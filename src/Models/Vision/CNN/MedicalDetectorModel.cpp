#include "MedicalDetectorModel.hpp"
#include "training/BBoxSupervision.hpp"
#include "runtimes/ops_loss_and_grad.hpp"
#include "include/json.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using json = nlohmann::json;

int convOut(int input, int kernel, int stride, int padding) {
    return (input + 2 * padding - kernel) / stride + 1;
}

double sigmoid(double value) {
    if (value >= 0.0) {
        const double exp_neg = std::exp(-value);
        return 1.0 / (1.0 + exp_neg);
    }
    const double exp_pos = std::exp(value);
    return exp_pos / (1.0 + exp_pos);
}

double bceWithLogits(double logit, double target) {
    return std::max(logit, 0.0) - logit * target +
           std::log1p(std::exp(-std::abs(logit)));
}

double smoothL1(double difference, double* derivative) {
    const double magnitude = std::abs(difference);
    if (magnitude < 1.0) {
        *derivative = difference;
        return 0.5 * difference * difference;
    }
    *derivative = difference < 0.0 ? -1.0 : 1.0;
    return magnitude - 0.5;
}

bool readTextFile(const std::filesystem::path& path, std::string* contents) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    *contents = buffer.str();
    return true;
}

} // namespace

MedicalDetectorModel::MedicalDetectorModel() {
    setModelName("MedicalDetectorModel");
    setHasEncoder(false);
}

void MedicalDetectorModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void MedicalDetectorModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("MedicalDetectorModel");

    const int image_w = std::max(1, cfg.image_w);
    const int image_h = std::max(1, cfg.image_h);
    const int image_c = std::max(1, cfg.image_c);
    const int base_channels = std::max(4, cfg.base_channels);
    const int num_classes = std::max(1, cfg.num_classes);
    const int output_channels = 5 + num_classes;

    model.modelConfig["type"] = "medical_detector";
    model.modelConfig["task"] = "object_detection";
    model.modelConfig["image_w"] = image_w;
    model.modelConfig["image_h"] = image_h;
    model.modelConfig["image_c"] = image_c;
    model.modelConfig["base_channels"] = base_channels;
    model.modelConfig["num_classes"] = num_classes;
    model.modelConfig["object_loss_weight"] = std::max(0.0f, cfg.object_loss_weight);
    model.modelConfig["box_loss_weight"] = std::max(0.0f, cfg.box_loss_weight);
    model.modelConfig["class_loss_weight"] = std::max(0.0f, cfg.class_loss_weight);
    model.modelConfig["object_pos_weight"] = std::max(1.0f, cfg.object_pos_weight);
    model.modelConfig["input_dim"] = image_w * image_h * image_c;
    model.modelConfig["output_channels"] = output_channels;
    model.modelConfig["detection_layout"] = "chw_grid_obj_xywh_classes";

    model.push("medical_detector/raw_in", "Identity", 0);
    if (auto* layer = model.getLayerByName("medical_detector/raw_in")) {
        layer->inputs = {"__input__"};
        layer->output = "medical_detector/in";
    }

    std::string current = "medical_detector/in";
    int current_c = image_c;
    int current_h = image_h;
    int current_w = image_w;

    for (int block = 1; block <= 5; ++block) {
        const int out_c = base_channels * std::min(1 << (block - 1), 8);
        const int out_h = convOut(current_h, 3, 2, 1);
        const int out_w = convOut(current_w, 3, 2, 1);
        const std::string prefix = "medical_detector/b" + std::to_string(block);
        const std::string conv_output = prefix + "/conv_y";

        model.push(prefix + "/conv", "Conv2d",
                   static_cast<size_t>(out_c) * static_cast<size_t>(current_c) * 9U);
        if (auto* layer = model.getLayerByName(prefix + "/conv")) {
            layer->inputs = {current};
            layer->output = conv_output;
            layer->in_channels = current_c;
            layer->out_channels = out_c;
            layer->input_height = current_h;
            layer->input_width = current_w;
            layer->kernel_size = 3;
            layer->stride = 2;
            layer->padding = 1;
            layer->use_bias = false;
        }

        model.push(prefix + "/norm", "LayerNorm", 0);
        if (auto* layer = model.getLayerByName(prefix + "/norm")) {
            layer->inputs = {conv_output};
            layer->output = prefix + "/norm_y";
            layer->in_channels = out_c;
            layer->input_height = out_h;
            layer->input_width = out_w;
            layer->in_features = std::max(1, out_c * out_h * out_w);
            layer->affine = false;
            layer->use_bias = false;
        }

        model.push(prefix + "/relu", "ReLU", 0);
        if (auto* layer = model.getLayerByName(prefix + "/relu")) {
            layer->inputs = {prefix + "/norm_y"};
            layer->output = prefix + "/out";
        }

        current = prefix + "/out";
        current_c = out_c;
        current_h = out_h;
        current_w = out_w;
    }

    model.push("medical_detector/head", "Conv2d",
               static_cast<size_t>(output_channels) * static_cast<size_t>(current_c) +
                   static_cast<size_t>(output_channels));
    if (auto* layer = model.getLayerByName("medical_detector/head")) {
        layer->inputs = {current};
        layer->output = "x";
        layer->in_channels = current_c;
        layer->out_channels = output_channels;
        layer->input_height = current_h;
        layer->input_width = current_w;
        layer->kernel_size = 1;
        layer->stride = 1;
        layer->padding = 0;
        layer->use_bias = true;
    }

    model.modelConfig["grid_h"] = current_h;
    model.modelConfig["grid_w"] = current_w;
    model.modelConfig["output_dim"] = output_channels * current_h * current_w;
}

std::optional<Model::TrainStepResult> MedicalDetectorModel::trainStep(
    const TrainStepRequest& request) {
    applyRuntimeConfiguration();
    if (!request.optimizer || !request.target) return std::nullopt;

    const std::vector<float>* input = nullptr;
    if (const auto it = request.float_inputs.find("__input__");
        it != request.float_inputs.end() && it->second) {
        input = it->second;
    } else if (const auto it = request.float_inputs.find("image");
               it != request.float_inputs.end() && it->second) {
        input = it->second;
    }
    if (!input) return std::nullopt;
    if (layers.empty() || layer_weight_blocks.empty()) {
        throw std::runtime_error(
            "MedicalDetectorModel::trainStep: model not built or weights not allocated");
    }

    const int grid_h = modelConfig.value("grid_h", 0);
    const int grid_w = modelConfig.value("grid_w", 0);
    const int num_classes = modelConfig.value("num_classes", 0);
    const int channels = 5 + num_classes;
    const size_t cells = static_cast<size_t>(grid_h) * static_cast<size_t>(grid_w);
    const size_t expected = cells * static_cast<size_t>(channels);
    if (grid_h <= 0 || grid_w <= 0 || num_classes <= 0 ||
        request.target->size() != expected) {
        throw std::invalid_argument(
            "MedicalDetectorModel::trainStep: target size does not match detection grid");
    }

    if (request.mode == TrainStepMode::Optimize) zeroGradients();
    const std::vector<float>& prediction = forwardPassView(*input, true);
    if (prediction.size() != expected) {
        throw std::runtime_error(
            "MedicalDetectorModel::trainStep: model output does not match detection grid");
    }

    const float object_weight = modelConfig.value("object_loss_weight", 1.0f);
    const float box_weight = modelConfig.value("box_loss_weight", 5.0f);
    const float class_weight = modelConfig.value("class_loss_weight", 1.0f);
    const float positive_weight = modelConfig.value("object_pos_weight", 8.0f);
    const std::vector<float>& target = *request.target;
    auto spatial = BBoxSupervision::makeWeights(request.annotation,grid_w,grid_h,channels,
        BBoxSupervision::Layout::CHW,modelConfig.value("bbox_loss_weight",1.0f), &modelConfig);
    if (request.loss_weights) BBoxSupervision::combineMask(spatial,*request.loss_weights);
    if (!spatial.values.empty() && spatial.values.size()!=expected)
        throw std::invalid_argument("MedicalDetectorModel: spatial weights size mismatch");
    auto weight = [&](size_t index) -> double {
        return spatial.values.empty() ? 1.0 : std::max(0.0f,spatial.values[index]);
    };


    size_t positives = 0;
    for (size_t cell = 0; cell < cells; ++cell) {
        if (target[cell] >= 0.5f) ++positives;
    }

    const double object_scale = 1.0 / static_cast<double>(std::max<size_t>(1, cells));
    const double box_scale = 1.0 / static_cast<double>(std::max<size_t>(1, positives * 4U));
    const double class_scale =
        1.0 / static_cast<double>(std::max<size_t>(1, positives * static_cast<size_t>(num_classes)));

    std::vector<float> gradient(expected, 0.0f);
    double object_loss = 0.0;
    double box_loss = 0.0;
    double class_loss = 0.0;

    for (size_t cell = 0; cell < cells; ++cell) {
        const double object_target = target[cell] >= 0.5f ? 1.0 : 0.0;
        const double object_logit = prediction[cell];
        const double object_factor = (object_target > 0.5 ? positive_weight : 1.0)*weight(cell);
        object_loss += bceWithLogits(object_logit, object_target) *
                       object_factor * object_scale;
        gradient[cell] = static_cast<float>(
            (sigmoid(object_logit) - object_target) * object_factor *
            object_scale * object_weight);

        if (object_target < 0.5) continue;

        for (int channel = 1; channel <= 4; ++channel) {
            const size_t index = static_cast<size_t>(channel) * cells + cell;
            const double activated = sigmoid(prediction[index]);
            const double difference = activated - static_cast<double>(target[index]);
            double derivative = 0.0;
            box_loss += smoothL1(difference, &derivative) * box_scale * weight(index);
            gradient[index] = static_cast<float>(
                derivative * activated * (1.0 - activated) * box_scale * box_weight * weight(index));
        }

        for (int class_id = 0; class_id < num_classes; ++class_id) {
            const size_t index = static_cast<size_t>(5 + class_id) * cells + cell;
            const double class_target = std::clamp<double>(target[index], 0.0, 1.0);
            const double class_logit = prediction[index];
            class_loss += bceWithLogits(class_logit, class_target) * class_scale * weight(index);
            gradient[index] = static_cast<float>(
                (sigmoid(class_logit) - class_target) * class_scale * class_weight * weight(index));
        }
    }

    double relation_loss=0;
    if (!spatial.relations.empty()) {
        std::vector<float> probabilities(prediction.size());
        for(size_t i=0;i<prediction.size();++i) probabilities[i]=sigmoid(prediction[i]);
        RuntimeLossGrad::PixelLossOptions relation_options;
        relation_options.weights=&spatial.values;relation_options.relations=&spatial.relations;
        relation_options.relation_only=true;
        const auto relational=RuntimeLossGrad::pixel_loss_and_grad(probabilities,target,"mse",relation_options);
        relation_loss=relational.relation_loss;
        for(size_t i=0;i<gradient.size();++i)
            gradient[i]+=relational.grad[i]*probabilities[i]*(1-probabilities[i]);
    }

    if (request.grad_scale != 1.0f) {
        for (float& value : gradient) value *= request.grad_scale;
    }
    backwardPass(gradient);

    TrainStepResult result;
    result.metrics["bbox_count"]=static_cast<float>(spatial.bbox_count);
    result.metrics["bbox_loss_weight"]=spatial.strength;
    result.metrics["relation_loss"]=static_cast<float>(relation_loss);
    result.metrics["relation_count"]=static_cast<float>(spatial.relations.size());
    result.loss = static_cast<float>(object_loss * object_weight + box_loss * box_weight +
                                     class_loss * class_weight + relation_loss);
    result.metrics["object_loss"] = static_cast<float>(object_loss);
    result.metrics["box_loss"] = static_cast<float>(box_loss);
    result.metrics["class_loss"] = static_cast<float>(class_loss);
    result.metrics["positive_cells"] = static_cast<float>(positives);
    if (request.capture_prediction) result.prediction = prediction;

    double gradient_sum_sq = 0.0;
    for (const auto& layer : layers) {
        for (float value : layer.grad_weights) {
            gradient_sum_sq += static_cast<double>(value) * static_cast<double>(value);
            result.grad_max_abs = std::max(result.grad_max_abs, std::abs(value));
        }
    }
    result.grad_norm = static_cast<float>(std::sqrt(gradient_sum_sq));

    if (request.mode == TrainStepMode::Optimize) {
        optimizerStep(*request.optimizer, request.learning_rate);
    }
    return result;
}

bool MedicalDetectorModel::parseAnnotation(const std::string& json_text,
                                           int fallback_width,
                                           int fallback_height,
                                           int num_classes,
                                           std::vector<Box>* boxes,
                                           std::string* error) {
    if (!boxes) return false;
    boxes->clear();
    try {
        const json document = json::parse(json_text);
        if (!document.is_object()) {
            if (error) *error = "annotation root must be an object";
            return false;
        }
        if (document.contains("entities")) {
            (void)BBoxSupervision::makeWeights(&document,1,1,1,BBoxSupervision::Layout::CHW);
            for (const auto& entity:document["entities"]) {
                if (!entity.contains("bbox")) continue;
                Box box;
                box.x1=entity["bbox"][0]; box.y1=entity["bbox"][1];
                box.x2=entity["bbox"][2]; box.y2=entity["bbox"][3];
                box.class_id=entity.value("class_id",0);
                if (box.class_id<0 || box.class_id>=num_classes)
                    throw std::invalid_argument("Entity class_id outside detector class range");
                box.label=entity.value("phrase",std::string("anomaly"));
                boxes->push_back(std::move(box));
            }
            return true;
        }
        const int width = std::max(1, document.value("width", fallback_width));
        const int height = std::max(1, document.value("height", fallback_height));
        const bool normalized = document.value("normalized", false);
        const auto it = document.find("boxes");
        if (it == document.end() || !it->is_array()) {
            if (error) *error = "annotation must contain a boxes array";
            return false;
        }

        for (const auto& value : *it) {
            if (!value.is_object()) continue;
            Box box;
            if (value.contains("x1") && value.contains("y1") &&
                value.contains("x2") && value.contains("y2")) {
                box.x1 = value.value("x1", 0.0f);
                box.y1 = value.value("y1", 0.0f);
                box.x2 = value.value("x2", 0.0f);
                box.y2 = value.value("y2", 0.0f);
            } else {
                const float x = value.value("x", 0.0f);
                const float y = value.value("y", 0.0f);
                box.x1 = x;
                box.y1 = y;
                box.x2 = x + value.value("w", 0.0f);
                box.y2 = y + value.value("h", 0.0f);
            }
            if (!normalized) {
                box.x1 /= static_cast<float>(width);
                box.x2 /= static_cast<float>(width);
                box.y1 /= static_cast<float>(height);
                box.y2 /= static_cast<float>(height);
            }
            box.x1 = std::clamp(box.x1, 0.0f, 1.0f);
            box.y1 = std::clamp(box.y1, 0.0f, 1.0f);
            box.x2 = std::clamp(box.x2, 0.0f, 1.0f);
            box.y2 = std::clamp(box.y2, 0.0f, 1.0f);
            box.class_id = value.value("class_id", value.value("class", 0));
            box.class_id = std::clamp(box.class_id, 0, std::max(0, num_classes - 1));
            box.label = value.value("label", std::string("anomaly"));
            if (box.x2 > box.x1 && box.y2 > box.y1) boxes->push_back(std::move(box));
        }
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

bool MedicalDetectorModel::loadAnnotationForImage(const std::string& image_path,
                                                  int fallback_width,
                                                  int fallback_height,
                                                  int num_classes,
                                                  std::vector<Box>* boxes,
                                                  std::string* annotation_path,
                                                  std::string* error) {
    const std::filesystem::path image(image_path);
    const std::filesystem::path base = image.parent_path() / image.stem();
    const std::vector<std::filesystem::path> candidates = {
        std::filesystem::path(base.string() + ".json"),
        std::filesystem::path(base.string() + ".boxes.json")
    };
    std::string contents;
    for (const auto& candidate : candidates) {
        if (!readTextFile(candidate, &contents)) continue;
        if (annotation_path) *annotation_path = candidate.string();
        return parseAnnotation(contents, fallback_width, fallback_height,
                               num_classes, boxes, error);
    }
    if (boxes) boxes->clear();
    if (error) *error = "annotation sidecar not found (.json or .boxes.json)";
    return false;
}

size_t MedicalDetectorModel::encodeTarget(const std::vector<Box>& boxes,
                                          int grid_h,
                                          int grid_w,
                                          int num_classes,
                                          std::vector<float>* target) {
    if (!target || grid_h <= 0 || grid_w <= 0 || num_classes <= 0) return 0;
    const size_t cells = static_cast<size_t>(grid_h) * static_cast<size_t>(grid_w);
    target->assign(static_cast<size_t>(5 + num_classes) * cells, 0.0f);
    std::vector<float> cell_areas(cells, -1.0f);
    size_t assigned = 0;
    for (const Box& box : boxes) {
        const float cx = 0.5f * (box.x1 + box.x2);
        const float cy = 0.5f * (box.y1 + box.y2);
        const float width = box.x2 - box.x1;
        const float height = box.y2 - box.y1;
        if (!(width > 0.0f && height > 0.0f)) continue;
        const int grid_x = std::clamp(static_cast<int>(cx * grid_w), 0, grid_w - 1);
        const int grid_y = std::clamp(static_cast<int>(cy * grid_h), 0, grid_h - 1);
        const size_t cell = static_cast<size_t>(grid_y * grid_w + grid_x);
        const float area = width * height;
        if (area <= cell_areas[cell]) continue;
        if (cell_areas[cell] < 0.0f) ++assigned;
        cell_areas[cell] = area;
        (*target)[cell] = 1.0f;
        (*target)[cells + cell] = std::clamp(cx * grid_w - grid_x, 0.0f, 1.0f);
        (*target)[2 * cells + cell] = std::clamp(cy * grid_h - grid_y, 0.0f, 1.0f);
        (*target)[3 * cells + cell] = std::clamp(width, 0.0f, 1.0f);
        (*target)[4 * cells + cell] = std::clamp(height, 0.0f, 1.0f);
        for (int class_id = 0; class_id < num_classes; ++class_id) {
            (*target)[static_cast<size_t>(5 + class_id) * cells + cell] =
                class_id == box.class_id ? 1.0f : 0.0f;
        }
    }
    return assigned;
}

float MedicalDetectorModel::intersectionOverUnion(const Box& a, const Box& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float union_area = area_a + area_b - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

std::vector<MedicalDetectorModel::Detection> MedicalDetectorModel::decodeDetections(
    const std::vector<float>& output,
    int grid_h,
    int grid_w,
    int num_classes,
    float score_threshold,
    float nms_iou_threshold,
    int max_detections) {
    std::vector<Detection> candidates;
    if (grid_h <= 0 || grid_w <= 0 || num_classes <= 0) return candidates;
    const size_t cells = static_cast<size_t>(grid_h) * static_cast<size_t>(grid_w);
    if (output.size() != static_cast<size_t>(5 + num_classes) * cells) return candidates;

    for (int grid_y = 0; grid_y < grid_h; ++grid_y) {
        for (int grid_x = 0; grid_x < grid_w; ++grid_x) {
            const size_t cell = static_cast<size_t>(grid_y * grid_w + grid_x);
            const float objectness = static_cast<float>(sigmoid(output[cell]));
            int best_class = 0;
            float best_class_score = 0.0f;
            for (int class_id = 0; class_id < num_classes; ++class_id) {
                const float probability = static_cast<float>(
                    sigmoid(output[static_cast<size_t>(5 + class_id) * cells + cell]));
                if (probability > best_class_score) {
                    best_class_score = probability;
                    best_class = class_id;
                }
            }
            const float score = objectness * best_class_score;
            if (score < score_threshold) continue;
            const float cx = (grid_x + static_cast<float>(sigmoid(output[cells + cell]))) / grid_w;
            const float cy = (grid_y + static_cast<float>(sigmoid(output[2 * cells + cell]))) / grid_h;
            const float width = static_cast<float>(sigmoid(output[3 * cells + cell]));
            const float height = static_cast<float>(sigmoid(output[4 * cells + cell]));
            Detection detection;
            detection.x1 = std::clamp(cx - 0.5f * width, 0.0f, 1.0f);
            detection.y1 = std::clamp(cy - 0.5f * height, 0.0f, 1.0f);
            detection.x2 = std::clamp(cx + 0.5f * width, 0.0f, 1.0f);
            detection.y2 = std::clamp(cy + 0.5f * height, 0.0f, 1.0f);
            detection.class_id = best_class;
            detection.label = "anomaly_" + std::to_string(best_class);
            detection.score = score;
            if (detection.x2 > detection.x1 && detection.y2 > detection.y1) {
                candidates.push_back(std::move(detection));
            }
        }
    }

    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Detection& left, const Detection& right) {
                         return left.score > right.score;
                     });
    std::vector<Detection> kept;
    for (const Detection& candidate : candidates) {
        bool suppressed = false;
        for (const Detection& accepted : kept) {
            if (candidate.class_id == accepted.class_id &&
                intersectionOverUnion(candidate, accepted) > nms_iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept.push_back(candidate);
        if (max_detections > 0 && static_cast<int>(kept.size()) >= max_detections) break;
    }
    return kept;
}