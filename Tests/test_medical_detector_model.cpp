#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Models/Vision/CNN/MedicalDetectorModel.hpp"
#include "include/json.hpp"

#include <cmath>
#include <vector>

using json = nlohmann::json;

int main() {
    json cfg = ModelArchitectures::defaultConfig("medical_detector");
    cfg["image_w"] = 64;
    cfg["image_h"] = 64;
    cfg["image_c"] = 3;
    cfg["base_channels"] = 4;
    cfg["num_classes"] = 2;

    auto model = ModelArchitectures::create("medical_detector", cfg);
    TASSERT_TRUE(model != nullptr);
    TASSERT_TRUE(model->modelConfig["type"].get<std::string>() == "medical_detector");
    TASSERT_TRUE(model->modelConfig["task"].get<std::string>() == "object_detection");
    TASSERT_TRUE(model->modelConfig["grid_h"].get<int>() == 2);
    TASSERT_TRUE(model->modelConfig["grid_w"].get<int>() == 2);
    TASSERT_TRUE(model->modelConfig["output_channels"].get<int>() == 7);
    TASSERT_TRUE(model->modelConfig["output_dim"].get<int>() == 28);

    const Layer* head = model->getLayerByName("medical_detector/head");
    TASSERT_TRUE(head != nullptr);
    TASSERT_TRUE(head->type == "Conv2d");
    TASSERT_TRUE(head->out_channels == 7);
    TASSERT_TRUE(head->output == "x");

    model->allocateParams();
    model->initializeWeights("xavier", 42);
    const std::vector<float> input(64 * 64 * 3, 0.125f);
    const std::vector<float> output = model->forwardPass(input, true);
    TASSERT_TRUE(output.size() == 28);
    model->backwardPass(std::vector<float>(output.size(), 1.0f));

    std::vector<float> target(output.size(), 0.0f);
    const size_t cells = 4;
    target[0] = 1.0f;
    target[cells + 0] = 0.5f;
    target[2 * cells + 0] = 0.5f;
    target[3 * cells + 0] = 0.25f;
    target[4 * cells + 0] = 0.25f;
    target[5 * cells + 0] = 1.0f;

    Optimizer optimizer;
    optimizer.type = OptimizerType::ADAMW;
    optimizer.weight_decay = 0.0f;
    Model::TrainStepRequest request;
    request.float_inputs["__input__"] = &input;
    request.target = &target;
    request.optimizer = &optimizer;
    request.learning_rate = 1e-4f;
    request.capture_prediction = true;
    model->setVizTapsEnabled(true);
    model->setVizTapsLimits(64, 64);
    const auto step = model->trainStep(request);
    TASSERT_TRUE(step.has_value());
    TASSERT_TRUE(std::isfinite(step->loss));
    TASSERT_TRUE(std::isfinite(step->grad_norm));
    TASSERT_TRUE(step->loss > 0.0f);
    TASSERT_TRUE(step->metrics.at("positive_cells") == 1.0f);
    TASSERT_TRUE(step->prediction.size() == output.size());
    TASSERT_TRUE(optimizer.step == 1);
    const auto taps = model->consumeVizTaps();
    TASSERT_TRUE(!taps.empty());
    bool has_medical_section = false;
    for (const auto& tap : taps) {
        if (tap.label.find("medical_detector/") == 0) has_medical_section = true;
    }
    TASSERT_TRUE(has_medical_section);

    const std::string annotation = R"({
        "width": 100,
        "height": 200,
        "boxes": [
            {"x1": 10, "y1": 20, "x2": 30, "y2": 80, "class_id": 1, "label": "mass"}
        ]
    })";
    std::vector<MedicalDetectorModel::Box> boxes;
    std::string parse_error;
    TASSERT_TRUE(MedicalDetectorModel::parseAnnotation(
        annotation, 100, 200, 2, &boxes, &parse_error));
    TASSERT_TRUE(boxes.size() == 1);
    TASSERT_NEAR(boxes[0].x1, 0.1f, 1e-6f);
    TASSERT_NEAR(boxes[0].y2, 0.4f, 1e-6f);
    TASSERT_TRUE(boxes[0].class_id == 1);

    std::vector<float> encoded;
    TASSERT_TRUE(MedicalDetectorModel::encodeTarget(boxes, 2, 2, 2, &encoded) == 1);
    TASSERT_TRUE(encoded.size() == 28);

    std::vector<float> synthetic(28, -10.0f);
    const size_t box_cell = 0;
    synthetic[box_cell] = 10.0f;
    synthetic[cells + box_cell] = 0.0f;
    synthetic[2 * cells + box_cell] = 0.0f;
    synthetic[3 * cells + box_cell] = -1.3862944f;
    synthetic[4 * cells + box_cell] = -0.8472979f;
    synthetic[6 * cells + box_cell] = 10.0f;
    const auto detections = MedicalDetectorModel::decodeDetections(
        synthetic, 2, 2, 2, 0.5f, 0.5f, 10);
    TASSERT_TRUE(detections.size() == 1);
    TASSERT_TRUE(detections[0].class_id == 1);
    TASSERT_TRUE(detections[0].score > 0.9f);

    return 0;
}