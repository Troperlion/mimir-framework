#include "LumenLatentDiffusionModel.hpp"
#include "Models/Vision/VAEConvModel.hpp"
#include "Serialization/Serialization.hpp"
#include "training/BBoxSupervision.hpp"
#include "runtimes/ops_loss_and_grad.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace {

// One numerical contract for corruption, previews and deterministic DDIM.
class DiffusionSchedule {
public:
    explicit DiffusionSchedule(const LumenLatentDiffusionModel::Config& cfg) {
        if (cfg.diffusion_steps < 2 || !std::isfinite(cfg.beta_start) ||
            !std::isfinite(cfg.beta_end) || cfg.beta_start <= 0.f ||
            cfg.beta_end < cfg.beta_start || cfg.beta_end >= 1.f) {
            throw std::invalid_argument("Lumen: invalid diffusion schedule");
        }
        double cumulative = 1.;
        for (int t = 0; t < cfg.diffusion_steps; ++t) {
            const double beta = cfg.beta_start + double(t) /
                (cfg.diffusion_steps - 1) * (cfg.beta_end - cfg.beta_start);
            cumulative *= 1. - beta;
            alpha_.push_back(static_cast<float>(cumulative));
        }
    }
    int size() const { return static_cast<int>(alpha_.size()); }
    float signal(int t) const { return std::sqrt(std::max(alpha_.at(t), 1e-8f)); }
    float noise(int t) const { return std::sqrt(std::max(0.f, 1.f - alpha_.at(t))); }
    std::vector<float> corrupt(const std::vector<float>& clean,
                               const std::vector<float>& epsilon, int t) const {
        if (clean.size() != epsilon.size()) throw std::invalid_argument("Lumen: noise shape mismatch");
        const float signal_scale = signal(t), noise_scale = noise(t);
        std::vector<float> result(clean.size());
        for (size_t i = 0; i < result.size(); ++i)
            result[i] = signal_scale * clean[i] + noise_scale * epsilon[i];
        return result;
    }
    void advance(std::vector<float>& sample, const std::vector<float>& epsilon,
                 int t, int previous) const {
        if (sample.size() != epsilon.size()) throw std::invalid_argument("Lumen: prediction shape mismatch");
        const float signal_scale = signal(t), noise_scale = noise(t);
        const float a = previous < 0 ? 1.f : signal(previous);
        const float b = previous < 0 ? 0.f : noise(previous);
        for (size_t i = 0; i < sample.size(); ++i) {
            const float clean = (sample[i] - noise_scale * epsilon[i]) / signal_scale;
            sample[i] = a * clean + b * epsilon[i];
        }
    }
private:
    std::vector<float> alpha_;
};

// Sample before allocating: full-resolution decoder outputs stay outside the UI snapshot.
template<class Value, class Colour>
Model::VizFrame rgbTip(int width, int height, int max_side, const std::string& label,
                       const std::string& info, Value value, Colour colour) {
    Model::VizFrame frame;
    const double scale = std::min(1., double(std::max(1, max_side)) / std::max(width, height));
    frame.w = std::max(1, int(std::lround(width * scale)));
    frame.h = std::max(1, int(std::lround(height * scale)));
    frame.channels = frame.tensor_sample_channels = 3;
    frame.label = label;
    frame.tensor_info = info + " | source=HWC " + std::to_string(height) + "x" + std::to_string(width) + "x3";
    const size_t count = static_cast<size_t>(frame.w) * frame.h * 3;
    frame.pixels.reserve(count);
    frame.tensor_values.reserve(count);
    frame.tensor_indices.reserve(count);
    for (int y = 0; y < frame.h; ++y) for (int x = 0; x < frame.w; ++x) {
        const size_t source = (static_cast<size_t>(y * height / frame.h) * width + x * width / frame.w) * 3;
        for (int c = 0; c < 3; ++c) {
            const float raw = value(source + c);
            frame.tensor_values.push_back(raw);
            frame.tensor_indices.push_back(source + c);
            frame.pixels.push_back(colour(raw));
        }
    }
    return frame;
}

uint8_t decodedColour(float value) {
    if (!std::isfinite(value)) return 0;
    return static_cast<uint8_t>(std::lround(std::clamp(value * .5f + .5f, 0.f, 1.f) * 255.f));
}

size_t parameterCount(int out_channels, int in_channels, int kernel_size) {
    return static_cast<size_t>(out_channels) * static_cast<size_t>(in_channels) *
           static_cast<size_t>(kernel_size) * static_cast<size_t>(kernel_size);
}

struct DistributionMoments {
    double mean = 0.0;
    double variance = 0.0;
    double skew = 0.0;
};

DistributionMoments distributionMoments(const std::vector<float>& values) {
    DistributionMoments moments;
    if (values.empty()) return moments;
    for (float value : values) moments.mean += value;
    moments.mean /= static_cast<double>(values.size());
    double third_moment = 0.0;
    for (float value : values) {
        const double delta = static_cast<double>(value) - moments.mean;
        moments.variance += delta * delta;
        third_moment += delta * delta * delta;
    }
    moments.variance /= static_cast<double>(values.size());
    const double deviation = std::sqrt(std::max(0.0, moments.variance));
    if (deviation > 1e-12) {
        moments.skew = (third_moment / static_cast<double>(values.size())) /
                       (deviation * deviation * deviation);
    }
    return moments;
}

double adjacentDifference(const std::vector<float>& values) {
    if (values.size() < 2) return 0.0;
    double total = 0.0;
    for (size_t index = 1; index < values.size(); ++index) {
        total += std::abs(static_cast<double>(values[index]) - values[index - 1]);
    }
    return total / static_cast<double>(values.size() - 1);
}

double correlation(const std::vector<float>& left, const std::vector<float>& right) {
    const size_t count = std::min(left.size(), right.size());
    if (count == 0) return 0.0;
    double left_mean = 0.0;
    double right_mean = 0.0;
    for (size_t index = 0; index < count; ++index) {
        left_mean += left[index];
        right_mean += right[index];
    }
    left_mean /= static_cast<double>(count);
    right_mean /= static_cast<double>(count);
    double covariance = 0.0;
    double left_variance = 0.0;
    double right_variance = 0.0;
    for (size_t index = 0; index < count; ++index) {
        const double left_delta = left[index] - left_mean;
        const double right_delta = right[index] - right_mean;
        covariance += left_delta * right_delta;
        left_variance += left_delta * left_delta;
        right_variance += right_delta * right_delta;
    }
    const double denominator = std::sqrt(left_variance * right_variance);
    return denominator > 1e-12 ? covariance / denominator : 0.0;
}

std::unordered_map<std::string, float> diffusionDiagnostics(
    const std::vector<float>& prediction,
    const std::vector<float>& target) {
    std::unordered_map<std::string, float> metrics;
    const DistributionMoments prediction_moments = distributionMoments(prediction);
    const DistributionMoments target_moments = distributionMoments(target);
    const double prediction_variance = std::max(prediction_moments.variance, 1e-12);
    const double target_variance = std::max(target_moments.variance, 1e-12);
    const double mean_delta = target_moments.mean - prediction_moments.mean;
    metrics["kl"] = static_cast<float>(std::max(0.0, 0.5 * (
        std::log(prediction_variance / target_variance) +
        (target_variance + mean_delta * mean_delta) / prediction_variance - 1.0)));
    const double deviation_delta =
        std::sqrt(target_variance) - std::sqrt(prediction_variance);
    metrics["wasserstein"] = static_cast<float>(std::sqrt(std::max(
        0.0, mean_delta * mean_delta + deviation_delta * deviation_delta)));
    metrics["entropy_diff"] = static_cast<float>(
        0.5 * std::log(prediction_variance / target_variance));
    metrics["moment_mismatch"] = static_cast<float>(
        std::abs(prediction_moments.skew - target_moments.skew));
    metrics["spatial_coherence"] = static_cast<float>(std::abs(
        adjacentDifference(prediction) - adjacentDifference(target)));
    metrics["temporal_consistency"] = static_cast<float>(correlation(prediction, target));
    return metrics;
}

struct DiffusionLoss {
    RuntimeLossGrad::LossWithGrad objective;
    float mse_unweighted=0.0f;
};

DiffusionLoss diffusionLoss(const std::vector<float>& prediction,
                            const std::vector<float>& target,
                            const std::vector<float>* weights,
                            const std::vector<SpatialRelation>* relations) {
    if(prediction.size()!=target.size())
        throw std::invalid_argument("Lumen: diffusion prediction and target sizes differ");
    if(weights && weights->size()!=prediction.size())
        throw std::invalid_argument("Lumen: spatial loss weights size mismatch");
    RuntimeLossGrad::PixelLossOptions options;
    options.weights=weights;options.relations=relations;
    DiffusionLoss result;
    // Keep epsilon prediction on MSE. The runtime owns both its scalar value
    // and its VJP, including spatial weights and relational supervision.
    result.objective=RuntimeLossGrad::pixel_loss_and_grad(prediction,target,"mse",options);
    const bool weighted=weights || (relations && !relations->empty());
    result.mse_unweighted=static_cast<float>(weighted
        ? RuntimeLossGrad::pixel_loss_and_grad(prediction,target,"mse").loss
        : result.objective.loss);
    return result;
}

} // namespace

LumenLatentDiffusionModel::LumenLatentDiffusionModel() {
    setModelName("LumenLatentDiffusionModel");
    setHasEncoder(false);
}

std::optional<Model::TrainStepResult> LumenLatentDiffusionModel::trainStep(
    const TrainStepRequest& request) {
    applyRuntimeConfiguration();
    if (request.optimizer == nullptr || request.target == nullptr) return std::nullopt;
    if (request.annotation) {
        auto weights = BBoxSupervision::makeWeights(request.annotation,cfg_.latent_w,cfg_.latent_h,
            cfg_.latent_c,BBoxSupervision::Layout::CHW,modelConfig.value("bbox_loss_weight",1.0f), &modelConfig);
        if (request.loss_weights) BBoxSupervision::combineMask(weights,*request.loss_weights);
        auto adapted=request; adapted.annotation=nullptr;
        adapted.relations = &weights.relations;
        adapted.loss_weights=weights.values.empty()?nullptr:&weights.values;
        auto result=trainStep(adapted);
        if (result) {
            result->metrics["bbox_count"]=static_cast<float>(weights.bbox_count);
            result->metrics["skeleton_region_count"]=static_cast<float>(weights.skeleton_region_count);
            result->metrics["bbox_loss_weight"]=weights.strength;
            result->metrics["relation_count"]=static_cast<float>(weights.relations.size());
        }
        return result;
    }


    std::unordered_map<std::string, std::vector<float>> float_inputs;
    for (const auto& [name, values] : request.float_inputs) {
        if (values == nullptr) return std::nullopt;
        float_inputs.emplace(name, *values);
    }
    std::unordered_map<std::string, std::vector<int>> int_inputs;
    for (const auto& [name, values] : request.int_inputs) {
        if (values == nullptr) return std::nullopt;
        int_inputs.emplace(name, *values);
    }
    if (float_inputs.empty()) return std::nullopt;

    if (request.mode == TrainStepMode::Optimize) zeroGradients();
    (void)forwardPassNamedView(float_inputs, int_inputs, true);
    const auto& prediction = getTensor("x"); // internal epsilon head, public output is z

    TrainStepResult result;
    result.metrics = diffusionDiagnostics(prediction, *request.target);
    auto loss=diffusionLoss(prediction,*request.target,request.loss_weights,request.relations);
    result.loss=static_cast<float>(loss.objective.loss);
    result.metrics["relation_loss"]=static_cast<float>(loss.objective.relation_loss);
    std::vector<float> loss_gradient=std::move(loss.objective.grad);
    result.metrics["relation_count"] = request.relations ? static_cast<float>(request.relations->size()) : 0.0f;
    result.metrics["mse_unweighted"]=loss.mse_unweighted;
    result.metrics["mse"] = result.loss;
    if (request.mode == TrainStepMode::Accumulate && request.grad_scale != 1.0f) {
        for (float& gradient : loss_gradient) gradient *= request.grad_scale;
    }
    backwardPass(loss_gradient);

    double gradient_sum = 0.0;
    for (const Layer& layer : getLayers()) {
        for (float gradient : layer.grad_weights) {
            gradient_sum += static_cast<double>(gradient) * gradient;
            result.grad_max_abs = std::max(result.grad_max_abs, std::abs(gradient));
        }
    }
    result.grad_norm = static_cast<float>(std::sqrt(gradient_sum));
    if (request.mode == TrainStepMode::Optimize) {
        optimizerStep(*request.optimizer, request.learning_rate);
    }
    return result;
}

static bool graphProduces(const Model& model, const std::string& tensor_name) {
    return std::any_of(
        model.getLayers().begin(), model.getLayers().end(),
        [&](const Layer& layer) { return layer.output == tensor_name; });
}

static std::string resolveVaeEncoderOutput(const Model& model) {
    for (const char* tensor_name : {"vae_conv/z_biased", "vae_conv/z"}) {
        if (graphProduces(model, tensor_name)) return tensor_name;
    }
    throw std::runtime_error(
        "LumenLatentDiffusionModel: VAE encoder graph has no usable latent output");
}

static nlohmann::json external_execution_graph(
    const Model& model,
    const nlohmann::json& bindings
) {
    nlohmann::json nodes = nlohmann::json::array();
    for (const Layer& layer : model.getLayers()) {
        nodes.push_back({
            {"name", layer.name},
            {"type", layer.type},
            {"inputs", layer.inputs},
            {"output", layer.output},
        });
    }
    return {
        {"model_name", model.getModelName()},
        {"bindings", bindings},
        {"nodes", std::move(nodes)},
    };
}

static void adaptLumenConfigToVaeCheckpoint(
    LumenLatentDiffusionModel::Config& cfg,
    const nlohmann::json& vae_config
) {
    const std::string type = vae_config.value("type", "");
    if (!type.empty() && type != "vae_conv") {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: checkpoint type must be 'vae_conv', got '" +
            type + "'");
    }

    auto adapt_int = [&](const char* key, int& value) {
        if (vae_config.contains(key) && vae_config[key].is_number_integer()) {
            value = vae_config[key].get<int>();
        }
    };
    auto adapt_bool = [&](const char* key, bool& value) {
        if (vae_config.contains(key) && vae_config[key].is_boolean()) {
            value = vae_config[key].get<bool>();
        }
    };
    auto adapt_string = [&](const char* key, std::string& value) {
        if (vae_config.contains(key) && vae_config[key].is_string()) {
            value = vae_config[key].get<std::string>();
        }
    };

    adapt_int("image_w", cfg.image_w);
    adapt_int("image_h", cfg.image_h);
    adapt_int("image_c", cfg.image_c);
    adapt_int("latent_w", cfg.latent_w);
    adapt_int("latent_h", cfg.latent_h);
    adapt_int("latent_c", cfg.latent_c);
    adapt_int("base_channels", cfg.vae_base_channels);
    adapt_bool("latent_spatial_context", cfg.vae_latent_spatial_context);
    adapt_bool("latent_detail_path", cfg.vae_latent_detail_path);
    adapt_int("latent_context_channels", cfg.vae_latent_context_channels);
    adapt_bool("stochastic_latent", cfg.vae_stochastic_latent);
    adapt_bool("resnet", cfg.vae_resnet);
    adapt_bool("attention", cfg.vae_attention);
    adapt_bool("use_skip_connections", cfg.vae_use_skip_connections);
    adapt_bool("use_encoder_prior", cfg.vae_use_encoder_prior);
    adapt_string("enc_norm", cfg.vae_enc_norm);
    adapt_string("dec_norm", cfg.vae_dec_norm);
    adapt_string("decoder_upsample", cfg.vae_decoder_upsample);
    adapt_int("enc_gn_groups", cfg.vae_enc_gn_groups);
    adapt_int("dec_gn_groups", cfg.vae_dec_gn_groups);
    adapt_int("attn_heads", cfg.vae_attn_heads);
    adapt_int("attn_max_tokens", cfg.vae_attn_max_tokens);
    adapt_int("resnet_max_tokens", cfg.vae_resnet_max_tokens);

    if (cfg.patch_size > 0) {
        cfg.patch_size = std::gcd(cfg.patch_size, std::gcd(cfg.latent_w, cfg.latent_h));
    }
}

bool LumenLatentDiffusionModel::InitVizTips() {
    clearVizTipsRegistry();
    // Keep canonical layer labels so graph selection and tensor picking stay linked.
    for (const auto& layer : getLayers()) {
        const auto& name = layer.name;
        std::string tip;
        if (name == "lumen/output/z") tip = "role=model_output tensor=z | reconstructed image";
        else if (name == "lumen/text/token_embedding") tip = "Prompt/tokens";
        else if (name == "lumen/text/final_norm") tip = "Prompt/context";
        else if (name == "lumen/time/output") tip = "Diffusion/time_condition";
        else if (name == "lumen/dit/patch_embed") tip = "DiT/latent_patches";
        else if (name == "lumen/dit/final_norm") tip = "DiT/features";
        else if (name == "lumen/dit/unpatchify") tip = "DiT/epsilon_latent";
        else if (name.rfind("lumen/dit/block", 0) == 0) {
            const auto slash = name.find('/', std::string("lumen/dit/").size());
            const auto block = name.substr(std::string("lumen/dit/").size(), slash - std::string("lumen/dit/").size());
            const auto operation = name.substr(slash + 1);
            if (operation == "self_attention" || operation == "cross_attention" || operation == "add_mlp")
                tip = "DiT/" + block + "/" + operation;
        }
        if (!tip.empty()) registerVizTip(name, tip);
    }
    return true;
}

bool LumenLatentDiffusionModel::UpdateVizTips(const Layer& layer, VizFrame& frame) {
    const bool selected = Model::UpdateVizTips(layer, frame);
    if (!selected) return false;
    if (layer.name == "lumen/output/z" && cfg_.image_c == 3 && !layer.inputs.empty() && hasTensor(layer.inputs[0])) {
        const auto& image = getTensor(layer.inputs[0]);
        if (image.size() == size_t(cfg_.image_w) * cfg_.image_h * 3)
            frame = rgbTip(cfg_.image_w, cfg_.image_h, getVizTapsMaxSide(), frame.label,
                "z: single-pass reconstructed RGB [-1,1]", [&](size_t i) { return image[i]; }, decodedColour);
        return true;
    }
    if (layer.name.rfind("lumen/text/", 0) == 0) {
        frame.tensor_info += " | prompt_tokens=" + std::to_string(
            modelConfig.value("text_active_seq_len", cfg_.text_seq_len));
    } else if (layer.name == "lumen/dit/unpatchify") {
        frame.tensor_info += " | epsilon CHW " + std::to_string(cfg_.latent_c) + "x" +
            std::to_string(cfg_.latent_h) + "x" + std::to_string(cfg_.latent_w);
    } else if (layer.name.rfind("lumen/dit/block", 0) == 0) {
        frame.tensor_info += " | latent_patch_grid=" +
            std::to_string(cfg_.latent_h / cfg_.patch_size) + "x" +
            std::to_string(cfg_.latent_w / cfg_.patch_size);
    }
    return true;
}

void LumenLatentDiffusionModel::addDiffusionVizTips(
    const std::vector<float>& generated_image,
    const std::vector<unsigned char>& original_image) {
    if (!isVizTapsEnabled() || cfg_.image_c != 3) return;
    const size_t count = static_cast<size_t>(cfg_.image_w) * cfg_.image_h * 3;
    if (generated_image.size() != count || original_image.size() != count) return;
    auto emit = [&](const char* label, const char* info, auto value, auto colour) {
        addVizTapFrame(rgbTip(cfg_.image_w, cfg_.image_h, getVizTapsMaxSide(), label, info, value, colour));
    };
    auto delta = [&](size_t i) { return float(int(decodedColour(generated_image[i])) - int(original_image[i])); };
    emit("resdiff_abs", "absolute pixel residual [0,255]",
        [&](size_t i) { return std::abs(delta(i)); },
        [](float d) { return uint8_t(std::abs(d)); });
    emit("resdiff_norm", "signed pixel residual [-255,255]", delta,
        [](float d) { return uint8_t(std::lround(127.5f + .5f * d)); });
    emit("resdiff_max", "positive pixel residual [0,255]",
        [&](size_t i) { return std::max(0.f, delta(i)); }, [](float d) { return uint8_t(d); });
    emit("resdiff_min", "negative pixel residual magnitude [0,255]",
        [&](size_t i) { return std::max(0.f, -delta(i)); }, [](float d) { return uint8_t(d); });
    emit("diffusion_out", "decoded model prediction [-1,1]",
        [&](size_t i) { return generated_image[i]; }, decodedColour);
    emit("diffusion/reference", "dataset pixels [0,255]",
        [&](size_t i) { return float(original_image[i]); }, [](float v) { return uint8_t(v); });
}

void LumenLatentDiffusionModel::addDiffusionComparisonVizTips(
    const std::vector<float>& oracle_image,
    const std::vector<float>& noisy_baseline_image,
    const std::vector<float>& predicted_image, int timestep) {
    if (!isVizTapsEnabled() || cfg_.image_c != 3) return;
    const size_t count = static_cast<size_t>(cfg_.image_w) * cfg_.image_h * 3;
    if (oracle_image.size() != count || noisy_baseline_image.size() != count || predicted_image.size() != count) return;
    auto emit = [&](const char* name, const std::vector<float>& image) {
        const std::string label = std::string("diffusion/compare/") + name + " | timestep=" + std::to_string(timestep);
        addVizTapFrame(rgbTip(cfg_.image_w, cfg_.image_h, getVizTapsMaxSide(), label,
            "decoded latent [-1,1] | timestep=" + std::to_string(timestep),
            [&](size_t i) { return image[i]; }, decodedColour));
    };
    emit("A_oracle_decode_z0", oracle_image);
    emit("B_noisy_baseline", noisy_baseline_image);
    emit("C_model_decode_z0_pred", predicted_image);
}

void LumenLatentDiffusionModel::buildFromConfig(const Config& cfg) {
    const DiffusionSchedule validated_schedule(cfg);
    cfg_ = cfg;
    if (!cfg_.vae_checkpoint.empty()) {
        Model checkpoint_metadata;
        Mimir::Serialization::LoadOptions metadata_options;
        metadata_options.format = std::filesystem::is_directory(cfg_.vae_checkpoint)
            ? Mimir::Serialization::CheckpointFormat::RawFolder
            : Mimir::Serialization::CheckpointFormat::SafeTensors;
        metadata_options.load_optimizer = false;
        metadata_options.load_tokenizer = false;
        metadata_options.load_encoder = false;
        metadata_options.strict_mode = true;
        metadata_options.validate_checksums = false;
        metadata_options.apply_model_name = false;
        metadata_options.apply_model_config = true;
        metadata_options.metadata_only = true;

        std::string metadata_error;
        if (!Mimir::Serialization::load_checkpoint(
                checkpoint_metadata, cfg_.vae_checkpoint, metadata_options,
                &metadata_error)) {
            throw std::runtime_error(
                "LumenLatentDiffusionModel: cannot read VAE configuration: " +
                metadata_error);
        }
        adaptLumenConfigToVaeCheckpoint(cfg_, checkpoint_metadata.modelConfig);
    }
    if (cfg_.image_w <= 0 || cfg_.image_h <= 0 || cfg_.image_c <= 0) {
        throw std::runtime_error("LumenLatentDiffusionModel: invalid image dimensions");
    }
    if (cfg_.latent_w <= 0 || cfg_.latent_h <= 0 || cfg_.latent_c <= 0) {
        throw std::runtime_error("LumenLatentDiffusionModel: invalid latent dimensions");
    }
    if (cfg_.patch_size <= 0 || cfg_.latent_w % cfg_.patch_size != 0 ||
        cfg_.latent_h % cfg_.patch_size != 0) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: latent dimensions must be divisible by patch_size");
    }
    if (modelConfig.is_object()) {
        modelConfig.erase("external_graphs");
        modelConfig.erase("graph_bindings");
    }
    vae_encoder_.reset();
    vae_decoder_.reset();
    vae_encoder_output_.clear();
    vae_skip_bindings_.clear();
    vae_encoder_skips_.clear();
    if (!cfg_.vae_checkpoint.empty()) {
        VAEConvModel::Config vae_cfg;
        vae_cfg.image_w = cfg_.image_w;
        vae_cfg.image_h = cfg_.image_h;
        vae_cfg.image_c = cfg_.image_c;
        vae_cfg.latent_w = cfg_.latent_w;
        vae_cfg.latent_h = cfg_.latent_h;
        vae_cfg.latent_c = cfg_.latent_c;
        vae_cfg.base_channels = cfg_.vae_base_channels;
        vae_cfg.latent_spatial_context = cfg_.vae_latent_spatial_context;
        vae_cfg.latent_detail_path = cfg_.vae_latent_detail_path;
        vae_cfg.latent_context_channels = cfg_.vae_latent_context_channels;
        vae_cfg.stochastic_latent = cfg_.vae_stochastic_latent;
        vae_cfg.resnet = cfg_.vae_resnet;
        vae_cfg.attention = cfg_.vae_attention;
        vae_cfg.use_skip_connections = cfg_.vae_use_skip_connections;
        vae_cfg.use_encoder_prior = cfg_.vae_use_encoder_prior;
        vae_cfg.enc_norm = cfg_.vae_enc_norm;
        vae_cfg.dec_norm = cfg_.vae_dec_norm;
        vae_cfg.decoder_upsample = cfg_.vae_decoder_upsample;
        vae_cfg.enc_gn_groups = cfg_.vae_enc_gn_groups;
        vae_cfg.dec_gn_groups = cfg_.vae_dec_gn_groups;
        vae_cfg.attn_heads = cfg_.vae_attn_heads;
        vae_cfg.attn_max_tokens = cfg_.vae_attn_max_tokens;
        vae_cfg.resnet_max_tokens = cfg_.vae_resnet_max_tokens;
        vae_cfg.d_model = 1024;
        vae_cfg.text_cond = false;

        vae_encoder_ = std::make_unique<VAEConvModel>();
        static_cast<VAEConvModel&>(*vae_encoder_).buildFromConfig(vae_cfg);
        vae_encoder_->allocateParams();

        vae_decoder_ = std::make_unique<Model>();
        VAEConvModel::buildDecoderInto(*vae_decoder_, vae_cfg);
        vae_decoder_->allocateParams();

        Mimir::Serialization::LoadOptions load_options;
        load_options.format = std::filesystem::is_directory(cfg_.vae_checkpoint)
            ? Mimir::Serialization::CheckpointFormat::RawFolder
            : Mimir::Serialization::CheckpointFormat::SafeTensors;
        load_options.load_optimizer = false;
        load_options.load_tokenizer = false;
        load_options.load_encoder = false;
        load_options.validate_checksums = true;
        load_options.apply_model_name = false;
        load_options.apply_model_config = false;

        std::string load_error;
        // A full VAE checkpoint also contains decoder and optimizer tensors that
        // do not belong to this encoder-only graph. Known tensors still undergo
        // checksum, dtype and shape validation in non-strict mode.
        load_options.strict_mode = false;
        if (!Mimir::Serialization::load_checkpoint(
                *vae_encoder_, cfg_.vae_checkpoint, load_options, &load_error)) {
            throw std::runtime_error(
                "LumenLatentDiffusionModel: cannot load VAE encoder: " + load_error);
        }
        load_options.strict_mode = false;
        if (!Mimir::Serialization::load_checkpoint(
                *vae_decoder_, cfg_.vae_checkpoint, load_options, &load_error)) {
            throw std::runtime_error(
                "LumenLatentDiffusionModel: cannot load VAE decoder: " + load_error);
        }
        vae_encoder_->freezeParameters();
        vae_decoder_->freezeParameters();
        vae_encoder_output_ = resolveVaeEncoderOutput(*vae_encoder_);
        if (cfg_.vae_use_skip_connections &&
            vae_encoder_->modelConfig.contains("encoder_skip_outputs")) {
            for (const auto& skip :
                 vae_encoder_->modelConfig["encoder_skip_outputs"]) {
                VaeSkipBinding binding;
                binding.decoder_input = skip.value("decoder_input", "");
                binding.encoder_output = skip.value("encoder_output", "");
                binding.values = static_cast<size_t>(skip.value("channels", 0)) *
                    static_cast<size_t>(skip.value("height", 0)) *
                    static_cast<size_t>(skip.value("width", 0));
                if (binding.decoder_input.empty() ||
                    binding.encoder_output.empty() || binding.values == 0) {
                    throw std::runtime_error(
                        "LumenLatentDiffusionModel: invalid VAE skip binding");
                }
                vae_skip_bindings_.push_back(std::move(binding));
            }
        }
        if (!graphProduces(*vae_decoder_, "x")) {
            throw std::runtime_error(
                "LumenLatentDiffusionModel: VAE decoder graph has no image output 'x'");
        }
    }

    setHasEncoder(false);
    buildInto(*this, cfg_);
    appendReconstructionGraph();

    if (vae_encoder_ && vae_decoder_) {
        // Frozen codec: show its decoded comparisons, avoid capturing every internal
        // layer three times for one preview. The external graph remains available.
        vae_encoder_->modelConfig["viz_taps_suppressed"] = true;
        vae_decoder_->modelConfig["viz_taps_suppressed"] = true;
        nlohmann::json decoder_skip_inputs = nlohmann::json::array();
        for (const auto& binding : vae_skip_bindings_) {
            decoder_skip_inputs.push_back(binding.decoder_input);
        }
        modelConfig["external_graphs"]["vae_encoder"] =
            external_execution_graph(*vae_encoder_, {
                {"inputs", {{"image", "__input__"}}},
                {"outputs", {{"latent", vae_encoder_output_}}},
                {"feeds", {{"model", "lumen"}, {"input", "__input__"}}},
            });
        modelConfig["vae_skip_bindings"] = nlohmann::json::array();
        for (const auto& binding : vae_skip_bindings_) {
            modelConfig["vae_skip_bindings"].push_back({
                {"encoder_output", binding.encoder_output},
                {"decoder_input", binding.decoder_input},
                {"values", binding.values},
            });
        }
        modelConfig["graph_bindings"] = {
            {"image_encoder", {
                {"model", "vae_encoder"},
                {"input", "__input__"},
                {"output", vae_encoder_output_},
            }},
            {"denoiser", {
                {"model", "lumen"},
                {"input", "__input__"},
                {"output", "x"},
            }},
            {"image_decoder", {
                {"model", "lumen"},
                {"input", "lumen/reconstruction/unshifted"},
                {"skip_inputs", decoder_skip_inputs},
                {"output", "z"},
            }},
        };
    }
    modelConfig["graph_bindings"]["model_output"] = {{"model", "lumen"}, {"output", "z"}};
}

void LumenLatentDiffusionModel::appendReconstructionGraph() {
    auto binary = [&](const std::string& name, const std::string& type,
                      const std::string& left, const std::string& right) {
        push(name, type, 0);
        auto* layer = getLayerByName(name);
        layer->inputs = {left, right};
        layer->output = name;
    };
    const std::string root = "lumen/reconstruction/";
    binary(root + "scaled_epsilon", "Multiply", "x", "diffusion_noise_scale");
    binary(root + "signal", "Subtract", "__input__", root + "scaled_epsilon");
    binary(root + "clean_latent", "Divide", root + "signal", "diffusion_signal_scale");
    std::string image;
    if (vae_decoder_) {
        binary(root + "unscaled", "Divide", root + "clean_latent", "vae_decode_scale");
        binary(root + "unshifted", "Add", root + "unscaled", "vae_decode_shift");
        const std::string prefix = root + "decoder/";
        for (const auto& source : vae_decoder_->getLayers()) {
            Layer layer = source;
            layer.name = prefix + source.name;
            layer.output = prefix + (source.output.empty() ? "x" : source.output);
            for (auto& input : layer.inputs)
                input = input == "__input__" ? root + "unshifted" : prefix + input;
            if (!layer.shared_weights_from.empty()) layer.shared_weights_from = prefix + layer.shared_weights_from;
            if (source.getWeightsSize())
                layer.weights.assign(source.getWeights(), source.getWeights() + source.getWeightsSize());
            layer.weight_block = nullptr;
            layer.grad_weights.clear();
            getMutableLayers().push_back(std::move(layer));
        }
        modelConfig["frozen_layer_prefixes"] = json::array({prefix});
        modelConfig["initialize_from_layer_weights"] = true;
        image = prefix + "x";
    } else {
        push(root + "to_hwc", "Permute", 0);
        auto* layer = getLayerByName(root + "to_hwc");
        layer->inputs = {root + "clean_latent"};
        layer->output = root + "image_hwc";
        layer->shape = {cfg_.latent_c, cfg_.latent_h, cfg_.latent_w};
        layer->permute_dims = {1, 2, 0};
        image = layer->output;
    }
    push("lumen/output/z", "Identity", 0);
    auto* output = getLayerByName("lumen/output/z");
    output->inputs = {image};
    output->output = "z";
    output->output_width = cfg_.image_w;
    output->output_height = cfg_.image_h;
    output->out_channels = cfg_.image_c;
    modelConfig["forward_output_tensor"] = "z";
    modelConfig["inference_output_tensor"] = "z";
    modelConfig["output_tensor"] = "z";
    modelConfig["diffusion_prediction_tensor"] = "x";
    modelConfig["output_dim"] = cfg_.image_w * cfg_.image_h * cfg_.image_c;
    modelConfig["architecture_version"] = 5;
    modelConfig["graph_bindings"]["model_output"] = {{"model", "lumen"}, {"output", "z"}};
}

LumenLatentDiffusionModel::DiffusionExample LumenLatentDiffusionModel::prepareExample(
    const std::vector<unsigned char>& pixels, unsigned int seed) {
    const size_t values = static_cast<size_t>(cfg_.image_w) * cfg_.image_h * cfg_.image_c;
    if (pixels.size() != values) throw std::runtime_error("Lumen: image size mismatch");
    std::vector<float> image(values);
    std::transform(pixels.begin(), pixels.end(), image.begin(),
                   [](unsigned char value) { return value / 127.5f - 1.f; });
    DiffusionExample example;
    example.clean = encodeImage(image);
    const size_t latent_values = static_cast<size_t>(cfg_.latent_w) * cfg_.latent_h * cfg_.latent_c;
    if (example.clean.size() != latent_values) throw std::runtime_error("Lumen: latent size mismatch");
    const DiffusionSchedule schedule(cfg_);
    std::mt19937 random(seed);
    std::uniform_int_distribution<int> timestep(0, schedule.size() - 1);
    std::normal_distribution<float> normal(0.f, 1.f);
    example.timestep = timestep(random);
    example.noise.resize(latent_values);
    for (float& value : example.noise) value = normal(random);
    example.noisy = schedule.corrupt(example.clean, example.noise, example.timestep);
    return example;
}

LumenLatentDiffusionModel::TrainStats LumenLatentDiffusionModel::trainDiffusionStep(
    const std::vector<unsigned char>& rgb_image,
    const std::string& prompt,
    unsigned int seed,
    Optimizer& optimizer,
    float learning_rate,
    const json* annotation,
    float bbox_loss_weight,
    bool drop_conditioning
) {
    applyRuntimeConfiguration();
    if (!std::isfinite(learning_rate) || learning_rate <= 0.f)
        throw std::invalid_argument("Lumen: learning rate must be positive");
    const auto example = prepareExample(rgb_image, seed);
    const auto& clean_latent = example.clean;
    const auto& noise = example.noise;
    const auto& noisy_latent = example.noisy;
    const int timestep = example.timestep;
    const int context_dim = std::max(8, cfg_.hidden_size);
    const std::vector<float> timestep_embedding = timestepEmbedding(timestep, context_dim);
    const std::vector<int> text_ids = (drop_conditioning ? promptIds("") : conditionedPromptIds(prompt,annotation));
    const auto bbox_weights=BBoxSupervision::makeWeights(annotation,cfg_.latent_w,cfg_.latent_h,
        cfg_.latent_c,BBoxSupervision::Layout::CHW,bbox_loss_weight, &modelConfig);
    TrainStepRequest request;
    if (!bbox_weights.values.empty()) request.loss_weights=&bbox_weights.values;
    request.relations=&bbox_weights.relations;
    request.float_inputs = {{"__input__", &noisy_latent}, {"timestep", &timestep_embedding}};
    request.int_inputs = {{"text_ids", &text_ids}};
    request.target = &noise;
    request.optimizer = &optimizer;
    request.learning_rate = learning_rate;
    const DiffusionSchedule schedule(cfg_);
    const std::vector<float> signal_scale{schedule.signal(timestep)}, noise_scale{schedule.noise(timestep)};
    request.float_inputs["diffusion_signal_scale"] = &signal_scale;
    request.float_inputs["diffusion_noise_scale"] = &noise_scale;
    const auto step_result = trainStep(request);
    if (!step_result.has_value()) {
        throw std::runtime_error("LumenLatentDiffusionModel: training request rejected");
    }
    // z is reconstructed by the same forward graph, without an inference loop.
    // Additional oracle/baseline comparisons belong only to validation.

    TrainStats stats;
    const auto metric = [&](const char* name) {
        const auto it = step_result->metrics.find(name);
        return it != step_result->metrics.end() ? it->second : 0.0f;
    };
    stats.loss = step_result->loss;
    stats.mse = metric("mse");
    stats.mse_unweighted = metric("mse_unweighted");
    stats.skeleton_region_count=bbox_weights.skeleton_region_count;
    stats.relation_count=bbox_weights.relation_count;
    stats.bbox_count=bbox_weights.bbox_count;
    stats.bbox_loss_weight=bbox_weights.strength;
    stats.relation_loss=metric("relation_loss");
    stats.kl = metric("kl");
    stats.kl_beta_effective = 0.0f;
    stats.grad_norm = step_result->grad_norm;
    stats.grad_max_abs = step_result->grad_max_abs;
    stats.wasserstein = metric("wasserstein");
    stats.entropy_diff = metric("entropy_diff");
    stats.moment_mismatch = metric("moment_mismatch");
    stats.spatial_coherence = metric("spatial_coherence");
    stats.temporal_consistency = metric("temporal_consistency");
    stats.timestep = timestep;
    return stats;
}

LumenLatentDiffusionModel::TrainStats LumenLatentDiffusionModel::validateDiffusionStep(
    const std::vector<unsigned char>& rgb_image,
    const std::string& prompt,
    unsigned int seed,
    const json* annotation,
    float bbox_loss_weight
) {
    applyRuntimeConfiguration();
    const auto example = prepareExample(rgb_image, seed);
    const auto& clean_latent = example.clean;
    const auto& noise = example.noise;
    const auto& noisy_latent = example.noisy;
    const int timestep = example.timestep;
    const DiffusionSchedule schedule(cfg_);
    const float signal_scale = schedule.signal(timestep);
    const float noise_scale = schedule.noise(timestep);
    const int context_dim = std::max(8, cfg_.hidden_size);
    (void)forwardPassNamed(
        {{"__input__", noisy_latent},
         {"timestep", timestepEmbedding(timestep, context_dim)},
         {"diffusion_signal_scale", {signal_scale}}, {"diffusion_noise_scale", {noise_scale}}},
        {{"text_ids", conditionedPromptIds(prompt,annotation)}}, false);
    const auto prediction = getTensor("x");
    if (prediction.size() != noise.size()) {
        throw std::runtime_error("LumenLatentDiffusionModel: validation output size mismatch");
    }
    std::vector<float> predicted_clean(prediction.size());
    std::vector<float> noisy_baseline(prediction.size());
    for (size_t index = 0; index < prediction.size(); ++index) {
        predicted_clean[index] =
            (noisy_latent[index] - noise_scale * prediction[index]) / signal_scale;
        noisy_baseline[index] = noisy_latent[index] / signal_scale;
    }
    const std::vector<float> reconstructed = hasTensor("z") ? getTensor("z") : decodeImage(predicted_clean, true);
    if (isVizTapsEnabled()) {
        const std::vector<float> oracle_image = decodeImage(clean_latent, true);
        const std::vector<float> noisy_baseline_image = decodeImage(noisy_baseline, true);
        addDiffusionComparisonVizTips(
            oracle_image,
            noisy_baseline_image,
            reconstructed,
            timestep);
        addDiffusionVizTips(reconstructed, rgb_image);
    }

    std::vector<float> generated_image(reconstructed.size()),target_image(reconstructed.size());
    for(size_t index=0;index<reconstructed.size();++index) {
        generated_image[index]=static_cast<float>(std::clamp(
            static_cast<double>(reconstructed[index])*.5+.5,0.0,1.0));
        target_image[index]=static_cast<float>(static_cast<double>(rgb_image[index])/255.0);
    }
    const auto reconstruction_mae=RuntimeLossGrad::pixel_loss_and_grad(generated_image,target_image,"mae");
    const auto reconstruction_mse=RuntimeLossGrad::pixel_loss_and_grad(generated_image,target_image,"mse");
    const auto bbox_weights=BBoxSupervision::makeWeights(annotation,cfg_.latent_w,cfg_.latent_h,
        cfg_.latent_c,BBoxSupervision::Layout::CHW,bbox_loss_weight, &modelConfig);
    const auto loss=diffusionLoss(prediction,noise,
        bbox_weights.values.empty()?nullptr:&bbox_weights.values,&bbox_weights.relations);
    TrainStats stats;
    stats.loss=static_cast<float>(loss.objective.loss);
    stats.mse_unweighted=loss.mse_unweighted;
    stats.relation_loss=static_cast<float>(loss.objective.relation_loss);
    stats.skeleton_region_count=bbox_weights.skeleton_region_count;
    stats.relation_count=bbox_weights.relation_count;
    stats.bbox_count=bbox_weights.bbox_count; stats.bbox_loss_weight=bbox_weights.strength;
    stats.mse = stats.loss;
    const auto diagnostics = diffusionDiagnostics(prediction, noise);
    stats.kl = diagnostics.at("kl");
    stats.wasserstein = diagnostics.at("wasserstein");
    stats.entropy_diff = diagnostics.at("entropy_diff");
    stats.moment_mismatch = diagnostics.at("moment_mismatch");
    stats.spatial_coherence = diagnostics.at("spatial_coherence");
    stats.temporal_consistency = diagnostics.at("temporal_consistency");
    stats.reconstruction_mae=static_cast<float>(reconstruction_mae.loss);
    stats.reconstruction_mse=static_cast<float>(reconstruction_mse.loss);
    stats.timestep = timestep;
    return stats;
}

void LumenLatentDiffusionModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("LumenLatentDiffusionModel");
    model.modelConfig["type"] = "lumen_diffusion";
    model.modelConfig["task"] = "text_to_image_rgb_diffusion_transformer";

    const int latent_w = std::max(1, cfg.latent_w);
    const int latent_h = std::max(1, cfg.latent_h);
    const int latent_c = std::max(1, cfg.latent_c);
    const int patch_size = std::max(1, cfg.patch_size);
    const int patch_w = latent_w / patch_size;
    const int patch_h = latent_h / patch_size;
    const int patch_tokens = patch_w * patch_h;
    const int patch_dim = latent_c * patch_size * patch_size;
    const int depth = std::max(1, cfg.depth);
    const int context_dim = std::max(8, cfg.hidden_size);
    const int mlp_hidden = std::max(context_dim,
        static_cast<int>(std::lround(context_dim * std::max(1.0f, cfg.mlp_ratio))));
    const int text_len = std::max(1, cfg.text_seq_len);
    const int vocab = std::max(8, cfg.vocab_size);
    int heads = std::max(1, std::min(cfg.num_heads, context_dim));
    while (heads > 1 && context_dim % heads != 0) --heads;

    model.modelConfig["image_w"] = cfg.image_w;
    model.modelConfig["image_h"] = cfg.image_h;
    model.modelConfig["image_c"] = cfg.image_c;
    model.modelConfig["latent_w"] = latent_w;
    model.modelConfig["latent_h"] = latent_h;
    model.modelConfig["latent_c"] = latent_c;
    model.modelConfig["architecture"] = "dit_latent_vae_conv";
    model.modelConfig["architecture_version"] = 4;
    model.modelConfig["time_projection_layout"] = "packed_per_block";
    model.modelConfig["viz_registered_tips_only"] = true;
    model.modelConfig["preview_source"] = "validation";
    model.modelConfig["prediction_type"] = "epsilon";
    model.modelConfig["sampler"] = "ddim";
    model.modelConfig["patch_size"] = patch_size;
    model.modelConfig["patch_tokens"] = patch_tokens;
    model.modelConfig["hidden_size"] = context_dim;
    model.modelConfig["mlp_ratio"] = cfg.mlp_ratio;
    model.modelConfig["depth"] = depth;
    model.modelConfig["context_dim"] = context_dim;
    model.modelConfig["vocab_size"] = vocab;
    model.modelConfig["text_seq_len"] = text_len;
    model.modelConfig["dynamic_text_seq_len"] = cfg.dynamic_text_seq_len;
    model.modelConfig["text_layers"] = std::max(1, cfg.text_layers);
    model.modelConfig["num_heads"] = heads;
    model.modelConfig["diffusion_steps"] = std::max(2, cfg.diffusion_steps);
    model.modelConfig["beta_start"] = cfg.beta_start;
    model.modelConfig["beta_end"] = cfg.beta_end;
    model.modelConfig["preview_timestep"] = std::clamp(
        cfg.preview_timestep, 0, std::max(1, cfg.diffusion_steps) - 1);
    model.modelConfig["kl_beta"] = 0.0f;
    model.modelConfig["kl_warmup_steps"] = 0;
    model.modelConfig["vae_scale"] = cfg.vae_scale;
    model.modelConfig["vae_shift"] = cfg.vae_shift;
    model.modelConfig["vae_decoder_upsample"] = cfg.vae_decoder_upsample;
    model.modelConfig["vae_latent_spatial_context"] = cfg.vae_latent_spatial_context;
    model.modelConfig["vae_latent_detail_path"] = cfg.vae_latent_detail_path;
    model.modelConfig["vae_latent_context_channels"] = cfg.vae_latent_context_channels;
    model.modelConfig["vae_calibrated"] =
        std::isfinite(cfg.vae_scale) && cfg.vae_scale > 0.0f;
    model.modelConfig["input_dim"] = latent_w * latent_h * latent_c;
    model.modelConfig["output_dim"] = latent_w * latent_h * latent_c;

    auto addConv = [&](const std::string& name,
                       const std::string& input,
                       const std::string& output,
                       int in_channels,
                       int out_channels,
                       int height,
                       int width,
                       int kernel,
                       int stride,
                       int padding) {
        model.push(name, "Conv2d", parameterCount(out_channels, in_channels, kernel));
        if (auto* layer = model.getLayerByName(name)) {
            layer->inputs = {input};
            layer->output = output;
            layer->in_channels = in_channels;
            layer->out_channels = out_channels;
            layer->input_height = height;
            layer->input_width = width;
            layer->kernel_size = kernel;
            layer->stride = stride;
            layer->padding = padding;
            layer->use_bias = false;
        }
    };

    auto addActivation = [&](const std::string& name,
                             const std::string& input,
                             const std::string& output) {
        model.push(name, "SiLU", 0);
        if (auto* layer = model.getLayerByName(name)) {
            layer->inputs = {input};
            layer->output = output;
        }
    };

    model.push("lumen/text/token_embedding", "Embedding",
               static_cast<size_t>(vocab) * static_cast<size_t>(context_dim));
    if (auto* layer = model.getLayerByName("lumen/text/token_embedding")) {
        layer->inputs = {"text_ids"};
        layer->output = "lumen/text/tokens";
        layer->vocab_size = vocab;
        layer->embed_dim = context_dim;
        layer->seq_len = text_len;
        layer->padding_idx = 0;
    }

    model.push("lumen/text/position", "Constant",
               static_cast<size_t>(text_len) * static_cast<size_t>(context_dim));
    if (auto* layer = model.getLayerByName("lumen/text/position")) {
        layer->output = "lumen/text/position_values";
        layer->seq_len = text_len;
        layer->embed_dim = context_dim;
    }
    model.push("lumen/text/add_position", "Add", 0);
    if (auto* layer = model.getLayerByName("lumen/text/add_position")) {
        layer->inputs = {"lumen/text/tokens", "lumen/text/position_values"};
        layer->output = "lumen/text/context0";
    }

    std::string context = "lumen/text/context0";
    for (int index = 0; index < std::max(1, cfg.text_layers); ++index) {
        const std::string prefix = "lumen/text/block" + std::to_string(index + 1);
        model.push(prefix + "/norm1", "LayerNorm", static_cast<size_t>(2 * context_dim));
        if (auto* layer = model.getLayerByName(prefix + "/norm1")) {
            layer->inputs = {context};
            layer->output = prefix + "/norm1_out";
            layer->in_features = context_dim;
            layer->affine = true;
            layer->use_bias = true;
            layer->eps = 1e-5f;
        }
        model.push(prefix + "/attention", "SelfAttention",
                   static_cast<size_t>(4) * static_cast<size_t>(context_dim) *
                       static_cast<size_t>(context_dim));
        if (auto* layer = model.getLayerByName(prefix + "/attention")) {
            layer->inputs = {prefix + "/norm1_out"};
            layer->output = prefix + "/attention_out";
            layer->seq_len = text_len;
            layer->embed_dim = context_dim;
            layer->num_heads = heads;
            layer->causal = false;
        }
        model.push(prefix + "/add", "Add", 0);
        if (auto* layer = model.getLayerByName(prefix + "/add")) {
            layer->inputs = {context, prefix + "/attention_out"};
            layer->output = prefix + "/res1";
        }
        model.push(prefix + "/norm2", "LayerNorm", static_cast<size_t>(2 * context_dim));
        if (auto* layer = model.getLayerByName(prefix + "/norm2")) {
            layer->inputs = {prefix + "/res1"};
            layer->output = prefix + "/norm2_out";
            layer->in_features = context_dim;
            layer->affine = true;
            layer->use_bias = true;
            layer->eps = 1e-5f;
        }
        model.push(prefix + "/mlp_fc1", "Linear",
                   static_cast<size_t>(context_dim) * static_cast<size_t>(mlp_hidden) +
                       static_cast<size_t>(mlp_hidden));
        if (auto* layer = model.getLayerByName(prefix + "/mlp_fc1")) {
            layer->inputs = {prefix + "/norm2_out"};
            layer->output = prefix + "/mlp_hidden";
            layer->seq_len = text_len;
            layer->in_features = context_dim;
            layer->out_features = mlp_hidden;
            layer->use_bias = true;
        }
        model.push(prefix + "/mlp_gelu", "GELU", 0);
        if (auto* layer = model.getLayerByName(prefix + "/mlp_gelu")) {
            layer->inputs = {prefix + "/mlp_hidden"};
            layer->output = prefix + "/mlp_activated";
        }
        model.push(prefix + "/mlp_fc2", "Linear",
                   static_cast<size_t>(mlp_hidden) * static_cast<size_t>(context_dim) +
                       static_cast<size_t>(context_dim));
        if (auto* layer = model.getLayerByName(prefix + "/mlp_fc2")) {
            layer->inputs = {prefix + "/mlp_activated"};
            layer->output = prefix + "/mlp_out";
            layer->seq_len = text_len;
            layer->in_features = mlp_hidden;
            layer->out_features = context_dim;
            layer->use_bias = true;
        }
        model.push(prefix + "/add_mlp", "Add", 0);
        if (auto* layer = model.getLayerByName(prefix + "/add_mlp")) {
            layer->inputs = {prefix + "/res1", prefix + "/mlp_out"};
            layer->output = prefix + "/out";
        }
        context = prefix + "/out";
    }
    model.push("lumen/text/final_norm", "LayerNorm",
               static_cast<size_t>(2 * context_dim));
    if (auto* layer = model.getLayerByName("lumen/text/final_norm")) {
        layer->inputs = {context};
        layer->output = "lumen/text/context";
        layer->in_features = context_dim;
        layer->affine = true;
        layer->use_bias = true;
        layer->eps = 1e-5f;
    }
    context = "lumen/text/context";

    model.push("lumen/time/input", "Identity", 0);
    if (auto* layer = model.getLayerByName("lumen/time/input")) {
        layer->inputs = {"timestep"};
        layer->output = "lumen/time/embedding";
    }
    model.push("lumen/time/projection", "Linear",
               static_cast<size_t>(context_dim) * static_cast<size_t>(context_dim) +
                   static_cast<size_t>(context_dim));
    if (auto* layer = model.getLayerByName("lumen/time/projection")) {
        layer->inputs = {"lumen/time/embedding"};
        layer->output = "lumen/time/projected";
        layer->in_features = context_dim;
        layer->out_features = context_dim;
        layer->use_bias = true;
    }
    addActivation("lumen/time/activation", "lumen/time/projected", "lumen/time/activated");
    model.push("lumen/time/output", "Linear", static_cast<size_t>(context_dim) * context_dim + context_dim);
    if (auto* layer = model.getLayerByName("lumen/time/output")) {
        layer->inputs = {"lumen/time/activated"};
        layer->output = "lumen/time/condition";
        layer->in_features = context_dim;
        layer->out_features = context_dim;
        layer->use_bias = true;
    }

    model.push("lumen/dit/input", "Identity", 0);
    if (auto* layer = model.getLayerByName("lumen/dit/input")) {
        layer->inputs = {"__input__"};
        layer->output = "lumen/dit/latent_chw";
    }
    addConv("lumen/dit/patch_embed", "lumen/dit/latent_chw", "lumen/dit/patch_chw",
            latent_c, context_dim, latent_h, latent_w, patch_size, patch_size, 0);
    model.push("lumen/dit/to_tokens", "Permute", 0);
    if (auto* layer = model.getLayerByName("lumen/dit/to_tokens")) {
        layer->inputs = {"lumen/dit/patch_chw"};
        layer->output = "lumen/dit/patch_tokens";
        layer->shape = {context_dim, patch_h, patch_w};
        layer->permute_dims = {1, 2, 0};
    }
    model.push("lumen/dit/position", "Constant",
               static_cast<size_t>(patch_tokens) * static_cast<size_t>(context_dim));
    if (auto* layer = model.getLayerByName("lumen/dit/position")) {
        layer->output = "lumen/dit/position_values";
        layer->seq_len = patch_tokens;
        layer->embed_dim = context_dim;
    }
    model.push("lumen/dit/add_position", "Add", 0);
    if (auto* layer = model.getLayerByName("lumen/dit/add_position")) {
        layer->inputs = {"lumen/dit/patch_tokens", "lumen/dit/position_values"};
        layer->output = "lumen/dit/positioned";
    }
    auto addNorm = [&](const std::string& name,
                       const std::string& input,
                       const std::string& output) {
        model.push(name, "LayerNorm", static_cast<size_t>(2 * context_dim));
        if (auto* layer = model.getLayerByName(name)) {
            layer->inputs = {input};
            layer->output = output;
            layer->in_features = context_dim;
            layer->affine = true;
            layer->use_bias = true;
            layer->eps = 1e-5f;
        }
    };

    model.push("lumen/time/block_projection", "Linear",
               static_cast<size_t>(depth) * context_dim * (context_dim + 1));
    if (auto* layer = model.getLayerByName("lumen/time/block_projection")) {
        layer->inputs = {"lumen/time/condition"};
        layer->output = "lumen/time/blocks_packed";
        layer->in_features = context_dim;
        layer->out_features = depth * context_dim;
        layer->use_bias = true;
    }
    model.push("lumen/time/split_blocks", "Split", 0);
    if (auto* layer = model.getLayerByName("lumen/time/split_blocks")) {
        layer->inputs = {"lumen/time/blocks_packed"};
        layer->output = "lumen/time/blocks";
        layer->split_sizes.assign(depth, context_dim);
        layer->split_axis = 0;
    }
    std::string tokens = "lumen/dit/positioned";
    const size_t attention_params = static_cast<size_t>(4) *
        static_cast<size_t>(context_dim) * static_cast<size_t>(context_dim);
    for (int index = 0; index < depth; ++index) {
        const std::string prefix = "lumen/dit/block" + std::to_string(index + 1);
        model.push(prefix + "/add_time", "Add", 0);
        if (auto* layer = model.getLayerByName(prefix + "/add_time")) {
            layer->inputs = {tokens, "lumen/time/blocks_" + std::to_string(index)};
            layer->output = prefix + "/conditioned";
        }
        tokens = prefix + "/conditioned";
        addNorm(prefix + "/norm1", tokens, prefix + "/norm1_out");
        model.push(prefix + "/self_attention", "SelfAttention", attention_params);
        if (auto* layer = model.getLayerByName(prefix + "/self_attention")) {
            layer->inputs = {prefix + "/norm1_out"};
            layer->output = prefix + "/self_out";
            layer->seq_len = patch_tokens;
            layer->embed_dim = context_dim;
            layer->num_heads = heads;
            layer->causal = false;
        }
        model.push(prefix + "/add_self", "Add", 0);
        if (auto* layer = model.getLayerByName(prefix + "/add_self")) {
            layer->inputs = {tokens, prefix + "/self_out"};
            layer->output = prefix + "/res1";
        }

        addNorm(prefix + "/norm2", prefix + "/res1", prefix + "/norm2_out");
        model.push(prefix + "/cross_attention", "CrossAttention", attention_params);
        if (auto* layer = model.getLayerByName(prefix + "/cross_attention")) {
            layer->inputs = {prefix + "/norm2_out", context};
            layer->output = prefix + "/cross_out";
            layer->embed_dim = context_dim;
            layer->in_features = context_dim;
            layer->num_heads = heads;
            layer->causal = false;
        }
        model.push(prefix + "/add_cross", "Add", 0);
        if (auto* layer = model.getLayerByName(prefix + "/add_cross")) {
            layer->inputs = {prefix + "/res1", prefix + "/cross_out"};
            layer->output = prefix + "/res2";
        }

        addNorm(prefix + "/norm3", prefix + "/res2", prefix + "/norm3_out");
        model.push(prefix + "/mlp_fc1", "Linear",
                   static_cast<size_t>(context_dim) * static_cast<size_t>(mlp_hidden) +
                       static_cast<size_t>(mlp_hidden));
        if (auto* layer = model.getLayerByName(prefix + "/mlp_fc1")) {
            layer->inputs = {prefix + "/norm3_out"};
            layer->output = prefix + "/mlp_hidden";
            layer->seq_len = patch_tokens;
            layer->in_features = context_dim;
            layer->out_features = mlp_hidden;
            layer->use_bias = true;
        }
        model.push(prefix + "/mlp_gelu", "GELU", 0);
        if (auto* layer = model.getLayerByName(prefix + "/mlp_gelu")) {
            layer->inputs = {prefix + "/mlp_hidden"};
            layer->output = prefix + "/mlp_activated";
        }
        model.push(prefix + "/mlp_fc2", "Linear",
                   static_cast<size_t>(mlp_hidden) * static_cast<size_t>(context_dim) +
                       static_cast<size_t>(context_dim));
        if (auto* layer = model.getLayerByName(prefix + "/mlp_fc2")) {
            layer->inputs = {prefix + "/mlp_activated"};
            layer->output = prefix + "/mlp_out";
            layer->seq_len = patch_tokens;
            layer->in_features = mlp_hidden;
            layer->out_features = context_dim;
            layer->use_bias = true;
        }
        model.push(prefix + "/add_mlp", "Add", 0);
        if (auto* layer = model.getLayerByName(prefix + "/add_mlp")) {
            layer->inputs = {prefix + "/res2", prefix + "/mlp_out"};
            layer->output = prefix + "/out";
        }
        tokens = prefix + "/out";
    }

    addNorm("lumen/dit/final_norm", tokens, "lumen/dit/final_norm_out");
    model.push("lumen/dit/patch_output", "Linear",
               static_cast<size_t>(context_dim) * static_cast<size_t>(patch_dim) +
                   static_cast<size_t>(patch_dim));
    if (auto* layer = model.getLayerByName("lumen/dit/patch_output")) {
        layer->inputs = {"lumen/dit/final_norm_out"};
        layer->output = "lumen/dit/patch_noise";
        layer->seq_len = patch_tokens;
        layer->in_features = context_dim;
        layer->out_features = patch_dim;
        layer->use_bias = true;
    }
    model.push("lumen/dit/to_patch_chw", "Permute", 0);
    if (auto* layer = model.getLayerByName("lumen/dit/to_patch_chw")) {
        layer->inputs = {"lumen/dit/patch_noise"};
        layer->output = "lumen/dit/patch_noise_chw";
        layer->shape = {patch_h, patch_w, patch_dim};
        layer->permute_dims = {2, 0, 1};
    }
    model.push("lumen/dit/unpatchify", "PixelShuffle", 0);
    if (auto* layer = model.getLayerByName("lumen/dit/unpatchify")) {
        layer->inputs = {"lumen/dit/patch_noise_chw"};
        layer->output = "x";
        layer->scale_h = static_cast<float>(patch_size);
        layer->in_channels = patch_dim;
        layer->input_height = patch_h;
        layer->input_width = patch_w;
        layer->output_width = latent_w;
        layer->output_height = latent_h;
    }
    // Explicit tip geometry avoids treating token-major features as full images.
    for (auto& layer : model.getMutableLayers()) {
        if (layer.name.rfind("lumen/text/", 0) == 0) {
            layer.output_width = context_dim;
            layer.output_height = text_len;
        } else if (layer.name.rfind("lumen/time/", 0) == 0) {
            layer.output_width = layer.output_height = 1;
        } else if (layer.name.rfind("lumen/dit/block", 0) == 0 || layer.name == "lumen/dit/final_norm") {
            layer.output_width = patch_w;
            layer.output_height = patch_h;
        }
    }
}

std::vector<int> LumenLatentDiffusionModel::conditionedPromptIds(const std::string& prompt, const json* annotation) const {
    const auto context=BBoxSupervision::groundedPrompt(prompt,annotation);
    if(!modelConfig.value("dynamic_text_seq_len",cfg_.dynamic_text_seq_len) && annotation && annotation->is_object() && annotation->contains("entities") && !(*annotation)["entities"].empty()) {
        const auto count=getTokenizer().tokenize(context).size();
        if(count>static_cast<size_t>(cfg_.text_seq_len))throw std::invalid_argument(
            "Lumen annotation context requires "+std::to_string(count)+" tokens; increase text_seq_len (current "+std::to_string(cfg_.text_seq_len)+")");
    }
    return promptIds(context);
}

std::vector<int> LumenLatentDiffusionModel::promptIds(const std::string& prompt) const {
    std::vector<int> ids = getTokenizer().tokenize(prompt);
    const int pad = std::max(0, getTokenizer().getPadId());
    const int unknown = std::max(0, getTokenizer().getUnkId());
    if (!modelConfig.value("dynamic_text_seq_len",cfg_.dynamic_text_seq_len)) {
        ids.resize(static_cast<size_t>(std::max(1,cfg_.text_seq_len)),pad);
    } else if(ids.empty()) {
        ids.push_back(pad);
    }
    for (int& id : ids) {
        if (id < 0 || id >= cfg_.vocab_size) id = unknown;
    }
    return ids;
}

const std::vector<float>& LumenLatentDiffusionModel::forwardPassNamedView(
    const std::unordered_map<std::string,std::vector<float>>& float_inputs,
    const std::unordered_map<std::string,std::vector<int>>& int_inputs,
    bool training) {
    if(modelConfig.value("dynamic_text_seq_len",cfg_.dynamic_text_seq_len)) {
        // Match Model's legacy alias precedence and normalize an absent/empty context.
        auto ids=int_inputs.find("seq");
        if(ids==int_inputs.end())ids=int_inputs.find("text_ids");
        if(ids==int_inputs.end() || ids->second.empty()) {
            auto normalized=int_inputs;normalized.erase("seq");
            normalized["text_ids"]={std::max(0,getTokenizer().getPadId())};
            return forwardPassNamedView(float_inputs,normalized,training);
        }
        if(ids!=int_inputs.end()) {
            const size_t count=std::max<size_t>(1,ids->second.size());
            if(count>size_t(std::numeric_limits<int>::max()))throw std::invalid_argument("Lumen: text context too long");
            bool changed=false;
            for(auto& layer:getMutableLayers())if(layer.name.rfind("lumen/text/",0)==0) {
                if(layer.seq_len>0 && layer.seq_len!=int(count)){layer.seq_len=int(count);changed=true;}
                if (layer.output_width != std::max(8, cfg_.hidden_size) || layer.output_height != int(count)) {
                    layer.output_width = std::max(8, cfg_.hidden_size);
                    layer.output_height = int(count);
                    changed = true;
                }
                if(layer.name=="lumen/text/position" && layer.shape!=std::vector<int>{-1,layer.embed_dim}) {
                    layer.shape={-1,layer.embed_dim};changed=true;
                }
            }
            if(changed)static_plan_.built=false;
            modelConfig["text_active_seq_len"]=count;
        }
    }
    if (!getLayerByName("lumen/output/z"))
        return Model::forwardPassNamedView(float_inputs, int_inputs, training); // serialized legacy graph
    // Topology owns these fields, including when loading an older epsilon-only
    // checkpoint into the already-built reconstruction graph.
    modelConfig["forward_output_tensor"] = "z";
    modelConfig["output_tensor"] = "z";
    modelConfig["output_dim"] = cfg_.image_w * cfg_.image_h * cfg_.image_c;
    modelConfig["architecture_version"] = 5;
    if (!modelConfig.contains("inference_output_tensor")) modelConfig["inference_output_tensor"] = "z";
    if (vae_decoder_) {
        auto& prefixes = modelConfig["frozen_layer_prefixes"];
        if (!prefixes.is_array()) prefixes = json::array();
        const std::string prefix = "lumen/reconstruction/decoder/";
        if (std::find(prefixes.begin(), prefixes.end(), json(prefix)) == prefixes.end()) prefixes.push_back(prefix);
    }
    auto inputs = float_inputs;
    const DiffusionSchedule schedule(cfg_);
    if (!inputs.count("diffusion_signal_scale")) inputs["diffusion_signal_scale"] = {schedule.signal(0)};
    if (!inputs.count("diffusion_noise_scale")) inputs["diffusion_noise_scale"] = {schedule.noise(0)};
    const bool decoder_tail = training || modelConfig.value("inference_output_tensor", std::string("z")) != "x";
    inputs["vae_decode_scale"] = {vae_decoder_ ? modelConfig.value("vae_scale", cfg_.vae_scale) : 1.f};
    inputs["vae_decode_shift"] = {vae_decoder_ ? modelConfig.value("vae_shift", cfg_.vae_shift) : 0.f};
    if (decoder_tail && vae_decoder_ && (!std::isfinite(inputs["vae_decode_scale"][0]) || inputs["vae_decode_scale"][0] <= 0.f))
        throw std::runtime_error("Lumen: calibrate the VAE before reconstructing z");
    // Binary runtime operators consume matching tensor shapes; expand schedule
    // and calibration scalars here instead of relying on backend broadcasting.
    const size_t latent_values = size_t(cfg_.latent_w) * cfg_.latent_h * cfg_.latent_c;
    for (const char* name : {"diffusion_signal_scale", "diffusion_noise_scale", "vae_decode_scale", "vae_decode_shift"}) {
        auto& values = inputs[name];
        if (values.size() == 1) values.assign(latent_values, values[0]);
        if (values.size() != latent_values) throw std::invalid_argument(std::string("Lumen: invalid coefficient shape: ") + name);
    }
    return Model::forwardPassNamedView(inputs,int_inputs,training);
}

std::vector<float> LumenLatentDiffusionModel::timestepEmbedding(int timestep, int dim) const {
    std::vector<float> embedding(static_cast<size_t>(dim), 0.0f);
    const int half = std::max(1, dim / 2);
    for (int index = 0; index < half; ++index) {
        const float exponent = -std::log(10000.0f) * static_cast<float>(index) /
                               static_cast<float>(std::max(1, half - 1));
        const float angle = static_cast<float>(timestep) * std::exp(exponent);
        embedding[static_cast<size_t>(index)] = std::sin(angle);
        if (index + half < dim) {
            embedding[static_cast<size_t>(index + half)] = std::cos(angle);
        }
    }
    return embedding;
}

std::vector<float> LumenLatentDiffusionModel::encodeImageUncalibrated(
    const std::vector<float>& image
) {
    const size_t expected = static_cast<size_t>(cfg_.image_w) * cfg_.image_h * cfg_.image_c;
    if (image.size() != expected) {
        throw std::runtime_error("LumenLatentDiffusionModel: input image size mismatch");
    }
    if (vae_encoder_) {
        vae_encoder_skips_.clear();
        (void)vae_encoder_->forwardPassNamed(
            {{"__input__", image}}, {}, false);
        if (!vae_encoder_->hasTensor(vae_encoder_output_)) {
            throw std::runtime_error(
                "LumenLatentDiffusionModel: VAE encoder did not produce " +
                vae_encoder_output_);
        }
        const auto& latent = vae_encoder_->getTensor(vae_encoder_output_);
        const size_t latent_values = static_cast<size_t>(cfg_.latent_w) *
            cfg_.latent_h * cfg_.latent_c;
        if (latent.size() != latent_values) {
            throw std::runtime_error("LumenLatentDiffusionModel: VAE latent size mismatch");
        }
        for (const auto& binding : vae_skip_bindings_) {
            if (!vae_encoder_->hasTensor(binding.encoder_output)) {
                throw std::runtime_error(
                    "LumenLatentDiffusionModel: VAE encoder did not produce skip " +
                    binding.encoder_output);
            }
            const auto& skip = vae_encoder_->getTensor(binding.encoder_output);
            if (skip.size() != binding.values) {
                throw std::runtime_error(
                    "LumenLatentDiffusionModel: VAE encoder skip size mismatch for " +
                    binding.encoder_output);
            }
            vae_encoder_skips_[binding.decoder_input] =
                std::vector<float>(skip.begin(), skip.end());
        }
        return std::vector<float>(latent.begin(), latent.end());
    }
    const size_t latent_values = static_cast<size_t>(cfg_.latent_w) *
        cfg_.latent_h * cfg_.latent_c;
    if (latent_values != expected) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: vae_checkpoint is required to encode images");
    }
    std::vector<float> rgb_chw(expected);
    const size_t spatial = static_cast<size_t>(cfg_.image_w) * cfg_.image_h;
    for (int y = 0; y < cfg_.image_h; ++y) {
        for (int x = 0; x < cfg_.image_w; ++x) {
            for (int channel = 0; channel < cfg_.image_c; ++channel) {
                const size_t hwc = (static_cast<size_t>(y) * cfg_.image_w + x) *
                                   cfg_.image_c + channel;
                const size_t chw = static_cast<size_t>(channel) * spatial +
                                   static_cast<size_t>(y) * cfg_.image_w + x;
                rgb_chw[chw] = image[hwc];
            }
        }
    }
    return rgb_chw;
}

std::vector<float> LumenLatentDiffusionModel::encodeImage(const std::vector<float>& image) {
    std::vector<float> latent = encodeImageUncalibrated(image);
    if (!vae_encoder_) return latent;

    const float scale = modelConfig.value("vae_scale", cfg_.vae_scale);
    const float shift = modelConfig.value("vae_shift", cfg_.vae_shift);
    if (!std::isfinite(scale) || scale <= 0.0f || !std::isfinite(shift)) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: VAE latent calibration is unavailable; "
            "calibrate the VAE before training");
    }
    for (float& value : latent) value = (value - shift) * scale;
    return latent;
}

void LumenLatentDiffusionModel::beginVaeCalibration() {
    if (!vae_encoder_) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: vae_checkpoint is required for calibration");
    }
    vae_calibration_items_ = 0;
    vae_calibration_values_ = 0;
    vae_calibration_mean_ = 0.0;
    vae_calibration_m2_ = 0.0;
    modelConfig["vae_calibrated"] = false;
}

LumenLatentDiffusionModel::VaeCalibrationStats
LumenLatentDiffusionModel::addVaeCalibrationImage(
    const std::vector<unsigned char>& rgb_image
) {
    const size_t expected = static_cast<size_t>(cfg_.image_w) * cfg_.image_h * cfg_.image_c;
    if (rgb_image.size() != expected) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: calibration image size mismatch; expected " +
            std::to_string(expected) + " RGB values");
    }
    std::vector<float> image(expected);
    for (size_t index = 0; index < expected; ++index) {
        image[index] = static_cast<float>(rgb_image[index]) / 127.5f - 1.0f;
    }
    const std::vector<float> latent = encodeImageUncalibrated(image);
    for (float value : latent) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "LumenLatentDiffusionModel: non-finite VAE latent during calibration");
        }
        ++vae_calibration_values_;
        const double delta = static_cast<double>(value) - vae_calibration_mean_;
        vae_calibration_mean_ += delta / static_cast<double>(vae_calibration_values_);
        const double delta2 = static_cast<double>(value) - vae_calibration_mean_;
        vae_calibration_m2_ += delta * delta2;
    }
    ++vae_calibration_items_;
    return {vae_calibration_items_, vae_calibration_values_,
            static_cast<float>(vae_calibration_mean_), 0.0f};
}

LumenLatentDiffusionModel::VaeCalibrationStats
LumenLatentDiffusionModel::finishVaeCalibration() {
    if (vae_calibration_items_ == 0 || vae_calibration_values_ == 0) {
        throw std::runtime_error("LumenLatentDiffusionModel: VAE calibration has no samples");
    }
    const double variance = vae_calibration_m2_ /
        static_cast<double>(vae_calibration_values_);
    const float shift = static_cast<float>(vae_calibration_mean_);
    const float stddev = static_cast<float>(std::sqrt(std::max(0.0, variance)));
    const float scale = std::clamp(1.0f / std::max(stddev, 1e-6f), 1e-3f, 1e3f);
    modelConfig["vae_scale"] = scale;
    modelConfig["vae_shift"] = shift;
    modelConfig["vae_calibrated"] = true;
    modelConfig["vae_calibration_items"] = vae_calibration_items_;
    modelConfig["vae_calibration_values"] = vae_calibration_values_;
    return {vae_calibration_items_, vae_calibration_values_, shift, scale};
}

std::vector<float> LumenLatentDiffusionModel::decodeImage(
    const std::vector<float>& latent_chw,
    bool use_encoder_skips
) {
    const size_t latent_values = static_cast<size_t>(cfg_.latent_w) *
        cfg_.latent_h * cfg_.latent_c;
    if (latent_chw.size() != latent_values) {
        throw std::runtime_error("LumenLatentDiffusionModel: latent diffusion tensor size mismatch");
    }
    if (vae_decoder_) {
        const float scale = modelConfig.value("vae_scale", cfg_.vae_scale);
        const float shift = modelConfig.value("vae_shift", cfg_.vae_shift);
        if (!std::isfinite(scale) || scale <= 0.0f || !std::isfinite(shift)) {
            throw std::runtime_error(
                "LumenLatentDiffusionModel: VAE latent calibration is unavailable");
        }
        std::vector<float> decoder_latent(latent_chw.size());
        for (size_t index = 0; index < latent_chw.size(); ++index) {
            decoder_latent[index] = latent_chw[index] / scale + shift;
        }
        std::unordered_map<std::string, std::vector<float>> decoder_inputs = {
            {"__input__", std::move(decoder_latent)},
        };
        for (const auto& binding : vae_skip_bindings_) {
            const auto cached = vae_encoder_skips_.find(binding.decoder_input);
            if (use_encoder_skips && cached != vae_encoder_skips_.end()) {
                decoder_inputs.emplace(binding.decoder_input, cached->second);
            } else {
                decoder_inputs.emplace(
                    binding.decoder_input, std::vector<float>(binding.values, 0.0f));
            }
        }
        const std::vector<float> decoded = vae_decoder_->forwardPassNamed(
            decoder_inputs, {}, false);
        const size_t image_values = static_cast<size_t>(cfg_.image_w) *
            cfg_.image_h * cfg_.image_c;
        if (decoded.size() != image_values) {
            throw std::runtime_error("LumenLatentDiffusionModel: VAE decoded image size mismatch");
        }
        return decoded;
    }
    const size_t expected = static_cast<size_t>(cfg_.image_w) * cfg_.image_h * cfg_.image_c;
    if (latent_values != expected) {
        throw std::runtime_error(
            "LumenLatentDiffusionModel: vae_checkpoint is required to decode latents");
    }
    std::vector<float> image(expected);
    const size_t spatial = static_cast<size_t>(cfg_.image_w) * cfg_.image_h;
    for (int y = 0; y < cfg_.image_h; ++y) {
        for (int x = 0; x < cfg_.image_w; ++x) {
            for (int channel = 0; channel < cfg_.image_c; ++channel) {
                const size_t chw = static_cast<size_t>(channel) * spatial +
                                   static_cast<size_t>(y) * cfg_.image_w + x;
                const size_t hwc = (static_cast<size_t>(y) * cfg_.image_w + x) *
                                   cfg_.image_c + channel;
                image[hwc] = latent_chw[chw];
            }
        }
    }
    return image;
}

LumenLatentDiffusionModel::GeneratedImage LumenLatentDiffusionModel::generate(
    const std::string& prompt, int seed, int sample_steps, float guidance_scale,
    const std::function<void(int, int)>& progress) {
    if (getLayers().empty()) throw std::runtime_error("Lumen: model is not built");
    if (sample_steps < 1 || !std::isfinite(guidance_scale))
        throw std::invalid_argument("Lumen: invalid sampling settings");
    struct RestoreOutput {
        json& config;
        json saved;
        ~RestoreOutput() { if (saved.is_null()) config.erase("inference_output_tensor"); else config["inference_output_tensor"] = saved; }
    } restore{modelConfig, modelConfig.value("inference_output_tensor", json())};
    modelConfig["inference_output_tensor"] = "x";
    const DiffusionSchedule schedule(cfg_);
    const int steps = std::min(sample_steps, schedule.size());
    std::vector<int> timesteps;
    for (int i = 0; i < steps; ++i) {
        timesteps.push_back(steps == 1 ? schedule.size() - 1 :
            static_cast<int>(std::lround(double(steps - 1 - i) *
                (schedule.size() - 1) / (steps - 1))));
    }
    const size_t count = static_cast<size_t>(cfg_.latent_w) * cfg_.latent_h * cfg_.latent_c;
    std::mt19937 random(static_cast<unsigned int>(seed));
    std::normal_distribution<float> normal(0.f, 1.f);
    std::vector<float> sample(count);
    for (float& value : sample) value = normal(random);
    const auto conditioned_ids = promptIds(prompt);
    const auto empty_ids = promptIds("");
    for (int i = 0; i < steps; ++i) {
        const int t = timesteps[i];
        const std::unordered_map<std::string, std::vector<float>> inputs{
            {"__input__", sample}, {"timestep", timestepEmbedding(t, std::max(8, cfg_.hidden_size))}};
        std::vector<float> unconditional;
        if (guidance_scale != 1.f)
            unconditional = forwardPassNamed(inputs, {{"text_ids", empty_ids}}, false);
        auto epsilon = forwardPassNamed(inputs, {{"text_ids", conditioned_ids}}, false);
        if (epsilon.size() != count || (guidance_scale != 1.f && unconditional.size() != count))
            throw std::runtime_error("Lumen: denoiser output size mismatch");
        for (size_t j = 0; j < count; ++j) {
            if (guidance_scale != 1.f)
                epsilon[j] = unconditional[j] + guidance_scale * (epsilon[j] - unconditional[j]);
            if (!std::isfinite(epsilon[j])) throw std::runtime_error("Lumen: non-finite noise prediction");
        }
        schedule.advance(sample, epsilon, t, i + 1 < steps ? timesteps[i + 1] : -1);
        if (progress) progress(i + 1, steps);
    }
    const auto decoded = decodeImage(sample);
    const size_t expected = static_cast<size_t>(cfg_.image_w) * cfg_.image_h * cfg_.image_c;
    if (decoded.size() != expected) throw std::runtime_error("Lumen: decoded image size mismatch");
    storeTensor("z", decoded);
    if (isVizTapsEnabled() && cfg_.image_c == 3) {
        addVizTapFrame(rgbTip(cfg_.image_w, cfg_.image_h, getVizTapsMaxSide(),
            "lumen_diffusion/blocks/lumen/output/z/Identity | role=model_output tensor=z",
            "z: sampled RGB [-1,1]", [&](size_t i) { return decoded[i]; }, decodedColour));
    }
    GeneratedImage image;
    image.w = cfg_.image_w;
    image.h = cfg_.image_h;
    image.channels = cfg_.image_c;
    image.pixels.reserve(expected);
    for (float value : decoded) {
        if (!std::isfinite(value)) throw std::runtime_error("Lumen: non-finite decoded image");
        image.pixels.push_back(static_cast<unsigned char>(std::lround(
            std::clamp(value * .5f + .5f, 0.f, 1.f) * 255.f)));
    }
    return image;
}

std::shared_ptr<SkipConnectionControl> LumenLatentDiffusionModel::skipConnectionControl() {
    if (vae_decoder_) return vae_decoder_->skipConnectionControl();
    return Model::skipConnectionControl();
}
