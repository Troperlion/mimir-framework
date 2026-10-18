#include "runtimes/losses/LossDispatch.hpp"
#include "VAETraining.hpp"

#include "../runtimes/ops_loss_and_grad.hpp"
#include "Registry/ModelArchitectures.hpp"
#include "../Serialization/Serialization.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace {

struct DistMoments {
    double mean = 0.0;
    double var = 0.0;
    double skew = 0.0;
};

DistMoments compute_moments_prefix(const std::vector<float>& values, size_t offset, int count) {
    DistMoments moments;
    if (count <= 0 || offset >= values.size()) return moments;
    const size_t end = std::min(values.size(), offset + static_cast<size_t>(count));
    const size_t actual_count = end - offset;
    if (actual_count == 0) return moments;

    for (size_t i = offset; i < end; ++i) moments.mean += static_cast<double>(values[i]);
    moments.mean /= static_cast<double>(actual_count);

    double third = 0.0;
    for (size_t i = offset; i < end; ++i) {
        const double delta = static_cast<double>(values[i]) - moments.mean;
        moments.var += delta * delta;
        third += delta * delta * delta;
    }
    moments.var /= static_cast<double>(actual_count);
    const double stddev = std::sqrt(std::max(0.0, moments.var));
    if (stddev > 1e-12) {
        moments.skew = (third / static_cast<double>(actual_count)) / (stddev * stddev * stddev);
    }
    return moments;
}

double pearson_corr_prefix(const std::vector<float>& lhs, size_t lhs_offset,
                           const std::vector<float>& rhs, size_t rhs_offset,
                           int count) {
    if (count < 2 || lhs_offset >= lhs.size() || rhs_offset >= rhs.size()) return 0.0;
    const size_t actual_count = std::min({static_cast<size_t>(count), lhs.size() - lhs_offset, rhs.size() - rhs_offset});
    if (actual_count < 2) return 0.0;

    double lhs_mean = 0.0;
    double rhs_mean = 0.0;
    for (size_t i = 0; i < actual_count; ++i) {
        lhs_mean += static_cast<double>(lhs[lhs_offset + i]);
        rhs_mean += static_cast<double>(rhs[rhs_offset + i]);
    }
    lhs_mean /= static_cast<double>(actual_count);
    rhs_mean /= static_cast<double>(actual_count);

    double numerator = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    for (size_t i = 0; i < actual_count; ++i) {
        const double a = static_cast<double>(lhs[lhs_offset + i]) - lhs_mean;
        const double b = static_cast<double>(rhs[rhs_offset + i]) - rhs_mean;
        numerator += a * b;
        lhs_norm += a * a;
        rhs_norm += b * b;
    }
    const double denominator = std::sqrt(lhs_norm) * std::sqrt(rhs_norm);
    if (denominator <= 1e-18) return 0.0;
    return std::clamp(numerator / denominator, -1.0, 1.0);
}

double mean_abs_adjacent_diff_prefix(const std::vector<float>& values, size_t offset, size_t count) {
    if (offset >= values.size() || count < 2) return 0.0;
    const size_t actual_count = std::min(count, values.size() - offset);
    if (actual_count < 2) return 0.0;
    double total = 0.0;
    for (size_t i = 1; i < actual_count; ++i) {
        total += std::abs(static_cast<double>(values[offset + i]) - static_cast<double>(values[offset + i - 1]));
    }
    return total / static_cast<double>(actual_count - 1);
}

float dot_f(const std::vector<float>& lhs, size_t lhs_offset,
            const std::vector<float>& rhs, size_t rhs_offset, int count) {
    double total = 0.0;
    for (int i = 0; i < count; ++i) {
        total += static_cast<double>(lhs[lhs_offset + static_cast<size_t>(i)]) *
                 static_cast<double>(rhs[rhs_offset + static_cast<size_t>(i)]);
    }
    return static_cast<float>(total);
}

float norm2_f(const std::vector<float>& values, size_t offset, int count) {
    return std::sqrt(std::max(0.0f, dot_f(values, offset, values, offset, count)));
}

uint64_t mix_u64(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

std::vector<float> make_text_hash_target_unigram(const std::vector<int>& ids,
                                                  int padding_id,
                                                  int dimension,
                                                  uint64_t seed) {
    std::vector<float> target(static_cast<size_t>(std::max(0, dimension)), 0.0f);
    if (dimension <= 0) return target;
    int valid = 0;
    for (int token : ids) {
        if (padding_id >= 0 && token == padding_id) continue;
        const uint64_t hash = mix_u64(static_cast<uint64_t>(static_cast<uint32_t>(token)) ^ seed);
        target[static_cast<size_t>(hash % static_cast<uint64_t>(dimension))] += (hash >> 63) ? 1.0f : -1.0f;
        ++valid;
    }
    if (valid > 0) {
        const float inverse = 1.0f / static_cast<float>(valid);
        for (float& value : target) value *= inverse;
    }
    const float norm = std::max(1e-8f, norm2_f(target, 0, dimension));
    for (float& value : target) value /= norm;
    return target;
}

std::vector<float> make_text_hash_target_bigram(const std::vector<int>& ids,
                                                 int padding_id,
                                                 int dimension,
                                                 uint64_t seed) {
    std::vector<float> target(static_cast<size_t>(std::max(0, dimension)), 0.0f);
    if (dimension <= 0) return target;
    int previous = -1;
    int valid = 0;
    for (int token : ids) {
        if (padding_id >= 0 && token == padding_id) continue;
        if (previous >= 0) {
            const uint64_t pair = (static_cast<uint64_t>(static_cast<uint32_t>(previous)) << 32) ^
                                  static_cast<uint64_t>(static_cast<uint32_t>(token));
            const uint64_t hash = mix_u64(pair ^ seed);
            target[static_cast<size_t>(hash % static_cast<uint64_t>(dimension))] += ((hash >> 62) & 1ULL) ? 1.0f : -1.0f;
            ++valid;
        }
        previous = token;
    }
    if (valid > 0) {
        const float inverse = 1.0f / static_cast<float>(valid);
        for (float& value : target) value *= inverse;
    }
    const float norm = std::max(1e-8f, norm2_f(target, 0, dimension));
    for (float& value : target) value /= norm;
    return target;
}

void validate_model(Model& model, const char* operation) {
    if (model.parametersFrozen()) throw std::runtime_error(std::string(operation) + ": parameters are frozen");
    if (model.getLayers().empty()) throw std::runtime_error(std::string(operation) + ": model not built");
    if (model.layer_weight_blocks.empty()) {
        throw std::runtime_error(std::string(operation) + ": weights not allocated (call allocateParams/initWeights)");
    }
}

void apply_accumulation_scale(std::vector<float>& gradient,
                              Model::TrainStepMode mode,
                              float grad_scale) {
    if (mode != Model::TrainStepMode::Accumulate) return;
    if (!std::isfinite(grad_scale) || grad_scale <= 0.0f) grad_scale = 1.0f;
    for (float& value : gradient) value *= grad_scale;
}

void gradient_diagnostics(const Model& model, float& norm, float& max_abs) {
    double sum_squares = 0.0;
    max_abs = 0.0f;
    for (const auto& layer : model.getLayers()) {
        for (float gradient : layer.grad_weights) {
            sum_squares += static_cast<double>(gradient) * static_cast<double>(gradient);
            max_abs = std::max(max_abs, std::abs(gradient));
        }
        for (float gradient : layer.grad_bias) {
            sum_squares += static_cast<double>(gradient) * static_cast<double>(gradient);
            max_abs = std::max(max_abs, std::abs(gradient));
        }
    }
    norm = static_cast<float>(std::sqrt(sum_squares));
}

void set_metrics(Model::TrainStepResult& result,
                 float recon, float kl, float wass, float spatial, float temporal,
                 float timestep, float align, float beta, int latent_dim,
                 float entropy_diff, float moment_mismatch) {
    result.metrics["mse"] = recon;
    result.metrics["kl"] = kl;
    result.metrics["wass"] = wass;
    result.metrics["wasserstein"] = wass;
    result.metrics["spatial_coherence"] = spatial;
    result.metrics["temp"] = temporal;
    result.metrics["temporal_consistency"] = temporal;
    result.metrics["timestep"] = timestep;
    result.metrics["align"] = align;
    result.metrics["kl_beta_effective"] = beta;
    result.metrics["latent_dim"] = static_cast<float>(latent_dim);
    result.metrics["entropy_diff"] = entropy_diff;
    result.metrics["moment_mismatch"] = moment_mismatch;
}

} // namespace

Model::TrainStepResult VAETraining::trainImage(Model& model,
                                               const std::vector<float>& input,
                                               const std::vector<float>* target,
                                               const std::vector<float>* loss_weights,
                                               Optimizer& optimizer,
                                               float learning_rate,
                                               Model::TrainStepMode mode,
                                               float grad_scale,
                                               const std::vector<SpatialRelation>* relations) {
    model.applyRuntimeConfiguration();
    validate_model(model, "VAETraining::trainImage");

    const std::vector<float>& expected = target ? *target : input;

    int image_dim = model.modelConfig.contains("image_dim")
        ? std::max(0, model.modelConfig["image_dim"].get<int>())
        : static_cast<int>(expected.size());
    if (image_dim <= 0) image_dim = static_cast<int>(expected.size());
    int latent_dim = model.modelConfig.contains("latent_dim")
        ? std::max(0, model.modelConfig["latent_dim"].get<int>()) : 0;

    float kl_beta = 1.0f;
    if (model.modelConfig.contains("kl_beta")) kl_beta = model.modelConfig["kl_beta"].get<float>();
    else if (model.modelConfig.contains("vae_kl_beta")) kl_beta = model.modelConfig["vae_kl_beta"].get<float>();
    kl_beta = std::max(0.0f, kl_beta);
    const int kl_warmup_steps = model.modelConfig.contains("kl_warmup_steps")
        ? std::max(0, model.modelConfig["kl_warmup_steps"].get<int>()) : 0;
    const float kl_progress = kl_warmup_steps > 0
        ? std::min(1.0f, static_cast<float>(optimizer.step + 1) / static_cast<float>(kl_warmup_steps)) : 1.0f;
    const float beta_effective = kl_beta * kl_progress;

    float marker_wass_scale = model.modelConfig.contains("marker_wass_scale")
        ? std::max(0.0f, model.modelConfig["marker_wass_scale"].get<float>()) : 0.0f;
    float marker_temp_scale = model.modelConfig.contains("marker_temp_scale")
        ? std::max(0.0f, model.modelConfig["marker_temp_scale"].get<float>()) : 0.0f;
    const float marker_scale_max = model.modelConfig.contains("marker_scale_max")
        ? std::max(1.0f, model.modelConfig["marker_scale_max"].get<float>()) : 10.0f;
    const int marker_warmup_steps = model.modelConfig.contains("marker_warmup_steps")
        ? std::max(0, model.modelConfig["marker_warmup_steps"].get<int>()) : 0;
    const float marker_progress = marker_warmup_steps > 0
        ? std::min(1.0f, static_cast<float>(optimizer.step + 1) / static_cast<float>(marker_warmup_steps)) : 1.0f;
    marker_wass_scale *= marker_progress;
    marker_temp_scale *= marker_progress;

    float logvar_min = model.modelConfig.contains("logvar_clip_min") ? model.modelConfig["logvar_clip_min"].get<float>() : -10.0f;
    float logvar_max = model.modelConfig.contains("logvar_clip_max") ? model.modelConfig["logvar_clip_max"].get<float>() : 10.0f;
    if (logvar_min > logvar_max) std::swap(logvar_min, logvar_max);

    if (!pending_perceptual_prior_.empty()) {
        Layer* prior = model.getLayerByName("vae_conv/z_prior_bias");
        if (prior && prior->getWeights() && prior->getWeightsSize() == pending_perceptual_prior_.size()) {
            std::copy(pending_perceptual_prior_.begin(), pending_perceptual_prior_.end(), prior->getWeights());
            model.modelConfig["perceptual_prior_hydrated"] = true;
        }
        pending_perceptual_prior_.clear();
    }

    if (mode == Model::TrainStepMode::Optimize) model.zeroGradients();
    const std::vector<float>& prediction = model.forwardPassView(input, true);
    const int output_dim = static_cast<int>(prediction.size());
    if (latent_dim <= 0 && output_dim > image_dim + 2 && ((output_dim - image_dim) % 2) == 0) {
        latent_dim = std::max(1, (output_dim - image_dim) / 2);
    }
    if (image_dim <= 0 || output_dim < image_dim + 2) {
        throw std::runtime_error("VAETraining::trainImage: invalid output/image_dim");
    }
    if (latent_dim <= 0 || output_dim < image_dim + 2 * latent_dim) {
        const int tail = output_dim - image_dim;
        if (tail < 2 || (tail % 2) != 0) throw std::runtime_error("VAETraining::trainImage: cannot infer latent_dim");
        latent_dim = std::max(1, tail / 2);
    }

    const int recon_count = std::min(image_dim, static_cast<int>(expected.size()));
    if (recon_count <= 0) {
        throw std::runtime_error("VAETraining::trainImage: empty reconstruction target");
    }
    std::string recon_loss = "mse";
    if (model.modelConfig.contains("recon_loss")) {
        try { recon_loss = model.modelConfig["recon_loss"].get<std::string>(); } catch (...) {}
    }
    recon_loss = RuntimeLossGrad::canonical_pixel_loss_name(recon_loss);

    const float ssim_weight = model.modelConfig.contains("ssim_weight")
        ? std::max(0.0f, model.modelConfig["ssim_weight"].get<float>()) : 0.0f;
    const float spectral_weight = model.modelConfig.contains("spectral_weight")
        ? std::max(0.0f, model.modelConfig["spectral_weight"].get<float>()) : 0.0f;
    const float perceptual_weight = model.modelConfig.contains("perceptual_weight")
        ? std::max(0.0f, model.modelConfig["perceptual_weight"].get<float>()) : 0.0f;

    int image_w = model.modelConfig.contains("image_w") ? std::max(0, model.modelConfig["image_w"].get<int>()) : 0;
    int image_h = model.modelConfig.contains("image_h") ? std::max(0, model.modelConfig["image_h"].get<int>()) : 0;
    int image_c = model.modelConfig.contains("image_c") ? std::max(0, model.modelConfig["image_c"].get<int>()) : 0;
    if (image_w <= 0 || image_h <= 0 || image_c <= 0) {
        image_c = 1;
        image_w = recon_count;
        image_h = 1;
    }
    const bool reconstruction_is_hwc = recon_count == image_w * image_h * image_c;

    RuntimeLossGrad::PixelLossOptions pixel_options;
    if (model.modelConfig.contains("huber_delta")) pixel_options.huber_delta = std::max(1e-6f, model.modelConfig["huber_delta"].get<float>());
    if (model.modelConfig.contains("smoothl1_delta")) pixel_options.huber_delta = std::max(1e-6f, model.modelConfig["smoothl1_delta"].get<float>());
    if (model.modelConfig.contains("smoothl1_beta")) pixel_options.huber_delta = std::max(1e-6f, model.modelConfig["smoothl1_beta"].get<float>());
    if (model.modelConfig.contains("charbonnier_eps")) pixel_options.charbonnier_eps = std::max(1e-12f, model.modelConfig["charbonnier_eps"].get<float>());
    if (model.modelConfig.contains("nll_sigma")) pixel_options.gaussian_nll_sigma = std::max(1e-6f, model.modelConfig["nll_sigma"].get<float>());
    if (model.modelConfig.contains("gaussian_nll_sigma")) pixel_options.gaussian_nll_sigma = std::max(1e-6f, model.modelConfig["gaussian_nll_sigma"].get<float>());
    pixel_options.weights = loss_weights;
    pixel_options.relations = relations;

    auto pixel = RuntimeLossGrad::pixel_loss_and_grad(prediction.data(), expected.data(), static_cast<size_t>(recon_count), recon_loss, pixel_options);
    double relation_loss = pixel.relation_loss;
    double reconstruction = pixel.loss;
    std::vector<float> reconstruction_gradient = std::move(pixel.grad);

    if (ssim_weight > 0.0f && reconstruction_is_hwc) {
        float k1 = model.modelConfig.contains("ssim_k1") ? model.modelConfig["ssim_k1"].get<float>() : 0.01f;
        float k2 = model.modelConfig.contains("ssim_k2") ? model.modelConfig["ssim_k2"].get<float>() : 0.03f;
        float dynamic_range = model.modelConfig.contains("ssim_L") ? model.modelConfig["ssim_L"].get<float>() : 2.0f;
        std::string ssim_mode = "ssim";
        if (model.modelConfig.contains("ssim_mode")) {
            try { ssim_mode = model.modelConfig["ssim_mode"].get<std::string>(); } catch (...) {}
        }
        if (ssim_mode == "ms_ssim" || ssim_mode == "ms-ssim" || recon_loss == "ms_ssim" || recon_loss == "ms-ssim") {
            std::vector<std::vector<float>> prediction_scales{std::vector<float>(prediction.begin(), prediction.begin() + recon_count)};
            std::vector<std::vector<float>> target_scales{std::vector<float>(expected.begin(), expected.begin() + recon_count)};
            std::vector<int> widths{image_w};
            std::vector<int> heights{image_h};
            int current_w = image_w;
            int current_h = image_h;
            for (int scale = 1; scale < 5; ++scale) {
                if (current_w < 8 || current_h < 8) break;
                std::vector<float> down_prediction;
                std::vector<float> down_target;
                int next_w = 0;
                int next_h = 0;
                RuntimeLossGrad::avgpool2x2_hwc(prediction_scales.back(), current_w, current_h, image_c, down_prediction, next_w, next_h);
                RuntimeLossGrad::avgpool2x2_hwc(target_scales.back(), current_w, current_h, image_c, down_target, next_w, next_h);
                current_w = next_w;
                current_h = next_h;
                prediction_scales.push_back(std::move(down_prediction));
                target_scales.push_back(std::move(down_target));
                widths.push_back(current_w);
                heights.push_back(current_h);
            }
            static const double weights[5] = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333};
            double weight_sum = 0.0;
            for (size_t scale = 0; scale < prediction_scales.size(); ++scale) weight_sum += weights[scale];
            double ms_loss = 0.0;
            std::vector<float> full_gradient(static_cast<size_t>(recon_count), 0.0f);
            for (size_t scale = 0; scale < prediction_scales.size(); ++scale) {
                const double weight = weights[scale] / std::max(1e-12, weight_sum);
                const auto result = RuntimeLossGrad::ssim_global_hwc(prediction_scales[scale], target_scales[scale], widths[scale], heights[scale], image_c, k1, k2, dynamic_range);
                ms_loss += weight * result.loss;
                std::vector<float> gradient = result.grad;
                for (int back = static_cast<int>(scale) - 1; back >= 0; --back) {
                    std::vector<float> upsampled;
                    RuntimeLossGrad::avgpool2x2_back_hwc(gradient, widths[static_cast<size_t>(back)], heights[static_cast<size_t>(back)], image_c, upsampled);
                    gradient.swap(upsampled);
                }
                if (gradient.size() == full_gradient.size()) {
                    for (size_t i = 0; i < gradient.size(); ++i) full_gradient[i] += static_cast<float>(weight) * gradient[i];
                }
            }
            reconstruction += static_cast<double>(ssim_weight) * ms_loss;
            for (int i = 0; i < recon_count; ++i) reconstruction_gradient[static_cast<size_t>(i)] += ssim_weight * full_gradient[static_cast<size_t>(i)];
        } else {
            const std::vector<float> recon(prediction.begin(), prediction.begin() + recon_count);
            const std::vector<float> expected_image(expected.begin(), expected.begin() + recon_count);
            const auto result = RuntimeLossGrad::ssim_global_hwc(recon, expected_image, image_w, image_h, image_c, k1, k2, dynamic_range);
            reconstruction += static_cast<double>(ssim_weight) * result.loss;
            for (int i = 0; i < recon_count; ++i) reconstruction_gradient[static_cast<size_t>(i)] += ssim_weight * result.grad[static_cast<size_t>(i)];
        }
    }

    if (spectral_weight > 0.0f && reconstruction_is_hwc) {
        const int requested_scales = model.modelConfig.contains("spectral_scales")
            ? std::max(1, model.modelConfig["spectral_scales"].get<int>()) : 1;
        std::vector<std::vector<float>> prediction_scales{std::vector<float>(prediction.begin(), prediction.begin() + recon_count)};
        std::vector<std::vector<float>> target_scales{std::vector<float>(expected.begin(), expected.begin() + recon_count)};
        std::vector<int> widths{image_w};
        std::vector<int> heights{image_h};
        int current_w = image_w;
        int current_h = image_h;
        for (int scale = 1; scale < requested_scales; ++scale) {
            if (current_w < 8 || current_h < 8) break;
            std::vector<float> down_prediction;
            std::vector<float> down_target;
            int next_w = 0;
            int next_h = 0;
            RuntimeLossGrad::avgpool2x2_hwc(prediction_scales.back(), current_w, current_h, image_c, down_prediction, next_w, next_h);
            RuntimeLossGrad::avgpool2x2_hwc(target_scales.back(), current_w, current_h, image_c, down_target, next_w, next_h);
            current_w = next_w;
            current_h = next_h;
            prediction_scales.push_back(std::move(down_prediction));
            target_scales.push_back(std::move(down_target));
            widths.push_back(current_w);
            heights.push_back(current_h);
        }
        double weight_sum = 0.0;
        for (size_t scale = 0; scale < prediction_scales.size(); ++scale) weight_sum += std::pow(0.5, static_cast<double>(scale));
        double spectral_loss = 0.0;
        std::vector<float> full_gradient(static_cast<size_t>(recon_count), 0.0f);
        for (size_t scale = 0; scale < prediction_scales.size(); ++scale) {
            const double weight = std::pow(0.5, static_cast<double>(scale)) / std::max(1e-12, weight_sum);
            const auto result = RuntimeLossGrad::spectral_dct_l1_hwc(prediction_scales[scale], target_scales[scale], widths[scale], heights[scale], image_c);
            spectral_loss += weight * result.loss;
            std::vector<float> gradient = result.grad;
            for (int back = static_cast<int>(scale) - 1; back >= 0; --back) {
                std::vector<float> upsampled;
                RuntimeLossGrad::avgpool2x2_back_hwc(gradient, widths[static_cast<size_t>(back)], heights[static_cast<size_t>(back)], image_c, upsampled);
                gradient.swap(upsampled);
            }
            if (gradient.size() == full_gradient.size()) {
                for (size_t i = 0; i < gradient.size(); ++i) full_gradient[i] += static_cast<float>(weight) * gradient[i];
            }
        }
        reconstruction += static_cast<double>(spectral_weight) * spectral_loss;
        for (int i = 0; i < recon_count; ++i) reconstruction_gradient[static_cast<size_t>(i)] += spectral_weight * full_gradient[static_cast<size_t>(i)];
    }

    if (perceptual_weight > 0.0f && reconstruction_is_hwc) {
        std::string architecture = "vgg16_feat";
        if (model.modelConfig.contains("perceptual_arch")) {
            try { architecture = model.modelConfig["perceptual_arch"].get<std::string>(); } catch (...) {}
        }
        if (!perceptual_model_) {
            json config = ModelArchitectures::defaultConfig(architecture);
            config["image_w"] = image_w;
            config["image_h"] = image_h;
            config["image_c"] = image_c;
            if (model.modelConfig.contains("perceptual_base_channels")) {
                int channels = std::max(1, model.modelConfig["perceptual_base_channels"].get<int>());
                if (architecture == "vgg16_feat") channels = std::max(4, channels);
                config["base_channels"] = channels;
            }
            perceptual_model_ = ModelArchitectures::create(architecture, config);
            perceptual_model_->allocateParams();
            try { perceptual_model_->initializeWeights("xavier", 1337u); } catch (...) {}

            std::string checkpoint;
            if (model.modelConfig.contains("perceptual_checkpoint")) {
                try { checkpoint = model.modelConfig["perceptual_checkpoint"].get<std::string>(); } catch (...) {}
            }
            if (!checkpoint.empty()) {
                Mimir::Serialization::LoadOptions options;
                options.format = Mimir::Serialization::detect_format(checkpoint);
                options.load_tokenizer = false;
                options.load_encoder = false;
                options.load_optimizer = false;
                options.strict_mode = false;
                options.validate_checksums = false;
                std::string error;
                if (!Mimir::Serialization::load_checkpoint(*perceptual_model_, checkpoint, options, &error)) {
                    bool retried = false;
                    if (options.format == Mimir::Serialization::CheckpointFormat::RawFolder) {
                        try {
                            const std::filesystem::path architecture_path = std::filesystem::path(checkpoint) / "model" / "architecture.json";
                            if (std::filesystem::exists(architecture_path)) {
                                std::ifstream stream(architecture_path);
                                json saved_architecture;
                                stream >> saved_architecture;
                                int inferred_channels = 0;
                                if (saved_architecture.is_object() && saved_architecture.contains("layers") && saved_architecture["layers"].is_array()) {
                                    for (const auto& layer : saved_architecture["layers"]) {
                                        if (layer.is_object() && layer.value("name", std::string()) == "vgg16_feat/b1/c1") {
                                            inferred_channels = layer.value("out_channels", 0);
                                            break;
                                        }
                                    }
                                }
                                const int configured_channels = config.contains("base_channels") ? config["base_channels"].get<int>() : 0;
                                if (inferred_channels > 0 && inferred_channels != configured_channels) {
                                    json retry_config = config;
                                    retry_config["base_channels"] = inferred_channels;
                                    perceptual_model_ = ModelArchitectures::create(architecture, retry_config);
                                    perceptual_model_->allocateParams();
                                    try { perceptual_model_->initializeWeights("xavier", 1337u); } catch (...) {}
                                    std::string retry_error;
                                    retried = Mimir::Serialization::load_checkpoint(*perceptual_model_, checkpoint, options, &retry_error);
                                    if (!retried) error = retry_error;
                                }
                            }
                        } catch (...) {}
                    }
                    if (!retried) std::cerr << "Perceptual checkpoint load failed: " << checkpoint << " | " << error << std::endl;
                }
            } else {
                std::cerr << "Perceptual loss active but perceptual_checkpoint is empty: using fixed random Xavier init (seed=1337)." << std::endl;
            }
        }

        const std::vector<float> reconstructed(prediction.begin(), prediction.begin() + recon_count);
        const std::vector<float> expected_image(expected.begin(), expected.begin() + recon_count);
        perceptual_model_->zeroGradients();
        const std::vector<float>& real_view = perceptual_model_->forwardPassView(expected_image, true);
        std::vector<float> real_features(real_view.begin(), real_view.end());
        if (!real_features.empty()) {
            Layer* prior = model.getLayerByName("vae_conv/z_prior_bias");
            if (prior && prior->getWeights() && prior->getWeightsSize() > 0) {
                float momentum = model.modelConfig.contains("perceptual_prior_momentum")
                    ? model.modelConfig["perceptual_prior_momentum"].get<float>() : 0.95f;
                float scale = model.modelConfig.contains("perceptual_prior_scale")
                    ? model.modelConfig["perceptual_prior_scale"].get<float>() : 0.05f;
                momentum = std::clamp(momentum, 0.0f, 0.9999f);
                scale = std::max(0.0f, scale);
                double mean = 0.0;
                for (float value : real_features) mean += static_cast<double>(value);
                mean /= static_cast<double>(real_features.size());
                double variance = 0.0;
                for (float value : real_features) {
                    const double delta = static_cast<double>(value) - mean;
                    variance += delta * delta;
                }
                variance /= static_cast<double>(real_features.size());
                const float inverse_stddev = 1.0f / std::sqrt(static_cast<float>(variance) + 1e-6f);
                pending_perceptual_prior_.resize(prior->getWeightsSize());
                const float* current = prior->getWeights();
                for (size_t i = 0; i < prior->getWeightsSize(); ++i) {
                    const float feature = (real_features[i % real_features.size()] - static_cast<float>(mean)) * inverse_stddev;
                    const float desired = scale * std::clamp(feature, -3.0f, 3.0f);
                    pending_perceptual_prior_[i] = momentum * current[i] + (1.0f - momentum) * desired;
                }
            }
        }

        perceptual_model_->zeroGradients();
        const std::vector<float>& fake_features = perceptual_model_->forwardPassView(reconstructed, true);
        const size_t feature_count = std::min(fake_features.size(), real_features.size());
        if (feature_count > 0) {
            auto feature_loss=RuntimeLossGrad::pixel_loss_and_grad(fake_features.data(),real_features.data(),feature_count,"mse");
            reconstruction += static_cast<double>(perceptual_weight)*feature_loss.loss;
            perceptual_model_->backwardPass(feature_loss.grad);
            if (perceptual_model_->hasLastInputGradient()) {
                const auto& input_gradient = perceptual_model_->getLastInputGradient();
                if (input_gradient.size() >= static_cast<size_t>(recon_count)) {
                    for (int i = 0; i < recon_count; ++i) reconstruction_gradient[static_cast<size_t>(i)] += perceptual_weight * input_gradient[static_cast<size_t>(i)];
                }
            }
        }
        perceptual_model_->releaseTrainingWorkingSet(optimizer.step);
    }

    const int mu_offset = image_dim;
    const int logvar_offset = image_dim + latent_dim;
    double kl = 0.0;
    for (int i = 0; i < latent_dim; ++i) {
        const float mu = prediction[static_cast<size_t>(mu_offset + i)];
        const float raw_logvar = prediction[static_cast<size_t>(logvar_offset + i)];
        const float logvar = std::clamp(raw_logvar, logvar_min, logvar_max);
        kl += 0.5 * (static_cast<double>(mu) * mu + std::exp(static_cast<double>(logvar)) - 1.0 - logvar);
    }
    kl /= static_cast<double>(std::max(1, latent_dim));

    const auto prediction_moments = compute_moments_prefix(prediction, 0, recon_count);
    const auto target_moments = compute_moments_prefix(expected, 0, recon_count);
    const double prediction_variance = std::max(prediction_moments.var, 1e-12);
    const double target_variance = std::max(target_moments.var, 1e-12);
    const double wasserstein_squared = std::pow(target_moments.mean - prediction_moments.mean, 2.0) +
        std::pow(std::sqrt(target_variance) - std::sqrt(prediction_variance), 2.0);
    const float wasserstein = static_cast<float>(std::sqrt(std::max(0.0, wasserstein_squared)));
    const float spatial = static_cast<float>(std::abs(
        mean_abs_adjacent_diff_prefix(prediction, 0, static_cast<size_t>(recon_count)) -
        mean_abs_adjacent_diff_prefix(expected, 0, static_cast<size_t>(recon_count))));
    const float temporal = static_cast<float>(pearson_corr_prefix(prediction, 0, expected, 0, recon_count));
    const float temporal_penalty = 1.0f - std::clamp(temporal, -1.0f, 1.0f);
    const float entropy_diff = static_cast<float>(0.5 * (std::log(prediction_variance) - std::log(target_variance)));
    const float moment_mismatch = static_cast<float>(std::abs(prediction_moments.skew - target_moments.skew));

    float marker_scale = 1.0f;
    if (marker_wass_scale > 0.0f || marker_temp_scale > 0.0f) {
        marker_scale = std::clamp(1.0f + marker_wass_scale * wasserstein + marker_temp_scale * temporal_penalty,
                                  0.1f, marker_scale_max);
    }

    static thread_local std::vector<float> packed_gradient;
    packed_gradient.assign(static_cast<size_t>(output_dim), 0.0f);
    for (int i = 0; i < recon_count; ++i) packed_gradient[static_cast<size_t>(i)] = marker_scale * reconstruction_gradient[static_cast<size_t>(i)];
    const float kl_scale = latent_dim > 0 ? beta_effective / static_cast<float>(latent_dim) : 0.0f;
    for (int i = 0; i < latent_dim; ++i) {
        const float mu = prediction[static_cast<size_t>(mu_offset + i)];
        const float raw_logvar = prediction[static_cast<size_t>(logvar_offset + i)];
        const float logvar = std::clamp(raw_logvar, logvar_min, logvar_max);
        const float in_range = raw_logvar >= logvar_min && raw_logvar <= logvar_max ? 1.0f : 0.0f;
        packed_gradient[static_cast<size_t>(mu_offset + i)] = kl_scale * mu;
        packed_gradient[static_cast<size_t>(logvar_offset + i)] = kl_scale * 0.5f * (std::exp(logvar) - 1.0f) * in_range;
    }
    apply_accumulation_scale(packed_gradient, mode, grad_scale);
    model.backwardPass(packed_gradient);

    Model::TrainStepResult result;
    result.metrics["relation_loss"] = static_cast<float>(relation_loss);
    result.metrics["relation_count"] = relations ? static_cast<float>(relations->size()) : 0.0f;
    result.loss = static_cast<float>(static_cast<double>(marker_scale) * reconstruction + static_cast<double>(beta_effective) * kl);
    gradient_diagnostics(model, result.grad_norm, result.grad_max_abs);
    if (mode == Model::TrainStepMode::Optimize) model.optimizerStep(optimizer, learning_rate);
    const float timestep = std::clamp(static_cast<float>(optimizer.step + 1) / static_cast<float>(std::max(1, optimizer.total_steps)), 0.0f, 1.0f);
    set_metrics(result, static_cast<float>(reconstruction), static_cast<float>(kl), wasserstein, spatial,
                temporal, timestep, 0.0f, beta_effective, latent_dim, entropy_diff, moment_mismatch);
    return result;
}

Model::TrainStepResult VAETraining::trainText(Model& model,
                                              const std::vector<float>& input,
                                              const std::vector<int>& text_ids,
                                              const std::vector<float>* explicit_target,
                                              const std::vector<float>* loss_weights,
                                              Optimizer& optimizer,
                                              float learning_rate,
                                              Model::TrainStepMode mode,
                                              float grad_scale,
                                               const std::vector<SpatialRelation>* relations) {
    model.applyRuntimeConfiguration();
    validate_model(model, "VAETraining::trainText");

    std::string recon_loss = "mse";
    if (model.modelConfig.contains("recon_loss")) {
        try { recon_loss = model.modelConfig["recon_loss"].get<std::string>(); } catch (...) {}
    }
    const bool reconstruction_is_ce = recon_loss == "ce" || recon_loss == "cross_entropy" || recon_loss == "xent";
    const std::vector<float>* target = explicit_target ? explicit_target : &input;
    bool using_internal_target = false;

    int image_dim = model.modelConfig.contains("image_dim") ? std::max(0, model.modelConfig["image_dim"].get<int>()) : 0;
    const int seq_len = model.modelConfig.contains("seq_len") ? std::max(0, model.modelConfig["seq_len"].get<int>()) : 0;
    const int vocab_size = model.modelConfig.contains("vocab_size") ? std::max(0, model.modelConfig["vocab_size"].get<int>()) : 0;
    const int padding_id = model.modelConfig.contains("padding_idx") ? model.modelConfig["padding_idx"].get<int>() : -1;
    if (reconstruction_is_ce) image_dim = std::max(1, std::max(1, seq_len) * std::max(2, vocab_size));
    else if (explicit_target) image_dim = static_cast<int>(explicit_target->size());
    else if (!input.empty()) image_dim = image_dim > 0 ? image_dim : static_cast<int>(input.size());

    int latent_dim = model.modelConfig.contains("latent_dim") ? std::max(0, model.modelConfig["latent_dim"].get<int>()) : 0;
    if (model.modelConfig.contains("latent_tokens") && model.modelConfig.contains("d_model")) {
        latent_dim = std::max(1, model.modelConfig["latent_tokens"].get<int>()) * std::max(1, model.modelConfig["d_model"].get<int>());
    }
    int projection_dim = model.modelConfig.contains("proj_dim") ? std::max(0, model.modelConfig["proj_dim"].get<int>()) : 0;
    int semantic_dim = model.modelConfig.contains("context_semantic_dim") ? std::max(0, model.modelConfig["context_semantic_dim"].get<int>()) : 0;
    int thematic_dim = model.modelConfig.contains("context_thematic_dim") ? std::max(0, model.modelConfig["context_thematic_dim"].get<int>()) : 0;
    int dialog_dim = model.modelConfig.contains("context_dialog_dim") ? std::max(0, model.modelConfig["context_dialog_dim"].get<int>()) : 0;
    const float semantic_weight = model.modelConfig.contains("context_semantic_weight") ? std::max(0.0f, model.modelConfig["context_semantic_weight"].get<float>()) : 0.0f;
    const float thematic_weight = model.modelConfig.contains("context_thematic_weight") ? std::max(0.0f, model.modelConfig["context_thematic_weight"].get<float>()) : 0.0f;
    const float dialog_weight = model.modelConfig.contains("context_dialog_weight") ? std::max(0.0f, model.modelConfig["context_dialog_weight"].get<float>()) : 0.0f;
    const float align_weight = model.modelConfig.contains("align_weight") ? std::max(0.0f, model.modelConfig["align_weight"].get<float>()) : 0.1f;

    float kl_beta = model.modelConfig.contains("kl_beta") ? model.modelConfig["kl_beta"].get<float>()
        : (model.modelConfig.contains("vae_kl_beta") ? model.modelConfig["vae_kl_beta"].get<float>() : 1.0f);
    kl_beta = std::max(0.0f, kl_beta);
    const int kl_warmup_steps = model.modelConfig.contains("kl_warmup_steps") ? std::max(0, model.modelConfig["kl_warmup_steps"].get<int>()) : 0;
    const float beta_effective = kl_beta * (kl_warmup_steps > 0
        ? std::min(1.0f, static_cast<float>(optimizer.step + 1) / static_cast<float>(kl_warmup_steps)) : 1.0f);

    float marker_wass_scale = model.modelConfig.contains("marker_wass_scale") ? std::max(0.0f, model.modelConfig["marker_wass_scale"].get<float>()) : 0.0f;
    float marker_temp_scale = model.modelConfig.contains("marker_temp_scale") ? std::max(0.0f, model.modelConfig["marker_temp_scale"].get<float>()) : 0.0f;
    const float marker_scale_max = model.modelConfig.contains("marker_scale_max") ? std::max(1.0f, model.modelConfig["marker_scale_max"].get<float>()) : 10.0f;
    const int marker_warmup_steps = model.modelConfig.contains("marker_warmup_steps") ? std::max(0, model.modelConfig["marker_warmup_steps"].get<int>()) : 0;
    const float marker_progress = marker_warmup_steps > 0
        ? std::min(1.0f, static_cast<float>(optimizer.step + 1) / static_cast<float>(marker_warmup_steps)) : 1.0f;
    marker_wass_scale *= marker_progress;
    marker_temp_scale *= marker_progress;

    float logvar_min = model.modelConfig.contains("logvar_clip_min") ? model.modelConfig["logvar_clip_min"].get<float>() : -10.0f;
    float logvar_max = model.modelConfig.contains("logvar_clip_max") ? model.modelConfig["logvar_clip_max"].get<float>() : 10.0f;
    if (logvar_min > logvar_max) std::swap(logvar_min, logvar_max);

    if (mode == Model::TrainStepMode::Optimize) model.zeroGradients();
    std::unordered_map<std::string, std::vector<float>> float_inputs{{"__input__", input}};
    std::unordered_map<std::string, std::vector<int>> int_inputs{{"text_ids", text_ids}};
    const std::vector<float>& prediction = model.forwardPassNamedView(float_inputs, int_inputs, true);
    const int output_dim = static_cast<int>(prediction.size());

    if (!reconstruction_is_ce && !explicit_target && input.empty()) {
        std::string target_name;
        if (model.modelConfig.contains("target_tensor")) {
            try { target_name = model.modelConfig["target_tensor"].get<std::string>(); } catch (...) {}
        }
        if (!target_name.empty() && model.hasTensor(target_name)) target = &model.getTensor(target_name);
        else if (model.hasTensor("vae_text/target")) target = &model.getTensor("vae_text/target");
        using_internal_target = target != &input && !target->empty();
        if (using_internal_target) image_dim = static_cast<int>(target->size());
        else if (model.modelConfig.contains("seq_len") && model.modelConfig.contains("d_model")) {
            image_dim = std::max(1, model.modelConfig["seq_len"].get<int>()) * std::max(1, model.modelConfig["d_model"].get<int>());
        }
    }

    if (latent_dim <= 0) {
        const int tail = output_dim - image_dim;
        if (tail >= 4 && (tail % 2) == 0) latent_dim = std::max(1, tail / 2);
    }
    if (image_dim <= 0 || latent_dim <= 0 || output_dim < image_dim + 2 * latent_dim + 2) {
        throw std::runtime_error("VAETraining::trainText: invalid dimensions");
    }
    const int projection_tail = output_dim - (image_dim + 2 * latent_dim);
    if (projection_dim <= 0 && projection_tail > 0 && (projection_tail % 2) == 0) projection_dim = std::max(1, projection_tail / 2);
    if (projection_dim <= 0 || output_dim < image_dim + 2 * latent_dim + 2 * projection_dim) {
        throw std::runtime_error("VAETraining::trainText: missing/invalid proj_dim");
    }

    const int base_dim = image_dim + 2 * latent_dim + 2 * projection_dim;
    const int extra_tail = std::max(0, output_dim - base_dim);
    if (semantic_dim + thematic_dim + dialog_dim <= 0 && extra_tail > 0) semantic_dim = extra_tail;
    const int requested_extra = semantic_dim + thematic_dim + dialog_dim;
    if (requested_extra > extra_tail) {
        const int overflow = requested_extra - extra_tail;
        if (dialog_dim >= overflow) dialog_dim -= overflow;
        else if (thematic_dim >= overflow) thematic_dim -= overflow;
        else semantic_dim = std::max(0, semantic_dim - overflow);
    }

    double reconstruction = 0.0;
    float wasserstein = 0.0f;
    float spatial = 0.0f;
    float temporal = 0.0f;
    float entropy_diff = 0.0f;
    float moment_mismatch = 0.0f;
    float marker_scale = 1.0f;
    const int recon_count = reconstruction_is_ce ? 0 : std::min(image_dim, static_cast<int>(target->size()));
    std::vector<float> reconstruction_gradient;

    double relation_loss = 0;
    if (reconstruction_is_ce) {
        auto ce=RuntimeLossGrad::text_cross_entropy(prediction.data(),prediction.size(),text_ids,seq_len,vocab_size,padding_id);
        reconstruction=ce.loss;
        reconstruction_gradient=std::move(ce.grad);
    } else {
        RuntimeLossGrad::PixelLossOptions pixel_options;
        pixel_options.weights = loss_weights;
        pixel_options.relations = relations;
        auto pixel = RuntimeLossGrad::pixel_loss_and_grad(
            prediction.data(), target->data(), static_cast<size_t>(recon_count), recon_loss, pixel_options);
        relation_loss = pixel.relation_loss;
        reconstruction = pixel.loss;
        reconstruction_gradient = std::move(pixel.grad);
        const auto prediction_moments = compute_moments_prefix(prediction, 0, recon_count);
        const auto target_moments = compute_moments_prefix(*target, 0, recon_count);
        const double prediction_variance = std::max(prediction_moments.var, 1e-12);
        const double target_variance = std::max(target_moments.var, 1e-12);
        wasserstein = static_cast<float>(std::sqrt(std::max(0.0,
            std::pow(target_moments.mean - prediction_moments.mean, 2.0) +
            std::pow(std::sqrt(target_variance) - std::sqrt(prediction_variance), 2.0))));
        spatial = static_cast<float>(std::abs(mean_abs_adjacent_diff_prefix(prediction, 0, static_cast<size_t>(recon_count)) -
                                              mean_abs_adjacent_diff_prefix(*target, 0, static_cast<size_t>(recon_count))));
        temporal = static_cast<float>(pearson_corr_prefix(prediction, 0, *target, 0, recon_count));
        entropy_diff = static_cast<float>(0.5 * (std::log(prediction_variance) - std::log(target_variance)));
        moment_mismatch = static_cast<float>(std::abs(prediction_moments.skew - target_moments.skew));
        if (marker_wass_scale > 0.0f || marker_temp_scale > 0.0f) {
            marker_scale = std::clamp(1.0f + marker_wass_scale * wasserstein +
                                      marker_temp_scale * (1.0f - std::clamp(temporal, -1.0f, 1.0f)),
                                      0.1f, marker_scale_max);
        }
    }

    const int mu_offset = image_dim;
    const int logvar_offset = image_dim + latent_dim;
    double kl = 0.0;
    for (int i = 0; i < latent_dim; ++i) {
        const float mu = prediction[static_cast<size_t>(mu_offset + i)];
        const float logvar = std::clamp(prediction[static_cast<size_t>(logvar_offset + i)], logvar_min, logvar_max);
        kl += 0.5 * (static_cast<double>(mu) * mu + std::exp(static_cast<double>(logvar)) - 1.0 - logvar);
    }
    kl /= static_cast<double>(std::max(1, latent_dim));

    const size_t image_projection_offset = static_cast<size_t>(image_dim + 2 * latent_dim);
    const size_t text_projection_offset = image_projection_offset + static_cast<size_t>(projection_dim);
    const size_t semantic_offset = text_projection_offset + static_cast<size_t>(projection_dim);
    const size_t thematic_offset = semantic_offset + static_cast<size_t>(semantic_dim);
    const size_t dialog_offset = thematic_offset + static_cast<size_t>(thematic_dim);
    const float epsilon = 1e-8f;
    const float image_norm = std::max(epsilon, norm2_f(prediction, image_projection_offset, projection_dim));
    const float text_norm = std::max(epsilon, norm2_f(prediction, text_projection_offset, projection_dim));
    const float cosine = dot_f(prediction, image_projection_offset, prediction, text_projection_offset, projection_dim) / (image_norm * text_norm);
    const float alignment = align_weight > 0.0f ? align_weight * (1.0f - cosine) : 0.0f;

    const auto semantic_target = make_text_hash_target_unigram(text_ids, padding_id, semantic_dim, 0x9e3779b97f4a7c15ULL);
    const auto thematic_target = make_text_hash_target_bigram(text_ids, padding_id, thematic_dim, 0xbf58476d1ce4e5b9ULL);
    const auto dialog_target = make_text_hash_target_unigram(text_ids, padding_id, dialog_dim, 0x94d049bb133111ebULL);
    auto context_loss = [&](size_t offset, int dimension, const std::vector<float>& expected, float weight) {
        if (dimension <= 0 || weight <= 0.0f) return 0.0;
        double loss = 0.0;
        for (int i = 0; i < dimension; ++i) {
            const double delta = static_cast<double>(prediction[offset + static_cast<size_t>(i)]) - expected[static_cast<size_t>(i)];
            loss += delta * delta;
        }
        return loss / static_cast<double>(dimension);
    };
    const double semantic_loss = context_loss(semantic_offset, semantic_dim, semantic_target, semantic_weight);
    const double thematic_loss = context_loss(thematic_offset, thematic_dim, thematic_target, thematic_weight);
    const double dialog_loss = context_loss(dialog_offset, dialog_dim, dialog_target, dialog_weight);

    static thread_local std::vector<float> packed_gradient;
    packed_gradient.assign(static_cast<size_t>(output_dim), 0.0f);
    if (reconstruction_is_ce) {
        for(size_t i=0;i<reconstruction_gradient.size();++i)packed_gradient[i]=marker_scale*reconstruction_gradient[i];
    } else {
        for (int i = 0; i < recon_count; ++i) {
            packed_gradient[static_cast<size_t>(i)] = marker_scale * reconstruction_gradient[static_cast<size_t>(i)];
        }
    }

    const float kl_scale = latent_dim > 0 ? beta_effective / static_cast<float>(latent_dim) : 0.0f;
    for (int i = 0; i < latent_dim; ++i) {
        const float mu = prediction[static_cast<size_t>(mu_offset + i)];
        const float raw_logvar = prediction[static_cast<size_t>(logvar_offset + i)];
        const float logvar = std::clamp(raw_logvar, logvar_min, logvar_max);
        packed_gradient[static_cast<size_t>(mu_offset + i)] = kl_scale * mu;
        packed_gradient[static_cast<size_t>(logvar_offset + i)] = kl_scale * 0.5f * (std::exp(logvar) - 1.0f) *
            (raw_logvar >= logvar_min && raw_logvar <= logvar_max ? 1.0f : 0.0f);
    }
    if (align_weight > 0.0f) {
        const float inverse_image_norm = 1.0f / image_norm;
        const float inverse_text_norm = 1.0f / text_norm;
        for (int i = 0; i < projection_dim; ++i) {
            const float image_value = prediction[image_projection_offset + static_cast<size_t>(i)];
            const float text_value = prediction[text_projection_offset + static_cast<size_t>(i)];
            const float image_unit = image_value * inverse_image_norm;
            const float text_unit = text_value * inverse_text_norm;
            packed_gradient[image_projection_offset + static_cast<size_t>(i)] = -align_weight * (text_unit - cosine * image_unit) * inverse_image_norm;
            packed_gradient[text_projection_offset + static_cast<size_t>(i)] = -align_weight * (image_unit - cosine * text_unit) * inverse_text_norm;
        }
    }
    auto write_context_gradient = [&](size_t offset, int dimension, const std::vector<float>& expected, float weight) {
        if (dimension <= 0 || weight <= 0.0f) return;
        const float scale = 2.0f * weight / static_cast<float>(dimension);
        for (int i = 0; i < dimension; ++i) {
            const size_t index = offset + static_cast<size_t>(i);
            packed_gradient[index] = scale * (prediction[index] - expected[static_cast<size_t>(i)]);
        }
    };
    write_context_gradient(semantic_offset, semantic_dim, semantic_target, semantic_weight);
    write_context_gradient(thematic_offset, thematic_dim, thematic_target, thematic_weight);
    write_context_gradient(dialog_offset, dialog_dim, dialog_target, dialog_weight);

    apply_accumulation_scale(packed_gradient, mode, grad_scale);
    model.backwardPass(packed_gradient);

    Model::TrainStepResult result;
    result.metrics["relation_loss"] = static_cast<float>(relation_loss);
    result.metrics["relation_count"] = relations ? static_cast<float>(relations->size()) : 0.0f;
    result.loss = static_cast<float>(static_cast<double>(marker_scale) * reconstruction +
                                     static_cast<double>(beta_effective) * kl + alignment +
                                     static_cast<double>(semantic_weight) * semantic_loss +
                                     static_cast<double>(thematic_weight) * thematic_loss +
                                     static_cast<double>(dialog_weight) * dialog_loss);
    gradient_diagnostics(model, result.grad_norm, result.grad_max_abs);
    if (mode == Model::TrainStepMode::Optimize) model.optimizerStep(optimizer, learning_rate);
    const float timestep = std::clamp(static_cast<float>(optimizer.step + 1) / static_cast<float>(std::max(1, optimizer.total_steps)), 0.0f, 1.0f);
    set_metrics(result, static_cast<float>(reconstruction), static_cast<float>(kl), wasserstein, spatial,
                temporal, timestep, alignment, beta_effective, latent_dim, entropy_diff, moment_mismatch);
    return result;
}
