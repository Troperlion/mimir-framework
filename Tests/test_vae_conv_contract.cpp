#include "test_utils.hpp"

#include "Model.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Models/Vision/VAEConvModel.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>

int main() {
    // Canonical keys independently control residual blocks and attention.
    auto registry_vae = ModelArchitectures::create("vae_conv", {
        {"image_w", 4}, {"image_h", 4}, {"image_c", 1},
        {"latent_w", 2}, {"latent_h", 2}, {"latent_c", 2},
        {"base_channels", 8}, {"resnet", true}, {"attention", false},
        {"enc_norm", "none"}, {"dec_norm", "none"}
    });
    bool saw_residual_add = false;
    bool saw_self_attention = false;
    for (const auto& layer : registry_vae->getLayers()) {
        saw_residual_add = saw_residual_add || layer.name.find("/res/add") != std::string::npos;
        saw_self_attention = saw_self_attention || layer.type == "SelfAttention";
    }
    TASSERT_TRUE(saw_residual_add);
    TASSERT_TRUE(!saw_self_attention);

    auto legacy_registry_vae = ModelArchitectures::create("vae_conv", {
        {"image_w", 4}, {"image_h", 4}, {"image_c", 1},
        {"latent_w", 2}, {"latent_h", 2}, {"latent_c", 2},
        {"base_channels", 8}, {"use_attention", false},
        {"use_attn", true}, {"enc_norm", "none"}, {"dec_norm", "none"}
    });
    bool legacy_saw_residual_add = false;
    bool legacy_saw_self_attention = false;
    for (const auto& layer : legacy_registry_vae->getLayers()) {
        legacy_saw_residual_add = legacy_saw_residual_add ||
                                  layer.name.find("/res/add") != std::string::npos;
        legacy_saw_self_attention = legacy_saw_self_attention ||
                                    layer.type == "SelfAttention";
    }
    TASSERT_TRUE(!legacy_saw_residual_add);
    TASSERT_TRUE(legacy_saw_self_attention);

    // Rectangular inputs are valid when both axes use the same power-of-two
    // downsampling ratio (8x4 -> 4x2 here).
    VAEConvModel::Config rectangular_cfg;
    rectangular_cfg.image_w = 8;
    rectangular_cfg.image_h = 4;
    rectangular_cfg.image_c = 1;
    rectangular_cfg.latent_w = 4;
    rectangular_cfg.latent_h = 2;
    rectangular_cfg.latent_c = 1;
    rectangular_cfg.base_channels = 8;
    rectangular_cfg.resnet = false;
    rectangular_cfg.attention = false;
    VAEConvModel rectangular_vae;
    rectangular_vae.buildFromConfig(rectangular_cfg);
    TASSERT_TRUE(!rectangular_vae.getLayers().empty());

    // Decoder-only checkpoints must preserve the full model topology even for
    // base channel counts below the historical decoder minimum of eight.
    VAEConvModel::Config checkpoint_cfg = rectangular_cfg;
    checkpoint_cfg.base_channels = 6;
    checkpoint_cfg.dec_norm = "layernorm";
    VAEConvModel checkpoint_full_model;
    checkpoint_full_model.buildFromConfig(checkpoint_cfg);
    Model checkpoint_decoder_model;
    VAEConvModel::buildDecoderInto(checkpoint_decoder_model, checkpoint_cfg);
    for (const Layer& full_layer : checkpoint_full_model.getLayers()) {
        if (full_layer.name.find("vae_conv/dec/") != 0) continue;
        const Layer* decoder_layer = checkpoint_decoder_model.getLayerByName(full_layer.name);
        TASSERT_TRUE(decoder_layer != nullptr);
        TASSERT_TRUE(decoder_layer->params_count == full_layer.params_count);
    }

    // Decoder upsampling is selectable while preserving the same output shape.
    const auto assert_upsample_graph = [&](const std::string& mode,
                                           const std::string& up_type) -> int {
        VAEConvModel::Config upsample_cfg = rectangular_cfg;
        upsample_cfg.decoder_upsample = mode;

        VAEConvModel full_model;
        full_model.buildFromConfig(upsample_cfg);
        Model decoder_model;
        VAEConvModel::buildDecoderInto(decoder_model, upsample_cfg);

        for (Model* candidate : std::vector<Model*>{&full_model, &decoder_model}) {
            const Layer* up = candidate->getLayerByName("vae_conv/dec/up1/up");
            TASSERT_TRUE(up != nullptr);
            TASSERT_TRUE(up->type == up_type);

            const Layer* conv = candidate->getLayerByName("vae_conv/dec/up1/conv");
            TASSERT_TRUE(conv != nullptr);
            TASSERT_TRUE(conv->type == "Conv2d");
            if (mode == "pixel_shuffle") {
                TASSERT_TRUE(conv->out_channels == 4 * upsample_cfg.base_channels);
                TASSERT_TRUE(up->in_channels == 4 * upsample_cfg.base_channels);
                TASSERT_TRUE(up->out_channels == upsample_cfg.base_channels);
            }
        }

        decoder_model.allocateParams();
        decoder_model.initializeWeights("xavier", 123u);
        const std::vector<float> latent(
            static_cast<size_t>(upsample_cfg.latent_w) * upsample_cfg.latent_h *
                upsample_cfg.latent_c,
            0.25f);
        const std::vector<float> decoded = decoder_model.forwardPass(latent, false);
        TASSERT_TRUE(decoded.size() ==
                     static_cast<size_t>(upsample_cfg.image_w) * upsample_cfg.image_h *
                         upsample_cfg.image_c);
        return 0;
    };
    TASSERT_TRUE(assert_upsample_graph("nearest_conv", "UpsampleNearest") == 0);
    TASSERT_TRUE(assert_upsample_graph("bilinear_conv", "UpsampleBilinear") == 0);
    TASSERT_TRUE(assert_upsample_graph("pixel_shuffle", "PixelShuffle") == 0);

    VAEConvModel::Config layernorm_cfg = rectangular_cfg;
    layernorm_cfg.enc_norm = "layernorm";
    layernorm_cfg.dec_norm = "layernorm";
    Model layernorm_decoder;
    VAEConvModel::buildDecoderInto(layernorm_decoder, layernorm_cfg);
    for (const Layer& layer : layernorm_decoder.getLayers()) {
        if (layer.type_enum != LayerType::LayerNorm) continue;
        const int spatial_size = layer.input_height * layer.input_width;
        TASSERT_TRUE(spatial_size > 0);
        TASSERT_TRUE(layer.in_features == spatial_size);
        TASSERT_TRUE(layer.params_count == static_cast<size_t>(spatial_size) * 2);
    }
    layernorm_decoder.allocateParams();
    layernorm_decoder.initializeWeights("xavier", 123u);
    const std::vector<float> layernorm_latent(
        static_cast<size_t>(layernorm_cfg.latent_w) * layernorm_cfg.latent_h *
            layernorm_cfg.latent_c,
        0.25f);
    const std::vector<float> layernorm_output =
        layernorm_decoder.forwardPass(layernorm_latent, false);
    TASSERT_TRUE(layernorm_output.size() ==
                 static_cast<size_t>(layernorm_cfg.image_w) * layernorm_cfg.image_h *
                     layernorm_cfg.image_c);
    for (float value : layernorm_output) TASSERT_TRUE(std::isfinite(value));

    bool invalid_upsample_rejected = false;
    try {
        VAEConvModel::Config invalid_cfg = rectangular_cfg;
        invalid_cfg.decoder_upsample = "unknown";
        VAEConvModel invalid_vae;
        invalid_vae.buildFromConfig(invalid_cfg);
    } catch (const std::runtime_error&) {
        invalid_upsample_rejected = true;
    }
    TASSERT_TRUE(invalid_upsample_rejected);

    // Decoder normalization can be disabled independently from the encoder.
    // This protects the CLI/config contract `enc_norm=groupnorm, dec_norm=none`.
    VAEConvModel::Config decoder_no_norm_cfg = rectangular_cfg;
    decoder_no_norm_cfg.resnet = true;
    decoder_no_norm_cfg.resnet_max_tokens = 0;
    decoder_no_norm_cfg.enc_norm = "groupnorm";
    decoder_no_norm_cfg.dec_norm = "none";
    VAEConvModel decoder_no_norm_vae;
    decoder_no_norm_vae.buildFromConfig(decoder_no_norm_cfg);
    bool saw_encoder_norm = false;
    bool saw_decoder_norm = false;
    for (const auto& layer : decoder_no_norm_vae.getLayers()) {
        const bool is_norm = layer.type == "GroupNorm" || layer.type == "LayerNorm";
        if (!is_norm) continue;
        saw_encoder_norm = saw_encoder_norm || layer.name.rfind("vae_conv/enc/", 0) == 0;
        saw_decoder_norm = saw_decoder_norm || layer.name.rfind("vae_conv/dec/", 0) == 0;
    }
    TASSERT_TRUE(saw_encoder_norm);
    TASSERT_TRUE(!saw_decoder_norm);

    // The log-variance head receives spatially indexed, activated and
    // normalized latent features before its final linear 1x1 projection.
    VAEConvModel::Config latent_stats_cfg = rectangular_cfg;
    latent_stats_cfg.enc_norm = "groupnorm";
    latent_stats_cfg.resnet = false;
    VAEConvModel latent_stats_vae;
    latent_stats_vae.buildFromConfig(latent_stats_cfg);
    const Layer* logvar_pre_conv =
        latent_stats_vae.getLayerByName("vae_conv/enc/logvar_pre_conv");
    const Layer* logvar_pre_act =
        latent_stats_vae.getLayerByName("vae_conv/enc/logvar_pre_conv/act");
    const Layer* logvar_pre_norm =
        latent_stats_vae.getLayerByName("vae_conv/enc/logvar_pre_norm/gn");
    const Layer* logvar_head = latent_stats_vae.getLayerByName("vae_conv/enc/logvar");
    TASSERT_TRUE(logvar_pre_conv != nullptr);
    TASSERT_TRUE(logvar_pre_conv->type == "Conv2d");
    TASSERT_TRUE(logvar_pre_conv->kernel_size == 3);
    TASSERT_TRUE(logvar_pre_conv->padding == 1);
    TASSERT_TRUE(logvar_pre_conv->input_height == latent_stats_cfg.latent_h);
    TASSERT_TRUE(logvar_pre_conv->input_width == latent_stats_cfg.latent_w);
    TASSERT_TRUE(logvar_pre_act != nullptr);
    TASSERT_TRUE(logvar_pre_act->type == "SiLU");
    TASSERT_TRUE(logvar_pre_norm != nullptr);
    TASSERT_TRUE(logvar_pre_norm->type == "GroupNorm");
    TASSERT_TRUE(logvar_head != nullptr);
    TASSERT_TRUE(logvar_head->inputs.size() == 1);
    TASSERT_TRUE(logvar_head->inputs[0] == "vae_conv/enc/logvar_pre_norm/gn_out");

    // A fixed Constant must remain fixed, while an explicitly trainable one
    // receives the exact upstream gradient and is updated by the optimizer.
    Model parameter_model;
    parameter_model.push("learned", "Constant", 3);
    Layer* learned = parameter_model.getLayerByName("learned");
    TASSERT_TRUE(learned != nullptr);
    learned->inputs = {};
    learned->output = "x";
    learned->trainable_parameter = true;

    parameter_model.allocateParams();
    float* learned_weights = learned->getWeights();
    TASSERT_TRUE(learned_weights != nullptr);
    std::fill(learned_weights, learned_weights + 3, 0.0f);

    const auto parameter_out = parameter_model.forwardPass(std::vector<float>{}, true);
    TASSERT_TRUE(parameter_out.size() == 3);
    parameter_model.backwardPass({1.0f, -2.0f, 0.5f});
    TASSERT_TRUE(learned->grad_weights.size() == 3);
    TASSERT_NEAR(learned->grad_weights[0], 1.0f, 1e-6f);
    TASSERT_NEAR(learned->grad_weights[1], -2.0f, 1e-6f);
    TASSERT_NEAR(learned->grad_weights[2], 0.5f, 1e-6f);

    Optimizer opt;
    opt.type = OptimizerType::SGD;
    opt.decay_strategy = LRDecayStrategy::NONE;
    parameter_model.optimizerStep(opt, 0.1f);
    TASSERT_NEAR(learned_weights[0], -0.1f, 1e-6f);
    TASSERT_NEAR(learned_weights[1], 0.2f, 1e-6f);
    TASSERT_NEAR(learned_weights[2], -0.05f, 1e-6f);

    // The packed VAE output must expose mu even when the decoder consumes a
    // stochastic, prior-biased z.
    VAEConvModel::Config cfg;
    cfg.image_w = 4;
    cfg.image_h = 4;
    cfg.image_c = 1;
    cfg.latent_w = 2;
    cfg.latent_h = 2;
    cfg.latent_c = 2;
    cfg.base_channels = 8;
    cfg.stochastic_latent = true;
    cfg.resnet = false;
    cfg.attention = false;
    cfg.enc_norm = "none";
    cfg.dec_norm = "none";
    cfg.use_encoder_prior = true;

    VAEConvModel vae;
    vae.buildFromConfig(cfg);
    Layer* prior = vae.getLayerByName("vae_conv/z_prior_bias");
    TASSERT_TRUE(prior != nullptr);
    TASSERT_TRUE(prior->trainable_parameter);

    vae.allocateParams();
    vae.initializeWeights("xavier", 123u);
    std::vector<float> input(16, 0.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) / 15.0f - 0.5f;
    }

    const std::vector<float> inference_output = vae.forwardPass(input, false);
    const std::vector<float> inference_recon = vae.getTensor("vae_conv/recon");
    TASSERT_TRUE(inference_output == inference_recon);
    TASSERT_TRUE(inference_output.size() == input.size());
    TASSERT_TRUE(vae.getTensor("x") == input);

    const std::vector<float> packed = vae.forwardPass(input, true);
    const std::vector<float> mu = vae.getTensor("vae_conv/mu");
    const std::vector<float> z = vae.getTensor("vae_conv/z");
    const std::vector<float> prior_values = vae.getTensor("vae_conv/prior_bias_out");
    const std::vector<float> z_biased = vae.getTensor("vae_conv/z_biased");
    TASSERT_TRUE(mu.size() == 8);
    TASSERT_TRUE(packed.size() == 16 + 2 * mu.size());
    for (size_t i = 0; i < mu.size(); ++i) {
        TASSERT_NEAR(packed[16 + i], mu[i], 1e-6f);
    }
    TASSERT_TRUE(z_biased.size() == mu.size());
    TASSERT_TRUE(z.size() == z_biased.size());
    TASSERT_TRUE(prior_values.size() == z_biased.size());
    for (size_t i = 0; i < z_biased.size(); ++i) {
        TASSERT_NEAR(z_biased[i], z[i] + prior_values[i], 1e-6f);
    }

    // The generic hook must dispatch through Model, honor an explicit target,
    // and accumulate gradients without advancing the optimizer.
    std::vector<float> target(input.size(), 0.25f);
    vae.modelConfig["recon_loss"] = "mse";
    Optimizer train_optimizer;
    train_optimizer.type = OptimizerType::SGD;
    train_optimizer.decay_strategy = LRDecayStrategy::NONE;
    Model::TrainStepRequest train_request;
    train_request.float_inputs["__input__"] = &input;
    train_request.target = &target;
    train_request.optimizer = &train_optimizer;
    train_request.mode = Model::TrainStepMode::Accumulate;
    train_request.grad_scale = 0.5f;

    Model& generic_vae = vae;
    const auto train_result = generic_vae.trainStep(train_request);
    TASSERT_TRUE(train_result.has_value());
    TASSERT_TRUE(train_result->metrics.count("mse") == 1);
    TASSERT_TRUE(train_result->metrics.count("kl") == 1);
    TASSERT_TRUE(train_optimizer.step == 0);
    TASSERT_TRUE(vae.hasTensor("vae_conv/recon"));

    const auto& trained_recon = vae.getTensor("vae_conv/recon");
    double expected_mse = 0.0;
    for (size_t i = 0; i < target.size(); ++i) {
        const double delta = static_cast<double>(trained_recon[i]) - target[i];
        expected_mse += delta * delta;
    }
    expected_mse /= static_cast<double>(target.size());
    TASSERT_NEAR(train_result->metrics.at("mse"), static_cast<float>(expected_mse), 1e-5f);

    // VIZ contract: a deliberately tiny historical limit must not hide graph
    // layers. Every node gets one canonical <model>/blocks/... label.
    vae.setVizTapsEnabled(true);
    vae.setVizTapsLimits(1, 32);
    (void)vae.forwardPass(input, false);
    const auto vae_taps = vae.consumeVizTaps();
    std::unordered_set<std::string> tapped_layers;
    for (const auto& frame : vae_taps) {
        const size_t bar = frame.label.find(" | ");
        const std::string base_label = frame.label.substr(0, bar);
        const size_t blocks = base_label.find("/blocks/");
        if (blocks == std::string::npos) continue; // custom recon/error frames
        size_t type_sep = base_label.find_last_of('/');
        if (base_label.compare(type_sep + 1, std::string::npos, "vec") == 0) {
            type_sep = base_label.find_last_of('/', type_sep - 1);
        }
        TASSERT_TRUE(type_sep != std::string::npos && type_sep > blocks + 8);
        std::string layer_name = base_label.substr(blocks + 8, type_sep - (blocks + 8));
        tapped_layers.insert(std::move(layer_name));
    }
    const std::string inference_tensor =
        vae.modelConfig.value("inference_output_tensor", std::string("x"));
    for (const auto& layer : vae.getLayers()) {
        TASSERT_TRUE(tapped_layers.find(layer.name) != tapped_layers.end());
        const std::string output_name = layer.output.empty() ? "x" : layer.output;
        if (output_name == inference_tensor) break;
    }

    // Training RGB previews must preserve geometry and channel order through
    // HWC->CHW, decoder Tanh and CHW->HWC. Compare the actual rendered bytes.
    {
        VAEConvModel::Config rgb_cfg;
        rgb_cfg.image_w = 8;
        rgb_cfg.image_h = 8;
        rgb_cfg.image_c = 3;
        rgb_cfg.latent_w = 4;
        rgb_cfg.latent_h = 4;
        rgb_cfg.latent_c = 4;
        rgb_cfg.base_channels = 8;
        rgb_cfg.resnet = false;
        rgb_cfg.attention = false;
        rgb_cfg.enc_norm = "none";
        rgb_cfg.dec_norm = "none";
        VAEConvModel rgb_model;
        rgb_model.buildFromConfig(rgb_cfg);
        rgb_model.allocateParams();
        rgb_model.initializeWeights("xavier", 123u);
        rgb_model.setVizTapsEnabled(true);
        rgb_model.setVizTapsLimits(100, 32);
        std::vector<float> rgb_input(8 * 8 * 3);
        for (size_t i = 0; i < rgb_input.size(); i += 3) {
            rgb_input[i] = 1.0f;
            rgb_input[i + 1] = 0.0f;
            rgb_input[i + 2] = -1.0f;
        }
        (void)rgb_model.forwardPass(rgb_input, true);
        const auto frames = rgb_model.consumeVizTaps();
        const auto find_frame = [&](const std::string& name) -> const Model::VizFrame* {
            for (const auto& frame : frames) {
                if (frame.label.find("/blocks/" + name + "/") != std::string::npos)
                    return &frame;
            }
            return nullptr;
        };
        const auto* raw = find_frame("vae_conv/raw_in");
        const auto* chw = find_frame("vae_conv/in_to_chw");
        const auto* tanh = find_frame("vae_conv/dec/tanh");
        const auto* recon = find_frame("vae_conv/recon_to_hwc");
        TASSERT_TRUE(raw && chw && tanh && recon);
        for (const auto* frame : {raw, chw, tanh, recon}) {
            TASSERT_TRUE(frame->tensor_sample_channels == 3);
            TASSERT_TRUE(frame->tensor_values.size() == static_cast<size_t>(frame->w)*frame->h*3);
            TASSERT_TRUE(frame->tensor_indices.size() == frame->tensor_values.size());
        }
        TASSERT_NEAR(raw->tensor_values[0],1.f,1e-6f);
        TASSERT_NEAR(raw->tensor_values[1],0.f,1e-6f);
        TASSERT_NEAR(raw->tensor_values[2],-1.f,1e-6f);
        TASSERT_TRUE(chw->tensor_indices[0]==0 && chw->tensor_indices[1]==64 && chw->tensor_indices[2]==128);
        const auto* feature = find_frame("vae_conv/enc/conv_in");
        TASSERT_TRUE(feature != nullptr);
        TASSERT_TRUE(feature->heatmap_kind == 1);
        TASSERT_TRUE(feature->pixels_real.size() == static_cast<size_t>(feature->w) * feature->h * 3);
        TASSERT_TRUE(feature->tensor_info.find("Capture: entrainement") != std::string::npos);
        TASSERT_TRUE(feature->tensor_info.find("non finies: 0") != std::string::npos);
        bool distinct_channels = false;
        for (size_t i = 0; i < feature->pixels_real.size(); i += 3)
            distinct_channels = distinct_channels || feature->pixels_real[i] != feature->pixels_real[i + 1] ||
                feature->pixels_real[i + 1] != feature->pixels_real[i + 2];
        TASSERT_TRUE(distinct_channels);
        for (const auto* frame : {raw, chw, tanh, recon}) {
            TASSERT_TRUE(frame->w == 8 && frame->h == 8 && frame->channels == 3);
            TASSERT_TRUE(frame->pixels.size() == 8 * 8 * 3);
            TASSERT_TRUE(frame->pixels_real.empty() || frame->pixels_real.size() == 8 * 8 * 3);
        }
        TASSERT_TRUE(raw->pixels == chw->pixels);
        TASSERT_TRUE(tanh->pixels == recon->pixels);
        TASSERT_TRUE(raw->pixels[0] > raw->pixels[1] && raw->pixels[1] > raw->pixels[2]);
        TASSERT_TRUE(rgb_model.getTensor("vae_conv/in_hwc") == rgb_input);
        TASSERT_TRUE(rgb_model.getTensor("vae_conv/recon").size() == rgb_input.size());
    }

    // Rectangular feature previews retain their RGB alternative after resizing.
    {
        Model features;
        features.modelConfig["image_w"] = 4;
        features.modelConfig["image_h"] = 2;
        features.push("features", "Identity", 0);
        features.getLayerByName("features")->inputs = {"__input__"};
        features.getLayerByName("features")->output = "features";
        features.allocateParams();
        features.setVizTapsEnabled(true);
        features.setVizTapsLimits(10, 32);
        std::vector<float> values(32);
        for (size_t i = 0; i < values.size(); i += 4) {
            values[i] = -1; values[i + 1] = 0; values[i + 2] = 1; values[i + 3] = 0;
        }
        (void)features.forwardPass(values, true);
        const auto frames = features.consumeVizTaps();
        TASSERT_TRUE(frames.size() == 1);
        TASSERT_TRUE(frames[0].w == 4 && frames[0].h == 4);
        TASSERT_TRUE(frames[0].pixels_real.size() == 48);
        TASSERT_TRUE(frames[0].tensor_sample_channels == 4);
        TASSERT_TRUE(frames[0].tensor_values.size() == 64);
        for (size_t i=0; i<frames[0].tensor_values.size(); ++i) {
            TASSERT_TRUE(frames[0].tensor_indices[i] < values.size());
            TASSERT_NEAR(frames[0].tensor_values[i],values[frames[0].tensor_indices[i]],1e-6f);
        }
        TASSERT_TRUE(frames[0].tensor_indices[0]==0 && frames[0].tensor_indices[16]==0);
        TASSERT_TRUE(frames[0].tensor_indices[32]==16);
        TASSERT_TRUE(frames[0].pixels_real[0] < frames[0].pixels_real[1]);
        TASSERT_TRUE(frames[0].pixels_real[1] < frames[0].pixels_real[2]);
    }

    // Composition contract: a child model executed before the root taps are
    // consumed automatically contributes its own canonical thumbnails.
    Model root;
    root.setModelName("root_model");
    root.modelConfig["type"] = "root_model";
    root.push("root/layer", "Identity", 0);
    root.getLayerByName("root/layer")->inputs = {"__input__"};
    root.getLayerByName("root/layer")->output = "x";
    root.allocateParams();
    root.setVizTapsEnabled(true);
    root.setVizTapsLimits(1, 16);
    (void)root.forwardPass(std::vector<float>{1.0f, 2.0f, 3.0f}, false);

    Model child;
    child.setModelName("child_model");
    child.modelConfig["type"] = "child_model";
    child.push("child/layer", "Identity", 0);
    child.getLayerByName("child/layer")->inputs = {"__input__"};
    child.getLayerByName("child/layer")->output = "x";
    child.allocateParams();
    (void)child.forwardPass(std::vector<float>{4.0f, 5.0f, 6.0f}, false);

    const auto composed_taps = root.consumeVizTaps();
    bool saw_root = false;
    bool saw_child = false;
    for (const auto& frame : composed_taps) {
        saw_root = saw_root || frame.label.find("root_model/blocks/root/layer/") == 0;
        saw_child = saw_child || frame.label.find("child_model/blocks/child/layer/") == 0;
    }
    TASSERT_TRUE(saw_root);
    TASSERT_TRUE(saw_child);

    // A reconstruction-only loss must train the prior through the decoder
    // feature path (prior -> Add -> decoder convolutions -> reconstruction).
    std::vector<float> reconstruction_grad(packed.size(), 0.0f);
    std::fill(reconstruction_grad.begin(), reconstruction_grad.begin() + 16, 1.0f / 16.0f);
    vae.backwardPass(reconstruction_grad);
    TASSERT_TRUE(prior->grad_weights.size() == mu.size());
    float prior_grad_l1 = 0.0f;
    for (float g : prior->grad_weights) {
        prior_grad_l1 += std::fabs(g);
    }
    TASSERT_TRUE(prior_grad_l1 > 1e-8f);

    return 0;
}
