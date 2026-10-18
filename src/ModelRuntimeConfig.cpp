#include "Model.hpp"
#include "viz/ModelGraph.hpp"
#include <limits>
#include "Serialization/DebugJsonDump.hpp"

void Model::publishRuntimeConfiguration(Optimizer* optimizer) {
    if (!runtime_config_->modelGraph())
        runtime_config_->publishModelGraph(Mimir::Viz::modelGraphSnapshot(layers,nullptr,"en attente",false,false,false));
    using Owner = LiveModelConfig::Owner;
    using Entry = LiveModelConfig::Entry;
    if (!modelConfig.is_object()) modelConfig = json::object();
    std::map<std::string, Entry> fields;
    std::function<void(const json&, const std::string&)> inspect = [&](const json& object, const std::string& prefix) {
        for (const auto& item : object.items()) {
            const std::string key = prefix + item.key();
            if (item.value().is_object() && !item.value().empty()) inspect(item.value(), key + "/");
            else {
                Entry field;
                field.key = key; field.value = item.value();
                fields[key] = std::move(field);
            }
        }
    };
    inspect(modelConfig, "");
    auto live = [&](const std::string& key, json value, Owner owner, double lo, double hi,
                    const std::string& note, std::vector<std::string> choices = {}) {
        fields[key] = {key, std::move(value), owner, lo, hi, std::move(choices), note, {}};
    };
    auto number = [&](const std::string& key, double fallback, double lo, double hi) {
        json value = modelConfig.contains(key) ? modelConfig[key] : json(fallback);
        if (value.is_number()) value = value.get<double>();
        live(key, value, Owner::Model, lo, hi,
             "Direct; nombre dans [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    };
    auto boolean = [&](const std::string& key, bool value) {
        live(key, value, Owner::Model, 0, 1, "Direct; true / false");
    };
    number("grad_clip_norm", modelConfig.value("clip_norm", 0.0), 0, 1e6);
    const auto skips = skipConnectionControl();
    if (skips->available.load()) boolean("skip_connections_enabled", skips->enabled.load());
    bool has_reparameterize = false;
    for (const auto& layer : layers) has_reparameterize |= layer.type_enum == LayerType::Reparameterize;
    const Layer* dropout = nullptr;
    const Layer* nms = nullptr;
    for (const auto& layer : layers) {
        if (layer.type_enum == LayerType::Dropout || layer.type_enum == LayerType::Dropout2d ||
            layer.type_enum == LayerType::AlphaDropout) dropout = &layer;
        if (layer.type_enum == LayerType::NMS) nms = &layer;
    }
    if (dropout) number("dropout", dropout->dropout_p, 0, 0.999999);
    if (nms) {
        number("nms_iou_threshold", nms->nms_iou_threshold, 0, 1);
        number("nms_score_threshold", nms->nms_score_threshold, 0, 1);
        live("nms_max_detections", nms->nms_max_detections, Owner::Model, 0, 1000000000, "Direct; entier >= 0");
        boolean("nms_class_agnostic", nms->nms_class_agnostic);
    }
    if (has_reparameterize) boolean("stochastic_latent", modelConfig.value("stochastic_latent", true));
    if (modelConfig.contains("use_kv_cache")) boolean("use_kv_cache", modelConfig["use_kv_cache"].get<bool>());
    const std::string type = modelConfig.value("type", "");
    if (modelConfig.contains("image_w") || modelConfig.contains("image_dim") ||
        modelConfig.contains("bbox_loss_weight")) {
        number("bbox_loss_weight", 1.0, 0, 1e6);
        number("relation_primary_weight", 3.0, 0, 1e6);
        number("relation_secondary_weight", 2.0, 0, 1e6);
        number("relation_other_weight", 1.0, 0, 1e6);
        fields["bbox_loss_weight"].note = "Direct; poids bbox >= 0 (0 desactive la supervision bbox)";
    }
    if (type == "vae" || type == "vae_conv" || type == "vae_text") {
        number("kl_beta", modelConfig.value("vae_kl_beta", 1.0), 0, 1e6);
        live("kl_warmup_steps", modelConfig.value("kl_warmup_steps", 0), Owner::Model,
             0, 1000000000, "Direct; entier >= 0");
        live("recon_loss", modelConfig.value("recon_loss", "mse"), Owner::Model, 0, 0,
             "mse, mae, huber, charbonnier, gaussian_nll, bce",
             {"mse", "mae", "l1", "huber", "charbonnier", "gaussian_nll", "bce"});
        if (type != "vae_text") {
            for (const auto* key : {"ssim_weight", "spectral_weight", "marker_wass_scale", "marker_temp_scale"})
                number(key, 0.0, 0, 1e6);
            number("huber_delta", modelConfig.value("smoothl1_beta", modelConfig.value("smoothl1_delta", 1.0)), 1e-6, 1e6);
            number("charbonnier_eps", 1e-3, 1e-12, 1e6);
            number("gaussian_nll_sigma", modelConfig.value("nll_sigma", 1.0), 1e-6, 1e6);
        }
    }
    if (optimizer) {
        for (const auto& pair : std::vector<std::pair<std::string, float>>{
            {"beta1", optimizer->beta1}, {"beta2", optimizer->beta2},
            {"epsilon", optimizer->eps}, {"weight_decay", optimizer->weight_decay},
            {"rmsprop_alpha", optimizer->rmsprop_alpha}}) {
            const bool bounded = pair.first == "beta1" || pair.first == "beta2" || pair.first == "rmsprop_alpha";
            live(pair.first, pair.second, Owner::Optimizer, pair.first == "epsilon" ? 1e-12 : 0,
                 bounded ? 0.999999 : 1e6,
                 bounded ? "Direct; 0 <= valeur < 1" : "Direct; nombre positif (epsilon > 0)");
        }
    }
    std::vector<Entry> entries;
    for (auto& item : fields) entries.push_back(std::move(item.second));
    runtime_config_->publish(std::move(entries));
}

void Model::applyRuntimeConfiguration() {
    std::string debug_path;
    if (runtime_config_->consumeDebugJson(&debug_path)) {
        try {
            Mimir::Serialization::DebugJsonOptions options;
            options.include_gradients = true;
            options.include_optimizer_state = true;
            options.include_activations = true;
            options.include_checksums = true;
            Mimir::Serialization::DebugJsonDump dumper;
            std::string error;
            const bool ok = dumper.save_enhanced(debug_path, *this, options, &error);
            runtime_config_->finishDebugJson(ok ? "DebugJSON : " + debug_path : "DebugJSON erreur : " + error);
        } catch (const std::exception& e) {
            runtime_config_->finishDebugJson(std::string("DebugJSON erreur : ") + e.what());
        }
    }
    runtime_config_->apply(LiveModelConfig::Owner::Model, [&](const std::string& key, const json& value) {
        modelConfig[key] = value;
        if (key == "skip_connections_enabled") skipConnectionControl()->enabled = value.get<bool>();
        if (key == "huber_delta") {
            // These legacy aliases otherwise take precedence in VAETraining.
            if (modelConfig.contains("smoothl1_delta")) modelConfig["smoothl1_delta"] = value;
            if (modelConfig.contains("smoothl1_beta")) modelConfig["smoothl1_beta"] = value;
        }
        for (auto& layer : layers) {
            if (key == "dropout" && (layer.type_enum == LayerType::Dropout ||
                layer.type_enum == LayerType::Dropout2d || layer.type_enum == LayerType::AlphaDropout))
                layer.dropout_p = value.get<float>();
            if (layer.type_enum == LayerType::NMS) {
                if (key == "nms_iou_threshold") layer.nms_iou_threshold = value.get<float>();
                if (key == "nms_score_threshold") layer.nms_score_threshold = value.get<float>();
                if (key == "nms_max_detections") layer.nms_max_detections = value.get<int>();
                if (key == "nms_class_agnostic") layer.nms_class_agnostic = value.get<bool>();
            }
        }
    });
}

void Model::applyRuntimeOptimizerConfiguration(Optimizer& optimizer) {
    runtime_config_->apply(LiveModelConfig::Owner::Optimizer, [&](const std::string& key, const json& value) {
        modelConfig[key] = value;
        if (key == "epsilon" && modelConfig.contains("eps")) modelConfig["eps"] = value;
    });
    // Only explicit UI edits override caller defaults; preserve optimizer state.
    const json overrides = runtime_config_->overrides();
    if (overrides.contains("beta1")) optimizer.beta1 = overrides["beta1"].get<float>();
    if (overrides.contains("beta2")) optimizer.beta2 = overrides["beta2"].get<float>();
    if (overrides.contains("epsilon")) optimizer.eps = overrides["epsilon"].get<float>();
    if (overrides.contains("weight_decay")) optimizer.weight_decay = overrides["weight_decay"].get<float>();
    if (overrides.contains("rmsprop_alpha")) optimizer.rmsprop_alpha = overrides["rmsprop_alpha"].get<float>();
}
