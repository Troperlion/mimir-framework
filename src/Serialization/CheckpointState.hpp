#pragma once

#include "Serialization.hpp"
#include "../Model.hpp"

namespace Mimir::Serialization {

// Explicitly disabled assets take precedence over compatibility defaults.
inline bool component_enabled(const Model& model, const char* name, bool active) {
    const auto& cfg = model.modelConfig;
    for (const std::string& key : {std::string(name), std::string("has_") + name}) {
        if (cfg.contains(key)) {
            const auto& value = cfg[key];
            if (value.is_boolean() && !value.get<bool>()) return false;
            if (value.is_object() && value.contains("enabled") && value["enabled"] == false) return false;
        }
    }
    return active;
}

inline SaveOptions effective_save_options(const Model& model, SaveOptions options) {
    options.save_encoder = options.save_encoder && component_enabled(model, "encoder", model.getHasEncoder());
    options.save_tokenizer = options.save_tokenizer && component_enabled(model, "tokenizer", model.getHasTokenizer());
    options.save_optimizer = options.save_optimizer && model.getSerializedOptimizer();
    options.include_optimizer_state = (options.include_optimizer_state || options.save_optimizer) && model.getSerializedOptimizer();
    return options;
}

inline json checkpoint_components(const SaveOptions& options) {
    return {{"encoder", options.save_encoder}, {"tokenizer", options.save_tokenizer},
            {"optimizer", options.save_optimizer}};
}

inline json optimizer_metadata(const Optimizer& opt) {
    json j = {{"schema_version", 2}, {"has_optimizer", true},
        {"type", static_cast<int>(opt.type)}, {"name", optimizerTypeName(opt.type)},
        {"step", opt.step}, {"lr_current", opt.getCurrentLR()},
        {"decay_strategy", static_cast<int>(opt.decay_strategy)},
        {"initial_lr", opt.initial_lr}, {"min_lr", opt.min_lr},
        {"decay_rate", opt.decay_rate}, {"decay_steps", opt.decay_steps},
        {"total_steps", opt.total_steps}, {"warmup_steps", opt.warmup_steps}};
    if (opt.type != OptimizerType::SGD && opt.type != OptimizerType::LION) j["eps"] = opt.eps;
    if (opt.usesFirstMoment()) j["beta1"] = opt.beta1;
    if (opt.type != OptimizerType::SGD && opt.type != OptimizerType::RMSPROP && opt.type != OptimizerType::ADAFACTOR) j["beta2"] = opt.beta2;
    if (opt.type != OptimizerType::SGD && opt.type != OptimizerType::ADAM) j["weight_decay"] = opt.weight_decay;
    if (opt.type == OptimizerType::RMSPROP) j["rmsprop_alpha"] = opt.rmsprop_alpha;
    if (opt.type == OptimizerType::ADAFACTOR) {
        j["adafactor_clip_threshold"] = opt.adafactor_clip_threshold;
        j["adafactor_decay_rate"] = opt.adafactor_decay_rate;
        j["adafactor_eps2"] = opt.adafactor_eps2;
        j["adafactor_beta1"] = opt.adafactor_beta1;
        j["adafactor_scale_parameter"] = opt.adafactor_scale_parameter;
        j["adafactor_relative_step"] = opt.adafactor_relative_step;
    }
    j["state_sizes"] = json::object();
    if (!opt.m.empty()) j["state_sizes"]["m"] = opt.m.size();
    if (!opt.v.empty()) j["state_sizes"]["v"] = opt.v.size();
    j["parameter_layout"] = json::array();
    for (const auto& block : opt.parameter_layout) {
        j["parameter_layout"].push_back({{"name", block.name}, {"offset", block.offset}, {"size", block.size}});
    }
    return j;
}

inline Optimizer optimizer_from_metadata(const json& j) {
    Optimizer opt;
    const int type = j.value("type", static_cast<int>(opt.type));
    if (type < static_cast<int>(OptimizerType::SGD) || type > static_cast<int>(OptimizerType::LAMB)) {
        throw std::runtime_error("Invalid optimizer type");
    }
    opt.type = static_cast<OptimizerType>(type);
    opt.step = j.value("step", size_t{0});
#define MIMIR_OPT_FIELD(field) opt.field = j.value(#field, opt.field)
    MIMIR_OPT_FIELD(beta1); MIMIR_OPT_FIELD(beta2); MIMIR_OPT_FIELD(eps);
    MIMIR_OPT_FIELD(weight_decay); MIMIR_OPT_FIELD(rmsprop_alpha);
    MIMIR_OPT_FIELD(adafactor_clip_threshold); MIMIR_OPT_FIELD(adafactor_decay_rate);
    MIMIR_OPT_FIELD(adafactor_eps2); MIMIR_OPT_FIELD(adafactor_beta1);
    MIMIR_OPT_FIELD(adafactor_scale_parameter); MIMIR_OPT_FIELD(adafactor_relative_step);
    MIMIR_OPT_FIELD(initial_lr); MIMIR_OPT_FIELD(min_lr); MIMIR_OPT_FIELD(decay_rate);
    MIMIR_OPT_FIELD(decay_steps); MIMIR_OPT_FIELD(total_steps); MIMIR_OPT_FIELD(warmup_steps);
#undef MIMIR_OPT_FIELD
    const int decay = j.value("decay_strategy", static_cast<int>(opt.decay_strategy));
    if (decay < 0 || decay > static_cast<int>(LRDecayStrategy::LINEAR)) throw std::runtime_error("Invalid LR schedule");
    opt.decay_strategy = static_cast<LRDecayStrategy>(decay);
    if (j.contains("parameter_layout")) {
        for (const auto& block : j["parameter_layout"]) {
            opt.parameter_layout.push_back({block.at("name").get<std::string>(),
                block.at("offset").get<size_t>(), block.at("size").get<size_t>()});
        }
    }
    return opt;
}

inline void validate_optimizer_state(const Model& model, const Optimizer& opt, const json& metadata) {
    if (metadata.value("schema_version", 0) < 2) return; // Legacy metadata remains readable.
    const auto sizes = metadata.value("state_sizes", json::object());
    if (opt.m.size() != sizes.value("m", size_t{0}) || opt.v.size() != sizes.value("v", size_t{0})) {
        throw std::runtime_error("Missing or truncated optimizer state tensors");
    }
    Optimizer check = opt;
    model.restoreOptimizerState(check); // Validate names and shapes before declaring a successful load.
}

} // namespace Mimir::Serialization
