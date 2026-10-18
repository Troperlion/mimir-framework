#include "CausalLMModel.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
size_t checkedMul(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::overflow_error("CausalLMModel: parameter count overflow");
    }
    return a * b;
}

void configureLinear(Model& model, const std::string& name, const std::string& input,
                     const std::string& output, int in_features, int out_features,
                     int seq_len) {
    model.push(name, "Linear", checkedMul(static_cast<size_t>(in_features),
                                          static_cast<size_t>(out_features)));
    auto* layer = model.getLayerByName(name);
    layer->inputs = {input};
    layer->output = output;
    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->seq_len = seq_len;
    layer->use_bias = false;
}
} // namespace

CausalLMModel::CausalLMModel() {
    setModelName("CausalLMModel");
    setHasEncoder(false);
}

void CausalLMModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void CausalLMModel::buildInto(Model& model, const Config& cfg) {
    if (cfg.vocab_size < 2 || cfg.seq_len < 2 || cfg.d_model < 2 ||
        cfg.num_layers < 1 || cfg.num_heads < 1 || cfg.mlp_hidden < 1) {
        throw std::invalid_argument("CausalLMModel: invalid non-positive dimensions");
    }
    if (cfg.d_model % cfg.num_heads != 0 || cfg.num_heads % cfg.num_kv_heads != 0) {
        throw std::invalid_argument("CausalLMModel: d_model must be divisible by num_heads");
    }

    model.getMutableLayers().clear();
    model.setModelName("CausalLMModel");
    model.setHasEncoder(false);
    model.modelConfig = {
        {"type", "causal_lm"},
        {"task", "causal_language_modeling"},
        {"input_dim", cfg.seq_len},
        {"output_dim", cfg.seq_len * cfg.vocab_size},
        {"vocab_size", cfg.vocab_size},
        {"seq_len", cfg.seq_len},
        {"d_model", cfg.d_model},
        {"num_layers", cfg.num_layers},
        {"num_heads", cfg.num_heads},
        {"num_kv_heads", cfg.num_kv_heads},
        {"mlp_hidden", cfg.mlp_hidden},
        {"padding_idx", cfg.padding_idx},
        {"norm_eps", cfg.norm_eps},
        {"rope_theta", cfg.rope_theta},
        {"causal", true},
    };

    model.push("causal_lm/token_embedding", "Embedding",
               checkedMul(cfg.vocab_size, cfg.d_model));
    if (auto* layer = model.getLayerByName("causal_lm/token_embedding")) {
        layer->inputs = {"__input__"};
        layer->output = "causal_lm/token_vectors";
        layer->vocab_size = cfg.vocab_size;
        layer->embed_dim = cfg.d_model;
        layer->padding_idx = cfg.padding_idx;
        layer->seq_len = cfg.seq_len;
    }

    std::string x = "causal_lm/token_vectors";
    for (int index = 0; index < cfg.num_layers; ++index) {
        const std::string prefix = "causal_lm/block" + std::to_string(index + 1);

        model.push(prefix + "/attn_norm", "RMSNorm", cfg.d_model);
        if (auto* layer = model.getLayerByName(prefix + "/attn_norm")) {
            layer->inputs = {x};
            layer->output = prefix + "/attn_norm_out";
            layer->in_features = cfg.d_model;
            layer->eps = cfg.norm_eps;
            layer->affine = true;
            layer->use_bias = false;
        }

        const int head_dim = cfg.d_model / cfg.num_heads;
        const int kv_dim = cfg.num_kv_heads * head_dim;
        const size_t attention_params = checkedMul(
            cfg.d_model, static_cast<size_t>(2 * cfg.d_model + 2 * kv_dim));
        model.push(prefix + "/attention", "MultiHeadAttention", attention_params);
        if (auto* layer = model.getLayerByName(prefix + "/attention")) {
            layer->inputs = {prefix + "/attn_norm_out"};
            layer->output = prefix + "/attention_out";
            layer->seq_len = cfg.seq_len;
            layer->embed_dim = cfg.d_model;
            layer->num_heads = cfg.num_heads;
            layer->num_kv_heads = cfg.num_kv_heads;
            layer->causal = true;
            layer->rope_theta = cfg.rope_theta;
            layer->use_bias = false;
        }

        model.push(prefix + "/attn_residual", "Add", 0);
        if (auto* layer = model.getLayerByName(prefix + "/attn_residual")) {
            layer->inputs = {x, prefix + "/attention_out"};
            layer->output = prefix + "/post_attention";
        }

        model.push(prefix + "/ffn_norm", "RMSNorm", cfg.d_model);
        if (auto* layer = model.getLayerByName(prefix + "/ffn_norm")) {
            layer->inputs = {prefix + "/post_attention"};
            layer->output = prefix + "/ffn_norm_out";
            layer->in_features = cfg.d_model;
            layer->eps = cfg.norm_eps;
            layer->affine = true;
            layer->use_bias = false;
        }

        configureLinear(model, prefix + "/gate", prefix + "/ffn_norm_out",
                        prefix + "/gate_pre", cfg.d_model, cfg.mlp_hidden, cfg.seq_len);
        model.push(prefix + "/gate_silu", "SiLU", 0);
        if (auto* layer = model.getLayerByName(prefix + "/gate_silu")) {
            layer->inputs = {prefix + "/gate_pre"};
            layer->output = prefix + "/gate_out";
        }
        configureLinear(model, prefix + "/up", prefix + "/ffn_norm_out",
                        prefix + "/up_out", cfg.d_model, cfg.mlp_hidden, cfg.seq_len);
        model.push(prefix + "/swiglu", "Multiply", 0);
        if (auto* layer = model.getLayerByName(prefix + "/swiglu")) {
            layer->inputs = {prefix + "/gate_out", prefix + "/up_out"};
            layer->output = prefix + "/swiglu_out";
        }
        configureLinear(model, prefix + "/down", prefix + "/swiglu_out",
                        prefix + "/ffn_out", cfg.mlp_hidden, cfg.d_model, cfg.seq_len);

        model.push(prefix + "/ffn_residual", "Add", 0);
        if (auto* layer = model.getLayerByName(prefix + "/ffn_residual")) {
            layer->inputs = {prefix + "/post_attention", prefix + "/ffn_out"};
            layer->output = prefix + "/output";
        }
        x = prefix + "/output";
    }

    model.push("causal_lm/final_norm", "RMSNorm", cfg.d_model);
    if (auto* layer = model.getLayerByName("causal_lm/final_norm")) {
        layer->inputs = {x};
        layer->output = "causal_lm/final_hidden";
        layer->in_features = cfg.d_model;
        layer->eps = cfg.norm_eps;
        layer->affine = true;
        layer->use_bias = false;
    }
    model.push("causal_lm/lm_head", "Linear", 0);
    if (auto* layer = model.getLayerByName("causal_lm/lm_head")) {
        layer->inputs = {"causal_lm/final_hidden"};
        layer->output = "x";
        layer->in_features = cfg.d_model;
        layer->out_features = cfg.vocab_size;
        layer->seq_len = cfg.seq_len;
        layer->use_bias = false;
        layer->shared_weights_from = "causal_lm/token_embedding";
    }
}
