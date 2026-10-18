#include "ModelArchitectures.hpp"

#include "Models/MLP/BasicMLPModel.hpp"
#include "Models/NLP/HFCLIPTextEncoder1Model.hpp"
#include "Models/NLP/HFCLIPTextEncoder2Model.hpp"
#include "Models/NLP/TransformerModel.hpp"
#include "Models/NLP/CausalLMModel.hpp"
#include "Models/NLP/VAETextModel.hpp"
#include "Models/External/ExternalSafeTensorsModel.hpp"
#include "Models/Vision/HFVaeDecoderModel.hpp"
#include "Models/Vision/ViTModel.hpp"
#include "Models/Vision/VAEModel.hpp"
#include "Models/Vision/VAEConvModel.hpp"
#include "Models/Vision/GanLatentModel.hpp"
#include "Models/Vision/GanStyleModel.hpp"
#include "Models/Vision/CNN/ResNetModel.hpp"
#include "Models/Vision/UNetModel.hpp"
#include "Models/Vision/CNN/MobileNetModel.hpp"
#include "Models/Vision/CNN/VGG16Model.hpp"
#include "Models/Vision/CNN/VGG19Model.hpp"
#include "Models/Vision/CNN/VGG16FeatModel.hpp"
#include "Models/Vision/CNN/MedicalDetectorModel.hpp"
#include "Models/Diffusion/DiffusionModel.hpp"
#include "Models/Diffusion/CondDiffusionModel.hpp"
#include "Models/Diffusion/HFSDXLTransformerBlockModel.hpp"
#include "Models/Diffusion/LumenLatentDiffusionModel.hpp"
#include "Models/Diffusion/SD35Model.hpp"
#include "Models/Vision/PatchDiscriminatorModel.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace ModelArchitectures {

namespace {
static inline std::string canonicalArchName(const std::string& name) {
    // Alias convivial: "SD3.5" -> "sd3_5".
    if (name == "sd3.5" || name == "SD3.5" || name == "sd3_5" || name == "SD3_5") return "sd3_5";
    return name;
}
} // namespace

// Merge helper for cfgFromConfig (keep behavior consistent with Registry::create).
static void mergeIntoCfg(json& base, const json& overrides) {
    if (!base.is_object() || !overrides.is_object()) return;
    for (auto it = overrides.begin(); it != overrides.end(); ++it) {
        const std::string& key = it.key();
        const json& val = it.value();
        if (base.contains(key) && base[key].is_object() && val.is_object()) {
            mergeIntoCfg(base[key], val);
        } else {
            base[key] = val;
        }
    }
}

Registry& Registry::instance() {
    static Registry inst;
    return inst;
}

void Registry::registerArchitecture(Entry entry) {
    if (entry.name.empty()) {
        throw std::runtime_error("ModelArchitectures::Registry: entry.name is empty");
    }
    if (!entry.create) {
        throw std::runtime_error("ModelArchitectures::Registry: entry.create is null for: " + entry.name);
    }
    ensureBuiltinsRegistered();
    entries_[entry.name] = std::move(entry);
}

const Entry* Registry::find(const std::string& name) const {
    ensureBuiltinsRegistered();
    const std::string key = canonicalArchName(name);
    auto it = entries_.find(key);
    if (it == entries_.end()) return nullptr;
    return &it->second;
}

std::vector<std::string> Registry::list() const {
    ensureBuiltinsRegistered();
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto& kv : entries_) out.push_back(kv.first);
    std::sort(out.begin(), out.end());
    return out;
}

json Registry::defaultConfig(const std::string& name) const {
    ensureBuiltinsRegistered();
    const std::string key = canonicalArchName(name);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        throw std::runtime_error("ModelArchitectures::defaultConfig: unknown architecture: " + name);
    }
    return it->second.default_config;
}

namespace {

template <typename T>
static T jget(const json& j, const char* key, T def) {
    if (!j.is_object()) return def;
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) return def;
    try {
        return it->get<T>();
    } catch (...) {
        return def;
    }
}

static void mergeInto(json& base, const json& overrides) {
    if (!base.is_object() || !overrides.is_object()) return;
    for (auto it = overrides.begin(); it != overrides.end(); ++it) {
        const std::string& key = it.key();
        const json& val = it.value();
        if (base.contains(key) && base[key].is_object() && val.is_object()) {
            mergeInto(base[key], val);
        } else {
            base[key] = val;
        }
    }
}

static BasicMLPModel::Config basicMlpCfgFromJson(const json& cfg) {
    BasicMLPModel::Config out;
    out.input_dim = jget<int>(cfg, "input_dim", out.input_dim);
    out.hidden_dim = jget<int>(cfg, "hidden_dim", out.hidden_dim);
    out.output_dim = jget<int>(cfg, "output_dim", out.output_dim);
    out.hidden_layers = jget<int>(cfg, "hidden_layers", out.hidden_layers);
    out.dropout = jget<float>(cfg, "dropout", out.dropout);
    return out;
}

static json basicMlpDefaultConfigJson() {
    BasicMLPModel::Config d;
    return json{
        {"input_dim", d.input_dim},
        {"hidden_dim", d.hidden_dim},
        {"output_dim", d.output_dim},
        {"hidden_layers", d.hidden_layers},
        {"dropout", d.dropout},
    };
}

static TransformerModel::Config transformerCfgFromJson(const json& cfg) {
    TransformerModel::Config out;
    out.seq_len = jget<int>(cfg, "seq_len", out.seq_len);
    out.d_model = jget<int>(cfg, "d_model", out.d_model);
    out.vocab_size = jget<int>(cfg, "vocab_size", out.vocab_size);
    out.padding_idx = jget<int>(cfg, "padding_idx", out.padding_idx);
    out.num_layers = jget<int>(cfg, "num_layers", out.num_layers);
    out.num_heads = jget<int>(cfg, "num_heads", out.num_heads);
    out.mlp_hidden = jget<int>(cfg, "mlp_hidden", out.mlp_hidden);
    out.output_dim = jget<int>(cfg, "output_dim", out.output_dim);
    out.causal = jget<bool>(cfg, "causal", out.causal);
    return out;
}

static HFCLIPTextEncoder2Model::Config hfClipTextEncoder2CfgFromJson(const json& cfg) {
    HFCLIPTextEncoder2Model::Config out;
    out.vocab_size = jget<int>(cfg, "vocab_size", out.vocab_size);
    out.padding_idx = jget<int>(cfg, "padding_idx", out.padding_idx);
    out.seq_len = jget<int>(cfg, "seq_len", out.seq_len);
    out.d_model = jget<int>(cfg, "d_model", out.d_model);
    out.num_layers = jget<int>(cfg, "num_layers", out.num_layers);
    out.num_heads = jget<int>(cfg, "num_heads", out.num_heads);
    out.mlp_hidden = jget<int>(cfg, "mlp_hidden", out.mlp_hidden);
    out.proj_dim = jget<int>(cfg, "proj_dim", out.proj_dim);
    out.causal = jget<bool>(cfg, "causal", out.causal);
    out.include_logit_scale = jget<bool>(cfg, "include_logit_scale", out.include_logit_scale);
    return out;
}

static HFCLIPTextEncoder1Model::Config hfClipTextEncoder1CfgFromJson(const json& cfg) {
    HFCLIPTextEncoder1Model::Config out;
    out.vocab_size = jget<int>(cfg, "vocab_size", out.vocab_size);
    out.padding_idx = jget<int>(cfg, "padding_idx", out.padding_idx);
    out.seq_len = jget<int>(cfg, "seq_len", out.seq_len);
    out.d_model = jget<int>(cfg, "d_model", out.d_model);
    out.num_layers = jget<int>(cfg, "num_layers", out.num_layers);
    out.num_heads = jget<int>(cfg, "num_heads", out.num_heads);
    out.mlp_hidden = jget<int>(cfg, "mlp_hidden", out.mlp_hidden);
    out.causal = jget<bool>(cfg, "causal", out.causal);
    return out;
}

static json hfClipTextEncoder1DefaultConfigJson() {
    HFCLIPTextEncoder1Model::Config d;
    return json{{"vocab_size", d.vocab_size},
                {"padding_idx", d.padding_idx},
                {"seq_len", d.seq_len},
                {"d_model", d.d_model},
                {"num_layers", d.num_layers},
                {"num_heads", d.num_heads},
                {"mlp_hidden", d.mlp_hidden},
                {"causal", d.causal}};
}

static HFVaeDecoderModel::Config hfVaeDecoderCfgFromJson(const json& cfg) {
    HFVaeDecoderModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.latent_w = jget<int>(cfg, "latent_w", out.latent_w);
    out.latent_h = jget<int>(cfg, "latent_h", out.latent_h);
    out.latent_c = jget<int>(cfg, "latent_c", out.latent_c);
    out.num_heads = jget<int>(cfg, "num_heads", out.num_heads);
    out.norm_groups = jget<int>(cfg, "norm_groups", out.norm_groups);
    return out;
}

static json hfVaeDecoderDefaultConfigJson() {
    HFVaeDecoderModel::Config d;
    return json{{"image_w", d.image_w},
                {"image_h", d.image_h},
                {"image_c", d.image_c},
                {"latent_w", d.latent_w},
                {"latent_h", d.latent_h},
                {"latent_c", d.latent_c},
                {"num_heads", d.num_heads},
                {"norm_groups", d.norm_groups}};
}

static json hfClipTextEncoder2DefaultConfigJson() {
    HFCLIPTextEncoder2Model::Config d;
    return json{{"vocab_size", d.vocab_size},
                {"padding_idx", d.padding_idx},
                {"seq_len", d.seq_len},
                {"d_model", d.d_model},
                {"num_layers", d.num_layers},
                {"num_heads", d.num_heads},
                {"mlp_hidden", d.mlp_hidden},
                {"proj_dim", d.proj_dim},
                {"causal", d.causal},
                {"include_logit_scale", d.include_logit_scale}};
}

static json transformerDefaultConfigJson() {
    TransformerModel::Config d;
    return json{{"seq_len", d.seq_len}, {"d_model", d.d_model}, {"vocab_size", d.vocab_size}, {"padding_idx", d.padding_idx}, {"num_layers", d.num_layers}, {"num_heads", d.num_heads}, {"mlp_hidden", d.mlp_hidden}, {"output_dim", d.output_dim}, {"causal", d.causal}};
}

static CausalLMModel::Config causalLMCfgFromJson(const json& cfg) {
    CausalLMModel::Config out;
    out.vocab_size = jget<int>(cfg, "vocab_size", out.vocab_size);
    out.seq_len = jget<int>(cfg, "seq_len", out.seq_len);
    out.d_model = jget<int>(cfg, "d_model", out.d_model);
    out.num_layers = jget<int>(cfg, "num_layers", out.num_layers);
    out.num_heads = jget<int>(cfg, "num_heads", out.num_heads);
    out.num_kv_heads = jget<int>(cfg, "num_kv_heads", out.num_kv_heads);
    out.mlp_hidden = jget<int>(cfg, "mlp_hidden", out.mlp_hidden);
    out.padding_idx = jget<int>(cfg, "padding_idx", out.padding_idx);
    out.norm_eps = jget<float>(cfg, "norm_eps", out.norm_eps);
    out.rope_theta = jget<float>(cfg, "rope_theta", out.rope_theta);
    return out;
}

static json causalLMDefaultConfigJson() {
    CausalLMModel::Config d;
    return json{{"vocab_size", d.vocab_size},
                {"seq_len", d.seq_len},
                {"d_model", d.d_model},
                {"num_layers", d.num_layers},
                {"num_heads", d.num_heads},
                {"num_kv_heads", d.num_kv_heads},
                {"mlp_hidden", d.mlp_hidden},
                {"padding_idx", d.padding_idx},
                {"norm_eps", d.norm_eps},
                {"rope_theta", d.rope_theta}};
}

static VAETextModel::Config vaeTextCfgFromJson(const json& cfg) {
    VAETextModel::Config out;
    out.vocab_size = jget<int>(cfg, "vocab_size", out.vocab_size);
    out.padding_idx = jget<int>(cfg, "padding_idx", out.padding_idx);
    out.seq_len = jget<int>(cfg, "seq_len", out.seq_len);
    out.d_model = jget<int>(cfg, "d_model", out.d_model);
    out.num_layers = jget<int>(cfg, "num_layers", out.num_layers);
    out.num_heads = jget<int>(cfg, "num_heads", out.num_heads);
    out.mlp_hidden = jget<int>(cfg, "mlp_hidden", out.mlp_hidden);
    out.latent_tokens = jget<int>(cfg, "latent_tokens", out.latent_tokens);
    out.proj_dim = jget<int>(cfg, "proj_dim", out.proj_dim);
    out.decoder_causal = jget<bool>(cfg, "decoder_causal", out.decoder_causal);
    out.tokenizer_integrated = jget<bool>(cfg, "tokenizer_integrated", out.tokenizer_integrated);
    out.enable_conditional_encoder = jget<bool>(cfg, "enable_conditional_encoder", out.enable_conditional_encoder);
    out.enable_context_heads = jget<bool>(cfg, "enable_context_heads", out.enable_context_heads);
    out.context_semantic_dim = jget<int>(cfg, "context_semantic_dim", out.context_semantic_dim);
    out.context_thematic_dim = jget<int>(cfg, "context_thematic_dim", out.context_thematic_dim);
    out.context_dialog_dim = jget<int>(cfg, "context_dialog_dim", out.context_dialog_dim);
    out.stochastic_latent = jget<bool>(cfg, "stochastic_latent", out.stochastic_latent);
    out.dropout = jget<float>(cfg, "dropout", out.dropout);
    return out;
}

static json vaeTextDefaultConfigJson() {
    VAETextModel::Config d;
    const int latent_dim = std::max(1, d.latent_tokens * d.d_model);
    const int logits_dim = std::max(1, d.seq_len * d.vocab_size);
    const int sem_dim = d.enable_context_heads ? std::max(1, d.context_semantic_dim) : 0;
    const int them_dim = d.enable_context_heads ? std::max(1, d.context_thematic_dim) : 0;
    const int dialog_dim = d.enable_context_heads ? std::max(1, d.context_dialog_dim) : 0;
    const int output_dim = logits_dim + 2 * latent_dim + 2 * std::max(1, d.proj_dim) + sem_dim + them_dim + dialog_dim;
    return json{
        {"vocab_size", d.vocab_size},
        {"padding_idx", d.padding_idx},
        {"seq_len", d.seq_len},
        {"d_model", d.d_model},
        {"num_layers", d.num_layers},
        {"num_heads", d.num_heads},
        {"mlp_hidden", d.mlp_hidden},
        {"latent_tokens", d.latent_tokens},
        {"latent_dim", latent_dim},
        {"proj_dim", d.proj_dim},
        {"decoder_causal", d.decoder_causal},
        {"tokenizer_integrated", d.tokenizer_integrated},
        {"enable_conditional_encoder", d.enable_conditional_encoder},
        {"enable_context_heads", d.enable_context_heads},
        {"context_semantic_dim", d.context_semantic_dim},
        {"context_thematic_dim", d.context_thematic_dim},
        {"context_dialog_dim", d.context_dialog_dim},
        {"stochastic_latent", d.stochastic_latent},
        {"dropout", d.dropout},

        // VAE text training defaults consumed by the model trainStep override.
        {"image_dim", logits_dim},
        {"output_dim", output_dim},
        {"target_tensor", "vae_text/target"},
        {"align_weight", 0.0},
        {"context_semantic_weight", 0.08},
        {"context_thematic_weight", 0.05},
        {"context_dialog_weight", 0.10},
        {"kl_beta", 0.01},
        {"kl_warmup_steps", 0},
        {"recon_loss", "ce"},
        {"logvar_clip_min", -6.0},
        {"logvar_clip_max", 2.0},
    };
}

static json vaeTextDecodeDefaultConfigJson() {
    VAETextModel::Config d;
    const int latent_dim = std::max(1, d.latent_tokens * d.d_model);
    const int logits_dim = std::max(1, d.seq_len * d.vocab_size);
    return json{
        {"vocab_size", d.vocab_size},
        {"padding_idx", d.padding_idx},
        {"seq_len", d.seq_len},
        {"d_model", d.d_model},
        {"num_layers", d.num_layers},
        {"num_heads", d.num_heads},
        {"mlp_hidden", d.mlp_hidden},
        {"latent_tokens", d.latent_tokens},
        {"latent_dim", latent_dim},
        {"proj_dim", d.proj_dim},
        {"decoder_causal", d.decoder_causal},
        {"tokenizer_integrated", d.tokenizer_integrated},
        {"enable_conditional_encoder", d.enable_conditional_encoder},
        {"enable_context_heads", d.enable_context_heads},
        {"context_semantic_dim", d.context_semantic_dim},
        {"context_thematic_dim", d.context_thematic_dim},
        {"context_dialog_dim", d.context_dialog_dim},
        {"stochastic_latent", d.stochastic_latent},
        {"dropout", d.dropout},

        // Decoder-only I/O
        {"input_dim", latent_dim},
        {"image_dim", logits_dim},
        {"output_dim", logits_dim},
    };
}

static ViTModel::Config vitCfgFromJson(const json& cfg) {
    ViTModel::Config out;
    out.num_tokens = jget<int>(cfg, "num_tokens", out.num_tokens);
    out.d_model = jget<int>(cfg, "d_model", out.d_model);
    out.num_layers = jget<int>(cfg, "num_layers", out.num_layers);
    out.num_heads = jget<int>(cfg, "num_heads", out.num_heads);
    out.mlp_hidden = jget<int>(cfg, "mlp_hidden", out.mlp_hidden);
    out.output_dim = jget<int>(cfg, "output_dim", out.output_dim);
    out.causal = jget<bool>(cfg, "causal", out.causal);
    return out;
}

static json vitDefaultConfigJson() {
    ViTModel::Config d;
    return json{{"num_tokens", d.num_tokens}, {"d_model", d.d_model}, {"num_layers", d.num_layers}, {"num_heads", d.num_heads}, {"mlp_hidden", d.mlp_hidden}, {"output_dim", d.output_dim}, {"causal", d.causal}};
}

static VAEModel::Config vaeCfgFromJson(const json& cfg) {
    VAEModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.latent_dim = jget<int>(cfg, "latent_dim", out.latent_dim);
    out.hidden_dim = jget<int>(cfg, "hidden_dim", out.hidden_dim);
    return out;
}

static json vaeDefaultConfigJson() {
    VAEModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"latent_dim", d.latent_dim}, {"hidden_dim", d.hidden_dim}};
}

static VAEConvModel::Config vaeConvCfgFromJson(const json& cfg) {
    VAEConvModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.latent_h = jget<int>(cfg, "latent_h", out.latent_h);
    out.latent_w = jget<int>(cfg, "latent_w", out.latent_w);
    out.latent_c = jget<int>(cfg, "latent_c", out.latent_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.latent_spatial_context = jget<bool>(cfg, "latent_spatial_context", out.latent_spatial_context);
    out.latent_detail_path = jget<bool>(cfg, "latent_detail_path", out.latent_detail_path);
    out.latent_context_channels = jget<int>(cfg, "latent_context_channels", out.latent_context_channels);

    // Latent stochasticity (reparameterization noise)
    out.stochastic_latent = jget<bool>(cfg, "stochastic_latent", out.stochastic_latent);

    // Blocs optionnels
    out.resnet = jget<bool>(cfg, "use_resnet",
                     jget<bool>(cfg, "use_attention",
                         jget<bool>(cfg, "resnet", out.resnet)));
    out.attention = jget<bool>(cfg, "use_attn",
                        jget<bool>(cfg, "attention", out.attention));
    out.enc_norm           = jget<std::string>(cfg, "enc_norm", out.enc_norm);
    out.enc_gn_groups      = jget<int>(cfg, "enc_gn_groups", out.enc_gn_groups);
    out.dec_norm           = jget<std::string>(cfg, "dec_norm", out.dec_norm);
    out.dec_gn_groups      = jget<int>(cfg, "dec_gn_groups", out.dec_gn_groups);
    out.decoder_upsample   = jget<std::string>(cfg, "decoder_upsample", out.decoder_upsample);
    out.attn_heads         = jget<int>(cfg, "attn_heads", out.attn_heads);
    out.resnet_max_tokens  = jget<int>(cfg, "resnet_max_tokens", out.resnet_max_tokens);
    out.attn_max_tokens        = jget<int>(cfg, "attn_max_tokens", out.attn_max_tokens);
    out.use_skip_connections   = jget<bool>(cfg, "use_skip_connections", out.use_skip_connections);
    out.use_encoder_prior      = jget<bool>(cfg, "use_encoder_prior", out.use_encoder_prior);
    out.d_model                = jget<int>(cfg, "d_model", out.d_model);

    // Optional text conditioning
    out.text_cond = jget<bool>(cfg, "text_cond", out.text_cond);
    out.vocab_size = jget<int>(cfg, "vocab_size", out.vocab_size);
    out.seq_len = jget<int>(cfg, "seq_len", out.seq_len);
    out.text_d_model = jget<int>(cfg, "text_d_model", out.text_d_model);
    out.proj_dim = jget<int>(cfg, "proj_dim", out.proj_dim);
    return out;
}

static json vaeConvDefaultConfigJson() {
    VAEConvModel::Config d;
    return json{{"image_w", d.image_w},
                {"image_h", d.image_h},
                {"image_c", d.image_c},
                {"latent_h", d.latent_h},
                {"latent_w", d.latent_w},
                {"latent_c", d.latent_c},
                {"base_channels", d.base_channels},
                {"latent_spatial_context", d.latent_spatial_context},
                {"latent_detail_path", d.latent_detail_path},
                {"latent_context_channels", d.latent_context_channels},
                {"stochastic_latent", d.stochastic_latent},
                {"resnet", d.resnet},
                {"attention", d.attention},
                {"enc_norm", d.enc_norm},
                {"enc_gn_groups", d.enc_gn_groups},
                {"dec_norm", d.dec_norm},
                {"dec_gn_groups", d.dec_gn_groups},
                {"decoder_upsample", d.decoder_upsample},
                {"attn_heads", d.attn_heads},
                {"resnet_max_tokens", d.resnet_max_tokens},
                {"attn_max_tokens", d.attn_max_tokens},
                {"use_skip_connections", d.use_skip_connections},
                {"use_encoder_prior", d.use_encoder_prior},
                {"d_model", d.d_model},
                {"text_cond", d.text_cond},
                {"vocab_size", d.vocab_size},
                {"seq_len", d.seq_len},
                {"text_d_model", d.text_d_model},
                {"proj_dim", d.proj_dim}};
}

static ExternalSafeTensorsModel::Config externalSafeTensorsCfgFromJson(const json& cfg) {
    ExternalSafeTensorsModel::Config out;
    out.source_safetensors = jget<std::string>(cfg, "source_safetensors", out.source_safetensors);
    out.max_tensors = jget<int>(cfg, "max_tensors", out.max_tensors);
    if (cfg.is_object()) {
        auto parse_prefixes = [&](const char* key, std::vector<std::string>& out_vec) {
            auto it = cfg.find(key);
            if (it == cfg.end() || it->is_null()) return;
            if (it->is_string()) {
                out_vec.push_back(it->get<std::string>());
                return;
            }
            if (it->is_array()) {
                for (const auto& v : *it) {
                    if (v.is_string()) out_vec.push_back(v.get<std::string>());
                }
            }
        };
        parse_prefixes("include_prefixes", out.include_prefixes);
        parse_prefixes("exclude_prefixes", out.exclude_prefixes);
    }
    return out;
}

static json externalSafeTensorsDefaultConfigJson() {
    ExternalSafeTensorsModel::Config d;
    return json{{"source_safetensors", d.source_safetensors},
                {"include_prefixes", d.include_prefixes},
                {"exclude_prefixes", d.exclude_prefixes},
                {"max_tensors", d.max_tensors}};
}

static ResNetModel::Config resnetCfgFromJson(const json& cfg) {
    ResNetModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.num_classes = jget<int>(cfg, "num_classes", out.num_classes);
    out.blocks1 = jget<int>(cfg, "blocks1", out.blocks1);
    out.blocks2 = jget<int>(cfg, "blocks2", out.blocks2);
    out.blocks3 = jget<int>(cfg, "blocks3", out.blocks3);
    out.blocks4 = jget<int>(cfg, "blocks4", out.blocks4);
    return out;
}

static json resnetDefaultConfigJson() {
    ResNetModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}, {"num_classes", d.num_classes}, {"blocks1", d.blocks1}, {"blocks2", d.blocks2}, {"blocks3", d.blocks3}, {"blocks4", d.blocks4}};
}

static UNetModel::Config unetCfgFromJson(const json& cfg) {
    UNetModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.depth = jget<int>(cfg, "depth", out.depth);
    return out;
}

static json unetDefaultConfigJson() {
    UNetModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}, {"depth", d.depth}};
}

static MobileNetModel::Config mobilenetCfgFromJson(const json& cfg) {
    MobileNetModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.num_classes = jget<int>(cfg, "num_classes", out.num_classes);
    return out;
}

static json mobilenetDefaultConfigJson() {
    MobileNetModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}, {"num_classes", d.num_classes}};
}

static VGG16Model::Config vgg16CfgFromJson(const json& cfg) {
    VGG16Model::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.num_classes = jget<int>(cfg, "num_classes", out.num_classes);
    out.fc_hidden = jget<int>(cfg, "fc_hidden", out.fc_hidden);
    out.dropout = jget<float>(cfg, "dropout", out.dropout);
    return out;
}

static json vgg16DefaultConfigJson() {
    VGG16Model::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}, {"num_classes", d.num_classes}, {"fc_hidden", d.fc_hidden}, {"dropout", d.dropout}};
}

static MedicalDetectorModel::Config medicalDetectorCfgFromJson(const json& cfg) {
    MedicalDetectorModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.num_classes = jget<int>(cfg, "num_classes", out.num_classes);
    out.object_loss_weight = jget<float>(cfg, "object_loss_weight", out.object_loss_weight);
    out.box_loss_weight = jget<float>(cfg, "box_loss_weight", out.box_loss_weight);
    out.class_loss_weight = jget<float>(cfg, "class_loss_weight", out.class_loss_weight);
    out.object_pos_weight = jget<float>(cfg, "object_pos_weight", out.object_pos_weight);
    return out;
}

static json medicalDetectorDefaultConfigJson() {
    MedicalDetectorModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h},
                {"image_c", d.image_c}, {"base_channels", d.base_channels},
                {"num_classes", d.num_classes},
                {"object_loss_weight", d.object_loss_weight},
                {"box_loss_weight", d.box_loss_weight},
                {"class_loss_weight", d.class_loss_weight},
                {"object_pos_weight", d.object_pos_weight}};
}

static VGG19Model::Config vgg19CfgFromJson(const json& cfg) {
    VGG19Model::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.num_classes = jget<int>(cfg, "num_classes", out.num_classes);
    out.fc_hidden = jget<int>(cfg, "fc_hidden", out.fc_hidden);
    out.dropout = jget<float>(cfg, "dropout", out.dropout);
    return out;
}

static json vgg19DefaultConfigJson() {
    VGG19Model::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}, {"num_classes", d.num_classes}, {"fc_hidden", d.fc_hidden}, {"dropout", d.dropout}};
}

static VGG16FeatModel::Config vgg16FeatCfgFromJson(const json& cfg) {
    VGG16FeatModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.enc_norm = jget<std::string>(cfg, "enc_norm", out.enc_norm);
    out.enc_gn_groups = jget<int>(cfg, "enc_gn_groups", out.enc_gn_groups);
    return out;
}

static json vgg16FeatDefaultConfigJson() {
    VGG16FeatModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c},
                {"base_channels", d.base_channels}, {"enc_norm", d.enc_norm},
                {"enc_gn_groups", d.enc_gn_groups}};
}

static PatchDiscriminatorModel::Config patchDiscCfgFromJson(const json& cfg) {
    PatchDiscriminatorModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.num_down = jget<int>(cfg, "num_down", out.num_down);
    return out;
}

static json patchDiscDefaultConfigJson() {
    PatchDiscriminatorModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}, {"num_down", d.num_down}};
}

static DiffusionModel::Config diffusionCfgFromJson(const json& cfg) {
    DiffusionModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.time_dim = jget<int>(cfg, "time_dim", out.time_dim);
    out.hidden_dim = jget<int>(cfg, "hidden_dim", out.hidden_dim);
    out.dropout = jget<float>(cfg, "dropout", out.dropout);
    return out;
}

static json diffusionDefaultConfigJson() {
    DiffusionModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"time_dim", d.time_dim}, {"hidden_dim", d.hidden_dim}, {"dropout", d.dropout}};
}

static GanLatentModel::Config ganLatentCfgFromJson(const json& cfg) {
    GanLatentModel::Config out;
    out.prompt_dim = jget<int>(cfg, "prompt_dim", out.prompt_dim);
    out.noise_dim = jget<int>(cfg, "noise_dim", out.noise_dim);
    out.latent_dim = jget<int>(cfg, "latent_dim", out.latent_dim);
    out.hidden_dim = jget<int>(cfg, "hidden_dim", out.hidden_dim);
    out.num_hidden_layers = jget<int>(cfg, "num_hidden_layers", out.num_hidden_layers);
    out.dropout = jget<float>(cfg, "dropout", out.dropout);
    return out;
}

static json ganLatentDefaultConfigJson() {
    GanLatentModel::Config d;
    return json{{"prompt_dim", d.prompt_dim},
                {"noise_dim", d.noise_dim},
                {"latent_dim", d.latent_dim},
                {"hidden_dim", d.hidden_dim},
                {"num_hidden_layers", d.num_hidden_layers},
                {"dropout", d.dropout}};
}

static CondDiffusionModel::Config condDiffusionCfgFromJson(const json& cfg) {
    CondDiffusionModel::Config out;
    out.prompt_dim = jget<int>(cfg, "prompt_dim", out.prompt_dim);
    out.latent_w = jget<int>(cfg, "latent_w", out.latent_w);
    out.latent_h = jget<int>(cfg, "latent_h", out.latent_h);
    out.latent_c = jget<int>(cfg, "latent_c", out.latent_c);
    out.time_dim = jget<int>(cfg, "time_dim", out.time_dim);
    out.hidden_dim = jget<int>(cfg, "hidden_dim", out.hidden_dim);
    out.dropout = jget<float>(cfg, "dropout", out.dropout);
    return out;
}

static json condDiffusionDefaultConfigJson() {
    CondDiffusionModel::Config d;
    return json{{"prompt_dim", d.prompt_dim},
                {"latent_w", d.latent_w},
                {"latent_h", d.latent_h},
                {"latent_c", d.latent_c},
                {"time_dim", d.time_dim},
                {"hidden_dim", d.hidden_dim},
                {"dropout", d.dropout}};
}

static LumenLatentDiffusionModel::Config lumenCfgFromJson(const json& cfg) {
    LumenLatentDiffusionModel::Config out;
    out.seed = jget<int>(cfg, "seed", out.seed);
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.latent_w = jget<int>(cfg, "latent_w", out.latent_w);
    out.latent_h = jget<int>(cfg, "latent_h", out.latent_h);
    out.latent_c = jget<int>(cfg, "latent_c", out.latent_c);
    out.vae_base_channels = jget<int>(cfg, "vae_base_channels", out.vae_base_channels);
    out.vae_latent_spatial_context = jget<bool>(cfg, "vae_latent_spatial_context", out.vae_latent_spatial_context);
    out.vae_latent_detail_path = jget<bool>(cfg, "vae_latent_detail_path", out.vae_latent_detail_path);
    out.vae_latent_context_channels = jget<int>(cfg, "vae_latent_context_channels", out.vae_latent_context_channels);
    out.vae_stochastic_latent = jget<bool>(cfg, "vae_stochastic_latent", out.vae_stochastic_latent);
    out.vae_resnet = jget<bool>(cfg, "vae_use_resnet",
                         jget<bool>(cfg, "vae_resnet", out.vae_resnet));
    out.vae_attention = jget<bool>(cfg, "vae_use_attn",
                            jget<bool>(cfg, "vae_attention", out.vae_attention));
    out.vae_use_skip_connections = jget<bool>(cfg, "vae_use_skip_connections", out.vae_use_skip_connections);
    out.vae_use_encoder_prior = jget<bool>(cfg, "vae_use_encoder_prior", out.vae_use_encoder_prior);
    out.vae_enc_norm = jget<std::string>(cfg, "vae_enc_norm", out.vae_enc_norm);
    out.vae_dec_norm = jget<std::string>(cfg, "vae_dec_norm", out.vae_dec_norm);
    out.vae_decoder_upsample = jget<std::string>(cfg, "vae_decoder_upsample", out.vae_decoder_upsample);
    out.vae_enc_gn_groups = jget<int>(cfg, "vae_enc_gn_groups", out.vae_enc_gn_groups);
    out.vae_dec_gn_groups = jget<int>(cfg, "vae_dec_gn_groups", out.vae_dec_gn_groups);
    out.vae_attn_heads = jget<int>(cfg, "vae_attn_heads", out.vae_attn_heads);
    out.vae_attn_max_tokens = jget<int>(cfg, "vae_attn_max_tokens", out.vae_attn_max_tokens);
    out.vae_resnet_max_tokens = jget<int>(cfg, "vae_resnet_max_tokens", out.vae_resnet_max_tokens);
    out.vae_scale = jget<float>(cfg, "vae_scale", out.vae_scale);
    out.vae_shift = jget<float>(cfg, "vae_shift", out.vae_shift);
    out.vae_checkpoint = jget<std::string>(cfg, "vae_checkpoint", out.vae_checkpoint);
    out.patch_size = jget<int>(cfg, "patch_size", out.patch_size);
    out.hidden_size = jget<int>(cfg, "hidden_size", out.hidden_size);
    out.depth = jget<int>(cfg, "depth", out.depth);
    out.mlp_ratio = jget<float>(cfg, "mlp_ratio", out.mlp_ratio);
    out.vocab_size = jget<int>(cfg, "vocab_size", out.vocab_size);
    out.text_seq_len = jget<int>(cfg, "text_seq_len", out.text_seq_len);
    out.dynamic_text_seq_len = jget<bool>(cfg,"dynamic_text_seq_len",out.dynamic_text_seq_len);
    out.text_layers = jget<int>(cfg, "text_layers", out.text_layers);
    out.num_heads = jget<int>(cfg, "num_heads", out.num_heads);
    out.diffusion_steps = jget<int>(cfg, "diffusion_steps", out.diffusion_steps);
    out.beta_start = jget<float>(cfg, "beta_start", out.beta_start);
    out.beta_end = jget<float>(cfg, "beta_end", out.beta_end);
    out.preview_timestep = jget<int>(cfg, "preview_timestep", out.preview_timestep);
    out.kl_beta = jget<float>(cfg, "kl_beta", out.kl_beta);
    out.kl_warmup_steps = jget<int>(cfg, "kl_warmup_steps", out.kl_warmup_steps);
    return out;
}

static json lumenDefaultConfigJson() {
    const LumenLatentDiffusionModel::Config d;
    return json{
        {"seed", d.seed}, {"image_w", d.image_w}, {"image_h", d.image_h},
        {"image_c", d.image_c}, {"latent_w", d.latent_w}, {"latent_h", d.latent_h},
        {"latent_c", d.latent_c}, {"vae_base_channels", d.vae_base_channels},
        {"vae_latent_spatial_context", d.vae_latent_spatial_context},
        {"vae_latent_detail_path", d.vae_latent_detail_path},
        {"vae_latent_context_channels", d.vae_latent_context_channels},
        {"vae_stochastic_latent", d.vae_stochastic_latent},
        {"vae_resnet", d.vae_resnet}, {"vae_attention", d.vae_attention},
        {"vae_use_skip_connections", d.vae_use_skip_connections},
        {"vae_use_encoder_prior", d.vae_use_encoder_prior},
        {"vae_enc_norm", d.vae_enc_norm}, {"vae_dec_norm", d.vae_dec_norm},
        {"vae_decoder_upsample", d.vae_decoder_upsample},
        {"vae_enc_gn_groups", d.vae_enc_gn_groups},
        {"vae_dec_gn_groups", d.vae_dec_gn_groups},
        {"vae_attn_heads", d.vae_attn_heads},
        {"vae_attn_max_tokens", d.vae_attn_max_tokens},
        {"vae_resnet_max_tokens", d.vae_resnet_max_tokens},
        {"vae_scale", d.vae_scale}, {"vae_shift", d.vae_shift},
        {"vae_checkpoint", d.vae_checkpoint}, {"patch_size", d.patch_size},
        {"hidden_size", d.hidden_size}, {"depth", d.depth},
        {"mlp_ratio", d.mlp_ratio}, {"vocab_size", d.vocab_size},
        {"text_seq_len", d.text_seq_len}, {"dynamic_text_seq_len", d.dynamic_text_seq_len}, {"text_layers", d.text_layers},
        {"num_heads", d.num_heads}, {"diffusion_steps", d.diffusion_steps},
        {"beta_start", d.beta_start}, {"beta_end", d.beta_end},
        {"preview_timestep", d.preview_timestep}, {"kl_beta", d.kl_beta},
        {"kl_warmup_steps", d.kl_warmup_steps},
    };
}

static SD35Model::Config sd35CfgFromJson(const json& cfg) {
    SD35Model::Config out;
    out.stub_only = jget<bool>(cfg, "stub_only", out.stub_only);
    out.q_len = jget<int>(cfg, "q_len", out.q_len);
    out.kv_len = jget<int>(cfg, "kv_len", out.kv_len);
    out.d_model = jget<int>(cfg, "d_model", out.d_model);
    out.num_heads = jget<int>(cfg, "num_heads", out.num_heads);
    out.num_layers = jget<int>(cfg, "num_layers", out.num_layers);
    out.mlp_hidden = jget<int>(cfg, "mlp_hidden", out.mlp_hidden);
    out.causal = jget<bool>(cfg, "causal", out.causal);

    // Compat: si un ancien config passait input_dim/output_dim, on peut en déduire
    // q_len/kv_len seulement si d_model est fourni et que l'input est divisible.
    // (On garde un comportement safe: si ambigu, on ignore.)
    const int legacy_input_dim = jget<int>(cfg, "input_dim", 0);
    if (legacy_input_dim > 0 && out.d_model > 0) {
        const int dm = std::max(1, out.d_model);
        const int total_tokens = legacy_input_dim / dm;
        if (total_tokens * dm == legacy_input_dim && total_tokens >= 2) {
            // Heuristique: split moitié/moitié
            out.q_len = std::max(1, total_tokens / 2);
            out.kv_len = std::max(1, total_tokens - out.q_len);
        }
    }
    return out;
}

static json sd35DefaultConfigJson() {
    SD35Model::Config d;
    return json{
        {"stub_only", d.stub_only},
        {"q_len", d.q_len},
        {"kv_len", d.kv_len},
        {"d_model", d.d_model},
        {"num_heads", d.num_heads},
        {"num_layers", d.num_layers},
        {"mlp_hidden", d.mlp_hidden},
        {"causal", d.causal},
    };
}

static HFSDXLTransformerBlockModel::Config hfSdxlTransformerBlockCfgFromJson(const json& cfg) {
    HFSDXLTransformerBlockModel::Config out;
    out.q_len = jget<int>(cfg, "q_len", out.q_len);
    out.kv_len = jget<int>(cfg, "kv_len", out.kv_len);
    out.d_model = jget<int>(cfg, "d_model", out.d_model);
    out.context_dim = jget<int>(cfg, "context_dim", out.context_dim);
    out.num_heads = jget<int>(cfg, "num_heads", out.num_heads);
    out.ff_hidden = jget<int>(cfg, "ff_hidden", out.ff_hidden);
    out.self_attn_qkv_bias = jget<bool>(cfg, "self_attn_qkv_bias", out.self_attn_qkv_bias);
    out.self_attn_out_bias = jget<bool>(cfg, "self_attn_out_bias", out.self_attn_out_bias);
    out.cross_attn_out_bias = jget<bool>(cfg, "cross_attn_out_bias", out.cross_attn_out_bias);
    return out;
}

static json hfSdxlTransformerBlockDefaultConfigJson() {
    HFSDXLTransformerBlockModel::Config d;
    return json{{"q_len", d.q_len},
                {"kv_len", d.kv_len},
                {"d_model", d.d_model},
                {"context_dim", d.context_dim},
                {"num_heads", d.num_heads},
                {"ff_hidden", d.ff_hidden},
                {"self_attn_qkv_bias", d.self_attn_qkv_bias},
                {"self_attn_out_bias", d.self_attn_out_bias},
                {"cross_attn_out_bias", d.cross_attn_out_bias}};
}

} // namespace

std::string resolveArchitectureFromConfig(const json& full_config, const std::string& default_arch) {
    std::string arch = default_arch;
    if (!full_config.is_object()) return arch;

    auto pickString = [&](const char* key) -> std::string {
        auto it = full_config.find(key);
        if (it != full_config.end() && it->is_string()) return it->get<std::string>();
        return std::string();
    };

    {
        std::string a = pickString("architecture");
        if (!a.empty()) return a;
    }
    {
        std::string t = pickString("type");
        if (!t.empty()) return t;
    }

    // Fallback: config.model.{architecture|type}
    auto mit = full_config.find("model");
    if (mit != full_config.end() && mit->is_object()) {
        const json& m = *mit;
        auto ait = m.find("architecture");
        if (ait != m.end() && ait->is_string()) return ait->get<std::string>();
        auto tit = m.find("type");
        if (tit != m.end() && tit->is_string()) return tit->get<std::string>();
    }

    return arch;
}

json cfgFromConfig(const json& full_config, std::string* out_arch, const std::string& default_arch) {
    const std::string arch = resolveArchitectureFromConfig(full_config, default_arch);
    if (out_arch) *out_arch = arch;

    // Start from default config for the selected architecture.
    json cfg = ModelArchitectures::defaultConfig(arch);

    // Flatten the common override sections into the root config (so buildFromConfig sees them).
    if (full_config.is_object()) {
        auto it_model = full_config.find("model");
        if (it_model != full_config.end() && it_model->is_object()) {
            mergeIntoCfg(cfg, *it_model);
        }

        auto it_arch = full_config.find(arch);
        if (it_arch != full_config.end() && it_arch->is_object()) {
            mergeIntoCfg(cfg, *it_arch);
        }

        // Preserve all parent keys verbatim so the framework can read them later.
        // This intentionally includes `model` and the architecture section name.
        for (auto it = full_config.begin(); it != full_config.end(); ++it) {
            cfg[it.key()] = it.value();
        }
    }

    return cfg;
}

std::shared_ptr<Model> createFromConfig(const json& full_config,
                                        json* out_cfg,
                                        std::string* out_arch,
                                        const std::string& default_arch) {
    std::string arch;
    json cfg = cfgFromConfig(full_config, &arch, default_arch);
    if (out_arch) *out_arch = arch;
    if (out_cfg) *out_cfg = cfg;
    return ModelArchitectures::create(arch, cfg);
}

std::shared_ptr<Model> Registry::create(const std::string& name, const json& config) const {
    ensureBuiltinsRegistered();
    const std::string key = canonicalArchName(name);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        throw std::runtime_error("ModelArchitectures::create: unknown architecture: " + name);
    }

    if (config.is_object() && !config.empty()) {
        std::cerr << "[registry] create name=" << key
                  << " config_dtype=" << config.value("dtype", std::string("float32"))
                  << std::endl;
    } else {
        std::cerr << "[registry] create name=" << key << std::endl;
    }

    json cfg = it->second.default_config;
    if (config.is_object() && !config.empty()) {
        mergeInto(cfg, config);
    }

    // Standardize metadata for serialization/rebuild.
    if (!cfg.is_object()) cfg = json::object();
    cfg["type"] = key;

    auto model = it->second.create(cfg);
    if (!model) {
        throw std::runtime_error("ModelArchitectures::create: factory returned null for: " + name);
    }

    // Preserve graph-derived metadata while retaining parent configuration
    // sections used by serialization and scripting.
    json effective_config = cfg;
    mergeInto(effective_config, model->modelConfig);
    model->modelConfig = std::move(effective_config);

    // Propagate dtype from config -> runtime default dtype.
    // This must be done here (framework-level) so callers don't need to manually call Model.dtype().
    auto it_dtype = cfg.find("dtype");
    if (it_dtype != cfg.end() && it_dtype->is_string()) {
        // setDefaultDType validates the value and throws on unknown dtype.
        model->setDefaultDType(it_dtype->get<std::string>());
    }
    if (!Model::frameworkLogsSuppressed()) {
        std::cerr << "[registry] create done name=" << key
                  << " layers=" << model->getLayers().size()
                  << " params=" << model->totalParamCount() << std::endl;
    }
    return model;
}

void Registry::ensureBuiltinsRegistered() const {
    std::call_once(builtins_once_, [&]() {
        entries_.emplace(
            "basic_mlp",
            Entry{
                "basic_mlp",
                "Basic MLP (régression: input->MLP->output)",
                basicMlpDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<BasicMLPModel>();
                    m->buildFromConfig(basicMlpCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "transformer",
            Entry{
                "transformer",
                "Transformer encoder (float-only: input=seq_len*d_model)",
                transformerDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<TransformerModel>();
                    m->buildFromConfig(transformerCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "causal_lm",
            Entry{
                "causal_lm",
                "Decoder-only causal language model (native Mimir layers)",
                causalLMDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<CausalLMModel>();
                    m->buildFromConfig(causalLMCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "vae_text",
            Entry{
                "vae_text",
                "VAEText (text_ids -> recon||mu||logvar||img_proj||text_proj)",
                vaeTextDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<VAETextModel>();
                    m->buildFromConfig(vaeTextCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "vae_text_decode",
            Entry{
                "vae_text_decode",
                "VAEText decoder-only (input=z latent, output=logits seq_len*vocab)",
                vaeTextDecodeDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<Model>();
                    VAETextModel::buildDecoderInto(*m, vaeTextCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "vit",
            Entry{
                "vit",
                "ViT (float-only: input=patch embeddings num_tokens*d_model)",
                vitDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<ViTModel>();
                    m->buildFromConfig(vitCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "vae",
            Entry{
                "vae",
                "VAE-style autoencoder (output packs recon||mu||logvar)",
                vaeDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<VAEModel>();
                    m->buildFromConfig(vaeCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "vae_conv",
            Entry{
                "vae_conv",
                "Convolutional VAE (output packs recon||mu||logvar; spatial latent)",
                vaeConvDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<VAEConvModel>();
                    m->buildFromConfig(vaeConvCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "vae_conv_decode",
            Entry{
                "vae_conv_decode",
                "Convolutional VAE decoder-only (input=z latent, output=recon RGB)",
                vaeConvDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<Model>();
                    VAEConvModel::buildDecoderInto(*m, vaeConvCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "resnet",
            Entry{
                "resnet",
                "ResNet (simplified ResNet18-like)",
                resnetDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<ResNetModel>();
                    m->buildFromConfig(resnetCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "unet",
            Entry{
                "unet",
                "UNet (simplified encoder-decoder with skip concatenations)",
                unetDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<UNetModel>();
                    m->buildFromConfig(unetCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "mobilenet",
            Entry{
                "mobilenet",
                "MobileNetV1-style (DepthwiseConv2d + pointwise Conv2d)",
                mobilenetDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<MobileNetModel>();
                    m->buildFromConfig(mobilenetCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "vgg16",
            Entry{
                "vgg16",
                "VGG16 (simplified, downsample via stride-2 conv)",
                vgg16DefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<VGG16Model>();
                    m->buildFromConfig(vgg16CfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "vgg19",
            Entry{
                "vgg19",
                "VGG19 (simplified, downsample via stride-2 conv)",
                vgg19DefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<VGG19Model>();
                    m->buildFromConfig(vgg19CfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "medical_detector",
            Entry{
                "medical_detector",
                "Medical anomaly detector with a native spatial detection grid",
                medicalDetectorDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<MedicalDetectorModel>();
                    m->buildFromConfig(medicalDetectorCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "vgg16_feat",
            Entry{
                "vgg16_feat",
                "VGG16 feature extractor (GAP features for perceptual loss)",
                vgg16FeatDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<VGG16FeatModel>();
                    m->buildFromConfig(vgg16FeatCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "hf_clip_text_encoder_1",
            Entry{
                "hf_clip_text_encoder_1",
                "Encodeur texte CLIP/SDXL exécutable pour checkpoints HuggingFace/PyTorch (conditioner.embedders.0)",
                hfClipTextEncoder1DefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<HFCLIPTextEncoder1Model>();
                    m->buildFromConfig(hfClipTextEncoder1CfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "hf_clip_text_encoder_2",
            Entry{
                "hf_clip_text_encoder_2",
                "Encodeur texte OpenCLIP/SDXL exécutable pour checkpoints HuggingFace/PyTorch",
                hfClipTextEncoder2DefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<HFCLIPTextEncoder2Model>();
                    m->buildFromConfig(hfClipTextEncoder2CfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "hf_vae_decoder",
            Entry{
                "hf_vae_decoder",
                "Décodeur VAE SDXL/HuggingFace exécutable pour le composant first_stage_model.decoder",
                hfVaeDecoderDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<HFVaeDecoderModel>();
                    m->buildFromConfig(hfVaeDecoderCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "hf_sdxl_transformer_block",
            Entry{
                "hf_sdxl_transformer_block",
                "Bloc transformer SDXL/HuggingFace exécutable avec SelfAttention, CrossAttention et GEGLU",
                hfSdxlTransformerBlockDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<HFSDXLTransformerBlockModel>();
                    m->buildFromConfig(hfSdxlTransformerBlockCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "external_safetensors_base",
            Entry{
                "external_safetensors_base",
                "Base non-executable qui reflète exactement les clés d'un checkpoint safetensors externe",
                externalSafeTensorsDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<ExternalSafeTensorsModel>();
                    m->buildFromConfig(externalSafeTensorsCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "patch_discriminator",
            Entry{
                "patch_discriminator",
                "PatchGAN-like discriminator (image -> patch logits)",
                patchDiscDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<PatchDiscriminatorModel>();
                    m->buildFromConfig(patchDiscCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "diffusion",
            Entry{
                "diffusion",
                "Diffusion epsilon predictor (baseline MLP: input=t_embed||x_t)",
                diffusionDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<DiffusionModel>();
                    m->buildFromConfig(diffusionCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace("Gan_Style", Entry{
            "Gan_Style", "Native image-to-image style GAN (convolutional generator and patch discriminator)",
            {{"image_w",64},{"image_h",64},{"image_c",3},{"base_channels",16},
             {"discriminator_channels",16},{"content_weight",10.f},{"discriminator_lr",0.001f}},
            [](const json& cfg) -> std::shared_ptr<Model> {
                GanStyleModel::Config c;
                c.image_w=cfg.value("image_w",64); c.image_h=cfg.value("image_h",64);
                c.image_c=cfg.value("image_c",3); c.base_channels=cfg.value("base_channels",16);
                c.discriminator_channels=cfg.value("discriminator_channels",16);
                c.content_weight=cfg.value("content_weight",10.f);
                c.discriminator_lr=cfg.value("discriminator_lr",0.001f);
                auto model=std::make_shared<GanStyleModel>(); model->buildFromConfig(c); return model;
            }});

        entries_.emplace(
            "gan_latent",
            Entry{
                "gan_latent",
                "GAN-like latent generator (baseline MLP: input=prompt||noise -> latent)",
                ganLatentDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<GanLatentModel>();
                    m->buildFromConfig(ganLatentCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "cond_diffusion",
            Entry{
                "cond_diffusion",
                "Conditional diffusion epsilon predictor (baseline MLP: input=prompt||t_embed||x_t)",
                condDiffusionDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<CondDiffusionModel>();
                    m->buildFromConfig(condDiffusionCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "lumen_diffusion",
            Entry{
                "lumen_diffusion",
                "Latent text-to-image Diffusion Transformer with a frozen VAE",
                lumenDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<LumenLatentDiffusionModel>();
                    m->buildFromConfig(lumenCfgFromJson(cfg));
                    return m;
                },
            }
        );

        entries_.emplace(
            "sd3_5",
            Entry{
                "sd3_5",
                "Stable Diffusion 3.5 (placeholder: registry + skeleton model)",
                sd35DefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<SD35Model>();
                    m->buildFromConfig(sd35CfgFromJson(cfg));
                    return m;
                },
            }
        );
    });
}

} // namespace ModelArchitectures
