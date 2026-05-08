#include "ModelArchitectures.hpp"

#include "Models/MLP/BasicMLPModel.hpp"
#include "Models/NLP/TransformerModel.hpp"
#include "Models/NLP/VAETextModel.hpp"
#include "Models/Vision/ViTModel.hpp"
#include "Models/Vision/VAEModel.hpp"
#include "Models/Vision/VAEConvModel.hpp"
#include "Models/Vision/GanLatentModel.hpp"
#include "Models/Vision/CNN/ResNetModel.hpp"
#include "Models/Vision/UNetModel.hpp"
#include "Models/Vision/CNN/MobileNetModel.hpp"
#include "Models/Vision/CNN/VGG16Model.hpp"
#include "Models/Vision/CNN/VGG19Model.hpp"
#include "Models/Vision/CNN/VGG16FeatModel.hpp"
#include "Models/Diffusion/DiffusionModel.hpp"
#include "Models/Diffusion/CondDiffusionModel.hpp"
#include "Models/Diffusion/PonyXLDDPMModel.hpp"
#include "Models/Diffusion/SD35Model.hpp"
#include "Models/Vision/PatchDiscriminatorModel.hpp"

#include <algorithm>
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

static json transformerDefaultConfigJson() {
    TransformerModel::Config d;
    return json{{"seq_len", d.seq_len}, {"d_model", d.d_model}, {"vocab_size", d.vocab_size}, {"padding_idx", d.padding_idx}, {"num_layers", d.num_layers}, {"num_heads", d.num_heads}, {"mlp_hidden", d.mlp_hidden}, {"output_dim", d.output_dim}, {"causal", d.causal}};
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
    out.stochastic_latent = jget<bool>(cfg, "stochastic_latent", out.stochastic_latent);
    out.dropout = jget<float>(cfg, "dropout", out.dropout);
    return out;
}

static json vaeTextDefaultConfigJson() {
    VAETextModel::Config d;
    const int latent_dim = std::max(1, d.latent_tokens * d.d_model);
    const int logits_dim = std::max(1, d.seq_len * d.vocab_size);
    const int output_dim = logits_dim + 2 * latent_dim + 2 * std::max(1, d.proj_dim);
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
        {"stochastic_latent", d.stochastic_latent},
        {"dropout", d.dropout},

        // Training helper defaults (Model::trainStepVAEText)
        {"image_dim", logits_dim},
        {"output_dim", output_dim},
        {"target_tensor", "vae_text/target"},
        {"align_weight", 0.0},
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

static PonyXLDDPMModel::Config ponyxlDdpmCfgFromJson(const json& cfg) {
    PonyXLDDPMModel::Config out;

    out.seed = jget<int>(cfg, "seed", out.seed);
    out.d_model = jget<int>(cfg, "d_model", out.d_model);
    out.max_vocab = jget<int>(cfg, "max_vocab", out.max_vocab);
    out.text_ctx_len = jget<int>(cfg, "text_ctx_len", out.text_ctx_len);
    out.text_clip_like = jget<bool>(cfg, "text_clip_like", out.text_clip_like);
    out.global_ctx_tokens = jget<int>(cfg, "global_ctx_tokens", out.global_ctx_tokens);
    out.text_bottleneck_meanpool = jget<bool>(cfg, "text_bottleneck_meanpool", out.text_bottleneck_meanpool);

    out.latent_seq_len = jget<int>(cfg, "latent_seq_len", out.latent_seq_len);
    out.latent_in_dim = jget<int>(cfg, "latent_in_dim", out.latent_in_dim);
    out.num_heads = jget<int>(cfg, "num_heads", out.num_heads);

    out.sdxl_time_cond = jget<bool>(cfg, "sdxl_time_cond", out.sdxl_time_cond);
    out.timestep_cond = jget<std::string>(cfg, "timestep_cond", out.timestep_cond);
    out.unet_layers = jget<int>(cfg, "unet_layers", out.unet_layers);
    out.text_layers = jget<int>(cfg, "text_layers", out.text_layers);
    out.mlp_hidden = jget<int>(cfg, "mlp_hidden", out.mlp_hidden);

    out.latent_h = jget<int>(cfg, "latent_h", out.latent_h);
    out.latent_w = jget<int>(cfg, "latent_w", out.latent_w);
    out.unet_depth = jget<int>(cfg, "unet_depth", out.unet_depth);
    out.unet_blocks_per_level = jget<int>(cfg, "unet_blocks_per_level", out.unet_blocks_per_level);
    out.unet_bottleneck_blocks = jget<int>(cfg, "unet_bottleneck_blocks", out.unet_bottleneck_blocks);

    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);

    out.ddpm_steps = jget<int>(cfg, "ddpm_steps", out.ddpm_steps);
    out.ddpm_beta_start = jget<float>(cfg, "ddpm_beta_start", out.ddpm_beta_start);
    out.ddpm_beta_end = jget<float>(cfg, "ddpm_beta_end", out.ddpm_beta_end);
    out.ddpm_steps_per_image = jget<int>(cfg, "ddpm_steps_per_image", out.ddpm_steps_per_image);

    out.recon_loss = jget<std::string>(cfg, "recon_loss", out.recon_loss);

    // KL regularization on eps_pred distribution (optional)
    out.kl_beta = jget<float>(cfg, "kl_beta", out.kl_beta);
    out.kl_warmup_steps = jget<int>(cfg, "kl_warmup_steps", out.kl_warmup_steps);
    out.logvar_clip_min = jget<float>(cfg, "logvar_clip_min", out.logvar_clip_min);
    out.logvar_clip_max = jget<float>(cfg, "logvar_clip_max", out.logvar_clip_max);

    out.loss_weighting = jget<std::string>(cfg, "loss_weighting", out.loss_weighting);
    out.min_snr_gamma = jget<float>(cfg, "min_snr_gamma", out.min_snr_gamma);

    out.output_activation = jget<std::string>(cfg, "output_activation", out.output_activation);

    out.peltier_noise = jget<bool>(cfg, "peltier_noise", out.peltier_noise);
    out.peltier_mix = jget<float>(cfg, "peltier_mix", out.peltier_mix);
    out.peltier_blur_radius = jget<int>(cfg, "peltier_blur_radius", out.peltier_blur_radius);

    out.vae_arch = jget<std::string>(cfg, "vae_arch", out.vae_arch);
    out.vae_checkpoint = jget<std::string>(cfg, "vae_checkpoint", out.vae_checkpoint);
    out.vae_scale = jget<float>(cfg, "vae_scale", out.vae_scale);
    out.vae_base_channels = jget<int>(cfg, "vae_base_channels", out.vae_base_channels);

    out.cfg_dropout_prob = jget<float>(cfg, "cfg_dropout_prob", out.cfg_dropout_prob);
    out.max_text_chars = jget<int>(cfg, "max_text_chars", out.max_text_chars);
    out.dropout = jget<float>(cfg, "dropout", out.dropout);

    // Captions structurées
    out.caption_structured_enable = jget<bool>(cfg, "caption_structured_enable", out.caption_structured_enable);
    out.caption_structured_canonicalize = jget<bool>(cfg, "caption_structured_canonicalize", out.caption_structured_canonicalize);
    out.caption_tags_dropout_prob = jget<float>(cfg, "caption_tags_dropout_prob", out.caption_tags_dropout_prob);
    out.caption_contexte_dropout_prob = jget<float>(cfg, "caption_contexte_dropout_prob", out.caption_contexte_dropout_prob);
    out.caption_mentalite_dropout_prob = jget<float>(cfg, "caption_mentalite_dropout_prob", out.caption_mentalite_dropout_prob);
    out.caption_texte_dropout_prob = jget<float>(cfg, "caption_texte_dropout_prob", out.caption_texte_dropout_prob);

    // Captions key/value (caption/theme/keywords/tokens)
    out.caption_kv_enable = jget<bool>(cfg, "caption_kv_enable", out.caption_kv_enable);

    // Majoration des termes fréquents
    out.term_freq_boost_enable = jget<bool>(cfg, "term_freq_boost_enable", out.term_freq_boost_enable);
    out.term_freq_boost_use_tokens = jget<bool>(cfg, "term_freq_boost_use_tokens", out.term_freq_boost_use_tokens);
    out.term_freq_boost_use_keywords = jget<bool>(cfg, "term_freq_boost_use_keywords", out.term_freq_boost_use_keywords);
    out.term_freq_boost_top_k = jget<int>(cfg, "term_freq_boost_top_k", out.term_freq_boost_top_k);
    out.term_freq_boost_repeat = jget<int>(cfg, "term_freq_boost_repeat", out.term_freq_boost_repeat);
    out.term_freq_boost_start_step = jget<int>(cfg, "term_freq_boost_start_step", out.term_freq_boost_start_step);
    out.term_freq_boost_update_every_steps = jget<int>(cfg, "term_freq_boost_update_every_steps", out.term_freq_boost_update_every_steps);

    out.viz_ddpm_every_steps = jget<int>(cfg, "viz_ddpm_every_steps", out.viz_ddpm_every_steps);
    out.viz_ddpm_num_steps = jget<int>(cfg, "viz_ddpm_num_steps", out.viz_ddpm_num_steps);

    // Optional image-space auxiliary loss (via VAE decoder)
    out.img_loss_weight = jget<float>(cfg, "img_loss_weight", out.img_loss_weight);
    out.img_loss_every_steps = jget<int>(cfg, "img_loss_every_steps", out.img_loss_every_steps);

    return out;
}

static json ponyxlDdpmDefaultConfigJson() {
    PonyXLDDPMModel::Config d;
    return json{
        {"seed", d.seed},
        {"d_model", d.d_model},
        {"max_vocab", d.max_vocab},
        {"text_ctx_len", d.text_ctx_len},
        {"text_clip_like", d.text_clip_like},
        {"global_ctx_tokens", d.global_ctx_tokens},
        {"text_bottleneck_meanpool", d.text_bottleneck_meanpool},

        {"latent_seq_len", d.latent_seq_len},
        {"latent_in_dim", d.latent_in_dim},
        {"num_heads", d.num_heads},

        {"sdxl_time_cond", d.sdxl_time_cond},
        {"timestep_cond", d.timestep_cond},
        {"unet_layers", d.unet_layers},
        {"text_layers", d.text_layers},
        {"mlp_hidden", d.mlp_hidden},

        {"latent_h", d.latent_h},
        {"latent_w", d.latent_w},
        {"unet_depth", d.unet_depth},
        {"unet_blocks_per_level", d.unet_blocks_per_level},
        {"unet_bottleneck_blocks", d.unet_bottleneck_blocks},

        {"image_w", d.image_w},
        {"image_h", d.image_h},
        {"image_c", d.image_c},

        {"ddpm_steps", d.ddpm_steps},
        {"ddpm_beta_start", d.ddpm_beta_start},
        {"ddpm_beta_end", d.ddpm_beta_end},
        {"ddpm_steps_per_image", d.ddpm_steps_per_image},

        {"recon_loss", d.recon_loss},

        // KL regularization on eps_pred distribution (optional)
        {"kl_beta", d.kl_beta},
        {"kl_warmup_steps", d.kl_warmup_steps},
        {"logvar_clip_min", d.logvar_clip_min},
        {"logvar_clip_max", d.logvar_clip_max},

        {"loss_weighting", d.loss_weighting},
        {"min_snr_gamma", d.min_snr_gamma},

        {"output_activation", d.output_activation},

        {"peltier_noise", d.peltier_noise},
        {"peltier_mix", d.peltier_mix},
        {"peltier_blur_radius", d.peltier_blur_radius},

        {"vae_arch", d.vae_arch},
        {"vae_checkpoint", d.vae_checkpoint},
        {"vae_scale", d.vae_scale},
        {"vae_base_channels", d.vae_base_channels},

        {"cfg_dropout_prob", d.cfg_dropout_prob},
        {"max_text_chars", d.max_text_chars},
        {"dropout", d.dropout},

        // Captions structurées
        {"caption_structured_enable", d.caption_structured_enable},
        {"caption_structured_canonicalize", d.caption_structured_canonicalize},
        {"caption_tags_dropout_prob", d.caption_tags_dropout_prob},
        {"caption_contexte_dropout_prob", d.caption_contexte_dropout_prob},
        {"caption_mentalite_dropout_prob", d.caption_mentalite_dropout_prob},
        {"caption_texte_dropout_prob", d.caption_texte_dropout_prob},

        // Captions key/value
        {"caption_kv_enable", d.caption_kv_enable},

        // Majoration des termes fréquents
        {"term_freq_boost_enable", d.term_freq_boost_enable},
        {"term_freq_boost_use_tokens", d.term_freq_boost_use_tokens},
        {"term_freq_boost_use_keywords", d.term_freq_boost_use_keywords},
        {"term_freq_boost_top_k", d.term_freq_boost_top_k},
        {"term_freq_boost_repeat", d.term_freq_boost_repeat},
        {"term_freq_boost_start_step", d.term_freq_boost_start_step},
        {"term_freq_boost_update_every_steps", d.term_freq_boost_update_every_steps},

        {"viz_ddpm_every_steps", d.viz_ddpm_every_steps},
        {"viz_ddpm_num_steps", d.viz_ddpm_num_steps},

        // Optional image-space auxiliary loss (via VAE decoder)
        {"img_loss_weight", d.img_loss_weight},
        {"img_loss_every_steps", d.img_loss_every_steps},
    };
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

    // Encoder normalization (optional)
    out.enc_norm = jget<std::string>(cfg, "enc_norm", out.enc_norm);
    out.enc_gn_groups = jget<int>(cfg, "enc_gn_groups", out.enc_gn_groups);

    // Compat: `use_attention` pilote désormais les blocs ResNet 2D injectés dans le décodeur.
    out.use_attention = jget<bool>(cfg, "use_attention", out.use_attention);

    // Nouvelle option: attention (SelfAttention) en plus des blocs ResNet.
    out.use_attn = jget<bool>(cfg, "use_attn", out.use_attn);
    out.attn_heads = jget<int>(cfg, "attn_heads", out.attn_heads);
    out.attn_max_tokens = jget<int>(cfg, "attn_max_tokens", out.attn_max_tokens);

    // Nouveau: gating séparé pour les blocs ResNet.
    // Compat: si absent, on reprend l'ancien comportement (gating couplé à attn_max_tokens).
    if (cfg.is_object() && cfg.contains("resnet_max_tokens")) {
        out.resnet_max_tokens = jget<int>(cfg, "resnet_max_tokens", out.resnet_max_tokens);
    } else {
        out.resnet_max_tokens = out.attn_max_tokens;
    }

    // Latent stochasticity (reparameterization noise)
    out.stochastic_latent = jget<bool>(cfg, "stochastic_latent", out.stochastic_latent);

    // Optional text conditioning
    out.text_cond = jget<bool>(cfg, "text_cond", out.text_cond);
    out.vocab_size = jget<int>(cfg, "vocab_size", out.vocab_size);
    out.seq_len = jget<int>(cfg, "seq_len", out.seq_len);
    out.text_d_model = jget<int>(cfg, "text_d_model", out.text_d_model);
    out.proj_dim = jget<int>(cfg, "proj_dim", out.proj_dim);

    // Perceptual loss (optional, consumed by Model::trainStepVAE)
    out.perceptual_weight = jget<float>(cfg, "perceptual_weight", out.perceptual_weight);
    out.perceptual_arch = jget<std::string>(cfg, "perceptual_arch", out.perceptual_arch);
    out.perceptual_checkpoint = jget<std::string>(cfg, "perceptual_checkpoint", out.perceptual_checkpoint);
    out.perceptual_base_channels = jget<int>(cfg, "perceptual_base_channels", out.perceptual_base_channels);
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

                // Encoder normalization (optional)
                {"enc_norm", d.enc_norm},
                {"enc_gn_groups", d.enc_gn_groups},

                {"use_attention", d.use_attention},
                {"resnet_max_tokens", d.resnet_max_tokens},
                {"use_attn", d.use_attn},
                {"attn_heads", d.attn_heads},
                {"attn_max_tokens", d.attn_max_tokens},
                {"stochastic_latent", d.stochastic_latent},
                {"text_cond", d.text_cond},
                {"vocab_size", d.vocab_size},
                {"seq_len", d.seq_len},
                {"text_d_model", d.text_d_model},
                {"proj_dim", d.proj_dim},

                // Perceptual loss (optional)
                {"perceptual_weight", d.perceptual_weight},
                {"perceptual_arch", d.perceptual_arch},
                {"perceptual_checkpoint", d.perceptual_checkpoint},
                {"perceptual_base_channels", d.perceptual_base_channels}};
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
    return out;
}

static json vgg16DefaultConfigJson() {
    VGG16Model::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}, {"num_classes", d.num_classes}, {"fc_hidden", d.fc_hidden}};
}

static VGG19Model::Config vgg19CfgFromJson(const json& cfg) {
    VGG19Model::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    out.num_classes = jget<int>(cfg, "num_classes", out.num_classes);
    out.fc_hidden = jget<int>(cfg, "fc_hidden", out.fc_hidden);
    return out;
}

static json vgg19DefaultConfigJson() {
    VGG19Model::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}, {"num_classes", d.num_classes}, {"fc_hidden", d.fc_hidden}};
}

static VGG16FeatModel::Config vgg16FeatCfgFromJson(const json& cfg) {
    VGG16FeatModel::Config out;
    out.image_w = jget<int>(cfg, "image_w", out.image_w);
    out.image_h = jget<int>(cfg, "image_h", out.image_h);
    out.image_c = jget<int>(cfg, "image_c", out.image_c);
    out.base_channels = jget<int>(cfg, "base_channels", out.base_channels);
    return out;
}

static json vgg16FeatDefaultConfigJson() {
    VGG16FeatModel::Config d;
    return json{{"image_w", d.image_w}, {"image_h", d.image_h}, {"image_c", d.image_c}, {"base_channels", d.base_channels}};
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
    model->modelConfig = cfg;
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
            "ponyxl_ddpm",
            Entry{
                "ponyxl_ddpm",
                "PonyXL SDXL-like DDPM latent diffusion (trainStepSdxlLatentDiffusion)",
                ponyxlDdpmDefaultConfigJson(),
                [](const json& cfg) -> std::shared_ptr<Model> {
                    auto m = std::make_shared<PonyXLDDPMModel>();
                    m->buildFromConfig(ponyxlDdpmCfgFromJson(cfg));
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
