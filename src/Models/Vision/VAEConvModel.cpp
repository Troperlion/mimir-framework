#include "VAEConvModel.hpp"

#include <algorithm>
#include <vector>
#include <stdexcept>
#include <cctype>

VAEConvModel::VAEConvModel() {
    setModelName("VAEConvModel");
    setHasEncoder(false);
}

void VAEConvModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

static inline void check_divisible(int a, int b, const std::string& msg) {
    if (b <= 0 || a <= 0 || (a % b) != 0) {
        throw std::runtime_error(msg);
    }
}

void VAEConvModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("VAEConvModel");
    model.modelConfig["type"] = "vae_conv";

    const int W = std::max(1, cfg.image_w);
    const int H = std::max(1, cfg.image_h);
    const int C = std::max(1, cfg.image_c);

    const int LH = std::max(1, cfg.latent_h);
    const int LW = std::max(1, cfg.latent_w);
    const int LC = std::max(1, cfg.latent_c);

    const bool text_cond = cfg.text_cond;
    const bool stochastic_latent = cfg.stochastic_latent;
    const int vocab = std::max(1, cfg.vocab_size);
    const int seq_len = std::max(1, cfg.seq_len);
    const int d_model = std::max(1, cfg.text_d_model);
    const int proj_dim = std::max(1, cfg.proj_dim);

    // Perceptual loss config (consumed by Model::trainStepVAE)
    const float perceptual_weight = std::max(0.0f, cfg.perceptual_weight);
    const std::string perceptual_arch = cfg.perceptual_arch.empty() ? std::string("vgg16_feat") : cfg.perceptual_arch;
    const std::string perceptual_checkpoint = cfg.perceptual_checkpoint;
    const int perceptual_base_channels = std::max(1, cfg.perceptual_base_channels);

    // Compat: `use_attention` pilote les blocs ResNet (ex-attention).
    const bool use_resnet = cfg.use_attention;
    // Nouveau: attention optionnelle en plus.
    const bool use_attn = cfg.use_attn;
    const int attn_heads_cfg = std::max(1, cfg.attn_heads);
    const int attn_max_tokens = std::max(0, cfg.attn_max_tokens);
    const int resnet_max_tokens = std::max(0, cfg.resnet_max_tokens);

    check_divisible(H, LH, "VAEConvModel: image_h must be divisible by latent_h");
    check_divisible(W, LW, "VAEConvModel: image_w must be divisible by latent_w");

    int down_h = H;
    int down_w = W;
    int downsamples = 0;
    while (down_h > LH && down_w > LW) {
        if ((down_h % 2) != 0 || (down_w % 2) != 0) break;
        down_h /= 2;
        down_w /= 2;
        ++downsamples;
    }
    if (down_h != LH || down_w != LW) {
        throw std::runtime_error("VAEConvModel: cannot reach latent_h/latent_w with /2 downsamples");
    }

    const int image_dim = W * H * C;
    const int latent_dim = LH * LW * LC;
    const int base = std::max(8, cfg.base_channels);

    // Encoder normalization options
    auto to_lower = [](std::string s) {
        for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return s;
    };
    const std::string enc_norm_cfg = to_lower(cfg.enc_norm);
    enum class EncNormKind { None, LayerNorm, GroupNorm };
    EncNormKind enc_norm_kind = EncNormKind::None;
    if (enc_norm_cfg == "layernorm" || enc_norm_cfg == "ln") enc_norm_kind = EncNormKind::LayerNorm;
    else if (enc_norm_cfg == "groupnorm" || enc_norm_cfg == "gn") enc_norm_kind = EncNormKind::GroupNorm;

    const int enc_gn_groups_cfg = std::max(1, cfg.enc_gn_groups);

    model.modelConfig["task"] = text_cond ? "vae_conv_text_autoencoder" : "vae_conv_autoencoder";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["image_dim"] = image_dim;
    model.modelConfig["latent_h"] = LH;
    model.modelConfig["latent_w"] = LW;
    model.modelConfig["latent_c"] = LC;
    model.modelConfig["latent_dim"] = latent_dim;
    model.modelConfig["base_channels"] = base;
    model.modelConfig["enc_norm"] = cfg.enc_norm;
    model.modelConfig["enc_gn_groups"] = enc_gn_groups_cfg;
    model.modelConfig["downsamples"] = downsamples;
    model.modelConfig["input_dim"] = image_dim;
    // Recon loss: avoid MSE for VAEConv (better behaved on images in [-1,1]).
    model.modelConfig["recon_loss"] = "l1";
    model.modelConfig["text_cond"] = text_cond;
    model.modelConfig["stochastic_latent"] = stochastic_latent;
    model.modelConfig["use_attention"] = use_resnet;
    model.modelConfig["resnet_max_tokens"] = resnet_max_tokens;
    model.modelConfig["use_attn"] = use_attn;
    model.modelConfig["attn_heads"] = attn_heads_cfg;
    model.modelConfig["attn_max_tokens"] = attn_max_tokens;

    // Optional perceptual loss
    model.modelConfig["perceptual_weight"] = perceptual_weight;
    model.modelConfig["perceptual_arch"] = perceptual_arch;
    model.modelConfig["perceptual_checkpoint"] = perceptual_checkpoint;
    model.modelConfig["perceptual_base_channels"] = perceptual_base_channels;
    if (text_cond) {
        model.modelConfig["vocab_size"] = vocab;
        model.modelConfig["seq_len"] = seq_len;
        model.modelConfig["text_d_model"] = d_model;
        model.modelConfig["proj_dim"] = proj_dim;
        model.modelConfig["output_dim"] = image_dim + 2 * latent_dim + 2 * proj_dim;
    } else {
        model.modelConfig["output_dim"] = image_dim + 2 * latent_dim;
    }

    auto sat_mul = [](size_t a, size_t b) -> size_t {
        if (a == 0 || b == 0) return 0;
        if (a > (static_cast<size_t>(-1) / b)) return static_cast<size_t>(-1);
        return a * b;
    };

    auto conv2d = [&](const std::string& name,
                      const std::string& in,
                      const std::string& out,
                      int in_c,
                      int out_c,
                      int in_h,
                      int in_w,
                      int k,
                      int s,
                      int p,
                      bool act) {
        model.push(name, "Conv2d",
                   sat_mul(static_cast<size_t>(out_c), sat_mul(static_cast<size_t>(in_c), sat_mul(static_cast<size_t>(k), static_cast<size_t>(k)))));
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {in};
            L->output = out;
            L->in_channels = in_c;
            L->out_channels = out_c;
            L->input_height = in_h;
            L->input_width = in_w;
            L->kernel_size = k;
            L->stride = s;
            L->padding = p;
            L->use_bias = false;
        }
        std::string y = out;
        if (act) {
            model.push(name + "/act", "SiLU", 0);
            if (auto* A = model.getLayerByName(name + "/act")) {
                A->inputs = {out};
                A->output = out + "_act";
            }
            y = out + "_act";
        }
        return y;
    };

    auto best_divisor_leq = [](int x, int max_d) {
        if (x <= 0) return 1;
        int d = std::min(std::max(1, max_d), x);
        for (; d >= 1; --d) {
            if ((x % d) == 0) return d;
        }
        return 1;
    };

    auto maybe_enc_norm2d_chw = [&](const std::string& name,
                                    const std::string& in_chw,
                                    int Cc,
                                    int Hh,
                                    int Ww) -> std::string {
        if (enc_norm_kind == EncNormKind::None) return in_chw;
        if (Cc <= 0 || Hh <= 0 || Ww <= 0) return in_chw;

        if (enc_norm_kind == EncNormKind::GroupNorm) {
            const int groups = best_divisor_leq(Cc, enc_gn_groups_cfg);
            model.push(name + "/gn", "GroupNorm", static_cast<size_t>(2) * static_cast<size_t>(Cc));
            if (auto* L = model.getLayerByName(name + "/gn")) {
                L->inputs = {in_chw};
                L->output = name + "/gn_out";
                L->num_groups = groups;
                L->in_channels = Cc;
                L->input_height = Hh;
                L->input_width = Ww;
                L->affine = true;
                L->use_bias = true;
            }

            // Ajout demandé: LayerNorm après GroupNorm.
            // CHW -> HWC -> LN(normalized=C) -> CHW
            model.push(name + "/to_hwc", "Permute", 0);
            if (auto* P = model.getLayerByName(name + "/to_hwc")) {
                P->inputs = {name + "/gn_out"};
                P->output = name + "/hwc";
                P->shape = {Cc, Hh, Ww};
                P->permute_dims = {1, 2, 0};
            }

            model.push(name + "/ln", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(Cc));
            if (auto* L = model.getLayerByName(name + "/ln")) {
                L->inputs = {name + "/hwc"};
                L->output = name + "/ln_hwc";
                L->in_features = Cc; // normalized_size
                L->affine = true;
                L->use_bias = true;
            }

            model.push(name + "/to_chw", "Permute", 0);
            if (auto* P = model.getLayerByName(name + "/to_chw")) {
                P->inputs = {name + "/ln_hwc"};
                P->output = name + "/out";
                P->shape = {Hh, Ww, Cc};
                P->permute_dims = {2, 0, 1};
            }
            return name + "/out";
        }

        // LayerNorm: CHW -> HWC (tokens contiguous) -> LN(normalized=C) -> CHW
        model.push(name + "/to_hwc", "Permute", 0);
        if (auto* P = model.getLayerByName(name + "/to_hwc")) {
            P->inputs = {in_chw};
            P->output = name + "/hwc";
            P->shape = {Cc, Hh, Ww};
            P->permute_dims = {1, 2, 0};
        }

        model.push(name + "/ln", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(Cc));
        if (auto* L = model.getLayerByName(name + "/ln")) {
            L->inputs = {name + "/hwc"};
            L->output = name + "/ln_hwc";
            L->in_features = Cc; // normalized_size
            L->affine = true;
            L->use_bias = true;
        }

        model.push(name + "/to_chw", "Permute", 0);
        if (auto* P = model.getLayerByName(name + "/to_chw")) {
            P->inputs = {name + "/ln_hwc"};
            P->output = name + "/out";
            P->shape = {Hh, Ww, Cc};
            P->permute_dims = {2, 0, 1};
        }
        return name + "/out";
    };

    auto silu = [&](const std::string& name, const std::string& in, const std::string& out) {
        model.push(name, "SiLU", 0);
        if (auto* A = model.getLayerByName(name)) {
            A->inputs = {in};
            A->output = out;
        }
    };

    auto enc_conv2d_norm_act = [&](const std::string& name,
                                   const std::string& in,
                                   const std::string& out,
                                   int in_c,
                                   int out_c,
                                   int in_h,
                                   int in_w,
                                   int k,
                                   int s,
                                   int p) -> std::string {
        // Conv
        (void)conv2d(name, in, out, in_c, out_c, in_h, in_w, k, s, p, false);
        // Norm (encoder only)
        const std::string n = maybe_enc_norm2d_chw(name + "/norm", out, out_c, (s == 2 ? std::max(1, (in_h + 2 * p - k) / s + 1) : in_h), (s == 2 ? std::max(1, (in_w + 2 * p - k) / s + 1) : in_w));
        // Act (keep historical layer name)
        silu(name + "/act", n, out + "_act");
        return out + "_act";
    };

    auto upsample2x = [&](const std::string& name,
                          const std::string& in,
                          const std::string& out,
                          int channels,
                          int in_h,
                          int in_w) {
        model.push(name, "UpsampleNearest", 0);
        if (auto* U = model.getLayerByName(name)) {
            U->inputs = {in};
            U->output = out;
            U->in_channels = channels;
            U->out_h = in_h;
            U->out_w = in_w;
            U->scale_h = 2.0f;
            U->scale_w = 2.0f;
        }
        return out;
    };

    // Attention spatiale 2D (CHW) via SelfAttention sur séquence H*W, embed_dim=C.
    // Implémentation: CHW -> HWC (tokens-major) -> SelfAttention -> HWC -> CHW + résiduel.
    auto maybe_self_attention2d_chw = [&](const std::string& name,
                                          const std::string& in_chw,
                                          int Cc,
                                          int Hh,
                                          int Ww) -> std::string {
        if (!use_attn) return in_chw;
        if (attn_max_tokens > 0 && (Hh * Ww) > attn_max_tokens) return in_chw;
        if (Cc <= 0 || Hh <= 0 || Ww <= 0) return in_chw;

        int heads = std::max(1, attn_heads_cfg);
        if ((Cc % heads) != 0) {
            int best = 1;
            for (int h = 1; h <= heads; ++h) {
                if ((Cc % h) == 0) best = h;
            }
            heads = best;
        }

        model.push(name + "/to_hwc", "Permute", 0);
        if (auto* P = model.getLayerByName(name + "/to_hwc")) {
            P->inputs = {in_chw};
            P->output = name + "/hwc";
            P->shape = {Cc, Hh, Ww};
            P->permute_dims = {1, 2, 0};
        }

        // SelfAttention params: (QKV + OUT) = (3*E*E + E*E) = 4*E*E
        model.push(name + "/attn", "SelfAttention",
                   sat_mul(static_cast<size_t>(4), sat_mul(static_cast<size_t>(Cc), static_cast<size_t>(Cc))));
        if (auto* A = model.getLayerByName(name + "/attn")) {
            A->inputs = {name + "/hwc"};
            A->output = name + "/attn_hwc";
            A->seq_len = Hh * Ww;
            A->embed_dim = Cc;
            A->num_heads = heads;
            A->causal = false;
        }

        model.push(name + "/to_chw", "Permute", 0);
        if (auto* P = model.getLayerByName(name + "/to_chw")) {
            P->inputs = {name + "/attn_hwc"};
            P->output = name + "/attn_chw";
            P->shape = {Hh, Ww, Cc};
            P->permute_dims = {2, 0, 1};
        }

        model.push(name + "/add", "Add", 0);
        if (auto* L = model.getLayerByName(name + "/add")) {
            L->inputs = {in_chw, name + "/attn_chw"};
            L->output = name + "/sum";
        }

        model.push(name + "/act", "SiLU", 0);
        if (auto* S = model.getLayerByName(name + "/act")) {
            S->inputs = {name + "/sum"};
            S->output = name + "/out";
        }

        return name + "/out";
    };

    // Remplace les anciens blocs d'attention par un bloc ResNet 2D (CHW):
    // conv3x3 -> SiLU -> conv3x3 + skip -> SiLU.
    // NOTE compat: historiquement le gating des blocs ResNet utilisait `attn_max_tokens`.
    // Désormais, `attn_max_tokens` ne concerne que SelfAttention, et `resnet_max_tokens` gate les blocs ResNet.
    auto maybe_resnet2d_chw = [&](const std::string& name,
                                  const std::string& in_chw,
                                  int Cc,
                                  int Hh,
                                  int Ww) -> std::string {
        if (!use_resnet) return in_chw;
        if (resnet_max_tokens > 0 && (Hh * Ww) > resnet_max_tokens) return in_chw;
        if (Cc <= 0 || Hh <= 0 || Ww <= 0) return in_chw;

        const std::string y1 = conv2d(name + "/conv1", in_chw, name + "/c1", Cc, Cc, Hh, Ww, 3, 1, 1, true);
        const std::string y2 = conv2d(name + "/conv2", y1, name + "/c2", Cc, Cc, Hh, Ww, 3, 1, 1, false);

        model.push(name + "/add", "Add", 0);
        if (auto* L = model.getLayerByName(name + "/add")) {
            L->inputs = {in_chw, y2};
            L->output = name + "/sum";
        }

        model.push(name + "/act", "SiLU", 0);
        if (auto* A = model.getLayerByName(name + "/act")) {
            A->inputs = {name + "/sum"};
            A->output = name + "/out";
        }

        return name + "/out";
    };

    // ===== Optional text encoder (ids -> pooled embedding) =====
    // Input ids are expected under IntTensorStore key "text_ids" when text_cond=true.
    // Outputs:
    // - vae_conv/text/pooled (d_model)
    // - vae_conv/text/cond (LC) used to bias z spatial channels
    // - vae_conv/text/proj (proj_dim) used for alignment loss
    std::string text_pooled = "";
    if (text_cond) {
        model.push("vae_conv/text/tok_embed", "Embedding", static_cast<size_t>(vocab) * static_cast<size_t>(d_model));
        if (auto* L = model.getLayerByName("vae_conv/text/tok_embed")) {
            L->inputs = {"text_ids"};
            L->output = "vae_conv/text/tok";
            L->vocab_size = vocab;
            L->embed_dim = d_model;
            L->padding_idx = 0;
        }

        model.push("vae_conv/text/pool", "TokenMeanPool", 0);
        if (auto* L = model.getLayerByName("vae_conv/text/pool")) {
            L->inputs = {"vae_conv/text/tok"};
            L->output = "vae_conv/text/pooled";
            L->seq_len = seq_len;
            L->embed_dim = d_model;
        }
        text_pooled = "vae_conv/text/pooled";

        // Text -> conditioning vector for latent channels
        model.push("vae_conv/text/cond_fc", "Linear",
                   sat_mul(static_cast<size_t>(d_model), static_cast<size_t>(LC)) + static_cast<size_t>(LC));
        if (auto* L = model.getLayerByName("vae_conv/text/cond_fc")) {
            L->inputs = {text_pooled};
            L->output = "vae_conv/text/cond_pre";
            L->in_features = d_model;
            L->out_features = LC;
            L->use_bias = true;
        }
        model.push("vae_conv/text/cond_act", "Tanh", 0);
        if (auto* L = model.getLayerByName("vae_conv/text/cond_act")) {
            L->inputs = {"vae_conv/text/cond_pre"};
            L->output = "vae_conv/text/cond";
        }

        // Text -> projection head (for align loss)
        model.push("vae_conv/text/proj", "Linear",
                   sat_mul(static_cast<size_t>(d_model), static_cast<size_t>(proj_dim)) + static_cast<size_t>(proj_dim));
        if (auto* L = model.getLayerByName("vae_conv/text/proj")) {
            L->inputs = {text_pooled};
            L->output = "vae_conv/text/proj_vec";
            L->in_features = d_model;
            L->out_features = proj_dim;
            L->use_bias = true;
        }
    }

    // Input vector -> HWC -> CHW
    model.push("vae_conv/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("vae_conv/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "vae_conv/in_vec";
    }

    model.push("vae_conv/in_reshape", "Reshape", 0);
    if (auto* R = model.getLayerByName("vae_conv/in_reshape")) {
        R->inputs = {"vae_conv/in_vec"};
        R->output = "vae_conv/in_hwc";
        R->target_shape = {H, W, C};
    }

    model.push("vae_conv/in_to_chw", "Permute", 0);
    if (auto* P = model.getLayerByName("vae_conv/in_to_chw")) {
        P->inputs = {"vae_conv/in_hwc"};
        P->output = "vae_conv/in_chw";
        P->shape = {H, W, C};
        P->permute_dims = {2, 0, 1};
    }

    // Encoder
    std::string x = "vae_conv/in_chw";
    int cur_h = H;
    int cur_w = W;
    int cur_c = C;

    int ch = base;
    x = enc_conv2d_norm_act("vae_conv/enc/conv_in", x, "vae_conv/enc/c0", cur_c, ch, cur_h, cur_w, 3, 1, 1);
    cur_c = ch;

    // Renforce le pouvoir convolutionnel à résolution constante (sans changer les shapes
    // ni les noms historiques des layers existants).
    x = enc_conv2d_norm_act("vae_conv/enc/conv_in2", x, "vae_conv/enc/c0b", cur_c, cur_c, cur_h, cur_w, 3, 1, 1);

    for (int i = 0; i < downsamples; ++i) {
        const std::string b = "vae_conv/enc/down" + std::to_string(i + 1);
        x = enc_conv2d_norm_act(b + "/conv", x, b + "/y", cur_c, cur_c, cur_h, cur_w, 3, 2, 1);
        cur_h = std::max(1, (cur_h + 2 * 1 - 3) / 2 + 1);
        cur_w = std::max(1, (cur_w + 2 * 1 - 3) / 2 + 1);

        // Deuxième conv 3x3 à cette échelle (plus de non-linéarités / mélange spatial)
        x = enc_conv2d_norm_act(b + "/conv2", x, b + "/y2", cur_c, cur_c, cur_h, cur_w, 3, 1, 1);
    }

    // Project to mu/logvar at latent resolution
    x = enc_conv2d_norm_act("vae_conv/enc/proj", x, "vae_conv/enc/h", cur_c, cur_c, cur_h, cur_w, 3, 1, 1);
    model.push("vae_conv/enc/mu", "Conv2d",
               sat_mul(static_cast<size_t>(LC), sat_mul(static_cast<size_t>(cur_c), sat_mul(static_cast<size_t>(1), static_cast<size_t>(1)))));
    if (auto* L = model.getLayerByName("vae_conv/enc/mu")) {
        L->inputs = {"vae_conv/enc/h_act"};
        L->output = "vae_conv/mu";
        L->in_channels = cur_c;
        L->out_channels = LC;
        L->input_height = LH;
        L->input_width = LW;
        L->kernel_size = 1;
        L->stride = 1;
        L->padding = 0;
        L->use_bias = false;
    }

    model.push("vae_conv/enc/logvar", "Conv2d",
               sat_mul(static_cast<size_t>(LC), sat_mul(static_cast<size_t>(cur_c), sat_mul(static_cast<size_t>(1), static_cast<size_t>(1)))));
    if (auto* L = model.getLayerByName("vae_conv/enc/logvar")) {
        L->inputs = {"vae_conv/enc/h_act"};
        L->output = "vae_conv/logvar";
        L->in_channels = cur_c;
        L->out_channels = LC;
        L->input_height = LH;
        L->input_width = LW;
        L->kernel_size = 1;
        L->stride = 1;
        L->padding = 0;
        L->use_bias = false;
    }

    // Reparameterize (spatial latent CHW)
    // NOTE: le layer Reparameterize respecte modelConfig["stochastic_latent"].
    model.push("vae_conv/reparam", "Reparameterize", 0);
    if (auto* L = model.getLayerByName("vae_conv/reparam")) {
        L->inputs = {"vae_conv/mu", "vae_conv/logvar"};
        L->output = "vae_conv/z";
    }

    // Image projection head (for align loss): mu -> proj_dim
    if (text_cond) {
        model.push("vae_conv/img/proj", "Linear",
                   sat_mul(static_cast<size_t>(latent_dim), static_cast<size_t>(proj_dim)) + static_cast<size_t>(proj_dim));
        if (auto* L = model.getLayerByName("vae_conv/img/proj")) {
            L->inputs = {"vae_conv/mu"};
            L->output = "vae_conv/img/proj_vec";
            L->in_features = latent_dim;
            L->out_features = proj_dim;
            L->use_bias = true;
        }
    }

    // Optional conditioning: bias z channels with text-derived vector (broadcast over spatial).
    std::string z_in = "vae_conv/z";
    if (text_cond) {
        model.push("vae_conv/z_add_text", "Add", 0);
        if (auto* L = model.getLayerByName("vae_conv/z_add_text")) {
            L->inputs = {"vae_conv/z", "vae_conv/text/cond"};
            L->output = "vae_conv/z_cond";
        }
        z_in = "vae_conv/z_cond";
    }

    std::string y = z_in;
    int dy_h = LH;
    int dy_w = LW;
    int dy_c = LC;

    // Bring to base channels
    y = conv2d("vae_conv/dec/conv_in", y, "vae_conv/dec/c0", dy_c, base, dy_h, dy_w, 3, 1, 1, true);
    dy_c = base;

    // Conv additionnelle au bottleneck (même résolution)
    y = conv2d("vae_conv/dec/conv_in2", y, "vae_conv/dec/c0b", dy_c, dy_c, dy_h, dy_w, 3, 1, 1, true);

    // Ex-attention au bottleneck -> ResNet
    y = maybe_resnet2d_chw("vae_conv/res/latent", y, dy_c, dy_h, dy_w);
    // Attention en plus (optionnelle)
    y = maybe_self_attention2d_chw("vae_conv/attn/latent", y, dy_c, dy_h, dy_w);

    for (int i = downsamples - 1; i >= 0; --i) {
        const std::string b = "vae_conv/dec/up" + std::to_string(i + 1);
        const int in_h = dy_h;
        const int in_w = dy_w;
        y = upsample2x(b + "/up", y, b + "/up_y", dy_c, in_h, in_w);
        dy_h = in_h * 2;
        dy_w = in_w * 2;
        y = conv2d(b + "/conv", y, b + "/c", dy_c, dy_c, dy_h, dy_w, 3, 1, 1, true);

        // Deuxième conv 3x3 à cette échelle (plus de capacité, sans changer les shapes)
        y = conv2d(b + "/conv2", y, b + "/c2", dy_c, dy_c, dy_h, dy_w, 3, 1, 1, true);

        // Ex-self-attn à cette échelle -> ResNet
        y = maybe_resnet2d_chw(b + "/res", y, dy_c, dy_h, dy_w);

        // Attention en plus (optionnelle)
        y = maybe_self_attention2d_chw(b + "/attn", y, dy_c, dy_h, dy_w);
    }

    // Final to RGB
    y = conv2d("vae_conv/dec/out", y, "vae_conv/dec/out_pre", dy_c, C, dy_h, dy_w, 3, 1, 1, false);

    model.push("vae_conv/dec/tanh", "Tanh", 0);
    if (auto* T = model.getLayerByName("vae_conv/dec/tanh")) {
        T->inputs = {"vae_conv/dec/out_pre"};
        T->output = "vae_conv/recon_chw";
    }

    // CHW -> HWC (image vector)
    model.push("vae_conv/recon_to_hwc", "Permute", 0);
    if (auto* P = model.getLayerByName("vae_conv/recon_to_hwc")) {
        P->inputs = {"vae_conv/recon_chw"};
        P->output = "vae_conv/recon";
        P->shape = {C, H, W};
        P->permute_dims = {1, 2, 0};
    }

    // Pack output recon || mu || logvar
    model.push("vae_conv/out_concat", "Concat", 0);
    if (auto* L = model.getLayerByName("vae_conv/out_concat")) {
        if (text_cond) {
            L->inputs = {"vae_conv/recon", "vae_conv/mu", "vae_conv/logvar", "vae_conv/img/proj_vec", "vae_conv/text/proj_vec"};
        } else {
            L->inputs = {"vae_conv/recon", "vae_conv/mu", "vae_conv/logvar"};
        }
        L->output = "x";
        L->concat_axis = 0;
    }
}

void VAEConvModel::buildDecoderInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("VAEConvModel");
    model.modelConfig["type"] = "vae_conv_decode";

    const int W = std::max(1, cfg.image_w);
    const int H = std::max(1, cfg.image_h);
    const int C = std::max(1, cfg.image_c);

    const int LH = std::max(1, cfg.latent_h);
    const int LW = std::max(1, cfg.latent_w);
    const int LC = std::max(1, cfg.latent_c);

    const bool use_resnet = cfg.use_attention;
    const bool use_attn = cfg.use_attn;
    const int attn_max_tokens = std::max(0, cfg.attn_max_tokens);
    const int attn_heads_cfg = std::max(1, cfg.attn_heads);
    const int resnet_max_tokens = std::max(0, cfg.resnet_max_tokens);

    check_divisible(H, LH, "VAEConvModel(decode): image_h must be divisible by latent_h");
    check_divisible(W, LW, "VAEConvModel(decode): image_w must be divisible by latent_w");

    int down_h = H;
    int down_w = W;
    int downsamples = 0;
    while (down_h > LH && down_w > LW) {
        if ((down_h % 2) != 0 || (down_w % 2) != 0) break;
        down_h /= 2;
        down_w /= 2;
        ++downsamples;
    }
    if (down_h != LH || down_w != LW) {
        throw std::runtime_error("VAEConvModel(decode): cannot reach latent_h/latent_w with /2 downsamples");
    }

    const int image_dim = W * H * C;
    const int latent_dim = LH * LW * LC;
    const int base = std::max(8, cfg.base_channels);

    model.modelConfig["task"] = "vae_conv_decoder";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["image_dim"] = image_dim;
    model.modelConfig["latent_h"] = LH;
    model.modelConfig["latent_w"] = LW;
    model.modelConfig["latent_c"] = LC;
    model.modelConfig["latent_dim"] = latent_dim;
    model.modelConfig["base_channels"] = base;
    model.modelConfig["downsamples"] = downsamples;
    model.modelConfig["input_dim"] = latent_dim;
    model.modelConfig["output_dim"] = image_dim;
    model.modelConfig["use_attention"] = use_resnet;
    model.modelConfig["resnet_max_tokens"] = resnet_max_tokens;
    model.modelConfig["use_attn"] = use_attn;
    model.modelConfig["attn_heads"] = attn_heads_cfg;
    model.modelConfig["attn_max_tokens"] = attn_max_tokens;

    auto sat_mul = [](size_t a, size_t b) -> size_t {
        if (a == 0 || b == 0) return 0;
        if (a > (static_cast<size_t>(-1) / b)) return static_cast<size_t>(-1);
        return a * b;
    };

    auto conv2d = [&](const std::string& name,
                      const std::string& in,
                      const std::string& out,
                      int in_c,
                      int out_c,
                      int in_h,
                      int in_w,
                      int k,
                      int s,
                      int p,
                      bool act) {
        model.push(name, "Conv2d",
                   sat_mul(static_cast<size_t>(out_c), sat_mul(static_cast<size_t>(in_c), sat_mul(static_cast<size_t>(k), static_cast<size_t>(k)))));
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {in};
            L->output = out;
            L->in_channels = in_c;
            L->out_channels = out_c;
            L->input_height = in_h;
            L->input_width = in_w;
            L->kernel_size = k;
            L->stride = s;
            L->padding = p;
            L->use_bias = false;
        }
        std::string y = out;
        if (act) {
            model.push(name + "/act", "SiLU", 0);
            if (auto* A = model.getLayerByName(name + "/act")) {
                A->inputs = {out};
                A->output = out + "_act";
            }
            y = out + "_act";
        }
        return y;
    };

    auto upsample2x = [&](const std::string& name,
                          const std::string& in,
                          const std::string& out,
                          int channels,
                          int in_h,
                          int in_w) {
        model.push(name, "UpsampleNearest", 0);
        if (auto* U = model.getLayerByName(name)) {
            U->inputs = {in};
            U->output = out;
            U->in_channels = channels;
            U->out_h = in_h;
            U->out_w = in_w;
            U->scale_h = 2.0f;
            U->scale_w = 2.0f;
        }
        return out;
    };

    auto maybe_self_attention2d_chw = [&](const std::string& name,
                                          const std::string& in_chw,
                                          int Cc,
                                          int Hh,
                                          int Ww) -> std::string {
        if (!use_attn) return in_chw;
        if (attn_max_tokens > 0 && (Hh * Ww) > attn_max_tokens) return in_chw;
        if (Cc <= 0 || Hh <= 0 || Ww <= 0) return in_chw;

        int heads = std::max(1, attn_heads_cfg);
        if ((Cc % heads) != 0) {
            int best = 1;
            for (int h = 1; h <= heads; ++h) {
                if ((Cc % h) == 0) best = h;
            }
            heads = best;
        }

        model.push(name + "/to_hwc", "Permute", 0);
        if (auto* P = model.getLayerByName(name + "/to_hwc")) {
            P->inputs = {in_chw};
            P->output = name + "/hwc";
            P->shape = {Cc, Hh, Ww};
            P->permute_dims = {1, 2, 0};
        }

        model.push(name + "/attn", "SelfAttention",
                   sat_mul(static_cast<size_t>(4), sat_mul(static_cast<size_t>(Cc), static_cast<size_t>(Cc))));
        if (auto* A = model.getLayerByName(name + "/attn")) {
            A->inputs = {name + "/hwc"};
            A->output = name + "/attn_hwc";
            A->seq_len = Hh * Ww;
            A->embed_dim = Cc;
            A->num_heads = heads;
            A->causal = false;
        }

        model.push(name + "/to_chw", "Permute", 0);
        if (auto* P = model.getLayerByName(name + "/to_chw")) {
            P->inputs = {name + "/attn_hwc"};
            P->output = name + "/attn_chw";
            P->shape = {Hh, Ww, Cc};
            P->permute_dims = {2, 0, 1};
        }

        model.push(name + "/add", "Add", 0);
        if (auto* L = model.getLayerByName(name + "/add")) {
            L->inputs = {in_chw, name + "/attn_chw"};
            L->output = name + "/sum";
        }

        model.push(name + "/act", "SiLU", 0);
        if (auto* S = model.getLayerByName(name + "/act")) {
            S->inputs = {name + "/sum"};
            S->output = name + "/out";
        }

        return name + "/out";
    };

    // Même remplacement que dans buildInto(): ex-attention -> bloc ResNet 2D (CHW)
    // avec gating dédié (resnet_max_tokens).
    auto maybe_resnet2d_chw = [&](const std::string& name,
                                  const std::string& in_chw,
                                  int Cc,
                                  int Hh,
                                  int Ww) -> std::string {
        if (!use_resnet) return in_chw;
        if (resnet_max_tokens > 0 && (Hh * Ww) > resnet_max_tokens) return in_chw;
        if (Cc <= 0 || Hh <= 0 || Ww <= 0) return in_chw;

        const std::string y1 = conv2d(name + "/conv1", in_chw, name + "/c1", Cc, Cc, Hh, Ww, 3, 1, 1, true);
        const std::string y2 = conv2d(name + "/conv2", y1, name + "/c2", Cc, Cc, Hh, Ww, 3, 1, 1, false);

        model.push(name + "/add", "Add", 0);
        if (auto* L = model.getLayerByName(name + "/add")) {
            L->inputs = {in_chw, y2};
            L->output = name + "/sum";
        }

        model.push(name + "/act", "SiLU", 0);
        if (auto* A = model.getLayerByName(name + "/act")) {
            A->inputs = {name + "/sum"};
            A->output = name + "/out";
        }

        return name + "/out";
    };

    // Input latent vector -> vae_conv/z
    model.push("vae_conv/raw_z", "Identity", 0);
    if (auto* L = model.getLayerByName("vae_conv/raw_z")) {
        L->inputs = {"__input__"};
        L->output = "vae_conv/z";
    }

    std::string y = "vae_conv/z";
    int dy_h = LH;
    int dy_w = LW;
    int dy_c = LC;

    y = conv2d("vae_conv/dec/conv_in", y, "vae_conv/dec/c0", dy_c, base, dy_h, dy_w, 3, 1, 1, true);
    dy_c = base;

    // Conv additionnelle au bottleneck (même résolution)
    y = conv2d("vae_conv/dec/conv_in2", y, "vae_conv/dec/c0b", dy_c, dy_c, dy_h, dy_w, 3, 1, 1, true);

    // Ex-attention au bottleneck -> ResNet
    y = maybe_resnet2d_chw("vae_conv/res/latent", y, dy_c, dy_h, dy_w);
    // Attention en plus (optionnelle)
    y = maybe_self_attention2d_chw("vae_conv/attn/latent", y, dy_c, dy_h, dy_w);

    for (int i = downsamples - 1; i >= 0; --i) {
        const std::string b = "vae_conv/dec/up" + std::to_string(i + 1);
        const int in_h = dy_h;
        const int in_w = dy_w;
        y = upsample2x(b + "/up", y, b + "/up_y", dy_c, in_h, in_w);
        dy_h = in_h * 2;
        dy_w = in_w * 2;
        y = conv2d(b + "/conv", y, b + "/c", dy_c, dy_c, dy_h, dy_w, 3, 1, 1, true);

        // Deuxième conv 3x3 à cette échelle
        y = conv2d(b + "/conv2", y, b + "/c2", dy_c, dy_c, dy_h, dy_w, 3, 1, 1, true);

        // Ex-self-attn à cette échelle -> ResNet
        y = maybe_resnet2d_chw(b + "/res", y, dy_c, dy_h, dy_w);

        // Attention en plus (optionnelle)
        y = maybe_self_attention2d_chw(b + "/attn", y, dy_c, dy_h, dy_w);
    }

    y = conv2d("vae_conv/dec/out", y, "vae_conv/dec/out_pre", dy_c, C, dy_h, dy_w, 3, 1, 1, false);

    model.push("vae_conv/dec/tanh", "Tanh", 0);
    if (auto* T = model.getLayerByName("vae_conv/dec/tanh")) {
        T->inputs = {"vae_conv/dec/out_pre"};
        T->output = "vae_conv/recon_chw";
    }

    model.push("vae_conv/recon_to_hwc", "Permute", 0);
    if (auto* P = model.getLayerByName("vae_conv/recon_to_hwc")) {
        P->inputs = {"vae_conv/recon_chw"};
        P->output = "x";
        P->shape = {C, H, W};
        P->permute_dims = {1, 2, 0};
    }
}
