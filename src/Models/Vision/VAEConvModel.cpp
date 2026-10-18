#include "VAEConvModel.hpp"
#include "training/BBoxSupervision.hpp"

#include <algorithm>
#include <stdexcept>

namespace {
std::string latentConv(Model& model, const std::string& name, const std::string& input,
                       int in_c, int out_c, int h, int w, int kernel = 1, int dilation = 1,
                       bool activate = false) {
    model.push(name, "Conv2d", static_cast<size_t>(in_c) * out_c * kernel * kernel);
    auto* layer = model.getLayerByName(name);
    layer->inputs = {input}; layer->output = name + "/out";
    layer->in_channels = in_c; layer->out_channels = out_c;
    layer->input_height = h; layer->input_width = w;
    layer->kernel_size = kernel; layer->stride = 1;
    layer->dilation = layer->dilation_h = layer->dilation_w = dilation;
    layer->padding = dilation * (kernel / 2);
    layer->use_bias = false;
    if (!activate) return layer->output;
    model.push(name + "/act", "SiLU", 0);
    auto* act = model.getLayerByName(name + "/act");
    act->inputs = {name + "/out"}; act->output = name + "/activated";
    return act->output;
}

std::string latentContext(Model& model, const std::string& prefix, const std::string& input,
                          int in_c, int channels, int h, int w) {
    const auto local = latentConv(model, prefix + "/project", input, in_c, channels, h, w);
    const int branch_channels = std::max(2, channels / 4);
    std::vector<std::string> branches;
    // A dense local branch prevents the gaps of dilated sampling; the other
    // branches expand context without shrinking the spatial latent.
    for (int dilation : {1, 2, 4})
        branches.push_back(latentConv(model, prefix + "/scale" + std::to_string(dilation),
            local, channels, branch_channels, h, w, 3, dilation, true));
    model.push(prefix + "/concat", "Concat", 0);
    auto* concat = model.getLayerByName(prefix + "/concat");
    concat->inputs = branches; concat->output = prefix + "/scales"; concat->concat_axis = 0;
    const auto mixed = latentConv(model, prefix + "/fuse", prefix + "/scales",
        3 * branch_channels, channels, h, w);
    model.push(prefix + "/add", "Add", 0);
    auto* add = model.getLayerByName(prefix + "/add");
    add->inputs = {local, mixed}; add->output = prefix + "/out";
    return add->output;
}

std::string phaseRearrange(Model& model, const std::string& prefix, const std::string& input,
                          int c, int h, int w, int lh, int lw, bool unpack) {
    const int rh = h / lh, rw = w / lw;
    model.push(prefix + "/reshape", "Reshape", 0);
    auto* reshape = model.getLayerByName(prefix + "/reshape");
    reshape->inputs = {input}; reshape->output = prefix + "/shaped";
    reshape->target_shape = unpack ? std::vector<int>{c, rh, rw, lh, lw}
                                   : std::vector<int>{c, lh, rh, lw, rw};
    model.push(prefix + "/permute", "Permute", 0);
    auto* permute = model.getLayerByName(prefix + "/permute");
    permute->inputs = {prefix + "/shaped"}; permute->output = prefix + "/permuted";
    permute->shape = unpack ? std::vector<int>{c, rh, rw, lh, lw}
                            : std::vector<int>{c, lh, rh, lw, rw};
    permute->permute_dims = unpack ? std::vector<int>{0, 3, 1, 4, 2}
                                  : std::vector<int>{0, 2, 4, 1, 3};
    model.push(prefix + "/view", "Reshape", 0);
    auto* view = model.getLayerByName(prefix + "/view");
    view->inputs = {prefix + "/permuted"}; view->output = prefix + "/out";
    view->target_shape = unpack ? std::vector<int>{c, h, w}
                               : std::vector<int>{c * rh * rw, lh, lw};
    return view->output;
}

std::string detailReconstruction(Model& model, const std::string& input,
                                 const std::string& coarse, int in_c,
                                 int c, int h, int w, int lh, int lw) {
    const auto phases = latentConv(model, "vae_conv/dec/detail/project", input, in_c,
        c * (h / lh) * (w / lw), lh, lw);
    const auto detail = phaseRearrange(model, "vae_conv/dec/detail/unpack", phases,
        c, h, w, lh, lw, true);
    model.push("vae_conv/dec/detail/add", "Add", 0);
    auto* add = model.getLayerByName("vae_conv/dec/detail/add");
    add->inputs = {coarse, detail}; add->output = "vae_conv/dec/detail/refined";
    return add->output;
}

void configureLatent(Model& model, const VAEConvModel::Config& cfg) {
    if ((cfg.latent_spatial_context || cfg.latent_detail_path) && cfg.latent_context_channels < 2)
        throw std::runtime_error("VAEConvModel: latent_context_channels must be >= 2");
    model.modelConfig["latent_spatial_context"] = cfg.latent_spatial_context;
    model.modelConfig["latent_context_channels"] = cfg.latent_context_channels;
    model.modelConfig["latent_detail_path"] = cfg.latent_detail_path;
}
} // namespace

VAEConvModel::VAEConvModel() {
    setModelName("VAEConvModel");
    setHasEncoder(false);
}

bool VAEConvModel::InitVizTips() {
    clearVizTipsRegistry();
    clearVizTaps();

    registerVizTip("vae_conv/raw_in", "Dataset/raw");
    registerVizTip("vae_conv/in_reshape", "Preprocess/reshape");
    registerVizTip("vae_conv/in_to_chw", "Preprocess/chw");
    registerVizTip("vae_conv/enc/conv_in", "Encoder/in");
    registerVizTip("vae_conv/enc/proj", "Encoder/proj");
    registerVizTip("vae_conv/enc/mu", "Latent/mu");
    registerVizTip("vae_conv/enc/logvar_pre_conv", "Latent/logvar/features");
    registerVizTip("vae_conv/enc/logvar_pre_norm", "Latent/logvar/norm");
    registerVizTip("vae_conv/enc/logvar", "Latent/logvar");
    registerVizTip("vae_conv/reparam", "Latent/sample");
    registerVizTip("vae_conv/z_prior_add", "Latent/prior_add");
    registerVizTip("vae_conv/dec/conv_in", "Decoder/in");
    registerVizTip("vae_conv/dec/tanh", "Output/tanh");
    registerVizTip("vae_conv/recon_to_hwc", "Output/recon_hwc");
    registerVizTip("vae_conv/out_concat", "Output/pack");
    registerVizTip("vae_conv/img_proj", "Txt/img_proj");
    registerVizTip("vae_conv/txt/tok_emb", "Txt/tok_emb");
    registerVizTip("vae_conv/txt/pool", "Txt/pool");
    registerVizTip("vae_conv/txt_proj", "Txt/txt_proj");

    return true;
}

bool VAEConvModel::UpdateVizTips(const Layer& layer, VizFrame& frame) {

    if (Model::UpdateVizTips(layer, frame)) return true;

    if (layer.name.empty()) return false;

    auto attach_tip = [&](const std::string& tip) {
        if (tip.empty()) return false;
        if (frame.label.empty()) frame.label = tip;
        else frame.label += " | " + tip;
        return true;
    };

    const std::string& name = layer.name;
    if (name.find("/enc/conv_in") != std::string::npos) return attach_tip("Encoder/in");
    if (name.find("/enc/proj") != std::string::npos) return attach_tip("Encoder/proj");
    if (name.find("/enc/logvar_pre_conv") != std::string::npos) return attach_tip("Latent/logvar/features");
    if (name.find("/enc/logvar_pre_norm") != std::string::npos) return attach_tip("Latent/logvar/norm");
    if (name.find("/enc/down") != std::string::npos) {
        if (name.find("/conv") != std::string::npos) return attach_tip("Encoder/down/conv");
        if (name.find("/res") != std::string::npos) return attach_tip("Encoder/down/res");
        if (name.find("/attn") != std::string::npos) return attach_tip("Encoder/down/attn");
        return attach_tip("Encoder/down");
    }
    if (name.find("/enc/bot") != std::string::npos) {
        if (name.find("/attn") != std::string::npos) return attach_tip("Encoder/bottleneck/attn");
        if (name.find("/res") != std::string::npos) return attach_tip("Encoder/bottleneck/res");
        return attach_tip("Encoder/bottleneck");
    }
    if (name.find("/dec/conv_in") != std::string::npos) return attach_tip("Decoder/in");
    if (name.find("/dec/up") != std::string::npos) {
        if (name.find("/skip_cat") != std::string::npos) return attach_tip("Decoder/up/skip_concat");
        if (name.find("/skip_proj") != std::string::npos) return attach_tip("Decoder/up/skip_projection");
        if (name.find("/conv") != std::string::npos) return attach_tip("Decoder/up/conv");
        if (name.find("/res") != std::string::npos) return attach_tip("Decoder/up/res");
        if (name.find("/attn") != std::string::npos) return attach_tip("Decoder/up/attn");
        if (name.size() >= 3 && name.compare(name.size() - 3, 3, "/up") == 0) return attach_tip("Decoder/up/sample");
        return attach_tip("Decoder/up");
    }
    if (name.find("/dec/bot") != std::string::npos) {
        if (name.find("/attn") != std::string::npos) return attach_tip("Decoder/bottleneck/attn");
        if (name.find("/res") != std::string::npos) return attach_tip("Decoder/bottleneck/res");
        return attach_tip("Decoder/bottleneck");
    }
    if (name.find("/skip") != std::string::npos) {
        return attach_tip("Skip/fusion");
    }

    if (name.find("/txt/") != std::string::npos ||
        name.find("/txt_") != std::string::npos ||
        name.find("/img_proj") != std::string::npos) {
        if (name.find("/tok_emb") != std::string::npos) return attach_tip("Txt/tok_emb");
        if (name.find("/pool") != std::string::npos) return attach_tip("Txt/pool");
        if (name.find("/img_proj") != std::string::npos) return attach_tip("Txt/img_proj");
        if (name.find("/txt_proj") != std::string::npos) return attach_tip("Txt/txt_proj");
        return attach_tip("Txt");
    }

    return false;
}

void VAEConvModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
    // ConditioningEncoder externe : embeddings mag/mod/seq appris de dimension d_model.
    // dim=d_model << latent_dim => ensureVocabSize raisonnable (~200 Mo max).
    if (cfg_.d_model > 0) {
        setHasEncoder(true);
        getMutableEncoder().ensureDim(cfg_.d_model);
        getMutableEncoder().ensureSpecialEmbeddings();
    }
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

    const bool stochastic_latent = cfg.stochastic_latent;
    const bool text_cond = cfg.text_cond;
    const int vocab_size = std::max(2, cfg.vocab_size);
    const int seq_len = std::max(1, cfg.seq_len);
    const int text_d_model = std::max(1, cfg.text_d_model);
    const int proj_dim = std::max(1, cfg.proj_dim);

    // ===== Modèle convolutionnel avec blocs optionnels =====
    // Le cœur reste convolutionnel (Conv2d / ConvTranspose2d / SiLU / Add /
    // UpsampleNearest / Reparameterize). On peut en plus activer des blocs
    // ResNet et SelfAttention spatiale (H*W tokens, embed_dim=channels).
    const bool use_resnet       = cfg.resnet;
    const bool attention         = cfg.attention;
    const bool use_skip_conn    = cfg.use_skip_connections;
    const bool use_enc_prior    = cfg.use_encoder_prior;
    // Normalisation configurable. Le décodeur reprend la normalisation encodeur
    // lorsque dec_norm est vide.
    const std::string enc_norm_str = cfg.enc_norm.empty() ? "none" : cfg.enc_norm;
    const int enc_gn_groups_val    = std::max(1, cfg.enc_gn_groups);
    const std::string dec_norm_str = cfg.dec_norm.empty() ? enc_norm_str : cfg.dec_norm;
    const int dec_gn_groups_val    = std::max(1, cfg.dec_gn_groups > 0 ? cfg.dec_gn_groups : cfg.enc_gn_groups);
    const std::string dec_upsample = cfg.decoder_upsample.empty() ? "conv_transpose" : cfg.decoder_upsample;
    if (dec_upsample != "conv_transpose" && dec_upsample != "nearest_conv" &&
        dec_upsample != "bilinear_conv" && dec_upsample != "pixel_shuffle") {
        throw std::runtime_error("VAEConvModel: unsupported decoder_upsample '" + dec_upsample + "'");
    }
    const int resnet_max_tok       = cfg.resnet_max_tokens;  // 0 = no gate
    const int attn_max_tok         = cfg.attn_max_tokens;    // 0 = no gate
    int attn_heads                 = std::max(1, cfg.attn_heads);

    auto resnet_gate = [&](int h, int w) -> bool {
        if (!use_resnet) return false;
        if (resnet_max_tok <= 0) return true;
        return (h * w) <= resnet_max_tok;
    };

    auto attn_gate = [&](int h, int w) -> bool {
        if (!attention) return false;
        if (attn_max_tok <= 0) return true;
        return (h * w) <= attn_max_tok;
    };

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
        throw std::runtime_error(
            "VAEConvModel: cannot reach latent " + std::to_string(LW) + "x" + std::to_string(LH) +
            " from image " + std::to_string(W) + "x" + std::to_string(H) +
            " with identical /2 downsamples (ratios width=" + std::to_string(W / LW) +
            ", height=" + std::to_string(H / LH) + ")");
    }

    const int image_dim = W * H * C;
    const int latent_dim = LH * LW * LC;
    const int base = std::max(2, cfg.base_channels);

    model.modelConfig["task"] = "vae_conv_autoencoder";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["image_dim"] = image_dim;
    model.modelConfig["latent_h"] = LH;
    model.modelConfig["latent_w"] = LW;
    model.modelConfig["latent_c"] = LC;
    model.modelConfig["latent_dim"] = latent_dim;
    model.modelConfig["base_channels"] = base;
    configureLatent(model, cfg);
    model.modelConfig["downsamples"] = downsamples;
    model.modelConfig["input_dim"] = image_dim;
    model.modelConfig["inference_output_tensor"] = "vae_conv/recon";
    // Recon loss: avoid MSE for VAEConv (better behaved on images in [-1,1]).
    model.modelConfig["recon_loss"] = "l1";
    model.modelConfig["text_cond"] = text_cond;
    model.modelConfig["vocab_size"] = vocab_size;
    model.modelConfig["seq_len"] = seq_len;
    model.modelConfig["text_d_model"] = text_d_model;
    model.modelConfig["proj_dim"] = proj_dim;
    model.modelConfig["padding_idx"] = 0;
    model.modelConfig["stochastic_latent"] = stochastic_latent;
    model.modelConfig["enc_norm"] = enc_norm_str;
    model.modelConfig["enc_gn_groups"] = enc_gn_groups_val;
    model.modelConfig["dec_norm"] = dec_norm_str;
    model.modelConfig["dec_gn_groups"] = dec_gn_groups_val;
    model.modelConfig["decoder_upsample"] = dec_upsample;
    model.modelConfig["use_skip_connections"] = use_skip_conn;
    model.modelConfig["use_encoder_prior"] = use_enc_prior;
    model.modelConfig["resnet"] = use_resnet;
    model.modelConfig["attention"] = attention;
    model.modelConfig["attn_heads"] = attn_heads;
    model.modelConfig["attn_max_tokens"] = attn_max_tok;
    model.modelConfig["resnet_max_tokens"] = resnet_max_tok;
    if (cfg.d_model > 0) model.modelConfig["d_model"] = cfg.d_model;
    model.modelConfig["output_dim"] = image_dim + 2 * latent_dim + (text_cond ? 2 * proj_dim : 0);

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
                          int in_w,
                          const std::string& layer_type) {
        const int out_h = std::max(1, in_h * 2);
        const int out_w = std::max(1, in_w * 2);
        model.push(name, layer_type, 0);
        if (auto* U = model.getLayerByName(name)) {
            U->inputs = {in};
            U->output = out;
            U->in_channels = channels;
            U->input_height = in_h;
            U->input_width = in_w;
            U->output_height = out_h;
            U->output_width = out_w;
            U->out_h = out_h;
            U->out_w = out_w;
            U->scale_h = 2.0f;
            U->scale_w = 2.0f;
        }
        return out;
    };

    auto pixel_shuffle2x = [&](const std::string& name,
                               const std::string& in,
                               const std::string& out,
                               int out_channels,
                               int in_h,
                               int in_w) {
        model.push(name, "PixelShuffle", 0);
        if (auto* U = model.getLayerByName(name)) {
            U->inputs = {in};
            U->output = out;
            U->in_channels = out_channels * 4;
            U->out_channels = out_channels;
            U->input_height = in_h;
            U->input_width = in_w;
            U->output_height = in_h * 2;
            U->output_width = in_w * 2;
            U->out_h = in_h * 2;
            U->out_w = in_w * 2;
            U->scale_h = 2.0f;
            U->scale_w = 2.0f;
        }
        return out;
    };

    auto deconv2d = [&](const std::string& name,
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
        const int out_h = std::max(1, (in_h - 1) * s - 2 * p + k);
        const int out_w = std::max(1, (in_w - 1) * s - 2 * p + k);
        model.push(name, "ConvTranspose2d",
                   sat_mul(static_cast<size_t>(out_c), sat_mul(static_cast<size_t>(in_c), sat_mul(static_cast<size_t>(k), static_cast<size_t>(k)))));
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {in};
            L->output = out;
            L->in_channels = in_c;
            L->out_channels = out_c;
            L->input_height = in_h;
            L->input_width = in_w;
            L->output_height = out_h;
            L->output_width = out_w;
            L->out_h = out_h;
            L->out_w = out_w;
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

    // ===== Bloc ResNet : conv3x3 -> SiLU -> conv3x3 + skip =====
    auto resblock = [&](const std::string& prefix, const std::string& in, int ch, int h, int w) -> std::string {
        const size_t ps = sat_mul(static_cast<size_t>(ch), sat_mul(static_cast<size_t>(ch), 9));
        model.push(prefix + "/conv1", "Conv2d", ps);
        if (auto* L = model.getLayerByName(prefix + "/conv1")) {
            L->inputs = {in};        L->output = prefix + "/c1";
            L->in_channels = ch;    L->out_channels = ch;
            L->input_height = h;    L->input_width = w;
            L->kernel_size = 3;     L->stride = 1;  L->padding = 1;
            L->use_bias = false;
        }
        model.push(prefix + "/act1", "SiLU", 0);
        if (auto* A = model.getLayerByName(prefix + "/act1")) {
            A->inputs = {prefix + "/c1"};
            A->output = prefix + "/c1a";
        }
        model.push(prefix + "/conv2", "Conv2d", ps);
        if (auto* L = model.getLayerByName(prefix + "/conv2")) {
            L->inputs = {prefix + "/c1a"};  L->output = prefix + "/c2";
            L->in_channels = ch;            L->out_channels = ch;
            L->input_height = h;            L->input_width = w;
            L->kernel_size = 3;             L->stride = 1;  L->padding = 1;
            L->use_bias = false;
        }
        model.push(prefix + "/add", "Add", 0);
        if (auto* A = model.getLayerByName(prefix + "/add")) {
            A->inputs = {prefix + "/c2", in};
            A->output = prefix + "/out";
        }
        return prefix + "/out";
    };

    // ===== Normalisation encodeur (GroupNorm ou LayerNorm spatial) =====
    auto add_enc_norm = [&](const std::string& prefix, const std::string& in, int ch, int h, int w) -> std::string {
        if (enc_norm_str == "none" || enc_norm_str.empty()) return in;
        if (enc_norm_str == "groupnorm" || enc_norm_str == "gn") {
            int groups = enc_gn_groups_val;
            while (groups > 1 && ch % groups != 0) --groups;
            model.push(prefix + "/gn", "GroupNorm", static_cast<size_t>(ch) * 2);
            if (auto* L = model.getLayerByName(prefix + "/gn")) {
                L->inputs = {in};       L->output = prefix + "/gn_out";
                L->in_channels = ch;    L->num_groups = groups;
                L->input_height = h;    L->input_width = w;
            }
            return prefix + "/gn_out";
        }
        if (enc_norm_str == "layernorm" || enc_norm_str == "ln") {
            const int normalized_size = h * w;
            model.push(prefix + "/ln", "LayerNorm", static_cast<size_t>(normalized_size) * 2);
            if (auto* L = model.getLayerByName(prefix + "/ln")) {
                L->inputs = {in};       L->output = prefix + "/ln_out";
                L->in_channels = ch;    L->input_height = h;  L->input_width = w;
                L->in_features = normalized_size;
            }
            return prefix + "/ln_out";
        }
        return in;
    };

    auto add_dec_norm = [&](const std::string& prefix, const std::string& in, int ch, int h, int w) -> std::string {
        if (dec_norm_str == "none" || dec_norm_str.empty()) return in;
        if (dec_norm_str == "groupnorm" || dec_norm_str == "gn") {
            int groups = dec_gn_groups_val;
            while (groups > 1 && ch % groups != 0) --groups;
            model.push(prefix + "/gn", "GroupNorm", static_cast<size_t>(ch) * 2);
            if (auto* L = model.getLayerByName(prefix + "/gn")) {
                L->inputs = {in};       L->output = prefix + "/gn_out";
                L->in_channels = ch;    L->num_groups = groups;
                L->input_height = h;    L->input_width = w;
            }
            return prefix + "/gn_out";
        }
        if (dec_norm_str == "layernorm" || dec_norm_str == "ln") {
            const int normalized_size = h * w;
            model.push(prefix + "/ln", "LayerNorm", static_cast<size_t>(normalized_size) * 2);
            if (auto* L = model.getLayerByName(prefix + "/ln")) {
                L->inputs = {in};       L->output = prefix + "/ln_out";
                L->in_channels = ch;    L->input_height = h;  L->input_width = w;
                L->in_features = normalized_size;
            }
            return prefix + "/ln_out";
        }
        return in;
    };

    // SelfAttention spatiale (CHW -> HWC -> SelfAttention -> CHW) + résiduel.
    auto self_attn = [&](const std::string& prefix, const std::string& in, int ch, int h, int w) -> std::string {
        int heads = std::max(1, std::min(attn_heads, ch));
        while (heads > 1 && (ch % heads) != 0) --heads;

        model.push(prefix + "/to_hwc", "Permute", 0);
        if (auto* P = model.getLayerByName(prefix + "/to_hwc")) {
            P->inputs = {in};
            P->output = prefix + "/hwc";
            P->shape = {ch, h, w};
            P->permute_dims = {1, 2, 0};
        }

        const size_t attn_params = sat_mul(static_cast<size_t>(ch), sat_mul(static_cast<size_t>(ch), static_cast<size_t>(4)));
        model.push(prefix + "/attn", "SelfAttention", attn_params);
        if (auto* A = model.getLayerByName(prefix + "/attn")) {
            A->inputs = {prefix + "/hwc"};
            A->output = prefix + "/attn_out";
            A->seq_len = h * w;
            A->embed_dim = ch;
            A->num_heads = heads;
            A->causal = false;
        }

        model.push(prefix + "/to_chw", "Permute", 0);
        if (auto* P = model.getLayerByName(prefix + "/to_chw")) {
            P->inputs = {prefix + "/attn_out"};
            P->output = prefix + "/attn_chw";
            P->shape = {h, w, ch};
            P->permute_dims = {2, 0, 1};
        }

        model.push(prefix + "/add", "Add", 0);
        if (auto* A = model.getLayerByName(prefix + "/add")) {
            A->inputs = {in, prefix + "/attn_chw"};
            A->output = prefix + "/out";
        }
        return prefix + "/out";
    };

    // Skip connections (style U-Net) : enc_skips[i] est la feature map de l'encodeur
    // à la résolution H/2^i × W/2^i, sauvegardée avant le downsampling du niveau i.
    std::vector<std::string> enc_skips;
    std::vector<int> enc_skip_h, enc_skip_w;

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

    // ConditioningEncoder
    std::string x = "vae_conv/in_chw";
    int cur_h = H;
    int cur_w = W;
    int cur_c = C;

    int ch = base;
    x = conv2d("vae_conv/enc/conv_in", x, "vae_conv/enc/c0", cur_c, ch, cur_h, cur_w, 3, 1, 1, true);
    cur_c = ch;
    if (resnet_gate(cur_h, cur_w)) {
        x = add_enc_norm("vae_conv/enc/n0", x, cur_c, cur_h, cur_w);
        x = resblock("vae_conv/enc/res0", x, cur_c, cur_h, cur_w);
    }
    if (attn_gate(cur_h, cur_w)) {
        x = self_attn("vae_conv/enc/attn0", x, cur_c, cur_h, cur_w);
    }

    for (int i = 0; i < downsamples; ++i) {
        // Sauvegarde du skip avant downsampling (résolution H/2^i × W/2^i)
        if (use_skip_conn) {
            enc_skips.push_back(x);
            enc_skip_h.push_back(cur_h);
            enc_skip_w.push_back(cur_w);
        }
        const std::string b = "vae_conv/enc/down" + std::to_string(i + 1);
        x = conv2d(b + "/conv", x, b + "/y", cur_c, cur_c, cur_h, cur_w, 3, 2, 1, true);
        cur_h = std::max(1, (cur_h + 2 * 1 - 3) / 2 + 1);
        cur_w = std::max(1, (cur_w + 2 * 1 - 3) / 2 + 1);
        if (resnet_gate(cur_h, cur_w)) {
            x = add_enc_norm(b + "/n", x, cur_c, cur_h, cur_w);
            x = resblock(b + "/res", x, cur_c, cur_h, cur_w);
        }
        if (attn_gate(cur_h, cur_w)) {
            x = self_attn(b + "/attn", x, cur_c, cur_h, cur_w);
        }
    }

    // Project to mu/logvar at latent resolution
    x = conv2d("vae_conv/enc/proj", x, "vae_conv/enc/h", cur_c, cur_c, cur_h, cur_w, 3, 1, 1, true);
    // Bottleneck : ResNet et/ou SelfAttention optionnels
    if (resnet_gate(cur_h, cur_w)) {
        x = add_enc_norm("vae_conv/enc/bot_n", x, cur_c, cur_h, cur_w);
        x = resblock("vae_conv/enc/bot_res", x, cur_c, cur_h, cur_w);
    }
    if (attn_gate(cur_h, cur_w)) {
        x = self_attn("vae_conv/enc/bot_attn", x, cur_c, cur_h, cur_w);
    }
    if (cfg.latent_detail_path) {
        const auto packed = phaseRearrange(model, "vae_conv/enc/detail/pack", "vae_conv/in_chw",
            C, H, W, LH, LW, false);
        const auto detail = latentConv(model, "vae_conv/enc/detail/project", packed,
            C * (H / LH) * (W / LW), cfg.latent_context_channels, LH, LW);
        model.push("vae_conv/enc/detail/concat", "Concat", 0);
        auto* concat = model.getLayerByName("vae_conv/enc/detail/concat");
        concat->inputs = {x, detail}; concat->output = "vae_conv/enc/detail/combined";
        concat->concat_axis = 0;
        x = concat->output; cur_c += cfg.latent_context_channels;
    }
    if (cfg.latent_spatial_context) {
        x = latentContext(model, "vae_conv/enc/latent_context", x, cur_c,
            cfg.latent_context_channels, LH, LW);
        cur_c = cfg.latent_context_channels;
    } else if (cfg.latent_detail_path) {
        x = latentConv(model, "vae_conv/enc/detail/fuse", x, cur_c,
            cfg.latent_context_channels, LH, LW);
        cur_c = cfg.latent_context_channels;
    }
    model.push("vae_conv/enc/mu", "Conv2d",
               sat_mul(static_cast<size_t>(LC), sat_mul(static_cast<size_t>(cur_c), sat_mul(static_cast<size_t>(1), static_cast<size_t>(1)))));
    if (auto* L = model.getLayerByName("vae_conv/enc/mu")) {
        L->inputs = {x};
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

    std::string logvar_features = conv2d(
        "vae_conv/enc/logvar_pre_conv", x, "vae_conv/enc/logvar_pre",
        cur_c, cur_c, cur_h, cur_w, 3, 1, 1, true);
    logvar_features = add_enc_norm(
        "vae_conv/enc/logvar_pre_norm", logvar_features, cur_c, cur_h, cur_w);

    model.push("vae_conv/enc/logvar", "Conv2d",
               sat_mul(static_cast<size_t>(LC), sat_mul(static_cast<size_t>(cur_c), sat_mul(static_cast<size_t>(1), static_cast<size_t>(1)))));
    if (auto* L = model.getLayerByName("vae_conv/enc/logvar")) {
        L->inputs = {logvar_features};
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

    // Prior appris dans le graphe : couche Constant de taille latent_dim
    // ajoutée additivement à z avant le décodeur.
    // Les poids de cette couche (un vecteur CHW de taille LC*LH*LW) sont
    // entraînés par backprop : ils capturent un biais global appris sur
    // l'espace latent, améliorant la représentation sans interaction
    // avec le tokenizer ni la table d'embeddings.
    std::string z_in = "vae_conv/z";
    if (use_enc_prior) {
        model.push("vae_conv/z_prior_bias", "Constant",
                   static_cast<size_t>(LC) * static_cast<size_t>(LH) * static_cast<size_t>(LW));
        if (auto* L = model.getLayerByName("vae_conv/z_prior_bias")) {
            L->inputs  = {};
            L->output  = "vae_conv/prior_bias_out";
            L->trainable_parameter = true;
            L->in_channels  = LC;
            L->out_channels = LC;
            L->input_height = LH;
            L->input_width  = LW;
        }
        model.push("vae_conv/z_prior_add", "Add", 0);
        if (auto* L = model.getLayerByName("vae_conv/z_prior_add")) {
            L->inputs = {"vae_conv/z", "vae_conv/prior_bias_out"};
            L->output = "vae_conv/z_biased";
        }
        z_in = "vae_conv/z_biased";
    }

    std::string y = z_in;
    int dy_h = LH;
    int dy_w = LW;
    int dy_c = LC;
    if (cfg.latent_spatial_context) {
        y = latentContext(model, "vae_conv/dec/latent_context", y, dy_c,
            cfg.latent_context_channels, LH, LW);
        dy_c = cfg.latent_context_channels;
    }
    const std::string detail_features = y;
    const int detail_channels = dy_c;


    // Bring to base channels
    y = conv2d("vae_conv/dec/conv_in", y, "vae_conv/dec/c0", dy_c, base, dy_h, dy_w, 3, 1, 1, true);
    dy_c = base;
    y = add_dec_norm("vae_conv/dec/n0", y, dy_c, dy_h, dy_w);

    // Bottleneck décodeur : ResNet et/ou SelfAttention optionnels
    if (resnet_gate(dy_h, dy_w)) {
        y = add_dec_norm("vae_conv/dec/bot_n", y, dy_c, dy_h, dy_w);
        y = resblock("vae_conv/dec/bot_res", y, dy_c, dy_h, dy_w);
    }
    if (attn_gate(dy_h, dy_w)) {
        y = self_attn("vae_conv/dec/bot_attn", y, dy_c, dy_h, dy_w);
    }

    for (int i = downsamples - 1; i >= 0; --i) {
        const std::string b = "vae_conv/dec/up" + std::to_string(i + 1);
        const int in_h = dy_h;
        const int in_w = dy_w;
        if (dec_upsample == "nearest_conv" || dec_upsample == "bilinear_conv") {
            const std::string layer_type = dec_upsample == "nearest_conv"
                ? "UpsampleNearest" : "UpsampleBilinear";
            y = upsample2x(b + "/up", y, b + "/up_y", dy_c, in_h, in_w, layer_type);
            dy_h = in_h * 2;
            dy_w = in_w * 2;
            y = conv2d(b + "/conv", y, b + "/c", dy_c, dy_c, dy_h, dy_w, 3, 1, 1, true);
        } else if (dec_upsample == "pixel_shuffle") {
            y = conv2d(b + "/conv", y, b + "/c", dy_c, dy_c * 4, in_h, in_w, 3, 1, 1, false);
            y = pixel_shuffle2x(b + "/up", y, b + "/up_y", dy_c, in_h, in_w);
            dy_h = in_h * 2;
            dy_w = in_w * 2;
        } else {
            y = deconv2d(b + "/up", y, b + "/up_y", dy_c, dy_c, in_h, in_w, 4, 2, 1, true);
            dy_h = in_h * 2;
            dy_w = in_w * 2;
        }
        // Skip connection encodeur→décodeur : concat + projection 1×1 (2*base → base)
        if (use_skip_conn && i < static_cast<int>(enc_skips.size())) {
            model.push(b + "/skip_cat", "Concat", 0);
            if (auto* L = model.getLayerByName(b + "/skip_cat")) {
                L->skip_input_index = 1;
                L->inputs      = {y, enc_skips[static_cast<size_t>(i)]};
                L->output      = b + "/sc";
                L->concat_axis = 0;
            }
            // Conv 1×1 linéaire : comprime 2*base canaux → base
            y = conv2d(b + "/skip_proj", b + "/sc", b + "/sp",
                       2 * dy_c, dy_c, dy_h, dy_w, 1, 1, 0, /*act=*/false);
        }
        y = add_dec_norm(b + "/n", y, dy_c, dy_h, dy_w);
        if (resnet_gate(dy_h, dy_w)) {
            y = resblock(b + "/res", y, dy_c, dy_h, dy_w);
        }
        if (attn_gate(dy_h, dy_w)) {
            y = self_attn(b + "/attn", y, dy_c, dy_h, dy_w);
        }
    }

    // Final to RGB
    y = add_dec_norm("vae_conv/dec/out_n", y, dy_c, dy_h, dy_w);
    y = conv2d("vae_conv/dec/out", y, "vae_conv/dec/out_pre", dy_c, C, dy_h, dy_w, 3, 1, 1, false);

    if (cfg.latent_detail_path)
        y = detailReconstruction(model, detail_features, y, detail_channels, C, H, W, LH, LW);

    model.push("vae_conv/dec/tanh", "Tanh", 0);
    if (auto* T = model.getLayerByName("vae_conv/dec/tanh")) {
        T->inputs = {y};
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

    // Projections multi-modales optionnelles pour trainStep.
    if (text_cond) {
        model.push("vae_conv/txt/tok_emb", "Embedding",
                   static_cast<size_t>(vocab_size) * static_cast<size_t>(text_d_model));
        if (auto* E = model.getLayerByName("vae_conv/txt/tok_emb")) {
            E->inputs = {"text_ids"};
            E->output = "vae_conv/txt/tok_emb_out";
            E->vocab_size = vocab_size;
            E->embed_dim = text_d_model;
            E->padding_idx = 0;
            E->seq_len = seq_len;
        }

        model.push("vae_conv/txt/pool", "TokenMeanPool", 0);
        if (auto* P = model.getLayerByName("vae_conv/txt/pool")) {
            P->inputs = {"vae_conv/txt/tok_emb_out"};
            P->output = "vae_conv/txt/pooled";
            P->seq_len = seq_len;
            P->embed_dim = text_d_model;
        }

        model.push("vae_conv/img_proj", "Linear",
                   static_cast<size_t>(latent_dim) * static_cast<size_t>(proj_dim) + static_cast<size_t>(proj_dim));
        if (auto* L = model.getLayerByName("vae_conv/img_proj")) {
            L->inputs = {z_in};
            L->output = "vae_conv/img_proj";
            L->in_features = latent_dim;
            L->out_features = proj_dim;
            L->use_bias = true;
        }

        model.push("vae_conv/txt_proj", "Linear",
                   static_cast<size_t>(text_d_model) * static_cast<size_t>(proj_dim) + static_cast<size_t>(proj_dim));
        if (auto* L = model.getLayerByName("vae_conv/txt_proj")) {
            L->inputs = {"vae_conv/txt/pooled"};
            L->output = "vae_conv/txt_proj";
            L->in_features = text_d_model;
            L->out_features = proj_dim;
            L->use_bias = true;
        }
    }

    // Contrat VAE: recon || mu || logvar || img_proj || txt_proj (optionnel).
    // z_in reste le latent opérationnel passé au décodeur et à img_proj, mais la
    // loss KL doit impérativement recevoir la moyenne de q(z|x), pas un sample.
    model.push("vae_conv/out_concat", "Concat", 0);
    if (auto* L = model.getLayerByName("vae_conv/out_concat")) {
        L->inputs = {"vae_conv/recon", "vae_conv/mu", "vae_conv/logvar"};
        if (text_cond) {
            L->inputs.push_back("vae_conv/img_proj");
            L->inputs.push_back("vae_conv/txt_proj");
        }
        L->output = "x";
        L->concat_axis = 0;
    }
}

std::optional<Model::TrainStepResult> VAEConvModel::trainStep(const TrainStepRequest& request) {
    applyRuntimeConfiguration();
    if (!request.optimizer) {
        return std::nullopt;
    }

    const std::vector<float>* input = nullptr;
    if (const auto it = request.float_inputs.find("__input__"); it != request.float_inputs.end() && it->second) {
        input = it->second;
    } else if (const auto it = request.float_inputs.find("image"); it != request.float_inputs.end() && it->second) {
        input = it->second;
    }
    if (!input) return std::nullopt;

    if (request.annotation) {
        auto weights = BBoxSupervision::makeWeights(request.annotation, cfg_.image_w, cfg_.image_h,
            cfg_.image_c, BBoxSupervision::Layout::HWC, modelConfig.value("bbox_loss_weight",1.0f), &modelConfig);
        if (request.loss_weights) BBoxSupervision::combineMask(weights,*request.loss_weights);
        auto adapted = request;
        adapted.annotation = nullptr;
        adapted.relations = &weights.relations;
        adapted.loss_weights = weights.values.empty() ? nullptr : &weights.values;
        auto result = trainStep(adapted);
        if (result) {
            result->metrics["bbox_count"] = static_cast<float>(weights.bbox_count);
            result->metrics["bbox_loss_weight"] = weights.strength;
            result->metrics["relation_count"] = static_cast<float>(weights.relations.size());
        }
        return result;
    }

    if (const auto it = request.int_inputs.find("text_ids"); it != request.int_inputs.end() && it->second) {
        return vae_training_.trainText(*this, *input, *it->second, request.target, request.loss_weights,
                           *request.optimizer, request.learning_rate,
                           request.mode, request.grad_scale, request.relations);
    }
        return vae_training_.trainImage(*this, *input, request.target, request.loss_weights,
                        *request.optimizer, request.learning_rate,
                        request.mode, request.grad_scale, request.relations);
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
        throw std::runtime_error(
            "VAEConvModel(decode): cannot reach latent " + std::to_string(LW) + "x" + std::to_string(LH) +
            " from image " + std::to_string(W) + "x" + std::to_string(H) +
            " with identical /2 downsamples (ratios width=" + std::to_string(W / LW) +
            ", height=" + std::to_string(H / LH) + ")");
    }

    const int image_dim = W * H * C;
    const int latent_dim = LH * LW * LC;
    const int base = std::max(2, cfg.base_channels);
    // Décodeur convolutionnel avec blocs optionnels (ResNet + SelfAttention).
    const std::string dec_norm_str =
        cfg.dec_norm.empty() ? (cfg.enc_norm.empty() ? "none" : cfg.enc_norm) : cfg.dec_norm;
    const int dec_gn_groups_val = std::max(1, cfg.dec_gn_groups > 0 ? cfg.dec_gn_groups : cfg.enc_gn_groups);
    const std::string dec_upsample = cfg.decoder_upsample.empty() ? "conv_transpose" : cfg.decoder_upsample;
    if (dec_upsample != "conv_transpose" && dec_upsample != "nearest_conv" &&
        dec_upsample != "bilinear_conv" && dec_upsample != "pixel_shuffle") {
        throw std::runtime_error("VAEConvModel(decode): unsupported decoder_upsample '" + dec_upsample + "'");
    }

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
    configureLatent(model, cfg);
    model.modelConfig["downsamples"] = downsamples;
    model.modelConfig["input_dim"] = latent_dim;
    model.modelConfig["output_dim"] = image_dim;
    model.modelConfig["inference_output_tensor"] = "x";
    model.modelConfig["dec_norm"] = dec_norm_str;
    model.modelConfig["dec_gn_groups"] = dec_gn_groups_val;
    model.modelConfig["decoder_upsample"] = dec_upsample;

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
                          int in_w,
                          const std::string& layer_type) {
        const int out_h = std::max(1, in_h * 2);
        const int out_w = std::max(1, in_w * 2);
        model.push(name, layer_type, 0);
        if (auto* U = model.getLayerByName(name)) {
            U->inputs = {in};
            U->output = out;
            U->in_channels = channels;
            U->input_height = in_h;
            U->input_width = in_w;
            U->output_height = out_h;
            U->output_width = out_w;
            U->out_h = out_h;
            U->out_w = out_w;
            U->scale_h = 2.0f;
            U->scale_w = 2.0f;
        }
        return out;
    };

    auto pixel_shuffle2x = [&](const std::string& name,
                               const std::string& in,
                               const std::string& out,
                               int out_channels,
                               int in_h,
                               int in_w) {
        model.push(name, "PixelShuffle", 0);
        if (auto* U = model.getLayerByName(name)) {
            U->inputs = {in};
            U->output = out;
            U->in_channels = out_channels * 4;
            U->out_channels = out_channels;
            U->input_height = in_h;
            U->input_width = in_w;
            U->output_height = in_h * 2;
            U->output_width = in_w * 2;
            U->out_h = in_h * 2;
            U->out_w = in_w * 2;
            U->scale_h = 2.0f;
            U->scale_w = 2.0f;
        }
        return out;
    };

    auto deconv2d = [&](const std::string& name,
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
        const int out_h = std::max(1, (in_h - 1) * s - 2 * p + k);
        const int out_w = std::max(1, (in_w - 1) * s - 2 * p + k);
        model.push(name, "ConvTranspose2d",
                   sat_mul(static_cast<size_t>(out_c), sat_mul(static_cast<size_t>(in_c), sat_mul(static_cast<size_t>(k), static_cast<size_t>(k)))));
        if (auto* L = model.getLayerByName(name)) {
            L->inputs = {in};
            L->output = out;
            L->in_channels = in_c;
            L->out_channels = out_c;
            L->input_height = in_h;
            L->input_width = in_w;
            L->output_height = out_h;
            L->output_width = out_w;
            L->out_h = out_h;
            L->out_w = out_w;
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

    // Options blocs optionnels
    const bool use_resnet_dec  = cfg.resnet;
    const int resnet_max_dec   = cfg.resnet_max_tokens;
    const bool use_attn_dec    = cfg.attention;
    const int attn_max_dec     = cfg.attn_max_tokens;
    int attn_heads_dec         = std::max(1, cfg.attn_heads);
    // Skip connections : si le checkpoint a été entraîné avec use_skip_connections=true,
    // on reconstruit le même graphe décodeur mais l'encodeur skip est remplacé par un
    // tenseur zéro (Constant non présent dans le checkpoint → initialisé à 0).
    // Résultat : skip_proj reçoit [decoder_feats, 0] → W_dec * decoder_feats
    // C'est bien meilleur que bypasser skip_proj entièrement.
    const bool use_skip_dec = cfg.use_skip_connections;
    // Stocker dans modelConfig pour permettre de reconstruire une architecture identique.
    model.modelConfig["resnet"] = use_resnet_dec;
    model.modelConfig["resnet_max_tokens"] = resnet_max_dec;
    model.modelConfig["attention"] = use_attn_dec;
    model.modelConfig["attn_heads"] = attn_heads_dec;
    model.modelConfig["attn_max_tokens"] = attn_max_dec;
    model.modelConfig["use_skip_connections"] = use_skip_dec;

    auto resnet_gate_dec = [&](int h, int w) -> bool {
        if (!use_resnet_dec) return false;
        return resnet_max_dec <= 0 || (h * w) <= resnet_max_dec;
    };

    auto attn_gate_dec = [&](int h, int w) -> bool {
        if (!use_attn_dec) return false;
        return attn_max_dec <= 0 || (h * w) <= attn_max_dec;
    };

    auto resblock_dec = [&](const std::string& prefix, const std::string& in, int ch, int h, int w) -> std::string {
        const size_t ps = sat_mul(static_cast<size_t>(ch), sat_mul(static_cast<size_t>(ch), 9));
        model.push(prefix + "/conv1", "Conv2d", ps);
        if (auto* L = model.getLayerByName(prefix + "/conv1")) {
            L->inputs = {in};        L->output = prefix + "/c1";
            L->in_channels = ch;    L->out_channels = ch;
            L->input_height = h;    L->input_width = w;
            L->kernel_size = 3;     L->stride = 1;  L->padding = 1;
            L->use_bias = false;
        }
        model.push(prefix + "/act1", "SiLU", 0);
        if (auto* A = model.getLayerByName(prefix + "/act1")) {
            A->inputs = {prefix + "/c1"};  A->output = prefix + "/c1a";
        }
        model.push(prefix + "/conv2", "Conv2d", ps);
        if (auto* L = model.getLayerByName(prefix + "/conv2")) {
            L->inputs = {prefix + "/c1a"};  L->output = prefix + "/c2";
            L->in_channels = ch;            L->out_channels = ch;
            L->input_height = h;            L->input_width = w;
            L->kernel_size = 3;             L->stride = 1;  L->padding = 1;
            L->use_bias = false;
        }
        model.push(prefix + "/add", "Add", 0);
        if (auto* A = model.getLayerByName(prefix + "/add")) {
            A->inputs = {prefix + "/c2", in};  A->output = prefix + "/out";
        }
        return prefix + "/out";
    };

    auto add_dec_norm = [&](const std::string& prefix, const std::string& in, int ch, int h, int w) -> std::string {
        if (dec_norm_str == "none" || dec_norm_str.empty()) return in;
        if (dec_norm_str == "groupnorm" || dec_norm_str == "gn") {
            int groups = dec_gn_groups_val;
            while (groups > 1 && ch % groups != 0) --groups;
            model.push(prefix + "/gn", "GroupNorm", static_cast<size_t>(ch) * 2);
            if (auto* L = model.getLayerByName(prefix + "/gn")) {
                L->inputs = {in};       L->output = prefix + "/gn_out";
                L->in_channels = ch;    L->num_groups = groups;
                L->input_height = h;    L->input_width = w;
            }
            return prefix + "/gn_out";
        }
        if (dec_norm_str == "layernorm" || dec_norm_str == "ln") {
            const int normalized_size = h * w;
            model.push(prefix + "/ln", "LayerNorm", static_cast<size_t>(normalized_size) * 2);
            if (auto* L = model.getLayerByName(prefix + "/ln")) {
                L->inputs = {in};       L->output = prefix + "/ln_out";
                L->in_channels = ch;    L->input_height = h;  L->input_width = w;
                L->in_features = normalized_size;
            }
            return prefix + "/ln_out";
        }
        return in;
    };

    auto self_attn_dec = [&](const std::string& prefix, const std::string& in, int ch, int h, int w) -> std::string {
        int heads = std::max(1, std::min(attn_heads_dec, ch));
        while (heads > 1 && (ch % heads) != 0) --heads;

        model.push(prefix + "/to_hwc", "Permute", 0);
        if (auto* P = model.getLayerByName(prefix + "/to_hwc")) {
            P->inputs = {in};
            P->output = prefix + "/hwc";
            P->shape = {ch, h, w};
            P->permute_dims = {1, 2, 0};
        }

        const size_t attn_params = sat_mul(static_cast<size_t>(ch), sat_mul(static_cast<size_t>(ch), static_cast<size_t>(4)));
        model.push(prefix + "/attn", "SelfAttention", attn_params);
        if (auto* A = model.getLayerByName(prefix + "/attn")) {
            A->inputs = {prefix + "/hwc"};
            A->output = prefix + "/attn_out";
            A->seq_len = h * w;
            A->embed_dim = ch;
            A->num_heads = heads;
            A->causal = false;
        }

        model.push(prefix + "/to_chw", "Permute", 0);
        if (auto* P = model.getLayerByName(prefix + "/to_chw")) {
            P->inputs = {prefix + "/attn_out"};
            P->output = prefix + "/attn_chw";
            P->shape = {h, w, ch};
            P->permute_dims = {2, 0, 1};
        }

        model.push(prefix + "/add", "Add", 0);
        if (auto* A = model.getLayerByName(prefix + "/add")) {
            A->inputs = {in, prefix + "/attn_chw"};
            A->output = prefix + "/out";
        }
        return prefix + "/out";
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
    if (cfg.latent_spatial_context) {
        y = latentContext(model, "vae_conv/dec/latent_context", y, dy_c,
            cfg.latent_context_channels, LH, LW);
        dy_c = cfg.latent_context_channels;
    }
    const std::string detail_features = y;
    const int detail_channels = dy_c;


    y = conv2d("vae_conv/dec/conv_in", y, "vae_conv/dec/c0", dy_c, base, dy_h, dy_w, 3, 1, 1, true);
    dy_c = base;
    y = add_dec_norm("vae_conv/dec/n0", y, dy_c, dy_h, dy_w);

    // Bottleneck décodeur : ResNet et/ou SelfAttention
    if (resnet_gate_dec(dy_h, dy_w)) {
        y = add_dec_norm("vae_conv/dec/bot_n", y, dy_c, dy_h, dy_w);
        y = resblock_dec("vae_conv/dec/bot_res", y, dy_c, dy_h, dy_w);
    }
    if (attn_gate_dec(dy_h, dy_w)) {
        y = self_attn_dec("vae_conv/dec/bot_attn", y, dy_c, dy_h, dy_w);
    }

    for (int i = downsamples - 1; i >= 0; --i) {
        const std::string b = "vae_conv/dec/up" + std::to_string(i + 1);
        const int in_h = dy_h;
        const int in_w = dy_w;
        if (dec_upsample == "nearest_conv" || dec_upsample == "bilinear_conv") {
            const std::string layer_type = dec_upsample == "nearest_conv"
                ? "UpsampleNearest" : "UpsampleBilinear";
            y = upsample2x(b + "/up", y, b + "/up_y", dy_c, in_h, in_w, layer_type);
            dy_h = in_h * 2;
            dy_w = in_w * 2;
            y = conv2d(b + "/conv", y, b + "/c", dy_c, dy_c, dy_h, dy_w, 3, 1, 1, true);
        } else if (dec_upsample == "pixel_shuffle") {
            y = conv2d(b + "/conv", y, b + "/c", dy_c, dy_c * 4, in_h, in_w, 3, 1, 1, false);
            y = pixel_shuffle2x(b + "/up", y, b + "/up_y", dy_c, in_h, in_w);
            dy_h = in_h * 2;
            dy_w = in_w * 2;
        } else {
            y = deconv2d(b + "/up", y, b + "/up_y", dy_c, dy_c, in_h, in_w, 4, 2, 1, true);
            dy_h = in_h * 2;
            dy_w = in_w * 2;
        }
        // Skip connection avec encodeur zéro : charge les poids skip_proj du checkpoint
        // et applique à [decoder_feats, 0]. Le Constant n'est PAS dans le checkpoint
        // donc initialisé à zéro par l'allocateur → skip_proj applique uniquement
        // la moitié "decoder" de ses poids, ce qui est bien plus fidèle au modèle
        // entraîné qu'ignorer skip_proj entièrement.
        if (use_skip_dec) {
            const std::string zero_name = b + "/zero_enc_skip";
            model.push(zero_name, "Constant",
                       static_cast<size_t>(dy_c) * static_cast<size_t>(dy_h) * static_cast<size_t>(dy_w));
            if (auto* L = model.getLayerByName(zero_name)) {
                L->inputs  = {};
                L->output  = zero_name + "_out";
                L->in_channels  = dy_c;
                L->out_channels = dy_c;
                L->input_height = dy_h;
                L->input_width  = dy_w;
            }
            model.push(b + "/skip_cat", "Concat", 0);
            if (auto* L = model.getLayerByName(b + "/skip_cat")) {
                L->skip_input_index = 1;
                L->inputs      = {y, zero_name + "_out"};
                L->output      = b + "/sc";
                L->concat_axis = 0;
            }
            y = conv2d(b + "/skip_proj", b + "/sc", b + "/sp",
                       2 * dy_c, dy_c, dy_h, dy_w, 1, 1, 0, /*act=*/false);
        }
        y = add_dec_norm(b + "/n", y, dy_c, dy_h, dy_w);
        if (resnet_gate_dec(dy_h, dy_w)) {
            y = resblock_dec(b + "/res", y, dy_c, dy_h, dy_w);
        }
        if (attn_gate_dec(dy_h, dy_w)) {
            y = self_attn_dec(b + "/attn", y, dy_c, dy_h, dy_w);
        }
    }

    y = add_dec_norm("vae_conv/dec/out_n", y, dy_c, dy_h, dy_w);
    y = conv2d("vae_conv/dec/out", y, "vae_conv/dec/out_pre", dy_c, C, dy_h, dy_w, 3, 1, 1, false);

    if (cfg.latent_detail_path)
        y = detailReconstruction(model, detail_features, y, detail_channels, C, H, W, LH, LW);

    model.push("vae_conv/dec/tanh", "Tanh", 0);
    if (auto* T = model.getLayerByName("vae_conv/dec/tanh")) {
        T->inputs = {y};
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
