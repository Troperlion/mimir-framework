#include "VGG16Model.hpp"

#include <algorithm>

bool VGG16Model::InitVizTips() {
    clearVizTipsRegistry();
    clearVizTaps();

    // Points clés visibles dans le panneau Blocks/Layers du visualiseur.
    registerVizTip("vgg16/raw_in",         "Dataset/raw");
    registerVizTip("vgg16/b1/down/conv",   "Block1/down");
    registerVizTip("vgg16/b1/down/relu",   "Block1/out");
    registerVizTip("vgg16/b2/down/conv",   "Block2/down");
    registerVizTip("vgg16/b2/down/relu",   "Block2/out");
    registerVizTip("vgg16/b3/down/conv",   "Block3/down");
    registerVizTip("vgg16/b3/down/relu",   "Block3/out");
    registerVizTip("vgg16/b4/down/conv",   "Block4/down");
    registerVizTip("vgg16/b4/down/relu",   "Block4/out");
    registerVizTip("vgg16/b5/down/conv",   "Block5/down");
    registerVizTip("vgg16/b5/down/relu",   "Block5/out");
    registerVizTip("vgg16/gap",            "Head/gap");
    registerVizTip("vgg16/fc1_act",        "Head/fc1");
    registerVizTip("vgg16/fc2_act",        "Head/fc2");
    registerVizTip("vgg16/head",           "Output/logits");

    return true;
}

bool VGG16Model::UpdateVizTips(const Layer& layer, VizFrame& frame) {
    if (Model::UpdateVizTips(layer, frame)) return true;
    if (layer.name.empty()) return false;

    auto attach_tip = [&](const std::string& tip) -> bool {
        if (tip.empty()) return false;
        frame.label = tip + "|" + frame.label;
        return true;
    };

    const std::string& n = layer.name;

    // Blocs convolutifs 1–5
    for (int b = 1; b <= 5; ++b) {
        const std::string bn = "/b" + std::to_string(b) + "/";
        if (n.find(bn) != std::string::npos) {
            const std::string block = "Block" + std::to_string(b);
            if (n.find("/down") != std::string::npos) {
                if (n.find("/relu") != std::string::npos) return attach_tip(block + "/down/act");
                if (n.find("/ln")   != std::string::npos) return attach_tip(block + "/down/norm");
                return attach_tip(block + "/down");
            }
            if (n.find("/pool") != std::string::npos) return attach_tip(block + "/pool");
            if (n.find("/relu") != std::string::npos) return attach_tip(block + "/act");
            if (n.find("/ln")   != std::string::npos) return attach_tip(block + "/norm");
            return attach_tip(block + "/conv");
        }
    }

    if (n.find("/gap")  != std::string::npos) return attach_tip("Head/gap");
    if (n.find("/drop") != std::string::npos) return attach_tip("Head/dropout");
    if (n.find("/fc2")  != std::string::npos) return attach_tip("Head/fc2");
    if (n.find("/fc1")  != std::string::npos) return attach_tip("Head/fc1");
    if (n.find("/head") != std::string::npos) return attach_tip("Output/logits");

    return false;
}

VGG16Model::VGG16Model() {
    setModelName("VGG16Model");
    setHasEncoder(false);
}

void VGG16Model::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

static inline int conv_out(int in, int k, int s, int p) {
    return (in + 2 * p - k) / s + 1;
}

void VGG16Model::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("VGG16Model");
    model.modelConfig["type"] = "vgg16";

    int W = std::max(1, cfg.image_w);
    int H = std::max(1, cfg.image_h);
    int C = std::max(1, cfg.image_c);
    const int base = std::max(4, cfg.base_channels);
    const int num_classes = std::max(1, cfg.num_classes);
    const int fc_hidden = std::max(16, cfg.fc_hidden);
    const float dropout = std::clamp(cfg.dropout, 0.0f, 0.95f);
    const int image_dim = W * H * C;

    model.modelConfig["task"] = "image_classification";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["base_channels"] = base;
    model.modelConfig["num_classes"] = num_classes;
    model.modelConfig["input_dim"] = image_dim;
    model.modelConfig["output_dim"] = num_classes;
    model.modelConfig["dropout"] = dropout;

    model.push("vgg16/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("vgg16/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "vgg16/in";
    }

    auto add_conv_ln_act = [&](const std::string& name,
                               const std::string& in,
                               const std::string& out,
                               int in_c,
                               int out_c,
                               int in_h,
                               int in_w,
                               int k,
                               int s,
                               int p,
                               int out_h,
                               int out_w) {
        model.push(name, "Conv2d", static_cast<size_t>(out_c) * static_cast<size_t>(in_c) * static_cast<size_t>(k) * static_cast<size_t>(k));
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
        // LayerNorm (affine=false) : stabilise les activations entre convolutions.
        // GroupNorm/BatchNorm2d ont un backward non fonctionnel dans ce framework
        // → LayerNorm retenu (cf. commentaires VGG16FeatModel).
        const std::string ln = name + "/ln";
        model.push(ln, "LayerNorm", 0);
        if (auto* N = model.getLayerByName(ln)) {
            N->inputs = {out};
            N->output = out + "_ln";
            N->in_channels = out_c;
            N->input_height = out_h;
            N->input_width = out_w;
            N->in_features = std::max(1, out_c * out_h * out_w);
            N->affine = false;
            N->use_bias = false;
        }
        model.push(name + "/relu", "ReLU", 0);
        if (auto* R = model.getLayerByName(name + "/relu")) {
            R->inputs = {out + "_ln"};
            R->output = out + "_act";
        }
        return out + "_act";
    };

    std::string x = "vgg16/in";
    int cur_c = C;
    int cur_h = H;
    int cur_w = W;

    // VGG16 conv counts: [2,2,3,3,3] with downsample between blocks.
    auto block = [&](int bi, int convs, int out_c) {
        const std::string p = "vgg16/b" + std::to_string(bi);
        for (int i = 0; i < convs; ++i) {
            // conv 3×3, stride=1, pad=1 → dimensions spatiales inchangées
            x = add_conv_ln_act(p + "/c" + std::to_string(i + 1), x, p + "/y" + std::to_string(i + 1),
                                cur_c, out_c, cur_h, cur_w, 3, 1, 1, cur_h, cur_w);
            cur_c = out_c;
        }
        // Downsample via conv stride-2 (seul downsampling au backward correct)
        const int dh = conv_out(cur_h, 3, 2, 1);
        const int dw = conv_out(cur_w, 3, 2, 1);
        x = add_conv_ln_act(p + "/down", x, p + "/down_y", cur_c, cur_c, cur_h, cur_w, 3, 2, 1, dh, dw);
        cur_h = dh;
        cur_w = dw;
    };

    block(1, 2, base);
    block(2, 2, base * 2);
    block(3, 3, base * 4);
    block(4, 3, base * 8);
    block(5, 3, base * 8);

    model.push("vgg16/gap", "GlobalAvgPool2d", 0);
    if (auto* L = model.getLayerByName("vgg16/gap")) {
        L->inputs = {x};
        L->output = "vgg16/gap_y";
        L->in_channels = cur_c;
        L->input_height = cur_h;
        L->input_width = cur_w;
    }

    std::string cur = "vgg16/gap_y";

    // Dropout + FC1 + GELU
    if (dropout > 0.0f) {
        model.push("vgg16/drop1", "Dropout", 0);
        if (auto* L = model.getLayerByName("vgg16/drop1")) {
            L->inputs = {cur};
            L->output = "vgg16/drop1_y";
            L->dropout_p = dropout;
        }
        cur = "vgg16/drop1_y";
    }
    model.push("vgg16/fc1", "Linear", static_cast<size_t>(cur_c) * static_cast<size_t>(fc_hidden) + static_cast<size_t>(fc_hidden));
    if (auto* L = model.getLayerByName("vgg16/fc1")) {
        L->inputs = {cur};
        L->output = "vgg16/fc1_y";
        L->in_features = cur_c;
        L->out_features = fc_hidden;
        L->use_bias = true;
    }
    model.push("vgg16/fc1_act", "GELU", 0);
    if (auto* L = model.getLayerByName("vgg16/fc1_act")) {
        L->inputs = {"vgg16/fc1_y"};
        L->output = "vgg16/fc1_h";
    }
    cur = "vgg16/fc1_h";

    // Dropout + FC2 + GELU
    if (dropout > 0.0f) {
        model.push("vgg16/drop2", "Dropout", 0);
        if (auto* L = model.getLayerByName("vgg16/drop2")) {
            L->inputs = {cur};
            L->output = "vgg16/drop2_y";
            L->dropout_p = dropout;
        }
        cur = "vgg16/drop2_y";
    }
    model.push("vgg16/fc2", "Linear", static_cast<size_t>(fc_hidden) * static_cast<size_t>(fc_hidden) + static_cast<size_t>(fc_hidden));
    if (auto* L = model.getLayerByName("vgg16/fc2")) {
        L->inputs = {cur};
        L->output = "vgg16/fc2_y";
        L->in_features = fc_hidden;
        L->out_features = fc_hidden;
        L->use_bias = true;
    }
    model.push("vgg16/fc2_act", "GELU", 0);
    if (auto* L = model.getLayerByName("vgg16/fc2_act")) {
        L->inputs = {"vgg16/fc2_y"};
        L->output = "vgg16/fc2_h";
    }

    // Tête de classification
    model.push("vgg16/head", "Linear", static_cast<size_t>(fc_hidden) * static_cast<size_t>(num_classes) + static_cast<size_t>(num_classes));
    if (auto* L = model.getLayerByName("vgg16/head")) {
        L->inputs = {"vgg16/fc2_h"};
        L->output = "x";
        L->in_features = fc_hidden;
        L->out_features = num_classes;
        L->use_bias = true;
    }
}
