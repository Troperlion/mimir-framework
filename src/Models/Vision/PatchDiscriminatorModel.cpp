#include "PatchDiscriminatorModel.hpp"

#include <algorithm>
#include <stdexcept>

static inline int conv_out(int in, int k, int s, int p) {
    return (in + 2 * p - k) / s + 1;
}

PatchDiscriminatorModel::PatchDiscriminatorModel() {
    setModelName("PatchDiscriminatorModel");
    setHasEncoder(false);
}

void PatchDiscriminatorModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

void PatchDiscriminatorModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("PatchDiscriminatorModel");
    model.modelConfig["type"] = "patch_discriminator";

    const int W = std::max(1, cfg.image_w);
    const int H = std::max(1, cfg.image_h);
    const int C = std::max(1, cfg.image_c);
    const int base = std::max(4, cfg.base_channels);
    const int num_down = std::max(1, cfg.num_down);

    const int image_dim = W * H * C;
    model.modelConfig["task"] = "patch_discriminator";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["base_channels"] = base;
    model.modelConfig["num_down"] = num_down;
    model.modelConfig["input_dim"] = image_dim;

    auto sat_mul = [](size_t a, size_t b) -> size_t {
        if (a == 0 || b == 0) return 0;
        if (a > (static_cast<size_t>(-1) / b)) return static_cast<size_t>(-1);
        return a * b;
    };

    // Input vector -> HWC -> CHW
    model.push("patch_disc/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("patch_disc/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "patch_disc/in_vec";
    }
    model.push("patch_disc/in_reshape", "Reshape", 0);
    if (auto* R = model.getLayerByName("patch_disc/in_reshape")) {
        R->inputs = {"patch_disc/in_vec"};
        R->output = "patch_disc/in_hwc";
        R->target_shape = {H, W, C};
    }
    model.push("patch_disc/in_to_chw", "Permute", 0);
    if (auto* P = model.getLayerByName("patch_disc/in_to_chw")) {
        P->inputs = {"patch_disc/in_hwc"};
        P->output = "patch_disc/in_chw";
        P->shape = {H, W, C};
        P->permute_dims = {2, 0, 1};
    }

    auto conv_act = [&](const std::string& name,
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
            L->use_bias = true;
        }
        std::string y = out;
        if (act) {
            model.push(name + "/act", "LeakyReLU", 0);
            if (auto* A = model.getLayerByName(name + "/act")) {
                A->inputs = {out};
                A->output = out + "_act";
                A->alpha = 0.2f;
            }
            y = out + "_act";
        }
        return y;
    };

    std::string x = "patch_disc/in_chw";
    int cur_c = C;
    int cur_h = H;
    int cur_w = W;

    // Stem
    x = conv_act("patch_disc/c1", x, "patch_disc/y1", cur_c, base, cur_h, cur_w, 4, 2, 1, true);
    cur_c = base;
    cur_h = conv_out(cur_h, 4, 2, 1);
    cur_w = conv_out(cur_w, 4, 2, 1);

    for (int i = 0; i < num_down - 1; ++i) {
        const int out_c = std::min(base * 8, cur_c * 2);
        const std::string p = "patch_disc/d" + std::to_string(i + 1);
        x = conv_act(p + "/conv", x, p + "/y", cur_c, out_c, cur_h, cur_w, 4, 2, 1, true);
        cur_c = out_c;
        cur_h = conv_out(cur_h, 4, 2, 1);
        cur_w = conv_out(cur_w, 4, 2, 1);
    }

    // Head: stride 1
    x = conv_act("patch_disc/head", x, "patch_disc/head_y", cur_c, cur_c, cur_h, cur_w, 3, 1, 1, true);

    model.push("patch_disc/out_conv", "Conv2d",
               sat_mul(static_cast<size_t>(1), sat_mul(static_cast<size_t>(cur_c), sat_mul(static_cast<size_t>(3), static_cast<size_t>(3)))));
    if (auto* L = model.getLayerByName("patch_disc/out_conv")) {
        L->inputs = {x};
        L->output = "patch_disc/logits";
        L->in_channels = cur_c;
        L->out_channels = 1;
        L->input_height = cur_h;
        L->input_width = cur_w;
        L->kernel_size = 3;
        L->stride = 1;
        L->padding = 1;
        L->use_bias = true;
    }

    // Flatten to vector
    model.push("patch_disc/flat", "Flatten", 0);
    if (auto* F = model.getLayerByName("patch_disc/flat")) {
        F->inputs = {"patch_disc/logits"};
        F->output = "x";
    }

    const int out_dim = std::max(1, cur_h * cur_w);
    model.modelConfig["output_dim"] = out_dim;
    model.modelConfig["bbox_output_w"] = cur_w;
    model.modelConfig["bbox_output_h"] = cur_h;
    model.modelConfig["bbox_output_c"] = 1;
    model.modelConfig["bbox_output_layout"] = "CHW";
}
