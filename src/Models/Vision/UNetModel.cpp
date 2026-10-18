#include "UNetModel.hpp"

#include <algorithm>

UNetModel::UNetModel() {
    setModelName("UNetModel");
    setHasEncoder(false);
}

void UNetModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

static inline int conv_out(int in, int k, int s, int p) {
    return (in + 2 * p - k) / s + 1;
}

void UNetModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("UNetModel");
    model.modelConfig["type"] = "unet";

    int W = std::max(1, cfg.image_w);
    int H = std::max(1, cfg.image_h);
    int C = std::max(1, cfg.image_c);
    const int base = std::max(4, cfg.base_channels);
    const int depth = std::max(1, cfg.depth);
    const int image_dim = W * H * C;

    model.modelConfig["task"] = "unet_denoiser";
    model.modelConfig["bbox_output_layout"] = "CHW";
    model.modelConfig["image_w"] = W;
    model.modelConfig["image_h"] = H;
    model.modelConfig["image_c"] = C;
    model.modelConfig["base_channels"] = base;
    model.modelConfig["depth"] = depth;
    model.modelConfig["input_dim"] = image_dim;
    model.modelConfig["output_dim"] = image_dim;

    model.push("unet/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("unet/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "unet/in";
    }

    auto add_conv_relu = [&](const std::string& name,
                             const std::string& in,
                             const std::string& out,
                             int in_c,
                             int out_c,
                             int in_h,
                             int in_w,
                             int k,
                             int s,
                             int p) {
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
        model.push(name + "/relu", "ReLU", 0);
        if (auto* R = model.getLayerByName(name + "/relu")) {
            R->inputs = {out};
            R->output = out + "_act";
        }
        return out + "_act";
    };

    auto add_upsample = [&](const std::string& name,
                            const std::string& in,
                            const std::string& out,
                            int channels,
                            int in_h,
                            int in_w) {
        const int out_h = std::max(1, in_h * 2);
        const int out_w = std::max(1, in_w * 2);
        model.push(name, "UpsampleNearest", 0);
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

    // Down path (store skip tensors)
    std::vector<std::string> skips;
    std::vector<int> skip_c, skip_h, skip_w;

    std::string x = "unet/in";
    int cur_c = C;
    int cur_h = H;
    int cur_w = W;

    for (int d = 0; d < depth; ++d) {
        const int out_c = base * (1 << d);
        const std::string b = "unet/down" + std::to_string(d + 1);

        x = add_conv_relu(b + "/conv1", x, b + "/c1", cur_c, out_c, cur_h, cur_w, 3, 1, 1);
        x = add_conv_relu(b + "/conv2", x, b + "/c2", out_c, out_c, cur_h, cur_w, 3, 1, 1);

        skips.push_back(x);
        skip_c.push_back(out_c);
        skip_h.push_back(cur_h);
        skip_w.push_back(cur_w);

        // Downsample with stride-2 conv
        const std::string ds = b + "/downsample";
        const int nh = conv_out(cur_h, 3, 2, 1);
        const int nw = conv_out(cur_w, 3, 2, 1);
        x = add_conv_relu(ds, x, ds + "/y", out_c, out_c, cur_h, cur_w, 3, 2, 1);
        cur_c = out_c;
        cur_h = nh;
        cur_w = nw;
    }

    // Bottleneck
    {
        const std::string b = "unet/bottleneck";
        const int out_c = base * (1 << depth);
        x = add_conv_relu(b + "/conv1", x, b + "/c1", cur_c, out_c, cur_h, cur_w, 3, 1, 1);
        x = add_conv_relu(b + "/conv2", x, b + "/c2", out_c, out_c, cur_h, cur_w, 3, 1, 1);
        cur_c = out_c;
    }

    // Up path
    for (int d = depth - 1; d >= 0; --d) {
        const std::string b = "unet/up" + std::to_string(d + 1);
        // Upsample
        const int up_h = cur_h;
        const int up_w = cur_w;
        x = add_upsample(b + "/up", x, b + "/up_y", cur_c, up_h, up_w);
        cur_h = up_h * 2;
        cur_w = up_w * 2;

        // Concat skip
        model.push(b + "/concat", "Concat", 0);
        if (auto* L = model.getLayerByName(b + "/concat")) {
            L->skip_input_index = 1;
            L->inputs = {x, skips[static_cast<size_t>(d)]};
            L->output = b + "/cat";
            L->concat_axis = 0;
        }

        // Reduce channels back
        const int out_c = base * (1 << d);
        const int in_cat_c = cur_c + skip_c[static_cast<size_t>(d)];
        x = add_conv_relu(b + "/conv1", b + "/cat", b + "/c1", in_cat_c, out_c, cur_h, cur_w, 3, 1, 1);
        x = add_conv_relu(b + "/conv2", x, b + "/c2", out_c, out_c, cur_h, cur_w, 3, 1, 1);
        cur_c = out_c;
    }

    // Output projection
    model.push("unet/out", "Conv2d", static_cast<size_t>(C) * static_cast<size_t>(cur_c) * 1ULL * 1ULL);
    if (auto* L = model.getLayerByName("unet/out")) {
        L->inputs = {x};
        L->output = "unet/out_pre";
        L->in_channels = cur_c;
        L->out_channels = C;
        L->input_height = H;
        L->input_width = W;
        L->kernel_size = 1;
        L->stride = 1;
        L->padding = 0;
        L->use_bias = false;
    }
    model.push("unet/out_tanh", "Tanh", 0);
    if (auto* L = model.getLayerByName("unet/out_tanh")) {
        L->inputs = {"unet/out_pre"};
        L->output = "x";
    }
}
