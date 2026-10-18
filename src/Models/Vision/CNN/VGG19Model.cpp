#include "VGG19Model.hpp"

#include <algorithm>

bool VGG19Model::InitVizTips() {
    clearVizTipsRegistry();
    clearVizTaps();

    registerVizTip("vgg19/raw_in",   "Dataset/raw");
    registerVizTip("vgg19/conv1_2",  "Block1/conv");
    registerVizTip("vgg19/pool1",    "Block1/out");
    registerVizTip("vgg19/conv2_2",  "Block2/conv");
    registerVizTip("vgg19/pool2",    "Block2/out");
    registerVizTip("vgg19/conv3_4",  "Block3/conv");
    registerVizTip("vgg19/pool3",    "Block3/out");
    registerVizTip("vgg19/conv4_4",  "Block4/conv");
    registerVizTip("vgg19/pool4",    "Block4/out");
    registerVizTip("vgg19/conv5_4",  "Block5/conv");
    registerVizTip("vgg19/pool5",    "Block5/out");
    registerVizTip("vgg19/flatten",  "Head/flatten");
    registerVizTip("vgg19/fc1_relu", "Head/fc1");
    registerVizTip("vgg19/fc2_relu", "Head/fc2");
    registerVizTip("vgg19/head",     "Output/logits");
    registerVizTip("vgg19/softmax",  "Output/probabilities");

    return true;
}

bool VGG19Model::UpdateVizTips(const Layer& layer, VizFrame& frame) {
    if (Model::UpdateVizTips(layer, frame)) return true;
    if (layer.name.empty()) return false;

    auto attach_tip = [&](const std::string& tip) -> bool {
        if (tip.empty()) return false;
        frame.label = tip + "|" + frame.label;
        return true;
    };

    const std::string& n = layer.name;

    auto attach_block = [&](int block) -> bool {
        const std::string prefix = "Block" + std::to_string(block);
        if (n.find("pool" + std::to_string(block)) != std::string::npos) return attach_tip(prefix + "/pool");
        if (n.find("relu" + std::to_string(block) + "_") != std::string::npos) return attach_tip(prefix + "/act");
        if (n.find("conv" + std::to_string(block) + "_") != std::string::npos) return attach_tip(prefix + "/conv");
        return false;
    };

    if (n.find("vgg19/") == 0) {
        if (attach_block(1)) return true;
        if (attach_block(2)) return true;
        if (attach_block(3)) return true;
        if (attach_block(4)) return true;
        if (attach_block(5)) return true;
    }

    if (n.find("/flatten") != std::string::npos) return attach_tip("Head/flatten");
    if (n.find("/drop")    != std::string::npos) return attach_tip("Head/dropout");
    if (n.find("/fc2")     != std::string::npos) return attach_tip("Head/fc2");
    if (n.find("/fc1")     != std::string::npos) return attach_tip("Head/fc1");
    if (n == "vgg19/head") return attach_tip("Output/logits");
    if (n == "vgg19/softmax") return attach_tip("Output/probabilities");

    return false;
}

VGG19Model::VGG19Model() {
    setModelName("VGG19Model");
    setHasEncoder(false);
}

void VGG19Model::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

static inline int pool_out(int in, int k, int s, int p) {
    return (in + 2 * p - k) / s + 1;
}

void VGG19Model::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("VGG19Model");
    model.modelConfig["type"] = "vgg19";

    int W = std::max(1, cfg.image_w);
    int H = std::max(1, cfg.image_h);
    int C = std::max(1, cfg.image_c);
    const int base = 64;
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

    model.push("vgg19/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("vgg19/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "vgg19/in";
    }

    auto add_conv_relu = [&](const std::string& conv_name,
                             const std::string& relu_name,
                             const std::string& in,
                             const std::string& conv_out_name,
                             const std::string& relu_out_name,
                             int in_c,
                             int out_c,
                             int in_h,
                             int in_w) {
        model.push(conv_name, "Conv2d", static_cast<size_t>(out_c) * static_cast<size_t>(in_c) * 9u + static_cast<size_t>(out_c));
        if (auto* L = model.getLayerByName(conv_name)) {
            L->inputs = {in};
            L->output = conv_out_name;
            L->in_channels = in_c;
            L->out_channels = out_c;
            L->input_height = in_h;
            L->input_width = in_w;
            L->kernel_size = 3;
            L->stride = 1;
            L->padding = 1;
            L->use_bias = true;
        }

        model.push(relu_name, "ReLU", 0);
        if (auto* R = model.getLayerByName(relu_name)) {
            R->inputs = {conv_out_name};
            R->output = relu_out_name;
        }
        return relu_out_name;
    };

    auto add_pool = [&](const std::string& pool_name,
                        const std::string& in,
                        const std::string& out,
                        int in_c,
                        int in_h,
                        int in_w,
                        int& out_h,
                        int& out_w) {
        out_h = pool_out(in_h, 2, 2, 0);
        out_w = pool_out(in_w, 2, 2, 0);
        model.push(pool_name, "MaxPool2d", 0);
        if (auto* P = model.getLayerByName(pool_name)) {
            P->inputs = {in};
            P->output = out;
            P->in_channels = in_c;
            P->input_height = in_h;
            P->input_width = in_w;
            P->kernel_size = 2;
            P->stride = 2;
            P->padding = 0;
        }
    };

    std::string x = "vgg19/in";
    int cur_c = C;
    int cur_h = H;
    int cur_w = W;

    // VGG19 canonical channels per block.
    const int c1 = 64;
    const int c2 = 128;
    const int c3 = 256;
    const int c4 = 512;
    const int c5 = 512;

    // Block 1
    x = add_conv_relu("vgg19/conv1_1", "vgg19/relu1_1", x, "vgg19/conv1_1_y", "vgg19/relu1_1_y", cur_c, c1, cur_h, cur_w);
    cur_c = c1;
    x = add_conv_relu("vgg19/conv1_2", "vgg19/relu1_2", x, "vgg19/conv1_2_y", "vgg19/relu1_2_y", cur_c, c1, cur_h, cur_w);
    cur_c = c1;
    add_pool("vgg19/pool1", x, "vgg19/pool1_y", cur_c, cur_h, cur_w, cur_h, cur_w);
    x = "vgg19/pool1_y";

    // Block 2
    x = add_conv_relu("vgg19/conv2_1", "vgg19/relu2_1", x, "vgg19/conv2_1_y", "vgg19/relu2_1_y", cur_c, c2, cur_h, cur_w);
    cur_c = c2;
    x = add_conv_relu("vgg19/conv2_2", "vgg19/relu2_2", x, "vgg19/conv2_2_y", "vgg19/relu2_2_y", cur_c, c2, cur_h, cur_w);
    cur_c = c2;
    add_pool("vgg19/pool2", x, "vgg19/pool2_y", cur_c, cur_h, cur_w, cur_h, cur_w);
    x = "vgg19/pool2_y";

    // Block 3
    x = add_conv_relu("vgg19/conv3_1", "vgg19/relu3_1", x, "vgg19/conv3_1_y", "vgg19/relu3_1_y", cur_c, c3, cur_h, cur_w);
    cur_c = c3;
    x = add_conv_relu("vgg19/conv3_2", "vgg19/relu3_2", x, "vgg19/conv3_2_y", "vgg19/relu3_2_y", cur_c, c3, cur_h, cur_w);
    x = add_conv_relu("vgg19/conv3_3", "vgg19/relu3_3", x, "vgg19/conv3_3_y", "vgg19/relu3_3_y", cur_c, c3, cur_h, cur_w);
    x = add_conv_relu("vgg19/conv3_4", "vgg19/relu3_4", x, "vgg19/conv3_4_y", "vgg19/relu3_4_y", cur_c, c3, cur_h, cur_w);
    add_pool("vgg19/pool3", x, "vgg19/pool3_y", cur_c, cur_h, cur_w, cur_h, cur_w);
    x = "vgg19/pool3_y";

    // Block 4
    x = add_conv_relu("vgg19/conv4_1", "vgg19/relu4_1", x, "vgg19/conv4_1_y", "vgg19/relu4_1_y", cur_c, c4, cur_h, cur_w);
    cur_c = c4;
    x = add_conv_relu("vgg19/conv4_2", "vgg19/relu4_2", x, "vgg19/conv4_2_y", "vgg19/relu4_2_y", cur_c, c4, cur_h, cur_w);
    x = add_conv_relu("vgg19/conv4_3", "vgg19/relu4_3", x, "vgg19/conv4_3_y", "vgg19/relu4_3_y", cur_c, c4, cur_h, cur_w);
    x = add_conv_relu("vgg19/conv4_4", "vgg19/relu4_4", x, "vgg19/conv4_4_y", "vgg19/relu4_4_y", cur_c, c4, cur_h, cur_w);
    add_pool("vgg19/pool4", x, "vgg19/pool4_y", cur_c, cur_h, cur_w, cur_h, cur_w);
    x = "vgg19/pool4_y";

    // Block 5
    x = add_conv_relu("vgg19/conv5_1", "vgg19/relu5_1", x, "vgg19/conv5_1_y", "vgg19/relu5_1_y", cur_c, c5, cur_h, cur_w);
    cur_c = c5;
    x = add_conv_relu("vgg19/conv5_2", "vgg19/relu5_2", x, "vgg19/conv5_2_y", "vgg19/relu5_2_y", cur_c, c5, cur_h, cur_w);
    x = add_conv_relu("vgg19/conv5_3", "vgg19/relu5_3", x, "vgg19/conv5_3_y", "vgg19/relu5_3_y", cur_c, c5, cur_h, cur_w);
    x = add_conv_relu("vgg19/conv5_4", "vgg19/relu5_4", x, "vgg19/conv5_4_y", "vgg19/relu5_4_y", cur_c, c5, cur_h, cur_w);
    add_pool("vgg19/pool5", x, "vgg19/pool5_y", cur_c, cur_h, cur_w, cur_h, cur_w);
    x = "vgg19/pool5_y";

    model.push("vgg19/flatten", "Flatten", 0);
    if (auto* F = model.getLayerByName("vgg19/flatten")) {
        F->inputs = {x};
        F->output = "vgg19/flat_y";
    }

    const int flatten_dim = std::max(1, cur_c * cur_h * cur_w);
    model.modelConfig["flatten_dim"] = flatten_dim;

    std::string cur = "vgg19/flat_y";

    // Dropout + FC1 + ReLU
    if (dropout > 0.0f) {
        model.push("vgg19/drop1", "Dropout", 0);
        if (auto* L = model.getLayerByName("vgg19/drop1")) {
            L->inputs = {cur};
            L->output = "vgg19/drop1_y";
            L->dropout_p = dropout;
        }
        cur = "vgg19/drop1_y";
    }
    model.push("vgg19/fc1", "Linear", static_cast<size_t>(flatten_dim) * static_cast<size_t>(fc_hidden) + static_cast<size_t>(fc_hidden));
    if (auto* L = model.getLayerByName("vgg19/fc1")) {
        L->inputs = {cur};
        L->output = "vgg19/fc1_y";
        L->in_features = flatten_dim;
        L->out_features = fc_hidden;
        L->use_bias = true;
    }
    model.push("vgg19/fc1_relu", "ReLU", 0);
    if (auto* L = model.getLayerByName("vgg19/fc1_relu")) {
        L->inputs = {"vgg19/fc1_y"};
        L->output = "vgg19/fc1_h";
    }
    cur = "vgg19/fc1_h";

    // Dropout + FC2 + ReLU
    if (dropout > 0.0f) {
        model.push("vgg19/drop2", "Dropout", 0);
        if (auto* L = model.getLayerByName("vgg19/drop2")) {
            L->inputs = {cur};
            L->output = "vgg19/drop2_y";
            L->dropout_p = dropout;
        }
        cur = "vgg19/drop2_y";
    }
    model.push("vgg19/fc2", "Linear", static_cast<size_t>(fc_hidden) * static_cast<size_t>(fc_hidden) + static_cast<size_t>(fc_hidden));
    if (auto* L = model.getLayerByName("vgg19/fc2")) {
        L->inputs = {cur};
        L->output = "vgg19/fc2_y";
        L->in_features = fc_hidden;
        L->out_features = fc_hidden;
        L->use_bias = true;
    }
    model.push("vgg19/fc2_relu", "ReLU", 0);
    if (auto* L = model.getLayerByName("vgg19/fc2_relu")) {
        L->inputs = {"vgg19/fc2_y"};
        L->output = "vgg19/fc2_h";
    }

    // Classification head + softmax
    model.push("vgg19/head", "Linear", static_cast<size_t>(fc_hidden) * static_cast<size_t>(num_classes) + static_cast<size_t>(num_classes));
    if (auto* L = model.getLayerByName("vgg19/head")) {
        L->inputs = {"vgg19/fc2_h"};
        L->output = "vgg19/logits";
        L->in_features = fc_hidden;
        L->out_features = num_classes;
        L->use_bias = true;
    }

    model.push("vgg19/softmax", "Softmax", 0);
    if (auto* S = model.getLayerByName("vgg19/softmax")) {
        S->inputs = {"vgg19/logits"};
        S->output = "x";
    }
}
