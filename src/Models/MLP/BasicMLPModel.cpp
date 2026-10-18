#include "BasicMLPModel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

BasicMLPModel::BasicMLPModel() {
    setModelName("BasicMLPModel");
    setHasEncoder(false);
}

void BasicMLPModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;
    buildInto(*this, cfg_);
}

BasicMLPModel::StepStats BasicMLPModel::trainStep(const std::vector<float>& input,
                                                  const std::vector<float>& target,
                                                  Optimizer& opt,
                                                  float learning_rate) {
    if (layers.empty()) {
        throw std::runtime_error("BasicMLPModel::trainStep: model not built");
    }
    if (layer_weight_blocks.empty()) {
        throw std::runtime_error("BasicMLPModel::trainStep: weights not allocated (call allocateParams/initWeights)");
    }

    zeroGradients();

    const std::vector<float>& prediction = forwardPassView(input, true);

    StepStats stats;
    stats.loss = computeLoss(prediction, target, "mse");

    static thread_local std::vector<float> loss_grad;
    computeLossGradientInto(prediction, target, loss_grad, "mse");
    backwardPass(loss_grad);

    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (const auto& layer : layers) {
        for (float g : layer.grad_weights) {
            sum_sq += static_cast<double>(g) * static_cast<double>(g);
            const float a = std::abs(g);
            if (a > max_abs) max_abs = a;
        }
    }
    stats.grad_norm = static_cast<float>(std::sqrt(sum_sq));
    stats.grad_max_abs = max_abs;

    optimizerStep(opt, learning_rate);
    return stats;
}

std::optional<Model::TrainStepResult> BasicMLPModel::trainStep(const TrainStepRequest& request) {
    if (request.optimizer == nullptr || request.target == nullptr) {
        return std::nullopt;
    }

    const std::vector<float>* input = nullptr;
    if (const auto it = request.float_inputs.find("__input__"); it != request.float_inputs.end()) {
        input = it->second;
    } else if (const auto it = request.float_inputs.find("input"); it != request.float_inputs.end()) {
        input = it->second;
    }
    if (input == nullptr) {
        return std::nullopt;
    }

    if (request.mode == TrainStepMode::Optimize) {
        const StepStats stats = trainStep(*input, *request.target, *request.optimizer, request.learning_rate);
        TrainStepResult result;
        result.loss = stats.loss;
        result.grad_norm = stats.grad_norm;
        result.grad_max_abs = stats.grad_max_abs;
        result.metrics["mse"] = stats.loss;
        return result;
    }

    if (layers.empty()) {
        throw std::runtime_error("BasicMLPModel::trainStep: model not built");
    }
    if (layer_weight_blocks.empty()) {
        throw std::runtime_error("BasicMLPModel::trainStep: weights not allocated (call allocateParams/initWeights)");
    }

    const std::vector<float>& prediction = forwardPassView(*input, true);
    TrainStepResult result;
    result.loss = computeLoss(prediction, *request.target, "mse");
    result.metrics["mse"] = result.loss;

    static thread_local std::vector<float> loss_grad;
    computeLossGradientInto(prediction, *request.target, loss_grad, "mse");
    if (request.grad_scale != 1.0f) {
        for (float& gradient : loss_grad) gradient *= request.grad_scale;
    }
    backwardPass(loss_grad);

    double sum_sq = 0.0;
    for (const auto& layer : layers) {
        for (float gradient : layer.grad_weights) {
            sum_sq += static_cast<double>(gradient) * static_cast<double>(gradient);
            result.grad_max_abs = std::max(result.grad_max_abs, std::abs(gradient));
        }
    }
    result.grad_norm = static_cast<float>(std::sqrt(sum_sq));
    return result;
}

void BasicMLPModel::buildInto(Model& model, const Config& cfg) {
    model.getMutableLayers().clear();
    model.setModelName("BasicMLPModel");
    model.modelConfig["type"] = "basic_mlp";

    const int in_dim = std::max(1, cfg.input_dim);
    const int hidden = std::max(1, cfg.hidden_dim);
    const int out_dim = std::max(1, cfg.output_dim);
    const int blocks = std::max(0, cfg.hidden_layers);
    const float drop = std::clamp(cfg.dropout, 0.0f, 0.95f);

    model.modelConfig["task"] = "mlp_regression";
    model.modelConfig["input_dim"] = in_dim;
    model.modelConfig["hidden_dim"] = hidden;
    model.modelConfig["output_dim"] = out_dim;
    model.modelConfig["hidden_layers"] = blocks;
    model.modelConfig["dropout"] = drop;

    // Routing input: __input__ -> basic/in
    model.push("basic_mlp/raw_in", "Identity", 0);
    if (auto* L = model.getLayerByName("basic_mlp/raw_in")) {
        L->inputs = {"__input__"};
        L->output = "basic_mlp/in";
    }

    int cur_dim = in_dim;
    std::string cur_tensor = "basic_mlp/in";

    for (int i = 0; i < blocks; ++i) {
        const std::string fc = "basic_mlp/fc" + std::to_string(i + 1);
        const std::string h = "basic_mlp/h" + std::to_string(i + 1);
        const std::string act = h + "_act";

        model.push(fc, "Linear", static_cast<size_t>(cur_dim) * static_cast<size_t>(hidden) + static_cast<size_t>(hidden));
        if (auto* L = model.getLayerByName(fc)) {
            L->inputs = {cur_tensor};
            L->output = h;
            L->in_features = cur_dim;
            L->out_features = hidden;
            L->use_bias = true;
        }

        model.push(fc + "/gelu", "GELU", 0);
        if (auto* L = model.getLayerByName(fc + "/gelu")) {
            L->inputs = {h};
            L->output = act;
        }

        if (drop > 0.0f) {
            const std::string dn = fc + "/drop";
            const std::string dout = act + "_drop";
            model.push(dn, "Dropout", 0);
            if (auto* L = model.getLayerByName(dn)) {
                L->inputs = {act};
                L->output = dout;
                L->dropout_p = drop;
            }
            cur_tensor = dout;
        } else {
            cur_tensor = act;
        }

        cur_dim = hidden;
    }

    model.push("basic_mlp/out", "Linear", static_cast<size_t>(cur_dim) * static_cast<size_t>(out_dim) + static_cast<size_t>(out_dim));
    if (auto* L = model.getLayerByName("basic_mlp/out")) {
        L->inputs = {cur_tensor};
        L->output = "x";
        L->in_features = cur_dim;
        L->out_features = out_dim;
        L->use_bias = true;
    }
}
