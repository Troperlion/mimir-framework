#include "test_utils.hpp"

#include "Layers.hpp"
#include "runtimes/LayerOps.hpp"
#include "runtimes/cpu/RuntimeLayerDispatch.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {

float dot(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::quiet_NaN();
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
    return result;
}

std::vector<float> forward(
    const std::vector<std::vector<float>>& owned_inputs,
    const Layer& layer
) {
    std::vector<const std::vector<float>*> inputs;
    for (const auto& input : owned_inputs) inputs.push_back(&input);
    std::vector<std::vector<float>> outputs;
    const bool ok = RuntimeLayerDispatch::cpu_forward_layer(inputs, outputs, layer, true);
    if (!ok || outputs.size() != 1) return {};
    return outputs[0];
}

std::vector<std::vector<float>> backward(
    const std::vector<std::vector<float>>& owned_inputs,
    const std::vector<float>& grad_output,
    Layer& layer
) {
    std::vector<const std::vector<float>*> inputs;
    for (const auto& input : owned_inputs) inputs.push_back(&input);
    const std::vector<const std::vector<float>*> grad_outputs = {&grad_output};
    std::vector<std::vector<float>> grad_inputs;
    const bool ok = RuntimeLayerDispatch::cpu_backward_layer(
        inputs, grad_outputs, grad_inputs, layer, true);
    if (!ok) return {};
    return grad_inputs;
}

bool gradient_check(
    Layer layer,
    std::vector<std::vector<float>> inputs,
    const std::vector<float>& grad_output,
    float tolerance = 2e-3f
) {
    const auto analytical = backward(inputs, grad_output, layer);
    if (analytical.size() != inputs.size()) return false;

    constexpr float h = 1e-3f;
    for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
        if (analytical[input_idx].size() != inputs[input_idx].size()) return false;
        for (size_t i = 0; i < inputs[input_idx].size(); ++i) {
            const float original = inputs[input_idx][i];
            inputs[input_idx][i] = original + h;
            const auto plus = forward(inputs, layer);
            inputs[input_idx][i] = original - h;
            const auto minus = forward(inputs, layer);
            inputs[input_idx][i] = original;
            if (plus.size() != grad_output.size() || minus.size() != grad_output.size()) return false;
            const float numerical = (dot(plus, grad_output) - dot(minus, grad_output)) / (2.0f * h);
            if (!std::isfinite(numerical) ||
                std::fabs(numerical - analytical[input_idx][i]) > tolerance) {
                std::cerr << "gradient mismatch input=" << input_idx << " element=" << i
                          << " analytical=" << analytical[input_idx][i]
                          << " numerical=" << numerical << '\n';
                return false;
            }
        }
    }
    return true;
}

Layer elementwise_layer(LayerType type) {
    Layer layer;
    layer.type_enum = type;
    layer.type = LayerRegistry::type_to_string(type);
    return layer;
}

} // namespace

int main() {
    const std::vector<float> x = {-2.0f, -0.7f, 0.4f, 2.0f};
    const std::vector<float> go = {0.3f, -0.8f, 1.1f, 0.5f};

    for (const LayerType type : {
            LayerType::LeakyReLU, LayerType::Sigmoid, LayerType::Tanh,
            LayerType::SiLU, LayerType::GELU, LayerType::Softplus,
            LayerType::Mish, LayerType::HardSigmoid, LayerType::HardSwish}) {
        TASSERT_TRUE(gradient_check(elementwise_layer(type), {x}, go));
    }

    const std::vector<float> a = {-1.2f, 0.5f, 2.0f, -0.8f};
    const std::vector<float> b = {0.7f, -1.5f, 2.5f, -0.4f};
    for (const LayerType type : {
            LayerType::Add, LayerType::Subtract, LayerType::Multiply, LayerType::Divide}) {
        TASSERT_TRUE(gradient_check(elementwise_layer(type), {a, b}, go));
    }

    {
        Layer layer = elementwise_layer(LayerType::LayerNorm);
        layer.in_features = 4;
        layer.eps = 1e-5f;
        layer.affine = false;
        TASSERT_TRUE(gradient_check(layer, {{-1.0f, 0.2f, 1.7f, 2.3f}}, go, 4e-3f));
    }

    {
        Layer layer = elementwise_layer(LayerType::RMSNorm);
        layer.in_features = 4;
        layer.eps = 1e-5f;
        layer.affine = false;
        TASSERT_TRUE(gradient_check(layer, {{-1.0f, 0.2f, 1.7f, 2.3f}}, go, 4e-3f));
    }

    {
        Layer layer = elementwise_layer(LayerType::SelfAttention);
        layer.seq_len = 2;
        layer.embed_dim = 2;
        layer.num_heads = 1;
        layer.causal = false;
        layer.params_count = 16;
        layer.weights.assign(16, 0.0f);
        // Wqkv layout is [3 * embed_dim, input_dim], followed by Wout.
        for (int block = 0; block < 3; ++block) {
            layer.weights[static_cast<size_t>(block * 4)] = 1.0f;
            layer.weights[static_cast<size_t>(block * 4 + 3)] = 1.0f;
        }
        layer.weights[12] = layer.weights[15] = 1.0f;
        TASSERT_TRUE(gradient_check(
            layer,
            {{0.8f, -0.3f, 0.2f, 1.1f}},
            {0.4f, -0.7f, 1.2f, 0.3f},
            7e-3f));
    }

    {
        const std::vector<float> extremes = {-100.0f, -25.0f, 25.0f, 100.0f};
        for (const LayerType type : {LayerType::Softplus, LayerType::Mish}) {
            const auto y = forward({extremes}, elementwise_layer(type));
            TASSERT_TRUE(y.size() == extremes.size());
            for (float value : y) TASSERT_TRUE(std::isfinite(value));
        }
    }

    {
        std::vector<float> y;
        RuntimeLayerOps::unaryForwardHost({-3.0f, -2.5f, 0.0f, 2.5f, 3.0f}, y, 8, 0.0f);
        TASSERT_NEAR(y[1], 1.0f / 12.0f, 1e-6f);
        TASSERT_NEAR(y[3], 11.0f / 12.0f, 1e-6f);
    }

    std::cout << "autograd numerical checks: OK\n";
    return 0;
}
