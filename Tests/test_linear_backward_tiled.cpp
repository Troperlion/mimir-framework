#include "Model.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void close(float actual, float expected) {
    if (std::abs(actual - expected) > 2e-5f * std::max(1.0f, std::abs(expected)))
        throw std::runtime_error("Linear backward differs from scalar reference");
}

static void check(int batch, int in, int out, bool bias, bool explicit_sequence) {
    Model model;
    model.push("linear", "Linear", 0);
    auto& layer = model.getMutableLayers().front();
    layer.in_features = in;
    layer.out_features = out;
    layer.seq_len = explicit_sequence ? batch : 0;
    layer.use_bias = bias;
    const size_t weight_count = size_t(in) * out;
    layer.params_count = weight_count + (bias ? out : 0);
    model.allocateParams();
    float* weights = layer.getWeights();
    for (size_t i = 0; i < layer.getWeightsSize(); ++i)
        weights[i] = float(int(i % 23) - 11) / 23;
    std::vector<float> x(size_t(batch) * in), go(size_t(batch) * out);
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i % 29) - 14) / 29;
    for (size_t i = 0; i < go.size(); ++i) go[i] = float(int(i % 17) - 8) / 17;
    std::vector<float> gx(x.size()), gw(weight_count), gb(bias ? out : 0);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < in; ++i)
            for (int o = 0; o < out; ++o)
                gx[size_t(b) * in + i] += go[size_t(b) * out + o] * weights[size_t(o) * in + i];
    for (int o = 0; o < out; ++o)
        for (int i = 0; i < in; ++i)
            for (int b = 0; b < batch; ++b)
                gw[size_t(o) * in + i] += go[size_t(b) * out + o] * x[size_t(b) * in + i];
    if (bias)
        for (int b = 0; b < batch; ++b)
            for (int o = 0; o < out; ++o) gb[o] += go[size_t(b) * out + o];

    layer.grad_weights.assign(layer.getWeightsSize(), .25f);
    layer.grad_bias.assign(bias ? out : 0, .125f);
    CpuRuntime runtime;
    runtime.initialize(RuntimeConfig{});
    std::vector<std::vector<float>> gradients;
    for (int repeat = 1; repeat <= 2; ++repeat) {
        if (!runtime.backwardLayer({&x}, {&go}, gradients, layer, true))
            throw std::runtime_error("Linear backward rejected valid dimensions");
        if (gradients.size() != 1 || gradients[0].size() != gx.size())
            throw std::runtime_error("Wrong input gradient shape");
        for (size_t i = 0; i < gx.size(); ++i) close(gradients[0][i], gx[i]);
        for (size_t i = 0; i < gw.size(); ++i) close(layer.grad_weights[i], .25f + repeat * gw[i]);
        for (size_t i = 0; i < gb.size(); ++i) {
            close(layer.grad_weights[weight_count + i], .25f + repeat * gb[i]);
            close(layer.grad_bias[i], .125f + repeat * gb[i]);
        }
    }
}

int main() {
    for (bool bias : {false, true})
        for (bool sequence : {false, true}) {
            check(1, 3, 2, bias, sequence);
            check(7, 65, 11, bias, sequence);
            check(33, 129, 67, bias, sequence);
            check(4, 257, 3, bias, sequence);
        }
    std::cout << "LINEAR_BACKWARD_TILED_OK\n";
}
