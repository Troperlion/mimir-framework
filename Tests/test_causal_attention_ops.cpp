#include "Models/NLP/CausalAttentionOps.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    constexpr int seq = 3, model = 8, q_heads = 4, kv_heads = 2;
    constexpr int head_dim = model / q_heads;
    constexpr int kv_dim = kv_heads * head_dim;
    const size_t weight_count =
        static_cast<size_t>(model) * (2 * model + 2 * kv_dim);
    std::vector<float> x(seq * model), weights(weight_count), dy(seq * model);
    for (size_t i = 0; i < x.size(); ++i) x[i] = 0.03f * static_cast<float>(i + 1);
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] = 0.01f * static_cast<float>(static_cast<int>(i % 11) - 5);
    for (size_t i = 0; i < dy.size(); ++i) dy[i] = 0.02f * static_cast<float>(i % 7);

    std::vector<float> output, dx, dweights(weight_count, 0.0f);
    if (!CausalAttentionOps::run(x, dy, seq, model, q_heads, kv_heads, true,
                                 10000.0f, weights.data(), dweights.data(), output, &dx))
        return 1;
    if (output.size() != x.size() || dx.size() != x.size()) return 2;

    const size_t probe = 17;
    const float epsilon = 1e-3f;
    auto loss = [&](float delta) {
        weights[probe] += delta;
        std::vector<float> y;
        const bool ok = CausalAttentionOps::run(
            x, {}, seq, model, q_heads, kv_heads, true, 10000.0f,
            weights.data(), nullptr, y, nullptr);
        weights[probe] -= delta;
        if (!ok) return 0.0f;
        float value = 0.0f;
        for (size_t i = 0; i < y.size(); ++i) value += y[i] * dy[i];
        return value;
    };
    const float numerical = (loss(epsilon) - loss(-epsilon)) / (2.0f * epsilon);
    if (std::fabs(numerical - dweights[probe]) > 2e-3f) {
        std::cerr << "gradient mismatch analytical=" << dweights[probe]
                  << " numerical=" << numerical << '\n';
        return 3;
    }
    std::cout << "PASS GQA+RoPE forward/backward numerical gradient\n";
    return 0;
}
