#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace CausalAttentionOps {

inline void rope(float* x, int seq, int heads, int dim, float theta, bool inverse = false) {
    if (dim % 2 != 0) throw std::runtime_error("RoPE requires an even head dimension");
    for (int t = 0; t < seq; ++t) {
        for (int h = 0; h < heads; ++h) {
            float* row = x + (t * heads + h) * dim;
            for (int d = 0; d < dim; d += 2) {
                const float angle = static_cast<float>(t) /
                    std::pow(theta, static_cast<float>(d) / static_cast<float>(dim));
                const float c = std::cos(angle);
                const float s = (inverse ? -1.0f : 1.0f) * std::sin(angle);
                const float a = row[d], b = row[d + 1];
                row[d] = a * c - b * s;
                row[d + 1] = a * s + b * c;
            }
        }
    }
}

// Weight layout: Wq[model,model], Wk[kv_dim,model], Wv[kv_dim,model],
// Wo[model,model], each stored output-major like Mimir Linear.
inline bool run(const std::vector<float>& x, const std::vector<float>& dy,
                int seq, int model, int q_heads, int kv_heads, bool causal,
                float theta, const float* weights, float* dweights,
                std::vector<float>& output, std::vector<float>* dx) {
    if (seq <= 0 || model <= 0 || q_heads <= 0 || kv_heads <= 0 ||
        model % q_heads != 0 || q_heads % kv_heads != 0 ||
        x.size() != static_cast<size_t>(seq * model)) return false;
    const bool backward = dx != nullptr;
    if (backward && dy.size() != x.size()) return false;
    const int hd = model / q_heads;
    const int kv_dim = kv_heads * hd;
    const size_t qn = static_cast<size_t>(model) * model;
    const size_t kn = static_cast<size_t>(kv_dim) * model;
    const float* Wq = weights;
    const float* Wk = Wq + qn;
    const float* Wv = Wk + kn;
    const float* Wo = Wv + kn;
    float* dWq = dweights;
    float* dWk = backward ? dWq + qn : nullptr;
    float* dWv = backward ? dWk + kn : nullptr;
    float* dWo = backward ? dWv + kn : nullptr;

    auto project = [&](const float* w, int out_dim, std::vector<float>& out) {
        out.assign(static_cast<size_t>(seq) * out_dim, 0.0f);
        for (int t = 0; t < seq; ++t)
            for (int o = 0; o < out_dim; ++o)
                for (int i = 0; i < model; ++i)
                    out[static_cast<size_t>(t) * out_dim + o] +=
                        x[static_cast<size_t>(t) * model + i] *
                        w[static_cast<size_t>(o) * model + i];
    };
    std::vector<float> q, k, v;
    project(Wq, model, q); project(Wk, kv_dim, k); project(Wv, kv_dim, v);
    if (theta > 0.0f) {
        rope(q.data(), seq, q_heads, hd, theta);
        rope(k.data(), seq, kv_heads, hd, theta);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    std::vector<float> probs(static_cast<size_t>(q_heads) * seq * seq, 0.0f);
    std::vector<float> context(static_cast<size_t>(seq) * model, 0.0f);
    for (int h = 0; h < q_heads; ++h) {
        const int kh = h / (q_heads / kv_heads);
        for (int i = 0; i < seq; ++i) {
            float max_score = -1e30f;
            for (int j = 0; j < seq; ++j) {
                float score = -1e9f;
                if (!causal || j <= i) {
                    score = 0.0f;
                    for (int d = 0; d < hd; ++d)
                        score += q[(i * q_heads + h) * hd + d] *
                                 k[(j * kv_heads + kh) * hd + d];
                    score *= scale;
                }
                const size_t p = (static_cast<size_t>(h) * seq + i) * seq + j;
                probs[p] = score;
                max_score = std::max(max_score, score);
            }
            float sum = 0.0f;
            for (int j = 0; j < seq; ++j) {
                const size_t p = (static_cast<size_t>(h) * seq + i) * seq + j;
                probs[p] = std::exp(probs[p] - max_score);
                sum += probs[p];
            }
            for (int j = 0; j < seq; ++j) {
                const size_t p = (static_cast<size_t>(h) * seq + i) * seq + j;
                probs[p] /= sum;
                for (int d = 0; d < hd; ++d)
                    context[(i * q_heads + h) * hd + d] +=
                        probs[p] * v[(j * kv_heads + kh) * hd + d];
            }
        }
    }
    output.assign(static_cast<size_t>(seq) * model, 0.0f);
    for (int t = 0; t < seq; ++t)
        for (int o = 0; o < model; ++o)
            for (int i = 0; i < model; ++i)
                output[static_cast<size_t>(t) * model + o] +=
                    context[static_cast<size_t>(t) * model + i] *
                    Wo[static_cast<size_t>(o) * model + i];
    if (!backward) return true;

    std::vector<float> dc(context.size(), 0.0f), dq(q.size(), 0.0f);
    std::vector<float> dk(k.size(), 0.0f), dv(v.size(), 0.0f);
    for (int t = 0; t < seq; ++t) for (int o = 0; o < model; ++o) {
        const float g = dy[static_cast<size_t>(t) * model + o];
        for (int i = 0; i < model; ++i) {
            dWo[static_cast<size_t>(o) * model + i] +=
                g * context[static_cast<size_t>(t) * model + i];
            dc[static_cast<size_t>(t) * model + i] +=
                g * Wo[static_cast<size_t>(o) * model + i];
        }
    }
    for (int h = 0; h < q_heads; ++h) {
        const int kh = h / (q_heads / kv_heads);
        std::vector<float> dp(static_cast<size_t>(seq) * seq, 0.0f);
        for (int i = 0; i < seq; ++i) for (int j = 0; j < seq; ++j) {
            for (int d = 0; d < hd; ++d) {
                const float g = dc[(i * q_heads + h) * hd + d];
                dp[static_cast<size_t>(i) * seq + j] +=
                    g * v[(j * kv_heads + kh) * hd + d];
                dv[(j * kv_heads + kh) * hd + d] +=
                    probs[(static_cast<size_t>(h) * seq + i) * seq + j] * g;
            }
        }
        for (int i = 0; i < seq; ++i) {
            float dot = 0.0f;
            for (int j = 0; j < seq; ++j)
                dot += dp[static_cast<size_t>(i) * seq + j] *
                       probs[(static_cast<size_t>(h) * seq + i) * seq + j];
            for (int j = 0; j < seq; ++j) {
                if (causal && j > i) continue;
                const float ds = (dp[static_cast<size_t>(i) * seq + j] - dot) *
                    probs[(static_cast<size_t>(h) * seq + i) * seq + j] * scale;
                for (int d = 0; d < hd; ++d) {
                    dq[(i * q_heads + h) * hd + d] +=
                        ds * k[(j * kv_heads + kh) * hd + d];
                    dk[(j * kv_heads + kh) * hd + d] +=
                        ds * q[(i * q_heads + h) * hd + d];
                }
            }
        }
    }
    if (theta > 0.0f) {
        rope(dq.data(), seq, q_heads, hd, theta, true);
        rope(dk.data(), seq, kv_heads, hd, theta, true);
    }
    dx->assign(x.size(), 0.0f);
    auto projection_backward = [&](const std::vector<float>& grad, const float* w,
                                   float* dw, int out_dim) {
        for (int t = 0; t < seq; ++t) for (int o = 0; o < out_dim; ++o) {
            const float g = grad[static_cast<size_t>(t) * out_dim + o];
            for (int i = 0; i < model; ++i) {
                dw[static_cast<size_t>(o) * model + i] +=
                    g * x[static_cast<size_t>(t) * model + i];
                (*dx)[static_cast<size_t>(t) * model + i] +=
                    g * w[static_cast<size_t>(o) * model + i];
            }
        }
    };
    projection_backward(dq, Wq, dWq, model);
    projection_backward(dk, Wk, dWk, kv_dim);
    projection_backward(dv, Wv, dWv, kv_dim);
    return true;
}

} // namespace CausalAttentionOps
