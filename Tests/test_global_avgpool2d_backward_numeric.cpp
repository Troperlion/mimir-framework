#include "test_utils.hpp"

#include "LayerOps.hpp"
#include "Layers.hpp"

#include <cmath>
#include <random>
#include <vector>

static float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

int main() {
    // Small but non-trivial shape (non-square spatial) to avoid accidental assumptions.
    const int C = 5;
    const int H = 4;
    const int W = 7;
    const size_t N = static_cast<size_t>(C) * static_cast<size_t>(H) * static_cast<size_t>(W);

    Layer layer;
    layer.type = "GlobalAvgPool2d";
    layer.type_enum = LayerType::GlobalAvgPool2d;
    layer.in_channels = C;
    layer.input_height = H;
    layer.input_width = W;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> x(N);
    for (auto& v : x) v = dist(rng);

    std::vector<float> grad_out(static_cast<size_t>(C));
    for (auto& v : grad_out) v = dist(rng);

    const auto y = LayerOps::global_avgpool2d_forward(x, layer);
    TASSERT_TRUE(y.size() == static_cast<size_t>(C));

    // Define a scalar loss: L = dot(y, grad_out).
    const float L0 = dot(y, grad_out);
    (void)L0;

    const auto grad_in = LayerOps::global_avgpool2d_backward(grad_out, layer, x.size());
    TASSERT_TRUE(grad_in.size() == x.size());

    // Analytical expectation: grad_in[c,h,w] = grad_out[c] / (H*W)
    const float inv = 1.0f / static_cast<float>(H * W);
    for (int c = 0; c < C; ++c) {
        const float expected = grad_out[static_cast<size_t>(c)] * inv;
        const size_t base = static_cast<size_t>(c) * static_cast<size_t>(H) * static_cast<size_t>(W);
        for (size_t i = 0; i < static_cast<size_t>(H) * static_cast<size_t>(W); ++i) {
            TASSERT_NEAR(grad_in[base + i], expected, 1e-6f);
        }
    }

    // Numerical gradient check (finite differences) on a subset of elements.
    const float eps = 1e-3f;
    const float tol = 2e-3f; // float + finite diff tolerance

    for (size_t i = 0; i < x.size(); i += 3) { // sample every 3rd element
        std::vector<float> xp = x;
        std::vector<float> xm = x;
        xp[i] += eps;
        xm[i] -= eps;

        const auto yp = LayerOps::global_avgpool2d_forward(xp, layer);
        const auto ym = LayerOps::global_avgpool2d_forward(xm, layer);

        const float Lp = dot(yp, grad_out);
        const float Lm = dot(ym, grad_out);
        const float num = (Lp - Lm) / (2.0f * eps);

        TASSERT_NEAR(num, grad_in[i], tol);
    }

    return 0;
}
