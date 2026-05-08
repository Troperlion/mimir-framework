#include "test_utils.hpp"

#include "Model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static void require_or_throw(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

static float dot(const std::vector<float>& a, const std::vector<float>& b) {
    require_or_throw(a.size() == b.size(), "dot: size mismatch");
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    return static_cast<float>(s);
}

static void fill_deterministic(std::vector<float>& v, float scale = 0.1f) {
    for (size_t i = 0; i < v.size(); ++i) {
        // Valeurs petites, non symétriques.
        v[i] = scale * std::sin(static_cast<float>(i + 1) * 0.37f);
    }
}

static void init_weights_sequential(Layer& L, float base = 0.01f) {
    float* w = L.getWeights();
    const size_t n = L.getWeightsSize();
    if (!w || n == 0) return;
    for (size_t i = 0; i < n; ++i) w[i] = base * static_cast<float>(i + 1);
}

static void numeric_check_last_input_grad(
    Model& m,
    std::unordered_map<std::string, std::vector<float>> fin,
    const std::vector<float>& grad_out,
    float eps,
    float abs_tol,
    float rel_tol,
    int max_samples
) {
    // Forward+backward analytique (training=true => forward_state complet pour le backward)
    m.zeroGradients();
    const std::vector<float> y = m.forwardPassNamed(fin, {}, /*training=*/true);
    require_or_throw(y.size() == grad_out.size(), "numeric_check: grad_out size mismatch");

    m.backwardPass(grad_out);
    require_or_throw(m.hasLastInputGradient(), "numeric_check: missing last input gradient (__input__ not routed?)");
    const std::vector<float> gin = m.getLastInputGradient();

    auto it = fin.find("__input__");
    require_or_throw(it != fin.end(), "numeric_check: __input__ missing from inputs");
    const size_t n_in = it->second.size();
    require_or_throw(gin.size() == n_in, "numeric_check: last input gradient size mismatch");

    // Indices échantillonnés pour rester rapide
    std::vector<size_t> indices;
    indices.reserve(static_cast<size_t>(std::max(1, max_samples)));
    if (n_in == 0) return;

    if (static_cast<int>(n_in) <= max_samples) {
        for (size_t i = 0; i < n_in; ++i) indices.push_back(i);
    } else {
        // Échantillonnage déterministe: pas fixe
        const size_t step = std::max<size_t>(1, n_in / static_cast<size_t>(max_samples));
        for (size_t i = 0; i < n_in && static_cast<int>(indices.size()) < max_samples; i += step) {
            indices.push_back(i);
        }
        // Assurer qu'on teste aussi la fin.
        if (!indices.empty() && indices.back() != n_in - 1) indices.push_back(n_in - 1);
    }

    auto eval_loss = [&](const std::vector<float>& x) -> float {
        fin["__input__"] = x;
        const std::vector<float> yy = m.forwardPassNamed(fin, {}, /*training=*/true);
        return dot(yy, grad_out);
    };

    std::vector<float> x0 = fin["__input__"];

    for (size_t idx : indices) {
        std::vector<float> xp = x0;
        std::vector<float> xm = x0;
        xp[idx] += eps;
        xm[idx] -= eps;

        const float lp = eval_loss(xp);
        const float lm = eval_loss(xm);
        const float g_num = (lp - lm) / (2.0f * eps);
        const float g_ana = gin[idx];

        const float denom = std::max(1.0f, std::max(std::fabs(g_num), std::fabs(g_ana)));
        const float rel_err = std::fabs(g_num - g_ana) / denom;
        const float abs_err = std::fabs(g_num - g_ana);

        if (!(abs_err <= abs_tol || rel_err <= rel_tol)) {
            throw std::runtime_error(
                "numeric grad mismatch at i=" + std::to_string(idx) +
                " g_num=" + std::to_string(g_num) +
                " g_ana=" + std::to_string(g_ana) +
                " abs_err=" + std::to_string(abs_err) +
                " rel_err=" + std::to_string(rel_err)
            );
        }
    }
}

static void run_case_unary(
    const std::string& type,
    const std::vector<float>& x,
    const std::function<void(Layer&)>& configure,
    const std::function<void(Layer&)>& init
) {
    Model m;
    m.push("t0", type, 0);
    auto& layers = m.getMutableLayers();
    require_or_throw(layers.size() == 1, "run_case_unary: expected 1 layer");
    Layer& L = layers[0];
    L.inputs = {"__input__"};
    if (configure) configure(L);
    m.allocateParams();
    if (init) init(L);

    std::unordered_map<std::string, std::vector<float>> fin;
    fin["__input__"] = x;

    // grad_out déterministe (basé sur la taille de la sortie)
    const std::vector<float> y = m.forwardPassNamed(fin, {}, /*training=*/true);
    std::vector<float> grad_out(y.size());
    fill_deterministic(grad_out, 0.2f);

    // Paramètres de diff finie: assez stricts, mais robustes en float.
    numeric_check_last_input_grad(m, fin, grad_out, /*eps=*/1e-3f, /*abs_tol=*/2e-2f, /*rel_tol=*/2e-2f, /*max_samples=*/24);
}

static void run_case_binary(
    const std::string& type,
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    Model m;
    m.push("t0", type, 0);
    auto& layers = m.getMutableLayers();
    require_or_throw(layers.size() == 1, "run_case_binary: expected 1 layer");
    Layer& L = layers[0];
    L.inputs = {"__input__", "b"};
    m.allocateParams();

    std::unordered_map<std::string, std::vector<float>> fin;
    fin["__input__"] = a;
    fin["b"] = b;

    const std::vector<float> y = m.forwardPassNamed(fin, {}, /*training=*/true);
    std::vector<float> grad_out(y.size());
    fill_deterministic(grad_out, 0.2f);

    numeric_check_last_input_grad(m, fin, grad_out, /*eps=*/1e-3f, /*abs_tol=*/2e-2f, /*rel_tol=*/2e-2f, /*max_samples=*/24);
}

int main() {
    try {
        // ------------------------------------------------------------
        // Activations (unaires)
        // ------------------------------------------------------------
        {
            std::vector<float> x(32);
            fill_deterministic(x, 0.7f);
            run_case_unary("Sigmoid", x, {}, {});
            run_case_unary("Tanh", x, {}, {});
            run_case_unary("SiLU", x, {}, {});
            run_case_unary("GELU", x, {}, {});
        }

        // ------------------------------------------------------------
        // Linear
        // ------------------------------------------------------------
        {
            const int in_f = 8;
            const int out_f = 5;
            std::vector<float> x(static_cast<size_t>(in_f));
            fill_deterministic(x, 0.5f);

            std::function<void(Layer&)> cfg = [&](Layer& L) {
                L.in_features = in_f;
                L.out_features = out_f;
                L.use_bias = true;
                L.params_count = static_cast<size_t>(out_f) * static_cast<size_t>(in_f) + static_cast<size_t>(out_f);
            };
            std::function<void(Layer&)> init = [&](Layer& L) { init_weights_sequential(L, 0.01f); };

            run_case_unary("Linear", x, cfg, init);
        }

        // ------------------------------------------------------------
        // LayerNorm
        // ------------------------------------------------------------
        {
            const int n = 16;
            std::vector<float> x(static_cast<size_t>(n));
            fill_deterministic(x, 0.6f);

            std::function<void(Layer&)> cfg = [&](Layer& L) {
                L.in_features = n;
                L.affine = true;
                // gamma + beta
                L.params_count = static_cast<size_t>(2 * n);
                L.eps = 1e-5f;
            };
            std::function<void(Layer&)> init = [&](Layer& L) {
                float* w = L.getWeights();
                const size_t sz = L.getWeightsSize();
                require_or_throw(w && sz == static_cast<size_t>(2 * n), "LayerNorm init: weights size mismatch");
                for (int i = 0; i < n; ++i) w[static_cast<size_t>(i)] = 1.0f + 0.01f * static_cast<float>(i + 1); // gamma
                for (int i = 0; i < n; ++i) w[static_cast<size_t>(n + i)] = 0.01f * std::sin(static_cast<float>(i + 1)); // beta
            };

            run_case_unary("LayerNorm", x, cfg, init);
        }

        // ------------------------------------------------------------
        // TokenMeanPool
        // ------------------------------------------------------------
        {
            const int seq_len = 4;
            const int embed_dim = 6;
            std::vector<float> x(static_cast<size_t>(seq_len * embed_dim));
            fill_deterministic(x, 0.5f);

            std::function<void(Layer&)> cfg = [&](Layer& L) {
                L.seq_len = seq_len;
                L.embed_dim = embed_dim;
            };

            run_case_unary("TokenMeanPool", x, cfg, {});
        }

        // ------------------------------------------------------------
        // GlobalAvgPool2d
        // ------------------------------------------------------------
        {
            const int C = 3;
            const int H = 4;
            const int W = 5;
            std::vector<float> x(static_cast<size_t>(C * H * W));
            fill_deterministic(x, 0.4f);

            std::function<void(Layer&)> cfg = [&](Layer& L) {
                L.in_channels = C;
                L.input_height = H;
                L.input_width = W;
            };

            run_case_unary("GlobalAvgPool2d", x, cfg, {});
        }

        // ------------------------------------------------------------
        // Conv2d (petit cas, padding=same)
        // ------------------------------------------------------------
        {
            const int Cin = 2;
            const int Cout = 3;
            const int H = 5;
            const int W = 4;
            const int K = 3;
            const int stride = 1;
            const int padding = 1;

            std::vector<float> x(static_cast<size_t>(Cin * H * W));
            fill_deterministic(x, 0.3f);

            std::function<void(Layer&)> cfg = [&](Layer& L) {
                L.in_channels = Cin;
                L.out_channels = Cout;
                L.input_height = H;
                L.input_width = W;
                L.kernel_size = K;
                L.stride = stride;
                L.padding = padding;
                L.use_bias = true;
                const size_t w = static_cast<size_t>(Cout) * static_cast<size_t>(Cin) * static_cast<size_t>(K) * static_cast<size_t>(K);
                const size_t b = static_cast<size_t>(Cout);
                L.params_count = w + b;
            };
            std::function<void(Layer&)> init = [&](Layer& L) { init_weights_sequential(L, 0.005f); };

            run_case_unary("Conv2d", x, cfg, init);
        }

        // ------------------------------------------------------------
        // Ops binaires (grad wrt __input__)
        // ------------------------------------------------------------
        {
            std::vector<float> a(24), b(24);
            fill_deterministic(a, 0.5f);
            fill_deterministic(b, 0.4f);
            // Éviter les divisions proches de zéro
            for (float& v : b) v = (v >= 0.0f) ? (v + 0.7f) : (v - 0.7f);

            run_case_binary("Add", a, b);
            run_case_binary("Subtract", a, b);
            run_case_binary("Multiply", a, b);
            run_case_binary("Divide", a, b);
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
