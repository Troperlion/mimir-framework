// test_runtime_math_norms.cpp
// Solidité mathématique des paths LayerNorm et RMSNorm : CPU + GPU optionnel.
//
// Propriétés vérifiées :
//
// LayerNorm (γ=1, β=0) :
//   P1. mean(output) ≈ 0
//   P2. std(output)  ≈ 1
//   P3. Invariance par translation : LN(x + c) = LN(x)
//   P4. Scaling de γ : γ=2 → std(output) ≈ 2
//   P5. Biais β : mean(output) ≈ β  (avec γ=1)
//
// RMSNorm (γ=1) :
//   P6. rms(output) ≈ 1  (pour entrée quelconque non nulle)
//   P7. Invariant échelle doublée : même direction que l'entrée
//   P8. Valeur analytique exacte : input = [3, 4]
//         rms = √((9+16)/2) = √12.5 ≈ 3.5355
//         output ≈ [0.8485, 1.1314]
//
// GPU (ENABLE_CUDA / ENABLE_ROCM) : résultats ≈ CPU ±1e-4.

#include "test_utils.hpp"
#include "Model.hpp"

#define CHECK(cond) do { if (!(cond)) { std::cerr << "FATAL: " #cond "\n"; std::abort(); } } while(0)

#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef ENABLE_CUDA
#  include "runtimes/cuda/CudaRuntime.hpp"
#endif
#ifdef ENABLE_ROCM
#  include "runtimes/rocm/RocmRuntime.hpp"
#endif
#ifdef ENABLE_OPENCL
#  include "runtimes/opencl/OpenCLRuntime.hpp"
#endif
#ifdef ENABLE_VULKAN
#  include "runtimes/vulkan/VulkanRuntime.hpp"
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Statistiques de base
// ─────────────────────────────────────────────────────────────────────────────
static float vec_mean(const std::vector<float>& v) {
    float s = 0.f;
    for (float x : v) s += x;
    return s / static_cast<float>(v.size());
}

static float vec_std(const std::vector<float>& v) {
    float m = vec_mean(v);
    float var = 0.f;
    for (float x : v) { float d = x - m; var += d * d; }
    return std::sqrt(var / static_cast<float>(v.size()));
}

static float vec_rms(const std::vector<float>& v) {
    float ss = 0.f;
    for (float x : v) ss += x * x;
    return std::sqrt(ss / static_cast<float>(v.size()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper : LayerNorm via Model
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> run_layernorm_cpu(
    const std::vector<float>& input,    // doit être de taille in_f
    int in_f,
    bool use_bias,
    const std::vector<float>& gamma,    // taille in_f
    const std::vector<float>& beta      // taille in_f (ignoré si !use_bias)
) {
    Model m;
    m.push("x0", "LayerNorm", 0);
    auto& L = m.getMutableLayers()[0];
    L.in_features = in_f;
    L.affine      = true;
    L.use_bias    = use_bias;
    L.eps         = 1e-5f;
    L.params_count = static_cast<size_t>(in_f) * (use_bias ? 2ULL : 1ULL);
    m.allocateParams();

    float* w = L.getWeights();
    CHECK(w != nullptr);
    for (int i = 0; i < in_f; ++i) w[i] = gamma[i];
    if (use_bias) {
        for (int i = 0; i < in_f; ++i) w[in_f + i] = beta[i];
    }

    std::unordered_map<std::string, std::vector<float>> fin;
    fin["x0"] = input;
    std::unordered_map<std::string, std::vector<int>> iin;
    return m.forwardPassNamed(fin, iin, false);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper : RMSNorm via Model
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> run_rmsnorm_cpu(
    const std::vector<float>& input,
    const std::vector<float>& gamma    // taille = input.size()
) {
    const int n = static_cast<int>(input.size());
    Model m;
    m.push("x0", "RMSNorm", 0);
    auto& L = m.getMutableLayers()[0];
    L.in_features  = n;
    L.affine       = true;
    L.use_bias     = false;
    L.eps          = 1e-5f;
    L.params_count = static_cast<size_t>(n);
    m.allocateParams();

    float* w = L.getWeights();
    CHECK(w != nullptr);
    for (int i = 0; i < n; ++i) w[i] = gamma[i];

    std::unordered_map<std::string, std::vector<float>> fin;
    fin["x0"] = input;
    std::unordered_map<std::string, std::vector<int>> iin;
    return m.forwardPassNamed(fin, iin, false);
}

#if defined(ENABLE_CUDA) || defined(ENABLE_ROCM) || defined(ENABLE_OPENCL) || defined(ENABLE_VULKAN)
static int check_layernorm_runtime(AbstractRuntime& runtime) {
    RuntimeConfig cfg;
    cfg.norm_enabled = true;
    cfg.norm_min_elements = 0;
    if (!runtime.initialize(cfg)) {
        std::cout << "SKIP: no " << runtime.name() << " device (Norms)\n";
        return 0;
    }

    const std::vector<float> input = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    const std::vector<float> gamma = {1.f, 2.f, 0.5f, 1.5f};
    const std::vector<float> beta = {0.5f, -1.f, 2.f, 0.f};
    Model model;
    model.push("x0", "LayerNorm", 0);
    auto& layer = model.getMutableLayers()[0];
    layer.in_features = 4;
    layer.affine = true;
    layer.use_bias = true;
    layer.eps = 1e-5f;
    layer.params_count = 8;
    model.allocateParams();
    std::copy(gamma.begin(), gamma.end(), layer.getWeights());
    std::copy(beta.begin(), beta.end(), layer.getWeights() + 4);

    const std::vector<const std::vector<float>*> inputs = {&input};
    TASSERT_TRUE(runtimeCapabilityIsNative(
        runtime.queryForwardOperationCapability(layer, inputs, false)));
    std::vector<std::vector<float>> output;
    TASSERT_TRUE(runtime.forwardLayer(inputs, output, layer, false));
    const auto reference = run_layernorm_cpu(input, 4, true, gamma, beta);
    TASSERT_TRUE(output.size() == 1 && output[0].size() == reference.size());
    for (size_t index = 0; index < reference.size(); ++index) {
        TASSERT_NEAR(output[0][index], reference[index], 1e-4f);
    }

    layer.affine = false;
    layer.use_bias = false;
    TASSERT_TRUE(runtime.forwardLayer(inputs, output, layer, false));
    const std::vector<float> identity_gamma(4, 1.0f);
    const std::vector<float> zero_beta(4, 0.0f);
    const auto non_affine_reference =
        run_layernorm_cpu(input, 4, false, identity_gamma, zero_beta);
    for (size_t index = 0; index < non_affine_reference.size(); ++index) {
        TASSERT_NEAR(output[0][index], non_affine_reference[index], 1e-4f);
    }
    return 0;
}
#endif

int main() {
    // =========================================================================
    // LAYERNORM
    // =========================================================================
    const std::vector<float> inp4 = {0.f, 1.f, 2.f, 3.f};
    const std::vector<float> gamma1(4, 1.f);
    const std::vector<float> beta0(4, 0.f);

    // --- P1 & P2 : mean≈0, std≈1 ---
    {
        const auto out = run_layernorm_cpu(inp4, 4, true, gamma1, beta0);
        TASSERT_TRUE(out.size() == 4);
        TASSERT_NEAR(vec_mean(out), 0.f, 1e-5f);
        TASSERT_NEAR(vec_std(out),  1.f, 1e-5f);
    }

    // --- P3 : invariance par translation c = 10 ---
    {
        std::vector<float> shifted(inp4);
        for (float& v : shifted) v += 10.f;
        const auto out_base    = run_layernorm_cpu(inp4,   4, true, gamma1, beta0);
        const auto out_shifted = run_layernorm_cpu(shifted, 4, true, gamma1, beta0);
        TASSERT_TRUE(out_base.size() == 4 && out_shifted.size() == 4);
        for (int i = 0; i < 4; ++i) {
            TASSERT_NEAR(out_base[i], out_shifted[i], 1e-5f);
        }
    }

    // --- P4 : γ=2 → std(output) ≈ 2 ---
    {
        const std::vector<float> gamma2(4, 2.f);
        const auto out = run_layernorm_cpu(inp4, 4, true, gamma2, beta0);
        TASSERT_TRUE(out.size() == 4);
        TASSERT_NEAR(vec_std(out), 2.f, 1e-5f);
    }

    // --- P5 : β=5, γ=1 → mean(output) ≈ 5 ---
    {
        const std::vector<float> beta5(4, 5.f);
        const auto out = run_layernorm_cpu(inp4, 4, true, gamma1, beta5);
        TASSERT_TRUE(out.size() == 4);
        TASSERT_NEAR(vec_mean(out), 5.f, 1e-5f);
    }

    // --- P4-bis : valeur analytique de la normalisation ---
    // input = [0,1,2,3], mean=1.5, var=1.25
    // x̂ = (x - 1.5) / sqrt(1.25 + eps) ≈ [-1.342, -0.447, 0.447, 1.342]
    {
        const auto out = run_layernorm_cpu(inp4, 4, true, gamma1, beta0);
        const float inv_std = 1.f / std::sqrt(1.25f + 1e-5f);
        TASSERT_NEAR(out[0], (0.f - 1.5f) * inv_std, 1e-5f);
        TASSERT_NEAR(out[1], (1.f - 1.5f) * inv_std, 1e-5f);
        TASSERT_NEAR(out[2], (2.f - 1.5f) * inv_std, 1e-5f);
        TASSERT_NEAR(out[3], (3.f - 1.5f) * inv_std, 1e-5f);
    }

    // =========================================================================
    // RMSNORM
    // =========================================================================
    const std::vector<float> gamma1_2(2, 1.f);

    // --- P8 : valeur analytique [3, 4] ---
    // rms = sqrt((9+16)/2 + eps) ≈ 3.5355
    {
        const auto out = run_rmsnorm_cpu({3.f, 4.f}, gamma1_2);
        TASSERT_TRUE(out.size() == 2);
        const float rms = std::sqrt((9.f + 16.f) / 2.f + 1e-5f);
        TASSERT_NEAR(out[0], 3.f / rms, 1e-5f);
        TASSERT_NEAR(out[1], 4.f / rms, 1e-5f);
    }

    // --- P6 : rms(output) ≈ 1 (γ=1) ---
    {
        const auto out = run_rmsnorm_cpu({3.f, 4.f}, gamma1_2);
        TASSERT_NEAR(vec_rms(out), 1.f, 1e-5f);
    }

    // --- P7 : γ=2 → rms(output) ≈ 2 ---
    {
        const std::vector<float> gamma2_2(2, 2.f);
        const auto out = run_rmsnorm_cpu({3.f, 4.f}, gamma2_2);
        TASSERT_NEAR(vec_rms(out), 2.f, 1e-5f);
    }

    // --- Invariance sens : doubler l'entrée ne change pas la direction ---
    {
        const auto out1x = run_rmsnorm_cpu({3.f, 4.f},     gamma1_2);
        const auto out2x = run_rmsnorm_cpu({6.f, 8.f},     gamma1_2);
        TASSERT_NEAR(out1x[0], out2x[0], 1e-5f);
        TASSERT_NEAR(out1x[1], out2x[1], 1e-5f);
    }

    // =========================================================================
    // GPU CUDA – LayerNorm : résultat ≈ CPU
    // =========================================================================
#ifdef ENABLE_CUDA
    {
        CudaRuntime rt;
        RuntimeConfig cfg;
        cfg.norm_enabled        = true;
        cfg.norm_min_elements   = 0;
        if (!rt.initialize(cfg)) {
            std::cout << "SKIP: no CUDA device (Norms)\n";
        } else {
            Model m;
            m.push("x0", "LayerNorm", 0);
            auto& L = m.getMutableLayers()[0];
            L.in_features  = 4;
            L.affine       = true;
            L.use_bias     = true;
            L.eps          = 1e-5f;
            L.params_count = 8;   // 4 gamma + 4 beta
            m.allocateParams();
            float* w = L.getWeights();
            TASSERT_TRUE(w != nullptr);
            for (int i = 0; i < 4; ++i) w[i]     = 1.f;  // gamma = 1
            for (int i = 0; i < 4; ++i) w[4 + i] = 0.f;  // beta  = 0

            const std::vector<float> x = {0.f, 1.f, 2.f, 3.f};
            const std::vector<const std::vector<float>*> inputs = {&x};
            std::vector<std::vector<float>> gpu_out;
            const bool ok = rt.forwardLayer(inputs, gpu_out, L, false);
            TASSERT_TRUE(ok);
            TASSERT_TRUE(gpu_out.size() == 1 && gpu_out[0].size() == 4);
            // Référence CPU
            const auto cpu_out = run_layernorm_cpu(x, 4, true, gamma1, beta0);
            for (int i = 0; i < 4; ++i) {
                TASSERT_NEAR(gpu_out[0][i], cpu_out[i], 1e-4f);
            }
        }
    }

    // GPU CUDA – RMSNorm
    {
        CudaRuntime rt;
        RuntimeConfig cfg;
        cfg.norm_enabled      = true;
        cfg.norm_min_elements = 0;
        if (rt.initialize(cfg)) {
            Model m;
            m.push("x0", "RMSNorm", 0);
            auto& L = m.getMutableLayers()[0];
            L.in_features  = 2;
            L.affine       = true;
            L.use_bias     = false;
            L.eps          = 1e-5f;
            L.params_count = 2;
            m.allocateParams();
            L.getWeights()[0] = 1.f;
            L.getWeights()[1] = 1.f;

            const std::vector<float> x = {3.f, 4.f};
            const std::vector<const std::vector<float>*> inputs = {&x};
            std::vector<std::vector<float>> gpu_out;
            const bool ok = rt.forwardLayer(inputs, gpu_out, L, false);
            TASSERT_TRUE(ok);
            TASSERT_TRUE(gpu_out.size() == 1 && gpu_out[0].size() == 2);
            const auto cpu_out = run_rmsnorm_cpu(x, gamma1_2);
            TASSERT_NEAR(gpu_out[0][0], cpu_out[0], 1e-4f);
            TASSERT_NEAR(gpu_out[0][1], cpu_out[1], 1e-4f);
        }
    }
#endif  // ENABLE_CUDA

#ifdef ENABLE_ROCM
    {
        RocmRuntime rt;
        RuntimeConfig cfg;
        cfg.norm_enabled      = true;
        cfg.norm_min_elements = 0;
        if (!rt.initialize(cfg)) {
            std::cout << "SKIP: no ROCm device (Norms)\n";
        } else {
            Model m;
            m.push("x0", "LayerNorm", 0);
            auto& L = m.getMutableLayers()[0];
            L.in_features  = 4;
            L.affine       = true;
            L.use_bias     = true;
            L.eps          = 1e-5f;
            L.params_count = 8;
            m.allocateParams();
            float* w = L.getWeights();
            TASSERT_TRUE(w != nullptr);
            for (int i = 0; i < 4; ++i) w[i]     = 1.f;
            for (int i = 0; i < 4; ++i) w[4 + i] = 0.f;

            const std::vector<float> x = {0.f, 1.f, 2.f, 3.f};
            const std::vector<const std::vector<float>*> inputs = {&x};
            std::vector<std::vector<float>> gpu_out;
            const bool ok = rt.forwardLayer(inputs, gpu_out, L, false);
            TASSERT_TRUE(ok);
            TASSERT_TRUE(gpu_out.size() == 1 && gpu_out[0].size() == 4);
            const auto cpu_out = run_layernorm_cpu(x, 4, true, gamma1, beta0);
            for (int i = 0; i < 4; ++i) {
                TASSERT_NEAR(gpu_out[0][i], cpu_out[i], 1e-4f);
            }
        }
    }
#endif  // ENABLE_ROCM

#ifdef ENABLE_CUDA
    {
        CudaRuntime runtime;
        TASSERT_TRUE(check_layernorm_runtime(runtime) == 0);
    }
#endif

#ifdef ENABLE_ROCM
    {
        RocmRuntime runtime;
        TASSERT_TRUE(check_layernorm_runtime(runtime) == 0);
    }
#endif

#ifdef ENABLE_OPENCL
    {
        OpenCLRuntime runtime;
        TASSERT_TRUE(check_layernorm_runtime(runtime) == 0);
    }
#endif

#ifdef ENABLE_VULKAN
    {
        VulkanRuntime runtime;
        TASSERT_TRUE(check_layernorm_runtime(runtime) == 0);
    }
#endif

    return 0;
}
