// test_runtime_math_linear.cpp
// Solidité mathématique du path Linear : CPU + GPU optionnel.
//
// Valeurs analytiques vérifiables à la main :
//  - W = [[1,2],[3,4]], x = [1,0], b = none  →  y = [1, 3]
//  - W = I,             x = [5,-3], b = [1,-1] →  y = [6,-4]
//  - batch = 2 : W = diag(2,3), x = [1,1 | 2,2]  →  y = [2,3 | 4,6]
//
// GPU (si ENABLE_CUDA / ENABLE_ROCM) : même valeurs, tolérance 1e-4.

#include "test_utils.hpp"
#include "Model.hpp"
#include "runtimes/cpu/SIMD_Ops.hpp"

#define CHECK(cond) do { if (!(cond)) { std::cerr << "FATAL: " #cond "\n"; std::abort(); } } while(0)

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef ENABLE_CUDA
#  include "runtimes/cuda/CudaRuntime.hpp"
#endif
#ifdef ENABLE_ROCM
#  include "runtimes/rocm/RocmRuntime.hpp"
#endif
#ifdef ENABLE_VULKAN
#  include "runtimes/vulkan/VulkanRuntime.hpp"
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Helpers repris du pattern de test_layers_all_types.cpp
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<float> run_linear_cpu(
    int in_f, int out_f, bool use_bias,
    const std::vector<float>& weights_and_bias,
    const std::vector<float>& input
) {
    Model m;
    m.push("x0", "Linear", 0);
    auto& L = m.getMutableLayers()[0];
    L.in_features = in_f;
    L.out_features = out_f;
    L.use_bias = use_bias;
    L.params_count = static_cast<size_t>(out_f) * static_cast<size_t>(in_f)
                     + (use_bias ? static_cast<size_t>(out_f) : 0ULL);
    m.allocateParams();
    {
        float* w = L.getWeights();
        CHECK(w != nullptr);
        CHECK(L.getWeightsSize() >= weights_and_bias.size());
        for (size_t i = 0; i < weights_and_bias.size(); ++i) w[i] = weights_and_bias[i];
    }

    std::unordered_map<std::string, std::vector<float>> fin;
    fin["x0"] = input;
    std::unordered_map<std::string, std::vector<int>> iin;
    return m.forwardPassNamed(fin, iin, /*training=*/false);
}

// Reuse one runtime across growing/shrinking batches and in-place parameter
// mutations. A scratch cache must never become a stale parameter cache.
template <class Runtime>
static void check_linear_workspace(Runtime& runtime) {
    for (int batch : {1, 7, 3, 7}) {
        for (int in : {3, 65}) {
            const int out = 5;
            std::vector<float> x(size_t(batch) * in), weights(size_t(out) * in), bias(out), y(size_t(batch) * out);
            for (int iteration = 0; iteration < 3; ++iteration) {
                for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i % 13) - 6 + iteration) / 13;
                for (size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 17) - 8 - iteration) / 17;
                for (int o = 0; o < out; ++o) bias[o] = float(o + iteration) / 11;
                const float* b = iteration == 1 ? nullptr : bias.data();
                CHECK(runtime.linearForward(x.data(), weights.data(), b, y.data(), batch, in, out));
                for (int row = 0; row < batch; ++row) {
                    for (int o = 0; o < out; ++o) {
                        float expected = 0;
                        for (int i = 0; i < in; ++i)
                            expected += x[size_t(row) * in + i] * weights[size_t(o) * in + i];
                        if (b) expected += b[o];
                        CHECK(std::abs(y[size_t(row) * out + o] - expected) < 1e-4f);
                    }
                }
            }
        }
    }
}

int main() {
    // Odd matrix dimensions exercise the SIMD tails and adaptive OpenMP teams.
    {
        constexpr size_t m=256,n=67,k=129;
        std::vector<float> a(m*k),b(k*n),bt(n*k),c(m*n),ct(m*n);
        for(size_t i=0;i<a.size();++i)a[i]=std::sin(float(i%137)*.13f)*.1f;
        for(size_t i=0;i<b.size();++i)b[i]=std::cos(float(i%173)*.17f)*.1f;
        for(size_t i=0;i<k;++i)for(size_t j=0;j<n;++j)bt[j*k+i]=b[i*n+j];
        SIMD::matmul_avx2(c.data(),a.data(),b.data(),m,n,k);
        SIMD::matmul_transpose_avx2(ct.data(),a.data(),bt.data(),m,n,k);
        for(size_t i=0;i<m;++i)for(size_t j=0;j<n;++j){
            double expected=0;for(size_t r=0;r<k;++r)expected+=double(a[i*k+r])*b[r*n+j];
            TASSERT_NEAR(c[i*n+j],float(expected),1e-5f);
            TASSERT_NEAR(ct[i*n+j],float(expected),1e-5f);
        }
        HardwareOpt::matmul_add_fma_saturated(c.data(),a.data(),b.data(),m,n,k);
        for(size_t i=0;i<c.size();++i)TASSERT_NEAR(c[i],2.f*ct[i],1e-5f);
    }
    // =========================================================================
    // 1. W = [[1,2],[3,4]], x = [1,0], sans biais
    //    y[0] = 1·1 + 2·0 = 1
    //    y[1] = 3·1 + 4·0 = 3
    // =========================================================================
    {
        // Poids stockés [row0: 1 2] [row1: 3 4]
        const auto out = run_linear_cpu(2, 2, false,
            {1.f, 2.f,  3.f, 4.f},
            {1.f, 0.f});
        TASSERT_TRUE(out.size() == 2);
        TASSERT_NEAR(out[0], 1.f, 1e-5f);
        TASSERT_NEAR(out[1], 3.f, 1e-5f);
    }

    // =========================================================================
    // 2. W = I, x = [5,-3], biais = [1,-1]
    //    y = [5+1, -3-1] = [6, -4]
    // =========================================================================
    {
        const auto out = run_linear_cpu(2, 2, true,
            {1.f, 0.f,  0.f, 1.f,  /*biais*/1.f, -1.f},
            {5.f, -3.f});
        TASSERT_TRUE(out.size() == 2);
        TASSERT_NEAR(out[0],  6.f, 1e-5f);
        TASSERT_NEAR(out[1], -4.f, 1e-5f);
    }

    // =========================================================================
    // 3. W = [[2,0],[0,3]], x = [1,1], sans biais
    //    y = [2·1+0·1, 0·1+3·1] = [2, 3]
    // =========================================================================
    {
        const auto out = run_linear_cpu(2, 2, false,
            {2.f, 0.f,  0.f, 3.f},
            {1.f, 1.f});
        TASSERT_TRUE(out.size() == 2);
        TASSERT_NEAR(out[0], 2.f, 1e-5f);
        TASSERT_NEAR(out[1], 3.f, 1e-5f);
    }

    // =========================================================================
    // 4. Batch = 2 : W = diag(2,3), x = [1,1 | 2,2], sans biais
    //    token0 → [2, 3]
    //    token1 → [4, 6]
    // =========================================================================
    {
        const auto out = run_linear_cpu(2, 2, false,
            {2.f, 0.f,  0.f, 3.f},
            {1.f, 1.f,  2.f, 2.f});   // batch=2 concaténé
        TASSERT_TRUE(out.size() == 4);
        TASSERT_NEAR(out[0], 2.f, 1e-5f);
        TASSERT_NEAR(out[1], 3.f, 1e-5f);
        TASSERT_NEAR(out[2], 4.f, 1e-5f);
        TASSERT_NEAR(out[3], 6.f, 1e-5f);
    }

    // =========================================================================
    // 5. Linéarité : y(a·x) = a·y(x) pour a = 3
    // =========================================================================
    {
        const std::vector<float> W = {1.f, 2.f,  3.f, 4.f};
        const auto y1 = run_linear_cpu(2, 2, false, W, {1.f, 2.f});
        const auto y3 = run_linear_cpu(2, 2, false, W, {3.f, 6.f});
        TASSERT_TRUE(y1.size() == 2 && y3.size() == 2);
        TASSERT_NEAR(y3[0], 3.f * y1[0], 1e-4f);
        TASSERT_NEAR(y3[1], 3.f * y1[1], 1e-4f);
    }

    // =========================================================================
    // 6. GPU CUDA – même test que cas 1 via linearForward() direct
    // =========================================================================
#ifdef ENABLE_CUDA
    {
        CudaRuntime rt;
        RuntimeConfig cfg;
        cfg.linear_enabled = true;
        cfg.linear_min_ops = 0;
        if (!rt.initialize(cfg)) {
            std::cout << "SKIP: no CUDA device\n";
        } else {
            check_linear_workspace(rt);
            const float W[4] = {1.f, 2.f,  3.f, 4.f};
            const float x[2] = {1.f, 0.f};
            float y[2]       = {0.f, 0.f};
            const bool ok = rt.linearForward(x, W, nullptr, y, /*batch=*/1, 2, 2);
            TASSERT_TRUE(ok);
            TASSERT_NEAR(y[0], 1.f, 1e-4f);
            TASSERT_NEAR(y[1], 3.f, 1e-4f);
        }
    }

    // GPU : cas avec biais
    {
        CudaRuntime rt;
        RuntimeConfig cfg;
        cfg.linear_enabled = true;
        cfg.linear_min_ops = 0;
        if (rt.initialize(cfg)) {
            const float W[4]  = {1.f, 0.f,  0.f, 1.f};   // identité
            const float b[2]  = {1.f, -1.f};
            const float x[2]  = {5.f, -3.f};
            float y[2]        = {0.f, 0.f};
            const bool ok = rt.linearForward(x, W, b, y, 1, 2, 2);
            TASSERT_TRUE(ok);
            TASSERT_NEAR(y[0],  6.f, 1e-4f);
            TASSERT_NEAR(y[1], -4.f, 1e-4f);
        }
    }

    // GPU : batch = 2 via forwardLayer
    {
        CudaRuntime rt;
        RuntimeConfig cfg;
        cfg.linear_enabled = true;
        cfg.linear_min_ops = 0;
        if (rt.initialize(cfg)) {
            // Construction du Layer via Model pour allocation correcte des poids
            Model m;
            m.push("x0", "Linear", 0);
            auto& L = m.getMutableLayers()[0];
            L.in_features  = 2;
            L.out_features = 2;
            L.use_bias     = false;
            L.params_count = 4;
            m.allocateParams();
            float* w = L.getWeights();
            TASSERT_TRUE(w != nullptr);
            w[0] = 2.f; w[1] = 0.f;
            w[2] = 0.f; w[3] = 3.f;

            const std::vector<float> x = {1.f, 1.f,  2.f, 2.f};
            const std::vector<const std::vector<float>*> inputs = {&x};
            std::vector<std::vector<float>> outputs;
            const bool ok = rt.forwardLayer(inputs, outputs, L, false);
            TASSERT_TRUE(ok);
            TASSERT_TRUE(outputs.size() == 1 && outputs[0].size() == 4);
            TASSERT_NEAR(outputs[0][0], 2.f, 1e-4f);
            TASSERT_NEAR(outputs[0][1], 3.f, 1e-4f);
            TASSERT_NEAR(outputs[0][2], 4.f, 1e-4f);
            TASSERT_NEAR(outputs[0][3], 6.f, 1e-4f);
        }
    }
#endif  // ENABLE_CUDA

    // =========================================================================
    // 7. GPU ROCm – cas 1 via linearForward() direct
    // =========================================================================
#ifdef ENABLE_ROCM
    {
        RocmRuntime rt;
        RuntimeConfig cfg;
        cfg.linear_enabled = true;
        cfg.linear_min_ops = 0;
        if (!rt.initialize(cfg)) {
            std::cout << "SKIP: no ROCm device\n";
        } else {
            check_linear_workspace(rt);
            const float W[4] = {1.f, 2.f,  3.f, 4.f};
            const float x[2] = {1.f, 0.f};
            float y[2]       = {0.f, 0.f};
            const bool ok = rt.linearForward(x, W, nullptr, y, 1, 2, 2);
            TASSERT_TRUE(ok);
            TASSERT_NEAR(y[0], 1.f, 1e-4f);
            TASSERT_NEAR(y[1], 3.f, 1e-4f);
        }
    }
#endif  // ENABLE_ROCM

    // =========================================================================
    // 8. GPU Vulkan – cas 1 via linearForward() direct
    // =========================================================================
#ifdef ENABLE_VULKAN
    {
        VulkanRuntime rt;
        RuntimeConfig cfg;
        cfg.linear_enabled = true;
        cfg.linear_min_ops = 0;
        if (!rt.initialize(cfg)) {
            std::cout << "SKIP: no Vulkan compute device\n";
        } else {
            check_linear_workspace(rt);
            const float W[4] = {1.f, 2.f,  3.f, 4.f};
            const float x[2] = {1.f, 0.f};
            float y[2]       = {0.f, 0.f};
            const bool ok = rt.linearForward(x, W, nullptr, y, /*batch=*/1, 2, 2);
            TASSERT_TRUE(ok);
            TASSERT_NEAR(y[0], 1.f, 1e-4f);
            TASSERT_NEAR(y[1], 3.f, 1e-4f);
        }
    }

    // GPU Vulkan : cas avec biais
    {
        VulkanRuntime rt;
        RuntimeConfig cfg;
        cfg.linear_enabled = true;
        cfg.linear_min_ops = 0;
        if (rt.initialize(cfg)) {
            const float W[4]  = {1.f, 0.f,  0.f, 1.f};
            const float b[2]  = {1.f, -1.f};
            const float x[2]  = {5.f, -3.f};
            float y[2]        = {0.f, 0.f};
            const bool ok = rt.linearForward(x, W, b, y, 1, 2, 2);
            TASSERT_TRUE(ok);
            TASSERT_NEAR(y[0],  6.f, 1e-4f);
            TASSERT_NEAR(y[1], -4.f, 1e-4f);
        }
    }

    // GPU Vulkan : batch = 2 via forwardLayer
    {
        VulkanRuntime rt;
        RuntimeConfig cfg;
        cfg.linear_enabled = true;
        cfg.linear_min_ops = 0;
        if (rt.initialize(cfg)) {
            Model m;
            m.push("x0", "Linear", 0);
            auto& L = m.getMutableLayers()[0];
            L.in_features  = 2;
            L.out_features = 2;
            L.use_bias     = false;
            L.params_count = 4;
            m.allocateParams();
            float* w = L.getWeights();
            TASSERT_TRUE(w != nullptr);
            w[0] = 2.f; w[1] = 0.f;
            w[2] = 0.f; w[3] = 3.f;

            const std::vector<float> x = {1.f, 1.f,  2.f, 2.f};
            const std::vector<const std::vector<float>*> inputs = {&x};
            std::vector<std::vector<float>> outputs;
            const bool ok = rt.forwardLayer(inputs, outputs, L, false);
            TASSERT_TRUE(ok);
            TASSERT_TRUE(outputs.size() == 1 && outputs[0].size() == 4);
            TASSERT_NEAR(outputs[0][0], 2.f, 1e-4f);
            TASSERT_NEAR(outputs[0][1], 3.f, 1e-4f);
            TASSERT_NEAR(outputs[0][2], 4.f, 1e-4f);
            TASSERT_NEAR(outputs[0][3], 6.f, 1e-4f);
        }
    }

    // GPU Vulkan : deux kernels unary chaînés avec un seul upload/download.
    {
        VulkanRuntime rt;
        RuntimeConfig cfg;
        if (rt.initialize(cfg)) {
            const float x[4] = {-2.f, -0.5f, 0.f, 2.f};
            float y[4] = {};
            const bool ok = rt.unaryChainForwardResident(
                x, y, 4, {LayerType::ReLU, LayerType::Sigmoid});
            TASSERT_TRUE(ok);
            TASSERT_NEAR(y[0], 0.5f, 1e-5f);
            TASSERT_NEAR(y[1], 0.5f, 1e-5f);
            TASSERT_NEAR(y[2], 0.5f, 1e-5f);
            TASSERT_NEAR(y[3], 1.f / (1.f + std::exp(-2.f)), 1e-5f);
        }
    }
#endif  // ENABLE_VULKAN

    return 0;
}
