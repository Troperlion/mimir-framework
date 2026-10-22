// test_runtime_math_conv2d.cpp
// Solidité mathématique du path Conv2d : CPU + GPU optionnel.
//
// Cas analytiques :
//  A) 1ch, 2×2, kernel 1×1 = 2.0, sans biais  →  output = 2 × input
//  B) 1ch, 3×3, kernel 3×3 identité (centre=1), pad=1  →  output ≈ input
//  C) 2ch, 2×2, kernels diagonaux indépendants  →  canaux scalés séparément
//  D) stride=2 : 4×4 → 2×2 avec kernel 1×1
//
// GPU (si ENABLE_CUDA / ENABLE_ROCM) : cas A comparé au CPU, tolérance 1e-4.

#include "test_utils.hpp"
#include "Model.hpp"
#include "runtimes/cpu/ConvolutionKernels.hpp"

#define CHECK(cond) do { if (!(cond)) { std::cerr << "FATAL: " #cond "\n"; std::abort(); } } while(0)

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

// ─────────────────────────────────────────────────────────────────────────────
// Helper : construit un modèle Conv2d, alloue les poids, renvoie le résultat.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> run_conv2d_cpu(
    int in_c, int out_c, int H, int W, int k, int stride, int pad,
    const std::vector<float>& kernel_weights,   // [out_c, in_c, k, k]
    const std::vector<float>& bias_or_empty,
    const std::vector<float>& input             // [in_c, H, W]
) {
    const bool use_bias = !bias_or_empty.empty();
    const size_t kernel_sz = static_cast<size_t>(out_c) * in_c * k * k;

    Model m;
    m.push("x0", "Conv2d", 0);
    auto& L = m.getMutableLayers()[0];
    L.in_channels  = in_c;
    L.out_channels = out_c;
    L.input_height = H;
    L.input_width  = W;
    L.kernel_size  = k;
    L.stride       = stride;
    L.padding      = pad;
    L.use_bias     = use_bias;
    L.params_count = kernel_sz + (use_bias ? static_cast<size_t>(out_c) : 0ULL);
    m.allocateParams();

    float* w = L.getWeights();
    CHECK(w != nullptr);
    CHECK(L.getWeightsSize() >= kernel_sz);
    for (size_t i = 0; i < kernel_sz; ++i) w[i] = kernel_weights[i];
    if (use_bias) {
        for (int i = 0; i < out_c; ++i) w[kernel_sz + i] = bias_or_empty[i];
    }

    std::unordered_map<std::string, std::vector<float>> fin;
    fin["x0"] = input;
    std::unordered_map<std::string, std::vector<int>> iin;
    return m.forwardPassNamed(fin, iin, false);
}

int main() {
    // Finite differences cover packed forward/backward, border windows, stride
    // and dilation, including the partially filled final tile.
    for(int stride:{1,2})for(int dilation:{1,2}) {
        Layer layer;layer.use_bias=true;layer.weights.resize(2*2*9+2);
        for(size_t i=0;i<layer.weights.size();++i)layer.weights[i]=std::sin(float(i)*.37f)*.1f;
        std::vector<float> x(2*7*9);for(size_t i=0;i<x.size();++i)x[i]=std::cos(float(i)*.21f)*.3f;
        auto forward=[&](){std::vector<float> y;CpuConvolution::forward(x,y,layer.weights.data(),layer.weights.data()+36,7,9,2,2,3,stride,2,dilation);return y;};
        auto output=forward();std::vector<float> go(output.size());
        for(size_t i=0;i<go.size();++i)go[i]=std::sin(float(i)*.43f)*.2f;
        auto objective=[&](){auto y=forward();double sum=0;for(size_t i=0;i<y.size();++i)sum+=double(y[i])*go[i];return sum;};
        std::vector<float> dx;CpuConvolution::backward(x,go,dx,layer,7,9,2,2,3,stride,2,dilation);
        constexpr float epsilon=.001f;
        for(size_t i:{size_t(0),size_t(8),size_t(35),size_t(36),size_t(37)}){
            const float saved=layer.weights[i];layer.weights[i]=saved+epsilon;const auto plus=objective();layer.weights[i]=saved-epsilon;const auto minus=objective();layer.weights[i]=saved;
            TASSERT_NEAR(layer.grad_weights[i],float((plus-minus)/(2*epsilon)),2e-4f);
        }
        for(size_t i:{size_t(0),size_t(31),size_t(125)}){
            const float saved=x[i];x[i]=saved+epsilon;const auto plus=objective();x[i]=saved-epsilon;const auto minus=objective();x[i]=saved;
            TASSERT_NEAR(dx[i],float((plus-minus)/(2*epsilon)),2e-4f);
        }
    }
    // =========================================================================
    // A. Kernel 1×1 = 2 : sortie = 2 × entrée
    // =========================================================================
    {
        // kernel[0] = 2.0  (seul élément pour out=1, in=1, k=1)
        const auto out = run_conv2d_cpu(1, 1, 2, 2, 1, 1, 0,
            {2.f}, {}, {1.f, 2.f, 3.f, 4.f});
        TASSERT_TRUE(out.size() == 4);
        TASSERT_NEAR(out[0], 2.f, 1e-5f);
        TASSERT_NEAR(out[1], 4.f, 1e-5f);
        TASSERT_NEAR(out[2], 6.f, 1e-5f);
        TASSERT_NEAR(out[3], 8.f, 1e-5f);
    }

    // =========================================================================
    // B. Kernel 3×3 identité (centre = 1, reste = 0), pad = 1, stride = 1 :
    //    output[i,j] == input[i,j]  pour tout i,j
    // =========================================================================
    {
        // 3×3 input arbitraire
        const std::vector<float> inp = {
            1.f, 2.f, 3.f,
            4.f, 5.f, 6.f,
            7.f, 8.f, 9.f
        };
        // kernel [out=1, in=1, 3, 3] : seul le centre vaut 1
        std::vector<float> id_kernel(9, 0.f);
        id_kernel[4] = 1.f;   // position (1,1) = centre

        const auto out = run_conv2d_cpu(1, 1, 3, 3, 3, 1, 1,
            id_kernel, {}, inp);
        TASSERT_TRUE(out.size() == 9);
        for (int i = 0; i < 9; ++i) {
            TASSERT_NEAR(out[i], inp[i], 1e-5f);
        }
    }

    // =========================================================================
    // C. 2 canaux, kernel 1×1 par canal
    //    Canal 0 : poids = 3.0  → output_ch0 = 3 × input_ch0
    //    Canal 1 : poids = 0.5  → output_ch1 = 0.5 × input_ch1
    // =========================================================================
    {
        // kernel [out=2, in=2, 1, 1] :
        //   W[0,0,0,0] = 3, W[0,1,0,0] = 0  (out0 ← 3·ch0 + 0·ch1)
        //   W[1,0,0,0] = 0, W[1,1,0,0] = 0.5  (out1 ← 0·ch0 + 0.5·ch1)
        const std::vector<float> kernel = {3.f, 0.f,  0.f, 0.5f};
        // input [ch0: 1 2 | ch1: 10 20]
        const std::vector<float> inp = {1.f, 2.f,  10.f, 20.f};

        const auto out = run_conv2d_cpu(2, 2, 1, 2, 1, 1, 0, kernel, {}, inp);
        // out [ch0: 3 6 | ch1: 5 10]
        TASSERT_TRUE(out.size() == 4);
        TASSERT_NEAR(out[0],  3.f, 1e-5f);  // ch0 pix0
        TASSERT_NEAR(out[1],  6.f, 1e-5f);  // ch0 pix1
        TASSERT_NEAR(out[2],  5.f, 1e-5f);  // ch1 pix0
        TASSERT_NEAR(out[3], 10.f, 1e-5f);  // ch1 pix1
    }

    // =========================================================================
    // D. Stride = 2 : 4×4 → 2×2 avec kernel 1×1 = 1.0 (sélectionne 1 pixel/2)
    //    output(0,0) = input(0,0), output(0,1) = input(0,2), etc.
    // =========================================================================
    {
        const std::vector<float> inp = {
            1.f,  2.f,  3.f,  4.f,
            5.f,  6.f,  7.f,  8.f,
            9.f, 10.f, 11.f, 12.f,
           13.f, 14.f, 15.f, 16.f
        };
        const auto out = run_conv2d_cpu(1, 1, 4, 4, 1, 2, 0, {1.f}, {}, inp);
        // Positions retenues (stride=2 sur grille 4×4) : (0,0),(0,2),(2,0),(2,2)
        TASSERT_TRUE(out.size() == 4);
        TASSERT_NEAR(out[0],  1.f, 1e-5f);
        TASSERT_NEAR(out[1],  3.f, 1e-5f);
        TASSERT_NEAR(out[2],  9.f, 1e-5f);
        TASSERT_NEAR(out[3], 11.f, 1e-5f);
    }

    // =========================================================================
    // E. Biais additif : kernel = 0, biais = 7  → output = 7 partout
    // =========================================================================
    {
        const auto out = run_conv2d_cpu(1, 1, 2, 2, 1, 1, 0,
            {0.f}, {7.f}, {100.f, 200.f, 300.f, 400.f});
        TASSERT_TRUE(out.size() == 4);
        for (float v : out) TASSERT_NEAR(v, 7.f, 1e-5f);
    }

    // =========================================================================
    // F. GPU CUDA – même test que cas A, comparaison via forwardLayer()
    // =========================================================================
#ifdef ENABLE_CUDA
    {
        CudaRuntime rt;
        RuntimeConfig cfg;
        cfg.conv_enabled   = true;
        cfg.conv_min_ops   = 0;
        if (!rt.initialize(cfg)) {
            std::cout << "SKIP: no CUDA device (Conv2d)\n";
        } else {
            Model m;
            m.push("x0", "Conv2d", 0);
            auto& L = m.getMutableLayers()[0];
            L.in_channels  = 1;
            L.out_channels = 1;
            L.input_height = 2;
            L.input_width  = 2;
            L.kernel_size  = 1;
            L.stride       = 1;
            L.padding      = 0;
            L.use_bias     = false;
            L.params_count = 1;
            m.allocateParams();
            L.getWeights()[0] = 2.f;

            const std::vector<float> x = {1.f, 2.f, 3.f, 4.f};
            const std::vector<const std::vector<float>*> inputs = {&x};
            std::vector<std::vector<float>> outputs;
            const bool ok = rt.forwardLayer(inputs, outputs, L, false);
            TASSERT_TRUE(ok);
            TASSERT_TRUE(outputs.size() == 1 && outputs[0].size() == 4);
            TASSERT_NEAR(outputs[0][0], 2.f, 1e-4f);
            TASSERT_NEAR(outputs[0][1], 4.f, 1e-4f);
            TASSERT_NEAR(outputs[0][2], 6.f, 1e-4f);
            TASSERT_NEAR(outputs[0][3], 8.f, 1e-4f);
        }
    }
#endif  // ENABLE_CUDA

#ifdef ENABLE_ROCM
    {
        RocmRuntime rt;
        RuntimeConfig cfg;
        cfg.conv_enabled = true;
        cfg.conv_min_ops = 0;
        if (!rt.initialize(cfg)) {
            std::cout << "SKIP: no ROCm device (Conv2d)\n";
        } else {
            Model m;
            m.push("x0", "Conv2d", 0);
            auto& L = m.getMutableLayers()[0];
            L.in_channels  = 1;
            L.out_channels = 1;
            L.input_height = 2;
            L.input_width  = 2;
            L.kernel_size  = 1;
            L.stride       = 1;
            L.padding      = 0;
            L.use_bias     = false;
            L.params_count = 1;
            m.allocateParams();
            L.getWeights()[0] = 2.f;

            const std::vector<float> x = {1.f, 2.f, 3.f, 4.f};
            const std::vector<const std::vector<float>*> inputs = {&x};
            std::vector<std::vector<float>> outputs;
            const bool ok = rt.forwardLayer(inputs, outputs, L, false);
            TASSERT_TRUE(ok);
            TASSERT_TRUE(outputs.size() == 1 && outputs[0].size() == 4);
            TASSERT_NEAR(outputs[0][0], 2.f, 1e-4f);
            TASSERT_NEAR(outputs[0][1], 4.f, 1e-4f);
            TASSERT_NEAR(outputs[0][2], 6.f, 1e-4f);
            TASSERT_NEAR(outputs[0][3], 8.f, 1e-4f);
        }
    }
#endif  // ENABLE_ROCM

    return 0;
}
