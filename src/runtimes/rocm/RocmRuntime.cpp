#include "runtimes/optimizers/RocmOptimizer.hpp"
#include "runtimes/rocm/RocmRuntime.hpp"

#include "runtimes/LayerOps.hpp"
#include "runtimes/cpu/RuntimeLayerDispatch.hpp"

#ifdef ENABLE_ROCM
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>
#include <mutex>
#include <array>
#include <vector>

#ifdef ENABLE_ROCM
static void im2col_cpu(
    const float* __restrict__ src, float* __restrict__ col,
    int C, int H, int W,
    int k, int stride, int pad, int dil,
    int H_out, int W_out)
{
    const int HW = H_out * W_out;
    for (int c = 0; c < C; ++c) {
        const float* src_c = src + c * H * W;
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                float* col_row = col + (c * k * k + kh * k + kw) * HW;
                for (int oh = 0; oh < H_out; ++oh) {
                    const int ih = oh * stride + kh * dil - pad;
                    const bool ih_ok = (ih >= 0 && ih < H);
                    for (int ow = 0; ow < W_out; ++ow) {
                        const int iw = ow * stride + kw * dil - pad;
                        col_row[oh * W_out + ow] =
                            (ih_ok && iw >= 0 && iw < W) ? src_c[ih * W + iw] : 0.0f;
                    }
                }
            }
        }
    }
}

static void softmax_rows(float* A, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        float* row = A + i * cols;
        float mx = row[0];
        for (int j = 1; j < cols; ++j) if (row[j] > mx) mx = row[j];
        float s = 0.0f;
        for (int j = 0; j < cols; ++j) { row[j] = std::exp(row[j] - mx); s += row[j]; }
        const float inv = 1.0f / (s > 0.0f ? s : 1.0f);
        for (int j = 0; j < cols; ++j) row[j] *= inv;
    }
}
#endif // ENABLE_ROCM

struct RocmRuntime::Impl {
#ifdef ENABLE_ROCM
    rocblas_handle handle = nullptr;

    struct DeviceBuf {
        void* ptr = nullptr;
        size_t bytes = 0;

        ~DeviceBuf() { reset(); }
        DeviceBuf() = default;
        DeviceBuf(const DeviceBuf&) = delete;
        DeviceBuf& operator=(const DeviceBuf&) = delete;

        void reset() {
            if (ptr) {
                hipFree(ptr);
                ptr = nullptr;
                bytes = 0;
            }
        }

        bool alloc(size_t nbytes) {
            if (nbytes && ptr && nbytes <= bytes) return true;
            reset();
            if (nbytes == 0) return true;
            if (hipMalloc(&ptr, nbytes) != hipSuccess) {
                ptr = nullptr;
                bytes = 0;
                return false;
            }
            bytes = nbytes;
            return true;
        }

        bool copyFromHost(const void* src, size_t nbytes) {
            if (!ptr || nbytes > bytes) return false;
            return hipMemcpy(ptr, src, nbytes, hipMemcpyHostToDevice) == hipSuccess;
        }

        bool copyToHost(void* dst, size_t nbytes) {
            if (!ptr || nbytes > bytes) return false;
            return hipMemcpy(dst, ptr, nbytes, hipMemcpyDeviceToHost) == hipSuccess;
        }
    };

    struct CachedBuf {
        DeviceBuf buf;
        size_t count = 0;

        bool ensure(size_t n) {
            if (count == n && buf.ptr) return true;
            if (!buf.alloc(n * sizeof(float))) return false;
            count = n;
            return true;
        }
    };

    // Scratch is shared by consecutive Linear calls, never by tensor identity:
    // parameters are uploaded every time, including external in-place updates.
    std::array<DeviceBuf, 5> linear_scratch;
    std::mutex linear_scratch_mutex;
    size_t ones_ready_count = 0;
    CachedBuf device_bias;
    CachedBuf device_ones;
    std::vector<float> host_ones;
#endif

    bool initialized = false;
};

#ifdef ENABLE_ROCM
enum class RocmUnaryOp : int {
    ReLU = 0,
    LeakyReLU = 1,
    Sigmoid = 2,
    Tanh = 3,
    SiLU = 4,
    GELU = 5,
    Softplus = 6,
    Mish = 7,
    HardSigmoid = 8,
    HardSwish = 9,
};

enum class RocmBinaryOp : int {
    Add = 0,
    Subtract = 1,
    Multiply = 2,
    Divide = 3,
};

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_HCC__)
__global__ static void rocm_unary_kernel(const float* in, float* out, int n, int op, float alpha) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const float x = in[i];
    switch (op) {
        case static_cast<int>(RocmUnaryOp::ReLU):
            out[i] = x > 0.0f ? x : 0.0f;
            break;
        case static_cast<int>(RocmUnaryOp::LeakyReLU):
            out[i] = x > 0.0f ? x : alpha * x;
            break;
        case static_cast<int>(RocmUnaryOp::Sigmoid):
            out[i] = 1.0f / (1.0f + expf(-x));
            break;
        case static_cast<int>(RocmUnaryOp::Tanh):
            out[i] = tanhf(x);
            break;
        case static_cast<int>(RocmUnaryOp::SiLU): {
            const float s = 1.0f / (1.0f + expf(-x));
            out[i] = x * s;
            break;
        }
        case static_cast<int>(RocmUnaryOp::GELU): {
            const float c = 0.7978845608f;
            const float x3 = x * x * x;
            out[i] = 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x3)));
            break;
        }
        case static_cast<int>(RocmUnaryOp::Softplus):
            out[i] = x > 20.0f ? x : log1pf(expf(x));
            break;
        case static_cast<int>(RocmUnaryOp::Mish): {
            const float sp = x > 20.0f ? x : log1pf(expf(x));
            out[i] = x * tanhf(sp);
            break;
        }
        case static_cast<int>(RocmUnaryOp::HardSigmoid): {
            const float hs = (x + 3.0f) / 6.0f;
            out[i] = fminf(1.0f, fmaxf(0.0f, hs));
            break;
        }
        case static_cast<int>(RocmUnaryOp::HardSwish): {
            const float hs = fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
            out[i] = x * hs;
            break;
        }
        default:
            out[i] = x;
            break;
    }
}

__global__ static void rocm_binary_kernel(const float* a, const float* b, float* out, int n, int op) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    switch (op) {
        case static_cast<int>(RocmBinaryOp::Add):
            out[i] = a[i] + b[i];
            break;
        case static_cast<int>(RocmBinaryOp::Subtract):
            out[i] = a[i] - b[i];
            break;
        case static_cast<int>(RocmBinaryOp::Multiply):
            out[i] = a[i] * b[i];
            break;
        case static_cast<int>(RocmBinaryOp::Divide): {
            const float d = b[i];
            out[i] = (fabsf(d) > 1e-12f) ? (a[i] / d) : 0.0f;
            break;
        }
        default:
            out[i] = a[i];
            break;
    }
}
#endif

static void rocm_apply_unary_host(const float* input, float* output, int n, RocmUnaryOp op, float alpha) {
    for (int i = 0; i < n; ++i) {
        const float x = input[i];
        switch (op) {
            case RocmUnaryOp::ReLU:
                output[i] = x > 0.0f ? x : 0.0f;
                break;
            case RocmUnaryOp::LeakyReLU:
                output[i] = x > 0.0f ? x : alpha * x;
                break;
            case RocmUnaryOp::Sigmoid:
                output[i] = 1.0f / (1.0f + std::exp(-x));
                break;
            case RocmUnaryOp::Tanh:
                output[i] = std::tanh(x);
                break;
            case RocmUnaryOp::SiLU: {
                const float s = 1.0f / (1.0f + std::exp(-x));
                output[i] = x * s;
                break;
            }
            case RocmUnaryOp::GELU: {
                const float c = 0.7978845608f;
                const float x3 = x * x * x;
                output[i] = 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x3)));
                break;
            }
            case RocmUnaryOp::Softplus:
                output[i] = std::log1p(std::exp(x));
                break;
            case RocmUnaryOp::Mish: {
                const float sp = std::log1p(std::exp(x));
                output[i] = x * std::tanh(sp);
                break;
            }
            case RocmUnaryOp::HardSigmoid: {
                const float hs = (x + 3.0f) / 6.0f;
                output[i] = std::min(1.0f, std::max(0.0f, hs));
                break;
            }
            case RocmUnaryOp::HardSwish: {
                const float hs = std::min(1.0f, std::max(0.0f, (x + 3.0f) / 6.0f));
                output[i] = x * hs;
                break;
            }
            default:
                output[i] = x;
                break;
        }
    }
}

static void rocm_apply_binary_host(const float* a, const float* b, float* output, int n, RocmBinaryOp op) {
    for (int i = 0; i < n; ++i) {
        switch (op) {
            case RocmBinaryOp::Add:
                output[i] = a[i] + b[i];
                break;
            case RocmBinaryOp::Subtract:
                output[i] = a[i] - b[i];
                break;
            case RocmBinaryOp::Multiply:
                output[i] = a[i] * b[i];
                break;
            case RocmBinaryOp::Divide: {
                const float d = b[i];
                output[i] = (std::fabs(d) > 1e-12f) ? (a[i] / d) : 0.0f;
                break;
            }
            default:
                output[i] = a[i];
                break;
        }
    }
}

static bool rocm_run_unary(const std::vector<float>& input, std::vector<float>& output, RocmUnaryOp op, float alpha) {
    const int n = static_cast<int>(input.size());
    if (n <= 0) return false;

    float* d_in = nullptr;
    float* d_out = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    if (hipMalloc(&d_in, bytes) != hipSuccess) return false;
    if (hipMalloc(&d_out, bytes) != hipSuccess) {
        hipFree(d_in);
        return false;
    }

    bool ok = true;
    if (hipMemcpy(d_in, input.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) ok = false;

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_HCC__)
    if (ok) {
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_unary_kernel, dim3(blocks), dim3(threads), 0, 0, d_in, d_out, n, static_cast<int>(op), alpha);
        if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) ok = false;
    }

    if (ok) {
        output.resize(static_cast<size_t>(n));
        if (hipMemcpy(output.data(), d_out, bytes, hipMemcpyDeviceToHost) != hipSuccess) ok = false;
    }
#else
    output.resize(static_cast<size_t>(n));
    rocm_apply_unary_host(input.data(), output.data(), n, op, alpha);
    ok = true;
#endif

    static_cast<void>(hipFree(d_in));
    static_cast<void>(hipFree(d_out));
    return ok;
}

static bool rocm_run_binary(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& output, RocmBinaryOp op) {
    const int n = static_cast<int>(a.size());
    if (n <= 0 || a.size() != b.size()) return false;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_out = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    if (hipMalloc(&d_a, bytes) != hipSuccess) return false;
    if (hipMalloc(&d_b, bytes) != hipSuccess) {
        hipFree(d_a);
        return false;
    }
    if (hipMalloc(&d_out, bytes) != hipSuccess) {
        hipFree(d_a);
        hipFree(d_b);
        return false;
    }

    bool ok = true;
    if (hipMemcpy(d_a, a.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) ok = false;
    if (ok && hipMemcpy(d_b, b.data(), bytes, hipMemcpyHostToDevice) != hipSuccess) ok = false;

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_HCC__)
    if (ok) {
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_binary_kernel, dim3(blocks), dim3(threads), 0, 0, d_a, d_b, d_out, n, static_cast<int>(op));
        if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) ok = false;
    }

    if (ok) {
        output.resize(static_cast<size_t>(n));
        if (hipMemcpy(output.data(), d_out, bytes, hipMemcpyDeviceToHost) != hipSuccess) ok = false;
    }
#else
    output.resize(static_cast<size_t>(n));
    rocm_apply_binary_host(a.data(), b.data(), output.data(), n, op);
    ok = true;
#endif

    static_cast<void>(hipFree(d_a));
    static_cast<void>(hipFree(d_b));
    static_cast<void>(hipFree(d_out));
    return ok;
}
#endif

bool RocmRuntime::initialize(const RuntimeConfig& cfg) {
    config_ = cfg;

#ifndef ENABLE_ROCM
    (void)cfg;
    shutdown();
    return false;
#else
    if (impl_ && impl_->initialized) return true;

    if (config_.disabled) {
        shutdown();
        return false;
    }

    if (!impl_) {
        impl_ = new (std::nothrow) Impl();
        if (!impl_) return false;
    }

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0) {
        shutdown();
        return false;
    }

    int device_index = config_.device_index;
    if (device_index < 0 || device_index >= device_count) {
        device_index = 0;
    }

    if (hipSetDevice(device_index) != hipSuccess) {
        shutdown();
        return false;
    }

    if (rocblas_create_handle(&impl_->handle) != rocblas_status_success) {
        shutdown();
        return false;
    }

    impl_->initialized = true;
    return true;
#endif
}

void RocmRuntime::shutdown() {
#ifdef ENABLE_ROCM
    if (impl_) {
        if (impl_->handle) {
            rocblas_destroy_handle(impl_->handle);
            impl_->handle = nullptr;
        }
        impl_->device_bias.buf.reset();
        impl_->device_ones.buf.reset();
        impl_->host_ones.clear();
        impl_->initialized = false;
    }
#endif

    delete impl_;
    impl_ = nullptr;
}

bool RocmRuntime::isInitialized() const {
    return impl_ && impl_->initialized;
}

bool RocmRuntime::supportsForwardLayerType(const LayerType type) const {
    switch (config_.disabled) {
        case true:
            return false;
        case false:
            break;
    }

    switch (type) {
        case LayerType::UNKNOWN:
            return false;
        default:
            return RuntimeLayerDispatch::cpu_supports_forward_layer_type(type);
    }
}

bool RocmRuntime::supportsBackwardLayerType(const LayerType type) const {
    switch (config_.disabled) {
        case true:
            return false;
        case false:
            break;
    }

    switch (type) {
        case LayerType::UNKNOWN:
            return false;
        default:
            return RuntimeLayerDispatch::cpu_supports_backward_layer_type(type);
    }
}

RuntimeCapabilityLevel RocmRuntime::queryForwardCapability(const LayerType type) const {
    if (!supportsForwardLayerType(type)) return RuntimeCapabilityLevel::Unsupported;
#ifndef ENABLE_ROCM
    return RuntimeCapabilityLevel::HostFallback;
#else
    switch (type) {
        case LayerType::Linear:
        case LayerType::MatMul:
        case LayerType::BatchMatMul:
        case LayerType::Conv2d:
        case LayerType::LayerNorm:
        case LayerType::RMSNorm:
        case LayerType::SelfAttention:
        case LayerType::MultiHeadAttention:
            return RuntimeCapabilityLevel::NativeOptimized;
        default:
            return RuntimeCapabilityLevel::HostFallback;
    }
#endif
}

RuntimeCapabilityLevel RocmRuntime::queryBackwardCapability(const LayerType type) const {
    if (!supportsBackwardLayerType(type)) return RuntimeCapabilityLevel::Unsupported;
#ifndef ENABLE_ROCM
    return RuntimeCapabilityLevel::HostFallback;
#else
    switch (type) {
        case LayerType::Linear:
        case LayerType::MatMul:
        case LayerType::BatchMatMul:
        case LayerType::Add:
        case LayerType::Subtract:
        case LayerType::Multiply:
        case LayerType::Divide:
        case LayerType::ReLU:
        case LayerType::LeakyReLU:
        case LayerType::Sigmoid:
        case LayerType::Tanh:
        case LayerType::SiLU:
        case LayerType::GELU:
        case LayerType::Softplus:
        case LayerType::Mish:
        case LayerType::HardSigmoid:
        case LayerType::HardSwish:
            return RuntimeCapabilityLevel::Native;
        default:
            return RuntimeCapabilityLevel::HostFallback;
    }
#endif
}

RuntimeCapabilityLevel RocmRuntime::queryForwardOperationCapability(
    const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs,
    bool training
) const {
    (void)training;
    return queryConfiguredForwardOperationCapability(layer, inputs, true);
}

bool RocmRuntime::linearForward(
    const float* input,
    const float* weights,
    const float* bias_or_null,
    float* output,
    int batch,
    int in_f,
    int out_f
) {
#ifndef ENABLE_ROCM
    (void)input; (void)weights; (void)bias_or_null; (void)output; (void)batch; (void)in_f; (void)out_f;
    return false;
#else
    if (!isInitialized() || !impl_->handle) return false;
    if (!input || !weights || !output) return false;
    if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;

    const size_t bytes_in = static_cast<size_t>(batch) * static_cast<size_t>(in_f) * sizeof(float);
    const size_t bytes_w = static_cast<size_t>(out_f) * static_cast<size_t>(in_f) * sizeof(float);
    const size_t bytes_out = static_cast<size_t>(batch) * static_cast<size_t>(out_f) * sizeof(float);

    std::lock_guard<std::mutex> scratch_lock(impl_->linear_scratch_mutex);
    auto& d_in = impl_->linear_scratch[0];
    auto& d_w = impl_->linear_scratch[1];
    auto& d_out = impl_->linear_scratch[2];

    if (!d_in.alloc(bytes_in)) return false;
    if (!d_w.alloc(bytes_w)) return false;
    if (!d_out.alloc(bytes_out)) return false;

    if (!d_in.copyFromHost(input, bytes_in)) return false;
    if (!d_w.copyFromHost(weights, bytes_w)) return false;

    // rocBLAS column-major: C(out_f x batch) = A^T(out_f x in_f) * B(in_f x batch)
    const float alpha = 1.0f;
    const float beta0 = 0.0f;

    if (rocblas_sgemm(
            impl_->handle,
            rocblas_operation_transpose,
            rocblas_operation_none,
            out_f,
            batch,
            in_f,
            &alpha,
            reinterpret_cast<const float*>(d_w.ptr),
            in_f,
            reinterpret_cast<const float*>(d_in.ptr),
            in_f,
            &beta0,
            reinterpret_cast<float*>(d_out.ptr),
            out_f
        ) != rocblas_status_success) {
        return false;
    }

    if (bias_or_null) {
        if (!impl_->device_bias.ensure(static_cast<size_t>(out_f))) return false;
        if (!impl_->device_bias.buf.copyFromHost(bias_or_null, static_cast<size_t>(out_f) * sizeof(float))) return false;

        // A failed growth may have released the old allocation. Invalidate the
        // upload marker before ensure(), so a later smaller call reinitializes it.
        if (impl_->device_ones.count != static_cast<size_t>(batch) || !impl_->device_ones.buf.ptr)
            impl_->ones_ready_count = 0;
        if (!impl_->device_ones.ensure(static_cast<size_t>(batch))) return false;
        if (static_cast<int>(impl_->host_ones.size()) != batch) {
            impl_->host_ones.assign(static_cast<size_t>(batch), 1.0f);
        }
        if (impl_->ones_ready_count != static_cast<size_t>(batch)) {
            if (!impl_->device_ones.buf.copyFromHost(impl_->host_ones.data(), static_cast<size_t>(batch) * sizeof(float))) return false;
            impl_->ones_ready_count = static_cast<size_t>(batch);
        }

        const float beta1 = 1.0f;
        const int k = 1;
        if (rocblas_sgemm(
                impl_->handle,
                rocblas_operation_none,
                rocblas_operation_none,
                out_f,
                batch,
                k,
                &alpha,
                reinterpret_cast<const float*>(impl_->device_bias.buf.ptr),
                out_f,
                reinterpret_cast<const float*>(impl_->device_ones.buf.ptr),
                1,
                &beta1,
                reinterpret_cast<float*>(d_out.ptr),
                out_f
            ) != rocblas_status_success) {
            return false;
        }
    }

    // copyToHost is blocking and orders the preceding GEMMs; an additional
    // device-wide synchronization would also stall unrelated work.
    return d_out.copyToHost(output, bytes_out);
#endif
}

bool RocmRuntime::forwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training
) {
    if (!isInitialized()) return false;
    if (config_.disabled) return false;
    if (!supportsForwardLayerType(layer.type_enum)) return false;

    switch (layer.type_enum) {
        case LayerType::Linear:
            if (config_.linear_enabled) {
                if (!inputs.empty() && inputs[0]) {
                    const std::vector<float>& x = *inputs[0];
                    const int in_f = layer.in_features;
                    const int out_f = layer.out_features;
                    const int seq_len = layer.seq_len;
                    const int batch = (seq_len > 0) ? seq_len : (in_f > 0 ? static_cast<int>(x.size()) / in_f : 0);

                    if (batch > 0 && in_f > 0 && out_f > 0) {
                        const size_t expected_in = static_cast<size_t>(batch) * static_cast<size_t>(in_f);
                        const size_t expected_w = static_cast<size_t>(out_f) * static_cast<size_t>(in_f);
                        const size_t expected_total_w = expected_w + (layer.use_bias ? static_cast<size_t>(out_f) : 0ULL);

                        const long long ops = static_cast<long long>(batch) * static_cast<long long>(in_f) * static_cast<long long>(out_f);
                        if (ops >= std::max(0, config_.linear_min_ops) && x.size() == expected_in) {
                            const float* w = layer.getWeights();
                            if (w && layer.getWeightsSize() >= expected_total_w) {
                                const float* b = layer.use_bias ? (w + expected_w) : nullptr;
                                outputs.resize(1);
                                outputs[0].assign(static_cast<size_t>(batch) * static_cast<size_t>(out_f), 0.0f);
                                if (linearForward(x.data(), w, b, outputs[0].data(), batch, in_f, out_f)) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
            break;
        default:
            break;
    }

    // Fast-path: MatMul / BatchMatMul sur GPU (piloté par le flag linear).
    switch (layer.type_enum) {
        case LayerType::MatMul:
        case LayerType::BatchMatMul:
            if (!config_.linear_enabled) break;
            do {
            if (inputs.size() < 2 || !inputs[0] || !inputs[1]) break;
            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& B = *inputs[1];

            const int M = layer.in_features;
            const int K = layer.out_features;
            const int N = layer.embed_dim;
            if (M <= 0 || K <= 0 || N <= 0) break;

            int batches = 1;
            bool valid_batches = true;
            switch (layer.type_enum) {
                case LayerType::BatchMatMul:
                    batches = layer.seq_len;
                    valid_batches = (batches > 0);
                    break;
                case LayerType::MatMul:
                    break;
                default:
                    break;
            }
            if (!valid_batches) break;

            const size_t a_batch_elems = static_cast<size_t>(M) * static_cast<size_t>(K);
            const size_t b_batch_elems = static_cast<size_t>(K) * static_cast<size_t>(N);
            const size_t c_batch_elems = static_cast<size_t>(M) * static_cast<size_t>(N);

            if (A.size() != static_cast<size_t>(batches) * a_batch_elems) break;
            if (B.size() != static_cast<size_t>(batches) * b_batch_elems) break;

            const long long ops = static_cast<long long>(batches) * static_cast<long long>(M) *
                                  static_cast<long long>(K) * static_cast<long long>(N);
            if (ops < static_cast<long long>(std::max(0, config_.linear_min_ops))) break;

#ifdef ENABLE_ROCM
            Impl::DeviceBuf d_a, d_b, d_c;
            if (!d_a.alloc(static_cast<size_t>(batches) * a_batch_elems * sizeof(float))) break;
            if (!d_b.alloc(static_cast<size_t>(batches) * b_batch_elems * sizeof(float))) break;
            if (!d_c.alloc(static_cast<size_t>(batches) * c_batch_elems * sizeof(float))) break;
            if (!d_a.copyFromHost(A.data(), static_cast<size_t>(batches) * a_batch_elems * sizeof(float))) break;
            if (!d_b.copyFromHost(B.data(), static_cast<size_t>(batches) * b_batch_elems * sizeof(float))) break;

            // Row-major mapping via column-major GEMM:
            // C[M,N] = A[M,K] * B[K,N]  <=>  C_cm[N,M] = B_cm[N,K] * A_cm[K,M]
            const float alpha = 1.0f;
            const float beta = 0.0f;
            bool ok = true;
            for (int bi = 0; bi < batches; ++bi) {
                const float* a_ptr = reinterpret_cast<const float*>(d_a.ptr) + static_cast<size_t>(bi) * a_batch_elems;
                const float* b_ptr = reinterpret_cast<const float*>(d_b.ptr) + static_cast<size_t>(bi) * b_batch_elems;
                float* c_ptr = reinterpret_cast<float*>(d_c.ptr) + static_cast<size_t>(bi) * c_batch_elems;
                if (rocblas_sgemm(
                        impl_->handle,
                        rocblas_operation_none,
                        rocblas_operation_none,
                        N,
                        M,
                        K,
                        &alpha,
                        b_ptr,
                        N,
                        a_ptr,
                        K,
                        &beta,
                        c_ptr,
                        N
                    ) != rocblas_status_success) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
            if (hipDeviceSynchronize() != hipSuccess) break;

            outputs.resize(1);
            outputs[0].resize(static_cast<size_t>(batches) * c_batch_elems);
            if (!d_c.copyToHost(outputs[0].data(), static_cast<size_t>(batches) * c_batch_elems * sizeof(float))) break;
            return true;
#endif
            } while (false);
            break;
        default:
            break;
    }

#ifdef ENABLE_ROCM
    // ─────────────────────────────────────────────────────────────────────────
    // Fast-path: Conv2d  (im2col + rocBLAS SGEMM)
    // ─────────────────────────────────────────────────────────────────────────
    switch (layer.type_enum) {
        case LayerType::Conv2d:
            if (!config_.conv_enabled) break;
            do {
            if (inputs.empty() || !inputs[0]) break;
            const std::vector<float>& xc = *inputs[0];
            const int C_in  = std::max(1, layer.in_channels);
            const int C_out = std::max(1, layer.out_channels);
            const int k     = std::max(1, layer.kernel_h > 0 ? layer.kernel_h : layer.get_kernel_h());
            const int str   = std::max(1, layer.stride_h > 0 ? layer.stride_h : layer.get_stride_h());
            const int pad   = std::max(0, layer.pad_h >= 0 ? layer.pad_h : layer.get_pad_h());
            const int dil   = std::max(1, layer.dilation_h > 0 ? layer.dilation_h : 1);
            const int H     = std::max(1, layer.input_height);
            const int W     = std::max(1, layer.input_width);
            const int H_out = (H + 2*pad - dil*(k-1) - 1) / str + 1;
            const int W_out = (W + 2*pad - dil*(k-1) - 1) / str + 1;
            if (H_out <= 0 || W_out <= 0) break;

            const int K  = C_in * k * k;
            const int HW = H_out * W_out;
            const long long ops = static_cast<long long>(C_out) * K * HW;
            if (ops < static_cast<long long>(config_.conv_min_ops)) break;
            if (static_cast<int>(xc.size()) != C_in * H * W) break;

            const size_t w_k = static_cast<size_t>(C_out) * K;
            const float* wc = layer.getWeights();
            if (!wc || layer.getWeightsSize() < w_k + (layer.use_bias ? C_out : 0u)) break;

            const size_t col_sz = static_cast<size_t>(K) * HW;
            std::vector<float> col_buf(col_sz);
            im2col_cpu(xc.data(), col_buf.data(), C_in, H, W, k, str, pad, dil, H_out, W_out);

            Impl::DeviceBuf d_col, d_w, d_out;
            if (!d_col.alloc(col_sz * sizeof(float))) break;
            if (!d_w.alloc(w_k * sizeof(float))) break;
            if (!d_out.alloc(static_cast<size_t>(C_out) * HW * sizeof(float))) break;
            if (!d_col.copyFromHost(col_buf.data(), col_sz * sizeof(float))) break;
            if (!d_w.copyFromHost(wc, w_k * sizeof(float))) break;

            const float cv_a = 1.0f, cv_b0 = 0.0f;
            if (rocblas_sgemm(impl_->handle,
                    rocblas_operation_none, rocblas_operation_none,
                    HW, C_out, K, &cv_a,
                    reinterpret_cast<const float*>(d_col.ptr), HW,
                    reinterpret_cast<const float*>(d_w.ptr), K,
                    &cv_b0,
                    reinterpret_cast<float*>(d_out.ptr), HW) != rocblas_status_success) break;

            if (layer.use_bias) {
                std::vector<float> ones_hw(static_cast<size_t>(HW), 1.0f);
                Impl::DeviceBuf d_bias, d_ones_hw;
                if (!d_bias.alloc(static_cast<size_t>(C_out) * sizeof(float))) break;
                if (!d_ones_hw.alloc(static_cast<size_t>(HW) * sizeof(float))) break;
                if (!d_bias.copyFromHost(wc + w_k, static_cast<size_t>(C_out) * sizeof(float))) break;
                if (!d_ones_hw.copyFromHost(ones_hw.data(), static_cast<size_t>(HW) * sizeof(float))) break;
                if (rocblas_sger(impl_->handle, HW, C_out, &cv_a,
                        reinterpret_cast<const float*>(d_ones_hw.ptr), 1,
                        reinterpret_cast<const float*>(d_bias.ptr), 1,
                        reinterpret_cast<float*>(d_out.ptr), HW) != rocblas_status_success) break;
            }

            if (hipDeviceSynchronize() != hipSuccess) break;
            outputs.resize(1);
            outputs[0].resize(static_cast<size_t>(C_out) * HW);
            if (!d_out.copyToHost(outputs[0].data(), static_cast<size_t>(C_out) * HW * sizeof(float))) break;
            return true;
            } while (false);
            break;
        default:
            break;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fast-path: LayerNorm / RMSNorm  (stats sur host, affine sur GPU)
    // ─────────────────────────────────────────────────────────────────────────
    switch (layer.type_enum) {
        case LayerType::LayerNorm:
        case LayerType::RMSNorm:
            if (!config_.norm_enabled) break;
            do {
            if (inputs.empty() || !inputs[0]) break;
            const std::vector<float>& xn = *inputs[0];
            const int N = static_cast<int>(xn.size());
            if (N < config_.norm_min_elements) break;

            const int norm_sz = (layer.in_features > 0) ? layer.in_features : N;
            if (norm_sz <= 0 || N % norm_sz != 0) break;
            const int groups = N / norm_sz;

            const float* gw = layer.getWeights();
            const size_t required_weights = static_cast<size_t>(norm_sz) *
                (layer.use_bias ? 2ULL : 1ULL);
            if (layer.affine && (!gw || layer.getWeightsSize() < required_weights)) break;

            const float eps = std::max(layer.eps, 0.0f);
            std::vector<float> x_hat(static_cast<size_t>(N));

            switch (layer.type_enum) {
                case LayerType::RMSNorm:
                    for (int g = 0; g < groups; ++g) {
                        const int base = g * norm_sz;
                        float ss = 0.0f;
                        for (int i = 0; i < norm_sz; ++i) { float v = xn[base+i]; ss += v*v; }
                        const float inv = 1.0f / std::sqrt(ss / norm_sz + eps);
                        for (int i = 0; i < norm_sz; ++i) x_hat[base+i] = xn[base+i] * inv;
                    }
                    break;
                case LayerType::LayerNorm:
                    for (int g = 0; g < groups; ++g) {
                        const int base = g * norm_sz;
                        float mean = 0.0f;
                        for (int i = 0; i < norm_sz; ++i) mean += xn[base+i];
                        mean /= norm_sz;
                        float var = 0.0f;
                        for (int i = 0; i < norm_sz; ++i) { float d = xn[base+i]-mean; var += d*d; }
                        const float inv = 1.0f / std::sqrt(var/norm_sz + eps);
                        for (int i = 0; i < norm_sz; ++i) x_hat[base+i] = (xn[base+i]-mean)*inv;
                    }
                    break;
                default:
                    break;
            }

            if (!layer.affine) {
                outputs = {std::move(x_hat)};
                return true;
            }

            Impl::DeviceBuf d_xh, d_gamma, d_out;
            if (!d_xh.alloc(static_cast<size_t>(N) * sizeof(float))) break;
            if (!d_gamma.alloc(static_cast<size_t>(norm_sz) * sizeof(float))) break;
            if (!d_out.alloc(static_cast<size_t>(N) * sizeof(float))) break;
            if (!d_xh.copyFromHost(x_hat.data(), static_cast<size_t>(N) * sizeof(float))) break;
            if (!d_gamma.copyFromHost(gw, static_cast<size_t>(norm_sz) * sizeof(float))) break;

            bool ok = true;
            for (int g = 0; g < groups && ok; ++g) {
                const int base = g * norm_sz;
                if (rocblas_sdgmm(impl_->handle, rocblas_side_left,
                        norm_sz, 1,
                        reinterpret_cast<const float*>(d_xh.ptr) + base, norm_sz,
                        reinterpret_cast<const float*>(d_gamma.ptr), 1,
                        reinterpret_cast<float*>(d_out.ptr) + base, norm_sz
                    ) != rocblas_status_success) ok = false;
            }
            if (!ok) break;

            if (layer.use_bias && static_cast<int>(layer.getWeightsSize()) >= 2 * norm_sz) {
                Impl::DeviceBuf d_beta;
                if (!d_beta.alloc(static_cast<size_t>(norm_sz) * sizeof(float))) break;
                if (!d_beta.copyFromHost(gw + norm_sz, static_cast<size_t>(norm_sz) * sizeof(float))) break;
                const float na = 1.0f;
                for (int g = 0; g < groups && ok; ++g) {
                    const int base = g * norm_sz;
                    if (rocblas_saxpy(impl_->handle, norm_sz, &na,
                            reinterpret_cast<const float*>(d_beta.ptr), 1,
                            reinterpret_cast<float*>(d_out.ptr) + base, 1
                        ) != rocblas_status_success) ok = false;
                }
                if (!ok) break;
            }

            if (hipDeviceSynchronize() != hipSuccess) break;
            outputs.resize(1);
            outputs[0].resize(static_cast<size_t>(N));
            if (!d_out.copyToHost(outputs[0].data(), static_cast<size_t>(N) * sizeof(float))) break;
            return true;
            } while (false);
            break;
        default:
            break;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fast-path: Element-wise binaire + activations unaires
    // ─────────────────────────────────────────────────────────────────────────
    switch (layer.type_enum) {
        case LayerType::Add:
        case LayerType::Subtract:
        case LayerType::Multiply:
        case LayerType::Divide:
            if (!config_.linear_enabled) break;
            do {
            if (inputs.size() < 2 || !inputs[0] || !inputs[1]) break;
            const std::vector<float>& a = *inputs[0];
            const std::vector<float>& b = *inputs[1];
            if (a.empty() || a.size() != b.size()) break;
            if (static_cast<long long>(a.size()) < static_cast<long long>(std::max(0, config_.linear_min_ops))) break;

            int op_code = 0;
            if (!RuntimeLayerOps::resolveBinaryOp(layer.type_enum, op_code)) break;
            const RocmBinaryOp op = static_cast<RocmBinaryOp>(op_code);

            outputs.resize(1);
            if (!rocm_run_binary(a, b, outputs[0], op)) {
                RuntimeLayerOps::binaryForwardHost(a, b, outputs[0], op_code);
            }
            return true;
            } while (false);
            break;

        case LayerType::ReLU:
        case LayerType::LeakyReLU:
        case LayerType::Sigmoid:
        case LayerType::Tanh:
        case LayerType::SiLU:
        case LayerType::GELU:
        case LayerType::Softplus:
        case LayerType::Mish:
        case LayerType::HardSigmoid:
        case LayerType::HardSwish:
            if (!config_.linear_enabled) break;
            do {
            if (inputs.empty() || !inputs[0]) break;
            const std::vector<float>& in = *inputs[0];
            if (in.empty()) break;
            if (static_cast<long long>(in.size()) < static_cast<long long>(std::max(0, config_.linear_min_ops))) break;

            int op_code = 0;
            float alpha = 0.01f;
            if (!RuntimeLayerOps::resolveUnaryOp(layer.type_enum, layer, op_code, alpha)) break;
            const RocmUnaryOp op = static_cast<RocmUnaryOp>(op_code);

            outputs.resize(1);
            if (!rocm_run_unary(in, outputs[0], op, alpha)) {
                RuntimeLayerOps::unaryForwardHost(in, outputs[0], op_code, alpha);
            }
            return true;
            } while (false);
            break;

        default:
            break;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fast-path: SelfAttention / MultiHeadAttention
    // ─────────────────────────────────────────────────────────────────────────
    switch (layer.type_enum) {
        case LayerType::SelfAttention:
        case LayerType::MultiHeadAttention:
            if (!config_.attention_enabled) break;
            do {
            if (inputs.empty() || !inputs[0]) break;
            const std::vector<float>& xa = *inputs[0];

            const int seq = layer.seq_len > 0 ? layer.seq_len : 1;
            const int ed  = layer.embed_dim > 0 ? layer.embed_dim : static_cast<int>(xa.size());
            const int nh  = layer.num_heads > 0 ? layer.num_heads : 1;
            if (ed <= 0 || nh <= 0 || ed % nh != 0) break;
            const int hd = ed / nh;
            const long long attn_ops = 4LL * seq * ed * ed;
            if (attn_ops < static_cast<long long>(config_.attention_min_ops)) break;
            if (static_cast<int>(xa.size()) != seq * ed) break;

            const float* wa = layer.getWeights();
            if (!wa || static_cast<int>(layer.getWeightsSize()) < ed * ed * 4) break;
            const int qkv_d = ed * 3;
            const float* wqkv = wa;
            const float* wout = wa + ed * ed * 3;

            const float aa = 1.0f, ab0 = 0.0f;

            Impl::DeviceBuf d_xa, d_wqkv, d_qkv;
            if (!d_xa.alloc(static_cast<size_t>(seq) * ed * sizeof(float))) break;
            if (!d_wqkv.alloc(static_cast<size_t>(ed) * qkv_d * sizeof(float))) break;
            if (!d_qkv.alloc(static_cast<size_t>(seq) * qkv_d * sizeof(float))) break;
            if (!d_xa.copyFromHost(xa.data(), static_cast<size_t>(seq) * ed * sizeof(float))) break;
            if (!d_wqkv.copyFromHost(wqkv, static_cast<size_t>(ed) * qkv_d * sizeof(float))) break;

            if (rocblas_sgemm(impl_->handle,
                    rocblas_operation_none, rocblas_operation_none,
                    qkv_d, seq, ed, &aa,
                    reinterpret_cast<const float*>(d_wqkv.ptr), qkv_d,
                    reinterpret_cast<const float*>(d_xa.ptr), ed,
                    &ab0,
                    reinterpret_cast<float*>(d_qkv.ptr), qkv_d) != rocblas_status_success) break;

            if (hipDeviceSynchronize() != hipSuccess) break;
            std::vector<float> qkv_h(static_cast<size_t>(seq) * qkv_d);
            if (!d_qkv.copyToHost(qkv_h.data(), static_cast<size_t>(seq) * qkv_d * sizeof(float))) break;

            std::vector<float> Q(seq*ed), Km(seq*ed), Vm(seq*ed);
            for (int t = 0; t < seq; ++t) {
                const float* src = &qkv_h[static_cast<size_t>(t) * qkv_d];
                std::copy(src,        src + ed,    &Q[t*ed]);
                std::copy(src + ed,   src + 2*ed,  &Km[t*ed]);
                std::copy(src + 2*ed, src + qkv_d, &Vm[t*ed]);
            }

            std::vector<float> attended(static_cast<size_t>(seq) * ed, 0.0f);
            const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
            bool aok = true;

            for (int h = 0; h < nh && aok; ++h) {
                const int hoff = h * hd;
                std::vector<float> Qh(seq*hd), Kh(seq*hd), Vh(seq*hd);
                for (int t = 0; t < seq; ++t) {
                    std::copy(&Q[t*ed+hoff],  &Q[t*ed+hoff+hd],  &Qh[t*hd]);
                    std::copy(&Km[t*ed+hoff], &Km[t*ed+hoff+hd], &Kh[t*hd]);
                    std::copy(&Vm[t*ed+hoff], &Vm[t*ed+hoff+hd], &Vh[t*hd]);
                }

                Impl::DeviceBuf d_qh, d_kh, d_vh, d_sc, d_ctx;
                if (!d_qh.alloc(seq*hd*sizeof(float)) ||
                    !d_kh.alloc(seq*hd*sizeof(float)) ||
                    !d_vh.alloc(seq*hd*sizeof(float)) ||
                    !d_sc.alloc(seq*seq*sizeof(float)) ||
                    !d_ctx.alloc(seq*hd*sizeof(float))) { aok=false; break; }
                if (!d_qh.copyFromHost(Qh.data(), seq*hd*sizeof(float)) ||
                    !d_kh.copyFromHost(Kh.data(), seq*hd*sizeof(float)) ||
                    !d_vh.copyFromHost(Vh.data(), seq*hd*sizeof(float))) { aok=false; break; }

                if (rocblas_sgemm(impl_->handle,
                        rocblas_operation_transpose, rocblas_operation_none,
                        seq, seq, hd, &scale,
                        reinterpret_cast<const float*>(d_kh.ptr), hd,
                        reinterpret_cast<const float*>(d_qh.ptr), hd,
                        &ab0,
                        reinterpret_cast<float*>(d_sc.ptr), seq) != rocblas_status_success) { aok=false; break; }

                if (hipDeviceSynchronize() != hipSuccess) { aok=false; break; }
                std::vector<float> sc_h(seq * seq);
                if (!d_sc.copyToHost(sc_h.data(), seq*seq*sizeof(float))) { aok=false; break; }

                if (layer.causal) {
                    for (int i = 0; i < seq; ++i) {
                        float* row = &sc_h[i * seq];
                        for (int j = i+1; j < seq; ++j) row[j] = -1e9f;
                    }
                }
                softmax_rows(sc_h.data(), seq, seq);
                if (!d_sc.copyFromHost(sc_h.data(), seq*seq*sizeof(float))) { aok=false; break; }

                if (rocblas_sgemm(impl_->handle,
                        rocblas_operation_none, rocblas_operation_none,
                        hd, seq, seq, &aa,
                        reinterpret_cast<const float*>(d_vh.ptr), hd,
                        reinterpret_cast<const float*>(d_sc.ptr), seq,
                        &ab0,
                        reinterpret_cast<float*>(d_ctx.ptr), hd) != rocblas_status_success) { aok=false; break; }

                if (hipDeviceSynchronize() != hipSuccess) { aok=false; break; }
                std::vector<float> ctx_h(seq * hd);
                if (!d_ctx.copyToHost(ctx_h.data(), seq*hd*sizeof(float))) { aok=false; break; }
                for (int t = 0; t < seq; ++t)
                    std::copy(&ctx_h[t*hd], &ctx_h[(t+1)*hd], &attended[t*ed+hoff]);
            }
            if (!aok) break;

            Impl::DeviceBuf d_att, d_wo, d_final;
            if (!d_att.alloc(static_cast<size_t>(seq)*ed*sizeof(float))) break;
            if (!d_wo.alloc(static_cast<size_t>(ed)*ed*sizeof(float))) break;
            if (!d_final.alloc(static_cast<size_t>(seq)*ed*sizeof(float))) break;
            if (!d_att.copyFromHost(attended.data(), static_cast<size_t>(seq)*ed*sizeof(float))) break;
            if (!d_wo.copyFromHost(wout, static_cast<size_t>(ed)*ed*sizeof(float))) break;

            if (rocblas_sgemm(impl_->handle,
                    rocblas_operation_none, rocblas_operation_none,
                    ed, seq, ed, &aa,
                    reinterpret_cast<const float*>(d_wo.ptr), ed,
                    reinterpret_cast<const float*>(d_att.ptr), ed,
                    &ab0,
                    reinterpret_cast<float*>(d_final.ptr), ed) != rocblas_status_success) break;

            if (hipDeviceSynchronize() != hipSuccess) break;
            outputs.resize(1);
            outputs[0].resize(static_cast<size_t>(seq) * ed);
            if (!d_final.copyToHost(outputs[0].data(), static_cast<size_t>(seq)*ed*sizeof(float))) break;
            return true;
            } while (false);
            break;
        default:
            break;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fast-path: CrossAttention
    // ─────────────────────────────────────────────────────────────────────────
    switch (layer.type_enum) {
        case LayerType::CrossAttention:
            if (!config_.attention_enabled) break;
            if (inputs.size() < 2 || inputs[1] == nullptr) break;
            do {
            const std::vector<float>& q_in  = *inputs[0];
            const std::vector<float>& kv_in = *inputs[1];

            const int nh = layer.num_heads > 0 ? layer.num_heads : 1;
            int ed = layer.embed_dim;
            if (ed <= 0 && layer.head_dim > 0) ed = layer.head_dim * nh;
            if (ed <= 0 || nh <= 0 || ed % nh != 0) break;
            const int hd = ed / nh;

            if (q_in.size() % static_cast<size_t>(ed) != 0) break;
            if (kv_in.size() % static_cast<size_t>(ed) != 0) break;
            const int qlen  = static_cast<int>(q_in.size())  / ed;
            const int kvlen = static_cast<int>(kv_in.size()) / ed;

            const long long cattn_ops = 2LL * qlen * kvlen * ed;
            if (cattn_ops < static_cast<long long>(config_.attention_min_ops)) break;

            const float* wca = layer.getWeights();
            const int q_sz  = ed * ed;
            const int kv_sz = ed * 2 * ed;
            const int o_sz  = ed * ed;
            if (!wca || static_cast<int>(layer.getWeightsSize()) < q_sz + kv_sz + o_sz) break;
            const float* wq  = wca;
            const float* wkv = wca + q_sz;
            const float* wo  = wca + q_sz + kv_sz;

            const float ca = 1.0f, cb0 = 0.0f;

            Impl::DeviceBuf d_qin, d_wq, d_Q;
            if (!d_qin.alloc(qlen*ed*sizeof(float))) break;
            if (!d_wq.alloc(q_sz*sizeof(float))) break;
            if (!d_Q.alloc(qlen*ed*sizeof(float))) break;
            if (!d_qin.copyFromHost(q_in.data(), qlen*ed*sizeof(float))) break;
            if (!d_wq.copyFromHost(wq, q_sz*sizeof(float))) break;
            if (rocblas_sgemm(impl_->handle,
                    rocblas_operation_none, rocblas_operation_none,
                    ed, qlen, ed, &ca,
                    reinterpret_cast<const float*>(d_wq.ptr), ed,
                    reinterpret_cast<const float*>(d_qin.ptr), ed,
                    &cb0,
                    reinterpret_cast<float*>(d_Q.ptr), ed) != rocblas_status_success) break;

            Impl::DeviceBuf d_kvin, d_wkv, d_KV;
            if (!d_kvin.alloc(kvlen*ed*sizeof(float))) break;
            if (!d_wkv.alloc(kv_sz*sizeof(float))) break;
            if (!d_KV.alloc(kvlen*2*ed*sizeof(float))) break;
            if (!d_kvin.copyFromHost(kv_in.data(), kvlen*ed*sizeof(float))) break;
            if (!d_wkv.copyFromHost(wkv, kv_sz*sizeof(float))) break;
            if (rocblas_sgemm(impl_->handle,
                    rocblas_operation_none, rocblas_operation_none,
                    2*ed, kvlen, ed, &ca,
                    reinterpret_cast<const float*>(d_wkv.ptr), 2*ed,
                    reinterpret_cast<const float*>(d_kvin.ptr), ed,
                    &cb0,
                    reinterpret_cast<float*>(d_KV.ptr), 2*ed) != rocblas_status_success) break;

            if (hipDeviceSynchronize() != hipSuccess) break;
            std::vector<float> Q_h(qlen*ed), KV_h(kvlen*2*ed);
            if (!d_Q.copyToHost(Q_h.data(), qlen*ed*sizeof(float))) break;
            if (!d_KV.copyToHost(KV_h.data(), kvlen*2*ed*sizeof(float))) break;

            std::vector<float> K_h(kvlen*ed), V_h(kvlen*ed);
            for (int t = 0; t < kvlen; ++t) {
                std::copy(&KV_h[t*2*ed],     &KV_h[t*2*ed+ed],  &K_h[t*ed]);
                std::copy(&KV_h[t*2*ed+ed],  &KV_h[(t+1)*2*ed], &V_h[t*ed]);
            }

            std::vector<float> attended_c(qlen*ed, 0.0f);
            const float cscale = 1.0f / std::sqrt(static_cast<float>(hd));
            bool cok = true;

            for (int h = 0; h < nh && cok; ++h) {
                const int hoff = h * hd;
                std::vector<float> Qh(qlen*hd), Kh(kvlen*hd), Vh(kvlen*hd);
                for (int t = 0; t < qlen;  ++t) std::copy(&Q_h[t*ed+hoff], &Q_h[t*ed+hoff+hd], &Qh[t*hd]);
                for (int t = 0; t < kvlen; ++t) std::copy(&K_h[t*ed+hoff], &K_h[t*ed+hoff+hd], &Kh[t*hd]);
                for (int t = 0; t < kvlen; ++t) std::copy(&V_h[t*ed+hoff], &V_h[t*ed+hoff+hd], &Vh[t*hd]);

                Impl::DeviceBuf d_qh, d_kh, d_vh, d_sc, d_ctx;
                if (!d_qh.alloc(qlen*hd*sizeof(float))   || !d_kh.alloc(kvlen*hd*sizeof(float)) ||
                    !d_vh.alloc(kvlen*hd*sizeof(float))   || !d_sc.alloc(qlen*kvlen*sizeof(float)) ||
                    !d_ctx.alloc(qlen*hd*sizeof(float))) { cok=false; break; }
                if (!d_qh.copyFromHost(Qh.data(), qlen*hd*sizeof(float))   ||
                    !d_kh.copyFromHost(Kh.data(), kvlen*hd*sizeof(float))  ||
                    !d_vh.copyFromHost(Vh.data(), kvlen*hd*sizeof(float)))  { cok=false; break; }

                if (rocblas_sgemm(impl_->handle,
                        rocblas_operation_transpose, rocblas_operation_none,
                        kvlen, qlen, hd, &cscale,
                        reinterpret_cast<const float*>(d_kh.ptr), hd,
                        reinterpret_cast<const float*>(d_qh.ptr), hd,
                        &cb0,
                        reinterpret_cast<float*>(d_sc.ptr), kvlen) != rocblas_status_success) { cok=false; break; }

                if (hipDeviceSynchronize() != hipSuccess) { cok=false; break; }
                std::vector<float> sc_h(qlen * kvlen);
                if (!d_sc.copyToHost(sc_h.data(), qlen*kvlen*sizeof(float))) { cok=false; break; }
                softmax_rows(sc_h.data(), qlen, kvlen);
                if (!d_sc.copyFromHost(sc_h.data(), qlen*kvlen*sizeof(float))) { cok=false; break; }

                if (rocblas_sgemm(impl_->handle,
                        rocblas_operation_none, rocblas_operation_none,
                        hd, qlen, kvlen, &ca,
                        reinterpret_cast<const float*>(d_vh.ptr), hd,
                        reinterpret_cast<const float*>(d_sc.ptr), kvlen,
                        &cb0,
                        reinterpret_cast<float*>(d_ctx.ptr), hd) != rocblas_status_success) { cok=false; break; }

                if (hipDeviceSynchronize() != hipSuccess) { cok=false; break; }
                std::vector<float> ctx_h(qlen * hd);
                if (!d_ctx.copyToHost(ctx_h.data(), qlen*hd*sizeof(float))) { cok=false; break; }
                for (int t = 0; t < qlen; ++t)
                    std::copy(&ctx_h[t*hd], &ctx_h[(t+1)*hd], &attended_c[t*ed+hoff]);
            }
            if (!cok) break;

            Impl::DeviceBuf d_att_c, d_wo_c, d_final_c;
            if (!d_att_c.alloc(qlen*ed*sizeof(float))) break;
            if (!d_wo_c.alloc(o_sz*sizeof(float))) break;
            if (!d_final_c.alloc(qlen*ed*sizeof(float))) break;
            if (!d_att_c.copyFromHost(attended_c.data(), qlen*ed*sizeof(float))) break;
            if (!d_wo_c.copyFromHost(wo, o_sz*sizeof(float))) break;
            if (rocblas_sgemm(impl_->handle,
                    rocblas_operation_none, rocblas_operation_none,
                    ed, qlen, ed, &ca,
                    reinterpret_cast<const float*>(d_wo_c.ptr), ed,
                    reinterpret_cast<const float*>(d_att_c.ptr), ed,
                    &cb0,
                    reinterpret_cast<float*>(d_final_c.ptr), ed) != rocblas_status_success) break;

            if (hipDeviceSynchronize() != hipSuccess) break;
            outputs.resize(1);
            outputs[0].resize(static_cast<size_t>(qlen) * ed);
            if (!d_final_c.copyToHost(outputs[0].data(), static_cast<size_t>(qlen)*ed*sizeof(float))) break;
            return true;
            } while (false);
            break;
        default:
            break;
    }
#endif // ENABLE_ROCM

    return RuntimeLayerDispatch::cpu_forward_layer(inputs, outputs, layer, training);
}

bool RocmRuntime::backwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs,
    std::vector<std::vector<float>>& grad_inputs,
    Layer& layer,
    bool training
) {
    if (!isInitialized()) return false;
    if (config_.disabled) return false;
    if (!supportsBackwardLayerType(layer.type_enum)) return false;

    switch (layer.type_enum) {
        case LayerType::Linear:
            if (!config_.linear_enabled) return false;
            break;
        case LayerType::MatMul:
        case LayerType::BatchMatMul:
        case LayerType::Add:
        case LayerType::Subtract:
        case LayerType::Multiply:
        case LayerType::Divide:
        case LayerType::ReLU:
        case LayerType::LeakyReLU:
        case LayerType::Sigmoid:
        case LayerType::Tanh:
        case LayerType::SiLU:
        case LayerType::GELU:
        case LayerType::Softplus:
        case LayerType::Mish:
        case LayerType::HardSigmoid:
        case LayerType::HardSwish:
            if (!config_.linear_enabled) return false;
            break;
        case LayerType::Conv2d:
            if (!config_.conv_enabled) return false;
            break;
        case LayerType::LayerNorm:
        case LayerType::RMSNorm:
            if (!config_.norm_enabled) return false;
            break;
        case LayerType::SelfAttention:
        case LayerType::MultiHeadAttention:
        case LayerType::CrossAttention:
            if (!config_.attention_enabled) return false;
            break;
        case LayerType::MaxPool2d:
        case LayerType::MaxPool1d:
        case LayerType::AvgPool2d:
        case LayerType::AvgPool1d:
        case LayerType::GlobalAvgPool2d:
        case LayerType::AdaptiveAvgPool2d:
            break;
        default:
            return false;
    }

    return RuntimeLayerDispatch::cpu_backward_layer(inputs, grad_outputs, grad_inputs, layer, training);
}

bool RocmRuntime::supportsOptimizer() const {
#if defined(ENABLE_ROCM) && __has_include(<hip/hiprtc.h>)
    return isInitialized() && !config_.disabled;
#else
    return false;
#endif
}
bool RocmRuntime::optimizerStep(const RuntimeOptimizerRequest& request) {
#if defined(ENABLE_ROCM) && __has_include(<hip/hiprtc.h>)
    return supportsOptimizer() && MimirRocmOptimizer::run(request,config_.device_index);
#else
    (void)request;return false;
#endif
}

#include "runtimes/losses/LossDispatch.hpp"
#include "runtimes/losses/LossShaderData.hpp"
bool RocmRuntime::reconstructionLoss(const RuntimeLossGrad::Request& r,RuntimeLossGrad::LossWithGrad& out) {
#if defined(ENABLE_ROCM) && __has_include(<hip/hiprtc.h>)
    if(!supportsReconstructionLoss() || !RuntimeLossGrad::deviceSizeFits(r) || hipSetDevice(config_.device_index)!=hipSuccess)return false;
    static thread_local std::map<int,std::unique_ptr<MimirRocmOptimizer::Compiler>> compilers;
    auto& compiler=compilers[config_.device_index];
    if(!compiler){auto candidate=std::make_unique<MimirRocmOptimizer::Compiler>();if(!candidate->initialize(MimirLossKernels::hip,"mimir_reconstruction"))return false;compiler=std::move(candidate);}
    auto memory=RuntimeLossGrad::pack(r);auto args=RuntimeLossGrad::parameters(r);
    MimirRocmOptimizer::Buffer data,params;
    if(!data.alloc(memory.size()*4) || !params.alloc(args.size()*4) ||
       hipMemcpy(data.ptr,memory.data(),memory.size()*4,hipMemcpyHostToDevice)!=hipSuccess ||
       hipMemcpy(params.ptr,args.data(),args.size()*4,hipMemcpyHostToDevice)!=hipSuccess)return false;
    for(int pass:RuntimeLossGrad::passes(args)) {
        if(hipMemcpy(params.ptr,&pass,4,hipMemcpyHostToDevice)!=hipSuccess)return false;
        void* parameters[]={&data.ptr,&params.ptr};
        if(hipModuleLaunchKernel(compiler->kernel,pass==91?1:(args[1]+255)/256,1,1,256,1,1,0,nullptr,parameters,nullptr)!=hipSuccess || hipDeviceSynchronize()!=hipSuccess)return false;
    }
    if(hipMemcpy(memory.data(),data.ptr,memory.size()*4,hipMemcpyDeviceToHost)!=hipSuccess)return false;
    return RuntimeLossGrad::unpack(r,memory,out);
#else
    (void)r;(void)out;return false;
#endif
}
