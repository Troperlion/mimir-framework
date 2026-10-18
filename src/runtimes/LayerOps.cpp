#include "runtimes/LayerOps.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "SIMD_Ops.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace {

std::atomic<bool> g_hardware_acceleration{true};

bool runtimeHasAVX2() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    static const bool available = [] {
        unsigned int eax = 0;
        unsigned int ebx = 0;
        unsigned int ecx = 0;
        unsigned int edx = 0;
#if defined(_MSC_VER)
        int regs[4] = {0, 0, 0, 0};
        __cpuidex(regs, 1, 0);
        ecx = static_cast<unsigned int>(regs[2]);
#else
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) return false;
#endif
        if ((ecx & (1u << 27)) == 0 || (ecx & (1u << 28)) == 0) return false;
#if defined(_MSC_VER)
        if ((_xgetbv(0) & 0x6ull) != 0x6ull) return false;
        __cpuidex(regs, 7, 0);
        ebx = static_cast<unsigned int>(regs[1]);
#else
        uint32_t xcr0_lo = 0;
        uint32_t xcr0_hi = 0;
        __asm__ volatile ("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
        (void)xcr0_hi;
        if ((xcr0_lo & 0x6u) != 0x6u) return false;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) return false;
#endif
        return (ebx & (1u << 5)) != 0;
    }();
    return available;
#else
    return false;
#endif
}

bool useHardware(const bool requested) {
    return requested && g_hardware_acceleration.load(std::memory_order_relaxed) && runtimeHasAVX2();
}

void softmaxInPlace(std::vector<float>& values) {
    if (values.empty()) return;
    const float maximum = *std::max_element(values.begin(), values.end());
    float sum = 0.0f;
    for (float& value : values) {
        value = std::exp(value - maximum);
        sum += value;
    }
    for (float& value : values) value /= sum;
}

} // namespace

namespace RuntimeLayerOps {

void setHardwareAcceleration(const bool enable) {
    g_hardware_acceleration.store(enable, std::memory_order_relaxed);
}

bool hardwareAccelerationEnabled() {
    return g_hardware_acceleration.load(std::memory_order_relaxed);
}

void computeConv2D(const std::vector<float>& input, std::vector<float>& output,
                   const LayerParams& params, const int in_h, const int in_w,
                   const int in_c, const int out_c, bool) {
    Conv::conv2d(input, output, params.weights, params.bias, in_h, in_w, in_c, out_c,
                 params.kernel_size, params.stride, params.padding, params.dilation);
}

void computeLinear(const std::vector<float>& input, std::vector<float>& output,
                   const LayerParams& params, const bool use_hardware) {
    if (params.in_features < 0 || params.out_features < 0 ||
        input.size() < static_cast<size_t>(params.in_features) ||
        params.weights.size() < static_cast<size_t>(params.in_features) * params.out_features) {
        throw std::invalid_argument("RuntimeLayerOps::computeLinear: dimensions invalides");
    }
    output.assign(static_cast<size_t>(params.out_features), 0.0f);
#if defined(__AVX2__) || defined(_M_AVX2)
    if (useHardware(use_hardware)) {
        SIMD::matmul_avx2(output.data(), input.data(), params.weights.data(),
                         1, params.out_features, params.in_features);
        if (!params.bias.empty()) {
            SIMD::add_vectors_avx2(output.data(), output.data(), params.bias.data(), params.out_features);
        }
        return;
    }
#else
    (void)use_hardware;
#endif
    for (int output_index = 0; output_index < params.out_features; ++output_index) {
        float sum = 0.0f;
        for (int input_index = 0; input_index < params.in_features; ++input_index) {
            sum += input[input_index] * params.weights[output_index * params.in_features + input_index];
        }
        if (!params.bias.empty()) sum += params.bias[output_index];
        output[output_index] = sum;
    }
}

void computeMaxPool2D(const std::vector<float>& input, std::vector<float>& output,
                      const int in_h, const int in_w, const int channels,
                      const int kernel_size, int stride, bool) {
    if (stride < 0) stride = kernel_size;
    Pooling::maxpool2d(input, output, in_h, in_w, channels, kernel_size, stride);
}

void computeAvgPool2D(const std::vector<float>& input, std::vector<float>& output,
                      const int in_h, const int in_w, const int channels,
                      const int kernel_size, int stride, bool) {
    if (stride < 0) stride = kernel_size;
    Pooling::avgpool2d(input, output, in_h, in_w, channels, kernel_size, stride);
}

void computeActivation(std::vector<float>& data, const std::string& activation_type,
                       const float param, const bool use_hardware) {
#if defined(__AVX2__) || defined(_M_AVX2)
    const bool hardware = useHardware(use_hardware);
    if (activation_type == "gelu" && hardware) {
        SIMD::gelu_forward_avx2(data.data(), data.data(), data.size());
        return;
    }
    if (activation_type == "relu" && hardware) {
        const size_t vectorized_size = data.size() & ~static_cast<size_t>(7);
        const __m256 zero = _mm256_setzero_ps();
        size_t index = 0;
        for (; index < vectorized_size; index += 8) {
            const __m256 values = _mm256_loadu_ps(&data[index]);
            _mm256_storeu_ps(&data[index], _mm256_max_ps(values, zero));
        }
        for (; index < data.size(); ++index) data[index] = std::max(0.0f, data[index]);
        return;
    }
    if (activation_type == "softmax" && hardware) {
        if (!data.empty()) SIMD::softmax_avx2(data.data(), data.data(), data.size());
        return;
    }
#else
    (void)use_hardware;
#endif
    if (activation_type == "gelu") {
        constexpr float scale = 0.7978845608f;
        for (float& value : data) {
            value = 0.5f * value * (1.0f + std::tanh(scale * (value + 0.044715f * value * value * value)));
        }
    } else if (activation_type == "relu") {
        for (float& value : data) value = std::max(0.0f, value);
    } else if (activation_type == "leaky_relu") {
        for (float& value : data) if (value < 0.0f) value *= param;
    } else if (activation_type == "tanh") {
        for (float& value : data) value = std::tanh(value);
    } else if (activation_type == "sigmoid") {
        for (float& value : data) value = 1.0f / (1.0f + std::exp(-value));
    } else if (activation_type == "softmax") {
        softmaxInPlace(data);
    } else if (activation_type == "elu") {
        for (float& value : data) value = value >= 0.0f ? value : param * (std::exp(value) - 1.0f);
    }
}

void computeBatchNorm(std::vector<float>& data, const std::vector<float>& gamma,
                      const std::vector<float>& beta, const std::vector<float>& running_mean,
                      const std::vector<float>& running_var, const int batch_size,
                      const int channels, const int spatial_size, const float eps,
                      const bool training, bool) {
    Normalization::batch_norm(data, gamma, beta, running_mean, running_var,
                              batch_size, channels, spatial_size, eps, training);
}

void computeLayerNorm(std::vector<float>& data, const std::vector<float>& gamma,
                      const std::vector<float>& beta, const int normalized_size,
                      const float eps, bool) {
    Normalization::layer_norm(data, gamma, beta, normalized_size, eps);
}

bool prepareLayerNorm(const std::vector<float>& input, const int normalized_size,
                      const float eps, std::vector<float>& normalized) {
    if (normalized_size <= 0 || input.empty() ||
        input.size() % static_cast<size_t>(normalized_size) != 0) {
        return false;
    }

    normalized.resize(input.size());
    const int groups = static_cast<int>(input.size() / static_cast<size_t>(normalized_size));
    for (int group = 0; group < groups; ++group) {
        const size_t base = static_cast<size_t>(group) * normalized_size;
        float mean = 0.0f;
        for (int index = 0; index < normalized_size; ++index) {
            mean += input[base + static_cast<size_t>(index)];
        }
        mean /= static_cast<float>(normalized_size);

        float variance = 0.0f;
        for (int index = 0; index < normalized_size; ++index) {
            const float centered = input[base + static_cast<size_t>(index)] - mean;
            variance += centered * centered;
        }
        const float inverse_stddev =
            1.0f / std::sqrt(variance / static_cast<float>(normalized_size) + eps);
        for (int index = 0; index < normalized_size; ++index) {
            normalized[base + static_cast<size_t>(index)] =
                (input[base + static_cast<size_t>(index)] - mean) * inverse_stddev;
        }
    }
    return true;
}

void computeConvTranspose2D(const std::vector<float>& input, std::vector<float>& output,
                            const LayerParams& params, const int in_h, const int in_w,
                            const int in_c, const int out_c, bool) {
    Conv::conv_transpose2d(input, output, params.weights, params.bias, in_h, in_w, in_c, out_c,
                           params.kernel_size, params.stride, params.padding);
}

void computeAttention(const std::vector<float>& query, const std::vector<float>& key,
                      const std::vector<float>& value, std::vector<float>& output,
                      const int seq_len, const int d_model, const int num_heads,
                      const bool use_hardware) {
    if (seq_len < 0 || d_model < 0 || num_heads <= 0 || d_model % num_heads != 0) {
        throw std::invalid_argument("RuntimeLayerOps::computeAttention: dimensions invalides");
    }
    const size_t tensor_size = static_cast<size_t>(seq_len) * d_model;
    if (query.size() < tensor_size || key.size() < tensor_size || value.size() < tensor_size) {
        throw std::invalid_argument("RuntimeLayerOps::computeAttention: tenseur trop petit");
    }
    const int head_dim = d_model / num_heads;
    output.assign(tensor_size, 0.0f);
    std::vector<float> attention_scores(static_cast<size_t>(seq_len) * seq_len);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int head = 0; head < num_heads; ++head) {
#if defined(__AVX2__) || defined(_M_AVX2)
        if (useHardware(use_hardware)) {
            SIMD::matmul_transpose_avx2(attention_scores.data(), &query[head * head_dim],
                                        &key[head * head_dim], seq_len, seq_len, head_dim);
        } else
#else
        (void)use_hardware;
#endif
        {
            for (int query_index = 0; query_index < seq_len; ++query_index) {
                for (int key_index = 0; key_index < seq_len; ++key_index) {
                    float sum = 0.0f;
                    for (int component = 0; component < head_dim; ++component) {
                        sum += query[query_index * d_model + head * head_dim + component] *
                               key[key_index * d_model + head * head_dim + component];
                    }
                    attention_scores[query_index * seq_len + key_index] = sum * scale;
                }
            }
        }
        for (int query_index = 0; query_index < seq_len; ++query_index) {
            std::vector<float> row(attention_scores.begin() + query_index * seq_len,
                                   attention_scores.begin() + (query_index + 1) * seq_len);
            softmaxInPlace(row);
            std::copy(row.begin(), row.end(), attention_scores.begin() + query_index * seq_len);
        }
        for (int query_index = 0; query_index < seq_len; ++query_index) {
            for (int component = 0; component < head_dim; ++component) {
                float sum = 0.0f;
                for (int key_index = 0; key_index < seq_len; ++key_index) {
                    sum += attention_scores[query_index * seq_len + key_index] *
                           value[key_index * d_model + head * head_dim + component];
                }
                output[query_index * d_model + head * head_dim + component] = sum;
            }
        }
    }
}

void conv2dSame(const std::vector<float>& input, std::vector<float>& output,
                const int width, const int height, const std::vector<float>& kernel,
                const int kernel_size) {
    if (width < 0 || height < 0 || kernel_size <= 0 ||
        input.size() < static_cast<size_t>(width) * height ||
        kernel.size() < static_cast<size_t>(kernel_size) * kernel_size) {
        throw std::invalid_argument("RuntimeLayerOps::conv2dSame: dimensions invalides");
    }
    LayerParams params;
    params.weights = kernel;
    params.kernel_size = kernel_size;
    params.padding = kernel_size / 2;
    computeConv2D(input, output, params, height, width, 1, 1, true);
}

void branchMerge(const std::vector<float>& branch1, const std::vector<float>& branch2,
                 std::vector<float>& output, const MergeOperation merge_op,
                 const bool use_hardware) {
    if (merge_op == MergeOperation::CONCATENATE) {
        output = branch1;
        output.insert(output.end(), branch2.begin(), branch2.end());
        return;
    }
    if (branch1.size() != branch2.size()) {
        throw std::invalid_argument("RuntimeLayerOps::branchMerge: tailles incompatibles");
    }
    output.resize(branch1.size());
    size_t index = 0;
#if defined(__AVX2__) || defined(_M_AVX2)
    if (useHardware(use_hardware)) {
        for (; index + 8 <= branch1.size(); index += 8) {
            const __m256 left = _mm256_loadu_ps(&branch1[index]);
            const __m256 right = _mm256_loadu_ps(&branch2[index]);
            __m256 result;
            switch (merge_op) {
                case MergeOperation::MULTIPLY: result = _mm256_mul_ps(left, right); break;
                case MergeOperation::MAX: result = _mm256_max_ps(left, right); break;
                case MergeOperation::AVERAGE:
                    result = _mm256_mul_ps(_mm256_add_ps(left, right), _mm256_set1_ps(0.5f));
                    break;
                default: result = _mm256_add_ps(left, right); break;
            }
            _mm256_storeu_ps(&output[index], result);
        }
    }
#else
    (void)use_hardware;
#endif
    for (; index < branch1.size(); ++index) {
        switch (merge_op) {
            case MergeOperation::MULTIPLY: output[index] = branch1[index] * branch2[index]; break;
            case MergeOperation::MAX: output[index] = std::max(branch1[index], branch2[index]); break;
            case MergeOperation::AVERAGE: output[index] = (branch1[index] + branch2[index]) * 0.5f; break;
            default: output[index] = branch1[index] + branch2[index]; break;
        }
    }
}

void branchSplit(const std::vector<float>& input, std::vector<std::vector<float>>& outputs,
                 const std::vector<int>& split_sizes) {
    size_t total_size = 0;
    for (const int split_size : split_sizes) {
        if (split_size < 0 || static_cast<size_t>(split_size) > input.size() - total_size) {
            throw std::invalid_argument("RuntimeLayerOps::branchSplit: taille de split invalide");
        }
        total_size += static_cast<size_t>(split_size);
    }
    if (total_size != input.size()) {
        throw std::invalid_argument("RuntimeLayerOps::branchSplit: les splits ne couvrent pas l'entree");
    }
    outputs.clear();
    outputs.reserve(split_sizes.size());
    auto begin = input.begin();
    for (const int split_size : split_sizes) {
        const auto end = begin + split_size;
        outputs.emplace_back(begin, end);
        begin = end;
    }
}

bool resolveUnaryOp(const LayerType type, const Layer& layer, int& op_code, float& alpha) {
    alpha = 0.01f;
    switch (type) {
        case LayerType::ReLU: op_code = 0; return true;
        case LayerType::LeakyReLU:
            op_code = 1;
            alpha = layer.leaky_relu_alpha > 0.0f ? layer.leaky_relu_alpha : 0.01f;
            return true;
        case LayerType::Sigmoid: op_code = 2; return true;
        case LayerType::Tanh: op_code = 3; return true;
        case LayerType::SiLU: op_code = 4; return true;
        case LayerType::GELU: op_code = 5; return true;
        case LayerType::Softplus: op_code = 6; return true;
        case LayerType::Mish: op_code = 7; return true;
        case LayerType::HardSigmoid: op_code = 8; return true;
        case LayerType::HardSwish: op_code = 9; return true;
        default:
            return false;
    }
}

bool resolveBinaryOp(const LayerType type, int& op_code) {
    switch (type) {
        case LayerType::Add: op_code = 0; return true;
        case LayerType::Subtract: op_code = 1; return true;
        case LayerType::Multiply: op_code = 2; return true;
        case LayerType::Divide: op_code = 3; return true;
        default:
            return false;
    }
}

void unaryForwardHost(const std::vector<float>& input, std::vector<float>& output, const int op_code, const float alpha) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const float x = input[i];
        switch (op_code) {
            case 0: output[i] = x > 0.0f ? x : 0.0f; break;
            case 1: output[i] = x > 0.0f ? x : alpha * x; break;
            case 2: output[i] = 1.0f / (1.0f + std::exp(-x)); break;
            case 3: output[i] = std::tanh(x); break;
            case 4: {
                const float s = 1.0f / (1.0f + std::exp(-x));
                output[i] = x * s;
                break;
            }
            case 5: {
                const float c = 0.7978845608f;
                const float x3 = x * x * x;
                output[i] = 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x3)));
                break;
            }
            case 6: output[i] = x > 20.0f ? x : std::log1p(std::exp(x)); break;
            case 7: {
                const float sp = x > 20.0f ? x : std::log1p(std::exp(x));
                output[i] = x * std::tanh(sp);
                break;
            }
            case 8: {
                const float hs = (x + 3.0f) / 6.0f;
                output[i] = std::min(1.0f, std::max(0.0f, hs));
                break;
            }
            case 9: {
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

void binaryForwardHost(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& output, const int op_code) {
    output.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        switch (op_code) {
            case 0: output[i] = a[i] + b[i]; break;
            case 1: output[i] = a[i] - b[i]; break;
            case 2: output[i] = a[i] * b[i]; break;
            case 3: {
                const float d = b[i];
                const float safe_d = std::fabs(d) < 1e-8f
                    ? (d < 0.0f ? -1e-8f : 1e-8f)
                    : d;
                output[i] = a[i] / safe_d;
                break;
            }
            default:
                output[i] = a[i];
                break;
        }
    }
}

} // namespace RuntimeLayerOps
