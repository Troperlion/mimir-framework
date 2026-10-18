#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <omp.h>
#include "Layers.hpp"

// ============================================================================
// LAYER OPERATIONS - Extensions (layers manquants prioritaires)
// ============================================================================
// 
// Ce fichier implémente les layers déclarés mais non implémentés dans Model.cpp
// Pour intégration dans le switch-case du runtime
// 
// Priorité 1: Conv1d, DepthwiseConv2d, LeakyReLU, InstanceNorm2d
// Priorité 2: Subtract, Divide, Squeeze, Unsqueeze
// Priorité 3: RMSNorm, Mish, Softplus
// ============================================================================

namespace LayerOpsExt {

// ============================================================================
// ACTIVATIONS MANQUANTES
// ============================================================================

// LeakyReLU: f(x) = x if x > 0 else alpha * x
inline std::vector<float> leaky_relu_forward(
    const std::vector<float>& input,
    float alpha = 0.01f
) {
    std::vector<float> output(input.size());
    
    #pragma omp simd
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] > 0.0f ? input[i] : alpha * input[i];
    }
    
    return output;
}

// Mish: f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
inline std::vector<float> mish_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    #pragma omp simd
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        // softplus(x) = ln(1 + e^x) ≈ x pour x grand, évite overflow
        float sp = x > 20.0f ? x : std::log1p(std::exp(x));
        output[i] = x * std::tanh(sp);
    }
    
    return output;
}

// Softplus: f(x) = ln(1 + e^x)
inline std::vector<float> softplus_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    #pragma omp simd
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        // Pour x > 20, softplus(x) ≈ x (évite overflow)
        output[i] = x > 20.0f ? x : std::log1p(std::exp(x));
    }
    
    return output;
}

// HardSigmoid: f(x) = clip((x + 3) / 6, 0, 1)
inline std::vector<float> hard_sigmoid_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    #pragma omp simd
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float val = (x + 3.0f) / 6.0f;
        output[i] = std::max(0.0f, std::min(1.0f, val));
    }
    
    return output;
}

// HardSwish: f(x) = x * HardSigmoid(x)
inline std::vector<float> hard_swish_forward(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    #pragma omp simd
    for (size_t i = 0; i < input.size(); ++i) {
        float x = input[i];
        float hs = std::max(0.0f, std::min(1.0f, (x + 3.0f) / 6.0f));
        output[i] = x * hs;
    }
    
    return output;
}

// ============================================================================
// ELEMENT-WISE OPERATIONS MANQUANTES
// ============================================================================

// Subtract: a - b
inline std::vector<float> subtract_forward(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    if (a.size() != b.size()) {
        throw std::runtime_error(
            "Subtract: size mismatch: " + std::to_string(a.size()) +
            " vs " + std::to_string(b.size())
        );
    }
    
    std::vector<float> output(a.size());
    
    #pragma omp simd
    for (size_t i = 0; i < a.size(); ++i) {
        output[i] = a[i] - b[i];
    }
    
    return output;
}

// Divide: a / b (avec protection division par zéro)
inline std::vector<float> divide_forward(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float eps = 1e-8f
) {
    if (a.size() != b.size()) {
        throw std::runtime_error(
            "Divide: size mismatch: " + std::to_string(a.size()) +
            " vs " + std::to_string(b.size())
        );
    }
    
    std::vector<float> output(a.size());
    
    for (size_t i = 0; i < a.size(); ++i) {
        // Protection division par zéro
        float divisor = b[i];
        if (std::abs(divisor) < eps) {
            divisor = divisor >= 0 ? eps : -eps;
        }
        output[i] = a[i] / divisor;
    }
    
    return output;
}

// ============================================================================
// SHAPE OPERATIONS MANQUANTES
// ============================================================================

// Squeeze: Remove dimensions of size 1
// Input shape: [N, 1, H, W] → [N, H, W]
// Note: Data unchanged, shape tracking external
inline std::vector<float> squeeze_forward(
    const std::vector<float>& input,
    const std::vector<int>& input_shape,
    std::vector<int>& output_shape,
    int dim = -1  // -1 = squeeze all dims of size 1
) {
    output_shape.clear();
    
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (dim == -1) {
            // Squeeze all dims == 1
            if (input_shape[i] != 1) {
                output_shape.push_back(input_shape[i]);
            }
        } else {
            // Squeeze specific dim
            if (static_cast<int>(i) != dim || input_shape[i] != 1) {
                output_shape.push_back(input_shape[i]);
            }
        }
    }
    
    // Data unchanged
    return input;
}

// Unsqueeze: Add dimension of size 1
// Input shape: [N, H, W] → [N, 1, H, W] (dim=1)
inline std::vector<float> unsqueeze_forward(
    const std::vector<float>& input,
    const std::vector<int>& input_shape,
    std::vector<int>& output_shape,
    int dim
) {
    if (dim < 0) dim = input_shape.size() + dim + 1;
    if (dim < 0 || dim > static_cast<int>(input_shape.size())) {
        throw std::runtime_error(
            "Unsqueeze: invalid dim " + std::to_string(dim) +
            " for shape with " + std::to_string(input_shape.size()) + " dims"
        );
    }
    
    output_shape = input_shape;
    output_shape.insert(output_shape.begin() + dim, 1);
    
    // Data unchanged
    return input;
}

// ============================================================================
// TENSOR OPERATIONS MANQUANTES
// ============================================================================

// Chunk: Split tensor into N equal chunks along an axis
// Retourne vector de vectors
inline std::vector<std::vector<float>> chunk_forward(
    const std::vector<float>& input,
    int chunks,
    int axis = 0
) {
    (void)axis;
    if (chunks <= 0) {
        throw std::runtime_error("Chunk: chunks must be > 0");
    }
    
    // Simplification: assume 1D ou axis=0
    size_t chunk_size = input.size() / chunks;
    size_t remainder = input.size() % chunks;
    
    std::vector<std::vector<float>> outputs;
    outputs.reserve(chunks);
    
    size_t offset = 0;
    for (int c = 0; c < chunks; ++c) {
        size_t size = chunk_size + (c < static_cast<int>(remainder) ? 1 : 0);
        std::vector<float> chunk(input.begin() + offset, input.begin() + offset + size);
        outputs.push_back(std::move(chunk));
        offset += size;
    }
    
    return outputs;
}

// Stack: Stack N tensors along a new axis
// Inputs: [[a1, a2, ...], [b1, b2, ...], ...]
// Output: Stacked along axis (default 0)
inline std::vector<float> stack_forward(
    const std::vector<std::vector<float>>& inputs,
    int axis = 0
) {
    (void)axis;
    if (inputs.empty()) {
        throw std::runtime_error("Stack: no inputs provided");
    }
    
    // Vérifier que toutes les entrées ont la même taille
    size_t size = inputs[0].size();
    for (const auto& inp : inputs) {
        if (inp.size() != size) {
            throw std::runtime_error("Stack: all inputs must have same size");
        }
    }
    
    // Simplification: concat séquentiel (axis=0)
    std::vector<float> output;
    output.reserve(inputs.size() * size);
    
    for (const auto& inp : inputs) {
        output.insert(output.end(), inp.begin(), inp.end());
    }
    
    return output;
}

// ============================================================================
// NORMALIZATION MANQUANTES
// ============================================================================

// InstanceNorm2d: Normalize per instance per channel
// Input shape: [C, H, W]
inline std::vector<float> instance_norm2d_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    const int channels = layer.in_channels > 0 ? layer.in_channels : 3;
    const int height = layer.input_height > 0 ? layer.input_height : 32;
    const int width = layer.input_width > 0 ? layer.input_width : 32;
    const float eps = layer.eps;
    
    if (input.size() != static_cast<size_t>(channels * height * width)) {
        throw std::runtime_error(
            "InstanceNorm2d: input size mismatch. Expected " +
            std::to_string(channels * height * width) + ", got " +
            std::to_string(input.size())
        );
    }
    
    std::vector<float> output = input;
    const int spatial_size = height * width;
    
    // Normalize each channel independently
    for (int c = 0; c < channels; ++c) {
        const float* channel_data = &input[c * spatial_size];
        float* out_channel = &output[c * spatial_size];
        
        // Compute mean
        float mean = 0.0f;
        #pragma omp simd reduction(+:mean)
        for (int i = 0; i < spatial_size; ++i) {
            mean += channel_data[i];
        }
        mean /= spatial_size;
        
        // Compute variance
        float var = 0.0f;
        #pragma omp simd reduction(+:var)
        for (int i = 0; i < spatial_size; ++i) {
            float diff = channel_data[i] - mean;
            var += diff * diff;
        }
        var /= spatial_size;
        float std = std::sqrt(var + eps);
        
        // Normalize
        #pragma omp simd
        for (int i = 0; i < spatial_size; ++i) {
            out_channel[i] = (channel_data[i] - mean) / std;
        }
        
        // Apply affine transform if enabled
        if (layer.affine && layer.getWeights()) {
            const float* weights = layer.getWeights();
            float gamma = weights[c];
            float beta = weights[channels + c];
            
            #pragma omp simd
            for (int i = 0; i < spatial_size; ++i) {
                out_channel[i] = gamma * out_channel[i] + beta;
            }
        }
    }
    
    return output;
}

// RMSNorm: Root Mean Square Normalization
// f(x) = x / sqrt(mean(x^2) + eps) * gamma
inline std::vector<float> rms_norm_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    const float eps = layer.eps;
    int normalized_size = 0;
    if (layer.in_features > 0) {
        normalized_size = layer.in_features;
    } else if (layer.embed_dim > 0) {
        normalized_size = layer.embed_dim;
    } else if (!layer.shape.empty() && layer.shape.back() > 0) {
        normalized_size = layer.shape.back();
    } else if (!layer.target_shape.empty() && layer.target_shape.back() > 0) {
        normalized_size = layer.target_shape.back();
    } else {
        normalized_size = static_cast<int>(input.size());
    }

    if (normalized_size <= 0 || (input.size() % static_cast<size_t>(normalized_size)) != 0ULL) {
        throw std::runtime_error("RMSNorm: invalid normalized_size");
    }

    std::vector<float> output = input;
    std::vector<float> gamma(static_cast<size_t>(normalized_size), 1.0f);
    if (layer.affine) {
        const float* weights = layer.getWeights();
        const size_t weights_size = weights ? layer.getWeightsSize() : 0ULL;
        const size_t copy_n = std::min(static_cast<size_t>(normalized_size), weights_size);
        if (copy_n > 0) {
            std::copy(weights, weights + copy_n, gamma.begin());
        }
    }

    Normalization::rms_norm(output, gamma, normalized_size, eps);
    return output;
}

// ============================================================================
// CONVOLUTION MANQUANTES
// ============================================================================

// Conv1d: 1D Convolution
// Input: [in_channels, length]
// Kernel: [out_channels, in_channels, kernel_size]
// Output: [out_channels, out_length]
inline std::vector<float> conv1d_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    const int in_channels = layer.in_channels > 0 ? layer.in_channels : 1;
    const int out_channels = layer.out_channels > 0 ? layer.out_channels : 1;
    const int kernel_size = layer.kernel_h > 0 ? layer.kernel_h : 3;
    const int length = static_cast<int>(input.size()) / in_channels;
    const int stride = layer.stride_h > 0 ? layer.stride_h : 1;
    const int padding = layer.pad_h >= 0 ? layer.pad_h : 0;
    
    const int out_length = (length + 2 * padding - kernel_size) / stride + 1;
    const size_t output_size = out_channels * out_length;
    
    std::vector<float> output(output_size, 0.0f);
    const float* weights = layer.getWeights();
    const int weights_size = static_cast<int>(layer.getWeightsSize());
    const float* bias = layer.use_bias ? 
                        (weights + out_channels * in_channels * kernel_size) : nullptr;
    (void)weights_size;
    
    const long long conv1d_work = static_cast<long long>(out_channels) * out_length * in_channels * kernel_size;
    #pragma omp parallel for if(conv1d_work >= 262144) schedule(static)
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ol = 0; ol < out_length; ++ol) {
            float sum = 0.0f;
            
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int k = 0; k < kernel_size; ++k) {
                    int il = ol * stride + k - padding;
                    
                    if (il >= 0 && il < length) {
                        int in_idx = ic * length + il;
                        int w_idx = ((oc * in_channels + ic) * kernel_size) + k;
                        
                        if (in_idx < static_cast<int>(input.size()) &&
                            w_idx < static_cast<int>(layer.getWeightsSize())) {
                            sum += input[in_idx] * weights[w_idx];
                        }
                    }
                }
            }
            
            // Add bias
            if (bias) sum += bias[oc];
            
            output[oc * out_length + ol] = sum;
        }
    }
    
    return output;
}

// DepthwiseConv2d: Each input channel convolved separately
// Kernel: [in_channels, 1, kernel_h, kernel_w]
inline std::vector<float> depthwise_conv2d_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    const int channels = layer.in_channels > 0 ? layer.in_channels : 3;
    const int height = layer.input_height > 0 ? layer.input_height : 32;
    const int width = layer.input_width > 0 ? layer.input_width : 32;
    const int kernel_size = layer.kernel_h > 0 ? layer.kernel_h : 3;
    const int stride = layer.stride_h > 0 ? layer.stride_h : 1;
    const int padding = layer.pad_h >= 0 ? layer.pad_h : 0;
    
    const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    const int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    const size_t output_size = channels * out_height * out_width;
    
    std::vector<float> output(output_size, 0.0f);
    const float* weights = layer.getWeights();
    const int weights_size = static_cast<int>(layer.getWeightsSize());
    const float* bias = layer.use_bias ? (weights + channels * kernel_size * kernel_size) : nullptr;
    (void)weights_size;
    
    // Each channel processed independently
    const long long dwconv_work = static_cast<long long>(channels) * out_height * out_width * kernel_size * kernel_size;
    #pragma omp parallel for if(dwconv_work >= 262144) schedule(static)
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                float sum = 0.0f;
                
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int ih = oh * stride + kh - padding;
                        int iw = ow * stride + kw - padding;
                        
                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                            int in_idx = c * (height * width) + ih * width + iw;
                            int w_idx = c * (kernel_size * kernel_size) + kh * kernel_size + kw;
                            
                            if (in_idx < static_cast<int>(input.size()) &&
                                w_idx < static_cast<int>(layer.getWeightsSize())) {
                                sum += input[in_idx] * weights[w_idx];
                            }
                        }
                    }
                }
                
                // Add bias
                if (bias) sum += bias[c];
                
                int out_idx = c * (out_height * out_width) + oh * out_width + ow;
                output[out_idx] = sum;
            }
        }
    }
    
    return output;
}

// ============================================================================
// POOLING MANQUANTES
// ============================================================================

// MaxPool1d: Max pooling 1D
inline std::vector<float> maxpool1d_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    const int channels = layer.in_channels > 0 ? layer.in_channels : 1;
    const int length = static_cast<int>(input.size()) / channels;
    const int kernel_size = layer.kernel_h > 0 ? layer.kernel_h : 2;
    const int stride = layer.stride_h > 0 ? layer.stride_h : kernel_size;
    const int padding = layer.pad_h >= 0 ? layer.pad_h : 0;
    
    const int out_length = (length + 2 * padding - kernel_size) / stride + 1;
    std::vector<float> output(channels * out_length, -std::numeric_limits<float>::infinity());
    
    for (int c = 0; c < channels; ++c) {
        for (int ol = 0; ol < out_length; ++ol) {
            float max_val = -std::numeric_limits<float>::infinity();
            
            for (int k = 0; k < kernel_size; ++k) {
                int il = ol * stride + k - padding;
                
                if (il >= 0 && il < length) {
                    int in_idx = c * length + il;
                    if (in_idx < static_cast<int>(input.size())) {
                        max_val = std::max(max_val, input[in_idx]);
                    }
                }
            }
            
            output[c * out_length + ol] = max_val;
        }
    }
    
    return output;
}

// AvgPool1d: Average pooling 1D
inline std::vector<float> avgpool1d_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    const int channels = layer.in_channels > 0 ? layer.in_channels : 1;
    const int length = static_cast<int>(input.size()) / channels;
    const int kernel_size = layer.kernel_h > 0 ? layer.kernel_h : 2;
    const int stride = layer.stride_h > 0 ? layer.stride_h : kernel_size;
    const int padding = layer.pad_h >= 0 ? layer.pad_h : 0;
    
    const int out_length = (length + 2 * padding - kernel_size) / stride + 1;
    std::vector<float> output(channels * out_length, 0.0f);
    
    for (int c = 0; c < channels; ++c) {
        for (int ol = 0; ol < out_length; ++ol) {
            float sum = 0.0f;
            int count = 0;
            
            for (int k = 0; k < kernel_size; ++k) {
                int il = ol * stride + k - padding;
                
                if (il >= 0 && il < length) {
                    int in_idx = c * length + il;
                    if (in_idx < static_cast<int>(input.size())) {
                        sum += input[in_idx];
                        count++;
                    }
                }
            }
            
            output[c * out_length + ol] = count > 0 ? sum / count : 0.0f;
        }
    }
    
    return output;
}

// ============================================================================
// PADDING OPERATIONS
// ============================================================================

// ZeroPad2d: Pad tensor with zeros
inline std::vector<float> zero_pad2d_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    const int channels = layer.in_channels > 0 ? layer.in_channels : 3;
    const int in_height = layer.input_height > 0 ? layer.input_height : 32;
    const int in_width = layer.input_width > 0 ? layer.input_width : 32;
    const int pad_top = layer.pad_h >= 0 ? layer.pad_h : 1;
    const int pad_bottom = layer.pad_h >= 0 ? layer.pad_h : 1;
    const int pad_left = layer.pad_w >= 0 ? layer.pad_w : 1;
    const int pad_right = layer.pad_w >= 0 ? layer.pad_w : 1;
    
    const int out_height = in_height + pad_top + pad_bottom;
    const int out_width = in_width + pad_left + pad_right;
    
    std::vector<float> output(channels * out_height * out_width, 0.0f);
    
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < in_height; ++h) {
            for (int w = 0; w < in_width; ++w) {
                int in_idx = c * (in_height * in_width) + h * in_width + w;
                int out_idx = c * (out_height * out_width) + 
                              (h + pad_top) * out_width + (w + pad_left);
                output[out_idx] = input[in_idx];
            }
        }
    }
    
    return output;
}

// ReflectionPad2d: Reflection padding
inline std::vector<float> reflection_pad2d_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    const int channels = layer.in_channels > 0 ? layer.in_channels : 3;
    const int in_height = layer.input_height > 0 ? layer.input_height : 32;
    const int in_width = layer.input_width > 0 ? layer.input_width : 32;
    const int pad = layer.pad_h >= 0 ? layer.pad_h : 1;
    
    const int out_height = in_height + 2 * pad;
    const int out_width = in_width + 2 * pad;
    
    std::vector<float> output(channels * out_height * out_width, 0.0f);
    
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                // Map output position to input with reflection
                int ih = oh - pad;
                int iw = ow - pad;
                
                // Reflect boundary
                if (ih < 0) ih = -ih;
                if (ih >= in_height) ih = 2 * in_height - ih - 2;
                if (iw < 0) iw = -iw;
                if (iw >= in_width) iw = 2 * in_width - iw - 2;
                
                // Clamp to valid range
                ih = std::max(0, std::min(ih, in_height - 1));
                iw = std::max(0, std::min(iw, in_width - 1));
                
                int in_idx = c * (in_height * in_width) + ih * in_width + iw;
                int out_idx = c * (out_height * out_width) + oh * out_width + ow;
                
                output[out_idx] = input[in_idx];
            }
        }
    }
    
    return output;
}

// ReplicationPad2d: Replicate border values
inline std::vector<float> replication_pad2d_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    const int channels = layer.in_channels > 0 ? layer.in_channels : 3;
    const int in_height = layer.input_height > 0 ? layer.input_height : 32;
    const int in_width = layer.input_width > 0 ? layer.input_width : 32;
    const int pad = layer.pad_h >= 0 ? layer.pad_h : 1;
    
    const int out_height = in_height + 2 * pad;
    const int out_width = in_width + 2 * pad;
    
    std::vector<float> output(channels * out_height * out_width, 0.0f);
    
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                // Map to input with clamping (replication)
                int ih = std::max(0, std::min(oh - pad, in_height - 1));
                int iw = std::max(0, std::min(ow - pad, in_width - 1));
                
                int in_idx = c * (in_height * in_width) + ih * in_width + iw;
                int out_idx = c * (out_height * out_width) + oh * out_width + ow;
                
                output[out_idx] = input[in_idx];
            }
        }
    }
    
    return output;
}

// ============================================================================
// UPSAMPLING MANQUANTES
// ============================================================================

// UpsampleBicubic: Bicubic interpolation upsampling
// Implémentation simplifiée (bicubic complet nécessite interpolation cubique)
inline std::vector<float> upsample_bicubic_forward(
    const std::vector<float>& input,
    int in_h, int in_w, int channels,
    int out_h, int out_w
) {
    std::vector<float> output(channels * out_h * out_w, 0.0f);

    auto cubic_weight = [](float x) -> float {
        constexpr float a = -0.75f;
        x = std::abs(x);
        if (x <= 1.0f) {
            return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
        }
        if (x < 2.0f) {
            return (((a * x - 5.0f * a) * x + 8.0f * a) * x) - 4.0f * a;
        }
        return 0.0f;
    };

    const float scale_h = static_cast<float>(in_h) / out_h;
    const float scale_w = static_cast<float>(in_w) / out_w;

    const size_t out_hw = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
    const size_t in_hw = static_cast<size_t>(in_h) * static_cast<size_t>(in_w);
    const size_t total = static_cast<size_t>(channels) * out_hw;

    #pragma omp parallel for if(total >= 65536) schedule(static)
    for (size_t idx = 0; idx < total; ++idx) {
        const int c = static_cast<int>(idx / out_hw);
        const size_t rem = idx - static_cast<size_t>(c) * out_hw;
        const int oh = static_cast<int>(rem / static_cast<size_t>(out_w));
        const int ow = static_cast<int>(rem - static_cast<size_t>(oh) * static_cast<size_t>(out_w));

        const float ih_f = (static_cast<float>(oh) + 0.5f) * scale_h - 0.5f;
        const float iw_f = (static_cast<float>(ow) + 0.5f) * scale_w - 0.5f;
        const int ih_base = static_cast<int>(std::floor(ih_f));
        const int iw_base = static_cast<int>(std::floor(iw_f));

        const size_t base = static_cast<size_t>(c) * in_hw;
        float sum = 0.0f;
        float weight_sum = 0.0f;
        for (int mh = -1; mh <= 2; ++mh) {
            const int ih = std::max(0, std::min(ih_base + mh, in_h - 1));
            const float wh = cubic_weight(ih_f - static_cast<float>(ih_base + mh));
            for (int mw = -1; mw <= 2; ++mw) {
                const int iw = std::max(0, std::min(iw_base + mw, in_w - 1));
                const float ww = cubic_weight(iw_f - static_cast<float>(iw_base + mw));
                const float w = wh * ww;
                sum += w * input[base + static_cast<size_t>(ih) * static_cast<size_t>(in_w) + static_cast<size_t>(iw)];
                weight_sum += w;
            }
        }

        output[idx] = (weight_sum != 0.0f) ? (sum / weight_sum) : 0.0f;
    }

    return output;
}

// PixelShuffle: Rearrange tensor from (C*r^2, H, W) to (C, H*r, W*r)
inline std::vector<float> pixel_shuffle_forward(
    const std::vector<float>& input,
    const Layer& layer
) {
    const int upscale_factor = layer.scale_h > 0 ? layer.scale_h : 2;
    const int in_channels = layer.in_channels > 0 ? layer.in_channels : 12;  // Must be divisible by r^2
    const int height = layer.input_height > 0 ? layer.input_height : 32;
    const int width = layer.input_width > 0 ? layer.input_width : 32;
    
    const int r = upscale_factor;
    const int r2 = r * r;
    
    if (in_channels % r2 != 0) {
        throw std::runtime_error(
            "PixelShuffle: in_channels (" + std::to_string(in_channels) +
            ") must be divisible by upscale_factor^2 (" + std::to_string(r2) + ")"
        );
    }
    
    const int out_channels = in_channels / r2;
    const int out_height = height * r;
    const int out_width = width * r;
    
    std::vector<float> output(out_channels * out_height * out_width, 0.0f);
    
    const size_t out_hw = static_cast<size_t>(out_height) * static_cast<size_t>(out_width);
    const size_t in_hw = static_cast<size_t>(height) * static_cast<size_t>(width);
    const size_t total = static_cast<size_t>(out_channels) * out_hw;
    #pragma omp simd
    for (size_t out_idx = 0; out_idx < total; ++out_idx) {
        const int oc = static_cast<int>(out_idx / out_hw);
        const size_t rem = out_idx - static_cast<size_t>(oc) * out_hw;
        const int oh = static_cast<int>(rem / static_cast<size_t>(out_width));
        const int ow = static_cast<int>(rem - static_cast<size_t>(oh) * static_cast<size_t>(out_width));

        const int ih = oh / r;
        const int iw = ow / r;
        const int sub_h = oh - ih * r;
        const int sub_w = ow - iw * r;
        const int ic = oc * r2 + sub_h * r + sub_w;

        const size_t in_idx = static_cast<size_t>(ic) * in_hw
            + static_cast<size_t>(ih) * static_cast<size_t>(width)
            + static_cast<size_t>(iw);
        if (in_idx < input.size()) {
            output[out_idx] = input[in_idx];
        }
    }
    
    return output;
}

} // namespace LayerOpsExt
