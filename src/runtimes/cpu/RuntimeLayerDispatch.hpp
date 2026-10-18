#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../Layers.hpp"
#include "LayerOps.hpp"
#include "../ops_sequence_constant.hpp"
#include "LayerOpsExt.hpp"
#include "ConvolutionKernels.hpp"
#include "MatMulKernels.hpp"
#include "FlatJoinKernels.hpp"

namespace RuntimeLayerDispatch {

inline bool cpu_supports_forward_layer_type(LayerType type) {
    return type != LayerType::UNKNOWN;
}

inline bool cpu_supports_backward_layer_type(LayerType type) {
    return type != LayerType::UNKNOWN;
}

inline bool cpu_forward_layer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training,
    std::vector<uint8_t>* output_mask = nullptr
) {
    if (layer.type_enum != LayerType::Constant && (inputs.empty() || inputs[0] == nullptr)) return false;
    static const std::vector<float> empty;
    const std::vector<float>& x = inputs.empty() ? empty : *inputs[0];
    outputs.clear();

    try {
        switch (layer.type_enum) {
            // ====================================================================
            // CONVOLUTION
            // ====================================================================
            case LayerType::Conv2d: {
                const int in_c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int out_c = layer.out_channels > 0 ? layer.out_channels : 1;
                const int k = layer.kernel_h > 0 ? layer.kernel_h : layer.get_kernel_h();
                const int stride = layer.get_stride_h();
                const int pad = layer.get_pad_h();
                const int dilation = layer.dilation_h != 1 ? layer.dilation_h : layer.dilation;
                int H = layer.input_height > 0 ? layer.input_height : 0;
                int W = layer.input_width > 0 ? layer.input_width : 0;

                if (in_c > 0 && !x.empty() && (x.size() % static_cast<size_t>(in_c)) == 0) {
                    const size_t hw = x.size() / static_cast<size_t>(in_c);
                    const size_t cfg_hw = static_cast<size_t>(std::max(1, H)) * static_cast<size_t>(std::max(1, W));
                    if (H <= 0 || W <= 0 || cfg_hw != hw) {
                        bool fixed = false;
                        if (H > 0 && (hw % static_cast<size_t>(H)) == 0) {
                            W = static_cast<int>(hw / static_cast<size_t>(H));
                            fixed = true;
                        } else if (W > 0 && (hw % static_cast<size_t>(W)) == 0) {
                            H = static_cast<int>(hw / static_cast<size_t>(W));
                            fixed = true;
                        }
                        if (!fixed) {
                            const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                            if (s > 0 && s * s == hw) {
                                H = static_cast<int>(s);
                                W = static_cast<int>(s);
                                fixed = true;
                            }
                        }
                        if (!fixed) {
                            H = 1;
                            W = static_cast<int>(hw);
                        }
                    }
                }
                if (H <= 0 || W <= 0) return false;

                const float* w = layer.getWeights();
                if (!w) return false;

                const size_t w_kernel = static_cast<size_t>(out_c) * static_cast<size_t>(in_c) * static_cast<size_t>(k) * static_cast<size_t>(k);
                const size_t need = w_kernel + (layer.use_bias ? static_cast<size_t>(out_c) : 0ULL);
                if (layer.getWeightsSize() < need) return false;

                outputs.resize(1);
                CpuConvolution::forward(x, outputs[0], w, layer.use_bias ? w + w_kernel : nullptr, H, W, in_c, out_c, k, stride, pad, dilation);
                return true;
            }

            case LayerType::ConvTranspose2d: {
                const int in_c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int out_c = layer.out_channels > 0 ? layer.out_channels : 1;
                const int k = layer.kernel_h > 0 ? layer.kernel_h : layer.get_kernel_h();
                const int stride = layer.get_stride_h();
                const int pad = layer.get_pad_h();
                int H = layer.input_height > 0 ? layer.input_height : 0;
                int W = layer.input_width > 0 ? layer.input_width : 0;

                if (in_c > 0 && !x.empty() && (x.size() % static_cast<size_t>(in_c)) == 0) {
                    const size_t hw = x.size() / static_cast<size_t>(in_c);
                    const size_t cfg_hw = static_cast<size_t>(std::max(1, H)) * static_cast<size_t>(std::max(1, W));
                    if (H <= 0 || W <= 0 || cfg_hw != hw) {
                        bool fixed = false;
                        if (H > 0 && (hw % static_cast<size_t>(H)) == 0) {
                            W = static_cast<int>(hw / static_cast<size_t>(H));
                            fixed = true;
                        } else if (W > 0 && (hw % static_cast<size_t>(W)) == 0) {
                            H = static_cast<int>(hw / static_cast<size_t>(W));
                            fixed = true;
                        }
                        if (!fixed) {
                            const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                            if (s > 0 && s * s == hw) {
                                H = static_cast<int>(s);
                                W = static_cast<int>(s);
                                fixed = true;
                            }
                        }
                        if (!fixed) {
                            H = 1;
                            W = static_cast<int>(hw);
                        }
                    }
                }
                if (H <= 0 || W <= 0) return false;

                const float* w = layer.getWeights();
                if (!w) return false;
                const size_t w_kernel = static_cast<size_t>(out_c) * static_cast<size_t>(in_c) * static_cast<size_t>(k) * static_cast<size_t>(k);
                const size_t need = w_kernel + (layer.use_bias ? static_cast<size_t>(out_c) : 0ULL);
                if (layer.getWeightsSize() < need) return false;

                std::vector<float> kernel(w, w + w_kernel);
                std::vector<float> bias;
                if (layer.use_bias) {
                    bias.assign(w + w_kernel, w + w_kernel + static_cast<size_t>(out_c));
                }

                outputs.resize(1);
                Conv::conv_transpose2d(x, outputs[0], kernel, bias, H, W, in_c, out_c, k, stride, pad);
                return true;
            }

            case LayerType::Conv1d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::conv1d_forward(x, layer);
                return true;
            }

            case LayerType::DepthwiseConv2d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::depthwise_conv2d_forward(x, layer);
                return true;
            }

            // ====================================================================
            // BASIC / LINEAR
            // ====================================================================
            case LayerType::Identity: {
                outputs.resize(1);
                outputs[0] = LayerOps::identity_forward(x);
                return true;
            }

            case LayerType::Linear: {
                outputs.resize(1);
                outputs[0] = LayerOps::linear_forward(x, layer);
                return true;
            }

            case LayerType::Bilinear: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                const std::vector<float>& y = *inputs[1];
                const int in1 = layer.in_features;
                const int in2 = layer.out_features;
                const int out = (layer.embed_dim > 0) ? layer.embed_dim : (!layer.target_shape.empty() ? layer.target_shape[0] : 0);
                if (in1 <= 0 || in2 <= 0 || out <= 0) return false;

                const int batch = static_cast<int>(x.size()) / in1;
                if (batch <= 0) return false;
                if (static_cast<int>(x.size()) != batch * in1) return false;
                if (static_cast<int>(y.size()) != batch * in2) return false;

                const float* w = layer.getWeights();
                if (!w) return false;

                const size_t w_sz = static_cast<size_t>(out) * static_cast<size_t>(in1) * static_cast<size_t>(in2);
                const bool has_bias = layer.use_bias;
                const float* b = has_bias ? (w + w_sz) : nullptr;

                outputs.resize(1);
                outputs[0].assign(static_cast<size_t>(batch) * static_cast<size_t>(out), 0.0f);
                for (int n = 0; n < batch; ++n) {
                    const float* x1 = &x[static_cast<size_t>(n) * static_cast<size_t>(in1)];
                    const float* x2 = &y[static_cast<size_t>(n) * static_cast<size_t>(in2)];
                    for (int o = 0; o < out; ++o) {
                        float sum = has_bias ? b[static_cast<size_t>(o)] : 0.0f;
                        const float* Wo = w + static_cast<size_t>(o) * static_cast<size_t>(in1) * static_cast<size_t>(in2);
                        for (int i = 0; i < in1; ++i) {
                            const float xi = x1[static_cast<size_t>(i)];
                            const float* Woi = Wo + static_cast<size_t>(i) * static_cast<size_t>(in2);
                            for (int j = 0; j < in2; ++j) {
                                sum += xi * Woi[static_cast<size_t>(j)] * x2[static_cast<size_t>(j)];
                            }
                        }
                        outputs[0][static_cast<size_t>(n) * static_cast<size_t>(out) + static_cast<size_t>(o)] = sum;
                    }
                }
                return true;
            }

            // ====================================================================
            // EMBEDDING
            // ====================================================================
            case LayerType::Embedding: {
                // Convention CPU (mode float): x contient des ids (arrondis).
                const int vocab = std::max(1, layer.vocab_size);
                const int dim = std::max(1, layer.embed_dim > 0 ? layer.embed_dim : layer.out_features);
                const int pad = layer.padding_idx;
                const float* w = layer.getWeights();
                if (!w) return false;

                outputs.resize(1);
                outputs[0].assign(x.size() * static_cast<size_t>(dim), 0.0f);
                for (size_t t = 0; t < x.size(); ++t) {
                    const int id = static_cast<int>(std::llround(static_cast<double>(x[t])));
                    if (pad >= 0 && id == pad) continue;
                    if (id < 0 || id >= vocab) continue;
                    const size_t base_w = static_cast<size_t>(id) * static_cast<size_t>(dim);
                    const size_t base_o = t * static_cast<size_t>(dim);
                    for (int d = 0; d < dim; ++d) {
                        outputs[0][base_o + static_cast<size_t>(d)] = w[base_w + static_cast<size_t>(d)];
                    }
                }
                return true;
            }

            case LayerType::EmbeddingBag: {
                if (inputs.empty() || inputs[0] == nullptr) return false;
                const std::vector<float>& ids_f = *inputs[0];
                const std::vector<float>* offsets_f = (inputs.size() >= 2) ? inputs[1] : nullptr;

                const int vocab = std::max(1, layer.vocab_size);
                const int dim = std::max(1, layer.embed_dim);
                const int pad = layer.padding_idx;
                const float* w = layer.getWeights();
                if (!w) return false;

                std::vector<int> offsets;
                int num_bags = 1;
                if (offsets_f && !offsets_f->empty()) {
                    offsets.reserve(offsets_f->size());
                    for (float v : *offsets_f) offsets.push_back(static_cast<int>(std::llround(static_cast<double>(v))));
                    if (offsets.size() < 2) return false;
                    num_bags = static_cast<int>(offsets.size()) - 1;
                } else {
                    offsets = {0, static_cast<int>(ids_f.size())};
                    num_bags = 1;
                }

                outputs.resize(1);
                outputs[0].assign(static_cast<size_t>(num_bags) * static_cast<size_t>(dim), 0.0f);
                for (int b = 0; b < num_bags; ++b) {
                    const int start = std::clamp(offsets[static_cast<size_t>(b)], 0, static_cast<int>(ids_f.size()));
                    const int end = std::clamp(offsets[static_cast<size_t>(b + 1)], start, static_cast<int>(ids_f.size()));
                    float* outp = &outputs[0][static_cast<size_t>(b) * static_cast<size_t>(dim)];
                    for (int t = start; t < end; ++t) {
                        const int id = static_cast<int>(std::llround(static_cast<double>(ids_f[static_cast<size_t>(t)])));
                        if (pad >= 0 && id == pad) continue;
                        if (id < 0 || id >= vocab) continue;
                        const size_t base_w = static_cast<size_t>(id) * static_cast<size_t>(dim);
                        for (int d = 0; d < dim; ++d) {
                            outp[static_cast<size_t>(d)] += w[base_w + static_cast<size_t>(d)];
                        }
                    }
                }
                return true;
            }

            // ====================================================================
            // NORMALIZATION
            // ====================================================================
            case LayerType::BatchNorm2d:
            case LayerType::BatchNorm1d: {
                outputs.resize(1);
                outputs[0] = LayerOps::batchnorm_forward(x, layer, training);
                return true;
            }

            case LayerType::LayerNorm: {
                outputs.resize(1);
                outputs[0] = LayerOps::layernorm_forward(x, layer, training);
                return true;
            }

            case LayerType::GroupNorm: {
                outputs.resize(1);
                outputs[0] = LayerOps::groupnorm_forward(x, layer, training);
                return true;
            }

            case LayerType::InstanceNorm2d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::instance_norm2d_forward(x, layer);
                return true;
            }

            case LayerType::RMSNorm: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::rms_norm_forward(x, layer);
                return true;
            }

            // ====================================================================
            // ACTIVATIONS
            // ====================================================================
            case LayerType::ReLU: {
                outputs.resize(1);
                outputs[0] = LayerOps::relu_forward(x);
                return true;
            }
            case LayerType::LeakyReLU: {
                const float alpha = layer.leaky_relu_alpha > 0 ? layer.leaky_relu_alpha : 0.01f;
                outputs.resize(1);
                outputs[0] = LayerOpsExt::leaky_relu_forward(x, alpha);
                return true;
            }
            case LayerType::GELU: {
                outputs.resize(1);
                outputs[0] = LayerOps::gelu_forward(x);
                return true;
            }
            case LayerType::GEGLU: {
                if (layer.seq_len <= 0 || layer.out_features <= 0) return false;
                outputs.resize(1);
                outputs[0] = LayerOps::geglu_forward(x, layer.seq_len, layer.out_features);
                return true;
            }
            case LayerType::SiLU: {
                outputs.resize(1);
                outputs[0] = LayerOps::silu_forward(x);
                return true;
            }
            case LayerType::Tanh: {
                outputs.resize(1);
                outputs[0] = LayerOps::tanh_forward(x);
                return true;
            }
            case LayerType::Sigmoid: {
                outputs.resize(1);
                outputs[0] = LayerOps::sigmoid_forward(x);
                return true;
            }
            case LayerType::Softmax:
            case LayerType::LogSoftmax: {
                outputs.resize(1);
                outputs[0] = LayerOps::softmax_forward(x, layer);
                return true;
            }
            case LayerType::Softplus: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::softplus_forward(x);
                return true;
            }
            case LayerType::Mish: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::mish_forward(x);
                return true;
            }
            case LayerType::HardSigmoid: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::hard_sigmoid_forward(x);
                return true;
            }
            case LayerType::HardSwish: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::hard_swish_forward(x);
                return true;
            }

            // ====================================================================
            // POOLING
            // ====================================================================
            case LayerType::MaxPool2d: {
                const int kernel_size = layer.get_kernel_h();
                const int stride = layer.get_stride_h();
                const int padding = layer.get_pad_h();

                const int in_channels = layer.in_channels > 0 ? layer.in_channels : 1;
                const int height = layer.input_height > 0 ? layer.input_height : 1;
                const int width = layer.input_width > 0 ? layer.input_width : 1;

                if (kernel_size <= 0 || stride <= 0) return false;

                const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
                const int out_width = (width + 2 * padding - kernel_size) / stride + 1;
                if (out_height <= 0 || out_width <= 0) return false;

                outputs.resize(1);
                outputs[0].assign(static_cast<size_t>(in_channels) * static_cast<size_t>(out_height) * static_cast<size_t>(out_width),
                                  -std::numeric_limits<float>::infinity());

                for (int c = 0; c < in_channels; ++c) {
                    for (int oh = 0; oh < out_height; ++oh) {
                        for (int ow = 0; ow < out_width; ++ow) {
                            float max_val = -std::numeric_limits<float>::infinity();
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    const int ih = oh * stride + kh - padding;
                                    const int iw = ow * stride + kw - padding;
                                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                        const int in_idx = c * (height * width) + ih * width + iw;
                                        if (in_idx >= 0 && static_cast<size_t>(in_idx) < x.size()) {
                                            max_val = std::max(max_val, x[static_cast<size_t>(in_idx)]);
                                        }
                                    }
                                }
                            }
                            const int out_idx = c * (out_height * out_width) + oh * out_width + ow;
                            outputs[0][static_cast<size_t>(out_idx)] = max_val;
                        }
                    }
                }
                return true;
            }

            case LayerType::MaxPool1d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::maxpool1d_forward(x, layer);
                return true;
            }

            case LayerType::AvgPool2d: {
                outputs.resize(1);
                outputs[0] = LayerOps::avgpool2d_forward(x, layer);
                return true;
            }

            case LayerType::AvgPool1d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::avgpool1d_forward(x, layer);
                return true;
            }

            case LayerType::TokenMeanPool: {
                const int seq_len = layer.seq_len > 0 ? layer.seq_len : 0;
                const int embed_dim = layer.embed_dim > 0 ? layer.embed_dim : 0;
                if (seq_len <= 0 || embed_dim <= 0) return false;
                if (static_cast<int>(x.size()) != seq_len * embed_dim) return false;

                outputs.resize(1);
                outputs[0].assign(static_cast<size_t>(embed_dim), 0.0f);
                for (int t = 0; t < seq_len; ++t) {
                    const int base = t * embed_dim;
                    for (int d = 0; d < embed_dim; ++d) {
                        outputs[0][static_cast<size_t>(d)] += x[static_cast<size_t>(base + d)];
                    }
                }
                const float inv = 1.0f / static_cast<float>(seq_len);
                for (int d = 0; d < embed_dim; ++d) {
                    outputs[0][static_cast<size_t>(d)] *= inv;
                }
                return true;
            }

            case LayerType::GlobalAvgPool2d:
            case LayerType::AdaptiveAvgPool2d: {
                outputs.resize(1);
                outputs[0] = LayerOps::global_avgpool2d_forward(x, layer);
                return true;
            }

            // ====================================================================
            // DROPOUT
            // ====================================================================
            case LayerType::Dropout:
            case LayerType::Dropout2d: {
                outputs.resize(1);
                outputs[0] = LayerOps::dropout_forward(x, layer, training, output_mask);
                return true;
            }

            case LayerType::AlphaDropout: {
                if (output_mask) output_mask->assign(x.size(),1);
                if (!training) {
                    outputs.resize(1);
                    outputs[0] = x;
                    return true;
                }

                const float p = std::clamp(layer.dropout_p, 0.0f, 1.0f);
                if (p >= 1.0f) {
                    outputs.assign(1,std::vector<float>(x.size(),0.0f));
                    if (output_mask) output_mask->assign(x.size(),0);
                    return true;
                }
                const float alpha = 1.6732632423543772848170429916717f;
                const float scale = 1.0507009873554804934193349852946f;
                const float alpha_p = -alpha * scale;

                const float a = 1.0f / std::sqrt((1.0f - p) * (1.0f + p * alpha_p * alpha_p));
                const float b = -a * alpha_p * p;

                outputs.resize(1);
                outputs[0].resize(x.size());
                auto& rng = MimirRng::generator();
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                for (size_t i = 0; i < x.size(); ++i) {
                    const bool keep = (dist(rng) > p);
                    if (output_mask) (*output_mask)[i]=keep;
                    const float v = keep ? x[i] : alpha_p;
                    outputs[0][i] = a * v + b;
                }
                return true;
            }

            // ====================================================================
            // SHAPE
            // ====================================================================
            case LayerType::Flatten: {
                outputs.resize(1);
                outputs[0] = LayerOps::flatten_forward(x, layer);
                return true;
            }

            case LayerType::Reshape:
            case LayerType::View: {
                outputs.resize(1);
                outputs[0] = LayerOps::reshape_forward(x, layer);
                return true;
            }

            case LayerType::Transpose: {
                if (layer.in_features <= 0 || layer.out_features <= 0) return false;
                outputs.resize(1);
                outputs[0] = LayerOps::transpose_forward(x, layer.in_features, layer.out_features);
                return true;
            }

            case LayerType::Permute: {
                if (layer.permute_dims.empty()) return false;
                std::vector<int> shape = layer.shape;
                if (shape.empty()) {
                    shape = {1, static_cast<int>(x.size())};
                }
                outputs.resize(1);
                outputs[0] = LayerOps::permute_forward(x, layer.permute_dims, shape);
                return true;
            }

            case LayerType::Squeeze: {
                std::vector<int> in_shape = {static_cast<int>(x.size())};
                std::vector<int> out_shape;
                outputs.resize(1);
                outputs[0] = LayerOpsExt::squeeze_forward(x, in_shape, out_shape, layer.squeeze_dim);
                return true;
            }

            case LayerType::Unsqueeze: {
                std::vector<int> in_shape = {static_cast<int>(x.size())};
                std::vector<int> out_shape;
                outputs.resize(1);
                outputs[0] = LayerOpsExt::unsqueeze_forward(x, in_shape, out_shape, layer.unsqueeze_dim);
                return true;
            }

            case LayerType::Lambda: {
                // Non supporté via runtime (callbacks Lua).
                return false;
            }

            // ====================================================================
            // CUSTOM (Mímir)
            // ====================================================================
            case LayerType::PatchEmbed: {
                const int d_model = layer.embed_dim > 0 ? layer.embed_dim : layer.out_features;
                const int seq_text = std::max(1, layer.seq_text);
                const int num_patches = std::max(1, layer.num_patches);
                const int patch_dim = std::max(1, layer.patch_dim);

                const int text_dim = (seq_text + 1) * d_model;
                const int in_dim = text_dim + num_patches * patch_dim;
                const int out_dim = (seq_text + 1 + num_patches) * d_model;
                if (static_cast<int>(x.size()) != in_dim) return false;

                const float* w = layer.getWeights();
                if (!w) return false;
                const int expected_w = patch_dim * d_model + d_model;
                if (static_cast<int>(layer.getWeightsSize()) != expected_w) return false;

                outputs.resize(1);
                outputs[0].assign(static_cast<size_t>(out_dim), 0.0f);
                std::copy(x.begin(), x.begin() + text_dim, outputs[0].begin());

                const float* b = w + patch_dim * d_model;
                const float inv = 1.0f / std::sqrt(static_cast<float>(patch_dim));
                for (int p = 0; p < num_patches; ++p) {
                    const int in_off = text_dim + p * patch_dim;
                    const int out_off = (seq_text + 1 + p) * d_model;
                    for (int d = 0; d < d_model; ++d) {
                        float sum = b[d];
                        for (int k = 0; k < patch_dim; ++k) {
                            sum += (x[static_cast<size_t>(in_off + k)] * inv) * w[static_cast<size_t>(k * d_model + d)];
                        }
                        outputs[0][static_cast<size_t>(out_off + d)] = sum;
                    }
                }
                return true;
            }

            // ====================================================================
            // ELEMENT-WISE
            // ====================================================================
            case LayerType::Add: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                outputs.resize(1);
                outputs[0] = LayerOps::add_forward(*inputs[0], *inputs[1]);
                return true;
            }

            case LayerType::Subtract: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                outputs.resize(1);
                outputs[0] = LayerOpsExt::subtract_forward(*inputs[0], *inputs[1]);
                return true;
            }

            case LayerType::Multiply: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                outputs.resize(1);
                outputs[0] = LayerOps::multiply_forward(*inputs[0], *inputs[1]);
                return true;
            }

            case LayerType::Divide: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                outputs.resize(1);
                outputs[0] = LayerOpsExt::divide_forward(*inputs[0], *inputs[1]);
                return true;
            }

            case LayerType::Reparameterize: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                const std::vector<float>& mu = *inputs[0];
                const std::vector<float>& logvar = *inputs[1];
                if (mu.size() != logvar.size()) return false;

                outputs.resize(1);
                outputs[0].resize(mu.size());

                if (!training) {
                    outputs[0] = mu;
                    return true;
                }

                static thread_local std::mt19937 rng(1337);
                std::normal_distribution<float> n01(0.0f, 1.0f);
                for (size_t i = 0; i < mu.size(); ++i) {
                    const float lv = std::clamp(logvar[i], -20.0f, 20.0f);
                    const float stdv = std::exp(0.5f * lv);
                    outputs[0][i] = mu[i] + stdv * n01(rng);
                }
                return true;
            }

            // ====================================================================
            // TENSOR OPS
            // ====================================================================
            case LayerType::Concat: {
                if (inputs.size() < 2) return false;
                outputs.resize(1);
                return RuntimeFlatJoin::forward(inputs, outputs[0], false);
            }

            case LayerType::Split: {
                if (!layer.split_sizes.empty()) {
                    outputs = LayerOps::split_forward(x, layer.split_sizes, layer.split_axis);
                    return true;
                }
                if (layer.num_splits > 0) {
                    outputs = LayerOps::split_forward(x, layer.num_splits, layer.split_axis);
                    return true;
                }
                return false;
            }

            case LayerType::Chunk: {
                if (layer.num_chunks <= 0) return false;
                outputs = LayerOpsExt::chunk_forward(x, layer.num_chunks, layer.split_axis);
                return true;
            }

            case LayerType::Stack: {
                if (inputs.size() < 2) return false;
                outputs.resize(1);
                return RuntimeFlatJoin::forward(inputs, outputs[0], true);
            }

            case LayerType::MatMul: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                if (layer.in_features <= 0 || layer.out_features <= 0 || layer.embed_dim <= 0) return false;
                const int M = layer.in_features;
                const int K = layer.out_features;
                const int N = layer.embed_dim;
                outputs.resize(1);
                outputs[0] = LayerOps::matmul_forward(*inputs[0], *inputs[1], M, K, N);
                return true;
            }

            case LayerType::BatchMatMul: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                if (layer.seq_len <= 0 || layer.in_features <= 0 || layer.out_features <= 0 || layer.embed_dim <= 0) return false;

                const int B = layer.seq_len;
                const int M = layer.in_features;
                const int K = layer.out_features;
                const int N = layer.embed_dim;
                const std::vector<float>& A = *inputs[0];
                const std::vector<float>& Bm = *inputs[1];
                if (A.size() != static_cast<size_t>(B) * M * K) return false;
                if (Bm.size() != static_cast<size_t>(B) * K * N) return false;

                outputs.resize(1);
                outputs[0].assign(static_cast<size_t>(B) * static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);

                RuntimeMatMul::forwardBatched(A.data(), Bm.data(), outputs[0].data(),
                    static_cast<size_t>(B), static_cast<size_t>(M),
                    static_cast<size_t>(K), static_cast<size_t>(N));
                return true;
            }

            case LayerType::NMS: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                const std::vector<float>& boxes = *inputs[0];
                const std::vector<float>& scores = *inputs[1];
                if (boxes.size() % 4 != 0) return false;

                const size_t count = boxes.size() / 4;
                if (scores.size() != count) return false;
                const std::vector<float>* classes =
                    (inputs.size() >= 3) ? inputs[2] : nullptr;
                if (classes != nullptr && classes->size() != count) return false;
                if (!std::isfinite(layer.nms_iou_threshold) ||
                    layer.nms_iou_threshold < 0.0f ||
                    layer.nms_iou_threshold > 1.0f ||
                    !std::isfinite(layer.nms_score_threshold) ||
                    layer.nms_max_detections < 0) {
                    return false;
                }

                struct Candidate {
                    size_t index;
                    float score;
                    int class_id;
                };
                std::vector<Candidate> candidates;
                candidates.reserve(count);
                for (size_t i = 0; i < count; ++i) {
                    const float score = scores[i];
                    const size_t b = i * 4;
                    if (!std::isfinite(score) ||
                        score < layer.nms_score_threshold ||
                        !std::isfinite(boxes[b]) ||
                        !std::isfinite(boxes[b + 1]) ||
                        !std::isfinite(boxes[b + 2]) ||
                        !std::isfinite(boxes[b + 3])) {
                        continue;
                    }
                    int class_id = 0;
                    if (classes != nullptr) {
                        const float raw_class = (*classes)[i];
                        if (!std::isfinite(raw_class)) continue;
                        class_id = static_cast<int>(std::llround(raw_class));
                    }
                    candidates.push_back({i, score, class_id});
                }

                std::stable_sort(
                    candidates.begin(),
                    candidates.end(),
                    [](const Candidate& a, const Candidate& b) {
                        if (a.score != b.score) return a.score > b.score;
                        return a.index < b.index;
                    });

                auto iou = [&boxes](size_t lhs, size_t rhs) {
                    const size_t a = lhs * 4;
                    const size_t b = rhs * 4;
                    const float ax1 = boxes[a];
                    const float ay1 = boxes[a + 1];
                    const float ax2 = boxes[a + 2];
                    const float ay2 = boxes[a + 3];
                    const float bx1 = boxes[b];
                    const float by1 = boxes[b + 1];
                    const float bx2 = boxes[b + 2];
                    const float by2 = boxes[b + 3];

                    const float area_a =
                        std::max(0.0f, ax2 - ax1) * std::max(0.0f, ay2 - ay1);
                    const float area_b =
                        std::max(0.0f, bx2 - bx1) * std::max(0.0f, by2 - by1);
                    const float iw =
                        std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
                    const float ih =
                        std::max(0.0f, std::min(ay2, by2) - std::max(ay1, by1));
                    const float intersection = iw * ih;
                    const float union_area = area_a + area_b - intersection;
                    return union_area > 0.0f ? intersection / union_area : 0.0f;
                };

                std::vector<Candidate> kept;
                kept.reserve(candidates.size());
                for (const Candidate& candidate : candidates) {
                    bool suppressed = false;
                    for (const Candidate& accepted : kept) {
                        const bool same_class =
                            layer.nms_class_agnostic ||
                            classes == nullptr ||
                            candidate.class_id == accepted.class_id;
                        if (same_class &&
                            iou(candidate.index, accepted.index) >
                                layer.nms_iou_threshold) {
                            suppressed = true;
                            break;
                        }
                    }
                    if (suppressed) continue;
                    kept.push_back(candidate);
                    if (layer.nms_max_detections > 0 &&
                        kept.size() >=
                            static_cast<size_t>(layer.nms_max_detections)) {
                        break;
                    }
                }

                outputs.resize(1);
                outputs[0].reserve(kept.size());
                for (const Candidate& candidate : kept) {
                    outputs[0].push_back(static_cast<float>(candidate.index));
                }
                return true;
            }

            // ====================================================================
            // ATTENTION
            // ====================================================================
            case LayerType::Constant: {
                outputs.resize(1);
                return SequenceConstant::forward(layer,outputs[0]);
            }

            case LayerType::SelfAttention:
            case LayerType::MultiHeadAttention: {
                const float* weights = layer.getWeights();
                if (!weights) return false;

                const int seq_len = layer.seq_len > 0 ? layer.seq_len : 1;
                const int embed_dim = layer.embed_dim > 0 ? layer.embed_dim : static_cast<int>(x.size());
                const int num_heads = layer.num_heads > 0 ? layer.num_heads : 1;
                const bool causal = layer.causal;

                const int qkv_size = embed_dim * embed_dim * 3;
                const int out_size = embed_dim * embed_dim;
                const int qkv_bias_size = embed_dim * 3;
                const int out_bias_size = embed_dim;
                const size_t expected_no_bias = static_cast<size_t>(qkv_size + out_size);
                const size_t expected_out_bias_only = expected_no_bias + static_cast<size_t>(out_bias_size);
                const size_t expected_with_bias = expected_no_bias + static_cast<size_t>(qkv_bias_size + out_bias_size);
                const size_t actual_size = layer.getWeightsSize();
                const bool has_full_bias = (actual_size >= expected_with_bias);
                const bool has_out_bias_only = (!has_full_bias && actual_size >= expected_out_bias_only);

                std::vector<float> qkv_weight(weights, weights + qkv_size);
                std::vector<float> out_weight(weights + qkv_size, weights + qkv_size + out_size);
                std::vector<float> qkv_bias;
                std::vector<float> out_bias;
                if (has_full_bias) {
                    const float* bias_ptr = weights + qkv_size + out_size;
                    qkv_bias.assign(bias_ptr, bias_ptr + qkv_bias_size);
                    out_bias.assign(bias_ptr + qkv_bias_size, bias_ptr + qkv_bias_size + out_bias_size);
                } else if (has_out_bias_only) {
                    const float* bias_ptr = weights + qkv_size + out_size;
                    out_bias.assign(bias_ptr, bias_ptr + out_bias_size);
                }

                outputs.resize(1);
                if (layer.type_enum == LayerType::SelfAttention) {
                    outputs[0] = LayerOps::self_attention_forward(x, qkv_weight, out_weight, qkv_bias, out_bias, seq_len, embed_dim, num_heads, causal);
                } else {
                    outputs[0] = LayerOps::multihead_attention_forward(x, qkv_weight, out_weight, qkv_bias, out_bias, seq_len, embed_dim, num_heads, causal);
                }
                return true;
            }

            case LayerType::CrossAttention: {
                if (inputs.size() < 2 || inputs[1] == nullptr) return false;
                const float* weights = layer.getWeights();
                if (!weights) return false;

                const std::vector<float>& q_in = *inputs[0];
                const std::vector<float>& kv_in = *inputs[1];
                const int num_heads = layer.num_heads > 0 ? layer.num_heads : 1;
                const bool causal = layer.causal;

                int embed_dim = layer.embed_dim;
                if (embed_dim <= 0 && layer.head_dim > 0 && num_heads > 0) {
                    embed_dim = layer.head_dim * num_heads;
                }
                const int kv_embed_dim = layer.in_features > 0 ? layer.in_features : embed_dim;
                if (embed_dim <= 0) return false;
                if ((q_in.size() % static_cast<size_t>(embed_dim)) != 0) return false;
                if ((kv_in.size() % static_cast<size_t>(kv_embed_dim)) != 0) return false;

                const int query_len = static_cast<int>(q_in.size() / static_cast<size_t>(embed_dim));
                const int kv_len = static_cast<int>(kv_in.size() / static_cast<size_t>(kv_embed_dim));
                if (query_len <= 0 || kv_len <= 0) return false;

                const int q_size = embed_dim * embed_dim;
                const int kv_size = kv_embed_dim * (2 * embed_dim);
                const int out_size = embed_dim * embed_dim;
                const int out_bias_size = embed_dim;

                std::vector<float> q_weight(weights, weights + q_size);
                std::vector<float> kv_weight(weights + q_size, weights + q_size + kv_size);
                std::vector<float> out_weight(weights + q_size + kv_size, weights + q_size + kv_size + out_size);
                std::vector<float> out_bias;
                const size_t expected_with_bias = static_cast<size_t>(q_size + kv_size + out_size + out_bias_size);
                if (layer.getWeightsSize() >= expected_with_bias) {
                    const float* bias_ptr = weights + q_size + kv_size + out_size;
                    out_bias.assign(bias_ptr, bias_ptr + out_bias_size);
                }

                outputs.resize(1);
                outputs[0] = LayerOps::cross_attention_forward(
                    q_in, kv_in, q_weight, kv_weight, out_weight, out_bias,
                    query_len, kv_len, embed_dim, kv_embed_dim, num_heads, causal
                );
                return true;
            }

            // ====================================================================
            // UPSAMPLING
            // ====================================================================
            case LayerType::UpsampleNearest: {
                if (layer.in_channels <= 0) return false;
                const int in_h = layer.input_height > 0 ? layer.input_height : layer.out_h;
                const int in_w = layer.input_width > 0 ? layer.input_width : layer.out_w;
                if (in_h <= 0 || in_w <= 0) return false;
                const int c = layer.in_channels;
                const int out_h = layer.output_height > 0 ? layer.output_height : (layer.out_h > 0 ? layer.out_h : 0);
                const int out_w = layer.output_width > 0 ? layer.output_width : (layer.out_w > 0 ? layer.out_w : 0);
                const int sh = layer.scale_h > 0 ? static_cast<int>(std::lround(layer.scale_h))
                                                 : ((out_h > 0) ? std::max(1, out_h / in_h) : 2);
                const int sw = layer.scale_w > 0 ? static_cast<int>(std::lround(layer.scale_w))
                                                 : ((out_w > 0) ? std::max(1, out_w / in_w) : 2);
                outputs.resize(1);
                outputs[0] = LayerOps::upsample_nearest_forward(x, in_h, in_w, c, sh, sw);
                return true;
            }

            case LayerType::UpsampleBilinear: {
                if (layer.in_channels <= 0) return false;
                const int in_h = layer.input_height > 0 ? layer.input_height : layer.out_h;
                const int in_w = layer.input_width > 0 ? layer.input_width : layer.out_w;
                if (in_h <= 0 || in_w <= 0) return false;
                const int c = layer.in_channels;
                const int out_h = layer.output_height > 0 ? layer.output_height : (layer.out_h > 0 ? layer.out_h : std::max(1, static_cast<int>(std::lround(static_cast<float>(in_h) * std::max(0.0f, layer.scale_h)))));
                const int out_w = layer.output_width > 0 ? layer.output_width : (layer.out_w > 0 ? layer.out_w : std::max(1, static_cast<int>(std::lround(static_cast<float>(in_w) * std::max(0.0f, layer.scale_w)))));
                outputs.resize(1);
                outputs[0] = LayerOps::upsample_bilinear_forward(x, in_h, in_w, c, out_h, out_w);
                return true;
            }

            case LayerType::UpsampleBicubic: {
                const int in_h = layer.input_height > 0 ? layer.input_height : layer.out_h;
                const int in_w = layer.input_width > 0 ? layer.input_width : layer.out_w;
                if (in_h <= 0 || in_w <= 0) return false;
                const int c = layer.in_channels > 0 ? layer.in_channels : 3;
                const int out_h = layer.output_height > 0 ? layer.output_height : (layer.out_h > 0 ? layer.out_h : std::max(1, static_cast<int>(std::lround(static_cast<float>(in_h) * std::max(0.0f, layer.scale_h)))));
                const int out_w = layer.output_width > 0 ? layer.output_width : (layer.out_w > 0 ? layer.out_w : std::max(1, static_cast<int>(std::lround(static_cast<float>(in_w) * std::max(0.0f, layer.scale_w)))));
                outputs.resize(1);
                outputs[0] = LayerOpsExt::upsample_bicubic_forward(x, in_h, in_w, c, out_h, out_w);
                return true;
            }

            case LayerType::PixelShuffle: {
                const int r = static_cast<int>(layer.scale_h);
                const int in_channels = layer.in_channels;
                const int in_h = layer.input_height;
                const int in_w = layer.input_width;
                if (r < 1 || static_cast<float>(r) != layer.scale_h ||
                    in_channels < 1 || in_h < 1 || in_w < 1 ||
                    in_channels % (r * r) != 0 ||
                    x.size() != static_cast<size_t>(in_channels) *
                                    static_cast<size_t>(in_h) *
                                    static_cast<size_t>(in_w)) {
                    return false;
                }
                outputs.resize(1);
                outputs[0] = LayerOpsExt::pixel_shuffle_forward(x, layer);
                return true;
            }

            // ====================================================================
            // PADDING
            // ====================================================================
            case LayerType::ZeroPad2d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::zero_pad2d_forward(x, layer);
                return true;
            }
            case LayerType::ReflectionPad2d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::reflection_pad2d_forward(x, layer);
                return true;
            }
            case LayerType::ReplicationPad2d: {
                outputs.resize(1);
                outputs[0] = LayerOpsExt::replication_pad2d_forward(x, layer);
                return true;
            }

            // ====================================================================
            // RECURRENT
            // ====================================================================
            case LayerType::LSTM:
            case LayerType::GRU:
            case LayerType::RNN: {
                // Pour rester fonctionnel et déterministe, on réutilise l'implémentation CPU de Model (naïve).
                const int T = layer.seq_len;
                const int I = layer.in_features;
                const int H = layer.out_features;
                if (T <= 0 || I <= 0 || H <= 0) return false;
                if (static_cast<int>(x.size()) != T * I) return false;
                const float* W = layer.getWeights();
                if (!W) return false;

                auto sigmoid_scalar = [](float v) -> float { return 1.0f / (1.0f + std::exp(-v)); };

                if (layer.type_enum == LayerType::RNN) {
                    const bool use_bias = layer.use_bias;
                    const size_t Wih_sz = static_cast<size_t>(H) * static_cast<size_t>(I);
                    const size_t Whh_sz = static_cast<size_t>(H) * static_cast<size_t>(H);
                    const size_t bih_sz = use_bias ? static_cast<size_t>(H) : 0ULL;
                    const size_t bhh_sz = use_bias ? static_cast<size_t>(H) : 0ULL;
                    const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                    if (layer.getWeightsSize() < need) return false;

                    const float* Wih = W;
                    const float* Whh = Wih + Wih_sz;
                    const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                    const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                    std::vector<float> h_prev(static_cast<size_t>(H), 0.0f);
                    outputs.resize(1);
                    outputs[0].assign(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                    for (int t = 0; t < T; ++t) {
                        const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                        float* ht = &outputs[0][static_cast<size_t>(t) * static_cast<size_t>(H)];
                        for (int h = 0; h < H; ++h) {
                            const float* wih = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                            const float* whh = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                            float sum = 0.0f;
                            for (int i = 0; i < I; ++i) sum += wih[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                            for (int k = 0; k < H; ++k) sum += whh[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                            if (bih) sum += bih[static_cast<size_t>(h)];
                            if (bhh) sum += bhh[static_cast<size_t>(h)];
                            ht[static_cast<size_t>(h)] = std::tanh(sum);
                        }
                        std::copy(ht, ht + H, h_prev.begin());
                    }
                    return true;
                }

                if (layer.type_enum == LayerType::GRU) {
                    const bool use_bias = layer.use_bias;
                    const size_t Wih_sz = static_cast<size_t>(3 * H) * static_cast<size_t>(I);
                    const size_t Whh_sz = static_cast<size_t>(3 * H) * static_cast<size_t>(H);
                    const size_t bih_sz = use_bias ? static_cast<size_t>(3 * H) : 0ULL;
                    const size_t bhh_sz = use_bias ? static_cast<size_t>(3 * H) : 0ULL;
                    const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                    if (layer.getWeightsSize() < need) return false;

                    const float* Wih = W;
                    const float* Whh = Wih + Wih_sz;
                    const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                    const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                    std::vector<float> h_prev(static_cast<size_t>(H), 0.0f);
                    std::vector<float> r(static_cast<size_t>(H), 0.0f);
                    std::vector<float> z(static_cast<size_t>(H), 0.0f);
                    std::vector<float> n(static_cast<size_t>(H), 0.0f);

                    outputs.resize(1);
                    outputs[0].assign(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                    for (int t = 0; t < T; ++t) {
                        const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];

                        for (int h = 0; h < H; ++h) {
                            const float* w_ir = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                            const float* w_hr = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                            float sr = 0.0f;
                            for (int i = 0; i < I; ++i) sr += w_ir[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                            for (int k = 0; k < H; ++k) sr += w_hr[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                            if (bih) sr += bih[static_cast<size_t>(h)];
                            if (bhh) sr += bhh[static_cast<size_t>(h)];
                            r[static_cast<size_t>(h)] = sigmoid_scalar(sr);

                            const float* w_iz = Wih + static_cast<size_t>(H + h) * static_cast<size_t>(I);
                            const float* w_hz = Whh + static_cast<size_t>(H + h) * static_cast<size_t>(H);
                            float sz = 0.0f;
                            for (int i = 0; i < I; ++i) sz += w_iz[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                            for (int k = 0; k < H; ++k) sz += w_hz[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                            if (bih) sz += bih[static_cast<size_t>(H + h)];
                            if (bhh) sz += bhh[static_cast<size_t>(H + h)];
                            z[static_cast<size_t>(h)] = sigmoid_scalar(sz);

                            const float* w_in = Wih + static_cast<size_t>(2 * H + h) * static_cast<size_t>(I);
                            const float* w_hn = Whh + static_cast<size_t>(2 * H + h) * static_cast<size_t>(H);
                            float sn = 0.0f;
                            for (int i = 0; i < I; ++i) sn += w_in[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                            for (int k = 0; k < H; ++k) sn += w_hn[static_cast<size_t>(k)] * (r[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)]);
                            if (bih) sn += bih[static_cast<size_t>(2 * H + h)];
                            if (bhh) sn += bhh[static_cast<size_t>(2 * H + h)];
                            n[static_cast<size_t>(h)] = std::tanh(sn);
                        }

                        float* ht = &outputs[0][static_cast<size_t>(t) * static_cast<size_t>(H)];
                        for (int h = 0; h < H; ++h) {
                            const float zh = z[static_cast<size_t>(h)];
                            const float nh = n[static_cast<size_t>(h)];
                            const float hp = h_prev[static_cast<size_t>(h)];
                            ht[static_cast<size_t>(h)] = (1.0f - zh) * nh + zh * hp;
                        }
                        std::copy(ht, ht + H, h_prev.begin());
                    }
                    return true;
                }

                // LSTM
                {
                    const bool use_bias = layer.use_bias;
                    const size_t Wih_sz = static_cast<size_t>(4 * H) * static_cast<size_t>(I);
                    const size_t Whh_sz = static_cast<size_t>(4 * H) * static_cast<size_t>(H);
                    const size_t bih_sz = use_bias ? static_cast<size_t>(4 * H) : 0ULL;
                    const size_t bhh_sz = use_bias ? static_cast<size_t>(4 * H) : 0ULL;
                    const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                    if (layer.getWeightsSize() < need) return false;

                    const float* Wih = W;
                    const float* Whh = Wih + Wih_sz;
                    const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                    const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                    std::vector<float> h_prev(static_cast<size_t>(H), 0.0f);
                    std::vector<float> c_prev(static_cast<size_t>(H), 0.0f);
                    outputs.resize(1);
                    outputs[0].assign(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);

                    for (int t = 0; t < T; ++t) {
                        const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                        float* ht = &outputs[0][static_cast<size_t>(t) * static_cast<size_t>(H)];

                        for (int h = 0; h < H; ++h) {
                            auto dot_in = [&](const float* wrow) -> float {
                                float s = 0.0f;
                                for (int i = 0; i < I; ++i) s += wrow[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                                return s;
                            };
                            auto dot_h = [&](const float* wrow) -> float {
                                float s = 0.0f;
                                for (int k = 0; k < H; ++k) s += wrow[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                                return s;
                            };

                            const float* wii = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                            const float* wif = Wih + static_cast<size_t>(H + h) * static_cast<size_t>(I);
                            const float* wig = Wih + static_cast<size_t>(2 * H + h) * static_cast<size_t>(I);
                            const float* wio = Wih + static_cast<size_t>(3 * H + h) * static_cast<size_t>(I);

                            const float* whi = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                            const float* whf = Whh + static_cast<size_t>(H + h) * static_cast<size_t>(H);
                            const float* whg = Whh + static_cast<size_t>(2 * H + h) * static_cast<size_t>(H);
                            const float* who = Whh + static_cast<size_t>(3 * H + h) * static_cast<size_t>(H);

                            float si = dot_in(wii) + dot_h(whi);
                            float sf = dot_in(wif) + dot_h(whf);
                            float sg = dot_in(wig) + dot_h(whg);
                            float so = dot_in(wio) + dot_h(who);
                            if (bih) {
                                si += bih[static_cast<size_t>(h)];
                                sf += bih[static_cast<size_t>(H + h)];
                                sg += bih[static_cast<size_t>(2 * H + h)];
                                so += bih[static_cast<size_t>(3 * H + h)];
                            }
                            if (bhh) {
                                si += bhh[static_cast<size_t>(h)];
                                sf += bhh[static_cast<size_t>(H + h)];
                                sg += bhh[static_cast<size_t>(2 * H + h)];
                                so += bhh[static_cast<size_t>(3 * H + h)];
                            }

                            const float i_gate = sigmoid_scalar(si);
                            const float f_gate = sigmoid_scalar(sf);
                            const float g_gate = std::tanh(sg);
                            const float o_gate = sigmoid_scalar(so);

                            const float c = f_gate * c_prev[static_cast<size_t>(h)] + i_gate * g_gate;
                            c_prev[static_cast<size_t>(h)] = c;
                            ht[static_cast<size_t>(h)] = o_gate * std::tanh(c);
                        }

                        std::copy(ht, ht + H, h_prev.begin());
                    }
                }
                return true;
            }

            case LayerType::UNKNOWN:
                return false;

            default:
                break;
        }

        // Si un layer n'est pas pris en charge ici, on renvoie false.
        return false;
    } catch (...) {
        return false;
    }
}

inline void accumulate_packed_bias(Layer& layer, size_t index, float gradient) {
    layer.grad_bias[index] += gradient;
    if (layer.grad_weights.size() >= layer.grad_bias.size())
        layer.grad_weights[layer.grad_weights.size() - layer.grad_bias.size() + index] += gradient;
}

inline bool cpu_backward_layer(
    const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs,
    std::vector<std::vector<float>>& grad_inputs,
    Layer& layer,
    bool training
) {
    (void)training;
    grad_inputs.clear();

    if ((layer.type_enum != LayerType::Constant) && (inputs.empty() || inputs[0] == nullptr)) {
        return false;
    }
    if (grad_outputs.empty() || grad_outputs[0] == nullptr) {
        return false;
    }

    try {
        auto sigmoid_scalar = [](float v) -> float {
            return 1.0f / (1.0f + std::exp(-v));
        };

        auto gelu_tanh_grad = [](float x) -> float {
            const float c = std::sqrt(2.0f / 3.14159265358979323846f);
            const float x2 = x * x;
            const float x3 = x2 * x;
            const float u = c * (x + 0.044715f * x3);
            const float t = std::tanh(u);
            const float sech2 = 1.0f - t * t;
            const float du = c * (1.0f + 3.0f * 0.044715f * x2);
            return 0.5f * (1.0f + t) + 0.5f * x * sech2 * du;
        };

        auto passthrough_grad = [](const std::vector<float>& x,
                                   const std::vector<float>& go,
                                   std::vector<std::vector<float>>& grad_inputs_ref) -> bool {
            if (go.size() != x.size()) return false;
            grad_inputs_ref.resize(1);
            grad_inputs_ref[0] = go;
            return true;
        };

        switch (layer.type_enum) {
            case LayerType::NMS: {
                // Hard NMS returns discrete indices. Backward explicitly stops
                // gradients; no differentiable selection surrogate is used.
                if (inputs.size() < 2 || inputs.size() > 3 || grad_outputs.size() != 1) return false;
                std::vector<std::vector<float>> selected;
                if (!cpu_forward_layer(inputs, selected, layer, false) ||
                    grad_outputs[0]->size() != selected[0].size()) return false;
                grad_inputs.resize(inputs.size());
                for (size_t i = 0; i < inputs.size(); ++i) {
                    if (inputs[i]) grad_inputs[i].assign(inputs[i]->size(), 0.0f);
                }
                return true;
            }

            case LayerType::Identity: {
                return passthrough_grad(*inputs[0], *grad_outputs[0], grad_inputs);
            }

            case LayerType::Flatten: {
                return passthrough_grad(*inputs[0], *grad_outputs[0], grad_inputs);
            }

            case LayerType::Reshape: {
                return passthrough_grad(*inputs[0], *grad_outputs[0], grad_inputs);
            }

            case LayerType::View: {
                return passthrough_grad(*inputs[0], *grad_outputs[0], grad_inputs);
            }

            case LayerType::Squeeze: {
                return passthrough_grad(*inputs[0], *grad_outputs[0], grad_inputs);
            }

            case LayerType::Unsqueeze: {
                return passthrough_grad(*inputs[0], *grad_outputs[0], grad_inputs);
            }

            case LayerType::Linear: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];

                const int in_f = layer.in_features;
                const int out_f = layer.out_features;
                const int batch = (layer.seq_len > 0)
                    ? layer.seq_len
                    : (in_f > 0 ? static_cast<int>(x.size()) / in_f : 0);
                if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;

                const size_t x_elems = static_cast<size_t>(batch) * static_cast<size_t>(in_f);
                const size_t go_elems = static_cast<size_t>(batch) * static_cast<size_t>(out_f);
                const size_t w_elems = static_cast<size_t>(out_f) * static_cast<size_t>(in_f);
                if (x.size() != x_elems || go.size() != go_elems) return false;

                const float* w = layer.getWeights();
                if (!w || layer.getWeightsSize() < w_elems) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x_elems, 0.0f);

                // Tile the contiguous feature dimension. The old b/i/o and
                // o/i/b loops read a different cache line for almost every
                // multiply. Each lane still reduces in the original o/b order.
                constexpr int tile = 64;
                const long long work = static_cast<long long>(batch) * in_f * out_f;
                #pragma omp parallel for schedule(static) if(work > 262144)
                for (int b = 0; b < batch; ++b) {
                    const float* row_go = go.data() + static_cast<size_t>(b) * out_f;
                    float* row_gx = grad_inputs[0].data() + static_cast<size_t>(b) * in_f;
                    for (int base = 0; base < in_f; base += tile) {
                        alignas(64) float sums[tile] = {};
                        const int count = std::min(tile, in_f - base);
                        for (int o = 0; o < out_f; ++o) {
                            const float g = row_go[o];
                            const float* row_w = w + static_cast<size_t>(o) * in_f + base;
                            #pragma omp simd
                            for (int i = 0; i < count; ++i) sums[i] += g * row_w[i];
                        }
                        std::copy_n(sums, count, row_gx + base);
                    }
                }

                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }
                #pragma omp parallel for schedule(static) if(work > 262144)
                for (int o = 0; o < out_f; ++o) {
                    float* row_gw = layer.grad_weights.data() + static_cast<size_t>(o) * in_f;
                    for (int base = 0; base < in_f; base += tile) {
                        alignas(64) float sums[tile] = {};
                        const int count = std::min(tile, in_f - base);
                        for (int b = 0; b < batch; ++b) {
                            const float g = go[static_cast<size_t>(b) * out_f + o];
                            const float* row_x = x.data() + static_cast<size_t>(b) * in_f + base;
                            #pragma omp simd
                            for (int i = 0; i < count; ++i) sums[i] += g * row_x[i];
                        }
                        #pragma omp simd
                        for (int i = 0; i < count; ++i) row_gw[base + i] += sums[i];
                    }
                }

                if (layer.use_bias && layer.getWeightsSize() >= w_elems + static_cast<size_t>(out_f)) {
                    if (layer.grad_bias.size() != static_cast<size_t>(out_f)) {
                        layer.grad_bias.assign(static_cast<size_t>(out_f), 0.0f);
                    }
                    for (int b = 0; b < batch; ++b) {
                        for (int o = 0; o < out_f; ++o) {
                            accumulate_packed_bias(layer, static_cast<size_t>(o), go[static_cast<size_t>(b) * static_cast<size_t>(out_f) + static_cast<size_t>(o)]);
                        }
                    }
                }
                return true;
            }

            case LayerType::Embedding: {
                const std::vector<float>& ids_f = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int vocab = std::max(1, layer.vocab_size);
                const int dim = std::max(1, layer.embed_dim > 0 ? layer.embed_dim : layer.out_features);
                const int pad = layer.padding_idx;
                if (go.size() != ids_f.size() * static_cast<size_t>(dim)) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(ids_f.size(), 0.0f);

                const size_t w_need = static_cast<size_t>(vocab) * static_cast<size_t>(dim);
                if (layer.getWeightsSize() < w_need) return false;
                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }

                for (size_t t = 0; t < ids_f.size(); ++t) {
                    const int id = static_cast<int>(std::llround(static_cast<double>(ids_f[t])));
                    if ((pad >= 0 && id == pad) || id < 0 || id >= vocab) continue;
                    const size_t w0 = static_cast<size_t>(id) * static_cast<size_t>(dim);
                    const size_t g0 = t * static_cast<size_t>(dim);
                    for (int d = 0; d < dim; ++d) {
                        layer.grad_weights[w0 + static_cast<size_t>(d)] += go[g0 + static_cast<size_t>(d)];
                    }
                }
                return true;
            }

            case LayerType::EmbeddingBag: {
                const std::vector<float>& ids_f = *inputs[0];
                const std::vector<float>* offsets_f = (inputs.size() >= 2) ? inputs[1] : nullptr;
                const std::vector<float>& go = *grad_outputs[0];

                const int vocab = std::max(1, layer.vocab_size);
                const int dim = std::max(1, layer.embed_dim);
                const int pad = layer.padding_idx;

                std::vector<int> offsets;
                int num_bags = 1;
                if (offsets_f && !offsets_f->empty()) {
                    offsets.reserve(offsets_f->size());
                    for (float v : *offsets_f) offsets.push_back(static_cast<int>(std::llround(static_cast<double>(v))));
                    if (offsets.size() < 2) return false;
                    num_bags = static_cast<int>(offsets.size()) - 1;
                } else {
                    offsets = {0, static_cast<int>(ids_f.size())};
                    num_bags = 1;
                }
                if (go.size() != static_cast<size_t>(num_bags) * static_cast<size_t>(dim)) return false;

                grad_inputs.resize(offsets_f ? 2 : 1);
                grad_inputs[0].assign(ids_f.size(), 0.0f);
                if (offsets_f) grad_inputs[1].assign(offsets_f->size(), 0.0f);

                const size_t w_need = static_cast<size_t>(vocab) * static_cast<size_t>(dim);
                if (layer.getWeightsSize() < w_need) return false;
                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }

                for (int b = 0; b < num_bags; ++b) {
                    const int start = std::clamp(offsets[static_cast<size_t>(b)], 0, static_cast<int>(ids_f.size()));
                    const int end = std::clamp(offsets[static_cast<size_t>(b + 1)], start, static_cast<int>(ids_f.size()));
                    const size_t g0 = static_cast<size_t>(b) * static_cast<size_t>(dim);
                    for (int t = start; t < end; ++t) {
                        const int id = static_cast<int>(std::llround(static_cast<double>(ids_f[static_cast<size_t>(t)])));
                        if ((pad >= 0 && id == pad) || id < 0 || id >= vocab) continue;
                        const size_t w0 = static_cast<size_t>(id) * static_cast<size_t>(dim);
                        for (int d = 0; d < dim; ++d) {
                            layer.grad_weights[w0 + static_cast<size_t>(d)] += go[g0 + static_cast<size_t>(d)];
                        }
                    }
                }
                return true;
            }

            case LayerType::Bilinear: {
                if (inputs.size() < 2 || !inputs[1]) return false;
                const std::vector<float>& x1 = *inputs[0];
                const std::vector<float>& x2 = *inputs[1];
                const std::vector<float>& go = *grad_outputs[0];

                const int in1 = layer.in_features;
                const int in2 = layer.out_features;
                const int out = (layer.embed_dim > 0) ? layer.embed_dim : (!layer.target_shape.empty() ? layer.target_shape[0] : 0);
                if (in1 <= 0 || in2 <= 0 || out <= 0) return false;

                const int batch = static_cast<int>(x1.size()) / in1;
                if (batch <= 0) return false;
                if (static_cast<int>(x1.size()) != batch * in1) return false;
                if (static_cast<int>(x2.size()) != batch * in2) return false;
                if (static_cast<int>(go.size()) != batch * out) return false;

                const float* w = layer.getWeights();
                if (!w) return false;
                const size_t w_sz = static_cast<size_t>(out) * static_cast<size_t>(in1) * static_cast<size_t>(in2);
                if (layer.getWeightsSize() < w_sz) return false;

                grad_inputs.resize(2);
                grad_inputs[0].assign(x1.size(), 0.0f);
                grad_inputs[1].assign(x2.size(), 0.0f);

                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }

                for (int n = 0; n < batch; ++n) {
                    const float* a = &x1[static_cast<size_t>(n) * static_cast<size_t>(in1)];
                    const float* b = &x2[static_cast<size_t>(n) * static_cast<size_t>(in2)];
                    const float* g = &go[static_cast<size_t>(n) * static_cast<size_t>(out)];

                    for (int o = 0; o < out; ++o) {
                        const float go_o = g[static_cast<size_t>(o)];
                        const float* Wo = w + static_cast<size_t>(o) * static_cast<size_t>(in1) * static_cast<size_t>(in2);
                        for (int i = 0; i < in1; ++i) {
                            const float xi = a[static_cast<size_t>(i)];
                            const float* Woi = Wo + static_cast<size_t>(i) * static_cast<size_t>(in2);
                            for (int j = 0; j < in2; ++j) {
                                const float wv = Woi[static_cast<size_t>(j)];
                                grad_inputs[0][static_cast<size_t>(n) * static_cast<size_t>(in1) + static_cast<size_t>(i)] += go_o * wv * b[static_cast<size_t>(j)];
                                grad_inputs[1][static_cast<size_t>(n) * static_cast<size_t>(in2) + static_cast<size_t>(j)] += go_o * wv * xi;
                                layer.grad_weights[static_cast<size_t>(o) * static_cast<size_t>(in1) * static_cast<size_t>(in2)
                                    + static_cast<size_t>(i) * static_cast<size_t>(in2)
                                    + static_cast<size_t>(j)] += go_o * xi * b[static_cast<size_t>(j)];
                            }
                        }
                    }
                }

                if (layer.use_bias && layer.getWeightsSize() >= w_sz + static_cast<size_t>(out)) {
                    if (layer.grad_bias.size() != static_cast<size_t>(out)) {
                        layer.grad_bias.assign(static_cast<size_t>(out), 0.0f);
                    }
                    for (int n = 0; n < batch; ++n) {
                        for (int o = 0; o < out; ++o) {
                            accumulate_packed_bias(layer, static_cast<size_t>(o), go[static_cast<size_t>(n) * static_cast<size_t>(out) + static_cast<size_t>(o)]);
                        }
                    }
                }
                return true;
            }

            case LayerType::MatMul:
            case LayerType::BatchMatMul: {
                if (inputs.size() < 2 || !inputs[1]) return false;
                const std::vector<float>& A = *inputs[0];
                const std::vector<float>& B = *inputs[1];
                const std::vector<float>& go = *grad_outputs[0];

                const int batches = (layer.type_enum == LayerType::BatchMatMul) ? layer.seq_len : 1;
                const int M = layer.in_features;
                const int K = layer.out_features;
                const int N = layer.embed_dim;
                if (batches <= 0 || M <= 0 || K <= 0 || N <= 0) return false;

                const size_t a_batch = static_cast<size_t>(M) * static_cast<size_t>(K);
                const size_t b_batch = static_cast<size_t>(K) * static_cast<size_t>(N);
                const size_t c_batch = static_cast<size_t>(M) * static_cast<size_t>(N);
                const size_t a_elems = static_cast<size_t>(batches) * a_batch;
                const size_t b_elems = static_cast<size_t>(batches) * b_batch;
                const size_t c_elems = static_cast<size_t>(batches) * c_batch;
                if (A.size() != a_elems || B.size() != b_elems || go.size() != c_elems) return false;

                grad_inputs.resize(2);
                grad_inputs[0].assign(a_elems, 0.0f);
                grad_inputs[1].assign(b_elems, 0.0f);

                for (int bi = 0; bi < batches; ++bi) {
                    const size_t a0 = static_cast<size_t>(bi) * a_batch;
                    const size_t b0 = static_cast<size_t>(bi) * b_batch;
                    const size_t c0 = static_cast<size_t>(bi) * c_batch;

                    RuntimeMatMul::backward(A.data() + a0, B.data() + b0,
                        go.data() + c0, grad_inputs[0].data() + a0,
                        grad_inputs[1].data() + b0, M, K, N);
                }
                return true;
            }

            case LayerType::Add: {
                if (inputs.size() != 2 || !inputs[1]) return false;
                const size_t a = inputs[0]->size(), b = inputs[1]->size();
                const auto& go = *grad_outputs[0];
                const size_t large = std::max(a,b), small = std::min(a,b);
                if (!small || go.size() != large || large % small != 0) return false;
                grad_inputs = {std::vector<float>(a,0.0f), std::vector<float>(b,0.0f)};
                for (size_t i = 0; i < large; ++i) {
                    grad_inputs[0][i % a] += go[i];
                    grad_inputs[1][i % b] += go[i];
                }
                return true;
            }

            case LayerType::Subtract: {
                if (inputs.size() < 2 || !inputs[1]) return false;
                const std::vector<float>& a = *inputs[0];
                const std::vector<float>& b = *inputs[1];
                const std::vector<float>& go = *grad_outputs[0];
                if (a.size() != b.size() || go.size() != a.size()) return false;

                grad_inputs.resize(2);
                grad_inputs[0].assign(a.size(), 0.0f);
                grad_inputs[1].assign(a.size(), 0.0f);
                for (size_t i = 0; i < a.size(); ++i) {
                    const float g = go[i];
                    grad_inputs[0][i] = g;
                    grad_inputs[1][i] = -g;
                }
                return true;
            }

            case LayerType::Multiply: {
                if (inputs.size() < 2 || !inputs[1]) return false;
                const std::vector<float>& a = *inputs[0];
                const std::vector<float>& b = *inputs[1];
                const std::vector<float>& go = *grad_outputs[0];
                if (a.size() != b.size() || go.size() != a.size()) return false;

                grad_inputs.resize(2);
                grad_inputs[0].assign(a.size(), 0.0f);
                grad_inputs[1].assign(a.size(), 0.0f);
                for (size_t i = 0; i < a.size(); ++i) {
                    const float g = go[i];
                    grad_inputs[0][i] = g * b[i];
                    grad_inputs[1][i] = g * a[i];
                }
                return true;
            }

            case LayerType::Divide: {
                if (inputs.size() < 2 || !inputs[1]) return false;
                const std::vector<float>& a = *inputs[0];
                const std::vector<float>& b = *inputs[1];
                const std::vector<float>& go = *grad_outputs[0];
                if (a.size() != b.size() || go.size() != a.size()) return false;

                grad_inputs.resize(2);
                grad_inputs[0].assign(a.size(), 0.0f);
                grad_inputs[1].assign(a.size(), 0.0f);

                const float eps = 1e-8f;
                for (size_t i = 0; i < a.size(); ++i) {
                    const float g = go[i];
                    const float ai = a[i];
                    const float bi = b[i];
                    const float d = (std::abs(bi) < eps) ? (bi >= 0.0f ? eps : -eps) : bi;
                    grad_inputs[0][i] = g / d;
                    grad_inputs[1][i] = -g * ai / (d * d);
                }
                return true;
            }

            case LayerType::ReLU: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                for (size_t i = 0; i < x.size(); ++i) {
                    grad_inputs[0][i] = go[i] * ((x[i] > 0.0f) ? 1.0f : 0.0f);
                }
                return true;
            }

            case LayerType::LeakyReLU: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;
                const float alpha = layer.leaky_relu_alpha > 0.0f ? layer.leaky_relu_alpha : 0.01f;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (size_t i = 0; i < x.size(); ++i) {
                    grad_inputs[0][i] = go[i] * ((x[i] > 0.0f) ? 1.0f : alpha);
                }
                return true;
            }

            case LayerType::Sigmoid: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (size_t i = 0; i < x.size(); ++i) {
                    const float s = sigmoid_scalar(x[i]);
                    grad_inputs[0][i] = go[i] * s * (1.0f - s);
                }
                return true;
            }

            case LayerType::Tanh: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (size_t i = 0; i < x.size(); ++i) {
                    const float t = std::tanh(x[i]);
                    grad_inputs[0][i] = go[i] * (1.0f - t * t);
                }
                return true;
            }

            case LayerType::SiLU: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (size_t i = 0; i < x.size(); ++i) {
                    const float s = sigmoid_scalar(x[i]);
                    grad_inputs[0][i] = go[i] * (s + x[i] * s * (1.0f - s));
                }
                return true;
            }

            case LayerType::GELU: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (size_t i = 0; i < x.size(); ++i) {
                    grad_inputs[0][i] = go[i] * gelu_tanh_grad(x[i]);
                }
                return true;
            }

            case LayerType::GEGLU: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int seq = layer.seq_len;
                const int hid = layer.out_features;
                if (seq <= 0 || hid <= 0) return false;
                if (x.size() != static_cast<size_t>(seq) * static_cast<size_t>(hid) * 2ULL) return false;
                if (go.size() != static_cast<size_t>(seq) * static_cast<size_t>(hid)) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                const float c = std::sqrt(2.0f / 3.14159265358979323846f);
                for (int t = 0; t < seq; ++t) {
                    const size_t xb = static_cast<size_t>(t) * static_cast<size_t>(hid) * 2ULL;
                    const size_t gb = static_cast<size_t>(t) * static_cast<size_t>(hid);
                    for (int i = 0; i < hid; ++i) {
                        const float a = x[xb + static_cast<size_t>(i)];
                        const float z = x[xb + static_cast<size_t>(hid) + static_cast<size_t>(i)];
                        const float g = go[gb + static_cast<size_t>(i)];

                        const float z2 = z * z;
                        const float z3 = z2 * z;
                        const float u = c * (z + 0.044715f * z3);
                        const float th = std::tanh(u);
                        const float gelu = z * 0.5f * (1.0f + th);
                        const float dgelu = 0.5f * (1.0f + th) + 0.5f * z * (1.0f - th * th) * c * (1.0f + 3.0f * 0.044715f * z2);

                        grad_inputs[0][xb + static_cast<size_t>(i)] = g * gelu;
                        grad_inputs[0][xb + static_cast<size_t>(hid) + static_cast<size_t>(i)] = g * a * dgelu;
                    }
                }
                return true;
            }

            case LayerType::Softplus: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (size_t i = 0; i < x.size(); ++i) {
                    grad_inputs[0][i] = go[i] * sigmoid_scalar(x[i]);
                }
                return true;
            }

            case LayerType::Mish: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (size_t i = 0; i < x.size(); ++i) {
                    const float xi = x[i];
                    const float sp = xi > 20.0f ? xi : std::log1p(std::exp(xi));
                    const float tsp = std::tanh(sp);
                    const float sig = sigmoid_scalar(xi);
                    const float d = tsp + xi * (1.0f - tsp * tsp) * sig;
                    grad_inputs[0][i] = go[i] * d;
                }
                return true;
            }

            case LayerType::HardSigmoid: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (size_t i = 0; i < x.size(); ++i) {
                    const float d = (x[i] > -3.0f && x[i] < 3.0f) ? (1.0f / 6.0f) : 0.0f;
                    grad_inputs[0][i] = go[i] * d;
                }
                return true;
            }

            case LayerType::HardSwish: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (size_t i = 0; i < x.size(); ++i) {
                    float d;
                    if (x[i] <= -3.0f) d = 0.0f;
                    else if (x[i] >= 3.0f) d = 1.0f;
                    else d = x[i] / 3.0f + 0.5f;
                    grad_inputs[0][i] = go[i] * d;
                }
                return true;
            }

            case LayerType::Softmax: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                const std::vector<int> shape = LayerOps::infer_shape_for_axis_op(x, layer);
                if (shape.empty()) return false;
                int axis = layer.axis;
                if (axis < 0) axis += static_cast<int>(shape.size());
                if (axis < 0 || axis >= static_cast<int>(shape.size())) axis = static_cast<int>(shape.size()) - 1;

                size_t outer = 1;
                for (int i = 0; i < axis; ++i) outer *= static_cast<size_t>(shape[static_cast<size_t>(i)]);
                size_t axis_size = static_cast<size_t>(shape[static_cast<size_t>(axis)]);
                size_t inner = 1;
                for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) inner *= static_cast<size_t>(shape[i]);

                std::vector<float> y = LayerOps::softmax_forward(x, layer);
                if (y.size() != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                for (size_t o = 0; o < outer; ++o) {
                    for (size_t in = 0; in < inner; ++in) {
                        const size_t base = o * axis_size * inner + in;
                        float dot = 0.0f;
                        for (size_t a = 0; a < axis_size; ++a) {
                            dot += go[base + a * inner] * y[base + a * inner];
                        }
                        for (size_t a = 0; a < axis_size; ++a) {
                            const float yi = y[base + a * inner];
                            grad_inputs[0][base + a * inner] = yi * (go[base + a * inner] - dot);
                        }
                    }
                }
                return true;
            }

            case LayerType::LogSoftmax: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                const std::vector<int> shape = LayerOps::infer_shape_for_axis_op(x, layer);
                if (shape.empty()) return false;
                int axis = layer.axis;
                if (axis < 0) axis += static_cast<int>(shape.size());
                if (axis < 0 || axis >= static_cast<int>(shape.size())) axis = static_cast<int>(shape.size()) - 1;

                size_t outer = 1;
                for (int i = 0; i < axis; ++i) outer *= static_cast<size_t>(shape[static_cast<size_t>(i)]);
                size_t axis_size = static_cast<size_t>(shape[static_cast<size_t>(axis)]);
                size_t inner = 1;
                for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) inner *= static_cast<size_t>(shape[i]);

                std::vector<float> y = LayerOps::softmax_forward(x, layer);
                if (y.size() != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                for (size_t o = 0; o < outer; ++o) {
                    for (size_t in = 0; in < inner; ++in) {
                        const size_t base = o * axis_size * inner + in;
                        float s = 0.0f;
                        for (size_t a = 0; a < axis_size; ++a) s += go[base + a * inner];
                        for (size_t a = 0; a < axis_size; ++a) {
                            grad_inputs[0][base + a * inner] = go[base + a * inner] - y[base + a * inner] * s;
                        }
                    }
                }
                return true;
            }

            case LayerType::LayerNorm: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                const int N = static_cast<int>(x.size());
                const int norm = (layer.in_features > 0) ? layer.in_features : N;
                if (norm <= 0 || (N % norm) != 0) return false;
                const int groups = N / norm;
                const float eps = (layer.eps > 0.0f) ? layer.eps : 1e-5f;

                const float* w = (layer.affine ? layer.getWeights() : nullptr);
                const bool use_affine = layer.affine && w != nullptr;
                const bool use_bias = use_affine && layer.use_bias && layer.getWeightsSize() >= static_cast<size_t>(2 * norm);

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                if (use_affine) {
                    if (layer.grad_weights.size() != layer.getWeightsSize()) {
                        layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                    }
                    if (use_bias && layer.grad_bias.size() != static_cast<size_t>(norm)) {
                        layer.grad_bias.assign(static_cast<size_t>(norm), 0.0f);
                    }
                }

                for (int g = 0; g < groups; ++g) {
                    const int base = g * norm;

                    float mean = 0.0f;
                    for (int i = 0; i < norm; ++i) mean += x[static_cast<size_t>(base + i)];
                    mean /= static_cast<float>(norm);

                    float var = 0.0f;
                    for (int i = 0; i < norm; ++i) {
                        const float d = x[static_cast<size_t>(base + i)] - mean;
                        var += d * d;
                    }
                    var /= static_cast<float>(norm);
                    const float inv = 1.0f / std::sqrt(var + eps);

                    std::vector<float> xhat(static_cast<size_t>(norm), 0.0f);
                    std::vector<float> dxhat(static_cast<size_t>(norm), 0.0f);
                    float dvar = 0.0f;
                    float dmean = 0.0f;
                    for (int i = 0; i < norm; ++i) {
                        const float xi = x[static_cast<size_t>(base + i)];
                        const float xh = (xi - mean) * inv;
                        xhat[static_cast<size_t>(i)] = xh;
                        const float gamma = use_affine ? w[static_cast<size_t>(i)] : 1.0f;
                        const float dxi = go[static_cast<size_t>(base + i)] * gamma;
                        dxhat[static_cast<size_t>(i)] = dxi;
                        dvar += dxi * (xi - mean) * -0.5f * inv * inv * inv;
                    }
                    for (int i = 0; i < norm; ++i) {
                        dmean += dxhat[static_cast<size_t>(i)] * (-inv);
                    }
                    float sum_xm = 0.0f;
                    for (int i = 0; i < norm; ++i) sum_xm += x[static_cast<size_t>(base + i)] - mean;
                    dmean += dvar * (-2.0f * sum_xm / static_cast<float>(norm));

                    for (int i = 0; i < norm; ++i) {
                        const float xi = x[static_cast<size_t>(base + i)];
                        grad_inputs[0][static_cast<size_t>(base + i)] =
                            dxhat[static_cast<size_t>(i)] * inv +
                            dvar * 2.0f * (xi - mean) / static_cast<float>(norm) +
                            dmean / static_cast<float>(norm);
                    }

                    if (use_affine) {
                        for (int i = 0; i < norm; ++i) {
                            const float gi = go[static_cast<size_t>(base + i)];
                            layer.grad_weights[static_cast<size_t>(i)] += gi * xhat[static_cast<size_t>(i)];
                            if (use_bias) accumulate_packed_bias(layer, static_cast<size_t>(i), gi);
                        }
                    }
                }
                return true;
            }

            case LayerType::RMSNorm: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                const int N = static_cast<int>(x.size());
                const int norm = (layer.in_features > 0) ? layer.in_features : N;
                if (norm <= 0 || (N % norm) != 0) return false;
                const int groups = N / norm;
                const float eps = (layer.eps > 0.0f) ? layer.eps : 1e-5f;

                const float* w = (layer.affine ? layer.getWeights() : nullptr);
                const bool use_affine = layer.affine && w != nullptr;
                const bool use_bias = use_affine && layer.use_bias && layer.getWeightsSize() >= static_cast<size_t>(2 * norm);

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                if (use_affine) {
                    if (layer.grad_weights.size() != layer.getWeightsSize()) {
                        layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                    }
                    if (use_bias && layer.grad_bias.size() != static_cast<size_t>(norm)) {
                        layer.grad_bias.assign(static_cast<size_t>(norm), 0.0f);
                    }
                }

                for (int g = 0; g < groups; ++g) {
                    const int base = g * norm;
                    float ms = 0.0f;
                    for (int i = 0; i < norm; ++i) {
                        const float xi = x[static_cast<size_t>(base + i)];
                        ms += xi * xi;
                    }
                    ms /= static_cast<float>(norm);
                    const float inv = 1.0f / std::sqrt(ms + eps);

                    float sum_dxhat_x = 0.0f;
                    for (int i = 0; i < norm; ++i) {
                        const float gamma = use_affine ? w[static_cast<size_t>(i)] : 1.0f;
                        const float dxh = go[static_cast<size_t>(base + i)] * gamma;
                        sum_dxhat_x += dxh * x[static_cast<size_t>(base + i)];
                    }
                    const float coeff = inv * inv * inv / static_cast<float>(norm);
                    for (int i = 0; i < norm; ++i) {
                        const float xi = x[static_cast<size_t>(base + i)];
                        const float gamma = use_affine ? w[static_cast<size_t>(i)] : 1.0f;
                        const float dxh = go[static_cast<size_t>(base + i)] * gamma;
                        grad_inputs[0][static_cast<size_t>(base + i)] = dxh * inv - xi * coeff * sum_dxhat_x;
                    }

                    if (use_affine) {
                        for (int i = 0; i < norm; ++i) {
                            const float xh = x[static_cast<size_t>(base + i)] * inv;
                            const float gi = go[static_cast<size_t>(base + i)];
                            layer.grad_weights[static_cast<size_t>(i)] += gi * xh;
                            if (use_bias) accumulate_packed_bias(layer, static_cast<size_t>(i), gi);
                        }
                    }
                }
                return true;
            }

            case LayerType::GroupNorm: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                const int channels = layer.in_channels > 0 ? layer.in_channels : 1;
                const int height = layer.input_height > 0 ? layer.input_height : 1;
                const int width = layer.input_width > 0 ? layer.input_width : 1;
                const int spatial = std::max(1, height * width);
                if (channels <= 0 || spatial <= 0) return false;
                if ((x.size() % (static_cast<size_t>(channels) * static_cast<size_t>(spatial))) != 0ULL) return false;
                const int batch = static_cast<int>(x.size() / (static_cast<size_t>(channels) * static_cast<size_t>(spatial)));
                if (batch <= 0) return false;

                const int groups = std::max(1, std::min(layer.num_groups > 0 ? layer.num_groups : 1, channels));
                if ((channels % groups) != 0) return false;
                const int cpg = channels / groups;
                const int M = cpg * spatial;
                const float eps = layer.eps > 0.0f ? layer.eps : 1e-5f;

                const bool affine = layer.affine && (layer.getWeights() != nullptr) && (layer.getWeightsSize() >= static_cast<size_t>(channels));
                const float* w = affine ? layer.getWeights() : nullptr;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                if (affine) {
                    if (layer.grad_weights.size() != layer.getWeightsSize()) {
                        layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                    }
                    if (layer.use_bias && layer.grad_bias.size() != static_cast<size_t>(channels)) {
                        layer.grad_bias.assign(static_cast<size_t>(channels), 0.0f);
                    }
                }

                for (int n = 0; n < batch; ++n) {
                    for (int g = 0; g < groups; ++g) {
                        const int c0 = g * cpg;
                        float mean = 0.0f;
                        for (int c = 0; c < cpg; ++c) {
                            const int ch = c0 + c;
                            const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                            for (int s = 0; s < spatial; ++s) mean += x[base + static_cast<size_t>(s)];
                        }
                        mean /= static_cast<float>(M);

                        float var = 0.0f;
                        for (int c = 0; c < cpg; ++c) {
                            const int ch = c0 + c;
                            const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                            for (int s = 0; s < spatial; ++s) {
                                const float d = x[base + static_cast<size_t>(s)] - mean;
                                var += d * d;
                            }
                        }
                        var /= static_cast<float>(M);
                        const float invstd = 1.0f / std::sqrt(var + eps);

                        float sum_dxhat = 0.0f;
                        float sum_dxhat_xhat = 0.0f;
                        for (int c = 0; c < cpg; ++c) {
                            const int ch = c0 + c;
                            const float gamma = affine ? w[static_cast<size_t>(ch)] : 1.0f;
                            const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                            for (int s = 0; s < spatial; ++s) {
                                const size_t idx = base + static_cast<size_t>(s);
                                const float xhat = (x[idx] - mean) * invstd;
                                const float dxhat = go[idx] * gamma;
                                sum_dxhat += dxhat;
                                sum_dxhat_xhat += dxhat * xhat;
                            }
                        }

                        const float invM = 1.0f / static_cast<float>(M);
                        for (int c = 0; c < cpg; ++c) {
                            const int ch = c0 + c;
                            const float gamma = affine ? w[static_cast<size_t>(ch)] : 1.0f;
                            const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                            for (int s = 0; s < spatial; ++s) {
                                const size_t idx = base + static_cast<size_t>(s);
                                const float xhat = (x[idx] - mean) * invstd;
                                const float dxhat = go[idx] * gamma;
                                grad_inputs[0][idx] = invstd * invM * (static_cast<float>(M) * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
                            }
                        }
                    }
                }

                if (affine) {
                    for (int n = 0; n < batch; ++n) {
                        for (int g = 0; g < groups; ++g) {
                            const int c0 = g * cpg;
                            float mean = 0.0f;
                            for (int c = 0; c < cpg; ++c) {
                                const int ch = c0 + c;
                                const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                    + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                                for (int s = 0; s < spatial; ++s) mean += x[base + static_cast<size_t>(s)];
                            }
                            mean /= static_cast<float>(M);
                            float var = 0.0f;
                            for (int c = 0; c < cpg; ++c) {
                                const int ch = c0 + c;
                                const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                    + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                                for (int s = 0; s < spatial; ++s) {
                                    const float d = x[base + static_cast<size_t>(s)] - mean;
                                    var += d * d;
                                }
                            }
                            var /= static_cast<float>(M);
                            const float invstd = 1.0f / std::sqrt(var + eps);

                            for (int c = 0; c < cpg; ++c) {
                                const int ch = c0 + c;
                                const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                    + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                                for (int s = 0; s < spatial; ++s) {
                                    const size_t idx = base + static_cast<size_t>(s);
                                    const float xhat = (x[idx] - mean) * invstd;
                                    layer.grad_weights[static_cast<size_t>(ch)] += go[idx] * xhat;
                                    if (layer.use_bias && ch < static_cast<int>(layer.grad_bias.size())) {
                                        accumulate_packed_bias(layer, static_cast<size_t>(ch), go[idx]);
                                    }
                                }
                            }
                        }
                    }
                }
                return true;
            }

            case LayerType::InstanceNorm2d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                const int channels = layer.in_channels > 0 ? layer.in_channels : 1;
                const int height = layer.input_height > 0 ? layer.input_height : 1;
                const int width = layer.input_width > 0 ? layer.input_width : 1;
                const int spatial = std::max(1, height * width);
                if (channels <= 0 || spatial <= 0) return false;
                if ((x.size() % (static_cast<size_t>(channels) * static_cast<size_t>(spatial))) != 0ULL) return false;
                const int batch = static_cast<int>(x.size() / (static_cast<size_t>(channels) * static_cast<size_t>(spatial)));
                if (batch <= 0) return false;

                const int groups = channels;
                const int cpg = channels / groups;
                const int M = cpg * spatial;
                const float eps = layer.eps > 0.0f ? layer.eps : 1e-5f;

                const bool affine = layer.affine && (layer.getWeights() != nullptr) && (layer.getWeightsSize() >= static_cast<size_t>(channels));
                const float* w = affine ? layer.getWeights() : nullptr;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                if (affine) {
                    if (layer.grad_weights.size() != layer.getWeightsSize()) {
                        layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                    }
                    if (layer.use_bias && layer.grad_bias.size() != static_cast<size_t>(channels)) {
                        layer.grad_bias.assign(static_cast<size_t>(channels), 0.0f);
                    }
                }

                for (int n = 0; n < batch; ++n) {
                    for (int g = 0; g < groups; ++g) {
                        const int c0 = g * cpg;
                        float mean = 0.0f;
                        for (int c = 0; c < cpg; ++c) {
                            const int ch = c0 + c;
                            const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                            for (int s = 0; s < spatial; ++s) mean += x[base + static_cast<size_t>(s)];
                        }
                        mean /= static_cast<float>(M);

                        float var = 0.0f;
                        for (int c = 0; c < cpg; ++c) {
                            const int ch = c0 + c;
                            const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                            for (int s = 0; s < spatial; ++s) {
                                const float d = x[base + static_cast<size_t>(s)] - mean;
                                var += d * d;
                            }
                        }
                        var /= static_cast<float>(M);
                        const float invstd = 1.0f / std::sqrt(var + eps);

                        float sum_dxhat = 0.0f;
                        float sum_dxhat_xhat = 0.0f;
                        for (int c = 0; c < cpg; ++c) {
                            const int ch = c0 + c;
                            const float gamma = affine ? w[static_cast<size_t>(ch)] : 1.0f;
                            const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                            for (int s = 0; s < spatial; ++s) {
                                const size_t idx = base + static_cast<size_t>(s);
                                const float xhat = (x[idx] - mean) * invstd;
                                const float dxhat = go[idx] * gamma;
                                sum_dxhat += dxhat;
                                sum_dxhat_xhat += dxhat * xhat;
                            }
                        }

                        const float invM = 1.0f / static_cast<float>(M);
                        for (int c = 0; c < cpg; ++c) {
                            const int ch = c0 + c;
                            const float gamma = affine ? w[static_cast<size_t>(ch)] : 1.0f;
                            const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                            for (int s = 0; s < spatial; ++s) {
                                const size_t idx = base + static_cast<size_t>(s);
                                const float xhat = (x[idx] - mean) * invstd;
                                const float dxhat = go[idx] * gamma;
                                grad_inputs[0][idx] = invstd * invM * (static_cast<float>(M) * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
                            }
                        }
                    }
                }

                if (affine) {
                    for (int n = 0; n < batch; ++n) {
                        for (int g = 0; g < groups; ++g) {
                            const int c0 = g * cpg;
                            float mean = 0.0f;
                            for (int c = 0; c < cpg; ++c) {
                                const int ch = c0 + c;
                                const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                    + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                                for (int s = 0; s < spatial; ++s) mean += x[base + static_cast<size_t>(s)];
                            }
                            mean /= static_cast<float>(M);
                            float var = 0.0f;
                            for (int c = 0; c < cpg; ++c) {
                                const int ch = c0 + c;
                                const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                    + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                                for (int s = 0; s < spatial; ++s) {
                                    const float d = x[base + static_cast<size_t>(s)] - mean;
                                    var += d * d;
                                }
                            }
                            var /= static_cast<float>(M);
                            const float invstd = 1.0f / std::sqrt(var + eps);

                            for (int c = 0; c < cpg; ++c) {
                                const int ch = c0 + c;
                                const size_t base = static_cast<size_t>(n) * static_cast<size_t>(channels * spatial)
                                    + static_cast<size_t>(ch) * static_cast<size_t>(spatial);
                                for (int s = 0; s < spatial; ++s) {
                                    const size_t idx = base + static_cast<size_t>(s);
                                    const float xhat = (x[idx] - mean) * invstd;
                                    layer.grad_weights[static_cast<size_t>(ch)] += go[idx] * xhat;
                                    if (layer.use_bias && ch < static_cast<int>(layer.grad_bias.size())) {
                                        accumulate_packed_bias(layer, static_cast<size_t>(ch), go[idx]);
                                    }
                                }
                            }
                        }
                    }
                }
                return true;
            }

            case LayerType::Conv2d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& grad_out = *grad_outputs[0];

                const int kernel_size = layer.get_kernel_h();
                const int in_channels = layer.in_channels > 0 ? layer.in_channels : 1;
                const int out_channels = layer.out_channels > 0 ? layer.out_channels : 1;
                int height = layer.input_height > 0 ? layer.input_height : 0;
                int width = layer.input_width > 0 ? layer.input_width : 0;
                const int stride = layer.get_stride_h();
                const int padding = layer.get_pad_h();
                const int dilation = layer.dilation_h != 1 ? layer.dilation_h : layer.dilation;

                if (in_channels <= 0 || out_channels <= 0 || kernel_size <= 0) return false;

                if ((height <= 0 || width <= 0) && !x.empty() && (x.size() % static_cast<size_t>(in_channels)) == 0) {
                    const size_t hw = x.size() / static_cast<size_t>(in_channels);
                    const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                    if (s > 0 && s * s == hw) {
                        height = static_cast<int>(s);
                        width = static_cast<int>(s);
                    }
                }
                if (height <= 0 || width <= 0) return false;

                const int out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
                const int out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
                const int out_spatial = out_h * out_w;
                if (out_h <= 0 || out_w <= 0) return false;
                if (grad_out.size() != static_cast<size_t>(out_channels) * static_cast<size_t>(out_spatial)) return false;

                const size_t in_size = static_cast<size_t>(in_channels) * static_cast<size_t>(height) * static_cast<size_t>(width);
                if (x.size() != in_size) return false;

                const float* w = layer.getWeights();
                const size_t w_need = static_cast<size_t>(out_channels) * static_cast<size_t>(in_channels)
                    * static_cast<size_t>(kernel_size) * static_cast<size_t>(kernel_size);
                if (!w || layer.getWeightsSize() < w_need) return false;

                if (layer.use_bias && layer.getWeightsSize() < w_need + static_cast<size_t>(out_channels)) return false;
                grad_inputs.resize(1);
                CpuConvolution::backward(x,grad_out,grad_inputs[0],layer,height,width,in_channels,out_channels,kernel_size,stride,padding,dilation);
                return true;
            }

            case LayerType::Conv1d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& grad_out = *grad_outputs[0];
                const int in_channels = layer.in_channels > 0 ? layer.in_channels : 1;
                const int out_channels = layer.out_channels > 0 ? layer.out_channels : 1;
                const int kernel_size = layer.get_kernel_h();
                const int stride = layer.get_stride_h();
                const int padding = layer.get_pad_h();
                const int length = static_cast<int>(x.size()) / std::max(1, in_channels);
                const int out_length = (length + 2 * padding - kernel_size) / stride + 1;
                if (length <= 0 || out_length <= 0) return false;
                if (x.size() != static_cast<size_t>(in_channels) * static_cast<size_t>(length)) return false;
                if (grad_out.size() != static_cast<size_t>(out_channels) * static_cast<size_t>(out_length)) return false;

                const float* weights = layer.getWeights();
                const size_t w_main = static_cast<size_t>(out_channels) * static_cast<size_t>(in_channels) * static_cast<size_t>(kernel_size);
                if (!weights || layer.getWeightsSize() < w_main) return false;

                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }
                if (layer.use_bias && layer.grad_bias.size() != static_cast<size_t>(out_channels)) {
                    layer.grad_bias.assign(static_cast<size_t>(out_channels), 0.0f);
                }

                std::vector<float> grad_in(static_cast<size_t>(in_channels) * static_cast<size_t>(length), 0.0f);
                for (int oc = 0; oc < out_channels; ++oc) {
                    for (int ol = 0; ol < out_length; ++ol) {
                        const float g = grad_out[static_cast<size_t>(oc) * static_cast<size_t>(out_length) + static_cast<size_t>(ol)];
                        if (layer.use_bias && oc < static_cast<int>(layer.grad_bias.size())) accumulate_packed_bias(layer, static_cast<size_t>(oc), g);
                        for (int ic = 0; ic < in_channels; ++ic) {
                            for (int kk = 0; kk < kernel_size; ++kk) {
                                const int il = ol * stride + kk - padding;
                                if (il < 0 || il >= length) continue;
                                const size_t in_idx = static_cast<size_t>(ic) * static_cast<size_t>(length) + static_cast<size_t>(il);
                                const size_t w_idx = (static_cast<size_t>(oc) * static_cast<size_t>(in_channels) + static_cast<size_t>(ic)) * static_cast<size_t>(kernel_size) + static_cast<size_t>(kk);
                                layer.grad_weights[w_idx] += g * x[in_idx];
                                grad_in[in_idx] += g * weights[w_idx];
                            }
                        }
                    }
                }
                grad_inputs.resize(1);
                grad_inputs[0] = std::move(grad_in);
                return true;
            }

            case LayerType::DepthwiseConv2d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& grad_out = *grad_outputs[0];
                const int channels = layer.in_channels > 0 ? layer.in_channels : 0;
                int height = layer.input_height > 0 ? layer.input_height : 0;
                int width = layer.input_width > 0 ? layer.input_width : 0;
                const int kernel_size = layer.get_kernel_h();
                const int stride = layer.get_stride_h();
                const int padding = layer.get_pad_h();
                if (channels <= 0) return false;
                if ((height <= 0 || width <= 0) && !x.empty() && (x.size() % static_cast<size_t>(channels)) == 0) {
                    const size_t hw = x.size() / static_cast<size_t>(channels);
                    const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                    if (s > 0 && s * s == hw) { height = static_cast<int>(s); width = static_cast<int>(s); }
                }
                if (height <= 0 || width <= 0) return false;
                const int out_h = (height + 2 * padding - kernel_size) / stride + 1;
                const int out_w = (width + 2 * padding - kernel_size) / stride + 1;
                if (out_h <= 0 || out_w <= 0) return false;
                if (x.size() != static_cast<size_t>(channels) * static_cast<size_t>(height) * static_cast<size_t>(width)) return false;
                if (grad_out.size() != static_cast<size_t>(channels) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w)) return false;

                const float* weights = layer.getWeights();
                const size_t w_main = static_cast<size_t>(channels) * static_cast<size_t>(kernel_size) * static_cast<size_t>(kernel_size);
                if (!weights || layer.getWeightsSize() < w_main) return false;

                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }
                if (layer.use_bias && layer.grad_bias.size() != static_cast<size_t>(channels)) {
                    layer.grad_bias.assign(static_cast<size_t>(channels), 0.0f);
                }

                std::vector<float> grad_in(static_cast<size_t>(channels) * static_cast<size_t>(height) * static_cast<size_t>(width), 0.0f);
                for (int c = 0; c < channels; ++c) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            const float g = grad_out[static_cast<size_t>(c) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w)
                                + static_cast<size_t>(oh) * static_cast<size_t>(out_w) + static_cast<size_t>(ow)];
                            if (layer.use_bias && c < static_cast<int>(layer.grad_bias.size())) accumulate_packed_bias(layer, static_cast<size_t>(c), g);
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    const int ih = oh * stride + kh - padding;
                                    const int iw = ow * stride + kw - padding;
                                    if (ih < 0 || ih >= height || iw < 0 || iw >= width) continue;
                                    const size_t in_idx = static_cast<size_t>(c) * static_cast<size_t>(height) * static_cast<size_t>(width)
                                        + static_cast<size_t>(ih) * static_cast<size_t>(width) + static_cast<size_t>(iw);
                                    const size_t w_idx = static_cast<size_t>(c) * static_cast<size_t>(kernel_size) * static_cast<size_t>(kernel_size)
                                        + static_cast<size_t>(kh) * static_cast<size_t>(kernel_size) + static_cast<size_t>(kw);
                                    layer.grad_weights[w_idx] += g * x[in_idx];
                                    grad_in[in_idx] += g * weights[w_idx];
                                }
                            }
                        }
                    }
                }
                grad_inputs.resize(1);
                grad_inputs[0] = std::move(grad_in);
                return true;
            }

            case LayerType::ConvTranspose2d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& grad_out = *grad_outputs[0];
                const int in_c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int out_c = layer.out_channels > 0 ? layer.out_channels : 1;
                const int k = layer.get_kernel_h();
                const int stride = layer.get_stride_h();
                const int pad = layer.get_pad_h();
                int H = layer.input_height > 0 ? layer.input_height : 0;
                int W = layer.input_width > 0 ? layer.input_width : 0;
                if ((H <= 0 || W <= 0) && !x.empty() && (x.size() % static_cast<size_t>(in_c)) == 0) {
                    const size_t hw = x.size() / static_cast<size_t>(in_c);
                    const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                    if (s > 0 && s * s == hw) { H = static_cast<int>(s); W = static_cast<int>(s); }
                }
                if (H <= 0 || W <= 0 || k <= 0 || stride <= 0) return false;
                if (x.size() != static_cast<size_t>(in_c) * static_cast<size_t>(H) * static_cast<size_t>(W)) return false;

                const int out_h = (H - 1) * stride - 2 * pad + k;
                const int out_w = (W - 1) * stride - 2 * pad + k;
                if (out_h <= 0 || out_w <= 0) return false;
                if (grad_out.size() != static_cast<size_t>(out_c) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w)) return false;

                const float* w = layer.getWeights();
                const size_t w_main = static_cast<size_t>(out_c) * static_cast<size_t>(in_c) * static_cast<size_t>(k) * static_cast<size_t>(k);
                if (!w || layer.getWeightsSize() < w_main) return false;

                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }
                if (layer.use_bias && layer.grad_bias.size() != static_cast<size_t>(out_c)) {
                    layer.grad_bias.assign(static_cast<size_t>(out_c), 0.0f);
                }

                std::vector<float> grad_in(x.size(), 0.0f);

                for (int oc = 0; oc < out_c; ++oc) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            const float g = grad_out[static_cast<size_t>(oc) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w)
                                + static_cast<size_t>(oh) * static_cast<size_t>(out_w) + static_cast<size_t>(ow)];
                            if (layer.use_bias && oc < static_cast<int>(layer.grad_bias.size())) {
                                accumulate_packed_bias(layer, static_cast<size_t>(oc), g);
                            }
                        }
                    }
                }

                for (int oc = 0; oc < out_c; ++oc) {
                    for (int ic = 0; ic < in_c; ++ic) {
                        for (int ih = 0; ih < H; ++ih) {
                            for (int iw = 0; iw < W; ++iw) {
                                const float xv = x[(static_cast<size_t>(ic) * static_cast<size_t>(H) + static_cast<size_t>(ih)) * static_cast<size_t>(W) + static_cast<size_t>(iw)];
                                for (int kh = 0; kh < k; ++kh) {
                                    for (int kw = 0; kw < k; ++kw) {
                                        const int oh = ih * stride + kh - pad;
                                        const int ow = iw * stride + kw - pad;
                                        if (oh < 0 || oh >= out_h || ow < 0 || ow >= out_w) continue;
                                        const size_t go_idx = static_cast<size_t>(oc) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w)
                                            + static_cast<size_t>(oh) * static_cast<size_t>(out_w) + static_cast<size_t>(ow);
                                        const size_t w_idx = ((static_cast<size_t>(oc) * static_cast<size_t>(in_c) + static_cast<size_t>(ic)) * static_cast<size_t>(k) + static_cast<size_t>(kh)) * static_cast<size_t>(k) + static_cast<size_t>(kw);
                                        layer.grad_weights[w_idx] += grad_out[go_idx] * xv;
                                    }
                                }
                            }
                        }
                    }
                }

                for (int ic = 0; ic < in_c; ++ic) {
                    for (int ih = 0; ih < H; ++ih) {
                        for (int iw = 0; iw < W; ++iw) {
                            float sum = 0.0f;
                            for (int oc = 0; oc < out_c; ++oc) {
                                for (int kh = 0; kh < k; ++kh) {
                                    for (int kw = 0; kw < k; ++kw) {
                                        const int oh = ih * stride + kh - pad;
                                        const int ow = iw * stride + kw - pad;
                                        if (oh < 0 || oh >= out_h || ow < 0 || ow >= out_w) continue;
                                        const size_t go_idx = static_cast<size_t>(oc) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w)
                                            + static_cast<size_t>(oh) * static_cast<size_t>(out_w) + static_cast<size_t>(ow);
                                        const size_t w_idx = ((static_cast<size_t>(oc) * static_cast<size_t>(in_c) + static_cast<size_t>(ic)) * static_cast<size_t>(k) + static_cast<size_t>(kh)) * static_cast<size_t>(k) + static_cast<size_t>(kw);
                                        sum += grad_out[go_idx] * w[w_idx];
                                    }
                                }
                            }
                            grad_in[(static_cast<size_t>(ic) * static_cast<size_t>(H) + static_cast<size_t>(ih)) * static_cast<size_t>(W) + static_cast<size_t>(iw)] = sum;
                        }
                    }
                }

                grad_inputs.resize(1);
                grad_inputs[0] = std::move(grad_in);
                return true;
            }

            case LayerType::MaxPool2d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];

                const int kernel_h = layer.get_kernel_h();
                const int kernel_w = layer.get_kernel_w();
                const int stride_h = layer.get_stride_h();
                const int stride_w = layer.get_stride_w();
                const int pad_h = layer.get_pad_h();
                const int pad_w = layer.get_pad_w();
                const int c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int h = layer.input_height > 0 ? layer.input_height : 1;
                const int w = layer.input_width > 0 ? layer.input_width : 1;
                const int oh = (h + 2 * pad_h - kernel_h) / stride_h + 1;
                const int ow = (w + 2 * pad_w - kernel_w) / stride_w + 1;

                if (kernel_h <= 0 || kernel_w <= 0 || stride_h <= 0 || stride_w <= 0) return false;
                if (x.size() != static_cast<size_t>(c) * static_cast<size_t>(h) * static_cast<size_t>(w)) return false;
                if (go.size() != static_cast<size_t>(c) * static_cast<size_t>(oh) * static_cast<size_t>(ow)) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                for (int ch = 0; ch < c; ++ch) {
                    for (int y = 0; y < oh; ++y) {
                        for (int xw = 0; xw < ow; ++xw) {
                            int best_ih = -1;
                            int best_iw = -1;
                            float best_v = -std::numeric_limits<float>::infinity();
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    const int ih = y * stride_h + kh - pad_h;
                                    const int iw = xw * stride_w + kw - pad_w;
                                    if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                    const size_t idx = static_cast<size_t>(ch) * static_cast<size_t>(h * w)
                                                     + static_cast<size_t>(ih) * static_cast<size_t>(w)
                                                     + static_cast<size_t>(iw);
                                    const float v = x[idx];
                                    if (v > best_v) {
                                        best_v = v;
                                        best_ih = ih;
                                        best_iw = iw;
                                    }
                                }
                            }
                            if (best_ih >= 0 && best_iw >= 0) {
                                const size_t in_idx = static_cast<size_t>(ch) * static_cast<size_t>(h * w)
                                                    + static_cast<size_t>(best_ih) * static_cast<size_t>(w)
                                                    + static_cast<size_t>(best_iw);
                                const size_t out_idx = static_cast<size_t>(ch) * static_cast<size_t>(oh * ow)
                                                     + static_cast<size_t>(y) * static_cast<size_t>(ow)
                                                     + static_cast<size_t>(xw);
                                grad_inputs[0][in_idx] += go[out_idx];
                            }
                        }
                    }
                }
                return true;
            }

            case LayerType::AvgPool2d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];

                const int kernel_h = layer.get_kernel_h();
                const int kernel_w = layer.get_kernel_w();
                const int stride_h = layer.get_stride_h();
                const int stride_w = layer.get_stride_w();
                const int pad_h = layer.get_pad_h();
                const int pad_w = layer.get_pad_w();
                const int c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int h = layer.input_height > 0 ? layer.input_height : 1;
                const int w = layer.input_width > 0 ? layer.input_width : 1;
                const int oh = (h + 2 * pad_h - kernel_h) / stride_h + 1;
                const int ow = (w + 2 * pad_w - kernel_w) / stride_w + 1;

                if (kernel_h <= 0 || kernel_w <= 0 || stride_h <= 0 || stride_w <= 0) return false;
                if (x.size() != static_cast<size_t>(c) * static_cast<size_t>(h) * static_cast<size_t>(w)) return false;
                if (go.size() != static_cast<size_t>(c) * static_cast<size_t>(oh) * static_cast<size_t>(ow)) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                for (int ch = 0; ch < c; ++ch) {
                    for (int y = 0; y < oh; ++y) {
                        for (int xw = 0; xw < ow; ++xw) {
                            int count = 0;
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    const int ih = y * stride_h + kh - pad_h;
                                    const int iw = xw * stride_w + kw - pad_w;
                                    if (ih >= 0 && ih < h && iw >= 0 && iw < w) ++count;
                                }
                            }
                            if (count <= 0) continue;
                            const size_t out_idx = static_cast<size_t>(ch) * static_cast<size_t>(oh * ow)
                                                 + static_cast<size_t>(y) * static_cast<size_t>(ow)
                                                 + static_cast<size_t>(xw);
                            const float g = go[out_idx] / static_cast<float>(count);
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                for (int kw = 0; kw < kernel_w; ++kw) {
                                    const int ih = y * stride_h + kh - pad_h;
                                    const int iw = xw * stride_w + kw - pad_w;
                                    if (ih < 0 || ih >= h || iw < 0 || iw >= w) continue;
                                    const size_t in_idx = static_cast<size_t>(ch) * static_cast<size_t>(h * w)
                                                        + static_cast<size_t>(ih) * static_cast<size_t>(w)
                                                        + static_cast<size_t>(iw);
                                    grad_inputs[0][in_idx] += g;
                                }
                            }
                        }
                    }
                }
                return true;
            }

            case LayerType::GlobalAvgPool2d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                std::vector<float> gx = LayerOps::global_avgpool2d_backward(go, layer, x.size());
                if (gx.empty()) return false;
                grad_inputs.resize(1);
                grad_inputs[0] = std::move(gx);
                return true;
            }

            case LayerType::AdaptiveAvgPool2d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                std::vector<float> gx = LayerOps::global_avgpool2d_backward(go, layer, x.size());
                if (gx.empty()) return false;
                grad_inputs.resize(1);
                grad_inputs[0] = std::move(gx);
                return true;
            }

            case LayerType::MaxPool1d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int l = static_cast<int>(x.size()) / c;
                const int k = layer.kernel_h > 0 ? layer.kernel_h : 2;
                const int s = layer.stride_h > 0 ? layer.stride_h : k;
                const int p = layer.pad_h >= 0 ? layer.pad_h : 0;
                const int ol = (l + 2 * p - k) / s + 1;
                if (l <= 0 || k <= 0 || s <= 0) return false;
                if (static_cast<int>(x.size()) != c * l) return false;
                if (static_cast<int>(go.size()) != c * ol) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (int ch = 0; ch < c; ++ch) {
                    for (int o = 0; o < ol; ++o) {
                        const float g = go[static_cast<size_t>(ch) * static_cast<size_t>(ol) + static_cast<size_t>(o)];
                        int best = -1;
                        float best_v = -std::numeric_limits<float>::infinity();
                        for (int kk = 0; kk < k; ++kk) {
                            const int il = o * s + kk - p;
                            if (il < 0 || il >= l) continue;
                            const float v = x[static_cast<size_t>(ch) * static_cast<size_t>(l) + static_cast<size_t>(il)];
                            if (v > best_v) { best_v = v; best = il; }
                        }
                        if (best >= 0) {
                            grad_inputs[0][static_cast<size_t>(ch) * static_cast<size_t>(l) + static_cast<size_t>(best)] += g;
                        }
                    }
                }
                return true;
            }

            case LayerType::AvgPool1d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int l = static_cast<int>(x.size()) / c;
                const int k = layer.kernel_h > 0 ? layer.kernel_h : 2;
                const int s = layer.stride_h > 0 ? layer.stride_h : k;
                const int p = layer.pad_h >= 0 ? layer.pad_h : 0;
                const int ol = (l + 2 * p - k) / s + 1;
                if (l <= 0 || k <= 0 || s <= 0) return false;
                if (static_cast<int>(x.size()) != c * l) return false;
                if (static_cast<int>(go.size()) != c * ol) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (int ch = 0; ch < c; ++ch) {
                    for (int o = 0; o < ol; ++o) {
                        const float g = go[static_cast<size_t>(ch) * static_cast<size_t>(ol) + static_cast<size_t>(o)];
                        int count = 0;
                        for (int kk = 0; kk < k; ++kk) {
                            const int il = o * s + kk - p;
                            if (il >= 0 && il < l) ++count;
                        }
                        if (count <= 0) continue;
                        const float gv = g / static_cast<float>(count);
                        for (int kk = 0; kk < k; ++kk) {
                            const int il = o * s + kk - p;
                            if (il < 0 || il >= l) continue;
                            grad_inputs[0][static_cast<size_t>(ch) * static_cast<size_t>(l) + static_cast<size_t>(il)] += gv;
                        }
                    }
                }
                return true;
            }

            case LayerType::TokenMeanPool: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int seq = layer.seq_len > 0 ? layer.seq_len : 0;
                const int emb = layer.embed_dim > 0 ? layer.embed_dim : 0;
                if (seq <= 0 || emb <= 0) return false;
                if (x.size() != static_cast<size_t>(seq) * static_cast<size_t>(emb)) return false;
                if (go.size() != static_cast<size_t>(emb)) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                const float inv = 1.0f / static_cast<float>(seq);
                for (int t = 0; t < seq; ++t) {
                    for (int d = 0; d < emb; ++d) {
                        grad_inputs[0][static_cast<size_t>(t) * static_cast<size_t>(emb) + static_cast<size_t>(d)] =
                            go[static_cast<size_t>(d)] * inv;
                    }
                }
                return true;
            }

            case LayerType::Dropout: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                const float p = std::clamp(layer.dropout_p, 0.0f, 1.0f);
                if (!training || p == 0.0f) { grad_inputs={go}; return true; }
                if (p == 1.0f) { grad_inputs={std::vector<float>(go.size(),0.0f)}; return true; }
                const float scale = (p < 1.0f) ? (1.0f / (1.0f - p)) : 0.0f;

                grad_inputs.resize(1);
                grad_inputs[0].assign(go.size(), 0.0f);

                const std::vector<float>* y = (inputs.size() >= 2 && inputs[1] && inputs[1]->size() == go.size()) ? inputs[1] : nullptr;
                if (!y) {
                    // Fallback déterministe (inference/no-mask): identité.
                    return false;
                }

                for (size_t i = 0; i < go.size(); ++i) {
                    const bool kept = ((*y)[i] != 0.0f);
                    grad_inputs[0][i] = kept ? (go[i] * scale) : 0.0f;
                }
                return true;
            }

            case LayerType::Dropout2d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                const float p = std::clamp(layer.dropout_p, 0.0f, 1.0f);
                if (!training || p == 0.0f) { grad_inputs={go}; return true; }
                if (p == 1.0f) { grad_inputs={std::vector<float>(go.size(),0.0f)}; return true; }
                const float scale = (p < 1.0f) ? (1.0f / (1.0f - p)) : 0.0f;

                grad_inputs.resize(1);
                grad_inputs[0].assign(go.size(), 0.0f);

                const std::vector<float>* y = (inputs.size() >= 2 && inputs[1] && inputs[1]->size() == go.size()) ? inputs[1] : nullptr;
                if (!y) {
                    return false;
                }

                for (size_t i = 0; i < go.size(); ++i) {
                    const bool kept = ((*y)[i] != 0.0f);
                    grad_inputs[0][i] = kept ? (go[i] * scale) : 0.0f;
                }
                return true;
            }

            case LayerType::AlphaDropout: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;

                const float p = std::clamp(layer.dropout_p, 0.0f, 1.0f);
                if (!training || p == 0.0f) { grad_inputs={go}; return true; }
                if (p == 1.0f) { grad_inputs={std::vector<float>(go.size(),0.0f)}; return true; }
                const float alpha = 1.6732632423543772848170429916717f;
                const float scale_selu = 1.0507009873554804934193349852946f;
                const float alpha_p = -alpha * scale_selu;
                const float a = 1.0f / std::sqrt((1.0f - p) * (1.0f + p * alpha_p * alpha_p));
                const float b = -a * alpha_p * p;
                const float dropped_out = a * alpha_p + b;

                grad_inputs.resize(1);
                grad_inputs[0].assign(go.size(), 0.0f);

                const std::vector<float>* y = (inputs.size() >= 2 && inputs[1] && inputs[1]->size() == go.size()) ? inputs[1] : nullptr;
                if (!y) {
                    return false;
                }

                for (size_t i = 0; i < go.size(); ++i) {
                    const bool kept = std::fabs((*y)[i] - dropped_out) > 1e-6f;
                    grad_inputs[0][i] = kept ? (go[i] * a) : 0.0f;
                }
                return true;
            }

            case LayerType::UpsampleNearest: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int in_h = layer.input_height > 0 ? layer.input_height : layer.out_h;
                const int in_w = layer.input_width > 0 ? layer.input_width : layer.out_w;
                if (in_h <= 0 || in_w <= 0) return false;
                const int out_h = layer.output_height > 0 ? layer.output_height : (layer.out_h > 0 ? layer.out_h : 0);
                const int out_w = layer.output_width > 0 ? layer.output_width : (layer.out_w > 0 ? layer.out_w : 0);
                const int sh = layer.scale_h > 0 ? static_cast<int>(std::lround(layer.scale_h)) : ((out_h > 0) ? std::max(1, out_h / in_h) : 2);
                const int sw = layer.scale_w > 0 ? static_cast<int>(std::lround(layer.scale_w)) : ((out_w > 0) ? std::max(1, out_w / in_w) : 2);
                const int oh = in_h * sh;
                const int ow = in_w * sw;
                if (x.size() != static_cast<size_t>(c) * static_cast<size_t>(in_h) * static_cast<size_t>(in_w)) return false;
                if (go.size() != static_cast<size_t>(c) * static_cast<size_t>(oh) * static_cast<size_t>(ow)) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (int ch = 0; ch < c; ++ch) {
                    for (int y = 0; y < oh; ++y) {
                        const int iy = y / sh;
                        for (int xw = 0; xw < ow; ++xw) {
                            const int ix = xw / sw;
                            const size_t out_idx = static_cast<size_t>(ch) * static_cast<size_t>(oh * ow)
                                                 + static_cast<size_t>(y) * static_cast<size_t>(ow)
                                                 + static_cast<size_t>(xw);
                            const size_t in_idx = static_cast<size_t>(ch) * static_cast<size_t>(in_h * in_w)
                                                + static_cast<size_t>(iy) * static_cast<size_t>(in_w)
                                                + static_cast<size_t>(ix);
                            grad_inputs[0][in_idx] += go[out_idx];
                        }
                    }
                }
                return true;
            }

            case LayerType::UpsampleBilinear: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int in_h = layer.input_height > 0 ? layer.input_height : layer.out_h;
                const int in_w = layer.input_width > 0 ? layer.input_width : layer.out_w;
                if (in_h <= 0 || in_w <= 0 || c <= 0) return false;
                const int out_h = layer.output_height > 0 ? layer.output_height : (layer.out_h > 0 ? layer.out_h : std::max(1, static_cast<int>(std::lround(static_cast<float>(in_h) * std::max(0.0f, layer.scale_h)))));
                const int out_w = layer.output_width > 0 ? layer.output_width : (layer.out_w > 0 ? layer.out_w : std::max(1, static_cast<int>(std::lround(static_cast<float>(in_w) * std::max(0.0f, layer.scale_w)))));
                if (out_h <= 0 || out_w <= 0) return false;
                if (x.size() != static_cast<size_t>(c) * static_cast<size_t>(in_h) * static_cast<size_t>(in_w)) return false;
                if (go.size() != static_cast<size_t>(c) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w)) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                const float scale_h = static_cast<float>(in_h) / static_cast<float>(out_h);
                const float scale_w = static_cast<float>(in_w) / static_cast<float>(out_w);
                for (int ch = 0; ch < c; ++ch) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            const float ih_f = static_cast<float>(oh) * scale_h;
                            const float iw_f = static_cast<float>(ow) * scale_w;
                            const int ih0 = static_cast<int>(std::floor(ih_f));
                            const int iw0 = static_cast<int>(std::floor(iw_f));
                            const int ih1 = std::min(ih0 + 1, in_h - 1);
                            const int iw1 = std::min(iw0 + 1, in_w - 1);
                            const float dh = ih_f - static_cast<float>(ih0);
                            const float dw = iw_f - static_cast<float>(iw0);
                            const float g = go[(static_cast<size_t>(ch) * static_cast<size_t>(out_h) + static_cast<size_t>(oh)) * static_cast<size_t>(out_w) + static_cast<size_t>(ow)];

                            const size_t b = static_cast<size_t>(ch) * static_cast<size_t>(in_h) * static_cast<size_t>(in_w);
                            grad_inputs[0][b + static_cast<size_t>(ih0) * static_cast<size_t>(in_w) + static_cast<size_t>(iw0)] += g * (1.0f - dh) * (1.0f - dw);
                            grad_inputs[0][b + static_cast<size_t>(ih0) * static_cast<size_t>(in_w) + static_cast<size_t>(iw1)] += g * (1.0f - dh) * dw;
                            grad_inputs[0][b + static_cast<size_t>(ih1) * static_cast<size_t>(in_w) + static_cast<size_t>(iw0)] += g * dh * (1.0f - dw);
                            grad_inputs[0][b + static_cast<size_t>(ih1) * static_cast<size_t>(in_w) + static_cast<size_t>(iw1)] += g * dh * dw;
                        }
                    }
                }
                return true;
            }

            case LayerType::UpsampleBicubic: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];

                const int c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int in_h = layer.input_height > 0 ? layer.input_height : layer.out_h;
                const int in_w = layer.input_width > 0 ? layer.input_width : layer.out_w;
                if (in_h <= 0 || in_w <= 0 || c <= 0) return false;
                const int out_h = layer.output_height > 0 ? layer.output_height : (layer.out_h > 0 ? layer.out_h : std::max(1, static_cast<int>(std::lround(static_cast<float>(in_h) * std::max(0.0f, layer.scale_h)))));
                const int out_w = layer.output_width > 0 ? layer.output_width : (layer.out_w > 0 ? layer.out_w : std::max(1, static_cast<int>(std::lround(static_cast<float>(in_w) * std::max(0.0f, layer.scale_w)))));
                if (out_h <= 0 || out_w <= 0) return false;
                if (x.size() != static_cast<size_t>(c) * static_cast<size_t>(in_h) * static_cast<size_t>(in_w)) return false;
                if (go.size() != static_cast<size_t>(c) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w)) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                auto cubic_weight = [](float xx) -> float {
                    constexpr float a = -0.75f;
                    xx = std::abs(xx);
                    if (xx <= 1.0f) return ((a + 2.0f) * xx - (a + 3.0f)) * xx * xx + 1.0f;
                    if (xx < 2.0f) return (((a * xx - 5.0f * a) * xx + 8.0f * a) * xx) - 4.0f * a;
                    return 0.0f;
                };

                const float scale_h = static_cast<float>(in_h) / static_cast<float>(out_h);
                const float scale_w = static_cast<float>(in_w) / static_cast<float>(out_w);
                for (int ch = 0; ch < c; ++ch) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            const float ih_f = (static_cast<float>(oh) + 0.5f) * scale_h - 0.5f;
                            const float iw_f = (static_cast<float>(ow) + 0.5f) * scale_w - 0.5f;
                            const int ih_base = static_cast<int>(std::floor(ih_f));
                            const int iw_base = static_cast<int>(std::floor(iw_f));
                            const float g = go[(static_cast<size_t>(ch) * static_cast<size_t>(out_h) + static_cast<size_t>(oh)) * static_cast<size_t>(out_w) + static_cast<size_t>(ow)];

                            float wsum = 0.0f;
                            float ws[4][4]{};
                            int ihs[4]{};
                            int iws[4]{};
                            for (int mh = -1; mh <= 2; ++mh) {
                                const int ridx = mh + 1;
                                ihs[ridx] = std::max(0, std::min(ih_base + mh, in_h - 1));
                                const float wh = cubic_weight(ih_f - static_cast<float>(ih_base + mh));
                                for (int mw = -1; mw <= 2; ++mw) {
                                    const int cidx = mw + 1;
                                    iws[cidx] = std::max(0, std::min(iw_base + mw, in_w - 1));
                                    const float ww = cubic_weight(iw_f - static_cast<float>(iw_base + mw));
                                    ws[ridx][cidx] = wh * ww;
                                    wsum += ws[ridx][cidx];
                                }
                            }
                            if (wsum == 0.0f) continue;
                            const size_t base = static_cast<size_t>(ch) * static_cast<size_t>(in_h) * static_cast<size_t>(in_w);
                            for (int mh = 0; mh < 4; ++mh) {
                                for (int mw = 0; mw < 4; ++mw) {
                                    const size_t idx = base + static_cast<size_t>(ihs[mh]) * static_cast<size_t>(in_w) + static_cast<size_t>(iws[mw]);
                                    grad_inputs[0][idx] += g * (ws[mh][mw] / wsum);
                                }
                            }
                        }
                    }
                }
                return true;
            }

            case LayerType::PixelShuffle: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int r = layer.scale_h > 0 ? static_cast<int>(layer.scale_h) : 2;
                const int in_channels = layer.in_channels > 0 ? layer.in_channels : 0;
                const int in_h = layer.input_height > 0 ? layer.input_height : 0;
                const int in_w = layer.input_width > 0 ? layer.input_width : 0;
                if (r <= 0 || static_cast<float>(r) != layer.scale_h ||
                    in_channels <= 0 || in_h <= 0 || in_w <= 0) return false;
                const int out_channels = in_channels / (r * r);
                const int out_h = in_h * r;
                const int out_w = in_w * r;
                if (out_channels <= 0 || (in_channels % (r * r)) != 0) return false;
                if (x.size() != static_cast<size_t>(in_channels) * static_cast<size_t>(in_h) * static_cast<size_t>(in_w)) return false;
                if (go.size() != static_cast<size_t>(out_channels) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w)) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                const size_t in_hw = static_cast<size_t>(in_h) * static_cast<size_t>(in_w);
                for (int oc = 0; oc < out_channels; ++oc) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            const int ih = oh / r;
                            const int iw = ow / r;
                            const int sub_h = oh - ih * r;
                            const int sub_w = ow - iw * r;
                            const int ic = oc * r * r + sub_h * r + sub_w;
                            const size_t in_idx = static_cast<size_t>(ic) * in_hw + static_cast<size_t>(ih) * static_cast<size_t>(in_w) + static_cast<size_t>(iw);
                            const size_t out_idx = (static_cast<size_t>(oc) * static_cast<size_t>(out_h) + static_cast<size_t>(oh)) * static_cast<size_t>(out_w) + static_cast<size_t>(ow);
                            grad_inputs[0][in_idx] += go[out_idx];
                        }
                    }
                }
                return true;
            }

            case LayerType::ZeroPad2d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int in_h = layer.input_height > 0 ? layer.input_height : 1;
                const int in_w = layer.input_width > 0 ? layer.input_width : 1;
                const int ph = layer.pad_h >= 0 ? layer.pad_h : 1;
                const int pw = layer.pad_w >= 0 ? layer.pad_w : ph;
                const int out_h = in_h + 2 * ph;
                const int out_w = in_w + 2 * pw;
                if (x.size() != static_cast<size_t>(c) * static_cast<size_t>(in_h) * static_cast<size_t>(in_w)) return false;
                if (go.size() != static_cast<size_t>(c) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w)) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (int ch = 0; ch < c; ++ch) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            int ih = oh - ph;
                            int iw = ow - pw;
                            if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) continue;

                            const size_t out_idx = static_cast<size_t>(ch) * static_cast<size_t>(out_h * out_w)
                                                 + static_cast<size_t>(oh) * static_cast<size_t>(out_w)
                                                 + static_cast<size_t>(ow);
                            const size_t in_idx = static_cast<size_t>(ch) * static_cast<size_t>(in_h * in_w)
                                                + static_cast<size_t>(ih) * static_cast<size_t>(in_w)
                                                + static_cast<size_t>(iw);
                            grad_inputs[0][in_idx] += go[out_idx];
                        }
                    }
                }
                return true;
            }

            case LayerType::ReflectionPad2d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int in_h = layer.input_height > 0 ? layer.input_height : 1;
                const int in_w = layer.input_width > 0 ? layer.input_width : 1;
                const int ph = layer.pad_h >= 0 ? layer.pad_h : 1;
                const int pw = layer.pad_w >= 0 ? layer.pad_w : ph;
                const int out_h = in_h + 2 * ph;
                const int out_w = in_w + 2 * pw;
                if (x.size() != static_cast<size_t>(c) * static_cast<size_t>(in_h) * static_cast<size_t>(in_w)) return false;
                if (go.size() != static_cast<size_t>(c) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w)) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (int ch = 0; ch < c; ++ch) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            int ih = oh - ph;
                            int iw = ow - pw;
                            if (ih < 0) ih = -ih;
                            if (ih >= in_h) ih = 2 * in_h - ih - 2;
                            if (iw < 0) iw = -iw;
                            if (iw >= in_w) iw = 2 * in_w - iw - 2;
                            ih = std::max(0, std::min(ih, in_h - 1));
                            iw = std::max(0, std::min(iw, in_w - 1));

                            const size_t out_idx = static_cast<size_t>(ch) * static_cast<size_t>(out_h * out_w)
                                                 + static_cast<size_t>(oh) * static_cast<size_t>(out_w)
                                                 + static_cast<size_t>(ow);
                            const size_t in_idx = static_cast<size_t>(ch) * static_cast<size_t>(in_h * in_w)
                                                + static_cast<size_t>(ih) * static_cast<size_t>(in_w)
                                                + static_cast<size_t>(iw);
                            grad_inputs[0][in_idx] += go[out_idx];
                        }
                    }
                }
                return true;
            }

            case LayerType::ReplicationPad2d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int c = layer.in_channels > 0 ? layer.in_channels : 1;
                const int in_h = layer.input_height > 0 ? layer.input_height : 1;
                const int in_w = layer.input_width > 0 ? layer.input_width : 1;
                const int ph = layer.pad_h >= 0 ? layer.pad_h : 1;
                const int pw = layer.pad_w >= 0 ? layer.pad_w : ph;
                const int out_h = in_h + 2 * ph;
                const int out_w = in_w + 2 * pw;
                if (x.size() != static_cast<size_t>(c) * static_cast<size_t>(in_h) * static_cast<size_t>(in_w)) return false;
                if (go.size() != static_cast<size_t>(c) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w)) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                for (int ch = 0; ch < c; ++ch) {
                    for (int oh = 0; oh < out_h; ++oh) {
                        for (int ow = 0; ow < out_w; ++ow) {
                            int ih = oh - ph;
                            int iw = ow - pw;
                            ih = std::max(0, std::min(ih, in_h - 1));
                            iw = std::max(0, std::min(iw, in_w - 1));

                            const size_t out_idx = static_cast<size_t>(ch) * static_cast<size_t>(out_h * out_w)
                                                 + static_cast<size_t>(oh) * static_cast<size_t>(out_w)
                                                 + static_cast<size_t>(ow);
                            const size_t in_idx = static_cast<size_t>(ch) * static_cast<size_t>(in_h * in_w)
                                                + static_cast<size_t>(ih) * static_cast<size_t>(in_w)
                                                + static_cast<size_t>(iw);
                            grad_inputs[0][in_idx] += go[out_idx];
                        }
                    }
                }
                return true;
            }

            case LayerType::Transpose: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int rows = layer.in_features;
                const int cols = layer.out_features;
                if (rows <= 0 || cols <= 0) return false;
                if (x.size() != static_cast<size_t>(rows) * static_cast<size_t>(cols)) return false;
                if (go.size() != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].assign(go.size(), 0.0f);
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        grad_inputs[0][static_cast<size_t>(i) * static_cast<size_t>(cols) + static_cast<size_t>(j)] =
                            go[static_cast<size_t>(j) * static_cast<size_t>(rows) + static_cast<size_t>(i)];
                    }
                }
                return true;
            }

            case LayerType::Permute: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                if (go.size() != x.size()) return false;
                if (layer.permute_dims.empty()) return false;

                std::vector<int> in_shape = layer.shape;
                if (in_shape.empty()) {
                    in_shape = {1, static_cast<int>(x.size())};
                }
                if (in_shape.empty()) return false;

                std::vector<int> out_shape(in_shape.size(), 0);
                for (size_t i = 0; i < layer.permute_dims.size(); ++i) {
                    const int d = layer.permute_dims[i];
                    if (d < 0 || d >= static_cast<int>(in_shape.size())) return false;
                    out_shape[i] = in_shape[static_cast<size_t>(d)];
                }

                std::vector<int> inv(layer.permute_dims.size(), 0);
                for (size_t i = 0; i < layer.permute_dims.size(); ++i) {
                    inv[static_cast<size_t>(layer.permute_dims[i])] = static_cast<int>(i);
                }

                grad_inputs.resize(1);
                grad_inputs[0] = LayerOps::permute_forward(go, inv, out_shape);
                return grad_inputs[0].size() == x.size();
            }

            case LayerType::Concat: {
                if (inputs.size() < 2) return false;
                const std::vector<float>& go = *grad_outputs[0];
                size_t total = 0;
                for (const auto* p : inputs) {
                    if (!p) return false;
                    total += p->size();
                }
                if (go.size() != total) return false;

                grad_inputs.resize(inputs.size());
                size_t off = 0;
                for (size_t i = 0; i < inputs.size(); ++i) {
                    const size_t n = inputs[i]->size();
                    grad_inputs[i].assign(go.begin() + static_cast<std::ptrdiff_t>(off), go.begin() + static_cast<std::ptrdiff_t>(off + n));
                    off += n;
                }
                return true;
            }

            case LayerType::Split: {
                const std::vector<float>& x = *inputs[0];
                size_t total = 0;
                for (const auto* g : grad_outputs) {
                    if (!g) return false;
                    total += g->size();
                }
                if (total != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].clear();
                grad_inputs[0].reserve(total);
                for (const auto* g : grad_outputs) {
                    grad_inputs[0].insert(grad_inputs[0].end(), g->begin(), g->end());
                }
                return true;
            }

            case LayerType::Chunk: {
                const std::vector<float>& x = *inputs[0];
                size_t total = 0;
                for (const auto* g : grad_outputs) {
                    if (!g) return false;
                    total += g->size();
                }
                if (total != x.size()) return false;

                grad_inputs.resize(1);
                grad_inputs[0].clear();
                grad_inputs[0].reserve(total);
                for (const auto* g : grad_outputs) {
                    grad_inputs[0].insert(grad_inputs[0].end(), g->begin(), g->end());
                }
                return true;
            }

            case LayerType::Stack: {
                const auto& go = *grad_outputs[0];
                size_t total = 0;
                for (auto* input : inputs) { if (!input) return false; total += input->size(); }
                if (go.size() != total) return false;
                grad_inputs.resize(inputs.size());
                size_t offset = 0;
                for (size_t i = 0; i < inputs.size(); ++i) {
                    grad_inputs[i].assign(go.begin() + offset, go.begin() + offset + inputs[i]->size());
                    offset += inputs[i]->size();
                }
                return true;
            }

            case LayerType::BatchNorm1d:
            case LayerType::BatchNorm2d: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& grad_out = *grad_outputs[0];
                if (x.size() != grad_out.size()) return false;

                int channels = layer.in_channels > 0 ? layer.in_channels : layer.out_channels;
                if (channels <= 0 && !layer.running_mean.empty()) channels = static_cast<int>(layer.running_mean.size());
                if (channels <= 0 && layer.affine && layer.getWeights()) {
                    const size_t w_sz = layer.getWeightsSize();
                    channels = static_cast<int>((layer.use_bias && (w_sz % 2ULL) == 0ULL) ? (w_sz / 2ULL) : w_sz);
                }
                const int total = static_cast<int>(x.size());
                if (channels <= 0 || (total % channels) != 0) return false;

                int spatial = 1;
                if (layer.type_enum == LayerType::BatchNorm2d && layer.input_height > 0 && layer.input_width > 0) {
                    spatial = layer.input_height * layer.input_width;
                }
                if (spatial <= 0 || (total % (channels * spatial)) != 0) {
                    spatial = (total % channels == 0) ? (total / channels) : 1;
                }
                const int batch = total / (channels * spatial);
                const int N = batch * spatial;
                if (batch <= 0 || N <= 0) return false;

                const float eps = layer.eps > 0.0f ? layer.eps : 1e-5f;
                const float* layer_weights = layer.getWeights();
                const bool affine = layer.affine;
                if (affine && layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                auto idx_at = [&](int b, int c, int s) -> size_t {
                    return static_cast<size_t>(b) * static_cast<size_t>(channels) * static_cast<size_t>(spatial)
                        + static_cast<size_t>(c) * static_cast<size_t>(spatial)
                        + static_cast<size_t>(s);
                };

                for (int c = 0; c < channels; ++c) {
                    float mean = 0.0f;
                    for (int b = 0; b < batch; ++b) {
                        for (int s = 0; s < spatial; ++s) {
                            mean += x[idx_at(b, c, s)];
                        }
                    }
                    mean /= static_cast<float>(N);

                    float var = 0.0f;
                    for (int b = 0; b < batch; ++b) {
                        for (int s = 0; s < spatial; ++s) {
                            const float d = x[idx_at(b, c, s)] - mean;
                            var += d * d;
                        }
                    }
                    var /= static_cast<float>(N);
                    const float invstd = 1.0f / std::sqrt(var + eps);

                    const float gamma = (affine && layer_weights && c < static_cast<int>(layer.getWeightsSize()))
                        ? layer_weights[static_cast<size_t>(c)] : 1.0f;

                    float grad_gamma = 0.0f;
                    float grad_beta = 0.0f;
                    float sum_dxhat = 0.0f;
                    float sum_dxhat_xhat = 0.0f;
                    for (int b = 0; b < batch; ++b) {
                        for (int s = 0; s < spatial; ++s) {
                            const size_t i = idx_at(b, c, s);
                            const float xhat = (x[i] - mean) * invstd;
                            const float dy = grad_out[i];
                            const float dxhat = dy * gamma;
                            grad_gamma += dy * xhat;
                            grad_beta += dy;
                            sum_dxhat += dxhat;
                            sum_dxhat_xhat += dxhat * xhat;
                        }
                    }

                    if (affine && c < static_cast<int>(layer.grad_weights.size())) {
                        layer.grad_weights[static_cast<size_t>(c)] += grad_gamma;
                        if (layer.use_bias && (channels + c) < static_cast<int>(layer.grad_weights.size())) {
                            layer.grad_weights[static_cast<size_t>(channels + c)] += grad_beta;
                        }
                    }

                    const float invN = 1.0f / static_cast<float>(N);
                    for (int b = 0; b < batch; ++b) {
                        for (int s = 0; s < spatial; ++s) {
                            const size_t i = idx_at(b, c, s);
                            const float xhat = (x[i] - mean) * invstd;
                            const float dxhat = grad_out[i] * gamma;
                            grad_inputs[0][i] = invstd * invN * (static_cast<float>(N) * dxhat - sum_dxhat - xhat * sum_dxhat_xhat);
                        }
                    }
                }
                return true;
            }

            case LayerType::Reparameterize: {
                if (inputs.size() < 2 || !inputs[1]) return false;
                const std::vector<float>& mu = *inputs[0];
                const std::vector<float>& logvar = *inputs[1];
                const std::vector<float>& go = *grad_outputs[0];
                if (mu.size() != logvar.size() || go.size() != mu.size()) return false;

                // Backward exact si z (ou epsilon) est fourni en input[2], sinon fallback déterministe (dz/dlogvar=0).
                grad_inputs.resize(2);
                grad_inputs[0] = go;
                grad_inputs[1].assign(mu.size(), 0.0f);

                if (inputs.size() >= 3 && inputs[2] && inputs[2]->size() == mu.size()) {
                    const std::vector<float>& z_or_eps = *inputs[2];
                    for (size_t i = 0; i < mu.size(); ++i) {
                        const float lv = std::clamp(logvar[i], -20.0f, 20.0f);
                        const float stdv = std::exp(0.5f * lv);
                        const float inv_std = (stdv > 1e-12f) ? (1.0f / stdv) : 0.0f;
                        const float eps_recon = (z_or_eps[i] - mu[i]) * inv_std;
                        grad_inputs[1][i] = (logvar[i] >= -20.0f && logvar[i] <= 20.0f) ? go[i] * 0.5f * stdv * eps_recon : 0.0f;
                    }
                    return true;
                }

                return true;
            }

            case LayerType::Constant: {
                if(!SequenceConstant::backward(layer,*grad_outputs[0]))return false;
                grad_inputs.clear();
                return true;
            }

            case LayerType::PatchEmbed: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];

                const int d_model = layer.embed_dim > 0 ? layer.embed_dim : layer.out_features;
                const int seq_text = std::max(1, layer.seq_text);
                const int num_patches = std::max(1, layer.num_patches);
                const int patch_dim = std::max(1, layer.patch_dim);
                if (d_model <= 0) return false;

                const int text_dim = (seq_text + 1) * d_model;
                const int in_dim = text_dim + num_patches * patch_dim;
                const int out_dim = (seq_text + 1 + num_patches) * d_model;
                if (static_cast<int>(x.size()) != in_dim || static_cast<int>(go.size()) != out_dim) return false;

                const float* w = layer.getWeights();
                if (!w || static_cast<int>(layer.getWeightsSize()) < (patch_dim * d_model + d_model)) return false;
                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }

                const float inv = 1.0f / std::sqrt(static_cast<float>(patch_dim));
                std::vector<float> grad_in(static_cast<size_t>(in_dim), 0.0f);

                for (int i = 0; i < text_dim; ++i) grad_in[static_cast<size_t>(i)] = go[static_cast<size_t>(i)];

                float* gradW = layer.grad_weights.data();
                float* gradB = layer.grad_weights.data() + static_cast<size_t>(patch_dim) * static_cast<size_t>(d_model);

                for (int p = 0; p < num_patches; ++p) {
                    const int in_off = text_dim + p * patch_dim;
                    const int out_off = (seq_text + 1 + p) * d_model;

                    for (int d = 0; d < d_model; ++d) {
                        gradB[d] += go[static_cast<size_t>(out_off + d)];
                    }

                    for (int k = 0; k < patch_dim; ++k) {
                        const float xk = x[static_cast<size_t>(in_off + k)] * inv;
                        const size_t row = static_cast<size_t>(k) * static_cast<size_t>(d_model);
                        for (int d = 0; d < d_model; ++d) {
                            gradW[row + static_cast<size_t>(d)] += xk * go[static_cast<size_t>(out_off + d)];
                        }
                    }

                    for (int k = 0; k < patch_dim; ++k) {
                        float acc = 0.0f;
                        const size_t row = static_cast<size_t>(k) * static_cast<size_t>(d_model);
                        for (int d = 0; d < d_model; ++d) {
                            acc += w[row + static_cast<size_t>(d)] * go[static_cast<size_t>(out_off + d)];
                        }
                        grad_in[static_cast<size_t>(in_off + k)] += acc * inv;
                    }
                }

                grad_inputs.resize(1);
                grad_inputs[0] = std::move(grad_in);
                return true;
            }

            case LayerType::SelfAttention: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& grad_out = *grad_outputs[0];
                const int seq_len = layer.seq_len > 0 ? layer.seq_len : 1;
                const int embed_dim = layer.embed_dim > 0 ? layer.embed_dim : static_cast<int>(x.size());
                const int num_heads = layer.num_heads > 0 ? layer.num_heads : 1;
                const bool causal = layer.causal;
                if (seq_len <= 0 || embed_dim <= 0 || num_heads <= 0) return false;
                if ((embed_dim % num_heads) != 0) return false;
                if (static_cast<int>(x.size()) != seq_len * embed_dim) return false;
                if (static_cast<int>(grad_out.size()) != seq_len * embed_dim) return false;

                const float* weights = layer.getWeights();
                if (!weights) return false;
                const int qkv_size = embed_dim * embed_dim * 3;
                const int out_size = embed_dim * embed_dim;
                if (layer.getWeightsSize() < static_cast<size_t>(qkv_size + out_size)) return false;

                const float* Wqkv = weights;
                const float* Wout = weights + qkv_size;
                const int head_dim = embed_dim / num_heads;
                const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

                std::vector<float> qkv(static_cast<size_t>(seq_len) * static_cast<size_t>(3 * embed_dim), 0.0f);
                for (int i = 0; i < seq_len; ++i) {
                    const float* xrow = &x[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
                    float* yrow = &qkv[static_cast<size_t>(i) * static_cast<size_t>(3 * embed_dim)];
                    for (int n = 0; n < 3 * embed_dim; ++n) {
                        float sum = 0.0f;
                        for (int k = 0; k < embed_dim; ++k) {
                            sum += xrow[static_cast<size_t>(k)] *
                                   Wqkv[static_cast<size_t>(n) * static_cast<size_t>(embed_dim) +
                                         static_cast<size_t>(k)];
                        }
                        yrow[static_cast<size_t>(n)] = sum;
                    }
                }

                auto q_at = [&](int t, int d) -> float {
                    return qkv[static_cast<size_t>(t) * static_cast<size_t>(3 * embed_dim) + static_cast<size_t>(d)];
                };
                auto k_at = [&](int t, int d) -> float {
                    return qkv[static_cast<size_t>(t) * static_cast<size_t>(3 * embed_dim) + static_cast<size_t>(embed_dim + d)];
                };
                auto v_at = [&](int t, int d) -> float {
                    return qkv[static_cast<size_t>(t) * static_cast<size_t>(3 * embed_dim) + static_cast<size_t>(2 * embed_dim + d)];
                };

                std::vector<float> P(static_cast<size_t>(num_heads) * static_cast<size_t>(seq_len) * static_cast<size_t>(seq_len), 0.0f);
                std::vector<float> attended(static_cast<size_t>(seq_len) * static_cast<size_t>(embed_dim), 0.0f);

                for (int h = 0; h < num_heads; ++h) {
                    const int hoff = h * head_dim;
                    for (int i = 0; i < seq_len; ++i) {
                        float maxv = -1e30f;
                        for (int j = 0; j < seq_len; ++j) {
                            float s = -1e9f;
                            if (!(causal && j > i)) {
                                float dot = 0.0f;
                                for (int k = 0; k < head_dim; ++k) dot += q_at(i, hoff + k) * k_at(j, hoff + k);
                                s = dot * scale;
                            }
                            const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(seq_len) + static_cast<size_t>(i)) * static_cast<size_t>(seq_len) + static_cast<size_t>(j);
                            P[idx] = s;
                            if (s > maxv) maxv = s;
                        }
                        float sum = 0.0f;
                        for (int j = 0; j < seq_len; ++j) {
                            const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(seq_len) + static_cast<size_t>(i)) * static_cast<size_t>(seq_len) + static_cast<size_t>(j);
                            const float e = std::exp(P[idx] - maxv);
                            P[idx] = e;
                            sum += e;
                        }
                        const float inv = 1.0f / (sum + 1e-9f);
                        for (int j = 0; j < seq_len; ++j) {
                            const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(seq_len) + static_cast<size_t>(i)) * static_cast<size_t>(seq_len) + static_cast<size_t>(j);
                            P[idx] *= inv;
                        }

                        for (int k = 0; k < head_dim; ++k) {
                            float acc = 0.0f;
                            for (int j = 0; j < seq_len; ++j) {
                                const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(seq_len) + static_cast<size_t>(i)) * static_cast<size_t>(seq_len) + static_cast<size_t>(j);
                                acc += P[idx] * v_at(j, hoff + k);
                            }
                            attended[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] = acc;
                        }
                    }
                }

                std::vector<float> dAtt(static_cast<size_t>(seq_len) * static_cast<size_t>(embed_dim), 0.0f);
                for (int i = 0; i < seq_len; ++i) {
                    const float* dy = &grad_out[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
                    float* da = &dAtt[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
                    for (int k = 0; k < embed_dim; ++k) {
                        float sum = 0.0f;
                        for (int n = 0; n < embed_dim; ++n) {
                            sum += dy[static_cast<size_t>(n)] * Wout[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
                        }
                        da[static_cast<size_t>(k)] = sum;
                    }
                }

                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }

                float* gradWqkv = layer.grad_weights.data();
                float* gradWout = layer.grad_weights.data() + static_cast<size_t>(qkv_size);

                for (int k = 0; k < embed_dim; ++k) {
                    for (int n = 0; n < embed_dim; ++n) {
                        float sum = 0.0f;
                        for (int i = 0; i < seq_len; ++i) {
                            sum += attended[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(k)]
                                 * grad_out[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
                        }
                        gradWout[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)] += sum;
                    }
                }

                std::vector<float> dQ(static_cast<size_t>(seq_len) * static_cast<size_t>(embed_dim), 0.0f);
                std::vector<float> dK(static_cast<size_t>(seq_len) * static_cast<size_t>(embed_dim), 0.0f);
                std::vector<float> dV(static_cast<size_t>(seq_len) * static_cast<size_t>(embed_dim), 0.0f);

                for (int h = 0; h < num_heads; ++h) {
                    const int hoff = h * head_dim;

                    for (int j = 0; j < seq_len; ++j) {
                        for (int k = 0; k < head_dim; ++k) {
                            float sum = 0.0f;
                            for (int i = 0; i < seq_len; ++i) {
                                const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(seq_len) + static_cast<size_t>(i)) * static_cast<size_t>(seq_len) + static_cast<size_t>(j);
                                sum += P[idx] * dAtt[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)];
                            }
                            dV[static_cast<size_t>(j) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] += sum;
                        }
                    }

                    std::vector<float> dP(static_cast<size_t>(seq_len) * static_cast<size_t>(seq_len), 0.0f);
                    for (int i = 0; i < seq_len; ++i) {
                        for (int j = 0; j < seq_len; ++j) {
                            float sum = 0.0f;
                            for (int k = 0; k < head_dim; ++k) {
                                sum += dAtt[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] * v_at(j, hoff + k);
                            }
                            dP[static_cast<size_t>(i) * static_cast<size_t>(seq_len) + static_cast<size_t>(j)] = sum;
                        }
                    }

                    std::vector<float> dS(static_cast<size_t>(seq_len) * static_cast<size_t>(seq_len), 0.0f);
                    for (int i = 0; i < seq_len; ++i) {
                        const float* p_row = &P[(static_cast<size_t>(h) * static_cast<size_t>(seq_len) + static_cast<size_t>(i)) * static_cast<size_t>(seq_len)];
                        const float* dp_row = &dP[static_cast<size_t>(i) * static_cast<size_t>(seq_len)];
                        float dot = 0.0f;
                        for (int j = 0; j < seq_len; ++j) dot += dp_row[j] * p_row[j];
                        for (int j = 0; j < seq_len; ++j) {
                            if (causal && j > i) {
                                dS[static_cast<size_t>(i) * static_cast<size_t>(seq_len) + static_cast<size_t>(j)] = 0.0f;
                                continue;
                            }
                            dS[static_cast<size_t>(i) * static_cast<size_t>(seq_len) + static_cast<size_t>(j)] = (dp_row[j] - dot) * p_row[j];
                        }
                    }

                    for (int i = 0; i < seq_len; ++i) {
                        for (int k = 0; k < head_dim; ++k) {
                            float sum = 0.0f;
                            for (int j = 0; j < seq_len; ++j) {
                                sum += dS[static_cast<size_t>(i) * static_cast<size_t>(seq_len) + static_cast<size_t>(j)] * k_at(j, hoff + k);
                            }
                            dQ[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] += sum * scale;
                        }
                    }

                    for (int j = 0; j < seq_len; ++j) {
                        for (int k = 0; k < head_dim; ++k) {
                            float sum = 0.0f;
                            for (int i = 0; i < seq_len; ++i) {
                                sum += dS[static_cast<size_t>(i) * static_cast<size_t>(seq_len) + static_cast<size_t>(j)] * q_at(i, hoff + k);
                            }
                            dK[static_cast<size_t>(j) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] += sum * scale;
                        }
                    }
                }

                std::vector<float> dqkv(static_cast<size_t>(seq_len) * static_cast<size_t>(3 * embed_dim), 0.0f);
                for (int t = 0; t < seq_len; ++t) {
                    float* row = &dqkv[static_cast<size_t>(t) * static_cast<size_t>(3 * embed_dim)];
                    for (int d = 0; d < embed_dim; ++d) {
                        row[static_cast<size_t>(d)] = dQ[static_cast<size_t>(t) * static_cast<size_t>(embed_dim) + static_cast<size_t>(d)];
                        row[static_cast<size_t>(embed_dim + d)] = dK[static_cast<size_t>(t) * static_cast<size_t>(embed_dim) + static_cast<size_t>(d)];
                        row[static_cast<size_t>(2 * embed_dim + d)] = dV[static_cast<size_t>(t) * static_cast<size_t>(embed_dim) + static_cast<size_t>(d)];
                    }
                }

                for (int n = 0; n < 3 * embed_dim; ++n) {
                    for (int k = 0; k < embed_dim; ++k) {
                        float sum = 0.0f;
                        for (int i = 0; i < seq_len; ++i) {
                            sum += x[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(k)]
                                 * dqkv[static_cast<size_t>(i) * static_cast<size_t>(3 * embed_dim) + static_cast<size_t>(n)];
                        }
                        gradWqkv[static_cast<size_t>(n) * static_cast<size_t>(embed_dim) +
                                 static_cast<size_t>(k)] += sum;
                    }
                }

                grad_inputs.resize(1);
                grad_inputs[0].assign(static_cast<size_t>(seq_len) * static_cast<size_t>(embed_dim), 0.0f);
                for (int i = 0; i < seq_len; ++i) {
                    const float* drow = &dqkv[static_cast<size_t>(i) * static_cast<size_t>(3 * embed_dim)];
                    float* xrow = &grad_inputs[0][static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
                    for (int k = 0; k < embed_dim; ++k) {
                        float sum = 0.0f;
                        for (int n = 0; n < 3 * embed_dim; ++n) {
                            sum += drow[n] *
                                   Wqkv[static_cast<size_t>(n) * static_cast<size_t>(embed_dim) +
                                         static_cast<size_t>(k)];
                        }
                        xrow[static_cast<size_t>(k)] = sum;
                    }
                }
                return true;
            }

            case LayerType::MultiHeadAttention: {
                // Même paramétrisation que SelfAttention dans ce runtime.
                layer.type_enum = LayerType::SelfAttention;
                const bool ok = cpu_backward_layer(inputs, grad_outputs, grad_inputs, layer, training);
                layer.type_enum = LayerType::MultiHeadAttention;
                return ok;
            }

            case LayerType::CrossAttention: {
                if (inputs.size() < 2 || !inputs[1]) return false;
                const std::vector<float>& q_in = *inputs[0];
                const std::vector<float>& kv_in = *inputs[1];
                const std::vector<float>& grad_out = *grad_outputs[0];
                const int num_heads = layer.num_heads > 0 ? layer.num_heads : 1;
                const bool causal = layer.causal;
                int embed_dim = layer.embed_dim;
                if (embed_dim <= 0 && layer.head_dim > 0 && num_heads > 0) {
                    embed_dim = layer.head_dim * num_heads;
                }
                const int kv_embed_dim = layer.in_features > 0 ? layer.in_features : embed_dim;
                if (embed_dim <= 0 || kv_embed_dim != embed_dim) return false;
                if ((q_in.size() % static_cast<size_t>(embed_dim)) != 0) return false;
                if ((kv_in.size() % static_cast<size_t>(embed_dim)) != 0) return false;

                const int query_len = static_cast<int>(q_in.size() / static_cast<size_t>(embed_dim));
                const int kv_len = static_cast<int>(kv_in.size() / static_cast<size_t>(embed_dim));
                if (query_len <= 0 || kv_len <= 0) return false;
                if (static_cast<int>(grad_out.size()) != query_len * embed_dim) return false;
                if ((embed_dim % num_heads) != 0) return false;

                const float* weights = layer.getWeights();
                if (!weights) return false;
                const int q_size = embed_dim * embed_dim;
                const int kv_size = embed_dim * (2 * embed_dim);
                const int out_size = embed_dim * embed_dim;
                if (layer.getWeightsSize() < static_cast<size_t>(q_size + kv_size + out_size)) return false;

                const float* Wq = weights;
                const float* Wkv = weights + q_size;
                const float* Wout = weights + q_size + kv_size;

                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }
                float* gWq = layer.grad_weights.data();
                float* gWkv = layer.grad_weights.data() + static_cast<size_t>(q_size);
                float* gWout = layer.grad_weights.data() + static_cast<size_t>(q_size + kv_size);

                const int head_dim = embed_dim / num_heads;
                const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

                std::vector<float> Q(static_cast<size_t>(query_len) * static_cast<size_t>(embed_dim), 0.0f);
                std::vector<float> KV(static_cast<size_t>(kv_len) * static_cast<size_t>(2 * embed_dim), 0.0f);
                for (int i = 0; i < query_len; ++i) {
                    const float* xrow = &q_in[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
                    float* yrow = &Q[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
                    for (int n = 0; n < embed_dim; ++n) {
                        float sum = 0.0f;
                        for (int k = 0; k < embed_dim; ++k) sum += xrow[k] * Wq[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
                        yrow[n] = sum;
                    }
                }
                for (int i = 0; i < kv_len; ++i) {
                    const float* xrow = &kv_in[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
                    float* yrow = &KV[static_cast<size_t>(i) * static_cast<size_t>(2 * embed_dim)];
                    for (int n = 0; n < 2 * embed_dim; ++n) {
                        float sum = 0.0f;
                        for (int k = 0; k < embed_dim; ++k) sum += xrow[k] * Wkv[static_cast<size_t>(k) * static_cast<size_t>(2 * embed_dim) + static_cast<size_t>(n)];
                        yrow[n] = sum;
                    }
                }

                auto q_at = [&](int t, int d) -> float { return Q[static_cast<size_t>(t) * static_cast<size_t>(embed_dim) + static_cast<size_t>(d)]; };
                auto k_at = [&](int t, int d) -> float { return KV[static_cast<size_t>(t) * static_cast<size_t>(2 * embed_dim) + static_cast<size_t>(d)]; };
                auto v_at = [&](int t, int d) -> float { return KV[static_cast<size_t>(t) * static_cast<size_t>(2 * embed_dim) + static_cast<size_t>(embed_dim + d)]; };

                std::vector<float> P(static_cast<size_t>(num_heads) * static_cast<size_t>(query_len) * static_cast<size_t>(kv_len), 0.0f);
                std::vector<float> attended(static_cast<size_t>(query_len) * static_cast<size_t>(embed_dim), 0.0f);
                for (int h = 0; h < num_heads; ++h) {
                    const int hoff = h * head_dim;
                    for (int i = 0; i < query_len; ++i) {
                        float maxv = -1e30f;
                        for (int j = 0; j < kv_len; ++j) {
                            float s = -1e9f;
                            if (!(causal && j > i)) {
                                float dot = 0.0f;
                                for (int k = 0; k < head_dim; ++k) dot += q_at(i, hoff + k) * k_at(j, hoff + k);
                                s = dot * scale;
                            }
                            const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(query_len) + static_cast<size_t>(i)) * static_cast<size_t>(kv_len) + static_cast<size_t>(j);
                            P[idx] = s;
                            if (s > maxv) maxv = s;
                        }
                        float sum = 0.0f;
                        for (int j = 0; j < kv_len; ++j) {
                            const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(query_len) + static_cast<size_t>(i)) * static_cast<size_t>(kv_len) + static_cast<size_t>(j);
                            const float e = std::exp(P[idx] - maxv);
                            P[idx] = e;
                            sum += e;
                        }
                        const float inv = 1.0f / (sum + 1e-9f);
                        for (int j = 0; j < kv_len; ++j) {
                            const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(query_len) + static_cast<size_t>(i)) * static_cast<size_t>(kv_len) + static_cast<size_t>(j);
                            P[idx] *= inv;
                        }
                        for (int k = 0; k < head_dim; ++k) {
                            float acc = 0.0f;
                            for (int j = 0; j < kv_len; ++j) {
                                const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(query_len) + static_cast<size_t>(i)) * static_cast<size_t>(kv_len) + static_cast<size_t>(j);
                                acc += P[idx] * v_at(j, hoff + k);
                            }
                            attended[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] = acc;
                        }
                    }
                }

                std::vector<float> dAtt(static_cast<size_t>(query_len) * static_cast<size_t>(embed_dim), 0.0f);
                for (int i = 0; i < query_len; ++i) {
                    const float* dy = &grad_out[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
                    float* da = &dAtt[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
                    for (int k = 0; k < embed_dim; ++k) {
                        float sum = 0.0f;
                        for (int n = 0; n < embed_dim; ++n) sum += dy[n] * Wout[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
                        da[k] = sum;
                    }
                }

                for (int k = 0; k < embed_dim; ++k) {
                    for (int n = 0; n < embed_dim; ++n) {
                        float sum = 0.0f;
                        for (int i = 0; i < query_len; ++i) {
                            sum += attended[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(k)]
                                 * grad_out[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
                        }
                        gWout[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)] += sum;
                    }
                }

                std::vector<float> dQ(static_cast<size_t>(query_len) * static_cast<size_t>(embed_dim), 0.0f);
                std::vector<float> dK(static_cast<size_t>(kv_len) * static_cast<size_t>(embed_dim), 0.0f);
                std::vector<float> dV(static_cast<size_t>(kv_len) * static_cast<size_t>(embed_dim), 0.0f);

                for (int h = 0; h < num_heads; ++h) {
                    const int hoff = h * head_dim;
                    for (int j = 0; j < kv_len; ++j) {
                        for (int k = 0; k < head_dim; ++k) {
                            float sum = 0.0f;
                            for (int i = 0; i < query_len; ++i) {
                                const size_t idx = (static_cast<size_t>(h) * static_cast<size_t>(query_len) + static_cast<size_t>(i)) * static_cast<size_t>(kv_len) + static_cast<size_t>(j);
                                sum += P[idx] * dAtt[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)];
                            }
                            dV[static_cast<size_t>(j) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] += sum;
                        }
                    }

                    std::vector<float> dP(static_cast<size_t>(query_len) * static_cast<size_t>(kv_len), 0.0f);
                    for (int i = 0; i < query_len; ++i) {
                        for (int j = 0; j < kv_len; ++j) {
                            float sum = 0.0f;
                            for (int k = 0; k < head_dim; ++k) {
                                sum += dAtt[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] * v_at(j, hoff + k);
                            }
                            dP[static_cast<size_t>(i) * static_cast<size_t>(kv_len) + static_cast<size_t>(j)] = sum;
                        }
                    }

                    std::vector<float> dS(static_cast<size_t>(query_len) * static_cast<size_t>(kv_len), 0.0f);
                    for (int i = 0; i < query_len; ++i) {
                        const float* p_row = &P[(static_cast<size_t>(h) * static_cast<size_t>(query_len) + static_cast<size_t>(i)) * static_cast<size_t>(kv_len)];
                        const float* dp_row = &dP[static_cast<size_t>(i) * static_cast<size_t>(kv_len)];
                        float dot = 0.0f;
                        for (int j = 0; j < kv_len; ++j) dot += dp_row[j] * p_row[j];
                        for (int j = 0; j < kv_len; ++j) {
                            if (causal && j > i) {
                                dS[static_cast<size_t>(i) * static_cast<size_t>(kv_len) + static_cast<size_t>(j)] = 0.0f;
                                continue;
                            }
                            dS[static_cast<size_t>(i) * static_cast<size_t>(kv_len) + static_cast<size_t>(j)] = (dp_row[j] - dot) * p_row[j];
                        }
                    }

                    for (int i = 0; i < query_len; ++i) {
                        for (int k = 0; k < head_dim; ++k) {
                            float sum = 0.0f;
                            for (int j = 0; j < kv_len; ++j) sum += dS[static_cast<size_t>(i) * static_cast<size_t>(kv_len) + static_cast<size_t>(j)] * k_at(j, hoff + k);
                            dQ[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] += sum * scale;
                        }
                    }
                    for (int j = 0; j < kv_len; ++j) {
                        for (int k = 0; k < head_dim; ++k) {
                            float sum = 0.0f;
                            for (int i = 0; i < query_len; ++i) sum += dS[static_cast<size_t>(i) * static_cast<size_t>(kv_len) + static_cast<size_t>(j)] * q_at(i, hoff + k);
                            dK[static_cast<size_t>(j) * static_cast<size_t>(embed_dim) + static_cast<size_t>(hoff + k)] += sum * scale;
                        }
                    }
                }

                for (int k = 0; k < embed_dim; ++k) {
                    for (int n = 0; n < embed_dim; ++n) {
                        float sum = 0.0f;
                        for (int i = 0; i < query_len; ++i) {
                            sum += q_in[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(k)]
                                 * dQ[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
                        }
                        gWq[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)] += sum;
                    }
                }

                std::vector<float> dKV(static_cast<size_t>(kv_len) * static_cast<size_t>(2 * embed_dim), 0.0f);
                for (int i = 0; i < kv_len; ++i) {
                    float* row = &dKV[static_cast<size_t>(i) * static_cast<size_t>(2 * embed_dim)];
                    for (int d = 0; d < embed_dim; ++d) {
                        row[d] = dK[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(d)];
                        row[embed_dim + d] = dV[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(d)];
                    }
                }
                for (int k = 0; k < embed_dim; ++k) {
                    for (int n = 0; n < 2 * embed_dim; ++n) {
                        float sum = 0.0f;
                        for (int i = 0; i < kv_len; ++i) {
                            sum += kv_in[static_cast<size_t>(i) * static_cast<size_t>(embed_dim) + static_cast<size_t>(k)]
                                 * dKV[static_cast<size_t>(i) * static_cast<size_t>(2 * embed_dim) + static_cast<size_t>(n)];
                        }
                        gWkv[static_cast<size_t>(k) * static_cast<size_t>(2 * embed_dim) + static_cast<size_t>(n)] += sum;
                    }
                }

                grad_inputs.resize(2);
                grad_inputs[0].assign(static_cast<size_t>(query_len) * static_cast<size_t>(embed_dim), 0.0f);
                grad_inputs[1].assign(static_cast<size_t>(kv_len) * static_cast<size_t>(embed_dim), 0.0f);

                for (int i = 0; i < query_len; ++i) {
                    const float* drow = &dQ[static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
                    float* xrow = &grad_inputs[0][static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
                    for (int k = 0; k < embed_dim; ++k) {
                        float sum = 0.0f;
                        for (int n = 0; n < embed_dim; ++n) sum += drow[n] * Wq[static_cast<size_t>(k) * static_cast<size_t>(embed_dim) + static_cast<size_t>(n)];
                        xrow[k] = sum;
                    }
                }
                for (int i = 0; i < kv_len; ++i) {
                    const float* drow = &dKV[static_cast<size_t>(i) * static_cast<size_t>(2 * embed_dim)];
                    float* xrow = &grad_inputs[1][static_cast<size_t>(i) * static_cast<size_t>(embed_dim)];
                    for (int k = 0; k < embed_dim; ++k) {
                        float sum = 0.0f;
                        for (int n = 0; n < 2 * embed_dim; ++n) sum += drow[n] * Wkv[static_cast<size_t>(k) * static_cast<size_t>(2 * embed_dim) + static_cast<size_t>(n)];
                        xrow[k] = sum;
                    }
                }
                return true;
            }

            case LayerType::Lambda: {
                if (inputs.empty() || inputs[0] == nullptr) return false;
                return passthrough_grad(*inputs[0], *grad_outputs[0], grad_inputs);
            }

            case LayerType::LSTM: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int T = layer.seq_len;
                const int I = layer.in_features;
                const int H = layer.out_features;
                if (T <= 0 || I <= 0 || H <= 0) return false;
                if (static_cast<int>(x.size()) != T * I) return false;
                if (static_cast<int>(go.size()) != T * H) return false;
                const bool use_bias = layer.use_bias;
                const float* W = layer.getWeights();
                if (!W) return false;

                const size_t Wih_sz = static_cast<size_t>(4 * H) * static_cast<size_t>(I);
                const size_t Whh_sz = static_cast<size_t>(4 * H) * static_cast<size_t>(H);
                const size_t bih_sz = use_bias ? static_cast<size_t>(4 * H) : 0ULL;
                const size_t bhh_sz = use_bias ? static_cast<size_t>(4 * H) : 0ULL;
                const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                if (layer.getWeightsSize() < need) return false;

                const float* Wih = W;
                const float* Whh = Wih + Wih_sz;
                const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                std::vector<float> i_gate(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                std::vector<float> f_gate(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                std::vector<float> g_gate(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                std::vector<float> o_gate(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                std::vector<float> c_state(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                std::vector<float> h_state(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);

                std::vector<float> h_prev(static_cast<size_t>(H), 0.0f);
                std::vector<float> c_prev(static_cast<size_t>(H), 0.0f);
                for (int t = 0; t < T; ++t) {
                    const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                    for (int h = 0; h < H; ++h) {
                        auto dot_in = [&](const float* wrow) -> float {
                            float s = 0.0f;
                            for (int ii = 0; ii < I; ++ii) s += wrow[static_cast<size_t>(ii)] * xt[static_cast<size_t>(ii)];
                            return s;
                        };
                        auto dot_h = [&](const float* wrow) -> float {
                            float s = 0.0f;
                            for (int k = 0; k < H; ++k) s += wrow[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                            return s;
                        };
                        const float* wii = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                        const float* wif = Wih + static_cast<size_t>(H + h) * static_cast<size_t>(I);
                        const float* wig = Wih + static_cast<size_t>(2 * H + h) * static_cast<size_t>(I);
                        const float* wio = Wih + static_cast<size_t>(3 * H + h) * static_cast<size_t>(I);
                        const float* whi = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                        const float* whf = Whh + static_cast<size_t>(H + h) * static_cast<size_t>(H);
                        const float* whg = Whh + static_cast<size_t>(2 * H + h) * static_cast<size_t>(H);
                        const float* who = Whh + static_cast<size_t>(3 * H + h) * static_cast<size_t>(H);

                        float si = dot_in(wii) + dot_h(whi);
                        float sf = dot_in(wif) + dot_h(whf);
                        float sg = dot_in(wig) + dot_h(whg);
                        float so = dot_in(wio) + dot_h(who);
                        if (bih) {
                            si += bih[static_cast<size_t>(h)];
                            sf += bih[static_cast<size_t>(H + h)];
                            sg += bih[static_cast<size_t>(2 * H + h)];
                            so += bih[static_cast<size_t>(3 * H + h)];
                        }
                        if (bhh) {
                            si += bhh[static_cast<size_t>(h)];
                            sf += bhh[static_cast<size_t>(H + h)];
                            sg += bhh[static_cast<size_t>(2 * H + h)];
                            so += bhh[static_cast<size_t>(3 * H + h)];
                        }

                        const float ig = sigmoid_scalar(si);
                        const float fg = sigmoid_scalar(sf);
                        const float gg = std::tanh(sg);
                        const float og = sigmoid_scalar(so);
                        const float c = fg * c_prev[static_cast<size_t>(h)] + ig * gg;
                        const float hv = og * std::tanh(c);

                        i_gate[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] = ig;
                        f_gate[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] = fg;
                        g_gate[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] = gg;
                        o_gate[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] = og;
                        c_state[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] = c;
                        h_state[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] = hv;
                        c_prev[static_cast<size_t>(h)] = c;
                        h_prev[static_cast<size_t>(h)] = hv;
                    }
                }

                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }
                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                std::vector<float> dh_next(static_cast<size_t>(H), 0.0f);
                std::vector<float> dc_next(static_cast<size_t>(H), 0.0f);

                for (int t = T - 1; t >= 0; --t) {
                    const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                    const float* hprev = (t > 0) ? &h_state[static_cast<size_t>(t - 1) * static_cast<size_t>(H)] : nullptr;
                    const float* cprev = (t > 0) ? &c_state[static_cast<size_t>(t - 1) * static_cast<size_t>(H)] : nullptr;

                    std::vector<float> dh_prev(static_cast<size_t>(H), 0.0f);
                    std::vector<float> dc_prev(static_cast<size_t>(H), 0.0f);

                    for (int h = 0; h < H; ++h) {
                        const size_t th = static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h);
                        const float ig = i_gate[th];
                        const float fg = f_gate[th];
                        const float gg = g_gate[th];
                        const float og = o_gate[th];
                        const float c = c_state[th];
                        const float tanh_c = std::tanh(c);
                        const float h_prev_v = hprev ? hprev[static_cast<size_t>(h)] : 0.0f;
                        const float c_prev_v = cprev ? cprev[static_cast<size_t>(h)] : 0.0f;

                        const float dht = go[th] + dh_next[static_cast<size_t>(h)];
                        const float do_gate = dht * tanh_c;
                        const float dc = dc_next[static_cast<size_t>(h)] + dht * og * (1.0f - tanh_c * tanh_c);
                        const float di_gate = dc * gg;
                        const float dg_gate = dc * ig;
                        const float df_gate = dc * c_prev_v;
                        dc_prev[static_cast<size_t>(h)] = dc * fg;

                        const float dai = di_gate * ig * (1.0f - ig);
                        const float daf = df_gate * fg * (1.0f - fg);
                        const float dag = dg_gate * (1.0f - gg * gg);
                        const float dao = do_gate * og * (1.0f - og);

                        const size_t row_i = static_cast<size_t>(h);
                        const size_t row_f = static_cast<size_t>(H + h);
                        const size_t row_g = static_cast<size_t>(2 * H + h);
                        const size_t row_o = static_cast<size_t>(3 * H + h);

                        for (int ii = 0; ii < I; ++ii) {
                            const float xv = xt[static_cast<size_t>(ii)];
                            layer.grad_weights[row_i * static_cast<size_t>(I) + static_cast<size_t>(ii)] += dai * xv;
                            layer.grad_weights[row_f * static_cast<size_t>(I) + static_cast<size_t>(ii)] += daf * xv;
                            layer.grad_weights[row_g * static_cast<size_t>(I) + static_cast<size_t>(ii)] += dag * xv;
                            layer.grad_weights[row_o * static_cast<size_t>(I) + static_cast<size_t>(ii)] += dao * xv;

                            grad_inputs[0][static_cast<size_t>(t) * static_cast<size_t>(I) + static_cast<size_t>(ii)] +=
                                dai * Wih[row_i * static_cast<size_t>(I) + static_cast<size_t>(ii)] +
                                daf * Wih[row_f * static_cast<size_t>(I) + static_cast<size_t>(ii)] +
                                dag * Wih[row_g * static_cast<size_t>(I) + static_cast<size_t>(ii)] +
                                dao * Wih[row_o * static_cast<size_t>(I) + static_cast<size_t>(ii)];
                        }

                        for (int k = 0; k < H; ++k) {
                            const float hp = hprev ? hprev[static_cast<size_t>(k)] : 0.0f;
                            layer.grad_weights[Wih_sz + row_i * static_cast<size_t>(H) + static_cast<size_t>(k)] += dai * hp;
                            layer.grad_weights[Wih_sz + row_f * static_cast<size_t>(H) + static_cast<size_t>(k)] += daf * hp;
                            layer.grad_weights[Wih_sz + row_g * static_cast<size_t>(H) + static_cast<size_t>(k)] += dag * hp;
                            layer.grad_weights[Wih_sz + row_o * static_cast<size_t>(H) + static_cast<size_t>(k)] += dao * hp;

                            dh_prev[static_cast<size_t>(k)] +=
                                dai * Whh[row_i * static_cast<size_t>(H) + static_cast<size_t>(k)] +
                                daf * Whh[row_f * static_cast<size_t>(H) + static_cast<size_t>(k)] +
                                dag * Whh[row_g * static_cast<size_t>(H) + static_cast<size_t>(k)] +
                                dao * Whh[row_o * static_cast<size_t>(H) + static_cast<size_t>(k)];
                        }

                        if (use_bias) {
                            const size_t b0 = Wih_sz + Whh_sz;
                            layer.grad_weights[b0 + row_i] += dai;
                            layer.grad_weights[b0 + row_f] += daf;
                            layer.grad_weights[b0 + row_g] += dag;
                            layer.grad_weights[b0 + row_o] += dao;
                            layer.grad_weights[b0 + bih_sz + row_i] += dai;
                            layer.grad_weights[b0 + bih_sz + row_f] += daf;
                            layer.grad_weights[b0 + bih_sz + row_g] += dag;
                            layer.grad_weights[b0 + bih_sz + row_o] += dao;
                        }
                    }

                    dh_next.swap(dh_prev);
                    dc_next.swap(dc_prev);
                }
                return true;
            }

            case LayerType::GRU: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int T = layer.seq_len;
                const int I = layer.in_features;
                const int H = layer.out_features;
                if (T <= 0 || I <= 0 || H <= 0) return false;
                if (static_cast<int>(x.size()) != T * I) return false;
                if (static_cast<int>(go.size()) != T * H) return false;

                const bool use_bias = layer.use_bias;
                const float* W = layer.getWeights();
                if (!W) return false;
                const size_t Wih_sz = static_cast<size_t>(3 * H) * static_cast<size_t>(I);
                const size_t Whh_sz = static_cast<size_t>(3 * H) * static_cast<size_t>(H);
                const size_t bih_sz = use_bias ? static_cast<size_t>(3 * H) : 0ULL;
                const size_t bhh_sz = use_bias ? static_cast<size_t>(3 * H) : 0ULL;
                const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                if (layer.getWeightsSize() < need) return false;
                const float* Wih = W;
                const float* Whh = Wih + Wih_sz;
                const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                std::vector<float> h_state(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                std::vector<float> r_state(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                std::vector<float> z_state(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                std::vector<float> n_state(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);

                std::vector<float> h_prev(static_cast<size_t>(H), 0.0f);
                for (int t = 0; t < T; ++t) {
                    const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                    for (int h = 0; h < H; ++h) {
                        const float* w_ir = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                        const float* w_hr = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                        float sr = 0.0f;
                        for (int i = 0; i < I; ++i) sr += w_ir[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                        for (int k = 0; k < H; ++k) sr += w_hr[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                        if (bih) sr += bih[static_cast<size_t>(h)];
                        if (bhh) sr += bhh[static_cast<size_t>(h)];
                        const float r = sigmoid_scalar(sr);

                        const float* w_iz = Wih + static_cast<size_t>(H + h) * static_cast<size_t>(I);
                        const float* w_hz = Whh + static_cast<size_t>(H + h) * static_cast<size_t>(H);
                        float sz = 0.0f;
                        for (int i = 0; i < I; ++i) sz += w_iz[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                        for (int k = 0; k < H; ++k) sz += w_hz[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                        if (bih) sz += bih[static_cast<size_t>(H + h)];
                        if (bhh) sz += bhh[static_cast<size_t>(H + h)];
                        const float z = sigmoid_scalar(sz);

                        const float* w_in = Wih + static_cast<size_t>(2 * H + h) * static_cast<size_t>(I);
                        const float* w_hn = Whh + static_cast<size_t>(2 * H + h) * static_cast<size_t>(H);
                        float sn = 0.0f;
                        for (int i = 0; i < I; ++i) sn += w_in[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                        for (int k = 0; k < H; ++k) sn += w_hn[static_cast<size_t>(k)] * (r * h_prev[static_cast<size_t>(k)]);
                        if (bih) sn += bih[static_cast<size_t>(2 * H + h)];
                        if (bhh) sn += bhh[static_cast<size_t>(2 * H + h)];
                        const float n = std::tanh(sn);

                        const float hv = (1.0f - z) * n + z * h_prev[static_cast<size_t>(h)];

                        const size_t th = static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h);
                        r_state[th] = r;
                        z_state[th] = z;
                        n_state[th] = n;
                        h_state[th] = hv;
                    }
                    for (int h = 0; h < H; ++h) {
                        h_prev[static_cast<size_t>(h)] = h_state[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)];
                    }
                }

                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }
                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);

                std::vector<float> dh_next(static_cast<size_t>(H), 0.0f);
                for (int t = T - 1; t >= 0; --t) {
                    const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                    const float* hprev = (t > 0) ? &h_state[static_cast<size_t>(t - 1) * static_cast<size_t>(H)] : nullptr;
                    std::vector<float> dh_prev(static_cast<size_t>(H), 0.0f);

                    std::vector<float> da_r(static_cast<size_t>(H), 0.0f);
                    std::vector<float> da_z(static_cast<size_t>(H), 0.0f);
                    std::vector<float> da_n(static_cast<size_t>(H), 0.0f);

                    for (int h = 0; h < H; ++h) {
                        const size_t th = static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h);
                        const float r = r_state[th];
                        const float z = z_state[th];
                        const float n = n_state[th];
                        const float hp = hprev ? hprev[static_cast<size_t>(h)] : 0.0f;
                        const float dht = go[th] + dh_next[static_cast<size_t>(h)];
                        const float dn = dht * (1.0f - z);
                        const float dz = dht * (hp - n);
                        dh_prev[static_cast<size_t>(h)] += dht * z;
                        da_n[static_cast<size_t>(h)] = dn * (1.0f - n * n);
                        da_z[static_cast<size_t>(h)] = dz * z * (1.0f - z);
                        (void)r;
                    }

                    std::vector<float> drh(static_cast<size_t>(H), 0.0f);
                    for (int h = 0; h < H; ++h) {
                        const size_t row_n = static_cast<size_t>(2 * H + h);
                        for (int i = 0; i < I; ++i) {
                            const float xv = xt[static_cast<size_t>(i)];
                            layer.grad_weights[row_n * static_cast<size_t>(I) + static_cast<size_t>(i)] += da_n[static_cast<size_t>(h)] * xv;
                            grad_inputs[0][static_cast<size_t>(t) * static_cast<size_t>(I) + static_cast<size_t>(i)] +=
                                da_n[static_cast<size_t>(h)] * Wih[row_n * static_cast<size_t>(I) + static_cast<size_t>(i)];
                        }
                        for (int k = 0; k < H; ++k) {
                            const float hp = hprev ? hprev[static_cast<size_t>(k)] : 0.0f;
                            const float r = r_state[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(k)];
                            layer.grad_weights[Wih_sz + row_n * static_cast<size_t>(H) + static_cast<size_t>(k)] += da_n[static_cast<size_t>(h)] * (r * hp);
                            drh[static_cast<size_t>(k)] += da_n[static_cast<size_t>(h)] * Whh[row_n * static_cast<size_t>(H) + static_cast<size_t>(k)];
                        }
                        if (use_bias) {
                            const size_t b0 = Wih_sz + Whh_sz;
                            layer.grad_weights[b0 + row_n] += da_n[static_cast<size_t>(h)];
                            layer.grad_weights[b0 + bih_sz + row_n] += da_n[static_cast<size_t>(h)];
                        }
                    }

                    for (int k = 0; k < H; ++k) {
                        const float hp = hprev ? hprev[static_cast<size_t>(k)] : 0.0f;
                        const float r = r_state[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(k)];
                        const float dr = drh[static_cast<size_t>(k)] * hp;
                        dh_prev[static_cast<size_t>(k)] += drh[static_cast<size_t>(k)] * r;
                        da_r[static_cast<size_t>(k)] = dr * r * (1.0f - r);
                    }

                    for (int h = 0; h < H; ++h) {
                        const size_t row_r = static_cast<size_t>(h);
                        const size_t row_z = static_cast<size_t>(H + h);

                        for (int i = 0; i < I; ++i) {
                            const float xv = xt[static_cast<size_t>(i)];
                            layer.grad_weights[row_r * static_cast<size_t>(I) + static_cast<size_t>(i)] += da_r[static_cast<size_t>(h)] * xv;
                            layer.grad_weights[row_z * static_cast<size_t>(I) + static_cast<size_t>(i)] += da_z[static_cast<size_t>(h)] * xv;
                            grad_inputs[0][static_cast<size_t>(t) * static_cast<size_t>(I) + static_cast<size_t>(i)] +=
                                da_r[static_cast<size_t>(h)] * Wih[row_r * static_cast<size_t>(I) + static_cast<size_t>(i)] +
                                da_z[static_cast<size_t>(h)] * Wih[row_z * static_cast<size_t>(I) + static_cast<size_t>(i)];
                        }

                        for (int k = 0; k < H; ++k) {
                            const float hp = hprev ? hprev[static_cast<size_t>(k)] : 0.0f;
                            layer.grad_weights[Wih_sz + row_r * static_cast<size_t>(H) + static_cast<size_t>(k)] += da_r[static_cast<size_t>(h)] * hp;
                            layer.grad_weights[Wih_sz + row_z * static_cast<size_t>(H) + static_cast<size_t>(k)] += da_z[static_cast<size_t>(h)] * hp;

                            dh_prev[static_cast<size_t>(k)] +=
                                da_r[static_cast<size_t>(h)] * Whh[row_r * static_cast<size_t>(H) + static_cast<size_t>(k)] +
                                da_z[static_cast<size_t>(h)] * Whh[row_z * static_cast<size_t>(H) + static_cast<size_t>(k)];
                        }

                        if (use_bias) {
                            const size_t b0 = Wih_sz + Whh_sz;
                            layer.grad_weights[b0 + row_r] += da_r[static_cast<size_t>(h)];
                            layer.grad_weights[b0 + row_z] += da_z[static_cast<size_t>(h)];
                            layer.grad_weights[b0 + bih_sz + row_r] += da_r[static_cast<size_t>(h)];
                            layer.grad_weights[b0 + bih_sz + row_z] += da_z[static_cast<size_t>(h)];
                        }
                    }

                    dh_next.swap(dh_prev);
                }
                return true;
            }

            case LayerType::RNN: {
                const std::vector<float>& x = *inputs[0];
                const std::vector<float>& go = *grad_outputs[0];
                const int T = layer.seq_len;
                const int I = layer.in_features;
                const int H = layer.out_features;
                if (T <= 0 || I <= 0 || H <= 0) return false;
                if (static_cast<int>(x.size()) != T * I) return false;
                if (static_cast<int>(go.size()) != T * H) return false;
                const bool use_bias = layer.use_bias;
                const float* W = layer.getWeights();
                if (!W) return false;
                const size_t Wih_sz = static_cast<size_t>(H) * static_cast<size_t>(I);
                const size_t Whh_sz = static_cast<size_t>(H) * static_cast<size_t>(H);
                const size_t bih_sz = use_bias ? static_cast<size_t>(H) : 0ULL;
                const size_t bhh_sz = use_bias ? static_cast<size_t>(H) : 0ULL;
                const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                if (layer.getWeightsSize() < need) return false;

                const float* Wih = W;
                const float* Whh = W + Wih_sz;
                const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                std::vector<float> h_state(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                std::vector<float> h_prev(static_cast<size_t>(H), 0.0f);
                for (int t = 0; t < T; ++t) {
                    const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                    float* ht = &h_state[static_cast<size_t>(t) * static_cast<size_t>(H)];
                    for (int h = 0; h < H; ++h) {
                        float sum = 0.0f;
                        const float* wih = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                        const float* whh = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                        for (int i = 0; i < I; ++i) sum += wih[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                        for (int k = 0; k < H; ++k) sum += whh[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                        if (bih) sum += bih[static_cast<size_t>(h)];
                        if (bhh) sum += bhh[static_cast<size_t>(h)];
                        ht[static_cast<size_t>(h)] = std::tanh(sum);
                    }
                    for (int h = 0; h < H; ++h) h_prev[static_cast<size_t>(h)] = ht[static_cast<size_t>(h)];
                }

                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }

                grad_inputs.resize(1);
                grad_inputs[0].assign(x.size(), 0.0f);
                std::vector<float> dh_next(static_cast<size_t>(H), 0.0f);

                for (int t = T - 1; t >= 0; --t) {
                    const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                    const float* hprev = (t > 0) ? &h_state[static_cast<size_t>(t - 1) * static_cast<size_t>(H)] : nullptr;
                    std::vector<float> dh_prev(static_cast<size_t>(H), 0.0f);
                    for (int h = 0; h < H; ++h) {
                        const float h_t = h_state[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)];
                        const float dht = go[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] + dh_next[static_cast<size_t>(h)];
                        const float dpre = dht * (1.0f - h_t * h_t);

                        const float* wih = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                        const float* whh = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);

                        for (int i = 0; i < I; ++i) {
                            layer.grad_weights[static_cast<size_t>(h) * static_cast<size_t>(I) + static_cast<size_t>(i)] += dpre * xt[static_cast<size_t>(i)];
                            grad_inputs[0][static_cast<size_t>(t) * static_cast<size_t>(I) + static_cast<size_t>(i)] += dpre * wih[static_cast<size_t>(i)];
                        }
                        for (int k = 0; k < H; ++k) {
                            const float hp = hprev ? hprev[static_cast<size_t>(k)] : 0.0f;
                            layer.grad_weights[Wih_sz + static_cast<size_t>(h) * static_cast<size_t>(H) + static_cast<size_t>(k)] += dpre * hp;
                            dh_prev[static_cast<size_t>(k)] += dpre * whh[static_cast<size_t>(k)];
                        }
                        if (use_bias) {
                            layer.grad_weights[Wih_sz + Whh_sz + static_cast<size_t>(h)] += dpre;
                            layer.grad_weights[Wih_sz + Whh_sz + bih_sz + static_cast<size_t>(h)] += dpre;
                        }
                    }
                    dh_next.swap(dh_prev);
                }
                return true;
            }

            case LayerType::UNKNOWN:
                return false;

            default:
                return false;
        }
    } catch (...) {
        return false;
    }
}

} // namespace RuntimeLayerDispatch
