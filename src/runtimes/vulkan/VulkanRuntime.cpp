#include "runtimes/optimizers/PortableOptimizer.hpp"
#include "runtimes/vulkan/VulkanRuntime.hpp"

#ifdef ENABLE_VULKAN
#include "runtimes/vulkan/VulkanCompute.hpp"
#endif

#include "Layers.hpp"
#include "runtimes/LayerOps.hpp"
#include "runtimes/portable/PortableRuntime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

struct VulkanRuntime::Impl {
    PortableRuntime reference{"VULKAN"};
#ifdef ENABLE_VULKAN
    VulkanCompute::ComputeEngine engine;
#endif
    bool initialized = false;
};

bool VulkanRuntime::initialize(const RuntimeConfig& cfg) {
    config_ = cfg;

#ifndef ENABLE_VULKAN
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

    impl_->initialized = impl_->engine.initialize();
    if (!impl_->initialized) {
        shutdown();
        return false;
    }

    return true;
#endif
}

void VulkanRuntime::shutdown() {
#ifdef ENABLE_VULKAN
    if (impl_) {
        impl_->engine.cleanup();
        impl_->initialized = false;
    }
#endif
    delete impl_;
    impl_ = nullptr;
}

bool VulkanRuntime::isInitialized() const {
    return impl_ && impl_->initialized;
}

bool VulkanRuntime::supportsForwardLayerType(const LayerType type) const {
#ifdef ENABLE_VULKAN
    return !config_.disabled && PortableRuntime("VULKAN").supportsForwardLayerType(type);
#else
    (void)type; return false;
#endif
}

RuntimeCapabilityLevel VulkanRuntime::queryForwardCapability(const LayerType type) const {
#ifndef ENABLE_VULKAN
    return supportsForwardLayerType(type) ? RuntimeCapabilityLevel::HostFallback
                                          : RuntimeCapabilityLevel::Unsupported;
#else
    if(config_.disabled)return RuntimeCapabilityLevel::Unsupported;
    switch (type) {
        case LayerType::Linear:
        case LayerType::MatMul:
        case LayerType::BatchMatMul:
        case LayerType::Conv2d:
        case LayerType::Add:
        case LayerType::Multiply:
        case LayerType::ReLU:
        case LayerType::SiLU:
        case LayerType::GELU:
        case LayerType::Sigmoid:
        case LayerType::Tanh:
            return RuntimeCapabilityLevel::NativeOptimized;
        case LayerType::Subtract:
        case LayerType::Divide:
        case LayerType::LeakyReLU:
        case LayerType::Softplus:
        case LayerType::Mish:
        case LayerType::HardSigmoid:
        case LayerType::HardSwish:
        case LayerType::LayerNorm:
            return RuntimeCapabilityLevel::Native;
        default:
            return PortableRuntime("VULKAN").queryForwardCapability(type);
    }
#endif
}

bool VulkanRuntime::supportsBackwardLayerType(const LayerType type) const {
#ifdef ENABLE_VULKAN
    return !config_.disabled && PortableRuntime("VULKAN").supportsBackwardLayerType(type);
#else
    (void)type; return false;
#endif
}

RuntimeCapabilityLevel VulkanRuntime::queryBackwardCapability(const LayerType type) const {
#ifdef ENABLE_VULKAN
    if(config_.disabled)return RuntimeCapabilityLevel::Unsupported;
    PortableRuntime reference("VULKAN"); reference.configure(config_);
    return reference.queryBackwardCapability(type);
#else
    return RuntimeCapabilityLevel::Unsupported;
#endif
}

RuntimeCapabilityLevel VulkanRuntime::queryBackwardOperationCapability(
    const Layer& layer, const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs, bool training) const {
#ifdef ENABLE_VULKAN
    if(config_.disabled)return RuntimeCapabilityLevel::Unsupported;
    PortableRuntime reference("VULKAN"); reference.configure(config_);
    return reference.queryBackwardOperationCapability(layer,inputs,grad_outputs,training);
#else
    return RuntimeCapabilityLevel::Unsupported;
#endif
}

RuntimeCapabilityLevel VulkanRuntime::queryForwardOperationCapability(
    const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs,
    bool training
) const {
#ifdef ENABLE_VULKAN
    if(config_.disabled)return RuntimeCapabilityLevel::Unsupported;
    PortableRuntime reference("VULKAN"); reference.configure(config_);
    // Parallel fast paths share geometry validation, not the serial work limit.
    switch (layer.type_enum) {
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
        case LayerType::Conv2d:
            return runtimeCapabilityIsNative(reference.queryForwardGeometryCapability(layer,inputs,training))
                ? queryForwardCapability(layer.type_enum) : RuntimeCapabilityLevel::Unsupported;
        default: break;
    }
    return reference.queryForwardOperationCapability(layer,inputs,training);
#else
    return RuntimeCapabilityLevel::Unsupported;
#endif
}

bool VulkanRuntime::linearForward(
    const float* input,
    const float* weights,
    const float* bias_or_null,
    float* output,
    int batch,
    int in_f,
    int out_f
) {
#ifndef ENABLE_VULKAN
    (void)input; (void)weights; (void)bias_or_null; (void)output; (void)batch; (void)in_f; (void)out_f;
    return false;
#else
    if (!isInitialized() || !impl_) return false;
    return impl_->engine.linearForward(input, weights, bias_or_null, output, batch, in_f, out_f);
#endif
}

bool VulkanRuntime::unaryChainForwardResident(
    const float* input,
    float* output,
    int elements,
    const std::vector<LayerType>& operations
) {
#ifndef ENABLE_VULKAN
    (void)input; (void)output; (void)elements; (void)operations;
    return false;
#else
    if (!isInitialized() || !impl_ || !input || !output || elements <= 0 || operations.empty()) {
        return false;
    }
    std::vector<int> op_codes;
    op_codes.reserve(operations.size());
    for (const LayerType type : operations) {
        switch (type) {
            case LayerType::ReLU: op_codes.push_back(0); break;
            case LayerType::SiLU: op_codes.push_back(1); break;
            case LayerType::GELU: op_codes.push_back(2); break;
            case LayerType::Sigmoid: op_codes.push_back(3); break;
            case LayerType::Tanh: op_codes.push_back(4); break;
            default: return false;
        }
    }
    return impl_->engine.unaryChainForwardResident(input, output, elements, op_codes);
#endif
}

bool VulkanRuntime::forwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training
) {
#ifndef ENABLE_VULKAN
    (void)inputs; (void)outputs; (void)layer;
    return false;
#else
    if (!isInitialized() || !impl_) return false;
    if (config_.disabled) return false;
    if (!supportsForwardLayerType(layer.type_enum)) return false;

    if(layer.type_enum==LayerType::LayerNorm || layer.type_enum==LayerType::ConvTranspose2d) {
        if(!impl_->reference.initialize(config_))return false;
        return impl_->reference.forwardLayer(inputs,outputs,layer,training);
    }

    switch (layer.type_enum) {
        case LayerType::Linear: {
            if (!config_.linear_enabled) return false;
            if (inputs.empty() || !inputs[0]) return false;
            const std::vector<float>& x = *inputs[0];

            const int in_f = layer.in_features;
            const int out_f = layer.out_features;
            const int seq_len = layer.seq_len;
            const int batch = (seq_len > 0) ? seq_len : (in_f > 0 ? static_cast<int>(x.size()) / in_f : 0);
            if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;

            const size_t expected_in = static_cast<size_t>(batch) * static_cast<size_t>(in_f);
            const size_t expected_w = static_cast<size_t>(out_f) * static_cast<size_t>(in_f);
            const size_t expected_total_w = expected_w + (layer.use_bias ? static_cast<size_t>(out_f) : 0ULL);
            const long long ops = static_cast<long long>(batch) * static_cast<long long>(in_f) * static_cast<long long>(out_f);

            if (ops < std::max(0, config_.linear_min_ops) || x.size() != expected_in) return false;
            const float* w = layer.getWeights();
            if (!w || layer.getWeightsSize() < expected_total_w) return false;

            const float* b = layer.use_bias ? (w + expected_w) : nullptr;
            outputs.resize(1);
            outputs[0].assign(static_cast<size_t>(batch) * static_cast<size_t>(out_f), 0.0f);
            return linearForward(x.data(), w, b, outputs[0].data(), batch, in_f, out_f);
        }
        case LayerType::MatMul: {
            if (!config_.linear_enabled) return false;
            if (inputs.size() < 2 || !inputs[0] || !inputs[1]) return false;
            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& B = *inputs[1];

            const int M = layer.in_features;
            const int K = layer.out_features;
            const int N = layer.embed_dim;
            if (M <= 0 || K <= 0 || N <= 0) return false;

            if (A.size() != static_cast<size_t>(M) * static_cast<size_t>(K)) return false;
            if (B.size() != static_cast<size_t>(K) * static_cast<size_t>(N)) return false;

            const long long ops = static_cast<long long>(M) * static_cast<long long>(K) * static_cast<long long>(N);
            if (ops < std::max(0, config_.linear_min_ops)) return false;

            outputs.resize(1);
            outputs[0].assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
            return impl_->engine.matmulForward(A.data(), B.data(), outputs[0].data(), M, K, N);
        }
        case LayerType::BatchMatMul: {
            if (!config_.linear_enabled) return false;
            if (inputs.size() < 2 || !inputs[0] || !inputs[1]) return false;
            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& B = *inputs[1];

            const int batches = layer.seq_len;
            const int M = layer.in_features;
            const int K = layer.out_features;
            const int N = layer.embed_dim;
            if (batches <= 0 || M <= 0 || K <= 0 || N <= 0) return false;

            if (A.size() != static_cast<size_t>(batches) * static_cast<size_t>(M) * static_cast<size_t>(K)) return false;
            if (B.size() != static_cast<size_t>(batches) * static_cast<size_t>(K) * static_cast<size_t>(N)) return false;

            const long long ops = static_cast<long long>(batches) * static_cast<long long>(M) * static_cast<long long>(K) * static_cast<long long>(N);
            if (ops < std::max(0, config_.linear_min_ops)) return false;

            outputs.resize(1);
            outputs[0].assign(static_cast<size_t>(batches) * static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
            return impl_->engine.batchMatMulForward(A.data(), B.data(), outputs[0].data(), batches, M, K, N);
        }
        case LayerType::Conv2d: {
            if (!config_.conv_enabled) return false;
            if (inputs.empty() || !inputs[0]) return false;
            const std::vector<float>& x = *inputs[0];

            const int in_c = std::max(1, layer.in_channels);
            const int out_c = std::max(1, layer.out_channels);
            const int k = std::max(1, layer.get_kernel_h());
            const int stride = std::max(1, layer.get_stride_h());
            const int pad = std::max(0, layer.get_pad_h());
            const int dilation = std::max(1, (layer.dilation_h != 1) ? layer.dilation_h : layer.dilation);

            int in_h = std::max(1, layer.input_height);
            int in_w = std::max(1, layer.input_width);
            if (in_c > 0 && x.size() % static_cast<size_t>(in_c) == 0) {
                const size_t hw = x.size() / static_cast<size_t>(in_c);
                const size_t cfg_hw = static_cast<size_t>(in_h) * static_cast<size_t>(in_w);
                if (cfg_hw != hw) {
                    const int cfg_h = layer.input_height;
                    const int cfg_w = layer.input_width;
                    bool fixed = false;
                    if (cfg_h > 0 && (hw % static_cast<size_t>(cfg_h)) == 0) {
                        in_h = cfg_h;
                        in_w = static_cast<int>(hw / static_cast<size_t>(cfg_h));
                        fixed = true;
                    } else if (cfg_w > 0 && (hw % static_cast<size_t>(cfg_w)) == 0) {
                        in_w = cfg_w;
                        in_h = static_cast<int>(hw / static_cast<size_t>(cfg_w));
                        fixed = true;
                    }
                    if (!fixed) {
                        const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                        if (s > 0 && s * s == hw) {
                            in_h = static_cast<int>(s);
                            in_w = static_cast<int>(s);
                        }
                    }
                }
            }

            const int out_h = (in_h + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
            const int out_w = (in_w + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
            if (out_h <= 0 || out_w <= 0) return false;

            const long long ops = static_cast<long long>(out_c) * static_cast<long long>(in_c) * static_cast<long long>(k) * static_cast<long long>(k) * static_cast<long long>(out_h) * static_cast<long long>(out_w);
            if (ops < std::max(0, config_.conv_min_ops)) return false;

            const size_t expected_in = static_cast<size_t>(in_c) * static_cast<size_t>(in_h) * static_cast<size_t>(in_w);
            if (x.size() != expected_in) return false;

            const size_t expected_w = static_cast<size_t>(out_c) * static_cast<size_t>(in_c) * static_cast<size_t>(k) * static_cast<size_t>(k);
            const size_t expected_total_w = expected_w + (layer.use_bias ? static_cast<size_t>(out_c) : 0ULL);
            const float* w = layer.getWeights();
            if (!w || layer.getWeightsSize() < expected_total_w) return false;
            const float* b = layer.use_bias ? (w + expected_w) : nullptr;

            outputs.resize(1);
            outputs[0].assign(static_cast<size_t>(out_c) * static_cast<size_t>(out_h) * static_cast<size_t>(out_w), 0.0f);
            return impl_->engine.conv2dForward(x.data(), w, b, outputs[0].data(), in_h, in_w, in_c, out_c, k, stride, pad, dilation);
        }

        case LayerType::Add: {
            if (inputs.size() < 2 || !inputs[0] || !inputs[1]) return false;
            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& B = *inputs[1];
            if (A.size() != B.size() || A.empty()) return false;
            if (static_cast<long long>(A.size()) < std::max(0, config_.linear_min_ops)) return false;
            outputs.resize(1);
            outputs[0].assign(A.size(), 0.0f);
            return impl_->engine.addForward(A.data(), B.data(), outputs[0].data(), static_cast<int>(A.size()));
        }
        case LayerType::Multiply: {
            if (inputs.size() < 2 || !inputs[0] || !inputs[1]) return false;
            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& B = *inputs[1];
            if (A.size() != B.size() || A.empty()) return false;
            if (static_cast<long long>(A.size()) < std::max(0, config_.linear_min_ops)) return false;
            outputs.resize(1);
            outputs[0].assign(A.size(), 0.0f);
            return impl_->engine.mulForward(A.data(), B.data(), outputs[0].data(), static_cast<int>(A.size()));
        }

        case LayerType::Subtract:
        case LayerType::Divide: {
            if (inputs.size() < 2 || !inputs[0] || !inputs[1]) return false;
            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& B = *inputs[1];
            if (A.size() != B.size() || A.empty()) return false;
            if (static_cast<long long>(A.size()) < std::max(0, config_.linear_min_ops)) return false;

            int op = 0;
            if (!RuntimeLayerOps::resolveBinaryOp(layer.type_enum, op)) return false;

            outputs.resize(1);
            outputs[0].resize(A.size());
            return impl_->engine.binaryForward(A.data(),B.data(),outputs[0].data(),static_cast<int>(A.size()),op);
        }
        case LayerType::ReLU: {
            if (inputs.empty() || !inputs[0]) return false;
            const std::vector<float>& A = *inputs[0];
            if (A.empty()) return false;
            if (static_cast<long long>(A.size()) < std::max(0, config_.linear_min_ops)) return false;
            outputs.resize(1);
            outputs[0].assign(A.size(), 0.0f);
            return impl_->engine.reluForward(A.data(), outputs[0].data(), static_cast<int>(A.size()));
        }
        case LayerType::SiLU: {
            if (inputs.empty() || !inputs[0]) return false;
            const std::vector<float>& A = *inputs[0];
            if (A.empty()) return false;
            if (static_cast<long long>(A.size()) < std::max(0, config_.linear_min_ops)) return false;
            outputs.resize(1);
            outputs[0].assign(A.size(), 0.0f);
            return impl_->engine.siluForward(A.data(), outputs[0].data(), static_cast<int>(A.size()));
        }
        case LayerType::GELU: {
            if (inputs.empty() || !inputs[0]) return false;
            const std::vector<float>& A = *inputs[0];
            if (A.empty()) return false;
            if (static_cast<long long>(A.size()) < std::max(0, config_.linear_min_ops)) return false;
            outputs.resize(1);
            outputs[0].assign(A.size(), 0.0f);
            return impl_->engine.geluForward(A.data(), outputs[0].data(), static_cast<int>(A.size()));
        }
        case LayerType::Sigmoid: {
            if (inputs.empty() || !inputs[0]) return false;
            const std::vector<float>& A = *inputs[0];
            if (A.empty()) return false;
            if (static_cast<long long>(A.size()) < std::max(0, config_.linear_min_ops)) return false;
            outputs.resize(1);
            outputs[0].assign(A.size(), 0.0f);
            return impl_->engine.sigmoidForward(A.data(), outputs[0].data(), static_cast<int>(A.size()));
        }
        case LayerType::Tanh: {
            if (inputs.empty() || !inputs[0]) return false;
            const std::vector<float>& A = *inputs[0];
            if (A.empty()) return false;
            if (static_cast<long long>(A.size()) < std::max(0, config_.linear_min_ops)) return false;
            outputs.resize(1);
            outputs[0].assign(A.size(), 0.0f);
            return impl_->engine.tanhForward(A.data(), outputs[0].data(), static_cast<int>(A.size()));
        }
        case LayerType::LeakyReLU:
        case LayerType::Softplus:
        case LayerType::Mish:
        case LayerType::HardSigmoid:
        case LayerType::HardSwish: {
            if (inputs.empty() || !inputs[0]) return false;
            const std::vector<float>& A = *inputs[0];
            if (A.empty()) return false;
            if (static_cast<long long>(A.size()) < std::max(0, config_.linear_min_ops)) return false;

            int op = 0;
            float alpha = 0.01f;
            if (!RuntimeLayerOps::resolveUnaryOp(layer.type_enum, layer, op, alpha)) return false;

            outputs.resize(1);
            outputs[0].resize(A.size());
            return impl_->engine.binaryForward(A.data(),A.data(),outputs[0].data(),static_cast<int>(A.size()),10+op,alpha);
        }
        default:
            if(!impl_->reference.initialize(config_))return false;
            return impl_->reference.forwardLayer(inputs,outputs,layer,training);
    }
#endif
}

bool VulkanRuntime::backwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs,
    std::vector<std::vector<float>>& grad_inputs, Layer& layer, bool training) {
#ifdef ENABLE_VULKAN
    if(!isInitialized() || !runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
    if(!impl_->reference.initialize(config_))return false;
    return impl_->reference.backwardLayer(inputs,grad_outputs,grad_inputs,layer,training);
#else
    (void)inputs;(void)grad_outputs;(void)grad_inputs;(void)layer;(void)training;return false;
#endif
}

bool VulkanRuntime::forwardLayerWithContext(
    const std::vector<const std::vector<float>*>& inputs,std::vector<std::vector<float>>& outputs,
    const Layer& layer,bool training,const RuntimeForwardContext& context) {
#ifdef ENABLE_VULKAN
    if(layer.type_enum==LayerType::Dropout || layer.type_enum==LayerType::Dropout2d || layer.type_enum==LayerType::AlphaDropout) {
        if(!isInitialized() || config_.disabled || !impl_->reference.initialize(config_))return false;
        return impl_->reference.forwardLayerWithContext(inputs,outputs,layer,training,context);
    }
#endif
    return AbstractRuntime::forwardLayerWithContext(inputs,outputs,layer,training,context);
}

bool VulkanRuntime::optimizerStep(const RuntimeOptimizerRequest& request) {
    return supportsOptimizer() && portableOptimizer(request,name(),config_.device_index);
}

#include "runtimes/losses/LossDispatch.hpp"
bool VulkanRuntime::reconstructionLoss(const RuntimeLossGrad::Request& r,RuntimeLossGrad::LossWithGrad& out) {
    return supportsReconstructionLoss() && RuntimeLossGrad::portable(r,out,name(),config_.device_index);
}
