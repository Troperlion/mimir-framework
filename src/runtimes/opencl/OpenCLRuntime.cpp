#include "runtimes/optimizers/PortableOptimizer.hpp"
#include "runtimes/opencl/OpenCLRuntime.hpp"

#ifdef ENABLE_OPENCL
#include "runtimes/opencl/OpenCLCompute.hpp"
#endif

#include "Layers.hpp"
#include "runtimes/LayerOps.hpp"
#include "runtimes/portable/PortableRuntime.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <new>
#include <vector>

struct OpenCLRuntime::Impl {
    PortableRuntime reference{"OPENCL"};
#ifdef ENABLE_OPENCL
    OpenCLCompute::ComputeEngine engine;
#endif
    bool initialized = false;
};

bool OpenCLRuntime::initialize(const RuntimeConfig& cfg) {
    config_ = cfg;

#ifndef ENABLE_OPENCL
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

void OpenCLRuntime::shutdown() {
#ifdef ENABLE_OPENCL
    if (impl_) {
        impl_->engine.cleanup();
        impl_->initialized = false;
    }
#endif
    delete impl_;
    impl_ = nullptr;
}

bool OpenCLRuntime::isInitialized() const {
    return impl_ && impl_->initialized;
}

bool OpenCLRuntime::supportsForwardLayerType(const LayerType type) const {
#ifdef ENABLE_OPENCL
    return !config_.disabled && PortableRuntime("OPENCL").supportsForwardLayerType(type);
#else
    (void)type; return false;
#endif
}

RuntimeCapabilityLevel OpenCLRuntime::queryForwardCapability(const LayerType type) const {
#ifdef ENABLE_OPENCL
    if(config_.disabled)return RuntimeCapabilityLevel::Unsupported;
    PortableRuntime reference("OPENCL"); reference.configure(config_);
    return reference.queryForwardCapability(type);
#else
    return RuntimeCapabilityLevel::Unsupported;
#endif
}

bool OpenCLRuntime::supportsBackwardLayerType(const LayerType type) const {
#ifdef ENABLE_OPENCL
    return !config_.disabled && PortableRuntime("OPENCL").supportsBackwardLayerType(type);
#else
    (void)type; return false;
#endif
}

RuntimeCapabilityLevel OpenCLRuntime::queryBackwardCapability(const LayerType type) const {
#ifdef ENABLE_OPENCL
    if(config_.disabled)return RuntimeCapabilityLevel::Unsupported;
    PortableRuntime reference("OPENCL"); reference.configure(config_);
    return reference.queryBackwardCapability(type);
#else
    return RuntimeCapabilityLevel::Unsupported;
#endif
}

RuntimeCapabilityLevel OpenCLRuntime::queryBackwardOperationCapability(
    const Layer& layer, const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs, bool training) const {
#ifdef ENABLE_OPENCL
    if(config_.disabled)return RuntimeCapabilityLevel::Unsupported;
    PortableRuntime reference("OPENCL"); reference.configure(config_);
    return reference.queryBackwardOperationCapability(layer,inputs,grad_outputs,training);
#else
    return RuntimeCapabilityLevel::Unsupported;
#endif
}

RuntimeCapabilityLevel OpenCLRuntime::queryForwardOperationCapability(
    const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs,
    bool training
) const {
#ifdef ENABLE_OPENCL
    if(config_.disabled)return RuntimeCapabilityLevel::Unsupported;
    PortableRuntime reference("OPENCL"); reference.configure(config_);
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
            return runtimeCapabilityIsNative(reference.queryForwardGeometryCapability(layer,inputs,training))
                ? queryForwardCapability(layer.type_enum) : RuntimeCapabilityLevel::Unsupported;
        default: break;
    }
    return reference.queryForwardOperationCapability(layer,inputs,training);
#else
    return RuntimeCapabilityLevel::Unsupported;
#endif
}

bool OpenCLRuntime::linearForward(
    const float* input,
    const float* weights,
    const float* bias_or_null,
    float* output,
    int batch,
    int in_f,
    int out_f
) {
#ifndef ENABLE_OPENCL
    (void)input; (void)weights; (void)bias_or_null; (void)output; (void)batch; (void)in_f; (void)out_f;
    return false;
#else
    if (!isInitialized() || !impl_) return false;
    return impl_->engine.linearForward(input, weights, bias_or_null, output, batch, in_f, out_f);
#endif
}

bool OpenCLRuntime::forwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training
) {
    (void)training;
#ifndef ENABLE_OPENCL
    (void)inputs; (void)outputs; (void)layer;
    return false;
#else
    if (!isInitialized() || !impl_) return false;
    if (config_.disabled) return false;
    if (!supportsForwardLayerType(layer.type_enum)) return false;

    if(layer.type_enum==LayerType::LayerNorm) {
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
        case LayerType::Add:
        case LayerType::Subtract:
        case LayerType::Multiply:
        case LayerType::Divide: {
            if (!config_.linear_enabled) return false;
            if (inputs.size() < 2 || !inputs[0] || !inputs[1]) return false;
            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& B = *inputs[1];
            if (A.empty() || A.size() != B.size()) return false;
            if (static_cast<long long>(A.size()) < std::max(0, config_.linear_min_ops)) return false;

            int op = 0;
            if (!RuntimeLayerOps::resolveBinaryOp(layer.type_enum, op)) return false;

            outputs.resize(1);
            outputs[0].assign(A.size(), 0.0f);
            return impl_->engine.binaryForward(
                A.data(), B.data(), outputs[0].data(), static_cast<int>(A.size()), op);
        }
        case LayerType::ReLU:
        case LayerType::LeakyReLU:
        case LayerType::Sigmoid:
        case LayerType::Tanh:
        case LayerType::SiLU:
        case LayerType::GELU:
        case LayerType::Softplus:
        case LayerType::Mish:
        case LayerType::HardSigmoid:
        case LayerType::HardSwish: {
            if (!config_.linear_enabled) return false;
            if (inputs.empty() || !inputs[0]) return false;
            const std::vector<float>& A = *inputs[0];
            if (A.empty()) return false;
            if (static_cast<long long>(A.size()) < std::max(0, config_.linear_min_ops)) return false;

            int op = 0;
            float alpha = 0.01f;
            if (!RuntimeLayerOps::resolveUnaryOp(layer.type_enum, layer, op, alpha)) return false;

            outputs.resize(1);
            outputs[0].assign(A.size(), 0.0f);
            return impl_->engine.unaryForward(
                A.data(), outputs[0].data(), static_cast<int>(A.size()), op, alpha);
        }

        default:
            if(!impl_->reference.initialize(config_))return false;
            return impl_->reference.forwardLayer(inputs,outputs,layer,training);
    }
#endif
}


bool OpenCLRuntime::backwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs,
    std::vector<std::vector<float>>& grad_inputs, Layer& layer, bool training) {
#ifdef ENABLE_OPENCL
    if(!isInitialized() || !runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
    if(!impl_->reference.initialize(config_))return false;
    return impl_->reference.backwardLayer(inputs,grad_outputs,grad_inputs,layer,training);
#else
    (void)inputs;(void)grad_outputs;(void)grad_inputs;(void)layer;(void)training;return false;
#endif
}

bool OpenCLRuntime::forwardLayerWithContext(
    const std::vector<const std::vector<float>*>& inputs,std::vector<std::vector<float>>& outputs,
    const Layer& layer,bool training,const RuntimeForwardContext& context) {
#ifdef ENABLE_OPENCL
    if(layer.type_enum==LayerType::Dropout || layer.type_enum==LayerType::Dropout2d || layer.type_enum==LayerType::AlphaDropout) {
        if(!isInitialized() || config_.disabled || !impl_->reference.initialize(config_))return false;
        return impl_->reference.forwardLayerWithContext(inputs,outputs,layer,training,context);
    }
#endif
    return AbstractRuntime::forwardLayerWithContext(inputs,outputs,layer,training,context);
}

bool OpenCLRuntime::optimizerStep(const RuntimeOptimizerRequest& request) {
    return supportsOptimizer() && portableOptimizer(request,name(),config_.device_index);
}

#include "runtimes/losses/LossDispatch.hpp"
bool OpenCLRuntime::reconstructionLoss(const RuntimeLossGrad::Request& r,RuntimeLossGrad::LossWithGrad& out) {
    return supportsReconstructionLoss() && RuntimeLossGrad::portable(r,out,name(),config_.device_index);
}
