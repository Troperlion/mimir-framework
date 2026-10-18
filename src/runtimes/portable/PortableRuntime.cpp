#include "../ops_sequence_constant.hpp"
#include "runtimes/portable/PortableRuntime.hpp"
#include "PortableStaging.hpp"
#define ENABLE_PORTABLE_REFERENCE 1

#include "runtimes/LayerOps.hpp"
#include "runtimes/cuda/CudaOperationGeometry.hpp"
#include "runtimes/cpu/RuntimeLayerDispatch.hpp"

#include "PortableStaging.hpp"

#include "runtimes/portable/PortableNativeOps.hpp"
#include "runtimes/portable/PortableTrainingOps.hpp"
#include "runtimes/portable/PortableAttentionOps.hpp"
#include "runtimes/portable/PortableRecurrentOps.hpp"
#include "runtimes/portable/PortableDetectionOps.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <new>
#include <vector>
using namespace PortableStaging;

struct PortableRuntime::Impl {
    cublasHandle_t handle = nullptr;

    struct DeviceBuf {
        void* ptr = nullptr;
        size_t bytes = 0;

        ~DeviceBuf() { reset(); }
        DeviceBuf() = default;
        DeviceBuf(const DeviceBuf&) = delete;
        DeviceBuf& operator=(const DeviceBuf&) = delete;

        void reset() {
            if (ptr) {
                cudaFree(ptr);
                ptr = nullptr;
                bytes = 0;
            }
        }

        bool alloc(size_t nbytes) {
            reset();
            if (nbytes == 0) return true;
            if (cudaMalloc(&ptr, nbytes) != cudaSuccess) {
                ptr = nullptr;
                bytes = 0;
                return false;
            }
            bytes = nbytes;
            return true;
        }

        bool copyFromHost(const void* src, size_t nbytes) {
            if (!ptr || nbytes > bytes) return false;
            return cudaMemcpy(ptr, src, nbytes, cudaMemcpyHostToDevice) == cudaSuccess;
        }

        bool copyToHost(void* dst, size_t nbytes) {
            if (!ptr || nbytes > bytes) return false;
            return cudaMemcpy(dst, ptr, nbytes, cudaMemcpyDeviceToHost) == cudaSuccess;
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

    CachedBuf device_bias;
    CachedBuf device_ones;
    std::vector<float> host_ones;

    bool initialized = false;
};

enum class CudaUnaryOp : int {
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

enum class CudaBinaryOp : int {
    Add = 0,
    Subtract = 1,
    Multiply = 2,
    Divide = 3,
};





static bool cuda_run_unary(const std::vector<float>& input, std::vector<float>& output, CudaUnaryOp op, float alpha) {
    const int n = static_cast<int>(input.size());
    if (n <= 0) return false;

    float* d_in = nullptr;
    float* d_out = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    if (cudaMalloc(&d_in, bytes) != cudaSuccess) return false;
    if (cudaMalloc(&d_out, bytes) != cudaSuccess) {
        cudaFree(d_in);
        return false;
    }

    bool ok = true;
    if (cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) ok = false;

    if (ok) {
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        launch("cuda_unary_kernel", blocks, threads, d_in, d_out, n, static_cast<int>(op), alpha);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) ok = false;
    }

    if (ok) {
        output.resize(static_cast<size_t>(n));
        if (cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) ok = false;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return ok;
}

static bool cuda_run_binary(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& output, CudaBinaryOp op) {
    const int n = static_cast<int>(a.size());
    if (n <= 0 || a.size() != b.size()) return false;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_out = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    if (cudaMalloc(&d_a, bytes) != cudaSuccess) return false;
    if (cudaMalloc(&d_b, bytes) != cudaSuccess) {
        cudaFree(d_a);
        return false;
    }
    if (cudaMalloc(&d_out, bytes) != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        return false;
    }

    bool ok = true;
    if (cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
    if (ok && cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) ok = false;

    if (ok) {
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        launch("cuda_binary_kernel", blocks, threads, d_a, d_b, d_out, n, static_cast<int>(op));
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) ok = false;
    }

    if (ok) {
        output.resize(static_cast<size_t>(n));
        if (cudaMemcpy(output.data(), d_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) ok = false;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return ok;
}





static bool cuda_run_unary_backward(
    const std::vector<float>& in,
    const std::vector<float>& grad_out,
    std::vector<float>& grad_in,
    CudaUnaryOp op,
    float alpha
) {
    const int n = static_cast<int>(in.size());
    if (n <= 0 || grad_out.size() != in.size()) return false;

    float* d_in = nullptr;
    float* d_go = nullptr;
    float* d_gi = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    if (cudaMalloc(&d_in, bytes) != cudaSuccess) return false;
    if (cudaMalloc(&d_go, bytes) != cudaSuccess) { cudaFree(d_in); return false; }
    if (cudaMalloc(&d_gi, bytes) != cudaSuccess) { cudaFree(d_in); cudaFree(d_go); return false; }

    bool ok = true;
    if (cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
    if (ok && cudaMemcpy(d_go, grad_out.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) ok = false;

    if (ok) {
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        launch("cuda_unary_backward_kernel", blocks, threads, d_in, d_go, d_gi, n, static_cast<int>(op), alpha);
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) ok = false;
    }

    if (ok) {
        grad_in.resize(static_cast<size_t>(n));
        if (cudaMemcpy(grad_in.data(), d_gi, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) ok = false;
    }

    cudaFree(d_in);
    cudaFree(d_go);
    cudaFree(d_gi);
    return ok;
}

static bool cuda_run_binary_backward(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& grad_out,
    std::vector<float>& grad_a,
    std::vector<float>& grad_b,
    CudaBinaryOp op
) {
    const int n = static_cast<int>(a.size());
    if (n <= 0 || b.size() != a.size() || grad_out.size() != a.size()) return false;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_go = nullptr;
    float* d_ga = nullptr;
    float* d_gb = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    if (cudaMalloc(&d_a, bytes) != cudaSuccess) return false;
    if (cudaMalloc(&d_b, bytes) != cudaSuccess) { cudaFree(d_a); return false; }
    if (cudaMalloc(&d_go, bytes) != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); return false; }
    if (cudaMalloc(&d_ga, bytes) != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); cudaFree(d_go); return false; }
    if (cudaMalloc(&d_gb, bytes) != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); cudaFree(d_go); cudaFree(d_ga); return false; }

    bool ok = true;
    if (cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
    if (ok && cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
    if (ok && cudaMemcpy(d_go, grad_out.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) ok = false;

    if (ok) {
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        launch("cuda_binary_backward_kernel", blocks, threads, d_a, d_b, d_go, d_ga, d_gb, n, static_cast<int>(op));
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) ok = false;
    }

    if (ok) {
        grad_a.resize(static_cast<size_t>(n));
        grad_b.resize(static_cast<size_t>(n));
        if (cudaMemcpy(grad_a.data(), d_ga, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) ok = false;
        if (ok && cudaMemcpy(grad_b.data(), d_gb, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) ok = false;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_go);
    cudaFree(d_ga);
    cudaFree(d_gb);
    return ok;
}

bool PortableRuntime::initialize(const RuntimeConfig& cfg) {
    DeviceScope device_scope(backend_,cfg.device_index);
    config_ = cfg;

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
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        shutdown();
        return false;
    }

    int device_index = config_.device_index;
    if (device_index < 0 || device_index >= device_count) {
        device_index = 0;
    }

    if (cudaSetDevice(device_index) != cudaSuccess) {
        shutdown();
        return false;
    }

    if (cublasCreate(&impl_->handle) != CUBLAS_STATUS_SUCCESS) {
        shutdown();
        return false;
    }

    (void)cublasSetStream(impl_->handle, nullptr);

    impl_->initialized = true;
    return true;
}

void PortableRuntime::shutdown() {
    if (impl_) {
        if (impl_->handle) {
            cublasDestroy(impl_->handle);
            impl_->handle = nullptr;
        }
        impl_->device_bias.buf.reset();
        impl_->device_ones.buf.reset();
        impl_->host_ones.clear();
        impl_->initialized = false;
    }

    delete impl_;
    impl_ = nullptr;
}

bool PortableRuntime::isInitialized() const {
    return impl_ && impl_->initialized;
}

bool PortableRuntime::supportsForwardLayerType(const LayerType type) const {
    switch (config_.disabled) {
        case true:
            return false;
        case false:
            break;
    }

    switch (type) {
        case LayerType::UNKNOWN:
        case LayerType::Lambda:
            return false;
        default:
            return RuntimeLayerDispatch::cpu_supports_forward_layer_type(type);
    }
}

bool PortableRuntime::supportsBackwardLayerType(const LayerType type) const {
    switch (config_.disabled) {
        case true:
            return false;
        case false:
            break;
    }

    if(type==LayerType::NMS)return true;
    switch (type) {
        case LayerType::UNKNOWN:
            return false;
        default:
            return RuntimeLayerDispatch::cpu_supports_backward_layer_type(type);
    }
}

RuntimeCapabilityLevel PortableRuntime::queryForwardCapability(const LayerType type) const {
    if (!supportsForwardLayerType(type)) return RuntimeCapabilityLevel::Unsupported;
    if(type==LayerType::NMS)return RuntimeCapabilityLevel::Native;
    if(CudaOperationGeometry::poolType(type) || CudaOperationGeometry::movementType(type) || CudaOperationGeometry::projectionType(type) || CudaOperationGeometry::extendedConvType(type) || CudaOperationGeometry::stochasticType(type) || CudaOperationGeometry::batchNormType(type) || CudaOperationGeometry::attentionType(type) || CudaOperationGeometry::recurrentType(type))return RuntimeCapabilityLevel::Native;
    switch (type) {
        case LayerType::Linear:
        case LayerType::MatMul:
        case LayerType::BatchMatMul:
        case LayerType::Conv2d:
        case LayerType::LayerNorm:
        case LayerType::RMSNorm:
        case LayerType::SelfAttention:
        case LayerType::MultiHeadAttention:
            return RuntimeCapabilityLevel::Native;
        case LayerType::InstanceNorm2d:
        case LayerType::GEGLU:
        case LayerType::GroupNorm:
        case LayerType::UpsampleNearest:
        case LayerType::UpsampleBicubic:
        case LayerType::UpsampleBilinear:
        case LayerType::Softmax:
        case LayerType::LogSoftmax:
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
}

RuntimeCapabilityLevel PortableRuntime::queryBackwardCapability(const LayerType type) const {
    if (!supportsBackwardLayerType(type)) return RuntimeCapabilityLevel::Unsupported;
    if(type==LayerType::Lambda)return RuntimeCapabilityLevel::Native;
    if(type==LayerType::NMS)return RuntimeCapabilityLevel::Native;
    if(CudaOperationGeometry::poolType(type) || CudaOperationGeometry::movementType(type) || CudaOperationGeometry::projectionType(type) || CudaOperationGeometry::extendedConvType(type) || CudaOperationGeometry::stochasticType(type) || CudaOperationGeometry::batchNormType(type) || CudaOperationGeometry::attentionType(type) || CudaOperationGeometry::recurrentType(type))return RuntimeCapabilityLevel::Native;
    switch (type) {
        case LayerType::LayerNorm:
        case LayerType::RMSNorm:
        case LayerType::Conv2d:
        case LayerType::InstanceNorm2d:
        case LayerType::GEGLU:
        case LayerType::GroupNorm:
        case LayerType::UpsampleNearest:
        case LayerType::UpsampleBicubic:
        case LayerType::UpsampleBilinear:
        case LayerType::Softmax:
        case LayerType::LogSoftmax:
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
}

RuntimeCapabilityLevel PortableRuntime::queryForwardOperationCapability(
    const Layer& layer, const std::vector<const std::vector<float>*>& inputs, bool training) const {
    if(layer.type_enum==LayerType::Constant && SequenceConstant::enabled(layer))return RuntimeCapabilityLevel::HostFallback;
    if (!referenceInputsFit(layer.getWeightsSize(), inputs)) return RuntimeCapabilityLevel::Unsupported;
    return queryForwardGeometryCapability(layer, inputs, training);
}

RuntimeCapabilityLevel PortableRuntime::queryForwardGeometryCapability(
    const Layer& layer, const std::vector<const std::vector<float>*>& inputs, bool training) const {
    if(layer.type_enum==LayerType::NMS){
        if(config_.disabled || inputs.size()<2 || !inputs[0] || !inputs[1] || inputs[0]->size()%4 || inputs[0]->size()>INT_MAX || inputs[1]->size()>(1U<<24) || inputs[1]->size()!=inputs[0]->size()/4 ||
            (inputs.size()>2 && inputs[2] && inputs[2]->size()!=inputs[1]->size()) || !std::isfinite(layer.nms_iou_threshold) || layer.nms_iou_threshold<0 || layer.nms_iou_threshold>1 ||
            !std::isfinite(layer.nms_score_threshold) || layer.nms_max_detections<0)return RuntimeCapabilityLevel::Unsupported;
        return queryForwardCapability(layer.type_enum);
    }
    (void)training;
    if (config_.disabled || inputs.empty() || !inputs[0] || inputs[0]->empty() || inputs[0]->size()>INT_MAX)
        return RuntimeCapabilityLevel::Unsupported;
    if(CudaOperationGeometry::recurrentType(layer.type_enum)){
        CudaOperationGeometry::Recurrent g;return g.valid(layer,inputs[0]->size())?queryForwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
    }
    if(CudaOperationGeometry::attentionType(layer.type_enum)){
        CudaOperationGeometry::Attention g;return config_.attention_enabled && g.valid(layer,inputs) && 2LL*g.q*g.k*g.ed>=std::max(0,config_.attention_min_ops)?queryForwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
    }
    if(CudaOperationGeometry::extendedConvType(layer.type_enum)){
        CudaOperationGeometry::ExtendedConv g;
        return config_.conv_enabled && g.valid(layer,inputs[0]->size()) && static_cast<long long>(g.outputSize())*g.kh*g.kw*(g.mode==2?1:g.ic)>=std::max(0,config_.conv_min_ops)?queryForwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
    }
    if(CudaOperationGeometry::batchNormType(layer.type_enum)){
        CudaOperationGeometry::BatchNorm g;return config_.norm_enabled && inputs[0]->size()>=static_cast<size_t>(std::max(0,config_.norm_min_elements)) && g.valid(layer,inputs[0]->size())?queryForwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
    }
    if(CudaOperationGeometry::stochasticType(layer.type_enum)){
        if(layer.type_enum==LayerType::Reparameterize && (inputs.size()<2 || !inputs[1] || inputs[1]->size()!=inputs[0]->size()))return RuntimeCapabilityLevel::Unsupported;
        if(layer.type_enum!=LayerType::Reparameterize && !std::isfinite(layer.dropout_p))return RuntimeCapabilityLevel::Unsupported;
        return queryForwardCapability(layer.type_enum);
    }
    if(layer.type_enum==LayerType::GEGLU){return layer.seq_len>0 && layer.out_features>0 && inputs[0]->size()==2ULL*layer.seq_len*layer.out_features?queryForwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;}
    if(CudaOperationGeometry::projectionType(layer.type_enum)){
        CudaOperationGeometry::Projection g;return g.valid(layer,inputs)?queryForwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
    }
    if(layer.type_enum==LayerType::LayerNorm || layer.type_enum==LayerType::RMSNorm){
        const size_t n=inputs[0]->size(),width=layer.in_features>0?static_cast<size_t>(layer.in_features):n;
        if(!config_.norm_enabled || n<static_cast<size_t>(std::max(0,config_.norm_min_elements)) || !width || n%width ||
            (layer.affine && (!layer.getWeights() || layer.getWeightsSize()<width*(layer.type_enum==LayerType::LayerNorm && layer.use_bias?2ULL:1ULL))) || !(layer.eps>0.f) || (layer.type_enum==LayerType::RMSNorm && layer.in_features<=0 && (layer.embed_dim>0 || !layer.shape.empty() || !layer.target_shape.empty())))return RuntimeCapabilityLevel::Unsupported;
        return queryForwardCapability(layer.type_enum);
    }
    if(CudaOperationGeometry::poolType(layer.type_enum)) {
        CudaOperationGeometry::Pool g;return g.valid(layer,inputs[0]->size())?queryForwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
    }
    if(CudaOperationGeometry::movementType(layer.type_enum)) {
        CudaOperationGeometry::Movement g;return g.valid(layer,inputs)?queryForwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
    }
    const size_t count=inputs[0]->size();
    bool valid=true;
    switch (layer.type_enum) {
        case LayerType::InstanceNorm2d:
        case LayerType::GroupNorm: {
            CudaOperationGeometry::Norm shape;
            valid=config_.norm_enabled && count>=static_cast<size_t>(std::max(0,config_.norm_min_elements)) && shape.valid(layer,count);
            break;
        }
        case LayerType::UpsampleNearest:
        case LayerType::UpsampleBicubic:
        case LayerType::UpsampleBilinear: {
            CudaOperationGeometry::Resize shape; valid=shape.valid(layer,count); break;
        }
        case LayerType::Softmax:
        case LayerType::LogSoftmax:
            valid=config_.linear_enabled && count>=static_cast<size_t>(std::max(0,config_.linear_min_ops)); break;
        case LayerType::Conv2d: {
            CudaOperationGeometry::Conv shape; valid=shape.valid(layer,count); break;
        }
        default: break;
    }
    return valid ? queryConfiguredForwardOperationCapability(layer, inputs, true) : RuntimeCapabilityLevel::Unsupported;
}


namespace {
bool cuda_axis_geometry(const std::vector<float>& x, const Layer& layer, int& size, int& inner) {
    if (x.empty() || x.size()>INT_MAX) return false;
    const auto shape=LayerOps::infer_shape_for_axis_op(x,layer);
    if (shape.empty()) return false;
    int axis=layer.axis;
    if (axis<0) axis+=static_cast<int>(shape.size());
    if (axis<0 || axis>=static_cast<int>(shape.size())) axis=static_cast<int>(shape.size())-1;
    size=shape[axis]; long long stride=1;
    for (size_t i=axis+1;i<shape.size();++i) { if (shape[i]<=0 || stride>INT_MAX/shape[i]) return false; stride*=shape[i]; }
    inner=stride;
    return size>0 && static_cast<long long>(size)*inner<=INT_MAX && x.size()%(static_cast<size_t>(size)*inner)==0;
}
void cuda_accumulate_parameters(Layer& layer, const std::vector<float>& weights, const std::vector<float>& bias) {
    if (layer.grad_weights.size()!=layer.getWeightsSize()) layer.grad_weights.assign(layer.getWeightsSize(),0.f);
    for (size_t i=0;i<weights.size();++i) layer.grad_weights[i]+=weights[i];
    if (!bias.empty()) {
        if (layer.grad_bias.size()!=bias.size()) layer.grad_bias.assign(bias.size(),0.f);
        const size_t offset=layer.getWeightsSize()-bias.size();
        for (size_t i=0;i<bias.size();++i) { layer.grad_bias[i]+=bias[i]; layer.grad_weights[offset+i]+=bias[i]; }
    }
}
}

RuntimeCapabilityLevel PortableRuntime::queryBackwardOperationCapability(
    const Layer& layer, const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs, bool training) const {
    if(layer.type_enum==LayerType::Constant && SequenceConstant::enabled(layer))return RuntimeCapabilityLevel::HostFallback;
    if (!referenceInputsFit(layer.getWeightsSize(), inputs) || !referenceInputsFit(0, grad_outputs)) return RuntimeCapabilityLevel::Unsupported;
    if(layer.type_enum==LayerType::Lambda) {
        return !config_.disabled && inputs.size()>=1 && inputs[0] && grad_outputs.size()==1 && grad_outputs[0] && inputs[0]->size()==grad_outputs[0]->size()?RuntimeCapabilityLevel::Native:RuntimeCapabilityLevel::Unsupported;
    }

    if(layer.type_enum==LayerType::NMS){
        return inputs.size()>=2 && inputs.size()<=3 && grad_outputs.size()==1 && grad_outputs[0] &&
            runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training)) && grad_outputs[0]->size()<=inputs[1]->size()
            ?RuntimeCapabilityLevel::Native:RuntimeCapabilityLevel::Unsupported;
    }

    if (inputs.empty() || !inputs[0] || grad_outputs.empty() || !grad_outputs[0]) return RuntimeCapabilityLevel::Unsupported;
    if(CudaOperationGeometry::recurrentType(layer.type_enum)){
        CudaOperationGeometry::Recurrent g;return !config_.disabled && grad_outputs.size()==1 && g.valid(layer,inputs[0]->size()) && grad_outputs[0]->size()==g.outputSize()?queryBackwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
    }
    if(CudaOperationGeometry::attentionType(layer.type_enum)){
        CudaOperationGeometry::Attention g;return grad_outputs.size()==1 && g.valid(layer,inputs,true) && grad_outputs[0]->size()==g.output && runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training))?queryBackwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
    }
    if(CudaOperationGeometry::extendedConvType(layer.type_enum)){
        CudaOperationGeometry::ExtendedConv g;return grad_outputs.size()==1 && g.valid(layer,inputs[0]->size()) && grad_outputs[0]->size()==g.outputSize() && runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training))?queryBackwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
    }
    if(CudaOperationGeometry::batchNormType(layer.type_enum) || CudaOperationGeometry::stochasticType(layer.type_enum)){
        if(grad_outputs.size()!=1 || grad_outputs[0]->size()!=inputs[0]->size() || !runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training)))return RuntimeCapabilityLevel::Unsupported;
        return queryBackwardCapability(layer.type_enum);
    }
    if(layer.type_enum==LayerType::GEGLU){return grad_outputs.size()==1 && grad_outputs[0]->size()*2==inputs[0]->size() && runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training))?queryBackwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;}
    if(CudaOperationGeometry::projectionType(layer.type_enum)){
        CudaOperationGeometry::Projection g;return !config_.disabled && g.valid(layer,inputs) && grad_outputs.size()==1 && grad_outputs[0]->size()==g.output?queryBackwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
    }
    if(CudaOperationGeometry::poolType(layer.type_enum)) {
        if(layer.type_enum==LayerType::MaxPool2d && (layer.get_kernel_w()!=layer.get_kernel_h() || layer.get_stride_w()!=layer.get_stride_h() || layer.get_pad_w()!=layer.get_pad_h()))return RuntimeCapabilityLevel::Unsupported;
        CudaOperationGeometry::Pool g;return !config_.disabled && g.valid(layer,inputs[0]->size()) && grad_outputs.size()==1 && grad_outputs[0]->size()==g.outputSize()?queryBackwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
    }
    if(CudaOperationGeometry::movementType(layer.type_enum)) {
        CudaOperationGeometry::Movement g;if(config_.disabled || !g.valid(layer,inputs) || grad_outputs.size()!=g.sizes.size())return RuntimeCapabilityLevel::Unsupported;
        for(size_t i=0;i<g.sizes.size();++i)if(!grad_outputs[i] || grad_outputs[i]->size()!=g.sizes[i])return RuntimeCapabilityLevel::Unsupported;
        return queryBackwardCapability(layer.type_enum);
    }
    if(layer.type_enum==LayerType::LayerNorm || layer.type_enum==LayerType::RMSNorm){
        return grad_outputs.size()==1 && grad_outputs[0]->size()==inputs[0]->size() && runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training))?queryBackwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
    }
    switch (layer.type_enum) {
        case LayerType::InstanceNorm2d:
        case LayerType::GroupNorm:
        case LayerType::Softmax:
        case LayerType::LogSoftmax:
            if (grad_outputs[0]->size()!=inputs[0]->size()) return RuntimeCapabilityLevel::Unsupported;
            break;
        case LayerType::UpsampleNearest:
        case LayerType::UpsampleBicubic:
        case LayerType::UpsampleBilinear: {
            CudaOperationGeometry::Resize shape;
            if (!shape.valid(layer,inputs[0]->size()) || grad_outputs[0]->size()!=shape.outputSize()) return RuntimeCapabilityLevel::Unsupported;
            break;
        }
        case LayerType::Conv2d: {
            CudaOperationGeometry::Conv shape;
            if (!shape.valid(layer,inputs[0]->size()) || grad_outputs[0]->size()!=shape.outputSize()) return RuntimeCapabilityLevel::Unsupported;
            break;
        }
        default: {
            if(!runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training)) || grad_outputs.size()!=1)return RuntimeCapabilityLevel::Unsupported;
            size_t expected=inputs[0]->size();
            if(layer.type_enum==LayerType::Linear)expected=inputs[0]->size()/static_cast<size_t>(layer.in_features)*static_cast<size_t>(layer.out_features);
            if(layer.type_enum==LayerType::MatMul || layer.type_enum==LayerType::BatchMatMul)expected=static_cast<size_t>(layer.in_features)*layer.embed_dim*(layer.type_enum==LayerType::BatchMatMul?layer.seq_len:1);
            return grad_outputs[0]->size()==expected?queryBackwardCapability(layer.type_enum):RuntimeCapabilityLevel::Unsupported;
        }
    }
    return runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training))
        ? queryBackwardCapability(layer.type_enum) : RuntimeCapabilityLevel::Unsupported;
}
bool PortableRuntime::linearForward(
    const float* input,
    const float* weights,
    const float* bias_or_null,
    float* output,
    int batch,
    int in_f,
    int out_f
) {
    DeviceScope device_scope(backend_,config_.device_index);
    if (!isInitialized() || !impl_->handle) return false;
    if (!input || !weights || !output) return false;
    if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;

    const size_t bytes_in = static_cast<size_t>(batch) * static_cast<size_t>(in_f) * sizeof(float);
    const size_t bytes_w = static_cast<size_t>(out_f) * static_cast<size_t>(in_f) * sizeof(float);
    const size_t bytes_out = static_cast<size_t>(batch) * static_cast<size_t>(out_f) * sizeof(float);

    Impl::DeviceBuf d_in;
    Impl::DeviceBuf d_w;
    Impl::DeviceBuf d_out;

    if (!d_in.alloc(bytes_in)) return false;
    if (!d_w.alloc(bytes_w)) return false;
    if (!d_out.alloc(bytes_out)) return false;

    if (!d_in.copyFromHost(input, bytes_in)) return false;
    if (!d_w.copyFromHost(weights, bytes_w)) return false;

    // GEMM 1: output = input * weights^T
    // cublas column-major: C(out_f x batch) = A^T(out_f x in_f) * B(in_f x batch)
    const float alpha = 1.0f;
    const float beta0 = 0.0f;

    if (cublasSgemm(
            impl_->handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
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
        ) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    if (bias_or_null) {
        // output += bias[:,1] * ones[1,batch]
        if (!impl_->device_bias.ensure(static_cast<size_t>(out_f))) return false;
        if (!impl_->device_bias.buf.copyFromHost(bias_or_null, static_cast<size_t>(out_f) * sizeof(float))) return false;

        if (!impl_->device_ones.ensure(static_cast<size_t>(batch))) return false;
        if (static_cast<int>(impl_->host_ones.size()) != batch) {
            impl_->host_ones.assign(static_cast<size_t>(batch), 1.0f);
        }
        if (!impl_->device_ones.buf.copyFromHost(impl_->host_ones.data(), static_cast<size_t>(batch) * sizeof(float))) return false;

        const float beta1 = 1.0f;
        const int k = 1;
        if (cublasSgemm(
                impl_->handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
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
            ) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        return false;
    }

    return d_out.copyToHost(output, bytes_out);
}

bool PortableRuntime::forwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training
) {
    if (!referenceInputsFit(layer.getWeightsSize(), inputs)) return false;
    DeviceScope device_scope(backend_,config_.device_index);
    (void)training;
    if (!isInitialized()) return false;
    if (config_.disabled) return false;
    if (!supportsForwardLayerType(layer.type_enum)) return false;

    if(layer.type_enum==LayerType::NMS){
        if(!runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training)))return false;
        return PortableNativeOps::nms(inputs,outputs,layer);
    }
    if(CudaOperationGeometry::recurrentType(layer.type_enum)){
        if(!runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training)))return false;
        CudaOperationGeometry::Recurrent g;g.valid(layer,inputs[0]->size());outputs.resize(1);return PortableNativeOps::recurrent(*inputs[0],outputs[0],layer,g);
    }
    if(CudaOperationGeometry::attentionType(layer.type_enum)){
        if(!runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training)))return false;
        CudaOperationGeometry::Attention g;g.valid(layer,inputs);return PortableNativeOps::attention(inputs,outputs,layer,g,nullptr,nullptr,impl_->handle);
    }
    if(CudaOperationGeometry::extendedConvType(layer.type_enum)){
        if(!runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training)))return false;
        CudaOperationGeometry::ExtendedConv g;g.valid(layer,inputs[0]->size());outputs.resize(1);return PortableNativeOps::extended_conv(*inputs[0],layer,g,outputs[0]);
    }
    if(CudaOperationGeometry::batchNormType(layer.type_enum)){
        if(!runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training)))return false;
        CudaOperationGeometry::BatchNorm g;g.valid(layer,inputs[0]->size());outputs.resize(1);return PortableNativeOps::batch_norm(*inputs[0],outputs[0],layer,g,training);
    }
    if(CudaOperationGeometry::stochasticType(layer.type_enum)){
        if(!runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training)))return false;
        if(layer.type_enum==LayerType::Reparameterize)return PortableNativeOps::reparameterize(inputs,outputs,training);
        outputs.resize(1);return PortableNativeOps::dropout(*inputs[0],outputs[0],layer,training);
    }
    if(layer.type_enum==LayerType::GEGLU || layer.type_enum==LayerType::UpsampleBicubic){
        if(!runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training)))return false;
        CudaOperationGeometry::Resize g;if(layer.type_enum==LayerType::UpsampleBicubic)g.valid(layer,inputs[0]->size());
        outputs.resize(1);return layer.type_enum==LayerType::GEGLU?PortableNativeOps::geglu(*inputs[0],outputs[0],layer.out_features):PortableNativeOps::bicubic(*inputs[0],outputs[0],g);
    }
    if(CudaOperationGeometry::projectionType(layer.type_enum)){
        CudaOperationGeometry::Projection g;if(!g.valid(layer,inputs))return false;return PortableNativeOps::projection(inputs,outputs,layer,g);
    }
    if(layer.type_enum==LayerType::LayerNorm || layer.type_enum==LayerType::RMSNorm){
        if(!runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training)))return false;
        outputs.resize(1);return PortableNativeOps::row_norm(*inputs[0],layer,outputs[0]);
    }
    if(CudaOperationGeometry::poolType(layer.type_enum)) {
        if(inputs.empty() || !inputs[0])return false;
        CudaOperationGeometry::Pool g;if(!g.valid(layer,inputs[0]->size()))return false;
        outputs.resize(1);return PortableNativeOps::pool(*inputs[0],outputs[0],g);
    }
    if(CudaOperationGeometry::movementType(layer.type_enum)) {
        CudaOperationGeometry::Movement g;if(!g.valid(layer,inputs))return false;
        if(layer.type_enum==LayerType::Constant){std::vector<float> weights(layer.getWeights(),layer.getWeights()+layer.getWeightsSize());return PortableNativeOps::movement({&weights},outputs,layer,g);}
        return PortableNativeOps::movement(inputs,outputs,layer,g);
    }
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
            // Fallback pour les chemins non encore couverts en backward GPU.
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

        case LayerType::MatMul:
            if (!config_.linear_enabled) break;
            do {
            if (inputs.size() < 2 || !inputs[0] || !inputs[1]) break;
            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& B = *inputs[1];

            const int M = layer.in_features;
            const int K = layer.out_features;
            const int N = layer.embed_dim;
            if (M <= 0 || K <= 0 || N <= 0) break;

            const int batches = 1;

            const size_t a_batch_elems = static_cast<size_t>(M) * static_cast<size_t>(K);
            const size_t b_batch_elems = static_cast<size_t>(K) * static_cast<size_t>(N);
            const size_t c_batch_elems = static_cast<size_t>(M) * static_cast<size_t>(N);

            if (A.size() != static_cast<size_t>(batches) * a_batch_elems) break;
            if (B.size() != static_cast<size_t>(batches) * b_batch_elems) break;

            const long long ops = static_cast<long long>(batches) * static_cast<long long>(M) *
                                  static_cast<long long>(K) * static_cast<long long>(N);
            if (ops < static_cast<long long>(std::max(0, config_.linear_min_ops))) break;

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
                if (cublasSgemm(
                        impl_->handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
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
                    ) != CUBLAS_STATUS_SUCCESS) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
            if (cudaDeviceSynchronize() != cudaSuccess) break;

            outputs.resize(1);
            outputs[0].resize(static_cast<size_t>(batches) * c_batch_elems);
            if (!d_c.copyToHost(outputs[0].data(), static_cast<size_t>(batches) * c_batch_elems * sizeof(float))) break;
            return true;
            } while (false);
            break;

        case LayerType::BatchMatMul:
            if (!config_.linear_enabled) break;
            do {
            if (inputs.size() < 2 || !inputs[0] || !inputs[1]) break;
            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& B = *inputs[1];

            const int M = layer.in_features;
            const int K = layer.out_features;
            const int N = layer.embed_dim;
            const int batches = layer.seq_len;
            if (M <= 0 || K <= 0 || N <= 0 || batches <= 0) break;

            const size_t a_batch_elems = static_cast<size_t>(M) * static_cast<size_t>(K);
            const size_t b_batch_elems = static_cast<size_t>(K) * static_cast<size_t>(N);
            const size_t c_batch_elems = static_cast<size_t>(M) * static_cast<size_t>(N);

            if (A.size() != static_cast<size_t>(batches) * a_batch_elems) break;
            if (B.size() != static_cast<size_t>(batches) * b_batch_elems) break;

            const long long ops = static_cast<long long>(batches) * static_cast<long long>(M) *
                                  static_cast<long long>(K) * static_cast<long long>(N);
            if (ops < static_cast<long long>(std::max(0, config_.linear_min_ops))) break;

            Impl::DeviceBuf d_a, d_b, d_c;
            if (!d_a.alloc(static_cast<size_t>(batches) * a_batch_elems * sizeof(float))) break;
            if (!d_b.alloc(static_cast<size_t>(batches) * b_batch_elems * sizeof(float))) break;
            if (!d_c.alloc(static_cast<size_t>(batches) * c_batch_elems * sizeof(float))) break;
            if (!d_a.copyFromHost(A.data(), static_cast<size_t>(batches) * a_batch_elems * sizeof(float))) break;
            if (!d_b.copyFromHost(B.data(), static_cast<size_t>(batches) * b_batch_elems * sizeof(float))) break;

            const float alpha = 1.0f;
            const float beta = 0.0f;
            bool ok = true;
            for (int bi = 0; bi < batches; ++bi) {
                const float* a_ptr = reinterpret_cast<const float*>(d_a.ptr) + static_cast<size_t>(bi) * a_batch_elems;
                const float* b_ptr = reinterpret_cast<const float*>(d_b.ptr) + static_cast<size_t>(bi) * b_batch_elems;
                float* c_ptr = reinterpret_cast<float*>(d_c.ptr) + static_cast<size_t>(bi) * c_batch_elems;
                if (cublasSgemm(
                        impl_->handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
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
                    ) != CUBLAS_STATUS_SUCCESS) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
            if (cudaDeviceSynchronize() != cudaSuccess) break;

            outputs.resize(1);
            outputs[0].resize(static_cast<size_t>(batches) * c_batch_elems);
            if (!d_c.copyToHost(outputs[0].data(), static_cast<size_t>(batches) * c_batch_elems * sizeof(float))) break;
            return true;
            } while (false);
            break;


        case LayerType::InstanceNorm2d:
        case LayerType::GroupNorm: {
            if (!runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training))) return false;
            const auto& x=*inputs[0]; CudaOperationGeometry::Norm shape; if (!shape.valid(layer,x.size())) return false;
            std::vector<float> result;
            const float* weights=layer.affine?layer.getWeights():nullptr;
            if (!PortableNativeOps::groupnorm(x,weights,layer.getWeightsSize(),shape.channels,shape.spatial,shape.groups,layer.eps,
                                         layer.affine && layer.use_bias,result)) return false;
            outputs={std::move(result)}; return true;
        }
        case LayerType::UpsampleNearest:
        case LayerType::UpsampleBicubic:
        case LayerType::UpsampleBilinear: {
            if (!runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training))) return false;
            const auto& x=*inputs[0]; CudaOperationGeometry::Resize shape; if (!shape.valid(layer,x.size())) return false;
            std::vector<float> result;
            if (!PortableNativeOps::resize(x,result,shape.c,shape.ih,shape.iw,shape.oh,shape.ow,shape.sh,shape.sw,
                                      layer.type_enum==LayerType::UpsampleBilinear,false)) return false;
            outputs={std::move(result)}; return true;
        }
        case LayerType::Softmax:
        case LayerType::LogSoftmax: {
            if (!runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training))) return false;
            const auto& x=*inputs[0]; int size=0,inner=0; if (!cuda_axis_geometry(x,layer,size,inner)) return false;
            std::vector<float> result;
            if (!PortableNativeOps::softmax(x,result,size,inner,layer.type_enum==LayerType::LogSoftmax)) return false;
            outputs={std::move(result)}; return true;
        }

        case LayerType::Conv2d:
            if (!config_.conv_enabled) return false;
            do {
            if (inputs.empty() || !inputs[0]) return false;
            const std::vector<float>& xc = *inputs[0];
            CudaOperationGeometry::Conv geometry;
            if (!geometry.valid(layer, xc.size())) return false;
            const int C_in  = std::max(1, layer.in_channels);
            const int C_out = std::max(1, layer.out_channels);
            const int k     = std::max(1, layer.kernel_h > 0 ? layer.kernel_h : layer.get_kernel_h());
            const int str   = std::max(1, layer.get_stride_h());
            const int pad   = std::max(0, layer.get_pad_h());
            const int dil   = std::max(1, layer.dilation_h != 1 ? layer.dilation_h : layer.dilation);
            const int H     = std::max(1, layer.input_height);
            const int W     = std::max(1, layer.input_width);
            const int H_out = (H + 2*pad - dil*(k-1) - 1) / str + 1;
            const int W_out = (W + 2*pad - dil*(k-1) - 1) / str + 1;
            if (H_out <= 0 || W_out <= 0) return false;

            const int K  = C_in * k * k;
            const int HW = H_out * W_out;
            const long long ops = static_cast<long long>(C_out) * K * HW;
            if (ops < static_cast<long long>(config_.conv_min_ops)) return false;
            if (static_cast<int>(xc.size()) != C_in * H * W) return false;

            const size_t w_k = static_cast<size_t>(C_out) * K;
            const float* wc = layer.getWeights();
            if (!wc || layer.getWeightsSize() < w_k + (layer.use_bias ? C_out : 0u)) return false;

            // Im2col runs on the device too: col[K, HW].
            const size_t col_sz = static_cast<size_t>(K) * HW;
            Impl::DeviceBuf d_input, d_col, d_w, d_out;
            if (!d_input.alloc(xc.size() * sizeof(float))) return false;
            if (!d_input.copyFromHost(xc.data(), xc.size() * sizeof(float))) return false;
            if (!d_col.alloc(col_sz * sizeof(float))) return false;
            if (!d_w.alloc(w_k * sizeof(float))) return false;
            if (!d_out.alloc(static_cast<size_t>(C_out) * HW * sizeof(float))) return false;
            PortableNativeOps::launch("im2col", (col_sz + PortableNativeOps::threads - 1) / PortableNativeOps::threads,
                                    PortableNativeOps::threads, 
                reinterpret_cast<const float*>(d_input.ptr), reinterpret_cast<float*>(d_col.ptr),
                static_cast<int>(col_sz), H, W, k, str, pad, dil, H_out, W_out);
            if (cudaGetLastError() != cudaSuccess) return false;
            if (!d_w.copyFromHost(wc, w_k * sizeof(float))) return false;

            // SGEMM: output_rm[C_out,HW] = weights_rm[C_out,K] × col_rm[K,HW]
            // cuBLAS (col-major): C[HW,C_out] = B[HW,K] × A[K,C_out]
            // → cublasSgemm(OP_N,OP_N, HW, C_out, K, 1, col, HW, w, K, 0, out, HW)
            const float cv_a = 1.0f, cv_b0 = 0.0f;
            if (cublasSgemm(impl_->handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    HW, C_out, K, &cv_a,
                    reinterpret_cast<const float*>(d_col.ptr), HW,
                    reinterpret_cast<const float*>(d_w.ptr), K,
                    &cv_b0,
                    reinterpret_cast<float*>(d_out.ptr), HW) != CUBLAS_STATUS_SUCCESS) return false;

            // Bias: d_out_cm[HW,C_out] += ones_HW[HW] × bias[C_out]^T  (cublasSger)
            if (layer.use_bias) {
                std::vector<float> ones_hw(static_cast<size_t>(HW), 1.0f);
                Impl::DeviceBuf d_bias, d_ones_hw;
                if (!d_bias.alloc(static_cast<size_t>(C_out) * sizeof(float))) return false;
                if (!d_ones_hw.alloc(static_cast<size_t>(HW) * sizeof(float))) return false;
                if (!d_bias.copyFromHost(wc + w_k, static_cast<size_t>(C_out) * sizeof(float))) return false;
                if (!d_ones_hw.copyFromHost(ones_hw.data(), static_cast<size_t>(HW) * sizeof(float))) return false;
                if (cublasSger(impl_->handle, HW, C_out, &cv_a,
                        reinterpret_cast<const float*>(d_ones_hw.ptr), 1,
                        reinterpret_cast<const float*>(d_bias.ptr), 1,
                        reinterpret_cast<float*>(d_out.ptr), HW) != CUBLAS_STATUS_SUCCESS) return false;
            }

            if (cudaDeviceSynchronize() != cudaSuccess) return false;
            outputs.resize(1);
            outputs[0].resize(static_cast<size_t>(C_out) * HW);
            if (!d_out.copyToHost(outputs[0].data(), static_cast<size_t>(C_out) * HW * sizeof(float))) return false;
            return true;
            } while (false);
            return false;

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
            const CudaBinaryOp op = static_cast<CudaBinaryOp>(op_code);

            outputs.resize(1);
            if (!cuda_run_binary(a, b, outputs[0], op)) return false;
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
            const CudaUnaryOp op = static_cast<CudaUnaryOp>(op_code);

            outputs.resize(1);
            if (!cuda_run_unary(in, outputs[0], op, alpha)) return false;
            return true;
            } while (false);
            break;

        default:
            break;
    }

    // Never hide a device failure behind a CPU result labelled CUDA.
    return false;
}

bool PortableRuntime::forwardLayerWithContext(const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,const Layer& layer,bool training,const RuntimeForwardContext& context){
    DeviceScope device_scope(backend_,config_.device_index);
    if(layer.type_enum==LayerType::Dropout || layer.type_enum==LayerType::Dropout2d || layer.type_enum==LayerType::AlphaDropout){
        if(context.output_mask)context.output_mask->clear();
        if(!isInitialized() || !runtimeCapabilityIsNative(queryForwardOperationCapability(layer,inputs,training)))return false;
        outputs.resize(1);return PortableNativeOps::dropout(*inputs[0],outputs[0],layer,training,context.output_mask);
    }
    return AbstractRuntime::forwardLayerWithContext(inputs,outputs,layer,training,context);
}

bool PortableRuntime::backwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs,
    std::vector<std::vector<float>>& grad_inputs,
    Layer& layer,
    bool training
) {
    if (!referenceInputsFit(layer.getWeightsSize(), inputs) || !referenceInputsFit(0, grad_outputs)) return false;
    DeviceScope device_scope(backend_,config_.device_index);
    (void)training;
    if (!isInitialized()) return false;
    if (config_.disabled) return false;
    if (!supportsBackwardLayerType(layer.type_enum)) return false;
    if (!impl_ || !impl_->handle) return false;
    if (inputs.empty() || !inputs[0] || grad_outputs.empty() || !grad_outputs[0]) return false;

    if(layer.type_enum==LayerType::NMS){
        if(!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
        return PortableNativeOps::nmsBackward(inputs,grad_outputs,grad_inputs,layer);
    }
    if(layer.type_enum==LayerType::Lambda) {
        if(!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
        Layer identity=layer;identity.type_enum=LayerType::Identity;
        CudaOperationGeometry::Movement shape;if(!shape.valid(identity,grad_outputs))return false;
        return PortableNativeOps::movement(grad_outputs,grad_inputs,identity,shape);
    }
    if(CudaOperationGeometry::recurrentType(layer.type_enum)){
        if(!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
        CudaOperationGeometry::Recurrent g;g.valid(layer,inputs[0]->size());std::vector<float> gi,dw;
        if(!PortableNativeOps::recurrent(*inputs[0],gi,layer,g,grad_outputs[0],&dw))return false;
        cuda_accumulate_parameters(layer,dw,{});grad_inputs={std::move(gi)};return true;
    }
    if(CudaOperationGeometry::attentionType(layer.type_enum)){
        if(!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
        CudaOperationGeometry::Attention g;g.valid(layer,inputs,true);std::vector<float> dw;
        if(!PortableNativeOps::attention(inputs,grad_inputs,layer,g,grad_outputs[0],&dw,impl_->handle))return false;
        cuda_accumulate_parameters(layer,dw,{});return true;
    }
    if(CudaOperationGeometry::extendedConvType(layer.type_enum)){
        if(!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
        CudaOperationGeometry::ExtendedConv g;g.valid(layer,inputs[0]->size());std::vector<float> gi,dw;
        if(!PortableNativeOps::extended_conv(*inputs[0],layer,g,gi,grad_outputs[0],&dw))return false;
        std::vector<float> db;if(g.bias){db.assign(dw.begin()+g.parameters,dw.end());dw.resize(g.parameters);}
        cuda_accumulate_parameters(layer,dw,db);grad_inputs={std::move(gi)};return true;
    }
    if(CudaOperationGeometry::batchNormType(layer.type_enum)){
        if(!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
        CudaOperationGeometry::BatchNorm g;g.valid(layer,inputs[0]->size());std::vector<float> gi,dw;
        if(!PortableNativeOps::batch_norm(*inputs[0],gi,layer,g,training,grad_outputs[0],&dw))return false;
        if(g.affine)cuda_accumulate_parameters(layer,dw,{}); // CPU BatchNorm stores beta only in grad_weights.
        grad_inputs={std::move(gi)};return true;
    }
    if(CudaOperationGeometry::stochasticType(layer.type_enum)){
        if(!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
        if(layer.type_enum==LayerType::Reparameterize)return PortableNativeOps::reparameterize(inputs,grad_inputs,training,grad_outputs[0]);
        const auto* saved=inputs.size()>1?inputs[1]:nullptr;std::vector<float> gi;
        if(!PortableNativeOps::dropout(*grad_outputs[0],gi,layer,training,nullptr,saved,true))return false;
        grad_inputs={std::move(gi)};return true;
    }
    if(layer.type_enum==LayerType::GEGLU || layer.type_enum==LayerType::UpsampleBicubic){
        if(!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
        CudaOperationGeometry::Resize g;if(layer.type_enum==LayerType::UpsampleBicubic)g.valid(layer,inputs[0]->size());
        grad_inputs.resize(1);return layer.type_enum==LayerType::GEGLU?PortableNativeOps::geglu(*inputs[0],grad_inputs[0],layer.out_features,grad_outputs[0]):PortableNativeOps::bicubic(*inputs[0],grad_inputs[0],g,grad_outputs[0]);
    }
    if(CudaOperationGeometry::projectionType(layer.type_enum)){
        if(!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
        CudaOperationGeometry::Projection g;g.valid(layer,inputs);std::vector<float> dw;
        if(!PortableNativeOps::projection(inputs,grad_inputs,layer,g,grad_outputs[0],&dw))return false;
        if(g.mode==4 && g.bias){std::vector<float> db(dw.end()-g.dim,dw.end());dw.resize(dw.size()-g.dim);cuda_accumulate_parameters(layer,dw,db);}
        else cuda_accumulate_parameters(layer,dw,{});
        return true;
    }
    if(layer.type_enum==LayerType::LayerNorm || layer.type_enum==LayerType::RMSNorm){
        if(!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
        std::vector<float> gi,dw,db;if(!PortableNativeOps::row_norm(*inputs[0],layer,gi,grad_outputs[0],&dw,&db))return false;
        if(!dw.empty())cuda_accumulate_parameters(layer,dw,db);
        grad_inputs={std::move(gi)};return true;
    }
    if(CudaOperationGeometry::poolType(layer.type_enum) || CudaOperationGeometry::movementType(layer.type_enum)) {
        if(!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training)))return false;
        if(CudaOperationGeometry::poolType(layer.type_enum)){
            CudaOperationGeometry::Pool g;g.valid(layer,inputs[0]->size());grad_inputs.resize(1);return PortableNativeOps::pool(*inputs[0],grad_inputs[0],g,grad_outputs[0]);
        }
        if(layer.type_enum==LayerType::Constant){
            if(layer.trainable_parameter){
                if(layer.grad_weights.size()!=layer.getWeightsSize())layer.grad_weights.assign(layer.getWeightsSize(),0.f);
                for(size_t i=0;i<layer.grad_weights.size();++i)layer.grad_weights[i]+=(*grad_outputs[0])[i];
            }
            grad_inputs.clear();return true;
        }
        CudaOperationGeometry::Movement g;g.valid(layer,inputs);g.sizes.clear();for(auto* x:inputs)g.sizes.push_back(x->size());
        return PortableNativeOps::movement(grad_outputs,grad_inputs,layer,g,true);
    }
    switch (layer.type_enum) {
        case LayerType::Linear: {
            if (!config_.linear_enabled) return false;
            const std::vector<float>& x = *inputs[0];
            const std::vector<float>& go = *grad_outputs[0];

            const int in_f = layer.in_features;
            const int out_f = layer.out_features;
            const int batch = (layer.seq_len > 0) ? layer.seq_len : (in_f > 0 ? static_cast<int>(x.size()) / in_f : 0);
            if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;

            const size_t x_elems = static_cast<size_t>(batch) * static_cast<size_t>(in_f);
            const size_t go_elems = static_cast<size_t>(batch) * static_cast<size_t>(out_f);
            const size_t w_elems = static_cast<size_t>(out_f) * static_cast<size_t>(in_f);
            if (x.size() != x_elems || go.size() != go_elems) return false;

            const float* w = layer.getWeights();
            if (!w || layer.getWeightsSize() < w_elems) return false;

            Impl::DeviceBuf d_x, d_go, d_w, d_gx, d_gw;
            Impl::DeviceBuf d_ones, d_bias;
            if (layer.use_bias && (!d_ones.alloc(batch*sizeof(float)) || !d_bias.alloc(out_f*sizeof(float)))) return false;
            if (!d_x.alloc(x_elems * sizeof(float))) return false;
            if (!d_go.alloc(go_elems * sizeof(float))) return false;
            if (!d_w.alloc(w_elems * sizeof(float))) return false;
            if (!d_gx.alloc(x_elems * sizeof(float))) return false;
            if (!d_gw.alloc(w_elems * sizeof(float))) return false;
            if (!d_x.copyFromHost(x.data(), x_elems * sizeof(float))) return false;
            if (!d_go.copyFromHost(go.data(), go_elems * sizeof(float))) return false;
            if (!d_w.copyFromHost(w, w_elems * sizeof(float))) return false;

            const float alpha = 1.0f;
            const float beta = 0.0f;

            // grad_input = grad_out * W
            if (cublasSgemm(impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    in_f, batch, out_f, &alpha,
                    reinterpret_cast<const float*>(d_w.ptr), in_f,
                    reinterpret_cast<const float*>(d_go.ptr), out_f,
                    &beta,
                    reinterpret_cast<float*>(d_gx.ptr), in_f) != CUBLAS_STATUS_SUCCESS) {
                return false;
            }

            // grad_W = grad_out^T * X
            if (cublasSgemm(impl_->handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    in_f, out_f, batch, &alpha,
                    reinterpret_cast<const float*>(d_x.ptr), in_f,
                    reinterpret_cast<const float*>(d_go.ptr), out_f,
                    &beta,
                    reinterpret_cast<float*>(d_gw.ptr), in_f) != CUBLAS_STATUS_SUCCESS) {
                return false;
            }

            if (cudaDeviceSynchronize() != cudaSuccess) return false;

            grad_inputs.resize(1);
            grad_inputs[0].resize(x_elems);
            std::vector<float> grad_w(w_elems, 0.0f);
            if (!d_gx.copyToHost(grad_inputs[0].data(), x_elems * sizeof(float))) return false;
            if (!d_gw.copyToHost(grad_w.data(), w_elems * sizeof(float))) return false;

            if (layer.grad_weights.size() != layer.getWeightsSize()) {
                layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
            }
            for (size_t i = 0; i < w_elems; ++i) layer.grad_weights[i] += grad_w[i];

            if (layer.use_bias && layer.getWeightsSize() >= w_elems + static_cast<size_t>(out_f)) {
                if (layer.grad_bias.size() != static_cast<size_t>(out_f)) {
                    layer.grad_bias.assign(static_cast<size_t>(out_f), 0.0f);
                }
                std::vector<float> ones(batch,1.f), bias_gradient(out_f);
                if(!d_ones.copyFromHost(ones.data(),batch*sizeof(float)))return false;
                if(cublasSgemm(impl_->handle,CUBLAS_OP_N,CUBLAS_OP_N,out_f,1,batch,&alpha,reinterpret_cast<const float*>(d_go.ptr),out_f,reinterpret_cast<const float*>(d_ones.ptr),batch,&beta,reinterpret_cast<float*>(d_bias.ptr),out_f)!=CUBLAS_STATUS_SUCCESS)return false;
                if(!d_bias.copyToHost(bias_gradient.data(),out_f*sizeof(float)))return false;
                for(int o=0;o<out_f;++o){layer.grad_bias[o]+=bias_gradient[o];layer.grad_weights[w_elems+o]+=bias_gradient[o];}
            }
            return true;
        }

        case LayerType::MatMul:
        case LayerType::BatchMatMul: {
            if (!config_.linear_enabled) return false;
            if (inputs.size() < 2 || !inputs[1]) return false;

            const std::vector<float>& A = *inputs[0];
            const std::vector<float>& B = *inputs[1];
            const std::vector<float>& go = *grad_outputs[0];
            const int M = layer.in_features;
            const int K = layer.out_features;
            const int N = layer.embed_dim;
            const int batches = (layer.type_enum == LayerType::BatchMatMul) ? layer.seq_len : 1;
            if (M <= 0 || K <= 0 || N <= 0 || batches <= 0) return false;

            const size_t a_batch = static_cast<size_t>(M) * static_cast<size_t>(K);
            const size_t b_batch = static_cast<size_t>(K) * static_cast<size_t>(N);
            const size_t c_batch = static_cast<size_t>(M) * static_cast<size_t>(N);
            const size_t a_elems = static_cast<size_t>(batches) * a_batch;
            const size_t b_elems = static_cast<size_t>(batches) * b_batch;
            const size_t c_elems = static_cast<size_t>(batches) * c_batch;
            if (A.size() != a_elems || B.size() != b_elems || go.size() != c_elems) return false;

            Impl::DeviceBuf d_a, d_b, d_go, d_ga, d_gb;
            if (!d_a.alloc(a_elems * sizeof(float))) return false;
            if (!d_b.alloc(b_elems * sizeof(float))) return false;
            if (!d_go.alloc(c_elems * sizeof(float))) return false;
            if (!d_ga.alloc(a_elems * sizeof(float))) return false;
            if (!d_gb.alloc(b_elems * sizeof(float))) return false;
            if (!d_a.copyFromHost(A.data(), a_elems * sizeof(float))) return false;
            if (!d_b.copyFromHost(B.data(), b_elems * sizeof(float))) return false;
            if (!d_go.copyFromHost(go.data(), c_elems * sizeof(float))) return false;

            const float alpha = 1.0f;
            const float beta = 0.0f;
            for (int bi = 0; bi < batches; ++bi) {
                const float* a_ptr = reinterpret_cast<const float*>(d_a.ptr) + static_cast<size_t>(bi) * a_batch;
                const float* b_ptr = reinterpret_cast<const float*>(d_b.ptr) + static_cast<size_t>(bi) * b_batch;
                const float* go_ptr = reinterpret_cast<const float*>(d_go.ptr) + static_cast<size_t>(bi) * c_batch;
                float* ga_ptr = reinterpret_cast<float*>(d_ga.ptr) + static_cast<size_t>(bi) * a_batch;
                float* gb_ptr = reinterpret_cast<float*>(d_gb.ptr) + static_cast<size_t>(bi) * b_batch;

                // dA = dC * B^T
                if (cublasSgemm(impl_->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        K, M, N, &alpha,
                        b_ptr, N,
                        go_ptr, N,
                        &beta,
                        ga_ptr, K) != CUBLAS_STATUS_SUCCESS) {
                    return false;
                }

                // dB = A^T * dC
                if (cublasSgemm(impl_->handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        N, K, M, &alpha,
                        go_ptr, N,
                        a_ptr, K,
                        &beta,
                        gb_ptr, N) != CUBLAS_STATUS_SUCCESS) {
                    return false;
                }
            }

            if (cudaDeviceSynchronize() != cudaSuccess) return false;

            grad_inputs.resize(2);
            grad_inputs[0].resize(a_elems);
            grad_inputs[1].resize(b_elems);
            if (!d_ga.copyToHost(grad_inputs[0].data(), a_elems * sizeof(float))) return false;
            if (!d_gb.copyToHost(grad_inputs[1].data(), b_elems * sizeof(float))) return false;
            return true;
        }

        case LayerType::Add:
        case LayerType::Subtract:
        case LayerType::Multiply:
        case LayerType::Divide: {
            if (!config_.linear_enabled) return false;
            if (inputs.size() < 2 || !inputs[1]) return false;
            const std::vector<float>& a = *inputs[0];
            const std::vector<float>& b = *inputs[1];
            const std::vector<float>& go = *grad_outputs[0];
            if (a.empty() || a.size() != b.size() || go.size() != a.size()) return false;

            int op_code = 0;
            if (!RuntimeLayerOps::resolveBinaryOp(layer.type_enum, op_code)) return false;
            const CudaBinaryOp op = static_cast<CudaBinaryOp>(op_code);

            grad_inputs.resize(2);
            return cuda_run_binary_backward(a, b, go, grad_inputs[0], grad_inputs[1], op);
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
            const std::vector<float>& in = *inputs[0];
            const std::vector<float>& go = *grad_outputs[0];
            if (in.empty() || go.size() != in.size()) return false;

            int op_code = 0;
            float alpha = 0.01f;
            if (!RuntimeLayerOps::resolveUnaryOp(layer.type_enum, layer, op_code, alpha)) return false;
            const CudaUnaryOp op = static_cast<CudaUnaryOp>(op_code);

            grad_inputs.resize(1);
            return cuda_run_unary_backward(in, go, grad_inputs[0], op, alpha);
        }

        case LayerType::Conv2d: {
            if (!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training))) return false;
            const auto& x=*inputs[0]; const auto& go=*grad_outputs[0]; CudaOperationGeometry::Conv shape;
            if (!shape.valid(layer,x.size())) return false;
            std::vector<float> gi,gw,gb;
            if (!PortableNativeOps::conv_backward(impl_->handle,x,go,layer.getWeights(),shape.ic,shape.oc,shape.h,shape.w,
                shape.k,shape.stride,shape.pad,shape.dilation,shape.oh,shape.ow,layer.use_bias,gi,gw,gb)) return false;
            cuda_accumulate_parameters(layer,gw,gb); grad_inputs={std::move(gi)}; return true;
        }
        case LayerType::InstanceNorm2d:
        case LayerType::GroupNorm: {
            if (!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training))) return false;
            const auto& x=*inputs[0]; CudaOperationGeometry::Norm shape; if (!shape.valid(layer,x.size())) return false;
            std::vector<float> gi,gw,gb;
            if (!PortableNativeOps::groupnorm(x,layer.affine?layer.getWeights():nullptr,layer.getWeightsSize(),shape.channels,
                shape.spatial,shape.groups,layer.eps,layer.affine && layer.use_bias,gi,grad_outputs[0],&gw,&gb)) return false;
            if (layer.affine) cuda_accumulate_parameters(layer,gw,gb);
            grad_inputs={std::move(gi)}; return true;
        }
        case LayerType::UpsampleNearest:
        case LayerType::UpsampleBicubic:
        case LayerType::UpsampleBilinear: {
            if (!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training))) return false;
            CudaOperationGeometry::Resize shape; if (!shape.valid(layer,inputs[0]->size())) return false;
            std::vector<float> gi;
            if (!PortableNativeOps::resize(*grad_outputs[0],gi,shape.c,shape.ih,shape.iw,shape.oh,shape.ow,shape.sh,shape.sw,
                                      layer.type_enum==LayerType::UpsampleBilinear,true)) return false;
            grad_inputs={std::move(gi)}; return true;
        }
        case LayerType::Softmax:
        case LayerType::LogSoftmax: {
            if (!runtimeCapabilityIsNative(queryBackwardOperationCapability(layer,inputs,grad_outputs,training))) return false;
            int size=0,inner=0; if (!cuda_axis_geometry(*inputs[0],layer,size,inner)) return false;
            std::vector<float> gi;
            if (!PortableNativeOps::softmax(*inputs[0],gi,size,inner,layer.type_enum==LayerType::LogSoftmax,grad_outputs[0])) return false;
            grad_inputs={std::move(gi)}; return true;
        }
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


    return false;
}
