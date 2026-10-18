#include "runtimes/cpu/CpuRuntime.hpp"

#include "SIMD_Ops.hpp"
#include "runtimes/LayerOps.hpp"
#include "runtimes/cpu/RuntimeLayerDispatch.hpp"

#include <cstddef>

bool CpuRuntime::initialize(const RuntimeConfig& cfg) {
    // CPU est toujours disponible; on conserve cfg pour homogénéité.
    config_ = cfg;
    initialized_ = true;
    return true;
}

void CpuRuntime::shutdown() {
    initialized_ = false;
}

bool CpuRuntime::linearForward(
    const float* input,
    const float* weights,
    const float* bias_or_null,
    float* output,
    int batch,
    int in_f,
    int out_f
) {
    if (!initialized_) return false;
    if (!input || !weights || !output) return false;
    if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;

    // output[batch x out_f] = input[batch x in_f] @ weights[out_f x in_f]^T
    SIMD::matmul_transpose_avx2(
        output,
        input,
        weights,
        static_cast<size_t>(batch),
        static_cast<size_t>(out_f),
        static_cast<size_t>(in_f)
    );

    if (bias_or_null) {
        for (int b = 0; b < batch; ++b) {
            float* row = output + static_cast<size_t>(b) * static_cast<size_t>(out_f);
            for (int o = 0; o < out_f; ++o) {
                row[o] += bias_or_null[o];
            }
        }
    }

    return true;
}

bool CpuRuntime::linearForwardTyped(
    const Mimir::TypedTensor& input,
    const Mimir::TypedTensor& weights,
    const Mimir::TypedTensor* bias,
    Mimir::TypedTensor& output
) {
    if (!initialized_ || input.shape().size() != 2 || weights.shape().size() != 2)
        return false;
    const int batch = input.shape()[0];
    const int in_f = input.shape()[1];
    const int out_f = weights.shape()[0];
    if (weights.shape()[1] != in_f || output.shape() != std::vector<int>({batch, out_f}))
        return false;
    if (input.dtype() != weights.dtype() || output.dtype() != input.dtype())
        return false;
    if (bias && (bias->dtype() != input.dtype() ||
                 bias->shape() != std::vector<int>({out_f})))
        return false;

    for (int row = 0; row < batch; ++row) {
        for (int out = 0; out < out_f; ++out) {
            // F64 keeps double accumulation. Lower precision floating and
            // integer kernels accumulate in at least float64/int64 range.
            long double sum = bias ? bias->get(static_cast<size_t>(out)) : 0.0L;
            for (int in = 0; in < in_f; ++in) {
                sum += static_cast<long double>(
                           input.get(static_cast<size_t>(row) * in_f + in)) *
                       static_cast<long double>(
                           weights.get(static_cast<size_t>(out) * in_f + in));
            }
            output.set(static_cast<size_t>(row) * out_f + out,
                       static_cast<double>(sum));
        }
    }
    return true;
}

bool CpuRuntime::forwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs,
    const Layer& layer,
    bool training
) {
    if (!initialized_) return false;
    if (!supportsForwardLayerType(layer.type_enum)) return false;
    return RuntimeLayerDispatch::cpu_forward_layer(inputs, outputs, layer, training);
}

bool CpuRuntime::forwardLayerWithContext(
    const std::vector<const std::vector<float>*>& inputs,
    std::vector<std::vector<float>>& outputs, const Layer& layer, bool training,
    const RuntimeForwardContext& context) {
    if (context.output_mask && (layer.type_enum == LayerType::Dropout ||
        layer.type_enum == LayerType::Dropout2d || layer.type_enum == LayerType::AlphaDropout)) {
        context.output_mask->clear();
        return initialized_ && RuntimeLayerDispatch::cpu_forward_layer(inputs,outputs,layer,training,context.output_mask);
    }
    return AbstractRuntime::forwardLayerWithContext(inputs,outputs,layer,training,context);
}

bool CpuRuntime::supportsForwardLayerType(const LayerType type) const {
    return RuntimeLayerDispatch::cpu_supports_forward_layer_type(type);
}

bool CpuRuntime::supportsBackwardLayerType(const LayerType type) const {
    return RuntimeLayerDispatch::cpu_supports_backward_layer_type(type);
}

RuntimeCapabilityLevel CpuRuntime::queryForwardCapability(const LayerType type) const {
    return supportsForwardLayerType(type) ? RuntimeCapabilityLevel::Native
                                          : RuntimeCapabilityLevel::Unsupported;
}

RuntimeCapabilityLevel CpuRuntime::queryBackwardCapability(const LayerType type) const {
    return supportsBackwardLayerType(type) ? RuntimeCapabilityLevel::Native
                                           : RuntimeCapabilityLevel::Unsupported;
}

bool CpuRuntime::backwardLayer(
    const std::vector<const std::vector<float>*>& inputs,
    const std::vector<const std::vector<float>*>& grad_outputs,
    std::vector<std::vector<float>>& grad_inputs,
    Layer& layer,
    bool training
) {
    if (!initialized_) return false;
    if (!supportsBackwardLayerType(layer.type_enum)) return false;
    return RuntimeLayerDispatch::cpu_backward_layer(inputs, grad_outputs, grad_inputs, layer, training);
}

bool CpuRuntime::mergeBranches(const std::vector<float>& left,
                               const std::vector<float>& right,
                               std::vector<float>& output, const Layer& layer) {
    if (!initialized_) return false;
    RuntimeLayerOps::branchMerge(left, right, output, layer.merge_op, true);
    return true;
}
