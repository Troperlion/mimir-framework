#pragma once

#include "runtimes/AbstractRuntime.hpp"

class VulkanRuntime final : public AbstractRuntime {
public:
    bool supportsReconstructionLoss() const override { return isInitialized() && !config_.disabled; }
    bool reconstructionLoss(const RuntimeLossGrad::Request&, RuntimeLossGrad::LossWithGrad&) override;
    bool supportsOptimizer() const override { return isInitialized() && !config_.disabled; }
    bool optimizerStep(const RuntimeOptimizerRequest&) override;
    ~VulkanRuntime() override { shutdown(); }

    const char* name() const override { return "VULKAN"; }

    bool initialize(const RuntimeConfig& cfg) override;
    void shutdown() override;

    bool isInitialized() const override;

    bool linearForward(
        const float* input,
        const float* weights,
        const float* bias_or_null,
        float* output,
        int batch,
        int in_f,
        int out_f
    ) override;

    bool forwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training
    ) override;

    bool forwardLayerWithContext(
        const std::vector<const std::vector<float>*>& inputs,std::vector<std::vector<float>>& outputs,
        const Layer& layer,bool training,const RuntimeForwardContext& context = {}) override;

    RuntimeCapabilityLevel queryBackwardOperationCapability(
        const Layer& layer, const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs, bool training) const override;
    bool supportsBackwardLayerType(LayerType type) const override;
    bool backwardLayer(const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs,
        std::vector<std::vector<float>>& grad_inputs, Layer& layer, bool training) override;

    bool supportsForwardLayerType(LayerType type) const override;
    RuntimeCapabilityLevel queryForwardCapability(LayerType type) const override;
    RuntimeCapabilityLevel queryBackwardCapability(LayerType type) const override;
    RuntimeCapabilityLevel queryForwardOperationCapability(
        const Layer& layer,
        const std::vector<const std::vector<float>*>& inputs,
        bool training) const override;

    // Initial residency slice: one upload, N native Vulkan unary kernels and
    // one final download. No intermediate std::vector is materialized.
    bool unaryChainForwardResident(
        const float* input,
        float* output,
        int elements,
        const std::vector<LayerType>& operations
    );

private:
    struct Impl;
    Impl* impl_ = nullptr;
};
