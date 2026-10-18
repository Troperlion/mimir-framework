#pragma once

#include "runtimes/AbstractRuntime.hpp"

class CpuRuntime final : public AbstractRuntime {
public:
    ~CpuRuntime() override = default;

    const char* name() const override { return "CPU"; }

    bool initialize(const RuntimeConfig& cfg) override;
    void shutdown() override;

    bool isInitialized() const override { return initialized_; }

    bool linearForward(
        const float* input,
        const float* weights,
        const float* bias_or_null,
        float* output,
        int batch,
        int in_f,
        int out_f
    ) override;

    bool linearForwardTyped(
        const Mimir::TypedTensor& input,
        const Mimir::TypedTensor& weights,
        const Mimir::TypedTensor* bias_or_null,
        Mimir::TypedTensor& output
    ) override;

    bool mergeBranches(const std::vector<float>& left,
                       const std::vector<float>& right,
                       std::vector<float>& output, const Layer& layer) override;

    bool forwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training
    ) override;

    bool forwardLayerWithContext(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs, const Layer& layer, bool training,
        const RuntimeForwardContext& context = {}) override;

    bool supportsForwardLayerType(LayerType type) const override;
    bool supportsBackwardLayerType(LayerType type) const override;
    RuntimeCapabilityLevel queryForwardCapability(LayerType type) const override;
    RuntimeCapabilityLevel queryBackwardCapability(LayerType type) const override;

    bool backwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs,
        std::vector<std::vector<float>>& grad_inputs,
        Layer& layer,
        bool training
    ) override;

private:
    bool initialized_ = false;
};
