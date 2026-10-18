#pragma once

#include "runtimes/AbstractRuntime.hpp"

class PortableRuntime final : public AbstractRuntime {
public:
    ~PortableRuntime() override { shutdown(); }
    explicit PortableRuntime(const char* backend) : backend_(backend) {}
    const char* name() const override { return backend_; }

    void configure(const RuntimeConfig& cfg) { config_ = cfg; }
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
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,const Layer& layer,bool training,
        const RuntimeForwardContext& context = {}) override;

    bool supportsForwardLayerType(LayerType type) const override;
    bool supportsBackwardLayerType(LayerType type) const override;
    RuntimeCapabilityLevel queryForwardCapability(LayerType type) const override;
    RuntimeCapabilityLevel queryBackwardCapability(LayerType type) const override;
    RuntimeCapabilityLevel queryForwardOperationCapability(
        const Layer& layer,
        const std::vector<const std::vector<float>*>& inputs,
        bool training) const override;

    // Geometry/config validation for adapters with parallel forward fast paths.
    // This does not authorize execution through the serial reference kernels.
    RuntimeCapabilityLevel queryForwardGeometryCapability(
        const Layer& layer, const std::vector<const std::vector<float>*>& inputs,
        bool training) const;

    RuntimeCapabilityLevel queryBackwardOperationCapability(
        const Layer& layer,
        const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs,
        bool training) const override;

    bool backwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs,
        std::vector<std::vector<float>>& grad_inputs,
        Layer& layer,
        bool training
    ) override;

private:
    const char* backend_;
    struct Impl;
    Impl* impl_ = nullptr;
};
