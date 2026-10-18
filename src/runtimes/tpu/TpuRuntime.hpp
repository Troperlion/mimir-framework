#pragma once
#include "runtimes/AbstractRuntime.hpp"
#include "include/json.hpp"
#include <mutex>

// Mímir owns layers, parameters, optimizer and checkpoints. The persistent XLA
// worker compiles these layer operations through JAX/PJRT onto a real TPU.
class TpuRuntime final : public AbstractRuntime {
public:
    bool supportsReconstructionLoss() const override { return isInitialized() && !config_.disabled; }
    bool reconstructionLoss(const RuntimeLossGrad::Request&, RuntimeLossGrad::LossWithGrad&) override;
    bool supportsOptimizer() const override { return isInitialized() && !config_.disabled; }
    bool optimizerStep(const RuntimeOptimizerRequest&) override;
    ~TpuRuntime() override;
    const char* name() const override { return "TPU"; }
    bool initialize(const RuntimeConfig&) override;
    void shutdown() override;
    bool isInitialized() const override { return initialized_; }
    bool linearForward(const float*, const float*, const float*, float*, int, int, int) override;
    bool linearForwardTyped(const Mimir::TypedTensor&, const Mimir::TypedTensor&,
                            const Mimir::TypedTensor*, Mimir::TypedTensor&) override;
    bool forwardLayer(const std::vector<const std::vector<float>*>&,
                      std::vector<std::vector<float>>&, const Layer&, bool) override;
    bool forwardLayerWithContext(const std::vector<const std::vector<float>*>&,
                      std::vector<std::vector<float>>&, const Layer&, bool,
                      const RuntimeForwardContext& = {}) override;
    bool backwardLayer(const std::vector<const std::vector<float>*>&,
                       const std::vector<const std::vector<float>*>&,
                       std::vector<std::vector<float>>&, Layer&, bool) override;
    bool backwardLayerWithContext(const std::vector<const std::vector<float>*>&,
                       const std::vector<const std::vector<float>*>&,
                       std::vector<std::vector<float>>&, Layer&, bool,
                       const RuntimeBackwardContext& = {}) override;
    bool supportsForwardLayerType(LayerType) const override;
    bool supportsBackwardLayerType(LayerType) const override;
    bool mergeBranches(const std::vector<float>&, const std::vector<float>&,
                       std::vector<float>&, const Layer&) override;
    const std::string& lastError() const { return last_error_; }
    const std::string& deviceDescription() const { return device_; }
private:
    bool exchange(nlohmann::json, const std::vector<const std::vector<float>*>&,
                  nlohmann::json&, std::vector<std::vector<float>>&);
    bool forward(const std::vector<const std::vector<float>*>&,
                 std::vector<std::vector<float>>&, const Layer&, bool,
                 bool, std::vector<uint8_t>*);
    void closeWorker();
    bool initialized_ = false;
    int socket_ = -1;
    int child_ = -1;
    std::mutex mutex_;
    std::string last_error_, device_;
};
namespace Mimir {
// Lazy, opt-in (MIMIR_TPU=1). Never probes or imports JAX on CPU/GPU runs.
TpuRuntime* tpuSessionRuntime();
nlohmann::json tpuLayerDescriptor(const Layer&);
}
