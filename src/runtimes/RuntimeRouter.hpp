#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

#include "runtimes/AbstractRuntime.hpp"

struct Layer;

class RuntimeRouter {
public:
    static RuntimeRouter& instance();
    const std::unordered_map<std::string,size_t>& optimizerDispatchCounts() const;
    bool dispatchOptimizer(const RuntimeOptimizerRequest&, AbstractRuntime** selected_runtime=nullptr) const;

    bool dispatchReconstructionLoss(const RuntimeLossGrad::Request&, RuntimeLossGrad::LossWithGrad&,
                                    AbstractRuntime** selected_runtime=nullptr) const;
    const std::unordered_map<std::string,size_t>& reconstructionLossDispatchCounts() const;

    // Priority is [ROCM, CUDA, VULKAN, OPENCL, optional TPU, CPU].
    void setRuntimes(
        AbstractRuntime* rocm,
        AbstractRuntime* cuda,
        AbstractRuntime* vulkan,
        AbstractRuntime* opencl,
        AbstractRuntime* cpu
    );
    void setRuntimes(
        AbstractRuntime* rocm,
        AbstractRuntime* cuda,
        AbstractRuntime* vulkan,
        AbstractRuntime* opencl,
        AbstractRuntime* tpu,
        AbstractRuntime* cpu
    );

    AbstractRuntime* selectForwardRuntimeForLayer(
        const Layer& layer, bool allow_host_fallback = false) const;
    AbstractRuntime* selectBackwardRuntimeForLayer(
        const Layer& layer, bool allow_host_fallback = false) const;

    // Activation callbacks are also ordered [ROCM, CUDA, VULKAN, OPENCL, CPU].
    // They may initialize runtimes lazily before dispatch.
    void setActivators(
        std::function<bool()> rocm,
        std::function<bool()> cuda,
        std::function<bool()> vulkan,
        std::function<bool()> opencl,
        std::function<bool()> cpu
    );

    void activateAvailableRuntimes() const;

    // Prépare une route map forward par layer avant exécution.
    // La map est ensuite réutilisée pendant le forward.
    void planForwardLayerRoutes(const std::vector<Layer>& layers) const;

    // Indique si une route runtime exploitable existe pour ce layer.
    bool hasForwardRouteForLayer(const Layer& layer) const;

    // Vote de support agrégé (sans calcul): true si au moins un runtime
    // configuré prend en charge ce LayerType.
    bool voteForwardLayerType(LayerType type) const;
    bool voteBackwardLayerType(LayerType type) const;

    bool dispatchBranchMerge(const std::vector<float>& left,
                             const std::vector<float>& right,
                             std::vector<float>& output, const Layer& layer) const;

    bool dispatchForwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training,
        AbstractRuntime** selected_runtime = nullptr,
        const RuntimeForwardContext& context = {}
    ) const;

    // Execute the planner choice first, then lower native routes without
    // retrying the failed preferred runtime.
    bool dispatchForwardLayerPlanned(
        AbstractRuntime* preferred,
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training,
        AbstractRuntime** selected_runtime = nullptr,
        const RuntimeForwardContext& context = {}
    ) const;

    bool dispatchBackwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs,
        std::vector<std::vector<float>>& grad_inputs,
        Layer& layer,
        bool training,
        AbstractRuntime** selected_runtime = nullptr,
        const RuntimeBackwardContext& context = {}
    ) const;

private:
    RuntimeRouter() = default;

    static size_t layerTypeIndex(LayerType type);
    void composeRoutes() const;
    void ensureActivatedAndComposed() const;
    std::vector<AbstractRuntime*> buildForwardRouteForLayer(const Layer& layer) const;
    std::vector<AbstractRuntime*> buildBackwardRouteForLayer(const Layer& layer) const;

    std::vector<AbstractRuntime*> runtime_priority_;
    mutable std::vector<uint8_t> forward_vote_;
    mutable std::vector<uint8_t> backward_vote_;
    mutable std::vector<std::vector<AbstractRuntime*>> forward_routes_;
    mutable std::vector<std::vector<AbstractRuntime*>> backward_routes_;
    mutable std::unordered_map<const Layer*, std::vector<AbstractRuntime*>> forward_layer_routes_;
    mutable std::unordered_map<const Layer*, std::vector<AbstractRuntime*>> backward_layer_routes_;
    mutable bool runtimes_activated_ = false;
    std::function<bool()> activate_rocm_;
    std::function<bool()> activate_cuda_;
    std::function<bool()> activate_vulkan_;
    std::function<bool()> activate_opencl_;
    std::function<bool()> activate_cpu_;
};
