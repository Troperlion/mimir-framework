#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "LayerTypes.hpp"
#include "TypedTensor.hpp"

struct Layer;
struct RuntimeOptimizerRequest;
namespace RuntimeLossGrad {struct Request;struct LossWithGrad;}

struct RuntimeForwardContext {
    bool skip_connections_enabled = true;
    std::vector<uint8_t>* output_mask = nullptr;
    bool stochastic_latent = true;
};

// State captured by a training forward, independent of backend selection.
struct RuntimeBackwardContext {
    bool skip_connections_enabled = true;
    const std::vector<float>* output = nullptr;
    const std::vector<uint8_t>* output_mask = nullptr;
};

enum class RuntimeKind : uint8_t {
    Unknown = 0,
    CPU,
    Vulkan,
    OpenCL,
    CUDA,
    ROCm,
    FPGA, // Legacy FPGA I/O identity; no longer a layer backend.
    TPU,
};

enum class RuntimeCapabilityLevel : uint8_t {
    Unsupported = 0,
    HostFallback,
    Native,
    NativeOptimized,
};

struct RuntimeCapability {
    RuntimeCapabilityLevel forward = RuntimeCapabilityLevel::Unsupported;
    RuntimeCapabilityLevel backward = RuntimeCapabilityLevel::Unsupported;
};

inline bool runtimeCapabilityIsNative(const RuntimeCapabilityLevel level) {
    return level == RuntimeCapabilityLevel::Native ||
           level == RuntimeCapabilityLevel::NativeOptimized;
}

struct RuntimeConfig {
    // Nom du backend (ex: "CUDA", "ROCM", "VULKAN", "OPENCL").
    std::string backend;

    // Kill-switch runtime (ex: MIMIR_DISABLE_CUDA=1)
    bool disabled = false;

    // Logs best-effort (ex: MIMIR_ACCEL_VERBOSE=1)
    bool verbose = false;

    // Fast-paths (opt-in)
    bool linear_enabled   = false;
    int  linear_min_ops   = 1 << 20;

    bool conv_enabled     = false;
    int  conv_min_ops     = 1 << 18;  // ~256K MACs

    bool norm_enabled         = false;
    int  norm_min_elements    = 1 << 12; // 4096 éléments

    bool attention_enabled  = false;
    int  attention_min_ops  = 1 << 18;  // ~256K MACs

    // Optionnel: sélection de device (ex: MIMIR_CUDA_DEVICE=0)
    int device_index = 0;

    // Construit une config commune depuis les variables d'environnement.
    // Exemples pour backend="CUDA":
    // - MIMIR_DISABLE_CUDA
    // - MIMIR_CUDA_LINEAR
    // - MIMIR_CUDA_LINEAR_MIN_OPS
    // - MIMIR_CUDA_DEVICE
    // - MIMIR_ACCEL_VERBOSE
    static RuntimeConfig fromEnv(const char* backend_upper);
};

class AbstractRuntime {
public:
    virtual ~AbstractRuntime() = default;

    virtual const char* name() const = 0;

    // Initialisation/shutdown du backend. La config est conservée pour pilotage runtime.
    virtual bool initialize(const RuntimeConfig& cfg) = 0;
    virtual void shutdown() = 0;

    virtual bool isInitialized() const = 0;

    const RuntimeConfig& config() const { return config_; }

    // API minimale alignée avec l'usage actuel (fast-path Linear)
    virtual bool linearForward(
        const float* input,
        const float* weights,
        const float* bias_or_null,
        float* output,
        int batch,
        int in_f,
        int out_f
    ) = 0;

    // Typed path. Backends must return false when they do not implement the
    // requested dtype; callers must not silently relabel float32 output.
    virtual bool linearForwardTyped(
        const Mimir::TypedTensor& input,
        const Mimir::TypedTensor& weights,
        const Mimir::TypedTensor* bias_or_null,
        Mimir::TypedTensor& output
    ) {
        (void)input; (void)weights; (void)bias_or_null; (void)output;
        return false;
    }

    // API générique (objectif: couvrir tous les LayerType via un switch). 
    // - `inputs`: tenseurs float (multi-input)
    // - `outputs`: 1 ou plusieurs tenseurs float produits (Split/Chunk peuvent produire N sorties)
    // Retourne false si le runtime ne supporte pas ce layer.
    virtual bool forwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training
    ) = 0;

    // Complete layer semantics, shared by direct and planned dispatch.
    virtual bool forwardLayerWithContext(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs, const Layer& layer, bool training,
        const RuntimeForwardContext& context = {});

    // Legacy branch merges have an explicit runtime entry point as well.
    virtual bool mergeBranches(const std::vector<float>& left,
                               const std::vector<float>& right,
                               std::vector<float>& output, const Layer& layer) {
        (void)left; (void)right; (void)output; (void)layer;
        return false;
    }

    // API backward générique. Retourne false si non supporté par ce runtime.
    // Convention:
    // - grad_outputs[0] = gradient en sortie du layer
    // - grad_inputs reçoit les gradients d'entrée (même ordre que `inputs`)
    // - `layer` est non-const pour permettre l'accumulation grad_weights/grad_bias
    virtual bool backwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs,
        std::vector<std::vector<float>>& grad_inputs,
        Layer& layer,
        bool training
    );

    virtual bool backwardLayerWithContext(
        const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs,
        std::vector<std::vector<float>>& grad_inputs,
        Layer& layer, bool training, const RuntimeBackwardContext& context);

    // Vote de support (sans calcul): indique si ce runtime prend en charge
    // la famille d'ops d'un LayerType donné.
    virtual bool supportsForwardLayerType(LayerType type) const;
    virtual bool supportsBackwardLayerType(LayerType type) const;

    virtual RuntimeCapabilityLevel queryForwardCapability(LayerType type) const;
    virtual RuntimeCapabilityLevel queryBackwardCapability(LayerType type) const;
    virtual RuntimeCapabilityLevel queryForwardOperationCapability(
        const Layer& layer,
        const std::vector<const std::vector<float>*>& inputs,
        bool training) const;
    virtual RuntimeCapabilityLevel queryBackwardOperationCapability(
        const Layer& layer,
        const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs,
        bool training) const;
    RuntimeCapability queryCapability(LayerType type) const {
        return {queryForwardCapability(type), queryBackwardCapability(type)};
    }
    virtual bool supportsKernelFusion(LayerType producer, LayerType consumer) const;

    // Routeur central: interroge les runtimes par ordre de priorité fourni.
    // Sélectionne le premier runtime initialisé qui supporte l'op.
    static bool dispatchForwardLayer(
        const std::vector<AbstractRuntime*>& runtime_priority,
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training,
        AbstractRuntime** selected_runtime = nullptr
    );

    // Routeur central backward: même logique que dispatchForwardLayer.
    static bool dispatchBackwardLayer(
        const std::vector<AbstractRuntime*>& runtime_priority,
        const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs,
        std::vector<std::vector<float>>& grad_inputs,
        Layer& layer,
        bool training,
        AbstractRuntime** selected_runtime = nullptr
    );

    // CPU inherits the host optimizer adapter; accelerator runtimes override it.
    virtual bool supportsOptimizer() const;
    virtual bool optimizerStep(const RuntimeOptimizerRequest&);

    virtual bool supportsReconstructionLoss() const;
    virtual bool reconstructionLoss(const RuntimeLossGrad::Request&, RuntimeLossGrad::LossWithGrad&);

protected:
    RuntimeCapabilityLevel queryConfiguredForwardOperationCapability(
        const Layer& layer,
        const std::vector<const std::vector<float>*>& inputs,
        bool elementwise_requires_linear_flag) const;
    RuntimeConfig config_{};
};
