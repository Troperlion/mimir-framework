#include "training/SpatialRelation.hpp"
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <filesystem>
#include <optional>
#include <cstdint>
#include "include/json.hpp"
#include "Helpers.hpp"    // contient MagicToken, DatasetItem, loadDataset, imageToEmbedding, write_u32_le
#include "tensors.hpp"
#include "Tokenizer.hpp"
#include "Encoder.hpp"
#include "Autograd.hpp"   // Pour la structure Gradients
#include "HardwareOpt.hpp" // Optimisations hardware avancées
#include "SkipConnectionControl.hpp"
#include "LiveModelConfig.hpp"
#include "Layers.hpp"      // Pour la structure Layer
#include "MemoryGuard.hpp" // Pour le strict mode
#include "DType.hpp"
#include "Planning/Planner.hpp"
#include <chrono>

using json = nlohmann::json;
namespace fs = std::filesystem;

// LR Decay strategies
enum class LRDecayStrategy {
    NONE,           // Pas de decay
    COSINE,         // Cosine annealing
    STEP,           // Step decay (réduction par paliers)
    EXPONENTIAL,    // Exponential decay
    LINEAR          // Linear decay
};

// Optimizer types
enum class OptimizerType {
    SGD = 0,      // Stochastic Gradient Descent
    ADAM = 1,     // Adam optimizer
    ADAMW = 2,    // Adam with decoupled weight decay
    LION = 3,
    ADAFACTOR = 4,
    RADAM = 5,
    NADAM = 6,
    RMSPROP = 7,
    LAMB = 8
};

// Optimizer (Adam-like) state with LR decay
struct Optimizer {

    OptimizerType type = OptimizerType::ADAM;

    // Adam moments are stored per parameter block (stable within a run) using the
    // base pointer of the parameter buffer as key.
    struct MomentBlock {
        std::vector<float> m;
        std::vector<float> v;
    };
    std::unordered_map<std::uintptr_t, MomentBlock> mv_by_param_ptr;

    // Flat legacy state used for checkpoint save/load.
    // The runtime optimizerStep uses mv_by_param_ptr.
    std::vector<float> m;
    std::vector<float> v;
    // Stable checkpoint mapping; runtime pointers are never persisted.
    struct StateBlock {
        std::string name;
        size_t offset = 0;
        size_t size = 0;
    };
    std::vector<StateBlock> parameter_layout;
    bool usesFirstMoment() const {
        return type != OptimizerType::SGD && type != OptimizerType::RMSPROP &&
            (type != OptimizerType::ADAFACTOR || adafactor_beta1 > 0.0f);
    }
    bool usesSecondMoment() const {
        return type != OptimizerType::SGD && type != OptimizerType::LION;
    }
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;  // Pour AdamW
    float rmsprop_alpha = 0.99f;
    float adafactor_clip_threshold = 1.0f;
    float adafactor_decay_rate = -0.8f;
    float adafactor_eps2 = 1e-3f;
    float adafactor_beta1 = 0.0f;
    bool adafactor_scale_parameter = true;
    bool adafactor_relative_step = false;
    size_t step = 0;
    
    // LR Decay parameters
    LRDecayStrategy decay_strategy = LRDecayStrategy::COSINE;
    float initial_lr = 5e-5f;
    float min_lr = 1e-6f;          // Learning rate minimum
    float decay_rate = 0.95f;       // Pour exponential/step decay
    int decay_steps = 100;          // Nombre de steps entre chaque decay (step decay)
    int total_steps = 1000;         // Total steps pour cosine/linear
    int warmup_steps = 0;           // Warmup optionnel
    
    void ensure(size_t n) {
        if (m.size() < n) m.resize(n, 0.0f);
        if (v.size() < n) v.resize(n, 0.0f);
    }

    MomentBlock& ensureMomentsFor(const float* param_ptr, size_t n) {
        const std::uintptr_t key = reinterpret_cast<std::uintptr_t>(param_ptr);
        MomentBlock& blk = mv_by_param_ptr[key];
        if (blk.m.size() < n) blk.m.resize(n, 0.0f);
        if (blk.v.size() < n) blk.v.resize(n, 0.0f);
        return blk;
    }
    
    // Calcule le learning rate actuel avec decay
    float getCurrentLR() const {
        const size_t wu = warmup_steps > 0 ? static_cast<size_t>(warmup_steps) : 0ULL;
        if (wu > 0 && step < wu) {
            // Warmup linéaire (évite LR=0 au tout premier step)
            return initial_lr * (static_cast<float>(step + 1) / static_cast<float>(wu));
        }
        
        const int safe_warmup = std::max(0, warmup_steps);
        const int effective_step = std::max(0, static_cast<int>(step) - safe_warmup);
        int effective_total = total_steps - safe_warmup;
        if (effective_total <= 0) effective_total = 1;
        const int safe_decay_steps = std::max(1, decay_steps);
        
        switch (decay_strategy) {
            case LRDecayStrategy::NONE:
                return initial_lr;
                
            case LRDecayStrategy::COSINE: {
                // Cosine annealing: lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * t / T))
                float progress = std::min(1.0f, static_cast<float>(effective_step) / effective_total);
                float cosine = 0.5f * (1.0f + std::cos(3.14159265359f * progress));
                return min_lr + (initial_lr - min_lr) * cosine;
            }
            
            case LRDecayStrategy::STEP: {
                // Step decay: lr *= decay_rate chaque decay_steps
                int num_decays = effective_step / safe_decay_steps;
                return std::max(min_lr, initial_lr * std::pow(decay_rate, static_cast<float>(num_decays)));
            }
            
            case LRDecayStrategy::EXPONENTIAL: {
                // Exponential decay: lr = initial_lr * decay_rate^(step / decay_steps)
                float exponent = static_cast<float>(effective_step) / static_cast<float>(safe_decay_steps);
                return std::max(min_lr, initial_lr * std::pow(decay_rate, exponent));
            }
            
            case LRDecayStrategy::LINEAR: {
                // Linear decay: lr décroit linéairement de initial_lr à min_lr
                float progress = std::min(1.0f, static_cast<float>(effective_step) / effective_total);
                return initial_lr - (initial_lr - min_lr) * progress;
            }
            
            default:
                return initial_lr;
        }
    }
};

bool optimizerTypeFromString(const std::string& name, OptimizerType& type);
const char* optimizerTypeName(OptimizerType type);
void configureOptimizerFromJson(Optimizer& optimizer, const json& config);

// -------------------- Model class --------------------
class Model {
public:
    virtual std::shared_ptr<SkipConnectionControl> skipConnectionControl();
    std::shared_ptr<LiveModelConfig> runtimeConfiguration() { return runtime_config_; }
    void publishRuntimeConfiguration(Optimizer* optimizer = nullptr);
    void applyRuntimeConfiguration();
    void applyRuntimeOptimizerConfiguration(Optimizer& optimizer);

    Model();
    virtual ~Model();

    // Freeze/unfreeze parameters: when frozen, any training operation that would
    // mutate params/gradients (backward/optimizer/weight init) is blocked.
    void freezeParameters(bool freeze = true) { params_frozen_ = freeze; }
    bool parametersFrozen() const { return params_frozen_; }

    void setDensity(double d);
    double getDensity() const;

    // Default storage dtype propagated to every layer. FP16/BF16 accumulate
    // in FP32; FP64 accumulates in FP64. Integer/bool models are forward-only.
    void setDefaultDType(const std::string& dtype);
    const std::string& getDefaultDType() const { return default_dtype_; }

    void build();
    void autoBuildFromDataset(const std::string &dataset_dir);

    // topology hooks (to be overridden)
    virtual void buildBackboneUNet(int stages, int blocks_per_stage, int bottleneck_depth);
    virtual void injectMagicToken(const MagicToken &tok);
    virtual void buildTextBranch(const MagicToken &tok);
    virtual void buildAudioBranch(const MagicToken &tok);
    virtual void buildImageBranch(const MagicToken &tok);
    virtual void buildVideoBranch(const MagicToken &tok);

    std::vector<uint16_t> getWeights() const;
    void setTokenizer(const Tokenizer &t);
    void setEncoder(const ConditioningEncoder &e);

    size_t totalParamCount() const;
    void allocateParams();
    void initializeWeights(const std::string &method = "xavier", unsigned int seed = 0);
    
    // Nouveau forward/backward pass complet
    std::vector<float> forwardPass(const std::vector<float> &input, bool training = true);

    // Variante "view": évite la copie/allocation du std::vector de sortie.
    // Le buffer retourné appartient au TensorStore interne (valide jusqu'au prochain forward).
    const std::vector<float>& forwardPassView(const std::vector<float> &input, bool training = true);

    // Convenience: forward with explicit (encoded prompt, image) and a seed.
    // The seed is used to make stochastic ops (e.g. Dropout in training) deterministic.
    std::vector<float> forwardPromptImageSeed(const std::vector<float>& prompt_vec,
                                              const std::vector<float>& image_vec,
                                              uint32_t seed,
                                              bool training = false);
    // Forward en entrée tokens int: délègue au chemin float après conversion des ids.
    std::vector<float> forwardPass(const std::vector<int> &input_ids, bool training = true);

    // Variante "view": conserve la même délégation au chemin float.
    const std::vector<float>& forwardPassView(const std::vector<int> &input_ids, bool training = true);

    // Nouveau: forward multi-entrées (floats + ids) via TensorStore.
    // Utile pour des archis qui combinent des tenseurs float (ex: latent) et des ids int (ex: texte).
    std::vector<float> forwardPassNamed(const std::unordered_map<std::string, std::vector<float>>& float_inputs,
                                        const std::unordered_map<std::string, std::vector<int>>& int_inputs,
                                        bool training = true);

    // Variante "view": évite la copie/allocation du std::vector de sortie.
    virtual const std::vector<float>& forwardPassNamedView(const std::unordered_map<std::string, std::vector<float>>& float_inputs,
                                                   const std::unordered_map<std::string, std::vector<int>>& int_inputs,
                                                   bool training = true);

    // KV cache pour décodage auto-régressif (SelfAttention en inférence).
    void setKVCacheEnabled(bool enabled);
    bool isKVCacheEnabled() const { return kv_cache_enabled_; }
    void clearKVCache();
    size_t getKVCacheTokenCount() const;

    // ========================================================================
    // Viz taps: capture d'images intermédiaires par bloc/layer (best-effort)
    // ========================================================================
    struct VizFrame {
        std::vector<uint8_t> pixels;      // heatmap (ou naturel pour image_like)
        std::vector<uint8_t> pixels_real; // RGB des canaux, vide si image_like
        int w = 0;
        int h = 0;
        int channels = 1;
        std::string label;
        int heatmap_kind = 0; // 0: image/scalar, 1: signed mean + energy
        std::string tensor_info;
        std::vector<float> tensor_values; // interleaved raw samples, before colour mapping
        std::vector<size_t> tensor_indices;
        int tensor_sample_channels = 0;
    };

    void setVizTapsEnabled(bool enabled) {
        if (viz_taps_enabled_ == enabled) return;
        viz_taps_enabled_ = enabled;
        viz_tips_init_done_ = false;
        viz_tips_custom_enabled_ = false;
        clearVizTipsRegistry();
    }
    bool isVizTapsEnabled() const { return viz_taps_enabled_; }
    int getVizTapsMaxSide() const { return viz_taps_max_side_; }
    void setVizTapsLimits(int max_frames, int max_side) {
        viz_taps_max_frames_ = std::max(0, max_frames);
        viz_taps_max_side_ = std::max(1, max_side);
    }
    // Permet à des modèles spécialisés d'ajouter des vignettes (recon/dénoise/etc.).
    // Respecte le mode enabled, la déduplication par label, et évince en fin de liste si plein.
    void addVizTapFrame(VizFrame vf);
    void clearVizTaps() {
        viz_taps_.clear();
        viz_tips_init_done_ = false;
        viz_tips_custom_enabled_ = false;
        clearVizTipsRegistry();
    }
    // Retourne le snapshot complet sans vider le cache. Les frames sont mises à
    // jour en place par label; clearVizTaps() réalise l'effacement explicite.
    std::vector<VizFrame> consumeVizTaps();

    // Hooks de personnalisation des tips Viz (par modèle enfant).
    // Par défaut: désactivés (retournent false / ne modifient rien).
    virtual bool InitVizTips();
    virtual bool UpdateVizTips(const Layer& layer, VizFrame& frame);

    enum class TrainStepMode {
        Optimize,
        Accumulate
    };

    struct TrainStepRequest {
        std::unordered_map<std::string, const std::vector<float>*> float_inputs;
        std::unordered_map<std::string, const std::vector<int>*> int_inputs;
        const std::vector<float>* target = nullptr;
        const std::vector<float>* loss_weights = nullptr;
        const std::vector<SpatialRelation>* relations = nullptr;
        const json* annotation = nullptr; // Optional structured dataset annotation.
        Optimizer* optimizer = nullptr;
        float learning_rate = 0.0f;
        TrainStepMode mode = TrainStepMode::Optimize;
        float grad_scale = 1.0f;
        bool capture_prediction = false;
    };

    struct TrainStepResult {
        float loss = 0.0f;
        float grad_norm = 0.0f;
        float grad_max_abs = 0.0f;
        std::unordered_map<std::string, float> metrics;
        std::vector<float> prediction;
    };

    // Optional high-level training hook. Models without a training contract return nullopt.
    virtual std::optional<TrainStepResult> trainStep(const TrainStepRequest& request);
    Gradients backwardPass(const std::vector<float> &loss_gradient);
    // Gradient d'entrée capturé au dernier backward (si l'architecture route "__input__")
    bool hasLastInputGradient() const { return has_last_input_gradient_; }
    const std::vector<float>& getLastInputGradient() const { return last_input_gradient_; }
    void zeroGradients();  // Réinitialise tous les gradients à zéro
    void releaseTrainingWorkingSet(size_t completed_step);
    Gradients getGradients() const;  // Récupère les gradients actuels
    float computeLoss(const std::vector<float> &prediction, const std::vector<float> &target, const std::string &loss_type = "mse");
    std::vector<float> computeLossGradient(const std::vector<float> &prediction, const std::vector<float> &target, const std::string &loss_type = "mse");

    // Version in-place: réutilise le buffer de sortie (évite les allocations répétées).
    void computeLossGradientInto(const std::vector<float> &prediction,
                                 const std::vector<float> &target,
                                 std::vector<float> &gradient,
                                 const std::string &loss_type = "mse");
    
    void push(const std::string &name, const std::string &type, size_t params_count);
    void optimizerStep(Optimizer &opt, float learning_rate, const Gradients* gradients = nullptr);
    
    // Fonctions obsolètes (conservées pour compatibilité temporaire)
    void updateWeightsWithNoise(float learning_rate, float noise_std = 0.01f);
    void forward(std::vector<uint8_t> &) const;
    void setOutputTarget(const std::vector<uint8_t> &target);
    void applyParamUpdate(float learning_rate);

    struct DecoderOutput {
        std::vector<int> tokens;
        double mse = -1.0;
        std::vector<float> logits;
    };



        // Optional training state (for checkpoint/debug)
        void setSerializedOptimizer(Optimizer opt);
        Optimizer optimizerSnapshot(const Optimizer& opt) const;
        void restoreOptimizerState(Optimizer& opt) const;
        const Optimizer* getSerializedOptimizer() const { return serialized_optimizer_ ? &(*serialized_optimizer_) : nullptr; }
        Optimizer* getMutableSerializedOptimizer() { return serialized_optimizer_ ? &(*serialized_optimizer_) : nullptr; }
        void clearSerializedOptimizer() { serialized_optimizer_.reset(); }
    DecoderOutput eval(const std::vector<uint8_t> &target) const;
    void setLastEncoding(const std::vector<float> &e);


        std::optional<Optimizer> serialized_optimizer_;
    // accessors used elsewhere
    int width() const { return tw; }
    int height() const { return th; }
    const Tokenizer &getTokenizer() const { return tokenizer; }
    Tokenizer &getMutableTokenizer() { return tokenizer; }
    const ConditioningEncoder &getEncoder() const { return encoder; }
    ConditioningEncoder &getMutableEncoder() { return encoder; }
    
    // Serialization-friendly accessors
    const std::vector<Layer>& getLayers() const { return layers; }
    std::vector<Layer>& getMutableLayers() { return layers; }
    bool getHasEncoder() const { return hasEncoder; }
    bool getHasTokenizer() const { return hasTokenizer; }
    void setHasTokenizer(bool val) { hasTokenizer = val; }
    void setHasEncoder(bool val) { hasEncoder = val; }
    const std::string& getModelName() const { return model_name; }
    void setModelName(const std::string& name) { model_name = name; }

    // static helpers for saving/loading
    bool saveCheckpoint(const Tokenizer &tokenizer, const std::vector<MagicToken> &magic_tokens, const fs::path &dir, int epoch);
    bool packToSafetensor(const fs::path &outpath, const std::unordered_map<std::string, std::vector<float>> &tensors) const;
    bool tryLoadExistingModel(const fs::path &ckdir, const fs::path &safep, Tokenizer &outTok, ConditioningEncoder &outEnc, std::vector<MagicToken> &outMagic);
    bool hasOpenCLCompute() const;
    bool initializeOpenCLComputeEngine();
    void shutdownOpenCLComputeEngine();

    bool hasCudaCompute() const;
    bool initializeCudaComputeEngine();
    void shutdownCudaComputeEngine();

    bool hasRocmCompute() const;
    bool initializeRocmComputeEngine();
    void shutdownRocmComputeEngine();

    bool hasCpuCompute() const;
    bool initializeCpuComputeEngine();
    void shutdownCpuComputeEngine();
    //           Helpers
    // =============================

    // fonction utilitaire : convertit Weight (uint16) -> float [-1,1]
    static inline float weightToFloat(uint16_t w)
    {
        return (static_cast<float>(w) / 65535.0f) * 2.0f - 1.0f;
    }

    static inline float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }
    
    // =============================
    // Hardware Acceleration
    // =============================
    
    // Détection des capacités CPU au runtime
    static bool hasAVX2();
    static bool hasFMA();
    static bool hasF16C();
    static bool hasBMI2();
    
    bool hasVulkanCompute() const;
    bool initializeComputeEngine();
    void shutdownComputeEngine();
    
    // =============================
    // Branch Operations (pour résiduals, skip connections, etc.)
    // =============================
    
    // Détection et exécution automatique des branches pendant forward/backward
    void detectAndSetupBranches();
    void executeBranchComputation(int layer_idx, 
                                  std::vector<std::vector<float>>& layer_outputs,
                                  bool training = false);
    
    // Gestion des gradients pour les branches pendant le backward pass
    void backpropThroughBranch(int layer_idx,
                              const std::vector<float>& grad_output,
                              std::vector<std::vector<float>>& layer_gradients);
    
    static void setFrameworkLogsSuppressed(bool enable);
    static bool frameworkLogsSuppressed();

    // Configuration du modèle (pour dimensionnement dynamique des layers)
    json modelConfig;

    // Preferred dtype for future storage/allocations (string for config interop).
    std::string default_dtype_ = "float32";

    // Dernier gradient d'entrée (debug/usage avancé)
    std::vector<float> last_input_gradient_;
    bool has_last_input_gradient_ = false;

    // Chaque layer a son propre bloc de poids (weight_block)
    std::vector<tensor> layer_weight_blocks;  // 1 tensor = tous les poids d'un layer

    void setName(std::string name) {

        model_name = name;
    }
    
    // Activations du forward pass (pour le backward)
    struct ForwardState {
        // Legacy: premier input seulement (conservé pour compatibilité/debug)
        std::vector<std::vector<float>> layer_inputs;

        // Multi-input: liste des inputs (copiés) par layer, dans l'ordre de layer.inputs
        std::vector<std::vector<std::vector<float>>> layer_inputs_multi;

        // Multi-input: tailles des inputs par layer (utile quand on ne snapshot pas les valeurs)
        std::vector<std::vector<size_t>> layer_input_sizes_multi;

        // Multi-input: noms des inputs utilisés au forward (après défaut "x")
        std::vector<std::vector<std::string>> layer_input_names;

        std::vector<std::vector<float>> layer_outputs;

        // Masques optionnels pour le backward (ex: ReLU/Dropout). Un masque non vide signifie "keep/active".
        std::vector<std::vector<uint8_t>> layer_output_masks;

        std::vector<std::vector<float>> activations;
        std::vector<float> final_output;
        bool skip_connections_enabled = true;
        bool is_valid = false;
        
        void clear() {
            layer_inputs.clear();
            layer_inputs_multi.clear();
            layer_input_sizes_multi.clear();
            layer_input_names.clear();
            layer_outputs.clear();
            layer_output_masks.clear();
            activations.clear();
            final_output.clear();
            is_valid = false;
        }
    };
    ForwardState forward_state;
    std::shared_ptr<SkipConnectionControl> skip_control_ = std::make_shared<SkipConnectionControl>();
    bool skip_control_initialized_ = false;
    std::shared_ptr<LiveModelConfig> runtime_config_ = std::make_shared<LiveModelConfig>();
    
    // ========================================================================
    // TENSOR STORE (Multi-input/Branch Support)
    // ========================================================================
    
    // TensorStore : stockage nommé des tensors pour routing
    std::unordered_map<std::string, std::vector<float>> tensor_store;
    // Authoritative typed representation. The float store above is a
    // compatibility compute view for kernels not migrated to TypedTensor yet.
    std::unordered_map<std::string, Mimir::TypedTensor> typed_tensor_store;
    // Nouveau: TensorStore d'IDs (int) pour layers type Embedding
    std::unordered_map<std::string, std::vector<int>> tensor_store_int;

    // Entrées additionnelles (utilisées par forwardPassNamed, consommées par forwardPass(float)).
    std::optional<std::unordered_map<std::string, std::vector<float>>> pending_float_inputs_;
    std::optional<std::unordered_map<std::string, std::vector<int>>> pending_int_inputs_;
    
    // Helper pour récupérer un tensor (avec erreur explicite si manquant)
    const std::vector<float>& getTensor(const std::string& name) const;
    std::vector<float>& getTensorMutable(const std::string& name);
    const Mimir::TypedTensor& getTypedTensor(const std::string& name) const;
    bool hasTypedTensor(const std::string& name) const;

    const std::vector<int>& getTensorInt(const std::string& name) const;
    std::vector<int>& getTensorIntMutable(const std::string& name);

    // Safe existence checks (évite les logs d'erreur dans getTensor*).
    bool hasTensor(const std::string& name) const;
    bool hasTensorInt(const std::string& name) const;
    
    // Helper pour stocker un tensor
    void storeTensor(const std::string& name, const std::vector<float>& data);
    void storeTensor(const std::string& name, std::vector<float>&& data);

    void storeTensorInt(const std::string& name, const std::vector<int>& data);
    void storeTensorInt(const std::string& name, std::vector<int>&& data);
    
    // Debug : liste tous les tensors disponibles
    std::vector<std::string> getAvailableTensors() const;
    std::vector<std::string> getAvailableIntTensors() const;
    
    // Clear tensor store (appelé au début de chaque forward)
    void clearTensorStore();
    void clearTensorStoreInt();
    
    // Helper pour récupérer un layer par nom
    Layer* getLayerByName(const std::string& name);

    // Buffers scratch pour éviter les allocations dans les hot paths (forward/train step)
    std::vector<const std::vector<float>*> scratch_input_ptrs_;
    std::vector<float> scratch_embedding_ids_tmp_;
    std::vector<int> scratch_embedding_ids_fallback_;
    std::vector<float> scratch_loss_grad_;

    // Registre des tips Viz (layer.name -> label tip personnalisé)
    void clearVizTipsRegistry();
    void registerVizTip(const std::string& layer_name, const std::string& tip_label);
    bool applyVizTipByLayerName(const Layer& layer, VizFrame& frame) const;

    // Viz taps state
    bool viz_taps_enabled_ = false;
    int viz_taps_max_frames_ = 12;
    int viz_taps_max_side_ = 64;
    std::vector<VizFrame> viz_taps_;
    bool viz_tips_init_done_ = false;
    bool viz_tips_custom_enabled_ = false;
    std::unordered_map<std::string, std::string> viz_tips_by_layer_name_;

protected:
    std::vector<Layer> layers;
    int tw = 64, th = 64;
    Tokenizer tokenizer;
    ConditioningEncoder encoder;
    bool hasTokenizer = false;
    bool hasEncoder = false;
    std::vector<float> lastEncoding;
    double densityFactor = 1.0;
    std::string model_name;
    
    // =============================
    // STRICT MODE: Memory Management
    // =============================
    // 0 => suivre la limite MemoryGuard déjà configurée (ex: via Lua)
    // IMPORTANT: éviter toute ré-assignation implicite après build/allocation.
    size_t max_ram_mb_ = 0;
    // MemoryGuard est un singleton, on utilise une référence via instance()

    // When true, blocks any op that would mutate weights/grad buffers.
    bool params_frozen_ = false;

    // =============================
    // Planner / Fusion (framework)
    // =============================
    struct StaticPlanCache {
        bool built = false;
        bool built_for_training = false;
        bool built_with_fusion = false;
        bool built_with_buffer_reuse = false;
        Mimir::Planning::PlannerMode mode = Mimir::Planning::PlannerMode::Legacy;
        bool dumped = false;
        bool runtime_scan_dumped = false;
        std::string runtime_scan_signature;
        Mimir::Planning::ExecutionPlan execution;
    };
    StaticPlanCache static_plan_;
    std::chrono::steady_clock::time_point graph_snapshot_time_{};

    // Cache du résultat de graph_uses_mag_mod() — recalculé à chaque push().
    // Évite le scan O(layers*inputs) à chaque forward pass.
    mutable bool uses_mag_mod_cached_ = false;
    mutable bool uses_mag_mod_         = false;

    struct KVCacheEntry {
        int embed_dim = 0;
        int num_heads = 0;
        int head_dim = 0;
        int seq_len = 0;
        std::vector<float> key;
        std::vector<float> value;

        void clear() {
            embed_dim = 0;
            num_heads = 0;
            head_dim = 0;
            seq_len = 0;
            key.clear();
            value.clear();
        }
    };

    bool kv_cache_enabled_ = false;
    std::unordered_map<size_t, KVCacheEntry> kv_cache_by_layer_;
};
