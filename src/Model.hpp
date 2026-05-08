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
#include "Layers.hpp"      // Pour la structure Layer
#include "MemoryGuard.hpp" // Pour le strict mode

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
    SGD,      // Stochastic Gradient Descent
    ADAM,     // Adam optimizer
    ADAMW     // Adam with weight decay (L2 regularization)
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
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;  // Pour AdamW
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
        
        int effective_step = static_cast<int>(step) - warmup_steps;
        int effective_total = total_steps - warmup_steps;
        if (effective_total <= 0) effective_total = 1;
        
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
                int num_decays = effective_step / decay_steps;
                return std::max(min_lr, initial_lr * std::pow(decay_rate, static_cast<float>(num_decays)));
            }
            
            case LRDecayStrategy::EXPONENTIAL: {
                // Exponential decay: lr = initial_lr * decay_rate^(step / decay_steps)
                float exponent = static_cast<float>(effective_step) / decay_steps;
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

// -------------------- Model class --------------------
class Model {
public:
    Model();
    virtual ~Model();

    // Freeze/unfreeze parameters: when frozen, any training operation that would
    // mutate params/gradients (backward/optimizer/weight init) is blocked.
    void freezeParameters(bool freeze = true) { params_frozen_ = freeze; }
    bool parametersFrozen() const { return params_frozen_; }

    void setDensity(double d);
    double getDensity() const;

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
    void setEncoder(const Encoder &e);

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
    // Nouveau: forward en entrée tokens int (Embedding consomme des ids)
    std::vector<float> forwardPass(const std::vector<int> &input_ids, bool training = true);

    // Variante "view": évite la copie/allocation du std::vector de sortie.
    const std::vector<float>& forwardPassView(const std::vector<int> &input_ids, bool training = true);

    // Nouveau: forward multi-entrées (floats + ids) via TensorStore.
    // Utile pour des archis qui combinent des tenseurs float (ex: latent) et des ids int (ex: texte).
    std::vector<float> forwardPassNamed(const std::unordered_map<std::string, std::vector<float>>& float_inputs,
                                        const std::unordered_map<std::string, std::vector<int>>& int_inputs,
                                        bool training = true);

    // Variante "view": évite la copie/allocation du std::vector de sortie.
    const std::vector<float>& forwardPassNamedView(const std::unordered_map<std::string, std::vector<float>>& float_inputs,
                                                   const std::unordered_map<std::string, std::vector<int>>& int_inputs,
                                                   bool training = true);

    // ========================================================================
    // Viz taps: capture d'images intermédiaires par bloc/layer (best-effort)
    // ========================================================================
    struct VizFrame {
        std::vector<uint8_t> pixels; // généralement grayscale
        int w = 0;
        int h = 0;
        int channels = 1;
        std::string label;
    };

    void setVizTapsEnabled(bool enabled) { viz_taps_enabled_ = enabled; }
    bool isVizTapsEnabled() const { return viz_taps_enabled_; }
    int getVizTapsMaxSide() const { return viz_taps_max_side_; }
    void setVizTapsLimits(int max_frames, int max_side) {
        viz_taps_max_frames_ = std::max(0, max_frames);
        viz_taps_max_side_ = std::max(1, max_side);
    }
    // Permet à des modèles spécialisés d'ajouter des vignettes (recon/dénoise/etc.).
    // Respecte le mode enabled, la déduplication par label, et évince en fin de liste si plein.
    void addVizTapFrame(VizFrame vf);
    void clearVizTaps() { viz_taps_.clear(); }
    std::vector<VizFrame> consumeVizTaps() {
        auto out = std::move(viz_taps_);
        viz_taps_.clear();
        return out;
    }

    // Variante trainStep pour forwardPassNamed.
    // Calcule loss (MSE) puis backward+optimizerStep.
    struct StepStats {
        float loss = 0.0f;
        float grad_norm = 0.0f;
        float grad_max_abs = 0.0f;

        // Métriques de divergence / cohérence (best-effort) basées sur prediction vs target.
        float kl_divergence = 0.0f;
        float wasserstein = 0.0f;
        float entropy_diff = 0.0f;
        float moment_mismatch = 0.0f;
        float spatial_coherence = 0.0f;
        float temporal_consistency = 0.0f;
    };
    StepStats trainStepNamed(const std::unordered_map<std::string, std::vector<float>>& float_inputs,
                             const std::unordered_map<std::string, std::vector<int>>& int_inputs,
                             const std::vector<float>& target,
                             Optimizer& opt,
                             float learning_rate);

    // VAE helper: assumes the model output packs [recon(image_dim), mu(latent_dim), logvar(latent_dim)].
    struct VAEStepStats {
        float loss = 0.0f;
        // Reconstruction metric (historical name kept for compatibility).
        // Depending on modelConfig["recon_loss"], this may be MSE or L1/MAE.
        float mse = 0.0f;
        float kl = 0.0f;
        // Monitoring/marker metrics computed on recon vs target image.
        float wass = 0.0f;
        float temp = 0.0f;
        float align = 0.0f; // loss d'alignement image/texte (si activé)
        float kl_beta_effective = 0.0f;
        int latent_dim = 0;
        float grad_norm = 0.0f;
        float grad_max_abs = 0.0f;
    };
    VAEStepStats trainStepVAE(const std::vector<float>& x, Optimizer& opt, float learning_rate);

    // VAE texte: output packs [recon(image_dim), mu(latent_dim), logvar(latent_dim), img_proj(proj_dim), text_proj(proj_dim)]
    // Loss: recon (MSE/L1 via modelConfig["recon_loss"]) + beta*KL + align_weight*(1-cos(img_proj, text_proj))
    VAEStepStats trainStepVAEText(const std::vector<float>& x,
                                 const std::vector<int>& text_ids,
                                 Optimizer& opt,
                                 float learning_rate);

    // Accumulate gradients for VAE (no zeroGradients, no optimizerStep). Use grad_scale=1/accum_steps.
    VAEStepStats backwardStepVAE(const std::vector<float>& x, Optimizer& opt, float grad_scale = 1.0f);

    // Accumulate gradients for VAE+text (no zeroGradients, no optimizerStep). Use grad_scale=1/accum_steps.
    VAEStepStats backwardStepVAEText(const std::vector<float>& x,
                                     const std::vector<int>& text_ids,
                                     Optimizer& opt,
                                     float grad_scale = 1.0f);
    Gradients backwardPass(const std::vector<float> &loss_gradient);
    // Gradient d'entrée capturé au dernier backward (si l'architecture route "__input__")
    bool hasLastInputGradient() const { return has_last_input_gradient_; }
    const std::vector<float>& getLastInputGradient() const { return last_input_gradient_; }
    void zeroGradients();  // Réinitialise tous les gradients à zéro
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
    const Encoder &getEncoder() const { return encoder; }
    Encoder &getMutableEncoder() { return encoder; }
    
    // Serialization-friendly accessors
    const std::vector<Layer>& getLayers() const { return layers; }
    std::vector<Layer>& getMutableLayers() { return layers; }
    bool getHasEncoder() const { return hasEncoder; }
    // Invariant framework: tous les modèles doivent avoir un encoder.
    // On autorise uniquement l'activation explicite; ignorer les tentatives de désactivation.
    void setHasEncoder(bool val) { if (val) hasEncoder = true; }
    const std::string& getModelName() const { return model_name; }
    void setModelName(const std::string& name) { model_name = name; }

    // static helpers for saving/loading
    bool saveCheckpoint(const Tokenizer &tokenizer, const std::vector<MagicToken> &magic_tokens, const fs::path &dir, int epoch);
    bool packToSafetensor(const fs::path &outpath, const std::unordered_map<std::string, std::vector<float>> &tensors) const;
    bool tryLoadExistingModel(const fs::path &ckdir, const fs::path &safep, Tokenizer &outTok, Encoder &outEnc, std::vector<MagicToken> &outMagic);
                            bool hasOpenCLCompute() const;

                            bool initializeOpenCLComputeEngine();
    // ============================= 
                            void shutdownOpenCLComputeEngine();
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
    // Layer Operations (Hardware/Software Dispatch)
    // =============================
    
    // Structure pour paramètres de layer
    struct LayerParams {
        std::vector<float> weights;
        std::vector<float> bias;
        int in_features = 0;
        int out_features = 0;
        int kernel_size = 3;
        int stride = 1;
        int padding = 0;
        int dilation = 1;
        int groups = 1;
        bool use_hardware = true;  // Dynamic dispatch
    };
    
    // Convolution 2D avec dispatch hardware/software
    static void computeConv2D(const std::vector<float>& input, std::vector<float>& output,
                             const LayerParams& params, int in_h, int in_w, int in_c, int out_c,
                             bool use_hardware = true);
    
    // Linear/Dense avec dispatch
    static void computeLinear(const std::vector<float>& input, std::vector<float>& output,
                             const LayerParams& params, bool use_hardware = true);
    
    // Pooling avec dispatch
    static void computeMaxPool2D(const std::vector<float>& input, std::vector<float>& output,
                                int in_h, int in_w, int channels, int kernel_size, int stride,
                                bool use_hardware = true);
    
    static void computeAvgPool2D(const std::vector<float>& input, std::vector<float>& output,
                                int in_h, int in_w, int channels, int kernel_size, int stride,
                                bool use_hardware = true);
    
    // Activation avec dispatch
    static void computeActivation(std::vector<float>& data, const std::string& activation_type,
                                 float param = 0.0f, bool use_hardware = true);
    
    // Batch Normalization avec dispatch
    static void computeBatchNorm(std::vector<float>& data, const std::vector<float>& gamma,
                                const std::vector<float>& beta, const std::vector<float>& running_mean,
                                const std::vector<float>& running_var, int batch_size, int channels,
                                int spatial_size, float eps = 1e-5f, bool training = false,
                                bool use_hardware = true);
    
    // Layer Normalization avec dispatch
    static void computeLayerNorm(std::vector<float>& data, const std::vector<float>& gamma,
                                const std::vector<float>& beta, int normalized_size,
                                float eps = 1e-5f, bool use_hardware = true);
    
    // Transpose Convolution avec dispatch
    static void computeConvTranspose2D(const std::vector<float>& input, std::vector<float>& output,
                                      const LayerParams& params, int in_h, int in_w, int in_c, int out_c,
                                      bool use_hardware = true);
    
    // Attention mechanism (pour transformers)
    static void computeAttention(const std::vector<float>& query, const std::vector<float>& key,
                                const std::vector<float>& value, std::vector<float>& output,
                                int seq_len, int d_model, int num_heads, bool use_hardware = true);
    
    // =============================
    // Branch Operations (pour résiduals, skip connections, etc.)
    // =============================
    
    // Calcul des opérations de branche avec dispatch automatique
    static void computeBranchMerge(const std::vector<float>& branch1, 
                                   const std::vector<float>& branch2,
                                   std::vector<float>& output,
                                   MergeOperation merge_op,
                                   bool use_hardware = true);
    
    // Split d'un tensor en plusieurs branches
    static void computeBranchSplit(const std::vector<float>& input,
                                   std::vector<std::vector<float>>& outputs,
                                   const std::vector<int>& split_sizes);
    
    // Détection et exécution automatique des branches pendant forward/backward
    void detectAndSetupBranches();
    void executeBranchComputation(int layer_idx, 
                                  std::vector<std::vector<float>>& layer_outputs,
                                  bool training = false);
    
    // Gestion des gradients pour les branches pendant le backward pass
    void backpropThroughBranch(int layer_idx,
                              const std::vector<float>& grad_output,
                              std::vector<std::vector<float>>& layer_gradients);
    
    // Configuration globale hardware
    static inline bool global_use_hardware = true;
    static void setHardwareAcceleration(bool enable) { global_use_hardware = enable; }

    // Configuration du modèle (pour dimensionnement dynamique des layers)
    json modelConfig;

    // Dernier gradient d'entrée (debug/usage avancé)
    std::vector<float> last_input_gradient_;
    bool has_last_input_gradient_ = false;

    // Aux models used by some training steps (perceptual).
    // Stored here to avoid rebuild/alloc each step.
    std::shared_ptr<Model> aux_perceptual_;

    static void conv2d_same(const std::vector<float> &in, std::vector<float> &out, int W, int H, const std::vector<float> &kernel, int ksize);

    static inline void add_inplace(std::vector<float> &a, const std::vector<float> &b)
    {
        const size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i)
            a[i] += b[i];
    }

    static inline void relu_inplace(std::vector<float> &x)
    {
        for (auto &v : x)
            if (v < 0.0f)
                v = 0.0f;
    }

    static inline void leaky_relu_inplace(std::vector<float> &x, float slope = 0.01f)
    {
        for (auto &v : x)
            if (v < 0.0f)
                v *= slope;
    }

    static inline void tanh_inplace(std::vector<float> &x)
    {
        for (auto &v : x)
            v = std::tanh(v);
    }

    static inline void elu_inplace(std::vector<float> &x, float alpha = 1.0f)
    {
        for (auto &v : x)
            v = (v >= 0.0f) ? v : alpha * (std::exp(v) - 1.0f);
    }

    // Sigmoid « image » (centre 128, échelle 32 -> [0..255])
    static inline void sigmoid_image_inplace(std::vector<float> &x)
    {
        for (auto &v : x)
        {
            float z = (v - 128.0f) / 32.0f;
            v = 255.0f * sigmoidf(z);
        }
    }

    // Softmax in-place (stable) sur un petit vecteur
    static inline void softmax_inplace(std::vector<float> &x)
    {
        if (x.empty())
            return;
        float m = *std::max_element(x.begin(), x.end());
        float s = 0.f;
        for (auto &v : x)
        {
            v = std::exp(v - m);
            s += v;
        }
        for (auto &v : x)
            v /= s;
    }

    // BatchNorm « globale » : (x-mean)/std * gamma + beta
    static inline void batchnorm_global_inplace(std::vector<float> &x, float gamma, float beta, float eps = 1e-5f)
    {
        if (x.empty())
            return;
        double mean = 0.0;
        for (float v : x)
            mean += v;
        mean /= double(x.size());
        double var = 0.0;
        for (float v : x)
        {
            double d = v - mean;
            var += d * d;
        }
        var = var / double(x.size());
        float inv_std = 1.0f / std::sqrt(float(var) + eps);
        for (auto &v : x)
            v = gamma * ((v - float(mean)) * inv_std) + beta;
    }

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
    
    // ========================================================================
    // TENSOR STORE (Multi-input/Branch Support)
    // ========================================================================
    
    // TensorStore : stockage nommé des tensors pour routing
    std::unordered_map<std::string, std::vector<float>> tensor_store;
    // Nouveau: TensorStore d'IDs (int) pour layers type Embedding
    std::unordered_map<std::string, std::vector<int>> tensor_store_int;

    // Entrées additionnelles (utilisées par forwardPassNamed, consommées par forwardPass(float)).
    std::optional<std::unordered_map<std::string, std::vector<float>>> pending_float_inputs_;
    std::optional<std::unordered_map<std::string, std::vector<int>>> pending_int_inputs_;
    
    // Helper pour récupérer un tensor (avec erreur explicite si manquant)
    const std::vector<float>& getTensor(const std::string& name) const;
    std::vector<float>& getTensorMutable(const std::string& name);

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
    std::vector<float> scratch_packed_grad_;

    // Viz taps state
    bool viz_taps_enabled_ = false;
    int viz_taps_max_frames_ = 12;
    int viz_taps_max_side_ = 64;
    std::vector<VizFrame> viz_taps_;

protected:
    std::vector<Layer> layers;
    int tw = 64, th = 64;
    Tokenizer tokenizer;
    Encoder encoder;
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
};