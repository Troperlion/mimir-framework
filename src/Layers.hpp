#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <functional>
#include <immintrin.h>  // AVX2
#include "LayerTypes.hpp"  // Enum central des types de layers

// Needed for inline access to weight_block storage
#include "tensors.hpp"
#include "DType.hpp"

// ============================================================================
// Énumérations et structures
// ============================================================================

enum class ActivationType {
    NONE,
    RELU,
    RELU6,
    LEAKY_RELU,
    ELU,
    SELU,
    GELU,
    SWISH,
    MISH,
    TANH,
    SIGMOID,
    SOFTMAX,
    SOFTPLUS,
    SOFTSIGN
};

enum class PoolingType {
    MAX,
    AVERAGE,
    MIN,
    SUM
};

enum class PaddingMode {
    ZEROS,
    REFLECT,
    REPLICATE,
    CIRCULAR
};

enum class NormalizationType {
    BATCH_NORM,
    LAYER_NORM,
    INSTANCE_NORM,
    GROUP_NORM,
    RMS_NORM
};

// Type de branche pour les layers avec bifurcations
enum class BranchType {
    NONE,              // Pas de branche
    RESIDUAL,          // Connexion résiduelle (y = F(x) + x)
    SKIP_CONNECTION,   // Skip connection (concaténation)
    DENSE_CONNECTION,  // DenseNet-style (concat avec tous les précédents)
    ATTENTION_BRANCH,  // Branche d'attention parallèle
    MULTI_SCALE,       // Branches multi-échelles (Inception-style)
    GATE,              // Gating mechanism (LSTM-style)
    SPLIT,             // Division en branches parallèles
    MERGE              // Fusion de plusieurs branches
};

// Opération de fusion pour les branches
enum class MergeOperation {
    ADD,               // Addition élément par élément
    MULTIPLY,          // Multiplication élément par élément
    CONCATENATE,       // Concaténation le long du canal
    MAX,               // Maximum élément par élément
    AVERAGE,           // Moyenne élément par élément
    GATED,             // Fusion avec gating
    ATTENTION_WEIGHTED // Fusion pondérée par attention
};

// ============================================================================
// Layer de base
// ============================================================================

struct Layer {
    std::string name;
    std::string type;  // String type for backward compat
    LayerType type_enum = LayerType::UNKNOWN;  // Enum type (NEW)
    size_t params_count;
    Mimir::DType dtype = Mimir::DType::F32;
    Mimir::DType accumulation_dtype = Mimir::DType::F32;
    
    // NOUVEAU: Pointeur vers le tensor de poids unifié pour ce layer
    tensor* weight_block = nullptr;  // Tous les poids du layer dans un seul tensor
    std::string shared_weights_from; // Optional source layer for tied parameters
    
    // Données des paramètres (weights, bias) - conservé pour compatibilité
    std::vector<float> weights;
    std::vector<float> bias;
    
    // Gradients (pour backprop)
    std::vector<float> grad_weights;
    std::vector<float> grad_bias;

    // Les couches Constant sont fixes par défaut. Une Constant marquée
    // trainable_parameter se comporte comme un tenseur paramètre sans entrée.
    bool trainable_parameter = false;
    
    // État interne (pour BatchNorm, etc.)
    std::vector<float> running_mean;
    std::vector<float> running_var;
    
    // ========================================================================
    // PARAMETRES UNIVERSELS (tous les layers)
    // ========================================================================
    
    // === Dimensions de base ===
    int in_features = 0;          // Linear, Embedding
    int out_features = 0;         // Linear
    int in_channels = 0;          // Conv, Norm
    int out_channels = 0;         // Conv
    
    // === Dimensions spatiales (pour 2D) ===
    int input_height = 0;
    int input_width = 0;
    int output_height = 0;
    int output_width = 0;
    
    // === Convolution / Pooling ===
    int kernel_size = 0;          // Kernel unique (carré)
    int kernel_h = 0;             // Kernel height (rectangulaire)
    int kernel_w = 0;             // Kernel width
    int stride = 1;               // Stride unique
    int stride_h = 1;             // Stride height
    int stride_w = 1;             // Stride width
    int padding = 0;              // Padding unique
    int pad_h = 0;                // Padding height
    int pad_w = 0;                // Padding width
    int dilation = 1;             // Dilation
    int dilation_h = 1;
    int dilation_w = 1;
    int groups = 1;               // Grouped convolution
    
    // === Normalization ===
    float eps = 1e-5f;            // Epsilon pour stabilité numérique
    int num_groups = 1;           // GroupNorm
    float momentum = 0.1f;        // BatchNorm momentum
    bool affine = true;           // Affine transform (gamma/beta)
    bool track_running_stats = true;  // BatchNorm stats tracking
    
    // === Dropout ===
    float dropout_p = 0.5f;       // Dropout probability
    
    // === Embedding ===
    int vocab_size = 0;           // Embedding vocabulary size
    int embed_dim = 0;            // Embedding dimension
    int padding_idx = -1;         // Padding index for embedding
    
    // === Softmax / LogSoftmax ===
    int axis = -1;                // Axis pour softmax (default: dernier)
    bool use_mask = false;        // Causal mask pour attention

    // === Non-Maximum Suppression ===
    // Entrées: boxes [N,4] en xyxy, scores [N], classes [N] optionnelles.
    // Sortie: indices originaux conservés, encodés en float.
    float nms_iou_threshold = 0.5f;
    float nms_score_threshold = 0.0f;
    int nms_max_detections = 0;       // 0 = aucune limite
    bool nms_class_agnostic = false;   // classes ignorées si true
    
    // === Reshape / View ===
    std::vector<int> target_shape;  // Shape cible pour reshape
    std::vector<int> shape;         // Shape actuel (pour permute, etc.)
    std::vector<int> permute_dims;  // Ordre des dimensions pour permute [0,2,1]
    
    // === Concat / Split ===
    // Explicit graph annotation; -1 preserves ordinary merge semantics.
    int skip_input_index = -1;
    int skipInputIndex() const {
        if (type_enum != LayerType::Concat && type_enum != LayerType::Add) return -1;
        if (skip_input_index >= 0 && static_cast<size_t>(skip_input_index) < inputs.size())
            return skip_input_index;
        // Older checkpoints predate the annotation. Recognize only native
        // encoder-decoder skips, never arbitrary concatenations/residuals.
        if (type_enum == LayerType::Concat && inputs.size() == 2 &&
            ((name.rfind("vae_conv/dec/", 0) == 0 && name.size() >= 9 &&
              name.compare(name.size() - 9, 9, "/skip_cat") == 0) ||
             (name.rfind("unet/up", 0) == 0 && name.size() >= 7 &&
              name.compare(name.size() - 7, 7, "/concat") == 0))) return 1;
        return -1;
    }
    int concat_axis = 1;               // Axis de concatenation (default: channels)
    int num_splits = 2;                // Nombre de splits (splits égaux si split_sizes vide)
    std::vector<int> split_sizes;      // Tailles explicites de chaque split (si vide: splits égaux)
    int split_axis = 0;                // Axis de split (default: 0 = batch dimension)
    
    // === Upsample ===
    float scale_h = 2.0f;         // Scale factor height
    float scale_w = 2.0f;         // Scale factor width
    int out_h = 0;                // Output height explicit
    int out_w = 0;                // Output width explicit
    
    // === Attention ===
    int num_heads = 8;            // Multi-head attention
    int num_kv_heads = 0;         // 0 = num_heads; smaller enables GQA
    int head_dim = 64;            // Dimension par head
    int seq_len = 0;              // Sequence length pour attention
    bool causal = false;          // Causal mask
    float rope_theta = 0.0f;      // > 0 applies RoPE to Q and K

    // === Custom: PatchEmbed (projection learnable des patches) ===
    int patch_dim = 0;            // Dimension d'un patch aplati (p*p)
    int num_patches = 0;          // Nombre total de patches
    int seq_text = 0;             // Longueur de la partie texte (sans le time token)
    
    // === Activation ===
    float alpha = 0.01f;          // LeakyReLU alpha
    float negative_slope = 0.01f; // Alias pour alpha
    float leaky_relu_alpha = 0.01f;  // LeakyReLU alpha (explicit)
    
    // === Shape Operations (Strict Mode) ===
    int squeeze_dim = -1;         // Dimension à squeeze (-1 = auto)
    int unsqueeze_dim = -1;       // Dimension où unsqueeze
    int num_chunks = 2;           // Nombre de chunks pour Chunk operation
    int stack_axis = 0;           // Axis pour Stack operation
    
    // === Bias ===
    bool use_bias = true;
    
    // ========================================================================
    // MULTI-INPUT / TENSOR ROUTING (Nouveau système)
    // ========================================================================
    
    std::vector<std::string> inputs;   // Noms des tensors d'entrée (vide = {"x"} par défaut)
    std::string output = "x";          // Nom du tensor de sortie (default "x")
    
    // ========================================================================
    // CONFIGURATION DES BRANCHES
    // ========================================================================
    
    ActivationType activation = ActivationType::NONE;
    float activation_param = 0.0f; // Pour LeakyReLU alpha, ELU alpha, etc.
    
    // Configuration des branches (pour les architectures avec skip connections, etc.)
    BranchType branch_type = BranchType::NONE;
    MergeOperation merge_op = MergeOperation::ADD;
    std::vector<int> branch_sources;  // Indices des layers sources pour les branches
    int branch_target = -1;           // Indice du layer cible pour cette branche
    bool is_branch_point = false;     // Ce layer est un point de bifurcation
    bool is_merge_point = false;      // Ce layer est un point de fusion
    
    Layer() = default;
    
    Layer(const std::string& n, const std::string& t, size_t pc)
        : name(n), type(t), params_count(pc) {
        // Convertir string -> enum automatiquement
        std::string normalized = LayerRegistry::normalize_type(t);
        type_enum = LayerRegistry::string_to_type(normalized);
        type = normalized;  // Stocker le type normalisé
    }
    
    // Accesseur pour récupérer les données du weight_block
    inline float* getWeights() {
        if (weight_block) return weight_block->getData();
        return weights.data();
    }

    inline const float* getWeights() const {
        if (weight_block) return weight_block->getData();
        return weights.data();
    }

    inline size_t getWeightsSize() const {
        // Hot-path: when using unified weight blocks, params_count is the authoritative size.
        // This avoids calling tensor::getSize() in tight loops.
        if (weight_block) return params_count ? params_count : weight_block->getSize();
        return weights.size();
    }
    
    // Helper: obtenir kernel effectif (kernel_size prend priorité)
    int get_kernel_h() const { return kernel_h > 0 ? kernel_h : (kernel_size > 0 ? kernel_size : 3); }
    int get_kernel_w() const { return kernel_w > 0 ? kernel_w : (kernel_size > 0 ? kernel_size : 3); }
    // NOTE: stride_h/stride_w valent 1 par défaut. Si on n'a configuré que `stride`,
    // on doit utiliser `stride` (sinon le stride reste bloqué à 1).
    int get_stride_h() const { return (stride_h != 1) ? stride_h : stride; }
    int get_stride_w() const { return (stride_w != 1) ? stride_w : stride; }
    int get_pad_h() const { return pad_h > 0 ? pad_h : padding; }
    int get_pad_w() const { return pad_w > 0 ? pad_w : padding; }
    
    // Détection automatique du type de branche basé sur le nom du layer
    void detectBranchType() {
        if (name.find("residual") != std::string::npos || 
            name.find("shortcut") != std::string::npos ||
            name.find("skip") != std::string::npos) {
            branch_type = BranchType::RESIDUAL;
            merge_op = MergeOperation::ADD;
        }
        else if (name.find("concat") != std::string::npos) {
            branch_type = BranchType::SKIP_CONNECTION;
            merge_op = MergeOperation::CONCATENATE;
        }
        else if (name.find("dense") != std::string::npos && name.find("connect") != std::string::npos) {
            branch_type = BranchType::DENSE_CONNECTION;
            merge_op = MergeOperation::CONCATENATE;
        }
        else if (name.find("attention") != std::string::npos && name.find("branch") != std::string::npos) {
            branch_type = BranchType::ATTENTION_BRANCH;
            merge_op = MergeOperation::ATTENTION_WEIGHTED;
        }
        else if (name.find("inception") != std::string::npos || name.find("multiscale") != std::string::npos) {
            branch_type = BranchType::MULTI_SCALE;
            merge_op = MergeOperation::CONCATENATE;
        }
        else if (name.find("gate") != std::string::npos) {
            branch_type = BranchType::GATE;
            merge_op = MergeOperation::GATED;
        }
        else if (name.find("split") != std::string::npos) {
            branch_type = BranchType::SPLIT;
            is_branch_point = true;
        }
        else if (name.find("merge") != std::string::npos || name.find("fusion") != std::string::npos) {
            branch_type = BranchType::MERGE;
            is_merge_point = true;
            // L'opération de fusion par défaut peut être précisée
            if (name.find("add") != std::string::npos) {
                merge_op = MergeOperation::ADD;
            } else if (name.find("mul") != std::string::npos) {
                merge_op = MergeOperation::MULTIPLY;
            } else if (name.find("concat") != std::string::npos) {
                merge_op = MergeOperation::CONCATENATE;
            }
        }
    }
    
    // Détermine si ce layer nécessite un calcul de branche
    bool requiresBranchComputation() const {
        return branch_type != BranchType::NONE || is_branch_point || is_merge_point;
    }
};

// ============================================================================
// Fonctions d'activation
// ============================================================================

namespace Activation {

// ReLU: max(0, x)
inline float relu(float x) {
    return std::max(0.0f, x);
}

inline void relu2d(std::vector<float>& data, int width, int height) {
    size_t size = data.size();
    size_t i = 0;
    
#ifdef __AVX2__
    __m256 zero = _mm256_setzero_ps();
    // Traiter 8 floats à la fois avec AVX2
    for (; i + 8 <= size; i += 8) {
        __m256 vals = _mm256_loadu_ps(&data[i]);
        __m256 result = _mm256_max_ps(vals, zero);
        _mm256_storeu_ps(&data[i], result);
    }
#endif
    
    // Remaining elements
    for (; i < size; ++i) {
        data[i] = relu(data[i]);
    }
}

inline void relu3d(std::vector<float>& data, int width, int height, int depth) {
    size_t size = data.size();
    size_t i = 0;
    
#ifdef __AVX2__
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 vals = _mm256_loadu_ps(&data[i]);
        __m256 result = _mm256_max_ps(vals, zero);
        _mm256_storeu_ps(&data[i], result);
    }
#endif
    
    for (; i < size; ++i) {
        data[i] = relu(data[i]);
    }
}

// ReLU6: min(max(0, x), 6)
inline float relu6(float x) {
    return std::min(std::max(0.0f, x), 6.0f);
}

// Leaky ReLU: x if x > 0 else alpha * x
inline float leaky_relu(float x, float alpha = 0.01f) {
    return x > 0.0f ? x : alpha * x;
}

// ELU: x if x > 0 else alpha * (exp(x) - 1)
inline float elu(float x, float alpha = 1.0f) {
    return x > 0.0f ? x : alpha * (std::exp(x) - 1.0f);
}

// SELU: scaled exponential linear unit
inline float selu(float x) {
    constexpr float alpha = 1.6732632423543772848170429916717f;
    constexpr float scale = 1.0507009873554804934193349852946f;
    return scale * (x > 0.0f ? x : alpha * (std::exp(x) - 1.0f));
}

// GELU: Gaussian Error Linear Unit
inline float gelu(float x) {
    constexpr float sqrt_2_pi = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_pi * (x + 0.044715f * x * x * x)));
}

// Swish (SiLU): x * sigmoid(x)
inline float swish(float x) {
    return x / (1.0f + std::exp(-x));
}

// Mish: x * tanh(softplus(x))
inline float mish(float x) {
    return x * std::tanh(std::log(1.0f + std::exp(x)));
}

// Tanh
inline float tanh_act(float x) {
    return std::tanh(x);
}

// Sigmoid: 1 / (1 + exp(-x))
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// Softplus: log(1 + exp(x))
inline float softplus(float x) {
    return std::log(1.0f + std::exp(x));
}

// Softsign: x / (1 + |x|)
inline float softsign(float x) {
    return x / (1.0f + std::abs(x));
}

// Softmax (pour un vecteur)
inline void softmax(std::vector<float>& logits) {
    if (logits.empty()) return;
    size_t size = logits.size();
    
    // Trouver le max pour stabilité numérique avec AVX2
    float max_val = logits[0];
#ifdef __AVX2__
    if (size >= 8) {
        __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        size_t i = 0;
        for (; i + 8 <= size; i += 8) {
            __m256 vals = _mm256_loadu_ps(&logits[i]);
            max_vec = _mm256_max_ps(max_vec, vals);
        }
        // Réduction horizontale
        float temp[8];
        _mm256_storeu_ps(temp, max_vec);
        for (int j = 0; j < 8; ++j) max_val = std::max(max_val, temp[j]);
        for (; i < size; ++i) max_val = std::max(max_val, logits[i]);
    } else
#endif
    {
        max_val = *std::max_element(logits.begin(), logits.end());
    }
    
    // Exp(x - max) avec AVX2
    float sum = 0.0f;
    size_t i = 0;
#ifdef __AVX2__
    __m256 max_vec = _mm256_set1_ps(max_val);
    __m256 sum_vec = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 vals = _mm256_loadu_ps(&logits[i]);
        vals = _mm256_sub_ps(vals, max_vec);
        // Approximation rapide d'exp avec AVX2 (à améliorer si nécessaire)
        // Pour l'instant utilisons le fallback scalaire pour exp
        float temp[8];
        _mm256_storeu_ps(temp, vals);
        for (int j = 0; j < 8; ++j) {
            temp[j] = std::exp(temp[j]);
            sum += temp[j];
        }
        vals = _mm256_loadu_ps(temp);
        _mm256_storeu_ps(&logits[i], vals);
    }
#endif
    for (; i < size; ++i) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }
    
    // Normaliser avec AVX2
    float inv_sum = 1.0f / sum;
    i = 0;
#ifdef __AVX2__
    __m256 inv_vec = _mm256_set1_ps(inv_sum);
    for (; i + 8 <= size; i += 8) {
        __m256 vals = _mm256_loadu_ps(&logits[i]);
        vals = _mm256_mul_ps(vals, inv_vec);
        _mm256_storeu_ps(&logits[i], vals);
    }
#endif
    for (; i < size; ++i) {
        logits[i] *= inv_sum;
    }
}

// Application générique d'activation
inline float apply(float x, ActivationType type, float param = 0.0f) {
    switch (type) {
        case ActivationType::NONE: return x;
        case ActivationType::RELU: return relu(x);
        case ActivationType::RELU6: return relu6(x);
        case ActivationType::LEAKY_RELU: return leaky_relu(x, param);
        case ActivationType::ELU: return elu(x, param);
        case ActivationType::SELU: return selu(x);
        case ActivationType::GELU: return gelu(x);
        case ActivationType::SWISH: return swish(x);
        case ActivationType::MISH: return mish(x);
        case ActivationType::TANH: return tanh_act(x);
        case ActivationType::SIGMOID: return sigmoid(x);
        case ActivationType::SOFTPLUS: return softplus(x);
        case ActivationType::SOFTSIGN: return softsign(x);
        default: return x;
    }
}

inline void apply_inplace(std::vector<float>& data, ActivationType type, float param = 0.0f) {
    if (type == ActivationType::SOFTMAX) {
        softmax(data);
        return;
    }
    
    size_t size = data.size();
    size_t i = 0;
    
#ifdef __AVX2__
    // Optimisations AVX2 pour activations simples
    if (type == ActivationType::RELU) {
        __m256 zero = _mm256_setzero_ps();
        for (; i + 8 <= size; i += 8) {
            __m256 vals = _mm256_loadu_ps(&data[i]);
            vals = _mm256_max_ps(vals, zero);
            _mm256_storeu_ps(&data[i], vals);
        }
    } else if (type == ActivationType::RELU6) {
        __m256 zero = _mm256_setzero_ps();
        __m256 six = _mm256_set1_ps(6.0f);
        for (; i + 8 <= size; i += 8) {
            __m256 vals = _mm256_loadu_ps(&data[i]);
            vals = _mm256_max_ps(vals, zero);
            vals = _mm256_min_ps(vals, six);
            _mm256_storeu_ps(&data[i], vals);
        }
    } else if (type == ActivationType::LEAKY_RELU) {
        __m256 zero = _mm256_setzero_ps();
        __m256 alpha = _mm256_set1_ps(param);
        for (; i + 8 <= size; i += 8) {
            __m256 vals = _mm256_loadu_ps(&data[i]);
            __m256 neg_part = _mm256_mul_ps(vals, alpha);
            __m256 mask = _mm256_cmp_ps(vals, zero, _CMP_GT_OQ);
            vals = _mm256_blendv_ps(neg_part, vals, mask);
            _mm256_storeu_ps(&data[i], vals);
        }
    } else if (type == ActivationType::TANH) {
        // TANH approximation rapide avec AVX2
        __m256 three = _mm256_set1_ps(3.0f);
        __m256 neg_three = _mm256_set1_ps(-3.0f);
        __m256 c1 = _mm256_set1_ps(27.0f);
        __m256 c2 = _mm256_set1_ps(9.0f);
        for (; i + 8 <= size; i += 8) {
            __m256 x = _mm256_loadu_ps(&data[i]);
            // Clipping
            x = _mm256_min_ps(_mm256_max_ps(x, neg_three), three);
            __m256 x2 = _mm256_mul_ps(x, x);
            __m256 num = _mm256_fmadd_ps(x2, x, c1);
            __m256 den = _mm256_fmadd_ps(c2, x2, c1);
            x = _mm256_div_ps(_mm256_mul_ps(x, num), den);
            _mm256_storeu_ps(&data[i], x);
        }
    }
#endif
    
    // Fallback scalaire pour éléments restants ou activations non vectorisées
    for (; i < size; ++i) {
        data[i] = apply(data[i], type, param);
    }
}

} // namespace Activation

// ============================================================================
// Opérations de convolution
// ============================================================================

namespace Conv {

// Convolution 1D
inline void conv1d(const std::vector<float>& input, std::vector<float>& output,
                   const std::vector<float>& kernel, const std::vector<float>& bias,
                   int in_length, int in_channels, int out_channels,
                   int kernel_size, int stride = 1, int padding = 0, int dilation = 1) {
    
    int out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    output.resize(out_length * out_channels, 0.0f);
    
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ol = 0; ol < out_length; ++ol) {
            float sum = 0.0f;
            
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int k = 0; k < kernel_size; ++k) {
                    int il = ol * stride - padding + k * dilation;
                    
                    if (il >= 0 && il < in_length) {
                        int in_idx = ic * in_length + il;
                        int kernel_idx = ((oc * in_channels + ic) * kernel_size) + k;
                        sum += input[in_idx] * kernel[kernel_idx];
                    }
                }
            }
            
            if (!bias.empty()) {
                sum += bias[oc];
            }
            
            output[oc * out_length + ol] = sum;
        }
    }
}

// Convolution 2D - Vraie implémentation optimisée
inline void conv2d(const std::vector<float>& input, std::vector<float>& output,
                   const std::vector<float>& kernel, const std::vector<float>& bias,
                   int in_height, int in_width, int in_channels, int out_channels,
                   int kernel_size, int stride = 1, int padding = 0, int dilation = 1) {
    
    // Calcul des dimensions de sortie
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    output.resize(out_height * out_width * out_channels, 0.0f);
    
    #ifdef __AVX2__
    // Version SIMD optimisée
        #pragma omp parallel for collapse(2) if(static_cast<long long>(out_channels) * out_height * out_width * in_channels * kernel_size * kernel_size > 262144) schedule(static)
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                __m256 acc = _mm256_setzero_ps();
                float scalar_tail = 0.0f;
                
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        int ih = oh * stride - padding + kh * dilation;
                        if (ih < 0 || ih >= in_height) continue;
                        
                        int kw = 0;
                        // Boucle vectorisée par 8
                        for (; kw + 7 < kernel_size; kw += 8) {
                            int iw_base = ow * stride - padding + kw * dilation;
                            
                            // Charger 8 valeurs d'entrée
                            float in_vals[8];
                            for (int i = 0; i < 8; ++i) {
                                int iw = iw_base + i * dilation;
                                in_vals[i] = (iw >= 0 && iw < in_width) ? 
                                    input[(ic * in_height + ih) * in_width + iw] : 0.0f;
                            }
                            __m256 in_vec = _mm256_loadu_ps(in_vals);
                            
                            // Charger 8 poids du kernel
                            int kernel_base = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                            __m256 k_vec = _mm256_loadu_ps(&kernel[kernel_base]);
                            
                            // FMA: acc += in_vec * k_vec
                            acc = _mm256_fmadd_ps(in_vec, k_vec, acc);
                        }
                        
                        // Reste (non vectorisé)
                        for (; kw < kernel_size; ++kw) {
                            int iw = ow * stride - padding + kw * dilation;
                            if (iw >= 0 && iw < in_width) {
                                int in_idx = (ic * in_height + ih) * in_width + iw;
                                int kernel_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                float prod = input[in_idx] * kernel[kernel_idx];
                                scalar_tail += prod;
                            }
                        }
                    }
                }
                
                // Réduction horizontale de l'accumulateur
                __m128 sum_high = _mm256_extractf128_ps(acc, 1);
                __m128 sum_low = _mm256_castps256_ps128(acc);
                __m128 sum = _mm_add_ps(sum_low, sum_high);
                __m128 shuf = _mm_movehdup_ps(sum);
                __m128 sums = _mm_add_ps(sum, shuf);
                shuf = _mm_movehl_ps(shuf, sums);
                sums = _mm_add_ss(sums, shuf);
                float result = _mm_cvtss_f32(sums) + scalar_tail;
                
                // Ajout du bias
                if (!bias.empty()) {
                    result += bias[oc];
                }
                
                output[(oc * out_height + oh) * out_width + ow] = result;
            }
        }
    }
    #else
    // Version CPU standard optimisée
        #pragma omp parallel for collapse(2) if(static_cast<long long>(out_channels) * out_height * out_width * in_channels * kernel_size * kernel_size > 262144) schedule(static)
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                float sum = 0.0f;
                
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        int ih = oh * stride - padding + kh * dilation;
                        if (ih < 0 || ih >= in_height) continue;
                        
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int iw = ow * stride - padding + kw * dilation;
                            
                            if (iw >= 0 && iw < in_width) {
                                int in_idx = (ic * in_height + ih) * in_width + iw;
                                int kernel_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                sum += input[in_idx] * kernel[kernel_idx];
                            }
                        }
                    }
                }
                
                if (!bias.empty()) {
                    sum += bias[oc];
                }
                
                output[(oc * out_height + oh) * out_width + ow] = sum;
            }
        }
    }
    #endif
}

// Convolution 3D
inline void conv3d(const std::vector<float>& input, std::vector<float>& output,
                   const std::vector<float>& kernel, const std::vector<float>& bias,
                   int in_depth, int in_height, int in_width, int in_channels, int out_channels,
                   int kernel_size, int stride = 1, int padding = 0, int dilation = 1) {
    
    int out_depth = (in_depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    output.resize(out_depth * out_height * out_width * out_channels, 0.0f);
    
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int od = 0; od < out_depth; ++od) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0f;
                    
                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kd = 0; kd < kernel_size; ++kd) {
                            for (int kh = 0; kh < kernel_size; ++kh) {
                                for (int kw = 0; kw < kernel_size; ++kw) {
                                    int id = od * stride - padding + kd * dilation;
                                    int ih = oh * stride - padding + kh * dilation;
                                    int iw = ow * stride - padding + kw * dilation;
                                    
                                    if (id >= 0 && id < in_depth && 
                                        ih >= 0 && ih < in_height && 
                                        iw >= 0 && iw < in_width) {
                                        
                                        int in_idx = ((ic * in_depth + id) * in_height + ih) * in_width + iw;
                                        int kernel_idx = (((oc * in_channels + ic) * kernel_size + kd) * kernel_size + kh) * kernel_size + kw;
                                        sum += input[in_idx] * kernel[kernel_idx];
                                    }
                                }
                            }
                        }
                    }
                    
                    if (!bias.empty()) {
                        sum += bias[oc];
                    }
                    
                    output[((oc * out_depth + od) * out_height + oh) * out_width + ow] = sum;
                }
            }
        }
    }
}

// Transposed Convolution 2D (Deconvolution)
inline void conv_transpose2d(const std::vector<float>& input, std::vector<float>& output,
                             const std::vector<float>& kernel, const std::vector<float>& bias,
                             int in_height, int in_width, int in_channels, int out_channels,
                             int kernel_size, int stride = 1, int padding = 0) {
    
    int out_height = (in_height - 1) * stride - 2 * padding + kernel_size;
    int out_width = (in_width - 1) * stride - 2 * padding + kernel_size;
    
    output.assign(out_height * out_width * out_channels, 0.0f);
    
    // Ajouter les bias
    if (!bias.empty()) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int i = 0; i < out_height * out_width; ++i) {
                output[oc * out_height * out_width + i] = bias[oc];
            }
        }
    }
    
    // Convolution transposée
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int ih = 0; ih < in_height; ++ih) {
                for (int iw = 0; iw < in_width; ++iw) {
                    float in_val = input[(ic * in_height + ih) * in_width + iw];
                    
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int oh = ih * stride + kh - padding;
                            int ow = iw * stride + kw - padding;
                            
                            if (oh >= 0 && oh < out_height && ow >= 0 && ow < out_width) {
                                int kernel_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                int out_idx = (oc * out_height + oh) * out_width + ow;
                                output[out_idx] += in_val * kernel[kernel_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace Conv

// ============================================================================
// Opérations de pooling
// ============================================================================

namespace Pooling {

// Max Pooling 2D
inline void maxpool2d(const std::vector<float>& input, std::vector<float>& output,
                      int in_height, int in_width, int channels,
                      int kernel_size, int stride = -1) {
    if (stride < 0) stride = kernel_size;
    
    int out_height = (in_height - kernel_size) / stride + 1;
    int out_width = (in_width - kernel_size) / stride + 1;
    
    output.resize(out_height * out_width * channels);
    
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                float max_val = -std::numeric_limits<float>::infinity();
                
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        
                        int in_idx = (c * in_height + ih) * in_width + iw;
                        max_val = std::max(max_val, input[in_idx]);
                    }
                }
                
                output[(c * out_height + oh) * out_width + ow] = max_val;
            }
        }
    }
}

// Average Pooling 2D
inline void avgpool2d(const std::vector<float>& input, std::vector<float>& output,
                      int in_height, int in_width, int channels,
                      int kernel_size, int stride = -1) {
    if (stride < 0) stride = kernel_size;
    
    int out_height = (in_height - kernel_size) / stride + 1;
    int out_width = (in_width - kernel_size) / stride + 1;
    
    output.resize(out_height * out_width * channels);
    
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                float sum = 0.0f;
                
                for (int kh = 0; kh < kernel_size; ++kh) {
                    for (int kw = 0; kw < kernel_size; ++kw) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        
                        int in_idx = (c * in_height + ih) * in_width + iw;
                        sum += input[in_idx];
                    }
                }
                
                output[(c * out_height + oh) * out_width + ow] = sum / (kernel_size * kernel_size);
            }
        }
    }
}

// Adaptive Average Pooling 2D
inline void adaptive_avgpool2d(const std::vector<float>& input, std::vector<float>& output,
                                int in_height, int in_width, int channels,
                                int out_height, int out_width) {
    output.resize(out_height * out_width * channels);
    
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                int h_start = static_cast<int>(std::floor(static_cast<float>(oh * in_height) / out_height));
                int h_end = static_cast<int>(std::ceil(static_cast<float>((oh + 1) * in_height) / out_height));
                int w_start = static_cast<int>(std::floor(static_cast<float>(ow * in_width) / out_width));
                int w_end = static_cast<int>(std::ceil(static_cast<float>((ow + 1) * in_width) / out_width));
                
                float sum = 0.0f;
                int count = 0;
                
                for (int ih = h_start; ih < h_end; ++ih) {
                    for (int iw = w_start; iw < w_end; ++iw) {
                        int in_idx = (c * in_height + ih) * in_width + iw;
                        sum += input[in_idx];
                        count++;
                    }
                }
                
                output[(c * out_height + oh) * out_width + ow] = sum / count;
            }
        }
    }
}

} // namespace Pooling

// ============================================================================
// Opérations de normalisation
// ============================================================================

namespace Normalization {

// Batch Normalization
inline void batch_norm(std::vector<float>& data, const std::vector<float>& gamma,
                       const std::vector<float>& beta, const std::vector<float>& running_mean,
                       const std::vector<float>& running_var, int batch_size, int channels,
                       int spatial_size, float eps = 1e-5f, bool training = false) {
    
    for (int c = 0; c < channels; ++c) {
        float mean = running_mean[c];
        float var = running_var[c];
        
        if (training) {
            // Calculer mean et var pour ce batch
            mean = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                for (int s = 0; s < spatial_size; ++s) {
                    mean += data[b * channels * spatial_size + c * spatial_size + s];
                }
            }
            mean /= (batch_size * spatial_size);
            
            var = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                for (int s = 0; s < spatial_size; ++s) {
                    float diff = data[b * channels * spatial_size + c * spatial_size + s] - mean;
                    var += diff * diff;
                }
            }
            var /= (batch_size * spatial_size);
        }
        
        float inv_std = 1.0f / std::sqrt(var + eps);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < spatial_size; ++s) {
                int idx = b * channels * spatial_size + c * spatial_size + s;
                data[idx] = (data[idx] - mean) * inv_std * gamma[c] + beta[c];
            }
        }
    }
}

// Layer Normalization
inline void layer_norm(std::vector<float>& data, const std::vector<float>& gamma,
                       const std::vector<float>& beta, int normalized_size,
                       float eps = 1e-5f) {
    
    int num_groups = data.size() / normalized_size;
    
    for (int g = 0; g < num_groups; ++g) {
        float mean = 0.0f;
        for (int i = 0; i < normalized_size; ++i) {
            mean += data[g * normalized_size + i];
        }
        mean /= normalized_size;
        
        float var = 0.0f;
        for (int i = 0; i < normalized_size; ++i) {
            float diff = data[g * normalized_size + i] - mean;
            var += diff * diff;
        }
        var /= normalized_size;
        
        float inv_std = 1.0f / std::sqrt(var + eps);
        
        for (int i = 0; i < normalized_size; ++i) {
            int idx = g * normalized_size + i;
            data[idx] = (data[idx] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

// RMS Normalization (Root Mean Square)
inline void rms_norm(std::vector<float>& data, const std::vector<float>& gamma,
                     int normalized_size, float eps = 1e-6f) {
    
    int num_groups = data.size() / normalized_size;
    
    for (int g = 0; g < num_groups; ++g) {
        float rms = 0.0f;
        for (int i = 0; i < normalized_size; ++i) {
            float val = data[g * normalized_size + i];
            rms += val * val;
        }
        rms = std::sqrt(rms / normalized_size + eps);
        
        for (int i = 0; i < normalized_size; ++i) {
            int idx = g * normalized_size + i;
            data[idx] = data[idx] / rms * gamma[i];
        }
    }
}

} // namespace Normalization

// ============================================================================
// Opérations d'accumulation et de comparaison
// ============================================================================

namespace Operations {

// Addition élément par élément
inline void add(std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        a[i] += b[i];
    }
}

// Soustraction élément par élément
inline void subtract(std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        a[i] -= b[i];
    }
}

// Multiplication élément par élément (Hadamard)
inline void multiply(std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        a[i] *= b[i];
    }
}

// Division élément par élément
inline void divide(std::vector<float>& a, const std::vector<float>& b, float eps = 1e-8f) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        a[i] /= (b[i] + eps);
    }
}

// Multiplication scalaire
inline void scale(std::vector<float>& a, float scalar) {
    for (auto& val : a) {
        val *= scalar;
    }
}

// Accumulation pondérée: a = alpha * a + beta * b
inline void accumulate(std::vector<float>& a, const std::vector<float>& b, 
                       float alpha = 1.0f, float beta = 1.0f) {
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        a[i] = alpha * a[i] + beta * b[i];
    }
}

// Clipping
inline void clip(std::vector<float>& a, float min_val, float max_val) {
    for (auto& val : a) {
        val = std::clamp(val, min_val, max_val);
    }
}

// Comparaison: retourne 1.0 si condition vraie, 0.0 sinon
inline void greater_than(const std::vector<float>& a, const std::vector<float>& b, 
                         std::vector<float>& result) {
    result.resize(std::min(a.size(), b.size()));
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (a[i] > b[i]) ? 1.0f : 0.0f;
    }
}

inline void less_than(const std::vector<float>& a, const std::vector<float>& b,
                      std::vector<float>& result) {
    result.resize(std::min(a.size(), b.size()));
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (a[i] < b[i]) ? 1.0f : 0.0f;
    }
}

inline void equal(const std::vector<float>& a, const std::vector<float>& b,
                  std::vector<float>& result, float tolerance = 1e-6f) {
    result.resize(std::min(a.size(), b.size()));
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = (std::abs(a[i] - b[i]) < tolerance) ? 1.0f : 0.0f;
    }
}

// Réduction: somme
inline float sum(const std::vector<float>& a) {
    float s = 0.0f;
    for (auto val : a) s += val;
    return s;
}

// Réduction: moyenne
inline float mean(const std::vector<float>& a) {
    return a.empty() ? 0.0f : sum(a) / a.size();
}

// Réduction: max
inline float max(const std::vector<float>& a) {
    return a.empty() ? 0.0f : *std::max_element(a.begin(), a.end());
}

// Réduction: min
inline float min(const std::vector<float>& a) {
    return a.empty() ? 0.0f : *std::min_element(a.begin(), a.end());
}

// Norme L2
inline float norm_l2(const std::vector<float>& a) {
    float s = 0.0f;
    for (auto val : a) s += val * val;
    return std::sqrt(s);
}

// Produit scalaire (dot product)
inline float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

} // namespace Operations
