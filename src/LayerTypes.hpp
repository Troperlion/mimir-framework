#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <iostream>
#include <vector>

// ============================================================================
// LAYER TYPE REGISTRY - Tous les types de layers supportés
// ============================================================================

enum class LayerType {
    // === Convolution ===
    Conv2d,
    ConvTranspose2d,
    Conv1d,
    DepthwiseConv2d,
    
    // === Linear / Dense ===
    Linear,
    Bilinear,
    
    // === Embedding ===
    Embedding,
    EmbeddingBag,
    
    // === Normalization ===
    BatchNorm2d,
    BatchNorm1d,
    LayerNorm,
    GroupNorm,
    InstanceNorm2d,
    RMSNorm,
    
    // === Activation ===
    ReLU,
    LeakyReLU,
    GELU,
    GEGLU,
    SiLU,           // Swish
    Tanh,
    Sigmoid,
    Softmax,
    LogSoftmax,
    Softplus,
    Mish,
    HardSigmoid,
    HardSwish,
    
    // === Pooling ===
    MaxPool2d,
    AvgPool2d,
    AdaptiveAvgPool2d,
    GlobalAvgPool2d,
    MaxPool1d,
    AvgPool1d,
    TokenMeanPool,
    
    // === Dropout / Regularization ===
    Dropout,
    Dropout2d,
    AlphaDropout,
    
    // === Shape Operations ===
    Flatten,
    Reshape,
    Transpose,
    Permute,
    Squeeze,
    Unsqueeze,
    View,
    
    // === Element-wise Operations ===
    Add,
    Subtract,
    Multiply,
    Divide,

    // === VAE / Probabilistic ===
    Reparameterize,
    
    // === Tensor Operations ===
    Concat,
    Split,
    Chunk,
    Stack,
    MatMul,
    BatchMatMul,
    NMS,
    
    // === Attention ===
    SelfAttention,
    MultiHeadAttention,
    CrossAttention,
    
    // === Upsampling ===
    UpsampleNearest,
    UpsampleBilinear,
    UpsampleBicubic,
    PixelShuffle,
    
    // === Recurrent (si nécessaire) ===
    LSTM,
    GRU,
    RNN,
    
    // === Padding ===
    ZeroPad2d,
    ReflectionPad2d,
    ReplicationPad2d,
    
    // === Special ===
    Identity,
    Constant,
    Lambda,

    // === Custom (Mímir) ===
    PatchEmbed,
    
    // === Total ===
    UNKNOWN
};

// ============================================================================
// MAPPING STRING <-> ENUM
// ============================================================================

namespace LayerRegistry {

// Normalisation des alias
inline std::string normalize_type(const std::string& type) {
    // Alias communs
    if (type == "BatchNorm") return "BatchNorm2d";
    if (type == "BN") return "BatchNorm2d";
    if (type == "BN2d") return "BatchNorm2d";
    if (type == "BN1d") return "BatchNorm1d";
    if (type == "LN") return "LayerNorm";
    if (type == "GN") return "GroupNorm";
    if (type == "IN") return "InstanceNorm2d";
    
    // Pooling
    if (type == "MaxPool") return "MaxPool2d";
    if (type == "AvgPool") return "AvgPool2d";
    if (type == "GlobalAvgPool") return "GlobalAvgPool2d";
    if (type == "AdaptiveAvgPool") return "AdaptiveAvgPool2d";
    
    // Activation
    if (type == "Swish") return "SiLU";
    if (type == "ReLu") return "ReLU";  // Typo commun
    if (type == "Relu") return "ReLU";
    if (type == "RELU") return "ReLU";
    if (type == "Gelu") return "GELU";
    if (type == "Geglu") return "GEGLU";
    if (type == "Silu") return "SiLU";
    if (type == "silu") return "SiLU";
    if (type == "swish") return "SiLU";
    
    // Shape ops
    if (type == "Concat") return "Concat";
    if (type == "Concatenate") return "Concat";
    if (type == "Cat") return "Concat";

    // Custom
    if (type == "PatchProjection") return "PatchEmbed";
    
    // Convolution
    if (type == "ConvTranspose") return "ConvTranspose2d";
    if (type == "Deconv2d") return "ConvTranspose2d";
    
    return type;  // Pas d'alias
}

// String -> Enum
inline LayerType string_to_type(const std::string& str) {
    static const std::unordered_map<std::string, LayerType> mapping = {
        // Convolution
        {"Conv2d", LayerType::Conv2d},
        {"ConvTranspose2d", LayerType::ConvTranspose2d},
        {"Conv1d", LayerType::Conv1d},
        {"DepthwiseConv2d", LayerType::DepthwiseConv2d},
        
        // Linear
        {"Linear", LayerType::Linear},
        {"Bilinear", LayerType::Bilinear},
        
        // Embedding
        {"Embedding", LayerType::Embedding},
        {"EmbeddingBag", LayerType::EmbeddingBag},
        
        // Normalization
        {"BatchNorm2d", LayerType::BatchNorm2d},
        {"BatchNorm1d", LayerType::BatchNorm1d},
        {"LayerNorm", LayerType::LayerNorm},
        {"GroupNorm", LayerType::GroupNorm},
        {"InstanceNorm2d", LayerType::InstanceNorm2d},
        {"RMSNorm", LayerType::RMSNorm},
        
        // Activation
        {"ReLU", LayerType::ReLU},
        {"LeakyReLU", LayerType::LeakyReLU},
        {"GELU", LayerType::GELU},
        {"GEGLU", LayerType::GEGLU},
        {"SiLU", LayerType::SiLU},
        {"Tanh", LayerType::Tanh},
        {"Sigmoid", LayerType::Sigmoid},
        {"Softmax", LayerType::Softmax},
        {"LogSoftmax", LayerType::LogSoftmax},
        {"Softplus", LayerType::Softplus},
        {"Mish", LayerType::Mish},
        {"HardSigmoid", LayerType::HardSigmoid},
        {"HardSwish", LayerType::HardSwish},
        
        // Pooling
        {"MaxPool2d", LayerType::MaxPool2d},
        {"AvgPool2d", LayerType::AvgPool2d},
        {"AdaptiveAvgPool2d", LayerType::AdaptiveAvgPool2d},
        {"GlobalAvgPool2d", LayerType::GlobalAvgPool2d},
        {"MaxPool1d", LayerType::MaxPool1d},
        {"AvgPool1d", LayerType::AvgPool1d},
        {"TokenMeanPool", LayerType::TokenMeanPool},
        
        // Dropout
        {"Dropout", LayerType::Dropout},
        {"Dropout2d", LayerType::Dropout2d},
        {"AlphaDropout", LayerType::AlphaDropout},
        
        // Shape
        {"Flatten", LayerType::Flatten},
        {"Reshape", LayerType::Reshape},
        {"Transpose", LayerType::Transpose},
        {"Permute", LayerType::Permute},
        {"Squeeze", LayerType::Squeeze},
        {"Unsqueeze", LayerType::Unsqueeze},
        {"View", LayerType::View},
        
        // Element-wise
        {"Add", LayerType::Add},
        {"Subtract", LayerType::Subtract},
        {"Multiply", LayerType::Multiply},
        {"Divide", LayerType::Divide},

        // VAE
        {"Reparameterize", LayerType::Reparameterize},
        
        // Tensor ops
        {"Concat", LayerType::Concat},
        {"Split", LayerType::Split},
        {"Chunk", LayerType::Chunk},
        {"Stack", LayerType::Stack},
        {"MatMul", LayerType::MatMul},
        {"BatchMatMul", LayerType::BatchMatMul},
        {"NMS", LayerType::NMS},
        {"NonMaxSuppression", LayerType::NMS},
        
        // Attention
        {"SelfAttention", LayerType::SelfAttention},
        {"MultiHeadAttention", LayerType::MultiHeadAttention},
        {"CrossAttention", LayerType::CrossAttention},
        
        // Upsampling
        {"UpsampleNearest", LayerType::UpsampleNearest},
        {"UpsampleBilinear", LayerType::UpsampleBilinear},
        {"UpsampleBicubic", LayerType::UpsampleBicubic},
        {"PixelShuffle", LayerType::PixelShuffle},
        
        // Recurrent
        {"LSTM", LayerType::LSTM},
        {"GRU", LayerType::GRU},
        {"RNN", LayerType::RNN},
        
        // Padding
        {"ZeroPad2d", LayerType::ZeroPad2d},
        {"ReflectionPad2d", LayerType::ReflectionPad2d},
        {"ReplicationPad2d", LayerType::ReplicationPad2d},
        
        // Special
        {"Identity", LayerType::Identity},
        {"Constant", LayerType::Constant},
        {"Lambda", LayerType::Lambda},

        // Custom
        {"PatchEmbed", LayerType::PatchEmbed}
    };
    
    std::string normalized = normalize_type(str);
    auto it = mapping.find(normalized);
    if (it != mapping.end()) {
        return it->second;
    }
    return LayerType::UNKNOWN;
}

// Enum -> String
inline std::string type_to_string(LayerType type) {
    switch (type) {
        // Convolution
        case LayerType::Conv2d: return "Conv2d";
        case LayerType::ConvTranspose2d: return "ConvTranspose2d";
        case LayerType::Conv1d: return "Conv1d";
        case LayerType::DepthwiseConv2d: return "DepthwiseConv2d";
        
        // Linear
        case LayerType::Linear: return "Linear";
        case LayerType::Bilinear: return "Bilinear";
        
        // Embedding
        case LayerType::Embedding: return "Embedding";
        case LayerType::EmbeddingBag: return "EmbeddingBag";
        
        // Normalization
        case LayerType::BatchNorm2d: return "BatchNorm2d";
        case LayerType::BatchNorm1d: return "BatchNorm1d";
        case LayerType::LayerNorm: return "LayerNorm";
        case LayerType::GroupNorm: return "GroupNorm";
        case LayerType::InstanceNorm2d: return "InstanceNorm2d";
        case LayerType::RMSNorm: return "RMSNorm";
        
        // Activation
        case LayerType::ReLU: return "ReLU";
        case LayerType::LeakyReLU: return "LeakyReLU";
        case LayerType::GELU: return "GELU";
        case LayerType::GEGLU: return "GEGLU";
        case LayerType::SiLU: return "SiLU";
        case LayerType::Tanh: return "Tanh";
        case LayerType::Sigmoid: return "Sigmoid";
        case LayerType::Softmax: return "Softmax";
        case LayerType::LogSoftmax: return "LogSoftmax";
        case LayerType::Softplus: return "Softplus";
        case LayerType::Mish: return "Mish";
        case LayerType::HardSigmoid: return "HardSigmoid";
        case LayerType::HardSwish: return "HardSwish";
        
        // Pooling
        case LayerType::MaxPool2d: return "MaxPool2d";
        case LayerType::AvgPool2d: return "AvgPool2d";
        case LayerType::AdaptiveAvgPool2d: return "AdaptiveAvgPool2d";
        case LayerType::GlobalAvgPool2d: return "GlobalAvgPool2d";
        case LayerType::MaxPool1d: return "MaxPool1d";
        case LayerType::AvgPool1d: return "AvgPool1d";
        case LayerType::TokenMeanPool: return "TokenMeanPool";
        
        // Dropout
        case LayerType::Dropout: return "Dropout";
        case LayerType::Dropout2d: return "Dropout2d";
        case LayerType::AlphaDropout: return "AlphaDropout";
        
        // Shape
        case LayerType::Flatten: return "Flatten";
        case LayerType::Reshape: return "Reshape";
        case LayerType::Transpose: return "Transpose";
        case LayerType::Permute: return "Permute";
        case LayerType::Squeeze: return "Squeeze";
        case LayerType::Unsqueeze: return "Unsqueeze";
        case LayerType::View: return "View";
        
        // Element-wise
        case LayerType::Add: return "Add";
        case LayerType::Subtract: return "Subtract";
        case LayerType::Multiply: return "Multiply";
        case LayerType::Divide: return "Divide";
        
        // Tensor ops
        case LayerType::Concat: return "Concat";
        case LayerType::Split: return "Split";
        case LayerType::Chunk: return "Chunk";
        case LayerType::Stack: return "Stack";
        case LayerType::MatMul: return "MatMul";
        case LayerType::BatchMatMul: return "BatchMatMul";
        case LayerType::NMS: return "NMS";
        
        // Attention
        case LayerType::SelfAttention: return "SelfAttention";
        case LayerType::MultiHeadAttention: return "MultiHeadAttention";
        case LayerType::CrossAttention: return "CrossAttention";
        
        // Upsampling
        case LayerType::UpsampleNearest: return "UpsampleNearest";
        case LayerType::UpsampleBilinear: return "UpsampleBilinear";
        case LayerType::UpsampleBicubic: return "UpsampleBicubic";
        case LayerType::PixelShuffle: return "PixelShuffle";
        
        // Recurrent
        case LayerType::LSTM: return "LSTM";
        case LayerType::GRU: return "GRU";
        case LayerType::RNN: return "RNN";
        
        // Padding
        case LayerType::ZeroPad2d: return "ZeroPad2d";
        case LayerType::ReflectionPad2d: return "ReflectionPad2d";
        case LayerType::ReplicationPad2d: return "ReplicationPad2d";
        
        // Special
        case LayerType::Identity: return "Identity";
        case LayerType::Constant: return "Constant";
        case LayerType::Lambda: return "Lambda";
        
        default: return "UNKNOWN";
    }
}

// Vérification du support
inline bool is_supported(LayerType type) {
    return type != LayerType::UNKNOWN;
}

inline bool is_supported(const std::string& type_str) {
    return string_to_type(type_str) != LayerType::UNKNOWN;
}

// Liste de tous les types supportés (pour messages d'erreur)
inline std::vector<std::string> get_all_supported_types() {
    return {
        "Conv2d", "ConvTranspose2d", "Conv1d", "DepthwiseConv2d",
        "Linear", "Bilinear",
        "Embedding", "EmbeddingBag", "PatchEmbed",
        "BatchNorm2d", "BatchNorm1d", "LayerNorm", "GroupNorm", "InstanceNorm2d", "RMSNorm",
        "ReLU", "LeakyReLU", "GELU", "GEGLU", "SiLU", "Tanh", "Sigmoid", "Softmax", "LogSoftmax", 
        "Softplus", "Mish", "HardSigmoid", "HardSwish",
        "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "GlobalAvgPool2d", "MaxPool1d", "AvgPool1d", "TokenMeanPool",
        "Dropout", "Dropout2d", "AlphaDropout",
        "Flatten", "Reshape", "Transpose", "Permute", "Squeeze", "Unsqueeze", "View",
        "Add", "Subtract", "Multiply", "Divide",
        "Concat", "Split", "Chunk", "Stack", "MatMul", "BatchMatMul", "NMS",
        "SelfAttention", "MultiHeadAttention", "CrossAttention",
        "UpsampleNearest", "UpsampleBilinear", "UpsampleBicubic", "PixelShuffle",
        "LSTM", "GRU", "RNN",
        "ZeroPad2d", "ReflectionPad2d", "ReplicationPad2d",
        "Identity", "Constant", "Reparameterize", "Lambda"
    };
}

// Log des types supportés
inline void log_supported_types() {
    auto types = get_all_supported_types();
    std::cerr << "📋 Layers supportés (" << types.size() << ") :\n";
    for (size_t i = 0; i < types.size(); ++i) {
        std::cerr << "  " << types[i];
        if ((i + 1) % 6 == 0) std::cerr << "\n";
        else if (i + 1 < types.size()) std::cerr << ", ";
    }
    std::cerr << std::endl;
}

} // namespace LayerRegistry
