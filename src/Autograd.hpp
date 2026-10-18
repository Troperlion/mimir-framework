#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "TypedTensor.hpp"

// Structure pour stocker les activations du forward pass
struct ComputationGraph {
    // Activations intermédiaires
    std::vector<float> token_embeddings;
    std::vector<float> pos_encodings;
    std::vector<float> combined_input;
    
    // Par layer transformer
    struct LayerActivations {
        std::vector<float> input;
        std::vector<float> normed1;
        std::vector<float> attn_out;
        std::vector<float> after_attn;
        std::vector<float> normed2;
        std::vector<float> ffn_out;
        std::vector<float> output;
    };
    std::vector<LayerActivations> layers;
    
    std::vector<float> final_output;
    
    // Métadonnées
    std::vector<int> input_tokens;
    int sequence_length;
    int latent_dim;
};

// Structure pour stocker les gradients
struct Gradients {
    // Gradients par paramètre (indexés par offset dans params)
    std::unordered_map<size_t, float> param_grads;
    
    // Ajouter un gradient
    void add(size_t param_idx, float grad) {
        param_grads[param_idx] += grad;
    }
    
    // Récupérer un gradient
    float get(size_t param_idx) const {
        auto it = param_grads.find(param_idx);
        return it != param_grads.end() ? it->second : 0.0f;
    }
    
    // Réinitialiser
    void zero() {
        param_grads.clear();
    }
    
    // Clipper les gradients
    void clip(float max_norm) {
        if (!std::isfinite(max_norm) || max_norm < 0.0f) {
            throw std::invalid_argument("Gradients::clip: max_norm must be finite and non-negative");
        }
        double total_norm_sq = 0.0;
        for (const auto& [idx, grad] : param_grads) {
            (void)idx;
            if (!std::isfinite(grad)) {
                throw std::runtime_error("Gradients::clip: non-finite gradient");
            }
            total_norm_sq += static_cast<double>(grad) * static_cast<double>(grad);
        }
        const double total_norm = std::sqrt(total_norm_sq);
        
        if (total_norm > static_cast<double>(max_norm) && total_norm > 0.0) {
            const float scale = static_cast<float>(static_cast<double>(max_norm) / total_norm);
            for (auto& [idx, grad] : param_grads) {
                (void)idx;
                grad *= scale;
            }
        }
    }
};

// Fonctions de backprop pour chaque opération
namespace Autograd {
    inline Mimir::DType gradient_dtype(Mimir::DType primal) {
        if (!Mimir::dtype_is_floating(primal)) {
            throw std::invalid_argument(
                std::string("Autograd: dtype '") + Mimir::dtype_to_string(primal) +
                "' is not differentiable");
        }
        return primal == Mimir::DType::F64 ? Mimir::DType::F64 : Mimir::DType::F32;
    }

    inline Mimir::TypedTensor mse_backward(const Mimir::TypedTensor& pred,
                                           const Mimir::TypedTensor& target) {
        if (pred.shape() != target.shape())
            throw std::invalid_argument("Autograd::mse_backward: shape mismatch");
        const Mimir::DType grad_type = gradient_dtype(pred.dtype());
        if (!Mimir::dtype_is_floating(target.dtype()))
            throw std::invalid_argument("Autograd::mse_backward: target must be floating");
        Mimir::TypedTensor gradient(pred.shape(), grad_type);
        if (pred.numel() == 0) return gradient;
        const double inv_n = 1.0 / static_cast<double>(pred.numel());
        for (size_t index = 0; index < pred.numel(); ++index)
            gradient.set(index, 2.0 * (pred.get(index) - target.get(index)) * inv_n);
        return gradient;
    }

    // Gradient de MSE: dL/dx = 2(x - target) / n
    inline std::vector<float> mse_backward(const std::vector<float>& pred, 
                                           const std::vector<float>& target) {
        if (pred.size() != target.size()) {
            throw std::invalid_argument("Autograd::mse_backward: size mismatch");
        }
        if (pred.empty()) {
            return {};
        }
        std::vector<float> grad(pred.size());
        const float inv_n = 1.0f / static_cast<float>(pred.size());
        #pragma omp simd
        for (size_t i = 0; i < pred.size(); ++i) {
            grad[i] = 2.0f * (pred[i] - target[i]) * inv_n;
        }
        return grad;
    }
    
    // Compatibility entry points. Layer derivatives are dispatched to runtimes;
    // configure RuntimeRouter (normally through Model) before using these helpers.
    std::vector<float> layernorm_backward(const std::vector<float>& grad_output,
                                         const std::vector<float>& input,
                                         const std::vector<float>& normalized);
    float gelu_backward(float x, float grad_output);

    inline float sigmoid(float x) {
        if (x >= 0.0f) {
            const float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(x);
        return z / (1.0f + z);
    }

    inline float softplus(float x) {
        return x > 20.0f ? x : std::log1p(std::exp(x));
    }

    inline bool all_finite(const std::vector<float>& values) {
        for (float value : values) {
            if (!std::isfinite(value)) return false;
        }
        return true;
    }
    
    std::vector<float> residual_backward(const std::vector<float>& grad_output);
}
