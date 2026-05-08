#include "DebugJsonDump.hpp"
#include "../Model.hpp"
#include "../Tokenizer.hpp"
#include "../Encoder.hpp"
#include <fstream>
#include <cmath>
#include <iomanip>
#include <ctime>
#include <algorithm>

namespace Mimir {
namespace Serialization {

DebugJsonDump::DebugJsonDump() {
}

DebugJsonDump::~DebugJsonDump() {
}

bool DebugJsonDump::save(
    Model& model,
    const std::string& path,
    const SaveOptions& options,
    std::string* error
) {
    try {
        // Build JSON representation
        json debug_json = build_json(model, options);
        
        // Ensure directory exists
        fs::path file_path(path);
        if (file_path.has_parent_path()) {
            fs::create_directories(file_path.parent_path());
        }
        
        // Write to file
        std::ofstream file(path);
        if (!file) {
            if (error) {
                *error = "Failed to create file: " + path;
            }
            return false;
        }
        
        file << std::setw(2) << debug_json;
        file.close();
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Debug JSON dump error: ") + e.what();
        }
        return false;
    }
}

DebugJsonDump::TensorStats DebugJsonDump::calculate_stats(
    const float* data,
    size_t size
) {
    TensorStats stats;
    stats.total_elements = size;
    
    if (size == 0 || data == nullptr) {
        stats.min = stats.max = stats.mean = stats.std = stats.l2_norm = 0.0f;
        return stats;
    }
    
    // Calculate min, max, mean, L2 norm
    float sum = 0.0f;
    double l2_sum = 0.0;
    stats.min = data[0];
    stats.max = data[0];
    
    for (size_t i = 0; i < size; ++i) {
        float val = data[i];
        sum += val;
        l2_sum += static_cast<double>(val) * static_cast<double>(val);
        if (val < stats.min) stats.min = val;
        if (val > stats.max) stats.max = val;
    }
    
    stats.mean = sum / static_cast<float>(size);
    stats.l2_norm = std::sqrt(l2_sum);
    
    // Calculate standard deviation
    float var_sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = data[i] - stats.mean;
        var_sum += diff * diff;
    }
    stats.std = std::sqrt(var_sum / static_cast<float>(size));
    
    return stats;
}

DebugJsonDump::WeightDelta DebugJsonDump::calculate_delta(
    const float* before,
    const float* after,
    size_t size
) {
    WeightDelta delta;
    delta.changed = false;
    delta.delta_l2_norm = 0.0;
    delta.delta_max_abs = 0.0;
    delta.relative_change = 0.0;
    
    if (size == 0 || before == nullptr || after == nullptr) {
        return delta;
    }
    
    double l2_sum = 0.0;
    double max_abs = 0.0;
    double before_l2 = 0.0;
    
    for (size_t i = 0; i < size; ++i) {
        double diff = static_cast<double>(after[i]) - static_cast<double>(before[i]);
        double abs_diff = std::abs(diff);
        
        l2_sum += diff * diff;
        if (abs_diff > max_abs) {
            max_abs = abs_diff;
        }
        
        before_l2 += static_cast<double>(before[i]) * static_cast<double>(before[i]);
    }
    
    delta.delta_l2_norm = std::sqrt(l2_sum);
    delta.delta_max_abs = max_abs;
    delta.changed = (delta.delta_max_abs > 1e-9);  // Threshold for float precision
    
    // Relative change
    double before_norm = std::sqrt(before_l2);
    if (before_norm > 1e-9) {
        delta.relative_change = delta.delta_l2_norm / before_norm;
    }
    
    return delta;
}

uint64_t DebugJsonDump::compute_checksum(const float* data, size_t size) {
    if (data == nullptr || size == 0) return 0;
    
    // Simple hash - sample every Nth element for speed
    uint64_t hash = 0;
    size_t step = std::max(size_t(1), size / 1000);
    
    for (size_t i = 0; i < size; i += step) {
        uint32_t bits;
        std::memcpy(&bits, &data[i], sizeof(float));
        hash = hash * 31 + bits;
    }
    
    return hash;
}

json DebugJsonDump::build_json(
    Model& model,
    const SaveOptions& options
) {
    json root;
    
    // Metadata
    root["format"] = "mimir_debug_dump";
    root["format_version"] = "1.0.0";
    root["mimir_version"] = get_mimir_version();
    
    if (options.include_git_info) {
        std::string git_commit = get_git_commit();
        if (!git_commit.empty()) {
            root["git_commit"] = git_commit;
        }
    }
    
    auto now = std::time(nullptr);
    root["created_at"] = static_cast<long long>(now);
    
    root["warning"] = "This is a debug dump. NOT for production use or large models.";
    
    // Model info
    json model_info;
    model_info["name"] = model.getModelName();
    model_info["total_params"] = model.totalParamCount();
    model_info["num_layers"] = model.getLayers().size();
    
    if (model.width() > 0 && model.height() > 0) {
        model_info["image_width"] = model.width();
        model_info["image_height"] = model.height();
    }
    
    root["model"] = model_info;
    
    // Layers
    json layers_array = json::array();
    for (const auto& layer : model.getLayers()) {
        json layer_obj;
        layer_obj["name"] = layer.name;
        layer_obj["type"] = layer.type;
        layer_obj["params_count"] = layer.params_count;
        layers_array.push_back(layer_obj);
    }
    root["layers"] = layers_array;
    
    // Tensors (truncated)
    json tensors_array = json::array();
    
    const auto& layers = model.getLayers();
    
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto& layer = layers[layer_idx];
        
        if (!layer.weight_block) {
            continue;  // No weights
        }
        
        const float* data = layer.weight_block->getData();
        size_t size = layer.weight_block->getSize();
        
        if (data == nullptr || size == 0) {
            continue;  // Skip empty tensors
        }
        
        json tensor_obj;
        tensor_obj["name"] = layer.name + "/weights";
        tensor_obj["dtype"] = "F32";
        
        tensor_obj["shape"] = json::array({size});
        tensor_obj["total_elements"] = size;
        
        // Statistics
        TensorStats stats = calculate_stats(data, size);
        json stats_obj;
        stats_obj["min"] = stats.min;
        stats_obj["max"] = stats.max;
        stats_obj["mean"] = stats.mean;
        stats_obj["std"] = stats.std;
        tensor_obj["stats"] = stats_obj;
        
        // Sample values (first N)
        size_t sample_size = std::min(size, options.debug_max_values);
        json sample_array = json::array();
        for (size_t j = 0; j < sample_size; ++j) {
            sample_array.push_back(data[j]);
        }
        tensor_obj["sample_values"] = sample_array;
        tensor_obj["sample_size"] = sample_size;
        
        if (size > sample_size) {
            tensor_obj["truncated"] = true;
        }
        
        tensors_array.push_back(tensor_obj);
    }
    
    root["tensors"] = tensors_array;
    
    // Tokenizer info (if present)
    if (options.save_tokenizer) {
        const auto& tokenizer = model.getTokenizer();
        json tok_info;
        tok_info["vocab_size"] = tokenizer.getVocabSize();
        tok_info["has_vocab"] = (tokenizer.getVocabSize() > 0);
        root["tokenizer"] = tok_info;
    }
    
    // Encoder info (if present)
    if (options.save_encoder && model.getHasEncoder()) {
        const auto& encoder = model.getEncoder();
        json enc_info;
        enc_info["dim"] = encoder.dim;
        enc_info["vocab_size"] = encoder.vocab_size;
        enc_info["embeddings_size"] = encoder.token_embeddings.size();
        
        // Sample embeddings stats
        if (!encoder.token_embeddings.empty()) {
            TensorStats stats = calculate_stats(
                encoder.token_embeddings.data(),
                encoder.token_embeddings.size()
            );
            json stats_obj;
            stats_obj["min"] = stats.min;
            stats_obj["max"] = stats.max;
            stats_obj["mean"] = stats.mean;
            stats_obj["std"] = stats.std;
            enc_info["embeddings_stats"] = stats_obj;
        }
        
        root["encoder"] = enc_info;
    }
    
    return root;
}

json DebugJsonDump::extract_layer_config(const Layer& layer) {
    json config;
    
    // Common fields
    if (layer.in_channels > 0) config["in_channels"] = layer.in_channels;
    if (layer.out_channels > 0) config["out_channels"] = layer.out_channels;
    
    // Type-specific configurations
    switch (layer.type_enum) {
        case LayerType::Conv2d:
            config["kernel_h"] = layer.get_kernel_h();
            config["kernel_w"] = layer.get_kernel_w();
            config["stride_h"] = layer.get_stride_h();
            config["stride_w"] = layer.get_stride_w();
            config["pad_h"] = layer.get_pad_h();
            config["pad_w"] = layer.get_pad_w();
            config["dilation"] = layer.dilation;
            config["groups"] = layer.groups;
            config["has_bias"] = layer.use_bias;
            break;
            
        case LayerType::Linear:
            config["in_features"] = layer.in_features;
            config["out_features"] = layer.out_features;
            config["has_bias"] = !layer.bias.empty();
            break;
            
        case LayerType::BatchNorm2d:
        case LayerType::LayerNorm:
        case LayerType::InstanceNorm2d:
        case LayerType::GroupNorm:
        case LayerType::RMSNorm:
            if (layer.in_channels > 0) config["num_features"] = layer.in_channels;
            config["eps"] = layer.eps;
            if (layer.momentum > 0) config["momentum"] = layer.momentum;
            if (layer.num_groups > 0) config["num_groups"] = layer.num_groups;
            break;
            
        case LayerType::MaxPool2d:
        case LayerType::AvgPool2d:
            config["kernel_size"] = layer.kernel_size;
            config["stride"] = layer.stride;
            config["padding"] = layer.padding;
            break;
            
        case LayerType::SelfAttention:
        case LayerType::MultiHeadAttention:
            config["num_heads"] = layer.num_heads;
            config["head_dim"] = layer.head_dim;
            if (layer.seq_len > 0) config["seq_len"] = layer.seq_len;
            config["causal"] = layer.causal;
            break;
            
        case LayerType::LeakyReLU:
            config["negative_slope"] = layer.negative_slope;
            break;
            
        case LayerType::Dropout:
            config["dropout_p"] = layer.dropout_p;
            break;
            
        default:
            // For other layers, add generic info
            break;
    }
    
    return config;
}

std::vector<size_t> DebugJsonDump::get_tensor_shape(const Layer& layer, bool is_bias) {
    std::vector<size_t> shape;
    
    if (is_bias) {
        // Bias is always 1D (taille gérée au callsite en mode weight_block)
        if (!layer.bias.empty()) {
            shape.push_back(layer.bias.size());
        }
        return shape;
    }
    
    // Weight shapes depend on layer type
    switch (layer.type_enum) {
        case LayerType::Conv2d:
            // [out_channels, in_channels, kernel_h, kernel_w]
            if (layer.out_channels > 0) shape.push_back(static_cast<size_t>(layer.out_channels));
            if (layer.in_channels > 0) {
                int groups = layer.groups > 0 ? layer.groups : 1;
                shape.push_back(static_cast<size_t>(layer.in_channels / groups));
            }
            shape.push_back(static_cast<size_t>(layer.get_kernel_h()));
            shape.push_back(static_cast<size_t>(layer.get_kernel_w()));
            break;
            
        case LayerType::Linear:
            // [out_features, in_features] or [in_features, out_features]
            // Using [out, in] convention
            shape.push_back(layer.out_features);
            shape.push_back(layer.in_features);
            break;
            
        case LayerType::BatchNorm2d:
        case LayerType::LayerNorm:
        case LayerType::InstanceNorm2d:
        case LayerType::GroupNorm:
        case LayerType::RMSNorm:
            // [num_features]
            if (layer.in_channels > 0) {
                shape.push_back(layer.in_channels);
            }
            break;
            
        case LayerType::Embedding:
            // [vocab_size, embedding_dim]
            shape.push_back(layer.vocab_size);
            shape.push_back(layer.embed_dim);
            break;
            
        default:
            // Fallback: 1D shape
            if (!layer.weights.empty()) {
                shape.push_back(layer.weights.size());
            }
            break;
    }
    
    return shape;
}

void DebugJsonDump::add_tensor_enhanced(
    json& parent,
    const std::string& name,
    const float* data,
    const std::vector<size_t>& shape,
    const float* grad_data,
    size_t max_values,
    bool include_grads
) {
    json tensor_obj;
    tensor_obj["name"] = name;
    tensor_obj["dtype"] = "F32";
    
    // Real shape (not just size)
    json shape_array = json::array();
    size_t total_size = 1;
    for (size_t dim : shape) {
        shape_array.push_back(dim);
        total_size *= dim;
    }
    tensor_obj["shape"] = shape_array;
    tensor_obj["total_elements"] = total_size;
    
    if (data != nullptr && total_size > 0) {
        // Statistics
        TensorStats stats = calculate_stats(data, total_size);
        json stats_obj;
        stats_obj["min"] = stats.min;
        stats_obj["max"] = stats.max;
        stats_obj["mean"] = stats.mean;
        stats_obj["std"] = stats.std;
        stats_obj["l2_norm"] = stats.l2_norm;
        tensor_obj["stats"] = stats_obj;
        
        // Sample values (first N)
        size_t sample_size = std::min(total_size, max_values);
        json sample_array = json::array();
        for (size_t j = 0; j < sample_size; ++j) {
            sample_array.push_back(data[j]);
        }
        tensor_obj["sample_values"] = sample_array;
        tensor_obj["sample_size"] = sample_size;
        
        if (total_size > sample_size) {
            tensor_obj["truncated"] = true;
        }
        
        // Gradients (if requested and available)
        if (include_grads && grad_data != nullptr) {
            TensorStats grad_stats = calculate_stats(grad_data, total_size);
            json grad_stats_obj;
            grad_stats_obj["min"] = grad_stats.min;
            grad_stats_obj["max"] = grad_stats.max;
            grad_stats_obj["mean"] = grad_stats.mean;
            grad_stats_obj["std"] = grad_stats.std;
            grad_stats_obj["l2_norm"] = grad_stats.l2_norm;
            
            // Check if gradients are all zero
            bool all_zero = (grad_stats.l2_norm < 1e-9);
            grad_stats_obj["all_zero"] = all_zero;
            
            tensor_obj["gradients"] = grad_stats_obj;
            
            // Sample gradient values
            json grad_sample = json::array();
            for (size_t j = 0; j < sample_size; ++j) {
                grad_sample.push_back(grad_data[j]);
            }
            tensor_obj["gradient_sample"] = grad_sample;
        }
    }
    
    parent.push_back(tensor_obj);
}

void DebugJsonDump::add_tensor_info(
    json& parent,
    const std::string& name,
    const float* data,
    size_t size,
    size_t max_values
) {
    json tensor_obj;
    tensor_obj["name"] = name;
    tensor_obj["size"] = size;
    
    if (data != nullptr && size > 0) {
        TensorStats stats = calculate_stats(data, size);
        json stats_obj;
        stats_obj["min"] = stats.min;
        stats_obj["max"] = stats.max;
        stats_obj["mean"] = stats.mean;
        stats_obj["std"] = stats.std;
        tensor_obj["stats"] = stats_obj;
        
        size_t sample_size = std::min(size, max_values);
        json sample = json::array();
        for (size_t i = 0; i < sample_size; ++i) {
            sample.push_back(data[i]);
        }
        tensor_obj["sample"] = sample;
        
        if (size > sample_size) {
            tensor_obj["truncated"] = true;
        }
    }
    
    parent.push_back(tensor_obj);
}

json DebugJsonDump::build_json_enhanced(const Model& model, const DebugJsonOptions& options) {
    json root;
    
    // Version and format
    root["format"] = "mimir_debug_dump";
    root["format_version"] = "1.1.0";
    root["timestamp"] = std::time(nullptr);
    root["mimir_version"] = get_mimir_version();
    root["model_name"] = model.getModelName();
    root["model_type"] = model.getModelName();
    root["debug_only"] = true;
    root["warning"] = "This serialization is DEBUG-ONLY (debug_json). It is write-only, not intended for production checkpoints or large models.";
    
    // Features supported in this version
    json features = json::array();
    features.push_back("layer_config");
    features.push_back("real_shapes");
    if (options.include_gradients) features.push_back("gradients");
    if (options.include_weight_deltas) features.push_back("weight_deltas");
    if (options.include_optimizer_state) features.push_back("optimizer_state");
    if (options.include_checksums) features.push_back("checksums");
    root["features"] = features;
    
    // Model statistics
    root["total_params"] = model.totalParamCount();
    root["num_layers"] = model.getLayers().size();

    // Optional captured gradients: no longer supported (VAEModel removed)
    const std::unordered_map<std::string, std::vector<float>>* captured_grads = nullptr;
    
    // Optimizer state (if requested)
    if (options.include_optimizer_state) {
        json opt_state;
        if (const Optimizer* opt = model.getSerializedOptimizer()) {
            auto type_to_string = [&](OptimizerType t) {
                switch (t) {
                    case OptimizerType::SGD: return "sgd";
                    case OptimizerType::ADAM: return "adam";
                    case OptimizerType::ADAMW: return "adamw";
                    default: return "unknown";
                }
            };

            opt_state["type"] = type_to_string(opt->type);
            opt_state["step"] = opt->step;
            opt_state["lr_current"] = opt->getCurrentLR();
            opt_state["beta1"] = opt->beta1;
            opt_state["beta2"] = opt->beta2;
            opt_state["eps"] = opt->eps;
            opt_state["weight_decay"] = opt->weight_decay;
            opt_state["decay_strategy"] = static_cast<int>(opt->decay_strategy);
            opt_state["initial_lr"] = opt->initial_lr;
            opt_state["min_lr"] = opt->min_lr;
            opt_state["decay_rate"] = opt->decay_rate;
            opt_state["decay_steps"] = opt->decay_steps;
            opt_state["total_steps"] = opt->total_steps;
            opt_state["warmup_steps"] = opt->warmup_steps;

            // State vectors (debug-only): stats + small sample
            if (!opt->m.empty()) {
                TensorStats s = calculate_stats(opt->m.data(), opt->m.size());
                json mj;
                mj["size"] = opt->m.size();
                mj["stats"] = {
                    {"min", s.min}, {"max", s.max}, {"mean", s.mean}, {"std", s.std},
                    {"l2_norm", s.l2_norm}, {"total_elements", s.total_elements}
                };
                const size_t sample_n = std::min<size_t>(options.max_values_per_tensor, opt->m.size());
                json sample = json::array();
                for (size_t i = 0; i < sample_n; ++i) sample.push_back(opt->m[i]);
                mj["sample_values"] = sample;
                mj["truncated"] = (opt->m.size() > sample_n);
                opt_state["m"] = mj;
            }
            if (!opt->v.empty()) {
                TensorStats s = calculate_stats(opt->v.data(), opt->v.size());
                json vj;
                vj["size"] = opt->v.size();
                vj["stats"] = {
                    {"min", s.min}, {"max", s.max}, {"mean", s.mean}, {"std", s.std},
                    {"l2_norm", s.l2_norm}, {"total_elements", s.total_elements}
                };
                const size_t sample_n = std::min<size_t>(options.max_values_per_tensor, opt->v.size());
                json sample = json::array();
                for (size_t i = 0; i < sample_n; ++i) sample.push_back(opt->v[i]);
                vj["sample_values"] = sample;
                vj["truncated"] = (opt->v.size() > sample_n);
                opt_state["v"] = vj;
            }
        } else {
            opt_state["available"] = false;
            opt_state["note"] = "No serialized optimizer found on model";
        }
        root["optimizer"] = opt_state;
    }
    
    // Layer information with configs
    json layers_array = json::array();
    const auto& layers = model.getLayers();
    
    for (size_t i = 0; i < layers.size(); ++i) {
        const Layer& layer = layers[i];
        
        json layer_obj;
        layer_obj["index"] = i;
        layer_obj["name"] = layer.name;
        layer_obj["type"] = layer.type;
        layer_obj["params_count"] = layer.params_count;
        
        // Extract layer-specific configuration
        json config = extract_layer_config(layer);
        if (!config.empty()) {
            layer_obj["config"] = config;
        }
        
        // Tensors array for this layer
        json tensors = json::array();

        // Weights/Bias (supporte weight_block)
        const float* wb = layer.getWeights();
        const size_t wb_size = layer.getWeightsSize();
        if (wb && wb_size > 0) {
            // Heuristique de séparation weights/bias pour Conv2d (layout: weights puis bias)
            size_t weight_count = wb_size;
            size_t bias_count = 0;

            if (layer.type_enum == LayerType::Conv2d && layer.in_channels > 0 && layer.out_channels > 0) {
                const size_t oc = static_cast<size_t>(layer.out_channels);
                const size_t ic = static_cast<size_t>(layer.in_channels);
                const size_t groups = static_cast<size_t>(layer.groups > 0 ? layer.groups : 1);
                const size_t kh = static_cast<size_t>(layer.get_kernel_h());
                const size_t kw = static_cast<size_t>(layer.get_kernel_w());

                const size_t conv_w = oc * (ic / groups) * kh * kw;
                if (conv_w > 0 && wb_size >= conv_w) {
                    weight_count = conv_w;
                    // Bias optionnel si le buffer contient oc éléments supplémentaires
                    if (layer.use_bias && wb_size >= conv_w + oc) {
                        bias_count = oc;
                    } else if (wb_size > conv_w) {
                        bias_count = wb_size - conv_w;
                    }
                }
            }

            // --- Weight tensor ---
            {
                std::vector<size_t> weight_shape = get_tensor_shape(layer, false);
                const float* grad_ptr = nullptr;
                if (options.include_gradients) {
                    // Prefer captured gradients (pre-optimizerStep), fall back to live grad_weights.
                    if (captured_grads) {
                        auto itg = captured_grads->find(layer.name);
                        if (itg != captured_grads->end() && itg->second.size() >= weight_count) {
                            grad_ptr = itg->second.data();
                        }
                    }
                    if (!grad_ptr && !layer.grad_weights.empty() && layer.grad_weights.size() >= weight_count) {
                        grad_ptr = layer.grad_weights.data();
                    }
                }

                add_tensor_enhanced(
                    tensors,
                    layer.name + ".weight",
                    wb,
                    weight_shape.empty() ? std::vector<size_t>{weight_count} : weight_shape,
                    grad_ptr,
                    options.max_values_per_tensor,
                    options.include_gradients
                );

                if (options.include_checksums) {
                    uint64_t checksum = compute_checksum(wb, weight_count);
                    layer_obj["weight_checksum"] = std::to_string(checksum);
                }

                if (options.include_weight_deltas) {
                    std::string snapshot_key = layer.name + ".weight";
                    auto it = weight_snapshots_.find(snapshot_key);
                    if (it != weight_snapshots_.end() && it->second.size() == weight_count) {
                        WeightDelta delta = calculate_delta(
                            it->second.data(),
                            wb,
                            weight_count
                        );

                        json delta_obj;
                        delta_obj["changed"] = delta.changed;
                        delta_obj["delta_l2_norm"] = delta.delta_l2_norm;
                        delta_obj["delta_max_abs"] = delta.delta_max_abs;
                        delta_obj["relative_change"] = delta.relative_change;
                        layer_obj["weight_delta"] = delta_obj;
                    }

                    weight_snapshots_[snapshot_key] = std::vector<float>(wb, wb + weight_count);
                }
            }

            // --- Bias tensor (si séparé) ---
            if (bias_count > 0 && wb_size >= weight_count + bias_count) {
                const float* bias_ptr = wb + weight_count;
                const float* grad_bias_ptr = nullptr;
                if (options.include_gradients) {
                    if (captured_grads) {
                        auto itg = captured_grads->find(layer.name);
                        if (itg != captured_grads->end() && itg->second.size() >= weight_count + bias_count) {
                            grad_bias_ptr = itg->second.data() + weight_count;
                        }
                    }
                    if (!grad_bias_ptr && !layer.grad_weights.empty() && layer.grad_weights.size() >= weight_count + bias_count) {
                        grad_bias_ptr = layer.grad_weights.data() + weight_count;
                    }
                }

                add_tensor_enhanced(
                    tensors,
                    layer.name + ".bias",
                    bias_ptr,
                    std::vector<size_t>{bias_count},
                    grad_bias_ptr,
                    options.max_values_per_tensor,
                    options.include_gradients
                );

                if (options.include_checksums) {
                    uint64_t checksum = compute_checksum(bias_ptr, bias_count);
                    layer_obj["bias_checksum"] = std::to_string(checksum);
                }
            }
        }
        
        layer_obj["tensors"] = tensors;
        layers_array.push_back(layer_obj);
    }
    
    root["layers"] = layers_array;
    
    // Tokenizer section (ONLY if model actually has a tokenizer)
    if (options.save_tokenizer) {
        const Tokenizer& tok = model.getTokenizer();
        size_t vocab_size = tok.getVocabSize();
        if (vocab_size > 0) {
            json tokenizer_obj;
            tokenizer_obj["vocab_size"] = vocab_size;
            tokenizer_obj["has_vocab"] = true;
            root["tokenizer"] = tokenizer_obj;
        }
    }
    
    // Encoder section (ONLY if model actually has an encoder)
    if (options.save_encoder && model.getHasEncoder()) {
        const Encoder& enc = model.getEncoder();
        json encoder_obj;
        encoder_obj["has_encoder"] = true;
        // The current Encoder is a learned token-embedding table (not a ViT).
        encoder_obj["type"] = "token_embedding_table";
        encoder_obj["dim"] = enc.dim;
        encoder_obj["vocab_size"] = enc.vocab_size;
        root["encoder"] = encoder_obj;
    }
    
    // Git info (if requested)
    if (options.include_git_info) {
        std::string git_commit = get_git_commit();
        if (!git_commit.empty()) {
            root["git_commit"] = git_commit;
        }
    }
    
    return root;
}

bool DebugJsonDump::save_enhanced(
    const std::string& path,
    const Model& model,
    const DebugJsonOptions& options,
    std::string* error
) {
    try {
        json j = build_json_enhanced(model, options);
        
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cerr << "[DebugJsonDump] Failed to open file: " << path << std::endl;
            return false;
        }
        
        // Write with pretty formatting (4 spaces indent)
        file << std::setw(4) << j << std::endl;
        file.close();
        
        std::cout << "[DebugJsonDump] Saved enhanced debug JSON to: " << path << std::endl;
        std::cout << "[DebugJsonDump] NOTE: debug_json is DEBUG-ONLY (write-only) and not meant for production checkpoints." << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[DebugJsonDump] Error saving enhanced JSON: " << e.what() << std::endl;
        return false;
    }
}

} // namespace Serialization
} // namespace Mimir
