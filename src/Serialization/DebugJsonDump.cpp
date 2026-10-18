#include "DebugJsonDump.hpp"
#include "../Model.hpp"
#include "CheckpointState.hpp"
#include "../Tokenizer.hpp"
#include "../Encoder.hpp"
#include "../LayerTypes.hpp"
#include "../MemoryGuard.hpp"
#include "../runtimes/cpu/DynamicTensorAllocator.hpp"
#include "../AdvancedRAMManager.hpp"
#include "../runtimes/AbstractRuntime.hpp"
#include <fstream>
#include <cmath>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <unordered_set>

namespace {

static nlohmann::json runtime_config_to_json(const RuntimeConfig& cfg) {
    nlohmann::json out;
    out["backend"] = cfg.backend;
    out["disabled"] = cfg.disabled;
    out["verbose"] = cfg.verbose;
    out["device_index"] = cfg.device_index;
    out["linear_enabled"] = cfg.linear_enabled;
    out["linear_min_ops"] = cfg.linear_min_ops;
    out["conv_enabled"] = cfg.conv_enabled;
    out["conv_min_ops"] = cfg.conv_min_ops;
    out["norm_enabled"] = cfg.norm_enabled;
    out["norm_min_elements"] = cfg.norm_min_elements;
    out["attention_enabled"] = cfg.attention_enabled;
    out["attention_min_ops"] = cfg.attention_min_ops;
    return out;
}

static void set_env_if_present(nlohmann::json& env, const char* key) {
    if (!key) return;
    if (const char* value = std::getenv(key)) {
        env[key] = value;
    }
}

static size_t shape_element_count(const std::vector<size_t>& shape) {
    if (shape.empty()) return 0;
    size_t count = 1;
    for (const size_t dim : shape) {
        if (dim == 0 || count > std::numeric_limits<size_t>::max() / dim) return 0;
        count *= dim;
    }
    return count;
}

static nlohmann::json json_number_or_null(double value) {
    return std::isfinite(value) ? nlohmann::json(value) : nlohmann::json(nullptr);
}

} // namespace

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
        json debug_json = build_json(model, effective_save_options(model, options));
        
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
    stats.finite_elements = 0;
    stats.zero_elements = 0;
    stats.nan_elements = 0;
    stats.pos_inf_elements = 0;
    stats.neg_inf_elements = 0;
    
    if (size == 0 || data == nullptr) {
        stats.min = stats.max = stats.mean = stats.std = stats.l2_norm = 0.0;
        return stats;
    }
    
    double sum = 0.0;
    double l2_sum = 0.0;
    stats.min = std::numeric_limits<double>::infinity();
    stats.max = -std::numeric_limits<double>::infinity();
    
    for (size_t i = 0; i < size; ++i) {
        const double val = static_cast<double>(data[i]);
        if (std::isnan(val)) {
            ++stats.nan_elements;
            continue;
        }
        if (std::isinf(val)) {
            if (val > 0.0) ++stats.pos_inf_elements;
            else ++stats.neg_inf_elements;
            continue;
        }
        ++stats.finite_elements;
        if (val == 0.0) ++stats.zero_elements;
        sum += val;
        l2_sum += val * val;
        if (val < stats.min) stats.min = val;
        if (val > stats.max) stats.max = val;
    }
    
    if (stats.finite_elements == 0) {
        stats.min = stats.max = stats.mean = stats.std = stats.l2_norm = 0.0;
        return stats;
    }
    stats.mean = sum / static_cast<double>(stats.finite_elements);
    stats.l2_norm = std::sqrt(l2_sum);
    
    double var_sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        const double val = static_cast<double>(data[i]);
        if (!std::isfinite(val)) continue;
        const double diff = val - stats.mean;
        var_sum += diff * diff;
    }
    stats.std = std::sqrt(var_sum / static_cast<double>(stats.finite_elements));
    
    return stats;
}

json DebugJsonDump::tensor_stats_to_json(const TensorStats& stats) const {
    json out;
    out["total_elements"] = stats.total_elements;
    out["finite_elements"] = stats.finite_elements;
    out["zero_elements"] = stats.zero_elements;
    out["nan_elements"] = stats.nan_elements;
    out["pos_inf_elements"] = stats.pos_inf_elements;
    out["neg_inf_elements"] = stats.neg_inf_elements;
    out["has_non_finite"] =
        stats.nan_elements + stats.pos_inf_elements + stats.neg_inf_elements > 0;
    if (stats.finite_elements > 0) {
        out["min"] = stats.min;
        out["max"] = stats.max;
        out["mean"] = stats.mean;
        out["std"] = stats.std;
        out["l2_norm"] = stats.l2_norm;
    } else {
        out["min"] = nullptr;
        out["max"] = nullptr;
        out["mean"] = nullptr;
        out["std"] = nullptr;
        out["l2_norm"] = nullptr;
    }
    return out;
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

    std::string tensor_dtype_tag = "F32";
    try {
        tensor_dtype_tag = dtype_to_string(string_to_dtype(model.getDefaultDType()));
    } catch (...) {
        tensor_dtype_tag = "F32";
    }
    
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
    std::string model_type = "";
    try {
        if (model.modelConfig.contains("type") && model.modelConfig["type"].is_string()) {
            model_type = model.modelConfig["type"].get<std::string>();
        }
    } catch (...) {
        model_type.clear();
    }
    if (!model_type.empty()) {
        model_info["type"] = model_type;
        root["model_type"] = model_type;
    }
    model_info["total_params"] = model.totalParamCount();
    model_info["num_layers"] = model.getLayers().size();
    
    root["model"] = model_info;
    
    // Layers
    json layers_array = json::array();
    for (const auto& layer : model.getLayers()) {
        json layer_obj;
        layer_obj["name"] = layer.name;
        layer_obj["type"] = layer.type;
        layer_obj["params_count"] = layer.params_count;
        layer_obj["trainable_parameter"] = layer.trainable_parameter;
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
        tensor_obj["dtype"] = tensor_dtype_tag;
        
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
            sample_array.push_back(json_number_or_null(data[j]));
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
    
    // ConditioningEncoder info (if present)
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

    auto set_if_positive = [&](const char* key, int value) {
        if (value > 0) config[key] = value;
    };
    auto set_vec_if_not_empty = [&](const char* key, const std::vector<int>& values) {
        if (!values.empty()) config[key] = values;
    };
    
    // Common fields
    if (layer.in_channels > 0) config["in_channels"] = layer.in_channels;
    if (layer.out_channels > 0) config["out_channels"] = layer.out_channels;
    set_if_positive("in_features", layer.in_features);
    set_if_positive("out_features", layer.out_features);
    set_if_positive("input_height", layer.input_height);
    set_if_positive("input_width", layer.input_width);
    set_if_positive("output_height", layer.output_height);
    set_if_positive("output_width", layer.output_width);
    
    // Type-specific configurations
    switch (layer.type_enum) {
        case LayerType::Conv2d:
        case LayerType::ConvTranspose2d:
        case LayerType::Conv1d:
        case LayerType::DepthwiseConv2d:
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
        case LayerType::Bilinear:
            config["in_features"] = layer.in_features;
            config["out_features"] = layer.out_features;
            config["has_bias"] = !layer.bias.empty();
            break;

        case LayerType::Embedding:
        case LayerType::EmbeddingBag:
            set_if_positive("vocab_size", layer.vocab_size);
            set_if_positive("embed_dim", layer.embed_dim);
            if (layer.padding_idx >= 0) config["padding_idx"] = layer.padding_idx;
            break;
            
        case LayerType::BatchNorm2d:
        case LayerType::BatchNorm1d:
        case LayerType::LayerNorm:
        case LayerType::InstanceNorm2d:
        case LayerType::GroupNorm:
        case LayerType::RMSNorm:
            if (layer.in_channels > 0) config["num_features"] = layer.in_channels;
            else if (layer.in_features > 0) config["num_features"] = layer.in_features;
            else if (layer.embed_dim > 0) config["num_features"] = layer.embed_dim;
            config["eps"] = layer.eps;
            if (layer.momentum > 0) config["momentum"] = layer.momentum;
            if (layer.num_groups > 0) config["num_groups"] = layer.num_groups;
            break;
            
        case LayerType::MaxPool2d:
        case LayerType::AvgPool2d:
        case LayerType::AdaptiveAvgPool2d:
        case LayerType::GlobalAvgPool2d:
        case LayerType::MaxPool1d:
        case LayerType::AvgPool1d:
            config["kernel_size"] = layer.kernel_size;
            config["stride"] = layer.stride;
            config["padding"] = layer.padding;
            break;
            
        case LayerType::SelfAttention:
        case LayerType::MultiHeadAttention:
        case LayerType::CrossAttention:
            config["num_heads"] = layer.num_heads;
            config["head_dim"] = layer.head_dim;
            if (layer.seq_len > 0) config["seq_len"] = layer.seq_len;
            if (layer.embed_dim > 0) config["embed_dim"] = layer.embed_dim;
            config["causal"] = layer.causal;
            config["use_mask"] = layer.use_mask;
            break;

        case LayerType::Softmax:
        case LayerType::LogSoftmax:
            config["axis"] = layer.axis;
            break;
            
        case LayerType::LeakyReLU:
            config["negative_slope"] = layer.negative_slope;
            break;
            
        case LayerType::Dropout:
        case LayerType::Dropout2d:
        case LayerType::AlphaDropout:
            config["dropout_p"] = layer.dropout_p;
            break;

        case LayerType::Reshape:
        case LayerType::View:
            set_vec_if_not_empty("target_shape", layer.target_shape);
            set_vec_if_not_empty("shape", layer.shape);
            break;

        case LayerType::Transpose:
        case LayerType::Permute:
            set_vec_if_not_empty("shape", layer.shape);
            set_vec_if_not_empty("permute_dims", layer.permute_dims);
            break;

        case LayerType::Squeeze:
            if (layer.squeeze_dim >= 0) config["squeeze_dim"] = layer.squeeze_dim;
            break;

        case LayerType::Unsqueeze:
            if (layer.unsqueeze_dim >= 0) config["unsqueeze_dim"] = layer.unsqueeze_dim;
            break;

        case LayerType::Concat:
            config["concat_axis"] = layer.concat_axis;
            break;

        case LayerType::Split:
            config["split_axis"] = layer.split_axis;
            config["num_splits"] = layer.num_splits;
            set_vec_if_not_empty("split_sizes", layer.split_sizes);
            break;

        case LayerType::Chunk:
            config["axis"] = layer.axis;
            config["num_chunks"] = layer.num_chunks;
            break;

        case LayerType::Stack:
            config["stack_axis"] = layer.stack_axis;
            break;

        case LayerType::MatMul:
        case LayerType::BatchMatMul:
            if (layer.embed_dim > 0) config["embed_dim"] = layer.embed_dim;
            break;

        case LayerType::UpsampleNearest:
        case LayerType::UpsampleBilinear:
        case LayerType::UpsampleBicubic:
            config["scale_h"] = layer.scale_h;
            config["scale_w"] = layer.scale_w;
            if (layer.out_h > 0) config["out_h"] = layer.out_h;
            if (layer.out_w > 0) config["out_w"] = layer.out_w;
            break;

        case LayerType::PatchEmbed:
            set_if_positive("patch_dim", layer.patch_dim);
            set_if_positive("num_patches", layer.num_patches);
            set_if_positive("seq_text", layer.seq_text);
            break;
            
        default:
            // For other layers, add generic info
            break;
    }
    
    if (layer.skip_input_index >= 0) config["skip_input_index"] = layer.skip_input_index;
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
        case LayerType::ConvTranspose2d:
            // [out_channels, in_channels, kernel_h, kernel_w]
            if (layer.out_channels > 0) shape.push_back(static_cast<size_t>(layer.out_channels));
            if (layer.in_channels > 0) {
                int groups = layer.groups > 0 ? layer.groups : 1;
                shape.push_back(static_cast<size_t>(layer.in_channels / groups));
            }
            shape.push_back(static_cast<size_t>(layer.get_kernel_h()));
            shape.push_back(static_cast<size_t>(layer.get_kernel_w()));
            break;

        case LayerType::Conv1d:
            if (layer.out_channels > 0) shape.push_back(static_cast<size_t>(layer.out_channels));
            if (layer.in_channels > 0) shape.push_back(static_cast<size_t>(layer.in_channels));
            shape.push_back(static_cast<size_t>(std::max(1, layer.kernel_size)));
            break;

        case LayerType::DepthwiseConv2d:
            if (layer.out_channels > 0) shape.push_back(static_cast<size_t>(layer.out_channels));
            shape.push_back(1);
            shape.push_back(static_cast<size_t>(layer.get_kernel_h()));
            shape.push_back(static_cast<size_t>(layer.get_kernel_w()));
            break;
            
        case LayerType::Linear:
        case LayerType::Bilinear:
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
        case LayerType::EmbeddingBag:
            // [vocab_size, embedding_dim]
            shape.push_back(layer.vocab_size);
            shape.push_back(layer.embed_dim);
            break;

        case LayerType::PatchEmbed:
            if (layer.out_features > 0) shape.push_back(static_cast<size_t>(layer.out_features));
            if (layer.patch_dim > 0) shape.push_back(static_cast<size_t>(layer.patch_dim));
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
    size_t element_count,
    const std::vector<size_t>& shape,
    const float* grad_data,
    size_t max_values,
    bool include_grads,
    const std::string& dtype_tag
) {
    json tensor_obj;
    tensor_obj["name"] = name;
    tensor_obj["dtype"] = dtype_tag;
    
    // Real shape (not just size)
    json shape_array = json::array();
    std::vector<size_t> safe_shape = shape;
    if (shape_element_count(safe_shape) != element_count) {
        safe_shape = {element_count};
        tensor_obj["shape_inferred"] = false;
    } else {
        tensor_obj["shape_inferred"] = true;
    }
    for (size_t dim : safe_shape) {
        shape_array.push_back(dim);
    }
    tensor_obj["shape"] = shape_array;
    tensor_obj["total_elements"] = element_count;
    
    if (data != nullptr && element_count > 0) {
        // Statistics
        TensorStats stats = calculate_stats(data, element_count);
        tensor_obj["stats"] = tensor_stats_to_json(stats);
        
        // Sample values (first N)
        size_t sample_size = std::min(element_count, max_values);
        json sample_array = json::array();
        for (size_t j = 0; j < sample_size; ++j) {
            sample_array.push_back(json_number_or_null(data[j]));
        }
        tensor_obj["sample_values"] = sample_array;
        tensor_obj["sample_size"] = sample_size;
        
        if (element_count > sample_size) {
            tensor_obj["truncated"] = true;
        }
        
        // Gradients (if requested and available)
        if (include_grads && grad_data != nullptr) {
            TensorStats grad_stats = calculate_stats(grad_data, element_count);
            json grad_stats_obj = tensor_stats_to_json(grad_stats);
            
            // Check if gradients are all zero
            bool all_zero = (grad_stats.l2_norm < 1e-9);
            grad_stats_obj["all_zero"] = all_zero;
            
            tensor_obj["gradients"] = grad_stats_obj;
            
            // Sample gradient values
            json grad_sample = json::array();
            for (size_t j = 0; j < sample_size; ++j) {
                grad_sample.push_back(json_number_or_null(grad_data[j]));
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
            sample.push_back(json_number_or_null(data[i]));
        }
        tensor_obj["sample"] = sample;
        
        if (size > sample_size) {
            tensor_obj["truncated"] = true;
        }
    }
    
    parent.push_back(tensor_obj);
}

    json DebugJsonDump::build_framework_state_snapshot(const Model& model) {
        json framework;

        framework["snapshot_version"] = "1.0.0";
        framework["snapshot_timestamp"] = std::time(nullptr);

        // Build/runtime section.
        json runtime;
        runtime["framework_logs_suppressed"] = Model::frameworkLogsSuppressed();

        json backends;
        {
        json cpu;
        cpu["compiled"] = true;
        cpu["available"] = model.hasCpuCompute();
        cpu["config"] = runtime_config_to_json(RuntimeConfig::fromEnv("CPU"));
        backends["cpu"] = cpu;
        }
        {
        json cuda;
    #ifdef ENABLE_CUDA
        cuda["compiled"] = true;
    #else
        cuda["compiled"] = false;
    #endif
        cuda["available"] = model.hasCudaCompute();
        cuda["config"] = runtime_config_to_json(RuntimeConfig::fromEnv("CUDA"));
        backends["cuda"] = cuda;
        }
        {
        json rocm;
    #ifdef ENABLE_ROCM
        rocm["compiled"] = true;
    #else
        rocm["compiled"] = false;
    #endif
        rocm["available"] = model.hasRocmCompute();
        rocm["config"] = runtime_config_to_json(RuntimeConfig::fromEnv("ROCM"));
        backends["rocm"] = rocm;
        }
        {
        json opencl;
    #ifdef ENABLE_OPENCL
        opencl["compiled"] = true;
    #else
        opencl["compiled"] = false;
    #endif
        opencl["available"] = model.hasOpenCLCompute();
        backends["opencl"] = opencl;
        }
        {
        json vulkan;
    #ifdef ENABLE_VULKAN
        vulkan["compiled"] = true;
    #else
        vulkan["compiled"] = false;
    #endif
        vulkan["available"] = model.hasVulkanCompute();
        backends["vulkan"] = vulkan;
        }
        runtime["backends"] = backends;

        json cpu_features;
        cpu_features["avx2"] = Model::hasAVX2();
        cpu_features["fma"] = Model::hasFMA();
        cpu_features["f16c"] = Model::hasF16C();
        cpu_features["bmi2"] = Model::hasBMI2();
        runtime["cpu_features"] = cpu_features;

        json env = json::object();
        set_env_if_present(env, "MIMIR_ACCEL_VERBOSE");
        set_env_if_present(env, "MIMIR_DISABLE_CPU");
        set_env_if_present(env, "MIMIR_DISABLE_CUDA");
        set_env_if_present(env, "MIMIR_DISABLE_ROCM");
        set_env_if_present(env, "MIMIR_DISABLE_OPENCL");
        set_env_if_present(env, "MIMIR_DISABLE_VULKAN");
        set_env_if_present(env, "MIMIR_CPU_LINEAR");
        set_env_if_present(env, "MIMIR_CPU_LINEAR_MIN_OPS");
        set_env_if_present(env, "MIMIR_CUDA_LINEAR");
        set_env_if_present(env, "MIMIR_CUDA_LINEAR_MIN_OPS");
        set_env_if_present(env, "MIMIR_CUDA_DEVICE");
        set_env_if_present(env, "MIMIR_ROCM_LINEAR");
        set_env_if_present(env, "MIMIR_ROCM_LINEAR_MIN_OPS");
        set_env_if_present(env, "MIMIR_ROCM_DEVICE");
        runtime["environment_overrides"] = env;

        framework["runtime"] = runtime;

        // Memory section.
        json memory;
        {
        auto& guard = MemoryGuard::instance();
        json mg;
        const size_t current_bytes = guard.getCurrentBytes();
        const size_t peak_bytes = guard.getPeakBytes();
        const size_t limit_bytes = guard.getLimit();
        mg["current_bytes"] = current_bytes;
        mg["peak_bytes"] = peak_bytes;
        mg["limit_bytes"] = limit_bytes;
        mg["usage_percent"] = guard.getUsagePercent();
        mg["current_mb"] = static_cast<double>(current_bytes) / (1024.0 * 1024.0);
        mg["peak_mb"] = static_cast<double>(peak_bytes) / (1024.0 * 1024.0);
        mg["limit_mb"] = static_cast<double>(limit_bytes) / (1024.0 * 1024.0);
        mg["allocations_blocked"] = guard.isBlocked();
        mg["freeze_mode"] = guard.isFrozen();
        memory["memory_guard"] = mg;
        }
        {
        auto& allocator = DynamicTensorAllocator::instance();
        json da;
        da["tensor_count"] = allocator.getTensorCount();
        da["loaded_count"] = allocator.getLoadedCount();
        memory["dynamic_tensor_allocator"] = da;
        }
        {
        auto& ram = AdvancedRAMManager::instance();
        json rm;
        const size_t current_bytes = ram.getCurrentRAM();
        const size_t peak_bytes = ram.getPeakRAM();
        rm["current_bytes"] = current_bytes;
        rm["peak_bytes"] = peak_bytes;
        rm["usage_percent"] = ram.getUsagePercent();
        rm["current_mb"] = static_cast<double>(current_bytes) / (1024.0 * 1024.0);
        rm["peak_mb"] = static_cast<double>(peak_bytes) / (1024.0 * 1024.0);
        rm["allocations_blocked"] = ram.isBlocked();
        rm["freeze_mode"] = ram.isFrozen();
        memory["advanced_ram_manager"] = rm;
        }
        framework["memory"] = memory;

        // Registry section.
        json registry;
        const auto supported_layers = LayerRegistry::get_all_supported_types();
        registry["supported_layer_types"] = supported_layers;
        registry["supported_layer_count"] = supported_layers.size();
        framework["registry"] = registry;

        // Model-local state.
        json model_state;
        model_state["model_name"] = model.getModelName();
        model_state["default_dtype"] = model.getDefaultDType();
        model_state["total_params"] = model.totalParamCount();
        model_state["num_layers"] = model.getLayers().size();
        model_state["has_encoder"] = model.getHasEncoder();
        model_state["tokenizer_vocab_size"] = model.getTokenizer().getVocabSize();
        model_state["parameters_frozen"] = model.parametersFrozen();
        model_state["model_config"] = model.modelConfig.is_object() ? model.modelConfig : json::object();
        framework["model"] = model_state;

        return framework;
    }

json DebugJsonDump::build_json_enhanced(const Model& model, const DebugJsonOptions& options) {
    json root;

    std::string tensor_dtype_tag = "F32";
    size_t serialized_element_bytes = sizeof(float);
    try {
        const auto serialized_dtype = string_to_dtype(model.getDefaultDType());
        tensor_dtype_tag = dtype_to_string(serialized_dtype);
        serialized_element_bytes = dtype_size(serialized_dtype);
    } catch (...) {
        tensor_dtype_tag = "F32";
        serialized_element_bytes = sizeof(float);
    }
    
    // Version and format
    const auto timestamp = std::time(nullptr);
    const std::string mimir_version = get_mimir_version();
    const std::string git_commit = options.include_git_info ? get_git_commit() : std::string();
    root["format"] = "mimir_debug_dump";
    root["format_version"] = "1.4.0";
    root["timestamp"] = timestamp;
    root["model_name"] = model.getModelName();

    json metadata;
    metadata["format"] = "mimir_debug_dump";
    metadata["format_version"] = "1.4.0";
    metadata["mimir_version"] = mimir_version;
    metadata["created_at_epoch_seconds"] = static_cast<long long>(timestamp);
    metadata["inspection_only"] = true;
    metadata["tensor_values_truncated"] = true;
    metadata["max_values_per_tensor"] = options.max_values_per_tensor;
    if (!git_commit.empty()) metadata["git_commit"] = git_commit;
    root["metadata"] = metadata;
    root["mimir_version"] = mimir_version;
    
    // Features supported in this version
    json features = json::array();
    features.push_back("layer_config");
    features.push_back("real_shapes");
    features.push_back("layer_io");
    features.push_back("extended_layer_types");
    features.push_back("framework_state");
    features.push_back("export_metrics");
    features.push_back("non_finite_metrics");
    if (options.include_gradients) features.push_back("gradients");
    if (options.include_weight_deltas) features.push_back("weight_deltas");
    if (options.include_optimizer_state) features.push_back("optimizer_state");
    if (options.include_checksums) features.push_back("checksums");
    root["features"] = features;
    
    // Model statistics. Keep legacy top-level fields and provide a scoped object.
    const size_t logical_params = model.totalParamCount();
    root["total_params"] = logical_params;
    root["num_layers"] = model.getLayers().size();

    // DType + config snapshot (kept in sync by Model::setDefaultDType)
    root["default_dtype"] = model.getDefaultDType();
    {
        json cfg = model.modelConfig;
        if (!cfg.is_object()) cfg = json::object();
        if (!cfg.contains("dtype")) {
            cfg["dtype"] = model.getDefaultDType();
        }
        root["model_config"] = cfg;
    }
    json model_info;
    model_info["name"] = model.getModelName();
    model_info["default_dtype"] = model.getDefaultDType();
    model_info["layer_count"] = model.getLayers().size();
    model_info["num_layers"] = model.getLayers().size();
    model_info["logical_parameter_elements"] = logical_params;
    model_info["total_params"] = logical_params;
    model_info["parameters_frozen"] = model.parametersFrozen();
    model_info["has_encoder"] = options.save_encoder;
    if (options.save_tokenizer) model_info["tokenizer_vocab_size"] = model.getTokenizer().getVocabSize();
    if (root["model_config"].contains("type") && root["model_config"]["type"].is_string()) {
        model_info["type"] = root["model_config"]["type"];
    }
    root["model"] = model_info;

    // Optional captured gradients: no longer supported (VAEModel removed)
    const std::unordered_map<std::string, std::vector<float>>* captured_grads = nullptr;
    
    // Optimizer state (if requested)
    if (options.include_optimizer_state) {
        json opt_state;
        if (const Optimizer* live = model.getSerializedOptimizer()) {
            const auto snapshot = model.optimizerSnapshot(*live);
            const Optimizer* opt = &snapshot;
            opt_state = optimizer_metadata(*opt);
            opt_state["type"] = optimizerTypeName(opt->type);

            // State vectors (debug-only): stats + small sample
            if (!opt->m.empty()) {
                TensorStats s = calculate_stats(opt->m.data(), opt->m.size());
                json mj;
                mj["size"] = opt->m.size();
                mj["stats"] = tensor_stats_to_json(s);
                const size_t sample_n = std::min<size_t>(options.max_values_per_tensor, opt->m.size());
                json sample = json::array();
                for (size_t i = 0; i < sample_n; ++i) sample.push_back(json_number_or_null(opt->m[i]));
                mj["sample_values"] = sample;
                mj["truncated"] = (opt->m.size() > sample_n);
                opt_state["m"] = mj;
            }
            if (!opt->v.empty()) {
                TensorStats s = calculate_stats(opt->v.data(), opt->v.size());
                json vj;
                vj["size"] = opt->v.size();
                vj["stats"] = tensor_stats_to_json(s);
                const size_t sample_n = std::min<size_t>(options.max_values_per_tensor, opt->v.size());
                json sample = json::array();
                for (size_t i = 0; i < sample_n; ++i) sample.push_back(json_number_or_null(opt->v[i]));
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
    size_t parameterized_layer_count = 0;
    size_t exported_tensor_count = 0;
    size_t exported_parameter_elements = 0;
    size_t unique_parameter_elements = 0;
    std::unordered_set<const float*> unique_parameter_buffers;
    
    for (size_t i = 0; i < layers.size(); ++i) {
        const Layer& layer = layers[i];
        
        json layer_obj;
        layer_obj["index"] = i;
        layer_obj["name"] = layer.name;
        layer_obj["type"] = layer.type;
        layer_obj["params_count"] = layer.params_count;
        layer_obj["trainable_parameter"] = layer.trainable_parameter;
        layer_obj["output"] = layer.output;
        if (!layer.inputs.empty()) {
            layer_obj["inputs"] = layer.inputs;
        }
        
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
            ++parameterized_layer_count;
            exported_parameter_elements += wb_size;
            if (unique_parameter_buffers.insert(wb).second) {
                unique_parameter_elements += wb_size;
            }

            // Split a trailing bias only when dimensions prove its exact layout.
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
            } else if (layer.type_enum == LayerType::Linear &&
                       layer.in_features > 0 && layer.out_features > 0) {
                const size_t linear_w =
                    static_cast<size_t>(layer.in_features) *
                    static_cast<size_t>(layer.out_features);
                const size_t linear_b = static_cast<size_t>(layer.out_features);
                if (linear_w <= wb_size) {
                    weight_count = linear_w;
                    if (layer.use_bias && wb_size == linear_w + linear_b) {
                        bias_count = linear_b;
                    } else if (wb_size != linear_w) {
                        // Unknown packed layout: describe the complete buffer safely.
                        weight_count = wb_size;
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
                    weight_count,
                    weight_shape.empty() ? std::vector<size_t>{weight_count} : weight_shape,
                    grad_ptr,
                    options.max_values_per_tensor,
                    options.include_gradients,
                    tensor_dtype_tag
                );
                ++exported_tensor_count;

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
                    bias_count,
                    std::vector<size_t>{bias_count},
                    grad_bias_ptr,
                    options.max_values_per_tensor,
                    options.include_gradients,
                    tensor_dtype_tag
                );
                ++exported_tensor_count;

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

    json export_metrics;
    export_metrics["layer_count"] = layers.size();
    export_metrics["parameterized_layer_count"] = parameterized_layer_count;
    export_metrics["exported_parameter_tensor_count"] = exported_tensor_count;
    export_metrics["logical_parameter_elements"] = logical_params;
    export_metrics["referenced_parameter_elements"] = exported_parameter_elements;
    export_metrics["unique_parameter_elements"] = unique_parameter_elements;
    export_metrics["shared_parameter_elements"] =
        exported_parameter_elements >= unique_parameter_elements
            ? exported_parameter_elements - unique_parameter_elements
            : 0;
    export_metrics["runtime_parameter_bytes"] = unique_parameter_elements * sizeof(float);
    export_metrics["serialized_parameter_bytes"] =
        unique_parameter_elements * serialized_element_bytes;
    export_metrics["serialized_dtype"] = tensor_dtype_tag;
    export_metrics["serialized_element_bytes"] = serialized_element_bytes;
    export_metrics["sampled_value_limit_per_tensor"] = options.max_values_per_tensor;
    export_metrics["counts_include_optimizer"] = false;
    export_metrics["counts_include_tokenizer"] = false;
    export_metrics["counts_include_encoder"] = false;
    root["export_metrics"] = export_metrics;
    
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
    
    // ConditioningEncoder section (ONLY if model actually has an encoder)
    if (options.save_encoder && model.getHasEncoder()) {
        const ConditioningEncoder& enc = model.getEncoder();
        json encoder_obj;
        encoder_obj["has_encoder"] = true;
        encoder_obj["architecture"] = "ViT";  // Or get from encoder
        root["encoder"] = encoder_obj;
    }
    
    // Git info (if requested)
    if (options.include_git_info) {
        json git;
        git["available"] = !git_commit.empty();
        if (!git_commit.empty()) git["commit"] = git_commit;
        root["git"] = git;
    }

    // Full framework snapshot at dump time.
    root["framework_state"] = build_framework_state_snapshot(model);
    
    return root;
}

bool DebugJsonDump::save_enhanced(
    const std::string& path,
    const Model& model,
    const DebugJsonOptions& requested_options,
    std::string* error
) {
    auto options = requested_options;
    options.save_encoder = options.save_encoder && component_enabled(model, "encoder", model.getHasEncoder());
    options.save_tokenizer = options.save_tokenizer && component_enabled(model, "tokenizer", model.getHasTokenizer());
    options.include_optimizer_state = options.include_optimizer_state && model.getSerializedOptimizer();

    try {
        json j = build_json_enhanced(model, options);

        fs::path file_path(path);
        if (file_path.has_parent_path()) {
            fs::create_directories(file_path.parent_path());
        }
        std::ofstream file(path);
        if (!file.is_open()) {
            const std::string message = "Failed to create DebugJson file: " + path;
            if (error) *error = message;
            std::cerr << "[DebugJsonDump] " << message << std::endl;
            return false;
        }
        
        // Write with pretty formatting (4 spaces indent)
        file << std::setw(4) << j << std::endl;
        if (!file.good()) {
            const std::string message = "Failed while writing DebugJson file: " + path;
            if (error) *error = message;
            return false;
        }
        file.close();
        
        std::cerr << "[DebugJsonDump] Saved enhanced debug JSON to: " << path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        if (error) *error = std::string("DebugJson export error: ") + e.what();
        std::cerr << "[DebugJsonDump] Error saving enhanced JSON: " << e.what() << std::endl;
        return false;
    }
}

} // namespace Serialization
} // namespace Mimir
