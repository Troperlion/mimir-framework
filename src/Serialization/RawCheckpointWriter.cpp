#include "runtimes/fpga/FpgaIo.hpp"
#include "RawCheckpointWriter.hpp"
#include "../Model.hpp"
#include "CheckpointState.hpp"
#include "../Tokenizer.hpp"
#include "../Encoder.hpp"
#include "HardwareOpt.hpp"
#include "../Sha256.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <iostream>
#include <chrono>
#include <atomic>

namespace Mimir {
namespace Serialization {

namespace {
struct StagedRawFolder {
    fs::path target, temporary, backup;
    explicit StagedRawFolder(const fs::path& path) : target(path) {
        if (fs::exists(target) && (!fs::is_directory(target) ||
            (!fs::is_empty(target) && !fs::exists(target / "manifest.json")))) {
            throw std::runtime_error("Refusing to replace a non-checkpoint directory: " + target.string());
        }
        static std::atomic<unsigned long long> sequence{0};
        const auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(sequence++);
        temporary = target.string() + ".writing-" + suffix;
        backup = target.string() + ".previous-" + suffix;
    }
    ~StagedRawFolder() {
        std::error_code ignored;
        fs::remove_all(temporary, ignored);
    }
    void commit() {
        const bool existed = fs::exists(target);
        if (existed) fs::rename(target, backup);
        try {
            fs::rename(temporary, target);
        } catch (...) {
            if (existed) fs::rename(backup, target);
            throw;
        }
        std::error_code ignored;
        if (existed) fs::remove_all(backup, ignored);
    }
};
} // namespace

RawCheckpointWriter::RawCheckpointWriter() {
}

RawCheckpointWriter::~RawCheckpointWriter() {
}

bool RawCheckpointWriter::save(
    Model& model,
    const std::string& path,
    const SaveOptions& requested_options,
    std::string* error
) {
    const auto options = effective_save_options(model, requested_options);
    try {
        auto destination = fs::path(path).lexically_normal();
        if (destination.filename().empty()) destination = destination.parent_path();
        StagedRawFolder staged(destination);
        const fs::path root = staged.temporary;
        std::cerr << "[serialization] raw save path=" << root.string()
                  << " save_tokenizer=" << options.save_tokenizer
                  << " save_encoder=" << options.save_encoder
                  << " save_optimizer=" << options.save_optimizer << std::endl;
        
        // Create directory structure
        if (!create_structure(root.string(), error)) {
            return false;
        }
        
        // Collect tensors
        std::vector<TensorData> tensors = collect_tensors(model, options);
        
        // Save each tensor
        for (const auto& tensor : tensors) {
            if (!save_tensor(root.string(), tensor, error)) {
                return false;
            }
        }
        
        // Save architecture
        if (!save_architecture(root.string(), model, error)) {
            return false;
        }

        // Training state is independent of optional debug gradients.
        if (options.save_optimizer || model.modelConfig.contains("training_state")) {
            if (!save_training(root.string(), model, error)) {
                return false;
            }
        }
        
        // Save tokenizer if requested
        if (options.save_tokenizer) {
            if (!save_tokenizer(root.string(), model, error)) {
                return false;
            }
        }
        
        // Save encoder if requested
        if (options.save_encoder && model.getHasEncoder()) {
            if (!save_encoder(root.string(), model, error)) {
                return false;
            }
        }
        
        // Save manifest
        if (!save_manifest(root.string(), tensors, options, error)) {
            return false;
        }
        
        staged.commit();
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Raw checkpoint write error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointWriter::create_structure(
    const std::string& root,
    std::string* error
) {
    try {
        fs::path root_path(root);
        std::cerr << "[serialization] raw create_structure root=" << root_path.string() << std::endl;
        
        // Create directories
        fs::create_directories(root_path);
        fs::create_directories(root_path / "tensors");
        fs::create_directories(root_path / "model");
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Failed to create directory structure: ") + e.what();
        }
        return false;
    }
}

std::vector<RawCheckpointWriter::TensorData> RawCheckpointWriter::collect_tensors(
    Model& model,
    const SaveOptions& options
) {
    std::vector<TensorData> tensors;

    owned_f32_buffers_.clear();
    optimizer_metadata_ = json::object();
    owned_u16_buffers_.clear();
    owned_f64_buffers_.clear();

    DType float_storage = DType::Float32;
    try {
        float_storage = string_to_dtype(model.getDefaultDType());
    } catch (...) {
        float_storage = DType::Float32;
    }
    if (float_storage != DType::Float16 && float_storage != DType::BFloat16 &&
        float_storage != DType::Float32 && float_storage != DType::Float64) {
        float_storage = DType::Float32;
    }

    auto push_float_tensor = [&](const std::string& name,
                                 const std::vector<size_t>& shape,
                                 const float* data_ptr,
                                 size_t count) {
        if (!data_ptr || count == 0) return;

        TensorData td;
        td.name = name;
        td.shape = shape;

        if (float_storage == DType::Float16) {
            owned_u16_buffers_.emplace_back(count);
            if (!Mimir::fpgaEncode16(owned_u16_buffers_.back().data(), data_ptr, count, false)) HardwareOpt::fp32_to_fp16_f16c(owned_u16_buffers_.back().data(), data_ptr, count);
            td.dtype = DType::Float16;
            td.byte_size = count * sizeof(uint16_t);
            td.data_ptr = owned_u16_buffers_.back().data();
        } else if (float_storage == DType::BFloat16) {
            owned_u16_buffers_.emplace_back(count);
            if (!Mimir::fpgaEncode16(owned_u16_buffers_.back().data(), data_ptr, count, true)) HardwareOpt::fp32_to_bf16(owned_u16_buffers_.back().data(), data_ptr, count);
            td.dtype = DType::BFloat16;
            td.byte_size = count * sizeof(uint16_t);
            td.data_ptr = owned_u16_buffers_.back().data();
        } else if (float_storage == DType::Float64) {
            owned_f64_buffers_.emplace_back(count);
            auto& buf = owned_f64_buffers_.back();
            for (size_t i = 0; i < count; ++i) buf[i] = static_cast<double>(data_ptr[i]);
            td.dtype = DType::Float64;
            td.byte_size = count * sizeof(double);
            td.data_ptr = buf.data();
        } else {
            td.dtype = DType::Float32;
            td.byte_size = count * sizeof(float);
            td.data_ptr = data_ptr;
        }

        tensors.push_back(std::move(td));
    };
    
    const auto& layers = model.getLayers();
    
    // Collect layer weight blocks (modern allocation)
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto& layer = layers[layer_idx];
        
        if (!layer.weight_block) {
            continue;  // Layer has no weights
        }
        
        const float* data_ptr = layer.weight_block->getData();
        size_t size = layer.weight_block->getSize();
        
        if (data_ptr == nullptr || size == 0) {
            continue;  // Skip empty
        }
        
        push_float_tensor(layer.name + "_weights", {size}, data_ptr, size);
    }
    
    // Add encoder embeddings
    if (options.save_encoder && model.getHasEncoder()) {
        const auto& enc = model.getEncoder();
        if (!enc.token_embeddings.empty()) {
            push_float_tensor(
                "encoder_token_embeddings",
                {static_cast<size_t>(enc.vocab_size), static_cast<size_t>(enc.dim)},
                enc.token_embeddings.data(),
                enc.token_embeddings.size());
        }
    }

    // Optimizer state tensors (debug/resume): m and v
    if (options.save_optimizer) {
        if (const Optimizer* live = model.getSerializedOptimizer()) {
            auto snapshot = model.optimizerSnapshot(*live);
            optimizer_metadata_ = optimizer_metadata(snapshot);
            auto add_state = [&](const std::string& name, std::vector<float>& values) {
                if (values.empty()) return;
                owned_f32_buffers_.push_back(std::move(values));
                const auto& data = owned_f32_buffers_.back();
                TensorData td;
                td.name = name;
                td.dtype = DType::Float32; // Optimizer arithmetic is F32, even for F16/BF16 weights.
                td.shape = {data.size()};
                td.byte_size = data.size() * sizeof(float);
                td.data_ptr = data.data();
                tensors.push_back(td);
            };
            add_state("optimizer/m", snapshot.m);
            add_state("optimizer/v", snapshot.v);
        }
    }

    // Gradient snapshot tensors (debug only).
    if (options.include_gradients) {
        // Generic: layer grad_weights snapshots
        for (const auto& layer : layers) {
            if (layer.grad_weights.empty()) continue;
            push_float_tensor("grads/" + layer.name + "/weights", {layer.grad_weights.size()}, layer.grad_weights.data(), layer.grad_weights.size());
        }
    }
    
    return tensors;
}

bool RawCheckpointWriter::save_training(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path training_path = fs::path(root) / "model" / "training.json";

        json j = optimizer_metadata_;
        if (!j.contains("has_optimizer")) j["has_optimizer"] = false;
        if (model.modelConfig.contains("training_state")) j["loop"] = model.modelConfig["training_state"];

        std::ofstream file(training_path);
        if (!file) {
            if (error) {
                *error = "Failed to create training.json";
            }
            return false;
        }

        file << std::setw(2) << j;
        file.close();
        return true;
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Training save error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointWriter::save_tensor(
    const std::string& root,
    const TensorData& tensor,
    std::string* error
) {
    try {
        fs::path root_path(root);
        fs::path bin_path = root_path / "tensors" / (tensor.name + ".bin");
        fs::path json_path = root_path / "tensors" / (tensor.name + ".json");
        std::cerr << "[serialization] raw save_tensor name=" << tensor.name
                  << " dtype=" << dtype_to_string(tensor.dtype)
                  << " bytes=" << tensor.byte_size << std::endl;

        // Ensure directories exist (tensor.name may contain subpaths like "vae/enc_fc0_weights")
        fs::create_directories(bin_path.parent_path());
        fs::create_directories(json_path.parent_path());
        
        // Write binary data
        std::ofstream bin_file(bin_path, std::ios::binary);
        if (!bin_file) {
            if (error) {
                *error = "Failed to create tensor file: " + bin_path.string();
            }
            return false;
        }
        
        bin_file.write(static_cast<const char*>(tensor.data_ptr), tensor.byte_size);
        bin_file.close();
        
        // Calculate checksum
        std::string checksum = calculate_checksum(tensor.data_ptr, tensor.byte_size);
        
        // Write JSON metadata
        json tensor_json;
        tensor_json["name"] = tensor.name;
        tensor_json["dtype"] = dtype_to_string(tensor.dtype);
        tensor_json["shape"] = tensor.shape;
        tensor_json["byte_size"] = tensor.byte_size;
        tensor_json["checksum"] = checksum;
        tensor_json["checksum_algo"] = "sha256";
        tensor_json["data_file"] = tensor.name + ".bin";
        
        std::ofstream json_file(json_path);
        if (!json_file) {
            if (error) {
                *error = "Failed to create tensor JSON: " + json_path.string();
            }
            return false;
        }
        
        json_file << std::setw(2) << tensor_json;
        json_file.close();
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Tensor save error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointWriter::save_architecture(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path arch_path = fs::path(root) / "model" / "architecture.json";
        std::cerr << "[serialization] raw save_architecture path=" << arch_path.string() << std::endl;
        
        json arch;
        arch["model_name"] = model.getModelName();
        arch["total_params"] = model.totalParamCount();
        arch["num_layers"] = model.getLayers().size();
        arch["model_config"] = model.modelConfig;
        try {
            if (model.modelConfig.contains("type") && model.modelConfig["type"].is_string()) {
                arch["model_type"] = model.modelConfig["type"].get<std::string>();
            }
        } catch (...) {
        }
        {
            DType dt = DType::Float32;
            try { dt = string_to_dtype(model.getDefaultDType()); } catch (...) {}
            if (dt != DType::Float16 && dt != DType::BFloat16 && dt != DType::Float64)
                dt = DType::Float32;
            arch["model_config"]["dtype"] = dtype_to_string(dt);
        }
        
        // Save layer info
        json layers_array = json::array();
        for (const auto& layer : model.getLayers()) {
            json layer_obj;
            layer_obj["name"] = layer.name;
            layer_obj["type"] = layer.type;
            layer_obj["params_count"] = layer.params_count;
            layer_obj["trainable_parameter"] = layer.trainable_parameter;
            layer_obj["inputs"] = layer.inputs;
            if (layer.skip_input_index >= 0) layer_obj["skip_input_index"] = layer.skip_input_index;
            layer_obj["output"] = layer.output;
            // Common shape fields
            layer_obj["in_features"] = layer.in_features;
            layer_obj["out_features"] = layer.out_features;
            layer_obj["in_channels"] = layer.in_channels;
            layer_obj["out_channels"] = layer.out_channels;
            layer_obj["kernel_size"] = layer.kernel_size;
            layer_obj["stride"] = layer.stride;
            layer_obj["padding"] = layer.padding;
            layer_obj["seq_len"] = layer.seq_len;
            layer_obj["embed_dim"] = layer.embed_dim;
            layer_obj["num_heads"] = layer.num_heads;
            layer_obj["weights_size"] = layer.getWeightsSize();
            layers_array.push_back(layer_obj);
        }
        arch["layers"] = layers_array;
        
        std::ofstream file(arch_path);
        if (!file) {
            if (error) {
                *error = "Failed to create architecture.json";
            }
            return false;
        }
        
        file << std::setw(2) << arch;
        file.close();
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Architecture save error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointWriter::save_tokenizer(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path tok_path = fs::path(root) / "tokenizer" / "tokenizer.json";
        std::cerr << "[serialization] raw save_tokenizer path=" << tok_path.string() << std::endl;
        
        fs::create_directories(tok_path.parent_path());
        const auto& tokenizer = model.getTokenizer();
        json tok_json = tokenizer.to_json();
        
        std::ofstream file(tok_path);
        if (!file) {
            if (error) {
                *error = "Failed to create tokenizer.json";
            }
            return false;
        }
        
        file << std::setw(2) << tok_json;
        file.close();
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Tokenizer save error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointWriter::save_encoder(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path enc_path = fs::path(root) / "encoder" / "encoder.json";
        std::cerr << "[serialization] raw save_encoder path=" << enc_path.string() << std::endl;
        
        fs::create_directories(enc_path.parent_path());
        const auto& encoder = model.getEncoder();
        json enc_json = encoder.to_json();
        
        std::ofstream file(enc_path);
        if (!file) {
            if (error) {
                *error = "Failed to create encoder.json";
            }
            return false;
        }
        
        file << std::setw(2) << enc_json;
        file.close();
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("ConditioningEncoder save error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointWriter::save_manifest(
    const std::string& root,
    const std::vector<TensorData>& tensors,
    const SaveOptions& options,
    std::string* error
) {
    try {
        fs::path manifest_path = fs::path(root) / "manifest.json";
        std::cerr << "[serialization] raw save_manifest path=" << manifest_path.string()
                  << " tensors=" << tensors.size() << std::endl;
        
        json manifest;
        manifest["format"] = "mimir_raw_checkpoint";
        manifest["format_version"] = "1.0.0";
        manifest["mimir_version"] = get_mimir_version();
        
        if (options.include_git_info) {
            std::string git_commit = get_git_commit();
            if (!git_commit.empty()) {
                manifest["git_commit"] = git_commit;
            }
        }
        
        auto now = std::time(nullptr);
        manifest["created_at"] = static_cast<long long>(now);
        
        // List all components
        json components;
        components["tensors"] = tensors.size();
        components["model_architecture"] = true;
        components["tokenizer"] = options.save_tokenizer;
        components["encoder"] = options.save_encoder;
        components["optimizer"] = options.save_optimizer;
        manifest["components"] = components;
        
        // Tensor index
        json tensor_index = json::array();
        for (const auto& tensor : tensors) {
            json t_entry;
            t_entry["name"] = tensor.name;
            t_entry["bin_file"] = "tensors/" + tensor.name + ".bin";
            t_entry["json_file"] = "tensors/" + tensor.name + ".json";
            tensor_index.push_back(t_entry);
        }
        manifest["tensor_index"] = tensor_index;
        
        std::ofstream file(manifest_path);
        if (!file) {
            if (error) {
                *error = "Failed to create manifest.json";
            }
            return false;
        }
        
        file << std::setw(2) << manifest;
        file.close();
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Manifest save error: ") + e.what();
        }
        return false;
    }
}

std::string RawCheckpointWriter::calculate_checksum(
    const void* data,
    size_t size
) {
    // Use SHA256 for checksums
    std::string input(static_cast<const char*>(data), size);
    return sha256(input);
}

} // namespace Serialization
} // namespace Mimir
