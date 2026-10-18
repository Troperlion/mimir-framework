#include "runtimes/fpga/FpgaIo.hpp"
#include "SafeTensorsWriter.hpp"
#include "../Model.hpp"
#include "CheckpointState.hpp"
#include "../Tokenizer.hpp"
#include "../Encoder.hpp"
#include "HardwareOpt.hpp"
#include <fstream>
#include <cstring>
#include <iomanip>
#include <ctime>
#include <iostream>

namespace Mimir {
namespace Serialization {

SafeTensorsWriter::SafeTensorsWriter() {
}

SafeTensorsWriter::~SafeTensorsWriter() {
}

bool SafeTensorsWriter::save(
    Model& model,
    const std::string& path,
    const SaveOptions& requested_options,
    std::string* error
) {
    const auto options = effective_save_options(model, requested_options);
    try {
        const uint16_t endian_probe = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian_probe) != 1) {
            if (error) *error = "SafeTensors serialization requires little-endian host support";
            return false;
        }
        // Collect all tensors
        std::vector<TensorData> tensors = collect_tensors(model, options);
        std::cerr << "[serialization] safetensors save path=" << path
                  << " tensors=" << tensors.size() << std::endl;
        
        if (tensors.empty()) {
            if (error) {
                *error = "No tensors to save";
            }
            return false;
        }
        
        // Build header
        json header = build_header(tensors, options);
        
        // Write file
        return write_file(path, header, tensors, error);
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("SafeTensors write error: ") + e.what();
        }
        return false;
    }
}

std::vector<SafeTensorsWriter::TensorData> SafeTensorsWriter::collect_tensors(
    Model& model,
    const SaveOptions& options
) {
    std::vector<TensorData> tensors;

    // Reset owned buffers for this save call
    owned_buffers_.clear();
    owned_f32_buffers_.clear();
    optimizer_metadata_ = json::object();
    owned_u16_buffers_.clear();
    owned_f64_buffers_.clear();

    // Determine preferred float storage dtype for serialization.
    // Runtime remains float32-first; we optionally downcast for disk.
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

    // Architecture / shapes / config (toujours, pour traçabilité et reconstruction)
    {
        json arch;
        arch["model_name"] = model.getModelName();
        arch["total_params"] = model.totalParamCount();
        arch["num_layers"] = layers.size();
        arch["model_config"] = model.modelConfig;
        try {
            if (model.modelConfig.contains("type") && model.modelConfig["type"].is_string()) {
                arch["model_type"] = model.modelConfig["type"].get<std::string>();
            }
        } catch (...) {
        }
        arch["model_config"]["dtype"] = dtype_to_string(float_storage);

        json layers_array = json::array();
        for (const auto& layer : layers) {
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
            // Weight size (if allocated)
            layer_obj["weights_size"] = layer.getWeightsSize();
            layers_array.push_back(layer_obj);
        }
        arch["layers"] = layers_array;

        std::string s = arch.dump();
        owned_buffers_.push_back(std::vector<uint8_t>(s.begin(), s.end()));
        TensorData td;
        td.name = "model/architecture_json";
        td.dtype = DType::Uint8;
        td.shape = {owned_buffers_.back().size()};
        td.byte_size = owned_buffers_.back().size();
        td.data_ptr = owned_buffers_.back().data();
        tensors.push_back(td);
    }
    
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
        
        push_float_tensor(layer.name + "/weights", {size}, data_ptr, size);
    }
    
    // Add encoder embeddings if present
    if (options.save_encoder && model.getHasEncoder()) {
        const auto& enc = model.getEncoder();
        if (!enc.token_embeddings.empty()) {
            push_float_tensor(
                "encoder/token_embeddings",
                {static_cast<size_t>(enc.vocab_size), static_cast<size_t>(enc.dim)},
                enc.token_embeddings.data(),
                enc.token_embeddings.size());
        }
    }

    // Add tokenizer JSON (as raw bytes)
    if (options.save_tokenizer) {
        const auto& tok = model.getTokenizer();
        std::string tok_str = tok.to_json().dump();
        owned_buffers_.push_back(std::vector<uint8_t>(tok_str.begin(), tok_str.end()));
        TensorData td;
        td.name = "tokenizer/json";
        td.dtype = DType::Uint8;
        td.shape = {owned_buffers_.back().size()};
        td.byte_size = owned_buffers_.back().size();
        td.data_ptr = owned_buffers_.back().data();
        tensors.push_back(td);
    }

    // Add encoder JSON (dim/vocab + special embeddings) if requested
    if (options.save_encoder && model.getHasEncoder()) {
        const auto& enc = model.getEncoder();
        std::string enc_str = enc.to_json().dump();
        owned_buffers_.push_back(std::vector<uint8_t>(enc_str.begin(), enc_str.end()));
        TensorData td;
        td.name = "encoder/json";
        td.dtype = DType::Uint8;
        td.shape = {owned_buffers_.back().size()};
        td.byte_size = owned_buffers_.back().size();
        td.data_ptr = owned_buffers_.back().data();
        tensors.push_back(td);
    }

    // Add optimizer state (if requested and available)
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
            const std::string metadata = optimizer_metadata_.dump();
            owned_buffers_.emplace_back(metadata.begin(), metadata.end());
            TensorData td;
            td.name = "optimizer/json";
            td.dtype = DType::Uint8;
            td.shape = {owned_buffers_.back().size()};
            td.byte_size = owned_buffers_.back().size();
            td.data_ptr = owned_buffers_.back().data();
            tensors.push_back(td);
        }
    }

    // Gradient snapshot tensors (debug-only).
    if (options.include_gradients) {
        // Generic: layer grad_weights snapshots
        for (const auto& layer : layers) {
            if (layer.grad_weights.empty()) continue;
            push_float_tensor("grads/" + layer.name + "/weights", {layer.grad_weights.size()}, layer.grad_weights.data(), layer.grad_weights.size());
        }
    }
    
    return tensors;
}

json SafeTensorsWriter::build_header(
    const std::vector<TensorData>& tensors,
    const SaveOptions& options
) {
    json header;
    
    // Calculate offsets
    size_t current_offset = 0;
    
    for (const auto& tensor : tensors) {
        json tensor_entry;
        tensor_entry["dtype"] = dtype_to_string(tensor.dtype);
        tensor_entry["shape"] = tensor.shape;
        
        // data_offsets: [begin, end) relative to start of data section
        json offsets = json::array();
        offsets.push_back(current_offset);
        offsets.push_back(current_offset + tensor.byte_size);
        tensor_entry["data_offsets"] = offsets;
        
        header[tensor.name] = tensor_entry;
        current_offset += tensor.byte_size;
    }
    
    // Add metadata
    json metadata;
    metadata["format"] = "safetensors";
    metadata["format_version"] = "0.4.0";
    metadata["components"] = checkpoint_components(options).dump();
    metadata["mimir_version"] = get_mimir_version();
    
    if (options.include_git_info) {
        std::string git_commit = get_git_commit();
        if (!git_commit.empty()) {
            metadata["git_commit"] = git_commit;
        }
    }
    
    // Timestamp
    auto now = std::time(nullptr);
    metadata["created_at"] = std::to_string(static_cast<long long>(now));
    
    // Total size
    metadata["total_size"] = std::to_string(current_offset);
    
    // Custom metadata
    if (!options.custom_metadata.empty()) {
        try {
            // SafeTensors requires __metadata__ to be string-to-string.
            // Keep structured user metadata as a canonical JSON string.
            metadata["custom"] = json::parse(options.custom_metadata).dump();
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Invalid custom_metadata JSON: ") + e.what());
        }
    }
    
    header["__metadata__"] = metadata;
    
    return header;
}

bool SafeTensorsWriter::write_file(
    const std::string& path,
    const json& header,
    const std::vector<TensorData>& tensors,
    std::string* error
) {
    try {
        // Ensure directory exists
        fs::path file_path(path);
        std::cerr << "[serialization] safetensors write_file path=" << file_path.string()
                  << " tensors=" << tensors.size() << std::endl;
        if (file_path.has_parent_path()) {
            fs::create_directories(file_path.parent_path());
        }
        
        // Open file for binary writing
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            if (error) {
                *error = "Failed to open file for writing: " + path;
            }
            return false;
        }
        
        // Serialize header to JSON string
        std::string header_str = header.dump();
        // Padding is part of N and must consist solely of ASCII spaces.
        // Aligning the data section to 8 bytes matches the reference writers.
        const size_t padding = (8 - (header_str.size() % 8)) % 8;
        header_str.append(padding, ' ');
        uint64_t header_len = static_cast<uint64_t>(header_str.size());
        
        // Write header length (8 bytes, little-endian)
        write_u64_le(file, header_len);
        
        // Write header JSON
        file.write(header_str.c_str(), header_len);
        if (!file) {
            if (error) {
                *error = "Failed to write header";
            }
            return false;
        }
        
        // Write tensor data in order
        for (const auto& tensor : tensors) {
            if (tensor.byte_size > 0 && tensor.data_ptr == nullptr) {
                if (error) {
                    *error = "Invalid tensor data for: " + tensor.name;
                }
                return false;
            }
            
            if (tensor.byte_size > 0) {
                // Numeric payloads are emitted little-endian on supported hosts.
                file.write(static_cast<const char*>(tensor.data_ptr), tensor.byte_size);
            }
            if (!file) {
                if (error) {
                    *error = "Failed to write tensor data: " + tensor.name;
                }
                return false;
            }
        }
        
        file.close();
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("File write exception: ") + e.what();
        }
        return false;
    }
}

void SafeTensorsWriter::write_u64_le(std::ofstream& f, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<uint8_t>((value >> (i * 8)) & 0xFF);
    }
    f.write(reinterpret_cast<const char*>(bytes), 8);
}

size_t SafeTensorsWriter::calculate_total_size(
    const std::vector<TensorData>& tensors
) const {
    size_t total = 0;
    for (const auto& tensor : tensors) {
        total += tensor.byte_size;
    }
    return total;
}

} // namespace Serialization
} // namespace Mimir
