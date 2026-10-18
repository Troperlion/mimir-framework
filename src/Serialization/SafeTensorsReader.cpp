#include "runtimes/fpga/FpgaIo.hpp"
#include "SafeTensorsReader.hpp"
#include "../Model.hpp"
#include "CheckpointState.hpp"
#include "../Encoder.hpp"
#include "HardwareOpt.hpp"
#include "../Sha256.hpp"
#include <fstream>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <unordered_set>

namespace Mimir {
namespace Serialization {

SafeTensorsReader::SafeTensorsReader() {
}

SafeTensorsReader::~SafeTensorsReader() {
}

bool SafeTensorsReader::load(
    Model& model,
    const std::string& path,
    const LoadOptions& requested_options,
    std::string* error
) {
    hash_tree_.begin(path);
    CheckpointHashTree::Scope hash_scope{hash_tree_};
    auto options = requested_options;
    try {
        const uint16_t endian_probe = 1;
        if (*reinterpret_cast<const uint8_t*>(&endian_probe) != 1) {
            if (error) *error = "SafeTensors loading requires little-endian host support";
            return false;
        }
        // Check if file exists
        if (!fs::exists(path)) {
            if (error) {
                *error = "File not found: " + path;
            }
            return false;
        }
        
        // Parse header
        json header;
        std::vector<ParsedTensor> tensors;
        size_t data_offset;
        
        if (!parse_header(path, header, tensors, data_offset, error)) {
            return false;
        }
        
        if (header.contains("__metadata__") && header["__metadata__"].contains("components")) {
            const auto components = json::parse(header["__metadata__"]["components"].get<std::string>());
            if (!components.value("encoder", true) && options.load_encoder) {
                options.load_encoder = false;
                model.setHasEncoder(false);
            }
            if (!components.value("tokenizer", true) && options.load_tokenizer) {
                options.load_tokenizer = false;
                model.setHasTokenizer(false);
            }
        }

        // Apply tensors to model
        return apply_tensors_to_model(model, tensors, path, data_offset, options, error);
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("SafeTensors load error: ") + e.what();
        }
        return false;
    }
}

bool SafeTensorsReader::parse_header(
    const std::string& path,
    json& header_out,
    std::vector<ParsedTensor>& tensors_out,
    size_t& data_offset_out,
    std::string* error
) {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            if (error) {
                *error = "Failed to open file: " + path;
            }
            return false;
        }
        
        // Read header length (8 bytes, little-endian)
        uint64_t header_len = read_u64_le(file);
        if (!file || header_len == 0 || header_len > 100 * 1024 * 1024 ||
            header_len > std::numeric_limits<size_t>::max()) {  // Max 100MB header
            if (error) {
                *error = "Invalid header length";
            }
            return false;
        }
        
        const uintmax_t file_size = fs::file_size(path);
        if (header_len > file_size - 8) {
            if (error) *error = "Header length exceeds file size";
            return false;
        }

        // Read header JSON
        std::vector<char> header_data(header_len);
        file.read(header_data.data(), header_len);
        if (!file) {
            if (error) {
                *error = "Failed to read header";
            }
            return false;
        }
        
        if (header_data.empty() || header_data.front() != '{') {
            if (error) *error = "SafeTensors header must begin with '{'";
            return false;
        }

        // Parse JSON and reject duplicate keys at every object depth.
        std::string header_str(header_data.begin(), header_data.end());
        bool duplicate_key = false;
        std::unordered_map<int, std::unordered_set<std::string>> keys_by_depth;
        json::parser_callback_t callback =
            [&](int depth, json::parse_event_t event, json& parsed) {
                if (event == json::parse_event_t::object_start) {
                    keys_by_depth[depth].clear();
                } else if (event == json::parse_event_t::key) {
                    auto& keys = keys_by_depth[depth - 1];
                    if (!keys.insert(parsed.get<std::string>()).second) duplicate_key = true;
                } else if (event == json::parse_event_t::object_end) {
                    keys_by_depth.erase(depth);
                }
                return true;
            };
        header_out = json::parse(header_str, callback);
        if (duplicate_key) {
            if (error) *error = "Duplicate key in SafeTensors header";
            return false;
        }
        if (!header_out.is_object()) {
            if (error) *error = "SafeTensors header root must be an object";
            return false;
        }

        if (header_out.contains("__metadata__")) {
            const auto& metadata = header_out["__metadata__"];
            if (!metadata.is_object()) {
                if (error) *error = "__metadata__ must be an object";
                return false;
            }
            for (auto it = metadata.begin(); it != metadata.end(); ++it) {
                if (!it.value().is_string()) {
                    if (error) *error = "__metadata__ values must all be strings";
                    return false;
                }
            }
        }
        
        // Data starts after 8-byte length + header
        data_offset_out = 8 + header_len;
        
        auto checked_element_count = [](const std::vector<size_t>& shape, size_t& count) {
            count = 1; // [] is a scalar
            for (const size_t dim : shape) {
                if (dim != 0 && count > std::numeric_limits<size_t>::max() / dim) return false;
                count *= dim;
            }
            return true;
        };
        auto is_official_supported_dtype = [](const std::string& dtype) {
            return dtype == "F64" || dtype == "F32" || dtype == "F16" ||
                   dtype == "BF16" || dtype == "I64" || dtype == "I32" ||
                   dtype == "I16" || dtype == "I8" || dtype == "U64" ||
                   dtype == "U32" || dtype == "U16" || dtype == "U8" ||
                   dtype == "BOOL";
        };

        // Extract tensor information
        for (auto it = header_out.begin(); it != header_out.end(); ++it) {
            if (it.key() == "__metadata__") {
                continue;  // Skip metadata
            }
            
            const json& tensor_entry = it.value();
            if (!tensor_entry.is_object()) {
                if (error) *error = "Tensor entry must be an object: " + it.key();
                return false;
            }
            
            ParsedTensor tensor;
            tensor.name = it.key();
            
            // Parse dtype
            if (!tensor_entry.contains("dtype")) {
                if (error) {
                    *error = "Missing dtype for tensor: " + tensor.name;
                }
                return false;
            }
            if (!tensor_entry["dtype"].is_string()) {
                if (error) *error = "Invalid dtype for tensor: " + tensor.name;
                return false;
            }
            std::string dtype_str = tensor_entry["dtype"].get<std::string>();
            if (!is_official_supported_dtype(dtype_str)) {
                if (error) *error = "Unsupported or non-canonical SafeTensors dtype: " + dtype_str;
                return false;
            }
            tensor.dtype = string_to_dtype(dtype_str);
            
            // Parse shape
            if (!tensor_entry.contains("shape")) {
                if (error) {
                    *error = "Missing shape for tensor: " + tensor.name;
                }
                return false;
            }
            if (!tensor_entry["shape"].is_array()) {
                if (error) *error = "Invalid shape for tensor: " + tensor.name;
                return false;
            }
            tensor.shape = tensor_entry["shape"].get<std::vector<size_t>>();
            
            // Parse data_offsets
            if (!tensor_entry.contains("data_offsets")) {
                if (error) {
                    *error = "Missing data_offsets for tensor: " + tensor.name;
                }
                return false;
            }
            auto offsets = tensor_entry["data_offsets"];
            if (!offsets.is_array() || offsets.size() != 2) {
                if (error) {
                    *error = "Invalid data_offsets for tensor: " + tensor.name;
                }
                return false;
            }
            if (!offsets[0].is_number_unsigned() || !offsets[1].is_number_unsigned()) {
                if (error) *error = "Non-unsigned data_offsets for tensor: " + tensor.name;
                return false;
            }
            tensor.data_begin = offsets[0].get<size_t>();
            tensor.data_end = offsets[1].get<size_t>();
            if (tensor.data_end < tensor.data_begin) {
                if (error) *error = "Reversed data_offsets for tensor: " + tensor.name;
                return false;
            }
            
            // Validate size
            size_t element_count = 0;
            if (!checked_element_count(tensor.shape, element_count) ||
                (dtype_size(tensor.dtype) != 0 &&
                 element_count > std::numeric_limits<size_t>::max() / dtype_size(tensor.dtype))) {
                if (error) *error = "Tensor shape size overflow: " + tensor.name;
                return false;
            }
            const size_t expected_size = element_count * dtype_size(tensor.dtype);
            
            if (tensor.data_end - tensor.data_begin != expected_size) {
                if (error) {
                    *error = "Size mismatch for tensor: " + tensor.name;
                }
                return false;
            }
            
            tensors_out.push_back(tensor);
        }

        std::sort(tensors_out.begin(), tensors_out.end(),
                  [](const ParsedTensor& a, const ParsedTensor& b) {
                      if (a.data_begin != b.data_begin) return a.data_begin < b.data_begin;
                      return a.data_end < b.data_end;
                  });
        size_t expected_begin = 0;
        for (const auto& tensor : tensors_out) {
            if (tensor.data_begin != expected_begin) {
                if (error) *error = tensor.data_begin < expected_begin
                    ? "Overlapping tensor data offsets"
                    : "Hole in tensor data offsets";
                return false;
            }
            expected_begin = tensor.data_end;
        }
        const uintmax_t actual_data_size = file_size - data_offset_out;
        if (expected_begin != actual_data_size) {
            if (error) *error = expected_begin < actual_data_size
                ? "Unindexed trailing bytes in SafeTensors buffer"
                : "Tensor data offsets exceed file size";
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Header parse error: ") + e.what();
        }
        return false;
    }
}

bool SafeTensorsReader::load_tensor_data(
    const std::string& path,
    size_t data_offset,
    const ParsedTensor& tensor,
    void* dest_buffer,
    size_t dest_size,
    std::string* error
) {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            if (error) {
                *error = "Failed to open file: " + path;
            }
            return false;
        }
        
        // Seek to tensor data
        size_t absolute_offset = data_offset + tensor.data_begin;
        file.seekg(absolute_offset);
        if (!file) {
            if (error) {
                *error = "Failed to seek to tensor data: " + tensor.name;
            }
            return false;
        }
        
        // Read tensor data
        size_t bytes_to_read = tensor.data_end - tensor.data_begin;
        if (bytes_to_read > dest_size) {
            if (error) {
                *error = "Buffer too small for tensor: " + tensor.name;
            }
            return false;
        }
        
        file.read(static_cast<char*>(dest_buffer), bytes_to_read);
        if (!file) {
            if (error) {
                *error = "Failed to read tensor data: " + tensor.name;
            }
            return false;
        }

        hash_tree_.add(tensor.name, "sha256", sha256(dest_buffer, bytes_to_read));
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Tensor load error: ") + e.what();
        }
        return false;
    }
}

bool SafeTensorsReader::apply_tensors_to_model(
    Model& model,
    const std::vector<ParsedTensor>& tensors,
    const std::string& path,
    size_t data_offset,
    const LoadOptions& options,
    std::string* error
) {
    try {
        const auto& layers = model.getLayers();
        
        // Create a map of tensor names for quick lookup
        std::unordered_map<std::string, const ParsedTensor*> tensor_map;
        for (const auto& tensor : tensors) {
            tensor_map[tensor.name] = &tensor;
        }

        json mapping_root = json::object();
        json mapped_layers = json::object();
        if (!options.mapping_json.empty()) {
            std::ifstream mf(options.mapping_json);
            if (!mf) {
                if (error) *error = "Failed to open mapping JSON: " + options.mapping_json;
                return false;
            }
            try {
                mf >> mapping_root;
                if (mapping_root.is_object() && mapping_root.contains("layers") && mapping_root["layers"].is_object()) {
                    mapped_layers = mapping_root["layers"];
                } else if (mapping_root.is_object()) {
                    mapped_layers = mapping_root;
                } else {
                    if (error) *error = "Invalid mapping JSON root: " + options.mapping_json;
                    return false;
                }
            } catch (const std::exception& e) {
                if (error) *error = std::string("Failed to parse mapping JSON: ") + e.what();
                return false;
            }
        }

        auto load_tensor_as_fp32 = [&](const std::string& source_name, std::vector<float>& out) -> bool {
            auto it = tensor_map.find(source_name);
            if (it == tensor_map.end()) {
                if (error) *error = "Missing mapped source tensor: " + source_name;
                return false;
            }
            const ParsedTensor& tensor = *it->second;
            size_t element_count = 1;
            for (size_t dim : tensor.shape) element_count *= dim;
            out.resize(element_count);

            if (tensor.dtype == DType::Float32) {
                return load_tensor_data(path, data_offset, tensor, out.data(), out.size() * sizeof(float), error);
            }
            if (tensor.dtype == DType::Float16) {
                std::vector<uint16_t> tmp(element_count);
                if (!load_tensor_data(path, data_offset, tensor, tmp.data(), tmp.size() * sizeof(uint16_t), error)) return false;
                if (!Mimir::fpgaDecode16(out.data(), tmp.data(), element_count, false)) HardwareOpt::fp16_to_fp32_f16c(out.data(), tmp.data(), element_count);
                return true;
            }
            if (tensor.dtype == DType::BFloat16) {
                std::vector<uint16_t> tmp(element_count);
                if (!load_tensor_data(path, data_offset, tensor, tmp.data(), tmp.size() * sizeof(uint16_t), error)) return false;
                if (!Mimir::fpgaDecode16(out.data(), tmp.data(), element_count, true)) HardwareOpt::bf16_to_fp32(out.data(), tmp.data(), element_count);
                return true;
            }
            if (tensor.dtype == DType::Float64) {
                std::vector<double> tmp(element_count);
                if (!load_tensor_data(path, data_offset, tensor, tmp.data(), tmp.size() * sizeof(double), error)) return false;
                for (size_t i = 0; i < element_count; ++i) out[i] = static_cast<float>(tmp[i]);
                return true;
            }

            if (error) *error = "Unsupported dtype for mapped tensor: " + source_name;
            return false;
        };

        auto apply_mapped_layer = [&](Layer& layer, const json& spec) -> bool {
            if (!spec.is_object()) {
                if (error) *error = "Invalid mapping spec for layer: " + layer.name;
                return false;
            }

            std::vector<float> packed;
            auto append_tensor = [&](const char* key) -> bool {
                auto it = spec.find(key);
                if (it == spec.end()) return true;
                if (!it->is_string()) {
                    if (error) *error = std::string("Mapping key '") + key + "' must be a string for layer: " + layer.name;
                    return false;
                }
                std::vector<float> tmp;
                if (!load_tensor_as_fp32(it->get<std::string>(), tmp)) return false;
                packed.insert(packed.end(), tmp.begin(), tmp.end());
                return true;
            };

            if (spec.contains("tensor")) {
                if (!append_tensor("tensor")) return false;
            } else if (spec.contains("q_weight") || spec.contains("k_weight") || spec.contains("v_weight") ||
                       spec.contains("q_bias") || spec.contains("k_bias") || spec.contains("v_bias")) {
                if (!append_tensor("q_weight")) return false;
                if (!append_tensor("k_weight")) return false;
                if (!append_tensor("v_weight")) return false;
                if (!append_tensor("out_weight")) return false;
                if (!append_tensor("q_bias")) return false;
                if (!append_tensor("k_bias")) return false;
                if (!append_tensor("v_bias")) return false;
                if (!append_tensor("out_bias")) return false;
            } else if (spec.contains("qkv_weight") || spec.contains("out_weight")) {
                if (!append_tensor("qkv_weight")) return false;
                if (!append_tensor("out_weight")) return false;
                if (!append_tensor("qkv_bias")) return false;
                if (!append_tensor("out_bias")) return false;
            } else {
                if (!append_tensor("weight")) return false;
                if (!append_tensor("bias")) return false;
            }

            const size_t actual_size = layer.weight_block->getSize();
            if (packed.size() != actual_size) {
                if (error) {
                    *error = "Mapped size mismatch for " + layer.name + ": expected " + std::to_string(actual_size) + " got " + std::to_string(packed.size());
                }
                return false;
            }

            float* data_ptr = layer.weight_block->getData();
            if (!data_ptr) {
                if (error) *error = "Failed to get data pointer for mapped layer: " + layer.name;
                return false;
            }
            std::copy(packed.begin(), packed.end(), data_ptr);
            return true;
        };

        // Load architecture/config JSON (if present)
        {
            auto it = tensor_map.find("model/architecture_json");
            if (it != tensor_map.end()) {
                const ParsedTensor* tensor = it->second;
                if (tensor->dtype == DType::Uint8) {
                    const size_t n = tensor->data_end - tensor->data_begin;
                    std::vector<uint8_t> buf(n);
                    if (!load_tensor_data(path, data_offset, *tensor, buf.data(), buf.size(), error)) {
                        return false;
                    }
                    try {
                        std::string s(reinterpret_cast<const char*>(buf.data()), buf.size());
                        json arch = json::parse(s);
                        if (!options.metadata_only && arch.contains("layers")) {
                            for (const auto& entry : arch["layers"]) {
                                if (entry.contains("skip_input_index")) {
                                    if (auto* layer = model.getLayerByName(entry.value("name", "")))
                                        layer->skip_input_index = entry["skip_input_index"].get<int>();
                                }
                            }
                        }
                        if (options.apply_model_name && arch.contains("model_name")) {
                            model.setModelName(arch["model_name"].get<std::string>());
                        }
                        if (options.apply_model_config && arch.contains("model_config")) {
                            model.modelConfig = arch["model_config"];

                            // Keep runtime default dtype consistent with the loaded config.
                            // This ensures subsequent saves use the same float storage policy.
                            try {
                                if (model.modelConfig.contains("dtype") && model.modelConfig["dtype"].is_string()) {
                                    const std::string raw_dtype = model.modelConfig["dtype"].get<std::string>();
                                    const auto dt = ::Mimir::parse_dtype_safetensors(raw_dtype);
                                    if (dt == ::Mimir::DType::UNKNOWN) {
                                        throw std::runtime_error("unknown dtype in model_config: " + raw_dtype);
                                    }
                                    const std::string canonical_dtype = ::Mimir::dtype_to_string(dt);
                                    model.setDefaultDType(canonical_dtype);
                                }
                            } catch (...) {
                                if (options.strict_mode) {
                                    if (error) {
                                        *error = "Invalid dtype in model_config";
                                    }
                                    return false;
                                }
                            }
                        }
                    } catch (...) {
                        if (options.strict_mode) {
                            if (error) {
                                *error = "Invalid model/architecture_json";
                            }
                            return false;
                        }
                        // Backward compatibility: ignore invalid JSON in non-strict mode
                    }
                } else if (options.strict_mode) {
                    if (error) {
                        *error = "Invalid dtype for model/architecture_json";
                    }
                    return false;
                }
            }
        }

        if (options.metadata_only) {
            return true;
        }
        
        // Load layer weight blocks
        for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
            auto& layer = const_cast<Layer&>(layers[layer_idx]);
            
            if (!layer.weight_block) {
                continue;  // Layer has no weights
            }

            auto map_it = mapped_layers.find(layer.name);
            if (map_it != mapped_layers.end()) {
                if (!apply_mapped_layer(layer, map_it.value())) {
                    return false;
                }
                continue;
            }
            
            // Standard Mimir checkpoints use "<layer>/weights".
            // External/base models may mirror the original safetensors keys exactly.
            std::string tensor_name = layer.name + "/weights";

            auto it = tensor_map.find(tensor_name);
            if (it == tensor_map.end()) {
                it = tensor_map.find(layer.name);
                if (it != tensor_map.end()) {
                    tensor_name = layer.name;
                }
            }
            if (it != tensor_map.end()) {
                const ParsedTensor* tensor = it->second;
                
                // Get expected size
                size_t expected_size = 1;
                for (size_t dim : tensor->shape) {
                    expected_size *= dim;
                }
                
                // Verify size matches
                size_t actual_size = layer.weight_block->getSize();
                if (actual_size != expected_size) {
                    if (error) {
                        *error = "Size mismatch for " + tensor_name + ": expected " + 
                                 std::to_string(expected_size) + " got " + std::to_string(actual_size);
                    }
                    return false;
                }
                
                // Get data pointer
                float* data_ptr = layer.weight_block->getData();
                if (!data_ptr) {
                    if (error) {
                        *error = "Failed to get data pointer for: " + tensor_name;
                    }
                    return false;
                }

                // Load data (support F16/BF16/F64 -> convert to f32)
                if (tensor->dtype == DType::Float32) {
                    if (!load_tensor_data(
                        path, data_offset, *tensor,
                        data_ptr, actual_size * sizeof(float),
                        error
                    )) {
                        return false;
                    }
                } else if (tensor->dtype == DType::Float16) {
                    std::vector<uint16_t> tmp(actual_size);
                    if (!load_tensor_data(
                        path, data_offset, *tensor,
                        tmp.data(), tmp.size() * sizeof(uint16_t),
                        error
                    )) {
                        return false;
                    }
                    if (!Mimir::fpgaDecode16(data_ptr, tmp.data(), actual_size, false)) HardwareOpt::fp16_to_fp32_f16c(data_ptr, tmp.data(), actual_size);
                } else if (tensor->dtype == DType::BFloat16) {
                    std::vector<uint16_t> tmp(actual_size);
                    if (!load_tensor_data(
                        path, data_offset, *tensor,
                        tmp.data(), tmp.size() * sizeof(uint16_t),
                        error
                    )) {
                        return false;
                    }
                    if (!Mimir::fpgaDecode16(data_ptr, tmp.data(), actual_size, true)) HardwareOpt::bf16_to_fp32(data_ptr, tmp.data(), actual_size);
                } else if (tensor->dtype == DType::Float64) {
                    std::vector<double> tmp(actual_size);
                    if (!load_tensor_data(
                        path, data_offset, *tensor,
                        tmp.data(), tmp.size() * sizeof(double),
                        error
                    )) {
                        return false;
                    }
                    for (size_t i = 0; i < actual_size; ++i) data_ptr[i] = static_cast<float>(tmp[i]);
                } else if (options.strict_mode) {
                    if (error) {
                        *error = "Unsupported dtype for tensor: " + tensor_name;
                    }
                    return false;
                }
            } else if (options.strict_mode) {
                if (error) {
                    *error = "Missing tensor in strict mode: " + tensor_name;
                }
                return false;
            }
        }

        // Load tokenizer JSON (as raw bytes)
        if (options.load_tokenizer) {
            auto it = tensor_map.find("tokenizer/json");
            if (it != tensor_map.end()) {
                const ParsedTensor* tensor = it->second;
                if (tensor->dtype == DType::Uint8) {
                    const size_t n = tensor->data_end - tensor->data_begin;
                    std::vector<uint8_t> buf(n);
                    if (!load_tensor_data(path, data_offset, *tensor, buf.data(), buf.size(), error)) {
                        return false;
                    }
                    try {
                        std::string s(reinterpret_cast<const char*>(buf.data()), buf.size());
                        json j = json::parse(s);
                        model.getMutableTokenizer().from_json(j);
                        model.setHasTokenizer(true);
                    } catch (...) {
                        // Ignore invalid tokenizer JSON for backward compatibility
                    }
                }
            } else if (options.strict_mode) {
                if (error) {
                    *error = "Missing tensor in strict mode: tokenizer/json";
                }
                return false;
            }
        }

        // Load encoder JSON (preferred for full encoder state)
        if (options.load_encoder) {
            auto it = tensor_map.find("encoder/json");
            if (it != tensor_map.end()) {
                const ParsedTensor* tensor = it->second;
                if (tensor->dtype == DType::Uint8) {
                    const size_t n = tensor->data_end - tensor->data_begin;
                    std::vector<uint8_t> buf(n);
                    if (!load_tensor_data(path, data_offset, *tensor, buf.data(), buf.size(), error)) {
                        return false;
                    }
                    try {
                        std::string s(reinterpret_cast<const char*>(buf.data()), buf.size());
                        json j = json::parse(s);
                        model.getMutableEncoder().from_json(j);
                        model.setHasEncoder(true);
                    } catch (...) {
                        // Ignore invalid encoder JSON and fall back to embeddings tensor
                    }
                }
            } else if (options.strict_mode) {
                if (error) {
                    *error = "Missing tensor in strict mode: encoder/json";
                }
                return false;
            }
        }
        
        // Load encoder embeddings
        if (options.load_encoder) {
            auto it = tensor_map.find("encoder/token_embeddings");
            if (it != tensor_map.end()) {
                const ParsedTensor* tensor = it->second;
                
                auto& enc = model.getMutableEncoder();
                if (tensor->shape.size() == 2) {
                    enc.vocab_size = static_cast<int>(tensor->shape[0]);
                    enc.dim = static_cast<int>(tensor->shape[1]);
                }
                size_t expected_size = 1;
                for (size_t dim : tensor->shape) {
                    expected_size *= dim;
                }
                
                if (enc.token_embeddings.size() < expected_size) {
                    enc.token_embeddings.resize(expected_size);
                }

                if (tensor->dtype == DType::Float32) {
                    if (!load_tensor_data(
                        path, data_offset, *tensor,
                        enc.token_embeddings.data(),
                        enc.token_embeddings.size() * sizeof(float),
                        error
                    )) {
                        return false;
                    }
                } else if (tensor->dtype == DType::Float16) {
                    std::vector<uint16_t> tmp(expected_size);
                    if (!load_tensor_data(
                        path, data_offset, *tensor,
                        tmp.data(), tmp.size() * sizeof(uint16_t),
                        error
                    )) {
                        return false;
                    }
                    if (!Mimir::fpgaDecode16(enc.token_embeddings.data(), tmp.data(), expected_size, false)) HardwareOpt::fp16_to_fp32_f16c(enc.token_embeddings.data(), tmp.data(), expected_size);
                } else if (tensor->dtype == DType::BFloat16) {
                    std::vector<uint16_t> tmp(expected_size);
                    if (!load_tensor_data(
                        path, data_offset, *tensor,
                        tmp.data(), tmp.size() * sizeof(uint16_t),
                        error
                    )) {
                        return false;
                    }
                    if (!Mimir::fpgaDecode16(enc.token_embeddings.data(), tmp.data(), expected_size, true)) HardwareOpt::bf16_to_fp32(enc.token_embeddings.data(), tmp.data(), expected_size);
                } else if (tensor->dtype == DType::Float64) {
                    std::vector<double> tmp(expected_size);
                    if (!load_tensor_data(
                        path, data_offset, *tensor,
                        tmp.data(), tmp.size() * sizeof(double),
                        error
                    )) {
                        return false;
                    }
                    for (size_t i = 0; i < expected_size; ++i) enc.token_embeddings[i] = static_cast<float>(tmp[i]);
                } else if (options.strict_mode) {
                    if (error) {
                        *error = "Unsupported dtype for encoder/token_embeddings";
                    }
                    return false;
                }

                model.setHasEncoder(true);
            }
        }

        // Load optimizer (json + state vectors) if requested
        if (options.load_optimizer) {
            json optimizer_meta = json::object();
            // Parse optimizer/json
            auto itj = tensor_map.find("optimizer/json");
            if (itj != tensor_map.end()) {
                const ParsedTensor* tensor = itj->second;
                if (tensor->dtype == DType::Uint8) {
                    const size_t n = tensor->data_end - tensor->data_begin;
                    std::vector<uint8_t> buf(n);
                    if (!load_tensor_data(path, data_offset, *tensor, buf.data(), buf.size(), error)) {
                        return false;
                    }
                    try {
                        std::string s(reinterpret_cast<const char*>(buf.data()), buf.size());
                        json j = json::parse(s);
                        Optimizer opt = optimizer_from_metadata(j);
                        optimizer_meta = j;
                        model.setSerializedOptimizer(opt);
                    } catch (...) {
                        // Invalid optimizer JSON
                        if (options.strict_mode) {
                            if (error) {
                                *error = "Invalid optimizer/json";
                            }
                            return false;
                        }
                    }
                }
            } else if (options.strict_mode) {
                if (error) {
                    *error = "Missing tensor in strict mode: optimizer/json";
                }
                return false;
            }

            // Load optimizer state vectors (optional; if present without json, create default container)
            auto load_opt_vec = [&](const std::string& name, std::vector<float>& dst) -> bool {
                auto it = tensor_map.find(name);
                if (it == tensor_map.end()) return true;
                const ParsedTensor* t = it->second;
                size_t expected = 1;
                for (size_t dim : t->shape) expected *= dim;
                dst.resize(expected);

                if (t->dtype == DType::Float32) {
                    return load_tensor_data(path, data_offset, *t, dst.data(), dst.size() * sizeof(float), error);
                }
                if (t->dtype == DType::Float16) {
                    std::vector<uint16_t> tmp(expected);
                    if (!load_tensor_data(path, data_offset, *t, tmp.data(), tmp.size() * sizeof(uint16_t), error)) {
                        return false;
                    }
                    if (!Mimir::fpgaDecode16(dst.data(), tmp.data(), expected, false)) HardwareOpt::fp16_to_fp32_f16c(dst.data(), tmp.data(), expected);
                    return true;
                }

                if (t->dtype == DType::BFloat16) {
                    std::vector<uint16_t> tmp(expected);
                    if (!load_tensor_data(path, data_offset, *t, tmp.data(), tmp.size() * sizeof(uint16_t), error)) {
                        return false;
                    }
                    if (!Mimir::fpgaDecode16(dst.data(), tmp.data(), expected, true)) HardwareOpt::bf16_to_fp32(dst.data(), tmp.data(), expected);
                    return true;
                }

                if (t->dtype == DType::Float64) {
                    std::vector<double> tmp(expected);
                    if (!load_tensor_data(path, data_offset, *t, tmp.data(), tmp.size() * sizeof(double), error)) {
                        return false;
                    }
                    for (size_t i = 0; i < expected; ++i) dst[i] = static_cast<float>(tmp[i]);
                    return true;
                }

                if (options.strict_mode) {
                    if (error) {
                        *error = "Unsupported dtype for optimizer tensor: " + name;
                    }
                    return false;
                }
                return true;
            };

            Optimizer* optp = model.getMutableSerializedOptimizer();
            if (!optp) {
                Optimizer tmp;
                model.setSerializedOptimizer(tmp);
                optp = model.getMutableSerializedOptimizer();
            }

            if (!load_opt_vec("optimizer/m", optp->m)) return false;
            if (!load_opt_vec("optimizer/v", optp->v)) return false;
            validate_optimizer_state(model, *optp, optimizer_meta);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Model apply error: ") + e.what();
        }
        return false;
    }
}

uint64_t SafeTensorsReader::read_u64_le(std::ifstream& f) {
    uint8_t bytes[8];
    f.read(reinterpret_cast<char*>(bytes), 8);
    
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= (static_cast<uint64_t>(bytes[i]) << (i * 8));
    }
    return value;
}

} // namespace Serialization
} // namespace Mimir
