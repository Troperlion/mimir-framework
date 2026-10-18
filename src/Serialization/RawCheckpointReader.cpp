#include "runtimes/fpga/FpgaIo.hpp"
#include "RawCheckpointReader.hpp"
#include "../Model.hpp"
#include "CheckpointState.hpp"
#include "../Tokenizer.hpp"
#include "../Encoder.hpp"
#include "HardwareOpt.hpp"
#include "../Sha256.hpp"
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace Mimir {
namespace Serialization {

RawCheckpointReader::RawCheckpointReader() {
}

RawCheckpointReader::~RawCheckpointReader() {
}

bool RawCheckpointReader::load(
    Model& model,
    const std::string& path,
    const LoadOptions& requested_options,
    std::string* error
) {
    hash_tree_.begin(path);
    CheckpointHashTree::Scope hash_scope{hash_tree_};
    auto options = requested_options;
    try {
        fs::path root(path);
        std::cerr << "[serialization] raw load path=" << root.string()
                  << " load_tokenizer=" << options.load_tokenizer
                  << " load_encoder=" << options.load_encoder
                  << " load_optimizer=" << options.load_optimizer << std::endl;
        
        // Check if directory exists
        if (!fs::exists(root) || !fs::is_directory(root)) {
            if (error) {
                *error = "Checkpoint directory not found: " + path;
            }
            return false;
        }
        
        // Load manifest
        json manifest;
        if (!load_manifest(root.string(), manifest, error)) {
            return false;
        }
        
        if (manifest.contains("components")) {
            const auto& components = manifest["components"];
            if (components.value("encoder", true) == false && options.load_encoder) {
                options.load_encoder = false;
                model.setHasEncoder(false);
            }
            if (components.value("tokenizer", true) == false && options.load_tokenizer) {
                options.load_tokenizer = false;
                model.setHasTokenizer(false);
            }
        }

        // Load architecture
        if (!load_architecture(root.string(), model, options, error)) {
            return false;
        }

        if (options.metadata_only) {
            return true;
        }
        
        // Load tokenizer if requested
        if (options.load_tokenizer) {
            if (!load_tokenizer(root.string(), model, error)) {
                // Non-fatal if tokenizer missing in non-strict mode
                if (options.strict_mode) {
                    return false;
                }
            }
        }
        
        // Load encoder if requested
        if (options.load_encoder) {
            if (!load_encoder(root.string(), model, error)) {
                // Non-fatal if encoder missing in non-strict mode
                if (options.strict_mode) {
                    return false;
                }
            }
        }

        // Load training/optimizer state if requested
        if (options.load_optimizer) {
            // Non-fatal if absent unless strict_mode
            if (!load_training(root.string(), model, error)) {
                if (options.strict_mode) {
                    return false;
                }
            }
        }
        
        // Apply tensors to model
        if (!apply_tensors_to_model(root.string(), model, manifest, options, error)) return false;
        if (options.load_optimizer && model.getSerializedOptimizer()) {
            std::ifstream input(root / "model" / "training.json");
            if (input) {
                json metadata; input >> metadata;
                validate_optimizer_state(model, *model.getSerializedOptimizer(), metadata);
            }
        }
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Raw checkpoint load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_training(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path training_path = fs::path(root) / "model" / "training.json";
        std::cerr << "[serialization] raw load_training path=" << training_path.string() << std::endl;
        if (!fs::exists(training_path)) {
            if (error) {
                *error = "training.json not found";
            }
            return false;
        }

        std::ifstream file(training_path);
        if (!file) {
            if (error) {
                *error = "Failed to open training.json";
            }
            return false;
        }

        json j;
        file >> j;

        if (j.contains("loop")) model.modelConfig["training_state"] = j["loop"];
        if (!j.value("has_optimizer", false)) {
            model.clearSerializedOptimizer();
            return true;
        }

        Optimizer opt = optimizer_from_metadata(j);
        if (j.contains("loop")) model.modelConfig["training_state"] = j["loop"];
        model.setSerializedOptimizer(opt);
        return true;
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Training load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_manifest(
    const std::string& root,
    json& manifest_out,
    std::string* error
) {
    try {
        fs::path manifest_path = fs::path(root) / "manifest.json";
        std::cerr << "[serialization] raw load_manifest path=" << manifest_path.string() << std::endl;
        
        if (!fs::exists(manifest_path)) {
            if (error) {
                *error = "manifest.json not found";
            }
            return false;
        }
        
        std::ifstream file(manifest_path);
        if (!file) {
            if (error) {
                *error = "Failed to open manifest.json";
            }
            return false;
        }
        
        file >> manifest_out;
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Manifest load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_architecture(
    const std::string& root,
    Model& model,
    const LoadOptions& options,
    std::string* error
) {
    try {
        fs::path arch_path = fs::path(root) / "model" / "architecture.json";
        std::cerr << "[serialization] raw load_architecture path=" << arch_path.string() << std::endl;
        
        if (!fs::exists(arch_path)) {
            if (error) {
                *error = "architecture.json not found";
            }
            return false;
        }
        
        std::ifstream file(arch_path);
        if (!file) {
            if (error) {
                *error = "Failed to open architecture.json";
            }
            return false;
        }
        
        json arch;
        file >> arch;
        
        // Load basic info
        if (options.apply_model_name && arch.contains("model_name")) {
            model.setModelName(arch["model_name"].get<std::string>());
        }

        // Load model config (critical for downstream consumers relying on model.modelConfig)
        if (options.apply_model_config && arch.contains("model_config")) {
            try {
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
            } catch (...) {
                if (error) {
                    *error = "Invalid model_config in architecture.json";
                }
                return false;
            }
        }
        
        // Load layers structure
        // IMPORTANT: l'architecture sauvegardée ne contient pas toutes les métadonnées
        // (in_features/out_features, kernels, etc.). Si le caller a déjà construit
        // l'architecture via un builder (recommandé), on ne l'écrase pas.
        if (!options.metadata_only && arch.contains("layers") && model.getLayers().empty()) {
            model.getMutableLayers().clear();
            for (const auto& layer_obj : arch["layers"]) {
                const std::string name = layer_obj.value("name", "");
                const std::string type = layer_obj.value("type", "");
                const size_t params_count = layer_obj.value("params_count", 0);
                Layer layer(name, type, params_count); // initialise type_enum + normalise le type
                layer.trainable_parameter = layer_obj.value("trainable_parameter", false);
                layer.skip_input_index = layer_obj.value("skip_input_index", -1);
                model.getMutableLayers().push_back(std::move(layer));
            }

            // Allocate parameters based on architecture (si on vient de reconstruire)
            model.allocateParams();
        }
        
        if (!options.metadata_only && arch.contains("layers")) {
            for (const auto& entry : arch["layers"]) {
                if (entry.contains("skip_input_index")) {
                    if (auto* layer = model.getLayerByName(entry.value("name", "")))
                        layer->skip_input_index = entry["skip_input_index"].get<int>();
                }
            }
        }

        // Load I/O dimensions
        // Note: tw and th are read-only via width()/height(), set during model construction
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Architecture load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_tokenizer(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path tok_path = fs::path(root) / "tokenizer" / "tokenizer.json";
        std::cerr << "[serialization] raw load_tokenizer path=" << tok_path.string() << std::endl;
        
        if (!fs::exists(tok_path)) {
            if (error) {
                *error = "tokenizer.json not found";
            }
            return false;
        }
        
        std::ifstream file(tok_path);
        if (!file) {
            if (error) {
                *error = "Failed to open tokenizer.json";
            }
            return false;
        }
        
        json tok_json;
        file >> tok_json;
        
        auto& tokenizer = model.getMutableTokenizer();
        tokenizer.from_json(tok_json);
        model.setHasTokenizer(true);
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Tokenizer load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_encoder(
    const std::string& root,
    Model& model,
    std::string* error
) {
    try {
        fs::path enc_path = fs::path(root) / "encoder" / "encoder.json";
        std::cerr << "[serialization] raw load_encoder path=" << enc_path.string() << std::endl;
        
        if (!fs::exists(enc_path)) {
            if (error) {
                *error = "encoder.json not found";
            }
            return false;
        }
        
        std::ifstream file(enc_path);
        if (!file) {
            if (error) {
                *error = "Failed to open encoder.json";
            }
            return false;
        }
        
        json enc_json;
        file >> enc_json;
        
        auto& encoder = model.getMutableEncoder();
        encoder.from_json(enc_json);
        
        model.setHasEncoder(true);
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("ConditioningEncoder load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_tensor_metadata(
    const std::string& json_path,
    TensorMetadata& metadata_out,
    std::string* error
) {
    try {
        if (!fs::exists(json_path)) {
            if (error) {
                *error = "Tensor JSON not found: " + json_path;
            }
            return false;
        }
        
        std::ifstream file(json_path);
        if (!file) {
            if (error) {
                *error = "Failed to open tensor JSON: " + json_path;
            }
            return false;
        }
        
        json tensor_json;
        file >> tensor_json;
        
        metadata_out.name = tensor_json.value("name", "");
        metadata_out.dtype = string_to_dtype(tensor_json.value("dtype", "F32"));
        metadata_out.shape = tensor_json.value("shape", std::vector<size_t>{});
        metadata_out.byte_size = tensor_json.value("byte_size", 0);
        metadata_out.checksum = tensor_json.value("checksum", "");
        metadata_out.checksum_algo = tensor_json.value("checksum_algo", "sha256");
        metadata_out.data_file = tensor_json.value("data_file", "");
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Tensor metadata load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::load_tensor_data(
    const std::string& bin_path,
    const TensorMetadata& metadata,
    void* dest_buffer,
    size_t dest_size,
    std::string* error
) {
    try {
        if (!fs::exists(bin_path)) {
            if (error) {
                *error = "Tensor data file not found: " + bin_path;
            }
            return false;
        }
        
        std::ifstream file(bin_path, std::ios::binary);
        if (!file) {
            if (error) {
                *error = "Failed to open tensor data: " + bin_path;
            }
            return false;
        }
        
        // Check size
        if (metadata.byte_size > dest_size) {
            if (error) {
                *error = "Buffer too small for tensor: " + metadata.name;
            }
            return false;
        }
        
        // Read data
        file.read(static_cast<char*>(dest_buffer), metadata.byte_size);
        if (!file) {
            if (error) {
                *error = "Failed to read tensor data: " + metadata.name;
            }
            return false;
        }

        const std::string checksum_algo = metadata.checksum_algo.empty()
            ? "sha256"
            : metadata.checksum_algo;
        const std::string checksum = metadata.checksum.empty()
            ? sha256(dest_buffer, metadata.byte_size)
            : metadata.checksum;
        hash_tree_.add(metadata.name, checksum_algo, checksum);
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Tensor data load error: ") + e.what();
        }
        return false;
    }
}

bool RawCheckpointReader::verify_checksum(
    const void* data,
    size_t size,
    const std::string& expected_checksum,
    const std::string& algo
) {
    if (algo == "sha256") {
        std::string input(static_cast<const char*>(data), size);
        std::string computed = sha256(input);
        return computed == expected_checksum;
    }
    // Unknown algorithm - skip verification
    return true;
}

bool RawCheckpointReader::apply_tensors_to_model(
    const std::string& root,
    Model& model,
    const json& manifest,
    const LoadOptions& options,
    std::string* error
) {
    try {
        fs::path root_path(root);
        
        if (!manifest.contains("tensor_index")) {
            if (error) {
                *error = "Missing tensor_index in manifest";
            }
            return false;
        }
        
        const auto& layers = model.getLayers();
        
        // Build map: tensor name -> layer index
        std::unordered_map<std::string, size_t> layer_map;
        for (size_t i = 0; i < layers.size(); ++i) {
            std::string tensor_name = layers[i].name + "_weights";
            layer_map[tensor_name] = i;
        }
        
        // Load each tensor
        for (const auto& t_entry : manifest["tensor_index"]) {
            std::string tensor_name = t_entry.value("name", "");
            std::string json_file = t_entry.value("json_file", "");
            
            fs::path json_path = root_path / json_file;
            
            // Load metadata
            TensorMetadata t_metadata;
            if (!load_tensor_metadata(json_path.string(), t_metadata, error)) {
                if (options.strict_mode) {
                    return false;
                }
                continue;
            }
            
            // Find corresponding layer
            auto it = layer_map.find(tensor_name);
            if (it == layer_map.end()) {
                // Optimizer tensors (optional)
                if ((tensor_name == "optimizer/m" || tensor_name == "optimizer/v")) {
                    if (!options.load_optimizer) {
                        // Known-but-ignored tensor
                        continue;
                    }

                    Optimizer* opt = model.getMutableSerializedOptimizer();
                    if (!opt) {
                        // If training.json wasn't loaded, create a default optimizer container
                        Optimizer tmp;
                        model.setSerializedOptimizer(tmp);
                        opt = model.getMutableSerializedOptimizer();
                    }

                    size_t expected_size = 1;
                    for (size_t dim : t_metadata.shape) expected_size *= dim;

                    std::vector<float>* dst = (tensor_name == "optimizer/m") ? &opt->m : &opt->v;
                    dst->resize(expected_size);

                    fs::path bin_path = root_path / "tensors" / t_metadata.data_file;

                    if (t_metadata.dtype == DType::Float32) {
                        if (!load_tensor_data(
                            bin_path.string(), t_metadata,
                            dst->data(),
                            dst->size() * sizeof(float),
                            error
                        )) {
                            return false;
                        }

                        if (options.validate_checksums && !t_metadata.checksum.empty()) {
                            if (!verify_checksum(
                                dst->data(),
                                t_metadata.byte_size,
                                t_metadata.checksum,
                                t_metadata.checksum_algo
                            )) {
                                if (error) *error = "Checksum mismatch for: " + tensor_name;
                                return false;
                            }
                        }
                    } else if (t_metadata.dtype == DType::Float16) {
                        if (t_metadata.byte_size != expected_size * sizeof(uint16_t)) {
                            if (error) *error = "Byte size mismatch for: " + tensor_name;
                            return false;
                        }
                        std::vector<uint16_t> tmp(expected_size);
                        if (!load_tensor_data(
                            bin_path.string(), t_metadata,
                            tmp.data(),
                            tmp.size() * sizeof(uint16_t),
                            error
                        )) {
                            return false;
                        }

                        if (options.validate_checksums && !t_metadata.checksum.empty()) {
                            if (!verify_checksum(
                                tmp.data(),
                                tmp.size() * sizeof(uint16_t),
                                t_metadata.checksum,
                                t_metadata.checksum_algo
                            )) {
                                if (error) *error = "Checksum mismatch for: " + tensor_name;
                                return false;
                            }
                        }

                        if (!Mimir::fpgaDecode16(dst->data(), tmp.data(), expected_size, false)) HardwareOpt::fp16_to_fp32_f16c(dst->data(), tmp.data(), expected_size);
                    } else if (t_metadata.dtype == DType::BFloat16) {
                        if (t_metadata.byte_size != expected_size * sizeof(uint16_t)) {
                            if (error) *error = "Byte size mismatch for: " + tensor_name;
                            return false;
                        }
                        std::vector<uint16_t> tmp(expected_size);
                        if (!load_tensor_data(
                            bin_path.string(), t_metadata,
                            tmp.data(),
                            tmp.size() * sizeof(uint16_t),
                            error
                        )) {
                            return false;
                        }

                        if (options.validate_checksums && !t_metadata.checksum.empty()) {
                            if (!verify_checksum(
                                tmp.data(),
                                tmp.size() * sizeof(uint16_t),
                                t_metadata.checksum,
                                t_metadata.checksum_algo
                            )) {
                                if (error) *error = "Checksum mismatch for: " + tensor_name;
                                return false;
                            }
                        }

                        if (!Mimir::fpgaDecode16(dst->data(), tmp.data(), expected_size, true)) HardwareOpt::bf16_to_fp32(dst->data(), tmp.data(), expected_size);
                    } else if (t_metadata.dtype == DType::Float64) {
                        if (t_metadata.byte_size != expected_size * sizeof(double)) {
                            if (error) *error = "Byte size mismatch for: " + tensor_name;
                            return false;
                        }
                        std::vector<double> tmp(expected_size);
                        if (!load_tensor_data(
                            bin_path.string(), t_metadata,
                            tmp.data(),
                            tmp.size() * sizeof(double),
                            error
                        )) {
                            return false;
                        }

                        if (options.validate_checksums && !t_metadata.checksum.empty()) {
                            if (!verify_checksum(
                                tmp.data(),
                                tmp.size() * sizeof(double),
                                t_metadata.checksum,
                                t_metadata.checksum_algo
                            )) {
                                if (error) *error = "Checksum mismatch for: " + tensor_name;
                                return false;
                            }
                        }

                        for (size_t i = 0; i < expected_size; ++i) (*dst)[i] = static_cast<float>(tmp[i]);
                    } else if (options.strict_mode) {
                        if (error) *error = "Unsupported dtype for: " + tensor_name;
                        return false;
                    }
                    continue;
                }

                // Gradient snapshots (debug-only) - ignore on load
                if (tensor_name.rfind("grads/", 0) == 0) {
                    continue;
                }

                if (tensor_name == "encoder_token_embeddings" && !options.load_encoder) continue;
                // Check if it's encoder embeddings
                if (tensor_name == "encoder_token_embeddings" && options.load_encoder) {
                    auto& enc = model.getMutableEncoder();
                    
                    size_t expected_size = 1;
                    for (size_t dim : t_metadata.shape) {
                        expected_size *= dim;
                    }
                    
                    if (enc.token_embeddings.size() < expected_size) {
                        enc.token_embeddings.resize(expected_size);
                    }
                    
                    fs::path bin_path = root_path / "tensors" / t_metadata.data_file;

                    if (t_metadata.dtype == DType::Float32) {
                        if (!load_tensor_data(
                            bin_path.string(), t_metadata,
                            enc.token_embeddings.data(),
                            enc.token_embeddings.size() * sizeof(float),
                            error
                        )) {
                            return false;
                        }

                        if (options.validate_checksums && !t_metadata.checksum.empty()) {
                            if (!verify_checksum(
                                enc.token_embeddings.data(),
                                t_metadata.byte_size,
                                t_metadata.checksum,
                                t_metadata.checksum_algo
                            )) {
                                if (error) *error = "Checksum mismatch for: " + tensor_name;
                                return false;
                            }
                        }
                    } else if (t_metadata.dtype == DType::Float16) {
                        if (t_metadata.byte_size != expected_size * sizeof(uint16_t)) {
                            if (error) *error = "Byte size mismatch for: " + tensor_name;
                            return false;
                        }
                        std::vector<uint16_t> tmp(expected_size);
                        if (!load_tensor_data(
                            bin_path.string(), t_metadata,
                            tmp.data(),
                            tmp.size() * sizeof(uint16_t),
                            error
                        )) {
                            return false;
                        }

                        if (options.validate_checksums && !t_metadata.checksum.empty()) {
                            if (!verify_checksum(
                                tmp.data(),
                                tmp.size() * sizeof(uint16_t),
                                t_metadata.checksum,
                                t_metadata.checksum_algo
                            )) {
                                if (error) *error = "Checksum mismatch for: " + tensor_name;
                                return false;
                            }
                        }

                        if (!Mimir::fpgaDecode16(enc.token_embeddings.data(), tmp.data(), expected_size, false)) HardwareOpt::fp16_to_fp32_f16c(enc.token_embeddings.data(), tmp.data(), expected_size);
                    } else if (t_metadata.dtype == DType::BFloat16) {
                        if (t_metadata.byte_size != expected_size * sizeof(uint16_t)) {
                            if (error) *error = "Byte size mismatch for: " + tensor_name;
                            return false;
                        }
                        std::vector<uint16_t> tmp(expected_size);
                        if (!load_tensor_data(
                            bin_path.string(), t_metadata,
                            tmp.data(),
                            tmp.size() * sizeof(uint16_t),
                            error
                        )) {
                            return false;
                        }

                        if (options.validate_checksums && !t_metadata.checksum.empty()) {
                            if (!verify_checksum(
                                tmp.data(),
                                tmp.size() * sizeof(uint16_t),
                                t_metadata.checksum,
                                t_metadata.checksum_algo
                            )) {
                                if (error) *error = "Checksum mismatch for: " + tensor_name;
                                return false;
                            }
                        }

                        if (!Mimir::fpgaDecode16(enc.token_embeddings.data(), tmp.data(), expected_size, true)) HardwareOpt::bf16_to_fp32(enc.token_embeddings.data(), tmp.data(), expected_size);
                    } else if (t_metadata.dtype == DType::Float64) {
                        if (t_metadata.byte_size != expected_size * sizeof(double)) {
                            if (error) *error = "Byte size mismatch for: " + tensor_name;
                            return false;
                        }
                        std::vector<double> tmp(expected_size);
                        if (!load_tensor_data(
                            bin_path.string(), t_metadata,
                            tmp.data(),
                            tmp.size() * sizeof(double),
                            error
                        )) {
                            return false;
                        }

                        if (options.validate_checksums && !t_metadata.checksum.empty()) {
                            if (!verify_checksum(
                                tmp.data(),
                                tmp.size() * sizeof(double),
                                t_metadata.checksum,
                                t_metadata.checksum_algo
                            )) {
                                if (error) *error = "Checksum mismatch for: " + tensor_name;
                                return false;
                            }
                        }

                        for (size_t i = 0; i < expected_size; ++i) enc.token_embeddings[i] = static_cast<float>(tmp[i]);
                    } else if (options.strict_mode) {
                        if (error) *error = "Unsupported dtype for: " + tensor_name;
                        return false;
                    }
                    
                    continue;
                }
                
                if (options.strict_mode) {
                    if (error) {
                        *error = "Unknown tensor in strict mode: " + tensor_name;
                    }
                    return false;
                }
                continue;
            }
            
            // Load tensor into layer weight block
            size_t layer_idx = it->second;
            auto& layer = const_cast<Layer&>(layers[layer_idx]);
            
            if (!layer.weight_block) {
                if (error) {
                    *error = "Layer has no weight block: " + layer.name;
                }
                return false;
            }
            
            // Get expected size
            size_t expected_size = 1;
            for (size_t dim : t_metadata.shape) {
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
            
            // Load data
            float* data_ptr = layer.weight_block->getData();
            if (!data_ptr) {
                if (error) {
                    *error = "Failed to get data pointer for: " + tensor_name;
                }
                return false;
            }
            
            fs::path bin_path = root_path / "tensors" / t_metadata.data_file;

            if (t_metadata.dtype == DType::Float32) {
                if (!load_tensor_data(
                    bin_path.string(), t_metadata,
                    data_ptr,
                    actual_size * sizeof(float),
                    error
                )) {
                    return false;
                }

                if (options.validate_checksums && !t_metadata.checksum.empty()) {
                    if (!verify_checksum(
                        data_ptr,
                        t_metadata.byte_size,
                        t_metadata.checksum,
                        t_metadata.checksum_algo
                    )) {
                        if (error) *error = "Checksum mismatch for: " + tensor_name;
                        return false;
                    }
                }
            } else if (t_metadata.dtype == DType::Float16) {
                if (t_metadata.byte_size != actual_size * sizeof(uint16_t)) {
                    if (error) *error = "Byte size mismatch for: " + tensor_name;
                    return false;
                }

                std::vector<uint16_t> tmp(actual_size);
                if (!load_tensor_data(
                    bin_path.string(), t_metadata,
                    tmp.data(),
                    tmp.size() * sizeof(uint16_t),
                    error
                )) {
                    return false;
                }

                if (options.validate_checksums && !t_metadata.checksum.empty()) {
                    if (!verify_checksum(
                        tmp.data(),
                        tmp.size() * sizeof(uint16_t),
                        t_metadata.checksum,
                        t_metadata.checksum_algo
                    )) {
                        if (error) *error = "Checksum mismatch for: " + tensor_name;
                        return false;
                    }
                }

                if (!Mimir::fpgaDecode16(data_ptr, tmp.data(), actual_size, false)) HardwareOpt::fp16_to_fp32_f16c(data_ptr, tmp.data(), actual_size);
            } else if (t_metadata.dtype == DType::BFloat16) {
                if (t_metadata.byte_size != actual_size * sizeof(uint16_t)) {
                    if (error) *error = "Byte size mismatch for: " + tensor_name;
                    return false;
                }

                std::vector<uint16_t> tmp(actual_size);
                if (!load_tensor_data(
                    bin_path.string(), t_metadata,
                    tmp.data(),
                    tmp.size() * sizeof(uint16_t),
                    error
                )) {
                    return false;
                }

                if (options.validate_checksums && !t_metadata.checksum.empty()) {
                    if (!verify_checksum(
                        tmp.data(),
                        tmp.size() * sizeof(uint16_t),
                        t_metadata.checksum,
                        t_metadata.checksum_algo
                    )) {
                        if (error) *error = "Checksum mismatch for: " + tensor_name;
                        return false;
                    }
                }

                if (!Mimir::fpgaDecode16(data_ptr, tmp.data(), actual_size, true)) HardwareOpt::bf16_to_fp32(data_ptr, tmp.data(), actual_size);
            } else if (t_metadata.dtype == DType::Float64) {
                if (t_metadata.byte_size != actual_size * sizeof(double)) {
                    if (error) *error = "Byte size mismatch for: " + tensor_name;
                    return false;
                }

                std::vector<double> tmp(actual_size);
                if (!load_tensor_data(
                    bin_path.string(), t_metadata,
                    tmp.data(),
                    tmp.size() * sizeof(double),
                    error
                )) {
                    return false;
                }

                if (options.validate_checksums && !t_metadata.checksum.empty()) {
                    if (!verify_checksum(
                        tmp.data(),
                        tmp.size() * sizeof(double),
                        t_metadata.checksum,
                        t_metadata.checksum_algo
                    )) {
                        if (error) *error = "Checksum mismatch for: " + tensor_name;
                        return false;
                    }
                }

                for (size_t i = 0; i < actual_size; ++i) data_ptr[i] = static_cast<float>(tmp[i]);
            } else if (options.strict_mode) {
                if (error) *error = "Unsupported dtype for: " + tensor_name;
                return false;
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Apply tensors error: ") + e.what();
        }
        return false;
    }
}

} // namespace Serialization
} // namespace Mimir
