#pragma once

#include "Serialization.hpp"
#include "CheckpointHashTree.hpp"
#include "../include/json.hpp"
#include <unordered_map>
#include <vector>

class Model;

namespace Mimir {
namespace Serialization {

using json = nlohmann::json;

/**
 * RawCheckpointReader - Read checkpoint from folder format
 */
class RawCheckpointReader {
public:
    RawCheckpointReader();
    ~RawCheckpointReader();
    
    /**
     * Load model from raw checkpoint format.
     */
    bool load(
        Model& model,
        const std::string& path,
        const LoadOptions& options,
        std::string* error = nullptr
    );
    
private:
    CheckpointHashTree hash_tree_;
    struct TensorMetadata {
        std::string name;
        DType dtype;
        std::vector<size_t> shape;
        size_t byte_size;
        std::string checksum;
        std::string checksum_algo;
        std::string data_file;
    };
    
    /**
     * Load manifest.json.
     */
    bool load_manifest(
        const std::string& root,
        json& manifest_out,
        std::string* error
    );
    
    /**
     * Load architecture.
     */
    bool load_architecture(
        const std::string& root,
        Model& model,
        const LoadOptions& options,
        std::string* error
    );
    
    /**
     * Load tokenizer.
     */
    bool load_tokenizer(
        const std::string& root,
        Model& model,
        std::string* error
    );
    
    /**
     * Load encoder.
     */
    bool load_encoder(
        const std::string& root,
        Model& model,
        std::string* error
    );

    /**
     * Load training state (optimizer) from model/training.json.
     */
    bool load_training(
        const std::string& root,
        Model& model,
        std::string* error
    );
    
    /**
     * Load tensor metadata from JSON.
     */
    bool load_tensor_metadata(
        const std::string& json_path,
        TensorMetadata& metadata_out,
        std::string* error
    );
    
    /**
     * Load tensor data from .bin file.
     */
    bool load_tensor_data(
        const std::string& bin_path,
        const TensorMetadata& metadata,
        void* dest_buffer,
        size_t dest_size,
        std::string* error
    );
    
    /**
     * Verify checksum.
     */
    bool verify_checksum(
        const void* data,
        size_t size,
        const std::string& expected_checksum,
        const std::string& algo
    );
    
    /**
     * Apply tensors to model.
     */
    bool apply_tensors_to_model(
        const std::string& root,
        Model& model,
        const json& manifest,
        const LoadOptions& options,
        std::string* error
    );
};

} // namespace Serialization
} // namespace Mimir
