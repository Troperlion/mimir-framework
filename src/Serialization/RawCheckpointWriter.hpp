#pragma once

#include "Serialization.hpp"
#include "../include/json.hpp"
#include <unordered_map>
#include <vector>

class Model;

namespace Mimir {
namespace Serialization {

using json = nlohmann::json;

/**
 * RawCheckpointWriter - Write checkpoint in folder format
 * 
 * Directory structure:
 * checkpoint_root/
 *   manifest.json          (root config)
 *   tensors/
 *     <name>.bin           (raw bytes)
 *     <name>.json          (per-tensor config)
 *   model/
 *     architecture.json    (layers, params_count, etc)
 *     training.json        (optimizer state if exists)
 *   tokenizer/
 *     tokenizer.json
 *   encoder/
 *     encoder.json
 */
class RawCheckpointWriter {
public:
    RawCheckpointWriter();
    ~RawCheckpointWriter();
    
    /**
     * Save model to raw checkpoint format.
     */
    bool save(
        Model& model,
        const std::string& path,
        const SaveOptions& options,
        std::string* error = nullptr
    );
    
private:
    struct TensorData {
        std::string name;
        DType dtype;
        std::vector<size_t> shape;
        const void* data_ptr;
        size_t byte_size;
    };

    // Owned FP16 payloads (aligned) for float tensors when saving as Float16.
    std::vector<std::vector<float>> owned_f32_buffers_;
    json optimizer_metadata_;
    std::vector<std::vector<uint16_t>> owned_u16_buffers_;

    // Owned FP64 payloads for float tensors when saving as Float64.
    std::vector<std::vector<double>> owned_f64_buffers_;
    
    /**
     * Create checkpoint directory structure.
     */
    bool create_structure(const std::string& root, std::string* error);
    
    /**
     * Collect tensors from model.
     */
    std::vector<TensorData> collect_tensors(
        Model& model,
        const SaveOptions& options
    );
    
    /**
     * Save tensor to .bin and .json files.
     */
    bool save_tensor(
        const std::string& root,
        const TensorData& tensor,
        std::string* error
    );
    
    /**
     * Save model architecture.
     */
    bool save_architecture(
        const std::string& root,
        Model& model,
        std::string* error
    );

    /**
     * Save training state (optimizer).
     */
    bool save_training(
        const std::string& root,
        Model& model,
        std::string* error
    );
    
    /**
     * Save tokenizer.
     */
    bool save_tokenizer(
        const std::string& root,
        Model& model,
        std::string* error
    );
    
    /**
     * Save encoder.
     */
    bool save_encoder(
        const std::string& root,
        Model& model,
        std::string* error
    );
    
    /**
     * Create manifest.json.
     */
    bool save_manifest(
        const std::string& root,
        const std::vector<TensorData>& tensors,
        const SaveOptions& options,
        std::string* error
    );
    
    /**
     * Calculate checksum (xxhash64 or sha256).
     */
    std::string calculate_checksum(const void* data, size_t size);
};

} // namespace Serialization
} // namespace Mimir
