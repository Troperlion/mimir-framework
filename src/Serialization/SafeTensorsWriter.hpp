#pragma once

#include "Serialization.hpp"
#include "../include/json.hpp"
#include <unordered_map>
#include <vector>

class Model;
class Tokenizer;
class ConditioningEncoder;

namespace Mimir {
namespace Serialization {

using json = nlohmann::json;

/**
 * SafeTensorsWriter - Write SafeTensors format files
 * 
 * Implements the SafeTensors specification:
 * - 8 bytes: header length (uint64 little-endian)
 * - N bytes: JSON header (UTF-8)
 * - M bytes: tensor data (contiguous, little-endian)
 * 
 * Header format:
 * {
 *   "tensor_name": {
 *     "dtype": "F32",
 *     "shape": [dim1, dim2, ...],
 *     "data_offsets": [begin, end]
 *   },
 *   "__metadata__": {
 *     "format": "safetensors",
 *     "mimir_version": "3.5.0",
 *     ...
 *   }
 * }
 */
class SafeTensorsWriter {
public:
    SafeTensorsWriter();
    ~SafeTensorsWriter();
    
    /**
     * Save model to SafeTensors format.
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

    // Buffers owned by the writer (used for non-float tensor payloads like tokenizer/encoder JSON)
    std::vector<std::vector<uint8_t>> owned_buffers_;

    // Owned FP16 payloads (aligned) for float tensors when saving as Float16.
    std::vector<std::vector<float>> owned_f32_buffers_;
    json optimizer_metadata_;
    std::vector<std::vector<uint16_t>> owned_u16_buffers_;

    // Owned FP64 payloads for float tensors when saving as Float64.
    std::vector<std::vector<double>> owned_f64_buffers_;
    
    /**
     * Collect all tensors from model.
     */
    std::vector<TensorData> collect_tensors(
        Model& model,
        const SaveOptions& options
    );
    
    /**
     * Build JSON header.
     */
    json build_header(
        const std::vector<TensorData>& tensors,
        const SaveOptions& options
    );
    
    /**
     * Write SafeTensors file.
     */
    bool write_file(
        const std::string& path,
        const json& header,
        const std::vector<TensorData>& tensors,
        std::string* error
    );
    
    /**
     * Write uint64 as little-endian.
     */
    static void write_u64_le(std::ofstream& f, uint64_t value);
    
    /**
     * Calculate total data size.
     */
    size_t calculate_total_size(const std::vector<TensorData>& tensors) const;
};

} // namespace Serialization
} // namespace Mimir
