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
 * SafeTensorsReader - Read SafeTensors format files
 */
class SafeTensorsReader {
public:
    SafeTensorsReader();
    ~SafeTensorsReader();
    
    /**
     * Load model from SafeTensors format.
     */
    bool load(
        Model& model,
        const std::string& path,
        const LoadOptions& options,
        std::string* error = nullptr
    );
    
private:
    CheckpointHashTree hash_tree_;
    struct ParsedTensor {
        std::string name;
        DType dtype;
        std::vector<size_t> shape;
        size_t data_begin;
        size_t data_end;
    };
    
    /**
     * Parse SafeTensors file header.
     */
    bool parse_header(
        const std::string& path,
        json& header_out,
        std::vector<ParsedTensor>& tensors_out,
        size_t& data_offset_out,
        std::string* error
    );
    
    /**
     * Load tensor data from file.
     */
    bool load_tensor_data(
        const std::string& path,
        size_t data_offset,
        const ParsedTensor& tensor,
        void* dest_buffer,
        size_t dest_size,
        std::string* error
    );
    
    /**
     * Apply loaded tensors to model.
     */
    bool apply_tensors_to_model(
        Model& model,
        const std::vector<ParsedTensor>& tensors,
        const std::string& path,
        size_t data_offset,
        const LoadOptions& options,
        std::string* error
    );
    
    /**
     * Read uint64 as little-endian.
     */
    static uint64_t read_u64_le(std::ifstream& f);
};

} // namespace Serialization
} // namespace Mimir
