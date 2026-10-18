#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <filesystem>

// Forward declarations
class Model;
class Tokenizer;
class ConditioningEncoder;

namespace Mimir {
namespace Serialization {

namespace fs = std::filesystem;

// ============================================================================
// Enums & Configuration Structures
// ============================================================================

enum class CheckpointFormat {
    SafeTensors,    // SafeTensors spec-compliant single-file format
    RawFolder,      // Human-readable folder with JSON manifests
    DebugJson       // Debug JSON dump (not for production)
};

enum class DType {
    Float32,
    Float16,
    BFloat16,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bool
};

struct SaveOptions {
    CheckpointFormat format = CheckpointFormat::SafeTensors;
    bool save_optimizer = false;
    bool save_tokenizer = true;
    bool save_encoder = true;
    bool save_dataset_info = false;
    size_t debug_max_values = 100;  // For DebugJson format (legacy)
    bool include_git_info = true;
    std::string custom_metadata;    // Optional custom JSON metadata
    
    // Enhanced DebugJson options (v1.1.0)
    bool include_gradients = false;
    bool include_optimizer_state = false;
    size_t max_values_per_tensor = 20;
    bool include_activations = false;
    bool include_checksums = false;
    bool include_weight_deltas = false;
};

struct LoadOptions {
    CheckpointFormat format = CheckpointFormat::SafeTensors;
    bool load_optimizer = false;
    bool load_tokenizer = true;
    bool load_encoder = true;
    bool strict_mode = true;        // Fail on missing tensors
    bool validate_checksums = true;
    std::string mapping_json;       // Optional JSON mapping for external tensor names -> Mimir executable layers

    // When checkpoints include architecture/config metadata, callers sometimes
    // want to load ONLY weights into an already-constructed model (e.g. load a
    // standalone VAE checkpoint into a composite model).
    bool apply_model_name = true;
    bool apply_model_config = true;
    bool metadata_only = false;     // Load architecture/config without tensors or auxiliary state
};

// ============================================================================
// Tensor Info Structure
// ============================================================================

struct TensorInfo {
    std::string name;
    DType dtype;
    std::vector<size_t> shape;
    size_t byte_size;
    size_t data_offset;  // For SafeTensors format
    std::string checksum;  // xxhash64 or sha256
};

// ============================================================================
// High-Level API
// ============================================================================

/**
 * Save a model checkpoint to disk.
 * 
 * @param model The model to save
 * @param path Output path (file for SafeTensors, directory for RawFolder)
 * @param options Save options
 * @param error Optional error message output
 * @return true if successful, false otherwise
 */
bool save_checkpoint(
    Model& model,
    const std::string& path,
    const SaveOptions& options,
    std::string* error = nullptr
);

/**
 * Load a model checkpoint from disk.
 * 
 * @param model The model to load into
 * @param path Input path (file for SafeTensors, directory for RawFolder)
 * @param options Load options
 * @param error Optional error message output
 * @return true if successful, false otherwise
 */
bool load_checkpoint(
    Model& model,
    const std::string& path,
    const LoadOptions& options,
    std::string* error = nullptr
);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get the string representation of a dtype.
 */
std::string dtype_to_string(DType dtype);

/**
 * Parse dtype from string.
 */
DType string_to_dtype(const std::string& str);

/**
 * Get size in bytes for a dtype.
 */
size_t dtype_size(DType dtype);

/**
 * Detect format from path.
 */
CheckpointFormat detect_format(const std::string& path);

/**
 * Get version string.
 */
std::string get_mimir_version();

/**
 * Get git commit hash (if available).
 */
std::string get_git_commit();

} // namespace Serialization
} // namespace Mimir
