#include "runtimes/fpga/FpgaIo.hpp"
#include "Serialization.hpp"
#include "SafeTensorsWriter.hpp"
#include "SafeTensorsReader.hpp"
#include "RawCheckpointWriter.hpp"
#include "RawCheckpointReader.hpp"
#include "DebugJsonDump.hpp"
#include "../Model.hpp"
#include "../DType.hpp"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cctype>

namespace Mimir {
namespace Serialization {

#ifndef MIMIR_PROJECT_VERSION
#define MIMIR_PROJECT_VERSION "0.0.0"
#endif

// ============================================================================
// High-Level API Implementation
// ============================================================================

bool save_checkpoint(
    Model& model,
    const std::string& path,
    const SaveOptions& options,
    std::string* error
) {
    try {
        FpgaCheckpointTransaction fpga_transaction(true);
        const bool completed = [&]() {
        switch (options.format) {
            case CheckpointFormat::SafeTensors: {
                SafeTensorsWriter writer;
                return writer.save(model, path, options, error);
            }
            
            case CheckpointFormat::RawFolder: {
                RawCheckpointWriter writer;
                return writer.save(model, path, options, error);
            }
            
            case CheckpointFormat::DebugJson: {
                DebugJsonDump dumper;
                
                // Convert SaveOptions to DebugJsonOptions
                DebugJsonOptions debug_opts;
                debug_opts.include_gradients = options.include_gradients;
                debug_opts.include_optimizer_state = options.include_optimizer_state || options.save_optimizer;
                debug_opts.max_values_per_tensor = options.max_values_per_tensor;
                debug_opts.include_activations = options.include_activations;
                debug_opts.include_checksums = options.include_checksums;
                debug_opts.include_weight_deltas = options.include_weight_deltas;
                debug_opts.include_git_info = options.include_git_info;
                debug_opts.save_tokenizer = options.save_tokenizer;
                debug_opts.save_encoder = options.save_encoder;
                
                return dumper.save_enhanced(path, model, debug_opts, error);
            }
            
            default:
                if (error) {
                    *error = "Unknown checkpoint format";
                }
                return false;
        }
        }();
        if(completed) {
            fpga_transaction.verifyFiles(path);
            fpga_transaction.finish();
        }
        return completed;
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Exception during save: ") + e.what();
        }
        return false;
    }
}

bool load_checkpoint(
    Model& model,
    const std::string& path,
    const LoadOptions& options,
    std::string* error
) {
    try {
        FpgaCheckpointTransaction fpga_transaction(false);
        fpga_transaction.verifyFiles(path);
        const bool completed = [&]() {
        switch (options.format) {
            case CheckpointFormat::SafeTensors: {
                SafeTensorsReader reader;
                return reader.load(model, path, options, error);
            }
            
            case CheckpointFormat::RawFolder: {
                RawCheckpointReader reader;
                return reader.load(model, path, options, error);
            }
            
            case CheckpointFormat::DebugJson: {
                if (error) {
                    *error = "DebugJson format is write-only (for debugging)";
                }
                return false;
            }
            
            default:
                if (error) {
                    *error = "Unknown checkpoint format";
                }
                return false;
        }
        }();
        if(completed) {
            fpga_transaction.finish();
        }
        return completed;
    } catch (const std::exception& e) {
        if (error) {
            *error = std::string("Exception during load: ") + e.what();
        }
        return false;
    }
}

// ============================================================================
// Utility Functions Implementation
// ============================================================================

std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "F32";
        case DType::Float16: return "F16";
        case DType::BFloat16: return "BF16";
        case DType::Float64: return "F64";
        case DType::Int8: return "I8";
        case DType::Int32: return "I32";
        case DType::Int16: return "I16";
        case DType::Int64: return "I64";
        case DType::Uint8: return "U8";
        case DType::Uint16: return "U16";
        case DType::Uint32: return "U32";
        case DType::Uint64: return "U64";
        case DType::Bool: return "BOOL";
        default: return "UNKNOWN";
    }
}

DType string_to_dtype(const std::string& str) {
    // Accept all common aliases/cases used across Lua/config/runtime.
    // Examples: f16, fp16, F16, float16, bf16, f32, float32, etc.
    const Mimir::DType rt = Mimir::parse_dtype_safetensors(str);
    switch (rt) {
        case Mimir::DType::F32: return DType::Float32;
        case Mimir::DType::F16: return DType::Float16;
        case Mimir::DType::BF16: return DType::BFloat16;
        case Mimir::DType::F64: return DType::Float64;
        case Mimir::DType::I8: return DType::Int8;
        case Mimir::DType::I32: return DType::Int32;
        case Mimir::DType::I16: return DType::Int16;
        case Mimir::DType::I64: return DType::Int64;
        case Mimir::DType::U8: return DType::Uint8;
        case Mimir::DType::U16: return DType::Uint16;
        case Mimir::DType::U32: return DType::Uint32;
        case Mimir::DType::U64: return DType::Uint64;
        case Mimir::DType::BOOL: return DType::Bool;
        default: break;
    }

    // Last-resort case-insensitive fallback for serialized tags like "f32".
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    if (s == "F32") return DType::Float32;
    if (s == "F16") return DType::Float16;
    if (s == "BF16") return DType::BFloat16;
    if (s == "F64") return DType::Float64;
    if (s == "I8") return DType::Int8;
    if (s == "I32") return DType::Int32;
    if (s == "I16") return DType::Int16;
    if (s == "I64") return DType::Int64;
    if (s == "U8") return DType::Uint8;
    if (s == "U16") return DType::Uint16;
    if (s == "U32") return DType::Uint32;
    if (s == "U64") return DType::Uint64;
    if (s == "BOOL") return DType::Bool;

    throw std::runtime_error("Unknown dtype string: " + str);
}

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::BFloat16: return 2;
        case DType::Float64: return 8;
        case DType::Int8: return 1;
        case DType::Int32: return 4;
        case DType::Int16: return 2;
        case DType::Int64: return 8;
        case DType::Uint8: return 1;
        case DType::Uint16: return 2;
        case DType::Uint32: return 4;
        case DType::Uint64: return 8;
        case DType::Bool: return 1;
        default: return 0;
    }
}

CheckpointFormat detect_format(const std::string& path) {
    fs::path p(path);
    
    if (fs::is_directory(p)) {
        // Check for manifest.json to distinguish RawFolder
        if (fs::exists(p / "manifest.json")) {
            return CheckpointFormat::RawFolder;
        }
    } else if (fs::is_regular_file(p)) {
        // Check extension
        std::string ext = p.extension().string();
        if (ext == ".safetensors" || ext == ".st") {
            return CheckpointFormat::SafeTensors;
        } else if (ext == ".json") {
            return CheckpointFormat::DebugJson;
        }
    }
    
    // Default to SafeTensors
    return CheckpointFormat::SafeTensors;
}

std::string get_mimir_version() {
    // Try to read from VERSION file
    fs::path version_file = fs::current_path() / "VERSION";
    if (fs::exists(version_file)) {
        std::ifstream ifs(version_file);
        std::string version;
        if (std::getline(ifs, version)) {
            const auto first = version.find_first_not_of(" \t\r\n");
            if (first != std::string::npos) {
                const auto last = version.find_last_not_of(" \t\r\n");
                return version.substr(first, last - first + 1);
            }
        }
    }
    return MIMIR_PROJECT_VERSION;
}

std::string get_git_commit() {
    // Try to get git commit hash
#ifdef _WIN32
    const char* cmd = "git rev-parse --short HEAD 2>NUL";
    FILE* pipe = _popen(cmd, "r");
#else
    const char* cmd = "git rev-parse --short HEAD 2>/dev/null";
    FILE* pipe = popen(cmd, "r");
#endif
    if (!pipe) return "";
    
    char buffer[128];
    std::string result;
    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result = buffer;
        // Remove trailing newline
        if (!result.empty() && result.back() == '\n') {
            result.pop_back();
        }
    }
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return result;
}

} // namespace Serialization
} // namespace Mimir
