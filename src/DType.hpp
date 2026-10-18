#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace Mimir {

enum class DType : uint8_t {
    UNKNOWN = 0,
    BOOL,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    BF16,
    F32,
    F64,
};

inline constexpr bool dtype_is_floating(DType dtype) {
    return dtype == DType::F16 || dtype == DType::BF16 ||
           dtype == DType::F32 || dtype == DType::F64;
}

inline constexpr bool dtype_is_integral(DType dtype) {
    return dtype >= DType::BOOL && dtype <= DType::I64;
}

inline constexpr size_t dtype_size_bytes(DType dt) {
    switch (dt) {
        case DType::BOOL: return 1;
        case DType::U8: return 1;
        case DType::I8: return 1;
        case DType::U16: return 2;
        case DType::I16: return 2;
        case DType::U32: return 4;
        case DType::I32: return 4;
        case DType::U64: return 8;
        case DType::I64: return 8;
        case DType::F16: return 2;
        case DType::BF16: return 2;
        case DType::F32: return 4;
        case DType::F64: return 8;
        default: return 0;
    }
}

inline constexpr const char* dtype_to_string(DType dt) {
    switch (dt) {
        case DType::BOOL: return "bool";
        case DType::U8: return "uint8";
        case DType::I8: return "int8";
        case DType::U16: return "uint16";
        case DType::I16: return "int16";
        case DType::U32: return "uint32";
        case DType::I32: return "int32";
        case DType::U64: return "uint64";
        case DType::I64: return "int64";
        case DType::F16: return "float16";
        case DType::BF16: return "bfloat16";
        case DType::F32: return "float32";
        case DType::F64: return "float64";
        default: return "unknown";
    }
}

inline DType parse_dtype(std::string_view s) {
    if (s == "float" || s == "f32" || s == "float32") return DType::F32;
    if (s == "double" || s == "f64" || s == "float64") return DType::F64;
    if (s == "f16" || s == "float16" || s == "fp16") return DType::F16;
    if (s == "bf16" || s == "bfloat16") return DType::BF16;

    if (s == "i8" || s == "int8") return DType::I8;
    if (s == "u8" || s == "uint8") return DType::U8;
    if (s == "i16" || s == "int16") return DType::I16;
    if (s == "u16" || s == "uint16") return DType::U16;
    if (s == "i32" || s == "int32") return DType::I32;
    if (s == "u32" || s == "uint32") return DType::U32;
    if (s == "i64" || s == "int64") return DType::I64;
    if (s == "u64" || s == "uint64") return DType::U64;

    if (s == "bool" || s == "b1") return DType::BOOL;

    return DType::UNKNOWN;
}

// Convenience: map common safetensors dtype codes.
inline DType parse_dtype_safetensors(std::string_view s) {
    // safetensors canonical strings include e.g. "F32", "F16", "I64", "U16".
    if (s == "F32") return DType::F32;
    if (s == "F16") return DType::F16;
    if (s == "BF16") return DType::BF16;
    if (s == "F64") return DType::F64;
    if (s == "I8") return DType::I8;
    if (s == "U8") return DType::U8;
    if (s == "I16") return DType::I16;
    if (s == "U16") return DType::U16;
    if (s == "I32") return DType::I32;
    if (s == "U32") return DType::U32;
    if (s == "I64") return DType::I64;
    if (s == "U64") return DType::U64;
    if (s == "BOOL") return DType::BOOL;
    return parse_dtype(s);
}

} // namespace Mimir
