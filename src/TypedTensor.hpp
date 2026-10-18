#pragma once

#include "DType.hpp"
#include "runtimes/cpu/HardwareOpt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace Mimir {

struct Float16 {
    uint16_t bits = 0;
};
struct BFloat16 {
    uint16_t bits = 0;
};
struct Bool8 {
    uint8_t value = 0;
};

using TensorStorage = std::variant<
    std::vector<Bool8>,
    std::vector<uint8_t>, std::vector<int8_t>,
    std::vector<uint16_t>, std::vector<int16_t>,
    std::vector<uint32_t>, std::vector<int32_t>,
    std::vector<uint64_t>, std::vector<int64_t>,
    std::vector<Float16>, std::vector<BFloat16>,
    std::vector<float>, std::vector<double>>;

template <typename T>
inline constexpr bool is_tensor_scalar_v =
    std::is_same_v<T, Bool8> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t> ||
    std::is_same_v<T, uint16_t> || std::is_same_v<T, int16_t> ||
    std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t> ||
    std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t> ||
    std::is_same_v<T, Float16> || std::is_same_v<T, BFloat16> ||
    std::is_same_v<T, float> || std::is_same_v<T, double>;

class TypedTensor {
public:
    TypedTensor() = default;
    TypedTensor(std::vector<int> shape, DType dtype)
        : shape_(std::move(shape)), dtype_(dtype) {
        if (dtype_ == DType::UNKNOWN) throw std::invalid_argument("TypedTensor: unknown dtype");
        numel_ = 1;
        for (int dimension : shape_) {
            if (dimension < 0) throw std::invalid_argument("TypedTensor: negative dimension");
            numel_ *= static_cast<size_t>(dimension);
        }
        storage_ = makeStorage(dtype_, numel_);
    }

    DType dtype() const { return dtype_; }
    const std::vector<int>& shape() const { return shape_; }
    size_t numel() const { return numel_; }
    size_t size_bytes() const { return numel_ * dtype_size_bytes(dtype_); }
    bool differentiable() const { return dtype_is_floating(dtype_); }

    TensorStorage& storage() { return storage_; }
    const TensorStorage& storage() const { return storage_; }

    template <typename Visitor>
    decltype(auto) visit(Visitor&& visitor) {
        return std::visit(std::forward<Visitor>(visitor), storage_);
    }
    template <typename Visitor>
    decltype(auto) visit(Visitor&& visitor) const {
        return std::visit(std::forward<Visitor>(visitor), storage_);
    }

    template <typename T>
    std::vector<T>& values() {
        static_assert(is_tensor_scalar_v<T>, "unsupported tensor scalar");
        return std::get<std::vector<T>>(storage_);
    }
    template <typename T>
    const std::vector<T>& values() const {
        static_assert(is_tensor_scalar_v<T>, "unsupported tensor scalar");
        return std::get<std::vector<T>>(storage_);
    }

    void* data() {
        return visit([](auto& values) -> void* { return values.data(); });
    }
    const void* data() const {
        return visit([](const auto& values) -> const void* { return values.data(); });
    }

    double get(size_t index) const {
        check(index);
        return visit([&](const auto& values) -> double {
            using T = typename std::decay_t<decltype(values)>::value_type;
            const T value = values[index];
            if constexpr (std::is_same_v<T, Float16>) {
                float decoded;
                HardwareOpt::fp16_to_fp32_f16c(&decoded, &value.bits, 1);
                return decoded;
            } else if constexpr (std::is_same_v<T, BFloat16>) {
                float decoded;
                HardwareOpt::bf16_to_fp32(&decoded, &value.bits, 1);
                return decoded;
            } else if constexpr (std::is_same_v<T, Bool8>) {
                return value.value != 0;
            } else {
                return static_cast<double>(value);
            }
        });
    }

    void set(size_t index, double value) {
        check(index);
        visit([&](auto& values) {
            using T = typename std::decay_t<decltype(values)>::value_type;
            if constexpr (std::is_same_v<T, Float16>) {
                const float source = static_cast<float>(value);
                HardwareOpt::fp32_to_fp16_f16c(&values[index].bits, &source, 1);
            } else if constexpr (std::is_same_v<T, BFloat16>) {
                const float source = static_cast<float>(value);
                HardwareOpt::fp32_to_bf16(&values[index].bits, &source, 1);
            } else if constexpr (std::is_same_v<T, Bool8>) {
                values[index].value = value != 0.0;
            } else if constexpr (std::is_integral_v<T>) {
                values[index] = castInteger<T>(value);
            } else {
                values[index] = static_cast<T>(value);
            }
        });
    }

    std::vector<float> toFloat32() const {
        if (dtype_ == DType::F32) return values<float>();
        std::vector<float> result(numel_);
        for (size_t index = 0; index < numel_; ++index)
            result[index] = static_cast<float>(get(index));
        return result;
    }

    static TypedTensor fromFloat32(const std::vector<float>& values,
                                   const std::vector<int>& shape, DType dtype) {
        if (dtype == DType::F32) {
            // Construct the owned F32 vector directly: allocating a zero-filled
            // tensor first would touch every byte twice before any computation.
            TypedTensor result;
            result.shape_ = shape;
            result.dtype_ = dtype;
            result.numel_ = 1;
            for (int dimension : shape) {
                if (dimension < 0) throw std::invalid_argument("TypedTensor: negative dimension");
                result.numel_ *= static_cast<size_t>(dimension);
            }
            if (result.numel_ != values.size())
                throw std::invalid_argument("TypedTensor::fromFloat32: shape mismatch");
            result.storage_ = values;
            return result;
        }
        TypedTensor result(shape, dtype);
        if (result.numel() != values.size())
            throw std::invalid_argument("TypedTensor::fromFloat32: shape mismatch");
        for (size_t index = 0; index < values.size(); ++index) result.set(index, values[index]);
        return result;
    }

private:
    static TensorStorage makeStorage(DType dtype, size_t size) {
        switch (dtype) {
            case DType::BOOL: return std::vector<Bool8>(size);
            case DType::U8: return std::vector<uint8_t>(size);
            case DType::I8: return std::vector<int8_t>(size);
            case DType::U16: return std::vector<uint16_t>(size);
            case DType::I16: return std::vector<int16_t>(size);
            case DType::U32: return std::vector<uint32_t>(size);
            case DType::I32: return std::vector<int32_t>(size);
            case DType::U64: return std::vector<uint64_t>(size);
            case DType::I64: return std::vector<int64_t>(size);
            case DType::F16: return std::vector<Float16>(size);
            case DType::BF16: return std::vector<BFloat16>(size);
            case DType::F32: return std::vector<float>(size);
            case DType::F64: return std::vector<double>(size);
            default: throw std::invalid_argument("TypedTensor: unknown dtype");
        }
    }

    template <typename T> static T castInteger(double value) {
        if (!std::isfinite(value)) throw std::invalid_argument("TypedTensor: non-finite integer value");
        const long double rounded = std::nearbyint(static_cast<long double>(value));
        const long double low = static_cast<long double>(std::numeric_limits<T>::lowest());
        const long double high = static_cast<long double>(std::numeric_limits<T>::max());
        return static_cast<T>(std::clamp(rounded, low, high));
    }
    void check(size_t index) const {
        if (index >= numel_) throw std::out_of_range("TypedTensor: index out of range");
    }

    std::vector<int> shape_;
    DType dtype_ = DType::UNKNOWN;
    size_t numel_ = 0;
    TensorStorage storage_ = std::vector<float>{};
};

} // namespace Mimir
