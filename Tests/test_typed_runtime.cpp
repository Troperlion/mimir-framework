#include <cstring>
#include <limits>
#include "Autograd.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/cpu/TypedKernels.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    // F32 conversion is an exact, independent bulk copy (including special bits).
    {
        const std::vector<float> source = {0.0f, -0.0f, 1.25f,
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN()};
        auto tensor = Mimir::TypedTensor::fromFloat32(source, {5}, Mimir::DType::F32);
        const auto restored = tensor.toFloat32();
        if (std::memcmp(source.data(), restored.data(), source.size() * sizeof(float)))
            throw std::runtime_error("F32 round trip changed scalar bits");
        tensor.values<float>()[2] = 7.0f;
        if (source[2] != 1.25f || restored[2] != 1.25f)
            throw std::runtime_error("F32 conversion aliases its source");
    }

    {
        auto empty = Mimir::TypedTensor::fromFloat32({}, {2,0,3}, Mimir::DType::F32);
        if (empty.numel() != 0 || !empty.toFloat32().empty()) return 8;
        auto scalar = Mimir::TypedTensor::fromFloat32({2.5f}, {}, Mimir::DType::F32);
        if (scalar.numel() != 1 || scalar.get(0) != 2.5) return 9;
        for (const auto& shape : {std::vector<int>{-1}, std::vector<int>{2}}) {
            bool rejected = false;
            try { (void)Mimir::TypedTensor::fromFloat32({1.0f}, shape, Mimir::DType::F32); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) return 10;
        }
    }

    // Bulk conversions must match scalar set/get, including SIMD tails, empty
    // tensors and non-finite floating values; integer rounding stays unchanged.
    for (auto dtype : {Mimir::DType::F16, Mimir::DType::BF16, Mimir::DType::F64,
                       Mimir::DType::I8, Mimir::DType::U64, Mimir::DType::BOOL}) {
        for (int count : {0, 1, 7, 8, 255, 256, 257, 521}) {
            std::vector<float> source(count);
            for (int i = 0; i < count; ++i) source[i] = float(i % 101 - 50) * 0.125f;
            if (Mimir::dtype_is_floating(dtype) && count > 7) {
                source[0] = -0.0f;
                source[1] = std::numeric_limits<float>::infinity();
                source[2] = -std::numeric_limits<float>::infinity();
                source[3] = std::numeric_limits<float>::quiet_NaN();
                source[4] = std::numeric_limits<float>::denorm_min();
            }
            Mimir::TypedTensor scalar({count}, dtype);
            for (int i = 0; i < count; ++i) scalar.set(i, source[i]);
            auto bulk = Mimir::TypedTensor::fromFloat32(source, {count}, dtype);
            if (count && std::memcmp(scalar.data(), bulk.data(), scalar.size_bytes()))
                throw std::runtime_error("Bulk conversion differs from scalar storage");
            auto restored = bulk.toFloat32();
            for (int i = 0; i < count; ++i) {
                float expected = static_cast<float>(scalar.get(i));
                if (std::memcmp(&expected, &restored[i], sizeof(float)))
                    throw std::runtime_error("Bulk decode differs from scalar get");
            }
        }
    }

    CpuRuntime runtime;
    RuntimeConfig runtime_config;
    runtime_config.backend = "CPU";
    if (!runtime.initialize(runtime_config)) return 1;

    const std::vector<Mimir::DType> dtypes = {
        Mimir::DType::F32, Mimir::DType::F16, Mimir::DType::BF16, Mimir::DType::F64,
        Mimir::DType::I8, Mimir::DType::I16, Mimir::DType::I32, Mimir::DType::I64,
        Mimir::DType::U8, Mimir::DType::U16, Mimir::DType::U32, Mimir::DType::U64,
        Mimir::DType::BOOL,
    };
    for (const auto dtype : dtypes) {
        Mimir::TypedTensor input({1, 2}, dtype);
        Mimir::TypedTensor weights({1, 2}, dtype);
        Mimir::TypedTensor output({1, 1}, dtype);
        input.set(0, 1); input.set(1, 2);
        weights.set(0, 2); weights.set(1, 3);
        if (!runtime.linearForwardTyped(input, weights, nullptr, output)) return 2;
        const double expected = dtype == Mimir::DType::BOOL ? 1.0 : 8.0;
        if (std::fabs(output.get(0) - expected) > 0.05) {
            std::cerr << Mimir::dtype_to_string(dtype) << " output=" << output.get(0) << '\n';
            return 3;
        }
    }

    Mimir::TypedTensor pred({2}, Mimir::DType::F16);
    Mimir::TypedTensor target({2}, Mimir::DType::F16);
    pred.set(0, 2); pred.set(1, 4);
    target.set(0, 1); target.set(1, 2);
    auto gradient = Autograd::mse_backward(pred, target);
    if (gradient.dtype() != Mimir::DType::F32 ||
        std::fabs(gradient.get(0) - 1.0) > 1e-6 ||
        std::fabs(gradient.get(1) - 2.0) > 1e-6) return 4;

    bool integer_rejected = false;
    try {
        Mimir::TypedTensor integers({1}, Mimir::DType::I32);
        (void)Autograd::mse_backward(integers, integers);
    } catch (const std::invalid_argument&) {
        integer_rejected = true;
    }
    if (!integer_rejected) return 5;

    for (const auto dtype : {Mimir::DType::F16, Mimir::DType::BF16,
                             Mimir::DType::F32, Mimir::DType::F64}) {
        Mimir::TypedTensor x({2, 2}, dtype), gamma({2}, dtype), normalized({2, 2}, dtype);
        x.set(0, 1); x.set(1, 3); x.set(2, -2); x.set(3, 4);
        gamma.set(0, 1); gamma.set(1, 1);
        Mimir::TypedKernels::rmsNorm(x, gamma, normalized, 2, 1e-5);
        Mimir::TypedTensor activated({2, 2}, dtype);
        Mimir::TypedKernels::unary(
            normalized, activated, Mimir::TypedKernels::UnaryOp::SiLU);
        for (size_t i = 0; i < activated.numel(); ++i)
            if (!std::isfinite(activated.get(i))) return 6;
    }

    Mimir::TypedTensor ids({2}, Mimir::DType::I32);
    Mimir::TypedTensor table({3, 2}, Mimir::DType::F16);
    Mimir::TypedTensor embedded({2, 2}, Mimir::DType::F16);
    ids.set(0, 1); ids.set(1, 2);
    for (size_t i = 0; i < table.numel(); ++i) table.set(i, static_cast<double>(i));
    Mimir::TypedKernels::embedding(ids, table, embedded, 3, 2, -1);
    if (std::fabs(embedded.get(0) - 2.0) > 0.01 ||
        std::fabs(embedded.get(3) - 5.0) > 0.01) return 7;

    std::cout << "PASS typed CPU runtime and autograd dtypes\n";
    return 0;
}
