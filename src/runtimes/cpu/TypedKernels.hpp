#pragma once

#include "TypedTensor.hpp"

#include <cmath>
#include <stdexcept>

namespace Mimir::TypedKernels {

enum class BinaryOp { Add, Subtract, Multiply, Divide };
enum class UnaryOp { ReLU, SiLU, Tanh, Sigmoid };

inline void requireSameLayout(const TypedTensor& a, const TypedTensor& b,
                              const TypedTensor& out) {
    if (a.dtype() != b.dtype() || a.dtype() != out.dtype() ||
        a.shape() != b.shape() || a.shape() != out.shape())
        throw std::invalid_argument("typed kernel: dtype/shape mismatch");
}

inline void binary(const TypedTensor& a, const TypedTensor& b,
                   TypedTensor& out, BinaryOp op) {
    requireSameLayout(a, b, out);
    for (size_t i = 0; i < a.numel(); ++i) {
        const double av = a.get(i), bv = b.get(i);
        double value = 0.0;
        switch (op) {
            case BinaryOp::Add: value = av + bv; break;
            case BinaryOp::Subtract: value = av - bv; break;
            case BinaryOp::Multiply: value = av * bv; break;
            case BinaryOp::Divide:
                if (bv == 0.0) throw std::domain_error("typed divide by zero");
                value = av / bv;
                break;
        }
        out.set(i, value);
    }
}

inline void unary(const TypedTensor& input, TypedTensor& output, UnaryOp op) {
    if (!dtype_is_floating(input.dtype()))
        throw std::invalid_argument("nonlinear activation requires floating dtype");
    if (input.dtype() != output.dtype() || input.shape() != output.shape())
        throw std::invalid_argument("typed unary: dtype/shape mismatch");
    for (size_t i = 0; i < input.numel(); ++i) {
        const double x = input.get(i);
        double value = 0.0;
        switch (op) {
            case UnaryOp::ReLU: value = x > 0.0 ? x : 0.0; break;
            case UnaryOp::SiLU: value = x / (1.0 + std::exp(-x)); break;
            case UnaryOp::Tanh: value = std::tanh(x); break;
            case UnaryOp::Sigmoid: value = 1.0 / (1.0 + std::exp(-x)); break;
        }
        output.set(i, value);
    }
}

inline void rmsNorm(const TypedTensor& input, const TypedTensor& weight,
                    TypedTensor& output, int width, double epsilon) {
    if (!dtype_is_floating(input.dtype()) || input.dtype() != weight.dtype() ||
        input.dtype() != output.dtype() || width <= 0 ||
        input.numel() % static_cast<size_t>(width) != 0 ||
        weight.numel() != static_cast<size_t>(width) ||
        output.shape() != input.shape())
        throw std::invalid_argument("typed RMSNorm: invalid tensors");
    const size_t rows = input.numel() / static_cast<size_t>(width);
    for (size_t row = 0; row < rows; ++row) {
        long double square_sum = 0.0;
        for (int column = 0; column < width; ++column) {
            const double value = input.get(row * width + column);
            square_sum += static_cast<long double>(value) * value;
        }
        const double scale = 1.0 / std::sqrt(
            static_cast<double>(square_sum / width) + epsilon);
        for (int column = 0; column < width; ++column)
            output.set(row * width + column,
                       input.get(row * width + column) * scale * weight.get(column));
    }
}

inline void embedding(const TypedTensor& ids, const TypedTensor& table,
                      TypedTensor& output, int vocabulary, int width, int padding) {
    if (!dtype_is_integral(ids.dtype()) || !dtype_is_floating(table.dtype()) ||
        table.shape() != std::vector<int>({vocabulary, width}) ||
        output.dtype() != table.dtype() ||
        output.numel() != ids.numel() * static_cast<size_t>(width))
        throw std::invalid_argument("typed embedding: invalid tensors");
    for (size_t token = 0; token < ids.numel(); ++token) {
        const int64_t id = static_cast<int64_t>(ids.get(token));
        if (id == padding) continue;
        if (id < 0 || id >= vocabulary) throw std::out_of_range("embedding id");
        for (int column = 0; column < width; ++column)
            output.set(token * width + column, table.get(id * width + column));
    }
}

inline void binaryBackward(const TypedTensor& a, const TypedTensor& b,
                           const TypedTensor& grad_output,
                           TypedTensor& grad_a, TypedTensor& grad_b, BinaryOp op) {
    if (!dtype_is_floating(a.dtype()) || a.dtype() != b.dtype() ||
        grad_a.dtype() != grad_output.dtype() || grad_b.dtype() != grad_output.dtype() ||
        a.shape() != b.shape() || a.shape() != grad_output.shape())
        throw std::invalid_argument("typed binary backward: invalid tensors");
    for (size_t i = 0; i < a.numel(); ++i) {
        const double g = grad_output.get(i);
        switch (op) {
            case BinaryOp::Add: grad_a.set(i, g); grad_b.set(i, g); break;
            case BinaryOp::Subtract: grad_a.set(i, g); grad_b.set(i, -g); break;
            case BinaryOp::Multiply:
                grad_a.set(i, g * b.get(i)); grad_b.set(i, g * a.get(i)); break;
            case BinaryOp::Divide:
                if (b.get(i) == 0.0) throw std::domain_error("typed divide backward by zero");
                grad_a.set(i, g / b.get(i));
                grad_b.set(i, -g * a.get(i) / (b.get(i) * b.get(i)));
                break;
        }
    }
}

inline void unaryBackward(const TypedTensor& input, const TypedTensor& grad_output,
                          TypedTensor& grad_input, UnaryOp op) {
    if (!dtype_is_floating(input.dtype()) ||
        grad_input.dtype() != grad_output.dtype() ||
        input.shape() != grad_output.shape() || input.shape() != grad_input.shape())
        throw std::invalid_argument("typed unary backward: invalid tensors");
    for (size_t i = 0; i < input.numel(); ++i) {
        const double x = input.get(i), g = grad_output.get(i);
        double derivative = 0.0;
        switch (op) {
            case UnaryOp::ReLU: derivative = x > 0.0 ? 1.0 : 0.0; break;
            case UnaryOp::SiLU: {
                const double sigmoid = 1.0 / (1.0 + std::exp(-x));
                derivative = sigmoid * (1.0 + x * (1.0 - sigmoid));
                break;
            }
            case UnaryOp::Tanh: {
                const double value = std::tanh(x);
                derivative = 1.0 - value * value;
                break;
            }
            case UnaryOp::Sigmoid: {
                const double value = 1.0 / (1.0 + std::exp(-x));
                derivative = value * (1.0 - value);
                break;
            }
        }
        grad_input.set(i, g * derivative);
    }
}

} // namespace Mimir::TypedKernels
