#pragma once

#include "Layers.hpp"
#include <climits>
#include <cmath>
#include <vector>

// Backend-independent scheduling of native kernels. No CPU layer dispatcher is
// used here; transposition only packs matrix layouts for the device kernels.
namespace NativeBackward {
inline int unaryOperation(LayerType type) {
    switch (type) {
        case LayerType::ReLU: return 0;
        case LayerType::LeakyReLU: return 1;
        case LayerType::Sigmoid: return 2;
        case LayerType::Tanh: return 3;
        case LayerType::SiLU: return 4;
        case LayerType::GELU: return 5;
        case LayerType::Softplus: return 6;
        case LayerType::Mish: return 7;
        case LayerType::HardSigmoid: return 8;
        case LayerType::HardSwish: return 9;
        default: return -1;
    }
}
inline bool supports(LayerType type) {
    return unaryOperation(type) >= 0 || type == LayerType::Linear ||
        type == LayerType::MatMul || type == LayerType::BatchMatMul ||
        type == LayerType::Add || type == LayerType::Subtract ||
        type == LayerType::Multiply || type == LayerType::Divide;
}
inline std::vector<float> transpose(const float* x, int rows, int columns) {
    std::vector<float> out(static_cast<size_t>(rows) * columns);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < columns; ++c)
            out[static_cast<size_t>(c)*rows+r] = x[static_cast<size_t>(r)*columns+c];
    return out;
}
inline bool finite(const std::vector<float>& x) {
    for (float value : x) if (!std::isfinite(value)) return false;
    return true;
}

template<class Engine>
bool execute(Engine& engine, const std::vector<const std::vector<float>*>& inputs,
             const std::vector<const std::vector<float>*>& grad_outputs,
             std::vector<std::vector<float>>& grad_inputs, Layer& layer) {
    if (inputs.empty() || !inputs[0] || grad_outputs.size()!=1 || !grad_outputs[0]) return false;
    const auto& x = *inputs[0];
    const auto& go = *grad_outputs[0];
    if (x.size() > INT_MAX || go.size() > INT_MAX) return false;
    const auto type = layer.type_enum;
    const int unary = unaryOperation(type);
    std::vector<std::vector<float>> result;
    std::vector<float> dw, db;
    if (unary >= 0) {
        if (go.size()!=x.size()) return false;
        result.resize(1); result[0].resize(x.size());
        if (!x.empty() && !engine.binaryForward(x.data(),go.data(),result[0].data(),
                static_cast<int>(x.size()),20+unary,
                layer.leaky_relu_alpha > 0 ? layer.leaky_relu_alpha : 0.01f)) return false;
    } else if (type == LayerType::Linear) {
        const int in = layer.in_features, out = layer.out_features;
        if (in <= 0 || out <= 0 || x.empty() || x.size()%in != 0) return false;
        const int batch = static_cast<int>(x.size()/in);
        const size_t nw = static_cast<size_t>(in)*out;
        if (go.size()!=static_cast<size_t>(batch)*out || !layer.getWeights() ||
            layer.getWeightsSize()<nw+(layer.use_bias ? out : 0)) return false;
        result.resize(1); result[0].resize(x.size());
        dw.resize(nw);
        auto gt = transpose(go.data(),batch,out);
        if (!engine.matmulForward(go.data(),layer.getWeights(),result[0].data(),batch,out,in) ||
            !engine.matmulForward(gt.data(),x.data(),dw.data(),out,batch,in)) return false;
        if (layer.use_bias) {
            std::vector<float> ones(batch,1.0f);
            db.resize(out);
            if (!engine.matmulForward(ones.data(),go.data(),db.data(),1,batch,out)) return false;
        }
    } else if (type == LayerType::MatMul || type == LayerType::BatchMatMul) {
        if (inputs.size()!=2 || !inputs[1]) return false;
        const int m=layer.in_features, k=layer.out_features, n=layer.embed_dim;
        const int batches=type==LayerType::BatchMatMul ? layer.seq_len : 1;
        if (m<=0 || k<=0 || n<=0 || batches<=0) return false;
        const auto& b=*inputs[1];
        const size_t as=static_cast<size_t>(m)*k, bs=static_cast<size_t>(k)*n, gs=static_cast<size_t>(m)*n;
        if (x.size()!=as*batches || b.size()!=bs*batches || go.size()!=gs*batches) return false;
        result={std::vector<float>(x.size()),std::vector<float>(b.size())};
        for (int batch=0;batch<batches;++batch) {
            const float* a=x.data()+batch*as; const float* bv=b.data()+batch*bs; const float* g=go.data()+batch*gs;
            auto at=transpose(a,m,k), bt=transpose(bv,k,n);
            if (!engine.matmulForward(g,bt.data(),result[0].data()+batch*as,m,n,k) ||
                !engine.matmulForward(at.data(),g,result[1].data()+batch*bs,k,m,n)) return false;
        }
    } else {
        if (inputs.size()!=2 || !inputs[1] || inputs[1]->size()!=x.size() || go.size()!=x.size()) return false;
        const auto& b=*inputs[1];
        const int count=static_cast<int>(x.size());
        result={std::vector<float>(x.size()),std::vector<float>(x.size())};
        if (!count) { grad_inputs=std::move(result); return true; }
        if (type==LayerType::Add || type==LayerType::Subtract) {
            std::vector<float> factor(x.size(),1.0f);
            if (!engine.binaryForward(go.data(),factor.data(),result[0].data(),count,2)) return false;
            if (type==LayerType::Subtract) std::fill(factor.begin(),factor.end(),-1.0f);
            if (!engine.binaryForward(go.data(),factor.data(),result[1].data(),count,2)) return false;
        } else if (type==LayerType::Multiply) {
            if (!engine.binaryForward(go.data(),b.data(),result[0].data(),count,2) ||
                !engine.binaryForward(go.data(),x.data(),result[1].data(),count,2)) return false;
        } else if (type==LayerType::Divide) {
            std::vector<float> product(x.size());
            if (!engine.binaryForward(go.data(),b.data(),result[0].data(),count,3) ||
                !engine.binaryForward(go.data(),x.data(),product.data(),count,2) ||
                !engine.binaryForward(product.data(),b.data(),result[1].data(),count,4)) return false;
        } else return false;
    }
    for (const auto& gradient : result) if (!finite(gradient)) return false;
    if (!finite(dw) || !finite(db)) return false;
    // Commit once, after every device operation succeeds. A failed device can
    // be retried on another backend without double-accumulating parameters.
    if (!dw.empty()) {
        if (layer.grad_weights.size()!=layer.getWeightsSize()) layer.grad_weights.assign(layer.getWeightsSize(),0.0f);
        for (size_t i=0;i<dw.size();++i) layer.grad_weights[i]+=dw[i];
        if (!db.empty()) {
            if (layer.grad_bias.size()!=db.size()) layer.grad_bias.assign(db.size(),0.0f);
            for (size_t i=0;i<db.size();++i) {
                layer.grad_weights[dw.size()+i]+=db[i];
                layer.grad_bias[i]+=db[i];
            }
        }
    }
    grad_inputs=std::move(result);
    return true;
}
} // namespace NativeBackward
