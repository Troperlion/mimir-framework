#pragma once
#include "PortableNativeOps.hpp"
#include "../ops_sequence_constant.hpp"

namespace PortableConstantOps {
inline bool fits(const Layer& layer) {
    const size_t weights=layer.getWeightsSize(),output=SequenceConstant::outputSize(layer);
    constexpr size_t maxElements=8*1024*1024;
    if (!layer.getWeights() || !weights || !output ||
        weights>=maxElements) return false;
    if (!SequenceConstant::enabled(layer)) return true;
    const size_t rows=static_cast<size_t>(layer.seq_len);
    return rows<=65536 && output<maxElements-weights &&
        2*rows<maxElements-weights-output;
}
inline bool execute(const Layer& layer,std::vector<float>& output,const std::vector<float>* gradient=nullptr) {
    using namespace PortableStaging;
    using PortableNativeOps::Buffer;
    if (!fits(layer)) return false;
    const size_t count=SequenceConstant::outputSize(layer),weights=layer.getWeightsSize();
    if (gradient && gradient->size()!=count) return false;
    if (!SequenceConstant::enabled(layer)) {
        Buffer values;
        if (!values.alloc(weights) || !values.upload(gradient?gradient->data():layer.getWeights(),weights)) return false;
        launch("portable_copy",1,1,values.ptr,values.ptr,static_cast<int>(weights));
        output.resize(weights);
        return PortableNativeOps::finished() && values.download(output.data(),weights);
    }
    // Only row indices and interpolation coordinates are prepared on the host.
    // Interpolation and its adjoint operate on device-resident tensor values.
    std::vector<float> rows(2*static_cast<size_t>(layer.seq_len));
    SequenceConstant::rows(layer,[&](size_t row,size_t first,size_t,float blend) {
        rows[2*row]=static_cast<float>(first);rows[2*row+1]=blend;
    });
    Buffer source,mapping,result;
    const size_t input_count=gradient?count:weights,result_count=gradient?weights:count;
    if (!source.alloc(input_count) || !source.upload(gradient?gradient->data():layer.getWeights(),input_count) ||
        !mapping.alloc(rows.size()) || !mapping.upload(rows.data(),rows.size()) || !result.alloc(result_count)) return false;
    if (gradient && cudaMemset(result.ptr,0,result_count*sizeof(float))!=cudaSuccess) return false;
    launch(gradient?"portable_sequence_constant_backward":"portable_sequence_constant_forward",
        static_cast<int>(count),1,source.ptr,mapping.ptr,result.ptr,layer.embed_dim,
        static_cast<int>(weights/static_cast<size_t>(layer.embed_dim)),static_cast<int>(count));
    output.resize(result_count);
    return PortableNativeOps::finished() && result.download(output.data(),result_count);
}
}
