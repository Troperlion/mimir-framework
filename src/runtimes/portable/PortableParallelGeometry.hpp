#pragma once
#include "PortableLayerGeometry.hpp"
#include "runtimes/cuda/CudaOperationGeometry.hpp"
#include "runtimes/LayerOps.hpp"

namespace PortableParallelGeometry {
// Match the parallel kernels' reduction and arena limits. Intrinsically serial
// recurrent/NMS algorithms retain their reference budget and fallback behavior.
inline bool fits(const Layer& layer,const std::vector<const std::vector<float>*>& inputs,bool backward) {
    if(inputs.empty() || !inputs[0] || inputs[0]->empty() || inputs[0]->size()>16*1024*1024)return false;
    int op=0;float alpha=0;
    if(RuntimeLayerOps::resolveUnaryOp(layer.type_enum,layer,op,alpha))return true;
    if(layer.type_enum==LayerType::LayerNorm || layer.type_enum==LayerType::RMSNorm){
        const int width=PortableLayerGeometry::rowNormWidth(layer,inputs[0]->size(),backward);
        return width>0 && width<=1024*1024 && inputs[0]->size()/width<=1024*1024 && layer.getWeightsSize()<=16*1024*1024;
    }
    if(layer.type_enum==LayerType::Linear){
        return layer.in_features>0 && layer.in_features<=4096 && layer.out_features>0 && layer.out_features<=4096 && inputs[0]->size()%layer.in_features==0 && inputs[0]->size()/layer.in_features<=4096 && layer.getWeightsSize()<=16*1024*1024;
    }
    // Geometry validation remains in PortableRuntime. These families now use
    // parallel kernels (CAS for scatter), while sequential algorithms stay bounded.
    if(layer.type_enum==LayerType::NMS || CudaOperationGeometry::recurrentType(layer.type_enum))return false;
    size_t total=layer.getWeightsSize();if(total>8*1024*1024)return false;
    for(const auto* input:inputs){if(input){if(input->size()>8*1024*1024-total)return false;total+=input->size();}}
    return total<=8*1024*1024;
}
}
