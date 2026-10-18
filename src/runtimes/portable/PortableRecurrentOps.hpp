#pragma once
#include "PortableNativeOps.hpp"
namespace PortableNativeOps {
using namespace PortableStaging;
// Compatibility reference kernels: recurrent time/order dependencies execute
// serially on device. This is Native coverage, not an optimized recurrent engine.
// Separate forward/backward formulas intentionally follow the existing CPU
// contracts, including their legacy gate/state update ordering.
inline bool recurrent(const std::vector<float>& x,std::vector<float>& y,const Layer& l,const CudaOperationGeometry::Recurrent& g,
    const std::vector<float>* dy=nullptr,std::vector<float>* dw=nullptr){
    Buffer bx,bw,bg,by,bdw,workspace;int* status=nullptr;
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !bw.alloc(l.getWeightsSize()) || !bw.upload(l.getWeights(),l.getWeightsSize()) || !workspace.alloc(g.workspace))return false;
    y.resize(dy?x.size():static_cast<size_t>(g.steps)*g.hidden);if(!by.alloc(y.size()))return false;
    if(dy && (!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size()) || !bdw.alloc(l.getWeightsSize()) || cudaMemset(bdw.ptr,0,l.getWeightsSize()*sizeof(float))!=cudaSuccess))return false;
    if(cudaMalloc(&status,sizeof(int))!=cudaSuccess)return false;
    launch("recurrent_kernel", 1,1, bx.ptr,bw.ptr,bg.ptr,by.ptr,bdw.ptr,workspace.ptr,status,l.type_enum,g,l.use_bias,l.getWeightsSize());
    int ok=0;const bool done=finished() && cudaMemcpy(&ok,status,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(status);
    if(!done || !ok || !by.download(y.data(),y.size()))return false;
    if(dy){dw->resize(l.getWeightsSize());return bdw.download(dw->data(),dw->size());}return true;
}
} // namespace PortableNativeOps
