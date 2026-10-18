#pragma once
#include "PortableNativeOps.hpp"
namespace PortableNativeOps {
using namespace PortableStaging;

// Sequential greedy selection preserves CPU score ordering and lowest-index ties.
// Filtering, comparisons, class-aware IoU and suppression run entirely on device.

inline bool nms(const std::vector<const std::vector<float>*>& in,std::vector<std::vector<float>>& out,const Layer& l){
    const size_t n=in[1]->size();out={std::vector<float>()};if(!n)return true;
    Buffer boxes,scores,classes,visited,result;int* count=nullptr;
    if(!boxes.alloc(in[0]->size()) || !boxes.upload(in[0]->data(),in[0]->size()) || !scores.alloc(n) || !scores.upload(in[1]->data(),n) || !visited.alloc(n) || !result.alloc(n))return false;
    if(in.size()>2 && in[2] && (!classes.alloc(n) || !classes.upload(in[2]->data(),n)))return false;
    if(cudaMalloc(&count,sizeof(int))!=cudaSuccess)return false;
    launch("nms_kernel", 1,1, boxes.ptr,scores.ptr,classes.ptr,visited.ptr,result.ptr,count,n,l.nms_score_threshold,l.nms_iou_threshold,l.nms_max_detections,l.nms_class_agnostic);
    int used=0;const bool ok=finished() && cudaMemcpy(&used,count,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(count);
    if(!ok || used<0 || static_cast<size_t>(used)>n)return false;
    out[0].resize(used);return used==0 || result.download(out[0].data(),used);
}
// Hard NMS returns discrete indices. Its explicit backward is stop-gradient;
// selection boundaries have no ordinary derivative (no surrogate is introduced).

inline bool nmsBackward(const std::vector<const std::vector<float>*>& in,
    const std::vector<const std::vector<float>*>& go,std::vector<std::vector<float>>& gi,const Layer& layer){
    std::vector<std::vector<float>> selected;
    if(go.size()!=1 || !go[0] || !nms(in,selected,layer) || go[0]->size()!=selected[0].size())return false;
    std::vector<std::vector<float>> result(in.size());
    for(size_t i=0;i<in.size();++i){
        if(!in[i] || in[i]->empty())continue;
        const size_t count=in[i]->size();if(count>static_cast<size_t>(INT_MAX))return false;
        Buffer gradient;if(!gradient.alloc(count))return false;
        launch("nms_stop_gradient_kernel", static_cast<int>((count+255)/256),256, gradient.ptr,static_cast<int>(count));
        result[i].resize(count);if(!finished() || !gradient.download(result[i].data(),count))return false;
    }
    gi=std::move(result);return true;
}
} // namespace PortableNativeOps
