#pragma once
#include "CudaNativeOps.cuh"
namespace CudaNativeOps {
__device__ inline float box_iou(const float* boxes,int a,int b){
    a*=4;b*=4;
    const float aa=fmaxf(0.f,boxes[a+2]-boxes[a])*fmaxf(0.f,boxes[a+3]-boxes[a+1]);
    const float ab=fmaxf(0.f,boxes[b+2]-boxes[b])*fmaxf(0.f,boxes[b+3]-boxes[b+1]);
    const float iw=fmaxf(0.f,fminf(boxes[a+2],boxes[b+2])-fmaxf(boxes[a],boxes[b]));
    const float ih=fmaxf(0.f,fminf(boxes[a+3],boxes[b+3])-fmaxf(boxes[a+1],boxes[b+1]));
    const float intersection=iw*ih,area=aa+ab-intersection;return area>0?intersection/area:0.f;
}
// Sequential greedy selection preserves CPU score ordering and lowest-index ties.
// Filtering, comparisons, class-aware IoU and suppression run entirely on device.
__global__ void nms_kernel(const float* boxes,const float* scores,const float* classes,float* visited,float* output,int* count,
    int n,float score_threshold,float iou_threshold,int maximum,bool agnostic){
    if(blockIdx.x || threadIdx.x)return;*count=0;
    for(int i=0;i<n;++i){bool valid=isfinite(scores[i]) && scores[i]>=score_threshold;
        for(int d=0;d<4;++d)valid=valid && isfinite(boxes[i*4+d]);if(classes)valid=valid && isfinite(classes[i]);visited[i]=valid?0.f:1.f;}
    for(int pass=0;pass<n;++pass){
        int best=-1;for(int i=0;i<n;++i)if(!visited[i] && (best<0 || scores[i]>scores[best]))best=i;
        if(best<0)break;visited[best]=1;bool suppress=false;
        const int cls=classes?static_cast<int>(llround(static_cast<double>(classes[best]))):0;
        for(int j=0;j<*count;++j){const int accepted=static_cast<int>(output[j]);
            const bool same=agnostic || !classes || cls==static_cast<int>(llround(static_cast<double>(classes[accepted])));
            if(same && box_iou(boxes,best,accepted)>iou_threshold){suppress=true;break;}}
        if(suppress)continue;output[(*count)++]=static_cast<float>(best);if(maximum>0 && *count>=maximum)break;
    }
}
inline bool nms(const std::vector<const std::vector<float>*>& in,std::vector<std::vector<float>>& out,const Layer& l){
    const size_t n=in[1]->size();out={std::vector<float>()};if(!n)return true;
    Buffer boxes,scores,classes,visited,result;int* count=nullptr;
    if(!boxes.alloc(in[0]->size()) || !boxes.upload(in[0]->data(),in[0]->size()) || !scores.alloc(n) || !scores.upload(in[1]->data(),n) || !visited.alloc(n) || !result.alloc(n))return false;
    if(in.size()>2 && in[2] && (!classes.alloc(n) || !classes.upload(in[2]->data(),n)))return false;
    if(cudaMalloc(&count,sizeof(int))!=cudaSuccess)return false;
    nms_kernel<<<1,1>>>(boxes.ptr,scores.ptr,classes.ptr,visited.ptr,result.ptr,count,n,l.nms_score_threshold,l.nms_iou_threshold,l.nms_max_detections,l.nms_class_agnostic);
    int used=0;const bool ok=finished() && cudaMemcpy(&used,count,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(count);
    if(!ok || used<0 || static_cast<size_t>(used)>n)return false;
    out[0].resize(used);return used==0 || result.download(out[0].data(),used);
}
// Hard NMS returns discrete indices. Its explicit backward is stop-gradient;
// selection boundaries have no ordinary derivative (no surrogate is introduced).
__global__ void nms_stop_gradient_kernel(float* gradient,int count){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x)gradient[i]=0.f;
}
inline bool nmsBackward(const std::vector<const std::vector<float>*>& in,
    const std::vector<const std::vector<float>*>& go,std::vector<std::vector<float>>& gi,const Layer& layer){
    std::vector<std::vector<float>> selected;
    if(go.size()!=1 || !go[0] || !nms(in,selected,layer) || go[0]->size()!=selected[0].size())return false;
    std::vector<std::vector<float>> result(in.size());
    for(size_t i=0;i<in.size();++i){
        if(!in[i] || in[i]->empty())continue;
        const size_t count=in[i]->size();if(count>static_cast<size_t>(INT_MAX))return false;
        Buffer gradient;if(!gradient.alloc(count))return false;
        nms_stop_gradient_kernel<<<static_cast<int>((count+255)/256),256>>>(gradient.ptr,static_cast<int>(count));
        result[i].resize(count);if(!finished() || !gradient.download(result[i].data(),count))return false;
    }
    gi=std::move(result);return true;
}
} // namespace CudaNativeOps
