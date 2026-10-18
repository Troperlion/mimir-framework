#pragma once

// Device implementations only. No CPU kernels or CPU runtime changes.
#include "PortableStaging.hpp"
#include "runtimes/cuda/CudaOperationGeometry.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <vector>

namespace PortableNativeOps {
using namespace PortableStaging;
constexpr int threads = 1;
struct Buffer {
    float* ptr = nullptr;
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    bool alloc(size_t count) { return count && cudaMalloc(&ptr, count * sizeof(float)) == cudaSuccess; }
    bool upload(const float* data, size_t count) {
        return cudaMemcpy(ptr, data, count * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    bool download(float* data, size_t count) {
        return cudaMemcpy(data, ptr, count * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    }
};
inline bool finished() {
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}



// One block per group: two-pass variance avoids cancellation for large means.



inline bool groupnorm(const std::vector<float>& x, const float* w, size_t weight_count,
                      int channels, int spatial, int groups, float eps, bool bias,
                      std::vector<float>& output, const std::vector<float>* dy = nullptr,
                      std::vector<float>* dw = nullptr, std::vector<float>* db = nullptr) {
    Buffer input, weights, result, grad, normalized, invstd, gamma_grad, bias_grad;
    const size_t count = x.size();
    if (!input.alloc(count) || !result.alloc(count) || !input.upload(x.data(), count)) return false;
    if (w && (!weights.alloc(weight_count) || !weights.upload(w, weight_count))) return false;
    const int group_size = channels / groups * spatial;
    if (!dy) {
        launch("norm_forward", groups, threads, input.ptr, weights.ptr, result.ptr, nullptr, nullptr,
                                         group_size, spatial, channels, eps, bias);
    } else {
        if (!grad.alloc(count) || !normalized.alloc(count) || !invstd.alloc(groups) ||
            !grad.upload(dy->data(), count)) return false;
        launch("norm_forward", groups, threads, input.ptr, weights.ptr, result.ptr, normalized.ptr, invstd.ptr,
                                         group_size, spatial, channels, eps, bias);
        launch("norm_backward", groups, threads, normalized.ptr, invstd.ptr, grad.ptr, weights.ptr, result.ptr,
                                          group_size, spatial, channels);
        if (w) {
            if (!gamma_grad.alloc(channels) || (bias && !bias_grad.alloc(channels))) return false;
            launch("channel_gradient", channels, threads, grad.ptr, normalized.ptr, gamma_grad.ptr, bias_grad.ptr, spatial);
        }
    }
    if (!finished()) return false;
    std::vector<float> staged(count);
    if (!result.download(staged.data(), count)) return false;
    if (dy && w) {
        dw->resize(channels);
        if (!gamma_grad.download(dw->data(), channels)) return false;
        if (bias) { db->resize(channels); if (!bias_grad.download(db->data(), channels)) return false; }
    }
    output = std::move(staged);
    return true;
}




inline bool resize(const std::vector<float>& source, std::vector<float>& output, int c,
                   int ih, int iw, int oh, int ow, int sh, int sw, bool bilinear, bool backward) {
    Buffer input, result;
    const size_t in_count = static_cast<size_t>(c)*ih*iw, out_count = static_cast<size_t>(c)*oh*ow;
    const size_t count = backward ? in_count : out_count;
    if (!input.alloc(source.size()) || !result.alloc(count) || !input.upload(source.data(), source.size())) return false;
    if (!backward) launch("resize_forward", (out_count+threads-1)/threads, threads, input.ptr, result.ptr, out_count, ih, iw, oh, ow, sh, sw, bilinear);
    else if (!bilinear) launch("nearest_backward", (in_count+threads-1)/threads, threads, input.ptr, result.ptr, in_count, ih, iw, sh, sw);
    else {
        if (cudaMemset(result.ptr, 0, count*sizeof(float)) != cudaSuccess) return false;
        launch("bilinear_backward", (out_count+threads-1)/threads, threads, input.ptr, result.ptr, out_count, ih, iw, oh, ow);
    }
    if (!finished()) return false;
    std::vector<float> staged(count);
    if (!result.download(staged.data(), count)) return false;
    output = std::move(staged); return true;
}

// Arbitrary axis: outer x axis_size x inner, including non-contiguous reductions.


inline bool softmax(const std::vector<float>& x, std::vector<float>& output, int axis_size,
                    int inner, bool logarithmic, const std::vector<float>* dy = nullptr) {
    Buffer input, result, gradient;
    if (!input.alloc(x.size()) || !result.alloc(x.size()) || !input.upload(x.data(),x.size())) return false;
    if (dy && (!gradient.alloc(x.size()) || !gradient.upload(dy->data(),dy->size()))) return false;
    const int rows = x.size() / (static_cast<size_t>(axis_size)*inner);
    launch("softmax_kernel", rows*inner, threads, input.ptr, gradient.ptr, result.ptr, axis_size, inner, logarithmic);
    if (!finished()) return false;
    std::vector<float> staged(x.size());
    if (!result.download(staged.data(),x.size())) return false;
    output=std::move(staged); return true;
}



inline bool conv_backward(cublasHandle_t handle, const std::vector<float>& x,
                          const std::vector<float>& dy, const float* w, int ic, int oc,
                          int h, int width, int k, int stride, int pad, int dilation,
                          int oh, int ow, bool bias, std::vector<float>& dx,
                          std::vector<float>& dw, std::vector<float>& db) {
    const int spatial=oh*ow, inner=ic*k*k;
    const size_t col_count=static_cast<size_t>(inner)*spatial, weight_count=static_cast<size_t>(oc)*inner;
    Buffer input, grad, weights, col, dcol, result, weight_grad, bias_grad;
    if (!input.alloc(x.size()) || !grad.alloc(dy.size()) || !weights.alloc(weight_count) ||
        !col.alloc(col_count) || !dcol.alloc(col_count) || !result.alloc(x.size()) || !weight_grad.alloc(weight_count) ||
        !input.upload(x.data(),x.size()) || !grad.upload(dy.data(),dy.size()) || !weights.upload(w,weight_count)) return false;
    launch("im2col", (col_count+threads-1)/threads,threads, input.ptr,col.ptr,col_count,h,width,k,stride,pad,dilation,oh,ow);
    if (cudaGetLastError()!=cudaSuccess) return false;
    const float one=1.f,zero=0.f;
    // dW^T = col * dY^T (K x S times S x O).
    if (cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,inner,oc,spatial,&one,col.ptr,spatial,
                    grad.ptr,spatial,&zero,weight_grad.ptr,inner)!=CUBLAS_STATUS_SUCCESS) return false;
    // dcol^T = dY^T * W (S x O times O x K).
    if (cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,spatial,inner,oc,&one,grad.ptr,spatial,
                    weights.ptr,inner,&zero,dcol.ptr,spatial)!=CUBLAS_STATUS_SUCCESS) return false;
    launch("col2im", (x.size()+threads-1)/threads,threads, dcol.ptr,result.ptr,x.size(),h,width,k,stride,pad,dilation,oh,ow);
    if (bias) {
        if (!bias_grad.alloc(oc)) return false;
        launch("channel_gradient", oc,threads, grad.ptr,nullptr,nullptr,bias_grad.ptr,spatial);
    }
    if (!finished()) return false;
    dx.resize(x.size()); dw.resize(weight_count);
    if (!result.download(dx.data(),dx.size()) || !weight_grad.download(dw.data(),dw.size())) return false;
    if (bias) { db.resize(oc); if (!bias_grad.download(db.data(),oc)) return false; }
    return true;
}
// One block per row. Statistics, affine transform and derivatives stay on device.


inline bool row_norm(const std::vector<float>& x,const Layer& l,std::vector<float>& y,const std::vector<float>* dy=nullptr,std::vector<float>* dw=nullptr,std::vector<float>* db=nullptr) {
    const int width=l.in_features>0?l.in_features:x.size();const int rows=x.size()/width;
    const float* w=l.affine?l.getWeights():nullptr;const bool bias=w && l.use_bias && l.getWeightsSize()>=2ULL*width;
    Buffer bx,bw,bg,by,bh,bdw,bdb;
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !by.alloc(x.size()))return false;
    if(w && (!bw.alloc(l.getWeightsSize()) || !bw.upload(w,l.getWeightsSize())))return false;
    if(dy && (!bg.alloc(x.size()) || !bg.upload(dy->data(),x.size()) || !bh.alloc(x.size())))return false;
    launch("row_norm_kernel", rows,threads, bx.ptr,bw.ptr,bg.ptr,by.ptr,bh.ptr,width,l.eps>0?l.eps:1e-5f,l.type_enum==LayerType::RMSNorm,bias && (dy || l.type_enum!=LayerType::RMSNorm));
    if(dy && w){
        if(!bdw.alloc(width) || (bias && !bdb.alloc(width)))return false;
        launch("row_norm_grad", (width+threads-1)/threads,threads, bg.ptr,bh.ptr,bdw.ptr,bdb.ptr,rows,width);
    }
    if(!finished())return false;
    y.resize(x.size());if(!by.download(y.data(),y.size()))return false;
    if(dy && w){dw->resize(width);if(!bdw.download(dw->data(),width))return false;if(bias){db->resize(width);if(!bdb.download(db->data(),width))return false;}}
    return true;
}

inline bool projection(const std::vector<const std::vector<float>*>& in,std::vector<std::vector<float>>& out,const Layer& l,const CudaOperationGeometry::Projection& g,
                       const std::vector<float>* dy=nullptr,std::vector<float>* dw=nullptr) {
    Buffer a,b,w,y,grad,da,db,weights_grad;const bool second=in.size()>1 && in[1] && !in[1]->empty();
    if(!a.alloc(in[0]->size()) || !a.upload(in[0]->data(),in[0]->size()) || !w.alloc(l.getWeightsSize()) || !w.upload(l.getWeights(),l.getWeightsSize()))return false;
    if(second && (!b.alloc(in[1]->size()) || !b.upload(in[1]->data(),in[1]->size())))return false;
    if(dy){
        if(!grad.alloc(dy->size()) || !grad.upload(dy->data(),dy->size()) || !da.alloc(in[0]->size()) || !weights_grad.alloc(l.getWeightsSize()) ||
           cudaMemset(da.ptr,0,in[0]->size()*sizeof(float))!=cudaSuccess || cudaMemset(weights_grad.ptr,0,l.getWeightsSize()*sizeof(float))!=cudaSuccess)return false;
        if(g.mode==4 && (!db.alloc(in[1]->size()) || cudaMemset(db.ptr,0,in[1]->size()*sizeof(float))!=cudaSuccess))return false;
    }else if(!y.alloc(g.output))return false;
    launch("projection_kernel", (g.output+threads-1)/threads,threads, a.ptr,b.ptr,w.ptr,grad.ptr,y.ptr,da.ptr,db.ptr,weights_grad.ptr,g,g.mode==2 && second?in[1]->size():0);
    if(!finished())return false;
    if(dy){
        out={std::vector<float>(in[0]->size())};if(!da.download(out[0].data(),out[0].size()))return false;
        if((g.mode==2 || g.mode==4) && in.size()>1 && in[1]){out.emplace_back(in[1]->size(),0.f);if(g.mode==4 && !db.download(out[1].data(),out[1].size()))return false;}
        dw->resize(l.getWeightsSize());return weights_grad.download(dw->data(),dw->size());
    }
    out={std::vector<float>(g.output)};return y.download(out[0].data(),g.output);
}

inline bool geglu(const std::vector<float>& x,std::vector<float>& y,int width,const std::vector<float>* dy=nullptr){
    Buffer bx,bg,by;const int count=x.size()/2;y.resize(dy?x.size():count);
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !by.alloc(y.size()))return false;
    if(dy && (!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size())))return false;
    launch("geglu_kernel", (count+threads-1)/threads,threads, bx.ptr,bg.ptr,by.ptr,width,count);return finished() && by.download(y.data(),y.size());
}


inline bool bicubic(const std::vector<float>& x,std::vector<float>& y,const CudaOperationGeometry::Resize& g,const std::vector<float>* dy=nullptr){
    Buffer bx,bg,by;y.resize(dy?x.size():g.outputSize());
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !by.alloc(y.size()))return false;
    if(dy && (!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size()) || cudaMemset(by.ptr,0,y.size()*sizeof(float))!=cudaSuccess))return false;
    const int count=g.outputSize();launch("bicubic_kernel", (count+threads-1)/threads,threads, bx.ptr,bg.ptr,by.ptr,g,count);return finished() && by.download(y.data(),y.size());
}
// Parallel pooling: independent windows forward; atomics sum overlapping gradients.

inline bool pool(const std::vector<float>& x,std::vector<float>& y,const CudaOperationGeometry::Pool& g,
                 const std::vector<float>* dy=nullptr) {
    Buffer bx,by,bg;if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()))return false;
    y.resize(dy?x.size():g.outputSize());if(!by.alloc(y.size()))return false;
    if(dy && (!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size()) || cudaMemset(by.ptr,0,y.size()*sizeof(float))!=cudaSuccess))return false;
    const int n=g.outputSize();launch("pool_kernel", (n+threads-1)/threads,threads, bx.ptr,dy?bg.ptr:nullptr,dy?nullptr:by.ptr,dy?by.ptr:nullptr,g,n);
    return finished() && by.download(y.data(),y.size());
}


inline bool movement(const std::vector<const std::vector<float>*>& in,std::vector<std::vector<float>>& out,
                     const Layer& l,const CudaOperationGeometry::Movement& g,bool backward=false) {
    if(g.padding || g.shuffle){
        const auto& x=*in[0];const size_t count=g.shuffle?static_cast<size_t>(g.c)*g.h*g.w:static_cast<size_t>(g.c)*g.oh*g.ow;
        const size_t n=backward?static_cast<size_t>(g.c)*g.h*g.w:count;
        Buffer bx,by;out={std::vector<float>(n)};
        if(!bx.alloc(x.size()) || !by.alloc(n) || !bx.upload(x.data(),x.size()) || cudaMemset(by.ptr,0,n*sizeof(float))!=cudaSuccess)return false;
        launch("spatial_move", (count+threads-1)/threads,threads, bx.ptr,by.ptr,count,g.c,g.h,g.w,g.oh,g.ow,g.ph,g.pw,g.shuffle,g.padding,backward);
        return finished() && by.download(out[0].data(),n);
    }
    if(!g.shape.empty()) {
        const auto& x=*in[0];out={std::vector<float>(x.size())};if(x.empty())return true;
        Buffer bx,by;int *bs=nullptr,*bt=nullptr;
        std::vector<int> shape(g.shape.size()),stride(g.shape.size());
        for(size_t d=0;d<shape.size();++d){shape[d]=g.shape[g.dims[d]];stride[d]=g.strides[g.dims[d]];}
        if(!bx.alloc(x.size()) || !by.alloc(x.size()) || !bx.upload(x.data(),x.size()))return false;
        if(cudaMalloc(&bs,shape.size()*sizeof(int))!=cudaSuccess)return false;
        if(cudaMalloc(&bt,stride.size()*sizeof(int))!=cudaSuccess){cudaFree(bs);return false;}
        bool ok=cudaMemcpy(bs,shape.data(),shape.size()*sizeof(int),cudaMemcpyHostToDevice)==cudaSuccess && cudaMemcpy(bt,stride.data(),stride.size()*sizeof(int),cudaMemcpyHostToDevice)==cudaSuccess;
        if(ok){launch("permute_kernel", (x.size()+threads-1)/threads,threads, bx.ptr,by.ptr,bs,bt,shape.size(),x.size(),backward);ok=finished() && by.download(out[0].data(),x.size());}
        cudaFree(bs);cudaFree(bt);return ok;
    }
    // All copies pass through device memory; empty chunks need no device allocation.
    size_t total=0;for(auto* x:in){if(!x)return false;total+=x->size();}
    if(total>INT_MAX)return false;
    if(!total){out.clear();for(auto n:g.sizes)out.emplace_back(n);return true;}
    Buffer b;if(!b.alloc(total))return false;size_t off=0;
    for(auto* x:in){if(x->size() && cudaMemcpy(b.ptr+off,x->data(),x->size()*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)return false;off+=x->size();}
    launch("portable_copy",1,1,b.ptr,b.ptr,total);if(!finished())return false;
    out.clear();off=0;for(auto n:g.sizes){if(n>total-off)return false;out.emplace_back(n);if(n && cudaMemcpy(out.back().data(),b.ptr+off,n*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)return false;off+=n;}
    return off==total && finished();
}
} // namespace PortableNativeOps
