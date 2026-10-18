#pragma once

// Device implementations only. No CPU kernels or CPU runtime changes.
#include <cuda_runtime.h>
#include "CudaOperationGeometry.hpp"
#include <cublas_v2.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <vector>

namespace CudaNativeOps {
constexpr int threads = 256;
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
__device__ inline float reduce_sum(float value, float* scratch) {
    const int t = threadIdx.x;
    scratch[t] = value;
    __syncthreads();
    for (int step = threads / 2; step; step /= 2) {
        if (t < step) scratch[t] += scratch[t + step];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}
__device__ inline float reduce_max(float value, float* scratch) {
    const int t = threadIdx.x;
    scratch[t] = value;
    __syncthreads();
    for (int step = threads / 2; step; step /= 2) {
        if (t < step) scratch[t] = fmaxf(scratch[t], scratch[t + step]);
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

// One block per group: two-pass variance avoids cancellation for large means.
__global__ void norm_forward(const float* x, const float* weights, float* y,
                             float* normalized, float* invstd, int group_size,
                             int spatial, int channels, float eps, bool bias) {
    __shared__ float scratch[threads];
    const int base = blockIdx.x * group_size;
    float sum = 0;
    for (int i = threadIdx.x; i < group_size; i += threads) sum += x[base + i];
    const float mean = reduce_sum(sum, scratch) / group_size;
    float variance = 0;
    for (int i = threadIdx.x; i < group_size; i += threads) {
        const float d = x[base + i] - mean;
        variance += d * d;
    }
    const float inv = rsqrtf(reduce_sum(variance, scratch) / group_size + eps);
    if (threadIdx.x == 0 && invstd) invstd[blockIdx.x] = inv;
    for (int i = threadIdx.x; i < group_size; i += threads) {
        const int index = base + i;
        const int channel = (index / spatial) % channels;
        const float value = (x[index] - mean) * inv;
        if (normalized) normalized[index] = value;
        y[index] = weights ? value * weights[channel] + (bias ? weights[channels + channel] : 0.f) : value;
    }
}
__global__ void norm_backward(const float* xhat, const float* invstd, const float* dy,
                              const float* weights, float* dx, int group_size,
                              int spatial, int channels) {
    __shared__ float scratch[threads];
    const int base = blockIdx.x * group_size;
    float sum = 0, product = 0;
    for (int i = threadIdx.x; i < group_size; i += threads) {
        const int index = base + i;
        const float g = dy[index] * (weights ? weights[(index / spatial) % channels] : 1.f);
        sum += g;
        product += g * xhat[index];
    }
    const float s1 = reduce_sum(sum, scratch);
    const float s2 = reduce_sum(product, scratch);
    for (int i = threadIdx.x; i < group_size; i += threads) {
        const int index = base + i;
        const float g = dy[index] * (weights ? weights[(index / spatial) % channels] : 1.f);
        dx[index] = invstd[blockIdx.x] * (g - s1 / group_size - xhat[index] * s2 / group_size);
    }
}
__global__ void channel_gradient(const float* dy, const float* xhat, float* dw, float* db, int spatial) {
    __shared__ float scratch[threads];
    const int channel = blockIdx.x;
    float sum = 0, product = 0;
    for (int s = threadIdx.x; s < spatial; s += threads) {
        const int index = channel * spatial + s;
        sum += dy[index];
        if (xhat) product += dy[index] * xhat[index];
    }
    const float bias = reduce_sum(sum, scratch);
    const float gamma = reduce_sum(product, scratch);
    if (threadIdx.x == 0) {
        if (dw) dw[channel] = gamma;
        if (db) db[channel] = bias;
    }
}
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
        norm_forward<<<groups, threads>>>(input.ptr, weights.ptr, result.ptr, nullptr, nullptr,
                                         group_size, spatial, channels, eps, bias);
    } else {
        if (!grad.alloc(count) || !normalized.alloc(count) || !invstd.alloc(groups) ||
            !grad.upload(dy->data(), count)) return false;
        norm_forward<<<groups, threads>>>(input.ptr, weights.ptr, result.ptr, normalized.ptr, invstd.ptr,
                                         group_size, spatial, channels, eps, bias);
        norm_backward<<<groups, threads>>>(normalized.ptr, invstd.ptr, grad.ptr, weights.ptr, result.ptr,
                                          group_size, spatial, channels);
        if (w) {
            if (!gamma_grad.alloc(channels) || (bias && !bias_grad.alloc(channels))) return false;
            channel_gradient<<<channels, threads>>>(grad.ptr, normalized.ptr, gamma_grad.ptr, bias_grad.ptr, spatial);
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

__global__ void resize_forward(const float* x, float* y, int count, int ih, int iw,
                               int oh, int ow, int sh, int sw, bool bilinear) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int c = i / (oh * ow), h = (i / ow) % oh, w = i % ow;
    if (!bilinear) { y[i] = x[(c * ih + h / sh) * iw + w / sw]; return; }
    const float hf = h * (static_cast<float>(ih) / oh), wf = w * (static_cast<float>(iw) / ow);
    const int h0 = static_cast<int>(floorf(hf)), w0 = static_cast<int>(floorf(wf));
    const int h1 = min(h0 + 1, ih - 1), w1 = min(w0 + 1, iw - 1);
    const float dh = hf - h0, dd = wf - w0;
    const int base = c * ih * iw;
    y[i] = x[base + h0 * iw + w0] * (1-dh) * (1-dd) + x[base + h0 * iw + w1] * (1-dh) * dd
         + x[base + h1 * iw + w0] * dh * (1-dd) + x[base + h1 * iw + w1] * dh * dd;
}
__global__ void nearest_backward(const float* dy, float* dx, int count, int ih, int iw, int sh, int sw) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int c = i / (ih * iw), h = (i / iw) % ih, w = i % iw;
    float sum = 0;
    for (int y = 0; y < sh; ++y) for (int x = 0; x < sw; ++x)
        sum += dy[(c * ih * sh + h * sh + y) * iw * sw + w * sw + x];
    dx[i] = sum;
}
__global__ void bilinear_backward(const float* dy, float* dx, int count, int ih, int iw, int oh, int ow) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int c = i / (oh * ow), h = (i / ow) % oh, w = i % ow;
    const float hf = h * (static_cast<float>(ih) / oh), wf = w * (static_cast<float>(iw) / ow);
    const int h0 = static_cast<int>(floorf(hf)), w0 = static_cast<int>(floorf(wf));
    const int h1 = min(h0+1, ih-1), w1 = min(w0+1, iw-1);
    const float dh = hf-h0, dd = wf-w0, g = dy[i];
    const int base = c*ih*iw;
    atomicAdd(dx+base+h0*iw+w0, g*(1-dh)*(1-dd));
    atomicAdd(dx+base+h0*iw+w1, g*(1-dh)*dd);
    atomicAdd(dx+base+h1*iw+w0, g*dh*(1-dd));
    atomicAdd(dx+base+h1*iw+w1, g*dh*dd);
}
inline bool resize(const std::vector<float>& source, std::vector<float>& output, int c,
                   int ih, int iw, int oh, int ow, int sh, int sw, bool bilinear, bool backward) {
    Buffer input, result;
    const size_t in_count = static_cast<size_t>(c)*ih*iw, out_count = static_cast<size_t>(c)*oh*ow;
    const size_t count = backward ? in_count : out_count;
    if (!input.alloc(source.size()) || !result.alloc(count) || !input.upload(source.data(), source.size())) return false;
    if (!backward) resize_forward<<<(out_count+threads-1)/threads, threads>>>(input.ptr, result.ptr, out_count, ih, iw, oh, ow, sh, sw, bilinear);
    else if (!bilinear) nearest_backward<<<(in_count+threads-1)/threads, threads>>>(input.ptr, result.ptr, in_count, ih, iw, sh, sw);
    else {
        if (cudaMemset(result.ptr, 0, count*sizeof(float)) != cudaSuccess) return false;
        bilinear_backward<<<(out_count+threads-1)/threads, threads>>>(input.ptr, result.ptr, out_count, ih, iw, oh, ow);
    }
    if (!finished()) return false;
    std::vector<float> staged(count);
    if (!result.download(staged.data(), count)) return false;
    output = std::move(staged); return true;
}

// Arbitrary axis: outer x axis_size x inner, including non-contiguous reductions.
__global__ void causal_mask(float* scores,int rows,int columns){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*columns && i%columns>i/columns)scores[i]=-1e9f;
}
__global__ void softmax_kernel(const float* x, const float* dy, float* y, int axis_size,
                               int inner, bool logarithmic) {
    __shared__ float scratch[threads];
    const int row = blockIdx.x;
    const int base = (row / inner) * axis_size * inner + row % inner;
    float maximum = -INFINITY;
    for (int a = threadIdx.x; a < axis_size; a += threads) maximum = fmaxf(maximum, x[base+a*inner]);
    maximum = reduce_max(maximum, scratch);
    float sum = 0;
    for (int a = threadIdx.x; a < axis_size; a += threads) sum += expf(x[base+a*inner]-maximum);
    sum = reduce_sum(sum, scratch);
    float reduction = 0;
    if (dy) {
        for (int a = threadIdx.x; a < axis_size; a += threads)
            reduction += dy[base+a*inner] * (logarithmic ? 1.f : expf(x[base+a*inner]-maximum)/sum);
        reduction = reduce_sum(reduction, scratch);
    }
    for (int a = threadIdx.x; a < axis_size; a += threads) {
        const int i = base+a*inner;
        const float probability = expf(x[i]-maximum)/sum;
        // Preserve the existing CPU LogSoftmax backward contract, which uses
        // the log output rather than its exponential in the reduction term.
        const float log_probability = x[i]-maximum-logf(sum);
        y[i] = dy ? (logarithmic ? dy[i]-log_probability*reduction : probability*(dy[i]-reduction))
                  : (logarithmic ? x[i]-maximum-logf(sum) : probability);
    }
}
inline bool softmax(const std::vector<float>& x, std::vector<float>& output, int axis_size,
                    int inner, bool logarithmic, const std::vector<float>* dy = nullptr) {
    Buffer input, result, gradient;
    if (!input.alloc(x.size()) || !result.alloc(x.size()) || !input.upload(x.data(),x.size())) return false;
    if (dy && (!gradient.alloc(x.size()) || !gradient.upload(dy->data(),dy->size()))) return false;
    const int rows = x.size() / (static_cast<size_t>(axis_size)*inner);
    softmax_kernel<<<rows*inner, threads>>>(input.ptr, gradient.ptr, result.ptr, axis_size, inner, logarithmic);
    if (!finished()) return false;
    std::vector<float> staged(x.size());
    if (!result.download(staged.data(),x.size())) return false;
    output=std::move(staged); return true;
}

__global__ void im2col(const float* x, float* col, int count, int h, int w, int k,
                       int stride, int pad, int dilation, int oh, int ow) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int s=i%(oh*ow), row=i/(oh*ow), c=row/(k*k), kh=(row/k)%k, kw=row%k;
    const int ih=(s/ow)*stride+kh*dilation-pad, iw=(s%ow)*stride+kw*dilation-pad;
    col[i] = ih>=0 && ih<h && iw>=0 && iw<w ? x[(c*h+ih)*w+iw] : 0.f;
}
__global__ void col2im(const float* col, float* dx, int count, int h, int w, int k,
                        int stride, int pad, int dilation, int oh, int ow) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int c=i/(h*w), ih=(i/w)%h, iw=i%w;
    float sum=0;
    for (int kh=0; kh<k; ++kh) for (int kw=0; kw<k; ++kw) {
        const int yh=ih+pad-kh*dilation, yw=iw+pad-kw*dilation;
        if (yh<0 || yw<0 || yh%stride || yw%stride) continue;
        const int y=yh/stride, x=yw/stride;
        if (y<oh && x<ow) sum += col[((c*k+kh)*k+kw)*oh*ow+y*ow+x];
    }
    dx[i]=sum;
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
    im2col<<<(col_count+threads-1)/threads,threads>>>(input.ptr,col.ptr,col_count,h,width,k,stride,pad,dilation,oh,ow);
    if (cudaGetLastError()!=cudaSuccess) return false;
    const float one=1.f,zero=0.f;
    // dW^T = col * dY^T (K x S times S x O).
    if (cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,inner,oc,spatial,&one,col.ptr,spatial,
                    grad.ptr,spatial,&zero,weight_grad.ptr,inner)!=CUBLAS_STATUS_SUCCESS) return false;
    // dcol^T = dY^T * W (S x O times O x K).
    if (cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,spatial,inner,oc,&one,grad.ptr,spatial,
                    weights.ptr,inner,&zero,dcol.ptr,spatial)!=CUBLAS_STATUS_SUCCESS) return false;
    col2im<<<(x.size()+threads-1)/threads,threads>>>(dcol.ptr,result.ptr,x.size(),h,width,k,stride,pad,dilation,oh,ow);
    if (bias) {
        if (!bias_grad.alloc(oc)) return false;
        channel_gradient<<<oc,threads>>>(grad.ptr,nullptr,nullptr,bias_grad.ptr,spatial);
    }
    if (!finished()) return false;
    dx.resize(x.size()); dw.resize(weight_count);
    if (!result.download(dx.data(),dx.size()) || !weight_grad.download(dw.data(),dw.size())) return false;
    if (bias) { db.resize(oc); if (!bias_grad.download(db.data(),oc)) return false; }
    return true;
}
// One block per row. Statistics, affine transform and derivatives stay on device.
__global__ void row_norm_kernel(const float* x,const float* w,const float* dy,float* y,float* xhat,int width,float eps,bool rms,bool bias) {
    __shared__ float scratch[threads];const int base=blockIdx.x*width;float sum=0;
    if(!rms)for(int i=threadIdx.x;i<width;i+=threads)sum+=x[base+i];
    const float mean=rms?0.f:reduce_sum(sum,scratch)/width;float variance=0;
    for(int i=threadIdx.x;i<width;i+=threads){const float d=x[base+i]-mean;variance+=d*d;}
    const float inv=rsqrtf(reduce_sum(variance,scratch)/width+eps);
    if(!dy){for(int i=threadIdx.x;i<width;i+=threads){const float v=(x[base+i]-mean)*inv;y[base+i]=v*(w?w[i]:1.f)+(bias?w[width+i]:0.f);}return;}
    float sg=0,sgx=0;
    for(int i=threadIdx.x;i<width;i+=threads){const float v=(x[base+i]-mean)*inv;const float g=dy[base+i]*(w?w[i]:1.f);xhat[base+i]=v;sg+=g;sgx+=g*v;}
    const float a=reduce_sum(sg,scratch)/width,b=reduce_sum(sgx,scratch)/width;
    for(int i=threadIdx.x;i<width;i+=threads){const float g=dy[base+i]*(w?w[i]:1.f);y[base+i]=inv*(g-(rms?0.f:a)-xhat[base+i]*b);}
}
__global__ void row_norm_grad(const float* dy,const float* xhat,float* dw,float* db,int rows,int width) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=width)return;float a=0,b=0;
    for(int r=0;r<rows;++r){const int j=r*width+i;a+=dy[j]*xhat[j];b+=dy[j];}dw[i]=a;if(db)db[i]=b;
}
inline bool row_norm(const std::vector<float>& x,const Layer& l,std::vector<float>& y,const std::vector<float>* dy=nullptr,std::vector<float>* dw=nullptr,std::vector<float>* db=nullptr) {
    const int width=l.in_features>0?l.in_features:x.size();const int rows=x.size()/width;
    const float* w=l.affine?l.getWeights():nullptr;const bool bias=w && l.use_bias && l.getWeightsSize()>=2ULL*width;
    Buffer bx,bw,bg,by,bh,bdw,bdb;
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !by.alloc(x.size()))return false;
    if(w && (!bw.alloc(l.getWeightsSize()) || !bw.upload(w,l.getWeightsSize())))return false;
    if(dy && (!bg.alloc(x.size()) || !bg.upload(dy->data(),x.size()) || !bh.alloc(x.size())))return false;
    row_norm_kernel<<<rows,threads>>>(bx.ptr,bw.ptr,bg.ptr,by.ptr,bh.ptr,width,l.eps>0?l.eps:1e-5f,l.type_enum==LayerType::RMSNorm,bias && (dy || l.type_enum!=LayerType::RMSNorm));
    if(dy && w){
        if(!bdw.alloc(width) || (bias && !bdb.alloc(width)))return false;
        row_norm_grad<<<(width+threads-1)/threads,threads>>>(bg.ptr,bh.ptr,bdw.ptr,bdb.ptr,rows,width);
    }
    if(!finished())return false;
    y.resize(x.size());if(!by.download(y.data(),y.size()))return false;
    if(dy && w){dw->resize(width);if(!bdw.download(dw->data(),width))return false;if(bias){db->resize(width);if(!bdb.download(db->data(),width))return false;}}
    return true;
}
__global__ void projection_kernel(const float* a,const float* b,const float* w,const float* dy,float* y,float* da,float* db,float* dw,
                                  CudaOperationGeometry::Projection g,int offset_count) {
    const int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=g.output)return;
    if(g.mode==3 && o<g.prefix){if(dy)da[o]=dy[o];else y[o]=a[o];return;}
    const int d=(o-(g.mode==3?g.prefix:0))%g.dim,r=(o-(g.mode==3?g.prefix:0))/g.dim;
    const float grad=dy?dy[o]:0.f;float value=0;
    if(g.mode<=2){
        int start=g.mode==1?r:0,end=g.mode==1?r+1:g.ids;
        if(g.mode==2 && offset_count){start=max(0,min(static_cast<int>(llround(static_cast<double>(b[r]))),g.ids));end=max(start,min(static_cast<int>(llround(static_cast<double>(b[r+1]))),g.ids));}
        for(int i=start;i<end;++i){const int id=static_cast<int>(llround(static_cast<double>(a[i])));if(id<0 || id>=g.vocab || id==g.pad)continue;
            const int j=id*g.dim+d;if(dy)atomicAdd(dw+j,grad);else value+=w[j];}
    }else if(g.mode==3){
        const float inv=rsqrtf(static_cast<float>(g.features));const int base=g.prefix+r*g.features;
        if(dy)atomicAdd(dw+g.features*g.dim+d,grad);else value=w[g.features*g.dim+d];
        for(int k=0;k<g.features;++k){const int j=k*g.dim+d;if(dy){atomicAdd(da+base+k,grad*w[j]*inv);atomicAdd(dw+j,grad*a[base+k]*inv);}else value+=(a[base+k]*inv)*w[j];}
    }else{
        const int base=d*g.features*g.second;
        if(g.bias){if(dy)atomicAdd(dw+g.dim*g.features*g.second+d,grad);else value=w[g.dim*g.features*g.second+d];}
        for(int i=0;i<g.features;++i)for(int j=0;j<g.second;++j){const int wi=base+i*g.second+j,ai=r*g.features+i,bi=r*g.second+j;
            if(dy){atomicAdd(da+ai,grad*w[wi]*b[bi]);atomicAdd(db+bi,grad*w[wi]*a[ai]);atomicAdd(dw+wi,grad*a[ai]*b[bi]);}else value+=a[ai]*w[wi]*b[bi];}
    }
    if(!dy)y[o]=value;
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
    projection_kernel<<<(g.output+threads-1)/threads,threads>>>(a.ptr,b.ptr,w.ptr,grad.ptr,y.ptr,da.ptr,db.ptr,weights_grad.ptr,g,g.mode==2 && second?in[1]->size():0);
    if(!finished())return false;
    if(dy){
        out={std::vector<float>(in[0]->size())};if(!da.download(out[0].data(),out[0].size()))return false;
        if((g.mode==2 || g.mode==4) && in.size()>1 && in[1]){out.emplace_back(in[1]->size(),0.f);if(g.mode==4 && !db.download(out[1].data(),out[1].size()))return false;}
        dw->resize(l.getWeightsSize());return weights_grad.download(dw->data(),dw->size());
    }
    out={std::vector<float>(g.output)};return y.download(out[0].data(),g.output);
}
__global__ void geglu_kernel(const float* x,const float* dy,float* y,int width,int count){
    const int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=count)return;const int a=(o/width)*2*width+o%width,b=a+width;
    const float z=x[b],th=tanhf(.7978845608028654f*(z+.044715f*z*z*z)),gelu=z*.5f*(1+th);
    if(!dy)y[o]=x[a]*gelu;else{y[a]=dy[o]*gelu;y[b]=dy[o]*x[a]*(.5f*(1+th)+.5f*z*(1-th*th)*.7978845608028654f*(1+3*.044715f*z*z));}
}
inline bool geglu(const std::vector<float>& x,std::vector<float>& y,int width,const std::vector<float>* dy=nullptr){
    Buffer bx,bg,by;const int count=x.size()/2;y.resize(dy?x.size():count);
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !by.alloc(y.size()))return false;
    if(dy && (!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size())))return false;
    geglu_kernel<<<(count+threads-1)/threads,threads>>>(bx.ptr,bg.ptr,by.ptr,width,count);return finished() && by.download(y.data(),y.size());
}
__device__ inline float cubic(float x){x=fabsf(x);const float a=-.75f;if(x<=1)return ((a+2)*x-(a+3))*x*x+1;if(x<2)return (((a*x-5*a)*x+8*a)*x)-4*a;return 0;}
__global__ void bicubic_kernel(const float* x,const float* dy,float* y,CudaOperationGeometry::Resize g,int count){
    const int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=count)return;const int c=o/(g.oh*g.ow),oy=(o/g.ow)%g.oh,ox=o%g.ow;
    const float sy=(oy+.5f)*(static_cast<float>(g.ih)/g.oh)-.5f,sx=(ox+.5f)*(static_cast<float>(g.iw)/g.ow)-.5f;
    const int iy=static_cast<int>(floorf(sy)),ix=static_cast<int>(floorf(sx));float sum=0,wsum=0;
    for(int h=-1;h<=2;++h)for(int w=-1;w<=2;++w){const float weight=cubic(sy-(iy+h))*cubic(sx-(ix+w));wsum+=weight;if(!dy)sum+=weight*x[(c*g.ih+max(0,min(iy+h,g.ih-1)))*g.iw+max(0,min(ix+w,g.iw-1))];}
    if(!dy){y[o]=wsum?sum/wsum:0.f;return;}if(!wsum)return;
    for(int h=-1;h<=2;++h)for(int w=-1;w<=2;++w){const float weight=cubic(sy-(iy+h))*cubic(sx-(ix+w));const int i=(c*g.ih+max(0,min(iy+h,g.ih-1)))*g.iw+max(0,min(ix+w,g.iw-1));atomicAdd(y+i,dy[o]*(weight/wsum));}
}
inline bool bicubic(const std::vector<float>& x,std::vector<float>& y,const CudaOperationGeometry::Resize& g,const std::vector<float>* dy=nullptr){
    Buffer bx,bg,by;y.resize(dy?x.size():g.outputSize());
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !by.alloc(y.size()))return false;
    if(dy && (!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size()) || cudaMemset(by.ptr,0,y.size()*sizeof(float))!=cudaSuccess))return false;
    const int count=g.outputSize();bicubic_kernel<<<(count+threads-1)/threads,threads>>>(bx.ptr,bg.ptr,by.ptr,g,count);return finished() && by.download(y.data(),y.size());
}
// Parallel pooling: independent windows forward; atomics sum overlapping gradients.
__global__ void pool_kernel(const float* x,const float* dy,float* y,float* dx,
                           CudaOperationGeometry::Pool g,int count) {
    const int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=count)return;
    const int c=o/(g.oh*g.ow),oy=(o/g.ow)%g.oh,ox=o%g.ow;
    float value=g.maximum?-INFINITY:0.f;int valid=0,best=-1;
    for(int ky=0;ky<g.kh;++ky)for(int kx=0;kx<g.kw;++kx) {
        const int iy=oy*g.sh+ky-g.ph,ix=ox*g.sw+kx-g.pw;
        if(iy<0 || iy>=g.h || ix<0 || ix>=g.w)continue;
        const int i=g.token?iy*g.c+c:(c*g.h+iy)*g.w+ix;
        ++valid;if(g.maximum){if(x[i]>value){value=x[i];best=i;}}else value+=x[i];
    }
    if(!dy){y[o]=g.maximum?value:(valid?value/valid:0.f);return;}
    if(g.maximum){if(best>=0)atomicAdd(dx+best,dy[o]);return;}
    if(!valid)return;const float grad=dy[o]/valid;
    for(int ky=0;ky<g.kh;++ky)for(int kx=0;kx<g.kw;++kx){
        const int iy=oy*g.sh+ky-g.ph,ix=ox*g.sw+kx-g.pw;
        if(iy>=0 && iy<g.h && ix>=0 && ix<g.w)atomicAdd(dx+(g.token?iy*g.c+c:(c*g.h+iy)*g.w+ix),grad);
    }
}
inline bool pool(const std::vector<float>& x,std::vector<float>& y,const CudaOperationGeometry::Pool& g,
                 const std::vector<float>* dy=nullptr) {
    Buffer bx,by,bg;if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()))return false;
    y.resize(dy?x.size():g.outputSize());if(!by.alloc(y.size()))return false;
    if(dy && (!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size()) || cudaMemset(by.ptr,0,y.size()*sizeof(float))!=cudaSuccess))return false;
    const int n=g.outputSize();pool_kernel<<<(n+threads-1)/threads,threads>>>(bx.ptr,dy?bg.ptr:nullptr,dy?nullptr:by.ptr,dy?by.ptr:nullptr,g,n);
    return finished() && by.download(y.data(),y.size());
}
__global__ void permute_kernel(const float* x,float* y,const int* out_shape,const int* in_stride,int rank,int count,bool backward) {
    const int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=count)return;int tmp=o,i=0;
    for(int d=rank-1;d>=0;--d){i+=(tmp%out_shape[d])*in_stride[d];tmp/=out_shape[d];}
    if(backward)y[i]=x[o];else y[o]=x[i];
}
__global__ void spatial_move(const float* x,float* y,int count,int c,int h,int w,int oh,int ow,int ph,int pw,int r,int padding,bool backward) {
    const int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=count)return;
    const int ch=o/(oh*ow),oy=(o/ow)%oh,ox=o%ow;int i=-1;
    if(r) {const int ic=ch*r*r+(oy%r)*r+ox%r;i=(ic*h+oy/r)*w+ox/r;}
    else {
        int iy=oy-ph,ix=ox-pw;
        if(padding==2){if(iy<0)iy=-iy;if(iy>=h)iy=2*h-iy-2;if(ix<0)ix=-ix;if(ix>=w)ix=2*w-ix-2;}
        if(padding!=1){iy=max(0,min(iy,h-1));ix=max(0,min(ix,w-1));}
        if(iy>=0 && iy<h && ix>=0 && ix<w)i=(ch*h+iy)*w+ix;
    }
    if(backward){if(i>=0)atomicAdd(y+i,x[o]);}else y[o]=i>=0?x[i]:0.f;
}
inline bool movement(const std::vector<const std::vector<float>*>& in,std::vector<std::vector<float>>& out,
                     const Layer& l,const CudaOperationGeometry::Movement& g,bool backward=false) {
    if(g.padding || g.shuffle){
        const auto& x=*in[0];const size_t count=g.shuffle?static_cast<size_t>(g.c)*g.h*g.w:static_cast<size_t>(g.c)*g.oh*g.ow;
        const size_t n=backward?static_cast<size_t>(g.c)*g.h*g.w:count;
        Buffer bx,by;out={std::vector<float>(n)};
        if(!bx.alloc(x.size()) || !by.alloc(n) || !bx.upload(x.data(),x.size()) || cudaMemset(by.ptr,0,n*sizeof(float))!=cudaSuccess)return false;
        spatial_move<<<(count+threads-1)/threads,threads>>>(bx.ptr,by.ptr,count,g.c,g.h,g.w,g.oh,g.ow,g.ph,g.pw,g.shuffle,g.padding,backward);
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
        if(ok){permute_kernel<<<(x.size()+threads-1)/threads,threads>>>(bx.ptr,by.ptr,bs,bt,shape.size(),x.size(),backward);ok=finished() && by.download(out[0].data(),x.size());}
        cudaFree(bs);cudaFree(bt);return ok;
    }
    // All copies pass through device memory; empty chunks need no device allocation.
    size_t total=0;for(auto* x:in){if(!x)return false;total+=x->size();}
    if(total>INT_MAX)return false;
    if(!total){out.clear();for(auto n:g.sizes)out.emplace_back(n);return true;}
    Buffer b;if(!b.alloc(total))return false;size_t off=0;
    for(auto* x:in){if(x->size() && cudaMemcpy(b.ptr+off,x->data(),x->size()*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)return false;off+=x->size();}
    out.clear();off=0;for(auto n:g.sizes){if(n>total-off)return false;out.emplace_back(n);if(n && cudaMemcpy(out.back().data(),b.ptr+off,n*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)return false;off+=n;}
    return off==total && finished();
}
} // namespace CudaNativeOps
