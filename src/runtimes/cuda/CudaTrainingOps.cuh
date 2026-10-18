#pragma once
#include "CudaNativeOps.cuh"
#include "RngContext.hpp"
#include <random>

namespace CudaNativeOps {
// Reference convolution kernels, independent of the CPU runtime. Each output
// gathers its source pixels; gradients scatter into zeroed device buffers.
__global__ void extended_conv_kernel(const float* x,const float* weights,const float* dy,
    float* y,float* dx,float* dw,CudaOperationGeometry::ExtendedConv g,int count){
    const int o=blockIdx.x*blockDim.x+threadIdx.x;if(o>=count)return;
    const int oc=o/(g.oh*g.ow),oy=(o/g.ow)%g.oh,ox=o%g.ow;
    float value=g.bias?weights[g.parameters+oc]:0.f;
    const float grad=dy?dy[o]:0.f;
    if(dy && g.bias)atomicAdd(dw+g.parameters+oc,grad);
    const int first=g.mode==2?oc:0,last=g.mode==2?oc+1:g.ic;
    for(int ic=first;ic<last;++ic)for(int kh=0;kh<g.kh;++kh)for(int kw=0;kw<g.kw;++kw){
        int iy=oy*g.stride+kh-g.pad,ix=g.mode==1?0:ox*g.stride+kw-g.pad;
        if(g.mode==3){iy=oy+g.pad-kh;ix=ox+g.pad-kw;if(iy<0 || ix<0 || iy%g.stride || ix%g.stride)continue;iy/=g.stride;ix/=g.stride;}
        if(iy<0 || iy>=g.h || ix<0 || ix>=g.w)continue;
        const int i=(ic*g.h+iy)*g.w+ix;
        const int wi=((oc*(g.mode==2?1:g.ic)+(g.mode==2?0:ic))*g.kh+kh)*g.kw+kw;
        if(dy){atomicAdd(dx+i,grad*weights[wi]);atomicAdd(dw+wi,grad*x[i]);}else value+=x[i]*weights[wi];
    }
    if(!dy)y[o]=value;
}
inline bool extended_conv(const std::vector<float>& x,const Layer& l,const CudaOperationGeometry::ExtendedConv& g,
    std::vector<float>& y,const std::vector<float>* dy=nullptr,std::vector<float>* dw=nullptr){
    Buffer bx,bw,bg,by,bdw;
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !bw.alloc(l.getWeightsSize()) || !bw.upload(l.getWeights(),l.getWeightsSize()))return false;
    y.resize(dy?x.size():g.outputSize());if(!by.alloc(y.size()))return false;
    if(dy && (!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size()) || !bdw.alloc(l.getWeightsSize()) ||
        cudaMemset(by.ptr,0,y.size()*sizeof(float))!=cudaSuccess || cudaMemset(bdw.ptr,0,l.getWeightsSize()*sizeof(float))!=cudaSuccess))return false;
    const int n=g.outputSize();extended_conv_kernel<<<(n+threads-1)/threads,threads>>>(bx.ptr,bw.ptr,bg.ptr,dy?nullptr:by.ptr,dy?by.ptr:nullptr,bdw.ptr,g,n);
    if(!finished() || !by.download(y.data(),y.size()))return false;
    if(dy){dw->resize(l.getWeightsSize());return bdw.download(dw->data(),dw->size());}return true;
}

// Masks/noise use Mimir's host RNG stream; all transformations and derivatives
// run on device. Explicit masks preserve gradients for retained zero-valued data.
__global__ void dropout_kernel(const float* x,const float* mask,float* y,int n,float p,bool alpha,bool backward,bool active){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(!active || p==0.f){y[i]=x[i];return;}if(p==1.f){y[i]=0;return;}
    const float ap=-1.6732632423543772848f*1.0507009873554804934f;
    const float scale=alpha?rsqrtf((1-p)*(1+p*ap*ap)):1.f/(1-p);
    if(backward)y[i]=mask[i]?x[i]*scale:0.f;
    else y[i]=alpha?scale*(mask[i]?x[i]:ap)-scale*ap*p:(mask[i]?x[i]*scale:0.f);
}
inline bool dropout(const std::vector<float>& x,std::vector<float>& y,const Layer& l,bool training,
    std::vector<uint8_t>* output_mask=nullptr,const std::vector<float>* saved_output=nullptr,bool backward=false){
    const float p=std::clamp(l.dropout_p,0.f,1.f);const bool alpha=l.type_enum==LayerType::AlphaDropout;
    std::vector<float> mask(x.size(),1.f);
    if(training){
        if(backward && p>0 && p<1){
            if(!saved_output || saved_output->size()!=x.size())return false;
            const float ap=-1.6732632423543772848f*1.0507009873554804934f;
            const float dropped=alpha?ap*(1-p)/std::sqrt((1-p)*(1+p*ap*ap)):0.f;
            for(size_t i=0;i<x.size();++i)mask[i]=alpha?(std::abs((*saved_output)[i]-dropped)>1e-6f):((*saved_output)[i]!=0.f);
        }else if(!backward){
            if(p==1){std::fill(mask.begin(),mask.end(),0.f);}else{
                auto& rng=MimirRng::generator();std::uniform_real_distribution<float> distribution(0.f,1.f);
                for(float& kept:mask)kept=distribution(rng)>p;
            }
        }
    }
    if(output_mask){output_mask->resize(mask.size());for(size_t i=0;i<mask.size();++i)(*output_mask)[i]=mask[i]!=0;}
    Buffer bx,bm,by;if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !bm.alloc(mask.size()) || !bm.upload(mask.data(),mask.size()) || !by.alloc(x.size()))return false;
    dropout_kernel<<<(x.size()+threads-1)/threads,threads>>>(bx.ptr,bm.ptr,by.ptr,x.size(),p,alpha,backward,training);
    if(!finished())return false;
    y.resize(x.size());return by.download(y.data(),y.size());
}
__global__ void reparameterize_kernel(const float* mu,const float* lv,const float* noise,const float* dy,float* a,float* b,int n,bool training){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const float sigma=expf(.5f*fmaxf(-20.f,fminf(20.f,lv[i])));
    if(!dy){a[i]=training?mu[i]+sigma*noise[i]:mu[i];return;}
    a[i]=dy[i];b[i]=0.f;
    if(noise && lv[i]>=-20.f && lv[i]<=20.f){const float inv=sigma>1e-12f?1.f/sigma:0.f;b[i]=dy[i]*.5f*sigma*((noise[i]-mu[i])*inv);}
}
inline bool reparameterize(const std::vector<const std::vector<float>*>& in,std::vector<std::vector<float>>& out,bool training,const std::vector<float>* dy=nullptr){
    const size_t n=in[0]->size();Buffer mu,lv,noise,grad,a,b;
    if(!mu.alloc(n) || !mu.upload(in[0]->data(),n) || !lv.alloc(n) || !lv.upload(in[1]->data(),n) || !a.alloc(n))return false;
    if(dy){if(!grad.alloc(n) || !grad.upload(dy->data(),n) || !b.alloc(n))return false;
        if(in.size()>2 && in[2] && in[2]->size()==n && (!noise.alloc(n) || !noise.upload(in[2]->data(),n)))return false;
    }else if(training){
        // Same independent, fixed-seed stream as the existing CPU contract.
        static thread_local std::mt19937 rng(1337);std::normal_distribution<float> normal(0.f,1.f);
        std::vector<float> epsilon(n);for(float& v:epsilon)v=normal(rng);
        if(!noise.alloc(n) || !noise.upload(epsilon.data(),n))return false;
    }
    reparameterize_kernel<<<(n+threads-1)/threads,threads>>>(mu.ptr,lv.ptr,noise.ptr,grad.ptr,a.ptr,b.ptr,n,training);
    if(!finished())return false;
    out={std::vector<float>(n)};if(!a.download(out[0].data(),n))return false;
    if(dy){out.emplace_back(n);return b.download(out[1].data(),n);}return true;
}

__global__ void batch_norm_kernel(const float* x,const float* w,const float* running,const float* dy,
    float* y,float* dw,CudaOperationGeometry::BatchNorm g,float eps,bool training){
    __shared__ float scratch[threads];const int c=blockIdx.x,N=g.batch*g.spatial;float sum=0;
    const bool estimate=dy || training || !g.running;
    if(estimate)for(int j=threadIdx.x;j<N;j+=threads)sum+=x[(j/g.spatial*g.channels+c)*g.spatial+j%g.spatial];
    const float mean=estimate?reduce_sum(sum,scratch)/N:running[c];float variance=0;
    if(estimate)for(int j=threadIdx.x;j<N;j+=threads){const float d=x[(j/g.spatial*g.channels+c)*g.spatial+j%g.spatial]-mean;variance+=d*d;}
    const float var=estimate?reduce_sum(variance,scratch)/N:running[g.channels+c],inv=rsqrtf(var+eps);
    const float gamma=g.affine?w[c]:1.f,beta=g.bias?w[g.channels+c]:0.f;
    if(!dy){for(int j=threadIdx.x;j<N;j+=threads){const int i=(j/g.spatial*g.channels+c)*g.spatial+j%g.spatial;y[i]=(x[i]-mean)*inv*gamma+beta;}return;}
    float s1=0,s2=0;
    for(int j=threadIdx.x;j<N;j+=threads){const int i=(j/g.spatial*g.channels+c)*g.spatial+j%g.spatial;s1+=dy[i];s2+=dy[i]*(x[i]-mean)*inv;}
    s1=reduce_sum(s1,scratch);s2=reduce_sum(s2,scratch);
    if(threadIdx.x==0 && g.affine){dw[c]=s2;if(g.bias)dw[g.channels+c]=s1;}
    for(int j=threadIdx.x;j<N;j+=threads){const int i=(j/g.spatial*g.channels+c)*g.spatial+j%g.spatial;const float xhat=(x[i]-mean)*inv;y[i]=inv*gamma*(dy[i]-s1/N-xhat*s2/N);}
}
inline bool batch_norm(const std::vector<float>& x,std::vector<float>& y,const Layer& l,const CudaOperationGeometry::BatchNorm& g,bool training,const std::vector<float>* dy=nullptr,std::vector<float>* dw=nullptr){
    Buffer bx,bw,br,bg,by,bdw;
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !by.alloc(x.size()))return false;
    if(g.affine && (!bw.alloc(l.getWeightsSize()) || !bw.upload(l.getWeights(),l.getWeightsSize())))return false;
    if(g.running){std::vector<float> stats=l.running_mean;stats.insert(stats.end(),l.running_var.begin(),l.running_var.end());if(!br.alloc(stats.size()) || !br.upload(stats.data(),stats.size()))return false;}
    if(dy && (!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size())))return false;
    if(dy && g.affine && (!bdw.alloc(l.getWeightsSize()) || cudaMemset(bdw.ptr,0,l.getWeightsSize()*sizeof(float))!=cudaSuccess))return false;
    batch_norm_kernel<<<g.channels,threads>>>(bx.ptr,bw.ptr,br.ptr,bg.ptr,by.ptr,bdw.ptr,g,l.eps,training);
    if(!finished())return false;
    y.resize(x.size());if(!by.download(y.data(),y.size()))return false;
    if(dy && g.affine){dw->resize(l.getWeightsSize());return bdw.download(dw->data(),dw->size());}return true;
}
} // namespace CudaNativeOps
