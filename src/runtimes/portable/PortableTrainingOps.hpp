#pragma once
#include "PortableNativeOps.hpp"
#include "RngContext.hpp"
#include <random>

namespace PortableNativeOps {
using namespace PortableStaging;
// Reference convolution kernels, independent of the CPU runtime. Each output
// gathers its source pixels; gradients scatter into zeroed device buffers.

inline bool extended_conv(const std::vector<float>& x,const Layer& l,const CudaOperationGeometry::ExtendedConv& g,
    std::vector<float>& y,const std::vector<float>* dy=nullptr,std::vector<float>* dw=nullptr){
    Buffer bx,bw,bg,by,bdw;
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !bw.alloc(l.getWeightsSize()) || !bw.upload(l.getWeights(),l.getWeightsSize()))return false;
    y.resize(dy?x.size():g.outputSize());if(!by.alloc(y.size()))return false;
    if(dy && (!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size()) || !bdw.alloc(l.getWeightsSize()) ||
        cudaMemset(by.ptr,0,y.size()*sizeof(float))!=cudaSuccess || cudaMemset(bdw.ptr,0,l.getWeightsSize()*sizeof(float))!=cudaSuccess))return false;
    const int n=g.outputSize();launch("extended_conv_kernel", (n+threads-1)/threads,threads, bx.ptr,bw.ptr,bg.ptr,dy?nullptr:by.ptr,dy?by.ptr:nullptr,bdw.ptr,g,n);
    if(!finished() || !by.download(y.data(),y.size()))return false;
    if(dy){dw->resize(l.getWeightsSize());return bdw.download(dw->data(),dw->size());}return true;
}

// Masks/noise use Mimir's host RNG stream; all transformations and derivatives
// run on device. Explicit masks preserve gradients for retained zero-valued data.

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
    launch("dropout_kernel", (x.size()+threads-1)/threads,threads, bx.ptr,bm.ptr,by.ptr,x.size(),p,alpha,backward,training);
    if(!finished())return false;
    y.resize(x.size());return by.download(y.data(),y.size());
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
    launch("reparameterize_kernel", (n+threads-1)/threads,threads, mu.ptr,lv.ptr,noise.ptr,grad.ptr,a.ptr,b.ptr,n,training);
    if(!finished())return false;
    out={std::vector<float>(n)};if(!a.download(out[0].data(),n))return false;
    if(dy){out.emplace_back(n);return b.download(out[1].data(),n);}return true;
}


inline bool batch_norm(const std::vector<float>& x,std::vector<float>& y,const Layer& l,const CudaOperationGeometry::BatchNorm& g,bool training,const std::vector<float>* dy=nullptr,std::vector<float>* dw=nullptr){
    Buffer bx,bw,br,bg,by,bdw;
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !by.alloc(x.size()))return false;
    if(g.affine && (!bw.alloc(l.getWeightsSize()) || !bw.upload(l.getWeights(),l.getWeightsSize())))return false;
    if(g.running){std::vector<float> stats=l.running_mean;stats.insert(stats.end(),l.running_var.begin(),l.running_var.end());if(!br.alloc(stats.size()) || !br.upload(stats.data(),stats.size()))return false;}
    if(dy && (!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size())))return false;
    if(dy && g.affine && (!bdw.alloc(l.getWeightsSize()) || cudaMemset(bdw.ptr,0,l.getWeightsSize()*sizeof(float))!=cudaSuccess))return false;
    launch("batch_norm_kernel", g.channels,threads, bx.ptr,bw.ptr,br.ptr,bg.ptr,by.ptr,bdw.ptr,g,l.eps,training);
    if(!finished())return false;
    y.resize(x.size());if(!by.download(y.data(),y.size()))return false;
    if(dy && g.affine){dw->resize(l.getWeightsSize());return bdw.download(dw->data(),dw->size());}return true;
}
} // namespace PortableNativeOps
