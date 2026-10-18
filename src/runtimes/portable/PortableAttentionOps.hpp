#pragma once
#include "PortableNativeOps.hpp"
namespace PortableNativeOps {
using namespace PortableStaging;
// Native attention. cuBLAS projections/context and per-head device reductions;
// no Q/K/V packing, softmax or derivative calculation on the host.









inline bool attention(const std::vector<const std::vector<float>*>& in,std::vector<std::vector<float>>& out,const Layer& l,
    const CudaOperationGeometry::Attention& g,const std::vector<float>* dy=nullptr,std::vector<float>* dw=nullptr,cublasHandle_t handle=nullptr){
    Buffer x,kv,w,q,k,v,p,ctx,y,grad,dc,ds,dq,dk,dv,weights_grad,dx,dkv;
    const size_t nq=static_cast<size_t>(g.q)*g.ed,nk=static_cast<size_t>(g.k)*g.ed,np=static_cast<size_t>(g.heads)*g.q*g.k;
    if(!x.alloc(in[0]->size()) || !x.upload(in[0]->data(),in[0]->size()) || !w.alloc(l.getWeightsSize()) || !w.upload(l.getWeights(),l.getWeightsSize()) ||
        !q.alloc(nq) || !k.alloc(nk) || !v.alloc(nk) || !p.alloc(np) || !ctx.alloc(nq))return false;
    if(g.cross && (!kv.alloc(in[1]->size()) || !kv.upload(in[1]->data(),in[1]->size())))return false;
    const float one=1.f,zero=0.f;
    for(int part=0;part<3;++part){
        const size_t n=part?nk:nq;float* result=part==0?q.ptr:(part==1?k.ptr:v.ptr);
        if(handle){
            const float* input=g.cross && part?kv.ptr:x.ptr;
            const int features=g.cross && part?g.ked:g.ed,length=part?g.k:g.q;
            const int offset=!g.cross?part*g.ed*g.ed:(part?g.ed*g.ed+(part-1)*g.ed:0);
            const int lda=g.cross && part?2*g.ed:g.ed;
            if(cublasSgemm(handle,g.cross?CUBLAS_OP_N:CUBLAS_OP_T,CUBLAS_OP_N,g.ed,length,features,&one,w.ptr+offset,lda,input,features,&zero,result,g.ed)!=CUBLAS_STATUS_SUCCESS)return false;
            if(!dy && !g.cross && g.qbias>=0)launch("attention_add_bias", (n+threads-1)/threads,threads, result,w.ptr,n,g.ed,g.qbias+part*g.ed);
        }else launch("attention_project", (n+threads-1)/threads,threads, g.cross && part?kv.ptr:x.ptr,w.ptr,result,g,part,dy!=nullptr,n);
    }
    launch("attention_probabilities", g.heads*g.q,threads, q.ptr,k.ptr,p.ptr,g);
    if(handle){
        const int width=g.ed/g.heads;
        for(int h=0;h<g.heads;++h)if(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,width,g.q,g.k,&one,v.ptr+h*width,g.ed,p.ptr+static_cast<size_t>(h)*g.q*g.k,g.k,&zero,ctx.ptr+h*width,g.ed)!=CUBLAS_STATUS_SUCCESS)return false;
    }else launch("attention_context", (nq+threads-1)/threads,threads, p.ptr,v.ptr,ctx.ptr,g,nq);
    // CPU backward recomputes attention without bias and leaves its bias gradients untouched.
    // Preserve that compatibility contract while forward supports both bias layouts.
    if(!dy){if(!y.alloc(nq))return false;
        if(handle){if(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,g.ed,g.q,g.ed,&one,w.ptr+g.mainParameters-g.ed*g.ed,g.ed,ctx.ptr,g.ed,&zero,y.ptr,g.ed)!=CUBLAS_STATUS_SUCCESS)return false;
            if(g.obias>=0)launch("attention_add_bias", (nq+threads-1)/threads,threads, y.ptr,w.ptr,nq,g.ed,g.obias);
        }else launch("attention_output", (nq+threads-1)/threads,threads, ctx.ptr,w.ptr,nullptr,y.ptr,g,nq);
        if(!finished())return false;
        out={std::vector<float>(nq)};return y.download(out[0].data(),nq);}
    if(!grad.alloc(nq) || !grad.upload(dy->data(),nq) || !dc.alloc(nq) || !ds.alloc(np) || !dq.alloc(nq) || !dk.alloc(nk) || !dv.alloc(nk) ||
        !weights_grad.alloc(l.getWeightsSize()) || cudaMemset(weights_grad.ptr,0,l.getWeightsSize()*sizeof(float))!=cudaSuccess || !dx.alloc(in[0]->size()))return false;
    if(g.cross && !dkv.alloc(in[1]->size()))return false;
    if(handle){if(cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,g.ed,g.q,g.ed,&one,w.ptr+g.mainParameters-g.ed*g.ed,g.ed,grad.ptr,g.ed,&zero,dc.ptr,g.ed)!=CUBLAS_STATUS_SUCCESS)return false;}
    else launch("attention_output", (nq+threads-1)/threads,threads, ctx.ptr,w.ptr,grad.ptr,dc.ptr,g,nq);
    launch("attention_score_grad", g.heads*g.q,threads, p.ptr,v.ptr,dc.ptr,ds.ptr,g);
    for(int part=0;part<3;++part){const size_t n=part?nk:nq;launch("attention_projected_grad", (n+threads-1)/threads,threads, q.ptr,k.ptr,dc.ptr,p.ptr,ds.ptr,part==0?dq.ptr:(part==1?dk.ptr:dv.ptr),g,part,n);}
    launch("attention_weight_grad", (g.mainParameters+threads-1)/threads,threads, x.ptr,kv.ptr,ctx.ptr,grad.ptr,dq.ptr,dk.ptr,dv.ptr,weights_grad.ptr,g,g.mainParameters);
    launch("attention_input_grad", (in[0]->size()+threads-1)/threads,threads, w.ptr,dq.ptr,dk.ptr,dv.ptr,dx.ptr,g,false,in[0]->size());
    if(g.cross)launch("attention_input_grad", (in[1]->size()+threads-1)/threads,threads, w.ptr,dq.ptr,dk.ptr,dv.ptr,dkv.ptr,g,true,in[1]->size());
    if(!finished())return false;
    out={std::vector<float>(in[0]->size())};if(!dx.download(out[0].data(),out[0].size()))return false;
    if(g.cross){out.emplace_back(in[1]->size());if(!dkv.download(out[1].data(),out[1].size()))return false;}
    dw->resize(l.getWeightsSize());return weights_grad.download(dw->data(),dw->size());
}
} // namespace PortableNativeOps
