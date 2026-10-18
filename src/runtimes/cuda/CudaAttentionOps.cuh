#pragma once
#include "CudaNativeOps.cuh"
namespace CudaNativeOps {
// Native attention. cuBLAS projections/context and per-head device reductions;
// no Q/K/V packing, softmax or derivative calculation on the host.
__global__ void attention_add_bias(float* y,const float* w,int n,int width,int offset){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n && offset>=0)y[i]+=w[offset+i%width];
}
__global__ void attention_project(const float* input,const float* w,float* result,
    CudaOperationGeometry::Attention g,int part,bool backward,int n){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const int t=i/g.ed,d=i%g.ed;
    const int features=part && g.cross?g.ked:g.ed;float value=0;
    if(!backward && !g.cross && g.qbias>=0)value=w[g.qbias+part*g.ed+d];
    for(int f=0;f<features;++f){
        const int wi=!g.cross?(part*g.ed+d)*g.ed+f:(part?g.ed*g.ed+f*2*g.ed+(part-1)*g.ed+d:f*g.ed+d);
        value+=input[t*features+f]*w[wi];
    }result[i]=value;
}
__global__ void attention_probabilities(const float* q,const float* k,float* p,CudaOperationGeometry::Attention g){
    __shared__ float scratch[threads];const int h=blockIdx.x/g.q,t=blockIdx.x%g.q,width=g.ed/g.heads,base=(h*g.q+t)*g.k;
    const float scale=rsqrtf(static_cast<float>(width));float maximum=-1e30f;
    for(int j=threadIdx.x;j<g.k;j+=threads){float score=-1e9f;if(!g.causal || j<=t){score=0;for(int d=0;d<width;++d)score+=q[t*g.ed+h*width+d]*k[j*g.ed+h*width+d];score*=scale;}p[base+j]=score;maximum=fmaxf(maximum,score);}
    maximum=reduce_max(maximum,scratch);float sum=0;
    for(int j=threadIdx.x;j<g.k;j+=threads){const float v=expf(p[base+j]-maximum);p[base+j]=v;sum+=v;}
    const float inv=1.f/(reduce_sum(sum,scratch)+1e-9f);for(int j=threadIdx.x;j<g.k;j+=threads)p[base+j]*=inv;
}
__global__ void attention_context(const float* p,const float* v,float* ctx,CudaOperationGeometry::Attention g,int n){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const int t=i/g.ed,d=i%g.ed,h=d/(g.ed/g.heads);float sum=0;
    for(int j=0;j<g.k;++j)sum+=p[(h*g.q+t)*g.k+j]*v[j*g.ed+d];ctx[i]=sum;
}
__global__ void attention_output(const float* ctx,const float* w,const float* dy,float* y,CudaOperationGeometry::Attention g,int n){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const int t=i/g.ed,d=i%g.ed,offset=g.mainParameters-g.ed*g.ed;float sum=0;
    if(!dy && g.obias>=0)sum=w[g.obias+d];
    for(int f=0;f<g.ed;++f)sum+=dy?dy[t*g.ed+f]*w[offset+d*g.ed+f]:ctx[t*g.ed+f]*w[offset+f*g.ed+d];y[i]=sum;
}
__global__ void attention_score_grad(const float* p,const float* v,const float* dc,float* ds,CudaOperationGeometry::Attention g){
    __shared__ float scratch[threads];const int h=blockIdx.x/g.q,t=blockIdx.x%g.q,width=g.ed/g.heads,base=(h*g.q+t)*g.k;float dot=0;
    for(int j=threadIdx.x;j<g.k;j+=threads){float dp=0;for(int f=0;f<width;++f)dp+=dc[t*g.ed+h*width+f]*v[j*g.ed+h*width+f];ds[base+j]=dp;dot+=dp*p[base+j];}
    dot=reduce_sum(dot,scratch);for(int j=threadIdx.x;j<g.k;j+=threads)ds[base+j]=g.causal && j>t?0.f:(ds[base+j]-dot)*p[base+j];
}
__global__ void attention_projected_grad(const float* q,const float* k,const float* dc,const float* p,const float* ds,float* result,
    CudaOperationGeometry::Attention g,int part,int n){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const int t=i/g.ed,d=i%g.ed,h=d/(g.ed/g.heads);float sum=0;
    if(part==0)for(int j=0;j<g.k;++j)sum+=ds[(h*g.q+t)*g.k+j]*k[j*g.ed+d];
    else if(part==1)for(int j=0;j<g.q;++j)sum+=ds[(h*g.q+j)*g.k+t]*q[j*g.ed+d];
    else for(int j=0;j<g.q;++j)sum+=p[(h*g.q+j)*g.k+t]*dc[j*g.ed+d];
    result[i]=part==2?sum:sum*rsqrtf(static_cast<float>(g.ed/g.heads));
}
__global__ void attention_weight_grad(const float* x,const float* kv,const float* ctx,const float* dy,const float* dq,const float* dk,const float* dv,
    float* dw,CudaOperationGeometry::Attention g,int count){
    const int wi=blockIdx.x*blockDim.x+threadIdx.x;if(wi>=count)return;float sum=0;const int output_offset=g.mainParameters-g.ed*g.ed;
    if(wi>=output_offset){const int r=(wi-output_offset)/g.ed,d=(wi-output_offset)%g.ed;for(int t=0;t<g.q;++t)sum+=ctx[t*g.ed+r]*dy[t*g.ed+d];}
    else if(!g.cross){const int part=(wi/g.ed)/g.ed,d=(wi/g.ed)%g.ed,f=wi%g.ed;const float* grad=part==0?dq:(part==1?dk:dv);for(int t=0;t<g.q;++t)sum+=x[t*g.ed+f]*grad[t*g.ed+d];}
    else if(wi<g.ed*g.ed){const int f=wi/g.ed,d=wi%g.ed;for(int t=0;t<g.q;++t)sum+=x[t*g.ed+f]*dq[t*g.ed+d];}
    else{const int off=wi-g.ed*g.ed,f=off/(2*g.ed),col=off%(2*g.ed),d=col%g.ed;const float* grad=col<g.ed?dk:dv;for(int t=0;t<g.k;++t)sum+=kv[t*g.ked+f]*grad[t*g.ed+d];}
    dw[wi]=sum;
}
__global__ void attention_input_grad(const float* w,const float* dq,const float* dk,const float* dv,float* dx,CudaOperationGeometry::Attention g,bool second,int count){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const int width=second?g.ked:g.ed,t=i/width,f=i%width;float sum=0;
    if(!g.cross)for(int d=0;d<g.ed;++d)sum+=dq[t*g.ed+d]*w[d*g.ed+f]+dk[t*g.ed+d]*w[(g.ed+d)*g.ed+f]+dv[t*g.ed+d]*w[(2*g.ed+d)*g.ed+f];
    else if(!second)for(int d=0;d<g.ed;++d)sum+=dq[t*g.ed+d]*w[f*g.ed+d];
    else for(int d=0;d<g.ed;++d)sum+=dk[t*g.ed+d]*w[g.ed*g.ed+f*2*g.ed+d]+dv[t*g.ed+d]*w[g.ed*g.ed+f*2*g.ed+g.ed+d];
    dx[i]=sum;
}
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
            if(!dy && !g.cross && g.qbias>=0)attention_add_bias<<<(n+threads-1)/threads,threads>>>(result,w.ptr,n,g.ed,g.qbias+part*g.ed);
        }else attention_project<<<(n+threads-1)/threads,threads>>>(g.cross && part?kv.ptr:x.ptr,w.ptr,result,g,part,dy!=nullptr,n);
    }
    attention_probabilities<<<g.heads*g.q,threads>>>(q.ptr,k.ptr,p.ptr,g);
    if(handle){
        const int width=g.ed/g.heads;
        for(int h=0;h<g.heads;++h)if(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,width,g.q,g.k,&one,v.ptr+h*width,g.ed,p.ptr+static_cast<size_t>(h)*g.q*g.k,g.k,&zero,ctx.ptr+h*width,g.ed)!=CUBLAS_STATUS_SUCCESS)return false;
    }else attention_context<<<(nq+threads-1)/threads,threads>>>(p.ptr,v.ptr,ctx.ptr,g,nq);
    // CPU backward recomputes attention without bias and leaves its bias gradients untouched.
    // Preserve that compatibility contract while forward supports both bias layouts.
    if(!dy){if(!y.alloc(nq))return false;
        if(handle){if(cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,g.ed,g.q,g.ed,&one,w.ptr+g.mainParameters-g.ed*g.ed,g.ed,ctx.ptr,g.ed,&zero,y.ptr,g.ed)!=CUBLAS_STATUS_SUCCESS)return false;
            if(g.obias>=0)attention_add_bias<<<(nq+threads-1)/threads,threads>>>(y.ptr,w.ptr,nq,g.ed,g.obias);
        }else attention_output<<<(nq+threads-1)/threads,threads>>>(ctx.ptr,w.ptr,nullptr,y.ptr,g,nq);
        if(!finished())return false;
        out={std::vector<float>(nq)};return y.download(out[0].data(),nq);}
    if(!grad.alloc(nq) || !grad.upload(dy->data(),nq) || !dc.alloc(nq) || !ds.alloc(np) || !dq.alloc(nq) || !dk.alloc(nk) || !dv.alloc(nk) ||
        !weights_grad.alloc(l.getWeightsSize()) || cudaMemset(weights_grad.ptr,0,l.getWeightsSize()*sizeof(float))!=cudaSuccess || !dx.alloc(in[0]->size()))return false;
    if(g.cross && !dkv.alloc(in[1]->size()))return false;
    if(handle){if(cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,g.ed,g.q,g.ed,&one,w.ptr+g.mainParameters-g.ed*g.ed,g.ed,grad.ptr,g.ed,&zero,dc.ptr,g.ed)!=CUBLAS_STATUS_SUCCESS)return false;}
    else attention_output<<<(nq+threads-1)/threads,threads>>>(ctx.ptr,w.ptr,grad.ptr,dc.ptr,g,nq);
    attention_score_grad<<<g.heads*g.q,threads>>>(p.ptr,v.ptr,dc.ptr,ds.ptr,g);
    for(int part=0;part<3;++part){const size_t n=part?nk:nq;attention_projected_grad<<<(n+threads-1)/threads,threads>>>(q.ptr,k.ptr,dc.ptr,p.ptr,ds.ptr,part==0?dq.ptr:(part==1?dk.ptr:dv.ptr),g,part,n);}
    attention_weight_grad<<<(g.mainParameters+threads-1)/threads,threads>>>(x.ptr,kv.ptr,ctx.ptr,grad.ptr,dq.ptr,dk.ptr,dv.ptr,weights_grad.ptr,g,g.mainParameters);
    attention_input_grad<<<(in[0]->size()+threads-1)/threads,threads>>>(w.ptr,dq.ptr,dk.ptr,dv.ptr,dx.ptr,g,false,in[0]->size());
    if(g.cross)attention_input_grad<<<(in[1]->size()+threads-1)/threads,threads>>>(w.ptr,dq.ptr,dk.ptr,dv.ptr,dkv.ptr,g,true,in[1]->size());
    if(!finished())return false;
    out={std::vector<float>(in[0]->size())};if(!dx.download(out[0].data(),out[0].size()))return false;
    if(g.cross){out.emplace_back(in[1]->size());if(!dkv.download(out[1].data(),out[1].size()))return false;}
    dw->resize(l.getWeightsSize());return weights_grad.download(dw->data(),dw->size());
}
} // namespace CudaNativeOps
