#pragma once
#include "SIMD_Ops.hpp"
#include "RuntimeAllocator.hpp"
#include "runtimes/LayerOps.hpp"

namespace CpuConvolution {
// Bounded im2col tiles keep the model's GEMM path inside the CPU runtime.
inline void multiply(float* out,const float* a,const float* b,int m,int n,int k,bool transpose_b) {
    if (RuntimeLayerOps::hardwareAccelerationEnabled()) {
        if (transpose_b) SIMD::matmul_transpose_avx2(out,a,b,m,n,k);
        else SIMD::matmul_avx2(out,a,b,m,n,k);
    } else {
        for(int i=0;i<m;++i) for(int j=0;j<n;++j) {
            float sum=0;
            for(int t=0;t<k;++t) sum+=a[static_cast<size_t>(i)*k+t]*b[transpose_b?static_cast<size_t>(j)*k+t:static_cast<size_t>(t)*n+j];
            out[static_cast<size_t>(i)*n+j]=sum;
        }
    }
}
// C += A * B without a full m*n intermediate, using the original GEMM
// reduction order and bounded register accumulators.
inline void multiply_add(float* out,const float* a,const float* b,int m,int n,int k) {
    if(!RuntimeLayerOps::hardwareAccelerationEnabled()) {
        for(int row=0;row<m;++row) for(int column=0;column<n;++column) {
            float sum=0;
            for(int t=0;t<k;++t) sum+=a[static_cast<size_t>(row)*k+t]*b[static_cast<size_t>(t)*n+column];
            out[static_cast<size_t>(row)*n+column]+=sum;
        }
        return;
    }
    HardwareOpt::matmul_add_fma_saturated(out,a,b,m,n,k);
}
inline void pack(const float* x,float* col,int start,int count,int h,int w,int channels,int kernel,int stride,int pad,int dilation,int out_w) {
    const int k=channels*kernel*kernel;
    for(int r=0;r<count;++r) {
        const int oh=(start+r)/out_w,ow=(start+r)%out_w;
        for(int c=0;c<channels;++c) for(int kh=0;kh<kernel;++kh) for(int kw=0;kw<kernel;++kw) {
            const int ih=oh*stride-pad+kh*dilation,iw=ow*stride-pad+kw*dilation;
            col[static_cast<size_t>(r)*k+(c*kernel+kh)*kernel+kw]=
                ih>=0&&ih<h&&iw>=0&&iw<w?x[(static_cast<size_t>(c)*h+ih)*w+iw]:0.0f;
        }
    }
}
inline int tile_size(int k,int out_c,int spatial) {
    return std::max(1,std::min(spatial,static_cast<int>(std::min<size_t>(1024, (4u<<20)/(sizeof(float)*(static_cast<size_t>(k)+out_c))))));
}
inline void forward(const std::vector<float>& x,std::vector<float>& y,const float* weights,const float* bias,
                    int h,int w,int in_c,int out_c,int kernel,int stride,int pad,int dilation) {
    const int oh=(h+2*pad-dilation*(kernel-1)-1)/stride+1;
    const int ow=(w+2*pad-dilation*(kernel-1)-1)/stride+1;
    if(oh<=0||ow<=0) throw std::invalid_argument("Conv2d: invalid output dimensions");
    const int spatial=oh*ow,k=in_c*kernel*kernel,tile=tile_size(k,out_c,spatial);
    // CHW 1x1 is already a matrix: no im2col, temporary rows or layout copy.
    // Wide, long rows keep the packed path to avoid strided cache conflicts.
    if(kernel==1 && stride==1 && pad==0 && (in_c<=32 || spatial<1024)) {
        y.resize(static_cast<size_t>(out_c)*spatial);
        multiply(y.data(),weights,x.data(),out_c,spatial,in_c,false);
        if(bias) for(int c=0;c<out_c;++c) {
            float* row=y.data()+static_cast<size_t>(c)*spatial;
            for(int i=0;i<spatial;++i) row[i]+=bias[c];
        }
        return;
    }
    auto& guard=MemoryGuard::instance();RuntimeAllocator allocator(guard,guard.getLimit()/(1024*1024));
    auto col=allocator.get_scratchpad_uninitialized(static_cast<size_t>(tile)*k*sizeof(float),"conv/im2col");
    auto rows=allocator.get_scratchpad_uninitialized(static_cast<size_t>(tile)*out_c*sizeof(float),"conv/output_rows");
    y.resize(static_cast<size_t>(out_c)*spatial);
    for(int start=0;start<spatial;start+=tile) {
        const int count=std::min(tile,spatial-start);
        pack(x.data(),col.data(),start,count,h,w,in_c,kernel,stride,pad,dilation,ow);
        multiply(rows.data(),col.data(),weights,count,out_c,k,true);
        for(int r=0;r<count;++r) for(int c=0;c<out_c;++c)
            y[static_cast<size_t>(c)*spatial+start+r]=rows.data()[static_cast<size_t>(r)*out_c+c]+(bias?bias[c]:0.0f);
    }
}
inline void backward(const std::vector<float>& x,const std::vector<float>& go,std::vector<float>& dx,Layer& layer,
                     int h,int w,int in_c,int out_c,int kernel,int stride,int pad,int dilation) {
    const int oh=(h+2*pad-dilation*(kernel-1)-1)/stride+1;
    const int ow=(w+2*pad-dilation*(kernel-1)-1)/stride+1;
    const int spatial=oh*ow,k=in_c*kernel*kernel,tile=tile_size(k,out_c,spatial);
    auto& guard=MemoryGuard::instance();RuntimeAllocator allocator(guard,guard.getLimit()/(1024*1024));
    auto col=allocator.get_scratchpad_uninitialized(static_cast<size_t>(tile)*k*sizeof(float),"conv/bwd_im2col");
    auto dy=allocator.get_scratchpad_uninitialized(static_cast<size_t>(tile)*out_c*sizeof(float),"conv/dy");
    auto dy_t=allocator.get_scratchpad_uninitialized(static_cast<size_t>(tile)*out_c*sizeof(float),"conv/dy_transpose");
    dx.assign(x.size(),0.0f);
    if(layer.grad_weights.size()!=layer.getWeightsSize()) layer.grad_weights.assign(layer.getWeightsSize(),0.0f);
    if(layer.use_bias && layer.grad_bias.size()!=static_cast<size_t>(out_c)) layer.grad_bias.assign(out_c,0.0f);
    for(int start=0;start<spatial;start+=tile) {
        const int count=std::min(tile,spatial-start);
        pack(x.data(),col.data(),start,count,h,w,in_c,kernel,stride,pad,dilation,ow);
        for(int r=0;r<count;++r) for(int c=0;c<out_c;++c) {
            const float g=go[static_cast<size_t>(c)*spatial+start+r];
            dy.data()[static_cast<size_t>(r)*out_c+c]=g;
            dy_t.data()[static_cast<size_t>(c)*count+r]=g;
            if(layer.use_bias) { layer.grad_weights[static_cast<size_t>(out_c)*k+c]+=g;layer.grad_bias[c]+=g; }
        }
        multiply_add(layer.grad_weights.data(),dy_t.data(),col.data(),out_c,k,count);
        // The packed input is dead after dW. Its allocation now holds dCol.
        multiply(col.data(),dy.data(),layer.getWeights(),count,k,out_c,false);
        for(int r=0;r<count;++r) {
            const int row=(start+r)/ow,column=(start+r)%ow;
            for(int c=0;c<in_c;++c) for(int kh=0;kh<kernel;++kh) for(int kw=0;kw<kernel;++kw) {
                const int ih=row*stride-pad+kh*dilation,iw=column*stride-pad+kw*dilation;
                if(ih>=0&&ih<h&&iw>=0&&iw<w)
                    dx[(static_cast<size_t>(c)*h+ih)*w+iw]+=col.data()[static_cast<size_t>(r)*k+(c*kernel+kh)*kernel+kw];
            }
        }
    }
}
} // namespace CpuConvolution
