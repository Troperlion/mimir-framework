#include "runtimes/cpu/ConvolutionKernels.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

static void require(bool value) { if(!value) throw std::runtime_error("convolution scratch regression"); }
static void near(float a,float b) { require(std::abs(a-b)<3e-5f*std::max(1.0f,std::abs(b))); }
static void check(int h,int width,int in,int out,int kernel,int stride,int pad,int dilation,bool bias) {
    auto& guard=MemoryGuard::instance();
    require(guard.getCurrentBytes()==0);guard.reset();
    const int oh=(h+2*pad-dilation*(kernel-1)-1)/stride+1;
    const int ow=(width+2*pad-dilation*(kernel-1)-1)/stride+1;
    const int spatial=oh*ow,k=in*kernel*kernel;
    Layer layer("conv","Conv2d",0);layer.use_bias=bias;
    layer.weights.resize(size_t(out)*k+(bias?out:0));
    for(size_t i=0;i<layer.weights.size();++i)layer.weights[i]=float(int(i%17)-8)/128;
    std::vector<float> x(size_t(in)*h*width),g(size_t(out)*spatial),reference(g.size()),dx_ref(x.size()),dw_ref(size_t(out)*k),db_ref(bias?out:0);
    for(size_t i=0;i<x.size();++i)x[i]=float(int(i%19)-9)/32;
    for(size_t i=0;i<g.size();++i)g[i]=float(int(i%23)-11)/64;
    for(int oc=0;oc<out;++oc)for(int y=0;y<oh;++y)for(int z=0;z<ow;++z) {
        const size_t oi=size_t(oc)*spatial+y*ow+z;
        float sum=0;
        for(int ic=0;ic<in;++ic)for(int kh=0;kh<kernel;++kh)for(int kw=0;kw<kernel;++kw) {
            const int iy=y*stride-pad+kh*dilation,ix=z*stride-pad+kw*dilation;
            if(iy<0||ix<0||iy>=h||ix>=width)continue;
            const size_t ii=(size_t(ic)*h+iy)*width+ix,wi=((size_t(oc)*in+ic)*kernel+kh)*kernel+kw;
            sum+=x[ii]*layer.weights[wi];dx_ref[ii]+=g[oi]*layer.weights[wi];dw_ref[wi]+=g[oi]*x[ii];
        }
        if(bias){sum+=layer.weights[size_t(out)*k+oc];db_ref[oc]+=g[oi];}
        reference[oi]=sum;
    }
    std::vector<float> y,dx;
    CpuConvolution::forward(x,y,layer.getWeights(),bias?layer.getWeights()+size_t(out)*k:nullptr,h,width,in,out,kernel,stride,pad,dilation);
    require(y.size()==reference.size());for(size_t i=0;i<y.size();++i)near(y[i],reference[i]);
    if(kernel==1&&stride==1&&pad==0)require(guard.getAllocationsCount()==0);
    layer.grad_weights.assign(layer.weights.size(),.125f);layer.grad_bias.assign(bias?out:0,.25f);
    for(int repeat=1;repeat<=2;++repeat) {
        CpuConvolution::backward(x,g,dx,layer,h,width,in,out,kernel,stride,pad,dilation);
        for(size_t i=0;i<dx.size();++i)near(dx[i],dx_ref[i]);
        for(size_t i=0;i<dw_ref.size();++i)near(layer.grad_weights[i],.125f+repeat*dw_ref[i]);
        for(size_t i=0;i<db_ref.size();++i){near(layer.grad_weights[size_t(out)*k+i],.125f+repeat*db_ref[i]);near(layer.grad_bias[i],.25f+repeat*db_ref[i]);}
        require(guard.getCurrentBytes()==0);
    }
    const int tile=CpuConvolution::tile_size(k,out,spatial);
    require(guard.getPeakBytes()<=(size_t(tile)*(k+2*out)+32)*sizeof(float));
}
int main() {
    {
        std::vector<float> output(2*31,7.0f);
        HardwareOpt::matmul_fma_saturated(output.data(),nullptr,nullptr,2,31,0);
        for(float value:output)require(value==0.0f);
        std::fill(output.begin(),output.end(),.125f);
        HardwareOpt::matmul_add_fma_saturated(output.data(),nullptr,nullptr,2,31,0);
        for(float value:output)require(value==.125f);
        SIMD::matmul_transpose_avx2(output.data(),nullptr,nullptr,2,31,0);
        for(float value:output)require(value==0.0f);
    }
    for(bool accelerated:{false,true}) {
        RuntimeLayerOps::setHardwareAcceleration(accelerated);
        for(bool bias:{false,true}) {
            check(9,13,3,5,1,1,0,1,bias);
            check(9,13,3,5,1,2,0,1,bias);
            check(9,13,3,5,3,1,1,1,bias);
            check(9,13,3,5,3,2,2,2,bias);
            check(31,35,2,3,3,1,1,1,bias); // second im2col tile and tail
        }
    }
    // The old pair of im2col/dCol buffers plus dW exceeded this scratch budget.
    {
        auto& guard=MemoryGuard::instance();guard.reset();
        const size_t previous_limit=guard.getLimit();guard.setLimit(6u<<20);
        Layer layer("bounded","Conv2d",0);layer.use_bias=true;
        layer.weights.assign(size_t(256)*128*9+256,.01f);
        std::vector<float> x(size_t(128)*32*32,.02f),go(size_t(256)*32*32,.01f),dx;
        CpuConvolution::backward(x,go,dx,layer,32,32,128,256,3,1,1,1);
        require(dx.size()==x.size()&&guard.getCurrentBytes()==0);
        require(guard.getPeakBytes()<=(6u<<20));
        guard.setLimit(previous_limit);
    }
    std::cout<<"CONVOLUTION_SCRATCH_OK\n";
}
