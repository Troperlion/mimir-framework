#include "Layers.hpp"
#include "runtimes/LayerOps.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/cpu/MatMulKernels.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
static void check(int batch,int m,int k,int n) {
    Layer layer("batch","BatchMatMul",0);
    layer.seq_len=batch;layer.in_features=m;layer.out_features=k;layer.embed_dim=n;
    std::vector<float> a(size_t(batch)*m*k),b(size_t(batch)*k*n);
    for(size_t i=0;i<a.size();++i)a[i]=float(int(i%31)-15)/31;
    for(size_t i=0;i<b.size();++i)b[i]=float(int(i%37)-18)/37;
    CpuRuntime cpu;cpu.initialize(RuntimeConfig{});
    std::vector<std::vector<float>> result;
    if(!cpu.forwardLayer({&a,&b},result,layer,false)||result.size()!=1)
        throw std::runtime_error("BatchMatMul dispatch failed");
    std::vector<float> direct(size_t(batch)*m*n,-99);
    RuntimeMatMul::forwardBatched(a.data(),b.data(),direct.data(),batch,m,k,n);
    for(int z=0;z<batch;++z)for(int i=0;i<m;++i)for(int j=0;j<n;++j) {
        double ref=0;
        for(int c=0;c<k;++c)ref+=double(a[(size_t(z)*m+i)*k+c])*b[(size_t(z)*k+c)*n+j];
        const size_t index=(size_t(z)*m+i)*n+j;
        for(float v:{result[0][index],direct[index]})
            if(std::abs(double(v)-ref)>2e-5*(1+std::abs(ref)))
                throw std::runtime_error("BatchMatMul differs from double reference");
    }
    b.pop_back();
    if(cpu.forwardLayer({&a,&b},result,layer,false))throw std::runtime_error("Invalid B accepted");
    if(cpu.forwardLayer({&a,nullptr},result,layer,false))throw std::runtime_error("Null B accepted");
    layer.seq_len=0;
    if(cpu.forwardLayer({&a,&b},result,layer,false))throw std::runtime_error("Zero batch accepted");
}
int main() {
    for(bool accelerated:{false,true}) {
        RuntimeLayerOps::setHardwareAcceleration(accelerated);
        check(1,1,1,1);check(3,7,13,7);check(2,11,17,8);check(3,13,29,23);
        check(3,17,31,24);check(2,19,37,25);check(4,31,67,49);check(8,65,67,71);
    }
    std::cout<<"BATCH_MATMUL_FORWARD_OK\n";
}
