#include "Layers.hpp"
#include "runtimes/cpu/MatMulKernels.hpp"
#include "runtimes/LayerOps.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
static void check(int m, int k, int n, int batch) {
    Layer layer("mm", batch == 1 ? "MatMul" : "BatchMatMul", 0);
    layer.in_features=m; layer.out_features=k; layer.embed_dim=n; layer.seq_len=batch;
    std::vector<float> a(size_t(batch)*m*k), b(size_t(batch)*k*n), dy(size_t(batch)*m*n);
    for(size_t i=0;i<a.size();++i) a[i]=float(int(i%31)-15)/31;
    for(size_t i=0;i<b.size();++i) b[i]=float(int(i%37)-18)/37;
    for(size_t i=0;i<dy.size();++i) dy[i]=float(int(i%41)-20)/41;
    CpuRuntime cpu; cpu.initialize(RuntimeConfig{});
    std::vector<std::vector<float>> result;
    if(!cpu.backwardLayer({&a,&b},{&dy},result,layer,true) || result.size()!=2)
        throw std::runtime_error("MatMul dispatch failed");
    std::vector<std::vector<float>> direct{std::vector<float>(a.size()),std::vector<float>(b.size())};
    for(int z=0;z<batch;++z)
        RuntimeMatMul::backward(a.data()+size_t(z)*m*k,b.data()+size_t(z)*k*n,
            dy.data()+size_t(z)*m*n,direct[0].data()+size_t(z)*m*k,
            direct[1].data()+size_t(z)*k*n,m,k,n);
    auto close=[](float x, double ref) {
        if(std::abs(double(x)-ref)>2e-5*(1+std::abs(ref))) throw std::runtime_error("MatMul gradient mismatch");
    };
    for(int z=0;z<batch;++z) {
        for(int i=0;i<m;++i) for(int c=0;c<k;++c) {
            double ref=0;
            for(int j=0;j<n;++j) ref+=double(dy[(z*m+i)*n+j])*b[(z*k+c)*n+j];
            close(result[0][(z*m+i)*k+c],ref);
            close(direct[0][(z*m+i)*k+c],ref);
        }
        for(int c=0;c<k;++c) for(int j=0;j<n;++j) {
            double ref=0;
            for(int i=0;i<m;++i) ref+=double(a[(z*m+i)*k+c])*dy[(z*m+i)*n+j];
            close(result[1][(z*k+c)*n+j],ref);
            close(direct[1][(z*k+c)*n+j],ref);
        }
    }
    b.pop_back();
    if(cpu.backwardLayer({&a,&b},{&dy},result,layer,true)) throw std::runtime_error("Invalid shape accepted");
}
int main() {
    for(bool accelerate : {false,true}) {
        RuntimeLayerOps::setHardwareAcceleration(accelerate);
        check(1,1,1,1); check(3,7,23,2); check(17,13,25,3);
        check(31,37,49,1); check(65,67,71,2);
    }
    std::cout << "MATMUL_BACKWARD_OK\n";
}
