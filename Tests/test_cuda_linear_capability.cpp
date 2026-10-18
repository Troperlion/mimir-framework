#include "runtimes/cuda/CudaRuntime.hpp"
#include "Layers.hpp"
#include <stdexcept>
#include <iostream>
// Exercise real metadata checks without requiring a CUDA device.
int main() {
    CudaRuntime runtime;
    RuntimeConfig config;config.linear_enabled=true;
    (void)runtime.initialize(config);
    Layer layer("linear","Linear",15);
    layer.in_features=4;layer.out_features=3;layer.seq_len=2;layer.use_bias=true;
    layer.weights.assign(15,.1f);
    std::vector<float> x(8),good(6),short_grad(5),long_grad(7);
    auto check=[&](bool valid,const std::vector<const std::vector<float>*>& inputs,
                   const std::vector<const std::vector<float>*>& grads) {
        const auto capability=runtime.queryBackwardOperationCapability(layer,inputs,grads,true);
        if((capability!=RuntimeCapabilityLevel::Unsupported)!=valid)
            throw std::runtime_error("CUDA Linear shape capability mismatch");
    };
    check(true,{&x},{&good});check(false,{&x},{&short_grad});check(false,{&x},{&long_grad});
    check(false,{&x},{&good,&good});check(false,{&x},{nullptr});check(false,{&x,&x},{&good});
    layer.seq_len=0;check(true,{&x},{&good});
    x.resize(9);check(false,{&x},{&good});x.resize(8);
    layer.in_features=0;check(false,{&x},{&good});layer.in_features=4;
    layer.weights.resize(12);check(false,{&x},{&good});
    layer.use_bias=false;check(true,{&x},{&good});
    std::cout<<"CUDA_LINEAR_CAPABILITY_OK\n";
}
