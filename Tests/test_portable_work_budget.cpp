#include "runtimes/portable/PortableWorkBudget.hpp"
#include "runtimes/portable/PortableRuntime.hpp"
#include "Layers.hpp"
#include "runtimes/portable/PortableStaging.hpp"
#include <iostream>
#include <stdexcept>
#include <climits>

static void check(bool value, int line) {
    if (!value) throw std::runtime_error("Serial GPU workload budget regression at line " + std::to_string(line));
}
#define require(value) check((value), __LINE__)
struct CountingDevice : PortableStaging::Device {
    int calls = 0,batches=0;
    bool runBatch(std::vector<float>& m,const std::vector<std::vector<int>>& q,const std::vector<PortableStaging::MemoryRange>& ranges={}) override {++batches;return Device::runBatch(m,q,ranges);}
    bool run(std::vector<float>&, const std::vector<int>&) override { ++calls; return true; }
};
int main() {
    using namespace PortableStaging;
    require(referenceDispatchFits(128, {34, 256, 1}));
    require(referenceDispatchFits(4096, {41, 1, 1024}));
    require(referenceDispatchFits(4096, {42, 1, 1024}));
    require(!referenceDispatchFits(referenceMaxElements+1, {41, 1, 1024}));
    require(!referenceDispatchFits(3 * 1024 * 1024, {34, 256, 12288}));
    require(!referenceDispatchFits(16384, {34, INT_MAX, INT_MAX}));
    require(!referenceDispatchFits(128, {34, 0, 1}));
    require(referenceDispatchFits(1024, {39, 1, 1, 1, 2, 3, 8, 8, 8}));
    require(!referenceDispatchFits(1024, {39, 1, 1, 1, 2, 3, 256, 256, 256}));
    require(!referenceDispatchFits(1024, {39, 1, 1, 1, 2, 3, INT_MAX, INT_MAX, INT_MAX}));
    require(parallelDispatchFits(3*1024*1024,{36,256,12288}));
    require(!parallelDispatchFits(65*1024*1024,{36,256,12288}));
    require(!parallelDispatchFits(1024,{39,1,1,1,2,3,8,8,65537}));
    require(!parallelDispatchFits(1024,{10,1,1,1,2,3,4,5,1048577}));
    require(!parallelDispatchFits(1024,{42,1,1024})); // malformed scatter arguments are rejected
    std::vector<float> image(3 * 1024 * 1024), small(16);
    require(referenceInputsFit(16, {&small}));
    require(!referenceInputsFit(0, {&image}));
    require(!referenceInputsFit(referenceMaxElements, {&small}));
    Context& context = contexts["TEST"];
    auto device = std::make_unique<CountingDevice>();
    auto* counter = device.get(); context.device = std::move(device);
    current = &context;
    float* buffer = nullptr;
    require(cudaMalloc(&buffer, 128 * sizeof(float)) == cudaSuccess);
    float host[129]{};
    require(cudaMemcpy(buffer,host,sizeof(host),cudaMemcpyHostToDevice)!=cudaSuccess);
    context.failed=false;
    require(cudaMemcpy(host,buffer+127,2*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess);
    context.failed=false;
    require(cudaMemset(buffer+128,0,sizeof(float))!=cudaSuccess);
    context.failed=false;
    require(cudaMemcpy(buffer+127,host,sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess);
    launch("cuda_unary_kernel", 1, 256, buffer, buffer, 128, 0, 0.f);
    require(counter->calls == 0 && cudaGetLastError() == cudaSuccess);
    launch("cuda_unary_kernel",1,256,buffer,buffer,128,0,0.f);
    require(counter->calls==0 && context.pending.size()==2);
    require(cudaDeviceSynchronize()==cudaSuccess && counter->calls==2 && counter->batches==1);
    launch("cuda_unary_kernel", INT_MAX, INT_MAX, buffer, buffer, 128, 0, 0.f);
    require(counter->calls == 2 && cudaGetLastError() != cudaSuccess);
    context.failed = false;
    launch("portable_gemm", 1, 1, buffer, buffer, buffer, 256, 256, 256, 256, 256, 256, false, false, 1.f, 0.f);
    require(counter->calls == 2 && cudaGetLastError() == cudaSuccess);
    require(cudaDeviceSynchronize()==cudaSuccess && counter->calls==3);
    cudaFree(buffer); context.failed = false;
    require(cudaMalloc(&buffer, image.size() * sizeof(float)) == cudaSuccess);
    launch("cuda_unary_kernel", 12288, 256, buffer, buffer, int(image.size()), 0, 0.f);
    require(counter->calls == 3 && cudaGetLastError() == cudaSuccess);
    require(cudaDeviceSynchronize()==cudaSuccess && counter->calls==4);
    cudaFree(buffer); current = nullptr;
    for (const char* backend : {"VULKAN", "OPENCL"}) {
        PortableRuntime runtime(backend);
        Layer layer; layer.type_enum = LayerType::Identity;
        require(runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(layer, {&image}, true)));
        require(runtime.queryBackwardOperationCapability(layer, {&small}, {&image}, true) == RuntimeCapabilityLevel::Unsupported);
        // Uninitialized operations still fail before touching a GPU or output buffers.
        std::vector<std::vector<float>> output{{42}};
        require(!runtime.forwardLayer({&image}, output, layer, true));
        require(output == std::vector<std::vector<float>>{{42}});
        require(!runtime.backwardLayer({&small}, {&image}, output, layer, true));
        require(output == std::vector<std::vector<float>>{{42}});
    }
    Layer relu; relu.type_enum = LayerType::ReLU;
    PortableRuntime geometry("VULKAN"); RuntimeConfig config;
    config.linear_enabled = true; config.linear_min_ops = 0; geometry.configure(config);
    require(runtimeCapabilityIsNative(geometry.queryForwardGeometryCapability(relu, {&image}, true)));
    require(runtimeCapabilityIsNative(geometry.queryForwardOperationCapability(relu, {&image}, true)));
    std::cout << "PORTABLE_WORK_BUDGET_OK\n";
}
