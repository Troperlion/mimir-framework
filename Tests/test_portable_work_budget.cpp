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
    int calls = 0;
    bool run(std::vector<float>&, const std::vector<int>&) override { ++calls; return true; }
};
int main() {
    using namespace PortableStaging;
    require(referenceDispatchFits(128, {34, 256, 1}));
    require(!referenceDispatchFits(3 * 1024 * 1024, {34, 256, 12288}));
    require(!referenceDispatchFits(16384, {34, INT_MAX, INT_MAX}));
    require(!referenceDispatchFits(128, {34, 0, 1}));
    require(referenceDispatchFits(1024, {39, 1, 1, 1, 2, 3, 8, 8, 8}));
    require(!referenceDispatchFits(1024, {39, 1, 1, 1, 2, 3, 256, 256, 256}));
    require(!referenceDispatchFits(1024, {39, 1, 1, 1, 2, 3, INT_MAX, INT_MAX, INT_MAX}));
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
    launch("cuda_unary_kernel", 1, 256, buffer, buffer, 128, 0, 0.f);
    require(counter->calls == 1 && cudaGetLastError() == cudaSuccess);
    launch("cuda_unary_kernel", INT_MAX, INT_MAX, buffer, buffer, 128, 0, 0.f);
    require(counter->calls == 1 && cudaGetLastError() != cudaSuccess);
    context.failed = false;
    launch("portable_gemm", 1, 1, buffer, buffer, buffer, 256, 256, 256, 256, 256, 256, false, false, 1.f, 0.f);
    require(counter->calls == 1 && cudaGetLastError() != cudaSuccess);
    cudaFree(buffer); context.failed = false;
    require(cudaMalloc(&buffer, image.size() * sizeof(float)) == cudaSuccess);
    launch("cuda_unary_kernel", 12288, 256, buffer, buffer, int(image.size()), 0, 0.f);
    require(counter->calls == 1 && cudaGetLastError() != cudaSuccess);
    cudaFree(buffer); current = nullptr;
    for (const char* backend : {"VULKAN", "OPENCL"}) {
        PortableRuntime runtime(backend);
        Layer layer; layer.type_enum = LayerType::Identity;
        require(runtime.queryForwardOperationCapability(layer, {&image}, true) == RuntimeCapabilityLevel::Unsupported);
        require(runtime.queryBackwardOperationCapability(layer, {&small}, {&image}, true) == RuntimeCapabilityLevel::Unsupported);
        // No initialization/device is needed: rejection happens before GPU access.
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
    require(geometry.queryForwardOperationCapability(relu, {&image}, true) == RuntimeCapabilityLevel::Unsupported);
    std::cout << "PORTABLE_WORK_BUDGET_OK\n";
}
