#include "test_utils.hpp"

#include "Layers.hpp"
#include "Model.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/opencl/OpenCLRuntime.hpp"
#include "runtimes/vulkan/VulkanRuntime.hpp"

class CapabilityRuntime final : public AbstractRuntime {
public:
    CapabilityRuntime(const char* runtime_name,
                      RuntimeCapabilityLevel forward,
                                            RuntimeCapabilityLevel backward,
                                            bool forward_succeeds = true,
                                            size_t min_operation_values = 0)
                : name_(runtime_name), forward_(forward), backward_(backward),
                    forward_succeeds_(forward_succeeds),
                    min_operation_values_(min_operation_values) {}

    const char* name() const override { return name_; }
    bool initialize(const RuntimeConfig& cfg) override { config_ = cfg; initialized_ = true; return true; }
    void shutdown() override { initialized_ = false; }
    bool isInitialized() const override { return initialized_; }
    bool linearForward(const float*, const float*, const float*, float*, int, int, int) override { return false; }
    bool forwardLayer(const std::vector<const std::vector<float>*>& inputs,
                      std::vector<std::vector<float>>& outputs,
                      const Layer&, bool) override {
        ++forward_calls_;
        if (!forward_succeeds_ || !runtimeCapabilityIsNative(forward_) ||
            inputs.empty() || !inputs[0]) return false;
        outputs = {{static_cast<float>(name_[0]), inputs[0]->front()}};
        return true;
    }
    int forwardCalls() const { return forward_calls_; }
    bool backwardLayer(const std::vector<const std::vector<float>*>& inputs,
                       const std::vector<const std::vector<float>*>& grad_outputs,
                       std::vector<std::vector<float>>& grad_inputs,
                       Layer&, bool) override {
        ++backward_calls_;
        if (!runtimeCapabilityIsNative(backward_) || inputs.empty() || !inputs[0] ||
            grad_outputs.empty() || !grad_outputs[0]) return false;
        grad_inputs = {*grad_outputs[0]};
        return true;
    }
    int backwardCalls() const { return backward_calls_; }
    bool supportsForwardLayerType(LayerType type) const override {
        return type == LayerType::Linear && forward_ != RuntimeCapabilityLevel::Unsupported;
    }
    bool supportsBackwardLayerType(LayerType type) const override {
        return type == LayerType::Linear && backward_ != RuntimeCapabilityLevel::Unsupported;
    }
    RuntimeCapabilityLevel queryForwardCapability(LayerType type) const override {
        return type == LayerType::Linear ? forward_ : RuntimeCapabilityLevel::Unsupported;
    }
    RuntimeCapabilityLevel queryBackwardCapability(LayerType type) const override {
        return type == LayerType::Linear ? backward_ : RuntimeCapabilityLevel::Unsupported;
    }
    RuntimeCapabilityLevel queryForwardOperationCapability(
        const Layer& layer,
        const std::vector<const std::vector<float>*>& inputs,
        bool training) const override {
        const auto capability = AbstractRuntime::queryForwardOperationCapability(
            layer, inputs, training);
        return runtimeCapabilityIsNative(capability) && inputs[0]->size() >= min_operation_values_
            ? capability : RuntimeCapabilityLevel::Unsupported;
    }
    RuntimeCapabilityLevel queryBackwardOperationCapability(
        const Layer& layer,
        const std::vector<const std::vector<float>*>& inputs,
        const std::vector<const std::vector<float>*>& grad_outputs,
        bool training) const override {
        const auto capability = AbstractRuntime::queryBackwardOperationCapability(
            layer, inputs, grad_outputs, training);
        return runtimeCapabilityIsNative(capability) && inputs[0]->size() >= min_operation_values_
            ? capability : RuntimeCapabilityLevel::Unsupported;
    }

private:
    const char* name_;
    RuntimeCapabilityLevel forward_;
    RuntimeCapabilityLevel backward_;
    bool forward_succeeds_ = true;
    size_t min_operation_values_ = 0;
    int forward_calls_ = 0;
    int backward_calls_ = 0;
    bool initialized_ = false;
};

int main() {
    Model primary_model;
    TASSERT_TRUE(primary_model.hasCpuCompute());
    {
        Model metadata_model;
        TASSERT_TRUE(metadata_model.hasCpuCompute());
    }
    TASSERT_TRUE(primary_model.hasCpuCompute());

    CpuRuntime concrete_cpu;
    TASSERT_TRUE(concrete_cpu.queryForwardCapability(LayerType::Linear) ==
                 RuntimeCapabilityLevel::Native);

#ifdef ENABLE_VULKAN
    VulkanRuntime concrete_vulkan;
    TASSERT_TRUE(concrete_vulkan.queryForwardCapability(LayerType::Add) ==
                 RuntimeCapabilityLevel::NativeOptimized);
    TASSERT_TRUE(concrete_vulkan.queryForwardCapability(LayerType::Subtract) ==
                 RuntimeCapabilityLevel::Native);
    TASSERT_TRUE(concrete_vulkan.queryBackwardCapability(LayerType::Linear) ==
                 RuntimeCapabilityLevel::Native);
#endif

#ifdef ENABLE_OPENCL
    OpenCLRuntime concrete_opencl;
    TASSERT_TRUE(concrete_opencl.queryForwardCapability(LayerType::Add) ==
                 RuntimeCapabilityLevel::Native);
    TASSERT_TRUE(concrete_opencl.queryBackwardCapability(LayerType::Add) ==
                 RuntimeCapabilityLevel::Native);
#endif

    CapabilityRuntime cuda("CUDA", RuntimeCapabilityLevel::HostFallback,
                           RuntimeCapabilityLevel::HostFallback);
    CapabilityRuntime vulkan("VULKAN", RuntimeCapabilityLevel::Native,
                             RuntimeCapabilityLevel::Unsupported);
    CapabilityRuntime cpu("CPU", RuntimeCapabilityLevel::Native,
                          RuntimeCapabilityLevel::Native);
    RuntimeConfig cfg;
    TASSERT_TRUE(cuda.initialize(cfg));
    TASSERT_TRUE(vulkan.initialize(cfg));
    TASSERT_TRUE(cpu.initialize(cfg));

    Layer linear;
    linear.type_enum = LayerType::Linear;

    auto& router = RuntimeRouter::instance();
    router.setRuntimes(nullptr, &cuda, &vulkan, nullptr, nullptr, &cpu);

    // A higher-priority host fallback must not hide a lower native route.
    TASSERT_TRUE(router.selectForwardRuntimeForLayer(linear) == &vulkan);
    TASSERT_TRUE(router.selectBackwardRuntimeForLayer(linear) == &cpu);

    const std::vector<float> input = {3.0f};
    const std::vector<const std::vector<float>*> inputs = {&input};
    std::vector<std::vector<float>> outputs;
    AbstractRuntime* executed = nullptr;
    TASSERT_TRUE(router.dispatchForwardLayerPlanned(
        &cpu, inputs, outputs, linear, false, &executed));
    TASSERT_TRUE(executed == &cpu);
    TASSERT_TRUE(outputs.size() == 1 && outputs[0].size() == 2);

    CapabilityRuntime failing_vulkan(
        "VULKAN", RuntimeCapabilityLevel::Native,
        RuntimeCapabilityLevel::Unsupported, false);
    TASSERT_TRUE(failing_vulkan.initialize(cfg));
    router.setRuntimes(nullptr, nullptr, &failing_vulkan, nullptr, nullptr, &cpu);
    TASSERT_TRUE(router.dispatchForwardLayerPlanned(
        &failing_vulkan, inputs, outputs, linear, false, &executed));
    TASSERT_TRUE(executed == &cpu);
    TASSERT_TRUE(failing_vulkan.forwardCalls() == 1);
    TASSERT_TRUE(router.dispatchForwardLayerPlanned(
        &failing_vulkan, inputs, outputs, linear, false, &executed));
    TASSERT_TRUE(executed == &cpu);
    TASSERT_TRUE(failing_vulkan.forwardCalls() == 1);

    CapabilityRuntime shape_sensitive(
        "VULKAN", RuntimeCapabilityLevel::Native,
        RuntimeCapabilityLevel::Native, true, 2);
    TASSERT_TRUE(shape_sensitive.initialize(cfg));
    router.setRuntimes(nullptr, nullptr, &shape_sensitive, nullptr, nullptr, &cpu);

    // Refusing one concrete operation must skip the runtime without banning it.
    TASSERT_TRUE(router.dispatchForwardLayerPlanned(
        &shape_sensitive, inputs, outputs, linear, false, &executed));
    TASSERT_TRUE(executed == &cpu);
    TASSERT_TRUE(shape_sensitive.forwardCalls() == 0);

    const std::vector<float> large_input = {3.0f, 4.0f};
    const std::vector<const std::vector<float>*> large_inputs = {&large_input};
    TASSERT_TRUE(router.dispatchForwardLayerPlanned(
        &shape_sensitive, large_inputs, outputs, linear, false, &executed));
    TASSERT_TRUE(executed == &shape_sensitive);
    TASSERT_TRUE(shape_sensitive.forwardCalls() == 1);

    const std::vector<float> grad_output = {1.0f};
    const std::vector<const std::vector<float>*> grad_outputs = {&grad_output};
    std::vector<std::vector<float>> grad_inputs;
    TASSERT_TRUE(router.dispatchBackwardLayer(
        inputs, grad_outputs, grad_inputs, linear, true, &executed));
    TASSERT_TRUE(executed == &cpu);
    TASSERT_TRUE(shape_sensitive.backwardCalls() == 0);

    const std::vector<float> large_grad_output = {1.0f, 1.0f};
    const std::vector<const std::vector<float>*> large_grad_outputs = {&large_grad_output};
    TASSERT_TRUE(router.dispatchBackwardLayer(
        large_inputs, large_grad_outputs, grad_inputs, linear, true, &executed));
    TASSERT_TRUE(executed == &shape_sensitive);
    TASSERT_TRUE(shape_sensitive.backwardCalls() == 1);

    // Compatibility mode may still select a host fallback when no native
    // implementation exists.
    router.setRuntimes(nullptr, &cuda, nullptr, nullptr, nullptr, nullptr);
    TASSERT_TRUE(router.selectForwardRuntimeForLayer(linear) == nullptr);
    TASSERT_TRUE(router.selectForwardRuntimeForLayer(linear, true) == &cuda);

    return 0;
}
