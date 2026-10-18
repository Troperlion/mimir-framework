#include "test_utils.hpp"

#include "Layers.hpp"
#include "Planning/Planner.hpp"
#include "runtimes/RuntimeTensorHandle.hpp"

#include <string>
#include <vector>

class PlanRuntime final : public AbstractRuntime {
public:
    explicit PlanRuntime(const char* name) : name_(name) {}
    const char* name() const override { return name_; }
    bool initialize(const RuntimeConfig& cfg) override { config_ = cfg; return true; }
    void shutdown() override {}
    bool isInitialized() const override { return true; }
    bool linearForward(const float*, const float*, const float*, float*, int, int, int) override { return false; }
    bool forwardLayer(const std::vector<const std::vector<float>*>&,
                      std::vector<std::vector<float>>&, const Layer&, bool) override { return false; }
    bool supportsForwardLayerType(LayerType) const override { return true; }
    RuntimeCapabilityLevel queryForwardCapability(LayerType) const override {
        return RuntimeCapabilityLevel::Native;
    }
private:
    const char* name_;
};

int main() {
    Layer first;
    first.name = "first\"layer";
    first.type_enum = LayerType::Identity;
    first.inputs = {"x"};
    first.output = "a";
    first.dtype = Mimir::DType::F32;
    first.shape = {2, 3};

    Layer overwrite;
    overwrite.name = "overwrite";
    overwrite.type_enum = LayerType::Identity;
    overwrite.inputs = {"a"};
    overwrite.output = "a";
    overwrite.dtype = Mimir::DType::F32;
    overwrite.shape = {2, 3};

    Layer branch;
    branch.name = "branch";
    branch.type_enum = LayerType::Add;
    branch.inputs = {"a", "x"};
    branch.output = "y";
    branch.dtype = Mimir::DType::F32;
    branch.shape = {2, 3};

    const std::vector<Layer> layers = {first, overwrite, branch};
    const auto plan = Mimir::Planning::build_execution_plan_global(
        layers, true, Mimir::Planning::PlannerMode::Static);

    TASSERT_TRUE(plan.layers.size() == 3);
    TASSERT_TRUE(plan.layers[0].output != plan.layers[1].output);
    TASSERT_TRUE(plan.layers[1].inputs[0] == plan.layers[0].output);
    TASSERT_TRUE(plan.layers[2].inputs[0] == plan.layers[1].output);
    TASSERT_TRUE(plan.tensors.at(plan.layers[0].output).generation == 0);
    TASSERT_TRUE(plan.tensors.at(plan.layers[1].output).generation == 1);
    TASSERT_TRUE(plan.tensors.at(plan.layers[0].output).bytes == 24);
    TASSERT_TRUE(plan.tensors.at(plan.layers[0].output).required_for_backward);

    const std::string text1 = Mimir::Planning::dump_execution_plan_text(plan);
    const std::string text2 = Mimir::Planning::dump_execution_plan_text(plan);
    const std::string json1 = Mimir::Planning::dump_execution_plan_json(plan);
    const std::string json2 = Mimir::Planning::dump_execution_plan_json(plan);
    TASSERT_TRUE(text1 == text2);
    TASSERT_TRUE(json1 == json2);
    TASSERT_TRUE(json1.find("first\\\"layer") != std::string::npos);
    TASSERT_TRUE(json1.find("\"generation\":1") != std::string::npos);

    const auto no_fusion_plan = Mimir::Planning::build_execution_plan_global(
        layers, false, Mimir::Planning::PlannerMode::Static, false, false);
    TASSERT_TRUE(std::none_of(no_fusion_plan.skip_layer.begin(), no_fusion_plan.skip_layer.end(),
                             [](uint8_t value) { return value != 0; }));

    // Inference may reuse non-overlapping, compatible host tensors.
    Layer c0 = first;
    c0.name = "c0"; c0.inputs = {"x"}; c0.output = "t0";
    Layer c1 = first;
    c1.name = "c1"; c1.inputs = {"t0"}; c1.output = "t1";
    Layer c2 = first;
    c2.name = "c2"; c2.inputs = {"t1"}; c2.output = "t2";
    Layer c3 = first;
    c3.name = "c3"; c3.inputs = {"t2"}; c3.output = "y";
    auto reuse_plan = Mimir::Planning::build_execution_plan_global({c0, c1, c2, c3}, false);
    Mimir::Planning::plan_host_buffer_reuse(reuse_plan, true);
    const auto t0_slot = reuse_plan.tensors.at(reuse_plan.layers[0].output).physical_buffer_id;
    const auto t2_slot = reuse_plan.tensors.at(reuse_plan.layers[2].output).physical_buffer_id;
    TASSERT_TRUE(t0_slot == t2_slot);
    TASSERT_TRUE(reuse_plan.stats.buffer_reuse_saved >= 24);

    // Training extends lifetimes through backward and disables that reuse.
    auto training_plan = Mimir::Planning::build_execution_plan_global({c0, c1, c2, c3}, true);
    Mimir::Planning::plan_host_buffer_reuse(training_plan, true);
    TASSERT_TRUE(training_plan.tensors.at(training_plan.layers[0].output).physical_buffer_id !=
                 training_plan.tensors.at(training_plan.layers[2].output).physical_buffer_id);

    // A branch producer stays alive until both consumers have executed.
    Layer b0 = first; b0.name = "branch_source"; b0.inputs = {"x"}; b0.output = "shared";
    Layer b1 = first; b1.name = "left"; b1.inputs = {"shared"}; b1.output = "left_out";
    Layer b2 = first; b2.name = "right"; b2.inputs = {"shared"}; b2.output = "right_out";
    Layer b3 = branch; b3.name = "merge"; b3.inputs = {"left_out", "right_out"}; b3.output = "y";
    auto branch_plan = Mimir::Planning::build_execution_plan_global({b0, b1, b2, b3}, false);
    Mimir::Planning::plan_host_buffer_reuse(branch_plan, true);
    const auto& shared = branch_plan.tensors.at(branch_plan.layers[0].output);
    const auto& left = branch_plan.tensors.at(branch_plan.layers[1].output);
    TASSERT_TRUE(shared.last_use == 2);
    TASSERT_TRUE(shared.physical_buffer_id != left.physical_buffer_id);
    TASSERT_TRUE(branch_plan.skip_layer[1] == 0 && branch_plan.skip_layer[2] == 0);

    PlanRuntime vulkan("VULKAN");
    PlanRuntime cpu("CPU");
    auto placed = Mimir::Planning::build_execution_plan_global({c0, c1, c2}, false,
                                                                Mimir::Planning::PlannerMode::Static);
    Mimir::Planning::apply_runtime_placement(placed, {&vulkan, &vulkan, &cpu});
    TASSERT_TRUE(placed.regions.size() == 2);
    TASSERT_TRUE(placed.regions[0].runtime == RuntimeKind::Vulkan);
    TASSERT_TRUE(placed.regions[1].runtime == RuntimeKind::CPU);
    TASSERT_TRUE(placed.tensors.at(placed.layers[0].output).preferred_location ==
                 Mimir::Planning::MemoryLocation::Vulkan);
    bool found_download = false;
    for (const auto& transfer : placed.transfers) {
        if (transfer.from == Mimir::Planning::MemoryLocation::Vulkan &&
            transfer.to == Mimir::Planning::MemoryLocation::Host &&
            transfer.bytes == 24) {
            found_download = true;
        }
    }
    TASSERT_TRUE(found_download);

    auto cost_placed = Mimir::Planning::build_execution_plan_global(
        {c0}, false, Mimir::Planning::PlannerMode::Cost);
    Mimir::Planning::apply_runtime_placement(cost_placed, {&vulkan, &cpu});
    TASSERT_TRUE(cost_placed.layers[0].runtime == RuntimeKind::CPU);
    TASSERT_TRUE(cost_placed.layers[0].estimated_compute_cost > 0.0);

    Mimir::RuntimeTensorHandle resident({2, 3}, Mimir::DType::F32);
    TASSERT_TRUE(resident.bytes() == 24);
    resident.prepareHostWrite();
    TASSERT_TRUE(resident.hostValid());
    TASSERT_TRUE(!resident.deviceValid());
    resident.markDeviceWritten({RuntimeKind::Vulkan, 1, 24});
    TASSERT_TRUE(!resident.hostValid());
    TASSERT_TRUE(resident.deviceValid());

    Mimir::Planning::BenchmarkCache benchmark_cache;
    Mimir::Planning::BenchmarkRecord record;
    record.layer_type = LayerType::Linear;
    record.runtime = RuntimeKind::Vulkan;
    record.dtype = Mimir::DType::F32;
    record.shape = {2, 3};
    record.parameters_key = "in=3,out=4";
    record.observed_seconds = 0.001;
    TASSERT_TRUE(!benchmark_cache.lookup(record).has_value());
    benchmark_cache.record(record);
    TASSERT_TRUE(benchmark_cache.lookup(record).has_value());
    TASSERT_NEAR(*benchmark_cache.lookup(record), 0.001, 1e-12);
    Mimir::Planning::BenchmarkRecord cpu_record = record;
    cpu_record.layer_type = c0.type_enum;
    cpu_record.runtime = RuntimeKind::CPU;
    cpu_record.parameters_key = Mimir::Planning::benchmark_parameters_key(c0);
    cpu_record.observed_seconds = 0.0005;
    Mimir::Planning::BenchmarkRecord vulkan_record = cpu_record;
    vulkan_record.runtime = RuntimeKind::Vulkan;
    vulkan_record.observed_seconds = 0.002;
    benchmark_cache.record(cpu_record);
    benchmark_cache.record(vulkan_record);
    TASSERT_TRUE(Mimir::Planning::select_runtime_by_benchmark(
        c0,
        placed.tensors.at(placed.layers[0].output),
        {&vulkan, &cpu},
        benchmark_cache) == &cpu);

    auto scratch_plan = Mimir::Planning::build_execution_plan_global({c0}, false);
    Layer conv;
    conv.type_enum = LayerType::Conv2d;
    conv.in_channels = 3;
    conv.out_channels = 8;
    conv.kernel_size = 3;
    conv.input_height = 16;
    conv.input_width = 16;
    scratch_plan = Mimir::Planning::build_execution_plan_global({conv}, false);
    Mimir::Planning::apply_global_scratch_plan(scratch_plan, {conv});
    TASSERT_TRUE(scratch_plan.stats.scratch_peak > 0);
    TASSERT_TRUE(scratch_plan.layers[0].scratch_bytes > 0);
    TASSERT_TRUE(!vulkan.supportsKernelFusion(LayerType::Linear, LayerType::GELU));
    TASSERT_TRUE(resident.location() == RuntimeKind::Vulkan);
    resident.markHostSynchronized(std::vector<uint8_t>(24, 0));
    TASSERT_TRUE(resident.hostValid());
    TASSERT_TRUE(resident.deviceValid());

    return 0;
}
