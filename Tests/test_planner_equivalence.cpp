#include "test_utils.hpp"

#include "Model.hpp"

#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <vector>

struct RunResult {
    std::vector<float> output;
    std::vector<float> input_gradient;
    std::vector<float> weight_gradient;
};

static RunResult run(const char* mode) {
    setTestEnvironment("MIMIR_ENABLE_PLANNER", "1");
    setTestEnvironment("MIMIR_PLANNER_MODE", mode);
    setTestEnvironment("MIMIR_ENABLE_FUSION", "0");

    Model model;
    model.push("linear", "Linear", 0);
    Layer& layer = model.getMutableLayers()[0];
    layer.inputs = {"__input__"};
    layer.output = "x";
    layer.in_features = 2;
    layer.out_features = 2;
    layer.use_bias = false;
    layer.params_count = 4;
    model.allocateParams();
    float* weights = layer.getWeights();
    if (!weights) throw std::runtime_error("planner equivalence: weights unavailable");
    weights[0] = 1.0f; weights[1] = 2.0f;
    weights[2] = 3.0f; weights[3] = 4.0f;

    const std::vector<float> input = {0.25f, -0.5f};
    RunResult result;
    result.output = model.forwardPass(input, true);
    (void)model.backwardPass({0.75f, -0.25f});
    if (!model.hasLastInputGradient()) {
        throw std::runtime_error("planner equivalence: input gradient unavailable");
    }
    result.input_gradient = model.getLastInputGradient();
    result.weight_gradient = layer.grad_weights;
    return result;
}

static std::vector<float> run_reuse_chain(const char* mode, const bool reuse) {
    setTestEnvironment("MIMIR_ENABLE_PLANNER", "1");
    setTestEnvironment("MIMIR_PLANNER_MODE", mode);
    setTestEnvironment("MIMIR_ENABLE_FUSION", "0");
    setTestEnvironment("MIMIR_PLANNER_BUFFER_REUSE", reuse ? "1" : "0");
    setTestEnvironment("MIMIR_PLANNER_BUFFER_POISON", reuse ? "1" : "0");

    Model model;
    const char* outputs[] = {"t0", "t1", "t2", "x"};
    const char* inputs[] = {"x", "t0", "t1", "t2"};
    for (size_t i = 0; i < 4; ++i) {
        model.push(std::string("identity_") + std::to_string(i), "Identity", 0);
        Layer& layer = model.getMutableLayers().back();
        layer.inputs = {inputs[i]};
        layer.output = outputs[i];
        layer.shape = {4};
    }
    model.allocateParams();
    return model.forwardPass(std::vector<float>{1.0f, -2.0f, 3.5f, 4.0f}, false);
}

static std::vector<float> run_resident_unary_chain() {
    setTestEnvironment("MIMIR_ENABLE_PLANNER", "1");
    setTestEnvironment("MIMIR_PLANNER_MODE", "static");
    setTestEnvironment("MIMIR_ENABLE_FUSION", "0");
    setTestEnvironment("MIMIR_PLANNER_DEVICE_RESIDENCY", "1");
    setTestEnvironment("MIMIR_VULKAN_LINEAR", "1");
    setTestEnvironment("MIMIR_VULKAN_LINEAR_MIN_OPS", "0");

    Model model;
    model.push("relu", "ReLU", 0);
    model.getMutableLayers().back().inputs = {"x"};
    model.getMutableLayers().back().output = "t0";
    model.getMutableLayers().back().shape = {4};
    model.push("sigmoid", "Sigmoid", 0);
    model.getMutableLayers().back().inputs = {"t0"};
    model.getMutableLayers().back().output = "x";
    model.getMutableLayers().back().shape = {4};
    model.allocateParams();
    return model.forwardPass(std::vector<float>{-2.0f, -0.5f, 0.0f, 2.0f}, false);
}

int main() {
    const auto resident = run_resident_unary_chain();
    TASSERT_TRUE(resident.size() == 4);
    TASSERT_NEAR(resident[0], 0.5f, 1e-5f);
    TASSERT_NEAR(resident[1], 0.5f, 1e-5f);
    TASSERT_NEAR(resident[2], 0.5f, 1e-5f);
    TASSERT_NEAR(resident[3], 1.f / (1.f + std::exp(-2.f)), 1e-5f);

    const RunResult legacy = run("legacy");
    const RunResult planned = run("static");
    TASSERT_TRUE(legacy.output.size() == planned.output.size());
    TASSERT_TRUE(legacy.input_gradient.size() == planned.input_gradient.size());
    TASSERT_TRUE(legacy.weight_gradient.size() == planned.weight_gradient.size());
    for (size_t i = 0; i < legacy.output.size(); ++i) {
        TASSERT_NEAR(legacy.output[i], planned.output[i], 1e-6f);
    }
    for (size_t i = 0; i < legacy.input_gradient.size(); ++i) {
        TASSERT_NEAR(legacy.input_gradient[i], planned.input_gradient[i], 1e-6f);
    }
    for (size_t i = 0; i < legacy.weight_gradient.size(); ++i) {
        TASSERT_NEAR(legacy.weight_gradient[i], planned.weight_gradient[i], 1e-6f);
    }


    const auto chain_legacy = run_reuse_chain("legacy", false);
    const auto chain_reused = run_reuse_chain("static", true);
    TASSERT_TRUE(chain_legacy.size() == chain_reused.size());
    for (size_t i = 0; i < chain_legacy.size(); ++i) {
        TASSERT_NEAR(chain_legacy[i], chain_reused[i], 1e-6f);
    }
    return 0;
}
