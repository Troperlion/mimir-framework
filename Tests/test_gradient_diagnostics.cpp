#include "test_utils.hpp"
#include "Model.hpp"
#include "AsyncMonitor.hpp"
#include <limits>

int main() {
    GradientDiagnostics rolling;
    TASSERT_TRUE(!rolling.requested() && rolling.snapshot().samples == 0);
    rolling.record(2, false);
    rolling.record(4, true);
    auto stats = rolling.snapshot();
    TASSERT_NEAR(stats.grad_norm_median, 3, 1e-6f);
    TASSERT_NEAR(stats.grad_norm_p95, 4, 1e-6f);
    TASSERT_NEAR(stats.clip_rate, 50, 1e-6f);
    // An old spike and its clipping decision expire together.
    rolling.record(10000, true);
    for (int i = 1; i <= 256; ++i) rolling.record(i, i % 2 == 0);
    stats = rolling.snapshot();
    TASSERT_TRUE(stats.samples == 256);
    TASSERT_NEAR(stats.grad_norm_median, 128.5f, 1e-6f);
    TASSERT_NEAR(stats.grad_norm_p95, 244, 1e-6f);
    TASSERT_NEAR(stats.grad_norm_unclipped_max, 256, 1e-6f);
    TASSERT_NEAR(stats.clip_rate, 50, 1e-6f);
    rolling.record(std::numeric_limits<double>::quiet_NaN(), false);
    rolling.record(std::numeric_limits<double>::infinity(), false);
    stats = rolling.snapshot();
    TASSERT_TRUE(stats.nonfinite == 2 && std::isfinite(stats.grad_norm_median));

    for (const auto* backend : {"CUDA", "ROCM", "VULKAN", "OPENCL"})
        setTestEnvironment((std::string("MIMIR_DISABLE_") + backend).c_str(), "1");
    setTestEnvironment("MIMIR_DISABLE_CPU", "0");
    Model model;
    model.push("active", "Linear", 2);
    model.push("frozen", "Linear", 1);
    model.allocateParams();
    auto& layers = model.getMutableLayers();
    std::fill(layers[0].getWeights(), layers[0].getWeights() + 2, 0.f);
    layers[0].grad_weights = {3, 4};
    layers[1].grad_weights = {1000};
    model.modelConfig["frozen_layer_prefixes"] = {"frozen"};
    model.modelConfig["grad_clip_norm"] = 1.f;
    Optimizer opt;
    opt.type = OptimizerType::SGD;
    opt.weight_decay = 0;
    model.optimizerStep(opt, .1f);
    auto config = model.runtimeConfiguration();
    TASSERT_TRUE(config->gradient_diagnostics.snapshot().samples == 0);
    // Binding the real monitor opts in; UI ticks do not count as optimizer steps.
    AsyncMonitor monitor;
    monitor.bindRuntimeConfiguration(config);
    TASSERT_TRUE(config->gradient_diagnostics.requested());
    layers[0].grad_weights = {3, 4};
    model.optimizerStep(opt, .1f);
    stats = config->gradient_diagnostics.snapshot();
    TASSERT_TRUE(stats.samples == 1);
    TASSERT_NEAR(stats.grad_norm_unclipped_max, 5, 1e-6f);
    TASSERT_NEAR(stats.clip_rate, 100, 1e-6f);
    TASSERT_NEAR(layers[0].grad_weights[0], .6f, 1e-6f);
    TASSERT_NEAR(layers[0].grad_weights[1], .8f, 1e-6f);
    TASSERT_NEAR(layers[1].grad_weights[0], 1000, 1e-6f);
    model.modelConfig["grad_clip_norm"] = 0.f;
    layers[0].grad_weights = {6, 8};
    model.optimizerStep(opt, .1f);
    for (int i = 0; i < 10; ++i) monitor.updateMetrics(AsyncMonitor::Metrics{});
    stats = config->gradient_diagnostics.snapshot();
    TASSERT_TRUE(stats.samples == 2);
    TASSERT_NEAR(stats.clip_rate, 50, 1e-6f);
    TASSERT_NEAR(stats.grad_norm_median, 7.5f, 1e-6f);
    TASSERT_NEAR(stats.grad_norm_p95, 10, 1e-6f);
    TASSERT_NEAR(stats.grad_norm_unclipped_max, 10, 1e-6f);
    return 0;
}
