#pragma once
#include <cstddef>
#include <climits>
#include <vector>
struct Optimizer;
// Internal host boundary. Public Optimizer/Lua/script contracts are unchanged.
// Runtimes commit all three buffers only after successful synchronized execution.
struct RuntimeOptimizerRequest {
    const Optimizer* optimizer=nullptr;
    float learning_rate=0;
    float* weights=nullptr;
    const float* gradients=nullptr;
    float* first_moment=nullptr;
    float* second_moment=nullptr;
    size_t count=0;
};
inline bool optimizerDeviceSizeFits(const RuntimeOptimizerRequest& r) { return r.count<=size_t((INT_MAX-2)/6); }
bool optimizerRequestValid(const RuntimeOptimizerRequest&);
void optimizerHostStep(const RuntimeOptimizerRequest&);
// Compact device layout: results are a contiguous [weights, active moments]
// prefix, followed by gradients and (only for reductions) temporary workspace.
struct OptimizerDeviceLayout {
    size_t first_moment=0, second_moment=0, gradients=0, updates=0, reductions=0;
    size_t result_count=0, total_count=0;
    unsigned moment_flags=0;
};
OptimizerDeviceLayout optimizerDeviceLayout(const RuntimeOptimizerRequest&);
std::vector<int> optimizerParameters(const RuntimeOptimizerRequest&, bool compact = false);
inline size_t optimizerInputCount(const std::vector<int>& parameters) {
    return static_cast<size_t>(parameters.at(22)) + static_cast<size_t>(parameters.at(1));
}
inline size_t optimizerResultCount(const std::vector<int>& parameters) {
    return static_cast<size_t>(parameters.at(19));
}
std::vector<float> optimizerPack(const RuntimeOptimizerRequest&, bool compact = false);
bool optimizerCommit(const RuntimeOptimizerRequest&,const std::vector<float>&, bool compact = false);
