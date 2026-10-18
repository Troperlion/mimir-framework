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
std::vector<int> optimizerParameters(const RuntimeOptimizerRequest&);
std::vector<float> optimizerPack(const RuntimeOptimizerRequest&);
bool optimizerCommit(const RuntimeOptimizerRequest&,const std::vector<float>&);
