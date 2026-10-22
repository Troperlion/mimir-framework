#pragma once
#include "runtimes/AbstractRuntime.hpp"
#include <functional>
#include <unordered_map>

namespace Mimir::Planning::Calibration {
struct Measurement { AbstractRuntime* runtime=nullptr; size_t configuration=0; double seconds=0; };
struct Ranking { size_t key=0; bool calibrated=false; std::vector<AbstractRuntime*> route; };
using Probe=std::function<bool(AbstractRuntime&,Layer&,std::vector<std::vector<float>>&)>;
extern thread_local std::unordered_map<const Layer*,std::vector<Measurement>> forward_profiles;
bool enabled();
size_t configuration(const AbstractRuntime& runtime);
void invalidate(size_t key);
Ranking rank(const std::vector<AbstractRuntime*>& candidates,const Layer& layer,
    const std::vector<const std::vector<float>*>& inputs,const std::vector<const std::vector<float>*>& gradients,
    bool backward,bool training,uint64_t generation,const Probe& execute);
}
