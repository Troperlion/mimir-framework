#include "OptimizerDispatch.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "Model.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <limits>
#include <tuple>
namespace {
std::unordered_map<std::string,size_t>& optimizerCounts(){static thread_local std::unordered_map<std::string,size_t> counts;return counts;}
struct Trial {
    RuntimeOptimizerRequest request;
    std::vector<float> w,m,v;
    explicit Trial(const RuntimeOptimizerRequest& r):request(r),w(r.weights,r.weights+r.count) {
        request.weights=w.data();
        if(r.first_moment){m.assign(r.first_moment,r.first_moment+r.count);request.first_moment=m.data();}
        if(r.second_moment){v.assign(r.second_moment,r.second_moment+r.count);request.second_moment=v.data();}
    }
    void commit(const RuntimeOptimizerRequest& r)const {
        std::copy(w.begin(),w.end(),r.weights);
        if(r.first_moment)std::copy(m.begin(),m.end(),r.first_moment);
        if(r.second_moment)std::copy(v.begin(),v.end(),r.second_moment);
    }
};
bool execute(AbstractRuntime* runtime,const RuntimeOptimizerRequest& r) {
    return runtime->optimizerStep(r);
}
}
bool RuntimeRouter::dispatchOptimizer(const RuntimeOptimizerRequest& request,AbstractRuntime** selected) const {
    if(selected)*selected=nullptr;
    if(!optimizerRequestValid(request))return false;
    ensureActivatedAndComposed();
    std::vector<AbstractRuntime*> candidates;
    for(auto* runtime:runtime_priority_)
        if(runtime && runtime->isInitialized() && !runtime->config().disabled &&
           runtime->supportsOptimizer() && (std::string(runtime->name())=="CPU" || optimizerDeviceSizeFits(request)))candidates.push_back(runtime);
    if(candidates.empty())return false;
    using Key=std::tuple<int,size_t,std::vector<AbstractRuntime*>>;
    static thread_local std::map<Key,AbstractRuntime*> fastest;
    const Key key{int(request.optimizer->type),request.count,candidates};
    auto found=fastest.find(key);
    if(found==fastest.end() && candidates.size()>1) {
        double best=std::numeric_limits<double>::infinity();AbstractRuntime* winner=nullptr;
        // Measure the whole synchronized host/device boundary, on disposable
        // copies of THIS update. Compilation warmup never updates real state.
        for(auto* runtime:candidates) {
            bool ok=true;double elapsed=0;
            for(int sample=0;sample<2;++sample) {
                Trial destination(request);const auto start=std::chrono::steady_clock::now();
                if(std::string(runtime->name())=="CPU") {
                    if(!execute(runtime,destination.request)){ok=false;break;}
                } else {
                    Trial trial(request);
                    if(!execute(runtime,trial.request)){ok=false;break;}
                    trial.commit(destination.request);
                }
                elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            }
            if(ok && elapsed<best){best=elapsed;winner=runtime;}
        }
        if(!winner)return false;
        found=fastest.emplace(key,winner).first;
        std::cerr<<"[optimizer] backend="<<winner->name()<<" type="<<int(request.optimizer->type)
                 <<" parameters="<<request.count<<" selection=measured_host_roundtrip\n";
    }
    if(found==fastest.end())found=fastest.emplace(key,candidates.front()).first;
    std::vector<AbstractRuntime*> ordered{found->second};
    for(auto* runtime:candidates)if(runtime!=found->second)ordered.push_back(runtime);
    for(auto* runtime:ordered) {
        if(std::string(runtime->name())=="CPU") {
            if(!execute(runtime,request)) continue;
            ++optimizerCounts()[runtime->name()];
            if(selected) *selected=runtime;
            fastest[key]=runtime;
            return true;
        }
        Trial trial(request);
        if(execute(runtime,trial.request)) {
            trial.commit(request);
            ++optimizerCounts()[runtime->name()];
            if(selected) *selected=runtime;
            fastest[key]=runtime;
            return true;
        }
    }
    return false;
}

const std::unordered_map<std::string,size_t>& RuntimeRouter::optimizerDispatchCounts() const { return optimizerCounts(); }
