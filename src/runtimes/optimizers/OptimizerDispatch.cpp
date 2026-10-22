#include "OptimizerDispatch.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "Model.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
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
    void reset(const RuntimeOptimizerRequest& r) {
        std::copy_n(r.weights,r.count,w.data());
        if(r.first_moment)std::copy_n(r.first_moment,r.count,m.data());
        if(r.second_moment)std::copy_n(r.second_moment,r.count,v.data());
    }
    void commit(const RuntimeOptimizerRequest& r)const {
        std::copy(w.begin(),w.end(),r.weights);
        if(r.first_moment)std::copy(m.begin(),m.end(),r.first_moment);
        if(r.second_moment)std::copy(v.begin(),v.end(),r.second_moment);
    }
};
bool cpu(const AbstractRuntime* runtime) { return std::strcmp(runtime->name(),"CPU")==0; }
bool update(AbstractRuntime* runtime,const RuntimeOptimizerRequest& r) {
    if(cpu(runtime))return runtime->optimizerStep(r);
    // Device failure must never partially commit parameters or moments.
    Trial trial(r);
    if(!runtime->optimizerStep(trial.request))return false;
    trial.commit(r);
    return true;
}
constexpr size_t maxRuntimes=6;
constexpr size_t refreshUpdates=1024;
using Identity=std::pair<uintptr_t,int>;
using Key=std::tuple<int,size_t,unsigned,unsigned>;
struct Route {
    std::array<AbstractRuntime*,maxRuntimes> order{};
    size_t count=0,updates=0;
};
struct Cache {
    uint64_t generation=0;
    std::array<Identity,maxRuntimes> available{};
    std::map<Key,Route> routes;
};
// Same algorithm variant, state transfer layout and exact block size share a route.
unsigned variant(const Optimizer& o) {
    if(o.type==OptimizerType::ADAFACTOR)
        return unsigned(o.adafactor_beta1>0.f) | (unsigned(o.adafactor_scale_parameter)<<1) |
            (unsigned(o.adafactor_relative_step)<<2);
    if(o.type==OptimizerType::RADAM)return unsigned(o.step>5);
    return 0;
}
Route calibrate(const RuntimeOptimizerRequest& request,
                const std::array<AbstractRuntime*,maxRuntimes>& candidates,size_t count) {
    Route route;
    std::array<std::pair<double,AbstractRuntime*>,maxRuntimes> timings{};
    size_t valid=0;
    Trial destination(request);
    for(size_t index=0;index<count;++index) {
        AbstractRuntime* runtime=candidates[index];
        // Warm compilation/context creation on disposable state, never on the model.
        destination.reset(request);
        if(!update(runtime,destination.request))continue;
        std::array<double,5> samples{};
        bool ok=true;
        for(auto& sample:samples) {
            destination.reset(request); // outside the timer, just like the model's input state
            const auto start=std::chrono::steady_clock::now();
            if(!update(runtime,destination.request)){ok=false;break;}
            sample=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        }
        if(!ok)continue;
        std::sort(samples.begin(),samples.end());
        timings[valid++]={samples[samples.size()/2],runtime};
    }
    std::stable_sort(timings.begin(),timings.begin()+valid,
        [](const auto& a,const auto& b){return a.first<b.first;});
    route.count=valid;
    for(size_t i=0;i<valid;++i)route.order[i]=timings[i].second;
    if(valid) {
        std::cerr<<"[optimizer] backend="<<route.order[0]->name()<<" type="<<int(request.optimizer->type)
            <<" parameters="<<request.count<<" selection=median_host_roundtrip samples=5"
            <<" refresh_updates="<<refreshUpdates<<" candidates_seconds=";
        for(size_t i=0;i<valid;++i)std::cerr<<(i?",":"")<<timings[i].second->name()<<":"<<timings[i].first;
        std::cerr<<'\n';
    }
    return route;
}
}

bool RuntimeRouter::dispatchOptimizer(const RuntimeOptimizerRequest& request,AbstractRuntime** selected) const {
    if(selected)*selected=nullptr;
    if(!optimizerRequestValid(request))return false;
    ensureActivatedAndComposed();
    std::array<AbstractRuntime*,maxRuntimes> candidates{};
    std::array<Identity,maxRuntimes> identities{};
    size_t count=0;
    for(auto* runtime:runtime_priority_) {
        if(!runtime || !runtime->isInitialized() || runtime->config().disabled ||
           !runtime->supportsOptimizer() || (!cpu(runtime) && !optimizerDeviceSizeFits(request)))continue;
        if(std::find(candidates.begin(),candidates.begin()+count,runtime)!=candidates.begin()+count)continue;
        candidates[count]=runtime;identities[count++]={reinterpret_cast<uintptr_t>(runtime),runtime->config().device_index};
    }
    static thread_local Cache cache;
    if(cache.generation!=optimizer_routing_generation_ || cache.available!=identities) {
        cache.routes.clear();cache.generation=optimizer_routing_generation_;cache.available=identities;
    }
    if(!count)return false;
    if(count==1) {
        if(!update(candidates[0],request))return false;
        ++optimizerCounts()[candidates[0]->name()];
        if(selected)*selected=candidates[0];
        return true;
    }
    const unsigned moments=unsigned(request.first_moment!=nullptr) | (unsigned(request.second_moment!=nullptr)<<1);
    const Key key{int(request.optimizer->type),request.count,moments,variant(*request.optimizer)};
    auto found=cache.routes.find(key);
    if(found==cache.routes.end() || found->second.updates>=refreshUpdates) {
        Route route=calibrate(request,candidates,count);
        if(!route.count){cache.routes.erase(key);return false;}
        // Dynamic block shapes must not grow a thread's cache indefinitely.
        if(found==cache.routes.end() && cache.routes.size()>=512)cache.routes.erase(cache.routes.begin());
        found=cache.routes.insert_or_assign(key,route).first;
    }
    auto& route=found->second;
    bool failed=false;
    for(size_t i=0;i<route.count;++i) {
        auto* runtime=route.order[i];
        if(!update(runtime,request)){failed=true;continue;}
        ++optimizerCounts()[runtime->name()];
        if(selected)*selected=runtime;
        // A transient failure must not permanently pin the model to its fallback.
        route.updates=failed?refreshUpdates:route.updates+1;
        return true;
    }
    cache.routes.erase(found);
    return false;
}

const std::unordered_map<std::string,size_t>& RuntimeRouter::optimizerDispatchCounts() const { return optimizerCounts(); }
