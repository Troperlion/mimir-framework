#include "Model.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/optimizers/OptimizerDispatch.hpp"
#include <chrono>
#include <thread>
#include <iostream>
#include <stdexcept>
static void require(bool b,const char* why){if(!b)throw std::runtime_error(why);}
class TimedRuntime final: public AbstractRuntime {
public:
    const char* label;int calls=0,delay_us=0;bool fail=false,ready=true,enabled=true;
    bool last_sample_outlier=false;
    explicit TimedRuntime(const char* n):label(n){}
    const char* name()const override{return label;}
    bool initialize(const RuntimeConfig& c)override{config_=c;return true;}
    void shutdown()override{ready=false;}
    bool isInitialized()const override{return ready;}
    bool linearForward(const float*,const float*,const float*,float*,int,int,int)override{return false;}
    bool forwardLayer(const std::vector<const std::vector<float>*>&,std::vector<std::vector<float>>&,const Layer&,bool)override{return false;}
    bool supportsOptimizer()const override{return enabled;}
    bool optimizerStep(const RuntimeOptimizerRequest& r)override {
        ++calls;
        int delay=delay_us;
        if(last_sample_outlier)delay=(calls==1||calls==6)?0:2000;
        if(delay) {
            const auto until=std::chrono::steady_clock::now()+std::chrono::microseconds(delay);
            while(std::chrono::steady_clock::now()<until)std::this_thread::yield();
        }
        if(fail){r.weights[0]=900;if(r.first_moment)r.first_moment[0]=900;return false;}
        optimizerHostStep(r);return true;
    }
};
struct State {
    Optimizer opt;std::vector<float>w,g,m,v;
    explicit State(size_t n=37):w(n,.25f),g(n,.125f),m(n,.01f),v(n,.02f){opt.type=OptimizerType::ADAM;opt.step=3;}
    RuntimeOptimizerRequest request(){return {&opt,.01f,w.data(),g.data(),m.data(),v.data(),w.size()};}
};
static void step(RuntimeRouter& router,State& s,AbstractRuntime* expected) {
    auto w=s.w,m=s.m,v=s.v;const auto counter=s.opt.step;
    RuntimeOptimizerRequest ref{&s.opt,.01f,w.data(),s.g.data(),m.data(),v.data(),w.size()};optimizerHostStep(ref);
    AbstractRuntime* selected=nullptr;
    require(router.dispatchOptimizer(s.request(),&selected),"optimizer dispatch failed");
    require(selected==expected,"fastest measured runtime was not selected");
    require(s.w==w&&s.m==m&&s.v==v,"calibration changed real state or updated twice");
    require(s.opt.step==counter,"calibration changed optimizer counter");
}
int main(){try{
    auto& router=RuntimeRouter::instance();
    TimedRuntime first("CUDA"),second("OPENCL");State s;
    first.delay_us=2000;second.delay_us=0;
    router.setRuntimes(nullptr,&first,nullptr,&second,nullptr);
    step(router,s,&second);int probes=first.calls;int wins=second.calls;
    step(router,s,&second);require(first.calls==probes&&second.calls==wins+1,"cached route reprobed on every update");
    // Exactly 1024 committed updates per shape trigger refresh, not 1024 probe calls.
    for(int i=2;i<1024;++i)step(router,s,&second);
    first.delay_us=0;second.delay_us=2000;
    step(router,s,&first);require(first.calls>probes,"changed performance was never remeasured");
    // Same pointers and same shape, but replaced runtime configuration.
    first.delay_us=2000;second.delay_us=0;
    router.setRuntimes(nullptr,&first,nullptr,&second,nullptr);step(router,s,&second);
    // Losing the previously selected backend must not alter live moments.
    second.fail=true;step(router,s,&first);second.fail=false;
    step(router,s,&second); // failure forces the next update to recalibrate
    // Device changes also separate cache entries without requiring setRuntimes.
    first.delay_us=0;second.delay_us=2000;RuntimeConfig device;device.device_index=1;
    first.initialize(device);step(router,s,&first);
    // Temporarily losing a candidate invalidates routes even without setRuntimes.
    second.enabled=false;step(router,s,&first);second.enabled=true;
    first.delay_us=2000;second.delay_us=0;step(router,s,&second);
    // A single anomalously fast final measurement must not beat a stable backend.
    first.calls=0;first.last_sample_outlier=true;second.delay_us=200;
    router.setRuntimes(nullptr,&first,nullptr,&second,nullptr);step(router,s,&second);
    first.last_sample_outlier=false;first.delay_us=0;second.delay_us=2000;
    router.setRuntimes(nullptr,&first,nullptr,&second,nullptr);step(router,s,&first);
    // State transfer layout and exact size are part of the performance key.
    probes=second.calls;s.opt.type=OptimizerType::SGD;step(router,s,&first);
    auto request=s.request();request.first_moment=nullptr;request.second_moment=nullptr;
    require(router.dispatchOptimizer(request),"SGD without moments failed");
    require(second.calls>=probes+12,"different optimizer/state layouts shared a stale route");
    State larger(41);probes=second.calls;step(router,larger,&first);require(second.calls>=probes+6,"different sizes shared a stale route");
    TimedRuntime host("CPU");State cpu_state;
    first.delay_us=2000;
    router.setRuntimes(nullptr,&first,nullptr,nullptr,&host);step(router,cpu_state,&host);
    router.setRuntimes(nullptr,&first,nullptr,&second,nullptr);
    first.fail=true;second.fail=true;auto before=larger.w;auto before_m=larger.m;AbstractRuntime* selected=&first;
    require(!router.dispatchOptimizer(larger.request(),&selected)&&!selected,"failed routes accepted");
    require(larger.w==before&&larger.m==before_m,"failed calibration/dispatch corrupted state");
    router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
    std::cout<<"OPTIMIZER_SELECTION_OK median, refresh, generation, device, atomic state\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
