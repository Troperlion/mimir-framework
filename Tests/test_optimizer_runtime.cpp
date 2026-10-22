#include "Model.hpp"
#include "runtimes/optimizers/OptimizerDispatch.hpp"
#include "runtimes/portable/PortableDevice.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/opencl/OpenCLRuntime.hpp"
#include "runtimes/vulkan/VulkanRuntime.hpp"
#include "runtimes/tpu/TpuRuntime.hpp"
#ifdef ENABLE_CUDA
#include "runtimes/cuda/CudaRuntime.hpp"
#endif
#ifdef ENABLE_ROCM
#include "runtimes/rocm/RocmRuntime.hpp"
#endif
#include <fstream>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <iostream>
static void require(bool b,const char* what){if(!b)throw std::runtime_error(what);}
struct Buffers {
    std::vector<float> w,g,m,v;
    explicit Buffers(size_t n):w(n),g(n),m(n),v(n){for(size_t i=0;i<n;++i){w[i]=std::sin(float(i+1))*.3f;g[i]=float(int(i%11)-5)*.02f;m[i]=.001f;v[i]=.01f;}}
    RuntimeOptimizerRequest request(Optimizer& o){return {&o,.01f,w.data(),g.data(),o.usesFirstMoment()?m.data():nullptr,o.usesSecondMoment()?v.data():nullptr,w.size()};}
};
static void compare(const Buffers& a,const Buffers& b){
    auto compareVector=[](const std::vector<float>& actual,const std::vector<float>& expected) {
        require(actual.size()==expected.size(),"optimizer buffer size");
        for(size_t i=0;i<actual.size();++i) {
            const float x=actual[i],y=expected[i];
            if(!std::isfinite(x) || std::abs(x-y)>4e-5f+5e-4f*std::abs(y)) {
                std::cerr<<i<<" got="<<x<<" expected="<<y<<'\n';
                throw std::runtime_error("optimizer numerical parity");
            }
        }
    };
    compareVector(a.w,b.w);
    compareVector(a.m,b.m);
    compareVector(a.v,b.v);
}
class MockRuntime final:public AbstractRuntime {
public:
    bool fail=false;int calls=0;const char* label;
    explicit MockRuntime(const char* value):label(value){}
    const char* name()const override{return label;}
    bool initialize(const RuntimeConfig& c)override{config_=c;return true;}
    void shutdown()override{}
    bool isInitialized()const override{return true;}
    bool linearForward(const float*,const float*,const float*,float*,int,int,int)override{return false;}
    bool forwardLayer(const std::vector<const std::vector<float>*>&,std::vector<std::vector<float>>&,const Layer&,bool)override{return false;}
    bool supportsOptimizer()const override{return true;}
    bool optimizerStep(const RuntimeOptimizerRequest& r)override{++calls;if(fail){r.weights[0]=1000;r.first_moment[0]=1000;return false;}optimizerHostStep(r);return true;}
};
int main(int argc,char**argv){try {
    const std::string mode=argc>1?argv[1]:"router";
    for (size_t count : {size_t{1}, size_t{257}, size_t{4096}}) {
        Optimizer sgd;
        sgd.type = OptimizerType::SGD;
        sgd.step = 1;
        sgd.weight_decay = 0.5f; // Existing plain SGD semantics ignore this field.
        Buffers actual(count), expected = actual;
        auto request = actual.request(sgd);
        // Optional stale moments must remain untouched by the compact path.
        request.first_moment = actual.m.data();
        request.second_moment = actual.v.data();
        optimizerHostStep(expected.request(sgd));
        auto compact = optimizerPack(request, true);
        require(compact.size() == 2*count, "SGD staging contains unused state");
        auto parameters = optimizerParameters(request);
        require(parameters.size() == 19 && parameters[2] == 0, "SGD parameter ABI");
        float lr;
        std::memcpy(&lr, &parameters[3], sizeof(float));
        require(lr == request.learning_rate, "SGD learning rate");
        for (size_t i = 0; i < count; ++i) compact[i] -= lr*compact[count+i];
        require(optimizerCommit(request, compact), "compact SGD commit");
        compare(actual, expected);
        // The default layout stays compatible with the TPU worker protocol.
        auto legacy = optimizerPack(request);
        require(legacy.size() == 5*count+2*((count+255)/256)+2, "legacy staging layout");
        require(optimizerCommit(request, legacy), "legacy SGD commit");
        compare(actual, expected);
        compact.pop_back();
        require(!optimizerCommit(request, compact), "truncated SGD staging accepted");
    }

    nlohmann::json fixtures=nlohmann::json::array();
    std::unique_ptr<AbstractRuntime> runtime;
#ifdef ENABLE_OPENCL
    if(mode=="OPENCL")runtime=std::make_unique<OpenCLRuntime>();
#endif
#ifdef ENABLE_VULKAN
    if(mode=="VULKAN")runtime=std::make_unique<VulkanRuntime>();
#endif
    if(mode=="TPU")runtime=std::make_unique<TpuRuntime>();
#ifdef ENABLE_CUDA
    if(mode=="CUDA")runtime=std::make_unique<CudaRuntime>();
#endif
#ifdef ENABLE_ROCM
    if(mode=="ROCM")runtime=std::make_unique<RocmRuntime>();
#endif
    if(!runtime && mode!="router" && mode!="fixture"){std::cout<<"SKIP: runtime not compiled\n";return 77;}
    if(runtime){RuntimeConfig cfg;cfg.backend=mode;if(!runtime->initialize(cfg)){std::cout<<"SKIP: device unavailable\n";return 77;}}
    if(mode=="OPENCL" || mode=="VULKAN") {
        auto physical_gpu=PortableStaging::createOptimizerDevice(mode,0);
        if(!physical_gpu){std::cout<<"SKIP: native optimizer GPU unavailable (software devices excluded)\n";return 77;}
    }
    if(mode=="router"){
        Optimizer o;o.type=OptimizerType::ADAM;o.step=1;Buffers a(257),reference=a;auto r=a.request(o);optimizerHostStep(reference.request(o));
        MockRuntime broken("BROKEN"),native("NATIVE");broken.fail=true;
        auto& router=RuntimeRouter::instance();router.setRuntimes(&broken,&native,nullptr,nullptr,nullptr);
        AbstractRuntime* selected=nullptr;require(router.dispatchOptimizer(r,&selected),"dispatch native optimizer");require(selected==&native,"wrong selected backend");compare(a,reference);
        require(o.step==1,"calibration changed step");require(broken.calls>0 && native.calls>0,"no capability probing");
        const int native_calls=native.calls,failed_calls=broken.calls;o.step=2;
        optimizerHostStep(reference.request(o));
        require(router.dispatchOptimizer(a.request(o),&selected),"cached optimizer route");
        require(native.calls==native_calls+1 && broken.calls==failed_calls,"calibration cache not reused");compare(a,reference);
        require(!router.dispatchOptimizer(RuntimeOptimizerRequest{},&selected) && !selected,"invalid request accepted");
        // Failure must not corrupt weights/moments when every native route fails.
        Buffers failed(257),before=failed;router.setRuntimes(&broken,nullptr,nullptr,nullptr,nullptr);
        require(!router.dispatchOptimizer(failed.request(o)),"failing optimizer accepted");compare(failed,before);
        MockRuntime cpu("CPU");Buffers fallback(257),fallback_reference=fallback;
        optimizerHostStep(fallback_reference.request(o));
        router.setRuntimes(&broken,nullptr,nullptr,nullptr,&cpu);
        require(router.dispatchOptimizer(fallback.request(o),&selected) && selected==&cpu,"CPU fallback route");
        compare(fallback,fallback_reference);
        require(router.optimizerDispatchCounts().at("CPU")>0,"optimizer runtime audit missing");
        router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
    }
    // Stable device norms must match the CPU's double accumulators even when
    // naive float squares would underflow (LAMB) or overflow (Adafactor).
    for(int kind:{4,8}) {
        Optimizer o;o.type=OptimizerType(kind);o.step=1;o.weight_decay=0.f;
        Buffers actual(257);
        if(kind==4)for(size_t i=0;i<actual.w.size();++i)actual.w[i]=1e25f*float(i%3+1);
        else {std::fill(actual.g.begin(),actual.g.end(),1e-30f);std::fill(actual.m.begin(),actual.m.end(),0.f);std::fill(actual.v.begin(),actual.v.end(),0.f);}
        Buffers reference=actual;auto request=actual.request(o);auto input=optimizerPack(request);
        optimizerHostStep(reference.request(o));
        if(runtime)require(runtime->optimizerStep(request),"stable norm execution");else optimizerHostStep(request);
        compare(actual,reference);
        if(mode=="fixture") {const auto p=optimizerParameters(request);std::vector<float> params(16);
            for(size_t i=0;i<16;++i)std::memcpy(&params[i],&p[3+i],4);
            fixtures.push_back({{"kind",kind},{"count",257},{"parameters",params},{"input",input},{"expected",optimizerPack(reference.request(o))}});
        }
    }
    for(int kind=0;kind<9;++kind)for(size_t count:{size_t(1),size_t(257),size_t(16385)})for(int variant=0;variant<2;++variant){
        Optimizer o;o.type=OptimizerType(kind);o.decay_strategy=LRDecayStrategy::NONE;o.beta1=.8f;o.beta2=.9f;
        o.adafactor_beta1=variant?.7f:0.f;o.adafactor_relative_step=variant;o.adafactor_scale_parameter=!variant;
        Buffers actual(count),reference=actual;
        for(int step=1;step<=12;++step){
            o.step=step;auto r=actual.request(o);auto ref=reference.request(o);
            if(mode=="fixture" && count==257 && step<=3){auto p=optimizerParameters(r);std::vector<float> params(16);for(size_t i=0;i<16;++i)std::memcpy(&params[i],&p[i+3],4);
                auto input=optimizerPack(r);optimizerHostStep(ref);
                fixtures.push_back({{"kind",kind},{"count",count},{"parameters",params},{"input",input},{"expected",optimizerPack(ref)}});
                actual=reference;
            }else{
                optimizerHostStep(ref);
                if(runtime){require(runtime->supportsOptimizer(),"optimizer support missing");require(runtime->optimizerStep(r),"native optimizer execution failed");}
                else optimizerHostStep(r);
                compare(actual,reference);
            }
        }
    }
    if(mode=="fixture"){require(argc==3,"fixture output path required");std::ofstream(argv[2])<<fixtures.dump();}
    std::cout<<"OPTIMIZER_RUNTIME_OK "<<mode<<" all=9 checkpoint_moments=preserved\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
