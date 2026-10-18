#include "runtimes/losses/LossDispatch.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/opencl/OpenCLRuntime.hpp"
#include "runtimes/vulkan/VulkanRuntime.hpp"
#include "runtimes/cuda/CudaRuntime.hpp"
#include "runtimes/rocm/RocmRuntime.hpp"
#include "runtimes/tpu/TpuRuntime.hpp"
#include "include/json.hpp"
#include "runtimes/portable/PortableDevice.hpp"
#include <array>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace RuntimeLossGrad;
static void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static void compare(const LossWithGrad& a,const LossWithGrad& b){
    require(a.grad.size()==b.grad.size(),"gradient size");
    if(!std::isfinite(a.loss) || std::abs(a.loss-b.loss)>3e-5+3e-5*std::abs(b.loss)) {
        std::cerr<<"loss got="<<a.loss<<" expected="<<b.loss<<'\n';throw std::runtime_error("loss parity");
    }
    require(std::abs(a.relation_loss-b.relation_loss)<3e-5+3e-5*std::abs(b.relation_loss),"relation loss parity");
    for(size_t i=0;i<a.grad.size();++i)if(!std::isfinite(a.grad[i]) || std::abs(a.grad[i]-b.grad[i])>2e-5f+5e-4f*std::abs(b.grad[i])){
        std::cerr<<"gradient "<<i<<" got="<<a.grad[i]<<" expected="<<b.grad[i]<<'\n';throw std::runtime_error("gradient parity");
    }
}
class FailingLossRuntime final:public AbstractRuntime {
public:
    const char* name()const override{return "FAILING_LOSS";}
    bool initialize(const RuntimeConfig& cfg)override{config_=cfg;return true;}
    void shutdown()override{}
    bool isInitialized()const override{return true;}
    bool linearForward(const float*,const float*,const float*,float*,int,int,int)override{return false;}
    bool forwardLayer(const std::vector<const std::vector<float>*>&,std::vector<std::vector<float>>&,const Layer&,bool)override{return false;}
    bool supportsReconstructionLoss()const override{return true;}
    bool reconstructionLoss(const Request&,LossWithGrad& out)override{out.loss=1000;out.grad={1000};return false;}
};
int main(int argc,char** argv){try {
    std::string mode=argc>1?argv[1]:"CPU";
    std::unique_ptr<AbstractRuntime> rt;
    if(mode=="CPU")rt=std::make_unique<CpuRuntime>();
#ifdef ENABLE_OPENCL
    if(mode=="OPENCL")rt=std::make_unique<OpenCLRuntime>();
#endif
#ifdef ENABLE_VULKAN
    if(mode=="VULKAN")rt=std::make_unique<VulkanRuntime>();
#endif
#ifdef ENABLE_CUDA
    if(mode=="CUDA")rt=std::make_unique<CudaRuntime>();
#endif
#ifdef ENABLE_ROCM
    if(mode=="ROCM")rt=std::make_unique<RocmRuntime>();
#endif
    if(mode=="TPU")rt=std::make_unique<TpuRuntime>();
    if(!rt && mode!="fixture"){std::cout<<"SKIP: backend not compiled\n";return 77;}
    if(rt){RuntimeConfig cfg;cfg.backend=mode;if(!rt->initialize(cfg)){std::cout<<"SKIP: backend unavailable\n";return 77;}}
    if(mode=="OPENCL" || mode=="VULKAN") {
        auto gpu=PortableStaging::createLossDevice(mode,0);
        if(!gpu){std::cout<<"SKIP: physical GPU unavailable\n";return 77;}
    }
    nlohmann::json fixtures=nlohmann::json::array();int cases=0;
    auto check=[&](Request r){
        LossWithGrad expected;require(host(r,expected),"host reference");
        if(mode=="fixture") {
            auto args=parameters(r);std::vector<float> params;
            for(size_t i=9;i<15;++i){float f;std::memcpy(&f,&args[i],4);params.push_back(f);}
            fixtures.push_back({{"memory",pack(r)},{"count",r.count},{"op",int(r.operation)},{"kind",args[7]},
                {"relation_only",args[8]!=0},{"width",r.width},{"height",r.height},{"channels",r.channels},
                {"parameters",params},{"relation_loss",expected.relation_loss},{"loss",expected.loss},{"grad",expected.grad}});
        } else {LossWithGrad actual;require(rt->reconstructionLoss(r,actual),"native reconstruction execution");try{compare(actual,expected);}catch(...){std::cerr<<"op="<<int(r.operation)<<" shape="<<r.width<<'x'<<r.height<<'x'<<r.channels<<" case="<<cases<<'\n';throw;}}
        ++cases;
    };
    std::vector<float> p,t,weights;
    for(size_t n:{size_t(1),size_t(257),size_t(16385)}) {
        p.resize(n);t.resize(n);weights.resize(n);
        for(size_t i=0;i<n;++i){p[i]=float(std::sin(i*.371+1.1)*.47+.5);t[i]=float(std::cos(i*.217+.5)*.45+.5);weights[i]=i%5==0?-.1f:float(i%7)*.3f;}
        if(n>4){p[0]=0;p[1]=1;p[2]=1e-8f;p[3]=std::nextafter(1.f,0.f);p[4]=std::nextafter(p[3],0.f);}
        for(const char* name:{"mse","l2","mae","l1","huber","smoothl1","smooth_l1","charbonnier","bce","gaussian_nll","gaussian-nll","nll_gaussian","unknown"})
            for(bool weighted:{false,true}) {
                Request r;r.prediction=p.data();r.target=t.data();r.count=n;r.type=name;r.options.weights=weighted?&weights:nullptr;
                r.options.huber_delta=.15f;r.options.charbonnier_eps=3e-5f;r.options.gaussian_nll_sigma=.7f;check(r);
                r.options.relation_only=true;check(r);
            }
    }
    for(bool chw:{false,true})for(const char* predicate:{"left_of","above","overlaps"}) {
        const int w=7,h=5,c=3;size_t n=size_t(w)*h*c;
        p.resize(n);t.resize(n);weights.resize(n);
        for(size_t i=0;i<n;++i){p[i]=float(std::sin(i*.13));t[i]=float(std::cos(i*.31));weights[i]=i%7?1.f:0.f;}
        SpatialRelation relation;relation.width=w;relation.height=h;relation.channels=c;relation.chw=chw;relation.weight=.7f;relation.predicate=predicate;
        relation.subject={.03,.1,.55,.83};relation.object={.34,.27,.98,1.};
        auto other=relation;other.weight=.2f;other.subject={0,0,.8,.9};
        std::vector<SpatialRelation> relations{relation,other};
        std::vector<float> partial(weights.begin(),weights.begin()+n/2);
        for(bool only:{false,true})for(int weighting:{0,1,2}) {
            Request r;r.prediction=p.data();r.target=t.data();r.count=n;r.type="huber";
            r.options.relations=&relations;r.options.relation_only=only;
            r.options.weights=weighting==1?&weights:weighting==2?&partial:nullptr;check(r);
        }
    }
    for(auto shape:{std::array<int,3>{1,1,3},{1,7,2},{7,1,3},{13,11,3},{32,32,1},{3,2,8}}) {
        int w=shape[0],h=shape[1],c=shape[2];size_t n=size_t(w)*h*c;
        p.resize(n);t.resize(n);
        for(size_t i=0;i<n;++i){p[i]=float(std::sin(i*.371+.73));t[i]=float(std::cos(i*.217+.31));}
        Request r;r.prediction=p.data();r.target=t.data();r.count=n;r.width=w;r.height=h;r.channels=c;
        for(auto op:{Operation::SSIM,Operation::Spectral,Operation::Pool,Operation::PoolBackward}){r.operation=op;check(r);}
        r.operation=Operation::SSIM;std::fill(p.begin(),p.end(),.3f);std::fill(t.begin(),t.end(),.3f);check(r);
        r.operation=Operation::Spectral;std::fill(t.begin(),t.end(),0.f);check(r);
    }
    for(int vocab:{2,17,513}) {
        const int tokens=7;size_t n=size_t(vocab)*tokens;p.resize(n);t.assign(n,-1.f);
        for(size_t i=0;i<n;++i)p[i]=1000.f+float(std::sin(i*.13)*3.7);
        t[0]=0;t[1]=vocab-1;t[3]=vocab/2;t[6]=1;
        Request r;r.operation=Operation::CrossEntropy;r.count=n;r.width=vocab;r.height=tokens;r.channels=1;r.prediction=p.data();r.target=t.data();check(r);
        std::fill(t.begin(),t.end(),-1.f);check(r);
    }
    if(mode=="fixture"){require(argc>2,"fixture path");std::ofstream(argv[2])<<fixtures.dump();}
    else {
        // Public entry points must use the registered runtime, including GPU-only setups.
        auto& router=RuntimeRouter::instance();
        if(mode=="TPU")router.setRuntimes(nullptr,nullptr,nullptr,nullptr,rt.get(),nullptr);
        else router.setRuntimes(nullptr,nullptr,mode=="VULKAN"?rt.get():nullptr,mode=="OPENCL"?rt.get():nullptr,mode=="CPU"?rt.get():nullptr);
        if(mode=="CUDA")router.setRuntimes(nullptr,rt.get(),nullptr,nullptr,nullptr);
        if(mode=="ROCM")router.setRuntimes(rt.get(),nullptr,nullptr,nullptr,nullptr);
        auto actual=pixel_loss_and_grad(p,t,"huber");
        compare(actual,Host::pixel_loss_and_grad(p,t,"huber",{}));
        p.resize(13*11*3);t.resize(p.size());
        for(size_t i=0;i<p.size();++i){p[i]=float(std::sin(i*.13));t[i]=float(std::cos(i*.31));}
        auto ssim=ssim_global_hwc(p,t,13,11,3,.01f,.03f,2.f);
        auto hs=Host::ssim_global_hwc(p,t,13,11,3,.01f,.03f,2.f);
        compare({ssim.loss,ssim.grad,0},{hs.loss,hs.grad,0});
        compare(spectral_dct_l1_hwc(p,t,13,11,3),Host::spectral_dct_l1_hwc(p,t,13,11,3));
        std::vector<float> pooled,hp,back,hb;int ow,oh,hw,hh;
        avgpool2x2_hwc(p,13,11,3,pooled,ow,oh);Host::avgpool2x2_hwc(p,13,11,3,hp,hw,hh);
        require(ow==hw && oh==hh,"pool dimensions");compare({0,pooled,0},{0,hp,0});
        avgpool2x2_back_hwc(pooled,13,11,3,back);Host::avgpool2x2_back_hwc(hp,13,11,3,hb);compare({0,back,0},{0,hb,0});
        auto ce=text_cross_entropy(p.data(),p.size(),{2,-1,1,0,2},5,3,0);
        std::vector<float> labels(15,-1.f);labels[0]=2;labels[2]=1;labels[4]=2;
        Request cr;cr.operation=Operation::CrossEntropy;cr.prediction=p.data();cr.target=labels.data();cr.width=3;cr.height=5;cr.channels=1;cr.count=15;
        LossWithGrad hc;host(cr,hc);compare(ce,hc);
        require(router.reconstructionLossDispatchCounts().at(mode)>=6,"public losses not routed to selected runtime");
        if(mode=="CPU") {
            FailingLossRuntime failing;CpuRuntime cpu;RuntimeConfig cfg;cpu.initialize(cfg);
            router.setRuntimes(&failing,nullptr,nullptr,nullptr,&cpu);
            Request r;r.prediction=p.data();r.target=t.data();r.count=p.size();
            LossWithGrad result,expected;host(r,expected);AbstractRuntime* selected=nullptr;
            require(router.dispatchReconstructionLoss(r,result,&selected),"CPU fallback missing");
            require(selected==&cpu,"failed native route selected");compare(result,expected);
            router.setRuntimes(&failing,nullptr,nullptr,nullptr,nullptr);result.loss=7;result.grad={8};
            require(!router.dispatchReconstructionLoss(r,result),"unexpected implicit CPU fallback");
            require(result.loss==7 && result.grad==std::vector<float>{8},"failed route overwrote output");
        }
    }
    std::cout<<"RECONSTRUCTION_RUNTIME_OK "<<mode<<" cases="<<cases<<" loss_and_grad=validated\n";return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
