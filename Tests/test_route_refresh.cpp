#include "test_utils.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "Layers.hpp"
#include <limits>
class MutableRuntime final : public AbstractRuntime {
public:
    bool ready=true, enabled=true, poison=false, fail=false;
    const char* name() const override { return "mutable"; }
    bool initialize(const RuntimeConfig&) override {ready=true;return true;}
    void shutdown() override {ready=false;}
    bool isInitialized() const override {return ready;}
    bool linearForward(const float*,const float*,const float*,float*,int,int,int) override {return false;}
    RuntimeCapabilityLevel queryForwardCapability(LayerType t) const override {
        return enabled && t==LayerType::Identity ? RuntimeCapabilityLevel::Native : RuntimeCapabilityLevel::Unsupported;
    }
    RuntimeCapabilityLevel queryBackwardCapability(LayerType t) const override {return queryForwardCapability(t);}
    bool forwardLayer(const std::vector<const std::vector<float>*>& in,std::vector<std::vector<float>>& out,const Layer&,bool) override {
        out={*in[0]}; if(poison) out[0][0]=std::numeric_limits<float>::infinity(); return !fail;
    }
    bool backwardLayer(const std::vector<const std::vector<float>*>&,const std::vector<const std::vector<float>*>& dy,std::vector<std::vector<float>>& dx,Layer&,bool) override {
        dx={*dy[0]}; if(poison) dx[0][0]=std::numeric_limits<float>::quiet_NaN(); return !fail;
    }
};
int main() {
    MutableRuntime fast; CpuRuntime cpu; cpu.initialize(RuntimeConfig{});
    auto& router=RuntimeRouter::instance();router.setRuntimes(&fast,nullptr,nullptr,nullptr,&cpu);
    std::vector<Layer> layers;layers.emplace_back("id","Identity",0);
    std::vector<float> x{-2,3},dy{4,5};std::vector<std::vector<float>> out;
    AbstractRuntime* selected=nullptr;
    for(int mode=0;mode<6;++mode) {
        fast.enabled=mode!=1;fast.ready=mode!=2;fast.poison=mode==3;fast.fail=mode==4;
        router.planForwardLayerRoutes(layers);
        TASSERT_TRUE(router.dispatchForwardLayer({&x},out,layers[0],false,&selected));
        TASSERT_TRUE(out==std::vector<std::vector<float>>({x}));
        TASSERT_TRUE(selected==((mode==0||mode==5)?static_cast<AbstractRuntime*>(&fast):&cpu));
        TASSERT_TRUE(router.dispatchBackwardLayer({&x},{&dy},out,layers[0],true,&selected));
        TASSERT_TRUE(out==std::vector<std::vector<float>>({dy}));
        TASSERT_TRUE(selected==((mode==0||mode==5)?static_cast<AbstractRuntime*>(&fast):&cpu));
    }
    // Same object address, changed operation; all route metadata must refresh.
    layers[0]=Layer("relu","ReLU",0);router.planForwardLayerRoutes(layers);
    TASSERT_TRUE(router.dispatchForwardLayer({&x},out,layers[0],false,&selected));
    TASSERT_TRUE(selected==&cpu && out[0]==std::vector<float>({0,3}));
    TASSERT_TRUE(router.dispatchBackwardLayer({&x},{&dy},out,layers[0],true,&selected));
    TASSERT_TRUE(selected==&cpu && out[0]==std::vector<float>({0,5}));
    router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
}
