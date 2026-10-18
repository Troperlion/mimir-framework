#include "test_utils.hpp"
#include "Models/MLP/BasicMLPModel.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/tpu/TpuRuntime.hpp"
#include <cstdlib>
#include <iostream>
int main() {
    const char* enabled=std::getenv("MIMIR_TPU");
    if(!enabled || std::string(enabled)!="1") { std::cout<<"SKIP: set MIMIR_TPU=1 on a TPU host\n";return 77; }
    BasicMLPModel model;BasicMLPModel::Config cfg;
    cfg.input_dim=4;cfg.hidden_dim=8;cfg.output_dim=2;cfg.hidden_layers=1;
    model.buildFromConfig(cfg);model.allocateParams();model.initializeWeights("xavier",123);
    TpuRuntime* tpu=Mimir::tpuSessionRuntime();TASSERT_TRUE(tpu && tpu->isInitialized());
    const std::vector<float> input={.5f,-1,.25f,2},target={1,-1};
    Optimizer opt;opt.type=OptimizerType::SGD;opt.decay_strategy=LRDecayStrategy::NONE;
    std::vector<float> before;
    for(const auto& l:model.getLayers())if(l.getWeightsSize())before.insert(before.end(),l.getWeights(),l.getWeights()+l.getWeightsSize());
    float initial=0,last=0;
    for(int step=0;step<30;++step) {
        auto result=model.trainStep(input,target,opt,.03f);
        if(!step)initial=result.loss;
        last=result.loss;
        TASSERT_TRUE(std::isfinite(result.loss));
        TASSERT_TRUE(result.grad_norm>0);
    }
    TASSERT_TRUE(last<initial*.5f);TASSERT_TRUE(opt.step==30);
    size_t offset=0;bool changed=false;
    for(const auto& l:model.getLayers()) {
        TASSERT_TRUE(RuntimeRouter::instance().selectForwardRuntimeForLayer(l)==tpu);
        TASSERT_TRUE(RuntimeRouter::instance().selectBackwardRuntimeForLayer(l)==tpu);
        for(size_t i=0;i<l.getWeightsSize();++i)if(l.getWeights()[i]!=before[offset+i])changed=true;
        offset+=l.getWeightsSize();
    }
    TASSERT_TRUE(changed);
    std::cout<<"TPU_MODEL_TRAIN_OK selected_runtime=TPU initial="<<initial<<" final="<<last<<'\n';
    return 0;
}
