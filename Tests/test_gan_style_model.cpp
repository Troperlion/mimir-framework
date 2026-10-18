#include "test_utils.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Models/Vision/GanStyleModel.hpp"
#include <cmath>
#include <limits>

int main() {
    auto cfg=ModelArchitectures::defaultConfig("Gan_Style");
    cfg["image_w"]=8; cfg["image_h"]=8; cfg["base_channels"]=4; cfg["discriminator_channels"]=4;
    cfg["content_weight"]=0.f; // Prove D -> image -> G differentiation without the L1 term.
    auto model=ModelArchitectures::create("Gan_Style",cfg);
    TASSERT_TRUE(dynamic_cast<GanStyleModel*>(model.get())!=nullptr);
    model->allocateParams(); model->initializeWeights("xavier",42);
    std::vector<float> source(192), style(192);
    for(size_t i=0;i<source.size();++i) { source[i]=std::sin(i*.13f)*.5f; style[i]=std::cos(i*.17f)*.5f; }
    auto* discriminator=model->getLayerByName("d/conv1");
    std::vector<float> d_before(discriminator->weight_block->getData(),
        discriminator->weight_block->getData()+discriminator->getWeightsSize());
    const auto before=model->forwardPass(source,false);
    TASSERT_TRUE(before.size()==source.size());
    Optimizer opt; opt.type=OptimizerType::ADAM; opt.decay_strategy=LRDecayStrategy::NONE;
    Model::TrainStepRequest req;
    req.float_inputs["__input__"]=&source; req.target=&style;
    req.optimizer=&opt; req.learning_rate=.001f; req.capture_prediction=true;
    const auto result=model->trainStep(req);
    TASSERT_TRUE(result.has_value() && std::isfinite(result->loss));
    TASSERT_TRUE(result->prediction.size()==source.size());
    TASSERT_TRUE(result->metrics.at("d_real")>0);
    TASSERT_TRUE(result->grad_norm>0);
    bool d_changed=false;
    for(size_t i=0;i<d_before.size();++i) d_changed |= d_before[i]!=discriminator->weight_block->getData()[i];
    TASSERT_TRUE(d_changed);
    const auto after=model->forwardPass(source,false);
    double difference=0; for(size_t i=0;i<before.size();++i) difference+=std::abs(before[i]-after[i]);
    TASSERT_TRUE(difference>1e-6);
    TASSERT_TRUE(model->getLayerByName("gan_style/output")->inputs[0]=="g/image");
    req.mode=Model::TrainStepMode::Accumulate;
    bool rejected=false; try { model->trainStep(req); } catch(const std::invalid_argument&) { rejected=true; }
    TASSERT_TRUE(rejected);
    // Inference must not execute the discriminator, even if its weights are invalid.
    for(size_t i=0;i<discriminator->getWeightsSize();++i)
        discriminator->weight_block->getData()[i]=std::numeric_limits<float>::quiet_NaN();
    const auto inference=model->forwardPass(source,false);
    for(size_t i=0;i<inference.size();++i) TASSERT_TRUE(std::isfinite(inference[i]) && inference[i]==after[i]);
    return 0;
}
