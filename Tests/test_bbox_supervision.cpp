#include "test_utils.hpp"
#include "training/BBoxSupervision.hpp"
#include "runtimes/ops_loss_and_grad.hpp"
#include <numeric>

int main() {
    using namespace BBoxSupervision;
    using nlohmann::json;
    json a={{"entities", {{{"bbox",{0.,0.,0.5,1.}}}}}};
    const auto hwc=makeWeights(&a,4,2,3,Layout::HWC);
    const auto chw=makeWeights(&a,4,2,3,Layout::CHW);
    TASSERT_TRUE(hwc.bbox_count==1 && hwc.strength==1);
    for(size_t p=0;p<8;++p) for(size_t c=0;c<3;++c) {
        TASSERT_NEAR(hwc.values[p*3+c],p%4<2?1.5f:0.5f,1e-6f);
        TASSERT_NEAR(hwc.values[p*3+c],chw.values[c*8+p],0.f);
    }
    TASSERT_TRUE(makeWeights(nullptr,4,2,3,Layout::HWC).values.empty());
    TASSERT_TRUE(makeWeights(&a,4,2,3,Layout::HWC,0).values.empty());
    json empty={{"entities",json::array()}};
    TASSERT_TRUE(makeWeights(&empty,4,2,3,Layout::HWC).values.empty());
    a["entities"][0]["bbox"]={0.001,0.001,0.002,0.002};
    const auto tiny=makeWeights(&a,4,2,1,Layout::CHW);
    TASSERT_NEAR(tiny.values[0],4.5f,1e-6f);
    TASSERT_NEAR(std::accumulate(tiny.values.begin(),tiny.values.end(),0.f),8.f,1e-6f);
    a["entities"]=json::array({{{"bbox",{0.,0.,0.25,1.}}},{{"bbox",{0.5,0.,1.,1.}}}});
    const auto unequal=makeWeights(&a,4,2,1,Layout::CHW);
    TASSERT_NEAR(unequal.values[0],1.5f,1e-6f);
    TASSERT_NEAR(unequal.values[2],1.f,1e-6f);
    TASSERT_NEAR(unequal.values[1],0.5f,1e-6f);
    std::vector<float> pred={1,2,3,4,5,6,7,8},target(8,0);
    RuntimeLossGrad::PixelLossOptions options; options.weights=&unequal.values;
    const auto loss=RuntimeLossGrad::pixel_loss_and_grad(pred,target,"mse",options);
    for(size_t i=0;i<pred.size();++i) {
        const float original=pred[i], eps=0.001f;
        pred[i]=original+eps; const double plus=RuntimeLossGrad::pixel_loss_and_grad(pred,target,"mse",options).loss;
        pred[i]=original-eps; const double minus=RuntimeLossGrad::pixel_loss_and_grad(pred,target,"mse",options).loss;
        pred[i]=original;
        TASSERT_NEAR(loss.grad[i],static_cast<float>((plus-minus)/(2*eps)),0.001f);
    }
    auto masked=unequal; std::vector<float> mask(8,1); mask[0]=0; combineMask(masked,mask);
    TASSERT_TRUE(masked.values[0]==0 && masked.values[2]==unequal.values[2]);
    json relational={{"entities", {{{"id","a"},{"phrase","object A"},{"bbox",{0.,0.,0.5,1.}}},
                                     {{"id","b"},{"phrase","object B"},{"bbox",{0.5,0.,1.,1.}}}}},
                     {"relations", {{{"subject","a"},{"predicate","left_of"},{"object","b"}}}}};
    const auto pair_weights=makeWeights(&relational,4,2,1,Layout::HWC);
    TASSERT_TRUE(pair_weights.relations.size()==1);
    TASSERT_NEAR(std::accumulate(pair_weights.values.begin(),pair_weights.values.end(),0.f),8.f,1e-5f);
    options.weights=&pair_weights.values;options.relations=&pair_weights.relations;
    target={0,1,0,1,2,0,2,0};pred={1,3,2,0,0,1,4,1};
    const auto relational_loss=RuntimeLossGrad::pixel_loss_and_grad(pred,target,"huber",options);
    TASSERT_TRUE(relational_loss.relation_loss>0);
    auto plain_options=options;plain_options.relations=nullptr;
    const auto no_pair=RuntimeLossGrad::pixel_loss_and_grad(pred,target,"huber",plain_options);
    bool coupled_gradient=false;
    for(size_t i=0;i<pred.size();++i) {
        coupled_gradient|=std::abs(relational_loss.grad[i]-no_pair.grad[i])>1e-6f;
        const float original=pred[i],eps=0.001f;
        pred[i]=original+eps;const double plus=RuntimeLossGrad::pixel_loss_and_grad(pred,target,"huber",options).loss;
        pred[i]=original-eps;const double minus=RuntimeLossGrad::pixel_loss_and_grad(pred,target,"huber",options).loss;
        pred[i]=original;
        TASSERT_NEAR(relational_loss.grad[i],float((plus-minus)/(2*eps)),0.001f);
    }
    TASSERT_TRUE(coupled_gradient);
    // The latent CHW path must match image HWC pooling for every channel.
    std::vector<float> hp(16),ht(16),cp(16),ct(16);
    for(size_t p=0;p<8;++p) for(size_t c=0;c<2;++c) {
        hp[p*2+c]=pred[p]+float(c); ht[p*2+c]=target[p];
        cp[c*8+p]=hp[p*2+c]; ct[c*8+p]=ht[p*2+c];
    }
    const auto hw=makeWeights(&relational,4,2,2,Layout::HWC);
    const auto cw=makeWeights(&relational,4,2,2,Layout::CHW);
    RuntimeLossGrad::PixelLossOptions ho,co;
    ho.weights=&hw.values;ho.relations=&hw.relations;
    co.weights=&cw.values;co.relations=&cw.relations;
    const auto hl=RuntimeLossGrad::pixel_loss_and_grad(hp,ht,"mse",ho);
    const auto cl=RuntimeLossGrad::pixel_loss_and_grad(cp,ct,"mse",co);
    TASSERT_NEAR(hl.loss,cl.loss,1e-6);
    for(size_t p=0;p<8;++p) for(size_t c=0;c<2;++c)
        TASSERT_NEAR(hl.grad[p*2+c],cl.grad[c*8+p],1e-6f);

    // Skeleton-only annotations work without bboxes, including unplaced Lua points.
    json skeleton={{"entities",json::array({{{"id","pose"},{"phrase","a pose"},{"skeleton",{
        {"name","custom"},{"points",json::array({{{"name","a"},{"position",{0.,0.}}},{{"name","b"},{"position",{1.,1.}},{"parent","a"}},{{"name","unset"}}})},
        {"edges",json::object()}
    }}}})}};
    const auto sw=makeWeights(&skeleton,8,4,2,Layout::HWC);
    const auto sc=makeWeights(&skeleton,8,4,2,Layout::CHW);
    TASSERT_TRUE(sw.bbox_count==0 && !sw.values.empty());
    TASSERT_NEAR(std::accumulate(sw.values.begin(),sw.values.end(),0.f),64.f,1e-4f);
    for(size_t p=0;p<32;++p)for(size_t c=0;c<2;++c)TASSERT_NEAR(sw.values[p*2+c],sc.values[c*32+p],0.f);
    TASSERT_TRUE(sw.values.front()>sw.values[14]);
    TASSERT_TRUE(makeWeights(&skeleton,8,4,2,Layout::CHW,0).values.empty());
    TASSERT_TRUE(groundedPrompt("caption",&skeleton).find("a at 0.000,0.000")!=std::string::npos);

    // Role importance scales the auxiliary loss AND its gradients directly.
    relational["entities"][0]["role"]="primary_subjet";
    relational["entities"][1]["role"]="secondary_subject";
    json role_config={{"relation_primary_weight",6.0},{"relation_secondary_weight",2.0},{"relation_other_weight",1.0}};
    const auto ranked=makeWeights(&relational,4,2,1,Layout::HWC,1,&role_config);
    RuntimeLossGrad::PixelLossOptions ranked_options;
    ranked_options.relation_only=true; ranked_options.relations=&ranked.relations;
    auto base_options=ranked_options;base_options.relations=&pair_weights.relations;
    const auto base_pair=RuntimeLossGrad::pixel_loss_and_grad(pred,target,"mse",base_options);
    const auto ranked_pair=RuntimeLossGrad::pixel_loss_and_grad(pred,target,"mse",ranked_options);
    TASSERT_NEAR(ranked_pair.relation_loss,base_pair.relation_loss*4,1e-6);
    for(size_t i=0;i<pred.size();++i) TASSERT_NEAR(ranked_pair.grad[i],base_pair.grad[i]*4,1e-6f);
    TASSERT_NEAR(roleWeight("secondary_subjet",nullptr),2.f,0.f);
    TASSERT_NEAR(roleWeight("background",nullptr),1.f,0.f);
    role_config["relation_primary_weight"]=0;role_config["relation_secondary_weight"]=0;
    const auto muted=makeWeights(&relational,4,2,1,Layout::HWC,1,&role_config);
    ranked_options.relations=&muted.relations;
    TASSERT_TRUE(RuntimeLossGrad::pixel_loss_and_grad(pred,target,"mse",ranked_options).relation_loss==0);
    role_config["relation_primary_weight"]=-1;
    bool bad_role=false;
    try { (void)makeWeights(&relational,4,2,1,Layout::HWC,1,&role_config); } catch(const std::invalid_argument&) { bad_role=true; }
    TASSERT_TRUE(bad_role);
    // Predicate orientation changes the objective; triplets survive short captions.
    relational["relations"][0]["predicate"]="above";
    const auto vertical=makeWeights(&relational,4,2,1,Layout::HWC);
    options.relations=&vertical.relations;
    const auto above_loss=RuntimeLossGrad::pixel_loss_and_grad(pred,target,"huber",options);
    TASSERT_TRUE(std::abs(above_loss.relation_loss-relational_loss.relation_loss)>1e-5);
    const auto context=groundedPrompt("caption",&relational);
    TASSERT_TRUE(context.find("entity a object A role primary_subjet bbox 0.000,0.000,0.500,1.000")!=std::string::npos);
    TASSERT_TRUE(context.find("relation a above b: object A above object B.")!=std::string::npos);
    TASSERT_TRUE(groundedPrompt("caption",nullptr)=="caption");
    const auto off=makeWeights(&relational,4,2,1,Layout::HWC,0);
    TASSERT_TRUE(off.values.empty() && off.relations.empty());
    // A masked-out pixel cannot acquire a gradient through the pair objective.
    auto masked_pair=pair_weights;std::vector<float> pair_mask(8,1);pair_mask[0]=0;combineMask(masked_pair,pair_mask);
    options.weights=&masked_pair.values;options.relations=&masked_pair.relations;
    TASSERT_TRUE(RuntimeLossGrad::pixel_loss_and_grad(pred,target,"mse",options).grad[0]==0);
    bool rejected=false;
    a["entities"][0]["bbox"]={0.5,0.,0.25,1.};
    try { (void)makeWeights(&a,4,2,1,Layout::CHW); } catch(const std::invalid_argument&) { rejected=true; }
    TASSERT_TRUE(rejected);
    std::cout << "PASS bbox rasterization, layout, small boxes, equal entity mass, finite difference gradients and masks\n";
}
