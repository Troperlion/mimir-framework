#include "test_utils.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Models/Vision/CNN/MedicalDetectorModel.hpp"
#include "training/BBoxSupervision.hpp"
#include <cmath>

int main() {
    setTestEnvironment("OMP_NUM_THREADS","1");
    setTestEnvironment("MIMIR_RUNTIME","cpu");
    const json annotation={{"entities", {{{"phrase","subject"},{"bbox",{0.,0.,0.25,1.}}}}}};
    for(const std::string type:{"resnet","mobilenet","vgg16","vgg19","vgg16_feat","unet","vae","vit","gan_latent","medical_detector"}) {
        json cfg=ModelArchitectures::defaultConfig(type);
        cfg.update(json{{"image_w",32},{"image_h",32},{"image_c",3},{"base_channels",4},
            {"num_classes",2},{"fc_hidden",8},{"dropout",0},{"depth",1},
            {"latent_dim",8},{"hidden_dim",16},{"num_hidden_layers",1},
            {"prompt_dim",4},{"noise_dim",4},{"num_tokens",5},{"d_model",8},
            {"num_heads",2},{"num_layers",1},{"mlp_hidden",16},{"output_dim",2}});
        if(type=="vit") cfg.update(json{{"bbox_input_w",2},{"bbox_input_h",2},{"bbox_input_c",8},{"bbox_input_layout","HWC"}});
        if(type=="gan_latent") cfg.update(json{{"bbox_output_w",4},{"bbox_output_h",2},{"bbox_output_c",1}});
        if(type=="medical_detector") { cfg["image_w"]=64; cfg["image_h"]=64; }
        auto model=ModelArchitectures::create(type,cfg);
        TASSERT_TRUE(model);
        model->allocateParams(); model->initializeWeights("xavier",123U);
        std::vector<float> input(model->modelConfig.value("input_dim",0));
        for(size_t i=0;i<input.size();++i) input[i]=0.25f+0.1f*std::sin(static_cast<float>(i));
        std::vector<float> target(model->modelConfig.value("output_dim",0),0.3f);
        Optimizer optimizer; optimizer.type=OptimizerType::SGD;
        Model::TrainStepRequest request; request.float_inputs["__input__"]=&input;
        request.target=&target; request.optimizer=&optimizer; request.mode=Model::TrainStepMode::Accumulate;
        const auto historical=model->trainStep(request);
        TASSERT_TRUE(historical && std::isfinite(historical->loss));
        model->zeroGradients(); request.annotation=&annotation;
        const auto weighted=model->trainStep(request);
        TASSERT_TRUE(weighted && std::isfinite(weighted->loss) && std::isfinite(weighted->grad_norm));
        TASSERT_TRUE(weighted->metrics.at("bbox_count")==1 && weighted->metrics.at("bbox_loss_weight")==1);
        json linked=annotation;
        linked["entities"][0]["id"]="a";
        linked["entities"].push_back(json{{"id","b"},{"phrase","object"},{"bbox",{0.5,0.,1.,1.}}});
        linked["relations"]=json::array({json{{"subject","a"},{"predicate","left_of"},{"object","b"}}});
        model->zeroGradients(); request.annotation=&linked;
        const auto relational=model->trainStep(request);
        TASSERT_TRUE(relational && std::isfinite(relational->loss) && std::isfinite(relational->grad_norm));
        TASSERT_TRUE(relational->metrics.at("relation_count")==1);
        model->zeroGradients(); model->modelConfig["bbox_loss_weight"]=0;
        const auto disabled=model->trainStep(request);
        TASSERT_TRUE(disabled);
        TASSERT_NEAR(disabled->loss,historical->loss,1e-6f);
        std::cout << "PASS " << type << " native bbox training and disabled fallback\n";
    }
    // PatchDiscriminator currently fails its historical CPU forward before any
    // bbox use (first Conv2d); verify its spatial contract separately.
    json patch_cfg=ModelArchitectures::defaultConfig("patch_discriminator");
    patch_cfg.update(json{{"image_w",32},{"image_h",16},{"base_channels",4}});
    auto patch=ModelArchitectures::create("patch_discriminator",patch_cfg);
    bool patch_output=false;
    const auto patch_weights=BBoxSupervision::forModel(&annotation,patch->modelConfig,
        patch->modelConfig["input_dim"],patch->modelConfig["output_dim"],&patch_output);
    TASSERT_TRUE(patch_output && patch_weights.values.size()==patch->modelConfig["output_dim"].get<size_t>());
    const json decoder_cfg={{"type","hf_vae_decoder"},{"image_w",16},{"image_h",8},{"image_c",3}};
    bool decoder_output=false;
    const auto decoder_weights=BBoxSupervision::forModel(&annotation,decoder_cfg,24,384,&decoder_output);
    TASSERT_TRUE(decoder_output && decoder_weights.values.size()==384);
    bool output=false;
    const json token_config={{"type","vit"},{"d_model",8},{"bbox_input_w",2},{"bbox_input_h",2}};
    const auto token_weights=BBoxSupervision::forModel(&annotation,token_config,40,2,&output);
    TASSERT_TRUE(!output && token_weights.values.size()==40);
    for(int i=0;i<8;++i) TASSERT_NEAR(token_weights.values[i],1.f,0.f);
    std::vector<MedicalDetectorModel::Box> boxes;
    std::string error;
    TASSERT_TRUE(MedicalDetectorModel::parseAnnotation(annotation.dump(),32,32,1,&boxes,&error));
    TASSERT_TRUE(boxes.size()==1 && boxes[0].class_id==0 && boxes[0].x2==0.25f);
    return 0;
}
