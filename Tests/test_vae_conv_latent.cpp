#include "test_utils.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Models/Vision/VAEConvModel.hpp"
#include "Serialization/Serialization.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <vector>

static double squaredError(const std::vector<float>& output, const std::vector<float>& target) {
    double sum = 0;
    for (size_t i=0; i<target.size(); ++i) {
        const double d = output[i]-target[i]; sum += 0.5*d*d;
    }
    return sum;
}

static int phaseRoundtrip() {
    VAEConvModel::Config cfg;
    cfg.image_w=8;cfg.image_h=4;cfg.image_c=3;
    cfg.latent_w=4;cfg.latent_h=2;cfg.latent_c=12;
    cfg.base_channels=4;cfg.latent_context_channels=12;
    cfg.latent_detail_path=true;cfg.resnet=false;
    cfg.enc_norm=cfg.dec_norm="none";
    VAEConvModel model;model.buildFromConfig(cfg);model.allocateParams();
    for (auto& l : model.getMutableLayers())
        if (l.getWeights()) std::fill_n(l.getWeights(),l.getWeightsSize(),0.f);
    for (const auto& name : {"vae_conv/enc/detail/project","vae_conv/enc/mu","vae_conv/dec/detail/project"}) {
        auto* l = model.getLayerByName(name);
        for (int c=0;c<12;++c) l->getWeights()[c*l->in_channels+c]=1.f;
    }
    auto* fuse=model.getLayerByName("vae_conv/enc/detail/fuse");
    for (int c=0;c<12;++c) fuse->getWeights()[c*fuse->in_channels+cfg.base_channels+c]=1.f;
    // Every RGB channel and every subpixel phase has a distinct value.
    std::vector<float> input(8*4*3);
    for (size_t i=0;i<input.size();++i) input[i]=static_cast<float>(i)*0.01f-0.48f;
    const auto output=model.forwardPass(input,false);
    TASSERT_TRUE(output.size()==input.size());
    for (size_t i=0;i<input.size();++i) TASSERT_NEAR(output[i],std::tanh(input[i]),1e-6f);
    const auto& packed=model.getTensor("vae_conv/enc/detail/pack/out");
    for (int c=0;c<3;++c) for (int py=0;py<2;++py) for (int px=0;px<2;++px)
        for (int y=0;y<2;++y) for (int x=0;x<4;++x) {
            const size_t phase_index=((c*2+py)*2+px)*8+y*4+x;
            const size_t image_index=((y*2+py)*8+x*2+px)*3+c;
            TASSERT_NEAR(packed[phase_index],input[image_index],1e-6f);
        }
    return 0;
}

int main() {
    TASSERT_TRUE(phaseRoundtrip()==0);
    json cfg={{"image_w",16},{"image_h",8},{"image_c",3},
        {"latent_w",8},{"latent_h",4},{"latent_c",6},{"base_channels",4},
        {"latent_spatial_context",true},{"latent_context_channels",8},{"latent_detail_path",true},
        {"resnet",false},{"attention",false},{"enc_norm","none"},{"dec_norm","none"},
        {"stochastic_latent",false},{"decoder_upsample","nearest_conv"}};
    auto model=ModelArchitectures::create("vae_conv",cfg);
    model->allocateParams();model->initializeWeights("xavier",631u);
    std::vector<float> input(16*8*3),target(input.size());
    for (size_t i=0;i<input.size();++i) input[i]=target[i]=0.4f*std::sin(static_cast<float>(i)*0.71f);
    auto output=model->forwardPass(input,true);
    TASSERT_TRUE(output.size()==input.size()+2*8*4*6);
    std::vector<float> gradient(output.size(),0.f);
    for (size_t i=0;i<input.size();++i) gradient[i]=output[i]-target[i];
    model->backwardPass(gradient);
    // Reconstruction gradients reach encoder detail, local/context branches,
    // and decoder detail. Check numerical derivatives through packing/dilation.
    for (const char* name : {"vae_conv/enc/detail/project",
            "vae_conv/enc/latent_context/scale1", "vae_conv/enc/latent_context/scale2",
            "vae_conv/enc/latent_context/scale4", "vae_conv/dec/latent_context/scale2",
            "vae_conv/dec/detail/project"}) {
        auto* layer=model->getLayerByName(name);
        TASSERT_TRUE(layer && layer->grad_weights.size()==layer->getWeightsSize());
        size_t index=0;
        for (size_t i=0;i<layer->grad_weights.size();++i) {
            TASSERT_TRUE(std::isfinite(layer->grad_weights[i]));
            if (std::fabs(layer->grad_weights[i])>std::fabs(layer->grad_weights[index])) index=i;
        }
        const float analytic=layer->grad_weights[index];
        TASSERT_TRUE(std::fabs(analytic)>1e-7f);
        float* weights=layer->getWeights();const float saved=weights[index],epsilon=0.001f;
        weights[index]=saved+epsilon;
        const double plus=squaredError(model->forwardPass(input,false),target);
        weights[index]=saved-epsilon;
        const double minus=squaredError(model->forwardPass(input,false),target);
        weights[index]=saved;
        const double numeric=(plus-minus)/(2*epsilon);
        TASSERT_TRUE(std::fabs(analytic-numeric)<0.003+0.02*std::fabs(numeric));
    }

    // Decoder-only inference uses identical layer names, geometry and weights.
    const auto expected=model->forwardPass(input,false);
    const auto latent=model->getTensor("vae_conv/z");
    auto decoder=ModelArchitectures::create("vae_conv_decode",cfg);
    decoder->allocateParams();
    for (auto& layer : decoder->getMutableLayers()) {
        const auto* source=model->getLayerByName(layer.name);
        if (!layer.getWeightsSize()) continue;
        TASSERT_TRUE(source && source->getWeightsSize()==layer.getWeightsSize());
        std::copy_n(source->getWeights(),source->getWeightsSize(),layer.getWeights());
    }
    const auto decoded=decoder->forwardPass(latent,false);
    TASSERT_TRUE(decoded.size()==expected.size());
    for (size_t i=0;i<expected.size();++i) TASSERT_NEAR(decoded[i],expected[i],1e-6f);
    std::vector<float> shifted=latent;shifted[0]+=0.1f;
    const auto changed=decoder->forwardPass(shifted,false);
    double effect=0;for (size_t i=0;i<changed.size();++i) effect+=std::fabs(changed[i]-decoded[i]);
    TASSERT_TRUE(effect>1e-6);

    // Native training must reduce reconstruction error on a fine-pattern fixture.
    model->modelConfig["recon_loss"]="mse";model->modelConfig["kl_beta"]=0.f;
    Optimizer optimizer;optimizer.type=OptimizerType::ADAM;optimizer.initial_lr=0.003f;
    optimizer.decay_strategy=LRDecayStrategy::NONE;
    Model::TrainStepRequest request;request.float_inputs["__input__"]=&input;
    request.target=&target;request.optimizer=&optimizer;request.learning_rate=0.003f;
    const double initial=squaredError(model->forwardPass(input,false),target);
    for (int i=0;i<40;++i) {
        const auto result=model->trainStep(request);
        TASSERT_TRUE(result && std::isfinite(result->loss));
    }
    const auto trained=model->forwardPass(input,false);
    const double final=squaredError(trained,target);
    std::cout << "LATENT_FINE_PATTERN initial=" << initial << " final=" << final << '\n';
    TASSERT_TRUE(final<0.6*initial);

    // All new graph fields/parameters survive strict checkpoint reload.
    using namespace Mimir::Serialization;
    const auto path=std::filesystem::temp_directory_path()/("mimir_latent_context_"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    SaveOptions save;save.format=CheckpointFormat::RawFolder;save.save_encoder=save.save_tokenizer=false;
    std::string error;TASSERT_TRUE(save_checkpoint(*model,path.string(),save,&error));
    auto restored=ModelArchitectures::create("vae_conv",model->modelConfig);restored->allocateParams();
    LoadOptions load;load.format=CheckpointFormat::RawFolder;load.load_encoder=load.load_tokenizer=false;
    TASSERT_TRUE(load_checkpoint(*restored,path.string(),load,&error));
    const auto resumed=restored->forwardPass(input,false);
    TASSERT_TRUE(resumed.size()==trained.size());
    for (size_t i=0;i<resumed.size();++i) TASSERT_NEAR(resumed[i],trained[i],1e-6f);
    std::filesystem::remove_all(path);
    return 0;
}
