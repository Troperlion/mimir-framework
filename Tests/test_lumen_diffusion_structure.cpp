#include "test_utils.hpp"
#include "viz/LumenOutputRole.hpp"

#include "Models/Diffusion/LumenLatentDiffusionModel.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Models/Vision/VAEConvModel.hpp"
#include "Serialization/Serialization.hpp"
#include "runtimes/ops_sequence_constant.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/losses/LossDispatch.hpp"
#include "include/json.hpp"

#include <cmath>
#include <limits>
#include <filesystem>
#include <string>
#include <unordered_set>

using json = nlohmann::json;

class TestableLumenModel : public LumenLatentDiffusionModel {
public:
    int denoiser_forwards = 0;
    int inference_forwards = 0;
    const std::vector<float>& forwardPassNamedView(
        const std::unordered_map<std::string,std::vector<float>>& floats,
        const std::unordered_map<std::string,std::vector<int>>& ints,
        bool training = true) override {
        ++denoiser_forwards;
        if (!training) ++inference_forwards;
        return LumenLatentDiffusionModel::forwardPassNamedView(floats, ints, training);
    }
    void addTestTips(const std::vector<float>& generated_image,
                     const std::vector<unsigned char>& original_image) {
        addDiffusionVizTips(generated_image, original_image);
    }

    void addTestComparisonTips(const std::vector<float>& oracle_image,
                               const std::vector<float>& noisy_baseline_image,
                               const std::vector<float>& predicted_image,
                               int timestep) {
        addDiffusionComparisonVizTips(
            oracle_image, noisy_baseline_image, predicted_image, timestep);
    }

    std::vector<float> encodeTestImage(const std::vector<float>& image) {
        return encodeImage(image);
    }

    std::vector<float> decodeTestImage(const std::vector<float>& rgb_chw) {
        return decodeImage(rgb_chw);
    }
};

int main() {
    json cfg = ModelArchitectures::defaultConfig("lumen_diffusion");
    TASSERT_TRUE(cfg["latent_w"].get<int>() == 64);
    TASSERT_TRUE(cfg["latent_h"].get<int>() == 64);
    TASSERT_TRUE(cfg["latent_c"].get<int>() == 4);
    TASSERT_TRUE(cfg["vae_base_channels"].get<int>() == 8);
    TASSERT_TRUE(cfg["vae_resnet"].get<bool>());
    TASSERT_TRUE(!cfg["vae_attention"].get<bool>());
    TASSERT_TRUE(!cfg["vae_use_skip_connections"].get<bool>());
    TASSERT_TRUE(!cfg["vae_use_encoder_prior"].get<bool>());
    TASSERT_TRUE(cfg["vae_resnet_max_tokens"].get<int>() == 4096);
    TASSERT_TRUE(cfg["vae_decoder_upsample"].get<std::string>() == "nearest_conv");
    TASSERT_TRUE(cfg["patch_size"].get<int>() == 4);
    cfg["image_w"] = 16;
    cfg["image_h"] = 16;
    cfg["latent_w"] = 16;
    cfg["latent_h"] = 16;
    cfg["latent_c"] = 3;
    cfg["patch_size"] = 4;
    cfg["hidden_size"] = 8;
    cfg["depth"] = 1;
    cfg["mlp_ratio"] = 2.0f;
    cfg["vocab_size"] = 32;
    cfg["text_seq_len"] = 4;
    cfg["text_layers"] = 1;
    cfg["num_heads"] = 2;
    cfg["kl_beta"] = 0.5f;
    cfg["kl_warmup_steps"] = 10;
    cfg["vae_decoder_upsample"] = "nearest_conv";

    auto model = ModelArchitectures::create("lumen_diffusion", cfg);
    TASSERT_TRUE(static_cast<bool>(model));
    auto* lumen = dynamic_cast<LumenLatentDiffusionModel*>(model.get());
    TASSERT_TRUE(lumen != nullptr);
    TASSERT_TRUE(lumen->getConfig().image_w * lumen->getConfig().image_h *
                     lumen->getConfig().image_c == 768);
    TASSERT_TRUE(model->modelConfig["type"].get<std::string>() == "lumen_diffusion");
    TASSERT_TRUE(model->modelConfig.contains("patch_size"));
    TASSERT_TRUE(model->modelConfig["patch_size"].is_number_integer());
    TASSERT_TRUE(model->modelConfig["patch_size"].get<int>() == 4);
    TASSERT_TRUE(model->modelConfig.contains("architecture_version"));
    TASSERT_TRUE(model->modelConfig["architecture_version"].is_number_integer());
    TASSERT_TRUE(model->modelConfig["architecture_version"].get<int>() == 5);
    TASSERT_TRUE(model->modelConfig.contains("kl_beta"));
    TASSERT_TRUE(model->modelConfig["kl_beta"].is_number());
    TASSERT_TRUE(model->modelConfig["kl_beta"].get<float>() == 0.0f);
    TASSERT_TRUE(model->modelConfig.contains("kl_warmup_steps"));
    TASSERT_TRUE(model->modelConfig["kl_warmup_steps"].is_number_integer());
    TASSERT_TRUE(model->modelConfig["kl_warmup_steps"].get<int>() == 0);
    TASSERT_TRUE(model->modelConfig["vae_decoder_upsample"].get<std::string>() ==
                 "nearest_conv");
    TASSERT_TRUE(model->getLayerByName("lumen/text/token_embedding") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/text/block1/norm1") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/text/block1/mlp_fc1") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/text/block1/mlp_gelu") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/text/block1/mlp_fc2") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/text/block1/add_mlp") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/text/final_norm") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/time/input") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/dit/patch_embed") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/dit/block1/self_attention") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/dit/block1/cross_attention") != nullptr);
    TASSERT_TRUE(model->getLayerByName("lumen/dit/unpatchify") != nullptr);
    TASSERT_TRUE(lumen->InitVizTips());

    TestableLumenModel tips_model;
    LumenLatentDiffusionModel::Config tips_cfg;
    tips_cfg.image_w = 16;
    tips_cfg.image_h = 16;
    tips_cfg.latent_w = 16;
    tips_cfg.latent_h = 16;
    tips_cfg.latent_c = 3;
    tips_cfg.patch_size = 4;
    tips_cfg.hidden_size = 8;
    tips_cfg.depth = 1;
    tips_cfg.mlp_ratio = 2.0f;
    tips_cfg.vocab_size = 32;
    tips_cfg.text_seq_len = 4;
    tips_cfg.text_layers = 1;
    tips_cfg.num_heads = 2;
    tips_model.buildFromConfig(tips_cfg);
    tips_model.setVizTapsEnabled(true);
    tips_model.setVizTapsLimits(16, 8);
    std::vector<float> generated_image(16 * 16 * 3, -1.0f);
    std::vector<unsigned char> original_image(16 * 16 * 3, 0);
    generated_image[1] = 0.0f;
    generated_image[2] = 1.0f;
    original_image[0] = 255;
    original_image[1] = 128;
    tips_model.addTestTips(generated_image, original_image);
    std::unordered_set<std::string> tip_labels;
    for (const auto& frame : tips_model.consumeVizTaps()) {
        tip_labels.insert(frame.label);
        TASSERT_TRUE(frame.w == 8);
        TASSERT_TRUE(frame.h == 8);
        TASSERT_TRUE(frame.channels == 3);
        TASSERT_TRUE(frame.pixels.size() ==
                     static_cast<size_t>(8 * 8 * 3));
        TASSERT_TRUE(frame.tensor_values.size() == frame.pixels.size());
        TASSERT_TRUE(frame.tensor_indices.size() == frame.pixels.size());
        TASSERT_TRUE(frame.tensor_sample_channels == 3);
        TASSERT_TRUE(frame.tensor_indices[3] == 6);
        if (frame.label == "diffusion_out") {
            TASSERT_NEAR(frame.tensor_values[0], -1.f, 1e-7f);
            TASSERT_NEAR(frame.tensor_values[1], 0.f, 1e-7f);
            TASSERT_TRUE(frame.pixels[0] == 0);
            TASSERT_TRUE(frame.pixels[1] == 128);
            TASSERT_TRUE(frame.pixels[2] == 255);
        } else if (frame.label == "resdiff_abs") {
            TASSERT_TRUE(frame.pixels[0] == 255);
            TASSERT_TRUE(frame.pixels[1] == 0);
            TASSERT_TRUE(frame.pixels[2] == 255);
        } else if (frame.label == "resdiff_norm") {
            TASSERT_TRUE(frame.pixels[0] == 0);
            TASSERT_TRUE(frame.pixels[1] == 128);
            TASSERT_TRUE(frame.pixels[2] == 255);
        } else if (frame.label == "resdiff_max") {
            TASSERT_TRUE(frame.pixels[0] == 0);
            TASSERT_TRUE(frame.pixels[1] == 0);
            TASSERT_TRUE(frame.pixels[2] == 255);
        } else if (frame.label == "resdiff_min") {
            TASSERT_TRUE(frame.pixels[0] == 255);
            TASSERT_TRUE(frame.pixels[1] == 0);
            TASSERT_TRUE(frame.pixels[2] == 0);
        }
    }
    TASSERT_TRUE(tip_labels.count("resdiff_abs") == 1);
    TASSERT_TRUE(tip_labels.count("resdiff_norm") == 1);
    TASSERT_TRUE(tip_labels.count("resdiff_max") == 1);
    TASSERT_TRUE(tip_labels.count("resdiff_min") == 1);
    TASSERT_TRUE(tip_labels.count("diffusion_out") == 1);
    TASSERT_TRUE(tip_labels.count("diffusion/reference") == 1);

    const std::vector<float> oracle_preview(16 * 16 * 3, -1.0f);
    const std::vector<float> baseline_preview(16 * 16 * 3, 0.0f);
    const std::vector<float> predicted_preview(16 * 16 * 3, 1.0f);
    tips_model.addTestComparisonTips(
        oracle_preview, baseline_preview, predicted_preview, 50);
    const auto comparison_frames = tips_model.consumeVizTaps();
    std::vector<Model::VizFrame> comparison_triptych;
    for (const auto& frame : comparison_frames) {
        if (frame.label.find("diffusion/compare/") == 0) {
            comparison_triptych.push_back(frame);
        }
    }
    TASSERT_TRUE(comparison_triptych.size() == 3);
    TASSERT_TRUE(comparison_triptych[0].label ==
                 "diffusion/compare/A_oracle_decode_z0 | timestep=50");
    TASSERT_TRUE(comparison_triptych[1].label ==
                 "diffusion/compare/B_noisy_baseline | timestep=50");
    TASSERT_TRUE(comparison_triptych[2].label ==
                 "diffusion/compare/C_model_decode_z0_pred | timestep=50");
    TASSERT_TRUE(comparison_triptych[0].pixels[0] == 0);
    TASSERT_TRUE(comparison_triptych[1].pixels[0] == 128);
    TASSERT_TRUE(comparison_triptych[2].pixels[0] == 255);

    TestableLumenModel direct_model;
    LumenLatentDiffusionModel::Config direct_cfg = tips_cfg;
    direct_cfg.image_w = 8;
    direct_cfg.image_h = 4;
    direct_cfg.image_c = 3;
    direct_cfg.latent_w = 8;
    direct_cfg.latent_h = 4;
    direct_cfg.latent_c = 3;
    direct_model.modelConfig["external_graphs"] = {{"stale", true}};
    direct_model.modelConfig["graph_bindings"] = {{"stale", true}};
    direct_model.buildFromConfig(direct_cfg);
    TASSERT_TRUE(!direct_model.modelConfig.contains("external_graphs"));
    TASSERT_TRUE(!direct_model.modelConfig["graph_bindings"].contains("stale"));
    TASSERT_TRUE(direct_model.modelConfig["output_tensor"] == "z");
    TASSERT_TRUE(direct_model.modelConfig["architecture"].get<std::string>() ==
                 "dit_latent_vae_conv");
    TASSERT_TRUE(direct_model.modelConfig["input_dim"].get<int>() == 8 * 4 * 3);
    TASSERT_TRUE(direct_model.modelConfig["output_dim"].get<int>() == 8 * 4 * 3);
    std::vector<float> direct_image(8 * 4 * 3);
    for (size_t index = 0; index < direct_image.size(); ++index) {
        direct_image[index] = static_cast<float>(index) / direct_image.size();
    }
    const auto direct_chw = direct_model.encodeTestImage(direct_image);
    const auto direct_round_trip = direct_model.decodeTestImage(direct_chw);
    TASSERT_TRUE(direct_round_trip == direct_image);

    direct_model.allocateParams();
    direct_model.initializeWeights("xavier", 123U);
    const auto position_parameters=direct_model.getLayerByName("lumen/text/position")->getWeightsSize();
    Model* polymorphic=&direct_model;
    for(int length:{2,13,1,7,4,0}) {
        const auto prediction=polymorphic->forwardPassNamed({{"__input__",direct_chw},{"timestep",std::vector<float>(8,0.f)}},
            {{"text_ids",std::vector<int>(size_t(length),1)}},false);
        TASSERT_TRUE(prediction.size()==direct_chw.size());
        TASSERT_TRUE(direct_model.getTensor("lumen/text/context").size()==size_t(std::max(1,length))*8);
        TASSERT_TRUE(direct_model.getTensor("lumen/text/position_values").size()==size_t(std::max(1,length))*8);
        TASSERT_TRUE(direct_model.getLayerByName("lumen/text/position")->getWeightsSize()==position_parameters);
        TASSERT_TRUE(direct_model.modelConfig["text_active_seq_len"]==std::max(1,length));
    }
    auto* positions=direct_model.getLayerByName("lumen/text/position");
    positions->seq_len=11;positions->trainable_parameter=true;
    std::vector<float> interpolated;TASSERT_TRUE(SequenceConstant::forward(*positions,interpolated));
    std::vector<float> adjoint(interpolated.size(),1.f);
    positions->grad_weights.assign(position_parameters,0.f);
    TASSERT_TRUE(SequenceConstant::backward(*positions,adjoint));
    const float analytical=positions->grad_weights[9], original=positions->getWeights()[9];
    auto total=[&](){std::vector<float> v;SequenceConstant::forward(*positions,v);double sum=0;for(float x:v)sum+=x;return sum;};
    positions->getWeights()[9]=original+.001f;const double plus=total();
    positions->getWeights()[9]=original-.001f;const double minus=total();positions->getWeights()[9]=original;
    TASSERT_NEAR(analytical,(plus-minus)/.002,1e-3);
    Optimizer train_optimizer;
    train_optimizer.type = OptimizerType::SGD;
    train_optimizer.decay_strategy = LRDecayStrategy::NONE;
    std::vector<unsigned char> train_image(8 * 4 * 3, 127);
    const auto loss_dispatches=[] {
        size_t count=0;
        for(const auto& entry:RuntimeRouter::instance().reconstructionLossDispatchCounts())count+=entry.second;
        return count;
    };
    const auto optimizer_dispatches=[] {
        size_t count=0;
        for(const auto& entry:RuntimeRouter::instance().optimizerDispatchCounts())count+=entry.second;
        return count;
    };
    const auto losses_before_training=loss_dispatches();
    const auto optimizers_before_training=optimizer_dispatches();
    const auto train_stats = direct_model.trainDiffusionStep(
        train_image, "test", 123U, train_optimizer, 1e-4f);
    TASSERT_TRUE(std::isfinite(train_stats.loss));
    TASSERT_TRUE(std::isfinite(train_stats.grad_norm));
    TASSERT_TRUE(train_stats.kl_beta_effective == 0.0f);
    TASSERT_TRUE(loss_dispatches()==losses_before_training+1);
    TASSERT_TRUE(optimizer_dispatches()>optimizers_before_training);
    TASSERT_TRUE(train_optimizer.step==1);
    TASSERT_NEAR(train_stats.mse_unweighted,train_stats.loss,1e-7f);
    const auto losses_before_validation=loss_dispatches();
    const auto optimizers_before_validation=optimizer_dispatches();
    const auto validation_stats=direct_model.validateDiffusionStep(train_image,"test",123U);
    TASSERT_TRUE(loss_dispatches()==losses_before_validation+3);
    TASSERT_TRUE(optimizer_dispatches()==optimizers_before_validation);
    TASSERT_TRUE(train_optimizer.step==1);
    TASSERT_TRUE(std::isfinite(validation_stats.loss));
    TASSERT_NEAR(validation_stats.mse_unweighted,validation_stats.loss,1e-7f);
    TASSERT_TRUE(validation_stats.reconstruction_mae>=0 && validation_stats.reconstruction_mae<=1);
    TASSERT_TRUE(validation_stats.reconstruction_mse>=0 && validation_stats.reconstruction_mse<=1);
    TASSERT_TRUE(validation_stats.reconstruction_mae*validation_stats.reconstruction_mae<=validation_stats.reconstruction_mse+1e-6f);

    // Weighted epsilon prediction keeps the same objective and supports gradient
    // accumulation without advancing the optimizer or updating its parameters.
    std::vector<float> timestep_values(8,0.f),noise_target(direct_chw.size()),spatial_weights(direct_chw.size(),2.f);
    for(size_t i=0;i<noise_target.size();++i){noise_target[i]=std::sin(float(i)*.17f);if(i%7==0)spatial_weights[i]=0.f;}
    SpatialRelation relation;relation.width=8;relation.height=4;relation.channels=3;relation.chw=true;
    relation.subject={0,0,.6,1};relation.object={.4,0,1,1};relation.weight=.4f;relation.predicate="left_of";
    std::vector<SpatialRelation> relations{relation};
    std::vector<int> training_ids{1,2};
    Model::TrainStepRequest weighted_request;
    weighted_request.float_inputs={{"__input__",&direct_chw},{"timestep",&timestep_values}};
    weighted_request.int_inputs={{"text_ids",&training_ids}};
    weighted_request.target=&noise_target;weighted_request.loss_weights=&spatial_weights;weighted_request.relations=&relations;
    weighted_request.optimizer=&train_optimizer;weighted_request.learning_rate=1e-4f;
    weighted_request.mode=Model::TrainStepMode::Accumulate;weighted_request.grad_scale=.5f;
    const auto losses_before_weighted=loss_dispatches();
    const auto weighted=direct_model.trainStep(weighted_request);
    TASSERT_TRUE(weighted.has_value());
    TASSERT_TRUE(loss_dispatches()==losses_before_weighted+2);
    TASSERT_TRUE(optimizer_dispatches()==optimizers_before_validation);
    TASSERT_TRUE(train_optimizer.step==1);
    RuntimeLossGrad::PixelLossOptions expected_options;expected_options.weights=&spatial_weights;expected_options.relations=&relations;
    const auto& predicted_noise=direct_model.getTensor("x");
    const auto expected_weighted=RuntimeLossGrad::Host::pixel_loss_and_grad(predicted_noise,noise_target,"mse",expected_options);
    const auto expected_unweighted=RuntimeLossGrad::Host::pixel_loss_and_grad(predicted_noise,noise_target,"mse",{});
    TASSERT_NEAR(weighted->loss,expected_weighted.loss,1e-5);
    TASSERT_NEAR(weighted->metrics.at("relation_loss"),expected_weighted.relation_loss,1e-5);
    TASSERT_NEAR(weighted->metrics.at("mse_unweighted"),expected_unweighted.loss,1e-5);
    TASSERT_TRUE(weighted->metrics.at("relation_count")==1);

    // Packed projection keeps distinct parameters and gradients for every block.
    TestableLumenModel packed_model;
    auto packed_cfg = direct_cfg;
    packed_cfg.depth = 3;
    packed_model.buildFromConfig(packed_cfg);
    packed_model.allocateParams();
    packed_model.initializeWeights("xavier", 123U);
    auto* packed_projection = packed_model.getLayerByName("lumen/time/block_projection");
    TASSERT_TRUE(packed_projection != nullptr);
    TASSERT_TRUE(packed_projection->getWeightsSize() == size_t(3 * 8 * 9));
    TASSERT_TRUE(packed_model.getLayerByName("lumen/dit/block1/time_projection") == nullptr);
    Optimizer packed_optimizer;
    packed_optimizer.type = OptimizerType::SGD;
    packed_optimizer.decay_strategy = LRDecayStrategy::NONE;
    packed_model.setVizTapsEnabled(true);
    packed_model.setVizTapsLimits(64, 4);
    packed_model.modelConfig["preview_every_steps"] = 1; // ignored legacy setting
    TASSERT_TRUE(Mimir::Viz::isLumenIntermediate("lumen_diffusion/blocks/lumen/time/output/Linear"));
    TASSERT_TRUE(!Mimir::Viz::isLumenModelOutput("lumen_diffusion/blocks/lumen/time/output/Linear"));
    TASSERT_TRUE(!Mimir::Viz::isLumenModelOutput("lumen_diffusion/blocks/lumen/dit/patch_output/Linear"));
    TASSERT_TRUE(Mimir::Viz::isLumenModelOutput("lumen_diffusion/blocks/lumen/output/z/Identity"));
    const auto packed_stats = packed_model.trainDiffusionStep(train_image, "test", 123U, packed_optimizer, 1e-4f);
    TASSERT_TRUE(std::isfinite(packed_stats.loss));
    TASSERT_TRUE(packed_model.denoiser_forwards == 1);
    (void)packed_model.consumeVizTaps(); // release the capture root before the independent run
    TestableLumenModel headless_model;
    headless_model.buildFromConfig(packed_cfg);
    headless_model.allocateParams();
    headless_model.initializeWeights("xavier", 123U);
    Optimizer headless_optimizer;
    headless_optimizer.type = OptimizerType::SGD;
    headless_optimizer.decay_strategy = LRDecayStrategy::NONE;
    const auto headless_stats = headless_model.trainDiffusionStep(train_image, "test", 123U, headless_optimizer, 1e-4f);
    TASSERT_NEAR(packed_stats.loss, headless_stats.loss, 1e-7f);
    const auto& captured_layers = packed_model.getLayers();
    const auto& headless_layers = headless_model.getLayers();
    for (size_t layer = 0; layer < captured_layers.size(); ++layer)
        for (size_t i = 0; i < captured_layers[layer].getWeightsSize(); ++i)
            TASSERT_NEAR(captured_layers[layer].getWeights()[i], headless_layers[layer].getWeights()[i], 1e-7f);
    const auto& all_times = packed_model.getTensor("lumen/time/blocks_packed");
    for (int block = 0; block < 3; ++block) {
        const auto& block_time = packed_model.getTensor("lumen/time/blocks_" + std::to_string(block));
        TASSERT_TRUE(block_time.size() == 8);
        double gradient_energy = 0;
        for (int i = 0; i < 8; ++i) {
            TASSERT_NEAR(block_time[i], all_times[block * 8 + i], 1e-7f);
            for (int j = 0; j < 8; ++j) {
                const float gradient = packed_projection->grad_weights[(block * 8 + i) * 8 + j];
                TASSERT_TRUE(std::isfinite(gradient));
                gradient_energy += gradient * gradient;
            }
        }
        TASSERT_TRUE(gradient_energy > 0);
    }
    int transformer_tips = 0;
    for (const auto& frame : packed_model.consumeVizTaps()) {
        TASSERT_TRUE(frame.label.find("diffusion/compare/") != 0);
        TASSERT_TRUE(frame.label != "diffusion_out");
        if (frame.label.find("/blocks/") != std::string::npos) {
            ++transformer_tips;
            TASSERT_TRUE(frame.label.find("/mlp_fc1/") == std::string::npos);
            TASSERT_TRUE(frame.label.find("/norm1/") == std::string::npos);
        }
    }
    TASSERT_TRUE(packed_model.hasTensor("z"));
    TASSERT_TRUE(packed_model.getTensor("z").size() == train_image.size());
    const auto& noisy_values = packed_model.getTensor("__input__");
    const auto& epsilon_values = packed_model.getTensor("x");
    const auto& latent_values = packed_model.getTensor("lumen/reconstruction/clean_latent");
    const float a = packed_model.getTensor("diffusion_signal_scale")[0];
    const float b = packed_model.getTensor("diffusion_noise_scale")[0];
    for (size_t i = 0; i < latent_values.size(); ++i)
        TASSERT_NEAR(latent_values[i], (noisy_values[i] - b * epsilon_values[i]) / a, 1e-5f);
    bool z_tip = false;
    for (const auto& frame : packed_model.consumeVizTaps()) {
        if (!Mimir::Viz::isLumenModelOutput(frame.label)) continue;
        z_tip = true;
        TASSERT_TRUE(frame.w == 4 && frame.h == 2);
        const auto& image = packed_model.getTensor("z");
        for (size_t i = 0; i < frame.tensor_values.size(); ++i)
            TASSERT_NEAR(frame.tensor_values[i], image[frame.tensor_indices[i]], 1e-7f);
    }
    TASSERT_TRUE(z_tip);
    TASSERT_TRUE(transformer_tips == 7 + 3 * packed_cfg.depth);
    packed_model.trainDiffusionStep(train_image, "test", 124U, packed_optimizer, 1e-4f);
    TASSERT_TRUE(packed_model.denoiser_forwards == 2);
    TASSERT_TRUE(packed_model.inference_forwards == 0);
    for (const auto& frame : packed_model.consumeVizTaps()) {
        TASSERT_TRUE(frame.label.find("diffusion/compare/") != 0);
        TASSERT_TRUE(frame.label != "diffusion_out");
    }
    const auto validation_step = packed_optimizer.step;
    const auto validation_preview = packed_model.validateDiffusionStep(train_image, "test", 125U);
    TASSERT_TRUE(packed_model.denoiser_forwards == 3);
    TASSERT_TRUE(packed_model.inference_forwards == 1);
    TASSERT_TRUE(packed_optimizer.step == validation_step);
    bool decoded_validation = false;
    for (const auto& frame : packed_model.consumeVizTaps()) {
        if (frame.label.find("diffusion/compare/C_") == 0) {
            decoded_validation = true;
            TASSERT_TRUE(frame.w == 4 && frame.h == 2);
            TASSERT_TRUE(frame.tensor_values.size() == size_t(4 * 2 * 3));
            TASSERT_TRUE(frame.label.find("timestep=" + std::to_string(validation_preview.timestep)) != std::string::npos);
        }
        if (frame.label.find("/lumen/dit/block2/cross_attention/") != std::string::npos) {
            const auto& raw = packed_model.getTensor("lumen/dit/block2/cross_out");
            for (size_t i = 0; i < frame.tensor_values.size(); ++i) {
                TASSERT_TRUE(frame.tensor_indices[i] < raw.size());
                TASSERT_NEAR(frame.tensor_values[i], raw[frame.tensor_indices[i]], 1e-7f);
            }
        }
    }
    TASSERT_TRUE(decoded_validation);
    packed_model.clearVizTaps();
    packed_model.trainDiffusionStep(train_image, "test", 126U, packed_optimizer, 1e-4f);
    TASSERT_TRUE(packed_model.denoiser_forwards == 4);
    TASSERT_TRUE(packed_model.inference_forwards == 1);
    for (const auto& frame : packed_model.consumeVizTaps()) {
        TASSERT_TRUE(frame.label.find("diffusion/compare/") != 0);
        TASSERT_TRUE(frame.label != "diffusion_out");
    }

    std::string long_prompt;for(int i=0;i<19;++i)long_prompt+="token ";
    const auto generated_long=direct_model.generate(long_prompt,17,2,1.f);
    TASSERT_TRUE(generated_long.pixels.size()==train_image.size());
    TASSERT_TRUE(direct_model.modelConfig["text_active_seq_len"]==direct_model.getTokenizer().tokenize(long_prompt).size());
    (void)direct_model.generate("",17,2,1.f);
    TASSERT_TRUE(direct_model.modelConfig["text_active_seq_len"]==std::max<size_t>(1,direct_model.getTokenizer().tokenize("").size()));
    // A single DDIM step starts at the noisiest endpoint, and seeded sampling is repeatable.
    const auto one_step = direct_model.generate("test", 17, 1, 1.f);
    TASSERT_NEAR(direct_model.getTensor("lumen/time/embedding")[0],
        std::sin(float(direct_cfg.diffusion_steps - 1)), 1e-6f);
    const auto one_step_again = direct_model.generate("test", 17, 1, 1.f);
    TASSERT_TRUE(one_step.pixels == one_step_again.pixels);
    TASSERT_TRUE(one_step.pixels.size() == train_image.size());
    bool bad_sampling = false;
    try { direct_model.generate("test", 17, 0, 1.f); }
    catch (const std::invalid_argument&) { bad_sampling = true; }
    TASSERT_TRUE(bad_sampling);
    // Annotation enrichment must never leak into classifier-free training inputs.
    const json cfg_annotation = {{"entities", json::array({{{"label", "person"}, {"bbox", {0., 0., 1., 1.}}}})}};
    const auto dropped = direct_model.trainDiffusionStep(
        train_image, long_prompt, 123U, train_optimizer, 1e-4f, &cfg_annotation, 1.f, true);
    TASSERT_TRUE(std::isfinite(dropped.loss));
    TASSERT_TRUE(direct_model.modelConfig["text_active_seq_len"] ==
        std::max<size_t>(1, direct_model.getTokenizer().tokenize("").size()));
    const auto dynamic_checkpoint=std::filesystem::temp_directory_path()/"mimir_lumen_dynamic_context_raw";
    std::filesystem::remove_all(dynamic_checkpoint);
    const std::unordered_map<std::string,std::vector<float>> context_input={{"__input__",direct_chw},{"timestep",std::vector<float>(8,0.f)}};
    const std::unordered_map<std::string,std::vector<int>> context_ids={{"text_ids",std::vector<int>(9,1)}};
    const auto before_save=direct_model.forwardPassNamed(context_input,context_ids,false);
    Mimir::Serialization::SaveOptions dynamic_save;
    dynamic_save.format=Mimir::Serialization::CheckpointFormat::RawFolder;
    std::string dynamic_error;
    TASSERT_TRUE(Mimir::Serialization::save_checkpoint(direct_model,dynamic_checkpoint.string(),dynamic_save,&dynamic_error));
    TestableLumenModel restored_dynamic;restored_dynamic.buildFromConfig(direct_cfg);restored_dynamic.allocateParams();
    Mimir::Serialization::LoadOptions dynamic_load;dynamic_load.format=Mimir::Serialization::CheckpointFormat::RawFolder;
    TASSERT_TRUE(Mimir::Serialization::load_checkpoint(restored_dynamic,dynamic_checkpoint.string(),dynamic_load,&dynamic_error));
    const auto after_load=restored_dynamic.forwardPassNamed(context_input,context_ids,false);
    TASSERT_TRUE(before_save.size()==after_load.size());
    for(size_t i=0;i<before_save.size();++i)TASSERT_NEAR(before_save[i],after_load[i],1e-6f);
    TASSERT_TRUE(restored_dynamic.getLayerByName("lumen/text/position")->getWeightsSize()==position_parameters);
    std::filesystem::remove_all(dynamic_checkpoint);


    for (int invalid_case = 0; invalid_case < 3; ++invalid_case) {
        auto invalid_cfg = direct_cfg;
        if (invalid_case == 0) invalid_cfg.diffusion_steps = 1;
        if (invalid_case == 1) invalid_cfg.beta_start = 0.f;
        if (invalid_case == 2) invalid_cfg.beta_end = std::numeric_limits<float>::quiet_NaN();
        bool rejected_schedule = false;
        try { TestableLumenModel invalid; invalid.buildFromConfig(invalid_cfg); }
        catch (const std::invalid_argument&) { rejected_schedule = true; }
        TASSERT_TRUE(rejected_schedule);
    }

    bool rejected_invalid_image = false;
    try {
        Optimizer optimizer;
        lumen->trainDiffusionStep({}, "test", 123U, optimizer, 1e-4f);
    } catch (const std::runtime_error&) {
        rejected_invalid_image = true;
    }
    TASSERT_TRUE(rejected_invalid_image);

    bool rejected_invalid_patch_shape = false;
    try {
        LumenLatentDiffusionModel invalid_model;
        auto invalid_cfg = tips_cfg;
        invalid_cfg.latent_w = 10;
        invalid_model.buildFromConfig(invalid_cfg);
    } catch (const std::runtime_error&) {
        rejected_invalid_patch_shape = true;
    }
    TASSERT_TRUE(rejected_invalid_patch_shape);

    const auto vae_checkpoint = std::filesystem::temp_directory_path() /
        "mimir_lumen_adaptive_vae.safetensors";
    std::filesystem::remove(vae_checkpoint);
    VAEConvModel adaptive_vae;
    VAEConvModel::Config adaptive_vae_cfg;
    adaptive_vae_cfg.image_w = 8;
    adaptive_vae_cfg.image_h = 8;
    adaptive_vae_cfg.image_c = 3;
    adaptive_vae_cfg.latent_w = 2;
    adaptive_vae_cfg.latent_h = 2;
    adaptive_vae_cfg.latent_c = 2;
    adaptive_vae_cfg.base_channels = 8;
    adaptive_vae_cfg.latent_spatial_context = true;
    adaptive_vae_cfg.latent_detail_path = true;
    adaptive_vae_cfg.latent_context_channels = 8;
    adaptive_vae_cfg.stochastic_latent = false;
    adaptive_vae_cfg.resnet = false;
    adaptive_vae_cfg.attention = false;
    adaptive_vae_cfg.use_skip_connections = false;
    adaptive_vae_cfg.enc_norm = "none";
    adaptive_vae_cfg.dec_norm = "none";
    adaptive_vae_cfg.decoder_upsample = "nearest_conv";
    adaptive_vae.buildFromConfig(adaptive_vae_cfg);
    adaptive_vae.allocateParams();
    adaptive_vae.initializeWeights("xavier", 321U);

    Mimir::Serialization::SaveOptions save_options;
    save_options.format = Mimir::Serialization::CheckpointFormat::SafeTensors;
    save_options.save_tokenizer = false;
    save_options.save_encoder = false;
    save_options.save_optimizer = false;
    std::string save_error;
    TASSERT_TRUE(Mimir::Serialization::save_checkpoint(
        adaptive_vae, vae_checkpoint.string(), save_options, &save_error));

    TestableLumenModel adaptive_lumen;
    auto adaptive_lumen_cfg = tips_cfg;
    adaptive_lumen_cfg.image_w = 16;
    adaptive_lumen_cfg.image_h = 16;
    adaptive_lumen_cfg.latent_w = 4;
    adaptive_lumen_cfg.latent_h = 4;
    adaptive_lumen_cfg.latent_c = 3;
    adaptive_lumen_cfg.patch_size = 4;
    adaptive_lumen_cfg.vae_checkpoint = vae_checkpoint.string();
    adaptive_lumen.buildFromConfig(adaptive_lumen_cfg);

    const auto& adapted = adaptive_lumen.getConfig();
    TASSERT_TRUE(adapted.image_w == adaptive_vae_cfg.image_w);
    TASSERT_TRUE(adapted.image_h == adaptive_vae_cfg.image_h);
    TASSERT_TRUE(adapted.image_c == adaptive_vae_cfg.image_c);
    TASSERT_TRUE(adapted.latent_w == adaptive_vae_cfg.latent_w);
    TASSERT_TRUE(adapted.latent_h == adaptive_vae_cfg.latent_h);
    TASSERT_TRUE(adapted.latent_c == adaptive_vae_cfg.latent_c);
    TASSERT_TRUE(adapted.vae_stochastic_latent ==
                 adaptive_vae_cfg.stochastic_latent);
    TASSERT_TRUE(adapted.vae_decoder_upsample ==
                 adaptive_vae_cfg.decoder_upsample);
    TASSERT_TRUE(adapted.vae_latent_spatial_context && adapted.vae_latent_detail_path);
    TASSERT_TRUE(adapted.vae_latent_context_channels == 8);
    TASSERT_TRUE(adapted.patch_size == 2);
    TASSERT_TRUE(adaptive_lumen.modelConfig["input_dim"].get<int>() == 2 * 2 * 2);
    TASSERT_TRUE(adaptive_lumen.modelConfig["image_w"].get<int>() == 8);
    TASSERT_TRUE(adaptive_lumen.modelConfig["latent_c"].get<int>() == 2);
    adaptive_lumen.allocateParams();
    adaptive_lumen.initializeWeights("xavier", 45U);
    adaptive_lumen.modelConfig["vae_scale"] = 1.f;
    adaptive_lumen.setVizTapsEnabled(true);
    adaptive_lumen.setVizTapsLimits(64, 4);
    Optimizer vae_preview_optimizer;
    vae_preview_optimizer.type = OptimizerType::SGD;
    vae_preview_optimizer.decay_strategy = LRDecayStrategy::NONE;
    vae_preview_optimizer.weight_decay = .1f;
    const auto vae_preview_stats = adaptive_lumen.trainDiffusionStep(
        std::vector<unsigned char>(8 * 8 * 3, 127), "test", 123U, vae_preview_optimizer, 1e-4f);
    TASSERT_TRUE(std::isfinite(vae_preview_stats.loss));
    TASSERT_TRUE(adaptive_lumen.denoiser_forwards == 1);
    TASSERT_TRUE(adaptive_lumen.inference_forwards == 0);
    TASSERT_TRUE(adaptive_lumen.getTensor("x").size() == 2 * 2 * 2);
    TASSERT_TRUE(adaptive_lumen.getTensor("z").size() == 8 * 8 * 3);
    TASSERT_TRUE(adaptive_lumen.modelConfig["output_dim"] == 8 * 8 * 3);
    const auto one_pass_z = adaptive_lumen.getTensor("z");
    const auto independently_decoded = adaptive_lumen.decodeTestImage(
        adaptive_lumen.getTensor("lumen/reconstruction/clean_latent"));
    TASSERT_TRUE(independently_decoded.size() == one_pass_z.size());
    for (size_t i = 0; i < one_pass_z.size(); ++i)
        TASSERT_NEAR(one_pass_z[i], independently_decoded[i], 1e-6f);
    for (const auto& layer : adaptive_lumen.getLayers()) {
        if (layer.name.rfind("lumen/reconstruction/decoder/", 0) != 0) continue;
        for (size_t i = 0; i < layer.getWeightsSize(); ++i)
            TASSERT_NEAR(layer.getWeights()[i], layer.weights[i], 1e-7f);
    }
    const auto public_output = adaptive_lumen.forwardPassNamed(
        {{"__input__", adaptive_lumen.getTensor("__input__")},
         {"timestep", adaptive_lumen.getTensor("lumen/time/embedding")},
         {"diffusion_signal_scale", adaptive_lumen.getTensor("diffusion_signal_scale")},
         {"diffusion_noise_scale", adaptive_lumen.getTensor("diffusion_noise_scale")}},
        {{"text_ids", {1}}}, true);
    TASSERT_TRUE(public_output.size() == 8 * 8 * 3);
    TASSERT_TRUE(public_output == adaptive_lumen.getTensor("z"));
    TASSERT_TRUE(adaptive_lumen.inference_forwards == 0);
    for (const auto& frame : adaptive_lumen.consumeVizTaps()) {
        TASSERT_TRUE(frame.label != "diffusion_out");
        TASSERT_TRUE(frame.label.find("diffusion/compare/") != 0);
    }
    const auto vae_validation_stats = adaptive_lumen.validateDiffusionStep(
        std::vector<unsigned char>(8 * 8 * 3, 127), "test", 123U);
    TASSERT_TRUE(std::isfinite(vae_validation_stats.loss));
    bool decoded_vae_preview = false;
    for (const auto& frame : adaptive_lumen.consumeVizTaps()) {
        TASSERT_TRUE(frame.label.find("vae_conv/blocks/") == std::string::npos);
        if (frame.label == "diffusion_out") decoded_vae_preview = true;
    }
    TASSERT_TRUE(decoded_vae_preview);
    std::filesystem::remove(vae_checkpoint);

    const auto raw_vae_checkpoint = std::filesystem::temp_directory_path() /
        "mimir_lumen_adaptive_vae_raw";
    std::filesystem::remove_all(raw_vae_checkpoint);
    save_options.format = Mimir::Serialization::CheckpointFormat::RawFolder;
    TASSERT_TRUE(Mimir::Serialization::save_checkpoint(
        adaptive_vae, raw_vae_checkpoint.string(), save_options, &save_error));
    LumenLatentDiffusionModel raw_adaptive_lumen;
    adaptive_lumen_cfg.vae_checkpoint = raw_vae_checkpoint.string();
    raw_adaptive_lumen.buildFromConfig(adaptive_lumen_cfg);
    TASSERT_TRUE(raw_adaptive_lumen.getConfig().image_w == 8);
    TASSERT_TRUE(raw_adaptive_lumen.getConfig().latent_w == 2);
    TASSERT_TRUE(raw_adaptive_lumen.getConfig().latent_c == 2);
    TASSERT_TRUE(raw_adaptive_lumen.getConfig().patch_size == 2);
    TASSERT_TRUE(raw_adaptive_lumen.modelConfig["input_dim"].get<int>() == 2 * 2 * 2);
    std::filesystem::remove_all(raw_vae_checkpoint);

    return 0;
}
