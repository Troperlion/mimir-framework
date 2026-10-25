#include "test_utils.hpp"
#include "Models/VAETraining.hpp"

int main() {
    for (bool text : {false, true}) {
        Model model;
        model.modelConfig = {{"type", text ? "vae_text" : "vae"}, {"image_dim", 1},
            {"latent_dim", 1}, {"proj_dim", 1}, {"align_weight", 0},
            {"kl_beta", 0.8}, {"kl_warmup_steps", 4}, {"recon_loss", "mse"}};
        model.push("packed", "Linear", text ? 5 : 3);
        auto& layer = model.getMutableLayers()[0];
        layer.inputs = {"__input__"}; layer.output = "x";
        layer.in_features = 1; layer.out_features = text ? 5 : 3; layer.use_bias = false;
        model.allocateParams();
        const float initial[] = {1.f, 2.f, .5f, 1.f, 1.f};
        std::copy(initial, initial + layer.out_features, layer.getWeights());
        Optimizer optimizer;
        VAETraining training;
        const std::vector<float> input{1.f};
        auto run = [&]() {
            model.zeroGradients();
            return text ? training.trainText(model, input, {}, nullptr, nullptr, optimizer, 0,
                                             Model::TrainStepMode::Accumulate, 1)
                        : training.trainImage(model, input, nullptr, nullptr, optimizer, 0,
                                              Model::TrainStepMode::Accumulate, 1);
        };
        const float beta = .2f; // first step of the shared warmup
        const float lv = .5f * (std::exp(.5f) - 1.f - .5f);
        auto baseline = run();
        TASSERT_NEAR(baseline.loss, beta * (2.f + lv), 1e-6f);
        TASSERT_NEAR(layer.grad_weights[1], beta * 2.f, 1e-6f);
        TASSERT_NEAR(layer.grad_weights[2], beta * .5f * (std::exp(.5f)-1.f), 1e-6f);
        model.publishRuntimeConfiguration();
        auto config = model.runtimeConfiguration();
        std::string error;
        TASSERT_TRUE(!config->request("kl_mu_weight", "-1", error));
        TASSERT_TRUE(!config->request("kl_logvar_weight", "nan", error));
        TASSERT_TRUE(config->request("kl_mu_weight", "0", error));
        TASSERT_TRUE(!model.modelConfig.contains("kl_mu_weight"));
        auto logvar_only = run();
        TASSERT_NEAR(logvar_only.loss, beta * lv, 1e-6f);
        TASSERT_NEAR(layer.grad_weights[1], 0, 1e-6f);
        TASSERT_NEAR(layer.grad_weights[2], beta * .5f * (std::exp(.5f)-1.f), 1e-6f);
        TASSERT_NEAR(logvar_only.metrics.at("kl_mu_beta_effective"), 0, 1e-6f);
        TASSERT_TRUE(config->request("kl_mu_weight", "2", error));
        TASSERT_TRUE(config->request("kl_logvar_weight", "0", error));
        auto mu_only = run();
        TASSERT_NEAR(mu_only.loss, beta * 4.f, 1e-6f);
        TASSERT_NEAR(layer.grad_weights[1], beta * 4.f, 1e-6f);
        TASSERT_NEAR(layer.grad_weights[2], 0, 1e-6f);
        TASSERT_NEAR(mu_only.metrics.at("kl_logvar_beta_effective"), 0, 1e-6f);
        TASSERT_TRUE(config->request("kl_mu_weight", "0", error));
        auto disabled = run();
        TASSERT_NEAR(disabled.loss, 0, 1e-6f);
        TASSERT_NEAR(layer.grad_weights[1], 0, 1e-6f);
        TASSERT_NEAR(layer.grad_weights[2], 0, 1e-6f);
    }
    return 0;
}
