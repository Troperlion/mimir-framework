#include "test_utils.hpp"
#include "Model.hpp"
#include "AsyncMonitor.hpp"
#include "Models/Vision/UNetModel.hpp"
#include "Models/Vision/VAEConvModel.hpp"
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    // Optional terminal integration smoke: press S twice in a real PTY.
    if (argc > 1 && std::string(argv[1]) == "--htop") {
        auto state = std::make_shared<SkipConnectionControl>();
        state->available = true;
        AsyncMonitor monitor;
        monitor.bindSkipConnectionControl(state);
        monitor.configureMetricsCsv("", false);
        monitor.start(true, false);
        bool saw_off = false, saw_on_again = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (std::chrono::steady_clock::now() < deadline) {
            if (!state->enabled.load()) saw_off = true;
            if (saw_off && state->enabled.load()) { saw_on_again = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        monitor.stop();
        TASSERT_TRUE(saw_off && saw_on_again);
        return 0;
    }
    for (const bool legacy : {false, true}) {
        Model model;
        model.push(legacy ? "vae_conv/dec/up0/skip_cat" : "custom_skip", "Concat", 0);
        auto& layer = model.getMutableLayers().front();
        layer.inputs = {"__input__", "__input__"};
        layer.output = "x";
        layer.concat_axis = 0;
        if (!legacy) layer.skip_input_index = 1;
        model.allocateParams();
        const auto control = model.skipConnectionControl();
        TASSERT_TRUE(control->available.load());
        TASSERT_TRUE(control->enabled.load());
        const std::vector<float> x{2, 3};
        TASSERT_TRUE(model.forwardPass(x, true) == std::vector<float>({2, 3, 2, 3}));
        // A click between forward and backward must affect only the next pass.
        control->toggle();
        model.backwardPass({1, 1, 1, 1});
        TASSERT_TRUE(model.getLastInputGradient() == std::vector<float>({2, 2}));
        TASSERT_TRUE(model.forwardPass(x, true) == std::vector<float>({2, 3, 0, 0}));
        TASSERT_TRUE(!model.modelConfig.at("skip_connections_enabled").get<bool>());
        control->toggle();
        model.backwardPass({1, 1, 1, 1});
        TASSERT_TRUE(model.getLastInputGradient() == std::vector<float>({1, 1}));
        TASSERT_TRUE(model.forwardPass(x, false) == std::vector<float>({2, 3, 2, 3}));
        // Rebinding interfaces must never reinitialize a manual choice.
        control->toggle();
        AsyncMonitor monitor;
        monitor.bindSkipConnectionControl(model.skipConnectionControl());
        monitor.bindSkipConnectionControl(model.skipConnectionControl());
        TASSERT_TRUE(!control->enabled.load());
    }
    Model plain;
    plain.push("pack_outputs", "Concat", 0);
    plain.getMutableLayers()[0].inputs = {"__input__", "__input__"};
    plain.allocateParams();
    auto unavailable = plain.skipConnectionControl();
    TASSERT_TRUE(!unavailable->available.load());
    unavailable->toggle();
    TASSERT_TRUE(unavailable->enabled.load());
    unavailable->enabled = false;
    TASSERT_TRUE(plain.forwardPass(std::vector<float>{7}, false) == std::vector<float>({7, 7}));

    Model annotated;
    annotated.modelConfig["skip_connections_enabled"] = false;
    annotated.push("residual", "Add", 0);
    auto& sum = annotated.getMutableLayers()[0];
    sum.inputs = {"__input__", "__input__"};
    sum.skip_input_index = 0;
    annotated.allocateParams();
    TASSERT_TRUE(!annotated.skipConnectionControl()->enabled.load());
    TASSERT_TRUE(annotated.forwardPass(std::vector<float>{4}, true) == std::vector<float>({4}));
    annotated.backwardPass({1});
    TASSERT_TRUE(annotated.getLastInputGradient() == std::vector<float>({1}));
    // Exercise the real reconstruction graph, not just an isolated merge.
    VAEConvModel vae;
    VAEConvModel::Config cfg;
    cfg.image_w = cfg.image_h = 4;
    cfg.image_c = cfg.latent_c = 1;
    cfg.latent_w = cfg.latent_h = 2;
    cfg.base_channels = 8;
    cfg.resnet = cfg.attention = cfg.stochastic_latent = false;
    cfg.enc_norm = cfg.dec_norm = "none";
    cfg.use_skip_connections = true;
    vae.buildFromConfig(cfg);
    vae.allocateParams();
    vae.initializeWeights("xavier", 123u);
    const auto vae_control = vae.skipConnectionControl();
    TASSERT_TRUE(vae_control->available.load());
    const std::vector<float> pixels(16, 0.5f);
    const auto on = vae.forwardPass(pixels, false);
    const Layer* merge = nullptr;
    for (const auto& layer : vae.getLayers()) if (layer.skipInputIndex() >= 0) merge = &layer;
    TASSERT_TRUE(merge != nullptr);
    const auto on_cat = vae.getTensor(merge->output);
    TASSERT_TRUE(std::any_of(on_cat.begin() + on_cat.size()/2, on_cat.end(),
        [](float x) { return x != 0; }));
    vae_control->toggle();
    const auto off = vae.forwardPass(pixels, false);
    const auto off_cat = vae.getTensor(merge->output);
    TASSERT_TRUE(on.size() == off.size());
    TASSERT_TRUE(std::all_of(off_cat.begin() + off_cat.size()/2, off_cat.end(),
        [](float x) { return x == 0; }));
    vae_control->toggle();
    TASSERT_TRUE(vae.forwardPass(pixels, false) == on);

    UNetModel unet;
    UNetModel::Config uc;
    uc.image_w = uc.image_h = 4;
    uc.image_c = 1; uc.base_channels = 2; uc.depth = 1;
    unet.buildFromConfig(uc);
    TASSERT_TRUE(unet.skipConnectionControl()->available.load());
    for (auto& layer : unet.getMutableLayers()) layer.skip_input_index = -1;
    TASSERT_TRUE(unet.skipConnectionControl()->available.load());
    return 0;
}
