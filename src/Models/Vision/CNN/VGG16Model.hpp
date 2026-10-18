#pragma once

#include "../Model.hpp"

// VGG16Model (simplified, float-only API)
// Input: float[image_dim]
// Output: float[num_classes]

class VGG16Model : public Model {
public:
    struct Config {
        int image_w = 64;
        int image_h = 64;
        int image_c = 3;
        int base_channels = 32;
        int num_classes = 1000;
        int fc_hidden = 512;
        float dropout = 0.5f;
    };

    VGG16Model();
    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);

    bool InitVizTips() override;
    bool UpdateVizTips(const Layer& layer, VizFrame& frame) override;

private:
    Config cfg_;
};
