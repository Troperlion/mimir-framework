#pragma once

#include "../Model.hpp"

// VGG19Model (canonical VGG19 classifier, float-only API)
// Input: float[image_dim]
// Output: float[num_classes]

class VGG19Model : public Model {
public:
    struct Config {
        int image_w = 224;
        int image_h = 224;
        int image_c = 3;
        int base_channels = 64;
        int num_classes = 1000;
        int fc_hidden = 4096;
        float dropout = 0.5f;
    };

    VGG19Model();
    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);

    bool InitVizTips() override;
    bool UpdateVizTips(const Layer& layer, VizFrame& frame) override;

private:
    Config cfg_;
};
