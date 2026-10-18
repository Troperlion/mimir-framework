#pragma once
#include "Model.hpp"

// CHW images in [-1,1]. target is an unpaired image from the target style.
// Both networks belong to this graph/checkpoint. D uses stateless SGD; G uses
// the caller's optimizer. No discriminator work is needed for inference.
class GanStyleModel : public Model {
public:
    struct Config {
        int image_w=64, image_h=64, image_c=3;
        int base_channels=16, discriminator_channels=16;
        float content_weight=10.f, discriminator_lr=0.001f;
    };
    GanStyleModel();
    void buildFromConfig(const Config& cfg);
    std::optional<TrainStepResult> trainStep(const TrainStepRequest& request) override;
private:
    void selectOutput(bool discriminator);
};
