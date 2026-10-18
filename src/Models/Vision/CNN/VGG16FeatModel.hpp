#pragma once

#include "Model.hpp"
#include <string>

// VGG16FeatModel: small VGG-like feature extractor for perceptual loss.
// Input: HWC float image vector in [-1,1]. Output: concatenated GAP features across blocks.
class VGG16FeatModel : public Model {
public:
    struct Config {
        int image_w = 64;
        int image_h = 64;
        int image_c = 3;
        int base_channels = 8;
        // "lineargroup": LayerNorm globale C*H*W (topologie historique).
        // "groupnorm": GroupNorm par groupes de canaux.
        std::string enc_norm = "lineargroup";
        int enc_gn_groups = 32;
    };

    VGG16FeatModel();
    void buildFromConfig(const Config& cfg);
    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_{};
};
