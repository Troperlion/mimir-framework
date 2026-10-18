#pragma once
#include <string_view>

namespace Mimir::Viz {
inline bool isLumenModelOutput(std::string_view label) {
    return label.find("lumen/output/z/") != std::string_view::npos ||
           label.find("role=model_output tensor=z") != std::string_view::npos;
}
inline bool isLumenIntermediate(std::string_view label) {
    return !isLumenModelOutput(label) && label.find("lumen/") != std::string_view::npos;
}
// Use the semantic Lumen path, not words in the model name (lumen_diffusion)
// or the human caption ("reconstructed image" on the public z output).
inline std::string_view lumenTipSection(std::string_view label) {
    if (isLumenModelOutput(label)) return "Outputs";
    if (label.rfind("diffusion/compare/", 0) == 0 ||
        label.rfind("diffusion/reference", 0) == 0 ||
        label.rfind("diffusion_out", 0) == 0 ||
        label.rfind("resdiff_", 0) == 0) return "Validation";
    const auto wrapper = label.find("/blocks/");
    if (wrapper != std::string_view::npos) label.remove_prefix(wrapper + std::string_view("/blocks/").size());
    const auto begin = label.find("lumen/");
    if (begin == std::string_view::npos) return {};
    const auto path = label.substr(begin);
    if (path.rfind("lumen/text/", 0) == 0) return "Text / Cond";
    if (path.rfind("lumen/time/", 0) == 0 || path.rfind("lumen/dit/", 0) == 0) return "Diffusion";
    if (path.rfind("lumen/reconstruction/decoder/", 0) == 0) return "VAE / Decoder";
    if (path.rfind("lumen/reconstruction/clean_latent", 0) == 0) return "Latent";
    if (path.rfind("lumen/reconstruction/", 0) == 0) return "Reconstruction";
    return {};
}

}
