#pragma once
#include "Layers.hpp"
#include <climits>
#include <cmath>

namespace PortableLayerGeometry {
// Match each CPU entry point, including the historical backward convention
// which uses in_features or the whole input, rather than forward shape hints.
inline int rowNormWidth(const Layer& layer, size_t count, bool backward) {
    if (count > INT_MAX) return 0;
    if (layer.in_features > 0) return layer.in_features;
    if (!backward && layer.type_enum == LayerType::RMSNorm) {
        if (layer.embed_dim > 0) return layer.embed_dim;
        if (!layer.shape.empty() && layer.shape.back() > 0) return layer.shape.back();
        if (!layer.target_shape.empty() && layer.target_shape.back() > 0) return layer.target_shape.back();
    }
    return static_cast<int>(count);
}
inline bool rowNormFits(const Layer& layer, size_t count, bool backward) {
    const int width = rowNormWidth(layer, count, backward);
    if (width <= 0 || count % static_cast<size_t>(width) ||
        !std::isfinite(layer.eps) || layer.eps <= 0) return false;
    if (!backward && layer.type_enum==LayerType::LayerNorm && layer.affine)
        return layer.getWeights() && layer.getWeightsSize()>=static_cast<size_t>(width)*(layer.use_bias?2:1);
    // RMS forward fills missing affine coefficients with 1; backward requires a
    // complete gamma vector whenever a non-null parameter block is present.
    return !backward || !layer.affine || !layer.getWeights() ||
        layer.getWeightsSize() >= static_cast<size_t>(width);
}
}
