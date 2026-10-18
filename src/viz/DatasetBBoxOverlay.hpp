#pragma once
#include "include/json.hpp"
#include <array>
#include <vector>
#include <cmath>

namespace DatasetBBoxOverlay {
using Boxes = std::vector<std::array<float, 4>>;
inline Boxes fromAnnotation(const nlohmann::json* annotation) {
    Boxes boxes;
    if (!annotation || !annotation->is_object() || !annotation->contains("entities") ||
        !(*annotation)["entities"].is_array()) return boxes;
    for (const auto& entity : (*annotation)["entities"]) {
        if (!entity.is_object() || !entity.contains("bbox")) continue;
        const auto& b = entity["bbox"];
        if (!b.is_array() || b.size()!=4) continue;
        std::array<float,4> box{};
        bool valid=true;
        for (size_t i=0;i<4;++i) {
            if (!b[i].is_number()) { valid=false; break; }
            box[i]=b[i].get<float>();
            valid=valid && std::isfinite(box[i]) && box[i]>=0 && box[i]<=1;
        }
        if (valid && box[0]<box[2] && box[1]<box[3]) boxes.push_back(box);
    }
    return boxes;
}
// Coordinates follow the displayed sprite, including its aspect ratio and 1:1 scale.
inline std::array<float,4> displayed(const std::array<float,4>& b, float x,float y,float w,float h) {
    return {x+b[0]*w,y+b[1]*h,(b[2]-b[0])*w,(b[3]-b[1])*h};
}
}
