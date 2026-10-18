#pragma once
#include "Graphics.hpp"
#include <array>
namespace mimir::viz {
struct BlockToolbarLayout {
    std::array<vizgfx::FloatRect,6> controls;
    float height=0;
};
inline BlockToolbarLayout blockToolbarLayout(vizgfx::Vector2f origin,float width) {
    constexpr std::array<float,6> widths{84,98,100,84,92,98};
    BlockToolbarLayout result;
    float x=8,y=26;
    for(std::size_t i=0;i<widths.size();++i) {
        if(x>8 && x+widths[i]>width-8) {x=8;y+=22;}
        result.controls[i]={{origin.x+x,origin.y+y},{widths[i],18}};
        x+=widths[i]+6;
    }
    result.height=y+22;
    return result;
}
}
