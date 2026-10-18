#pragma once
#include "Graphics.hpp"
#include <cmath>

namespace mimir::viz::ui {
// Rounded geometry stays native in every renderer; no bitmap button assets.
template<class Target>
void roundedRect(Target& target, vizgfx::FloatRect bounds, vizgfx::Color color, float radius=5.f) {
    const float r=std::min(radius,std::min(bounds.size.x,bounds.size.y)*.5f);
    auto strip=[&](float x,float y,float w,float h) {
        if(w<=0 || h<=0)return;
        vizgfx::RectangleShape shape({w,h});shape.setPosition({x,y});shape.setFillColor(color);target.draw(shape);
    };
    strip(bounds.position.x,bounds.position.y+r,bounds.size.x,bounds.size.y-2*r);
    for(int row=0;row<int(std::ceil(r));++row) {
        const float dy=r-row-.5f;
        const float inset=r-std::sqrt(std::max(0.f,r*r-dy*dy));
        strip(bounds.position.x+inset,bounds.position.y+row,bounds.size.x-2*inset,1);
        strip(bounds.position.x+inset,bounds.position.y+bounds.size.y-row-1,bounds.size.x-2*inset,1);
    }
}
inline vizgfx::Color brighten(vizgfx::Color c,int amount) {
    return {uint8_t(std::min(255,int(c.r)+amount)),uint8_t(std::min(255,int(c.g)+amount)),uint8_t(std::min(255,int(c.b)+amount)),c.a};
}
template<class Target>
void buttonSurface(Target& target, vizgfx::FloatRect bounds, vizgfx::Color fill, bool hover=false) {
    roundedRect(target,{{bounds.position.x,bounds.position.y+2},bounds.size},{12,15,20,90});
    roundedRect(target,bounds,hover?vizgfx::Color(126,151,178):vizgfx::Color(77,85,98));
    roundedRect(target,{{bounds.position.x+1,bounds.position.y+1},{bounds.size.x-2,bounds.size.y-2}},hover?brighten(fill,13):fill,4);
}
}
