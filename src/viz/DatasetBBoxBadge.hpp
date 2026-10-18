#pragma once
#include "Graphics.hpp"

namespace mimir::viz {
// Backend-neutral vector icon: no Unicode glyph/font dependency.
template<class Target>
void drawDatasetBBoxBadge(Target& target, float x, float y, float panel_size, bool active) {
    if (!active) return;
    const float side=24.f, bx=x+panel_size-side-5.f, by=y+5.f;
    vizgfx::RectangleShape background({side,side});
    background.setPosition({bx,by});
    background.setFillColor(vizgfx::Color(15,35,45,235));
    background.setOutlineColor(vizgfx::Color(85,220,240));
    background.setOutlineThickness(1.f);
    target.draw(background);
    auto bar=[&](float dx,float dy,float w,float h) {
        vizgfx::RectangleShape shape({w,h});
        shape.setPosition({bx+dx,by+dy});
        shape.setFillColor(vizgfx::Color(110,240,255));
        target.draw(shape);
    };
    for(float dx:{5.f,13.f}) for(float dy:{5.f,17.f}) bar(dx,dy,6.f,2.f);
    for(float dx:{5.f,17.f}) for(float dy:{5.f,13.f}) bar(dx,dy,2.f,6.f);
    bar(10.f,10.f,4.f,4.f);
}
}
