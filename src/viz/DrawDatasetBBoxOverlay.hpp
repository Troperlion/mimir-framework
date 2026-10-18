#pragma once
#include "DatasetBBoxOverlay.hpp"
#include "Graphics.hpp"

namespace DatasetBBoxOverlay {
template<class Target>
void draw(Target& target, const Boxes& boxes, float x, float y, float w, float h) {
    for (const auto& box : boxes) {
        const auto r = displayed(box, x, y, w, h);
        vizgfx::RectangleShape outline(vizgfx::Vector2f(r[2], r[3]));
        outline.setPosition(vizgfx::Vector2f(r[0], r[1]));
        outline.setFillColor(vizgfx::Color(0, 0, 0, 0));
        outline.setOutlineColor(vizgfx::Color(70, 220, 255, 255));
        outline.setOutlineThickness(-2.0f);
        target.draw(outline);
    }
}
}
