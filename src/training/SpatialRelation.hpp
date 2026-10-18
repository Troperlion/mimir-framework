#pragma once
#include <array>
#include <string>

// A directed pair of annotated regions, expressed on the actual supervised grid.
struct SpatialRelation {
    std::array<double,4> subject{}, object{};
    int width=0, height=0, channels=0;
    bool chw=false;
    float weight=0;
    std::string predicate;
};
