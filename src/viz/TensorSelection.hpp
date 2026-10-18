#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TensorSelection {
struct Summary {
    int x0=0, y0=0, x1=0, y1=0;
    size_t finite=0, zeros=0, nan=0, inf=0;
    double mean=0, stddev=0, l2=0;
    float min=0, max=0;
    std::vector<std::pair<size_t,float>> values;
};
// Values are the raw, interleaved preview samples, with original tensor indices.
// Deduplicate indices introduced by nearest resizing / vector padding.
inline Summary analyze(const std::vector<float>& values, const std::vector<size_t>& indices,
                       int w, int h, int channels, int ax, int ay, int bx, int by) {
    Summary result;
    if (w<=0 || h<=0 || channels<=0) return result;
    result.x0=std::clamp(std::min(ax,bx),0,w-1); result.x1=std::clamp(std::max(ax,bx),0,w-1);
    result.y0=std::clamp(std::min(ay,by),0,h-1); result.y1=std::clamp(std::max(ay,by),0,h-1);
    std::unordered_set<size_t> seen;
    double squared=0, m2=0;
    for (int y=result.y0;y<=result.y1;++y) for(int x=result.x0;x<=result.x1;++x)
        for(int c=0;c<channels;++c) {
            const size_t offset=(static_cast<size_t>(y)*w+x)*channels+c;
            if(offset>=values.size()) continue;
            const size_t index=offset<indices.size()?indices[offset]:offset;
            if(!seen.insert(index).second) continue;
            const float value=values[offset]; result.values.emplace_back(index,value);
            if(std::isnan(value)) {++result.nan;continue;}
            if(!std::isfinite(value)) {++result.inf;continue;}
            if(result.finite==0) result.min=result.max=value;
            result.min=std::min(result.min,value);result.max=std::max(result.max,value);
            ++result.finite; if(value==0.f) ++result.zeros;
            const double delta=static_cast<double>(value)-result.mean;
            result.mean+=delta/static_cast<double>(result.finite);
            m2+=delta*(static_cast<double>(value)-result.mean);
            squared+=static_cast<double>(value)*value;
        }
    if(result.finite) result.stddev=std::sqrt(std::max(0.0,m2/result.finite));
    result.l2=std::sqrt(squared);
    return result;
}
}
