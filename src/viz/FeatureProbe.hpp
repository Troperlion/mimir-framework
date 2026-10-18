#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include <utility>

namespace FeatureProbe {
struct Result {
    std::vector<size_t> cells;
    std::vector<std::pair<int,float>> channels;
};
// Connected spatial section matching the hovered raw activation vector.
// This measures feature similarity, not semantic detection or attribution.
inline Result analyze(const std::vector<float>& values,int w,int h,int channels,int x,int y) {
    Result result;
    if(w<=0 || h<=0 || channels<2 || x<0 || y<0 || x>=w || y>=h ||
       values.size()!=size_t(w)*h*channels) return result;
    const size_t seed=size_t(y)*w+x, base=seed*channels;
    double seed_norm=0;
    for(int c=0;c<channels;++c) {
        const float v=values[base+c];
        if(!std::isfinite(v))return {};
        seed_norm+=double(v)*v;
        result.channels.emplace_back(c,v);
    }
    if(seed_norm<=1e-20)return {};
    std::partial_sort(result.channels.begin(),result.channels.begin()+std::min<size_t>(3,result.channels.size()),result.channels.end(),
        [](const auto& a,const auto& b){return std::abs(a.second)>std::abs(b.second);});
    result.channels.resize(std::min<size_t>(3,result.channels.size()));
    std::vector<unsigned char> visited(size_t(w)*h,0);
    std::vector<size_t> pending{seed};visited[seed]=1;
    while(!pending.empty()) {
        const auto p=pending.back();pending.pop_back();
        double norm=0,dot=0;
        for(int c=0;c<channels;++c){const double v=values[p*channels+c];norm+=v*v;dot+=v*values[base+c];}
        if(!std::isfinite(norm) || !std::isfinite(dot) || norm<=1e-20 || dot/std::sqrt(norm*seed_norm)<.85)continue;
        result.cells.push_back(p);
        auto enqueue=[&](size_t q){if(!visited[q]){visited[q]=1;pending.push_back(q);}};
        if(p%w>0)enqueue(p-1);
        if(p%w+1<size_t(w))enqueue(p+1);
        if(p>=size_t(w))enqueue(p-w);
        if(p+size_t(w)<visited.size())enqueue(p+w);
    }
    std::sort(result.cells.begin(),result.cells.end());
    return result;
}
}
