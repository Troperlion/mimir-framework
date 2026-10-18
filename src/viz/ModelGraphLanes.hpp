#pragma once
#include "Graphics.hpp"
#include "include/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace Mimir::Viz {
struct ModelGraphLanes {
    struct Edge { size_t from,to,rail; };
    std::vector<size_t> lanes;
    std::vector<Edge> edges;
    size_t columns=1;
    static constexpr float spacing=26.f,row=66.f,left=24.f,top=32.f;
    float labelsX() const { return left+float(columns)*spacing+24.f; }
    vizgfx::Vector2f position(size_t i) const { return {left+float(lanes[i])*spacing,top+float(i)*row}; }
};

// Keep an outgoing branch's rail reserved until its destination is visited.
// This prevents skip connections from running through unrelated node markers.
inline ModelGraphLanes modelGraphLanes(const nlohmann::json& graph) {
    ModelGraphLanes result;
    const size_t count=graph.at("nodes").size(),none=std::numeric_limits<size_t>::max();
    std::vector<std::vector<size_t>> outgoing(count);
    for(const auto& e:graph.at("edges")) {
        const size_t from=e.at("from"),to=e.at("to");
        if(from<count && to<count && from<to)outgoing[from].push_back(to);
    }
    std::vector<size_t> tracks;
    auto freeRail=[&]() {
        const auto found=std::find(tracks.begin(),tracks.end(),none);
        if(found!=tracks.end())return size_t(found-tracks.begin());
        tracks.push_back(none);return tracks.size()-1;
    };
    for(size_t i=0;i<count;++i) {
        const auto found=std::find(tracks.begin(),tracks.end(),i);
        const size_t lane=found==tracks.end()?freeRail():size_t(found-tracks.begin());
        result.lanes.push_back(lane);
        for(auto& target:tracks)if(target==i)target=none;
        auto& children=outgoing[i];
        std::sort(children.begin(),children.end());children.erase(std::unique(children.begin(),children.end()),children.end());
        for(size_t j=0;j<children.size();++j) {
            const size_t rail=j==0 && tracks[lane]==none?lane:freeRail();
            tracks[rail]=children[j];result.edges.push_back({i,children[j],rail});
        }
        result.columns=std::max(result.columns,tracks.size());
    }
    return result;
}

inline vizgfx::Color graphLaneColor(size_t lane,bool dim=false) {
    static const std::array<vizgfx::Color,8> palette={vizgfx::Color(87,204,190),vizgfx::Color(172,136,255),
        vizgfx::Color(255,184,99),vizgfx::Color(99,173,255),vizgfx::Color(246,122,160),
        vizgfx::Color(171,211,103),vizgfx::Color(232,209,105),vizgfx::Color(104,206,238)};
    auto color=palette[lane%palette.size()];
    if(dim){color.r=uint8_t(color.r*.53f);color.g=uint8_t(color.g*.53f);color.b=uint8_t(color.b*.53f);}
    return color;
}

inline std::vector<vizgfx::Vector2f> graphLaneCurve(const ModelGraphLanes& layout,const ModelGraphLanes::Edge& edge) {
    const auto a=layout.position(edge.from),b=layout.position(edge.to);
    const float rail=ModelGraphLanes::left+float(edge.rail)*ModelGraphLanes::spacing;
    const float bend=std::min(24.f,(b.y-a.y)*.35f);
    std::vector<vizgfx::Vector2f> path;path.reserve(35);
    auto curve=[&](vizgfx::Vector2f p,vizgfx::Vector2f c1,vizgfx::Vector2f c2,vizgfx::Vector2f q) {
        for(int k=0;k<=16;++k) {
            const float t=float(k)/16.f,u=1.f-t;
            path.push_back({u*u*u*p.x+3*u*u*t*c1.x+3*u*t*t*c2.x+t*t*t*q.x,
                            u*u*u*p.y+3*u*u*t*c1.y+3*u*t*t*c2.y+t*t*t*q.y});
        }
    };
    curve(a,{a.x,a.y+bend*.65f},{rail,a.y+bend*.35f},{rail,a.y+bend});
    curve({rail,b.y-bend},{rail,b.y-bend*.35f},{b.x,b.y-bend*.65f},b);
    return path;
}

template<class Canvas> inline void graphDisc(Canvas& canvas,vizgfx::Vector2f p,float radius,vizgfx::Color color) {
    // Horizontal chords work on every Viz backend, including the Cairo renderer.
    for(float y=-radius;y<=radius;y+=1.f) {
        const float half=std::sqrt(std::max(0.f,radius*radius-y*y));
        vizgfx::RectangleShape chord({2*half,1.f});chord.setPosition({p.x-half,p.y+y});
        chord.setFillColor(color);canvas.draw(chord);
    }
}

template<class Canvas> inline void drawModelGraphLanes(Canvas& canvas,const ModelGraphLanes& layout,
    size_t selected,float firstY,float lastY) {
    for(const auto& edge:layout.edges) {
        if(layout.position(edge.to).y<firstY || layout.position(edge.from).y>lastY)continue;
        const auto path=graphLaneCurve(layout,edge);
        const bool active=edge.from==selected || edge.to==selected;
        const auto color=graphLaneColor(edge.rail,!active);
        // Multiple parallel strokes retain consistent width on SFML and Cairo.
        for(int offset=-1;offset<=1;++offset) {
            vizgfx::VertexArray vertices(vizgfx::PrimitiveType::LineStrip,path.size());
            for(size_t i=0;i<path.size();++i){vertices[i].position={path[i].x+float(offset),path[i].y};vertices[i].color=color;}
            canvas.draw(vertices);
        }
    }
    for(size_t i=0;i<layout.lanes.size();++i) {
        const auto p=layout.position(i);
        if(p.y<firstY-12 || p.y>lastY+12)continue;
        if(i==selected)graphDisc(canvas,p,11.f,vizgfx::Color(228,238,248));
        graphDisc(canvas,p,i==selected?8.f:6.f,graphLaneColor(layout.lanes[i]));
        graphDisc(canvas,p,2.f,vizgfx::Color(20,26,36));
    }
}
}
