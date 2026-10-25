#pragma once
#include "Graphics.hpp"
#include "include/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
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

struct ModelGraphTree {
    struct Edge { size_t from, to; };
    std::vector<size_t> depths, lanes;
    std::vector<Edge> edges;
    size_t columns=0, rows=0;
    float card_width=190.f;
    float card_height=70.f,column_gap=80.f,top=35.f;
    static constexpr float row_gap=40.f;
    static constexpr float left=120.f;
    float width() const { return left+float(columns)*(card_width+column_gap)+24.f; }
    float height() const {
        return top+float(rows)*card_height+(rows>0?float(rows-1)*row_gap:0.f)+24.f;
    }
    vizgfx::Vector2f position(size_t i) const {
        return {left+float(depths[i])*(card_width+column_gap)+card_width*.5f,
                this->top+float(lanes[i])*(card_height+row_gap)+card_height*.5f};
    }
};

inline ModelGraphTree modelGraphTree(const nlohmann::json& graph) {
    ModelGraphTree result;
    const auto& nodes=graph.at("nodes");
    result.depths.reserve(nodes.size());
    result.lanes.reserve(nodes.size());
    for(const auto& node:nodes) {
        const size_t depth=node.value("depth",size_t(0));
        const size_t lane=node.value("lane",size_t(0));
        std::string badges="params "+node.value("parameters",nlohmann::json(0)).dump();
        if(node.value("graph_fused",false))badges+="  | fusion";
        if(node.value("kernel_fused",false))badges+="  | kernel";
        if(node.value("skip",false))badges+="  | skip";
        const std::string details=node.value("type",std::string())+"  |  "+
            node.value("runtime",std::string());
        const size_t label_width=std::max({node.value("name",std::string()).size()+12,
            details.size(),badges.size()});
        result.card_width=std::max(result.card_width,float(label_width)*8.f+20.f);
        result.depths.push_back(depth);
        result.lanes.push_back(lane);
        result.columns=std::max(result.columns,depth+1);
        result.rows=std::max(result.rows,lane+1);
    }
    for(const auto& edge:graph.at("edges")) {
        const size_t from=edge.at("from"),to=edge.at("to");
        if(from<nodes.size() && to<nodes.size()) result.edges.push_back({from,to});
    }
    std::vector<size_t> outgoing(nodes.size()),incoming(nodes.size());
    for(const auto& edge:result.edges) {
        ++outgoing[edge.from];
        ++incoming[edge.to];
    }
    size_t max_degree=0;
    for(size_t i=0;i<nodes.size();++i)
        max_degree=std::max({max_degree,outgoing[i],incoming[i]});
    result.card_height=std::max(70.f,float(max_degree+1)*14.f);
    result.column_gap=std::max(80.f,float(max_degree+1)*16.f);
    const size_t long_edges=static_cast<size_t>(std::count_if(result.edges.begin(),result.edges.end(),
        [&](const auto& edge){return result.depths[edge.to]>result.depths[edge.from]+1;}));
    result.top+=float(long_edges)*10.f;
    return result;
}

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

inline std::vector<vizgfx::Vector2f> modelGraphTreeEdgePath(const ModelGraphTree& layout,
    const ModelGraphTree::Edge& edge) {
    const auto from=layout.position(edge.from),to=layout.position(edge.to);
    const float half=layout.card_width*.5f;
    const float source_right=from.x+half;
    const float target_left=to.x-half;
    auto is_long=[&](size_t i) {
        return layout.depths[layout.edges[i].to]>layout.depths[layout.edges[i].from]+1;
    };
    auto edge_less=[&](size_t a,size_t b,bool source) {
        if(is_long(a)!=is_long(b)) return is_long(a);
        const auto pa=layout.position(source?layout.edges[a].to:layout.edges[a].from);
        const auto pb=layout.position(source?layout.edges[b].to:layout.edges[b].from);
        return pa.y==pb.y ? (source?layout.edges[a].to<layout.edges[b].to:
            layout.edges[a].from<layout.edges[b].from) : pa.y<pb.y;
    };
    size_t edge_index=layout.edges.size();
    for(size_t i=0;i<layout.edges.size();++i)
        if(layout.edges[i].from==edge.from && layout.edges[i].to==edge.to) { edge_index=i;break; }
    if(edge_index==layout.edges.size()) return {};

    std::vector<size_t> outgoing,incoming;
    for(size_t i=0;i<layout.edges.size();++i) {
        if(layout.edges[i].from==edge.from)outgoing.push_back(i);
        if(layout.edges[i].to==edge.to)incoming.push_back(i);
    }
    std::sort(outgoing.begin(),outgoing.end(),[&](size_t a,size_t b){return edge_less(a,b,true);});
    std::sort(incoming.begin(),incoming.end(),[&](size_t a,size_t b){return edge_less(a,b,false);});
    const auto output=std::find(outgoing.begin(),outgoing.end(),edge_index);
    const auto input=std::find(incoming.begin(),incoming.end(),edge_index);
    const size_t output_rank=static_cast<size_t>(output-outgoing.begin());
    const size_t input_rank=static_cast<size_t>(input-incoming.begin());
    const float output_offset=(float(output_rank+1)/float(outgoing.size()+1)-.5f)*
        layout.card_height*.72f;
    const float input_offset=(float(input_rank+1)/float(incoming.size()+1)-.5f)*
        layout.card_height*.72f;
    const float source_y=from.y+output_offset;
    const float target_y=to.y+input_offset;
    if(layout.depths[edge.to]>layout.depths[edge.from]+1) {
        size_t route_rank=0;
        for(size_t i=0;i<edge_index;++i)
            if(layout.depths[layout.edges[i].to]>layout.depths[layout.edges[i].from]+1)++route_rank;
        const float track_y=layout.top-12.f-float(route_rank)*10.f;
        const float source_route=source_right+layout.column_gap*
            float(output_rank+1)/float(outgoing.size()+1);
        const float target_route=target_left-layout.column_gap*
            float(input_rank+1)/float(incoming.size()+1);
        return {{source_right+1.f,source_y},{source_route,source_y},{source_route,track_y},
            {target_route,track_y},{target_route,target_y},{target_left-2.f,target_y}};
    }
    if(outgoing.size()==1 && incoming.size()==1 && source_y==target_y &&
       layout.depths[edge.to]==layout.depths[edge.from]+1) {
        return {{source_right+1.f,source_y},{target_left-2.f,target_y}};
    }
    const float gap=layout.column_gap;
    const float source_route=source_right+gap*float(output_rank+1)/float(outgoing.size()+1);
    return {{source_right+1.f,source_y},{source_route,source_y},
        {source_route,target_y},{target_left-2.f,target_y}};
}

template<class Canvas> inline void drawModelGraphTreeEdges(Canvas& canvas,const ModelGraphTree& layout,
    size_t selected) {
    for(const auto& edge:layout.edges) {
        const auto path=modelGraphTreeEdgePath(layout,edge);
        const auto color=graphLaneColor(layout.lanes[edge.to],edge.from!=selected && edge.to!=selected);
        vizgfx::VertexArray vertices(vizgfx::PrimitiveType::LineStrip,path.size());
        for(size_t i=0;i<path.size();++i) {
            vertices[i].position=path[i];
            vertices[i].color=color;
        }
        canvas.draw(vertices);
        const auto& tip=path.back();
        vizgfx::VertexArray arrow(vizgfx::PrimitiveType::LineStrip,3);
        arrow[0].position=tip;arrow[1].position={tip.x-7.f,tip.y-4.f};
        arrow[2].position={tip.x-7.f,tip.y+4.f};
        arrow[0].color=arrow[1].color=arrow[2].color=color;
        canvas.draw(arrow);
    }
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
