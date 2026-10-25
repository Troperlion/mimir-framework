#pragma once
#include "include/json.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace Mimir::Viz {
// Terminal-only presentation of the immutable model snapshot; never reads layers.
struct TerminalModelGraphView {
    bool visible=false, follow=true;
    size_t selected=0;
    int pan_x=0,pan_y=0;
    void move(int delta,size_t count) {
        if(!count){selected=0;return;}
        selected=static_cast<size_t>(std::clamp<long long>(static_cast<long long>(selected)+delta,0,count-1));
        follow=true;
    }
    static std::string safe(std::string value) {
        for(char& c:value)if(static_cast<unsigned char>(c)<32 || c==127)c='?';
        return value;
    }
    std::vector<std::string> lines(const nlohmann::json* snapshot,int width,int height) {
        const int columns=std::max(1,width-4),body=std::max(1,height-13);
        std::vector<std::string> result;
        result.push_back(std::string(columns,'='));
        result.push_back("GRAPHE DU MODELE / PLANNER  [G / Esc] fermer");
        result.push_back("Haut/bas/Tab: layer | PgUp/PgDn: page | Gauche/droite: vue | R: origine");
        if(!snapshot) {
            result.push_back("Aucun modele attache : en attente d'un instantane.");return result;
        }
        const auto& graph=*snapshot;const auto& nodes=graph.at("nodes");
        if(!nodes.empty())selected=std::min(selected,nodes.size()-1);else selected=0;
        result.push_back("Planner: "+graph.value("mode",std::string("inconnu"))+
            " | "+(graph.value("training",false)?"training":"inference")+
            " | fusion: "+(graph.value("fusion",false)?"on":"off")+
            " | reuse: "+(graph.value("buffer_reuse",false)?"on":"off")+
            " | layers: "+std::to_string(nodes.size()));
        if(graph.value("planned",false)) {
            const auto& stats=graph.at("plan").at("stats");
            result.push_back("Estimations octets: tenseurs="+stats.at("tensor_logical_bytes").dump()+
                " pic="+stats.at("tensor_physical_peak").dump()+" reuse="+stats.at("buffer_reuse_saved").dump()+
                " transferts="+stats.at("transfer_bytes").dump()+" scratch="+stats.at("scratch_peak").dump());
        } else result.push_back("Topologie seule : plan indisponible.");
        result.push_back("Arbre nodal ; fleches distinctes ; > layer selectionne ; placements PREVUS (pas une trace GPU).");

        size_t card_width_size=32;
        for(size_t i=0;i<nodes.size();++i) {
            const auto& node=nodes[i];
            const std::string name=std::to_string(i)+" "+node.value("name",std::string());
            const std::string details=node.value("type",std::string())+" / "+
                node.value("runtime",std::string());
            card_width_size=std::max({card_width_size,name.size()+2,details.size()+2});
        }
        const int card_width=static_cast<int>(std::min(card_width_size,
            static_cast<size_t>(std::numeric_limits<int>::max()-1)));
        struct Edge { size_t from,to; };
        std::vector<Edge> edges;
        std::vector<size_t> depths,lanes;
        depths.reserve(nodes.size());lanes.reserve(nodes.size());
        size_t max_degree=0,max_depth=0;
        for(const auto& node:nodes) {
            const size_t depth=node.value("depth",size_t(0));
            const size_t lane=node.value("lane",size_t(0));
            depths.push_back(depth);lanes.push_back(lane);
            max_depth=std::max(max_depth,depth);
        }
        for(const auto& edge:graph.at("edges")) {
            const size_t from=edge.at("from"),to=edge.at("to");
            if(from>=nodes.size() || to>=nodes.size())continue;
            edges.push_back({from,to});
        }
        std::vector<std::vector<size_t>> outgoing(nodes.size()),incoming(nodes.size());
        for(size_t i=0;i<edges.size();++i) {
            outgoing[edges[i].from].push_back(i);
            incoming[edges[i].to].push_back(i);
        }
        for(size_t i=0;i<nodes.size();++i)
            max_degree=std::max({max_degree,outgoing[i].size(),incoming[i].size()});
        const int card_height=std::max(4,static_cast<int>(max_degree)+2);
        const int row_step=card_height+2;
        size_t long_edge_count=0;
        for(const auto& edge:edges)
            if(depths[edge.to]>depths[edge.from]+1)++long_edge_count;
        const int graph_top=2+static_cast<int>(long_edge_count)*2;

        std::vector<size_t> edges_per_depth(max_depth+1,0);
        for(const auto& edge:edges)
            if(depths[edge.to]==depths[edge.from]+1)++edges_per_depth[depths[edge.from]];
        const int gap_width=static_cast<int>(std::max<size_t>(3,
            *std::max_element(edges_per_depth.begin(),edges_per_depth.end())+2));
        const int column_step=card_width+gap_width;
        std::vector<std::pair<int,int>> pos;
        pos.reserve(nodes.size());
        int max_x=0,max_y=0;
        for(size_t i=0;i<nodes.size();++i) {
            const int x=2+static_cast<int>(depths[i])*column_step;
            const int y=graph_top+static_cast<int>(lanes[i])*row_step;
            pos.push_back({x,y});
            max_x=std::max(max_x,x+card_width);
            max_y=std::max(max_y,y+card_height);
        }
        std::vector<size_t> edge_channel(edges.size(),std::numeric_limits<size_t>::max());
        for(size_t depth=0;depth<edges_per_depth.size();++depth) {
            std::vector<size_t> group;
            for(size_t i=0;i<edges.size();++i)
                if(depths[edges[i].to]==depths[edges[i].from]+1 &&
                   depths[edges[i].from]==depth)group.push_back(i);
            std::sort(group.begin(),group.end(),[&](size_t a,size_t b) {
                const auto [ax,ay]=pos[edges[a].from];
                const auto [bx,by]=pos[edges[b].from];
                const auto [atx,aty]=pos[edges[a].to];
                const auto [btx,bty]=pos[edges[b].to];
                if(ay!=by)return ay<by;
                if(aty!=bty)return aty<bty;
                if(ax!=bx)return ax<bx;
                return atx<btx;
            });
            for(size_t rank=0;rank<group.size();++rank)
                edge_channel[group[rank]]=group.size()-rank-1;
        }
        for(size_t i=0;i<nodes.size();++i) {
            std::sort(outgoing[i].begin(),outgoing[i].end(),[&](size_t a,size_t b) {
                return pos[edges[a].to].second<pos[edges[b].to].second;
            });
            std::sort(incoming[i].begin(),incoming[i].end(),[&](size_t a,size_t b) {
                return pos[edges[a].from].second<pos[edges[b].from].second;
            });
        }
        if(follow && !nodes.empty()) {
            const auto [x,y]=pos[selected];
            if(x<pan_x || x+card_width>pan_x+columns)pan_x=std::max(0,x-columns/2+card_width/2);
            if(y<pan_y || y+card_height>pan_y+body)pan_y=std::max(0,y-body/2+card_height/2);
            follow=false;
        }
        pan_x=std::clamp(pan_x,0,std::max(0,max_x-columns));
        pan_y=std::clamp(pan_y,0,std::max(0,max_y-body));
        std::vector<std::string> canvas(body,std::string(columns,' '));
        auto put=[&](int x,int y,char c) {
            x-=pan_x;y-=pan_y;
            if(x>=0 && x<columns && y>=0 && y<body)canvas[y][x]=c;
        };
        auto write=[&](int x,int y,std::string text,int limit) {
            text=safe(std::move(text));
            for(size_t i=0;i<text.size() && i<static_cast<size_t>(limit);++i)put(x+int(i),y,text[i]);
        };
        auto port_row=[&](size_t rank,size_t count) {
            return 1+static_cast<int>((rank+1)*static_cast<size_t>(card_height-2)/(count+1));
        };
        auto segment=[&](int x1,int y1,int x2,int y2) {
            if(y1==y2) {
                const int first=std::max(std::min(x1,x2),pan_x);
                const int last=std::min(std::max(x1,x2),pan_x+columns-1);
                for(int x=first;x<=last;++x)put(x,y1,'-');
            } else if(x1==x2) {
                const int first=std::max(std::min(y1,y2),pan_y);
                const int last=std::min(std::max(y1,y2),pan_y+body-1);
                for(int y=first;y<=last;++y)put(x1,y,'|');
            }
        };
        for(size_t i=0;i<edges.size();++i) {
            const auto& edge=edges[i];
            const auto [sx,sy]=pos[edge.from];const auto [tx,ty]=pos[edge.to];
            const size_t out_rank=static_cast<size_t>(
                std::find(outgoing[edge.from].begin(),outgoing[edge.from].end(),i)-outgoing[edge.from].begin());
            const size_t in_rank=static_cast<size_t>(
                std::find(incoming[edge.to].begin(),incoming[edge.to].end(),i)-incoming[edge.to].begin());
            const int source_y=sy+port_row(out_rank,outgoing[edge.from].size());
            const int target_y=ty+port_row(in_rank,incoming[edge.to].size());
            const int source_x=sx+card_width-1,target_x=tx;
            if(outgoing[edge.from].size()==1 && incoming[edge.to].size()==1 &&
               depths[edge.to]==depths[edge.from]+1 && source_y==target_y) {
                segment(source_x+1,source_y,target_x-1,target_y);
                put(source_x,source_y,'>');
                put(target_x-1,target_y,'>');
                continue;
            }
            if(depths[edge.to]>depths[edge.from]+1) {
                const size_t skip_rank=static_cast<size_t>(std::count_if(edges.begin(),
                    edges.begin()+static_cast<std::ptrdiff_t>(i),[&](const Edge& previous) {
                        return depths[previous.to]>depths[previous.from]+1;
                    }));
                const int route_y=1+static_cast<int>(skip_rank)*2;
                segment(source_x+1,source_y,source_x+1,route_y);
                segment(source_x+1,route_y,target_x-1,route_y);
                segment(target_x-1,route_y,target_x-1,target_y);
                put(source_x,source_y,'>');
                put(target_x-1,target_y,'v');
            } else {
                const int route_x=sx+card_width+1+static_cast<int>(edge_channel[i]);
                segment(source_x+1,source_y,route_x,source_y);
                segment(route_x,source_y,route_x,target_y);
                segment(route_x,target_y,target_x-1,target_y);
                put(source_x,source_y,'>');
                put(target_x-1,target_y,'v');
            }
        }
        for(size_t i=0;i<nodes.size();++i) {
            const auto [x,y]=pos[i];const auto& n=nodes[i];
            write(x,y,(i==selected?">":"+")+std::string(card_width-2,'-')+"+",card_width);
            for(int row=1;row<card_height-1;++row)
                write(x,y+row,"|"+std::string(card_width-2,' ')+"|",card_width);
            write(x+1,y+1,std::to_string(i)+" "+n.value("name",std::string()),card_width-2);
            if(card_height>3)
                write(x+1,y+2,n.value("type",std::string())+" / "+
                    n.value("runtime",std::string()),card_width-2);
            write(x,y+card_height-1,"+"+std::string(card_width-2,'-')+"+",card_width);
        }
        if(nodes.empty())canvas[0]="Le modele ne contient aucun layer.";
        result.insert(result.end(),canvas.begin(),canvas.end());
        result.push_back(std::string(columns,'-'));
        if(!nodes.empty()) {
            const auto& n=nodes[selected];
            result.push_back(std::to_string(selected)+" "+n.at("name").get<std::string>()+" | "+n.at("type").get<std::string>()+" | params="+n.at("parameters").dump());
            result.push_back("Entrees "+n.at("inputs").dump()+" -> "+n.at("output").get<std::string>());
            result.push_back("Runtime: "+n.at("runtime").get<std::string>()+" | capacite: "+n.at("capability").get<std::string>());
            result.push_back("Fusion graphe="+std::string(n.value("graph_fused",false)?"oui":"non")+
                " kernel="+(n.value("kernel_fused",false)?std::string("oui"):"non")+
                " skip="+(n.value("skip",false)?std::string("oui"):"non")+
                " vers="+(n.contains("fused_into")?n.at("fused_into").dump():"aucune"));
            result.push_back("Shape="+(n.contains("shape")?n.at("shape").dump():"inconnue")+
                " octets="+(n.contains("bytes")?n.at("bytes").dump():"inconnus")+
                " scratch="+std::to_string(n.value("scratch_bytes",size_t(0))));
        }
        result.push_back(std::string(columns,'='));
        for(auto& line:result)line=safe(std::move(line));
        return result;
    }
};
}
