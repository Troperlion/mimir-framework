#pragma once
#include "include/json.hpp"
#include <algorithm>
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
        result.push_back("Connexions nommees ; > layer selectionne ; placements PREVUS (pas une trace GPU).");
        std::vector<std::pair<int,int>> pos;
        int max_x=0,max_y=0;
        for(const auto& n:nodes) {
            const int x=n.at("lane").get<int>()*32,y=n.at("depth").get<int>()*4;
            pos.push_back({x,y});max_x=std::max(max_x,x+30);max_y=std::max(max_y,y+3);
        }
        if(follow && !nodes.empty()) {
            const auto [x,y]=pos[selected];
            if(x<pan_x || x+30>pan_x+columns)pan_x=std::max(0,x-columns/2+15);
            if(y<pan_y || y+3>pan_y+body)pan_y=std::max(0,y-body/2+1);
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
        for(const auto& edge:graph.at("edges")) {
            const size_t a=edge.at("from"),b=edge.at("to");
            if(a>=pos.size() || b>=pos.size())continue;
            const auto [ax,ay]=pos[a];const auto [bx,by]=pos[b];
            const int source=ax+15,target=bx+15,mid=by-1;
            // Clip loops before rasterizing even very long skip connections.
            for(int y=std::max(ay+3,pan_y);y<=std::min(mid,pan_y+body-1);++y)put(source,y,'|');
            for(int x=std::max(std::min(source,target),pan_x);x<=std::min(std::max(source,target),pan_x+columns-1);++x)put(x,mid,'-');
            put(source,mid,'+');put(target,mid,'v');
            if(a==selected || b==selected)write(std::min(source,target)+2,mid,edge.at("tensor"),24);
        }
        for(size_t i=0;i<nodes.size();++i) {
            const auto [x,y]=pos[i];const auto& n=nodes[i];
            write(x,y,(i==selected?">":"+")+std::string(28,'-')+"+",30);
            write(x,y+1,"|"+std::string(28,' ')+"|",30);
            write(x+1,y+1,std::to_string(i)+" "+n.at("name").get<std::string>(),28);
            write(x,y+2,"+"+std::string(28,'-')+"+",30);
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
