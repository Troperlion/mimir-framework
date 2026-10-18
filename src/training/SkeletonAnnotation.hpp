#pragma once
#include "include/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace SkeletonAnnotation {
using nlohmann::json;
inline bool list(const json& v, bool strict) {
    return v.is_array() || (!strict && v.is_object() && v.empty());
}
inline void validate(const json& a, bool strict = false) {
    if (!a.contains("entities")) return;
    const auto& entities=a.at("entities");
    if (!list(entities,strict)) throw std::runtime_error("Dataset skeleton: entities must be a list");
    auto check=[](bool ok,const char* message) {if(!ok)throw std::runtime_error(std::string("Dataset skeleton: ")+message);};
    auto nonempty=[](const json& v){return v.is_string() && v.get<std::string>().find_first_not_of(" \t\r\n")!=std::string::npos;};
    std::unordered_set<std::string> ids;
    for(const auto& e:entities) if(e.is_object() && e.contains("id") && e["id"].is_string()) ids.insert(e["id"].get<std::string>());
    for(const auto& e:entities) {
        if(!e.is_object() || !e.contains("skeleton"))continue;
        const auto& s=e["skeleton"];
        check(s.is_object() && s.contains("name") && nonempty(s["name"]) && s.contains("points") && list(s["points"],strict),"invalid name or points");
        std::unordered_set<std::string> names;
        for(const auto& p:s["points"]) {
            check(p.is_object() && p.contains("name") && nonempty(p["name"]),"invalid point name");
            check(names.insert(p["name"].get<std::string>()).second,"duplicate point name");
            check(!strict || p.contains("position"),"missing point position");
            if(p.contains("position") && !p["position"].is_null()) {
                const auto& pos=p["position"];check(pos.is_array() && pos.size()==2,"position must be [x,y]");
                for(const auto& v:pos) {check(v.is_number(),"non-numeric position");const double n=v.get<double>();check(std::isfinite(n) && n>=0 && n<=1,"position outside [0,1]");}
            }
            if(p.contains("entities")) {
                check(list(p["entities"],strict),"linked entities must be a list");std::unordered_set<std::string> linked;
                for(const auto& id:p["entities"]) {check(id.is_string(),"invalid linked entity");const auto key=id.get<std::string>();check(ids.count(key) && linked.insert(key).second,"unknown or duplicate linked entity");}
            }
        }
        for(const auto& p:s["points"]) if(p.contains("parent") && !p["parent"].is_null()) {
            check(p["parent"].is_string(),"invalid parent");const auto parent=p["parent"].get<std::string>();check(names.count(parent) && parent!=p["name"].get<std::string>(),"unknown or self parent");
        }
        if(s.contains("edges")) {
            check(list(s["edges"],strict),"edges must be a list");
            for(const auto& edge:s["edges"]) {
                check(edge.is_object() && edge.contains("from") && edge["from"].is_string() && edge.contains("to") && edge["to"].is_string(),"invalid edge");
                const auto from=edge["from"].get<std::string>(),to=edge["to"].get<std::string>();check(names.count(from) && names.count(to) && from!=to,"unknown or self edge");
                check(!edge.contains("label") || edge["label"].is_string(),"invalid edge label");
            }
        }
    }
}

// Regional emphasis, not a pose prediction head. Coordinates refer to the whole
// image and are rasterized on the caller's latent/pixel grid.
inline std::vector<std::array<double,4>> regions(const json& a) {
    validate(a);
    std::vector<std::array<double,4>> result;
    if(!a.contains("entities"))return result;
    const auto& entities=a["entities"];
    auto patch=[&](double x,double y){result.push_back({std::max(0.,x-.025),std::max(0.,y-.025),std::min(1.,x+.025),std::min(1.,y+.025)});};
    auto segment=[&](const json& p,const json& q) {
        const double x=p[0].get<double>(),y=p[1].get<double>(),dx=q[0].get<double>()-x,dy=q[1].get<double>()-y;
        const int steps=std::max(1,int(std::ceil(std::hypot(dx,dy)/.025)));
        for(int i=0;i<=steps;++i)patch(x+dx*i/steps,y+dy*i/steps);
    };
    for(const auto& e:entities) {
        if(!e.contains("skeleton"))continue;
        const auto& s=e["skeleton"];std::unordered_map<std::string,json> placed;
        for(const auto& p:s["points"]) if(p.contains("position") && !p["position"].is_null()) {placed[p["name"].get<std::string>()]=p["position"];patch(p["position"][0].get<double>(),p["position"][1].get<double>());}
        auto connect=[&](const std::string& from,const std::string& to){if(placed.count(from) && placed.count(to))segment(placed.at(from),placed.at(to));};
        for(const auto& p:s["points"]) {
            const auto name=p["name"].get<std::string>();
            if(p.contains("parent") && p["parent"].is_string())connect(name,p["parent"].get<std::string>());
            if(placed.count(name) && p.contains("entities")) for(const auto& id:p["entities"]) for(const auto& target:entities) {
                if(target.value("id",std::string())!=id.get<std::string>() || target.value("id",std::string())==e.value("id",std::string()) || !target.contains("bbox"))continue;
                const auto& b=target["bbox"];
                if(b.is_array() && b.size()==4)segment(placed.at(name),json::array({(b[0].get<double>()+b[2].get<double>())*.5,(b[1].get<double>()+b[3].get<double>())*.5}));
            }
        }
        if(s.contains("edges"))for(const auto& edge:s["edges"])connect(edge["from"].get<std::string>(),edge["to"].get<std::string>());
    }
    return result;
}
inline std::string prompt(const json& a) {
    validate(a);if(!a.contains("entities"))return {};
    std::unordered_map<std::string,std::string> phrases;
    for(const auto& e:a["entities"])if(e.contains("id")){
        const auto id=e["id"].get<std::string>();auto phrase=e.value("phrase",std::string());phrases[id]=phrase.empty()?id:phrase;
    }
    std::ostringstream out;out<<std::fixed<<std::setprecision(3);
    for(const auto& e:a["entities"]) {
        if(!e.contains("skeleton"))continue;
        const auto& s=e["skeleton"];out<<"entity "<<e["id"].get<std::string>()<<" "<<phrases[e["id"].get<std::string>()]<<" skeleton "<<s["name"].get<std::string>()<<": ";
        for(const auto& p:s["points"]) {
            out<<p["name"].get<std::string>();
            if(p.contains("position") && !p["position"].is_null())out<<" at "<<p["position"][0].get<double>()<<","<<p["position"][1].get<double>();
            if(p.contains("parent") && p["parent"].is_string())out<<" connected to "<<p["parent"].get<std::string>();
            if(p.contains("entities"))for(const auto& id:p["entities"])out<<" linked to "<<phrases[id.get<std::string>()]<<" ["<<id.get<std::string>()<<"]";
            out<<". ";
        }
        if(s.contains("edges"))for(const auto& edge:s["edges"])out<<edge["from"].get<std::string>()<<" "<<edge.value("label",std::string())<<" connected to "<<edge["to"].get<std::string>()<<". ";
    }
    return out.str();
}
} // namespace SkeletonAnnotation
