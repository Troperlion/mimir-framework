#pragma once
#include "Planning/Planner.hpp"
#include "include/json.hpp"
#include <unordered_map>

namespace Mimir::Viz {
// Called on the model thread only. Tensor names are resolved in execution order,
// so in-place writes connect to the previous producer rather than to themselves.
inline nlohmann::json modelGraphSnapshot(const std::vector<Layer>& layers,
    const Planning::ExecutionPlan* plan, const std::string& mode,
    bool training, bool fusion, bool reuse) {
    using Json = nlohmann::json;
    Json graph={{"mode",mode},{"planned",plan!=nullptr},{"training",training},
        {"fusion",fusion},{"buffer_reuse",reuse},{"nodes",Json::array()},
        {"edges",Json::array()},{"external_inputs",Json::array()}};
    if(plan) graph["plan"]=Json::parse(Planning::dump_execution_plan_json(*plan));
    std::unordered_map<std::string,size_t> producers;
    std::vector<size_t> depths(layers.size(),0), lanes(layers.size()+1,0);
    for(size_t i=0;i<layers.size();++i) {
        const auto& layer=layers[i];
        for(const auto& input:Planning::planner_inputs_for(layer)) {
            auto found=producers.find(input);
            if(found!=producers.end()) {
                graph["edges"].push_back({{"from",found->second},{"to",i},{"tensor",input}});
                depths[i]=std::max(depths[i],depths[found->second]+1);
            } else graph["external_inputs"].push_back({{"to",i},{"tensor",input}});
        }
        Json node={{"index",i},{"name",layer.name},{"type",layer.type},
            {"inputs",Planning::planner_inputs_for(layer)},
            {"output",Planning::planner_output_name_for(layer)},
            {"parameters",layer.getWeightsSize()},{"depth",depths[i]},
            {"lane",lanes[depths[i]]++},{"runtime","non planifie"},
            {"capability","inconnue"},{"graph_fused",false},{"kernel_fused",false},{"skip",false}};
        if(plan && i<plan->layers.size()) {
            const auto& p=plan->layers[i];
            // Legacy plans do not run the placement pass; don't label their
            // default CPU fields as actual runtime decisions.
            if(mode!="legacy") {
                node["runtime"]=Planning::runtime_kind_name(p.runtime);
                node["capability"]=Planning::capability_level_name(p.capability);
            } else node["runtime"]="dispatcher (legacy)";
            node["graph_fused"]=p.graph_fused;node["kernel_fused"]=p.kernel_fused;
            node["skip"]=p.skip;node["scratch_bytes"]=p.scratch_bytes;
            if(p.fused_into)node["fused_into"]=*p.fused_into;
            const auto tensor=plan->tensors.find(p.output);
            if(tensor!=plan->tensors.end()) {
                node["shape"]=tensor->second.shape;node["bytes"]=tensor->second.bytes;
                node["generation"]=tensor->second.generation;
            }
        }
        graph["nodes"].push_back(std::move(node));
        producers[Planning::planner_output_name_for(layer)]=i;
    }
    return graph;
}
}
