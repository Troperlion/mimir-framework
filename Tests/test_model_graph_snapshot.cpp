#include "viz/ModelGraph.hpp"
#include "LiveModelConfig.hpp"
#include <iostream>
#include <stdexcept>
#define CHECK(c) do { if(!(c)) throw std::runtime_error(#c); } while(0)
int main() {
    std::vector<Layer> layers;
    layers.emplace_back("first","Linear",6);
    layers.back().inputs={"x"};layers.back().output="a";
    layers.emplace_back("rewrite","ReLU",0);
    layers.back().inputs={"a"};layers.back().output="a";
    layers.emplace_back("branch","Add",0);
    layers.back().inputs={"a","x"};layers.back().output="b";
    auto graph=Mimir::Viz::modelGraphSnapshot(layers,nullptr,"desactive",false,false,false);
    CHECK(graph["nodes"].size()==3);CHECK(graph["edges"].size()==2);
    CHECK(graph["edges"][0]["from"]==0 && graph["edges"][0]["to"]==1);
    CHECK(graph["edges"][1]["from"]==1 && graph["edges"][1]["to"]==2);
    CHECK(graph["nodes"][2]["depth"]==2);CHECK(!graph["planned"].get<bool>());
    auto plan=Mimir::Planning::build_execution_plan_global(layers,false,Mimir::Planning::PlannerMode::Static,false,false);
    graph=Mimir::Viz::modelGraphSnapshot(layers,&plan,"static",false,false,false);
    CHECK(graph["planned"].get<bool>());CHECK(graph.contains("plan"));
    CHECK(graph["nodes"][0]["runtime"]=="CPU");
    auto legacy=Mimir::Viz::modelGraphSnapshot(layers,&plan,"legacy",false,false,false);
    CHECK(legacy["nodes"][0]["runtime"]=="dispatcher (legacy)");
    layers.emplace_back("parallel","Identity",0);
    layers.back().inputs={"x"};layers.back().output="parallel";
    layers.emplace_back("join","Add",0);
    layers.back().inputs={"b","parallel"};layers.back().output="result";
    auto branch=Mimir::Viz::modelGraphSnapshot(layers,nullptr,"desactive",false,false,false);
    CHECK(branch["nodes"][3]["depth"]==0 && branch["nodes"][3]["lane"]==1);
    CHECK(branch["nodes"][4]["depth"]==3 && branch["edges"].size()==4);
    auto empty=Mimir::Viz::modelGraphSnapshot({},nullptr,"en attente",false,false,false);
    CHECK(empty["nodes"].empty() && empty["edges"].empty());
    LiveModelConfig config;config.publishModelGraph(graph);auto old=config.modelGraph();
    config.publishModelGraph(legacy);CHECK((*old)["mode"]=="static");
    CHECK((*config.modelGraph())["mode"]=="legacy");
    config.requestModelGraph(true);CHECK(config.modelGraphRequested());
    std::cout<<"MODEL_GRAPH_SNAPSHOT_OK\n";
}
