#include "Visualizer.hpp"
#include <iostream>
#include <thread>
#include <chrono>

int main(int argc, char** argv) {
    Visualizer viz(json{{"visualization", {{"enabled",true},{"window_width",1200},
        {"window_height",800},{"window_title","Mimir Script UI test"},{"fps_limit",30}}}});
    viz.setLossLogEnabled(false);
    for (const auto& bad : {
        json{{"panels", {{{"id",7}}}}},
        json{{"panels", {{{"id",1.5}}}}},
        json{{"panels", {{{"id",4294967296LL}}}}},
        json{{"controls", {{{"id","duplicate"}}, {{"id","duplicate"}}}}},
        json{{"controls", {{{"id","bad"},{"w",-1}}}}},
        json{{"events", "yes"}}}) {
        bool rejected = false;
        try { viz.configureScript(bad); } catch (const std::exception&) { rejected=true; }
        if (!rejected) return 2;
    }
    if (!viz.initialize()) return 3;
    // Exercise producer/consumer thread ownership, including queue coalescing.
    std::thread producer([&] {
        viz.configureScript({{"panels", {{{"id",2},{"title","Lua preview"},{"x",20},{"y",140},{"w",900},{"h",450}}}},
            {"controls", {{{"id","apply"},{"label","Apply edit"},{"x",20},{"y",20},{"w",180},{"h",40}}}},
            {"help","Test script controls"},{"events",true}});
        viz.configureScript({{"help","Callbacks execute on the Lua thread"}});
    });
    producer.join();
    bool configured=false, clicked=false;
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(argc>1?std::stoi(argv[1]):10);
    while(viz.isOpen() && std::chrono::steady_clock::now()<deadline) {
        viz.processEvents();
        viz.update();
        for(const auto& event:viz.pollScriptEvents()) {
            if(event.value("type","")=="configured") configured=true;
            if(event.value("type","")=="click" && event.value("id","")=="apply") clicked=true;
            std::cout<<"SCRIPT_EVENT "<<event.dump()<<std::endl;
        }
    }
    viz.shutdown();
    std::cout<<"SCRIPT_UI_RESULT "<<configured<<" "<<clicked<<std::endl;
    return configured && clicked ? 0 : 4;
}
