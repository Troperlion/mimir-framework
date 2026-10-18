#pragma once
#include "Ipc.hpp"
#include "Scene.hpp"
#include <atomic>
#include <mutex>
#include <thread>
#include <unistd.h>
using namespace mimir::viz;
namespace mimir::viz::desktop {
std::mutex frameMutex;
Frame latest;
Scene latestScene;
bool updated=false;
std::atomic<bool> ended{false};
void emitEvent(Event e) { if(!transfer(STDOUT_FILENO,&e,sizeof(e),true)) ended=true; }
void readFrames() {
    while(!ended) {
        FrameHeader h;
        if(!transfer(STDIN_FILENO,&h,sizeof(h),false) || (!validSize(h.width,h.height) || h.sceneBytes>maxSceneBytes)) break;
        Frame f; f.width=h.width; f.height=h.height; f.cursor=h.cursor; f.scene.resize(h.sceneBytes);
        if(!transfer(STDIN_FILENO,f.scene.data(),f.scene.size(),false)) break;
        // JSON parsing can be substantial for tensor-heavy scenes. Keep it off
        // the toolkit thread so input remains serviceable during reception.
        Scene scene=Scene::parse(f.scene,nullptr,false);
        if(scene.is_discarded()) break;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            std::swap(latest,f); latestScene.swap(scene); updated=true;
        }
    }
    ended=true;
}
bool takeFrame(Frame& f,Scene& scene) {
    std::lock_guard<std::mutex> lock(frameMutex);
    if(!updated) return false;
    std::swap(f,latest); scene.swap(latestScene); updated=false;
    return true;
}
}
