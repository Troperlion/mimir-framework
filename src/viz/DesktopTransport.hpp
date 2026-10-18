#pragma once
#include "Ipc.hpp"
#include <atomic>
#include <mutex>
#include <thread>
#include <unistd.h>
using namespace mimir::viz;
namespace mimir::viz::desktop {
std::mutex frameMutex;
Frame latest;
bool updated=false;
std::atomic<bool> ended{false};
void emitEvent(Event e) { if(!transfer(STDOUT_FILENO,&e,sizeof(e),true)) ended=true; }
void readFrames() {
    while(!ended) {
        FrameHeader h;
        if(!transfer(STDIN_FILENO,&h,sizeof(h),false) || (!validSize(h.width,h.height) || h.sceneBytes>maxSceneBytes)) break;
        Frame f; f.width=h.width; f.height=h.height; f.cursor=h.cursor; f.scene.resize(h.sceneBytes);
        if(!transfer(STDIN_FILENO,f.scene.data(),f.scene.size(),false)) break;
        std::lock_guard<std::mutex> lock(frameMutex); latest=std::move(f); updated=true;
    }
    ended=true;
}
bool takeFrame(Frame& f) { std::lock_guard<std::mutex> lock(frameMutex); if(!updated) return false; f=std::move(latest); updated=false; return true; }
}
