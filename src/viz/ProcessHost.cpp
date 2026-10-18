#include "Host.hpp"
#include "Ipc.hpp"
#include "HostEnvironment.hpp"
#include <spawn.h>
#include <sys/wait.h>
#include <poll.h>
#include <unistd.h>
#include <signal.h>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <thread>
extern char** environ;
namespace mimir::viz {
class ProcessHost final : public Host {
    int fd_=-1; pid_t pid_=-1;
    std::thread writer_; std::mutex mutex_; std::condition_variable wake_;
    Frame frame_; bool pending_=false; std::atomic<bool> stopped_{false};
    std::array<char,sizeof(Event)> incoming_{}; size_t used_=0;
public:
    explicit ProcessHost(const std::string& title) {
        auto environment = desktopHostEnvironment(environ);
        std::vector<char*> childEnvironment;
        childEnvironment.reserve(environment.size() + 1);
        for (auto& variable : environment) childEnvironment.push_back(variable.data());
        childEnvironment.push_back(nullptr);
        int pair[2];
        if(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,pair)) throw std::runtime_error("Viz socketpair failed");
        std::string path;
        try { path=(std::filesystem::read_symlink("/proc/self/exe").parent_path()/"mimir_viz_host").string(); }
        catch (...) { ::close(pair[0]); ::close(pair[1]); throw; }
        if (const char* overridePath=std::getenv("MIMIR_VIZ_HOST")) path=overridePath;
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions,pair[1],STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions,pair[1],STDOUT_FILENO);
        char* args[]={path.data(),const_cast<char*>(title.c_str()),nullptr};
        int error=posix_spawn(&pid_,path.c_str(),&actions,nullptr,args,childEnvironment.data());
        posix_spawn_file_actions_destroy(&actions); ::close(pair[1]);
        if(error) { ::close(pair[0]); throw std::runtime_error("Cannot launch Viz host: " + std::string(strerror(error))); }
        fd_=pair[0];
        pollfd ready{fd_,POLLIN,0}; Event handshake;
        if(::poll(&ready,1,10000)<=0 || !transfer(fd_,&handshake,sizeof(handshake),false) || handshake.type!=sceneProtocolHandshake) {
            ::close(fd_); kill(pid_,SIGTERM); waitpid(pid_,nullptr,0);
            throw std::runtime_error("Viz desktop host did not initialize");
        }
        writer_=std::thread([this] {
            for (;;) {
                Frame f;
                { std::unique_lock<std::mutex> lock(mutex_); wake_.wait(lock,[this]{return stopped_ || pending_;});
                  if(stopped_) return;
                  f=std::move(frame_); pending_=false; }
                FrameHeader h{f.width,f.height,f.cursor,uint32_t(f.scene.size())};
                if(f.scene.size()>maxSceneBytes) { stopped_=true; return; }
                if(!transfer(fd_,&h,sizeof(h),true) || !transfer(fd_,f.scene.data(),f.scene.size(),true)) { stopped_=true; return; }
            }
        });
    }
    ~ProcessHost() override {
        stopped_=true; wake_.notify_all(); shutdown(fd_,SHUT_RDWR);
        if(writer_.joinable()) writer_.join();
        ::close(fd_);
        // EOF normally terminates the GUI; bound cleanup if its event loop is stuck.
        for(int i=0;i<50;++i) {
            if(waitpid(pid_,nullptr,WNOHANG)==pid_) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        kill(pid_,SIGTERM); waitpid(pid_,nullptr,0);
    }
    void present(Frame f) override { std::lock_guard<std::mutex> lock(mutex_); frame_=std::move(f); pending_=true; wake_.notify_one(); }
    std::optional<Event> poll() override {
        if(stopped_) return Event{Close};
        auto n=recv(fd_,incoming_.data()+used_,incoming_.size()-used_,MSG_DONTWAIT);
        if(n==0 || (n<0 && errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR)) { stopped_=true; return Event{Close}; }
        if(n>0) used_+=size_t(n);
        if(used_==incoming_.size()) { Event e; memcpy(&e,incoming_.data(),sizeof(e)); used_=0; return e; }
        return {};
    }
};
std::unique_ptr<Host> makeHost(const std::string& title) { return std::make_unique<ProcessHost>(title); }
}
