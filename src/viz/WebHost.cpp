#include "Host.hpp"
#include "Ipc.hpp"
#include "renderers/WebScene.hpp"
#include "backends/WebPage.hpp"
#include "Scene.hpp"
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <atomic>
#include <cmath>
#include <charconv>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
namespace mimir::viz {
namespace {
bool parseEvent(const std::string& query, Event& e) {
    if(query.size()>256) return false;
    std::istringstream input(query);
    const char* names[]={"t=","x=","y=","v=","m="};
    int32_t* values[]={&e.type,&e.x,&e.y,&e.value,&e.modifiers};
    std::string part;
    for(int i=0;i<5;++i) {
        if(!std::getline(input,part,'&') || part.rfind(names[i],0)!=0) return false;
        auto result=std::from_chars(part.data()+2,part.data()+part.size(),*values[i]);
        if(result.ec!=std::errc{} || result.ptr!=part.data()+part.size()) return false;
    }
    if(!std::getline(input,part) || part.rfind("d=",0)!=0) return false;
    std::istringstream delta(part.substr(2));
    delta.imbue(std::locale::classic());
    delta >> std::noskipws >> e.delta;
    return !delta.fail() && delta.eof();
}
}
class WebHost final : public Host {
    int server_=-1; std::atomic<bool> stopped_{false}; std::thread worker_;
    std::mutex mutex_; Frame frame_; std::deque<Event> events_; std::string token_;
    void reply(int fd,const std::string& code,const std::string& type,const void* bytes,size_t count,unsigned cursor=0) {
        std::string h="HTTP/1.1 "+code+"\r\nContent-Type: "+type+"\r\nContent-Length: "+std::to_string(count)+
            "\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\nX-Frame-Options: DENY\r\nConnection: close\r\nX-Viz-Cursor: "+std::to_string(cursor)+"\r\n\r\n";
        if(transfer(fd,h.data(),h.size(),true) && count) transfer(fd,const_cast<void*>(bytes),count,true);
    }
    void serve(int fd) {
        timeval timeout{0,250000}; setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout)); setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
        std::string request; char buf[2048];
        // Bound both bytes and wall time, including a client trickling bytes.
        auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(500);
        while(request.find("\r\n\r\n")==std::string::npos) {
            auto n=recv(fd,buf,sizeof(buf),0); if(n<=0)return; request.append(buf,n);
            if(request.size()>8192 || std::chrono::steady_clock::now()>deadline)return;
        }
        std::istringstream first(request); std::string method,path,version; first>>method>>path>>version;
        const std::string root="/"+token_;
        if(method=="GET" && (path==root || path==root+"/")) { reply(fd,"200 OK","text/html; charset=utf-8",web::page,std::char_traits<char>::length(web::page));return; }
        if(method=="GET" && path==root+"/renderer.js") {
            reply(fd,"200 OK","text/javascript; charset=utf-8",web::renderer,std::char_traits<char>::length(web::renderer));return;
        }
        if(method=="GET" && path==root+"/frame") {
            Frame f; {std::lock_guard<std::mutex> lock(mutex_);f=frame_;}
            if(f.scene.empty()){reply(fd,"204 No Content","application/json",nullptr,0);return;}
            // Embed the already serialized scene without reparsing its texture assets.
            const std::string body="{\"width\":"+std::to_string(f.width)+",\"height\":"+std::to_string(f.height)+",\"scene\":"+f.scene+"}";
            reply(fd,"200 OK","application/json",body.data(),body.size(),f.cursor);return;
        }
        if(method=="POST" && path.rfind(root+"/event?",0)==0) {
            Event e;
            const auto query=path.substr(root.size()+7);
            if(!parseEvent(query,e) ||
               e.type<Close || e.type>Blur || e.x< -32768 || e.x>32768 || e.y< -32768 || e.y>32768 || !std::isfinite(e.delta) || std::abs(e.delta)>10000 ||
               e.value<0 || e.value>0x10ffff || e.modifiers<0 || e.modifiers>15) {reply(fd,"400 Bad Request","text/plain",nullptr,0);return;}
            bool full=false;
            {std::lock_guard<std::mutex> lock(mutex_);
             if(e.type==Move && !events_.empty() && events_.back().type==Move) events_.back()=e;
             else if(events_.size()<1024)events_.push_back(e);else full=true;}
            reply(fd,full?"429 Too Many Requests":"204 No Content","text/plain",nullptr,0);return;
        }
        reply(fd,"404 Not Found","text/plain",nullptr,0);
    }
public:
    explicit WebHost(const std::string&) {
        unsigned long port=0;
        if(const char* s=std::getenv("MIMIR_VIZ_WEB_PORT")) { char* end=nullptr;port=strtoul(s,&end,10);if(!*s||*end||port>65535)throw std::runtime_error("Invalid MIMIR_VIZ_WEB_PORT"); }
        std::random_device rng;std::ostringstream token;token<<std::hex<<std::setfill('0');for(int i=0;i<4;++i)token<<std::setw(8)<<rng();token_=token.str();
        server_=socket(AF_INET,SOCK_STREAM|SOCK_CLOEXEC,0);if(server_<0)throw std::runtime_error("Viz Web socket failed");
        sockaddr_in address{}; address.sin_family=AF_INET;address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);address.sin_port=htons(port);
        if(bind(server_,reinterpret_cast<sockaddr*>(&address),sizeof(address)) || listen(server_,8)) {::close(server_);throw std::runtime_error("Cannot bind Viz Web loopback port");}
        socklen_t len=sizeof(address);getsockname(server_,reinterpret_cast<sockaddr*>(&address),&len);
        std::cerr<<"[Viz WEB] http://127.0.0.1:"<<ntohs(address.sin_port)<<"/"<<token_<<"/\n";
        worker_=std::thread([this]{while(!stopped_){pollfd p{server_,POLLIN,0};if(::poll(&p,1,100)<=0)continue;int fd=accept4(server_,nullptr,nullptr,SOCK_CLOEXEC);if(fd<0)continue;serve(fd);::close(fd);}});
    }
    ~WebHost() override { stopped_=true;shutdown(server_,SHUT_RDWR);if(worker_.joinable())worker_.join();::close(server_); }
    void present(Frame f) override {if(f.scene.size()>maxSceneBytes)throw std::runtime_error("Viz scene exceeds transport limit");std::lock_guard<std::mutex> lock(mutex_);frame_=std::move(f);}
    std::optional<Event> poll() override {std::lock_guard<std::mutex> lock(mutex_);if(events_.empty())return {};auto e=events_.front();events_.pop_front();return e;}
};
std::unique_ptr<Host> makeHost(const std::string& title) {return std::make_unique<WebHost>(title);}
}
