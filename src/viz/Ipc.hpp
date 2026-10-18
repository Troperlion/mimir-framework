#pragma once
#include "Host.hpp"
#include <sys/socket.h>
#include <cerrno>
namespace mimir::viz {
inline bool transfer(int fd, void* data, size_t size, bool write) {
    auto* p=static_cast<char*>(data);
    while(size) {
        auto n=write ? ::send(fd,p,size,MSG_NOSIGNAL) : ::recv(fd,p,size,0);
        if(n<0 && errno==EINTR) continue;
        if(n<=0) return false;
        p+=n; size-=size_t(n);
    }
    return true;
}
}
