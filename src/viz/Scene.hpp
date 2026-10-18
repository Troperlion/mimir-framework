#pragma once
#include "../include/json.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace mimir::viz {
using Scene = nlohmann::json;
// Only texture assets are raster images. Geometry, clipping and text stay native.
inline std::string base64(const uint8_t* data, size_t count) {
    static constexpr char alphabet[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out; out.reserve((count+2)/3*4);
    for(size_t i=0;i<count;i+=3) {
        const uint32_t v=uint32_t(data[i])<<16 | (i+1<count?uint32_t(data[i+1])<<8:0) | (i+2<count?data[i+2]:0);
        out+=alphabet[(v>>18)&63]; out+=alphabet[(v>>12)&63];
        out+=i+1<count?alphabet[(v>>6)&63]:'='; out+=i+2<count?alphabet[v&63]:'=';
    }
    return out;
}
inline std::vector<uint8_t> unbase64(const std::string& text) {
    std::vector<uint8_t> out; out.reserve(text.size()/4*3); uint32_t value=0; int bits=0;
    for(unsigned char c:text) {
        int n=c>='A'&&c<='Z'?c-'A':c>='a'&&c<='z'?c-'a'+26:c>='0'&&c<='9'?c-'0'+52:c=='+'?62:c=='/'?63:-1;
        if(n<0)break;
        value=(value<<6)|uint32_t(n);bits+=6;
        if(bits>=8){bits-=8;out.push_back(uint8_t(value>>bits));}
    }
    return out;
}
}
