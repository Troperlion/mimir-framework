#include "runtimes/FiniteCheck.hpp"
#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <iostream>
int main() {
    using Mimir::RuntimeChecks::allFinite;
    if(!allFinite(nullptr,0)||allFinite(nullptr,1)) throw std::runtime_error("Empty check failed");
    std::vector<float> values(2057,0.0f);
    for(size_t position : {size_t(0),size_t(7),size_t(1023),size_t(1024),size_t(2056)})
        for(uint32_t bits : {0u,0x80000000u,1u,0x007fffffu,0x7f7fffffu,0x7f800000u,0xff800000u,0x7fc00001u,0x7f800001u}) {
            std::memcpy(&values[position],&bits,sizeof(bits));
            if(allFinite(values.data(),values.size()) != std::isfinite(values[position]))
                throw std::runtime_error("Classification mismatch");
            values[position]=0;
        }
    std::mt19937 rng(42);
    for(int trial=0;trial<10000;++trial) {
        const uint32_t bits=rng(); float value; std::memcpy(&value,&bits,sizeof(bits));
        if(allFinite(&value,1)!=std::isfinite(value)) throw std::runtime_error("Random bits mismatch");
    }
    std::cout << "FINITE_CHECK_OK\n";
}
