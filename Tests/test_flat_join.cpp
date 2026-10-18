#include "Layers.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/cpu/FlatJoinKernels.hpp"
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <iostream>
static void require(bool ok){if(!ok)throw std::runtime_error("Flat join failed");}
int main() {
    CpuRuntime cpu;cpu.initialize(RuntimeConfig{});
    const uint32_t bits[]={0,0x80000000u,0x7f800000u,0x7fc00001u,1u};
    std::vector<float> a(5);std::memcpy(a.data(),bits,sizeof(bits));
    std::vector<float> b=a,empty,shorter{2,3};
    std::vector<std::vector<float>> output;
    for(const char* type:{"Concat","Stack"})for(int axis:{-1,0,1,2}) {
        Layer l("join",type,0);l.concat_axis=axis;l.stack_axis=axis;
        require(cpu.forwardLayer({&a,&b,&a},output,l,false));
        require(output.size()==1&&output[0].size()==a.size()*3);
        for(size_t i=0;i<3;++i)require(std::memcmp(output[0].data()+i*a.size(),a.data(),sizeof(bits))==0);
        require(!cpu.forwardLayer({&a,nullptr},output,l,false));
        require(!cpu.forwardLayer({&a},output,l,false));
        require(cpu.forwardLayer({&empty,&empty},output,l,false)&&output[0].empty());
        require(cpu.forwardLayer({&a,&shorter},output,l,false)==(l.type_enum==LayerType::Concat));
    }
    std::vector<float> out{99};
    require(!RuntimeFlatJoin::forward({&a,&shorter},out,true)&&out==std::vector<float>{99});
    require(RuntimeFlatJoin::forward({&empty,&a,&shorter,&empty},out,false));
    require(out.size()==7&&out[5]==2&&out[6]==3);
    require(!RuntimeFlatJoin::forward({},out,false));
    // Safe even if the destination is also one of the borrowed inputs.
    require(RuntimeFlatJoin::forward({&shorter,&shorter},shorter,true));
    require(shorter==std::vector<float>({2,3,2,3}));
    std::cout<<"FLAT_JOIN_OK\n";
}
