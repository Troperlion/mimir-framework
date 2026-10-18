#include "runtimes/cpu/RuntimeAllocator.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
static void require(bool value) { if(!value)throw std::runtime_error("scratch allocator regression"); }
int main() {
    auto& guard=MemoryGuard::instance();guard.reset();guard.setLimit(1u<<20);
    {
        RuntimeAllocator allocator(guard,1);
        for(size_t bytes:{size_t(0),size_t(1),size_t(3),size_t(4),size_t(7),size_t(65)}) {
            auto buffer=allocator.allocate_buffer(bytes,"zeroed");
            const auto* p=reinterpret_cast<const unsigned char*>(buffer.data());
            for(size_t i=0;i<bytes;++i)require(p[i]==0);
            std::memset(buffer.data(),0x5a,bytes); // tail must have real allocated storage
        }
        auto buffer=allocator.get_scratchpad_uninitialized(65,"work");
        std::memset(buffer.data(),0x3c,65);const auto* address=buffer.data();
        allocator.return_scratchpad(std::move(buffer));
        auto reused=allocator.get_scratchpad_uninitialized(33,"work");
        require(reused.data()==address&&reused.size_bytes()==65);
        for(size_t i=0;i<65;++i)require(reinterpret_cast<unsigned char*>(reused.data())[i]==0x3c);
        require(guard.getCurrentBytes()==65);
    }
    {
        RuntimeAllocator allocator(guard,1);
        const std::string long_tag(100,'t');
        auto original=allocator.allocate_buffer(16,long_tag);
        const char* tag_data=original.tag().data();
        auto moved=std::move(original);
        require(moved.tag().data()==tag_data&&original.data()==nullptr);
        auto assigned=allocator.allocate_buffer(8,"discarded");
        assigned=std::move(moved);
        require(assigned.tag().data()==tag_data&&moved.data()==nullptr);
        auto tensor=allocator.allocate_tensor({2,3},"float32",long_tag);
        const int* shape_data=tensor.descriptor().shape.data();
        auto moved_tensor=std::move(tensor);
        require(moved_tensor.descriptor().shape.data()==shape_data);
        auto raw=allocator.allocate_raw_tensor({2,3},"uint8",long_tag);
        shape_data=raw.descriptor().shape.data();
        auto moved_raw=std::move(raw);
        require(moved_raw.descriptor().shape.data()==shape_data);
    }
    require(guard.getCurrentBytes()==0);
    {
        RuntimeAllocator allocator(guard,1);bool rejected=false;
        try{auto buffer=allocator.allocate_buffer_uninitialized((1u<<20)+1,"too big");}
        catch(const std::runtime_error&){rejected=true;}
        require(rejected&&guard.getCurrentBytes()==0);
    }
    std::cout<<"ALLOCATOR_SCRATCH_OK\n";
}
