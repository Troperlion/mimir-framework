#include "runtimes/fpga/FpgaIo.hpp"
#include "runtimes/fpga/FpgaSession.hpp"
#include "runtimes/fpga/FpgaRuntime.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace {
std::recursive_mutex io_mutex;
bool io_failed=false;
uint8_t active_operation=0;
uint64_t transferred=0;
uint32_t stream_crc=0xffffffffU;
void update(const uint8_t* bytes,size_t size) {
    if(!active_operation) return;
    transferred+=size;
    for(size_t i=0;i<size;++i) {
        stream_crc^=bytes[i];
        for(int bit=0;bit<8;++bit) stream_crc=(stream_crc>>1)^((stream_crc&1)?0xedb88320U:0);
    }
}
FpgaRuntime* engine() {
    const char* enabled=std::getenv("MIMIR_FPGA_IO");
    if(enabled && std::string(enabled)=="0") return nullptr;
    if(io_failed) throw std::runtime_error("FPGA I/O session failed; restart or explicitly disable MIMIR_FPGA_IO");
    Mimir::initializeFpgaSession();
    auto* runtime=Mimir::fpgaSessionRuntime();
    return runtime && (runtime->capabilities().operations & Mimir::FpgaProtocol::BufferedIoConversion)?runtime:nullptr;
}
void transfer(FpgaRuntime& runtime,uint8_t mode,const uint8_t* bytes,size_t size,std::vector<uint8_t>& result) {
    if(!runtime.transferBlock(mode,bytes,size,result)) {
        io_failed=true;
        std::cerr<<"[FPGA] transfer_failed mode="<<unsigned(mode)<<" bytes="<<size<<"; integrity required, no CPU fallback\n";
        throw std::runtime_error("FPGA: transfer/conversion integrity check failed (no silent CPU fallback)");
    }
    update(bytes,size);
}
}

bool Mimir::fpgaEncode16(uint16_t* output,const float* input,size_t count,bool bf16) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex);
    auto* runtime=engine();if(!runtime) return false;
    std::vector<uint8_t> result;
    for(size_t offset=0;offset<count;offset+=64) {
        const size_t elements=std::min<size_t>(64,count-offset);
        transfer(*runtime,bf16?1:3,reinterpret_cast<const uint8_t*>(input+offset),elements*4,result);
        std::memcpy(output+offset,result.data(),result.size());
    }
    std::cerr<<"[FPGA] checkpoint_convert F32->"<<(bf16?"BF16":"F16")<<" elements="<<count<<" native=YES\n";
    return true;
}
bool Mimir::fpgaDecode16(float* output,const uint16_t* input,size_t count,bool bf16) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex);
    auto* runtime=engine();if(!runtime) return false;
    std::vector<uint8_t> result;
    for(size_t offset=0;offset<count;offset+=128) {
        const size_t elements=std::min<size_t>(128,count-offset);
        transfer(*runtime,bf16?2:4,reinterpret_cast<const uint8_t*>(input+offset),elements*2,result);
        std::memcpy(output+offset,result.data(),result.size());
    }
    std::cerr<<"[FPGA] checkpoint_convert "<<(bf16?"BF16":"F16")<<"->F32 elements="<<count<<" native=YES\n";
    return true;
}
void Mimir::fpgaStageDataset(std::vector<uint8_t>& bytes) {
    std::lock_guard<std::recursive_mutex> lock(io_mutex);
    auto* runtime=engine();if(!runtime || bytes.empty()) return;
    std::vector<uint8_t> result;
    for(size_t offset=0;offset<bytes.size();offset+=256) {
        const size_t size=std::min<size_t>(256,bytes.size()-offset);
        transfer(*runtime,0,bytes.data()+offset,size,result);
        if(!std::equal(result.begin(),result.end(),bytes.begin()+offset))
            throw std::runtime_error("FPGA dataset BRAM readback mismatch");
        std::copy(result.begin(),result.end(),bytes.begin()+offset);
    }
    std::cerr<<"[FPGA] dataset_buffer bytes="<<bytes.size()<<" storage=BRAM chunks="<<(bytes.size()+255)/256<<" verified=YES\n";
}
struct Mimir::FpgaCheckpointTransaction::State {
    std::unique_lock<std::recursive_mutex> lock{io_mutex};
    FpgaRuntime* runtime=nullptr;
    uint8_t operation=0;
    bool done=false;
};
Mimir::FpgaCheckpointTransaction::FpgaCheckpointTransaction(bool saving):state_(new State) {
    state_->runtime=engine();
    if(!state_->runtime || active_operation) { state_->runtime=nullptr;return; }
    state_->operation=saving?2:1;
    if(!state_->runtime->beginCheckpoint(state_->operation)) throw std::runtime_error("FPGA checkpoint controller begin failed");
    active_operation=state_->operation;transferred=0;stream_crc=0xffffffffU;
}
Mimir::FpgaCheckpointTransaction::~FpgaCheckpointTransaction() {
    if(state_->runtime && !state_->done) {
        (void)state_->runtime->beginCheckpoint(0);
        active_operation=0;
    }
}
void Mimir::FpgaCheckpointTransaction::verifyFiles(const std::string& path) {
    if(!state_->runtime) return;
    namespace fs=std::filesystem;
    std::vector<fs::path> files;
    if(fs::is_regular_file(path)) files.emplace_back(path);
    else if(fs::is_directory(path)) {
        for(const auto& entry:fs::recursive_directory_iterator(path)) if(entry.is_regular_file()) files.push_back(entry.path());
        std::sort(files.begin(),files.end());
    } else throw std::runtime_error("FPGA checkpoint path not found: "+path);
    std::vector<uint8_t> input(256),result;
    for(const auto& file:files) {
        std::ifstream stream(file,std::ios::binary);
        if(!stream) throw std::runtime_error("FPGA checkpoint cannot read: "+file.string());
        while(stream) {
            stream.read(reinterpret_cast<char*>(input.data()),input.size());
            const size_t count=static_cast<size_t>(stream.gcount());
            if(!count) break;
            transfer(*state_->runtime,5,input.data(),count,result);
            if(!std::equal(result.begin(),result.end(),input.begin())) throw std::runtime_error("FPGA checkpoint readback mismatch");
        }
        if(stream.bad()) throw std::runtime_error("FPGA checkpoint read error: "+file.string());
    }
}
void Mimir::FpgaCheckpointTransaction::finish() {
    if(!state_->runtime) return;
    uint8_t operation=0;uint64_t count=0;uint32_t crc=0;
    if(!state_->runtime->endCheckpoint(operation,count,crc) || operation!=state_->operation || count!=transferred || crc!=~stream_crc)
        throw std::runtime_error("FPGA checkpoint stream byte count/CRC mismatch");
    if(!state_->runtime->beginCheckpoint(0)) throw std::runtime_error("FPGA checkpoint controller reset failed");
    state_->done=true;active_operation=0;
    std::cerr<<"[FPGA] checkpoint_"<<(operation==2?"save":"load")<<" bytes="<<count<<" crc32=OK controller=FPGA\n";
}
