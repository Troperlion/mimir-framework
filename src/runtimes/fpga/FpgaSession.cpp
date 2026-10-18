#include "runtimes/fpga/FpgaSession.hpp"
#include "runtimes/fpga/FpgaRuntime.hpp"
#include <cstdlib>
#include <iostream>
#include <mutex>

namespace {
struct Session {
    FpgaRuntime runtime;
    std::once_flag once;
    nlohmann::json status={{"detected",false},{"connected",false},
        {"layer_acceleration",false},{"status","not_checked"}};
    ~Session() { runtime.shutdown(); }
};
Session& session() { static Session instance; return instance; }
}

void Mimir::initializeFpgaSession() {
    auto& s=session();
    std::call_once(s.once,[&] {
        const auto cfg=RuntimeConfig::fromEnv("FPGA");
        if(cfg.disabled) { s.status["status"]="disabled"; return; }
        const char* root=std::getenv("MIMIR_FPGA_SYSFS_ROOT");
        const auto device=FpgaRuntime::detectDevice(root && *root?root:"/sys/bus/usb/devices");
        if(!device) {
            s.status["status"]="not_detected";
            std::cerr<<"[FPGA] not_detected; dataset/checkpoint I/O stays on host\n";
            return;
        }
        s.status["detected"]=true;
        s.status["device"]=device->tty_path; s.status["serial"]=device->serial;
        if(!s.runtime.initialize(cfg)) {
            s.status["status"]="handshake_failed";
            std::cerr<<"[FPGA] detected device="<<device->tty_path
                     <<" connection=FAILED (check tty permissions, serial port ownership and Mimir firmware)\n";
            return;
        }
        const auto& caps=s.runtime.capabilities();
        s.status["connected"]=true; s.status["status"]="connected_diagnostics_only";
        s.status["protocol"]=std::to_string(caps.protocol_major)+"."+std::to_string(caps.protocol_minor);
        s.status["operations"]=caps.operations;
        s.status["max_vector_elements"]=caps.max_vector_elements;
        const bool io=(caps.operations & Mimir::FpgaProtocol::BufferedIoConversion)!=0;
        s.status["dataset_buffer"]=io; s.status["checkpoint_controller"]=io;
        s.status["tensor_conversion"]=io; s.status["buffer_input_bytes"]=io?256:0;
        if(io) s.status["status"]="buffer_and_checkpoint_io_ready";
        std::cerr<<"[FPGA] dataset_buffer="<<(io?"BRAM/256-byte-chunks":"unsupported_by_firmware")
                 <<" checkpoint_controller="<<(io?"CRC32/byte-count":"unsupported_by_firmware")
                 <<" tensor_conversion="<<(io?"F32<->F16/BF16":"unsupported_by_firmware")<<"\n";
        if((caps.operations & Mimir::FpgaProtocol::DotProductInt8) && caps.max_vector_elements>=4) {
            const int8_t a[]={-128,127,-1,1}, b[]={1,-1,127,127};
            int32_t result=0;
            const bool passed=s.runtime.int8DotProduct(a,b,4,result) && result==-255;
            s.status["int8_self_test"]=passed?"PASS":"FAIL";
            if(!passed) {
                s.runtime.shutdown(); s.status["connected"]=false; s.status["status"]="self_test_failed";
            }
        }
        std::cerr<<"[FPGA] detected device="<<device->tty_path
                 <<" protocol="<<s.status["protocol"].get<std::string>()
                 <<" connection="<<(s.runtime.isInitialized()?"OK":"FAILED")
                 <<" int8_self_test="<<s.status.value("int8_self_test",std::string("not_run"))
                 <<"\n[FPGA] role=dataset_buffer/checkpoint_io; neural_layers=CPU/GPU\n";
    });
}

FpgaRuntime* Mimir::fpgaSessionRuntime() {
    auto& s=session();
    return s.runtime.isInitialized()?&s.runtime:nullptr;
}
nlohmann::json Mimir::fpgaSessionStatus() {
    initializeFpgaSession();
    return session().status;
}
