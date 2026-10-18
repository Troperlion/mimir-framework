#pragma once
#include "OptimizerDispatch.hpp"
#include "runtimes/portable/PortableDevice.hpp"
#include <map>
#include <memory>
#include <string>
inline bool portableOptimizer(const RuntimeOptimizerRequest& request,const char* backend,int index) {
    if((!optimizerDeviceSizeFits(request) || !optimizerRequestValid(request)))return false;
    static thread_local std::map<std::pair<std::string,int>,std::unique_ptr<PortableStaging::Device>> contexts;
    auto& device=contexts[{backend,index}];
    if(!device)device=PortableStaging::createOptimizerDevice(backend,index);
    if(!device)return false;
    auto memory=optimizerPack(request);const auto arguments=optimizerParameters(request);
    if(!device->run(memory,arguments)){device.reset();return false;}
    return optimizerCommit(request,memory);
}
