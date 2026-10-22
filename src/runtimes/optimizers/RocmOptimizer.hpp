#pragma once
#include "OptimizerDispatch.hpp"
#if defined(ENABLE_ROCM) && __has_include(<hip/hiprtc.h>)
#include "OptimizerShaderData.hpp"
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <dlfcn.h>
#include <map>
#include <memory>
#include <iostream>
namespace MimirRocmOptimizer {
struct Compiler {
    void* library=nullptr;hipModule_t module=nullptr;hipFunction_t kernel=nullptr;
    decltype(&hiprtcCreateProgram) create=nullptr;
    decltype(&hiprtcCompileProgram) compile=nullptr;
    decltype(&hiprtcGetCodeSize) size=nullptr;
    decltype(&hiprtcGetCode) code=nullptr;
    decltype(&hiprtcDestroyProgram) destroy=nullptr;
    ~Compiler(){if(module)(void)hipModuleUnload(module);if(library)dlclose(library);}
    bool initialize(const char* source=MimirOptimizerKernels::hip,const char* kernelName="mimir_optimizer") {
        library=dlopen("libhiprtc.so",RTLD_NOW|RTLD_LOCAL);if(!library)return false;
#define SYMBOL(field,name) field=reinterpret_cast<decltype(field)>(dlsym(library,name));if(!field)return false
        SYMBOL(create,"hiprtcCreateProgram");SYMBOL(compile,"hiprtcCompileProgram");
        SYMBOL(size,"hiprtcGetCodeSize");SYMBOL(code,"hiprtcGetCode");SYMBOL(destroy,"hiprtcDestroyProgram");
#undef SYMBOL
        hiprtcProgram program=nullptr;
        if(create(&program,source,"mimir_optimizer.hip",0,nullptr,nullptr)!=HIPRTC_SUCCESS)return false;
        const char* options[]={"--std=c++17"};
        const auto status=compile(program,1,options);
        size_t bytes=0;std::vector<char> binary;
        bool ok=status==HIPRTC_SUCCESS && size(program,&bytes)==HIPRTC_SUCCESS;
        if(ok){binary.resize(bytes);ok=code(program,binary.data())==HIPRTC_SUCCESS;}
        destroy(&program);
        return ok && hipModuleLoadData(&module,binary.data())==hipSuccess &&
            hipModuleGetFunction(&kernel,module,kernelName)==hipSuccess;
    }
};
struct Buffer {void* ptr=nullptr;~Buffer(){if(ptr)(void)hipFree(ptr);}bool alloc(size_t n){return hipMalloc(&ptr,n)==hipSuccess;}};
inline bool run(const RuntimeOptimizerRequest& request,int index) {
    if((!optimizerDeviceSizeFits(request) || !optimizerRequestValid(request)) || hipSetDevice(index)!=hipSuccess)return false;
    static thread_local std::map<int,std::unique_ptr<Compiler>> compilers;
    auto& compiler=compilers[index];
    if(!compiler){auto candidate=std::make_unique<Compiler>();if(!candidate->initialize())return false;compiler=std::move(candidate);}
    auto memory=optimizerPack(request, true);auto args=optimizerParameters(request, true);
    Buffer data,params;
    if(!data.alloc(memory.size()*4) || !params.alloc(args.size()*4) ||
       hipMemcpy(data.ptr,memory.data(),optimizerInputCount(args)*sizeof(float),hipMemcpyHostToDevice)!=hipSuccess ||
       hipMemcpy(params.ptr,args.data(),args.size()*4,hipMemcpyHostToDevice)!=hipSuccess)return false;
    std::vector<int> passes{0};if(args[2]==4 || args[2]==8){passes.push_back(1);passes.push_back(3);passes.push_back(2);}
    for(int pass:passes) {
        void* parameters[]={&data.ptr,&params.ptr,&pass};
        if(hipModuleLaunchKernel(compiler->kernel,pass==3?1:(request.count+255)/256,1,1,256,1,1,0,nullptr,parameters,nullptr)!=hipSuccess)return false;
    }
    const size_t result_count = optimizerResultCount(args);
    if(hipMemcpy(memory.data(),data.ptr,result_count*sizeof(float),hipMemcpyDeviceToHost)!=hipSuccess)return false;
    return optimizerCommit(request,memory,true);
}
}
#endif
