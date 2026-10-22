#pragma once
// Host buffers only stage inputs/results. All tensor arithmetic is dispatched
// to the selected OpenCL/Vulkan device; no CPU arithmetic fallback is provided.
#include "PortableDevice.hpp"
#include "PortableWorkBudget.hpp"
#include "runtimes/cuda/CudaOperationGeometry.hpp"
#include <cstdlib>
#include <cstring>
#include <map>
#include <limits>
#include <cstdint>
#include <stdexcept>
namespace PortableStaging {
struct Context {
    std::unique_ptr<Device> device;
    std::map<uintptr_t,size_t> allocations;
    std::map<uintptr_t,int> offsets;
    bool failed=false;
    size_t elements=1;
    std::vector<std::vector<int>> pending;
};
inline thread_local std::map<std::string,Context> contexts;
inline thread_local Context* current=nullptr;
inline std::vector<int> kernelWriteArguments(int id);
// Preserve device intermediates until the first host observation. A batch has
// stable arena offsets even when a wrapper allocates more scratch between kernels.
inline bool flush(Context& context){
    if(context.pending.empty()){context.offsets.clear();context.elements=1;return !context.failed;}
    bool ok=!context.failed && bool(context.device);
    try{
        if(ok){
            for(const auto& a:context.allocations)if(!context.offsets.count(a.first)){
                if(a.second/4>size_t(64*1024*1024)-context.elements)throw std::runtime_error("GPU arena budget exceeded");
                context.offsets[a.first]=static_cast<int>(context.elements);context.elements+=a.second/4;
            }
            std::vector<float> arena(context.elements,0.f);
            for(const auto& a:context.allocations)std::memcpy(arena.data()+context.offsets.at(a.first),reinterpret_cast<const void*>(a.first),a.second);
            std::map<int,uintptr_t> locations;
            for(const auto& entry:context.offsets)locations.emplace(entry.second,entry.first);
            std::map<uintptr_t,MemoryRange> written;
            for(const auto& command:context.pending)for(int argument:kernelWriteArguments(command[0])){
                if(size_t(argument)>=command.size())throw std::runtime_error("Missing GPU pointer argument");
                const int offset=command[argument];if(!offset)continue;
                auto at=locations.upper_bound(offset);
                if(at==locations.begin())throw std::runtime_error("Invalid GPU write pointer");
                --at;const auto size=context.allocations.at(at->second)/sizeof(float);
                if(size_t(offset-at->first)>=size)throw std::runtime_error("Invalid GPU write range");
                written[at->second]={size_t(at->first),size};
            }
            std::vector<MemoryRange> downloads;
            for(const auto& entry:written)downloads.push_back(entry.second);
            ok=context.device->runBatch(arena,context.pending,downloads);
            if(ok)for(const auto& entry:written)std::memcpy(reinterpret_cast<void*>(entry.first),arena.data()+entry.second.offset,entry.second.count*sizeof(float));
        }
    }catch(...){ok=false;}
    context.pending.clear();context.offsets.clear();context.elements=1;context.failed=!ok;return ok;
}
inline bool flush(){return current && flush(*current);}
inline bool allocationContains(const void* pointer,size_t bytes){
    if(!current || !pointer)return bytes==0;
    const auto address=reinterpret_cast<uintptr_t>(pointer);
    auto at=current->allocations.upper_bound(address);
    if(at==current->allocations.begin())return false;
    --at;
    const auto offset=address-at->first;
    return offset<=at->second && bytes<=at->second-offset;
}
inline bool scheduledPointer(const void* pointer){
    if(!current || current->pending.empty() || !allocationContains(pointer,1))return false;
    auto at=current->allocations.upper_bound(reinterpret_cast<uintptr_t>(pointer));
    return at!=current->allocations.begin() && current->offsets.count((--at)->first);
}
struct DeviceScope {
    Context* previous;
    DeviceScope(const char* backend,int index=0):previous(current) {
        if(current)flush(*current);
        current=&contexts[std::string(backend)+":"+std::to_string(index)];
        if(!current->device)current->device=createDevice(backend,index);
        current->failed=(current==previous && current->failed) || !current->device;
    }
    ~DeviceScope(){if(current)flush(*current);current=previous;}
};
constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1,cudaMemcpyDeviceToDevice=2;
constexpr int CUBLAS_STATUS_SUCCESS=0,CUBLAS_OP_N=0,CUBLAS_OP_T=1;
using cublasHandle_t=void*;
inline int cudaGetDeviceCount(int* n){*n=current && current->device?1:0;return *n?0:1;}
inline int cudaSetDevice(int){return current && current->device?0:1;}
inline int cublasCreate(void** p){*p=current;return current && current->device?0:1;}
inline int cublasDestroy(void*){return 0;}
inline int cublasSetStream(void*,void*){return 0;}
template<class T> int cudaMalloc(T** p,size_t bytes){
    if(!current || !current->device || !bytes || bytes%4 || bytes/4>INT_MAX)return 1;
    *p=static_cast<T*>(std::calloc(1,bytes));if(!*p)return 1;
    current->allocations[reinterpret_cast<uintptr_t>(*p)]=bytes;return 0;
}
inline int cudaFree(void* p){if(p){for(auto& entry:contexts){if(entry.second.allocations.count(reinterpret_cast<uintptr_t>(p)))flush(entry.second);entry.second.allocations.erase(reinterpret_cast<uintptr_t>(p));}std::free(p);}return 0;}
inline int cudaMemcpy(void* dest,const void* source,size_t bytes,int direction){
    if(direction<cudaMemcpyHostToDevice || direction>cudaMemcpyDeviceToDevice ||
        (bytes && (!dest || !source)) ||
        (bytes && direction!=cudaMemcpyDeviceToHost && !allocationContains(dest,bytes)) ||
        (bytes && direction!=cudaMemcpyHostToDevice && !allocationContains(source,bytes))){
        if(current)current->failed=true;
        return 1;
    }
    if(current && ((direction!=cudaMemcpyHostToDevice) || scheduledPointer(dest)) && !flush(*current))return 1;
    if(bytes)std::memmove(dest,source,bytes);
    return 0;
}
inline int cudaMemset(void* p,int value,size_t bytes){
    if(bytes && !allocationContains(p,bytes)){if(current)current->failed=true;return 1;}
    if(scheduledPointer(p) && !flush())return 1;
    if(bytes)std::memset(p,value,bytes);
    return 0;
}
inline int cudaGetLastError(){return !current || current->failed?1:0;}
inline int cudaDeviceSynchronize(){return flush()?cudaSuccess:1;}
inline void pack(std::vector<int>& p,float v){int bits;std::memcpy(&bits,&v,4);p.push_back(bits);}
inline void pack(std::vector<int>& p,bool v){p.push_back(v?1:0);}
inline void pack(std::vector<int>& p,int v){p.push_back(v);}
inline void pack(std::vector<int>& p,size_t v){if(v>INT_MAX)throw std::runtime_error("GPU argument overflow");p.push_back(static_cast<int>(v));}
inline void pack(std::vector<int>& p,LayerType v){p.push_back(static_cast<int>(v));}
inline void pack(std::vector<int>& p,std::nullptr_t){p.push_back(0);}
template<class T> void pack(std::vector<int>& p,T* ptr){
    if(!ptr){p.push_back(0);return;}
    uintptr_t addr=reinterpret_cast<uintptr_t>(ptr);auto it=current->allocations.upper_bound(addr);
    if(it==current->allocations.begin())throw std::runtime_error("Invalid GPU staging pointer");
    --it;
    if(addr-it->first>=it->second || (addr-it->first)%4)throw std::runtime_error("Invalid GPU staging offset");
    p.push_back(current->offsets.at(it->first)+static_cast<int>((addr-it->first)/4));
}
#include "PortableKernelArguments.hpp"
template<class... Args> void launch(const char* kernel,int blocks,int threads,const Args&... args){
    if(!current || !current->device || current->failed)return;
    try {
        if(blocks<=0 || threads<=0 || 1LL*blocks*threads>INT_MAX)throw std::runtime_error("Invalid GPU grid");
        for(const auto& a:current->allocations)if(!current->offsets.count(a.first)){
            if(a.second/4>size_t(64*1024*1024)-current->elements)throw std::runtime_error("GPU arena budget exceeded");
            current->offsets[a.first]=static_cast<int>(current->elements);current->elements+=a.second/4;
        }
        int id=kernelId(kernel);if(id<0)throw std::runtime_error("Unknown GPU reference kernel");
        std::vector<int> parameters{id,threads,blocks};(pack(parameters,args),...);
        if(!parallelDispatchFits(current->elements,parameters) && !referenceDispatchFits(current->elements,parameters)){current->failed=true;return;}
        current->pending.push_back(std::move(parameters));
    }catch(...){current->failed=true;}
}
inline int cublasSgemm(void*,int ta,int tb,int m,int n,int k,const float* alpha,const float* a,int lda,const float* b,int ldb,const float* beta,float* c,int ldc){
    launch("portable_gemm",1,1,a,b,c,m,n,k,lda,ldb,ldc,ta!=0,tb!=0,*alpha,*beta);return cudaGetLastError();
}
inline int cublasSger(void*,int m,int n,const float* alpha,const float* x,int sx,const float* y,int sy,float* a,int lda){
    launch("portable_ger",1,1,x,y,a,m,n,sx,sy,lda,*alpha);return cudaGetLastError();
}
}
