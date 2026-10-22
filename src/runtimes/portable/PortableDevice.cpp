#include "PortableWorkBudget.hpp"
#include "PortableDevice.hpp"
#include "PortableShaderData.hpp"
#include "runtimes/losses/LossShaderData.hpp"
#include "runtimes/losses/LossDispatch.hpp"
#include "runtimes/optimizers/OptimizerShaderData.hpp"
#include "runtimes/optimizers/OptimizerDispatch.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <map>
#ifdef ENABLE_OPENCL
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif
#ifdef ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif
namespace PortableStaging {
#ifdef ENABLE_OPENCL
class OpenCLDevice final : public Device {
    bool optimizer_=false,loss_=false;
    cl_context context=nullptr;
    cl_command_queue queue=nullptr;
    cl_program program=nullptr;
    cl_kernel kernel=nullptr;
    cl_device_id selectedDevice=nullptr;
    std::map<int,std::pair<cl_program,cl_kernel>> specialized;
    cl_mem cachedData=nullptr,cachedParams=nullptr;
    size_t dataCapacity=0,paramCapacity=0;
    std::vector<std::pair<cl_mem,size_t>> batchParams;
public:
    ~OpenCLDevice() override {
        for(const auto& entry:specialized){clReleaseKernel(entry.second.second);clReleaseProgram(entry.second.first);}
        for(const auto& parameter:batchParams)if(parameter.first)clReleaseMemObject(parameter.first);
        if(cachedData)clReleaseMemObject(cachedData);
        if(cachedParams)clReleaseMemObject(cachedParams);
        if(kernel)clReleaseKernel(kernel);
        if(program)clReleaseProgram(program);
        if(queue)clReleaseCommandQueue(queue);
        if(context)clReleaseContext(context);
    }
    bool initialize(int index,bool optimizer=false,bool loss=false) {
        loss_=loss;optimizer_=optimizer || loss;
        cl_uint count=0;if(clGetPlatformIDs(0,nullptr,&count)!=CL_SUCCESS || !count)return false;
        std::vector<cl_platform_id> platforms(count);if(clGetPlatformIDs(count,platforms.data(),nullptr)!=CL_SUCCESS)return false;
        std::vector<cl_device_id> devices;
        for(auto platform:platforms){cl_uint n=0;if(clGetDeviceIDs(platform,(optimizer_?CL_DEVICE_TYPE_GPU:CL_DEVICE_TYPE_ALL),0,nullptr,&n)!=CL_SUCCESS)continue;
            std::vector<cl_device_id> found(n);if(clGetDeviceIDs(platform,(optimizer_?CL_DEVICE_TYPE_GPU:CL_DEVICE_TYPE_ALL),n,found.data(),nullptr)==CL_SUCCESS)devices.insert(devices.end(),found.begin(),found.end());}
        if(devices.empty())return false;
        if(index<0 || static_cast<size_t>(index)>=devices.size()){if(optimizer_)return false;index=0;}
        cl_device_id device=devices[index];selectedDevice=device;cl_int error=0;
        if(std::getenv("MIMIR_ACCEL_VERBOSE")) {
            char name[256]={};cl_device_type type=0;
            clGetDeviceInfo(device,CL_DEVICE_NAME,sizeof(name),name,nullptr);
            clGetDeviceInfo(device,CL_DEVICE_TYPE,sizeof(type),&type,nullptr);
            std::cerr<<"[OPENCL reference] device="<<name<<" gpu="<<bool(type&CL_DEVICE_TYPE_GPU)<<'\n';
        }
        context=clCreateContext(nullptr,1,&device,nullptr,nullptr,&error);if(error!=CL_SUCCESS)return false;
        queue=clCreateCommandQueue(context,device,0,&error);if(error!=CL_SUCCESS)return false;
        const char* source=loss_?MimirLossKernels::opencl:optimizer_?MimirOptimizerKernels::opencl:referenceOpenCL;program=clCreateProgramWithSource(context,1,&source,nullptr,&error);if(error!=CL_SUCCESS)return false;
        error=clBuildProgram(program,1,&device,"-cl-std=CL1.2",nullptr,nullptr);
        if(error!=CL_SUCCESS){size_t n=0;clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,nullptr,&n);std::string log(n,'\0');clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,n,log.data(),nullptr);std::cerr<<"[OPENCL reference] "<<log<<'\n';return false;}
        kernel=clCreateKernel(program,loss_?"reconstruction":optimizer_?"optimizer":"reference",&error);return error==CL_SUCCESS;
    }
    cl_kernel selectKernel(const std::vector<int>& arguments){
        cl_kernel kernel=this->kernel;
        if(!optimizer_ && parallelInvocationCount(arguments)){
            auto found=specialized.find(arguments[0]);
            if(found==specialized.end()){
                cl_int error=CL_SUCCESS;const char* source=referenceOpenCL;
                cl_program program=clCreateProgramWithSource(context,1,&source,nullptr,&error);
                if(error!=CL_SUCCESS)return nullptr;
                const std::string options="-cl-std=CL1.2 -DspecializedOperation="+std::to_string(arguments[0]);
                if(clBuildProgram(program,1,&selectedDevice,options.c_str(),nullptr,nullptr)!=CL_SUCCESS){clReleaseProgram(program);return nullptr;}
                cl_kernel compiled=clCreateKernel(program,"reference",&error);
                if(error!=CL_SUCCESS){clReleaseProgram(program);return nullptr;}
                found=specialized.emplace(arguments[0],std::make_pair(program,compiled)).first;
            }
            kernel=found->second.second;
        }
        return kernel;
    }
    bool run(std::vector<float>& memory,const std::vector<int>& arguments) override {
        cl_kernel kernel=selectKernel(arguments);if(!kernel)return false;
        const size_t upload_bytes = optimizer_ && !loss_
            ? optimizerInputCount(arguments)*sizeof(float) : memory.size()*sizeof(float);
        cl_int error=CL_SUCCESS;
        auto reserve=[&](cl_mem& buffer,size_t& capacity,size_t bytes,cl_mem_flags flags){
            if(capacity>=bytes)return true;
            cl_mem replacement=clCreateBuffer(context,flags,bytes,nullptr,&error);
            if(error!=CL_SUCCESS)return false;
            if(buffer)clReleaseMemObject(buffer);
            buffer=replacement;capacity=bytes;return true;
        };
        if(!reserve(cachedData,dataCapacity,memory.size()*sizeof(float),CL_MEM_READ_WRITE) ||
           !reserve(cachedParams,paramCapacity,arguments.size()*sizeof(int),CL_MEM_READ_ONLY))return false;
        cl_mem data=cachedData,params=cachedParams;
        if(clEnqueueWriteBuffer(queue,data,CL_TRUE,0,upload_bytes,memory.data(),0,nullptr,nullptr)!=CL_SUCCESS ||
           clEnqueueWriteBuffer(queue,params,CL_TRUE,0,arguments.size()*sizeof(int),arguments.data(),0,nullptr,nullptr)!=CL_SUCCESS)return false;
        bool ok=clSetKernelArg(kernel,0,sizeof(data),&data)==CL_SUCCESS && clSetKernelArg(kernel,1,sizeof(params),&params)==CL_SUCCESS;
        if(optimizer_) {
            std::vector<int> passes{0};if(!loss_ && (arguments[2]==4 || arguments[2]==8)){passes.push_back(1);passes.push_back(3);passes.push_back(2);}
            if(loss_)passes=RuntimeLossGrad::passes(arguments);
            for(int pass:passes) {
                if(!ok)break;
                ok=clEnqueueWriteBuffer(queue,params,CL_TRUE,0,sizeof(pass),&pass,0,nullptr,nullptr)==CL_SUCCESS;
                const size_t local=256,global=(loss_?pass==91:pass==3)?256:((size_t(arguments[1])+255)/256)*256;
                if(ok)ok=clEnqueueNDRangeKernel(queue,kernel,1,nullptr,&global,&local,0,nullptr,nullptr)==CL_SUCCESS;
            }
        } else {
            const size_t local=256,count=std::max(size_t(1),parallelInvocationCount(arguments)),size=((count+255)/256)*256;if(ok)ok=clEnqueueNDRangeKernel(queue,kernel,1,nullptr,&size,&local,0,nullptr,nullptr)==CL_SUCCESS;
        }
        const size_t result_bytes = optimizer_ && !loss_
            ? optimizerResultCount(arguments)*sizeof(float) : memory.size()*sizeof(float);
        if(ok)ok=clEnqueueReadBuffer(queue,data,CL_TRUE,0,result_bytes,memory.data(),0,nullptr,nullptr)==CL_SUCCESS;
        return ok;
    }
    bool runBatch(std::vector<float>& memory,const std::vector<std::vector<int>>& commands,const std::vector<MemoryRange>& downloads={}) override {
        if(optimizer_)return Device::runBatch(memory,commands,downloads);
        if(commands.empty())return true;
        std::vector<cl_kernel> kernels;for(const auto& args:commands){auto k=selectKernel(args);if(!k)return false;kernels.push_back(k);}
        cl_int error=CL_SUCCESS;const size_t bytes=memory.size()*sizeof(float);
        if(dataCapacity<bytes){cl_mem replacement=clCreateBuffer(context,CL_MEM_READ_WRITE,bytes,nullptr,&error);if(error!=CL_SUCCESS)return false;if(cachedData)clReleaseMemObject(cachedData);cachedData=replacement;dataCapacity=bytes;}
        if(batchParams.size()<commands.size())batchParams.resize(commands.size(),{nullptr,0});
        for(size_t i=0;i<commands.size();++i){auto& p=batchParams[i];const size_t size=commands[i].size()*sizeof(int);
            if(p.second<size){cl_mem replacement=clCreateBuffer(context,CL_MEM_READ_ONLY,size,nullptr,&error);if(error!=CL_SUCCESS)return false;if(p.first)clReleaseMemObject(p.first);p={replacement,size};}}
        bool ok=clEnqueueWriteBuffer(queue,cachedData,CL_FALSE,0,bytes,memory.data(),0,nullptr,nullptr)==CL_SUCCESS;
        for(size_t i=0;i<commands.size() && ok;++i){
            const auto& args=commands[i];cl_mem params=batchParams[i].first;cl_kernel kernel=kernels[i];
            ok=clEnqueueWriteBuffer(queue,params,CL_FALSE,0,args.size()*sizeof(int),args.data(),0,nullptr,nullptr)==CL_SUCCESS &&
                clSetKernelArg(kernel,0,sizeof(cachedData),&cachedData)==CL_SUCCESS && clSetKernelArg(kernel,1,sizeof(params),&params)==CL_SUCCESS;
            const size_t local=256,count=std::max(size_t(1),parallelInvocationCount(args)),global=((count+255)/256)*256;
            if(ok)ok=clEnqueueNDRangeKernel(queue,kernel,1,nullptr,&global,&local,0,nullptr,nullptr)==CL_SUCCESS;
        }
        size_t read_bytes=0;
        const std::vector<MemoryRange> full{{0,memory.size()}};
        const auto& ranges=downloads.empty()?full:downloads;
        for(const auto& range:ranges){
            if(!ok)break;
            if(range.offset>memory.size() || range.count>memory.size()-range.offset){ok=false;break;}
            ok=clEnqueueReadBuffer(queue,cachedData,CL_FALSE,range.offset*sizeof(float),range.count*sizeof(float),memory.data()+range.offset,0,nullptr,nullptr)==CL_SUCCESS;
            read_bytes+=range.count*sizeof(float);
        }
        // Drain uploads/readbacks on every path before their host storage dies.
        const bool drained=clFinish(queue)==CL_SUCCESS;ok=ok && drained;
        if(ok){++batch_statistics_.submissions;batch_statistics_.kernels+=commands.size();batch_statistics_.uploaded_bytes+=bytes;batch_statistics_.downloaded_bytes+=read_bytes;}
        return ok;
    }

};
#endif
#ifdef ENABLE_VULKAN
class VulkanDevice final : public Device {
    bool optimizer_=false,loss_=false;
    VkInstance instance=VK_NULL_HANDLE;
    VkPhysicalDevice physical=VK_NULL_HANDLE;
    VkDevice device=VK_NULL_HANDLE;
    VkQueue queue=VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors=VK_NULL_HANDLE;
    VkPipelineLayout layout=VK_NULL_HANDLE;
    VkPipeline pipeline=VK_NULL_HANDLE;
    std::map<int,VkPipeline> specialized;
    VkDescriptorPool descriptorPool=VK_NULL_HANDLE;
    VkCommandPool commandPool=VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memoryTypes{};
    VkDeviceSize maxRange=0;
    uint32_t maxGroups=0,maxGroupsY=0;
    struct Buffer {
        VkDevice device=VK_NULL_HANDLE;VkBuffer buffer=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;void* mapped=nullptr;
        ~Buffer(){if(mapped)vkUnmapMemory(device,memory);if(buffer)vkDestroyBuffer(device,buffer,nullptr);if(memory)vkFreeMemory(device,memory,nullptr);}
    };
    std::unique_ptr<Buffer> cachedData,cachedParams;
    std::vector<std::unique_ptr<Buffer>> batchParams;
    std::vector<size_t> batchParamCapacity;
    size_t descriptorCapacity=1;
    size_t dataCapacity=0,paramCapacity=0;
    VkCommandBuffer cachedCommand=VK_NULL_HANDLE;
    bool createBuffer(Buffer& b,size_t bytes,const void* source,size_t initialized_bytes=size_t(-1)) {
        if(bytes>maxRange)return false;
        b.device=device;
        VkBufferCreateInfo create{};create.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;create.size=bytes;create.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;create.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        if(vkCreateBuffer(device,&create,nullptr,&b.buffer)!=VK_SUCCESS)return false;
        VkMemoryRequirements requirements;vkGetBufferMemoryRequirements(device,b.buffer,&requirements);
        int type=-1;for(uint32_t i=0;i<memoryTypes.memoryTypeCount;++i){auto flags=memoryTypes.memoryTypes[i].propertyFlags;
            if((requirements.memoryTypeBits&(1u<<i)) && (flags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){type=i;break;}}
        if(type<0)return false;
        VkMemoryAllocateInfo alloc{};alloc.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;alloc.allocationSize=requirements.size;alloc.memoryTypeIndex=type;
        if(vkAllocateMemory(device,&alloc,nullptr,&b.memory)!=VK_SUCCESS || vkBindBufferMemory(device,b.buffer,b.memory,0)!=VK_SUCCESS || vkMapMemory(device,b.memory,0,bytes,0,&b.mapped)!=VK_SUCCESS)return false;
        std::memcpy(b.mapped,source,std::min(bytes,initialized_bytes));return true;
    }
public:
    ~VulkanDevice() override {
        if(device){vkDeviceWaitIdle(device);cachedData.reset();cachedParams.reset();batchParams.clear();for(const auto& entry:specialized)vkDestroyPipeline(device,entry.second,nullptr);if(commandPool)vkDestroyCommandPool(device,commandPool,nullptr);if(descriptorPool)vkDestroyDescriptorPool(device,descriptorPool,nullptr);if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);if(layout)vkDestroyPipelineLayout(device,layout,nullptr);if(descriptors)vkDestroyDescriptorSetLayout(device,descriptors,nullptr);vkDestroyDevice(device,nullptr);}if(instance)vkDestroyInstance(instance,nullptr);
    }
    bool initialize(int index,bool optimizer=false,bool loss=false) {
        loss_=loss;optimizer_=optimizer || loss;
        VkApplicationInfo app{};app.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;app.pApplicationName="Mimir native reference";app.apiVersion=VK_API_VERSION_1_0;
        VkInstanceCreateInfo info{};info.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;info.pApplicationInfo=&app;if(vkCreateInstance(&info,nullptr,&instance)!=VK_SUCCESS)return false;
        uint32_t n=0;if(vkEnumeratePhysicalDevices(instance,&n,nullptr)!=VK_SUCCESS || !n)return false;std::vector<VkPhysicalDevice> devices(n);if(vkEnumeratePhysicalDevices(instance,&n,devices.data())!=VK_SUCCESS)return false;
        if(optimizer_) {
            devices.erase(std::remove_if(devices.begin(),devices.end(),[](VkPhysicalDevice d){VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(d,&p);return p.deviceType==VK_PHYSICAL_DEVICE_TYPE_CPU;}),devices.end());
            if(devices.empty())return false;
        }
        if(index<0 || static_cast<size_t>(index)>=devices.size()){if(optimizer_)return false;index=0;}
        physical=devices[index];VkPhysicalDeviceProperties properties;vkGetPhysicalDeviceProperties(physical,&properties);if(std::getenv("MIMIR_ACCEL_VERBOSE"))std::cerr<<"[VULKAN reference] device="<<properties.deviceName<<" type="<<properties.deviceType<<'\n';maxRange=properties.limits.maxStorageBufferRange;maxGroups=properties.limits.maxComputeWorkGroupCount[0];maxGroupsY=properties.limits.maxComputeWorkGroupCount[1];vkGetPhysicalDeviceMemoryProperties(physical,&memoryTypes);
        vkGetPhysicalDeviceQueueFamilyProperties(physical,&n,nullptr);std::vector<VkQueueFamilyProperties> families(n);vkGetPhysicalDeviceQueueFamilyProperties(physical,&n,families.data());uint32_t family=n;for(uint32_t i=0;i<n;++i)if(families[i].queueFlags&VK_QUEUE_COMPUTE_BIT){family=i;break;}if(family==n)return false;
        float priority=1;VkDeviceQueueCreateInfo qi{};qi.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;VkDeviceCreateInfo di{};di.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;if(vkCreateDevice(physical,&di,nullptr,&device)!=VK_SUCCESS)return false;vkGetDeviceQueue(device,family,0,&queue);
        VkDescriptorSetLayoutBinding bindings[2]{};for(int i=0;i<2;++i){bindings[i].binding=i;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;bindings[i].descriptorCount=1;bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
        VkDescriptorSetLayoutCreateInfo dli{};dli.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;dli.bindingCount=2;dli.pBindings=bindings;if(vkCreateDescriptorSetLayout(device,&dli,nullptr,&descriptors)!=VK_SUCCESS)return false;
        VkPipelineLayoutCreateInfo pli{};pli.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pli.setLayoutCount=1;pli.pSetLayouts=&descriptors;if(vkCreatePipelineLayout(device,&pli,nullptr,&layout)!=VK_SUCCESS)return false;
        VkShaderModuleCreateInfo smi{};smi.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;smi.codeSize=loss_?sizeof(MimirLossKernels::spirv):optimizer_?sizeof(MimirOptimizerKernels::spirv):sizeof(referenceSpirv);smi.pCode=loss_?MimirLossKernels::spirv:optimizer_?MimirOptimizerKernels::spirv:referenceSpirv;VkShaderModule module;if(vkCreateShaderModule(device,&smi,nullptr,&module)!=VK_SUCCESS)return false;
        VkComputePipelineCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;pi.layout=layout;pi.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;pi.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pi.stage.module=module;pi.stage.pName="main";VkResult result=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pi,nullptr,&pipeline);vkDestroyShaderModule(device,module,nullptr);if(result!=VK_SUCCESS)return false;
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2};VkDescriptorPoolCreateInfo dpi{};dpi.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;dpi.maxSets=1;dpi.poolSizeCount=1;dpi.pPoolSizes=&ps;if(vkCreateDescriptorPool(device,&dpi,nullptr,&descriptorPool)!=VK_SUCCESS)return false;
        VkCommandPoolCreateInfo cpi{};cpi.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;cpi.queueFamilyIndex=family;cpi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;return vkCreateCommandPool(device,&cpi,nullptr,&commandPool)==VK_SUCCESS;
    }
    VkPipeline selectPipeline(const std::vector<int>& arguments){
        VkPipeline pipeline=this->pipeline;
        if(!optimizer_ && parallelInvocationCount(arguments)){
            auto found=specialized.find(arguments[0]);
            if(found==specialized.end()){
                VkShaderModuleCreateInfo shader{};shader.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;shader.codeSize=sizeof(referenceSpirv);shader.pCode=referenceSpirv;
                VkShaderModule module=VK_NULL_HANDLE;if(vkCreateShaderModule(device,&shader,nullptr,&module)!=VK_SUCCESS)return VK_NULL_HANDLE;
                const VkSpecializationMapEntry entry{0,0,sizeof(int)};
                const VkSpecializationInfo specialization{1,&entry,sizeof(int),arguments.data()};
                VkComputePipelineCreateInfo create{};create.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;create.layout=layout;
                create.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;create.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;create.stage.module=module;create.stage.pName="main";create.stage.pSpecializationInfo=&specialization;
                VkPipeline compiled=VK_NULL_HANDLE;const VkResult result=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&create,nullptr,&compiled);
                vkDestroyShaderModule(device,module,nullptr);if(result!=VK_SUCCESS)return VK_NULL_HANDLE;
                found=specialized.emplace(arguments[0],compiled).first;
            }
            pipeline=found->second;
        }
        return pipeline;
    }
    bool run(std::vector<float>& memory,const std::vector<int>& arguments) override {
        VkPipeline pipeline=selectPipeline(arguments);if(!pipeline)return false;
        const uint64_t groups=(std::max(size_t(1),parallelInvocationCount(arguments))+255)/256;
        const uint64_t groupsY=(groups+maxGroups-1)/maxGroups;
        if((optimizer_ && (uint64_t(arguments[1])+255)/256>maxGroups) || groupsY>maxGroupsY)return false;
        const uint32_t groupsX=uint32_t((groups+groupsY-1)/groupsY);
        const size_t upload_bytes = optimizer_ && !loss_
            ? optimizerInputCount(arguments)*sizeof(float) : memory.size()*sizeof(float);
        auto reserve=[&](std::unique_ptr<Buffer>& buffer,size_t& capacity,size_t bytes,const void* source,size_t initialized){
            if(capacity<bytes){auto replacement=std::make_unique<Buffer>();if(!createBuffer(*replacement,bytes,source,initialized))return false;buffer=std::move(replacement);capacity=bytes;}
            else std::memcpy(buffer->mapped,source,initialized);
            return true;
        };
        if(!reserve(cachedData,dataCapacity,memory.size()*4,memory.data(),upload_bytes) || !reserve(cachedParams,paramCapacity,arguments.size()*4,arguments.data(),arguments.size()*4))return false;
        Buffer& data=*cachedData;Buffer& params=*cachedParams;
        if(vkResetDescriptorPool(device,descriptorPool,0)!=VK_SUCCESS)return false;
        VkDescriptorSetAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;ai.descriptorPool=descriptorPool;ai.descriptorSetCount=1;ai.pSetLayouts=&descriptors;VkDescriptorSet set;if(vkAllocateDescriptorSets(device,&ai,&set)!=VK_SUCCESS)return false;
        VkDescriptorBufferInfo buffers[2]={{data.buffer,0,memory.size()*4},{params.buffer,0,arguments.size()*4}};VkWriteDescriptorSet writes[2]{};
        for(int i=0;i<2;++i){writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&buffers[i];}vkUpdateDescriptorSets(device,2,writes,0,nullptr);
        VkCommandBufferAllocateInfo cai{};cai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;cai.commandPool=commandPool;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;if(!cachedCommand && vkAllocateCommandBuffers(device,&cai,&cachedCommand)!=VK_SUCCESS)return false;VkCommandBuffer command=cachedCommand;
        bool ok=true;
        std::vector<int> passes{0};if(optimizer_ && !loss_ && (arguments[2]==4 || arguments[2]==8)){passes.push_back(1);passes.push_back(3);passes.push_back(2);}
        if(loss_)passes=RuntimeLossGrad::passes(arguments);
        for(int pass:passes) {
            if(!ok)break;
            if(optimizer_)std::memcpy(params.mapped,&pass,sizeof(pass));
            if(vkResetCommandBuffer(command,0)!=VK_SUCCESS){ok=false;break;}
        VkCommandBufferBeginInfo begin{};begin.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;ok=vkBeginCommandBuffer(command,&begin)==VK_SUCCESS;
        if(ok){vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);vkCmdDispatch(command,optimizer_?((loss_?pass==91:pass==3)?1:(uint32_t(arguments[1])+255)/256):groupsX,optimizer_?1:uint32_t(groupsY),1);if(optimizer_){VkMemoryBarrier barrier{};barrier.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);}ok=vkEndCommandBuffer(command)==VK_SUCCESS;}
        if(ok){VkSubmitInfo submit{};submit.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;submit.commandBufferCount=1;submit.pCommandBuffers=&command;ok=vkQueueSubmit(queue,1,&submit,VK_NULL_HANDLE)==VK_SUCCESS && vkQueueWaitIdle(queue)==VK_SUCCESS;}
        }
        const size_t result_bytes = optimizer_ && !loss_
            ? optimizerResultCount(arguments)*sizeof(float) : memory.size()*sizeof(float);
        if(ok)std::memcpy(memory.data(),data.mapped,result_bytes);
        return ok;
    }
    bool runBatch(std::vector<float>& memory,const std::vector<std::vector<int>>& commands,const std::vector<MemoryRange>& downloads={}) override {
        if(optimizer_)return Device::runBatch(memory,commands,downloads);
        if(commands.empty())return true;
        const size_t count=commands.size(),bytes=memory.size()*sizeof(float);
        std::vector<VkPipeline> pipelines;std::vector<std::pair<uint32_t,uint32_t>> grids;
        for(const auto& args:commands){auto pipeline=selectPipeline(args);if(!pipeline)return false;pipelines.push_back(pipeline);
            const uint64_t groups=(std::max(size_t(1),parallelInvocationCount(args))+255)/256,gy=(groups+maxGroups-1)/maxGroups;
            if(gy>maxGroupsY)return false;
            grids.push_back({uint32_t((groups+gy-1)/gy),uint32_t(gy)});
        }
        if(dataCapacity<bytes){auto replacement=std::make_unique<Buffer>();if(!createBuffer(*replacement,bytes,memory.data()))return false;cachedData=std::move(replacement);dataCapacity=bytes;}
        else std::memcpy(cachedData->mapped,memory.data(),bytes);
        if(batchParams.size()<count){batchParams.resize(count);batchParamCapacity.resize(count,0);}
        for(size_t i=0;i<count;++i){const size_t size=commands[i].size()*sizeof(int);
            if(batchParamCapacity[i]<size){auto replacement=std::make_unique<Buffer>();if(!createBuffer(*replacement,size,commands[i].data()))return false;batchParams[i]=std::move(replacement);batchParamCapacity[i]=size;}
            else std::memcpy(batchParams[i]->mapped,commands[i].data(),size);}
        if(descriptorCapacity<count){VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,uint32_t(2*count)};VkDescriptorPoolCreateInfo info{};info.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;info.maxSets=count;info.poolSizeCount=1;info.pPoolSizes=&size;
            VkDescriptorPool replacement=VK_NULL_HANDLE;if(vkCreateDescriptorPool(device,&info,nullptr,&replacement)!=VK_SUCCESS)return false;
            vkDestroyDescriptorPool(device,descriptorPool,nullptr);descriptorPool=replacement;descriptorCapacity=count;}
        if(vkResetDescriptorPool(device,descriptorPool,0)!=VK_SUCCESS)return false;
        std::vector<VkDescriptorSetLayout> layouts(count,descriptors);std::vector<VkDescriptorSet> sets(count);
        VkDescriptorSetAllocateInfo allocate{};allocate.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;allocate.descriptorPool=descriptorPool;allocate.descriptorSetCount=count;allocate.pSetLayouts=layouts.data();
        if(vkAllocateDescriptorSets(device,&allocate,sets.data())!=VK_SUCCESS)return false;
        std::vector<VkDescriptorBufferInfo> buffers(2*count);std::vector<VkWriteDescriptorSet> writes(2*count);
        for(size_t i=0;i<count;++i){buffers[2*i]={cachedData->buffer,0,bytes};buffers[2*i+1]={batchParams[i]->buffer,0,commands[i].size()*sizeof(int)};
            for(size_t b=0;b<2;++b){auto& w=writes[2*i+b];w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w.dstSet=sets[i];w.dstBinding=b;w.descriptorCount=1;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&buffers[2*i+b];}}
        vkUpdateDescriptorSets(device,writes.size(),writes.data(),0,nullptr);
        if(!cachedCommand){VkCommandBufferAllocateInfo info{};info.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;info.commandPool=commandPool;info.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;info.commandBufferCount=1;if(vkAllocateCommandBuffers(device,&info,&cachedCommand)!=VK_SUCCESS)return false;}
        if(vkResetCommandBuffer(cachedCommand,0)!=VK_SUCCESS)return false;
        VkCommandBufferBeginInfo begin{};begin.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;if(vkBeginCommandBuffer(cachedCommand,&begin)!=VK_SUCCESS)return false;
        for(size_t i=0;i<count;++i){
            vkCmdBindPipeline(cachedCommand,VK_PIPELINE_BIND_POINT_COMPUTE,pipelines[i]);vkCmdBindDescriptorSets(cachedCommand,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&sets[i],0,nullptr);
            vkCmdDispatch(cachedCommand,grids[i].first,grids[i].second,1);
            VkMemoryBarrier barrier{};barrier.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cachedCommand,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        }
        if(vkEndCommandBuffer(cachedCommand)!=VK_SUCCESS)return false;
        VkSubmitInfo submit{};submit.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;submit.commandBufferCount=1;submit.pCommandBuffers=&cachedCommand;
        if(vkQueueSubmit(queue,1,&submit,VK_NULL_HANDLE)!=VK_SUCCESS || vkQueueWaitIdle(queue)!=VK_SUCCESS)return false;
        size_t read_bytes=0;
        const std::vector<MemoryRange> full{{0,memory.size()}};
        for(const auto& range:downloads.empty()?full:downloads){
            if(range.offset>memory.size() || range.count>memory.size()-range.offset)return false;
            std::memcpy(memory.data()+range.offset,static_cast<const float*>(cachedData->mapped)+range.offset,range.count*sizeof(float));
            read_bytes+=range.count*sizeof(float);
        }
        ++batch_statistics_.submissions;batch_statistics_.kernels+=count;batch_statistics_.uploaded_bytes+=bytes;batch_statistics_.downloaded_bytes+=read_bytes;return true;
    }

};
#endif
std::unique_ptr<Device> createLossDevice(const std::string& backend,int index) {
#ifdef ENABLE_OPENCL
    if(backend=="OPENCL"){auto d=std::make_unique<OpenCLDevice>();if(d->initialize(index,false,true))return d;return nullptr;}
#endif
#ifdef ENABLE_VULKAN
    if(backend=="VULKAN"){auto d=std::make_unique<VulkanDevice>();if(d->initialize(index,false,true))return d;return nullptr;}
#endif
    return nullptr;
}
std::unique_ptr<Device> createOptimizerDevice(const std::string& backend,int index) {
#ifdef ENABLE_OPENCL
    if(backend=="OPENCL"){auto d=std::make_unique<OpenCLDevice>();if(d->initialize(index,true))return d;return nullptr;}
#endif
#ifdef ENABLE_VULKAN
    if(backend=="VULKAN"){auto d=std::make_unique<VulkanDevice>();if(d->initialize(index,true))return d;return nullptr;}
#endif
    return nullptr;
}
std::unique_ptr<Device> createDevice(const std::string& backend,int index) {
#ifdef ENABLE_OPENCL
    if(backend=="OPENCL"){auto d=std::make_unique<OpenCLDevice>();if(d->initialize(index))return d;return nullptr;}
#endif
#ifdef ENABLE_VULKAN
    if(backend=="VULKAN"){auto d=std::make_unique<VulkanDevice>();if(d->initialize(index))return d;return nullptr;}
#endif
    return nullptr;
}
}
