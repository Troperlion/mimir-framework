#include "PortableDevice.hpp"
#include "PortableShaderData.hpp"
#include "runtimes/losses/LossShaderData.hpp"
#include "runtimes/losses/LossDispatch.hpp"
#include "runtimes/optimizers/OptimizerShaderData.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
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
public:
    ~OpenCLDevice() override {
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
        cl_device_id device=devices[index];cl_int error=0;
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
    bool run(std::vector<float>& memory,const std::vector<int>& arguments) override {
        cl_int error=0;cl_mem data=clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,memory.size()*4,memory.data(),&error);if(error!=CL_SUCCESS)return false;
        cl_mem params=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,arguments.size()*4,const_cast<int*>(arguments.data()),&error);
        if(error!=CL_SUCCESS){clReleaseMemObject(data);return false;}
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
            size_t size=1;if(ok)ok=clEnqueueNDRangeKernel(queue,kernel,1,nullptr,&size,&size,0,nullptr,nullptr)==CL_SUCCESS;
        }
        if(ok)ok=clEnqueueReadBuffer(queue,data,CL_TRUE,0,memory.size()*4,memory.data(),0,nullptr,nullptr)==CL_SUCCESS;
        clReleaseMemObject(params);clReleaseMemObject(data);return ok;
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
    VkDescriptorPool descriptorPool=VK_NULL_HANDLE;
    VkCommandPool commandPool=VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memoryTypes{};
    VkDeviceSize maxRange=0;
    uint32_t maxGroups=0;
    struct Buffer {
        VkDevice device=VK_NULL_HANDLE;VkBuffer buffer=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;void* mapped=nullptr;
        ~Buffer(){if(mapped)vkUnmapMemory(device,memory);if(buffer)vkDestroyBuffer(device,buffer,nullptr);if(memory)vkFreeMemory(device,memory,nullptr);}
    };
    bool createBuffer(Buffer& b,size_t bytes,const void* source) {
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
        std::memcpy(b.mapped,source,bytes);return true;
    }
public:
    ~VulkanDevice() override {
        if(device){vkDeviceWaitIdle(device);if(commandPool)vkDestroyCommandPool(device,commandPool,nullptr);if(descriptorPool)vkDestroyDescriptorPool(device,descriptorPool,nullptr);if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);if(layout)vkDestroyPipelineLayout(device,layout,nullptr);if(descriptors)vkDestroyDescriptorSetLayout(device,descriptors,nullptr);vkDestroyDevice(device,nullptr);}if(instance)vkDestroyInstance(instance,nullptr);
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
        physical=devices[index];VkPhysicalDeviceProperties properties;vkGetPhysicalDeviceProperties(physical,&properties);if(std::getenv("MIMIR_ACCEL_VERBOSE"))std::cerr<<"[VULKAN reference] device="<<properties.deviceName<<" type="<<properties.deviceType<<'\n';maxRange=properties.limits.maxStorageBufferRange;maxGroups=properties.limits.maxComputeWorkGroupCount[0];vkGetPhysicalDeviceMemoryProperties(physical,&memoryTypes);
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
    bool run(std::vector<float>& memory,const std::vector<int>& arguments) override {
        if(optimizer_ && (uint64_t(arguments[1])+255)/256>maxGroups)return false;
        Buffer data,params;if(!createBuffer(data,memory.size()*4,memory.data()) || !createBuffer(params,arguments.size()*4,arguments.data()))return false;
        if(vkResetDescriptorPool(device,descriptorPool,0)!=VK_SUCCESS)return false;
        VkDescriptorSetAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;ai.descriptorPool=descriptorPool;ai.descriptorSetCount=1;ai.pSetLayouts=&descriptors;VkDescriptorSet set;if(vkAllocateDescriptorSets(device,&ai,&set)!=VK_SUCCESS)return false;
        VkDescriptorBufferInfo buffers[2]={{data.buffer,0,memory.size()*4},{params.buffer,0,arguments.size()*4}};VkWriteDescriptorSet writes[2]{};
        for(int i=0;i<2;++i){writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&buffers[i];}vkUpdateDescriptorSets(device,2,writes,0,nullptr);
        VkCommandBufferAllocateInfo cai{};cai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;cai.commandPool=commandPool;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;VkCommandBuffer command;if(vkAllocateCommandBuffers(device,&cai,&command)!=VK_SUCCESS)return false;
        bool ok=true;
        std::vector<int> passes{0};if(optimizer_ && !loss_ && (arguments[2]==4 || arguments[2]==8)){passes.push_back(1);passes.push_back(3);passes.push_back(2);}
        if(loss_)passes=RuntimeLossGrad::passes(arguments);
        for(int pass:passes) {
            if(!ok)break;
            if(optimizer_)std::memcpy(params.mapped,&pass,sizeof(pass));
            if(vkResetCommandBuffer(command,0)!=VK_SUCCESS){ok=false;break;}
        VkCommandBufferBeginInfo begin{};begin.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;ok=vkBeginCommandBuffer(command,&begin)==VK_SUCCESS;
        if(ok){vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);vkCmdDispatch(command,optimizer_?((loss_?pass==91:pass==3)?1:(uint32_t(arguments[1])+255)/256):1,1,1);if(optimizer_){VkMemoryBarrier barrier{};barrier.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_HOST_READ_BIT;vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);}ok=vkEndCommandBuffer(command)==VK_SUCCESS;}
        if(ok){VkSubmitInfo submit{};submit.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;submit.commandBufferCount=1;submit.pCommandBuffers=&command;ok=vkQueueSubmit(queue,1,&submit,VK_NULL_HANDLE)==VK_SUCCESS && vkQueueWaitIdle(queue)==VK_SUCCESS;}
        }
        if(ok)std::memcpy(memory.data(),data.mapped,memory.size()*4);
        vkFreeCommandBuffers(device,commandPool,1,&command);return ok;
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
