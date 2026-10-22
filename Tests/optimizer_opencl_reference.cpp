// Kept separate from Model.hpp so optional OpenCL headers do not conflict
// with the framework's opaque types in builds with ENABLE_OPENCL=OFF.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include "runtimes/optimizers/OptimizerDispatch.hpp"
#include "runtimes/optimizers/OptimizerShaderData.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct OpenCLReference {
    cl_context context=nullptr;
    cl_command_queue queue=nullptr;
    cl_program program=nullptr;
    cl_kernel kernel=nullptr;
    ~OpenCLReference() {
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
    bool initialize() {
        cl_uint count=0;
        if (clGetPlatformIDs(0,nullptr,&count)!=CL_SUCCESS || !count) return false;
        std::vector<cl_platform_id> platforms(count);
        require(clGetPlatformIDs(count,platforms.data(),nullptr)==CL_SUCCESS,"platform enumeration");
        cl_device_id device=nullptr;
        for (auto platform:platforms)
            if (clGetDeviceIDs(platform,CL_DEVICE_TYPE_ALL,1,&device,nullptr)==CL_SUCCESS) break;
        if (!device) return false;
        char name[256]={};
        clGetDeviceInfo(device,CL_DEVICE_NAME,sizeof(name),name,nullptr);
        std::cout << "OpenCL numerical reference device: " << name << '\n';
        cl_int error=0;
        context=clCreateContext(nullptr,1,&device,nullptr,nullptr,&error);
        require(error==CL_SUCCESS,"context creation");
        queue=clCreateCommandQueue(context,device,0,&error);
        require(error==CL_SUCCESS,"queue creation");
        const char* source=MimirOptimizerKernels::opencl;
        program=clCreateProgramWithSource(context,1,&source,nullptr,&error);
        require(error==CL_SUCCESS,"program creation");
        if (clBuildProgram(program,1,&device,"-cl-std=CL1.2",nullptr,nullptr)!=CL_SUCCESS) {
            size_t size=0;clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,nullptr,&size);
            std::string log(size,'\0');clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,size,log.data(),nullptr);
            throw std::runtime_error(log);
        }
        kernel=clCreateKernel(program,"optimizer",&error);
        require(error==CL_SUCCESS,"kernel creation");
        return true;
    }
    void run(std::vector<float>& memory, std::vector<int> parameters) {
        cl_int error=0;
        cl_mem data=clCreateBuffer(context,CL_MEM_READ_WRITE,
            memory.size()*sizeof(float),nullptr,&error);
        require(error==CL_SUCCESS,"data buffer");
        cl_mem args=clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
            parameters.size()*sizeof(int),parameters.data(),&error);
        if (error!=CL_SUCCESS) {clReleaseMemObject(data);throw std::runtime_error("parameter buffer");}
        // Poison scratch to verify that each reduction initializes its device
        // workspace before reading it; upload only parameters/moments/gradients.
        const cl_uint poison=0x7fc00000u;
        bool uploaded=clEnqueueFillBuffer(queue,data,&poison,sizeof(poison),0,
            memory.size()*sizeof(float),0,nullptr,nullptr)==CL_SUCCESS &&
            clEnqueueWriteBuffer(queue,data,CL_TRUE,0,optimizerInputCount(parameters)*sizeof(float),
                memory.data(),0,nullptr,nullptr)==CL_SUCCESS;
        bool ok=uploaded && clSetKernelArg(kernel,0,sizeof(data),&data)==CL_SUCCESS &&
            clSetKernelArg(kernel,1,sizeof(args),&args)==CL_SUCCESS;
        std::vector<int> passes{0};
        if (parameters[2]==4 || parameters[2]==8) passes={0,1,3,2};
        for (int pass:passes) {
            const size_t local=256,global=pass==3?256:((size_t(parameters[1])+255)/256)*256;
            if (ok) ok=clEnqueueWriteBuffer(queue,args,CL_TRUE,0,sizeof(pass),&pass,0,nullptr,nullptr)==CL_SUCCESS;
            if (ok) ok=clEnqueueNDRangeKernel(queue,kernel,1,nullptr,&global,&local,0,nullptr,nullptr)==CL_SUCCESS;
        }
        if (ok) ok=clEnqueueReadBuffer(queue,data,CL_TRUE,0,optimizerResultCount(parameters)*sizeof(float),
            memory.data(),0,nullptr,nullptr)==CL_SUCCESS;
        clReleaseMemObject(args);clReleaseMemObject(data);
        require(ok,"compact OpenCL execution");
    }
};
static OpenCLReference& reference() { static OpenCLReference instance; return instance; }
bool optimizerOpenCLReferenceInitialize() { return reference().initialize(); }
void optimizerOpenCLReferenceRun(std::vector<float>& memory, const std::vector<int>& parameters) {
    reference().run(memory, parameters);
}
