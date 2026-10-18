#ifndef __VULKAN_COMPUTE_HPP__
#define __VULKAN_COMPUTE_HPP__

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <optional>
#include <filesystem>
#include <mutex>

namespace VulkanCompute {

namespace fs = std::filesystem;

static inline std::vector<uint32_t> read_spirv_u32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    const std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    if (size <= 0 || (size % 4) != 0) return {};
    std::vector<uint32_t> out(static_cast<size_t>(size / 4));
    if (!f.read(reinterpret_cast<char*>(out.data()), size)) return {};
    return out;
}

static inline std::optional<std::string> find_shader_path_linear() {
    if (const char* env = std::getenv("MIMIR_VULKAN_LINEAR_SPV")) {
        if (env[0] != '\0' && fs::exists(env)) return std::string(env);
    }

    // Candidats relatifs au cwd (souvent la racine du repo)
    const char* candidates[] = {
        "./bin/shaders/linear_forward.comp.spv",
        "./shaders/linear_forward.comp.spv",
        "../bin/shaders/linear_forward.comp.spv",
        "../shaders/linear_forward.comp.spv",
        "./build/shaders/linear_forward.comp.spv",
        "./build_static/shaders/linear_forward.comp.spv",
        "./build_sfml/shaders/linear_forward.comp.spv",
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) return std::string(c);
    }
    return std::nullopt;
}

static inline std::optional<std::string> find_shader_path_add() {
    const char* candidates[] = {
        "./bin/shaders/add_forward.comp.spv",
        "./shaders/add_forward.comp.spv",
        "../bin/shaders/add_forward.comp.spv",
        "../shaders/add_forward.comp.spv",
        "./build/shaders/add_forward.comp.spv",
        "./build_static/shaders/add_forward.comp.spv",
        "./build_sfml/shaders/add_forward.comp.spv",
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) return std::string(c);
    }
    return std::nullopt;
}

static inline std::optional<std::string> find_shader_path_elementwise() {
    const char* candidates[] = {
        "./bin/shaders/elementwise.comp.spv",
        "./shaders/elementwise.comp.spv",
        "../bin/shaders/elementwise.comp.spv",
        "../shaders/elementwise.comp.spv",
        "./build/shaders/elementwise.comp.spv",
        "./build_static/shaders/elementwise.comp.spv",
        "./build_sfml/shaders/elementwise.comp.spv",
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) return std::string(c);
    }
    return std::nullopt;
}

static inline std::optional<std::string> find_shader_path_mul() {
    const char* candidates[] = {
        "./bin/shaders/mul_forward.comp.spv",
        "./shaders/mul_forward.comp.spv",
        "../bin/shaders/mul_forward.comp.spv",
        "../shaders/mul_forward.comp.spv",
        "./build/shaders/mul_forward.comp.spv",
        "./build_static/shaders/mul_forward.comp.spv",
        "./build_sfml/shaders/mul_forward.comp.spv",
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) return std::string(c);
    }
    return std::nullopt;
}

static inline std::optional<std::string> find_shader_path_relu() {
    const char* candidates[] = {
        "./bin/shaders/relu_forward.comp.spv",
        "./shaders/relu_forward.comp.spv",
        "../bin/shaders/relu_forward.comp.spv",
        "../shaders/relu_forward.comp.spv",
        "./build/shaders/relu_forward.comp.spv",
        "./build_static/shaders/relu_forward.comp.spv",
        "./build_sfml/shaders/relu_forward.comp.spv",
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) return std::string(c);
    }
    return std::nullopt;
}

static inline std::optional<std::string> find_shader_path_silu() {
    const char* candidates[] = {
        "./bin/shaders/silu_forward.comp.spv",
        "./shaders/silu_forward.comp.spv",
        "../bin/shaders/silu_forward.comp.spv",
        "../shaders/silu_forward.comp.spv",
        "./build/shaders/silu_forward.comp.spv",
        "./build_static/shaders/silu_forward.comp.spv",
        "./build_sfml/shaders/silu_forward.comp.spv",
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) return std::string(c);
    }
    return std::nullopt;
}

static inline std::optional<std::string> find_shader_path_gelu() {
    const char* candidates[] = {
        "./bin/shaders/gelu_forward.comp.spv",
        "./shaders/gelu_forward.comp.spv",
        "../bin/shaders/gelu_forward.comp.spv",
        "../shaders/gelu_forward.comp.spv",
        "./build/shaders/gelu_forward.comp.spv",
        "./build_static/shaders/gelu_forward.comp.spv",
        "./build_sfml/shaders/gelu_forward.comp.spv",
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) return std::string(c);
    }
    return std::nullopt;
}

static inline std::optional<std::string> find_shader_path_sigmoid() {
    const char* candidates[] = {
        "./bin/shaders/sigmoid_forward.comp.spv",
        "./shaders/sigmoid_forward.comp.spv",
        "../bin/shaders/sigmoid_forward.comp.spv",
        "../shaders/sigmoid_forward.comp.spv",
        "./build/shaders/sigmoid_forward.comp.spv",
        "./build_static/shaders/sigmoid_forward.comp.spv",
        "./build_sfml/shaders/sigmoid_forward.comp.spv",
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) return std::string(c);
    }
    return std::nullopt;
}

static inline std::optional<std::string> find_shader_path_tanh() {
    const char* candidates[] = {
        "./bin/shaders/tanh_forward.comp.spv",
        "./shaders/tanh_forward.comp.spv",
        "../bin/shaders/tanh_forward.comp.spv",
        "../shaders/tanh_forward.comp.spv",
        "./build/shaders/tanh_forward.comp.spv",
        "./build_static/shaders/tanh_forward.comp.spv",
        "./build_sfml/shaders/tanh_forward.comp.spv",
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) return std::string(c);
    }
    return std::nullopt;
}

static inline std::optional<std::string> find_shader_path_conv2d() {
    const char* candidates[] = {
        "./bin/shaders/conv2d_forward.comp.spv",
        "./shaders/conv2d_forward.comp.spv",
        "../bin/shaders/conv2d_forward.comp.spv",
        "../shaders/conv2d_forward.comp.spv",
        "./build/shaders/conv2d_forward.comp.spv",
        "./build_static/shaders/conv2d_forward.comp.spv",
        "./build_sfml/shaders/conv2d_forward.comp.spv",
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) return std::string(c);
    }
    return std::nullopt;
}

static inline std::optional<std::string> find_shader_path_conv_transpose2d() {
    const char* candidates[] = {
        "./bin/shaders/conv_transpose2d_forward.comp.spv",
        "./shaders/conv_transpose2d_forward.comp.spv",
        "../bin/shaders/conv_transpose2d_forward.comp.spv",
        "../shaders/conv_transpose2d_forward.comp.spv",
        "./build/shaders/conv_transpose2d_forward.comp.spv",
        "./build_static/shaders/conv_transpose2d_forward.comp.spv",
        "./build_sfml/shaders/conv_transpose2d_forward.comp.spv",
    };
    for (const char* c : candidates) {
        if (fs::exists(c)) return std::string(c);
    }
    return std::nullopt;
}

static inline uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1u << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

class ComputeEngine {
private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t queueFamilyIndex;
    bool initialized;

    // Resources for Linear kernel
    bool linear_ready_ = false;
    VkDescriptorSetLayout linear_dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout linear_pl_ = VK_NULL_HANDLE;
    VkPipeline linear_pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool linear_dp_ = VK_NULL_HANDLE;

    bool elementwise_ready_ = false;
    VkDescriptorSetLayout elementwise_dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout elementwise_pl_ = VK_NULL_HANDLE;
    VkPipeline elementwise_pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool elementwise_dp_ = VK_NULL_HANDLE;

    bool add_ready_ = false;
    VkDescriptorSetLayout add_dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout add_pl_ = VK_NULL_HANDLE;
    VkPipeline add_pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool add_dp_ = VK_NULL_HANDLE;

    bool mul_ready_ = false;
    VkDescriptorSetLayout mul_dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout mul_pl_ = VK_NULL_HANDLE;
    VkPipeline mul_pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool mul_dp_ = VK_NULL_HANDLE;

    bool relu_ready_ = false;
    VkDescriptorSetLayout relu_dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout relu_pl_ = VK_NULL_HANDLE;
    VkPipeline relu_pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool relu_dp_ = VK_NULL_HANDLE;

    bool silu_ready_ = false;
    VkDescriptorSetLayout silu_dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout silu_pl_ = VK_NULL_HANDLE;
    VkPipeline silu_pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool silu_dp_ = VK_NULL_HANDLE;

    bool gelu_ready_ = false;
    VkDescriptorSetLayout gelu_dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout gelu_pl_ = VK_NULL_HANDLE;
    VkPipeline gelu_pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool gelu_dp_ = VK_NULL_HANDLE;

    bool sigmoid_ready_ = false;
    VkDescriptorSetLayout sigmoid_dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout sigmoid_pl_ = VK_NULL_HANDLE;
    VkPipeline sigmoid_pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool sigmoid_dp_ = VK_NULL_HANDLE;

    bool tanh_ready_ = false;
    VkDescriptorSetLayout tanh_dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout tanh_pl_ = VK_NULL_HANDLE;
    VkPipeline tanh_pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool tanh_dp_ = VK_NULL_HANDLE;

    bool conv2d_ready_ = false;
    VkDescriptorSetLayout conv2d_dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout conv2d_pl_ = VK_NULL_HANDLE;
    VkPipeline conv2d_pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool conv2d_dp_ = VK_NULL_HANDLE;

    bool conv_transpose2d_ready_ = false;
    VkDescriptorSetLayout conv_transpose2d_dsl_ = VK_NULL_HANDLE;
    VkPipelineLayout conv_transpose2d_pl_ = VK_NULL_HANDLE;
    VkPipeline conv_transpose2d_pipe_ = VK_NULL_HANDLE;
    VkDescriptorPool conv_transpose2d_dp_ = VK_NULL_HANDLE;

    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;

    struct ReusableBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        size_t capacity = 0;
    };

    ReusableBuffer conv_input_buffer_;
    ReusableBuffer conv_weights_buffer_;
    ReusableBuffer conv_bias_buffer_;
    ReusableBuffer conv_output_buffer_;

    std::recursive_mutex linear_mutex_;
    
public:
    ComputeEngine() : instance(VK_NULL_HANDLE), physicalDevice(VK_NULL_HANDLE),
                      device(VK_NULL_HANDLE), computeQueue(VK_NULL_HANDLE),
                      queueFamilyIndex(0), initialized(false) {}
    
    ~ComputeEngine() {
        cleanup();
    }
    
    bool initialize() {
        // 1. Create Vulkan instance
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Mimir Compute";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "MimirEngine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;
        
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan instance" << std::endl;
            return false;
        }
        
        // 2. Select physical device with compute capability
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        
        if (deviceCount == 0) {
            std::cerr << "No Vulkan devices found" << std::endl;
            return false;
        }
        
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        // Pick first device with compute queue
        for (const auto& dev : devices) {
            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);
            
            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, queueFamilies.data());
            
            for (uint32_t i = 0; i < queueFamilyCount; ++i) {
                if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physicalDevice = dev;
                    queueFamilyIndex = i;
                    break;
                }
            }
            
            if (physicalDevice != VK_NULL_HANDLE) break;
        }
        
        if (physicalDevice == VK_NULL_HANDLE) {
            std::cerr << "No compute-capable device found" << std::endl;
            return false;
        }
        
        // 3. Create logical device
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        
        VkDeviceCreateInfo deviceCreateInfo = {};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        
        if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
            std::cerr << "Failed to create logical device" << std::endl;
            return false;
        }
        
        // 4. Get compute queue
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);

        // Create command pool
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &cmd_pool_) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan command pool" << std::endl;
            return false;
        }
        
        initialized = true;
        return true;
    }
    
    void cleanup() {
        releaseReusableBuffer(conv_input_buffer_);
        releaseReusableBuffer(conv_weights_buffer_);
        releaseReusableBuffer(conv_bias_buffer_);
        releaseReusableBuffer(conv_output_buffer_);
        cleanupLinearKernel();
        cleanupAddKernel();
        cleanupMulKernel();
        cleanupReluKernel();
        cleanupSiluKernel();
        cleanupGeluKernel();
        cleanupSigmoidKernel();
        cleanupTanhKernel();
        cleanupConv2dKernel();
        cleanupConvTranspose2dKernel();
        if (cmd_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, cmd_pool_, nullptr);
            cmd_pool_ = VK_NULL_HANDLE;
        }
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
        initialized = false;
    }
    
    bool isInitialized() const { return initialized; }
    VkDevice getDevice() const { return device; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
    VkQueue getComputeQueue() const { return computeQueue; }
    uint32_t getQueueFamilyIndex() const { return queueFamilyIndex; }

    struct LinearDims {
        uint32_t batch;
        uint32_t in_f;
        uint32_t out_f;
    };

    struct VecDims {
        uint32_t n;
    };

    struct ConvDims {
        uint32_t in_h;
        uint32_t in_w;
        uint32_t in_c;
        uint32_t out_c;
        uint32_t k;
        uint32_t stride;
        uint32_t pad;
        uint32_t dilation;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t use_bias;
    };

    bool linearForward(const float* input, const float* weights, const float* bias_or_null,
                       float* output, int batch, int in_f, int out_f) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized) return false;
        if (!input || !weights || !output) return false;
        if (batch <= 0 || in_f <= 0 || out_f <= 0) return false;
        if (!ensureLinearKernel()) return false;

        const size_t bytes_in = static_cast<size_t>(batch) * static_cast<size_t>(in_f) * sizeof(float);
        const size_t bytes_w  = static_cast<size_t>(out_f) * static_cast<size_t>(in_f) * sizeof(float);
        const size_t bytes_b  = static_cast<size_t>(out_f) * sizeof(float);
        const size_t bytes_o  = static_cast<size_t>(batch) * static_cast<size_t>(out_f) * sizeof(float);

        VkBuffer buf_in = VK_NULL_HANDLE, buf_w = VK_NULL_HANDLE, buf_b = VK_NULL_HANDLE, buf_o = VK_NULL_HANDLE;
        VkDeviceMemory mem_in = VK_NULL_HANDLE, mem_w = VK_NULL_HANDLE, mem_b = VK_NULL_HANDLE, mem_o = VK_NULL_HANDLE;

        auto make_buffer = [&](size_t size, VkBufferUsageFlags usage, VkBuffer& buf, VkDeviceMemory& mem) -> bool {
            VkBufferCreateInfo bi = {};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = size;
            bi.usage = usage;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer(device, &bi, nullptr, &buf) != VK_SUCCESS) return false;

            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(device, buf, &req);
            const uint32_t mt = findMemoryType(physicalDevice, req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (mt == UINT32_MAX) return false;

            VkMemoryAllocateInfo ai = {};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = mt;
            if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
            if (vkBindBufferMemory(device, buf, mem, 0) != VK_SUCCESS) return false;
            return true;
        };

        auto destroy_buf = [&](VkBuffer& b, VkDeviceMemory& m) {
            if (b != VK_NULL_HANDLE) { vkDestroyBuffer(device, b, nullptr); b = VK_NULL_HANDLE; }
            if (m != VK_NULL_HANDLE) { vkFreeMemory(device, m, nullptr); m = VK_NULL_HANDLE; }
        };

        if (!make_buffer(bytes_in, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buf_in, mem_in)) { destroy_buf(buf_in, mem_in); return false; }
        if (!make_buffer(bytes_w,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buf_w,  mem_w))  { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); return false; }
        if (!make_buffer(bytes_b,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buf_b,  mem_b))  { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); return false; }
        if (!make_buffer(bytes_o,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buf_o,  mem_o))  { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false; }

        auto upload = [&](VkDeviceMemory mem, const void* src, size_t sz) -> bool {
            void* mapped = nullptr;
            if (vkMapMemory(device, mem, 0, sz, 0, &mapped) != VK_SUCCESS) return false;
            std::memcpy(mapped, src, sz);
            vkUnmapMemory(device, mem);
            return true;
        };

        if (!upload(mem_in, input, bytes_in)) { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false; }
        if (!upload(mem_w,  weights, bytes_w)) { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false; }
        if (bias_or_null) {
            if (!upload(mem_b, bias_or_null, bytes_b)) { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false; }
        } else {
            std::vector<float> z(static_cast<size_t>(out_f), 0.0f);
            if (!upload(mem_b, z.data(), bytes_b)) { destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false; }
        }

        // Descriptor set
        VkDescriptorSet ds = VK_NULL_HANDLE;
        if (!allocAndWriteLinearDescriptorSet(
            buf_in, bytes_in,
            buf_w,  bytes_w,
            buf_b,  bytes_b,
            buf_o,  bytes_o,
            ds)) {
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }

        // Command buffer
        VkCommandBufferAllocateInfo cbai = {};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmd_pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &cbai, &cmd) != VK_SUCCESS) {
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }

        VkCommandBufferBeginInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, linear_pipe_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, linear_pl_, 0, 1, &ds, 0, nullptr);

        LinearDims dims{ static_cast<uint32_t>(batch), static_cast<uint32_t>(in_f), static_cast<uint32_t>(out_f) };
        vkCmdPushConstants(cmd, linear_pl_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LinearDims), &dims);

        const uint32_t gx = (static_cast<uint32_t>(batch) + 15u) / 16u;
        const uint32_t gy = (static_cast<uint32_t>(out_f) + 15u) / 16u;
        vkCmdDispatch(cmd, gx, gy, 1);
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }

        VkFence fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device, &fci, nullptr, &fence) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }

        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        if (vkQueueSubmit(computeQueue, 1, &si, fence) != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }
        if (vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }

        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);

        // Download output
        void* mapped = nullptr;
        if (vkMapMemory(device, mem_o, 0, bytes_o, 0, &mapped) != VK_SUCCESS) {
            destroy_buf(buf_in, mem_in); destroy_buf(buf_w, mem_w); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
            return false;
        }
        std::memcpy(output, mapped, bytes_o);
        vkUnmapMemory(device, mem_o);

        destroy_buf(buf_in, mem_in);
        destroy_buf(buf_w,  mem_w);
        destroy_buf(buf_b,  mem_b);
        destroy_buf(buf_o,  mem_o);
        return true;
    }

    // MatMul: C[M,N] = A[M,K] @ B[K,N]
    // Implémentation pragmatique: réutilise linearForward en transposant B -> [N,K]
    // et en injectant un biais nul.
    bool matmulForward(
        const float* a,
        const float* b,
        float* c,
        int M,
        int K,
        int N
    ) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized) return false;
        if (!a || !b || !c) return false;
        if (M <= 0 || K <= 0 || N <= 0) return false;

        std::vector<float> b_t(static_cast<size_t>(N) * static_cast<size_t>(K), 0.0f);
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                b_t[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)] =
                    b[static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(n)];
            }
        }

        std::vector<float> zero_bias(static_cast<size_t>(N), 0.0f);
        return linearForward(a, b_t.data(), zero_bias.data(), c, M, K, N);
    }

    // BatchMatMul: C[B,M,N] = A[B,M,K] @ Bm[B,K,N]
    bool batchMatMulForward(
        const float* a,
        const float* b,
        float* c,
        int B,
        int M,
        int K,
        int N
    ) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized) return false;
        if (!a || !b || !c) return false;
        if (B <= 0 || M <= 0 || K <= 0 || N <= 0) return false;

        const size_t a_stride = static_cast<size_t>(M) * static_cast<size_t>(K);
        const size_t b_stride = static_cast<size_t>(K) * static_cast<size_t>(N);
        const size_t c_stride = static_cast<size_t>(M) * static_cast<size_t>(N);

        for (int bi = 0; bi < B; ++bi) {
            const float* a_ptr = a + static_cast<size_t>(bi) * a_stride;
            const float* b_ptr = b + static_cast<size_t>(bi) * b_stride;
            float* c_ptr = c + static_cast<size_t>(bi) * c_stride;
            if (!matmulForward(a_ptr, b_ptr, c_ptr, M, K, N)) {
                return false;
            }
        }
        return true;
    }

    bool binaryForward(const float* a, const float* b, float* out, int n, int op, float alpha = 0.01f) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized || !a || !b || !out || n <= 0) return false;
        struct Parameters { uint32_t n; int32_t op; float alpha; } params{static_cast<uint32_t>(n),op,alpha};
        if (!ensureVectorKernel(elementwise_ready_,elementwise_dsl_,elementwise_pl_,elementwise_pipe_,
                elementwise_dp_,find_shader_path_elementwise(),3,"elementwise",sizeof(params))) return false;
        return runBinaryVectorKernel(elementwise_pipe_,elementwise_pl_,elementwise_dsl_,elementwise_dp_,
            a,b,out,n,&params,sizeof(params));
    }

    bool addForward(const float* a, const float* b, float* out, int n) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized) return false;
        if (!a || !b || !out || n <= 0) return false;
        if (!ensureAddKernel()) return false;
        return runBinaryVectorKernel(add_pipe_, add_pl_, add_dsl_, add_dp_, a, b, out, n);
    }

    bool mulForward(const float* a, const float* b, float* out, int n) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized) return false;
        if (!a || !b || !out || n <= 0) return false;
        if (!ensureMulKernel()) return false;
        return runBinaryVectorKernel(mul_pipe_, mul_pl_, mul_dsl_, mul_dp_, a, b, out, n);
    }

    bool reluForward(const float* in, float* out, int n) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized) return false;
        if (!in || !out || n <= 0) return false;
        if (!ensureReluKernel()) return false;
        return runUnaryVectorKernel(relu_pipe_, relu_pl_, relu_dsl_, relu_dp_, in, out, n);
    }

    bool siluForward(const float* in, float* out, int n) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized) return false;
        if (!in || !out || n <= 0) return false;
        if (!ensureSiluKernel()) return false;
        return runUnaryVectorKernel(silu_pipe_, silu_pl_, silu_dsl_, silu_dp_, in, out, n);
    }

    bool geluForward(const float* in, float* out, int n) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized) return false;
        if (!in || !out || n <= 0) return false;
        if (!ensureGeluKernel()) return false;
        return runUnaryVectorKernel(gelu_pipe_, gelu_pl_, gelu_dsl_, gelu_dp_, in, out, n);
    }

    bool sigmoidForward(const float* in, float* out, int n) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized) return false;
        if (!in || !out || n <= 0) return false;
        if (!ensureSigmoidKernel()) return false;
        return runUnaryVectorKernel(sigmoid_pipe_, sigmoid_pl_, sigmoid_dsl_, sigmoid_dp_, in, out, n);
    }

    bool tanhForward(const float* in, float* out, int n) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized) return false;
        if (!in || !out || n <= 0) return false;
        if (!ensureTanhKernel()) return false;
        return runUnaryVectorKernel(tanh_pipe_, tanh_pl_, tanh_dsl_, tanh_dp_, in, out, n);
    }

    // Executes a sequence of unary kernels while the intermediate values stay
    // in Vulkan buffers. op codes: 0=ReLU, 1=SiLU, 2=GELU, 3=Sigmoid, 4=Tanh.
    bool unaryChainForwardResident(
        const float* in,
        float* out,
        int n,
        const std::vector<int>& operations
    ) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized || !in || !out || n <= 0 || operations.empty()) return false;

        struct KernelSelection {
            VkPipeline pipe = VK_NULL_HANDLE;
            VkPipelineLayout layout = VK_NULL_HANDLE;
            VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
            VkDescriptorPool pool = VK_NULL_HANDLE;
        };
        auto select = [&](const int op, KernelSelection& selected) -> bool {
            switch (op) {
                case 0:
                    if (!ensureReluKernel()) return false;
                    selected = {relu_pipe_, relu_pl_, relu_dsl_, relu_dp_};
                    return true;
                case 1:
                    if (!ensureSiluKernel()) return false;
                    selected = {silu_pipe_, silu_pl_, silu_dsl_, silu_dp_};
                    return true;
                case 2:
                    if (!ensureGeluKernel()) return false;
                    selected = {gelu_pipe_, gelu_pl_, gelu_dsl_, gelu_dp_};
                    return true;
                case 3:
                    if (!ensureSigmoidKernel()) return false;
                    selected = {sigmoid_pipe_, sigmoid_pl_, sigmoid_dsl_, sigmoid_dp_};
                    return true;
                case 4:
                    if (!ensureTanhKernel()) return false;
                    selected = {tanh_pipe_, tanh_pl_, tanh_dsl_, tanh_dp_};
                    return true;
                default:
                    return false;
            }
        };

        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        VkBuffer buffers[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
        VkDeviceMemory memories[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
        auto cleanup_buffers = [&]() {
            for (int i = 0; i < 2; ++i) {
                if (buffers[i] != VK_NULL_HANDLE) vkDestroyBuffer(device, buffers[i], nullptr);
                if (memories[i] != VK_NULL_HANDLE) vkFreeMemory(device, memories[i], nullptr);
                buffers[i] = VK_NULL_HANDLE;
                memories[i] = VK_NULL_HANDLE;
            }
        };
        auto allocate_buffer = [&](const int index) -> bool {
            VkBufferCreateInfo bi = {};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = bytes;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer(device, &bi, nullptr, &buffers[index]) != VK_SUCCESS) return false;
            VkMemoryRequirements req = {};
            vkGetBufferMemoryRequirements(device, buffers[index], &req);
            const uint32_t mt = findMemoryType(
                physicalDevice, req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (mt == UINT32_MAX) return false;
            VkMemoryAllocateInfo ai = {};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = mt;
            if (vkAllocateMemory(device, &ai, nullptr, &memories[index]) != VK_SUCCESS) return false;
            return vkBindBufferMemory(device, buffers[index], memories[index], 0) == VK_SUCCESS;
        };
        if (!allocate_buffer(0) || !allocate_buffer(1)) {
            cleanup_buffers();
            return false;
        }
        void* mapped = nullptr;
        if (vkMapMemory(device, memories[0], 0, bytes, 0, &mapped) != VK_SUCCESS) {
            cleanup_buffers();
            return false;
        }
        std::memcpy(mapped, in, bytes);
        vkUnmapMemory(device, memories[0]);

        int source = 0;
        for (const int operation : operations) {
            const int destination = 1 - source;
            KernelSelection kernel;
            if (!select(operation, kernel)) {
                cleanup_buffers();
                return false;
            }
            VkDescriptorSet ds = VK_NULL_HANDLE;
            std::vector<VkDescriptorBufferInfo> infos(2);
            infos[0] = {buffers[source], 0, static_cast<VkDeviceSize>(bytes)};
            infos[1] = {buffers[destination], 0, static_cast<VkDeviceSize>(bytes)};
            if (!allocAndWriteDescriptorSet(kernel.pool, kernel.descriptors, infos, ds)) {
                cleanup_buffers();
                return false;
            }
            VkCommandBufferAllocateInfo cbai = {};
            cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cbai.commandPool = cmd_pool_;
            cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbai.commandBufferCount = 1;
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device, &cbai, &cmd) != VK_SUCCESS) {
                cleanup_buffers();
                return false;
            }
            VkCommandBufferBeginInfo begin = {};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            bool ok = vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS;
            if (ok) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipe);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        kernel.layout, 0, 1, &ds, 0, nullptr);
                const VecDims dims{static_cast<uint32_t>(n)};
                vkCmdPushConstants(cmd, kernel.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(VecDims), &dims);
                vkCmdDispatch(cmd, (static_cast<uint32_t>(n) + 255u) / 256u, 1, 1);
                ok = vkEndCommandBuffer(cmd) == VK_SUCCESS;
            }
            VkFence fence = VK_NULL_HANDLE;
            if (ok) {
                VkFenceCreateInfo fci = {};
                fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                ok = vkCreateFence(device, &fci, nullptr, &fence) == VK_SUCCESS;
            }
            if (ok) {
                VkSubmitInfo submit = {};
                submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit.commandBufferCount = 1;
                submit.pCommandBuffers = &cmd;
                ok = vkQueueSubmit(computeQueue, 1, &submit, fence) == VK_SUCCESS;
                if (ok) ok = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
            }
            if (fence != VK_NULL_HANDLE) vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            if (!ok) {
                cleanup_buffers();
                return false;
            }
            source = destination;
        }

        if (vkMapMemory(device, memories[source], 0, bytes, 0, &mapped) != VK_SUCCESS) {
            cleanup_buffers();
            return false;
        }
        std::memcpy(out, mapped, bytes);
        vkUnmapMemory(device, memories[source]);
        cleanup_buffers();
        return true;
    }

    bool conv2dForward(
        const float* in,
        const float* w,
        const float* b,
        float* out,
        int in_h,
        int in_w,
        int in_c,
        int out_c,
        int k,
        int stride,
        int pad,
        int dilation
    ) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized) return false;
        if (!in || !w || !out) return false;
        if (in_h <= 0 || in_w <= 0 || in_c <= 0 || out_c <= 0 || k <= 0 || stride <= 0 || dilation <= 0) return false;
        if (!ensureConv2dKernel()) return false;

        const int out_h = (in_h + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        const int out_w = (in_w + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
        if (out_h <= 0 || out_w <= 0) return false;

        ConvDims dims{};
        dims.in_h = static_cast<uint32_t>(in_h);
        dims.in_w = static_cast<uint32_t>(in_w);
        dims.in_c = static_cast<uint32_t>(in_c);
        dims.out_c = static_cast<uint32_t>(out_c);
        dims.k = static_cast<uint32_t>(k);
        dims.stride = static_cast<uint32_t>(stride);
        dims.pad = static_cast<uint32_t>(pad);
        dims.dilation = static_cast<uint32_t>(dilation);
        dims.out_h = static_cast<uint32_t>(out_h);
        dims.out_w = static_cast<uint32_t>(out_w);
        dims.use_bias = (b != nullptr) ? 1u : 0u;

        const size_t in_bytes = static_cast<size_t>(in_h) * static_cast<size_t>(in_w) * static_cast<size_t>(in_c) * sizeof(float);
        const size_t w_bytes = static_cast<size_t>(out_c) * static_cast<size_t>(in_c) * static_cast<size_t>(k) * static_cast<size_t>(k) * sizeof(float);
        const size_t b_bytes = static_cast<size_t>(out_c) * sizeof(float);
        const size_t out_bytes = static_cast<size_t>(out_h) * static_cast<size_t>(out_w) * static_cast<size_t>(out_c) * sizeof(float);
        return runConvKernel(conv2d_pipe_, conv2d_pl_, conv2d_dsl_, conv2d_dp_, in, in_bytes, w, w_bytes, b, b_bytes, out, out_bytes, dims);
    }

    bool convTranspose2dForward(
        const float* in,
        const float* w,
        const float* b,
        float* out,
        int in_h,
        int in_w,
        int in_c,
        int out_c,
        int k,
        int stride,
        int pad,
        int dilation
    ) {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (!initialized) return false;
        if (!in || !w || !out) return false;
        if (in_h <= 0 || in_w <= 0 || in_c <= 0 || out_c <= 0 || k <= 0 || stride <= 0 || dilation <= 0) return false;
        if (!ensureConvTranspose2dKernel()) return false;

        const int out_h = (in_h - 1) * stride - 2 * pad + dilation * (k - 1) + 1;
        const int out_w = (in_w - 1) * stride - 2 * pad + dilation * (k - 1) + 1;
        if (out_h <= 0 || out_w <= 0) return false;

        ConvDims dims{};
        dims.in_h = static_cast<uint32_t>(in_h);
        dims.in_w = static_cast<uint32_t>(in_w);
        dims.in_c = static_cast<uint32_t>(in_c);
        dims.out_c = static_cast<uint32_t>(out_c);
        dims.k = static_cast<uint32_t>(k);
        dims.stride = static_cast<uint32_t>(stride);
        dims.pad = static_cast<uint32_t>(pad);
        dims.dilation = static_cast<uint32_t>(dilation);
        dims.out_h = static_cast<uint32_t>(out_h);
        dims.out_w = static_cast<uint32_t>(out_w);
        dims.use_bias = (b != nullptr) ? 1u : 0u;

        const size_t in_bytes = static_cast<size_t>(in_h) * static_cast<size_t>(in_w) * static_cast<size_t>(in_c) * sizeof(float);
        const size_t w_bytes = static_cast<size_t>(out_c) * static_cast<size_t>(in_c) * static_cast<size_t>(k) * static_cast<size_t>(k) * sizeof(float);
        const size_t b_bytes = static_cast<size_t>(out_c) * sizeof(float);
        const size_t out_bytes = static_cast<size_t>(out_h) * static_cast<size_t>(out_w) * static_cast<size_t>(out_c) * sizeof(float);
        return runConvKernel(conv_transpose2d_pipe_, conv_transpose2d_pl_, conv_transpose2d_dsl_, conv_transpose2d_dp_, in, in_bytes, w, w_bytes, b, b_bytes, out, out_bytes, dims);
    }

private:
    static bool env_verbose_enabled() {
        if (const char* v = std::getenv("MIMIR_ACCEL_VERBOSE")) {
            return (v[0] != '\0' && !(v[0] == '0' && v[1] == '\0'));
        }
        return false;
    }

    void cleanupVectorKernel(
        bool& ready,
        VkDescriptorSetLayout& dsl,
        VkPipelineLayout& pl,
        VkPipeline& pipe,
        VkDescriptorPool& dp
    ) {
        if (dp != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, dp, nullptr);
            dp = VK_NULL_HANDLE;
        }
        if (pipe != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipe, nullptr);
            pipe = VK_NULL_HANDLE;
        }
        if (pl != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pl, nullptr);
            pl = VK_NULL_HANDLE;
        }
        if (dsl != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, dsl, nullptr);
            dsl = VK_NULL_HANDLE;
        }
        ready = false;
    }

    bool ensureVectorKernel(
        bool& ready,
        VkDescriptorSetLayout& dsl,
        VkPipelineLayout& pl,
        VkPipeline& pipe,
        VkDescriptorPool& dp,
        const std::optional<std::string>& shader_path,
        uint32_t binding_count,
        const char* kernel_name,
        uint32_t push_constant_size = sizeof(VecDims)
    ) {
        if (ready) return true;
        if (!shader_path.has_value()) {
            if (env_verbose_enabled()) {
                std::cerr << "Vulkan " << kernel_name << " shader not found (SPIR-V missing).\n";
            }
            return false;
        }

        std::vector<uint32_t> spirv = read_spirv_u32(*shader_path);
        if (spirv.empty()) return false;

        VkShaderModule shader = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo smci = {};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spirv.size() * sizeof(uint32_t);
        smci.pCode = spirv.data();
        if (vkCreateShaderModule(device, &smci, nullptr, &shader) != VK_SUCCESS) {
            return false;
        }

        std::vector<VkDescriptorSetLayoutBinding> bindings(binding_count);
        for (uint32_t i = 0; i < binding_count; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo dslci = {};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = binding_count;
        dslci.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsl) != VK_SUCCESS) {
            vkDestroyShaderModule(device, shader, nullptr);
            return false;
        }

        VkPushConstantRange pcr = {};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = push_constant_size;

        VkPipelineLayoutCreateInfo plci = {};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(device, &plci, nullptr, &pl) != VK_SUCCESS) {
            vkDestroyShaderModule(device, shader, nullptr);
            return false;
        }

        VkPipelineShaderStageCreateInfo stage = {};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpci = {};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = pl;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe) != VK_SUCCESS) {
            vkDestroyShaderModule(device, shader, nullptr);
            return false;
        }
        vkDestroyShaderModule(device, shader, nullptr);

        VkDescriptorPoolSize ps = {};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = binding_count;

        VkDescriptorPoolCreateInfo dpci = {};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(device, &dpci, nullptr, &dp) != VK_SUCCESS) {
            return false;
        }

        ready = true;
        return true;
    }

    bool allocAndWriteDescriptorSet(
        VkDescriptorPool dp,
        VkDescriptorSetLayout dsl,
        const std::vector<VkDescriptorBufferInfo>& infos,
        VkDescriptorSet& outDS
    ) {
        if (!dp || !dsl || infos.empty()) return false;

        vkResetDescriptorPool(device, dp, 0);

        VkDescriptorSetAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = dp;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &dsl;
        if (vkAllocateDescriptorSets(device, &ai, &outDS) != VK_SUCCESS) {
            return false;
        }

        std::vector<VkWriteDescriptorSet> writes(infos.size());
        for (uint32_t i = 0; i < infos.size(); ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = outDS;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return true;
    }

    bool runBinaryVectorKernel(
        VkPipeline pipe,
        VkPipelineLayout pl,
        VkDescriptorSetLayout dsl,
        VkDescriptorPool dp,
        const float* a,
        const float* b,
        float* out,
        int n,
        const void* parameters = nullptr,
        uint32_t parameter_size = 0
    ) {
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);

        VkBuffer buf_a = VK_NULL_HANDLE, buf_b = VK_NULL_HANDLE, buf_o = VK_NULL_HANDLE;
        VkDeviceMemory mem_a = VK_NULL_HANDLE, mem_b = VK_NULL_HANDLE, mem_o = VK_NULL_HANDLE;

        auto make_buffer = [&](size_t size, VkBuffer& buf, VkDeviceMemory& mem) -> bool {
            VkBufferCreateInfo bi = {};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = size;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer(device, &bi, nullptr, &buf) != VK_SUCCESS) return false;

            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(device, buf, &req);
            const uint32_t mt = findMemoryType(physicalDevice, req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (mt == UINT32_MAX) return false;

            VkMemoryAllocateInfo ai = {};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = mt;
            if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
            if (vkBindBufferMemory(device, buf, mem, 0) != VK_SUCCESS) return false;
            return true;
        };

        auto destroy_buf = [&](VkBuffer& b0, VkDeviceMemory& m0) {
            if (b0 != VK_NULL_HANDLE) { vkDestroyBuffer(device, b0, nullptr); b0 = VK_NULL_HANDLE; }
            if (m0 != VK_NULL_HANDLE) { vkFreeMemory(device, m0, nullptr); m0 = VK_NULL_HANDLE; }
        };

        auto upload = [&](VkDeviceMemory mem, const void* src, size_t sz) -> bool {
            void* mapped = nullptr;
            if (vkMapMemory(device, mem, 0, sz, 0, &mapped) != VK_SUCCESS) return false;
            std::memcpy(mapped, src, sz);
            vkUnmapMemory(device, mem);
            return true;
        };

        if (!make_buffer(bytes, buf_a, mem_a)) { destroy_buf(buf_a, mem_a); return false; }
        if (!make_buffer(bytes, buf_b, mem_b)) { destroy_buf(buf_a, mem_a); destroy_buf(buf_b, mem_b); return false; }
        if (!make_buffer(bytes, buf_o, mem_o)) { destroy_buf(buf_a, mem_a); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false; }

        if (!upload(mem_a, a, bytes) || !upload(mem_b, b, bytes)) {
            destroy_buf(buf_a, mem_a); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false;
        }

        VkDescriptorSet ds = VK_NULL_HANDLE;
        std::vector<VkDescriptorBufferInfo> infos(3);
        infos[0] = { buf_a, 0, static_cast<VkDeviceSize>(bytes) };
        infos[1] = { buf_b, 0, static_cast<VkDeviceSize>(bytes) };
        infos[2] = { buf_o, 0, static_cast<VkDeviceSize>(bytes) };
        if (!allocAndWriteDescriptorSet(dp, dsl, infos, ds)) {
            destroy_buf(buf_a, mem_a); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false;
        }

        VkCommandBufferAllocateInfo cbai = {};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmd_pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &cbai, &cmd) != VK_SUCCESS) {
            destroy_buf(buf_a, mem_a); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false;
        }

        VkCommandBufferBeginInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_a, mem_a); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);

        VecDims dims{ static_cast<uint32_t>(n) };
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0,
            parameters ? parameter_size : sizeof(VecDims), parameters ? parameters : &dims);
        const uint32_t gx = (static_cast<uint32_t>(n) + 255u) / 256u;
        vkCmdDispatch(cmd, gx, 1, 1);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_a, mem_a); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false;
        }

        VkFence fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device, &fci, nullptr, &fence) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_a, mem_a); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false;
        }

        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        if (vkQueueSubmit(computeQueue, 1, &si, fence) != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_a, mem_a); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false;
        }
        if (vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_a, mem_a); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false;
        }

        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);

        void* mapped = nullptr;
        if (vkMapMemory(device, mem_o, 0, bytes, 0, &mapped) != VK_SUCCESS) {
            destroy_buf(buf_a, mem_a); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o); return false;
        }
        std::memcpy(out, mapped, bytes);
        vkUnmapMemory(device, mem_o);

        destroy_buf(buf_a, mem_a); destroy_buf(buf_b, mem_b); destroy_buf(buf_o, mem_o);
        return true;
    }

    bool runUnaryVectorKernel(
        VkPipeline pipe,
        VkPipelineLayout pl,
        VkDescriptorSetLayout dsl,
        VkDescriptorPool dp,
        const float* in,
        float* out,
        int n
    ) {
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);

        VkBuffer buf_in = VK_NULL_HANDLE, buf_o = VK_NULL_HANDLE;
        VkDeviceMemory mem_in = VK_NULL_HANDLE, mem_o = VK_NULL_HANDLE;

        auto make_buffer = [&](size_t size, VkBuffer& buf, VkDeviceMemory& mem) -> bool {
            VkBufferCreateInfo bi = {};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = size;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer(device, &bi, nullptr, &buf) != VK_SUCCESS) return false;

            VkMemoryRequirements req;
            vkGetBufferMemoryRequirements(device, buf, &req);
            const uint32_t mt = findMemoryType(physicalDevice, req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (mt == UINT32_MAX) return false;

            VkMemoryAllocateInfo ai = {};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = mt;
            if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
            if (vkBindBufferMemory(device, buf, mem, 0) != VK_SUCCESS) return false;
            return true;
        };

        auto destroy_buf = [&](VkBuffer& b0, VkDeviceMemory& m0) {
            if (b0 != VK_NULL_HANDLE) { vkDestroyBuffer(device, b0, nullptr); b0 = VK_NULL_HANDLE; }
            if (m0 != VK_NULL_HANDLE) { vkFreeMemory(device, m0, nullptr); m0 = VK_NULL_HANDLE; }
        };

        auto upload = [&](VkDeviceMemory mem, const void* src, size_t sz) -> bool {
            void* mapped = nullptr;
            if (vkMapMemory(device, mem, 0, sz, 0, &mapped) != VK_SUCCESS) return false;
            std::memcpy(mapped, src, sz);
            vkUnmapMemory(device, mem);
            return true;
        };

        if (!make_buffer(bytes, buf_in, mem_in)) { destroy_buf(buf_in, mem_in); return false; }
        if (!make_buffer(bytes, buf_o, mem_o)) { destroy_buf(buf_in, mem_in); destroy_buf(buf_o, mem_o); return false; }
        if (!upload(mem_in, in, bytes)) {
            destroy_buf(buf_in, mem_in); destroy_buf(buf_o, mem_o); return false;
        }

        VkDescriptorSet ds = VK_NULL_HANDLE;
        std::vector<VkDescriptorBufferInfo> infos(2);
        infos[0] = { buf_in, 0, static_cast<VkDeviceSize>(bytes) };
        infos[1] = { buf_o, 0, static_cast<VkDeviceSize>(bytes) };
        if (!allocAndWriteDescriptorSet(dp, dsl, infos, ds)) {
            destroy_buf(buf_in, mem_in); destroy_buf(buf_o, mem_o); return false;
        }

        VkCommandBufferAllocateInfo cbai = {};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmd_pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &cbai, &cmd) != VK_SUCCESS) {
            destroy_buf(buf_in, mem_in); destroy_buf(buf_o, mem_o); return false;
        }

        VkCommandBufferBeginInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_o, mem_o); return false;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);

        VecDims dims{ static_cast<uint32_t>(n) };
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VecDims), &dims);
        const uint32_t gx = (static_cast<uint32_t>(n) + 255u) / 256u;
        vkCmdDispatch(cmd, gx, 1, 1);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_o, mem_o); return false;
        }

        VkFence fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device, &fci, nullptr, &fence) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_o, mem_o); return false;
        }

        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        if (vkQueueSubmit(computeQueue, 1, &si, fence) != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_o, mem_o); return false;
        }
        if (vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            destroy_buf(buf_in, mem_in); destroy_buf(buf_o, mem_o); return false;
        }

        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);

        void* mapped = nullptr;
        if (vkMapMemory(device, mem_o, 0, bytes, 0, &mapped) != VK_SUCCESS) {
            destroy_buf(buf_in, mem_in); destroy_buf(buf_o, mem_o); return false;
        }
        std::memcpy(out, mapped, bytes);
        vkUnmapMemory(device, mem_o);

        destroy_buf(buf_in, mem_in); destroy_buf(buf_o, mem_o);
        return true;
    }

    bool runConvKernel(
        VkPipeline pipe,
        VkPipelineLayout pl,
        VkDescriptorSetLayout dsl,
        VkDescriptorPool dp,
        const float* in,
        size_t in_bytes,
        const float* w,
        size_t w_bytes,
        const float* b,
        size_t b_bytes,
        float* out,
        size_t out_bytes,
        const ConvDims& dims
    ) {
        if (!ensureReusableBuffer(conv_input_buffer_, in_bytes) ||
            !ensureReusableBuffer(conv_weights_buffer_, w_bytes) ||
            !ensureReusableBuffer(conv_bias_buffer_, b_bytes) ||
            !ensureReusableBuffer(conv_output_buffer_, out_bytes)) {
            return false;
        }

        const VkBuffer buf_in = conv_input_buffer_.buffer;
        const VkBuffer buf_w = conv_weights_buffer_.buffer;
        const VkBuffer buf_b = conv_bias_buffer_.buffer;
        const VkBuffer buf_o = conv_output_buffer_.buffer;
        const VkDeviceMemory mem_in = conv_input_buffer_.memory;
        const VkDeviceMemory mem_w = conv_weights_buffer_.memory;
        const VkDeviceMemory mem_b = conv_bias_buffer_.memory;
        const VkDeviceMemory mem_o = conv_output_buffer_.memory;

        auto upload = [&](VkDeviceMemory mem, const void* src, size_t sz) -> bool {
            void* mapped = nullptr;
            if (vkMapMemory(device, mem, 0, sz, 0, &mapped) != VK_SUCCESS) return false;
            std::memcpy(mapped, src, sz);
            vkUnmapMemory(device, mem);
            return true;
        };

        if (!upload(mem_in, in, in_bytes) || !upload(mem_w, w, w_bytes)) {
            return false;
        }

        if (b) {
            if (!upload(mem_b, b, b_bytes)) {
                return false;
            }
        } else {
            std::vector<float> zb(b_bytes / sizeof(float), 0.0f);
            if (!upload(mem_b, zb.data(), b_bytes)) {
                return false;
            }
        }

        VkDescriptorSet ds = VK_NULL_HANDLE;
        std::vector<VkDescriptorBufferInfo> infos(4);
        infos[0] = { buf_in, 0, static_cast<VkDeviceSize>(in_bytes) };
        infos[1] = { buf_w,  0, static_cast<VkDeviceSize>(w_bytes) };
        infos[2] = { buf_b,  0, static_cast<VkDeviceSize>(b_bytes) };
        infos[3] = { buf_o,  0, static_cast<VkDeviceSize>(out_bytes) };
        if (!allocAndWriteDescriptorSet(dp, dsl, infos, ds)) {
            return false;
        }

        VkCommandBufferAllocateInfo cbai = {};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmd_pool_;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &cbai, &cmd) != VK_SUCCESS) {
            return false;
        }

        VkCommandBufferBeginInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            return false;
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ConvDims), &dims);
        const uint32_t gx = (dims.out_w + 7u) / 8u;
        const uint32_t gy = (dims.out_h + 7u) / 8u;
        const uint32_t gz = dims.out_c;
        vkCmdDispatch(cmd, gx, gy, gz);

        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            return false;
        }

        VkFence fence = VK_NULL_HANDLE;
        VkFenceCreateInfo fci = {};
        fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (vkCreateFence(device, &fci, nullptr, &fence) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            return false;
        }

        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        if (vkQueueSubmit(computeQueue, 1, &si, fence) != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            return false;
        }
        if (vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);
            return false;
        }

        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, cmd_pool_, 1, &cmd);

        void* mapped = nullptr;
        if (vkMapMemory(device, mem_o, 0, out_bytes, 0, &mapped) != VK_SUCCESS) {
            return false;
        }
        std::memcpy(out, mapped, out_bytes);
        vkUnmapMemory(device, mem_o);

        return true;
    }

    void releaseReusableBuffer(ReusableBuffer& reusable) {
        if (reusable.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, reusable.buffer, nullptr);
            reusable.buffer = VK_NULL_HANDLE;
        }
        if (reusable.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, reusable.memory, nullptr);
            reusable.memory = VK_NULL_HANDLE;
        }
        reusable.capacity = 0;
    }

    bool ensureReusableBuffer(ReusableBuffer& reusable, size_t required) {
        if (reusable.buffer != VK_NULL_HANDLE && reusable.capacity >= required) return true;

        releaseReusableBuffer(reusable);
        VkBufferCreateInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = required;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &bi, nullptr, &reusable.buffer) != VK_SUCCESS) return false;

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, reusable.buffer, &req);
        const uint32_t memory_type = findMemoryType(
            physicalDevice,
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (memory_type == UINT32_MAX) {
            releaseReusableBuffer(reusable);
            return false;
        }

        VkMemoryAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memory_type;
        if (vkAllocateMemory(device, &ai, nullptr, &reusable.memory) != VK_SUCCESS ||
            vkBindBufferMemory(device, reusable.buffer, reusable.memory, 0) != VK_SUCCESS) {
            releaseReusableBuffer(reusable);
            return false;
        }
        reusable.capacity = required;
        return true;
    }

    bool ensureAddKernel() {
        return ensureVectorKernel(add_ready_, add_dsl_, add_pl_, add_pipe_, add_dp_, find_shader_path_add(), 3, "add");
    }

    bool ensureMulKernel() {
        return ensureVectorKernel(mul_ready_, mul_dsl_, mul_pl_, mul_pipe_, mul_dp_, find_shader_path_mul(), 3, "mul");
    }

    bool ensureReluKernel() {
        return ensureVectorKernel(relu_ready_, relu_dsl_, relu_pl_, relu_pipe_, relu_dp_, find_shader_path_relu(), 2, "relu");
    }

    bool ensureSiluKernel() {
        return ensureVectorKernel(silu_ready_, silu_dsl_, silu_pl_, silu_pipe_, silu_dp_, find_shader_path_silu(), 2, "silu");
    }

    bool ensureGeluKernel() {
        return ensureVectorKernel(gelu_ready_, gelu_dsl_, gelu_pl_, gelu_pipe_, gelu_dp_, find_shader_path_gelu(), 2, "gelu");
    }

    bool ensureSigmoidKernel() {
        return ensureVectorKernel(sigmoid_ready_, sigmoid_dsl_, sigmoid_pl_, sigmoid_pipe_, sigmoid_dp_, find_shader_path_sigmoid(), 2, "sigmoid");
    }

    bool ensureTanhKernel() {
        return ensureVectorKernel(tanh_ready_, tanh_dsl_, tanh_pl_, tanh_pipe_, tanh_dp_, find_shader_path_tanh(), 2, "tanh");
    }

    bool ensureConv2dKernel() {
        return ensureVectorKernel(conv2d_ready_, conv2d_dsl_, conv2d_pl_, conv2d_pipe_, conv2d_dp_, find_shader_path_conv2d(), 4, "conv2d", sizeof(ConvDims));
    }

    bool ensureConvTranspose2dKernel() {
        return ensureVectorKernel(conv_transpose2d_ready_, conv_transpose2d_dsl_, conv_transpose2d_pl_, conv_transpose2d_pipe_, conv_transpose2d_dp_, find_shader_path_conv_transpose2d(), 4, "conv_transpose2d", sizeof(ConvDims));
    }

    void cleanupLinearKernel() {
        if (linear_dp_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, linear_dp_, nullptr);
            linear_dp_ = VK_NULL_HANDLE;
        }
        if (linear_pipe_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, linear_pipe_, nullptr);
            linear_pipe_ = VK_NULL_HANDLE;
        }
        if (linear_pl_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, linear_pl_, nullptr);
            linear_pl_ = VK_NULL_HANDLE;
        }
        if (linear_dsl_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, linear_dsl_, nullptr);
            linear_dsl_ = VK_NULL_HANDLE;
        }
        linear_ready_ = false;
    }

    void cleanupAddKernel() {
        cleanupVectorKernel(elementwise_ready_, elementwise_dsl_, elementwise_pl_, elementwise_pipe_, elementwise_dp_);
        cleanupVectorKernel(add_ready_, add_dsl_, add_pl_, add_pipe_, add_dp_);
    }

    void cleanupMulKernel() {
        cleanupVectorKernel(mul_ready_, mul_dsl_, mul_pl_, mul_pipe_, mul_dp_);
    }

    void cleanupReluKernel() {
        cleanupVectorKernel(relu_ready_, relu_dsl_, relu_pl_, relu_pipe_, relu_dp_);
    }

    void cleanupSiluKernel() {
        cleanupVectorKernel(silu_ready_, silu_dsl_, silu_pl_, silu_pipe_, silu_dp_);
    }

    void cleanupGeluKernel() {
        cleanupVectorKernel(gelu_ready_, gelu_dsl_, gelu_pl_, gelu_pipe_, gelu_dp_);
    }

    void cleanupSigmoidKernel() {
        cleanupVectorKernel(sigmoid_ready_, sigmoid_dsl_, sigmoid_pl_, sigmoid_pipe_, sigmoid_dp_);
    }

    void cleanupTanhKernel() {
        cleanupVectorKernel(tanh_ready_, tanh_dsl_, tanh_pl_, tanh_pipe_, tanh_dp_);
    }

    void cleanupConv2dKernel() {
        cleanupVectorKernel(conv2d_ready_, conv2d_dsl_, conv2d_pl_, conv2d_pipe_, conv2d_dp_);
    }

    void cleanupConvTranspose2dKernel() {
        cleanupVectorKernel(conv_transpose2d_ready_, conv_transpose2d_dsl_, conv_transpose2d_pl_, conv_transpose2d_pipe_, conv_transpose2d_dp_);
    }

    bool ensureLinearKernel() {
        std::lock_guard<std::recursive_mutex> lk(linear_mutex_);
        if (linear_ready_) return true;

        const auto shader_path = find_shader_path_linear();
        if (!shader_path.has_value()) {
            if (const char* v = std::getenv("MIMIR_ACCEL_VERBOSE")) {
                if (v[0] != '\0' && !(v[0] == '0' && v[1] == '\0')) {
                    std::cerr << "Vulkan linear shader not found. Set MIMIR_VULKAN_LINEAR_SPV or build shaders via glslangValidator.\n";
                }
            }
            return false;
        }

        std::vector<uint32_t> spirv = read_spirv_u32(*shader_path);
        if (spirv.empty()) {
            return false;
        }

        VkShaderModule shader = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo smci = {};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spirv.size() * sizeof(uint32_t);
        smci.pCode = spirv.data();
        if (vkCreateShaderModule(device, &smci, nullptr, &shader) != VK_SUCCESS) {
            return false;
        }

        // Descriptor set layout (4 storage buffers)
        VkDescriptorSetLayoutBinding bindings[4] = {};
        for (uint32_t i = 0; i < 4; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci = {};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 4;
        dslci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &dslci, nullptr, &linear_dsl_) != VK_SUCCESS) {
            vkDestroyShaderModule(device, shader, nullptr);
            return false;
        }

        VkPushConstantRange pcr = {};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(LinearDims);

        VkPipelineLayoutCreateInfo plci = {};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &linear_dsl_;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(device, &plci, nullptr, &linear_pl_) != VK_SUCCESS) {
            vkDestroyShaderModule(device, shader, nullptr);
            return false;
        }

        VkPipelineShaderStageCreateInfo stage = {};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpci = {};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = linear_pl_;
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &linear_pipe_) != VK_SUCCESS) {
            vkDestroyShaderModule(device, shader, nullptr);
            return false;
        }
        vkDestroyShaderModule(device, shader, nullptr);

        // Descriptor pool
        VkDescriptorPoolSize ps = {};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = 4;
        VkDescriptorPoolCreateInfo dpci = {};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(device, &dpci, nullptr, &linear_dp_) != VK_SUCCESS) {
            return false;
        }

        linear_ready_ = true;
        return true;
    }

    bool allocAndWriteLinearDescriptorSet(
        VkBuffer inB, size_t inBytes,
        VkBuffer wB,  size_t wBytes,
        VkBuffer bB,  size_t bBytes,
        VkBuffer oB,  size_t oBytes,
        VkDescriptorSet& outDS
    ) {
        if (!linear_dp_ || !linear_dsl_) return false;

        // Important: on réutilise le même descriptor pool à chaque appel.
        // Sans reset/free, on épuise le pool après 1 allocation (maxSets=1) et
        // on retombe en fallback CPU après avoir déjà payé le coût des buffers.
        // Ici c'est safe car on attend la fence (pas d'utilisation en vol).
        vkResetDescriptorPool(device, linear_dp_, 0);

        VkDescriptorSetAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = linear_dp_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &linear_dsl_;
        if (vkAllocateDescriptorSets(device, &ai, &outDS) != VK_SUCCESS) {
            return false;
        }

        VkDescriptorBufferInfo infos[4] = {};
        infos[0].buffer = inB; infos[0].offset = 0; infos[0].range = static_cast<VkDeviceSize>(inBytes);
        infos[1].buffer = wB;  infos[1].offset = 0; infos[1].range = static_cast<VkDeviceSize>(wBytes);
        infos[2].buffer = bB;  infos[2].offset = 0; infos[2].range = static_cast<VkDeviceSize>(bBytes);
        infos[3].buffer = oB;  infos[3].offset = 0; infos[3].range = static_cast<VkDeviceSize>(oBytes);

        VkWriteDescriptorSet writes[4] = {};
        for (uint32_t i = 0; i < 4; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = outDS;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
        return true;
    }
};

// Buffer pour calculs GPU
class ComputeBuffer {
private:
    VkDevice device;
    VkBuffer buffer;
    VkDeviceMemory memory;
    size_t size;
    
public:
    ComputeBuffer(VkDevice dev) : device(dev), buffer(VK_NULL_HANDLE),
                                  memory(VK_NULL_HANDLE), size(0) {}
    
    ~ComputeBuffer() {
        cleanup();
    }
    
    bool allocate(VkPhysicalDevice physicalDevice, size_t byteSize, VkBufferUsageFlags usage) {
        size = byteSize;
        
        // Create buffer
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            return false;
        }
        
        // Get memory requirements
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        
        // Find suitable memory type
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        
        uint32_t memoryTypeIndex = UINT32_MAX;
        VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if ((memRequirements.memoryTypeBits & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                memoryTypeIndex = i;
                break;
            }
        }
        
        if (memoryTypeIndex == UINT32_MAX) {
            return false;
        }
        
        // Allocate memory
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = memoryTypeIndex;
        
        if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
            return false;
        }
        
        // Bind buffer to memory
        vkBindBufferMemory(device, buffer, memory, 0);
        
        return true;
    }
    
    void cleanup() {
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }
    
    bool upload(const void* data, size_t offset = 0) {
        void* mapped;
        if (vkMapMemory(device, memory, offset, size, 0, &mapped) != VK_SUCCESS) {
            return false;
        }
        std::memcpy(mapped, data, size);
        vkUnmapMemory(device, memory);
        return true;
    }
    
    bool download(void* data, size_t offset = 0) {
        void* mapped;
        if (vkMapMemory(device, memory, offset, size, 0, &mapped) != VK_SUCCESS) {
            return false;
        }
        std::memcpy(data, mapped, size);
        vkUnmapMemory(device, memory);
        return true;
    }
    
    VkBuffer getBuffer() const { return buffer; }
    size_t getSize() const { return size; }
};

// Shader compute SPIR-V
class ComputeShader {
private:
    VkDevice device;
    VkShaderModule shaderModule;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    
public:
    ComputeShader(VkDevice dev) : device(dev), shaderModule(VK_NULL_HANDLE),
                                  pipeline(VK_NULL_HANDLE), pipelineLayout(VK_NULL_HANDLE),
                                  descriptorSetLayout(VK_NULL_HANDLE),
                                  descriptorPool(VK_NULL_HANDLE),
                                  descriptorSet(VK_NULL_HANDLE) {}
    
    ~ComputeShader() {
        cleanup();
    }
    
    bool loadFromSPIRV(const std::vector<uint32_t>& spirv) {
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = spirv.size() * sizeof(uint32_t);
        createInfo.pCode = spirv.data();
        
        return vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) == VK_SUCCESS;
    }
    
    void cleanup() {
        if (descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        }
        if (descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        }
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        if (pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        }
        if (shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shaderModule, nullptr);
        }
    }
};

} // namespace VulkanCompute

#endif // __VULKAN_COMPUTE_HPP__
