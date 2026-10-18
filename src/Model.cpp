#include "runtimes/tpu/TpuRuntime.hpp"
#include "runtimes/fpga/FpgaSession.hpp"
#include "runtimes/fpga/FpgaRuntime.hpp"
#include "Model.hpp"
#include "runtimes/optimizers/OptimizerDispatch.hpp"
#include "viz/ModelGraph.hpp"
#include "training/BBoxSupervision.hpp"
#include "HardwareOpt.hpp"
#include "SIMD_Ops.hpp"
#include "Layers.hpp"
#include "LayerTypes.hpp"
#include "LayerOps.hpp"
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#ifdef ENABLE_VULKAN
#include "runtimes/vulkan/VulkanRuntime.hpp"
#endif
#ifdef ENABLE_OPENCL
#include "runtimes/opencl/OpenCLRuntime.hpp"
#endif
#include "runtimes/AbstractRuntime.hpp"
#include "runtimes/LayerOps.hpp"
#include "runtimes/RuntimeRouter.hpp"
#ifdef ENABLE_CUDA
#include "runtimes/cuda/CudaRuntime.hpp"
#endif
#ifdef ENABLE_ROCM
#include "runtimes/rocm/RocmRuntime.hpp"
#endif
#include "RngContext.hpp"
#include "RuntimeAllocator.hpp"
#include "LayerOpsExt.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/ops_loss_and_grad.hpp"
#include "Planning/Planner.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Models/NLP/CausalAttentionOps.hpp"
#include "Serialization/Serialization.hpp"
#include <fstream>
#include <iomanip>
#include <ctime>
#include <iostream>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <cstdlib>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <unordered_map>
#include <mutex>
#include <atomic>

void framework_log_write_file_only(const char* data, size_t size);

// Racine de capture par thread. Elle reste active jusqu'a consumeVizTaps(),
// ce qui permet aux sous-modeles executes apres le forward principal (VAE,
// perception, discriminateur...) de publier dans la meme VIZ.
static thread_local Model* g_viz_capture_root = nullptr;

#ifdef _OPENMP
#include <omp.h>
#endif


namespace {
static inline long long omp_work_threshold() {
#ifdef _OPENMP
    // Seuil adaptatif modéré: on amortit le runtime OpenMP sans basculer trop vite en séquentiel.
    const int nt = std::max(1, omp_get_max_threads());
    const long long base = 262144LL;
    long long factor = 1LL;
    if (nt >= 4) factor = 2LL;
    if (nt >= 12) factor = 3LL;
    if (nt >= 24) factor = 4LL;
    return base * factor;
#else
    return 262144LL;
#endif
}
} // namespace
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <random>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <atomic>
#include <mutex>
#include <unordered_set>

#if defined(_MSC_VER)
#define MIMIR_RESTRICT __restrict
#else
#define MIMIR_RESTRICT __restrict__
#endif

// ============================================================================
// Registry centralisé (via LayerTypes.hpp)
// ============================================================================

using namespace LayerRegistry;

// ============================================================================
// Implémentation des méthodes Layer
// ============================================================================

// ============================================================================
// Détection des capacités CPU au runtime
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
static inline bool mimir_get_cpuid(unsigned int leaf, unsigned int* eax, unsigned int* ebx,
                                   unsigned int* ecx, unsigned int* edx) {
#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, static_cast<int>(leaf), 0);
    *eax = static_cast<unsigned int>(regs[0]);
    *ebx = static_cast<unsigned int>(regs[1]);
    *ecx = static_cast<unsigned int>(regs[2]);
    *edx = static_cast<unsigned int>(regs[3]);
    return true;
#else
    return __get_cpuid(leaf, eax, ebx, ecx, edx) != 0;
#endif
}

static inline bool mimir_get_cpuid_count(unsigned int leaf, unsigned int subleaf, unsigned int* eax,
                                         unsigned int* ebx, unsigned int* ecx, unsigned int* edx) {
#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    *eax = static_cast<unsigned int>(regs[0]);
    *ebx = static_cast<unsigned int>(regs[1]);
    *ecx = static_cast<unsigned int>(regs[2]);
    *edx = static_cast<unsigned int>(regs[3]);
    return true;
#else
    return __get_cpuid_count(leaf, subleaf, eax, ebx, ecx, edx) != 0;
#endif
}

static inline bool mimir_os_supports_avx_state() {
    // Vérifie que l'OS a activé le sauvegarde/restauration XMM/YMM (AVX).
    // Nécessaire pour utiliser AVX/AVX2/FMA/F16C sans #UD.
    unsigned int eax, ebx, ecx, edx;
    if (!mimir_get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;

    const bool osxsave = (ecx & (1u << 27)) != 0;
    const bool avx_hw = (ecx & (1u << 28)) != 0;
    if (!osxsave || !avx_hw) return false;

    // XGETBV(0): bits 1 (XMM) et 2 (YMM) doivent être à 1
#if defined(_MSC_VER)
    const unsigned __int64 xcr0 = _xgetbv(0);
    return (xcr0 & 0x6ull) == 0x6ull;
#else
    uint32_t xcr0_lo = 0;
    uint32_t xcr0_hi = 0;
    __asm__ volatile ("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    (void)xcr0_hi;
    return (xcr0_lo & 0x6u) == 0x6u;
#endif
}
#endif

bool Model::hasAVX2() {
    static bool detected = false;
    static bool result = false;
    
    if (!detected) {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            unsigned int eax, ebx, ecx, edx;
            const bool os_ok = mimir_os_supports_avx_state();
            if (os_ok && mimir_get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                result = (ebx & (1u << 5)) != 0; // EBX bit 5 = AVX2
            } else {
                result = false;
            }
        #endif
        detected = true;
    }
    
    return result;
}

bool Model::hasFMA() {
    static bool detected = false;
    static bool result = false;
    
    if (!detected) {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            unsigned int eax, ebx, ecx, edx;
            const bool os_ok = mimir_os_supports_avx_state();
            if (os_ok && mimir_get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                result = (ecx & (1u << 12)) != 0; // ECX bit 12 = FMA
            } else {
                result = false;
            }
        #endif
        detected = true;
    }
    
    return result;
}

bool Model::hasF16C() {
    static bool detected = false;
    static bool result = false;
    
    if (!detected) {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            unsigned int eax, ebx, ecx, edx;
            const bool os_ok = mimir_os_supports_avx_state();
            if (os_ok && mimir_get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
                result = (ecx & (1u << 29)) != 0; // ECX bit 29 = F16C
            } else {
                result = false;
            }
        #endif
        detected = true;
    }
    
    return result;
}

bool Model::hasBMI2() {
    static bool detected = false;
    static bool result = false;
    
    if (!detected) {
        #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
            unsigned int eax, ebx, ecx, edx;
            if (mimir_get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                result = (ebx & (1 << 8)) != 0; // EBX bit 8 = BMI2
            }
        #endif
        detected = true;
    }
    
    return result;
}

// Global compute engine (initialized on demand)
#ifdef ENABLE_VULKAN
static std::unique_ptr<VulkanRuntime> g_compute_engine = nullptr;
#endif
static bool g_compute_available = false;
static bool g_suppress_framework_logs = false;

// OpenCL compute engine (initialized on demand)
#ifdef ENABLE_OPENCL
static std::unique_ptr<OpenCLRuntime> g_opencl_engine = nullptr;
#endif
static bool g_opencl_available = false;

// CUDA compute engine (initialized on demand)
#ifdef ENABLE_CUDA
static std::unique_ptr<CudaRuntime> g_cuda_engine = nullptr;
#endif
static bool g_cuda_available = false;

// ROCm compute engine (initialized on demand)
#ifdef ENABLE_ROCM
static std::unique_ptr<RocmRuntime> g_rocm_engine = nullptr;
#endif
static bool g_rocm_available = false;

// CPU runtime (always available unless explicitly disabled)
static std::unique_ptr<CpuRuntime> g_cpu_engine = nullptr;
static bool g_cpu_available = false;

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {
static inline bool env_flag_true(const char* name, bool default_value) {
    if (!name) return default_value;
    const char* v = std::getenv(name);
    if (!v) return default_value;
    if (v[0] == '\0') return default_value;
    if (v[0] == '0' && v[1] == '\0') return false;
    // tout autre valeur non vide => true
    return true;
}

static inline int env_int(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') return default_value;
    try {
        return std::stoi(v);
    } catch (...) {
        return default_value;
    }
}

static inline bool any_runtime_fastpath_enabled(const RuntimeConfig& cfg) {
    return cfg.linear_enabled || cfg.conv_enabled || cfg.norm_enabled || cfg.attention_enabled;
}

static inline void refresh_runtime_router_bindings() {
    AbstractRuntime* rocm = nullptr;
    AbstractRuntime* cuda = nullptr;
    AbstractRuntime* vulkan = nullptr;
    AbstractRuntime* opencl = nullptr;
    AbstractRuntime* cpu = nullptr;

#ifdef ENABLE_ROCM
    rocm = g_rocm_engine.get();
#endif
#ifdef ENABLE_CUDA
    cuda = g_cuda_engine.get();
#endif
#ifdef ENABLE_VULKAN
    vulkan = g_compute_engine.get();
#endif
#ifdef ENABLE_OPENCL
    opencl = g_opencl_engine.get();
#endif
    cpu = g_cpu_engine.get();

    RuntimeRouter::instance().setRuntimes(rocm, cuda, vulkan, opencl, Mimir::tpuSessionRuntime(), cpu);
}
}

// === constructeurs / destructeurs (déjà présents) ===
Model::Model()
    : tokenizer(20000), encoder(256, 20000), hasTokenizer(true), hasEncoder(true),
    max_ram_mb_(0)
{
    RuntimeRouter::instance().setActivators(
        [this]() -> bool { return this->initializeRocmComputeEngine(); },
        [this]() -> bool { return this->initializeCudaComputeEngine(); },
        [this]() -> bool { return this->initializeComputeEngine(); },
        [this]() -> bool { return this->initializeOpenCLComputeEngine(); },
        [this]() -> bool { return this->initializeCpuComputeEngine(); }
    );

    tw = 64; th = 64;
    // ConditioningEncoder toujours présent + embeddings spéciaux (SEQ/MOD/MAG) disponibles.
    encoder.ensureSpecialEmbeddings();
    Mimir::initializeFpgaSession();
    // Tenter d'initialiser les runtimes disponibles
    RuntimeRouter::instance().activateAvailableRuntimes();
}

Model::~Model() = default;

void Model::setDefaultDType(const std::string& dtype) {
    const auto dt = Mimir::parse_dtype(dtype);
    if (dt == Mimir::DType::UNKNOWN) {
        throw std::runtime_error("Model.setDefaultDType: dtype non supporté: '" + dtype + "'");
    }
    default_dtype_ = Mimir::dtype_to_string(dt);
    const auto accumulation = (dt == Mimir::DType::F64)
        ? Mimir::DType::F64
        : Mimir::DType::F32;
    for (auto& layer : layers) {
        layer.dtype = dt;
        layer.accumulation_dtype = accumulation;
    }
    if (!Model::frameworkLogsSuppressed()) {
        std::cerr << "[dtype] default=" << default_dtype_
                  << " accumulation=" << Mimir::dtype_to_string(accumulation) << std::endl;
    }
    // Keep a canonical copy in config for serialization/planner/Lua.
    try {
        modelConfig["dtype"] = default_dtype_;
    } catch (...) {
    }
}

// ===== Hardware Acceleration =====

bool Model::hasVulkanCompute() const {
    return g_compute_available;
}

void Model::setFrameworkLogsSuppressed(bool enable) {
    g_suppress_framework_logs = enable;
}

bool Model::frameworkLogsSuppressed() {
    return g_suppress_framework_logs;
}

bool Model::hasOpenCLCompute() const {
    return g_opencl_available;
}

bool Model::hasCudaCompute() const {
    return g_cuda_available;
}

bool Model::hasRocmCompute() const {
    return g_rocm_available;
}

bool Model::hasCpuCompute() const {
    return g_cpu_available;
}

bool Model::initializeCpuComputeEngine() {
    // Pattern init_once thread-safe avec atomic
    static std::atomic<bool> initialized{false};
    static std::mutex init_mutex;

    if (initialized.load(std::memory_order_acquire)) {
        return g_cpu_available;
    }

    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized.load(std::memory_order_relaxed)) {
        return g_cpu_available;
    }

    const RuntimeConfig cfg_from_env = RuntimeConfig::fromEnv("CPU");
        if (cfg_from_env.disabled) {
            g_cpu_available = false;
            g_cpu_engine.reset();
        refresh_runtime_router_bindings();
        initialized.store(true, std::memory_order_release);
        if (cfg_from_env.verbose && !Model::frameworkLogsSuppressed()) {
            std::cerr << "⚠ CPU runtime disabled via MIMIR_DISABLE_CPU" << std::endl;
        }
        return false;
    }

    // CPU est le fallback de base; on active Linear par défaut.
    RuntimeConfig cfg = cfg_from_env;
    cfg.linear_enabled = env_flag_true("MIMIR_CPU_LINEAR", true);
    cfg.linear_min_ops = env_int("MIMIR_CPU_LINEAR_MIN_OPS", 0);

    try {
        g_cpu_engine = std::make_unique<CpuRuntime>();
        g_cpu_available = g_cpu_engine->initialize(cfg);
        refresh_runtime_router_bindings();
        if (g_cpu_available && cfg.verbose && !Model::frameworkLogsSuppressed()) {
            std::cerr << "✓ CPU Runtime initialized" << std::endl;
        }
        if (!g_cpu_available) {
            g_cpu_engine.reset();
            refresh_runtime_router_bindings();
        }
    } catch (const std::exception& e) {
        if (!Model::frameworkLogsSuppressed()) {
            std::cerr << "⚠ CPU Runtime unavailable: " << e.what() << std::endl;
        }
        g_cpu_available = false;
        g_cpu_engine.reset();
        refresh_runtime_router_bindings();
    }

    initialized.store(true, std::memory_order_release);
    return g_cpu_available;
}

bool Model::initializeComputeEngine() {
#ifndef ENABLE_VULKAN
    g_compute_available = false;
    return false;
#else
    // Pattern init_once thread-safe avec atomic
    static std::atomic<bool> initialized{false};
    static std::mutex init_mutex;

    // Permet de forcer le mode CPU pour diagnostic/stabilité.
    // Toute valeur non vide et différente de "0" désactive Vulkan.
    if (const char* v = std::getenv("MIMIR_DISABLE_VULKAN")) {
        if (v[0] != '\0' && !(v[0] == '0' && v[1] == '\0')) {
            g_compute_available = false;
            g_compute_engine.reset();
            refresh_runtime_router_bindings();
            initialized.store(true, std::memory_order_release);
            if (!Model::frameworkLogsSuppressed()) {
                std::cerr << "⚠ Vulkan Compute disabled via MIMIR_DISABLE_VULKAN" << std::endl;
            }
            return false;
        }
    }
    
    if (initialized.load(std::memory_order_acquire)) {
        return g_compute_available;
    }
    
    std::lock_guard<std::mutex> lock(init_mutex);
    
    // Double-check après lock
    if (initialized.load(std::memory_order_relaxed)) {
        return g_compute_available;
    }

    RuntimeConfig cfg = RuntimeConfig::fromEnv("VULKAN");
    cfg.linear_enabled = env_flag_true("MIMIR_VULKAN_LINEAR", true);
    cfg.linear_min_ops = env_int("MIMIR_VULKAN_LINEAR_MIN_OPS", 0);
    cfg.conv_enabled = env_flag_true("MIMIR_VULKAN_CONV", true);
    cfg.conv_min_ops = env_int("MIMIR_VULKAN_CONV_MIN_OPS", 0);
    if (cfg.disabled || !any_runtime_fastpath_enabled(cfg)) {
        g_compute_available = false;
        g_compute_engine.reset();
        refresh_runtime_router_bindings();
        initialized.store(true, std::memory_order_release);
        return false;
    }

    try {
        g_compute_engine = std::make_unique<VulkanRuntime>();
        g_compute_available = g_compute_engine->initialize(cfg);
        refresh_runtime_router_bindings();

        if (g_compute_available) {
            if (!Model::frameworkLogsSuppressed()) {
            std::cerr << "✓ Vulkan Compute initialized" << std::endl;
            }
        } else {
            if (!Model::frameworkLogsSuppressed()) {
                std::cerr << "⚠ Vulkan Compute initialization failed, using CPU fallback" << std::endl;
            }
            g_compute_engine.reset();
            refresh_runtime_router_bindings();
        }
    } catch (const std::exception& e) {
        if (!Model::frameworkLogsSuppressed()) {
            std::cerr << "⚠ Vulkan Compute unavailable: " << e.what() << std::endl;
        }
        g_compute_available = false;
        g_compute_engine.reset();
        refresh_runtime_router_bindings();
    }
    
    initialized.store(true, std::memory_order_release);
    return g_compute_available;
#endif
}

bool Model::initializeOpenCLComputeEngine() {
#ifndef ENABLE_OPENCL
    g_opencl_available = false;
    return false;
#else
    static std::atomic<bool> initialized{false};
    static std::mutex init_mutex;

    // Toute valeur non vide et différente de "0" désactive OpenCL.
    if (const char* v = std::getenv("MIMIR_DISABLE_OPENCL")) {
        if (v[0] != '\0' && !(v[0] == '0' && v[1] == '\0')) {
            g_opencl_available = false;
            g_opencl_engine.reset();
            refresh_runtime_router_bindings();
            initialized.store(true, std::memory_order_release);
            std::cerr << "⚠ OpenCL Compute disabled via MIMIR_DISABLE_OPENCL" << std::endl;
            return false;
        }
    }

    if (initialized.load(std::memory_order_acquire)) {
            return g_opencl_available; 
    }

    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized.load(std::memory_order_relaxed)) {
        return g_opencl_available;
    }

    RuntimeConfig cfg = RuntimeConfig::fromEnv("OPENCL");
    cfg.linear_enabled = env_flag_true("MIMIR_OPENCL_LINEAR", true);
    cfg.linear_min_ops = env_int("MIMIR_OPENCL_LINEAR_MIN_OPS", 0);
    if (cfg.disabled || !cfg.linear_enabled) {
        g_opencl_available = false;
        g_opencl_engine.reset();
        refresh_runtime_router_bindings();
        initialized.store(true, std::memory_order_release);
        return false;
    }

    try {
        g_opencl_engine = std::make_unique<OpenCLRuntime>();
        g_opencl_available = g_opencl_engine->initialize(cfg);
        refresh_runtime_router_bindings();
        if (g_opencl_available) {
            std::cerr << "✓ OpenCL Compute initialized" << std::endl;
        } else {
            if (env_flag_true("MIMIR_ACCEL_VERBOSE", false)) {
                std::cerr << "⚠ OpenCL Compute unavailable, using CPU fallback" << std::endl;
            }
            g_opencl_engine.reset();
            refresh_runtime_router_bindings();
        }
    } catch (const std::exception& e) {
        std::cerr << "⚠ OpenCL Compute unavailable: " << e.what() << std::endl;
        g_opencl_available = false;
        g_opencl_engine.reset();
        refresh_runtime_router_bindings();
    }

    initialized.store(true, std::memory_order_release);
    return g_opencl_available;
#endif
}

bool Model::initializeCudaComputeEngine() {
#ifndef ENABLE_CUDA
    g_cuda_available = false;
    return false;
#else
    static std::atomic<bool> initialized{false};
    static std::mutex init_mutex;

    RuntimeConfig cfg = RuntimeConfig::fromEnv("CUDA");
    cfg.linear_enabled = env_flag_true("MIMIR_CUDA_LINEAR", true);
    cfg.linear_min_ops = env_int("MIMIR_CUDA_LINEAR_MIN_OPS", 0);
    cfg.conv_enabled = env_flag_true("MIMIR_CUDA_CONV", true);
    cfg.conv_min_ops = env_int("MIMIR_CUDA_CONV_MIN_OPS", 0);
    cfg.norm_enabled = env_flag_true("MIMIR_CUDA_NORM", true);
    cfg.norm_min_elements = env_int("MIMIR_CUDA_NORM_MIN_ELEMS", 0);
    cfg.attention_enabled = env_flag_true("MIMIR_CUDA_ATTENTION", true);
    cfg.attention_min_ops = env_int("MIMIR_CUDA_ATTENTION_MIN_OPS", 0);

    if (cfg.disabled || !any_runtime_fastpath_enabled(cfg)) {
        g_cuda_available = false;
        g_cuda_engine.reset();
        refresh_runtime_router_bindings();
        initialized.store(true, std::memory_order_release);
        if (cfg.verbose && cfg.disabled) {
            std::cerr << "⚠ CUDA Compute disabled via MIMIR_DISABLE_CUDA" << std::endl;
        }
        return false;
    }

    if (initialized.load(std::memory_order_acquire)) {
        return g_cuda_available;
    }

    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized.load(std::memory_order_relaxed)) {
        return g_cuda_available;
    }

    try {
        g_cuda_engine = std::make_unique<CudaRuntime>();
        g_cuda_available = g_cuda_engine->initialize(cfg);
        refresh_runtime_router_bindings();
        if (g_cuda_available) {
            if (cfg.verbose) {
                std::cerr << "✓ CUDA Compute initialized" << std::endl;
            }
        } else {
            g_cuda_engine.reset();
            refresh_runtime_router_bindings();
        }
    } catch (const std::exception& e) {
        std::cerr << "⚠ CUDA Compute unavailable: " << e.what() << std::endl;
        g_cuda_available = false;
        g_cuda_engine.reset();
        refresh_runtime_router_bindings();
    }

    initialized.store(true, std::memory_order_release);
    return g_cuda_available;
#endif
}

bool Model::initializeRocmComputeEngine() {
#ifndef ENABLE_ROCM
    g_rocm_available = false;
    return false;
#else
    static std::atomic<bool> initialized{false};
    static std::mutex init_mutex;

    RuntimeConfig cfg = RuntimeConfig::fromEnv("ROCM");
    cfg.linear_enabled = env_flag_true("MIMIR_ROCM_LINEAR", true);
    cfg.linear_min_ops = env_int("MIMIR_ROCM_LINEAR_MIN_OPS", 0);
    cfg.conv_enabled = env_flag_true("MIMIR_ROCM_CONV", true);
    cfg.conv_min_ops = env_int("MIMIR_ROCM_CONV_MIN_OPS", 0);
    cfg.norm_enabled = env_flag_true("MIMIR_ROCM_NORM", true);
    cfg.norm_min_elements = env_int("MIMIR_ROCM_NORM_MIN_ELEMS", 0);
    cfg.attention_enabled = env_flag_true("MIMIR_ROCM_ATTENTION", true);
    cfg.attention_min_ops = env_int("MIMIR_ROCM_ATTENTION_MIN_OPS", 0);

    if (cfg.disabled || !any_runtime_fastpath_enabled(cfg)) {
        g_rocm_available = false;
        g_rocm_engine.reset();
        refresh_runtime_router_bindings();
        initialized.store(true, std::memory_order_release);
        if (cfg.verbose && cfg.disabled) {
            std::cerr << "⚠ ROCm Compute disabled via MIMIR_DISABLE_ROCM" << std::endl;
        }
        return false;
    }

    if (initialized.load(std::memory_order_acquire)) {
        return g_rocm_available;
    }

    std::lock_guard<std::mutex> lock(init_mutex);
    if (initialized.load(std::memory_order_relaxed)) {
        return g_rocm_available;
    }

    try {
        g_rocm_engine = std::make_unique<RocmRuntime>();
        g_rocm_available = g_rocm_engine->initialize(cfg);
        refresh_runtime_router_bindings();
        if (g_rocm_available) {
            if (cfg.verbose) {
                std::cerr << "✓ ROCm Compute initialized" << std::endl;
            }
        } else {
            g_rocm_engine.reset();
            refresh_runtime_router_bindings();
        }
    } catch (const std::exception& e) {
        std::cerr << "⚠ ROCm Compute unavailable: " << e.what() << std::endl;
        g_rocm_available = false;
        g_rocm_engine.reset();
        refresh_runtime_router_bindings();
    }

    initialized.store(true, std::memory_order_release);
    return g_rocm_available;
#endif
}

void Model::shutdownComputeEngine() {
#ifdef ENABLE_VULKAN
    if (g_compute_engine) {
        g_compute_engine->shutdown();
        g_compute_engine.reset();
        g_compute_available = false;
        refresh_runtime_router_bindings();
    }
#else
    g_compute_available = false;
#endif
}

void Model::shutdownOpenCLComputeEngine() {
#ifdef ENABLE_OPENCL
    if (g_opencl_engine) {
        g_opencl_engine->shutdown();
        g_opencl_engine.reset();
        g_opencl_available = false;
        refresh_runtime_router_bindings();
    }
#else
    g_opencl_available = false;
#endif
}

void Model::shutdownCudaComputeEngine() {
#ifdef ENABLE_CUDA
    if (g_cuda_engine) {
        g_cuda_engine->shutdown();
        g_cuda_engine.reset();
        g_cuda_available = false;
        refresh_runtime_router_bindings();
    }
#else
    g_cuda_available = false;
#endif
}

void Model::shutdownRocmComputeEngine() {
#ifdef ENABLE_ROCM
    if (g_rocm_engine) {
        g_rocm_engine->shutdown();
        g_rocm_engine.reset();
        g_rocm_available = false;
        refresh_runtime_router_bindings();
    }
#else
    g_rocm_available = false;
#endif
}

void Model::shutdownCpuComputeEngine() {
    if (g_cpu_engine) {
        g_cpu_engine->shutdown();
        g_cpu_engine.reset();
        g_cpu_available = false;
        refresh_runtime_router_bindings();
    } else {
        g_cpu_available = false;
        refresh_runtime_router_bindings();
    }
}

void Model::zeroGradients() {
    if (params_frozen_) {
        throw std::runtime_error("Model::zeroGradients: parameters are frozen");
    }
    // Réinitialiser tous les gradients des layers à zéro
    for (auto& layer : layers) {
        std::fill(layer.grad_weights.begin(), layer.grad_weights.end(), 0.0f);
        std::fill(layer.grad_bias.begin(), layer.grad_bias.end(), 0.0f);
    }
    
    // Réinitialiser l'état du forward pour le prochain backward
    forward_state.clear();
}

void Model::releaseTrainingWorkingSet(size_t completed_step) {
    forward_state.clear();
    clearTensorStore();
    clearTensorStoreInt();

#if defined(__GLIBC__)
    static const size_t trim_every = []() {
        const char* value = std::getenv("MIMIR_MALLOC_TRIM_EVERY");
        if (!value || *value == '\0') return size_t{1};
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        return (end != value) ? static_cast<size_t>(parsed) : size_t{1};
    }();
    if (trim_every > 0 && completed_step % trim_every == 0) {
        malloc_trim(0);
    }
#else
    (void)completed_step;
#endif
}

Gradients Model::getGradients() const {
    Gradients grads;
    
    // Collecter tous les gradients des layers
    size_t param_idx = 0;
    for (const auto& layer : layers) {
        // Ajouter les gradients de poids
        for (const auto& grad : layer.grad_weights) {
            grads.param_grads[param_idx++] = grad;
        }
        
        // Ajouter les gradients de biais
        for (const auto& grad : layer.grad_bias) {
            grads.param_grads[param_idx++] = grad;
        }
    }
    
    return grads;
}

// ============================================================================
// TENSOR STORE (Multi-input/Branch Support)
// ============================================================================

const std::vector<float>& Model::getTensor(const std::string& name) const {
    auto it = tensor_store.find(name);
    if (it == tensor_store.end()) {
        std::cerr << "❌ ERROR: Tensor '" << name << "' not found in TensorStore" << std::endl;
        std::cerr << "Available tensors: ";
        for (const auto& kv : tensor_store) {
            std::cerr << "'" << kv.first << "' ";
        }
        std::cerr << std::endl;
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

bool Model::hasTensor(const std::string& name) const {
    return tensor_store.find(name) != tensor_store.end();
}

const Mimir::TypedTensor& Model::getTypedTensor(const std::string& name) const {
    auto it = typed_tensor_store.find(name);
    if (it == typed_tensor_store.end())
        throw std::runtime_error("Typed tensor not found: " + name);
    return it->second;
}

bool Model::hasTypedTensor(const std::string& name) const {
    return typed_tensor_store.find(name) != typed_tensor_store.end();
}

const std::vector<int>& Model::getTensorInt(const std::string& name) const {
    auto it = tensor_store_int.find(name);
    if (it == tensor_store_int.end()) {
        std::cerr << "❌ ERROR: IntTensor '" << name << "' not found in IntTensorStore" << std::endl;
        std::cerr << "Available int tensors: ";
        for (const auto& kv : tensor_store_int) {
            std::cerr << "'" << kv.first << "' ";
        }
        std::cerr << std::endl;
        throw std::runtime_error("Int tensor not found: " + name);
    }
    return it->second;
}

bool Model::hasTensorInt(const std::string& name) const {
    return tensor_store_int.find(name) != tensor_store_int.end();
}

std::vector<int>& Model::getTensorIntMutable(const std::string& name) {
    auto it = tensor_store_int.find(name);
    if (it == tensor_store_int.end()) {
        std::cerr << "❌ ERROR: IntTensor '" << name << "' not found in IntTensorStore" << std::endl;
        std::cerr << "Available int tensors: ";
        for (const auto& kv : tensor_store_int) {
            std::cerr << "'" << kv.first << "' ";
        }
        std::cerr << std::endl;
        throw std::runtime_error("Int tensor not found: " + name);
    }
    return it->second;
}

std::vector<float>& Model::getTensorMutable(const std::string& name) {
    auto it = tensor_store.find(name);
    if (it == tensor_store.end()) {
        std::cerr << "❌ ERROR: Tensor '" << name << "' not found in TensorStore" << std::endl;
        std::cerr << "Available tensors: ";
        for (const auto& kv : tensor_store) {
            std::cerr << "'" << kv.first << "' ";
        }
        std::cerr << std::endl;
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

void Model::storeTensor(const std::string& name, const std::vector<float>& data) {
    tensor_store[name] = data;
    const auto dtype = Mimir::parse_dtype(default_dtype_);
    typed_tensor_store.insert_or_assign(
        name, Mimir::TypedTensor::fromFloat32(data, {static_cast<int>(data.size())}, dtype));
}

void Model::storeTensorInt(const std::string& name, const std::vector<int>& data) {
    tensor_store_int[name] = data;
}

void Model::storeTensor(const std::string& name, std::vector<float>&& data) {
    const auto dtype = Mimir::parse_dtype(default_dtype_);
    typed_tensor_store.insert_or_assign(
        name, Mimir::TypedTensor::fromFloat32(data, {static_cast<int>(data.size())}, dtype));
    tensor_store[name] = std::move(data);
}

void Model::storeTensorInt(const std::string& name, std::vector<int>&& data) {
    tensor_store_int[name] = std::move(data);
}

std::vector<std::string> Model::getAvailableTensors() const {
    std::vector<std::string> names;
    names.reserve(tensor_store.size());
    for (const auto& kv : tensor_store) {
        names.push_back(kv.first);
    }
    return names;
}

std::vector<std::string> Model::getAvailableIntTensors() const {
    std::vector<std::string> names;
    names.reserve(tensor_store_int.size());
    for (const auto& kv : tensor_store_int) {
        names.push_back(kv.first);
    }
    return names;
}

void Model::clearTensorStore() {
    tensor_store.clear();
    typed_tensor_store.clear();
}

void Model::clearTensorStoreInt() {
    tensor_store_int.clear();
}

void Model::setKVCacheEnabled(bool enabled) {
    kv_cache_enabled_ = enabled;
    if (!enabled) {
        clearKVCache();
    }
}

void Model::clearKVCache() {
    kv_cache_by_layer_.clear();
}

size_t Model::getKVCacheTokenCount() const {
    size_t total = 0;
    for (const auto& kv : kv_cache_by_layer_) {
        total += static_cast<size_t>(std::max(0, kv.second.seq_len));
    }
    return total;
}

// === Forward pass (tokens int -> float delegation) ===

std::vector<float> Model::forwardPass(const std::vector<int> &input_ids, bool training) {
    return forwardPassView(input_ids, training);
}

const std::vector<float>& Model::forwardPassView(const std::vector<int> &input_ids, bool training) {
    std::vector<float> input_f;
    input_f.reserve(input_ids.size());
    for (int v : input_ids) {
        input_f.push_back(static_cast<float>(v));
    }
    return forwardPassView(input_f, training);
}

// === Forward pass (multi-entrées: floats + tokens int) ===

std::vector<float> Model::forwardPassNamed(
    const std::unordered_map<std::string, std::vector<float>>& float_inputs,
    const std::unordered_map<std::string, std::vector<int>>& int_inputs,
    bool training
) {
    return forwardPassNamedView(float_inputs, int_inputs, training);
}

void Model::addVizTapFrame(VizFrame vf) {
    if (!viz_taps_enabled_) return;
    if (viz_taps_max_frames_ <= 0) return;
    if (vf.w <= 0 || vf.h <= 0 || vf.channels <= 0) return;
    if (vf.pixels.empty()) return;

    auto trim_local = [](const std::string& s) -> std::string {
        size_t b = 0;
        while (b < s.size() && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) ++b;
        if (b >= s.size()) return std::string();
        size_t e = s.size();
        while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' || s[e - 1] == '\r')) --e;
        return s.substr(b, e - b);
    };

    auto base_label = [&](const std::string& label) -> std::string {
        const size_t bar = label.find('|');
        if (bar == std::string::npos) return trim_local(label);
        return trim_local(label.substr(0, bar));
    };

    const std::string base = base_label(vf.label);
    if (base.empty()) return;

    // Dedup by *base label* (keep last): the part after "|" is metadata that may change
    // every step (e.g. stats, filenames). We want the UI tile to update in-place.
    auto it = std::find_if(viz_taps_.begin(), viz_taps_.end(), [&](const VizFrame& existing) {
        return base_label(existing.label) == base;
    });
    if (it != viz_taps_.end()) {
        *it = std::move(vf);
        return;
    }

    if (static_cast<int>(viz_taps_.size()) < viz_taps_max_frames_) {
        viz_taps_.push_back(std::move(vf));
        return;
    }

    // Evict last (best-effort) to guarantee key frames can be shown.
    viz_taps_.back() = std::move(vf);
}

std::vector<Model::VizFrame> Model::consumeVizTaps() {
    // Le cache est dédupliqué et mis à jour en place par addVizTapFrame().
    // Retourner un snapshot complet évite qu'un forward partiel fasse disparaître
    // de l'interface toutes les couches qui n'ont pas été recapturées ce cycle-ci.
    auto out = viz_taps_;
    if (g_viz_capture_root == this) g_viz_capture_root = nullptr;
    return out;
}

bool Model::InitVizTips() {
    return false;
}

bool Model::UpdateVizTips(const Layer& layer, VizFrame& frame) {
    return applyVizTipByLayerName(layer, frame);
}

void Model::clearVizTipsRegistry() {
    viz_tips_by_layer_name_.clear();
}

void Model::registerVizTip(const std::string& layer_name, const std::string& tip_label) {
    auto trim_local = [](const std::string& s) -> std::string {
        size_t b = 0;
        while (b < s.size() && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) ++b;
        if (b >= s.size()) return std::string();
        size_t e = s.size();
        while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' || s[e - 1] == '\r')) --e;
        return s.substr(b, e - b);
    };

    const std::string key = trim_local(layer_name);
    const std::string val = trim_local(tip_label);
    if (key.empty() || val.empty()) return;
    viz_tips_by_layer_name_[key] = val;
}

bool Model::applyVizTipByLayerName(const Layer& layer, VizFrame& frame) const {
    if (layer.name.empty()) return false;
    auto it = viz_tips_by_layer_name_.find(layer.name);
    if (it == viz_tips_by_layer_name_.end()) return false;

    const std::string& tip = it->second;
    if (tip.empty()) return false;

    // Le chemin canonique doit rester avant le separateur: le Visualizer le
    // parse pour reconnaitre le layer dans architecture.json. Le tip humain
    // est une information secondaire, placee apres "|".
    if (frame.label.empty()) frame.label = tip;
    else frame.label += " | " + tip;
    return true;
}

const std::vector<float>& Model::forwardPassNamedView(
    const std::unordered_map<std::string, std::vector<float>>& float_inputs,
    const std::unordered_map<std::string, std::vector<int>>& int_inputs,
    bool training
) {
    // On injecte les entrées supplémentaires via un canal interne, puis on
    // réutilise le forward float principal (qui exécute le graphe complet).
    pending_float_inputs_ = float_inputs;
    pending_int_inputs_ = int_inputs;

    auto itx = float_inputs.find("x");
    if (itx != float_inputs.end()) {
        return forwardPassView(itx->second, training);
    }

    // Fallback: si "x" absent, utiliser __input__ si présent, sinon vecteur vide.
    auto iti = float_inputs.find("__input__");
    if (iti != float_inputs.end()) {
        return forwardPassView(iti->second, training);
    }

    // Compat: beaucoup de tests/unités injectent une seule entrée nommée (ex: x0)
    // sans alias explicite vers "x". Utiliser cette unique entrée comme ancre du forward.
    if (float_inputs.size() == 1) {
        return forwardPassView(float_inputs.begin()->second, training);
    }

    static const std::vector<float> empty;
    return forwardPassView(empty, training);
}

std::optional<Model::TrainStepResult> Model::trainStep(const TrainStepRequest& request) {
    applyRuntimeConfiguration();
    if (request.optimizer == nullptr || request.target == nullptr) return std::nullopt;

    const std::vector<float>* input = nullptr;
    for (const char* name : {"__input__", "image", "input", "x"}) {
        const auto it = request.float_inputs.find(name);
        if (it != request.float_inputs.end() && it->second != nullptr) {
            input = it->second;
            break;
        }
    }
    if (input == nullptr) return std::nullopt;

    const std::vector<float>* forward_input = input;
    std::vector<float> masked_input;
    const std::vector<float>* output_weights = nullptr;
    if (request.loss_weights != nullptr && !request.loss_weights->empty()) {
        if (request.loss_weights->size() == request.target->size()) {
            output_weights = request.loss_weights;
        } else if (request.loss_weights->size() == input->size()) {
            masked_input.resize(input->size());
            for (size_t i = 0; i < input->size(); ++i) {
                masked_input[i] = (*input)[i] * std::max(0.0f, (*request.loss_weights)[i]);
            }
            forward_input = &masked_input;
        } else {
            throw std::invalid_argument(
                "Model::trainStep: loss_weights size must match the input or target size");
        }
    }

    BBoxSupervision::Weights bbox_weights;
    std::vector<float> combined_output_weights;
    if (request.annotation) {
        bool at_output = true;
        bbox_weights = BBoxSupervision::forModel(request.annotation, modelConfig,
            input->size(), request.target->size(), &at_output);
        if (!bbox_weights.values.empty()) {
            if (at_output) {
                combined_output_weights = bbox_weights.values;
                if (output_weights) for (size_t i=0;i<combined_output_weights.size();++i)
                    combined_output_weights[i] *= (*output_weights)[i];
                output_weights = &combined_output_weights;
            } else {
                const std::vector<float> original = *forward_input;
                masked_input.resize(original.size());
                for (size_t i=0;i<original.size();++i) masked_input[i]=original[i]*bbox_weights.values[i];
                forward_input=&masked_input;
            }
        }
    }

    if (request.mode == TrainStepMode::Optimize) zeroGradients();
    const std::vector<float>& prediction = forwardPassView(*forward_input, true);
    if (prediction.size() != request.target->size()) {
        throw std::invalid_argument(
            "Model::trainStep: prediction and target sizes differ");
    }

    std::string loss_type = "mse";
    try {
        if (modelConfig.contains("loss")) loss_type = modelConfig["loss"].get<std::string>();
        else if (modelConfig.contains("recon_loss")) loss_type = modelConfig["recon_loss"].get<std::string>();
    } catch (...) {
    }

    RuntimeLossGrad::PixelLossOptions options;
    options.weights = output_weights;
    options.relations = request.relations ? request.relations : &bbox_weights.relations;
    try {
        if (modelConfig.contains("huber_delta")) options.huber_delta = std::max(1e-6f, modelConfig["huber_delta"].get<float>());
        if (modelConfig.contains("smoothl1_delta")) options.huber_delta = std::max(1e-6f, modelConfig["smoothl1_delta"].get<float>());
        if (modelConfig.contains("smoothl1_beta")) options.huber_delta = std::max(1e-6f, modelConfig["smoothl1_beta"].get<float>());
        if (modelConfig.contains("charbonnier_eps")) options.charbonnier_eps = std::max(1e-12f, modelConfig["charbonnier_eps"].get<float>());
        if (modelConfig.contains("nll_sigma")) options.gaussian_nll_sigma = std::max(1e-6f, modelConfig["nll_sigma"].get<float>());
        if (modelConfig.contains("gaussian_nll_sigma")) options.gaussian_nll_sigma = std::max(1e-6f, modelConfig["gaussian_nll_sigma"].get<float>());
    } catch (...) {
    }

    auto loss = RuntimeLossGrad::pixel_loss_and_grad(
        prediction, *request.target, loss_type, options);
    if (request.grad_scale != 1.0f) {
        for (float& gradient : loss.grad) gradient *= request.grad_scale;
    }
    backwardPass(loss.grad);

    TrainStepResult result;
    result.loss = static_cast<float>(loss.loss);
    result.metrics["bbox_count"] = static_cast<float>(bbox_weights.bbox_count);
    result.metrics["bbox_loss_weight"] = bbox_weights.strength;
    result.metrics["relation_loss"] = static_cast<float>(loss.relation_loss);
    result.metrics["relation_count"] = static_cast<float>(request.relations ? request.relations->size() : bbox_weights.relation_count);
    result.metrics[RuntimeLossGrad::canonical_pixel_loss_name(loss_type)] = result.loss;
    double sum_squares = 0.0;
    for (const auto& layer : layers) {
        for (float gradient : layer.grad_weights) {
            sum_squares += static_cast<double>(gradient) * static_cast<double>(gradient);
            result.grad_max_abs = std::max(result.grad_max_abs, std::abs(gradient));
        }
    }
    result.grad_norm = static_cast<float>(std::sqrt(sum_squares));
    if (request.mode == TrainStepMode::Optimize) {
        optimizerStep(*request.optimizer, request.learning_rate);
    }
    return result;
}

Layer* Model::getLayerByName(const std::string& name) {
    for (auto& layer : layers) {
        if (layer.name == name) {
            return &layer;
        }
    }
    return nullptr;  // Layer not found
}

// === méthodes utilitaires simples (déjà présentes) ===
void Model::setDensity(double d) { densityFactor = (d > 0.0 ? d : 1.0); }
double Model::getDensity() const { return densityFactor; }

void Model::push(const std::string &name, const std::string &type, size_t params_count) {
    // Normaliser le type et créer le layer avec enum
    std::string normalized_type = normalize_type(type);
    Layer layer(name, normalized_type, params_count);
    layer.dtype = Mimir::parse_dtype(default_dtype_);
    layer.accumulation_dtype = layer.dtype == Mimir::DType::F64
        ? Mimir::DType::F64 : Mimir::DType::F32;
    
    // Le constructeur Layer a déjà converti string -> enum
    // Vérifier que c'est supporté
    if (layer.type_enum == LayerType::UNKNOWN) {
        std::cerr << "❌ ERROR: Unknown layer type '" << type << "' (normalized: '" 
                  << normalized_type << "')" << std::endl;
        log_supported_types();
        throw std::runtime_error("Unknown layer type: " + type);
    }
    
    // Si des dimensions sont configurées dans modelConfig, les appliquer
    if (modelConfig.contains("in_channels")) {
        layer.in_channels = modelConfig["in_channels"];
    }
    if (modelConfig.contains("out_channels")) {
        layer.out_channels = modelConfig["out_channels"];
    }
    if (modelConfig.contains("height")) {
        layer.input_height = modelConfig["height"];
    }
    if (modelConfig.contains("width")) {
        layer.input_width = modelConfig["width"];
    }
    if (modelConfig.contains("kernel")) {
        layer.kernel_size = modelConfig["kernel"];
    }
    if (modelConfig.contains("stride")) {
        layer.stride = modelConfig["stride"];
    }
    if (modelConfig.contains("padding")) {
        layer.padding = modelConfig["padding"];
    }
    if (normalized_type == "NMS") {
        if (modelConfig.contains("nms_iou_threshold")) {
            layer.nms_iou_threshold = modelConfig["nms_iou_threshold"];
        }
        if (modelConfig.contains("nms_score_threshold")) {
            layer.nms_score_threshold = modelConfig["nms_score_threshold"];
        }
        if (modelConfig.contains("nms_max_detections")) {
            layer.nms_max_detections = modelConfig["nms_max_detections"];
        }
        if (modelConfig.contains("nms_class_agnostic")) {
            layer.nms_class_agnostic = modelConfig["nms_class_agnostic"];
        }
    }
    
    // Calculer les dimensions de sortie pour Conv2D
    if ((normalized_type == "Conv2d" || normalized_type == "ConvTranspose2d") && layer.kernel_size > 0) {
        if (normalized_type == "Conv2d") {
            layer.output_height = (layer.input_height + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
            layer.output_width = (layer.input_width + 2 * layer.padding - layer.kernel_size) / layer.stride + 1;
        } else { // ConvTranspose2d
            layer.output_height = (layer.input_height - 1) * layer.stride - 2 * layer.padding + layer.kernel_size;
            layer.output_width = (layer.input_width - 1) * layer.stride - 2 * layer.padding + layer.kernel_size;
        }
    }
    
    // Détecter automatiquement le type de branche basé sur le nom du layer
    layer.detectBranchType();
    
    layers.push_back(layer);
    // Invalider le cache uses_mag_mod : un nouveau layer vient d'être ajouté.
    uses_mag_mod_cached_ = false;
}

size_t Model::totalParamCount() const {
    size_t s = 0;
    for (const auto &L : layers) s += L.params_count;
    return s;
}

void Model::allocateParams() {
    size_t tot = totalParamCount();
    
    auto& allocator = DynamicTensorAllocator::instance();
    
    if (!Model::frameworkLogsSuppressed()) {
        std::cerr << "📦 Allocation de " << layers.size() << " blocs de poids (" << tot << " paramètres au total)..." << std::endl;
    }
    
    // NOUVEAU: Allouer un tensor par layer au lieu d'un tensor par paramètre
    layer_weight_blocks.clear();
    layer_weight_blocks.resize(layers.size());
    
    for (size_t i = 0; i < layers.size(); ++i) {
        size_t layer_param_count = layers[i].params_count;
        
        if (layer_param_count > 0) {
            // ⚠️ CRITIQUE: Allocation dynamique via MemoryGuard (passe par DynamicTensorAllocator)
            // Le flag 'true' force l'allocation à passer par requestAllocation()
            layer_weight_blocks[i] = tensor(layer_param_count, true);
            
            // Lier le tensor au layer
            layers[i].weight_block = &layer_weight_blocks[i];
            // Frozen imported graph components may carry their checkpoint values.
            if (modelConfig.is_object() && modelConfig.value("initialize_from_layer_weights", false) &&
                layers[i].weights.size() == layer_param_count) {
                std::copy(layers[i].weights.begin(), layers[i].weights.end(), layers[i].getWeights());
            }
            
            if (!Model::frameworkLogsSuppressed()) {
                std::cerr << "  Layer " << i << " (" << layers[i].name << "): " 
                          << layer_param_count << " paramètres dans 1 tensor" << std::endl;
            }
        }
    }
    
    if (!Model::frameworkLogsSuppressed()) {
        std::cerr << "✓ " << layers.size() << " blocs de poids créés (1 tensor par layer)" << std::endl;
    }
}

void Model::initializeWeights(const std::string &method, unsigned int seed) {
    if (params_frozen_) {
        throw std::runtime_error("Model::initializeWeights: parameters are frozen");
    }
    if (layer_weight_blocks.empty()) {
        std::cerr << "⚠️  Cannot initialize weights: weight blocks not allocated" << std::endl;
        return;
    }
    
    auto& allocator = DynamicTensorAllocator::instance();
    std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
    
    std::cerr << "🎲 Initializing weights using " << method << " method (bloc par layer)..." << std::endl;
#ifdef _OPENMP
    std::cerr << "🧵 OpenMP: initialisation des poids jusqu'à " << omp_get_max_threads() << " threads" << std::endl;
#endif

    auto mix_seed = [](unsigned int base_seed, size_t layer_idx, unsigned int stream) -> unsigned int {
        // SplitMix64-like mixing to derive independent deterministic seeds.
        uint64_t x = (static_cast<uint64_t>(base_seed) << 1) ^ 0x9E3779B97F4A7C15ull;
        x ^= (static_cast<uint64_t>(layer_idx) + 0xD1B54A32D192ED03ull) * 0xBF58476D1CE4E5B9ull;
        x ^= (static_cast<uint64_t>(stream) + 0x94D049BB133111EBull) * 0x94D049BB133111EBull;
        x ^= (x >> 30);
        x *= 0xBF58476D1CE4E5B9ull;
        x ^= (x >> 27);
        x *= 0x94D049BB133111EBull;
        x ^= (x >> 31);
        return static_cast<unsigned int>(x & 0xFFFFFFFFu);
    };
    
    auto frozen_prefixes = [&]() -> std::vector<std::string> {
        std::vector<std::string> out;
        if (modelConfig.contains("frozen_layer_prefixes") && modelConfig["frozen_layer_prefixes"].is_array()) {
            for (const auto& v : modelConfig["frozen_layer_prefixes"]) {
                if (v.is_string()) out.push_back(v.get<std::string>());
            }
        }
        return out;
    }();

    auto is_frozen_layer = [&](const Layer& l) -> bool {
        if (frozen_prefixes.empty()) return false;
        for (const auto& p : frozen_prefixes) {
            if (p.empty()) continue;
            if (l.name.rfind(p, 0) == 0) return true;
        }
        return false;
    };

    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto &layer = layers[layer_idx];
        
        if (layer.params_count == 0 || !layer.weight_block) continue;
        if (is_frozen_layer(layer)) continue;
        
        // Afficher progression tous les 10 layers
        if (layer_idx % 10 == 0) {
            std::cerr << "  Initializing layer " << layer_idx << "/" << layers.size() 
                      << " (" << layer.name << ")..." << std::endl;
        }
        
        // fan_in/fan_out: utiliser les dimensions réelles quand disponibles
        int fan_in = 0;
        int fan_out = 0;

        if (layer.type_enum == LayerType::Linear && layer.in_features > 0 && layer.out_features > 0) {
            fan_in = layer.in_features;
            fan_out = layer.out_features;
        } else {
            // Estimation fan_in/fan_out depuis params_count
            int fan_estimate = static_cast<int>(std::sqrt(static_cast<float>(layer.params_count)));
            fan_in = std::max(fan_estimate, 32);
            fan_out = std::max(fan_estimate, 32);
        }
        
        float std_dev = 0.01f;
        
        if (method == "xavier" || method == "glorot") {
            std_dev = std::sqrt(2.0f / (fan_in + fan_out));
        }
        else if (method == "he" || method == "kaiming") {
            std_dev = 1.5f * std::sqrt(2.0f / fan_in);
        }
        else if (method == "normal") {
            std_dev = 0.05f;
        }
        
        // Déterminer précisément la zone bias quand possible
        const size_t num_weights = layer.params_count;
        size_t num_pure_weights = num_weights;

        if (layer.type_enum == LayerType::Linear && layer.in_features > 0 && layer.out_features > 0) {
            const size_t expected_w = static_cast<size_t>(layer.in_features) * static_cast<size_t>(layer.out_features);
            const size_t expected_b = layer.use_bias ? static_cast<size_t>(layer.out_features) : 0;
            if (expected_w + expected_b == num_weights) {
                num_pure_weights = expected_w;
            } else {
                // Fallback si le comptage ne correspond pas exactement
                size_t estimated_bias = std::min(expected_b, num_weights / 10);
                num_pure_weights = num_weights - estimated_bias;
            }
        } else {
            // Heuristique générique
            size_t estimated_bias = static_cast<size_t>(fan_out);
            if (estimated_bias > num_weights / 10) {
                estimated_bias = num_weights / 10;
            }
            num_pure_weights = num_weights - estimated_bias;
        }
        
        // Initialiser directement le weight_block du layer
        float* weights_data = layer.weight_block->getData();
        if (!weights_data) continue;

#ifdef _OPENMP
        // Paralléliser uniquement quand c'est suffisamment gros pour amortir l'overhead.
        const bool use_parallel = (num_pure_weights >= 1u << 16);
        if (use_parallel) {
            const unsigned int base_seed = (seed == 0 ? 0xC0FFEEu : seed);
            #pragma omp parallel
            {
                const unsigned int tid = static_cast<unsigned int>(omp_get_thread_num());
                std::mt19937 gen_local(mix_seed(base_seed, layer_idx, tid));
                std::normal_distribution<float> dist_local(0.0f, std_dev);

                #pragma omp for schedule(static)
                for (size_t i = 0; i < num_weights; ++i) {
                    float value = (i >= num_pure_weights) ? 0.0f : dist_local(gen_local);
                    value = std::clamp(value, -3.0f, 3.0f);
                    weights_data[i] = value;
                }
            }
            continue;
        }
#endif

        // Fallback séquentiel (préserve la séquence actuelle basée sur `gen`).
        std::normal_distribution<float> dist(0.0f, std_dev);
        for (size_t i = 0; i < num_weights; ++i) {
            float value = (i >= num_pure_weights) ? 0.0f : dist(gen);
            value = std::clamp(value, -3.0f, 3.0f);  // ±3σ capture 99.7%
            weights_data[i] = value;
        }
    }
    
    std::cerr << "✓ Weights initialized (" << layers.size() << " layers, " << totalParamCount() << " parameters)" << std::endl;
}

void Model::updateWeightsWithNoise(float learning_rate, float noise_std) {
    // NOTE: Fonction obsolète utilisant l'ancienne structure params
    std::cerr << "⚠️ updateWeightsWithNoise() est obsolète" << std::endl;
}

std::vector<uint16_t> Model::getWeights() const {
    // NOTE: Fonction obsolète utilisant l'ancienne structure params
    return std::vector<uint16_t>();
}

void Model::setTokenizer(const Tokenizer &t) {
    tokenizer = t;
    hasTokenizer = true;
    // Garder l'encoder compatible avec la taille vocab du tokenizer.
    // Utile lorsque le tokenizer est chargé après construction.
    encoder.ensureVocabSize(tokenizer.getVocabSize());
    encoder.ensureSpecialEmbeddings();
    if (!Model::frameworkLogsSuppressed()) {
        std::cerr << "[registry] tokenizer attached vocab=" << tokenizer.getVocabSize()
                  << " encoder_dim=" << encoder.dim << std::endl;
    }
}

void Model::setEncoder(const ConditioningEncoder &e) {
    encoder = e;
    encoder.ensureSpecialEmbeddings();
    hasEncoder = true;
    if (!Model::frameworkLogsSuppressed()) {
        std::cerr << "[encoder] attached dim=" << encoder.dim
                  << " vocab=" << encoder.vocab_size << std::endl;
    }
}

void Model::forward(std::vector<uint8_t> &out_uint8) const {
    // NOTE: Fonction obsolète utilisant l'ancienne structure params
    // Utilisez forwardPass() à la place
    out_uint8.clear();
}

void Model::setOutputTarget(const std::vector<uint8_t> &target) {
    // NOTE: Fonction obsolète utilisant l'ancienne structure params
}

bool optimizerTypeFromString(const std::string& name, OptimizerType& type) {
    std::string normalized = name;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (normalized == "sgd") type = OptimizerType::SGD;
    else if (normalized == "adam") type = OptimizerType::ADAM;
    else if (normalized == "adamw") type = OptimizerType::ADAMW;
    else if (normalized == "lion") type = OptimizerType::LION;
    else if (normalized == "adafactor") type = OptimizerType::ADAFACTOR;
    else if (normalized == "radam") type = OptimizerType::RADAM;
    else if (normalized == "nadam") type = OptimizerType::NADAM;
    else if (normalized == "rmsprop") type = OptimizerType::RMSPROP;
    else if (normalized == "lamb") type = OptimizerType::LAMB;
    else return false;
    return true;
}

const char* optimizerTypeName(OptimizerType type) {
    switch (type) {
        case OptimizerType::SGD: return "sgd";
        case OptimizerType::ADAM: return "adam";
        case OptimizerType::ADAMW: return "adamw";
        case OptimizerType::LION: return "lion";
        case OptimizerType::ADAFACTOR: return "adafactor";
        case OptimizerType::RADAM: return "radam";
        case OptimizerType::NADAM: return "nadam";
        case OptimizerType::RMSPROP: return "rmsprop";
        case OptimizerType::LAMB: return "lamb";
    }
    return "unknown";
}

void configureOptimizerFromJson(Optimizer& optimizer, const json& config) {
    if (config.contains("optimizer") && config["optimizer"].is_string()) {
        OptimizerType parsed;
        if (!optimizerTypeFromString(config["optimizer"].get<std::string>(), parsed)) {
            throw std::invalid_argument("optimiseur inconnu: " + config["optimizer"].get<std::string>());
        }
        optimizer.type = parsed;
    }
    if (config.contains("beta1")) optimizer.beta1 = config["beta1"].get<float>();
    if (config.contains("beta2")) optimizer.beta2 = config["beta2"].get<float>();
    if (config.contains("epsilon")) optimizer.eps = config["epsilon"].get<float>();
    if (config.contains("eps")) optimizer.eps = config["eps"].get<float>();
    if (config.contains("weight_decay")) optimizer.weight_decay = config["weight_decay"].get<float>();
    if (config.contains("rmsprop_alpha")) optimizer.rmsprop_alpha = config["rmsprop_alpha"].get<float>();
    if (config.contains("adafactor_clip_threshold")) optimizer.adafactor_clip_threshold = config["adafactor_clip_threshold"].get<float>();
    if (config.contains("adafactor_decay_rate")) optimizer.adafactor_decay_rate = config["adafactor_decay_rate"].get<float>();
    if (config.contains("adafactor_eps2")) optimizer.adafactor_eps2 = config["adafactor_eps2"].get<float>();
    if (config.contains("adafactor_beta1")) optimizer.adafactor_beta1 = config["adafactor_beta1"].get<float>();
    if (config.contains("adafactor_scale_parameter")) optimizer.adafactor_scale_parameter = config["adafactor_scale_parameter"].get<bool>();
    if (config.contains("adafactor_relative_step")) optimizer.adafactor_relative_step = config["adafactor_relative_step"].get<bool>();
}

Optimizer Model::optimizerSnapshot(const Optimizer& source) const {
    Optimizer opt = source;
    const bool use_m = opt.usesFirstMoment();
    const bool use_v = opt.usesSecondMoment();
    if (!source.mv_by_param_ptr.empty()) {
        opt.m.clear();
        opt.v.clear();
        opt.parameter_layout.clear();
        size_t offset = 0;
        for (const auto& layer : layers) {
            if (!layer.weight_block || layer.params_count == 0) continue;
            const auto key = reinterpret_cast<std::uintptr_t>(layer.weight_block->getData());
            auto it = source.mv_by_param_ptr.find(key);
            if (it == source.mv_by_param_ptr.end()) continue;
            if (!use_m && !use_v) continue;
            const size_t n = layer.getWeightsSize();
            const auto& block = it->second;
            if ((use_m && block.m.size() != n) || (use_v && block.v.size() != n)) {
                throw std::runtime_error("Invalid optimizer moment size for " + layer.name);
            }
            opt.parameter_layout.push_back({layer.name, offset, n});
            if (use_m) opt.m.insert(opt.m.end(), block.m.begin(), block.m.end());
            if (use_v) opt.v.insert(opt.v.end(), block.v.begin(), block.v.end());
            offset += n;
        }
    }
    if (!use_m) opt.m.clear();
    if (!use_v) opt.v.clear();
    if (!use_m && !use_v) opt.parameter_layout.clear();
    opt.mv_by_param_ptr.clear();
    return opt;
}

void Model::restoreOptimizerState(Optimizer& opt) const {
    if (!opt.mv_by_param_ptr.empty() || (opt.m.empty() && opt.v.empty() && opt.parameter_layout.empty())) return;
    std::vector<Optimizer::StateBlock> layout = opt.parameter_layout;
    if (layout.empty()) {
        // Legacy checkpoints stored all layer blocks in graph order.
        size_t offset = 0;
        for (const auto& layer : layers) {
            if (!layer.weight_block || layer.params_count == 0) continue;
            layout.push_back({layer.name, offset, layer.getWeightsSize()});
            offset += layer.getWeightsSize();
        }
    }
    size_t total = 0;
    std::unordered_set<std::string> names;
    for (const auto& entry : layout) {
        if (entry.offset != total || !names.insert(entry.name).second || entry.size == 0) {
            throw std::runtime_error("Invalid optimizer parameter layout");
        }
        auto it = std::find_if(layers.begin(), layers.end(), [&](const Layer& l) { return l.name == entry.name; });
        if (it == layers.end() || !it->weight_block || it->getWeightsSize() != entry.size) {
            throw std::runtime_error("Optimizer checkpoint topology mismatch: " + entry.name);
        }
        total += entry.size;
    }
    if ((opt.usesFirstMoment() && opt.m.size() != total) ||
        (opt.usesSecondMoment() && opt.v.size() != total)) {
        throw std::runtime_error("Missing or incompatible optimizer moments");
    }
    for (const auto& entry : layout) {
        const auto it = std::find_if(layers.begin(), layers.end(), [&](const Layer& l) { return l.name == entry.name; });
        auto& block = opt.ensureMomentsFor(it->weight_block->getData(), entry.size);
        if (opt.usesFirstMoment()) std::copy_n(opt.m.begin() + entry.offset, entry.size, block.m.begin());
        if (opt.usesSecondMoment()) std::copy_n(opt.v.begin() + entry.offset, entry.size, block.v.begin());
    }
    opt.m.clear();
    opt.v.clear();
    opt.parameter_layout.clear();
}

void Model::setSerializedOptimizer(Optimizer opt) {
    opt = optimizerSnapshot(opt);
    serialized_optimizer_ = std::move(opt);
    if (!Model::frameworkLogsSuppressed()) {
        std::cerr << "[memory] optimizer_state step=" << serialized_optimizer_->step
                  << " m=" << serialized_optimizer_->m.size()
                  << " v=" << serialized_optimizer_->v.size() << std::endl;
    }
}

void Model::applyParamUpdate(float learning_rate) {
    // NOTE: Fonction obsolète - utilisez optimizerStep() pour l'entraînement moderne avec layer_weight_blocks
    std::cerr << "[DEPRECATED] applyParamUpdate() est obsolète. Utilisez optimizerStep() à la place.\n";
    return;
}

// Multi-optimizer step
void Model::optimizerStep(Optimizer &opt, float learning_rate, const Gradients* gradients) {
    if (params_frozen_) {
        throw std::runtime_error("Model::optimizerStep: parameters are frozen");
    }
    applyRuntimeOptimizerConfiguration(opt);
    publishRuntimeConfiguration(&opt);
    // NOUVEAU: Utiliser les weight_blocks au lieu de params
    if (layer_weight_blocks.empty()) return;

    auto frozen_prefixes = [&]() -> std::vector<std::string> {
        std::vector<std::string> out;
        if (modelConfig.contains("frozen_layer_prefixes") && modelConfig["frozen_layer_prefixes"].is_array()) {
            for (const auto& v : modelConfig["frozen_layer_prefixes"]) {
                if (v.is_string()) out.push_back(v.get<std::string>());
            }
        }
        return out;
    }();

    auto is_frozen_layer = [&](const Layer& l) -> bool {
        if (frozen_prefixes.empty()) return false;
        for (const auto& p : frozen_prefixes) {
            if (p.empty()) continue;
            if (l.name.rfind(p, 0) == 0) return true;
        }
        return false;
    };

    // Sécurité numérique: eps doit être strictement positif et fini.
    // Certains checkpoints/configs peuvent contenir NaN/0 ou 0, ce qui casse Adam/AdamW.
    if (!std::isfinite(opt.eps) || opt.eps <= 0.0f) {
        opt.eps = 1e-8f;
    }

    // Optional global grad clipping (L2 norm), configured via modelConfig.
    // This makes clipping work for any training loop (including grad accumulation).
    float grad_clip_norm = 0.0f;
    if (modelConfig.contains("grad_clip_norm")) {
        grad_clip_norm = std::max(0.0f, modelConfig["grad_clip_norm"].get<float>());
    } else if (modelConfig.contains("clip_norm")) {
        grad_clip_norm = std::max(0.0f, modelConfig["clip_norm"].get<float>());
    }
    if (grad_clip_norm > 0.0f) {
        double sum_sq = 0.0;
        for (const auto& layer : layers) {
            if (is_frozen_layer(layer)) continue;
            for (float g : layer.grad_weights) sum_sq += static_cast<double>(g) * static_cast<double>(g);
            for (float g : layer.grad_bias) sum_sq += static_cast<double>(g) * static_cast<double>(g);
        }
        const float norm = static_cast<float>(std::sqrt(sum_sq));
        if (norm > grad_clip_norm && norm > 1e-12f) {
            const float scale = grad_clip_norm / norm;
            for (auto& layer : layers) {
                if (is_frozen_layer(layer)) continue;
                for (auto& g : layer.grad_weights) g *= scale;
                for (auto& g : layer.grad_bias) g *= scale;
            }
        }
    }
    
    // Appliquer le scheduler exactement une fois. `learning_rate` reste la LR de
    // base demandée par l'appelant (éventuellement modulée par le feedback), et
    // getCurrentLR() fournit uniquement le facteur warmup/decay correspondant.
    float effective_lr = learning_rate;
    if (opt.warmup_steps > 0 || opt.decay_strategy != LRDecayStrategy::NONE) {
        const float base = std::max(1e-12f, opt.initial_lr);
        effective_lr = opt.getCurrentLR() * (learning_rate / base);
    }
    
    // Restore named checkpoint blocks once; preserve the active runtime state thereafter.
    restoreOptimizerState(opt);

    opt.step += 1;
    
    // NOUVEAU: Appliquer l'optimiseur sur chaque weight_block du layer
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        auto &layer = layers[layer_idx];
        
        if (!layer.weight_block || layer.params_count == 0) continue;
        if (is_frozen_layer(layer)) continue;
        if (layer.grad_weights.empty()) continue;
        
        float* weights = layer.weight_block->getData();
        size_t weight_count = layer.getWeightsSize();
        const size_t n = std::min(weight_count, layer.grad_weights.size());
        
        Optimizer::MomentBlock* moments = nullptr;
        if (opt.type != OptimizerType::SGD) {
            moments = &opt.ensureMomentsFor(weights, weight_count);
        }
        
        RuntimeOptimizerRequest update{&opt,effective_lr,weights,layer.grad_weights.data(),
            moments && !moments->m.empty()?moments->m.data():nullptr,
            moments && !moments->v.empty()?moments->v.data():nullptr,n};
        AbstractRuntime* optimizer_runtime=nullptr;
        if(n && !RuntimeRouter::instance().dispatchOptimizer(update,&optimizer_runtime))
            throw std::runtime_error("No runtime completed optimizer update");
        
        // NOTE: ne pas réinitialiser ici.
        // Les boucles d'entraînement modernes appellent zeroGradients() en début de step.
        // Garder le dernier gradient permet la sérialisation/debug (include_gradients).
    }
}

Model::DecoderOutput Model::eval(const std::vector<uint8_t> &target) const {
    DecoderOutput out;
    std::vector<uint8_t> gen;
    forward(gen);
    if (gen.size() != target.size() || gen.empty()) { out.mse = -1.0; return out; }
    double s = 0.0;
    for (size_t i = 0; i < gen.size(); ++i) {
        double d = double(gen[i]) - double(target[i]);
        s += d * d;
    }
    out.mse = s / double(gen.size());

    if (!hasTokenizer) return out;
    size_t vs = tokenizer.getVocabSize();
    if (vs == 0) return out;
    // produce trivial logits from generated image
    out.logits.assign(vs, 0.0f);
    for (size_t i = 0; i < out.logits.size(); ++i) out.logits[i] = 1.0f / float(out.logits.size());
    // top-k tokens
    for (size_t i = 0; i < std::min<size_t>(8, out.logits.size()); ++i) out.tokens.push_back(int(i));
    return out;
}

void Model::setLastEncoding(const std::vector<float> &e) { lastEncoding = e; }

// ---------------- file helpers ----------------
// convert MagicToken vector to JSON
[[maybe_unused]] static json magic_tokens_to_json(const std::vector<MagicToken> &mvec) {
    json a = json::array();
    for (const auto &m : mvec) {
        json mj;
        mj["modality_mask"] = m.modality_mask;
        mj["seed"] = m.seed;
        mj["embed"] = json::array();
        for (int i = 0; i < 8; ++i) mj["embed"].push_back(m.embed[i]);
        a.push_back(mj);
    }
    return a;
}

// read magic tokens from JSON
static void json_to_magic_tokens(const json &j, std::vector<MagicToken> &outMagic) {
    if (!j.is_array()) return;
    for (const auto &m : j) {
        MagicToken mt{};
        mt.modality_mask = m.value("modality_mask", 0u);
        mt.seed = m.value("seed", 0u);
        if (m.contains("embed") && m["embed"].is_array()) {
            for (size_t i = 0; i < 8 && i < m["embed"].size(); ++i) mt.embed[i] = m["embed"][i].get<float>();
        }
        outMagic.push_back(mt);
    }
}

// ---------------- static persistence helpers ----------------
// helper: sanitize strings in id2token array (replace control chars by '<NL>' or space)
[[maybe_unused]] static void sanitize_id2token_json(json &tokj) {
    if (!tokj.is_object() || !tokj.contains("id2token")) return;
    try {
        auto &arr = tokj["id2token"];
        if (!arr.is_array()) return;
        for (auto &el : arr) {
            if (!el.is_string()) continue;
            std::string s = el.get<std::string>();
            bool changed = false;
            for (char &c : s) {
                if (static_cast<unsigned char>(c) <= 0x1F) { // control chars
                    changed = true;
                    c = ' '; // replace with space to avoid embedded newlines
                }
            }
            if (changed) el = s;
        }
    } catch (...) { /* best-effort */ }
}

bool Model::saveCheckpoint(const Tokenizer &tokenizer, const std::vector<MagicToken> &magic_tokens, const fs::path &dir, int epoch) {
    // NOTE: Cette fonction est obsolète et a été remplacée par le module Serialization
    // Utilisez maintenant checkpoint.save() depuis Lua ou Mimir::Serialization::save_checkpoint() depuis C++
    std::cerr << "⚠️ Model::saveCheckpoint() est obsolète! Utilisez Mimir::Serialization::save_checkpoint()" << std::endl;
    std::cerr << "   Depuis Lua: checkpoint.save(model, path, {format='safetensors'})" << std::endl;
    return false;
}

static void write_u64_le(std::ofstream &f, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
    f.write(reinterpret_cast<char*>(b), 8);
}

// writer for a set of float32 tensors into a safetensors-like file.
// Format written:
// [8 bytes little-endian u64] header_length
// [header_length bytes UTF-8 JSON header]
// [binary blob of tensors concatenated as raw little-endian float32]
//
// Header format (JSON object) follows safetensors style:
// { "metadata": {}, "tensors": { "name": { "dtype":"f32", "shape":[N], "data":[offset, length] }, ... } }
static bool write_safetensors_file(const fs::path &outpath, const std::unordered_map<std::string, std::vector<float>> &tensors, std::string *err = nullptr) {
    try {
        // prepare metadata and compute offsets
        json header;
        header["metadata"] = json::object();
        json tensors_meta = json::object();

        uint64_t offset = 0; // data offset after header
        std::vector<std::pair<const std::string*, const std::vector<float>*>> order;
        order.reserve(tensors.size());
        for (const auto &kv : tensors) order.emplace_back(&kv.first, &kv.second);

        // compute total data size to help building header offsets (we don't need it here)
        for (const auto &p : order) {
            const std::string &name = *p.first;
            const std::vector<float> &buf = *p.second;
            uint64_t byte_len = static_cast<uint64_t>(buf.size()) * sizeof(float);
            // record meta: data = [offset, length]
            json m;
            m["dtype"] = "f32";
            m["shape"] = json::array({ static_cast<uint64_t>(buf.size()) });
            m["data"] = json::array({ offset, byte_len });
            tensors_meta[name] = m;
            offset += byte_len;
        }
        header["tensors"] = tensors_meta;

        std::string header_str = header.dump();
        uint64_t header_len = static_cast<uint64_t>(header_str.size());

        // open file and write header length + header
        std::ofstream ofs(outpath.string(), std::ios::binary);
        if (!ofs) {
            if (err) *err = "failed to open output file";
            return false;
        }

        write_u64_le(ofs, header_len);
        ofs.write(header_str.data(), static_cast<std::streamsize>(header_len));

        // now write tensor data in the same order as header (order vector)
        for (const auto &p : order) {
            const std::vector<float> &buf = *p.second;
            if (!buf.empty()) {
                // write raw floats (assume host is little-endian; if not, convert)
                ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size() * sizeof(float)));
            }
        }

        ofs.close();
        return true;
    } catch (const std::exception &e) {
        if (err) *err = e.what();
        return false;
    } catch (...) {
        if (err) *err = "unknown error";
        return false;
    }
}

// Model::packToSafetensor implementation that delegates to writer above.
// Utilise une map fournie par l'appelant (nom -> float buffer).
bool Model::packToSafetensor(const fs::path &outpath, const std::unordered_map<std::string, std::vector<float>> &tensors) const {
    // create parent dir
    try {
        if (outpath.has_parent_path()) fs::create_directories(outpath.parent_path());
    } catch (...) { /* ignore */ }

    std::string err;
    if (!write_safetensors_file(outpath, tensors, &err)) {
        std::cerr << "packToSafetensor: failed to write " << outpath << " : " << err << "\n";
        return false;
    }
    return true;
}

bool Model::tryLoadExistingModel(const fs::path &ckdir, const fs::path &safep, Tokenizer &outTok, ConditioningEncoder &outEnc, std::vector<MagicToken> &outMagic) {
    bool loaded_any = false;
    try {
        fs::path sjson = safep; sjson += ".json";
        if (fs::exists(sjson) && fs::is_regular_file(sjson)) {
            try {
                std::ifstream f(sjson);
                if (f) {
                    json full; f >> full;
                    if (full.contains("tokenizer")) { try { outTok.from_json(full["tokenizer"]); loaded_any = true; } catch(...) {} }
                    else if (full.contains("id2token")) { json tj; tj["id2token"] = full["id2token"]; try { outTok.from_json(tj); loaded_any = true; } catch(...) {} }
                    if (full.contains("magic_tokens")) { try { json_to_magic_tokens(full["magic_tokens"], outMagic); loaded_any = true; } catch(...) {} }
                    if (full.contains("encoder")) {
                        try {
                            auto ej = full["encoder"];
                            outEnc.dim = ej.value("dim", outEnc.dim);
                            if (ej.contains("embeddings") && ej["embeddings"].is_array()) {
                                auto &rows = ej["embeddings"];
                                outEnc.vocab_size = (int)rows.size();
                                outEnc.token_embeddings.assign((size_t)outEnc.dim * (size_t)outEnc.vocab_size, 0.0f);
                                for (size_t r = 0; r < rows.size(); ++r)
                                    for (int d = 0; d < outEnc.dim && d < (int)rows[r].size(); ++d)
                                        outEnc.token_embeddings[r * (size_t)outEnc.dim + d] = rows[r][d].get<float>();
                                loaded_any = true;
                            }
                        } catch (...) {}
                    }
                    if (loaded_any) return true;
                }
            } catch (...) {
                // fallback: safep json is invalid/corrupted, ignore and continue to checkpoint folders
            }
        }
    } catch (...) {}

    try {
        if (fs::exists(ckdir) && fs::is_directory(ckdir)) {
            int best_epoch = -1; fs::path best_dir;
            for (auto &p : fs::directory_iterator(ckdir)) {
                if (!p.is_directory()) continue;
                std::string n = p.path().filename().string();
                if (n.rfind("epoch_", 0) == 0) {
                    try { int e = std::stoi(n.substr(6)); if (e > best_epoch) { best_epoch = e; best_dir = p.path(); } } catch(...) {}
                }
            }
            if (best_epoch >= 0 && !best_dir.empty()) {
                fs::path tokp = best_dir / "tokenizer.json";
                fs::path encp = best_dir / "encoder.json";
                fs::path mp = best_dir / "metadata.json";
                fs::path layersp = best_dir / "layers.json";
                fs::path embp = best_dir / "embeddings.bin";
                fs::path paramsp = best_dir / "params_data.bin";
                
                if (fs::exists(tokp)) {
                    try {
                        std::ifstream tf(tokp);
                        json tj;
                        tf >> tj;
                        outTok.from_json(tj);
                        loaded_any = true;
                    } catch(...) {
                        // tokenizer.json is invalid -> fallback to minimal tokenizer
                        try {
                            json minimal;
                            minimal["id2token"] = json::array({ "<PAD>", "<UNK>", "<SEQ>", "<MOD>", "<MAG>", "<NL>" });
                            outTok.from_json(minimal);
                            loaded_any = true;
                        } catch(...) {}
                    }
                }
                if (fs::exists(encp)) {
                    try { std::ifstream ef(encp); json ej; ef >> ej;
                        outEnc.dim = ej.value("dim", outEnc.dim);
                        if (ej.contains("embeddings") && ej["embeddings"].is_array()) {
                            auto &rows = ej["embeddings"];
                            outEnc.vocab_size = (int)rows.size();
                            outEnc.token_embeddings.assign((size_t)outEnc.dim * (size_t)outEnc.vocab_size, 0.0f);
                            for (size_t r = 0; r < rows.size(); ++r)
                                for (int d = 0; d < outEnc.dim && d < (int)rows[r].size(); ++d)
                                    outEnc.token_embeddings[r * (size_t)outEnc.dim + d] = rows[r][d].get<float>();
                            loaded_any = true;
                        }
                    } catch(...) {}
                }
                if (fs::exists(mp)) {
                    try { std::ifstream mf(mp); json mj; mf >> mj; if (mj.contains("magic_tokens")) { json_to_magic_tokens(mj["magic_tokens"], outMagic); loaded_any = true; } } catch(...) {}
                }
                
                // NOTE: Anciennes méthodes de sauvegarde/chargement supprimées
                // Utiliser maintenant le module Serialization avec checkpoint.load()
                /*
                // Charger la structure des layers
                if (fs::exists(layersp)) {
                    try {
                        if (loadLayersStructure(layersp)) {
                            std::cerr << "✓ Structure des layers chargée depuis " << layersp << std::endl;
                            loaded_any = true;
                        }
                    } catch(...) {
                        std::cerr << "⚠️ Échec du chargement de layers.json" << std::endl;
                    }
                }
                
                // Charger les embeddings
                if (fs::exists(embp)) {
                    try {
                        if (loadEmbeddings(embp)) {
                            std::cerr << "✓ Embeddings chargés depuis " << embp << std::endl;
                            loaded_any = true;
                        }
                    } catch(...) {
                        std::cerr << "⚠️ Échec du chargement des embeddings" << std::endl;
                    }
                }
                
                // Charger les données des paramètres
                if (fs::exists(paramsp)) {
                    try {
                        if (loadParamsData(paramsp)) {
                            std::cerr << "✓ Données des paramètres chargées depuis " << paramsp << std::endl;
                            loaded_any = true;
                        }
                    } catch(...) {
                        std::cerr << "⚠️ Échec du chargement des params_data" << std::endl;
                    }
                }
                */
                
                if (loaded_any) return true;
            }
        }
    } catch (...) {}

    return loaded_any;
}

// --- Définitions vides pour méthodes virtuelles afin de fournir la vtable ---
void Model::buildBackboneUNet(int /*stages*/, int /*blocks_per_stage*/, int /*bottleneck_depth*/) { /* noop */ }
void Model::injectMagicToken(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildTextBranch(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildAudioBranch(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildImageBranch(const MagicToken & /*tok*/) { /* noop */ }
void Model::buildVideoBranch(const MagicToken & /*tok*/) { /* noop */ }

void Model::detectAndSetupBranches() {
    // Parcourir tous les layers et détecter automatiquement les types de branches
    for (auto& layer : layers) {
        layer.detectBranchType();
    }
    
    // Analyser la structure pour identifier les connexions entre branches
    for (size_t i = 0; i < layers.size(); ++i) {
        auto& layer = layers[i];
        
        // Si c'est un layer résiduel, chercher le layer source
        if (layer.branch_type == BranchType::RESIDUAL) {
            // Par convention, le shortcut se connecte généralement plusieurs layers en arrière
            // Chercher un layer avec un nom similaire mais sans "shortcut" ou "residual"
            std::string base_name = layer.name;
            size_t pos = base_name.find("_shortcut");
            if (pos == std::string::npos) {
                pos = base_name.find("_residual");
            }
            
            if (pos != std::string::npos) {
                base_name = base_name.substr(0, pos);
                
                // Chercher le layer de base correspondant
                for (int j = static_cast<int>(i) - 1; j >= 0; --j) {
                    if (layers[j].name.find(base_name) != std::string::npos && 
                        j != static_cast<int>(i)) {
                        layer.branch_sources.push_back(j);
                        layers[j].is_branch_point = true;
                        break;
                    }
                }
            }
        }
    }
    
    std::cerr << "✓ Détection des branches terminée. Trouvé:" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].requiresBranchComputation()) {
            std::cerr << "  - Layer " << i << " (" << layers[i].name << "): ";
            if (layers[i].branch_type == BranchType::RESIDUAL) {
                std::cerr << "RESIDUAL";
            } else if (layers[i].branch_type == BranchType::SKIP_CONNECTION) {
                std::cerr << "SKIP_CONNECTION";
            } else if (layers[i].is_branch_point) {
                std::cerr << "BRANCH_POINT";
            } else if (layers[i].is_merge_point) {
                std::cerr << "MERGE_POINT";
            }
            std::cerr << std::endl;
        }
    }
}

void Model::executeBranchComputation(int layer_idx, 
                                    std::vector<std::vector<float>>& layer_outputs,
                                    bool training) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers.size())) {
        return;
    }
    
    auto& layer = layers[layer_idx];
    
    if (!layer.requiresBranchComputation()) {
        return;
    }
    
    // Si c'est un point de fusion (residual, skip connection, etc.)
    if (layer.branch_type == BranchType::RESIDUAL && !layer.branch_sources.empty()) {
        // Récupérer la sortie du layer source
        int source_idx = layer.branch_sources[0];
        if (source_idx >= 0 && source_idx < static_cast<int>(layer_outputs.size())) {
            // Fusionner avec l'opération spécifiée
            std::vector<float> merged_output;
            if (!RuntimeRouter::instance().dispatchBranchMerge(
                    layer_outputs[layer_idx], layer_outputs[source_idx], merged_output, layer)) {
                throw std::runtime_error("Runtime branch merge failed: " + layer.name);
            }
            layer_outputs[layer_idx] = std::move(merged_output);
        }
    }
    else if (layer.branch_type == BranchType::SPLIT) {
        // Pour les splits, on doit diviser la sortie
        // Ceci sera géré au niveau du forward pass principal
    }
}

void Model::backpropThroughBranch(int layer_idx,
                                 const std::vector<float>& grad_output,
                                 std::vector<std::vector<float>>& layer_gradients) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layers.size())) {
        return;
    }
    
    auto& layer = layers[layer_idx];
    
    if (!layer.requiresBranchComputation()) {
        return;
    }
    
    // Backprop à travers les connexions de branche
    if (layer.branch_type == BranchType::RESIDUAL && !layer.branch_sources.empty()) {
        // Pour une connexion résiduelle, le gradient se propage vers les deux branches
        int source_idx = layer.branch_sources[0];
        if (source_idx >= 0 && source_idx < static_cast<int>(layer_gradients.size())) {
            // Le gradient du résidual se propage tel quel vers la branche source
            if (layer_gradients[source_idx].empty()) {
                layer_gradients[source_idx] = grad_output;
            } else {
                // Accumuler les gradients
                for (size_t i = 0; i < grad_output.size() && i < layer_gradients[source_idx].size(); ++i) {
                    layer_gradients[source_idx][i] += grad_output[i];
                }
            }
        }
    }
}

// === Forward/Backward Pass Complet ===

std::shared_ptr<SkipConnectionControl> Model::skipConnectionControl() {
    if (!skip_control_initialized_) {
        skip_control_->enabled = !modelConfig.is_object() || modelConfig.value("skip_connections_enabled", true);
        skip_control_initialized_ = true;
    }
    skip_control_->available = std::any_of(layers.begin(), layers.end(),
        [](const Layer& layer) { return layer.skipInputIndex() >= 0; });
    return skip_control_;
}

std::vector<float> Model::forwardPass(const std::vector<float> &input, bool training) {
    return forwardPassView(input, training);
}

const std::vector<float>& Model::forwardPassView(const std::vector<float> &input, bool training) {
    applyRuntimeConfiguration();
    const bool skip_connections_enabled = skipConnectionControl()->enabled.load();
    if (skip_control_->available.load()) modelConfig["skip_connections_enabled"] = skip_connections_enabled;
    const bool viz_capture_allowed = !modelConfig.is_object() || !modelConfig.value("viz_taps_suppressed", false);
    const bool viz_capture_requested = viz_capture_allowed && viz_taps_enabled_ && viz_taps_max_frames_ > 0;
    if (viz_capture_requested && g_viz_capture_root == nullptr) {
        g_viz_capture_root = this;
    }
    Model* const viz_capture_root = g_viz_capture_root;
    const bool nested_viz_model = viz_capture_allowed && viz_capture_root != nullptr && viz_capture_root != this;
    if (nested_viz_model) {
        setVizTapsEnabled(true);
        setVizTapsLimits(viz_capture_root->viz_taps_max_frames_,
                         viz_capture_root->viz_taps_max_side_);
        // Reserver aussi la place necessaire dans la racine pour ce graphe
        // enfant. Une limite historique trop basse ne doit pas masquer des (L).
        const size_t required = viz_capture_root->viz_taps_.size() + layers.size() + 64ULL;
        viz_capture_root->viz_taps_max_frames_ = std::max(
            viz_capture_root->viz_taps_max_frames_,
            static_cast<int>(std::min(required, static_cast<size_t>(std::numeric_limits<int>::max()))));
    }
    // Vérifications préliminaires
    if (layers.empty()) {
        std::cerr << "⚠️  Cannot perform forward pass: no layers defined" << std::endl;
        return input;
    }
    
    if (layer_weight_blocks.empty()) {
        std::cerr << "⚠️  Cannot perform forward pass: weights not allocated" << std::endl;
        std::cerr << "    Call allocate_params() and init_weights() first" << std::endl;
        return input;
    }

    if (modelConfig.contains("use_kv_cache")) {
        try {
            setKVCacheEnabled(modelConfig["use_kv_cache"].get<bool>());
        } catch (...) {
        }
    }
    if (training) {
        clearKVCache();
    }
    
    // ========================================================================
    // INITIALIZATION: TensorStore + Validation
    // ========================================================================
    
    // Clear et initialiser TensorStore avec l'input principal
    clearTensorStore();
    storeTensor("x", input);
    // Alias immuable de l'entrée (utile si le graphe réutilise le nom "x" pour la sortie avec une taille différente)
    storeTensor("__input__", input);

    // Injection optionnelle d'entrées nommées (forwardPassNamed)
    if (pending_int_inputs_.has_value()) {
        clearTensorStoreInt();
        for (const auto& kv : *pending_int_inputs_) {
            storeTensorInt(kv.first, kv.second);
        }

        // Convention: "seq" = sortie tokenizer (ids). Alias utile pour modèles existants.
        // Les modèles texte utilisent "text_ids".
        const auto it_seq = pending_int_inputs_->find("seq");
        if (it_seq != pending_int_inputs_->end()) {
            // store under both keys (safe overwrite ok)
            storeTensorInt("text_ids", it_seq->second);
            // Compat: beaucoup de graphes lisent "__input__" (et parfois "x") comme entrée ids.
            // Permet d'utiliser forwardPassNamed({seq=...}) avec ces architectures.
            storeTensorInt("__input__", it_seq->second);
            storeTensorInt("x", it_seq->second);
        }
    }
    if (pending_float_inputs_.has_value()) {
        for (const auto& kv : *pending_float_inputs_) {
            storeTensor(kv.first, kv.second);
        }
    }

    // Convention: "mag" et "mod" = embeddings float (médias + liaison).
    // On les injecte par défaut depuis l'ConditioningEncoder si non fournis explicitement.
    if (hasEncoder) {
        const auto& mag = encoder.getMagEmbedding();
        const auto& mod = encoder.getModEmbedding();

        // Cache du scan O(layers*inputs) : recalculé seulement après un push().
        if (!uses_mag_mod_cached_) {
            uses_mag_mod_ = false;
            for (const auto& lyr : layers) {
                for (const auto& in : lyr.inputs) {
                    if (in == "mag" || in == "mod") { uses_mag_mod_ = true; break; }
                }
                if (uses_mag_mod_) break;
            }
            uses_mag_mod_cached_ = true;
        }

        if (uses_mag_mod_) {
            int expected = 0;
            if (modelConfig.contains("d_model")) expected = std::max(0, modelConfig["d_model"].get<int>());
            else if (modelConfig.contains("text_d_model")) expected = std::max(0, modelConfig["text_d_model"].get<int>());
            if (expected > 0) {
                if (!mag.empty() && static_cast<int>(mag.size()) != expected) {
                    throw std::runtime_error("ConditioningEncoder mag dim mismatch: have=" + std::to_string(mag.size()) + ", expected=" + std::to_string(expected));
                }
                if (!mod.empty() && static_cast<int>(mod.size()) != expected) {
                    throw std::runtime_error("ConditioningEncoder mod dim mismatch: have=" + std::to_string(mod.size()) + ", expected=" + std::to_string(expected));
                }
            }
        }

        if (!mag.empty() && tensor_store.find("mag") == tensor_store.end()) {
            storeTensor("mag", mag);
        }
        if (!mod.empty() && tensor_store.find("mod") == tensor_store.end()) {
            storeTensor("mod", mod);
        }
    }
    pending_float_inputs_.reset();
    pending_int_inputs_.reset();

    // VALIDATION: Vérifier que tous les layers sont supportés (une seule fois)
    static bool validated = false;
    if (!validated) {
        for (size_t i = 0; i < layers.size(); ++i) {
            if (layers[i].type_enum == LayerType::UNKNOWN) {
                std::cerr << "❌ ERROR: Unsupported layer type '" << layers[i].type 
                          << "' at index " << i << " (" << layers[i].name << ")" << std::endl;
                log_supported_types();
                throw std::runtime_error("Unsupported layer type: " + layers[i].type);
            }
        }
        validated = true;
        std::cerr << "✓ All " << layers.size() << " layers validated" << std::endl;
    }

    // ========================================================================
    // STATIC SCHEDULING + PLANNERS (framework)
    // ========================================================================
    const bool planner_enabled = env_flag_true("MIMIR_ENABLE_PLANNER", true);
    const bool fusion_enabled = env_flag_true(
        "MIMIR_PLANNER_FUSION", env_flag_true("MIMIR_ENABLE_FUSION", true));
    const bool fusion_in_training = env_flag_true("MIMIR_ENABLE_FUSION_TRAIN", false);
    const bool planner_buffer_reuse = env_flag_true("MIMIR_PLANNER_BUFFER_REUSE", false);
    const bool planner_cost_model = env_flag_true("MIMIR_PLANNER_COST_MODEL", true);
    const std::string planner_mode_value = [] {
        const char* value = std::getenv("MIMIR_PLANNER_MODE");
        return std::string(value ? value : "legacy");
    }();
    const Mimir::Planning::PlannerMode planner_mode =
        planner_mode_value == "cost" && planner_cost_model ? Mimir::Planning::PlannerMode::Cost :
        planner_mode_value == "static" ? Mimir::Planning::PlannerMode::Static :
        Mimir::Planning::PlannerMode::Legacy;
    // Par défaut, la planification reste silencieuse dans le terminal.
    // Opt-in explicite via MIMIR_PLANNER_STDOUT=1 (compatibilité conservée avec l'ancien nom).
    const bool planner_to_terminal = env_flag_true(
        "MIMIR_PLANNER_STDOUT",
        env_flag_true("MIMIR_PLANNER_TERMINAL", false)
    );
    auto emit_planner_line = [&](const std::string& line) {
        if (planner_to_terminal) {
            std::cerr << line << std::endl;
        } else {
            framework_log_write_file_only((line + "\n").c_str(), line.size() + 1);
        }
    };
    bool graph_plan_changed = false;
    if (planner_enabled) {
        if (!static_plan_.built ||
            static_plan_.built_for_training != training ||
            static_plan_.built_with_fusion != fusion_enabled ||
            static_plan_.built_with_buffer_reuse != planner_buffer_reuse ||
            static_plan_.mode != planner_mode ||
            static_plan_.execution.ops.size() != layers.size()) {
            static_plan_.execution = Mimir::Planning::build_execution_plan_global(
                layers, training, planner_mode, fusion_enabled, planner_buffer_reuse);
            Mimir::Planning::plan_host_buffer_reuse(static_plan_.execution, planner_buffer_reuse);
            Mimir::Planning::apply_global_scratch_plan(static_plan_.execution, layers);
            static_plan_.built = true;
            graph_plan_changed = true;
            static_plan_.built_for_training = training;
            static_plan_.built_with_fusion = fusion_enabled;
            static_plan_.built_with_buffer_reuse = planner_buffer_reuse;
            static_plan_.mode = planner_mode;
            static_plan_.dumped = false;
            static_plan_.runtime_scan_dumped = false;
        }

        // Runtime activation and placement must precede every dump so the
        // selected runtime, transfers and regions describe the executable plan.
        RuntimeRouter::instance().planForwardLayerRoutes(layers);
        if (planner_mode != Mimir::Planning::PlannerMode::Legacy) {
            std::vector<AbstractRuntime*> planner_runtimes;
#ifdef ENABLE_ROCM
            if (g_rocm_engine) planner_runtimes.push_back(g_rocm_engine.get());
#endif
#ifdef ENABLE_CUDA
            if (g_cuda_engine) planner_runtimes.push_back(g_cuda_engine.get());
#endif
#ifdef ENABLE_VULKAN
            if (g_compute_engine) planner_runtimes.push_back(g_compute_engine.get());
#endif
#ifdef ENABLE_OPENCL
            if (g_opencl_engine) planner_runtimes.push_back(g_opencl_engine.get());
#endif
            if (auto* tpu = Mimir::tpuSessionRuntime(); tpu && tpu->isInitialized())
                planner_runtimes.push_back(tpu);
            if (g_cpu_engine) planner_runtimes.push_back(g_cpu_engine.get());
            Mimir::Planning::apply_runtime_placement(static_plan_.execution, planner_runtimes);
        }

        if (!static_plan_.dumped && env_flag_true("MIMIR_PLANNER_DUMP", false)) {
            static_plan_.dumped = true;
            const auto lifetimes = Mimir::Planning::analyze_tensor_lifetimes(layers);
            const auto scratch = Mimir::Planning::plan_conv2d_fastpath_scratch(layers);
            size_t generic_activation_fusions = 0;
            size_t generic_unary_fusions = 0;
            size_t generic_split_fusions = 0;
            size_t generic_chain_edges = 0;
            for (size_t i = 0; i < static_plan_.execution.fuse_activation_consumer.size(); ++i) {
                if (static_plan_.execution.fuse_activation_consumer[i] >= 0) ++generic_activation_fusions;
                if (static_plan_.execution.fuse_unary_consumer[i] >= 0) ++generic_unary_fusions;
                if (static_plan_.execution.fuse_split_consumer[i] >= 0) ++generic_split_fusions;
            }
            for (size_t i = 0; i < static_plan_.execution.fuse_chain_next.size(); ++i) {
                if (static_plan_.execution.fuse_chain_next[i] >= 0) ++generic_chain_edges;
            }
            std::ostringstream planner_summary;
            planner_summary << "[planner] tensors=" << lifetimes.size()
                            << " generic_activation_fusions=" << generic_activation_fusions
                            << " generic_unary_fusions=" << generic_unary_fusions
                            << " generic_split_fusions=" << generic_split_fusions
                            << " generic_chain_edges=" << generic_chain_edges
                            << " conv2d_scratch_bytes={wT=" << scratch.wT_bytes
                            << ", xcol=" << scratch.xcol_bytes
                            << ", c=" << scratch.c_bytes
                            << "}";
            emit_planner_line(planner_summary.str());

            const std::string global_dump = Mimir::Planning::dump_execution_plan_text(static_plan_.execution);
            std::istringstream global_lines(global_dump);
            std::string global_line;
            while (std::getline(global_lines, global_line)) {
                if (!global_line.empty()) emit_planner_line(global_line);
            }
            if (const char* json_path = std::getenv("MIMIR_PLANNER_JSON"); json_path && json_path[0] != '\0') {
                std::ofstream json(json_path, std::ios::binary | std::ios::trunc);
                if (json) json << Mimir::Planning::dump_execution_plan_json(static_plan_.execution) << '\n';
            }
        }
    } else {
        graph_plan_changed = static_plan_.built;
        static_plan_.built = false;
    }

    // Keep the last built plan available even if the UI opens after inference.
    // Otherwise refresh metadata at most twice per second while inspected.
    const auto graph_now=std::chrono::steady_clock::now();
    if (graph_plan_changed || graph_snapshot_time_.time_since_epoch().count()==0 ||
        (runtime_config_->modelGraphRequested() &&
         graph_now-graph_snapshot_time_ >= std::chrono::milliseconds(500))) {
        runtime_config_->publishModelGraph(Mimir::Viz::modelGraphSnapshot(layers,
            planner_enabled && static_plan_.built ? &static_plan_.execution : nullptr,
            planner_enabled ? (planner_mode==Mimir::Planning::PlannerMode::Cost ? "cost" :
                planner_mode==Mimir::Planning::PlannerMode::Static ? "static" : "legacy") : "desactive",
            training,fusion_enabled,planner_buffer_reuse));
        graph_snapshot_time_=graph_now;
    }

    const bool runtime_verbose = !Model::frameworkLogsSuppressed();
    const bool runtime_trace = runtime_verbose;
    if (runtime_verbose) {
        auto join_strings = [](const std::vector<std::string>& items, const char* sep) -> std::string {
            if (items.empty()) return std::string();
            std::ostringstream oss;
            for (size_t i = 0; i < items.size(); ++i) {
                if (i > 0) oss << sep;
                oss << items[i];
            }
            return oss.str();
        };

        auto fusion_to_string = [](Mimir::Planning::FusionKind fusion) -> const char* {
            switch (fusion) {
                case Mimir::Planning::FusionKind::NONE: return "NONE";
                case Mimir::Planning::FusionKind::CONV2D_RELU: return "CONV2D_RELU";
                case Mimir::Planning::FusionKind::GENERIC_ACTIVATION: return "GENERIC_ACTIVATION";
                case Mimir::Planning::FusionKind::GENERIC_SPLIT: return "GENERIC_SPLIT";
                case Mimir::Planning::FusionKind::GENERIC_CHUNK: return "GENERIC_CHUNK";
                case Mimir::Planning::FusionKind::GENERIC_ACTIVATION_SPLIT: return "GENERIC_ACTIVATION_SPLIT";
                case Mimir::Planning::FusionKind::GENERIC_ACTIVATION_CHUNK: return "GENERIC_ACTIVATION_CHUNK";
                case Mimir::Planning::FusionKind::GENERIC_UNARY_SHAPE: return "GENERIC_UNARY_SHAPE";
                default: return "UNKNOWN";
            }
        };

        std::vector<std::string> active_runtimes;
        active_runtimes.reserve(5);

#ifdef ENABLE_ROCM
        if (g_rocm_available && g_rocm_engine && g_rocm_engine->isInitialized()) {
            active_runtimes.emplace_back(g_rocm_engine->name());
        }
#endif

#ifdef ENABLE_CUDA
        if (g_cuda_available && g_cuda_engine && g_cuda_engine->isInitialized()) {
            active_runtimes.emplace_back(g_cuda_engine->name());
        }
#endif

    #ifdef ENABLE_VULKAN
        if (g_compute_available && g_compute_engine && g_compute_engine->isInitialized()) {
            active_runtimes.emplace_back(g_compute_engine->name());
        }
    #endif

    #ifdef ENABLE_OPENCL
        if (g_opencl_available && g_opencl_engine && g_opencl_engine->isInitialized()) {
            active_runtimes.emplace_back(g_opencl_engine->name());
        }
    #endif

        if (g_cpu_available && g_cpu_engine && g_cpu_engine->isInitialized()) {
            active_runtimes.emplace_back(g_cpu_engine->name());
        }

        const std::string selected_hardware = active_runtimes.empty() ? std::string("NONE") : active_runtimes.front();
        std::unordered_map<std::string, size_t> layer_type_count;
        layer_type_count.reserve(layers.size());
        for (const auto& layer : layers) {
            ++layer_type_count[layer.type.empty() ? type_to_string(layer.type_enum) : layer.type];
        }

        std::vector<std::pair<std::string, size_t>> sorted_types(layer_type_count.begin(), layer_type_count.end());
        std::sort(sorted_types.begin(), sorted_types.end(), [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second > b.second;
            return a.first < b.first;
        });

        std::vector<int> fused_into(layers.size(), -1);
        if (planner_enabled && static_plan_.built) {
            const auto& plan = static_plan_.execution;
            for (size_t i = 0; i < plan.fuse_chain_next.size(); ++i) {
                const int consumer = plan.fuse_chain_next[i];
                if (consumer >= 0 && static_cast<size_t>(consumer) < fused_into.size()) {
                    fused_into[static_cast<size_t>(consumer)] = static_cast<int>(i);
                }
            }
            for (size_t i = 0; i < plan.fuse_activation_consumer.size(); ++i) {
                const int consumer = plan.fuse_activation_consumer[i];
                if (consumer >= 0 && static_cast<size_t>(consumer) < fused_into.size()) fused_into[static_cast<size_t>(consumer)] = static_cast<int>(i);
            }
            for (size_t i = 0; i < plan.fuse_unary_consumer.size(); ++i) {
                const int consumer = plan.fuse_unary_consumer[i];
                if (consumer >= 0 && static_cast<size_t>(consumer) < fused_into.size()) fused_into[static_cast<size_t>(consumer)] = static_cast<int>(i);
            }
            for (size_t i = 0; i < plan.fuse_split_consumer.size(); ++i) {
                const int consumer = plan.fuse_split_consumer[i];
                if (consumer >= 0 && static_cast<size_t>(consumer) < fused_into.size()) fused_into[static_cast<size_t>(consumer)] = static_cast<int>(i);
            }
        }

        std::ostringstream signature;
        signature << "hw=" << selected_hardware
                  << " train=" << (training ? 1 : 0)
                  << " planner=" << (planner_enabled ? 1 : 0)
                  << " layers=" << layers.size();
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            signature << '|' << i << ':' << layer.name << ':' << layer.type << ':' << Mimir::Planning::planner_output_name_for(layer);
            if (planner_enabled && static_plan_.built && i < static_plan_.execution.skip_layer.size()) {
                signature << ":skip=" << static_cast<int>(static_plan_.execution.skip_layer[i]);
            }
        }

        if (!static_plan_.runtime_scan_dumped || static_plan_.runtime_scan_signature != signature.str()) {
            static_plan_.runtime_scan_dumped = true;
            static_plan_.runtime_scan_signature = signature.str();

            std::cerr << "[runtime] selected_hardware=" << selected_hardware
                      << " active_priority=[" << join_strings(active_runtimes, ",") << "]"
                      << " planner_enabled=" << (planner_enabled ? 1 : 0)
                      << " fusion_enabled=" << (fusion_enabled ? 1 : 0)
                      << " training=" << (training ? 1 : 0)
                      << std::endl;

            std::ostringstream type_scan;
            type_scan << "[runtime] layer_scan total=" << layers.size() << " types=";
            for (size_t i = 0; i < sorted_types.size(); ++i) {
                if (i > 0) type_scan << ", ";
                type_scan << sorted_types[i].first << ':' << sorted_types[i].second;
            }
            std::cerr << type_scan.str() << std::endl;

            if (planner_enabled && static_plan_.built) {
                const auto& plan = static_plan_.execution;
                emit_planner_line("[runtime] planner_map begin");
                for (size_t i = 0; i < layers.size(); ++i) {
                    const auto& layer = layers[i];
                    const auto& input_names = Mimir::Planning::planner_inputs_for(layer);

                    std::vector<std::string> io_parts;
                    io_parts.reserve(input_names.size());
                    for (const auto& in : input_names) io_parts.push_back(in);

                    const bool skipped = i < plan.skip_layer.size() && plan.skip_layer[i] != 0;
                    std::string call_path;
                    if (skipped) {
                        const int producer = fused_into[i];
                        call_path = (producer >= 0)
                            ? (std::string("fused_skip<-layer#") + std::to_string(producer))
                            : std::string("fused_skip");
                    } else {
                        call_path = "runtime_router.dispatchForwardLayer";
                    }

                    const char* fusion = "NONE";
                    if (i < plan.ops.size()) {
                        fusion = fusion_to_string(plan.ops[i].fusion);
                    }

                    std::ostringstream planner_line;
                    planner_line << "[runtime] planner_map layer#" << i
                                 << " name='" << layer.name
                                 << "' type='" << (layer.type.empty() ? type_to_string(layer.type_enum) : layer.type)
                                 << "' inputs=[" << join_strings(io_parts, ",")
                                 << "] output='" << Mimir::Planning::planner_output_name_for(layer)
                                 << "' fusion=" << fusion
                                 << " call=" << call_path;
                    emit_planner_line(planner_line.str());
                }
                emit_planner_line("[runtime] planner_map end");
            } else {
                emit_planner_line("[runtime] planner_map unavailable (planner disabled)");
            }
        }
    }
    
    if (!planner_enabled) RuntimeRouter::instance().planForwardLayerRoutes(layers);

    // État du forward
    if (training) {
        forward_state.clear();
        forward_state.skip_connections_enabled = skip_connections_enabled;
        forward_state.is_valid = true;
    }

    const bool has_branches = training && std::any_of(
        layers.begin(), layers.end(),
        [](const Layer& l) { return l.requiresBranchComputation(); }
    );

    std::vector<std::vector<float>> all_layer_outputs;
    if (training && has_branches) {
        all_layer_outputs.reserve(layers.size());
    }
    
    // Conservation pour backward (à migrer vers TensorStore)
    if (training) {
        forward_state.layer_outputs.clear();
        forward_state.layer_outputs.reserve(layers.size());
        forward_state.layer_output_masks.clear();
        forward_state.layer_output_masks.reserve(layers.size());
        forward_state.layer_inputs_multi.clear();
        forward_state.layer_inputs_multi.reserve(layers.size());
        forward_state.layer_input_names.clear();
        forward_state.layer_input_names.reserve(layers.size());
        forward_state.layer_input_sizes_multi.clear();
        forward_state.layer_input_sizes_multi.reserve(layers.size());
    }

    auto needs_input_value_snapshot = [](const Layer& layer) -> bool {
        // Par défaut: snapshot valeurs (sécurité), sauf si le backward n'utilise que des tailles.
        // Important: pour Add/Concat/Split/Subtract/TokenMeanPool/Upsample/Identity, on évite la copie.
        if (layer.type == "Add" || layer.type == "Concat" || layer.type == "Split" || layer.type == "Subtract" ||
            layer.type == "TokenMeanPool" || layer.type == "UpsampleNearest" || layer.type == "Identity" || layer.type == "Constant") {
            return false;
        }
        // Dropout backward se base sur un masque (output), pas sur l'input.
        if (layer.type == "Dropout" || layer.type == "Dropout2d" || layer.type == "AlphaDropout") {
            return false;
        }
        return true;
    };

    auto needs_output_mask = [](const Layer& layer) -> bool {
        if (layer.type == "Dropout" || layer.type == "Dropout2d" || layer.type == "AlphaDropout") return true;
        if ((layer.type == "Conv2d" || layer.type == "ConvTranspose2d") && layer.activation != ActivationType::NONE) return true;
        return false;
    };

    auto needs_output_snapshot = [](const Layer& layer) -> bool {
        // Reparameterize backward a besoin de z (output) pour reconstruire eps.
        if (layer.type == "Reparameterize") return true;
        return false;
    };
    
    // ========================================================================
    // FORWARD PASS: Routing via TensorStore
    // ========================================================================

    MemoryGuard& guard = MemoryGuard::instance();
    const size_t guard_mb = guard.getLimit() / (1024ULL * 1024ULL);
    const size_t cap_mb = (max_ram_mb_ > 0) ? max_ram_mb_ : guard_mb;
    RuntimeAllocator allocator(guard, cap_mb);
    const bool allocator_log = env_flag_true("MIMIR_ALLOCATOR_LOG", false);
    const bool allocator_log_verbose = env_flag_true("MIMIR_ALLOCATOR_LOG_VERBOSE", false);
    RuntimeAllocator::BackendMemoryAttribution backend_mem_attrib{};

    static const std::vector<std::string> kDefaultInputNameX = {"x"};

    // VIZ: dernier HxW "connu" pendant ce forward, utile pour les layers
    // qui ne renseignent pas output_width/output_height (ex: activations).
    int viz_last_w = 0;
    int viz_last_h = 0;

    auto apply_fused_layer = [&](std::vector<float>& data, const Layer& fused_layer) {
        std::vector<const std::vector<float>*> inputs{&data};
        std::vector<std::vector<float>> outputs;
        if (!RuntimeRouter::instance().dispatchForwardLayer(inputs, outputs, fused_layer, training))
            throw std::runtime_error("Runtime fused layer failed: " + fused_layer.name);
        if (fused_layer.type_enum == LayerType::Split || fused_layer.type_enum == LayerType::Chunk) {
            const std::string base = fused_layer.output.empty() ? "x" : fused_layer.output;
            for (size_t i = 0; i < outputs.size(); ++i) storeTensor(base + "_" + std::to_string(i), outputs[i]);
        }
        data = std::move(outputs.at(0));
    };

    const bool exhaustive_model_viz = viz_taps_enabled_ && viz_taps_max_frames_ > 0;
    if (exhaustive_model_viz && (!modelConfig.is_object() || !modelConfig.value("viz_registered_tips_only", false))) {
        viz_taps_max_frames_ = std::max(
            viz_taps_max_frames_,
            static_cast<int>(std::min(layers.size() + static_cast<size_t>(64),
                                      static_cast<size_t>(std::numeric_limits<int>::max()))));
    }

    const bool planned_host_reuse_enabled =
        planner_enabled && static_plan_.built && planner_buffer_reuse && !training &&
        !exhaustive_model_viz &&
        std::none_of(static_plan_.execution.skip_layer.begin(),
                     static_plan_.execution.skip_layer.end(),
                     [](const uint8_t value) { return value != 0; });
    const bool poison_reused_buffers = env_flag_true("MIMIR_PLANNER_BUFFER_POISON", false);
    std::unordered_map<size_t, std::vector<float>> planned_host_buffer_pool;
    size_t actual_buffer_reuse_count = 0;
    size_t actual_buffer_reuse_bytes = 0;
    const bool planner_device_residency = env_flag_true("MIMIR_PLANNER_DEVICE_RESIDENCY", false);
    std::vector<uint8_t> resident_chain_skip(layers.size(), 0);

    std::string forward_output_name = modelConfig.is_object()
        ? modelConfig.value("forward_output_tensor", std::string("x")) : "x";
    size_t forward_layer_limit = layers.size();
    if (!training && modelConfig.contains("inference_output_tensor")) {
        forward_output_name = modelConfig.value("inference_output_tensor", std::string("x"));
        bool producer_found = false;
        for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
            const std::string output_name = layers[layer_idx].output.empty() ? "x" : layers[layer_idx].output;
            if (output_name == forward_output_name) {
                forward_layer_limit = layer_idx + 1;
                producer_found = true;
            }
        }
        if (!producer_found) {
            throw std::runtime_error(
                "Inference output tensor has no producer: " + forward_output_name);
        }
    }

    for (size_t layer_idx = 0; layer_idx < forward_layer_limit; ++layer_idx) {
        if (resident_chain_skip[layer_idx] != 0) continue;
        // Une fusion retire les consumers du parcours. En mode VIZ exhaustif,
        // executer chaque noeud garantit une vignette (L) par layer.
        if (!exhaustive_model_viz && planner_enabled && static_plan_.built &&
            layer_idx < static_plan_.execution.skip_layer.size() &&
            static_plan_.execution.skip_layer[layer_idx] != 0) {
            if (runtime_trace) {
                const auto& skipped_layer = layers[layer_idx];
                std::cerr << "[runtime-trace] layer#" << layer_idx
                          << " name='" << skipped_layer.name
                          << "' type='" << (skipped_layer.type.empty() ? type_to_string(skipped_layer.type_enum) : skipped_layer.type)
                          << "' backend=FUSED_SKIP call=fused_skip output_size=0"
                          << std::endl;
            }
            continue;
        }

        auto &layer = layers[layer_idx];
        if (!layer.shared_weights_from.empty()) {
            Layer* owner = getLayerByName(layer.shared_weights_from);
            if (!owner || !owner->weight_block) throw std::runtime_error("Shared weight source unavailable: " + layer.name);
            layer.weight_block = owner->weight_block;
        }

        // ====================================================================
        // RETRIEVE INPUTS (multi-input support)
        // ====================================================================
        
        const std::vector<std::string>& input_names =
            (layer.inputs.empty() && layer.type_enum != LayerType::Constant) ? kDefaultInputNameX : layer.inputs;

        auto& inputs = scratch_input_ptrs_;
        inputs.clear();
        inputs.reserve(input_names.size());

        // Pour Embedding: accepter un input fourni en int (tensor_store_int) et le projeter en float ids.
        // IMPORTANT: on évite d'appeler getTensor() en premier (qui loggue une erreur) car
        // le chemin normal pour Embedding peut être via tensor_store_int.
        auto& embedding_ids_tmp = scratch_embedding_ids_tmp_;
        auto& embedding_ids_fallback = scratch_embedding_ids_fallback_;

        for (size_t in_i = 0; in_i < input_names.size(); ++in_i) {
            const auto& name = input_names[in_i];

            // 1) Chemin standard: float TensorStore
            auto itf = tensor_store.find(name);
            if (itf != tensor_store.end()) {
                inputs.push_back(&itf->second);
                continue;
            }

            // 2) Cas spécial Embedding: ids int -> float
            if (layer.type_enum == LayerType::Embedding && in_i == 0) {
                auto iti = tensor_store_int.find(name);
                if (iti == tensor_store_int.end()) {
                    // Fallback: si text_ids n'est pas fourni (ex: smoke tests / entraînement sans prompt),
                    // on génère des ids de padding de longueur seq_len pour stabiliser les shapes.
                    const int L = (layer.seq_len > 0) ? layer.seq_len : 1;
                    const int pad = (layer.padding_idx >= 0) ? layer.padding_idx : 0;
                    embedding_ids_fallback.assign(static_cast<size_t>(L), pad);
                    iti = tensor_store_int.emplace(name, embedding_ids_fallback).first;
                }

                const std::vector<int>& ids = iti->second;
                embedding_ids_tmp.clear();
                embedding_ids_tmp.reserve(ids.size());
                for (int v : ids) embedding_ids_tmp.push_back(static_cast<float>(v));
                inputs.push_back(&embedding_ids_tmp);
                continue;
            }

            // 3) Erreur: input manquant
            std::cerr << "❌ ERROR in layer " << layer_idx << " (" << layer.name
                      << "): Cannot find input tensor '" << name << "'" << std::endl;
            std::cerr << "Available tensors: ";
            auto available = getAvailableTensors();
            for (const auto& t : available) std::cerr << "'" << t << "' ";
            std::cerr << std::endl;
            std::cerr << "Available int tensors: ";
            auto available_i = getAvailableIntTensors();
            for (const auto& t : available_i) std::cerr << "'" << t << "' ";
            std::cerr << std::endl;
            throw std::runtime_error("Missing input tensor: " + name);
        }
        
        // Snapshot inputs for backward (multi-input)
        if (training) {
            forward_state.layer_input_names.push_back(input_names);

            std::vector<size_t> sizes;
            sizes.reserve(inputs.size());
            for (const auto* inp : inputs) sizes.push_back(inp ? inp->size() : 0ULL);
            forward_state.layer_input_sizes_multi.push_back(std::move(sizes));

            std::vector<std::vector<float>> snap;
            if (needs_input_value_snapshot(layer)) {
                snap.reserve(inputs.size());
                for (const auto* inp : inputs) {
                    snap.push_back(*inp);
                }
            }
            forward_state.layer_inputs_multi.push_back(std::move(snap));

            // placeholders alignés par layer
            forward_state.layer_outputs.emplace_back();
            forward_state.layer_output_masks.emplace_back();
        }

        // Pour compatibilité: x est généralement inputs[0], sauf pour des layers sans entrée comme Constant.
        static const std::vector<float> kEmptyInput;
        const std::vector<float>& x = (!inputs.empty() && inputs[0] != nullptr) ? *inputs[0] : kEmptyInput;

        std::string executed_backend = "CPU";
        std::string executed_call = "runtime_router.dispatchForwardLayer";
        
        std::vector<float> layer_output;
        bool resident_chain_executed = false;
        std::string resident_final_output_name;

#ifdef ENABLE_VULKAN
        if (planner_device_residency && planner_enabled && static_plan_.built && !training &&
            !exhaustive_model_viz && g_compute_engine && g_compute_engine->isInitialized() &&
            g_compute_engine->config().linear_enabled &&
            layer_idx < static_plan_.execution.layers.size() &&
            static_plan_.execution.layers[layer_idx].runtime == RuntimeKind::Vulkan &&
            static_cast<long long>(x.size()) >= std::max(0, g_compute_engine->config().linear_min_ops)) {
            auto resident_unary = [](const LayerType type) {
                return type == LayerType::ReLU || type == LayerType::SiLU ||
                       type == LayerType::GELU || type == LayerType::Sigmoid ||
                       type == LayerType::Tanh;
            };
            if (resident_unary(layer.type_enum)) {
                std::vector<LayerType> operations;
                size_t chain_end = layer_idx;
                std::string expected_input = input_names.empty() ? std::string() : input_names.front();
                for (size_t candidate = layer_idx; candidate < layers.size(); ++candidate) {
                    const Layer& chain_layer = layers[candidate];
                    if (!resident_unary(chain_layer.type_enum) ||
                        candidate >= static_plan_.execution.layers.size() ||
                        static_plan_.execution.layers[candidate].runtime != RuntimeKind::Vulkan) break;
                    const auto& chain_inputs = Mimir::Planning::planner_inputs_for(chain_layer);
                    if (chain_inputs.size() != 1 || chain_inputs.front() != expected_input) break;
                    if (candidate > layer_idx) {
                        const auto& previous_planned = static_plan_.execution.layers[candidate - 1];
                        const auto tensor_it = static_plan_.execution.tensors.find(previous_planned.output);
                        if (tensor_it == static_plan_.execution.tensors.end() ||
                            tensor_it->second.persistent || tensor_it->second.graph_output ||
                            tensor_it->second.last_use != candidate) break;
                    }
                    operations.push_back(chain_layer.type_enum);
                    chain_end = candidate;
                    expected_input = Mimir::Planning::planner_output_name_for(chain_layer);
                }
                if (operations.size() >= 2) {
                    layer_output.assign(x.size(), 0.0f);
                    if (g_compute_engine->unaryChainForwardResident(
                            x.data(), layer_output.data(), static_cast<int>(x.size()), operations)) {
                        resident_chain_executed = true;
                        resident_final_output_name = Mimir::Planning::planner_output_name_for(layers[chain_end]);
                        for (size_t skipped = layer_idx + 1; skipped <= chain_end; ++skipped) {
                            resident_chain_skip[skipped] = 1;
                        }
                        executed_backend = "VULKAN";
                        executed_call = "vulkan.unaryChainForwardResident";
                    } else {
                        layer_output.clear();
                    }
                }
            }
        }
#endif

        // Dispatch runtime générique (CUDA/ROCm) pour les layers supportés via forwardLayer().
        // Retourne true si un runtime a produit une sortie valide (outputs[0]).
        auto try_runtime_forward_layer = [&](std::vector<float>& out) -> bool {
            const bool has_preplanned_route = RuntimeRouter::instance().hasForwardRouteForLayer(layer);
            if (!has_preplanned_route) {
                if (runtime_trace) {
                    std::cerr << "[runtime-fallback] layer#" << layer_idx
                              << " name='" << layer.name
                              << "' type='" << (layer.type.empty() ? type_to_string(layer.type_enum) : layer.type)
                              << "' reason=unsupported_by_preplanned_route"
                              << std::endl;
                }
                return false;
            }

            std::vector<std::vector<float>> runtime_outputs;
            AbstractRuntime* selected_runtime = nullptr;
            AbstractRuntime* planned_runtime = nullptr;
            if (planner_enabled && static_plan_.built &&
                planner_mode != Mimir::Planning::PlannerMode::Legacy &&
                layer_idx < static_plan_.execution.layers.size()) {
                switch (static_plan_.execution.layers[layer_idx].runtime) {
#ifdef ENABLE_ROCM
                    case RuntimeKind::ROCm: planned_runtime = g_rocm_engine.get(); break;
#endif
#ifdef ENABLE_CUDA
                    case RuntimeKind::CUDA: planned_runtime = g_cuda_engine.get(); break;
#endif
#ifdef ENABLE_VULKAN
                    case RuntimeKind::Vulkan: planned_runtime = g_compute_engine.get(); break;
#endif
#ifdef ENABLE_OPENCL
                    case RuntimeKind::OpenCL: planned_runtime = g_opencl_engine.get(); break;
#endif
                    case RuntimeKind::TPU: planned_runtime = Mimir::tpuSessionRuntime(); break;
                    case RuntimeKind::CPU: planned_runtime = g_cpu_engine.get(); break;
                    default: break;
                }
            }
            RuntimeForwardContext context;
            context.skip_connections_enabled = skip_connections_enabled;
            if (layer.type_enum == LayerType::Reparameterize) {
                if (modelConfig.contains("stochastic_latent")) {
                    try { context.stochastic_latent = modelConfig["stochastic_latent"].get<bool>(); } catch (...) {}
                } else if (modelConfig.contains("vae_stochastic_latent")) {
                    try { context.stochastic_latent = modelConfig["vae_stochastic_latent"].get<bool>(); } catch (...) {}
                }
            }
            if (training) context.output_mask=&forward_state.layer_output_masks.back();
            const bool runtime_ok = planned_runtime
                ? RuntimeRouter::instance().dispatchForwardLayerPlanned(
                    planned_runtime, inputs, runtime_outputs, layer, training, &selected_runtime, context)
                : RuntimeRouter::instance().dispatchForwardLayer(
                    inputs, runtime_outputs, layer, training, &selected_runtime, context);
            if (!runtime_ok) {
                if (runtime_trace) {
                    std::vector<std::string> active_runtimes;
                    active_runtimes.reserve(5);
#ifdef ENABLE_ROCM
                    if (g_rocm_available && g_rocm_engine && g_rocm_engine->isInitialized()) active_runtimes.emplace_back(g_rocm_engine->name());
#endif
#ifdef ENABLE_CUDA
                    if (g_cuda_available && g_cuda_engine && g_cuda_engine->isInitialized()) active_runtimes.emplace_back(g_cuda_engine->name());
#endif
#ifdef ENABLE_VULKAN
                    if (g_compute_available && g_compute_engine && g_compute_engine->isInitialized()) active_runtimes.emplace_back(g_compute_engine->name());
#endif
#ifdef ENABLE_OPENCL
                    if (g_opencl_available && g_opencl_engine && g_opencl_engine->isInitialized()) active_runtimes.emplace_back(g_opencl_engine->name());
#endif
                    if (g_cpu_available && g_cpu_engine && g_cpu_engine->isInitialized()) active_runtimes.emplace_back(g_cpu_engine->name());

                    auto join_runtime_names = [](const std::vector<std::string>& names) -> std::string {
                        if (names.empty()) return std::string();
                        std::ostringstream oss;
                        for (size_t i = 0; i < names.size(); ++i) {
                            if (i) oss << ",";
                            oss << names[i];
                        }
                        return oss.str();
                    };

                    std::ostringstream input_sizes;
                    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
                        if (input_index) input_sizes << ",";
                        input_sizes << (inputs[input_index] ? inputs[input_index]->size() : 0);
                    }

                    const char* reason = active_runtimes.empty()
                        ? "no_active_runtime"
                        : "runtime_execution_error_route_pruned";

                    std::cerr << "[runtime-fallback] layer#" << layer_idx
                              << " name='" << layer.name
                              << "' type='" << (layer.type.empty() ? type_to_string(layer.type_enum) : layer.type)
                              << "' reason=" << reason
                              << " active_priority=[" << join_runtime_names(active_runtimes) << "]"
                              << " input_sizes=[" << input_sizes.str() << "]"
                              << std::endl;
                }
                return false;
            }
            if (layer.type_enum == LayerType::Split || layer.type_enum == LayerType::Chunk) {
                const std::string base = layer.output.empty() ? "x" : layer.output;
                for (size_t i = 0; i < runtime_outputs.size(); ++i)
                    storeTensor(base + "_" + std::to_string(i), runtime_outputs[i]);
            }
            out = std::move(runtime_outputs[0]);
            executed_call = planned_runtime
                ? "runtime_router.dispatchForwardLayerPlanned"
                : "runtime_router.dispatchForwardLayer";
            executed_backend = selected_runtime ? std::string(selected_runtime->name()) : std::string("runtime_router(unknown)");
            return true;
        };
        
        // ====================================================================
        // DISPATCH PRINCIPAL VIA SWITCH/CASE SUR LayerType (MODE STRICT)
        // ====================================================================
        
        try {
            if (!resident_chain_executed && !try_runtime_forward_layer(layer_output)) {
                throw std::runtime_error("No runtime could execute layer '" + layer.name + "' (" + layer.type + ")");
            }
        } catch (const std::exception& e) {
            std::cerr << "❌ ERROR in layer " << layer_idx << " (" << layer.name
                      << ", type: " << type_to_string(layer.type_enum) << "): "
                      << e.what() << std::endl;
            throw;
        }

        // ====================================================================
        // VIZ TAPS (best-effort): capturer des vignettes intermédiaires par bloc/layer
        // ====================================================================
        // NOTE: Beaucoup d'archis ne renseignent pas output_w/output_h pour tous les layers
        // (ex: activations). On garde un "dernier HxW connu" pour afficher plus de blocs
        // sans modifier les builders.
        if (viz_capture_allowed && viz_taps_enabled_ && viz_taps_max_frames_ > 0) {
            if (!viz_tips_init_done_) {
                clearVizTipsRegistry();
                viz_tips_custom_enabled_ = InitVizTips();
                viz_tips_init_done_ = true;
            }

            auto is_viz_candidate_layer = [&](LayerType t) {
                switch (t) {
                    // Spatial ops (déjà supportés)
                    case LayerType::Conv2d:
                    case LayerType::ConvTranspose2d:
                    case LayerType::DepthwiseConv2d:
                    case LayerType::Reparameterize:
                    case LayerType::MaxPool2d:
                    case LayerType::AvgPool2d:
                    case LayerType::AdaptiveAvgPool2d:
                    case LayerType::GlobalAvgPool2d:
                    case LayerType::UpsampleNearest:
                    case LayerType::UpsampleBilinear:
                    case LayerType::UpsampleBicubic:
                    case LayerType::PixelShuffle:
                    case LayerType::ZeroPad2d:
                    case LayerType::ReflectionPad2d:
                    case LayerType::ReplicationPad2d:
                        return true;

                    // Layers qui conservent souvent la forme spatiale (permet d'afficher + de blocs)
                    case LayerType::BatchNorm2d:
                    case LayerType::InstanceNorm2d:
                    case LayerType::GroupNorm:
                    case LayerType::LayerNorm:
                    case LayerType::RMSNorm:
                    case LayerType::Dropout2d:
                    case LayerType::Dropout:
                    case LayerType::AlphaDropout:
                    case LayerType::Identity:
                    case LayerType::Permute:
                    case LayerType::Transpose:
                    case LayerType::Reshape:
                    case LayerType::View:
                    case LayerType::Squeeze:
                    case LayerType::Unsqueeze:
                    case LayerType::Concat:
                    case LayerType::Add:
                    case LayerType::Subtract:
                    case LayerType::Multiply:
                    case LayerType::Divide:

                    // Attention (affichage best-effort; utile pour diagnostiquer la mémoire K/V)
                    case LayerType::SelfAttention:
                    case LayerType::MultiHeadAttention:
                    case LayerType::CrossAttention:

                    // Activations (souvent omettent output_w/output_h dans les builders)
                    case LayerType::ReLU:
                    case LayerType::LeakyReLU:
                    case LayerType::GELU:
                    case LayerType::GEGLU:
                    case LayerType::SiLU:
                    case LayerType::Tanh:
                    case LayerType::Sigmoid:
                    case LayerType::Softmax:
                    case LayerType::LogSoftmax:
                    case LayerType::Softplus:
                    case LayerType::Mish:
                    case LayerType::HardSigmoid:
                    case LayerType::HardSwish:
                        return true;

                    default:
                        // La VIZ demande une vue exhaustive du graphe. Les
                        // sorties non spatiales utiliseront la heatmap vectorielle.
                        return exhaustive_model_viz && t != LayerType::UNKNOWN;
                }
            };

            const bool is_packed_output_concat =
                (layer.type_enum == LayerType::Concat) &&
                (layer.name.find("out_concat") != std::string::npos ||
                 layer.name.find("out_pack") != std::string::npos ||
                 layer.output == "x");

            // Opt-in sparse capture: skip conversion/allocation for unregistered layers.
            const bool registered_tips_only = modelConfig.is_object() && modelConfig.value("viz_registered_tips_only", false);
            const bool registered_tip = viz_tips_by_layer_name_.count(layer.name) != 0;
            if ((!registered_tips_only || registered_tip) &&
                (is_viz_candidate_layer(layer.type_enum) || registered_tip) &&
                (!is_packed_output_concat || exhaustive_model_viz)) {

                auto canonical_viz_label = [&]() -> std::string {
                    std::string model_key = modelConfig.value("type", std::string());
                    if (model_key.empty()) model_key = getModelName();
                    if (model_key.empty()) model_key = "model";
                    const std::string layer_key = layer.name.empty()
                        ? ("layer_" + std::to_string(layer_idx))
                        : layer.name;
                    return model_key + "/blocks/" + layer_key + "/" +
                           type_to_string(layer.type_enum);
                };

                auto resize_viz_frame_nearest = [&](VizFrame& vf, int target_w, int target_h) {
                    if (target_w <= 0 || target_h <= 0) return;
                    if (vf.w <= 0 || vf.h <= 0 || vf.channels <= 0) return;
                    if (vf.w == target_w && vf.h == target_h) return;

                    const int ch = vf.channels;
                    const size_t expected = static_cast<size_t>(vf.w) * static_cast<size_t>(vf.h) * static_cast<size_t>(ch);
                    if (vf.pixels.size() != expected) return;

                    std::vector<uint8_t> up;
                    up.resize(static_cast<size_t>(target_w) * static_cast<size_t>(target_h) * static_cast<size_t>(ch));
                    for (int y = 0; y < target_h; ++y) {
                        const int sy = (target_h > 1) ? ((y * vf.h) / target_h) : 0;
                        for (int x = 0; x < target_w; ++x) {
                            const int sx = (target_w > 1) ? ((x * vf.w) / target_w) : 0;
                            const size_t src_base = (static_cast<size_t>(sy) * static_cast<size_t>(vf.w) + static_cast<size_t>(sx)) * static_cast<size_t>(ch);
                            const size_t dst_base = (static_cast<size_t>(y) * static_cast<size_t>(target_w) + static_cast<size_t>(x)) * static_cast<size_t>(ch);
                            for (int cidx = 0; cidx < ch; ++cidx) {
                                up[dst_base + static_cast<size_t>(cidx)] = vf.pixels[src_base + static_cast<size_t>(cidx)];
                            }
                        }
                    }

                    if (!vf.pixels_real.empty()) {
                        const size_t rc = vf.pixels_real.size() / (static_cast<size_t>(vf.w) * vf.h);
                        std::vector<uint8_t> real(static_cast<size_t>(target_w) * target_h * rc);
                        for (int y = 0; y < target_h; ++y) for (int x = 0; x < target_w; ++x)
                            for (size_t channel = 0; channel < rc; ++channel)
                                real[(static_cast<size_t>(y) * target_w + x) * rc + channel] =
                                    vf.pixels_real[(static_cast<size_t>(y * vf.h / target_h) * vf.w + x * vf.w / target_w) * rc + channel];
                        vf.pixels_real = std::move(real);
                    }
                    if (vf.tensor_sample_channels > 0) {
                        const size_t tc = static_cast<size_t>(vf.tensor_sample_channels);
                        std::vector<float> values(static_cast<size_t>(target_w) * target_h * tc);
                        std::vector<size_t> indices(values.size());
                        for (int y = 0; y < target_h; ++y) for (int x = 0; x < target_w; ++x)
                            for (size_t channel = 0; channel < tc; ++channel) {
                                const size_t src = (static_cast<size_t>(y * vf.h / target_h) * vf.w + x * vf.w / target_w) * tc + channel;
                                const size_t dst = (static_cast<size_t>(y) * target_w + x) * tc + channel;
                                values[dst] = vf.tensor_values[src]; indices[dst] = vf.tensor_indices[src];
                            }
                        vf.tensor_values = std::move(values); vf.tensor_indices = std::move(indices);
                    }
                    vf.pixels = std::move(up);
                    vf.w = target_w;
                    vf.h = target_h;
                };

                auto describe_tensor = [&]() {
                    double sum = 0.0;
                    float lo = std::numeric_limits<float>::infinity(), hi = -lo;
                    size_t finite = 0;
                    for (float value : layer_output) if (std::isfinite(value)) {
                        lo = std::min(lo, value); hi = std::max(hi, value); sum += value; ++finite;
                    }
                    std::ostringstream info;
                    info << (training ? "Capture: entrainement" : "Capture: inference")
                         << "\nValeurs: " << layer_output.size() << "; non finies: " << (layer_output.size() - finite);
                    if (finite) info << "\nMin / max: " << lo << " / " << hi << "\nMoyenne: " << sum / finite;
                    return info.str();
                };

                auto viz_preview_prefers_chw = [&](const Layer& lyr) -> bool {
                    switch (lyr.type_enum) {
                        case LayerType::Conv2d:
                        case LayerType::ConvTranspose2d:
                        case LayerType::DepthwiseConv2d:
                        case LayerType::GroupNorm:
                        case LayerType::BatchNorm2d:
                        case LayerType::InstanceNorm2d:
                        case LayerType::UpsampleNearest:
                        case LayerType::UpsampleBilinear:
                        case LayerType::UpsampleBicubic:
                        case LayerType::PixelShuffle:
                            return true;
                        case LayerType::LayerNorm:
                            return lyr.in_channels > 0 && lyr.input_height > 0 && lyr.input_width > 0;
                        default:
                            break;
                    }

                    if (lyr.name.find("_chw") != std::string::npos || lyr.name.find("/chw") != std::string::npos) {
                        return true;
                    }
                    if (lyr.output.find("_chw") != std::string::npos || lyr.output.find("/chw") != std::string::npos) {
                        return true;
                    }
                    if (lyr.name.find("_hwc") != std::string::npos || lyr.name.find("/hwc") != std::string::npos) {
                        return false;
                    }
                    if (lyr.output.find("_hwc") != std::string::npos || lyr.output.find("/hwc") != std::string::npos) {
                        return false;
                    }
                    return false;
                };

                auto infer_hw = [&](const Layer& lyr, size_t out_size, int& ow, int& oh, int& c, bool& channels_first) -> bool {
                    channels_first = viz_preview_prefers_chw(lyr);

                    auto ok = [&](int w, int h) -> bool {
                        if (w <= 0 || h <= 0) return false;
                        const size_t spatial = static_cast<size_t>(w) * static_cast<size_t>(h);
                        if (spatial == 0) return false;
                        if (out_size < spatial) return false;
                        if ((out_size % spatial) != 0) return false;
                        c = static_cast<int>(out_size / spatial);
                        ow = w;
                        oh = h;
                        return true;
                    };

                    auto ok_derived_conv2d = [&]() -> bool {
                        if (lyr.input_width <= 0 || lyr.input_height <= 0) return false;
                        const int kw = lyr.get_kernel_w();
                        const int kh = lyr.get_kernel_h();
                        const int sw = lyr.get_stride_w();
                        const int sh = lyr.get_stride_h();
                        const int pw = lyr.get_pad_w();
                        const int ph = lyr.get_pad_h();
                        if (kw <= 0 || kh <= 0 || sw <= 0 || sh <= 0) return false;
                        const int w = (lyr.input_width + 2 * pw - kw) / sw + 1;
                        const int h = (lyr.input_height + 2 * ph - kh) / sh + 1;
                        return ok(w, h);
                    };

                    auto ok_derived_deconv2d = [&]() -> bool {
                        if (lyr.input_width <= 0 || lyr.input_height <= 0) return false;
                        const int kw = lyr.get_kernel_w();
                        const int kh = lyr.get_kernel_h();
                        const int sw = lyr.get_stride_w();
                        const int sh = lyr.get_stride_h();
                        const int pw = lyr.get_pad_w();
                        const int ph = lyr.get_pad_h();
                        if (kw <= 0 || kh <= 0 || sw <= 0 || sh <= 0) return false;
                        const int w = (lyr.input_width - 1) * sw - 2 * pw + kw;
                        const int h = (lyr.input_height - 1) * sh - 2 * ph + kh;
                        return ok(w, h);
                    };

                    auto ok_derived_upsample = [&]() -> bool {
                        if (lyr.out_w > 0 && lyr.out_h > 0) return ok(lyr.out_w, lyr.out_h);
                        if (lyr.input_width <= 0 || lyr.input_height <= 0) return false;
                        const int w = std::max(1, static_cast<int>(std::lround(static_cast<double>(lyr.input_width) * std::max(0.0f, lyr.scale_w))));
                        const int h = std::max(1, static_cast<int>(std::lround(static_cast<double>(lyr.input_height) * std::max(0.0f, lyr.scale_h))));
                        return ok(w, h);
                    };

                    // 1) Métadonnées explicites
                    if (ok(lyr.output_width, lyr.output_height)) return true;
                    if (ok(lyr.out_w, lyr.out_h)) return true;

                    // 2) Dérivation à partir des paramètres spatiaux du layer.
                    if (lyr.type_enum == LayerType::ConvTranspose2d && ok_derived_deconv2d()) return true;
                    if (lyr.type_enum == LayerType::Conv2d && ok_derived_conv2d()) return true;
                    if ((lyr.type_enum == LayerType::UpsampleNearest ||
                         lyr.type_enum == LayerType::UpsampleBilinear ||
                         lyr.type_enum == LayerType::UpsampleBicubic) && ok_derived_upsample()) return true;

                    // 3) Shapes connues (si elles matchent exactement la taille).
                    // Priorité: ces shapes reflètent souvent les dimensions réelles
                    // du tenseur, alors que input_width/input_height peut rester une
                    // valeur de config (ex: 1024x1024) et fausser la première vignette.
                    auto try_shape = [&](const std::vector<int>& s) -> bool {
                        if (s.size() != 3) return false;
                        const int a = s[0];
                        const int b = s[1];
                        const int d = s[2];
                        if (a <= 0 || b <= 0 || d <= 0) return false;
                        const size_t n = static_cast<size_t>(a) * static_cast<size_t>(b) * static_cast<size_t>(d);
                        if (n != out_size) return false;

                        // Cas fréquent: Permute CHW->HWC avec shape stockée en CHW
                        // (ex: recon_to_hwc avec shape={C,H,W}, permute={1,2,0}).
                        // Ici la preview doit être interprétée en HWC: H=b, W=d, C=a.
                        if (lyr.type_enum == LayerType::Permute &&
                            lyr.permute_dims.size() == 3 &&
                            lyr.permute_dims[0] == 1 &&
                            lyr.permute_dims[1] == 2 &&
                            lyr.permute_dims[2] == 0) {
                            channels_first = false;
                            c = a;
                            ow = d;
                            oh = b;
                            return true;
                        }

                        // Permute HWC->CHW: shape décrit l'entrée, pas la sortie.
                        if (lyr.type_enum == LayerType::Permute &&
                            lyr.permute_dims.size() == 3 &&
                            lyr.permute_dims[0] == 2 &&
                            lyr.permute_dims[1] == 0 &&
                            lyr.permute_dims[2] == 1) {
                            channels_first = true;
                            c = d;
                            ow = b;
                            oh = a;
                            return true;
                        }

                        // Deux interprétations courantes: HWC (H=a,W=b,C=d) ou CHW (C=a,H=b,W=d).
                        const int out_c = lyr.out_channels;

                        // Si out_channels est renseigné, on choisit celle qui match.
                        if (out_c > 0) {
                            if (out_c == d) {
                                channels_first = false;
                                c = d;
                                ow = b;
                                oh = a;
                                return true;
                            }
                            if (out_c == a) {
                                channels_first = true;
                                c = a;
                                ow = d;
                                oh = b;
                                return true;
                            }
                        }

                        // Sinon, essayer de coller aux dims input si dispo.
                        if (lyr.input_height == a && lyr.input_width == b) {
                            channels_first = false;
                            c = d;
                            ow = b;
                            oh = a;
                            return true;
                        }
                        if (lyr.input_height == b && lyr.input_width == d) {
                            channels_first = true;
                            c = a;
                            ow = d;
                            oh = b;
                            return true;
                        }

                        if (channels_first) {
                            c = a;
                            ow = d;
                            oh = b;
                        } else {
                            c = d;
                            ow = b;
                            oh = a;
                        }
                        return true;
                    };
                    if (try_shape(lyr.shape)) return true;
                    if (try_shape(lyr.target_shape)) return true;

                    // 4) Heuristique auto-calibrée pour les entrées "raw" et tenseurs plats:
                    // tenter de reconstruire HxW via la config image + canaux usuels.
                    auto try_common_channels = [&](int w, int h, bool force_hwc = true) -> bool {
                        if (w <= 0 || h <= 0) return false;
                        const size_t spatial = static_cast<size_t>(w) * static_cast<size_t>(h);
                        if (spatial == 0) return false;
                        for (int cc : {3, 4, 1}) {
                            if (out_size == spatial * static_cast<size_t>(cc)) {
                                ow = w;
                                oh = h;
                                c = cc;
                                if (force_hwc) channels_first = false;
                                return true;
                            }
                        }
                        return false;
                    };

                    auto cfg_int = [&](const char* k) -> int {
                        if (!modelConfig.contains(k)) return 0;
                        try {
                            return std::max(0, modelConfig[k].get<int>());
                        } catch (...) {
                            return 0;
                        }
                    };

                    const int cfg_w = std::max(cfg_int("image_w"), cfg_int("width"));
                    const int cfg_h = std::max(cfg_int("image_h"), cfg_int("height"));

                    // Cas idéal: largeur/hauteur connues dans la config dataset/modèle.
                    // Conserver CHW lorsqu'il est connu (ex: recon_chw après Tanh).
                    if (try_common_channels(cfg_w, cfg_h, false)) return true;

                    // Si une seule dimension est connue, inférer l'autre avec un canal usuel.
                    if (cfg_w > 0) {
                        for (int cc : {3, 4, 1}) {
                            const size_t denom = static_cast<size_t>(cfg_w) * static_cast<size_t>(cc);
                            if (denom > 0 && (out_size % denom) == 0) {
                                const int h = static_cast<int>(out_size / denom);
                                if (h > 0) {
                                    ow = cfg_w;
                                    oh = h;
                                    c = cc;
                                    channels_first = false;
                                    return true;
                                }
                            }
                        }
                    }
                    if (cfg_h > 0) {
                        for (int cc : {3, 4, 1}) {
                            const size_t denom = static_cast<size_t>(cfg_h) * static_cast<size_t>(cc);
                            if (denom > 0 && (out_size % denom) == 0) {
                                const int w = static_cast<int>(out_size / denom);
                                if (w > 0) {
                                    ow = w;
                                    oh = cfg_h;
                                    c = cc;
                                    channels_first = false;
                                    return true;
                                }
                            }
                        }
                    }

                    // Dernière tentative stable: image carrée pour 3/4/1 canaux.
                    for (int cc : {3, 4, 1}) {
                        if ((out_size % static_cast<size_t>(cc)) != 0) continue;
                        const size_t hw = out_size / static_cast<size_t>(cc);
                        const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                        if (s > 0 && s * s == hw) {
                            ow = static_cast<int>(s);
                            oh = static_cast<int>(s);
                            c = cc;
                            channels_first = false;
                            return true;
                        }
                    }

                    // 5) En dernier recours: réutiliser les dims d'entrée déclarées.
                    if (ok(lyr.input_width, lyr.input_height)) return true;

                    // 6) Fallback final: dernier HxW connu dans ce thread
                    // (souvent valable pour activations/norm).
                    if (ok(viz_last_w, viz_last_h)) return true;

                    return false;
                };

                int ow = 0;
                int oh = 0;
                int c = 0;
                bool channels_first = false;
                const bool has_hw = infer_hw(layer, layer_output.size(), ow, oh, c, channels_first);

                if (has_hw && ow > 0 && oh > 0) {
                    const size_t spatial = static_cast<size_t>(ow) * static_cast<size_t>(oh);
                    if (spatial > 0 && layer_output.size() >= spatial) {
                        // Mémoriser pour les layers suivants (activations, etc.).
                        viz_last_w = ow;
                        viz_last_h = oh;

                        const int max_side = std::max(1, viz_taps_max_side_);
                        const int sx = (ow > max_side) ? static_cast<int>((ow + max_side - 1) / max_side) : 1;
                        const int sy = (oh > max_side) ? static_cast<int>((oh + max_side - 1) / max_side) : 1;
                        const int vw = std::max(1, ow / sx);
                        const int vh = std::max(1, oh / sy);

                        VizFrame vf;

                        auto sample_value = [&](int sample_c, size_t base) -> float {
                            if (sample_c < 0) return 0.0f;
                            const size_t idx = channels_first
                                ? (static_cast<size_t>(sample_c) * spatial + base)
                                : (base * static_cast<size_t>(std::max(1, c)) + static_cast<size_t>(sample_c));
                            return (idx < layer_output.size()) ? layer_output[idx] : 0.0f;
                        };

                        const bool name_input_like =
                            layer.name.find("/in") != std::string::npos ||
                            layer.name.find("input") != std::string::npos ||
                            layer.name.find("raw") != std::string::npos ||
                            layer.output.find("/in") != std::string::npos ||
                            layer.output.find("input") != std::string::npos ||
                            layer.output.find("raw") != std::string::npos;

                        const bool image_like_preview = (c == 3 || c == 4) &&
                            (layer.out_channels == 3 || layer.out_channels == 4 ||
                             name_input_like ||
                             layer.name.find("recon") != std::string::npos ||
                             layer.name.find("/out") != std::string::npos ||
                             layer.name.find("output") != std::string::npos ||
                             layer.output.find("recon") != std::string::npos ||
                             layer.output.find("/out") != std::string::npos ||
                             layer.output.find("output") != std::string::npos);

                        // UX: toutes les vignettes de tips sont rendues en RGB.
                        // - image_like_preview: rendu "naturel" (canaux image)
                        // - sinon: rendu faux-couleur (signed mean / énergie)
                        const bool prefer_rgb_preview = true;

                        if (prefer_rgb_preview) {
                            std::vector<float> map;
                            map.resize(static_cast<size_t>(vw) * static_cast<size_t>(vh) * 3, 0.0f);

                            for (int y = 0; y < vh; ++y) {
                                const int yy = y * sy;
                                for (int x = 0; x < vw; ++x) {
                                    const int xx = x * sx;
                                    const size_t base = (static_cast<size_t>(yy) * static_cast<size_t>(ow) + static_cast<size_t>(xx));

                                    if (image_like_preview) {
                                        for (int cc = 0; cc < 3; ++cc) {
                                            const float v = sample_value(cc, base);
                                            map[(static_cast<size_t>(y) * static_cast<size_t>(vw) + static_cast<size_t>(x)) * 3ULL + static_cast<size_t>(cc)] = v;
                                        }
                                    } else {
                                        const int c_take = std::max(1, std::min(c, 32));
                                        double sum = 0.0;
                                        double energy = 0.0;
                                        for (int chan = 0; chan < c_take; ++chan) {
                                            const double v = static_cast<double>(sample_value(chan, base));
                                            sum += v;
                                            energy += std::fabs(v);
                                        }
                                        const float mean_v = static_cast<float>(sum / static_cast<double>(c_take));
                                        const float energy_v = static_cast<float>(energy / static_cast<double>(c_take));

                                        const size_t dst = (static_cast<size_t>(y) * static_cast<size_t>(vw) + static_cast<size_t>(x)) * 3ULL;
                                        map[dst + 0] = mean_v;
                                        map[dst + 1] = energy_v;
                                        map[dst + 2] = -mean_v;
                                    }
                                }
                            }

                            float max_abs = 0.0f;
                            float max_energy = 0.0f;
                            if (image_like_preview) {
                                for (float v : map) max_abs = std::max(max_abs, std::fabs(v));
                            } else {
                                for (size_t i = 0; i < map.size(); i += 3) {
                                    max_abs = std::max(max_abs, std::max(std::fabs(map[i + 0]), std::fabs(map[i + 2])));
                                    max_energy = std::max(max_energy, std::fabs(map[i + 1]));
                                }
                            }
                            const float inv = 1.0f / (max_abs + 1e-6f);
                            const float inv_energy = 1.0f / (max_energy + 1e-6f);

                            std::vector<uint8_t> px;
                            px.resize(map.size());
                            if (image_like_preview) {
                                for (size_t i = 0; i < map.size(); ++i) {
                                    const float s = map[i] * inv;
                                    const float t = 0.5f + 0.5f * std::tanh(s);
                                    const int p = static_cast<int>(std::lround(std::clamp(t, 0.0f, 1.0f) * 255.0f));
                                    px[i] = static_cast<uint8_t>(std::clamp(p, 0, 255));
                                }
                            } else {
                                for (size_t i = 0; i < map.size(); i += 3) {
                                    const float signed_mean = map[i + 0] * inv;
                                    const float energy_norm = std::clamp(std::tanh(map[i + 1] * inv_energy), 0.0f, 1.0f);
                                    const float t = std::clamp(0.5f + 0.5f * std::tanh(signed_mean), 0.0f, 1.0f);

                                    const float neg_r = 70.0f,  neg_g = 120.0f, neg_b = 210.0f;
                                    const float mid_r = 175.0f, mid_g = 175.0f, mid_b = 175.0f;
                                    const float pos_r = 220.0f, pos_g = 65.0f,  pos_b = 65.0f;

                                    float r = 0.0f, g = 0.0f, b = 0.0f;
                                    if (t < 0.5f) {
                                        const float u = t / 0.5f;
                                        r = neg_r + (mid_r - neg_r) * u;
                                        g = neg_g + (mid_g - neg_g) * u;
                                        b = neg_b + (mid_b - neg_b) * u;
                                    } else {
                                        const float u = (t - 0.5f) / 0.5f;
                                        r = mid_r + (pos_r - mid_r) * u;
                                        g = mid_g + (pos_g - mid_g) * u;
                                        b = mid_b + (pos_b - mid_b) * u;
                                    }

                                    px[i + 0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(r * energy_norm)), 0, 255));
                                    px[i + 1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(g * energy_norm)), 0, 255));
                                    px[i + 2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(b * energy_norm)), 0, 255));
                                }
                            }

                            // RGB naturel: conserver trois canaux distincts, jamais leur moyenne.
                            if (!image_like_preview) {
                                vf.heatmap_kind = 1; // moyenne signée + énergie
                                vf.pixels_real.resize(static_cast<size_t>(vw) * vh * 3);
                                float real_max = 0.0f;
                                for (int y = 0; y < vh; ++y) for (int x = 0; x < vw; ++x) {
                                    const size_t base = static_cast<size_t>(y * sy) * ow + x * sx;
                                    for (int cc = 0; cc < std::min(c, 3); ++cc)
                                        real_max = std::max(real_max, std::fabs(sample_value(cc, base)));
                                }
                                for (int y = 0; y < vh; ++y) for (int x = 0; x < vw; ++x) {
                                    const size_t base = static_cast<size_t>(y * sy) * ow + x * sx;
                                    for (int cc = 0; cc < 3; ++cc) {
                                        const float value = sample_value(std::min(cc, c - 1), base);
                                        const float t = 0.5f + 0.5f * std::tanh(value / (real_max + 1e-6f));
                                        vf.pixels_real[(static_cast<size_t>(y) * vw + x) * 3 + cc] =
                                            static_cast<uint8_t>(std::lround(std::clamp(t, 0.0f, 1.0f) * 255.0f));
                                    }
                                }
                            }

                            vf.pixels = std::move(px);
                            vf.w = vw;
                            vf.h = vh;
                            vf.channels = 3;
                        } else {
                            // Fallback: heatmap 1 canal (moyenne de quelques canaux)
                            std::vector<float> map;
                            map.resize(static_cast<size_t>(vw) * static_cast<size_t>(vh), 0.0f);

                            const int c_take = std::max(1, std::min(c, 16));
                            for (int y = 0; y < vh; ++y) {
                                const int yy = y * sy;
                                for (int x = 0; x < vw; ++x) {
                                    const int xx = x * sx;
                                    const size_t base = (static_cast<size_t>(yy) * static_cast<size_t>(ow) + static_cast<size_t>(xx));
                                    double acc = 0.0;
                                    for (int cc = 0; cc < c_take; ++cc) {
                                        acc += static_cast<double>(sample_value(cc, base));
                                    }
                                    map[static_cast<size_t>(y) * static_cast<size_t>(vw) + static_cast<size_t>(x)] = static_cast<float>(acc / static_cast<double>(c_take));
                                }
                            }

                            float max_abs = 0.0f;
                            for (float v : map) max_abs = std::max(max_abs, std::fabs(v));
                            const float inv = 1.0f / (max_abs + 1e-6f);

                            std::vector<uint8_t> px;
                            px.resize(map.size());
                            for (size_t i = 0; i < map.size(); ++i) {
                                const float s = map[i] * inv;
                                const float t = 0.5f + 0.5f * std::tanh(s);
                                const int p = static_cast<int>(std::lround(std::clamp(t, 0.0f, 1.0f) * 255.0f));
                                px[i] = static_cast<uint8_t>(std::clamp(p, 0, 255));
                            }

                            vf.pixels = std::move(px);
                            vf.w = vw;
                            vf.h = vh;
                            vf.channels = 1;
                        }

                        vf.tensor_sample_channels = std::max(1, std::min(c, 16));
                        for (int y = 0; y < vh; ++y) for (int x = 0; x < vw; ++x) {
                            const size_t base = static_cast<size_t>(y * sy) * ow + x * sx;
                            for (int channel = 0; channel < vf.tensor_sample_channels; ++channel) {
                                vf.tensor_values.push_back(sample_value(channel, base));
                                vf.tensor_indices.push_back(channels_first
                                    ? static_cast<size_t>(channel) * spatial + base
                                    : base * std::max(1, c) + channel);
                            }
                        }

                        // UX: forcer toutes les vignettes tips en format carré XxX,
                        // avec X = largeur courante de la preview.
                        // Exemples: 32x3x3 -> 32x32x3, 128x36x3 -> 128x128x3.
                        if (vf.w > 0 && vf.h != vf.w) {
                            resize_viz_frame_nearest(vf, vf.w, vf.w);
                        }

                        vf.tensor_info = describe_tensor();
                        vf.label = canonical_viz_label();

                        if (viz_tips_custom_enabled_) {
                            (void)UpdateVizTips(layer, vf);
                        }

                        addVizTapFrame(std::move(vf));
                    }
                } else {
                    // Fallback vectoriel: projeter le tenseur 1D sur une petite heatmap 2D.
                    // Une bande 1xN devient quasiment invisible une fois mise à l'échelle dans la VIZ,
                    // ce qui donne l'impression d'une vignette noire.
                    const int max_side = std::max(1, viz_taps_max_side_);
                    const size_t nvals = layer_output.size();
                    const size_t max_pixels = static_cast<size_t>(max_side) * static_cast<size_t>(max_side);
                    const size_t sample_count = std::max<size_t>(1, std::min(nvals, max_pixels));

                    const int vw = std::max(1, std::min<int>(max_side, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(sample_count))))));
                    const int vh = std::max(1, std::min<int>(max_side, static_cast<int>((sample_count + static_cast<size_t>(vw) - 1) / static_cast<size_t>(vw))));

                    std::vector<float> map;
                    map.resize(static_cast<size_t>(vw) * static_cast<size_t>(vh), 0.0f);

                    const size_t map_count = map.size();
                    for (size_t i = 0; i < map_count; ++i) {
                        const size_t src_idx = (nvals <= map_count)
                            ? std::min(i, nvals - 1)
                            : ((i * nvals) / map_count);
                        map[i] = layer_output[std::min(src_idx, nvals - 1)];
                    }

                    float max_abs = 0.0f;
                    for (float v : map) max_abs = std::max(max_abs, std::fabs(v));
                    const float inv = 1.0f / (max_abs + 1e-6f);

                    VizFrame vf;
                    vf.pixels.resize(map.size());
                    for (size_t i = 0; i < map.size(); ++i) {
                        const float s = map[i] * inv;
                        const float t = 0.5f + 0.5f * std::tanh(s);
                        const int p = static_cast<int>(std::lround(std::clamp(t, 0.0f, 1.0f) * 255.0f));
                        vf.pixels[i] = static_cast<uint8_t>(std::clamp(p, 0, 255));
                    }
                    vf.w = vw;
                    vf.h = vh;
                    vf.channels = 1;

                    vf.tensor_values = map;
                    vf.tensor_sample_channels = 1;
                    for (size_t i = 0; i < map.size(); ++i) {
                        vf.tensor_indices.push_back(nvals <= map.size()
                            ? std::min(i, nvals - 1) : (i * nvals) / map.size());
                    }
                    vf.tensor_info = describe_tensor();
                    vf.label = canonical_viz_label() + "/vec";

                    if (viz_tips_custom_enabled_) {
                        (void)UpdateVizTips(layer, vf);
                    }

                    addVizTapFrame(std::move(vf));
                }
            }
        }

        // ====================================================================
        // STORE OUTPUT (multi-output support)
        // ====================================================================

        std::string output_name = resident_chain_executed
            ? resident_final_output_name
            : (layer.output.empty() ? "x" : layer.output);

        if (!resident_chain_executed && !exhaustive_model_viz && fusion_enabled && static_plan_.built && ((!training) || fusion_in_training)) {
            const auto& plan = static_plan_.execution;
            if (layer_idx < plan.fuse_chain_next.size()) {
                int chain_idx = plan.fuse_chain_next[layer_idx];
                while (chain_idx >= 0) {
                    const size_t fused_idx = static_cast<size_t>(chain_idx);
                    if (fused_idx >= layers.size()) break;

                    const Layer& fused_layer = layers[fused_idx];
                    if (Mimir::Planning::is_fusible_activation_layer(fused_layer)) {
                        apply_fused_layer(layer_output, fused_layer);
                    } else if (Mimir::Planning::is_fusible_unary_shape_layer(fused_layer)) {
                        apply_fused_layer(layer_output, fused_layer);
                    } else if (Mimir::Planning::is_fusible_split_layer(fused_layer)) {
                        apply_fused_layer(layer_output, fused_layer);
                    } else {
                        break;
                    }

                    output_name = fused_layer.output.empty() ? "x" : fused_layer.output;

                    if (fused_idx >= plan.fuse_chain_next.size()) break;
                    chain_idx = plan.fuse_chain_next[fused_idx];
                }
            } else {
                if (layer_idx < plan.fuse_activation_consumer.size()) {
                    const int activation_idx = plan.fuse_activation_consumer[layer_idx];
                    if (activation_idx >= 0) {
                        const Layer& activation_layer = layers[static_cast<size_t>(activation_idx)];
                        apply_fused_layer(layer_output, activation_layer);
                        output_name = activation_layer.output.empty() ? "x" : activation_layer.output;
                    }
                }
                if (layer_idx < plan.fuse_unary_consumer.size()) {
                    const int unary_idx = plan.fuse_unary_consumer[layer_idx];
                    if (unary_idx >= 0) {
                        const Layer& unary_layer = layers[static_cast<size_t>(unary_idx)];
                        apply_fused_layer(layer_output, unary_layer);
                        output_name = unary_layer.output.empty() ? "x" : unary_layer.output;
                    }
                }
                if (layer_idx < plan.fuse_split_consumer.size()) {
                    const int split_idx = plan.fuse_split_consumer[layer_idx];
                    if (split_idx >= 0) {
                        const Layer& split_layer = layers[static_cast<size_t>(split_idx)];
                        apply_fused_layer(layer_output, split_layer);
                        output_name = split_layer.output.empty() ? "x" : split_layer.output;
                    }
                }
            }
        }

        // Masque (ReLU/Dropout) ou snapshot output (Reparameterize) selon besoins.
        if (training) {
            if (needs_output_mask(layer) && forward_state.layer_output_masks.back().empty()) {
                std::vector<uint8_t> mask;
                mask.resize(layer_output.size());
                if ((layer.type == "Conv2d" || layer.type == "ConvTranspose2d") && layer.activation != ActivationType::NONE) {
                    for (size_t i = 0; i < layer_output.size(); ++i) mask[i] = (layer_output[i] > 0.0f) ? 1 : 0;
                } else {
                    for (size_t i = 0; i < layer_output.size(); ++i) mask[i] = (layer_output[i] != 0.0f) ? 1 : 0;
                }
                forward_state.layer_output_masks.back() = std::move(mask);
            }
            if (needs_output_snapshot(layer)) {
                forward_state.layer_outputs.back() = layer_output;
            }
        }

        if (planned_host_reuse_enabled && layer_idx < static_plan_.execution.layers.size()) {
            const auto& planned_layer = static_plan_.execution.layers[layer_idx];
            // Inputs whose last logical use is this layer can be returned to
            // their planned physical slot before the output is materialized.
            for (const auto& input_id : planned_layer.inputs) {
                const auto tensor_it = static_plan_.execution.tensors.find(input_id);
                if (tensor_it == static_plan_.execution.tensors.end()) continue;
                const auto& planned_tensor = tensor_it->second;
                if (planned_tensor.last_use != layer_idx || planned_tensor.persistent ||
                    planned_tensor.required_for_backward ||
                    planned_tensor.physical_buffer_id == std::numeric_limits<size_t>::max()) continue;
                auto live = tensor_store.find(planned_tensor.name);
                if (live == tensor_store.end()) continue;
                std::vector<float> released = std::move(live->second);
                tensor_store.erase(live);
                typed_tensor_store.erase(planned_tensor.name);
                if (poison_reused_buffers) {
                    std::fill(released.begin(), released.end(), std::numeric_limits<float>::quiet_NaN());
                }
                planned_host_buffer_pool[planned_tensor.physical_buffer_id] = std::move(released);
            }

            const auto output_it = static_plan_.execution.tensors.find(planned_layer.output);
            if (output_it != static_plan_.execution.tensors.end()) {
                const size_t slot = output_it->second.physical_buffer_id;
                auto reusable = planned_host_buffer_pool.find(slot);
                if (slot != std::numeric_limits<size_t>::max() && reusable != planned_host_buffer_pool.end() &&
                    reusable->second.capacity() >= layer_output.size()) {
                    std::vector<float> recycled = std::move(reusable->second);
                    planned_host_buffer_pool.erase(reusable);
                    recycled.assign(layer_output.begin(), layer_output.end());
                    layer_output.swap(recycled);
                    ++actual_buffer_reuse_count;
                    actual_buffer_reuse_bytes += layer_output.size() * sizeof(float);
                }
            }
        }

        storeTensor(output_name, std::move(layer_output));

        const auto& layer_out_view = getTensor(output_name);

        const size_t layer_out_bytes = hasTypedTensor(output_name)
            ? getTypedTensor(output_name).size_bytes()
            : layer_out_view.size() * sizeof(float);
        if (executed_backend.find("VULKAN") != std::string::npos) {
            backend_mem_attrib.vulkan_bytes += layer_out_bytes;
        } else if (executed_backend.find("CUDA") != std::string::npos) {
            backend_mem_attrib.cuda_bytes += layer_out_bytes;
        } else if (executed_backend.find("ROCM") != std::string::npos) {
            backend_mem_attrib.rocm_bytes += layer_out_bytes;
        } else if (executed_backend.find("CPU") != std::string::npos || executed_backend == "cpu_switch_kernel") {
            backend_mem_attrib.cpu_bytes += layer_out_bytes;
        } else {
            backend_mem_attrib.other_bytes += layer_out_bytes;
        }

        if (training && has_branches) {
            all_layer_outputs.push_back(layer_out_view);
        }

        if (runtime_trace) {
            std::cerr << "[runtime-trace] layer#" << layer_idx
                      << " name='" << layer.name
                      << "' type='" << (layer.type.empty() ? type_to_string(layer.type_enum) : layer.type)
                      << "' backend=" << executed_backend
                      << " call=" << executed_call
                      << " output_size=" << layer_out_view.size()
                      << std::endl;
        }

        // Gestion des branches
        if (layer.requiresBranchComputation() && training) {
            executeBranchComputation(layer_idx, all_layer_outputs, training);
            // Update tensor store avec le résultat mergé
            storeTensor(output_name, all_layer_outputs[layer_idx]);
        }
    }

    if (planned_host_reuse_enabled) {
        static_plan_.execution.stats.buffer_reuse_count = actual_buffer_reuse_count;
        static_plan_.execution.stats.buffer_reuse_bytes = actual_buffer_reuse_bytes;
        if (planner_to_terminal) {
            std::cerr << "[planner] host_buffer_reuse_count=" << actual_buffer_reuse_count
                      << " host_buffer_reuse_bytes=" << actual_buffer_reuse_bytes << std::endl;
        }
    }

    // Le résultat final est toujours dans "x" (ou dernier output)

    // ====================================================================
    // VAEConv extra viz frames: MU + resdiff (best-effort)
    // ====================================================================
    // Objectif UX: dans la Viz VAE_conv, toujours montrer le latent MU et
    // une heatmap d'erreur de reconstruction (|recon - input|).
    if (viz_taps_enabled_ && viz_taps_max_frames_ > 0) {
        auto downsample_mean_chw_to_gray = [&](
            const std::vector<float>& tensor,
            int W,
            int H,
            int C,
            int max_side,
            bool take_abs,
            bool symmetric
        ) -> VizFrame {
            VizFrame vf;
            if (W <= 0 || H <= 0 || C <= 0) return vf;
            const size_t expected = static_cast<size_t>(W) * static_cast<size_t>(H) * static_cast<size_t>(C);
            if (tensor.size() != expected) return vf;

            const int vw = std::max(1, std::min(max_side, W));
            const int vh = std::max(1, std::min(max_side, H));

            std::vector<float> map;
            map.resize(static_cast<size_t>(vw) * static_cast<size_t>(vh), 0.0f);

            for (int y = 0; y < vh; ++y) {
                const int sy = (vh > 1) ? (y * H) / vh : 0;
                for (int x = 0; x < vw; ++x) {
                    const int sx = (vw > 1) ? (x * W) / vw : 0;
                    double acc = 0.0;
                    const size_t base = (static_cast<size_t>(sy) * static_cast<size_t>(W) + static_cast<size_t>(sx)) * static_cast<size_t>(C);
                    for (int c = 0; c < C; ++c) {
                        float v = tensor[base + static_cast<size_t>(c)];
                        if (take_abs) v = std::fabs(v);
                        acc += static_cast<double>(v);
                    }
                    map[static_cast<size_t>(y) * static_cast<size_t>(vw) + static_cast<size_t>(x)] = static_cast<float>(acc / static_cast<double>(C));
                }
            }

            float max_abs = 0.0f;
            for (float v : map) {
                max_abs = std::max(max_abs, std::fabs(v));
            }
            const float inv = 1.0f / (max_abs + 1e-6f);

            vf.pixels.resize(map.size());
            for (size_t i = 0; i < map.size(); ++i) {
                const float s = map[i] * inv;
                float t = 0.0f;
                if (symmetric) {
                    // [-1, 1] -> [0, 1]
                    t = 0.5f + 0.5f * std::tanh(s);
                } else {
                    // [0, +] -> [0, 1]
                    t = std::clamp(s, 0.0f, 1.0f);
                }
                const int p = static_cast<int>(std::lround(std::clamp(t, 0.0f, 1.0f) * 255.0f));
                vf.pixels[i] = static_cast<uint8_t>(std::clamp(p, 0, 255));
            }
            vf.w = vw;
            vf.h = vh;
            vf.channels = 1;
            return vf;
        };

        auto add_mu_frame = [&](const char* prefix) {
            const std::string mu_name = std::string(prefix) + "/mu";
            const std::string mu_layer = std::string(prefix) + "/enc/mu";
            const bool has_canonical_mu_frame = std::any_of(
                viz_taps_.begin(), viz_taps_.end(), [&](const VizFrame& frame) {
                    return frame.label.find(mu_layer + "/") != std::string::npos;
                });
            if (has_canonical_mu_frame) return;
            if (!hasTensor(mu_name)) return;
            const auto& mu = getTensor(mu_name);
            int muW = 0, muH = 0, muC = 0;
            if (Layer* L = getLayerByName(mu_layer)) {
                muC = std::max(0, L->out_channels);
                muH = std::max(0, L->input_height);
                muW = std::max(0, L->input_width);
            }
            if (muC > 0 && (muW <= 0 || muH <= 0)) {
                const size_t hw = mu.size() / static_cast<size_t>(muC);
                const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                if (s > 0 && s * s == hw) {
                    muH = static_cast<int>(s);
                    muW = static_cast<int>(s);
                }
            }
            if (muW > 0 && muH > 0 && muC > 0 && mu.size() == static_cast<size_t>(muW) * static_cast<size_t>(muH) * static_cast<size_t>(muC)) {
                VizFrame vf = downsample_mean_chw_to_gray(mu, muW, muH, muC, std::max(1, viz_taps_max_side_), /*take_abs*/false, /*symmetric*/true);
                if (!vf.pixels.empty()) {
                    vf.label = std::string(prefix) + "/latent/mu";
                    addVizTapFrame(std::move(vf));
                }
            }
        };

        // MU: tensor spatial CHW (best-effort via layer config)
        add_mu_frame("vae_conv");

        auto add_resdiff_frame = [&](const char* prefix) {
            const std::string recon_name = std::string(prefix) + "/recon";
            const std::string in_name = std::string(prefix) + "/in_hwc";
            const std::string recon_to_hwc = std::string(prefix) + "/recon_to_hwc";
            if (!hasTensor(recon_name) || !hasTensor(in_name)) return;
            const auto& recon = getTensor(recon_name);
            const auto& in_hwc = getTensor(in_name);
            if (recon.size() != in_hwc.size() || recon.empty()) return;

            int W = 0, H = 0, C = 0;
            if (Layer* P = getLayerByName(recon_to_hwc)) {
                if (P->shape.size() == 3) {
                    C = std::max(0, P->shape[0]);
                    H = std::max(0, P->shape[1]);
                    W = std::max(0, P->shape[2]);
                }
            }
            if ((W <= 0 || H <= 0 || C <= 0) && recon.size() % 3 == 0) {
                C = 3;
                const size_t hw = recon.size() / 3ULL;
                const size_t s = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(hw))));
                if (s > 0 && s * s == hw) {
                    H = static_cast<int>(s);
                    W = static_cast<int>(s);
                }
            }
            if (W <= 0 || H <= 0 || C <= 0) return;
            if (recon.size() != static_cast<size_t>(W) * static_cast<size_t>(H) * static_cast<size_t>(C)) return;

            std::vector<float> diff;
            diff.resize(recon.size());
            #pragma omp simd
            for (size_t i = 0; i < diff.size(); ++i) {
                diff[i] = std::fabs(recon[i] - in_hwc[i]);
            }
            VizFrame vf = downsample_mean_chw_to_gray(diff, W, H, C, std::max(1, viz_taps_max_side_), /*take_abs*/false, /*symmetric*/false);
            if (!vf.pixels.empty()) {
                vf.label = std::string(prefix) + "/err/resdiff_abs";
                addVizTapFrame(std::move(vf));
            }
        };

        // resdiff: |recon - input| en image-space (HWC)
        add_resdiff_frame("vae_conv");
    }

    if (allocator_log) {
        allocator.log_stats(training ? "forward(training)" : "forward(inference)", allocator_log_verbose, &backend_mem_attrib);
    }

    if (nested_viz_model && viz_capture_root != nullptr) {
        auto child_frames = std::move(viz_taps_);
        viz_taps_.clear();
        for (auto& frame : child_frames) {
            viz_capture_root->addVizTapFrame(std::move(frame));
        }
    }

    return getTensor(forward_output_name);
}

std::vector<float> Model::forwardPromptImageSeed(const std::vector<float>& prompt_vec,
                                                 const std::vector<float>& image_vec,
                                                 uint32_t seed,
                                                 bool training) {
    std::vector<float> packed;
    packed.reserve(prompt_vec.size() + image_vec.size());
    packed.insert(packed.end(), prompt_vec.begin(), prompt_vec.end());
    packed.insert(packed.end(), image_vec.begin(), image_vec.end());

    MimirRng::ScopedSeed scoped(seed);
    return forwardPass(packed, training);
}

Gradients Model::backwardPass(const std::vector<float> &loss_gradient) {
    const auto model_dtype = Mimir::parse_dtype(default_dtype_);
    if (!Mimir::dtype_is_floating(model_dtype)) {
        throw std::runtime_error(
            std::string("Model::backwardPass: dtype '") +
            Mimir::dtype_to_string(model_dtype) + "' is not differentiable");
    }
    if (params_frozen_) {
        throw std::runtime_error("Model::backwardPass: parameters are frozen");
    }
    Gradients grads;

    if (!forward_state.is_valid) {
        std::cerr << "⚠️  Cannot perform backward pass: no valid forward state" << std::endl;
        std::cerr << "    Call forwardPass() in training mode first" << std::endl;
        return grads;
    }

    if (layers.empty()) {
        std::cerr << "⚠️  Cannot perform backward pass: layers or weights not initialized" << std::endl;
        return grads;
    }

    auto accumulate_grad = [](std::vector<float>& dst, const std::vector<float>& src) {
        if (dst.empty()) { dst = src; return; }
        if (dst.size() != src.size()) throw std::runtime_error("Gradient accumulate: size mismatch");
        for (size_t i = 0; i < dst.size(); ++i) dst[i] += src[i];
    };
    std::unordered_map<std::string, std::vector<float>> grad_store;
    grad_store.reserve(layers.size() + 2);
    grad_store["x"] = loss_gradient;
    for (size_t index = layers.size(); index-- > 0;) {
        Layer& layer = layers[index];
        if (!layer.shared_weights_from.empty()) {
            Layer* owner = getLayerByName(layer.shared_weights_from);
            if (!owner || !owner->weight_block) throw std::runtime_error("Shared weight source unavailable: " + layer.name);
            layer.weight_block = owner->weight_block;
        }
        const std::string output = layer.output.empty() ? "x" : layer.output;
        const bool split = layer.type_enum == LayerType::Split || layer.type_enum == LayerType::Chunk;
        std::vector<std::vector<float>> owned_grad_outputs;
        std::vector<std::string> consumed_outputs;
        if (split) {
            const bool explicit_split = layer.type_enum == LayerType::Split && !layer.split_sizes.empty();
            const size_t count = explicit_split ? layer.split_sizes.size() :
                static_cast<size_t>(std::max(0, layer.type_enum == LayerType::Split ? layer.num_splits : layer.num_chunks));
            bool active = grad_store.count(output) != 0;
            for (size_t i = 0; i < count; ++i) active |= grad_store.count(output + "_" + std::to_string(i)) != 0;
            if (!active) continue;
            const size_t total = forward_state.layer_input_sizes_multi.at(index).at(0);
            for (size_t i = 0; i < count; ++i) {
                const std::string name = output + "_" + std::to_string(i);
                const size_t n = explicit_split ? static_cast<size_t>(layer.split_sizes[i]) :
                    total / count + (i < total % count ? 1 : 0);
                owned_grad_outputs.emplace_back(n, 0.0f);
                auto it = grad_store.find(name);
                if (it != grad_store.end()) accumulate_grad(owned_grad_outputs.back(), it->second);
                if (i == 0 && grad_store.count(output)) accumulate_grad(owned_grad_outputs.back(), grad_store.at(output));
                consumed_outputs.push_back(name);
            }
        } else {
            auto it = grad_store.find(output);
            if (it == grad_store.end()) continue;
            owned_grad_outputs.push_back(std::move(it->second));
        }
        consumed_outputs.push_back(output);
        // Consume this tensor version before accumulating into its producers.
        // This matters when input and output are both named "x".
        for (const auto& name : consumed_outputs) grad_store.erase(name);
        const auto& input_names = forward_state.layer_input_names.at(index);
        const auto& snapshots = forward_state.layer_inputs_multi.at(index);
        const auto& sizes = forward_state.layer_input_sizes_multi.at(index);
        std::vector<std::vector<float>> shape_only_inputs;
        std::vector<const std::vector<float>*> inputs, grad_outputs;
        if (snapshots.size() == input_names.size()) {
            for (const auto& value : snapshots) inputs.push_back(&value);
        } else {
            // Shape-only backward kernels do not read these values. Never read
            // TensorStore here: later layers may have overwritten its contents.
            shape_only_inputs.reserve(sizes.size());
            inputs.reserve(sizes.size());
            for (size_t size : sizes) {
                const std::vector<float>* shape = nullptr;
                for (const auto& value : owned_grad_outputs)
                    if (value.size() == size) { shape = &value; break; }
                if (!shape) {
                    for (const auto& value : shape_only_inputs)
                        if (value.size() == size) { shape = &value; break; }
                }
                if (!shape) {
                    shape_only_inputs.emplace_back(size, 0.0f);
                    shape = &shape_only_inputs.back();
                }
                inputs.push_back(shape);
            }
        }
        for (const auto& value : owned_grad_outputs) grad_outputs.push_back(&value);
        RuntimeBackwardContext context;
        context.skip_connections_enabled = forward_state.skip_connections_enabled;
        const auto& saved_output = forward_state.layer_outputs.at(index);
        const auto& saved_mask = forward_state.layer_output_masks.at(index);
        if (!saved_output.empty()) context.output = &saved_output;
        if (!saved_mask.empty()) context.output_mask = &saved_mask;
        std::vector<std::vector<float>> grad_inputs;
        if (!RuntimeRouter::instance().dispatchBackwardLayer(inputs, grad_outputs, grad_inputs, layer, true, nullptr, context))
            throw std::runtime_error("Runtime backward failed for layer '" + layer.name + "' (" + layer.type + ")");
        // Runtime grad_bias is a compatibility view of packed gradients.
        // Model optimizers and norm/clipping calculations use the packed block once.
        layer.grad_bias.clear();
        if (grad_inputs.size() != input_names.size() && layer.type_enum != LayerType::Constant)
            throw std::runtime_error("Runtime backward input count mismatch: " + layer.name);
        if (!layer.shared_weights_from.empty()) {
            Layer* owner = getLayerByName(layer.shared_weights_from);
            if (!owner) throw std::runtime_error("Shared parameter owner unavailable: " + layer.name);
            accumulate_grad(owner->grad_weights, layer.grad_weights);
            layer.grad_weights.clear();
        }
        for (size_t i = 0; i < grad_inputs.size(); ++i) {
            auto& destination = grad_store[input_names[i]];
            if (destination.empty()) destination = std::move(grad_inputs[i]);
            else accumulate_grad(destination, grad_inputs[i]);
        }
    }

    // Exposer le gradient d'entrée si présent (architecture utilisant "__input__")
    has_last_input_gradient_ = false;
    last_input_gradient_.clear();
    auto it_in = grad_store.find("__input__");
    if (it_in != grad_store.end()) {
        last_input_gradient_ = std::move(it_in->second);
        has_last_input_gradient_ = true;
    }


    return grads;
}

float Model::computeLoss(const std::vector<float> &prediction, 
                        const std::vector<float> &target, 
                        const std::string &loss_type) {
    if (prediction.size() != target.size()) {
        std::cerr << "⚠️  Prediction and target size mismatch" << std::endl;
        return 0.0f;
    }
    return static_cast<float>(
        RuntimeLossGrad::pixel_loss_and_grad(prediction, target, loss_type).loss);
}

std::vector<float> Model::computeLossGradient(const std::vector<float> &prediction,
                                              const std::vector<float> &target,
                                              const std::string &loss_type) {
    std::vector<float> gradient;
    computeLossGradientInto(prediction, target, gradient, loss_type);
    return gradient;
}

void Model::computeLossGradientInto(const std::vector<float> &prediction,
                                    const std::vector<float> &target,
                                    std::vector<float> &gradient,
                                    const std::string &loss_type) {
    if (prediction.size() != target.size()) {
        gradient.resize(prediction.size());
        std::fill(gradient.begin(), gradient.end(), 0.0f);
        return;
    }
    gradient = RuntimeLossGrad::pixel_loss_and_grad(
        prediction, target, loss_type).grad;
}

// === build & autoBuildFromDataset ===

void Model::build()
{
    // Construction générique du modèle
    // Peut être surchargée pour définir une architecture spécifique
    
    // Exemple: backbone U-Net simple
    buildBackboneUNet(4, 2, 3);  // 4 stages, 2 blocs par stage, 3 blocs bottleneck
    
    std::cerr << "Model::build() - Architecture construite" << std::endl;
    std::cerr << "  Couches: " << layers.size() << std::endl;
    std::cerr << "  Paramètres totaux: " << totalParamCount() << std::endl;
    
    // Allocation automatique des paramètres
    size_t total = totalParamCount();
    if (total > 0) {
        std::cerr << "  Allocation des paramètres..." << std::endl;
        allocateParams();
        std::cerr << "  ✓ " << layer_weight_blocks.size() << " blocs de poids alloués" << std::endl;
        
        // Initialisation automatique des poids (méthode He par défaut)
        std::cerr << "  Initialisation des poids (He)..." << std::endl;
        initializeWeights("he", 0);
        std::cerr << "  ✓ Poids initialisés" << std::endl;
    }
}

void Model::autoBuildFromDataset(const std::string &dataset_dir)
{
    // Analyse automatique du dataset pour construire l'architecture appropriée
    
    std::cerr << "Model::autoBuildFromDataset(" << dataset_dir << ")" << std::endl;
    
    // Charger le dataset avec cache et validation flexible (min 1 modalité)
    std::vector<DatasetItem> items;
    try {
        items = loadDatasetCached(dataset_dir, 64, 64, 1);  // min_modalities = 1
    } catch (const std::exception &e) {
        std::cerr << "Erreur chargement dataset: " << e.what() << std::endl;
        // Fallback: construction par défaut
        build();
        return;
    }
    
    if (items.empty()) {
        std::cerr << "Dataset vide, construction par défaut" << std::endl;
        build();
        return;
    }
    
    std::cerr << "  Items trouvés: " << items.size() << std::endl;
    
    // Analyser les modalités présentes et les linkables
    bool has_text = false;
    bool has_image = false;
    bool has_audio = false;
    bool has_video = false;
    size_t linkable_count = 0;
    
    for (const auto &item : items) {
        if (!item.text_file.empty()) has_text = true;
        if (!item.image_file.empty()) has_image = true;
        if (!item.audio_file.empty()) has_audio = true;
        if (!item.video_file.empty()) has_video = true;
        if (item.is_linked && item.countModalities() >= 2) linkable_count++;
    }
    
    std::cerr << "  Modalités détectées:" << std::endl;
    std::cerr << "    - Texte:  " << (has_text ? "✓" : "✗") << std::endl;
    std::cerr << "    - Image:  " << (has_image ? "✓" : "✗") << std::endl;
    std::cerr << "    - Audio:  " << (has_audio ? "✓" : "✗") << std::endl;
    std::cerr << "    - Vidéo:  " << (has_video ? "✓" : "✗") << std::endl;
    std::cerr << "    - Linkables validés: " << linkable_count << std::endl;
    
    // Construire le backbone de base
    buildBackboneUNet(4, 2, 3);
    
    // Créer les magic tokens pour chaque modalité détectée
    std::vector<MagicToken> magic_tokens;
    
    if (has_text) {
        MagicToken tok;
        tok.modality_mask = 0x01;  // bit 0 = text
        tok.seed = 42;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.1f * (i + 1);
        magic_tokens.push_back(tok);
        buildTextBranch(tok);
        injectMagicToken(tok);
        std::cerr << "  → Branche texte ajoutée" << std::endl;
    }
    
    if (has_image) {
        MagicToken tok;
        tok.modality_mask = 0x02;  // bit 1 = image
        tok.seed = 43;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.2f * (i + 1);
        magic_tokens.push_back(tok);
        buildImageBranch(tok);
        injectMagicToken(tok);
        std::cerr << "  → Branche image ajoutée" << std::endl;
    }
    
    if (has_audio) {
        MagicToken tok;
        tok.modality_mask = 0x04;  // bit 2 = audio
        tok.seed = 44;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.3f * (i + 1);
        magic_tokens.push_back(tok);
        buildAudioBranch(tok);
        injectMagicToken(tok);
        std::cerr << "  → Branche audio ajoutée" << std::endl;
    }
    
    if (has_video) {
        MagicToken tok;
        tok.modality_mask = 0x08;  // bit 3 = video
        tok.seed = 45;
        for (int i = 0; i < 8; ++i) tok.embed[i] = 0.4f * (i + 1);
        magic_tokens.push_back(tok);
        buildVideoBranch(tok);
        injectMagicToken(tok);
        std::cerr << "  → Branche vidéo ajoutée" << std::endl;
    }
    
    std::cerr << "  Architecture auto-construite:" << std::endl;
    std::cerr << "    - Couches: " << layers.size() << std::endl;
    std::cerr << "    - Paramètres: " << totalParamCount() << std::endl;
    std::cerr << "    - Magic tokens: " << magic_tokens.size() << std::endl;
}

// --- Fin ---

// --- Fin ---
