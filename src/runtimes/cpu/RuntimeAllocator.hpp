#pragma once

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <iomanip>
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#include "tensors.hpp"
#include "DType.hpp"

// ============================================================================
// RuntimeAllocator - Gestionnaire mémoire strict pour Mímir
// ============================================================================
// 
// Objectif : TOUTE allocation runtime (activations, buffers temporaires, 
// intermédiaires) doit passer par ce gestionnaire pour respecter la limite RAM.
//
// Fonctionnalités :
// - Comptabilisation automatique via MemoryGuard
// - RAII handles pour auto-release
// - Pool de buffers réutilisables (scratchpad)
// - Tracking des allocations (debug)
// - Interdiction des allocations sauvages (std::vector)
//
// Usage :
//   RuntimeAllocator alloc(memory_guard, max_ram_mb);
//   auto tensor = alloc.allocate_tensor({batch, channels, h, w}, "conv1_out");
//   auto buffer = alloc.allocate_buffer(1024 * 1024, "temp_gemm");
//   // Auto-release via RAII
// ============================================================================

// Forward declarations
class MemoryGuard;
class DynamicTensorAllocator;

// Tensor descriptor pour shape + metadata
struct TensorDescriptor {
    std::vector<int> shape;         // [N, C, H, W] ou autre
    std::string dtype = "float32";  // Pour future support FP16/INT8
    Mimir::DType dtype_enum = Mimir::DType::F32;
    size_t size_bytes = 0;
    std::string name;
    
    TensorDescriptor() = default;
    TensorDescriptor(std::vector<int> s, const std::string& dt = "float32", const std::string& n = "")
        : shape(std::move(s)), dtype(dt), name(n) {
        size_t numel = 1;
        for (int dim : shape) numel *= dim;

        dtype_enum = Mimir::parse_dtype(dtype);
        const size_t elt = Mimir::dtype_size_bytes(dtype_enum);
        if (elt == 0) {
            throw std::runtime_error("Unsupported dtype: " + dtype);
        }
        size_bytes = numel * elt;
    }
    
    size_t numel() const {
        size_t n = 1;
        for (int dim : shape) n *= dim;
        return n;
    }
};

// RAII Handle pour buffer temporaire
class BufferHandle {
public:
    BufferHandle(float* ptr, size_t bytes, const std::string& tag, MemoryGuard* guard)
        : data_(ptr), size_bytes_(bytes), tag_(tag), guard_(guard) {}
    
    ~BufferHandle() {
        if (data_) {
            delete[] data_;
            data_ = nullptr;
        }
        if (guard_) {
            // Libération automatique (tracking)
            guard_->releaseAllocation(size_bytes_);
        }
    }
    
    // Non-copyable, movable
    BufferHandle(const BufferHandle&) = delete;
    BufferHandle& operator=(const BufferHandle&) = delete;
    
    BufferHandle(BufferHandle&& other) noexcept
        : data_(other.data_), size_bytes_(other.size_bytes_),
          tag_(std::move(other.tag_)), guard_(other.guard_) {
        other.data_ = nullptr;
        other.guard_ = nullptr;
    }
    
    BufferHandle& operator=(BufferHandle&& other) noexcept {
        if (this != &other) {
            // Release current
            if (data_) {
                delete[] data_;
                data_ = nullptr;
            }
            if (guard_) guard_->releaseAllocation(size_bytes_);
            
            data_ = other.data_;
            size_bytes_ = other.size_bytes_;
            tag_ = std::move(other.tag_);
            guard_ = other.guard_;
            
            other.data_ = nullptr;
            other.guard_ = nullptr;
        }
        return *this;
    }
    
    float* data() { return data_; }
    const float* data() const { return data_; }
    size_t size() const { return size_bytes_ / sizeof(float); }
    size_t size_bytes() const { return size_bytes_; }
    const std::string& tag() const { return tag_; }
    
private:
    float* data_;
    size_t size_bytes_;
    std::string tag_;
    MemoryGuard* guard_;
};

// RAII Handle pour Tensor (wrapper autour de std::vector mais comptabilisé)
class TensorHandle {
public:
    TensorHandle(std::vector<float>&& data, const TensorDescriptor& desc, MemoryGuard* guard)
        : data_(std::move(data)), descriptor_(desc), guard_(guard) {}
    
    ~TensorHandle() {
        if (guard_ && !data_.empty()) {
            guard_->releaseAllocation(descriptor_.size_bytes);
        }
    }
    
    // Non-copyable, movable
    TensorHandle(const TensorHandle&) = delete;
    TensorHandle& operator=(const TensorHandle&) = delete;
    
    TensorHandle(TensorHandle&& other) noexcept
        : data_(std::move(other.data_)), descriptor_(std::move(other.descriptor_)), guard_(other.guard_) {
        other.guard_ = nullptr;
    }
    
    TensorHandle& operator=(TensorHandle&& other) noexcept {
        if (this != &other) {
            if (guard_ && !data_.empty()) guard_->releaseAllocation(descriptor_.size_bytes);
            
            data_ = std::move(other.data_);
            descriptor_ = std::move(other.descriptor_);
            guard_ = other.guard_;
            
            other.guard_ = nullptr;
        }
        return *this;
    }
    
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }
    const TensorDescriptor& descriptor() const { return descriptor_; }
    
    size_t size() const { return data_.size(); }
    float* ptr() { return data_.data(); }
    const float* ptr() const { return data_.data(); }
    
private:
    std::vector<float> data_;
    TensorDescriptor descriptor_;
    MemoryGuard* guard_;
};

// RAII Handle pour Tensor "raw" (bytes) - permet de gérer tous les dtypes.
// Important: ce handle n'impose pas d'interprétation; il fournit juste une vue typée optionnelle.
class RawTensorHandle {
public:
    RawTensorHandle(std::unique_ptr<uint8_t[]>&& data, const TensorDescriptor& desc, MemoryGuard* guard)
        : data_(std::move(data)), descriptor_(desc), guard_(guard) {}

    ~RawTensorHandle() {
        if (guard_ && descriptor_.size_bytes) {
            guard_->releaseAllocation(descriptor_.size_bytes);
        }
    }

    RawTensorHandle(const RawTensorHandle&) = delete;
    RawTensorHandle& operator=(const RawTensorHandle&) = delete;

    RawTensorHandle(RawTensorHandle&& other) noexcept
        : data_(std::move(other.data_)), descriptor_(std::move(other.descriptor_)), guard_(other.guard_) {
        other.guard_ = nullptr;
    }

    RawTensorHandle& operator=(RawTensorHandle&& other) noexcept {
        if (this != &other) {
            if (guard_ && descriptor_.size_bytes) guard_->releaseAllocation(descriptor_.size_bytes);
            data_ = std::move(other.data_);
            descriptor_ = std::move(other.descriptor_);
            guard_ = other.guard_;
            other.guard_ = nullptr;
        }
        return *this;
    }

    uint8_t* data_bytes() { return data_.get(); }
    const uint8_t* data_bytes() const { return data_.get(); }
    size_t size_bytes() const { return descriptor_.size_bytes; }
    const TensorDescriptor& descriptor() const { return descriptor_; }

    template <typename T>
    T* data_as() { return reinterpret_cast<T*>(data_.get()); }
    template <typename T>
    const T* data_as() const { return reinterpret_cast<const T*>(data_.get()); }

private:
    std::unique_ptr<uint8_t[]> data_;
    TensorDescriptor descriptor_;
    MemoryGuard* guard_;
};

// ============================================================================
// RuntimeAllocator - Main class
// ============================================================================

class RuntimeAllocator {
public:
    struct BackendMemoryAttribution {
        size_t cpu_bytes = 0;
        size_t vulkan_bytes = 0;
        size_t cuda_bytes = 0;
        size_t rocm_bytes = 0;
        size_t other_bytes = 0;
    };

    RuntimeAllocator(MemoryGuard& guard, size_t max_ram_mb = 4096)
        : memory_guard_(guard), max_ram_bytes_(max_ram_mb * 1024ULL * 1024ULL) {
        // IMPORTANT: ne jamais modifier la limite globale du MemoryGuard ici.
        // Le RuntimeAllocator doit respecter la limite déjà configurée ailleurs (Lua/config).
        // max_ram_bytes_ sert uniquement de cap local optionnel.
    }
    
    // Allocation d'un Tensor avec shape
    TensorHandle allocate_tensor(const std::vector<int>& shape,
                                  const std::string& dtype = "float32",
                                  const std::string& name = "") {
        TensorDescriptor desc(shape, dtype, name);

        // Le framework gère les dtypes (taille, validation), mais ce handle float32
        // est volontairement limité à float32 pour éviter des interprétations silencieuses.
        if (desc.dtype_enum != Mimir::DType::F32) {
            throw std::runtime_error(
                "RuntimeAllocator: allocate_tensor only supports float32 for now (requested '" + dtype + "'). "
                "Use allocate_raw_tensor for other dtypes."
            );
        }

        // Cap local optionnel (sans toucher à la limite MemoryGuard)
        if (max_ram_bytes_ > 0 && (memory_guard_.getCurrentBytes() + desc.size_bytes) > max_ram_bytes_) {
            throw std::runtime_error(
                "RuntimeAllocator: Cannot allocate tensor '" + name + "' (" +
                std::to_string(desc.size_bytes / (1024*1024)) + " MB) - " +
                "would exceed local RAM cap. Current: " +
                std::to_string(memory_guard_.getCurrentBytes() / (1024*1024)) + " MB / " +
                std::to_string(max_ram_bytes_ / (1024*1024)) + " MB"
            );
        }
        
        // Vérifier limite avant allocation
        if (!memory_guard_.requestAllocation(desc.size_bytes, name)) {
            throw std::runtime_error(
                "RuntimeAllocator: Cannot allocate tensor '" + name + "' (" +
                std::to_string(desc.size_bytes / (1024*1024)) + " MB) - " +
                "would exceed RAM limit. Current: " +
                std::to_string(memory_guard_.getCurrentBytes() / (1024*1024)) + " MB / " +
                std::to_string(memory_guard_.getLimit() / (1024*1024)) + " MB"
            );
        }
        
        // Allouer (comptabilisé par MemoryGuard via reserve)
        std::vector<float> data(desc.numel(), 0.0f);
        
        // Tracking (optionnel, debug)
        total_allocated_bytes_ += desc.size_bytes;
        num_allocations_++;
        
        return TensorHandle(std::move(data), desc, &memory_guard_);
    }

    // Allocation d'un tensor raw (bytes) pour tous dtypes.
    RawTensorHandle allocate_raw_tensor(const std::vector<int>& shape,
                                        const std::string& dtype,
                                        const std::string& name = "") {
        TensorDescriptor desc(shape, dtype, name);

        // Cap local optionnel (sans toucher à la limite MemoryGuard)
        if (max_ram_bytes_ > 0 && (memory_guard_.getCurrentBytes() + desc.size_bytes) > max_ram_bytes_) {
            throw std::runtime_error(
                "RuntimeAllocator: Cannot allocate raw tensor '" + name + "' (" +
                std::to_string(desc.size_bytes / (1024*1024)) + " MB) - would exceed local RAM cap"
            );
        }

        if (!memory_guard_.requestAllocation(desc.size_bytes, name)) {
            throw std::runtime_error(
                "RuntimeAllocator: Cannot allocate raw tensor '" + name + "' (" +
                std::to_string(desc.size_bytes / (1024*1024)) + " MB) - would exceed RAM limit"
            );
        }

        auto mem = std::unique_ptr<uint8_t[]>(new uint8_t[desc.size_bytes]);
        std::memset(mem.get(), 0, desc.size_bytes);

        total_allocated_bytes_ += desc.size_bytes;
        num_allocations_++;

        return RawTensorHandle(std::move(mem), desc, &memory_guard_);
    }
    
    // Allocation d'un buffer brut (pour calculs temporaires)
    BufferHandle allocate_buffer(size_t bytes, const std::string& tag = "") {
        return allocate_buffer_impl(bytes, tag, true);
    }

    // Kernels using this entry point overwrite all live elements before reading.
    // Keep the zero-initialized public allocation path unchanged.
    BufferHandle allocate_buffer_uninitialized(size_t bytes, const std::string& tag = "") {
        return allocate_buffer_impl(bytes, tag, false);
    }

private:
    BufferHandle allocate_buffer_impl(size_t bytes, const std::string& tag, bool zero) {
        // Cap local optionnel (sans toucher à la limite MemoryGuard)
        if (max_ram_bytes_ > 0 && (memory_guard_.getCurrentBytes() + bytes) > max_ram_bytes_) {
            throw std::runtime_error(
                "RuntimeAllocator: Cannot allocate buffer '" + tag + "' (" +
                std::to_string(bytes / (1024*1024)) + " MB) - " +
                "would exceed local RAM cap. Current: " +
                std::to_string(memory_guard_.getCurrentBytes() / (1024*1024)) + " MB / " +
                std::to_string(max_ram_bytes_ / (1024*1024)) + " MB"
            );
        }
        if (!memory_guard_.requestAllocation(bytes, tag)) {
            throw std::runtime_error(
                "RuntimeAllocator: Cannot allocate buffer '" + tag + "' (" +
                std::to_string(bytes / (1024*1024)) + " MB) - " +
                "would exceed RAM limit. Current: " +
                std::to_string(memory_guard_.getCurrentBytes() / (1024*1024)) + " MB / " +
                std::to_string(memory_guard_.getLimit() / (1024*1024)) + " MB"
            );
        }
        
        try {
            // Byte-sized requests also need enough float storage for their tail.
            auto storage = std::unique_ptr<float[]>(
                new float[bytes / sizeof(float) + (bytes % sizeof(float) != 0)]);
            if (zero) std::memset(storage.get(), 0, bytes);
            BufferHandle result(storage.get(), bytes, tag, &memory_guard_);
            storage.release();
            total_allocated_bytes_ += bytes;
            num_allocations_++;
            return result;
        } catch (...) {
            // Covers both allocation and copying the handle's diagnostic tag.
            memory_guard_.releaseAllocation(bytes);
            throw;
        }
    }
    
public:
    // Pool de buffers réutilisables (scratchpad)
    // Pour éviter alloc/free répétitifs dans les boucles
    BufferHandle get_scratchpad(size_t min_bytes, const std::string& tag = "scratchpad") {
        // Chercher buffer existant de taille suffisante
        auto it = scratchpad_pool_.find(tag);
        if (it != scratchpad_pool_.end() && it->second.size_bytes() >= min_bytes) {
            // Réutiliser
            auto buffer = std::move(it->second);
            scratchpad_pool_.erase(it);
            return buffer;
        }
        
        // Allouer nouveau
        return allocate_buffer(min_bytes, tag);
    }
    
    BufferHandle get_scratchpad_uninitialized(size_t min_bytes, const std::string& tag = "scratchpad") {
        auto it = scratchpad_pool_.find(tag);
        if (it != scratchpad_pool_.end() && it->second.size_bytes() >= min_bytes) {
            auto buffer = std::move(it->second);
            scratchpad_pool_.erase(it);
            return buffer;
        }
        return allocate_buffer_uninitialized(min_bytes, tag);
    }

    // Retourner buffer au pool pour réutilisation
    void return_scratchpad(BufferHandle&& buffer) {
        std::string tag = buffer.tag();
        auto it = scratchpad_pool_.find(tag);
        if (it == scratchpad_pool_.end()) {
            scratchpad_pool_.emplace(std::move(tag), std::move(buffer));
            return;
        }

        // Conserver le plus gros buffer pour maximiser la réutilisation.
        if (it->second.size_bytes() >= buffer.size_bytes()) {
            // Drop the incoming buffer; its destructor releases memory accounting.
            return;
        }
        it->second = std::move(buffer);
    }
    
    // Stats (debug)
    size_t get_total_allocated() const { return total_allocated_bytes_; }
    size_t get_num_allocations() const { return num_allocations_; }
    size_t get_peak_usage() const { return memory_guard_.getPeakBytes(); }
    size_t get_current_usage() const { return memory_guard_.getCurrentBytes(); }
    size_t get_scratchpad_pool_count() const { return scratchpad_pool_.size(); }
    size_t get_scratchpad_pool_bytes() const {
        size_t total = 0;
        for (const auto& kv : scratchpad_pool_) {
            total += kv.second.size_bytes();
        }
        return total;
    }
    
    // Vérification post-forward
    bool check_no_leaks() const {
        size_t current = memory_guard_.getCurrentBytes() / (1024 * 1024);
        // Après forward, current devrait être proche de 0 (ou weights only)
        // Tolérance: 10 MB pour les poids/états persistants
        return current < 10;
    }
    
    void reset_stats() {
        total_allocated_bytes_ = 0;
        num_allocations_ = 0;
    }

    std::string build_stats_line(const std::string& stage = "runtime",
                                 const BackendMemoryAttribution* backend_mem = nullptr) const {
        std::ostringstream oss;
        const auto dyn = DynamicTensorAllocator::instance().getStatsSnapshot();

        const size_t total_bytes = total_allocated_bytes_;
        const size_t current_bytes = get_current_usage();
        const size_t peak_bytes = get_peak_usage();
        const double total_mb = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
        const double current_mb = static_cast<double>(current_bytes) / (1024.0 * 1024.0);
        const double peak_mb = static_cast<double>(peak_bytes) / (1024.0 * 1024.0);
        const size_t pool_count = get_scratchpad_pool_count();
        const size_t pool_bytes = get_scratchpad_pool_bytes();
        const double pool_mb = static_cast<double>(pool_bytes) / (1024.0 * 1024.0);
        const double dyn_loaded_mb = static_cast<double>(dyn.loaded_bytes) / (1024.0 * 1024.0);
        const double dyn_reserved_mb = static_cast<double>(dyn.reserved_bytes) / (1024.0 * 1024.0);
        const size_t guard_allocs = memory_guard_.getAllocationsCount();
        const size_t guard_deallocs = memory_guard_.getDeallocationsCount();
        const size_t backend_cpu_bytes = backend_mem ? backend_mem->cpu_bytes : 0;
        const size_t backend_vulkan_bytes = backend_mem ? backend_mem->vulkan_bytes : 0;
        const size_t backend_cuda_bytes = backend_mem ? backend_mem->cuda_bytes : 0;
        const size_t backend_rocm_bytes = backend_mem ? backend_mem->rocm_bytes : 0;
        const size_t backend_other_bytes = backend_mem ? backend_mem->other_bytes : 0;
        const double backend_cpu_mb = static_cast<double>(backend_cpu_bytes) / (1024.0 * 1024.0);
        const double backend_vulkan_mb = static_cast<double>(backend_vulkan_bytes) / (1024.0 * 1024.0);
        const double backend_cuda_mb = static_cast<double>(backend_cuda_bytes) / (1024.0 * 1024.0);
        const double backend_rocm_mb = static_cast<double>(backend_rocm_bytes) / (1024.0 * 1024.0);
        const double backend_other_mb = static_cast<double>(backend_other_bytes) / (1024.0 * 1024.0);

        oss << std::fixed << std::setprecision(2);

        oss << "[allocator] stage=" << stage
            << " allocations=" << num_allocations_
            << " total_allocated_mb=" << total_mb
            << " total_allocated_bytes=" << total_bytes
            << " current_mb=" << current_mb
            << " current_bytes=" << current_bytes
            << " peak_mb=" << peak_mb
            << " peak_bytes=" << peak_bytes
            << " scratchpad_pool_count=" << pool_count
            << " scratchpad_pool_mb=" << pool_mb
            << " scratchpad_pool_bytes=" << pool_bytes
            << " guard_allocs=" << guard_allocs
            << " guard_deallocs=" << guard_deallocs
            << " dyn_tensors=" << dyn.tensor_count
            << " dyn_loaded=" << dyn.loaded_count
            << " dyn_compressed=" << dyn.compressed_count
            << " dyn_reserved=" << dyn.reserved_count
            << " dyn_loaded_mb=" << dyn_loaded_mb
            << " dyn_loaded_bytes=" << dyn.loaded_bytes
            << " dyn_reserved_mb=" << dyn_reserved_mb
            << " dyn_reserved_bytes=" << dyn.reserved_bytes
            << " backend_cpu_mb=" << backend_cpu_mb
            << " backend_cpu_bytes=" << backend_cpu_bytes
            << " backend_vulkan_mb=" << backend_vulkan_mb
            << " backend_vulkan_bytes=" << backend_vulkan_bytes
            << " backend_cuda_mb=" << backend_cuda_mb
            << " backend_cuda_bytes=" << backend_cuda_bytes
            << " backend_rocm_mb=" << backend_rocm_mb
            << " backend_rocm_bytes=" << backend_rocm_bytes
            << " backend_other_mb=" << backend_other_mb
            << " backend_other_bytes=" << backend_other_bytes
            << " leaks=" << (check_no_leaks() ? 0 : 1);
        return oss.str();
    }

    void log_stats(const std::string& stage = "runtime",
                   bool verbose = false,
                   const BackendMemoryAttribution* backend_mem = nullptr) const {
        std::cerr << build_stats_line(stage, backend_mem) << std::endl;
        if (!verbose) return;

        for (const auto& kv : scratchpad_pool_) {
            std::cerr << "[allocator] scratchpad tag='" << kv.first
                      << "' size_mb=" << (kv.second.size_bytes() / (1024ULL * 1024ULL))
                      << std::endl;
        }
    }
    
    // Vidage du pool (fin de batch)
    void clear_scratchpad_pool() {
        scratchpad_pool_.clear();
    }
    
private:
    MemoryGuard& memory_guard_;
    size_t max_ram_bytes_;
    
    // Stats
    size_t total_allocated_bytes_ = 0;
    size_t num_allocations_ = 0;
    
    // Pool de buffers réutilisables
    std::unordered_map<std::string, BufferHandle> scratchpad_pool_;
};

// ============================================================================
// Helper pour créer des Tensors depuis std::vector existants
// ============================================================================

inline TensorHandle wrap_tensor(std::vector<float>&& data,
                                 const std::vector<int>& shape,
                                 const std::string& name,
                                 MemoryGuard& guard) {
    TensorDescriptor desc(shape, "float32", name);
    
    // data déjà alloué, mais comptabiliser quand même
    if (!guard.requestAllocation(desc.size_bytes)) {
        throw std::runtime_error("wrap_tensor: would exceed memory limit");
    }
    
    return TensorHandle(std::move(data), desc, &guard);
}

// ============================================================================
// Helper macro pour mode strict
// ============================================================================

#ifndef MIMIR_STRICT_MODE
#define MIMIR_STRICT_MODE 1  // Par défaut: mode strict
#endif

#define RUNTIME_ERROR_STRICT(msg) \
    do { \
        if (MIMIR_STRICT_MODE) { \
            throw std::runtime_error(msg); \
        } else { \
            std::cerr << "⚠️  [PERMISSIVE MODE] " << msg << std::endl; \
        } \
    } while(0)

#define RUNTIME_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            RUNTIME_ERROR_STRICT(msg); \
        } \
    } while(0)
