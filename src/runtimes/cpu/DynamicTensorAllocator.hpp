#ifndef __DYNAMIC_TENSOR_ALLOCATOR_HPP__
#define __DYNAMIC_TENSOR_ALLOCATOR_HPP__

#include "AdvancedRAMManager.hpp"
#include "MemoryGuard.hpp"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <iomanip>

/**
 * DynamicTensorAllocator - Allocation RAM dynamique pour tenseurs
 * 
 * Combine AdvancedRAMManager (compression, éviction) + MemoryGuard (limite stricte)
 * pour gérer intelligemment la mémoire des tenseurs avec:
 * - Allocation à la demande (lazy loading)
 * - Compression automatique des tenseurs inactifs
 * - Éviction LRU si limite atteinte
 * - Rechargement transparent depuis disque/mémoire compressée
 */
class DynamicTensorAllocator {
public:
    static DynamicTensorAllocator& instance() {
        static DynamicTensorAllocator alloc;
        return alloc;
    }
    
    struct TensorHandle {
        size_t size = 0;
        bool is_loaded = false;
        bool is_compressed = false;
        std::string cache_key;
        float* data_ptr = nullptr;  // Pointeur vers données actives
        bool reserved = false;      // Réservation comptabilisée dans MemoryGuard
        size_t reserved_bytes = 0;  // Bytes réservés (typiquement size*sizeof(float))
    };

    struct StatsSnapshot {
        size_t tensor_count = 0;
        size_t loaded_count = 0;
        size_t compressed_count = 0;
        size_t reserved_count = 0;
        size_t loaded_bytes = 0;
        size_t reserved_bytes = 0;
    };
    
    // Configuration
    void configure(double max_ram_gb, bool enable_compression = true, bool lazy_mode = true) {
        if (max_ram_gb < 0.0) max_ram_gb = 0.0;
        max_ram_bytes_ = static_cast<size_t>(max_ram_gb * 1024.0 * 1024.0 * 1024.0);
        compression_enabled_ = enable_compression;
        lazy_mode_ = lazy_mode;
        
        // Configurer MemoryGuard
        auto& guard = MemoryGuard::instance();
        guard.setLimit(max_ram_bytes_);
        
        // Configurer AdvancedRAMManager
        AdvancedRAMManager::Config ram_config;
        ram_config.max_ram_bytes = max_ram_bytes_;
        ram_config.enable_compression = enable_compression;
        ram_config.enable_async_loading = false;  // Synchrone pour contrôle strict
        ram_config.enable_prediction = false;
        ram_config.enable_statistics = true;
        ram_config.enable_disk_spill = true;
        ram_config.spill_dir = ".mimir-spill";
        ram_config.worker_threads = 2;
        
        auto& ram_mgr = AdvancedRAMManager::instance();
        ram_mgr.configure(ram_config);
        
        std::cerr << "🚀 DynamicTensorAllocator configuré:" << std::endl;
        std::cerr << "   - Limite RAM: " << std::fixed << std::setprecision(2) << max_ram_gb << " GB" << std::endl;
        std::cerr << "   - Compression: " << (enable_compression ? "activée" : "désactivée") << std::endl;
        std::cerr << "   - Lazy mode: " << (lazy_mode_ ? "activé" : "désactivé") << std::endl;
    }
    
    // Allouer un tenseur (retourne un handle)
    TensorHandle* allocateTensor(size_t num_elements, const std::string& tag = "") {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t bytes_needed = num_elements * sizeof(float);
        
        // Créer le handle
        auto handle = std::make_unique<TensorHandle>();
        handle->size = num_elements;
        handle->is_loaded = false;
        handle->is_compressed = false;
        handle->cache_key = tag + "_" + std::to_string(next_id_++);
        handle->data_ptr = nullptr;
        handle->reserved = false;
        handle->reserved_bytes = 0;

        // Mode non-lazy: réserver tout de suite (comptabilisation immédiate)
        if (!lazy_mode_) {
            auto& guard = MemoryGuard::instance();

            // Éviction proactive dès 90% de la limite (évite le refus dur).
            ensureBelowPressureLocked(0.90f, bytes_needed);

            if (!guard.requestAllocation(bytes_needed, tag)) {
                std::cerr << "⚠️  Mémoire insuffisante, tentative d'éviction..." << std::endl;
                evictLRULocked(bytes_needed);
                if (!guard.requestAllocation(bytes_needed, tag)) {
                    std::cerr << "❌ Impossible d'allouer tenseur même après éviction!" << std::endl;
                    return nullptr;
                }
            }
            handle->reserved = true;
            handle->reserved_bytes = bytes_needed;
        }
        
        // Sauvegarder la clé avant le move
        std::string cache_key = handle->cache_key;
        handles_[cache_key] = std::move(handle);
        return handles_[cache_key].get();
    }
    
    // Obtenir les données d'un tenseur (chargement à la demande)
    float* getTensorData(TensorHandle* handle) {
        if (!handle) return nullptr;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Si déjà chargé, retourner
        if (handle->is_loaded && handle->data_ptr) {
            access_order_[handle->cache_key] = ++access_counter_;
            return handle->data_ptr;
        }
        
        // Charger depuis AdvancedRAMManager ou allouer
        auto& ram_mgr = AdvancedRAMManager::instance();

        // Mode lazy: réserver au moment du chargement réel
        if (lazy_mode_ && !handle->reserved) {
            const size_t bytes_needed = handle->size * sizeof(float);
            auto& guard = MemoryGuard::instance();

            // Éviction proactive dès 90% de la limite (évite le refus dur).
            ensureBelowPressureLocked(0.90f, bytes_needed);

            if (!guard.requestAllocation(bytes_needed, handle->cache_key)) {
                std::cerr << "⚠️  Mémoire insuffisante, tentative d'éviction..." << std::endl;
                evictLRULocked(bytes_needed);
                if (!guard.requestAllocation(bytes_needed, handle->cache_key)) {
                    std::cerr << "❌ Impossible d'allouer tenseur même après éviction!" << std::endl;
                    return nullptr;
                }
            }
            handle->reserved = true;
            handle->reserved_bytes = bytes_needed;
        }
        
        if (handle->is_compressed) {
            // Décompresser depuis RAMManager
            auto data = ram_mgr.get(handle->cache_key);
            if (data.has_value()) {
                // ⚠️ IMPORTANT: malloc direct bypass MemoryGuard!
                // La comptabilisation est faite via handle->reserved/_bytes (lazy ou non-lazy)
                handle->data_ptr = reinterpret_cast<float*>(
                    malloc(handle->size * sizeof(float)));
                if (handle->data_ptr) {
                    memcpy(handle->data_ptr, data->data(), 
                           handle->size * sizeof(float));
                    handle->is_loaded = true;
                    handle->is_compressed = false;
                    access_order_[handle->cache_key] = ++access_counter_;
                    return handle->data_ptr;
                }
            }
        } else {
            // Allocation fraîche
            // ⚠️ IMPORTANT: malloc direct bypass MemoryGuard!
            // La comptabilisation est faite via handle->reserved/_bytes (lazy ou non-lazy)
            handle->data_ptr = reinterpret_cast<float*>(
                malloc(handle->size * sizeof(float)));
            if (handle->data_ptr) {
                // Initialiser à 0
                memset(handle->data_ptr, 0, handle->size * sizeof(float));
                handle->is_loaded = true;
                access_order_[handle->cache_key] = ++access_counter_;
                return handle->data_ptr;
            }
        }
        
        return nullptr;
    }
    
    // Compresser un tenseur (libère RAM active, stocke compressé)
    void compressTensor(TensorHandle* handle) {
        if (!handle) return;
        std::lock_guard<std::mutex> lock(mutex_);
        (void)compressTensorLocked(handle);
    }
    
    // Libérer complètement un tenseur
    void freeTensor(TensorHandle* handle) {
        if (!handle) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto& ram_mgr = AdvancedRAMManager::instance();
        auto& guard = MemoryGuard::instance();
        
        // Libérer données actives
        if (handle->data_ptr) {
            free(handle->data_ptr);
            handle->data_ptr = nullptr;
        }

        // Libérer la réservation (qu'il y ait eu chargement ou non)
        if (handle->reserved) {
            guard.releaseAllocation(handle->reserved_bytes);
            handle->reserved = false;
            handle->reserved_bytes = 0;
        }
        
        // Libérer depuis RAMManager
        if (handle->is_compressed) {
            ram_mgr.deallocate(handle->cache_key);
        }
        
        // Supprimer handle
        access_order_.erase(handle->cache_key);
        handles_.erase(handle->cache_key);
    }
    
    // Statistiques
    void printStats() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::cerr << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
        std::cerr << "║      DYNAMIC TENSOR ALLOCATOR - STATISTIQUES         ║" << std::endl;
        std::cerr << "╠═══════════════════════════════════════════════════════╣" << std::endl;
        
        size_t loaded_count = 0;
        size_t compressed_count = 0;
        size_t total_size = 0;
        
        for (const auto& [key, handle] : handles_) {
            if (handle->is_loaded) loaded_count++;
            if (handle->is_compressed) compressed_count++;
            total_size += handle->size * sizeof(float);
        }
        
        std::cerr << "║ Tenseurs totaux:  " << handles_.size() << std::endl;
        std::cerr << "║ Chargés:          " << loaded_count << std::endl;
        std::cerr << "║ Compressés:       " << compressed_count << std::endl;
        std::cerr << "║ Taille totale:    " << (total_size / 1024 / 1024) << " MB" << std::endl;
        std::cerr << "╚═══════════════════════════════════════════════════════╝" << std::endl;
        
        // Afficher stats MemoryGuard
        MemoryGuard::instance().printStats();
        
        // Afficher stats RAMManager
        AdvancedRAMManager::instance().printDetailedStats();
    }
    
    size_t getTensorCount() const { return handles_.size(); }
    size_t getLoadedCount() const {
        size_t count = 0;
        for (const auto& [k, h] : handles_) {
            if (h->is_loaded) count++;
        }
        return count;
    }

    StatsSnapshot getStatsSnapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        StatsSnapshot s;
        s.tensor_count = handles_.size();
        for (const auto& [k, h] : handles_) {
            if (h->is_loaded) {
                s.loaded_count++;
                s.loaded_bytes += h->size * sizeof(float);
            }
            if (h->is_compressed) {
                s.compressed_count++;
            }
            if (h->reserved) {
                s.reserved_count++;
                s.reserved_bytes += h->reserved_bytes;
            }
        }
        return s;
    }

private:
    DynamicTensorAllocator() = default;

    // Éviction proactive pour rester sous un ratio de la limite MemoryGuard.
    // Suppose mutex_ déjà tenu.
    void ensureBelowPressureLocked(float target_ratio, size_t upcoming_bytes) {
        auto& guard = MemoryGuard::instance();
        const size_t limit = guard.getLimit();
        if (limit == 0) return;

        const size_t target = static_cast<size_t>(static_cast<double>(limit) * static_cast<double>(target_ratio));
        const size_t current = guard.getCurrentBytes();
        if (current + upcoming_bytes <= target) return;

        const size_t need_free = (current + upcoming_bytes) - target;
        evictLRULocked(need_free);
    }

    // Compresser/spiller un tenseur (sans reprendre mutex_)
    // Retourne true si on a réellement libéré la réservation MemoryGuard.
    bool compressTensorLocked(TensorHandle* handle) {
        if (!handle || !handle->is_loaded || !handle->data_ptr) return false;

        auto& ram_mgr = AdvancedRAMManager::instance();

        // Spill direct sur disque (bloquant) pour libérer réellement la RAM.
        // Évite d'allouer un gros buffer intermédiaire en RAM.
        const size_t bytes = handle->size * sizeof(float);
        if (!ram_mgr.storeRawOnDisk(handle->cache_key, handle->data_ptr, bytes)) {
            // Fallback: tenter cache RAMManager (avec ou sans compression), puis spill.
            std::vector<uint8_t> data(bytes);
            memcpy(data.data(), handle->data_ptr, bytes);
            const bool do_compress = compression_enabled_;
            if (!ram_mgr.allocate(handle->cache_key, data, do_compress)) {
                return false;
            }
            (void)ram_mgr.forceSpillToDisk(handle->cache_key);
        }

        // Libérer mémoire active
        free(handle->data_ptr);
        handle->data_ptr = nullptr;
        handle->is_loaded = false;
        handle->is_compressed = true;

        // Mettre à jour MemoryGuard
        if (handle->reserved) {
            auto& guard = MemoryGuard::instance();
            guard.releaseAllocation(handle->reserved_bytes);
            handle->reserved = false;
            handle->reserved_bytes = 0;
            return true;
        }
        return false;
    }

    // Éviction LRU (suppose mutex_ déjà tenu).
    void evictLRULocked(size_t bytes_needed) {
        // Trier par ordre d'accès (LRU)
        std::vector<std::pair<std::string, uint64_t>> items;
        items.reserve(access_order_.size());
        for (const auto& [key, timestamp] : access_order_) {
            items.push_back({key, timestamp});
        }

        std::sort(items.begin(), items.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });

        auto& guard = MemoryGuard::instance();
        const size_t before_bytes = guard.getCurrentBytes();

        size_t freed_estimated = 0;
        for (const auto& [key, _] : items) {
            if (freed_estimated >= bytes_needed) break;

            auto it = handles_.find(key);
            if (it != handles_.end() && it->second->is_loaded) {
                const bool freed = compressTensorLocked(it->second.get());
                if (freed) {
                    freed_estimated += it->second->size * sizeof(float);
                }
            }
        }

        const size_t after_bytes = guard.getCurrentBytes();
        const size_t freed_actual = (before_bytes >= after_bytes) ? (before_bytes - after_bytes) : 0;

        std::cerr << "⟳ Éviction LRU: ~" << (freed_estimated / 1024 / 1024)
                  << " MB candidats, " << (freed_actual / 1024 / 1024)
                  << " MB libérés (MemoryGuard)" << std::endl;
    }
    
    mutable std::mutex mutex_;
    size_t max_ram_bytes_ = 10ULL * 1024 * 1024 * 1024;
    bool compression_enabled_ = true;
    bool lazy_mode_ = true;
    size_t next_id_ = 0;
    uint64_t access_counter_ = 0;
    
    std::unordered_map<std::string, std::unique_ptr<TensorHandle>> handles_;
    std::unordered_map<std::string, uint64_t> access_order_;
};

#endif // __DYNAMIC_TENSOR_ALLOCATOR_HPP__
