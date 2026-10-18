#ifndef __MEMORY_GUARD_HPP__
#define __MEMORY_GUARD_HPP__

#include <cstddef>
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <iomanip>

/**
 * MemoryGuard - Garde-fou strict pour limiter l'allocation RAM
 * Contrôle TOUTES les allocations de tenseurs et refuse si dépassement
 */
class MemoryGuard {
public:
    static MemoryGuard& instance() {
        static MemoryGuard guard;
        return guard;
    }
    
    // Configuration
    void setLimit(size_t bytes) {
        max_bytes_ = bytes;
        std::cerr << "🛡️  MemoryGuard: Limite stricte définie à " 
                  << (bytes / 1024 / 1024 / 1024) << " GB" << std::endl;
    }
    
    size_t getLimit() const { return max_bytes_; }
    
    // =============================
    // Système de Blocage d'Allocation
    // =============================
    
    // Bloquer toutes les nouvelles allocations
    void blockAllocations(bool block = true) {
        allocations_blocked_ = block;
        if (block) {
            std::cerr << "🔒 MemoryGuard: Allocations BLOQUÉES" << std::endl;
            std::cerr << "   Aucune nouvelle allocation ne sera autorisée" << std::endl;
        } else {
            std::cerr << "🔓 MemoryGuard: Allocations DÉBLOQUÉES" << std::endl;
        }
    }
    
    // Vérifier si les allocations sont bloquées
    bool isBlocked() const { return allocations_blocked_.load(); }
    
    // Bloquer temporairement avec timeout automatique
    void blockTemporary(size_t milliseconds) {
        blockAllocations(true);
        
        // Créer un thread pour débloquer après timeout
        std::thread([this, milliseconds]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
            blockAllocations(false);
        }).detach();
    }
    
    // Mode freeze: bloquer nouvelles allocations mais autoriser les libérations
    void freezeAllocations(bool freeze = true) {
        freeze_mode_ = freeze;
        if (freeze) {
            std::cerr << "❄️  MemoryGuard: Mode FREEZE activé" << std::endl;
            std::cerr << "   Nouvelles allocations bloquées, libérations autorisées" << std::endl;
        } else {
            std::cerr << "☀️  MemoryGuard: Mode FREEZE désactivé" << std::endl;
        }
    }
    
    bool isFrozen() const { return freeze_mode_.load(); }
    
    // Vérifier et enregistrer une allocation
    bool requestAllocation(size_t bytes, const std::string& tag = "") {
        // Vérifier si les allocations sont bloquées
        if (allocations_blocked_.load()) {
            blocked_attempts_++;
            std::cerr << "🔒 MemoryGuard: ALLOCATION BLOQUÉE (blocage actif)" << std::endl;
            std::cerr << "   Tentative #" << blocked_attempts_ << std::endl;
            std::cerr << "   Demandé: " << (bytes / 1024 / 1024) << " MB" << std::endl;
            if (!tag.empty()) {
                std::cerr << "   Tag: " << tag << std::endl;
            }
            return false;
        }
        
        // Vérifier si en mode freeze
        if (freeze_mode_.load()) {
            frozen_attempts_++;
            std::cerr << "❄️  MemoryGuard: ALLOCATION GELÉE (mode freeze)" << std::endl;
            std::cerr << "   Tentative #" << frozen_attempts_ << std::endl;
            std::cerr << "   Demandé: " << (bytes / 1024 / 1024) << " MB" << std::endl;
            if (!tag.empty()) {
                std::cerr << "   Tag: " << tag << std::endl;
            }
            return false;
        }
        
        size_t current = current_bytes_.load();
        size_t new_total = current + bytes;
        
        if (new_total > max_bytes_) {
            std::cerr << "❌ MemoryGuard: ALLOCATION REFUSÉE!" << std::endl;
            std::cerr << "   Demandé: " << (bytes / 1024 / 1024) << " MB" << std::endl;
            std::cerr << "   Actuel: " << (current / 1024 / 1024) << " MB" << std::endl;
            std::cerr << "   Nouveau total: " << (new_total / 1024 / 1024) << " MB" << std::endl;
            std::cerr << "   Limite: " << (max_bytes_ / 1024 / 1024) << " MB" << std::endl;
            if (!tag.empty()) {
                std::cerr << "   Tag: " << tag << std::endl;
            }
            return false;
        }
        
        current_bytes_ += bytes;
        if (current_bytes_ > peak_bytes_) {
            peak_bytes_ = current_bytes_.load();
        }
        
        allocations_count_++;
        
        if (!tag.empty() && bytes > 100 * 1024 * 1024) { // Log si > 100 MB
            std::cerr << "📊 Allocation: " << (bytes / 1024 / 1024) << " MB"
                      << " (" << tag << ")"
                      << " - Total: " << (current_bytes_ / 1024 / 1024) << " MB"
                      << " / " << (max_bytes_ / 1024 / 1024) << " MB" << std::endl;
        }
        
        return true;
    }
    
    // Libérer une allocation
    void releaseAllocation(size_t bytes) {
        if (current_bytes_ >= bytes) {
            current_bytes_ -= bytes;
        } else {
            current_bytes_ = 0;
        }
        deallocations_count_++;
    }
    
    // Statistiques
    size_t getCurrentBytes() const { return current_bytes_.load(); }
    size_t getPeakBytes() const { return peak_bytes_.load(); }
    size_t getAllocationsCount() const { return allocations_count_.load(); }
    size_t getDeallocationsCount() const { return deallocations_count_.load(); }
    size_t getBlockedAttempts() const { return blocked_attempts_.load(); }
    size_t getFrozenAttempts() const { return frozen_attempts_.load(); }
    float getUsagePercent() const {
        size_t max = max_bytes_.load();
        if (max == 0) return 0.0f;  // Éviter division par zéro
        return 100.0f * static_cast<float>(current_bytes_.load()) / static_cast<float>(max);
    }
    
    void printStats() const {
        std::cerr << "\n╔═══════════════════════════════════════════════════════╗" << std::endl;
        std::cerr << "║           MEMORY GUARD - STATISTIQUES                ║" << std::endl;
        std::cerr << "╠═══════════════════════════════════════════════════════╣" << std::endl;
        std::cerr << "║ Limite:     " << std::setw(10) << (max_bytes_ / 1024 / 1024) 
                  << " MB                         ║" << std::endl;
        std::cerr << "║ Actuel:     " << std::setw(10) << (current_bytes_ / 1024 / 1024) 
                  << " MB                         ║" << std::endl;
        std::cerr << "║ Pic:        " << std::setw(10) << (peak_bytes_ / 1024 / 1024) 
                  << " MB                         ║" << std::endl;
        std::cerr << "║ Utilisation:" << std::setw(9) << std::fixed << std::setprecision(1)
                  << getUsagePercent() << " %                          ║" << std::endl;
        std::cerr << "║ Allocations:" << std::setw(9) << allocations_count_ 
                  << "                              ║" << std::endl;
        std::cerr << "║ Libérations:" << std::setw(9) << deallocations_count_ 
                  << "                              ║" << std::endl;
        std::cerr << "╠═══════════════════════════════════════════════════════╣" << std::endl;
        std::cerr << "║ État:       " << (allocations_blocked_ ? "🔒 BLOQUÉ  " : "🔓 ACTIF   ")
                  << "                          ║" << std::endl;
        if (freeze_mode_) {
            std::cerr << "║ Mode:       ❄️  FREEZE                               ║" << std::endl;
        }
        if (blocked_attempts_ > 0) {
            std::cerr << "║ Tentatives bloquées: " << std::setw(9) << blocked_attempts_ 
                      << "                       ║" << std::endl;
        }
        if (frozen_attempts_ > 0) {
            std::cerr << "║ Tentatives gelées:   " << std::setw(9) << frozen_attempts_ 
                      << "                       ║" << std::endl;
        }
        std::cerr << "╚═══════════════════════════════════════════════════════╝" << std::endl;
    }
    
    void reset() {
        current_bytes_ = 0;
        peak_bytes_ = 0;
        allocations_count_ = 0;
        deallocations_count_ = 0;
    }
    
private:
    MemoryGuard() = default;
    ~MemoryGuard() = default;
    
    std::atomic<size_t> current_bytes_{0};
    std::atomic<size_t> peak_bytes_{0};
    std::atomic<size_t> max_bytes_{10ULL * 1024 * 1024 * 1024}; // 10 GB par défaut
    std::atomic<size_t> allocations_count_{0};
    std::atomic<size_t> deallocations_count_{0};
    
    // Système de blocage
    std::atomic<bool> allocations_blocked_{false};
    std::atomic<bool> freeze_mode_{false};
    std::atomic<size_t> blocked_attempts_{0};
    std::atomic<size_t> frozen_attempts_{0};
};

#endif // __MEMORY_GUARD_HPP__
