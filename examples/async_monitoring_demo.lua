#!/usr/bin/env lua5.3
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

--[[
    Exemple d'utilisation du monitoring asynchrone - Mímir Framework v2.0
    
    Démontre:
    - Monitoring htop non-bloquant
    - Mise à jour thread-safe des métriques
    - Pas besoin de render() explicite
]]

package.path = package.path .. ";./?.lua"

-- Simulation: Charger l'API Mímir (dans un vrai script, bin/mimir charge automatiquement)
-- Pour l'autocomplétion IDE, le fichier mimir-api.lua fournit les définitions

print("=== Mímir Framework v2.0 - Exemple Threading Asynchrone ===\n")

-- Configuration mémoire avec compression LZ4
print("📊 Configuration de l'allocateur mémoire...")
Mimir.Allocator.configure({
    max_ram_gb = 10.0,
    enable_compression = true,
    compression_threshold_mb = 100,
    enable_statistics = true
})

-- Démarrer le monitoring asynchrone
print("🚀 Démarrage du monitoring asynchrone (htop)...")
Mimir.Htop.create()  -- Lance un thread séparé, non-bloquant

print("📦 Création du modèle...")
Mimir.Model.create("encoder", {
    vocab_size = 50000,
    embed_dim = 512,
    num_layers = 6,
    num_heads = 8,
    d_ff = 2048
})

Mimir.Model.build()
Mimir.Model.allocate_params()
Mimir.Model.init_weights("he", 42)

-- Vérifier l'accélération GPU
if Mimir.Model.has_vulkan_compute() then
    print("✓ Accélération GPU activée (Vulkan Compute)")
else
    print("⚠ Mode CPU uniquement (Vulkan non disponible)")
end

print("\n🏋️  Démarrage de l'entraînement...\n")

-- Simulation d'un training loop
local total_epochs = 5
local batches_per_epoch = 100
local base_lr = 0.0002

for epoch = 1, total_epochs do
    local epoch_loss_sum = 0.0
    
    for batch = 1, batches_per_epoch do
        -- Simuler un batch de training
        local batch_start = os.clock()
        
        -- Simuler forward/backward (varie avec la taille du batch)
        local compute_time = 0.15 + math.random() * 0.05  -- 150-200ms
        
        -- Simuler du travail CPU/GPU
        local dummy_sum = 0
        for i = 1, 1000000 do
            dummy_sum = dummy_sum + i * 0.0001
        end
        
        local batch_time = os.clock() - batch_start
        
        -- Métriques simulées
        local loss = 2.5 * math.exp(-epoch * 0.3) + math.random() * 0.1
        epoch_loss_sum = epoch_loss_sum + loss
        local avg_loss = epoch_loss_sum / batch
        local lr = base_lr * math.cos(math.pi * (epoch - 1) / (2 * total_epochs))
        
        -- Statistiques mémoire simulées
        local memory_mb = 3500 + math.random() * 500
        local memory_freed = math.random() * 100
        local bps = 1.0 / batch_time
        local params = Mimir.Model.total_params()
        
        -- Métriques avancées simulées
        local kl = 0.05 + math.random() * 0.02
        local wass = 0.1 + math.random() * 0.05
        local ent = 0.03 + math.random() * 0.01
        local mse = loss * 0.8
        
        -- ⚡ MISE À JOUR ASYNCHRONE (thread-safe, non-bloquant)
        -- Le rendu se fait automatiquement dans le thread séparé toutes les 100ms
        Mimir.Htop.update(
            epoch, total_epochs,
            batch, batches_per_epoch,
            loss, avg_loss, lr,
            math.floor(batch_time * 1000),  -- batch_time_ms
            math.floor(memory_mb),
            math.floor(memory_freed),
            bps, params, 0.0,  -- timestep
            kl, wass, ent, 0.0, 0.0, 0.0, mse  -- mom, spat, temp
        )
        
        -- ✅ PAS BESOIN de Mimir.Htop.render()!
        -- Le rendu est automatique et n'impacte pas les performances du training
        
        -- Petit délai pour simuler le training (sans ce délai, la boucle serait trop rapide)
        os.execute("sleep 0.05")
    end
    
    print(string.format("Epoch %d/%d - Loss: %.4f", epoch, total_epochs, epoch_loss_sum / batches_per_epoch))
end

print("\n✓ Entraînement terminé!")

-- Statistiques finales
print("\n📊 Statistiques mémoire:")
Mimir.Allocator.print_stats()

local stats = Mimir.Allocator.get_stats()
print(string.format([[
  - Tensors alloués: %d
  - Tensors compressés: %d
  - Mémoire utilisée: %.2f GB / %.2f GB (%.1f%%)
  - Ratio compression: %.2fx
  - Évictions LRU: %d
]], 
    stats.allocated_tensors,
    stats.compressed_tensors,
    stats.used_bytes / (1024^3),
    stats.total_bytes / (1024^3),
    stats.used_percent,
    stats.compression_ratio,
    stats.evictions
))

print("\n💡 Note: Le monitoring htop continue de s'exécuter jusqu'à la fin du script")
print("   (thread asynchrone automatiquement joiné à la sortie)")
