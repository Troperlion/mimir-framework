#!/usr/bin/env lua
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

-- ══════════════════════════════════════════════════════════════
--  TEMPLATE SCRIPT - Mímir Framework v3.5.0
--  Utilisez ce template pour créer vos propres modèles
--  
--  📚 Synchronisé avec l'API enregistrée dans
--     src/scriptings/Lua/luaScripting/LuaScripting.cpp
-- ══════════════════════════════════════════════════════════════

---@type string Tag de run (sert aux chemins de sauvegarde, logs, etc.)
local RUN_TAG = "my_model"

---@type ModelType Type du modèle à construire (doit correspondre au registry C++)
-- Astuce: vous pouvez surcharger l'archi via env:
--   MIMIR_ARCH=vae_conv ./bin/mimir --lua scripts/templates/template_new_model.lua
-- Valeur par défaut: choisir une archi connue du repo.
local MODEL_TYPE = os.getenv("MIMIR_ARCH") or "vae_conv"

-- Récupérer une config par défaut depuis le registry C++
---@type table|nil, string?
local CONFIG, err_cfg = Mimir.Architectures.default_config(MODEL_TYPE)
if not CONFIG then
    log("⚠️ default_config() indisponible: " .. (err_cfg or "unknown"))
    -- Fallback: prendre une archi disponible (si l'API le permet)
    local ok_list, list_or_err = pcall(function()
        if type(Mimir.Architectures) == "table" and type(Mimir.Architectures.available) == "function" then
            return Mimir.Architectures.available()
        end
        return nil, "Architectures.available() indisponible"
    end)

    if ok_list and type(list_or_err) == "table" and #list_or_err > 0 then
        MODEL_TYPE = tostring(list_or_err[1])
        log("ℹ️  Fallback: utilisation de l'architecture disponible: " .. tostring(MODEL_TYPE))
        CONFIG, err_cfg = Mimir.Architectures.default_config(MODEL_TYPE)
    end

    if type(CONFIG) ~= "table" then
        CONFIG = {}
    end
end

local function apply_dtype(dtype)
    if dtype == nil then return true end
    local dtype_fn = nil
    if type(Mimir) == "table" then
        if type(Mimir.model) == "table" and type(Mimir.model.dtype) == "function" then
            dtype_fn = Mimir.model.dtype
        elseif type(Mimir.Model) == "table" and type(Mimir.Model.dtype) == "function" then
            dtype_fn = Mimir.Model.dtype
        end
    end
    if dtype_fn == nil then
        -- Runtime ancienne / non supportée: ignorer silencieusement.
        return true
    end
    local ok, dt_or_err = dtype_fn(dtype)
    if ok == false then
        return false, dt_or_err
    end
    return true
end

-- Ajustez ici selon votre dataset / contraintes mémoire
CONFIG.seq_len = CONFIG.seq_len or 64
-- Convention moderne: vocab_size (legacy: max_vocab)
CONFIG.vocab_size = CONFIG.vocab_size or CONFIG.max_vocab or 50000
CONFIG.max_vocab = CONFIG.max_vocab or CONFIG.vocab_size -- legacy (compat)
CONFIG.image_w = CONFIG.image_w or 256
CONFIG.image_h = CONFIG.image_h or 256
CONFIG.image_c = CONFIG.image_c or 4
-- Convention moderne: d_model (certaines archis legacy utilisent encore embed_dim)
CONFIG.d_model = CONFIG.d_model or CONFIG.embed_dim or 256
CONFIG.embed_dim = CONFIG.embed_dim or CONFIG.d_model -- legacy (compat)

-- Optionnel: dtype (préférence) via env/config.
-- Ex: MIMIR_DTYPE=float16 ./bin/mimir --lua scripts/templates/template_new_model.lua
CONFIG.dtype = CONFIG.dtype or os.getenv("MIMIR_DTYPE")

-- Training (consommés par vos scripts, pas forcément par le binding)
CONFIG.batch_size = CONFIG.batch_size or 8
CONFIG.learning_rate = CONFIG.learning_rate or 1e-4
CONFIG.epochs = CONFIG.epochs or 1

log("╔════════════════════════════════════════════════════════╗")
log("║      Template Script - Mímir Framework v3.5.0          ║")
log("╚════════════════════════════════════════════════════════╝")
log("Arch: " .. tostring(MODEL_TYPE))

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 1: CONFIGURATION SYSTÈME (OBLIGATOIRE!)
-- ══════════════════════════════════════════════════════════════

log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  1. Configuration Système")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- ⚠️ CRITIQUE: Toujours configurer l'allocateur et MemoryGuard en premier!
-- API moderne: MemoryGuard avec limite stricte
---@type boolean
local ok_guard = Mimir.MemoryGuard.setLimit(10)  -- 10 GB (valeur < 1000 = GB, sinon bytes)
if not ok_guard then
    log("❌ ERREUR: MemoryGuard.setLimit() a échoué")
    os.exit(1)
end
log("✓ MemoryGuard configuré (limite: 10 GB)")

-- Allocateur dynamique pour tensors
---@type boolean, string?
local ok, err = Mimir.Allocator.configure({
    max_ram_gb = 10.0,              -- Limite stricte (coordonnée avec MemoryGuard)
    enable_compression = true,       -- Compression LZ4 (~50% économie RAM)
    swap_strategy = "lru"
})

if not ok then
    log("❌ ERREUR: Configuration allocateur échouée: " .. (err or "unknown"))
    os.exit(1)
end
log("✓ Allocateur configuré (compression LZ4, offload enabled)")

-- Vérifier et activer l'accélération hardware
---@type HardwareCaps
local hw = Mimir.Model.hardware_caps()
log("\n🔧 Capacités Hardware:")
log(string.format("  • AVX2:  %s", hw.avx2 and "✓" or "✗"))
log(string.format("  • FMA:   %s", hw.fma and "✓" or "✗"))
log(string.format("  • F16C:  %s", hw.f16c and "✓" or "✗"))
log(string.format("  • BMI2:  %s", hw.bmi2 and "✓" or "✗"))

if hw.avx2 or hw.fma then
    local ok_hw, err_hw = Mimir.Model.set_hardware(true)
    if ok_hw then
        log("\n✓ Accélération hardware activée\n")
    else
        log("\n⚠️  Accélération hardware non activée: " .. (err_hw or "unknown"))
    end
end

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 2: CHOIX DE L'ARCHITECTURE
-- ══════════════════════════════════════════════════════════════

log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  2. Choix de l'Architecture")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- 💡 CONSEIL: Utilisez des valeurs raisonnables pour rester sous 10 GB
-- Exemples de configs qui rentrent dans 10 GB:
--   • Transformer: embed_dim=512, num_layers=6-8
--   • ResNet-50: ~25M params
--   • UNet: base_channels=64-128
--   • Diffusion: image_resolution=256

log("Configuration choisie:")
log(string.format("  • seq_len: %d", CONFIG.seq_len))
log(string.format("  • vocab_size/max_vocab: %d", CONFIG.vocab_size or CONFIG.max_vocab or 0))
log(string.format("  • d_model/embed_dim: %d", CONFIG.d_model))
log(string.format("  • image: %dx%dx%d", CONFIG.image_w, CONFIG.image_h, CONFIG.image_c))

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 3: CRÉATION DU MODÈLE
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  3. Création du Modèle")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

---@type boolean, string?
local ok_create, err_create = Mimir.Model.create(MODEL_TYPE, CONFIG)
if not ok_create then
    log("❌ ERREUR: Création du modèle échouée: " .. (err_create or "unknown"))
    os.exit(1)
end
log("✓ Modèle créé (type='" .. MODEL_TYPE .. "')")

-- Appliquer la préférence dtype (si supportée par la runtime)
do
    local ok_dt, dt_or_err = apply_dtype(CONFIG.dtype)
    if ok_dt == false then
        log("❌ ERREUR: dtype invalide: " .. tostring(dt_or_err))
        os.exit(1)
    end
    if CONFIG.dtype ~= nil then
        log("✓ dtype appliqué: " .. tostring(CONFIG.dtype))
    end
end

-- Construire l'architecture à partir de (MODEL_TYPE + CONFIG)
---@type boolean, integer|string?
local ok_build, params_built_or_err = Mimir.Model.build()
if not ok_build then
    log("❌ ERREUR: Construction du modèle échouée: " .. (params_built_or_err or "unknown"))
    os.exit(1)
end
log("✓ Architecture construite avec succès")
local params_built = tonumber(params_built_or_err)
if params_built then
    log(string.format("  Paramètres (scalars): %d (%.2fM)", params_built, params_built / 1e6))
end

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 4: ALLOCATION DES PARAMÈTRES (VÉRIFIER LE SUCCÈS!)
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  4. Allocation des Paramètres")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

---@type boolean, integer|string?
local ok_alloc, alloc_count_or_err = Mimir.Model.allocate_params()

-- ⚠️ TOUJOURS vérifier le succès!
if not ok_alloc then
    log("❌ ERREUR: Impossible d'allouer les paramètres!")
    log("⚠️  Raison: " .. (alloc_count_or_err or "La limite de 10 GB a été atteinte"))
    log("\n💡 Solutions possibles:")
    log("  1. Réduire d_model (ex: 256 → 128)")
    log("  2. Réduire seq_len (ex: 64 → 32)")
    log("  3. Réduire max_vocab (ex: 50000 → 20000)")
    log("  4. Choisir une architecture plus petite")
    Mimir.MemoryGuard.printStats()  -- Afficher les stats mémoire
    os.exit(1)
end

-- Récupérer le nombre total de paramètres
---@type integer
local param_count = Mimir.Model.total_params()

-- Calcul de la mémoire utilisée
local memory_mb = param_count * 4 / (1024 * 1024)
log(string.format("✓ Paramètres alloués: %d (%.2fM)", param_count, param_count / 1e6))
log(string.format("  Mémoire utilisée: %.2f MB", memory_mb))
log(string.format("  Compression LZ4: ~%.2f MB en RAM", memory_mb * 0.5))

-- Afficher les stats MemoryGuard
---@type MemoryGuardStats
local mem_stats = Mimir.MemoryGuard.getStats()
log(string.format("  MemoryGuard: %.2f MB / %.2f MB (%.1f%%)", 
    mem_stats.current_mb, mem_stats.limit_mb, mem_stats.usage_percent))

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 5: INITIALISATION DES POIDS
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  5. Initialisation des Poids")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Méthodes disponibles: "xavier" | "he" | "normal" | "uniform" | "zeros"
---@type WeightInit
local init_method = "xavier"  -- Valeur supportée par l'API
local seed = 42  -- Pour reproductibilité

---@type boolean, string?
local ok_init, err_init = Mimir.Model.init_weights(init_method, seed)

if not ok_init then
    log("❌ Échec de l'initialisation des poids: " .. (err_init or "unknown"))
    os.exit(1)
end

log(string.format("✓ Poids initialisés (méthode: %s, seed: %d)", init_method, seed))

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 6: TOKENIZER (SI MODÈLE NLP)
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  6. Tokenizer (optionnel pour NLP)")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

local wants_tokenizer = false
do
    local mt = tostring(MODEL_TYPE or "")
    if mt == "transformer" or mt:find("transformer", 1, true) ~= nil or mt:find("gpt", 1, true) ~= nil or mt:find("bert", 1, true) ~= nil then
        wants_tokenizer = true
    end
end

if wants_tokenizer and type(Mimir.Tokenizer) == "table" and type(Mimir.Tokenizer.create) == "function" then
    ---@type boolean
    local ok_tok = Mimir.Tokenizer.create(CONFIG.vocab_size or CONFIG.max_vocab)
    if not ok_tok then
        log("⚠️  Tokenizer.create() a échoué (skip tokenizer)")
    else
        -- Ajouter les tokens spéciaux (si exposés)
        if type(Mimir.Tokenizer.add_token) == "function" then
            Mimir.Tokenizer.add_token("[PAD]")
            Mimir.Tokenizer.add_token("[UNK]")
            Mimir.Tokenizer.add_token("[CLS]")
            Mimir.Tokenizer.add_token("[SEP]")
            Mimir.Tokenizer.add_token("[MASK]")  -- Pour masked language modeling
        end

        -- Configurer la longueur max des séquences
        if type(Mimir.Tokenizer.set_max_length) == "function" then
            Mimir.Tokenizer.set_max_length(CONFIG.seq_len)
        end

        log(string.format("✓ Tokenizer créé (vocab_size: %d)", CONFIG.vocab_size or CONFIG.max_vocab or 0))
        if type(Mimir.Tokenizer.pad_id) == "function" and type(Mimir.Tokenizer.unk_id) == "function" and type(Mimir.Tokenizer.seq_id) == "function" then
            log(string.format("  Tokens spéciaux: PAD=%d, UNK=%d, CLS/SEQ=%d",
                Mimir.Tokenizer.pad_id(), Mimir.Tokenizer.unk_id(), Mimir.Tokenizer.seq_id()))
        end
    end
else
    log("ℹ️  Tokenizer non activé pour cette arch (skip)")
end

-- Exemple d'utilisation (décommenter si besoin)
-- local test_text = "Hello world, this is a test."
-- local tokens = Mimir.Tokenizer.tokenize(test_text)
-- log("Test tokenization: " .. #tokens .. " tokens")
-- local decoded = Mimir.Tokenizer.detokenize(tokens)
-- log("Decoded: " .. decoded)

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 7: DATASET (SI TRAINING)
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  7. Dataset (optionnel pour training)")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Charger un dataset (décommenter si besoin)
-- local dataset_path = "data/my_dataset"
-- local ok_ds, err_ds = Mimir.Dataset.load(dataset_path)
-- if not ok_ds then
--     log("❌ Échec du chargement du dataset: " .. (err_ds or "unknown"))
--     os.exit(1)
-- end
--
-- -- Préparer les séquences pour le training
-- local ok_prep, err_prep = Mimir.Dataset.prepare_sequences(CONFIG.seq_len)
-- if not ok_prep then
--     log("❌ Échec de la préparation des séquences: " .. (err_prep or "unknown"))
--     os.exit(1)
-- end
--
-- -- Exemple d'accès aux items
-- ---@type DatasetItem|nil, string?
-- local item, err_item = Mimir.Dataset.get(1)  -- Premier item (indices 1-based)
-- if item then
--     log("  Premier item:")
--     if item.text_file then log("    text_file: " .. item.text_file) end
--     if item.image_file then log("    image_file: " .. item.image_file) end
--     if item.width then log(string.format("    dimensions: %dx%d", item.width, item.height)) end
-- end
--
-- log("✓ Dataset chargé et préparé")

log("⚠️  Dataset non configuré (décommentez le code ci-dessus si besoin)")

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 8: TRAINING / INFERENCE
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  8. Training / Inference")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Exemple de boucle de training manuelle
--[[
local function compute_loss(predictions, targets)
    -- Implémenter votre fonction de loss (MSE, CrossEntropy, etc.)
    local loss = 0.0
    for i = 1, #predictions do
        local diff = predictions[i] - targets[i]
        loss = loss + diff * diff
    end
    return loss / #predictions
end

local function training_loop()
    log("\n🚀 Démarrage du training...")
    
    for epoch = 1, CONFIG.epochs do
        local total_loss = 0.0
        local num_batches = 100  -- À adapter selon votre dataset
        
        for batch = 1, num_batches do
            -- 1. Préparer les données d'entrée (exemple)
            ---@type float[]
            local input_data = {}  -- Charger depuis dataset
            for i = 1, 512 do input_data[i] = math.random() end
            
            -- 2. Zero gradients
            Mimir.Model.zero_grads()
            
            -- 3. Forward pass (training mode)
            ---@type float[]|nil, string?
            local predictions, err_fwd = Mimir.Model.forward(input_data, true)
            if not predictions then
                log("❌ Forward pass échoué: " .. (err_fwd or "unknown"))
                break
            end
            
            -- 4. Calculer la loss
            local targets = {}  -- Vos targets
            for i = 1, #predictions do targets[i] = math.random() end
            local loss = compute_loss(predictions, targets)
            total_loss = total_loss + loss
            
            -- 5. Calculer les gradients de loss
            local loss_gradient = {}
            for i = 1, #predictions do
                loss_gradient[i] = 2.0 * (predictions[i] - targets[i]) / #predictions
            end
            
            -- 6. Backward pass
            local ok_bwd, err_bwd = Mimir.Model.backward(loss_gradient)
            if not ok_bwd then
                log("❌ Backward pass échoué: " .. (err_bwd or "unknown"))
                break
            end
            
            -- 7. Optimizer step
            local ok_opt, err_opt = Mimir.Model.optimizer_step(CONFIG.learning_rate)
            if not ok_opt then
                log("❌ Optimizer step échoué: " .. (err_opt or "unknown"))
                break
            end
        end
        
        local avg_loss = total_loss / num_batches
        log(string.format("epoch=%d | loss=%.6f", epoch, avg_loss))
        
        -- Afficher les stats mémoire périodiquement
        if epoch % 10 == 0 then
            local stats = Mimir.MemoryGuard.getStats()
            log(string.format("  RAM: %.2f MB / %.2f MB (%.1f%%)",
                stats.current_mb, stats.limit_mb, stats.usage_percent))
        end
    end
    
    log("\n✅ Training terminé!")
end

-- Décommenter pour lancer le training
-- training_loop()
]]

log("💡 Template de boucle de training prêt (décommentez le code ci-dessus)")
log("\nAPI disponible:")
log("  • Mimir.Model.forward(input, training?) → predictions")
log("  • Mimir.Model.backward(loss_gradient) → ok")
log("  • Mimir.Model.zero_grads() → ok")
log("  • Mimir.Model.optimizer_step(lr) → ok")
log("  • Mimir.Model.get_gradients() → gradients")
log("  • Mimir.Serialization.save(path, format, options) → ok")
log("  • Mimir.Serialization.load(path, format, options) → ok")

-- ══════════════════════════════════════════════════════════════
--  ÉTAPE 9: SAUVEGARDE (NOUVELLE API v2.4.0)
-- ══════════════════════════════════════════════════════════════

log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
log("  9. Sauvegarde (Serialization API v2.4)")
log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Sauvegarder le modèle avec la nouvelle API (décommenter si besoin)
-- local checkpoint_path = "checkpoints/" .. RUN_TAG .. "_epoch_" .. CONFIG.epochs .. ".safetensors"
-- 
-- -- Format SafeTensors (production, compatible HuggingFace)
-- local ok_save, err_save = Mimir.Serialization.save(checkpoint_path, "safetensors", {
--     save_tokenizer = true,
--     save_encoder = true,
--     save_optimizer = false
-- })
-- 
-- if not ok_save then
--     log("❌ Échec de la sauvegarde: " .. (err_save or "unknown"))
-- else
--     log("✓ Modèle sauvegardé: " .. checkpoint_path)
--     log("  Format: SafeTensors (compatible PyTorch/HuggingFace)")
-- end
--
-- -- Alternative: RawFolder (debug avec checksums SHA256)
-- -- local checkpoint_dir = "checkpoints/" .. RUN_TAG .. "_debug/"
-- -- Mimir.Serialization.save(checkpoint_dir, "raw_folder")
--
-- -- Alternative: DebugJson (inspection avec statistiques)
-- -- Mimir.Serialization.save("debug/" .. RUN_TAG .. ".json", "debug_json")

log("⚠️  Sauvegarde non configurée (décommentez le code ci-dessus si besoin)")
log("📚 Voir docs/03-API-Reference/02-Serialization.md et docs/03-API-Reference/16-Serialization.md")

-- ══════════════════════════════════════════════════════════════
--  RÉSUMÉ FINAL
-- ══════════════════════════════════════════════════════════════

log("\n╔════════════════════════════════════════════════════════╗")
log("║                    RÉSUMÉ                              ║")
log("╚════════════════════════════════════════════════════════╝\n")

log("✅ Modèle configuré avec succès!")
log(string.format("  • Tag: %s", RUN_TAG))
log(string.format("  • Type: %s", MODEL_TYPE))
log(string.format("  • Paramètres: %d (%.2fM)", param_count, param_count / 1e6))
log(string.format("  • Mémoire: %.2f MB (%.2f MB avec compression)", memory_mb, memory_mb * 0.5))

-- Stats MemoryGuard finales
Mimir.MemoryGuard.printStats()

-- Stats Allocator
log("\n📊 Stats Allocator:")
if type(Mimir.Allocator) == "table" then
    local stats_fn = nil
    if type(Mimir.Allocator.get_stats) == "function" then
        stats_fn = Mimir.Allocator.get_stats
    elseif type(Mimir.Allocator.getStats) == "function" then
        stats_fn = Mimir.Allocator.getStats
    end

    if stats_fn ~= nil then
        local ok_stats, stats = pcall(stats_fn)
        if ok_stats and type(stats) == "table" then
            -- Champs documentés: tensor_count, loaded_count (d'autres peuvent exister selon la runtime)
            log(string.format("  • tensor_count: %s", tostring(stats.tensor_count)))
            log(string.format("  • loaded_count: %s", tostring(stats.loaded_count)))
            -- Dump léger des clés additionnelles
            for k, v in pairs(stats) do
                if k ~= "tensor_count" and k ~= "loaded_count" then
                    log(string.format("  • %s: %s", tostring(k), tostring(v)))
                end
            end
        else
            log("  ⚠️  Allocator.get_stats() a échoué")
        end
    else
        log("  ℹ️  Allocator.get_stats() indisponible (utilisez Mimir.Allocator.print_stats() manuellement si besoin)")
    end
else
    log("  ℹ️  Allocator indisponible")
end

log("\n💡 Prochaines étapes:")
log("  1. Implémenter votre boucle de training (étape 8)")
log("  2. Charger et préparer un dataset (étape 7)")
log("  3. Tester avec forward/backward sur données synthétiques")
log("  4. Sauvegarder les checkpoints (étape 9)")

log("\n📚 Documentation:")
log("  • mimir-api.lua - Référence API complète (16 modules)")
log("  • docs/00-INDEX.md - Index principal (point d'entrée)")
log("  • docs/02-User-Guide/ - Guides utilisateur")
log("  • docs/03-API-Reference/10-Model.md - Multi-input (forward map) + patterns")
log("  • docs/03-API-Reference/20-Lua-API-Cpp-Mapping.md - Mapping Lua ↔ C++ (dont set_layer_io)")
log("  • README.md - Vue d'ensemble")

log("\n🆕 Nouveautés v2.4.0:")
log("  • Mode Strict activé (0 pass-through)")
log("  • Support multi-input avec Mimir.Model.set_layer_io()")
log("  • Résidual & skip connections")
log("  • 115+ fonctions API documentées")

log("\n✨ Bon apprentissage avec Mímir Framework v3.5.0! ✨\n")
