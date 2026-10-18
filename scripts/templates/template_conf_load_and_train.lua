#!/usr/bin/env lua
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

---@diagnostic disable: undefined-global

-- scripts/templates/template_conf_load_and_train.lua
--
-- Template: charger une config via --conf et entraîner un modèle
--
-- Usage:
--   ./bin/mimir --conf configs/example_conf_driven.json
--
-- Notes:
-- - CONF est automatiquement injecté par le runtime (--conf mode)
-- - Ce script utilise les valeurs dans CONF pour créer/entraîner un modèle

local log = print

if not CONF then
    error("CONF not injected! Use --conf mode to load this script.")
end

log("═══════════════════════════════════════════════")
log("📝 Template: Config-Driven Model Training")
log("═══════════════════════════════════════════════")
log("")

-- 1. Afficher la config chargée
log("📋 Configuration chargée:")
log("  • CONF_PATH: " .. (CONF_PATH or "?"))
log("  • CONF_DIR:  " .. (CONF_DIR or "?"))
log("")

-- 2. Extraire les paramètres du modèle depuis CONF
if not CONF.model then
    error("CONF.model not found in config!")
end

local model_cfg = CONF.model
log("🏗️  Paramètres du modèle depuis CONF:")
log("  • Architecture: " .. (model_cfg.architecture or "?"))
log("  • Vocab size:   " .. (model_cfg.vocab_size or "?"))
log("  • Seq length:   " .. (model_cfg.seq_len or "?"))
log("  • D_model:      " .. (model_cfg.d_model or "?"))
log("  • Num layers:   " .. (model_cfg.num_layers or "?"))
log("")

-- 3. Créer le modèle via l'architecte (v3.0+ moderne)
log("🔨 Création du modèle...")
local ok, err = Mimir.Architectures.available()
if not ok then
    error("Architectures.available() failed: " .. tostring(err))
end
log("  ✓ Architectures disponibles")

-- Obtenir la config par défaut pour cette architecture
local arch = model_cfg.architecture or "transformer"
local cfg, cfg_err = Mimir.Architectures.default_config(arch)
if not cfg then
    error("Cannot get default config for " .. arch .. ": " .. tostring(cfg_err))
end
log("  ✓ Config de base pour '" .. arch .. "' obtenue")

-- Overrider les valeurs depuis CONF
for k, v in pairs(model_cfg) do
    if k ~= "architecture" then
        cfg[k] = v
    end
end
log("  ✓ Overrides appliqués depuis CONF.model")

-- Créer le modèle
local model_ok, model_err = Mimir.Model.create(arch, cfg)
if not model_ok then
    error("Model.create failed: " .. tostring(model_err))
end
log("  ✓ Modèle créé")

-- 4. Allouer les paramètres
log("⚡ Allocation des paramètres...")
local alloc_ok, alloc_err = Mimir.Model.allocate_params()
if not alloc_ok then
    error("Model.allocate_params failed: " .. tostring(alloc_err))
end
log("  ✓ Paramètres alloués")

-- 5. Initialiser les poids
log("🎲 Initialisation des poids...")
local init_ok, init_err = Mimir.Model.init_weights("xavier", 42)
if not init_ok then
    error("Model.init_weights failed: " .. tostring(init_err))
end
log("  ✓ Poids initialisés (xavier, seed=42)")

-- 6. Tracer les paramètres d'entraînement depuis CONF
if CONF.training then
    log("")
    log("🎓 Paramètres d'entraînement depuis CONF.training:")
    log("  • Num epochs:      " .. (CONF.training.num_epochs or "?"))
    log("  • Batch size:      " .. (CONF.training.batch_size or "?"))
    log("  • Learning rate:   " .. (CONF.training.learning_rate or "?"))
    log("  • Optimizer:       " .. (CONF.training.optimizer or "?"))
end

-- 7. Exemple: forward pass simplifié
log("")
log("🚀 Test forward pass...")
local dummy_input = {}
for i = 1, (cfg.seq_len or 64) do
    table.insert(dummy_input, 100)
end

local forward_out = Mimir.Model.forward({ __input__ = dummy_input }, false)
if forward_out then
    log("  ✓ Forward pass réussi")
    if type(forward_out) == "table" then
        log("    Output length: " .. (#forward_out or "?"))
    else
        log("    Output type: " .. type(forward_out))
    end
else
    log("  ⚠️  Forward pass failed")
end

log("")
log("✅ Template exécuté avec succès!")
log("   Prochaines étapes:")
log("   - Implémenter la boucle d'entraînement (CONF.training)")
log("   - Charger un dataset (CONF.dataset)")
log("   - Sauvegarder des checkpoints")
log("")
