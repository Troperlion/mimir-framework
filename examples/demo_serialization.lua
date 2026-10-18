#!/usr/bin/env lua
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

--[[
    Démonstration du nouveau système de sérialisation Mímir
    
    Teste les 3 formats:
    - SafeTensors (production)
    - RawFolder (debug)
    - DebugJson (development)
]]

print("╔════════════════════════════════════════════════════════╗")
print("║  Démo Système de Sérialisation Mímir                  ║")
print("╚════════════════════════════════════════════════════════╝\n")

-- ══════════════════════════════════════════════════════════════
-- Configuration
-- ══════════════════════════════════════════════════════════════

local CONFIG = {
    model_name = "demo_serialization",
    embed_dim = 64,
    num_layers = 3,
    layer_size = 128,
    
    -- Paths de test
    safetensors_path = "/tmp/mimir_demo.safetensors",
    raw_folder_path = "/tmp/mimir_demo_raw",
    debug_json_path = "/tmp/mimir_demo_debug.json",
}

local checkpoint = Mimir.Serialization

-- ══════════════════════════════════════════════════════════════
-- Étape 1: Créer un modèle de test
-- ══════════════════════════════════════════════════════════════

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  1. Création du modèle")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Créer modèle
Mimir.Model.create("transformer", {
    embed_dim = CONFIG.embed_dim,
    num_layers = CONFIG.num_layers,
    ff_dim = CONFIG.layer_size * 2,
    num_heads = 4
})

-- Build
local ok, params = Mimir.Model.build()
if not ok then
    print("❌ Erreur build: " .. (params or "unknown"))
    os.exit(1)
end

print("✓ Modèle créé: " .. params .. " paramètres")

-- Init weights
Mimir.Model.init_weights("xavier", 42)
print("✓ Poids initialisés\n")

-- ══════════════════════════════════════════════════════════════
-- Étape 2: SafeTensors (Production)
-- ══════════════════════════════════════════════════════════════

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  2. Test SafeTensors (Production)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Sauvegarder
print("📦 Sauvegarde SafeTensors...")
local ok, err = checkpoint.save(
    CONFIG.safetensors_path,
    "safetensors",
    {
        save_tokenizer = false,  -- SafeTensors = weights only
        include_git_info = true
    }
)

if not ok then
    print("❌ Échec sauvegarde: " .. (err or "unknown"))
else
    print("✓ Sauvegardé: " .. CONFIG.safetensors_path)
    
    -- Afficher la taille
    local cmd = "ls -lh " .. CONFIG.safetensors_path .. " 2>/dev/null | awk '{print $5}'"
    local handle = io.popen(cmd)
    if handle then
        local size = handle:read("*a"):gsub("%s+", "")
        handle:close()
        if size ~= "" then
            print("  Taille: " .. size)
        end
    end
end

-- Charger dans un nouveau modèle
print("\n🔄 Chargement SafeTensors...")

-- Créer nouveau modèle avec même structure
Mimir.Model.create("transformer", {
    embed_dim = CONFIG.embed_dim,
    num_layers = CONFIG.num_layers,
    ff_dim = CONFIG.layer_size * 2,
    num_heads = 4
})
Mimir.Model.build()

local ok, err = checkpoint.load(
    CONFIG.safetensors_path,
    "safetensors",
    {}
)

if not ok then
    print("❌ Échec chargement: " .. (err or "unknown"))
else
    print("✓ Chargé avec succès")
end

print("")

-- ══════════════════════════════════════════════════════════════
-- Étape 3: RawFolder (Debug & Development)
-- ══════════════════════════════════════════════════════════════

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  3. Test RawFolder (Debug)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

-- Sauvegarder
print("📦 Sauvegarde RawFolder...")
local ok, err = checkpoint.save(
    CONFIG.raw_folder_path,
    "raw_folder",
    {
        save_tokenizer = true,
        save_encoder = false,
        include_git_info = true
    }
)

if not ok then
    print("❌ Échec sauvegarde: " .. (err or "unknown"))
else
    print("✓ Sauvegardé: " .. CONFIG.raw_folder_path)
    
    -- Afficher la structure
    print("\n  Structure:")
    os.execute("tree -L 2 " .. CONFIG.raw_folder_path .. " 2>/dev/null || find " .. CONFIG.raw_folder_path .. " -maxdepth 2 -type f | head -10")
    
    -- Afficher manifest.json
    print("\n  Contenu manifest.json:")
    os.execute("cat " .. CONFIG.raw_folder_path .. "/manifest.json | head -20")
end

-- Charger
print("\n🔄 Chargement RawFolder...")

Mimir.Model.create("transformer", {
    embed_dim = CONFIG.embed_dim,
    num_layers = CONFIG.num_layers,
    ff_dim = CONFIG.layer_size * 2,
    num_heads = 4
})

local ok, err = checkpoint.load(
    CONFIG.raw_folder_path,
    "raw_folder",
    {
        load_tokenizer = true,
        verify_checksums = true
    }
)

if not ok then
    print("❌ Échec chargement: " .. (err or "unknown"))
else
    print("✓ Chargé avec succès (avec validation checksums)")
end

print("")

-- ══════════════════════════════════════════════════════════════
-- Étape 4: DebugJson (Development only)
-- ══════════════════════════════════════════════════════════════

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  4. Test DebugJson (Development)")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

print("⚠️  Format DEBUG uniquement - Petits modèles seulement!")
print("📦 Sauvegarde DebugJson...")

local ok, err = checkpoint.save(
    CONFIG.debug_json_path,
    "debug_json",
    {
        debug_max_values = 50  -- Max 50 valeurs par tensor
    }
)

if not ok then
    print("❌ Échec sauvegarde: " .. (err or "unknown"))
else
    print("✓ Sauvegardé: " .. CONFIG.debug_json_path)
    
    -- Afficher la taille
    local cmd = "ls -lh " .. CONFIG.debug_json_path .. " 2>/dev/null | awk '{print $5}'"
    local handle = io.popen(cmd)
    if handle then
        local size = handle:read("*a"):gsub("%s+", "")
        handle:close()
        if size ~= "" then
            print("  Taille: " .. size)
        end
    end
    
    -- Afficher un extrait
    print("\n  Extrait (format JSON):")
    os.execute("cat " .. CONFIG.debug_json_path .. " | head -30")
end

print("")

-- ══════════════════════════════════════════════════════════════
-- Étape 5: Auto-détection de format
-- ══════════════════════════════════════════════════════════════

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  5. Auto-détection de format")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

print("🔍 Chargement avec auto-détection...")

-- SafeTensors (détecté par extension)
Mimir.Model.create("transformer", {
    embed_dim = CONFIG.embed_dim,
    num_layers = CONFIG.num_layers,
    ff_dim = CONFIG.layer_size * 2,
    num_heads = 4
})
Mimir.Model.build()

local ok, err = checkpoint.load(CONFIG.safetensors_path)
if ok then
    print("✓ Auto-détecté: SafeTensors")
else
    print("❌ Échec: " .. (err or "unknown"))
end

-- RawFolder (détecté par manifest.json)
local ok, err = checkpoint.load(CONFIG.raw_folder_path)
if ok then
    print("✓ Auto-détecté: RawFolder")
else
    print("❌ Échec: " .. (err or "unknown"))
end

print("")

-- ══════════════════════════════════════════════════════════════
-- Nettoyage
-- ══════════════════════════════════════════════════════════════

print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  Nettoyage")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

print("🧹 Suppression des fichiers de test...")
os.execute("rm -f " .. CONFIG.safetensors_path .. " 2>/dev/null")
os.execute("rm -rf " .. CONFIG.raw_folder_path .. " 2>/dev/null")
os.execute("rm -f " .. CONFIG.debug_json_path .. " 2>/dev/null")
print("✓ Nettoyage terminé\n")

-- ══════════════════════════════════════════════════════════════
-- Résumé
-- ══════════════════════════════════════════════════════════════

print("╔════════════════════════════════════════════════════════╗")
print("║  ✅ Démo Terminée                                      ║")
print("╠════════════════════════════════════════════════════════╣")
print("║  3 formats testés:                                     ║")
print("║    • SafeTensors (production, interopérable)           ║")
print("║    • RawFolder (debug, checksums)                      ║")
print("║    • DebugJson (dev, petits modèles)                   ║")
print("╚════════════════════════════════════════════════════════╝")
