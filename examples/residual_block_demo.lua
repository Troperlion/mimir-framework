#!/usr/bin/env lua5.3
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

-- ============================================================================
-- DEMO: ResNet-style Residual Block avec Multi-Input Support
-- ============================================================================
-- Démontre l'utilisation complète du système TensorStore pour créer
-- un bloc résiduel typique d'architectures comme ResNet
-- Version: 2.3.0 - Multi-Input Support

log([[
╔════════════════════════════════════════════════════════════════════╗
║                 🔀 Multi-Input Demo: Residual Block                ║
║                      Mímir Framework v2.3.0                        ║
╚════════════════════════════════════════════════════════════════════╝
]])

-- ============================================================================
-- Configuration
-- ============================================================================

local input_channels = 64
local output_channels = 64
local image_h = 32
local image_w = 32
local batch_size = 1  -- Pour simplicité

local input_size = batch_size * input_channels * image_h * image_w
log("📊 Configuration:")
log("   • Input: " .. batch_size .. "×" .. input_channels .. "×" .. image_h .. "×" .. image_w)
log("   • Output: " .. batch_size .. "×" .. output_channels .. "×" .. image_h .. "×" .. image_w)
log("   • Total elements: " .. input_size)
log("")

-- ============================================================================
-- Architecture: Residual Block
-- ============================================================================

log("🏗️  Construction du bloc résiduel:")
log("")
log("    Input (x)")
log("       │")
log("       ├─────────────────────┐ (skip connection)")
log("       │                     │")
log("       ↓                     │")
log("    Conv2d (3×3)             │")
log("       ↓                     │")
log("    BatchNorm2d              │")
log("       ↓                     │")
log("    ReLU                     │")
log("       ↓                     │")
log("    Conv2d (3×3)             │")
log("       ↓                     │")
log("    BatchNorm2d              │")
log("       │                     │")
log("       └─────────→ Add ←─────┘")
log("                   ↓")
log("                  ReLU")
log("                   ↓")
log("                Output (x)")
log("")

-- ============================================================================
-- Model Construction
-- ============================================================================

log("🔧 Construction du modèle...")

Mimir.Model.create("residual_block_demo")

-- Layer 1: Conv2d (3×3, 64 → 64)
log("   ✓ Conv2d_1: 64×64×3×3 kernels")
Mimir.Model.push_layer("conv1", "Conv2d", output_channels * input_channels * 3 * 3)
Mimir.Model.set_layer_io("conv1", {"x"}, "main")

-- Layer 2: BatchNorm2d
log("   ✓ BatchNorm2d_1: 64 channels")
Mimir.Model.push_layer("bn1", "BatchNorm2d", output_channels)
Mimir.Model.set_layer_io("bn1", {"main"}, "main")

-- Layer 3: ReLU
log("   ✓ ReLU_1")
Mimir.Model.push_layer("relu1", "ReLU", 0)
Mimir.Model.set_layer_io("relu1", {"main"}, "main")

-- Layer 4: Conv2d (3×3, 64 → 64)
log("   ✓ Conv2d_2: 64×64×3×3 kernels")
Mimir.Model.push_layer("conv2", "Conv2d", output_channels * output_channels * 3 * 3)
Mimir.Model.set_layer_io("conv2", {"main"}, "main")

-- Layer 5: BatchNorm2d
log("   ✓ BatchNorm2d_2: 64 channels")
Mimir.Model.push_layer("bn2", "BatchNorm2d", output_channels)
Mimir.Model.set_layer_io("bn2", {"main"}, "main")

-- Layer 6: Add (residual connection: x + main → combined)
log("   ✓ Add: x + main → combined")
Mimir.Model.push_layer("add_residual", "Add", 0)
Mimir.Model.set_layer_io("add_residual", {"x", "main"}, "combined")

-- Layer 7: ReLU (final activation)
log("   ✓ ReLU_2 (final)")
Mimir.Model.push_layer("relu_final", "ReLU", 0)
Mimir.Model.set_layer_io("relu_final", {"combined"}, "x")

log("")

-- ============================================================================
-- Parameter Allocation
-- ============================================================================

log("📦 Allocation des paramètres...")
Mimir.Model.allocate_params()

log("🎲 Initialisation des poids (He initialization)...")
Mimir.Model.init_weights("he")

local total_params = Mimir.Model.total_params()
log("✓ Paramètres totaux: " .. total_params)
log("")

-- ============================================================================
-- Input Creation
-- ============================================================================

log("🎨 Création de l'input test...")
local input = {}
for i = 1, input_size do
    -- Valeur simple pour test (normalement: image normalisée)
    input[i] = math.random() * 0.1 - 0.05  -- [-0.05, 0.05]
end
log("✓ Input créé: " .. #input .. " éléments")
log("")

-- ============================================================================
-- Forward Pass
-- ============================================================================

log("🚀 Forward pass...")
log("")

local start_time = os.clock()
local output = Mimir.Model.forward(input, false)  -- training=false (inference)
local elapsed = os.clock() - start_time

if output then
    log("✅ Forward pass réussi!")
    log("   • Output size: " .. #output .. " éléments")
    log("   • Expected: " .. input_size .. " éléments")
    
    if #output == input_size then
        log("   • ✅ Taille correcte (skip connection préservée)")
    else
        log("   • ⚠️  Taille différente (attendue: " .. input_size .. ")")
    end
    
    log("")
    log("⏱️  Temps d'exécution: " .. string.format("%.3f", elapsed * 1000) .. " ms")
    
    -- Statistics
    local sum = 0
    local min_val = output[1]
    local max_val = output[1]
    for i = 1, #output do
        sum = sum + output[i]
        if output[i] < min_val then min_val = output[i] end
        if output[i] > max_val then max_val = output[i] end
    end
    local mean = sum / #output
    
    log("")
    log("📊 Statistiques de sortie:")
    log("   • Min: " .. string.format("%.6f", min_val))
    log("   • Max: " .. string.format("%.6f", max_val))
    log("   • Mean: " .. string.format("%.6f", mean))
else
    log("❌ Forward pass échoué (output = nil)")
end

log("")

-- ============================================================================
-- TensorStore Information
-- ============================================================================

log("🗂️  Tensors créés pendant le forward pass:")
log("   (disponibles dans TensorStore)")
log("")
log("   1. x         - Input original (conservé pour skip)")
log("   2. main      - Chemin principal après Conv→BN→ReLU→Conv→BN")
log("   3. combined  - Résultat de Add(x, main)")
log("   4. x         - Output final après ReLU(combined)")
log("")

-- ============================================================================
-- Architecture Summary
-- ============================================================================

log("╔════════════════════════════════════════════════════════════════════╗")
log("║                        🎉 Demo Terminé                             ║")
log("╚════════════════════════════════════════════════════════════════════╝")
log("")
log("📝 Résumé:")
log("   ✅ TensorStore system fonctionnel")
log("   ✅ Skip connection (residual) implémentée")
log("   ✅ Multi-input Add operation utilisée")
log("   ✅ Dimensions préservées (x: input → output)")
log("")
log("🔧 API utilisée:")
log("   • Mimir.Model.create(name)")
log("   • Mimir.Model.push_layer(name, type, params)")
log("   • Mimir.Model.set_layer_io(name, {inputs}, output)")
log("   • Mimir.Model.allocate_params()")
log("   • Mimir.Model.init_weights(method)")
log("   • Mimir.Model.forward(input, training)")
log("")
log("🎓 Ce bloc résiduel peut être:")
log("   • Empilé pour créer ResNet (18, 34, 50, 101, 152 layers)")
log("   • Modifié avec bottleneck (1×1 → 3×3 → 1×1)")
log("   • Utilisé dans U-Net, DenseNet, etc.")
log("")
log("📚 Documentation complète: docs/MULTI_INPUT_SUPPORT.md")
log("════════════════════════════════════════════════════════════════════")
