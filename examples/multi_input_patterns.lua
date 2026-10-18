#!/usr/bin/env lua5.3
local Help = dofile(ROOTWORK.."/scripts/modules/help_cli.lua")
Help.auto_exit_help()

-- ============================================================================
-- EXAMPLES: Multi-Input Support - Cas d'Usage Courants
-- ============================================================================
-- Exemples pratiques d'utilisation du système TensorStore
-- Version: 2.3.0 - Multi-Input Support

log("╔════════════════════════════════════════════════════════════════════╗")
log("║            🎯 Multi-Input Examples - Mímir Framework v2.3          ║")
log("╚════════════════════════════════════════════════════════════════════╝\n")

-- ============================================================================
-- EXEMPLE 1: Residual Connection Simple (ResNet-style)
-- ============================================================================
log("📘 Exemple 1: Residual Connection")
log("   Pattern: Input → Conv → (Input + Conv) → Output")
log("")

Mimir.Model.create("resnet_example")

-- Path principal
Mimir.Model.push_layer("conv", "Conv2d", 64*64*3*3)
Mimir.Model.set_layer_io("conv", {"x"}, "conv_out")

-- Residual: add input + conv_out
Mimir.Model.push_layer("add_residual", "Add", 0)
Mimir.Model.set_layer_io("add_residual", {"x", "conv_out"}, "x")

log("✓ Architecture créée:")
log("   x → conv → conv_out")
log("   x + conv_out → x (residual)")
log("")

-- ============================================================================
-- EXEMPLE 2: Concat Multi-Branch (U-Net Decoder)
-- ============================================================================
log("📘 Exemple 2: Concat Multi-Branch")
log("   Pattern: Input → [Branch1, Branch2] → Concat → Output")
log("")

Mimir.Model.create("concat_example")

-- Branch 1: Encoder features
Mimir.Model.push_layer("encoder", "Conv2d", 32*64*3*3)
Mimir.Model.set_layer_io("encoder", {"x"}, "encoder_feat")

-- Branch 2: Decoder upsampled
Mimir.Model.push_layer("upsample", "UpsampleNearest", 0)
Mimir.Model.set_layer_io("upsample", {"encoder_feat"}, "upsampled")

-- Branch 3: Lateral connection
Mimir.Model.push_layer("lateral", "Conv2d", 32*32*1*1)
Mimir.Model.set_layer_io("lateral", {"x"}, "lateral_feat")

-- Concat toutes les features
Mimir.Model.push_layer("concat_all", "Concat", 0)
Mimir.Model.set_layer_io("concat_all", {"upsampled", "lateral_feat"}, "fused")

log("✓ Architecture créée:")
log("   x → encoder → encoder_feat → upsample → upsampled")
log("   x → lateral → lateral_feat")
log("   [upsampled, lateral_feat] → concat → fused")
log("")

-- ============================================================================
-- EXEMPLE 3: Attention Q/K/V Routing
-- ============================================================================
log("📘 Exemple 3: Attention Q/K/V")
log("   Pattern: Input → [Q, K, V] → Attention → Output")
log("")

Mimir.Model.create("attention_example")

-- Projections Q, K, V
Mimir.Model.push_layer("query_proj", "Linear", 512*512)
Mimir.Model.set_layer_io("query_proj", {"x"}, "Q")

Mimir.Model.push_layer("key_proj", "Linear", 512*512)
Mimir.Model.set_layer_io("key_proj", {"x"}, "K")

Mimir.Model.push_layer("value_proj", "Linear", 512*512)
Mimir.Model.set_layer_io("value_proj", {"x"}, "V")

-- Attention multi-head (conceptuel)
Mimir.Model.push_layer("attention", "MultiHeadAttention", 0)
Mimir.Model.set_layer_io("attention", {"Q", "K", "V"}, "attn_out")

log("✓ Architecture créée:")
log("   x → query_proj → Q")
log("   x → key_proj → K")
log("   x → value_proj → V")
log("   Attention(Q, K, V) → attn_out")
log("")

-- ============================================================================
-- EXEMPLE 4: DenseNet-style Dense Connection
-- ============================================================================
log("📘 Exemple 4: DenseNet Dense Connection")
log("   Pattern: Concat toutes les features précédentes")
log("")

Mimir.Model.create("densenet_example")

-- Layer 1
Mimir.Model.push_layer("conv1", "Conv2d", 32*3*3*3)
Mimir.Model.set_layer_io("conv1", {"x"}, "feat1")

-- Layer 2 (prend x et feat1)
Mimir.Model.push_layer("concat1", "Concat", 0)
Mimir.Model.set_layer_io("concat1", {"x", "feat1"}, "dense1")

Mimir.Model.push_layer("conv2", "Conv2d", 32*(3+32)*3*3)
Mimir.Model.set_layer_io("conv2", {"dense1"}, "feat2")

-- Layer 3 (prend tout: x, feat1, feat2)
Mimir.Model.push_layer("concat2", "Concat", 0)
Mimir.Model.set_layer_io("concat2", {"x", "feat1", "feat2"}, "dense2")

log("✓ Architecture créée:")
log("   x → conv1 → feat1")
log("   [x, feat1] → concat1 → dense1 → conv2 → feat2")
log("   [x, feat1, feat2] → concat2 → dense2 (dense connection)")
log("")

-- ============================================================================
-- EXEMPLE 5: Split + Parallel Processing
-- ============================================================================
log("📘 Exemple 5: Split + Parallel Processing")
log("   Pattern: Input → Split → [Path1, Path2] → Concat → Output")
log("")

Mimir.Model.create("split_example")

-- Split input en 2 parties
Mimir.Model.push_layer("split", "Split", 0)
Mimir.Model.set_layer_io("split", {"x"}, "parts")
-- Crée: parts_0, parts_1

-- Process chaque partie
Mimir.Model.push_layer("conv_part0", "Conv2d", 32*32*3*3)
Mimir.Model.set_layer_io("conv_part0", {"parts_0"}, "processed_0")

Mimir.Model.push_layer("conv_part1", "Conv2d", 32*32*3*3)
Mimir.Model.set_layer_io("conv_part1", {"parts_1"}, "processed_1")

-- Recombiner
Mimir.Model.push_layer("concat_parts", "Concat", 0)
Mimir.Model.set_layer_io("concat_parts", {"processed_0", "processed_1"}, "x")

log("✓ Architecture créée:")
log("   x → split → [parts_0, parts_1]")
log("   parts_0 → conv_part0 → processed_0")
log("   parts_1 → conv_part1 → processed_1")
log("   [processed_0, processed_1] → concat → x")
log("")

-- ============================================================================
-- EXEMPLE 6: FPN (Feature Pyramid Network)
-- ============================================================================
log("📘 Exemple 6: Feature Pyramid Network")
log("   Pattern: Multi-scale feature fusion")
log("")

Mimir.Model.create("fpn_example")

-- Bottom-up: extraire features à différentes échelles
Mimir.Model.push_layer("conv_low", "Conv2d", 64*3*3*3)
Mimir.Model.set_layer_io("conv_low", {"x"}, "feat_low")

Mimir.Model.push_layer("pool1", "MaxPool2d", 0)
Mimir.Model.set_layer_io("pool1", {"feat_low"}, "feat_mid_in")

Mimir.Model.push_layer("conv_mid", "Conv2d", 128*64*3*3)
Mimir.Model.set_layer_io("conv_mid", {"feat_mid_in"}, "feat_mid")

Mimir.Model.push_layer("pool2", "MaxPool2d", 0)
Mimir.Model.set_layer_io("pool2", {"feat_mid"}, "feat_high_in")

Mimir.Model.push_layer("conv_high", "Conv2d", 256*128*3*3)
Mimir.Model.set_layer_io("conv_high", {"feat_high_in"}, "feat_high")

-- Top-down: upsample et fusionner
Mimir.Model.push_layer("upsample_high", "UpsampleNearest", 0)
Mimir.Model.set_layer_io("upsample_high", {"feat_high"}, "high_up")

Mimir.Model.push_layer("concat_mid_high", "Concat", 0)
Mimir.Model.set_layer_io("concat_mid_high", {"feat_mid", "high_up"}, "pyramid_mid")

log("✓ Architecture créée:")
log("   Bottom-up: x → conv_low → pool → conv_mid → pool → conv_high")
log("   Top-down: feat_high → upsample → + feat_mid → pyramid")
log("")

-- ============================================================================
-- RÉSUMÉ
-- ============================================================================
log("╔════════════════════════════════════════════════════════════════════╗")
log("║                      📊 Résumé des Exemples                        ║")
log("╚════════════════════════════════════════════════════════════════════╝")
log("")
log("✅ 6 patterns d'architecture démontrés:")
log("   1. Residual Connection (ResNet)")
log("   2. Multi-Branch Concat (U-Net)")
log("   3. Attention Q/K/V Routing")
log("   4. Dense Connection (DenseNet)")
log("   5. Split + Parallel Processing")
log("   6. Feature Pyramid (FPN)")
log("")
log("🔧 API utilisée dans tous les exemples:")
log("   • Mimir.Model.create(name)")
log("   • Mimir.Model.push_layer(name, type, params)")
log("   • Mimir.Model.set_layer_io(name, {inputs}, output)")
log("")
log("📚 Pour plus d'infos:")
log("   • docs/MULTI_INPUT_SUPPORT.md - Guide complet")
log("   • examples/residual_block_demo.lua - Démo exécutable")
log("   • scripts/tests/test_branches.lua - Tests fonctionnels")
log("")
log("════════════════════════════════════════════════════════════════════")
