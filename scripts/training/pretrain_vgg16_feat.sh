#!/usr/bin/env bash
#
# ============================================================================
#  Fichier de configuration / lancement : pré-entraînement de `vgg16_feat`
# ============================================================================
#
# Objectif :
#   Ré-entraîner la *partie features* du modèle de perception (`vgg16_feat`)
#   afin de la réutiliser ensuite comme extracteur perceptuel lors de
#   l'entraînement du VAE conv (`train_vae_conv.lua --perceptual-ckpt ...`).
#
# Pipeline attendu :
#   1) ./scripts/training/pretrain_vgg16_feat.sh           # (ce script)
#   2) ./bin/mimir --lua scripts/training/train_vae_conv.lua -- \
#        --perceptual-weight 0.1 \
#        --perceptual-ckpt "$OUT_DIR" \
#        --perceptual-base-channels "$BASE_CHANNELS"
#
# Notes de cohérence avec `train_vae_conv.lua` :
#   - OUT_DIR        == valeur par défaut de `--perceptual-ckpt`
#                       ("./checkpoint/vgg16_feat_pretrain").
#   - BASE_CHANNELS  == `--perceptual-base-channels` côté VAEConv (>= 4 imposé
#                       par l'archi C++ de `vgg16_feat`).
#
# Usage :
#   ./scripts/training/pretrain_vgg16_feat.sh                # lancement standard
#   RESUME=1 ./scripts/training/pretrain_vgg16_feat.sh       # reprise checkpoint
#   EPOCHS=20 LR=5e-5 ./scripts/training/pretrain_vgg16_feat.sh
#   HTOP=1 ./scripts/training/pretrain_vgg16_feat.sh         # TUI htop seul (recommandé)
#   VIZ=1 ./scripts/training/pretrain_vgg16_feat.sh          # fenêtre SFML (peut segfault)
#
# Toutes les variables ci-dessous sont surchageables via l'environnement.
# ----------------------------------------------------------------------------

set -euo pipefail

# Se placer à la racine du dépôt (le binaire attend des chemins relatifs).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ----------------------------------------------------------------------------
#  Binaire & script Lua
# ----------------------------------------------------------------------------
MIMIR_BIN="${MIMIR_BIN:-./run_mimir.sh}"
LUA_SCRIPT="${LUA_SCRIPT:-scripts/training/pretrain_vgg16_feat.lua}"

# ----------------------------------------------------------------------------
#  Données
# ----------------------------------------------------------------------------
# Répertoire racine du dataset images (chargé par Mimir.Dataset.load).
DATASET_ROOT="${DATASET_ROOT:-datasets/coco}"

# Vocabulaire de référence (tags/captions) — fichier unique imposé.
# Utilisé pour le contexte de tags/captions associé à la perception ; il sert
# de source de vocabulaire unique pour rester aligné avec le VAE conv.
VOCAB_FILE="${VOCAB_FILE:-checkpoint/coco_tags_vocab_10k.txt}"

# ----------------------------------------------------------------------------
#  Sorties
# ----------------------------------------------------------------------------
# IMPORTANT : doit correspondre au `--perceptual-ckpt` par défaut du VAE conv.
OUT_DIR="${OUT_DIR:-checkpoint/vgg16_feat_pretrain}"

# ----------------------------------------------------------------------------
#  Hyperparamètres d'entraînement
# ----------------------------------------------------------------------------
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
SEED="${SEED:-1337}"
INIT="${INIT:-xavier}"

# Image de pré-entraînement : 256x256 (rapide). Les poids des convs sont
# réutilisables tels quels en 512x512 (reshape/pool sans poids).
IMAGE_W="${IMAGE_W:-256}"
IMAGE_H="${IMAGE_H:-256}"
IMAGE_C="${IMAGE_C:-3}"

# `base_channels` du vgg16_feat == `perceptual_base_channels` côté VAEConv.
# L'archi C++ force base_channels >= 4.
BASE_CHANNELS="${BASE_CHANNELS:-4}"

# Self-supervised pretrain grid.
PRETRAIN_GRID="${PRETRAIN_GRID:-8}"

# ----------------------------------------------------------------------------
#  Optimiseur
# ----------------------------------------------------------------------------
OPTIMIZER="${OPTIMIZER:-adamw}"
BETA1="${BETA1:-0.9}"
BETA2="${BETA2:-0.999}"
EPSILON="${EPSILON:-1e-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
DECAY_STRATEGY="${DECAY_STRATEGY:-cosine}"
# Warmup : monte le LR progressivement (anti-divergence en début d'entraînement).
WARMUP_STEPS="${WARMUP_STEPS:-500}"
# Gradient clipping (L2 global). CRITIQUE : sans clipping, la boucle vgg16_feat
# diverge (gradient explosion -> loss ~1e13 vers le step ~90). 0 = désactivé.
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"

# ----------------------------------------------------------------------------
#  Checkpoints / logs
# ----------------------------------------------------------------------------
AUTOSAVE_EVERY_EPOCHS="${AUTOSAVE_EVERY_EPOCHS:-1}"
MAX_ITEMS="${MAX_ITEMS:-0}"   # 0 = tout le dataset
LOG_EVERY="${LOG_EVERY:-10}"

# ----------------------------------------------------------------------------
#  Mémoire / matériel
# ----------------------------------------------------------------------------
MEM_GB="${MEM_GB:-10}"
ALLOC_GB="${ALLOC_GB:-$MEM_GB}"
COMPRESSION="${COMPRESSION:-true}"
HW="${HW:-true}"

# ----------------------------------------------------------------------------
#  Reprise / visualisation
# ----------------------------------------------------------------------------
RESUME="${RESUME:-0}"   # 1 pour reprendre depuis OUT_DIR

# Monitoring :
#   HTOP=1  -> TUI htop seul (sans fenêtre SFML).  Recommandé.
#   VIZ=1   -> fenêtre graphique SFML/OpenGL (+ htop).
#
# ⚠️  La fenêtre SFML (--viz) peut provoquer un `segmentation fault` au démarrage
#     sur certaines configs (driver OpenGL / threading SFML). Le crash est dans
#     le thread de rendu graphique, PAS dans l'entraînement ni la config.
#     En cas de segfault avec VIZ=1, utiliser HTOP=1 (ou aucun des deux).
HTOP="${HTOP:-0}"      # 1 pour le TUI htop seul (pas de fenêtre SFML)
VIZ="${VIZ:-0}"        # 1 pour activer la viz graphique SFML (+ htop)

# ----------------------------------------------------------------------------
#  Vérifications préalables
# ----------------------------------------------------------------------------
if [[ ! -x "$MIMIR_BIN" ]]; then
  echo "❌ Binaire introuvable ou non exécutable : $MIMIR_BIN" >&2
  echo "   Compiler d'abord : make build" >&2
  exit 1
fi
if [[ ! -f "$LUA_SCRIPT" ]]; then
  echo "❌ Script Lua introuvable : $LUA_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$VOCAB_FILE" ]]; then
  echo "❌ Fichier vocabulaire introuvable : $VOCAB_FILE" >&2
  exit 1
fi

# ----------------------------------------------------------------------------
#  Assemblage des arguments
# ----------------------------------------------------------------------------
ARGS=(
  --dataset-root "$DATASET_ROOT"
  --out-dir "$OUT_DIR"
  --epochs "$EPOCHS"
  --lr "$LR"
  --seed "$SEED"
  --init "$INIT"
  --image-w "$IMAGE_W"
  --image-h "$IMAGE_H"
  --image-c "$IMAGE_C"
  --base-channels "$BASE_CHANNELS"
  --pretrain-grid "$PRETRAIN_GRID"
  --optimizer "$OPTIMIZER"
  --beta1 "$BETA1"
  --beta2 "$BETA2"
  --epsilon "$EPSILON"
  --weight-decay "$WEIGHT_DECAY"
  --decay-strategy "$DECAY_STRATEGY"
  --warmup-steps "$WARMUP_STEPS"
  --grad-clip-norm "$GRAD_CLIP_NORM"
  --autosave-every-epochs "$AUTOSAVE_EVERY_EPOCHS"
  --max-items "$MAX_ITEMS"
  --log-every "$LOG_EVERY"
  --mem-gb "$MEM_GB"
  --alloc-gb "$ALLOC_GB"
  --compression "$COMPRESSION"
  --hw "$HW"
)

if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
  ARGS+=(--resume)
fi
if [[ "$VIZ" == "1" || "$VIZ" == "true" ]]; then
  # --viz démarre aussi htop (alias). Fenêtre graphique SFML => risque de segfault.
  ARGS+=(--viz)
elif [[ "$HTOP" == "1" || "$HTOP" == "true" ]]; then
  # TUI htop seul, sans la fenêtre SFML (--no-viz pour être explicite).
  ARGS+=(--htop --no-viz)
fi

# ----------------------------------------------------------------------------
#  Récapitulatif
# ----------------------------------------------------------------------------
echo "╔════════════════════════════════════════════════╗"
echo "║   PRÉ-ENTRAÎNEMENT vgg16_feat (perception)     ║"
echo "╚════════════════════════════════════════════════╝"
echo "  dataset_root   = $DATASET_ROOT"
echo "  vocab_file     = $VOCAB_FILE"
echo "  out_dir        = $OUT_DIR"
echo "  image          = ${IMAGE_W}x${IMAGE_H}x${IMAGE_C}"
echo "  base_channels  = $BASE_CHANNELS"
echo "  epochs / lr    = $EPOCHS / $LR"
echo "  optimizer      = $OPTIMIZER (wd=$WEIGHT_DECAY, decay=$DECAY_STRATEGY)"
echo "  warmup / clip  = $WARMUP_STEPS / $GRAD_CLIP_NORM"
echo "  resume         = $RESUME"
echo "  monitoring     = htop=$HTOP viz=$VIZ"
echo ""

# ----------------------------------------------------------------------------
#  Lancement
# ----------------------------------------------------------------------------
exec "$MIMIR_BIN" --lua "$LUA_SCRIPT" -- "${ARGS[@]}"
