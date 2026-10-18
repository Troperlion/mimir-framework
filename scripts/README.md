# Scripts Mímir v3.5.0

Organisation des scripts Lua pour le framework Mímir.

Cette page décrit les scripts utiles dans le workspace actuel. Elle ne reflète pas l’ancienne arborescence documentaire archivée.

## Structure

```text
scripts/
├── examples/        # Exemples ciblés (inspection, config, classification)
├── inferences/      # Inférence CLIP, SDXL et VAE
├── tests/           # Tests et validation
├── benchmarks/      # Benchmarks
├── training/        # Scripts d'entraînement
├── templates/       # Templates pour nouveaux modèles
├── modules/         # Modules partagés (args, ws server, tokenizer, etc.)
└── tools/           # Outils divers (inspection d'archis, analyse de modèles)
```

## Catégories

### 💡 Exemples (`examples/`)

- `inspect_vae_conv.lua` - construit et inspecte un petit VAEConv sans dataset
- `example_conf_inference.lua` - exemple d'inférence piloté par `CONF`
- `classify_vgg16_feat.lua` - classification à partir d'un checkpoint VGG16/VGG16Feat

### 🔎 Inférences (`inferences/`)

- encodeurs CLIP SDXL 1 et 2
- bloc transformer SDXL
- décodeur VAE Hugging Face
- inspection d'une base SafeTensors externe

Ces scripts nécessitent généralement des checkpoints compatibles. Afficher leur aide avant exécution.

### 🧪 Tests (`tests/`)

Scripts de validation et tests:

- `test_vae_conv_generate.lua` - Génération VAE Conv (smoke)
- `test_serialization_smoke.lua` - Smoke test sérialisation (SafeTensors)
- `test_vae_conv_attention_smoke.lua` / `test_vae_conv_resnet_smoke.lua` - chemins optionnels
- `test_vae_conv_decode_sensitivity.lua` - sensibilité du décodeur au latent
- `test_vae_conv_text_cond_train_smoke.lua` - branche texte conditionnelle

Usage recommandé : commence par `test_serialization_smoke.lua` si tu veux un signal rapide que le binaire, l’API Lua et la sérialisation sont tous opérationnels.

### 🛠️ Outils (`tools/`)

Outils d'inspection et d'analyse:

- `inspect_architectures.lua` - Liste les architectures + dtypes, inspecte une archi et peut l'exporter (`-a`, `-l <arch> -p`, `-d`, `-e <path>`)
- `analyze_model.lua` - Analyse un checkpoint/modèle (SafeTensors / RawFolder / DebugJson) avec graphes `table|blocks|mermaid|mlp_graph` et export image (`--graph-out`)
- `build_tags_vocab.lua` - Construit un vocab de tags depuis un dataset (`.txt` séparés par des points)
- `convert_checkpoint2safetensor.lua` - Convertit un checkpoint RawFolder → SafeTensors
- `convert_safetensors2raw_folder.lua` - Convertit un checkpoint SafeTensors → RawFolder
- `build_mpk.lua` - Construit une source `.mpk` moderne en pseudocode Visu
- `compile_mpk.lua` - Compile une source pseudocode `.mpk` en conteneur binaire v4 opaque `.mpk.bin`
- `../ide/vscode-mpk` - Stub et extension VS Code pour le pseudocode MPK
- `add_vision_mpk_architectures.lua` - Régénère les prototypes MPK `r_cnn`, `yolo`, `ssd` et `deeplab`
- `load_mpk.lua` - Charge/inspecte un `.mpk` et peut créer directement un modèle via le registre
- `mpk_node_wizard.lua` - Assistant interactif (questionnaire nodal) pour assembler des layers dynamiquement et generer un `.mpk`
- `export_arch_mpk.lua` - Exporte une architecture complete vers `.mpk` depuis le registre ou depuis un modele courant (meme hors registre)

Note export complet:

- L'export complet depuis un modèle/registre conserve aussi les types de layers non dynamiques (ex: couches spécialisées internes).
- Ces types peuvent nécessiter le registre au rechargement complet. Pour un import 100% hors-registre avec `--apply-graph`, utiliser des types supportés par `push_layer`.

Validation pre-build MPK:

- `build_mpk.lua` et `mpk_node_wizard.lua` valident maintenant les types de layers du graphe nodal avant ecriture.
- Si un type ne peut pas etre mappe vers un layer supporte par le framework, la generation est bloquee avec une erreur explicite.

Schémas utiles (SVG):

![Inspection et export architectures](../docs/graphs/10_inspection_export.svg)

![Flux Lua JSON inspection](../docs/graphs/26_lua_json_inspection.svg)

### ⚡ Benchmarks (`benchmarks/`)

Scripts de performance:

- `benchmark_official.lua` - Benchmark standard
- `benchmark_stress.lua` - Test de stress
- `benchmark.lua` / `benchmark_complet.lua` / `benchmark_conv_train.lua` - Benchmarks complémentaires
- `benchmark_attention.lua` - attention
- `benchmark_nms.lua` - NMS runtime, sans dataset, avec tailles et seuils configurables
- `dtype_api_smoke.lua` - API dtype
- `spill_cleanup_smoke.lua` - nettoyage des fichiers de spill

Benchmark NMS rapide :

```bash
./bin/mimir --lua scripts/benchmarks/benchmark_nms.lua -- --quick
```

Profil personnalisé :

```bash
./bin/mimir --lua scripts/benchmarks/benchmark_nms.lua -- \
  --boxes 4096 --classes 80 --warmup 5 --iters 50
```

### 🎓 Training (`training/`)

Scripts d'entraînement:

- `train_vae_conv.lua` - Entraînement VAE Conv
- `train_vae_texte.lua` - Entraînement VAE Texte
- `pretrain_vgg16_feat.lua` - préentraînement extracteur perceptuel
- `train_vgg16_tags_multilabel.lua` - classification multi-label
- `prepare_tokenizer_dataset.lua` - préparation tokenizer
- `train_face_recognition_mimir.lua` - reconnaissance faciale

### 📝 Templates (`templates/`)

Templates pour développement:

- `template_new_model.lua` - Lifecycle complet bas niveau (API Mimir directe)
- `template_pipeline_only.lua` - Pipeline minimal (via variables d'environnement)
- `template_pipeline_args.lua` - Pipeline + args CLI + mode registry-first

Choix rapide :

- `template_new_model.lua` si tu dois comprendre le cycle create/allocate/init/train/save détaillé.
- `template_pipeline_only.lua` si tu veux un exemple simple pilotable par env (pas d'args.lua).
- `template_pipeline_args.lua` si tu veux un template production-ready avec :
  - parsing d'arguments via `--flag value` et `--override key=val`,
  - mode **registry-first** : `--from-registry --arch my_arch` charge la config du registre,
  - puis fusionne tes overrides locaux avec `--d-model 256 --layers 4`, etc.

## Utilisation

### Aide CLI des scripts Lua

Les scripts qui chargent `scripts/modules/args.lua` bénéficient de l'aide automatique :

```bash
./bin/mimir --lua <script.lua> -- --help
```

Le flag `--help` affiche :

- une description du script,
- les options/flags detectes dans le script,
- les flags communs (ex: `--help`, et selon le script des flags `args.lua` comme `--viz`, `--htop`, `--override`).

Note: conserver le separateur `--` avant les arguments du script Lua.

### Exécution depuis la racine du projet

```bash
# Templates
./bin/mimir --lua scripts/templates/template_new_model.lua
./bin/mimir --lua scripts/templates/template_pipeline_only.lua
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- --no-train

# Mode registry-first (pipeline_args)
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry --arch transformer \
  --d-model 256 --layers 4 --heads 8 --seq-len 128 \
  --no-train --save /tmp/transformer_demo.safetensors

# Tests
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua

# Outils
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -l vae_conv -e /tmp/vae_conv_export.json
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -l vae_conv -e /tmp/vae_conv_export/
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -l vae_conv -e /tmp/vae_conv_export.safetensors
./bin/mimir --lua scripts/tools/analyze_model.lua -- --in checkpoint/vae_conv-generique/epoch_0018 --graph-format mlp_graph
./bin/mimir --lua scripts/tools/analyze_model.lua -- --in checkpoint/vae_conv-generique/epoch_0018 --graph-format mlp_graph --graph-out /tmp/arch.svg
./bin/mimir --lua scripts/tools/analyze_model.lua -- --in checkpoint/vae_conv-generique/epoch_0018 --graph-format visu
./bin/mimir --lua scripts/tools/analyze_model.lua -- --in checkpoint/vae_conv-generique/epoch_0018 --graph-format visu-bloc
./bin/mimir --lua scripts/tools/analyze_model.lua -- --in checkpoint/vae_conv-generique/epoch_0018 --graph-format visu-tree --graph-units 8

# MPK (Mimir Package Template)
# Le pseudocode lisible de style Visu est le conteneur par défaut.
./bin/mimir --lua scripts/tools/build_mpk.lua -- \
  --name vae_conv_base \
  --type vae_conv \
  --author bri45 \
  --from-registry \
  --template auto \
  --description "VAEConv base package" \
  --viz \
  --compile \
  --out exports/vae_conv_base.mpk

./bin/mimir --lua scripts/tools/load_mpk.lua -- --in exports/vae_conv_registry_pack.mpk --create

# Chargement direct du MPK comme plugin via le registre
./bin/mimir --lua scripts/tests/test_create_path_mpk.lua
./bin/mimir --lua scripts/tools/load_mpk.lua -- --in exports/vae_conv_base.mpk --show-config --create

# Compiler le pseudocode MPK vers le binaire v4 opaque
./bin/mimir --lua scripts/tools/build_mpk.lua -- \
  --name unet_compact \
  --type unet \
  --template unet \
  --description "UNet compact package" \
  --compile \
  --out exports/unet_compact.mpk

# Verification uniquement (checksum + structure)
./bin/mimir --lua scripts/tools/load_mpk.lua -- --in exports/unet_compact.mpk.bin --verify-only

# Wizard interactif nodal (questionnaire)
./bin/mimir --lua scripts/tools/mpk_node_wizard.lua --
./bin/mimir --lua scripts/tools/mpk_node_wizard.lua -- --list-layer-types

# Charger + assembler dynamiquement les layers du graphe nodal
./bin/mimir --lua scripts/tools/load_mpk.lua -- --in exports/custom_model_pack.mpk --create --apply-graph --init xavier --seed 42

# Import hors-registre (fallback automatique create_empty)
./bin/mimir --lua scripts/tools/load_mpk.lua -- --in exports/custom_model_pack.mpk --create

# Reconstruction complète hors-registre (appliquer le graphe)
./bin/mimir --lua scripts/tools/load_mpk.lua -- --in exports/custom_model_pack.mpk --create --apply-graph

# Export complet depuis une architecture du registre
./bin/mimir --lua scripts/tools/export_arch_mpk.lua -- \
  --arch vae_conv \
  --name vae_conv_export_full \
  --description "Export complet vae_conv" \
  --out exports/vae_conv_export_full.mpk

# Export complet depuis un modele courant (standalone/hors registre)
./bin/mimir --lua scripts/tools/export_arch_mpk.lua -- \
  --from-current-model \
  --type custom_graph \
  --name custom_graph_export \
  --out exports/custom_graph_export.mpk

# Benchmark
./bin/mimir --lua scripts/benchmarks/benchmark_official.lua -- --safe --iters 1

# Training
./bin/mimir --lua scripts/training/train_vae_conv.lua -- --help
```

Règles d'export pour `inspect_architectures.lua -e`:

- `*.json` -> export `debugJSON`
- chemin finissant par `/` -> export `rawFolder`
- `*.safetensors` (ou `*.st`) -> export `safetensors`

Lecture pratique de ces commandes :

- templates : découverte et prototypage,
- tests : validation ciblée d’un sous-système,
- benchmarks : mesure rapide et régression perf,
- training : scripts complets avec plus de paramètres et d’état.

### Avec `--conf` et `--run`

```bash
# Exécution de la section lua racine (tâche par défaut)
./bin/mimir --conf configs/vae_conv-config.json

# Sélection d'une tâche nommée
./bin/mimir --conf configs/vae_conv-config.json --run train
./bin/mimir --conf configs/vae_conv-config.json --run infer
./bin/mimir --conf configs/vae_conv-config.json --run resume

# Override à la volée sur une tâche
./bin/mimir --conf configs/vae_conv-config.json --run train --override training.learning_rate=5e-6
./bin/mimir --conf configs/vae_conv-config.json --run train --override model.base_channels=16

# Si la tâche est introuvable, mimir liste les tâches disponibles :
# ❌ --run: tâche 'xxx' introuvable
# 💡 Tâches disponibles: train (Entraînement complet), infer (Génération d'images), ...
```

La section `tasks` d'un fichier de conf définit des tâches nommées, chacune avec son propre bloc `lua`.
Sans `--run`, la section `lua` racine est utilisée.

Le schéma complet des fichiers de conf est documenté dans `configs/conf.schema.json`.

La section racine `env` applique ses variables avant l'exécution. Une valeur du
fichier de conf remplace la valeur héritée du shell ou initialisée par
`run_mimir.sh` :

```json
"env": {
  "OMP_NUM_THREADS": 10,
  "MIMIR_ACCEL_VERBOSE": 0
}
```

### Avec `run_mimir.sh`

```bash
./run_mimir.sh --lua scripts/templates/template_new_model.lua
./run_mimir.sh --conf configs/vae_conv-config.json --run train
```

### Bootstrap environnement (config.sh)

`config.sh` est un script utilitaire a la racine du repo (pas dans `scripts/`) pour Linux Debian/Ubuntu.

```bash
./config.sh
cmake --build build -j"$(nproc)"
```

Variables frequentes : `BUILD_TYPE=Debug`, `ENABLE_VULKAN=0`, `ENABLE_OPENCL=0`, `ENABLE_SFML=0`, `ENABLE_LZ4=0`.

## Statistiques

- **NB** : ce README reflète l'état du dossier `scripts/` à la date de la release.

En cas de doute sur un script, la référence principale reste le code Lua lui-même. Plusieurs fichiers sont pensés comme exemples exécutables avant d’être des tutoriels exhaustifs.

## Pipeline API (modules/pipeline.lua)

Module Lua pour piloter les modèles via le registre d'architectures du framework.

### Utilisation rapide

```lua
local P = dofile(ROOTWORK.."/scripts/modules/pipeline.lua")
local pipe = P.FromRegistry("transformer")  -- ou P.Transformer(cfg) pour la forme spécialisée
pipe:loadDefaultConfig("transformer")
pipe:patchConfig({ d_model = 256, num_layers = 4 })
pipe:build()
pipe:train("dataset.bin", 10, 0.0003)
pipe:save("model.safetensors")
```

### Constructeurs disponibles

- `P.FromRegistry(arch, config, options)` - générique, charge du registre
- `P.Transformer(config)` - constructeur spécialisé
- `P.UNet(config)`, `P.VAE(config)`, `P.ViT(config)`, `P.Diffusion(config)`, etc.

### Méthodes du pipeline

- `pipe:loadDefaultConfig(arch, patch?)` - charge la config du registre
- `pipe:patchConfig(patch)` - fusionne des overrides
- `pipe:getConfig()`, `pipe:getBaseConfig()` - lecture
- `pipe:build()` - create → dtype → build → allocate → init
- `pipe:train(dataset, epochs, lr)` - entraînement
- `pipe:infer(input)` - inférence
- `pipe:save(path)` - sauvegarde (format déduit depuis l'extension)

## Voir aussi

- [Documentation complète](../docs/00-INDEX.md)
- [Guide de démarrage rapide](../docs/01-Getting-Started/01-Quick-Start.md)
- [Vue d'ensemble API Lua](../docs/03-API-Reference/00-API-Overview.md)
- [Workflow modèle](../docs/02-User-Guide/02-Model-Lifecycle.md)
- [Catalogue détaillé des scripts et outils](../docs/02-User-Guide/10-Examples.md)
- [Parcours par niveau](../docs/01-Getting-Started/06-Learning-Paths.md)
- [Développeurs: nouvelles architectures et outils](../docs/06-Contributing/02-New-Architecture-And-Tools.md)

---

**Version**: 3.5.0 | **Date**: 1 octobre 2026
