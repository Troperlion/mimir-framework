# Changelog

Toutes les modifications notables du Mímir Framework sont documentées ici.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [Non publié]

## [3.6.0] - 2026-10-22

### Performances, Viz et benchmarks

- CPU : backward Linear/MatMul vectorisé, forward BatchMatMul par blocs,
  temporaires de convolution réutilisés, copies Concat/Stack supprimées et
  attention par blocs de 128 requêtes. API et modèles existants conservés.
- Routage : réutilisation des conteneurs avec rafraîchissement des capacités,
  contrôle NaN/Inf conservé ; scratchpads non initialisés explicites.
- Viz Qt/GTK : événements et focus reliés au flux commun, rendu des scènes
  actualisées et caches de textures encodées/décodées.
- Colab : sources embarquées synchronisées, helper de mesure autonome,
  identités distinctes CUDA/Vulkan/OpenCL et profil portable borné commun aux
  frameworks ; RSS externe et résultats MNIST séparés des microbenchmarks.
- Documentation : procédures Colab, limites de mesure, chemins forward/backward
  Vulkan/OpenCL et contrats d’initialisation des scratchpads actualisés.

### Routage, optimizers et compatibilité

- Planification `cost` par défaut et calibration forward/backward sur copies
  de travail, avec contrôle numérique, médiane de cinq mesures et budget mémoire.
  Les opérations non rejouables conservent leur parcours normal.
- Optimizers : sélection empirique du runtime, caches invalidés lors des changements
  de configuration, états compacts et sauvegarde canonique des moments actifs.
- Kernels portables : exécution parallèle par opération, buffers réutilisés,
  géométrie de convolution partagée et constantes séquentielles interpolées.
- Conversions typées F16/BF16/F32 par blocs et publication des contrôles runtime
  évitée lorsque leur état effectif n’a pas changé.
- Diagnostics optionnels des gradients : médiane, p95, maximum et taux de clipping
  sur les 256 dernières mises à jour, exposés au monitoring.
- Régressions ciblées : calibration, reprise des optimizers, tenseurs typés,
  kernels CPU/portables et limites des annotations de l’interface.
- `config.sh` : sélection explicite du backend Viz, répertoire de build séparé
  et options `--enable` / `--disable` pour les profils CI reproductibles.
  Les workflows Linux installent et configurent via `./config.sh`.

### Corrections de validation

- Le cache des routes forward conserve le rejet d’un runtime ayant échoué,
  jusqu’au rafraîchissement explicite des routes.
- Les tests de priorité et de parité de kernels sont indépendants de la
  calibration des vitesses ; les tests dédiés continuent de vérifier le choix mesuré.
- Les espacements d’alignement sont inclus dans le budget de scratch convolution.
- La compilation croisée n’importe plus les chemins Cairo du système hôte.
  Le bridge JavaScript POSIX reste désactivé sur Windows, tandis que les bridges
  C#/Rust utilisent les en-têtes de processus adaptés à cette plateforme.

### Validation et publication

- Documentation des nouvelles options de calibration et de la procédure de
  validation de version.
- Compilations GitHub limitées à deux tâches concurrentes pour borner la mémoire.
- Publications du wiki sérialisées entre `develop` et `main` pour éviter les
  conflits de push vers leur dépôt wiki commun.

## [3.5.0] - 2026-10-01

### ✨ Ajouté — CLI `--run <task>` et JSON Schema `--conf`

- **`--run <task>`** : nouvel argument CLI pour sélectionner une tâche nommée depuis la section `tasks` d'un fichier `--conf`.
  - `mimir --conf config.json --run train` exécute `tasks.train.lua.scripts`.
  - Si la tâche est introuvable, liste les tâches disponibles avec leur description.
  - Compatible avec `--override` : les overrides s'appliquent sur le JSON complet avant la résolution de tâche.
- **Section `tasks`** dans les fichiers `--conf` : chaque tâche est un objet `{ description, lua: { scripts: [...] } }`. Sans `--run`, la section `lua` racine est utilisée (rétrocompatibilité totale).
- **`configs/conf.schema.json`** : schéma JSON Schema draft-07 complet pour les fichiers `--conf`.
  - Toutes les architectures du registre documentées (24 architectures) avec leurs champs exacts issus des exports `exports/architectures_dumpjson/`.
  - Discriminant `if/then/allOf` par `architecture` pour l'autocomplétion IDE.
  - Nouvelles sections : `tasks`, `logging`, `env`.
- **`vae_conv-config.json`** : exemple de conf complet avec `_doc` inline, section `tasks` (train / resume / infer) et toutes les sections documentées.

### 🧾 Documentation

- Scripts Lua: documentation du support `--help` unifie pour les scripts executables (`scripts/` et `examples/`), avec description et options detectees.
- Mise à jour de la documentation `DebugJson` vers **enhanced v1.3**.
- Ajout de la documentation du bloc `framework_state` (runtime/backends, mémoire, registre layers, état modèle) dans :
  - `docs/03-API-Reference/16-Serialization.md`
  - `docs/03-API-Reference/02-Serialization.md`
  - `docs/05-Advanced/02-Debugging.md`
  - `docs/04-Architecture-Internals/15-Serialization-Internals.md`
  - `docs/05-Advanced/04-Source-Code-Map.md`
  - `docs/03-API-Reference/20-Lua-API-Cpp-Mapping.md`
- Stub API Lua (`mimir-api.lua`) aligné sur `DebugJson v1.3`.

## [3.1.0] - 2026-07-01

### ✨ Ajouté — Nouveaux modèles HuggingFace / SDXL

- **`hf_clip_text_encoder_1` / `hf_clip_text_encoder_2`** : encodeurs texte CLIP/OpenCLIP
  exécutables pour checkpoints HuggingFace/PyTorch (SDXL `conditioner.embedders.*`).
- **`hf_sdxl_transformer_block`** : bloc transformer SDXL exécutable (SelfAttention,
  CrossAttention, FeedForward).
- **`hf_vae_decoder`** : décodeur VAE SDXL exécutable (`first_stage_model.decoder`).
- **`external_safetensors_base`** : base non-exécutable reflétant exactement les clés
  d'un checkpoint safetensors externe.
- Scripts d'inférence associés dans `scripts/inferences/` (encodeurs HF, VAE decoder,
  bloc transformer, base safetensors, `ponyxl_ddpm_text2img_hf_vae`) et entraînement
  direct `scripts/training/ponyxl_ddpm_direct_train.lua`.
- Tests unitaires runtime math (`attention`, `conv2d`, `linear`, `norms`) et smoke
  `external_safetensors_base`.

### ✨ Ajouté — Introspection du registre d'architectures

- **`Mimir.Architectures.info([name])`** : lit toutes les infos du registry
  (`name`, `description`, `config`). Sans argument → liste complète ; avec un nom
  → entrée unique (ou `(nil, err)`). Seul accesseur exposant le champ `description`.
  - Binding C++ : `LuaScripting::lua_archInfo`.
- **`Mimir.Architectures.dtypes()`** : liste les dtypes pris en charge par le
  framework (`name`, `aliases`, `bytes`, `kind`), sourcés depuis `src/DType.hpp`.
  - Binding C++ : `LuaScripting::lua_archDtypes`.
- **Outil `scripts/tools/inspect_architectures.lua`** (anciennement
  `scripts/tests/test_list_archi_conf.lua`) : CLI d'inspection avec sorties en
  tableaux colorés — `-a/--show-archs` (archis + dtypes), `-l <arch> -p`
  (paramètres + dtype par défaut), `-d/--dtypes`. Parsing via `scripts/modules/args.lua`.

### ⚡ Performance

- **Fast-paths GPU** Conv2d / Norm / Attention dans les runtimes CUDA et ROCm.

### 🐛 Correctifs

- **Sérialisation** : correction du round-trip dtype (SafeTensors / RawFolder).
- **Stub EmmyLua (`mimir-api.lua`)** : réparation de l'en-tête corrompu — restauration
  de la déclaration `Mimir`, du bloc `@field`, des alias de base
  (`int`/`float`/`bool`/`bytes`/`TokenId`/`TokenIds`) et des alias `ModelType`/
  `ArchitectureName`, étendus avec les nouvelles architectures HF.

### 📝 Documentation

- Réécriture de la documentation (style MDN/Laravel) : introductions, callouts,
  exemples progressifs ; ajout du feedback de calibration.

## [3.0.1] - 2026-05-25

### 🐛 Correctifs

- **`Visualizer`** : ajout du membre manquant `live_ui_inited_` dans `Visualizer.hpp` — corrige une erreur de compilation avec `ENABLE_SFML=ON`.

## [3.0.0] - 2026-05-25

### 🚀 Montée de version majeure — API enrichie, nouveaux templates et robustesse

Cette version consolide les apports des sessions de développement de mai 2026 :
nouveau module I/O, helpers PonyXL-DDPM, robustesse dtype, nouveaux templates Lua
et mise à jour complète du stub EmmyLua.

#### Ajouté

- **`Mimir.IO` — Nouveau module** (I/O images)
  - `Mimir.IO.read_image_rgb_u8(path)` : charge une image et retourne ses pixels RGB u8 sous forme de table Lua `{R, G, B, …}`
  - Alias global `readImageRGBU8(path)` pour les scripts compacts
  - Documentation dédiée : `docs/03-API-Reference/21-IO.md`

- **`Mimir.Model.create_from_config(full_cfg)`**
  - Crée un modèle depuis une config complète (ex. : table `CONF` injectée par le mode CLI `--conf`)
  - Résout l'architecture depuis `full_cfg.architecture` (fallback `type`, puis `model.architecture|model.type`)
  - Retourne `(true, arch)` ou `(false, err)`

- **`Mimir.Model.dtype()` / `Mimir.Model.dtype(name)`**
  - Getter/setter de la préférence dtype du modèle (`"float32"`, `"float16"`, …)
  - Alias `Mimir.model` (lowercase) exposé globalement → `Mimir.model.dtype(...)` fonctionne

- **Helpers PonyXL-DDPM**
  - `ponyxl_ddpm_train_step(batch, lr, opts?)` — step d'entraînement complet
  - `ponyxl_ddpm_validate_step(batch)` — step de validation
  - `ponyxl_ddpm_viz_reconstruct_step(batch)` — forward pour visualisation
  - `ponyxl_ddpm_text2img(prompt, steps?, seed?)` — génération image depuis texte
  - `ponyxl_ddpm_set_vae_scale(scale)` / `ponyxl_ddpm_get_vae_scale()` — facteur d'échelle VAE
  - `ponyxl_ddpm_vae_mu_moments(input)` — moments μ du VAE encoder

- **Nouveaux templates Lua**
  - `scripts/templates/template_pipeline_only.lua` : démarrage minimal via `pipeline_api.lua`, sans args
  - `scripts/templates/template_pipeline_args.lua` : `args.lua` + `pipeline_api.lua` (CLI-driven)

- **Détection hardware au démarrage** (`bin/mimir --help`)
  - Affichage des optimisations CPU disponibles : AVX2, FMA, F16C, BMI2
  - Détection runtime CUDA / ROCm (compilé + disponible)

#### Modifié

- **`scripts/modules/pipeline_api.lua`** — `apply_model_dtype()` robuste
  - Supporte `Mimir.model.dtype` (v3.0+) **et** `Mimir.Model.dtype` (rétrocompat)
  - Gestion silencieuse si ni l'un ni l'autre n'est disponible

- **`scripts/templates/template_new_model.lua`** — mise à jour
  - Architecture par défaut : `vae_conv` (nom canonique du registre)
  - Fallback auto sur `Mimir.Architectures.available()[1]` si l'archi n'est pas trouvée
  - `apply_dtype()` robuste (supporte les deux chemins)
  - `vocab_size` / `max_vocab` harmonisés
  - Tokenizer conditionnel (NLP uniquement, non initialisé pour les archis image)
  - `Mimir.Allocator.get_stats()` à la place de `print_stats()` (non bloquant)

- **`mimir-api.lua`** — stub EmmyLua complet (130+ fonctions, 15 modules)
  - Ajout de toutes les nouvelles fonctions ci-dessus
  - Bump version header : 2.4.0 → **3.0.0**
  - Date de synchronisation : 25 mai 2026

- **Documentation `docs/`** — mise à jour complète
  - `docs/00-INDEX.md` : version 3.0.0, section "Nouveautés v3.0", nouveaux templates
  - `docs/03-API-Reference/00-API-Overview.md` : ajout `Mimir.IO`, `create_from_config`, `dtype`
  - `docs/03-API-Reference/19-Globals.md` : ajout alias `Mimir.model`
  - `docs/03-API-Reference/20-Lua-API-Cpp-Mapping.md` : nouvelles entrées Model + section IO + PonyXL-DDPM
  - `docs/03-API-Reference/21-IO.md` : nouvelle page Mimir.IO

- **`scripts/README.md`** : version 3.0.0, date 25 mai 2026, référence aux 2 nouveaux templates

## [2.4.0] - 2026-03-15

### 🧹 Release de maintenance

#### Modifié
- Documentation mise à jour (révision 2026-03-15)
- Scripts Lua : README corrigé + ajout d'un smoke test de sérialisation
- Stub IDE Lua (`mimir-api.lua`) synchronisé et bump de version
- Template Lua mis à jour (v2.4)

#### Packaging
- Ajout/renforcement des exclusions `.gitignore` (artefacts build/logs/datasets/secrets, exports Insomnia, graphs)

## [2.3.0] - 2025-12-28

### 💾 Sérialisation Moderne

Cette version introduit un système de sérialisation complet et retire tout le code legacy.

#### Ajouté
- **Module Mimir::Serialization** - API unifiée de sérialisation
  - `Serialization::save()` et `Serialization::load()` avec détection automatique de format
  - Support de 3 formats: SafeTensors, RawFolder, DebugJson
  
- **SafeTensors Format** - Format production compatible HuggingFace
  - Conformité spec SafeTensors 0.3.0
  - Header JSON + données binaires contiguës little-endian
  - Section `__metadata__` avec version Mímir et timestamps
  - Interopérable avec PyTorch/TensorFlow
  - Performance: 850 MB/s write, 1200 MB/s read

- **RawFolder Format** - Format debug avec validation d'intégrité
  - Structure de répertoires organisée (manifest, model, tensors, tokenizer)
  - Checksums SHA256 pour chaque tensor
  - Validation optionnelle à la lecture
  - Git-friendly pour versioning

- **DebugJson Format** - Format inspection avec statistiques
  - Export JSON complet lisible
  - Statistiques par tensor (min, max, mean, std)
  - Échantillons de valeurs pour analyse rapide

- **Documentation**
  - `docs/SAVE_LOAD.md` - Guide complet (480+ lignes)
  - `SERIALIZATION_COMPLETE.md` - Résumé technique
  - Exemples Lua dans tous les scripts

#### Modifié
- **Architecture des poids** - Optimisation mémoire et performance
  - Passage de `params` (un tenseur par paramètre) à `layer_weight_blocks` (un tenseur par couche)
  - Réduction de la fragmentation mémoire
  - Meilleure localité cache pour opérations SIMD
  - Accès optimisé via `layer.weight_block->getData()`

- **SafeTensorsReader** - Migration vers layer_weight_blocks
  - `apply_tensors_to_model()` utilise `layer.weight_block` au lieu de `params[]`
  - Accès via `model.getLayers()` au lieu de `getMutableParams()`

#### Retiré
- **Code legacy complet** - ~300 lignes supprimées
  - ❌ Structure `std::vector<tensor> params` obsolète
  - ❌ Méthode `getMutableParams()` accessor
  - ❌ `saveLayersStructure()` / `loadLayersStructure()`
  - ❌ `saveEmbeddings()` / `loadEmbeddings()`
  - ❌ `saveParamsData()` / `loadParamsData()`
  - ❌ `Model::saveCheckpoint()` legacy
  - ❌ Blocs `#ifdef MIMIR_ENABLE_LEGACY_PARAMS`

- **Fonctions obsolètes** - Remplacées par stubs avec messages de dépréciation
  - `updateWeightsWithNoise()` → utiliser `optimizerStep()`
  - `forward(std::vector<uint8_t>&)` → utiliser `forwardPass()`
  - `setOutputTarget()` → obsolète
  - `applyParamUpdate()` → utiliser `optimizerStep()`

#### Documentation
- **[LEGACY_CLEANUP_COMPLETE.md](LEGACY_CLEANUP_COMPLETE.md)** - Détails du cleanup
- **[docs/SAVE_LOAD.md](docs/SAVE_LOAD.md)** - Guide sérialisation
- **[TECHNICAL_STATUS.md](TECHNICAL_STATUS.md)** - État technique v2.3

### 🔀 Multi-Input / Branch Support

Cette version ajoute le support complet des architectures avec multi-inputs, branches et skip connections.

#### Ajouté
- **TensorStore System** - Système de routage de tensors nommés
  - `std::unordered_map<std::string, std::vector<float>>` pour le stockage
  - Méthodes `getTensor()`, `storeTensor()`, `getAvailableTensors()`, `clearTensorStore()`
  - Gestion d'erreurs explicite avec liste des tensors disponibles
  - Support move semantics pour optimisation mémoire

- **Layer I/O Configuration**
  - Nouveau champs `Layer.inputs` (vector<string>) : noms des tensors d'entrée
  - Nouveau champs `Layer.output` (string) : nom du tensor de sortie
  - Défauts: `inputs={}` → `{"x"}`, `output=""` → `"x"`
  - Rétrocompatibilité totale avec pipelines séquentiels existants

- **API Lua: `model.set_layer_io()`**
  ```lua
  model.set_layer_io("layer_name", {"input1", "input2"}, "output_name")
  ```
  - Configuration dynamique des entrées/sorties des layers
  - Validation existence du layer
  - Support tables Lua pour entrées multiples

- **Operations Multi-Input Complètes**
  - ✅ **Add**: Addition élément par élément de 2 tensors
  - ✅ **Multiply**: Multiplication élément par élément de 2 tensors
  - ✅ **Concat**: Concaténation de N tensors selon axis
  - ✅ **MatMul**: Multiplication matricielle de 2 matrices (M×K, K×N → M×N)
  - ✅ **Split**: Séparation 1 tensor → N outputs (`name_0`, `name_1`, ...)

- **LayerOps Extensions**
  - Surcharge `split_forward()` avec `vector<int> split_sizes` pour tailles explicites
  - Validation des dimensions et tailles dans toutes les operations
  - Messages d'erreur détaillés avec contexte

- **Documentation**
  - `docs/MULTI_INPUT_SUPPORT.md` - Guide complet (50+ exemples)
  - `scripts/tests/test_api_simple.lua` - Test API fonctionnel ✅
  - `scripts/tests/test_branches.lua` - Suite de tests complète

#### Cas d'Usage

**Residual Connections (ResNet, etc.)**
```lua
model.set_layer_io("conv1", {"x"}, "skip")
model.set_layer_io("add_res", {"x", "skip"}, "x")
```

**Feature Fusion (U-Net, FPN)**
```lua
model.set_layer_io("conv_a", {"x"}, "feat_a")
model.set_layer_io("conv_b", {"x"}, "feat_b")
model.set_layer_io("concat", {"feat_a", "feat_b"}, "fused")
```

#### Travail Futur

- ⏳ **Backward Pass**: Routage gradients pour multi-input ops
- ⏳ **Graph Optimization**: Tri topologique, parallélisation branches
- ⏳ **Lua Extensions**: `get_available_tensors()`, `get_tensor(name)`

## [2.1.0] - 2025-12-27

### 🗂️ Organisation et Qualité

Cette version se concentre sur l'organisation du projet et l'amélioration de la qualité de la documentation.

#### Ajouté
- **`scripts/README.md`** - Documentation complète de l'organisation des scripts
  - Structure organisée en 6 catégories
  - 40 scripts classés: demos (10), examples (5), tests (19), benchmarks (2), training (3), templates (1)
  - Documentation d'utilisation avec exemples
  - Statistiques et références croisées

- **`tools/README.md`** - Documentation des outils de validation
  - 3 scripts bash pour validation API
  - Exemples d'utilisation complets
  - Intégration CI/CD possible

#### Modifié
- **Structure `scripts/`** - Réorganisation complète
  - Création de sous-dossiers thématiques: `demos/`, `examples/`, `tests/`, `benchmarks/`, `training/`, `templates/`
  - Migration de 40 scripts vers leur catégorie appropriée
  - Navigation facilitée et découverte intuitive

- **Documentation** - Corrections et synchronisation
  - 33 corrections de liens cassés (8 fichiers)
  - Mise à jour architecture count: 8 → 9 (ajout Flux)
  - Mise à jour fonction count: 117 → 114 (compte exact)
  - Table des matières complétée dans API Complete

#### Synchronisation
- **API Lua** - Synchronisation 100% confirmée
  - `mimir-api.lua` ↔ `src/LuaScripting.cpp`
  - 114 fonctions validées sur 13 modules
  - Scripts de validation créés et testés
  - EmmyLua annotations complètes

#### Qualité
- 📁 **Meilleure organisation** des ressources projet
- 🔗 **Documentation navigable** sans liens cassés
- ✅ **Validation automatisée** de la synchronisation API
- 📊 **Statistiques exactes** partout dans la documentation

## [2.0.0-doc] - 2025-12-22

### 📚 Documentation - Mise à Jour Majeure de l'API Lua

Cette version apporte une refonte complète de la documentation de l'API Lua avec une couverture exhaustive de toutes les fonctionnalités du framework.

#### Ajouté
- **`docs/LUA_API_COMPLETE.md`** (50K+) - Documentation exhaustive complète
  - 11 modules documentés en détail (80+ fonctions)
  - Architecture générale et workflow standard
  - 13 sections couvrant tous les aspects de l'API
  - 4 exemples complets annotés et testables
  - Structures de données (HtopMetrics, VizMetrics, MemoryStats, etc.)
  - Notes de performance et optimisations
  - Best practices et guide de debugging
  
- **`docs/LUA_API_REFERENCE_QUICK.md`** (15K) - Référence rapide pratique
  - Tables de référence pour tous les modules
  - Workflow standard simplifié
  - Fonctions essentielles (quick reference)
  - Configurations types pour chaque architecture
  - 8 exemples rapides prêts à l'emploi
  - Section debugging et gestion d'erreurs
  
- **`docs/API_UPDATE_SUMMARY.md`** - Résumé technique de la mise à jour
  - Statistiques avant/après détaillées
  - Liste exhaustive des changements
  - Impact sur le développement
  - Vérifications de cohérence

#### Modifié
- **`mimir-api.lua`** - Stub IDE complètement refondu (40 → 80+ fonctions)
  - 6 nouveaux modules: `memory`, `guard`, `allocator`, `htop`, `viz`, globales
  - Module `tokenizer` étendu: 6 → 20+ fonctions
  - Annotations EmmyLua complètes (`@class`, `@param`, `@return`, `@alias`)
  - Documentation inline pour autocomplétion IDE
  - Types stricts pour validation statique
  
- **`docs/INDEX.md`** - Section "API Lua" restructurée
  - Table complète des 11 modules
  - Compteurs de fonctions actualisés
  - Liens vers nouvelles documentations
  
- **`README.md`** - Section "Documentation" refaite
  - Nouvelle sous-section "API Lua Complète"
  - Description détaillée des 11 modules
  - Liens vers documentation exhaustive
  - Statistiques: 80+ fonctions organisées

#### Modules Nouvellement Documentés

**memory (AdvancedRAMManager)** - 6 fonctions:
- `config(cfg)`, `get_stats()`, `print_stats()`, `clear()`, `get_usage()`, `set_limit(limit_mb)`

**guard (MemoryGuard)** - 4 fonctions:
- `set_limit(limit_mb)`, `get_stats()`, `print_stats()`, `reset()`

**allocator (DynamicTensorAllocator)** - 3 fonctions:
- `configure(config)`, `print_stats()`, `get_stats()`

**htop (HtopDisplay)** - 5 fonctions:
- `create(config)`, `enable(enabled)`, `update(metrics)`, `render()`, `clear()`

**viz (Visualizer SFML)** - 11 fonctions:
- `create(title, width, height)`, `initialize()`, `is_open()`, `process_events()`, `update()`
- `add_image(pixels, w, h, ch)`, `update_metrics(metrics)`, `add_loss_point(loss)`
- `clear()`, `set_enabled(enabled)`, `save_loss_history(path)`

**Fonctions globales** - 3 fonctions:
- `log(message)`, `read_json(filepath)`, `write_json(filepath, data)`

**tokenizer (extension)** - 14 nouvelles fonctions:
- Vocabulaire: `add_token()`, `ensure_vocab_from_text()`, `tokenize_ensure()`
- IDs spéciaux: `pad_id()`, `unk_id()`, `seq_id()`, `mod_id()`, `mag_id()`, `get_token_by_id()`
- BPE: `learn_bpe()`, `tokenize_bpe()`, `set_max_length()`, `pad_sequence()`, `batch_tokenize()`
- Analyse: `print_stats()`, `get_frequencies()`, `analyze_text()`, `extract_keywords()`

#### Impact Mesurable

| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| Fonctions documentées | 40 | 80+ | +100% |
| Modules | 5 | 11 | +120% |
| Documentation (lignes) | ~15K | ~65K+ | +333% |
| Exemples complets | 2 | 8 | +300% |
| Structures données | 2 | 7 | +250% |

#### Bénéfices
- 🎯 **Autocomplétion IDE complète** via annotations EmmyLua
- 📚 **Documentation à 3 niveaux**: Quick reference, Complète, Legacy
- 🔍 **Découverte facilitée** de l'API via exploration IDE
- ⏱️ **Gain de temps** pour les nouveaux utilisateurs
- 🤝 **Collaboration améliorée** avec documentation standardisée
- ✅ **Cohérence garantie** avec implémentation C++ (vérifiée)

#### Vérifications
- ✅ Cohérence complète avec `LuaScripting.cpp` / `LuaScripting.hpp`
- ✅ Toutes les fonctions `lua_setfield()` documentées
- ✅ Signatures vérifiées contre implémentations
- ✅ Retours `(ok, err)` standardisés
- ✅ Types Lua corrects (table, string, number, boolean)
- ✅ Tous les exemples testables

---

## [2.0.0] - 2025-12-19

### 🎯 Philosophy

**Mímir v2.0 est un framework CPU-only** conçu pour démocratiser le deep learning en le rendant accessible à tous sans nécessiter de GPU coûteux.

- ✅ **CPU-only par design**: Pas de CUDA, pas de ROCm, juste des CPU modernes
- 💰 **Accessible**: Fonctionne sur tout ordinateur avec un CPU moderne (>2013)
- ⚡ **Performant**: Optimisations AVX2/FMA/F16C pour exploiter le CPU au maximum
- 🎯 **Mission**: Permettre à quiconque de créer et entraîner des IA sans barrière financière

### 🎉 Ajouté

#### Architectures Modernes
- **8 architectures state-of-the-art** prêtes à l'emploi dans `ModelArchitectures` namespace
  - UNet: Encoder-Decoder avec skip connections pour segmentation
  - VAE: Variational Autoencoder pour génération et compression
  - ViT: Vision Transformer avec patch embedding
  - GAN: Generator + Discriminator (StyleGAN-inspired)
  - Diffusion: DDPM pour génération haute qualité
  - Transformer: GPT-style avec causal masking
  - ResNet: ResNet-50 avec bottleneck blocks
  - MobileNet: MobileNetV2 avec inverted residuals

#### API Lua Complète
- **60+ fonctions Lua** exposées dans 4 tables globales:
  - `model.*` (18 fonctions): Gestion complète du modèle
  - `architectures.*` (8 fonctions): Construction des 8 architectures
  - `layers.*` (8 fonctions): Opérations de layers
  - `tokenizer.*` (6 fonctions): Tokenization avec save/load
- Configuration des architectures via tables Lua
- Conversions automatiques Lua ↔ C++ (tables, configs)
- Gestion d'erreurs robuste avec retours appropriés

#### Optimisations Hardware
- Détection automatique des capacités CPU (AVX2, FMA, F16C, BMI2)
- Dispatch dynamique hardware/software pour chaque opération
- **Layer Operations optimisées**:
  - `computeConv2D()`: FMA saturé (3 accumulateurs), collapse(3) OpenMP
  - `computeLinear()`: matmul_avx2 avec FMA
  - `computeMaxPool2D()`: AVX2 avec horizontal max optimisé
  - `computeAvgPool2D()`: AVX2 avec somme vectorielle
  - `computeActivation()`: GELU, ReLU, Softmax avec SIMD
  - `computeBatchNorm()`: AVX2 pour mean/variance
  - `computeLayerNorm()`: Normalisation vectorisée
  - `computeAttention()`: Multi-head avec matmul_transpose_avx2
- Configuration globale: `Model::setHardwareAcceleration(bool)`
- Méthodes de détection: `hasAVX2()`, `hasFMA()`, `hasF16C()`, `hasBMI2()`

#### Interface Utilisateur
- Option `--lua` dans le main pour exécuter des scripts
- Option `--demo` pour tester les architectures
- Option `--config` pour charger depuis JSON
- Scripts d'exemple:
  - `test_lua_api.lua`: Tests complets (10 tests)
  - `example_simple.lua`: Exemple minimal
  - `example_gpt.lua`: Génération de texte
  - `example_training.lua`: Boucle d'entraînement

#### Documentation
- **README.md** complètement refait (800+ lignes)
- **docs/README.md**: Index consolidé de la documentation
- **docs/LUA_API.md**: Référence API Lua complète (550+ lignes)
- **docs/MODEL_ARCHITECTURES.md**: Détails techniques des architectures
- **docs/LAYER_OPERATIONS.md**: Documentation des opérations optimisées
- Exemples de code pour chaque architecture
- Benchmarks de performance
- Guide de contribution

### 🔄 Modifié

#### Model Class
- Refonte complète avec méthodes statiques pour layer operations
- Ajout de `LayerParams` structure pour configuration
- Support du dispatch hardware/software
- Amélioration de `allocateParams()` et `initializeWeights()`
- Méthodes `totalParamCount()` pour statistiques

#### LuaScripting
- Refonte complète de `registerAPI()` avec structure en tables
- ~800 lignes d'implémentations ajoutées
- Parsing de configurations Lua complexes
- Error handling cohérent
- Helpers pour conversions JSON ↔ Lua

#### Main
- Support de l'option `--lua` pour scripts
- Amélioration de l'aide (`--help`)
- Affichage des capacités hardware au démarrage
- Interface utilisateur améliorée

#### Makefile
- Optimisations de compilation (-mavx2 -mfma -mf16c -mbmi2)
- Support de Lua 5.3
- Binary optimisé: 1.4MB

### ❌ Supprimé

- Anciens modèles enfants de la classe Model (EncoderModel, DecoderModel, etc.)
- Fichiers de documentation obsolètes:
  - API_CPP.md
  - CHECKPOINT_FORMAT.md
  - FORWARD_BACKWARD_PASS.md
  - OPTIMIZATIONS.md
  - STATUS.md
  - TOKENIZER_TRAINING.md

### 🐛 Corrigé

- Gestion mémoire améliorée dans les opérations SIMD
- Alignement correct des données pour AVX2
- Conversions types Lua ↔ C++ robustes
- Gestion des erreurs dans le parsing de configs

### 🚀 Performance

- **2.5-4× speedup** sur training loop complet
- **1.5× speedup** vs PyTorch CPU sur Conv2D
- **1.3× speedup** vs PyTorch CPU sur MatMul
- **50% réduction mémoire** avec FP16 storage

### 📊 Statistiques

- **~5000 lignes** de code ajoutées
- **60+ fonctions** Lua exposées
- **8 architectures** modernes
- **9 opérations** de layers optimisées
- **4 méthodes** de détection hardware
- **10 scripts** d'exemple
- **800+ lignes** de documentation README
- **1400+ lignes** de documentation API

---

## [1.0.0] - 2025-11-15

### Ajouté
- Classe Model de base avec tensors
- Support OpenMP et AVX2
- Tokenizer BPE
- Visualiseur SFML
- Système Autograd basique
- Format SafeTensors
- 7 architectures de base (Encoder, Decoder, etc.)
- API Lua basique
- Documentation initiale

### Notes
Version initiale du framework avec fonctionnalités de base.

---

## Légende

- 🎉 **Ajouté**: Nouvelles fonctionnalités
- 🔄 **Modifié**: Changements dans des fonctionnalités existantes
- ❌ **Supprimé**: Fonctionnalités supprimées
- 🐛 **Corrigé**: Corrections de bugs
- 🚀 **Performance**: Améliorations de performance
- 📊 **Statistiques**: Métriques et chiffres

---

## Liens

- [Repository](https://github.com/brit45/mimir-framework)
- [Documentation](docs/README.md)
- [Issues](https://github.com/brit45/mimir-framework/issues)
