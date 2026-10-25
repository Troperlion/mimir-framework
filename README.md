# Mímir AI Engine

![logo](./logo.png)

## Framework Philosophy

**[Framework Philosophy](./docs/00-Framework-Philosophy.md)** — pourquoi C++, Lua, JSON, CPU-first, Registry, Planner, runtime independant et architectures compilees.

**[Scripting Contract](./docs/03-API-Reference/00-Scripting-Contract.md)** — specification officielle que tous les bridges doivent implementer.

| Branche | unit-tests | publish-wiki |
| --- | --- | --- |
| `main` | [![unit-tests main](https://github.com/brit45/mimir-framework/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/brit45/mimir-framework/actions/workflows/unit-tests.yml) | [![publish-wiki main](https://github.com/brit45/mimir-framework/actions/workflows/wiki.yml/badge.svg?branch=main)](https://github.com/brit45/mimir-framework/actions/workflows/wiki.yml) |
| `develop` | [![unit-tests develop](https://github.com/brit45/mimir-framework/actions/workflows/unit-tests.yml/badge.svg?branch=develop)](https://github.com/brit45/mimir-framework/actions/workflows/unit-tests.yml) | [![publish-wiki develop](https://github.com/brit45/mimir-framework/actions/workflows/wiki.yml/badge.svg?branch=develop)](https://github.com/brit45/mimir-framework/actions/workflows/wiki.yml) |

Version engine : **3.6.0**
Révision documentation : **2026-10-24**

Consultez d'abord l'[état réel du projet](./docs/00-PROJECT-STATUS.md) : cette
page distingue les chemins stables, partiels, expérimentaux, placeholders et
matériels testés dans le checkout courant.

**Mímir est un AI Engine C++ de conception, d'entraînement et d'analyse de systèmes IA, pilotable par Lua ou JSON, avec runtime, mémoire, dataset, visualisation et sérialisation intégrés, dans une approche CPU-first orientée recherche et expérimentation.**

---

## Vitrine (FR)

Mímir est un **AI Engine de recherche IA CPU-first** où tu conçois les architectures en **C++**, puis tu pilotes entraînement/inférence via **Lua** ou **JSON (`--conf`)**.

Points clés:

- AI engine de graphes autonome (runtime + mémoire + planner),
- registre central d'architectures (vision, NLP, diffusion),
- outillage intégré (datasets, Viz SFML/Qt/GTK/Web, sérialisation, benchmarks),
- approche orientée expérimentation et inspection interne des modèles.

---

## Showcase (EN)

Mímir is a **CPU-first C++ AI Engine**.
You define model architectures in C++, then orchestrate training/inference with Lua scripts or JSON-driven runs.

Core strengths:

- autonomous AI runtime and graph execution core,
- architecture registry (vision, NLP, diffusion),
- built-in tooling (datasets, visualization, serialization, benchmarking),
- research-focused workflow for fast experimentation and deep inspection.

---

## Version technique (FR)

### Positionnement

Mímir n'est pas un clone de PyTorch ou TensorFlow.

Le projet est pensé comme un **AI Engine autonome** :

- architecture des modèles en **C++**,
- orchestration en **Lua**,
- exécution reproductible via **JSON**,
- runtime et outillage intégrés dans un seul engine.

Philosophie:

```text
Architecture
  ↓
C++
  ↓
Registry
  ↓
Runtime
  ↓
Lua / JSON
  ↓
Entraînement / Inférence / Analyse
```

---

### Architecture générale

```text
            Lua / JSON / CLI
                    │
                    ▼
            API Engine Mímir
                    │
                    ▼
        Registry d'architectures
                    │
                    ▼
               Runtime IA
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      CPU         CUDA        ROCm
```

Schéma visuel (SVG):

![Vue d'ensemble du framework](./docs/graphs/00_framework_overview.svg)

![Exécution CPU-first](./docs/graphs/15_cpu_first_execution.svg)

---

### Quick Start (5 minutes)

Guide complet : **[GET STARTED](./docs/01-Getting-Started/00-GET-STARTED.md)**

```bash
# 1) Installer les dépendances et configurer la compilation Linux
./config.sh --gui NONE

# 2) Compiler le profil sélectionné
cmake --build build --parallel 2

# 3) Vérifier le binaire
./bin/mimir --help

# 4) Smoke test
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua

# 5) Premier modèle
./bin/mimir --lua scripts/templates/template_new_model.lua
```

#### Variante Docker Compose (Alpine)

Pour démarrer sans installer les dépendances C++ localement, construis l'image
puis lance un script depuis le dépôt monté :

```bash
docker compose build
MIMIR_UID="$(id -u)" MIMIR_GID="$(id -g)" \
  docker compose run --rm mimir \
  --lua scripts/templates/template_new_model.lua
```

Autre possibilité, sélectionner le script avec `SCRIPT` et passer ses options
Lua après `--` :

```bash
MIMIR_UID="$(id -u)" MIMIR_GID="$(id -g)" \
SCRIPT=scripts/benchmarks/benchmark_nms.lua \
  docker compose run --rm mimir -- --quick
```

Le conteneur embarque uniquement le binaire et ses bibliothèques ; scripts,
configurations, données et sorties sont dans le dépôt monté. Pour le détail
des options Docker et des exemples, voir [Docker (CPU/headless)](#docker-cpuheadless)
plus bas.

Notes rapides sur `config.sh`:

- Script Linux (Debian/Ubuntu) qui installe les dependances et configure CMake dans `build/`.
- Options : `--gui QT`, `--build-type Debug`, `--enable MIMIR_ENABLE_TESTS`,
  `--disable ENABLE_VULKAN`, `--disable ENABLE_OPENCL` et `--non-interactive`.
- `--skip-system-deps` omet apt/sudo seulement lorsque les dépendances sont déjà installées.
- Voir la doc detaillee: `docs/01-Getting-Started/02-Installation.md`.

Notes scripts Lua (cross-plateforme):

- Les scripts doivent utiliser `scripts/modules/fs.lua` pour les opérations fichiers/répertoires (`mkdir_p`, `join`, `dirname`, `list_dir`, `file_exists`).
- Éviter `os.execute("mkdir ...")`, `io.popen("ls ...")`, `test -d`, etc. dans les scripts métier.
- Les appels shell restent réservés aux besoins process externes (ex: ouvrir navigateur, lancer un outil), pas au filesystem applicatif.

#### Modules Lua réutilisables

Les fichiers de `scripts/modules/` sont chargés avec `dofile`; ils ne sont pas
injectés dans `Mimir` et ne sont pas des bindings C++. Le stub
[`mimir-api.lua`](./mimir-api.lua) décrit à la fois l'API native et les contrats
publics de ces helpers afin que LuaLS/EmmyLua puisse les compléter.

| Module | État | Rôle et contrainte principale |
|---|---|---|
| `args.lua` | actif | Parse la table globale `arg`, normalise `--foo-bar`/`--foo_bar`, applique les overrides et peut initialiser Htop/Viz. Il n'existe pas de `Mimir.Args`. |
| `fs.lua` | actif | Helpers filesystem portables; certaines opérations passent par les commandes natives de l'OS. |
| `checkpoint_resume.lua` | actif, ciblé | Résout un checkpoint `raw_folder` direct ou le dernier `epoch_N`; ne charge pas le modèle. |
| `base_tokenizer.lua` | actif | Charge ou crée le tokenizer partagé via `Mimir.Tokenizer`; exige le binaire Mímir. |
| `causal_lm_tokenizer.lua` | actif | Vérifie vocabulaire, tokens spéciaux et capacité avant un run causal LM. |
| `pipeline_api.lua` | expérimental | Façade registry-first et constructeurs historiques; la disponibilité effective dépend du registre compilé. `pipeline.lua` est son alias de compatibilité. |
| `mpk.lua` / `mpk_layers.lua` | actif, outillage | Lit/écrit/compile MPK et normalise les types de couches; le format écrit est `binary-v4`, les versions historiques sont lecture seule. |
| `help_cli.lua` | actif | Génère l'aide d'un script à partir de ses commentaires et de ses options détectables. |
| `api_ws_server.lua` | expérimental | Serveur HTTP/WebSocket autonome; requiert LuaSocket, et LuaSec lorsque TLS est demandé. |

Exemple minimal, lancé depuis la racine du dépôt :

```lua
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local FS = dofile(ROOTWORK.."/scripts/modules/fs.lua")

local opts = Args.parse(arg)
local out_dir = Args.get_str(opts, "out-dir", "checkpoint/demo")
assert(FS.mkdir_p(out_dir), "création du répertoire impossible")
```

```bash
./bin/mimir --lua mon_script.lua -- --out-dir checkpoint/essai
```

`args.lua` effectue aussi un premier parse lors de son chargement pour fournir
les helpers `opt_*`; appeler ensuite `Args.parse(arg)` est utile seulement si le
script a besoin de la table complète et des arguments positionnels.

### Docker (CPU/headless)

L'image Docker générique est basée sur Alpine Linux et compile Mímir sans
dépendre des instructions CPU de la machine de build. Elle active Lua, OpenMP,
LZ4 et FFmpeg, mais désactive SFML, CUDA, ROCm, Vulkan, OpenCL et les bridges
externes.

L'image runtime contient le binaire Mímir, ses bibliothèques et le lanceur
`run_mimir.sh` ; scripts, configurations, données et sorties sont fournis par
le montage du répertoire du projet dans le service Compose. Tous les lancements
du conteneur passent par ce lanceur afin d'appliquer la configuration runtime.

```bash
docker compose build
docker compose run --rm mimir --version

# Passer le script et ses options directement sur la ligne de commande.
docker compose run --rm mimir \
  --lua scripts/templates/template_new_model.lua

# Ou sélectionner le script avec SCRIPT.
SCRIPT=scripts/templates/template_new_model.lua \
  docker compose run --rm mimir

# Les arguments Lua sont transmis après le séparateur --.
SCRIPT=scripts/benchmarks/benchmark_nms.lua \
  docker compose run --rm mimir -- --quick
```

Compose monte le dépôt dans `/workspace` afin que le binaire puisse accéder
aux scripts et à leurs modules. Les UID/GID par défaut du processus sont 1000 ;
définir `MIMIR_UID` et `MIMIR_GID` pour que les fichiers produits appartiennent
à l'utilisateur hôte approprié.

La configuration JSON peut remplacer `OMP_NUM_THREADS` grâce à sa section
`env`. Les backends GPU nécessitent des images spécialisées avec leurs SDK et
un accès explicite au matériel; ce Dockerfile ne revendique pas ce support.

---

### Modes d'utilisation

Mímir expose trois modes principaux:

1. **C++** : définition d'architectures, noyau runtime et couches bas niveau.
2. **Lua** : pilotage des runs (train/inference/tools).
3. **JSON (`--conf`)** : exécution reproductible et scénarios paramétrés.

La [matrice de maturité](./docs/00-PROJECT-STATUS.md) indique pour chaque
sous-système s'il est actif, partiel, expérimental, placeholder ou en
construction. Elle doit être consultée avant toute revendication de support
GPU, FPGA, modèle externe ou bridge de langage.

Schéma des flux d'exécution (SVG):

![Workflow Lua JSON CLI](./docs/graphs/04_workflow_lua_json_cli.svg)

### CLI essentielle

```bash
./bin/mimir --lua <script.lua>
./bin/mimir --conf <config.json>
./bin/mimir --conf <config.json> --run <task>
./bin/mimir --conf <config.json> --override path.to.key=value
```

#### Mode `--conf`

Charge un fichier JSON et exécute la section `lua.scripts` séquentiellement.

Le runtime injecte automatiquement dans chaque script Lua :

| Variable Lua | Contenu |
|---|---|
| `CONF` | Table Lua contenant l'intégralité du JSON (après overrides) |
| `CONF_PATH` | Chemin absolu du fichier de conf |
| `CONF_DIR` | Répertoire du fichier de conf |
| `arg` | Arguments de script (`lua.scripts[i].args`) |

#### Mode `--run <task>` (avec `--conf`)

Sélectionne une tâche nommée définie dans la section `tasks` du fichier de conf.
Sans `--run`, la section `lua` racine est utilisée (comportement par défaut).

```bash
# Tâche par défaut (section lua racine)
./bin/mimir --conf config.json

# Tâche nommée
./bin/mimir --conf config.json --run train
./bin/mimir --conf config.json --run infer

# Tâche + override à la volée
./bin/mimir --conf config.json --run train --override training.learning_rate=5e-6
```

Si la tâche est introuvable, mimir liste les tâches disponibles avec leur description.

#### Exemple de config `--conf` avec tâches

```json
{
  "lua": {
    "scripts": ["scripts/training/train_vae_conv.lua"]
  },
  "tasks": {
    "train": {
      "description": "Entraînement complet",
      "lua": { "scripts": ["scripts/training/train_vae_conv.lua"] }
    },
    "infer": {
      "description": "Génération d'images",
      "lua": {
        "scripts": [{
          "script": "scripts/inferences/infer_vae_conv.lua",
          "args": ["--num-samples", "16"]
        }]
      }
    }
  },
  "model": {
    "architecture": "vae_conv",
    "image_w": 512, "image_h": 512, "image_c": 3,
    "latent_h": 64, "latent_w": 64, "latent_c": 128,
    "base_channels": 8
  },
  "training": {
    "num_epochs": 35,
    "learning_rate": 1e-5,
    "optimizer": "adamw"
  },
  "visualization": { "enabled": true }
}
```

#### JSON Schema (`--conf`)

Le fichier `configs/conf.schema.json` documente toutes les sections acceptées :

| Section | Rôle |
|---|---|
| `lua` | Scripts Lua par défaut (sans `--run`) |
| `tasks` | Tâches nommées sélectionnables via `--run` |
| `model` | Config architecture — `architecture` détermine les champs valides |
| `training` | Hyperparamètres d'entraînement |
| `dataset` | Chemins et paramètres du dataset |
| `inference` | Paramètres de génération |
| `visualization` | Fenêtre SFML (lue par le runtime C++) |
| `logging` | Affichage console (`show_htop_display`, etc.) |
| `env` | Variables appliquées au processus avant le run; elles remplacent les valeurs héritées du shell |

---

### Workflow modèle (v3.0+)

Cycle moderne recommandé:

1. `Mimir.Model.create(name, cfg)`
2. `Mimir.Model.allocate_params()`
3. `Mimir.Model.init_weights(...)` ou `Mimir.Serialization.load(...)`
4. `Mimir.Model.forward(...)`

`Model.build()` est conservé pour compatibilité legacy, mais **n'est plus requis** dans le flux moderne.

---

### Modèles disponibles

Le registre inclut notamment:

- MLP
- VAE, VAE Conv
- UNet
- Vision Transformer
- ResNet
- VGG16 / VGG19
- MobileNet
- GAN Latent
- Transformer
- Diffusion / Conditional Diffusion
- SDXL
- CLIP
- modèles HuggingFace
- modèles externes SafeTensors

---

### Runtime engine et performances

Capacités clés:

- OpenMP
- SIMD / AVX2 / FMA
- détection matérielle au démarrage
- dispatch CPU optimisé
- backends CUDA / ROCm (optionnels, en évolution)
- accélération Vulkan/OpenCL sur certains chemins

L'engine est **CPU-first** et optimisé pour les processeurs modernes.

Les chemins CPU réduisent les copies et allocations du backward, des convolutions
et de l’attention. Vulkan/OpenCL disposent aussi de calculs backward natifs,
sous réserve des formes et budgets acceptés. Voir la
[documentation performance](docs/05-Advanced/01-Performance.md) et les
[benchmarks Colab](docs/05-Advanced/07-Colab-Benchmarks.md) : profils partagés,
mesures distinctes des runtimes GPU, contrôles numériques et mémoire hôte.


---

### Gestion mémoire de l'engine

Mímir intègre son propre système mémoire engine:

- `MemoryGuard`
- `DynamicTensorAllocator`
- spill disque
- compression
- lazy allocation
- éviction LRU

Objectifs:

- limiter les OOM,
- contrôler finement la RAM,
- exécuter des charges dépassant la RAM disponible via offload.

---

### Planner et exécution de graphe engine

Le planner de graphes prend en charge:

- fusion d'opérations,
- optimisation des durées de vie des tenseurs,
- planification d'allocation des buffers,
- réutilisation mémoire,
- préparation du runtime.

Schémas internes (SVG):

![Runtime graphe](./docs/graphs/07_graph_runtime.svg)

![Sous-système mémoire](./docs/graphs/08_memory_subsystem.svg)

---

### Dataset, visualisation, sérialisation

### Dataset

Loader multimodal intégré (image/texte actuellement, audio/vidéo prévu), avec:

- association automatique des modalités,
- cache dataset,
- hash SHA256,
- lazy loading,
- gestion RAM.

### Visualisation

Interface temps réel (backend SFML, Qt, GTK ou Web choisi à la compilation) pour:

- loss et métriques,
- reconstructions et activations,
- heatmaps et latent,
- progression d'entraînement.

Fonctions Viz actuelles:

- rendu `Blocks / Layers` en mode `HEATMAP` ou `REEL` (`M`),
- palettes heatmap cyclables au clavier (`K`) : `CLASSIC`, `TURBO`, `INFERNO`, `VIRIDIS`,
- palette appliquée à tous les tips de `Blocks / Layers`, sauf les layers `Identity`, avec mise à jour immédiate,
- filtrage des tenseurs de packing final (`out_concat` / `out_pack`) pour éviter les faux aperçus,
- ordre visuel forcé dans `Outputs`: `recon` affiché juste avant `diff/resdiff` quand les deux existent,
- barre de progression globale (epochs) + barre dédiée batch courant (sous la globale),
- couleur de la barre batch pilotée par `batch_time_ms` (rapide=vert, plus lent=orange/rouge),
- graphe nodal interactif du modèle (`N`) avec export PNG complet (`X`), et export de snapshots configurables (`E`),
- vue du graphe du modèle et du planner dans HTOP (`G`).

Les connexions du graphe sont routées en lignes orthogonales distinctes :
les embranchements ont des rails dédiés, les connexions simples alignées
utilisent un segment, et les trajets ne passent pas sous les cartes ni ne se
coupent. Les images PNG du graphe sont rendues par tuiles afin de conserver la
topologie entière sans dépasser la limite de texture de rendu.

### Sérialisation

Formats:

- SafeTensors
- RawFolder
- DebugJson

Interop:

- checkpoints externes,
- compatibilité HuggingFace selon chemins supportés.

---

### Outils et tests

Outillage intégré:

- benchmarks,
- inspection d'architectures,
- analyse checkpoints / SafeTensors / DebugJson,
- profils mémoire/runtime,
- génération de rapports (HTML/PDF/Mermaid selon scripts).

Le repo inclut des tests unitaires, smoke tests, tests mémoire/runtime/Lua/registry/sérialisation.

---

### Build et dépendances

Prérequis principaux:

- CMake 3.15+
- compilateur C++17
- Lua 5.3+
- OpenMP

Build rapide:

```bash
# Bootstrap automatique (installe les dependances puis configure CMake)
./config.sh

# Compilation
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Exemple avec options:

```bash
WORKROOT=/home/user/tensor-2 ENABLE_VULKAN=1 ENABLE_OPENCL=1 ./config.sh
```

Installation détaillée (Linux/macOS/Windows, options CMake, troubleshooting):
**[docs/01-Getting-Started/02-Installation.md](./docs/01-Getting-Started/02-Installation.md)**

---

### Documentation

- **[Validation de la version 3.6.0](./docs/06-Contributing/08-Release-Validation.md)**

- **[GET STARTED](./docs/01-Getting-Started/00-GET-STARTED.md)**
- **[CLI](./docs/01-Getting-Started/03-CLI.md)**
- **[Config-driven scripting (`--conf`)](./docs/02-User-Guide/08-Config-Driven-Scripting.md)**
- **[Cycle de vie modèle](./docs/02-User-Guide/02-Model-Lifecycle.md)**
- **[Packages d’architecture MPK](./docs/02-User-Guide/15-MPK.md)**
- **[API Reference](./docs/03-API-Reference/00-API-Overview.md)**
- **[Index complet](./docs/00-INDEX.md)**

---

### Structure du dépôt

- `src/` : coeur C++ (runtime, layers, registry, mémoire)
- `scripts/` : templates Lua, workflows, benchmarks, tests
- `configs/` : exemples de config JSON pour mode `--conf`
- `docs/` : documentation technique
- `bin/mimir` : exécutable CLI

---

## Technical Snapshot (EN)

Mímir is organized as a layered execution stack:

1. C++ model architecture definitions
2. Architecture registry
3. Graph runtime + planner
4. Lua/JSON orchestration layer

Recommended v3.0+ model lifecycle:

1. `Mimir.Model.create(name, cfg)`
2. `Mimir.Model.allocate_params()`
3. `Mimir.Model.init_weights(...)` or `Mimir.Serialization.load(...)`
4. `Mimir.Model.forward(...)`

`Model.build()` remains for backward compatibility but is not required in modern flows.

Main capabilities:

- CPU-first runtime (OpenMP, SIMD, AVX2, FMA)
- optional GPU backends (CUDA/ROCm/Vulkan/OpenCL paths)
- integrated memory stack (`MemoryGuard`, `DynamicTensorAllocator`, spill/compression/LRU)
- multimodal dataset pipeline and selectable SFML/Qt/GTK/Web real-time visualization
- checkpoint/weights interoperability (SafeTensors, RawFolder, DebugJson)

Primary docs entry points:

- [GET STARTED](./docs/01-Getting-Started/00-GET-STARTED.md)
- [CLI](./docs/01-Getting-Started/03-CLI.md)
- [Config-driven scripting (`--conf`)](./docs/02-User-Guide/08-Config-Driven-Scripting.md)
- [API Overview](./docs/03-API-Reference/00-API-Overview.md)

---

## Licence

Voir [LICENSE](./LICENSE).
