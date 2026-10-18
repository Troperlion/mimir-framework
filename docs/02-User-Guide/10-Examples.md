# Scripts et outils Lua

Expliquer le rôle des dossiers `scripts/`, choisir le bon point de départ et utiliser chaque famille d’outils sans confondre inspection, test, benchmark, inférence et entraînement.

**Public concerné :** Débutant qui apprend par l’exécution, utilisateur qui cherche un workflow existant, ou développeur qui veut réutiliser les modules Lua du dépôt.

> **Prérequis**
>
> Exécuter les commandes depuis la racine du dépôt avec un binaire `./bin/mimir` compilé.

## Sur cette page

- [1. Convention de lancement](#1-convention-de-lancement)
- [2. Choisir une famille](#2-choisir-une-famille)
- [3. Parcours sans dataset](#3-parcours-sans-dataset)
- [4. Exemples](#4-exemples)
- [5. Templates](#5-templates)
- [6. Outils d’inspection](#6-outils-dinspection)
- [7. Conversion de checkpoints](#7-conversion-de-checkpoints)
- [8. MPK : packaging d’architecture](#8-mpk-packaging-darchitecture)
- [9. Scripts d’inférence](#9-scripts-dinférence)
- [10. Scripts d’entraînement](#10-scripts-dentraînement)
- [11. Modules réutilisables](#11-modules-réutilisables)
- [12. Dépannage](#12-dépannage)
- [Références](#références)
- [Étapes suivantes](#étapes-suivantes)

## 1. Convention de lancement

Forme générale :

```bash
./bin/mimir --lua chemin/script.lua -- arguments_du_script
```

Exemple :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- \
  --list vae_conv --params --layers
```

Le premier `--` sépare les options de Mímir des arguments Lua. Le runtime construit :

```lua
arg[0] = "scripts/tools/inspect_architectures.lua"
arg[1] = "--list"
arg[2] = "vae_conv"
```

Les scripts qui chargent `scripts/modules/args.lua` acceptent :

```text
--key value
--key=value
--flag
--no-flag
--override chemin.valeur=valeur
```

Pour afficher l’aide détectée :

```bash
./bin/mimir --lua scripts/training/train_vae_conv.lua -- --help
```

## 2. Choisir une famille

| Dossier | Rôle | Peut écrire ? | Peut demander des données ? |
| --- | --- | --- | --- |
| `scripts/examples/` | exemples ciblés et lisibles | parfois | selon l’exemple |
| `scripts/templates/` | base à copier pour un nouveau workflow | oui si save activé | optionnel |
| `scripts/tools/` | inspection, conversion, packaging, rapports | souvent | checkpoint/config selon l’outil |
| `scripts/tests/` | smoke tests fonctionnels | souvent dans un chemin temporaire | certains oui |
| `scripts/benchmarks/` | mesure performance et stress | logs/rapports possibles | généralement synthétique |
| `scripts/inferences/` | exécution de modèles ou checkpoints | images/rapports possibles | checkpoint et entrées |
| `scripts/training/` | entraînement complet | checkpoints, CSV, logs | oui, sauf scripts directs particuliers |
| `scripts/modules/` | bibliothèques chargées par `dofile` | dépend du module | non directement |

## 3. Parcours sans dataset

### Inspecter le registre

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- -a

./bin/mimir --lua scripts/tools/inspect_architectures.lua -- \
  --list vae_conv --params --layers --stats --ops --runtime
```

Fonctions utiles :

- `-a`, `--show-archs` : architectures et dtypes ;
- `--list <arch>` : sélection ;
- `--params` : config par défaut ;
- `--layers` : graphe construit ;
- `--stats` : paramètres et estimation d’opérations ;
- `--ops` : types de layers observés ;
- `--runtime` : capacités exposées par le runtime ;
- `--json` : export structuré du registre.

### Construire via la Pipeline API

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry --arch basic_mlp --no-train
```

Ce template utilise `scripts/modules/pipeline_api.lua` et exécute :

```text
default_config
  → patchConfig
  → Model.create
  → dtype éventuel
  → allocate_params
  → init_weights
```

### Inspecter VAEConv

```bash
./bin/mimir --lua scripts/examples/inspect_vae_conv.lua
```

Cet exemple construit un petit graphe avec GroupNorm et prior appris. Il ne charge aucune donnée.

### Tester la sérialisation

```bash
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
```

Ce smoke test écrit des artefacts temporaires. Utilise-le pour vérifier ensemble API Lua, poids et sérialisation.

### Mesurer le runtime

```bash
./bin/mimir --lua scripts/benchmarks/benchmark_official.lua -- \
  --safe --iters 1
```

`--safe` choisit un profil réduit. Un benchmark mesure le temps ; ce n’est pas une preuve de correction mathématique. Complète-le par les tests C++.

## 4. Exemples

| Script | Usage |
| --- | --- |
| `scripts/examples/inspect_vae_conv.lua` | structure VAEConv sans données |
| `scripts/examples/example_conf_inference.lua` | exemple piloté par la globale `CONF` |
| `scripts/examples/classify_vgg16_feat.lua` | classification depuis checkpoint et labels |

`classify_vgg16_feat.lua` lit des images et un checkpoint. Consulte son `--help` avant exécution.

## 5. Templates

### `template_new_model.lua`

Template pédagogique détaillé. Il montre mémoire, registre, allocation, initialisation, forward, boucle manuelle commentée et sauvegarde.

```bash
./bin/mimir --lua scripts/templates/template_new_model.lua
```

Utilise-le pour apprendre le cycle bas niveau, pas comme script de production sans relecture.

### `template_pipeline_only.lua`

Pipeline minimal piloté surtout par variables d’environnement :

```bash
./bin/mimir --lua scripts/templates/template_pipeline_only.lua
```

### `template_pipeline_args.lua`

Template recommandé pour commencer un nouveau script paramétrable :

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry --arch transformer \
  --d-model 128 --layers 2 --heads 4 \
  --no-train
```

Pour sauvegarder :

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry --arch transformer \
  --no-train \
  --save /tmp/transformer_demo.safetensors
```

### Templates `--conf`

- `template_conf_load_and_train.lua` : charge `CONF`, crée et entraîne ;
- `example_conf_inference.lua` : lit la même configuration pour l’inférence.

Voir [Config-driven scripting](08-Config-Driven-Scripting.md).

## 6. Outils d’inspection

### `analyze_model.lua`

Analyse RawFolder, SafeTensors ou DebugJson :

```bash
./bin/mimir --lua scripts/tools/analyze_model.lua -- \
  --in checkpoint/model.safetensors \
  --graph-format mermaid
```

Il lit les métadonnées et tensors du checkpoint. Il ne construit pas automatiquement un modèle entraînable.

### `show-graph.lua`

Transforme un ou plusieurs CSV d’entraînement en rapport HTML :

```bash
./bin/mimir --lua scripts/tools/show-graph.lua -- \
  --csv checkpoint/run/loss_history.csv \
  --out /tmp/mimir_training_report.html \
  --no-interactive
```

Le mode `--watch` surveille les changements et régénère le rapport.

### `inspect_z_prior_raw_folder.lua`

Inspecte le tensor du prior VAEConv dans un checkpoint RawFolder :

```bash
./bin/mimir --lua scripts/tools/inspect_z_prior_raw_folder.lua -- --help
```

Il peut produire statistiques, CSV ou aperçu PPM selon ses options.

## 7. Conversion de checkpoints

```bash
# RawFolder → SafeTensors
./bin/mimir --lua scripts/tools/convert_checkpoint2safetensor.lua -- --help

# SafeTensors → RawFolder
./bin/mimir --lua scripts/tools/convert_safetensors2raw_folder.lua -- --help
```

Avant conversion :

1. conserve une copie de l’original ;
2. vérifie l’architecture et le dtype ;
3. convertis vers un nouveau chemin ;
4. analyse la sortie ;
5. charge-la en mode strict si le workflow le permet.

## 8. MPK : packaging d’architecture

Les fichiers MPK encapsulent métadonnées, configuration et éventuellement structure de graphe.
Par défaut, ils sont écrits dans un pseudocode lisible inspiré de Visu :
déclarations `map`/`list`, puis opérations `.set(...)` et `.append(...)`.
L’écriture produit uniquement le pseudocode moderne. Celui-ci peut ensuite être
compilé en binaire v4 opaque avec `compile_mpk.lua`. La lecture reste compatible avec
les anciens MPK JSON et binaires pour migration.

Pour le format complet, l’écriture en Lua, l’export, le prototypage nodal et
l’autochargement via `_archi/`, voir
[MPK : packages d’architecture](15-MPK.md).

| Outil | Rôle |
| --- | --- |
| `build_mpk.lua` | construire un package |
| `compile_mpk.lua` | compiler une source pseudocode MPK vers le binaire v4 opaque |
| `load_mpk.lua` | vérifier, inspecter ou créer |
| `export_arch_mpk.lua` | exporter une architecture complète |
| `mpk_node_wizard.lua` | assembler interactivement un graphe |
| `scripts/modules/mpk.lua` | lecture, écriture, checksum, conteneur |
| `scripts/modules/mpk_layers.lua` | normalisation des types de layers |

Exemple :

```bash
./bin/mimir --lua scripts/tools/build_mpk.lua -- \
  --name vae_conv_demo \
  --type vae_conv \
  --from-registry \
  --out /tmp/vae_conv_demo.mpk

./bin/mimir --lua scripts/tools/load_mpk.lua -- \
  --in /tmp/vae_conv_demo.mpk \
  --create
```

Depuis Lua, le même chargement passe directement par le registre :

```lua
local ok, err = Mimir.Model.create("/tmp/vae_conv_demo.mpk")
assert(ok, err)
```

### Découverte automatique avec `_archi`

Quand un dossier `_archi/` existe à la racine du projet, Mímir inspecte
automatiquement ses fichiers `.mpk` au démarrage. Chaque package valide dont
l’architecture de base existe est ajouté à `Mimir.Architectures.available()`.

```text
_archi/
└── vae_conv_pseudocode.mpk
```

```lua
local cfg = Mimir.Architectures.default_config("vae_conv_pseudocode")
local ok, err = Mimir.Model.create("vae_conv_pseudocode", cfg)
assert(ok, err)
```

Le nom déclaré dans le MPK est utilisé en priorité. S’il entre en collision
avec une architecture native, le nom du fichier sans l’extension devient
l’alias. Un MPK invalide est ignoré avec un avertissement au démarrage.

`--apply-graph` reconstruit un graphe nodal hors registre uniquement si ses types sont compatibles avec `push_layer`.

## 9. Scripts d’inférence

Le dossier contient :

- encodeurs texte CLIP 1 et 2 ;
- bloc transformer SDXL ;
- décodeur VAE Hugging Face ;
- inspection d’une base SafeTensors externe.

Ces scripts nécessitent généralement des checkpoints précis. Commence toujours par :

```bash
./bin/mimir --lua scripts/inferences/hf_vae_decoder_infer.lua -- --help
```

Ne suppose pas qu’un checkpoint PyTorch/Hugging Face est directement compatible : les noms, shapes et mappings doivent correspondre.

## 10. Scripts d’entraînement

| Script | Domaine |
| --- | --- |
| `train_vae_conv.lua` | VAE image convolutionnel |
| `train_vae_texte.lua` | VAE texte |
| `pretrain_vgg16_feat.lua` | extracteur perceptuel |
| `train_vgg16_tags_multilabel.lua` | classification multi-label |
| `prepare_tokenizer_dataset.lua` | préparation tokenizer |
| `train_face_recognition_mimir.lua` | workflow reconnaissance faciale |

Les scripts d’entraînement peuvent écrire checkpoints, historique CSV, logs et artefacts de visualisation. Lis l’aide et choisis explicitement les chemins de sortie.

## 11. Modules réutilisables

| Module | Contrat principal |
| --- | --- |
| `args.lua` | parse flags, booléens, nombres et overrides |
| `help_cli.lua` | aide automatique dérivée du script |
| `fs.lua` | chemins et filesystem multiplateforme |
| `pipeline.lua` | façade compatible vers `pipeline_api.lua` |
| `pipeline_api.lua` | construction registry-first, train, infer, save |
| `checkpoint_resume.lua` | résolution d’un RawFolder ou du dernier `epoch_*` |
| `base_tokenizer.lua` | vocabulaire/tokenizer partagé |
| `mpk.lua` | format MPK |
| `mpk_layers.lua` | validation de graphes MPK |
| `api_ws_server.lua` | serveur HTTP/WebSocket ; dépend de LuaSocket |

Exemple de parsing :

```lua
local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")
local opts = Args.parse(arg) or {}

local epochs = Args.get_int(opts, "epochs", 1)
local train = Args.get_bool(opts, "train", true)
local cfg = Args.apply_overrides(base_cfg, opts)
```

## 12. Dépannage

| Problème | Cause probable | Action |
| --- | --- | --- |
| Le script ignore les flags | séparateur `--` absent | ajouter `--` avant les arguments Lua |
| Architecture inconnue | nom non enregistré/alias incorrect | lancer `inspect_architectures.lua -- -a` |
| Forward avant allocation | cycle incomplet | appeler `allocate_params`, puis init ou load |
| Checkpoint incompatible | config/topologie différente | analyser config, layers et shapes |
| GPU non utilisé | backend non compilé, désactivé ou op refusée | activer les logs runtime et vérifier le fallback |
| OOM | résolution, batch, attention ou cache trop grands | réduire progressivement et fixer MemoryGuard |

## Références

- [Parcours par niveau](../01-Getting-Started/06-Learning-Paths.md)
- [API Lua](06-Lua-Scripting.md)
- [Cycle de vie](02-Model-Lifecycle.md)
- [Checkpoints](08-Checkpoints.md)
- [VAEConv](14-VAEConv.md)
- [MPK : packages d’architecture](15-MPK.md)
- [Référence des architectures](../03-API-Reference/11-Architectures.md)

## Étapes suivantes

- [Page précédente : Mémoire (Allocator, MemoryGuard)](09-Memory.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Tutoriel : VAEText](11-VAEText.md)
