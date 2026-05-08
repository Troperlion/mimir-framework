# Globals & aliases

Cette page documente ce que le runtime Lua injecte **automatiquement** quand tu lances un script via `./bin/mimir --lua ...`.

Objectif :

- te permettre d’écrire des scripts courts (aliases globaux pratiques)
- rester compatible avec des scripts historiques
- éviter les ambiguïtés (`arg` vs `Mimir.Args`, collisions de noms, etc.)

Source de vérité : `src/LuaScripting.cpp`.

## 1bis) Globals supplémentaires en mode `--conf`

Quand un script est exécuté via `./bin/mimir --conf <config.json>`, le binaire injecte en plus :

| Nom | Type | Sens |
| --- | --- | --- |
| `CONF` | table | config JSON complète (convertie en table Lua) |
| `CONF_PATH` | string | chemin absolu du JSON |
| `CONF_DIR` | string | dossier parent du JSON |

But : permettre à un script de lire la conf complète et/ou d’instancier un modèle via `Mimir.Model.create_from_config(CONF)`.

## 1) Globals injectés

### Table `arg`

Le binaire remplit une table globale `arg` “à la façon de Lua” :

- `arg[0]` = chemin du script
- `arg[1..n]` = arguments

Conseil : préfère `Mimir.Args` (copie explicite) si tu veux être sûr de ne pas marcher sur une variable `arg` modifiée par un module.

### Table `Mimir`

Le runtime publie la table `Mimir` avec les sous-modules (`Mimir.Model`, `Mimir.Serialization`, etc.).

### Fonctions utilitaires globales

Ces fonctions existent pour éviter d’importer un module juste pour des tâches basiques.

| Nom | Signature | Retour | À utiliser quand | Notes |
| --- | --- | --- | --- | --- |
| `log` | `log(msg: string)` | rien | logs côté runtime Mímir | alias d’un `print` runtime |
| `read_json` | `read_json(path: string)` | `table` ou `(nil, err)` | charger une config/metadata | parse JSON -> table Lua |
| `write_json` | `write_json(path: string, table)` | `true` ou `(false, err)` | écrire un état/rapport | sérialise table Lua -> JSON |

## 2) Aliases globaux (rétrocompat + confort)

En plus de `Mimir.*`, le runtime publie des **aliases globaux** qui pointent vers les sous-tables `Mimir.*`.

| Alias global | Pointe vers | Pourquoi ça existe | Recommandation |
| --- | --- | --- | --- |
| `model` | `Mimir.Model` | scripts courts, legacy | OK pour scripts jetables, préfère `Mimir.Model` en projet long |
| `architectures` | `Mimir.Architectures` | idem | idem |
| `tokenizer` | `Mimir.Tokenizer` | idem | idem |
| `dataset` | `Mimir.Dataset` | idem | idem |
| `Memory` | `Mimir.Memory` | legacy + quick debug | préfère `Mimir.Memory` |
| `MemoryGuard` | `Mimir.MemoryGuard` | limite stricte mémoire | recommandé (sécurité) |
| `Allocator` | `Mimir.Allocator` | allocateur dynamique | recommandé (stabilité) |
| `htop` | `Mimir.Htop` | monitoring terminal | optionnel |
| `viz` | `Mimir.Viz` | visualisation SFML | optionnel |
| `visualiser` | `Mimir.visualiser` | alias FR de Viz | optionnel |
| `neuropulse` | `Mimir.NeuroPulse` | démo “texte->audio/lumière” | optionnel |

## 3) Conseils (éviter les pièges)

- Ne redéfinis pas ces noms globalement (`model = ...`, `MemoryGuard = ...`). Si tu veux une variable, fais `local model = ...`.
- Pour éviter les collisions, tu peux adopter ce pattern :

```lua
local M = Mimir
local Model = Mimir.Model
local Ser = Mimir.Serialization
```

- Quand tu as un doute sur l’API, la source de vérité est le fichier `src/LuaScripting.cpp`.
