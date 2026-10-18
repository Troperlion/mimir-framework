# Configuration pilotée par fichier avec `--conf`

Piloter des expériences via JSON avec --conf.

**Public concerné :** Débutant à intermédiaire (ops/automation).

> **Prérequis**
>
> Savoir éditer un fichier JSON.

## Sur cette page

- [Diagramme d'explication](#diagramme-dexplication)
- [QuickStart](#quickstart)
- [Structure de la config](#structure-de-la-config)
- [Sections personnalisables](#sections-personnalisables)
- [Variables injectées](#variables-injectées)
- [Variables d'environnement actives](#variables-denvironnement-actives)
- [Override avec --override](#override-avec---override)
- [Exemples de workflows](#exemples-de-workflows)
- [Bonnes pratiques](#bonnes-pratiques)
- [Troubleshooting](#troubleshooting)
- [Fichiers d'exemple](#fichiers-dexemple)
- [Étapes suivantes](#étapes-suivantes)

## Diagramme d'explication

![Inspection Lua et JSON](../graphs/26_lua_json_inspection.svg)


Le mode `--conf` permet de charger une configuration JSON et d'exécuter automatiquement des scripts Lua avec cette config injectée. C'est idéal pour **automatiser des workflows complexes**, **paramétrer des expériences** et **reproduire des résultats**.

---

## QuickStart

```bash
# 1. Créer une config
cat > my_exp.json << 'EOF'
{
  "lua": {
    "scripts": ["scripts/templates/template_conf_load_and_train.lua"]
  },
  "model": {
    "architecture": "transformer",
    "vocab_size": 4096,
    "d_model": 256,
    "num_layers": 4
  },
  "training": {
    "num_epochs": 5,
    "batch_size": 8,
    "learning_rate": 0.0001
  }
}
EOF

# 2. Exécuter
./bin/mimir --conf my_exp.json
```

---

## Structure de la config

Une config pour `--conf` doit contenir une section `lua.scripts` qui liste les scripts à exécuter:

### Format 1: Scripts simples (chaîne)

```json
{
  "lua": {
    "scripts": [
      "script1.lua",
      "script2.lua"
    ]
  }
}
```

Chaque script est exécuté dans l'ordre. Le binaire construit une nouvelle VM Lua
pour chaque entrée : les globals Lua ne persistent donc pas entre deux scripts.
Utilisez des fichiers, checkpoints ou une section de configuration pour échanger
un état durable entre étapes.

### Format 2: Scripts avec arguments

```json
{
  "lua": {
    "scripts": [
      {
        "script": "script_with_args.lua",
        "args": ["value1", "value2", "42", "true"]
      }
    ]
  }
}
```

Les arguments sont accessibles dans le script via `arg[]` (std Lua).
Le schéma public attend des chaînes. Le runtime sait sérialiser d'autres scalaires,
mais utiliser des chaînes garde la config valide et le parsing explicite.

### Format 3: Chemin alternatif (`run.lua.scripts`)

Si la section `lua.scripts` est profondément imbriquée, vous pouvez l'organiser sous `run.lua.scripts`:

```json
{
  "run": {
    "lua": {
      "scripts": [...]
    }
  }
}
```

---

## Sections personnalisables

Au-delà de `lua.scripts`, vous pouvez ajouter n'importe quelle section JSON. Tout est accessible dans le script via la table `CONF`:

```json
{
  "lua": {
    "scripts": ["my_script.lua"]
  },
  
  "model": {
    "architecture": "transformer",
    "vocab_size": 4096
  },
  
  "training": {
    "num_epochs": 10,
    "learning_rate": 0.0001
  },
  
  "inference": {
    "temperature": 0.8,
    "top_k": 40
  },
  
  "my_custom_section": {
    "key1": "value1",
    "nested": {
      "key2": 42
    }
  }
}
```

Dans `my_script.lua`:

```lua
print("Model arch:", CONF.model.architecture)
print("Epochs:", CONF.training.num_epochs)
print("Custom:", CONF.my_custom_section.key1)
```

---

## Variables injectées

Quand un script s'exécute via `--conf`, ces variables sont automatiquement injectées:

### `CONF` (table)
Toute la configuration JSON convertie en table Lua.

```lua
local model_config = CONF.model
local vocab = CONF.model.vocab_size
```

### `CONF_PATH` (string)
Chemin absolu du fichier config.

```lua
print("Config is at:", CONF_PATH)
-- Output: /home/user/project/my_exp.json
```

### `CONF_DIR` (string)
Répertoire parent du fichier config. **Utile pour chemins relatifs.**

```lua
-- Si config est à /home/user/data/exp.json, CONF_DIR = /home/user/data
local dataset_file = CONF_DIR .. "/dataset.csv"
local checkpoint_dir = CONF_DIR .. "/checkpoints"
```

### `arg[]` (table)
Arguments du script (si spécifiés dans `lua.scripts[i].args`).

```lua
for i = 1, #arg do
    print("Arg " .. i .. ":", arg[i])
end
```

---

## Override avec `--override`

Modifie des valeurs config avant exécution des scripts:

```bash
./bin/mimir --conf my_exp.json --override model.vocab_size=8192 --override training.num_epochs=20
```

Syntaxe: `--override path.to.key=value`

Les overrides sont appliqués en JSON path. Exemples:

```bash
# Scaler numérique
--override model.d_model=512

# Chaîne
--override model.architecture=gpt

# Booléen
--override training.use_mixed_precision=true

# Imbriqué
--override training.lr_decay.enabled=false
```

**Important:** Les overrides sont appliqués **avant** l'exécution des scripts, donc les scripts verront les valeurs modifiées dans `CONF`.

## Variables d'environnement actives

La section racine `env` n'est pas une annotation : Mímir applique réellement
ses valeurs au processus après les `--override` et avant de résoudre/exécuter le
run. Une valeur de config remplace une valeur héritée du shell ou exportée par
`run_mimir.sh`.

```json
{
  "$schema": "./configs/conf.schema.json",
  "env": {
    "OMP_NUM_THREADS": 10,
    "MIMIR_RUNTIME_TRACE": false
  },
  "lua": {
    "scripts": ["scripts/templates/template_conf_load_and_train.lua"]
  }
}
```

Pour `OMP_NUM_THREADS`, Mímir met aussi à jour le runtime OpenMP déjà chargé.
Les chaînes, nombres et booléens sont acceptés; un objet, un tableau, `null` ou
un nom de variable invalide arrête le run avec une erreur.

```bash
# Le shell propose 2, mais la config ci-dessus impose 10.
OMP_NUM_THREADS=2 ./run_mimir.sh --conf my_exp.json
```

Les variables sont globales au processus : tous les scripts de la config voient
la même valeur. Consultez la [référence des variables](../03-API-Reference/22-Environment-Variables.md)
pour savoir quelles variables sont effectivement lues par chaque sous-système.

---

## Exemples de workflows

Les chemins `scripts/workflows/*` et `scripts/stages/*` des exemples ci-dessous
sont des **noms illustratifs à créer dans votre projet** ; ils ne sont pas fournis
dans ce dépôt. Pour partir d’un fichier existant, copie
`scripts/templates/template_conf_load_and_train.lua` et adapte-le.

### Exemple 1: Entraînement paramétrisé

```jsonc
{
  "lua": {
    "scripts": [
      "scripts/workflows/train.lua"
    ]
  },
  "model": {
    "architecture": "transformer",
    "vocab_size": 4096,
    "d_model": 256,
    "num_layers": 4
  },
  "training": {
    "num_epochs": 10,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "checkpoint_dir": "./checkpoints/exp1"
  },
  "dataset": {
    "train_path": "/data/train.csv",
    "val_path": "/data/val.csv"
  }
}
```

Exécute:
```bash
./bin/mimir --conf my_exp.json
```

### Exemple 2: Pipeline multi-stage

```jsonc
{
  "lua": {
    "scripts": [
      "scripts/stages/1_preprocess.lua",
      "scripts/stages/2_train.lua",
      "scripts/stages/3_evaluate.lua",
      "scripts/stages/4_export.lua"
    ]
  },
  "dataset": { ... },
  "model": { ... },
  "training": { ... }
}
```

Chaque stage s'exécute dans l'ordre dans une VM Lua distincte. Faites persister
les résultats nécessaires dans un chemin explicite, puis relisez-les au stage
suivant.

### Exemple 3: Hyperparameter sweep

```bash
# Automatiser plusieurs runs avec overrides
for d_model in 128 256 512; do
  for lr in 0.00001 0.0001 0.001; do
    echo "Running: d_model=$d_model, lr=$lr"
    ./bin/mimir --conf my_exp.json \
      --override model.d_model=$d_model \
      --override training.learning_rate=$lr
  done
done
```

### Exemple 4: Sauvegarder le checkpoint en fonction de la config

```json
{
  "lua": {
    "scripts": [
      "scripts/workflows/train.lua",
      "scripts/workflows/save_checkpoint.lua"
    ]
  },
  "model": {
    "architecture": "transformer",
    "vocab_size": 4096
  },
  "checkpoint": {
    "format": "safetensors",
    "output_dir": "./checkpoints/my_exp",
    "compress": true
  }
}
```

Le script `save_checkpoint.lua` accède à `CONF.checkpoint` pour savoir où sauvegarder.

---

## Bonnes pratiques

### 1. **Utilise des chemins relatifs à `CONF_DIR`**

Mauvais:
```lua
local data = load_csv("../../data/train.csv")
```

Bon:
```lua
local data = load_csv(CONF_DIR .. "/../data/train.csv")
```

### 2. **Documente votre config avec un `description` field**

```json
{
  "description": "Exp #42: Transformer avec VAE pour text2img",
  "lua": { ... }
}
```

### 3. **Groupe les paramètres logiquement**

```json
{
  "lua": { "scripts": [...] },
  "model": { ... },
  "training": { ... },
  "inference": { ... },
  "logging": { ... }
}
```

### 4. **Valide la config au démarrage du script**

```lua
if not CONF.model or not CONF.model.vocab_size then
    error("CONF.model.vocab_size is required!")
end
```

### 5. **Utilise des valeurs par défaut dans le script**

```lua
local num_epochs = CONF.training.num_epochs or 10
local batch_size = CONF.training.batch_size or 32
```

---

## Troubleshooting

### Error: "CONF not injected! Use --conf mode."

**Problème:** Le script s'attend à `CONF` mais est exécuté avec `--lua`.

**Solution:** Utilise `--conf` au lieu de `--lua`:
```bash
# ❌ Mauvais
./bin/mimir --lua my_script.lua

# ✅ Correct
./bin/mimir --conf config.json
```

### Error: "aucune section lua trouvée"

**Problème:** Le JSON n'a pas de `lua.scripts` ou `run.lua.scripts`.

**Solution:** Ajoute une section `lua`:
```json
{
  "lua": {
    "scripts": ["script.lua"]
  }
}
```

### Variables de config ne sont pas mises à jour

**Problème:** Vous utilisez `--override` mais le script ne voit pas les changements.

**Solution:** Les overrides ne changent que `CONF`. Si le script capture une variable locale avant l'override, la modification ne sera pas visible. Toujours accéder via `CONF` au lieu de cacher en variable locale:

```lua
-- ❌ Mauvais: la variable locale n'est pas affectée par les overrides
local epochs = CONF.training.num_epochs
-- Même après override, 'epochs' garde l'ancienne valeur

-- ✅ Correct: accéder à CONF directement
print(CONF.training.num_epochs)  -- Voit l'override
```

---

## Fichiers d'exemple

Voir:
- `configs/example_conf_driven.json` — Exemple simple de config
- `scripts/templates/template_conf_load_and_train.lua` — Template pour entraînement
- `scripts/examples/example_conf_inference.lua` — Exemple d'inférence

Utilisation:
```bash
./bin/mimir --conf configs/example_conf_driven.json
```

## Étapes suivantes

- [Page précédente : Checkpoints / reprise d’entraînement](08-Checkpoints.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Mémoire (Allocator, MemoryGuard)](09-Memory.md)
