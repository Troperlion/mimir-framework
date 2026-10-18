# CLI (binaire `mimir`)

Comprendre les modes --lua et --conf sans ambiguïté.

**Public concerné :** Débutant à intermédiaire.

> **Prérequis**
>
> Binaire mimir compilé.

## Sur cette page

- [Diagrammes d'explication](#diagrammes-dexplication)
- [Usage](#usage)
- [Exemples d'utilisation](#exemples-dutilisation)
- [Variables injectées par --conf](#variables-injectées-par---conf)
- [Exécuter un script Lua avec arguments](#exécuter-un-script-lua-avec-arguments)
- [Sortie au démarrage](#sortie-au-démarrage)
- [Étapes suivantes](#étapes-suivantes)

## Diagrammes d'explication

![Workflow Lua JSON CLI](../graphs/04_workflow_lua_json_cli.svg)

![Détection des backends au démarrage](../graphs/14_backend_detection.svg)


Le binaire principal est `bin/mimir`.

## Usage

```bash
./bin/mimir --help
```

Options :

- `--lua <script.lua>` : exécute un script Lua.
- `--conf <config.json>` : charge une config JSON et exécute les scripts Lua déclarés dans `lua.scripts`.
- `--config <config.json>` : construit et entraîne directement depuis une configuration modèle historique.
- `--run <task>` : sélectionne une tâche sous `tasks` avec `--conf`.
### Options principales

| Option | Description |
| --- | --- |
| `--lua <script.lua>` | Exécute un script Lua standalone |
| `--conf <config.json>` | Charge une config JSON et exécute les scripts Lua spécifiés dans `lua.scripts` |
| `--config <config.json>` | Mode historique de construction/entraînement direct |
| `--run <task>` | Sélectionne `tasks.<task>` en mode `--conf` |
| `--override <path=value>` | Override des valeurs config (répétable, appliqué avant exécution) |

### Modes de fonctionnement

#### Mode `--lua` (standalone)

Exécute un script Lua directement. La config n'est pas injectée.

```bash
./bin/mimir --lua script.lua [args...]
```

**Injection disponible :**
- `arg[]` : arguments du script (std Lua)
- `Mimir` et les fonctions globales du framework

#### Mode `--conf` (config-driven)

Charge un JSON avec section `lua.scripts` et exécute les scripts dans l'ordre.

```bash
./bin/mimir --conf config.json [--override path=value ...]
```

**Injection disponible :**
- `CONF` : table Lua contenant toute la config JSON
- `CONF_PATH` : chemin absolu du fichier config
- `CONF_DIR` : répertoire parent du fichier config
- `arg[]` : arguments spécifiés pour chaque script dans `lua.scripts[i].args`

**Structure de config attendue :**

```json
{
	"lua": {
		"scripts": [
			"script1.lua",
			{
				"script": "script2.lua",
				"args": ["arg1", "arg2"]
			}
		]
	},
	"model": { ... },
	"training": { ... }
}
```

Ou le chemin `run.lua.scripts`:

```json
{
	"run": {
		"lua": {
			"scripts": [...]
		}
	},
	"model": { ... }
}
```
## Exécuter un script Lua avec arguments

## Exemples d'utilisation

### Exemple 1: Mode `--conf` avec config simple

```bash
./bin/mimir --conf configs/example_conf_driven.json
```

Charge la config, injecte `CONF`, `CONF_PATH`, `CONF_DIR` et exécute les scripts dans `lua.scripts`.

### Exemple 2: Mode `--conf` avec override

```bash
./bin/mimir --conf configs/example_conf_driven.json --override model.d_model=512 --override training.num_epochs=5
```

Les overrides sont appliqués avant l'exécution des scripts, donc les scripts verront les valeurs modifiées dans `CONF`.

### Exemple 3: Mode `--lua` avec arguments

```bash
./bin/mimir --lua scripts/examples/example_conf_inference.lua -- --num-samples 20
```

Les arguments après `--` sont accessibles dans `arg[]`. Utilisez
`scripts/modules/args.lua` pour les analyser.

### Exemple 4: Créer votre propre config

Crée un fichier `my_config.json`:

```json
{
	"lua": {
		"scripts": [
			"scripts/templates/template_conf_load_and_train.lua"
		]
	},
	"model": {
		"architecture": "transformer",
		"vocab_size": 4096,
		"seq_len": 256,
		"d_model": 512,
		"num_layers": 6,
		"num_heads": 8,
		"mlp_hidden": 2048
	},
	"training": {
		"num_epochs": 10,
		"batch_size": 16,
		"learning_rate": 0.0001,
		"optimizer": "adamw"
	}
}
```

Puis exécute:

```bash
./bin/mimir --conf my_config.json
```

## Variables injectées par --conf

Quand un script est exécuté via `--conf`, les variables suivantes sont automatiquement injectées dans le contexte Lua global:

| Variable | Type | Contenu |
| --- | --- | --- |
| `CONF` | table | Toute la config JSON parsée en table Lua |
| `CONF_PATH` | string | Chemin absolu du fichier config |
| `CONF_DIR` | string | Répertoire parent du fichier config (utile pour chemins relatifs) |
| `arg[]` | table | Arguments spécifiés dans `lua.scripts[i].args` (ou empty si pas d'args) |

**Accès dans le script:**

```lua
-- Récupérer la config du modèle
local model_config = CONF.model

-- Récupérer un training param
local epochs = CONF.training.num_epochs

-- Construire un chemin relatif à la config
local dataset_path = CONF_DIR .. "/" .. "data/train.csv"

-- Accéder aux args
for i, v in ipairs(arg) do
	print("Arg " .. i .. ": " .. v)
end
```

## Exécuter un script Lua avec arguments

Le programme injecte la table globale `arg`, comme l’interpréteur Lua.

Exemple :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- \
  --list vae_text --params --layers
```

Notes :

- Le séparateur `--` termine les options du CLI Mímir ; les valeurs suivantes sont transmises au script.
- Dans un script, parsage recommandé : `local Args = dofile(ROOTWORK.."/scripts/modules/args.lua")` puis `Args.parse(arg)`.

## Sortie au démarrage

Au lancement, Mímir affiche :

- la version et le chemin du journal complet ;
- le mode ou script demandé ;
- les erreurs et résultats importants.

La console est concise par défaut, tandis que `logs/*.log` conserve le détail.
Utilisez `MIMIR_CONSOLE_VERBOSE=1` pour restaurer toutes les lignes du framework
et `MIMIR_LAUNCH_VERBOSE=1` pour détailler le lanceur `run_mimir.sh`.

## Étapes suivantes

- [Page précédente : 🔧 Installation & Compilation détaillée](02-Installation.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Organisation du dépôt](04-Repo-Layout.md)
