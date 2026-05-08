# CLI (binaire `mimir`)

Le binaire principal est `bin/mimir`.

## Usage

```bash
./bin/mimir --help
```

Options :

- `--lua <script.lua>` : exécute un script Lua.
- `--config <config.json>` : charge une config JSON et **crée un modèle** via le registre (chemin “starter”).
- `--conf <config.json>` : charge une conf JSON et exécute `lua.scripts` (ou `run.lua.scripts`).
- `--override <path=value>` : override (répétable) appliqué à la config JSON avant exécution.

## Mode `--config` (créer un modèle depuis JSON)

Ce mode :

- lit `config.json`
- applique les `--override` (si fournis)
- détermine l’architecture à instancier
- appelle le registre d’architectures

Résolution d’architecture :

- clé top-level `architecture` (sinon `type`)
- fallback : `model.architecture` / `model.type`

Convention : les overrides “modèle” peuvent vivre dans :

- `model` (overrides génériques)
- et/ou dans la section nommée comme l’architecture (ex: `unet`, `vae_conv`, `transformer`)

Exemples :

```bash
./bin/mimir --config config.json
./bin/mimir --config config.json --override architecture="vae_conv"
./bin/mimir --config config.json --override tokenizer.max_vocab=64000
```

Note : ce mode crée le modèle, alloue et initialise les poids ; l’entraînement “end-to-end” est à implémenter selon votre workflow (Lua recommandé).

## Mode `--conf` (exécuter une liste de scripts)

Ce mode est fait pour piloter des scripts Lua depuis une conf JSON (batch / smoke / CI local).

Structure attendue :

- `lua.scripts` : tableau de scripts
- ou `run.lua.scripts` (même contenu, pour structurer la conf)

Chaque entrée de `scripts` peut être :

- une string : chemin du script
- un objet `{ "script": "...", "args": [...] }`

Exemple minimal :

```json
{
	"lua": {
		"scripts": [
			{"script": "scripts/tests/test_list_archi_conf.lua", "args": []}
		]
	}
}
```

En mode `--conf`, le runtime injecte aussi des globals : `CONF`, `CONF_PATH`, `CONF_DIR` (voir `docs/03-API-Reference/19-Globals.md`).

Exemples :

```bash
./bin/mimir --conf config.json
./bin/mimir --conf config.json --override architecture="unet" --override unet.depth=5
```

## Syntaxe des `--override`

Format : `path.to.key=value`.

Parsing des valeurs :

- `true` / `false` / `null`
- si la valeur commence par `{`, `[` ou `"` : tentative de parse JSON
- sinon : tentative `int`, puis `float`, sinon string

## Exécuter un script Lua avec arguments

Le programme injecte la table globale `arg` (comme l’interpréteur Lua) et aussi `Mimir.Args`.

Exemple :

```bash
./bin/mimir --lua scripts/examples/vae_text_sample.lua -- --ckpt checkpoint/_smoke_vae_text_ce --mode prior
```

Notes :

- Le séparateur `--` est un usage pratique pour “séparer” les args du binaire de ceux du script. Le binaire ne le consomme pas explicitement ; certains scripts l’acceptent/ignorent.
- Dans un script, parsage recommandé : `local Args = dofile("scripts/modules/args.lua")` puis `Args.parse(arg)`.

## Sortie au démarrage

Au lancement, Mímir affiche :

- tests d’intégrité mémoire
- capacités CPU (AVX2/FMA/F16C/BMI2)
- configuration OpenMP

Ces logs sont utiles pour diagnostiquer un build “lent” ou une machine non compatible AVX.
