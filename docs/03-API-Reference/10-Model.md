# API : `Mimir.Model`

Source : `src/LuaScripting.cpp`.

Cette page documente l’API brute, mais aussi le **workflow recommandé** pour éviter les erreurs courantes.

## Workflow recommandé (rappel)

| Étape | Appel |
| ---: | --- |
| 1 | `Mimir.Model.create(type, cfg)` |
| 2 | `Mimir.Model.build()` |
| 3 | `Mimir.Model.allocate_params()` |
| 4a | `Mimir.Model.init_weights(method, seed)` |
| 4b | `Mimir.Serialization.load(path)` |
| 5 | `Mimir.Model.forward(input, training)` |

Détails : `docs/02-User-Guide/02-Model-Lifecycle.md`.

## Create / build

### `Mimir.Model.create(name: string, cfg?: table) -> bool | (false, err)`

- Si `cfg` absent/vide : utilise la config par défaut du registre.
- Crée le modèle via `ModelArchitectures::create`.

### `Mimir.Model.create_from_config(full_cfg: table) -> (bool, arch|err)`

Crée le modèle depuis une **config complète** (ex: `CONF` injecté par le mode CLI `--conf`).

Comportement :

- résout l’architecture depuis `full_cfg.architecture` (fallback `type`, puis `model.architecture|model.type`)
- construit une config effective à partir de la config par défaut du registre + merges
- crée le modèle via le registre

Retour :

- `true, <arch>` si OK
- `false, <message>` sinon

### `Mimir.Model.build() -> (bool, params|err)`

- Recrée le modèle via le registre à partir de `ctx.modelType` + `ctx.modelConfig`.
- Ne fait **pas** `allocate_params` ni `init_weights`.

## Paramètres

### `Mimir.Model.allocate_params() -> (bool, total_params|err)`

Alloue les poids.

### `Mimir.Model.init_weights(method?: string="he", seed?: int=0) -> bool | (false, err)`

Initialise les poids. Méthodes usuelles : `he`, `xavier`.

### `Mimir.Model.total_params() -> int`

## Exécution

### `Mimir.Model.forward(input: table, training?: bool=true) -> table<float> | (nil, err)`

Deux formes d’inputs :

1) **liste** : `{1,2,3}` (int) ou `{0.1, 0.2}` (float)
2) **map** : `{ __input__ = {...}, text_ids = {...} }` (entrées mixtes float/int)

Conseil : utilise le format **map** dans tes scripts (même si tu n’as qu’une entrée). Ça rend ton code compatible avec les architectures multi-input.

| Forme | Exemple | À privilégier quand |
| --- | --- | --- |
| liste | `{0.1, 0.2, 0.3}` | quick tests mono-input |
| map | `{ __input__ = ids }` | scripts réutilisables |
| map multi-input | `{ text_ids = ids, __input__ = x }` | texte+image/latent |

Note sur `training` : certaines architectures/layers peuvent activer des comportements différents (dropout, stats, etc.). En inférence, passe `false`.

### `Mimir.Model.backward(grad_out: table<float>) -> bool | (false, err)`

`grad_out` est le gradient par rapport à la sortie principale.

### `Mimir.Model.optimizer_step() -> bool | (false, err)`

Applique l’étape d’optimizer (selon l’état/config).

### `Mimir.Model.zero_grads() -> bool | (false, err)`

## Entraînement “haut niveau”

### `Mimir.Model.train(epochs: int, lr: number) -> (bool, err)`

- Implémente des chemins spécifiques à certains modèles.
- Peut faire des checkpoints d’interruption Ctrl+C si `cfg.checkpoint_dir` est défini.

## Divers

### `Mimir.Model.encode_prompt(prompt?: string) -> table<float> | (nil, err)`

Encode un prompt texte en vecteur (dimension dépend de la config/modèle).

### `Mimir.Model.forward_prompt_image_seed(text_vec, image_vec, seed, training?: bool=false)`

Forward multi-input spécialisé.

### `Mimir.Model.hardware_caps() -> table`

### `Mimir.Model.set_hardware(enabled: bool)`

Active/désactive certains chemins d’accélération CPU.

## Legacy

### `Mimir.Model.infer(prompt: string) -> string | nil`

Chemin d’inférence historique ; pas recommandé pour les workflows modernes.

## Exemple minimal (forward)

```lua
local cfg, err = Mimir.Architectures.default_config("transformer")
assert(cfg, err)
cfg.seq_len = 16
cfg.vocab_size = 256

assert(Mimir.Model.create("transformer", cfg))
assert(Mimir.Model.build())
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 0))

local ids = {}
for i = 1, cfg.seq_len do ids[i] = 1 end
local out, ferr = Mimir.Model.forward({ __input__ = ids }, false)
assert(out, ferr)
print("out_len=", #out)
```
