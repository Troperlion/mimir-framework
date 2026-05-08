# API : `Mimir.Architectures`

Le registre d’architectures est la manière recommandée de créer des modèles.

Source : `src/Models/Registry/ModelArchitectures.cpp` et bindings `src/LuaScripting.cpp`.

Ce module fournit :

- une liste des architectures disponibles
- des configs par défaut (faciles à surcharger)
- une normalisation de noms (alias/rétrocompat)
- une construction “safe” de modèles (avec métadonnées standardisées)

## `available() -> table<string> | (nil, err)`

Retourne la liste triée des architectures connues.

Notes :

- La liste provient du registre C++ (`ModelArchitectures::available()`), qui construit la liste à partir des entrées enregistrées.
- Les noms retournés sont les **noms canoniques** (après normalisation), triés.

## `default_config(name: string) -> table | (nil, err)`

Retourne la config par défaut (JSON -> table Lua).

Comportement :

- `name` passe par une phase de **canonicalisation** (voir section Alias).
- si l’architecture est inconnue : erreur.
- la config retournée sert de base : on peut ensuite fournir des overrides à la création.

## `create(name: string, overrides?: table) -> Model | (nil, err)`

Crée un modèle depuis une architecture nommée.

Comportement (côté C++ `ModelArchitectures::Registry::create`) :

1) canonicalise `name` (alias/rétrocompat)
2) charge `default_config`
3) merge récursivement les champs de `overrides` dans la config par défaut
4) impose `cfg["type"] = <nom canonique>`
5) appelle la factory associée
6) écrit `model->modelConfig = cfg` (utile pour inspection/sérialisation)

Le merge est un merge “objet” :

- si `base[key]` et `overrides[key]` sont des objets, merge récursif
- sinon, la valeur override remplace la valeur base

Conséquence : vous pouvez surcharger une sous-clé sans recopier tout le bloc.

Exemple (pseudo-code Lua) :

```lua
local arch = "transformer"
local cfg = Mimir.Architectures.default_config(arch)

-- Override minimal
local m = Mimir.Architectures.create(arch, {
  d_model = 256,
  num_layers = 6,
  causal = true,
})
```

## Alias / rétrocompat

Certains noms historiques sont normalisés (ex: PonyXL). Voir `canonicalArchName` dans le code du registre.

Canonicalisation observée dans le registre C++ :

- anciens noms PonyXL (ex: `ponyxl_ddpm`, `t2i_autoencoder`, `ponyxl_sdxl_stub`, `ponyxl_sdxl_unet2d`) -> `ponyxl_sdxl`
- variantes conviviales `SD3.5` / `sd3.5` / `SD3_5` -> `sd3_5`

Recommandation : utilisez les noms canoniques pour les configs et la sérialisation (ils sont stables), et ne comptez sur les alias que pour charger des anciens scripts.

## Config complète (JSON) : résolution + merge

Le runtime (CLI `--config`) et certaines APIs utilisent une **config complète** qui contient plusieurs sections (tokenizer, viz, lua, etc.).

### Choix de l’architecture

Ordre de recherche :

1) `architecture` (top-level)
2) `type` (top-level)
3) `model.architecture`
4) `model.type`

### Où mettre les overrides modèle

Deux emplacements sont fusionnés dans la config effective du modèle :

- `model` (overrides génériques)
- la section nommée comme l’architecture (ex: `unet`, `vae_conv`, `transformer`)

Le runtime préserve aussi les clés parentes (ex: `tokenizer`, `lua`, `viz`) dans `model->modelConfig` afin que les scripts puissent les relire.
