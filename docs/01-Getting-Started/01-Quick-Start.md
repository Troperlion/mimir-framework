# Première exécution

Valider rapidement le chemin minimal: build, script Lua, création de modèle.

**Public concerné :** Débutant à intermédiaire.

> **Prérequis**
>
> Le dépôt est cloné localement.

Objectif : exécuter un script Lua, créer un modèle via le registre, initialiser les poids, faire un forward.

Si vous ne savez pas par où commencer, suis cette page de haut en bas sans improviser. Elle couvre le plus petit chemin utile pour vérifier que le runtime, l’API Lua et le registre d’architectures fonctionnent ensemble.

## Plan de progression

1. Vérifier que le binaire démarre.
2. Vérifier qu'un script Lua s'exécute.
3. Vérifier qu'un modèle peut être créé et exécuté.
4. Vérifier que la sérialisation fonctionne.

Temps moyen : 10 à 15 minutes.

## Pré-requis

- Linux
- CMake + compilateur C++17

## 1) Build

```bash
cmake -S . -B build
cmake --build build -j
```

Si le build passe mais que vous voulez un contrôle rapide avant de lire le reste de la doc :

```bash
./bin/mimir --help
```

Vous devez obtenir l’aide CLI et la détection matérielle au démarrage, sans crash immédiat.

Si la commande échoue :

1. Vérifier que vous êtes bien à la racine du dépôt.
2. Supprimer `build/` puis relancer `cmake -S . -B build`.
3. Relire les erreurs CMake avant de relancer le build.

## 2) Exécuter un script Lua

```bash
./bin/mimir --lua scripts/examples/inspect_vae_conv.lua
```

Cet exemple vérifie le câblage CLI → Lua, le registre et l’allocation d’un modèle sans charger de dataset. Pour une validation plus large :

```bash
./bin/mimir --lua scripts/templates/template_new_model.lua
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
```

Le template vérifie le chemin “création de modèle”, le smoke test vérifie le chemin “save/load”. Ensemble, ils donnent un meilleur signal qu’un simple `--help`.

Attendu :

1. Pas de crash.
2. Messages de run cohérents.
3. Retour shell à 0.

## 3) Créer un modèle (via le registre)

Exemple : Transformer simple.

```lua
local cfg, err = Mimir.Architectures.default_config("transformer")
if not cfg then error(err) end

cfg.vocab_size = 8000
cfg.seq_len = 128
cfg.d_model = 256
cfg.num_layers = 4
cfg.num_heads = 8
cfg.mlp_hidden = 1024
cfg.output_dim = 256

assert(Mimir.Model.create("transformer", cfg))
-- Model.build() n'est plus nécessaire (v3.0+: network construit automatiquement)
assert(Mimir.Model.allocate_params())
assert(Mimir.Model.init_weights("xavier", 42))

-- forward avec ids int (Embedding) si l’archi le supporte,
-- sinon forward float: utilise __input__.
local ids = {}
for i = 1, cfg.seq_len do ids[i] = 1 end
local out = Mimir.Model.forward({ __input__ = ids }, false)
print("out_len=", #out)
```

Pourquoi cet exemple est structuré comme ça :

- `default_config()` évite d’oublier des champs attendus par le builder.
- `create()` fixe le type et la config dans le contexte runtime, puis construit le réseau automatiquement (v3.0+).
- `allocate_params()` réserve les poids.
- `init_weights()` met le modèle dans un état exploitable avant le premier `forward()`.

Si vous remplacez l’entrée par une table map nommée, vous gardez un script compatible avec les architectures multi-input. C’est la convention recommandée même pour un seul tenseur d’entrée.

Astuce : si vous voulez un premier essai encore plus court, ne change que `seq_len` et garde le reste par défaut.

### Variante courte avec sauvegarde

```lua
local ok, save_err = Mimir.Serialization.save("checkpoint/quickstart_model", "safetensors")
assert(ok ~= false, save_err)
print("checkpoint écrit")
```

Cette étape est utile si vous voulez valider tout de suite que la config, l’allocation et la sérialisation sont cohérentes dans votre environnement.

## Erreurs fréquentes

1. Oublier `allocate_params()` avant `init_weights()`.
2. Passer une entrée incompatible (ids int attendus vs tenseur float).
3. Croire que `Mimir.Model.build()` est obligatoire (ce n'est plus le cas en mode moderne).

## 4) Où regarder ensuite

- `docs/02-User-Guide/02-Model-Lifecycle.md` pour comprendre le pipeline.
- `docs/03-API-Reference/02-Serialization.md` pour save/load.
- `docs/02-User-Guide/09-Memory.md` si votre premier vrai modèle dépasse vite la RAM disponible.

## Étapes suivantes

- [Page précédente : Démarrage rapide](00-GET-STARTED.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : 🔧 Installation & Compilation détaillée](02-Installation.md)
