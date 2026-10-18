# Internals de la sérialisation

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.


Cette page documente l’implémentation de la sérialisation côté C++ (save/load), et comment elle s’interface avec le runtime `Model`.

Source de vérité :

- API haut niveau : `src/Serialization/Serialization.hpp`, `src/Serialization/Serialization.cpp`
- SafeTensors : `src/Serialization/SafeTensorsWriter.hpp` et `src/Serialization/SafeTensorsWriter.cpp`, `SafeTensorsReader.hpp/.cpp`
- Raw folder : `src/Serialization/RawCheckpointWriter.hpp` et `src/Serialization/RawCheckpointWriter.cpp`, `RawCheckpointReader.hpp/.cpp`
- Debug JSON : `src/Serialization/DebugJsonDump.hpp` et `src/Serialization/DebugJsonDump.cpp`
- Checksum/crypto utilitaires : `src/Sha256.hpp` et `src/Sha256.cpp` (si utilisé)
- Modèle (accès weights/layers/tokenizer/encoder/optimizer) : `src/Model.hpp`, `src/Model.cpp`

## 1) Surface API : `Mimir::Serialization`

Dans `src/Serialization/Serialization.hpp` :

- `save_checkpoint(Model&, path, SaveOptions, error*)`
- `load_checkpoint(Model&, path, LoadOptions, error*)`

Formats :

- `SafeTensors` : fichier unique (interop).
- `RawFolder` : dossier + manifest JSON (lisible/debug).
- `DebugJson` : dump debug (write-only, non production).

## 2) Options de save/load : invariants

### `SaveOptions`

Points importants :

- `format` : type de sortie.
- `save_optimizer`, `save_tokenizer`, `save_encoder` : inclusion des états.
- `include_gradients`, `include_optimizer_state`, `include_activations`, `include_checksums`, `include_weight_deltas` : options “Enhanced DebugJson”.
- `include_git_info` : métadonnées de build.

### `LoadOptions`

- `format` : peut être explicitement donné.
- `detect_format(path)` : heuristique (dossier → RawFolder si `manifest.json`, fichier `.safetensors` → SafeTensors, `.json` → DebugJson).
- `strict_mode` : si vrai, l’absence de tenseurs attendus est une erreur.
- `validate_checksums` : si activé, vérifie la cohérence (selon le writer).

## 3) Dispatch réel (implémentation)

Dans `src/Serialization/Serialization.cpp` :

- `save_checkpoint` instancie un writer (`SafeTensorsWriter`, `RawCheckpointWriter`, `DebugJsonDump`).
- `load_checkpoint` instancie un reader (`SafeTensorsReader`, `RawCheckpointReader`).
- `DebugJson` est explicitement “write-only” côté load.

## 4) Qu’est-ce qui est sérialisé exactement ?

Ce point dépend du writer choisi, mais le contrat général est :

- Poids du modèle (par layer / tenseur nommé),
- Tokenizer (si `save_tokenizer`),
- Encoder (si `save_encoder`),
- Optimizer (si `save_optimizer` et support),
- Métadonnées : version Mímir, git commit, config modèle, etc.

**Note pratique** : côté runtime, les poids peuvent être stockés dans des blocs unifiés `Layer::weight_block` (voir `docs/04-Architecture-Internals/12-Tensor-Storage.md`). Le writer doit donc lire via les accesseurs “compatibles sérialisation”.

## 5) DebugJson “enhanced” (v1.4)

Le format debug est conçu pour :

- inspecter rapidement des poids / gradients,
- produire des dumps de compat tests,
- comparer des runs (checksums / deltas).

Depuis v1.3, il embarque aussi `framework_state` (snapshot au moment du dump) :

- runtime/backends + configs,
- capacités CPU,
- état mémoire global (guard/allocator/RAM manager),
- registre des layers supportés,
- état modèle (`default_dtype`, `model_config`, dimensions, etc.).

Le code convertit `SaveOptions` → `DebugJsonOptions` (voir `Serialization.cpp`).

## 6) Checklist : lorsque vous ajoutez un nouveau tensor / nouvel état à sauver

- Choisir une clé stable (nom) compatible entre versions.
- Mettre à jour writer + reader (SafeTensors et/ou RawFolder).
- Mettre à jour le smoke Lua `scripts/tests/test_serialization_smoke.lua` et les
  tests C++ `SerializationTest.*` concernés.
- Si l’état est gros, prévoir options “include_*” plutôt qu’un dump systématique.

## Étapes suivantes

- [Page précédente : Internals : layers, `LayerType`, `LayerOps` et layouts de poids (C++)](14-Layers-And-Ops.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals : Tokenizer / ConditioningEncoder (C++)](16-Tokenizer-Encoder-Internals.md)
