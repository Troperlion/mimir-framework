# Internals : bindings Lua (`LuaScripting` / `LuaContext`)

Cette page explique comment l’API Lua est exposée et comment elle appelle le runtime C++.

Source de vérité :

- Déclaration : `src/LuaScripting.hpp`
- Implémentation : `src/LuaScripting.cpp`
- Modèle : `src/Model.hpp`, `src/Model.cpp`
- Registre d’architectures : `src/Models/Registry/ModelArchitectures.hpp/.cpp`

## 1) `LuaScripting` : rôle

`LuaScripting` encapsule :

- l’état `lua_State* L`,
- le chargement/exécution d’un script Lua,
- l’injection d’arguments (équivalent de `arg`),
- l’enregistrement des bindings (fonction `registerAPI`).

## 2) Surface API (catégories)

Le header liste les endpoints principaux :

- **Model API** : create/build/train/infer/save/load + allocate/init/forward/backward/optimizer.
- **Serialization API** : save/load checkpoint + detect format + debug json.
- **Architectures** : `available`, `default_config`.
- **LayerOps** : helpers de calcul (conv/linear/pool/norm/attention) accessibles directement.
- **Viz/Htop** : création, update, rendu.
- **Tokenizer** : create/tokenize/detokenize + vocab/BPE + analyse.
- **Memory** : config, stats, limit, allocator, guard.
- **Dataset** : load/get/prepare.

## 3) `LuaContext` : singleton runtime

Le code définit un singleton `LuaContext` global, accessible depuis les callbacks Lua.

Rôle :

- stocker des objets C++ partagés (modèle courant, tokenizer, encoder, visualizer, dataset, logs…),
- éviter de repasser des pointeurs via la stack Lua pour tout.

ATTENTION : l’API Lua est **stateful**. Beaucoup de fonctions opèrent sur “le modèle courant”.

## 4) Conventions d’IO

- Entrée/sortie par défaut du modèle : `"x"`.
- Alias input immuable : `"__input__"`.
- Forward multi-input : tables Lua transformées en maps C++ (noms → vecteurs).

## 5) Pièges courants

- Appeler `forward` avant `allocate_params/init_weights` renvoie généralement un message d’erreur (poids non alloués).
- En mode htop/viz, écrire sur stdout peut casser le rendu (note dans `LuaScripting.hpp`).
- Les tailles doivent être cohérentes (seq_len, d_model, etc.) sinon le runtime lève des exceptions.