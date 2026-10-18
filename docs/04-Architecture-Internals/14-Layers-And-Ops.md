# Layers, opérations et disposition des poids

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.


Cette page documente l’API interne des layers : la structure `Layer`, le registre de types, les conventions de routing `inputs/output`, et les implémentations de forward/backward.

Source de vérité :

- Enum & mapping : `src/LayerTypes.hpp`
- Structure `Layer` : `src/Layers.hpp`
- Forward ops (principal) : `src/runtimes/cpu/LayerOps.hpp`
- Forward ops (extensions) : `src/runtimes/cpu/LayerOpsExt.hpp`
- Dispatch runtime : `src/Model.cpp` (switch `layer.type_enum`)

## Sur cette page

- [0) TL;DR](#0-tldr)
- [1) LayerType et normalisation de type](#1-layertype-et-normalisation-de-type)
- [2) struct Layer : champs “universels”](#2-struct-layer-champs-universels)
- [3) Helpers dans Layer](#3-helpers-dans-layer)
- [4) LayerOps : conventions et performance](#4-layerops-conventions-et-performance)
- [5) Attention : layout des poids (runtime)](#5-attention-layout-des-poids-runtime)
- [6) LayerOpsExt : extensions et statut](#6-layeropsext-extensions-et-statut)
- [7) Backward : où regarder](#7-backward-où-regarder)
- [8) Checklist lorsque vous ajoutez un nouveau layer](#8-checklist-lorsque-vous-ajoutez-un-nouveau-layer)
- [Étapes suivantes](#étapes-suivantes)

## 0) TL;DR

- Un `Layer` = un nœud du graphe, avec un `type_enum` (canonique) + des champs de config (dimensions, hyperparams).
- Le runtime dispatch en `switch (LayerType)` et appelle `LayerOps::*_forward`.
- Les poids d’un layer sont stockés dans **un bloc unifié** (`Layer::weight_block`) quand disponible, sinon fallback `weights/bias`.
- Le routing (graphe non-linéaire) se fait via `Layer.inputs` / `Layer.output` et un `TensorStore` interne “par nom”.

## 1) `LayerType` et normalisation de type

Dans `src/LayerTypes.hpp` :

- `enum class LayerType` liste les types supportés (Conv/Linear/Norm/Activation/Attention/Shape ops/…)
- `LayerRegistry::normalize_type(string)` gère les alias (ex: `Swish` → `SiLU`, `BN` → `BatchNorm2d`, `ConvTranspose` → `ConvTranspose2d`, …)
- `LayerRegistry::string_to_type(str)` donne l’enum.

Invariant : un type inconnu doit finir en `LayerType::UNKNOWN` et être rejeté lors de la construction (`Model::push`).

## 2) `struct Layer` : champs “universels”

Définition : `src/Layers.hpp`.

Champs structurants :

- `name` : identifiant stable.
- `type` : string normalisée (compat).
- `type_enum` : version canonique.
- `params_count` : taille du bloc de poids attendu.

Stockage des paramètres :

- `tensor* weight_block` : bloc unifié (recommandé).
- `std::vector<float> weights/bias` : fallback compat.
- `std::vector<float> grad_weights/grad_bias` : gradients.

Dimensions (extraits, non exhaustif) :

- Linear : `in_features`, `out_features`, `use_bias`, `seq_len` (mode séquence)
- Conv2d : `in_channels`, `out_channels`, `input_height`, `input_width`, `kernel_size`, `stride`, `padding`
- Embedding : `vocab_size`, `embed_dim`, `padding_idx`
- Attention : `seq_len`, `embed_dim`, `num_heads`, `causal`

Routing :

- `inputs : vector<string>` (vide ⇒ `{ "x" }`)
- `output : string` (vide ⇒ `"x"`)

## 3) Helpers dans `Layer`

- `getWeights()` / `getWeightsSize()` :
  - si `weight_block` est présent : pointer + taille = `params_count` (hot-path).
  - sinon : pointer = `weights.data()`.

- `get_stride_h/get_stride_w` :
  - gère la compat entre `stride` et `stride_h/stride_w`.

## 4) `LayerOps` : conventions et performance

### 4.1 Linear

Implémentation : `LayerOps::linear_forward`.

- Poids = matrice `[out_f, in_f]` *row-major par sortie*.
- Bias optionnel stocké juste après les poids :
  - `bias = weights + in_f*out_f`.

Support “séquence” :

- si `layer.seq_len > 0` et `input.size() == seq_len * in_f` :
  - l’entrée est interprétée comme `[seq_len, in_f]` aplati,
  - la sortie est `[seq_len, out_f]` aplati.

Optimisations :

- AVX2/FMA si dispo (`__AVX2__`), OpenMP si taille grande.

### 4.2 Ops élément-wise

- `Add` : support broadcast minimal (scalaire, répétition si divisible).
- `Multiply/Subtract/Divide` : tailles égales (Divide protège division par ~0).

### 4.3 Concat/Split/Chunk/Stack

`LayerOps::concat_forward` concatène simplement les buffers 1D (actuellement l’`axis` est surtout informatif).

`LayerOpsExt` contient des implémentations d’ops manquantes et des simplifications (1D/axis=0), utiles pour compléter le runtime.

## 5) Attention : layout des poids (runtime)

Pour `SelfAttention` / `MultiHeadAttention` :

- le runtime attend un bloc de poids contenant `Wqkv` puis `Wout`.
- tailles typiques :
  - `Wqkv` : `3 * E * E`
  - `Wout` : `E * E`
  - total : `4 * E * E`

Dans certains chemins (forward int), le runtime reconstruit des `std::vector` temporaires `qkv_weight/out_weight` à partir du bloc.

⚠️ Implication : `params_count` doit être configuré correctement, sinon lecture hors limites.

## 6) `LayerOpsExt` : extensions et statut

Ce fichier contient des layers déclarés dans `LayerType` mais pas toujours dispatchés dans `Model.cpp`.

Exemples :

- activations : `LeakyReLU`, `Mish`, `Softplus`, `HardSigmoid`, `HardSwish`
- élément-wise : `Subtract`, `Divide`
- shape ops : `Squeeze`, `Unsqueeze`
- tensor ops : `Chunk`, `Stack`

La présence dans `LayerOpsExt.hpp` ne garantit pas que le runtime principal appelle ces fonctions : il faut aussi un `case LayerType::...` dans `Model.cpp`.

## 7) Backward : où regarder

- Backward principal : `Model::backwardPass` dans `src/Model.cpp`.
- Certains backward “non-triviaux” (attention/conv) sont implémentés in situ.

Conventions :

- gradients d’un layer : écrits dans `layer.grad_weights` (taille = `getWeightsSize()`), et éventuellement `layer.grad_bias`.
- routing des gradients : via un `grad_store` par nom de tensor (miroir du forward).

## 8) Checklist lorsque vous ajoutez un nouveau layer

1) Ajouter l’enum dans `LayerType` (si absent).
2) Ajouter mapping string↔enum dans `LayerRegistry::string_to_type`.
3) Ajouter les champs nécessaires dans `Layer` (si pas déjà couvert par les champs “universels”).
4) Implémenter `LayerOps::<layer>_forward` (ou dans `LayerOpsExt`).
5) Ajouter le `case LayerType::<...>` dans le switch du forward runtime (et backward si nécessaire).
6) Définir le layout de poids + formule `params_count` et documenter.

## Étapes suivantes

- [Page précédente : Internals : Autograd + gradients + backward (C++)](13-Autograd-Gradients.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals : sérialisation (SafeTensors / RawFolder / DebugJson)](15-Serialization-Internals.md)
