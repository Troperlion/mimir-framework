# Autograd, gradients et passe arrière

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.


Cette page documente le système de gradients et le backward pass dans Mímir.

L’objectif est pragmatique : expliquer ce qui est *vraiment* supporté, comment les gradients sont stockés, et quelles informations doivent être snapshotées pendant le forward.

Source de vérité :

- Déclarations : `src/Model.hpp` (API training/gradients/optimizer)
- Orchestration du graphe : `src/Model.cpp` (`Model::backwardPass`, `zeroGradients`, `getGradients`)
- Dispatch forward/backward : `src/runtimes/RuntimeRouter.hpp`
- État sauvegardé et activations intégrées : `src/runtimes/AbstractRuntime.hpp`
- Backward natif OpenCL/Vulkan : `src/runtimes/portable/PortableRuntime.cpp` et kernels `Portable*Ops.hpp`
- Dispatch CPU partagé : `src/runtimes/cpu/RuntimeLayerDispatch.hpp`
- Types et pertes autograd : `src/Autograd.hpp` ; adaptateurs de layers : `src/runtimes/Autograd.cpp`
- Layout poids & champs layer : `src/Layers.hpp`
- Primitives mathématiques : `src/runtimes/LayerOps.hpp`, `src/runtimes/cpu/LayerOps.hpp` et `src/runtimes/cpu/LayerOpsExt.hpp`

## Sur cette page

- [1) Vue rapide (Lua → C++)](#1-vue-rapide-lua-c)
- [2) Deux notions à distinguer](#2-deux-notions-à-distinguer)
- [3) “Forward state” : pourquoi il faut snapshot](#3-forward-state-pourquoi-il-faut-snapshot)
- [4) Routing des gradients : gradstore (par nom)](#4-routing-des-gradients-gradstore-par-nom)
- [5) Ce qui est supporté (exemples concrets)](#5-ce-qui-est-supporté-exemples-concrets)
- [6) Invariants et garde-fous](#6-invariants-et-garde-fous)
- [7) Autograd.hpp : ce que c’est (et ce que ce n’est pas)](#7-autogradhpp-ce-que-cest-et-ce-que-ce-nest-pas)
- [8) Debug : comment vérifier que le backward “fait quelque chose”](#8-debug-comment-vérifier-que-le-backward-fait-quelque-chose)
- [Étapes suivantes](#étapes-suivantes)

## 1) Vue rapide (Lua → C++)

| Appel Lua | Binding C++ | Cible | Effet |
|---|---|---|---|
| `Mimir.Model.zero_gradients()` | `LuaScripting::lua_zeroGradients` | `Model::zeroGradients` | Met tous les gradients à zéro + invalide l’état forward. |
| `Mimir.Model.backward(loss_grad)` | `LuaScripting::lua_backwardPass` | `Model::backwardPass` | Traverse le graphe et délègue les dérivées au dispatcher. |
| `Mimir.Model.get_gradients()` | `LuaScripting::lua_getGradients` | `Model::getGradients` | Exporte un dictionnaire d’index→valeur (format plat). |

Notes :

- Les dérivées des layers sont implémentées dans les runtimes. Une opération refusée par tous les runtimes provoque une erreur explicite.
- Les utilitaires de pertes restent dans `src/Autograd.hpp`. Les adaptateurs GELU, LayerNorm et résiduel passent par `RuntimeRouter` et nécessitent un routeur configuré.

## 2) Deux notions à distinguer

### A) Gradients “par layer” (principal)

Chaque `Layer` stocke :

- `grad_weights` : gradient du bloc compact de paramètres, biais inclus.
- `grad_bias` : vue de compatibilité pour les appels directs aux runtimes.

Les runtimes accumulent ces buffers. `Model::backwardPass` efface ensuite la vue de compatibilité pour éviter de compter deux fois les biais dans les normes, le clipping et les gradients exportés.

### B) Gradients “plats” (`Gradients`)

`struct Gradients` (dans `src/Autograd.hpp`) stocke :

- `std::unordered_map<size_t, float> param_grads`

Dans le code actuel, `Model::getGradients()` reconstruit un index plat en itérant les layers et en appendant `grad_weights` puis `grad_bias`.

Conséquence :

- ce format est pratique pour Lua/serialization/debug,
- mais l’optimizer moderne du runtime travaille surtout avec les buffers des layers (ou des blocs par pointeur de paramètre).

## 3) “Forward state” : pourquoi il faut snapshot

Pour faire le backward correctement, il faut les *inputs* du forward (par layer) et parfois des masques (dropout/relu).

Dans `src/Model.cpp`, quand `training==true`, le forward capture dans une structure interne (`forward_state`) :

- `layer_input_names` : quels noms de tenseurs ont été lus (`Layer.inputs` ou `{ "x" }`).
- `layer_input_sizes_multi` : tailles des inputs (utile même sans copier les valeurs).
- `layer_inputs_multi` : copie des valeurs **uniquement** pour certains types (`needs_input_value_snapshot`).
- `layer_output_masks` : masques de sortie (Dropout) pour ne pas recopier toute la sortie.
- parfois `layer_outputs` (best-effort selon path).

**Piège** : le store de tenseurs est “par nom” (`TensorStore`). Si un nom est réutilisé, relire le store en backward peut donner des données différentes de celles du forward. D’où le snapshot.

## 4) Routing des gradients : `grad_store` (par nom)

Le backward ne propage pas uniquement un vecteur unique “dx”. Il route des gradients par nom, miroir du forward.

Pattern :

- initialisation : `grad_store["x"] = loss_gradient`
- pour chaque layer (ordre inverse) :
  - on lit `grad_out = grad_store[layer.output]`
  - on consomme cette version du gradient avant toute accumulation (important pour `x → x`)
  - le dispatcher calcule les gradients vers les inputs
  - on accumule dans `grad_store[input_name]`

Accumulation = somme, avec vérification de taille.

## 5) Ce qui est supporté (exemples concrets)

### Ops sans paramètres

- `Identity`, `Reshape` : gradient recopié (si tailles compatibles).
- `Add` : support du broadcast (même logique que `LayerOps::add_forward`), avec réduction du gradient pour l’entrée “petite”.
- `Concat` : découpe du gradient en tranches.
- `Split` et `Chunk` : recollent les gradients des sorties `base_i`, avec zéro pour les branches non utilisées.
- `Multiply`, `Subtract`, `Divide` : élément-wise (avec règles de base).

### Ops avec paramètres

- `Embedding` :
  - input snapshoté = ids stockés en float (dans le path tokens).
  - `grad_weights` accumule par index de vocab.
  - pas de gradient d’entrée float utile (ids).

- `LayerNorm` :
  - le runtime CPU recalcule mean/var et accumule `dgamma/dbeta` si `affine`.

- `Conv2d` : forward et backward CPU tuilés (im2col + GEMM), avec voie SIMD ou scalaire dans `src/runtimes/cpu/ConvolutionKernels.hpp`. Les tests numériques couvrent stride et dilation.
- Dropout, Dropout2d et AlphaDropout sauvegardent le masque réellement tiré, y compris lorsque l’entrée vaut zéro. Le backward d’entraînement refuse un masque manquant.

### Attention

Le backward attention est implémenté dans le dispatcher CPU partagé. Les poids sont layoutés en blocs (`Wqkv`, `Wout`) et les gradients sont écrits dans `grad_weights`.

### Réparamétrisation VAE

`Reparameterize` utilise deux entrées, `mu` et `logvar`. En entraînement stochastique, le forward calcule :

```text
z = mu + exp(0.5 * clamp(logvar, -20, 20)) * epsilon
```

Le forward state conserve `z`. Le runtime reconstruit ensuite :

```text
epsilon = (z - mu) / exp(0.5 * clamp(logvar, -20, 20))
```

puis calcule :

```text
grad_mu     = grad_z
grad_logvar = grad_z * 0.5 * epsilon * exp(0.5 * logvar)
```

Le facteur de clamp vaut zéro hors de `[-20,20]`. Sans snapshot exploitable, le fallback déterministe propage uniquement `grad_mu`.

### Paramètres sans entrée (`Constant`)

Une `Constant` est fixe par défaut. Le champ `Layer::trainable_parameter` permet d’en faire explicitement un paramètre appris :

- son output est son bloc de poids ;
- elle n’a aucun gradient d’entrée ;
- son gradient de poids est exactement le gradient amont ;
- l’optimizer step la traite comme les autres blocs paramétrés.

Cette distinction est utilisée par `vae_conv/z_prior_bias`. Elle évite de rendre apprenables toutes les constantes structurelles du graphe.

## 6) Invariants et garde-fous

- Si `Model::freezeParameters(true)` :
  - `backwardPass`, `zeroGradients`, `initializeWeights`, `optimizerStep` doivent refuser (exception) ou no-op.

- Si `forward_state.is_valid == false` :
  - le backward refuse (message “call forwardPass() in training mode first”).

- `grad_weights` doit avoir la même taille que `getWeightsSize()` pour éviter overflow.

## 7) Autograd.hpp : ce que c’est (et ce que ce n’est pas)

`src/Autograd.hpp` contient des briques math (ex: `mse_backward`, `gelu_backward`, `layernorm_backward`) et des structures (`ComputationGraph`, `Gradients`).

Dans le runtime actuel :

- `Model::backwardPass` orchestre les snapshots, les branches nommées et l’accumulation ; les runtimes calculent les dérivées.
- `Autograd.hpp` sert de bibliothèque d’outils et de types (notamment `Gradients`).

## 8) Debug : comment vérifier que le backward “fait quelque chose”

- Exécuter les tests numériques et de contrat :

```bash
ctest --test-dir build --output-on-failure \
  -R 'AutogradTest\.|ModelTest.VAEConvContract|RuntimeTest.Math|RuntimeTest.*BackwardParity'
```

- Vérifier :
  - `zeroGradients()` est appelé avant le step,
  - après backward, certains `grad_weights` ne sont pas tous zéro,
  - l’optimizer step change effectivement les poids.

Pour VAEConv, `ModelTest.VAEConvContract` vérifie aussi qu’un gradient de reconstruction traverse le décodeur et atteint le prior appris.

## Couverture OpenCL et Vulkan

Les backends Vulkan et OpenCL conservent leurs chemins spécialisés et utilisent
`PortableRuntime` pour les autres opérations et les passes arrière. Leur
couverture par type correspond au dispatcher CPU exécutable : **72 types en
forward et 73 en backward**, dont NMS avec stop-gradient. `Lambda` est refusé
en forward par le dispatcher CPU lui-même ; son backward identité est supporté
sur CPU, Vulkan et OpenCL. CUDA possède les mêmes familles de kernels mais
n'implémente pas le backward `Lambda`.

Les familles couvertes comprennent convolutions, normalisations, attention,
projections/embeddings, pooling, transformations de tenseurs, récurrence,
upsampling, dropout et reparamétrisation. Les configurations, formes de tenseurs
et limites du device restent vérifiées ; cette couverture par type ne garantit
pas que toute forme soit acceptée.

Les nouveaux kernels de référence OpenCL/Vulkan exécutent leur calcul sur le
device avec une invocation qui traite le travail logique séquentiellement.
Ils privilégient la conformité numérique ; leur couverture ne garantit pas des
performances supérieures au CPU. Le staging des tenseurs et l'accumulation des
gradients de paramètres retournés restent sur l'hôte. Les calculs forward et
backward ne sont pas remplacés par un appel arithmétique CPU caché.

Pour NMS, le backward restitue des gradients nuls de la forme de chaque entrée.
Il vérifie la longueur du gradient amont contre le nombre d'indices retenus.
CPU, CUDA, OpenCL et Vulkan utilisent ce contrat de stop-gradient ; aucune
surrogate différentiable n'est ajoutée aux frontières de sélection.

Les contextes de forward conservent le masque de dropout et les activations
nécessaires à la reparamétrisation pour reproduire le backward CPU.
`RuntimeTest.OPENCLNativeOps` et `RuntimeTest.VULKANNativeOps` vérifient les
valeurs, les gradients de paramètres et d'entrées, leur accumulation et le
runtime sélectionné. `RuntimeTest.NMS` vérifie le CPU ; les tests existants
`RuntimeTest.OPENCLBackwardParity` et `RuntimeTest.VULKANBackwardParity`
conservent la couverture des familles historiques. Les tests GPU ignorent
l'exécution avec le code 77 si le device est indisponible.

## Étapes suivantes

- [Page précédente : Internals : stockage `tensor` + allocation dynamique (C++)](12-Tensor-Storage.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals : layers, `LayerType`, `LayerOps` et layouts de poids (C++)](14-Layers-And-Ops.md)
