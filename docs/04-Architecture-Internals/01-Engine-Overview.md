# Moteur d’exécution

Comprendre le fonctionnement interne exact des composants runtime.

**Public concerné :** Développeur avancé qui modifie le moteur C/C++.

> **Prérequis**
>
> Connaître les bases C++ et la structure du dépôt.

## Diagrammes d'explication

![Runtime graphe](../graphs/07_graph_runtime.svg)

![Flux de données runtime](../graphs/27_runtime_dataflow.svg)


Cette page décrit le flux d’exécution réel du runtime (côté C++), du point de vue de `Model` : comment les entrées sont injectées, comment les tenseurs sont routés, et comment les layers sont évalués.

Sources principales :

- `src/Model.cpp` (forward, stores, mode training)
- `src/Layers.hpp` (structure d’un layer : type, entrées/sorties, poids)
- `src/LayerTypes.hpp` (enum des types + normalisation d’alias)
- `src/runtimes/cpu/RuntimeAllocator.hpp` (gestion strict/mémoire pour les activations)

## Vue d’ensemble

Le runtime exécute un modèle comme une *liste ordonnée* de layers. Le mécanisme clé est le **TensorStore** : un dictionnaire `nom -> Tensor` qui permet de chaîner les opérations sans construire un graphe dynamique complexe.

Deux stores coexistent :

- store **float** : activations principales et sorties des layers
- store **int** : entrées discrètes / identifiants (principalement tokens)

L’idée : certains layers (ex: `Embedding`) consomment des tenseurs int et produisent des tenseurs float. La majorité des layers consomment/produisent du float.

## Convention d’entrées et de noms

Le modèle accepte des entrées nommées (côté C++ et bindings Lua). Les architectures s’appuient sur des noms conventionnels, par exemple :

- `x` : activation courante principale
- `__input__` : copie “référence” de l’entrée quand on veut la garder accessible
- `text_ids`, `input_ids`, etc. : tenseurs int de tokens (selon l’architecture)

Chaque layer a typiquement :

- une liste `input_names` (0, 1 ou plusieurs)
- un champ `output` (nom sous lequel stocker le résultat)

Si un layer a `input_names` vide, le runtime applique souvent une valeur par défaut (classiquement `x`), mais ce détail dépend du chemin de code exact et du type de layer.

## Étapes d’un forward (haut niveau)

1) **Reset des stores**

Le forward commence en nettoyant les stores (au minimum les activations temporaires) pour éviter d’accumuler des tensors d’un appel à l’autre.

1) **Injection des entrées**

Les entrées passées à `forward` sont enregistrées dans le store adéquat :

- float -> store float
- int / tokens -> store int

Dans le code, vous verrez des helpers du style `storeTensor(...)` et `storeTensorInt(...)` utilisés pour maintenir cette séparation.

1) **(Optionnel) Injection de conditionnements**

Certaines architectures injectent des tenseurs additionnels (conditionnement type SDXL/UNet, etc.). Typiquement, ça se fait en écrivant dans le store sous des clés attendues par les layers suivants.

1) **Création d’un allocateur runtime**

Pour les activations/intermédiaires, le runtime utilise un gestionnaire mémoire strict (`RuntimeAllocator`) branché sur `MemoryGuard`.

Objectif : éviter les allocations “sauvages” hors contrôle et rendre les OOM explicites (exceptions/erreurs).

1) **Exécution séquentielle des layers**

Pour chaque layer (dans l’ordre) :

- résoudre ses entrées : pour chaque `input_name`, lire un tensor du store (float ou int selon le type du layer)
- exécuter l’opération (dispatch par `LayerType`)
- écrire le résultat sous `layer.output` dans le store float (la plupart du temps)

Le runtime supporte le **multi-input** en passant un vecteur d’entrées à certaines opérations (concat/add/attention/cross-attn, etc.).

1) **Sortie**

La sortie “finale” dépend de la convention de l’architecture :

- soit la sortie du dernier layer
- soit un tensor explicite (par exemple un nom attendu)

## Cas particulier : chemin “tokens/int”

Dans le forward “tokens”, on observe (dans `Model.cpp`) une séparation stricte :

- les IDs (tokens) sont stockés dans le store int
- `Embedding` lit dans le store int et produit un tensor float

C’est ce qui permet de décrire proprement des modèles NLP : `input_ids -> Embedding -> …`.

## Mode training : snapshots et backward

Quand `training=true`, le forward peut capturer des **snapshots** utiles au backward :

- shapes/tailles des entrées/sorties
- certaines valeurs (selon le type de layer et les options)
- des masques (ex: dropout) nécessaires pour reproduire exactement le forward au backward

Le but : rendre le backward déterministe et éviter de recalculer certains intermédiaires coûteux.

La quantité de données stockée dépend fortement de la stratégie choisie dans le code (certaines ops stockent plus, d’autres moins), et du besoin réel du backward.

## Attention (implémentation)

Les variants d’attention (self / multi-head / cross) existent côté runtime, mais il faut garder à l’esprit qu’il s’agit d’implémentations “framework” : elles ne visent pas forcément les mêmes performances que des kernels GPU spécialisés.

## Lien avec le registre d’architectures

Le registre d’architectures (`ModelArchitectures`) construit des graphes “valides” en :

- choisissant des noms d’entrées/sorties cohérents entre layers
- fixant les champs obligatoires (dimensions, poids, paramètres)
- produisant une `modelConfig` sérialisable (voir la page API dédiée)

## Étapes suivantes

- [Page précédente : Internals du framework](00-Internals-Index.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Internals - mémoire](02-Memory.md)
