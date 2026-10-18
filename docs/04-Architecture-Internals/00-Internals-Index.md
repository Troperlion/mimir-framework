# Internals du framework

Cette section décrit l'implémentation C++ de Mímir : stockage des tenseurs,
exécution des layers, autograd, mémoire, sérialisation, registre et runtimes.
Elle est destinée aux développeurs qui doivent vérifier ou modifier le
comportement du moteur.

## Moteur

- [Vue d'ensemble](01-Engine-Overview.md)
- [Classe `Model`](10-Model-Class.md)
- [Stockage des tenseurs](12-Tensor-Storage.md)
- [Layers et opérations](14-Layers-And-Ops.md)
- [Autograd et gradients](13-Autograd-Gradients.md)
- [Planification](22-Planning.md)

## Mémoire

- [Gestion de la mémoire](02-Memory.md)
- [AdvancedRAMManager](05-AdvancedRAMManager.md)
- [Runtime allocator et scratchpads](18-RuntimeAllocator-And-Scratchpads.md)

## Runtimes

- [Backends matériels](03-Hardware-Backends.md)
- [Runtimes GPU](21-GPU-Runtimes.md)
- [Monitoring, Htop et visualizer](04-Monitoring-Htop-Visualizer.md)

## Modèles et services

- [Registre et builders](19-Models-Registry-And-Builders.md)
- [Sérialisation](15-Serialization-Internals.md)
- [Tokenizer et encodeur](16-Tokenizer-Encoder-Internals.md)
- [Bindings Lua](17-Lua-Bindings-Internals.md)
- [Entrées CLI](20-CLI-EntryPoints.md)
- [Helpers C++](11-Helpers.md)

## Lire une page interne

Chaque page distingue autant que possible :

- la source de vérité ;
- les responsabilités du composant ;
- le flux d'exécution ;
- les invariants et formes de tenseurs ;
- les fallbacks ;
- les limites connues ;
- les tests qui couvrent le comportement.

Les structures historiques encore utilisées sont signalées comme telles. Une
capacité déclarée par un backend est distinguée d'un kernel matériel réellement
spécialisé.

## Étapes suivantes

Pour une modification guidée, utilisez le
[guide développeur](../07-Devs/00-INDEX.md). Pour ajouter un composant, consultez
les [tutoriels moteur et runtime](../08-Tuto/00-INDEX.md).
