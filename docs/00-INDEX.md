# Documentation Mímir 3.x

Mímir est un framework C++ de construction, d'entraînement et d'exécution de
modèles, pilotable depuis Lua. Sa documentation est progressive : elle permet
de lancer un premier modèle rapidement, puis d'approfondir le moteur, les
runtimes et les extensions sans changer de conventions en cours de route.

> **Note**
> Cette documentation décrit la branche 3.x actuelle. L'ancienne documentation
> est conservée dans `docs_archive/` à titre historique, mais ne doit pas être
> utilisée comme référence d'API.

## Sur cette page

- [Commencer ici](#commencer-ici)
- [Prologue](#prologue)
- [Bien démarrer](#bien-démarrer)
- [Concepts d'architecture](#concepts-darchitecture)
- [Les bases](#les-bases)
- [Modèles](#modèles)
- [Référence de l'API Lua](#référence-de-lapi-lua)
- [Runtimes et matériel](#runtimes-et-matériel)
- [Aller plus loin](#aller-plus-loin)
- [Développement et extension](#développement-et-extension)
- [Tutoriels](#tutoriels)
- [Conventions essentielles](#conventions-essentielles)
- [Étapes suivantes](#étapes-suivantes)

## Commencer ici

Si vous découvrez Mímir, suivez ces pages dans l'ordre :

1. [Installation](01-Getting-Started/02-Installation.md)
2. [Démarrage rapide](01-Getting-Started/00-GET-STARTED.md)
3. [Concepts essentiels](02-User-Guide/01-Core-Concepts.md)
4. [Cycle de vie d'un modèle](02-User-Guide/02-Model-Lifecycle.md)
5. [Scripting Lua](02-User-Guide/06-Lua-Scripting.md)

Pour choisir un parcours adapté à votre expérience, consultez les
[parcours d'apprentissage](01-Getting-Started/06-Learning-Paths.md).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
./bin/mimir --lua scripts/templates/template_new_model.lua
```

## Prologue

- [État réel du projet et matrice de maturité](00-PROJECT-STATUS.md)
- [Philosophie du framework](00-Framework-Philosophy.md)
- [Guide de style de la documentation](00-STYLE.md)
- [Notes de version](../CHANGELOG.md)
- [Guide de contribution](06-Contributing/01-Contributing.md)

## Bien démarrer

- [Démarrage rapide](01-Getting-Started/00-GET-STARTED.md)
- [Première exécution](01-Getting-Started/01-Quick-Start.md)
- [Installation et compilation](01-Getting-Started/02-Installation.md)
- [Interface en ligne de commande](01-Getting-Started/03-CLI.md)
- [Organisation du projet](01-Getting-Started/04-Repo-Layout.md)
- [Valider l'installation](01-Getting-Started/05-Smoketest.md)
- [Parcours par niveau](01-Getting-Started/06-Learning-Paths.md)

## Concepts d'architecture

- [Vue d'ensemble du moteur](04-Architecture-Internals/01-Engine-Overview.md)
- [Cycle de vie d'un modèle](02-User-Guide/02-Model-Lifecycle.md)
- [Classe `Model`](04-Architecture-Internals/10-Model-Class.md)
- [Registre et builders](04-Architecture-Internals/19-Models-Registry-And-Builders.md)
- [Planification de l'exécution](04-Architecture-Internals/22-Planning.md)
- [Autograd et gradients](04-Architecture-Internals/13-Autograd-Gradients.md)

## Les bases

- [Concepts essentiels](02-User-Guide/01-Core-Concepts.md)
- [Données et datasets](02-User-Guide/03-Data.md)
- [Entraînement](02-User-Guide/04-Training.md)
- [Inférence](02-User-Guide/05-Inference.md)
- [Scripting Lua](02-User-Guide/06-Lua-Scripting.md)
- [Tokenizer et encodeur](02-User-Guide/07-Tokenizer-Encoder.md)
- [Configuration pilotée par fichier](02-User-Guide/08-Config-Driven-Scripting.md)
- [Checkpoints](02-User-Guide/08-Checkpoints.md)
- [Gestion de la mémoire](02-User-Guide/09-Memory.md)
- [Scripts et exemples](02-User-Guide/10-Examples.md)
- [Packages d’architecture MPK](02-User-Guide/15-MPK.md)
- [Rapports Show-Graph et métriques](02-User-Guide/16-Show-Graph-Metrics.md)

## Modèles

- [Registre des architectures](03-API-Reference/11-Architectures.md)
- [VAE texte](02-User-Guide/11-VAEText.md)
- [Transformer et GPT](02-User-Guide/12-Transformer-GPT.md)
- [Modèles de diffusion](02-User-Guide/13-Diffusion.md)
- [VAE convolutionnel](02-User-Guide/14-VAEConv.md)

## Référence de l'API Lua

- [Vue d'ensemble](03-API-Reference/00-API-Overview.md)
- [Contrat du système de scripting](03-API-Reference/00-Scripting-Contract.md)
- [Variables globales et alias](03-API-Reference/19-Globals.md)
- [`Mimir.Model`](03-API-Reference/10-Model.md)
- [`Mimir.Architectures`](03-API-Reference/11-Architectures.md)
- [`Mimir.Layers`](03-API-Reference/18-Layers-Module.md)
- [`Mimir.Tokenizer`](03-API-Reference/12-Tokenizer.md)
- [`Mimir.Dataset`](03-API-Reference/13-Dataset.md)
- [`Mimir.IO`](03-API-Reference/21-IO.md)
- [Mémoire](03-API-Reference/14-Memory.md)
- [Visualisation et monitoring](03-API-Reference/15-Viz-Htop.md)
- [Interfaces Viz : SFML, Qt, GTK, Web](02-User-Guide/17-Viz-Backends.md)
- [Sérialisation](03-API-Reference/02-Serialization.md)
- [Variables d'environnement](03-API-Reference/22-Environment-Variables.md)
- [Bridge JavaScript](03-API-Reference/21-JavaScript-API.md)
- [Correspondance Lua vers C++](03-API-Reference/20-Lua-API-Cpp-Mapping.md)

## Runtimes et matériel

- [Backends matériels](04-Architecture-Internals/03-Hardware-Backends.md)
- [Runtimes GPU](04-Architecture-Internals/21-GPU-Runtimes.md)
- [Accélération GPU](05-Advanced/05-GPU-Acceleration.md)
- [Étude du runtime FPGA iCESugar Pro](05-Advanced/06-FPGA-iCESugar-Pro-Runtime-Study.md)
- [Runtime allocator et scratchpads](04-Architecture-Internals/18-RuntimeAllocator-And-Scratchpads.md)
- [Développer un runtime](07-Devs/04-Runtime-Development.md)

Le routeur essaie les backends disponibles dans cet ordre : ROCm, CUDA,
Vulkan, OpenCL, puis CPU. Le support déclaré d'une opération ne garantit pas
qu'elle possède un kernel GPU spécialisé ; les pages runtime distinguent ces
deux notions.

CPU, Vulkan et OpenCL couvrent 72 types exécutables en forward et 73 en
backward. [NMS](03-API-Reference/01-Layers.md#contrat-de-nms) possède un backward
stop-gradient sur CPU, CUDA, Vulkan et OpenCL. Les kernels de référence GPU
complètent la couverture ; ils ne garantissent pas une amélioration de vitesse.

## Aller plus loin

- [Performance](05-Advanced/01-Performance.md)
- [Débogage et stabilité numérique](05-Advanced/02-Debugging.md)
- [État de préparation LLM](05-Advanced/03-LLM-Readiness.md)
- [Carte du code source](05-Advanced/04-Source-Code-Map.md)
- [Internals de la mémoire](04-Architecture-Internals/02-Memory.md)
- [Layers et opérations](04-Architecture-Internals/14-Layers-And-Ops.md)
- [Sérialisation interne](04-Architecture-Internals/15-Serialization-Internals.md)
- [Bindings Lua internes](04-Architecture-Internals/17-Lua-Bindings-Internals.md)

## Développement et extension

- [Guide développeur](07-Devs/00-INDEX.md)
- [Construire des modèles et des layers](07-Devs/02-Building-Models-And-Layers.md)
- [Configuration et registre](07-Devs/03-Config-And-Registry.md)
- [Contrat du système de scripting](07-Devs/05-Scripting-System-Contract.md)
- [Ajouter une architecture et ses outils](06-Contributing/02-New-Architecture-And-Tools.md)
- [Étendre modèles, runtimes et fonctionnalités](06-Contributing/03-Extending-Models-Runtimes-And-Features.md)
- [Ajouter une entrée de scripting](06-Contributing/04-Tutorial-Add-Python-Scripting-Entry.md)
- [Préparer une publication GitHub](06-Contributing/05-Publishing-GitHub.md)

## Tutoriels

- [Tous les tutoriels](08-Tuto/00-INDEX.md)
- [Le framework en trois étapes](08-Tuto/01-Cours-Framework-3-Etapes.md)
- [Ajouter un modèle](08-Tuto/02-Tuto-Ajouter-Modele.md)
- [Écrire un script Lua](08-Tuto/03-Tuto-Coder-Script.md)
- [Ajouter ou modifier un runtime](08-Tuto/04-Tuto-Modifier-Ou-Ajouter-Runtime.md)
- [Ajouter une opération](08-Tuto/05-Tuto-Ajouter-Op.md)
- [Parcours complet](08-Tuto/06-Parcours-Complet-Framework.md)
- [Valider VAEConv sans dataset](08-Tuto/07-Tuto-VAEConv-Sans-Dataset.md)
- [Du registre au checkpoint](08-Tuto/08-Tuto-Registre-Pipeline-Checkpoint.md)
- [Créer et inspecter un package MPK](08-Tuto/09-Tuto-Creer-MPK.md)

## Conventions essentielles

| Nom | Rôle |
| --- | --- |
| `__input__` | Entrée par défaut, sous forme de flottants ou d'identifiants selon l'architecture |
| `text_ids` | Entrée d'identifiants dédiée aux architectures de texte |
| `x` | Sortie principale conventionnelle |

Les sources de vérité sont le
[registre des architectures](../src/Models/Registry/ModelArchitectures.cpp),
les [bindings Lua](../src/scriptings/Lua/luaScripting/LuaScripting.cpp), la
[classe Model](../src/Model.cpp) et les implémentations de
[runtimes](../src/runtimes/).

## Étapes suivantes

Pour une première utilisation, poursuivez avec le
[démarrage rapide](01-Getting-Started/00-GET-STARTED.md). Pour contribuer au
moteur, commencez par la [vue d'ensemble interne](04-Architecture-Internals/00-Internals-Index.md).
