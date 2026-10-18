# Développement du framework

Cette section s'adresse aux personnes qui modifient le moteur C++, les modèles,
les runtimes ou les bridges de scripting. Elle décrit les contrats à préserver
et relie chaque mécanisme à ses sources de vérité.

> **Limitation**
> Lua est le bridge de référence pour le scripting métier. Les bridges
> JavaScript, C# et Rust restent partiels tant que leur parité avec Lua n'est
> pas explicitement validée.

## Parcours recommandé

1. [Comprendre le fonctionnement du framework](01-How-The-Framework-Works.md)
2. [Construire des modèles et des layers](02-Building-Models-And-Layers.md)
3. [Comprendre la configuration et le registre](03-Config-And-Registry.md)
4. [Développer un runtime](04-Runtime-Development.md)
5. [Préserver le contrat de scripting](05-Scripting-System-Contract.md)

Exécutez les tests associés à un composant avant et après toute modification.
Un backend qui accepte un type de layer doit soit l'exécuter correctement, soit
le refuser afin que le routeur puisse essayer le backend suivant.

## Modèles et configuration

- [Construction de modèles et de layers](02-Building-Models-And-Layers.md)
- [Configuration et registre](03-Config-And-Registry.md)
- [Ajouter une architecture et ses outils](../06-Contributing/02-New-Architecture-And-Tools.md)

## Runtimes

- [Développement d'un runtime](04-Runtime-Development.md)
- [Internals des runtimes GPU](../04-Architecture-Internals/21-GPU-Runtimes.md)
- [Tutoriel : modifier ou ajouter un runtime](../08-Tuto/04-Tuto-Modifier-Ou-Ajouter-Runtime.md)
- [Tutoriel : ajouter une opération](../08-Tuto/05-Tuto-Ajouter-Op.md)

## Scripting et visualisation

- [Contrat du système de scripting](05-Scripting-System-Contract.md)
- [Étendre le visualizer](06-Visualizer-Tips-And-Features.md)
- [Runbook de débogage du visualizer](07-Visualizer-Debug-Runbook.md)

## Sources de vérité

| Domaine | Sources principales |
| --- | --- |
| Modèle | `src/Model.hpp`, `src/Model.cpp` |
| Registre | `src/Models/Registry/ModelArchitectures.hpp`, `ModelArchitectures.cpp` |
| Runtime | `src/runtimes/AbstractRuntime.hpp`, `RuntimeRouter.cpp` |
| Contrat de scripting | `src/scriptings/ScriptingContext.hpp`, `ScriptingRuntime.hpp` |
| Bridge Lua | `src/scriptings/Lua/luaScripting/` |

## Étapes suivantes

Consultez l'[index des internals](../04-Architecture-Internals/00-Internals-Index.md)
pour étudier l'implémentation composant par composant, ou le
[guide de contribution](../06-Contributing/01-Contributing.md) avant de
préparer une modification.
