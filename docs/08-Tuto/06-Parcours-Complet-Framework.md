# Parcours complet du framework

Apprendre le framework de bout en bout avec un chemin guide qui couvre la compilation, les donnees, les modeles, l'entrainement, l'inference, les checkpoints, le scripting, le runtime et l'extension du code.

**Public concerné :** - Debutants qui veulent une vue d'ensemble concrete.
- Utilisateurs intermediaires qui veulent passer d'un test simple a un workflow complet.
- Contributeurs qui veulent comprendre ou se situe chaque piece du framework.

> **Prérequis**
>
> 1. Compiler Mimir.
> 2. Verifier le smoketest.
> 3. Ouvrir les pages de reference si besoin:
> - [docs/00-Framework-Philosophy.md](../00-Framework-Philosophy.md)
> - [docs/01-Getting-Started/00-GET-STARTED.md](../01-Getting-Started/00-GET-STARTED.md)
> - [docs/02-User-Guide/01-Core-Concepts.md](../02-User-Guide/01-Core-Concepts.md)

## Vue d'ensemble

![Vue d'ensemble du framework](../graphs/00_framework_overview.svg)

![Architecture en couches](../graphs/01_layered_architecture.svg)

## Parcours recommande

### 1. Demarrage et philosophie

Commence par poser le cadre technique.

1. Lire [docs/00-Framework-Philosophy.md](../00-Framework-Philosophy.md).
2. Lire [docs/01-Getting-Started/00-GET-STARTED.md](../01-Getting-Started/00-GET-STARTED.md).
3. Suivre [docs/01-Getting-Started/01-Quick-Start.md](../01-Getting-Started/01-Quick-Start.md).
4. Lancer [docs/01-Getting-Started/05-Smoketest.md](../01-Getting-Started/05-Smoketest.md).

### 2. Donnees et modalites

Apprends comment Mimir voit un dataset et comment il relie texte, image, audio et video.

1. Lire [docs/02-User-Guide/03-Data.md](../02-User-Guide/03-Data.md).
2. Lire [docs/03-API-Reference/13-Dataset.md](../03-API-Reference/13-Dataset.md).
3. Lire [docs/02-User-Guide/09-Memory.md](../02-User-Guide/09-Memory.md) pour le comportement mémoire et les évictions.
4. Ne charger des données qu'après avoir vérifié le format attendu par le
   script choisi. `Mimir.Dataset.get` expose principalement des chemins et des
   métadonnées ; il ne constitue pas un décodeur multimédia générique.

### 3. Cycle de vie d'un modele

Comprenez le chemin moderne : `create` construit le graphe, `build` est un
no-op de compatibilité, puis `allocate`, `init`, `forward` et `backward`
exécutent le cycle du modèle.

1. Lire [docs/02-User-Guide/02-Model-Lifecycle.md](../02-User-Guide/02-Model-Lifecycle.md).
2. Lire [docs/03-API-Reference/01-Layers.md](../03-API-Reference/01-Layers.md).
3. Lire [docs/03-API-Reference/11-Architectures.md](../03-API-Reference/11-Architectures.md).
4. Lire [docs/03-API-Reference/20-Lua-API-Cpp-Mapping.md](../03-API-Reference/20-Lua-API-Cpp-Mapping.md).

### 4. Entrainement et inference

Passe du build au run reproductible.

1. Lire [docs/02-User-Guide/04-Training.md](../02-User-Guide/04-Training.md).
2. Lire [docs/02-User-Guide/05-Inference.md](../02-User-Guide/05-Inference.md).
3. Lire [docs/02-User-Guide/08-Checkpoints.md](../02-User-Guide/08-Checkpoints.md).
4. Commencer par le tutoriel sans dataset
   [du registre au checkpoint](08-Tuto-Registre-Pipeline-Checkpoint.md).
5. Effectuer un entraînement seulement avec un script, un format de données et
   une loss compatibles avec l'architecture.

### 5. Scripting et automatisation

Utilise Lua pour piloter le framework sans toucher au C++.

1. Lire [docs/02-User-Guide/06-Lua-Scripting.md](../02-User-Guide/06-Lua-Scripting.md).
2. Lire [docs/02-User-Guide/08-Config-Driven-Scripting.md](../02-User-Guide/08-Config-Driven-Scripting.md).
3. Tester un template dans [docs/08-Tuto/03-Tuto-Coder-Script.md](03-Tuto-Coder-Script.md).

### 6. Runtime, memoire et performance

Apprends ou le framework peut echouer ou ralentir.

1. Lire [docs/04-Architecture-Internals/01-Engine-Overview.md](../04-Architecture-Internals/01-Engine-Overview.md).
2. Lire [docs/04-Architecture-Internals/02-Memory.md](../04-Architecture-Internals/02-Memory.md).
3. Lire [docs/04-Architecture-Internals/03-Hardware-Backends.md](../04-Architecture-Internals/03-Hardware-Backends.md).
4. Lire [docs/05-Advanced/01-Performance.md](../05-Advanced/01-Performance.md) et [docs/05-Advanced/02-Debugging.md](../05-Advanced/02-Debugging.md).

### 7. Extension du framework

Quand il manque une piece, vous devez savoir la brancher proprement.

1. Ajouter un modele avec [docs/08-Tuto/02-Tuto-Ajouter-Modele.md](02-Tuto-Ajouter-Modele.md).
2. Modifier un runtime avec [docs/08-Tuto/04-Tuto-Modifier-Ou-Ajouter-Runtime.md](04-Tuto-Modifier-Ou-Ajouter-Runtime.md).
3. Ajouter une OP avec [docs/08-Tuto/05-Tuto-Ajouter-Op.md](05-Tuto-Ajouter-Op.md).
4. Lire [docs/06-Contributing/03-Extending-Models-Runtimes-And-Features.md](../06-Contributing/03-Extending-Models-Runtimes-And-Features.md).

## Atelier pratique

Suivez cette séquence sans dataset pour couvrir les blocs vérifiables :

1. Compilez le projet.
2. Lancez le smoketest.
3. Inspectez `basic_mlp` dans le registre.
4. Construisez-le avec `template_pipeline_args.lua`.
5. Sauvegardez un checkpoint.
6. Analysez ce checkpoint.
7. Exécutez les tests registre et sérialisation.
8. Lisez la section runtime si la mémoire ou la performance ne correspond pas
   aux attentes.

## Mini repere visuel

![Carte des composants](../graphs/02_component_map.svg)

## En cas de blocage

- Si le probleme est conceptuel, reviens a la philosophie et au cycle de vie.
- Si le probleme est runtime, regarde la memoire, les backends et le debugging.
- Si le probleme est d'extension, passe par la couche contrib et les tutos pratiques.

## Suite logique

- Parcours numerote: [docs/08-Tuto/01-Cours-Framework-3-Etapes.md](01-Cours-Framework-3-Etapes.md)
- Tuto modele: [docs/08-Tuto/02-Tuto-Ajouter-Modele.md](02-Tuto-Ajouter-Modele.md)
- Tuto scripting: [docs/08-Tuto/03-Tuto-Coder-Script.md](03-Tuto-Coder-Script.md)
- Tuto runtime: [docs/08-Tuto/04-Tuto-Modifier-Ou-Ajouter-Runtime.md](04-Tuto-Modifier-Ou-Ajouter-Runtime.md)
- Tuto op: [docs/08-Tuto/05-Tuto-Ajouter-Op.md](05-Tuto-Ajouter-Op.md)

## Étapes suivantes

- [Page précédente : Tuto - Ajouter une OPs](05-Tuto-Ajouter-Op.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : Tutoriel : valider VAEConv sans dataset](07-Tuto-VAEConv-Sans-Dataset.md)
