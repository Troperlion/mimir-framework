# Démarrage rapide

Cette page vous accompagne depuis les sources jusqu'à l'exécution d'un premier
modèle Mímir. Le parcours utilise uniquement les fichiers du dépôt et ne
nécessite aucun dataset.

## Sur cette page

- [Prérequis](#prérequis)
- [Compiler Mímir](#compiler-mímir)
- [Valider l'installation](#valider-linstallation)
- [Créer un premier modèle](#créer-un-premier-modèle)
- [Sauvegarder un checkpoint](#sauvegarder-un-checkpoint)
- [Résoudre les problèmes courants](#résoudre-les-problèmes-courants)
- [Étapes suivantes](#étapes-suivantes)

## Prérequis

Mímir nécessite CMake, un compilateur compatible C++17 et les bibliothèques Lua
de développement. Vérifiez d'abord les outils installés :

```bash
cmake --version
g++ --version
lua -v
```

Les backends CUDA, ROCm, Vulkan et OpenCL sont optionnels. Le runtime CPU suffit
pour suivre ce guide.

Pour installer les dépendances ou activer un backend particulier, consultez le
[guide d'installation](02-Installation.md).

## Compiler Mímir

Depuis la racine du dépôt :

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Le binaire produit est `./bin/mimir`. Affichez son aide pour vérifier qu'il
démarre et connaître les capacités détectées :

```bash
./bin/mimir --help
```

> **Note**
> La liste des backends dépend des options de compilation, des bibliothèques
> disponibles et du matériel détecté.

## Valider l'installation

Exécutez le test de sérialisation Lua :

```bash
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua
```

La commande doit se terminer sans erreur. Elle vérifie une chaîne minimale de
création, de sauvegarde et de relecture sans charger de données externes.

Les tests C++ peuvent ensuite être exécutés avec :

```bash
ctest --test-dir build --output-on-failure
```

## Créer un premier modèle

Le template minimal construit un modèle, alloue ses paramètres et exécute une
passe avant :

```bash
./bin/mimir --lua scripts/templates/template_new_model.lua
```

Le script illustre le cycle de vie commun aux modèles Mímir :

1. choisir une configuration ;
2. créer les layers ;
3. allouer et initialiser les paramètres ;
4. fournir les entrées ;
5. exécuter `forward()`.

Pour inspecter une architecture sans écrire de script :

```bash
./bin/mimir --lua scripts/tools/inspect_architectures.lua -- \
  --list basic_mlp --params --layers --stats
```

Les arguments après `--` sont transmis au script Lua. Les arguments placés
avant ce séparateur appartiennent au binaire `mimir`.

## Sauvegarder un checkpoint

La Pipeline API réunit la création depuis le registre et la sérialisation :

```bash
./bin/mimir --lua scripts/templates/template_pipeline_args.lua -- \
  --from-registry \
  --arch basic_mlp \
  --no-train \
  --save /tmp/mimir_basic_mlp.safetensors
```

Vérifiez ensuite le checkpoint :

```bash
./bin/mimir --lua scripts/tools/analyze_model.lua -- \
  --in /tmp/mimir_basic_mlp.safetensors
```

`analyze_model.lua` affiche la structure et les paramètres lisibles depuis
l'artefact. Le fichier placé dans `/tmp` peut être supprimé après le tutoriel.

## Résoudre les problèmes courants

### CMake ne trouve pas Lua

Installez le paquet de développement Lua de votre distribution, puis
reconfigurez le projet. Sur Debian ou Ubuntu :

```bash
sudo apt-get install lua5.3 liblua5.3-dev
cmake -S . -B build
```

### La compilation échoue avec OpenMP

OpenMP peut être désactivé pour isoler le problème :

```bash
cmake -S . -B build -DENABLE_OPENMP=OFF
cmake --build build -j
```

### Aucun backend GPU n'apparaît

Le backend doit être activé à la configuration CMake et ses bibliothèques
doivent être détectées. Continuez avec le CPU ou consultez
[l'accélération GPU](../05-Advanced/05-GPU-Acceleration.md).

### Le binaire `./bin/mimir` est absent

Relancez la compilation en mode verbeux :

```bash
cmake --build build --verbose
```

Lisez la première erreur de compilation. Évitez de supprimer le répertoire de
build avant d'avoir identifié la dépendance ou le fichier concerné.

## Étapes suivantes

- Découvrez les [concepts essentiels](../02-User-Guide/01-Core-Concepts.md).
- Comprenez le [cycle de vie des modèles](../02-User-Guide/02-Model-Lifecycle.md).
- Apprenez à écrire des [scripts Lua](../02-User-Guide/06-Lua-Scripting.md).
- Suivez le tutoriel [du registre au checkpoint](../08-Tuto/08-Tuto-Registre-Pipeline-Checkpoint.md).
- Choisissez un [parcours adapté à votre niveau](06-Learning-Paths.md).
