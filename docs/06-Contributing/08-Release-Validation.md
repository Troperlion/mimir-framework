# Validation de la version 3.6.0

La version est définie dans `VERSION` et lue par CMake. Les changements de cette
version concernent le routage mesuré, les optimizers, les kernels CPU/portables,
les diagnostics de gradients, les contrôles runtime et les benchmarks Colab.
Le [changelog](../../CHANGELOG.md) détaille ces surfaces.

## Reproduire les vérifications des workflows

Le workflow `.github/workflows/unit-tests.yml` compile Linux, Windows natif
(MSVC) et effectue une compilation croisée Windows depuis Linux. Les profils
Linux utilisent `./config.sh` pour installer les dépendances et préparer CMake. Les tests Windows sont compilés mais ne sont pas
exécutés sous Linux. Les runtimes CUDA/ROCm et les interfaces desktop ne sont
pas activés par ces deux profils ; leur validation reste distincte.

### Linux Release

Installer CMake, Ninja, G++, Lua 5.3, LZ4, Node.js 22, .NET 10, `dotnet-script`,
Rust et `rust-script`, comme indiqué dans le workflow. Les bridges externes
nécessitent leurs exécutables dans `PATH`. `config.sh` utilise apt/sudo pour les
dépendances système ; `--skip-system-deps` est réservé aux environnements où
elles sont déjà installées.

```bash
./config.sh --gui NONE --non-interactive --build-dir build_ci_linux \
  --generator Ninja --enable MIMIR_ENABLE_TESTS \
  --enable ENABLE_SCRIPTING_JS --enable ENABLE_SCRIPTING_RUST \
  --enable ENABLE_SCRIPTING_CSHARP \
  --disable ENABLE_VULKAN --disable ENABLE_OPENCL
cmake --build build_ci_linux --parallel 2
ctest --test-dir build_ci_linux --output-on-failure --timeout 120
```

### Windows depuis Linux

Installer MinGW-w64 et les dépendances `lua` et `lz4` du triplet vcpkg
`x64-mingw-dynamic`. Définir `VCPKG_ROOT` vers l’installation vcpkg.

```bash
./config.sh --gui NONE --non-interactive --build-dir build_ci_windows \
  --generator Ninja --enable MIMIR_ENABLE_TESTS \
  --disable ENABLE_SCRIPTING_JS --enable ENABLE_SCRIPTING_RUST \
  --enable ENABLE_SCRIPTING_CSHARP \
  --disable ENABLE_VULKAN --disable ENABLE_OPENCL --disable ENABLE_FFMPEG \
  --cmake-arg -DCMAKE_SYSTEM_NAME=Windows \
  --cmake-arg -DCMAKE_C_COMPILER=x86_64-w64-mingw32-gcc \
  --cmake-arg -DCMAKE_CXX_COMPILER=x86_64-w64-mingw32-g++ \
  --cmake-arg -DCMAKE_RC_COMPILER=x86_64-w64-mingw32-windres \
  --cmake-arg "-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  --cmake-arg -DVCPKG_TARGET_TRIPLET=x64-mingw-dynamic \
  --cmake-arg -DVCPKG_APPLOCAL_DEPS=OFF
cmake --build build_ci_windows --parallel 2
```

Le bridge JavaScript utilise un transport POSIX et reste désactivé sur Windows.
Les bridges C#/Rust sont compilés ; leurs tests peuvent être ignorés si les
interpréteurs externes ne sont pas détectés.

Vérifier la présence de `build_ci_windows/bin/mimir.exe` et de
`build_ci_windows/Tests/BridgeTest.API.exe`. Ces fichiers prouvent la compilation
croisée, pas leur exécution sur un Windows réel.

## Documentation et notebooks

```bash
python3 scripts/tools/verify_docs.py
./scripts/tools/verify_api_sync.sh
python3 -m unittest discover -s scripts/tests -p 'test_colab*.py'
python3 scripts/tests/test_config_cli.py
```

Les tests Colab distinguent les vérifications statiques de notebook et les tests
nécessitant une exécution native, un GPU ou les dépendances externes. Un test
ignoré n’est pas une preuve d’exécution sur le matériel concerné.

## Vulkan/OpenCL et autres configurations

Un profil supplémentaire avec `ENABLE_VULKAN=ON`, `ENABLE_OPENCL=ON` et
`MIMIR_ENABLE_TESTS=ON` vérifie la compilation des kernels portables et expose
les tests GPU CTest. Un backend sans périphérique disponible peut être ignoré
avec le code CTest 77. Une exécution Vulkan sur un périphérique logiciel ne
prouve pas des performances sur GPU physique.

CUDA requiert un toolkit et une compilation `ENABLE_CUDA=ON`. Une compilation
CPU ou Vulkan/OpenCL ne valide pas le code CUDA. Les backends Qt/GTK/SFML/Web
nécessitent aussi leurs dépendances et leurs tests d’interface propres.

## Publication

Les profils doivent être vérifiés sur le contenu final du commit. Ne pas
annoncer que GitHub Actions est vert avant la fin des jobs distants : les
réseaux, versions des runners, packages et permissions du wiki restent externes
au test local. Le workflow du wiki sérialise les publications de `main` et
`develop` vers le dépôt wiki commun.

Les journaux de validation sont conservés hors des sources ; les données,
checkpoints, builds et le hub local non suivi ne font pas partie de la release.


## Résultats locaux du 22 octobre 2026

| Profil ou contrôle | Résultat |
| --- | --- |
| Linux Release, C++/Lua et bridges JS/C#/Rust | Compilation réussie ; 107 tests CTest, aucun échec, 4 ignorés. |
| Windows x86-64 MinGW/vcpkg | Compilation réussie ; `mimir.exe` et 94 exécutables de tests produits. Exécution Windows non réalisée localement. |
| Linux Vulkan/OpenCL | Compilation réussie ; 114 tests CTest, aucun échec, 7 ignorés. |
| Qt 6 | `mimir`, hôte desktop et cibles de diagnostics compilés ; 3 tests ciblés réussis. Pas de validation interactive du bureau. |
| CLI `config.sh` | 3 tests réussis, syntaxe Bash et options de configuration contrôlées. |
| Notebooks Colab | 27 tests Python, aucun échec, 3 ignorés ; archive embarquée de 544 fichiers synchronisée. |
| Documentation et API | 100 pages actives et 623 liens locaux vérifiés ; 106 fonctions et 13 modules Lua synchronisés. |
| GitHub workflows | YAML et actionlint valides ; génération du wiki exécutée hors ligne sans push. |

Les périphériques utilisés pour les tests Vulkan/OpenCL sont logiciels
(llvmpipe/Lavapipe). La parité numérique et l’exécution des kernels portables ont
été contrôlées ; aucune accélération sur GPU physique n’est déduite de ces tests.
Les tests ignorés concernent notamment TPU, événements graphiques sans affichage
et certains parcours d’optimizer/perte non disponibles sur ces périphériques.
CUDA n’a pas été compilé faute de `nvcc`. Le profil MSVC du workflow doit encore
être exécuté sur son runner Windows ; les vérifications locales ne remplacent
pas son résultat.

Les compilations locales ont utilisé `./config.sh --skip-system-deps` car les
dépendances étaient déjà installées et sudo nécessitait une authentification.
Les workflows Linux utilisent `./config.sh` avec l’installation des dépendances
activée. L’installation apt a également été vérifiée par les tests du script,
sans mutation système dans les tests.
