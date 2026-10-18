# Installation et compilation

Installer les dépendances et compiler selon votre environnement.

**Public concerné :** Débutant qui veut une installation stable et reproductible.

> **Prérequis**
>
> Accès internet + installation de paquets système.


Pour compiler Mímir, il faut CMake, C++17 et quelques dépendances. Cette page couvre tout ce qu'il faut savoir.

---

## Sur cette page

- [Dépendances](#dépendances)
- [Google Colab](#google-colab-cuda)
- [🔨 Build — Guide par OS](#build-guide-par-os)
- [📋 Options CMake principales](#options-cmake-principales)
- [🧪 Vérifier l'installation](#vérifier-linstallation)
- [🐛 Troubleshooting](#troubleshooting)
- [💡 Tips avancés](#tips-avancés)
- [Checklist de déploiement en production](#checklist-de-déploiement-en-production)
- [Étapes suivantes](#étapes-suivantes)

## Dépendances

### Obligatoires

| Dépendance | Version min | Rôle | Installation |
| --- | --- | --- | --- |
| **CMake** | 3.15+ | Build system | `apt install cmake` ou [cmake.org](https://cmake.org) |
| **C++ compiler** | C++17 compatible | Compilation | GCC 7+, Clang 7+, MSVC 2017+ |
| **Lua** | 5.3+ | API scripting | `apt install lua5.3 liblua5.3-dev` |
| **OpenMP** | (bundled) | Parallelisation CPU | Généralement inclus avec le compilateur |

### Optionnelles (à bas bruit)

| Dépendance | Rôle | Défaut | Désactiver avec |
| --- | --- | --- | --- |
| **Vulkan SDK** | GPU Compute (SPIR-V shaders) | ON | `-DENABLE_VULKAN=OFF` |
| **glslangValidator** | Compilation shaders Vulkan | Auto-build | `-DMIMIR_FETCH_GLSLANG=OFF` |
| **SFML** | Visualisation graphique | ON | `-DENABLE_SFML=OFF` |
| **OpenCL** | GPU Compute (OpenCL) | ON | `-DENABLE_OPENCL=OFF` |
| **LZ4** | Compression checkpoints | ON | `-DENABLE_LZ4=OFF` |

### Extras (à haute spécialisation)

| Dépendance | Rôle | Utilité | Installer si |
| --- | --- | --- | --- |
| **CUDA Toolkit** | GPU NVIDIA (cuBLAS) | Inférence NVIDIA 100x+ rapide | Vous avez GPU NVIDIA + `-DENABLE_CUDA=ON` |
| **ROCm** | GPU AMD (rocBLAS) | Inférence AMD rapide | Vous avez GPU AMD + `-DENABLE_ROCM=ON` |

---

## 🔨 Build — Guide par OS

### Google Colab (CUDA)

Sélectionnez d'abord un environnement d'exécution avec GPU dans Colab, puis
vérifiez que le GPU NVIDIA est visible:

```bash
!nvidia-smi
```

Après avoir cloné le dépôt dans `/content/tensor-2`, installez les dépendances
de compilation:

```bash
!apt-get update -qq
!apt-get install -y ninja-build build-essential pkg-config \
  liblua5.3-dev liblz4-dev \
  libavcodec-dev libavformat-dev libavutil-dev \
  libswresample-dev libswscale-dev
```

Compilez ensuite le binaire headless. Le preset détecte automatiquement CUDA
ou ROCm selon le toolkit disponible et limite la compilation à deux tâches
parallèles afin de rester compatible avec la mémoire des sessions Colab
gratuites:

```bash
%cd /content/tensor-2
!cmake --preset colab-cuda
!cmake --build --preset build-colab-cuda
!./build-colab/bin/mimir --version
```

Le preset `colab-cuda` active l'auto-détection CUDA/ROCm, OpenMP, SIMD, FFmpeg
et LZ4. Il désactive Vulkan, OpenCL, l'interface graphique, les bridges
externes, les exemples et les tests. Vérifiez les lignes `CUDA Compute` et
`ROCm Compute` du résumé CMake. Si les deux valent `OFF`, vérifiez que le
notebook utilise bien un environnement GPU et que `nvcc` ou `hipcc` est
accessible.

### Linux (Ubuntu/Debian)

**Script automatique recommande:**

```bash
cd ~/path/to/tensor-2

# Installe les dependances systeme et configure CMake dans build/
./config.sh

# Compile le projet
cmake --build build -j$(nproc)
```

Variables utiles du script:

- `WORKROOT=/path/to/tensor-2` pour pointer explicitement la racine projet
- `BUILD_TYPE=Debug` pour preparer un build debug
- `ENABLE_VULKAN=0`, `ENABLE_OPENCL=0`, `ENABLE_SFML=0`, `ENABLE_LZ4=0` pour desactiver des dependances optionnelles

**Installation des dépendances:**

```bash
# Dépendances obligatoires
sudo apt-get update
sudo apt-get install -y \
  cmake \
  build-essential \
  lua5.3 liblua5.3-dev \
  libomp-dev

# Optionnel: Vulkan Compute
sudo apt-get install -y \
  vulkan-tools libvulkan-dev \
  glslang-tools

# Optionnel: SFML visualisation
sudo apt-get install -y \
  libsfml-dev

# Optionnel: OpenCL GPU
sudo apt-get install -y \
  ocl-icd-opencl-dev
```

**Compilation:**

```bash
cd ~/path/to/tensor-2
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_SIMD=ON \
  -DENABLE_OPENMP=ON
cmake --build build -j$(nproc)

# Résultat
ls -la bin/mimir
```

**Tests (optionnel):**
```bash
ctest --test-dir build --output-on-failure
```

---

### macOS

**Installation des dépendances:**

```bash
# Installer Homebrew si absent
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Dépendances
brew install cmake lua@5.3 libomp

# Optionnel: Vulkan (moins courant sur macOS)
brew install vulkan-loader molten-vk

# Optionnel: SFML
brew install sfml
```

**Compilation:**

```bash
cd ~/path/to/tensor-2

# Pour Intel Mac (x86_64):
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_SIMD=ON

# Pour Apple Silicon (arm64):
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_SIMD=OFF

cmake --build build -j$(sysctl -n hw.ncpu)
```

---

### Windows (MSVC)

**Requirements:**
- Visual Studio 2017+ ou Build Tools
- CMake (via installer ou `choco install cmake`)
- Lua 5.3+ (via installer ou prebuilt binaries)

**Compilation (PowerShell):**

```powershell
cd C:\path\to\tensor-2

# Configure
cmake -S . -B build `
  -G "Visual Studio 16 2019" `
  -DCMAKE_BUILD_TYPE=Release `
  -DLUA_INCLUDE_DIR="C:\Program Files\Lua\include" `
  -DLUA_LIBRARY="C:\Program Files\Lua\lua53.lib"

# Build
cmake --build build --config Release -j
```

⚠️ **Note:** OpenMP et SIMD sont généralement détectés automatiquement. Les GPUs ne sont pas (encore) supportés sur Windows.

---

## 📋 Options CMake principales

```bash
# Syntaxe générale:
cmake -S . -B build [OPTIONS]

# Exemples:

# Release optimisé (défaut)
-DCMAKE_BUILD_TYPE=Release

# Debug (symbols + no optimizations)
-DCMAKE_BUILD_TYPE=Debug

# Désactiver SIMD (utile pour débogage ou CPU ancien)
-DENABLE_SIMD=OFF

# Désactiver Vulkan (si problèmes de build)
-DENABLE_VULKAN=OFF

# Désactiver SFML (si dépendance problématique)
-DENABLE_SFML=OFF

# Activer CUDA (NVIDIA GPU)
-DENABLE_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda

# Activer ROCm (AMD GPU)
-DENABLE_ROCM=ON -DMIMIR_ROCM_ROOT=/opt/rocm

# Détecter automatiquement CUDA et ROCm
-DMIMIR_AUTO_DETECT_GPU_BACKENDS=ON

# Build static (pas de dépendances dynamiques)
-DBUILD_MIMIR_STATIC=ON

# Build exemples (dans build/examples/)
-DBUILD_EXAMPLES=ON

# Mode legacy params (attention: très RAM-intensive!)
-DMIMIR_ENABLE_LEGACY_PARAMS=ON
```

---

## 🧪 Vérifier l'installation

```bash
# Afficher l'aide + capacités hardware
./bin/mimir --help

# Expected output:
# ========== Mímir v3.5.0 ==========
# Capabilities: [AVX2] [FMA] [F16C] [BMI2] [OpenMP] [Vulkan] ...
# Usage: ./bin/mimir [OPTIONS]

# Exécuter un smoke test
./bin/mimir --lua scripts/tests/test_serialization_smoke.lua

# Si pas d'erreur → tout fonctionne!
```

---

## 🐛 Troubleshooting

### CMake error: "Lua not found"

```bash
# Solution 1: Installer Lua
sudo apt-get install lua5.3 liblua5.3-dev

# Solution 2: Spécifier le chemin manuellement
cmake -S . -B build \
  -DLUA_INCLUDE_DIR=/usr/include/lua5.3 \
  -DLUA_LIBRARY=/usr/lib/x86_64-linux-gnu/liblua5.3.so
```

### Compilation error: "OpenMP not found"

```bash
# Solution 1: Installer OpenMP
sudo apt-get install libomp-dev

# Solution 2: Compiler sans OpenMP
cmake -S . -B build -DENABLE_OPENMP=OFF
```

### Error: "glslangValidator not found" (Vulkan)

```bash
# Solution 1: Installer glslang
sudo apt-get install glslang-tools

# Solution 2: Désactiver Vulkan
cmake -S . -B build -DENABLE_VULKAN=OFF

# Solution 3: Auto-build glslang (défaut)
# CMake téléchargera et compilera glslang lui-même
```

### SFML compilation errors

```bash
# Solution: Désactiver SFML
cmake -S . -B build -DENABLE_SFML=OFF
```

### Build hangs or super slow

```bash
# Réduire parallélisation
cmake --build build -j2

# Ou nettoyer et recommencer
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

---

## 💡 Tips avancés

### Cross-compilation pour CPU plus ancien

```bash
# Pour CPU sans AVX2 (ex: Atom, Celeron):
cmake -S . -B build \
  -DENABLE_SIMD=OFF \
  -DCMAKE_CXX_FLAGS="-mtune=generic"
```

### Build completement statique (Linux)

```bash
cmake -S . -B build \
  -DBUILD_MIMIR_STATIC=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-static"

cmake --build build -j

# Résultat: ./bin/mimir_static (pas de dépendances système)
```

### Benchmark compilation

```bash
# Temps de build avec output détaillé:
time cmake --build build -j --verbose 2>&1 | tee build.log
```

### Nettoyer complètement (cache CMake)

```bash
rm -rf build
cmake -S . -B build --fresh
cmake --build build -j
```

---

## Checklist de déploiement en production

- [ ] Compiler en `-DCMAKE_BUILD_TYPE=Release`
- [ ] Vérifier capacités: `./bin/mimir --help`
- [ ] Exécuter smoke test: `./bin/mimir --lua scripts/tests/test_serialization_smoke.lua`
- [ ] Tester le modèle qu'on va utiliser
- [ ] Si CPU ancien: compiler sans `-DENABLE_SIMD`
- [ ] Si GPU requis: `-DENABLE_CUDA=ON` ou `-DENABLE_ROCM=ON`
- [ ] Archiver le binaire + config utilisée

## Étapes suivantes

- [Page précédente : Quick start](01-Quick-Start.md)
- [Index de la documentation](../00-INDEX.md)
- [Page suivante : CLI (binaire `mimir`)](03-CLI.md)
