#!/usr/bin/env bash
set -euo pipefail

WORKROOT="${WORKROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BUILD_DIR="$WORKROOT/build"
BUILD_TYPE="Release"
VIZ_BACKEND="${MIMIR_VIZ_BACKEND:-SFML}"

if [[ ! -f "$WORKROOT/CMakeLists.txt" ]]; then
  echo "[error] CMakeLists.txt introuvable dans WORKROOT=$WORKROOT"
  exit 1
fi

if [[ "$OSTYPE" != linux* ]]; then
  echo "[error] Ce script cible Linux (Debian/Ubuntu). OSTYPE=$OSTYPE"
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "[error] apt-get introuvable. Utilise le guide docs pour ton OS/distribution."
  exit 1
fi

SUDO=""
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "[error] Exécute en root ou installe sudo."
    exit 1
  fi
fi

declare -a FEATURE_KEYS=(
  ENABLE_SIMD
  ENABLE_OPENMP
  ENABLE_VULKAN
  ENABLE_OPENCL
  MIMIR_AUTO_DETECT_GPU_BACKENDS
  ENABLE_CUDA
  ENABLE_ROCM
  ENABLE_FFMPEG
  ENABLE_LZ4
  ENABLE_SCRIPTING_REST
  ENABLE_SCRIPTING_JS
  ENABLE_SCRIPTING_CSHARP
  ENABLE_SCRIPTING_RUST
  MIMIR_ENABLE_TESTS
)

declare -A FEATURE_LABELS=(
  [ENABLE_SIMD]="SIMD (AVX2/FMA/F16C/BMI2)"
  [ENABLE_OPENMP]="OpenMP"
  [ENABLE_VULKAN]="Vulkan Compute"
  [ENABLE_OPENCL]="OpenCL Compute"
  [MIMIR_AUTO_DETECT_GPU_BACKENDS]="Auto-détection GPU (CUDA/ROCm)"
  [ENABLE_CUDA]="CUDA Compute"
  [ENABLE_ROCM]="ROCm Compute"
  [ENABLE_FFMPEG]="FFmpeg (audio/vidéo)"
  [ENABLE_LZ4]="Compression LZ4"
  [ENABLE_SCRIPTING_REST]="Bridge REST (placeholder)"
  [ENABLE_SCRIPTING_JS]="Bridge JavaScript"
  [ENABLE_SCRIPTING_CSHARP]="Bridge C#"
  [ENABLE_SCRIPTING_RUST]="Bridge Rust"
  [MIMIR_ENABLE_TESTS]="Tests unitaires (CTest)"
)

declare -A FEATURE_DEFAULTS=(
  [ENABLE_SIMD]=1
  [ENABLE_OPENMP]=1
  [ENABLE_VULKAN]=1
  [ENABLE_OPENCL]=1
  [MIMIR_AUTO_DETECT_GPU_BACKENDS]=0
  [ENABLE_CUDA]=0
  [ENABLE_ROCM]=0
  [ENABLE_FFMPEG]=1
  [ENABLE_LZ4]=1
  [ENABLE_SCRIPTING_REST]=0
  [ENABLE_SCRIPTING_JS]=0
  [ENABLE_SCRIPTING_CSHARP]=0
  [ENABLE_SCRIPTING_RUST]=0
  [MIMIR_ENABLE_TESTS]=0
)

is_interactive=0
if [[ -t 0 && -t 1 ]]; then
  is_interactive=1
fi
if [[ "${NON_INTERACTIVE:-0}" == "1" ]]; then
  is_interactive=0
fi

get_feature() {
  local key="$1"
  echo "${FEATURE_DEFAULTS[$key]}"
}

declare -A FEATURE_VALUES
for key in "${FEATURE_KEYS[@]}"; do
  FEATURE_VALUES[$key]="$(get_feature "$key")"
done

cmake_bool() {
  if [[ "$1" == "1" ]]; then
    echo ON
  else
    echo OFF
  fi
}

print_header() {
  echo
  echo "╭──────────────────────────────────────────────────────────────╮"
  echo "│ Mímir Configure                                              │"
  echo "│ Sélectionne les features à compiler (style utilitaire)      │"
  echo "╰──────────────────────────────────────────────────────────────╯"
  echo
}

interactive_configure_whiptail() {
  local whiptail_bin
  whiptail_bin="$(command -v whiptail || true)"
  if [[ -z "$whiptail_bin" ]]; then
    return 1
  fi

  local checklist_args=()
  for key in "${FEATURE_KEYS[@]}"; do
    local state="OFF"
    if [[ "${FEATURE_VALUES[$key]}" == "1" ]]; then
      state="ON"
    fi
    checklist_args+=("$key" "${FEATURE_LABELS[$key]}" "$state")
  done

  local selected
  if ! selected="$($whiptail_bin \
      --title "Mímir Configure" \
      --checklist "Flèches: naviguer | Espace: (dé)sélectionner | Tab: OK" \
      26 110 16 \
      "${checklist_args[@]}" \
      3>&1 1>&2 2>&3)"; then
    echo "Annulé."
    exit 0
  fi

  for key in "${FEATURE_KEYS[@]}"; do
    FEATURE_VALUES[$key]=0
  done

  for key in $selected; do
    key="${key//\"/}"
    if [[ -n "${FEATURE_VALUES[$key]+x}" ]]; then
      FEATURE_VALUES[$key]=1
    fi
  done

  local bt
  if ! bt="$($whiptail_bin \
      --title "Mímir Configure" \
      --menu "Choisis le Build type" \
      18 80 8 \
      "Release" "Optimisé" \
      "Debug" "Debug symboles" \
      "RelWithDebInfo" "Optimisé + debug" \
      "MinSizeRel" "Optimisé taille" \
      3>&1 1>&2 2>&3)"; then
    echo "Annulé."
    exit 0
  fi
  BUILD_TYPE="$bt"
  return 0
}

interactive_configure() {
  if interactive_configure_whiptail; then
    return 0
  fi

  echo "[warn] whiptail introuvable, fallback mode texte simple."
  print_header
  echo "Installe whiptail pour une sélection flèches + espace: sudo apt-get install -y whiptail"
  echo
  local i=1
  for key in "${FEATURE_KEYS[@]}"; do
    local mark="[ ]"
    if [[ "${FEATURE_VALUES[$key]}" == "1" ]]; then
      mark="[x]"
    fi
    printf " %2d) %s %-26s (%s)\n" "$i" "$mark" "${FEATURE_LABELS[$key]}" "$key"
    ((i++))
  done
  echo
  echo "Aucune modification interactive effectuée (fallback)."
}

if [[ "$is_interactive" == "1" ]]; then
  interactive_configure
  if command -v whiptail >/dev/null 2>&1; then
    VIZ_BACKEND="$(whiptail --title "Mímir Viz" --menu "Interface Viz (scène commune, moteurs indépendants)" 18 80 5 \
      SFML "Fenêtre SFML" QT "Qt 6" GTK "GTK 3" WEB "Navigateur local" NONE "Sans interface" 3>&1 1>&2 2>&3)" || exit 0
  else
    read -r -p "Interface Viz [SFML/QT/GTK/WEB/NONE] ($VIZ_BACKEND): " selected_viz
    VIZ_BACKEND="${selected_viz:-$VIZ_BACKEND}"
  fi
fi
VIZ_BACKEND="${VIZ_BACKEND^^}"
case "$VIZ_BACKEND" in
  SFML) FEATURE_VALUES[ENABLE_SFML]=1 ;;
  QT|GTK|WEB|NONE) FEATURE_VALUES[ENABLE_SFML]=0 ;;
  *) echo "[error] Interface Viz inconnue: $VIZ_BACKEND"; exit 1 ;;
esac

detect_sfml_version() {
  if command -v pkg-config >/dev/null 2>&1; then
    if pkg-config --exists sfml-graphics; then
      pkg-config --modversion sfml-graphics 2>/dev/null || true
      return 0
    fi
    if pkg-config --exists sfml-all; then
      pkg-config --modversion sfml-all 2>/dev/null || true
      return 0
    fi
  fi
  echo ""
}

SFML_DETECTED_VERSION=""
SFML_MODE="disabled"

# SFML n'est requis que par son backend historique.
if [[ "$VIZ_BACKEND" == "SFML" ]]; then
  SFML_DETECTED_VERSION="$(detect_sfml_version)"
  if [[ -z "$SFML_DETECTED_VERSION" ]]; then
    SFML_MODE="missing"
    echo "[warn] SFML non détecté via pkg-config (sfml-graphics/sfml-all)."
    echo "[note] CMake exigera SFML 3 pour cette interface."
  else
    SFML_MAJOR="${SFML_DETECTED_VERSION%%.*}"
    if [[ "$SFML_MAJOR" =~ ^[0-9]+$ ]]; then
      if (( SFML_MAJOR >= 3 )); then
        SFML_MODE="sfml3-enabled"
        echo "[ok] SFML $SFML_DETECTED_VERSION détecté: visualizer SFML3 activé."
      else
        SFML_MODE="sfml2-incompatible"
        echo "[warn] SFML $SFML_DETECTED_VERSION détecté (v2): visualizer SFML3 non compatible."
        echo "[note] Installe SFML 3 ou choisis MIMIR_VIZ_BACKEND=NONE."
      fi
    else
      SFML_MODE="unknown-version"
      echo "[warn] Version SFML non interprétable: '$SFML_DETECTED_VERSION'."
      echo "[note] ENABLE_SFML conservé tel quel."
    fi
  fi

  # Variable legacy: garder une trace explicite pour éviter les confusions.
  if [[ "${MIMIR_FORCE_SFML3_VISUALIZER:-0}" == "1" ]]; then
    echo "[note] MIMIR_FORCE_SFML3_VISUALIZER=1 est désormais inutile (SFML3 auto-détecté)."
  fi
fi

echo
printf "Configuration retenue:\n"
for key in "${FEATURE_KEYS[@]}"; do
  printf "  - %-24s : %s\n" "$key" "$(cmake_bool "${FEATURE_VALUES[$key]}")"
done
if [[ -n "$SFML_DETECTED_VERSION" ]]; then
  printf "  - %-24s : %s\n" "SFML_DETECTED_VERSION" "$SFML_DETECTED_VERSION"
fi
printf "  - %-24s : %s\n" "MIMIR_VIZ_BACKEND" "$VIZ_BACKEND"
printf "  - %-24s : %s\n" "SFML_MODE" "$SFML_MODE"
printf "  - %-24s : %s\n" "CMAKE_BUILD_TYPE" "$BUILD_TYPE"
printf "  - %-24s : %s\n" "BUILD_DIR" "$BUILD_DIR"

echo

SKIP_SYSTEM_DEPS="${SKIP_SYSTEM_DEPS:-0}"

if [[ "$SKIP_SYSTEM_DEPS" == "1" ]]; then
  echo "[1/4] Installation des dépendances système... (skipped: SKIP_SYSTEM_DEPS=1)"
else
  echo "[1/4] Installation des dépendances système..."
  $SUDO apt-get update
fi

REQUIRED_PACKAGES=(
  cmake
  build-essential
  lua5.3
  liblua5.3-dev
  pkg-config
)

if [[ "${FEATURE_VALUES[ENABLE_OPENMP]}" == "1" ]]; then
  REQUIRED_PACKAGES+=(libomp-dev)
fi

OPTIONAL_PACKAGES=()
if [[ "$VIZ_BACKEND" == "QT" ]]; then OPTIONAL_PACKAGES+=(qt6-base-dev); fi
if [[ "$VIZ_BACKEND" == "GTK" ]]; then OPTIONAL_PACKAGES+=(libgtk-3-dev); fi
if [[ "$VIZ_BACKEND" == "QT" || "$VIZ_BACKEND" == "GTK" || "$VIZ_BACKEND" == "WEB" ]]; then
  OPTIONAL_PACKAGES+=(libcairo2-dev)
fi
if [[ "${FEATURE_VALUES[ENABLE_VULKAN]}" == "1" ]]; then
  OPTIONAL_PACKAGES+=(vulkan-tools libvulkan-dev glslang-tools)
fi
if [[ "${FEATURE_VALUES[ENABLE_OPENCL]}" == "1" ]]; then
  OPTIONAL_PACKAGES+=(ocl-icd-opencl-dev clinfo)
fi
if [[ "${FEATURE_VALUES[ENABLE_FFMPEG]}" == "1" ]]; then
  OPTIONAL_PACKAGES+=(ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev)
fi
if [[ "$VIZ_BACKEND" == "SFML" ]]; then
  # Privilégier SFML3 si disponible; fallback SFML2 uniquement si nécessaire.
  if apt-cache show libsfml3-dev >/dev/null 2>&1; then
    OPTIONAL_PACKAGES+=(libsfml3-dev)
  elif apt-cache show libsfml-dev >/dev/null 2>&1; then
    OPTIONAL_PACKAGES+=(libsfml-dev)
  else
    echo "[warn] Aucun package SFML apt détecté (libsfml3-dev/libsfml-dev)."
    echo "[note] Installe SFML3 manuellement puis relance ./config.sh."
  fi
fi
if [[ "${FEATURE_VALUES[ENABLE_LZ4]}" == "1" ]]; then
  OPTIONAL_PACKAGES+=(liblz4-dev)
fi

if [[ "$SKIP_SYSTEM_DEPS" != "1" ]]; then
  $SUDO apt-get install -y "${REQUIRED_PACKAGES[@]}" "${OPTIONAL_PACKAGES[@]}"
fi

echo "[2/4] Vérification des outils..."
cmake --version | head -n 1
c++ --version | head -n 1
lua -v || true
if [[ "${FEATURE_VALUES[ENABLE_VULKAN]}" == "1" ]]; then
  command -v glslangValidator >/dev/null 2>&1 && glslangValidator --version | head -n 1 || true
fi
if [[ "${FEATURE_VALUES[ENABLE_OPENCL]}" == "1" ]]; then
  command -v clinfo >/dev/null 2>&1 && clinfo | sed -n '1,20p' || true
fi
if [[ "${FEATURE_VALUES[MIMIR_AUTO_DETECT_GPU_BACKENDS]}" == "1" ]]; then
  echo "[note] Auto-détection GPU activée: CMake recherchera CUDA et ROCm/HIP."
elif [[ "${FEATURE_VALUES[ENABLE_CUDA]}" == "1" ]]; then
  echo "[note] CUDA activé: installe CUDA Toolkit si nécessaire (non installé automatiquement ici)."
fi
if [[ "${FEATURE_VALUES[MIMIR_AUTO_DETECT_GPU_BACKENDS]}" != "1" && "${FEATURE_VALUES[ENABLE_ROCM]}" == "1" ]]; then
  echo "[note] ROCm activé: installe ROCm/HIP si nécessaire (non installé automatiquement ici)."
fi

echo "[3/4] Configuration CMake..."

CMAKE_ARGS=(
  -S "$WORKROOT"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DMIMIR_VIZ_BACKEND="$VIZ_BACKEND"
  -DENABLE_SFML="$(cmake_bool "${FEATURE_VALUES[ENABLE_SFML]}")"
  -DBUILD_EXAMPLES=OFF
  -DMIMIR_ENABLE_LEGACY_PARAMS=OFF
)

for key in "${FEATURE_KEYS[@]}"; do
  CMAKE_ARGS+=("-D${key}=$(cmake_bool "${FEATURE_VALUES[$key]}")")
done

cmake "${CMAKE_ARGS[@]}"

echo "[4/4] Préparation terminée."
echo "Build prêt dans: $BUILD_DIR"
echo "Commande de compilation: cmake --build \"$BUILD_DIR\" -j\"$(nproc)\""
