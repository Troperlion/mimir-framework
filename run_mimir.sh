#!/bin/bash

# GPU runtime switches (1=enabled, 0=disabled), applied when compiled and available.
# Notebook/runtime MIMIR_DISABLE_* overrides remain authoritative for CPU/TPU runs.
MIMIR_GPU_CUDA="${MIMIR_GPU_CUDA:-0}"
MIMIR_GPU_VULKAN="${MIMIR_GPU_VULKAN:-1}"
MIMIR_GPU_OPENCL="${MIMIR_GPU_OPENCL:-1}"
for runtime in CUDA VULKAN OPENCL; do
    enable_key="MIMIR_GPU_${runtime}"
    disable_key="MIMIR_DISABLE_${runtime}"
    case "${!enable_key}" in
        0|1) ;;
        *) echo "$enable_key must be 0 or 1" >&2; exit 2 ;;
    esac
    if [ -z "${!disable_key}" ]; then
        export "$disable_key=$((1-${!enable_key}))"
    fi
done

export MIMIR_ALLOCATOR_LOG=0
export MIMIR_ALLOCATOR_LOG_VERBOSE=0

# Global execution planner (override possible via environment).
export MIMIR_ENABLE_PLANNER="${MIMIR_ENABLE_PLANNER:-1}"
export MIMIR_PLANNER_MODE="${MIMIR_PLANNER_MODE:-static}" # legacy | static | cost
export MIMIR_PLANNER_BUFFER_REUSE="${MIMIR_PLANNER_BUFFER_REUSE:-1}"
export MIMIR_PLANNER_FUSION="${MIMIR_PLANNER_FUSION:-1}"
export MIMIR_PLANNER_COST_MODEL="${MIMIR_PLANNER_COST_MODEL:-1}"
export MIMIR_PLANNER_DUMP="${MIMIR_PLANNER_DUMP:-1}"
export MIMIR_PLANNER_JSON="${MIMIR_PLANNER_JSON:-.mimir-spill/execution-plan.json}"
export MIMIR_PLANNER_DEVICE_RESIDENCY="${MIMIR_PLANNER_DEVICE_RESIDENCY:-1}"

# OpenCL Rusticl (AMD/Mesa): activer automatiquement sur iGPU AMD.
# Override possible: RUSTICL_ENABLE=... ./run_mimir.sh
if [ -z "${RUSTICL_ENABLE:-}" ]; then
	export RUSTICL_ENABLE=radeonsi
fi

# Configuration OpenMP (override possible via env)
NPROC=$(nproc 2>/dev/null || echo 4)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NPROC}"
export OMP_PROC_BIND=false
export OMP_PLACES=cores
export OMP_SCHEDULE="static"

# Limiter la rétention des gros buffers temporaires dans une arène glibc par worker.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# Désactiver nested parallelism
export OMP_NESTED=false
export OMP_MAX_ACTIVE_LEVELS=1

# CRITIQUE : Garder les threads en vie entre les régions parallèles
export OMP_WAIT_POLICY=active     # Les threads restent actifs (pas de sleep)
export OMP_DYNAMIC=false           # Nombre de threads fixe
# Pinning: ne pas brider par défaut. Si l'utilisateur n'a rien défini,
# épingler sur les OMP_NUM_THREADS premiers CPU logiques.
if [ -z "${GOMP_CPU_AFFINITY:-}" ]; then
	if [ "$OMP_NUM_THREADS" -gt 1 ] 2>/dev/null; then
		export GOMP_CPU_AFFINITY="0-$((OMP_NUM_THREADS-1))"
	else
		export GOMP_CPU_AFFINITY="0"
	fi
fi

# ROCm guard: sur certaines stacks AMD (ex: gfx90c + rocBLAS incomplet),
# l'initialisation peut terminer en IOT/core dump. Par défaut on garde
# Vulkan/CPU stables, et on n'active ROCm qu'en opt-in explicite.
if [ -z "${MIMIR_DISABLE_ROCM:-}" ]; then
	if [ "${MIMIR_FORCE_ROCM:-0}" = "1" ]; then
		export MIMIR_DISABLE_ROCM=0
	else
		export MIMIR_DISABLE_ROCM=1
	fi
fi

# Sortie concise par défaut. MIMIR_LAUNCH_VERBOSE=1 restaure le diagnostic
# complet du lanceur; MIMIR_CONSOLE_VERBOSE=1 restaure toutes les lignes du
# framework (les fichiers logs/*.log restent toujours complets).
if [ "${MIMIR_LAUNCH_VERBOSE:-0}" = "1" ]; then
	echo "Configuration OpenMP:"
	echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS/$NPROC"
	echo "  OMP_PROC_BIND=$OMP_PROC_BIND"
	echo "  OMP_PLACES=$OMP_PLACES"
	echo "  OMP_SCHEDULE=$OMP_SCHEDULE"
	echo "  OMP_NESTED=$OMP_NESTED"
	echo "  OMP_MAX_ACTIVE_LEVELS=$OMP_MAX_ACTIVE_LEVELS"
	echo "  OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
	echo "  OMP_DYNAMIC=$OMP_DYNAMIC"
	echo "  GOMP_CPU_AFFINITY=$GOMP_CPU_AFFINITY"
	echo "  MIMIR_DISABLE_ROCM=$MIMIR_DISABLE_ROCM"
	echo "  RUSTICL_ENABLE=$RUSTICL_ENABLE"
else
	echo "\n\n\t→\tMímir: threads=$OMP_NUM_THREADS/$NPROC rocm_disabled=$MIMIR_DISABLE_ROCM rusticl=$RUSTICL_ENABLE"
fi

# Lancer mimir
# Benchmarks may select another native Mímir entry point while retaining this
# exact runtime setup. Normal invocations keep the production executable.
MIMIR_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${MIMIR_EXECUTABLE:-$MIMIR_SCRIPT_DIR/bin/mimir}" "$@"
