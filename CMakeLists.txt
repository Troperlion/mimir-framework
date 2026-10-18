cmake_minimum_required(VERSION 3.15)

# Source de verite: fichier VERSION a la racine du repo.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" MIMIR_VERSION_RAW)
string(STRIP "${MIMIR_VERSION_RAW}" MIMIR_VERSION)
project(Mimir VERSION ${MIMIR_VERSION} LANGUAGES CXX)

# ============================================================================
# Configuration du projet
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options de compilation
option(ENABLE_SIMD "Enable SIMD optimizations (AVX2, FMA, F16C, BMI2)" ON)
option(ENABLE_OPENMP "Enable OpenMP parallel processing" ON)
option(ENABLE_VULKAN "Enable Vulkan Compute acceleration" ON)
option(ENABLE_OPENCL "Enable OpenCL GPU compute" ON)
option(ENABLE_CUDA "Enable CUDA GPU compute (cuBLAS)" OFF)
option(ENABLE_ROCM "Enable ROCm GPU compute (rocBLAS/HIP)" OFF)
option(MIMIR_AUTO_DETECT_GPU_BACKENDS "Auto-detect and enable available CUDA/ROCm backends" OFF)
option(ENABLE_LZ4 "Enable LZ4 compression" ON)
option(ENABLE_FFMPEG "Enable FFmpeg-backed audio/video loading" ON)
option(ENABLE_SFML "Enable SFML visualization" ON)
set(MIMIR_VIZ_BACKEND "SFML" CACHE STRING
    "Viz interface selected at compile time: SFML, QT, GTK, WEB, or NONE")
set_property(CACHE MIMIR_VIZ_BACKEND PROPERTY STRINGS SFML QT GTK WEB NONE)
option(ENABLE_SCRIPTING_REST "Enable REST scripting bridge" OFF)
option(ENABLE_SCRIPTING_RUST "Enable Rust scripting bridge" ON)
option(ENABLE_SCRIPTING_JS "Enable JavaScript scripting bridge" ON)
option(ENABLE_SCRIPTING_CSHARP "Enable C# scripting bridge" ON)
option(BUILD_EXAMPLES "Build example executables" OFF)
option(MIMIR_ENABLE_FACE_RECOGNITION_TOOL "Build OpenCV face detection/recognition tool" OFF)
option(MIMIR_PORTABLE_BUILD "Disable host-specific CPU optimizations for portable binaries" OFF)

string(TOUPPER "${MIMIR_VIZ_BACKEND}" MIMIR_VIZ_BACKEND)
set(_MIMIR_VIZ_BACKENDS SFML QT GTK WEB NONE)
if(NOT MIMIR_VIZ_BACKEND IN_LIST _MIMIR_VIZ_BACKENDS)
    message(FATAL_ERROR
        "MIMIR_VIZ_BACKEND='${MIMIR_VIZ_BACKEND}' invalide. "
        "Valeurs possibles: SFML, QT, GTK, WEB, NONE.")
endif()

# Compatibilité historique : avec le backend par défaut SFML,
# ENABLE_SFML=OFF signifie encore NONE. Les autres backends sont indépendants.
if(NOT ENABLE_SFML)
    if(MIMIR_VIZ_BACKEND STREQUAL "SFML")
        set(MIMIR_VIZ_BACKEND "NONE")
    endif()
endif()
if(MIMIR_VIZ_BACKEND STREQUAL "NONE")
    set(ENABLE_SFML OFF)
elseif(MIMIR_VIZ_BACKEND STREQUAL "SFML")
    set(ENABLE_SFML ON)
else()
    set(ENABLE_SFML OFF)
endif()
message(STATUS "Viz interface: ${MIMIR_VIZ_BACKEND}")

if(ENABLE_SCRIPTING_REST)
    message(WARNING "ENABLE_SCRIPTING_REST=ON demandé, mais aucun bridge REST natif n'est implémenté dans ce dépôt. Option forcée à OFF.")
    set(ENABLE_SCRIPTING_REST OFF)
endif()

# Build d'un binaire statique (sans dépendances dynamiques) : nécessite les variantes *.a
# des libs système (Lua, libstdc++, etc.). Sur certaines distros, un binaire 100% statique
# peut ne pas être possible sans musl.
option(BUILD_MIMIR_STATIC "Build fully static mimir_static binary (Linux only)" OFF)

# 🛡️ SÉCURITÉ MÉMOIRE: Legacy params structure (désactivé par défaut)
# ⚠️  Activer seulement pour debug/compatibilité - consomme énormément de RAM!
option(MIMIR_ENABLE_LEGACY_PARAMS "Enable legacy params structure (high RAM usage)" OFF)

# Type de build par défaut
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Le build principal alimente ./bin pour rester compatible avec run_mimir.sh.
# Les builds secondaires (notamment les builds CPU des tests du planner) gardent
# leurs exécutables dans leur propre arbre afin de ne jamais écraser le binaire
# de production avec une configuration matérielle différente.
if(CMAKE_BINARY_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/build")
    set(MIMIR_RUNTIME_OUTPUT_DIRECTORY_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/bin")
else()
    set(MIMIR_RUNTIME_OUTPUT_DIRECTORY_DEFAULT "${CMAKE_BINARY_DIR}/bin")
endif()
set(MIMIR_RUNTIME_OUTPUT_DIRECTORY
    "${MIMIR_RUNTIME_OUTPUT_DIRECTORY_DEFAULT}"
    CACHE PATH "Directory for Mimir runtime executables")

# ============================================================================
# Détection des capacités hardware
# ============================================================================

include(CheckCXXCompilerFlag)

# CUDA includes CPU dispatch headers: nvcc must pass SIMD options to its host compiler.
if(ENABLE_SIMD)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
    check_cxx_compiler_flag("-mf16c" COMPILER_SUPPORTS_F16C)
    check_cxx_compiler_flag("-mbmi2" COMPILER_SUPPORTS_BMI2)
    
    if(COMPILER_SUPPORTS_AVX2)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx2>)
        add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler=-mavx2>")
        message(STATUS "AVX2 support: Enabled")
    endif()
    
    if(COMPILER_SUPPORTS_FMA)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mfma>)
        add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler=-mfma>")
        message(STATUS "FMA support: Enabled")
    endif()
    
    if(COMPILER_SUPPORTS_F16C)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mf16c>)
        add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler=-mf16c>")
        message(STATUS "F16C support: Enabled")
    endif()
    
    if(COMPILER_SUPPORTS_BMI2)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mbmi2>)
        add_compile_options("$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler=-mbmi2>")
        message(STATUS "BMI2 support: Enabled")
    endif()
endif()

# ============================================================================
# Options de compilation
# ============================================================================

# Flags communs
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4100> # unreferenced formal parameter
        $<$<COMPILE_LANGUAGE:CXX>:/wd4101> # unreferenced local variable
        $<$<COMPILE_LANGUAGE:CXX>:/wd4996> # deprecated API warnings
    )
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-variable>
    )
endif()

# Flags de release
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    if(NOT MIMIR_PORTABLE_BUILD)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    endif()
endif()

# Flags de debug
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /Od /DDEBUG")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")
endif()

# ============================================================================
# Recherche des dépendances
# ============================================================================

# OpenMP
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    endif()
endif()

# Lua 5.3+
find_package(Lua 5.3 REQUIRED)
if(LUA_FOUND)
    message(STATUS "Lua found: ${LUA_VERSION_STRING}")
    include_directories(${LUA_INCLUDE_DIR})
endif()

# Vulkan (optionnel)
if(ENABLE_VULKAN)
    find_package(Vulkan)
    if(Vulkan_FOUND)
        message(STATUS "Vulkan found: ${Vulkan_VERSION}")
        include_directories(${Vulkan_INCLUDE_DIRS})
    else()
        message(WARNING "Vulkan not found - Vulkan Compute disabled")
        set(ENABLE_VULKAN OFF)
    endif()
endif()

# Vulkan shaders (SPIR-V) - best-effort
if(ENABLE_VULKAN)
    option(MIMIR_FETCH_GLSLANG "Fetch and build glslangValidator locally if not found" ON)

    find_program(GLSLANG_VALIDATOR NAMES glslangValidator)
    set(_MIMIR_GLSLANG_VALIDATOR_CMD "")

    if(GLSLANG_VALIDATOR)
        message(STATUS "glslangValidator found: ${GLSLANG_VALIDATOR}")
        set(_MIMIR_GLSLANG_VALIDATOR_CMD "${GLSLANG_VALIDATOR}")
    elseif(MIMIR_FETCH_GLSLANG)
        message(STATUS "glslangValidator not found - fetching glslang to build it locally")
        include(FetchContent)
        # Keep glslang build lean; we only need the validator binary.
        set(ENABLE_GLSLANG_BINARIES ON CACHE BOOL "" FORCE)
        set(ENABLE_SPVREMAPPER OFF CACHE BOOL "" FORCE)
        set(ENABLE_OPT OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

        FetchContent_Declare(glslang
            GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
            # Known-good commit (can be updated later if needed)
            GIT_TAG 617ecc40dc1d2e64b2796198769ef035d73b9e16
        )
        FetchContent_MakeAvailable(glslang)

        # glslang builds the CLI as target `glslang-standalone` (OUTPUT_NAME: glslang)
        # and creates a post-build symlink `glslangValidator`. Use the target directly.
        if(TARGET glslang-standalone)
            set(_MIMIR_GLSLANG_VALIDATOR_CMD $<TARGET_FILE:glslang-standalone>)
        endif()
    endif()

    if(_MIMIR_GLSLANG_VALIDATOR_CMD)
        set(MIMIR_SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/shaders")
        set(MIMIR_SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")
        file(MAKE_DIRECTORY "${MIMIR_SHADER_OUT_DIR}")

        set(MIMIR_VULKAN_SHADER_NAMES
            elementwise
            linear_forward
            add_forward
            mul_forward
            relu_forward
            silu_forward
            gelu_forward
            sigmoid_forward
            tanh_forward
            conv2d_forward
            conv_transpose2d_forward
        )

        set(MIMIR_VULKAN_SHADER_SPV_OUTPUTS)
        foreach(_shader_name IN LISTS MIMIR_VULKAN_SHADER_NAMES)
            set(_shader_src "${MIMIR_SHADER_SRC_DIR}/${_shader_name}.comp")
            set(_shader_spv "${MIMIR_SHADER_OUT_DIR}/${_shader_name}.comp.spv")
            list(APPEND MIMIR_VULKAN_SHADER_SPV_OUTPUTS "${_shader_spv}")

            add_custom_command(
                OUTPUT "${_shader_spv}"
                COMMAND ${_MIMIR_GLSLANG_VALIDATOR_CMD} -V "${_shader_src}" -o "${_shader_spv}"
                DEPENDS "${_shader_src}"
                COMMENT "Compiling Vulkan shader ${_shader_name}.comp -> SPIR-V"
                VERBATIM
            )
        endforeach()

        add_custom_target(mimir_shaders ALL DEPENDS ${MIMIR_VULKAN_SHADER_SPV_OUTPUTS})
        if(TARGET glslang-standalone)
            add_dependencies(mimir_shaders glslang-standalone)
        endif()
    else()
        message(WARNING "No glslangValidator available - Vulkan shaders will not be compiled (runtime will fallback to CPU).")
    endif()
endif()

# LZ4 (compression)
if(ENABLE_LZ4)
    find_library(LZ4_LIBRARY NAMES lz4)
    if(LZ4_LIBRARY)
        message(STATUS "LZ4 found: ${LZ4_LIBRARY}")
    else()
        message(WARNING "LZ4 not found - compression disabled")
        set(ENABLE_LZ4 OFF)
    endif()
endif()

# FFmpeg (audio/video decoding)
if(ENABLE_FFMPEG)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFMPEG QUIET IMPORTED_TARGET
            libavformat
            libavcodec
            libavutil
            libswscale
            libswresample
        )
    endif()

    if(FFMPEG_FOUND)
        message(STATUS "FFmpeg found: ${FFMPEG_VERSION}")
    else()
        message(WARNING "FFmpeg not found - audio/video loading will fall back to raw bytes")
    endif()
endif()

# Threads
find_package(Threads REQUIRED)

# SFML is used only by the legacy SFML backend. Alternative backends use the
# independent CPU renderer below.
if(MIMIR_VIZ_BACKEND STREQUAL "SFML")
    find_package(SFML 3 REQUIRED COMPONENTS Graphics Window System)
endif()
if(MIMIR_VIZ_BACKEND MATCHES "^(QT|GTK|WEB)$" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The QT/GTK/WEB Viz hosts currently require Linux")
endif()
if(MIMIR_VIZ_BACKEND STREQUAL "QT")
    find_package(Qt6 REQUIRED COMPONENTS Widgets)
elseif(MIMIR_VIZ_BACKEND STREQUAL "GTK")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(VIZ_GTK REQUIRED IMPORTED_TARGET gtk+-3.0)
endif()
if(MIMIR_VIZ_BACKEND MATCHES "^(QT|GTK|WEB)$")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(VIZ_CAIRO REQUIRED IMPORTED_TARGET cairo)
endif()

# OpenCL (calcul GPU)
if(ENABLE_OPENCL)
    find_library(OPENCL_LIBRARY NAMES OpenCL)
    if(OPENCL_LIBRARY)
        message(STATUS "OpenCL found: ${OPENCL_LIBRARY}")
    else()
        message(WARNING "OpenCL not found - GPU compute disabled")
        set(ENABLE_OPENCL OFF)
    endif()
endif()

# CUDA / ROCm roots (hints)
# Permet de configurer la détection quand les toolkits sont installés dans un chemin non standard.
set(MIMIR_CUDA_ROOT "" CACHE PATH "CUDA Toolkit root (e.g. /usr/local/cuda)")
set(MIMIR_ROCM_ROOT "" CACHE PATH "ROCm root (e.g. /opt/rocm)")

if(MIMIR_AUTO_DETECT_GPU_BACKENDS)
    message(STATUS "GPU backend auto-detection enabled (CUDA/ROCm)")
    set(ENABLE_CUDA ON)
    set(ENABLE_ROCM ON)
endif()

# CUDA (optionnel)
if(ENABLE_CUDA)
    # Aider find_package(CUDAToolkit) si CUDAToolkit_ROOT n'est pas fourni.
    if(NOT DEFINED CUDAToolkit_ROOT OR "${CUDAToolkit_ROOT}" STREQUAL "")
        if(NOT "${MIMIR_CUDA_ROOT}" STREQUAL "")
            set(CUDAToolkit_ROOT "${MIMIR_CUDA_ROOT}" CACHE PATH "" FORCE)
        elseif(DEFINED ENV{CUDA_HOME} AND NOT "$ENV{CUDA_HOME}" STREQUAL "")
            set(CUDAToolkit_ROOT "$ENV{CUDA_HOME}" CACHE PATH "" FORCE)
        elseif(DEFINED ENV{CUDA_PATH} AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
            set(CUDAToolkit_ROOT "$ENV{CUDA_PATH}" CACHE PATH "" FORCE)
        endif()
    endif()

    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        # CudaRuntime contains __global__ kernels and kernel launch syntax.
        # Finding the libraries alone does not enable CUDA compilation.
        if(NOT CMAKE_CUDA_COMPILER AND CUDAToolkit_NVCC_EXECUTABLE)
            set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}"
                CACHE FILEPATH "CUDA compiler")
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set_source_files_properties(src/runtimes/cuda/CudaRuntime.cpp
            PROPERTIES LANGUAGE CUDA)
        # Explicitly select CUDA syntax for this .cpp file, including on CMake
        # versions predating CMP0119. NVIDIA nvcc accepts -x cu.
        set_property(SOURCE src/runtimes/cuda/CudaRuntime.cpp APPEND PROPERTY
            COMPILE_OPTIONS "$<$<CUDA_COMPILER_ID:NVIDIA>:-x;cu>")
        message(STATUS "CUDAToolkit found: ${CUDAToolkit_VERSION}")
    else()
        message(WARNING "CUDAToolkit not found - CUDA compute disabled")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# ROCm/HIP (optionnel)
if(ENABLE_ROCM)
    set(MIMIR_HIP_PLATFORM "AMD" CACHE STRING "HIP platform (AMD or NVIDIA). Default: AMD")
    set_property(CACHE MIMIR_HIP_PLATFORM PROPERTY STRINGS AMD NVIDIA)

    set(_MIMIR_ROCM_HINTS "")
    if(NOT "${MIMIR_ROCM_ROOT}" STREQUAL "")
        list(APPEND _MIMIR_ROCM_HINTS "${MIMIR_ROCM_ROOT}")
    endif()
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        list(APPEND _MIMIR_ROCM_HINTS "$ENV{ROCM_PATH}")
    endif()
    if(DEFINED ENV{ROCM_HOME} AND NOT "$ENV{ROCM_HOME}" STREQUAL "")
        list(APPEND _MIMIR_ROCM_HINTS "$ENV{ROCM_HOME}")
    endif()
    if(DEFINED ENV{HIP_PATH} AND NOT "$ENV{HIP_PATH}" STREQUAL "")
        list(APPEND _MIMIR_ROCM_HINTS "$ENV{HIP_PATH}")
    endif()
    # Chemin standard sur Linux
    list(APPEND _MIMIR_ROCM_HINTS "/opt/rocm" "/usr/local/rocm")

    find_path(HIP_INCLUDE_DIR hip/hip_runtime.h
        HINTS ${_MIMIR_ROCM_HINTS}
        PATH_SUFFIXES include)
    find_library(HIP_LIBRARY NAMES amdhip64 hip
        HINTS ${_MIMIR_ROCM_HINTS}
        PATH_SUFFIXES lib lib64)
    find_path(ROCBLAS_INCLUDE_DIR rocblas/rocblas.h
        HINTS ${_MIMIR_ROCM_HINTS}
        PATH_SUFFIXES include)
    find_library(ROCBLAS_LIBRARY NAMES rocblas
        HINTS ${_MIMIR_ROCM_HINTS}
        PATH_SUFFIXES lib lib64)

    if(HIP_INCLUDE_DIR AND HIP_LIBRARY AND ROCBLAS_INCLUDE_DIR AND ROCBLAS_LIBRARY)
        message(STATUS "ROCm found: HIP=${HIP_LIBRARY} rocblas=${ROCBLAS_LIBRARY}")
    else()
        message(WARNING "ROCm/HIP not found - ROCm compute disabled")
        set(ENABLE_ROCM OFF)
    endif()
endif()

# ============================================================================
# Fichiers sources
# ============================================================================

set(MIMIR_VIZ_WINDOW_SOURCES)
if(MIMIR_VIZ_BACKEND STREQUAL "SFML")
    list(APPEND MIMIR_VIZ_WINDOW_SOURCES
        src/viz/backends/SfmlVizWindow.cpp)
elseif(MIMIR_VIZ_BACKEND STREQUAL "QT")
    list(APPEND MIMIR_VIZ_WINDOW_SOURCES
        src/viz/SoftwareVizWindow.cpp
        src/viz/backends/QtVizWindow.cpp)
elseif(MIMIR_VIZ_BACKEND STREQUAL "GTK")
    list(APPEND MIMIR_VIZ_WINDOW_SOURCES
        src/viz/SoftwareVizWindow.cpp
        src/viz/backends/GtkVizWindow.cpp)
elseif(MIMIR_VIZ_BACKEND STREQUAL "WEB")
    list(APPEND MIMIR_VIZ_WINDOW_SOURCES
        src/viz/SoftwareVizWindow.cpp
        src/viz/backends/WebVizWindow.cpp)
endif()

set(MIMIR_SOURCES
    src/runtimes/optimizers/OptimizerHost.cpp
    src/runtimes/optimizers/OptimizerDispatch.cpp
    src/runtimes/losses/LossDispatch.cpp
    src/Model.cpp
    src/ModelRuntimeConfig.cpp
    src/Models/VAETraining.cpp
    src/Encoder.cpp
    src/Tokenizer.cpp
    src/Visualizer.cpp
    ${MIMIR_VIZ_WINDOW_SOURCES}
    src/tensors.cpp
    src/Sha256.cpp
    src/scriptings/Lua/luaScripting/LuaScripting.cpp
    src/scriptings/Lua/luaScripting/LuaScriptingModelAndRegistry.cpp
    src/scriptings/Lua/luaScripting/LuaScriptingTokenizerDataset.cpp
    src/scriptings/Lua/luaScripting/LuaScriptingRuntimeAndViz.cpp
    src/scriptings/ScriptingBridgeCommon.cpp
    src/ConfigOverrides.cpp
    src/Helpers.cpp
    src/stb_image_impl.cpp
    src/runtimes/portable/PortableRuntime.cpp
    src/runtimes/portable/PortableDevice.cpp
    src/runtimes/AbstractRuntime.cpp
    src/runtimes/Autograd.cpp
    src/runtimes/RuntimeRouter.cpp
    src/runtimes/tpu/TpuRuntime.cpp
    src/runtimes/fpga/FpgaRuntime.cpp
    src/runtimes/fpga/FpgaSession.cpp
    src/runtimes/fpga/FpgaIo.cpp
    src/runtimes/ops_loss_and_grad.cpp
    src/runtimes/cpu/CpuRuntime.cpp
    src/Models/Registry/ModelArchitectures.cpp
    src/Models/External/ExternalSafeTensorsModel.cpp
    src/Models/MLP/BasicMLPModel.cpp
    src/Models/NLP/HFCLIPTextEncoder1Model.cpp
    src/Models/NLP/HFCLIPTextEncoder2Model.cpp
    src/Models/Vision/HFVaeDecoderModel.cpp
    src/Models/NLP/TransformerModel.cpp
    src/Models/NLP/CausalLMModel.cpp
    src/Models/NLP/VAETextModel.cpp
    src/Models/Vision/ViTModel.cpp
    src/Models/Vision/VAEModel.cpp
    src/Models/Vision/VAEConvModel.cpp
    src/Models/Vision/GanStyleModel.cpp
    src/Models/Vision/GanLatentModel.cpp
    src/Models/Vision/CNN/ResNetModel.cpp
    src/Models/Vision/UNetModel.cpp
    src/Models/Vision/CNN/MobileNetModel.cpp
    src/Models/Vision/CNN/VGG16Model.cpp
    src/Models/Vision/CNN/VGG19Model.cpp
    src/Models/Vision/CNN/VGG16FeatModel.cpp
    src/Models/Vision/CNN/MedicalDetectorModel.cpp
    src/Models/Vision/PatchDiscriminatorModel.cpp
    src/Models/Diffusion/DiffusionModel.cpp
    src/Models/Diffusion/CondDiffusionModel.cpp
    src/Models/Diffusion/HFSDXLTransformerBlockModel.cpp
    src/Models/Diffusion/LumenLatentDiffusionModel.cpp
    src/Models/Diffusion/SD35Model.cpp
    src/Serialization/Serialization.cpp
    src/Serialization/SafeTensorsWriter.cpp
    src/Serialization/SafeTensorsReader.cpp
    src/Serialization/RawCheckpointWriter.cpp
    src/Serialization/RawCheckpointReader.cpp
    src/Serialization/DebugJsonDump.cpp
)

set(MIMIR_HEADERS
    src/Model.hpp
    src/Models/VAETraining.hpp
    src/Encoder.hpp
    src/Tokenizer.hpp
    src/Visualizer.hpp
    src/tensors.hpp
    src/TypedTensor.hpp
    src/Sha256.hpp
    src/scriptings/ScriptingRuntime.hpp
    src/scriptings/ScriptingContext.hpp
    src/scriptings/ScriptingBridgeCommon.hpp
    src/scriptings/Lua/luaScripting/LuaScripting.hpp
    src/Helpers.hpp
    src/Layers.hpp
    src/Autograd.hpp
    src/runtimes/cpu/SIMD_Ops.hpp
    src/runtimes/cpu/HardwareOpt.hpp
    src/runtimes/vulkan/VulkanCompute.hpp
    src/runtimes/opencl/OpenCLCompute.hpp
    src/runtimes/AbstractRuntime.hpp
    src/runtimes/RuntimeRouter.hpp
    src/runtimes/LayerOps.hpp
    src/runtimes/ops_loss_and_grad.hpp
    src/runtimes/cpu/CpuRuntime.hpp
    src/runtimes/cuda/CudaRuntime.hpp
    src/runtimes/rocm/RocmRuntime.hpp
    src/runtimes/vulkan/VulkanRuntime.hpp
    src/runtimes/opencl/OpenCLRuntime.hpp
    src/runtimes/cpu/DynamicTensorAllocator.hpp
    src/AdvancedRAMManager.hpp
    src/MemoryGuard.hpp
    src/AsyncMonitor.hpp
    src/HtopDisplay.hpp
    src/runtimes/cpu/RuntimeAllocator.hpp
    src/runtimes/cpu/TypedKernels.hpp
    src/runtimes/cpu/LayerOpsExt.hpp
    src/runtimes/cpu/RuntimeLayerDispatch.hpp
    src/Models/Registry/ModelArchitectures.hpp
    src/Models/External/ExternalSafeTensorsModel.hpp
    src/Models/MLP/BasicMLPModel.hpp
    src/Models/NLP/HFCLIPTextEncoder1Model.hpp
    src/Models/NLP/HFCLIPTextEncoder2Model.hpp
    src/Models/Vision/HFVaeDecoderModel.hpp
    src/Models/NLP/TransformerModel.hpp
    src/Models/NLP/CausalLMModel.hpp
    src/Models/NLP/CausalAttentionOps.hpp
    src/Models/NLP/VAETextModel.hpp
    src/Models/Vision/ViTModel.hpp
    src/Models/Vision/VAEModel.hpp
    src/Models/Vision/GanStyleModel.hpp
    src/Models/Vision/GanLatentModel.hpp
    src/Models/Vision/CNN/ResNetModel.hpp
    src/Models/Vision/UNetModel.hpp
    src/Models/Vision/CNN/MobileNetModel.hpp
    src/Models/Vision/CNN/VGG16Model.hpp
    src/Models/Vision/CNN/VGG19Model.hpp
    src/Models/Vision/CNN/VGG16FeatModel.hpp
    src/Models/Vision/CNN/MedicalDetectorModel.hpp
    src/Models/Vision/PatchDiscriminatorModel.hpp
    src/Models/Diffusion/DiffusionModel.hpp
    src/Models/Diffusion/CondDiffusionModel.hpp
    src/Models/Diffusion/HFSDXLTransformerBlockModel.hpp
    src/Models/Diffusion/LumenLatentDiffusionModel.hpp
    src/Models/Diffusion/SD35Model.hpp
    src/Serialization/Serialization.hpp
    src/Serialization/SafeTensorsWriter.hpp
    src/Serialization/SafeTensorsReader.hpp
    src/Serialization/RawCheckpointWriter.hpp
    src/Serialization/RawCheckpointReader.hpp
    src/Serialization/DebugJsonDump.hpp
    src/stb_image.h
)

if(ENABLE_SCRIPTING_RUST)
    list(APPEND MIMIR_SOURCES src/scriptings/Rust/rustScripting/RustScripting.cpp)
    list(APPEND MIMIR_HEADERS src/scriptings/Rust/rustScripting/RustScripting.hpp)
endif()

if(ENABLE_SCRIPTING_JS)
    list(APPEND MIMIR_SOURCES src/scriptings/JavaScript/jsScripting/JSScripting.cpp)
    list(APPEND MIMIR_HEADERS src/scriptings/JavaScript/jsScripting/JSScripting.hpp)
endif()

if(ENABLE_SCRIPTING_CSHARP)
    list(APPEND MIMIR_SOURCES src/scriptings/CSharp/csharpScripting/CSharpScripting.cpp)
    list(APPEND MIMIR_HEADERS src/scriptings/CSharp/csharpScripting/CSharpScripting.hpp)
endif()

if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    list(APPEND MIMIR_SOURCES src/runtimes/cuda/CudaRuntime.cpp)
endif()

if(ENABLE_ROCM AND HIP_LIBRARY AND ROCBLAS_LIBRARY)
    list(APPEND MIMIR_SOURCES src/runtimes/rocm/RocmRuntime.cpp)
endif()

if(ENABLE_VULKAN AND Vulkan_FOUND)
    list(APPEND MIMIR_SOURCES src/runtimes/vulkan/VulkanRuntime.cpp)
endif()

if(ENABLE_OPENCL AND OPENCL_LIBRARY)
    list(APPEND MIMIR_SOURCES src/runtimes/opencl/OpenCLRuntime.cpp)
endif()

list(APPEND MIMIR_SOURCES src/runtimes/LayerOps.cpp)

# Variante de sources sans Viz (utile pour builds headless / statiques)
set(MIMIR_SOURCES_NO_SFML ${MIMIR_SOURCES})
list(REMOVE_ITEM MIMIR_SOURCES_NO_SFML src/Visualizer.cpp)
list(REMOVE_ITEM MIMIR_SOURCES_NO_SFML ${MIMIR_VIZ_WINDOW_SOURCES})

# Variante stricte pour build statique: sans runtimes GPU afin d'eviter des
# dependances dynamiques (Vulkan/OpenCL/CUDA/ROCm) lors du linkage statique.
set(MIMIR_SOURCES_STATIC ${MIMIR_SOURCES_NO_SFML})
list(REMOVE_ITEM MIMIR_SOURCES_STATIC
    src/runtimes/vulkan/VulkanRuntime.cpp
    src/runtimes/opencl/OpenCLRuntime.cpp
    src/runtimes/cuda/CudaRuntime.cpp
    src/runtimes/rocm/RocmRuntime.cpp
)

# ============================================================================
# Bibliothèque principale Mímir
# ============================================================================

if(NOT MIMIR_VIZ_BACKEND STREQUAL "NONE")
    add_library(mimir_core STATIC
        ${MIMIR_SOURCES}
        ${MIMIR_HEADERS}
    )
    target_compile_definitions(mimir_core PUBLIC ENABLE_VIZ)
else()
    add_library(mimir_core STATIC
        ${MIMIR_SOURCES_NO_SFML}
        ${MIMIR_HEADERS}
    )
endif()

if(NOT MIMIR_VIZ_BACKEND STREQUAL "NONE")
    # Graphics.hpp fait partie de l'interface de Visualizer.hpp : le backend doit
    # être visible par les consommateurs pour conserver exactement les mêmes
    # types et le même layout ABI que mimir_core.
    target_compile_definitions(mimir_core PUBLIC MIMIR_VIZ_BACKEND_${MIMIR_VIZ_BACKEND})
    if(MIMIR_VIZ_BACKEND MATCHES "^(QT|GTK|WEB)$")
        target_sources(mimir_core PRIVATE src/viz/SoftwareGraphics.cpp)
        target_link_libraries(mimir_core PUBLIC PkgConfig::VIZ_CAIRO)
    endif()
    if(MIMIR_VIZ_BACKEND STREQUAL "WEB")
        target_sources(mimir_core PRIVATE src/viz/WebHost.cpp)
    elseif(MIMIR_VIZ_BACKEND MATCHES "^(QT|GTK)$")
        target_sources(mimir_core PRIVATE src/viz/ProcessHost.cpp)
        if(MIMIR_VIZ_BACKEND STREQUAL "QT")
            add_executable(mimir_viz_host src/viz/backends/QtDesktop.cpp)
        else()
            add_executable(mimir_viz_host src/viz/backends/GtkDesktop.cpp)
        endif()
        target_compile_definitions(mimir_viz_host PRIVATE MIMIR_VIZ_BACKEND_${MIMIR_VIZ_BACKEND})
        target_link_libraries(mimir_viz_host PRIVATE Threads::Threads)
        if(MIMIR_VIZ_BACKEND STREQUAL "QT")
            target_link_libraries(mimir_viz_host PRIVATE Qt6::Widgets)
        else()
            target_link_libraries(mimir_viz_host PRIVATE PkgConfig::VIZ_GTK)
        endif()
        set_target_properties(mimir_viz_host PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${MIMIR_RUNTIME_OUTPUT_DIRECTORY}")
        install(TARGETS mimir_viz_host RUNTIME DESTINATION bin)
        add_dependencies(mimir_core mimir_viz_host)
    endif()
endif()

# Opt-in graphical integration executable (requires a display / Xvfb).
if(NOT MIMIR_VIZ_BACKEND STREQUAL "NONE")
    add_executable(mimir_viz_smoke EXCLUDE_FROM_ALL
        Tests/test_viz_window.cpp
        ${MIMIR_VIZ_WINDOW_SOURCES})
    target_include_directories(mimir_viz_smoke PRIVATE src)
    target_compile_definitions(mimir_viz_smoke PRIVATE MIMIR_VIZ_BACKEND_${MIMIR_VIZ_BACKEND})
    target_link_libraries(mimir_viz_smoke PRIVATE Threads::Threads)
    if(MIMIR_VIZ_BACKEND STREQUAL "SFML")
        target_link_libraries(mimir_viz_smoke PRIVATE SFML::Graphics SFML::Window SFML::System)
    else()
        target_sources(mimir_viz_smoke PRIVATE src/viz/SoftwareGraphics.cpp)
        target_link_libraries(mimir_viz_smoke PRIVATE PkgConfig::VIZ_CAIRO)
    endif()
    if(MIMIR_VIZ_BACKEND STREQUAL "WEB")
        target_sources(mimir_viz_smoke PRIVATE src/viz/WebHost.cpp)
    elseif(MIMIR_VIZ_BACKEND MATCHES "^(QT|GTK)$")
        target_sources(mimir_viz_smoke PRIVATE src/viz/ProcessHost.cpp)
        add_dependencies(mimir_viz_smoke mimir_viz_host)
    endif()
    set_target_properties(mimir_viz_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${MIMIR_RUNTIME_OUTPUT_DIRECTORY}")

    add_executable(mimir_viz_script_smoke EXCLUDE_FROM_ALL Tests/test_viz_script.cpp Tests/test_log_stub.cpp)
    target_link_libraries(mimir_viz_script_smoke PRIVATE mimir_core)
    set_target_properties(mimir_viz_script_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${MIMIR_RUNTIME_OUTPUT_DIRECTORY}")
    if(TARGET mimir_viz_host)
        add_dependencies(mimir_viz_script_smoke mimir_viz_host)
    endif()
    add_executable(mimir_viz_scene_smoke EXCLUDE_FROM_ALL Tests/test_viz_scene.cpp Tests/test_log_stub.cpp)
    target_link_libraries(mimir_viz_scene_smoke PRIVATE mimir_core)
    if(TARGET mimir_viz_host)
        add_dependencies(mimir_viz_scene_smoke mimir_viz_host)
    endif()
    set_target_properties(mimir_viz_scene_smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${MIMIR_RUNTIME_OUTPUT_DIRECTORY}")
endif()

target_include_directories(mimir_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtimes/cpu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

target_compile_definitions(mimir_core PUBLIC MIMIR_PROJECT_VERSION="${PROJECT_VERSION}")

if(ENABLE_SCRIPTING_RUST)
    target_compile_definitions(mimir_core PUBLIC MIMIR_ENABLE_SCRIPTING_RUST)
endif()
if(ENABLE_SCRIPTING_JS)
    target_compile_definitions(mimir_core PUBLIC MIMIR_ENABLE_SCRIPTING_JS)
endif()
if(ENABLE_SCRIPTING_CSHARP)
    target_compile_definitions(mimir_core PUBLIC MIMIR_ENABLE_SCRIPTING_CSHARP)
endif()

# 🛡️ Appliquer la macro MIMIR_ENABLE_LEGACY_PARAMS si activée
if(MIMIR_ENABLE_LEGACY_PARAMS)
    target_compile_definitions(mimir_core PRIVATE MIMIR_ENABLE_LEGACY_PARAMS)
    message(WARNING "⚠️  Legacy params structure ENABLED - High RAM usage!")
else()
    message(STATUS "✅ Legacy params structure DISABLED (recommended for production)")
endif()

# Lier les dépendances
target_link_libraries(mimir_core PUBLIC
    ${LUA_LIBRARIES}
    Threads::Threads
)

if(ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(mimir_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(ENABLE_VULKAN AND Vulkan_FOUND)
    target_compile_definitions(mimir_core PUBLIC ENABLE_VULKAN)
    target_link_libraries(mimir_core PUBLIC Vulkan::Vulkan)
endif()

if(ENABLE_OPENCL AND OPENCL_LIBRARY)
    target_compile_definitions(mimir_core PUBLIC ENABLE_OPENCL)
    target_link_libraries(mimir_core PUBLIC ${OPENCL_LIBRARY})
endif()

if(ENABLE_CUDA AND CUDAToolkit_FOUND)
    target_compile_definitions(mimir_core PUBLIC ENABLE_CUDA)
    if(TARGET CUDA::cudart)
        target_link_libraries(mimir_core PUBLIC CUDA::cudart)
    endif()
    if(TARGET CUDA::cublas)
        target_link_libraries(mimir_core PUBLIC CUDA::cublas)
    endif()
endif()

if(ENABLE_ROCM AND HIP_LIBRARY AND ROCBLAS_LIBRARY)
    target_compile_definitions(mimir_core PUBLIC ENABLE_ROCM)
    if(MIMIR_HIP_PLATFORM STREQUAL "NVIDIA")
        target_compile_definitions(mimir_core PUBLIC __HIP_PLATFORM_NVIDIA__)
    else()
        target_compile_definitions(mimir_core PUBLIC __HIP_PLATFORM_AMD__)
    endif()
    target_include_directories(mimir_core PUBLIC ${HIP_INCLUDE_DIR} ${ROCBLAS_INCLUDE_DIR})
    target_link_libraries(mimir_core PUBLIC ${HIP_LIBRARY} ${ROCBLAS_LIBRARY} ${CMAKE_DL_LIBS})
endif()

if(ENABLE_LZ4 AND LZ4_LIBRARY)
    target_compile_definitions(mimir_core PUBLIC ENABLE_LZ4)
    target_link_libraries(mimir_core PUBLIC ${LZ4_LIBRARY})
endif()

if(ENABLE_FFMPEG AND FFMPEG_FOUND)
    target_compile_definitions(mimir_core PUBLIC MIMIR_ENABLE_FFMPEG)
    target_link_libraries(mimir_core PUBLIC PkgConfig::FFMPEG)
endif()

# SFML (visualisation)
if(MIMIR_VIZ_BACKEND STREQUAL "SFML" AND SFML_FOUND)
    if(TARGET SFML::Graphics)
        target_link_libraries(mimir_core PUBLIC SFML::Graphics SFML::Window SFML::System)
    else()
        target_link_libraries(mimir_core PUBLIC sfml-graphics sfml-window sfml-system)
    endif()
endif()

# ============================================================================
# Exécutable principal
# ============================================================================

add_executable(mimir src/main.cpp
    src/runtimes/cpu/HardwareOpt.hpp)
target_link_libraries(mimir PRIVATE mimir_core)

# Copier les shaders compilés à côté du binaire (best-effort)
if(TARGET mimir_shaders)
    add_dependencies(mimir mimir_shaders)
    add_custom_command(TARGET mimir POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:mimir>/shaders"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/shaders"
            "$<TARGET_FILE_DIR:mimir>/shaders"
        COMMENT "Copy Vulkan shaders to bin/shaders"
    )
endif()

# Installation dans bin/
set_target_properties(mimir PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${MIMIR_RUNTIME_OUTPUT_DIRECTORY}"
)

# ==========================================================================
# Exécutable statique (optionnel) : mimir_static
# ==========================================================================

if(BUILD_MIMIR_STATIC)
    # Trouver la lib Lua statique si possible (évite le lien contre liblua*.so)
    set(_MIMIR_OLD_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    find_library(LUA_STATIC_LIBRARY NAMES lua5.3 lua)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_MIMIR_OLD_FIND_LIBRARY_SUFFIXES})

    if(NOT LUA_STATIC_LIBRARY)
        message(WARNING "BUILD_MIMIR_STATIC=ON mais liblua statique introuvable (*.a). Tentative avec ${LUA_LIBRARIES} (peut échouer).")
    else()
        message(STATUS "Lua static library: ${LUA_STATIC_LIBRARY}")
    endif()

    # Variante minimale de la lib (sans SFML ni runtimes GPU)
    # pour maximiser les chances d'un lien 100% statique.
    add_library(mimir_core_static STATIC
        ${MIMIR_SOURCES_STATIC}
        ${MIMIR_HEADERS}
    )

    target_include_directories(mimir_core_static PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/runtimes/cpu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    )

    target_compile_definitions(mimir_core_static PUBLIC MIMIR_PROJECT_VERSION="${PROJECT_VERSION}")

    if(ENABLE_SCRIPTING_RUST)
        target_compile_definitions(mimir_core_static PUBLIC MIMIR_ENABLE_SCRIPTING_RUST)
    endif()
    if(ENABLE_SCRIPTING_JS)
        target_compile_definitions(mimir_core_static PUBLIC MIMIR_ENABLE_SCRIPTING_JS)
    endif()
    if(ENABLE_SCRIPTING_CSHARP)
        target_compile_definitions(mimir_core_static PUBLIC MIMIR_ENABLE_SCRIPTING_CSHARP)
    endif()

    if(MIMIR_ENABLE_LEGACY_PARAMS)
        target_compile_definitions(mimir_core_static PRIVATE MIMIR_ENABLE_LEGACY_PARAMS)
    endif()

    if(ENABLE_FFMPEG AND FFMPEG_FOUND)
        target_compile_definitions(mimir_core_static PUBLIC MIMIR_ENABLE_FFMPEG)
        target_link_libraries(mimir_core_static PUBLIC PkgConfig::FFMPEG)
    endif()

    if(LUA_STATIC_LIBRARY)
        target_link_libraries(mimir_core_static PUBLIC
            ${LUA_STATIC_LIBRARY}
            Threads::Threads
        )
    else()
        target_link_libraries(mimir_core_static PUBLIC
            ${LUA_LIBRARIES}
            Threads::Threads
        )
    endif()

    add_executable(mimir_static src/main.cpp)
    target_link_libraries(mimir_static PRIVATE mimir_core_static)

    set_target_properties(mimir_static PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${MIMIR_RUNTIME_OUTPUT_DIRECTORY}"
    )

    if(UNIX AND NOT APPLE)
        # Tente un lien 100% statique. Si ça échoue, il manque probablement des *.a
        # (ex: liblua, libgomp, libc, etc.).
        target_link_options(mimir_static PRIVATE -static -static-libgcc -static-libstdc++)
        set_target_properties(mimir_static PROPERTIES
            LINK_SEARCH_START_STATIC ON
            LINK_SEARCH_END_STATIC ON
        )
    endif()
endif()

message(STATUS "Tests/exemples désactivés (obsolètes)")

# ============================================================================
# Outil C++ de détection/reconnaissance faciale (OpenCV)
# ============================================================================

if(MIMIR_ENABLE_FACE_RECOGNITION_TOOL)
    find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs objdetect highgui videoio dnn)
    if(OpenCV_FOUND)
        add_executable(face_recognition_tool examples/face_recognition_tool.cpp)
        target_link_libraries(face_recognition_tool PRIVATE ${OpenCV_LIBS})
        target_include_directories(face_recognition_tool PRIVATE ${OpenCV_INCLUDE_DIRS})
        set_target_properties(face_recognition_tool PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${MIMIR_RUNTIME_OUTPUT_DIRECTORY}"
        )
        message(STATUS "Face recognition tool: enabled (${OpenCV_VERSION})")
    endif()
endif()

# ============================================================================
# Tests unitaires (CTest)
# ============================================================================

option(MIMIR_ENABLE_TESTS "Build unit tests (CTest)" OFF)
if(MIMIR_ENABLE_TESTS)
    include(CTest)
    if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Tests/CMakeLists.txt")
        add_subdirectory(Tests)
    endif()
endif()

# ============================================================================
# Scripts Lua
# ============================================================================

# Copier les scripts Lua dans le dossier de build (optionnel)
file(GLOB LUA_SCRIPTS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.lua")
file(COPY ${LUA_SCRIPTS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/scripts)

# ============================================================================
# Installation
# ============================================================================

# Installer l'exécutable principal
install(TARGETS mimir
    RUNTIME DESTINATION bin
    COMPONENT Runtime
)

if(BUILD_MIMIR_STATIC)
    install(TARGETS mimir_static
        RUNTIME DESTINATION bin
    )
endif()

# Installer la bibliothèque
install(TARGETS mimir_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# Installer les headers
install(FILES ${MIMIR_HEADERS}
    DESTINATION include/mimir
)

# Installer les scripts Lua
install(DIRECTORY scripts/
    DESTINATION share/mimir/scripts
    COMPONENT Runtime
    FILES_MATCHING PATTERN "*.lua"
)

# Installer la documentation
install(DIRECTORY docs/
    DESTINATION share/mimir/docs
    FILES_MATCHING PATTERN "*.md"
)

# ============================================================================
# AppImage Linux portable (build non natif)
# ============================================================================

if(UNIX AND NOT APPLE AND NOT MIMIR_APPIMAGE_INNER_BUILD)
    set(MIMIR_LINUXDEPLOY "" CACHE FILEPATH
        "Path to linuxdeploy or linuxdeploy-x86_64.AppImage")
    if(MIMIR_LINUXDEPLOY)
        set(_MIMIR_LINUXDEPLOY "${MIMIR_LINUXDEPLOY}")
    else()
        find_program(_MIMIR_LINUXDEPLOY
            NAMES linuxdeploy linuxdeploy-x86_64.AppImage
            HINTS "${CMAKE_CURRENT_SOURCE_DIR}/tools")
    endif()

    set(MIMIR_APPIMAGE_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build_appimage")
    set(MIMIR_APPDIR "${MIMIR_APPIMAGE_BUILD_DIR}/AppDir")
    set(MIMIR_APPIMAGE_DIST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dist")
    set(MIMIR_APPIMAGE_DESKTOP "${CMAKE_CURRENT_BINARY_DIR}/appimage/mimir.desktop")
    set(MIMIR_APPIMAGE_ICON "${CMAKE_CURRENT_BINARY_DIR}/appimage/mimir.svg")
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" MIMIR_APPIMAGE_ARCH)
    if(MIMIR_APPIMAGE_ARCH MATCHES "^(amd64|x86_64)$")
        set(MIMIR_APPIMAGE_ARCH "x86_64")
        set(MIMIR_APPIMAGE_ENABLE_SIMD ON)
    elseif(MIMIR_APPIMAGE_ARCH MATCHES "^(aarch64|arm64)$")
        set(MIMIR_APPIMAGE_ARCH "aarch64")
        set(MIMIR_APPIMAGE_ENABLE_SIMD OFF)
    else()
        set(MIMIR_APPIMAGE_ENABLE_SIMD OFF)
    endif()

    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/appimage")
    file(WRITE "${MIMIR_APPIMAGE_DESKTOP}"
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Mimir\n"
        "Comment=Machine learning framework and scripting runtime\n"
        "Exec=mimir\n"
        "Icon=mimir\n"
        "Terminal=true\n"
        "Categories=Development;Science;\n")
    file(WRITE "${MIMIR_APPIMAGE_ICON}"
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"256\" height=\"256\" viewBox=\"0 0 256 256\">"
        "<rect width=\"256\" height=\"256\" rx=\"32\" fill=\"#15191f\"/>"
        "<path d=\"M48 192V64h32l48 72 48-72h32v128h-32v-76l-48 68-48-68v76z\" fill=\"#62d6a8\"/>"
        "</svg>\n")

    if(_MIMIR_LINUXDEPLOY)
        add_custom_target(appimage
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${MIMIR_APPIMAGE_BUILD_DIR}"
            COMMAND ${CMAKE_COMMAND} -S "${CMAKE_CURRENT_SOURCE_DIR}"
                    -B "${MIMIR_APPIMAGE_BUILD_DIR}"
                    -G "${CMAKE_GENERATOR}"
                    -DCMAKE_BUILD_TYPE=Release
                    -DCMAKE_INSTALL_PREFIX=/usr
                    -DMIMIR_APPIMAGE_INNER_BUILD=ON
                    -DMIMIR_PORTABLE_BUILD=ON
                    -DMIMIR_ENABLE_TESTS=OFF
                    -DENABLE_SIMD=${MIMIR_APPIMAGE_ENABLE_SIMD}
                    -DENABLE_VULKAN=ON
                    -DENABLE_OPENCL=ON
                    -DENABLE_CUDA=OFF
                    -DENABLE_ROCM=OFF
                    -DENABLE_SFML=ON
            COMMAND ${CMAKE_COMMAND} --build "${MIMIR_APPIMAGE_BUILD_DIR}"
                    --config Release --target mimir --parallel
            COMMAND ${CMAKE_COMMAND} -E env
                    DESTDIR=${MIMIR_APPDIR}
                    ${CMAKE_COMMAND} --install "${MIMIR_APPIMAGE_BUILD_DIR}"
                    --config Release --prefix /usr --component Runtime
            COMMAND ${CMAKE_COMMAND} -E make_directory "${MIMIR_APPIMAGE_DIST_DIR}"
            COMMAND ${CMAKE_COMMAND} -E env
                    VERSION=${PROJECT_VERSION}
                    ARCH=${MIMIR_APPIMAGE_ARCH}
                    OUTPUT=${MIMIR_APPIMAGE_DIST_DIR}/Mimir-${PROJECT_VERSION}-${MIMIR_APPIMAGE_ARCH}.AppImage
                    "${_MIMIR_LINUXDEPLOY}"
                    --appdir "${MIMIR_APPDIR}"
                    --executable "${MIMIR_APPDIR}/usr/bin/mimir"
                    --desktop-file "${MIMIR_APPIMAGE_DESKTOP}"
                    --icon-file "${MIMIR_APPIMAGE_ICON}"
                    --output appimage
            COMMENT "Building portable Mimir AppImage (no -march=native)"
            USES_TERMINAL
            VERBATIM
        )
    else()
        add_custom_target(appimage
            COMMAND ${CMAKE_COMMAND} -E echo
                    "linuxdeploy not found. Set -DMIMIR_LINUXDEPLOY=/path/to/linuxdeploy-x86_64.AppImage"
            COMMAND ${CMAKE_COMMAND} -E false
            COMMENT "Cannot build AppImage without linuxdeploy"
            VERBATIM
        )
    endif()
endif()

# ============================================================================
# Résumé de la configuration
# ============================================================================

message(STATUS "")
message(STATUS "╔═══════════════════════════════════════════════════════════════╗")
message(STATUS "║          Mímir Framework v${PROJECT_VERSION} - Configuration               ║")
message(STATUS "╚═══════════════════════════════════════════════════════════════╝")
message(STATUS "")
message(STATUS "Build type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler:        ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard:        C++${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Features:")
message(STATUS "  SIMD (AVX2/FMA):   ${ENABLE_SIMD}")
message(STATUS "  OpenMP:            ${ENABLE_OPENMP}")
message(STATUS "  Vulkan Compute:    ${ENABLE_VULKAN}")
message(STATUS "  OpenCL Compute:    ${ENABLE_OPENCL}")
message(STATUS "  CUDA Compute:      ${ENABLE_CUDA}")
message(STATUS "  ROCm Compute:      ${ENABLE_ROCM}")
message(STATUS "  LZ4 Compression:   ${ENABLE_LZ4}")
message(STATUS "  REST Bridge:       ${ENABLE_SCRIPTING_REST}")
message(STATUS "  Rust Bridge:       ${ENABLE_SCRIPTING_RUST}")
message(STATUS "  JavaScript Bridge: ${ENABLE_SCRIPTING_JS}")
message(STATUS "  C# Bridge:         ${ENABLE_SCRIPTING_CSHARP}")
message(STATUS "")
message(STATUS "Build options:")
message(STATUS "  Examples:          ${BUILD_EXAMPLES}")
message(STATUS "")
message(STATUS "Install prefix:      ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")

set_source_files_properties(src/runtimes/tpu/TpuRuntime.cpp PROPERTIES
    COMPILE_DEFINITIONS "MIMIR_TPU_WORKER=\"${CMAKE_CURRENT_SOURCE_DIR}/src/runtimes/tpu/worker.py\"")

install(FILES src/runtimes/tpu/worker.py DESTINATION share/mimir/tpu COMPONENT Runtime)
