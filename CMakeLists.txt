cmake_minimum_required(VERSION 3.15)
project(Mimir VERSION 2.4.0 LANGUAGES CXX)

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
option(ENABLE_LZ4 "Enable LZ4 compression" ON)
option(ENABLE_SFML "Enable SFML visualization" ON)
option(BUILD_EXAMPLES "Build example executables" OFF)

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

# ============================================================================
# Détection des capacités hardware
# ============================================================================

include(CheckCXXCompilerFlag)

if(ENABLE_SIMD)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
    check_cxx_compiler_flag("-mf16c" COMPILER_SUPPORTS_F16C)
    check_cxx_compiler_flag("-mbmi2" COMPILER_SUPPORTS_BMI2)
    
    if(COMPILER_SUPPORTS_AVX2)
        add_compile_options(-mavx2)
        message(STATUS "AVX2 support: Enabled")
    endif()
    
    if(COMPILER_SUPPORTS_FMA)
        add_compile_options(-mfma)
        message(STATUS "FMA support: Enabled")
    endif()
    
    if(COMPILER_SUPPORTS_F16C)
        add_compile_options(-mf16c)
        message(STATUS "F16C support: Enabled")
    endif()
    
    if(COMPILER_SUPPORTS_BMI2)
        add_compile_options(-mbmi2)
        message(STATUS "BMI2 support: Enabled")
    endif()
endif()

# ============================================================================
# Options de compilation
# ============================================================================

# Flags communs
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
    -Wno-unused-variable
)

# Flags de release
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG -march=native")

# Flags de debug
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")

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

        set(SHADER_LINEAR_SRC "${MIMIR_SHADER_SRC_DIR}/linear_forward.comp")
        set(SHADER_LINEAR_SPV "${MIMIR_SHADER_OUT_DIR}/linear_forward.comp.spv")

        add_custom_command(
            OUTPUT "${SHADER_LINEAR_SPV}"
            COMMAND ${_MIMIR_GLSLANG_VALIDATOR_CMD} -V "${SHADER_LINEAR_SRC}" -o "${SHADER_LINEAR_SPV}"
            DEPENDS "${SHADER_LINEAR_SRC}"
            COMMENT "Compiling Vulkan shader linear_forward.comp -> SPIR-V"
            VERBATIM
        )

        add_custom_target(mimir_shaders ALL DEPENDS "${SHADER_LINEAR_SPV}")
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

# Threads
find_package(Threads REQUIRED)

# SFML (visualisation)
if(ENABLE_SFML)
    find_package(SFML 2.5 COMPONENTS graphics window system)
    if(SFML_FOUND)
        message(STATUS "SFML found: ${SFML_VERSION}")
    else()
        message(WARNING "SFML not found - Visualization may be limited")
    endif()
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

# ============================================================================
# Fichiers sources
# ============================================================================

set(MIMIR_SOURCES
    src/Model.cpp
    src/Encoder.cpp
    src/Tokenizer.cpp
    src/ConfigOverrides.cpp
    src/Visualizer.cpp
    src/tensors.cpp
    src/Sha256.cpp
    src/LuaScripting.cpp
    src/ConfigOverrides.cpp
    src/Helpers.cpp
    src/stb_image_impl.cpp
    src/Models/Registry/ModelArchitectures.cpp
    src/Models/MLP/BasicMLPModel.cpp
    src/Models/NLP/TransformerModel.cpp
    src/Models/NLP/VAETextModel.cpp
    src/Models/Vision/ViTModel.cpp
    src/Models/Vision/VAEModel.cpp
    src/Models/Vision/VAEConvModel.cpp
    src/Models/Vision/GanLatentModel.cpp
    src/Models/Vision/CNN/ResNetModel.cpp
    src/Models/Vision/UNetModel.cpp
    src/Models/Vision/CNN/MobileNetModel.cpp
    src/Models/Vision/CNN/VGG16Model.cpp
    src/Models/Vision/CNN/VGG19Model.cpp
    src/Models/Vision/CNN/VGG16FeatModel.cpp
    src/Models/Vision/PatchDiscriminatorModel.cpp
    src/Models/Diffusion/DiffusionModel.cpp
    src/Models/Diffusion/CondDiffusionModel.cpp
    src/Models/Diffusion/SD35Model.cpp
    src/Models/Diffusion/PonyXLDDPMModel.cpp
    src/Serialization/Serialization.cpp
    src/Serialization/SafeTensorsWriter.cpp
    src/Serialization/SafeTensorsReader.cpp
    src/Serialization/RawCheckpointWriter.cpp
    src/Serialization/RawCheckpointReader.cpp
    src/Serialization/DebugJsonDump.cpp
)

set(MIMIR_HEADERS
    src/Model.hpp
    src/Encoder.hpp
    src/Tokenizer.hpp
    src/ConfigOverrides.hpp
    src/Visualizer.hpp
    src/tensors.hpp
    src/Sha256.hpp
    src/LuaScripting.hpp
    src/Helpers.hpp
    src/Layers.hpp
    src/Autograd.hpp
    src/SIMD_Ops.hpp
    src/HardwareOpt.hpp
    src/VulkanCompute.hpp
    src/DynamicTensorAllocator.hpp
    src/AdvancedRAMManager.hpp
    src/MemoryGuard.hpp
    src/AsyncMonitor.hpp
    src/HtopDisplay.hpp
    src/RuntimeAllocator.hpp
    src/LayerOpsExt.hpp
    src/Models/Registry/ModelArchitectures.hpp
    src/Models/MLP/BasicMLPModel.hpp
    src/Models/NLP/TransformerModel.hpp
    src/Models/NLP/VAETextModel.hpp
    src/Models/Vision/ViTModel.hpp
    src/Models/Vision/VAEModel.hpp
    src/Models/Vision/GanLatentModel.hpp
    src/Models/Vision/CNN/ResNetModel.hpp
    src/Models/Vision/UNetModel.hpp
    src/Models/Vision/CNN/MobileNetModel.hpp
    src/Models/Vision/CNN/VGG16Model.hpp
    src/Models/Vision/CNN/VGG19Model.hpp
    src/Models/Vision/CNN/VGG16FeatModel.hpp
    src/Models/Vision/PatchDiscriminatorModel.hpp
    src/Models/Diffusion/DiffusionModel.hpp
    src/Models/Diffusion/CondDiffusionModel.hpp
    src/Models/Diffusion/SD35Model.hpp
    src/Models/Diffusion/PonyXLDDPMModel.hpp
    src/Serialization/Serialization.hpp
    src/Serialization/SafeTensorsWriter.hpp
    src/Serialization/SafeTensorsReader.hpp
    src/Serialization/RawCheckpointWriter.hpp
    src/Serialization/RawCheckpointReader.hpp
    src/Serialization/DebugJsonDump.hpp
    src/stb_image.h
)

# Variante de sources sans SFML (utile pour builds headless / statiques)
set(MIMIR_SOURCES_NO_SFML ${MIMIR_SOURCES})
list(REMOVE_ITEM MIMIR_SOURCES_NO_SFML src/Visualizer.cpp)

# ============================================================================
# Bibliothèque principale Mímir
# ============================================================================

if(ENABLE_SFML AND SFML_FOUND)
    add_library(mimir_core STATIC
        ${MIMIR_SOURCES}
        ${MIMIR_HEADERS}
    )
    target_compile_definitions(mimir_core PUBLIC ENABLE_SFML)
else()
    add_library(mimir_core STATIC
        ${MIMIR_SOURCES_NO_SFML}
        ${MIMIR_HEADERS}
    )
endif()

target_include_directories(mimir_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

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

if(ENABLE_LZ4 AND LZ4_LIBRARY)
    target_compile_definitions(mimir_core PUBLIC ENABLE_LZ4)
    target_link_libraries(mimir_core PUBLIC ${LZ4_LIBRARY})
endif()

# SFML (visualisation)
if(ENABLE_SFML AND SFML_FOUND)
    target_link_libraries(mimir_core PUBLIC sfml-graphics sfml-window sfml-system)
endif()

# OpenCL (calcul GPU)
# (déjà géré via ENABLE_OPENCL)

# ============================================================================
# Exécutable principal
# ============================================================================

add_executable(mimir src/main.cpp)
target_link_libraries(mimir PRIVATE mimir_core)

# Copier les shaders compilés à côté du binaire (best-effort)
if(TARGET mimir_shaders)
    add_dependencies(mimir mimir_shaders)
    add_custom_command(TARGET mimir POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:mimir>/shaders"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/shaders/linear_forward.comp.spv"
            "$<TARGET_FILE_DIR:mimir>/shaders/linear_forward.comp.spv"
        COMMENT "Copy Vulkan shaders to bin/shaders"
    )
endif()

# Installation dans bin/
set_target_properties(mimir PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
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

    # Variante minimale de la lib (évite Vulkan/SFML/OpenCL/LZ4/OpenMP par défaut)
    # pour maximiser les chances d'un lien 100% statique.
    add_library(mimir_core_static STATIC
        ${MIMIR_SOURCES_NO_SFML}
        ${MIMIR_HEADERS}
    )

    target_include_directories(mimir_core_static PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    )

    if(MIMIR_ENABLE_LEGACY_PARAMS)
        target_compile_definitions(mimir_core_static PRIVATE MIMIR_ENABLE_LEGACY_PARAMS)
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
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
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
    FILES_MATCHING PATTERN "*.lua"
)

# Installer la documentation
install(DIRECTORY docs/
    DESTINATION share/mimir/docs
    FILES_MATCHING PATTERN "*.md"
)

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
message(STATUS "  LZ4 Compression:   ${ENABLE_LZ4}")
message(STATUS "")
message(STATUS "Build options:")
message(STATUS "  Examples:          ${BUILD_EXAMPLES}")
message(STATUS "")
message(STATUS "Install prefix:      ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
option(MIMIR_ENABLE_TESTS "Build unit tests (CTest)" ON)
if(MIMIR_ENABLE_TESTS)
    include(CTest)
    if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Tests/CMakeLists.txt")
        # Use a unique binary dir to avoid collisions when multiple subprojects
        # (or previous configurations) also add a 'Tests' directory.
        add_subdirectory(Tests "${CMAKE_CURRENT_BINARY_DIR}/_mimir_tests")
    endif()
endif()