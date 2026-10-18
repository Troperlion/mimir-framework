#include "LuaScripting.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"
#include "Serialization/DebugJsonDump.hpp"
#include "DType.hpp"
#include "AdvancedRAMManager.hpp"
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#include "AsyncMonitor.hpp"
#include "Helpers.hpp"
#include "scriptings/ScriptingBridgeCommon.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <unordered_set>
#include <type_traits>
#include <utility>

namespace {

template <typename T, typename = void>
struct _mimir_has_overrides_enabled : std::false_type {};

template <typename T>
struct _mimir_has_overrides_enabled<T, std::void_t<decltype(std::declval<T>().overrides_enabled)>>
    : std::true_type {};

template <typename T>
bool _mimir_live_params_overrides_enabled(const T& p) {
    if constexpr (_mimir_has_overrides_enabled<T>::value) {
        return static_cast<bool>(p.overrides_enabled);
    }
    // Compat: anciennes versions n'avaient pas de toggle => considérer "ON".
    return true;
}

}  // namespace

// ============================================================================
// Constructeur / Destructeur
// ============================================================================

LuaScripting::LuaScripting() {
    L = luaL_newstate();
    luaL_openlibs(L);  // Charger les bibliothèques standard Lua
    setString(ScriptingBridgeCommon::kEnvRootWork,
              ScriptingBridgeCommon::ensureRootWorkEnv());
    registerAPI();
}

void LuaScripting::setArgs(const std::string& script_path, const std::vector<std::string>& script_args) {
    cacheArgs(script_path, script_args);
    // Reproduit le comportement de l’interpréteur Lua en exposant une table globale `arg`.
    // Convention: arg[0] = chemin du script, arg[1..n] = arguments.
    lua_newtable(L);

    lua_pushstring(L, script_path.c_str());
    lua_rawseti(L, -2, 0);

    for (size_t i = 0; i < script_args.size(); ++i) {
        lua_pushstring(L, script_args[i].c_str());
        lua_rawseti(L, -2, static_cast<lua_Integer>(i + 1));
    }

    lua_setglobal(L, ScriptingContext::kGlobalArg);
}

LuaScripting::~LuaScripting() {
    if (L) {
        lua_close(L);
    }
}

// ============================================================================
// Chargement et exécution de scripts
// ============================================================================

bool LuaScripting::loadScript(const std::string& filepath) {
    if (luaL_dofile(L, filepath.c_str()) != LUA_OK) {
        std::cerr << "Erreur Lua: " << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
        return false;
    }
    return true;
}

bool LuaScripting::executeScript(const std::string& code) {
    if (luaL_dostring(L, code.c_str()) != LUA_OK) {
        std::cerr << "Erreur Lua: " << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
        return false;
    }
    return true;
}

bool LuaScripting::callFunction(const std::string& function_name) {
    lua_getglobal(L, function_name.c_str());
    if (!lua_isfunction(L, -1)) {
        std::cerr << "Fonction Lua introuvable: " << function_name << std::endl;
        lua_pop(L, 1);
        return false;
    }
    
    if (lua_pcall(L, 0, 0, 0) != LUA_OK) {
        std::cerr << "Erreur lors de l'appel de " << function_name << ": " 
                  << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
        return false;
    }
    
    return true;
}

// ============================================================================
// Getters / Setters
// ============================================================================

std::string LuaScripting::getString(const std::string& var_name) {
    lua_getglobal(L, var_name.c_str());
    std::string result = lua_isstring(L, -1) ? lua_tostring(L, -1) : "";
    lua_pop(L, 1);
    return result;
}

double LuaScripting::getNumber(const std::string& var_name) {
    lua_getglobal(L, var_name.c_str());
    double result = lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 0.0;
    lua_pop(L, 1);
    return result;
}

bool LuaScripting::getBoolean(const std::string& var_name) {
    lua_getglobal(L, var_name.c_str());
    bool result = lua_isboolean(L, -1) ? lua_toboolean(L, -1) : false;
    lua_pop(L, 1);
    return result;
}

void LuaScripting::setString(const std::string& var_name, const std::string& value) {
    lua_pushstring(L, value.c_str());
    lua_setglobal(L, var_name.c_str());
}

void LuaScripting::setNumber(const std::string& var_name, double value) {
    lua_pushnumber(L, value);
    lua_setglobal(L, var_name.c_str());
}

void LuaScripting::setBoolean(const std::string& var_name, bool value) {
    lua_pushboolean(L, value);
    lua_setglobal(L, var_name.c_str());
}

void LuaScripting::setSystemConfig(const json& conf, const std::string& conf_path, const std::string& conf_dir) {
    setString(ScriptingContext::kGlobalConfPath, conf_path);
    setString(ScriptingContext::kGlobalConfDir, conf_dir);
    jsonToLuaTable(L, conf);
    lua_setglobal(L, ScriptingContext::kGlobalConf);
}

// ============================================================================
// Enregistrement de l'API
// ============================================================================

void LuaScripting::registerAPI() {
    // ========== Table "Mimir" (namespace racine) ==========
    lua_newtable(L);
    
    // ========== Sous-table "Mimir.Model" ==========
    lua_newtable(L);
    
    // Gestion basique
    lua_pushcfunction(L, lua_createModel);
    lua_setfield(L, -2, "create");

    lua_pushcfunction(L, lua_createEmptyModel);
    lua_setfield(L, -2, "create_empty");

    lua_pushcfunction(L, lua_createModelFromConfig);
    lua_setfield(L, -2, "create_from_config");
    
    lua_pushcfunction(L, lua_buildModel);
    lua_setfield(L, -2, "build");
    
    lua_pushcfunction(L, lua_trainModel);
    lua_setfield(L, -2, "train");
    lua_pushcfunction(L, lua_visionTrainStep);
    lua_setfield(L, -2, "vision_train_step");
    
    lua_pushcfunction(L, lua_inferModel);
    lua_setfield(L, -2, "infer");
    
    lua_pushcfunction(L, lua_saveModel);
    lua_setfield(L, -2, "save");
    
    lua_pushcfunction(L, lua_loadModel);
    lua_setfield(L, -2, "load");
    
    // Gestion des paramètres
    lua_pushcfunction(L, lua_allocateParams);
    lua_setfield(L, -2, "allocate_params");
    
    lua_pushcfunction(L, lua_initWeights);
    lua_setfield(L, -2, "init_weights");
    
    lua_pushcfunction(L, lua_totalParams);
    lua_setfield(L, -2, "total_params");

    lua_pushcfunction(L, lua_modelGetConfig);
    lua_setfield(L, -2, "get_config");

    lua_pushcfunction(L, lua_getModelLayers);
    lua_setfield(L, -2, "get_layers");

    lua_pushcfunction(L, lua_clearModelLayers);
    lua_setfield(L, -2, "clear_layers");
    
    lua_pushcfunction(L, lua_pushLayer);
    lua_setfield(L, -2, "push_layer");
    
    lua_pushcfunction(L, lua_setLayerIO);
    lua_setfield(L, -2, "set_layer_io");
    
    // Forward/Backward
    lua_pushcfunction(L, lua_forwardPass);
    lua_setfield(L, -2, "forward");
    
    lua_pushcfunction(L, lua_backwardPass);
    lua_setfield(L, -2, "backward");
    
    lua_pushcfunction(L, lua_optimizerStep);
    lua_setfield(L, -2, "optimizer_step");
    
    lua_pushcfunction(L, lua_zeroGradients);
    lua_setfield(L, -2, "zero_grads");
    
    lua_pushcfunction(L, lua_getGradients);
    lua_setfield(L, -2, "get_gradients");

    lua_pushcfunction(L, lua_lumenText2Img);
    lua_setfield(L, -2, "lumen_text2img");

    lua_pushcfunction(L, lua_lumenBeginVaeCalibration);
    lua_setfield(L, -2, "lumen_begin_vae_calibration");

    lua_pushcfunction(L, lua_lumenAddVaeCalibrationImage);
    lua_setfield(L, -2, "lumen_add_vae_calibration_image");

    lua_pushcfunction(L, lua_lumenFinishVaeCalibration);
    lua_setfield(L, -2, "lumen_finish_vae_calibration");

    lua_pushcfunction(L, lua_lumenTrainStep);
    lua_setfield(L, -2, "lumen_train_step");

    lua_pushcfunction(L, lua_lumenValidateStep);
    lua_setfield(L, -2, "lumen_validate_step");
    
    // Hardware
    lua_pushcfunction(L, lua_setHardwareAccel);
    lua_setfield(L, -2, "set_hardware");
    
    lua_pushcfunction(L, lua_getHardwareCaps);
    lua_setfield(L, -2, "hardware_caps");

    lua_pushcfunction(L, lua_conditioningPrompt);
    lua_setfield(L, -2, "conditioning_prompt");

    lua_pushcfunction(L, lua_runtimeOps);
    lua_setfield(L, -2, "runtime_ops");

    // DType selection
    lua_pushcfunction(L, lua_modelDType);
    lua_setfield(L, -2, "dtype");

    // Expose both Mimir.Model and Mimir.model (lowercase alias)
    lua_pushvalue(L, -1);
    lua_setfield(L, -3, "model");  // Mimir.model

    lua_setfield(L, -2, "Model");  // Mimir.Model

    // ========== Sous-table "Mimir.Benchmark" ==========
    lua_newtable(L);
    lua_pushcfunction(L, lua_benchmarkNow);
    lua_setfield(L, -2, "now");
    lua_setfield(L, -2, "Benchmark");
    
    // ========== Sous-table "Mimir.Architectures" ==========
    lua_newtable(L);

    // Registry helpers
    lua_pushcfunction(L, lua_archAvailable);
    lua_setfield(L, -2, "available");

    lua_pushcfunction(L, lua_archDefaultConfig);
    lua_setfield(L, -2, "default_config");
    
    lua_pushcfunction(L, lua_archInfo);
    lua_setfield(L, -2, "info");
    
    lua_pushcfunction(L, lua_archDtypes);
    lua_setfield(L, -2, "dtypes");
    
    lua_setfield(L, -2, "Architectures");  // Mimir.Architectures
    
    // ========== Sous-table "Mimir.Layers" ==========
    lua_newtable(L);

    lua_pushcfunction(L, lua_layersAvailable);
    lua_setfield(L, -2, "available");

    lua_pushcfunction(L, lua_layersByType);
    lua_setfield(L, -2, "by_type");

    auto registerLayerType = [this](const std::string& field_name, const std::string& layer_type) {
        lua_pushstring(L, layer_type.c_str());
        lua_pushcclosure(L, lua_layersByType, 1);
        lua_setfield(L, -2, field_name.c_str());
    };

    for (const auto& layer_type : LayerRegistry::get_all_supported_types()) {
        registerLayerType(layer_type, layer_type);
        std::string lowercase = layer_type;
        std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        registerLayerType(lowercase, layer_type);
    }
    
    lua_pushstring(L, "Conv2d");
    lua_pushcclosure(L, lua_layersByType, 1);
    lua_setfield(L, -2, "conv2d");
    
    lua_pushstring(L, "Linear");
    lua_pushcclosure(L, lua_layersByType, 1);
    lua_setfield(L, -2, "linear");
    
    lua_pushstring(L, "MaxPool2d");
    lua_pushcclosure(L, lua_layersByType, 1);
    lua_setfield(L, -2, "maxpool2d");
    
    lua_pushstring(L, "AvgPool2d");
    lua_pushcclosure(L, lua_layersByType, 1);
    lua_setfield(L, -2, "avgpool2d");
    
    lua_pushcfunction(L, lua_computeActivation);
    lua_setfield(L, -2, "activation");
    
    lua_pushcfunction(L, lua_computeBatchNorm);
    lua_setfield(L, -2, "batchnorm");
    
    lua_pushstring(L, "LayerNorm");
    lua_pushcclosure(L, lua_layersByType, 1);
    lua_setfield(L, -2, "layernorm");
    
    lua_pushcfunction(L, lua_computeAttention);
    lua_setfield(L, -2, "attention");
    
    lua_setfield(L, -2, "Layers");  // Mimir.Layers
    
    // ========== Sous-table "Mimir.Checkpoint" (legacy, deprecated) ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_saveCheckpoint);
    lua_setfield(L, -2, "save");
    
    lua_pushcfunction(L, lua_loadCheckpoint);
    lua_setfield(L, -2, "load");
    
    lua_setfield(L, -2, "Checkpoint");  // Mimir.Checkpoint (legacy)
    
    // ========== Sous-table "Mimir.Tokenizer" ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_createTokenizer);
    lua_setfield(L, -2, "create");
    
    lua_pushcfunction(L, lua_tokenize);
    lua_setfield(L, -2, "tokenize");
    
    lua_pushcfunction(L, lua_detokenize);
    lua_setfield(L, -2, "detokenize");
    
    lua_pushcfunction(L, lua_getVocabSize);
    lua_setfield(L, -2, "vocab_size");

    lua_pushcfunction(L, lua_getMaxVocab);
    lua_setfield(L, -2, "get_max_vocab");

    lua_pushcfunction(L, lua_setMaxVocab);
    lua_setfield(L, -2, "set_max_vocab");
    
    lua_pushcfunction(L, lua_saveTokenizer);
    lua_setfield(L, -2, "save");
    
    lua_pushcfunction(L, lua_loadTokenizer);
    lua_setfield(L, -2, "load");
    
    // Méthodes de manipulation du vocabulaire
    lua_pushcfunction(L, lua_addToken);
    lua_setfield(L, -2, "add_token");
    
    lua_pushcfunction(L, lua_ensureVocabFromText);
    lua_setfield(L, -2, "ensure_vocab_from_text");
    
    lua_pushcfunction(L, lua_tokenizeEnsure);
    lua_setfield(L, -2, "tokenize_ensure");
    
    // Méthodes d'accès aux tokens spéciaux
    lua_pushcfunction(L, lua_getPadId);
    lua_setfield(L, -2, "pad_id");
    
    lua_pushcfunction(L, lua_getUnkId);
    lua_setfield(L, -2, "unk_id");
    
    lua_pushcfunction(L, lua_getSeqId);
    lua_setfield(L, -2, "seq_id");
    
    lua_pushcfunction(L, lua_getModId);
    lua_setfield(L, -2, "mod_id");
    
    lua_pushcfunction(L, lua_getMagId);
    lua_setfield(L, -2, "mag_id");
    
    lua_pushcfunction(L, lua_getTokenById);
    lua_setfield(L, -2, "get_token_by_id");
    
    // Méthodes BPE
    lua_pushcfunction(L, lua_learnBPEFromCorpus);
    lua_setfield(L, -2, "learn_bpe");
    
    lua_pushcfunction(L, lua_tokenizeBPE);
    lua_setfield(L, -2, "tokenize_bpe");
    
    lua_pushcfunction(L, lua_setMaxSequenceLength);
    lua_setfield(L, -2, "set_max_length");
    
    lua_pushcfunction(L, lua_padSequence);
    lua_setfield(L, -2, "pad_sequence");
    
    lua_pushcfunction(L, lua_batchTokenize);
    lua_setfield(L, -2, "batch_tokenize");
    
    // Statistiques et analyse
    lua_pushcfunction(L, lua_printVocabStats);
    lua_setfield(L, -2, "print_stats");
    
    lua_pushcfunction(L, lua_getTokenFrequencies);
    lua_setfield(L, -2, "get_frequencies");
    
    lua_pushcfunction(L, lua_analyzeText);
    lua_setfield(L, -2, "analyze_text");
    
    lua_pushcfunction(L, lua_extractKeywords);
    lua_setfield(L, -2, "extract_keywords");
    
    lua_setfield(L, -2, "Tokenizer");  // Mimir.Tokenizer
    
    // ========== Sous-table "Mimir.Dataset" ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_loadDataset);
    lua_setfield(L, -2, "load");
    
    lua_pushcfunction(L, lua_getDataset);
    lua_setfield(L, -2, "get");

    lua_pushcfunction(L, lua_releaseDatasetItem);
    lua_setfield(L, -2, "release");
    
    lua_pushcfunction(L, lua_prepareSequences);
    lua_setfield(L, -2, "prepare_sequences");
    
    lua_setfield(L, -2, "Dataset");  // Mimir.Dataset

    // ========== Sous-table "Mimir.Database" ==========
    // Builder: Mimir.Database.load(...).cache(...)
    lua_newtable(L);

    lua_pushcfunction(L, lua_databaseLoad);
    lua_setfield(L, -2, "load");

    lua_setfield(L, -2, "Database");  // Mimir.Database

    // ========== Sous-table "Mimir.IO" ==========
    lua_newtable(L);

    lua_pushcfunction(L, lua_readImageRGBU8);
    lua_setfield(L, -2, "read_image_rgb_u8");
    lua_pushcfunction(L, lua_readImageRGBU8);
    lua_setfield(L, -2, "readImageRGBU8");  // alias
    lua_pushcfunction(L, lua_setStdoutLogSuppressed);
    lua_setfield(L, -2, "suppress_stdout_logs");
    lua_pushcfunction(L, lua_setStdoutLogSuppressed);
    lua_setfield(L, -2, "suppressStdoutLogs");  // alias

    lua_setfield(L, -2, "IO");  // Mimir.IO
    
    // ========== Sous-table "Mimir.Memory" ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_memoryConfig);
    lua_setfield(L, -2, "config");
    
    lua_pushcfunction(L, lua_memoryGetStats);
    lua_setfield(L, -2, "get_stats");
    lua_pushcfunction(L, lua_memoryGetStats);
    lua_setfield(L, -2, "getStats");  // camelCase alias
    
    lua_pushcfunction(L, lua_memoryPrintStats);
    lua_setfield(L, -2, "print_stats");
    lua_pushcfunction(L, lua_memoryPrintStats);
    lua_setfield(L, -2, "printStats");  // camelCase alias
    
    lua_pushcfunction(L, lua_memoryClear);
    lua_setfield(L, -2, "clear");
    
    lua_pushcfunction(L, lua_memoryGetUsage);
    lua_setfield(L, -2, "get_usage");
    lua_pushcfunction(L, lua_memoryGetUsage);
    lua_setfield(L, -2, "getUsage");  // camelCase alias
    
    lua_pushcfunction(L, lua_memorySetLimit);
    lua_setfield(L, -2, "set_limit");
    lua_pushcfunction(L, lua_memorySetLimit);
    lua_setfield(L, -2, "setLimit");  // camelCase alias
    
    lua_setfield(L, -2, "Memory");  // Mimir.Memory
    
    // ========== Sous-table "Mimir.Guard" (strict memory enforcement) ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_guardSetLimit);
    lua_setfield(L, -2, "set_limit");
    lua_pushcfunction(L, lua_guardSetLimit);
    lua_setfield(L, -2, "setLimit");  // camelCase alias
    
    lua_pushcfunction(L, lua_guardGetStats);
    lua_setfield(L, -2, "get_stats");
    lua_pushcfunction(L, lua_guardGetStats);
    lua_setfield(L, -2, "getStats");  // camelCase alias
    
    lua_pushcfunction(L, lua_guardPrintStats);
    lua_setfield(L, -2, "print_stats");
    lua_pushcfunction(L, lua_guardPrintStats);
    lua_setfield(L, -2, "printStats");  // camelCase alias
    
    lua_pushcfunction(L, lua_guardReset);
    lua_setfield(L, -2, "reset");
    
    lua_setfield(L, -2, "Guard");  // Mimir.Guard
    
    // ========== Sous-table "Mimir.MemoryGuard" (nom moderne pour guard) ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_guardSetLimit);
    lua_setfield(L, -2, "setLimit");
    
    lua_pushcfunction(L, lua_memoryguardGetCurrentUsage);
    lua_setfield(L, -2, "getCurrentUsage");
    
    lua_pushcfunction(L, lua_memoryguardGetPeakUsage);
    lua_setfield(L, -2, "getPeakUsage");
    
    lua_pushcfunction(L, lua_memoryguardGetLimit);
    lua_setfield(L, -2, "getLimit");
    
    lua_pushcfunction(L, lua_guardGetStats);
    lua_setfield(L, -2, "getStats");
    
    lua_pushcfunction(L, lua_guardPrintStats);
    lua_setfield(L, -2, "printStats");
    
    lua_pushcfunction(L, lua_guardReset);
    lua_setfield(L, -2, "reset");
    
    lua_setfield(L, -2, "MemoryGuard");  // Mimir.MemoryGuard
    
    // ========== Sous-table "Mimir.Allocator" (dynamic tensor allocator) ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_allocatorConfigure);
    lua_setfield(L, -2, "configure");
    
    lua_pushcfunction(L, lua_allocatorPrintStats);
    lua_setfield(L, -2, "print_stats");
    lua_pushcfunction(L, lua_allocatorPrintStats);
    lua_setfield(L, -2, "printStats");  // camelCase alias
    
    lua_pushcfunction(L, lua_allocatorGetStats);
    lua_setfield(L, -2, "get_stats");
    lua_pushcfunction(L, lua_allocatorGetStats);
    lua_setfield(L, -2, "getStats");  // camelCase alias
    
    lua_setfield(L, -2, "Allocator");  // Mimir.Allocator
    
    // ========== Sous-table "Mimir.Htop" (HtopDisplay monitoring) ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_htopCreate);
    lua_setfield(L, -2, "create");
    
    lua_pushcfunction(L, lua_htopUpdate);
    lua_setfield(L, -2, "update");
    
    lua_pushcfunction(L, lua_htopRender);
    lua_setfield(L, -2, "render");
    
    lua_pushcfunction(L, lua_htopClear);
    lua_setfield(L, -2, "clear");
    
    lua_pushcfunction(L, lua_htopEnable);
    lua_setfield(L, -2, "enable");
    
    lua_setfield(L, -2, "Htop");  // Mimir.Htop
    
    // ========== Sous-table "Mimir.Viz" (backend sélectionné) ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_vizConfigure);
    lua_setfield(L, -2, "configure");
    lua_pushcfunction(L, lua_vizPollEvents);
    lua_setfield(L, -2, "poll_events");
    lua_pushcfunction(L, lua_vizBackend);
    lua_setfield(L, -2, "backend");
    lua_pushcfunction(L, lua_vizCreate);
    lua_setfield(L, -2, "create");
    
    lua_pushcfunction(L, lua_vizInitialize);
    lua_setfield(L, -2, "initialize");
    
    lua_pushcfunction(L, lua_vizIsOpen);
    lua_setfield(L, -2, "is_open");
    
    lua_pushcfunction(L, lua_vizProcessEvents);
    lua_setfield(L, -2, "process_events");
    
    lua_pushcfunction(L, lua_vizUpdate);
    lua_setfield(L, -2, "update");
    
    lua_pushcfunction(L, lua_vizAddImage);
    lua_setfield(L, -2, "add_image");

    lua_pushcfunction(L, lua_vizSetDetections);
    lua_setfield(L, -2, "set_detections");

    lua_pushcfunction(L, lua_vizShowDetections);
    lua_setfield(L, -2, "show_detections");

    lua_pushcfunction(L, lua_vizSetDetectionThreshold);
    lua_setfield(L, -2, "set_detection_threshold");
    
    lua_pushcfunction(L, lua_vizUpdateMetrics);
    lua_setfield(L, -2, "update_metrics");

    lua_pushcfunction(L, lua_vizSetValidation);
    lua_setfield(L, -2, "set_validation");

    lua_pushcfunction(L, lua_vizValidationEnabled);
    lua_setfield(L, -2, "validation_enabled");
    
    lua_pushcfunction(L, lua_vizAddLossPoint);
    lua_setfield(L, -2, "add_loss_point");
    
    lua_pushcfunction(L, lua_vizClear);
    lua_setfield(L, -2, "clear");
    
    lua_pushcfunction(L, lua_vizSetEnabled);
    lua_setfield(L, -2, "set_enabled");
    
    lua_pushcfunction(L, lua_vizSaveLossHistory);
    lua_setfield(L, -2, "save_loss_history");
    
    lua_setfield(L, -2, "Viz");  // Mimir.Viz
    
    // ========== Sous-table "Mimir.Serialization" ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_saveCheckpoint);
    lua_setfield(L, -2, "save");
    
    lua_pushcfunction(L, lua_loadCheckpoint);
    lua_setfield(L, -2, "load");
    
    lua_pushcfunction(L, lua_detectFormat);
    lua_setfield(L, -2, "detect_format");
    
    lua_pushcfunction(L, lua_saveEnhancedDebugJson);
    lua_setfield(L, -2, "save_enhanced_debug");
    
    lua_setfield(L, -2, "Serialization");  // Mimir.Serialization
    
    // Enregistrer la table Mimir comme globale
    lua_setglobal(L, ScriptingContext::kGlobalNamespace);
    
    // ========== Fonctions utilitaires globales ==========
    lua_pushcfunction(L, lua_print);
    lua_setglobal(L, "log");
    
    lua_pushcfunction(L, lua_readJSON);
    lua_setglobal(L, "read_json");
    
    lua_pushcfunction(L, lua_writeJSON);
    lua_setglobal(L, "write_json");
    
    // ========== Aliases globaux pour rétrocompatibilité et facilité d'usage ==========
    // Ces aliases permettent d'utiliser model.*, MemoryGuard.*, etc. directement
    // au lieu de Mimir.Model.*, Mimir.MemoryGuard.*, etc.
    
    lua_getglobal(L, ScriptingContext::kGlobalNamespace);
    
    // model = Mimir.Model
    lua_getfield(L, -1, "Model");
    lua_setglobal(L, ScriptingContext::kAliasModel);
    
    // architectures = Mimir.Architectures
    lua_getfield(L, -1, "Architectures");
    lua_setglobal(L, ScriptingContext::kAliasArchitectures);
    
    // tokenizer = Mimir.Tokenizer
    lua_getfield(L, -1, "Tokenizer");
    lua_setglobal(L, ScriptingContext::kAliasTokenizer);
    
    // dataset = Mimir.Dataset
    lua_getfield(L, -1, "Dataset");
    lua_setglobal(L, ScriptingContext::kAliasDataset);
    
    // Memory = Mimir.Memory
    lua_getfield(L, -1, "Memory");
    lua_setglobal(L, ScriptingContext::kAliasMemory);
    
    // MemoryGuard = Mimir.MemoryGuard (priorité)
    lua_getfield(L, -1, "MemoryGuard");
    lua_setglobal(L, ScriptingContext::kAliasMemoryGuard);
    
    // Allocator = Mimir.Allocator
    lua_getfield(L, -1, "Allocator");
    lua_setglobal(L, ScriptingContext::kAliasAllocator);
    
    // htop = Mimir.Htop
    lua_getfield(L, -1, "Htop");
    lua_setglobal(L, ScriptingContext::kAliasHtop);
    
    // viz = Mimir.Viz
    lua_getfield(L, -1, "Viz");
    lua_setglobal(L, ScriptingContext::kAliasViz);
    
    lua_pop(L, 1);  // Pop Mimir table
}

int LuaScripting::lua_benchmarkNow(lua_State* L) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const double seconds = std::chrono::duration<double>(now).count();
    lua_pushnumber(L, seconds);
    return 1;
}

// ============================================================================
// Implémentation des fonctions Lua
// ============================================================================



