#include "LuaScripting.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"
#include "Serialization/DebugJsonDump.hpp"
#include "AdvancedRAMManager.hpp"
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#include "AsyncMonitor.hpp"
#include "Models/Diffusion/PonyXLDDPMModel.hpp"
#include "Helpers.hpp"
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

// ============================================================================
// Constructeur / Destructeur
// ============================================================================

LuaScripting::LuaScripting() {
    L = luaL_newstate();
    luaL_openlibs(L);  // Charger les bibliothèques standard Lua
    registerAPI();
}

void LuaScripting::setArgs(const std::string& script_path, const std::vector<std::string>& script_args) {
    // Reproduit le comportement de l’interpréteur Lua en exposant une table globale `arg`.
    // Convention: arg[0] = chemin du script, arg[1..n] = arguments.
    lua_newtable(L);

    lua_pushstring(L, script_path.c_str());
    lua_rawseti(L, -2, 0);

    for (size_t i = 0; i < script_args.size(); ++i) {
        lua_pushstring(L, script_args[i].c_str());
        lua_rawseti(L, -2, static_cast<lua_Integer>(i + 1));
    }

    lua_setglobal(L, "arg");
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

    lua_pushcfunction(L, lua_createModelFromConfig);
    lua_setfield(L, -2, "create_from_config");
    
    lua_pushcfunction(L, lua_buildModel);
    lua_setfield(L, -2, "build");
    
    lua_pushcfunction(L, lua_trainModel);
    lua_setfield(L, -2, "train");
    
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
    
    // Hardware
    lua_pushcfunction(L, lua_setHardwareAccel);
    lua_setfield(L, -2, "set_hardware");
    
    lua_pushcfunction(L, lua_getHardwareCaps);
    lua_setfield(L, -2, "hardware_caps");

    // Helpers spécifiques (PonyXL / Diffusion)
    lua_pushcfunction(L, lua_ponyxlDdpmTrainStep);
    lua_setfield(L, -2, "ponyxl_ddpm_train_step");

    lua_pushcfunction(L, lua_ponyxlDdpmValidateStep);
    lua_setfield(L, -2, "ponyxl_ddpm_validate_step");

    lua_pushcfunction(L, lua_ponyxlDdpmVizReconstructStep);
    lua_setfield(L, -2, "ponyxl_ddpm_viz_reconstruct_step");

    lua_pushcfunction(L, lua_ponyxlDdpmSetVaeScale);
    lua_setfield(L, -2, "ponyxl_ddpm_set_vae_scale");

    lua_pushcfunction(L, lua_ponyxlDdpmGetVaeScale);
    lua_setfield(L, -2, "ponyxl_ddpm_get_vae_scale");

    lua_pushcfunction(L, lua_ponyxlDdpmVaeMuMoments);
    lua_setfield(L, -2, "ponyxl_ddpm_vae_mu_moments");
    
    lua_setfield(L, -2, "Model");  // Mimir.Model
    
    // ========== Sous-table "Mimir.Architectures" ==========
    lua_newtable(L);

    // Registry helpers
    lua_pushcfunction(L, lua_archAvailable);
    lua_setfield(L, -2, "available");

    lua_pushcfunction(L, lua_archDefaultConfig);
    lua_setfield(L, -2, "default_config");
    
    lua_setfield(L, -2, "Architectures");  // Mimir.Architectures
    
    // ========== Sous-table "Mimir.Layers" ==========
    lua_newtable(L);
    
    lua_pushcfunction(L, lua_computeConv2D);
    lua_setfield(L, -2, "conv2d");
    
    lua_pushcfunction(L, lua_computeLinear);
    lua_setfield(L, -2, "linear");
    
    lua_pushcfunction(L, lua_computeMaxPool2D);
    lua_setfield(L, -2, "maxpool2d");
    
    lua_pushcfunction(L, lua_computeAvgPool2D);
    lua_setfield(L, -2, "avgpool2d");
    
    lua_pushcfunction(L, lua_computeActivation);
    lua_setfield(L, -2, "activation");
    
    lua_pushcfunction(L, lua_computeBatchNorm);
    lua_setfield(L, -2, "batchnorm");
    
    lua_pushcfunction(L, lua_computeLayerNorm);
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
    
    // ========== Sous-table "Mimir.Viz" (Visualizer SFML) ==========
    lua_newtable(L);
    
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
    
    lua_pushcfunction(L, lua_vizUpdateMetrics);
    lua_setfield(L, -2, "update_metrics");

    lua_pushcfunction(L, lua_vizSetValidation);
    lua_setfield(L, -2, "set_validation");
    
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
    lua_setglobal(L, "Mimir");
    
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
    
    lua_getglobal(L, "Mimir");
    
    // model = Mimir.Model
    lua_getfield(L, -1, "Model");
    lua_setglobal(L, "model");
    
    // architectures = Mimir.Architectures
    lua_getfield(L, -1, "Architectures");
    lua_setglobal(L, "architectures");
    
    // tokenizer = Mimir.Tokenizer
    lua_getfield(L, -1, "Tokenizer");
    lua_setglobal(L, "tokenizer");
    
    // dataset = Mimir.Dataset
    lua_getfield(L, -1, "Dataset");
    lua_setglobal(L, "dataset");
    
    // Memory = Mimir.Memory
    lua_getfield(L, -1, "Memory");
    lua_setglobal(L, "Memory");
    
    // MemoryGuard = Mimir.MemoryGuard (priorité)
    lua_getfield(L, -1, "MemoryGuard");
    lua_setglobal(L, "MemoryGuard");
    
    // Allocator = Mimir.Allocator
    lua_getfield(L, -1, "Allocator");
    lua_setglobal(L, "Allocator");
    
    // htop = Mimir.Htop
    lua_getfield(L, -1, "Htop");
    lua_setglobal(L, "htop");
    
    // viz = Mimir.Viz
    lua_getfield(L, -1, "Viz");
    lua_setglobal(L, "viz");
    
    lua_pop(L, 1);  // Pop Mimir table
}

// ============================================================================
// Implémentation des fonctions Lua
// ============================================================================

namespace {

// Le registre d'architectures construit `Model::modelConfig` à partir d'un sous-ensemble
// des champs de config (ceux présents dans les structs Config). Or les boucles d'entraînement
// (trainStepVAE/optimizerStep/etc.) lisent certains hyperparamètres depuis `model.modelConfig`.
// On propage donc les clés pertinentes depuis la config Lua (`cfg`) vers `model.modelConfig`.
//
// Règles:
// - Par défaut: n'écrase pas une clé déjà définie par l'architecture.
// - Certaines clés sont explicitement autorisées à override (hyperparams de loss/grad/kl),
//   car elles sont attendues comme réglables côté training.
static void mergeLuaConfigIntoModelConfig(Model& model, const json& cfg) {
    if (!cfg.is_object()) return;

    static const std::unordered_set<std::string> kOverrideKeys = {
        // VAE losses/knobs
        "recon_loss",
        "kl_beta",
        "vae_kl_beta",
        "kl_warmup_steps",
        "marker_wass_scale",
        "marker_temp_scale",
        "marker_scale_max",
        "marker_warmup_steps",
        "logvar_clip_min",
        "logvar_clip_max",

        // Optional additive losses
        "ssim_weight",
        "ssim_mode",
        "ssim_k1",
        "ssim_k2",
        "ssim_L",
        "spectral_weight",
        "spectral_scales",
        "perceptual_weight",
        "perceptual_arch",
        "perceptual_checkpoint",
        "perceptual_base_channels",

        // Recon loss params
        "huber_delta",
        "smoothl1_delta",
        "smoothl1_beta",
        "charbonnier_eps",
        "nll_sigma",
        "gaussian_nll_sigma",

        // Text align
        "align_weight",

        // Optimizer/runtime knobs
        "grad_clip_norm",
        "clip_norm",
        "frozen_layer_prefixes",
    };

    for (auto it = cfg.begin(); it != cfg.end(); ++it) {
        const std::string key = it.key();
        const json& value = it.value();

        if (kOverrideKeys.count(key) > 0) {
            model.modelConfig[key] = value;
            continue;
        }

        // Ne pas écraser les clés déjà définies par l'architecture.
        if (!model.modelConfig.contains(key)) {
            model.modelConfig[key] = value;
        }
    }
}

} // namespace

int LuaScripting::lua_createModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Argument: type de modèle (string)
    const char* model_type = luaL_checkstring(L, 1);
    
    // Argument optionnel: config (table). Si absent: defaultConfig(model_type)
    json config;
    if (lua_istable(L, 2)) {
        config = luaTableToJson(L, 2);
    }
    
    try {
        const std::string name(model_type);
        if (!config.is_object() || config.empty()) {
            config = ModelArchitectures::defaultConfig(name);
        }

        // Création + construction via registre (le réseau est défini dans la classe du modèle).
        ctx.currentModel = ModelArchitectures::create(name, config);

        // Important: faire remonter les hyperparams de training (KL, recon_loss, clip, etc.)
        // dans `model.modelConfig` pour que Model::trainStepVAE/optimizerStep les voient.
        if (ctx.currentModel) {
            mergeLuaConfigIntoModelConfig(*ctx.currentModel, config);
        }

        // Propager les assets du contexte (tokenizer/encoder) au modèle.
        if (ctx.currentTokenizer) {
            ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
        }
        if (ctx.currentEncoder) {
            ctx.currentModel->setEncoder(*ctx.currentEncoder);
        }

        ctx.modelType = name;
        ctx.modelConfig = config;

        // Si la viz est active, activer automatiquement les viz taps sur le modèle.
        // (sinon, les "blocks" ne seront jamais produits et la Viz semblera vide.)
        if (ctx.asyncMonitor && ctx.asyncMonitor->getViz() != nullptr && ctx.currentModel) {
            ctx.currentModel->setVizTapsEnabled(true);
            try {
                int max_frames = 12;
                int max_side = 64;
                if (ctx.modelConfig.contains("viz_taps_max_frames")) max_frames = ctx.modelConfig["viz_taps_max_frames"].get<int>();
                if (ctx.modelConfig.contains("viz_taps_max_side")) max_side = ctx.modelConfig["viz_taps_max_side"].get<int>();
                // Safety: too-small limits degenerate to 1x1 previews (often perceived as white squares)
                // or to a single constantly-replaced frame.
                ctx.currentModel->setVizTapsLimits(std::max(16, max_frames), std::max(16, max_side));
            } catch (...) {
            }
        }

        ctx.addLog("Modèle créé via registre: " + name);
        lua_pushboolean(L, true);
    } catch (const std::exception& e) {
        ctx.addLog("Erreur création modèle: " + std::string(e.what()));
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

int LuaScripting::lua_createModelFromConfig(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    // Argument: config complète (table)
    luaL_checktype(L, 1, LUA_TTABLE);
    json full = luaTableToJson(L, 1);

    try {
        std::string arch;
        json cfg;
        ctx.currentModel = ModelArchitectures::createFromConfig(full, &cfg, &arch);

        // Même principe que lua_createModel: injecter les clés de config (training) dans
        // `model.modelConfig` en respectant les overrides autorisés.
        if (ctx.currentModel) {
            mergeLuaConfigIntoModelConfig(*ctx.currentModel, cfg);
        }

        // Propager les assets déjà présents.
        if (ctx.currentTokenizer) {
            ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
        }
        if (ctx.currentEncoder) {
            ctx.currentModel->setEncoder(*ctx.currentEncoder);
        }

        ctx.currentConfig = full;
        ctx.modelType = arch;
        ctx.modelConfig = cfg;

        ctx.addLog("Modèle créé via registre depuis config: " + arch);

        lua_pushboolean(L, true);
        lua_pushstring(L, arch.c_str());
        return 2;
    } catch (const std::exception& e) {
        ctx.addLog("Erreur création modèle depuis config: " + std::string(e.what()));
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_buildModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    try {
        // Framework moderne: Model.create(name, cfg) construit déjà le réseau via le registre.
        // On conserve Model.build() pour compat scripts, mais sans re-création (sinon on perd
        // les poids chargés, l'état optimiseur, etc.).
        if (ctx.currentTokenizer) {
            ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
        }
        if (ctx.currentEncoder) {
            ctx.currentModel->setEncoder(*ctx.currentEncoder);
        }

        const size_t params = ctx.currentModel->totalParamCount();
        ctx.addLog("Model.build: no-op (moderne). Params=" + std::to_string(params));

        lua_pushboolean(L, true);
        lua_pushinteger(L, params);
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_trainModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Arguments: epochs (number), learning_rate (number)
    int epochs = luaL_checkinteger(L, 1);
    double lr = luaL_checknumber(L, 2);
    
    ctx.addLog("Entraînement: " + std::to_string(epochs) + " epochs, LR=" + std::to_string(lr));

    try {
        // Si la viz est active, activer les "viz taps" côté modèle et préparer
        // un pont vers AsyncMonitor pour afficher les blocs/layers.
        const bool viz_active = (ctx.asyncMonitor && ctx.asyncMonitor->getViz() != nullptr);
        if (viz_active && ctx.currentModel) {
            ctx.currentModel->setVizTapsEnabled(true);
            try {
                int max_frames = 12;
                int max_side = 64;
                if (ctx.modelConfig.contains("viz_taps_max_frames")) max_frames = ctx.modelConfig["viz_taps_max_frames"].get<int>();
                if (ctx.modelConfig.contains("viz_taps_max_side")) max_side = ctx.modelConfig["viz_taps_max_side"].get<int>();
                ctx.currentModel->setVizTapsLimits(max_frames, max_side);
            } catch (...) {
            }
        }

        // Instancier l'Optimizer à partir de la configuration
        Optimizer opt;
        opt.initial_lr = static_cast<float>(lr);
        
        // Type d'optimizer depuis la config (défaut: ADAMW)
        if (ctx.modelConfig.contains("optimizer")) {
            std::string opt_type = ctx.modelConfig["optimizer"];
            if (opt_type == "sgd" || opt_type == "SGD") {
                opt.type = OptimizerType::SGD;
            } else if (opt_type == "adam" || opt_type == "ADAM") {
                opt.type = OptimizerType::ADAM;
            } else if (opt_type == "adamw" || opt_type == "ADAMW") {
                opt.type = OptimizerType::ADAMW;
            }
        } else {
            opt.type = OptimizerType::ADAMW;  // Défaut
        }
        
        // Paramètres de l'optimizer depuis la config
        if (ctx.modelConfig.contains("beta1")) {
            opt.beta1 = ctx.modelConfig["beta1"];
        }
        if (ctx.modelConfig.contains("beta2")) {
            opt.beta2 = ctx.modelConfig["beta2"];
        }
        if (ctx.modelConfig.contains("epsilon")) {
            opt.eps = ctx.modelConfig["epsilon"];
        }
        if (ctx.modelConfig.contains("weight_decay")) {
            opt.weight_decay = ctx.modelConfig["weight_decay"];
        }
        
        // Paramètres de LR decay depuis la config
        if (ctx.modelConfig.contains("min_lr")) {
            opt.min_lr = ctx.modelConfig["min_lr"];
        }
        if (ctx.modelConfig.contains("decay_rate")) {
            opt.decay_rate = ctx.modelConfig["decay_rate"];
        }
        if (ctx.modelConfig.contains("decay_steps")) {
            opt.decay_steps = ctx.modelConfig["decay_steps"];
        }
        if (ctx.modelConfig.contains("warmup_steps")) {
            opt.warmup_steps = ctx.modelConfig["warmup_steps"];
        }

        // Strategy (optionnel): "none" | "cosine" | "step" | "exponential" | "linear"
        if (ctx.modelConfig.contains("decay_strategy")) {
            std::string s = ctx.modelConfig["decay_strategy"].get<std::string>();
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            if (s == "none") opt.decay_strategy = LRDecayStrategy::NONE;
            else if (s == "cosine") opt.decay_strategy = LRDecayStrategy::COSINE;
            else if (s == "step") opt.decay_strategy = LRDecayStrategy::STEP;
            else if (s == "exponential") opt.decay_strategy = LRDecayStrategy::EXPONENTIAL;
            else if (s == "linear") opt.decay_strategy = LRDecayStrategy::LINEAR;
        }

        // Reprise éventuelle de l'état optimiseur (checkpoint)
        const Optimizer* saved_opt = ctx.currentModel->getSerializedOptimizer();
        const bool resume_optimizer_loaded = (saved_opt != nullptr);
        if (saved_opt) {
            opt = *saved_opt;
            // S'assurer que lr reflète l'argument (opt.initial_lr sert de base au scheduler)
            opt.initial_lr = static_cast<float>(lr);
        }
        
        ctx.addLog("Optimizer configuré: type=" + std::to_string(static_cast<int>(opt.type)) + 
                   ", beta1=" + std::to_string(opt.beta1) + 
                   ", beta2=" + std::to_string(opt.beta2) + 
                   ", weight_decay=" + std::to_string(opt.weight_decay));

        // -----------------------------------------------------------------
        // Entraînement moderne basé dataset, par type de modèle
        // -----------------------------------------------------------------
        std::string model_type;
        try {
            if (ctx.currentModel->modelConfig.contains("type")) {
                model_type = ctx.currentModel->modelConfig["type"].get<std::string>();
            }
        } catch (...) {
        }
        if (model_type.empty()) model_type = ctx.modelType;

        // Dataset requis pour la majorité des trains.
        if (ctx.currentDataset.empty()) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Aucun dataset chargé. Utilisez Dataset.load() d'abord.");
            return 2;
        }

        // Common knobs
        int max_items = 0;
        int log_every = 10;
        int seed = 1337;
        std::string checkpoint_dir;
        int autosave_every_epochs = 0;
        try {
            if (ctx.modelConfig.contains("max_items")) max_items = ctx.modelConfig["max_items"].get<int>();
            if (ctx.modelConfig.contains("log_every")) log_every = ctx.modelConfig["log_every"].get<int>();
            if (ctx.modelConfig.contains("seed")) seed = ctx.modelConfig["seed"].get<int>();
            if (ctx.modelConfig.contains("checkpoint_dir")) checkpoint_dir = ctx.modelConfig["checkpoint_dir"].get<std::string>();
            if (ctx.modelConfig.contains("autosave_every_epochs")) autosave_every_epochs = ctx.modelConfig["autosave_every_epochs"].get<int>();
            else if (ctx.modelConfig.contains("autosave_every_epoch")) autosave_every_epochs = ctx.modelConfig["autosave_every_epoch"].get<int>();
        } catch (...) {
        }
        max_items = std::max(0, max_items);
        log_every = std::max(1, log_every);
        autosave_every_epochs = std::max(0, autosave_every_epochs);

        // -----------------------------------------------------------------
        // Resume epoch offset (UX): si on reprend depuis checkpoint_dir/epoch_XXXX,
        // afficher/nommer les epochs à partir de l'epoch précédente.
        // Exemple: reprise à 3 => UI affiche 3 (pas 1) et autosave écrit epoch_0003+.
        // Active uniquement si un optimizer a été chargé via Serialization.load().
        // -----------------------------------------------------------------
        int epoch_offset = 0;          // Ajouté à epoch_1based (boucle) pour affichage + checkpoints
        if (resume_optimizer_loaded && !checkpoint_dir.empty()) {
            try {
                namespace fs = std::filesystem;
                const fs::path base(checkpoint_dir);
                int best = -1;
                if (fs::exists(base) && fs::is_directory(base)) {
                    for (auto& p : fs::directory_iterator(base)) {
                        if (!p.is_directory()) continue;
                        const std::string name = p.path().filename().string();
                        if (name.rfind("epoch_", 0) != 0) continue;
                        try {
                            size_t parsed = 0;
                            const int e = std::stoi(name.substr(std::string("epoch_").size()), &parsed);
                            if (parsed == 0) continue;
                            if (e > best) best = e;
                        } catch (...) {
                        }
                    }
                }
                if (best > 0) {
                    epoch_offset = best - 1;
                }
            } catch (...) {
                // best-effort: pas de resume epoch détectée => offset=0
            }
        }
        const int total_epochs_display = epoch_offset + epochs;

        auto do_checkpoint_save = [&](int epoch_1based, const std::string& suffix, std::string* err_out) -> bool {
            if (!ctx.currentModel) return false;
            if (checkpoint_dir.empty()) return false;

            const int epoch_abs = epoch_offset + epoch_1based;

            using namespace Mimir::Serialization;
            std::ostringstream name;
            name << "epoch_" << std::setw(4) << std::setfill('0') << epoch_abs;
            if (!suffix.empty()) name << suffix;

            const std::filesystem::path out = std::filesystem::path(checkpoint_dir) / name.str();
            std::error_code ec;
            std::filesystem::create_directories(out, ec);

            // Important: synchroniser l'état optimiseur dans le modèle avant la sauvegarde.
            // (train utilise un Optimizer local, la sérialisation lit l'état du modèle)
            if (ctx.currentModel) {
                Optimizer opt_snapshot = opt;
                ctx.currentModel->setSerializedOptimizer(std::move(opt_snapshot));
            }

            SaveOptions so;
            so.format = CheckpointFormat::RawFolder;
            so.save_optimizer = true;
            so.save_tokenizer = true;
            so.save_encoder = true;
            so.include_git_info = true;
            so.include_checksums = true;

            std::string error;
            const bool ok = save_checkpoint(*ctx.currentModel, out.string(), so, &error);
            if (!ok) {
                if (err_out) *err_out = error;
                return false;
            }
            ctx.addLog("✓ Autosave checkpoint: " + out.string());
            return true;
        };

        std::mt19937 rng((uint32_t)seed);

        std::string recon_loss_type;
        try {
            if (ctx.modelConfig.contains("recon_loss")) recon_loss_type = ctx.modelConfig["recon_loss"].get<std::string>();
        } catch (...) {
        }

        int global_step = 0;

        // -----------------------------------------------------------------
        // Live tuning via Viz (thread UI -> training thread)
        // -----------------------------------------------------------------
        uint64_t last_live_ver = 0;
        bool live_override_active = false;

        // Baselines (pour retour au comportement natif)
        const float baseline_lr = static_cast<float>(lr);
        const int baseline_lr_warmup_steps = std::max(0, opt.warmup_steps);
        float baseline_kl_beta = 0.0f;
        int baseline_kl_warmup_steps = 0;
        try {
            if (ctx.modelConfig.contains("kl_beta")) baseline_kl_beta = ctx.modelConfig["kl_beta"].get<float>();
            else if (ctx.modelConfig.contains("vae_kl_beta")) baseline_kl_beta = ctx.modelConfig["vae_kl_beta"].get<float>();
            if (ctx.modelConfig.contains("kl_warmup_steps")) baseline_kl_warmup_steps = ctx.modelConfig["kl_warmup_steps"].get<int>();
        } catch (...) {
        }
        baseline_kl_beta = std::max(0.0f, baseline_kl_beta);
        baseline_kl_warmup_steps = std::max(0, baseline_kl_warmup_steps);

        float baseline_pony_kl_beta = 0.0f;
        int baseline_pony_kl_warmup_steps = 0;
        if (auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get())) {
            const auto& pcfg = pony->getConfig();
            baseline_pony_kl_beta = std::max(0.0f, pcfg.kl_beta);
            baseline_pony_kl_warmup_steps = std::max(0, pcfg.kl_warmup_steps);
        }

        auto poll_viz_live_params = [&]() {
            if (!ctx.asyncMonitor) return;
            auto viz = ctx.asyncMonitor->getViz();
            if (!viz) return;

            const uint64_t ver = viz->liveTrainParamsVersion();
            if (ver == 0 || ver == last_live_ver) return;
            last_live_ver = ver;

            const auto p = viz->liveTrainParamsSnapshot();
            if (p.version == 0) return;

            // Si on repasse en mode natif, restaurer la baseline une fois.
            if (!p.overrides_enabled) {
                if (live_override_active) {
                    live_override_active = false;
                    opt.initial_lr = std::max(1e-12f, baseline_lr);
                    opt.warmup_steps = baseline_lr_warmup_steps;

                    ctx.modelConfig["kl_beta"] = baseline_kl_beta;
                    ctx.modelConfig["kl_warmup_steps"] = baseline_kl_warmup_steps;
                    if (ctx.currentModel) {
                        ctx.currentModel->modelConfig["kl_beta"] = baseline_kl_beta;
                        ctx.currentModel->modelConfig["kl_warmup_steps"] = baseline_kl_warmup_steps;
                    }

                    if (auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get())) {
                        pony->setLiveKL(baseline_pony_kl_beta, baseline_pony_kl_warmup_steps);
                    }
                }
                return;
            }

            live_override_active = true;

            if (std::isfinite(p.lr) && p.lr > 0.0f) {
                opt.initial_lr = std::max(1e-12f, p.lr);
            }
            opt.warmup_steps = std::max(0, p.lr_warmup_steps);

            const float kl_beta = (p.kl_enabled ? std::max(0.0f, p.kl_beta) : 0.0f);
            const int kl_warmup_steps = std::max(0, p.kl_warmup_steps);

            // Propager vers la config runtime (consommée par Model::trainStepVAE, etc.)
            ctx.modelConfig["kl_beta"] = kl_beta;
            ctx.modelConfig["kl_warmup_steps"] = kl_warmup_steps;
            if (ctx.currentModel) {
                ctx.currentModel->modelConfig["kl_beta"] = kl_beta;
                ctx.currentModel->modelConfig["kl_warmup_steps"] = kl_warmup_steps;
            }

            // PonyXLDDPM: KL est lu depuis cfg_ (pas via modelConfig) => setter dédié.
            if (auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get())) {
                pony->setLiveKL(kl_beta, kl_warmup_steps);
            }
        };

        auto step_learning_rate = [&]() -> float {
            // Avant toute interaction UI, conserver le comportement historique (arg `lr`).
            return live_override_active ? opt.getCurrentLR() : static_cast<float>(lr);
        };

        // Perf stats pour la viz: time/mem/bps (best-effort).
        // - time: ms entre 2 updates successifs (approx temps/batch)
        // - bps: batches/sec (approx)
        // - mem: MemoryGuard current bytes en MB
        std::chrono::steady_clock::time_point last_metrics_ts;
        bool has_last_metrics_ts = false;
        auto apply_perf_stats = [&](AsyncMonitor::Metrics& m) {
            const auto now = std::chrono::steady_clock::now();
            if (has_last_metrics_ts) {
                const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_metrics_ts).count();
                if (dt_ms > 0 && dt_ms < 60LL * 60LL * 1000LL) {
                    m.batch_time_ms = (int)dt_ms;
                    m.bps = 1000.0f / (float)m.batch_time_ms;
                }
            }
            last_metrics_ts = now;
            has_last_metrics_ts = true;

            auto& guard = MemoryGuard::instance();
            m.memory_mb = guard.getCurrentBytes() / 1024 / 1024;
        };

        auto log_step = [&](int global_step, const Model::VAEStepStats& st, const char* prefix) {
            if ((global_step % log_every) != 0) return;
            ctx.addLog(std::string(prefix) +
                       " step=" + std::to_string(global_step) +
                       " loss=" + std::to_string(st.loss) +
                       " mse=" + std::to_string(st.mse) +
                       " kl=" + std::to_string(st.kl) +
                       " beta_eff=" + std::to_string(st.kl_beta_effective) +
                       " grad_norm=" + std::to_string(st.grad_norm));
        };

        auto monitor_step = [&](int epoch_1based, int batch_1based, int total_batches, const Model::VAEStepStats& st) {
            if (!ctx.asyncMonitor) return;

            AsyncMonitor::Metrics m;
            m.epoch = epoch_offset + epoch_1based;
            m.total_epochs = total_epochs_display;
            m.batch = batch_1based;
            m.total_batches = total_batches;
            m.loss = st.loss;
            m.avg_loss = st.loss;
            m.lr = opt.getCurrentLR();
            m.mse = st.mse;
            m.kl = st.kl;
            m.kl_beta_effective = st.kl_beta_effective;
            m.wass = st.wass;
            m.temp = st.temp;
            m.grad_norm = st.grad_norm;
            m.grad_max = st.grad_max_abs;
            m.params = ctx.currentModel ? ctx.currentModel->totalParamCount() : 0;
            m.recon_loss_type = recon_loss_type;

            // Optimizer state (pour affichage Htop/Viz)
            m.opt_type = (int)opt.type;
            m.opt_step = (int)opt.step;
            m.opt_beta1 = opt.beta1;
            m.opt_beta2 = opt.beta2;
            m.opt_eps = opt.eps;
            m.opt_weight_decay = opt.weight_decay;

            apply_perf_stats(m);

            ctx.asyncMonitor->updateMetrics(m);
        };

        // -----------------------------------------
        // VAEConv (images) optional text
        // -----------------------------------------
        if (model_type == "vae_conv") {
            int image_w = 0, image_h = 0, image_c = 3;
            bool text_cond = false;
            int seq_len = 64;
            int pad_id = 0;
            try {
                if (ctx.modelConfig.contains("image_w")) image_w = ctx.modelConfig["image_w"].get<int>();
                if (ctx.modelConfig.contains("image_h")) image_h = ctx.modelConfig["image_h"].get<int>();
                if (ctx.modelConfig.contains("image_c")) image_c = ctx.modelConfig["image_c"].get<int>();
                if (ctx.modelConfig.contains("text_cond")) text_cond = ctx.modelConfig["text_cond"].get<bool>();
                if (ctx.modelConfig.contains("seq_len")) seq_len = ctx.modelConfig["seq_len"].get<int>();
            } catch (...) {
            }
            if (image_w <= 0 || image_h <= 0) {
                // fallback: dataset config
                try {
                    if (ctx.currentConfig.contains("dataset")) {
                        if (ctx.currentConfig["dataset"].contains("target_w")) image_w = ctx.currentConfig["dataset"]["target_w"].get<int>();
                        if (ctx.currentConfig["dataset"].contains("target_h")) image_h = ctx.currentConfig["dataset"]["target_h"].get<int>();
                    }
                } catch (...) {
                }
            }
            image_w = std::max(1, image_w);
            image_h = std::max(1, image_h);
            image_c = std::max(1, image_c);
            seq_len = std::max(1, seq_len);

            if (text_cond) {
                if (!ctx.currentTokenizer) {
                    lua_pushboolean(L, false);
                    lua_pushstring(L, "text_cond=true mais aucun tokenizer n'est chargé");
                    return 2;
                }
                pad_id = ctx.currentTokenizer->getPadId();
            }

            // Validation config (best-effort, optional)
            int validate_every_steps = 0;
            int validate_items = 0;
            float validate_holdout_frac = 0.0f;
            int validate_holdout_items = 0;
            bool validate_holdout = true;
            try {
                if (ctx.modelConfig.contains("validate_every_steps")) validate_every_steps = std::max(0, ctx.modelConfig["validate_every_steps"].get<int>());
                if (ctx.modelConfig.contains("validate_items")) validate_items = std::max(0, ctx.modelConfig["validate_items"].get<int>());
                if (ctx.modelConfig.contains("validate_holdout_frac")) validate_holdout_frac = ctx.modelConfig["validate_holdout_frac"].get<float>();
                if (ctx.modelConfig.contains("validate_holdout_items")) validate_holdout_items = std::max(0, ctx.modelConfig["validate_holdout_items"].get<int>());
                if (ctx.modelConfig.contains("validate_holdout")) validate_holdout = ctx.modelConfig["validate_holdout"].get<bool>();
            } catch (...) {
            }
            validate_holdout_frac = std::clamp(validate_holdout_frac, 0.0f, 0.95f);

            // -----------------------------------------------------------------
            // Détection automatique "VAE backbone prêt" (pour diffusion text→image)
            // Basée sur la validation: recon (MSE/L1) + KL, et un plateau sur une fenêtre.
            // Écrit un fichier JSON dans checkpoint_dir quand le critère est satisfait.
            // -----------------------------------------------------------------
            bool backbone_ready_enabled = true;
            bool backbone_ready_stop = false;
            int backbone_ready_window = 5;              // nb de validations consécutives
            float backbone_ready_plateau_rel = 0.01f;   // amélioration relative max (1% -> plateau)
            float backbone_ready_plateau_abs = 1e-4f;   // amélioration absolue max
            float backbone_ready_recon_target = 0.02f;  // 0 => ignorer seuil absolu
            float backbone_ready_kl_min = 0.01f;        // évite collapse (KL~0)
            float backbone_ready_kl_max = 5.0f;         // garde-fou
            int backbone_ready_min_steps = 0;           // ex: kl_warmup_steps
            std::string backbone_ready_file = "vae_backbone_ready.json";
            try {
                if (ctx.modelConfig.contains("backbone_ready")) backbone_ready_enabled = ctx.modelConfig["backbone_ready"].get<bool>();
                if (ctx.modelConfig.contains("backbone_ready_enabled")) backbone_ready_enabled = ctx.modelConfig["backbone_ready_enabled"].get<bool>();
                if (ctx.modelConfig.contains("backbone_ready_stop")) backbone_ready_stop = ctx.modelConfig["backbone_ready_stop"].get<bool>();
                if (ctx.modelConfig.contains("backbone_ready_window")) backbone_ready_window = std::max(2, ctx.modelConfig["backbone_ready_window"].get<int>());
                if (ctx.modelConfig.contains("backbone_ready_plateau_rel")) backbone_ready_plateau_rel = std::max(0.0f, ctx.modelConfig["backbone_ready_plateau_rel"].get<float>());
                if (ctx.modelConfig.contains("backbone_ready_plateau_abs")) backbone_ready_plateau_abs = std::max(0.0f, ctx.modelConfig["backbone_ready_plateau_abs"].get<float>());
                if (ctx.modelConfig.contains("backbone_ready_recon_target")) backbone_ready_recon_target = std::max(0.0f, ctx.modelConfig["backbone_ready_recon_target"].get<float>());
                if (ctx.modelConfig.contains("backbone_ready_kl_min")) backbone_ready_kl_min = std::max(0.0f, ctx.modelConfig["backbone_ready_kl_min"].get<float>());
                if (ctx.modelConfig.contains("backbone_ready_kl_max")) backbone_ready_kl_max = std::max(0.0f, ctx.modelConfig["backbone_ready_kl_max"].get<float>());
                if (ctx.modelConfig.contains("backbone_ready_min_steps")) backbone_ready_min_steps = std::max(0, ctx.modelConfig["backbone_ready_min_steps"].get<int>());
                if (ctx.modelConfig.contains("backbone_ready_file")) backbone_ready_file = ctx.modelConfig["backbone_ready_file"].get<std::string>();
            } catch (...) {
            }
            if (backbone_ready_min_steps <= 0) {
                try {
                    if (ctx.modelConfig.contains("kl_warmup_steps")) backbone_ready_min_steps = std::max(0, ctx.modelConfig["kl_warmup_steps"].get<int>());
                } catch (...) {
                }
            }
            if (backbone_ready_file.empty()) backbone_ready_file = "vae_backbone_ready.json";

            // On ne peut pas détecter la "readiness" sans validation.
            if (validate_every_steps <= 0 || validate_items <= 0) {
                backbone_ready_enabled = false;
            }

            bool backbone_ready_written = false;
            std::deque<float> backbone_recon_hist;
            std::deque<float> backbone_kl_hist;
            std::deque<int> backbone_step_hist;

            // Filtrer les items utilisables (au moins une image). Sans ça, `max_items` petit
            // peut tomber sur un item texte-only et faire 0 step.
            std::vector<int> indices;
            indices.reserve(ctx.currentDataset.size());
            for (size_t i = 0; i < ctx.currentDataset.size(); ++i) {
                if (!ctx.currentDataset[i].image_file.empty()) indices.push_back((int)i);
            }
            if (indices.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Dataset: aucun item avec image_file (requis pour vae_conv)");
                return 2;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
            if (max_items > 0 && (int)indices.size() > max_items) indices.resize((size_t)max_items);

            // Split train/val (holdout) only if validation is enabled.
            std::vector<int> train_indices = indices;
            std::vector<int> val_indices;
            if (validate_every_steps > 0 && validate_items > 0 && (int)indices.size() >= 2) {
                if (validate_holdout) {
                    const int n = (int)indices.size();
                    int val_n = 0;
                    if (validate_holdout_items > 0) {
                        val_n = std::clamp(validate_holdout_items, 1, n - 1);
                    } else if (validate_holdout_frac > 0.0f) {
                        val_n = (int)std::floor((double)n * (double)validate_holdout_frac);
                        val_n = std::clamp(val_n, 1, n - 1);
                    }

                    if (val_n > 0) {
                        val_indices.assign(indices.end() - val_n, indices.end());
                        train_indices.assign(indices.begin(), indices.end() - val_n);
                        ctx.addLog("Validation holdout: train=" + std::to_string((int)train_indices.size()) + " val=" + std::to_string((int)val_indices.size()) +
                                   " (every=" + std::to_string(validate_every_steps) + " steps, items=" + std::to_string(validate_items) + ")");
                    } else {
                        // holdout demandé mais taille nulle => fallback: val sur train
                        val_indices = train_indices;
                        ctx.addLog("Validation (no holdout): using train set for val (every=" + std::to_string(validate_every_steps) + " steps, items=" + std::to_string(validate_items) + ")");
                    }
                } else {
                    // No holdout: validate on the training pool
                    val_indices = train_indices;
                    ctx.addLog("Validation (no holdout): using train set for val (every=" + std::to_string(validate_every_steps) + " steps, items=" + std::to_string(validate_items) + ")");
                }
            }

            const int use_n = (int)train_indices.size();
            opt.total_steps = std::max(1, epochs * std::max(1, use_n));

            const size_t expected_u8 = (size_t)image_w * (size_t)image_h * (size_t)image_c;
            std::vector<float> x;
            x.resize(expected_u8);

            std::vector<uint8_t> recon_u8;
            recon_u8.resize(expected_u8);

            // Helpers for validation rendering/metrics.
            auto pack_f32_to_u8 = [&](const std::vector<float>& src, size_t off, std::vector<uint8_t>& dst) {
                const size_t n = std::min(dst.size(), src.size() > off ? (src.size() - off) : (size_t)0);
                for (size_t i = 0; i < n; ++i) {
                    const float v = std::clamp(src[off + i], -1.0f, 1.0f);
                    const float u = (v + 1.0f) * 127.5f;
                    const int q = (int)std::lround((double)u);
                    dst[i] = (uint8_t)std::clamp(q, 0, 255);
                }
            };

            auto compute_val_recon = [&](const std::vector<float>& pred, const std::vector<float>& target, int recon_n) -> float {
                if (recon_n <= 0) return 0.0f;
                double acc = 0.0;
                if (recon_loss_type == "l1" || recon_loss_type == "mae") {
                    for (int i = 0; i < recon_n; ++i) {
                        const double d = (double)pred[(size_t)i] - (double)target[(size_t)i];
                        acc += std::abs(d);
                    }
                } else {
                    for (int i = 0; i < recon_n; ++i) {
                        const double d = (double)pred[(size_t)i] - (double)target[(size_t)i];
                        acc += d * d;
                    }
                }
                acc /= (double)std::max(1, recon_n);
                return (float)acc;
            };

            auto compute_val_kl = [&](const std::vector<float>& pred, int image_dim, int latent_dim) -> float {
                if (image_dim <= 0 || latent_dim <= 0) return 0.0f;

                float logvar_min = -10.0f;
                float logvar_max = 10.0f;
                try {
                    if (ctx.modelConfig.contains("logvar_clip_min")) logvar_min = ctx.modelConfig["logvar_clip_min"].get<float>();
                    if (ctx.modelConfig.contains("logvar_clip_max")) logvar_max = ctx.modelConfig["logvar_clip_max"].get<float>();
                } catch (...) {
                }
                if (logvar_min > logvar_max) std::swap(logvar_min, logvar_max);

                const int mu_off = image_dim;
                const int lv_off = image_dim + latent_dim;
                if ((int)pred.size() < image_dim + 2 * latent_dim) return 0.0f;

                double kl = 0.0;
                for (int i = 0; i < latent_dim; ++i) {
                    const float mu_f = pred[(size_t)(mu_off + i)];
                    const float lv_raw = pred[(size_t)(lv_off + i)];
                    const float lv = std::clamp(lv_raw, logvar_min, logvar_max);
                    const double mu = (double)mu_f;
                    const double ev = std::exp((double)lv);
                    kl += 0.5 * (mu * mu + ev - 1.0 - (double)lv);
                }
                kl /= (double)std::max(1, latent_dim);
                return (float)kl;
            };

            const std::string step_prefix = "[" + model_type + "]";
            bool stopped_by_ui = false;

            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(train_indices.begin(), train_indices.end(), rng);
                ctx.addLog("Epoch " + std::to_string(epoch_offset + (epoch + 1)) + "/" + std::to_string(total_epochs_display) +
                           " (" + model_type + ") items=" + std::to_string(use_n));

                bool stop_requested = false;

                for (int k = 0; k < use_n; ++k) {
                    DatasetItem& item = ctx.currentDataset[(size_t)train_indices[(size_t)k]];
                    if (item.image_file.empty()) continue;

                    item.loadImageRGB(image_w, image_h);
                    if (!item.img_loaded || item.img.size() != expected_u8) {
                        continue;
                    }

                    // Normalize u8 -> [-1, 1]
                    for (size_t i = 0; i < expected_u8; ++i) {
                        x[i] = (float)((double)item.img[i] / 127.5 - 1.0);
                    }

                    Model::VAEStepStats st;
                    std::string prompt;
                    if (text_cond) {
                        if (!item.text_file.empty() && !item.text.has_value()) item.loadText();
                        prompt = item.text.has_value() ? item.text.value() : std::string();
                        std::vector<int> ids = ctx.currentTokenizer->tokenize(prompt);
                        if ((int)ids.size() < seq_len) ids.resize((size_t)seq_len, pad_id);
                        else if ((int)ids.size() > seq_len) ids.resize((size_t)seq_len);
                        poll_viz_live_params();
                        st = ctx.currentModel->trainStepVAEText(x, ids, opt, step_learning_rate());
                    } else {
                        poll_viz_live_params();
                        st = ctx.currentModel->trainStepVAE(x, opt, step_learning_rate());
                    }

                    global_step += 1;
                    log_step(global_step, st, step_prefix.c_str());
                    monitor_step(epoch + 1, k + 1, use_n, st);

                    // STOP depuis la Viz (bouton dans le panneau Metrics)
                    if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                        ctx.addLog("⛔ Stop demandé via Viz. Sauvegarde et arrêt...");
                        stop_requested = true;
                        stopped_by_ui = true;
                        break;
                    }

                    // Validation: forward-only sur un petit holdout, puis push dans Generated.
                    if (validate_every_steps > 0 && validate_items > 0 && !val_indices.empty() && (global_step % validate_every_steps) == 0) {
                        const int image_dim = (int)expected_u8;
                        int latent_dim = 0;
                        try {
                            if (ctx.modelConfig.contains("latent_dim")) latent_dim = std::max(0, ctx.modelConfig["latent_dim"].get<int>());
                        } catch (...) {
                        }

                        const int total = std::min((int)val_indices.size(), std::max(1, validate_items));
                        if (ctx.asyncMonitor) ctx.asyncMonitor->updateValidation(true, global_step, 0, total, false, false, 0.0f, 0.0f, 0.0f);

                        const bool taps_prev = ctx.currentModel->isVizTapsEnabled();

                        // Sample a subset each time (shuffle copy for randomness)
                        std::vector<int> val_pick = val_indices;
                        std::shuffle(val_pick.begin(), val_pick.end(), rng);
                        if ((int)val_pick.size() > total) val_pick.resize((size_t)total);

                        double acc_recon = 0.0;
                        double acc_kl = 0.0;
                        int done = 0;
                        bool val_ok = true;

                        for (int vi = 0; vi < (int)val_pick.size(); ++vi) {
                            if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                                val_ok = false;
                                stop_requested = true;
                                stopped_by_ui = true;
                                break;
                            }

                            DatasetItem& vitem = ctx.currentDataset[(size_t)val_pick[(size_t)vi]];
                            if (vitem.image_file.empty()) continue;

                            vitem.loadImageRGB(image_w, image_h);
                            if (!vitem.img_loaded || vitem.img.size() != expected_u8) continue;

                            // Normalize u8 -> [-1, 1]
                            for (size_t i = 0; i < expected_u8; ++i) {
                                x[i] = (float)((double)vitem.img[i] / 127.5 - 1.0);
                            }

                            const std::vector<float>* ppred = nullptr;
                            std::vector<int> ids;
                            std::string vprompt;
                            if (text_cond) {
                                if (!vitem.text_file.empty() && !vitem.text.has_value()) vitem.loadText();
                                vprompt = vitem.text.has_value() ? vitem.text.value() : std::string();
                                ids = ctx.currentTokenizer->tokenize(vprompt);
                                if ((int)ids.size() < seq_len) ids.resize((size_t)seq_len, pad_id);
                                else if ((int)ids.size() > seq_len) ids.resize((size_t)seq_len);

                                std::unordered_map<std::string, std::vector<float>> fin;
                                std::unordered_map<std::string, std::vector<int>> iin;
                                fin["__input__"] = x;
                                iin["text_ids"] = ids;
                                ppred = &ctx.currentModel->forwardPassNamedView(fin, iin, false);
                            } else {
                                ppred = &ctx.currentModel->forwardPassView(x, false);
                            }
                            if (!ppred) continue;
                            const std::vector<float>& pred = *ppred;
                            if ((int)pred.size() < image_dim + 2) continue;

                            const int recon_n = std::min(image_dim, (int)x.size());
                            const float vrecon = compute_val_recon(pred, x, recon_n);
                            float vkl = 0.0f;
                            if (latent_dim > 0) {
                                vkl = compute_val_kl(pred, image_dim, latent_dim);
                            }

                            acc_recon += (double)vrecon;
                            acc_kl += (double)vkl;
                            done += 1;

                            // Push images to Generated (target + recon)
                            if (ctx.asyncMonitor) {
                                const std::string idx = "i=" + std::to_string(val_pick[(size_t)vi]) + " step=" + std::to_string(global_step);
                                ctx.asyncMonitor->addImage(vitem.img, image_w, image_h, image_c, std::string("VAL target | ") + idx);
                                pack_f32_to_u8(pred, 0, recon_u8);
                                ctx.asyncMonitor->addImage(recon_u8, image_w, image_h, image_c, std::string("VAL recon | ") + idx);
                            }

                            if (ctx.asyncMonitor) {
                                const float avg_recon = (done > 0) ? (float)(acc_recon / (double)done) : 0.0f;
                                const float avg_kl = (done > 0) ? (float)(acc_kl / (double)done) : 0.0f;
                                ctx.asyncMonitor->updateValidation(true, global_step, done, total, true, false, avg_recon, avg_kl, 0.0f);
                            }
                        }

                        ctx.currentModel->setVizTapsEnabled(taps_prev);

                        const float final_recon = (done > 0) ? (float)(acc_recon / (double)done) : 0.0f;
                        const float final_kl = (done > 0) ? (float)(acc_kl / (double)done) : 0.0f;
                        if (ctx.asyncMonitor) ctx.asyncMonitor->updateValidation(false, global_step, done, total, true, val_ok, final_recon, final_kl, 0.0f);

                        // Backbone readiness: heuristique basée sur validation.
                        if (backbone_ready_enabled && val_ok && done > 0 && !backbone_ready_written) {
                            backbone_recon_hist.push_back(final_recon);
                            backbone_kl_hist.push_back(final_kl);
                            backbone_step_hist.push_back(global_step);
                            while ((int)backbone_recon_hist.size() > backbone_ready_window) backbone_recon_hist.pop_front();
                            while ((int)backbone_kl_hist.size() > backbone_ready_window) backbone_kl_hist.pop_front();
                            while ((int)backbone_step_hist.size() > backbone_ready_window) backbone_step_hist.pop_front();

                            const bool have_window = ((int)backbone_recon_hist.size() >= backbone_ready_window);
                            const bool warm_enough = (global_step >= backbone_ready_min_steps);

                            if (have_window && warm_enough) {
                                const float r0 = backbone_recon_hist.front();
                                const float r1 = backbone_recon_hist.back();
                                const float k1 = backbone_kl_hist.back();
                                const float denom = std::max(1e-12f, std::abs(r0));
                                const float rel_improve = (r0 - r1) / denom;
                                const float abs_improve = std::abs(r0 - r1);

                                const bool plateau = (rel_improve >= 0.0f) && (rel_improve <= backbone_ready_plateau_rel) && (abs_improve <= backbone_ready_plateau_abs);

                                bool recon_ok = true;
                                if (backbone_ready_recon_target > 0.0f) {
                                    recon_ok = (r1 <= backbone_ready_recon_target);
                                }

                                bool kl_ok = (k1 >= backbone_ready_kl_min);
                                if (backbone_ready_kl_max > 0.0f) {
                                    kl_ok = kl_ok && (k1 <= backbone_ready_kl_max);
                                }

                                if (plateau && recon_ok && kl_ok) {
                                    // Écrire un marqueur JSON dans checkpoint_dir.
                                    try {
                                        namespace fs = std::filesystem;
                                        fs::path out_base = checkpoint_dir.empty() ? fs::path(".") : fs::path(checkpoint_dir);
                                        std::error_code ec;
                                        fs::create_directories(out_base, ec);
                                        fs::path out_path = out_base / backbone_ready_file;

                                        json j;
                                        j["type"] = "vae_backbone_ready";
                                        j["model_type"] = model_type;
                                        j["global_step"] = global_step;
                                        j["epoch"] = epoch_offset + (epoch + 1);
                                        j["validate_items"] = done;
                                        j["val_recon"] = final_recon;
                                        j["val_kl"] = final_kl;
                                        j["plateau_rel"] = backbone_ready_plateau_rel;
                                        j["plateau_abs"] = backbone_ready_plateau_abs;
                                        j["recon_target"] = backbone_ready_recon_target;
                                        j["kl_min"] = backbone_ready_kl_min;
                                        j["kl_max"] = backbone_ready_kl_max;
                                        j["window"] = backbone_ready_window;
                                        j["min_steps"] = backbone_ready_min_steps;
                                        j["recon_history"] = backbone_recon_hist;
                                        j["kl_history"] = backbone_kl_hist;
                                        j["step_history"] = backbone_step_hist;

                                        // Contexte utile pour diffusion
                                        try {
                                            if (ctx.modelConfig.contains("image_w")) j["image_w"] = ctx.modelConfig["image_w"].get<int>();
                                            if (ctx.modelConfig.contains("image_h")) j["image_h"] = ctx.modelConfig["image_h"].get<int>();
                                            if (ctx.modelConfig.contains("image_c")) j["image_c"] = ctx.modelConfig["image_c"].get<int>();
                                            if (ctx.modelConfig.contains("latent_h")) j["latent_h"] = ctx.modelConfig["latent_h"].get<int>();
                                            if (ctx.modelConfig.contains("latent_w")) j["latent_w"] = ctx.modelConfig["latent_w"].get<int>();
                                            if (ctx.modelConfig.contains("latent_c")) j["latent_c"] = ctx.modelConfig["latent_c"].get<int>();
                                            if (ctx.modelConfig.contains("base_channels")) j["base_channels"] = ctx.modelConfig["base_channels"].get<int>();
                                            if (ctx.modelConfig.contains("stochastic_latent")) j["stochastic_latent"] = ctx.modelConfig["stochastic_latent"].get<bool>();
                                            if (ctx.modelConfig.contains("kl_beta")) j["kl_beta"] = ctx.modelConfig["kl_beta"].get<float>();
                                        } catch (...) {
                                        }

                                        std::ofstream f(out_path);
                                        f << j.dump(2);
                                        f.close();

                                        backbone_ready_written = true;
                                        ctx.addLog("✅ VAE backbone prêt pour diffusion (marqueur: " + out_path.string() + ")");
                                    } catch (...) {
                                        // Ne casse pas l'entraînement si l'écriture échoue.
                                        ctx.addLog("⚠️  VAE backbone readiness: échec écriture marqueur JSON");
                                    }

                                    if (backbone_ready_stop) {
                                        ctx.addLog("⛔ backbone_ready_stop=true: arrêt demandé (plateau atteint)");
                                        stop_requested = true;
                                        stopped_by_ui = true;
                                    }
                                }
                            }
                        }

                        if (stop_requested) {
                            break;
                        }
                    }

                    // VIZ: pousser le contexte dataset + blocs uniquement tous les log_every steps
                    // (réduit le coût; la viz reste “stale” entre deux updates, volontairement)
                    if (viz_active && ctx.asyncMonitor && ctx.currentModel && ctx.asyncMonitor->getViz() != nullptr && ((global_step % log_every) == 0)) {
                        std::string label = "vae_conv/input/dataset/rgb";
                        label += "/i=" + std::to_string(train_indices[(size_t)k]);

                        ctx.asyncMonitor->setDatasetSample(
                            item.img,
                            image_w,
                            image_h,
                            image_c,
                            label,
                            prompt,
                            std::string(),
                            std::string()
                        );

                        // Important UX:
                        // Par défaut, on affiche les "taps" en état réel (ceux produits par
                        // les passes réelles du modèle durant l'entraînement), sans forcer
                        // un forward additionnel en mode inference/freeze.
                        //
                        // Si l'utilisateur veut explicitement un snapshot inference (ex: pour
                        // rendre la recon plus lisible quand stochastic_latent=true), il peut
                        // activer `viz_taps_force_inference=true` dans la config.
                        bool viz_force_inference = false;
                        try {
                            if (ctx.modelConfig.contains("viz_taps_force_inference")) {
                                viz_force_inference = ctx.modelConfig["viz_taps_force_inference"].get<bool>();
                            }
                        } catch (...) {
                        }

                        if (viz_force_inference) {
                            ctx.currentModel->clearVizTaps();
                            try {
                                if (text_cond) {
                                    std::vector<int> ids = ctx.currentTokenizer ? ctx.currentTokenizer->tokenize(prompt) : std::vector<int>();
                                    if ((int)ids.size() < seq_len) ids.resize((size_t)seq_len, pad_id);
                                    else if ((int)ids.size() > seq_len) ids.resize((size_t)seq_len);

                                    std::unordered_map<std::string, std::vector<float>> fin;
                                    std::unordered_map<std::string, std::vector<int>> iin;
                                    fin["__input__"] = x;
                                    iin["text_ids"] = std::move(ids);
                                    (void)ctx.currentModel->forwardPassNamedView(fin, iin, false);
                                } else {
                                    (void)ctx.currentModel->forwardPassView(x, false);
                                }
                            } catch (...) {
                                // best-effort: ne jamais casser le training à cause de la viz
                            }
                        }

                        auto taps = ctx.currentModel->consumeVizTaps();
                        std::vector<Visualizer::BlockFrame> frames;
                        frames.reserve(taps.size());
                        for (auto& f : taps) {
                            Visualizer::BlockFrame bf;
                            bf.pixels = std::move(f.pixels);
                            bf.w = f.w;
                            bf.h = f.h;
                            bf.channels = f.channels;
                            bf.label = std::move(f.label);
                            frames.push_back(std::move(bf));
                        }
                        // Important UX: même si aucun tap n'est émis, vider les frames précédentes.
                        ctx.asyncMonitor->setLayerBlockImages(frames);
                    }
                }

                // Autosave à la fin de chaque epoch
                if (!stop_requested && autosave_every_epochs > 0 && ((epoch + 1) % autosave_every_epochs) == 0) {
                    std::string err_save;
                    (void)do_checkpoint_save(epoch + 1, std::string(), &err_save);
                    if (!err_save.empty()) {
                        ctx.addLog("⚠ Autosave échoué: " + err_save);
                    }
                }

                if (stop_requested) {
                    // Sauvegarde forcée même si autosave désactivé
                    std::string err_save;
                    (void)do_checkpoint_save(epoch + 1, "_stop", &err_save);
                    if (!err_save.empty()) {
                        ctx.addLog("⚠ Save(stop) échoué: " + err_save);
                    }
                    break;
                }
            }

            ctx.currentModel->setSerializedOptimizer(std::move(opt));

            if (stopped_by_ui) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "STOP_REQUESTED");
                return 2;
            }
            lua_pushboolean(L, true);
            lua_pushinteger(L, global_step);
            return 2;
        } else if (model_type == "vgg16_feat") {
            int image_w = 0, image_h = 0, image_c = 3;
            int grid = 8;
            int viz_taps_every_steps = 0;
            try {
                if (ctx.modelConfig.contains("image_w")) image_w = ctx.modelConfig["image_w"].get<int>();
                if (ctx.modelConfig.contains("image_h")) image_h = ctx.modelConfig["image_h"].get<int>();
                if (ctx.modelConfig.contains("image_c")) image_c = ctx.modelConfig["image_c"].get<int>();
                if (ctx.modelConfig.contains("pretrain_grid")) grid = ctx.modelConfig["pretrain_grid"].get<int>();
                if (ctx.modelConfig.contains("viz_taps_every_steps")) viz_taps_every_steps = ctx.modelConfig["viz_taps_every_steps"].get<int>();
                else if (ctx.modelConfig.contains("viz_every_steps")) viz_taps_every_steps = ctx.modelConfig["viz_every_steps"].get<int>();
            } catch (...) {
            }
            if (image_w <= 0 || image_h <= 0) {
                // fallback: dataset config
                try {
                    if (ctx.currentConfig.contains("dataset")) {
                        if (ctx.currentConfig["dataset"].contains("target_w")) image_w = ctx.currentConfig["dataset"]["target_w"].get<int>();
                        if (ctx.currentConfig["dataset"].contains("target_h")) image_h = ctx.currentConfig["dataset"]["target_h"].get<int>();
                    }
                } catch (...) {
                }
            }
            image_w = std::max(1, image_w);
            image_h = std::max(1, image_h);
            image_c = std::max(1, image_c);
            grid = std::clamp(grid, 2, 32);

            // Viz taps frequency (best-effort). Default: follow log_every.
            if (viz_active) {
                if (viz_taps_every_steps <= 0) viz_taps_every_steps = log_every;
                viz_taps_every_steps = std::max(1, viz_taps_every_steps);
            } else {
                viz_taps_every_steps = 0;
            }

            auto push_viz_taps = [&]() {
                if (!viz_active) return;
                if (!ctx.currentModel || !ctx.asyncMonitor) return;
                if (ctx.asyncMonitor->getViz() == nullptr) return;
                auto taps = ctx.currentModel->consumeVizTaps();
                std::vector<Visualizer::BlockFrame> frames;
                frames.reserve(taps.size());
                for (auto& f : taps) {
                    Visualizer::BlockFrame bf;
                    bf.pixels = std::move(f.pixels);
                    bf.w = f.w;
                    bf.h = f.h;
                    bf.channels = f.channels;
                    bf.label = std::move(f.label);
                    frames.push_back(std::move(bf));
                }
                // Important UX: même si aucun tap n'est émis, vider les frames précédentes.
                ctx.asyncMonitor->setLayerBlockImages(frames);
            };

            // Filter usable items
            std::vector<int> indices;
            indices.reserve(ctx.currentDataset.size());
            for (size_t i = 0; i < ctx.currentDataset.size(); ++i) {
                if (!ctx.currentDataset[i].image_file.empty()) indices.push_back((int)i);
            }
            if (indices.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Dataset: aucun item avec image_file (requis pour vgg16_feat)");
                return 2;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
            if (max_items > 0 && (int)indices.size() > max_items) indices.resize((size_t)max_items);

            const int use_n = (int)indices.size();
            opt.total_steps = std::max(1, epochs * std::max(1, use_n));

            const size_t expected_u8 = (size_t)image_w * (size_t)image_h * (size_t)image_c;
            std::vector<float> x;
            x.resize(expected_u8);

            auto compute_patch_means = [&](const std::vector<float>& src_hwc, int out_dim) -> std::vector<float> {
                // Features: grid*grid*image_c means (HWC input)
                const int feat_dim = grid * grid * image_c;
                std::vector<double> sum((size_t)feat_dim, 0.0);
                std::vector<int> cnt((size_t)(grid * grid), 0);

                for (int yy = 0; yy < image_h; ++yy) {
                    const int gy = (yy * grid) / image_h;
                    for (int xx = 0; xx < image_w; ++xx) {
                        const int gx = (xx * grid) / image_w;
                        const int cell = gy * grid + gx;
                        const int off = cell * image_c;
                        const size_t pix = (size_t)(yy * image_w + xx) * (size_t)image_c;
                        for (int cc = 0; cc < image_c; ++cc) {
                            sum[(size_t)off + (size_t)cc] += (double)src_hwc[pix + (size_t)cc];
                        }
                        cnt[(size_t)cell] += 1;
                    }
                }

                std::vector<float> feat((size_t)feat_dim, 0.0f);
                for (int cell = 0; cell < grid * grid; ++cell) {
                    const int n = std::max(1, cnt[(size_t)cell]);
                    const int off = cell * image_c;
                    for (int cc = 0; cc < image_c; ++cc) {
                        feat[(size_t)off + (size_t)cc] = (float)(sum[(size_t)off + (size_t)cc] / (double)n);
                    }
                }

                // Truncate/pad to out_dim
                out_dim = std::max(1, out_dim);
                std::vector<float> y((size_t)out_dim, 0.0f);
                const int m = std::min(out_dim, feat_dim);
                for (int i = 0; i < m; ++i) y[(size_t)i] = feat[(size_t)i];
                return y;
            };

            bool stopped_by_ui = false;

            // Perf stats vgg16_feat -> Viz.
            std::chrono::steady_clock::time_point last_vgg_metrics_ts;
            bool has_last_vgg_metrics_ts = false;
            auto apply_vgg_perf_stats = [&](AsyncMonitor::Metrics& m) {
                const auto now = std::chrono::steady_clock::now();
                if (has_last_vgg_metrics_ts) {
                    const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_vgg_metrics_ts).count();
                    if (dt_ms > 0 && dt_ms < 60LL * 60LL * 1000LL) {
                        m.batch_time_ms = (int)dt_ms;
                        m.bps = 1000.0f / (float)m.batch_time_ms;
                    }
                }
                last_vgg_metrics_ts = now;
                has_last_vgg_metrics_ts = true;

                auto& guard = MemoryGuard::instance();
                m.memory_mb = guard.getCurrentBytes() / 1024 / 1024;
            };

            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(indices.begin(), indices.end(), rng);
                ctx.addLog("Epoch " + std::to_string(epoch_offset + (epoch + 1)) + "/" + std::to_string(total_epochs_display) +
                           " (vgg16_feat) items=" + std::to_string(use_n));

                bool stop_requested = false;

                for (int k = 0; k < use_n; ++k) {
                    DatasetItem& item = ctx.currentDataset[(size_t)indices[(size_t)k]];
                    if (item.image_file.empty()) continue;

                    item.loadImageRGB(image_w, image_h);
                    if (!item.img_loaded || item.img.size() != expected_u8) {
                        continue;
                    }

                    // Normalize u8 -> [-1, 1]
                    for (size_t i = 0; i < expected_u8; ++i) {
                        x[i] = (float)((double)item.img[i] / 127.5 - 1.0);
                    }

                    // Forward
                    ctx.currentModel->zeroGradients();
                    const std::vector<float>& pred = ctx.currentModel->forwardPassView(x, true);
                    const int out_dim = std::max(1, (int)pred.size());

                    // Target is a deterministic downsample summary of the same input.
                    const std::vector<float> y = compute_patch_means(x, out_dim);

                    // Loss + grad
                    double loss = 0.0;
                    std::vector<float> grad;
                    grad.resize((size_t)out_dim);
                    const double inv = 1.0 / (double)std::max(1, out_dim);
                    for (int i = 0; i < out_dim; ++i) {
                        const double d = (double)pred[(size_t)i] - (double)y[(size_t)i];
                        loss += d * d;
                        grad[(size_t)i] = (float)(2.0 * d * inv);
                    }
                    loss *= inv;

                    // Backward + step
                    ctx.currentModel->backwardPass(grad);
                    poll_viz_live_params();
                    ctx.currentModel->optimizerStep(opt, step_learning_rate(), nullptr);

                    // Metrics for UI/log
                    global_step += 1;

                    // Viz taps (throttled)
                    if (viz_taps_every_steps > 0 && (global_step % viz_taps_every_steps) == 0) {
                        push_viz_taps();
                    }

                    if ((global_step % log_every) == 0) {
                        ctx.addLog(std::string("[vgg16_feat]") +
                                   " step=" + std::to_string(global_step) +
                                   " loss=" + std::to_string((float)loss) +
                                   " lr=" + std::to_string(opt.getCurrentLR()));
                    }
                    if (ctx.asyncMonitor) {
                        AsyncMonitor::Metrics m;
                        m.epoch = epoch_offset + (epoch + 1);
                        m.total_epochs = total_epochs_display;
                        m.batch = k + 1;
                        m.total_batches = use_n;
                        m.loss = (float)loss;
                        m.avg_loss = (float)loss;
                        m.lr = opt.getCurrentLR();
                        m.mse = (float)loss;
                        m.grad_norm = 0.0f;
                        m.grad_max = 0.0f;
                        m.params = ctx.currentModel ? ctx.currentModel->totalParamCount() : 0;
                        m.recon_loss_type = "mse";
                        m.opt_type = (int)opt.type;
                        m.opt_step = (int)opt.step;
                        m.opt_beta1 = opt.beta1;
                        m.opt_beta2 = opt.beta2;
                        m.opt_eps = opt.eps;
                        m.opt_weight_decay = opt.weight_decay;

                        apply_vgg_perf_stats(m);
                        ctx.asyncMonitor->updateMetrics(m);
                    }

                    // STOP via Viz
                    if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                        ctx.addLog("⛔ Stop demandé via Viz. Sauvegarde et arrêt...");
                        stop_requested = true;
                        stopped_by_ui = true;
                        break;
                    }
                }

                // Autosave per epoch
                if (!checkpoint_dir.empty() && autosave_every_epochs > 0) {
                    const int epoch_1based = epoch + 1;
                    if ((epoch_1based % autosave_every_epochs) == 0) {
                        std::string save_err;
                        if (!do_checkpoint_save(epoch_1based, std::string(), &save_err)) {
                            ctx.addLog("⚠️ Autosave failed: " + save_err);
                        }
                    }
                }

                if (stop_requested) break;
            }

            // Save optimizer state back into the model for later Serialization.save()
            if (ctx.currentModel) {
                Optimizer opt_snapshot = opt;
                ctx.currentModel->setSerializedOptimizer(std::move(opt_snapshot));
            }

            if (stopped_by_ui) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "STOP_REQUESTED");
                return 2;
            }

            lua_pushboolean(L, true);
            return 1;
        } else if (model_type == "vgg16" || model_type == "vgg19") {
            // Multi-label image tag classification.
            // Dataset items must have image_file + text, where text is dot-separated tags/short phrases.
            // Requires cfg.tags_vocab (array of strings) to define fixed class indices.

            int image_w = 0, image_h = 0, image_c = 3;
            int viz_taps_every_steps = 0;
            bool lowercase_tags = true;
            float bce_pos_weight = 1.0f;

            // Validation (optional)
            int validate_every_steps = 0;
            int validate_every_epochs = 0;
            int validate_items = 0;
            float validate_threshold = 0.5f;
            bool validate_holdout = false;
            float validate_holdout_frac = 0.0f;
            int validate_holdout_items = 0;
            int validate_seed = 12345;
            try {
                if (ctx.modelConfig.contains("image_w")) image_w = ctx.modelConfig["image_w"].get<int>();
                if (ctx.modelConfig.contains("image_h")) image_h = ctx.modelConfig["image_h"].get<int>();
                if (ctx.modelConfig.contains("image_c")) image_c = ctx.modelConfig["image_c"].get<int>();
                if (ctx.modelConfig.contains("viz_taps_every_steps")) viz_taps_every_steps = ctx.modelConfig["viz_taps_every_steps"].get<int>();
                else if (ctx.modelConfig.contains("viz_every_steps")) viz_taps_every_steps = ctx.modelConfig["viz_every_steps"].get<int>();
                if (ctx.modelConfig.contains("lowercase_tags")) lowercase_tags = ctx.modelConfig["lowercase_tags"].get<bool>();

                if (ctx.modelConfig.contains("pos_weight")) bce_pos_weight = ctx.modelConfig["pos_weight"].get<float>();
                else if (ctx.modelConfig.contains("bce_pos_weight")) bce_pos_weight = ctx.modelConfig["bce_pos_weight"].get<float>();

                if (ctx.modelConfig.contains("validate_every_steps")) validate_every_steps = std::max(0, ctx.modelConfig["validate_every_steps"].get<int>());
                else if (ctx.modelConfig.contains("validate_every")) validate_every_steps = std::max(0, ctx.modelConfig["validate_every"].get<int>());

                if (ctx.modelConfig.contains("validate_every_epochs")) validate_every_epochs = std::max(0, ctx.modelConfig["validate_every_epochs"].get<int>());
                else if (ctx.modelConfig.contains("validate_epochs")) validate_every_epochs = std::max(0, ctx.modelConfig["validate_epochs"].get<int>());

                if (ctx.modelConfig.contains("validate_items")) validate_items = std::max(0, ctx.modelConfig["validate_items"].get<int>());
                else if (ctx.modelConfig.contains("val_items")) validate_items = std::max(0, ctx.modelConfig["val_items"].get<int>());

                if (ctx.modelConfig.contains("validate_threshold")) validate_threshold = ctx.modelConfig["validate_threshold"].get<float>();
                else if (ctx.modelConfig.contains("val_threshold")) validate_threshold = ctx.modelConfig["val_threshold"].get<float>();

                if (ctx.modelConfig.contains("validate_holdout")) validate_holdout = ctx.modelConfig["validate_holdout"].get<bool>();
                else if (ctx.modelConfig.contains("val_holdout")) validate_holdout = ctx.modelConfig["val_holdout"].get<bool>();

                if (ctx.modelConfig.contains("validate_holdout_frac")) validate_holdout_frac = ctx.modelConfig["validate_holdout_frac"].get<float>();
                else if (ctx.modelConfig.contains("val_holdout_frac")) validate_holdout_frac = ctx.modelConfig["val_holdout_frac"].get<float>();

                if (ctx.modelConfig.contains("validate_holdout_items")) validate_holdout_items = std::max(0, ctx.modelConfig["validate_holdout_items"].get<int>());
                else if (ctx.modelConfig.contains("val_holdout_items")) validate_holdout_items = std::max(0, ctx.modelConfig["val_holdout_items"].get<int>());

                if (ctx.modelConfig.contains("validate_seed")) validate_seed = ctx.modelConfig["validate_seed"].get<int>();
                else if (ctx.modelConfig.contains("val_seed")) validate_seed = ctx.modelConfig["val_seed"].get<int>();
            } catch (...) {
            }

            bce_pos_weight = std::max(0.0f, bce_pos_weight);
            validate_threshold = std::clamp(validate_threshold, 0.0f, 1.0f);
            validate_holdout_frac = std::clamp(validate_holdout_frac, 0.0f, 0.95f);

            image_w = std::max(1, image_w);
            image_h = std::max(1, image_h);
            image_c = std::max(1, image_c);

            if (!ctx.modelConfig.contains("tags_vocab") || !ctx.modelConfig["tags_vocab"].is_array()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "vgg16/vgg19 multi-label: cfg.tags_vocab manquant (liste de tags/classes)");
                return 2;
            }

            const auto& vocab_json = ctx.modelConfig["tags_vocab"];
            const int num_classes = (int)vocab_json.size();
            if (num_classes <= 0) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "vgg16/vgg19 multi-label: cfg.tags_vocab vide");
                return 2;
            }

            auto trim = [](std::string& s) {
                auto is_ws = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
                size_t a = 0;
                while (a < s.size() && is_ws((unsigned char)s[a])) a++;
                size_t b = s.size();
                while (b > a && is_ws((unsigned char)s[b - 1])) b--;
                if (a != 0 || b != s.size()) s = s.substr(a, b - a);
            };

            auto trim_punct = [&](std::string& s) {
                // Retire ponctuation fréquente autour des tags (ex: "tag1," "(tag2)" "tag3\"")
                auto is_junk = [](unsigned char c) {
                    switch (c) {
                        case ' ': case '\t': case '\n': case '\r':
                        case ',': case ';': case ':':
                        case '!': case '?':
                        case '"': case '\'':
                        case '(': case ')':
                        case '[': case ']':
                        case '{': case '}':
                        case '<': case '>':
                        case '/': case '\\':
                        case '|':
                        case '-':
                            return true;
                        default:
                            return false;
                    }
                };
                size_t a = 0;
                while (a < s.size() && is_junk((unsigned char)s[a])) a++;
                size_t b = s.size();
                while (b > a && is_junk((unsigned char)s[b - 1])) b--;
                if (a != 0 || b != s.size()) s = s.substr(a, b - a);
                trim(s);
            };

            // Build label->class_id map.
            std::unordered_map<std::string, int> tag_to_id;
            tag_to_id.reserve((size_t)num_classes * 2ULL);
            for (int i = 0; i < num_classes; ++i) {
                try {
                    std::string t = vocab_json[(size_t)i].get<std::string>();
                    if (t.empty()) continue;
                    trim_punct(t);
                    if (t.empty()) continue;
                    if (lowercase_tags) {
                        for (char& ch : t) ch = (char)std::tolower((unsigned char)ch);
                    }
                    tag_to_id.emplace(std::move(t), i);
                } catch (...) {
                }
            }
            if (tag_to_id.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "vgg16/vgg19 multi-label: cfg.tags_vocab invalide (aucun tag utilisable)");
                return 2;
            }

            // Viz taps frequency (best-effort). Default: follow log_every.
            if (viz_active) {
                if (viz_taps_every_steps <= 0) viz_taps_every_steps = log_every;
                viz_taps_every_steps = std::max(1, viz_taps_every_steps);
            } else {
                viz_taps_every_steps = 0;
            }

            auto push_viz_taps = [&]() {
                if (!viz_active) return;
                if (!ctx.currentModel || !ctx.asyncMonitor) return;
                if (ctx.asyncMonitor->getViz() == nullptr) return;
                auto taps = ctx.currentModel->consumeVizTaps();
                std::vector<Visualizer::BlockFrame> frames;
                frames.reserve(taps.size());
                for (auto& f : taps) {
                    Visualizer::BlockFrame bf;
                    bf.pixels = std::move(f.pixels);
                    bf.w = f.w;
                    bf.h = f.h;
                    bf.channels = f.channels;
                    bf.label = std::move(f.label);
                    frames.push_back(std::move(bf));
                }
                ctx.asyncMonitor->setLayerBlockImages(frames);
            };

            // Filter usable items
            std::vector<int> indices;
            indices.reserve(ctx.currentDataset.size());
            for (size_t i = 0; i < ctx.currentDataset.size(); ++i) {
                const auto& it = ctx.currentDataset[i];
                // Note: text is often lazy-loaded. Require a text_file path instead.
                if (!it.image_file.empty() && !it.text_file.empty()) indices.push_back((int)i);
            }
            if (indices.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Dataset: aucun item avec image_file + text_file (requis pour vgg16/vgg19 multi-label)");
                return 2;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
            if (max_items > 0 && (int)indices.size() > max_items) indices.resize((size_t)max_items);

            const bool validation_enabled = (validate_items > 0) && ((validate_every_steps > 0) || (validate_every_epochs > 0));

            // Split train/val if requested; default is to validate on train indices (no holdout).
            std::vector<int> train_indices = indices;
            std::vector<int> val_indices;
            if (validation_enabled) {
                if (validate_holdout && (int)indices.size() >= 2) {
                    const int n = (int)indices.size();
                    int val_n = 0;
                    if (validate_holdout_items > 0) {
                        val_n = std::clamp(validate_holdout_items, 1, n - 1);
                    } else if (validate_holdout_frac > 0.0f) {
                        val_n = (int)std::floor((double)n * (double)validate_holdout_frac);
                        val_n = std::clamp(val_n, 1, n - 1);
                    }
                    if (val_n > 0) {
                        val_indices.assign(indices.end() - val_n, indices.end());
                        train_indices.assign(indices.begin(), indices.end() - val_n);
                    } else {
                        val_indices = train_indices;
                    }
                } else {
                    val_indices = train_indices;
                }

                ctx.addLog("Validation: every_steps=" + std::to_string(validate_every_steps) +
                           " every_epochs=" + std::to_string(validate_every_epochs) +
                           " items=" + std::to_string(validate_items) +
                           " threshold=" + std::to_string(validate_threshold) +
                           " holdout=" + std::string(validate_holdout ? "true" : "false") +
                           " train=" + std::to_string((int)train_indices.size()) +
                           " val=" + std::to_string((int)val_indices.size()));
            }

            const int use_n = (int)train_indices.size();

            opt.total_steps = std::max(1, epochs * std::max(1, use_n));
            const size_t expected_u8 = (size_t)image_w * (size_t)image_h * (size_t)image_c;
            std::vector<float> x_chw;
            x_chw.resize(expected_u8);
            std::vector<float> y;
            y.resize((size_t)num_classes);

            auto find_icase = [](const std::string& haystack, const std::string& needle) -> size_t {
                if (needle.empty()) return 0;
                if (haystack.size() < needle.size()) return std::string::npos;
                for (size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
                    bool ok = true;
                    for (size_t j = 0; j < needle.size(); ++j) {
                        const unsigned char a = (unsigned char)haystack[i + j];
                        const unsigned char b = (unsigned char)needle[j];
                        if ((char)std::tolower(a) != (char)std::tolower(b)) { ok = false; break; }
                    }
                    if (ok) return i;
                }
                return std::string::npos;
            };

            auto extract_tags_section = [&](const std::string& txt) -> std::string {
                // Supporte un format de caption type:
                // --- TAGS ---\n tag1.tag2 ... \n--- DESCRIPTION ---\n ...
                const std::string kTags = "--- TAGS ---";
                const std::string kDesc = "--- DESCRIPTION ---";
                size_t a = find_icase(txt, kTags);
                if (a == std::string::npos) return txt;
                a += kTags.size();
                size_t b = find_icase(txt, kDesc);
                if (b == std::string::npos || b < a) {
                    return txt.substr(a);
                }
                return txt.substr(a, b - a);
            };

            auto split_tags = [&](const std::string& txt0) {
                std::vector<std::string> out;
                out.reserve(16);

                const std::string txt = extract_tags_section(txt0);
                std::string cur;
                cur.reserve(txt.size());
                for (char ch : txt) {
                    const bool is_sep = (ch == '.' || ch == ',' || ch == ';' || ch == '\n' || ch == '\r' || ch == '\t' || ch == '|');
                    if (!is_sep) {
                        cur.push_back(ch);
                        continue;
                    }

                    trim_punct(cur);
                    if (!cur.empty()) {
                        if (lowercase_tags) {
                            for (char& c : cur) c = (char)std::tolower((unsigned char)c);
                        }
                        out.push_back(cur);
                    }
                    cur.clear();
                }

                trim_punct(cur);
                if (!cur.empty()) {
                    if (lowercase_tags) {
                        for (char& c : cur) c = (char)std::tolower((unsigned char)c);
                    }
                    out.push_back(cur);
                }
                return out;
            };

            auto sigmoid = [](double z) {
                if (z >= 0.0) {
                    const double ez = std::exp(-z);
                    return 1.0 / (1.0 + ez);
                } else {
                    const double ez = std::exp(z);
                    return ez / (1.0 + ez);
                }
            };

            struct ValStats {
                int items = 0;
                double loss = 0.0;
                double f1_micro = 0.0;
                double pos_true_rate = 0.0;
                double pos_pred_rate = 0.0;
                double avg_prob = 0.0;
            };

            std::mt19937 val_rng((uint32_t)validate_seed);

            auto run_validation = [&](int step, int epoch_1based) -> ValStats {
                ValStats st;
                if (!validation_enabled) return st;
                if (!ctx.currentModel) return st;
                if (val_indices.empty()) return st;

                std::vector<int> work = val_indices;
                std::shuffle(work.begin(), work.end(), val_rng);

                const int want = std::min(validate_items, (int)work.size());
                if (want <= 0) return st;

                long long tp = 0, fp = 0, fn = 0;
                long long true_pos = 0;
                long long pred_pos = 0;
                double prob_sum = 0.0;
                double loss_sum = 0.0;

                int done = 0;
                for (int j = 0; j < (int)work.size() && done < want; ++j) {
                    DatasetItem& item = ctx.currentDataset[(size_t)work[(size_t)j]];
                    if (item.image_file.empty() || item.text_file.empty()) continue;
                    if (!item.loadText() || !item.text.has_value()) continue;

                    item.loadImageRGB(image_w, image_h);
                    if (!item.img_loaded || item.img.size() != expected_u8) continue;

                    // Labels
                    std::fill(y.begin(), y.end(), 0.0f);
                    const std::vector<std::string> tags = split_tags(item.text.value());
                    for (const auto& t : tags) {
                        auto it = tag_to_id.find(t);
                        if (it != tag_to_id.end()) {
                            const int id = it->second;
                            if (id >= 0 && id < num_classes) {
                                y[(size_t)id] = 1.0f;
                            }
                        }
                    }

                    // Image -> CHW
                    for (int yy = 0; yy < image_h; ++yy) {
                        for (int xx = 0; xx < image_w; ++xx) {
                            const size_t pix = (size_t)(yy * image_w + xx) * (size_t)image_c;
                            for (int cc = 0; cc < image_c; ++cc) {
                                const float v = (float)((double)item.img[pix + (size_t)cc] / 127.5 - 1.0);
                                const size_t idx = (size_t)cc * (size_t)(image_h * image_w) + (size_t)(yy * image_w + xx);
                                x_chw[idx] = v;
                            }
                        }
                    }

                    const std::vector<float>& logits = ctx.currentModel->forwardPassView(x_chw, false);
                    if ((int)logits.size() != num_classes) continue;

                    double loss = 0.0;
                    for (int i = 0; i < num_classes; ++i) {
                        const double z = (double)logits[(size_t)i];
                        const double yi = (double)y[(size_t)i];
                        const double sp_pos = std::max(0.0, z) + std::log1p(std::exp(-std::fabs(z)));
                        const double sp_neg = std::max(0.0, -z) + std::log1p(std::exp(-std::fabs(z)));
                        const double wpos = (yi >= 0.5) ? (double)bce_pos_weight : 1.0;
                        loss += (1.0 - yi) * sp_pos + yi * wpos * sp_neg;

                        const double p = sigmoid(z);
                        prob_sum += p;
                        const bool y1 = (yi >= 0.5);
                        const bool p1 = (p >= (double)validate_threshold);
                        if (y1) true_pos++;
                        if (p1) pred_pos++;
                        if (p1 && y1) tp++;
                        else if (p1 && !y1) fp++;
                        else if (!p1 && y1) fn++;
                    }
                    loss *= (1.0 / (double)std::max(1, num_classes));
                    loss_sum += loss;
                    done++;
                }

                st.items = done;
                if (done <= 0) return st;

                const double denom = (double)done * (double)std::max(1, num_classes);
                st.loss = loss_sum / (double)done;
                st.pos_true_rate = denom > 0 ? (double)true_pos / denom : 0.0;
                st.pos_pred_rate = denom > 0 ? (double)pred_pos / denom : 0.0;
                st.avg_prob = denom > 0 ? prob_sum / denom : 0.0;
                const double f1_den = (2.0 * (double)tp + (double)fp + (double)fn);
                st.f1_micro = (f1_den > 0.0) ? (2.0 * (double)tp) / f1_den : 0.0;

                ctx.addLog(std::string("[") + model_type + "][val]" +
                           " step=" + std::to_string(step) +
                           " epoch=" + std::to_string(epoch_1based) +
                           " items=" + std::to_string(done) +
                           " loss=" + std::to_string((float)st.loss) +
                           " f1_micro=" + std::to_string((float)st.f1_micro) +
                           " pos_true=" + std::to_string((float)st.pos_true_rate) +
                           " pos_pred=" + std::to_string((float)st.pos_pred_rate) +
                           " avg_p=" + std::to_string((float)st.avg_prob));

                if (ctx.asyncMonitor) {
                    AsyncMonitor::Metrics vm;
                    vm.epoch = epoch_offset + epoch_1based;
                    vm.total_epochs = total_epochs_display;
                    vm.batch = 0;
                    vm.total_batches = use_n;
                    vm.loss = 0.0f;
                    vm.avg_loss = 0.0f;
                    vm.lr = opt.getCurrentLR();
                    vm.params = ctx.currentModel ? ctx.currentModel->totalParamCount() : 0;
                    vm.recon_loss_type = "bce_logits";

                    vm.val_has = true;
                    vm.val_ok = true;
                    vm.val_in_progress = false;
                    vm.val_step = step;
                    vm.val_items = done;
                    vm.val_done = done;
                    vm.val_total = done;
                    vm.val_recon = (float)st.loss;
                    vm.val_align = (float)st.f1_micro;

                    vm.opt_type = (int)opt.type;
                    vm.opt_step = (int)opt.step;
                    vm.opt_beta1 = opt.beta1;
                    vm.opt_beta2 = opt.beta2;
                    vm.opt_eps = opt.eps;
                    vm.opt_weight_decay = opt.weight_decay;

                    ctx.asyncMonitor->updateMetrics(vm);
                }

                // Collapse warnings (simple heuristics)
                if (st.pos_true_rate < 1e-9) {
                    ctx.addLog("⚠️  Labels: pos_true_rate=0 (aucun tag du texte ne matche tags_vocab). "
                               "Le modèle va naturellement converger vers des prédictions ~0.");
                } else if (st.pos_pred_rate < 1e-3 || st.pos_pred_rate > 1.0 - 1e-3) {
                    ctx.addLog("⚠️  Possible collapse: pos_pred_rate=" + std::to_string((float)st.pos_pred_rate) +
                               " (pos_true_rate=" + std::to_string((float)st.pos_true_rate) +
                               ", avg_p=" + std::to_string((float)st.avg_prob) + ")");
                }

                return st;
            };

            bool stopped_by_ui = false;

            // Perf stats vgg16_tags_multilabel -> Viz.
            std::chrono::steady_clock::time_point last_tags_metrics_ts;
            bool has_last_tags_metrics_ts = false;
            auto apply_tags_perf_stats = [&](AsyncMonitor::Metrics& m) {
                const auto now = std::chrono::steady_clock::now();
                if (has_last_tags_metrics_ts) {
                    const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_tags_metrics_ts).count();
                    if (dt_ms > 0 && dt_ms < 60LL * 60LL * 1000LL) {
                        m.batch_time_ms = (int)dt_ms;
                        m.bps = 1000.0f / (float)m.batch_time_ms;
                    }
                }
                last_tags_metrics_ts = now;
                has_last_tags_metrics_ts = true;

                auto& guard = MemoryGuard::instance();
                m.memory_mb = guard.getCurrentBytes() / 1024 / 1024;
            };
            bool warned_no_vocab_match = false;

            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(train_indices.begin(), train_indices.end(), rng);
                ctx.addLog("Epoch " + std::to_string(epoch_offset + (epoch + 1)) + "/" + std::to_string(total_epochs_display) +
                           " (" + std::string(model_type) + " multi-label) items=" + std::to_string(use_n));

                bool stop_requested = false;

                for (int k = 0; k < use_n; ++k) {
                    DatasetItem& item = ctx.currentDataset[(size_t)train_indices[(size_t)k]];
                    if (item.image_file.empty() || item.text_file.empty()) continue;

                    // Lazy-load text on demand (uses DatasetMemoryManager).
                    if (!item.loadText() || !item.text.has_value()) {
                        continue;
                    }

                    item.loadImageRGB(image_w, image_h);
                    if (!item.img_loaded || item.img.size() != expected_u8) {
                        continue;
                    }

                    // Build labels (multi-hot)
                    std::fill(y.begin(), y.end(), 0.0f);
                    const std::vector<std::string> tags = split_tags(item.text.value());
                    int matched_tags = 0;
                    for (const auto& t : tags) {
                        auto it = tag_to_id.find(t);
                        if (it != tag_to_id.end()) {
                            const int id = it->second;
                            if (id >= 0 && id < num_classes) {
                                y[(size_t)id] = 1.0f;
                                matched_tags++;
                            }
                        }
                    }

                    // Normalize u8 -> [-1, 1] AND permute HWC -> CHW (Conv2d expects CHW)
                    for (int yy = 0; yy < image_h; ++yy) {
                        for (int xx = 0; xx < image_w; ++xx) {
                            const size_t pix = (size_t)(yy * image_w + xx) * (size_t)image_c;
                            for (int cc = 0; cc < image_c; ++cc) {
                                const float v = (float)((double)item.img[pix + (size_t)cc] / 127.5 - 1.0);
                                const size_t idx = (size_t)cc * (size_t)(image_h * image_w) + (size_t)(yy * image_w + xx);
                                x_chw[idx] = v;
                            }
                        }
                    }

                    // Forward
                    ctx.currentModel->zeroGradients();
                    const std::vector<float>& logits = ctx.currentModel->forwardPassView(x_chw, true);
                    const int out_dim = (int)logits.size();
                    if (out_dim != num_classes) {
                        lua_pushboolean(L, false);
                        lua_pushstring(L, "vgg16/vgg19 multi-label: output_dim != num_classes (cfg.num_classes doit matcher tags_vocab)");
                        return 2;
                    }

                    // BCEWithLogits loss + gradient (+ optional positive class reweighting)
                    double loss = 0.0;
                    std::vector<float> grad;
                    grad.resize((size_t)num_classes);
                    const double inv = 1.0 / (double)std::max(1, num_classes);

                    long long tp = 0, fp = 0, fn = 0;
                    long long true_pos = 0;
                    long long pred_pos = 0;
                    double prob_sum = 0.0;
                    for (int i = 0; i < num_classes; ++i) {
                        const double z = (double)logits[(size_t)i];
                        const double yi = (double)y[(size_t)i];
                        const double sp_pos = std::max(0.0, z) + std::log1p(std::exp(-std::fabs(z)));
                        const double sp_neg = std::max(0.0, -z) + std::log1p(std::exp(-std::fabs(z)));
                        const double wpos = (yi >= 0.5) ? (double)bce_pos_weight : 1.0;
                        loss += (1.0 - yi) * sp_pos + yi * wpos * sp_neg;
                        const double p = sigmoid(z);
                        const double gw = (yi >= 0.5) ? (double)bce_pos_weight : 1.0;
                        grad[(size_t)i] = (float)((p - yi) * gw * inv);

                        prob_sum += p;
                        const bool y1 = (yi >= 0.5);
                        const bool p1 = (p >= (double)validate_threshold);
                        if (y1) true_pos++;
                        if (p1) pred_pos++;
                        if (p1 && y1) tp++;
                        else if (p1 && !y1) fp++;
                        else if (!p1 && y1) fn++;
                    }
                    loss *= inv;

                    ctx.currentModel->backwardPass(grad);
                    poll_viz_live_params();
                    ctx.currentModel->optimizerStep(opt, step_learning_rate(), nullptr);

                    global_step += 1;

                    if (validation_enabled && validate_every_steps > 0 && (global_step % validate_every_steps) == 0) {
                        (void)run_validation(global_step, (epoch + 1));
                    }

                    if (viz_taps_every_steps > 0 && (global_step % viz_taps_every_steps) == 0) {
                        push_viz_taps();
                    }

                    if ((global_step % log_every) == 0) {
                        const double denom = (double)std::max(1, num_classes);
                        const double pos_true_rate = denom > 0.0 ? (double)true_pos / denom : 0.0;
                        const double pos_pred_rate = denom > 0.0 ? (double)pred_pos / denom : 0.0;
                        const double avg_prob = denom > 0.0 ? prob_sum / denom : 0.0;
                        const double f1_den = (2.0 * (double)tp + (double)fp + (double)fn);
                        const double f1_micro = (f1_den > 0.0) ? (2.0 * (double)tp) / f1_den : 0.0;
                        ctx.addLog(std::string("[") + model_type + "]" +
                                   " step=" + std::to_string(global_step) +
                                   " loss=" + std::to_string((float)loss) +
                                   " lr=" + std::to_string(opt.getCurrentLR()) +
                                   " pos_true=" + std::to_string((float)pos_true_rate) +
                                   " pos_pred=" + std::to_string((float)pos_pred_rate) +
                                   " f1_micro=" + std::to_string((float)f1_micro) +
                                   " matched_tags=" + std::to_string(matched_tags) +
                                   " parsed_tags=" + std::to_string((int)tags.size()));

                        if (!warned_no_vocab_match && ((int)tags.size() > 0) && matched_tags == 0) {
                            warned_no_vocab_match = true;
                            ctx.addLog("⚠️  Aucun tag ne matche tags_vocab (matched_tags=0). "
                                       "Parsing supporté: ., , ; | et nouvelles lignes + section '--- TAGS ---'.");
                        }
                    }
                    if (ctx.asyncMonitor) {
                        AsyncMonitor::Metrics m;
                        m.epoch = epoch_offset + (epoch + 1);
                        m.total_epochs = total_epochs_display;
                        m.batch = k + 1;
                        m.total_batches = use_n;
                        m.loss = (float)loss;
                        m.avg_loss = (float)loss;
                        m.lr = opt.getCurrentLR();
                        m.mse = (float)loss;
                        m.grad_norm = 0.0f;
                        m.grad_max = 0.0f;
                        m.params = ctx.currentModel ? ctx.currentModel->totalParamCount() : 0;
                        m.recon_loss_type = "bce_logits";
                        m.opt_type = (int)opt.type;
                        m.opt_step = (int)opt.step;
                        m.opt_beta1 = opt.beta1;
                        m.opt_beta2 = opt.beta2;
                        m.opt_eps = opt.eps;
                        m.opt_weight_decay = opt.weight_decay;

                        apply_tags_perf_stats(m);
                        ctx.asyncMonitor->updateMetrics(m);
                    }

                    if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                        ctx.addLog("⛔ Stop demandé via Viz. Sauvegarde et arrêt...");
                        stop_requested = true;
                        stopped_by_ui = true;
                        break;
                    }
                }

                if (validation_enabled && validate_every_epochs > 0) {
                    const int epoch_1based = epoch + 1;
                    if ((epoch_1based % validate_every_epochs) == 0) {
                        (void)run_validation(global_step, epoch_1based);
                    }
                }

                // Autosave per epoch
                if (!checkpoint_dir.empty() && autosave_every_epochs > 0) {
                    const int epoch_1based = epoch + 1;
                    if ((epoch_1based % autosave_every_epochs) == 0) {
                        std::string save_err;
                        if (!do_checkpoint_save(epoch_1based, std::string(), &save_err)) {
                            ctx.addLog("⚠️ Autosave failed: " + save_err);
                        }
                    }
                }

                if (stop_requested) break;
            }

            if (ctx.currentModel) {
                Optimizer opt_snapshot = opt;
                ctx.currentModel->setSerializedOptimizer(std::move(opt_snapshot));
            }

            if (stopped_by_ui) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "STOP_REQUESTED");
                return 2;
            }

            lua_pushboolean(L, true);
            return 1;
        }

        // -------------------------------
        // VAEText (text_ids) -> recon
        // -------------------------------
        if (model_type == "ponyxl_ddpm") {
            auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
            if (!pony) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Le modèle courant n'est pas un PonyXLDDPMModel (type=ponyxl_ddpm attendu)");
                return 2;
            }

            int image_w = 0;
            int image_h = 0;
            try {
                if (ctx.modelConfig.contains("image_w")) image_w = ctx.modelConfig["image_w"].get<int>();
                if (ctx.modelConfig.contains("image_h")) image_h = ctx.modelConfig["image_h"].get<int>();
            } catch (...) {
            }
            if (image_w <= 0 || image_h <= 0) {
                try {
                    if (ctx.currentConfig.contains("dataset")) {
                        if (ctx.currentConfig["dataset"].contains("target_w")) image_w = ctx.currentConfig["dataset"]["target_w"].get<int>();
                        if (ctx.currentConfig["dataset"].contains("target_h")) image_h = ctx.currentConfig["dataset"]["target_h"].get<int>();
                    }
                } catch (...) {
                }
            }
            image_w = std::max(1, image_w);
            image_h = std::max(1, image_h);

            int steps_per_image = 1;
            try {
                if (ctx.modelConfig.contains("steps_per_image")) steps_per_image = ctx.modelConfig["steps_per_image"].get<int>();
                else if (ctx.modelConfig.contains("ddpm_steps_per_image")) steps_per_image = ctx.modelConfig["ddpm_steps_per_image"].get<int>();
            } catch (...) {
            }
            steps_per_image = std::max(1, steps_per_image);

            // Validation config (best-effort, optional)
            int validate_every_steps = 0;
            int validate_items = 0;
            float validate_holdout_frac = 0.0f;
            int validate_holdout_items = 0;
            bool validate_holdout = true;
            int validate_seed = 12345;
            int validate_t = -1;
            try {
                if (ctx.modelConfig.contains("validate_every_steps")) validate_every_steps = std::max(0, ctx.modelConfig["validate_every_steps"].get<int>());
                else if (ctx.modelConfig.contains("validate_every")) validate_every_steps = std::max(0, ctx.modelConfig["validate_every"].get<int>());

                if (ctx.modelConfig.contains("validate_items")) validate_items = std::max(0, ctx.modelConfig["validate_items"].get<int>());
                if (ctx.modelConfig.contains("validate_holdout_frac")) validate_holdout_frac = ctx.modelConfig["validate_holdout_frac"].get<float>();
                if (ctx.modelConfig.contains("validate_holdout_items")) validate_holdout_items = std::max(0, ctx.modelConfig["validate_holdout_items"].get<int>());
                if (ctx.modelConfig.contains("validate_holdout")) validate_holdout = ctx.modelConfig["validate_holdout"].get<bool>();

                if (ctx.modelConfig.contains("validate_seed")) validate_seed = ctx.modelConfig["validate_seed"].get<int>();
                else if (ctx.modelConfig.contains("val_seed")) validate_seed = ctx.modelConfig["val_seed"].get<int>();

                if (ctx.modelConfig.contains("validate_t")) validate_t = ctx.modelConfig["validate_t"].get<int>();
                else if (ctx.modelConfig.contains("validate_ddpm_step")) validate_t = ctx.modelConfig["validate_ddpm_step"].get<int>();
            } catch (...) {
            }
            validate_holdout_frac = std::clamp(validate_holdout_frac, 0.0f, 0.95f);
            validate_every_steps = std::max(0, validate_every_steps);
            validate_items = std::max(0, validate_items);

            std::vector<int> indices;
            indices.reserve(ctx.currentDataset.size());
            for (size_t i = 0; i < ctx.currentDataset.size(); ++i) {
                if (!ctx.currentDataset[i].image_file.empty()) indices.push_back((int)i);
            }
            if (indices.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Dataset: aucun item avec image_file (requis pour ponyxl_ddpm)");
                return 2;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
            if (max_items > 0 && (int)indices.size() > max_items) indices.resize((size_t)max_items);

            // Split train/val (holdout) only if validation is enabled.
            std::vector<int> train_indices = indices;
            std::vector<int> val_indices;
            if (validate_every_steps > 0 && validate_items > 0 && (int)indices.size() >= 2) {
                if (validate_holdout) {
                    const int n = (int)indices.size();
                    int val_n = 0;
                    if (validate_holdout_items > 0) {
                        val_n = std::clamp(validate_holdout_items, 1, n - 1);
                    } else if (validate_holdout_frac > 0.0f) {
                        val_n = (int)std::floor((double)n * (double)validate_holdout_frac);
                        val_n = std::clamp(val_n, 1, n - 1);
                    }

                    if (val_n > 0) {
                        val_indices.assign(indices.end() - val_n, indices.end());
                        train_indices.assign(indices.begin(), indices.end() - val_n);
                        ctx.addLog("Validation holdout: train=" + std::to_string((int)train_indices.size()) + " val=" + std::to_string((int)val_indices.size()) +
                                   " (every=" + std::to_string(validate_every_steps) + " steps, items=" + std::to_string(validate_items) + ")");
                    } else {
                        val_indices = train_indices;
                        ctx.addLog("Validation (no holdout): using train set for val (every=" + std::to_string(validate_every_steps) + " steps, items=" + std::to_string(validate_items) + ")");
                    }
                } else {
                    val_indices = train_indices;
                    ctx.addLog("Validation (no holdout): using train set for val (every=" + std::to_string(validate_every_steps) + " steps, items=" + std::to_string(validate_items) + ")");
                }
            }

            const int use_n = (int)train_indices.size();
            opt.total_steps = std::max(1, epochs * std::max(1, use_n) * steps_per_image);

            const size_t expected_u8 = (size_t)image_w * (size_t)image_h * 3ULL;

            auto log_step_ddpm = [&](int global_step, const PonyXLDDPMModel::StepStats& st) {
                if ((global_step % log_every) != 0) return;
                ctx.addLog(std::string("[ponyxl_ddpm]") +
                           " step=" + std::to_string(global_step) +
                           " loss=" + std::to_string(st.loss) +
                           " t=" + std::to_string(st.timestep) +
                           " kl=" + std::to_string(st.kl_divergence) +
                           " grad_norm=" + std::to_string(st.grad_norm) +
                           " grad_max=" + std::to_string(st.grad_max_abs));
            };

            auto monitor_step_ddpm = [&](int epoch_1based, int batch_1based, int total_batches, const PonyXLDDPMModel::StepStats& st) {
                if (!ctx.asyncMonitor) return;

                static std::chrono::steady_clock::time_point last_ddpm_metrics_ts;
                static bool has_last_ddpm_metrics_ts = false;

                AsyncMonitor::Metrics m;
                m.epoch = epoch_offset + epoch_1based;
                m.total_epochs = total_epochs_display;
                m.batch = batch_1based;
                m.total_batches = total_batches;
                m.loss = st.loss;
                m.avg_loss = st.loss;
                m.lr = opt.getCurrentLR();
                // Le training diffusion utilise une loss de type MSE sur eps (en pratique).
                // On la mappe aussi dans la colonne générique "mse" du logger.
                m.mse = st.loss;
                m.kl = st.kl_divergence;
                m.wass = st.wasserstein;
                m.ent = st.entropy_diff;
                m.mom = st.moment_mismatch;
                m.spat = st.spatial_coherence;
                m.temp = st.temporal_consistency;
                m.timestep = st.timestep;
                m.grad_norm = st.grad_norm;
                m.grad_max = st.grad_max_abs;
                m.kl_beta_effective = st.kl_beta_effective;
                m.params = ctx.currentModel ? ctx.currentModel->totalParamCount() : 0;
                m.recon_loss_type = recon_loss_type;

                // Mémoire (best-effort): expose l'usage courant du MemoryGuard si actif.
                {
                    auto& guard = MemoryGuard::instance();
                    m.memory_mb = guard.getCurrentBytes() / 1024 / 1024;
                }

                {
                    const auto now = std::chrono::steady_clock::now();
                    if (has_last_ddpm_metrics_ts) {
                        const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_ddpm_metrics_ts).count();
                        if (dt_ms > 0 && dt_ms < 60LL * 60LL * 1000LL) {
                            m.batch_time_ms = (int)dt_ms;
                            m.bps = 1000.0f / (float)m.batch_time_ms;
                        }
                    }
                    last_ddpm_metrics_ts = now;
                    has_last_ddpm_metrics_ts = true;
                }

                m.opt_type = (int)opt.type;
                m.opt_step = (int)opt.step;
                m.opt_beta1 = opt.beta1;
                m.opt_beta2 = opt.beta2;
                m.opt_eps = opt.eps;
                m.opt_weight_decay = opt.weight_decay;

                ctx.asyncMonitor->updateMetrics(m);
            };

            bool stopped_by_ui = false;

            auto get_prompt_for_item = [&](DatasetItem& it) -> std::string {
                if (!it.text_file.empty() && !it.text.has_value()) it.loadText();
                if (it.text.has_value()) return it.text.value();
                return std::string();
            };

            auto pick_wrong_prompt = [&](const std::string& ref, std::mt19937& prng) -> std::string {
                if (ref.empty()) return std::string();
                const std::vector<int>& pool = !val_indices.empty() ? val_indices : train_indices;
                if (pool.size() <= 1) return std::string();
                std::uniform_int_distribution<int> dist(0, (int)pool.size() - 1);
                for (int tries = 0; tries < 8; ++tries) {
                    const int idx = pool[(size_t)dist(prng)];
                    DatasetItem& cand = ctx.currentDataset[(size_t)idx];
                    std::string p = get_prompt_for_item(cand);
                    if (!p.empty() && p != ref) return p;
                }
                return std::string();
            };

            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(train_indices.begin(), train_indices.end(), rng);
                ctx.addLog("Epoch " + std::to_string(epoch_offset + (epoch + 1)) + "/" + std::to_string(total_epochs_display) +
                           " (ponyxl_ddpm) items=" + std::to_string(use_n));

                bool stop_requested = false;

                for (int k = 0; k < use_n; ++k) {
                    DatasetItem& item = ctx.currentDataset[(size_t)train_indices[(size_t)k]];
                    if (item.image_file.empty()) continue;

                    item.loadImageRGB(image_w, image_h);
                    if (!item.img_loaded || item.img.size() != expected_u8) {
                        continue;
                    }

                    std::string prompt = get_prompt_for_item(item);

                    poll_viz_live_params();
                    const PonyXLDDPMModel::StepStats st = pony->trainStepSdxlLatentDiffusion(
                        prompt, item.img, image_w, image_h, opt, step_learning_rate());

                    global_step += 1;
                    log_step_ddpm(global_step, st);
                    monitor_step_ddpm(epoch + 1, k + 1, use_n, st);

                    if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                        ctx.addLog("⛔ Stop demandé via Viz. Sauvegarde et arrêt...");
                        stop_requested = true;
                        stopped_by_ui = true;
                        break;
                    }

                    // Validation: forward-only périodique sur holdout, avec recon preview.
                    if (validate_every_steps > 0 && validate_items > 0 && !val_indices.empty() && (global_step % validate_every_steps) == 0) {
                        const int total = std::min((int)val_indices.size(), std::max(1, validate_items));
                        if (ctx.asyncMonitor) ctx.asyncMonitor->updateValidation(true, global_step, 0, total, false, false, 0.0f, 0.0f, 0.0f);

                        const bool taps_prev = ctx.currentModel->isVizTapsEnabled();
                        ctx.currentModel->setVizTapsEnabled(false);

                        std::vector<int> val_pick = val_indices;
                        std::shuffle(val_pick.begin(), val_pick.end(), rng);
                        if ((int)val_pick.size() > total) val_pick.resize((size_t)total);

                        std::mt19937 prng((uint32_t)(seed ^ (global_step * 2654435761u)));

                        double acc_img = 0.0;
                        double acc_eps = 0.0;
                        double acc_margin = 0.0;
                        int done = 0;
                        bool val_ok = true;

                        for (int vi = 0; vi < (int)val_pick.size(); ++vi) {
                            if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                                val_ok = false;
                                stop_requested = true;
                                stopped_by_ui = true;
                                break;
                            }

                            DatasetItem& vitem = ctx.currentDataset[(size_t)val_pick[(size_t)vi]];
                            if (vitem.image_file.empty()) continue;
                            vitem.loadImageRGB(image_w, image_h);
                            if (!vitem.img_loaded || vitem.img.size() != expected_u8) continue;

                            const std::string vprompt = get_prompt_for_item(vitem);
                            if (vprompt.empty()) continue;
                            const std::string wrong = pick_wrong_prompt(vprompt, prng);

                            const int vseed = validate_seed + val_pick[(size_t)vi];
                            const PonyXLDDPMModel::ValStats vst = pony->validateStepSdxlLatentDiffusion(
                                vprompt,
                                wrong,
                                vitem.img,
                                image_w,
                                image_h,
                                vseed,
                                validate_t
                            );

                            acc_img += vst.img_mse;
                            acc_eps += vst.eps_mse;
                            acc_margin += vst.assoc_margin;
                            done += 1;

                            if (ctx.asyncMonitor) {
                                const std::string idx = "i=" + std::to_string(val_pick[(size_t)vi]) + " step=" + std::to_string(global_step);
                                ctx.asyncMonitor->addImage(vitem.img, image_w, image_h, 3, std::string("VAL target | ") + idx);
                                auto prev = pony->reconstructPreviewSdxlLatentDiffusion(vprompt, vitem.img, image_w, image_h, 256, vseed, validate_t);
                                if (!prev.pixels.empty() && prev.w > 0 && prev.h > 0) {
                                    ctx.asyncMonitor->addImage(prev.pixels, prev.w, prev.h, prev.channels, std::string("VAL recon | ") + idx);
                                }

                                if (!wrong.empty()) {
                                    auto prev_wrong = pony->reconstructPreviewSdxlLatentDiffusion(wrong, vitem.img, image_w, image_h, 256, vseed, validate_t);
                                    if (!prev_wrong.pixels.empty() && prev_wrong.w > 0 && prev_wrong.h > 0) {
                                        ctx.asyncMonitor->addImage(prev_wrong.pixels, prev_wrong.w, prev_wrong.h, prev_wrong.channels,
                                                                  std::string("VAL recon WRONG | ") + idx);
                                    }
                                }
                            }

                            // VIZ: afficher le contexte + visuels d'assignation/dénoise émis par validateStep.
                            if (viz_active && ctx.asyncMonitor && ctx.currentModel && ctx.asyncMonitor->getViz() != nullptr) {
                                std::string label = "ponyxl_ddpm/val/input/dataset/rgb";
                                label += "/i=" + std::to_string(val_pick[(size_t)vi]);
                                ctx.asyncMonitor->setDatasetSample(
                                    vitem.img,
                                    image_w,
                                    image_h,
                                    3,
                                    label,
                                    vprompt,
                                    std::string(),
                                    std::string()
                                );

                                auto taps = ctx.currentModel->consumeVizTaps();
                                std::vector<Visualizer::BlockFrame> frames;
                                frames.reserve(taps.size());
                                for (auto& f : taps) {
                                    Visualizer::BlockFrame bf;
                                    bf.pixels = std::move(f.pixels);
                                    bf.w = f.w;
                                    bf.h = f.h;
                                    bf.channels = f.channels;
                                    bf.label = std::move(f.label);
                                    frames.push_back(std::move(bf));
                                }
                                ctx.asyncMonitor->setLayerBlockImages(frames);
                            }

                            if (ctx.asyncMonitor) {
                                const float avg_img = (done > 0) ? (float)(acc_img / (double)done) : 0.0f;
                                const float avg_eps = (done > 0) ? (float)(acc_eps / (double)done) : 0.0f;
                                const float avg_margin = (done > 0) ? (float)(acc_margin / (double)done) : 0.0f;
                                ctx.asyncMonitor->updateValidation(true, global_step, done, total, true, false, avg_img, avg_eps, avg_margin);
                            }
                        }

                        ctx.currentModel->setVizTapsEnabled(taps_prev);

                        const float final_img = (done > 0) ? (float)(acc_img / (double)done) : 0.0f;
                        const float final_eps = (done > 0) ? (float)(acc_eps / (double)done) : 0.0f;
                        const float final_margin = (done > 0) ? (float)(acc_margin / (double)done) : 0.0f;
                        if (ctx.asyncMonitor) ctx.asyncMonitor->updateValidation(false, global_step, done, total, true, val_ok, final_img, final_eps, final_margin);

                        if (stop_requested) {
                            break;
                        }
                    }

                    if (viz_active && ctx.asyncMonitor && ctx.currentModel && ctx.asyncMonitor->getViz() != nullptr) {
                        std::string label = "ponyxl_ddpm/input/dataset/rgb";
                        label += "/i=" + std::to_string(train_indices[(size_t)k]);

                        ctx.asyncMonitor->setDatasetSample(
                            item.img,
                            image_w,
                            image_h,
                            3,
                            label,
                            prompt,
                            std::string(),
                            std::string()
                        );

                        auto taps = ctx.currentModel->consumeVizTaps();
                        std::vector<Visualizer::BlockFrame> frames;
                        frames.reserve(taps.size());
                        for (auto& f : taps) {
                            Visualizer::BlockFrame bf;
                            bf.pixels = std::move(f.pixels);
                            bf.w = f.w;
                            bf.h = f.h;
                            bf.channels = f.channels;
                            bf.label = std::move(f.label);
                            frames.push_back(std::move(bf));
                        }
                        ctx.asyncMonitor->setLayerBlockImages(frames);
                    }
                }

                if (!stop_requested && autosave_every_epochs > 0 && ((epoch + 1) % autosave_every_epochs) == 0) {
                    std::string err_save;
                    (void)do_checkpoint_save(epoch + 1, std::string(), &err_save);
                    if (!err_save.empty()) {
                        ctx.addLog("⚠ Autosave échoué: " + err_save);
                    }
                }

                if (stop_requested) {
                    std::string err_save;
                    (void)do_checkpoint_save(epoch + 1, "_stop", &err_save);
                    if (!err_save.empty()) {
                        ctx.addLog("⚠ Save(stop) échoué: " + err_save);
                    }
                    break;
                }
            }

            ctx.currentModel->setSerializedOptimizer(std::move(opt));

            if (stopped_by_ui) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "STOP_REQUESTED");
                return 2;
            }
            lua_pushboolean(L, true);
            lua_pushinteger(L, global_step);
            return 2;
        }

        if (model_type == "vae_text") {
            if (!ctx.currentTokenizer) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Aucun tokenizer chargé (requis pour vae_text)");
                return 2;
            }

            std::vector<int> indices;
            indices.reserve(ctx.currentDataset.size());
            for (size_t i = 0; i < ctx.currentDataset.size(); ++i) {
                if (!ctx.currentDataset[i].text_file.empty()) indices.push_back((int)i);
            }
            if (indices.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Dataset: aucun item avec text_file (requis pour vae_text)");
                return 2;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
            if (max_items > 0 && (int)indices.size() > max_items) indices.resize((size_t)max_items);

            const int use_n = (int)indices.size();
            opt.total_steps = std::max(1, epochs * use_n);

            int seq_len = 256;
            int pad_id = ctx.currentTokenizer->getPadId();
            try {
                if (ctx.modelConfig.contains("seq_len")) seq_len = ctx.modelConfig["seq_len"].get<int>();
            } catch (...) {
            }
            seq_len = std::max(1, seq_len);

            std::vector<float> empty_x;

            bool stopped_by_ui = false;
            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(indices.begin(), indices.end(), rng);
                ctx.addLog("Epoch " + std::to_string(epoch_offset + (epoch + 1)) + "/" + std::to_string(total_epochs_display) +
                           " (vae_text) items=" + std::to_string(use_n));

                bool stop_requested = false;

                for (int k = 0; k < use_n; ++k) {
                    DatasetItem& item = ctx.currentDataset[(size_t)indices[(size_t)k]];
                    if (item.text_file.empty()) continue;
                    if (!item.text.has_value()) item.loadText();
                    if (!item.text.has_value()) continue;

                    std::vector<int> ids = ctx.currentTokenizer->tokenize(item.text.value());
                    if ((int)ids.size() < seq_len) ids.resize((size_t)seq_len, pad_id);
                    else if ((int)ids.size() > seq_len) ids.resize((size_t)seq_len);

                    poll_viz_live_params();
                    const Model::VAEStepStats st = ctx.currentModel->trainStepVAEText(empty_x, ids, opt, step_learning_rate());
                    global_step += 1;
                    log_step(global_step, st, "[vae_text]");
                    monitor_step(epoch + 1, k + 1, use_n, st);

                    if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                        ctx.addLog("⛔ Stop demandé via Viz. Sauvegarde et arrêt...");
                        stop_requested = true;
                        stopped_by_ui = true;
                        break;
                    }
                }

                if (!stop_requested && autosave_every_epochs > 0 && ((epoch + 1) % autosave_every_epochs) == 0) {
                    std::string err_save;
                    (void)do_checkpoint_save(epoch + 1, std::string(), &err_save);
                    if (!err_save.empty()) {
                        ctx.addLog("⚠ Autosave échoué: " + err_save);
                    }
                }

                if (stop_requested) {
                    std::string err_save;
                    (void)do_checkpoint_save(epoch + 1, "_stop", &err_save);
                    if (!err_save.empty()) {
                        ctx.addLog("⚠ Save(stop) échoué: " + err_save);
                    }
                    break;
                }
            }

            ctx.currentModel->setSerializedOptimizer(std::move(opt));

            if (stopped_by_ui) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "STOP_REQUESTED");
                return 2;
            }
            lua_pushboolean(L, true);
            lua_pushinteger(L, global_step);
            return 2;
        }

        lua_pushboolean(L, false);
        lua_pushstring(L, ("Model.train: type non supporté (type='" + model_type + "')").c_str());
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_inferModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Argument: input (string ou table)
    if (lua_isstring(L, 1)) {
        const char* input = lua_tostring(L, 1);
        ctx.addLog("Inférence sur: " + std::string(input));
        
        try {
            // Tokenize input
            std::vector<int> tokens;
            if (ctx.currentTokenizer) {
                tokens = ctx.currentTokenizer->tokenize(input);
            } else {
                // Simple word tokenization si pas de tokenizer
                tokens = {1, 2, 3, 4, 5}; // Placeholder
            }
            
            // Encode
            std::vector<float> encoding;
            if (ctx.currentEncoder) {
                encoding = ctx.currentEncoder->encode(tokens);
                ctx.currentModel->setLastEncoding(encoding);
            }
            
            // Forward pass
            std::vector<uint8_t> output;
            ctx.currentModel->forward(output);
            
            // Decode result
            auto result = ctx.currentModel->eval(output);
            
            // Convert tokens back to text
            if (ctx.currentTokenizer && !result.tokens.empty()) {
                std::string decoded = ctx.currentTokenizer->decode(result.tokens);
                lua_pushstring(L, decoded.c_str());
            } else {
                lua_pushstring(L, "[output tokens]");
            }
        } catch (const std::exception& e) {
            lua_pushstring(L, ("Error: " + std::string(e.what())).c_str());
        }
    } else {
        lua_pushnil(L);
    }
    
    return 1;
}

// ============================================================================
// ModelArchitectures Registry helpers (Lua)
// ============================================================================

int LuaScripting::lua_archAvailable(lua_State* L) {
    try {
        const auto names = ModelArchitectures::available();
        lua_createtable(L, static_cast<int>(names.size()), 0);
        int i = 1;
        for (const auto& name : names) {
            lua_pushstring(L, name.c_str());
            lua_rawseti(L, -2, i++);
        }
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_archDefaultConfig(lua_State* L) {
    const char* name = luaL_checkstring(L, 1);
    try {
        json cfg = ModelArchitectures::defaultConfig(name);
        jsonToLuaTable(L, cfg);
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_saveModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    const char* path = luaL_checkstring(L, 1);
    
    try {
        fs::path save_dir(path);
        
        // Créer le dossier si nécessaire
        if (!fs::exists(save_dir)) {
            fs::create_directories(save_dir);
        }
        
        // Sauvegarder le checkpoint
        std::vector<MagicToken> magic_tokens;  // Empty for now
        
        Tokenizer tokenizer = ctx.currentTokenizer ? *ctx.currentTokenizer : Tokenizer();
        
        bool success = ctx.currentModel->saveCheckpoint(
            tokenizer,
            magic_tokens,
            save_dir,
            0  // epoch 0
        );
        
        if (success) {
            ctx.addLog("Modèle sauvegardé: " + std::string(path));
            lua_pushboolean(L, true);
        } else {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Erreur lors de la sauvegarde");
            return 2;
        }
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

int LuaScripting::lua_loadModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    const char* path = luaL_checkstring(L, 1);
    
    try {
        fs::path load_dir(path);
        
        if (!fs::exists(load_dir)) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Le chemin n'existe pas");
            return 2;
        }
        
        // Chercher le fichier safetensors
        fs::path safetensor_path = load_dir / "weights.safetensors";
        if (!fs::exists(safetensor_path)) {
            safetensor_path = load_dir / "model.safetensors";
        }
        
        Tokenizer tokenizer;
        Encoder encoder;
        std::vector<MagicToken> magic_tokens;
        
        bool success = ctx.currentModel->tryLoadExistingModel(
            load_dir,
            safetensor_path,
            tokenizer,
            encoder,
            magic_tokens
        );
        
        if (success) {
            ctx.currentTokenizer = std::make_shared<Tokenizer>(tokenizer);
            ctx.currentEncoder = std::make_shared<Encoder>(encoder);

            // Si la viz est active, activer les taps pour permettre l'affichage des blocks.
            if (ctx.asyncMonitor && ctx.asyncMonitor->getViz() != nullptr && ctx.currentModel) {
                ctx.currentModel->setVizTapsEnabled(true);
                try {
                    int max_frames = 12;
                    int max_side = 64;
                    if (ctx.modelConfig.contains("viz_taps_max_frames")) max_frames = ctx.modelConfig["viz_taps_max_frames"].get<int>();
                    if (ctx.modelConfig.contains("viz_taps_max_side")) max_side = ctx.modelConfig["viz_taps_max_side"].get<int>();
                    ctx.currentModel->setVizTapsLimits(max_frames, max_side);
                } catch (...) {
                }
            }

            ctx.addLog("Modèle chargé: " + std::string(path));
            lua_pushboolean(L, true);
        } else {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Erreur lors du chargement");
            return 2;
        }
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

// ============================================================================
// New Serialization API
// ============================================================================

int LuaScripting::lua_saveCheckpoint(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Arguments: path, format (optionnel), options (optionnel)
    const char* path = luaL_checkstring(L, 1);
    const char* format_str = lua_isstring(L, 2) ? lua_tostring(L, 2) : "safetensors";
    
    using namespace Mimir::Serialization;
    
    // Parse format
    CheckpointFormat format = CheckpointFormat::SafeTensors;
    std::string fmt(format_str);
    if (fmt == "safetensors" || fmt == "st") {
        format = CheckpointFormat::SafeTensors;
    } else if (fmt == "raw_folder" || fmt == "raw" || fmt == "folder") {
        format = CheckpointFormat::RawFolder;
    } else if (fmt == "debug_json" || fmt == "debug" || fmt == "json") {
        format = CheckpointFormat::DebugJson;
    } else {
        lua_pushboolean(L, false);
        lua_pushstring(L, ("Format inconnu: " + fmt).c_str());
        return 2;
    }
    
    // Parse options from table (if provided)
    SaveOptions options;
    options.format = format;
    
    if (lua_istable(L, 3)) {
        lua_getfield(L, 3, "save_tokenizer");
        if (lua_isboolean(L, -1)) {
            options.save_tokenizer = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "save_encoder");
        if (lua_isboolean(L, -1)) {
            options.save_encoder = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "save_optimizer");
        if (lua_isboolean(L, -1)) {
            options.save_optimizer = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "debug_max_values");
        if (lua_isnumber(L, -1)) {
            options.debug_max_values = lua_tointeger(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "include_git_info");
        if (lua_isboolean(L, -1)) {
            options.include_git_info = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // Enhanced DebugJson options (v1.1.0)
        lua_getfield(L, 3, "include_gradients");
        if (lua_isboolean(L, -1)) {
            options.include_gradients = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "include_optimizer_state");
        if (lua_isboolean(L, -1)) {
            options.include_optimizer_state = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "max_values_per_tensor");
        if (lua_isnumber(L, -1)) {
            options.max_values_per_tensor = lua_tointeger(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "include_activations");
        if (lua_isboolean(L, -1)) {
            options.include_activations = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "include_checksums");
        if (lua_isboolean(L, -1)) {
            options.include_checksums = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        lua_getfield(L, 3, "include_weight_deltas");
        if (lua_isboolean(L, -1)) {
            options.include_weight_deltas = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
    }
    
    // Save
    std::string error;
    bool success = save_checkpoint(*ctx.currentModel, path, options, &error);
    
    if (success) {
        ctx.addLog("Checkpoint sauvegardé: " + std::string(path) + " (format: " + fmt + ")");
        lua_pushboolean(L, true);
        return 1;
    } else {
        lua_pushboolean(L, false);
        lua_pushstring(L, error.c_str());
        return 2;
    }
}

int LuaScripting::lua_loadCheckpoint(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    // Arguments: path, format (optionnel), options (optionnel)
    const char* path = luaL_checkstring(L, 1);
    const char* format_str = lua_isstring(L, 2) ? lua_tostring(L, 2) : nullptr;

    // Parse format (or auto-detect)
    Mimir::Serialization::CheckpointFormat format = Mimir::Serialization::CheckpointFormat::SafeTensors;
    if (format_str == nullptr) {
        format = Mimir::Serialization::detect_format(path);
    } else {
        std::string fmt(format_str);
        if (fmt == "auto") {
            format = Mimir::Serialization::detect_format(path);
        } else if (fmt == "safetensors" || fmt == "st") {
            format = Mimir::Serialization::CheckpointFormat::SafeTensors;
        } else if (fmt == "raw_folder" || fmt == "raw" || fmt == "folder") {
            format = Mimir::Serialization::CheckpointFormat::RawFolder;
        } else if (fmt == "debug_json" || fmt == "debug" || fmt == "json") {
            format = Mimir::Serialization::CheckpointFormat::DebugJson;
        } else {
            lua_pushboolean(L, false);
            lua_pushstring(L, ("Format inconnu: " + fmt).c_str());
            return 2;
        }
    }

    // Parse options from table (if provided)
    Mimir::Serialization::LoadOptions options;
    options.format = format;

    int options_idx = (format_str != nullptr) ? 3 : 2;
    if (lua_istable(L, options_idx)) {
        lua_getfield(L, options_idx, "load_tokenizer");
        if (lua_isboolean(L, -1)) options.load_tokenizer = lua_toboolean(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, options_idx, "load_encoder");
        if (lua_isboolean(L, -1)) options.load_encoder = lua_toboolean(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, options_idx, "load_optimizer");
        if (lua_isboolean(L, -1)) options.load_optimizer = lua_toboolean(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, options_idx, "strict_mode");
        if (lua_isboolean(L, -1)) options.strict_mode = lua_toboolean(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, options_idx, "validate_checksums");
        if (lua_isboolean(L, -1)) options.validate_checksums = lua_toboolean(L, -1);
        lua_pop(L, 1);
    }

    // Load
    std::string error;
    bool success = Mimir::Serialization::load_checkpoint(*ctx.currentModel, path, options, &error);

    if (success) {
        ctx.addLog("Checkpoint chargé: " + std::string(path));
        lua_pushboolean(L, true);
        return 1;
    }

    lua_pushboolean(L, false);
    lua_pushstring(L, error.c_str());
    return 2;
}

int LuaScripting::lua_detectFormat(lua_State* L) {
    const char* path = luaL_checkstring(L, 1);
    
    using namespace Mimir::Serialization;
    
    CheckpointFormat format = detect_format(path);
    
    switch (format) {
        case CheckpointFormat::SafeTensors:
            lua_pushstring(L, "SAFETENSORS");
            break;
        case CheckpointFormat::RawFolder:
            lua_pushstring(L, "RAWFOLDER");
            break;
        case CheckpointFormat::DebugJson:
            lua_pushstring(L, "DEBUGJSON");
            break;
        default:
            lua_pushnil(L);
            lua_pushstring(L, "Format inconnu");
            return 2;
    }
    
    return 1;
}

int LuaScripting::lua_saveEnhancedDebugJson(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Argument 1: path
    const char* path = luaL_checkstring(L, 1);
    
    // Argument 2 (optionnel): options table
    using namespace Mimir::Serialization;
    DebugJsonOptions options;
    
    if (lua_istable(L, 2)) {
        // include_gradients
        lua_getfield(L, 2, "include_gradients");
        if (lua_isboolean(L, -1)) {
            options.include_gradients = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // include_optimizer_state
        lua_getfield(L, 2, "include_optimizer_state");
        if (lua_isboolean(L, -1)) {
            options.include_optimizer_state = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // max_values_per_tensor
        lua_getfield(L, 2, "max_values_per_tensor");
        if (lua_isnumber(L, -1)) {
            options.max_values_per_tensor = lua_tointeger(L, -1);
        }
        lua_pop(L, 1);
        
        // include_activations
        lua_getfield(L, 2, "include_activations");
        if (lua_isboolean(L, -1)) {
            options.include_activations = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // include_checksums
        lua_getfield(L, 2, "include_checksums");
        if (lua_isboolean(L, -1)) {
            options.include_checksums = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // include_weight_deltas
        lua_getfield(L, 2, "include_weight_deltas");
        if (lua_isboolean(L, -1)) {
            options.include_weight_deltas = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // include_git_info
        lua_getfield(L, 2, "include_git_info");
        if (lua_isboolean(L, -1)) {
            options.include_git_info = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // save_tokenizer
        lua_getfield(L, 2, "save_tokenizer");
        if (lua_isboolean(L, -1)) {
            options.save_tokenizer = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
        
        // save_encoder
        lua_getfield(L, 2, "save_encoder");
        if (lua_isboolean(L, -1)) {
            options.save_encoder = lua_toboolean(L, -1);
        }
        lua_pop(L, 1);
    }
    
    // Save using enhanced debug JSON
    DebugJsonDump dumper;
    bool success = dumper.save_enhanced(path, *ctx.currentModel, options);
    
    if (success) {
        ctx.addLog("Enhanced debug JSON sauvegardé: " + std::string(path));
        lua_pushboolean(L, true);
        return 1;
    } else {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Erreur lors de la sauvegarde");
        return 2;
    }
}

// ============================================================================
// Tokenizer
// ============================================================================

int LuaScripting::lua_createTokenizer(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    int max_vocab = luaL_checkinteger(L, 1);
    
    ctx.currentTokenizer = std::make_shared<Tokenizer>(max_vocab);
    ctx.addLog("Tokenizer créé (vocab_max=" + std::to_string(max_vocab) + ")");
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_tokenize(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    auto tokens = ctx.currentTokenizer->tokenizeBPE(text);
    
    // Retourner une table Lua avec les tokens
    lua_newtable(L);
    for (size_t i = 0; i < tokens.size(); ++i) {
        lua_pushinteger(L, tokens[i]);
        lua_rawseti(L, -2, i + 1);  // Indices Lua commencent à 1
    }
    
    return 1;
}

int LuaScripting::lua_detokenize(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    // Lire table de tokens
    luaL_checktype(L, 1, LUA_TTABLE);
    
    std::vector<int> tokens;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        tokens.push_back(lua_tointeger(L, -1));
        lua_pop(L, 1);
    }
    
    std::string text = ctx.currentTokenizer->decode(tokens);
    lua_pushstring(L, text.c_str());
    
    return 1;
}

// ============================================================================
// Dataset
// ============================================================================

// ---------------------------------------------------------------------------
// Database (dataset loader builder with caching)
// ---------------------------------------------------------------------------

int LuaScripting::lua_databaseLoad(lua_State* L) {
    // Builder object returned to Lua. It can optionally capture initial args.
    // Example:
    //   Mimir.Database.load(dir, 64, 64, 1).cache("dataset_cache.json", 10240, true)
    // Also supports:
    //   Mimir.Database.load().cache(dir)

    const int top = lua_gettop(L);
    std::string dir;
    int target_w = 64;
    int target_h = 64;
    int min_modalities = 1;

    if (top >= 1 && lua_isstring(L, 1)) dir = lua_tostring(L, 1);
    if (top >= 2 && lua_isinteger(L, 2)) target_w = (int)lua_tointeger(L, 2);
    if (top >= 3 && lua_isinteger(L, 3)) target_h = (int)lua_tointeger(L, 3);
    if (top >= 4 && lua_isinteger(L, 4)) min_modalities = (int)lua_tointeger(L, 4);

    lua_newtable(L); // loader

    if (!dir.empty()) {
        lua_pushstring(L, dir.c_str());
        lua_setfield(L, -2, "dir");
    }
    lua_pushinteger(L, target_w);
    lua_setfield(L, -2, "target_w");
    lua_pushinteger(L, target_h);
    lua_setfield(L, -2, "target_h");
    lua_pushinteger(L, min_modalities);
    lua_setfield(L, -2, "min_modalities");

    // cache() method: closure captures the loader table as upvalue.
    lua_pushvalue(L, -1);
    lua_pushcclosure(L, lua_databaseLoad_cache, 1);
    lua_setfield(L, -2, "cache");

    return 1;
}

int LuaScripting::lua_databaseLoad_cache(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    // Upvalue: loader table
    lua_pushvalue(L, lua_upvalueindex(1));
    const int loader_idx = lua_gettop(L);

    // Allow both dot-call and colon-call.
    int argi = 1;
    if (lua_gettop(L) >= 1 && lua_istable(L, 1)) {
        // colon call: self at arg1
        argi = 2;
    }

    auto get_loader_string = [&](const char* key) -> std::string {
        lua_getfield(L, loader_idx, key);
        std::string v;
        if (lua_isstring(L, -1)) v = lua_tostring(L, -1);
        lua_pop(L, 1);
        return v;
    };
    auto get_loader_int = [&](const char* key, int def) -> int {
        lua_getfield(L, loader_idx, key);
        int v = def;
        if (lua_isinteger(L, -1)) v = (int)lua_tointeger(L, -1);
        lua_pop(L, 1);
        return v;
    };

    std::string dataset_dir = get_loader_string("dir");
    int target_w = get_loader_int("target_w", 64);
    int target_h = get_loader_int("target_h", 64);
    int min_modalities = get_loader_int("min_modalities", 1);

    // Overrides from args (optional)
    // Signature (flexible):
    //   cache([dir], [target_w], [target_h], [min_modalities], [cache_path], [max_ram_mb], [lazy_loading])
    if (lua_gettop(L) >= argi && lua_isstring(L, argi)) {
        dataset_dir = lua_tostring(L, argi);
        argi++;
    }
    if (lua_gettop(L) >= argi && lua_isinteger(L, argi)) {
        target_w = (int)lua_tointeger(L, argi);
        argi++;
    }
    if (lua_gettop(L) >= argi && lua_isinteger(L, argi)) {
        target_h = (int)lua_tointeger(L, argi);
        argi++;
    }
    if (lua_gettop(L) >= argi && lua_isinteger(L, argi)) {
        min_modalities = (int)lua_tointeger(L, argi);
        argi++;
    }

    std::string cache_path = "dataset_cache.json";
    if (lua_gettop(L) >= argi && lua_isstring(L, argi)) {
        cache_path = lua_tostring(L, argi);
        argi++;
    }

    int max_ram_mb = 10240;
    if (lua_gettop(L) >= argi && lua_isinteger(L, argi)) {
        max_ram_mb = (int)lua_tointeger(L, argi);
        argi++;
    }

    bool lazy_loading = true;
    if (lua_gettop(L) >= argi && lua_isboolean(L, argi)) {
        lazy_loading = lua_toboolean(L, argi);
        argi++;
    }

    // Fallback: if no dir was provided, try ctx.currentConfig.dataset.dir
    if (dataset_dir.empty()) {
        try {
            if (ctx.currentConfig.contains("dataset") && ctx.currentConfig["dataset"].contains("dir")) {
                dataset_dir = ctx.currentConfig["dataset"]["dir"].get<std::string>();
            }
        } catch (...) {
        }
    }

    if (dataset_dir.empty()) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Database.load().cache(): dataset_dir manquant (passez un chemin ou chargez un dataset d'abord)");
        return 2;
    }

    try {
        fs::path dataset_path(dataset_dir);
        if (!fs::exists(dataset_path)) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Le dossier dataset n'existe pas");
            return 2;
        }

        ctx.addLog("Chargement dataset (cached): " + dataset_dir);

        std::vector<DatasetItem> items = loadDatasetCached(
            dataset_dir,
            target_w,
            target_h,
            min_modalities,
            cache_path,
            (size_t)std::max(0, max_ram_mb),
            lazy_loading
        );

        if (items.empty()) {
            ctx.addLog("⚠️  Attention: Dataset vide (cached)");
        } else {
            ctx.addLog("✓ " + std::to_string(items.size()) + " items chargés (cached)");
        }

        ctx.currentDataset = std::move(items);

        if (!ctx.currentConfig.contains("dataset")) {
            ctx.currentConfig["dataset"] = json::object();
        }
        ctx.currentConfig["dataset"]["dir"] = dataset_dir;
        ctx.currentConfig["dataset"]["target_w"] = target_w;
        ctx.currentConfig["dataset"]["target_h"] = target_h;
        ctx.currentConfig["dataset"]["min_modalities"] = min_modalities;
        ctx.currentConfig["dataset"]["num_items"] = ctx.currentDataset.size();
        ctx.currentConfig["dataset"]["cache_path"] = cache_path;
        ctx.currentConfig["dataset"]["max_ram_mb"] = max_ram_mb;
        ctx.currentConfig["dataset"]["lazy_loading"] = lazy_loading;
        ctx.currentConfig["dataset"]["loader"] = "cached";

        lua_pushboolean(L, true);
        lua_pushinteger(L, ctx.currentDataset.size());
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    } catch (...) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "unknown error");
        return 2;
    }
}

int LuaScripting::lua_loadDataset(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    const char* dataset_dir = luaL_checkstring(L, 1);

    // Options:
    //  (dir, target_w, target_h, min_modalities, use_cache, cache_path, max_ram_mb, lazy_loading)
    const int top = lua_gettop(L);
    int target_w = (top >= 2 && lua_isinteger(L, 2)) ? (int)lua_tointeger(L, 2) : 64;
    int target_h = (top >= 3 && lua_isinteger(L, 3)) ? (int)lua_tointeger(L, 3) : 64;
    int min_modalities = (top >= 4 && lua_isinteger(L, 4)) ? (int)lua_tointeger(L, 4) : 1;

    const bool use_cache = (top >= 5 && lua_isboolean(L, 5)) ? (bool)lua_toboolean(L, 5) : false;
    std::string cache_path = (top >= 6 && lua_isstring(L, 6)) ? std::string(lua_tostring(L, 6)) : std::string("dataset_cache.json");
    int max_ram_mb = (top >= 7 && lua_isinteger(L, 7)) ? (int)lua_tointeger(L, 7) : 10240;
    bool lazy_loading = (top >= 8 && lua_isboolean(L, 8)) ? (bool)lua_toboolean(L, 8) : true;

    ctx.addLog(std::string("Chargement dataset") + (use_cache ? " (cached): " : ": ") + std::string(dataset_dir));
    
    try {
        fs::path dataset_path(dataset_dir);
        
        if (!fs::exists(dataset_path)) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Le dossier dataset n'existe pas");
            return 2;
        }
        
        // Charger les items du dataset
        std::vector<DatasetItem> items;
        if (use_cache) {
            items = loadDatasetCached(
                dataset_dir,
                target_w,
                target_h,
                min_modalities,
                cache_path,
                (size_t)std::max(0, max_ram_mb),
                lazy_loading
            );
        } else {
            items = loadDataset(dataset_dir, target_w, target_h, min_modalities);
        }
        
        if (items.empty()) {
            ctx.addLog(std::string("⚠️  Attention: Dataset vide") + (use_cache ? " (cached)" : ""));
        } else {
            ctx.addLog("✓ " + std::to_string(items.size()) + " items chargés" + (use_cache ? " (cached)" : ""));
        }
        
        // Stocker le dataset dans le contexte
        ctx.currentDataset = std::move(items);
        
        if (!ctx.currentConfig.contains("dataset")) {
            ctx.currentConfig["dataset"] = json::object();
        }
        ctx.currentConfig["dataset"]["dir"] = dataset_dir;
        ctx.currentConfig["dataset"]["target_w"] = target_w;
        ctx.currentConfig["dataset"]["target_h"] = target_h;
        ctx.currentConfig["dataset"]["min_modalities"] = min_modalities;
        ctx.currentConfig["dataset"]["num_items"] = ctx.currentDataset.size();

        ctx.currentConfig["dataset"]["use_cache"] = use_cache;
        if (use_cache) {
            ctx.currentConfig["dataset"]["cache_path"] = cache_path;
            ctx.currentConfig["dataset"]["max_ram_mb"] = max_ram_mb;
            ctx.currentConfig["dataset"]["lazy_loading"] = lazy_loading;
            ctx.currentConfig["dataset"]["loader"] = "cached";
        } else {
            if (ctx.currentConfig["dataset"].contains("cache_path")) ctx.currentConfig["dataset"].erase("cache_path");
            if (ctx.currentConfig["dataset"].contains("max_ram_mb")) ctx.currentConfig["dataset"].erase("max_ram_mb");
            if (ctx.currentConfig["dataset"].contains("lazy_loading")) ctx.currentConfig["dataset"].erase("lazy_loading");
            ctx.currentConfig["dataset"]["loader"] = "default";
        }
        
        lua_pushboolean(L, true);
        lua_pushinteger(L, ctx.currentDataset.size());
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_getDataset(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    int index = luaL_checkinteger(L, 1);
    
    if (ctx.currentDataset.empty()) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun dataset chargé. Utilisez dataset.load() d'abord.");
        return 2;
    }
    
    if (index < 1 || index > (int)ctx.currentDataset.size()) {
        lua_pushnil(L);
        lua_pushstring(L, "Index hors limites");
        return 2;
    }
    
    try {
        auto& item = ctx.currentDataset[index - 1]; // Lua est 1-indexed
        
        // Créer une table Lua avec les informations de l'item
        lua_newtable(L);
        
        // Ajouter les chemins de fichiers
        if (!item.text_file.empty()) {
            lua_pushstring(L, item.text_file.c_str());
            lua_setfield(L, -2, "text_file");
        }
        if (!item.image_file.empty()) {
            lua_pushstring(L, item.image_file.c_str());
            lua_setfield(L, -2, "image_file");
        }
        if (!item.audio_file.empty()) {
            lua_pushstring(L, item.audio_file.c_str());
            lua_setfield(L, -2, "audio_file");
        }
        if (!item.video_file.empty()) {
            lua_pushstring(L, item.video_file.c_str());
            lua_setfield(L, -2, "video_file");
        }
        
        // Déduire les dimensions cible du dataset (si connues)
        int target_w = 64;
        int target_h = 64;
        try {
            if (ctx.currentConfig.contains("dataset")) {
                if (ctx.currentConfig["dataset"].contains("target_w")) target_w = ctx.currentConfig["dataset"]["target_w"].get<int>();
                if (ctx.currentConfig["dataset"].contains("target_h")) target_h = ctx.currentConfig["dataset"]["target_h"].get<int>();
            }
        } catch (...) {
        }
        
        // Charger et ajouter le contenu texte si présent
        if (!item.text_file.empty() && !item.text.has_value()) {
            item.loadText();
        }
        if (!item.text_file.empty() && item.text.has_value()) {
            lua_pushstring(L, item.text.value().c_str());
            lua_setfield(L, -2, "text");
        }

        // Charger et retourner l'image en bytes RGB u8 (utile pour diffusion).
        if (!item.image_file.empty()) {
            item.loadImageRGB(target_w, target_h);
            if (item.img_loaded) {
                lua_newtable(L);
                for (size_t i = 0; i < item.img.size(); ++i) {
                    lua_pushinteger(L, (lua_Integer)item.img[i]);
                    lua_rawseti(L, -2, (int)i + 1);
                }
                lua_setfield(L, -2, "image");
            }

            lua_pushinteger(L, item.w > 0 ? item.w : target_w);
            lua_setfield(L, -2, "width");
            lua_pushinteger(L, item.h > 0 ? item.h : target_h);
            lua_setfield(L, -2, "height");
            lua_pushinteger(L, item.img_c > 0 ? item.img_c : 3);
            lua_setfield(L, -2, "channels");
        }
        
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_readImageRGBU8(lua_State* L) {
    // Args:
    // 1 path (string)
    // 2 target_w (int, optional, default 256)
    // 3 target_h (int, optional, default 256)
    const char* path = luaL_checkstring(L, 1);
    const int top = lua_gettop(L);
    const int target_w = (top >= 2 && lua_isinteger(L, 2)) ? (int)lua_tointeger(L, 2) : 256;
    const int target_h = (top >= 3 && lua_isinteger(L, 3)) ? (int)lua_tointeger(L, 3) : 256;

    if (!path || std::string(path).empty()) {
        lua_pushnil(L);
        lua_pushstring(L, "Chemin image vide");
        return 2;
    }
    if (target_w <= 0 || target_h <= 0) {
        lua_pushnil(L);
        lua_pushstring(L, "Dimensions invalides");
        return 2;
    }

    try {
        int w_img = 0, h_img = 0, c = 0;
        unsigned char* data = stbi_load(path, &w_img, &h_img, &c, 3);
        if (!data) {
            lua_pushnil(L);
            lua_pushfstring(L, "Impossible de charger l'image: %s", path);
            return 2;
        }

        std::vector<unsigned char> src((size_t)w_img * (size_t)h_img * 3);
        std::memcpy(src.data(), data, src.size());
        stbi_image_free(data);

        std::vector<uint8_t> dst((size_t)target_w * (size_t)target_h * 3);
        resizeBicubicRGB_SRGBLinear(src.data(), w_img, h_img, dst.data(), target_w, target_h);

        lua_newtable(L);

        lua_newtable(L);
        for (size_t i = 0; i < dst.size(); ++i) {
            lua_pushinteger(L, (lua_Integer)dst[i]);
            lua_rawseti(L, -2, (int)i + 1);
        }
        lua_setfield(L, -2, "image");

        lua_pushinteger(L, target_w);
        lua_setfield(L, -2, "width");
        lua_pushinteger(L, target_h);
        lua_setfield(L, -2, "height");
        lua_pushinteger(L, 3);
        lua_setfield(L, -2, "channels");

        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    } catch (...) {
        lua_pushnil(L);
        lua_pushstring(L, "unknown error");
        return 2;
    }
}

// ============================================================================
// PonyXL helpers
// ============================================================================

int LuaScripting::lua_ponyxlDdpmTrainStep(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    static int s_ponyxl_global_step = 0;

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    // Args:
    // 1 prompt (string)
    // 2 image (table of bytes u8)
    // 3 w (int)
    // 4 h (int)
    // 5 lr (number)
    // 6 optimizer (string, optional): "adamw" | "adam" | "sgd"
    // 7 meta (table, optional): {epoch,total_epochs,batch,total_batches,avg_loss,dataset_i}
    const char* prompt = luaL_checkstring(L, 1);
    luaL_checktype(L, 2, LUA_TTABLE);
    const int w = (int)luaL_checkinteger(L, 3);
    const int h = (int)luaL_checkinteger(L, 4);
    const float lr = (float)luaL_checknumber(L, 5);
    const char* opt_type = luaL_optstring(L, 6, "adamw");

    int meta_epoch = 0;
    int meta_total_epochs = 0;
    int meta_batch = 0;
    int meta_total_batches = 0;
    float meta_avg_loss = 0.0f;
    int meta_dataset_i = 0;

    const int nargs = lua_gettop(L);
    if (nargs >= 7 && lua_istable(L, 7)) {
        lua_getfield(L, 7, "epoch");
        if (lua_isnumber(L, -1)) meta_epoch = (int)lua_tointeger(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, 7, "total_epochs");
        if (lua_isnumber(L, -1)) meta_total_epochs = (int)lua_tointeger(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, 7, "batch");
        if (lua_isnumber(L, -1)) meta_batch = (int)lua_tointeger(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, 7, "total_batches");
        if (lua_isnumber(L, -1)) meta_total_batches = (int)lua_tointeger(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, 7, "avg_loss");
        if (lua_isnumber(L, -1)) meta_avg_loss = (float)lua_tonumber(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, 7, "dataset_i");
        if (lua_isnumber(L, -1)) meta_dataset_i = (int)lua_tointeger(L, -1);
        lua_pop(L, 1);
    }

    const size_t n = (size_t)lua_rawlen(L, 2);
    if (n == 0) {
        lua_pushnil(L);
        lua_pushstring(L, "Image vide");
        return 2;
    }

    std::vector<uint8_t> rgb;
    rgb.resize(n);
    for (size_t i = 0; i < n; ++i) {
        lua_rawgeti(L, 2, (lua_Integer)i + 1);
        int v = 0;
        if (lua_isinteger(L, -1)) v = (int)lua_tointeger(L, -1);
        else if (lua_isnumber(L, -1)) v = (int)lua_tonumber(L, -1);
        lua_pop(L, 1);
        v = std::clamp(v, 0, 255);
        rgb[i] = (uint8_t)v;
    }

    // Validation stricte: le modèle attend RGB packed (w*h*3)
    if (w <= 0 || h <= 0) {
        lua_pushnil(L);
        lua_pushstring(L, "w/h invalides (doivent être > 0)");
        return 2;
    }
    const size_t expected = (size_t)w * (size_t)h * 3ULL;
    if (rgb.size() != expected) {
        const std::string msg = "Taille image invalide: got=" + std::to_string(rgb.size()) +
                                " expected=" + std::to_string(expected);
        lua_pushnil(L);
        lua_pushstring(L, msg.c_str());
        return 2;
    }

    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Le modèle courant n'est pas un PonyXLDDPMModel (type=ponyxl_ddpm attendu)");
            return 2;
        }

        // Optimizer persistant: stocké dans le modèle pour compat avec save/load.
        if (!ctx.currentModel->getSerializedOptimizer()) {
            Optimizer opt;
            opt.initial_lr = lr;

            std::string t = opt_type ? std::string(opt_type) : std::string("adamw");
            std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            if (t == "sgd") opt.type = OptimizerType::SGD;
            else if (t == "adam") opt.type = OptimizerType::ADAM;
            else opt.type = OptimizerType::ADAMW;

            // Paramètres depuis modelConfig si présents
            try {
                if (ctx.modelConfig.contains("beta1")) opt.beta1 = ctx.modelConfig["beta1"].get<float>();
                if (ctx.modelConfig.contains("beta2")) opt.beta2 = ctx.modelConfig["beta2"].get<float>();
                if (ctx.modelConfig.contains("epsilon")) opt.eps = ctx.modelConfig["epsilon"].get<float>();
                if (ctx.modelConfig.contains("weight_decay")) opt.weight_decay = ctx.modelConfig["weight_decay"].get<float>();
            } catch (...) {
            }

            ctx.currentModel->setSerializedOptimizer(opt);
        } else {
            // S'assurer que lr reflète l'argument
            if (Optimizer* saved = ctx.currentModel->getMutableSerializedOptimizer()) {
                saved->initial_lr = lr;
            }
        }

        Optimizer* opt = ctx.currentModel->getMutableSerializedOptimizer();
        if (!opt) {
            throw std::runtime_error("optimizer state unavailable");
        }

        // If Viz is active, ensure taps are enabled before running the step so the model
        // can emit custom previews (noise/denoise, recon, etc.) during the step.
        const bool viz_active = (ctx.asyncMonitor && ctx.asyncMonitor->getViz() != nullptr);
        if (viz_active && ctx.currentModel) {
            if (!ctx.currentModel->isVizTapsEnabled()) {
                ctx.currentModel->setVizTapsEnabled(true);
            }
            // Optional per-model limits from config.json / modelConfig.
            try {
                int max_frames = 12;
                int max_side = 64;
                if (ctx.modelConfig.contains("viz_taps_max_frames")) max_frames = ctx.modelConfig["viz_taps_max_frames"].get<int>();
                if (ctx.modelConfig.contains("viz_taps_max_side")) max_side = ctx.modelConfig["viz_taps_max_side"].get<int>();
                ctx.currentModel->setVizTapsLimits(max_frames, max_side);
            } catch (...) {
            }
        }

        const PonyXLDDPMModel::StepStats st = pony->trainStepSdxlLatentDiffusion(prompt, rgb, w, h, *opt, lr);

        // Best-effort: si AsyncMonitor/Viz est actif, pousser l'input et les métriques.
        if (ctx.asyncMonitor) {
            s_ponyxl_global_step += 1;

            static std::chrono::steady_clock::time_point s_last_pony_metrics_ts;
            static bool s_has_last_pony_metrics_ts = false;

            AsyncMonitor::Metrics m;
            m.epoch = meta_epoch;
            m.total_epochs = meta_total_epochs;
            m.batch = (meta_batch > 0) ? meta_batch : s_ponyxl_global_step;
            m.total_batches = meta_total_batches;
            m.loss = st.loss;
            m.avg_loss = (meta_avg_loss > 0.0f) ? meta_avg_loss : st.loss;
            m.lr = lr;
            m.mse = st.loss;
            m.kl = st.kl_divergence;
            m.wass = st.wasserstein;
            m.ent = st.entropy_diff;
            m.mom = st.moment_mismatch;
            m.spat = st.spatial_coherence;
            m.temp = st.temporal_consistency;
            m.timestep = st.timestep;
            m.grad_norm = st.grad_norm;
            m.grad_max = st.grad_max_abs;
            m.kl_beta_effective = st.kl_beta_effective;
            m.params = ctx.currentModel ? ctx.currentModel->totalParamCount() : 0;
            m.opt_type = (int)opt->type;
            m.opt_step = opt->step;
            m.opt_beta1 = opt->beta1;
            m.opt_beta2 = opt->beta2;
            m.opt_eps = opt->eps;
            m.opt_weight_decay = opt->weight_decay;

            {
                auto& guard = MemoryGuard::instance();
                m.memory_mb = guard.getCurrentBytes() / 1024 / 1024;
            }

            {
                const auto now = std::chrono::steady_clock::now();
                if (s_has_last_pony_metrics_ts) {
                    const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - s_last_pony_metrics_ts).count();
                    if (dt_ms > 0 && dt_ms < 60LL * 60LL * 1000LL) {
                        m.batch_time_ms = (int)dt_ms;
                        m.bps = 1000.0f / (float)m.batch_time_ms;
                    }
                }
                s_last_pony_metrics_ts = now;
                s_has_last_pony_metrics_ts = true;
            }
            ctx.asyncMonitor->updateMetrics(m);

            if (ctx.asyncMonitor->getViz() != nullptr) {
                const bool viz_resync = [&]() -> bool {
                    try {
                        auto viz = ctx.asyncMonitor->getViz();
                        if (!viz) return false;
                        return viz->consumeResyncRequested();
                    } catch (...) {
                        return false;
                    }
                }();

                // Afficher l'input dataset (image + prompt) dans la Viz.
                std::string label = "ponyxl_ddpm/input/dataset/rgb";
                if (meta_dataset_i > 0) {
                    label += "/i=" + std::to_string(meta_dataset_i);
                }
                ctx.asyncMonitor->setDatasetSample(
                    rgb,
                    w,
                    h,
                    3,
                    label,
                    std::string(prompt ? prompt : ""),
                    std::string(),
                    std::string()
                );

                // Pousser les viz taps (frames) générés par le modèle pendant la step.
                // Sans ça, les recon/noise previews ajoutés via Model::addVizTapFrame ne s'affichent jamais.
                if (ctx.currentModel) {
                    auto taps = ctx.currentModel->consumeVizTaps();
                    std::vector<Visualizer::BlockFrame> frames;
                    frames.reserve(taps.size());
                    for (auto& f : taps) {
                        Visualizer::BlockFrame bf;
                        bf.pixels = std::move(f.pixels);
                        bf.w = f.w;
                        bf.h = f.h;
                        bf.channels = f.channels;
                        bf.label = std::move(f.label);
                        frames.push_back(std::move(bf));
                    }
                    // Important UX: si le dataset change mais que le modèle n'a pas émis de taps,
                    // ne pas conserver les frames précédentes (sinon rendu stale/mélangé).
                    ctx.asyncMonitor->setLayerBlockImages(frames);
                }

                // Si l'utilisateur a demandé une resynchronisation (touche R), forcer une preview
                // de reconstruction (best-effort) afin de rafraîchir la vue avec l'état courant.
                if (viz_resync) {
                    try {
                        // seed/t choisis pour être stables et peu coûteux à interpréter.
                        const int seed = 12345;
                        const int ddpm_step = -1;
                        auto prev = pony->reconstructPreviewSdxlLatentDiffusion(
                            std::string(prompt ? prompt : ""),
                            rgb,
                            w,
                            h,
                            256,
                            seed,
                            ddpm_step
                        );
                        if (!prev.pixels.empty() && prev.w > 0 && prev.h > 0) {
                            std::string tag = "RESYNC recon";
                            if (meta_dataset_i > 0) tag += " i=" + std::to_string(meta_dataset_i);
                            ctx.asyncMonitor->addImage(prev.pixels, prev.w, prev.h, prev.channels, tag);
                        }
                    } catch (...) {
                        // ignore
                    }
                }
            }
        }

        lua_newtable(L);
        lua_pushnumber(L, st.loss);
        lua_setfield(L, -2, "loss");
        lua_pushnumber(L, st.grad_norm);
        lua_setfield(L, -2, "grad_norm");
        lua_pushnumber(L, st.grad_max_abs);
        lua_setfield(L, -2, "grad_max_abs");
        lua_pushnumber(L, st.timestep);
        lua_setfield(L, -2, "timestep");
        lua_pushnumber(L, st.kl_divergence);
        lua_setfield(L, -2, "kl_divergence");
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_ponyxlDdpmValidateStep(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    // Args:
    // 1 prompt (string)
    // 2 wrong_prompt (string, optional)
    // 3 image (table of bytes u8)
    // 4 w (int)
    // 5 h (int)
    // 6 seed (int, optional) : graine pour eps (validation déterministe)
    // 7 ddpm_step (int, optional) : timestep t dans [0..T-1], -1 => T/2
    const char* prompt = luaL_checkstring(L, 1);
    const char* wrong_prompt = luaL_optstring(L, 2, "");
    luaL_checktype(L, 3, LUA_TTABLE);
    const int w = (int)luaL_checkinteger(L, 4);
    const int h = (int)luaL_checkinteger(L, 5);
    const int seed = (int)luaL_optinteger(L, 6, 12345);
    const int ddpm_step = (int)luaL_optinteger(L, 7, -1);

    const size_t n = (size_t)lua_rawlen(L, 3);
    if (n == 0) {
        lua_pushnil(L);
        lua_pushstring(L, "Image vide");
        return 2;
    }

    std::vector<uint8_t> rgb;
    rgb.resize(n);
    for (size_t i = 0; i < n; ++i) {
        lua_rawgeti(L, 3, (lua_Integer)i + 1);
        int v = 0;
        if (lua_isinteger(L, -1)) v = (int)lua_tointeger(L, -1);
        else if (lua_isnumber(L, -1)) v = (int)lua_tonumber(L, -1);
        lua_pop(L, 1);
        v = std::clamp(v, 0, 255);
        rgb[i] = (uint8_t)v;
    }

    if (w <= 0 || h <= 0) {
        lua_pushnil(L);
        lua_pushstring(L, "w/h invalides (doivent être > 0)");
        return 2;
    }
    const size_t expected = (size_t)w * (size_t)h * 3ULL;
    if (rgb.size() != expected) {
        const std::string msg = "Taille image invalide: got=" + std::to_string(rgb.size()) +
                                " expected=" + std::to_string(expected);
        lua_pushnil(L);
        lua_pushstring(L, msg.c_str());
        return 2;
    }

    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Le modèle courant n'est pas un PonyXLDDPMModel (type=ponyxl_ddpm attendu)");
            return 2;
        }

        const PonyXLDDPMModel::ValStats st = pony->validateStepSdxlLatentDiffusion(
            prompt ? std::string(prompt) : std::string(),
            wrong_prompt ? std::string(wrong_prompt) : std::string(),
            rgb,
            w,
            h,
            seed,
            ddpm_step
        );

        lua_newtable(L);
        lua_pushnumber(L, (lua_Number)st.eps_mse);
        lua_setfield(L, -2, "eps_mse");
        lua_pushnumber(L, (lua_Number)st.x0_mse);
        lua_setfield(L, -2, "x0_mse");
        lua_pushnumber(L, (lua_Number)st.img_mse);
        lua_setfield(L, -2, "img_mse");
        lua_pushnumber(L, (lua_Number)st.eps_mse_wrong);
        lua_setfield(L, -2, "eps_mse_wrong");
        lua_pushnumber(L, (lua_Number)st.assoc_margin);
        lua_setfield(L, -2, "assoc_margin");
        lua_pushnumber(L, (lua_Number)st.t_norm);
        lua_setfield(L, -2, "t_norm");
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_ponyxlDdpmSetVaeScale(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    const float s = (float)luaL_checknumber(L, 1);
    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Modèle courant n'est pas ponyxl_ddpm");
            return 2;
        }
        pony->setVaeScale(s);
        lua_pushboolean(L, 1);
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_ponyxlDdpmGetVaeScale(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Modèle courant n'est pas ponyxl_ddpm");
            return 2;
        }
        lua_pushnumber(L, (lua_Number)pony->getConfig().vae_scale);
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_ponyxlDdpmVaeMuMoments(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    // Args:
    // 1 image (table of bytes u8)
    // 2 w (int)
    // 3 h (int)
    luaL_checktype(L, 1, LUA_TTABLE);
    const int w = (int)luaL_checkinteger(L, 2);
    const int h = (int)luaL_checkinteger(L, 3);
    const size_t nbytes = (size_t)lua_rawlen(L, 1);
    if (nbytes == 0) {
        lua_pushnil(L);
        lua_pushstring(L, "Image vide");
        return 2;
    }

    std::vector<uint8_t> rgb;
    rgb.resize(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
        lua_rawgeti(L, 1, (lua_Integer)(i + 1));
        const int v = (int)luaL_checkinteger(L, -1);
        lua_pop(L, 1);
        rgb[i] = (uint8_t)std::clamp(v, 0, 255);
    }

    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Modèle courant n'est pas ponyxl_ddpm");
            return 2;
        }

        double sum = 0.0;
        double sumsq = 0.0;
        size_t n = 0;
        pony->accumulateVaeMuMoments(rgb, w, h, sum, sumsq, n);

        lua_newtable(L);
        lua_pushnumber(L, (lua_Number)sum);
        lua_setfield(L, -2, "sum");
        lua_pushnumber(L, (lua_Number)sumsq);
        lua_setfield(L, -2, "sumsq");
        lua_pushinteger(L, (lua_Integer)n);
        lua_setfield(L, -2, "n");
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_ponyxlDdpmVizReconstructStep(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    if (!ctx.asyncMonitor || ctx.asyncMonitor->getViz() == nullptr) {
        lua_pushnil(L);
        lua_pushstring(L, "Viz non actif (démarrer avec --viz)");
        return 2;
    }

    // Args:
    // 1 prompt (string)
    // 2 image (table of bytes u8)
    // 3 w (int)
    // 4 h (int)
    // 5 label (string, optional)
    // 6 max_side (int, optional)
    // 7 seed (int, optional)
    // 8 ddpm_step (int, optional)
    const char* prompt = luaL_checkstring(L, 1);
    luaL_checktype(L, 2, LUA_TTABLE);
    const int w = (int)luaL_checkinteger(L, 3);
    const int h = (int)luaL_checkinteger(L, 4);
    const char* label = luaL_optstring(L, 5, "VAL recon");
    const int max_side = (int)luaL_optinteger(L, 6, 256);
    const int seed = (int)luaL_optinteger(L, 7, 12345);
    const int ddpm_step = (int)luaL_optinteger(L, 8, -1);

    const size_t n = (size_t)lua_rawlen(L, 2);
    if (n == 0) {
        lua_pushnil(L);
        lua_pushstring(L, "Image vide");
        return 2;
    }

    std::vector<uint8_t> rgb;
    rgb.resize(n);
    for (size_t i = 0; i < n; ++i) {
        lua_rawgeti(L, 2, (lua_Integer)i + 1);
        int v = 0;
        if (lua_isinteger(L, -1)) v = (int)lua_tointeger(L, -1);
        else if (lua_isnumber(L, -1)) v = (int)lua_tonumber(L, -1);
        lua_pop(L, 1);
        v = std::clamp(v, 0, 255);
        rgb[i] = (uint8_t)v;
    }

    if (w <= 0 || h <= 0) {
        lua_pushnil(L);
        lua_pushstring(L, "w/h invalides (doivent être > 0)");
        return 2;
    }
    const size_t expected = (size_t)w * (size_t)h * 3ULL;
    if (rgb.size() != expected) {
        const std::string msg = "Taille image invalide: got=" + std::to_string(rgb.size()) +
                                " expected=" + std::to_string(expected);
        lua_pushnil(L);
        lua_pushstring(L, msg.c_str());
        return 2;
    }

    try {
        auto* pony = dynamic_cast<PonyXLDDPMModel*>(ctx.currentModel.get());
        if (!pony) {
            lua_pushnil(L);
            lua_pushstring(L, "Le modèle courant n'est pas un PonyXLDDPMModel (type=ponyxl_ddpm attendu)");
            return 2;
        }

        const PonyXLDDPMModel::ReconPreview rp = pony->reconstructPreviewSdxlLatentDiffusion(
            prompt ? std::string(prompt) : std::string(),
            rgb,
            w,
            h,
            max_side,
            seed,
            ddpm_step
        );

        if (rp.pixels.empty() || rp.w <= 0 || rp.h <= 0) {
            lua_pushnil(L);
            lua_pushstring(L, "Reconstruction preview vide");
            return 2;
        }

        ctx.asyncMonitor->addImage(rp.pixels, rp.w, rp.h, rp.channels, std::string(label ? label : "VAL recon"));
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_prepareSequences(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    int seq_length = luaL_checkinteger(L, 1);
    
    ctx.addLog("Préparation séquences (longueur=" + std::to_string(seq_length) + ")");
    
    try {
        if (!ctx.currentConfig.contains("dataset") || 
            !ctx.currentConfig["dataset"].contains("dir")) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Aucun dataset chargé. Utilisez dataset.load() d'abord.");
            return 2;
        }
        
        std::string dataset_dir = ctx.currentConfig["dataset"]["dir"];
        std::vector<DatasetItem> items = loadDataset(dataset_dir);
        
        // Créer des séquences à partir des items
        ctx.currentSequences.clear();
        
        for (auto& item : items) {
            // Charger le texte si nécessaire (lazy loading)
            if (!item.text_file.empty() && !item.text.has_value()) {
                item.loadText();
            }
            
            if (item.text.has_value() && !item.text.value().empty()) {
                // Tokenize le texte
                std::vector<int> tokens;
                if (ctx.currentTokenizer) {
                    tokens = ctx.currentTokenizer->tokenize(item.text.value());
                    
                    // Padding/truncation à seq_length
                    if (tokens.size() < static_cast<size_t>(seq_length)) {
                        tokens.resize(seq_length, ctx.currentTokenizer->getPadId());
                    } else if (tokens.size() > static_cast<size_t>(seq_length)) {
                        tokens.resize(seq_length);
                    }
                    
                    ctx.currentSequences.push_back(tokens);
                }
            }
        }
        
        ctx.addLog("✓ " + std::to_string(ctx.currentSequences.size()) + " séquences préparées");
        
        lua_pushboolean(L, true);
        lua_pushinteger(L, ctx.currentSequences.size());
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

// ============================================================================
// Utilitaires
// ============================================================================

int LuaScripting::lua_print(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    const char* msg = luaL_checkstring(L, 1);
    ctx.addLog(msg);
    
    return 0;
}

int LuaScripting::lua_readJSON(lua_State* L) {
    const char* filepath = luaL_checkstring(L, 1);
    
    try {
        std::ifstream f(filepath);
        json j;
        f >> j;
        
        jsonToLuaTable(L, j);
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_writeJSON(lua_State* L) {
    const char* filepath = luaL_checkstring(L, 1);
    luaL_checktype(L, 2, LUA_TTABLE);
    
    try {
        json j = luaTableToJson(L, 2);
        
        std::ofstream f(filepath);
        f << j.dump(2);
        
        lua_pushboolean(L, true);
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

// ============================================================================
// Conversion Lua <-> JSON
// ============================================================================

json LuaScripting::luaTableToJson(lua_State* L, int index) {
    // IMPORTANT: si l'appelant passe un index relatif (ex: -1), il devient instable
    // dès qu'on push/pop sur la stack. On le convertit donc en index absolu.
    index = lua_absindex(L, index);

    json result;
    
    // Vérifier si c'est un array ou un objet
    bool is_array = true;
    int max_index = 0;
    
    lua_pushnil(L);
    while (lua_next(L, index) != 0) {
        if (!lua_isnumber(L, -2)) {
            is_array = false;
        } else {
            int idx = lua_tointeger(L, -2);
            if (idx > max_index) max_index = idx;
        }
        lua_pop(L, 1);
    }
    
    if (is_array) {
        result = json::array();
        for (int i = 1; i <= max_index; ++i) {
            lua_rawgeti(L, index, i);
            
            if (lua_isnil(L, -1)) {
                result.push_back(nullptr);
            } else if (lua_isboolean(L, -1)) {
                result.push_back(lua_toboolean(L, -1) != 0);
            } else if (lua_isnumber(L, -1)) {
                result.push_back(lua_tonumber(L, -1));
            } else if (lua_isstring(L, -1)) {
                result.push_back(lua_tostring(L, -1));
            } else if (lua_istable(L, -1)) {
                result.push_back(luaTableToJson(L, lua_gettop(L)));
            }
            
            lua_pop(L, 1);
        }
    } else {
        result = json::object();
        lua_pushnil(L);
        while (lua_next(L, index) != 0) {
            std::string key;
            if (lua_isstring(L, -2)) {
                key = lua_tostring(L, -2);
            } else if (lua_isnumber(L, -2)) {
                key = std::to_string(lua_tointeger(L, -2));
            }
            
            if (lua_isnil(L, -1)) {
                result[key] = nullptr;
            } else if (lua_isboolean(L, -1)) {
                result[key] = (lua_toboolean(L, -1) != 0);
            } else if (lua_isnumber(L, -1)) {
                result[key] = lua_tonumber(L, -1);
            } else if (lua_isstring(L, -1)) {
                result[key] = lua_tostring(L, -1);
            } else if (lua_istable(L, -1)) {
                result[key] = luaTableToJson(L, lua_gettop(L));
            }
            
            lua_pop(L, 1);
        }
    }
    
    return result;
}

void LuaScripting::jsonToLuaTable(lua_State* L, const json& j) {
    if (j.is_null()) {
        lua_pushnil(L);
    } else if (j.is_boolean()) {
        lua_pushboolean(L, j.get<bool>());
    } else if (j.is_number_integer()) {
        lua_pushinteger(L, j.get<int>());
    } else if (j.is_number_float()) {
        lua_pushnumber(L, j.get<double>());
    } else if (j.is_string()) {
        lua_pushstring(L, j.get<std::string>().c_str());
    } else if (j.is_array()) {
        lua_newtable(L);
        for (size_t i = 0; i < j.size(); ++i) {
            jsonToLuaTable(L, j[i]);
            lua_rawseti(L, -2, i + 1);
        }
    } else if (j.is_object()) {
        lua_newtable(L);
        for (auto it = j.begin(); it != j.end(); ++it) {
            lua_pushstring(L, it.key().c_str());
            jsonToLuaTable(L, it.value());
            lua_settable(L, -3);
        }
    }
}

// ============================================================================
// Nouvelles implémentations - Model API étendue
// ============================================================================

int LuaScripting::lua_allocateParams(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    try {
        ctx.currentModel->allocateParams();
        size_t count = ctx.currentModel->totalParamCount();
        ctx.addLog("Paramètres alloués: " + std::to_string(count));
        
        lua_pushboolean(L, true);
        lua_pushinteger(L, count);
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_initWeights(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    const char* method = luaL_optstring(L, 1, "he");
    unsigned int seed = luaL_optinteger(L, 2, 0);
    
    try {
        ctx.currentModel->initializeWeights(method, seed);
        ctx.addLog("Poids initialisés: méthode=" + std::string(method));
        
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_totalParams(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushinteger(L, 0);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentModel->totalParamCount());
    return 1;
}

int LuaScripting::lua_pushLayer(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    const char* name = luaL_checkstring(L, 1);
    const char* type = luaL_checkstring(L, 2);
    size_t params_count = luaL_checkinteger(L, 3);
    
    ctx.currentModel->push(name, type, params_count);
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_setLayerIO(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Arguments:
    // 1. layer_name (string)
    // 2. inputs (table of strings, peut être vide)
    // 3. output (string, optionnel, défaut = "x")
    
    const char* layer_name = luaL_checkstring(L, 1);
    
    Layer* layer = ctx.currentModel->getLayerByName(layer_name);
    if (!layer) {
        lua_pushboolean(L, false);
        lua_pushfstring(L, "Layer '%s' not found", layer_name);
        return 2;
    }
    
    // Lire la table d'inputs
    layer->inputs.clear();
    if (lua_istable(L, 2)) {
        lua_pushnil(L);  // Premier key
        while (lua_next(L, 2) != 0) {
            // key à -2, value à -1
            if (lua_isstring(L, -1)) {
                layer->inputs.push_back(lua_tostring(L, -1));
            }
            lua_pop(L, 1);  // Pop value, garde key pour next
        }
    }
    
    // Lire l'output (optionnel)
    if (lua_gettop(L) >= 3 && lua_isstring(L, 3)) {
        layer->output = lua_tostring(L, 3);
    }
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_forwardPass(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Argument 1: input
    // - soit un tableau (array) de floats/ints
    // - soit une table { __input__ = <array> }
    luaL_checktype(L, 1, LUA_TTABLE);
    
    // Argument 2 (optionnel): training (bool, défaut: true)
    bool training = true;
    if (lua_gettop(L) >= 2 && lua_isboolean(L, 2)) {
        training = lua_toboolean(L, 2);
    }

    auto maybe_push_viz_taps = [&]() {
        if (!ctx.asyncMonitor || !ctx.currentModel) return;
        if (ctx.asyncMonitor->getViz() == nullptr) return;

        // Auto-enable taps when viz is active.
        if (!ctx.currentModel->isVizTapsEnabled()) {
            ctx.currentModel->setVizTapsEnabled(true);
        }

        auto taps = ctx.currentModel->consumeVizTaps();
        if (taps.empty()) return;

        std::vector<Visualizer::BlockFrame> frames;
        frames.reserve(taps.size());
        for (auto& f : taps) {
            Visualizer::BlockFrame bf;
            bf.pixels = std::move(f.pixels);
            bf.w = f.w;
            bf.h = f.h;
            bf.channels = f.channels;
            bf.label = std::move(f.label);
            frames.push_back(std::move(bf));
        }
        ctx.asyncMonitor->setLayerBlockImages(frames);
    };
    
    // Résoudre le tableau "array" à lire.
    // Si arg1 est une map {__input__=...}, on lit arg1.__input__.
    int input_index = 1;
    bool pushed_subtable = false;
    size_t n = lua_rawlen(L, input_index);
    if (n == 0) {
        lua_getfield(L, 1, "__input__");
        if (lua_istable(L, -1)) {
            input_index = lua_gettop(L);
            pushed_subtable = true;
            n = lua_rawlen(L, input_index);
        } else {
            lua_pop(L, 1);
        }
    }

    if (n == 0) {
        lua_pushnil(L);
        lua_pushstring(L, "Model.forward: expected an array of numbers or a table {__input__=<array>}");
        return 2;
    }

    // Détecter int vs float (et préserver l'ordre 1..n)
    bool all_int = true;
    for (size_t i = 1; i <= n; ++i) {
        lua_rawgeti(L, input_index, static_cast<lua_Integer>(i));
        const bool is_int = lua_isinteger(L, -1);
        lua_pop(L, 1);
        if (!is_int) {
            all_int = false;
            break;
        }
    }

    if (all_int) {
        std::vector<int> input_ids;
        input_ids.reserve(n);
        for (size_t i = 1; i <= n; ++i) {
            lua_rawgeti(L, input_index, static_cast<lua_Integer>(i));
            lua_Integer v = lua_tointeger(L, -1);
            lua_pop(L, 1);
            input_ids.push_back(static_cast<int>(v));
        }
        if (pushed_subtable) {
            lua_pop(L, 1);
            pushed_subtable = false;
        }
        try {
            std::vector<float> output = ctx.currentModel->forwardPass(input_ids, training);

            maybe_push_viz_taps();

            lua_newtable(L);
            for (size_t i = 0; i < output.size(); ++i) {
                lua_pushnumber(L, output[i]);
                lua_rawseti(L, -2, i + 1);
            }
            return 1;
        } catch (const std::exception& e) {
            lua_pushnil(L);
            lua_pushstring(L, e.what());
            return 2;
        }
    }

    std::vector<float> input;
    input.reserve(n);
    for (size_t i = 1; i <= n; ++i) {
        lua_rawgeti(L, input_index, static_cast<lua_Integer>(i));
        input.push_back(static_cast<float>(lua_tonumber(L, -1)));
        lua_pop(L, 1);
    }
    if (pushed_subtable) {
        lua_pop(L, 1);
        pushed_subtable = false;
    }
    
    try {
        std::vector<float> output = ctx.currentModel->forwardPass(input, training);

        maybe_push_viz_taps();
        
        // Retourner output comme table
        lua_newtable(L);
        for (size_t i = 0; i < output.size(); ++i) {
            lua_pushnumber(L, output[i]);
            lua_rawseti(L, -2, i + 1);
        }
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_backwardPass(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    // Argument: loss_gradient (table de floats)
    luaL_checktype(L, 1, LUA_TTABLE);
    
    std::vector<float> loss_grad;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        loss_grad.push_back(lua_tonumber(L, -1));
        lua_pop(L, 1);
    }
    
    try {
        Gradients grads = ctx.currentModel->backwardPass(loss_grad);
        ctx.addLog("Backward pass complété");
        
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_zeroGradients(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    try {
        ctx.currentModel->zeroGradients();
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_getGradients(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    try {
        Gradients grads = ctx.currentModel->getGradients();
        
        // Retourner les gradients comme table (ordonnée par index)
        lua_newtable(L);
        size_t lua_idx = 1;
        
        // Parcourir tous les indices dans l'ordre
        for (const auto& [param_idx, grad_value] : grads.param_grads) {
            lua_pushnumber(L, grad_value);
            lua_rawseti(L, -2, lua_idx++);
        }
        
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_optimizerStep(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    double lr = luaL_checknumber(L, 1);
    const char* opt_type = luaL_optstring(L, 2, "adamw");
    
    try {
        Optimizer opt;
        opt.initial_lr = lr;
        
        if (std::string(opt_type) == "sgd") {
            opt.type = OptimizerType::SGD;
        } else if (std::string(opt_type) == "adam") {
            opt.type = OptimizerType::ADAM;
        } else {
            opt.type = OptimizerType::ADAMW;
        }
        
        ctx.currentModel->optimizerStep(opt, lr, nullptr);
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_setHardwareAccel(lua_State* L) {
    bool enable = lua_toboolean(L, 1);
    Model::setHardwareAcceleration(enable);
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_getHardwareCaps(lua_State* L) {
    lua_newtable(L);
    
    lua_pushboolean(L, Model::hasAVX2());
    lua_setfield(L, -2, "avx2");
    
    lua_pushboolean(L, Model::hasFMA());
    lua_setfield(L, -2, "fma");
    
    lua_pushboolean(L, Model::hasF16C());
    lua_setfield(L, -2, "f16c");
    
    lua_pushboolean(L, Model::hasBMI2());
    lua_setfield(L, -2, "bmi2");
    
    return 1;
}

// ============================================================================
// Layer Operations API (stubs - implémentation complète optionnelle)
// ============================================================================

int LuaScripting::lua_computeConv2D(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeLinear(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeMaxPool2D(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeAvgPool2D(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeActivation(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeBatchNorm(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeLayerNorm(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

int LuaScripting::lua_computeAttention(lua_State* L) {
    lua_pushboolean(L, false);
    lua_pushstring(L, "Non implémenté - utilisez model.forward() à la place");
    return 2;
}

// ============================================================================
// Tokenizer API étendue
// ============================================================================

int LuaScripting::lua_getVocabSize(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 0);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentTokenizer->getVocabSize());
    return 1;
}

int LuaScripting::lua_getMaxVocab(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 0);
        return 1;
    }
    lua_pushinteger(L, static_cast<lua_Integer>(ctx.currentTokenizer->getMaxVocab()));
    return 1;
}

int LuaScripting::lua_setMaxVocab(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    if (!ctx.currentTokenizer) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }

    const lua_Integer v = luaL_checkinteger(L, 1);
    const size_t new_max = static_cast<size_t>(std::max<lua_Integer>(0, v));
    ctx.currentTokenizer->setMaxVocab(new_max);
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_saveTokenizer(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* filepath = luaL_checkstring(L, 1);
    
    try {
        json j = ctx.currentTokenizer->to_json();
        std::ofstream f(filepath);
        f << j.dump(2);
        
        ctx.addLog("Tokenizer sauvegardé: " + std::string(filepath));
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_loadTokenizer(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    const char* filepath = luaL_checkstring(L, 1);
    
    try {
        std::ifstream f(filepath);
        json j;
        f >> j;
        
        ctx.currentTokenizer = std::make_shared<Tokenizer>();
        ctx.currentTokenizer->from_json(j);
        
        ctx.addLog("Tokenizer chargé: " + std::string(filepath));
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}
// Extension des méthodes Tokenizer pour l'API Lua
// À ajouter à la fin de LuaScripting.cpp avant le dernier }

// ============================================================================
// Tokenizer API - Méthodes étendues
// ============================================================================

int LuaScripting::lua_addToken(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, -1);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* token = luaL_checkstring(L, 1);
    int id = ctx.currentTokenizer->addToken(token);
    
    lua_pushinteger(L, id);
    return 1;
}

int LuaScripting::lua_ensureVocabFromText(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    ctx.currentTokenizer->ensureVocabFromText(text);
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_tokenizeEnsure(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    auto tokens = ctx.currentTokenizer->tokenizeEnsure(text);
    
    lua_newtable(L);
    for (size_t i = 0; i < tokens.size(); ++i) {
        lua_pushinteger(L, tokens[i]);
        lua_rawseti(L, -2, i + 1);
    }
    
    return 1;
}

int LuaScripting::lua_getPadId(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 0);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentTokenizer->getPadId());
    return 1;
}

int LuaScripting::lua_getUnkId(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 1);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentTokenizer->getUnkId());
    return 1;
}

int LuaScripting::lua_getSeqId(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 2);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentTokenizer->getSeqId());
    return 1;
}

int LuaScripting::lua_getModId(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 3);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentTokenizer->getModId());
    return 1;
}

int LuaScripting::lua_getMagId(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushinteger(L, 4);
        return 1;
    }
    
    lua_pushinteger(L, ctx.currentTokenizer->getMagId());
    return 1;
}

int LuaScripting::lua_getTokenById(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushstring(L, "");
        return 1;
    }
    
    int id = luaL_checkinteger(L, 1);
    std::string token = ctx.currentTokenizer->getTokenById(id);
    
    lua_pushstring(L, token.c_str());
    return 1;
}

int LuaScripting::lua_learnBPEFromCorpus(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    // Lire table de textes
    luaL_checktype(L, 1, LUA_TTABLE);
    int num_merges = luaL_optinteger(L, 2, 1000);
    
    std::vector<std::string> corpus;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        corpus.push_back(lua_tostring(L, -1));
        lua_pop(L, 1);
    }
    
    ctx.currentTokenizer->learnBPEFromCorpus(corpus, num_merges);
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_tokenizeBPE(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    auto tokens = ctx.currentTokenizer->tokenizeBPE(text);
    
    lua_newtable(L);
    for (size_t i = 0; i < tokens.size(); ++i) {
        lua_pushinteger(L, tokens[i]);
        lua_rawseti(L, -2, i + 1);
    }
    
    return 1;
}

int LuaScripting::lua_setMaxSequenceLength(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushboolean(L, false);
        return 1;
    }
    
    int max_len = luaL_checkinteger(L, 1);
    ctx.currentTokenizer->setMaxSequenceLength(max_len);
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_padSequence(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    // Lire table de tokens
    luaL_checktype(L, 1, LUA_TTABLE);
    int target_len = luaL_optinteger(L, 2, -1);
    
    std::vector<int> tokens;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        tokens.push_back(lua_tointeger(L, -1));
        lua_pop(L, 1);
    }
    
    auto padded = ctx.currentTokenizer->padSequence(tokens, target_len);
    
    lua_newtable(L);
    for (size_t i = 0; i < padded.size(); ++i) {
        lua_pushinteger(L, padded[i]);
        lua_rawseti(L, -2, i + 1);
    }
    
    return 1;
}

int LuaScripting::lua_batchTokenize(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    // Lire table de textes
    luaL_checktype(L, 1, LUA_TTABLE);
    int max_len = luaL_optinteger(L, 2, 512);
    
    std::vector<std::string> texts;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        texts.push_back(lua_tostring(L, -1));
        lua_pop(L, 1);
    }
    
    auto batch = ctx.currentTokenizer->batchTokenize(texts, max_len);
    
    // Retourner table de tables
    lua_newtable(L);
    for (size_t i = 0; i < batch.size(); ++i) {
        lua_newtable(L);
        for (size_t j = 0; j < batch[i].size(); ++j) {
            lua_pushinteger(L, batch[i][j]);
            lua_rawseti(L, -2, j + 1);
        }
        lua_rawseti(L, -2, i + 1);
    }
    
    return 1;
}

int LuaScripting::lua_printVocabStats(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushboolean(L, false);
        return 1;
    }
    
    ctx.currentTokenizer->printVocabStats();
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_getTokenFrequencies(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    auto freqs = ctx.currentTokenizer->getTokenFrequencies(text);
    
    // Retourner table Lua
    lua_newtable(L);
    for (const auto& pair : freqs) {
        lua_pushstring(L, pair.first.c_str());
        lua_pushinteger(L, pair.second);
        lua_settable(L, -3);
    }
    
    return 1;
}

int LuaScripting::lua_analyzeText(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    auto analysis = ctx.currentTokenizer->analyzeText(text);
    
    // Retourner table Lua avec analyse
    lua_newtable(L);
    
    // entities
    lua_newtable(L);
    for (size_t i = 0; i < analysis.entities.size(); ++i) {
        lua_pushstring(L, analysis.entities[i].c_str());
        lua_rawseti(L, -2, i + 1);
    }
    lua_setfield(L, -2, "entities");
    
    // modifiers
    lua_newtable(L);
    for (size_t i = 0; i < analysis.modifiers.size(); ++i) {
        lua_pushstring(L, analysis.modifiers[i].c_str());
        lua_rawseti(L, -2, i + 1);
    }
    lua_setfield(L, -2, "modifiers");
    
    // actions
    lua_newtable(L);
    for (size_t i = 0; i < analysis.actions.size(); ++i) {
        lua_pushstring(L, analysis.actions[i].c_str());
        lua_rawseti(L, -2, i + 1);
    }
    lua_setfield(L, -2, "actions");
    
    // main_subject
    lua_pushstring(L, analysis.mainSubject.c_str());
    lua_setfield(L, -2, "main_subject");
    
    // context
    lua_pushstring(L, analysis.context.c_str());
    lua_setfield(L, -2, "context");
    
    // complexity
    lua_pushinteger(L, analysis.complexity);
    lua_setfield(L, -2, "complexity");
    
    return 1;
}

// ============================================================================
// Memory Manager API
// ============================================================================

int LuaScripting::lua_memoryConfig(lua_State* L) {
    auto& mgr = AdvancedRAMManager::instance();
    
    // Argument: table de configuration
    if (!lua_istable(L, 1)) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Argument 1 doit être une table de configuration");
        return 2;
    }
    
    AdvancedRAMManager::Config config;
    
    // max_ram_gb (en Go)
    lua_getfield(L, 1, "max_ram_gb");
    if (lua_isnumber(L, -1)) {
        double gb = lua_tonumber(L, -1);
        config.max_ram_bytes = static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
    }
    lua_pop(L, 1);
    
    // enable_compression
    lua_getfield(L, 1, "enable_compression");
    if (lua_isboolean(L, -1)) {
        config.enable_compression = lua_toboolean(L, -1);
    }
    lua_pop(L, 1);
    
    // enable_async_loading
    lua_getfield(L, 1, "enable_async_loading");
    if (lua_isboolean(L, -1)) {
        config.enable_async_loading = lua_toboolean(L, -1);
    }
    lua_pop(L, 1);
    
    // enable_prediction
    lua_getfield(L, 1, "enable_prediction");
    if (lua_isboolean(L, -1)) {
        config.enable_prediction = lua_toboolean(L, -1);
    }
    lua_pop(L, 1);
    
    // enable_statistics
    lua_getfield(L, 1, "enable_statistics");
    if (lua_isboolean(L, -1)) {
        config.enable_statistics = lua_toboolean(L, -1);
    }
    lua_pop(L, 1);
    
    // preload_queue_size
    lua_getfield(L, 1, "preload_queue_size");
    if (lua_isnumber(L, -1)) {
        config.preload_queue_size = static_cast<size_t>(lua_tonumber(L, -1));
    }
    lua_pop(L, 1);
    
    // worker_threads
    lua_getfield(L, 1, "worker_threads");
    if (lua_isnumber(L, -1)) {
        config.worker_threads = static_cast<size_t>(lua_tonumber(L, -1));
    }
    lua_pop(L, 1);
    
    // Appliquer la configuration
    mgr.configure(config);
    
    std::cout << "🔧 Gestionnaire de mémoire configuré:" << std::endl;
    std::cout << "   - Limite RAM: " << (config.max_ram_bytes / 1024 / 1024 / 1024) << " GB" << std::endl;
    std::cout << "   - Compression: " << (config.enable_compression ? "activée" : "désactivée") << std::endl;
    std::cout << "   - Chargement async: " << (config.enable_async_loading ? "activé" : "désactivé") << std::endl;
    std::cout << "   - Prédiction: " << (config.enable_prediction ? "activée" : "désactivée") << std::endl;
    std::cout << "   - Worker threads: " << config.worker_threads << std::endl;
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_memoryGetStats(lua_State* L) {
    auto& mgr = AdvancedRAMManager::instance();
    
    // Retourner une table avec les statistiques
    lua_newtable(L);
    
    // current_mb
    size_t current = mgr.getCurrentRAM();
    lua_pushnumber(L, static_cast<double>(current) / (1024.0 * 1024.0));
    lua_setfield(L, -2, "current_mb");
    
    // peak_mb
    size_t peak = mgr.getPeakRAM();
    lua_pushnumber(L, static_cast<double>(peak) / (1024.0 * 1024.0));
    lua_setfield(L, -2, "peak_mb");
    
    // usage_percent
    float usage = mgr.getUsagePercent();
    lua_pushnumber(L, static_cast<double>(usage));
    lua_setfield(L, -2, "usage_percent");
    
    return 1;
}

int LuaScripting::lua_memoryPrintStats(lua_State* L) {
    auto& mgr = AdvancedRAMManager::instance();
    mgr.printDetailedStats();
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_memoryClear(lua_State* L) {
    auto& mgr = AdvancedRAMManager::instance();
    mgr.clear();
    
    std::cout << "🧹 Mémoire effacée" << std::endl;
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_memoryGetUsage(lua_State* L) {
    auto& mgr = AdvancedRAMManager::instance();
    
    // Retourner current_mb, peak_mb, usage_percent
    size_t current = mgr.getCurrentRAM();
    size_t peak = mgr.getPeakRAM();
    float usage = mgr.getUsagePercent();
    
    lua_pushnumber(L, static_cast<double>(current) / (1024.0 * 1024.0));
    lua_pushnumber(L, static_cast<double>(peak) / (1024.0 * 1024.0));
    lua_pushnumber(L, static_cast<double>(usage));
    
    return 3;
}

int LuaScripting::lua_memorySetLimit(lua_State* L) {
    auto& mgr = AdvancedRAMManager::instance();
    
    // Argument: limite en GB
    double gb = luaL_checknumber(L, 1);
    
    if (gb <= 0) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Limite doit être > 0");
        return 2;
    }
    
    // Créer une config avec la nouvelle limite
    AdvancedRAMManager::Config config;
    config.max_ram_bytes = static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
    config.enable_compression = true;
    config.enable_async_loading = false;
    config.enable_prediction = false;
    config.enable_statistics = true;
    config.worker_threads = 2;
    
    mgr.configure(config);
    
    std::cout << "💾 Limite RAM définie à " << gb << " GB" << std::endl;
    
    lua_pushboolean(L, true);
    return 1;
}

// ============================================================================
// Memory Guard API (Strict Enforcement)
// ============================================================================

int LuaScripting::lua_guardSetLimit(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    
    // Argument: peut être en GB (float) ou en bytes (très grand nombre)
    double value = luaL_checknumber(L, 1);
    
    if (value <= 0) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Limite doit être > 0");
        return 2;
    }
    
    size_t bytes;
    // Si la valeur est petite (<= 1000), on assume que c'est en GB
    if (value <= 1000.0) {
        bytes = static_cast<size_t>(value * 1024.0 * 1024.0 * 1024.0);
    } else {
        // Sinon c'est directement en bytes
        bytes = static_cast<size_t>(value);
    }
    
    guard.setLimit(bytes);
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_guardGetStats(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    
    // Retourner une table avec les statistiques
    lua_newtable(L);
    
    // current_mb
    lua_pushnumber(L, static_cast<double>(guard.getCurrentBytes()) / (1024.0 * 1024.0));
    lua_setfield(L, -2, "current_mb");
    
    // peak_mb
    lua_pushnumber(L, static_cast<double>(guard.getPeakBytes()) / (1024.0 * 1024.0));
    lua_setfield(L, -2, "peak_mb");
    
    // limit_mb
    lua_pushnumber(L, static_cast<double>(guard.getLimit()) / (1024.0 * 1024.0));
    lua_setfield(L, -2, "limit_mb");
    
    // usage_percent
    lua_pushnumber(L, static_cast<double>(guard.getUsagePercent()));
    lua_setfield(L, -2, "usage_percent");
    
    return 1;
}

int LuaScripting::lua_guardPrintStats(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    guard.printStats();
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_guardReset(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    guard.reset();
    
    std::cout << "🔄 MemoryGuard réinitialisé" << std::endl;
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_memoryguardGetCurrentUsage(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    lua_pushnumber(L, static_cast<double>(guard.getCurrentBytes()));
    return 1;
}

int LuaScripting::lua_memoryguardGetPeakUsage(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    lua_pushnumber(L, static_cast<double>(guard.getPeakBytes()));
    return 1;
}

int LuaScripting::lua_memoryguardGetLimit(lua_State* L) {
    auto& guard = MemoryGuard::instance();
    lua_pushnumber(L, static_cast<double>(guard.getLimit()));
    return 1;
}

// ============================================================================
// Dynamic Tensor Allocator API
// ============================================================================

int LuaScripting::lua_allocatorConfigure(lua_State* L) {
    auto& allocator = DynamicTensorAllocator::instance();
    
    // Argument: table de configuration
    if (!lua_istable(L, 1)) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Argument 1 doit être une table de configuration");
        return 2;
    }
    
    // max_ram_gb
    lua_getfield(L, 1, "max_ram_gb");
    double max_ram_gb = lua_isnumber(L, -1) ? lua_tonumber(L, -1) : 10.0;
    lua_pop(L, 1);
    
    // enable_compression
    lua_getfield(L, 1, "enable_compression");
    bool enable_compression = lua_isboolean(L, -1) ? lua_toboolean(L, -1) : true;
    lua_pop(L, 1);
    
    // Configurer
    allocator.configure(static_cast<size_t>(max_ram_gb), enable_compression);
    
    std::cout << "✓ DynamicTensorAllocator configuré" << std::endl;
    std::cout << "   - Limite: " << max_ram_gb << " GB" << std::endl;
    std::cout << "   - Compression: " << (enable_compression ? "activée" : "désactivée") << std::endl;
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_allocatorPrintStats(lua_State* L) {
    auto& allocator = DynamicTensorAllocator::instance();
    allocator.printStats();
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_allocatorGetStats(lua_State* L) {
    auto& allocator = DynamicTensorAllocator::instance();
    
    // Retourner une table avec les statistiques
    lua_newtable(L);
    
    // tensor_count
    lua_pushnumber(L, static_cast<double>(allocator.getTensorCount()));
    lua_setfield(L, -2, "tensor_count");
    
    // loaded_count
    lua_pushnumber(L, static_cast<double>(allocator.getLoadedCount()));
    lua_setfield(L, -2, "loaded_count");
    
    return 1;
}

// ============================================================================
// HtopDisplay API
// ============================================================================

int LuaScripting::lua_htopCreate(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    try {
        if (!ctx.asyncMonitor) {
            ctx.asyncMonitor = std::make_shared<AsyncMonitor>();
        }

        // Signature supportée:
        //   - Htop.create(enable_viz?: boolean)
        //   - Htop.create({ enable_viz = bool, enable_htop = bool, viz_config = { ... } })
        bool enable_viz = false;
        bool enable_htop = true;
        json viz_config;
        std::optional<bool> csv_flag;
        std::optional<bool> csv_enabled;
        std::optional<std::string> csv_path;
        if (lua_istable(L, 1)) {
            lua_getfield(L, 1, "enable_viz");
            if (lua_isboolean(L, -1)) enable_viz = lua_toboolean(L, -1);
            lua_pop(L, 1);

            lua_getfield(L, 1, "enable_htop");
            if (lua_isboolean(L, -1)) enable_htop = lua_toboolean(L, -1);
            lua_pop(L, 1);

            // Alias de compat (certains scripts passent {viz=true}).
            lua_getfield(L, 1, "viz");
            if (!enable_viz && lua_isboolean(L, -1)) enable_viz = lua_toboolean(L, -1);
            lua_pop(L, 1);

            lua_getfield(L, 1, "viz_config");
            if (lua_istable(L, -1)) {
                viz_config = luaTableToJson(L, -1);
            }
            lua_pop(L, 1);

            // Options CSV htop (compat Lua):
            // - csv=true/false
            // - csv_enabled=true/false
            // - csv_path="..." (alias: csv_file)
            lua_getfield(L, 1, "csv");
            if (lua_isboolean(L, -1)) csv_flag = (bool)lua_toboolean(L, -1);
            lua_pop(L, 1);

            lua_getfield(L, 1, "csv_enabled");
            if (lua_isboolean(L, -1)) csv_enabled = (bool)lua_toboolean(L, -1);
            lua_pop(L, 1);

            lua_getfield(L, 1, "csv_path");
            if (lua_isstring(L, -1)) csv_path = std::string(lua_tostring(L, -1));
            lua_pop(L, 1);

            if (!csv_path.has_value()) {
                lua_getfield(L, 1, "csv_file");
                if (lua_isstring(L, -1)) csv_path = std::string(lua_tostring(L, -1));
                lua_pop(L, 1);
            }
        } else if (!lua_isnoneornil(L, 1)) {
            enable_viz = lua_toboolean(L, 1);
        }

        // Si l'utilisateur demande la viz sans fournir de config, on force l'activation.
        // (Le Visualizer est disabled par défaut si visualization.enabled n'est pas true.)
        if (enable_viz) {
            if (!viz_config.contains("visualization")) viz_config["visualization"] = json::object();
            if (!viz_config["visualization"].contains("enabled")) {
                viz_config["visualization"]["enabled"] = true;
            }
        }

        ctx.asyncMonitor->start(enable_htop, enable_viz, viz_config);

        // Appliquer les options CSV côté HtopDisplay si présent.
        // Par défaut, AsyncMonitor désactive le CSV Htop si la Viz est active,
        // pour éviter les écritures concurrentes.
        if (enable_htop) {
            auto h = ctx.asyncMonitor->getHtop();
            if (h) {
                if (csv_path.has_value() && !csv_path->empty()) {
                    h->setCsvLogFile(*csv_path);
                }

                bool enable_csv = !enable_viz;
                if (csv_path.has_value()) {
                    // Si l'utilisateur fournit un chemin, activer par défaut.
                    enable_csv = true;
                }
                if (csv_flag.has_value()) {
                    enable_csv = *csv_flag;
                }
                if (csv_enabled.has_value()) {
                    enable_csv = *csv_enabled;
                }

                h->setCsvEnabled(enable_csv);
            }
        }

        ctx.addLog(std::string("AsyncMonitor démarré (") + (enable_htop ? "htop enabled" : "htop disabled") + ")");
        lua_pushboolean(L, true);

        // Remonter un warning sans casser les scripts (ok=true + msg).
        if (enable_viz && !ctx.asyncMonitor->vizInitOk()) {
            const std::string err = ctx.asyncMonitor->vizInitError();
            if (!err.empty()) {
                lua_pushstring(L, (std::string("Viz init failed: ") + err).c_str());
                return 2;
            }
        }
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

int LuaScripting::lua_htopUpdate(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    AsyncMonitor::Metrics metrics;

    // Signature supportée:
    //   - Htop.update(tbl)
    //   - Htop.update(epoch, total_epochs, batch, total_batches, loss, avg_loss, lr, ...)
    if (lua_istable(L, 1)) {
        auto get_int = [&](const char* key, int def) -> int {
            lua_getfield(L, 1, key);
            int v = lua_isnumber(L, -1) ? (int)lua_tointeger(L, -1) : def;
            lua_pop(L, 1);
            return v;
        };
        auto get_num = [&](const char* key, float def) -> float {
            lua_getfield(L, 1, key);
            float v = lua_isnumber(L, -1) ? (float)lua_tonumber(L, -1) : def;
            lua_pop(L, 1);
            return v;
        };
        auto get_str = [&](const char* key, const char* def) -> std::string {
            lua_getfield(L, 1, key);
            std::string v = lua_isstring(L, -1) ? std::string(lua_tostring(L, -1)) : std::string(def);
            lua_pop(L, 1);
            return v;
        };

        metrics.epoch = get_int("epoch", 0);
        metrics.total_epochs = get_int("total_epochs", get_int("totalEpochs", 0));
        metrics.batch = get_int("batch", 0);
        metrics.total_batches = get_int("total_batches", get_int("totalBatches", 0));

        // Compat: certains scripts utilisent `step`.
        const int step = get_int("step", 0);
        if (metrics.batch == 0 && step > 0) metrics.batch = step;

        metrics.loss = get_num("loss", 0.0f);
        metrics.avg_loss = get_num("avg_loss", get_num("avgLoss", 0.0f));
        metrics.lr = get_num("lr", 0.0f);
        metrics.batch_time_ms = get_int("batch_time_ms", get_int("batchTimeMs", 0));
        metrics.memory_mb = (size_t)std::max(0, get_int("memory_mb", get_int("memoryMb", 0)));
        metrics.memory_freed = (size_t)std::max(0, get_int("memory_freed", get_int("memoryFreed", 0)));
        metrics.bps = get_num("bps", 0.0f);
        metrics.params = (size_t)std::max(0, get_int("params", 0));
        metrics.timestep = get_num("timestep", 0.0f);
        metrics.kl = get_num("kl", 0.0f);
        metrics.wass = get_num("wass", 0.0f);
        metrics.ent = get_num("ent", 0.0f);
        metrics.mom = get_num("mom", 0.0f);
        metrics.spat = get_num("spat", 0.0f);
        metrics.temp = get_num("temp", 0.0f);
        metrics.mse = get_num("mse", 0.0f);
        metrics.grad_norm = get_num("grad_norm", get_num("gradNorm", 0.0f));
        metrics.grad_max = get_num("grad_max", get_num("gradMax", 0.0f));
        metrics.recon_loss_type = get_str("recon_loss_type", "");
        if (metrics.recon_loss_type.empty()) {
            metrics.recon_loss_type = get_str("reconLoss", "");
        }

        // Optimizer (optionnel) : soit champs plats, soit sous-table `optimizer`.
        metrics.opt_type = get_int("opt_type", get_int("optType", 0));
        metrics.opt_step = get_int("opt_step", get_int("optStep", 0));
        metrics.opt_beta1 = get_num("opt_beta1", get_num("optBeta1", 0.0f));
        metrics.opt_beta2 = get_num("opt_beta2", get_num("optBeta2", 0.0f));
        metrics.opt_eps = get_num("opt_eps", get_num("optEps", 0.0f));
        metrics.opt_weight_decay = get_num("opt_weight_decay", get_num("optWeightDecay", 0.0f));

        auto parse_opt_type = [&](const std::string& s) -> int {
            std::string t = s;
            std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            if (t == "sgd") return 0;
            if (t == "adam") return 1;
            if (t == "adamw") return 2;
            return metrics.opt_type;
        };

        // Support: opt_type="adamw" (string)
        lua_getfield(L, 1, "opt_type");
        if (lua_isstring(L, -1)) {
            metrics.opt_type = parse_opt_type(lua_tostring(L, -1));
        }
        lua_pop(L, 1);
        lua_getfield(L, 1, "optType");
        if (lua_isstring(L, -1)) {
            metrics.opt_type = parse_opt_type(lua_tostring(L, -1));
        }
        lua_pop(L, 1);

        lua_getfield(L, 1, "optimizer");
        if (lua_isstring(L, -1)) {
            // Support: optimizer="adamw" (string)
            metrics.opt_type = parse_opt_type(lua_tostring(L, -1));
        } else if (lua_istable(L, -1)) {
            lua_getfield(L, -1, "type");
            if (lua_isnumber(L, -1)) metrics.opt_type = (int)lua_tointeger(L, -1);
            else if (lua_isstring(L, -1)) metrics.opt_type = parse_opt_type(lua_tostring(L, -1));
            lua_pop(L, 1);
            lua_getfield(L, -1, "step");
            if (lua_isnumber(L, -1)) metrics.opt_step = (int)lua_tointeger(L, -1);
            lua_pop(L, 1);
            lua_getfield(L, -1, "beta1");
            if (lua_isnumber(L, -1)) metrics.opt_beta1 = (float)lua_tonumber(L, -1);
            lua_pop(L, 1);
            lua_getfield(L, -1, "beta2");
            if (lua_isnumber(L, -1)) metrics.opt_beta2 = (float)lua_tonumber(L, -1);
            lua_pop(L, 1);
            lua_getfield(L, -1, "eps");
            if (lua_isnumber(L, -1)) metrics.opt_eps = (float)lua_tonumber(L, -1);
            lua_pop(L, 1);
            lua_getfield(L, -1, "weight_decay");
            if (lua_isnumber(L, -1)) metrics.opt_weight_decay = (float)lua_tonumber(L, -1);
            lua_pop(L, 1);
        }
        lua_pop(L, 1);
    } else {
        // Arguments positionnels (legacy)
        // epoch, total_epochs, batch, total_batches, loss, avg_loss, lr,
        // batch_time_ms, memory_mb, memory_freed, bps, params, timestep,
        // kl, wass, ent, mom, spat, temp, mse, grad_norm, grad_max,
        // [opt_type, opt_step, opt_beta1, opt_beta2, opt_eps, opt_weight_decay]
        metrics.epoch = luaL_checkinteger(L, 1);
        metrics.total_epochs = luaL_checkinteger(L, 2);
        metrics.batch = luaL_checkinteger(L, 3);
        metrics.total_batches = luaL_checkinteger(L, 4);
        metrics.loss = static_cast<float>(luaL_checknumber(L, 5));
        metrics.avg_loss = static_cast<float>(luaL_checknumber(L, 6));
        metrics.lr = static_cast<float>(luaL_checknumber(L, 7));
        metrics.batch_time_ms = luaL_optinteger(L, 8, 0);
        metrics.memory_mb = static_cast<size_t>(luaL_optinteger(L, 9, 0));
        metrics.memory_freed = static_cast<size_t>(luaL_optinteger(L, 10, 0));
        metrics.bps = static_cast<float>(luaL_optnumber(L, 11, 0.0));
        metrics.params = static_cast<size_t>(luaL_optinteger(L, 12, 0));
        metrics.timestep = static_cast<float>(luaL_optnumber(L, 13, 0.0));
        metrics.kl = static_cast<float>(luaL_optnumber(L, 14, 0.0));
        metrics.wass = static_cast<float>(luaL_optnumber(L, 15, 0.0));
        metrics.ent = static_cast<float>(luaL_optnumber(L, 16, 0.0));
        metrics.mom = static_cast<float>(luaL_optnumber(L, 17, 0.0));
        metrics.spat = static_cast<float>(luaL_optnumber(L, 18, 0.0));
        metrics.temp = static_cast<float>(luaL_optnumber(L, 19, 0.0));
        metrics.mse = static_cast<float>(luaL_optnumber(L, 20, 0.0));
        metrics.grad_norm = static_cast<float>(luaL_optnumber(L, 21, 0.0));
        metrics.grad_max = static_cast<float>(luaL_optnumber(L, 22, 0.0));

        // Optimizer (optionnel)
        metrics.opt_type = luaL_optinteger(L, 23, 0);
        metrics.opt_step = luaL_optinteger(L, 24, 0);
        metrics.opt_beta1 = static_cast<float>(luaL_optnumber(L, 25, 0.0));
        metrics.opt_beta2 = static_cast<float>(luaL_optnumber(L, 26, 0.0));
        metrics.opt_eps = static_cast<float>(luaL_optnumber(L, 27, 0.0));
        metrics.opt_weight_decay = static_cast<float>(luaL_optnumber(L, 28, 0.0));
    }
    
    ctx.asyncMonitor->updateMetrics(metrics);
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_htopRender(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Render est automatique dans AsyncMonitor, cette fonction est maintenant no-op
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_htopClear(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    auto htop = ctx.asyncMonitor->getHtop();
    if (htop) {
        htop->clearScreen();
    }
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_htopEnable(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    // Note: HtopDisplay n'a pas de setEnabled(), on peut juste démarrer/arrêter le monitor
    bool enabled = lua_toboolean(L, 1);
    if (!enabled) {
        ctx.asyncMonitor->stop();
    }
    
    lua_pushboolean(L, true);
    return 1;
}

// ============================================================================
// Visualizer API
// ============================================================================

int LuaScripting::lua_vizCreate(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Argument: table de configuration
    json config;
    if (lua_istable(L, 1)) {
        config = luaTableToJson(L, 1);
    }

    // Si l'utilisateur appelle explicitement Viz.create(), on active la viz par défaut.
    if (!config.contains("visualization")) config["visualization"] = json::object();
    if (!config["visualization"].contains("enabled")) {
        config["visualization"]["enabled"] = true;
    }
    
    try {
        if (!ctx.asyncMonitor) {
            ctx.asyncMonitor = std::make_shared<AsyncMonitor>();
        }
        
        // Démarrer avec viz activé (et htop désactivé si pas déjà démarré)
        ctx.asyncMonitor->start(false, true, config);

        if (!ctx.asyncMonitor->vizInitOk()) {
            lua_pushboolean(L, false);
            const std::string err = ctx.asyncMonitor->vizInitError();
            lua_pushstring(L, err.empty() ? "Visualizer init failed" : err.c_str());
            return 2;
        }
        
        ctx.addLog("AsyncMonitor démarré (visualizer enabled)");
        lua_pushboolean(L, true);
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
    
    return 1;
}

int LuaScripting::lua_vizInitialize(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    auto viz = ctx.asyncMonitor->getViz();
    bool success = viz && viz->isOpen();
    lua_pushboolean(L, success);
    return 1;
}

int LuaScripting::lua_vizIsOpen(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        return 1;
    }
    
    auto viz = ctx.asyncMonitor->getViz();
    lua_pushboolean(L, viz && viz->isOpen());
    return 1;
}

int LuaScripting::lua_vizProcessEvents(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Process events est automatique dans AsyncMonitor, cette fonction est maintenant no-op
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizUpdate(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Update est automatique dans AsyncMonitor, cette fonction est maintenant no-op
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizAddImage(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    // Arguments (compat):
    //  - add_image(pixels_table, prompt)
    //  - add_image(pixels_table, prompt, w, h, channels)
    //  - add_image(pixels_table, w, h, channels, prompt)
    if (!lua_istable(L, 1)) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Argument 1 doit être une table de pixels");
        return 2;
    }

    int w = 0;
    int h = 0;
    int channels = 0;
    std::string prompt;

    if (lua_isnumber(L, 2)) {
        // (pixels, w, h, channels, prompt)
        w = luaL_checkinteger(L, 2);
        h = luaL_checkinteger(L, 3);
        channels = luaL_optinteger(L, 4, 0);
        prompt = luaL_optstring(L, 5, "");
    } else {
        // (pixels, prompt[, w, h, channels])
        prompt = luaL_optstring(L, 2, "");
        if (lua_isnumber(L, 3)) {
            w = luaL_checkinteger(L, 3);
            h = luaL_checkinteger(L, 4);
            channels = luaL_optinteger(L, 5, 0);
        }
    }
    
    // Lire la table de pixels
    std::vector<uint8_t> pixels;
    lua_pushnil(L);
    while (lua_next(L, 1) != 0) {
        if (lua_isnumber(L, -1)) {
            pixels.push_back(static_cast<uint8_t>(lua_tointeger(L, -1)));
        }
        lua_pop(L, 1);
    }
    
    ctx.asyncMonitor->addImage(pixels, w, h, channels, prompt);
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizUpdateMetrics(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    // Signature supportée:
    //   - Viz.update_metrics(tbl)  (recommandé)
    //   - Viz.update_metrics(epoch, batch, loss, lr, mse, kl, wass, ent, mom, spat, temp, kl_beta_effective[, time_ms, mem_mb, bps, params])
    // Notes compat:
    //   - tbl.time / tbl.time_ms -> batch_time_ms
    //   - tbl.mem / tbl.mem_mb   -> memory_mb
    AsyncMonitor::Metrics metrics;

    if (lua_istable(L, 1)) {
        auto get_int = [&](const char* key, int def) -> int {
            lua_getfield(L, 1, key);
            int v = lua_isnumber(L, -1) ? (int)lua_tointeger(L, -1) : def;
            lua_pop(L, 1);
            return v;
        };
        auto get_num = [&](const char* key, float def) -> float {
            lua_getfield(L, 1, key);
            float v = lua_isnumber(L, -1) ? (float)lua_tonumber(L, -1) : def;
            lua_pop(L, 1);
            return v;
        };
        auto get_str = [&](const char* key, const char* def) -> std::string {
            lua_getfield(L, 1, key);
            std::string v = lua_isstring(L, -1) ? std::string(lua_tostring(L, -1)) : std::string(def);
            lua_pop(L, 1);
            return v;
        };

        metrics.epoch = get_int("epoch", 0);
        metrics.total_epochs = get_int("total_epochs", get_int("totalEpochs", 0));
        metrics.batch = get_int("batch", 0);
        metrics.total_batches = get_int("total_batches", get_int("totalBatches", 0));
        const int step = get_int("step", 0);
        if (metrics.batch == 0 && step > 0) metrics.batch = step;

        metrics.loss = get_num("loss", 0.0f);
        metrics.avg_loss = get_num("avg_loss", get_num("avgLoss", 0.0f));
        metrics.lr = get_num("lr", 0.0f);

        metrics.batch_time_ms = get_int(
            "batch_time_ms",
            get_int("batchTimeMs", get_int("time_ms", get_int("timeMs", get_int("time", 0))))
        );
        metrics.memory_mb = (size_t)std::max(
            0,
            get_int("memory_mb", get_int("memoryMb", get_int("mem_mb", get_int("memMb", get_int("mem", 0)))))
        );
        metrics.bps = get_num("bps", get_num("batches_per_sec", get_num("batchesPerSec", 0.0f)));
        metrics.params = (size_t)std::max(0, get_int("params", 0));

        metrics.mse = get_num("mse", 0.0f);
        metrics.kl = get_num("kl", 0.0f);
        metrics.wass = get_num("wass", 0.0f);
        metrics.ent = get_num("ent", 0.0f);
        metrics.mom = get_num("mom", 0.0f);
        metrics.spat = get_num("spat", 0.0f);
        metrics.temp = get_num("temp", 0.0f);
        metrics.timestep = get_num("timestep", 0.0f);
        metrics.grad_norm = get_num("grad_norm", get_num("gradNorm", 0.0f));
        metrics.grad_max = get_num("grad_max", get_num("gradMax", 0.0f));
        metrics.kl_beta_effective = get_num("kl_beta_effective", get_num("klBetaEffective", 0.0f));
        metrics.recon_loss_type = get_str("recon_loss_type", "");
        if (metrics.recon_loss_type.empty()) {
            metrics.recon_loss_type = get_str("reconLoss", "");
        }
    } else {
        metrics.epoch = luaL_checkinteger(L, 1);
        metrics.batch = luaL_checkinteger(L, 2);
        metrics.loss = static_cast<float>(luaL_checknumber(L, 3));
        metrics.lr = static_cast<float>(luaL_checknumber(L, 4));
        metrics.mse = static_cast<float>(luaL_optnumber(L, 5, 0.0));
        metrics.kl = static_cast<float>(luaL_optnumber(L, 6, 0.0));
        metrics.wass = static_cast<float>(luaL_optnumber(L, 7, 0.0));
        metrics.ent = static_cast<float>(luaL_optnumber(L, 8, 0.0));
        metrics.mom = static_cast<float>(luaL_optnumber(L, 9, 0.0));
        metrics.spat = static_cast<float>(luaL_optnumber(L, 10, 0.0));
        metrics.temp = static_cast<float>(luaL_optnumber(L, 11, 0.0));
        metrics.kl_beta_effective = static_cast<float>(luaL_optnumber(L, 12, 0.0));
        metrics.batch_time_ms = luaL_optinteger(L, 13, 0);
        {
            lua_Integer v = luaL_optinteger(L, 14, 0);
            if (v < 0) v = 0;
            metrics.memory_mb = (size_t)v;
        }
        metrics.bps = static_cast<float>(luaL_optnumber(L, 15, 0.0));
        {
            lua_Integer v = luaL_optinteger(L, 16, 0);
            if (v < 0) v = 0;
            metrics.params = (size_t)v;
        }
    }
    
    ctx.asyncMonitor->updateMetrics(metrics);
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizSetValidation(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }

    if (!lua_istable(L, 1)) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Argument 1 doit être une table (ex: {in_progress=true, step=123, done=1, total=8})");
        return 2;
    }

    auto getBoolField = [&](const char* key, bool def) -> bool {
        lua_getfield(L, 1, key);
        bool v = def;
        if (lua_isboolean(L, -1)) v = lua_toboolean(L, -1);
        lua_pop(L, 1);
        return v;
    };
    auto getIntField = [&](const char* key, int def) -> int {
        lua_getfield(L, 1, key);
        int v = def;
        if (lua_isnumber(L, -1)) v = static_cast<int>(lua_tointeger(L, -1));
        lua_pop(L, 1);
        return v;
    };
    auto getNumField = [&](const char* key, float def) -> float {
        lua_getfield(L, 1, key);
        float v = def;
        if (lua_isnumber(L, -1)) v = static_cast<float>(lua_tonumber(L, -1));
        lua_pop(L, 1);
        return v;
    };

    const bool in_progress = getBoolField("in_progress", false);
    const int step = getIntField("step", 0);
    const int done = getIntField("done", 0);
    const int total = getIntField("total", 0);
    const bool has = getBoolField("has", false);
    const bool ok = getBoolField("ok", true);
    const float recon = getNumField("recon", 0.0f);
    const float kl = getNumField("kl", 0.0f);
    const float align = getNumField("align", 0.0f);

    ctx.asyncMonitor->updateValidation(in_progress, step, done, total, has, ok, recon, kl, align);

    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizAddLossPoint(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // AddLossPoint est automatique dans AsyncMonitor via updateMetrics
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizClear(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        return 1;
    }
    
    auto viz = ctx.asyncMonitor->getViz();
    if (viz) {
        viz->clearImages();
    }
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizSetEnabled(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        return 1;
    }
    
    // Note: Visualizer n'a pas de setEnabled(), on peut juste démarrer/arrêter le monitor
    bool enabled = lua_toboolean(L, 1);
    if (!enabled) {
        ctx.asyncMonitor->stop();
    }
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_vizSaveLossHistory(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.asyncMonitor) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "AsyncMonitor non créé");
        return 2;
    }
    
    const char* filepath = luaL_checkstring(L, 1);
    auto viz = ctx.asyncMonitor->getViz();
    if (viz) {
        viz->saveLossHistory(filepath);
    }
    
    lua_pushboolean(L, true);
    return 1;
}

int LuaScripting::lua_extractKeywords(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentTokenizer) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun tokenizer créé");
        return 2;
    }
    
    const char* text = luaL_checkstring(L, 1);
    int topN = luaL_optinteger(L, 2, 5);
    
    auto keywords = ctx.currentTokenizer->extractKeywords(text, topN);
    
    lua_newtable(L);
    for (size_t i = 0; i < keywords.size(); ++i) {
        lua_pushstring(L, keywords[i].c_str());
        lua_rawseti(L, -2, i + 1);
    }
    
    return 1;
}
