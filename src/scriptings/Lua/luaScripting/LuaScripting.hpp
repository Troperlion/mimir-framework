#pragma once

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}
#include <string>
#include <vector>
#include <memory>
#include "Models/Registry/ModelArchitectures.hpp"
#include "scriptings/ScriptingRuntime.hpp"
#include "scriptings/ScriptingContext.hpp"

using json = nlohmann::json;

// ============================================================================
// LuaScripting - Interface Lua pour piloter le modèle
// ============================================================================

class LuaScripting : public ScriptingRuntime {
public:
    LuaScripting();
    ~LuaScripting();

    // Injecter les arguments (équivalent au `arg` de l'interpréteur Lua)
    void setArgs(const std::string& script_path, const std::vector<std::string>& script_args) override;

    // Charger et exécuter un script Lua
    bool loadScript(const std::string& filepath) override;
    bool executeScript(const std::string& code) override;
    
    // Exécuter une fonction Lua
    bool callFunction(const std::string& function_name);
    
    // Obtenir des valeurs depuis Lua
    std::string getString(const std::string& var_name);
    double getNumber(const std::string& var_name);
    bool getBoolean(const std::string& var_name);
    
    // Définir des valeurs dans Lua
    void setString(const std::string& var_name, const std::string& value);
    void setNumber(const std::string& var_name, double value);
    void setBoolean(const std::string& var_name, bool value);
    void setSystemConfig(const json& conf, const std::string& conf_path, const std::string& conf_dir);
    
    // Obtenir l'état Lua (pour usage avancé)
    lua_State* getState() { return L; }
    
    // Enregistrer les bindings
    void registerAPI() override;

    // Découvre <project_root>/_archi/*.mpk et expose chaque package valide
    // comme une entrée du registre d'architectures.
    static std::size_t autoRegisterMpkArchitectures(
        const std::string& project_root,
        std::vector<std::string>* warnings = nullptr);

private:
    lua_State* L;
    
    // === Model API ===
    static int lua_createModel(lua_State* L);
    static int lua_createEmptyModel(lua_State* L);
    static int lua_createModelFromConfig(lua_State* L);
    static int lua_buildModel(lua_State* L);
    static int lua_trainModel(lua_State* L);
    static int lua_visionTrainStep(lua_State* L);
    static int lua_inferModel(lua_State* L);
    static int lua_saveModel(lua_State* L);
    static int lua_loadModel(lua_State* L);
    static int lua_modelDType(lua_State* L);
    static int lua_modelGetConfig(lua_State* L);

    // === Benchmark API ===
    static int lua_benchmarkNow(lua_State* L);
    
    // === New Serialization API ===
    static int lua_saveCheckpoint(lua_State* L);    // Serialization API v2.3
    static int lua_loadCheckpoint(lua_State* L);    // Serialization API v2.3
    static int lua_detectFormat(lua_State* L);      // Serialization API v2.3
    static int lua_saveEnhancedDebugJson(lua_State* L);  // Enhanced Debug JSON v1.1.0
    
    static int lua_allocateParams(lua_State* L);
    static int lua_initWeights(lua_State* L);
    static int lua_totalParams(lua_State* L);
    static int lua_getModelLayers(lua_State* L);
    static int lua_clearModelLayers(lua_State* L);
    static int lua_pushLayer(lua_State* L);
    static int lua_setLayerIO(lua_State* L);  // NEW: Configure inputs/outputs
    static int lua_forwardPass(lua_State* L);
    static int lua_forwardPromptImageSeed(lua_State* L);
    static int lua_lumenText2Img(lua_State* L);
    static int lua_lumenBeginVaeCalibration(lua_State* L);
    static int lua_lumenAddVaeCalibrationImage(lua_State* L);
    static int lua_lumenFinishVaeCalibration(lua_State* L);
    static int lua_lumenTrainStep(lua_State* L);
    static int lua_lumenValidateStep(lua_State* L);

    // === Image IO helpers (Lua) ===
    // Charge une image depuis le disque via stb_image et renvoie des pixels RGB u8.
    static int lua_readImageRGBU8(lua_State* L);
    static int lua_encodePrompt(lua_State* L);
    static int lua_backwardPass(lua_State* L);
    static int lua_optimizerStep(lua_State* L);
    static int lua_getOptimizer(lua_State* L);
    static int lua_setOptimizer(lua_State* L);
    static int lua_resetOptimizerState(lua_State* L);
    static int lua_zeroGradients(lua_State* L);
    static int lua_getGradients(lua_State* L);
    static int lua_setHardwareAccel(lua_State* L);
    static int lua_getHardwareCaps(lua_State* L);
    static int lua_runtimeOps(lua_State* L);
    static int lua_conditioningPrompt(lua_State* L);
    
    // === ModelArchitectures API ===
    static int lua_archAvailable(lua_State* L);
    static int lua_archDefaultConfig(lua_State* L);
    static int lua_archInfo(lua_State* L);
    static int lua_archDtypes(lua_State* L);
    
    // === Layer Operations API ===
    static int lua_layersAvailable(lua_State* L);
    static int lua_layersByType(lua_State* L);
    static int lua_computeConv2D(lua_State* L);
    static int lua_computeLinear(lua_State* L);
    static int lua_computeMaxPool2D(lua_State* L);
    static int lua_computeAvgPool2D(lua_State* L);
    static int lua_computeActivation(lua_State* L);
    static int lua_computeBatchNorm(lua_State* L);
    static int lua_computeLayerNorm(lua_State* L);
    static int lua_computeAttention(lua_State* L);
    
    // === HtopDisplay API ===
    static int lua_htopCreate(lua_State* L);
    static int lua_htopUpdate(lua_State* L);
    static int lua_htopRender(lua_State* L);
    static int lua_htopClear(lua_State* L);
    static int lua_htopEnable(lua_State* L);
    
    // === Visualizer API ===
    static int lua_vizCreate(lua_State* L);
    static int lua_vizConfigure(lua_State* L);
    static int lua_vizPollEvents(lua_State* L);
    static int lua_vizBackend(lua_State* L);
    static int lua_vizInitialize(lua_State* L);
    static int lua_vizIsOpen(lua_State* L);
    static int lua_vizProcessEvents(lua_State* L);
    static int lua_vizUpdate(lua_State* L);
    static int lua_vizAddImage(lua_State* L);
    static int lua_vizSetDetections(lua_State* L);
    static int lua_vizShowDetections(lua_State* L);
    static int lua_vizSetDetectionThreshold(lua_State* L);
    static int lua_vizUpdateMetrics(lua_State* L);
    static int lua_vizSetValidation(lua_State* L);
    static int lua_vizValidationEnabled(lua_State* L);
    static int lua_vizAddLossPoint(lua_State* L);
    static int lua_vizClear(lua_State* L);
    static int lua_vizSetEnabled(lua_State* L);
    static int lua_vizSaveLossHistory(lua_State* L);
    
    // === Tokenizer API ===
    static int lua_createTokenizer(lua_State* L);
    static int lua_tokenize(lua_State* L);
    static int lua_detokenize(lua_State* L);
    static int lua_getVocabSize(lua_State* L);
    static int lua_getMaxVocab(lua_State* L);
    static int lua_setMaxVocab(lua_State* L);
    static int lua_saveTokenizer(lua_State* L);
    static int lua_loadTokenizer(lua_State* L);
    
    // Manipulation vocabulaire
    static int lua_addToken(lua_State* L);
    static int lua_ensureVocabFromText(lua_State* L);
    
    // === Memory Manager API ===
    static int lua_memoryConfig(lua_State* L);
    static int lua_memoryGetStats(lua_State* L);
    static int lua_memoryPrintStats(lua_State* L);
    static int lua_memoryClear(lua_State* L);
    static int lua_memoryGetUsage(lua_State* L);
    static int lua_memorySetLimit(lua_State* L);
    
    // === Dynamic Allocator API ===
    static int lua_allocatorConfigure(lua_State* L);
    static int lua_allocatorPrintStats(lua_State* L);
    static int lua_allocatorGetStats(lua_State* L);

    // === Scripting output control ===
    static int lua_setStdoutLogSuppressed(lua_State* L);
    
    // === Memory Guard API (strict enforcement) ===
    static int lua_guardSetLimit(lua_State* L);
    static int lua_guardGetStats(lua_State* L);
    static int lua_guardPrintStats(lua_State* L);
    static int lua_guardReset(lua_State* L);
    
    // === MemoryGuard helper functions ===
    static int lua_memoryguardGetCurrentUsage(lua_State* L);
    static int lua_memoryguardGetPeakUsage(lua_State* L);
    static int lua_memoryguardGetLimit(lua_State* L);
    
    static int lua_tokenizeEnsure(lua_State* L);
    
    // Tokens spéciaux
    static int lua_getPadId(lua_State* L);
    static int lua_getUnkId(lua_State* L);
    static int lua_getSeqId(lua_State* L);
    static int lua_getModId(lua_State* L);
    static int lua_getMagId(lua_State* L);
    static int lua_getTokenById(lua_State* L);
    
    // BPE
    static int lua_learnBPEFromCorpus(lua_State* L);
    static int lua_tokenizeBPE(lua_State* L);
    static int lua_setMaxSequenceLength(lua_State* L);
    static int lua_padSequence(lua_State* L);
    static int lua_batchTokenize(lua_State* L);
    
    // Statistiques et analyse
    static int lua_printVocabStats(lua_State* L);
    static int lua_getTokenFrequencies(lua_State* L);
    static int lua_analyzeText(lua_State* L);
    static int lua_extractKeywords(lua_State* L);
    
    // === Dataset API ===
    static int lua_loadDataset(lua_State* L);
    static int lua_getDataset(lua_State* L);
    static int lua_releaseDatasetItem(lua_State* L);
    static int lua_prepareSequences(lua_State* L);

    // === Database API (dataset loader with caching builder) ===
    // Usage: Mimir.Database.load(dir, w, h, min_modalities).cache(cache_path?, max_ram_mb?, lazy_loading?)
    static int lua_databaseLoad(lua_State* L);
    static int lua_databaseLoad_cache(lua_State* L);
    
    // === Utilitaires ===
    static int lua_print(lua_State* L);
    static int lua_log(lua_State* L);
    static int lua_readJSON(lua_State* L);
    static int lua_writeJSON(lua_State* L);
    
    // Helpers
    static json luaTableToJson(lua_State* L, int index);
    static void jsonToLuaTable(lua_State* L, const json& j);
};

// ============================================================================
// NOTE (TUX/htop): quand HtopDisplay est actif, écrire sur stdout casse le rendu.
// Les logs sont conservés en mémoire (ctx.logs) mais ne sont pas imprimés.
// ============================================================================

// ============================================================================
// Singleton global pour accès depuis les callbacks Lua
// ============================================================================
class LuaContext : public ScriptingContext {
public:
    static LuaContext& getInstance() {
        static LuaContext instance;
        return instance;
    }
    
private:
    LuaContext() = default;
    ~LuaContext() = default;
    LuaContext(const LuaContext&) = delete;
    LuaContext& operator=(const LuaContext&) = delete;
};
