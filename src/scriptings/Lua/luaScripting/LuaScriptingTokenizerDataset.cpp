#include "runtimes/fpga/FpgaIo.hpp"
#include "LuaScripting.hpp"
#include "VizTextPayload.hpp"
#include "training/TagAnnotation.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"
#include "Serialization/DebugJsonDump.hpp"
#include "DType.hpp"
#include "AdvancedRAMManager.hpp"
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#include "AsyncMonitor.hpp"
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
#include <type_traits>
#include <utility>
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
    const bool load_image = lua_gettop(L) < 2 || lua_toboolean(L, 2);
    
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
        if (!item.mask_file.empty()) {
            lua_pushstring(L, item.mask_file.c_str());
            lua_setfield(L, -2, "mask_file");
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
        if ((!item.text_file.empty() || !item.text_inline.empty()) && !item.text.has_value()) {
            if (!item.loadText() && !item.annotation_file.empty()) {
                throw std::runtime_error(item.annotation_error.empty() ? "Cannot load dataset annotation (RAM limit?)" : item.annotation_error);
            }
        }
        if (item.text.has_value()) {
            lua_pushstring(L, item.text.value().c_str());
            lua_setfield(L, -2, "text");
        }

        if (item.text.has_value()) {
            const auto labels=TagAnnotation::labels(*item.text,item.annotation ? &*item.annotation : nullptr,false);
            lua_newtable(L);
            for(size_t i=0;i<labels.size();++i) {
                lua_pushlstring(L,labels[i].data(),labels[i].size());
                lua_rawseti(L,-2,static_cast<int>(i+1));
            }
            lua_setfield(L,-2,"label_candidates");
        }
        if (!item.annotation_file.empty()) {
            lua_pushstring(L, item.annotation_file.c_str());
            lua_setfield(L, -2, "annotation_file");
            if (item.annotation) {
                jsonToLuaTable(L, *item.annotation);
                lua_setfield(L, -2, "annotation");
            }
        }

        // Charger et retourner l'image en bytes RGB u8 (utile pour diffusion).
        if (!item.image_file.empty()) {
            if (load_image) item.loadImageRGB(target_w, target_h);
            if (load_image && item.img_loaded) {
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

            if (load_image && item.img_loaded) {
                const size_t frame_count = item.animationFrameCount();
                lua_pushboolean(L, item.isAnimatedImage());
                lua_setfield(L, -2, "is_animated");
                lua_pushinteger(L, static_cast<lua_Integer>(frame_count));
                lua_setfield(L, -2, "frame_count");
                lua_pushinteger(L, item.animation_loop_count);
                lua_setfield(L, -2, "loop_count");

                if (item.isAnimatedImage()) {
                    lua_newtable(L);
                    for (size_t i = 0; i < item.animation_frame_delays_ms.size(); ++i) {
                        lua_pushinteger(L, item.animation_frame_delays_ms[i]);
                        lua_rawseti(L, -2, static_cast<lua_Integer>(i + 1));
                    }
                    lua_setfield(L, -2, "frame_delays_ms");

                    const size_t frame_size = static_cast<size_t>(item.w) *
                                              static_cast<size_t>(item.h) *
                                              static_cast<size_t>(item.img_c);
                    lua_newtable(L);
                    for (size_t frame = 0; frame < frame_count; ++frame) {
                        lua_newtable(L);
                        const size_t offset = frame * frame_size;
                        for (size_t i = 0; i < frame_size; ++i) {
                            lua_pushinteger(L, static_cast<lua_Integer>(item.animation_frames[offset + i]));
                            lua_rawseti(L, -2, static_cast<lua_Integer>(i + 1));
                        }
                        lua_rawseti(L, -2, static_cast<lua_Integer>(frame + 1));
                    }
                    lua_setfield(L, -2, "animation_frames");
                }
            }

            if (!item.mask_file.empty()) {
                if (load_image) item.loadMask(target_w, target_h);
                if (load_image && item.mask_loaded) {
                    lua_newtable(L);
                    for (size_t i = 0; i < item.mask.size(); ++i) {
                        lua_pushinteger(L, (lua_Integer)item.mask[i]);
                        lua_rawseti(L, -2, (int)i + 1);
                    }
                    lua_setfield(L, -2, "mask");
                }
            }
        }
        
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_releaseDatasetItem(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    const int index = luaL_checkinteger(L, 1);
    if (index < 1 || index > static_cast<int>(ctx.currentDataset.size())) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Index hors limites");
        return 2;
    }
    ctx.currentDataset[static_cast<size_t>(index - 1)].unload();
    lua_pushboolean(L, true);
    return 1;
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
        Mimir::fpgaStageDataset(dst);

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

