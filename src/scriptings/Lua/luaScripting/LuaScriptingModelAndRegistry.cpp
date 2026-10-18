#include "LuaScripting.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "Models/Diffusion/LumenLatentDiffusionModel.hpp"
#include "Models/Vision/CNN/MedicalDetectorModel.hpp"
#include "Serialization/Serialization.hpp"
#include "Serialization/DebugJsonDump.hpp"
#include "DType.hpp"
#include "AdvancedRAMManager.hpp"
#include "MemoryGuard.hpp"
#include "DynamicTensorAllocator.hpp"
#include "AsyncMonitor.hpp"
#include "VizTextPayload.hpp"
#include "Helpers.hpp"
#include "runtimes/ops_loss_and_grad.hpp"
#if defined(MIMIR_ENABLE_FPGA_RUNTIME)
#include "runtimes/fpga/FpgaRuntime.hpp"
#include "runtimes/fpga/FpgaValidationRunner.hpp"
#endif
#include "training/VaeValidationFeedback.hpp"
#include "training/BBoxSupervision.hpp"
#include "training/TagAnnotation.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <unordered_set>
#include <type_traits>
#include <utility>
#if defined(__linux__)
#include <unistd.h>
#endif

namespace {

static LumenLatentDiffusionModel& _mimir_current_lumen_model() {
    auto& ctx = LuaContext::getInstance();
    if (!ctx.currentModel) throw std::runtime_error("Aucun modèle créé");
    auto* lumen = dynamic_cast<LumenLatentDiffusionModel*>(ctx.currentModel.get());
    if (!lumen) {
        throw std::runtime_error(
            "Le modèle courant n'est pas un LumenLatentDiffusionModel");
    }
    return *lumen;
}

static std::vector<unsigned char> _mimir_lua_u8_array(lua_State* L, int index) {
    luaL_checktype(L, index, LUA_TTABLE);
    const size_t count = lua_rawlen(L, index);
    std::vector<unsigned char> values;
    values.reserve(count);
    for (size_t item = 1; item <= count; ++item) {
        lua_rawgeti(L, index, static_cast<lua_Integer>(item));
        if (!lua_isnumber(L, -1)) {
            lua_pop(L, 1);
            throw std::runtime_error("image Lumen: valeur non numérique");
        }
        const lua_Number value = lua_tonumber(L, -1);
        lua_pop(L, 1);
        if (!std::isfinite(static_cast<double>(value)) || value < 0.0 || value > 255.0) {
            throw std::runtime_error("image Lumen: les pixels doivent être dans [0,255]");
        }
        values.push_back(static_cast<unsigned char>(std::lround(value)));
    }
    return values;
}

static void _mimir_push_lumen_train_stats(
    lua_State* L,
    const LumenLatentDiffusionModel::TrainStats& stats,
    bool stop_requested
) {
    lua_createtable(L, 0, 15);
    auto number = [L](const char* name, double value) {
        lua_pushnumber(L, value);
        lua_setfield(L, -2, name);
    };
    number("loss", stats.loss);
    number("mse", stats.mse);
    number("mse_unweighted", stats.mse_unweighted);
    number("bbox_count", stats.bbox_count);
    number("skeleton_region_count", stats.skeleton_region_count);
    number("relation_count", stats.relation_count);
    number("relation_loss", stats.relation_loss);
    number("bbox_loss_weight", stats.bbox_loss_weight);
    number("kl", stats.kl);
    number("kl_beta_effective", stats.kl_beta_effective);
    number("grad_norm", stats.grad_norm);
    number("grad_max_abs", stats.grad_max_abs);
    number("reconstruction_mae", stats.reconstruction_mae);
    number("reconstruction_mse", stats.reconstruction_mse);
    number("wasserstein", stats.wasserstein);
    number("entropy_diff", stats.entropy_diff);
    number("moment_mismatch", stats.moment_mismatch);
    number("spatial_coherence", stats.spatial_coherence);
    number("temporal_consistency", stats.temporal_consistency);
    lua_pushinteger(L, stats.timestep);
    lua_setfield(L, -2, "timestep");
    lua_pushboolean(L, stop_requested);
    lua_setfield(L, -2, "stop_requested");
}

static void _mimir_push_lumen_calibration_stats(
    lua_State* L,
    const LumenLatentDiffusionModel::VaeCalibrationStats& stats
) {
    lua_createtable(L, 0, 4);
    lua_pushinteger(L, static_cast<lua_Integer>(stats.items));
    lua_setfield(L, -2, "items");
    lua_pushinteger(L, static_cast<lua_Integer>(stats.values));
    lua_setfield(L, -2, "values");
    lua_pushnumber(L, stats.shift);
    lua_setfield(L, -2, "shift");
    lua_pushnumber(L, stats.scale);
    lua_setfield(L, -2, "scale");
}

static Optimizer& _mimir_lumen_optimizer(
    Model& model,
    const std::string& optimizer_name,
    float learning_rate
) {
    if (!model.getSerializedOptimizer()) {
        Optimizer optimizer;
        optimizer.initial_lr = learning_rate;
        const auto& config = model.modelConfig;
        configureOptimizerFromJson(optimizer, config);
        if (!optimizerTypeFromString(optimizer_name, optimizer.type)) {
            throw std::runtime_error("optimiseur Lumen inconnu: " + optimizer_name);
        }
        model.setSerializedOptimizer(std::move(optimizer));
    }
    Optimizer* optimizer = model.getMutableSerializedOptimizer();
    if (!optimizer) throw std::runtime_error("état optimiseur Lumen indisponible");
    optimizer->initial_lr = learning_rate;
    return *optimizer;
}

static void _mimir_publish_viz_taps(ScriptingContext& ctx, bool validation) {
    if (!ctx.currentModel || !ctx.asyncMonitor || !ctx.asyncMonitor->getViz()) return;

    // Graph intermediates belong to Blocks, never to the validation gallery.
    if (validation) return;
    auto taps = ctx.currentModel->consumeVizTaps();

    std::vector<Visualizer::BlockFrame> frames;
    frames.reserve(taps.size());
    for (auto& tap : taps) {
        Visualizer::BlockFrame frame;
        frame.pixels = std::move(tap.pixels);
        frame.w = tap.w;
        frame.h = tap.h;
        frame.channels = tap.channels;
        frame.pixels_real = std::move(tap.pixels_real);
        frame.heatmap_kind = tap.heatmap_kind;
        frame.tensor_info = std::move(tap.tensor_info);
        frame.tensor_values = std::move(tap.tensor_values);
        frame.tensor_indices = std::move(tap.tensor_indices);
        frame.tensor_sample_channels = tap.tensor_sample_channels;
        frame.label = std::move(tap.label);
        frames.push_back(std::move(frame));
    }
    ctx.asyncMonitor->setLayerBlockImages(std::move(frames));
}

static void _mimir_publish_lumen_viz(
    ScriptingContext& ctx,
    LumenLatentDiffusionModel& lumen,
    const std::vector<unsigned char>& image,
    const std::string& prompt,
    const char* sample_label,
    bool validation,
    bool bbox_supervision = false,
    const json* annotation = nullptr,
    float spatial_strength = -1.f
) {
    if (!ctx.asyncMonitor || !ctx.asyncMonitor->getViz()) return;

    const auto& config = lumen.getConfig();
    const int pad_id = ctx.currentTokenizer ? ctx.currentTokenizer->getPadId() : -1;
    const auto text = VizTextPayload::buildDatasetTextPayload(
        ctx.currentModel.get(),
        ctx.currentTokenizer.get(),
        prompt,
        nullptr,
        pad_id);
    json spatial_config=lumen.modelConfig;
    if(spatial_strength>=0)spatial_config["bbox_loss_weight"]=spatial_strength;
    ctx.asyncMonitor->setDatasetSample(
        image,
        config.image_w,
        config.image_h,
        config.image_c,
        sample_label,
        text.prompt,
        text.tags,
        text.tokens,
        text.encoding, bbox_supervision, DatasetBBoxOverlay::fromAnnotation(annotation), annotation, &spatial_config);
    if (validation) {
        // Only the final decoded inference output is paired with this original.
        for (const auto& frame : ctx.currentModel->consumeVizTaps()) {
            if (frame.label != "diffusion_out") continue;
            if (frame.w <= 0 || frame.h <= 0 || frame.channels != config.image_c) continue;
            const size_t expected = static_cast<size_t>(frame.w) * frame.h * frame.channels;
            if (frame.pixels.size() != expected) continue;
            ctx.asyncMonitor->addImage(image, config.image_w, config.image_h, config.image_c,
                "VAL target | " + prompt, true);
            ctx.asyncMonitor->addImage(frame.pixels, frame.w, frame.h, frame.channels,
                "VAL recon | " + prompt, true);
            break;
        }
    }
    // Commit validation tips to Blocks too; the gallery only receives decoded images.
    _mimir_publish_viz_taps(ctx, false);
}

static size_t _mimir_process_resident_memory_mb() {
#if defined(__linux__)
    std::ifstream statm("/proc/self/statm");
    size_t total_pages = 0;
    size_t resident_pages = 0;
    if (statm >> total_pages >> resident_pages) {
        (void)total_pages;
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (page_size > 0) {
            const uint64_t resident_bytes = static_cast<uint64_t>(resident_pages) *
                                            static_cast<uint64_t>(page_size);
            return static_cast<size_t>(resident_bytes / (1024ULL * 1024ULL));
        }
    }
#endif
    return MemoryGuard::instance().getCurrentBytes() / 1024 / 1024;
}

static std::vector<float> _mimir_load_mask_weights(
    DatasetItem& item,
    int image_w,
    int image_h,
    int image_c
) {
    if (item.mask_file.empty() || !item.loadMask(image_w, image_h)) return {};
    return item.makeMaskLossWeights(image_c);
}

static BBoxSupervision::Weights _mimir_load_spatial_loss_weights(
    DatasetItem& item, int image_w, int image_h, int image_c, float strength, const json& config
) {
    if (!item.annotation_file.empty() && !item.annotation && !item.loadText())
        throw std::runtime_error(item.annotation_error.empty() ? "Cannot load bbox annotation" : item.annotation_error);
    auto weights=BBoxSupervision::makeWeights(item.annotation ? &*item.annotation : nullptr,
        image_w,image_h,image_c,BBoxSupervision::Layout::HWC,strength,&config);
    BBoxSupervision::combineMask(weights,_mimir_load_mask_weights(item,image_w,image_h,image_c));
    return weights;
}

static bool _mimir_dataset_bbox_active(const DatasetItem& item, const Model& model) {
    const auto weights=BBoxSupervision::makeWeights(item.annotation ? &*item.annotation : nullptr,
        1,1,1,BBoxSupervision::Layout::HWC,model.modelConfig.value("bbox_loss_weight",1.0f));
    return weights.bbox_count>0 && weights.strength>0;
}

static bool _mimir_ends_with(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

static bool _mimir_load_mpk_full_config(lua_State* L,
                                        const std::string& mpk_path,
                                        std::string* out_err) {
    const int stack_top = lua_gettop(L);
    auto cleanup = [&]() { lua_settop(L, stack_top); };

    lua_getglobal(L, "dofile");
    if (!lua_isfunction(L, -1)) {
        cleanup();
        if (out_err) *out_err = "dofile unavailable";
        return false;
    }

    lua_pushstring(L, "scripts/modules/mpk.lua");
    if (lua_pcall(L, 1, 1, 0) != LUA_OK) {
        if (out_err) *out_err = lua_tostring(L, -1);
        lua_pop(L, 1);
        cleanup();
        return false;
    }

    if (!lua_istable(L, -1)) {
        if (out_err) *out_err = "MPK module did not return a table";
        cleanup();
        return false;
    }

    const int mpk_mod = lua_gettop(L);

    lua_getfield(L, mpk_mod, "read");
    if (!lua_isfunction(L, -1)) {
        if (out_err) *out_err = "MPK.read unavailable";
        cleanup();
        return false;
    }
    lua_pushstring(L, mpk_path.c_str());
    if (lua_pcall(L, 1, 1, 0) != LUA_OK) {
        if (out_err) *out_err = lua_tostring(L, -1);
        lua_pop(L, 1);
        cleanup();
        return false;
    }
    if (!lua_istable(L, -1)) {
        if (out_err) *out_err = "MPK.read did not return a package table";
        cleanup();
        return false;
    }
    const int pkg_idx = lua_gettop(L);

    lua_getfield(L, mpk_mod, "to_registry_full_config");
    if (!lua_isfunction(L, -1)) {
        if (out_err) *out_err = "MPK.to_registry_full_config unavailable";
        cleanup();
        return false;
    }
    lua_pushvalue(L, pkg_idx);
    if (lua_pcall(L, 1, 1, 0) != LUA_OK) {
        if (out_err) *out_err = lua_tostring(L, -1);
        lua_pop(L, 1);
        cleanup();
        return false;
    }
    if (!lua_istable(L, -1)) {
        if (out_err) *out_err = "MPK.to_registry_full_config did not return a table";
        cleanup();
        return false;
    }

    lua_replace(L, stack_top + 1);
    lua_settop(L, stack_top + 1);
    return true;
}

}  // namespace

std::size_t LuaScripting::autoRegisterMpkArchitectures(
    const std::string& project_root,
    std::vector<std::string>* warnings) {
    const std::filesystem::path arch_dir =
        std::filesystem::path(project_root) / "_archi";
    std::error_code ec;
    if (!std::filesystem::is_directory(arch_dir, ec) || ec) {
        return 0;
    }

    std::vector<std::filesystem::path> files;
    for (std::filesystem::directory_iterator it(arch_dir, ec), end;
         !ec && it != end;
         it.increment(ec)) {
        if (!it->is_regular_file(ec) || ec) continue;
        std::string filename = it->path().filename().string();
        std::transform(filename.begin(), filename.end(), filename.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (_mimir_ends_with(filename, ".mpk") ||
            _mimir_ends_with(filename, ".mpk.bin")) {
            files.push_back(it->path());
        }
    }
    if (ec) {
        if (warnings) warnings->push_back("lecture de _archi impossible: " + ec.message());
        return 0;
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) return 0;

    lua_State* loader = luaL_newstate();
    if (!loader) {
        if (warnings) warnings->push_back("initialisation Lua impossible pour charger _archi");
        return 0;
    }
    luaL_openlibs(loader);

    std::size_t loaded = 0;
    auto& registry = ModelArchitectures::Registry::instance();
    for (const auto& path : files) {
        const std::string filename = path.filename().string();
        std::string file_alias = filename;
        std::string file_alias_lower = file_alias;
        std::transform(file_alias_lower.begin(), file_alias_lower.end(), file_alias_lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const std::size_t suffix_size =
            _mimir_ends_with(file_alias_lower, ".mpk.bin") ? 8u : 4u;
        file_alias.resize(file_alias.size() - suffix_size);

        std::string err;
        if (!_mimir_load_mpk_full_config(loader, path.string(), &err)) {
            if (warnings) warnings->push_back(path.filename().string() + ": " + err);
            continue;
        }

        json full_cfg;
        try {
            full_cfg = LuaScripting::luaTableToJson(loader, -1);
        } catch (const std::exception& e) {
            lua_pop(loader, 1);
            if (warnings) warnings->push_back(path.filename().string() + ": " + e.what());
            continue;
        }
        lua_pop(loader, 1);

        const std::string base_arch = full_cfg.value("architecture", std::string());
        const json model_cfg = full_cfg.value("model", json::object());
        const json mpk_meta = full_cfg.value("mpk", json::object());
        std::string registry_name = mpk_meta.value("name", file_alias);
        const std::string description =
            mpk_meta.value("description", std::string("Architecture MPK: ") + path.filename().string());

        if (registry_name.empty() || base_arch.empty() || !model_cfg.is_object()) {
            if (warnings) warnings->push_back(path.filename().string() + ": métadonnées de registre incomplètes");
            continue;
        }
        if (registry.find(base_arch) == nullptr) {
            if (warnings) warnings->push_back(
                path.filename().string() + ": architecture de base inconnue: " + base_arch);
            continue;
        }

        // Ne jamais remplacer silencieusement une architecture native. Un MPK
        // portant le même nom devient accessible sous le nom de son fichier.
        if (registry.find(registry_name) != nullptr) {
            registry_name = file_alias;
        }
        if (registry_name.empty() || registry.find(registry_name) != nullptr) {
            if (warnings) warnings->push_back(
                path.filename().string() + ": nom déjà présent dans le registre");
            continue;
        }

        ModelArchitectures::Entry entry;
        entry.name = registry_name;
        entry.description = description + " [MPK: " + path.string() + "]";
        entry.default_config = model_cfg;
        entry.create = [base_arch](const json& cfg) {
            return ModelArchitectures::create(base_arch, cfg);
        };
        entry.origin = "mpk";
        entry.source_path = path.string();
        registry.registerArchitecture(std::move(entry));
        ++loaded;
    }

    lua_close(loader);
    return loaded;
}

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
    return true;
}

static std::vector<float> latentChunkToEmbedding(const std::vector<float>& packed,
                                                 int image_dim,
                                                 int latent_dim,
                                                 int target_dim) {
    if (target_dim <= 0 || image_dim < 0 || latent_dim <= 0) return {};
    const size_t off = static_cast<size_t>(image_dim);
    if (packed.size() < off + static_cast<size_t>(latent_dim)) return {};

    std::vector<float> out(static_cast<size_t>(target_dim), 0.0f);
    for (int d = 0; d < target_dim; ++d) {
        const int lo = (d * latent_dim) / target_dim;
        int hi = ((d + 1) * latent_dim) / target_dim;
        if (hi <= lo) hi = lo + 1;
        const int hi_clamped = std::min(latent_dim, hi);
        float acc = 0.0f;
        int cnt = 0;
        for (int i = lo; i < hi_clamped; ++i) {
            acc += packed[off + static_cast<size_t>(i)];
            ++cnt;
        }
        out[static_cast<size_t>(d)] = (cnt > 0) ? (acc / static_cast<float>(cnt)) : 0.0f;
    }

    double n2 = 0.0;
    for (float v : out) n2 += static_cast<double>(v) * static_cast<double>(v);
    if (n2 > 1e-12) {
        const float inv = static_cast<float>(1.0 / std::sqrt(n2));
        for (float& v : out) v *= inv;
    }
    return out;
}

static void mergeLuaConfigIntoModelConfig(Model& model, const json& cfg) {
    if (!cfg.is_object()) return;

    static const std::unordered_set<std::string> kOverrideKeys = {
        "recon_loss", "kl_beta", "vae_kl_beta", "kl_warmup_steps",
        "marker_wass_scale", "marker_temp_scale", "marker_scale_max", "marker_warmup_steps",
        "logvar_clip_min", "logvar_clip_max", "ssim_weight", "ssim_mode", "ssim_k1", "ssim_k2",
        "ssim_L", "spectral_weight", "spectral_scales", "perceptual_weight", "perceptual_arch",
        "perceptual_checkpoint", "perceptual_base_channels", "perceptual_prior_momentum",
        "perceptual_prior_scale", "huber_delta", "smoothl1_delta",
        "smoothl1_beta", "charbonnier_eps", "nll_sigma", "gaussian_nll_sigma", "align_weight", "bbox_loss_weight",
        "relation_primary_weight", "relation_secondary_weight", "relation_other_weight",
        "grad_clip_norm", "clip_norm", "frozen_layer_prefixes",
    };

    for (auto it = cfg.begin(); it != cfg.end(); ++it) {
        const std::string key = it.key();
        const json& value = it.value();
        if (kOverrideKeys.count(key) > 0) {
            model.modelConfig[key] = value;
            continue;
        }
        if (!model.modelConfig.contains(key)) {
            model.modelConfig[key] = value;
        }
    }
}

int LuaScripting::lua_createModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    // Argument: type de modèle (string)
    const char* model_type = luaL_checkstring(L, 1);
    const std::string model_type_str = model_type ? std::string(model_type) : std::string();

    if (_mimir_ends_with(model_type_str, ".mpk") ||
        _mimir_ends_with(model_type_str, ".mpk.bin")) {
        const std::filesystem::path mpk_path(model_type_str);
        std::error_code ec;
        if (std::filesystem::exists(mpk_path, ec) && !ec) {
            std::string err;
            if (!_mimir_load_mpk_full_config(L, model_type_str, &err)) {
                ctx.addLog("Erreur chargement MPK pour create(path): " + err);
                lua_pushboolean(L, false);
                lua_pushstring(L, err.c_str());
                return 2;
            }

            json full_cfg = LuaScripting::luaTableToJson(L, -1);
            lua_pop(L, 1);

            try {
                std::string arch;
                json cfg;
                ctx.currentModel = ModelArchitectures::createFromConfig(full_cfg, &cfg, &arch);

                if (ctx.currentModel) {
                    mergeLuaConfigIntoModelConfig(*ctx.currentModel, cfg);
                }
                if (ctx.currentTokenizer) {
                    ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
                }
                if (ctx.currentEncoder) {
                    ctx.currentModel->setEncoder(*ctx.currentEncoder);
                }

                ctx.currentConfig = full_cfg;
                ctx.modelType = arch;
                ctx.modelConfig = cfg;

                ctx.addLog("Modèle créé depuis MPK: " + model_type_str);
                lua_pushboolean(L, true);
                return 1;
            } catch (const std::exception& e) {
                ctx.addLog("Erreur création modèle depuis MPK: " + std::string(e.what()));
                lua_pushboolean(L, false);
                lua_pushstring(L, e.what());
                return 2;
            }
        }
    }
    
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
        // dans `model.modelConfig` pour que le contrat trainStep du modèle et optimizerStep les voient.
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
        ctx.modelConfig = ctx.currentModel->modelConfig;

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

int LuaScripting::lua_modelGetConfig(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    if (!ctx.currentModel) {
        lua_pushnil(L);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    jsonToLuaTable(L, ctx.currentModel->modelConfig);
    return 1;
}

int LuaScripting::lua_createEmptyModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    const char* model_type = lua_isstring(L, 1) ? lua_tostring(L, 1) : "custom_graph";

    json config = json::object();
    if (lua_istable(L, 2)) {
        config = luaTableToJson(L, 2);
        if (!config.is_object()) config = json::object();
    }

    try {
        ctx.currentModel = std::make_shared<Model>();
        if (!ctx.currentModel) {
            lua_pushboolean(L, false);
            lua_pushstring(L, "Echec creation modele vide");
            return 2;
        }

        const std::string name(model_type ? model_type : "custom_graph");
        ctx.currentModel->setModelName(name);

        ctx.modelType = name;
        ctx.modelConfig = config;
        ctx.currentModel->modelConfig = config;
        ctx.currentModel->modelConfig["type"] = name;

        if (ctx.currentTokenizer) {
            ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
        }
        if (ctx.currentEncoder) {
            ctx.currentModel->setEncoder(*ctx.currentEncoder);
        }

        ctx.addLog("Modele vide cree: " + name);
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& e) {
        ctx.addLog("Erreur create_empty: " + std::string(e.what()));
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
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

int LuaScripting::lua_modelDType(lua_State* L) {
    auto& ctx = LuaContext::getInstance();

    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }

    const int nargs = lua_gettop(L);
    if (nargs <= 0) {
        lua_pushstring(L, ctx.currentModel->getDefaultDType().c_str());
        return 1;
    }

    const char* dtype = luaL_checkstring(L, 1);
    try {
        ctx.currentModel->setDefaultDType(dtype);
        // Keep Lua context config in sync (best-effort)
        try {
            ctx.modelConfig["dtype"] = std::string(dtype);
        } catch (...) {
        }
        ctx.addLog(std::string("Model.dtype = '") + dtype + "'");
        lua_pushboolean(L, true);
        lua_pushstring(L, dtype);
        return 2;
    } catch (const std::exception& e) {
        lua_pushboolean(L, false);
        lua_pushstring(L, e.what());
        return 2;
    }
}

int LuaScripting::lua_lumenText2Img(lua_State* L) {
    const char* prompt = luaL_checkstring(L, 1);
    const int seed = static_cast<int>(luaL_optinteger(L, 2, 1337));
    const int steps = static_cast<int>(luaL_optinteger(L, 3, 30));
    const float guidance = static_cast<float>(luaL_optnumber(L, 4, 5.0));
    try {
        if (steps < 1 || !std::isfinite(guidance) || guidance < 0.0f) {
            throw std::invalid_argument("Lumen: steps must be positive and guidance finite and non-negative");
        }
        const auto image = _mimir_current_lumen_model().generate(prompt, seed, steps, guidance);
        lua_createtable(L, static_cast<int>(image.pixels.size()), 0);
        for (size_t index = 0; index < image.pixels.size(); ++index) {
            lua_pushinteger(L, image.pixels[index]);
            lua_rawseti(L, -2, static_cast<lua_Integer>(index + 1));
        }
        lua_pushinteger(L, image.w);
        lua_pushinteger(L, image.h);
        lua_pushinteger(L, image.channels);
        return 4;
    } catch (const std::exception& error) {
        lua_pushnil(L);
        lua_pushstring(L, error.what());
        return 2;
    }
}

int LuaScripting::lua_lumenBeginVaeCalibration(lua_State* L) {
    try {
        _mimir_current_lumen_model().beginVaeCalibration();
        lua_pushboolean(L, true);
        return 1;
    } catch (const std::exception& error) {
        lua_pushboolean(L, false);
        lua_pushstring(L, error.what());
        return 2;
    }
}

int LuaScripting::lua_lumenAddVaeCalibrationImage(lua_State* L) {
    try {
        const auto stats = _mimir_current_lumen_model().addVaeCalibrationImage(
            _mimir_lua_u8_array(L, 1));
        _mimir_push_lumen_calibration_stats(L, stats);
        return 1;
    } catch (const std::exception& error) {
        lua_pushnil(L);
        lua_pushstring(L, error.what());
        return 2;
    }
}

int LuaScripting::lua_lumenFinishVaeCalibration(lua_State* L) {
    try {
        const auto stats = _mimir_current_lumen_model().finishVaeCalibration();
        _mimir_push_lumen_calibration_stats(L, stats);
        return 1;
    } catch (const std::exception& error) {
        lua_pushnil(L);
        lua_pushstring(L, error.what());
        return 2;
    }
}

int LuaScripting::lua_lumenTrainStep(lua_State* L) {
    try {
        auto& ctx = LuaContext::getInstance();
        auto& lumen = _mimir_current_lumen_model();
    if (ctx.asyncMonitor) {
        ctx.asyncMonitor->bindSkipConnectionControl(ctx.currentModel->skipConnectionControl());
        ctx.currentModel->publishRuntimeConfiguration();
        ctx.asyncMonitor->bindRuntimeConfiguration(ctx.currentModel->runtimeConfiguration());
    }

        if (ctx.asyncMonitor) ctx.asyncMonitor->waitWhileTrainingPaused();
        const auto step_started_at = std::chrono::steady_clock::now();
        const std::vector<unsigned char> image = _mimir_lua_u8_array(L, 1);
        const std::string prompt = luaL_checkstring(L, 2);
        const unsigned int seed = static_cast<unsigned int>(luaL_checkinteger(L, 3));
        const float requested_learning_rate = static_cast<float>(luaL_checknumber(L, 4));
        const std::string optimizer_name = luaL_optstring(L, 5, "adamw");
        float learning_rate = requested_learning_rate;
        int lr_warmup_steps = 0;
        if (lua_istable(L, 6)) {
            lua_getfield(L, 6, "lr_warmup_steps");
            lr_warmup_steps = std::max(0, static_cast<int>(lua_tointeger(L, -1)));
            lua_pop(L, 1);
        }
        if (ctx.asyncMonitor) {
            ctx.asyncMonitor->updateRuntimeTrainParams(
                requested_learning_rate, lr_warmup_steps, 0.0f, 0, "mse");
            const auto live = ctx.asyncMonitor->liveTrainParamsSnapshot();
            if (_mimir_live_params_overrides_enabled(live) &&
                std::isfinite(live.lr) && live.lr > 0.0f) {
                learning_rate = live.lr;
                lr_warmup_steps = live.lr_warmup_steps;
            }
        }
        Optimizer& optimizer = _mimir_lumen_optimizer(
            *ctx.currentModel, optimizer_name, learning_rate);
        optimizer.warmup_steps = lr_warmup_steps;
        const bool viz_active = ctx.asyncMonitor && ctx.asyncMonitor->getViz();
        if (viz_active && !ctx.currentModel->isVizTapsEnabled()) {
            ctx.currentModel->setVizTapsEnabled(true);
        }
        json annotation;
        bool drop_conditioning = false;
        ctx.currentModel->applyRuntimeConfiguration();
        float bbox_loss_weight=ctx.currentModel->modelConfig.value("bbox_loss_weight",1.0f);
        if (lua_istable(L,6)) {
            lua_getfield(L,6,"drop_conditioning");
            drop_conditioning = lua_toboolean(L,-1);
            lua_pop(L,1);
            lua_getfield(L,6,"annotation");
            if (!lua_isnil(L,-1)) annotation=luaTableToJson(L,-1);
            lua_pop(L,1);
            lua_getfield(L,6,"bbox_loss_weight");
            if (!lua_isnil(L,-1)) { if (!lua_isnumber(L,-1)) throw std::invalid_argument("bbox_loss_weight must be numeric"); bbox_loss_weight=static_cast<float>(lua_tonumber(L,-1)); }
            lua_pop(L,1);
        }
        const auto stats = lumen.trainDiffusionStep(
            image, prompt, seed, optimizer, learning_rate,annotation.is_null()?nullptr:&annotation,bbox_loss_weight,drop_conditioning);
        const auto step_finished_at = std::chrono::steady_clock::now();

        bool stop_requested = false;
        if (ctx.asyncMonitor) {
            AsyncMonitor::Metrics metrics;
            metrics.loss = stats.loss;
            metrics.avg_loss = stats.loss;
            metrics.lr = learning_rate;
            metrics.mse = stats.mse;
            metrics.kl = stats.kl;
            metrics.kl_beta_effective = stats.kl_beta_effective;
            metrics.wass = stats.wasserstein;
            metrics.ent = stats.entropy_diff;
            metrics.mom = stats.moment_mismatch;
            metrics.spat = stats.spatial_coherence;
            metrics.temp = stats.temporal_consistency;
            metrics.timestep = static_cast<float>(stats.timestep);
            metrics.grad_norm = stats.grad_norm;
            metrics.grad_max = stats.grad_max_abs;
            metrics.params = ctx.currentModel->totalParamCount();
            metrics.recon_loss_type = "MSE";
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                step_finished_at - step_started_at).count();
            metrics.batch_time_ms = static_cast<int>(std::max<int64_t>(1, elapsed_ms));
            metrics.bps = 1000.0f / static_cast<float>(metrics.batch_time_ms);
            metrics.memory_mb = _mimir_process_resident_memory_mb();
            metrics.allocator_memory_mb =
                static_cast<double>(MemoryGuard::instance().getCurrentBytes()) / (1024.0 * 1024.0);
            metrics.opt_type = static_cast<int>(optimizer.type);
            metrics.opt_step = static_cast<int>(optimizer.step);
            metrics.opt_beta1 = optimizer.beta1;
            metrics.opt_beta2 = optimizer.beta2;
            metrics.opt_eps = optimizer.eps;
            metrics.opt_weight_decay = optimizer.weight_decay;
            if (lua_istable(L, 6)) {
                auto integer_field = [L](const char* name) {
                    lua_getfield(L, 6, name);
                    const int value = static_cast<int>(lua_tointeger(L, -1));
                    lua_pop(L, 1);
                    return value;
                };
                metrics.epoch = integer_field("epoch");
                metrics.total_epochs = integer_field("total_epochs");
                metrics.batch = integer_field("batch");
                metrics.total_batches = integer_field("total_batches");
                const int completed_steps = integer_field("completed_steps");
                lua_getfield(L, 6, "loss_sum_before");
                const double loss_sum_before = lua_tonumber(L, -1);
                lua_pop(L, 1);
                metrics.avg_loss = static_cast<float>(
                    (loss_sum_before + stats.loss) / std::max(1, completed_steps + 1));
            }
            ctx.asyncMonitor->updateMetrics(metrics);
            if (auto viz = ctx.asyncMonitor->getViz()) {
                _mimir_publish_lumen_viz(
                    ctx, lumen, image, prompt, "Lumen training sample", false, stats.bbox_count>0 && stats.bbox_loss_weight>0, &annotation, stats.bbox_loss_weight);
            }
            stop_requested = ctx.asyncMonitor->consumeStopTrainingRequested();
        }
        _mimir_push_lumen_train_stats(L, stats, stop_requested);
        lua_pushnumber(L, learning_rate);
        lua_setfield(L, -2, "learning_rate");
        return 1;
    } catch (const std::exception& error) {
        lua_pushnil(L);
        lua_pushstring(L, error.what());
        return 2;
    }
}

int LuaScripting::lua_lumenValidateStep(lua_State* L) {
    try {
        auto& ctx = LuaContext::getInstance();
        auto& lumen = _mimir_current_lumen_model();
        const std::vector<unsigned char> image = _mimir_lua_u8_array(L, 1);
        const std::string prompt = luaL_checkstring(L, 2);
        if (ctx.asyncMonitor && ctx.asyncMonitor->getViz() &&
            !ctx.currentModel->isVizTapsEnabled()) {
            ctx.currentModel->setVizTapsEnabled(true);
        }
        ctx.currentModel->clearVizTaps();
        json annotation;
        ctx.currentModel->applyRuntimeConfiguration();
        float bbox_loss_weight=ctx.currentModel->modelConfig.value("bbox_loss_weight",1.0f);
        if (lua_istable(L,4)) {
            lua_getfield(L,4,"annotation");
            if (!lua_isnil(L,-1)) annotation=luaTableToJson(L,-1);
            lua_pop(L,1);
            lua_getfield(L,4,"bbox_loss_weight");
            if (!lua_isnil(L,-1)) { if (!lua_isnumber(L,-1)) throw std::invalid_argument("bbox_loss_weight must be numeric"); bbox_loss_weight=static_cast<float>(lua_tonumber(L,-1)); }
            lua_pop(L,1);
        }
        const auto stats = lumen.validateDiffusionStep(
            image,
            prompt,
            static_cast<unsigned int>(luaL_checkinteger(L, 3)),annotation.is_null()?nullptr:&annotation,bbox_loss_weight);
        _mimir_publish_lumen_viz(
            ctx, lumen, image, prompt, "Lumen validation sample", true, stats.bbox_count>0 && stats.bbox_loss_weight>0, &annotation, stats.bbox_loss_weight);
        _mimir_push_lumen_train_stats(L, stats, false);
        return 1;
    } catch (const std::exception& error) {
        lua_pushnil(L);
        lua_pushstring(L, error.what());
        return 2;
    }
}

int LuaScripting::lua_visionTrainStep(lua_State* L) {
    auto& ctx=LuaContext::getInstance();
    if (ctx.asyncMonitor) ctx.asyncMonitor->waitWhileTrainingPaused();
    luaL_checktype(L,1,LUA_TTABLE);
    luaL_checktype(L,2,LUA_TTABLE);
    try {
        if (!ctx.currentModel) throw std::runtime_error("No model created");
        auto input=luaTableToJson(L,1).get<std::vector<float>>();
        auto target=luaTableToJson(L,2).get<std::vector<float>>();
        json options=lua_istable(L,3)?luaTableToJson(L,3):json::object();
        if (!options.is_object()) throw std::invalid_argument("vision_train_step options must be an object");
        const float lr=options.value("learning_rate",1e-4f);
        if (!std::isfinite(lr) || lr<=0) throw std::invalid_argument("learning_rate must be finite and positive");
        auto& optimizer=_mimir_lumen_optimizer(*ctx.currentModel,options.value("optimizer",std::string("adamw")),lr);
        Model::TrainStepRequest request;
        request.float_inputs["__input__"]=&input;
        request.target=&target; request.optimizer=&optimizer; request.learning_rate=lr;
        if (options.contains("annotation")) request.annotation=&options["annotation"];
        std::vector<float> weights;
        if (options.contains("loss_weights")) {
            weights=options["loss_weights"].get<std::vector<float>>(); request.loss_weights=&weights;
        }
        std::vector<int> text_ids;
        if (options.contains("text_ids")) {
            text_ids=options["text_ids"].get<std::vector<int>>(); request.int_inputs["text_ids"]=&text_ids;
        }
        const auto result=ctx.currentModel->trainStep(request);
        if (!result) throw std::runtime_error("Model has no compatible training contract");
        if (ctx.asyncMonitor && ctx.asyncMonitor->getViz()) {
            const auto& config=ctx.currentModel->modelConfig;
            const int w=config.value("image_w",0), h=config.value("image_h",0), c=config.value("image_c",3);
            if (w>0 && h>0 && (c==1 || c==3 || c==4) && input.size()==static_cast<size_t>(w)*h*c) {
                const std::string type=config.value("type",std::string("vision"));
                const bool hwc=config.value("bbox_input_layout",type=="vae_conv" || type=="vae" ||
                    type=="vgg16_feat" || type=="patch_discriminator" ? std::string("HWC") : std::string("CHW"))=="HWC";
                std::vector<uint8_t> preview(input.size());
                const size_t spatial=static_cast<size_t>(w)*h;
                for(size_t p=0;p<spatial;++p) for(int channel=0;channel<c;++channel) {
                    const float value=input[hwc?p*c+channel:static_cast<size_t>(channel)*spatial+p];
                    preview[p*c+channel]=static_cast<uint8_t>(std::clamp((value+1.f)*127.5f,0.f,255.f));
                }
                std::string prompt;
                if (request.annotation && request.annotation->is_object() && request.annotation->contains("captions"))
                    prompt=(*request.annotation)["captions"].value("primary",std::string());
                const auto text=VizTextPayload::buildDatasetTextPayload(ctx.currentModel.get(),ctx.currentTokenizer.get(),prompt,nullptr,-1);
                const auto count=result->metrics.find("bbox_count"), strength=result->metrics.find("bbox_loss_weight");
                const bool active=count!=result->metrics.end() && count->second>0 &&
                    strength!=result->metrics.end() && strength->second>0;
                ctx.asyncMonitor->setDatasetSample(preview,w,h,c,type+"/input/dataset/rgb",text.prompt,text.tags,text.tokens,text.encoding,active,DatasetBBoxOverlay::fromAnnotation(request.annotation),request.annotation,&ctx.currentModel->modelConfig);
                _mimir_publish_viz_taps(ctx,false);
            }
        }
        json stats=result->metrics;
        stats["loss"]=result->loss; stats["grad_norm"]=result->grad_norm;
        stats["grad_max_abs"]=result->grad_max_abs;
        jsonToLuaTable(L,stats); return 1;
    } catch (const std::exception& error) {
        lua_pushnil(L); lua_pushstring(L,error.what()); return 2;
    }
}

int LuaScripting::lua_trainModel(lua_State* L) {
    auto& ctx = LuaContext::getInstance();
    
    if (!ctx.currentModel) {
        lua_pushboolean(L, false);
        lua_pushstring(L, "Aucun modèle créé");
        return 2;
    }
    
    if (ctx.asyncMonitor) {
        ctx.asyncMonitor->bindSkipConnectionControl(ctx.currentModel->skipConnectionControl());
        ctx.currentModel->publishRuntimeConfiguration();
        ctx.asyncMonitor->bindRuntimeConfiguration(ctx.currentModel->runtimeConfiguration());
    }

    // Arguments: epochs (number), learning_rate (number)
    int epochs = luaL_checkinteger(L, 1);
    double lr = luaL_checknumber(L, 2);
    
    ctx.addLog("Entraînement: " + std::to_string(epochs) + " epochs, LR=" + std::to_string(lr));

    try {
        // Si la viz est active, activer les "viz taps" côté modèle et préparer
        // un pont vers AsyncMonitor pour afficher les blocs/layers.
        bool viz_active = false;
        auto refresh_viz_activation = [&]() {
            if (viz_active || !ctx.asyncMonitor || !ctx.currentModel ||
                ctx.asyncMonitor->getViz() == nullptr) {
                return;
            }

            viz_active = true;
            ctx.currentModel->setVizTapsEnabled(true);
            try {
                int max_frames = 12;
                int max_side = 64;
                if (ctx.modelConfig.contains("viz_taps_max_frames")) max_frames = ctx.modelConfig["viz_taps_max_frames"].get<int>();
                if (ctx.modelConfig.contains("viz_taps_max_side")) max_side = ctx.modelConfig["viz_taps_max_side"].get<int>();
                ctx.currentModel->setVizTapsLimits(max_frames, max_side);
            } catch (...) {
            }
        };
        refresh_viz_activation();

        // Instancier l'Optimizer à partir de la configuration
        Optimizer opt;
        opt.initial_lr = static_cast<float>(lr);
        opt.type = OptimizerType::ADAMW;
        configureOptimizerFromJson(opt, ctx.modelConfig);
        
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
            else throw std::runtime_error("decay_strategy invalide: " + s);
        }

        // Reprise éventuelle de l'état optimiseur (checkpoint)
        const Optimizer* saved_opt = ctx.currentModel->getSerializedOptimizer();
        const bool resume_optimizer_loaded = (saved_opt != nullptr);
        if (saved_opt) {
            opt = *saved_opt;
            // S'assurer que lr reflète l'argument (opt.initial_lr sert de base au scheduler)
            opt.initial_lr = static_cast<float>(lr);

            // La reprise conserve les moments et le step, mais la configuration
            // d'entraînement courante reste autoritaire pour le scheduler.
            if (ctx.modelConfig.contains("min_lr")) opt.min_lr = ctx.modelConfig["min_lr"];
            if (ctx.modelConfig.contains("decay_rate")) opt.decay_rate = ctx.modelConfig["decay_rate"];
            if (ctx.modelConfig.contains("decay_steps")) opt.decay_steps = ctx.modelConfig["decay_steps"];
            if (ctx.modelConfig.contains("warmup_steps")) opt.warmup_steps = ctx.modelConfig["warmup_steps"];
            if (ctx.modelConfig.contains("decay_strategy")) {
                std::string s = ctx.modelConfig["decay_strategy"].get<std::string>();
                std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
                if (s == "none") opt.decay_strategy = LRDecayStrategy::NONE;
                else if (s == "cosine") opt.decay_strategy = LRDecayStrategy::COSINE;
                else if (s == "step") opt.decay_strategy = LRDecayStrategy::STEP;
                else if (s == "exponential") opt.decay_strategy = LRDecayStrategy::EXPONENTIAL;
                else if (s == "linear") opt.decay_strategy = LRDecayStrategy::LINEAR;
                else throw std::runtime_error("decay_strategy invalide: " + s);
            }
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
        if (checkpoint_dir.empty()) {
            std::string safe_model_type = model_type.empty() ? std::string("model") : model_type;
            for (char& character : safe_model_type) {
                const unsigned char value = static_cast<unsigned char>(character);
                if (!(std::isalnum(value) || character == '_' || character == '-')) character = '_';
            }
            checkpoint_dir = (std::filesystem::path("checkpoints") /
                              "emergency" / safe_model_type).string();
        }

        // -----------------------------------------------------------------
        // CSV : csv_file / csv_path / csv_dir depuis la config du modèle.
        // Appliqué ICI (commun à tous les types) sur le HtopDisplay ET le Visualizer.
        // Une section spécifique peut encore surcharger la destination via csv_dir.
        // (pattern partN_epochM.csv) si aucun csv_file explicite n'est fourni.
        // -----------------------------------------------------------------
        {
            std::string csv_cfg_file;
            std::string csv_cfg_dir;
            try {
                if (ctx.modelConfig.contains("csv_file"))        csv_cfg_file = ctx.modelConfig["csv_file"].get<std::string>();
                else if (ctx.modelConfig.contains("csv_path"))   csv_cfg_file = ctx.modelConfig["csv_path"].get<std::string>();
                if (ctx.modelConfig.contains("csv_dir"))         csv_cfg_dir  = ctx.modelConfig["csv_dir"].get<std::string>();
            } catch (...) {}

            // Résoudre le chemin final pour le Visualizer (si csv_file vide, construire depuis csv_dir)
            std::string csv_viz_file;  // chemin Visualizer résolu
            if (!csv_cfg_file.empty()) {
                csv_viz_file = csv_cfg_file;
            } else if (!csv_cfg_dir.empty()) {
                namespace fs = std::filesystem;
                std::error_code ec;
                fs::create_directories(csv_cfg_dir, ec);
                std::string base_name;
                try {
                    if (ctx.modelConfig.contains("name"))            base_name = ctx.modelConfig["name"].get<std::string>();
                    else if (ctx.modelConfig.contains("model_name")) base_name = ctx.modelConfig["model_name"].get<std::string>();
                } catch (...) {}
                if (base_name.empty()) base_name = ctx.modelType;
                if (base_name.empty()) base_name = "model";
                for (char& c : base_name) {
                    const unsigned char uc = static_cast<unsigned char>(c);
                    if (!(std::isalnum(uc) || c == '_' || c == '-')) c = '_';
                }
                csv_viz_file = (fs::path(csv_cfg_dir) / (base_name + "_loss_history.csv")).string();
            }

            if (ctx.asyncMonitor && !csv_viz_file.empty()) {
                ctx.asyncMonitor->configureMetricsCsv(csv_viz_file, true);
                ctx.addLog("CSV metrics (cfg): " + csv_viz_file);
            }
        }

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

            // Le tokenizer est enrichi pendant l'entraînement (tokenizeEnsure).
            // Le resynchroniser juste avant save garantit que le checkpoint contient
            // le vocabulaire effectivement composé avec le dataset.
            if (ctx.currentTokenizer) {
                ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
            }

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
        if (recon_loss_type.empty()) recon_loss_type = "mse";

        auto recon_metric_label = [&]() -> std::string {
            std::string t = recon_loss_type;
            std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
            if (t.empty() || t == "mse") return "mse";
            if (t == "mae") return "l1";
            if (t == "cross_entropy") return "ce";
            if (t == "nll_gaussian") return "gaussian_nll";
            return t;
        };

        int global_step = 0;

        // -----------------------------------------------------------------
        // Live tuning via Viz (thread UI -> training thread)
        // -----------------------------------------------------------------
        uint64_t last_live_ver = 0;
        bool live_override_active = false;

        // Baselines (pour retour au comportement natif)
        const float baseline_lr = static_cast<float>(lr);
        const int baseline_lr_warmup_steps = std::max(0, opt.warmup_steps);
        const std::string baseline_recon_loss = recon_loss_type;
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

        // ─────────────────────────────────────────────────────────────────────
        // Calibration par feedback de validation (récompense / punition sur LR)
        // Clés modelConfig: val_feedback_enabled, val_reward_factor,
        //   val_penalty_factor, val_lr_scale_min, val_lr_scale_max,
        //   val_improve_thresh, val_feedback_min_steps
        // La métrique utilisée est "inférieure = meilleure" (recon_loss, eps_mse, bce_loss…)
        // ─────────────────────────────────────────────────────────────────────
        bool  val_feedback_enabled   = true;
        float val_reward_factor      = 1.05f;   // boost LR si val s'améliore
        float val_penalty_factor     = 0.70f;   // réduction LR si val se dégrade
        float val_lr_scale_min       = 0.10f;   // plancher du facteur LR
        float val_lr_scale_max       = 1.50f;   // plafond du facteur LR
        float val_improve_thresh     = 0.001f;  // amélior. rel. min pour déclencher reward
        int   val_feedback_min_steps = 0;       // activer seulement après N steps
        float val_lr_scale           = 1.0f;    // facteur courant (multiplicateur step_lr)
        float val_best_metric        = std::numeric_limits<float>::max();
        try {
            if (ctx.modelConfig.contains("val_feedback_enabled"))   val_feedback_enabled   = ctx.modelConfig["val_feedback_enabled"].get<bool>();
            if (ctx.modelConfig.contains("val_reward_factor"))      val_reward_factor      = ctx.modelConfig["val_reward_factor"].get<float>();
            if (ctx.modelConfig.contains("val_penalty_factor"))     val_penalty_factor     = ctx.modelConfig["val_penalty_factor"].get<float>();
            if (ctx.modelConfig.contains("val_lr_scale_min"))       val_lr_scale_min       = ctx.modelConfig["val_lr_scale_min"].get<float>();
            if (ctx.modelConfig.contains("val_lr_scale_max"))       val_lr_scale_max       = ctx.modelConfig["val_lr_scale_max"].get<float>();
            if (ctx.modelConfig.contains("val_improve_thresh"))     val_improve_thresh     = ctx.modelConfig["val_improve_thresh"].get<float>();
            if (ctx.modelConfig.contains("val_feedback_min_steps")) val_feedback_min_steps = ctx.modelConfig["val_feedback_min_steps"].get<int>();
        } catch (...) {}

        VaeValidationFeedback::Config vae_feedback_config;
        vae_feedback_config.enabled = val_feedback_enabled;
        vae_feedback_config.lr_reward_factor = val_reward_factor;
        vae_feedback_config.lr_penalty_factor = val_penalty_factor;
        vae_feedback_config.lr_scale_min = val_lr_scale_min;
        vae_feedback_config.lr_scale_max = val_lr_scale_max;
        vae_feedback_config.improve_threshold = val_improve_thresh;
        vae_feedback_config.min_steps = val_feedback_min_steps;
        vae_feedback_config.initial_kl_beta = baseline_kl_beta;
        vae_feedback_config.kl_beta_min = baseline_kl_beta * 0.05f;
        vae_feedback_config.kl_beta_max = baseline_kl_beta * 2.0f;
        try {
            if (ctx.modelConfig.contains("val_kl_feedback_enabled")) vae_feedback_config.kl_enabled = ctx.modelConfig["val_kl_feedback_enabled"].get<bool>();
            if (ctx.modelConfig.contains("val_kl_beta_min")) vae_feedback_config.kl_beta_min = ctx.modelConfig["val_kl_beta_min"].get<float>();
            if (ctx.modelConfig.contains("val_kl_beta_max")) vae_feedback_config.kl_beta_max = ctx.modelConfig["val_kl_beta_max"].get<float>();
            if (ctx.modelConfig.contains("val_kl_target_min")) vae_feedback_config.kl_target_min = ctx.modelConfig["val_kl_target_min"].get<float>();
            if (ctx.modelConfig.contains("val_kl_target_max")) vae_feedback_config.kl_target_max = ctx.modelConfig["val_kl_target_max"].get<float>();
            if (ctx.modelConfig.contains("val_kl_reward_factor")) vae_feedback_config.kl_reward_factor = ctx.modelConfig["val_kl_reward_factor"].get<float>();
            if (ctx.modelConfig.contains("val_kl_penalty_factor")) vae_feedback_config.kl_penalty_factor = ctx.modelConfig["val_kl_penalty_factor"].get<float>();
            if (ctx.modelConfig.contains("val_kl_plateau_factor")) vae_feedback_config.kl_plateau_factor = ctx.modelConfig["val_kl_plateau_factor"].get<float>();
            if (ctx.modelConfig.contains("val_kl_collapse_factor")) vae_feedback_config.kl_collapse_factor = ctx.modelConfig["val_kl_collapse_factor"].get<float>();
            if (ctx.modelConfig.contains("val_kl_excess_factor")) vae_feedback_config.kl_excess_factor = ctx.modelConfig["val_kl_excess_factor"].get<float>();
        } catch (...) {}
        VaeValidationFeedback vae_validation_feedback(vae_feedback_config);

        auto poll_viz_live_params = [&]() {
            if (ctx.asyncMonitor) ctx.asyncMonitor->waitWhileTrainingPaused();
            ctx.currentModel->applyRuntimeConfiguration();
            if (!ctx.asyncMonitor) return;
            const uint64_t ver = ctx.asyncMonitor->liveTrainParamsVersion();
            if (ver == 0 || ver == last_live_ver) return;
            last_live_ver = ver;

            const auto p = ctx.asyncMonitor->liveTrainParamsSnapshot();
            if (p.version == 0) return;

            // Si on repasse en mode natif, restaurer la baseline une fois.
            if (!_mimir_live_params_overrides_enabled(p)) {
                if (live_override_active) {
                    live_override_active = false;
                    opt.initial_lr = std::max(1e-12f, baseline_lr);
                    opt.warmup_steps = baseline_lr_warmup_steps;

                    ctx.modelConfig["kl_beta"] = baseline_kl_beta;
                    ctx.modelConfig["kl_warmup_steps"] = baseline_kl_warmup_steps;
                    ctx.modelConfig["recon_loss"] = baseline_recon_loss;
                    recon_loss_type = baseline_recon_loss;
                    if (ctx.currentModel) {
                        ctx.currentModel->modelConfig["kl_beta"] = baseline_kl_beta;
                        ctx.currentModel->modelConfig["kl_warmup_steps"] = baseline_kl_warmup_steps;
                        ctx.currentModel->modelConfig["recon_loss"] = baseline_recon_loss;
                    }

                }
                return;
            }

            live_override_active = true;

            if (std::isfinite(p.lr) && p.lr > 0.0f) {
                opt.initial_lr = std::max(1e-12f, p.lr);
            }
            opt.warmup_steps = std::max(0, p.lr_warmup_steps);
            ctx.currentModel->modelConfig["learning_rate"] = opt.initial_lr;
            ctx.currentModel->modelConfig["lr_warmup_steps"] = opt.warmup_steps;

            const float kl_beta = (p.kl_enabled ? std::max(0.0f, p.kl_beta) : 0.0f);
            const int kl_warmup_steps = std::max(0, p.kl_warmup_steps);

            // Propager vers la config runtime consommée par le contrat trainStep du modèle.
            ctx.modelConfig["kl_beta"] = kl_beta;
            ctx.modelConfig["kl_warmup_steps"] = kl_warmup_steps;
            ctx.modelConfig["recon_loss"] = p.recon_loss;
            recon_loss_type = p.recon_loss;
            if (ctx.currentModel) {
                ctx.currentModel->modelConfig["kl_beta"] = kl_beta;
                ctx.currentModel->modelConfig["kl_warmup_steps"] = kl_warmup_steps;
                ctx.currentModel->modelConfig["recon_loss"] = p.recon_loss;
            }

        };

        auto optimizer_learning_rate = [&]() -> float {
            return opt.initial_lr * val_lr_scale;
        };

        auto step_learning_rate = [&]() -> float {
            return opt.getCurrentLR() * val_lr_scale;
        };

        auto train_metric = [](const Model::TrainStepResult& result, const char* name, float fallback = 0.0f) {
            const auto it = result.metrics.find(name);
            return it == result.metrics.end() ? fallback : it->second;
        };

        auto update_viz_runtime_params = [&](const Model::TrainStepResult* stats = nullptr) {
            if (!ctx.asyncMonitor) return;
            float kl_beta = 0.0f;
            int kl_warmup_steps = 0;
            try {
                if (ctx.modelConfig.contains("kl_beta")) kl_beta = ctx.modelConfig["kl_beta"].get<float>();
                else if (ctx.modelConfig.contains("vae_kl_beta")) kl_beta = ctx.modelConfig["vae_kl_beta"].get<float>();
                if (ctx.modelConfig.contains("kl_warmup_steps")) kl_warmup_steps = ctx.modelConfig["kl_warmup_steps"].get<int>();
            } catch (...) {}
            const float displayed_kl_beta = stats
                ? std::max(0.0f, train_metric(*stats, "kl_beta_effective"))
                : std::max(0.0f, kl_beta);
            ctx.asyncMonitor->updateRuntimeTrainParams(
                step_learning_rate(), opt.warmup_steps,
                displayed_kl_beta, kl_warmup_steps, recon_loss_type);
        };

        // Feedback de validation: récompense ou punit le modèle en ajustant val_lr_scale.
        // metric : valeur scalaire (inférieure = meilleure).
        auto apply_val_feedback = [&](float metric, int step) {
            if (!val_feedback_enabled) return;
            if (step < val_feedback_min_steps) return;
            if (!std::isfinite(metric) || metric < 0.0f) return;

            const float prev = val_best_metric;
            const bool is_first = (prev == std::numeric_limits<float>::max());

            if (is_first) {
                val_best_metric = metric;
                ctx.addLog("[val_feedback] premier point metric=" + std::to_string(metric) +
                           " lr_scale=" + std::to_string(val_lr_scale));
                return;
            }

            const float denom = std::max(1e-12f, std::abs(prev));
            const float rel   = (prev - metric) / denom; // positif = amélioration

            if (rel > val_improve_thresh) {
                // Récompense: la validation s'améliore
                val_lr_scale    = std::min(val_lr_scale_max, val_lr_scale * val_reward_factor);
                val_best_metric = metric;
                ctx.addLog("[val_feedback] ✓ reward: prev=" + std::to_string(prev) +
                           " cur=" + std::to_string(metric) +
                           " rel_improve=" + std::to_string(rel) +
                           " -> lr_scale=" + std::to_string(val_lr_scale));
            } else if (rel < -val_improve_thresh) {
                // Punition: la validation se dégrade
                val_lr_scale = std::max(val_lr_scale_min, val_lr_scale * val_penalty_factor);
                ctx.addLog("[val_feedback] ✗ penalty: prev=" + std::to_string(prev) +
                           " cur=" + std::to_string(metric) +
                           " rel_degrade=" + std::to_string(-rel) +
                           " -> lr_scale=" + std::to_string(val_lr_scale));
            } else {
                // Plateau / variation trop faible: neutre
                ctx.addLog("[val_feedback] ~ plateau: prev=" + std::to_string(prev) +
                           " cur=" + std::to_string(metric) +
                           " lr_scale=" + std::to_string(val_lr_scale));
            }
        };

        auto apply_vae_val_feedback = [&](float reconstruction, float kl, int step) -> std::string {
            const auto decision = vae_validation_feedback.update(
                reconstruction, kl, step, !live_override_active);
            val_lr_scale = decision.lr_scale;
            if (decision.kl_changed && !live_override_active) {
                ctx.modelConfig["kl_beta"] = decision.kl_beta;
                if (ctx.currentModel) ctx.currentModel->modelConfig["kl_beta"] = decision.kl_beta;
            }
            update_viz_runtime_params();
            if (ctx.asyncMonitor) {
                auto viz = ctx.asyncMonitor->getViz();
                if (viz) {
                    viz->showValidationFeedback(
                        decision.rewarded ? Visualizer::ValidationFeedbackIcon::Reward :
                        decision.penalized ? Visualizer::ValidationFeedbackIcon::Penalty :
                                  Visualizer::ValidationFeedbackIcon::None);
                }
            }
            if (decision.action == VaeValidationFeedback::Action::Ignored) return "none";

            const char* action = "first";
            switch (decision.action) {
                case VaeValidationFeedback::Action::Reward: action = "reward"; break;
                case VaeValidationFeedback::Action::Penalty: action = "penalty"; break;
                case VaeValidationFeedback::Action::Plateau: action = "plateau"; break;
                case VaeValidationFeedback::Action::KlCollapse: action = "kl_collapse"; break;
                case VaeValidationFeedback::Action::KlExcess: action = "kl_excess"; break;
                default: break;
            }
            ctx.addLog(std::string("[vae_val_feedback] action=") + action +
                       " recon=" + std::to_string(reconstruction) +
                       " kl=" + std::to_string(kl) +
                       " rel=" + std::to_string(decision.relative_change) +
                       " lr_scale=" + std::to_string(decision.lr_scale) +
                       " kl_beta=" + std::to_string(decision.kl_beta) +
                       (live_override_active ? " kl_override=viz" : ""));
            return action;
        };

        // Perf stats pour la viz: time/mem/bps (best-effort).
        // - time: ms entre 2 updates successifs (approx temps/batch)
        // - bps: batches/sec (approx)
        // - mem: mémoire réellement résidente du processus (RSS) en MB
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

            m.memory_mb = _mimir_process_resident_memory_mb();
            m.allocator_memory_mb =
                static_cast<double>(MemoryGuard::instance().getCurrentBytes()) / (1024.0 * 1024.0);
        };

        auto log_step = [&](int global_step, const Model::TrainStepResult& st, const char* prefix) {
            if ((global_step % log_every) != 0) return;
            const std::string recon_label = recon_metric_label();
            ctx.addLog(std::string(prefix) +
                       " step=" + std::to_string(global_step) +
                       " loss=" + std::to_string(st.loss) +
                       " " + recon_label + "=" + std::to_string(train_metric(st, "mse", st.loss)) +
                       " kl=" + std::to_string(train_metric(st, "kl")) +
                       " beta_eff=" + std::to_string(train_metric(st, "kl_beta_effective")) +
                       " grad_norm=" + std::to_string(st.grad_norm));
        };

        auto monitor_step = [&](int epoch_1based, int batch_1based, int total_batches, const Model::TrainStepResult& st) {
            if (!ctx.asyncMonitor) return;

            update_viz_runtime_params(&st);

            AsyncMonitor::Metrics m;
            m.epoch = epoch_offset + epoch_1based;
            m.total_epochs = total_epochs_display;
            m.batch = batch_1based;
            m.total_batches = total_batches;
            m.loss = st.loss;
            m.avg_loss = st.loss;
            m.lr = step_learning_rate();
            m.mse = train_metric(st, "mse", st.loss);
            m.kl = train_metric(st, "kl");
            m.kl_beta_effective = train_metric(st, "kl_beta_effective");
            m.wass = train_metric(st, "wass", train_metric(st, "wasserstein"));
            m.spat = train_metric(st, "spatial_coherence");
            m.temp = train_metric(st, "temp", train_metric(st, "temporal_consistency"));
            m.timestep = train_metric(st, "timestep");
            m.grad_norm = st.grad_norm;
            m.grad_max = st.grad_max_abs;
            m.ent = train_metric(st, "entropy_diff");
            m.mom = train_metric(st, "moment_mismatch");
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
            bool graph_text_cond = false;
            bool requested_text_cond = false;
            bool text_cond_runtime = false;
            bool text_pipeline_enabled = false;
            bool semantic_label_study = false;
            int seq_len = 64;
            int pad_id = 0;
            float text_encoder_lr = 0.01f;
            float text_special_lr = 0.005f;
            int semantic_label_topk = 6;
            int semantic_label_min_df = 2;
            try {
                if (ctx.modelConfig.contains("image_w")) image_w = ctx.modelConfig["image_w"].get<int>();
                if (ctx.modelConfig.contains("image_h")) image_h = ctx.modelConfig["image_h"].get<int>();
                if (ctx.modelConfig.contains("image_c")) image_c = ctx.modelConfig["image_c"].get<int>();
                // text_cond: use model's own config (ground truth of what was actually built),
                // not the Lua config which may differ (e.g. vae_conv hardcodes text_cond=false).
                if (ctx.currentModel && ctx.currentModel->modelConfig.contains("text_cond"))
                    graph_text_cond = ctx.currentModel->modelConfig["text_cond"].get<bool>();
                else if (ctx.modelConfig.contains("text_cond"))
                    graph_text_cond = ctx.modelConfig["text_cond"].get<bool>();
                if (ctx.modelConfig.contains("text_cond"))
                    requested_text_cond = ctx.modelConfig["text_cond"].get<bool>();
                if (ctx.modelConfig.contains("seq_len")) seq_len = ctx.modelConfig["seq_len"].get<int>();
                if (ctx.modelConfig.contains("text_encoder_lr")) text_encoder_lr = ctx.modelConfig["text_encoder_lr"].get<float>();
                if (ctx.modelConfig.contains("text_special_lr")) text_special_lr = ctx.modelConfig["text_special_lr"].get<float>();
                if (ctx.modelConfig.contains("semantic_label_study")) semantic_label_study = ctx.modelConfig["semantic_label_study"].get<bool>();
                if (ctx.modelConfig.contains("semantic_label_topk")) semantic_label_topk = ctx.modelConfig["semantic_label_topk"].get<int>();
                if (ctx.modelConfig.contains("semantic_label_min_df")) semantic_label_min_df = ctx.modelConfig["semantic_label_min_df"].get<int>();
            } catch (...) {
            }

            text_cond_runtime = (!graph_text_cond && requested_text_cond);
            text_pipeline_enabled = (graph_text_cond || text_cond_runtime);
            if (text_pipeline_enabled && !ctx.modelConfig.contains("semantic_label_study")) {
                semantic_label_study = true;
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
            text_encoder_lr = std::max(0.0f, text_encoder_lr);
            text_special_lr = std::max(0.0f, text_special_lr);
            semantic_label_topk = std::clamp(semantic_label_topk, 1, 32);
            semantic_label_min_df = std::max(1, semantic_label_min_df);

            auto extract_label_terms = [](const std::string& raw) -> std::vector<std::string> {
                std::string norm;
                norm.reserve(raw.size());
                bool last_space = true;
                for (char ch : raw) {
                    const unsigned char uc = static_cast<unsigned char>(ch);
                    bool sep = false;
                    switch (ch) {
                        case '.': case ',': case ';': case ':': case '|':
                        case '/': case '\\': case '_': case '-':
                        case '(': case ')': case '[': case ']':
                        case '{': case '}': case '"': case '\'':
                        case '\t': case '\n': case '\r':
                            sep = true;
                            break;
                        default:
                            sep = false;
                            break;
                    }
                    if (sep) {
                        if (!last_space) {
                            norm.push_back(' ');
                            last_space = true;
                        }
                        continue;
                    }
                    char outc = ch;
                    if (uc < 128) outc = static_cast<char>(std::tolower(uc));
                    if (std::isspace(static_cast<unsigned char>(outc))) {
                        if (!last_space) {
                            norm.push_back(' ');
                            last_space = true;
                        }
                    } else {
                        norm.push_back(outc);
                        last_space = false;
                    }
                }
                if (!norm.empty() && norm.back() == ' ') norm.pop_back();

                std::vector<std::string> terms;
                std::string cur;
                for (char c : norm) {
                    if (c == ' ') {
                        if (!cur.empty()) {
                            terms.push_back(cur);
                            cur.clear();
                        }
                    } else {
                        cur.push_back(c);
                    }
                }
                if (!cur.empty()) terms.push_back(cur);

                // Dedup en conservant l'ordre.
                std::unordered_set<std::string> seen;
                std::vector<std::string> uniq;
                uniq.reserve(terms.size());
                for (const auto& t : terms) {
                    if (t.empty()) continue;
                    if (seen.insert(t).second) uniq.push_back(t);
                }
                return uniq;
            };

            std::unordered_map<std::string, int> semantic_df;

            auto enrich_prompt_semantic = [&](const std::string& raw) -> std::string {
                if (!semantic_label_study) return raw;
                auto terms = extract_label_terms(raw);
                if (terms.empty()) return raw;

                struct TermSc {
                    std::string t;
                    int df;
                };
                std::vector<TermSc> scored;
                scored.reserve(terms.size());
                for (const auto& t : terms) {
                    auto it = semantic_df.find(t);
                    const int df = (it == semantic_df.end()) ? 1 : std::max(1, it->second);
                    if (df >= semantic_label_min_df) scored.push_back({t, df});
                }
                if (scored.empty()) {
                    for (const auto& t : terms) scored.push_back({t, 1});
                }

                std::stable_sort(scored.begin(), scored.end(), [](const TermSc& a, const TermSc& b) {
                    if (a.df != b.df) return a.df > b.df;
                    return a.t.size() > b.t.size();
                });

                std::string suffix;
                const int k = std::min<int>((int)scored.size(), semantic_label_topk);
                for (int i = 0; i < k; ++i) {
                    if (!suffix.empty()) suffix.push_back(' ');
                    suffix += scored[(size_t)i].t;
                }
                if (suffix.empty()) return raw;
                return raw + " ; semantic " + suffix;
            };

            if (text_pipeline_enabled) {
                if (!ctx.currentTokenizer) {
                    lua_pushboolean(L, false);
                    lua_pushstring(L, "text_cond=true mais aucun tokenizer n'est chargé");
                    return 2;
                }
                pad_id = ctx.currentTokenizer->getPadId();
            }

            if (text_cond_runtime) {
                ctx.addLog("ℹ️ vae_conv: text_cond runtime actif (graphe image-only conservé). Tokenizer+ConditioningEncoder seront entraînés pour l'alignement image↔texte.");
            }

            // Validation config (best-effort, optional)
            int validate_every_steps = 0;
            int validate_items = 0;
            float validate_holdout_frac = 0.0f;
            int validate_holdout_items = 0;
            bool validate_holdout = true;
            bool fpga_validation_enabled = true;
            int fpga_validation_every_steps = 0;
            try {
                if (ctx.modelConfig.contains("validate_every_steps")) validate_every_steps = std::max(0, ctx.modelConfig["validate_every_steps"].get<int>());
                if (ctx.modelConfig.contains("validate_items")) validate_items = std::max(0, ctx.modelConfig["validate_items"].get<int>());
                if (ctx.modelConfig.contains("validate_holdout_frac")) validate_holdout_frac = ctx.modelConfig["validate_holdout_frac"].get<float>();
                if (ctx.modelConfig.contains("validate_holdout_items")) validate_holdout_items = std::max(0, ctx.modelConfig["validate_holdout_items"].get<int>());
                if (ctx.modelConfig.contains("validate_holdout")) validate_holdout = ctx.modelConfig["validate_holdout"].get<bool>();
                if (ctx.modelConfig.contains("fpga_validation_enabled")) fpga_validation_enabled = ctx.modelConfig["fpga_validation_enabled"].get<bool>();
                if (ctx.modelConfig.contains("fpga_validation_every_steps")) fpga_validation_every_steps = std::max(0, ctx.modelConfig["fpga_validation_every_steps"].get<int>());
            } catch (...) {
            }
            if (fpga_validation_every_steps <= 0) fpga_validation_every_steps = validate_every_steps;
#if !defined(MIMIR_ENABLE_FPGA_RUNTIME)
            (void)fpga_validation_enabled;
#endif
            if (ctx.asyncMonitor) {
                if (auto viz = ctx.asyncMonitor->getViz()) {
                    viz->updateRuntimeValidationEnabled(
                        validate_every_steps > 0 && validate_items > 0);
                }
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
                if (!ctx.currentDataset[i].image_file.empty() ||
                    !ctx.currentDataset[i].video_file.empty()) {
                    indices.push_back((int)i);
                }
            }
            if (indices.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Dataset: aucun item image ou vidéo (requis pour vae_conv)");
                return 2;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
            if (max_items > 0 && (int)indices.size() > max_items) indices.resize((size_t)max_items);

            // Composer explicitement le tokenizer avec le dataset complet
            // avant l'entraînement, pour tout pipeline textuel.
            if (text_pipeline_enabled && ctx.currentTokenizer) {
                int composed_items = 0;
                std::vector<std::string> conditioning_corpus;
                for (size_t di = 0; di < ctx.currentDataset.size(); ++di) {
                    DatasetItem& ditem = ctx.currentDataset[di];
                    DatasetItemUnloadGuard unload_ditem(ditem);
                    if (!ditem.text_file.empty() && !ditem.text.has_value()) {
                        ditem.loadText();
                    }
                    const std::string dprompt = ditem.text.has_value() ? ditem.text.value() : std::string();
                    if (!dprompt.empty()) {
                        if (semantic_label_study) {
                            const auto terms = extract_label_terms(dprompt);
                            std::unordered_set<std::string> seen_doc;
                            for (const auto& t : terms) {
                                if (seen_doc.insert(t).second) semantic_df[t] += 1;
                            }
                        }
                        const auto context=BBoxSupervision::groundedPrompt(dprompt,ditem.annotation ? &*ditem.annotation : nullptr);
                        ctx.currentTokenizer->ensureVocabFromText(context);
                        conditioning_corpus.push_back(context);
                        ++composed_items;
                    }
                }

                // Ensure the exact final conditioning, including semantic suffixes,
                // before BPE; keep existing IDs and learn once on the whole corpus.
                for(auto& context:conditioning_corpus) {
                    context=enrich_prompt_semantic(context);
                    ctx.currentTokenizer->ensureVocabFromText(context);
                }
                ctx.currentTokenizer->learnBPEFromCorpus(conditioning_corpus);
                ctx.addLog("Tokenizer text conditioning: ensure + learn_bpe, corpus="+std::to_string(conditioning_corpus.size()));

                if (semantic_label_study) {
                    ctx.addLog("ℹ️  Semantic label study actif: concepts=" + std::to_string((int)semantic_df.size()) +
                               " topk=" + std::to_string(semantic_label_topk) +
                               " min_df=" + std::to_string(semantic_label_min_df));
                }

                if (ctx.currentModel) {
                    ctx.currentModel->setTokenizer(*ctx.currentTokenizer);
                }

                if (!checkpoint_dir.empty()) {
                    try {
                        namespace fs = std::filesystem;
                        const fs::path tok_path = fs::path(checkpoint_dir) / "tokenizer" / "tokenizer.json";
                        std::error_code ec_tok;
                        fs::create_directories(tok_path.parent_path(), ec_tok);

                        std::ofstream tok_out(tok_path);
                        if (tok_out.is_open()) {
                            const json jtok = ctx.currentTokenizer->to_json();
                            tok_out << jtok.dump(2);
                            tok_out.close();
                            ctx.addLog("✓ Tokenizer sauvegardé après composition complète dataset: " + tok_path.string() +
                                       " (items=" + std::to_string(composed_items) +
                                       ", vocab=" + std::to_string(ctx.currentTokenizer->getVocabSize()) + ")");
                        } else {
                            ctx.addLog("⚠️  Échec ouverture fichier tokenizer après composition complète dataset: " + tok_path.string());
                        }
                    } catch (...) {
                        ctx.addLog("⚠️  Échec sauvegarde tokenizer après composition complète dataset");
                    }
                }
            }

            // Split train/val (holdout) only if validation is enabled.
            std::vector<int> train_indices = indices;
            std::vector<int> val_indices;
            if (validate_every_steps > 0 && validate_items > 0 && (int)indices.size() >= 2) {
                if (validate_holdout) {
                    const int n = (int)indices.size();
                    int val_n = 0;
                    if (validate_holdout_items > 0 && validate_holdout_items < n) {
                        val_n = validate_holdout_items;
                    } else if (validate_holdout_items >= n) {
                        const float fraction = validate_holdout_frac > 0.0f ? validate_holdout_frac : 0.2f;
                        val_n = std::clamp((int)std::floor((double)n * fraction), 1, n - 1);
                        ctx.addLog("Validation holdout: requested items=" + std::to_string(validate_holdout_items) +
                                   " exceeds dataset=" + std::to_string(n) + "; using fraction=" +
                                   std::to_string(fraction) + " (val=" + std::to_string(val_n) + ")");
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

            // Le probe FPGA faisait partie du snapshot de référence, mais il
            // dépend de l'extension Model/FpgaRuntime optionnelle. Conserver la
            // boucle d'entraînement identique et compiler le probe seulement
            // lorsque cette extension est activée dans le cœur courant.
#if defined(MIMIR_ENABLE_FPGA_RUNTIME)
            constexpr size_t fpga_validation_rows = 8;
            constexpr size_t fpga_validation_columns = 64;
            std::vector<int8_t> fpga_validation_weights;
            std::unique_ptr<FpgaValidationRunner> fpga_validation_runner;
            bool fpga_validation_active = false;
            std::string fpga_validation_log_path;

            if (fpga_validation_enabled && fpga_validation_every_steps > 0 &&
                !val_indices.empty() && image_c == 3 && ctx.currentModel &&
                ctx.currentModel->initializeFpgaComputeEngine()) {
                FpgaRuntime* fpga_runtime = ctx.currentModel->fpgaComputeEngine();
                if (fpga_runtime &&
                    (fpga_runtime->capabilities().operations &
                     Mimir::FpgaProtocol::ResidentMatrixVectorInt8) != 0) {
                    fpga_validation_weights.resize(
                        fpga_validation_rows * fpga_validation_columns
                    );
                    for (size_t row = 0; row < fpga_validation_rows; ++row) {
                        for (size_t column = 0; column < fpga_validation_columns; ++column) {
                            fpga_validation_weights[row * fpga_validation_columns + column] =
                                static_cast<int8_t>(
                                    static_cast<int>((row * 13 + column * 5) % 31) - 15
                                );
                        }
                    }
                    if (fpga_runtime->uploadInt8Matrix(
                            fpga_validation_weights.data(),
                            fpga_validation_rows,
                            fpga_validation_columns)) {
                        fpga_validation_runner =
                            std::make_unique<FpgaValidationRunner>(*fpga_runtime);
                        fpga_validation_active = true;
                        const std::filesystem::path log_dir = checkpoint_dir.empty()
                            ? std::filesystem::path(".")
                            : std::filesystem::path(checkpoint_dir);
                        fpga_validation_log_path =
                            (log_dir / "fpga_validation.jsonl").string();
                        ctx.addLog(
                            "FPGA validation probe actif: holdout 8x8 INT8, every=" +
                            std::to_string(fpga_validation_every_steps) + " steps"
                        );
                    }
                }
            }

            auto collect_fpga_validation = [&](bool wait) {
                if (!fpga_validation_runner) return;
                if (wait) fpga_validation_runner->wait();
                const auto result = fpga_validation_runner->poll();
                if (!result) return;

                const bool exact = result->success && result->exact;
                json event;
                event["type"] = "fpga_validation_probe";
                event["training_step"] = result->training_step;
                event["sample"] = result->sample_id;
                event["input_checksum"] = result->input_checksum;
                event["transport_ok"] = result->success;
                event["exact"] = exact;
                event["outputs"] = result->outputs;
                if (!fpga_validation_log_path.empty()) {
                    try {
                        const std::filesystem::path log_path(fpga_validation_log_path);
                        std::error_code ec;
                        std::filesystem::create_directories(log_path.parent_path(), ec);
                        std::ofstream log_file(log_path, std::ios::app);
                        if (log_file) log_file << event.dump() << '\n';
                    } catch (...) {
                    }
                }

                ctx.addLog(
                    std::string("FPGA validation step=") +
                    std::to_string(result->training_step) +
                    " sample=" + result->sample_id +
                    (exact ? " PASS" : " FAIL")
                );
                if (!exact) {
                    fpga_validation_active = false;
                    ctx.addLog(
                        "FPGA validation désactivée pour cette session après un résultat non exact"
                    );
                }
            };

            auto submit_fpga_validation = [&](int step) {
                if (!fpga_validation_active || !fpga_validation_runner ||
                    fpga_validation_runner->busy() || val_indices.empty()) {
                    return;
                }

                const size_t sample_position =
                    (static_cast<size_t>(step / std::max(1, fpga_validation_every_steps)) - 1) %
                    val_indices.size();
                DatasetItem& fpga_item =
                    ctx.currentDataset[static_cast<size_t>(val_indices[sample_position])];
                DatasetItemUnloadGuard unload_fpga_item(fpga_item);
                if (!fpga_item.loadImageRGB(image_w, image_h) ||
                    !fpga_item.img_loaded ||
                    fpga_item.img.size() !=
                        static_cast<size_t>(image_w) * static_cast<size_t>(image_h) * 3) {
                    return;
                }

                std::vector<uint8_t> resized(fpga_validation_columns * 3);
                resizeBicubicRGB_SRGBLinear(
                    fpga_item.img.data(), image_w, image_h,
                    resized.data(), 8, 8
                );
                FpgaValidationRunner::DatasetInput dataset_input;
                dataset_input.image_rgb = std::move(resized);

                if ((!fpga_item.text_file.empty() || !fpga_item.text_inline.empty()) &&
                    !fpga_item.text.has_value()) {
                    fpga_item.loadText();
                }
                if (fpga_item.text.has_value() && ctx.currentTokenizer) {
                    // Le vocabulaire est compose/hydrate par le chemin dataset
                    // existant; le probe FPGA ne le modifie pas.
                    dataset_input.text_token_ids =
                        ctx.currentTokenizer->tokenize(*fpga_item.text);
                    dataset_input.tokenizer_vocab_size =
                        ctx.currentTokenizer->getVocabSize();
                }
                if (!fpga_item.audio_file.empty() && !fpga_item.audio_loaded) {
                    fpga_item.loadAudio();
                }
                if (fpga_item.audio_is_pcm_s16le_stereo_48k) {
                    dataset_input.audio_pcm_s16le_stereo = fpga_item.audio_bytes;
                }
                if (!fpga_item.video_file.empty() && !fpga_item.video_loaded) {
                    fpga_item.loadVideo();
                }
                if (fpga_item.video_is_rgb24_first_frame) {
                    dataset_input.video_rgb = fpga_item.video_bytes;
                }

                std::vector<float> normalized_input;
                if (!FpgaValidationRunner::composeDatasetInput(
                        dataset_input, fpga_validation_columns, normalized_input)) {
                    return;
                }
                std::vector<int32_t> expected(fpga_validation_rows, 0);
                std::vector<int8_t> input;
                if (!FpgaValidationRunner::quantizeNormalizedInput(
                        normalized_input, input)) {
                    return;
                }
                for (size_t column = 0; column < fpga_validation_columns; ++column) {
                    for (size_t row = 0; row < fpga_validation_rows; ++row) {
                        expected[row] += static_cast<int32_t>(input[column]) *
                            static_cast<int32_t>(
                                fpga_validation_weights[row * fpga_validation_columns + column]
                            );
                    }
                }

                (void)fpga_validation_runner->submit(
                    static_cast<uint64_t>(step),
                    std::move(input),
                    std::move(expected),
                    fpga_item.image_file
                );
            };
#else
            bool fpga_validation_active = false;
            auto collect_fpga_validation = [](bool) {};
            auto submit_fpga_validation = [](int) {};
#endif

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

            auto compute_val_recon = [&](const std::vector<float>& pred, const std::vector<float>& target, int recon_n, const std::vector<float>* loss_weights) -> float {
                if (recon_n <= 0) return 0.0f;
                RuntimeLossGrad::PixelLossOptions options; options.weights=loss_weights;
                try {
                    if (ctx.modelConfig.contains("charbonnier_eps")) {
                        options.charbonnier_eps = std::max(
                            1e-12f, ctx.modelConfig["charbonnier_eps"].get<float>());
                    }
                    if (ctx.modelConfig.contains("huber_delta")) {
                        options.huber_delta = std::max(
                            1e-6f, ctx.modelConfig["huber_delta"].get<float>());
                    }
                    if (ctx.modelConfig.contains("smoothl1_delta")) {
                        options.huber_delta = std::max(
                            1e-6f, ctx.modelConfig["smoothl1_delta"].get<float>());
                    }
                    if (ctx.modelConfig.contains("smoothl1_beta")) {
                        options.huber_delta = std::max(
                            1e-6f, ctx.modelConfig["smoothl1_beta"].get<float>());
                    }
                    if (ctx.modelConfig.contains("nll_sigma")) {
                        options.gaussian_nll_sigma = std::max(
                            1e-6f, ctx.modelConfig["nll_sigma"].get<float>());
                    }
                    if (ctx.modelConfig.contains("gaussian_nll_sigma")) {
                        options.gaussian_nll_sigma = std::max(
                            1e-6f, ctx.modelConfig["gaussian_nll_sigma"].get<float>());
                    }
                } catch (...) {
                }
                return static_cast<float>(RuntimeLossGrad::pixel_loss_and_grad(
                    pred.data(), target.data(), static_cast<size_t>(recon_n),
                    recon_loss_type, options).loss);
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

                const std::vector<float>* mu = nullptr;
                const std::vector<float>* logvar = nullptr;
                const int packed_size = image_dim + 2 * latent_dim;
                if ((int)pred.size() >= packed_size) {
                    mu = &pred;
                    logvar = &pred;
                } else if (ctx.currentModel->hasTensor("vae_conv/mu") &&
                           ctx.currentModel->hasTensor("vae_conv/logvar")) {
                    mu = &ctx.currentModel->getTensor("vae_conv/mu");
                    logvar = &ctx.currentModel->getTensor("vae_conv/logvar");
                    if ((int)mu->size() < latent_dim || (int)logvar->size() < latent_dim) {
                        return 0.0f;
                    }
                } else {
                    return 0.0f;
                }

                double kl = 0.0;
                for (int i = 0; i < latent_dim; ++i) {
                    const size_t mu_index = mu == &pred
                        ? static_cast<size_t>(image_dim + i)
                        : static_cast<size_t>(i);
                    const size_t logvar_index = logvar == &pred
                        ? static_cast<size_t>(image_dim + latent_dim + i)
                        : static_cast<size_t>(i);
                    const float mu_f = (*mu)[mu_index];
                    const float lv_raw = (*logvar)[logvar_index];
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
            int trained_steps_total = 0;
            bool bbox_log_emitted = false;

            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(train_indices.begin(), train_indices.end(), rng);
                ctx.addLog("Epoch " + std::to_string(epoch_offset + (epoch + 1)) + "/" + std::to_string(total_epochs_display) +
                           " (" + model_type + ") items=" + std::to_string(use_n));

                bool stop_requested = false;
                int trained_steps_epoch = 0;

                for (int k = 0; k < use_n; ++k) {
                    refresh_viz_activation();
                    DatasetItem& item = ctx.currentDataset[(size_t)train_indices[(size_t)k]];
                    if (item.image_file.empty() && item.video_file.empty()) continue;
                    DatasetItemUnloadGuard unload_item(item);

                    const bool stream_video = !item.video_file.empty();
#ifdef MIMIR_ENABLE_FFMPEG
                    std::unique_ptr<FFmpegVideoFrameStream> video_stream;
                    if (stream_video) {
                        video_stream = std::make_unique<FFmpegVideoFrameStream>(
                            item.video_file, image_w, image_h);
                        if (!video_stream->valid()) {
                            throw std::runtime_error(
                                "Dataset vidéo: impossible d'ouvrir '" + item.video_file +
                                "': " + video_stream->error());
                        }
                    }
#else
                    if (stream_video) {
                        throw std::runtime_error(
                            "Dataset vidéo: FFmpeg est requis (recompiler avec ENABLE_FFMPEG=ON)");
                    }
#endif

                    bool more_media_samples = true;
                    size_t video_frame_index = 0;
                    while (more_media_samples) {
                        if (stream_video) {
#ifdef MIMIR_ENABLE_FFMPEG
                            std::vector<uint8_t> frame;
                            if (!video_stream->nextFrame(frame)) break;
                            item.img = std::move(frame);
                            item.img_loaded = true;
                            item.w = image_w;
                            item.h = image_h;
                            item.img_c = 3;
                            ++video_frame_index;
#endif
                        } else {
                            more_media_samples = false;
                            item.loadImageRGB(image_w, image_h);
                        }
                        if (!item.img_loaded || item.img.size() != expected_u8) {
                            continue;
                        }

                    // Normalize u8 -> [-1, 1]
                    for (size_t i = 0; i < expected_u8; ++i) {
                        x[i] = (float)((double)item.img[i] / 127.5 - 1.0);
                    }

                    poll_viz_live_params();
                    const float bbox_strength=ctx.currentModel->modelConfig.value("bbox_loss_weight",1.0f);
                    auto spatial_weights=_mimir_load_spatial_loss_weights(item,image_w,image_h,image_c,bbox_strength,ctx.currentModel->modelConfig);

                    std::vector<int> ids;
                    std::string prompt;
                    if (text_pipeline_enabled) {
                        if (!item.text_file.empty() && !item.text.has_value()) item.loadText();
                        prompt = item.text.has_value() ? item.text.value() : std::string();
                        const std::string prompt_for_model = enrich_prompt_semantic(BBoxSupervision::groundedPrompt(prompt,item.annotation ? &*item.annotation : nullptr));
                        if (text_cond_runtime) {
                            ids = ctx.currentTokenizer->tokenizeEnsure(prompt_for_model);
                        } else {
                            ids = ctx.currentTokenizer->tokenize(prompt_for_model);
                        }
                        if(item.annotation && !item.annotation->value("entities",json::array()).empty() && (int)ids.size()>seq_len)
                            throw std::invalid_argument("VAE annotation context requires "+std::to_string(ids.size())+" tokens; increase seq_len (current "+std::to_string(seq_len)+")");
                        if ((int)ids.size() < seq_len) ids.resize((size_t)seq_len, pad_id);
                        else if ((int)ids.size() > seq_len) ids.resize((size_t)seq_len);
                    }

                    // Publish the input before the expensive training pass and before validation can unload it.
                    const bool publish_viz_sample = viz_active && ctx.asyncMonitor && ctx.currentModel &&
                        ctx.asyncMonitor->getViz() != nullptr &&
                        (trained_steps_epoch == 0 || k + 1 == use_n || ((global_step + 1) % log_every) == 0);
                    if (publish_viz_sample) {
                        std::string label = "vae_conv/input/dataset/rgb";
                        label += "/i=" + std::to_string(train_indices[(size_t)k]);

                        const std::vector<int>* ids_ptr = (text_pipeline_enabled && !ids.empty()) ? &ids : nullptr;
                        const auto viz_payload = VizTextPayload::buildDatasetTextPayload(
                            ctx.currentModel.get(),
                            ctx.currentTokenizer.get(),
                            item.text.has_value() ? item.text.value() : prompt,
                            ids_ptr,
                            text_pipeline_enabled ? pad_id : -1
                        );

                        ctx.asyncMonitor->setDatasetSample(
                            item.img,
                            image_w,
                            image_h,
                            image_c,
                            label,
                            viz_payload.prompt,
                            viz_payload.tags,
                            viz_payload.tokens,
                            viz_payload.encoding,
                            spatial_weights.bbox_count>0 && spatial_weights.strength>0,
                            DatasetBBoxOverlay::fromAnnotation(item.annotation ? &*item.annotation : nullptr),
                            item.annotation ? &*item.annotation : nullptr, &ctx.currentModel->modelConfig
                        );

                    }

                    Model::TrainStepRequest request;
                    request.float_inputs["__input__"] = &x;
                    if (graph_text_cond) request.int_inputs["text_ids"] = &ids;
                    if (!spatial_weights.values.empty()) request.loss_weights = &spatial_weights.values;
                    request.relations = &spatial_weights.relations;
                    request.optimizer = &opt;
                    request.learning_rate = optimizer_learning_rate();
                    const auto step_result = ctx.currentModel->trainStep(request);
                    if (!step_result) {
                        throw std::runtime_error("Le modèle '" + model_type + "' ne fournit pas de contrat d'entraînement compatible");
                    }
                    Model::TrainStepResult st = *step_result;
                    st.metrics["bbox_count"]=static_cast<float>(spatial_weights.bbox_count);
                    st.metrics["bbox_loss_weight"]=spatial_weights.strength;
                    st.metrics["relation_count"]=static_cast<float>(spatial_weights.relations.size());
                    if (!spatial_weights.relations.empty() && !bbox_log_emitted)
                        ctx.addLog("Relation supervision: pairs="+std::to_string(spatial_weights.relations.size()));
                    if (spatial_weights.bbox_count>0 && !bbox_log_emitted) {
                        ctx.addLog("BBox reconstruction supervision: boxes="+std::to_string(spatial_weights.bbox_count)+
                            " weight="+std::to_string(spatial_weights.strength)); bbox_log_emitted=true;
                    }

                    if (!graph_text_cond) {

                        // Mode runtime optionnel: apprend la corrélation image↔texte via l'ConditioningEncoder,
                        // sans modifier le chemin de loss/convergence du graphe vae_conv existant.
                        if (text_cond_runtime && ctx.currentModel) {
                            try {
                                auto& enc = ctx.currentModel->getMutableEncoder();
                                enc.ensureVocabSize(ctx.currentTokenizer->getVocabSize());
                                enc.ensureSpecialEmbeddings();

                                const bool single_modality_sample = (item.countModalities() <= 1);

                                // Cible "image" en espace embedding:
                                // priorité au latent packé [recon | z | logvar] du VAE.
                                // Quand use_encoder_prior=true, ce chunk correspond à z_biased
                                // (z + prior), donc mag_embedding capture aussi ce signal.
                                std::vector<float> img_emb;
                                try {
                                    const std::vector<float>& packed = ctx.currentModel->forwardPassView(x, false);
                                    int latent_for_pack = std::max(0, static_cast<int>(train_metric(st, "latent_dim")));
                                    const int image_dim_for_pack = static_cast<int>(expected_u8);
                                    const int packed_n = static_cast<int>(packed.size());
                                    if (latent_for_pack <= 0 && packed_n > image_dim_for_pack + 2 && ((packed_n - image_dim_for_pack) % 2) == 0) {
                                        latent_for_pack = std::max(1, (packed_n - image_dim_for_pack) / 2);
                                    }
                                    if (latent_for_pack > 0) {
                                        img_emb = latentChunkToEmbedding(packed, image_dim_for_pack, latent_for_pack, enc.dim);
                                    }
                                } catch (...) {
                                    // fallback ci-dessous
                                }
                                if (img_emb.empty()) {
                                    img_emb = imageToEmbedding(item.img, image_w, image_h, enc.dim);
                                }

                                // En mono-modalité image: remplir uniquement le vecteur image (mag).
                                enc.fillImageVectorSingleModality(img_emb, text_special_lr, single_modality_sample);

                                // En mono-modalité texte: remplir uniquement le vecteur texte (seq).
                                if (!ids.empty()) {
                                    enc.fillTextVectorSingleModality(ids, pad_id, text_special_lr, single_modality_sample);
                                }

                                // Mode multi-modalité: conserver l'alignement image↔texte historique.
                                if (!single_modality_sample && text_encoder_lr > 0.0f && !ids.empty()) {
                                    enc.trainOnTextTokens(ids, img_emb, text_encoder_lr);
                                }

                                if (!single_modality_sample && !ids.empty()) {
                                    // Lier mag_embedding à la modalité image du sample courant.
                                    std::string img_dir;
                                    try {
                                        img_dir = std::filesystem::path(item.image_file).parent_path().string();
                                    } catch (...) {
                                    }
                                    const MagicToken mt = makeMagicToken(MOD_IMAGE, img_dir.empty() ? item.image_file : img_dir);
                                    enc.setMagicFromToken(mt);

                                    // Gradient d'alignement sur seq/mod: encode(ids) -> img_emb.
                                    std::vector<float> txt_emb;
                                    enc.encodeInto(txt_emb, ids);
                                    if (txt_emb.size() == img_emb.size() && !txt_emb.empty() && text_special_lr > 0.0f) {
                                        std::vector<float> grad(txt_emb.size(), 0.0f);
                                        for (size_t d = 0; d < txt_emb.size(); ++d) {
                                            grad[d] = txt_emb[d] - img_emb[d];
                                        }
                                        enc.sgdUpdateSpecialEmbeddings(grad, text_special_lr, true, true, false);
                                    }

                                    // Compose mod_embedding comme pont seq <-> mag (image feature).
                                    const auto& seq = enc.getSeqEmbedding();
                                    const auto& mag = enc.getMagEmbedding();
                                    if (seq.size() == (size_t)enc.dim && mag.size() == (size_t)enc.dim) {
                                        std::vector<float> mod((size_t)enc.dim, 0.0f);
                                        for (int d = 0; d < enc.dim; ++d) {
                                            mod[(size_t)d] = 0.5f * (seq[(size_t)d] + mag[(size_t)d]);
                                        }
                                        enc.setModEmbedding(mod);
                                    }
                                }
                            } catch (...) {
                                // best-effort: ne jamais casser le train principal.
                            }
                        }
                    }

                    global_step += 1;
                    trained_steps_epoch += 1;
                    trained_steps_total += 1;
                    log_step(global_step, st, step_prefix.c_str());
                    monitor_step(epoch + 1, k + 1, use_n, st);

                    // Arrêt sécurisé depuis Htop, Viz ou SIGINT.
                    if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                        ctx.addLog("⛔ Arrêt sécurisé demandé. Sauvegarde et arrêt...");
                        stop_requested = true;
                        stopped_by_ui = true;
                        break;
                    }

                    // Capture training taps before a validation forward overwrites them.
                    if (publish_viz_sample) {
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
                                if (graph_text_cond) {
                                    std::vector<int> ids = ctx.currentTokenizer ? ctx.currentTokenizer->tokenize(enrich_prompt_semantic(BBoxSupervision::groundedPrompt(prompt,item.annotation ? &*item.annotation : nullptr))) : std::vector<int>();
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
                            bf.pixels_real = std::move(f.pixels_real);
                            bf.heatmap_kind = f.heatmap_kind;
                            bf.tensor_info = std::move(f.tensor_info);
                            bf.tensor_values = std::move(f.tensor_values);
                            bf.tensor_indices = std::move(f.tensor_indices);
                            bf.tensor_sample_channels = f.tensor_sample_channels;
                            bf.label = std::move(f.label);
                            frames.push_back(std::move(bf));
                        }
                        // Important UX: même si aucun tap n'est émis, vider les frames précédentes.
                        ctx.asyncMonitor->setLayerBlockImages(std::move(frames));
                    }

                    // Validation: forward-only sur un petit holdout, puis push dans Generated.
                    bool validation_runtime_enabled = validate_every_steps > 0 && validate_items > 0;
                    if (ctx.asyncMonitor) {
                        ctx.asyncMonitor->updateRuntimeValidationEnabled(validation_runtime_enabled);
                        validation_runtime_enabled = ctx.asyncMonitor->validationEnabledSnapshot();
                    }
                    if (validation_runtime_enabled && validate_every_steps > 0 && validate_items > 0 && !val_indices.empty() && (global_step % validate_every_steps) == 0) {
                        const int image_dim = (int)expected_u8;
                        int latent_dim = 0;
                        try {
                            if (ctx.modelConfig.contains("latent_dim")) latent_dim = std::max(0, ctx.modelConfig["latent_dim"].get<int>());
                        } catch (...) {
                        }

                        const int total = std::min((int)val_indices.size(), std::max(1, validate_items));
                        if (ctx.asyncMonitor) ctx.asyncMonitor->updateValidation(true, global_step, 0, total, false, false, 0.0f, 0.0f, 0.0f);

                        const bool taps_prev = ctx.currentModel->isVizTapsEnabled();
                        if (ctx.asyncMonitor && ctx.asyncMonitor->getViz()) {
                            ctx.currentModel->clearVizTaps();
                        }

                        // Sample a subset each time (shuffle copy for randomness)
                        std::vector<int> val_pick = val_indices;
                        std::shuffle(val_pick.begin(), val_pick.end(), rng);
                        if ((int)val_pick.size() > total) val_pick.resize((size_t)total);

                        double acc_recon = 0.0;
                        double acc_kl = 0.0;
                        int done = 0;
                        bool val_ok = true;

                        for (int vi = 0; vi < (int)val_pick.size(); ++vi) {
                            if (ctx.asyncMonitor && !ctx.asyncMonitor->validationEnabledSnapshot()) {
                                val_ok = false;
                                break;
                            }
                            if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                                val_ok = false;
                                stop_requested = true;
                                stopped_by_ui = true;
                                break;
                            }

                            DatasetItem& vitem = ctx.currentDataset[(size_t)val_pick[(size_t)vi]];
                            if (vitem.image_file.empty()) continue;
                            DatasetItemUnloadGuard unload_vitem(vitem);

                            vitem.loadImageRGB(image_w, image_h);
                            if (!vitem.img_loaded || vitem.img.size() != expected_u8) continue;

                            // Normalize u8 -> [-1, 1]
                            for (size_t i = 0; i < expected_u8; ++i) {
                                x[i] = (float)((double)vitem.img[i] / 127.5 - 1.0);
                            }

                            const std::vector<float>* ppred = nullptr;
                            std::vector<int> ids;
                            std::string vprompt;
                            if (text_pipeline_enabled) {
                                if (!vitem.text_file.empty() && !vitem.text.has_value()) vitem.loadText();
                                vprompt = vitem.text.has_value() ? vitem.text.value() : std::string();
                                const std::string vprompt_for_model = enrich_prompt_semantic(BBoxSupervision::groundedPrompt(vprompt,vitem.annotation ? &*vitem.annotation : nullptr));
                                ids = ctx.currentTokenizer->tokenize(vprompt_for_model);
                                if(vitem.annotation && !vitem.annotation->value("entities",json::array()).empty() && (int)ids.size()>seq_len)
                                    throw std::invalid_argument("VAE validation annotation context requires "+std::to_string(ids.size())+" tokens; increase seq_len (current "+std::to_string(seq_len)+")");
                                if ((int)ids.size() < seq_len) ids.resize((size_t)seq_len, pad_id);
                                else if ((int)ids.size() > seq_len) ids.resize((size_t)seq_len);

                                if (graph_text_cond) {
                                    std::unordered_map<std::string, std::vector<float>> fin;
                                    std::unordered_map<std::string, std::vector<int>> iin;
                                    fin["__input__"] = x;
                                    iin["text_ids"] = ids;
                                    ppred = &ctx.currentModel->forwardPassNamedView(fin, iin, false);
                                } else {
                                    ppred = &ctx.currentModel->forwardPassView(x, false);
                                }
                            } else {
                                ppred = &ctx.currentModel->forwardPassView(x, false);
                            }
                            if (!ppred) continue;
                            const std::vector<float>& pred = *ppred;
                            if ((int)pred.size() < image_dim) continue;

                            const int recon_n = std::min({
                                image_dim,
                                static_cast<int>(pred.size()),
                                static_cast<int>(x.size())
                            });
                            const auto spatial_weights=_mimir_load_spatial_loss_weights(vitem,image_w,image_h,image_c,
                                ctx.currentModel->modelConfig.value("bbox_loss_weight",1.0f),ctx.currentModel->modelConfig);
                            float vrecon = compute_val_recon(pred, x, recon_n,
                                spatial_weights.strength > 0 ? &spatial_weights.values : nullptr);
                            if (!spatial_weights.relations.empty()) {
                                RuntimeLossGrad::PixelLossOptions relation_options;
                                relation_options.weights = &spatial_weights.values;
                                relation_options.relations = &spatial_weights.relations;
                                relation_options.relation_only = true;
                                vrecon += static_cast<float>(RuntimeLossGrad::pixel_loss_and_grad(
                                    pred.data(),x.data(),static_cast<size_t>(recon_n),"mse",relation_options).relation_loss);
                            }

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
                                ctx.asyncMonitor->addImage(vitem.img, image_w, image_h, image_c, std::string("VAL target | ") + idx, true);
                                pack_f32_to_u8(pred, 0, recon_u8);
                                ctx.asyncMonitor->addImage(recon_u8, image_w, image_h, image_c, std::string("VAL recon | ") + idx, true);
                            }

                            if (ctx.asyncMonitor) {
                                const float avg_recon = (done > 0) ? (float)(acc_recon / (double)done) : 0.0f;
                                const float avg_kl = (done > 0) ? (float)(acc_kl / (double)done) : 0.0f;
                                ctx.asyncMonitor->updateValidation(true, global_step, done, total, true, false, avg_recon, avg_kl, 0.0f);
                            }
                        }

                        ctx.currentModel->setVizTapsEnabled(taps_prev);
                        _mimir_publish_viz_taps(ctx, true);

                        const float final_recon = (done > 0) ? (float)(acc_recon / (double)done) : 0.0f;
                        const float final_kl = (done > 0) ? (float)(acc_kl / (double)done) : 0.0f;
                        // Calibration: récompense / punition selon l'évolution de la loss de reconstruction VAE.
                        std::string val_feedback = "none";
                        if (val_ok && done > 0) {
                            val_feedback = apply_vae_val_feedback(final_recon, final_kl, global_step);
                        }
                        if (ctx.asyncMonitor) {
                            ctx.asyncMonitor->updateValidation(false, global_step, done, total,
                                                               true, val_ok, final_recon, final_kl,
                                                               0.0f, val_feedback);
                        }

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

                    collect_fpga_validation(false);
                    if (fpga_validation_active && fpga_validation_every_steps > 0 &&
                        (global_step % fpga_validation_every_steps) == 0) {
                        submit_fpga_validation(global_step);
                    }

                    }

                    if (stream_video && video_frame_index == 0) {
                        ctx.addLog("⚠ Dataset vidéo sans frame décodable: " + item.video_file);
                    }
                    if (stop_requested) break;
                }

                ctx.addLog("Epoch stats: trained_steps=" + std::to_string(trained_steps_epoch) +
                           "/" + std::to_string(use_n) +
                           " global_step=" + std::to_string(global_step));

                if (trained_steps_epoch == 0 && !stop_requested) {
                    ctx.addLog("⚠ Aucun batch entraine sur cette epoch (0/" + std::to_string(use_n) + ") - images invalides/inaccessibles ou dataset non charge correctement.");
                    break;
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

            collect_fpga_validation(true);

            if (trained_steps_total == 0) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Aucun batch n'a ete entraine (0 step). Verifiez dataset_root, acces fichiers image, et preload/caching des images.");
                return 2;
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
            if (viz_taps_every_steps <= 0) viz_taps_every_steps = log_every;
            viz_taps_every_steps = std::max(1, viz_taps_every_steps);

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
                    bf.pixels_real = std::move(f.pixels_real);
                    bf.heatmap_kind = f.heatmap_kind;
                    bf.tensor_info = std::move(f.tensor_info);
                    bf.tensor_values = std::move(f.tensor_values);
                    bf.tensor_indices = std::move(f.tensor_indices);
                    bf.tensor_sample_channels = f.tensor_sample_channels;
                            bf.label = std::move(f.label);
                    frames.push_back(std::move(bf));
                }
                // Important UX: même si aucun tap n'est émis, vider les frames précédentes.
                ctx.asyncMonitor->setLayerBlockImages(std::move(frames));
            };

            auto push_vgg_context_and_output = [&](const DatasetItem& item,
                                                   const std::vector<float>& features,
                                                   int dataset_index) {
                if (!viz_active || !ctx.asyncMonitor || ctx.asyncMonitor->getViz() == nullptr) return;

                // Context: image RGB réellement consommée par ce step.
                const size_t expected_context_u8 = static_cast<size_t>(image_w) *
                    static_cast<size_t>(image_h) * static_cast<size_t>(image_c);
                if (item.img_loaded && item.img.size() == expected_context_u8) {
                    ctx.asyncMonitor->stageDatasetImage(
                        item.img,
                        image_w,
                        image_h,
                        image_c,
                        "vgg16_feat/input/dataset/rgb/i=" + std::to_string(dataset_index),
                        _mimir_dataset_bbox_active(item,*ctx.currentModel),
                        DatasetBBoxOverlay::fromAnnotation(item.annotation ? &*item.annotation : nullptr),
                            item.annotation ? &*item.annotation : nullptr, &ctx.currentModel->modelConfig);
                }

                // Sortie vgg16_feat: le modèle produit un vecteur perceptuel et non
                // une image. On le projette en grille carrée 1 canal, normalisée
                // autour de zéro, pour alimenter Context/compréhension et Sortie.
                if (features.empty()) return;
                const int side = std::max(1, static_cast<int>(std::ceil(
                    std::sqrt(static_cast<double>(features.size())))));
                std::vector<uint8_t> heatmap(static_cast<size_t>(side * side), 127);
                float max_abs = 0.0f;
                for (float value : features) {
                    if (std::isfinite(value)) max_abs = std::max(max_abs, std::fabs(value));
                }
                const float inv = 1.0f / (max_abs + 1e-6f);
                for (size_t i = 0; i < features.size(); ++i) {
                    const float value = std::isfinite(features[i]) ? features[i] : 0.0f;
                    const float normalized = 0.5f + 0.5f * std::tanh(value * inv);
                    heatmap[i] = static_cast<uint8_t>(std::clamp(
                        static_cast<int>(std::lround(normalized * 255.0f)), 0, 255));
                }

                const std::string label = "vgg16_feat/output/perceptual_features";
                ctx.asyncMonitor->setUnderstandingImage(heatmap, side, side, 1, label);
                ctx.asyncMonitor->addImage(heatmap, side, side, 1, label);
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
            int stopped_epoch = 0;

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

                m.memory_mb = _mimir_process_resident_memory_mb();
                m.allocator_memory_mb =
                    static_cast<double>(MemoryGuard::instance().getCurrentBytes()) / (1024.0 * 1024.0);
            };

            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(indices.begin(), indices.end(), rng);
                ctx.addLog("Epoch " + std::to_string(epoch_offset + (epoch + 1)) + "/" + std::to_string(total_epochs_display) +
                           " (vgg16_feat) items=" + std::to_string(use_n));

                bool stop_requested = false;

                for (int k = 0; k < use_n; ++k) {
                    refresh_viz_activation();
                    DatasetItem& item = ctx.currentDataset[(size_t)indices[(size_t)k]];
                    if (item.image_file.empty()) continue;
                    DatasetItemUnloadGuard unload_item(item);

                    item.loadImageRGB(image_w, image_h);
                    if (!item.img_loaded || item.img.size() != expected_u8) {
                        continue;
                    }

                    // Normalize u8 -> [-1, 1]
                    for (size_t i = 0; i < expected_u8; ++i) {
                        x[i] = (float)((double)item.img[i] / 127.5 - 1.0);
                    }
                    ctx.currentModel->applyRuntimeConfiguration();
                    const std::vector<float> mask_weights =
                        _mimir_load_spatial_loss_weights(item, image_w, image_h, image_c,
                            ctx.currentModel->modelConfig.value("bbox_loss_weight",1.0f),ctx.currentModel->modelConfig).values;
                    if (mask_weights.size() == x.size()) {
                        for (size_t i = 0; i < x.size(); ++i) x[i] *= mask_weights[i];
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

                    // Vraie norme L2 globale + max abs des gradients (avant clipping/step).
                    // Sans ça, les métriques vgg16_feat affichaient grad_norm=grad_max=0
                    // (valeurs codées en dur), masquant toute explosion de gradient.
                    float grad_norm_val = 0.0f;
                    float grad_max_val = 0.0f;
                    {
                        double sum_sq = 0.0;
                        float max_abs = 0.0f;
                        for (const auto& layer : ctx.currentModel->getLayers()) {
                            for (float g : layer.grad_weights) {
                                sum_sq += (double)g * (double)g;
                                const float a = std::fabs(g);
                                if (a > max_abs) max_abs = a;
                            }
                            for (float g : layer.grad_bias) {
                                sum_sq += (double)g * (double)g;
                                const float a = std::fabs(g);
                                if (a > max_abs) max_abs = a;
                            }
                        }
                        grad_norm_val = (float)std::sqrt(sum_sq);
                        grad_max_val = max_abs;
                    }

                    poll_viz_live_params();
                    ctx.currentModel->optimizerStep(opt, optimizer_learning_rate(), nullptr);

                    // Metrics for UI/log
                    global_step += 1;

                    // Viz taps (throttled)
                    if (viz_taps_every_steps > 0 && (global_step % viz_taps_every_steps) == 0) {
                        push_vgg_context_and_output(item, pred, indices[(size_t)k]);
                        push_viz_taps();
                    }

                    if ((global_step % log_every) == 0) {
                        ctx.addLog("step=" + std::to_string(global_step) +
                                   " loss=" + std::to_string((float)loss) +
                                   " grad_norm=" + std::to_string(grad_norm_val) +
                                   " lr=" + std::to_string(opt.getCurrentLR()) +
                                   " (vgg16_feat)");
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
                        m.grad_norm = grad_norm_val;
                        m.grad_max = grad_max_val;
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

                    // Arrêt sécurisé demandé depuis une interface de monitoring.
                    if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                        ctx.addLog("⛔ Arrêt sécurisé demandé. Sauvegarde et arrêt...");
                        stop_requested = true;
                        stopped_by_ui = true;
                        stopped_epoch = epoch + 1;
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
                std::string save_error;
                if (!do_checkpoint_save(stopped_epoch, "_stop", &save_error) && !save_error.empty()) {
                    ctx.addLog("⚠ Save(stop) échoué: " + save_error);
                }
                lua_pushboolean(L, false);
                lua_pushstring(L, "STOP_REQUESTED");
                return 2;
            }

            lua_pushboolean(L, true);
            return 1;
        } else if (model_type == "medical_detector") {
            auto* detector = dynamic_cast<MedicalDetectorModel*>(ctx.currentModel.get());
            if (!detector) {
                throw std::runtime_error("medical_detector: type de modèle natif invalide");
            }

            const int image_w = std::max(1, ctx.modelConfig.value("image_w", 256));
            const int image_h = std::max(1, ctx.modelConfig.value("image_h", 256));
            const int image_c = std::max(1, ctx.modelConfig.value("image_c", 3));
            auto detector_grid_size = [](int size) {
                for (int block = 0; block < 5; ++block) size = (size + 1) / 2;
                return std::max(1, size);
            };
            const int grid_w = std::max(
                1, ctx.modelConfig.value("grid_w", detector_grid_size(image_w)));
            const int grid_h = std::max(
                1, ctx.modelConfig.value("grid_h", detector_grid_size(image_h)));
            const int num_classes = std::max(1, ctx.modelConfig.value("num_classes", 1));
            const float score_threshold = std::clamp(
                ctx.modelConfig.value("score_threshold", 0.25f), 0.0f, 1.0f);
            const float nms_iou_threshold = std::clamp(
                ctx.modelConfig.value("nms_iou_threshold", 0.45f), 0.0f, 1.0f);
            const float match_iou_threshold = std::clamp(
                ctx.modelConfig.value("match_iou_threshold", 0.3f), 0.0f, 1.0f);
            const float validation_fraction = std::clamp(
                ctx.modelConfig.value("validation_fraction", 0.2f), 0.0f, 0.9f);
            const int validation_items = std::max(1,
                ctx.modelConfig.value("validation_items", 32));
            const int validate_every_epochs = std::max(1,
                ctx.modelConfig.value("validate_every_epochs", 1));
            const int viz_every_steps = std::max(1,
                ctx.modelConfig.value("viz_every_steps", log_every));
            const int max_detections = std::max(1,
                ctx.modelConfig.value("max_detections", 100));

            std::vector<std::string> class_names((size_t)num_classes);
            for (int class_id = 0; class_id < num_classes; ++class_id) {
                class_names[(size_t)class_id] = "anomaly_" + std::to_string(class_id);
            }
            if (ctx.modelConfig.contains("class_names") &&
                ctx.modelConfig["class_names"].is_array()) {
                const auto& names = ctx.modelConfig["class_names"];
                for (int class_id = 0;
                     class_id < num_classes && class_id < (int)names.size(); ++class_id) {
                    if (names[(size_t)class_id].is_string()) {
                        class_names[(size_t)class_id] = names[(size_t)class_id].get<std::string>();
                    }
                }
            }

            struct MedicalSample {
                int dataset_index = -1;
                std::vector<MedicalDetectorModel::Box> boxes;
            };
            std::vector<MedicalSample> samples;
            int missing_annotations = 0;
            int malformed_annotations = 0;
            for (size_t index = 0; index < ctx.currentDataset.size(); ++index) {
                const DatasetItem& item = ctx.currentDataset[index];
                if (item.image_file.empty()) continue;
                MedicalSample sample;
                sample.dataset_index = static_cast<int>(index);
                std::string annotation_path;
                std::string annotation_error;
                if (!MedicalDetectorModel::loadAnnotationForImage(
                        item.image_file, image_w, image_h, num_classes,
                        &sample.boxes, &annotation_path, &annotation_error)) {
                    if (annotation_error.find("not found") != std::string::npos) {
                        ++missing_annotations;
                    } else {
                        ++malformed_annotations;
                        ctx.addLog("medical_detector: annotation invalide pour " +
                                   item.image_file + ": " + annotation_error);
                    }
                    continue;
                }
                samples.push_back(std::move(sample));
            }
            if (samples.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L,
                    "medical_detector: aucune paire image + annotation .json/.boxes.json valide");
                return 2;
            }
            std::shuffle(samples.begin(), samples.end(), rng);
            if (max_items > 0 && (int)samples.size() > max_items) {
                samples.resize((size_t)max_items);
            }

            int validation_count = 0;
            if (samples.size() >= 2 && validation_fraction > 0.0f) {
                validation_count = std::clamp(
                    static_cast<int>(std::lround(samples.size() * validation_fraction)),
                    1, static_cast<int>(samples.size()) - 1);
            }
            std::vector<MedicalSample> validation_samples;
            if (validation_count > 0) {
                validation_samples.assign(samples.end() - validation_count, samples.end());
                samples.erase(samples.end() - validation_count, samples.end());
            }
            ctx.addLog("medical_detector: train=" + std::to_string(samples.size()) +
                       " validation=" + std::to_string(validation_samples.size()) +
                       " annotations_absentes=" + std::to_string(missing_annotations) +
                       " annotations_invalides=" + std::to_string(malformed_annotations) +
                       " grille=" + std::to_string(grid_w) + "x" + std::to_string(grid_h));

            const size_t expected_pixels = static_cast<size_t>(image_w) *
                                           static_cast<size_t>(image_h) *
                                           static_cast<size_t>(image_c);
            std::vector<float> image_chw(expected_pixels);
            std::vector<float> target;
            auto load_image = [&](MedicalSample& sample, DatasetItem** loaded_item) -> bool {
                DatasetItem& item = ctx.currentDataset[(size_t)sample.dataset_index];
                item.loadImageRGB(image_w, image_h);
                if (!item.img_loaded || item.img.size() != expected_pixels) return false;
                for (int y = 0; y < image_h; ++y) {
                    for (int x = 0; x < image_w; ++x) {
                        const size_t pixel = static_cast<size_t>(y * image_w + x) * image_c;
                        for (int channel = 0; channel < image_c; ++channel) {
                            const size_t chw = static_cast<size_t>(channel) * image_h * image_w +
                                               static_cast<size_t>(y * image_w + x);
                            image_chw[chw] = static_cast<float>(item.img[pixel + channel]) /
                                             127.5f - 1.0f;
                        }
                    }
                }
                *loaded_item = &item;
                return true;
            };

            auto push_medical_viz_taps = [&]() {
                if (!ctx.asyncMonitor || !ctx.asyncMonitor->getViz()) return;
                auto taps = ctx.currentModel->consumeVizTaps();
                std::vector<Visualizer::BlockFrame> frames;
                frames.reserve(taps.size());
                for (auto& tap : taps) {
                    Visualizer::BlockFrame frame;
                    frame.pixels = std::move(tap.pixels);
                    frame.w = tap.w;
                    frame.h = tap.h;
                    frame.channels = tap.channels;
                    frame.pixels_real = std::move(tap.pixels_real);
                    frame.heatmap_kind = tap.heatmap_kind;
                    frame.tensor_info = std::move(tap.tensor_info);
                    frame.tensor_values = std::move(tap.tensor_values);
                    frame.tensor_indices = std::move(tap.tensor_indices);
                    frame.tensor_sample_channels = tap.tensor_sample_channels;
                    frame.label = std::move(tap.label);
                    frames.push_back(std::move(frame));
                }
                ctx.asyncMonitor->setLayerBlockImages(std::move(frames));
            };

            auto publish_detections = [&](DatasetItem& item,
                                          const std::vector<MedicalDetectorModel::Detection>& detections,
                                          const std::vector<MedicalDetectorModel::Box>& truth,
                                          const std::string& label) {
                if (!ctx.asyncMonitor || !ctx.asyncMonitor->getViz()) return;
                ctx.asyncMonitor->stageDatasetImage(item.img, image_w, image_h, image_c, label,
                    _mimir_dataset_bbox_active(item,*ctx.currentModel),
                        DatasetBBoxOverlay::fromAnnotation(item.annotation ? &*item.annotation : nullptr),
                            item.annotation ? &*item.annotation : nullptr, &ctx.currentModel->modelConfig);
                ctx.asyncMonitor->addImage(item.img, image_w, image_h, image_c, label,
                    label.find("/validation/") != std::string::npos);
                std::vector<Visualizer::Detection> overlay;
                overlay.reserve(truth.size() + detections.size());
                for (const auto& target : truth) {
                    Visualizer::Detection box;
                    box.x1 = target.x1;
                    box.y1 = target.y1;
                    box.x2 = target.x2;
                    box.y2 = target.y2;
                    box.score = -1.0f;
                    box.class_id = target.class_id;
                    const std::string class_name = target.class_id >= 0 && target.class_id < num_classes
                        ? class_names[(size_t)target.class_id]
                        : target.label;
                    box.label = "cible: " + class_name;
                    overlay.push_back(std::move(box));
                }
                for (const auto& detection : detections) {
                    Visualizer::Detection box;
                    box.x1 = detection.x1;
                    box.y1 = detection.y1;
                    box.x2 = detection.x2;
                    box.y2 = detection.y2;
                    box.score = detection.score;
                    box.class_id = detection.class_id;
                    const std::string class_name = detection.class_id >= 0 && detection.class_id < num_classes
                        ? class_names[(size_t)detection.class_id]
                        : detection.label;
                    box.label = "prédiction: " + class_name;
                    overlay.push_back(std::move(box));
                }
                ctx.asyncMonitor->setDetections(
                    overlay, Visualizer::DetectionTarget::Output);
                ctx.asyncMonitor->setDetectionOverlayEnabled(true);
            };

            struct DetectionValidation {
                int items = 0;
                int true_positive = 0;
                int false_positive = 0;
                int false_negative = 0;
                float mean_iou = 0.0f;
                float precision = 0.0f;
                float recall = 0.0f;
                float f1 = 0.0f;
            };
            auto run_validation = [&](int epoch_1based) {
                DetectionValidation stats;
                if (ctx.asyncMonitor && ctx.asyncMonitor->getViz()) {
                    ctx.currentModel->clearVizTaps();
                }
                double matched_iou_sum = 0.0;
                int matched_count = 0;
                const int count = std::min(validation_items,
                                           (int)validation_samples.size());
                for (int index = 0; index < count; ++index) {
                    MedicalSample& sample = validation_samples[(size_t)index];
                    DatasetItem* item = nullptr;
                    if (!load_image(sample, &item)) continue;
                    DatasetItemUnloadGuard unload(*item);
                    const std::vector<float>& output =
                        ctx.currentModel->forwardPassView(image_chw, false);
                    auto detections = MedicalDetectorModel::decodeDetections(
                        output, grid_h, grid_w, num_classes, score_threshold,
                        nms_iou_threshold, max_detections);
                    std::vector<bool> matched(sample.boxes.size(), false);
                    for (const auto& detection : detections) {
                        int best_index = -1;
                        float best_iou = 0.0f;
                        for (size_t truth_index = 0;
                             truth_index < sample.boxes.size(); ++truth_index) {
                            if (matched[truth_index] ||
                                sample.boxes[truth_index].class_id != detection.class_id) continue;
                            const float iou = MedicalDetectorModel::intersectionOverUnion(
                                detection, sample.boxes[truth_index]);
                            if (iou > best_iou) {
                                best_iou = iou;
                                best_index = static_cast<int>(truth_index);
                            }
                        }
                        if (best_index >= 0 && best_iou >= match_iou_threshold) {
                            matched[(size_t)best_index] = true;
                            ++stats.true_positive;
                            ++matched_count;
                            matched_iou_sum += best_iou;
                        } else {
                            ++stats.false_positive;
                        }
                    }
                    for (bool found : matched) {
                        if (!found) ++stats.false_negative;
                    }
                    ++stats.items;
                    if (index == 0) {
                        const auto display_detections = MedicalDetectorModel::decodeDetections(
                            output, grid_h, grid_w, num_classes, 0.0f,
                            nms_iou_threshold, max_detections);
                        publish_detections(*item, display_detections,
                            sample.boxes,
                            "medical_detector/validation/epoch=" +
                            std::to_string(epoch_offset + epoch_1based));
                    }
                }
                const int precision_den = stats.true_positive + stats.false_positive;
                const int recall_den = stats.true_positive + stats.false_negative;
                stats.precision = precision_den > 0
                    ? static_cast<float>(stats.true_positive) / precision_den : 0.0f;
                stats.recall = recall_den > 0
                    ? static_cast<float>(stats.true_positive) / recall_den : 0.0f;
                stats.f1 = stats.precision + stats.recall > 0.0f
                    ? 2.0f * stats.precision * stats.recall /
                          (stats.precision + stats.recall) : 0.0f;
                stats.mean_iou = matched_count > 0
                    ? static_cast<float>(matched_iou_sum / matched_count) : 0.0f;
                ctx.addLog("medical_detector val: epoch=" +
                           std::to_string(epoch_offset + epoch_1based) +
                           " items=" + std::to_string(stats.items) +
                           " precision=" + std::to_string(stats.precision) +
                           " recall=" + std::to_string(stats.recall) +
                           " f1=" + std::to_string(stats.f1) +
                           " mean_iou=" + std::to_string(stats.mean_iou));
                _mimir_publish_viz_taps(ctx, true);
                return stats;
            };

            opt.total_steps = std::max(1, epochs * std::max(1, (int)samples.size()));
            double running_loss = 0.0;
            int running_steps = 0;
            bool stopped_by_ui = false;
            int stopped_epoch = 0;
            bool positive_output_published = false;
            AsyncMonitor::Metrics last_medical_metrics;
            bool has_last_medical_metrics = false;
            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(samples.begin(), samples.end(), rng);
                for (size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
                    refresh_viz_activation();
                    MedicalSample& sample = samples[sample_index];
                    DatasetItem* item = nullptr;
                    if (!load_image(sample, &item)) continue;
                    DatasetItemUnloadGuard unload(*item);
                    const auto batch_started = std::chrono::steady_clock::now();
                    MedicalDetectorModel::encodeTarget(
                        sample.boxes, grid_h, grid_w, num_classes, &target);
                    poll_viz_live_params();
                    Model::TrainStepRequest request;
                    request.float_inputs["__input__"] = &image_chw;
                    request.target = &target;
                    if (!item->annotation_file.empty() && !item->annotation && !item->loadText())
                        throw std::runtime_error(item->annotation_error);
                    request.annotation=item->annotation?&*item->annotation:nullptr;
                    request.optimizer = &opt;
                    request.learning_rate = optimizer_learning_rate();
                    const bool publish_viz_layers = viz_active &&
                        (global_step == 0 || (global_step + 1) % viz_every_steps == 0);
                    const bool publish_positive_output = viz_active && !sample.boxes.empty() &&
                        (!positive_output_published || publish_viz_layers);
                    request.capture_prediction = publish_positive_output;
                    const auto result = ctx.currentModel->trainStep(request);
                    if (!result) {
                        throw std::runtime_error(
                            "medical_detector: contrat trainStep indisponible");
                    }
                    ++global_step;
                    ++running_steps;
                    running_loss += result->loss;

                    if (publish_positive_output) {
                        const auto detections = MedicalDetectorModel::decodeDetections(
                            result->prediction, grid_h, grid_w, num_classes, 0.0f,
                            nms_iou_threshold, max_detections);
                        publish_detections(*item, detections,
                            sample.boxes,
                            "medical_detector/train/step=" + std::to_string(global_step));
                        positive_output_published = true;
                    }
                    if (publish_viz_layers || publish_positive_output) push_medical_viz_taps();
                    if (global_step % log_every == 0) {
                        ctx.addLog("medical_detector step=" + std::to_string(global_step) +
                                   " loss=" + std::to_string(result->loss) +
                                   " object=" + std::to_string(result->metrics.at("object_loss")) +
                                   " box=" + std::to_string(result->metrics.at("box_loss")) +
                                   " class=" + std::to_string(result->metrics.at("class_loss")) +
                                   " positives=" + std::to_string(
                                       (int)result->metrics.at("positive_cells")));
                    }
                    if (ctx.asyncMonitor) {
                        AsyncMonitor::Metrics metrics;
                        metrics.epoch = epoch_offset + epoch + 1;
                        metrics.total_epochs = total_epochs_display;
                        metrics.batch = static_cast<int>(sample_index) + 1;
                        metrics.total_batches = static_cast<int>(samples.size());
                        metrics.loss = result->loss;
                        metrics.avg_loss = static_cast<float>(running_loss / running_steps);
                        metrics.lr = opt.getCurrentLR();
                        metrics.grad_norm = result->grad_norm;
                        metrics.grad_max = result->grad_max_abs;
                        metrics.params = ctx.currentModel->totalParamCount();
                        metrics.recon_loss_type = "detection_bce_smooth_l1";
                        metrics.opt_type = (int)opt.type;
                        metrics.opt_step = (int)opt.step;
                        metrics.opt_beta1 = opt.beta1;
                        metrics.opt_beta2 = opt.beta2;
                        metrics.opt_eps = opt.eps;
                        metrics.opt_weight_decay = opt.weight_decay;
                        metrics.batch_time_ms = static_cast<int>(
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - batch_started).count());
                        metrics.bps = metrics.batch_time_ms > 0
                            ? 1000.0f / static_cast<float>(metrics.batch_time_ms) : 0.0f;
                        metrics.memory_mb = _mimir_process_resident_memory_mb();
                        metrics.allocator_memory_mb =
                            static_cast<double>(MemoryGuard::instance().getCurrentBytes()) /
                            (1024.0 * 1024.0);
                        metrics.detection.enabled = true;
                        metrics.detection.object_loss = train_metric(*result, "object_loss");
                        metrics.detection.box_loss = train_metric(*result, "box_loss");
                        metrics.detection.class_loss = train_metric(*result, "class_loss");
                        metrics.detection.positive_cells = static_cast<int>(std::lround(
                            train_metric(*result, "positive_cells")));
                        last_medical_metrics = metrics;
                        has_last_medical_metrics = true;
                        ctx.asyncMonitor->updateMetrics(metrics);
                    }
                    if (ctx.asyncMonitor &&
                        ctx.asyncMonitor->consumeStopTrainingRequested()) {
                        stopped_by_ui = true;
                        stopped_epoch = epoch + 1;
                        break;
                    }
                }

                if (!validation_samples.empty() &&
                    ((epoch + 1) % validate_every_epochs) == 0) {
                    const DetectionValidation validation = run_validation(epoch + 1);
                    if (ctx.asyncMonitor && validation.items > 0) {
                        AsyncMonitor::Metrics metrics = has_last_medical_metrics
                            ? last_medical_metrics : AsyncMonitor::Metrics{};
                        metrics.epoch = epoch_offset + epoch + 1;
                        metrics.total_epochs = total_epochs_display;
                        metrics.val_has = true;
                        metrics.val_ok = true;
                        metrics.val_step = global_step;
                        metrics.val_items = validation.items;
                        metrics.val_recon = 1.0f - validation.f1;
                        metrics.val_align = validation.mean_iou;
                        metrics.recon_loss_type = "detection_f1_iou";
                        metrics.detection.enabled = true;
                        metrics.detection.validation_has = true;
                        metrics.detection.precision = validation.precision;
                        metrics.detection.recall = validation.recall;
                        metrics.detection.f1 = validation.f1;
                        metrics.detection.mean_iou = validation.mean_iou;
                        metrics.detection.true_positive = validation.true_positive;
                        metrics.detection.false_positive = validation.false_positive;
                        metrics.detection.false_negative = validation.false_negative;
                        last_medical_metrics = metrics;
                        has_last_medical_metrics = true;
                        ctx.asyncMonitor->updateMetrics(metrics);
                    }
                }

                if (autosave_every_epochs > 0 &&
                    ((epoch + 1) % autosave_every_epochs) == 0) {
                    std::string save_error;
                    if (!do_checkpoint_save(epoch + 1, std::string(), &save_error)) {
                        ctx.addLog("medical_detector autosave: " + save_error);
                    }
                }
                if (stopped_by_ui) break;
            }

            ctx.currentModel->setSerializedOptimizer(opt);
            if (stopped_by_ui) {
                std::string save_error;
                do_checkpoint_save(stopped_epoch, "_stop", &save_error);
                lua_pushboolean(L, false);
                lua_pushstring(L, "STOP_REQUESTED");
                return 2;
            }
            lua_pushboolean(L, true);
            lua_pushinteger(L, global_step);
            return 2;
        } else if (model_type == "vgg16" || model_type == "vgg19") {
            // Multi-label image tag classification.
            // Dataset items must have image_file + text, where text is dot-separated tags/short phrases.
            // Requires cfg.tags_vocab (array of strings) to define fixed class indices.

            int image_w = 0, image_h = 0, image_c = 3;
            int viz_taps_every_steps = 0;
            bool lowercase_tags = true;
            float bce_pos_weight = 1.0f;
            std::vector<float> class_pos_weights;

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

                if (ctx.modelConfig.contains("class_pos_weights") && ctx.modelConfig["class_pos_weights"].is_array()) {
                    const auto& arr = ctx.modelConfig["class_pos_weights"];
                    class_pos_weights.reserve(arr.size());
                    for (const auto& v : arr) {
                        try {
                            class_pos_weights.push_back(std::max(0.0f, v.get<float>()));
                        } catch (...) {
                            class_pos_weights.push_back(0.0f);
                        }
                    }
                }

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
            if (!class_pos_weights.empty() && (int)class_pos_weights.size() != num_classes) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "vgg16/vgg19 multi-label: cfg.class_pos_weights doit matcher cfg.tags_vocab");
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
                    bf.pixels_real = std::move(f.pixels_real);
                    bf.heatmap_kind = f.heatmap_kind;
                    bf.tensor_info = std::move(f.tensor_info);
                    bf.tensor_values = std::move(f.tensor_values);
                    bf.tensor_indices = std::move(f.tensor_indices);
                    bf.tensor_sample_channels = f.tensor_sample_channels;
                            bf.label = std::move(f.label);
                    frames.push_back(std::move(bf));
                }
                ctx.asyncMonitor->setLayerBlockImages(std::move(frames));
            };

            // Filter usable items
            std::vector<int> indices;
            indices.reserve(ctx.currentDataset.size());
            for (size_t i = 0; i < ctx.currentDataset.size(); ++i) {
                const auto& it = ctx.currentDataset[i];
                if (!it.image_file.empty() && (!it.text_file.empty() || !it.text_inline.empty() || it.text.has_value())) {
                    indices.push_back((int)i);
                }
            }
            if (indices.empty()) {
                lua_pushboolean(L, false);
                lua_pushstring(L, "Dataset: aucun item avec image_file + texte exploitable (text_file ou text_inline) pour vgg16/vgg19 multi-label");
                return 2;
            }
            std::shuffle(indices.begin(), indices.end(), rng);
            if (max_items > 0 && (int)indices.size() > max_items) indices.resize((size_t)max_items);

            const bool validation_enabled = (validate_items > 0) && ((validate_every_steps > 0) || (validate_every_epochs > 0));
            if (ctx.asyncMonitor) {
                ctx.asyncMonitor->updateRuntimeValidationEnabled(validation_enabled);
            }
            auto validation_runtime_enabled = [&]() {
                return validation_enabled &&
                    (!ctx.asyncMonitor || ctx.asyncMonitor->validationEnabledSnapshot());
            };

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
            double running_loss_sum = 0.0;
            int running_loss_count = 0;

            auto split_tags = [&](const DatasetItem& item) {
                auto out = ctx.modelConfig.value("use_dataset_annotation_labels", false)
                    ? TagAnnotation::labels(item.text.value(),item.annotation ? &*item.annotation : nullptr,lowercase_tags)
                    : PromptParsing::extractLabelCandidates(item.text.value(), lowercase_tags);
                for (auto& s : out) trim_punct(s);
                out.erase(std::remove_if(out.begin(), out.end(), [](const std::string& s) {
                    return s.empty();
                }), out.end());
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
            auto pos_weight_for_class = [&](int class_id) -> double {
                if (class_id >= 0 && class_id < (int)class_pos_weights.size()) {
                    const double w = (double)class_pos_weights[(size_t)class_id];
                    if (w > 0.0) return w;
                }
                return (double)bce_pos_weight;
            };

            auto run_validation = [&](int step, int epoch_1based) -> ValStats {
                ValStats st;
                if (!validation_runtime_enabled()) return st;
                if (!ctx.currentModel) return st;
                if (val_indices.empty()) return st;

                if (ctx.asyncMonitor && ctx.asyncMonitor->getViz()) {
                    ctx.currentModel->clearVizTaps();
                }

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
                    if (item.image_file.empty() || (!item.text.has_value() && item.text_file.empty() && item.text_inline.empty())) continue;
                    DatasetItemUnloadGuard unload_item(item);
                    if (!item.loadText() || !item.text.has_value()) continue;

                    item.loadImageRGB(image_w, image_h);
                    if (!item.img_loaded || item.img.size() != expected_u8) continue;

                    ctx.currentModel->applyRuntimeConfiguration();
                    const std::vector<float> mask_weights =
                        _mimir_load_spatial_loss_weights(item, image_w, image_h, image_c,
                            ctx.currentModel->modelConfig.value("bbox_loss_weight",1.0f),ctx.currentModel->modelConfig).values;

                    // Labels
                    std::fill(y.begin(), y.end(), 0.0f);
                    const std::vector<std::string> tags = split_tags(item);
                    for (const auto& t : tags) {
                        auto it = tag_to_id.find(t);
                        if (it != tag_to_id.end()) {
                            const int id = it->second;
                            if (id >= 0 && id < num_classes) {
                                if (y[(size_t)id] < 0.5f) y[(size_t)id] = 1.0f;
                            }
                        }
                    }

                    // Image -> CHW
                    for (int yy = 0; yy < image_h; ++yy) {
                        for (int xx = 0; xx < image_w; ++xx) {
                            const size_t pix = (size_t)(yy * image_w + xx) * (size_t)image_c;
                            for (int cc = 0; cc < image_c; ++cc) {
                                const size_t hwc_idx = pix + (size_t)cc;
                                float v = (float)((double)item.img[hwc_idx] / 127.5 - 1.0);
                                if (mask_weights.size() == expected_u8) v *= mask_weights[hwc_idx];
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
                        const double wpos = (yi >= 0.5) ? pos_weight_for_class(i) : 1.0;
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
                _mimir_publish_viz_taps(ctx, true);
                if (done <= 0) return st;

                const double denom = (double)done * (double)std::max(1, num_classes);
                st.loss = loss_sum / (double)done;
                st.pos_true_rate = denom > 0 ? (double)true_pos / denom : 0.0;
                st.pos_pred_rate = denom > 0 ? (double)pred_pos / denom : 0.0;
                st.avg_prob = denom > 0 ? prob_sum / denom : 0.0;
                const double f1_den = (2.0 * (double)tp + (double)fp + (double)fn);
                st.f1_micro = (f1_den > 0.0) ? (2.0 * (double)tp) / f1_den : 0.0;

                ctx.addLog(std::string("val") +
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
                    vm.loss = (float)st.loss;
                    vm.avg_loss = (float)st.loss;
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
            int stopped_epoch = 0;

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

                m.memory_mb = _mimir_process_resident_memory_mb();
                m.allocator_memory_mb =
                    static_cast<double>(MemoryGuard::instance().getCurrentBytes()) / (1024.0 * 1024.0);
            };
            bool warned_no_vocab_match = false;

            for (int epoch = 0; epoch < epochs; ++epoch) {
                std::shuffle(train_indices.begin(), train_indices.end(), rng);
                ctx.addLog("Epoch " + std::to_string(epoch_offset + (epoch + 1)) + "/" + std::to_string(total_epochs_display) +
                           " (" + std::string(model_type) + " multi-label) items=" + std::to_string(use_n));

                bool stop_requested = false;

                for (int k = 0; k < use_n; ++k) {
                    refresh_viz_activation();
                    DatasetItem& item = ctx.currentDataset[(size_t)train_indices[(size_t)k]];
                    if (item.image_file.empty() || (!item.text.has_value() && item.text_file.empty() && item.text_inline.empty())) continue;
                    DatasetItemUnloadGuard unload_item(item);

                    // Lazy-load text on demand (uses DatasetMemoryManager).
                    if (!item.loadText() || !item.text.has_value()) {
                        continue;
                    }

                    item.loadImageRGB(image_w, image_h);
                    if (!item.img_loaded || item.img.size() != expected_u8) {
                        continue;
                    }

                    ctx.currentModel->applyRuntimeConfiguration();
                    const std::vector<float> mask_weights =
                        _mimir_load_spatial_loss_weights(item, image_w, image_h, image_c,
                            ctx.currentModel->modelConfig.value("bbox_loss_weight",1.0f),ctx.currentModel->modelConfig).values;

                    // Build labels (multi-hot)
                    std::fill(y.begin(), y.end(), 0.0f);
                    const std::vector<std::string> tags = split_tags(item);
                    int active_labels = 0;
                    for (const auto& t : tags) {
                        auto it = tag_to_id.find(t);
                        if (it != tag_to_id.end()) {
                            const int id = it->second;
                            if (id >= 0 && id < num_classes) {
                                if (y[(size_t)id] < 0.5f) {
                                    y[(size_t)id] = 1.0f;
                                    active_labels++;
                                }
                            }
                        }
                    }

                    // Normalize u8 -> [-1, 1] AND permute HWC -> CHW (Conv2d expects CHW)
                    for (int yy = 0; yy < image_h; ++yy) {
                        for (int xx = 0; xx < image_w; ++xx) {
                            const size_t pix = (size_t)(yy * image_w + xx) * (size_t)image_c;
                            for (int cc = 0; cc < image_c; ++cc) {
                                const size_t hwc_idx = pix + (size_t)cc;
                                float v = (float)((double)item.img[hwc_idx] / 127.5 - 1.0);
                                if (mask_weights.size() == expected_u8) v *= mask_weights[hwc_idx];
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
                        const double wpos = (yi >= 0.5) ? pos_weight_for_class(i) : 1.0;
                        loss += (1.0 - yi) * sp_pos + yi * wpos * sp_neg;
                        const double p = sigmoid(z);
                        const double gw = (yi >= 0.5) ? pos_weight_for_class(i) : 1.0;
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
                    ctx.currentModel->optimizerStep(opt, optimizer_learning_rate(), nullptr);
                    running_loss_sum += loss;
                    running_loss_count += 1;

                    global_step += 1;

                    if (viz_active && ctx.asyncMonitor && ctx.asyncMonitor->getViz() != nullptr &&
                        viz_taps_every_steps > 0 && ((global_step % viz_taps_every_steps) == 0)) {
                        std::string label = std::string(model_type) + "/input/dataset/rgb";
                        label += "/i=" + std::to_string(train_indices[(size_t)k]);
                        const auto viz_payload = VizTextPayload::buildDatasetTextPayload(
                            ctx.currentModel.get(),
                            ctx.currentTokenizer.get(),
                            item.text.value(),
                            nullptr,
                            -1,
                            false,
                            true,
                            true
                        );
                        ctx.asyncMonitor->setDatasetSample(
                            item.img,
                            image_w,
                            image_h,
                            image_c,
                            label,
                            viz_payload.prompt,
                            viz_payload.tags,
                            viz_payload.tokens,
                            viz_payload.encoding,
                            _mimir_dataset_bbox_active(item,*ctx.currentModel),
                            DatasetBBoxOverlay::fromAnnotation(item.annotation ? &*item.annotation : nullptr),
                            item.annotation ? &*item.annotation : nullptr, &ctx.currentModel->modelConfig
                        );
                    }

                    if (viz_taps_every_steps > 0 && (global_step % viz_taps_every_steps) == 0) {
                        push_viz_taps();
                    }

                    if (validation_runtime_enabled() && validate_every_steps > 0 && (global_step % validate_every_steps) == 0) {
                        const auto vs = run_validation(global_step, (epoch + 1));
                        if (vs.items > 0) apply_val_feedback(static_cast<float>(vs.loss), global_step);
                    }

                    if ((global_step % log_every) == 0) {
                        const double denom = (double)std::max(1, num_classes);
                        const double pos_true_rate = denom > 0.0 ? (double)true_pos / denom : 0.0;
                        const double pos_pred_rate = denom > 0.0 ? (double)pred_pos / denom : 0.0;
                        const double avg_prob = denom > 0.0 ? prob_sum / denom : 0.0;
                        const double f1_den = (2.0 * (double)tp + (double)fp + (double)fn);
                        const double f1_micro = (f1_den > 0.0) ? (2.0 * (double)tp) / f1_den : 0.0;
                        const double avg_loss = (running_loss_count > 0) ? (running_loss_sum / (double)running_loss_count) : loss;
                        ctx.addLog("step=" + std::to_string(global_step) +
                                   " loss=" + std::to_string((float)loss) +
                                   " avg_loss=" + std::to_string((float)avg_loss) +
                                   " lr=" + std::to_string(opt.getCurrentLR()) +
                                   " pos_true=" + std::to_string((float)pos_true_rate) +
                                   " pos_pred=" + std::to_string((float)pos_pred_rate) +
                                   " f1_micro=" + std::to_string((float)f1_micro) +
                                   " active_labels=" + std::to_string(active_labels) +
                                   " label_candidates=" + std::to_string((int)tags.size()));

                        if (!warned_no_vocab_match && ((int)tags.size() > 0) && active_labels == 0) {
                            warned_no_vocab_match = true;
                            ctx.addLog("⚠️  Aucun label ne matche tags_vocab (active_labels=0). "
                                       "Parsing supporté: JSON standard (`tags`/`keywords`/`tokens`) puis fallback séparateurs `.,;|` et nouvelles lignes.");
                        }
                    }
                    if (ctx.asyncMonitor) {
                        AsyncMonitor::Metrics m;
                        m.epoch = epoch_offset + (epoch + 1);
                        m.total_epochs = total_epochs_display;
                        m.batch = k + 1;
                        m.total_batches = use_n;
                        m.loss = (float)loss;
                        m.avg_loss = (float)((running_loss_count > 0) ? (running_loss_sum / (double)running_loss_count) : loss);
                        m.lr = opt.getCurrentLR();
                        m.mse = 0.0f;
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
                        ctx.addLog("⛔ Arrêt sécurisé demandé. Sauvegarde et arrêt...");
                        stop_requested = true;
                        stopped_by_ui = true;
                        stopped_epoch = epoch + 1;
                        break;
                    }
                }

                if (validation_runtime_enabled() && validate_every_epochs > 0) {
                    const int epoch_1based = epoch + 1;
                    if ((epoch_1based % validate_every_epochs) == 0) {
                        const auto vse = run_validation(global_step, epoch_1based);
                        if (vse.items > 0) apply_val_feedback(static_cast<float>(vse.loss), global_step);
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
                std::string save_error;
                if (!do_checkpoint_save(stopped_epoch, "_stop", &save_error) && !save_error.empty()) {
                    ctx.addLog("⚠ Save(stop) échoué: " + save_error);
                }
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
                    refresh_viz_activation();
                    DatasetItem& item = ctx.currentDataset[(size_t)indices[(size_t)k]];
                    if (item.text_file.empty()) continue;
                    DatasetItemUnloadGuard unload_item(item);
                    if (!item.text.has_value()) item.loadText();
                    if (!item.text.has_value()) continue;

                    std::vector<int> ids = ctx.currentTokenizer->tokenize(item.text.value());
                    if ((int)ids.size() < seq_len) ids.resize((size_t)seq_len, pad_id);
                    else if ((int)ids.size() > seq_len) ids.resize((size_t)seq_len);

                    poll_viz_live_params();
                    Model::TrainStepRequest request;
                    request.int_inputs["text_ids"] = &ids;
                    request.optimizer = &opt;
                    request.learning_rate = optimizer_learning_rate();
                    const auto step_result = ctx.currentModel->trainStep(request);
                    if (!step_result) {
                        throw std::runtime_error("Le modèle '" + model_type + "' ne fournit pas de contrat d'entraînement compatible");
                    }
                    const Model::TrainStepResult& st = *step_result;

                    // Entraînement texte mono-modalité: remplir uniquement le vecteur texte (seq).
                    try {
                        auto& enc = ctx.currentModel->getMutableEncoder();
                        enc.ensureVocabSize(ctx.currentTokenizer->getVocabSize());
                        enc.ensureSpecialEmbeddings();
                        enc.fillTextVectorSingleModality(ids, pad_id, 0.005f, true);
                    } catch (...) {
                        // best-effort: ne jamais casser le train principal.
                    }

                    global_step += 1;
                    log_step(global_step, st, "[vae_text]");
                    monitor_step(epoch + 1, k + 1, use_n, st);

                    if (ctx.asyncMonitor && ctx.asyncMonitor->consumeStopTrainingRequested()) {
                        ctx.addLog("⛔ Arrêt sécurisé demandé. Sauvegarde et arrêt...");
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

// Renvoie toutes les infos présentes dans le registry pour une architecture.
// Sans argument: renvoie un tableau (liste) d'entrées complètes pour toutes
// les architectures. Avec un nom: renvoie l'entrée correspondante.
// Chaque entrée contient aussi origin="native"|"mpk" et, pour un MPK,
// source_path=<fichier chargé>.
int LuaScripting::lua_archInfo(lua_State* L) {
    auto pushEntry = [L](const ModelArchitectures::Entry& e) {
        lua_newtable(L);
        lua_pushstring(L, e.name.c_str());
        lua_setfield(L, -2, "name");
        lua_pushstring(L, e.description.c_str());
        lua_setfield(L, -2, "description");
        lua_pushstring(L, e.origin.c_str());
        lua_setfield(L, -2, "origin");
        if (!e.source_path.empty()) {
            lua_pushstring(L, e.source_path.c_str());
            lua_setfield(L, -2, "source_path");
        }
        jsonToLuaTable(L, e.default_config);
        lua_setfield(L, -2, "config");
    };

    try {
        auto& reg = ModelArchitectures::Registry::instance();

        // Cas 1: un nom est fourni -> entrée unique.
        if (lua_gettop(L) >= 1 && !lua_isnil(L, 1)) {
            const char* name = luaL_checkstring(L, 1);
            const ModelArchitectures::Entry* e = reg.find(name);
            if (e == nullptr) {
                lua_pushnil(L);
                lua_pushstring(L, (std::string("unknown architecture: ") + name).c_str());
                return 2;
            }
            pushEntry(*e);
            return 1;
        }

        // Cas 2: aucun nom -> liste complète de toutes les entrées.
        const auto names = ModelArchitectures::available();
        lua_createtable(L, static_cast<int>(names.size()), 0);
        int i = 1;
        for (const auto& name : names) {
            const ModelArchitectures::Entry* e = reg.find(name);
            if (e != nullptr) {
                pushEntry(*e);
                lua_rawseti(L, -2, i++);
            }
        }
        return 1;
    } catch (const std::exception& e) {
        lua_pushnil(L);
        lua_pushstring(L, e.what());
        return 2;
    }
}

// Renvoie la liste des dtypes pris en charge par le framework.
// Chaque entrée: { name=<canonique>, aliases=<csv>, bytes=<n>, kind="float|int|uint|bool" }.
int LuaScripting::lua_archDtypes(lua_State* L) {
    struct DtInfo {
        Mimir::DType dt;
        const char* aliases;
        const char* kind;
    };
    // Ordre logique d'affichage (flottants d'abord, puis entiers, puis bool).
    static const DtInfo kAll[] = {
        { Mimir::DType::F32,  "float, f32, float32",   "float" },
        { Mimir::DType::F16,  "f16, float16, fp16",    "float" },
        { Mimir::DType::BF16, "bf16, bfloat16",        "float" },
        { Mimir::DType::F64,  "double, f64, float64",  "float" },
        { Mimir::DType::I8,   "i8, int8",              "int"   },
        { Mimir::DType::I16,  "i16, int16",            "int"   },
        { Mimir::DType::I32,  "i32, int32",            "int"   },
        { Mimir::DType::I64,  "i64, int64",            "int"   },
        { Mimir::DType::U8,   "u8, uint8",             "uint"  },
        { Mimir::DType::U16,  "u16, uint16",           "uint"  },
        { Mimir::DType::U32,  "u32, uint32",           "uint"  },
        { Mimir::DType::U64,  "u64, uint64",           "uint"  },
        { Mimir::DType::BOOL, "bool, b1",              "bool"  },
    };

    const int n = static_cast<int>(sizeof(kAll) / sizeof(kAll[0]));
    lua_createtable(L, n, 0);
    for (int i = 0; i < n; ++i) {
        const auto& e = kAll[i];
        lua_newtable(L);
        lua_pushstring(L, Mimir::dtype_to_string(e.dt));
        lua_setfield(L, -2, "name");
        lua_pushstring(L, e.aliases);
        lua_setfield(L, -2, "aliases");
        lua_pushinteger(L, static_cast<lua_Integer>(Mimir::dtype_size_bytes(e.dt)));
        lua_setfield(L, -2, "bytes");
        lua_pushstring(L, e.kind);
        lua_setfield(L, -2, "kind");
        lua_rawseti(L, -2, i + 1);
    }
    return 1;
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
        ConditioningEncoder encoder;
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
            ctx.currentEncoder = std::make_shared<ConditioningEncoder>(encoder);

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
    
    // Caller-owned loop state (cursor, RNG, validation feedback) travels with model_config.
    if (lua_istable(L, 3)) {
        lua_getfield(L, 3, "training_state");
        if (lua_istable(L, -1)) ctx.currentModel->modelConfig["training_state"] = luaTableToJson(L, -1);
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

        lua_getfield(L, options_idx, "mapping_json");
        if (lua_isstring(L, -1)) options.mapping_json = lua_tostring(L, -1);
        lua_pop(L, 1);

        lua_getfield(L, options_idx, "tensor_mapping_json");
        if (options.mapping_json.empty() && lua_isstring(L, -1)) options.mapping_json = lua_tostring(L, -1);
        lua_pop(L, 1);
    }

    // Load
    std::string error;
    bool success = Mimir::Serialization::load_checkpoint(*ctx.currentModel, path, options, &error);

    if (success) {
        // En mode reprise, la sérialisation hydrate les assets dans le modèle.
        // On resynchronise le contexte Lua pour que l'entraînement utilise bien
        // le tokenizer/encoder du checkpoint (et non un asset précédemment chargé).
        if (options.load_tokenizer && ctx.currentModel) {
            ctx.currentTokenizer = ctx.currentModel->getHasTokenizer()
                ? std::make_shared<Tokenizer>(ctx.currentModel->getTokenizer()) : nullptr;
        }
        if (options.load_encoder && ctx.currentModel) {
            ctx.currentEncoder = ctx.currentModel->getHasEncoder()
                ? std::make_shared<ConditioningEncoder>(ctx.currentModel->getEncoder()) : nullptr;
        }

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

int LuaScripting::lua_conditioningPrompt(lua_State* L) {
    const std::string caption=luaL_checkstring(L,1);
    if(!lua_isnoneornil(L,2))luaL_checktype(L,2,LUA_TTABLE);
    try {
        const json annotation=lua_isnoneornil(L,2)?json():luaTableToJson(L,2);
        const auto prompt=BBoxSupervision::groundedPrompt(caption,annotation.is_null()?nullptr:&annotation);
        lua_pushlstring(L,prompt.data(),prompt.size());return 1;
    } catch(const std::exception& e) {
        lua_pushnil(L);lua_pushstring(L,e.what());return 2;
    }
}
