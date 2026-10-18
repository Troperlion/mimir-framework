#include "scriptings/ScriptingBridgeCommon.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include "DType.hpp"
#include "Models/Registry/ModelArchitectures.hpp"

namespace ScriptingBridgeCommon {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

std::string ensureRootWorkEnv() {
    std::filesystem::path root;
    if (const char* configured = std::getenv(kEnvRootWork); configured && *configured) {
        root = configured;
    } else {
        root = std::filesystem::current_path();
    }

    root = std::filesystem::absolute(root).lexically_normal();
    const std::string value = root.string();
#ifdef _WIN32
    _putenv_s(kEnvRootWork, value.c_str());
#else
    setenv(kEnvRootWork, value.c_str(), 1);
#endif
    return value;
}

std::string archCacheFilePath() {
    // Cache stable entre runs dans /tmp (non critique, best-effort).
    return std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
           "/mimir_bridge_arch_cache.json";
}

std::string loadArchCacheJson() {
    std::ifstream f(archCacheFilePath());
    if (!f.is_open()) return "{}";
    std::string s((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());
    return s.empty() ? "{}" : s;
}

static void updateArchCache(const std::string& archName, size_t totalParams) {
    json cache;
    {
        std::ifstream f(archCacheFilePath());
        if (f.is_open()) {
            try { f >> cache; } catch (...) { cache = json::object(); }
        } else {
            cache = json::object();
        }
    }
    cache[archName]["total_params"] = totalParams;
    std::ofstream out(archCacheFilePath());
    if (out.is_open()) out << cache.dump();
}

// ---------------------------------------------------------------------------
// Data builders
// ---------------------------------------------------------------------------

std::string buildArchitecturesInfoJson() {
    json out = json::array();
    auto& reg = ModelArchitectures::Registry::instance();
    for (const auto& name : ModelArchitectures::available()) {
        const auto* e = reg.find(name);
        if (!e) continue;
        out.push_back({
            {"name", e->name},
            {"description", e->description},
            {"config", e->default_config}
        });
    }
    return out.dump();
}

std::string buildAvailableJson() {
    json out = json::array();
    for (const auto& name : ModelArchitectures::available())
        out.push_back(name);
    return out.dump();
}

std::string buildDtypesJson() {
    json out = json::array();
    auto push = [&](Mimir::DType dt, const char* aliases_csv, const char* kind) {
        // Normalisation: aliases est un array de strings (pas une string CSV),
        // ce qui correspond a la semantique Lua (table indexee) tout en etant
        // idiomatique en JS/C#/Rust.
        json aliases = json::array();
        std::istringstream ss(aliases_csv);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            const auto s = tok.find_first_not_of(" \t");
            const auto e = tok.find_last_not_of(" \t");
            if (s != std::string::npos)
                aliases.push_back(tok.substr(s, e - s + 1));
        }
        out.push_back({
            {"name",    Mimir::dtype_to_string(dt)},
            {"aliases", aliases},
            {"bytes",   Mimir::dtype_size_bytes(dt)},
            {"kind",    kind}
        });
    };
    push(Mimir::DType::F32, "float, f32, float32", "float");
    push(Mimir::DType::F16, "f16, float16, fp16", "float");
    push(Mimir::DType::BF16, "bf16, bfloat16", "float");
    push(Mimir::DType::F64, "double, f64, float64", "float");
    push(Mimir::DType::I8, "i8, int8", "int");
    push(Mimir::DType::I16, "i16, int16", "int");
    push(Mimir::DType::I32, "i32, int32", "int");
    push(Mimir::DType::I64, "i64, int64", "int");
    push(Mimir::DType::U8, "u8, uint8", "uint");
    push(Mimir::DType::U16, "u16, uint16", "uint");
    push(Mimir::DType::U32, "u32, uint32", "uint");
    push(Mimir::DType::U64, "u64, uint64", "uint");
    push(Mimir::DType::BOOL, "bool, b1", "bool");
    return out.dump();
}

bool processBridgeCommands(ScriptingContext& ctx,
                           const std::filesystem::path& cmdFile,
                           const std::string& logPrefix) {
    std::ifstream in(cmdFile);
    if (!in.is_open()) return true;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;

        if (line.rfind("Model.create|", 0) == 0) {
            // Protocole: Model.create|name  ou  Model.create|name|{json_cfg}
            const std::string rest = line.substr(std::string("Model.create|").size());
            std::string name;
            json cfg;
            const auto pipe = rest.find('|');
            if (pipe == std::string::npos) {
                name = rest;
            } else {
                name = rest.substr(0, pipe);
                const std::string cfg_str = rest.substr(pipe + 1);
                if (!cfg_str.empty()) {
                    try { cfg = json::parse(cfg_str); } catch (...) {}
                }
            }
            if (name.empty()) {
                ctx.addLog(logPrefix + " bridge Model.create failed: empty architecture name");
                return false;
            }
            if (!cfg.is_object() || cfg.empty())
                cfg = ModelArchitectures::defaultConfig(name);
            ctx.currentModel = ModelArchitectures::create(name, cfg);
            ctx.modelType = name;
            ctx.modelConfig = cfg;
            ctx.addLog(logPrefix + " bridge Model.create(" + name + ")");
            continue;
        }

        if (line == "Model.allocate_params") {
            if (!ctx.currentModel) {
                ctx.addLog(logPrefix + " bridge Model.allocate_params failed: no current model");
                return false;
            }
            ctx.currentModel->allocateParams();
            const size_t n = ctx.currentModel->totalParamCount();
            ctx.addLog(logPrefix + " bridge Model.allocate_params -> " +
                       std::to_string(n) + " params");
            // Mettre à jour le cache inter-exécutions.
            if (!ctx.modelType.empty()) updateArchCache(ctx.modelType, n);
            continue;
        }

        if (line.rfind("Model.init_weights|", 0) == 0) {
            if (!ctx.currentModel) {
                ctx.addLog(logPrefix + " bridge Model.init_weights failed: no current model");
                return false;
            }
            const std::string payload = line.substr(std::string("Model.init_weights|").size());
            const auto sep = payload.find('|');
            const std::string method =
                (sep == std::string::npos) ? payload : payload.substr(0, sep);
            unsigned int seed = 0;
            if (sep != std::string::npos) {
                try {
                    seed = static_cast<unsigned int>(std::stoul(payload.substr(sep + 1)));
                } catch (...) {
                    seed = 0;
                }
            }
            ctx.currentModel->initializeWeights(method.empty() ? "he" : method, seed);
            ctx.addLog(logPrefix + " bridge Model.init_weights(method=" +
                       (method.empty() ? std::string("he") : method) + ")");
            continue;
        }

        if (line == "Model.total_params") {
            const size_t n = ctx.currentModel ? ctx.currentModel->totalParamCount() : 0;
            ctx.addLog(logPrefix + " bridge Model.total_params -> " + std::to_string(n));
            if (ctx.currentModel && !ctx.modelType.empty())
                updateArchCache(ctx.modelType, n);
            continue;
        }

        ctx.addLog(logPrefix + " bridge ignored unknown command: " + line);
    }
    return true;
}

}  // namespace ScriptingBridgeCommon
