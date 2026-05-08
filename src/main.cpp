#include "Model.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "LuaScripting.hpp"
#include "Helpers.hpp"
#include "HtopDisplay.hpp"
#include "Visualizer.hpp"
#include "MemorySafety.hpp"
#include "Tokenizer.hpp"
#include "Encoder.hpp"
#include "ConfigOverrides.hpp"
#include "include/json.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <cstdlib>
#include <cerrno>
#ifdef _OPENMP
#include <omp.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace ModelArchitectures;

static bool applyOverride(json& target, const std::string& expr, std::string& err)
{
    return Mimir::ConfigOverrides::applyOverride(target, expr, err);
}

static bool readJsonFile(const std::string& path, json& out, std::string& err)
{
    std::ifstream f(path);
    if (!f) {
        err = "Impossible d'ouvrir le fichier: " + path;
        return false;
    }
    try {
        f >> out;
    } catch (const std::exception& e) {
        err = std::string("Erreur JSON: ") + e.what();
        return false;
    }
    return true;
}

static void pushJsonToLua(lua_State* L, const json& v, int depth = 0)
{
    if (!L) return;
    if (depth > 128) {
        lua_pushstring(L, "<max-depth>");
        return;
    }

    if (v.is_null()) {
        lua_pushnil(L);
        return;
    }
    if (v.is_boolean()) {
        lua_pushboolean(L, v.get<bool>() ? 1 : 0);
        return;
    }
    if (v.is_number_integer()) {
        const long long x = v.get<long long>();
        lua_pushinteger(L, static_cast<lua_Integer>(x));
        return;
    }
    if (v.is_number_unsigned()) {
        const unsigned long long x = v.get<unsigned long long>();
        lua_pushinteger(L, static_cast<lua_Integer>(x));
        return;
    }
    if (v.is_number_float()) {
        lua_pushnumber(L, static_cast<lua_Number>(v.get<double>()));
        return;
    }
    if (v.is_string()) {
        const std::string s = v.get<std::string>();
        lua_pushlstring(L, s.c_str(), s.size());
        return;
    }
    if (v.is_array()) {
        lua_createtable(L, static_cast<int>(v.size()), 0);
        int idx = 1;
        for (const auto& it : v) {
            pushJsonToLua(L, it, depth + 1);
            lua_rawseti(L, -2, idx++);
        }
        return;
    }
    if (v.is_object()) {
        lua_createtable(L, 0, static_cast<int>(v.size()));
        for (auto it = v.begin(); it != v.end(); ++it) {
            const std::string& k = it.key();
            lua_pushlstring(L, k.c_str(), k.size());
            pushJsonToLua(L, it.value(), depth + 1);
            lua_settable(L, -3);
        }
        return;
    }

    // Fallback: dump as string.
    const std::string dumped = v.dump();
    lua_pushlstring(L, dumped.c_str(), dumped.size());
}

void printUsage(const char *prog)
{
    std::cout << "Usage: " << prog << " [OPTIONS]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --lua <script.lua>       Exécuter un script Lua\n";
    std::cout << "  --config <config.json>   Charger et entraîner depuis config\n";
    std::cout << "  --conf <config.json>     Charger une conf et exécuter lua.scripts\n";
    std::cout << "  --override <path=value>  Override (répétable) appliqué à la config du modèle\n";
    std::cout << "  --help                   Afficher cette aide\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " --lua scripts/test_lua_api.lua\n";
    std::cout << "  " << prog << " --config config.json\n";
    std::cout << "  " << prog << " --conf config.json\n";
    std::cout << "  " << prog << " --config config.json --override max_vocab=64000\n";
    std::cout << "  " << prog << " --config config.json --override optimizer=\"adamw\" --override weight_decay=0.01\n";
}

int main(int argc, char **argv)
{
    std::cout << "╔════════════════════════════════════════╗\n";
    std::cout << "║       Mímir Framework v2.4.0           ║\n";
    std::cout << "║     Deep Learning Architectures        ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    // 🛡️ SÉCURITÉ MÉMOIRE: Vérification au démarrage
    std::cout << "🛡️  Vérification de la sécurité mémoire...\n";
    MemorySafety::validateLegacyDisabled();
    MemorySafety::runMemoryIntegrityTest();
    std::cout << "\n";
    
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    std::cout << "🔧 OpenMP: " << num_threads << " threads disponibles\n";
    omp_set_num_threads(num_threads);
#endif
    
    std::cout << "🚀 Optimisations hardware:\n";
    std::cout << "  • AVX2: " << (Model::hasAVX2() ? "✓" : "✗") << "\n";
    std::cout << "  • FMA: " << (Model::hasFMA() ? "✓" : "✗") << "\n";
    std::cout << "  • F16C: " << (Model::hasF16C() ? "✓" : "✗") << "\n";
    std::cout << "  • BMI2: " << (Model::hasBMI2() ? "✓" : "✗") << "\n";
    std::cout << "\n";
    
    if (argc < 2) {
        std::cout << "💡 Utilisez --help pour voir les options\n";
        std::cout << "💡 Ou:      " << argv[0] << " --lua scripts/test_lua_api.lua\n\n";
        return 0;
    }

    // Mode --lua: on passe tous les args après le script à Lua (ne pas valider ici).
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        if (std::string(argv[i]) == "--lua" && i + 1 < argc) {
            const std::string lua_script = argv[++i];
            std::cout << "📜 Exécution du script Lua: " << lua_script << "\n";
            std::cout << "═══════════════════════════════════════════════\n\n";
            
            if (!fs::exists(lua_script)) {
                std::cerr << "❌ Fichier non trouvé: " << lua_script << "\n";
                return 1;
            }
            
            try {
                LuaScripting lua;
                std::vector<std::string> script_args;
                for (int j = i + 1; j < argc; ++j) {
                    script_args.emplace_back(argv[j]);
                }
                lua.setArgs(lua_script, script_args);
                lua.loadScript(lua_script);
                std::cout << "\n✅ Script Lua exécuté avec succès\n";
            } catch (const std::exception& e) {
                std::cerr << "❌ Erreur Lua: " << e.what() << "\n";
                return 1;
            }
            
            return 0;
        }
    }

    // Mode --config
    std::string config_path;
    std::string conf_path;
    std::vector<std::string> overrides;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (a == "--conf" && i + 1 < argc) {
            conf_path = argv[++i];
        } else if (a == "--override" && i + 1 < argc) {
            overrides.emplace_back(argv[++i]);
        } else if (a == "--help") {
            // déjà géré plus haut
        } else if (a == "--lua") {
            // déjà géré plus haut
            ++i;
        } else {
            // En mode config on reste strict pour éviter les typos silencieuses.
            if (!config_path.empty()) {
                std::cerr << "❌ Option inconnue en mode --config: " << a << "\n";
                std::cerr << "💡 Utilisez --help pour la liste des options\n";
                return 1;
            }
            if (!conf_path.empty()) {
                std::cerr << "❌ Option inconnue en mode --conf: " << a << "\n";
                std::cerr << "💡 Utilisez --help pour la liste des options\n";
                return 1;
            }
        }
    }

    if (!conf_path.empty()) {
        std::cout << "⚙️  Chargement de la conf: " << conf_path << "\n";

        if (!fs::exists(conf_path)) {
            std::cerr << "❌ Fichier non trouvé: " << conf_path << "\n";
            return 1;
        }

        json conf;
        {
            std::string err;
            if (!readJsonFile(conf_path, conf, err)) {
                std::cerr << "❌ " << err << "\n";
                return 1;
            }
        }

        if (!overrides.empty()) {
            std::cout << "🧩 Application des overrides (--override) sur la conf:\n";
            for (const auto& o : overrides) {
                std::string err;
                if (!applyOverride(conf, o, err)) {
                    std::cerr << "❌ " << err << "\n";
                    return 1;
                }
                std::cout << "  • " << o << "\n";
            }
            std::cout << "\n";
        }

        const json* lua_conf = nullptr;
        if (conf.contains("lua") && conf["lua"].is_object()) {
            lua_conf = &conf["lua"];
        } else if (conf.contains("run") && conf["run"].is_object() && conf["run"].contains("lua") && conf["run"]["lua"].is_object()) {
            lua_conf = &conf["run"]["lua"];
        }

        if (!lua_conf) {
            std::cerr << "❌ --conf: aucune section lua trouvée (attendu: lua.scripts ou run.lua.scripts)\n";
            return 1;
        }

        std::vector<json> scripts;
        if (lua_conf->contains("scripts") && (*lua_conf)["scripts"].is_array()) {
            for (const auto& it : (*lua_conf)["scripts"]) scripts.push_back(it);
        } else if (lua_conf->contains("script")) {
            scripts.push_back((*lua_conf)["script"]);
        }

        if (scripts.empty()) {
            std::cerr << "❌ --conf: aucune entrée de script (attendu: lua.scripts=[...] ou lua.script)\n";
            return 1;
        }

        const std::string conf_abs = fs::absolute(conf_path).string();
        const std::string conf_dir = fs::absolute(fs::path(conf_path)).parent_path().string();

        for (size_t si = 0; si < scripts.size(); ++si) {
            const json& s = scripts[si];

            std::string script_path;
            std::vector<std::string> script_args;

            if (s.is_string()) {
                script_path = s.get<std::string>();
            } else if (s.is_object()) {
                script_path = s.value("script", "");
                if (s.contains("args") && s["args"].is_array()) {
                    for (const auto& a : s["args"]) {
                        if (a.is_string()) script_args.push_back(a.get<std::string>());
                        else script_args.push_back(a.dump());
                    }
                }
            } else {
                std::cerr << "❌ --conf: script invalide (string ou objet attendu)\n";
                return 1;
            }

            if (script_path.empty()) {
                std::cerr << "❌ --conf: script vide\n";
                return 1;
            }
            if (!fs::exists(script_path)) {
                std::cerr << "❌ Script Lua non trouvé: " << script_path << "\n";
                return 1;
            }

            std::cout << "📜 [conf] Script Lua (" << (si + 1) << "/" << scripts.size() << "): " << script_path << "\n";

            try {
                LuaScripting lua;
                lua.setArgs(script_path, script_args);
                lua.setString("CONF_PATH", conf_abs);
                lua.setString("CONF_DIR", conf_dir);

                if (lua.getState()) {
                    pushJsonToLua(lua.getState(), conf);
                    lua_setglobal(lua.getState(), "CONF");
                }

                if (!lua.loadScript(script_path)) {
                    std::cerr << "❌ Échec exécution Lua: " << script_path << "\n";
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << "❌ Erreur Lua: " << e.what() << "\n";
                return 1;
            }
        }

        std::cout << "\n✅ --conf: scripts exécutés avec succès\n";
        return 0;
    }

    if (!config_path.empty()) {
        std::cout << "⚙️  Chargement de la configuration: " << config_path << "\n";

        if (!fs::exists(config_path)) {
            std::cerr << "❌ Fichier non trouvé: " << config_path << "\n";
            return 1;
        }

        json config;
        {
            std::string err;
            if (!readJsonFile(config_path, config, err)) {
                std::cerr << "❌ " << err << "\n";
                return 1;
            }
        }

        if (!overrides.empty()) {
            std::cout << "🧩 Application des overrides (--override) sur la config:\n";
            for (const auto& o : overrides) {
                std::string err;
                if (!applyOverride(config, o, err)) {
                    std::cerr << "❌ " << err << "\n";
                    return 1;
                }
                std::cout << "  • " << o << "\n";
            }
            std::cout << "\n";
        }

        std::string arch_name;
        json cfg = ModelArchitectures::cfgFromConfig(config, &arch_name);
        std::cout << "🏗️  Architecture: " << arch_name << "\n\n";

        // Construire le modèle selon la config résolue.
        std::shared_ptr<Model> model = ModelArchitectures::create(arch_name, cfg);

        // Brancher les composants framework (tokenizer/encoder) si présents dans la config.
        // (Ils restent aussi accessibles dans model->modelConfig via cfgFromConfig.)
        try {
            if (config.contains("tokenizer") && config["tokenizer"].is_object()) {
                const json& tj = config["tokenizer"];
                const int max_vocab = tj.value("max_vocab", 4096);
                auto tok = std::make_shared<Tokenizer>(static_cast<size_t>(std::max(1, max_vocab)));
                if (tj.contains("max_sequence_length") && tj["max_sequence_length"].is_number_integer()) {
                    tok->setMaxSequenceLength(tj["max_sequence_length"].get<int>());
                }
                model->setTokenizer(*tok);
            }

            if (config.contains("encoder") && config["encoder"].is_object()) {
                const json& ej = config["encoder"];
                const int dim = ej.value("embedding_dim", 64);
                int vocab_size = 4096;
                if (config.contains("tokenizer") && config["tokenizer"].is_object()) {
                    vocab_size = config["tokenizer"].value("max_vocab", vocab_size);
                }
                auto enc = std::make_shared<Encoder>(dim, std::max(1, vocab_size));
                // Optionnel: seed
                uint64_t seed = 0;
                if (config.contains("inference") && config["inference"].is_object()) {
                    seed = static_cast<uint64_t>(config["inference"].value("seed", 0));
                }
                enc->initRandom(seed);
                model->setEncoder(*enc);
            }
        } catch (...) {
            // Best-effort: ne pas bloquer la création du modèle si config partielle.
        }

        model->allocateParams();
        model->initializeWeights("he");

        std::cout << "✅ Modèle créé avec " << model->totalParamCount() << " paramètres\n";
        std::cout << "💡 Entraînement à implémenter selon vos besoins\n";

        return 0;
    }
    
    std::cout << "❌ Option invalide. Utilisez --help\n";
    return 1;

}