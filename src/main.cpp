#include "Model.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include "scriptings/Lua/luaScripting/LuaScripting.hpp"
#ifdef MIMIR_ENABLE_SCRIPTING_RUST
#include "scriptings/Rust/rustScripting/RustScripting.hpp"
#endif
#ifdef MIMIR_ENABLE_SCRIPTING_CSHARP
#include "scriptings/CSharp/csharpScripting/CSharpScripting.hpp"
#endif
#ifdef MIMIR_ENABLE_SCRIPTING_JS
#include "scriptings/JavaScript/jsScripting/JSScripting.hpp"
#endif
#include "AsyncMonitor.hpp"
#include "Helpers.hpp"
#include "HtopDisplay.hpp"
#include "Visualizer.hpp"
#include "MemorySafety.hpp"
#include "Tokenizer.hpp"
#include "Encoder.hpp"
#include "ConfigOverrides.hpp"
#include "runtimes/AbstractRuntime.hpp"
#include "Serialization/Serialization.hpp"
#include "include/json.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <ctime>
#include <cstring>
#include <cctype>
#include <iomanip>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <sstream>
#include <array>
#include <mutex>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;
using namespace ModelArchitectures;

// Spill disque: dossier attendu par le système en cas d'éviction/MemoryGuard.
static std::string g_mimir_spill_dir = ".mimir-spill";

namespace {
std::mutex g_framework_log_mutex;
std::ofstream* g_framework_log_stream = nullptr;
}

void framework_log_write(const char* data, size_t size) {
    std::lock_guard<std::mutex> lock(g_framework_log_mutex);
    if (!g_framework_log_stream || !g_framework_log_stream->is_open() || data == nullptr || size == 0) {
        return;
    }
    g_framework_log_stream->write(data, static_cast<std::streamsize>(size));
    g_framework_log_stream->flush();
}

void framework_log_write_file_only(const char* data, size_t size) {
    framework_log_write(data, size);
}

class FrameworkLogTee {
public:
    FrameworkLogTee() {
        try {
            const fs::path logs_dir = fs::current_path() / "logs";
            std::error_code ec;
            fs::create_directories(logs_dir, ec);

            const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::tm tm{};
            if (auto* local_tm = std::localtime(&now)) {
                tm = *local_tm;
            }

            std::ostringstream name;
            name << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << ".log";
            log_path_ = logs_dir / name.str();

            log_file_.open(log_path_, std::ios::out | std::ios::trunc);
            if (!log_file_.is_open()) {
                return;
            }

            saved_cout_ = std::cout.rdbuf();
            saved_cerr_ = std::cerr.rdbuf();
            const char* verbose_env = std::getenv("MIMIR_CONSOLE_VERBOSE");
            const bool concise_console = !verbose_env || std::string(verbose_env) != "1";
            tee_cout_ = std::make_unique<TeeStreamBuf>(
                saved_cout_, log_file_.rdbuf(), concise_console);
            tee_cerr_ = std::make_unique<TeeStreamBuf>(
                saved_cerr_, log_file_.rdbuf(), concise_console);
            std::cout.rdbuf(tee_cout_.get());
            std::cerr.rdbuf(tee_cerr_.get());
            g_framework_log_stream = &log_file_;
            active_ = true;
        } catch (...) {
            cleanup();
        }
    }

    ~FrameworkLogTee() {
        cleanup();
    }

    const fs::path& logPath() const { return log_path_; }
    bool active() const { return active_; }

private:
    class TeeStreamBuf final : public std::streambuf {
    public:
        TeeStreamBuf(std::streambuf* primary,
                     std::streambuf* secondary,
                     bool concise_console)
            : primary_(primary), secondary_(secondary),
              concise_console_(concise_console) {}

        ~TeeStreamBuf() override {
            flushConsoleLine();
        }

    protected:
        std::streamsize xsputn(const char* s, std::streamsize n) override {
            if (n <= 0) return 0;
            const std::streamsize b = secondary_ ? secondary_->sputn(s, n) : n;
            if (secondary_) secondary_->pubsync();
            if (!primary_) return b;
            if (!concise_console_) {
                const std::streamsize a = primary_->sputn(s, n);
                primary_->pubsync();
                return (a < b) ? a : b;
            }
            for (std::streamsize index = 0; index < n; ++index) {
                console_line_.push_back(s[index]);
                if (s[index] == '\n') flushConsoleLine();
            }
            return b;
        }

        int overflow(int ch) override {
            if (ch == traits_type::eof()) return traits_type::not_eof(ch);
            const char c = static_cast<char>(ch);
            return xsputn(&c, 1) == 1 ? ch : traits_type::eof();
        }

        int sync() override {
            int result = 0;
            if (primary_ && primary_->pubsync() != 0) result = -1;
            if (secondary_ && secondary_->pubsync() != 0) result = -1;
            return result;
        }

    private:
        static bool startsWith(const std::string& line, const char* prefix) {
            return line.rfind(prefix, 0) == 0;
        }

        static bool showOnConsole(const std::string& line) {
            if (startsWith(line, "[startup]")) return false;
            if (startsWith(line, "[memory]")) return false;
            if (startsWith(line, "[planner]")) return false;
            if (startsWith(line, "[runtime-trace]")) return false;
            if (startsWith(line, "[runtime]")) return false;
            if (startsWith(line, "[allocator]")) return false;
            if (startsWith(line, "[registry]")) return false;
            if (startsWith(line, "[encoder]")) return false;
            if (startsWith(line, "[serialization] raw ")) return false;
            if (startsWith(line, "[serialization] safetensors ")) return false;
            if (startsWith(line, "  Layer ")) return false;
            if (startsWith(line, "Test ")) return false;
            if (startsWith(line, "  • ")) return false;
            if (startsWith(line, "🛡️  Vérification")) return false;
            if (startsWith(line, "✅ Structure legacy")) return false;
            if (startsWith(line, "🧪 TEST D'INTÉGRITÉ")) return false;
            if (startsWith(line, "✅ TOUS LES TESTS PASSÉS")) return false;
            if (startsWith(line, "🔧 OpenMP:")) return false;
            if (startsWith(line, "🚀 Optimisations hardware:")) return false;
            if (startsWith(line, "═══════════════════════════")) return false;
            if (line.find("blocs de poids") != std::string::npos) return false;
            if (line.find("scratchpad tag=") != std::string::npos) return false;
            return true;
        }

        void flushConsoleLine() {
            if (console_line_.empty()) return;
            if (showOnConsole(console_line_)) {
                const bool blank = console_line_.find_first_not_of(" \t\r\n") == std::string::npos;
                // Les blocs filtrés laissent souvent plusieurs séparateurs
                // vides consécutifs. En conserver un seul garde le terminal
                // lisible sans compacter les vraies sections utiles.
                if (!blank || !previous_console_line_blank_) {
                    primary_->sputn(console_line_.data(),
                                    static_cast<std::streamsize>(console_line_.size()));
                    primary_->pubsync();
                }
                previous_console_line_blank_ = blank;
            }
            console_line_.clear();
        }

        std::streambuf* primary_ = nullptr;
        std::streambuf* secondary_ = nullptr;
        bool concise_console_ = true;
        std::string console_line_;
        bool previous_console_line_blank_ = false;
    }
    ;

    void restoreStandardStreams() {
        if (saved_cout_) {
            std::cout.rdbuf(saved_cout_);
        }
        if (saved_cerr_) {
            std::cerr.rdbuf(saved_cerr_);
        }
        g_framework_log_stream = nullptr;
    }

    void cleanup() {
        if (!active_) {
            restoreStandardStreams();
            if (log_file_.is_open()) log_file_.close();
            return;
        }

        std::cout.flush();
        std::cerr.flush();
        restoreStandardStreams();
        if (log_file_.is_open()) {
            log_file_.flush();
            log_file_.close();
        }
        active_ = false;
    }

    std::filesystem::path log_path_;
    std::ofstream log_file_;
    std::streambuf* saved_cout_ = nullptr;
    std::streambuf* saved_cerr_ = nullptr;
    std::unique_ptr<TeeStreamBuf> tee_cout_;
    std::unique_ptr<TeeStreamBuf> tee_cerr_;
    bool active_ = false;
};

class FrameworkExitSummary {
public:
    FrameworkExitSummary()
        : start_(std::chrono::steady_clock::now()) {}

    ~FrameworkExitSummary() {
        const auto end = std::chrono::steady_clock::now();
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        const auto elapsed_sec = elapsed_ms / 1000;
        const auto elapsed_rem_ms = elapsed_ms % 1000;

        auto& guard = MemoryGuard::instance();
        std::cerr << "\n[exit] execution_time=" << elapsed_sec << 's'
                  << ' ' << elapsed_rem_ms << "ms"
                  << " memory_current=" << (guard.getCurrentBytes() / 1024 / 1024) << "MB"
                  << " memory_peak=" << (guard.getPeakBytes() / 1024 / 1024) << "MB"
                  << " memory_usage=" << std::fixed << std::setprecision(1)
                  << guard.getUsagePercent() << "%" << std::endl;
    }

private:
    std::chrono::steady_clock::time_point start_;
};

static std::string cudaAccelStatus() {
#ifndef ENABLE_CUDA
    return "✗";
#else
    const RuntimeConfig cfg = RuntimeConfig::fromEnv("CUDA");
    if (cfg.disabled) {
        return "✗ (désactivé)";
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return "✗ (aucun device)";
    }

    int device_index = cfg.device_index;
    if (device_index < 0 || device_index >= device_count) {
        device_index = 0;
    }

    cudaDeviceProp prop;
    const bool have_props = (cudaGetDeviceProperties(&prop, device_index) == cudaSuccess);

    // Ici, on reflète l'intention d'usage: le backend existe, mais n'est "utilisé" que si opt-in.
    if (!cfg.linear_enabled) {
        return have_props ? ("— (device " + std::to_string(device_index) + ": " + prop.name + ")") : "— (device)";
    }

    return have_props ? ("✓ (device " + std::to_string(device_index) + ": " + prop.name + ")") : "✓";
#endif
}

static std::string rocmAccelStatus() {
#ifndef ENABLE_ROCM
    return "✗";
#else
    const RuntimeConfig cfg = RuntimeConfig::fromEnv("ROCM");
    if (cfg.disabled) {
        return "✗ (désactivé)";
    }

    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count <= 0) {
        return "✗ (aucun device)";
    }

    int device_index = cfg.device_index;
    if (device_index < 0 || device_index >= device_count) {
        device_index = 0;
    }

    hipDeviceProp_t prop;
    const bool have_props = (hipGetDeviceProperties(&prop, device_index) == hipSuccess);

    if (!cfg.linear_enabled) {
        return have_props ? ("— (device " + std::to_string(device_index) + ": " + prop.name + ")") : "— (device)";
    }

    return have_props ? ("✓ (device " + std::to_string(device_index) + ": " + prop.name + ")") : "✓";
#endif
}

static void clearSpillDirContentsNoThrow() noexcept {
    try {
        const fs::path dir(g_mimir_spill_dir);

        // Garde-fou: ne nettoyer QUE le dossier attendu (évite toute suppression accidentelle).
        if (dir.empty() || dir.filename() != ".mimir-spill") {
            return;
        }

        std::error_code ec;
        if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) {
            return;
        }

        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            std::error_code rm_ec;
            fs::remove_all(entry.path(), rm_ec);
        }
    } catch (...) {
    }
}

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

static bool isValidEnvironmentName(const std::string& name)
{
    if (name.empty()) return false;
    const auto is_alpha_or_underscore = [](unsigned char c) {
        return std::isalpha(c) != 0 || c == '_';
    };
    const auto is_alnum_or_underscore = [](unsigned char c) {
        return std::isalnum(c) != 0 || c == '_';
    };

    if (!is_alpha_or_underscore(static_cast<unsigned char>(name.front()))) return false;
    return std::all_of(name.begin() + 1, name.end(), [&](char c) {
        return is_alnum_or_underscore(static_cast<unsigned char>(c));
    });
}

static bool applyConfigEnvironment(const json& conf, std::string& err)
{
    if (!conf.contains("env")) return true;
    if (!conf["env"].is_object()) {
        err = "La section 'env' doit être un objet JSON";
        return false;
    }

    for (const auto& [name, value] : conf["env"].items()) {
        if (!isValidEnvironmentName(name)) {
            err = "Nom de variable d'environnement invalide dans 'env': " + name;
            return false;
        }

        std::string text;
        if (value.is_string()) {
            text = value.get<std::string>();
        } else if (value.is_boolean()) {
            text = value.get<bool>() ? "true" : "false";
        } else if (value.is_number()) {
            text = value.dump();
        } else {
            err = "Valeur invalide pour env." + name + " (chaîne, nombre ou booléen attendu)";
            return false;
        }

        int env_error = 0;
#ifdef _WIN32
        env_error = _putenv_s(name.c_str(), text.c_str());
#else
        if (setenv(name.c_str(), text.c_str(), 1) != 0) env_error = errno;
#endif
        if (env_error != 0) {
            err = "Impossible d'appliquer env." + name + ": " + std::strerror(env_error);
            return false;
        }

#ifdef _OPENMP
        if (name == "OMP_NUM_THREADS") {
            try {
                std::size_t parsed = 0;
                const long threads = std::stol(text, &parsed);
                if (parsed != text.size() || threads < 1 || threads > INT_MAX) {
                    throw std::out_of_range("OMP_NUM_THREADS");
                }
                omp_set_num_threads(static_cast<int>(threads));
            } catch (const std::exception&) {
                err = "Valeur invalide pour env.OMP_NUM_THREADS (entier positif attendu)";
                return false;
            }
        }
#endif
    }
    return true;
}

void printUsage(const char *prog)
{
    std::cout << "Usage: " << prog << " [OPTIONS]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --version, -v           Afficher la version et quitter\n";
    std::cout << "  --lua <script.lua>       Exécuter un script Lua\n";
#ifdef MIMIR_ENABLE_SCRIPTING_JS
    std::cout << "  --js <script.js>         Exécuter un script JavaScript (Node.js)\n";
#endif
#ifdef MIMIR_ENABLE_SCRIPTING_CSHARP
    std::cout << "  --csharp <script.csx>    Exécuter un script C# (dotnet-script/csi)\n";
#endif
#ifdef MIMIR_ENABLE_SCRIPTING_RUST
    std::cout << "  --rust <script.rs>       Exécuter un script Rust (rust-script)\n";
#endif
    std::cout << "  --config <config.json>   Charger et entraîner depuis config\n";
    std::cout << "  --conf <config.json>     Charger une conf et exécuter lua.scripts\n";
    std::cout << "  --run <task>             Exécuter une tâche nommée définie dans tasks.<task> (avec --conf)\n";
    std::cout << "  --override <path=value>  Override (répétable) appliqué à la config\n";
    std::cout << "  --help                   Afficher cette aide\n";
    std::cout << "\nTâches (avec --conf + --run):\n";
    std::cout << "  La section 'tasks' du fichier de conf définit des tâches nommées.\n";
    std::cout << "  Chaque tâche contient un bloc 'lua' identique au bloc racine.\n";
    std::cout << "  Sans --run, la section 'lua' racine est utilisée (comportement par défaut).\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << prog << " --lua scripts/test_lua_api.lua\n";
#ifdef MIMIR_ENABLE_SCRIPTING_JS
    std::cout << "  " << prog << " --js scripts/examples/example.js\n";
#endif
#ifdef MIMIR_ENABLE_SCRIPTING_CSHARP
    std::cout << "  " << prog << " --csharp scripts/examples/example.csx\n";
#endif
#ifdef MIMIR_ENABLE_SCRIPTING_RUST
    std::cout << "  " << prog << " --rust scripts/examples/example.rs\n";
#endif
    std::cout << "  " << prog << " --config config.json\n";
    std::cout << "  " << prog << " --conf config.json\n";
    std::cout << "  " << prog << " --conf config.json --run train\n";
    std::cout << "  " << prog << " --conf config.json --run infer\n";
    std::cout << "  " << prog << " --config config.json --override max_vocab=64000\n";
    std::cout << "  " << prog << " --config config.json --override optimizer=\"adamw\" --override weight_decay=0.01\n";
}

int main(int argc, char **argv)
{
    FrameworkLogTee framework_log;
    FrameworkExitSummary exit_summary;

    for (int i = 1; i < argc; ++i) {
        const std::string opt = argv[i];
        if (opt == "--version" || opt == "-v") {
            std::cout << Mimir::Serialization::get_mimir_version() << "\n";
            return 0;
        }
    }

    {
        const std::string ver = Mimir::Serialization::get_mimir_version();
        // Largeur interne de la boîte = 40 chars affichage.
        // Préfixe "       Mímir Framework v" = 24 chars affichage (ASCII sauf "í" qui occupe 2 octets
        // mais 1 char affichage — on calcule le padding sur la taille d'affichage).
        const int trailing = std::max(0, 40 - 24 - static_cast<int>(ver.size()));
        std::cerr << "╔════════════════════════════════════════╗\n";
        std::cerr << "║       Mímir Framework v" << ver << std::string(trailing, ' ') << "║\n";
        std::cerr << "║     Deep Learning Architectures        ║\n";
        std::cerr << "╚════════════════════════════════════════╝\n\n";
    }

    if (framework_log.active()) {
        std::cerr << "📝 Journal du framework: " << framework_log.logPath().string() << "\n\n";
    }

    {
        std::vector<std::string> mpk_warnings;
        const std::size_t mpk_loaded =
            LuaScripting::autoRegisterMpkArchitectures(fs::current_path().string(), &mpk_warnings);
        if (mpk_loaded > 0) {
            std::cerr << "[startup] mpk_architectures_loaded=" << mpk_loaded
                      << " from=" << (fs::current_path() / "_archi").string() << std::endl;
        }
        for (const auto& warning : mpk_warnings) {
            std::cerr << "[startup] MPK ignoré: " << warning << std::endl;
        }

        const auto registry_count = ModelArchitectures::Registry::instance().list().size();
        auto& guard = MemoryGuard::instance();
        std::cerr << "[startup] workspace=" << fs::current_path().string() << std::endl;
        std::cerr << "[startup] registry_architectures=" << registry_count << std::endl;
        std::cerr << "[memory] current=" << (guard.getCurrentBytes() / 1024 / 1024)
                  << "MB limit=" << (guard.getLimit() / 1024 / 1024)
                  << "MB usage=" << guard.getUsagePercent() << "%" << std::endl;
    }

    // Préparer le spill dir + nettoyage fin de run.
    {
        std::error_code ec;
        fs::create_directories(g_mimir_spill_dir, ec);
        // Nettoyer aussi au démarrage (résidus d'un run précédent).
        clearSpillDirContentsNoThrow();
        std::atexit(clearSpillDirContentsNoThrow);
    }
    
    // 🛡️ SÉCURITÉ MÉMOIRE: Vérification au démarrage
    std::cerr << "🛡️  Vérification de la sécurité mémoire...\n";
    MemorySafety::validateLegacyDisabled();
    MemorySafety::runMemoryIntegrityTest();
    std::cerr << "\n";
    
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    std::cerr << "🔧 OpenMP: " << num_threads << " threads disponibles\n";
    omp_set_num_threads(num_threads);
#endif
    
    std::cerr << "🚀 Optimisations hardware:\n";
    std::cerr << "  • AVX2: " << (Model::hasAVX2() ? "✓" : "✗") << "\n";
    std::cerr << "  • FMA: " << (Model::hasFMA() ? "✓" : "✗") << "\n";
    std::cerr << "  • F16C: " << (Model::hasF16C() ? "✓" : "✗") << "\n";
    std::cerr << "  • BMI2: " << (Model::hasBMI2() ? "✓" : "✗") << "\n";
    std::cerr << "  • CUDA: " << cudaAccelStatus() << "\n";
    std::cerr << "  • ROCM: " << rocmAccelStatus() << "\n";
    std::cerr << "\n";
    
    if (argc < 2) {
        std::cerr << "💡 Utilisez --help pour voir les options\n";
        std::cerr << "💡 Ou:      " << argv[0] << " --lua scripts/test_lua_api.lua\n\n";
        return 0;
    }

    auto waitForLuaVizCloseIfOpen = []() {
        auto& ctx = LuaContext::getInstance();
        if (ctx.asyncMonitor && ctx.asyncMonitor->getViz() && ctx.asyncMonitor->getViz()->isOpen()) {
            std::cerr << "\n🖼️  Viz ouverte — fermeture manuelle requise pour quitter le script...\n";
            ctx.asyncMonitor->waitForVizClose();
        }
    };

    // Modes scripting directs: on passe tous les args après le script au runtime.
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        const std::string opt = argv[i];
#ifndef MIMIR_ENABLE_SCRIPTING_JS
        if (opt == "--js") {
            std::cerr << "❌ Le bridge JavaScript n'est pas compilé dans ce binaire (ENABLE_SCRIPTING_JS=OFF)\n";
            return 1;
        }
#endif
#ifndef MIMIR_ENABLE_SCRIPTING_CSHARP
        if (opt == "--csharp") {
            std::cerr << "❌ Le bridge C# n'est pas compilé dans ce binaire (ENABLE_SCRIPTING_CSHARP=OFF)\n";
            return 1;
        }
#endif
#ifndef MIMIR_ENABLE_SCRIPTING_RUST
        if (opt == "--rust") {
            std::cerr << "❌ Le bridge Rust n'est pas compilé dans ce binaire (ENABLE_SCRIPTING_RUST=OFF)\n";
            return 1;
        }
#endif

        bool is_script_mode = (opt == "--lua");
    #ifdef MIMIR_ENABLE_SCRIPTING_JS
        is_script_mode = is_script_mode || (opt == "--js");
    #endif
    #ifdef MIMIR_ENABLE_SCRIPTING_CSHARP
        is_script_mode = is_script_mode || (opt == "--csharp");
    #endif
    #ifdef MIMIR_ENABLE_SCRIPTING_RUST
        is_script_mode = is_script_mode || (opt == "--rust");
    #endif

        if (is_script_mode && i + 1 < argc) {
            const std::string script_path = argv[++i];
            const std::string lang =
                (opt == "--lua") ? "Lua" :
                (opt == "--js") ? "JavaScript" :
                (opt == "--csharp") ? "C#" : "Rust";

            std::cerr << "📜 Exécution du script " << lang << ": " << script_path << "\n";
            std::cerr << "═══════════════════════════════════════════════\n\n";
            
            if (!fs::exists(script_path)) {
                std::cerr << "❌ Fichier non trouvé: " << script_path << "\n";
                return 1;
            }
            
            try {
                std::vector<std::string> script_args;
                for (int j = i + 1; j < argc; ++j) {
                    script_args.emplace_back(argv[j]);
                }

                if (opt == "--lua") {
                    LuaScripting lua;
                    lua.setArgs(script_path, script_args);
                    if (!lua.loadScript(script_path)) {
                        std::cerr << "❌ Échec exécution Lua: " << script_path << "\n";
                        return 1;
                    }

                    // UX: quand la Viz est active, ne pas fermer tant que l'utilisateur
                    // n'a pas explicitement fermé la fenêtre.
                    waitForLuaVizCloseIfOpen();

                    {
                        auto& ctx = LuaContext::getInstance();
                        ctx.resetRuntimeState();
                    }
#ifdef MIMIR_ENABLE_SCRIPTING_JS
                } else if (opt == "--js") {
                    JSScripting js;
                    js.registerAPI();
                    auto js_args = script_args;
                    if (!js_args.empty() && js_args.front() == "--") js_args.erase(js_args.begin());
                    js.setArgs(script_path, js_args);
                    if (!js.loadScript(script_path)) {
                        std::cerr << "❌ Échec exécution JS: " << script_path << "\n";
                        return 1;
                    }
                    JSContext::getInstance().resetRuntimeState();
#endif
#ifdef MIMIR_ENABLE_SCRIPTING_CSHARP
                } else if (opt == "--csharp") {
                    CSharpScripting cs;
                    cs.registerAPI();
                    cs.setArgs(script_path, script_args);
                    if (!cs.loadScript(script_path)) {
                        std::cerr << "❌ Échec exécution C#: " << script_path << "\n";
                        return 1;
                    }
                    CSharpContext::getInstance().resetRuntimeState();
#endif
#ifdef MIMIR_ENABLE_SCRIPTING_RUST
                } else if (opt == "--rust") {
                    RustScripting rs;
                    rs.registerAPI();
                    rs.setArgs(script_path, script_args);
                    if (!rs.loadScript(script_path)) {
                        std::cerr << "❌ Échec exécution Rust: " << script_path << "\n";
                        return 1;
                    }
                    RustContext::getInstance().resetRuntimeState();
#endif
                }

                std::cerr << "\n✅ Script " << lang << " exécuté avec succès\n";
            } catch (const std::exception& e) {
                std::cerr << "❌ Erreur script " << lang << ": " << e.what() << "\n";
                return 1;
            }
            
            return 0;
        }
    }

    // Mode --config
    std::string config_path;
    std::string conf_path;
    std::string run_task;          // tâche nommée sélectionnée via --run
    std::vector<std::string> overrides;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (a == "--conf" && i + 1 < argc) {
            conf_path = argv[++i];
        } else if (a == "--run" && i + 1 < argc) {
            run_task = argv[++i];
        } else if (a == "--override" && i + 1 < argc) {
            overrides.emplace_back(argv[++i]);
        } else if (a == "--help") {
            // déjà géré plus haut
        } else if (a == "--lua"
    #ifdef MIMIR_ENABLE_SCRIPTING_JS
               || a == "--js"
    #endif
    #ifdef MIMIR_ENABLE_SCRIPTING_CSHARP
               || a == "--csharp"
    #endif
    #ifdef MIMIR_ENABLE_SCRIPTING_RUST
               || a == "--rust"
    #endif
              ) {
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
                // --run sans valeur est traité plus haut; ici on rejette les vrais inconnus.
                std::cerr << "❌ Option inconnue en mode --conf: " << a << "\n";
                std::cerr << "💡 Utilisez --help pour la liste des options\n";
                return 1;
            }
        }
    }

    if (!conf_path.empty()) {
        std::cerr << "⚙️  Chargement de la conf: " << conf_path << "\n";

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
            std::cerr << "🧩 Application des overrides (--override) sur la conf:\n";
            for (const auto& o : overrides) {
                std::string err;
                if (!applyOverride(conf, o, err)) {
                    std::cerr << "❌ " << err << "\n";
                    return 1;
                }
                std::cerr << "  • " << o << "\n";
            }
            std::cerr << "\n";
        }

        {
            std::string err;
            if (!applyConfigEnvironment(conf, err)) {
                std::cerr << "❌ " << err << "\n";
                return 1;
            }
            if (conf.contains("env")) {
                std::cerr << "🌐 Environnement appliqué depuis la conf ("
                          << conf["env"].size() << " variable(s))\n";
            }
        }

        // ── Résolution de la tâche active ───────────────────────────────────────
        // Sans --run : utilise la section lua racine du fichier (comportement défaut).
        // Avec --run <name> : cherche conf["tasks"][name], qui doit contenir un bloc
        //   lua identique au bloc racine (lua.scripts[] ou lua.script).
        const json* task_conf = &conf;
        if (!run_task.empty()) {
            if (!conf.contains("tasks") || !conf["tasks"].is_object()) {
                std::cerr << "❌ --run '" << run_task << "': aucune section 'tasks' dans '" << conf_path << "'\n";
                std::cerr << "💡 Ajoutez une section \"tasks\": { \"" << run_task << "\": { \"lua\": { \"scripts\": [...] } } }\n";
                return 1;
            }
            const json& tasks_node = conf["tasks"];
            if (!tasks_node.contains(run_task) || !tasks_node[run_task].is_object()) {
                std::cerr << "❌ --run: tâche '" << run_task << "' introuvable\n";
                std::cerr << "💡 Tâches disponibles dans '" << conf_path << "':";
                bool first = true;
                for (auto& [k, v] : tasks_node.items()) {
                    std::cerr << (first ? " " : ", ") << k;
                    if (v.is_object() && v.contains("description") && v["description"].is_string())
                        std::cerr << " (" << v["description"].get<std::string>() << ")";
                    first = false;
                }
                std::cerr << "\n";
                return 1;
            }
            task_conf = &tasks_node[run_task];
            const std::string task_desc = task_conf->value("description", std::string{});
            std::cerr << "▶️  Tâche: " << run_task;
            if (!task_desc.empty()) std::cerr << " — " << task_desc;
            std::cerr << "\n";
        }

#ifdef MIMIR_ENABLE_SCRIPTING_JS
        // JS config-driven entry uses the same globals and script argument contract.
        const json* js_conf = nullptr;
        if (task_conf->contains("js") && (*task_conf)["js"].is_object())
            js_conf = &(*task_conf)["js"];
        else if (task_conf->contains("run") && (*task_conf)["run"].is_object()
                 && (*task_conf)["run"].contains("js") && (*task_conf)["run"]["js"].is_object())
            js_conf = &(*task_conf)["run"]["js"];
        if (js_conf) {
            std::vector<json> entries;
            if (js_conf->contains("scripts") && (*js_conf)["scripts"].is_array())
                for (const auto& entry : (*js_conf)["scripts"]) entries.push_back(entry);
            else if (js_conf->contains("script")) entries.push_back((*js_conf)["script"]);
            if (entries.empty()) {
                std::cerr << "--conf: expected js.script or js.scripts\n";
                return 1;
            }
            try {
                JSScripting js;
                for (const auto& entry : entries) {
                    const std::string path = entry.is_string() ? entry.get<std::string>() : entry.value("script", "");
                    std::vector<std::string> args;
                    if (entry.is_object() && entry.contains("args")) {
                        if (!entry.at("args").is_array())
                            throw std::runtime_error("js script args must be an array");
                        for (const auto& arg : entry.at("args"))
                            args.push_back(arg.is_string() ? arg.get<std::string>() : arg.dump());
                    }
                    js.setArgs(path, args);
                    js.setSystemConfig(conf, fs::absolute(conf_path).string(),
                                       fs::absolute(fs::path(conf_path)).parent_path().string());
                    if (path.empty()) throw std::runtime_error("js script path is empty");
                    if (!js.loadScript(path)) return 1;
                }
                JSContext::getInstance().resetRuntimeState();
                return 0;
            } catch (const std::exception& e) {
                std::cerr << "--conf JS: " << e.what() << "\n";
                return 1;
            }
        }
#endif

        // ── Résolution du bloc lua dans la tâche active ─────────────────────────
        const json* lua_conf = nullptr;
        if (task_conf->contains("lua") && (*task_conf)["lua"].is_object()) {
            lua_conf = &(*task_conf)["lua"];
        } else if (task_conf->contains("run") && (*task_conf)["run"].is_object()
                   && (*task_conf)["run"].contains("lua") && (*task_conf)["run"]["lua"].is_object()) {
            lua_conf = &(*task_conf)["run"]["lua"];
        }

        if (!lua_conf) {
            if (run_task.empty())
                std::cerr << "❌ --conf: aucune section lua trouvée (attendu: lua.scripts ou run.lua.scripts)\n";
            else
                std::cerr << "❌ --conf --run '" << run_task << "': aucune section lua dans la tâche\n";
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

            std::cerr << "📜 [conf] Script Lua (" << (si + 1) << "/" << scripts.size() << "): " << script_path << "\n";

            try {
                LuaScripting lua;
                lua.setArgs(script_path, script_args);
                lua.setSystemConfig(conf, conf_abs, conf_dir);

                if (!lua.loadScript(script_path)) {
                    std::cerr << "❌ Échec exécution Lua: " << script_path << "\n";
                    return 1;
                }

                // Même comportement qu'en --lua direct: si la Viz est ouverte,
                // bloquer jusqu'à fermeture explicite par l'utilisateur.
                waitForLuaVizCloseIfOpen();
            } catch (const std::exception& e) {
                std::cerr << "❌ Erreur Lua: " << e.what() << "\n";
                return 1;
            }
        }

        // IMPORTANT: libérer explicitement les ressources LuaContext avant exit.
        {
            auto& ctx = LuaContext::getInstance();
            ctx.resetRuntimeState();
        }

        std::cerr << "\n✅ --conf: scripts exécutés avec succès\n";
        return 0;
    }

    if (!config_path.empty()) {
        std::cerr << "⚙️  Chargement de la configuration: " << config_path << "\n";

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
            std::cerr << "🧩 Application des overrides (--override) sur la config:\n";
            for (const auto& o : overrides) {
                std::string err;
                if (!applyOverride(config, o, err)) {
                    std::cerr << "❌ " << err << "\n";
                    return 1;
                }
                std::cerr << "  • " << o << "\n";
            }
            std::cerr << "\n";
        }

        {
            std::string err;
            if (!applyConfigEnvironment(config, err)) {
                std::cerr << "❌ " << err << "\n";
                return 1;
            }
            if (config.contains("env")) {
                std::cerr << "🌐 Environnement appliqué depuis la configuration ("
                          << config["env"].size() << " variable(s))\n";
            }
        }

        std::string arch_name;
        json cfg = ModelArchitectures::cfgFromConfig(config, &arch_name);
        std::cerr << "🏗️  Architecture: " << arch_name << "\n\n";

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
                auto enc = std::make_shared<ConditioningEncoder>(dim, std::max(1, vocab_size));
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

        std::cerr << "✅ Modèle créé avec " << model->totalParamCount() << " paramètres\n";
        std::cerr << "💡 Entraînement à implémenter selon vos besoins\n";

        return 0;
    }
    
    std::cout << "❌ Option invalide. Utilisez --help\n";
    return 1;

}
