#include "RustScripting.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

#include "Models/Registry/ModelArchitectures.hpp"
#include "scriptings/ScriptingBridgeCommon.hpp"

namespace {

static std::string shellQuote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static bool commandExists(const char* cmd) {
    const std::string probe = std::string("command -v ") + cmd + " >/dev/null 2>&1";
    return std::system(probe.c_str()) == 0;
}

static void setEnvVar(const std::string& k, const std::string& v) {
#ifdef _WIN32
    _putenv_s(k.c_str(), v.c_str());
#else
    setenv(k.c_str(), v.c_str(), 1);
#endif
}


static std::string makeBootstrap(const std::string& user_code) {
    std::ostringstream ss;
    ss << "// mimir rust binding bootstrap\n";
    ss << "use std::env;\n";
    ss << "use std::fs::OpenOptions;\n";
    ss << "use std::io::Write;\n";
    ss << "fn __mimir_unimplemented(name: &str) -> ! {\n";
    ss << "    panic!(\"[mimir-rust] API not yet implemented: {}\", name);\n";
    ss << "}\n";
    ss << "fn __bridge_emit(line: &str) {\n";
    ss << "    if let Ok(path) = env::var(\"MIMIR_BRIDGE_CMD_FILE\") {\n";
    ss << "        if !path.is_empty() {\n";
    ss << "            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {\n";
    ss << "                let _ = writeln!(f, \"{}\", line);\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "    }\n";
    ss << "}\n";
    ss << "#[allow(non_snake_case, dead_code)]\n";
    ss << "mod Architectures {\n";
    ss << "    use std::env;\n";
    ss << "    pub fn info() -> String { env::var(\"MIMIR_BRIDGE_ARCH_INFO_JSON\").unwrap_or_else(|_| \"[]\".to_string()) }\n";
    ss << "    pub fn available() -> String { env::var(\"MIMIR_BRIDGE_ARCH_AVAILABLE_JSON\").unwrap_or_else(|_| \"[]\".to_string()) }\n";
    ss << "    /// Nombre d'architectures disponibles (sans parser JSON).\n";
    ss << "    pub fn available_count() -> usize {\n";
    ss << "        env::var(\"MIMIR_BRIDGE_ARCH_AVAIL_COUNT\").ok()\n";
    ss << "            .and_then(|v| v.parse().ok()).unwrap_or(0)\n";
    ss << "    }\n";
    ss << "    #[allow(dead_code)]\n";
    ss << "    pub fn default_config(name: &str) -> String {\n";
    ss << "        let raw = env::var(\"MIMIR_BRIDGE_ARCH_INFO_JSON\").unwrap_or_else(|_| \"[]\".to_string());\n";
    ss << "        let _ = name;\n";
    ss << "        raw\n";
    ss << "    }\n";
    ss << "    pub fn dtypes() -> String { env::var(\"MIMIR_BRIDGE_DTYPES_JSON\").unwrap_or_else(|_| \"[]\".to_string()) }\n";
    ss << "    /// Nombre de dtypes disponibles (sans parser JSON).\n";
    ss << "    pub fn dtypes_count() -> usize {\n";
    ss << "        env::var(\"MIMIR_BRIDGE_DTYPES_COUNT\").ok()\n";
    ss << "            .and_then(|v| v.parse().ok()).unwrap_or(0)\n";
    ss << "    }\n";
    ss << "    pub fn create(name: &str, cfg_json: &str) { super::__bridge_emit(&format!(\"Model.create|{}{}\", name, if !cfg_json.is_empty() { format!(\"|{}\", cfg_json) } else { String::new() })); }\n";
    ss << "}\n";
    ss << "struct Model;\n";
    ss << "impl Model {\n";
    ss << "    fn create(name: &str, cfg_json: &str) {\n";
    ss << "        __bridge_emit(&format!(\"Model.create|{}{}\", name, if !cfg_json.is_empty() { format!(\"|{}\", cfg_json) } else { String::new() }));\n";
    ss << "        CURRENT_MODEL_NAME.with(|c| *c.borrow_mut() = name.to_string());\n";
    ss << "    }\n";
    ss << "    fn allocate_params() { __bridge_emit(\"Model.allocate_params\"); }\n";
    ss << "    fn init_weights(method: &str, seed: u32) { __bridge_emit(&format!(\"Model.init_weights|{}|{}\", method, seed)); }\n";
    ss << "    fn total_params() -> u64 {\n";
    ss << "        __bridge_emit(\"Model.total_params\");\n";
    ss << "        let cache_raw = std::env::var(\"MIMIR_BRIDGE_ARCH_CACHE_JSON\").unwrap_or_default();\n";
    ss << "        let name = CURRENT_MODEL_NAME.with(|c| c.borrow().clone());\n";
    ss << "        let pat = format!(\"\\\"{}\\\"\", name);\n";
    ss << "        if let Some(pos) = cache_raw.find(&pat) {\n";
    ss << "            if let Some(tp_pos) = cache_raw[pos..].find(\"total_params\\\"\") {\n";
    ss << "                let after = &cache_raw[pos + tp_pos + 14..];\n";
    ss << "                let n: String = after.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect();\n";
    ss << "                return n.parse::<u64>().unwrap_or(0);\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "        0\n";
    ss << "    }\n";
    ss << "}\n";
    ss << "use std::cell::RefCell;\n";
    ss << "thread_local! { static CURRENT_MODEL_NAME: RefCell<String> = RefCell::new(String::new()); }\n";
    ss << "fn main() {\n";
    ss << "    let _arg_json = env::var(\"MIMIR_ARG_JSON\").unwrap_or_else(|_| \"[]\".to_string());\n";
    ss << "    let _conf_json = env::var(\"MIMIR_CONF_JSON\").unwrap_or_else(|_| \"{}\".to_string());\n";
    ss << "    let _conf_path = env::var(\"MIMIR_CONF_PATH\").unwrap_or_default();\n";
    ss << "    let _conf_dir = env::var(\"MIMIR_CONF_DIR\").unwrap_or_default();\n";
    ss << "    // NOTE: user code is inlined below and can consume env vars if needed.\n";
    ss << user_code << "\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace

void RustScripting::setArgs(const std::string& script_path,
                            const std::vector<std::string>& script_args) {
    cacheArgs(script_path, script_args);
}

bool RustScripting::loadScript(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        RustContext::getInstance().addLog("[rust] unable to open script: " + filepath);
        return false;
    }

    std::ostringstream ss;
    ss << f.rdbuf();
    loaded_file_ = filepath;
    return executeScript(ss.str());
}

bool RustScripting::executeScript(const std::string& code) {
    auto& ctx = RustContext::getInstance();
    if (!commandExists("rust-script")) {
        ctx.addLog("[rust] rust-script not found in PATH");
        return false;
    }

    const json arg_json = [&]() {
        json a = json::array();
        a.push_back(scriptPath());
        for (const auto& s : scriptArgs()) a.push_back(s);
        return a;
    }();

    setEnvVar("MIMIR_GLOBAL_NAMESPACE", ScriptingContext::kGlobalNamespace);
    setEnvVar("MIMIR_ARG_JSON", arg_json.dump());
    setEnvVar("MIMIR_CONF_JSON", ctx.currentConfig.is_null() ? "{}" : ctx.currentConfig.dump());
    setEnvVar("MIMIR_CONF_PATH", "");
    setEnvVar("MIMIR_CONF_DIR", "");
    setEnvVar("MIMIR_ALIAS_MODEL", ScriptingContext::kAliasModel);
    setEnvVar("MIMIR_ALIAS_ARCHITECTURES", ScriptingContext::kAliasArchitectures);
    setEnvVar("MIMIR_ALIAS_TOKENIZER", ScriptingContext::kAliasTokenizer);
    setEnvVar("MIMIR_ALIAS_DATASET", ScriptingContext::kAliasDataset);
    setEnvVar("MIMIR_ALIAS_MEMORY", ScriptingContext::kAliasMemory);
    setEnvVar("MIMIR_ALIAS_MEMORY_GUARD", ScriptingContext::kAliasMemoryGuard);
    setEnvVar("MIMIR_ALIAS_ALLOCATOR", ScriptingContext::kAliasAllocator);
    setEnvVar("MIMIR_ALIAS_HTOP", ScriptingContext::kAliasHtop);
    setEnvVar("MIMIR_ALIAS_VIZ", ScriptingContext::kAliasViz);
    setEnvVar(ScriptingBridgeCommon::kEnvBridgeArchInfoJson,
              ScriptingBridgeCommon::buildArchitecturesInfoJson());
    setEnvVar(ScriptingBridgeCommon::kEnvBridgeArchAvailJson,
              ScriptingBridgeCommon::buildAvailableJson());
    setEnvVar(ScriptingBridgeCommon::kEnvBridgeDtypesJson,
              ScriptingBridgeCommon::buildDtypesJson());
    setEnvVar(ScriptingBridgeCommon::kEnvBridgeArchCacheJson,
              ScriptingBridgeCommon::loadArchCacheJson());
    {
        const auto dtypes_json = ScriptingBridgeCommon::buildDtypesJson();
        const size_t dc = static_cast<size_t>(std::count(dtypes_json.begin(), dtypes_json.end(), '{'));
        const size_t ac = ModelArchitectures::available().size();
        setEnvVar(ScriptingBridgeCommon::kEnvBridgeDtypesCount,   std::to_string(dc));
        setEnvVar(ScriptingBridgeCommon::kEnvBridgeArchAvailCount, std::to_string(ac));
    }

    const auto tmp = std::filesystem::temp_directory_path() /
                     ("mimir_rs_" + std::to_string(static_cast<unsigned long long>(::getpid())) + ".rs");
    const auto cmd_file = std::filesystem::temp_directory_path() /
                          ("mimir_rs_bridge_" + std::to_string(static_cast<unsigned long long>(::getpid())) + ".cmd");
    setEnvVar(ScriptingBridgeCommon::kEnvBridgeCmdFile, cmd_file.string());
    {
        std::ofstream out(tmp);
        out << makeBootstrap(code);
    }

    std::string cmd = "rust-script " + shellQuote(tmp.string());
    for (const auto& a : scriptArgs()) cmd += " " + shellQuote(a);

    const int rc = std::system(cmd.c_str());
    const bool bridge_ok =
        (rc == 0) ? ScriptingBridgeCommon::processBridgeCommands(ctx, cmd_file, "[rust]") : false;
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    std::filesystem::remove(cmd_file, ec);
    if (rc != 0) {
        ctx.addLog("[rust] script execution failed with status=" + std::to_string(rc));
        return false;
    }
    if (!bridge_ok) {
        ctx.addLog("[rust] bridge command processing failed");
        return false;
    }
    return true;
}

void RustScripting::registerAPI() {
    RustContext::getInstance().addLog(
        "[rust] registerAPI called (system contract unchanged)");
}
