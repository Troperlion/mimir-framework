#include "CSharpScripting.hpp"

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
    ss << "using System;\n";
    ss << "using System.Dynamic;\n";
    ss << "using System.IO;\n";
    ss << "using System.Text.Json;\n";
    ss << "using System.Collections.Generic;\n\n";
    ss << "var __bridgeCmdFile = Environment.GetEnvironmentVariable(\"MIMIR_BRIDGE_CMD_FILE\") ?? \"\";\n";
    ss << "void __bridgeEmit(string line) { if (!string.IsNullOrEmpty(__bridgeCmdFile)) File.AppendAllText(__bridgeCmdFile, line + \"\\n\"); }\n";
    ss << "var __archInfo = JsonSerializer.Deserialize<List<Dictionary<string, object>>>(Environment.GetEnvironmentVariable(\"MIMIR_BRIDGE_ARCH_INFO_JSON\") ?? \"[]\") ?? new List<Dictionary<string, object>>();\n";
    ss << "var __archAvail = JsonSerializer.Deserialize<List<string>>(Environment.GetEnvironmentVariable(\"MIMIR_BRIDGE_ARCH_AVAILABLE_JSON\") ?? \"[]\") ?? new List<string>();\n";
    ss << "var __dtypes = JsonSerializer.Deserialize<List<Dictionary<string, object>>>(Environment.GetEnvironmentVariable(\"MIMIR_BRIDGE_DTYPES_JSON\") ?? \"[]\") ?? new List<Dictionary<string, object>>();\n";
    ss << "var __archCache = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, long>>>(Environment.GetEnvironmentVariable(\"MIMIR_BRIDGE_ARCH_CACHE_JSON\") ?? \"{}\") ?? new Dictionary<string, Dictionary<string, long>>();\n";
    ss << "string __currentModelName = \"\";\n";
    ss << "public sealed class MimirDyn : DynamicObject {\n";
    ss << "  private readonly string _path;\n";
    ss << "  public MimirDyn(string path) { _path = path; }\n";
    ss << "  public override bool TryGetMember(GetMemberBinder b, out object r) { r = new MimirDyn(_path + \".\" + b.Name); return true; }\n";
    ss << "  public override bool TryInvoke(InvokeBinder b, object[] a, out object r) { throw new Exception(\"[mimir-csharp] API not yet implemented: \" + _path); }\n";
    ss << "  public override bool TryInvokeMember(InvokeMemberBinder b, object[] a, out object r) { throw new Exception(\"[mimir-csharp] API not yet implemented: \" + _path + \".\" + b.Name); }\n";
    ss << "}\n\n";
    ss << "var arg = JsonSerializer.Deserialize<List<string>>(Environment.GetEnvironmentVariable(\"MIMIR_ARG_JSON\") ?? \"[]\") ?? new List<string>();\n";
    ss << "var CONF = JsonSerializer.Deserialize<Dictionary<string, object>>(Environment.GetEnvironmentVariable(\"MIMIR_CONF_JSON\") ?? \"{}\") ?? new Dictionary<string, object>();\n";
    ss << "var CONF_PATH = Environment.GetEnvironmentVariable(\"MIMIR_CONF_PATH\") ?? \"\";\n";
    ss << "var CONF_DIR = Environment.GetEnvironmentVariable(\"MIMIR_CONF_DIR\") ?? \"\";\n";
    ss << "dynamic Mimir = new MimirDyn(Environment.GetEnvironmentVariable(\"MIMIR_GLOBAL_NAMESPACE\") ?? \"Mimir\");\n";
    ss << "dynamic model = new MimirDyn(Environment.GetEnvironmentVariable(\"MIMIR_ALIAS_MODEL\") ?? \"model\");\n";
    ss << "dynamic architectures = new MimirDyn(Environment.GetEnvironmentVariable(\"MIMIR_ALIAS_ARCHITECTURES\") ?? \"architectures\");\n";
    ss << "dynamic tokenizer = new MimirDyn(Environment.GetEnvironmentVariable(\"MIMIR_ALIAS_TOKENIZER\") ?? \"tokenizer\");\n";
    ss << "dynamic dataset = new MimirDyn(Environment.GetEnvironmentVariable(\"MIMIR_ALIAS_DATASET\") ?? \"dataset\");\n";
    ss << "dynamic Memory = new MimirDyn(Environment.GetEnvironmentVariable(\"MIMIR_ALIAS_MEMORY\") ?? \"Memory\");\n";
    ss << "dynamic MemoryGuard = new MimirDyn(Environment.GetEnvironmentVariable(\"MIMIR_ALIAS_MEMORY_GUARD\") ?? \"MemoryGuard\");\n";
    ss << "dynamic Allocator = new MimirDyn(Environment.GetEnvironmentVariable(\"MIMIR_ALIAS_ALLOCATOR\") ?? \"Allocator\");\n";
    ss << "dynamic htop = new MimirDyn(Environment.GetEnvironmentVariable(\"MIMIR_ALIAS_HTOP\") ?? \"htop\");\n";
    ss << "dynamic viz = new MimirDyn(Environment.GetEnvironmentVariable(\"MIMIR_ALIAS_VIZ\") ?? \"viz\");\n\n";
    ss << "dynamic Architectures = new System.Dynamic.ExpandoObject();\n";
    ss << "Architectures.available = (Func<object>)(() => __archAvail);\n";
    ss << "Architectures.info = (Func<string, object>)((name) => {\n";
    ss << "  if (!string.IsNullOrEmpty(name)) {\n";
    ss << "    foreach (var e in __archInfo) { if (e.ContainsKey(\"name\") && ((e[\"name\"]?.ToString()) == name)) return e; }\n";
    ss << "    throw new Exception(\"unknown architecture: \" + name);\n";
    ss << "  }\n";
    ss << "  return __archInfo;\n";
    ss << "});\n";
    ss << "Architectures.default_config = (Func<string, object>)((name) => {\n";
    ss << "  foreach (var e in __archInfo) { if (e.ContainsKey(\"name\") && e[\"name\"]?.ToString() == name) return e.ContainsKey(\"config\") ? e[\"config\"] : new Dictionary<string,object>(); }\n";
    ss << "  throw new Exception(\"unknown architecture: \" + name);\n";
    ss << "});\n";
    ss << "Architectures.dtypes = (Func<object>)(() => __dtypes);\n";
    ss << "Architectures.create = (Func<string, string, bool>)((name, cfgJson) => {\n";
    ss << "  __currentModelName = name;\n";
    ss << "  __bridgeEmit(\"Model.create|\" + (name ?? \"\") + (!string.IsNullOrEmpty(cfgJson) ? \"|\" + cfgJson : \"\"));\n";
    ss << "  return true;\n";
    ss << "});\n";
    ss << "architectures = Architectures;\n";
    ss << "dynamic Model = new System.Dynamic.ExpandoObject();\n";
    ss << "Model.create = (Func<string, string, bool>)((name, cfgJson) => {\n";
    ss << "  __currentModelName = name;\n";
    ss << "  __bridgeEmit(\"Model.create|\" + (name ?? \"\") + (!string.IsNullOrEmpty(cfgJson) ? \"|\" + cfgJson : \"\"));\n";
    ss << "  return true;\n";
    ss << "});\n";
    ss << "Model.allocate_params = (Func<bool>)(() => { __bridgeEmit(\"Model.allocate_params\"); return true; });\n";
    ss << "Model.init_weights = (Func<string, int, bool>)((method, seed) => { __bridgeEmit(\"Model.init_weights|\" + (method ?? \"he\") + \"|\" + seed); return true; });\n";
    ss << "Model.total_params = (Func<long>)(() => {\n";
    ss << "  __bridgeEmit(\"Model.total_params\");\n";
    ss << "  return (__archCache.TryGetValue(__currentModelName, out var c) && c.TryGetValue(\"total_params\", out var v)) ? v : 0L;\n";
    ss << "});\n";
    ss << "model = Model;\n\n";
    ss << user_code << "\n";
    return ss.str();
}

}  // namespace

void CSharpScripting::setArgs(const std::string& script_path,
                              const std::vector<std::string>& script_args) {
    cacheArgs(script_path, script_args);
}

bool CSharpScripting::loadScript(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        CSharpContext::getInstance().addLog("[csharp] unable to open script: " + filepath);
        return false;
    }

    std::ostringstream ss;
    ss << f.rdbuf();
    loaded_file_ = filepath;
    return executeScript(ss.str());
}

bool CSharpScripting::executeScript(const std::string& code) {
    auto& ctx = CSharpContext::getInstance();

    const bool has_dotnet_script = commandExists("dotnet-script");
    const bool has_csi = commandExists("csi");
    if (!has_dotnet_script && !has_csi) {
        ctx.addLog("[csharp] dotnet-script/csi not found in PATH");
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
                     ("mimir_cs_" + std::to_string(static_cast<unsigned long long>(::getpid())) + ".csx");
    const auto cmd_file = std::filesystem::temp_directory_path() /
                          ("mimir_cs_bridge_" + std::to_string(static_cast<unsigned long long>(::getpid())) + ".cmd");
    setEnvVar(ScriptingBridgeCommon::kEnvBridgeCmdFile, cmd_file.string());
    {
        std::ofstream out(tmp);
        out << makeBootstrap(code);
    }

    std::string cmd;
    if (has_dotnet_script) {
        cmd = "dotnet-script " + shellQuote(tmp.string());
    } else {
        cmd = "csi " + shellQuote(tmp.string());
    }

    for (const auto& a : scriptArgs()) cmd += " " + shellQuote(a);

    const int rc = std::system(cmd.c_str());
    const bool bridge_ok =
        (rc == 0) ? ScriptingBridgeCommon::processBridgeCommands(ctx, cmd_file, "[csharp]") : false;
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    std::filesystem::remove(cmd_file, ec);
    if (rc != 0) {
        ctx.addLog("[csharp] script execution failed with status=" + std::to_string(rc));
        return false;
    }
    if (!bridge_ok) {
        ctx.addLog("[csharp] bridge command processing failed");
        return false;
    }
    return true;
}

void CSharpScripting::registerAPI() {
    CSharpContext::getInstance().addLog(
        "[csharp] registerAPI called (system contract unchanged)");
}
