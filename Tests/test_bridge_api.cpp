// Tests/test_bridge_api.cpp
// Teste la parite du bridge commun (ScriptingBridgeCommon) pour
// les runtimes JS, C# et Rust. Chaque sous-test:
//   1. Cree un script minimal inline.
//   2. L'execute via le runtime cible.
//   3. Verifie le contexte C++ (modele cree, params alloues).
//
#define MIMIR_ASSERT(cond, msg) TASSERT_TRUE((cond))
#define MIMIR_FAIL(msg)  do { std::cerr << "FAIL: " << (msg) << "\n"; return false; } while(0)
#include "test_utils.hpp"

#include "include/json.hpp"
#ifdef MIMIR_ENABLE_SCRIPTING_JS
#include "scriptings/JavaScript/jsScripting/JSScripting.hpp"
#include "scriptings/Lua/luaScripting/LuaScripting.hpp"
#endif
#ifdef MIMIR_ENABLE_SCRIPTING_CSHARP
#include "scriptings/CSharp/csharpScripting/CSharpScripting.hpp"
#endif
#ifdef MIMIR_ENABLE_SCRIPTING_RUST
#include "scriptings/Rust/rustScripting/RustScripting.hpp"
#endif
#include "scriptings/ScriptingBridgeCommon.hpp"
#include "Models/Registry/ModelArchitectures.hpp"

using json = nlohmann::json;

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

[[maybe_unused]] static bool commandAvailable(const char* cmd) {
    return std::system((std::string("command -v ") + cmd + " >/dev/null 2>&1").c_str()) == 0;
}

[[maybe_unused]] static bool dotnetRuntimeAvailable() {
    if (const char* root = std::getenv("DOTNET_ROOT")) {
        const fs::path fxr = fs::path(root) / "host" / "fxr";
        if (fs::exists(fxr)) return true;
    }
    if (const char* home = std::getenv("HOME")) {
        const fs::path fxr = fs::path(home) / ".dotnet" / "host" / "fxr";
        if (fs::exists(fxr)) return true;
    }
    return false;
}

static int currentPid() {
#if defined(_WIN32)
    return static_cast<int>(_getpid());
#else
    return static_cast<int>(::getpid());
#endif
}

[[maybe_unused]] static fs::path writeTempScript(const std::string& ext, const std::string& content) {
    const auto tmp = fs::temp_directory_path() /
                     ("mimir_test_bridge" + std::to_string(currentPid()) + "." + ext);
    std::ofstream f(tmp);
    f << content;
    return tmp;
}

// ---------------------------------------------------------------------------
// Assertions communes apres execution bridge
// ---------------------------------------------------------------------------

[[maybe_unused]] static bool assertModelCreated(const ScriptingContext& ctx, const std::string& label) {
    if (!ctx.currentModel) {
        MIMIR_FAIL(label + ": currentModel == nullptr apres bridge");
        return false;
    }
    if (ctx.modelType != "basic_mlp") {
        MIMIR_FAIL(label + ": modelType != basic_mlp, got=" + ctx.modelType);
        return false;
    }
    return true;
}

[[maybe_unused]] static bool assertParamsAllocated(const ScriptingContext& ctx, const std::string& label) {
    if (!ctx.currentModel) { 
        MIMIR_FAIL(label + ": no model"); 
        return false; 
    }
    const size_t n = ctx.currentModel->totalParamCount();
    if (n == 0) {
        MIMIR_FAIL(label + ": totalParamCount == 0 apres allocate_params");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// ScriptingBridgeCommon: test des builders de donnees
// ---------------------------------------------------------------------------

static int testBridgeCommonData() {
    const char* previous_rootwork = std::getenv(ScriptingBridgeCommon::kEnvRootWork);
    const std::string saved_rootwork = previous_rootwork ? previous_rootwork : "";
    const fs::path configured_root = fs::temp_directory_path() / "mimir bridge root";
#if defined(_WIN32)
    _putenv_s(ScriptingBridgeCommon::kEnvRootWork, configured_root.string().c_str());
#else
    setenv(ScriptingBridgeCommon::kEnvRootWork, configured_root.string().c_str(), 1);
#endif
    const std::string rootwork = ScriptingBridgeCommon::ensureRootWorkEnv();
    MIMIR_ASSERT(fs::path(rootwork) == fs::absolute(configured_root).lexically_normal(),
                 "ROOTWORK explicite non conserve");
#if defined(_WIN32)
    _putenv_s(ScriptingBridgeCommon::kEnvRootWork, saved_rootwork.c_str());
#else
    if (previous_rootwork) setenv(ScriptingBridgeCommon::kEnvRootWork, saved_rootwork.c_str(), 1);
    else unsetenv(ScriptingBridgeCommon::kEnvRootWork);
#endif

    const std::string avail = ScriptingBridgeCommon::buildAvailableJson();
    MIMIR_ASSERT(!avail.empty(), "buildAvailableJson vide");
    MIMIR_ASSERT(avail.find("basic_mlp") != std::string::npos,
                 "basic_mlp absent de buildAvailableJson");

    const std::string info = ScriptingBridgeCommon::buildArchitecturesInfoJson();
    MIMIR_ASSERT(info.find("description") != std::string::npos,
                 "buildArchitecturesInfoJson sans 'description'");

    // Normalisation: aliases doit etre un array ([ au lieu d'une string "...)
    const std::string dtypes = ScriptingBridgeCommon::buildDtypesJson();
    MIMIR_ASSERT(dtypes.find("float32") != std::string::npos,
                 "buildDtypesJson sans 'float32'");
    MIMIR_ASSERT(dtypes.find("\"aliases\":[") != std::string::npos,
                 "aliases n'est pas un array JSON (normalisation manquante)");
    MIMIR_ASSERT(dtypes.find("\"bytes\":4") != std::string::npos,
                 "bytes=4 absent pour float32");
    MIMIR_ASSERT(dtypes.find("\"kind\":\"float\"") != std::string::npos,
                 "kind=float absent");

    // Verifier la structure complete du premier dtype via parse JSON.
    const json dts = json::parse(dtypes);
    MIMIR_ASSERT(dts.is_array() && dts.size() == 13, "dtypes: 13 entrees attendues");
    const json& f32 = dts[0];  // float32 est le premier.
    MIMIR_ASSERT(f32["name"].get<std::string>() == "float32", "f32 name != float32");
    MIMIR_ASSERT(f32["aliases"].is_array(),                   "f32 aliases n'est pas un array");
    MIMIR_ASSERT(f32["aliases"].size() >= 2,                  "f32 aliases trop court");
    MIMIR_ASSERT(f32["aliases"][0].get<std::string>() == "float", "f32 aliases[0] != 'float'");
    MIMIR_ASSERT(f32["bytes"].get<int>() == 4,                "f32 bytes != 4");
    MIMIR_ASSERT(f32["kind"].get<std::string>() == "float",   "f32 kind != 'float'");

    // Cache: premiere lecture donne un JSON object valide (possiblement vide).
    const std::string cache = ScriptingBridgeCommon::loadArchCacheJson();
    MIMIR_ASSERT(cache.front() == '{', "loadArchCacheJson ne commence pas par '{'");

    return 0;
}

// ---------------------------------------------------------------------------
// Test: bridge commun direct (sans script)
// ---------------------------------------------------------------------------

static int testBridgeDirectCommands() {
    // Simuler l'ecriture d'un fichier de commandes et sa lecture par processBridgeCommands.
    const auto cmd_file = fs::temp_directory_path() /
                          ("mimir_test_cmds_" + std::to_string(currentPid()) + ".cmd");

    {
        std::ofstream f(cmd_file);
        f << "Model.create|basic_mlp\n";
        f << "Model.allocate_params\n";
        f << "Model.init_weights|he|0\n";
        f << "Model.total_params\n";
    }

    // Utiliser un contexte runtime disponible comme contexte generique.
#if !defined(MIMIR_ENABLE_SCRIPTING_JS) && \
    !defined(MIMIR_ENABLE_SCRIPTING_CSHARP) && \
    !defined(MIMIR_ENABLE_SCRIPTING_RUST)
    std::cerr << "[SKIP] BridgeTest.Direct: aucun bridge script activé\n";
    {
        std::error_code skip_ec;
        fs::remove(cmd_file, skip_ec);
    }
    return 0;
#else
#if defined(MIMIR_ENABLE_SCRIPTING_JS)
    JSContext& ctx = JSContext::getInstance();
#elif defined(MIMIR_ENABLE_SCRIPTING_CSHARP)
    CSharpContext& ctx = CSharpContext::getInstance();
#else
    RustContext& ctx = RustContext::getInstance();
#endif

    ctx.resetRuntimeState();

    const bool ok = ScriptingBridgeCommon::processBridgeCommands(ctx, cmd_file, "[test]");
    std::error_code ec;
    fs::remove(cmd_file, ec);

    MIMIR_ASSERT(ok, "processBridgeCommands a retourne false");
    if (!assertModelCreated(ctx, "bridge direct")) return 1;
    if (!assertParamsAllocated(ctx, "bridge direct")) return 1;

    ctx.resetRuntimeState();
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Test: bridge via runtime JS
// ---------------------------------------------------------------------------

static int testJsBridge() {
#ifndef MIMIR_ENABLE_SCRIPTING_JS
    std::cerr << "[SKIP] BridgeTest.JS: bridge JS désactivé à la compilation\n";
    return 0;
#else
    if (!commandAvailable("node")) {
        std::cerr << "[SKIP] BridgeTest.JS: node absent du PATH\n";
        return 0;
    }

    const auto script = writeTempScript("mjs",
        "Model.create('basic_mlp', {});\n"
        "Model.allocate_params();\n"
        "Model.init_weights('he', 0);\n"
        "Model.total_params();\n"
    );

    JSContext& ctx = JSContext::getInstance();
    ctx.resetRuntimeState();

    JSScripting js;
    js.registerAPI();
    js.setArgs(script.string(), {});
    const bool ok = js.loadScript(script.string());

    std::error_code ec;
    fs::remove(script, ec);

    MIMIR_ASSERT(ok, "JS bridge script a echoue");
    if (!assertModelCreated(ctx, "JS bridge")) return 1;
    if (!assertParamsAllocated(ctx, "JS bridge")) return 1;

    const auto artifacts = fs::temp_directory_path() /
        ("mimir js artifacts_" + std::to_string(currentPid()));
    fs::create_directories(artifacts / "dataset");
    {
        std::ofstream image(artifacts / "dataset" / "sample.ppm", std::ios::binary);
        image << "P6\n2 1\n255\n";
        const unsigned char pixels[] = {255, 0, 0, 0, 255, 0};
        image.write(reinterpret_cast<const char*>(pixels), sizeof(pixels));
        std::ofstream text(artifacts / "dataset" / "sample.txt");
        text << "bonjour monde";
    }
    // Loaded source files must work without write access to their directory.
    const auto readonly_dir = artifacts / "readonly";
    fs::create_directory(readonly_dir);
    const auto readonly_script = readonly_dir / "source.mjs";
    {
        std::ofstream source(readonly_script);
        source << "#!/usr/bin/env node\n"
                  "import {fileURLToPath} from 'node:url';\n"
                  "if (fileURLToPath(import.meta.url) !== process.argv[1]) throw new Error('identity');\n"
                  "if (Mimir.Model.total_params() <= 0) throw new Error('API unavailable');\n";
    }
    {
        struct RestorePermissions {
            fs::path path;
            fs::perms original;
            ~RestorePermissions() { std::error_code ec; fs::permissions(path, original, ec); }
        } permissions{readonly_dir, fs::status(readonly_dir).permissions()};
        fs::permissions(readonly_dir, fs::perms::owner_read | fs::perms::owner_exec);
        js.setArgs(readonly_script.string(), {});
        MIMIR_ASSERT(js.loadScript(readonly_script.string()), "read-only JS source failed");
    }
    {
        const auto commonjs = artifacts / "source.cjs";
        std::ofstream source(commonjs);
        source << "const fs = require('node:fs');\n"
                  "if (__filename !== process.argv[1] || typeof fs.readFileSync !== 'function') throw new Error('CommonJS');\n"
                  "if (Model.total_params() <= 0) throw new Error('API unavailable');\n";
        source.close();
        js.setArgs(commonjs.string(), {});
        MIMIR_ASSERT(js.loadScript(commonjs.string()), "CommonJS source failed");
    }
    // Exercise live results, the entire registered surface, closures, tuples,
    // exceptions and state persistence across executeScript calls.
    auto& lua_ctx = LuaContext::getInstance();
    const auto original_lua_model = lua_ctx.currentModel;
    struct RestoreContexts {
        JSContext& js;
        LuaContext& lua;
        std::shared_ptr<Model> original_model;
        ~RestoreContexts() {
            js.resetRuntimeState();
            lua.currentModel = original_model;
        }
    } restore_contexts{ctx, lua_ctx, original_lua_model};
    lua_ctx.currentModel = ctx.currentModel;
    const auto saved_lua_model = lua_ctx.currentModel;
    { std::ofstream dependency(artifacts / "dependency.mjs"); dependency << "export const value = 42;\n"; }
    js.setArgs((artifacts / "main.mjs").string(), {"argument with spaces", "é漢字"});
    js.setSystemConfig(json{{"marker", 42}, {"artifacts", artifacts.string()}}, "/tmp/conf.json", "/tmp");
    MIMIR_ASSERT(js.executeScript(R"JS(
      import fs from 'node:fs';
      import { value } from './dependency.mjs';
      function check(value, message) { if (!value) throw new Error(message); }
      check(value === 42 && typeof fs.readFileSync === 'function', 'ESM imports');
      check(arg[1] === 'argument with spaces' && arg[2] === 'é漢字', 'arg');
      check(CONF.marker === 42 && CONF_PATH === '/tmp/conf.json' && CONF_DIR === '/tmp', 'conf');
      check(Model === Mimir.Model && model === Model && viz === Mimir.Viz, 'aliases');
      for (const name of ['Model','Tokenizer','Dataset','Database','Memory','MemoryGuard','Allocator','Htop','Viz','Serialization','Layers','Architectures'])
        check(Mimir[name] && Object.keys(Mimir[name]).length > 0, name);
      check(Model.total_params() > 0, 'live parameter count');
      check(typeof Model.forward === 'function' && typeof Model.backward === 'function', 'execution API');
      let failed = false;
      try { Architectures.default_config('__missing_architecture__'); } catch (e) { failed = true; }
      check(failed, 'catchable native error');
      const raw = Architectures.default_config.results('__missing_architecture__');
      check(raw.length === 2 && raw[0] === null && typeof raw[1] === 'string', 'raw returns');
      Tokenizer.create(256);
      Tokenizer.tokenize_ensure('bonjour monde');
      const ids = Tokenizer.tokenize('bonjour monde');
      check(Array.isArray(ids) && ids.length > 0, 'tokenize');
      check(typeof Tokenizer.detokenize(new Int32Array(ids)) === 'string', 'typed array');
      const bufferIds = Buffer.from(ids);
      check(Tokenizer.detokenize(bufferIds) === Tokenizer.detokenize(Array.from(bufferIds)), 'Buffer transport');
      const marker = {$mimir_function: 1, nested: [0, false, '']};
      write_json(CONF.artifacts + '/marker.json', marker);
      check(read_json(CONF.artifacts + '/marker.json').$mimir_function === 1, 'JSON marker collision');
      for (const invalid of [new DataView(new ArrayBuffer(4)), [NaN], [Infinity], [1n], [Symbol('test')]]) {
        let rejected = false;
        try { Tokenizer.detokenize(invalid); } catch (error) { rejected = true; }
        check(rejected, 'invalid argument must fail');
      }
      const cyclic = {}; cyclic.self = cyclic;
      let cycleRejected = false;
      try { write_json(CONF.artifacts + '/cycle.json', cyclic); } catch (error) { cycleRejected = true; }
      check(cycleRejected, 'cyclic input must fail');
      const [loaded, count] = Dataset.load(CONF.artifacts + '/dataset', 2, 1, 2);
      check(loaded && count === 1, 'dataset load');
      const item = Dataset.get(1);
      check(item.text === 'bonjour monde' && item.channels === 3 && item.image.length === 6, 'dataset item');
      check(item.image[0] === 255 && item.image[4] === 255, 'RGB preserved');
      Dataset.release(1);
      const cached = Mimir.Database.load(CONF.artifacts + '/dataset', 2, 1, 2)
        .cache(CONF.artifacts + '/dataset', 2, 1, 2, CONF.artifacts + '/index.json', 16, true);
      check(cached[0] === true && cached[1] === 1, 'dataset cached builder');
      check(Dataset.get(1, false).image === undefined, 'metadata-only dataset read');
      const loader = Mimir.Database.load('/path/that/does/not/exist');
      check(typeof loader.cache === 'function', 'native returned closure');
      const cacheResult = loader.cache.results();
      check(cacheResult.length === 2 && cacheResult[0] === false && typeof cacheResult[1] === 'string', 'returned closure invocation');
      check(typeof Memory.get_stats() === 'object', 'memory');
      Model.create('basic_mlp', {input_dim: 4, hidden_dim: 8, output_dim: 2});
      check(Mimir.Layers.Linear().length === 3, 'registered native closure');
      Model.set_hardware(false);
      Model.allocate_params();
      Model.init_weights('he', 42);
      const input = new Float32Array([0.1, 0.2, 0.3, 0.4]);
      const output = Model.forward(input, true);
      check(Array.isArray(output) && output.length === 2 && output.every(Number.isFinite), 'forward');
      Model.zero_grads();
      Model.backward([0.1, -0.2]);
      Model.optimizer_step(0.001, 'adamw');
      const integralFloat = Model.forward(new Float32Array([1, 0, 1, 0]), false);
      check(integralFloat.length === 2 && integralFloat.every(Number.isFinite), 'integer-valued floating tensor');
      const trained = Model.forward(input, false);
      Mimir.Serialization.save(CONF.artifacts + '/checkpoint', 'raw_folder');
      Model.init_weights('he', 123);
      Mimir.Serialization.load(CONF.artifacts + '/checkpoint', 'raw_folder');
      const restored = Model.forward(input, false);
      check(restored.every((x, i) => Math.abs(x - trained[i]) < 1e-6), 'checkpoint round trip');
      write_json(CONF.artifacts + '/data.json', {text: 'é漢字 | \n', nested: [1, true, 0.25], integralFloat, large: 'é漢字'.repeat(30000)});
      check(read_json(CONF.artifacts + '/data.json').nested[2] === 0.25, 'JSON IO');
      check(read_json(CONF.artifacts + '/data.json').large.length === 90000, 'large UTF-8 frames');
      check(typeof Viz.backend() === 'string', 'Viz backend');
      console.log('[PASS] JS synchronous surface');
    )JS"), "JS synchronous surface failed");
    {
        std::ifstream file(artifacts / "data.json");
        json stored;
        file >> stored;
        const auto reference = ctx.currentModel->forwardPass(std::vector<float>{1, 0, 1, 0}, false);
        const auto actual = stored.at("integralFloat").get<std::vector<float>>();
        MIMIR_ASSERT(actual.size() == reference.size(), "JS/native forward shape mismatch");
        for (size_t i = 0; i < actual.size(); ++i)
            MIMIR_ASSERT(std::abs(actual[i] - reference[i]) < 1e-6f, "JS/native floating input mismatch");
    }
    MIMIR_ASSERT(lua_ctx.currentModel == saved_lua_model, "Lua state changed by JS");
    MIMIR_ASSERT(ctx.currentTokenizer != nullptr, "JS tokenizer state lost");
    MIMIR_ASSERT(js.executeScript("if (Tokenizer.vocab_size() <= 0) throw new Error('lost state');"), "JS state persistence");
    MIMIR_ASSERT(!js.executeScript("throw new Error('intentional JS failure');"), "uncaught JS error accepted");
    MIMIR_ASSERT(js.executeScript("if (Model.total_params() <= 0) throw new Error('lost model');"), "recovery after failure");
    lua_ctx.currentModel = original_lua_model;
    fs::remove_all(artifacts);
    ctx.resetRuntimeState();
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Test: bridge via runtime C#
// ---------------------------------------------------------------------------

static int testCSharpBridge() {
#ifndef MIMIR_ENABLE_SCRIPTING_CSHARP
    std::cerr << "[SKIP] BridgeTest.CSharp: bridge C# désactivé à la compilation\n";
    return 0;
#else
    const bool has_cs = commandAvailable("dotnet-script") || commandAvailable("csi");
    if (!has_cs || !dotnetRuntimeAvailable()) {
        std::cerr << "[SKIP] BridgeTest.CSharp: dotnet-script/csi absent du PATH\n";
        return 0;
    }

    const auto script = writeTempScript("csx",
        "Model.create(\"basic_mlp\", \"\");\n"
        "Model.allocate_params();\n"
        "Model.init_weights(\"he\", 0);\n"
        "long p = (long)Model.total_params();\n"
    );

    CSharpContext& ctx = CSharpContext::getInstance();
    ctx.resetRuntimeState();

    CSharpScripting cs;
    cs.registerAPI();
    cs.setArgs(script.string(), {});
    const bool ok = cs.loadScript(script.string());

    std::error_code ec;
    fs::remove(script, ec);

    MIMIR_ASSERT(ok, "C# bridge script a echoue");
    if (!assertModelCreated(ctx, "CSharp bridge")) return 1;
    if (!assertParamsAllocated(ctx, "CSharp bridge")) return 1;

    ctx.resetRuntimeState();
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Test: bridge via runtime Rust
// ---------------------------------------------------------------------------

static int testRustBridge() {
#ifndef MIMIR_ENABLE_SCRIPTING_RUST
    std::cerr << "[SKIP] BridgeTest.Rust: bridge Rust désactivé à la compilation\n";
    return 0;
#else
    if (!commandAvailable("rust-script")) {
        std::cerr << "[SKIP] BridgeTest.Rust: rust-script absent du PATH\n";
        return 0;
    }

    const auto script = writeTempScript("rs",
        "Model::create(\"basic_mlp\", \"\");\n"
        "Model::allocate_params();\n"
        "Model::init_weights(\"he\", 0);\n"
        "let _p = Model::total_params();\n"
    );

    RustContext& ctx = RustContext::getInstance();
    ctx.resetRuntimeState();

    RustScripting rs;
    rs.registerAPI();
    rs.setArgs(script.string(), {});
    const bool ok = rs.loadScript(script.string());

    std::error_code ec;
    fs::remove(script, ec);

    MIMIR_ASSERT(ok, "Rust bridge script a echoue");
    if (!assertModelCreated(ctx, "Rust bridge")) return 1;
    if (!assertParamsAllocated(ctx, "Rust bridge")) return 1;

    ctx.resetRuntimeState();
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Test: Model.create avec config JSON personnalisee
// ---------------------------------------------------------------------------

static int testBridgeCreateWithConfig() {
    const auto cmd_file = fs::temp_directory_path() /
                          ("mimir_test_cfg_" + std::to_string(currentPid()) + ".cmd");
    {
        std::ofstream f(cmd_file);
        // Config minimale valide pour basic_mlp
        f << "Model.create|basic_mlp|{\"hidden_size\":256,\"input_size\":512,\"output_size\":10}\n";
        f << "Model.allocate_params\n";
    }

#if !defined(MIMIR_ENABLE_SCRIPTING_JS) && \
    !defined(MIMIR_ENABLE_SCRIPTING_CSHARP) && \
    !defined(MIMIR_ENABLE_SCRIPTING_RUST)
    std::cerr << "[SKIP] BridgeTest.CreateWithConfig: aucun bridge script activé\n";
    {
        std::error_code skip_ec;
        fs::remove(cmd_file, skip_ec);
    }
    return 0;
#else
#if defined(MIMIR_ENABLE_SCRIPTING_JS)
    JSContext& ctx = JSContext::getInstance();
#elif defined(MIMIR_ENABLE_SCRIPTING_CSHARP)
    CSharpContext& ctx = CSharpContext::getInstance();
#else
    RustContext& ctx = RustContext::getInstance();
#endif

    ctx.resetRuntimeState();

    const bool ok = ScriptingBridgeCommon::processBridgeCommands(ctx, cmd_file, "[test-cfg]");
    std::error_code ec;
    fs::remove(cmd_file, ec);

    MIMIR_ASSERT(ok, "bridge create+cfg a echoue");
    if (!assertModelCreated(ctx, "bridge create+cfg")) return 1;

    ctx.resetRuntimeState();
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    int failures = 0;

    failures += testBridgeCommonData();
    failures += testBridgeDirectCommands();
    failures += testBridgeCreateWithConfig();
    failures += testJsBridge();
    failures += testCSharpBridge();
    failures += testRustBridge();

    if (failures == 0)
        std::cerr << "[PASS] BridgeTest: tous les sous-tests reussis\n";
    else
        std::cerr << "[FAIL] BridgeTest: " << failures << " echec(s)\n";

    return failures == 0 ? 0 : 1;
}
