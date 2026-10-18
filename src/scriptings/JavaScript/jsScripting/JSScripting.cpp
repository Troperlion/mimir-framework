#include "JSScripting.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <fcntl.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include "scriptings/Lua/luaScripting/LuaScripting.hpp"
#include "scriptings/ScriptingBridgeCommon.hpp"

extern char** environ;

namespace {
// The reference callbacks use LuaContext. Borrow it only during a JS execution,
// preserving the Lua session and keeping the JS runtime state in JSContext.
std::mutex execution_mutex;
struct ContextScope {
    ScriptingContext& js = JSContext::getInstance();
    ScriptingContext& lua = LuaContext::getInstance();
    void exchange() {
        using std::swap;
        swap(js.currentModel, lua.currentModel);
        swap(js.currentTokenizer, lua.currentTokenizer);
        swap(js.currentEncoder, lua.currentEncoder);
        swap(js.asyncMonitor, lua.asyncMonitor);
        swap(js.currentSequences, lua.currentSequences);
        swap(js.currentConfig, lua.currentConfig);
        swap(js.currentDataset, lua.currentDataset);
        swap(js.modelType, lua.modelType);
        swap(js.modelConfig, lua.modelConfig);
        swap(js.logs, lua.logs);
        swap(js.suppress_stdout_logs, lua.suppress_stdout_logs);
    }
    ContextScope() { exchange(); }
    ~ContextScope() { exchange(); }
};

struct Descriptor {
    int fd = -1;
    ~Descriptor() { if (fd >= 0) close(fd); }
};

struct Values {
    lua_State* L;
    std::vector<int> refs;
    std::string session;
    ~Values() { for (int r : refs) luaL_unref(L, LUA_REGISTRYINDEX, r); }

    json read(int index, int depth = 0) {
        if (depth > 64) throw std::runtime_error("bridge value nesting exceeds 64");
        index = lua_absindex(L, index);
        switch (lua_type(L, index)) {
        case LUA_TNIL: return nullptr;
        case LUA_TBOOLEAN: return bool(lua_toboolean(L, index));
        case LUA_TNUMBER:
            if (lua_isinteger(L, index)) return lua_tointeger(L, index);
            return lua_tonumber(L, index);
        case LUA_TSTRING: {
            size_t n = 0;
            const char* s = lua_tolstring(L, index, &n);
            return std::string(s, n);
        }
        case LUA_TFUNCTION: {
            lua_pushvalue(L, index);
            int ref = luaL_ref(L, LUA_REGISTRYINDEX);
            refs.push_back(ref);
            return json{{"$mimir_function", ref}, {"$mimir_session", session}};
        }
        case LUA_TTABLE: {
            // Lua inspection snapshots are empty read-only proxies. Serialize
            // their detached backing data so JS receives the same layer list.
            if (lua_getmetatable(L, index)) {
                lua_getfield(L, -1, "__mimir_layer_snapshot");
                const bool snapshot = lua_toboolean(L, -1);
                lua_pop(L, 1);
                if (snapshot) {
                    lua_getfield(L, -1, "__index");
                    if (lua_istable(L, -1)) {
                        json value = read(-1, depth + 1);
                        lua_pop(L, 2);
                        return value;
                    }
                    lua_pop(L, 1);
                }
                lua_pop(L, 1);
            }
            json object = json::object();
            size_t count = 0;
            bool sequence = true;
            const size_t length = lua_rawlen(L, index);
            lua_pushnil(L);
            while (lua_next(L, index)) {
                std::string key;
                if (lua_type(L, -2) == LUA_TSTRING) {
                    size_t n = 0;
                    const char* s = lua_tolstring(L, -2, &n);
                    key.assign(s, n);
                    sequence = false;
                } else if (lua_isinteger(L, -2)) {
                    const auto k = lua_tointeger(L, -2);
                    key = std::to_string(k);
                    if (k < 1 || static_cast<size_t>(k) > length) sequence = false;
                } else {
                    throw std::runtime_error("unsupported bridge table key");
                }
                object[key] = read(-1, depth + 1);
                ++count;
                lua_pop(L, 1);
            }
            if (sequence && count > 0 && count == length) {
                json array = json::array();
                for (size_t i = 1; i <= length; ++i) array.push_back(object.at(std::to_string(i)));
                return array;
            }
            return object;
        }
        default: throw std::runtime_error("unsupported native bridge value");
        }
    }

    void push(const json& value, int depth = 0) {
        if (depth > 64) throw std::runtime_error("bridge value nesting exceeds 64");
        if (value.is_object() && value.value("$mimir_session", json{}) == session &&
            value.value("$mimir_array_kind", json{}) == "float" && value.contains("data")) {
            const auto& data = value.at("data");
            if (!data.is_array()) throw std::runtime_error("typed array data must be an array");
            lua_newtable(L);
            for (size_t i = 0; i < data.size(); ++i) {
                lua_pushnumber(L, data.at(i).get<double>());
                lua_rawseti(L, -2, i + 1);
            }
        } else if (value.is_null()) lua_pushnil(L);
        else if (value.is_boolean()) lua_pushboolean(L, value.get<bool>());
        else if (value.is_number_integer()) lua_pushinteger(L, value.get<lua_Integer>());
        else if (value.is_number()) lua_pushnumber(L, value.get<double>());
        else if (value.is_string()) {
            const auto s = value.get<std::string>();
            lua_pushlstring(L, s.data(), s.size());
        } else if (value.is_array()) {
            lua_newtable(L);
            for (size_t i = 0; i < value.size(); ++i) {
                push(value[i], depth + 1);
                lua_rawseti(L, -2, i + 1);
            }
        } else {
            lua_newtable(L);
            for (auto it = value.begin(); it != value.end(); ++it) {
                lua_pushlstring(L, it.key().data(), it.key().size());
                push(it.value(), depth + 1);
                lua_rawset(L, -3);
            }
        }
    }

    json call(const json& request) {
        const int top = lua_gettop(L);
        try {
            const int ref = request.at("function").get<int>();
            if (std::find(refs.begin(), refs.end(), ref) == refs.end())
                throw std::runtime_error("unknown native bridge function");
            lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
            const auto& args = request.at("args");
            if (!args.is_array()) throw std::runtime_error("bridge arguments must be an array");
            for (const auto& arg : args) push(arg);
            if (lua_pcall(L, static_cast<int>(args.size()), LUA_MULTRET, 0) != LUA_OK) {
                const char* error = lua_tostring(L, -1);
                throw std::runtime_error(error ? error : "native API error");
            }
            json results = json::array();
            for (int i = top + 1; i <= lua_gettop(L); ++i) results.push_back(read(i));
            lua_settop(L, top);
            return json{{"results", results}};
        } catch (const std::exception& e) {
            lua_settop(L, top);
            return json{{"error", e.what()}};
        }
    }
};

bool sendLine(int fd, const json& value) {
    const std::string data = value.dump() + "\n";
    size_t offset = 0;
    while (offset < data.size()) {
        const auto n = send(fd, data.data() + offset, data.size() - offset, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        offset += static_cast<size_t>(n);
    }
    return true;
}

const char* bootstrap = R"JS(
import __mimirFs from 'node:fs';
{
const fs = __mimirFs;
const channel = 3;
let pending = Buffer.alloc(0);
function receive() {
  for (;;) {
    const end = pending.indexOf(10);
    if (end >= 0) {
      const line = pending.subarray(0, end).toString('utf8');
      pending = pending.subarray(end + 1);
      return JSON.parse(line);
    }
    const buffer = Buffer.allocUnsafe(65536);
    const count = fs.readSync(channel, buffer, 0, buffer.length, null);
    if (!count) throw new Error('Mimir native bridge disconnected');
    pending = Buffer.concat([pending, buffer.subarray(0, count)]);
  }
}
function normalize(value, ancestors = new Set(), depth = 0) {
  if (depth > 64) throw new TypeError('Mimir value nesting exceeds 64');
  if (ArrayBuffer.isView(value)) {
    if (value instanceof DataView) throw new TypeError('Use a typed array instead of DataView');
    const data = normalize(Array.from(value), ancestors, depth + 1);
    if (value instanceof Float32Array || value instanceof Float64Array)
      return {$mimir_array_kind: 'float', $mimir_session: initial.session, data};
    return data;
  }
  if (typeof value === 'function' || typeof value === 'bigint' || typeof value === 'symbol')
    throw new TypeError('Mimir arguments must be JSON values or typed arrays');
  if (typeof value === 'number' && !Number.isFinite(value))
    throw new TypeError('Mimir arguments must contain finite numbers');
  if (value === null || typeof value !== 'object') return value;
  if (ancestors.has(value)) throw new TypeError('Mimir arguments must not contain cycles');
  ancestors.add(value);
  try {
    if (Array.isArray(value)) return value.map(item => normalize(item, ancestors, depth + 1));
    if (typeof value.toJSON === 'function') return normalize(value.toJSON(), ancestors, depth + 1);
    const result = Object.create(null);
    for (const [key, item] of Object.entries(value)) result[key] = normalize(item, ancestors, depth + 1);
    return result;
  } finally { ancestors.delete(value); }
}
function invoke(id, args, raw = false) {
  // Normalize before JSON.stringify: Buffer.toJSON otherwise masks typed arrays.
  const data = Buffer.from(JSON.stringify({function: id, args: normalize(args)}) + '\n');
  let offset = 0;
  while (offset < data.length) {
    const count = fs.writeSync(channel, data, offset, data.length - offset);
    if (!count) throw new Error('Mimir native bridge disconnected');
    offset += count;
  }
  const reply = receive();
  if (reply.error !== undefined) throw new Error(reply.error);
  const results = reply.results.map(hydrate);
  if (raw) return results;
  // Lua's nil,error convention maps to a catchable JS exception. Successful
  // multiple returns remain tuples, single returns remain ordinary JS values.
  if (results.length >= 2 && (results[0] === null || results[0] === false) && typeof results[1] === 'string')
    throw new Error(results[1]);
  return results.length <= 1 ? results[0] : results;
}
function hydrate(value) {
  if (value === null || typeof value !== 'object') return value;
  if (value.$mimir_session === initial.session && Object.hasOwn(value, '$mimir_function')) {
    const id = value.$mimir_function;
    const fn = (...args) => invoke(id, args);
    fn.results = (...args) => invoke(id, args, true);
    return fn;
  }
  if (Array.isArray(value)) return value.map(hydrate);
  const result = {};
  for (const [key, item] of Object.entries(value))
    Object.defineProperty(result, key, {value: hydrate(item), enumerable: true, writable: true, configurable: true});
  return result;
}
const initial = receive();
Object.assign(globalThis, hydrate(initial.globals));
// Keep the original JS dtype representation and architecture creation alias.
const nativeDtypes = Mimir.Architectures.dtypes;
Mimir.Architectures.dtypes = () => nativeDtypes().map(dtype => ({...dtype,
  aliases: typeof dtype.aliases === 'string' ? dtype.aliases.split(',').map(s => s.trim()) : dtype.aliases}));
Mimir.Architectures.dtypes.results = nativeDtypes.results;
Mimir.Architectures.create = Mimir.Model.create;
for (const [name, value] of Object.entries(Mimir)) globalThis[name] = value;
for (const [alias, name] of Object.entries({model:'Model', architectures:'Architectures', tokenizer:'Tokenizer', dataset:'Dataset', Memory:'Memory', MemoryGuard:'MemoryGuard', Allocator:'Allocator', htop:'Htop', viz:'Viz'}))
  globalThis[alias] = Mimir[name];
}
)JS";
} // namespace

JSScripting::JSScripting() : native_(std::make_unique<LuaScripting>()) {}
JSScripting::~JSScripting() = default;

void JSScripting::setArgs(const std::string& path, const std::vector<std::string>& args) {
    cacheArgs(path, args);
}

void JSScripting::setSystemConfig(const json& conf, const std::string& path, const std::string& dir) {
    JSContext::getInstance().currentConfig = conf;
    conf_path_ = path;
    conf_dir_ = dir;
}

bool JSScripting::loadScript(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        JSContext::getInstance().addLog("[js] unable to open script: " + path);
        return false;
    }
    file.close();
    return runScript("", std::filesystem::absolute(path).string());
}

bool JSScripting::executeScript(const std::string& code) {
    return runScript(code, "");
}

bool JSScripting::runScript(const std::string& code, const std::string& source_file) {
    std::lock_guard<std::mutex> lock(execution_mutex);
    ContextScope context;
    auto& ctx = LuaContext::getInstance();
    try {
        lua_State* L = native_->getState();
        struct StackScope {
            lua_State* state;
            int top;
            ~StackScope() { lua_settop(state, top); }
        } stack{L, lua_gettop(L)};
        std::random_device random;
        std::string session;
        for (int i = 0; i < 4; ++i) session += std::to_string(random()) + ":";
        Values values{L, {}, session};
        json globals;
        for (const char* name : {ScriptingContext::kGlobalNamespace, "log", "read_json", "write_json", "ROOTWORK"}) {
            lua_getglobal(L, name);
            globals[name] = values.read(-1);
            lua_pop(L, 1);
        }
        globals[ScriptingContext::kGlobalArg] = json::array();
        globals[ScriptingContext::kGlobalArg].push_back(scriptPath());
        for (const auto& arg : scriptArgs()) globals[ScriptingContext::kGlobalArg].push_back(arg);
        globals[ScriptingContext::kGlobalConf] = ctx.currentConfig.is_null() ? json::object() : ctx.currentConfig;
        globals[ScriptingContext::kGlobalConfPath] = conf_path_;
        globals[ScriptingContext::kGlobalConfDir] = conf_dir_;
        native_->setArgs(scriptPath(), scriptArgs());
        native_->setSystemConfig(globals[ScriptingContext::kGlobalConf], conf_path_, conf_dir_);

        // Preload the bridge separately: loaded files keep their real module
        // identity, hashbangs, package scope and relative import resolution.
        struct RemoveFile {
            std::string path;
            ~RemoveFile() { if (!path.empty()) { std::error_code ec; std::filesystem::remove(path, ec); } }
        } preload, inline_module;
        auto writeModule = [](const std::filesystem::path& directory, const std::string& contents,
                              RemoveFile& cleanup) {
            std::string pattern = (directory / ".mimir-js-XXXXXX.mjs").string();
            std::vector<char> filename(pattern.begin(), pattern.end());
            filename.push_back('\0');
            Descriptor temporary{mkstemps(filename.data(), 4)};
            if (temporary.fd < 0) throw std::runtime_error("cannot create JS module: " + std::string(strerror(errno)));
            cleanup.path = filename.data();
            size_t offset = 0;
            while (offset < contents.size()) {
                const auto n = write(temporary.fd, contents.data() + offset, contents.size() - offset);
                if (n < 0 && errno == EINTR) continue;
                if (n <= 0) throw std::runtime_error("cannot write JS module");
                offset += static_cast<size_t>(n);
            }
        };
        writeModule(std::filesystem::temp_directory_path(), bootstrap, preload);
        std::string entry = source_file;
        if (entry.empty()) {
            // Inline execution retains the established scriptPath-relative ESM convention.
            const auto directory = scriptPath().empty() ? std::filesystem::current_path()
                : std::filesystem::absolute(scriptPath()).parent_path();
            writeModule(directory, code, inline_module);
            entry = inline_module.path;
        }
        int pair[2];
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair)) throw std::runtime_error("cannot create JS channel");
        Descriptor parent{pair[0]}, child{pair[1]};
        // Move the child endpoint away from fd 3 before installing spawn actions.
        Descriptor endpoint{fcntl(child.fd, F_DUPFD_CLOEXEC, 10)};
        if (endpoint.fd < 0) throw std::runtime_error("cannot duplicate JS channel");
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_addclose(&actions, parent.fd);
        posix_spawn_file_actions_addclose(&actions, child.fd);
        posix_spawn_file_actions_adddup2(&actions, endpoint.fd, 3);
        posix_spawn_file_actions_addclose(&actions, endpoint.fd);
        std::vector<std::string> arguments{"node", "--import", preload.path, entry};
        for (const auto& arg : scriptArgs()) arguments.push_back(arg);
        std::vector<char*> argv;
        for (auto& arg : arguments) argv.push_back(arg.data());
        argv.push_back(nullptr);
        pid_t pid;
        const int error = posix_spawnp(&pid, "node", &actions, nullptr, argv.data(), environ);
        posix_spawn_file_actions_destroy(&actions);
        if (error) throw std::runtime_error("cannot start node: " + std::string(strerror(error)));
        struct ChildProcess {
            pid_t pid;
            int channel;
            ~ChildProcess() {
                if (pid > 0) {
                    shutdown(channel, SHUT_RDWR);
                    int status;
                    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
                }
            }
        } process{pid, parent.fd};
        close(child.fd); child.fd = -1;
        close(endpoint.fd); endpoint.fd = -1;
        bool ok = sendLine(parent.fd, json{{"globals", globals}, {"session", session}});
        std::string pending;
        char buffer[65536];
        while (ok) {
            const auto n = recv(parent.fd, buffer, sizeof(buffer), 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { if (n < 0) ok = false; break; }
            pending.append(buffer, static_cast<size_t>(n));
            size_t end;
            while ((end = pending.find('\n')) != std::string::npos) {
                json reply;
                try { reply = values.call(json::parse(pending.substr(0, end))); }
                catch (const std::exception& e) { reply = json{{"error", e.what()}}; }
                pending.erase(0, end + 1);
                if (!sendLine(parent.fd, reply)) { ok = false; break; }
            }
        }
        shutdown(parent.fd, SHUT_RDWR);
        int status = 0;
        pid_t waited;
        do { waited = waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
        process.pid = -1;
        ok = ok && waited == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0;
        if (!ok) {
            std::string detail;
            if (waited == pid && WIFEXITED(status)) detail = " (exit=" + std::to_string(WEXITSTATUS(status)) + ")";
            else if (waited == pid && WIFSIGNALED(status)) detail = " (signal=" + std::to_string(WTERMSIG(status)) + ")";
            ctx.addLog("[js] script execution failed" + detail);
        }
        return ok;
    } catch (const std::exception& e) {
        ctx.addLog(std::string("[js] execution error: ") + e.what());
        return false;
    }
}

void JSScripting::registerAPI() {
    // LuaScripting registers the reference native API at construction. The JS
    // manifest is discovered from that registry on every execution.
}
