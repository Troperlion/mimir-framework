#pragma once

#include <memory>
#include <string>
#include <vector>

#include "scriptings/ScriptingRuntime.hpp"
#include "scriptings/ScriptingContext.hpp"

class LuaScripting;

class JSScripting : public ScriptingRuntime {
public:
    JSScripting();
    ~JSScripting() override;

    void setArgs(const std::string& script_path,
                 const std::vector<std::string>& script_args) override;
    bool loadScript(const std::string& filepath) override;
    bool executeScript(const std::string& code) override;
    void registerAPI() override;
    void setSystemConfig(const json& conf, const std::string& path, const std::string& dir);

private:
    bool runScript(const std::string& code, const std::string& source_file);
    std::unique_ptr<LuaScripting> native_;
    std::string conf_path_, conf_dir_;
};

class JSContext : public ScriptingContext {
public:
    static JSContext& getInstance() {
        static JSContext instance;
        return instance;
    }

private:
    JSContext() = default;
    ~JSContext() = default;
    JSContext(const JSContext&) = delete;
    JSContext& operator=(const JSContext&) = delete;
};
