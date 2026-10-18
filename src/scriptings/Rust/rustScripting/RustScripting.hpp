#pragma once

#include <string>
#include <vector>

#include "scriptings/ScriptingRuntime.hpp"
#include "scriptings/ScriptingContext.hpp"

class RustScripting : public ScriptingRuntime {
public:
    RustScripting() = default;
    ~RustScripting() override = default;

    void setArgs(const std::string& script_path,
                 const std::vector<std::string>& script_args) override;
    bool loadScript(const std::string& filepath) override;
    bool executeScript(const std::string& code) override;
    void registerAPI() override;

private:
    std::string loaded_file_;
};

class RustContext : public ScriptingContext {
public:
    static RustContext& getInstance() {
        static RustContext instance;
        return instance;
    }

private:
    RustContext() = default;
    ~RustContext() = default;
    RustContext(const RustContext&) = delete;
    RustContext& operator=(const RustContext&) = delete;
};
