#pragma once

#include <string>
#include <vector>

#include "scriptings/ScriptingRuntime.hpp"
#include "scriptings/ScriptingContext.hpp"

class CSharpScripting : public ScriptingRuntime {
public:
    CSharpScripting() = default;
    ~CSharpScripting() override = default;

    void setArgs(const std::string& script_path,
                 const std::vector<std::string>& script_args) override;
    bool loadScript(const std::string& filepath) override;
    bool executeScript(const std::string& code) override;
    void registerAPI() override;

private:
    std::string loaded_file_;
};

class CSharpContext : public ScriptingContext {
public:
    static CSharpContext& getInstance() {
        static CSharpContext instance;
        return instance;
    }

private:
    CSharpContext() = default;
    ~CSharpContext() = default;
    CSharpContext(const CSharpContext&) = delete;
    CSharpContext& operator=(const CSharpContext&) = delete;
};
