#include "test_utils.hpp"

#include "scriptings/Lua/luaScripting/LuaScripting.hpp"

int main() {
    LuaScripting lua;

    // Validate that the binding exists and can create a model from a full config.
    const std::string code = R"(
        TEST_OK = false
        TEST_ARCH = ""
        TEST_ERR = ""

        if type(Mimir) ~= "table" or type(Mimir.Model) ~= "table" then
            TEST_ERR = "Mimir API not registered"
            return
        end
        if type(Mimir.Model.create_from_config) ~= "function" then
            TEST_ERR = "create_from_config not found"
            return
        end

        local conf = {
            architecture = "basic_mlp",
            basic_mlp = { input_dim = 2, hidden_dim = 4, output_dim = 1, hidden_layers = 1, dropout = 0.0 },
            tokenizer = { max_vocab = 128, max_sequence_length = 16 },
            encoder = { embedding_dim = 32 },
            training = { optimizer = "adamw", learning_rate = 1e-3 },
        }

        local ok, arch_or_err = Mimir.Model.create_from_config(conf)
        if ok then
            TEST_OK = true
            TEST_ARCH = tostring(arch_or_err)
        else
            TEST_OK = false
            TEST_ERR = tostring(arch_or_err)
        end
    )";

    TASSERT_TRUE(lua.executeScript(code));
    TASSERT_TRUE(lua.getBoolean("TEST_OK"));
    TASSERT_TRUE(lua.getString("TEST_ARCH") == "basic_mlp");

    LuaContext::getInstance().resetRuntimeState();

    return 0;
}
