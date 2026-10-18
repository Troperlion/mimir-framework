#include "test_utils.hpp"

#include "scriptings/Lua/luaScripting/LuaScripting.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>

int main() {
    namespace fs = std::filesystem;

    const fs::path rootwork = fs::temp_directory_path() / "mimir lua rootwork";
    fs::create_directories(rootwork / "modules");
    fs::create_directories(rootwork / "scripts" / "modules");
    fs::copy_file(fs::current_path() / "scripts" / "modules" / "mpk.lua",
                  rootwork / "scripts" / "modules" / "mpk.lua",
                  fs::copy_options::overwrite_existing);
    {
        std::ofstream module(rootwork / "modules" / "dependency.lua");
        module << "return { value = 'loaded' }\n";
    }
#if defined(_WIN32)
    _putenv_s("ROOTWORK", rootwork.string().c_str());
#else
    setenv("ROOTWORK", rootwork.string().c_str(), 1);
#endif

    LuaScripting lua;

    TASSERT_TRUE(lua.executeScript(R"(
        local dependency = dofile(ROOTWORK.."/modules/dependency.lua")
        ROOTWORK_MODULE_OK = dependency.value == "loaded"
    )"));
    TASSERT_TRUE(lua.getBoolean("ROOTWORK_MODULE_OK"));

    const std::string code = R"(
        TEST_OK = false
        TEST_ERR = ""

        if type(Mimir) ~= "table" then TEST_ERR = "Mimir missing" return end
        if type(Mimir.Architectures) ~= "table" then TEST_ERR = "Architectures missing" return end
        if type(Mimir.Architectures.available) ~= "function" then TEST_ERR = "available missing" return end
        if type(Mimir.Architectures.default_config) ~= "function" then TEST_ERR = "default_config missing" return end
        if type(Mimir.Architectures.info) ~= "function" then TEST_ERR = "info missing" return end
        if type(Mimir.Architectures.dtypes) ~= "function" then TEST_ERR = "dtypes missing" return end
        if type(Mimir.Model) ~= "table" then TEST_ERR = "Model missing" return end
        if type(Mimir.Model.create) ~= "function" then TEST_ERR = "Model.create missing" return end
        if type(Mimir.Model.get_config) ~= "function" then TEST_ERR = "Model.get_config missing" return end
        if type(Mimir.Model.lumen_begin_vae_calibration) ~= "function" then TEST_ERR = "Lumen calibration missing" return end
        if type(Mimir.Model.lumen_add_vae_calibration_image) ~= "function" then TEST_ERR = "Lumen calibration image missing" return end
        if type(Mimir.Model.lumen_finish_vae_calibration) ~= "function" then TEST_ERR = "Lumen calibration finish missing" return end
        if type(Mimir.Model.lumen_train_step) ~= "function" then TEST_ERR = "Lumen train step missing" return end
        if type(Mimir.Benchmark) ~= "table" or type(Mimir.Benchmark.now) ~= "function" then
            TEST_ERR = "Benchmark.now missing"
            return
        end
        local benchmark_t0 = Mimir.Benchmark.now()
        local benchmark_t1 = Mimir.Benchmark.now()
        if type(benchmark_t0) ~= "number" or benchmark_t1 < benchmark_t0 then
            TEST_ERR = "Benchmark.now is not monotonic"
            return
        end
        assert(type(Mimir.Viz.backend()) == "string")
        assert(type(Mimir.Viz.poll_events()) == "table")
        assert(type(Mimir.Viz.set_detections) == "function")
        assert(type(Mimir.Viz.show_detections) == "function")
        assert(type(Mimir.Viz.set_detection_threshold) == "function")
        assert(not pcall(Mimir.Viz.set_detections, {{x1=-0.1,y1=0.1,x2=0.5,y2=0.5}}))
        assert(not pcall(Mimir.Viz.set_detections, {{x1=0.5,y1=0.1,x2=0.2,y2=0.5}}))
        assert(not pcall(Mimir.Viz.show_detections, "yes"))
        assert(not pcall(Mimir.Viz.set_detection_threshold, -0.1))
        assert(not pcall(Mimir.Viz.set_detection_threshold, 1.1))
        local threshold_ok, threshold_err = Mimir.Viz.set_detection_threshold(0.25)
        assert(threshold_ok == false and type(threshold_err) == "string")
        local detection_ok, detection_err = Mimir.Viz.set_detections({
            {x1=0.1,y1=0.2,x2=0.7,y2=0.8,label="object",score=0.9,class_id=1}
        }, "dataset")
        assert(detection_ok == false and type(detection_err) == "string")
        local scene_ok, scene_err = Mimir.Viz.configure({controls={}})
        assert(scene_ok == false and type(scene_err) == "string")
        assert(not pcall(Mimir.Viz.poll_events, -1))
        assert(not pcall(Mimir.Viz.poll_events, 1001))
        if type(Mimir.Viz.validation_enabled) ~= "function" then TEST_ERR = "Viz.validation_enabled missing" return end
        if type(Mimir.Model.lumen_validate_step) ~= "function" then TEST_ERR = "Lumen validation step missing" return end
        if type(Mimir.Dataset.release) ~= "function" then TEST_ERR = "Dataset.release missing" return end

        local release_ok, release_err = Mimir.Dataset.release(1)
        if release_ok ~= false or type(release_err) ~= "string" then
            TEST_ERR = "Dataset.release should reject an unloaded dataset"
            return
        end

        local archs = Mimir.Architectures.available()
        if type(archs) ~= "table" or #archs < 1 then TEST_ERR = "no architectures" return end

        local cfg = Mimir.Architectures.default_config("basic_mlp")
        if type(cfg) ~= "table" then TEST_ERR = "default_config not table" return end

        -- info() sans argument: liste complète
        local entries = Mimir.Architectures.info()
        if type(entries) ~= "table" or #entries < 1 then TEST_ERR = "info() not a list" return end
        if type(entries[1].name) ~= "string" then TEST_ERR = "info() entry has no name" return end
        if type(entries[1].config) ~= "table" then TEST_ERR = "info() entry has no config" return end

        -- info(name): entrée unique
        local one = Mimir.Architectures.info("basic_mlp")
        if type(one) ~= "table" or one.name ~= "basic_mlp" then TEST_ERR = "info(name) wrong" return end
        if type(one.description) ~= "string" then TEST_ERR = "info(name) has no description" return end

        -- info(unknown): (nil, err)
        local bad, berr = Mimir.Architectures.info("__no_such_arch__")
        if bad ~= nil or type(berr) ~= "string" then TEST_ERR = "info(unknown) should fail" return end

        -- dtypes(): liste de descripteurs
        local dts = Mimir.Architectures.dtypes()
        if type(dts) ~= "table" or #dts < 1 then TEST_ERR = "dtypes() not a list" return end
        if type(dts[1].name) ~= "string" or type(dts[1].bytes) ~= "number" then TEST_ERR = "dtypes() entry malformed" return end

        local ok = Mimir.Model.create("basic_mlp", cfg)
        if ok ~= true then TEST_ERR = "Model.create returned false" return end
        -- Every canonical type has an exact, read-only inspection filter.
        local available = Mimir.Layers.available()
        assert(#available == 73)
        local seen = {}
        for _, name in ipairs(available) do
            seen[name] = true
            assert(type(Mimir.Layers[name]) == "function")
            assert(type(Mimir.Layers[name:lower()]) == "function")
            local snapshot = Mimir.Layers[name:lower()]()
            local count = 0
            for _, layer in ipairs(Mimir.Model.get_layers()) do
                if layer.type == name then count = count + 1 end
            end
            assert(#snapshot == count, name .. " must filter by exact type")
            local pairs_count = 0
            for _, layer in pairs(snapshot) do
                pairs_count = pairs_count + 1
                assert(layer.type == name)
                assert(not pcall(function() layer.name = "changed" end))
                assert(not pcall(function() layer.inputs[1] = "changed" end))
                local fields = 0
                for key, value in pairs(layer) do fields = fields + 1 end
                assert(fields > 0)
            end
            assert(pairs_count == count)
            assert(not pcall(function() snapshot[1] = {} end))
            assert(not pcall(function() setmetatable(snapshot, {}) end))
        end
        assert(seen.PatchEmbed and seen.Reparameterize)
        assert(not pcall(function() available[1] = "changed" end))
        local backend_ops = Mimir.Model.runtime_ops()
        assert(#backend_ops == 6)
        local levels = {Native=true, NativeOptimized=true, HostFallback=true, Unsupported=true}
        for _, backend in ipairs(backend_ops) do
            assert(type(backend.compiled) == "boolean")
            assert(#backend.ops == #available)
            for _, op in ipairs(backend.ops) do
                assert(seen[op.type] and levels[op.forward] and levels[op.backward])
                if not backend.compiled then
                    assert(op.forward == "Unsupported" and op.backward == "Unsupported")
                end
                if backend.name == "CPU" and op.type == "Linear" then
                    assert(backend.compiled and op.forward == "Native" and op.backward == "Native")
                end
            end
        end
        local effective_cfg = Mimir.Model.get_config()
        if type(effective_cfg) ~= "table" or effective_cfg.type ~= "basic_mlp" then
            TEST_ERR = "Model.get_config returned an invalid effective config"
            return
        end

        local pixels, generation_err = Mimir.Model.lumen_text2img("test", 1, 1, 1.0)
        if pixels ~= nil or type(generation_err) ~= "string" then
            TEST_ERR = "Lumen generation should reject a non-Lumen model"
            return
        end
        local calibration_ok, calibration_err = Mimir.Model.lumen_begin_vae_calibration()
        if calibration_ok ~= false or type(calibration_err) ~= "string" then
            TEST_ERR = "Lumen calibration should reject a non-Lumen model"
            return
        end

        TEST_OK = true
    )";

    TASSERT_TRUE(lua.executeScript(code));
    TASSERT_TRUE(lua.getBoolean("TEST_OK"));

    const std::filesystem::path mpk_path = std::filesystem::temp_directory_path() / "mimir_create_path_test.mpk";
    const std::string mpk_code = R"(
        TEST_PATH_OK = false
        TEST_PATH_ERR = ""

        local MPK = dofile(ROOTWORK.."/scripts/modules/mpk.lua")
        local pkg = MPK.build({
            name = "path_plugin_test",
            type = "basic_mlp",
            author = "test",
            description = "path-based create test",
            base_config = { input_dim = 2, hidden_dim = 4, output_dim = 1, hidden_layers = 1, dropout = 0.0 },
            model_structure = MPK.model_structure_template("basic_mlp"),
            container = "pseudocode",
        })
        if not pkg then
            TEST_PATH_ERR = "MPK.build failed"
            return
        end

        local ok_write, err_write = MPK.write("__MPK_PATH__", pkg, { binary = false })
        if not ok_write then
            TEST_PATH_ERR = tostring(err_write)
            return
        end

        local mpk_text, err_text = MPK.read_text_file("__MPK_PATH__")
        if type(mpk_text) ~= "string"
            or not mpk_text:match("^# MPK")
            or not mpk_text:find("map mpk = %[%]")
            or not mpk_text:find('mpk.set%("payload",') then
            TEST_PATH_ERR = "MPK is not written as Visu-like pseudocode: " .. tostring(err_text)
            return
        end

        local loaded, err_loaded = MPK.read("__MPK_PATH__")
        if type(loaded) ~= "table" or loaded.container ~= "pseudocode" then
            TEST_PATH_ERR = "MPK pseudocode round-trip failed: " .. tostring(err_loaded)
            return
        end

        local rejected_json = MPK.write("__MPK_PATH__", pkg, { json = true })
        if rejected_json ~= nil then
            TEST_PATH_ERR = "MPK JSON output should be rejected"
            return
        end

        local compiled_path = "__MPK_PATH__.bin"
        local ok_compile, err_compile = MPK.compile("__MPK_PATH__", compiled_path)
        if not ok_compile then
            TEST_PATH_ERR = "MPK pseudocode compilation failed: " .. tostring(err_compile)
            return
        end

        local compiled_raw = MPK.read_text_file(compiled_path)
        if type(compiled_raw) ~= "string"
            or compiled_raw:sub(1, 4) ~= "MPKB"
            or compiled_raw:byte(5) ~= 4
            or compiled_raw:find("PSC3", 1, true)
            or compiled_raw:find("# MPK", 1, true) then
            TEST_PATH_ERR = "compiled MPK is not opaque binary-v4"
            return
        end
        if #compiled_raw >= #mpk_text then
            TEST_PATH_ERR = "compiled MPK should be smaller than pseudocode source"
            return
        end

        local compiled, err_compiled = MPK.read(compiled_path)
        if type(compiled) ~= "table" or compiled.container ~= "binary" then
            TEST_PATH_ERR = "compiled MPK round-trip failed: " .. tostring(err_compiled)
            return
        end

        local ok_create = Mimir.Model.create("__MPK_PATH__")
        if ok_create ~= true then
            TEST_PATH_ERR = "Mimir.Model.create(path) failed"
            return
        end

        local ok_create_compiled = Mimir.Model.create(compiled_path)
        if ok_create_compiled ~= true then
            TEST_PATH_ERR = "Mimir.Model.create(compiled path) failed"
            return
        end

        TEST_PATH_OK = true
    )";

    std::string mpk_code_filled = mpk_code;
    const std::string mpk_path_str = mpk_path.string();
    const std::string placeholder = "__MPK_PATH__";
    size_t pos = 0;
    while ((pos = mpk_code_filled.find(placeholder, pos)) != std::string::npos) {
        mpk_code_filled.replace(pos, placeholder.size(), mpk_path_str);
        pos += mpk_path_str.size();
    }

    TASSERT_TRUE(lua.executeScript(mpk_code_filled));
    TASSERT_TRUE(lua.getBoolean("TEST_PATH_OK"));

    LuaContext::getInstance().resetRuntimeState();

    return 0;
}
