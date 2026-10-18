#include "test_utils.hpp"
#include "Model.hpp"
#include "Serialization/Serialization.hpp"
#include "Serialization/CheckpointState.hpp"
#include <fstream>
#include <chrono>
#include <cstring>

using namespace Mimir::Serialization;

namespace {
json read_json_file(const fs::path& path) {
    std::ifstream input(path);
    json value; input >> value;
    return value;
}
json read_header(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    uint64_t size = 0;
    input.read(reinterpret_cast<char*>(&size), 8);
    std::string data(size, ' ');
    input.read(data.data(), size);
    return json::parse(data);
}
void build(Model& model, bool reverse = false) {
    model.setHasEncoder(false);
    model.setHasTokenizer(false);
    model.modelConfig["frozen_layer_prefixes"] = {"frozen"};
    model.modelConfig["training_state"] = {{"global_step", 2}, {"next_item", 3}, {"seed", 42}};
    std::vector<std::string> names{"q", "k", "v", "frozen"};
    if (reverse) std::reverse(names.begin(), names.end());
    for (const auto& name : names) {
        model.push(name, "Constant", 3);
        model.getLayerByName(name)->trainable_parameter = true;
    }
    model.allocateParams();
    for (auto& layer : model.getMutableLayers()) {
        std::copy_n(std::vector<float>{1.0f, -2.0f, 0.5f}.begin(), 3, layer.getWeights());
        layer.grad_weights = {0.2f, -0.4f, 0.1f};
    }
}
void step(Model& model, float scale = 1.0f) {
    for (auto& layer : model.getMutableLayers()) layer.grad_weights = {0.2f * scale, -0.4f * scale, 0.1f * scale};
    model.optimizerStep(*model.getMutableSerializedOptimizer(), 0.01f);
}
void require(bool ok, const std::string& error) {
    if (!ok) throw std::runtime_error(error);
}
}

int main() {
    const auto root = fs::temp_directory_path() / ("mimir-active-state-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    std::string error;
    for (int type = 0; type <= static_cast<int>(OptimizerType::LAMB); ++type) {
        for (auto format : {CheckpointFormat::RawFolder, CheckpointFormat::SafeTensors}) {
            Model model; build(model);
            Optimizer optimizer;
            optimizer.type = static_cast<OptimizerType>(type);
            optimizer.initial_lr = 0.01f;
            optimizer.warmup_steps = 3;
            optimizer.decay_strategy = LRDecayStrategy::COSINE;
            optimizer.beta1 = 0.85f;
            optimizer.beta2 = 0.95f;
            optimizer.weight_decay = 0.03f;
            if (type == 4 && format == CheckpointFormat::SafeTensors) optimizer.adafactor_beta1 = 0.8f;
            model.setSerializedOptimizer(optimizer);
            step(model); step(model, -0.5f);
            const auto original = model.optimizerSnapshot(*model.getSerializedOptimizer());
            const auto path = root / (std::to_string(type) + (format == CheckpointFormat::RawFolder ? "-raw" : ".safetensors"));
            SaveOptions save; save.format = format; save.save_optimizer = true; save.include_git_info = false;
            require(save_checkpoint(model, path.string(), save, &error), error);
            // Saving may not discard the live moments used by the next update.
            if (type != 0) TASSERT_TRUE(model.getSerializedOptimizer()->mv_by_param_ptr.size() == 3);
            if (format == CheckpointFormat::RawFolder) {
                TASSERT_TRUE(!fs::exists(path / "encoder"));
                TASSERT_TRUE(!fs::exists(path / "tokenizer"));
                TASSERT_TRUE(!fs::exists(path / "dataset"));
                const auto training = read_json_file(path / "model/training.json");
                TASSERT_TRUE(training["step"] == 2);
                TASSERT_TRUE(training["loop"]["next_item"] == 3);
                TASSERT_TRUE(training["state_sizes"].contains("m") == optimizer.usesFirstMoment());
                TASSERT_TRUE(training["state_sizes"].contains("v") == optimizer.usesSecondMoment());
                TASSERT_TRUE(training["parameter_layout"].size() == (type == 0 ? 0 : 3));
                if (optimizer.usesFirstMoment()) TASSERT_TRUE(read_json_file(path / "tensors/optimizer/m.json")["dtype"] == "F32");
            } else {
                const auto header = read_header(path);
                TASSERT_TRUE(!header.contains("encoder/json"));
                TASSERT_TRUE(!header.contains("encoder/token_embeddings"));
                TASSERT_TRUE(!header.contains("tokenizer/json"));
                TASSERT_TRUE(header.contains("optimizer/m") == optimizer.usesFirstMoment());
                TASSERT_TRUE(header.contains("optimizer/v") == optimizer.usesSecondMoment());
            }
            Model resumed; build(resumed, true); // Named moments must survive graph reordering.
            LoadOptions load; load.format = format; load.load_optimizer = true;
            require(load_checkpoint(resumed, path.string(), load, &error), error);
            const auto* restored = resumed.getSerializedOptimizer();
            TASSERT_TRUE(restored && restored->step == 2);
            TASSERT_TRUE(restored->m == original.m && restored->v == original.v);
            TASSERT_TRUE(resumed.modelConfig["training_state"]["next_item"] == 3);
            step(model, 1.5f); step(resumed, 1.5f);
            for (const auto& layer : model.getLayers()) {
                const auto* other = resumed.getLayerByName(layer.name);
                for (size_t i = 0; i < 3; ++i) TASSERT_NEAR(layer.getWeights()[i], other->getWeights()[i], 1e-7f);
            }
            // Enhanced debug output uses the current runtime moments too.
            save.format = CheckpointFormat::DebugJson;
            const auto debug = root / "debug.json";
            require(save_checkpoint(model, debug.string(), save, &error), error);
            const auto info = read_json_file(debug);
            TASSERT_TRUE(!info.contains("encoder") && !info.contains("tokenizer"));
            TASSERT_TRUE(info["optimizer"]["step"] == 3);
            TASSERT_TRUE(info["optimizer"].contains("m") == optimizer.usesFirstMoment());
            TASSERT_TRUE(info["optimizer"].contains("v") == optimizer.usesSecondMoment());
        }
    }
    // F16 weights may not downcast F32 optimizer history.
    {
        Model model; build(model);
        model.setDefaultDType("float16");
        Optimizer optimizer; optimizer.type = OptimizerType::ADAMW;
        model.setSerializedOptimizer(optimizer); step(model);
        for (auto format : {CheckpointFormat::RawFolder, CheckpointFormat::SafeTensors}) {
            SaveOptions save; save.format = format; save.save_optimizer = true; save.include_git_info = false;
            const auto path = root / (format == CheckpointFormat::RawFolder ? "half" : "half.safetensors");
            require(save_checkpoint(model, path.string(), save, &error), error);
            if (format == CheckpointFormat::RawFolder) {
                TASSERT_TRUE(read_json_file(path / "tensors/optimizer/m.json")["dtype"] == "F32");
                // Missing state tensors are an error, even if removed from the manifest.
                auto manifest = read_json_file(path / "manifest.json");
                auto& index = manifest["tensor_index"];
                index.erase(std::remove_if(index.begin(), index.end(), [](const json& j) { return j["name"] == "optimizer/m"; }), index.end());
                std::ofstream(path / "manifest.json") << manifest;
                Model resumed; build(resumed);
                LoadOptions load; load.format = format; load.load_optimizer = true;
                TASSERT_TRUE(!load_checkpoint(resumed, path.string(), load, &error));
            } else {
                TASSERT_TRUE(read_header(path)["optimizer/m"]["dtype"] == "F32");
            }
        }
    }
    // Reusing a folder must remove components previously saved, including tensors.
    {
        Model model; build(model);
        model.setHasEncoder(true); model.setHasTokenizer(true);
        const auto path = root / "replace";
        SaveOptions save; save.format = CheckpointFormat::RawFolder; save.include_git_info = false;
        require(save_checkpoint(model, path.string(), save, &error), error);
        TASSERT_TRUE(fs::exists(path / "encoder/encoder.json"));
        TASSERT_TRUE(fs::exists(path / "tokenizer/tokenizer.json"));
        model.modelConfig["encoder"] = false;
        model.modelConfig["tokenizer"] = false;
        require(save_checkpoint(model, path.string(), save, &error), error);
        TASSERT_TRUE(!fs::exists(path / "encoder"));
        TASSERT_TRUE(!fs::exists(path / "tokenizer"));
        TASSERT_TRUE(!fs::exists(path / "tensors/encoder_token_embeddings.bin"));
        TASSERT_TRUE(read_json_file(path / "manifest.json")["components"]["encoder"] == false);
        LoadOptions load; load.format = CheckpointFormat::RawFolder;
        Model resumed; build(resumed);
        require(load_checkpoint(resumed, path.string(), load, &error), error);
        TASSERT_TRUE(!resumed.getHasEncoder());
        // No optimizer placeholder in an untrained model's debug dump.
        save.format = CheckpointFormat::DebugJson; save.include_optimizer_state = true;
        require(save_checkpoint(model, (root / "empty-debug.json").string(), save, &error), error);
        TASSERT_TRUE(!read_json_file(root / "empty-debug.json").contains("optimizer"));
    }
    // Legacy flat moments are still imported in graph order.
    {
        Model model; build(model);
        Optimizer legacy;
        legacy.type = OptimizerType::ADAM;
        legacy.m.assign(12, 0.2f); legacy.v.assign(12, 0.4f);
        model.restoreOptimizerState(legacy);
        TASSERT_TRUE(legacy.mv_by_param_ptr.size() == 4);
        TASSERT_TRUE(legacy.m.empty() && legacy.v.empty());
    }
    fs::remove_all(root);
    return 0;
}
