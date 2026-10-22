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
// Rewrite only optimizer metadata, retaining all tensor payloads. This builds
// legacy fixtures without invoking today's optimizer-state filtering on save.
json optimizer_json(const fs::path& path, CheckpointFormat format) {
    if (format == CheckpointFormat::RawFolder) return read_json_file(path / "model/training.json");
    const auto header = read_header(path);
    const auto offsets = header.at("optimizer/json").at("data_offsets");
    std::ifstream input(path, std::ios::binary);
    uint64_t header_size=0;input.read(reinterpret_cast<char*>(&header_size),8);
    const size_t begin=offsets[0].get<size_t>(),end=offsets[1].get<size_t>();
    input.seekg(8+header_size+begin);
    std::string data(end-begin,' ');input.read(data.data(),data.size());
    return json::parse(data);
}
void rewrite_optimizer_json(const fs::path& path, CheckpointFormat format, const json& metadata) {
    if (format == CheckpointFormat::RawFolder) {
        std::ofstream(path / "model/training.json") << metadata;
        return;
    }
    auto header=read_header(path);
    std::ifstream input(path,std::ios::binary);
    uint64_t header_size=0;input.read(reinterpret_cast<char*>(&header_size),8);
    input.seekg(8+header_size);
    std::vector<char> payload((std::istreambuf_iterator<char>(input)),{});
    input.close();
    std::vector<char> replacement;
    for (auto& entry:header.items()) {
        if (entry.key()=="__metadata__") continue;
        auto& tensor=entry.value();
        const size_t begin=tensor["data_offsets"][0].get<size_t>();
        const size_t end=tensor["data_offsets"][1].get<size_t>();
        const size_t new_begin=replacement.size();
        if (entry.key()=="optimizer/json") {
            const auto text=metadata.dump();
            replacement.insert(replacement.end(),text.begin(),text.end());
            tensor["shape"]={text.size()};
        } else replacement.insert(replacement.end(),payload.begin()+begin,payload.begin()+end);
        tensor["data_offsets"]={new_begin,replacement.size()};
    }
    auto text=header.dump();text.resize((text.size()+7)/8*8,' ');
    const uint64_t size=text.size();
    std::ofstream output(path,std::ios::binary|std::ios::trunc);
    output.write(reinterpret_cast<const char*>(&size),8);output.write(text.data(),text.size());
    output.write(replacement.data(),replacement.size());
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
                TASSERT_TRUE(training["state_layout"] == "canonical_m_v_v1");
                TASSERT_TRUE(training["state_dtype"] == "F32");
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
            TASSERT_TRUE(info["optimizer"]["type_id"] == type);
            TASSERT_TRUE(info["optimizer"]["state_layout"] == "canonical_m_v_v1");
            TASSERT_TRUE(info["optimizer"]["active_moments"] == optimizer_metadata(original)["active_moments"]);
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
    // Legacy dense checkpoints carried both m and v for every optimizer, with
    // no runtime layout metadata. They must resume through the current compact
    // implementation and re-save only the states the optimizer actually uses.
    for (int kind=0;kind<=8;++kind) for (auto format:{CheckpointFormat::RawFolder,CheckpointFormat::SafeTensors})
        for (bool schema2:{false,true}) {
            Model source;build(source);
            Optimizer dense;
            dense.type=OptimizerType::ADAM;dense.step=7;
            dense.m.assign(12,.2f);dense.v.assign(12,.4f);
            source.setSerializedOptimizer(dense);
            const auto path=root/(std::string("legacy-")+std::to_string(kind)+(schema2?"-v2":"-v1")+
                (format==CheckpointFormat::RawFolder?"-raw":".safetensors"));
            SaveOptions save;save.format=format;save.save_optimizer=true;save.include_git_info=false;
            require(save_checkpoint(source,path.string(),save,&error),error);
            auto metadata=optimizer_json(path,format);
            for (const char* key:{"state_layout","state_dtype","active_moments","parameter_layout"}) metadata.erase(key);
            if (!schema2) {metadata.erase("schema_version");metadata.erase("state_sizes");}
            metadata["type"]=schema2?json(optimizerTypeName(static_cast<OptimizerType>(kind))):json(kind);
            metadata["name"]=optimizerTypeName(static_cast<OptimizerType>(kind));
            metadata["adafactor_beta1"]=schema2?.7f:0.f;
            rewrite_optimizer_json(path,format,metadata);
            Model expected;build(expected);
            Optimizer legacy=optimizer_from_metadata(metadata);
            legacy.m=dense.m;legacy.v=dense.v;
            expected.setSerializedOptimizer(legacy);
            Model resumed;build(resumed);
            LoadOptions load;load.format=format;load.load_optimizer=true;
            require(load_checkpoint(resumed,path.string(),load,&error),error);
            TASSERT_TRUE(resumed.getSerializedOptimizer()->step==7);
            TASSERT_TRUE(static_cast<int>(resumed.getSerializedOptimizer()->type)==kind);
            step(expected,-.75f);step(resumed,-.75f);
            for (const auto& layer:expected.getLayers()) {
                const auto* actual=resumed.getLayerByName(layer.name);
                for (size_t i=0;i<layer.getWeightsSize();++i)
                    TASSERT_NEAR(layer.getWeights()[i],actual->getWeights()[i],1e-7f);
            }
            const auto upgraded=root/("upgraded-"+std::to_string(kind)+(format==CheckpointFormat::RawFolder?"-raw":".safetensors"));
            require(save_checkpoint(resumed,upgraded.string(),save,&error),error);
            const auto upgraded_metadata=optimizer_json(upgraded,format);
            const auto* optimizer=resumed.getSerializedOptimizer();
            TASSERT_TRUE(upgraded_metadata["state_sizes"].contains("m")==optimizer->usesFirstMoment());
            TASSERT_TRUE(upgraded_metadata["state_sizes"].contains("v")==optimizer->usesSecondMoment());
            TASSERT_TRUE(upgraded_metadata["state_layout"]=="canonical_m_v_v1");
            Model reloaded;build(reloaded);
            require(load_checkpoint(reloaded,upgraded.string(),load,&error),error);
            step(resumed,.5f);step(reloaded,.5f);
            for (const auto& layer:resumed.getLayers()) {
                const auto* actual=reloaded.getLayerByName(layer.name);
                for (size_t i=0;i<layer.getWeightsSize();++i)
                    TASSERT_NEAR(layer.getWeights()[i],actual->getWeights()[i],1e-7f);
            }
        }
    // Unknown device layouts and missing active state are never silently reset.
    {
        Model model;build(model);
        Optimizer optimizer;optimizer.type=OptimizerType::ADAM;
        auto metadata=optimizer_metadata(optimizer);
        metadata["state_layout"]="compact_device_scratch";
        bool rejected=false;
        try {validate_optimizer_state(model,optimizer,metadata);} catch(const std::runtime_error&) {rejected=true;}
        TASSERT_TRUE(rejected);
        metadata=optimizer_metadata(optimizer);metadata["active_moments"]={"v"};
        rejected=false;
        try {validate_optimizer_state(model,optimizer,metadata);} catch(const std::runtime_error&) {rejected=true;}
        TASSERT_TRUE(rejected);
    }
    fs::remove_all(root);
    return 0;
}
