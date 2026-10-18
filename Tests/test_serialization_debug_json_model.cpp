#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"

#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

static json read_json_file(const std::filesystem::path& p) {
    std::ifstream f(p);
    TASSERT_TRUE((bool)f);
    json j;
    f >> j;
    return j;
}

int main() {
    using namespace Mimir::Serialization;

    json cfg = {
        {"input_dim", 4},
        {"hidden_dim", 8},
        {"output_dim", 2},
        {"hidden_layers", 1},
        {"dropout", 0.0},
    };

    auto model = ModelArchitectures::create("basic_mlp", cfg);
    TASSERT_TRUE(model != nullptr);

    model->setDefaultDType("bfloat16");
    model->allocateParams();
    model->initializeWeights("xavier", 123u);
    TASSERT_TRUE(!model->getLayers().empty());
    bool injected_nan = false;
    for (const auto& layer_view : model->getLayers()) {
        Layer* layer = model->getLayerByName(layer_view.name);
        if (layer && layer->getWeights() && layer->getWeightsSize() > 0) {
            layer->getWeights()[0] = std::numeric_limits<float>::quiet_NaN();
            injected_nan = true;
            break;
        }
    }
    TASSERT_TRUE(injected_nan);

    const std::filesystem::path tmp = std::filesystem::temp_directory_path();
    const std::filesystem::path p = tmp / "mimir_test_debug_json_dtype.json";
    std::filesystem::remove(p);

    SaveOptions sopts;
    sopts.format = CheckpointFormat::DebugJson;
    sopts.include_git_info = false;
    sopts.save_tokenizer = false;
    sopts.save_encoder = false;
    sopts.include_optimizer_state = false;
    sopts.include_checksums = false;
    sopts.include_weight_deltas = false;

    std::string err;
    TASSERT_TRUE(save_checkpoint(*model, p.string(), sopts, &err));

    json j = read_json_file(p);

    TASSERT_TRUE(j.contains("format"));
    TASSERT_TRUE(j["format"].get<std::string>() == "mimir_debug_dump");

    TASSERT_TRUE(j.contains("format_version"));
    TASSERT_TRUE(j["format_version"].get<std::string>() == "1.4.0");

    TASSERT_TRUE(j.contains("metadata"));
    TASSERT_TRUE(j["metadata"]["format"].get<std::string>() == "mimir_debug_dump");
    TASSERT_TRUE(j["metadata"]["format_version"].get<std::string>() == "1.4.0");
    TASSERT_TRUE(j["metadata"]["mimir_version"].is_string());
    TASSERT_TRUE(j["metadata"]["created_at_epoch_seconds"].get<long long>() > 0);
    TASSERT_TRUE(j["metadata"]["inspection_only"].get<bool>());

    TASSERT_TRUE(j.contains("model"));
    TASSERT_TRUE(j["model"]["name"].get<std::string>() == model->getModelName());
    TASSERT_TRUE(j["model"]["default_dtype"].get<std::string>() == "bfloat16");
    TASSERT_TRUE(j["model"]["layer_count"].get<size_t>() == model->getLayers().size());
    TASSERT_TRUE(!j["model"].contains("image_width"));
    TASSERT_TRUE(!j["model"].contains("image_height"));

    TASSERT_TRUE(j.contains("export_metrics"));
    const auto& metrics = j["export_metrics"];
    TASSERT_TRUE(metrics["logical_parameter_elements"].get<size_t>() == model->totalParamCount());
    TASSERT_TRUE(metrics["unique_parameter_elements"].get<size_t>() == model->totalParamCount());
    TASSERT_TRUE(metrics["runtime_parameter_bytes"].get<size_t>() == model->totalParamCount() * sizeof(float));
    TASSERT_TRUE(metrics["serialized_parameter_bytes"].get<size_t>() == model->totalParamCount() * 2);
    TASSERT_TRUE(metrics["serialized_dtype"].get<std::string>() == "BF16");
    TASSERT_TRUE(metrics["serialized_element_bytes"].get<size_t>() == 2);
    TASSERT_TRUE(!metrics["counts_include_optimizer"].get<bool>());

    TASSERT_TRUE(j.contains("framework_state"));
    TASSERT_TRUE(j["framework_state"].is_object());
    TASSERT_TRUE(j["framework_state"].contains("runtime"));
    TASSERT_TRUE(j["framework_state"].contains("memory"));
    TASSERT_TRUE(!j["framework_state"]["model"].contains("image_width"));
    TASSERT_TRUE(!j["framework_state"]["model"].contains("image_height"));

    TASSERT_TRUE(j.contains("default_dtype"));
    TASSERT_TRUE(j["default_dtype"].get<std::string>() == "bfloat16");

    TASSERT_TRUE(j.contains("model_config"));
    TASSERT_TRUE(j["model_config"].is_object());
    TASSERT_TRUE(j["model_config"].contains("dtype"));
    TASSERT_TRUE(j["model_config"]["dtype"].get<std::string>() == "bfloat16");

    // Tensor entries should reflect selected serialized dtype too.
    TASSERT_TRUE(j.contains("layers"));
    TASSERT_TRUE(j["layers"].is_array());
    bool saw_tensor_dtype = false;
    bool saw_non_finite_stats = false;
    for (const auto& layer : j["layers"]) {
        if (!layer.is_object() || !layer.contains("tensors") || !layer["tensors"].is_array()) continue;
        for (const auto& t : layer["tensors"]) {
            if (!t.is_object() || !t.contains("dtype")) continue;
            TASSERT_TRUE(t["dtype"].get<std::string>() == "BF16");
            size_t shape_elements = 1;
            for (const auto& dim : t["shape"]) shape_elements *= dim.get<size_t>();
            TASSERT_TRUE(shape_elements == t["total_elements"].get<size_t>());
            if (t.contains("stats") && t["stats"]["nan_elements"].get<size_t>() > 0) {
                TASSERT_TRUE(t["stats"]["has_non_finite"].get<bool>());
                TASSERT_TRUE(t["sample_values"][0].is_null());
                saw_non_finite_stats = true;
            }
            saw_tensor_dtype = true;
        }
    }
    TASSERT_TRUE(saw_tensor_dtype);
    TASSERT_TRUE(saw_non_finite_stats);

    std::filesystem::remove(p);
    return 0;
}
