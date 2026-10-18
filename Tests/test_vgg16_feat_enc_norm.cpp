#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "include/json.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

static size_t count_type(const Model& model, const std::string& type) {
    size_t count = 0;
    for (const auto& layer : model.getLayers()) {
        if (layer.type == type) ++count;
    }
    return count;
}

int main() {
    json base = ModelArchitectures::defaultConfig("vgg16_feat");
    TASSERT_TRUE(base["enc_norm"].get<std::string>() == "lineargroup");
    base["image_w"] = 8;
    base["image_h"] = 8;
    base["base_channels"] = 4;

    {
        json cfg = base;
        cfg["enc_norm"] = "lineargroup";
        auto model = ModelArchitectures::create("vgg16_feat", cfg);
        TASSERT_TRUE(model != nullptr);
        TASSERT_TRUE(model->modelConfig["enc_norm"].get<std::string>() == "lineargroup");
        TASSERT_TRUE(count_type(*model, "LayerNorm") > 0);
        TASSERT_TRUE(count_type(*model, "GroupNorm") == 0);
    }

    {
        json cfg = base;
        cfg["enc_norm"] = "groupnorm";
        cfg["enc_gn_groups"] = 3;
        auto model = ModelArchitectures::create("vgg16_feat", cfg);
        TASSERT_TRUE(model != nullptr);
        TASSERT_TRUE(model->modelConfig["enc_norm"].get<std::string>() == "groupnorm");
        TASSERT_TRUE(count_type(*model, "GroupNorm") > 0);
        TASSERT_TRUE(count_type(*model, "LayerNorm") == 0);
        for (const auto& layer : model->getLayers()) {
            if (layer.type == "GroupNorm") {
                TASSERT_TRUE(layer.num_groups >= 1);
                TASSERT_TRUE(layer.in_channels % layer.num_groups == 0);
            }
        }

        model->allocateParams();
        model->initializeWeights("xavier", 42);
        const std::vector<float> input(8 * 8 * 3, 0.125f);
        const std::vector<float> output = model->forwardPass(input, true);
        TASSERT_TRUE(!output.empty());
        model->backwardPass(std::vector<float>(output.size(), 1.0f));
    }

    bool rejected = false;
    try {
        json cfg = base;
        cfg["enc_norm"] = "batchnorm";
        (void)ModelArchitectures::create("vgg16_feat", cfg);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    TASSERT_TRUE(rejected);

    return 0;
}
