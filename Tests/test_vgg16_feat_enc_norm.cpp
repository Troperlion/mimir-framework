#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "include/json.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>

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

    // Repeated real forward/backward/update exercises scratch lifetimes, planner
    // recalibration and optimizer state, rather than just architecture creation.
    const int steps = std::getenv("MIMIR_VGG_STRESS_STEPS")
        ? std::max(1, std::atoi(std::getenv("MIMIR_VGG_STRESS_STEPS"))) : 3;
    const int side = std::getenv("MIMIR_VGG_STRESS_SIDE")
        ? std::clamp(std::atoi(std::getenv("MIMIR_VGG_STRESS_SIDE")), 32, 512) : 32;
    for (const char* norm : {"lineargroup", "groupnorm"}) {
        json cfg = base;
        cfg["enc_norm"] = norm;
        cfg["image_w"] = side;
        cfg["image_h"] = side;
        if (const char* channels = std::getenv("MIMIR_VGG_STRESS_CHANNELS"))
            cfg["base_channels"] = std::clamp(std::atoi(channels), 4, 64);
        auto model = ModelArchitectures::create("vgg16_feat", cfg);
        model->allocateParams();
        model->initializeWeights("xavier", 42);
        Optimizer optimizer;
        optimizer.type = OptimizerType::ADAMW;
        if (const char* kind = std::getenv("MIMIR_VGG_STRESS_OPTIMIZER"))
            optimizer.type = static_cast<OptimizerType>(std::clamp(std::atoi(kind), 0, 8));
        std::vector<float> input(side * side * 3);
        for (int step = 0; step < steps; ++step) {
            for (size_t i = 0; i < input.size(); ++i)
                input[i] = std::sin(float(i + step) * .031f);
            model->zeroGradients();
            const auto& prediction = model->forwardPassView(input, true);
            TASSERT_TRUE(!prediction.empty());
            std::vector<float> gradient(prediction.size());
            for (size_t i = 0; i < prediction.size(); ++i) {
                TASSERT_TRUE(std::isfinite(prediction[i]));
                gradient[i] = 2.f * (prediction[i] - .1f) / float(prediction.size());
            }
            model->backwardPass(gradient);
            model->optimizerStep(optimizer, 1e-5f);
            for (const auto& layer : model->getLayers())
                for (size_t i = 0; i < layer.getWeightsSize(); ++i)
                    TASSERT_TRUE(std::isfinite(layer.getWeights()[i]));
        }
    }

    return 0;
}
