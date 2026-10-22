#include "test_utils.hpp"

#include "Model.hpp"
#include "Models/MLP/BasicMLPModel.hpp"
#include "Models/Vision/UNetModel.hpp"

#include <vector>

int main() {
    Model base_model;
    Model::TrainStepRequest empty_request;
    TASSERT_TRUE(!base_model.trainStep(empty_request).has_value());

    BasicMLPModel model;
    BasicMLPModel::Config config;
    config.input_dim = 4;
    config.hidden_dim = 8;
    config.output_dim = 2;
    config.hidden_layers = 1;
    model.buildFromConfig(config);
    model.allocateParams();
    model.initializeWeights("xavier", 123u);

    std::vector<float> input = {0.5f, -1.0f, 0.25f, 2.0f};
    std::vector<float> target = {0.0f, 1.0f};
    Optimizer optimizer;
    optimizer.type = OptimizerType::SGD;
    optimizer.decay_strategy = LRDecayStrategy::NONE;

    Model::TrainStepRequest request;
    request.float_inputs["__input__"] = &input;
    request.target = &target;
    request.optimizer = &optimizer;
    request.learning_rate = 1e-2f;
    request.mode = Model::TrainStepMode::Accumulate;
    request.grad_scale = 0.5f;

    const auto accumulated = model.trainStep(request);
    TASSERT_TRUE(accumulated.has_value());
    TASSERT_TRUE(accumulated->loss >= 0.0f);
    TASSERT_TRUE(accumulated->metrics.count("mse") == 1);
    TASSERT_TRUE(optimizer.step == 0);

    request.mode = Model::TrainStepMode::Optimize;
    const auto optimized = model.trainStep(request);
    TASSERT_TRUE(optimized.has_value());
    TASSERT_TRUE(optimizer.step == 1);

    Model::TrainStepRequest missing_target = request;
    missing_target.target = nullptr;
    TASSERT_TRUE(!model.trainStep(missing_target).has_value());

    UNetModel vision_model;
    UNetModel::Config vision_config;
    vision_config.image_w = 2;
    vision_config.image_h = 2;
    vision_config.image_c = 1;
    vision_config.base_channels = 4;
    vision_config.depth = 1;
    vision_model.buildFromConfig(vision_config);
    vision_model.allocateParams();
    vision_model.initializeWeights("xavier", 321u);

    std::vector<float> image = {0.1f, 0.2f, 0.3f, 0.4f};
    std::vector<float> image_target = {0.0f, 0.0f, 3.0f, 3.0f};
    std::vector<float> image_mask = {0.4f, 0.4f, 1.6f, 1.6f};
    Model::TrainStepRequest vision_request;
    vision_request.float_inputs["__input__"] = &image;
    vision_request.target = &image_target;
    vision_request.loss_weights = &image_mask;
    vision_request.optimizer = &optimizer;
    vision_request.mode = Model::TrainStepMode::Accumulate;

    Model& generic_vision_model = vision_model;
    vision_request.loss_weights = nullptr;
    const auto unmasked_vision_step = generic_vision_model.trainStep(vision_request);
    TASSERT_TRUE(unmasked_vision_step.has_value());

    vision_request.loss_weights = &image_mask;
    const auto masked_vision_step = generic_vision_model.trainStep(vision_request);
    TASSERT_TRUE(masked_vision_step.has_value());
    TASSERT_TRUE(masked_vision_step->metrics.count("mse") == 1);
    TASSERT_TRUE(masked_vision_step->loss >= 0.0f);
    TASSERT_TRUE(!nearf(masked_vision_step->loss, unmasked_vision_step->loss));

    std::vector<float> invalid_mask = {1.0f, 1.0f, 1.0f};
    vision_request.loss_weights = &invalid_mask;
    bool invalid_mask_rejected = false;
    try {
        (void)generic_vision_model.trainStep(vision_request);
    } catch (const std::invalid_argument&) {
        invalid_mask_rejected = true;
    }
    TASSERT_TRUE(invalid_mask_rejected);
    return 0;
}