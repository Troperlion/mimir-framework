#pragma once

#include "../Model.hpp"

class VAETraining {
public:
    Model::TrainStepResult trainImage(Model& model,
                                      const std::vector<float>& input,
                                      const std::vector<float>* target,
                                      const std::vector<float>* loss_weights,
                                      Optimizer& optimizer,
                                      float learning_rate,
                                      Model::TrainStepMode mode,
                                      float grad_scale,
                                     const std::vector<SpatialRelation>* relations = nullptr);

    Model::TrainStepResult trainText(Model& model,
                                     const std::vector<float>& input,
                                     const std::vector<int>& text_ids,
                                     const std::vector<float>* target,
                                     const std::vector<float>* loss_weights,
                                     Optimizer& optimizer,
                                     float learning_rate,
                                     Model::TrainStepMode mode,
                                     float grad_scale,
                                     const std::vector<SpatialRelation>* relations = nullptr);

private:
    std::shared_ptr<Model> perceptual_model_;
    std::vector<float> pending_perceptual_prior_;
};