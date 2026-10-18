#pragma once

#include "../Model.hpp"

#include <string>
#include <vector>

// BasicMLPModel:
// - Modèle "de base" pour régression: y = MLP(x)
// - I/O: input = vecteur float[input_dim] -> output = float[output_dim]
// - Entraînement: MSE + backwardPass + optimizerStep

class BasicMLPModel : public Model {
public:
    struct Config {
        int input_dim = 256;
        int hidden_dim = 256;
        int output_dim = 256;
        int hidden_layers = 2;   // nombre de blocs (Linear->GELU->[Dropout])
        float dropout = 0.0f;    // dropout p (0..1)
    };

    struct StepStats {
        float loss = 0.0f;
        float grad_norm = 0.0f;
        float grad_max_abs = 0.0f;
    };

    BasicMLPModel();

    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    StepStats trainStep(const std::vector<float>& input,
                        const std::vector<float>& target,
                        Optimizer& opt,
                        float learning_rate);

    std::optional<TrainStepResult> trainStep(const TrainStepRequest& request) override;

    static void buildInto(Model& model, const Config& cfg);

private:
    Config cfg_;
};
