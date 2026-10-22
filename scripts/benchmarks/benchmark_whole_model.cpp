// Bounded whole-model latency probe, using the unchanged public training API.
// cmake --build build --target mimir_bench_whole_model
// CPU example (set these environment variables before running):
// OMP_NUM_THREADS=1 MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1
// ./build/Tests/mimir_bench_whole_model float32 200 2
// Optional trailing arguments: features (64), batch (16), Linear/SiLU pairs (8).
// Optimizer IDs: 0 SGD, 1 Adam, 2 AdamW, 3 Lion, 4 Adafactor,
// 5 RAdam, 6 NAdam, 7 RMSProp, 8 LAMB (default: SGD).
#include "Model.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const std::string dtype = argc > 1 ? argv[1] : "float32";
        const int steps = argc > 2 ? std::stoi(argv[2]) : 200;
        if (steps < 1 || steps > 10000) throw std::invalid_argument("steps must be in [1,10000]");
        const int features=argc>4?std::stoi(argv[4]):64;
        const int batch=argc>5?std::stoi(argv[5]):16;
        const int pairs=argc>6?std::stoi(argv[6]):8;
        if(features<1 || features>1024 || batch<1 || batch>1024 || pairs<1 || pairs>32)
            throw std::invalid_argument("features/batch must be in [1,1024], pairs in [1,32]");
        Model::setFrameworkLogsSuppressed(true);
        Model model;
        model.setDefaultDType(dtype);
        model.modelConfig = {{"loss", "mse"}};
        for (int i = 0; i < pairs; ++i) {
            model.push("fc" + std::to_string(i), "Linear", features*features+features);
            auto& layer = model.getMutableLayers().back();
            layer.in_features = layer.out_features = features;
            layer.seq_len = batch;
            layer.use_bias = true;
            model.push("act" + std::to_string(i), "SiLU", 0);
        }
        model.getMutableLayers()[0].inputs = {"__input__"};
        model.allocateParams();
        for (auto& layer : model.getMutableLayers()) {
            auto* weights = layer.getWeights();
            if (weights) for (size_t i = 0; i < layer.getWeightsSize(); ++i)
                weights[i] = float(int(i % 19)-9)*0.002f;
        }
        std::vector<float> input(batch*features, 0.3f), target(input.size(), 0.2f);
        Optimizer optimizer;
        const int optimizer_kind = argc > 3 ? std::stoi(argv[3]) : 0;
        if (optimizer_kind < 0 || optimizer_kind > 8) throw std::invalid_argument("optimizer must be in [0,8]");
        optimizer.type = static_cast<OptimizerType>(optimizer_kind);
        optimizer.weight_decay = 0;
        Model::TrainStepRequest request;
        request.float_inputs["__input__"] = &input;
        request.target = &target;
        request.optimizer = &optimizer;
        request.learning_rate = 0.001f;
        request.capture_prediction = false;
        auto train = [&]() {
            auto result = model.trainStep(request);
            if (!result || !std::isfinite(result->loss)) throw std::runtime_error("Invalid training result");
            return result->loss;
        };
        using Clock = std::chrono::steady_clock;
        const auto cold_start=Clock::now();train();
        const double cold_trainstep_ms=std::chrono::duration<double,std::milli>(Clock::now()-cold_start).count();
        for (int i = 1; i < 10; ++i) train();
        auto elapsed = [](Clock::time_point start) {
            return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        };
        auto measure = [&](auto operation) {
            std::vector<double> samples;
            samples.reserve(steps);
            for (int i = 0; i < steps; ++i) {
                auto start = Clock::now(); operation(); samples.push_back(elapsed(start));
            }
            const auto raw_samples=samples;
            std::sort(samples.begin(), samples.end());
            return json{{"median_ms",samples[samples.size()/2]},
                        {"p95_ms",samples[std::min(samples.size()-1,samples.size()*95/100)]},{"samples_ms",raw_samples}};
        };
        float loss = 0;
        const auto training = measure([&]() { loss = train(); });
        const auto inference = measure([&]() { (void)model.forwardPass(input, false); });
        double checksum = 0;
        for (const auto& layer : model.getLayers())
            if (layer.getWeights()) for (size_t i = 0; i < layer.getWeightsSize(); ++i)
                checksum += double(i+1)*layer.getWeights()[i];
        std::cout << json{{"dtype",dtype},{"steps",steps},{"layers",pairs*2},{"batch",batch},{"features",features},
            {"cold_trainstep_ms",cold_trainstep_ms},{"training",training},{"inference",inference},{"final_loss",loss},
            {"parameter_checksum",checksum},
            {"optimizer_kind",optimizer_kind},{"optimizer_backends",RuntimeRouter::instance().optimizerDispatchCounts()},
            {"loss_backends",RuntimeRouter::instance().reconstructionLossDispatchCounts()}}.dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
