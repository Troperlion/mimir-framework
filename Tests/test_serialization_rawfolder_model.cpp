#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"
#include "DType.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

static int compare_model_weights(const Model& a, const Model& b, float eps) {
    const auto& la = a.getLayers();
    const auto& lb = b.getLayers();

    TASSERT_TRUE(la.size() == lb.size());

    for (size_t i = 0; i < la.size(); ++i) {
        const auto& A = la[i];
        const auto& B = lb[i];
        TASSERT_TRUE(A.name == B.name);

        if (!A.weight_block && !B.weight_block) continue;
        TASSERT_TRUE(A.weight_block != nullptr);
        TASSERT_TRUE(B.weight_block != nullptr);

        const size_t nA = A.weight_block->getSize();
        const size_t nB = B.weight_block->getSize();
        TASSERT_TRUE(nA == nB);
        const float* pA = A.weight_block->getData();
        const float* pB = B.weight_block->getData();
        TASSERT_TRUE(pA != nullptr);
        TASSERT_TRUE(pB != nullptr);

        const size_t n = std::min<size_t>(nA, 64);
        for (size_t j = 0; j < n; ++j) {
            TASSERT_NEAR(pA[j], pB[j], eps);
        }
    }

    return 0;
}

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
        {"input_dim", 3},
        {"hidden_dim", 5},
        {"output_dim", 2},
        {"hidden_layers", 2},
        {"dropout", 0.0}
    };

    struct Case {
        const char* dtype;
        const char* expected_tag;
        float eps;
    };
    const Case cases[] = {
        {"float32", "F32", 1e-6f},
        {"f32", "F32", 1e-6f},
        {"float16", "F16", 5e-3f},
        {"f16", "F16", 5e-3f},
        {"fp16", "F16", 5e-3f},
        {"bfloat16", "BF16", 2e-2f},
        {"bf16", "BF16", 2e-2f},
        {"float64", "F64", 1e-6f},
    };

    const std::filesystem::path tmp = std::filesystem::temp_directory_path();
    std::error_code ec;

    for (const auto& c : cases) {
        json cfgA = cfg;
        cfgA["dtype"] = c.dtype;

        auto modelA = ModelArchitectures::create("basic_mlp", cfgA);
        TASSERT_TRUE(modelA != nullptr);
        TASSERT_TRUE(Mimir::parse_dtype(modelA->getDefaultDType()) == Mimir::parse_dtype(c.dtype));
        modelA->allocateParams();
        modelA->initializeWeights("xavier", 321u);

        Tokenizer tok(64);
        tok.setMaxSequenceLength(8);
        tok.tokenizeEnsure("alpha beta");

        ConditioningEncoder enc(8, 64);
        enc.ensureVocabSize(tok.getVocabSize(), 9u);
        enc.ensureSpecialEmbeddings(10u);

        modelA->setTokenizer(tok);
        modelA->setEncoder(enc);
        modelA->setHasEncoder(true);

        if (std::string(c.expected_tag) == "F32") {
            Optimizer optimizer;
            optimizer.type = OptimizerType::ADAFACTOR;
            optimizer.step = 17;
            optimizer.adafactor_clip_threshold = 1.25f;
            optimizer.adafactor_decay_rate = -0.7f;
            optimizer.adafactor_eps2 = 2e-3f;
            optimizer.adafactor_beta1 = 0.8f;
            optimizer.adafactor_scale_parameter = false;
            optimizer.adafactor_relative_step = true;
            size_t moment_count = 0;
            for (const auto& layer : modelA->getLayers()) moment_count += layer.getWeightsSize();
            optimizer.m.assign(moment_count, 0.1f);
            optimizer.v.assign(moment_count, 0.3f);
            optimizer.m[1] = 0.2f;
            optimizer.v[1] = 0.4f;
            modelA->setSerializedOptimizer(std::move(optimizer));
        }

        const std::filesystem::path dir = tmp / (std::string("mimir_test_rawfolder_") + c.expected_tag);
        std::filesystem::remove_all(dir, ec);

        SaveOptions sopts;
        sopts.format = CheckpointFormat::RawFolder;
        sopts.save_tokenizer = true;
        sopts.save_encoder = true;
        sopts.save_optimizer = std::string(c.expected_tag) == "F32";

        std::string err;
        TASSERT_TRUE(save_checkpoint(*modelA, dir.string(), sopts, &err));

        {
            const json architecture = read_json_file(dir / "model" / "architecture.json");
            TASSERT_TRUE(!architecture.contains("image_width"));
            TASSERT_TRUE(!architecture.contains("image_height"));
        }

        // Metadata JSON should advertise expected dtype on at least one weights tensor.
        {
            const auto& layers = modelA->getLayers();
            std::string first_weight;
            for (const auto& layer : layers) {
                if (layer.weight_block && layer.weight_block->getSize() > 0) {
                    first_weight = layer.name + "_weights";
                    break;
                }
            }
            TASSERT_TRUE(!first_weight.empty());
            const std::filesystem::path meta = dir / "tensors" / (first_weight + ".json");
            json tmeta = read_json_file(meta);
            TASSERT_TRUE(tmeta.contains("dtype"));
            TASSERT_TRUE(tmeta["dtype"].get<std::string>() == c.expected_tag);
        }

        auto modelB = ModelArchitectures::create("basic_mlp", cfg);
        TASSERT_TRUE(modelB != nullptr);
        modelB->allocateParams();
        modelB->initializeWeights("xavier", 111u);

        LoadOptions lopts;
        lopts.format = CheckpointFormat::RawFolder;
        lopts.strict_mode = true;
        lopts.load_tokenizer = true;
        lopts.load_encoder = true;
        lopts.load_optimizer = std::string(c.expected_tag) == "F32";

        std::string logged_tensor_name;
        for (const auto& layer : modelA->getLayers()) {
            if (layer.weight_block && layer.weight_block->getSize() > 0) {
                logged_tensor_name = layer.name + "_weights";
                break;
            }
        }
        std::ostringstream load_log;
        std::streambuf* previous_cerr = std::cerr.rdbuf(load_log.rdbuf());
        const bool loaded = load_checkpoint(*modelB, dir.string(), lopts, &err);
        std::cerr.rdbuf(previous_cerr);
        TASSERT_TRUE(loaded);
        TASSERT_TRUE(load_log.str().find(logged_tensor_name) != std::string::npos);
        TASSERT_TRUE(load_log.str().find(" | sha256:") != std::string::npos);

        // DType should be restored semantically (aliases allowed: f16/fp16/float16...).
        TASSERT_TRUE(Mimir::parse_dtype(modelB->getDefaultDType()) == Mimir::parse_dtype(c.dtype));

        // Basic sanity: model config restored and embeddings present.
        TASSERT_TRUE(modelB->modelConfig.contains("type"));
        TASSERT_TRUE(modelB->getEncoder().dim == modelA->getEncoder().dim);
        TASSERT_TRUE(!modelB->getEncoder().token_embeddings.empty());

        if (std::string(c.expected_tag) == "F32") {
            const Optimizer* optimizer = modelB->getSerializedOptimizer();
            TASSERT_TRUE(optimizer != nullptr);
            TASSERT_TRUE(optimizer->type == OptimizerType::ADAFACTOR);
            TASSERT_TRUE(optimizer->step == 17);
            TASSERT_NEAR(optimizer->adafactor_clip_threshold, 1.25f, 1e-6f);
            TASSERT_NEAR(optimizer->adafactor_decay_rate, -0.7f, 1e-6f);
            TASSERT_NEAR(optimizer->adafactor_eps2, 2e-3f, 1e-8f);
            TASSERT_NEAR(optimizer->adafactor_beta1, 0.8f, 1e-6f);
            TASSERT_TRUE(!optimizer->adafactor_scale_parameter);
            TASSERT_TRUE(optimizer->adafactor_relative_step);
            TASSERT_TRUE(optimizer->m.size() == modelA->getSerializedOptimizer()->m.size() && optimizer->v.size() == modelA->getSerializedOptimizer()->v.size());
        }

        // Weights should match (within dtype quantization error if any).
        TASSERT_TRUE(compare_model_weights(*modelA, *modelB, c.eps) == 0);

        std::filesystem::remove_all(dir, ec);
    }
    return 0;
}
