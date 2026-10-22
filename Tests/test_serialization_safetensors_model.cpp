#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"
#include "DType.hpp"

#include <algorithm>
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

static uint64_t read_u64_le(std::ifstream& f) {
    uint8_t bytes[8];
    f.read(reinterpret_cast<char*>(bytes), 8);
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= (static_cast<uint64_t>(bytes[i]) << (i * 8));
    }
    return value;
}

static json read_safetensors_header_json(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    TASSERT_TRUE((bool)f);
    const uint64_t header_len = read_u64_le(f);
    TASSERT_TRUE(header_len > 0);
    std::string header;
    header.resize(static_cast<size_t>(header_len));
    f.read(header.data(), static_cast<std::streamsize>(header.size()));
    TASSERT_TRUE((bool)f);
    return json::parse(header);
}

int main() {
    using namespace Mimir::Serialization;

    json cfg = {
        {"input_dim", 4},
        {"hidden_dim", 8},
        {"output_dim", 2},
        {"hidden_layers", 1},
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

    for (const auto& c : cases) {
        json cfgA = cfg;
        cfgA["dtype"] = c.dtype;

        auto modelA = ModelArchitectures::create("basic_mlp", cfgA);
        TASSERT_TRUE(modelA != nullptr);
        TASSERT_TRUE(Mimir::parse_dtype(modelA->getDefaultDType()) == Mimir::parse_dtype(c.dtype));
        modelA->allocateParams();
        modelA->initializeWeights("xavier", 123u);

        // Attach deterministic-ish tokenizer/encoder state (ensures JSON + embeddings tensors exist).
        Tokenizer tok(128);
        tok.setMaxSequenceLength(16);
        tok.tokenizeEnsure("hello world");

        ConditioningEncoder enc(8, 128);
        enc.ensureVocabSize(tok.getVocabSize(), 777u);
        enc.ensureSpecialEmbeddings(42u);

        modelA->setTokenizer(tok);
        modelA->setEncoder(enc);
        modelA->setHasEncoder(true);

        if (std::string(c.expected_tag) == "F32") {
            Optimizer optimizer;
            optimizer.type = OptimizerType::LAMB;
            optimizer.step = 23;
            optimizer.weight_decay = 0.02f;
            size_t moment_count = 0;
            for (const auto& layer : modelA->getLayers()) moment_count += layer.getWeightsSize();
            optimizer.m.assign(moment_count, 0.1f);
            optimizer.v.assign(moment_count, 0.3f);
            optimizer.m[1] = 0.2f;
            optimizer.v[1] = 0.4f;
            modelA->setSerializedOptimizer(std::move(optimizer));
        }

        const std::filesystem::path p = tmp / (std::string("mimir_test_model_") + c.expected_tag + ".safetensors");
        std::filesystem::remove(p);

        SaveOptions sopts;
        sopts.format = CheckpointFormat::SafeTensors;
        sopts.save_tokenizer = true;
        sopts.save_encoder = true;
        sopts.save_optimizer = std::string(c.expected_tag) == "F32";
        sopts.custom_metadata = R"({"run":"dtype-test","seed":123})";

        std::string err;
        TASSERT_TRUE(save_checkpoint(*modelA, p.string(), sopts, &err));

        {
            std::ifstream serialized(p, std::ios::binary);
            TASSERT_TRUE((bool)serialized);
            const std::string bytes((std::istreambuf_iterator<char>(serialized)),
                                    std::istreambuf_iterator<char>());
            TASSERT_TRUE(bytes.find("\"image_width\"") == std::string::npos);
            TASSERT_TRUE(bytes.find("\"image_height\"") == std::string::npos);
        }

        // Header should advertise expected dtype on at least one weights tensor.
        {
            const auto& layers = modelA->getLayers();
            std::string first_weight;
            for (const auto& layer : layers) {
                if (layer.weight_block && layer.weight_block->getSize() > 0) {
                    first_weight = layer.name + "/weights";
                    break;
                }
            }
            TASSERT_TRUE(!first_weight.empty());
            json header = read_safetensors_header_json(p);
            TASSERT_TRUE(header.contains("__metadata__"));
            for (auto it = header["__metadata__"].begin(); it != header["__metadata__"].end(); ++it) {
                TASSERT_TRUE(it.value().is_string());
            }
            TASSERT_TRUE(json::parse(header["__metadata__"]["custom"].get<std::string>())["seed"] == 123);
            TASSERT_TRUE(header.contains(first_weight));
            TASSERT_TRUE(header[first_weight].contains("dtype"));
            const std::string dt = header[first_weight]["dtype"].get<std::string>();
            TASSERT_TRUE(dt == c.expected_tag);

            std::ifstream raw(p, std::ios::binary);
            TASSERT_TRUE((bool)raw);
            const uint64_t padded_header_len = read_u64_le(raw);
            TASSERT_TRUE((padded_header_len % 8) == 0);
        }

        // Load into a fresh model with different init weights, to make sure load overwrites.
        auto modelB = ModelArchitectures::create("basic_mlp", cfg);
        TASSERT_TRUE(modelB != nullptr);
        modelB->allocateParams();
        modelB->initializeWeights("xavier", 999u);

        LoadOptions lopts;
        lopts.format = CheckpointFormat::SafeTensors;
        lopts.strict_mode = true;
        lopts.load_tokenizer = true;
        lopts.load_encoder = true;
        lopts.load_optimizer = std::string(c.expected_tag) == "F32";

        std::string logged_tensor_name;
        for (const auto& layer : modelA->getLayers()) {
            if (layer.weight_block && layer.weight_block->getSize() > 0) {
                logged_tensor_name = layer.name + "/weights";
                break;
            }
        }
        std::ostringstream load_log;
        std::streambuf* previous_cerr = std::cerr.rdbuf(load_log.rdbuf());
        const bool loaded = load_checkpoint(*modelB, p.string(), lopts, &err);
        std::cerr.rdbuf(previous_cerr);
        TASSERT_TRUE(loaded);
        // Hash-tree output splits tensor paths across directory/leaf lines.
        const auto leaf = logged_tensor_name.substr(logged_tensor_name.find_last_of("/.")+1);
        TASSERT_TRUE(load_log.str().find(leaf) != std::string::npos);
        TASSERT_TRUE(load_log.str().find("[checkpoint hashsums]") != std::string::npos);
        TASSERT_TRUE(load_log.str().find(" | ") != std::string::npos);
        TASSERT_TRUE(load_log.str().find("sha256:") != std::string::npos);

        // DType should be restored semantically (aliases allowed: f16/fp16/float16...).
        TASSERT_TRUE(Mimir::parse_dtype(modelB->getDefaultDType()) == Mimir::parse_dtype(c.dtype));

        // Weights should match (within dtype quantization error if any).
        TASSERT_TRUE(compare_model_weights(*modelA, *modelB, c.eps) == 0);

        // Tokenizer/encoder should have been restored.
        TASSERT_TRUE(modelB->getTokenizer().getVocabSize() == modelA->getTokenizer().getVocabSize());
        TASSERT_TRUE(modelB->getEncoder().dim == modelA->getEncoder().dim);
        TASSERT_TRUE(modelB->getEncoder().vocab_size == modelA->getEncoder().vocab_size);
        TASSERT_TRUE(!modelB->getEncoder().token_embeddings.empty());

        if (std::string(c.expected_tag) == "F32") {
            const Optimizer* optimizer = modelB->getSerializedOptimizer();
            TASSERT_TRUE(optimizer != nullptr);
            TASSERT_TRUE(optimizer->type == OptimizerType::LAMB);
            TASSERT_TRUE(optimizer->step == 23);
            TASSERT_NEAR(optimizer->weight_decay, 0.02f, 1e-6f);
            TASSERT_TRUE(optimizer->m.size() == modelA->getSerializedOptimizer()->m.size() && optimizer->v.size() == modelA->getSerializedOptimizer()->v.size());
        }

        std::filesystem::remove(p);
    }

    // Invalid custom JSON must be reported, never silently discarded.
    {
        auto model = ModelArchitectures::create("basic_mlp", cfg);
        TASSERT_TRUE(model != nullptr);
        model->allocateParams();
        const auto p = tmp / "mimir_test_invalid_custom_metadata.safetensors";
        SaveOptions options;
        options.format = CheckpointFormat::SafeTensors;
        options.save_tokenizer = false;
        options.save_encoder = false;
        options.custom_metadata = "{invalid";
        std::string error;
        TASSERT_TRUE(!save_checkpoint(*model, p.string(), options, &error));
        TASSERT_TRUE(!error.empty());
        std::filesystem::remove(p);
    }
    return 0;
}
