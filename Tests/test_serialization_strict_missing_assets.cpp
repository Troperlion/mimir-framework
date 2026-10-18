#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

int main() {
    using namespace Mimir::Serialization;

    json cfg = {
        {"input_dim", 4},
        {"hidden_dim", 8},
        {"output_dim", 2},
        {"hidden_layers", 1},
        {"dropout", 0.0}
    };

    // 1) Intentionally absent assets are allowed; legacy/undeclared missing assets still fail.
    {
        auto modelA = ModelArchitectures::create("basic_mlp", cfg);
        TASSERT_TRUE(modelA != nullptr);
        modelA->allocateParams();
        modelA->initializeWeights("xavier", 123u);

        const std::filesystem::path tmp = std::filesystem::temp_directory_path();
        const std::filesystem::path p = tmp / "mimir_test_missing_tokenizer.safetensors";

        SaveOptions sopts;
        sopts.format = CheckpointFormat::SafeTensors;
        sopts.save_tokenizer = false;
        sopts.save_encoder = false;
        sopts.save_optimizer = false;

        std::string err;
        TASSERT_TRUE(save_checkpoint(*modelA, p.string(), sopts, &err));

        auto modelB = ModelArchitectures::create("basic_mlp", cfg);
        TASSERT_TRUE(modelB != nullptr);
        modelB->allocateParams();
        modelB->initializeWeights("xavier", 999u);

        LoadOptions lopts;
        lopts.format = CheckpointFormat::SafeTensors;
        lopts.strict_mode = true;
        lopts.load_tokenizer = true;
        lopts.load_encoder = false;
        lopts.load_optimizer = false;

        err.clear();
        TASSERT_TRUE(load_checkpoint(*modelB, p.string(), lopts, &err));
        TASSERT_TRUE(!modelB->getHasTokenizer());

        // Remove the new declaration to emulate the legacy strict contract.
        std::ifstream input(p, std::ios::binary);
        uint64_t length = 0;
        input.read(reinterpret_cast<char*>(&length), 8);
        std::string text(length, ' '); input.read(text.data(), length);
        auto header = json::parse(text);
        std::string payload((std::istreambuf_iterator<char>(input)), {});
        input.close();
        header["__metadata__"].erase("components");
        text = header.dump();
        text.append((8 - text.size() % 8) % 8, ' ');
        length = text.size();
        std::ofstream output(p, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(&length), 8);
        output.write(text.data(), text.size());
        output.write(payload.data(), payload.size()); output.close();
        err.clear();
        TASSERT_TRUE(!load_checkpoint(*modelB, p.string(), lopts, &err));
        TASSERT_TRUE(!err.empty());

        std::filesystem::remove(p);
    }

    // 2) RawFolder distinguishes a disabled encoder from a declared but missing one.
    {
        auto modelA = ModelArchitectures::create("basic_mlp", cfg);
        TASSERT_TRUE(modelA != nullptr);
        modelA->allocateParams();
        modelA->initializeWeights("xavier", 321u);

        // Save a tokenizer but not an encoder.
        Tokenizer tok(64);
        tok.setMaxSequenceLength(8);
        tok.tokenizeEnsure("alpha beta");
        modelA->setTokenizer(tok);

        const std::filesystem::path tmp = std::filesystem::temp_directory_path();
        const std::filesystem::path dir = tmp / "mimir_test_missing_encoder";

        std::error_code ec;
        std::filesystem::remove_all(dir, ec);

        SaveOptions sopts;
        sopts.format = CheckpointFormat::RawFolder;
        sopts.save_tokenizer = true;
        sopts.save_encoder = false;
        sopts.save_optimizer = false;

        std::string err;
        TASSERT_TRUE(save_checkpoint(*modelA, dir.string(), sopts, &err));

        auto modelB = ModelArchitectures::create("basic_mlp", cfg);
        TASSERT_TRUE(modelB != nullptr);
        modelB->allocateParams();
        modelB->initializeWeights("xavier", 111u);

        LoadOptions lopts;
        lopts.format = CheckpointFormat::RawFolder;
        lopts.strict_mode = true;
        lopts.load_tokenizer = true;
        lopts.load_encoder = true;
        lopts.load_optimizer = false;

        err.clear();
        TASSERT_TRUE(load_checkpoint(*modelB, dir.string(), lopts, &err));
        TASSERT_TRUE(!modelB->getHasEncoder());
        json manifest;
        { std::ifstream input(dir / "manifest.json"); input >> manifest; }
        manifest["components"]["encoder"] = true;
        { std::ofstream output(dir / "manifest.json"); output << manifest; }
        err.clear();
        TASSERT_TRUE(!load_checkpoint(*modelB, dir.string(), lopts, &err));
        TASSERT_TRUE(!err.empty());

        std::filesystem::remove_all(dir, ec);
    }

    return 0;
}
