#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "include/json.hpp"

using json = nlohmann::json;

int main() {
    // 1) Architecture resolved from top-level `architecture`.
    {
        json full = {
            {"architecture", "unet"},
            {"unet", {{"image_w", 32}, {"image_h", 32}, {"image_c", 3}, {"base_channels", 24}, {"depth", 5}}},
            {"tokenizer", {{"max_vocab", 1234}, {"max_sequence_length", 77}}},
            {"training", {{"learning_rate", 1e-4}, {"batch_size", 2}}},
        };

        std::string arch;
        json cfg = ModelArchitectures::cfgFromConfig(full, &arch, "t2i_autoencoder");

        TASSERT_TRUE(arch == "unet");
        TASSERT_TRUE(cfg.is_object());

        // Flattened overrides must be visible at root.
        TASSERT_TRUE(cfg.contains("image_w") && cfg["image_w"].get<int>() == 32);
        TASSERT_TRUE(cfg.contains("image_h") && cfg["image_h"].get<int>() == 32);
        TASSERT_TRUE(cfg.contains("image_c") && cfg["image_c"].get<int>() == 3);
        TASSERT_TRUE(cfg.contains("base_channels") && cfg["base_channels"].get<int>() == 24);
        TASSERT_TRUE(cfg.contains("depth") && cfg["depth"].get<int>() == 5);

        // Parent keys must be preserved in the returned cfg.
        TASSERT_TRUE(cfg.contains("tokenizer") && cfg["tokenizer"].is_object());
        TASSERT_TRUE(cfg["tokenizer"].contains("max_vocab") && cfg["tokenizer"]["max_vocab"].get<int>() == 1234);
        TASSERT_TRUE(cfg.contains("training") && cfg["training"].is_object());
    }

    // 2) Architecture resolved from config.model.type if no top-level architecture.
    {
        json full = {
            {"model", {{"type", "unet"}}},
            {"unet", {{"image_w", 16}, {"image_h", 16}, {"image_c", 1}, {"base_channels", 8}, {"depth", 2}}},
        };

        std::string arch;
        json cfg = ModelArchitectures::cfgFromConfig(full, &arch, "t2i_autoencoder");
        TASSERT_TRUE(arch == "unet");
        TASSERT_TRUE(cfg.contains("base_channels") && cfg["base_channels"].get<int>() == 8);
        TASSERT_TRUE(cfg.contains("depth") && cfg["depth"].get<int>() == 2);
    }

    // 3) Architecture resolved from top-level `type`.
    {
        json full = {
            {"type", "basic_mlp"},
            {"basic_mlp", {{"input_dim", 3}, {"hidden_dim", 7}, {"output_dim", 2}, {"hidden_layers", 2}, {"dropout", 0.0}}},
        };

        std::string arch;
        json cfg = ModelArchitectures::cfgFromConfig(full, &arch, "t2i_autoencoder");
        TASSERT_TRUE(arch == "basic_mlp");
        TASSERT_TRUE(cfg.contains("input_dim") && cfg["input_dim"].get<int>() == 3);
    }

    // 4) Removed architectures are no longer exposed by the registry.
    {
        TASSERT_TRUE(ModelArchitectures::Registry::instance().find("ponyxl_ddpm") == nullptr);
    }

    return 0;
}
