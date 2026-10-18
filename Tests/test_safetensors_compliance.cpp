#include "test_utils.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

void write_u64_le(std::ofstream& out, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<uint8_t>((value >> (8 * i)) & 0xffu);
    }
    out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

bool write_fixture(
    const std::filesystem::path& path,
    std::string header,
    const std::vector<uint8_t>& data
) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    write_u64_le(out, static_cast<uint64_t>(header.size()));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!data.empty()) {
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size()));
    }
    return static_cast<bool>(out);
}

bool load_fixture(const std::filesystem::path& path, std::string& error) {
    json cfg = {
        {"input_dim", 1},
        {"hidden_dim", 1},
        {"output_dim", 1},
        {"hidden_layers", 0},
        {"dropout", 0.0}
    };
    auto model = ModelArchitectures::create("basic_mlp", cfg);
    if (!model) return false;
    model->allocateParams();

    Mimir::Serialization::LoadOptions options;
    options.format = Mimir::Serialization::CheckpointFormat::SafeTensors;
    options.strict_mode = false;
    options.load_tokenizer = false;
    options.load_encoder = false;
    options.load_optimizer = false;
    error.clear();
    return Mimir::Serialization::load_checkpoint(*model, path.string(), options, &error);
}

} // namespace

int main() {
    const auto root = std::filesystem::temp_directory_path() / "mimir_safetensors_compliance";
    std::filesystem::create_directories(root);
    std::string error;

    struct InvalidCase {
        const char* name;
        std::string header;
        std::vector<uint8_t> data;
    };

    const std::vector<InvalidCase> invalid_cases = {
        {
            "metadata_not_string",
            R"({"__metadata__":{"created_at":123}})",
            {}
        },
        {
            "duplicate_tensor_key",
            R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
            {0}
        },
        {
            "noncanonical_dtype",
            R"({"x":{"dtype":"float32","shape":[1],"data_offsets":[0,4]}})",
            {0, 0, 0, 0}
        },
        {
            "shape_overflow",
            std::string(R"({"x":{"dtype":"F64","shape":[)") +
                std::to_string(std::numeric_limits<size_t>::max()) +
                R"(,2],"data_offsets":[0,0]}})",
            {}
        },
        {
            "reversed_offsets",
            R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}})",
            {0}
        },
        {
            "hole",
            R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})",
            {0, 0}
        },
        {
            "overlap",
            R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}})",
            {0, 0}
        },
        {
            "trailing_bytes",
            R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
            {0, 1}
        }
    };

    for (const auto& c : invalid_cases) {
        const auto path = root / (std::string(c.name) + ".safetensors");
        TASSERT_TRUE(write_fixture(path, c.header, c.data));
        TASSERT_TRUE(!load_fixture(path, error));
        TASSERT_TRUE(!error.empty());
    }

    // Valid scalar and valid empty tensor: both are explicitly allowed by the format.
    {
        const auto path = root / "valid_scalar.safetensors";
        TASSERT_TRUE(write_fixture(
            path,
            R"({"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"empty":{"dtype":"U8","shape":[0],"data_offsets":[4,4]},"__metadata__":{"source":"test"}})",
            {0, 0, 0, 0}
        ));
        TASSERT_TRUE(load_fixture(path, error));
    }

    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    return 0;
}
