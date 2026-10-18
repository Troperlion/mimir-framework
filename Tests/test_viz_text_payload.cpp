#include "test_utils.hpp"

#include "VizTextPayload.hpp"

#include <string>
#include <vector>

int main() {
    Tokenizer tokenizer(128);
    std::vector<int> token_ids;
    for (int index = 0; index < 32; ++index) {
        token_ids.push_back(tokenizer.addToken(
            "token_" + std::to_string(index) + "_with_complete_text"));
    }
    const int quoted_id = tokenizer.addToken("quoted_\"token\\path");
    token_ids.push_back(quoted_id);

    const std::string tokens = VizTextPayload::formatTokens(&tokenizer, token_ids);
    TASSERT_TRUE(tokens.find("count=33") != std::string::npos);
    TASSERT_TRUE(tokens.find("[32] id=" + std::to_string(quoted_id)) != std::string::npos);
    TASSERT_TRUE(tokens.find("token_31_with_complete_text") != std::string::npos);
    TASSERT_TRUE(tokens.find("quoted_\\\"token\\\\path") != std::string::npos);
    TASSERT_TRUE(tokens.find(" more)") == std::string::npos);

    ConditioningEncoder encoder(12, 128);
    encoder.ensureVocabSize(tokenizer.getVocabSize(), 123U);
    encoder.initRandom(456U);
    const std::string encoding = VizTextPayload::formatEncoding(&encoder, token_ids);
    TASSERT_TRUE(encoding.find("dim=12") != std::string::npos);
    TASSERT_TRUE(encoding.find("tokens=33") != std::string::npos);
    TASSERT_TRUE(encoding.find("finite=") != std::string::npos);
    TASSERT_TRUE(encoding.find("zeros=") != std::string::npos);
    TASSERT_TRUE(encoding.find("nan=") != std::string::npos);
    TASSERT_TRUE(encoding.find("std=") != std::string::npos);
    TASSERT_TRUE(encoding.find("l1=") != std::string::npos);
    TASSERT_TRUE(encoding.find("l2=") != std::string::npos);
    TASSERT_TRUE(encoding.find("rms=") != std::string::npos);
    TASSERT_TRUE(encoding.find("[11]=") != std::string::npos);
    TASSERT_TRUE(encoding.find(",...") == std::string::npos);

    Model model;
    model.setHasEncoder(false);
    model.storeTensorInt("text_ids", {quoted_id, token_ids[0]});
    model.storeTensor("lumen/text/context", {0.25f, -2.5f, 4.f});
    const auto actual = VizTextPayload::buildDatasetTextPayload(&model, &tokenizer, "display prompt", nullptr, -1);
    TASSERT_TRUE(actual.tokens.find("count=2") != std::string::npos);
    TASSERT_TRUE(actual.encoding.find("Tensor: lumen/text/context") != std::string::npos);
    TASSERT_TRUE(actual.encoding.find("[1]=-2.50000000") != std::string::npos);
    return 0;
}