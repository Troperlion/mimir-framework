#ifndef __TENSOR_VIZ_TEXT_PAYLOAD_HPP__
#define __TENSOR_VIZ_TEXT_PAYLOAD_HPP__

#include "Model.hpp"
#include "PromptParsing.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace VizTextPayload {

struct Payload {
    std::string prompt;
    std::string tags;
    std::string tokens;
    std::string encoding;
};

inline std::string escapeTokenPiece(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

inline std::string formatTokens(const Tokenizer* tok, const std::vector<int>& token_ids) {
    if (!tok || token_ids.empty()) return std::string();

    std::ostringstream oss;
    oss << "count=" << token_ids.size();
    for (size_t i = 0; i < token_ids.size(); ++i) {
        oss << "\n[" << i << "] ";
        const int tid = token_ids[i];
        std::string piece;
        try {
            piece = tok->getTokenById(tid);
        } catch (...) {
            piece.clear();
        }
        piece = escapeTokenPiece(piece);
        oss << "id=" << tid << " text=\"" << piece << "\"";
    }
    return oss.str();
}

inline std::string formatValues(const std::vector<float>& values, size_t token_count) {
    try {
        if (values.empty()) return "dim=0";

        double sum = 0.0;
        double squared_sum = 0.0;
        double absolute_sum = 0.0;
        float vmin = std::numeric_limits<float>::infinity();
        float vmax = -std::numeric_limits<float>::infinity();
        size_t finite_count = 0;
        size_t zero_count = 0;
        size_t nan_count = 0;
        size_t positive_inf_count = 0;
        size_t negative_inf_count = 0;
        for (float x : values) {
            if (std::isnan(x)) {
                ++nan_count;
                continue;
            }
            if (std::isinf(x)) {
                if (x > 0.0f) ++positive_inf_count;
                else ++negative_inf_count;
                continue;
            }
            ++finite_count;
            if (x == 0.0f) ++zero_count;
            sum += static_cast<double>(x);
            squared_sum += static_cast<double>(x) * static_cast<double>(x);
            absolute_sum += std::fabs(static_cast<double>(x));
            if (x < vmin) vmin = x;
            if (x > vmax) vmax = x;
        }

        const double denominator = static_cast<double>(std::max<size_t>(1, finite_count));
        const double mean = sum / denominator;
        const double variance = std::max(0.0, squared_sum / denominator - mean * mean);
        const double standard_deviation = std::sqrt(variance);
        const double l2 = std::sqrt(std::max(0.0, squared_sum));
        const double rms = std::sqrt(std::max(0.0, squared_sum / denominator));

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(8);
        oss << "dim=" << values.size()
            << " tokens=" << token_count
            << " finite=" << finite_count
            << " zeros=" << zero_count
            << " nan=" << nan_count
            << " +inf=" << positive_inf_count
            << " -inf=" << negative_inf_count
            << " mean=" << mean
            << " std=" << standard_deviation
            << " min=" << (finite_count > 0 ? vmin : 0.0f)
            << " max=" << (finite_count > 0 ? vmax : 0.0f)
            << " l1=" << absolute_sum
            << " l2=" << l2
            << " rms=" << rms
            << "\nvalues:";

        for (size_t i = 0; i < values.size(); ++i) {
            oss << "\n[" << i << "]=" << values[i];
        }
        return oss.str();
    } catch (...) {
        return "encode_error";
    }
}

inline std::string formatEncoding(const ConditioningEncoder* enc, const std::vector<int>& token_ids) {
    if (!enc || token_ids.empty()) return {};
    try { std::vector<float> values; enc->encodeInto(values, token_ids); return formatValues(values, token_ids.size()); }
    catch (...) { return "encode_error"; }
}

inline Payload buildDatasetTextPayload(
    const Model* model,
    const Tokenizer* tok,
    const std::string& prompt,
    const std::vector<int>* ids_hint,
    int pad_id,
    bool caption_kv_enable = false,
    bool caption_structured_enable = true,
    bool caption_structured_canonicalize = true
) {
    Payload out;

    PromptParsing::ParseOptions options;
    options.kv_enable = caption_kv_enable;
    options.structured_enable = caption_structured_enable;
    options.structured_canonicalize = caption_structured_canonicalize;

    const PromptParsing::PromptAnalysis analysis = PromptParsing::analyzePrompt(prompt, options);
    out.prompt = analysis.display_prompt.empty() ? prompt : analysis.display_prompt;
    out.tags = analysis.tags;

    if (!tok || out.prompt.empty()) return out;

    std::vector<int> ids;
    if (ids_hint && !ids_hint->empty()) {
        ids = *ids_hint;
    } else {
        try {
            ids = tok->tokenize(out.prompt);
        } catch (...) {
            ids.clear();
        }
    }

    if (pad_id >= 0) {
        while (!ids.empty() && ids.back() == pad_id) ids.pop_back();
    }

    if (model && !ids_hint && model->hasTensor("lumen/text/context") && model->hasTensorInt("text_ids")) ids = model->getTensorInt("text_ids");
    out.tokens = formatTokens(tok, ids);

    const ConditioningEncoder* enc = nullptr;
    if (model && model->getHasEncoder()) enc = &model->getEncoder();
    out.encoding = formatEncoding(enc, ids);
    if (model && model->hasTensor("lumen/text/context")) {
        out.encoding = "Tensor: lumen/text/context\n" + formatValues(model->getTensor("lumen/text/context"), ids.size());
    }
    return out;
}

} // namespace VizTextPayload

#endif