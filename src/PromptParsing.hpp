#ifndef __TENSOR_PROMPT_PARSING_HPP__
#define __TENSOR_PROMPT_PARSING_HPP__

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "include/json.hpp"

namespace PromptParsing {

using json = nlohmann::json;

inline std::string trimWs(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

inline bool startsWith(const std::string& s, const char* lit) {
    if (!lit) return false;
    const size_t n = std::char_traits<char>::length(lit);
    if (s.size() < n) return false;
    return std::equal(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(n), lit);
}

inline bool endsWith(const std::string& s, const char* lit) {
    if (!lit) return false;
    const size_t n = std::char_traits<char>::length(lit);
    if (s.size() < n) return false;
    return std::equal(s.end() - static_cast<std::ptrdiff_t>(n), s.end(), lit);
}

inline std::string toUpperAscii(std::string s) {
    for (char& ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (c >= 'a' && c <= 'z') ch = static_cast<char>(c - ('a' - 'A'));
    }
    return s;
}

inline std::string toLowerAscii(std::string s) {
    for (char& ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (c >= 'A' && c <= 'Z') ch = static_cast<char>(c + ('a' - 'A'));
    }
    return s;
}

inline void splitLines(const std::string& s, std::vector<std::string>& out) {
    out.clear();
    std::string cur;
    cur.reserve(std::min<size_t>(s.size(), 256));
    for (char ch : s) {
        if (ch == '\r') continue;
        if (ch == '\n') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    out.push_back(cur);
}

inline std::string normalizeSpaces(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool prev_ws = false;
    for (char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        const bool ws = std::isspace(c) != 0;
        if (ws) {
            if (!prev_ws) out.push_back(' ');
            prev_ws = true;
        } else {
            out.push_back(ch);
            prev_ws = false;
        }
    }
    return trimWs(out);
}

inline bool containsSeparator(char ch, const char* separators) {
    if (!separators) return false;
    for (const char* p = separators; *p != '\0'; ++p) {
        if (ch == *p) return true;
    }
    return false;
}

inline std::string normalizeDelimitedList(const std::string& s, const char* separators, const std::string& joiner) {
    std::vector<std::string> parts;
    std::string cur;
    for (char ch : s) {
        if (ch == '\r') continue;
        if (containsSeparator(ch, separators)) {
            const std::string t = trimWs(cur);
            if (!t.empty()) parts.push_back(t);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }

    const std::string tail = trimWs(cur);
    if (!tail.empty()) parts.push_back(tail);

    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) out += joiner;
        out += parts[i];
    }
    return out;
}

inline std::string normalizeTagsBlock(const std::string& s) {
    return normalizeDelimitedList(s, ",;.|\n\t", ", ");
}

inline std::string normalizeKvListAsSpaces(const std::string& s) {
    std::string tmp;
    tmp.reserve(s.size());
    for (char ch : s) {
        if (ch == '\r') continue;
        if (ch == ',' || ch == ';' || ch == '\n' || ch == '\t') tmp.push_back(' ');
        else tmp.push_back(ch);
    }
    return normalizeSpaces(tmp);
}

inline bool tryParseJsonObject(const std::string& raw, json& out) {
    const std::string trimmed = trimWs(raw);
    if (trimmed.empty() || trimmed.front() != '{') return false;

    json parsed = json::parse(trimmed, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) return false;
    out = std::move(parsed);
    return true;
}

inline std::string jsonScalarToString(const json& value) {
    if (value.is_string()) return value.get<std::string>();
    if (value.is_boolean()) return value.get<bool>() ? "true" : "false";
    if (value.is_number_integer()) return std::to_string(value.get<long long>());
    if (value.is_number_unsigned()) return std::to_string(value.get<unsigned long long>());
    if (value.is_number_float()) return std::to_string(value.get<double>());
    return std::string();
}

inline std::string jsonFieldAsJoinedString(const json& obj,
                                           std::initializer_list<const char*> keys,
                                           const std::string& joiner,
                                           bool normalize_as_tags,
                                           bool normalize_as_spaces) {
    for (const char* key : keys) {
        if (!key || !obj.contains(key)) continue;
        const auto& value = obj.at(key);
        if (value.is_array()) {
            std::vector<std::string> parts;
            parts.reserve(value.size());
            for (const auto& item : value) {
                const std::string part = trimWs(jsonScalarToString(item));
                if (!part.empty()) parts.push_back(part);
            }
            std::string joined;
            for (size_t i = 0; i < parts.size(); ++i) {
                if (i > 0) joined += joiner;
                joined += parts[i];
            }
            if (normalize_as_tags) return normalizeTagsBlock(joined);
            if (normalize_as_spaces) return normalizeKvListAsSpaces(joined);
            return normalizeSpaces(joined);
        }

        if (value.is_string() || value.is_boolean() || value.is_number()) {
            const std::string scalar = jsonScalarToString(value);
            if (normalize_as_tags) return normalizeTagsBlock(scalar);
            if (normalize_as_spaces) return normalizeKvListAsSpaces(scalar);
            return normalizeSpaces(scalar);
        }
    }
    return std::string();
}

struct KvCaption {
    bool has_any = false;
    std::string caption;
    std::string theme;
    std::string keywords;
    std::string tokens;
};

inline bool parseKvLine(const std::string& line_in, std::string& out_key_lc, std::string& out_value) {
    const std::string line = trimWs(line_in);
    if (line.empty()) return false;
    const size_t colon = line.find(':');
    if (colon == std::string::npos || colon == 0) return false;

    std::string key = trimWs(line.substr(0, colon));
    std::string val = trimWs(line.substr(colon + 1));
    if (key.empty()) return false;
    out_key_lc = toLowerAscii(key);
    out_value = val;
    return true;
}

inline KvCaption parseKvCaption(const std::string& caption) {
    KvCaption out;

    json obj;
    if (tryParseJsonObject(caption, obj)) {
        out.caption = jsonFieldAsJoinedString(obj, {"caption", "prompt", "text", "description", "desc"}, " ", false, false);
        out.theme = jsonFieldAsJoinedString(obj, {"theme", "context", "contexte"}, " ", false, false);
        out.keywords = jsonFieldAsJoinedString(obj, {"keywords", "tags"}, " ", false, true);
        out.tokens = jsonFieldAsJoinedString(obj, {"tokens"}, " ", false, true);
        out.has_any = (!out.caption.empty() || !out.theme.empty() || !out.keywords.empty() || !out.tokens.empty());
        return out;
    }

    std::vector<std::string> lines;
    splitLines(caption, lines);
    for (const std::string& raw : lines) {
        std::string k;
        std::string v;
        if (!parseKvLine(raw, k, v)) continue;
        if (k == "caption" || k == "prompt" || k == "text") {
            out.caption = v;
            out.has_any = true;
        } else if (k == "theme" || k == "context") {
            out.theme = v;
            out.has_any = true;
        } else if (k == "keywords" || k == "tags") {
            out.keywords = v;
            out.has_any = true;
        } else if (k == "tokens") {
            out.tokens = v;
            out.has_any = true;
        }
    }

    out.keywords = normalizeKvListAsSpaces(out.keywords);
    out.tokens = normalizeKvListAsSpaces(out.tokens);
    out.caption = normalizeSpaces(out.caption);
    out.theme = normalizeSpaces(out.theme);
    return out;
}

inline std::string composeKvCaptionPrompt(const KvCaption& kv) {
    std::string out;
    auto add_part = [&](const std::string& p) {
        if (p.empty()) return;
        if (!out.empty()) out += " | ";
        out += p;
    };
    add_part(kv.caption);
    add_part(kv.theme);
    add_part(kv.keywords);
    add_part(kv.tokens);
    return normalizeSpaces(out);
}

inline std::vector<std::string> splitTerms(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(32);
    auto flush = [&]() {
        std::string t = trimWs(cur);
        if (!t.empty()) out.push_back(t);
        cur.clear();
    };
    for (char ch : s) {
        if (ch == '\r') continue;
        const unsigned char c = static_cast<unsigned char>(ch);
        const bool sep = (ch == ',' || ch == ';' || ch == '\n' || ch == '\t' || std::isspace(c));
        if (sep) flush();
        else cur.push_back(ch);
    }
    flush();
    return out;
}

struct StructuredPrompt {
    bool has_any_header = false;
    std::string tags;
    std::string context;
    std::string mentality;
    std::string text;
    std::vector<std::pair<std::string, std::string>> extras;
};

inline StructuredPrompt parseStructuredPrompt(const std::string& caption) {
    StructuredPrompt out;
    json obj;
    if (!tryParseJsonObject(caption, obj)) {
        return out;
    }

    out.has_any_header = true;
    out.tags = jsonFieldAsJoinedString(obj, {"tags", "keywords"}, ", ", true, false);
    out.context = jsonFieldAsJoinedString(obj, {"context", "contexte", "theme"}, " ", false, false);
    out.mentality = jsonFieldAsJoinedString(obj, {"mentality", "mentalite", "mood"}, " ", false, false);
    out.text = jsonFieldAsJoinedString(obj, {"text", "texte", "description", "desc", "prompt", "caption"}, " / ", false, false);

    static const std::unordered_set<std::string> known_keys = {
        "tags", "keywords", "context", "contexte", "theme",
        "mentality", "mentalite", "mood",
        "text", "texte", "description", "desc", "prompt", "caption",
        "tokens"
    };

    for (auto it = obj.begin(); it != obj.end(); ++it) {
        if (known_keys.find(it.key()) != known_keys.end()) continue;
        std::string value;
        if (it.value().is_array()) {
            value = jsonFieldAsJoinedString(obj, {it.key().c_str()}, " | ", false, false);
        } else if (it.value().is_object()) {
            value = it.value().dump();
        } else {
            value = normalizeSpaces(jsonScalarToString(it.value()));
        }
        if (!value.empty()) out.extras.emplace_back(toUpperAscii(it.key()), value);
    }

    return out;
}

inline std::string composeStructuredPrompt(const StructuredPrompt& prompt, bool canonicalize) {
    if (!prompt.has_any_header) return std::string();

    auto add_line = [](std::string& out, const std::string& key, const std::string& value) {
        if (value.empty()) return;
        if (!out.empty()) out.push_back('\n');
        out += key;
        out += ": ";
        out += value;
    };

    if (canonicalize) {
        std::string out;
        add_line(out, "TAGS", prompt.tags);
        add_line(out, "CONTEXT", prompt.context);
        add_line(out, "MENTALITY", prompt.mentality);
        add_line(out, "TEXT", prompt.text);
        for (const auto& kv : prompt.extras) {
            if (kv.first.empty() || kv.second.empty()) continue;
            add_line(out, kv.first, kv.second);
        }
        return out;
    }

    json out = json::object();
    if (!prompt.tags.empty()) out["tags"] = splitTerms(normalizeDelimitedList(prompt.tags, ",", " "));
    if (!prompt.context.empty()) out["context"] = prompt.context;
    if (!prompt.mentality.empty()) out["mentality"] = prompt.mentality;
    if (!prompt.text.empty()) out["text"] = prompt.text;
    for (const auto& kv : prompt.extras) {
        if (kv.first.empty() || kv.second.empty()) continue;
        out[toLowerAscii(kv.first)] = kv.second;
    }
    return out.dump();
}

inline void applyStructuredDropout(StructuredPrompt& prompt, std::mt19937& rng, float p_tags, float p_ctx, float p_ment, float p_txt) {
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    if (p_tags > 0.0f && u01(rng) < p_tags) prompt.tags.clear();
    if (p_ctx > 0.0f && u01(rng) < p_ctx) prompt.context.clear();
    if (p_ment > 0.0f && u01(rng) < p_ment) prompt.mentality.clear();
    if (p_txt > 0.0f && u01(rng) < p_txt) prompt.text.clear();
}

inline std::string buildTagsDisplay(const KvCaption& kv, const StructuredPrompt& structured) {
    std::vector<std::string> parts;
    std::unordered_set<std::string> seen;
    auto add = [&](const std::string& part) {
        const std::string clean = normalizeSpaces(part);
        if (clean.empty()) return;
        if (seen.insert(clean).second) parts.push_back(clean);
    };

    add(structured.tags);
    if (!kv.keywords.empty()) add(std::string("keywords: ") + kv.keywords);
    if (!kv.tokens.empty()) add(std::string("tokens: ") + kv.tokens);

    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) out += " | ";
        out += parts[i];
    }
    return out;
}

struct ParseOptions {
    bool kv_enable = false;
    bool structured_enable = true;
    bool structured_canonicalize = true;
};

struct PromptAnalysis {
    std::string normalized_prompt;
    std::string display_prompt;
    std::string tags;
    KvCaption kv;
    StructuredPrompt structured;
};

inline PromptAnalysis analyzePrompt(const std::string& raw_prompt, const ParseOptions& options);

inline std::vector<std::string> extractLabelCandidates(const std::string& raw_prompt, bool lowercase_tags = true) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    seen.reserve(64);

    auto push_norm = [&](std::string s) {
        s = normalizeSpaces(s);
        while (!s.empty()) {
            const unsigned char c = static_cast<unsigned char>(s.front());
            if (std::isalnum(c) || c >= 128 || s.front() == '_' || s.front() == '-') break;
            s.erase(s.begin());
        }
        while (!s.empty()) {
            const unsigned char c = static_cast<unsigned char>(s.back());
            if (std::isalnum(c) || c >= 128 || s.back() == '_' || s.back() == '-') break;
            s.pop_back();
        }
        s = normalizeSpaces(s);
        if (s.empty()) return;
        if (lowercase_tags) s = toLowerAscii(s);
        if (seen.insert(s).second) out.push_back(std::move(s));
    };

    auto add_word_tokens = [&](const std::string& s) {
        std::string tok;
        tok.reserve(s.size());
        auto flush = [&]() {
            if (tok.empty()) return;
            push_norm(tok);
            tok.clear();
        };
        for (char ch : s) {
            const unsigned char u = static_cast<unsigned char>(ch);
            const bool is_word = std::isalnum(u) || u >= 128 || ch == '_' || ch == '-' || ch == '\'';
            if (is_word) tok.push_back(ch);
            else flush();
        }
        flush();
    };

    auto emit_segment = [&](const std::string& seg) {
        push_norm(seg);
        add_word_tokens(seg);
    };

    const PromptAnalysis analysis = analyzePrompt(raw_prompt, ParseOptions{});
    std::string source = analysis.tags;
    if (source.empty() && !analysis.structured.text.empty()) source = analysis.structured.text;
    if (source.empty() && !analysis.kv.caption.empty()) source = analysis.kv.caption;
    if (source.empty()) source = raw_prompt;

    std::string cur;
    cur.reserve(source.size());
    for (char ch : source) {
        const bool is_sep = (ch == '.' || ch == ',' || ch == ';' || ch == '\n' || ch == '\r' || ch == '\t' || ch == '|');
        if (!is_sep) {
            cur.push_back(ch);
            continue;
        }
        emit_segment(cur);
        cur.clear();
    }
    emit_segment(cur);
    return out;
}

inline PromptAnalysis analyzePrompt(const std::string& raw_prompt, const ParseOptions& options) {
    PromptAnalysis out;
    out.normalized_prompt = raw_prompt;
    out.display_prompt = normalizeSpaces(raw_prompt);

    if (options.structured_enable) {
        out.structured = parseStructuredPrompt(out.normalized_prompt);
        if (out.structured.has_any_header) {
            const std::string composed = composeStructuredPrompt(out.structured, options.structured_canonicalize);
            if (!composed.empty()) out.normalized_prompt = composed;
        }
    }

    if (options.kv_enable) {
        out.kv = parseKvCaption(raw_prompt);
        if (out.kv.has_any && !out.structured.has_any_header) {
            out.normalized_prompt = composeKvCaptionPrompt(out.kv);
        }
    }

    out.tags = buildTagsDisplay(out.kv, out.structured);
    if (out.structured.has_any_header && !out.structured.text.empty()) {
        out.display_prompt = out.structured.text;
    } else if (out.kv.has_any && !out.kv.caption.empty()) {
        out.display_prompt = out.kv.caption;
    } else if (!out.normalized_prompt.empty()) {
        out.display_prompt = out.normalized_prompt;
    }
    if (out.display_prompt.empty()) out.display_prompt = raw_prompt;
    return out;
}

} // namespace PromptParsing

#endif