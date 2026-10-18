#include "viz/WorkbenchTheme.hpp"
#include "viz/UiPrimitives.hpp"
#include "viz/WorkbenchLayout.hpp"
#include "Visualizer.hpp"
#include "viz/DatasetBBoxBadge.hpp"
#include "viz/FeatureProbe.hpp"
#include "viz/LumenOutputRole.hpp"
#include "viz/ModelGraphLanes.hpp"
#include "viz/DrawDatasetBBoxOverlay.hpp"
#include "Model.hpp"
#include "LayerTypes.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <cctype>
#include <filesystem>
#include <limits>

static const char* kVizUISettingsFile = "viz_ui_settings.json";

namespace {
static inline vizgfx::FloatRect make_rect(float x, float y, float w, float h) {
    return vizgfx::FloatRect(vizgfx::Vector2f(x, y), vizgfx::Vector2f(w, h));
}

static vizgfx::Color detection_heatmap_color(float score) {
    static const std::array<vizgfx::Color, 5> heatmap = {
        vizgfx::Color(45, 70, 210), vizgfx::Color(25, 190, 235),
        vizgfx::Color(70, 225, 130), vizgfx::Color(250, 215, 55),
        vizgfx::Color(235, 55, 45)};
    const float scaled = std::clamp(score, 0.0f, 1.0f) *
        static_cast<float>(heatmap.size() - 1);
    const size_t lower = static_cast<size_t>(scaled);
    const size_t upper = std::min(lower + 1, heatmap.size() - 1);
    const float mix = scaled - static_cast<float>(lower);
    const auto channel = [mix](uint8_t from, uint8_t to) {
        return static_cast<uint8_t>(std::lround(
            static_cast<float>(from) +
            (static_cast<float>(to) - static_cast<float>(from)) * mix));
    };
    return vizgfx::Color(channel(heatmap[lower].r, heatmap[upper].r),
                         channel(heatmap[lower].g, heatmap[upper].g),
                         channel(heatmap[lower].b, heatmap[upper].b));
}

struct ParsedVizLabel {
    std::string raw;
    std::string path;
    std::string extra;
    std::vector<std::string> parts;

    std::string model;
    std::string layer_path;
    std::string layer_type;
    std::string tag;        // ex: DS/OUT/VAL/PRE/ACT/?
    std::string headline;   // ex: Dataset / Sortie / Validation / Prétraitement / Activations
    std::string short_text; // ex: unet2d/down_0/conv_in
};

static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(s.size());
    for (const char c : s) {
        if (c == delim) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

static std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' || s[b] == '\r')) ++b;
    if (b >= s.size()) return std::string();
    size_t e = s.size();
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\n' || s[e - 1] == '\r')) --e;
    return s.substr(b, e - b);
}

static std::string clamp_text_end(const std::string& s, size_t max_chars) {
    if (max_chars == 0) return std::string();
    if (s.size() <= max_chars) return s;
    if (max_chars <= 3) return s.substr(0, max_chars);
    return s.substr(0, max_chars - 3) + "...";
}

static std::string format_decimal(double v, int max_decimals = 8) {
    if (!std::isfinite(v)) {
        if (std::isnan(v)) return "nan";
        return (v > 0.0) ? "inf" : "-inf";
    }

    const double av = std::fabs(v);
    int prec = 0;
    if (av >= 1000.0) prec = 0;
    else if (av >= 100.0) prec = 1;
    else if (av >= 1.0) prec = 3;
    else if (av >= 0.01) prec = 5;
    else prec = std::min(max_decimals, 8);

    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss << std::setprecision(std::max(0, prec)) << v;
    std::string s = ss.str();

    if (prec > 0) {
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
        if (s.empty()) s = "0";
    }

    return s;
}

static std::string basename_like(const std::string& p) {
    if (p.empty()) return p;
    const size_t pos = p.find_last_of("/\\");
    if (pos == std::string::npos) return p;
    if (pos + 1 >= p.size()) return p;
    return p.substr(pos + 1);
}

static bool contains_part(const std::vector<std::string>& parts, const std::string& needle) {
    return std::any_of(parts.begin(), parts.end(), [&](const std::string& p) { return p == needle; });
}

static std::string join_parts(const std::vector<std::string>& parts, size_t start, size_t end) {
    if (end <= start || start >= parts.size()) return {};
    end = std::min(end, parts.size());
    std::string out;
    for (size_t i = start; i < end; ++i) {
        if (parts[i].empty()) continue;
        if (!out.empty()) out += "/";
        out += parts[i];
    }
    return out;
}

static bool is_norm_type_name(const std::string& t) {
    return t == "BatchNorm1d" || t == "BatchNorm2d" || t == "InstanceNorm2d" ||
           t == "GroupNorm" || t == "LayerNorm" || t == "RMSNorm";
}

static bool is_activation_type_name(const std::string& t) {
    return t == "ReLU" || t == "LeakyReLU" || t == "GELU" || t == "GEGLU" ||
           t == "SiLU" || t == "Tanh" || t == "Sigmoid" || t == "Softmax" ||
           t == "LogSoftmax" || t == "Softplus" || t == "Mish" ||
           t == "HardSigmoid" || t == "HardSwish";
}

static bool is_attention_type_name(const std::string& t) {
    return t == "SelfAttention" || t == "CrossAttention" || t == "MultiHeadAttention";
}

static bool is_upsample_type_name(const std::string& t) {
    return t == "ConvTranspose2d" || t == "UpsampleNearest" || t == "UpsampleBilinear" ||
           t == "UpsampleBicubic" || t == "PixelShuffle";
}

static std::string short_layer_type_name(const std::string& t) {
    if (t == "Conv2d") return "conv";
    if (t == "ConvTranspose2d") return "deconv";
    if (t == "DepthwiseConv2d") return "dwconv";
    if (t == "GroupNorm") return "gn";
    if (t == "LayerNorm") return "ln";
    if (t == "RMSNorm") return "rms";
    if (t == "BatchNorm1d" || t == "BatchNorm2d") return "bn";
    if (t == "InstanceNorm2d") return "in";
    if (t == "SelfAttention") return "attn";
    if (t == "CrossAttention") return "xattn";
    if (t == "MultiHeadAttention") return "mha";
    if (t == "Reparameterize") return "reparam";
    if (t == "UpsampleNearest") return "up";
    if (t == "UpsampleBilinear") return "upbil";
    if (t == "UpsampleBicubic") return "upbic";
    if (t == "PixelShuffle") return "pix";
    if (t == "Add") return "add";
    if (t == "Concat") return "cat";
    if (t == "Permute") return "perm";
    if (t == "Reshape" || t == "View") return "shape";
    if (t == "Identity") return "id";
    return {};
}

static const char* heatmap_palette_name(int palette_id) {
    switch (palette_id) {
        case 1: return "TURBO";
        case 2: return "INFERNO";
        case 3: return "VIRIDIS";
        default: return "CLASSIC";
    }
}

static std::vector<uint8_t> colorize_gray_heatmap_rgb(const std::vector<uint8_t>& gray, int palette_id) {
    std::vector<uint8_t> rgb;
    rgb.resize(gray.size() * 3ULL);
    for (size_t i = 0; i < gray.size(); ++i) {
        const float t = static_cast<float>(gray[i]) / 255.0f;

        float r = 0.0f, g = 0.0f, b = 0.0f;
        if (palette_id == 1) {
            // TURBO (approximation polynomiale compacte)
            const float x = t;
            r = 34.61f + x * (1172.33f + x * (-10793.56f + x * (33300.12f + x * (-38394.49f + x * 14825.05f))));
            g = 23.31f + x * (557.33f + x * (1225.33f + x * (-3574.96f + x * (1073.77f + x * 707.56f))));
            b = 27.2f + x * (3211.1f + x * (-15327.97f + x * (27814.0f + x * (-22569.18f + x * 6838.66f))));
        } else if (palette_id == 2) {
            // INFERNO (anchors)
            const float c0r = 20.f,  c0g = 11.f,  c0b = 52.f;
            const float c1r = 121.f, c1g = 28.f,  c1b = 109.f;
            const float c2r = 219.f, c2g = 82.f,  c2b = 59.f;
            const float c3r = 250.f, c3g = 229.f, c3b = 107.f;
            if (t < 0.33f) {
                const float u = t / 0.33f;
                r = c0r + (c1r - c0r) * u;
                g = c0g + (c1g - c0g) * u;
                b = c0b + (c1b - c0b) * u;
            } else if (t < 0.66f) {
                const float u = (t - 0.33f) / 0.33f;
                r = c1r + (c2r - c1r) * u;
                g = c1g + (c2g - c1g) * u;
                b = c1b + (c2b - c1b) * u;
            } else {
                const float u = (t - 0.66f) / 0.34f;
                r = c2r + (c3r - c2r) * u;
                g = c2g + (c3g - c2g) * u;
                b = c2b + (c3b - c2b) * u;
            }
        } else if (palette_id == 3) {
            // VIRIDIS (anchors)
            const float c0r = 68.f,  c0g = 1.f,   c0b = 84.f;
            const float c1r = 59.f,  c1g = 82.f,  c1b = 139.f;
            const float c2r = 33.f,  c2g = 145.f, c2b = 140.f;
            const float c3r = 253.f, c3g = 231.f, c3b = 37.f;
            if (t < 0.33f) {
                const float u = t / 0.33f;
                r = c0r + (c1r - c0r) * u;
                g = c0g + (c1g - c0g) * u;
                b = c0b + (c1b - c0b) * u;
            } else if (t < 0.66f) {
                const float u = (t - 0.33f) / 0.33f;
                r = c1r + (c2r - c1r) * u;
                g = c1g + (c2g - c1g) * u;
                b = c1b + (c2b - c1b) * u;
            } else {
                const float u = (t - 0.66f) / 0.34f;
                r = c2r + (c3r - c2r) * u;
                g = c2g + (c3g - c2g) * u;
                b = c2b + (c3b - c2b) * u;
            }
        } else {
            // CLASSIC: froid -> neutre -> chaud
            const float neg_r = 70.0f,  neg_g = 120.0f, neg_b = 210.0f;
            const float mid_r = 175.0f, mid_g = 175.0f, mid_b = 175.0f;
            const float pos_r = 220.0f, pos_g = 65.0f,  pos_b = 65.0f;
            if (t < 0.5f) {
                const float u = t / 0.5f;
                r = neg_r + (mid_r - neg_r) * u;
                g = neg_g + (mid_g - neg_g) * u;
                b = neg_b + (mid_b - neg_b) * u;
            } else {
                const float u = (t - 0.5f) / 0.5f;
                r = mid_r + (pos_r - mid_r) * u;
                g = mid_g + (pos_g - mid_g) * u;
                b = mid_b + (pos_b - mid_b) * u;
            }
        }

        const size_t base = i * 3ULL;
        rgb[base + 0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(r)), 0, 255));
        rgb[base + 1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(g)), 0, 255));
        rgb[base + 2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(b)), 0, 255));
    }
    return rgb;
}

static std::string strip_known_block_suffixes(const std::string& s) {
    if (s == "n" || s == "norm") return {};
    if (s == "bot_n") return "bot";
    if (s == "out_n") return "out";
    if (s.size() >= 2 && s[0] == 'n') {
        bool all_digits = true;
        for (size_t i = 1; i < s.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
                all_digits = false;
                break;
            }
        }
        if (all_digits) return {};
    }
    const std::array<std::string, 8> suffixes = {
        std::string("_norm"), std::string("_gn"), std::string("_ln"), std::string("_bn"),
        std::string("_in"), std::string("_rms"), std::string("/norm"), std::string("/n")
    };
    for (const auto& suffix : suffixes) {
        if (s.size() > suffix.size() && s.rfind(suffix) == s.size() - suffix.size()) {
            return s.substr(0, s.size() - suffix.size());
        }
    }
    return s;
}

static std::string canonical_path_role(const std::string& s) {
    if (s.empty()) return {};
    if (s == "self_attn" || s == "attn") return "attn";
    if (s == "cross_attn") return "xattn";
    if (s == "multi_attn") return "mha";
    if (s == "lm_head" || s == "classifier") return "head";
    if (s == "out_proj" || s == "text_proj" || s == "img_proj") return "proj";
    if (s == "final_ln" || s == "final_norm") return "final";
    if (s == "recon_to_hwc" || s == "out_to_hwc") return "out";
    if (s == "recon_chw" || s == "recon" || s == "out_concat" || s == "out_pack") return "out";
    if (s == "tok_emb") return "tok";
    if (s == "meanpool") return "pool";

    const std::array<std::pair<const char*, const char*>, 9> suffix_roles = {{
        {"_attn", "attn"},
        {"_proj", "proj"},
        {"_head", "head"},
        {"_out", "out"},
        {"_pool", "pool"},
        {"_norm", "norm"},
        {"_gn", "gn"},
        {"_ln", "ln"},
        {"_fc", "fc"},
    }};
    for (const auto& [suffix, role] : suffix_roles) {
        const std::string suf(suffix);
        if (s.size() > suf.size() && s.rfind(suf) == s.size() - suf.size()) {
            return role;
        }
    }

    return strip_known_block_suffixes(s);
}

static std::string find_label_anchor(const std::vector<std::string>& tail_parts,
                                     const std::string& alias,
                                     const std::string& role) {
    for (size_t i = tail_parts.size(); i > 0; --i) {
        const std::string token = canonical_path_role(tail_parts[i - 1]);
        if (token.empty()) continue;
        if (!alias.empty() && token == alias) continue;
        if (!role.empty() && token == role) continue;
        if (token == "vec" || token == "act" || token == "id") continue;
        return token;
    }
    return {};
}

static std::string format_block_short_text(const std::vector<std::string>& tail_parts,
                                           const std::string& tag,
                                           const std::string& alias,
                                           const std::string& layer_type,
                                           const std::string& fallback) {
    const std::string role = canonical_path_role(tail_parts.empty() ? std::string() : tail_parts.back());
    const std::string anchor = find_label_anchor(tail_parts, alias, role);

    if (tag == "N" && !alias.empty()) {
        return anchor.empty() ? alias : (anchor + "/" + alias);
    }
    if (tag == "AT" && !alias.empty()) {
        return anchor.empty() ? alias : (anchor + "/" + alias);
    }
    if (tag == "LAT" && !alias.empty()) {
        return anchor.empty() ? alias : (anchor + "/" + alias);
    }

    if (role == "proj" || role == "head" || role == "out" || role == "pool") {
        return anchor.empty() ? role : (anchor + "/" + role);
    }
    if (role == "final") {
        if (!alias.empty() && (alias == "ln" || alias == "gn" || alias == "bn" || alias == "rms")) {
            return std::string("final/") + alias;
        }
        return "final";
    }
    if (role == "fc") {
        return anchor.empty() ? std::string("fc") : (anchor + "/fc");
    }

    std::vector<std::string> normalized;
    normalized.reserve(tail_parts.size());
    for (const auto& part : tail_parts) {
        const std::string token = canonical_path_role(part);
        if (!token.empty()) normalized.push_back(token);
    }
    if (!normalized.empty()) {
        const size_t keep = std::min<size_t>(3, normalized.size());
        const size_t from = normalized.size() > keep ? (normalized.size() - keep) : 0;
        return join_parts(normalized, from, normalized.size());
    }

    (void)layer_type;
    return fallback;
}

static ParsedVizLabel parse_viz_label(const std::string& label_raw) {
    static thread_local std::unordered_map<std::string, ParsedVizLabel> label_cache;
    const auto it = label_cache.find(label_raw);
    if (it != label_cache.end()) {
        return it->second;
    }

    ParsedVizLabel out;
    out.raw = label_raw;

    // Convention: une partie "chemin" peut être suivie d'extra après " | ".
    // Ex: "vae_conv/input/dataset/rgb | /path/file.png"
    const std::string s = label_raw;
    const size_t bar = s.find("|");
    if (bar != std::string::npos) {
        out.path = trim(s.substr(0, bar));
        out.extra = trim(s.substr(bar + 1));
    } else {
        out.path = trim(s);
    }
    out.parts = split(out.path, '/');
    // Supprimer les segments vides.
    out.parts.erase(std::remove_if(out.parts.begin(), out.parts.end(), [](const std::string& p) { return p.empty(); }), out.parts.end());

    out.model = out.parts.empty() ? std::string() : out.parts.front();

    auto find_idx = [&](const std::string& v) -> int {
        for (size_t i = 0; i < out.parts.size(); ++i) {
            if (out.parts[i] == v) return static_cast<int>(i);
        }
        return -1;
    };

    const int idx_input = find_idx("input");
    const int idx_output = find_idx("output");
    const int idx_val = std::max(find_idx("validation"), find_idx("val"));
    const int idx_blocks = find_idx("blocks");
    const bool is_activation = (!out.parts.empty() && out.parts.back() == "activation");

    auto is_activation_type = [&](const std::string& t) { return is_activation_type_name(LayerRegistry::normalize_type(t)); };

    // Catégorisation heuristique (labels émis par LuaScripting + viz taps Model.cpp)
    // 1) Format viz taps: <model>/blocks/<path>/<LayerType> (ou .../<LayerType>/vec)
    if (idx_blocks >= 0) {
        // Déterminer le type (dernier segment, ou avant-dernier si suffixe /vec)
        std::string t = out.parts.empty() ? std::string() : out.parts.back();
        size_t drop = 1;
        if (t == "vec") {
            drop = 2;
            if (out.parts.size() >= 2) t = out.parts[out.parts.size() - 2];
        }

        out.layer_type = t;

        if (is_activation_type(t)) {
            out.tag = "ACT";
            out.headline = "Activations";
        } else if (is_norm_type_name(t)) {
            out.tag = "N";
            out.headline = "Norm";
        } else if (is_attention_type_name(t)) {
            out.tag = "AT";
            out.headline = "Attention";
        } else if (is_upsample_type_name(t)) {
            out.tag = "UP";
            out.headline = "Upsample";
        } else if (t == "Reparameterize") {
            out.tag = "LAT";
            out.headline = "Latent";
        } else if (t == "Add" || t == "Concat") {
            out.tag = "RES";
            out.headline = "Fusion";
        } else {
            out.tag = "L";
            out.headline = "Layer";
        }

        // Toute entree au format <model>/blocks/<layer>/<type> correspond a
        // un vrai noeud du graphe et porte le badge uniforme (L).
        out.tag = "L";
        out.headline = "Layer";

        const size_t start = static_cast<size_t>(idx_blocks + 1);
        const size_t end = (out.parts.size() > drop) ? (out.parts.size() - drop) : start;
        out.layer_path = join_parts(out.parts, start, end);

        std::vector<std::string> tail_parts;
        for (size_t i = start; i < end; ++i) tail_parts.push_back(out.parts[i]);

        const std::string alias = short_layer_type_name(t);
        out.short_text = format_block_short_text(
            tail_parts,
            out.tag,
            alias,
            t,
            out.layer_path.empty() ? std::string("blocks") : out.layer_path);
        if (out.short_text.empty()) out.short_text = out.layer_path.empty() ? std::string("blocks") : out.layer_path;
    } else if (idx_val >= 0) {
        out.tag = "VAL";
        out.headline = "Validation";
        std::string tail;
        for (size_t i = static_cast<size_t>(idx_val + 1); i < out.parts.size(); ++i) {
            if (!tail.empty()) tail += "/";
            tail += out.parts[i];
        }
        out.short_text = tail.empty() ? std::string("validation") : tail;
    } else if (idx_output >= 0) {
        out.tag = "OUT";
        out.headline = "Sortie";
        std::string tail;
        for (size_t i = static_cast<size_t>(idx_output + 1); i < out.parts.size(); ++i) {
            if (!tail.empty()) tail += "/";
            tail += out.parts[i];
        }
        out.short_text = tail.empty() ? std::string("output") : tail;
    } else if (idx_input >= 0) {
        // input/dataset vs input/* (prétraitement)
        const bool is_dataset = (idx_input + 1 < static_cast<int>(out.parts.size()) && out.parts[static_cast<size_t>(idx_input + 1)] == "dataset") ||
                                contains_part(out.parts, "dataset");
        if (is_dataset) {
            out.tag = "DS";
            out.headline = "Dataset";
        } else {
            out.tag = "PRE";
            out.headline = "Prétraitement";
        }
        std::string tail;
        for (size_t i = static_cast<size_t>(idx_input + 1); i < out.parts.size(); ++i) {
            if (!tail.empty()) tail += "/";
            tail += out.parts[i];
        }
        out.short_text = tail.empty() ? std::string("input") : tail;
    } else if (is_activation) {
        out.tag = "ACT";
        out.headline = "Activations";
        // Format attendu: model/block/component/activation (ou + profond)
        // On enlève model + activation et on garde une queue lisible.
        const size_t n = out.parts.size();
        size_t start = 1; // skip model
        size_t end = (n >= 1) ? (n - 1) : 0; // drop activation
        if (end <= start) {
            out.short_text = "activation";
        } else {
            // Garder jusqu'à 3 segments de fin (plus lisible dans les vignettes)
            const size_t keep = std::min<size_t>(3, end - start);
            const size_t from = end - keep;
            std::string tail;
            for (size_t i = from; i < end; ++i) {
                if (!tail.empty()) tail += "/";
                tail += out.parts[i];
            }
            out.short_text = tail;
        }
    } else {
        out.tag = "?";
        out.headline = "Image";
        // fallback: garder la fin du chemin
        if (!out.parts.empty()) {
            out.short_text = out.parts.back();
        }
    }

    // Amélioration: si extra est un chemin, on garde juste le basename.
    if (!out.extra.empty()) {
        const std::string bn = basename_like(out.extra);
        if (!bn.empty() && bn.size() < out.extra.size()) {
            out.extra = bn;
        }
    }

    if (label_cache.size() > 4096) {
        label_cache.clear();
    }
    label_cache.emplace(label_raw, out);
    return out;
}

static std::string lowercase_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static bool hidden_block_tip(const ParsedVizLabel& tip, bool hide_activation, bool hide_norm) {
    const std::string type=LayerRegistry::normalize_type(tip.layer_type);
    if (LayerRegistry::string_to_type(type)!=LayerType::UNKNOWN) {
        // Canonical graph captures carry their actual layer type; names are not authoritative.
        return (hide_activation && is_activation_type_name(type)) ||
               (hide_norm && is_norm_type_name(type));
    }
    bool activation=tip.tag=="ACT", norm=tip.tag=="N" || tip.headline=="Norm";
    for(const auto& raw:tip.parts) {
        const std::string part=lowercase_ascii(raw);
        activation=activation || part=="activation" || part=="act" ||
            is_activation_type_name(LayerRegistry::normalize_type(raw));
        const bool numbered_norm=part.size()>4 && part.rfind("norm",0)==0 &&
            std::all_of(part.begin()+4,part.end(),[](unsigned char c){return std::isdigit(c);});
        norm=norm || is_norm_type_name(LayerRegistry::normalize_type(raw)) || numbered_norm ||
            part=="norm" || part=="bn" || part=="gn" || part=="ln" || part=="rms";
        for(const char* suffix:{"_norm","_bn","_gn","_ln","_rms"}) {
            const std::string ending(suffix);
            norm=norm || (part.size()>ending.size() && part.compare(part.size()-ending.size(),ending.size(),ending)==0);
        }
    }
    return (hide_activation && activation) || (hide_norm && norm);
}

static std::string canonical_recon_loss_name(const std::string& raw_name) {
    const std::string name = lowercase_ascii(raw_name);
    if (name.empty() || name == "mse") return "mse";
    if (name == "mae") return "l1";
    if (name == "smoothl1") return "huber";
    if (name == "nll_gaussian") return "gaussian_nll";
    return name;
}

static int recon_loss_index(const std::string& raw_name) {
    const std::string name = canonical_recon_loss_name(raw_name);
    if (name == "l1" || name == "mae") return 1;
    if (name == "huber" || name == "smooth_l1" || name == "smoothl1") return 2;
    if (name == "charbonnier") return 3;
    if (name == "gaussian_nll" || name == "nll_gaussian") return 4;
    if (name == "bce") return 5;
    return 0;
}

static const char* recon_loss_name(int index) {
    static constexpr const char* names[] = {
        "mse", "mae", "huber", "charbonnier", "gaussian_nll", "bce"
    };
    return names[std::clamp(index, 0, 5)];
}

static float normalized_recon_loss(float value, const std::string& raw_name) {
    const std::string name = canonical_recon_loss_name(raw_name);
    if (name == "mse") return value * 0.25f;
    if (name == "l1" || name == "charbonnier") return value * 0.5f;
    return value;
}

static std::string display_tip_tag(const ParsedVizLabel& parsed) {
    if (parsed.tag == "VIZ") return "?";

    const std::string source = lowercase_ascii(
        parsed.path + " " + parsed.layer_path + " " + parsed.layer_type + " " +
        parsed.extra + " " + parsed.headline);
    const auto contains = [&](const char* token) {
        return source.find(token) != std::string::npos;
    };

    if (parsed.tag == "DS" || contains("dataset")) return "DS";
    if (Mimir::Viz::isLumenModelOutput(source)) return "ST";
    if (Mimir::Viz::lumenTipSection(source) == "Validation") return "VAL";
    if (!Mimir::Viz::isLumenIntermediate(source) &&
        (parsed.tag == "OUT" || contains("/output") || contains("output/") ||
         contains("sortie") || contains("diffusion_out") ||
         contains("recon_to_hwc") || contains("out_to_hwc"))) {
        return "ST";
    }
    if (contains("/txt/") || contains("/txt_") || contains("text") ||
        contains("token") || contains("tok_emb") || contains("prompt")) {
        return "TX";
    }
    if (contains("skip")) return "SK";
    if (parsed.layer_type == "Concat" || contains("concat")) return "CC";
    if (parsed.tag == "ACT" || contains("relu") || contains("gelu") ||
        contains("silu") || contains("sigmoid") || contains("softmax") ||
        contains("softplus") || contains("mish") || contains("hardswish")) {
        return "AC";
    }
    if (parsed.tag == "N" || is_norm_type_name(parsed.layer_type) ||
        contains("/norm") || contains("groupnorm") || contains("layernorm") ||
        contains("batchnorm") || contains("instancenorm") || contains("rmsnorm")) {
        return "NM";
    }
    return "L";
}

static std::string format_tip_display_label(const ParsedVizLabel& parsed) {
    const std::string path = !parsed.layer_path.empty()
        ? parsed.layer_path
        : (!parsed.path.empty() ? parsed.path : parsed.short_text);
    std::string label = parsed.extra;
    if (label.empty() && parsed.short_text != path) label = parsed.short_text;

    std::string text = "[" + display_tip_tag(parsed) + "] " + path;
    if (!label.empty() && label != path) text += " + " + label;
    return text;
}

static vizgfx::Color color_for_tag(const std::string& tag) {
    if (tag == "DS") return vizgfx::Color(120, 180, 220);
    if (tag == "TX") return vizgfx::Color(190, 165, 235);
    if (tag == "AC") return vizgfx::Color(160, 160, 220);
    if (tag == "NM") return vizgfx::Color(120, 205, 175);
    if (tag == "SK") return vizgfx::Color(235, 185, 105);
    if (tag == "CC") return vizgfx::Color(205, 190, 120);
    if (tag == "ST") return vizgfx::Color(105, 205, 145);
    if (tag == "VAL") return vizgfx::Color(115, 194, 204);
    if (tag == "PRE") return vizgfx::Color(140, 200, 220);
    if (tag == "OUT") return vizgfx::Color(220, 160, 120);
    if (tag == "VAL") return vizgfx::Color(160, 220, 160);
    if (tag == "ACT") return vizgfx::Color(160, 160, 220);
    if (tag == "N") return vizgfx::Color(120, 205, 175);
    if (tag == "AT") return vizgfx::Color(185, 145, 230);
    if (tag == "UP") return vizgfx::Color(235, 170, 105);
    if (tag == "LAT") return vizgfx::Color(230, 125, 165);
    if (tag == "RES") return vizgfx::Color(205, 190, 120);
    if (tag == "L") return vizgfx::Color(180, 180, 200);
    if (tag == "T") return vizgfx::Color(180, 170, 140);
    return vizgfx::Color(140, 140, 150);
}

static int workbenchPreviewSize(const vizgfx::FloatRect& area) {
    return std::max(32,std::min({200,int(area.size.x)-8,int(area.size.y)-36}));
}

static void position_sprite_centered_in_box(vizgfx::Sprite& sprite, float x, float y, float box_size) {
    const auto lb = sprite.getLocalBounds();
    if(lb.size.x>0.f && lb.size.y>0.f) {
        const float fit=box_size/std::max(lb.size.x,lb.size.y);
        sprite.setScale({fit,fit});
    }
    const auto sc = sprite.getScale();
    const float dw = lb.size.x * sc.x;
    const float dh = lb.size.y * sc.y;
    const float ox = (box_size - dw) * 0.5f;
    const float oy = (box_size - dh) * 0.5f;
    // Ajuster avec left/top au cas où localBounds n'est pas (0,0).
    const float px = x + ox - lb.position.x * sc.x;
    const float py = y + oy - lb.position.y * sc.y;
    // Snap pixel pour éviter les artefacts d'alignement/sub-pixel dans les grilles.
    sprite.setPosition(vizgfx::Vector2f(std::round(px), std::round(py)));
}

static bool is_layer_block_preview_smoothing_candidate(const ParsedVizLabel& p, int channels) {
    if (channels != 1 && channels != 3 && channels != 4) return false;
    if (p.tag == "DS" || p.tag == "PRE" || p.tag == "OUT" || p.tag == "VAL" || p.tag == "ACT") return false;

    const std::string& path = p.layer_path;
    if (path.find("raw_in") != std::string::npos ||
        path.find("in_reshape") != std::string::npos ||
        path.find("recon") != std::string::npos ||
        path.find("to_hwc") != std::string::npos) {
        return false;
    }

    return true;
}

static std::vector<uint8_t> smooth_preview_pixels(const std::vector<uint8_t>& pixels, int w, int h, int channels) {
    if (w <= 2 || h <= 2) return pixels;
    if (channels != 1 && channels != 3 && channels != 4) return pixels;

    std::vector<uint8_t> out = pixels;
    const int color_channels = (channels == 4) ? 3 : channels;

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const size_t dst = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * static_cast<size_t>(channels);
            for (int cc = 0; cc < color_channels; ++cc) {
                int acc = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        const size_t src = (static_cast<size_t>(y + ky) * static_cast<size_t>(w) + static_cast<size_t>(x + kx)) * static_cast<size_t>(channels) + static_cast<size_t>(cc);
                        acc += static_cast<int>(pixels[src]);
                    }
                }
                out[dst + static_cast<size_t>(cc)] = static_cast<uint8_t>(std::clamp(acc / 9, 0, 255));
            }
            if (channels == 4) {
                out[dst + 3] = pixels[dst + 3];
            }
        }
    }

    return out;
}

static int infer_channels_from_buffer_size(const std::vector<uint8_t>& pixels, int w, int h, int preferred) {
    if (w <= 0 || h <= 0) return 0;
    const size_t area = static_cast<size_t>(w) * static_cast<size_t>(h);
    if (area == 0) return 0;

    const size_t n = pixels.size();
    auto exact = [&](int c) -> bool {
        return n == area * static_cast<size_t>(c);
    };
    auto at_least = [&](int c) -> bool {
        return n >= area * static_cast<size_t>(c);
    };

    // 1) Correspondance exacte prioritaire.
    if (preferred == 1 || preferred == 3 || preferred == 4) {
        if (exact(preferred)) return preferred;
    }
    if (exact(4)) return 4;
    if (exact(3)) return 3;
    if (exact(1)) return 1;

    // 2) Fallback best-effort si le buffer contient plus de données que prévu.
    // Préférer d'abord le format demandé si plausible.
    if (preferred == 1 || preferred == 3 || preferred == 4) {
        if (at_least(preferred)) return preferred;
    }
    if (at_least(4)) return 4;
    if (at_least(3)) return 3;
    if (at_least(1)) return 1;
    return 0;
}

// -----------------------------------------------------------------------------
// Live tuning slider helpers (Metrics panel)
// -----------------------------------------------------------------------------
static constexpr float kLiveLRMin = 1e-7f;
static constexpr float kLiveLRMax = 1e-3f;
static constexpr int kLiveWarmupMax = 50000;

static float clamp01(float t) {
    return std::clamp(t, 0.0f, 1.0f);
}

[[maybe_unused]] static float lr_from_t(float t) {
    const float tt = clamp01(t);
    const float logmin = std::log10(kLiveLRMin);
    const float logmax = std::log10(kLiveLRMax);
    const float v = logmin + (logmax - logmin) * tt;
    return std::pow(10.0f, v);
}

static float t_from_lr(float lr) {
    float v = lr;
    if (!std::isfinite(v)) v = kLiveLRMin;
    v = std::clamp(v, kLiveLRMin, kLiveLRMax);
    const float logmin = std::log10(kLiveLRMin);
    const float logmax = std::log10(kLiveLRMax);
    const float lv = std::log10(v);
    const float denom = std::max(1e-6f, (logmax - logmin));
    return clamp01((lv - logmin) / denom);
}

[[maybe_unused]] static int warmup_from_t(float t) {
    const float tt = clamp01(t);
    const float v = tt * static_cast<float>(kLiveWarmupMax);
    return std::max(0, static_cast<int>(std::lround(v)));
}

static float t_from_warmup(int steps) {
    const int s = std::clamp(steps, 0, kLiveWarmupMax);
    return clamp01(static_cast<float>(s) / static_cast<float>(kLiveWarmupMax));
}

[[maybe_unused]] static float klbeta_from_t(float t) {
    return clamp01(t); // 0..1
}

static float t_from_klbeta(float beta) {
    float b = beta;
    if (!std::isfinite(b)) b = 0.0f;
    return clamp01(b);
}

static uint64_t steady_now_ms() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

[[maybe_unused]] static bool is_live_input_char(uint32_t cp) {
    if (cp >= static_cast<uint32_t>('0') && cp <= static_cast<uint32_t>('9')) return true;
    return cp == static_cast<uint32_t>('.') ||
           cp == static_cast<uint32_t>('e') ||
           cp == static_cast<uint32_t>('E') ||
           cp == static_cast<uint32_t>('+') ||
           cp == static_cast<uint32_t>('-');
}
} // namespace

Visualizer::Visualizer(const json& config)
    : loss_log_file("checkpoints/loss_history.csv")
    , enabled(false)
    , window_width(1280)
    , window_height(720)
    , window_title("FLUX Model Visualization")
    , fps_limit(60)
    , show_generated_images(true)
    , show_training_progress(true)
    , show_loss_graph(true)
    , graph_history_size(200)
    , image_grid_cols(3)
    , image_grid_rows(1)
    , current_epoch(0)
    , current_total_epochs(0)
    , current_batch(0)
    , current_total_batches(0)
    , current_loss(0.0f)
    , current_avg_loss(0.0f)
    , current_lr(0.0f)
    , current_mse(0.0f)
    , current_kl(0.0f)
    , current_wass(0.0f)
    , current_ent(0.0f)
    , current_mom(0.0f)
    , current_spat(0.0f)
    , current_temp(0.0f)
    , current_timestep(0.0f)
    , current_batch_time_ms(0)
    , current_memory_mb(0)
    , current_allocator_memory_mb(0)
    , current_bps(0.0f)
    , current_params(0)
    , current_grad_norm(0.0f)
    , current_grad_max(0.0f)
    , current_opt_type(0)
    , current_opt_step(0)
    , current_opt_beta1(0.0f)
    , current_opt_beta2(0.0f)
    , current_opt_eps(0.0f)
    , current_opt_weight_decay(0.0f)
    , current_val_has(false)
    , current_val_ok(false)
    , current_val_in_progress(false)
    , current_val_step(0)
    , current_val_items(0)
    , current_val_done(0)
    , current_val_total(0)
    , current_val_recon(0.0f)
    , current_val_kl(0.0f)
    , current_val_align(0.0f)
{
    if (config.contains("visualization")) {
        auto viz = config["visualization"];
        enabled = viz.value("enabled", false);
        window_width = viz.value("window_width", 1280);
        window_height = viz.value("window_height", 720);
        window_title = viz.value("window_title", "FLUX Model Visualization");
        fps_limit = viz.value("fps_limit", 60);
        show_generated_images = viz.value("show_generated_images", true);
        show_training_progress = viz.value("show_training_progress", true);
        show_loss_graph = viz.value("show_loss_graph", true);
        graph_history_size = viz.value("graph_history_size", 200);
        image_grid_cols = viz.value("image_grid_cols", 3);
        image_grid_rows = viz.value("image_grid_rows", 1);

        // CSV loss: optionnel, peut être remplacé pendant le run via AsyncMonitor::setLossLogFile().
        loss_log_file = viz.value("loss_log_file", loss_log_file);

        hide_activation_blocks = viz.value("hide_activation_blocks", false);
        hide_normalisation_blocks = viz.value("hide_normalisation_blocks", false);
        show_detection_overlay_ = viz.value("show_detection_overlay", true);
        show_feature_probe_ = viz.value("show_feature_probe", false);
        architecture_path = viz.value("architecture_path", std::string());
    }
    std::cerr << "[viz] config enabled=" << enabled
              << " window=" << window_width << "x" << window_height
              << " title='" << window_title << "'"
              << " loss_log='" << loss_log_file << "'" << std::endl;
}

void Visualizer::setLossLogFile(const std::string& filepath) {
    if (filepath.empty()) return;
    loss_log_file = filepath;
    std::cerr << "[viz] loss_log_file=" << loss_log_file << std::endl;
}

void Visualizer::setLossLogEnabled(bool enabled) {
    loss_log_enabled_ = enabled;
}

Visualizer::~Visualizer() {
    shutdown();
}

void Visualizer::shutdown() {
    std::atomic_load(&runtime_config_)->requestModelGraph(false);
    if (loss_log_enabled_ && pending_loss_log_flush_ && !loss_log_file.empty()) {
        saveLossHistory(loss_log_file);
        pending_loss_log_flush_ = false;
    }
    if (window) {
        if (window->isOpen()) {
            window->close();
        }
        window.reset();
    }
}

bool Visualizer::initialize() {
    if (!enabled) {
        return false;
    }

    try {
        window = createVizWindow(
            vizgfx::Vector2u(static_cast<unsigned>(window_width), static_cast<unsigned>(window_height)),
            window_title,
            static_cast<unsigned>(fps_limit));
        // Certains environnements laissent le contexte inactif au démarrage si la
        // fenêtre n'a pas encore reçu d'interaction: forcer l'activation.
        window->setActive(true);

        // Logo du programme + splash au lancement (best-effort)
        logo_loaded_ = false;
        try {
            vizgfx::Image logo_img;
            if (logo_img.loadFromFile("logo.png")) {
                const auto sz = logo_img.getSize();
                if (sz.x > 0 && sz.y > 0) {
                    // Certains WM ignorent les icônes trop grandes ou atypiques.
                    // On force une taille standard (64x64) via downscale nearest-neighbor.
                    const unsigned target = 64;
                    vizgfx::Image icon_img;
                    icon_img.resize(vizgfx::Vector2u(target, target));
                    for (unsigned yy = 0; yy < target; ++yy) {
                        for (unsigned xx = 0; xx < target; ++xx) {
                            const unsigned sx = (sz.x > 0) ? (xx * sz.x) / target : 0;
                            const unsigned sy = (sz.y > 0) ? (yy * sz.y) / target : 0;
                            icon_img.setPixel(vizgfx::Vector2u(static_cast<unsigned int>(xx), static_cast<unsigned int>(yy)), logo_img.getPixel(vizgfx::Vector2u(static_cast<unsigned int>(sx), static_cast<unsigned int>(sy))));
                        }
                    }
                    window->setIcon(icon_img);
                }
                if (logo_texture_.loadFromImage(logo_img)) {
                    logo_sprite_.setTexture(logo_texture_, true);
                    logo_sprite_.setColor(vizgfx::Color(255, 255, 255, 220));
                    logo_loaded_ = true;
                    logo_clock_.restart();
                }
            }
        } catch (...) {
            logo_loaded_ = false;
        }

        validation_reward_icon_loaded_ =
            validation_reward_texture_.loadFromFile("assets/viz/validation-reward.png");
        if (validation_reward_icon_loaded_) {
            validation_reward_sprite_.setTexture(validation_reward_texture_, true);
        }
        validation_penalty_icon_loaded_ =
            validation_penalty_texture_.loadFromFile("assets/viz/validation-penalty.png");
        if (validation_penalty_icon_loaded_) {
            validation_penalty_sprite_.setTexture(validation_penalty_texture_, true);
        }

        window->setFramerateLimit(fps_limit);
        syncUIView();
        first_texture_sync_done_ = false;

        // Best-effort: charger une police système pour afficher les métriques.
        // Priorité: Open Sans. Fallback: polices courantes.
        // Si indisponible, on garde un rendu sans texte.
        font_loaded = false;
        const std::vector<std::string> font_candidates = {
            // Open Sans (différentes distros)
            "/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf",
            "/usr/share/fonts/truetype/open-sans/OpenSans.ttf",
            "/usr/share/fonts/truetype/opensans/OpenSans-Regular.ttf",
            "/usr/share/fonts/truetype/opensans/OpenSans.ttf",
            "/usr/share/fonts/truetype/google-fonts/OpenSans-Regular.ttf",
            "/usr/share/fonts/truetype/google-fonts/OpenSans/OpenSans-Regular.ttf",

            // Fallbacks fréquents
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
        };
        for (const auto& path : font_candidates) {
            std::error_code ec;
            if (!std::filesystem::is_regular_file(path, ec)) continue;
            if (font.openFromFile(path)) {
                font_loaded = true;
                break;
            }
        }

        std::cerr << "✓ Visualisation " << vizBackendName() << " initialisée ("
                  << window_width << "x" << window_height << ")" << std::endl;

        // Initialiser et tenter de restaurer le dernier layout sauvegardé.
        initDefaultPanelsIfNeeded();
        try {
            json settings;
            if (loadUISettings(settings)) {
                const json* chosen = nullptr;
                const char* chosen_name = nullptr;

                if (settings.contains("last")) {
                    chosen = &settings["last"];
                    chosen_name = "last";
        std::cerr << "[viz] config enabled=" << enabled
                  << " window=" << window_width << "x" << window_height
                  << " title='" << window_title << "'"
                  << " loss_log='" << loss_log_file << "'" << std::endl;
                } else if (settings.contains("slot1")) {
                    // Compat: certains fichiers peuvent exposer un slot nommé "slot1" au top-level.
                    chosen = &settings["slot1"];
                    chosen_name = "slot1";
                } else if (settings.contains("slots") && settings["slots"].is_object() && settings["slots"].contains("1")) {
        std::cerr << "[viz] loss_log_file=" << loss_log_file << std::endl;
                    chosen = &settings["slots"]["1"];
                    chosen_name = "slots[1]";
                }

            std::cerr << "[viz] initialize skipped (disabled)" << std::endl;
                if (chosen && chosen->is_object()) {
                    // Appliquer la taille de fenêtre AVANT le layout (sinon clamp sur mauvaise taille).
                    try {
                        if (window && chosen->contains("window_width") && chosen->contains("window_height")) {
                            int ww = 0;
                            int hh = 0;
                            try {
                                ww = (*chosen)["window_width"].get<int>();
                                hh = (*chosen)["window_height"].get<int>();
                            } catch (...) {
                                ww = (int)(*chosen)["window_width"].get<double>();
                                hh = (int)(*chosen)["window_height"].get<double>();
                            }

                            ww = std::max(640, ww);
                            hh = std::max(480, hh);
                            window->setSize(vizgfx::Vector2u(static_cast<unsigned>(ww), static_cast<unsigned>(hh)));
                            window_width = ww;
                            window_height = hh;
                            syncUIView();
                        }
                    } catch (...) {
                        // ignore resize errors
                    }

                    const bool ok = applyUILayout(*chosen);
                    if (ok && chosen_name) {
                        std::cerr << "✓ Layout appliqué par défaut (" << chosen_name << ")" << std::endl;
                    }
                }
            }
        } catch (...) {
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors de l'initialisation Viz " << vizBackendName() << ": " << e.what() << std::endl;
        return false;
    }
}

json Visualizer::serializeUILayout() const {
    json out;
    out["version"] = 2;
    out["show_generated_images"] = show_generated_images;
    out["show_training_progress"] = show_training_progress;
    out["show_loss_graph"] = show_loss_graph;
    out["show_prompt_text"] = show_prompt_text_;
    out["show_detection_overlay"] = show_detection_overlay_;
    out["show_feature_probe"] = show_feature_probe_;
    out["heatmap_palette"] = static_cast<int>(heatmap_palette_);
    out["hide_activation_blocks"] = hide_activation_blocks;
    out["hide_normalisation_blocks"] = hide_normalisation_blocks;
    out["blocks_tips_visible_limit"] = std::max(0, blocks_tips_visible_limit_);

    // Taille de fenêtre (utile pour restaurer un slot de layout)
    out["window_width"] = window_width;
    out["window_height"] = window_height;

    json panels = json::array();
    for (size_t i = 0; i < static_cast<size_t>(PanelId::Count); ++i) {
        const PanelId id = static_cast<PanelId>(i);
        const auto& p = panels_[i];
        json jp;
        jp["id"] = static_cast<int>(id);
        jp["title"] = p.title;
        jp["x"] = p.pos.x;
        jp["y"] = p.pos.y;
        jp["w"] = p.size.x;
        jp["h"] = p.size.y;
        jp["visible"] = p.visible;
        jp["allow_drag"] = p.allow_drag;
        panels.push_back(std::move(jp));
    }
    out["panels"] = std::move(panels);
    return out;
}

bool Visualizer::applyUILayout(const json& layout) {
    initDefaultPanelsIfNeeded();

    try {
        if (layout.contains("show_generated_images")) show_generated_images = layout["show_generated_images"].get<bool>();
        if (layout.contains("show_training_progress")) show_training_progress = layout["show_training_progress"].get<bool>();
        if (layout.contains("show_loss_graph")) show_loss_graph = layout["show_loss_graph"].get<bool>();
        if (layout.contains("show_prompt_text")) show_prompt_text_ = layout["show_prompt_text"].get<bool>();
        if (layout.contains("show_feature_probe")) show_feature_probe_ = layout["show_feature_probe"].get<bool>();
        if (layout.contains("show_detection_overlay")) show_detection_overlay_ = layout["show_detection_overlay"].get<bool>();
        if (layout.contains("hide_activation_blocks")) hide_activation_blocks = layout["hide_activation_blocks"].get<bool>();
        if (layout.contains("hide_normalisation_blocks")) hide_normalisation_blocks = layout["hide_normalisation_blocks"].get<bool>();
        if (layout.contains("blocks_tips_visible_limit")) {
            blocks_tips_visible_limit_ = std::max(0, layout["blocks_tips_visible_limit"].get<int>());
        }
        if (layout.contains("heatmap_palette")) {
            const int pid = std::clamp(layout["heatmap_palette"].get<int>(), 0, 3);
            heatmap_palette_ = static_cast<Visualizer::HeatmapPalette>(pid);
        }

        if (!layout.contains("panels") || !layout["panels"].is_array()) {
            // Layout incomplet: appliquer seulement les toggles.
            panels_[static_cast<size_t>(PanelId::Generated)].visible = show_generated_images;
            panels_[static_cast<size_t>(PanelId::Training)].visible = show_training_progress;
            panels_[static_cast<size_t>(PanelId::Graph)].visible = show_loss_graph;
            return true;
        }

        bool output_panel_loaded = false;
        for (const auto& jp : layout["panels"]) {
            if (!jp.is_object()) continue;
            if (!jp.contains("id")) continue;
            const int raw_id = jp["id"].get<int>();
            if (raw_id < 0 || raw_id >= static_cast<int>(PanelId::Count)) continue;
            const size_t idx = static_cast<size_t>(raw_id);
            auto& p = panels_[idx];
            if (raw_id == static_cast<int>(PanelId::Output)) output_panel_loaded = true;

            if (jp.contains("title") && jp["title"].is_string()) p.title = jp["title"].get<std::string>();
            if (jp.contains("x")) p.pos.x = jp["x"].get<float>();
            if (jp.contains("y")) p.pos.y = jp["y"].get<float>();
            if (jp.contains("w")) p.size.x = std::max(kPanelMinW, jp["w"].get<float>());
            if (jp.contains("h")) p.size.y = std::max(kPanelMinH, jp["h"].get<float>());
            if (jp.contains("visible")) p.visible = jp["visible"].get<bool>();
            if (jp.contains("allow_drag")) p.allow_drag = jp["allow_drag"].get<bool>();
        }

        if (!output_panel_loaded) {
            auto& context = panels_[static_cast<size_t>(PanelId::Context)];
            auto& output = panels_[static_cast<size_t>(PanelId::Output)];
            const float gap = 20.0f;
            const float output_width = std::clamp(context.size.x * 0.30f, kPanelMinW, 320.0f);
            context.size.x = std::max(kPanelMinW, context.size.x - output_width - gap);
            output.pos = vizgfx::Vector2f(context.pos.x + context.size.x + gap, context.pos.y);
            output.size = vizgfx::Vector2f(output_width, context.size.y);
            output.visible = true;
        }

        // Visibilité liée aux toggles (reste la source de vérité)
        panels_[static_cast<size_t>(PanelId::Generated)].visible = show_generated_images;
        panels_[static_cast<size_t>(PanelId::Training)].visible = show_training_progress;
        panels_[static_cast<size_t>(PanelId::Graph)].visible = show_loss_graph;

        clampPanelsToWindow();
        rebuildAllTextures();
        return true;
    } catch (...) {
        return false;
    }
}

bool Visualizer::loadUISettings(json& out) const {
    out = json::object();
    try {
        std::ifstream f(kVizUISettingsFile);
        if (!f) return false;
        f >> out;
        if (!out.is_object()) {
            out = json::object();
            return false;
        }
        return true;
    } catch (...) {
        out = json::object();
        return false;
    }
}

bool Visualizer::saveUISettings(const json& s) const {
    try {
        std::ofstream f(kVizUISettingsFile);
        if (!f) return false;
        f << std::setw(2) << s << std::endl;
        return true;
    } catch (...) {
        return false;
    }
}

void Visualizer::saveUILayoutToLast() {
    json settings;
    loadUISettings(settings);
    if (!settings.is_object()) settings = json::object();
    settings["version"] = 1;
    settings["last"] = serializeUILayout();
    saveUISettings(settings);
    std::cerr << "✓ Layout sauvegardé (last) -> " << kVizUISettingsFile << std::endl;
}

void Visualizer::saveUILayoutToSlot(int slot) {
    slot = std::clamp(slot, 0, 9);
    json settings;
    loadUISettings(settings);
    if (!settings.is_object()) settings = json::object();
    settings["version"] = 1;
    if (!settings.contains("slots") || !settings["slots"].is_object()) {
        settings["slots"] = json::object();
    }
    settings["slots"][std::to_string(slot)] = serializeUILayout();
    saveUISettings(settings);
    std::cerr << "✓ Layout sauvegardé (slot " << slot << ") -> " << kVizUISettingsFile << std::endl;
}

void Visualizer::loadUILayoutFromSlot(int slot) {
    slot = std::clamp(slot, 0, 9);
    json settings;
    if (!loadUISettings(settings)) {
        std::cerr << "⚠️  Aucun settings UI trouvé: " << kVizUISettingsFile << std::endl;
        return;
    }
    try {
        if (!settings.contains("slots") || !settings["slots"].is_object()) {
            std::cerr << "⚠️  Aucun slot dans settings UI" << std::endl;
            return;
        }
        const std::string key = std::to_string(slot);
        if (!settings["slots"].contains(key)) {
            std::cerr << "⚠️  Slot UI inexistant: " << slot << std::endl;
            return;
        }

        const json& layout = settings["slots"][key];

        // Appliquer la taille de fenêtre du slot AVANT le layout (sinon clamp sur mauvaise taille).
        try {
            if (window && layout.is_object() && layout.contains("window_width") && layout.contains("window_height")) {
                int ww = 0;
                int hh = 0;
                try {
                    ww = layout["window_width"].get<int>();
                    hh = layout["window_height"].get<int>();
                } catch (...) {
                    // tolerate float/json numbers
                    ww = (int)layout["window_width"].get<double>();
                    hh = (int)layout["window_height"].get<double>();
                }

                ww = std::max(640, ww);
                hh = std::max(480, hh);
                window->setSize(vizgfx::Vector2u(static_cast<unsigned>(ww), static_cast<unsigned>(hh)));
                window_width = ww;
                window_height = hh;
                syncUIView();
            }
        } catch (...) {
            // ignore resize errors
        }

        const bool ok = applyUILayout(layout);
        if (ok) {
            std::cerr << "✓ Layout chargé (slot " << slot << ")" << std::endl;
        } else {
            std::cerr << "⚠️  Échec application layout (slot " << slot << ")" << std::endl;
        }
    } catch (...) {
        std::cerr << "⚠️  Settings UI invalides" << std::endl;
    }
}

void Visualizer::syncUIView() {
    if (!window) return;
    // View UI en coordonnées pixels: (0..w, 0..h)
    // On la force explicitement pour éviter toute déformation au resize.
    const float ww = static_cast<float>(std::max(1, window_width));
    const float hh = static_cast<float>(std::max(1, window_height));
    vizgfx::View v(make_rect(0.f, 0.f, ww, hh));
    v.setViewport(make_rect(0.f, 0.f, 1.f, 1.f));
    window->setView(v);
}

bool Visualizer::isOpen() const {
    return window && window->isOpen();
}

bool Visualizer::isPanelVisible(PanelId id) const {
    return panels_[static_cast<size_t>(id)].visible;
}

namespace {
vizgfx::Color with_alpha(vizgfx::Color c, uint8_t a) {
    c.a = a;
    return c;
}
} // namespace

vizgfx::Color Visualizer::panelAccent(PanelId id) const {
    switch (id) {
        case PanelId::Context: return vizgfx::Color(70, 130, 200, 255);
        case PanelId::Blocks: return vizgfx::Color(160, 110, 210, 255);
        case PanelId::Generated: return vizgfx::Color(70, 170, 170, 255);
        case PanelId::Training: return vizgfx::Color(90, 180, 100, 255);
        case PanelId::Metrics: return vizgfx::Color(210, 150, 70, 255);
        case PanelId::Graph: return vizgfx::Color(200, 90, 90, 255);
        case PanelId::Output: return vizgfx::Color(72, 190, 130, 255);
        default: return vizgfx::Color(140, 140, 150, 255);
    }
}

vizgfx::FloatRect Visualizer::panelRect(PanelId id) const {
    const auto& p = panels_[static_cast<size_t>(id)];
    return make_rect(p.pos.x, p.pos.y, p.size.x, p.size.y);
}

float Visualizer::panelHeaderHeight(PanelId id) const {
    const auto& p=panels_[static_cast<size_t>(id)];
    return std::min(p.size.y,id==PanelId::Blocks ? mimir::viz::blockToolbarLayout(p.pos,p.size.x).height : kPanelTitleH);
}

vizgfx::FloatRect Visualizer::panelTitleRect(PanelId id) const {
    const auto& p = panels_[static_cast<size_t>(id)];
    return make_rect(p.pos.x, p.pos.y, p.size.x, panelHeaderHeight(id));
}

vizgfx::FloatRect Visualizer::panelContentRect(PanelId id) const {
    const auto& p = panels_[static_cast<size_t>(id)];
    const float top = p.pos.y + panelHeaderHeight(id);
    const float inner_w = std::max(0.f, p.size.x - 2.f * kPanelPad);
    const float inner_h = std::max(0.f, p.size.y - panelHeaderHeight(id) - 2.f * kPanelPad);
    return make_rect(p.pos.x + kPanelPad, top + kPanelPad, inner_w, inner_h);
}

void Visualizer::setPanelContentHeight(PanelId id, float content_height) {
    if (id == PanelId::Blocks) return;
    const size_t index = static_cast<size_t>(id);
    const float viewport_height = panelContentRect(id).size.y;
    panel_scroll_max_[index] = std::max(0.0f, content_height - viewport_height);
    panel_scroll_y_[index] = std::clamp(
        panel_scroll_y_[index], 0.0f, panel_scroll_max_[index]);
}

void Visualizer::drawPanelScrollbar(PanelId id) {
    if (!window || id == PanelId::Blocks || !isPanelVisible(id)) return;
    const size_t index = static_cast<size_t>(id);
    panel_scroll_track_rects_[index] = make_rect(0.f, 0.f, 0.f, 0.f);
    panel_scroll_thumb_rects_[index] = make_rect(0.f, 0.f, 0.f, 0.f);
    const float scroll_max = panel_scroll_max_[index];
    if (scroll_max <= 0.0f) return;

    const auto area = panelContentRect(id);
    const float track_w = 10.0f;
    const float tx = area.position.x + std::max(0.0f, area.size.x - track_w - 2.0f);
    const float ty = area.position.y;
    const float th = std::max(0.0f, area.size.y);
    const float content_height = th + scroll_max;
    const float visible_ratio = content_height > 1.0f
        ? std::clamp(th / content_height, 0.05f, 1.0f) : 1.0f;
    const float thumb_h = std::max(24.0f, th * visible_ratio);
    const float travel = std::max(1.0f, th - thumb_h);
    const float ratio = std::clamp(panel_scroll_y_[index] / scroll_max, 0.0f, 1.0f);
    const float thumb_y = ty + ratio * travel;

    panel_scroll_track_rects_[index] = make_rect(tx, ty, track_w, th);
    panel_scroll_thumb_rects_[index] = make_rect(tx, thumb_y, track_w, thumb_h);

    vizgfx::RectangleShape track(vizgfx::Vector2f(track_w, th));
    track.setPosition(vizgfx::Vector2f(tx, ty));
    track.setFillColor(vizgfx::Color(35, 35, 42, 220));
    window->draw(track);

    vizgfx::RectangleShape thumb(vizgfx::Vector2f(track_w, thumb_h));
    thumb.setPosition(vizgfx::Vector2f(tx, thumb_y));
    thumb.setFillColor(panel_scroll_thumb_rects_[index].contains(picker_mouse_) ? vizgfx::Color(126,151,178) : vizgfx::Color(83,98,119));
    window->draw(thumb);
}

vizgfx::FloatRect Visualizer::panelResizeHandleRect(PanelId id) const {
    const auto& p = panels_[static_cast<size_t>(id)];
    const float w = std::max(0.f, p.size.x);
    const float h = std::max(0.f, p.size.y);
    const float hs = std::min(kPanelResizeHandle, std::min(w, h));
    return make_rect(p.pos.x + w - hs, p.pos.y + h - hs, hs, hs);
}

vizgfx::FloatRect Visualizer::panelCloseButtonRect(PanelId id) const {
    const auto tr = panelTitleRect(id);
    const float s = std::max(10.f, std::min(18.f, tr.size.y - 4.f));
    const float x = tr.position.x + tr.size.x - s - 4.f;
    const float y = tr.position.y + (tr.size.y - s) * 0.5f;
    return make_rect(x, y, s, s);
}

std::optional<Visualizer::PanelId> Visualizer::hitTestPanel(const vizgfx::Vector2f& mouse) const {
    // Reverse of the render order: covered controls must never receive a click.
    const PanelId order[] = {PanelId::Graph, PanelId::Metrics, PanelId::Training,
                            PanelId::Generated, PanelId::Blocks, PanelId::Output, PanelId::Context};
    for (PanelId id : order) {
        const auto& p = panels_[static_cast<size_t>(id)];
        if (isPanelVisible(id) && make_rect(p.pos.x,p.pos.y,p.size.x,p.size.y).contains(mouse)) return id;
    }
    return std::nullopt;
}

std::optional<Visualizer::PanelId> Visualizer::hitTestPanelTitle(const vizgfx::Vector2f& mouse) const {
    const auto id = hitTestPanel(mouse);
    if (id && panels_[static_cast<size_t>(*id)].allow_drag && panelTitleRect(*id).contains(mouse)) return id;
    return std::nullopt;
}

std::optional<Visualizer::PanelId> Visualizer::hitTestPanelResizeHandle(const vizgfx::Vector2f& mouse) const {
    const auto id = hitTestPanel(mouse);
    if (id && panels_[static_cast<size_t>(*id)].allow_drag && panelResizeHandleRect(*id).contains(mouse)) return id;
    return std::nullopt;
}

std::optional<Visualizer::PanelId> Visualizer::hitTestPanelCloseButton(const vizgfx::Vector2f& mouse) const {
    const auto id = hitTestPanel(mouse);
    if (id && panelCloseButtonRect(*id).contains(mouse)) return id;
    return std::nullopt;
}

void Visualizer::initCursorsIfNeeded() {
    if (cursors_loaded_) return;
    // Best-effort: certains backends peuvent refuser certains curseurs.
    cursor_arrow_ = vizgfx::Cursor::createFromSystem(vizgfx::Cursor::Type::Arrow);
    cursor_hand_ = vizgfx::Cursor::createFromSystem(vizgfx::Cursor::Type::Hand);
    cursor_cross_ = vizgfx::Cursor::createFromSystem(vizgfx::Cursor::Type::Cross);
    // Diagonal resize (NW-SE) pour handle bottom-right.
    cursor_resize_ = vizgfx::Cursor::createFromSystem(vizgfx::Cursor::Type::SizeTopLeftBottomRight);

    cursor_ok_arrow_ = cursor_arrow_.has_value();
    cursor_ok_hand_ = cursor_hand_.has_value();
    cursor_ok_cross_ = cursor_cross_.has_value();
    cursor_ok_resize_ = cursor_resize_.has_value();
    cursors_loaded_ = true;
}

void Visualizer::setCursor(CursorKind kind) {
    if (!window) return;
    if (cursor_kind_ == kind) return;
    initCursorsIfNeeded();

    const vizgfx::Cursor* c = nullptr;
    if (kind == CursorKind::Hand && cursor_ok_hand_ && cursor_hand_.has_value()) c = &(*cursor_hand_);
    else if (kind == CursorKind::Cross && cursor_ok_cross_ && cursor_cross_.has_value()) c = &(*cursor_cross_);
    else if (kind == CursorKind::Resize && cursor_ok_resize_ && cursor_resize_.has_value()) c = &(*cursor_resize_);
    else if (cursor_ok_arrow_ && cursor_arrow_.has_value()) c = &(*cursor_arrow_);

    if (c) {
        window->setMouseCursor(*c, static_cast<int>(kind));
    }
    cursor_kind_ = kind;
}

void Visualizer::clampPanelsToWindow() {
    const float min_margin = 4.f;
    for (auto& p : panels_) {
        if (p.size.x <= 0.f || p.size.y <= 0.f) continue;
        const float max_x = std::max(min_margin, static_cast<float>(window_width) - p.size.x - min_margin);
        const float max_y = std::max(min_margin, static_cast<float>(window_height) - p.size.y - min_margin);
        p.pos.x = std::clamp(p.pos.x, min_margin, max_x);
        p.pos.y = std::clamp(p.pos.y, min_margin, max_y);
    }
}

void Visualizer::initDefaultPanelsIfNeeded() {
    if (panels_initialized_) return;
    panels_initialized_ = true;

    const float margin = 6.f;
    const float right_w = std::clamp(float(window_width) * .30f, 320.f, 520.f);
    const float min_left_w = 520.f;

    const float win_w = static_cast<float>(window_width);
    const float win_h = static_cast<float>(window_height);

    const float left_w = std::clamp(win_w - right_w - 3.f * margin, min_left_w, std::max(min_left_w, win_w - 2.f * margin));
    const float right_x = margin + left_w + margin;
    const float right_safe_w = std::max(320.f, std::min(right_w, win_w - right_x - margin));

    const float usable_h = std::max(420.f, win_h - 2.f * margin);
    const float top_h = std::clamp(usable_h * .34f, 260.f, 320.f);
    const float bottom_h = std::clamp(usable_h * .25f, 140.f, 250.f);
    const float middle_h = std::max(140.f, usable_h - top_h - bottom_h - 2.f * margin);
    const float output_w = std::clamp(left_w * .34f, kPanelMinW, 320.f);
    const float context_w = std::max(kPanelMinW, left_w - output_w - margin);
    panels_[static_cast<size_t>(PanelId::Context)] = Panel{{margin, margin}, {context_w, top_h}, "Contexte", true, true};
    panels_[static_cast<size_t>(PanelId::Output)] = Panel{{margin + context_w + margin, margin}, {output_w, top_h}, "Sortie", true, true};
    panels_[static_cast<size_t>(PanelId::Blocks)] = Panel{{margin, margin + top_h + margin}, {left_w, middle_h}, "Tenseurs / Layers", true, true};
    panels_[static_cast<size_t>(PanelId::Generated)] = Panel{{margin, margin + top_h + middle_h + 2.f * margin}, {left_w, bottom_h}, "Images / Validation", true, true};
    const float training_h=90.f;
    const float graph_h=std::clamp(usable_h*.27f,160.f,260.f);
    panels_[static_cast<size_t>(PanelId::Training)] = Panel{{right_x, margin}, {right_safe_w, training_h}, "Entraînement", true, true};
    panels_[static_cast<size_t>(PanelId::Metrics)] = Panel{{right_x, margin + training_h + margin}, {right_safe_w, std::max(160.f, usable_h-training_h-graph_h-2.f*margin)}, "Propriétés / Métriques", true, true};
    panels_[static_cast<size_t>(PanelId::Graph)] = Panel{{right_x, win_h-margin-graph_h}, {right_safe_w, graph_h}, "Courbes / Loss", true, true};

    clampPanelsToWindow();
}

void Visualizer::drawPanelChrome(PanelId id) {
    if (!window) return;
    if (!isPanelVisible(id)) return;
    const auto& p = panels_[static_cast<size_t>(id)];
    if (p.size.x <= 1.f || p.size.y <= 1.f) return;

    vizgfx::RectangleShape bg(p.size);
    bg.setPosition(p.pos);
    bg.setFillColor(mimir::viz::theme::panel);
    bg.setOutlineColor(mimir::viz::theme::border);
    bg.setOutlineThickness(1);
    window->draw(bg);

    const float th = panelHeaderHeight(id);
    vizgfx::RectangleShape title(vizgfx::Vector2f(p.size.x, th));
    title.setPosition(p.pos);
    title.setFillColor(mimir::viz::theme::header);
    window->draw(title);

    if (id == PanelId::Blocks) {
        last_blocks_mode_box_.reset();
        last_blocks_smooth_box_.reset();
        last_blocks_palette_box_.reset();
        last_blocks_hide_act_box_.reset();
        last_blocks_hide_norm_box_.reset();
        last_blocks_tips_limit_box_.reset();
    }
    if (id == PanelId::Context) last_detection_toggle_box_.reset();
    if (id == PanelId::Output) {
        last_detection_threshold_track_.reset();
        last_detection_threshold_thumb_.reset();
    }

    if (font_loaded && !p.title.empty()) {
        vizgfx::Text t(font);
        t.setFont(font);
        t.setCharacterSize(14);
        t.setFillColor(vizgfx::Color(235, 235, 240));
        t.setPosition(vizgfx::Vector2f(p.pos.x + 8.f, p.pos.y + 2.f));
        t.setString(vizgfx::String::fromUtf8(p.title.begin(), p.title.end()));
        window->draw(t);

        // Badge de mode de rendu pour Blocks/Layers (heatmap vs réel).
        if (id == PanelId::Blocks) {
            const bool real_mode = !heatmap_mode_;
            const std::string badge = real_mode ? "REEL" : "HEATMAP";

            const vizgfx::Color badge_fill = real_mode
                ? vizgfx::Color(47, 71, 88)
                : vizgfx::Color(64, 58, 48);
            const vizgfx::Color badge_outline = real_mode
                ? vizgfx::Color(87, 122, 147)
                : vizgfx::Color(130, 111, 84);

            const bool smooth_on = smooth_layer_block_previews_;
            const std::string smooth_badge = smooth_on ? "LISSAGE ON" : "LISSAGE OFF";
            const vizgfx::Color smooth_fill = smooth_on
                ? vizgfx::Color(47, 71, 88)
                : vizgfx::Color(49, 56, 67);
            const vizgfx::Color smooth_outline = smooth_on
                ? vizgfx::Color(87, 122, 147)
                : vizgfx::Color(77, 85, 98);

            const std::string pal_badge = std::string("PAL ") + heatmap_palette_name(static_cast<int>(heatmap_palette_));
            const vizgfx::Color pal_fill = vizgfx::Color(54, 59, 76);
            const vizgfx::Color pal_outline = vizgfx::Color(90, 98, 126);

            const auto layout=mimir::viz::blockToolbarLayout(p.pos,p.size.x);
            const float badge_h=18.f;
            const float mode_w=layout.controls[0].size.x,mode_x=layout.controls[0].position.x,mode_y=layout.controls[0].position.y;
            const float smooth_w=layout.controls[1].size.x,smooth_x=layout.controls[1].position.x,smooth_y=layout.controls[1].position.y;
            const float pal_w=layout.controls[2].size.x,pal_x=layout.controls[2].position.x,pal_y=layout.controls[2].position.y;
            const float filt_act_w=layout.controls[3].size.x,act_x=layout.controls[3].position.x,act_y=layout.controls[3].position.y;
            const float filt_norm_w=layout.controls[4].size.x,norm_x=layout.controls[4].position.x,norm_y=layout.controls[4].position.y;
            const float tips_w=layout.controls[5].size.x,tips_x=layout.controls[5].position.x,tips_y=layout.controls[5].position.y;

            last_blocks_mode_box_ = make_rect(mode_x, mode_y, mode_w, badge_h);
            last_blocks_smooth_box_ = make_rect(smooth_x, smooth_y, smooth_w, badge_h);
            last_blocks_palette_box_ = make_rect(pal_x, pal_y, pal_w, badge_h);
            vizgfx::RectangleShape b(vizgfx::Vector2f(mode_w, badge_h));
            b.setPosition(vizgfx::Vector2f(mode_x, mode_y));
            b.setFillColor(badge_fill);
            b.setOutlineColor(badge_outline);
            b.setOutlineThickness(1.f);
            window->draw(b);

            vizgfx::Text bt(font);
            bt.setFont(font);
            bt.setCharacterSize(12);
            bt.setFillColor(vizgfx::Color(245, 245, 248));
            bt.setPosition(vizgfx::Vector2f(mode_x + 7.f, mode_y - 1.f));
            bt.setString(vizgfx::String::fromUtf8(badge.begin(), badge.end()));
            window->draw(bt);

            vizgfx::RectangleShape sb(vizgfx::Vector2f(smooth_w, badge_h));
            sb.setPosition(vizgfx::Vector2f(smooth_x, smooth_y));
            sb.setFillColor(smooth_fill);
            sb.setOutlineColor(smooth_outline);
            sb.setOutlineThickness(1.f);
            window->draw(sb);

            vizgfx::Text st(font);
            st.setFont(font);
            st.setCharacterSize(12);
            st.setFillColor(vizgfx::Color(245, 245, 248));
            st.setPosition(vizgfx::Vector2f(smooth_x + 6.f, smooth_y - 1.f));
            st.setString(vizgfx::String::fromUtf8(smooth_badge.begin(), smooth_badge.end()));
            window->draw(st);

            vizgfx::RectangleShape pb(vizgfx::Vector2f(pal_w, badge_h));
            pb.setPosition(vizgfx::Vector2f(pal_x, pal_y));
            pb.setFillColor(pal_fill);
            pb.setOutlineColor(pal_outline);
            pb.setOutlineThickness(1.f);
            window->draw(pb);

            vizgfx::Text pt(font);
            pt.setFont(font);
            pt.setCharacterSize(12);
            pt.setFillColor(vizgfx::Color(245, 245, 248));
            pt.setPosition(vizgfx::Vector2f(pal_x + 6.f, pal_y - 1.f));
            pt.setString(vizgfx::String::fromUtf8(pal_badge.begin(), pal_badge.end()));
            window->draw(pt);

            auto draw_filter_badge = [&](float x, float y, float w, const std::string& label, bool hidden,
                                         std::optional<vizgfx::FloatRect>& out_rect) {
                const vizgfx::Color fill = hidden ? vizgfx::Color(49, 56, 67)
                                              : vizgfx::Color(47, 71, 88);
                const vizgfx::Color outline = hidden ? vizgfx::Color(77, 85, 98)
                                                 : vizgfx::Color(87, 122, 147);
                vizgfx::RectangleShape fb(vizgfx::Vector2f(w, badge_h));
                fb.setPosition(vizgfx::Vector2f(x, y));
                fb.setFillColor(fill);
                fb.setOutlineColor(outline);
                fb.setOutlineThickness(1.f);
                window->draw(fb);

                vizgfx::Text ft(font);
                ft.setFont(font);
                ft.setCharacterSize(12);
                ft.setFillColor(vizgfx::Color(245, 245, 248));
                ft.setPosition(vizgfx::Vector2f(x + 6.f, y - 1.f));
                ft.setString(vizgfx::String::fromUtf8(label.begin(), label.end()));
                window->draw(ft);

                out_rect = make_rect(x, y, w, badge_h);
            };

            draw_filter_badge(act_x, act_y, filt_act_w,
                              hide_activation_blocks ? "ACT OFF" : "ACT ON",
                              hide_activation_blocks,
                              last_blocks_hide_act_box_);
            draw_filter_badge(norm_x, norm_y, filt_norm_w,
                              hide_normalisation_blocks ? "NORM OFF" : "NORM ON",
                              hide_normalisation_blocks,
                              last_blocks_hide_norm_box_);

            {
                const vizgfx::Color tips_fill = blocks_tips_limit_editing_
                    ? vizgfx::Color(76, 98, 146, 220)
                    : vizgfx::Color(74, 74, 92, 220);
                const vizgfx::Color tips_outline = blocks_tips_limit_editing_
                    ? vizgfx::Color(166, 208, 255, 235)
                    : vizgfx::Color(168, 168, 188, 235);

                vizgfx::RectangleShape tb(vizgfx::Vector2f(tips_w, badge_h));
                tb.setPosition(vizgfx::Vector2f(tips_x, tips_y));
                tb.setFillColor(tips_fill);
                tb.setOutlineColor(tips_outline);
                tb.setOutlineThickness(1.f);
                window->draw(tb);

                std::string tips_value;
                if (blocks_tips_limit_editing_) {
                    tips_value = blocks_tips_limit_input_;
                } else if (blocks_tips_visible_limit_ > 0) {
                    tips_value = std::to_string(blocks_tips_visible_limit_);
                } else {
                    tips_value = "ALL";
                }
                if (tips_value.empty()) tips_value = "_";

                const std::string tips_badge = "TIPS " + tips_value;

                vizgfx::Text tt(font);
                tt.setFont(font);
                tt.setCharacterSize(12);
                tt.setFillColor(vizgfx::Color(245, 245, 248));
                tt.setPosition(vizgfx::Vector2f(tips_x + 6.f, tips_y - 1.f));
                tt.setString(vizgfx::String::fromUtf8(tips_badge.begin(), tips_badge.end()));
                window->draw(tt);

                last_blocks_tips_limit_box_ = make_rect(tips_x, tips_y, tips_w, badge_h);
            }
        }
        if (id == PanelId::Context) {
            const float badge_w = 76.f;
            const float badge_h = std::max(14.f, th - 8.f);
            const float badge_x = p.pos.x + std::max(8.f, p.size.x - badge_w - 28.f);
            const float badge_y = p.pos.y + (th - badge_h) * 0.5f;
            const vizgfx::Color fill = show_detection_overlay_
                ? vizgfx::Color(42, 124, 86, 225) : vizgfx::Color(49, 56, 67);
            vizgfx::RectangleShape badge_shape(vizgfx::Vector2f(badge_w, badge_h));
            badge_shape.setPosition(vizgfx::Vector2f(badge_x, badge_y));
            badge_shape.setFillColor(fill);
            badge_shape.setOutlineColor(vizgfx::Color(165, 225, 195, 235));
            badge_shape.setOutlineThickness(1.f);
            window->draw(badge_shape);

            const std::string badge_text = show_detection_overlay_ ? "BOX ON" : "BOX OFF";
            vizgfx::Text badge_label(font);
            badge_label.setCharacterSize(12);
            badge_label.setFillColor(vizgfx::Color(245, 245, 248));
            badge_label.setPosition(vizgfx::Vector2f(badge_x + 8.f, badge_y - 1.f));
            badge_label.setString(vizgfx::String::fromUtf8(badge_text.begin(), badge_text.end()));
            window->draw(badge_label);
            last_detection_toggle_box_ = make_rect(badge_x, badge_y, badge_w, badge_h);
        }
    }

    if (id == PanelId::Output) {
        const float track_x = p.pos.x + 92.0f;
        const float track_w = std::max(36.0f, p.size.x - 128.0f);
        const float track_y = p.pos.y + th * 0.5f - 2.0f;
        constexpr int heatmap_segments = 32;
        const float segment_w = track_w / static_cast<float>(heatmap_segments);
        for (int segment = 0; segment < heatmap_segments; ++segment) {
            const float score = (static_cast<float>(segment) + 0.5f) /
                static_cast<float>(heatmap_segments);
            vizgfx::RectangleShape heatmap_segment(
                vizgfx::Vector2f(segment_w + 0.5f, 4.0f));
            heatmap_segment.setPosition(vizgfx::Vector2f(
                track_x + segment_w * static_cast<float>(segment), track_y));
            heatmap_segment.setFillColor(detection_heatmap_color(score));
            window->draw(heatmap_segment);
        }

        const float threshold = std::clamp(detection_score_threshold_, 0.0f, 1.0f);
        const float thumb_x = track_x + track_w * threshold;
        vizgfx::RectangleShape thumb(vizgfx::Vector2f(12.f, 12.f));
        thumb.setPosition(vizgfx::Vector2f(thumb_x - 6.0f, track_y - 4.0f));
        thumb.setFillColor(vizgfx::Color(230, 250, 238, 255));
        thumb.setOutlineColor(detection_heatmap_color(threshold));
        thumb.setOutlineThickness(1.0f);
        window->draw(thumb);

        if (font_loaded) {
            const std::string confidence = "CONF " +
                std::to_string(static_cast<int>(std::lround(threshold * 100.0f))) + "%";
            vizgfx::Text text(font);
            text.setCharacterSize(11);
            text.setFillColor(vizgfx::Color(235, 245, 240));
            text.setPosition(vizgfx::Vector2f(p.pos.x + 48.0f, p.pos.y + 1.0f));
            text.setString(vizgfx::String::fromUtf8(confidence.begin(), confidence.end()));
            window->draw(text);
        }
        last_detection_threshold_track_ = make_rect(track_x - 4.0f, track_y - 7.0f,
                                                     track_w + 8.0f, 18.0f);
        last_detection_threshold_thumb_ = make_rect(thumb_x - 8.0f, track_y - 6.0f,
                                                     16.0f, 16.0f);
    }

    // Poignée de redimensionnement (bas-droite)
    {
        const auto r = panelResizeHandleRect(id);
        vizgfx::RectangleShape h(vizgfx::Vector2f(r.size.x, r.size.y));
        h.setPosition(vizgfx::Vector2f(r.position.x, r.position.y));
        h.setFillColor(vizgfx::Color(80, 80, 95, 210));
        window->draw(h);
    }

    // Discreet close action; destructive color appears only on hover.
    {
        const auto r=panelCloseButtonRect(id);
        const bool hovered=r.contains(picker_mouse_);
        mimir::viz::ui::roundedRect(*window,r,hovered?vizgfx::Color(121,56,64):mimir::viz::theme::header,3);
        vizgfx::VertexArray cross(vizgfx::PrimitiveType::LineStrip,2);
        for(int direction:{-1,1}) {
            const auto center=vizgfx::Vector2f(r.position.x+r.size.x*.5f,r.position.y+r.size.y*.5f);
            cross[0]={{center.x-3,center.y-3*direction},vizgfx::Color(170,182,199)};
            cross[1]={{center.x+3,center.y+3*direction},vizgfx::Color(170,182,199)};window->draw(cross);
        }
    }
}

// Lua-facing mailbox: no Lua VM or graphics objects cross thread boundaries.
void Visualizer::configureScript(const json& config) {
    if (!config.is_object()) throw std::invalid_argument("Viz.configure expects an object");
    for (const char* field : {"panels", "controls"}) {
        if (!config.contains(field)) continue;
        if (!config[field].is_array() || config[field].size() > 64)
            throw std::invalid_argument(std::string(field) + " must be an array (max 64)");
        std::unordered_set<std::string> ids;
        for (const auto& item : config[field]) {
            if (!item.is_object()) throw std::invalid_argument("scene item must be an object");
            for (const char* key : {"x", "y", "w", "h"}) {
                if (item.contains(key) && (!item[key].is_number() ||
                    !std::isfinite(item[key].get<double>()) || std::abs(item[key].get<double>()) > 16384))
                    throw std::invalid_argument("invalid scene geometry");
            }
            for (const char* key : {"w", "h"})
                if (item.contains(key) && item[key].get<float>() <= 0)
                    throw std::invalid_argument("scene size must be positive");
            if (std::string(field) == "panels") {
                if (!item.contains("id") || !item["id"].is_number() ||
                    !std::isfinite(item["id"].get<double>()) ||
                    item["id"].get<double>() != std::floor(item["id"].get<double>()) ||
                    item["id"].get<double>() < 0 || item["id"].get<double>() >= 7)
                    throw std::invalid_argument("panel id must be 0..6");
                if (item.contains("visible") && !item["visible"].is_boolean())
                    throw std::invalid_argument("visible must be boolean");
            } else {
                if (!item.contains("id") || !item["id"].is_string() || item["id"].get<std::string>().empty() ||
                    !ids.insert(item["id"].get<std::string>()).second)
                    throw std::invalid_argument("control ids must be unique nonempty strings");
            }
            if (item.contains("title") && !item["title"].is_string()) throw std::invalid_argument("title must be string");
            if (item.contains("label") && !item["label"].is_string()) throw std::invalid_argument("label must be string");
        }
    }
    if (config.contains("image_size") && (!config["image_size"].is_number() ||
        !std::isfinite(config["image_size"].get<double>()) ||
        config["image_size"].get<double>() != std::floor(config["image_size"].get<double>()) ||
        config["image_size"].get<double>() < 32 || config["image_size"].get<double>() > 1024))
        throw std::invalid_argument("image_size must be 32..1024 pixels");
    if (config.contains("events") && !config["events"].is_boolean()) throw std::invalid_argument("events must be boolean");
    if (config.contains("help") && !config["help"].is_string()) throw std::invalid_argument("help must be string");
    std::lock_guard<std::mutex> lock(script_mutex_);
    if (!script_pending_) script_pending_ = json::object();
    json next = config;
    if (next.contains("panels") && script_pending_->contains("panels")) {
        json merged = (*script_pending_)["panels"];
        for (const auto& patch : next["panels"]) {
            bool found = false;
            for (auto& panel : merged) if (panel["id"] == patch["id"]) {
                panel.update(patch); found = true; break;
            }
            if (!found) merged.push_back(patch);
        }
        next["panels"] = std::move(merged);
    }
    script_pending_->update(next);
}

json Visualizer::pollScriptEvents() {
    std::lock_guard<std::mutex> lock(script_mutex_);
    json result = json::array();
    result.swap(script_events_);
    return result;
}

void Visualizer::emitScriptEvent(const json& event) {
    std::lock_guard<std::mutex> lock(script_mutex_);
    if (script_events_.size() >= 256) script_events_.erase(script_events_.begin());
    script_events_.push_back(event);
}

void Visualizer::applyScriptScene() {
    std::optional<json> pending;
    {
        std::lock_guard<std::mutex> lock(script_mutex_);
        pending.swap(script_pending_);
    }
    if (!pending) return;
    initDefaultPanelsIfNeeded();
    script_scene_.update(*pending);
    if (pending->contains("panels")) for (const auto& item : (*pending)["panels"]) {
        auto& p = panels_[item["id"].get<size_t>()];
        p.pos = {item.value("x", p.pos.x), item.value("y", p.pos.y)};
        p.size = {item.value("w", p.size.x), item.value("h", p.size.y)};
        p.visible = item.value("visible", p.visible);
        p.title = item.value("title", p.title);
    }
    show_generated_images = panels_[2].visible;
    show_training_progress = panels_[3].visible;
    show_loss_graph = panels_[5].visible;
    emitScriptEvent({{"type", "configured"}, {"width", window_width}, {"height", window_height}});
}

void Visualizer::renderScriptControls() {
    if (!font_loaded || !script_scene_.contains("controls")) return;
    for (const auto& c : script_scene_["controls"]) {
        const float x = c.value("x", 12.f), y = c.value("y", 12.f);
        vizgfx::RectangleShape box({c.value("w", 150.f), c.value("h", 36.f)});
        box.setPosition({x, y});
        box.setFillColor(vizgfx::Color(37, 65, 83));
        box.setOutlineColor(vizgfx::Color(80, 180, 200));
        box.setOutlineThickness(1.f);
        mimir::viz::ui::buttonSurface(*window,{{x,y},box.getSize()},mimir::viz::theme::control,vizgfx::FloatRect({x,y},box.getSize()).contains(picker_mouse_));
        vizgfx::Text t(font);
        t.setCharacterSize(14);
        t.setPosition({x + 10.f, y + 8.f});
        const auto label = c.value("label", c["id"].get<std::string>());
        t.setString(vizgfx::String::fromUtf8(label.begin(), label.end()));
        t.setFillColor(vizgfx::Color(235, 245, 250));
        window->draw(t);
    }
}

void Visualizer::renderDebugJsonDialog() {
    if (!font_loaded) return;
    const float width = std::max(180.f, std::min(760.f, static_cast<float>(window_width)-40.f));
    const float left = std::max(10.f, (static_cast<float>(window_width)-width)/2.f);
    const float top = std::max(20.f, (static_cast<float>(window_height)-280.f)/2.f);
    auto box = [&](float x, float y, float w, float h, vizgfx::Color color) {
        vizgfx::RectangleShape shape({w,h}); shape.setPosition({x,y}); shape.setFillColor(color); window->draw(shape);
    };
    auto text = [&](float x, float y, const std::string& value, vizgfx::Color color) {
        vizgfx::Text label(font); label.setCharacterSize(14); label.setPosition({x,y}); label.setFillColor(color);
        const auto limit = static_cast<size_t>(std::max(1.f, (width-32.f)/8.5f));
        const auto shown = value.size() > limit ? "..." + value.substr(value.size()-limit) : value;
        label.setString(vizgfx::String::fromUtf8(shown.begin(),shown.end())); window->draw(label);
    };
    box(0,0,static_cast<float>(window_width),static_cast<float>(window_height),vizgfx::Color(0,0,0,185));
    box(left,top,width,280.f,vizgfx::Color(22,29,39));
    text(left+12,top+12,"EXPORT DEBUGJSON",vizgfx::Color(100,215,210));
    text(left+12,top+40,"Tab : changer de champ. Entree : exporter. Esc : annuler.",vizgfx::Color(170,185,200));
    auto& form = debug_json_dialog_;
    for (int i=0;i<2;++i) {
        const float y = top+75.f+i*40.f;
        box(left+8,y,width-16.f,34.f,form.selected==i ? vizgfx::Color(42,75,95) : vizgfx::Color(30,40,55));
        text(left+14,y+7,(i==0 ? "Dossier : " + form.directory : "Fichier : " + form.filename) + (form.selected==i ? "_" : ""),vizgfx::Color(230,240,246));
    }
    text(left+12,top+175,"La saisie remplace le champ selectionne.",vizgfx::Color(170,185,200));
    text(left+12,top+200,"L'entrainement continue pendant la saisie.",vizgfx::Color(170,185,200));
    text(left+12,top+240,form.message,vizgfx::Color(245,185,110));
}

void Visualizer::renderModelGraphOverlay() {
    model_graph_hits_.clear();
    if(!font_loaded) return;
    const auto config=std::atomic_load(&runtime_config_);
    config->requestModelGraph(true); // also handles rebinding to a new model
    model_graph_display_=config->modelGraph();
    const float width=float(window_width),height=float(window_height);
    const vizgfx::Color white(230,240,246),muted(160,180,200),accent(100,215,210);
    auto box=[&](float x,float y,float w,float h,vizgfx::Color color) {
        vizgfx::RectangleShape r({w,h});r.setPosition({x,y});r.setFillColor(color);window->draw(r);
    };
    auto text=[&](float x,float y,std::string value,vizgfx::Color color,unsigned size=13) {
        vizgfx::Text t(font);t.setCharacterSize(size);t.setPosition({x,y});t.setFillColor(color);
        t.setString(vizgfx::String::fromUtf8(value.begin(),value.end()));window->draw(t);
    };
    box(0,0,width,height,vizgfx::Color(16,23,32));
    text(20,12,"GRAPHE DU MODELE / PLANNER   [N / Esc] Fermer",accent,17);
    text(20,38,"Fleches / molette : parcourir | R : origine | Clic ligne : details | Rails colores : branches / fusions",muted);
    if(!model_graph_display_) {
        text(20,72,"Aucun modele attache. En attente d'un instantane du modele.",white);return;
    }
    const auto& graph=*model_graph_display_;
    const auto& nodes=graph.at("nodes");
    const bool planned=graph.value("planned",false);
    text(20,64,"Planner : "+graph.value("mode",std::string("inconnu"))+
        " | "+(graph.value("training",false)?"training":"inference")+
        " | fusion : "+(graph.value("fusion",false)?"on":"off")+
        " | reuse : "+(graph.value("buffer_reuse",false)?"on":"off")+
        " | layers : "+std::to_string(nodes.size()),white);
    if(planned) {
        const auto& plan=graph.at("plan");const auto& stats=plan.at("stats");
        text(20,86,"Estimations octets : tenseurs "+stats.at("tensor_logical_bytes").dump()+
            " | pic "+stats.at("tensor_physical_peak").dump()+
            " | reutilises "+stats.at("buffer_reuse_saved").dump()+
            " | transferts "+stats.at("transfer_bytes").dump()+
            " | scratch "+stats.at("scratch_peak").dump(),muted);
    } else text(20,86,"Plan indisponible : topologie seule. Les placements ne sont pas mesures.",muted);
    const float top=114.f,bottom=std::max(top+1.f,height-174.f);
    const float view_h=bottom-top,view_w=std::max(1.f,width-40.f);
    const auto layout=Mimir::Viz::modelGraphLanes(graph);
    const float labels_x=layout.labelsX();
    const float extent_x=labels_x+520.f,extent_y=float(nodes.size())*Mimir::Viz::ModelGraphLanes::row+30.f;
    model_graph_max_x_=std::max(0.f,extent_x-view_w+20.f);
    model_graph_max_y_=std::max(0.f,extent_y-view_h+20.f);
    model_graph_x_=std::clamp(model_graph_x_,0.f,model_graph_max_x_);
    model_graph_y_=std::clamp(model_graph_y_,0.f,model_graph_max_y_);
    const auto old=window->getView();
    vizgfx::View view(make_rect(model_graph_x_,model_graph_y_,view_w,view_h));
    view.setViewport(make_rect(20.f/std::max(1.f,width),top/std::max(1.f,height),
        view_w/std::max(1.f,width),view_h/std::max(1.f,height)));
    window->setView(view);
    if(!nodes.empty())model_graph_selected_=std::min(model_graph_selected_,nodes.size()-1);
    // Row highlights stay behind the colored branch rails and merge markers.
    for(size_t i=0;i<nodes.size();++i) {
        const float y=layout.position(i).y-25.f;
        if(y+56<model_graph_y_ || y>model_graph_y_+view_h)continue;
        box(0,y,extent_x,56.f,i==model_graph_selected_?vizgfx::Color(36,48,65):
            (i%2?vizgfx::Color(19,27,38):vizgfx::Color(16,23,32)));
    }
    Mimir::Viz::drawModelGraphLanes(*window,layout,model_graph_selected_,model_graph_y_,model_graph_y_+view_h);
    for(size_t i=0;i<nodes.size();++i) {
        const auto& n=nodes[i];const float y=layout.position(i).y-25.f;
        if(y+56<model_graph_y_ || y>model_graph_y_+view_h)continue;
        const auto color=Mimir::Viz::graphLaneColor(layout.lanes[i]);
        auto clip=[](std::string v,size_t limit){return v.size()>limit?v.substr(0,limit-3)+"...":v;};
        text(labels_x,y+3,std::to_string(i)+"  "+clip(n.at("name"),44),white,14);
        text(labels_x,y+25,clip(n.at("type"),22)+"  |  "+n.at("runtime").get<std::string>(),color,12);
        std::string badges="params "+n.at("parameters").dump();
        if(n.value("graph_fused",false))badges+="  | fusion";
        if(n.value("kernel_fused",false))badges+="  | kernel";
        if(n.value("skip",false))badges+="  | skip";
        text(labels_x+270.f,y+27,badges,muted,11);
        const auto rect=make_rect(20-model_graph_x_,top+y-model_graph_y_,extent_x,56.f);
        if(auto hit=rect.findIntersection(make_rect(20,top,view_w,view_h)))model_graph_hits_.push_back({*hit,i});
    }
    window->setView(old);
    box(0,bottom,width,height-bottom,vizgfx::Color(22,32,44));
    text(20,bottom+6,"Placement prevu par le planner ; ne constitue pas une trace des kernels executes.",muted,12);
    if(!nodes.empty()) {
        const auto& n=nodes[model_graph_selected_];
        text(20,bottom+28,n.at("name").get<std::string>()+" : "+n.at("type").get<std::string>(),white);
        text(20,bottom+49,"Entrees : "+n.at("inputs").dump()+" -> "+n.at("output").get<std::string>(),white,12);
        text(20,bottom+70,"Runtime : "+n.at("runtime").get<std::string>()+" | capacite : "+n.at("capability").get<std::string>()+
            " | skip : "+(n.value("skip",false)?"oui":"non")+
            " | fusion vers : "+(n.contains("fused_into")?n.at("fused_into").dump():"aucune"),accent,12);
        text(20,bottom+91,"Shape : "+(n.contains("shape")?n.at("shape").dump():"inconnue")+
            " | octets : "+(n.contains("bytes")?n.at("bytes").dump():"inconnus")+
            " | scratch : "+std::to_string(n.value("scratch_bytes",size_t(0))),muted,12);
        text(20,bottom+113,"Entrees externes : "+std::to_string(graph.at("external_inputs").size())+
            " | mise a jour au prochain forward actif (max 2/s)",muted,12);
    } else text(20,bottom+32,"Le modele ne contient aucun layer.",white);
}

void Visualizer::renderConfigurationOverlay() {
    config_row_rects_.clear();
    if (!font_loaded) return;
    const auto rows = std::atomic_load(&runtime_config_)->snapshot();
    auto& editor = config_editor_;
    if (!rows.empty()) editor.selected = std::min(editor.selected, rows.size() - 1);
        if (editor.editing) for (size_t i = 0; i < rows.size(); ++i)
            if (rows[i].key == editor.editing_key) editor.selected = i;
    const float left = 20.f, top = 20.f;
    const float width = std::max(100.f, static_cast<float>(window_width) - 40.f);
    const float height = std::max(180.f, static_cast<float>(window_height) - 40.f);
    auto box = [&](float x, float y, float w, float h, vizgfx::Color color) {
        vizgfx::RectangleShape shape({w, h}); shape.setPosition({x,y}); shape.setFillColor(color); window->draw(shape);
    };
    auto text = [&](float x, float y, const std::string& value, vizgfx::Color color) {
        vizgfx::Text label(font); label.setCharacterSize(14); label.setPosition({x,y}); label.setFillColor(color);
        const size_t limit = static_cast<size_t>(std::max(1.f, (width - 32.f) / 8.5f));
        const auto clipped = value.size() > limit ? value.substr(0, limit > 3 ? limit - 3 : 0) + "..." : value;
        label.setString(vizgfx::String::fromUtf8(clipped.begin(), clipped.end())); window->draw(label);
    };
    const vizgfx::Color white(230,240,246), muted(150,163,178), accent(100,215,210);
    box(0, 0, static_cast<float>(window_width), static_cast<float>(window_height), vizgfx::Color(0,0,0,185));
    box(left, top, width, height, vizgfx::Color(22,29,39,255));
    text(left+12, top+10, "CONFIGURATION DU MODELE    [C] Fermer   [Esc] Annuler", accent);
    text(left+12, top+34, "Fleches / molette : choisir. Entree ou clic : modifier. Entree : valider.", muted);
    text(left+12, top+58, "DIRECT : modifiable maintenant. FIXE : reconstruction / redemarrage requis.", muted);
    const size_t count = static_cast<size_t>(std::max(1.f, (height - 166.f)/26.f));
    const size_t first = (editor.selected/count)*count;
    float y = top + 86.f;
    for (size_t i = first; i < rows.size() && i < first + count; ++i, y += 26.f) {
        const auto& e = rows[i];
        if (i == editor.selected) box(left+8, y-2, width-16, 25, vizgfx::Color(42,65,84));
        std::string value = LiveModelConfig::display(e.value);
        if (e.pending) value += " -> " + LiveModelConfig::display(*e.pending) + " (en attente)";
        text(left+14,y,(e.owner == LiveModelConfig::Owner::ReadOnly ? "FIXE   " : "DIRECT ") + e.key + " = " + value,
             e.owner == LiveModelConfig::Owner::ReadOnly ? muted : white);
        config_row_rects_.push_back({make_rect(left+8,y-2,width-16,25), i});
    }
    text(left+12,top+height-76,editor.message.empty() ? (rows.empty() ? "Aucun modele attache" : rows[editor.selected].note) : editor.message, accent);
    text(left+12,top+height-50,editor.editing ? editor.editing_key + " > " + editor.buffer + "_" : "Applique au prochain point sur du calcul.",white);
    text(left+12,top+height-26,std::to_string(rows.empty() ? 0 : editor.selected+1) + "/" + std::to_string(rows.size()),muted);
}

void Visualizer::processEvents() {
    if (!window) return;

    applyScriptScene();

    // Garantir que les coordonnées UI (pixels) sont actives avant tout hit-test.
    syncUIView();

    auto publish_live_params = [&](bool bump_version) {
        live_lr_.store(std::clamp(live_ui_lr_, kLiveLRMin, kLiveLRMax), std::memory_order_relaxed);
        live_lr_warmup_steps_.store(std::max(0, live_ui_lr_warmup_steps_), std::memory_order_relaxed);
        live_kl_beta_.store(std::clamp(live_ui_kl_beta_, 0.0f, 1.0f), std::memory_order_relaxed);
        live_kl_warmup_steps_.store(std::max(0, live_ui_kl_warmup_steps_), std::memory_order_relaxed);
        live_kl_enabled_.store(live_ui_kl_enabled_, std::memory_order_relaxed);
        live_recon_loss_index_.store(
            std::clamp(live_ui_recon_loss_index_, 0, 5), std::memory_order_relaxed);
        if (bump_version) {
            live_params_version_.fetch_add(1, std::memory_order_relaxed);
        }
    };

    auto apply_live_slider_from_mouse = [&](LiveDragTarget target, const vizgfx::Vector2f& mouse) {
        auto apply_t_on_track = [&](const std::optional<vizgfx::FloatRect>& track, auto fn) {
            if (!track.has_value()) return;
            const float denom = std::max(1.0f, track->size.x);
            const float t = clamp01((mouse.x - track->position.x) / denom);
            fn(t);
        };

        if (target == LiveDragTarget::LR) {
            apply_t_on_track(last_live_lr_track_, [&](float t) { live_ui_lr_ = lr_from_t(t); });
        } else if (target == LiveDragTarget::LRWarmup) {
            apply_t_on_track(last_live_lrwu_track_, [&](float t) { live_ui_lr_warmup_steps_ = warmup_from_t(t); });
        } else if (target == LiveDragTarget::KLBeta) {
            apply_t_on_track(last_live_klb_track_, [&](float t) { live_ui_kl_beta_ = klbeta_from_t(t); });
        } else if (target == LiveDragTarget::KLWarmup) {
            apply_t_on_track(last_live_klwu_track_, [&](float t) { live_ui_kl_warmup_steps_ = warmup_from_t(t); });
        }
    };

    auto apply_detection_threshold_from_mouse = [&](const vizgfx::Vector2f& mouse) {
        if (!last_detection_threshold_track_.has_value()) return;
        const auto& track = *last_detection_threshold_track_;
        const float inner_x = track.position.x + 4.0f;
        const float inner_w = std::max(1.0f, track.size.x - 8.0f);
        detection_score_threshold_ = clamp01((mouse.x - inner_x) / inner_w);
    };

    auto trim_local = [](const std::string& s) -> std::string {
        size_t a = 0;
        while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
        size_t b = s.size();
        while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
        return s.substr(a, b - a);
    };

    auto commit_blocks_tips_limit = [&]() {
        const std::string trimmed = trim_local(blocks_tips_limit_input_);
        if (trimmed.empty() || trimmed == "ALL" || trimmed == "all" || trimmed == "0") {
            blocks_tips_visible_limit_ = 0;
        } else {
            try {
                const int v = std::stoi(trimmed);
                blocks_tips_visible_limit_ = std::max(0, v);
            } catch (...) {
                // Entrée invalide: conserver l'ancienne valeur.
            }
        }
        blocks_tips_limit_editing_ = false;
        blocks_tips_limit_input_.clear();
        saveUILayoutToLast();
    };

    auto commit_live_input = [&]() {
        if (live_input_target_ == LiveInputTarget::None) return;
        const std::string trimmed = trim_local(live_input_buffer_);
        bool ok = false;

        try {
            if (live_input_target_ == LiveInputTarget::LR) {
                const float v = std::stof(trimmed);
                if (std::isfinite(v)) {
                    live_ui_lr_ = std::clamp(v, kLiveLRMin, kLiveLRMax);
                    ok = true;
                }
            } else if (live_input_target_ == LiveInputTarget::KLBeta) {
                const float v = std::stof(trimmed);
                if (std::isfinite(v)) {
                    live_ui_kl_beta_ = std::clamp(v, 0.0f, 1.0f);
                    ok = true;
                }
            } else if (live_input_target_ == LiveInputTarget::LRWarmup) {
                const int v = std::stoi(trimmed);
                live_ui_lr_warmup_steps_ = std::clamp(v, 0, kLiveWarmupMax);
                ok = true;
            } else if (live_input_target_ == LiveInputTarget::KLWarmup) {
                const int v = std::stoi(trimmed);
                live_ui_kl_warmup_steps_ = std::clamp(v, 0, kLiveWarmupMax);
                ok = true;
            }
        } catch (...) {
            ok = false;
        }

        if (ok) {
            live_overrides_enabled_.store(true, std::memory_order_relaxed);
            publish_live_params(true);
            live_input_target_ = LiveInputTarget::None;
            live_input_buffer_.clear();
            live_input_error_until_ms_ = 0;
        } else {
            live_input_error_until_ms_ = steady_now_ms() + 900;
        }
    };

    while (const std::optional<vizgfx::Event> event_opt = window->pollEvent()) {
        const vizgfx::Event& event = *event_opt;

        if (event.is<vizgfx::Event::Closed>()) {
            emitScriptEvent({{"type", "close"}});
            window->close();
            continue;
        }

        if (const auto* resized = event.getIf<vizgfx::Event::Resized>()) {
            emitScriptEvent({{"type", "resize"}, {"width", resized->size.x}, {"height", resized->size.y}});
            window_width = static_cast<int>(resized->size.x);
            window_height = static_cast<int>(resized->size.y);
            syncUIView();
            continue;
        }

        if (debug_json_dialog_.visible) {
            if (const auto* key = event.getIf<vizgfx::Event::KeyPressed>()) {
                if (key->code == vizgfx::Keyboard::Key::Escape) debug_json_dialog_.visible = false;
                else if (key->code == vizgfx::Keyboard::Key::Tab || key->code == vizgfx::Keyboard::Key::Up || key->code == vizgfx::Keyboard::Key::Down) debug_json_dialog_.move();
                else if (key->code == vizgfx::Keyboard::Key::Enter) debug_json_dialog_.submit(*std::atomic_load(&runtime_config_));
            } else if (const auto* text = event.getIf<vizgfx::Event::TextEntered>()) {
                debug_json_dialog_.text(text->unicode);
            } else if (const auto* click = event.getIf<vizgfx::Event::MouseButtonPressed>()) {
                const float top = std::max(20.f, (static_cast<float>(window_height)-280.f)/2.f);
                if (click->position.y >= top+75.f && click->position.y < top+155.f) {
                    debug_json_dialog_.selected = click->position.y < top+115.f ? 0 : 1;
                    debug_json_dialog_.replace = true;
                }
            }
            continue;
        }
        if (const auto* key = event.getIf<vizgfx::Event::KeyPressed>()) {
            if (show_model_graph_) {
                if (key->code==vizgfx::Keyboard::Key::N || key->code==vizgfx::Keyboard::Key::Escape) {
                    show_model_graph_=false;
                    std::atomic_load(&runtime_config_)->requestModelGraph(false);
                } else if (key->code==vizgfx::Keyboard::Key::Down) model_graph_y_+=100.f;
                else if (key->code==vizgfx::Keyboard::Key::Up) model_graph_y_-=100.f;
                else if (key->code==vizgfx::Keyboard::Key::Right) model_graph_x_+=150.f;
                else if (key->code==vizgfx::Keyboard::Key::Left) model_graph_x_-=150.f;
                else if (key->code==vizgfx::Keyboard::Key::R) model_graph_x_=model_graph_y_=0.f;
                continue;
            }
            if (!show_help_overlay_ && !config_editor_.visible && !zoom_active_ &&
                live_input_target_==LiveInputTarget::None && !blocks_tips_limit_editing_ &&
                key->code==vizgfx::Keyboard::Key::N && !key->control) {
                show_model_graph_=true;
                std::atomic_load(&runtime_config_)->requestModelGraph(true);
                continue;
            }
            if (show_help_overlay_) {
                if (key->code==vizgfx::Keyboard::Key::H || key->code==vizgfx::Keyboard::Key::Escape)
                    show_help_overlay_=false;
                else if (key->code==vizgfx::Keyboard::Key::Up || key->code==vizgfx::Keyboard::Key::Down)
                    help_scroll_y_=std::clamp(help_scroll_y_+(key->code==vizgfx::Keyboard::Key::Down?48.f:-48.f),0.f,help_scroll_max_);
                continue;
            }
            if (!config_editor_.editing && live_input_target_ == LiveInputTarget::None &&
                !blocks_tips_limit_editing_ && key->code == vizgfx::Keyboard::Key::C && !key->control) {
                config_editor_.toggle();
                continue;
            }
        }
        if (show_model_graph_) {
            if (const auto* wheel=event.getIf<vizgfx::Event::MouseWheelScrolled>())
                model_graph_y_-=wheel->delta*80.f;
            if (const auto* click=event.getIf<vizgfx::Event::MouseButtonPressed>()) {
                if(click->button==vizgfx::Mouse::Button::Left)
                    for(const auto& hit:model_graph_hits_)
                        if(hit.first.contains({float(click->position.x),float(click->position.y)}))
                            model_graph_selected_=hit.second;
            }
            continue;
        }
        if (config_editor_.visible) {
            auto config = std::atomic_load(&runtime_config_);
            const auto count = config->snapshot().size();
            if (const auto* key = event.getIf<vizgfx::Event::KeyPressed>()) {
                if (key->code == vizgfx::Keyboard::Key::Escape) config_editor_.cancel();
                else if (key->code == vizgfx::Keyboard::Key::Enter) config_editor_.enter(*config);
                else if (key->code == vizgfx::Keyboard::Key::Up) config_editor_.move(-1, count);
                else if (key->code == vizgfx::Keyboard::Key::Down || key->code == vizgfx::Keyboard::Key::Tab) config_editor_.move(1, count);
                else if (key->code == vizgfx::Keyboard::Key::Left) config_editor_.move(-10, count);
                else if (key->code == vizgfx::Keyboard::Key::Right) config_editor_.move(10, count);
            } else if (const auto* text = event.getIf<vizgfx::Event::TextEntered>()) {
                config_editor_.text(static_cast<uint32_t>(text->unicode));
            } else if (const auto* wheel = event.getIf<vizgfx::Event::MouseWheelScrolled>()) {
                config_editor_.move(wheel->delta > 0 ? -3 : 3, count);
            } else if (const auto* click = event.getIf<vizgfx::Event::MouseButtonPressed>()) {
                if (click->button == vizgfx::Mouse::Button::Left) {
                    const vizgfx::Vector2f mouse(static_cast<float>(click->position.x), static_cast<float>(click->position.y));
                    for (const auto& row : config_row_rects_) if (row.first.contains(mouse)) {
                        config_editor_.editing = false;
                        config_editor_.selected = row.second;
                        config_editor_.enter(*config);
                        break;
                    }
                }
            }
            continue;
        }

        if (script_scene_.value("events", false)) {
            if (const auto* key = event.getIf<vizgfx::Event::KeyPressed>())
                emitScriptEvent({{"type", "key"}, {"code", static_cast<int>(key->code)},
                    {"control", key->control}, {"shift", key->shift}, {"alt", key->alt}});
            if (const auto* text = event.getIf<vizgfx::Event::TextEntered>())
                emitScriptEvent({{"type", "text"}, {"unicode", static_cast<uint32_t>(text->unicode)}});
            if (const auto* move = event.getIf<vizgfx::Event::MouseMoved>())
                emitScriptEvent({{"type", "pointer_move"}, {"x", move->position.x}, {"y", move->position.y}});
            if (const auto* button = event.getIf<vizgfx::Event::MouseButtonPressed>())
                emitScriptEvent({{"type", "pointer_down"}, {"x", button->position.x}, {"y", button->position.y}, {"button", static_cast<int>(button->button)}});
            if (const auto* button = event.getIf<vizgfx::Event::MouseButtonReleased>())
                emitScriptEvent({{"type", "pointer_up"}, {"x", button->position.x}, {"y", button->position.y}, {"button", static_cast<int>(button->button)}});
            if (event.is<vizgfx::Event::KeyPressed>() && script_scene_.value("capture_keys", false)) {
                continue;
            }
        }
        if (const auto* mb = event.getIf<vizgfx::Event::MouseButtonPressed>()) {
            if (show_help_overlay_) {
                if (mb->button==vizgfx::Mouse::Button::Left && help_close_rect_ &&
                    help_close_rect_->contains(window->mapPixelToCoords(mb->position))) show_help_overlay_=false;
                continue;
            }
            if (zoom_active_ && !show_help_overlay_) {
                const auto point=window->mapPixelToCoords(mb->position);
                if (mb->button==vizgfx::Mouse::Button::Right) {
                    picker_selection_active_=picker_selecting_=false;
                    zoom_info_scroll_=0.f;
                } else if (mb->button==vizgfx::Mouse::Button::Left && picker_image_clip_ &&
                           picker_image_clip_->contains(point) && picker_source_.w>0) {
                    picker_selection_source_=picker_source_; picker_selection_raw_=picker_source_raw_;
                    picker_anchor_x_=std::clamp(static_cast<int>((point.x-picker_image_rect_->position.x)/picker_image_scale_),0,picker_source_.w-1);
                    picker_anchor_y_=std::clamp(static_cast<int>((point.y-picker_image_rect_->position.y)/picker_image_scale_),0,picker_source_.h-1);
                    picker_selection_active_=picker_selecting_=true;
                    zoom_info_scroll_=0.f;
                    updatePickerSelection(point);
                }
                continue;
            }
            if (show_help_overlay_ || zoom_active_) continue;
            initDefaultPanelsIfNeeded();
            syncUIView();
            const vizgfx::Vector2f mouse = window->mapPixelToCoords(mb->position);
            const auto clicked_panel = hitTestPanel(mouse);

            if (!show_help_overlay_ && !zoom_active_ && mb->button == vizgfx::Mouse::Button::Left && script_scene_.contains("controls")) {
                bool handled = false;
                for (auto it = script_scene_["controls"].rbegin(); it != script_scene_["controls"].rend(); ++it) {
                    const auto& c = *it;
                    if (make_rect(c.value("x", 12.f), c.value("y", 12.f), c.value("w", 150.f), c.value("h", 36.f)).contains(mouse)) {
                        emitScriptEvent({{"type", "click"}, {"id", c["id"]}, {"x", mouse.x}, {"y", mouse.y}});
                        handled = true;
                        break;
                    }
                }
                if (handled) continue;
            }
            if (mb->button == vizgfx::Mouse::Button::Left) {
                if (clicked_panel == PanelId::Output &&
                    ((last_detection_threshold_track_ && last_detection_threshold_track_->contains(mouse)) ||
                     (last_detection_threshold_thumb_ && last_detection_threshold_thumb_->contains(mouse)))) {
                    dragging_detection_threshold_ = true;
                    apply_detection_threshold_from_mouse(mouse);
                    continue;
                }
                const bool metrics_content_hit = clicked_panel == PanelId::Metrics &&
                    panelContentRect(PanelId::Metrics).contains(mouse);
                if (metrics_content_hit && last_debug_json_button_rect_ && last_debug_json_button_rect_->contains(mouse)) {
                    debug_json_dialog_.open();
                    continue;
                }
                if (metrics_content_hit && last_pause_button_rect_ && last_pause_button_rect_->contains(mouse)) {
                    auto paused = std::atomic_load(&training_paused_);
                    paused->store(!paused->load());
                    continue;
                }
                if (metrics_content_hit && last_stop_button_rect_.has_value() && last_stop_button_rect_->contains(mouse)) {
                    requestStopTraining();
                    continue;
                }
                if (metrics_content_hit && last_skip_button_rect_ && last_skip_button_rect_->contains(mouse)) {
                    std::atomic_load(&skip_control_)->toggle();
                    continue;
                }
                if (metrics_content_hit && last_validation_button_rect_.has_value() && last_validation_button_rect_->contains(mouse)) {
                    const bool enabled = !validation_enabled_.load(std::memory_order_relaxed);
                    validation_enabled_.store(enabled, std::memory_order_relaxed);
                    validation_control_version_.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                if (metrics_content_hit && last_live_overrides_box_.has_value() && last_live_overrides_box_->contains(mouse)) {
                    const bool next = !live_overrides_enabled_.load(std::memory_order_relaxed);
                    live_overrides_enabled_.store(next, std::memory_order_relaxed);
                    publish_live_params(true);
                    continue;
                }

                if (metrics_content_hit && last_live_kl_enable_box_.has_value() && last_live_kl_enable_box_->contains(mouse)) {
                    live_ui_kl_enabled_ = !live_ui_kl_enabled_;
                    live_overrides_enabled_.store(true, std::memory_order_relaxed);
                    publish_live_params(true);
                    continue;
                }

                if (metrics_content_hit && last_live_recon_loss_box_.has_value() && last_live_recon_loss_box_->contains(mouse)) {
                    live_ui_recon_loss_index_ = (live_ui_recon_loss_index_ + 1) % 6;
                    live_overrides_enabled_.store(true, std::memory_order_relaxed);
                    publish_live_params(true);
                    continue;
                }

                auto begin_live_text_edit = [&](LiveInputTarget target, const std::string& initial_value) {
                    live_input_target_ = target;
                    live_input_buffer_ = initial_value;
                    live_input_error_until_ms_ = 0;
                };

                if (metrics_content_hit && last_live_lr_value_box_.has_value() && last_live_lr_value_box_->contains(mouse)) {
                    begin_live_text_edit(LiveInputTarget::LR, format_decimal(live_ui_lr_, 10));
                    continue;
                }
                if (metrics_content_hit && last_live_lrwu_value_box_.has_value() && last_live_lrwu_value_box_->contains(mouse)) {
                    begin_live_text_edit(LiveInputTarget::LRWarmup, std::to_string(live_ui_lr_warmup_steps_));
                    continue;
                }
                if (metrics_content_hit && last_live_klb_value_box_.has_value() && last_live_klb_value_box_->contains(mouse)) {
                    begin_live_text_edit(LiveInputTarget::KLBeta, format_decimal(live_ui_kl_beta_, 8));
                    continue;
                }
                if (metrics_content_hit && last_live_klwu_value_box_.has_value() && last_live_klwu_value_box_->contains(mouse)) {
                    begin_live_text_edit(LiveInputTarget::KLWarmup, std::to_string(live_ui_kl_warmup_steps_));
                    continue;
                }

                auto begin_live_drag_if_hit = [&](LiveDragTarget target,
                                                  const std::optional<vizgfx::FloatRect>& track,
                                                  const std::optional<vizgfx::FloatRect>& thumb) -> bool {
                    if ((thumb.has_value() && thumb->contains(mouse)) || (track.has_value() && track->contains(mouse))) {
                        live_dragging_ = target;
                        live_overrides_enabled_.store(true, std::memory_order_relaxed);
                        apply_live_slider_from_mouse(target, mouse);
                        publish_live_params(true);
                        return true;
                    }
                    return false;
                };

                if (metrics_content_hit && begin_live_drag_if_hit(LiveDragTarget::LR, last_live_lr_track_, last_live_lr_thumb_)) continue;
                if (metrics_content_hit && begin_live_drag_if_hit(LiveDragTarget::LRWarmup, last_live_lrwu_track_, last_live_lrwu_thumb_)) continue;
                if (metrics_content_hit && begin_live_drag_if_hit(LiveDragTarget::KLBeta, last_live_klb_track_, last_live_klb_thumb_)) continue;
                if (metrics_content_hit && begin_live_drag_if_hit(LiveDragTarget::KLWarmup, last_live_klwu_track_, last_live_klwu_thumb_)) continue;

                if (clicked_panel == PanelId::Blocks && last_blocks_mode_box_ && last_blocks_mode_box_->contains(mouse)) {
                    heatmap_mode_ = !heatmap_mode_;
                    setLayerBlockImages(last_block_frames_);
                    continue;
                }
                if (clicked_panel == PanelId::Blocks && last_blocks_smooth_box_ && last_blocks_smooth_box_->contains(mouse)) {
                    smooth_layer_block_previews_ = !smooth_layer_block_previews_;
                    rebuildAllTextures();
                    continue;
                }
                if (clicked_panel == PanelId::Blocks && last_blocks_palette_box_ && last_blocks_palette_box_->contains(mouse)) {
                    heatmap_palette_ = static_cast<HeatmapPalette>((static_cast<int>(heatmap_palette_) + 1) % 4);
                    setLayerBlockImages(last_block_frames_);
                    saveUILayoutToLast();
                    continue;
                }
                if (clicked_panel == PanelId::Blocks && last_blocks_hide_act_box_.has_value() && last_blocks_hide_act_box_->contains(mouse)) {
                    hide_activation_blocks = !hide_activation_blocks;
                    setLayerBlockImages(last_block_frames_);
                    requestResync();
                    saveUILayoutToLast();
                    continue;
                }
                if (clicked_panel == PanelId::Blocks && last_blocks_hide_norm_box_.has_value() && last_blocks_hide_norm_box_->contains(mouse)) {
                    hide_normalisation_blocks = !hide_normalisation_blocks;
                    setLayerBlockImages(last_block_frames_);
                    requestResync();
                    saveUILayoutToLast();
                    continue;
                }
                if (clicked_panel == PanelId::Blocks && last_blocks_tips_limit_box_.has_value() && last_blocks_tips_limit_box_->contains(mouse)) {
                    blocks_tips_limit_editing_ = true;
                    blocks_tips_limit_input_ = (blocks_tips_visible_limit_ > 0) ? std::to_string(blocks_tips_visible_limit_) : std::string();
                    continue;
                }
                if (clicked_panel == PanelId::Context && last_detection_toggle_box_.has_value() && last_detection_toggle_box_->contains(mouse)) {
                    show_detection_overlay_ = !show_detection_overlay_;
                    saveUILayoutToLast();
                    continue;
                }
                if (blocks_tips_limit_editing_) {
                    commit_blocks_tips_limit();
                }

                // Sélection directe par clic sur les vignettes de tips (panel Blocks).
                bool selected_tip = false;
                if (clicked_panel == PanelId::Blocks &&
                    panelContentRect(PanelId::Blocks).contains(mouse) && !last_block_rects_.empty()) {
                    for (int i = static_cast<int>(last_block_rects_.size()) - 1; i >= 0; --i) {
                        const auto& r = last_block_rects_[static_cast<size_t>(i)];
                        if (r.size.x <= 0.f || r.size.y <= 0.f) continue;
                        if (r.contains(mouse)) {
                            focus_target_ = FocusTarget::LayerBlock;
                            focus_block_index_ = i;
                            selected_tip = true;
                            break;
                        }
                    }
                }
                if (selected_tip) continue;

                // Sélection directe des images générées au clic.
                bool selected_generated = false;
                if (clicked_panel == PanelId::Generated &&
                    panelContentRect(PanelId::Generated).contains(mouse) &&
                    !last_generated_rects_.empty() && last_generated_rects_.size() == last_generated_indices_.size()) {
                    for (int i = static_cast<int>(last_generated_rects_.size()) - 1; i >= 0; --i) {
                        const auto& r = last_generated_rects_[static_cast<size_t>(i)];
                        if (r.size.x <= 0.f || r.size.y <= 0.f) continue;
                        if (r.contains(mouse)) {
                            focus_target_ = FocusTarget::Generated;
                            focus_generated_index_ = std::max(0, last_generated_indices_[static_cast<size_t>(i)]);
                            selected_generated = true;
                            break;
                        }
                    }
                }
                if (selected_generated) continue;
                const std::pair<const std::optional<vizgfx::FloatRect>*, FocusTarget> image_targets[] = {
                    {&last_dataset_rect_,FocusTarget::Dataset}, {&last_projection_rect_,FocusTarget::Projection},
                    {&last_understanding_rect_,FocusTarget::Understanding}, {&last_output_rect_,FocusTarget::Output}};
                bool selected_image = false;
                for (const auto& target : image_targets) {
                    const auto owner = target.second == FocusTarget::Output ? PanelId::Output : PanelId::Context;
                    if (clicked_panel == owner && *target.first && (*target.first)->contains(mouse)) {
                        focus_target_ = target.second; selected_image = true; break;
                    }
                }
                if (selected_image) continue;


                if (const auto close_hit = hitTestPanelCloseButton(mouse)) {
                    const PanelId id = *close_hit;
                    if (id == PanelId::Generated) show_generated_images = false;
                    else if (id == PanelId::Training) show_training_progress = false;
                    else if (id == PanelId::Graph) show_loss_graph = false;
                    panels_[static_cast<size_t>(id)].visible = false;
                    saveUILayoutToLast();
                    continue;
                }

                bool panel_scroll_hit = false;
                for (size_t index = 0; index < static_cast<size_t>(PanelId::Count); ++index) {
                    const PanelId id = static_cast<PanelId>(index);
                    if (id == PanelId::Blocks || clicked_panel != id) continue;
                    const auto& thumb = panel_scroll_thumb_rects_[index];
                    const auto& track = panel_scroll_track_rects_[index];
                    if (thumb.contains(mouse)) {
                        dragging_panel_scrollbar_ = true;
                        scrolled_panel_ = id;
                        panel_scroll_drag_grab_y_ = mouse.y - thumb.position.y;
                        panel_scroll_hit = true;
                        break;
                    }
                    if (track.contains(mouse)) {
                        const float thumb_h = std::max(1.0f, thumb.size.y);
                        const float ratio = clamp01(
                            (mouse.y - track.position.y - thumb_h * 0.5f) /
                            std::max(1.0f, track.size.y - thumb_h));
                        panel_scroll_y_[index] = ratio * panel_scroll_max_[index];
                        panel_scroll_hit = true;
                        break;
                    }
                }
                if (panel_scroll_hit) continue;

                if (clicked_panel == PanelId::Blocks && last_blocks_scroll_thumb_rect_.contains(mouse)) {
                    dragging_blocks_scrollbar_ = true;
                    blocks_scroll_drag_grab_y_ = mouse.y - last_blocks_scroll_thumb_rect_.position.y;
                    continue;
                }
                if (clicked_panel == PanelId::Blocks && last_blocks_scroll_track_rect_.contains(mouse)) {
                    const float thumb_h = std::max(1.0f, last_blocks_scroll_thumb_rect_.size.y);
                    const float t = clamp01((mouse.y - last_blocks_scroll_track_rect_.position.y - thumb_h * 0.5f) /
                                            std::max(1.0f, last_blocks_scroll_track_rect_.size.y - thumb_h));
                    blocks_scroll_y_ = t * std::max(0.0f, blocks_scroll_max_);
                    continue;
                }

                if (const auto resize_hit = hitTestPanelResizeHandle(mouse)) {
                    resizing_panel_ = true;
                    resized_panel_ = *resize_hit;
                    resize_start_mouse_ = mouse;
                    resize_start_size_ = panels_[static_cast<size_t>(resized_panel_)].size;
                    setCursor(CursorKind::Resize);
                    continue;
                }

                if (const auto drag_hit = hitTestPanelTitle(mouse)) {
                    dragging_panel_ = true;
                    dragged_panel_ = *drag_hit;
                    const auto& p = panels_[static_cast<size_t>(dragged_panel_)];
                    drag_grab_offset_ = vizgfx::Vector2f(mouse.x - p.pos.x, mouse.y - p.pos.y);
                    setCursor(CursorKind::Hand);
                    continue;
                }
            }
            continue;
        }

        if (const auto* mm = event.getIf<vizgfx::Event::MouseMoved>()) {
            picker_mouse_ = window->mapPixelToCoords(mm->position);
            if (picker_selecting_ && zoom_active_ && !show_help_overlay_) updatePickerSelection(picker_mouse_);
            if (show_help_overlay_ || zoom_active_) continue;
            const vizgfx::Vector2f mouse = window->mapPixelToCoords(mm->position);

            if (dragging_panel_) {
                auto& p = panels_[static_cast<size_t>(dragged_panel_)];
                p.pos = vizgfx::Vector2f(mouse.x - drag_grab_offset_.x, mouse.y - drag_grab_offset_.y);
                clampPanelsToWindow();
            }

            if (resizing_panel_) {
                auto& p = panels_[static_cast<size_t>(resized_panel_)];
                const float dx = mouse.x - resize_start_mouse_.x;
                const float dy = mouse.y - resize_start_mouse_.y;
                p.size.x = std::max(kPanelMinW, resize_start_size_.x + dx);
                p.size.y = std::max(kPanelMinH, resize_start_size_.y + dy);
                clampPanelsToWindow();
            }

            if (dragging_blocks_scrollbar_ && isPanelVisible(PanelId::Blocks)) {
                const float thumb_h = std::max(1.0f, last_blocks_scroll_thumb_rect_.size.y);
                const float rel = mouse.y - last_blocks_scroll_track_rect_.position.y - blocks_scroll_drag_grab_y_;
                const float denom = std::max(1.0f, last_blocks_scroll_track_rect_.size.y - thumb_h);
                const float t = clamp01(rel / denom);
                blocks_scroll_y_ = t * std::max(0.0f, blocks_scroll_max_);
            }

            if (dragging_panel_scrollbar_ && isPanelVisible(scrolled_panel_)) {
                const size_t index = static_cast<size_t>(scrolled_panel_);
                const auto& track = panel_scroll_track_rects_[index];
                const auto& thumb = panel_scroll_thumb_rects_[index];
                const float travel = std::max(1.0f, track.size.y - thumb.size.y);
                const float ratio = clamp01(
                    (mouse.y - track.position.y - panel_scroll_drag_grab_y_) / travel);
                panel_scroll_y_[index] = ratio * panel_scroll_max_[index];
            }

            if (live_dragging_ != LiveDragTarget::None) {
                apply_live_slider_from_mouse(live_dragging_, mouse);
                publish_live_params(true);
            }
            if (dragging_detection_threshold_) {
                apply_detection_threshold_from_mouse(mouse);
            }

            if (dragging_panel_) {
                setCursor(CursorKind::Hand);
            } else if (resizing_panel_ || dragging_blocks_scrollbar_ || dragging_panel_scrollbar_) {
                setCursor(CursorKind::Resize);
            } else {
                bool is_hand = false;
                if (isPanelVisible(PanelId::Blocks) &&
                    (last_blocks_scroll_thumb_rect_.contains(mouse) || last_blocks_scroll_track_rect_.contains(mouse))) {
                    is_hand = true;
                }
                if (!is_hand) {
                    for (size_t index = 0; index < static_cast<size_t>(PanelId::Count); ++index) {
                        const PanelId id = static_cast<PanelId>(index);
                        if (id == PanelId::Blocks || !isPanelVisible(id)) continue;
                        if (panel_scroll_thumb_rects_[index].contains(mouse) ||
                            panel_scroll_track_rects_[index].contains(mouse)) {
                            is_hand = true;
                            break;
                        }
                    }
                }
                if (!is_hand && (hitTestPanelResizeHandle(mouse).has_value() || hitTestPanelTitle(mouse).has_value())) {
                    is_hand = true;
                }
                setCursor(is_hand ? CursorKind::Hand : CursorKind::Arrow);
            }
            continue;
        }

        if (const auto* mb_rel = event.getIf<vizgfx::Event::MouseButtonReleased>()) {
            if (mb_rel->button == vizgfx::Mouse::Button::Left) {
                if (picker_selecting_) {
                    updatePickerSelection(window->mapPixelToCoords(mb_rel->position));
                    picker_selecting_=false;
                }
                if (dragging_panel_ || resizing_panel_ || dragging_blocks_scrollbar_ || dragging_panel_scrollbar_) {
                    saveUILayoutToLast();
                }
                dragging_panel_ = false;
                resizing_panel_ = false;
                dragging_blocks_scrollbar_ = false;
                dragging_panel_scrollbar_ = false;
                live_dragging_ = LiveDragTarget::None;
                dragging_detection_threshold_ = false;
            }
            continue;
        }

        if (const auto* txt = event.getIf<vizgfx::Event::TextEntered>()) {
            const uint32_t cp = static_cast<uint32_t>(txt->unicode);

            if (blocks_tips_limit_editing_) {
                if (cp == 8u) {
                    if (!blocks_tips_limit_input_.empty()) blocks_tips_limit_input_.pop_back();
                } else if (cp >= static_cast<uint32_t>('0') && cp <= static_cast<uint32_t>('9')) {
                    blocks_tips_limit_input_.push_back(static_cast<char>(cp));
                }
                continue;
            }

            if (live_input_target_ != LiveInputTarget::None) {
                if (cp == 8u) {
                    if (!live_input_buffer_.empty()) live_input_buffer_.pop_back();
                } else if (is_live_input_char(cp)) {
                    live_input_buffer_.push_back(static_cast<char>(cp));
                }
                continue;
            }
        }

        if (const auto* wheel = event.getIf<vizgfx::Event::MouseWheelScrolled>()) {
            if (show_help_overlay_) {
                help_scroll_y_=std::clamp(help_scroll_y_-wheel->delta*48.f,0.f,help_scroll_max_);
                continue;
            }
            if (zoom_active_) {
                zoom_info_scroll_ = std::clamp(zoom_info_scroll_ - wheel->delta * 32.0f, 0.0f, zoom_info_scroll_max_);
                continue;
            }
            if (dragging_panel_ || resizing_panel_) continue;
            initDefaultPanelsIfNeeded();
            syncUIView();
            const vizgfx::Vector2f mouse = window->mapPixelToCoords(wheel->position);
            bool consumed_scroll = false;
            if (isPanelVisible(PanelId::Metrics) &&
                panelContentRect(PanelId::Metrics).contains(mouse)) {
                for (size_t i = 0; i < dataset_text_section_rects_.size(); ++i) {
                    if (!dataset_text_section_rects_[i].contains(mouse)) continue;
                    const float previous = dataset_text_scroll_y_[i];
                    dataset_text_scroll_y_[i] = std::clamp(
                        previous - wheel->delta * kDatasetTextScrollSpeed,
                        0.0f, dataset_text_scroll_max_[i]);
                    consumed_scroll = std::abs(dataset_text_scroll_y_[i] - previous) > 0.01f;
                    break;
                }
            }
            if (isPanelVisible(PanelId::Blocks)) {
                const auto content = panelContentRect(PanelId::Blocks);
                if (content.contains(mouse)) {
                    blocks_scroll_y_ -= wheel->delta * kBlocksScrollSpeed;
                    blocks_scroll_y_ = std::clamp(blocks_scroll_y_, 0.0f, std::max(0.0f, blocks_scroll_max_));
                    consumed_scroll = true;
                }
            }
            if (!consumed_scroll) {
                const PanelId order[] = {PanelId::Metrics, PanelId::Training,
                    PanelId::Output, PanelId::Generated, PanelId::Context};
                for (PanelId id : order) {
                    if (!isPanelVisible(id) || !panelContentRect(id).contains(mouse)) continue;
                    const size_t index = static_cast<size_t>(id);
                    panel_scroll_y_[index] -= wheel->delta * kPanelScrollSpeed;
                    panel_scroll_y_[index] = std::clamp(
                        panel_scroll_y_[index], 0.0f, panel_scroll_max_[index]);
                    break;
                }
            }
            continue;
        }

        if (const auto* key = event.getIf<vizgfx::Event::KeyPressed>()) {
            const vizgfx::Keyboard::Key code = key->code;
            const bool ctrl = key->control;
            const bool shift = key->shift;

            if (blocks_tips_limit_editing_) {
                if (code == vizgfx::Keyboard::Key::Enter) {
                    commit_blocks_tips_limit();
                } else if (code == vizgfx::Keyboard::Key::Escape) {
                    blocks_tips_limit_editing_ = false;
                    blocks_tips_limit_input_.clear();
                }
                continue;
            }

            if (live_input_target_ != LiveInputTarget::None) {
                if (code == vizgfx::Keyboard::Key::Enter) {
                    commit_live_input();
                } else if (code == vizgfx::Keyboard::Key::Escape) {
                    live_input_target_ = LiveInputTarget::None;
                    live_input_buffer_.clear();
                    live_input_error_until_ms_ = 0;
                }
                continue;
            }

            if (code == vizgfx::Keyboard::Key::H) {
                show_help_overlay_=true;
                help_scroll_y_=0.f;
            }
            if (code == vizgfx::Keyboard::Key::D)
                debug_json_dialog_.open();
            if (code == vizgfx::Keyboard::Key::E) snapshot_requested_ = true;
            if (code == vizgfx::Keyboard::Key::G) {
                show_loss_graph = !show_loss_graph;
                panels_[static_cast<size_t>(PanelId::Graph)].visible = show_loss_graph;
            }
            if (code == vizgfx::Keyboard::Key::O) {
                auto& output_panel = panels_[static_cast<size_t>(PanelId::Output)];
                output_panel.visible = !output_panel.visible;
                saveUILayoutToLast();
            }
            if (code == vizgfx::Keyboard::Key::P) show_prompt_text_ = !show_prompt_text_;
            if (code == vizgfx::Keyboard::Key::F) {
                show_feature_probe_ = !show_feature_probe_;
                saveUILayoutToLast();
            }
            if (code == vizgfx::Keyboard::Key::B) {
                show_detection_overlay_ = !show_detection_overlay_;
                saveUILayoutToLast();
            }
            if (code == vizgfx::Keyboard::Key::A) {
                smooth_layer_block_previews_ = !smooth_layer_block_previews_;
                rebuildAllTextures();
            }
            if (code == vizgfx::Keyboard::Key::M) {
                heatmap_mode_ = !heatmap_mode_;
                setLayerBlockImages(last_block_frames_);
            }
            if (code == vizgfx::Keyboard::Key::K) {
                const int pid = (static_cast<int>(heatmap_palette_) + 1) % 4;
                heatmap_palette_ = static_cast<Visualizer::HeatmapPalette>(pid);
                setLayerBlockImages(last_block_frames_);
                saveUILayoutToLast();
            }
            if (code == vizgfx::Keyboard::Key::Z || code == vizgfx::Keyboard::Key::Enter) zoom_active_ = !zoom_active_;
            if (code == vizgfx::Keyboard::Key::T && zoom_active_) zoom_actual_size_ = !zoom_actual_size_;
            if (code == vizgfx::Keyboard::Key::Escape) {
                if (picker_selection_active_) {
                    picker_selection_active_=picker_selecting_=false; zoom_info_scroll_=0.f;
                } else zoom_active_=false;
            }
            if (code == vizgfx::Keyboard::Key::R) {
                requestResync();
                rebuildAllTextures();
                architecture_loaded = false;
                arch_layer_names.clear();
                arch_tensor_outputs.clear();
                arch_tensor_sinks.clear();
            }
            if (code == vizgfx::Keyboard::Key::Tab) {
                if (focus_target_ == FocusTarget::Dataset) focus_target_ = FocusTarget::Projection;
                else if (focus_target_ == FocusTarget::Projection) focus_target_ = FocusTarget::Understanding;
                else if (focus_target_ == FocusTarget::Understanding) focus_target_ = FocusTarget::LayerBlock;
                else if (focus_target_ == FocusTarget::LayerBlock) focus_target_ = FocusTarget::Generated;
                else if (focus_target_ == FocusTarget::Generated) focus_target_ = FocusTarget::Output;
                else if (focus_target_ == FocusTarget::Output) focus_target_ = FocusTarget::Training;
                else if (focus_target_ == FocusTarget::Training) focus_target_ = FocusTarget::Metrics;
                else if (focus_target_ == FocusTarget::Metrics) focus_target_ = FocusTarget::Graph;
                else focus_target_ = FocusTarget::Dataset;
            }
            if (code == vizgfx::Keyboard::Key::F1) focus_target_ = FocusTarget::Dataset;
            if (code == vizgfx::Keyboard::Key::F2) focus_target_ = FocusTarget::Projection;
            if (code == vizgfx::Keyboard::Key::F3) focus_target_ = FocusTarget::Understanding;
            if (code == vizgfx::Keyboard::Key::F4) focus_target_ = FocusTarget::LayerBlock;
            if (code == vizgfx::Keyboard::Key::F5) focus_target_ = FocusTarget::Generated;
            if (code == vizgfx::Keyboard::Key::F6) focus_target_ = FocusTarget::Output;
            if (code == vizgfx::Keyboard::Key::F7) focus_target_ = FocusTarget::Training;
            if (code == vizgfx::Keyboard::Key::F8) focus_target_ = FocusTarget::Metrics;
            if (code == vizgfx::Keyboard::Key::F9) focus_target_ = FocusTarget::Graph;

            if (ctrl && (code == vizgfx::Keyboard::Key::Left || code == vizgfx::Keyboard::Key::Right ||
                         code == vizgfx::Keyboard::Key::Up || code == vizgfx::Keyboard::Key::Down)) {
                const int step = shift ? 128 : 64;
                const auto sz = window->getSize();
                int ww = static_cast<int>(sz.x);
                int hh = static_cast<int>(sz.y);
                if (code == vizgfx::Keyboard::Key::Left) ww -= step;
                if (code == vizgfx::Keyboard::Key::Right) ww += step;
                if (code == vizgfx::Keyboard::Key::Up) hh -= step;
                if (code == vizgfx::Keyboard::Key::Down) hh += step;
                ww = std::max(640, ww);
                hh = std::max(480, hh);
                window->setSize(vizgfx::Vector2u(static_cast<unsigned>(ww), static_cast<unsigned>(hh)));
                window_width = ww;
                window_height = hh;
                syncUIView();
            } else {
                if (focus_target_ == FocusTarget::LayerBlock &&
                    (code == vizgfx::Keyboard::Key::Left || code == vizgfx::Keyboard::Key::Right)) {
                    const int n = static_cast<int>(layer_block_images.size());
                    if (n > 0) {
                        if (code == vizgfx::Keyboard::Key::Left) focus_block_index_ = (focus_block_index_ - 1 + n) % n;
                        if (code == vizgfx::Keyboard::Key::Right) focus_block_index_ = (focus_block_index_ + 1) % n;
                    }
                } else if (focus_target_ == FocusTarget::Generated &&
                           (code == vizgfx::Keyboard::Key::Left || code == vizgfx::Keyboard::Key::Right)) {
                    const int n = static_cast<int>(generated_images.size());
                    if (n > 0) {
                        focus_generated_index_ = std::clamp(focus_generated_index_, 0, n - 1);
                        if (code == vizgfx::Keyboard::Key::Left) focus_generated_index_ = (focus_generated_index_ - 1 + n) % n;
                        if (code == vizgfx::Keyboard::Key::Right) focus_generated_index_ = (focus_generated_index_ + 1) % n;
                    }
                }
            }

            if (code == vizgfx::Keyboard::Key::S) {
                save_chord_armed_ = true;
                save_chord_consumed_ = false;
                save_chord_armed_ms_ = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
            }

            auto key_to_digit = [](vizgfx::Keyboard::Key k) -> int {
                if (k >= vizgfx::Keyboard::Key::Num0 && k <= vizgfx::Keyboard::Key::Num9) {
                    return static_cast<int>(k) - static_cast<int>(vizgfx::Keyboard::Key::Num0);
                }
                if (k >= vizgfx::Keyboard::Key::Numpad0 && k <= vizgfx::Keyboard::Key::Numpad9) {
                    return static_cast<int>(k) - static_cast<int>(vizgfx::Keyboard::Key::Numpad0);
                }
                return -1;
            };
            const int digit = key_to_digit(code);
            if (digit >= 0) {
                const bool s_down = window->isKeyPressed(vizgfx::Keyboard::Key::S);
                const uint64_t now_ms = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
                const bool chord_active = s_down || (save_chord_armed_ && (now_ms - save_chord_armed_ms_) < 800);
                initDefaultPanelsIfNeeded();
                if (chord_active) {
                    saveUILayoutToSlot(digit);
                    save_chord_consumed_ = true;
                } else {
                    loadUILayoutFromSlot(digit);
                }
            }
            continue;
        }

        if (const auto* key_rel = event.getIf<vizgfx::Event::KeyReleased>()) {
            if (key_rel->code == vizgfx::Keyboard::Key::S) {
                if (save_chord_armed_ && !save_chord_consumed_) {
                    initDefaultPanelsIfNeeded();
                    saveUILayoutToLast();
                }
                save_chord_armed_ = false;
                save_chord_consumed_ = false;
            }
            continue;
        }
    }
}

void Visualizer::requestResync() {
    resync_requested_.store(true, std::memory_order_relaxed);
}

bool Visualizer::consumeResyncRequested() {
    return resync_requested_.exchange(false, std::memory_order_relaxed);
}

void Visualizer::requestStopTraining() {
    stop_training_requested_.store(true, std::memory_order_relaxed);
}

bool Visualizer::consumeStopTrainingRequested() {
    return stop_training_requested_.exchange(false, std::memory_order_relaxed);
}

void Visualizer::updateRuntimeValidationEnabled(bool enabled) {
    // Dès que l'utilisateur a cliqué, ne jamais écraser son choix avec l'état natif.
    if (validation_control_version_.load(std::memory_order_relaxed) == 0) {
        validation_enabled_.store(enabled, std::memory_order_relaxed);
    }
}

void Visualizer::applyValidationControl(bool enabled) {
    validation_enabled_.store(enabled, std::memory_order_relaxed);
    validation_control_version_.fetch_add(1, std::memory_order_relaxed);
}

bool Visualizer::validationEnabledSnapshot() const {
    return validation_enabled_.load(std::memory_order_relaxed);
}

uint64_t Visualizer::validationControlVersion() const {
    return validation_control_version_.load(std::memory_order_relaxed);
}

uint64_t Visualizer::liveTrainParamsVersion() const {
    return live_params_version_.load(std::memory_order_relaxed);
}

Visualizer::LiveTrainParams Visualizer::liveTrainParamsSnapshot() const {
    LiveTrainParams p;
    p.overrides_enabled = live_overrides_enabled_.load(std::memory_order_relaxed);
    p.lr = live_lr_.load(std::memory_order_relaxed);
    p.lr_warmup_steps = live_lr_warmup_steps_.load(std::memory_order_relaxed);
    p.kl_beta = live_kl_beta_.load(std::memory_order_relaxed);
    p.kl_warmup_steps = live_kl_warmup_steps_.load(std::memory_order_relaxed);
    p.kl_enabled = live_kl_enabled_.load(std::memory_order_relaxed);
    p.recon_loss = recon_loss_name(live_recon_loss_index_.load(std::memory_order_relaxed));
    p.version = live_params_version_.load(std::memory_order_relaxed);
    return p;
}

void Visualizer::applyLiveTrainParams(const LiveTrainParams& params) {
    live_overrides_enabled_.store(params.overrides_enabled, std::memory_order_relaxed);
    live_lr_.store(std::max(0.0f, params.lr), std::memory_order_relaxed);
    live_lr_warmup_steps_.store(std::max(0, params.lr_warmup_steps), std::memory_order_relaxed);
    live_kl_beta_.store(std::max(0.0f, params.kl_beta), std::memory_order_relaxed);
    live_kl_warmup_steps_.store(std::max(0, params.kl_warmup_steps), std::memory_order_relaxed);
    live_kl_enabled_.store(params.kl_enabled, std::memory_order_relaxed);
    live_recon_loss_index_.store(recon_loss_index(params.recon_loss), std::memory_order_relaxed);
    external_live_params_version_.fetch_add(1, std::memory_order_relaxed);
    live_params_version_.fetch_add(1, std::memory_order_relaxed);
}

void Visualizer::updateRuntimeTrainParams(float lr, int lr_warmup_steps,
                                          float kl_beta, int kl_warmup_steps,
                                          const std::string& recon_loss) {
    if (std::isfinite(lr)) runtime_lr_.store(std::max(0.0f, lr), std::memory_order_relaxed);
    runtime_lr_warmup_steps_.store(std::max(0, lr_warmup_steps), std::memory_order_relaxed);
    if (std::isfinite(kl_beta)) runtime_kl_beta_.store(std::max(0.0f, kl_beta), std::memory_order_relaxed);
    runtime_kl_warmup_steps_.store(std::max(0, kl_warmup_steps), std::memory_order_relaxed);
    runtime_recon_loss_index_.store(recon_loss_index(recon_loss), std::memory_order_relaxed);
}

void Visualizer::showValidationFeedback(ValidationFeedbackIcon icon) {
    validation_feedback_icon_.store(static_cast<int>(icon), std::memory_order_relaxed);
}

void Visualizer::update() {
    if (!enabled) return;
    if (!window || !window->isOpen()) return;

    // Active la cible du backend dans le thread Viz. L'adaptateur SFML active
    // son contexte GL ; le moteur CPU des autres backends est toujours actif.
    window->setActive(true);

    // Premier frame: re-synchroniser les textures une fois la cible créée.
    if (!first_texture_sync_done_) {
        rebuildAllTextures();
        first_texture_sync_done_ = true;
    }

    // Auto-réparation: si certains blocks ont encore une texture invalide,
    // retenter un rebuild, mais jamais a chaque frame pour garder une UI fluide.
    if (has_layer_blocks && !layer_block_images.empty()) {
        bool has_invalid_block_tex = false;
        for (const auto& img : layer_block_images) {
            if (img.pixels.empty()) continue;
            if (img.texture.getSize().x == 0 || img.texture.getSize().y == 0) {
                has_invalid_block_tex = true;
                break;
            }
        }
        if (has_invalid_block_tex) {
            const uint64_t now_ms = steady_now_ms();
            if (last_block_texture_rebuild_ms_ == 0 || (now_ms - last_block_texture_rebuild_ms_) >= 250) {
                rebuildLayerBlockTextures();
                last_block_texture_rebuild_ms_ = now_ms;
            }
        }
    }

    // Refresh auto des textures (évite un reload UI)
    {
        const uint64_t now_ms = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
        if (last_auto_texture_refresh_ms_ == 0) {
            last_auto_texture_refresh_ms_ = now_ms;
        } else if ((now_ms - last_auto_texture_refresh_ms_) >= kAutoTextureRefreshPeriodMs) {
            rebuildAllTextures();
            last_auto_texture_refresh_ms_ = now_ms;
        }
    }

    // Toujours forcer la view UI avant de rendre.
    syncUIView();

    // Best-effort: si on a raté un événement Resized, ne jamais laisser
    // window_width/window_height et la view diverger (sinon étirement).
    {
        const auto sz = window->getSize();
        const int ww = static_cast<int>(sz.x);
        const int hh = static_cast<int>(sz.y);
        if (ww > 0 && hh > 0 && (ww != window_width || hh != window_height)) {
            window_width = ww;
            window_height = hh;
            syncUIView();
            // Ne pas toucher aux panneaux: ils restent tels quels.
        }
    }

    initDefaultPanelsIfNeeded();
    // Visibilité liée aux toggles
    panels_[static_cast<size_t>(PanelId::Generated)].visible = show_generated_images;
    panels_[static_cast<size_t>(PanelId::Training)].visible = show_training_progress;
    panels_[static_cast<size_t>(PanelId::Graph)].visible = show_loss_graph;

    maybeLoadArchitecture();

    window->clear(mimir::viz::theme::background);

    renderBackground();

    if (isPanelVisible(PanelId::Context)) {
        drawPanelChrome(PanelId::Context);
        renderContextImages();
        drawPanelScrollbar(PanelId::Context);
    } else {
        last_dataset_rect_.reset();
        last_projection_rect_.reset();
        last_understanding_rect_.reset();
    }

    if (isPanelVisible(PanelId::Output)) {
        drawPanelChrome(PanelId::Output);
        renderOutputImage();
        drawPanelScrollbar(PanelId::Output);
    } else {
        last_output_rect_.reset();
    }

    if (isPanelVisible(PanelId::Blocks)) {
        drawPanelChrome(PanelId::Blocks);
        renderLayerBlocks();
    } else {
        last_block_rects_.clear();
        last_blocks_scroll_track_rect_ = make_rect(0.f, 0.f, 0.f, 0.f);
        last_blocks_scroll_thumb_rect_ = make_rect(0.f, 0.f, 0.f, 0.f);
    }

    if (show_generated_images) {
        drawPanelChrome(PanelId::Generated);
        renderGeneratedImages();
        drawPanelScrollbar(PanelId::Generated);
    }

    if (show_training_progress) {
        drawPanelChrome(PanelId::Training);
        renderTrainingProgress();
        drawPanelScrollbar(PanelId::Training);
    }

    if (isPanelVisible(PanelId::Metrics)) {
        drawPanelChrome(PanelId::Metrics);
        renderMetrics();
        drawPanelScrollbar(PanelId::Metrics);
    } else {
        last_commands_rect_.reset();
        last_debug_json_button_rect_.reset();
        last_pause_button_rect_.reset();
        last_stop_button_rect_.reset();
        last_live_overrides_box_.reset();
        last_live_kl_enable_box_.reset();
        last_live_lr_track_.reset();
        last_live_lr_thumb_.reset();
        last_live_lrwu_track_.reset();
        last_live_lrwu_thumb_.reset();
        last_live_klb_track_.reset();
        last_live_klb_thumb_.reset();
        last_live_klwu_track_.reset();
        last_live_klwu_thumb_.reset();
    }

    if (show_loss_graph) {
        drawPanelChrome(PanelId::Graph);
        renderLossGraph();
    }

    renderFocusOutline();

    renderScriptControls();

    if (zoom_active_) {
        renderZoomOverlay();
    }
    if (show_help_overlay_) {
        renderHelpOverlay();
    }

    if (config_editor_.visible) renderConfigurationOverlay();
    if (show_model_graph_) renderModelGraphOverlay();
    if (debug_json_dialog_.visible) renderDebugJsonDialog();

    window->display();

    if (snapshot_requested_) {
        saveSnapshotPng();
        snapshot_requested_ = false;
    }
}

bool Visualizer::saveSnapshotPng() {
    if (!window || !window->isOpen()) return false;
    try {
        const vizgfx::Vector2u base_size = window->getSize();
        const vizgfx::Image base_image = window->captureImage();

        struct TipRef { const ImageData* image; std::string label; };
        std::vector<TipRef> tips;
        tips.reserve(layer_block_images.size() + 2);
        for (size_t i = 0; i < layer_block_images.size(); ++i) {
            tips.push_back(TipRef{&layer_block_images[i],
                i < layer_block_labels.size() ? layer_block_labels[i] : std::string("tip")});
        }
        if (has_projection_thumb_ && !projection_thumb_.pixels.empty())
            tips.push_back(TipRef{&projection_thumb_, projection_label});
        if (has_output_thumb_ && !output_thumb_.pixels.empty())
            tips.push_back(TipRef{&output_thumb_, output_thumb_label_});

        constexpr unsigned thumb = 120;
        constexpr unsigned cell_w = 140;
        constexpr unsigned cell_h = 158;
        constexpr unsigned margin = 16;
        constexpr unsigned header_h = 42;
        const unsigned sheet_width = std::max(base_size.x, 640u);
        const unsigned columns = std::max(1u, (sheet_width - 2 * margin) / cell_w);
        const unsigned rows = std::max(1u, static_cast<unsigned>(
            (tips.size() + columns - 1) / columns));
        const unsigned sheet_height = header_h + rows * cell_h + margin;

        vizgfx::RenderTexture sheet(vizgfx::Vector2u(sheet_width, sheet_height));
        sheet.clear(vizgfx::Color(24, 24, 30));
        if (font_loaded) {
            vizgfx::Text title(font, "Toutes les tips", 18);
            title.setPosition(vizgfx::Vector2f(static_cast<float>(margin), 8.0f));
            title.setFillColor(vizgfx::Color(240, 240, 245));
            sheet.draw(title);
        }

        for (size_t i = 0; i < tips.size(); ++i) {
            const unsigned column = static_cast<unsigned>(i) % columns;
            const unsigned row = static_cast<unsigned>(i) / columns;
            const float x = static_cast<float>(margin + column * cell_w);
            const float y = static_cast<float>(header_h + row * cell_h);

            vizgfx::RectangleShape frame(vizgfx::Vector2f(static_cast<float>(thumb + 4),
                                                   static_cast<float>(thumb + 4)));
            frame.setPosition(vizgfx::Vector2f(x - 2.0f, y - 2.0f));
            frame.setFillColor(vizgfx::Color::Transparent);
            frame.setOutlineColor(vizgfx::Color(150, 150, 175));
            frame.setOutlineThickness(1.0f);
            sheet.draw(frame);

            vizgfx::Sprite sprite(tips[i].image->texture);
            sprite.setScale(tips[i].image->sprite.getScale());
            position_sprite_centered_in_box(sprite, x, y, static_cast<float>(thumb));
            sheet.draw(sprite);

            if (font_loaded) {
                const ParsedVizLabel parsed = parse_viz_label(tips[i].label);
                std::string label = clamp_text_end(format_tip_display_label(parsed), 20);
                vizgfx::Text text(font, vizgfx::String::fromUtf8(label.begin(), label.end()), 12);
                text.setPosition(vizgfx::Vector2f(x + 2.0f, y + thumb + 5.0f));
                text.setFillColor(vizgfx::Color(235, 235, 240));
                sheet.draw(text);
            }
        }
        sheet.display();

        const vizgfx::Image sheet_image = sheet.getTexture().copyToImage();
        vizgfx::Image snapshot(vizgfx::Vector2u(sheet_width, base_size.y + sheet_height),
                           vizgfx::Color(24, 24, 30));
        if (!snapshot.copy(base_image, vizgfx::Vector2u(0, 0)) ||
            !snapshot.copy(sheet_image, vizgfx::Vector2u(0, base_size.y))) {
            std::cerr << "[Viz] Echec assemblage du snapshot PNG" << std::endl;
            return false;
        }

        const auto now = std::chrono::system_clock::now();
        const auto now_time = std::chrono::system_clock::to_time_t(now);
        const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count() % 1000;
        std::tm local_time{};
#ifdef _WIN32
        localtime_s(&local_time, &now_time);
#else
        localtime_r(&now_time, &local_time);
#endif
        std::ostringstream filename;
        filename << "viz_snapshot_" << std::put_time(&local_time, "%Y%m%d_%H%M%S")
                 << '_' << std::setw(3) << std::setfill('0') << millis << ".png";
        const std::filesystem::path directory("snapshots");
        std::error_code ec;
        std::filesystem::create_directories(directory, ec);
        const std::filesystem::path path = directory / filename.str();
        if (ec || !snapshot.saveToFile(path)) {
            std::cerr << "[Viz] Echec snapshot PNG: " << path.string() << std::endl;
            return false;
        }
        last_snapshot_path_ = path.string();
        std::cout << "[Viz] Snapshot PNG: " << last_snapshot_path_ << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[Viz] Echec snapshot PNG: " << e.what() << std::endl;
        return false;
    }
}

void Visualizer::renderFocusOutline() {
    if (!window) return;
    if (zoom_active_) return; // le zoom a déjà sa propre emphase

    // Ne pas afficher si le panneau associé est fermé
    if (focus_target_ == FocusTarget::LayerBlock) {
        if (!isPanelVisible(PanelId::Blocks)) return;
    } else if (focus_target_ == FocusTarget::Generated) {
        if (!isPanelVisible(PanelId::Generated)) return;
    } else if (focus_target_ == FocusTarget::Output) {
        if (!isPanelVisible(PanelId::Output)) return;
    } else if (focus_target_ == FocusTarget::Training) {
        if (!isPanelVisible(PanelId::Training)) return;
    } else if (focus_target_ == FocusTarget::Metrics) {
        if (!isPanelVisible(PanelId::Metrics)) return;
    } else if (focus_target_ == FocusTarget::Graph) {
        if (!isPanelVisible(PanelId::Graph)) return;
    } else {
        if (!isPanelVisible(PanelId::Context)) return;
    }

    std::optional<vizgfx::FloatRect> r;
    bool focus_whole_panel = false;
    if (focus_target_ == FocusTarget::Dataset) {
        r = last_dataset_rect_;
    } else if (focus_target_ == FocusTarget::Projection) {
        r = last_projection_rect_;
    } else if (focus_target_ == FocusTarget::Understanding) {
        r = last_understanding_rect_;
    } else if (focus_target_ == FocusTarget::LayerBlock) {
        if (!last_block_rects_.empty()) {
            const int n = static_cast<int>(last_block_rects_.size());
            const int idx = (n > 0) ? std::clamp(focus_block_index_, 0, n - 1) : 0;
            if (idx >= 0 && idx < n) r = last_block_rects_[static_cast<size_t>(idx)];
        }
    } else if (focus_target_ == FocusTarget::Generated) {
        if (!last_generated_rects_.empty() && last_generated_rects_.size() == last_generated_indices_.size()) {
            // Retrouver la rect correspondant à l'index sélectionné
            for (size_t i = 0; i < last_generated_indices_.size(); ++i) {
                if (last_generated_indices_[i] == focus_generated_index_) {
                    r = last_generated_rects_[i];
                    break;
                }
            }
            // Fallback: clamp sur la première rect visible
            if (!r.has_value() && !last_generated_rects_.empty()) {
                r = last_generated_rects_.front();
            }
        }
    } else if (focus_target_ == FocusTarget::Output) {
        r = last_output_rect_;
    } else if (focus_target_ == FocusTarget::Training) {
        r = panelRect(PanelId::Training);
        focus_whole_panel = true;
    } else if (focus_target_ == FocusTarget::Metrics) {
        r = panelRect(PanelId::Metrics);
        focus_whole_panel = true;
    } else if (focus_target_ == FocusTarget::Graph) {
        r = panelRect(PanelId::Graph);
        focus_whole_panel = true;
    }

    if (!r.has_value()) return;
    if (r->size.x <= 0.f || r->size.y <= 0.f) return;

    // Accent selon la zone focusée
    vizgfx::Color col = vizgfx::Color(245, 245, 250, 230);
    if (focus_target_ == FocusTarget::LayerBlock) {
        col = with_alpha(panelAccent(PanelId::Blocks), 235);
    } else if (focus_target_ == FocusTarget::Generated) {
        col = with_alpha(panelAccent(PanelId::Generated), 235);
    } else if (focus_target_ == FocusTarget::Output) {
        col = with_alpha(panelAccent(PanelId::Output), 235);
    } else if (focus_target_ == FocusTarget::Training) {
        col = with_alpha(panelAccent(PanelId::Training), 235);
    } else if (focus_target_ == FocusTarget::Metrics) {
        col = with_alpha(panelAccent(PanelId::Metrics), 235);
    } else if (focus_target_ == FocusTarget::Graph) {
        col = with_alpha(panelAccent(PanelId::Graph), 235);
    } else {
        col = with_alpha(panelAccent(PanelId::Context), 235);
    }

    PanelId focus_panel = PanelId::Context;
    if (focus_target_ == FocusTarget::LayerBlock) focus_panel = PanelId::Blocks;
    else if (focus_target_ == FocusTarget::Generated) focus_panel = PanelId::Generated;
    else if (focus_target_ == FocusTarget::Output) focus_panel = PanelId::Output;
    else if (focus_target_ == FocusTarget::Training) focus_panel = PanelId::Training;
    else if (focus_target_ == FocusTarget::Metrics) focus_panel = PanelId::Metrics;
    else if (focus_target_ == FocusTarget::Graph) focus_panel = PanelId::Graph;

    if (focus_whole_panel) {
        vizgfx::RectangleShape outline(vizgfx::Vector2f(r->size.x, r->size.y));
        outline.setPosition(r->position);
        outline.setFillColor(vizgfx::Color::Transparent);
        outline.setOutlineColor(col);
        outline.setOutlineThickness(3);
        window->draw(outline);
        return;
    }
    const auto area = panelContentRect(focus_panel);
    const vizgfx::FloatRect clip = make_rect(
        area.position.x, area.position.y, area.size.x, area.size.y);
    if (!clip.findIntersection(*r).has_value()) return;

    const vizgfx::View old_view = window->getView();
    const float ww = static_cast<float>(std::max(1, window_width));
    const float hh = static_cast<float>(std::max(1, window_height));
    vizgfx::View view(clip);
    view.setViewport(make_rect(
        area.position.x / ww, area.position.y / hh,
        area.size.x / ww, area.size.y / hh));
    window->setView(view);
    vizgfx::RectangleShape outline(vizgfx::Vector2f(r->size.x, r->size.y));
    outline.setPosition(vizgfx::Vector2f(r->position.x, r->position.y));
    outline.setFillColor(vizgfx::Color::Transparent);
    outline.setOutlineColor(col);
    outline.setOutlineThickness(3);
    window->draw(outline);
    window->setView(old_view);
}

void Visualizer::rebuildAllTextures() {
    auto rebuild_one = [&](ImageData& img) {
        if (img.w <= 0 || img.h <= 0 || img.display_size <= 0) return;
        if (img.channels != 1 && img.channels != 3 && img.channels != 4) return;
        if (img.pixels.empty()) return;
        createImageTexture(img, img.w, img.h, img.channels, img.display_size);
    };

    if (has_dataset_image) rebuild_one(dataset_image);
    if (has_projection_image) rebuild_one(projection_image);
    if (has_projection_thumb_) rebuild_one(projection_thumb_);
    if (has_understanding_image) rebuild_one(understanding_image);
    for (auto& img : generated_images) rebuild_one(img);
    if (has_output_thumb_) {
        rebuild_one(output_image_);
        rebuild_one(output_thumb_);
    }
    for (size_t i = 0; i < layer_block_images.size(); ++i) {
        const std::string label = (i < layer_block_labels.size()) ? layer_block_labels[i] : std::string();
        createLayerBlockTexture(layer_block_images[i], label);
    }
}

void Visualizer::renderHelpOverlay() {
    if (!window) return;
    help_row_rects_.clear();
    const float ww=static_cast<float>(std::max(1,window_width));
    const float hh=static_cast<float>(std::max(1,window_height));
    const float margin=ww<600.f?12.f:24.f;
    const float cw=std::min(1080.f,std::max(1.f,ww-2.f*margin));
    const float ch=std::min(900.f,std::max(1.f,hh-2.f*margin));
    const float cx=(ww-cw)*0.5f, cy=(hh-ch)*0.5f;
    const vizgfx::Color white(235,242,250), muted(175,190,208), accent(90,210,220);
    vizgfx::RectangleShape shade({ww,hh});shade.setFillColor(vizgfx::Color(0,0,0,190));window->draw(shade);
    vizgfx::RectangleShape card({cw,ch});card.setPosition({cx,cy});
    card.setFillColor(vizgfx::Color(22,30,43,250));card.setOutlineThickness(1.f);
    card.setOutlineColor(vizgfx::Color(65,95,120));window->draw(card);
    auto text=[&](float x,float y,unsigned size,const std::string& value,vizgfx::Color color,bool bold=false) {
        if(!font_loaded) return;
        vizgfx::Text t(font);t.setCharacterSize(size);t.setFillColor(color);
        if(bold) t.setStyle(vizgfx::Text::Bold);
        t.setPosition({x,y});t.setString(vizgfx::String::fromUtf8(value.begin(),value.end()));window->draw(t);
    };
    text(cx+20.f,cy+14.f,22,"Guide de la Viz",white,true);
    text(cx+20.f,cy+48.f,14,std::string("Raccourcis et commandes  |  ")+vizBackendName(),muted);
    help_close_rect_=make_rect(cx+cw-48.f,cy+12.f,28.f,28.f);
    vizgfx::RectangleShape close(help_close_rect_->size);close.setPosition(help_close_rect_->position);
    close.setFillColor(vizgfx::Color(70,85,105));window->draw(close);
    text(help_close_rect_->position.x+8.f,help_close_rect_->position.y+4.f,16,"X",white,true);
    const float body_x=cx+20.f, body_y=cy+82.f;
    const bool compact_footer=cw<620.f;
    const float body_w=std::max(1.f,cw-40.f), body_h=std::max(1.f,ch-(compact_footer?150.f:124.f));
    help_body_rect_=make_rect(body_x,body_y,body_w,body_h);
    help_columns_=body_w>=860.f?2:1;
    const float column_gap=32.f;
    const float column_w=(body_w-(help_columns_-1)*column_gap)/help_columns_;
    const bool stacked=column_w<360.f;
    const float key_w=stacked?column_w:136.f;
    const float description_w=stacked?column_w:std::max(1.f,column_w-key_w-12.f);
    if (!font_loaded) return;
    struct Section { std::string title; std::vector<std::pair<std::string,std::string>> rows; };
    const std::vector<Section> sections={
        {"Navigation",{{"Tab","Parcourir les éléments sélectionnables."},
            {"F1 / F2 / F3","Dataset / Projection / Understanding."},
            {"F4 / F5 / F6","Blocks / Generated / Sortie."},
            {"F7 / F8 / F9","Training / Metrics / Loss."},
            {"G / P / B","Afficher ou masquer le graphe, le prompt et les boîtes."}}},
        {"Zoom et inspection des tenseurs",{{"Z / Entrée","Agrandir l’élément sélectionné."},
            {"T","Basculer en taille réelle 1:1."},
            {"F","Picker : sections de features, canaux actifs et detections surlignees."},
            {"Gauche / Droite","Parcourir les tips ou les images."},
            {"Survol","Lire la valeur brute et son indice dans la cellule encadrée."},
            {"Glisser","Sélectionner une zone et analyser ses valeurs capturées."},
            {"Molette","Parcourir les informations et les valeurs sélectionnés."},
            {"Clic droit / Esc","Effacer la sélection. Esc ferme le zoom si elle est vide."}}},
        {"Commandes et configuration",{{"Commandes","En haut de Metrics : Play/Pause, Stop, Validation, Skips, Live/Native, KL et Recon."},
            {"Play / Pause","Suspendre ou reprendre entre deux étapes de calcul."},
            {"C","Ouvrir ou fermer la configuration du modèle."},
            {"N","Graphe du modele et planner actif. Fleches/molette : parcourir, clic : inspecter."},
            {"D / DebugJSON","Choisir le dossier et le fichier dans la boite modale, puis exporter."},
            {"Entrée / Esc","Valider une édition ou l'annuler."},
            {"Reglages","Cliquer sur une valeur pour la saisir, ou utiliser les curseurs."}}},
        {"Affichage des blocs",{{"M / K / A","Changer le rendu Heatmap/Reel, la palette et le lissage."},
            {"ACT / NORM","Afficher ou masquer les activations et normalisations."},
            {"TIPS","Limiter les vignettes : Entrée valide, vide signifie ALL."},
            {"R","Resynchroniser les textures et les informations d'architecture."}}},
        {"Panneaux et export",{{"Titre / Poignée","Glisser pour déplacer ou redimensionner un panneau."},
            {"[X]","Masquer un panneau."},
            {"S","Mémoriser la disposition."},
            {"S + 0-9","Sauvegarder un slot de disposition."},
            {"0-9","Charger un slot de disposition."},
            {"Ctrl + fleches","Modifier la taille de la fenêtre."},
            {"E","Capturer la scene en PNG."}}},
        {"Contrôles du script",{{"Viz.configure","Définir les panneaux et les boutons du script."},
            {"Viz.poll_events","Recevoir les événements du script."},
            {"Aide du script",script_scene_.value("help",std::string("Les contrôles du script apparaissent au-dessus des panneaux."))}}}
    };
    auto wrap=[&](const std::string& value,float width,unsigned size) {
        std::vector<std::string> lines;std::istringstream words(value);std::string word,line;
        vizgfx::Text measure(font);measure.setCharacterSize(size);
        while(words>>word) {
            const std::string candidate=line.empty()?word:line+" "+word;
            measure.setString(vizgfx::String::fromUtf8(candidate.begin(),candidate.end()));
            if(!line.empty() && measure.getLocalBounds().size.x>width) {lines.push_back(line);line=word;}
            else line=candidate;
            // Break oversized tokens at UTF-8 character boundaries as well.
            measure.setString(vizgfx::String::fromUtf8(line.begin(),line.end()));
            if(measure.getLocalBounds().size.x>width) {
                std::string chunk;
                for(size_t i=0;i<line.size();) {
                    const unsigned char lead=static_cast<unsigned char>(line[i]);
                    const size_t count=(lead&0x80)==0?1:((lead&0xE0)==0xC0?2:((lead&0xF0)==0xE0?3:4));
                    const auto character=line.substr(i,std::min(count,line.size()-i));
                    const auto next=chunk+character;
                    measure.setString(vizgfx::String::fromUtf8(next.begin(),next.end()));
                    if(!chunk.empty() && measure.getLocalBounds().size.x>width) {lines.push_back(chunk);chunk=character;}
                    else chunk=next;
                    i+=character.size();
                }
                line=chunk;
            }
        }
        if(!line.empty()) lines.push_back(line);
        if(lines.empty()) lines.push_back("");
        return lines;
    };
    const vizgfx::View saved_view=window->getView();
    vizgfx::View clipped(*help_body_rect_);
    clipped.setViewport(make_rect(body_x/ww,body_y/hh,body_w/ww,body_h/hh));window->setView(clipped);
    float max_height=0.f;
    for(int column=0;column<help_columns_;++column) {
        const float x=body_x+column*(column_w+column_gap);
        float y=body_y-help_scroll_y_;
        const size_t first=help_columns_==1?0:column*3;
        const size_t end=help_columns_==1?sections.size():first+3;
        for(size_t index=first;index<end;++index) {
            for(const auto& line:wrap(sections[index].title,column_w,16)) {text(x,y,16,line,accent,true);y+=24.f;}
            y+=8.f;
            for(const auto& item:sections[index].rows) {
                const auto key_lines=wrap(item.first,key_w,14);
                const auto lines=wrap(item.second,description_w,14);
                const float height=stacked?22.f*(key_lines.size()+lines.size()):22.f*std::max(key_lines.size(),lines.size());
                help_row_rects_.push_back(make_rect(x,y,column_w,height));
                for(size_t line=0;line<key_lines.size();++line) text(x,y+line*22.f,14,key_lines[line],white,true);
                const float dx=stacked?x:x+key_w+12.f;
                const float dy=stacked?y+key_lines.size()*22.f:y;
                for(size_t line=0;line<lines.size();++line) text(dx,dy+line*22.f,14,lines[line],muted);
                y+=height+10.f;
            }
            y+=16.f;
        }
        max_height=std::max(max_height,y+help_scroll_y_-body_y);
    }
    window->setView(saved_view);
    help_scroll_max_=std::max(0.f,max_height-body_h);
    help_scroll_y_=std::clamp(help_scroll_y_,0.f,help_scroll_max_);
    if(compact_footer) {
        text(cx+20.f,cy+ch-52.f,13,"H / Esc : fermer",muted);
        text(cx+20.f,cy+ch-28.f,13,"Molette / Haut / Bas : défiler",muted);
    } else text(cx+20.f,cy+ch-28.f,13,"H / Esc : fermer   |   Molette / Haut / Bas : défiler",muted);
    if(help_scroll_max_>0.f) {
        const float ratio=body_h/std::max(body_h,max_height);
        const float thumb_h=std::max(20.f,body_h*ratio);
        vizgfx::RectangleShape track({3.f,body_h});track.setPosition({cx+cw-10.f,body_y});
        track.setFillColor(vizgfx::Color(55,70,90));window->draw(track);
        vizgfx::RectangleShape thumb({3.f,thumb_h});
        thumb.setPosition({cx+cw-10.f,body_y+(body_h-thumb_h)*help_scroll_y_/help_scroll_max_});
        thumb.setFillColor(accent);window->draw(thumb);
    }
}

void Visualizer::updatePickerSelection(vizgfx::Vector2f point) {
    if (!picker_image_rect_ || !picker_image_clip_ || picker_image_scale_<=0.f) return;
    point.x=std::clamp(point.x,picker_image_clip_->position.x,
        picker_image_clip_->position.x+picker_image_clip_->size.x-0.001f);
    point.y=std::clamp(point.y,picker_image_clip_->position.y,
        picker_image_clip_->position.y+picker_image_clip_->size.y-0.001f);
    picker_end_x_=std::clamp(static_cast<int>((point.x-picker_image_rect_->position.x)/picker_image_scale_),0,picker_selection_source_.w-1);
    picker_end_y_=std::clamp(static_cast<int>((point.y-picker_image_rect_->position.y)/picker_image_scale_),0,picker_selection_source_.h-1);
    picker_selection_summary_=TensorSelection::analyze(picker_selection_source_.tensor_values,
        picker_selection_source_.tensor_indices,picker_selection_source_.w,picker_selection_source_.h,
        picker_selection_source_.tensor_sample_channels,picker_anchor_x_,picker_anchor_y_,picker_end_x_,picker_end_y_);
}

void Visualizer::renderZoomOverlay() {
    last_picker_text_.clear();
    last_picker_rect_.reset();
    picker_image_rect_.reset(); picker_image_clip_.reset(); picker_selection_rect_.reset();
    if (!window) return;

    std::optional<PanelId> zoomed_panel;
    if (focus_target_ == FocusTarget::Training) zoomed_panel = PanelId::Training;
    else if (focus_target_ == FocusTarget::Metrics) zoomed_panel = PanelId::Metrics;
    else if (focus_target_ == FocusTarget::Graph) zoomed_panel = PanelId::Graph;

    if (zoomed_panel.has_value()) {
        picker_selection_active_=picker_selecting_=false;
        vizgfx::RectangleShape overlay(vizgfx::Vector2f(
            static_cast<float>(window_width), static_cast<float>(window_height)));
        overlay.setPosition(vizgfx::Vector2f(0.0f, 0.0f));
        overlay.setFillColor(vizgfx::Color(0, 0, 0, 220));
        window->draw(overlay);

        const size_t panel_index = static_cast<size_t>(*zoomed_panel);
        const Panel saved_panel = panels_[panel_index];
        const float saved_scroll_y = panel_scroll_y_[panel_index];
        const float saved_scroll_max = panel_scroll_max_[panel_index];
        const vizgfx::FloatRect saved_scroll_track = panel_scroll_track_rects_[panel_index];
        const vizgfx::FloatRect saved_scroll_thumb = panel_scroll_thumb_rects_[panel_index];
        const float pad = 28.0f;
        auto& panel = panels_[panel_index];
        panel.pos = vizgfx::Vector2f(pad, pad);
        panel.size = vizgfx::Vector2f(
            std::max(kPanelMinW, static_cast<float>(window_width) - 2.0f * pad),
            std::max(kPanelMinH, static_cast<float>(window_height) - 2.0f * pad));
        panel.visible = true;

        drawPanelChrome(*zoomed_panel);
        if (*zoomed_panel == PanelId::Training) renderTrainingProgress();
        else if (*zoomed_panel == PanelId::Metrics) renderMetrics();
        else renderLossGraph();
        drawPanelScrollbar(*zoomed_panel);

        panels_[panel_index] = saved_panel;
    panel_scroll_y_[panel_index] = saved_scroll_y;
    panel_scroll_max_[panel_index] = saved_scroll_max;
    panel_scroll_track_rects_[panel_index] = saved_scroll_track;
    panel_scroll_thumb_rects_[panel_index] = saved_scroll_thumb;
        return;
    }

    const ImageData* img = nullptr;
    std::string label;
    if (focus_target_ == FocusTarget::Dataset && has_dataset_image) {
        img = &dataset_image;
        label = dataset_label.empty() ? std::string("dataset") : dataset_label;
    } else if (focus_target_ == FocusTarget::Projection && has_projection_image) {
        img = &projection_image;
        label = projection_label.empty() ? std::string("projection") : projection_label;
    } else if (focus_target_ == FocusTarget::Understanding && has_understanding_image) {
        img = &understanding_image;
        label = understanding_label.empty() ? std::string("understanding") : understanding_label;
    } else if (focus_target_ == FocusTarget::LayerBlock && !layer_block_images.empty()) {
        const int n = static_cast<int>(layer_block_images.size());
        const int idx = (n > 0) ? std::clamp(focus_block_index_, 0, n - 1) : 0;
        img = &layer_block_images[static_cast<size_t>(idx)];
        if (idx >= 0 && idx < static_cast<int>(layer_block_labels.size())) {
            label = layer_block_labels[static_cast<size_t>(idx)];
        }
    } else if (focus_target_ == FocusTarget::Generated && !generated_images.empty()) {
        const int n = static_cast<int>(generated_images.size());
        const int idx = (n > 0) ? std::clamp(focus_generated_index_, 0, n - 1) : 0;
        img = &generated_images[static_cast<size_t>(idx)];
        label = img->prompt.empty() ? std::string("generated") : (std::string("generated | ") + img->prompt);
    } else if (focus_target_ == FocusTarget::Output && has_output_thumb_) {
        img = &output_image_;
        label = output_thumb_label_.empty() ? std::string("sortie") : output_thumb_label_;
    }

    if (!img) { picker_selection_active_=picker_selecting_=false; return; }
    const auto ts = img->texture.getSize();
    if (ts.x == 0 || ts.y == 0) return;

    const BlockFrame* selected_frame = nullptr;
    for (const auto& frame : last_block_frames_) if (frame.label == label) {
        selected_frame = &frame;
        break;
    }

    const ParsedVizLabel parsed = parse_viz_label(label);
    const std::string display_tag = display_tip_tag(parsed);
    const vizgfx::Color accent = color_for_tag(display_tag);

    uint8_t pixel_min = 0;
    uint8_t pixel_max = 0;
    double pixel_mean = 0.0;
    double pixel_stddev = 0.0;
    double nonzero_ratio = 0.0;
    if (!img->pixels.empty()) {
        const size_t stride = std::max<size_t>(1, img->pixels.size() / 200000ULL);
        uint8_t min_value = 255;
        uint8_t max_value = 0;
        size_t sample_count = 0;
        size_t nonzero_count = 0;
        double sum = 0.0;
        double sum_squared = 0.0;
        for (size_t index = 0; index < img->pixels.size(); index += stride) {
            const uint8_t value = img->pixels[index];
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
            sum += static_cast<double>(value);
            sum_squared += static_cast<double>(value) * static_cast<double>(value);
            if (value != 0) ++nonzero_count;
            ++sample_count;
        }
        if (sample_count > 0) {
            pixel_min = min_value;
            pixel_max = max_value;
            pixel_mean = sum / static_cast<double>(sample_count);
            const double variance = std::max(
                0.0, sum_squared / static_cast<double>(sample_count) - pixel_mean * pixel_mean);
            pixel_stddev = std::sqrt(variance);
            nonzero_ratio = 100.0 * static_cast<double>(nonzero_count) /
                static_cast<double>(sample_count);
        }
    }

    vizgfx::RectangleShape bg(vizgfx::Vector2f(static_cast<float>(window_width), static_cast<float>(window_height)));
    bg.setPosition(vizgfx::Vector2f(0, 0));
    bg.setFillColor(vizgfx::Color(8, 9, 12, 238));
    window->draw(bg);

    const float pad = 28.0f;
    const float gap = 22.0f;
    const float sidebar_w = std::clamp(
        static_cast<float>(window_width) * 0.40f, 360.0f, 520.0f);
    const float image_area_w = std::max(
        120.0f, static_cast<float>(window_width) - 2.0f * pad - gap - sidebar_w);
    const float image_area_h = std::max(
        120.0f, static_cast<float>(window_height) - 2.0f * pad);
    const float box = std::min(image_area_w, image_area_h);

    vizgfx::Sprite spr(img->texture);
    const float sx = box / std::max(1.0f, static_cast<float>(ts.x));
    const float sy = box / std::max(1.0f, static_cast<float>(ts.y));
    const float sc = zoom_actual_size_ ? 1.0f : std::min(sx, sy);
    spr.setScale(vizgfx::Vector2f(sc, sc));

    const float dw = static_cast<float>(ts.x) * sc;
    const float dh = static_cast<float>(ts.y) * sc;
    const float image_x = pad + (image_area_w - dw) * 0.5f;
    const float image_y = pad + (image_area_h - dh) * 0.5f;
    picker_image_rect_=make_rect(image_x,image_y,dw,dh);
    picker_image_clip_=picker_image_rect_->findIntersection(make_rect(pad,pad,image_area_w,image_area_h));
    picker_image_scale_=sc;
    if (selected_frame && selected_frame->tensor_sample_channels>0 && !selected_frame->tensor_values.empty()) {
        picker_source_=*selected_frame; picker_source_raw_=true;
    } else {
        picker_source_={}; picker_source_.w=img->w; picker_source_.h=img->h; picker_source_.label=label;
        picker_source_.tensor_sample_channels=img->channels;
        picker_source_.tensor_values.assign(img->pixels.begin(),img->pixels.end());
        picker_source_raw_=false;
    }
    if (picker_selection_active_ && (picker_selection_source_.label!=label ||
        picker_selection_source_.w!=img->w || picker_selection_source_.h!=img->h)) {
        picker_selection_active_=picker_selecting_=false; zoom_info_scroll_=0.f;
    }


    vizgfx::RectangleShape image_frame(vizgfx::Vector2f(dw + 8.0f, dh + 8.0f));
    image_frame.setPosition(vizgfx::Vector2f(image_x - 4.0f, image_y - 4.0f));
    image_frame.setFillColor(vizgfx::Color(15, 16, 20, 255));
    image_frame.setOutlineColor(with_alpha(accent, 230));
    image_frame.setOutlineThickness(2.0f);
    window->draw(image_frame);

    spr.setPosition(vizgfx::Vector2f(image_x, image_y));
    window->draw(spr);
    if (focus_target_ == FocusTarget::Dataset && show_detection_overlay_) {
        DatasetBBoxOverlay::draw(*window, dataset_boxes_, image_x, image_y, dw, dh);
    }

    const float zoom_box_x = pad + (image_area_w - box) * 0.5f;
    const float zoom_box_y = pad + (image_area_h - box) * 0.5f;
    if (focus_target_ == FocusTarget::Dataset) {
        renderDetectionOverlay(*img, zoom_box_x, zoom_box_y, box,
                               DetectionTarget::Dataset);
    } else if (focus_target_ == FocusTarget::Output ||
               (focus_target_ == FocusTarget::Generated &&
                img->prompt == output_image_.prompt &&
                img->w == output_image_.w && img->h == output_image_.h)) {
        renderDetectionOverlay(*img, zoom_box_x, zoom_box_y, box,
                               DetectionTarget::Output);
    }

    const float sidebar_x = pad + image_area_w + gap;
    const float sidebar_h = image_area_h;
    vizgfx::RectangleShape sidebar(vizgfx::Vector2f(sidebar_w, sidebar_h));
    sidebar.setPosition(vizgfx::Vector2f(sidebar_x, pad));
    sidebar.setFillColor(vizgfx::Color(20, 22, 28, 250));
    sidebar.setOutlineColor(with_alpha(accent, 190));
    sidebar.setOutlineThickness(2.0f);
    window->draw(sidebar);

    vizgfx::RectangleShape accent_bar(vizgfx::Vector2f(6.0f, sidebar_h));
    accent_bar.setPosition(vizgfx::Vector2f(sidebar_x, pad));
    accent_bar.setFillColor(accent);
    window->draw(accent_bar);

    if (picker_selection_active_) {
        const auto& selection=picker_selection_summary_;
        const float x=image_x+selection.x0*sc, y=image_y+selection.y0*sc;
        const float w=(selection.x1-selection.x0+1)*sc, h=(selection.y1-selection.y0+1)*sc;
        picker_selection_rect_=make_rect(x,y,w,h);
        if (const auto clipped=picker_selection_rect_->findIntersection(make_rect(pad,pad,image_area_w,image_area_h))) {
            vizgfx::RectangleShape region(clipped->size);region.setPosition(clipped->position);
            region.setFillColor(vizgfx::Color(90,225,255,45));
            region.setOutlineColor(vizgfx::Color(100,235,255));region.setOutlineThickness(2.f);window->draw(region);
        }
        if (font_loaded) {
            float yrow=pad+16.f-zoom_info_scroll_;
            const size_t chars=static_cast<size_t>(std::max(12.f,(sidebar_w-40.f)/7.2f));
            auto row=[&](const std::string& line, bool heading=false) {
                if(yrow>=pad+8.f && yrow+20.f<pad+sidebar_h-8.f) {
                    vizgfx::Text text(font);text.setCharacterSize(heading?15:13);
                    text.setFillColor(heading?vizgfx::Color(100,235,255):vizgfx::Color(225,230,240));
                    text.setPosition({sidebar_x+20.f,yrow});
                    const auto shown=clamp_text_end(line,chars);
                    text.setString(vizgfx::String::fromUtf8(shown.begin(),shown.end()));window->draw(text);
                }
                yrow+=24.f;
            };
            auto numeric=[](double value) {std::ostringstream text;text<<std::setprecision(9)<<value;return text.str();};
            row("SELECTION / ANALYSE",true);
            row(picker_selection_raw_?"Tenseur : capture figee":"Pixels affiches uint8 : capture figee");
            row("Glisser : selectionner | Clic droit / Esc : effacer");
            row("Molette : parcourir toutes les valeurs");
            row("Zone ["+std::to_string(selection.x0)+","+std::to_string(selection.y0)+"] -> ["+
                std::to_string(selection.x1)+","+std::to_string(selection.y1)+"]");
            row("Canaux captures : "+std::to_string(picker_selection_source_.tensor_sample_channels));
            row("Valeurs uniques : "+std::to_string(selection.values.size()));
            row("Finies : "+std::to_string(selection.finite)+" | zeros : "+std::to_string(selection.zeros));
            row("NaN : "+std::to_string(selection.nan)+" | Inf : "+std::to_string(selection.inf));
            row("Min : "+(selection.finite?numeric(selection.min):"--")+" | Max : "+(selection.finite?numeric(selection.max):"--"));
            row("Moyenne : "+(selection.finite?numeric(selection.mean):"--"));
            row("Ecart-type : "+(selection.finite?numeric(selection.stddev):"--"));
            row("Norme L2 (valeurs finies) : "+numeric(selection.l2));
            if (picker_selection_raw_) row("Echantillons de la preview, avant colorisation");
            row("INDICES / VALEURS",true);
            for (const auto& entry:selection.values) {
                if(yrow>=pad+8.f && yrow+20.f<pad+sidebar_h-8.f)
                    row("["+std::to_string(entry.first)+"] = "+numeric(entry.second));
                else yrow+=24.f;
            }
            zoom_info_scroll_max_=std::max(0.f,yrow+zoom_info_scroll_-(pad+sidebar_h)+16.f);
        }
        return;
    }
    if (!font_loaded) return;

    const float text_x = sidebar_x + 20.0f;
    const float text_w = std::max(120.0f, sidebar_w - 40.0f);
    const float key_column_w = std::clamp(text_w * 0.30f, 104.0f, 132.0f);
    const size_t max_chars = static_cast<size_t>(std::max(12.0f, text_w / 7.2f));
    float text_y = pad + 16.0f - zoom_info_scroll_;
    auto visible_row = [&](float height) { return text_y >= pad + 8.0f && text_y + height < pad + sidebar_h - 8.0f; };

    auto draw_text = [&](const std::string& value, unsigned size, vizgfx::Color color, bool bold = false) {
        vizgfx::Text text(font);
        text.setCharacterSize(size);
        text.setFillColor(color);
        if (bold) text.setStyle(vizgfx::Text::Bold);
        text.setPosition(vizgfx::Vector2f(text_x, text_y));
        const std::string shown = clamp_text_end(value, max_chars);
        text.setString(vizgfx::String::fromUtf8(shown.begin(), shown.end()));
        if (visible_row(static_cast<float>(size) + 5.0f)) window->draw(text);
        text_y += static_cast<float>(size) + 5.0f;
    };

    auto draw_section = [&](const std::string& title) {
        text_y += 4.0f;
        vizgfx::RectangleShape rule(vizgfx::Vector2f(text_w, 1.0f));
        rule.setPosition(vizgfx::Vector2f(text_x, text_y));
        rule.setFillColor(with_alpha(accent, 120));
        if (visible_row(1.0f)) window->draw(rule);
        text_y += 5.0f;
        draw_text(title, 12, with_alpha(accent, 255), true);
    };

    auto draw_value = [&](const std::string& key, const std::string& value) {
        vizgfx::Text key_text(font);
        key_text.setCharacterSize(11);
        key_text.setStyle(vizgfx::Text::Bold);
        key_text.setFillColor(with_alpha(accent, 210));
        key_text.setPosition(vizgfx::Vector2f(text_x, text_y));
        key_text.setString(vizgfx::String::fromUtf8(key.begin(), key.end()));
        if (visible_row(20.0f)) window->draw(key_text);

        vizgfx::Text value_text(font);
        value_text.setCharacterSize(12);
        value_text.setFillColor(vizgfx::Color(220, 224, 232));
        value_text.setPosition(vizgfx::Vector2f(text_x + key_column_w, text_y - 1.0f));
        const size_t key_chars = static_cast<size_t>(key_column_w / 7.2f);
        const size_t value_chars = max_chars > key_chars ? max_chars - key_chars : max_chars;
        const std::string shown = clamp_text_end(value, value_chars);
        value_text.setString(vizgfx::String::fromUtf8(shown.begin(), shown.end()));
        if (visible_row(20.0f)) window->draw(value_text);
        text_y += 20.0f;
    };

    draw_text("[" + display_tag + "]  TIP INSPECTOR", 18, accent, true);
    draw_text(parsed.path.empty() ? std::string("tip") : parsed.path,
              14, vizgfx::Color(245, 245, 248), true);
    if (!parsed.extra.empty()) {
        draw_text(parsed.extra, 12, vizgfx::Color(164, 174, 190));
    }

    draw_section("INFO");
    draw_value("type", parsed.layer_type.empty() ? parsed.headline : parsed.layer_type);
    draw_value("model", parsed.model.empty() ? std::string("visualizer") : parsed.model);
    draw_value("source", display_tag == "?" ? std::string("synthetic") : std::string("model"));

    if (selected_frame && !selected_frame->tensor_info.empty()) {
        draw_section("STATISTIQUES DU TENSEUR");
        std::istringstream info(selected_frame->tensor_info);
        for (std::string line; std::getline(info, line);)
            draw_text(line, 11, vizgfx::Color(220,224,232));
    }
    draw_section("STATISTIQUES DES PIXELS AFFICHES");
    draw_value("shape", std::to_string(img->w) + " x " + std::to_string(img->h) +
        " x " + std::to_string(img->channels));
    draw_value("bytes", std::to_string(img->pixels.size()));
    draw_value("range", std::to_string(pixel_min) + " .. " + std::to_string(pixel_max));
    draw_value("mean / sd", format_decimal(pixel_mean, 4) + " / " +
        format_decimal(pixel_stddev, 4));
    draw_value("non-zero", format_decimal(nonzero_ratio, 3) + "%");

    if (heatmap_mode_ && selected_frame) {
        draw_section("LEGENDE HEATMAP");
        if (selected_frame->heatmap_kind == 1) {
            draw_text("Bleu: negatif | gris: zero | rouge: positif", 11, vizgfx::Color(220, 224, 232));
            draw_text("Moyenne des canaux; luminosite = energie", 11, vizgfx::Color(220, 224, 232));
            draw_text("Sombre: faible | clair: forte (normalisee)", 11, vizgfx::Color(220, 224, 232));
            const vizgfx::Color colors[] = {vizgfx::Color(70,120,210), vizgfx::Color(175,175,175), vizgfx::Color(220,65,65)};
            for (int i = 0; i < 3; ++i) {
                vizgfx::RectangleShape swatch(vizgfx::Vector2f(text_w / 3, 12));
                swatch.setPosition(vizgfx::Vector2f(text_x + i * text_w / 3, text_y));
                swatch.setFillColor(colors[i]);
                if (visible_row(16)) window->draw(swatch);
            }
            text_y += 18;
        } else if (selected_frame->channels == 1) {
            draw_text(std::string("Palette: ") + heatmap_palette_name(static_cast<int>(heatmap_palette_)), 11, vizgfx::Color(220,224,232));
            std::vector<uint8_t> levels(256);
            for (int i = 0; i < 256; ++i) levels[i] = static_cast<uint8_t>(i);
            const auto colors = colorize_gray_heatmap_rgb(levels, static_cast<int>(heatmap_palette_));
            for (int i = 0; i < 256; ++i) {
                vizgfx::RectangleShape swatch(vizgfx::Vector2f(text_w / 256 + 1, 12));
                swatch.setPosition(vizgfx::Vector2f(text_x + i * text_w / 256, text_y));
                swatch.setFillColor(vizgfx::Color(colors[i*3], colors[i*3+1], colors[i*3+2]));
                if (visible_row(16)) window->draw(swatch);
            }
            text_y += 18;
            draw_text("0: faible <--- intensite affichee ---> 255: forte", 11, vizgfx::Color(220,224,232));
        } else {
            draw_text("Image RGB: couleurs des canaux, sans palette", 11, vizgfx::Color(220,224,232));
        }
    }
    draw_section("TRAINING (GLOBAL ACTUEL)");
    draw_value("epoch", std::to_string(current_epoch) + " / " +
        std::to_string(std::max(1, current_total_epochs)));
    draw_value("batch", std::to_string(current_batch) + " / " +
        std::to_string(std::max(1, current_total_batches)));
    draw_value("loss / avg", format_decimal(current_loss, 8) + " / " +
        format_decimal(current_avg_loss, 8));
    const std::string recon_name = canonical_recon_loss_name(current_recon_loss_type);
    draw_value(recon_name, format_decimal(current_mse, 8));
    draw_value("recon norm", format_decimal(
        normalized_recon_loss(current_mse, current_recon_loss_type), 8));
    draw_value("lr", format_decimal(current_lr, 10));
    draw_value("time / grad", format_decimal(current_timestep, 6) + " / " +
        format_decimal(current_grad_norm, 6));
    draw_value("perf", std::to_string(current_batch_time_ms) + " ms / " +
        std::to_string(current_memory_mb) + " MB RSS / " +
        std::to_string(current_allocator_memory_mb) + " MB allocator");
    draw_value("optimiseur", optimizerTypeName(static_cast<OptimizerType>(current_opt_type)));
    draw_value("optim step", std::to_string(current_opt_step));
    draw_text(show_feature_probe_ ? "F: features ON | Molette: defiler" : "F: features OFF | Molette: defiler", 11, vizgfx::Color(164,174,190));
    zoom_info_scroll_max_ = std::max(0.0f, text_y + zoom_info_scroll_ - (pad + sidebar_h) + 16.0f);
    zoom_info_scroll_ = std::min(zoom_info_scroll_, zoom_info_scroll_max_);
    if (picker_mouse_.x >= pad && picker_mouse_.x < pad+image_area_w &&
        picker_mouse_.y >= pad && picker_mouse_.y < pad+image_area_h &&
        picker_mouse_.x >= image_x && picker_mouse_.y >= image_y &&
        picker_mouse_.x < image_x + dw && picker_mouse_.y < image_y + dh && sc > 0.f) {
        const int px = std::clamp(static_cast<int>((picker_mouse_.x-image_x)/sc), 0, img->w-1);
        const int py = std::clamp(static_cast<int>((picker_mouse_.y-image_y)/sc), 0, img->h-1);
        // Highlight the exact preview cell sampled by the picker at either zoom scale.
        const float cell_x = image_x+px*sc, cell_y = image_y+py*sc;
        last_picker_rect_ = make_rect(cell_x,cell_y,sc,sc);
        vizgfx::RectangleShape scan_cell({sc,sc}); scan_cell.setPosition({cell_x,cell_y});
        scan_cell.setFillColor(vizgfx::Color(90,225,255,35));
        scan_cell.setOutlineColor(vizgfx::Color(5,12,20,230)); scan_cell.setOutlineThickness(3.f);
        window->draw(scan_cell);
        scan_cell.setOutlineColor(vizgfx::Color(100,235,255,255)); scan_cell.setOutlineThickness(1.f);
        window->draw(scan_cell);
        const float cx=cell_x+sc*0.5f, cy=cell_y+sc*0.5f;
        vizgfx::RectangleShape horizontal({18.f,1.f}); horizontal.setPosition({cx-9.f,cy});
        horizontal.setFillColor(vizgfx::Color(100,235,255)); window->draw(horizontal);
        vizgfx::RectangleShape vertical({1.f,18.f}); vertical.setPosition({cx,cy-9.f});
        vertical.setFillColor(vizgfx::Color(100,235,255)); window->draw(vertical);
        std::ostringstream value;
        value << "(" << px << ", " << py << ")" << std::setprecision(9);
        int lines = 1;
        const size_t tc = selected_frame ? static_cast<size_t>(std::max(0,selected_frame->tensor_sample_channels)) : 0;
        const size_t base = (static_cast<size_t>(py)*img->w+px)*tc;
        if (selected_frame && tc && base+tc <= selected_frame->tensor_values.size()) {
            for (size_t c=0;c<tc;++c) {
                value << "\n[" << (base+c < selected_frame->tensor_indices.size() ? selected_frame->tensor_indices[base+c] : base+c)
                      << "] = " << selected_frame->tensor_values[base+c]; ++lines;
            }
        } else {
            value << "\nPixel affiche (uint8)"; ++lines;
            const size_t base_pixel = (static_cast<size_t>(py)*img->w+px)*img->channels;
            for (int c=0;c<img->channels && base_pixel+c<img->pixels.size();++c) value << " " << static_cast<int>(img->pixels[base_pixel+c]);
        }
        if (show_feature_probe_) {
            value << "\nFeatures [F]"; ++lines;
            const auto probe = selected_frame && label.find("/vec")==std::string::npos ? FeatureProbe::analyze(selected_frame->tensor_values,
                img->w,img->h,selected_frame->tensor_sample_channels,px,py) : FeatureProbe::Result{};
            if (!probe.cells.empty()) {
                // Merge adjacent cells into runs to keep draw calls bounded by rows/sections.
                for (size_t i=0;i<probe.cells.size();) {
                    const size_t first=probe.cells[i];size_t last=first;
                    while(++i<probe.cells.size() && probe.cells[i]==last+1 && probe.cells[i]/img->w==first/img->w)last=probe.cells[i];
                    vizgfx::RectangleShape glow({float(last-first+1)*sc,sc});
                    glow.setPosition({image_x+float(first%img->w)*sc,image_y+float(first/img->w)*sc});
                    glow.setFillColor(vizgfx::Color(255,235,100,85));window->draw(glow);
                }
                value << "\nSection similaire : " << probe.cells.size() << " cellules"; ++lines;
                value << "\nCosinus >= 0.85 (activations)"; ++lines;
                for (const auto& channel:probe.channels) {
                    value << "\nCanal " << channel.first << " : " << channel.second; ++lines;
                }
            } else { value << "\nFeatures spatiales indisponibles"; ++lines; }
            const bool dataset=focus_target_==FocusTarget::Dataset;
            const bool output=focus_target_==FocusTarget::Output ||
                (focus_target_==FocusTarget::Generated && img->prompt==output_image_.prompt && img->w==output_image_.w && img->h==output_image_.h);
            const bool aligned=(dataset && (detection_target_==DetectionTarget::Dataset || detection_target_==DetectionTarget::Both)) ||
                (output && (detection_target_==DetectionTarget::Output || detection_target_==DetectionTarget::Both));
            size_t hits=0;
            if(aligned) for(const auto& d:detections_) {
                if(!std::isfinite(d.score) || d.score<detection_score_threshold_)continue;
                const float nx=(px+.5f)/img->w,ny=(py+.5f)/img->h;
                if(nx<d.x1 || nx>d.x2 || ny<d.y1 || ny>d.y2)continue;
                vizgfx::RectangleShape glow({(d.x2-d.x1)*dw,(d.y2-d.y1)*dh});
                glow.setPosition({image_x+d.x1*dw,image_y+d.y1*dh});
                glow.setFillColor(vizgfx::Color(100,255,180,65));
                glow.setOutlineColor(vizgfx::Color(150,255,200));glow.setOutlineThickness(2.f);window->draw(glow);
                if(hits++<4){value << "\nDetecte : " << (d.label.empty()?"classe "+std::to_string(d.class_id):d.label)
                    << " " << std::fixed << std::setprecision(1) << d.score*100.f << "%"; ++lines;}
            }
            if(hits>4){value << "\n+ " << hits-4 << " detections"; ++lines;}
        }
        last_picker_text_ = value.str();
        const float bw=show_feature_probe_?360.f:270.f, bh=static_cast<float>(lines*17+14);
        const float bx=std::clamp(picker_mouse_.x+16.f,0.f,std::max(0.f,static_cast<float>(window_width)-bw));
        const float by=std::clamp(picker_mouse_.y+16.f,0.f,std::max(0.f,static_cast<float>(window_height)-bh));
        vizgfx::RectangleShape bubble({bw,bh}); bubble.setPosition({bx,by});
        bubble.setFillColor(vizgfx::Color(15,20,28,245)); bubble.setOutlineColor(accent); bubble.setOutlineThickness(1.f); window->draw(bubble);
        std::istringstream rows(value.str());
        std::string row;
        float row_y = by+5.f;
        while (std::getline(rows,row)) {
            vizgfx::Text text(font); text.setCharacterSize(13); text.setPosition({bx+7.f,row_y});
            text.setFillColor(vizgfx::Color::White); text.setString(row); window->draw(text);
            row_y += 17.f;
        }
    }

}

void Visualizer::maybeLoadArchitecture() {
    if (architecture_loaded) return;
    if (architecture_path.empty()) return;

    const uint64_t now_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());

    // Éviter de spammer le FS si le fichier n'existe pas encore.
    if (arch_last_check_ms != 0 && (now_ms - arch_last_check_ms) < 1000) {
        return;
    }
    arch_last_check_ms = now_ms;

    std::ifstream f(architecture_path);
    if (!f.is_open()) {
        return;
    }

    try {
        json j;
        f >> j;
        if (!j.contains("layers") || !j["layers"].is_array()) {
            return;
        }

        std::unordered_set<std::string> inputs;
        arch_layer_names.clear();
        arch_tensor_outputs.clear();
        arch_tensor_sinks.clear();

        for (const auto& layer : j["layers"]) {
            if (layer.contains("name") && layer["name"].is_string()) {
                arch_layer_names.insert(layer["name"].get<std::string>());
            }
            if (layer.contains("output") && layer["output"].is_string()) {
                arch_tensor_outputs.insert(layer["output"].get<std::string>());
            }
            if (layer.contains("inputs") && layer["inputs"].is_array()) {
                for (const auto& in : layer["inputs"]) {
                    if (in.is_string()) inputs.insert(in.get<std::string>());
                }
            }
        }

        // Sinks = outputs jamais réutilisés comme input
        for (const auto& out : arch_tensor_outputs) {
            if (inputs.find(out) == inputs.end()) {
                arch_tensor_sinks.insert(out);
            }
        }

        architecture_loaded = !arch_layer_names.empty();
        if (architecture_loaded) {
            std::cerr << "✓ Visualizer: architecture chargée: " << arch_layer_names.size()
                      << " layers (" << arch_tensor_sinks.size() << " sinks)" << std::endl;
        }
    } catch (...) {
        // Best-effort: on retentera plus tard.
    }
}

void Visualizer::addGeneratedImage(const std::vector<uint8_t>& image, const std::string& prompt) {
    // Legacy wrapper: on infère un carré grayscale.
    int img_size = static_cast<int>(std::sqrt(static_cast<double>(image.size())));
    addGeneratedImage(image, img_size, img_size, 1, prompt);
}

void Visualizer::addGeneratedImage(const std::vector<uint8_t>& image, int w, int h, int channels, const std::string& prompt, bool validation) {
    if (!enabled) return;

    int iw = w;
    int ih = h;
    int ic = channels;

    if (ic != 1 && ic != 3 && ic != 4) {
        // Best-effort: tenter de déduire les canaux si possible.
        ic = 0;
    }
    if (iw <= 0 || ih <= 0 || ic == 0) {
        // Best-effort: inférer un carré (grayscale ou RGB selon la taille).
        const size_t n = image.size();
        const int s1 = static_cast<int>(std::sqrt(static_cast<double>(n)));
        const int s3 = static_cast<int>(std::sqrt(static_cast<double>(n / 3)));
        if ((size_t)(s3 * s3 * 3) == n) {
            iw = s3; ih = s3; ic = 3;
        } else if ((size_t)(s1 * s1) == n) {
            iw = s1; ih = s1; ic = 1;
        }
    }

    if (iw <= 0 || ih <= 0 || (ic != 1 && ic != 3 && ic != 4)) {
        return;
    }
    const size_t expected = static_cast<size_t>(iw) * static_cast<size_t>(ih) * static_cast<size_t>(ic);
    if (image.size() < expected) {
        return;
    }

    ImageData img_data;
    img_data.pixels = image;
    img_data.prompt = prompt;
    img_data.w = iw;
    img_data.h = ih;
    img_data.channels = ic;
    img_data.display_size = 200;
    createImageTexture(img_data, iw, ih, ic, 200);

    if (validation) {
        generated_images.insert(generated_images.begin() + validation_image_count_, std::move(img_data));
        ++validation_image_count_;
    } else {
        generated_images.push_back(std::move(img_data));
    }

    // Validation originals/results belong only to Generated; retain the completed training output.
    if (validation) return;

    output_image_.pixels = image;
    output_image_.prompt = prompt;
    output_image_.w = iw;
    output_image_.h = ih;
    output_image_.channels = ic;
    output_image_.display_size = 200;
    createImageTexture(output_image_, iw, ih, ic, 200);

    // Conserver une vignette 120px pour Blocks/Layers.
    output_thumb_.pixels = image;
    output_thumb_.prompt = prompt;
    output_thumb_.w = iw;
    output_thumb_.h = ih;
    output_thumb_.channels = ic;
    output_thumb_.display_size = 120;
    createImageTexture(output_thumb_, iw, ih, ic, 120);
    has_output_thumb_ = true;
    output_thumb_label_ = std::string("mimir/output | ") + prompt;

    const size_t max_images = static_cast<size_t>(std::max(1, image_grid_cols * image_grid_rows));
    if (generated_images.size() - validation_image_count_ > max_images) {
        generated_images.erase(generated_images.begin() + validation_image_count_);
    }
}

void Visualizer::beginValidationImages() {
    generated_images.erase(generated_images.begin(), generated_images.begin() + validation_image_count_);
    validation_image_count_ = 0;
    focus_generated_index_ = 0;
    last_generated_rects_.clear();
    last_generated_indices_.clear();
    panel_scroll_y_[static_cast<size_t>(PanelId::Generated)] = 0.f;
}

void Visualizer::setDatasetImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label, bool bbox_supervision, const DatasetBBoxOverlay::Boxes& boxes) {
    if (!enabled) return;
    if (w <= 0 || h <= 0) return;
    if (channels != 1 && channels != 3 && channels != 4) return;

    dataset_image.pixels = pixels;
    dataset_image.prompt = label;
    dataset_label = label;
    dataset_bbox_supervision_ = bbox_supervision;
    dataset_boxes_ = boxes;
    dataset_image.w = w;
    dataset_image.h = h;
    dataset_image.channels = channels;
    dataset_image.display_size = 200;
    createImageTexture(dataset_image, w, h, channels, 200);
    has_dataset_image = true;
}

void Visualizer::setDatasetText(const std::string& raw_text, const std::string& tags, const std::string& tokenized, const std::string& encoded) {
    if (!enabled) return;
    const bool changed = dataset_text_raw != raw_text || dataset_text_tags != tags ||
                         dataset_text_tokens != tokenized || dataset_text_encoded != encoded;
    dataset_text_raw = raw_text;
    dataset_text_tags = tags;
    dataset_text_tokens = tokenized;
    dataset_text_encoded = encoded;
    has_dataset_text = (!dataset_text_raw.empty() || !dataset_text_tags.empty() || !dataset_text_tokens.empty() || !dataset_text_encoded.empty());
    if (changed) dataset_text_scroll_y_.fill(0.0f);
}

void Visualizer::setProjectionImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label) {
    if (!enabled) return;
    if (w <= 0 || h <= 0) return;
    if (channels != 1 && channels != 3 && channels != 4) return;

    projection_image.pixels = pixels;
    projection_image.prompt = label;
    projection_label = label;
    projection_image.w = w;
    projection_image.h = h;
    projection_image.channels = channels;
    projection_image.display_size = 200;
    createImageTexture(projection_image, w, h, channels, 200);
    has_projection_image = true;

    projection_thumb_.pixels = pixels;
    projection_thumb_.prompt = label;
    projection_thumb_.w = w;
    projection_thumb_.h = h;
    projection_thumb_.channels = channels;
    projection_thumb_.display_size = 120;
    createImageTexture(projection_thumb_, w, h, channels, 120);
    has_projection_thumb_ = true;
}

void Visualizer::setUnderstandingImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label) {
    if (!enabled) return;
    if (w <= 0 || h <= 0) return;
    if (channels != 1 && channels != 3 && channels != 4) return;

    understanding_image.pixels = pixels;
    understanding_image.prompt = label;
    understanding_label = label;
    understanding_image.w = w;
    understanding_image.h = h;
    understanding_image.channels = channels;
    understanding_image.display_size = 200;
    createImageTexture(understanding_image, w, h, channels, 200);
    has_understanding_image = true;
}

void Visualizer::setDetections(const std::vector<Detection>& detections, DetectionTarget target) {
    detections_ = detections;
    detection_target_ = target;
}

void Visualizer::setDetectionOverlayEnabled(bool enabled) {
    show_detection_overlay_ = enabled;
}

void Visualizer::setDetectionScoreThreshold(float threshold) {
    detection_score_threshold_ = std::clamp(threshold, 0.0f, 1.0f);
}

void Visualizer::setLayerBlockImages(const std::vector<BlockFrame>& frames) {
    if (!enabled) return;
    if (&frames != &last_block_frames_) last_block_frames_ = frames;

    layer_block_images.clear();
    layer_block_labels.clear();
    has_layer_blocks = false;

    if (frames.empty()) {
        has_projection_image = has_projection_thumb_ = has_understanding_image = false;
        return;
    }
    const BlockFrame* embedding = nullptr;
    const BlockFrame* context = nullptr;
    for (const auto& frame : frames) {
        const std::string name = lowercase_ascii(frame.label);
        if (name.find("token_embedding") != std::string::npos || name.find("conditioning/tokens") != std::string::npos) embedding = &frame;
        if (name.find("lumen/text/") != std::string::npos || name.find("conditioning/text") != std::string::npos) context = &frame;
    }
    if (embedding) setProjectionImage(embedding->pixels, embedding->w, embedding->h, embedding->channels, embedding->label);
    if (context) setUnderstandingImage(context->pixels, context->w, context->h, context->channels, context->label);

    // L'entraînement publie généralement la sortie du modèle comme viz tap et non
    // via addGeneratedImage(). Promouvoir le dernier tap [ST] image-compatible
    // vers le panneau Sortie, sans utiliser les vecteurs de packing finaux.
    const bool explicit_lumen_output = std::any_of(frames.begin(), frames.end(), [](const BlockFrame& frame) {
        return Mimir::Viz::isLumenModelOutput(frame.label);
    });
    for (auto it = frames.rbegin(); it != frames.rend(); ++it) {
        const BlockFrame& frame = *it;
        if (explicit_lumen_output && !Mimir::Viz::isLumenModelOutput(frame.label)) continue;
        if (frame.w <= 0 || frame.h <= 0 || frame.pixels.empty()) continue;
        const ParsedVizLabel parsed = parse_viz_label(frame.label);
        if (display_tip_tag(parsed) != "ST") continue;

        const std::string source = lowercase_ascii(
            parsed.path + " " + parsed.layer_path + " " + parsed.extra);
        if (source.find("out_concat") != std::string::npos ||
            source.find("out_pack") != std::string::npos ||
            source.find("/vec") != std::string::npos) {
            continue;
        }

        const std::vector<uint8_t>& output_pixels = frame.pixels_real.empty()
            ? frame.pixels : frame.pixels_real;
        const int channels = infer_channels_from_buffer_size(
            output_pixels, frame.w, frame.h, frame.channels);
        if (channels != 1 && channels != 3 && channels != 4) continue;

        output_image_.pixels = output_pixels;
        output_image_.prompt = frame.label;
        output_image_.w = frame.w;
        output_image_.h = frame.h;
        output_image_.channels = channels;
        output_image_.display_size = 200;
        createImageTexture(output_image_, frame.w, frame.h, channels, 200);

        output_thumb_.pixels = output_pixels;
        output_thumb_.prompt = frame.label;
        output_thumb_.w = frame.w;
        output_thumb_.h = frame.h;
        output_thumb_.channels = channels;
        output_thumb_.display_size = 120;
        createImageTexture(output_thumb_, frame.w, frame.h, channels, 120);
        has_output_thumb_ = true;
        output_thumb_label_ = frame.label;
        break;
    }

    layer_block_images.reserve(frames.size());
    layer_block_labels.reserve(frames.size());

    for (const auto& f : frames) {
        if (f.w <= 0 || f.h <= 0) continue;
        if (f.pixels.empty()) continue;

        // Inference robuste des canaux pour éviter les textures décalées
        // quand les métadonnées (channels) ne collent pas au buffer.
        const int heat_channels = infer_channels_from_buffer_size(f.pixels, f.w, f.h, f.channels);
        const int real_channels = f.pixels_real.empty()
            ? 0
            : infer_channels_from_buffer_size(f.pixels_real, f.w, f.h, 1);
        if (heat_channels != 1 && heat_channels != 3 && heat_channels != 4) continue;

        const auto p = parse_viz_label(f.label);

        // PS (user): ne pas afficher les blocs d'activation.
        const bool exhaustive_model_layer = !p.model.empty() && p.tag == "L" &&
                                            f.label.find("/blocks/") != std::string::npos;
        if (hidden_block_tip(p, hide_activation_blocks, hide_normalisation_blocks)) continue;

        // PS (user): ne pas afficher la validation dans Blocks/Layers.
        // Best-effort: tag heuristique + substring explicite.
        if (p.tag == "VAL" || f.label.find("/validation") != std::string::npos || f.label.find("/val/") != std::string::npos) {
            continue;
        }

        // Le vecteur packé final (ex: recon||mu||logvar) n'est pas un tip image utile.
        // L'afficher comme pseudo-image carrée produit un bruit trompeur dans Outputs.
        if (!exhaustive_model_layer && f.label.find("/vec") != std::string::npos &&
            (p.layer_path.find("out_concat") != std::string::npos ||
             p.layer_path.find("out_pack") != std::string::npos)) {
            continue;
        }

        // Les concat de packing final (ex: recon||z||logvar) ne sont pas des images
        // stables à afficher: elles provoquent des artefacts visuels (répétitions/décalage).
        if (!exhaustive_model_layer && (p.layer_path.find("out_concat") != std::string::npos ||
            p.layer_path.find("out_pack") != std::string::npos)) {
            continue;
        }

        ImageData img;
        img.prompt = f.label;
        img.w = f.w;
        img.h = f.h;
        img.display_size = 120;

        if (!heatmap_mode_ && !f.pixels_real.empty()) {
            // Mode réel: respecter les canaux réellement présents dans pixels_real
            // (certaines sources peuvent fournir du RGB/RGBA et pas uniquement du 1 canal).
            if (real_channels == 1 || real_channels == 3 || real_channels == 4) {
                img.pixels       = f.pixels_real;
                img.channels     = real_channels;
                img.pixels_alt   = f.pixels;
                img.channels_alt = heat_channels;
            } else {
                // Buffer real invalide/incomplet: fallback heatmap pour éviter les artefacts.
                img.pixels       = f.pixels;
                img.channels     = heat_channels;
                img.pixels_alt   = f.pixels_real;
                img.channels_alt = (real_channels == 1 || real_channels == 3 || real_channels == 4) ? real_channels : 1;
            }
        } else {
            // Mode heatmap (par défaut) ou pas de version naturelle
            img.pixels      = f.pixels;
            img.channels    = heat_channels;
            img.pixels_alt  = f.pixels_real;
            img.channels_alt = (real_channels == 1 || real_channels == 3 || real_channels == 4) ? real_channels : 1;

            // En mode heatmap, éviter l'aspect noir/blanc sur les tips 1 canal:
            // on colorise en RGB tout en conservant la version brute pour debug.
            if (heatmap_mode_ && img.channels == 1) {
                img.pixels_alt = img.pixels;
                img.channels_alt = 1;
                img.pixels = colorize_gray_heatmap_rgb(img.pixels, static_cast<int>(heatmap_palette_));
                img.channels = 3;
            }
        }

        // Même les tenseurs scalaires utilisent un buffer RGB en mode réel.
        // Répéter un canal ne prétend pas inventer une couleur absente du tenseur.
        if (!heatmap_mode_ && img.channels == 1) {
            std::vector<uint8_t> rgb(img.pixels.size() * 3);
            for (size_t i = 0; i < img.pixels.size(); ++i)
                rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = img.pixels[i];
            img.pixels = std::move(rgb);
            img.channels = 3;
        }
        createLayerBlockTexture(img, f.label);
        layer_block_images.push_back(std::move(img));
        layer_block_labels.push_back(f.label);
    }

    has_layer_blocks = !layer_block_images.empty();
}

void Visualizer::rebuildLayerBlockTextures() {
    for (size_t i = 0; i < layer_block_images.size(); ++i) {
        const std::string label = (i < layer_block_labels.size()) ? layer_block_labels[i] : std::string();
        createLayerBlockTexture(layer_block_images[i], label);
    }
}

void Visualizer::createLayerBlockTexture(ImageData& img, const std::string& label) {
    if (img.w <= 0 || img.h <= 0 || img.display_size <= 0) return;
    if (img.channels != 1 && img.channels != 3 && img.channels != 4) return;
    if (img.pixels.empty()) return;

    const ParsedVizLabel p = parse_viz_label(label);
    const bool smooth_preview = smooth_layer_block_previews_ &&
                                is_layer_block_preview_smoothing_candidate(p, img.channels);

    if (!smooth_preview) {
        createImageTexture(img, img.w, img.h, img.channels, img.display_size);
        return;
    }

    ImageData tmp = img;
    tmp.pixels = smooth_preview_pixels(img.pixels, img.w, img.h, img.channels);
    createImageTexture(tmp, tmp.w, tmp.h, tmp.channels, tmp.display_size);
    img.texture = std::move(tmp.texture);
    img.sprite = tmp.sprite;
    img.sprite.setTexture(img.texture, true);
}

void Visualizer::updateMetrics(int epoch, int batch, float loss, float lr, float mse,
                              float kl, float wass, float ent, float mom, float spat, float temp,
                              float timestep,
                              int total_epochs, int total_batches, float avg_loss,
                              int batch_time_ms, size_t memory_mb,
                              double allocator_memory_mb, float bps, size_t params,
                              float grad_norm, float grad_max,
                              int opt_type, int opt_step,
                              float opt_beta1, float opt_beta2,
                              float opt_eps, float opt_weight_decay,
                              bool val_has, bool val_ok, int val_step, int val_items,
                              float val_recon, float val_kl, float val_align,
                              const std::string& recon_loss_type,
                              bool val_in_progress,
                              int val_done,
                              int val_total,
                              float kl_beta_effective,
                              const std::string& val_feedback,
                              const DetectionMetrics* detection_metrics) {
    current_epoch = epoch;
    current_total_epochs = total_epochs;
    current_batch = batch;
    current_total_batches = total_batches;
    current_loss = loss;
    current_avg_loss = avg_loss;
    current_lr = lr;
    current_mse = mse;
    current_recon_loss_type = recon_loss_type;
    current_kl = kl;
    current_wass = wass;
    current_ent = ent;
    current_mom = mom;
    current_spat = spat;
    current_temp = temp;
    current_timestep = timestep;
    current_batch_time_ms = batch_time_ms;
    current_memory_mb = memory_mb;
    current_allocator_memory_mb = allocator_memory_mb;
    current_bps = bps;
    current_params = params;
    current_grad_norm = grad_norm;
    current_grad_max = grad_max;
    current_opt_type = opt_type;
    current_opt_step = opt_step;
    current_opt_beta1 = opt_beta1;
    current_opt_beta2 = opt_beta2;
    current_opt_eps = opt_eps;
    current_opt_weight_decay = opt_weight_decay;

    current_kl_beta_effective = kl_beta_effective;

    current_val_has = val_has;
    current_val_ok = val_ok;
    current_val_in_progress = val_in_progress;
    current_val_step = val_step;
    current_val_items = val_items;
    current_val_done = std::max(0, val_done);
    current_val_total = std::max(0, val_total);
    current_val_recon = val_recon;
    current_val_kl = val_kl;
    current_val_align = val_align;
    current_val_feedback = val_feedback;
    current_detection_metrics_ = detection_metrics
        ? *detection_metrics : DetectionMetrics{};

    if (!has_loss_stats_) {
        has_loss_stats_ = true;
        loss_min_ = loss;
        loss_max_ = loss;
    } else {
        loss_min_ = std::min(loss_min_, loss);
        loss_max_ = std::max(loss_max_, loss);
    }
    
    // Créer un record complet avec toutes les métriques
    LossRecord record;
    record.step = full_loss_history.size();
    record.epoch = epoch;
    record.total_epochs = total_epochs;
    record.batch = batch;
    record.total_batches = total_batches;
    record.loss = loss;
    record.avg_loss = avg_loss;
    record.lr = lr;
    record.batch_time_ms = batch_time_ms;
    record.bps = bps;
    record.memory_mb = memory_mb;
    record.allocator_memory_mb = allocator_memory_mb;
    record.params = params;
    record.mse = mse;
    record.kl_divergence = kl;
    record.kl_beta_effective = kl_beta_effective;
    record.wasserstein = wass;
    record.entropy_diff = ent;
    record.moment_mismatch = mom;
    record.spatial_coherence = spat;
    record.temporal_consistency = temp;
    record.timestep = timestep;
    record.grad_norm = grad_norm;
    record.grad_max = grad_max;
    record.opt_type = opt_type;
    record.opt_step = opt_step;
    record.opt_beta1 = opt_beta1;
    record.opt_beta2 = opt_beta2;
    record.opt_eps = opt_eps;
    record.opt_weight_decay = opt_weight_decay;
    // Métriques de validation : renseignées à la fin de chaque validation.
    // val_recon = loss primaire (img-space MSE pour DDPM, recon loss pour VAE).
    // val_kl    = second indicateur (eps-space MSE pour DDPM, KL pour VAE).
    // val_* reste affiché après la validation. Ne sérialiser toutefois qu'une
    // seule ligne par step de validation, pas une copie à chaque tick train.
    const bool new_validation = val_has && !val_in_progress && val_step >= 0 &&
                                val_step != last_recorded_validation_step_;
    record.is_val      = new_validation;
    record.val_loss    = new_validation ? val_recon : 0.f;
    record.val_mse     = new_validation ? val_kl    : 0.f;
    record.val_step_id = new_validation ? val_step  : -1;
    record.val_feedback = new_validation ? val_feedback : std::string();
    if (new_validation) last_recorded_validation_step_ = val_step;
    full_loss_history.push_back(record);
    
    // Flush CSV throttle: eviter une ecriture disque a chaque metric tick.
    pending_loss_log_flush_ = true;
    const uint64_t now_ms = steady_now_ms();
    if (loss_log_enabled_ && !loss_log_file.empty() &&
        (last_loss_log_flush_ms_ == 0 || (now_ms - last_loss_log_flush_ms_) >= 1000)) {
        saveLossHistory(loss_log_file);
        last_loss_log_flush_ms_ = now_ms;
        pending_loss_log_flush_ = false;
    }
}

void Visualizer::setMetricsHistory(const std::vector<MetricsRecord>& records, size_t offset) {
    offset = std::min(offset, full_loss_history.size());
    full_loss_history.resize(offset); // Remove updateMetrics' provisional UI record.
    full_loss_history.insert(full_loss_history.end(), records.begin(), records.end());
    if (offset == 0) has_loss_stats_ = false;
    for (const auto& record : records) {
        if (!has_loss_stats_) {loss_min_ = loss_max_ = record.loss; has_loss_stats_ = true;}
        else {loss_min_ = std::min(loss_min_, record.loss); loss_max_ = std::max(loss_max_, record.loss);}
        if (record.is_val) last_recorded_validation_step_ = record.val_step_id;
    }
    loss_history.clear();
    const size_t visible = std::min(full_loss_history.size(), static_cast<size_t>(graph_history_size));
    for (size_t i = full_loss_history.size() - visible; i < full_loss_history.size(); ++i)
        loss_history.push_back(full_loss_history[i].loss);
}

void Visualizer::addLossPoint(float loss) {
    loss_history.push_back(loss);
    if (loss_history.size() > static_cast<size_t>(graph_history_size)) {
        loss_history.pop_front();
    }
    
    // NE PAS ajouter à full_loss_history ici - c'est déjà fait par updateMetrics()
    // Cette fonction ne fait que mettre à jour le graphique visuel
}

void Visualizer::clearImages() {
    generated_images.clear();
    validation_image_count_ = 0;
    detections_.clear();
    has_output_thumb_ = false;
    output_image_.pixels.clear();
    output_thumb_.pixels.clear();
    output_thumb_label_.clear();
}

void Visualizer::setEnabled(bool en) {
    enabled = en;
}

bool Visualizer::isEnabled() const {
    return enabled;
}

// Méthodes privées de rendu

void Visualizer::renderBackground() {
    // Grille de fond subtile
    vizgfx::RectangleShape line(vizgfx::Vector2f(1, window_height));
    line.setFillColor(vizgfx::Color(40, 40, 45, 100));
    
    for (int x = 0; x < window_width; x += 50) {
        line.setPosition(vizgfx::Vector2f(x, 0));
        window->draw(line);
    }

    line.setSize(vizgfx::Vector2f(window_width, 1));
    for (int y = 0; y < window_height; y += 50) {
        line.setPosition(vizgfx::Vector2f(0, y));
        window->draw(line);
    }

    // Splash logo au lancement
    if (logo_loaded_) {
        const float t = logo_clock_.getElapsedTime().asSeconds();
        if (t >= 0.0f && t < logo_splash_seconds_) {
            const auto lb = logo_sprite_.getLocalBounds();
            const float iw = std::max(1.0f, lb.size.x);
            const float ih = std::max(1.0f, lb.size.y);
            const float desired = std::min<float>(
                std::min<float>(window_width, window_height) * 0.35f,
                320.0f
            );
            const float scale = desired / std::max(iw, ih);
            logo_sprite_.setScale(vizgfx::Vector2f(scale, scale));
            logo_sprite_.setOrigin(vizgfx::Vector2f(lb.position.x + iw * 0.5f, lb.position.y + ih * 0.5f));
            logo_sprite_.setPosition(vizgfx::Vector2f(window_width * 0.5f, window_height * 0.5f));
            window->draw(logo_sprite_);
        }
    }
}

void Visualizer::renderContextImages() {
    // Affiche (si disponibles) l'image du dataset + la projection associée.
    const auto area = panelContentRect(PanelId::Context);
    const int img_display_size = workbenchPreviewSize(area);
    const int margin = 16;
    const int start_x = static_cast<int>(area.position.x);
    const int label_h = 20;
    const int item_count = static_cast<int>(has_dataset_image) +
        static_cast<int>(has_projection_image) + static_cast<int>(has_understanding_image);
    const int cols = std::max(1, static_cast<int>(
        (area.size.x + margin) / (img_display_size + margin)));
    const int rows = item_count > 0 ? (item_count + cols - 1) / cols : 0;
    setPanelContentHeight(PanelId::Context,
        static_cast<float>(rows * (img_display_size + label_h + margin)));
    const int start_y = static_cast<int>(area.position.y -
        panel_scroll_y_[static_cast<size_t>(PanelId::Context)]);

    // Clip au contenu du panneau (overflow: clip)
    const vizgfx::View old_view = window->getView();
    struct ViewGuard {
        VizWindow* w;
        vizgfx::View v;
        ~ViewGuard() {
            if (w) w->setView(v);
        }
    } guard{window.get(), old_view};
    {
        const float ww = static_cast<float>(std::max(1, window_width));
        const float hh = static_cast<float>(std::max(1, window_height));

        const float left = std::clamp(area.position.x, 0.0f, ww);
        const float top = std::clamp(area.position.y, 0.0f, hh);
        const float right = std::clamp(area.position.x + area.size.x, 0.0f, ww);
        const float bottom = std::clamp(area.position.y + area.size.y, 0.0f, hh);
        const float w = std::max(0.0f, right - left);
        const float h = std::max(0.0f, bottom - top);

        if (w > 0.0f && h > 0.0f) {
            vizgfx::View v(make_rect(left, top, w, h));
            v.setViewport(make_rect(left / ww, top / hh, w / ww, h / hh));
            window->setView(v);
        }
    }

    auto toSfUtf8 = [](const std::string& s) -> vizgfx::String {
        return vizgfx::String::fromUtf8(s.begin(), s.end());
    };

    auto drawPanel = [&](int x, int y, ImageData& img, const std::string& label_text, vizgfx::Color outline) -> vizgfx::FloatRect {
        // Cadre
        vizgfx::RectangleShape frame(vizgfx::Vector2f(img_display_size + 4, img_display_size + 4));
        frame.setPosition(vizgfx::Vector2f(x - 2, y - 2));
        frame.setFillColor(vizgfx::Color::Transparent);
        frame.setOutlineColor(outline);
        frame.setOutlineThickness(2);
        window->draw(frame);

        const vizgfx::FloatRect frame_rect = make_rect(static_cast<float>(x - 2), static_cast<float>(y - 2), static_cast<float>(img_display_size + 4), static_cast<float>(img_display_size + 4));

        // Image
        position_sprite_centered_in_box(img.sprite, static_cast<float>(x), static_cast<float>(y), static_cast<float>(img_display_size));
        window->draw(img.sprite);
        if (&img == &dataset_image) {
            renderDetectionOverlay(img, static_cast<float>(x), static_cast<float>(y),
                                   static_cast<float>(img_display_size), DetectionTarget::Dataset);
            mimir::viz::drawDatasetBBoxBadge(*window, static_cast<float>(x),
                static_cast<float>(y), static_cast<float>(img_display_size), dataset_bbox_supervision_);
        }

        // Label bar
        vizgfx::RectangleShape label(vizgfx::Vector2f(img_display_size, label_h));
        label.setPosition(vizgfx::Vector2f(static_cast<float>(x), static_cast<float>(y + img_display_size + 5)));
        label.setFillColor(vizgfx::Color(50, 50, 60, 200));
        window->draw(label);

        if (font_loaded) {
            vizgfx::Text t(font);
            t.setFont(font);
            t.setCharacterSize(14);
            t.setFillColor(vizgfx::Color(230, 230, 235));
            t.setPosition(vizgfx::Vector2f(static_cast<float>(x + 6), static_cast<float>(y + img_display_size + 3)));
            t.setString(toSfUtf8(label_text));
            window->draw(t);
        }

        return frame_rect;
    };

    auto apply_arch_hint = [&](ParsedVizLabel& p) {
        if (!architecture_loaded) return;
        if (!p.path.empty()) {
            if (arch_tensor_sinks.find(p.path) != arch_tensor_sinks.end()) {
                p.tag = "OUT";
                p.headline = "Sortie";
                p.short_text = p.path;
            } else if (arch_layer_names.find(p.path) != arch_layer_names.end()) {
                p.tag = "L";
                p.headline = "Layer";
                // enlever le préfixe modèle
                if (!p.model.empty() && p.path.rfind(p.model + "/", 0) == 0) {
                    p.short_text = p.path.substr(p.model.size() + 1);
                } else {
                    p.short_text = p.path;
                }
            } else if (arch_tensor_outputs.find(p.path) != arch_tensor_outputs.end()) {
                p.tag = "T";
                p.headline = "Tensor";
                p.short_text = p.path;
            }
        }
    };

    int x = start_x;
    int y = start_y;
    auto advance = [&]() {
        x += img_display_size + margin;
        if (x + img_display_size > static_cast<int>(area.position.x + area.size.x)) {
            x = start_x;
            y += img_display_size + label_h + margin;
        }
    };

    if (has_dataset_image) {
        auto p = parse_viz_label(dataset_label.empty() ? std::string("dataset") : dataset_label);
        apply_arch_hint(p);
        p.tag = "DS";
        const std::string display_tag = display_tip_tag(p);
        last_dataset_rect_ = drawPanel(
            x, y, dataset_image, clamp_text_end(format_tip_display_label(p), 44),
            color_for_tag(display_tag));
        advance();
    }
    if (has_projection_image) {
        auto p = parse_viz_label(projection_label.empty() ? std::string("projection") : projection_label);
        apply_arch_hint(p);
        const std::string display_tag = display_tip_tag(p);
        last_projection_rect_ = drawPanel(
            x, y, projection_image, clamp_text_end(format_tip_display_label(p), 44),
            color_for_tag(display_tag));
        advance();
    }
    if (has_understanding_image) {
        auto p = parse_viz_label(understanding_label.empty() ? std::string("understanding") : understanding_label);
        apply_arch_hint(p);
        const std::string display_tag = display_tip_tag(p);
        last_understanding_rect_ = drawPanel(
            x, y, understanding_image, clamp_text_end(format_tip_display_label(p), 44),
            color_for_tag(display_tag));
    }

}

void Visualizer::renderOutputImage() {
    const auto area = panelContentRect(PanelId::Output);
    const int image_size = workbenchPreviewSize(area);
    const int label_height = 20;
    const int margin = 16;
    const float content_height = has_output_thumb_
        ? static_cast<float>(image_size + label_height + margin) : 0.0f;
    setPanelContentHeight(PanelId::Output, content_height);
    if (!has_output_thumb_ || output_thumb_.pixels.empty()) return;

    const float x = area.position.x;
    const float y = area.position.y - panel_scroll_y_[static_cast<size_t>(PanelId::Output)];
    const vizgfx::View old_view = window->getView();
    {
        const float window_w = static_cast<float>(std::max(1, window_width));
        const float window_h = static_cast<float>(std::max(1, window_height));
        const float left = std::clamp(area.position.x, 0.0f, window_w);
        const float top = std::clamp(area.position.y, 0.0f, window_h);
        const float right = std::clamp(area.position.x + area.size.x, 0.0f, window_w);
        const float bottom = std::clamp(area.position.y + area.size.y, 0.0f, window_h);
        const float width = std::max(0.0f, right - left);
        const float height = std::max(0.0f, bottom - top);
        if (width > 0.0f && height > 0.0f) {
            vizgfx::View view(make_rect(left, top, width, height));
            view.setViewport(make_rect(
                left / window_w, top / window_h, width / window_w, height / window_h));
            window->setView(view);
        }
    }

    vizgfx::RectangleShape frame(vizgfx::Vector2f(image_size + 4.0f, image_size + 4.0f));
    frame.setPosition(vizgfx::Vector2f(x - 2.0f, y - 2.0f));
    frame.setFillColor(vizgfx::Color::Transparent);
    frame.setOutlineColor(panelAccent(PanelId::Output));
    frame.setOutlineThickness(2.0f);
    window->draw(frame);
    last_output_rect_ = make_rect(x - 2.0f, y - 2.0f, image_size + 4.0f, image_size + 4.0f);

    position_sprite_centered_in_box(output_image_.sprite, x, y, static_cast<float>(image_size));
    window->draw(output_image_.sprite);
    renderDetectionOverlay(output_image_, x, y, static_cast<float>(image_size), DetectionTarget::Output);

    vizgfx::RectangleShape label(vizgfx::Vector2f(static_cast<float>(image_size), static_cast<float>(label_height)));
    label.setPosition(vizgfx::Vector2f(x, y + image_size + 5.0f));
    label.setFillColor(vizgfx::Color(50, 50, 60, 200));
    window->draw(label);

    if (font_loaded) {
        const ParsedVizLabel parsed = parse_viz_label(
            output_thumb_label_.empty() ? std::string("mimir/output") : output_thumb_label_);
        const std::string text = clamp_text_end(format_tip_display_label(parsed), 28);
        vizgfx::Text title(font);
        title.setCharacterSize(14);
        title.setFillColor(vizgfx::Color(230, 230, 235));
        title.setPosition(vizgfx::Vector2f(x + 6.0f, y + image_size + 3.0f));
        title.setString(vizgfx::String::fromUtf8(text.begin(), text.end()));
        window->draw(title);
    }

    window->setView(old_view);
}

void Visualizer::renderDetectionOverlay(const ImageData& image, float x, float y, float box_size,
                                        DetectionTarget rendered_target) {
    if (!show_detection_overlay_ || detections_.empty() || image.w <= 0 || image.h <= 0) return;
    if (detection_target_ != DetectionTarget::Both && detection_target_ != rendered_target) return;

    const float scale = std::min(box_size / static_cast<float>(image.w),
                                 box_size / static_cast<float>(image.h));
    const float draw_w = static_cast<float>(image.w) * scale;
    const float draw_h = static_cast<float>(image.h) * scale;
    const float image_x = x + (box_size - draw_w) * 0.5f;
    const float image_y = y + (box_size - draw_h) * 0.5f;
    static const std::array<vizgfx::Color, 6> colors = {
        vizgfx::Color(72, 220, 145), vizgfx::Color(255, 190, 70),
        vizgfx::Color(90, 190, 255), vizgfx::Color(245, 105, 130),
        vizgfx::Color(190, 125, 255), vizgfx::Color(235, 235, 90)};

    for (const Detection& detection : detections_) {
        if (detection.score >= 0.0f && detection.score < detection_score_threshold_) continue;
        const size_t color_index = detection.class_id >= 0
            ? static_cast<size_t>(detection.class_id) % colors.size() : 0;
        const vizgfx::Color color = detection.score >= 0.0f
            ? detection_heatmap_color(detection.score) : colors[color_index];
        const float left = image_x + detection.x1 * draw_w;
        const float top = image_y + detection.y1 * draw_h;
        const float width = (detection.x2 - detection.x1) * draw_w;
        const float height = (detection.y2 - detection.y1) * draw_h;
        vizgfx::RectangleShape outline(vizgfx::Vector2f(width, height));
        outline.setPosition(vizgfx::Vector2f(left, top));
        outline.setFillColor(vizgfx::Color::Transparent);
        outline.setOutlineColor(color);
        outline.setOutlineThickness(2.0f);
        window->draw(outline);

        if (!font_loaded) continue;
        std::string text = detection.label;
        if (text.empty() && detection.class_id >= 0) text = "class " + std::to_string(detection.class_id);
        if (detection.score >= 0.0f) {
            if (!text.empty()) text += " ";
            text += std::to_string(static_cast<int>(std::lround(detection.score * 100.0f))) + "%";
        }
        if (text.empty()) continue;
        const float label_w = std::min(draw_w,
            std::max(54.0f, 8.0f * static_cast<float>(text.size()) + 14.0f));
        const float label_h = 22.0f;
        const float label_x = std::clamp(left, image_x, image_x + draw_w - label_w);
        const float label_y = std::max(image_y, top - label_h);
        vizgfx::RectangleShape label_bg(vizgfx::Vector2f(label_w, label_h));
        label_bg.setPosition(vizgfx::Vector2f(label_x, label_y));
        label_bg.setFillColor(vizgfx::Color(12, 14, 18, 238));
        label_bg.setOutlineColor(color);
        label_bg.setOutlineThickness(2.0f);
        window->draw(label_bg);
        vizgfx::Text label(font);
        label.setCharacterSize(13);
        label.setStyle(vizgfx::Text::Bold);
        label.setFillColor(vizgfx::Color(248, 248, 250));
        label.setPosition(vizgfx::Vector2f(label_x + 6.0f, label_y));
        const std::string clipped = clamp_text_end(
            text, static_cast<size_t>(std::max(1.0f, (label_w - 12.0f) / 8.0f)));
        label.setString(vizgfx::String::fromUtf8(clipped.begin(), clipped.end()));
        window->draw(label);
    }
}

void Visualizer::renderLayerBlocks() {
    const bool has_blocks = (has_layer_blocks && !layer_block_images.empty());
    const bool show_projection = has_projection_thumb_ && has_projection_image && !projection_thumb_.pixels.empty() &&
        !hidden_block_tip(parse_viz_label(projection_label), hide_activation_blocks, hide_normalisation_blocks);
    const bool show_output = has_output_thumb_ && !output_thumb_.pixels.empty() &&
        !hidden_block_tip(parse_viz_label(output_thumb_label_), hide_activation_blocks, hide_normalisation_blocks);

    const int blocks_count = has_blocks ? static_cast<int>(layer_block_images.size()) : 0;
    if (!show_projection && !show_output && blocks_count <= 0) return;

    last_block_rects_.clear();

    const int thumb = 120;
    const int margin = 14;
    const auto area = panelContentRect(PanelId::Blocks);
    const int start_x = static_cast<int>(area.position.x);
    const int start_y = static_cast<int>(area.position.y);

    const int label_h = 18;
    const int section_h = 22;
    const int section_gap = 8;
    const int max_cols = std::max(1, static_cast<int>(area.size.x) / (thumb + margin));
    const int cell_h = thumb + label_h + margin;

    // Focus rectangles doivent correspondre aux vrais blocks (index = layer_block_images idx).
    last_block_rects_.assign(static_cast<size_t>(std::max(0, blocks_count)), make_rect(0.f, 0.f, 0.f, 0.f));

    // Clip au contenu du panneau (évite de dessiner en dehors de la zone visible pendant le scroll)
    const vizgfx::View old_view = window->getView();
    {
        const float ww = static_cast<float>(std::max(1, window_width));
        const float hh = static_cast<float>(std::max(1, window_height));

        // Clamp à la partie visible dans la fenêtre (le panneau peut être resizé hors écran)
        const float left = std::clamp(area.position.x, 0.0f, ww);
        const float top = std::clamp(area.position.y, 0.0f, hh);
        const float right = std::clamp(area.position.x + area.size.x, 0.0f, ww);
        const float bottom = std::clamp(area.position.y + area.size.y, 0.0f, hh);
        const float w = std::max(0.0f, right - left);
        const float h = std::max(0.0f, bottom - top);

        if (w > 0.0f && h > 0.0f) {
            vizgfx::View v(make_rect(left, top, w, h));
            v.setViewport(make_rect(left / ww, top / hh, w / ww, h / hh));
            window->setView(v);
        }
    }

    auto apply_arch_hint = [&](ParsedVizLabel& p) {
        if (!architecture_loaded) return;
        if (!p.path.empty()) {
            if (arch_tensor_sinks.find(p.path) != arch_tensor_sinks.end()) {
                p.tag = "OUT";
                p.headline = "Sortie";
                p.short_text = p.path;
            } else if (arch_layer_names.find(p.path) != arch_layer_names.end()) {
                p.tag = "L";
                p.headline = "Layer";
                // Enlever le préfixe "model/" si présent (plus lisible en vignette)
                if (!p.model.empty() && p.path.rfind(p.model + "/", 0) == 0) {
                    p.short_text = p.path.substr(p.model.size() + 1);
                } else {
                    p.short_text = p.path;
                }
            } else if (arch_tensor_outputs.find(p.path) != arch_tensor_outputs.end()) {
                p.tag = "T";
                p.headline = "Tensor";
                p.short_text = p.path;
            }
        }
    };

    struct BlockPanelEntry {
        bool is_header = false;
        bool is_extra = false;
        int block_index = -1;
        ImageData* img = nullptr;
        ParsedVizLabel parsed;
        std::string section;
        std::string text_override;
        std::string tag_override;
    };

    auto starts_with = [](const std::string& s, const std::string& prefix) {
        return s.rfind(prefix, 0) == 0;
    };

    auto section_color = [&](const std::string& section) -> vizgfx::Color {
        if (section == "Inputs") return vizgfx::Color(110, 145, 205, 210);
        if (section == "Text / Cond") return vizgfx::Color(122, 188, 142, 210);
        if (section == "UNet") return vizgfx::Color(104, 164, 216, 210);
        if (section == "VAE") return vizgfx::Color(214, 126, 166, 210);
        if (section == "VAE / Encoder") return vizgfx::Color(112, 174, 212, 210);
        if (section == "VAE / Latent") return vizgfx::Color(214, 106, 172, 210);
        if (section == "VAE / Decoder") return vizgfx::Color(233, 152, 92, 210);
        if (section == "Diffusion") return vizgfx::Color(146, 136, 222, 210);
        if (section == "Reconstruction") return vizgfx::Color(226, 162, 106, 210);
        if (section == "Validation") return vizgfx::Color(115, 194, 204, 210);
        if (section == "ConditioningEncoder") return vizgfx::Color(96, 168, 214, 210);
        if (section == "Down Blocks") return vizgfx::Color(88, 176, 198, 210);
        if (section == "Bottleneck") return vizgfx::Color(160, 118, 220, 210);
        if (section == "Backbone") return vizgfx::Color(132, 132, 218, 210);
        if (section == "Up Blocks") return vizgfx::Color(232, 170, 102, 210);
        if (section == "Latent") return vizgfx::Color(214, 106, 172, 210);
        if (section == "Decoder") return vizgfx::Color(233, 152, 92, 210);
        if (section == "Heads") return vizgfx::Color(206, 196, 110, 210);
        if (section == "Outputs") return vizgfx::Color(188, 188, 120, 210);
        return vizgfx::Color(100, 100, 112, 210);
    };

    auto section_for = [&](const ParsedVizLabel& p, bool is_extra) -> std::string {
        if (is_extra) return "Outputs";
        const auto lumen_section = Mimir::Viz::lumenTipSection(p.path + " " + p.layer_path + " " + p.extra);
        if (!lumen_section.empty()) return std::string(lumen_section);
        const std::string lp = !p.layer_path.empty() ? p.layer_path : p.short_text;
        if (lp.empty()) return "Blocks";

        const auto parts = split(lp, '/');
        const std::string first = parts.empty() ? std::string() : parts.front();
        const std::string last = parts.empty() ? std::string() : parts.back();

        bool has_latent_exact = false;
        bool has_latent_prefix = false;
        bool has_latent_substr = false;

        bool has_text_substr = false;
        bool has_text_exact = false;

        bool has_input_exact = false;
        bool has_input_prefix = false;

        bool has_encoder_exact_or_prefix = false;
        bool has_decoder_exact_or_prefix = false;
        bool has_down_prefix = false;
        bool has_up_prefix = false;

        bool has_bottleneck_exact = false;
        bool has_bottleneck_prefix = false;
        bool has_bottleneck_substr = false;

        bool has_backbone_exact = false;
        bool has_backbone_prefix = false;

        bool has_head_exact = false;
        bool has_head_prefix = false;
        bool has_head_substr = false;

        bool has_output_exact = false;
        bool has_output_prefix = false;
        bool has_output_substr = false;

        bool has_diff_exact = false;
        bool has_diff_prefix = false;
        bool has_diff_substr = false;

        bool has_unet_exact = false;
        bool has_unet_substr = false;
        bool has_vae_exact = false;
        bool has_vae_substr = false;
        bool has_diffusion_exact = false;
        bool has_diffusion_substr = false;
        bool has_recon_exact = false;
        bool has_recon_substr = false;

        for (const auto& part : parts) {
            if (part == "mu" || part == "logvar" || part == "latent" || part == "z") has_latent_exact = true;
            if (starts_with(part, "mu") || starts_with(part, "logvar") || starts_with(part, "latent") || starts_with(part, "reparam")) has_latent_prefix = true;
            if (part.find("latent") != std::string::npos || part.find("logvar") != std::string::npos || part.find("reparam") != std::string::npos) has_latent_substr = true;

            if (part == "tok_emb" || part == "meanpool" || part == "pool" || part == "add_pos" || part == "text_proj") has_text_exact = true;
            if (part.find("text") != std::string::npos || part.find("token") != std::string::npos || part.find("prompt") != std::string::npos ||
                part.find("context") != std::string::npos || part.find("embed") != std::string::npos || part.find("cond") != std::string::npos) {
                has_text_substr = true;
            }

            if (part == "input" || part == "in" || part == "raw_in" || part == "raw_z" || part == "in_vec" || part == "in_hwc" || part == "in_chw" || part == "init") has_input_exact = true;
            if (starts_with(part, "input") || starts_with(part, "raw_") || starts_with(part, "in_")) has_input_prefix = true;

            if (part == "encoder" || starts_with(part, "enc") || starts_with(part, "encoder")) has_encoder_exact_or_prefix = true;
            if (part == "decoder" || starts_with(part, "dec") || starts_with(part, "decoder")) has_decoder_exact_or_prefix = true;
            if (starts_with(part, "down")) has_down_prefix = true;
            if (starts_with(part, "up")) has_up_prefix = true;

            if (part == "mid" || part == "bottleneck" || part == "bridge") has_bottleneck_exact = true;
            if (starts_with(part, "mid") || starts_with(part, "bot") || starts_with(part, "bottleneck") || starts_with(part, "bridge")) has_bottleneck_prefix = true;
            if (part.find("bot_") != std::string::npos || part.find("mid_") != std::string::npos) has_bottleneck_substr = true;

            if (part == "unet" || part == "transformer" || part == "transformer_block" || part == "backbone" || part == "trunk" || part == "layers" || part == "blocks") has_backbone_exact = true;
            if (starts_with(part, "unet") || starts_with(part, "transformer") || starts_with(part, "block") || starts_with(part, "layer")) has_backbone_prefix = true;

            if (part == "head" || part == "lm_head" || part == "classifier" || part == "proj" || part == "out_proj" || part == "text_proj") has_head_exact = true;
            if (starts_with(part, "head") || starts_with(part, "proj") || starts_with(part, "final_")) has_head_prefix = true;
            if (part.find("_head") != std::string::npos || part.find("_proj") != std::string::npos) has_head_substr = true;

            if (part == "out" || part == "output" || part == "out_pack" || part == "out_concat" || part == "recon" ||
                part == "recon_chw" || part == "recon_to_hwc" || part == "out_to_hwc" || part == "final") {
                has_output_exact = true;
            }
            if (starts_with(part, "out") || starts_with(part, "output") || starts_with(part, "recon")) has_output_prefix = true;
            if (part.find("recon") != std::string::npos || part.find("to_hwc") != std::string::npos || part.find("out_concat") != std::string::npos || part.find("out_pack") != std::string::npos) {
                has_output_substr = true;
            }

            if (part == "diff" || part == "err" || part == "resdiff" || part == "resdiff_abs") {
                has_diff_exact = true;
            }
            if (starts_with(part, "diff") || starts_with(part, "err") || starts_with(part, "resdiff")) {
                has_diff_prefix = true;
            }
            if (part.find("diff") != std::string::npos || part.find("err") != std::string::npos) {
                has_diff_substr = true;
            }

            if (part == "unet" || part == "unet2d") has_unet_exact = true;
            if (part.find("unet") != std::string::npos) has_unet_substr = true;

            if (part == "vae" || part == "vae_conv" || part == "vae_text") has_vae_exact = true;
            if (part.find("vae") != std::string::npos || part.find("decoder") != std::string::npos || part.find("encoder") != std::string::npos) {
                has_vae_substr = true;
            }

            if (part == "diffusion" || part == "ddpm") has_diffusion_exact = true;
            if (part.find("diffusion") != std::string::npos || part.find("ddpm") != std::string::npos || part.find("denoise") != std::string::npos || part.find("noise") != std::string::npos) {
                has_diffusion_substr = true;
            }

            if (part == "recon" || part == "reconstruction") has_recon_exact = true;
            if (part.find("recon") != std::string::npos || part.find("resdiff") != std::string::npos) {
                has_recon_substr = true;
            }
        }

        const bool is_latent = p.tag == "LAT" || has_latent_exact || has_latent_prefix || has_latent_substr;
        const bool is_text_cond = has_text_substr || has_text_exact;
        const bool is_input = has_input_exact || has_input_prefix;
        const bool is_encoder = first == "enc" || has_encoder_exact_or_prefix;
        const bool is_decoder = first == "dec" || has_decoder_exact_or_prefix;
        const bool is_down = has_down_prefix;
        const bool is_up = has_up_prefix;
        const bool is_bottleneck = has_bottleneck_exact || has_bottleneck_prefix || has_bottleneck_substr;
        const bool is_backbone = has_backbone_exact || has_backbone_prefix;
        const bool is_head = has_head_exact || has_head_prefix || has_head_substr;
        const bool is_diff = has_diff_exact || has_diff_prefix || has_diff_substr;
        const bool is_output = p.tag == "OUT" || has_output_exact || has_output_prefix || has_output_substr || is_diff || last == "tanh";
        const bool is_unet = has_unet_exact || has_unet_substr;
        const bool is_vae = has_vae_exact || has_vae_substr;
        const bool is_diffusion = has_diffusion_exact || has_diffusion_substr;
        const bool is_recon = has_recon_exact || has_recon_substr;

        if (is_recon) return "Reconstruction";
        if (is_diffusion) return "Diffusion";
        if (is_unet) return "UNet";
        if (is_vae) {
            if (is_latent) return "VAE / Latent";
            if (is_encoder) return "VAE / Encoder";
            if (is_decoder) return "VAE / Decoder";
            return "VAE";
        }

        if (is_latent) {
            return "Latent";
        }
        if (is_text_cond) {
            return "Text / Cond";
        }
        if (is_input && !is_encoder && !is_decoder && !is_backbone) {
            return "Inputs";
        }
        if (is_bottleneck) {
            return "Bottleneck";
        }
        if (is_output) return "Outputs";
        if (is_encoder) return is_down ? "Down Blocks" : "ConditioningEncoder";
        if (is_down) return "Down Blocks";
        if (is_backbone && !is_up) return "Backbone";
        if (is_up) return "Up Blocks";
        if (is_decoder) return "Decoder";
        if (is_head && !is_output) return "Heads";
        if (is_input) return "Inputs";
        return "Blocks";
    };

    std::vector<BlockPanelEntry> entries;
    entries.reserve(static_cast<size_t>(blocks_count + 8));
    std::string current_section;

    auto push_header = [&](const std::string& section) {
        if (section.empty() || section == current_section) return;
        current_section = section;
        BlockPanelEntry h;
        h.is_header = true;
        h.section = section;
        entries.push_back(std::move(h));
    };

    if (show_projection || show_output) {
        if (show_projection) {
            const bool text_projection = Mimir::Viz::lumenTipSection(projection_label) == "Text / Cond";
            const std::string section = text_projection ? "Text / Cond" : "Outputs";
            push_header(section);
            BlockPanelEntry e;
            e.is_extra = true;
            e.img = &projection_thumb_;
            e.parsed = parse_viz_label(projection_label.empty() ? std::string("mimir/output/projection") : projection_label);
            e.section = section;
            e.text_override = "Projection";
            e.tag_override = text_projection ? "TX" : "OUT";
            entries.push_back(std::move(e));
        }
        if (show_output) {
            const bool validation_output = Mimir::Viz::lumenTipSection(output_thumb_label_) == "Validation";
            const std::string section = validation_output ? "Validation" : "Outputs";
            push_header(section);
            BlockPanelEntry e;
            e.is_extra = true;
            e.img = &output_thumb_;
            e.parsed = parse_viz_label(output_thumb_label_.empty() ? std::string("mimir/output") : output_thumb_label_);
            e.section = section;
            e.text_override = validation_output ? "Validation" : "Sortie";
            e.tag_override = validation_output ? "VAL" :
                (Mimir::Viz::isLumenModelOutput(output_thumb_label_) ? "ST" : "OUT");
            entries.push_back(std::move(e));
        }
    }

    for (int bi = 0; bi < blocks_count; ++bi) {
        const std::string parsed_label = (bi >= 0 && bi < static_cast<int>(layer_block_labels.size()))
            ? layer_block_labels[static_cast<size_t>(bi)]
            : std::string();
        ParsedVizLabel parsed = parse_viz_label(parsed_label);
        apply_arch_hint(parsed);
        const std::string section = parsed.path.find("annotation_")!=std::string::npos && !entries.empty()
            ? entries.back().section : section_for(parsed, false);
        push_header(section);

        BlockPanelEntry e;
        e.block_index = bi;
        e.img = &layer_block_images[static_cast<size_t>(bi)];
        e.parsed = std::move(parsed);
        e.section = section;
        entries.push_back(std::move(e));
    }

    // Règle UX: si une vignette "diff" existe, placer une vignette "recon"
    // immédiatement avant (dans la même section) pour comparaison visuelle directe.
    auto to_lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return s;
    };
    auto entry_key = [&](const BlockPanelEntry& e) -> std::string {
        return to_lower(e.parsed.layer_path + " " + e.parsed.path + " " + e.parsed.short_text + " " + e.text_override);
    };
    auto is_diff_entry = [&](const BlockPanelEntry& e) -> bool {
        if (e.is_header) return false;
        const std::string k = entry_key(e);
        if(k.find("annotation_")!=std::string::npos)return false;
        return k.find("resdiff") != std::string::npos ||
               k.find("/diff") != std::string::npos ||
               k.find("diff_") != std::string::npos ||
               k.find(" err/") != std::string::npos;
    };
    auto is_recon_entry = [&](const BlockPanelEntry& e) -> bool {
        if (e.is_header) return false;
        const std::string k = entry_key(e);
        return (k.find("recon") != std::string::npos) && (k.find("resdiff") == std::string::npos);
    };

    for (size_t i = 0; i < entries.size(); ++i) {
        if (!is_diff_entry(entries[i])) continue;
        const std::string section = entries[i].section;

        // Déjà juste avant ?
        if (i > 0 && !entries[i - 1].is_header && entries[i - 1].section == section && is_recon_entry(entries[i - 1])) {
            continue;
        }

        // Chercher d'abord un recon dans la même section avant diff.
        size_t recon_idx = entries.size();
        for (size_t j = 0; j < i; ++j) {
            if (entries[j].is_header) continue;
            if (entries[j].section != section) continue;
            if (is_recon_entry(entries[j])) recon_idx = j;
        }

        // Sinon, chercher après diff dans la même section.
        if (recon_idx == entries.size()) {
            for (size_t j = i + 1; j < entries.size(); ++j) {
                if (entries[j].is_header) continue;
                if (entries[j].section != section) continue;
                if (is_recon_entry(entries[j])) {
                    recon_idx = j;
                    break;
                }
            }
        }

        if (recon_idx == entries.size() || recon_idx == i) continue;

        BlockPanelEntry recon = std::move(entries[recon_idx]);
        entries.erase(entries.begin() + static_cast<std::ptrdiff_t>(recon_idx));
        if (recon_idx < i) {
            --i;
        }
        entries.insert(entries.begin() + static_cast<std::ptrdiff_t>(i), std::move(recon));
    }

    // Lumen's forward and validation frames arrive in computation order. Gather
    // each section once, retaining original block indices and annotation order.
    const bool lumen_sections = std::any_of(entries.begin(), entries.end(), [](const BlockPanelEntry& entry) {
        return !entry.is_header && !Mimir::Viz::lumenTipSection(
            entry.parsed.path + " " + entry.parsed.layer_path + " " + entry.parsed.extra).empty();
    });
    if (lumen_sections) {
        std::vector<std::string> sections = {"Inputs", "Text / Cond", "Diffusion", "Latent",
            "VAE / Encoder", "VAE / Latent", "VAE / Decoder", "Reconstruction", "Outputs", "Validation"};
        for (const auto& entry : entries)
            if (!entry.is_header && std::find(sections.begin(), sections.end(), entry.section) == sections.end())
                sections.push_back(entry.section);
        std::vector<BlockPanelEntry> grouped;
        grouped.reserve(entries.size());
        for (const auto& section : sections) {
            bool header_added = false;
            for (auto& entry : entries) {
                if (entry.is_header || entry.section != section) continue;
                if (!header_added) {
                    BlockPanelEntry header;
                    header.is_header = true;
                    header.section = section;
                    grouped.push_back(std::move(header));
                    header_added = true;
                }
                grouped.push_back(std::move(entry));
            }
        }
        entries = std::move(grouped);
    }

    if (blocks_tips_visible_limit_ > 0) {
        int remaining = blocks_tips_visible_limit_;
        std::vector<BlockPanelEntry> limited;
        limited.reserve(entries.size());

        std::optional<BlockPanelEntry> pending_header;
        for (const auto& entry : entries) {
            if (entry.is_header) {
                pending_header = entry;
                continue;
            }

            const bool keep = entry.is_extra || (remaining > 0);
            if (!keep) continue;

            if (pending_header.has_value()) {
                limited.push_back(*pending_header);
                pending_header.reset();
            }

            limited.push_back(entry);
            if (!entry.is_extra && remaining > 0) {
                --remaining;
            }
        }

        entries.swap(limited);
    }

    if (entries.empty()) return;

    std::vector<vizgfx::Vector2f> entry_positions;
    entry_positions.reserve(entries.size());

    int layout_col = 0;
    int layout_y = start_y;
    for (const auto& entry : entries) {
        if (entry.is_header) {
            if (layout_col != 0) {
                layout_col = 0;
                layout_y += cell_h;
            }
            entry_positions.emplace_back(static_cast<float>(start_x), static_cast<float>(layout_y));
            layout_y += section_h + section_gap;
        } else {
            const int x = start_x + layout_col * (thumb + margin);
            entry_positions.emplace_back(static_cast<float>(x), static_cast<float>(layout_y));
            layout_col += 1;
            if (layout_col >= max_cols) {
                layout_col = 0;
                layout_y += cell_h;
            }
        }
    }

    const float content_h = static_cast<float>((layout_y - start_y) + ((layout_col != 0) ? cell_h : 0));

    // PS (user): MAJ en temps réel des scrollers avec les valeurs runtime.
    // Ici: si le runtime ajoute/retire des items, on conserve la position relative
    // tant que l'utilisateur n'est pas en train de drag.
    const float prev_max = blocks_scroll_max_;
    const float prev_y = blocks_scroll_y_;

    blocks_scroll_max_ = std::max(0.0f, content_h - area.size.y);
    if (!dragging_blocks_scrollbar_ && prev_max > 1e-6f) {
        const float t = std::clamp(prev_y / prev_max, 0.0f, 1.0f);
        blocks_scroll_y_ = t * blocks_scroll_max_;
    }
    blocks_scroll_y_ = std::clamp(blocks_scroll_y_, 0.0f, std::max(0.0f, blocks_scroll_max_));

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        const int x = static_cast<int>(std::lround(entry_positions[i].x));
        const int base_y = static_cast<int>(std::lround(entry_positions[i].y));
        const int y = base_y - static_cast<int>(std::lround((double)blocks_scroll_y_));

        if (entry.is_header) {
            if ((y - section_h) > static_cast<int>(area.position.y + area.size.y)) continue;
            if ((y + section_h + section_gap) < static_cast<int>(area.position.y)) continue;

            const float bar_w = std::max(80.0f, area.size.x - 12.0f);
            vizgfx::RectangleShape bar(vizgfx::Vector2f(bar_w, static_cast<float>(section_h)));
            bar.setPosition(vizgfx::Vector2f(static_cast<float>(start_x), static_cast<float>(y)));
            bar.setFillColor(vizgfx::Color(28, 30, 38, 220));
            bar.setOutlineColor(section_color(entry.section));
            bar.setOutlineThickness(1.0f);
            window->draw(bar);

            vizgfx::RectangleShape accent(vizgfx::Vector2f(8.0f, static_cast<float>(section_h)));
            accent.setPosition(vizgfx::Vector2f(static_cast<float>(start_x), static_cast<float>(y)));
            accent.setFillColor(section_color(entry.section));
            window->draw(accent);

            if (font_loaded) {
                vizgfx::Text t(font);
                t.setFont(font);
                t.setCharacterSize(13);
                t.setStyle(vizgfx::Text::Bold);
                t.setFillColor(vizgfx::Color(236, 236, 240));
                t.setPosition(vizgfx::Vector2f(static_cast<float>(start_x + 14), static_cast<float>(y + 2)));
                t.setString(vizgfx::String::fromUtf8(entry.section.begin(), entry.section.end()));
                window->draw(t);
            }
            continue;
        }

        if (!entry.img) continue;

        // Cull (best-effort) pour limiter les draws
        if ((y - cell_h) > static_cast<int>(area.position.y + area.size.y)) continue;
        if ((y + 2 * cell_h) < static_cast<int>(area.position.y)) continue;

        ParsedVizLabel display_parsed = entry.parsed;
        if (entry.is_extra) {
            // Projection/Sortie ajoutées ici sont des tips de présentation créés
            // par le Visualizer, pas des nœuds provenant du graphe du modèle.
            display_parsed.tag = "VIZ";
            if (display_parsed.extra.empty()) display_parsed.extra = entry.text_override;
        } else if (entry.section.rfind("VAE / ", 0) == 0) {
            if (display_parsed.layer_path.rfind("vae_conv/", 0) == 0) {
                display_parsed.layer_path.erase(0, std::string("vae_conv/").size());
            }

            const std::string extra_lower = lowercase_ascii(display_parsed.extra);
            if (extra_lower.rfind("encoder/", 0) == 0 ||
                extra_lower.rfind("latent/", 0) == 0 ||
                extra_lower.rfind("decoder/", 0) == 0) {
                display_parsed.extra.clear();
            }
        }
        if(entry_key(entry).find("annotation_")!=std::string::npos)display_parsed.tag="VIZ";
        const std::string use_tag = display_tip_tag(display_parsed);

        // Frame
        vizgfx::RectangleShape frame(vizgfx::Vector2f(thumb + 4, thumb + 4));
        frame.setPosition(vizgfx::Vector2f(static_cast<float>(x - 2), static_cast<float>(y - 2)));
        frame.setFillColor(vizgfx::Color::Transparent);
        frame.setOutlineColor(color_for_tag(use_tag));
        frame.setOutlineThickness(1);
        window->draw(frame);

        if (!entry.is_extra) {
            const int bi = entry.block_index;
            if (bi >= 0 && bi < blocks_count) {
                last_block_rects_[static_cast<size_t>(bi)] = make_rect(
                    static_cast<float>(x - 2),
                    static_cast<float>(y - 2),
                    static_cast<float>(thumb + 4),
                    static_cast<float>(thumb + 4));
            }
        }

        position_sprite_centered_in_box(entry.img->sprite, static_cast<float>(x), static_cast<float>(y), static_cast<float>(thumb));
        window->draw(entry.img->sprite);

        vizgfx::RectangleShape label(vizgfx::Vector2f(static_cast<float>(thumb), static_cast<float>(label_h)));
        label.setPosition(vizgfx::Vector2f(static_cast<float>(x), static_cast<float>(y + thumb + 3)));
        label.setFillColor(vizgfx::Color(50, 50, 60, 200));
        window->draw(label);

        if (font_loaded) {
            vizgfx::Text t(font);
            t.setFont(font);
            t.setCharacterSize(12);
            t.setFillColor(vizgfx::Color(230, 230, 235));
            t.setPosition(vizgfx::Vector2f(static_cast<float>(x + 4), static_cast<float>(y + thumb + 1)));

            std::string text = format_tip_display_label(display_parsed);
            const auto name=entry_key(entry);
            if(name.find("annotation_features")!=std::string::npos)text="Features / liens";
            else if(name.find("annotation_impact")!=std::string::npos)text="Poids spatiaux";
            else if(name.find("resdiff_abs")!=std::string::npos)text="resdiff_abs";
            text = clamp_text_end(text, 18);
            t.setString(vizgfx::String::fromUtf8(text.begin(), text.end()));
            window->draw(t);
        }
    }

    // Restore view
    window->setView(old_view);

    // Scrollbar (slicer) cliquable (dessiné dans l'UI normale, hors view clipping)
    // NB: rects en coordonnées UI (pixels)
    last_blocks_scroll_track_rect_ = make_rect(0.f, 0.f, 0.f, 0.f);
    last_blocks_scroll_thumb_rect_ = make_rect(0.f, 0.f, 0.f, 0.f);
    if (blocks_scroll_max_ > 0.0f) {
        const float track_w = 10.0f;
        const float pad = 2.0f;
        const float tx = area.position.x + std::max(0.0f, area.size.x - track_w - pad);
        const float ty = area.position.y;
        const float th = std::max(0.0f, area.size.y);
        last_blocks_scroll_track_rect_ = make_rect(tx, ty, track_w, th);

        const float content_h2 = area.size.y + blocks_scroll_max_;
        const float visible_ratio = (content_h2 > 1.0f) ? std::clamp(area.size.y / content_h2, 0.05f, 1.0f) : 1.0f;
        const float thumb_h = std::max(24.0f, th * visible_ratio);
        const float denom = std::max(1.0f, th - thumb_h);
        const float t = (blocks_scroll_max_ > 0.0f) ? std::clamp(blocks_scroll_y_ / blocks_scroll_max_, 0.0f, 1.0f) : 0.0f;
        const float thumb_y = ty + t * denom;
        last_blocks_scroll_thumb_rect_ = make_rect(tx, thumb_y, track_w, thumb_h);

        vizgfx::RectangleShape track(vizgfx::Vector2f(track_w, th));
        track.setPosition(vizgfx::Vector2f(tx, ty));
        track.setFillColor(vizgfx::Color(35, 35, 42, 200));
        window->draw(track);

        vizgfx::RectangleShape thumb(vizgfx::Vector2f(track_w, thumb_h));
        thumb.setPosition(vizgfx::Vector2f(tx, thumb_y));
        thumb.setFillColor(vizgfx::Color(90, 90, 110, 220));
        window->draw(thumb);
    }
}

void Visualizer::renderGeneratedImages() {
    if (generated_images.empty()) {
        setPanelContentHeight(PanelId::Generated, 0.0f);
        last_generated_rects_.clear();
        last_generated_indices_.clear();
        return;
    }

    const auto area = panelContentRect(PanelId::Generated);

    const int img_display_size = script_scene_.value("image_size", workbenchPreviewSize(area));
    const int margin = 20;
    const int start_x = static_cast<int>(area.position.x);
    const int start_y = static_cast<int>(area.position.y -
        panel_scroll_y_[static_cast<size_t>(PanelId::Generated)]);

    const int cell_w = img_display_size + margin;
    const int cell_h = img_display_size + margin + 26;
    const int cols = std::max(1, static_cast<int>(area.size.x) / std::max(1, cell_w));
    const size_t n = generated_images.size();
    const size_t start_idx = 0;
    const int content_rows = std::max(1, static_cast<int>(
        (n + static_cast<size_t>(cols) - 1) / static_cast<size_t>(cols)));
    setPanelContentHeight(PanelId::Generated,
        static_cast<float>(content_rows * cell_h));

    last_generated_rects_.clear();
    last_generated_indices_.clear();
    last_generated_rects_.reserve(n);
    last_generated_indices_.reserve(n);

    // Clip au contenu du panneau (overflow: clip)
    const vizgfx::View old_view = window->getView();
    {
        const float ww = static_cast<float>(std::max(1, window_width));
        const float hh = static_cast<float>(std::max(1, window_height));

        const float left = std::clamp(area.position.x, 0.0f, ww);
        const float top = std::clamp(area.position.y, 0.0f, hh);
        const float right = std::clamp(area.position.x + area.size.x, 0.0f, ww);
        const float bottom = std::clamp(area.position.y + area.size.y, 0.0f, hh);
        const float w = std::max(0.0f, right - left);
        const float h = std::max(0.0f, bottom - top);

        if (w > 0.0f && h > 0.0f) {
            vizgfx::View v(make_rect(left, top, w, h));
            v.setViewport(make_rect(left / ww, top / hh, w / ww, h / hh));
            window->setView(v);
        }
    }

    for (size_t i = 0; i < n; ++i) {
        auto& img = generated_images[start_idx + i];
        
        int col = static_cast<int>(i % static_cast<size_t>(cols));
        int row = static_cast<int>(i / static_cast<size_t>(cols));
        
        int x = start_x + col * cell_w;
        int y = start_y + row * cell_h;

        // Cadre
        vizgfx::RectangleShape frame(vizgfx::Vector2f(img_display_size + 4, img_display_size + 4));
        frame.setPosition(vizgfx::Vector2f(x - 2, y - 2));
        frame.setFillColor(vizgfx::Color::Transparent);
        frame.setOutlineColor(vizgfx::Color(100, 150, 200));
        frame.setOutlineThickness(2);
        window->draw(frame);

        last_generated_rects_.push_back(make_rect(static_cast<float>(x - 2), static_cast<float>(y - 2), static_cast<float>(img_display_size + 4), static_cast<float>(img_display_size + 4)));
        last_generated_indices_.push_back(static_cast<int>(start_idx + i));

        // Fit to the script-selected preview size (texture sizes may differ).
        const auto bounds = img.sprite.getLocalBounds();
        if (bounds.size.x > 0.f && bounds.size.y > 0.f) {
            const float scale = std::min(img_display_size / bounds.size.x, img_display_size / bounds.size.y);
            img.sprite.setScale({scale, scale});
        }
        position_sprite_centered_in_box(img.sprite, static_cast<float>(x), static_cast<float>(y), static_cast<float>(img_display_size));
        window->draw(img.sprite);

        // Caption remains readable beneath each source/result preview.
        vizgfx::RectangleShape label(vizgfx::Vector2f(img_display_size, 20));
        label.setPosition(vizgfx::Vector2f(x, y + img_display_size + 5));
        label.setFillColor(vizgfx::Color(50, 50, 60, 200));
        window->draw(label);
        if (font_loaded) {
            vizgfx::Text caption(font);
            caption.setCharacterSize(12);
            caption.setFillColor(vizgfx::Color(230, 240, 248));
            caption.setPosition({static_cast<float>(x + 4), static_cast<float>(y + img_display_size + 7)});
            const std::string value = clamp_text_end(img.prompt, std::max(8, img_display_size / 8));
            caption.setString(vizgfx::String::fromUtf8(value.begin(), value.end()));
            window->draw(caption);
        }
    }

    window->setView(old_view);
}

void Visualizer::renderTrainingProgress() {
    const auto area = panelContentRect(PanelId::Training);
    setPanelContentHeight(PanelId::Training, 52.0f);
    const int bar_x = static_cast<int>(area.position.x);
    const int bar_width = std::max(10, static_cast<int>(area.size.x));
    const int global_h = std::clamp(static_cast<int>(area.size.y * 0.34f), 14, 24);
    const int batch_h = std::clamp(static_cast<int>(area.size.y * 0.22f), 10, 16);
    const int gap = 8;
    const int stack_h = global_h + gap + batch_h;
    const int global_y = static_cast<int>(area.position.y + std::max(0.0f, (area.size.y - static_cast<float>(stack_h)) * 0.5f));
    const int batch_y = global_y + global_h + gap;

    const bool has_epoch_info = (current_total_epochs > 0);
    const bool has_batch_info = (current_total_batches > 0);

    const int total_epochs = std::max(1, current_total_epochs);
    const int total_batches = std::max(1, current_total_batches);

    const int epoch_index = std::clamp(current_epoch - 1, 0, std::max(0, total_epochs - 1));
    const int batch_done = std::clamp(current_batch, 0, total_batches);

    const float batch_target = has_batch_info
        ? std::clamp(static_cast<float>(batch_done) / static_cast<float>(total_batches), 0.0f, 1.0f)
        : 0.0f;
    const float global_target = has_epoch_info
        ? std::clamp((static_cast<float>(epoch_index) + batch_target) / static_cast<float>(total_epochs), 0.0f, 1.0f)
        : 0.0f;

    // Lissage visuel (monotone local) pour éviter les sauts d'affichage.
    const auto smooth_to = [](float current, float target, float alpha) {
        if (!std::isfinite(current)) current = target;
        if (target + 0.25f < current) return target; // reset rapide si run repart de plus bas
        const float v = current + (target - current) * alpha;
        return (std::fabs(v - target) < 1e-4f) ? target : v;
    };

    progress_epoch_display_ = smooth_to(progress_epoch_display_, global_target, 0.24f);
    progress_batch_display_ = smooth_to(progress_batch_display_, batch_target, 0.30f);

    const auto batch_speed_color = [&](int batch_ms) -> vizgfx::Color {
        // 0 = rapide (vert), 1 = lent (rouge)
        const float lo = 20.0f;
        const float hi = 2000.0f;
        const float t = std::clamp((static_cast<float>(std::max(0, batch_ms)) - lo) / (hi - lo), 0.0f, 1.0f);

        const float fast_r = 124.0f, fast_g = 218.0f, fast_b = 120.0f;
        const float slow_r = 226.0f, slow_g = 116.0f, slow_b = 92.0f;

        const auto mix = [&](float a, float b) -> uint8_t {
            return static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(a + (b - a) * t)), 0, 255));
        };

        return vizgfx::Color(mix(fast_r, slow_r), mix(fast_g, slow_g), mix(fast_b, slow_b), 245);
    };

    // Barre globale (epochs)
    vizgfx::RectangleShape progress_bg(vizgfx::Vector2f(static_cast<float>(bar_width), static_cast<float>(global_h)));
    progress_bg.setPosition(vizgfx::Vector2f(static_cast<float>(bar_x), static_cast<float>(global_y)));
    progress_bg.setFillColor(vizgfx::Color(50, 50, 60, 230));
    progress_bg.setOutlineColor(vizgfx::Color(88, 88, 102, 235));
    progress_bg.setOutlineThickness(1.5f);
    window->draw(progress_bg);

    const float epoch_w = static_cast<float>(bar_width) * progress_epoch_display_;
    if (epoch_w > 0.0f) {
        vizgfx::RectangleShape progress_bar(vizgfx::Vector2f(epoch_w, static_cast<float>(global_h)));
        progress_bar.setPosition(vizgfx::Vector2f(static_cast<float>(bar_x), static_cast<float>(global_y)));
        progress_bar.setFillColor(vizgfx::Color(96, 184, 112, 240));
        window->draw(progress_bar);
    }

    // Barre batch (courant) sous la barre globale.
    vizgfx::RectangleShape batch_bg(vizgfx::Vector2f(static_cast<float>(bar_width), static_cast<float>(batch_h)));
    batch_bg.setPosition(vizgfx::Vector2f(static_cast<float>(bar_x), static_cast<float>(batch_y)));
    batch_bg.setFillColor(vizgfx::Color(42, 42, 52, 230));
    batch_bg.setOutlineColor(vizgfx::Color(78, 78, 94, 220));
    batch_bg.setOutlineThickness(1.0f);
    window->draw(batch_bg);

    if (has_batch_info) {
        const float batch_w = static_cast<float>(bar_width) * progress_batch_display_;
        if (batch_w > 0.0f) {
            vizgfx::RectangleShape batch_fg(vizgfx::Vector2f(batch_w, static_cast<float>(batch_h)));
            batch_fg.setPosition(vizgfx::Vector2f(static_cast<float>(bar_x), static_cast<float>(batch_y)));
            batch_fg.setFillColor(batch_speed_color(current_batch_time_ms));
            window->draw(batch_fg);
        }
    }

    // Fallback visuel si l'info d'epoch est absente: bande animée discrète.
    if (!has_epoch_info) {
        const uint64_t t = steady_now_ms();
        const float phase = static_cast<float>((t % 1600ULL) / 1600.0);
        const float seg_w = std::max(24.0f, static_cast<float>(bar_width) * 0.18f);
        const float x = static_cast<float>(bar_x) + (static_cast<float>(bar_width) + seg_w) * phase - seg_w;
        vizgfx::RectangleShape ind(vizgfx::Vector2f(seg_w, static_cast<float>(global_h)));
        ind.setPosition(vizgfx::Vector2f(x, static_cast<float>(global_y)));
        ind.setFillColor(vizgfx::Color(110, 165, 215, 120));
        window->draw(ind);
    }

    if (font_loaded) {
        int pct = static_cast<int>(std::lround(progress_epoch_display_ * 100.0f));
        pct = std::clamp(pct, 0, 100);
        std::string txt_global;
        if (has_epoch_info) {
            txt_global = std::to_string(pct) + "%";
        } else {
            txt_global = "Initialisation";
        }

        vizgfx::Text tg(font);
        tg.setFont(font);
        tg.setCharacterSize(12);
        tg.setFillColor(vizgfx::Color(235, 235, 240));
        tg.setString(vizgfx::String::fromUtf8(txt_global.begin(), txt_global.end()));
        const auto bg = tg.getLocalBounds();
        const float txg = static_cast<float>(bar_x) + (static_cast<float>(bar_width) - bg.size.x) * 0.5f - bg.position.x;
        const float tyg = static_cast<float>(global_y) + (static_cast<float>(global_h) - bg.size.y) * 0.5f - bg.position.y - 1.0f;
        tg.setPosition(vizgfx::Vector2f(std::round(txg), std::round(tyg)));
        window->draw(tg);

        std::string txt_batch;
        if (has_batch_info) {
            const int bpct = std::clamp(static_cast<int>(std::lround(progress_batch_display_ * 100.0f)), 0, 100);
            txt_batch = "batch " + std::to_string(bpct) + "%";
        } else {
            txt_batch = "batch --";
        }
        vizgfx::Text tb(font);
        tb.setFont(font);
        tb.setCharacterSize(11);
        tb.setFillColor(vizgfx::Color(220, 228, 210));
        tb.setString(vizgfx::String::fromUtf8(txt_batch.begin(), txt_batch.end()));
        const auto bb = tb.getLocalBounds();
        const float txb = static_cast<float>(bar_x) + (static_cast<float>(bar_width) - bb.size.x) * 0.5f - bb.position.x;
        const float tyb = static_cast<float>(batch_y) + (static_cast<float>(batch_h) - bb.size.y) * 0.5f - bb.position.y - 1.0f;
        tb.setPosition(vizgfx::Vector2f(std::round(txb), std::round(tyb)));
        window->draw(tb);
    }
}

void Visualizer::renderLossGraph() {
    if (!window) return;

    const bool have_full = !full_loss_history.empty();
    if (!have_full && loss_history.empty()) return;

    const auto area = panelContentRect(PanelId::Graph);
    const int graph_x = static_cast<int>(area.position.x);
    const int graph_y = static_cast<int>(area.position.y);
    const int graph_width = std::max(120, static_cast<int>(area.size.x));
    const int graph_height = std::max(120, static_cast<int>(area.size.y));

    // Fond du graphique
    vizgfx::RectangleShape graph_bg(vizgfx::Vector2f(static_cast<float>(graph_width), static_cast<float>(graph_height)));
    graph_bg.setPosition(vizgfx::Vector2f(static_cast<float>(graph_x), static_cast<float>(graph_y)));
    graph_bg.setFillColor(mimir::viz::theme::background);
    graph_bg.setOutlineColor(vizgfx::Color(100, 100, 120));
    graph_bg.setOutlineThickness(1);
    window->draw(graph_bg);

    // Zone de plot (padding pour l'échelle)
    const float pad_l = font_loaded ? 54.f : 10.f;
    const float pad_r = 10.f;
    const float pad_t = font_loaded ? 8.f : 10.f;
    const float pad_b = font_loaded ? 22.f : 10.f;
    const float plot_x = static_cast<float>(graph_x) + pad_l;
    const float plot_y = static_cast<float>(graph_y) + pad_t;
    const float plot_w = std::max(10.f, static_cast<float>(graph_width) - pad_l - pad_r);
    const float plot_h = std::max(10.f, static_cast<float>(graph_height) - pad_t - pad_b);

    // Min/Max pour normalisation (full history par défaut)
    float min_loss = 0.0f;
    float max_loss = 1.0f;
    if (have_full && has_loss_stats_) {
        min_loss = loss_min_;
        max_loss = loss_max_;
    } else {
        // fallback: calculer sur l'historique glissant
        min_loss = *std::min_element(loss_history.begin(), loss_history.end());
        max_loss = *std::max_element(loss_history.begin(), loss_history.end());
    }
    float range = max_loss - min_loss;
    if (!(range > 0.0f)) range = 1.0f;

    // Grille + échelle Y
    const int ticks = 5;
    for (int i = 0; i < ticks; ++i) {
        const float t = (ticks <= 1) ? 0.f : (static_cast<float>(i) / static_cast<float>(ticks - 1));
        const float y = plot_y + (1.f - t) * plot_h;

        vizgfx::RectangleShape grid(vizgfx::Vector2f(plot_w, 1.f));
        grid.setPosition(vizgfx::Vector2f(plot_x, y));
        grid.setFillColor(vizgfx::Color(70, 70, 80, (i == 0 || i == ticks - 1) ? 120 : 70));
        window->draw(grid);

        if (font_loaded) {
            const float v = min_loss + t * range;
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << v;
            const std::string label = ss.str();
            vizgfx::Text txt(font);
            txt.setFont(font);
            txt.setCharacterSize(12);
            txt.setFillColor(vizgfx::Color(220, 220, 228));
            txt.setPosition(vizgfx::Vector2f(static_cast<float>(graph_x) + 6.f, y - 8.f));
            txt.setString(vizgfx::String::fromUtf8(label.begin(), label.end()));
            window->draw(txt);
        }
    }

    // Données à tracer: historique complet (non-glissant)
    const size_t n_total = have_full ? full_loss_history.size() : loss_history.size();
    if (n_total < 2) return;

    // Downsample: un point par pixel (approx) sur l'axe X
    const size_t max_points = static_cast<size_t>(std::max(2.f, plot_w));
    const size_t n_draw = std::min(n_total, max_points);

    auto loss_at = [&](size_t idx) -> float {
        if (have_full) return full_loss_history[idx].loss;
        return loss_history[idx];
    };

    vizgfx::VertexArray line(vizgfx::PrimitiveType::LineStrip, n_draw);
    for (size_t i = 0; i < n_draw; ++i) {
        const size_t idx = (n_draw <= 1) ? 0 : (i * (n_total - 1)) / (n_draw - 1);
        const float loss = loss_at(idx);
        const float x = plot_x + (static_cast<float>(i) / static_cast<float>(n_draw - 1)) * plot_w;
        const float normalized = (loss - min_loss) / range;
        const float y = plot_y + (1.f - std::clamp(normalized, 0.f, 1.f)) * plot_h;

        line[i].position = vizgfx::Vector2f(x, y);
        line[i].color = getLossColor(loss);
    }
    window->draw(line);

    // Labels X (début/fin) + info
    if (font_loaded) {
        const size_t start_step = 0;
        const size_t end_step = (n_total > 0) ? (n_total - 1) : 0;
        auto drawSmall = [&](float x, float y, const std::string& s) {
            vizgfx::Text t(font);
            t.setFont(font);
            t.setCharacterSize(12);
            t.setFillColor(vizgfx::Color(220, 220, 228));
            t.setPosition(vizgfx::Vector2f(x, y));
            t.setString(vizgfx::String::fromUtf8(s.begin(), s.end()));
            window->draw(t);
        };

        drawSmall(plot_x, plot_y + plot_h + 4.f, std::to_string(start_step));
        const std::string end_s = std::to_string(end_step);
        drawSmall(plot_x + plot_w - 8.f * static_cast<float>(end_s.size()), plot_y + plot_h + 4.f, end_s);

        // Titre court / rappel
        std::ostringstream ss;
        ss << "steps=" << n_total;
        ss << "  min=" << std::fixed << std::setprecision(4) << min_loss;
        ss << "  max=" << std::fixed << std::setprecision(4) << max_loss;
        drawSmall(plot_x, static_cast<float>(graph_y) - 16.f, ss.str());
    }
}

void Visualizer::renderMetrics() {
    // Affichage des métriques textuelles (simulé avec rectangles colorés)
    const auto area = panelContentRect(PanelId::Metrics);
    const int panel_w = std::max(120, static_cast<int>(area.size.x));
    int metrics_x = static_cast<int>(area.position.x);
    int metrics_y = static_cast<int>(area.position.y -
        panel_scroll_y_[static_cast<size_t>(PanelId::Metrics)]);
    int line_height = 25;

    const vizgfx::View old_view = window->getView();
    {
        const float ww = static_cast<float>(std::max(1, window_width));
        const float hh = static_cast<float>(std::max(1, window_height));
        const float left = std::clamp(area.position.x, 0.0f, ww);
        const float top = std::clamp(area.position.y, 0.0f, hh);
        const float right = std::clamp(area.position.x + area.size.x, 0.0f, ww);
        const float bottom = std::clamp(area.position.y + area.size.y, 0.0f, hh);
        const float width = std::max(0.0f, right - left);
        const float height = std::max(0.0f, bottom - top);
        if (width > 0.0f && height > 0.0f) {
            vizgfx::View view(make_rect(left, top, width, height));
            view.setViewport(make_rect(left / ww, top / hh, width / ww, height / hh));
            window->setView(view);
        }
    }
    float metrics_content_height = 4.0f * static_cast<float>(line_height) + 20.0f;

    // Bouton stop: hitbox recalculée à chaque frame.
    last_debug_json_button_rect_.reset();
    last_pause_button_rect_.reset();
    last_stop_button_rect_.reset();
    last_validation_button_rect_.reset();
    last_skip_button_rect_.reset();

    // Live tuning controls: hitboxes recalculées à chaque frame.
    last_live_lr_track_.reset();
    last_live_lr_thumb_.reset();
    last_live_lrwu_track_.reset();
    last_live_lrwu_thumb_.reset();
    last_live_klb_track_.reset();
    last_live_klb_thumb_.reset();
    last_live_klwu_track_.reset();
    last_live_klwu_thumb_.reset();
    last_live_lr_value_box_.reset();
    last_live_lrwu_value_box_.reset();
    last_live_klb_value_box_.reset();
    last_live_klwu_value_box_.reset();
    last_live_recon_loss_box_.reset();
    last_live_kl_enable_box_.reset();
    last_live_overrides_box_.reset();
    for (auto& rect : dataset_text_section_rects_) rect = make_rect(0.f, 0.f, 0.f, 0.f);

    // Commands occupy their own card at the top, separate from numeric rows.
    const float command_y = static_cast<float>(metrics_y);
    const float command_gap = 8.f;
    const int command_cols = panel_w >= 460 ? 3 : (panel_w >= 300 ? 2 : 1);
    const int command_rows = (7 + command_cols - 1) / command_cols;
    const float command_width = static_cast<float>(panel_w) - 16.f;
    const float command_height = 110.f + command_rows * 40.f;
    last_commands_rect_ = make_rect(static_cast<float>(metrics_x), command_y,
                                   static_cast<float>(panel_w), command_height);
    vizgfx::RectangleShape command_card({static_cast<float>(panel_w), command_height});
    command_card.setPosition({static_cast<float>(metrics_x), command_y});
    command_card.setFillColor(mimir::viz::theme::header);
    command_card.setOutlineColor(vizgfx::Color(59, 68, 82));
    command_card.setOutlineThickness(1.f);
    window->draw(command_card);
    if (font_loaded) {
        vizgfx::Text title(font); title.setCharacterSize(14); title.setStyle(vizgfx::Text::Bold);
        title.setPosition({static_cast<float>(metrics_x)+8.f, command_y+5.f});
        title.setFillColor(mimir::viz::theme::text); title.setString("Pilotage"); window->draw(title);
    }
    auto drawCommand = [&](int index, const std::string& label, vizgfx::Color color,
                           std::optional<vizgfx::FloatRect>& hitbox) {
        const bool primary = index == 0;
        const float bw = primary ? command_width :
            (command_width - (command_cols-1)*command_gap) / command_cols;
        const float bh = primary ? 48.f : 34.f;
        const int grid_index = std::max(0,index-1);
        const float bx = static_cast<float>(metrics_x)+8.f +
            (primary ? 0.f : (grid_index%command_cols)*(bw+command_gap));
        const float by = command_y+28.f +
            (primary ? 0.f : 56.f+(grid_index/command_cols)*40.f);
        hitbox = make_rect(bx,by,bw,bh);
        const bool hovered=hitbox->contains(picker_mouse_) && !show_help_overlay_ && !zoom_active_ && !config_editor_.visible;
        const auto fill=primary?color:(index==1?vizgfx::Color(78,43,49):mimir::viz::theme::control);
        mimir::viz::ui::buttonSurface(*window,*hitbox,fill,hovered);
        const float text_x=bx+(primary?40.f:12.f);
        if(!primary && index>=2 && index<=5)
            mimir::viz::ui::roundedRect(*window,{{bx+3.f,by+8.f},{3.f,bh-16.f}},color,1.f);
        if(primary) {
            const bool paused=std::atomic_load(&training_paused_)->load();
            if(paused) {
                for(int i=0;i<10;++i) {
                    vizgfx::RectangleShape stroke({1.f,float(18-2*std::abs(i-1))});
                    stroke.setPosition({bx+15.f+i,by+15.f+std::abs(i-1)});stroke.setFillColor(vizgfx::Color::White);window->draw(stroke);
                }
            } else {
                for(float dx:{15.f,23.f}) {
                    vizgfx::RectangleShape bar({4.f,16.f});bar.setPosition({bx+dx,by+16.f});bar.setFillColor(vizgfx::Color::White);window->draw(bar);
                }
            }
        }
        if (font_loaded) {
            vizgfx::Text text(font);unsigned text_size=primary?14:12;text.setCharacterSize(text_size);
            text.setStyle(vizgfx::Text::Bold);text.setFillColor(mimir::viz::theme::text);
            text.setString(vizgfx::String::fromUtf8(label.begin(),label.end()));
            while(text.getLocalBounds().size.x>bw-(text_x-bx)-10.f && text_size>10)text.setCharacterSize(--text_size);
            text.setPosition({text_x,by+(primary?7.f:9.f)});window->draw(text);
            if(primary) {
                text.setCharacterSize(10);text.setStyle(vizgfx::Text::Regular);text.setFillColor(vizgfx::Color(198,218,239));
                const std::string hint=std::atomic_load(&training_paused_)->load()?"Entraînement en pause":"Entraînement en cours";text.setString(vizgfx::String::fromUtf8(hint.begin(),hint.end()));text.setPosition({text_x,by+27.f});window->draw(text);
            }
        }
    };
    const bool paused = std::atomic_load(&training_paused_)->load();
    drawCommand(0, paused ? "Reprendre" : "Mettre en pause",
                paused ? vizgfx::Color(35,145,95) : vizgfx::Color(40,110,170), last_pause_button_rect_);
    drawCommand(1, stop_training_requested_.load() ? "Arrêt demandé" : "Arrêter",
                vizgfx::Color(155,55,55), last_stop_button_rect_);
    const bool validation_on = validation_enabled_.load();
    drawCommand(2, validation_on ? "Validation : oui" : "Validation : non",
                validation_on ? vizgfx::Color(35,115,75) : vizgfx::Color(65,75,85), last_validation_button_rect_);
    const auto skips = std::atomic_load(&skip_control_);
    const bool skips_on = skips->available.load() && skips->enabled.load();
    drawCommand(3, !skips->available.load() ? "Skip : indisponible" : (skips_on ? "Skip : oui" : "Skip : non"),
                skips_on ? vizgfx::Color(35,115,75) : vizgfx::Color(65,75,85), last_skip_button_rect_);

    drawCommand(7, "Exporter JSON [D]", vizgfx::Color(65,95,155), last_debug_json_button_rect_);
    if (font_loaded) {
        const auto status = std::atomic_load(&runtime_config_)->debugJsonStatus();
        vizgfx::Text text(font); text.setCharacterSize(11);
        text.setPosition({static_cast<float>(metrics_x)+8.f, command_y+command_height-22.f});
        text.setFillColor(vizgfx::Color(145,215,235));
        text.setString(vizgfx::String::fromUtf8(status.begin(),status.end()));
        window->draw(text);
    }
    metrics_y += static_cast<int>(command_height)+28;
    metrics_content_height += command_height+28.f;

    auto drawMetricBox = [&](int y_offset, vizgfx::Color color) {
        vizgfx::RectangleShape box(vizgfx::Vector2f(static_cast<float>(panel_w), 20));
        box.setPosition(vizgfx::Vector2f(metrics_x, metrics_y + y_offset));
        box.setFillColor(vizgfx::Color(38,43,52));
        window->draw(box);
        vizgfx::RectangleShape marker({3.f,20.f});marker.setPosition(box.getPosition());marker.setFillColor(color);window->draw(marker);
    };

    // Epoch (bleu)
    drawMetricBox(0, vizgfx::Color(50, 100, 180, 200));

    // Batch (vert)
    drawMetricBox(line_height, vizgfx::Color(50, 150, 80, 200));

    // Loss (orange/rouge selon valeur)
    vizgfx::Color loss_color = current_loss > 100.0f ? vizgfx::Color(200, 80, 50, 200) : vizgfx::Color(180, 140, 50, 200);
    drawMetricBox(line_height * 2, loss_color);

    // Learning rate (violet)
    drawMetricBox(line_height * 3, vizgfx::Color(140, 80, 180, 200));



    // Texte best-effort
    if (font_loaded) {
        auto wrap_lines = [](const std::string& s, size_t max_chars, int max_lines) {
            std::vector<std::string> out;
            if (max_chars == 0) return out;
            const bool limit_lines = (max_lines > 0);
            std::string cur;
            cur.reserve(max_chars);
            size_t i = 0;
            while (i < s.size() && (!limit_lines || static_cast<int>(out.size()) < max_lines)) {
                // skip leading spaces
                while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
                if (i >= s.size()) break;
                cur.clear();
                while (i < s.size() && cur.size() < max_chars) {
                    const char c = s[i++];
                    if (c == '\n' || c == '\r') break;
                    cur.push_back(c);
                }
                if (!cur.empty()) out.push_back(cur);
                // skip until newline if we broke early
                while (i < s.size() && s[i] != '\n' && s[i] != '\r') {
                    if (cur.size() >= max_chars) break;
                    ++i;
                }
                while (i < s.size() && (s[i] == '\n' || s[i] == '\r')) ++i;
            }
            return out;
        };

        auto drawText = [&](int y_offset, const std::string& s) {
            vizgfx::Text t(font);
            t.setFont(font);
            t.setCharacterSize(14);
            t.setFillColor(vizgfx::Color(230, 230, 235));
            t.setPosition(vizgfx::Vector2f(static_cast<float>(metrics_x + 6), static_cast<float>(metrics_y + y_offset - 2)));
            t.setString(vizgfx::String::fromUtf8(s.begin(), s.end()));
            window->draw(t);
        };

        auto drawSection = [&](int y_offset, const std::string& title, vizgfx::Color color) {
            vizgfx::RectangleShape bar(vizgfx::Vector2f(static_cast<float>(panel_w), 18.f));
            bar.setPosition(vizgfx::Vector2f(static_cast<float>(metrics_x), static_cast<float>(metrics_y + y_offset)));
            bar.setFillColor(vizgfx::Color(color.r, color.g, color.b, 75));
            window->draw(bar);
            vizgfx::Text t(font);
            t.setCharacterSize(11);
            t.setStyle(vizgfx::Text::Bold);
            t.setFillColor(color);
            t.setPosition(vizgfx::Vector2f(static_cast<float>(metrics_x + 6), static_cast<float>(metrics_y + y_offset + 1)));
            t.setString(title);
            window->draw(t);
        };

        auto drawTag = [&](float x, float y, const std::string& label, vizgfx::Color color) {
            const float w = std::max(48.f, 8.f * static_cast<float>(label.size()) + 12.f);
            vizgfx::RectangleShape box(vizgfx::Vector2f(w, 18.f));
            box.setPosition(vizgfx::Vector2f(x, y));
            box.setFillColor(vizgfx::Color(color.r, color.g, color.b, 95));
            box.setOutlineColor(vizgfx::Color(color.r, color.g, color.b, 220));
            box.setOutlineThickness(1.f);
            window->draw(box);
            vizgfx::Text t(font);
            t.setCharacterSize(11);
            t.setFillColor(vizgfx::Color(245, 245, 248));
            t.setPosition(vizgfx::Vector2f(x + 6.f, y + 1.f));
            t.setString(label);
            window->draw(t);
        };

        auto drawTableRow = [&](int y_offset, const std::string& label,
                                const std::string& value, vizgfx::Color accent) {
            vizgfx::RectangleShape row(vizgfx::Vector2f(static_cast<float>(panel_w), 21.f));
            row.setPosition(vizgfx::Vector2f(static_cast<float>(metrics_x),
                                         static_cast<float>(metrics_y + y_offset - 2)));
            row.setFillColor(vizgfx::Color(48, 48, 48));
            window->draw(row);
            vizgfx::RectangleShape mark(vizgfx::Vector2f(3.f, 21.f));
            mark.setPosition(row.getPosition());
            mark.setFillColor(accent);
            window->draw(mark);
            vizgfx::Text key(font);
            key.setCharacterSize(12);
            key.setStyle(vizgfx::Text::Bold);
            key.setFillColor(accent);
            key.setPosition(vizgfx::Vector2f(static_cast<float>(metrics_x + 8),
                                         static_cast<float>(metrics_y + y_offset)));
            key.setString(label);
            window->draw(key);
            vizgfx::Text val(font);
            val.setCharacterSize(12);
            val.setFillColor(vizgfx::Color(228, 231, 238));
            val.setPosition(vizgfx::Vector2f(static_cast<float>(metrics_x + 112),
                                         static_cast<float>(metrics_y + y_offset)));
            val.setString(value);
            window->draw(val);
        };

        drawText(0, "Epoch " + std::to_string(current_epoch) + "/" + std::to_string(std::max(1, current_total_epochs)));
        drawText(line_height, "Batch " + std::to_string(current_batch) + "/" + std::to_string(std::max(1, current_total_batches)));
        {
            if (current_detection_metrics_.enabled) {
                drawText(line_height * 2,
                         "Loss=" + format_decimal(current_loss, 8) +
                             " avg=" + format_decimal(current_avg_loss, 8));
            } else {
                const std::string recon_name = canonical_recon_loss_name(current_recon_loss_type);
                const float recon_normalized = normalized_recon_loss(
                    current_mse, current_recon_loss_type);
                drawText(line_height * 2,
                         "Loss=" + format_decimal(current_loss, 8) +
                             " avg=" + format_decimal(current_avg_loss, 8) +
                             " " + recon_name + "_norm=" + format_decimal(recon_normalized, 8) +
                             " ent=" + format_decimal(current_ent, 8));
            }
        }
        { drawText(line_height * 3, "LR=" + format_decimal(current_lr, 10)); }

        // -------------------------
        // Live tuning controls
        // -------------------------
        const uint64_t external_version =
            external_live_params_version_.load(std::memory_order_relaxed);
        if (external_version != live_ui_external_version_) {
            live_ui_lr_ = std::clamp(live_lr_.load(std::memory_order_relaxed), kLiveLRMin, kLiveLRMax);
            live_ui_lr_warmup_steps_ = live_lr_warmup_steps_.load(std::memory_order_relaxed);
            live_ui_kl_beta_ = std::clamp(live_kl_beta_.load(std::memory_order_relaxed), 0.0f, 1.0f);
            live_ui_kl_warmup_steps_ = live_kl_warmup_steps_.load(std::memory_order_relaxed);
            live_ui_kl_enabled_ = live_kl_enabled_.load(std::memory_order_relaxed);
            live_ui_recon_loss_index_ = live_recon_loss_index_.load(std::memory_order_relaxed);
            live_ui_external_version_ = external_version;
            live_ui_inited_ = true;
        }
        if (!live_ui_inited_) {
            live_ui_lr_ = std::clamp(runtime_lr_.load(std::memory_order_relaxed), kLiveLRMin, kLiveLRMax);
            live_ui_lr_warmup_steps_ = runtime_lr_warmup_steps_.load(std::memory_order_relaxed);
            live_ui_kl_beta_ = std::clamp(runtime_kl_beta_.load(std::memory_order_relaxed), 0.0f, 1.0f);
            live_ui_kl_warmup_steps_ = runtime_kl_warmup_steps_.load(std::memory_order_relaxed);
            live_ui_kl_enabled_ = (live_ui_kl_beta_ > 0.0f);
            live_ui_recon_loss_index_ = runtime_recon_loss_index_.load(std::memory_order_relaxed);

            // Démarre en mode NATIVE: aucun override tant que l'utilisateur n'interagit pas.
            live_overrides_enabled_.store(false, std::memory_order_relaxed);
            live_lr_.store(live_ui_lr_, std::memory_order_relaxed);
            live_lr_warmup_steps_.store(live_ui_lr_warmup_steps_, std::memory_order_relaxed);
            live_kl_beta_.store(live_ui_kl_beta_, std::memory_order_relaxed);
            live_kl_warmup_steps_.store(live_ui_kl_warmup_steps_, std::memory_order_relaxed);
            live_kl_enabled_.store(live_ui_kl_enabled_, std::memory_order_relaxed);
            live_recon_loss_index_.store(live_ui_recon_loss_index_, std::memory_order_relaxed);
            // Ne pas bump la version: on ne veut pas changer le training sans interaction.
            live_ui_inited_ = true;
        }

        // PS (user): MAJ en temps réel des scrollers (sliders) avec les valeurs runtime associées.
        // En mode NATIVE, refléter current_lr/current_kl_beta_effective tant qu'on ne drag pas.
        const bool live_on_rt = live_overrides_enabled_.load(std::memory_order_relaxed);
        if (!live_on_rt && live_dragging_ == LiveDragTarget::None &&
            live_input_target_ == LiveInputTarget::None) {
            live_ui_lr_ = std::clamp(runtime_lr_.load(std::memory_order_relaxed), kLiveLRMin, kLiveLRMax);
            live_ui_lr_warmup_steps_ = runtime_lr_warmup_steps_.load(std::memory_order_relaxed);
            live_ui_kl_beta_ = std::clamp(runtime_kl_beta_.load(std::memory_order_relaxed), 0.0f, 1.0f);
            live_ui_kl_warmup_steps_ = runtime_kl_warmup_steps_.load(std::memory_order_relaxed);
            live_ui_kl_enabled_ = (live_ui_kl_beta_ > 0.0f);
            live_ui_recon_loss_index_ = runtime_recon_loss_index_.load(std::memory_order_relaxed);
        }

        const int live_y0 = line_height * 4 + 30;
        const int live_row_h = 42;
        const int label_w = 82;
        const int value_w = 110;
        const float track_h = 10.f;
        const float thumb_w = 10.f;
        const float thumb_h = 16.f;
        const int rows = 4;

        auto drawSliderRow = [&](int row,
                                 const std::string& label,
                                 float t,
                                 const std::string& value,
                                 LiveInputTarget input_target,
                                 const std::string& min_label,
                                 const std::string& max_label,
                                 std::optional<vizgfx::FloatRect>& out_track,
                                 std::optional<vizgfx::FloatRect>& out_thumb,
                                 std::optional<vizgfx::FloatRect>& out_value_box) {
            const float y = static_cast<float>(metrics_y + live_y0 + row * live_row_h);
            const float x0 = static_cast<float>(metrics_x + 6);
            const float x_track = static_cast<float>(metrics_x + label_w);
            const float x_value = static_cast<float>(metrics_x + panel_w - value_w);
            const float w_track = std::max(20.f, x_value - x_track - 10.f);

            // Label
            {
                vizgfx::Text tt(font);
                tt.setFont(font);
                tt.setCharacterSize(13);
                tt.setFillColor(vizgfx::Color(225, 225, 235));
                tt.setPosition(vizgfx::Vector2f(x0, y - 2.f));
                tt.setString(vizgfx::String::fromUtf8(label.begin(), label.end()));
                window->draw(tt);
            }

            // Track
            const float tt = clamp01(t);
            const float track_y = y + 7.f;
            vizgfx::RectangleShape track(vizgfx::Vector2f(w_track, track_h));
            track.setPosition(vizgfx::Vector2f(x_track, track_y));
            track.setFillColor(vizgfx::Color(55, 55, 60, 210));
            track.setOutlineColor(vizgfx::Color(90, 90, 95, 220));
            track.setOutlineThickness(1);
            window->draw(track);

            vizgfx::RectangleShape fill(vizgfx::Vector2f(w_track * tt, track_h));
            fill.setPosition(vizgfx::Vector2f(x_track, track_y));
            fill.setFillColor(vizgfx::Color(90, 140, 200, 220));
            window->draw(fill);

            // Graduation: 5 repères visibles pour guider le réglage.
            for (int gi = 0; gi <= 4; ++gi) {
                const float gt = static_cast<float>(gi) / 4.0f;
                const float gx = x_track + gt * w_track;
                vizgfx::RectangleShape tick(vizgfx::Vector2f(1.0f, 5.0f));
                tick.setPosition(vizgfx::Vector2f(gx, track_y + track_h + 1.0f));
                tick.setFillColor(vizgfx::Color(145, 145, 160, 210));
                window->draw(tick);
            }

            if (font_loaded) {
                vizgfx::Text tmin(font);
                tmin.setFont(font);
                tmin.setCharacterSize(10);
                tmin.setFillColor(vizgfx::Color(178, 178, 190));
                tmin.setPosition(vizgfx::Vector2f(x_track, track_y + track_h + 7.0f));
                tmin.setString(vizgfx::String::fromUtf8(min_label.begin(), min_label.end()));
                window->draw(tmin);

                vizgfx::Text tmax(font);
                tmax.setFont(font);
                tmax.setCharacterSize(10);
                tmax.setFillColor(vizgfx::Color(178, 178, 190));
                tmax.setPosition(vizgfx::Vector2f(x_track + w_track - 8.0f * static_cast<float>(max_label.size()), track_y + track_h + 7.0f));
                tmax.setString(vizgfx::String::fromUtf8(max_label.begin(), max_label.end()));
                window->draw(tmax);
            }

            const float thumb_x = x_track + w_track * tt - thumb_w * 0.5f;
            vizgfx::RectangleShape thumb(vizgfx::Vector2f(thumb_w, thumb_h));
            thumb.setPosition(vizgfx::Vector2f(thumb_x, track_y + track_h * 0.5f - thumb_h * 0.5f));
            thumb.setFillColor(vizgfx::Color(210, 210, 220, 235));
            thumb.setOutlineColor(vizgfx::Color(30, 30, 30, 220));
            thumb.setOutlineThickness(1);
            window->draw(thumb);

            out_track = make_rect(x_track, track_y, w_track, track_h);
            out_thumb = make_rect(thumb.getPosition().x, thumb.getPosition().y, thumb_w, thumb_h);

            // Cellule de saisie valeur (clic pour éditer)
            {
                const bool editing = (live_input_target_ == input_target);
                const bool invalid = editing && (live_input_error_until_ms_ > steady_now_ms());
                const float vb_h = 18.0f;
                const float vb_y = y + 1.0f;

                vizgfx::RectangleShape vb(vizgfx::Vector2f(static_cast<float>(value_w), vb_h));
                vb.setPosition(vizgfx::Vector2f(x_value, vb_y));
                vb.setFillColor(editing ? vizgfx::Color(44, 58, 74, 225) : vizgfx::Color(34, 36, 44, 215));
                vb.setOutlineColor(invalid ? vizgfx::Color(230, 110, 110, 235) : vizgfx::Color(98, 104, 120, 225));
                vb.setOutlineThickness(1.0f);
                window->draw(vb);

                out_value_box = make_rect(x_value, vb_y, static_cast<float>(value_w), vb_h);

                vizgfx::Text tt(font);
                tt.setFont(font);
                tt.setCharacterSize(13);
                tt.setFillColor(vizgfx::Color(225, 225, 235));
                tt.setPosition(vizgfx::Vector2f(x_value + 4.0f, y - 1.f));

                std::string shown = editing ? live_input_buffer_ : value;
                if (editing) shown += "_";
                tt.setString(vizgfx::String::fromUtf8(shown.begin(), shown.end()));
                window->draw(tt);
            }
        };

        drawCommand(4, live_on_rt ? "Réglages : manuels" : "Réglages : auto",
                    live_on_rt ? vizgfx::Color(35,115,75) : vizgfx::Color(65,75,85), last_live_overrides_box_);
        drawCommand(5, live_ui_kl_enabled_ ? "KL : oui" : "KL : non",
                    live_ui_kl_enabled_ ? vizgfx::Color(35,115,75) : vizgfx::Color(65,75,85), last_live_kl_enable_box_);
        drawCommand(6, std::string("Perte : ")+recon_loss_name(live_ui_recon_loss_index_),
                    vizgfx::Color(80,70,125), last_live_recon_loss_box_);
        drawSection(line_height*4+6, "Reglages", vizgfx::Color(150,190,235));

        // LR (log scale)
        {
            drawSliderRow(0,
                          "LR",
                          t_from_lr(live_ui_lr_),
                          format_decimal(live_ui_lr_, 10),
                          LiveInputTarget::LR,
                          "1e-7",
                          "1e-3",
                          last_live_lr_track_,
                          last_live_lr_thumb_,
                          last_live_lr_value_box_);
        }

        // LR warmup
        {
            const int warmup_total = std::max(0, live_ui_lr_warmup_steps_);
            const int warmup_done = std::min(std::max(0, current_opt_step), warmup_total);
            const bool native_progress = !live_on_rt && warmup_total > 0;
            drawSliderRow(1,
                          "LR wu",
                          native_progress
                              ? static_cast<float>(warmup_done) / static_cast<float>(warmup_total)
                              : t_from_warmup(live_ui_lr_warmup_steps_),
                          native_progress
                              ? std::to_string(warmup_done) + "/" + std::to_string(warmup_total)
                              : std::to_string(live_ui_lr_warmup_steps_) + " steps",
                          LiveInputTarget::LRWarmup,
                          "0",
                          "5e4",
                          last_live_lrwu_track_,
                          last_live_lrwu_thumb_,
                          last_live_lrwu_value_box_);
        }

        // KL beta
        {
            drawSliderRow(2,
                          "KL beta",
                          t_from_klbeta(live_ui_kl_beta_),
                          format_decimal(live_ui_kl_beta_, 8),
                          LiveInputTarget::KLBeta,
                          "0",
                          "1",
                          last_live_klb_track_,
                          last_live_klb_thumb_,
                          last_live_klb_value_box_);
        }

        // KL warmup
        {
            const int warmup_total = std::max(0, live_ui_kl_warmup_steps_);
            const int warmup_done = std::min(std::max(0, current_opt_step), warmup_total);
            const bool native_progress = !live_on_rt && warmup_total > 0;
            drawSliderRow(3,
                          "KL wu",
                          native_progress
                              ? static_cast<float>(warmup_done) / static_cast<float>(warmup_total)
                              : t_from_warmup(live_ui_kl_warmup_steps_),
                          native_progress
                              ? std::to_string(warmup_done) + "/" + std::to_string(warmup_total)
                              : std::to_string(live_ui_kl_warmup_steps_) + " steps",
                          LiveInputTarget::KLWarmup,
                          "0",
                          "5e4",
                          last_live_klwu_track_,
                          last_live_klwu_thumb_,
                          last_live_klwu_value_box_);
        }

        int extra_y = live_y0 + rows * live_row_h + 26;
        if (current_detection_metrics_.enabled) {
            drawSection(extra_y - 22, "DÉTECTION", vizgfx::Color(235, 135, 90));
            drawTableRow(extra_y, "LOSSES",
                "obj=" + format_decimal(current_detection_metrics_.object_loss, 8) +
                "  box=" + format_decimal(current_detection_metrics_.box_loss, 8) +
                "  cls=" + format_decimal(current_detection_metrics_.class_loss, 8),
                vizgfx::Color(235, 155, 90));
            drawTableRow(extra_y + line_height, "TARGETS",
                "positive_cells=" + std::to_string(current_detection_metrics_.positive_cells),
                vizgfx::Color(120, 205, 170));
            extra_y += 2 * line_height + 24;
        }
        drawSection(extra_y - 22, "RUNTIME / OPTIMIZER", vizgfx::Color(180, 145, 235));
        int after_validation_y = extra_y + 8 * line_height;
        metrics_content_height = command_height+28.f+static_cast<float>(after_validation_y + 20);
        {
            const std::string recon_name = canonical_recon_loss_name(current_recon_loss_type);
            std::string s = "t=" + format_decimal(current_timestep, 8) +
                "  " + recon_name + "=" + format_decimal(current_mse, 8) +
                "  norm=" + format_decimal(
                    normalized_recon_loss(current_mse, current_recon_loss_type), 8);
            drawTableRow(extra_y, "SIGNAL", s, vizgfx::Color(105, 190, 230));
        }

        if (current_kl_beta_effective > 0.0f) {
            drawTableRow(extra_y + line_height, "KL", "beta_eff=" + format_decimal(current_kl_beta_effective, 8) + "  kl=" + format_decimal(current_kl, 8), vizgfx::Color(205, 125, 230));
            extra_y += line_height;
        }
        drawTableRow(extra_y + line_height, "GRAD", "norm=" + format_decimal(current_grad_norm, 8) + "  max=" + format_decimal(current_grad_max, 8), vizgfx::Color(235, 155, 80));
        drawTableRow(extra_y + 2 * line_height, "PERF", std::to_string(current_batch_time_ms) + "ms  bps=" + format_decimal(current_bps, 8), vizgfx::Color(90, 205, 155));
        drawTableRow(extra_y + 3 * line_height, "MEM", std::to_string(current_memory_mb) + "MB RSS  alloc=" + std::to_string(current_allocator_memory_mb) + "MB  params=" + std::to_string(current_params), vizgfx::Color(105, 165, 235));
        drawTableRow(extra_y + 4 * line_height, "OPT", std::string(optimizerTypeName(static_cast<OptimizerType>(current_opt_type))) + "  step=" + std::to_string(current_opt_step) + "  wd=" + format_decimal(current_opt_weight_decay, 10), vizgfx::Color(215, 180, 80));

        {
            const int validation_header_y = extra_y + 5 * line_height;
            const int validation_status_y = validation_header_y + 22;
            const int validation_result_y = validation_status_y + line_height;
            std::string s;
            if (current_val_in_progress) {
                s = "Val: EN COURS";
                if (current_val_step > 0) s += " @" + std::to_string(current_val_step);
                if (current_val_total > 0) {
                    s += " " + std::to_string(std::clamp(current_val_done, 0, current_val_total)) + "/" + std::to_string(current_val_total);
                } else if (current_val_done > 0) {
                    s += " done=" + std::to_string(current_val_done);
                }
            } else if (!current_val_has) {
                s = "Val: —";
            } else {
                s = "Val@" + std::to_string(current_val_step) + " " + (current_val_ok ? "OK" : "FAIL") + " items=" + std::to_string(current_val_items);
            }
            drawSection(validation_header_y, "VALIDATION", vizgfx::Color(100, 190, 235));
            drawTableRow(validation_status_y, "STATUS", s, vizgfx::Color(100, 190, 235));

            const auto feedback_icon = static_cast<ValidationFeedbackIcon>(
                validation_feedback_icon_.load(std::memory_order_relaxed));
            vizgfx::Sprite* feedback_sprite = nullptr;
            if (feedback_icon == ValidationFeedbackIcon::Reward && validation_reward_icon_loaded_) {
                feedback_sprite = &validation_reward_sprite_;
            } else if (feedback_icon == ValidationFeedbackIcon::Penalty && validation_penalty_icon_loaded_) {
                feedback_sprite = &validation_penalty_sprite_;
            }
            if (feedback_sprite && current_val_has && !current_val_in_progress) {
                const auto bounds = feedback_sprite->getLocalBounds();
                const float icon_size = 20.0f;
                feedback_sprite->setScale(vizgfx::Vector2f(
                    icon_size / std::max(1.0f, bounds.size.x),
                    icon_size / std::max(1.0f, bounds.size.y)));
                feedback_sprite->setPosition(vizgfx::Vector2f(
                    static_cast<float>(metrics_x + panel_w - 30),
                    static_cast<float>(metrics_y + validation_status_y - 2)));
                window->draw(*feedback_sprite);
            }

            if (current_val_has && !current_val_in_progress && !current_val_feedback.empty()) {
                vizgfx::Color tag_color(145, 150, 165);
                if (current_val_feedback == "reward") tag_color = vizgfx::Color(65, 205, 115);
                else if (current_val_feedback == "penalty") tag_color = vizgfx::Color(235, 85, 85);
                else if (current_val_feedback == "plateau") tag_color = vizgfx::Color(225, 175, 65);
                else if (current_val_feedback == "kl_collapse" || current_val_feedback == "kl_excess") tag_color = vizgfx::Color(205, 105, 225);
                drawTag(static_cast<float>(metrics_x + std::max(120, panel_w - 130)),
                        static_cast<float>(metrics_y + validation_status_y),
                        current_val_feedback, tag_color);
            }

            // Afficher les métriques de validation dès qu'on en a (même partiellement).
            if (current_val_has || current_val_in_progress) {
                std::ostringstream ss;
                if (current_detection_metrics_.enabled &&
                    current_detection_metrics_.validation_has) {
                    ss << "P=" << format_decimal(current_detection_metrics_.precision * 100.0f, 2)
                       << "%  R=" << format_decimal(current_detection_metrics_.recall * 100.0f, 2)
                       << "%  F1=" << format_decimal(current_detection_metrics_.f1 * 100.0f, 2)
                       << "%  IoU=" << format_decimal(current_detection_metrics_.mean_iou * 100.0f, 2)
                       << "%";
                } else {
                    ss << "val recon=" << format_decimal(current_val_recon, 8)
                       << " kl=" << format_decimal(current_val_kl, 8);
                    if (std::fabs(current_val_align) > 1e-9f) {
                        ss << " align=" << format_decimal(current_val_align, 8);
                    }
                }
                drawTableRow(validation_result_y, "RESULT", ss.str(), vizgfx::Color(115, 210, 160));
                if (current_detection_metrics_.enabled &&
                    current_detection_metrics_.validation_has) {
                    drawTableRow(validation_result_y + line_height, "MATCHES",
                        "TP=" + std::to_string(current_detection_metrics_.true_positive) +
                        "  FP=" + std::to_string(current_detection_metrics_.false_positive) +
                        "  FN=" + std::to_string(current_detection_metrics_.false_negative),
                        vizgfx::Color(120, 190, 235));
                    after_validation_y += line_height;
                    metrics_content_height = command_height+28.f+static_cast<float>(after_validation_y + 20);
                }
            }
        }

        // Texte dataset : cartes indépendantes avec clipping et scroll interne.
        if (show_prompt_text_ && has_dataset_text) {
            int y = after_validation_y;
            const size_t wrap_chars = static_cast<size_t>(std::max(30, panel_w / 8));
            drawSection(y, "PROMPT / TOKENIZER / ENCODER", vizgfx::Color(100, 205, 190));
            y += 24;

            struct TextCard {
                const char* title;
                const std::string* value;
                vizgfx::Color color;
                DatasetTextSection section;
            };
            const TextCard cards[] = {
                {"PROMPT",  &dataset_text_raw,     vizgfx::Color(100, 190, 235), DatasetTextSection::Prompt},
                {"TAGS",    &dataset_text_tags,    vizgfx::Color(100, 215, 145), DatasetTextSection::Tags},
                {"TOKENS",  &dataset_text_tokens,  vizgfx::Color(225, 175, 70),  DatasetTextSection::Tokens},
                {"ENCODER", &dataset_text_encoded, vizgfx::Color(190, 125, 230), DatasetTextSection::Encoder},
            };
            const int card_h = 104;
            const int title_h = 22;
            const int inner_pad = 7;
            const int text_line_h = 18;

            for (const auto& card : cards) {
                if (!card.value || card.value->empty()) continue;
                const size_t index = static_cast<size_t>(card.section);
                const float card_x = static_cast<float>(metrics_x);
                const float card_y = static_cast<float>(metrics_y + y);
                const float card_w = static_cast<float>(panel_w);

                vizgfx::RectangleShape background(vizgfx::Vector2f(card_w, static_cast<float>(card_h)));
                background.setPosition(vizgfx::Vector2f(card_x, card_y));
                background.setFillColor(vizgfx::Color(24, 28, 36, 220));
                background.setOutlineColor(vizgfx::Color(card.color.r, card.color.g, card.color.b, 165));
                background.setOutlineThickness(1.f);
                window->draw(background);

                vizgfx::RectangleShape title_bg(vizgfx::Vector2f(card_w, static_cast<float>(title_h)));
                title_bg.setPosition(vizgfx::Vector2f(card_x, card_y));
                title_bg.setFillColor(vizgfx::Color(card.color.r, card.color.g, card.color.b, 72));
                window->draw(title_bg);
                vizgfx::Text title(font);
                title.setCharacterSize(11);
                title.setStyle(vizgfx::Text::Bold);
                title.setFillColor(card.color);
                title.setPosition(vizgfx::Vector2f(card_x + 7.f, card_y + 2.f));
                title.setString(card.title);
                window->draw(title);

                const auto lines = wrap_lines(*card.value, wrap_chars, 0);
                const float content_h = static_cast<float>(lines.size() * text_line_h + 2 * inner_pad);
                const float viewport_h = static_cast<float>(card_h - title_h);
                dataset_text_scroll_max_[index] = std::max(0.f, content_h - viewport_h);
                dataset_text_scroll_y_[index] = std::clamp(
                    dataset_text_scroll_y_[index], 0.f, dataset_text_scroll_max_[index]);
                dataset_text_section_rects_[index] = make_rect(
                    card_x, card_y + static_cast<float>(title_h), card_w, viewport_h);

                // View dédiée : seul le corps de cette carte est clippé.
                const vizgfx::View metrics_view = window->getView();
                const float ww = static_cast<float>(std::max(1, window_width));
                const float hh = static_cast<float>(std::max(1, window_height));
                const float clip_top = std::max(area.position.y, card_y + static_cast<float>(title_h));
                const float clip_bottom = std::min(area.position.y + area.size.y,
                                                   card_y + static_cast<float>(card_h));
                if (clip_bottom > clip_top) {
                    vizgfx::View clip_view(make_rect(card_x, clip_top, card_w, clip_bottom - clip_top));
                    clip_view.setViewport(make_rect(card_x / ww, clip_top / hh,
                                                    card_w / ww, (clip_bottom - clip_top) / hh));
                    window->setView(clip_view);
                    float text_y = card_y + static_cast<float>(title_h + inner_pad) -
                                   dataset_text_scroll_y_[index];
                    for (const auto& line : lines) {
                        vizgfx::Text text(font);
                        text.setCharacterSize(12);
                        text.setFillColor(vizgfx::Color(225, 229, 237));
                        text.setPosition(vizgfx::Vector2f(card_x + inner_pad, text_y));
                        text.setString(vizgfx::String::fromUtf8(line.begin(), line.end()));
                        window->draw(text);
                        text_y += static_cast<float>(text_line_h);
                    }
                    window->setView(metrics_view);
                }

                // Scrollbar interne toujours visible si le contenu déborde.
                if (dataset_text_scroll_max_[index] > 0.f) {
                    const float track_x = card_x + card_w - 5.f;
                    const float track_y = card_y + static_cast<float>(title_h + 2);
                    const float track_h = static_cast<float>(card_h - title_h - 4);
                    vizgfx::RectangleShape track(vizgfx::Vector2f(3.f, track_h));
                    track.setPosition(vizgfx::Vector2f(track_x, track_y));
                    track.setFillColor(vizgfx::Color(65, 70, 82, 210));
                    window->draw(track);
                    const float ratio = viewport_h / std::max(viewport_h, content_h);
                    const float thumb_h = std::max(14.f, track_h * ratio);
                    const float progress = dataset_text_scroll_y_[index] /
                                           dataset_text_scroll_max_[index];
                    vizgfx::RectangleShape thumb(vizgfx::Vector2f(3.f, thumb_h));
                    thumb.setPosition(vizgfx::Vector2f(track_x,
                        track_y + progress * (track_h - thumb_h)));
                    thumb.setFillColor(card.color);
                    window->draw(thumb);
                }
                y += card_h + 8;
            }
            metrics_content_height = std::max(
                metrics_content_height, command_height+28.f+static_cast<float>(y + line_height));
        }
    }
    setPanelContentHeight(PanelId::Metrics, metrics_content_height);
    window->setView(old_view);
}

void Visualizer::createImageTexture(ImageData& img_data, int w, int h, int channels, int display_size) {
    // Créer une image graphique à partir des pixels (grayscale/RGB/RGBA).
    if (w <= 0 || h <= 0) return;
    if (channels != 1 && channels != 3 && channels != 4) return;

    if (window && window->isOpen()) {
        // Upload texture garanti avec contexte actif (évite textures noires selon WM/driver).
        window->setActive(true);
    }

    img_data.w = w;
    img_data.h = h;
    img_data.channels = channels;
    img_data.display_size = display_size;

    vizgfx::Image sfml_image;
    sfml_image.resize(vizgfx::Vector2u(static_cast<unsigned>(w), static_cast<unsigned>(h)));

    // Si la taille ne colle pas exactement aux canaux annoncés, tenter une correction
    // best-effort pour éviter des previews noires au premier rendu.
    const size_t area = static_cast<size_t>(w) * static_cast<size_t>(h);
    const size_t declared_stride = static_cast<size_t>(channels);
    const size_t expected_declared = area * declared_stride;
    if (img_data.pixels.size() != expected_declared) {
        const size_t n = img_data.pixels.size();
        if (n == area * 4ULL) channels = 4;
        else if (n == area * 3ULL) channels = 3;
        else if (n == area) channels = 1;
        img_data.channels = channels;
    }

    const size_t stride = static_cast<size_t>(channels);
    const size_t expected = area * stride;
    const size_t n = std::min(expected, img_data.pixels.size());

    auto at = [&](size_t idx) -> uint8_t {
        return (idx < n) ? img_data.pixels[idx] : 0;
    };

    bool force_opaque_alpha = false;
    if (channels == 4) {
        bool any_alpha = false;
        for (size_t i = 3; i < n; i += 4) {
            if (img_data.pixels[i] != 0) {
                any_alpha = true;
                break;
            }
        }
        force_opaque_alpha = !any_alpha;
    }

    for (int yy = 0; yy < h; ++yy) {
        for (int xx = 0; xx < w; ++xx) {
            const size_t base = (static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(xx)) * stride;
            vizgfx::Color c;
            if (channels == 1) {
                const uint8_t g = at(base);
                c = vizgfx::Color(g, g, g, 255);
            } else if (channels == 3) {
                c = vizgfx::Color(at(base + 0), at(base + 1), at(base + 2), 255);
            } else {
                c = vizgfx::Color(at(base + 0), at(base + 1), at(base + 2), force_opaque_alpha ? 255 : at(base + 3));
            }
            sfml_image.setPixel(vizgfx::Vector2u(static_cast<unsigned int>(static_cast<unsigned>(xx)), static_cast<unsigned int>(static_cast<unsigned>(yy))), c);
        }
    }

    bool loaded = img_data.texture.loadFromImage(sfml_image);
    if (!loaded) {
        // Fallback: downscale agressif puis retry (utile si la texture dépasse les limites GPU/driver).
        const unsigned max_tex = vizgfx::Texture::getMaximumSize();
        const unsigned src_w = static_cast<unsigned>(std::max(1, w));
        const unsigned src_h = static_cast<unsigned>(std::max(1, h));

        const unsigned hard_cap = std::max(64u, std::min(1024u, max_tex > 0 ? max_tex : 1024u));
        const unsigned dw = std::max(1u, std::min(src_w, hard_cap));
        const unsigned dh = std::max(1u, std::min(src_h, hard_cap));

        if (dw != src_w || dh != src_h) {
            vizgfx::Image down;
            down.resize(vizgfx::Vector2u(dw, dh));
            for (unsigned yy = 0; yy < dh; ++yy) {
                const unsigned sy = (yy * src_h) / dh;
                for (unsigned xx = 0; xx < dw; ++xx) {
                    const unsigned sx = (xx * src_w) / dw;
                    down.setPixel(vizgfx::Vector2u(static_cast<unsigned int>(xx), static_cast<unsigned int>(yy)), sfml_image.getPixel(vizgfx::Vector2u(static_cast<unsigned int>(sx), static_cast<unsigned int>(sy))));
                }
            }
            loaded = img_data.texture.loadFromImage(down);
        }
    }

    if (!loaded) {
        return;
    }

    img_data.texture.setSmooth(true);
    img_data.sprite.setTexture(img_data.texture, true);

    // Mettre à l'échelle pour affichage (fit-to-square)
    const float sx = static_cast<float>(display_size) / static_cast<float>(w);
    const float sy = static_cast<float>(display_size) / static_cast<float>(h);
    const float scale = std::min(sx, sy);
    img_data.sprite.setScale(vizgfx::Vector2f(scale, scale));
}

vizgfx::Color Visualizer::getLossColor(float loss) {
    // Gradient vert -> jaune -> rouge basé sur la loss
    if (loss < 50.0f) {
        return vizgfx::Color(100, 200, 100); // Vert
    } else if (loss < 150.0f) {
        return vizgfx::Color(200, 200, 100); // Jaune
    } else {
        return vizgfx::Color(200, 100, 100); // Rouge
    }
}

void Visualizer::saveLossHistory(const std::string& filepath) const {
    if (!MetricsCsv::write(filepath, full_loss_history))
        std::cerr << "Erreur: impossible d'ouvrir " << filepath << " pour écriture" << std::endl;
}
