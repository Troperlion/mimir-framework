#include "PonyXLDDPMModel.hpp"

#include "Models/Registry/ModelArchitectures.hpp"
#include "Serialization/Serialization.hpp"

#include "Helpers.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <random>
#include <unordered_map>

namespace {
namespace fs = std::filesystem;

static fs::path resolve_latest_epoch_dir(const fs::path& base);

static std::string truncate_utf8_safe(const std::string& s, size_t max_bytes) {
    if (s.size() <= max_bytes) return s;
    if (max_bytes == 0) return std::string();

    // Keep a prefix, then rewind to a valid UTF-8 boundary.
    size_t end = max_bytes;
    // Rewind continuation bytes.
    while (end > 0) {
        const unsigned char c = static_cast<unsigned char>(s[end - 1]);
        if ((c & 0xC0) != 0x80) break;
        --end;
    }
    if (end == 0) return std::string();

    // Validate the last codepoint length.
    const unsigned char lead = static_cast<unsigned char>(s[end - 1]);
    int len = 1;
    if ((lead & 0x80) == 0x00) len = 1;
    else if ((lead & 0xE0) == 0xC0) len = 2;
    else if ((lead & 0xF0) == 0xE0) len = 3;
    else if ((lead & 0xF8) == 0xF0) len = 4;
    else len = 1;
    if (len > 1) {
        // If the prefix ends mid-codepoint, drop the lead.
        if (end + static_cast<size_t>(len - 1) > max_bytes) {
            --end;
        }
    }
    return s.substr(0, end);
}

static std::string strip_trailing_slashes(std::string s) {
    while (!s.empty() && (s.back() == '/' || s.back() == '\\')) {
        s.pop_back();
    }
    return s;
}

static inline std::string trim_ws(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

static inline bool starts_with(const std::string& s, const char* lit) {
    if (!lit) return false;
    const size_t n = std::char_traits<char>::length(lit);
    if (s.size() < n) return false;
    return std::equal(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(n), lit);
}

static inline bool ends_with(const std::string& s, const char* lit) {
    if (!lit) return false;
    const size_t n = std::char_traits<char>::length(lit);
    if (s.size() < n) return false;
    return std::equal(s.end() - static_cast<std::ptrdiff_t>(n), s.end(), lit);
}

static inline std::string to_upper_ascii(std::string s) {
    for (char& ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (c >= 'a' && c <= 'z') ch = static_cast<char>(c - ('a' - 'A'));
    }
    return s;
}

static inline void split_lines(const std::string& s, std::vector<std::string>& out) {
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

static inline std::string normalize_spaces(std::string s) {
    // Convert any whitespace runs to a single space.
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
    return trim_ws(out);
}

static inline std::string normalize_tags_block(const std::string& s) {
    // Split on common separators (incl. '.' used by some captioners), trim, and join with ", ".
    std::string tmp;
    tmp.reserve(s.size());
    for (char ch : s) {
        if (ch == '\r') continue;
        const bool sep = (ch == '\n' || ch == ';' || ch == '.' || ch == '|' || ch == '\t');
        if (sep) tmp.push_back(',');
        else tmp.push_back(ch);
    }

    std::vector<std::string> parts;
    std::string cur;
    for (char ch : tmp) {
        if (ch == ',') {
            const std::string t = trim_ws(cur);
            if (!t.empty()) parts.push_back(t);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    {
        const std::string t = trim_ws(cur);
        if (!t.empty()) parts.push_back(t);
    }

    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += ", ";
        out += parts[i];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Captions key/value: caption/theme/keywords/tokens
// ---------------------------------------------------------------------------

struct KvCaption {
    bool has_any = false;
    std::string caption;
    std::string theme;
    std::string keywords;
    std::string tokens;
};

static inline std::string to_lower_ascii(std::string s) {
    for (char& ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (c >= 'A' && c <= 'Z') ch = static_cast<char>(c + ('a' - 'A'));
    }
    return s;
}

static inline bool parse_kv_line(const std::string& line_in, std::string& out_key_lc, std::string& out_value) {
    // Format: key: value
    const std::string line = trim_ws(line_in);
    if (line.empty()) return false;
    const size_t colon = line.find(':');
    if (colon == std::string::npos) return false;
    if (colon == 0) return false;

    std::string key = trim_ws(line.substr(0, colon));
    std::string val = trim_ws(line.substr(colon + 1));
    if (key.empty()) return false;
    out_key_lc = to_lower_ascii(key);
    out_value = val;
    return true;
}

static inline std::string normalize_kv_list_as_spaces(const std::string& s) {
    // Convert separators (comma/semicolon/newlines/tabs) to spaces and normalize.
    std::string tmp;
    tmp.reserve(s.size());
    for (char ch : s) {
        if (ch == '\r') continue;
        if (ch == ',' || ch == ';' || ch == '\n' || ch == '\t') tmp.push_back(' ');
        else tmp.push_back(ch);
    }
    return normalize_spaces(tmp);
}

static inline KvCaption parse_kv_caption(const std::string& caption) {
    KvCaption out;
    std::vector<std::string> lines;
    split_lines(caption, lines);
    for (const std::string& raw : lines) {
        std::string k, v;
        if (!parse_kv_line(raw, k, v)) continue;
        if (k == "caption") {
            out.caption = v;
            out.has_any = true;
        } else if (k == "theme") {
            out.theme = v;
            out.has_any = true;
        } else if (k == "keywords") {
            out.keywords = v;
            out.has_any = true;
        } else if (k == "tokens") {
            out.tokens = v;
            out.has_any = true;
        }
    }

    // Normalize lists.
    out.keywords = normalize_kv_list_as_spaces(out.keywords);
    out.tokens = normalize_kv_list_as_spaces(out.tokens);
    out.caption = normalize_spaces(out.caption);
    out.theme = normalize_spaces(out.theme);
    return out;
}

static inline std::string compose_kv_caption_prompt(const KvCaption& kv) {
    // Compact form aimed at conditioning (no explicit "caption:" labels).
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
    return normalize_spaces(out);
}

static inline std::vector<std::string> split_terms_simple(const std::string& s) {
    // Split on common separators + whitespace. Intended for `tokens:`/`keywords:`.
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(32);
    auto flush = [&]() {
        std::string t = trim_ws(cur);
        if (!t.empty()) out.push_back(t);
        cur.clear();
    };
    for (char ch : s) {
        if (ch == '\r') continue;
        const unsigned char c = static_cast<unsigned char>(ch);
        const bool sep = (ch == ',' || ch == ';' || ch == '\n' || ch == '\t' || std::isspace(c));
        if (sep) {
            flush();
        } else {
            cur.push_back(ch);
        }
    }
    flush();
    return out;
}

struct StructuredCaption {
    bool has_any_header = false;
    std::string tags;
    std::string contexte;
    std::string mentalite;
    std::string texte;
    // Unknown sections in order of appearance.
    std::vector<std::pair<std::string, std::string>> extras;
};

static inline bool parse_section_header(const std::string& line_in, std::string& out_name_upper) {
    // Format: --- NAME --- (NAME may include parentheses)
    std::string line = trim_ws(line_in);
    if (!starts_with(line, "---")) return false;
    if (!ends_with(line, "---")) return false;
    if (line.size() < 6) return false;

    // Remove leading/trailing '---'
    std::string inner = line.substr(3, line.size() - 6);
    inner = trim_ws(inner);
    if (inner.empty()) return false;

    // Cut at '(' to accept "TEXTE (langue ...)".
    const size_t paren = inner.find('(');
    if (paren != std::string::npos) {
        inner = trim_ws(inner.substr(0, paren));
    }
    if (inner.empty()) return false;
    out_name_upper = to_upper_ascii(inner);
    return true;
}

static inline StructuredCaption parse_structured_caption(const std::string& caption) {
    StructuredCaption out;
    std::vector<std::string> lines;
    split_lines(caption, lines);

    enum class Sec { None, Tags, Contexte, Mentalite, Texte, Extra };
    Sec cur = Sec::None;
    int extra_idx = -1;

    for (const std::string& raw_line : lines) {
        std::string name;
        if (parse_section_header(raw_line, name)) {
            out.has_any_header = true;

            // Map header to known sections.
            if (starts_with(name, "TAGS") || starts_with(name, "TAG")) {
                cur = Sec::Tags;
                extra_idx = -1;
            } else if (starts_with(name, "CONTEXTE") || starts_with(name, "CONTEXT")) {
                cur = Sec::Contexte;
                extra_idx = -1;
            } else if (starts_with(name, "MENTALIT")) {
                cur = Sec::Mentalite;
                extra_idx = -1;
            } else if (starts_with(name, "TEXTE") || starts_with(name, "TEXT") || starts_with(name, "DESCRIPTION") || starts_with(name, "DESC")) {
                cur = Sec::Texte;
                extra_idx = -1;
            } else {
                cur = Sec::Extra;
                extra_idx = static_cast<int>(out.extras.size());
                out.extras.emplace_back(name, std::string());
            }
            continue;
        }

        // Content line.
        const std::string line = raw_line;
        auto append_line = [&](std::string& dst) {
            dst.append(line);
            dst.push_back('\n');
        };

        switch (cur) {
            case Sec::Tags: append_line(out.tags); break;
            case Sec::Contexte: append_line(out.contexte); break;
            case Sec::Mentalite: append_line(out.mentalite); break;
            case Sec::Texte: append_line(out.texte); break;
            case Sec::Extra:
                if (extra_idx >= 0 && extra_idx < static_cast<int>(out.extras.size())) {
                    out.extras[static_cast<size_t>(extra_idx)].second.append(line);
                    out.extras[static_cast<size_t>(extra_idx)].second.push_back('\n');
                } else {
                    append_line(out.contexte);
                }
                break;
            case Sec::None:
            default:
                // If no header yet, treat as contexte.
                append_line(out.contexte);
                break;
        }
    }

    // Normalize blocks.
    out.tags = normalize_tags_block(out.tags);
    out.contexte = normalize_spaces(out.contexte);
    out.mentalite = normalize_spaces(out.mentalite);

    // "Texte": keep some structure but avoid hard newlines.
    {
        std::string t;
        t.reserve(out.texte.size());
        for (char ch : out.texte) {
            if (ch == '\r') continue;
            if (ch == '\n') t.append(" / ");
            else t.push_back(ch);
        }
        out.texte = normalize_spaces(t);
    }

    for (auto& kv : out.extras) {
        kv.second = normalize_spaces(kv.second);
    }
    return out;
}

static inline std::string compose_structured_caption(const StructuredCaption& c, bool canonicalize) {
    if (!c.has_any_header) {
        // Not structured -> nothing to compose.
        return std::string();
    }

    auto add_line = [](std::string& out, const std::string& k, const std::string& v) {
        if (v.empty()) return;
        if (!out.empty()) out.push_back('\n');
        out += k;
        out += ": ";
        out += v;
    };

    if (canonicalize) {
        std::string out;
        add_line(out, "TAGS", c.tags);
        add_line(out, "CONTEXTE", c.contexte);
        add_line(out, "MENTALITE", c.mentalite);
        add_line(out, "TEXTE", c.texte);
        for (const auto& kv : c.extras) {
            if (kv.first.empty() || kv.second.empty()) continue;
            add_line(out, kv.first, kv.second);
        }
        return out;
    }

    // Raw-ish: keep explicit header blocks.
    auto add_block = [](std::string& out, const std::string& header, const std::string& body) {
        if (body.empty()) return;
        if (!out.empty()) out.push_back('\n');
        out += "--- ";
        out += header;
        out += " ---\n";
        out += body;
    };

    std::string out;
    add_block(out, "TAGS", c.tags);
    add_block(out, "CONTEXTE", c.contexte);
    add_block(out, "MENTALITE", c.mentalite);
    add_block(out, "TEXTE", c.texte);
    for (const auto& kv : c.extras) {
        if (kv.first.empty() || kv.second.empty()) continue;
        add_block(out, kv.first, kv.second);
    }
    return out;
}

static inline void apply_structured_dropout(StructuredCaption& c, std::mt19937& rng, float p_tags, float p_ctx, float p_ment, float p_txt) {
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    if (p_tags > 0.0f && u01(rng) < p_tags) c.tags.clear();
    if (p_ctx > 0.0f && u01(rng) < p_ctx) c.contexte.clear();
    if (p_ment > 0.0f && u01(rng) < p_ment) c.mentalite.clear();
    if (p_txt > 0.0f && u01(rng) < p_txt) c.texte.clear();
}

static std::string resolve_checkpoint_dir_for_loading(const std::string& ckpt_path_in) {
    std::string ckpt_path = strip_trailing_slashes(ckpt_path_in);
    if (ckpt_path.empty()) return ckpt_path;
    fs::path p(ckpt_path);

    // If user points at .../model, go up.
    if (p.filename() == "model") {
        p = p.parent_path();
    }

    // If directory contains epoch_* subdirs, use the latest.
    p = resolve_latest_epoch_dir(p);
    return p.string();
}

// NOTE: l'ancien chemin "peltier noise" (bruit corrélé/blur) a été retiré.
// PonyXLDDPMModel utilise désormais uniquement eps ~ N(0, I) (DDPM latent standard).

struct DistMoments {
    double mean = 0.0;
    double var = 0.0;
    double skew = 0.0;
};

static inline DistMoments compute_moments_local(const std::vector<float>& v) {
    DistMoments m;
    if (v.empty()) return m;
    double sum = 0.0;
    double sumsq = 0.0;
    double n = 0.0;
    for (float x : v) {
        if (!std::isfinite(x)) continue;
        const double xd = static_cast<double>(x);
        sum += xd;
        sumsq += xd * xd;
        n += 1.0;
    }
    if (n <= 0.0) return m;
    m.mean = sum / n;
    m.var = std::max(0.0, (sumsq / n) - m.mean * m.mean);

    // Skewness (best-effort)
    const double stdv = std::sqrt(std::max(1e-12, m.var));
    double acc3 = 0.0;
    for (float x : v) {
        if (!std::isfinite(x)) continue;
        const double z = (static_cast<double>(x) - m.mean) / std::max(1e-12, stdv);
        acc3 += z * z * z;
    }
    m.skew = acc3 / std::max(1.0, n);
    return m;
}

static inline std::string normalize_timestep_cond(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s.empty()) return "log_snr";
    if (s == "logsnr" || s == "log_snr_tanh" || s == "logsnr_tanh") return "log_snr";
    if (s == "tnorm" || s == "t" || s == "linear") return "t_norm";
    if (s == "tnorm_center" || s == "t_norm_center" || s == "centered" || s == "t_centered") return "t_norm_centered";
    return s;
}

static inline float compute_time_cond_value(const PonyXLDDPMModel::Config& cfg, float t_norm, float alpha_bar) {
    const std::string mode = normalize_timestep_cond(cfg.timestep_cond);
    if (mode == "t_norm") {
        return std::clamp(t_norm, 0.0f, 1.0f);
    }
    if (mode == "t_norm_centered") {
        const float tn = std::clamp(t_norm, 0.0f, 1.0f);
        return 2.0f * tn - 1.0f;
    }

    // Default: logSNR, compressed to roughly [-1,1].
    const float ab = std::clamp(alpha_bar, 1e-6f, 1.0f - 1e-6f);
    const float one_m = std::max(1e-6f, 1.0f - ab);
    const float logsnr = std::log(ab) - std::log(one_m);
    // Scale factor tuned so values stay in a sensible range across typical DDPM schedules.
    const float scaled = logsnr / 8.0f;
    return std::tanh(scaled);
}

static inline std::string normalize_loss_weighting(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s.empty()) return "none";
    if (s == "off" || s == "disabled") return "none";
    if (s == "minsnr" || s == "min_snr" || s == "min-snr") return "min_snr";
    return s;
}

static inline float compute_snr_from_alpha_bar(float alpha_bar) {
    const float ab = std::clamp(alpha_bar, 1e-6f, 1.0f - 1e-6f);
    const float one_m = std::max(1e-6f, 1.0f - ab);
    return ab / one_m;
}

static inline float compute_loss_weight(const PonyXLDDPMModel::Config& cfg, float alpha_bar) {
    const std::string mode = normalize_loss_weighting(cfg.loss_weighting);
    if (mode == "none") return 1.0f;

    const float snr = std::max(1e-6f, compute_snr_from_alpha_bar(alpha_bar));
    if (mode == "min_snr") {
        const float gamma = std::max(1e-6f, cfg.min_snr_gamma);
        const float w = std::min(snr, gamma) / snr;
        return std::isfinite(w) ? std::clamp(w, 0.0f, 1.0f) : 1.0f;
    }

    // Fallback: unknown mode => no weighting.
    return 1.0f;
}

static inline double mean_abs_adjacent_diff_local(const std::vector<float>& v) {
    if (v.size() < 2) return 0.0;
    double s = 0.0;
    for (size_t i = 1; i < v.size(); ++i) {
        s += std::abs(static_cast<double>(v[i]) - static_cast<double>(v[i - 1]));
    }
    return s / static_cast<double>(v.size() - 1);
}

static inline double pearson_corr_local(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = std::min(a.size(), b.size());
    if (n < 2) return 0.0;
    double sa = 0.0, sb = 0.0;
    double saa = 0.0, sbb = 0.0;
    double sab = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double xa = static_cast<double>(a[i]);
        const double xb = static_cast<double>(b[i]);
        sa += xa;
        sb += xb;
        saa += xa * xa;
        sbb += xb * xb;
        sab += xa * xb;
    }
    const double dn = static_cast<double>(n);
    const double ma = sa / dn;
    const double mb = sb / dn;
    const double va = std::max(0.0, saa / dn - ma * ma);
    const double vb = std::max(0.0, sbb / dn - mb * mb);
    const double cov = sab / dn - ma * mb;
    const double denom = std::sqrt(std::max(1e-12, va) * std::max(1e-12, vb));
    if (denom <= 0.0) return 0.0;
    return cov / denom;
}

static std::vector<uint8_t> to_rgb_preview_latent(const std::vector<float>& v,
                                                  int h,
                                                  int w,
                                                  int c,
                                                  int max_side,
                                                  const float* v_override = nullptr) {
    // v is H*W*C in "HWC" layout (tokens-major: (y*w+x)*c + ch).
    // We preview using first 3 channels if available, otherwise grayscale.
    if (h <= 0 || w <= 0 || c <= 0) return {};
    const size_t spatial = static_cast<size_t>(h) * static_cast<size_t>(w);
    const size_t need = spatial * static_cast<size_t>(c);
    if (v.size() < need) return {};

    const int ms = std::max(1, max_side);
    const int sx = (w > ms) ? static_cast<int>((w + ms - 1) / ms) : 1;
    const int sy = (h > ms) ? static_cast<int>((h + ms - 1) / ms) : 1;
    const int pw = std::max(1, w / sx);
    const int ph = std::max(1, h / sy);

    auto at = [&](int yy, int xx, int cc) -> float {
        // HWC
        const size_t idx = (static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(xx)) * static_cast<size_t>(c) + static_cast<size_t>(cc);
        const float* src = v_override ? v_override : v.data();
        if (idx >= need) return 0.0f;
        const float val = src[idx];
        return std::isfinite(val) ? val : 0.0f;
    };

    const bool rgb = (c >= 3);
    const int out_c = rgb ? 3 : 1;

    // Heuristic for better previews:
    // - Many latent channels are not equally informative. Showing channels (0,1,2)
    //   can look uniform even when other channels carry structure.
    // - We pick up to 3 channels with the highest variance (among a small prefix)
    //   on the sampled grid used for preview.
    std::array<int, 3> chan = {0, 1, 2};
    if (rgb && c > 3) {
        const int cand = std::max(3, std::min(c, 16));
        std::vector<double> sum(static_cast<size_t>(cand), 0.0);
        std::vector<double> sumsq(static_cast<size_t>(cand), 0.0);
        size_t nstat = 0;
        for (int y = 0; y < ph; ++y) {
            const int yy = y * sy;
            for (int x = 0; x < pw; ++x) {
                const int xx = x * sx;
                for (int cc = 0; cc < cand; ++cc) {
                    const float val = at(yy, xx, cc);
                    const double dv = static_cast<double>(val);
                    sum[static_cast<size_t>(cc)] += dv;
                    sumsq[static_cast<size_t>(cc)] += dv * dv;
                }
                nstat += 1;
            }
        }

        struct CVar { int ch; double var; };
        std::vector<CVar> vars;
        vars.reserve(static_cast<size_t>(cand));
        const double dn = std::max<size_t>(1, nstat);
        for (int cc = 0; cc < cand; ++cc) {
            const double m = sum[static_cast<size_t>(cc)] / dn;
            const double v2 = std::max(0.0, sumsq[static_cast<size_t>(cc)] / dn - m * m);
            vars.push_back({cc, v2});
        }
        std::sort(vars.begin(), vars.end(), [](const CVar& a, const CVar& b) {
            if (a.var != b.var) return a.var > b.var;
            return a.ch < b.ch;
        });
        // Keep unique channels; fall back to (0,1,2) if something goes wrong.
        if (vars.size() >= 3) {
            chan[0] = vars[0].ch;
            chan[1] = (vars[1].ch != chan[0]) ? vars[1].ch : 1;
            chan[2] = (vars[2].ch != chan[0] && vars[2].ch != chan[1]) ? vars[2].ch : 2;
        }
    }
    std::vector<float> map;
    map.assign(static_cast<size_t>(pw) * static_cast<size_t>(ph) * static_cast<size_t>(out_c), 0.0f);

    for (int y = 0; y < ph; ++y) {
        const int yy = y * sy;
        for (int x = 0; x < pw; ++x) {
            const int xx = x * sx;
            if (rgb) {
                for (int cc = 0; cc < 3; ++cc) {
                    const int src_ch = std::clamp(chan[static_cast<size_t>(cc)], 0, std::max(0, c - 1));
                    map[(static_cast<size_t>(y) * static_cast<size_t>(pw) + static_cast<size_t>(x)) * 3ULL + static_cast<size_t>(cc)] = at(yy, xx, src_ch);
                }
            } else {
                // Mean over a few channels.
                const int take = std::max(1, std::min(c, 8));
                double acc = 0.0;
                for (int cc = 0; cc < take; ++cc) acc += static_cast<double>(at(yy, xx, cc));
                map[static_cast<size_t>(y) * static_cast<size_t>(pw) + static_cast<size_t>(x)] = static_cast<float>(acc / static_cast<double>(take));
            }
        }
    }

    // Robust normalization: per-channel z-score on the sampled preview grid,
    // then compress with tanh. This avoids “flat” previews when one outlier
    // dominates max_abs.
    std::array<double, 3> mean = {0.0, 0.0, 0.0};
    std::array<double, 3> var = {0.0, 0.0, 0.0};
    {
        std::array<double, 3> sum = {0.0, 0.0, 0.0};
        std::array<double, 3> sumsq = {0.0, 0.0, 0.0};
        size_t nstat = 0;
        if (out_c == 1) {
            for (float x : map) {
                const double v0 = std::isfinite(x) ? static_cast<double>(x) : 0.0;
                sum[0] += v0;
                sumsq[0] += v0 * v0;
                nstat += 1;
            }
        } else {
            for (size_t i = 0; i + 2 < map.size(); i += 3) {
                for (int cc = 0; cc < 3; ++cc) {
                    const float x = map[i + static_cast<size_t>(cc)];
                    const double v0 = std::isfinite(x) ? static_cast<double>(x) : 0.0;
                    sum[static_cast<size_t>(cc)] += v0;
                    sumsq[static_cast<size_t>(cc)] += v0 * v0;
                }
                nstat += 1;
            }
        }
        const double dn = static_cast<double>(std::max<size_t>(1, nstat));
        for (int cc = 0; cc < out_c; ++cc) {
            mean[static_cast<size_t>(cc)] = sum[static_cast<size_t>(cc)] / dn;
            var[static_cast<size_t>(cc)] = std::max(0.0, sumsq[static_cast<size_t>(cc)] / dn - mean[static_cast<size_t>(cc)] * mean[static_cast<size_t>(cc)]);
        }
    }

    std::vector<uint8_t> px;
    px.resize(map.size());
    if (out_c == 1) {
        const double std0 = std::sqrt(std::max(1e-12, var[0]));
        for (size_t i = 0; i < map.size(); ++i) {
            const double x = std::isfinite(map[i]) ? static_cast<double>(map[i]) : 0.0;
            const double s = (x - mean[0]) / (3.0 * std0 + 1e-6);
            const double t = 0.5 + 0.5 * std::tanh(s);
            const int p = static_cast<int>(std::lround(std::clamp(t, 0.0, 1.0) * 255.0));
            px[i] = static_cast<uint8_t>(std::clamp(p, 0, 255));
        }
    } else {
        const double std0 = std::sqrt(std::max(1e-12, var[0]));
        const double std1 = std::sqrt(std::max(1e-12, var[1]));
        const double std2 = std::sqrt(std::max(1e-12, var[2]));
        for (size_t i = 0; i + 2 < map.size(); i += 3) {
            const double x0 = std::isfinite(map[i + 0]) ? static_cast<double>(map[i + 0]) : 0.0;
            const double x1 = std::isfinite(map[i + 1]) ? static_cast<double>(map[i + 1]) : 0.0;
            const double x2 = std::isfinite(map[i + 2]) ? static_cast<double>(map[i + 2]) : 0.0;

            const double s0 = (x0 - mean[0]) / (3.0 * std0 + 1e-6);
            const double s1 = (x1 - mean[1]) / (3.0 * std1 + 1e-6);
            const double s2 = (x2 - mean[2]) / (3.0 * std2 + 1e-6);

            const double t0 = 0.5 + 0.5 * std::tanh(s0);
            const double t1 = 0.5 + 0.5 * std::tanh(s1);
            const double t2 = 0.5 + 0.5 * std::tanh(s2);

            px[i + 0] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(std::clamp(t0, 0.0, 1.0) * 255.0)), 0, 255));
            px[i + 1] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(std::clamp(t1, 0.0, 1.0) * 255.0)), 0, 255));
            px[i + 2] = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(std::clamp(t2, 0.0, 1.0) * 255.0)), 0, 255));
        }
    }
    return px;
}

static inline bool latent_chw_to_tokens_hwc_scaled(std::vector<float>& out_tokens_hwc,
                                                   const float* in_chw,
                                                   int h,
                                                   int w,
                                                   int c,
                                                   float scale) {
    if (!in_chw || h <= 0 || w <= 0 || c <= 0) return false;
    const size_t need = static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(c);
    out_tokens_hwc.assign(need, 0.0f);
    for (int yy = 0; yy < h; ++yy) {
        for (int xx = 0; xx < w; ++xx) {
            for (int cc = 0; cc < c; ++cc) {
                const size_t src = (static_cast<size_t>(cc) * static_cast<size_t>(h) + static_cast<size_t>(yy)) * static_cast<size_t>(w) + static_cast<size_t>(xx);
                const size_t dst = (static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(xx)) * static_cast<size_t>(c) + static_cast<size_t>(cc);
                out_tokens_hwc[dst] = in_chw[src] * scale;
            }
        }
    }
    return true;
}

static inline bool latent_tokens_hwc_to_chw_scaled(std::vector<float>& out_chw,
                                                   const float* in_tokens_hwc,
                                                   int h,
                                                   int w,
                                                   int c,
                                                   float scale) {
    if (!in_tokens_hwc || h <= 0 || w <= 0 || c <= 0) return false;
    const size_t need = static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(c);
    out_chw.assign(need, 0.0f);
    for (int yy = 0; yy < h; ++yy) {
        for (int xx = 0; xx < w; ++xx) {
            for (int cc = 0; cc < c; ++cc) {
                const size_t src = (static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(xx)) * static_cast<size_t>(c) + static_cast<size_t>(cc);
                const size_t dst = (static_cast<size_t>(cc) * static_cast<size_t>(h) + static_cast<size_t>(yy)) * static_cast<size_t>(w) + static_cast<size_t>(xx);
                out_chw[dst] = in_tokens_hwc[src] * scale;
            }
        }
    }
    return true;
}

static std::vector<uint8_t> to_rgb_preview_image(const std::vector<float>& img,
                                                 int w,
                                                 int h,
                                                 int max_side) {
    if (w <= 0 || h <= 0) return {};
    const size_t need = static_cast<size_t>(w) * static_cast<size_t>(h) * 3ULL;
    if (img.size() < need) return {};

    const int ms = std::max(1, max_side);
    const int sx = (w > ms) ? static_cast<int>((w + ms - 1) / ms) : 1;
    const int sy = (h > ms) ? static_cast<int>((h + ms - 1) / ms) : 1;
    const int pw = std::max(1, w / sx);
    const int ph = std::max(1, h / sy);

    // Heuristic: the VAE decoder should output in [-1,1] (tanh), but if the range
    // is off we auto-normalize so the preview stays informative (avoids white squares).
    float minv = std::numeric_limits<float>::infinity();
    float maxv = -std::numeric_limits<float>::infinity();
    float max_abs = 0.0f;
    double sum = 0.0;
    double sumsq = 0.0;
    size_t nstat = 0;
    {
        // Compute stats on the sampled pixels only (same ones used for preview).
        for (int y = 0; y < ph; ++y) {
            const int yy = y * sy;
            for (int x = 0; x < pw; ++x) {
                const int xx = x * sx;
                const size_t in_base = (static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(xx)) * 3ULL;
                for (int cc = 0; cc < 3; ++cc) {
                    const float v = img[in_base + static_cast<size_t>(cc)];
                    if (!std::isfinite(v)) continue;
                    minv = std::min(minv, v);
                    maxv = std::max(maxv, v);
                    max_abs = std::max(max_abs, std::fabs(v));
                    sum += static_cast<double>(v);
                    sumsq += static_cast<double>(v) * static_cast<double>(v);
                    nstat += 1;
                }
            }
        }
        if (!std::isfinite(minv) || !std::isfinite(maxv)) {
            minv = -1.0f;
            maxv = 1.0f;
            max_abs = 1.0f;
            sum = 0.0;
            sumsq = 1.0;
            nstat = 1;
        }
        if (!(max_abs > 0.0f)) max_abs = 1.0f;
    }

    const double mean = (nstat > 0) ? (sum / static_cast<double>(nstat)) : 0.0;
    const double var = (nstat > 0) ? std::max(0.0, (sumsq / static_cast<double>(nstat)) - mean * mean) : 0.0;
    const float stdv = static_cast<float>(std::sqrt(std::max(1e-12, var)));

    enum class MapMode { MinusOneToOne, ZeroToOne, AutoTanh };
    MapMode mode = MapMode::MinusOneToOne;
    // If it's already [0,1], don't shift it (otherwise it becomes washed out).
    if (minv >= -1e-3f && maxv <= 1.0f + 1e-3f) {
        mode = MapMode::ZeroToOne;
    } else if (minv >= -1.0f - 5e-2f && maxv <= 1.0f + 5e-2f) {
        mode = MapMode::MinusOneToOne;
    } else {
        mode = MapMode::AutoTanh;
    }

    std::vector<uint8_t> px;
    px.resize(static_cast<size_t>(pw) * static_cast<size_t>(ph) * 3ULL);
    for (int y = 0; y < ph; ++y) {
        const int yy = y * sy;
        for (int x = 0; x < pw; ++x) {
            const int xx = x * sx;
            const size_t in_base = (static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(xx)) * 3ULL;
            const size_t out_base = (static_cast<size_t>(y) * static_cast<size_t>(pw) + static_cast<size_t>(x)) * 3ULL;
            for (int cc = 0; cc < 3; ++cc) {
                const float v0 = img[in_base + static_cast<size_t>(cc)];
                const float v = std::isfinite(v0) ? v0 : 0.0f;

                float t = 0.0f;
                if (mode == MapMode::MinusOneToOne) {
                    t = std::clamp(0.5f + 0.5f * v, 0.0f, 1.0f);
                } else if (mode == MapMode::ZeroToOne) {
                    t = std::clamp(v, 0.0f, 1.0f);
                } else {
                    // Robust scaling when values are out-of-range: center & scale (z-score), then compress via tanh.
                    const float s = (v - static_cast<float>(mean)) / (3.0f * stdv + 1e-6f);
                    t = 0.5f + 0.5f * std::tanh(s);
                    t = std::clamp(t, 0.0f, 1.0f);
                }

                px[out_base + static_cast<size_t>(cc)] = static_cast<uint8_t>(std::lround(t * 255.0f));
            }
        }
    }
    return px;
}

static bool try_read_json_file(const fs::path& p, nlohmann::json* out) {
    if (!out) return false;
    try {
        std::ifstream f(p);
        if (!f.is_open()) return false;
        nlohmann::json j;
        f >> j;
        *out = std::move(j);
        return true;
    } catch (...) {
        return false;
    }
}

static fs::path resolve_latest_epoch_dir(const fs::path& base) {
    // Best-effort: pick the highest epoch_XXXX directory.
    // If none found, return base.
    try {
        if (!fs::exists(base) || !fs::is_directory(base)) return base;
        int best_epoch = -1;
        fs::path best;
        for (const auto& it : fs::directory_iterator(base)) {
            if (!it.is_directory()) continue;
            const auto name = it.path().filename().string();
            if (name.rfind("epoch_", 0) != 0) continue;
            const auto num = name.substr(std::string("epoch_").size());
            int e = -1;
            try { e = std::stoi(num); } catch (...) { e = -1; }
            if (e > best_epoch) {
                best_epoch = e;
                best = it.path();
            }
        }
        if (best_epoch >= 0) return best;
    } catch (...) {
    }
    return base;
}

static bool load_vae_architecture_json(const std::string& ckpt_path_in, nlohmann::json* arch_out) {
    if (!arch_out) return false;

    std::string ckpt_path = strip_trailing_slashes(ckpt_path_in);
    if (ckpt_path.empty()) return false;

    fs::path p(ckpt_path);

    // If user points at .../model, go up.
    if (p.filename() == "model") {
        p = p.parent_path();
    }

    // If directory contains epoch_* subdirs, use the latest.
    p = resolve_latest_epoch_dir(p);

    // RawFolder layout: <root>/model/architecture.json
    fs::path arch1 = p / "model" / "architecture.json";
    if (try_read_json_file(arch1, arch_out)) return true;

    // Some callers may pass directly the model dir.
    fs::path arch2 = p / "architecture.json";
    if (try_read_json_file(arch2, arch_out)) return true;

    return false;
}

static void apply_vae_autodims_from_arch(PonyXLDDPMModel::Config& cfg, const nlohmann::json& arch) {
    // Prefer model_config if present.
    const nlohmann::json* mc = nullptr;
    if (arch.contains("model_config") && arch["model_config"].is_object()) mc = &arch["model_config"];
    if (!mc && arch.contains("modelConfig") && arch["modelConfig"].is_object()) mc = &arch["modelConfig"];

    int img_w = 0, img_h = 0, img_c = 0;
    int lh = 0, lw = 0, lc = 0;
    int base_channels = 0;

    auto jgeti = [](const nlohmann::json* j, const char* k) -> int {
        if (!j) return 0;
        try {
            if (j->contains(k)) return (*j)[k].get<int>();
        } catch (...) {
        }
        return 0;
    };

    img_w = jgeti(mc, "image_w");
    img_h = jgeti(mc, "image_h");
    img_c = jgeti(mc, "image_c");
    lh = jgeti(mc, "latent_h");
    lw = jgeti(mc, "latent_w");
    lc = jgeti(mc, "latent_c");
    base_channels = jgeti(mc, "base_channels");

    // Fallback: some dumps store partial information in layers.
    if (lc <= 0 && arch.contains("layers") && arch["layers"].is_array()) {
        try {
            for (const auto& L : arch["layers"]) {
                if (!L.is_object()) continue;
                if (L.contains("name") && L["name"].is_string() && L["name"].get<std::string>() == "vae_conv/enc/mu") {
                    if (L.contains("out_channels")) {
                        lc = L["out_channels"].get<int>();
                    }
                }
                if (base_channels <= 0 && L.contains("name") && L["name"].is_string() && L["name"].get<std::string>() == "vae_conv/enc/conv_in") {
                    if (L.contains("out_channels")) {
                        base_channels = L["out_channels"].get<int>();
                    }
                }
            }
        } catch (...) {
        }
    }

    const bool has_latent = (lh > 0 && lw > 0 && lc > 0);
    const bool has_image = (img_w > 0 && img_h > 0 && img_c > 0);

    if (has_image) {
        if (cfg.image_w != img_w || cfg.image_h != img_h || cfg.image_c != img_c) {
            std::cout << "✓ PonyXL: VAE auto-dims (image) "
                      << cfg.image_w << "x" << cfg.image_h << "x" << cfg.image_c
                      << " -> " << img_w << "x" << img_h << "x" << img_c << std::endl;
        }
        cfg.image_w = img_w;
        cfg.image_h = img_h;
        cfg.image_c = img_c;
    }

    if (has_latent) {
        const int prev_lh = cfg.latent_h;
        const int prev_lw = cfg.latent_w;
        const int prev_lc = cfg.latent_in_dim;
        if (prev_lh != lh || prev_lw != lw || prev_lc != lc) {
            std::cout << "✓ PonyXL: VAE auto-dims (latent) "
                      << "lh=" << prev_lh << " lw=" << prev_lw << " lc=" << prev_lc
                      << " -> lh=" << lh << " lw=" << lw << " lc=" << lc << std::endl;
        }
        cfg.latent_h = lh;
        cfg.latent_w = lw;
        cfg.latent_in_dim = lc;
        cfg.latent_seq_len = lh * lw;
    }

    if (cfg.vae_base_channels <= 0 && base_channels > 0) {
        cfg.vae_base_channels = base_channels;
    }
}
} // namespace

PonyXLDDPMModel::PonyXLDDPMModel() {
    setModelName("PonyXLSDXL");
    // SDXL-like: l'encoder est dans le graphe (tok_emb + blocks), pas l'Encoder externe.
    setHasEncoder(true);
}

void PonyXLDDPMModel::setLiveKL(float kl_beta, int kl_warmup_steps) {
    cfg_.kl_beta = std::max(0.0f, kl_beta);
    cfg_.kl_warmup_steps = std::max(0, kl_warmup_steps);

    // Garder modelConfig aligné (utile pour debug/viz/serialization).
    modelConfig["kl_beta"] = cfg_.kl_beta;
    modelConfig["kl_warmup_steps"] = cfg_.kl_warmup_steps;
}

namespace {
static std::string normalize_loss_name(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "" || s == "none") return "mse";
    if (s == "l2") return "mse";
    if (s == "l1") return "mae";
    if (s == "smooth_l1") return "smoothl1";
    if (s == "gaussian-nll") return "gaussian_nll";
    return s;
}

static std::string normalize_output_activation(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s.empty()) return "linear";
    if (s == "identity" || s == "none") return "linear";
    if (s == "tanh") return "tanh";
    if (s == "linear") return "linear";
    return s;
}
} // namespace

void PonyXLDDPMModel::buildFromConfig(const Config& cfg) {
    cfg_ = cfg;

    // Seed RNG (reproductibilité). On évite un seed codé en dur.
    const int s = cfg_.seed;
    rng_.seed(static_cast<uint32_t>(s < 0 ? -static_cast<int64_t>(s) : s));

    // Normaliser le nom de loss pour Model::computeLoss.
    cfg_.recon_loss = normalize_loss_name(cfg_.recon_loss);
    if (cfg_.recon_loss != "mse" && cfg_.recon_loss != "mae" && cfg_.recon_loss != "bce" &&
        cfg_.recon_loss != "huber" && cfg_.recon_loss != "smoothl1" && cfg_.recon_loss != "charbonnier" &&
        cfg_.recon_loss != "gaussian_nll" && cfg_.recon_loss != "nll_gaussian") {
        std::cout << "⚠️  PonyXLDDPM: recon_loss='" << cfg_.recon_loss << "' unsupported; falling back to mse" << std::endl;
        cfg_.recon_loss = "mse";
    }

    // Normaliser la pondération de loss en fonction du timestep.
    cfg_.loss_weighting = normalize_loss_weighting(cfg_.loss_weighting);
    if (cfg_.loss_weighting != "none" && cfg_.loss_weighting != "min_snr") {
        std::cout << "⚠️  PonyXLDDPM: loss_weighting='" << cfg_.loss_weighting << "' unsupported; falling back to none" << std::endl;
        cfg_.loss_weighting = "none";
    }
    if (!(cfg_.min_snr_gamma > 0.0f) || !std::isfinite(cfg_.min_snr_gamma)) {
        cfg_.min_snr_gamma = 5.0f;
    }

    cfg_.output_activation = normalize_output_activation(cfg_.output_activation);
    if (cfg_.output_activation != "linear" && cfg_.output_activation != "tanh") {
        std::cout << "⚠️  PonyXLDDPM: output_activation='" << cfg_.output_activation << "' unsupported; falling back to linear" << std::endl;
        cfg_.output_activation = "linear";
    }

    // Auto-align dims from VAE checkpoint when available (RawFolder architecture.json).
    // This keeps PonyXL perfectly compatible with the VAEConv checkpoint without modifying the VAE.
    const std::string vae_ckpt_resolved = resolve_checkpoint_dir_for_loading(cfg_.vae_checkpoint);
    if (!vae_ckpt_resolved.empty() && Mimir::Serialization::detect_format(vae_ckpt_resolved) == Mimir::Serialization::CheckpointFormat::RawFolder) {
        nlohmann::json arch;
        if (load_vae_architecture_json(vae_ckpt_resolved, &arch)) {
            apply_vae_autodims_from_arch(cfg_, arch);
        }
    }

    // Encoder externe: utilisé pour fournir mag/mod (broadcast add sur d_model).
    // Doit être aligné AVANT setTokenizer (qui peut allouer des embeddings encoder).
    getMutableEncoder().ensureDim(std::max(1, cfg_.d_model));

    // Garder tokenizer/embeddings cohérents avec la config.
    // Important quand on compose le vocab depuis le dataset.
    const int mv = std::max(8, cfg_.max_vocab);
    setTokenizer(Tokenizer(static_cast<size_t>(mv)));
    // Pour permettre des prompts longs (ex: 512 tokens), garder un max sequence length cohérent.
    getMutableTokenizer().setMaxSequenceLength(std::max(1, cfg_.text_ctx_len));

    // IMPORTANT: aligner la capacité du tokenizer sur la taille de l'embedding texte.
    // Sinon, tokenizeEnsure() peut produire des IDs >= max_vocab, et on perd la correspondance.
    getMutableTokenizer().setMaxVocab(static_cast<size_t>(std::max(1, cfg_.max_vocab)));

    // Réserver les tokens de contexte global (appris) tôt, pour stabiliser leurs IDs.
    const int gct = std::max(0, cfg_.global_ctx_tokens);
    for (int i = 0; i < gct; ++i) {
        (void)getMutableTokenizer().addToken("<GCTX" + std::to_string(i) + ">");
    }

    buildInto(*this, cfg_);
}

void PonyXLDDPMModel::accumulateVaeMuMoments(
    const std::vector<uint8_t>& rgb,
    int w,
    int h,
    double& sum,
    double& sumsq,
    size_t& n
) {
    const int W = std::max(1, w);
    const int H = std::max(1, h);
    const int latent_len = std::max(1, cfg_.latent_seq_len);
    const int latent_in_dim = std::max(1, cfg_.latent_in_dim);
    const int latent_raw_dim = latent_len * latent_in_dim;

    if (cfg_.vae_checkpoint.empty()) {
        throw std::runtime_error("PonyXLDDPMModel: vae_checkpoint is required (cannot compute std(z) without VAE)");
    }

    // Lazy-load VAE (same path as training)
    if (!vae_) {
        using json = nlohmann::json;
        json vae_cfg = ModelArchitectures::defaultConfig(cfg_.vae_arch);
        vae_cfg["image_w"] = W;
        vae_cfg["image_h"] = H;
        vae_cfg["image_c"] = 3;

        // Pour les pipelines type SD/SDXL, on veut un encodeur déterministe.
        // NOTE: même si on extrait mu pour x0, garder le VAE en mode déterministe évite
        // des reconstructions/viz bruitées et rend le comportement plus stable.
        if (cfg_.vae_arch == "vae_conv") {
            vae_cfg["stochastic_latent"] = false;
        }

        int lh = cfg_.latent_h;
        int lw = cfg_.latent_w;
        if (lh <= 0 && lw <= 0) {
            lh = 1;
            lw = latent_len;
        } else if (lh <= 0 && lw > 0) {
            lh = (latent_len % lw == 0) ? (latent_len / lw) : 1;
        } else if (lh > 0 && lw <= 0) {
            lw = (latent_len % lh == 0) ? (latent_len / lh) : latent_len;
        }
        lh = std::max(1, lh);
        lw = std::max(1, lw);
        vae_cfg["latent_h"] = lh;
        vae_cfg["latent_w"] = lw;
        vae_cfg["latent_c"] = latent_in_dim;

        if (cfg_.vae_base_channels > 0) {
            vae_cfg["base_channels"] = cfg_.vae_base_channels;
        }

        auto m = ModelArchitectures::create(cfg_.vae_arch, vae_cfg);
        if (!m) {
            throw std::runtime_error("Failed to create VAE model: " + cfg_.vae_arch);
        }
        m->allocateParams();

        Mimir::Serialization::LoadOptions opts;
        const std::string vae_ckpt = resolve_checkpoint_dir_for_loading(cfg_.vae_checkpoint);
        opts.format = Mimir::Serialization::detect_format(vae_ckpt);
        opts.load_tokenizer = false;
        opts.load_encoder = false;
        opts.load_optimizer = false;
        opts.strict_mode = false;
        opts.validate_checksums = false;

        std::string err;
        if (!Mimir::Serialization::load_checkpoint(*m, vae_ckpt, opts, &err)) {
            throw std::runtime_error("Failed to load VAE checkpoint: " + cfg_.vae_checkpoint + " (resolved=" + vae_ckpt + ") | " + err);
        }

        // Assure déterminisme même si le checkpoint embarque un flag différent.
        // Le layer Reparameterize lit modelConfig["stochastic_latent"] à l'exécution.
        if (cfg_.vae_arch == "vae_conv") {
            m->modelConfig["stochastic_latent"] = false;
        }

        // IMPORTANT: le VAE est un composant pré-entraîné utilisé uniquement pour encoder/décoder.
        // On gèle explicitement ses paramètres pour empêcher toute modification (entraînement/inférence).
        m->freezeParameters(true);
        vae_ = std::move(m);
    }

    const std::vector<float> img_f = imageBytesToFloatRGB(rgb, W, H);
    const std::vector<float> packed = vae_->forwardPass(img_f, false);

    int vae_image_dim = W * H * 3;
    int vae_latent_dim = 0;
    try {
        if (vae_->modelConfig.contains("latent_dim")) {
            vae_latent_dim = vae_->modelConfig["latent_dim"].get<int>();
        } else if (vae_->modelConfig.contains("latent_h") && vae_->modelConfig.contains("latent_w") && vae_->modelConfig.contains("latent_c")) {
            vae_latent_dim = vae_->modelConfig["latent_h"].get<int>() *
                             vae_->modelConfig["latent_w"].get<int>() *
                             vae_->modelConfig["latent_c"].get<int>();
        }
        if (vae_->modelConfig.contains("image_dim")) {
            vae_image_dim = vae_->modelConfig["image_dim"].get<int>();
        }
    } catch (...) {
    }

    if (vae_latent_dim <= 0) {
        throw std::runtime_error("VAE model missing latent_dim in modelConfig");
    }
    if (vae_latent_dim != latent_raw_dim) {
        throw std::runtime_error(
            "VAE latent_dim mismatch during std(z) calibration: VAE=" + std::to_string(vae_latent_dim) +
            " PonyXL(latent_seq_len*latent_in_dim)=" + std::to_string(latent_raw_dim)
        );
    }
    if (static_cast<int>(packed.size()) < (vae_image_dim + 2 * vae_latent_dim)) {
        throw std::runtime_error("VAE packed output too small (calibration)");
    }

    const size_t mu_off = static_cast<size_t>(vae_image_dim);
    for (int i = 0; i < vae_latent_dim; ++i) {
        const double x = static_cast<double>(packed[mu_off + static_cast<size_t>(i)]);
        sum += x;
        sumsq += x * x;
        n += 1;
    }
}

PonyXLDDPMModel::StepStats PonyXLDDPMModel::trainStepSdxlLatentDiffusion(
    const std::string& prompt,
    const std::vector<uint8_t>& rgb,
    int w,
    int h,
    Optimizer& opt,
    float learning_rate
) {
    if (layers.empty()) {
        throw std::runtime_error("PonyXLDDPMModel::trainStepSdxlLatentDiffusion: model not built");
    }
    if (layer_weight_blocks.empty()) {
        throw std::runtime_error("PonyXLDDPMModel::trainStepSdxlLatentDiffusion: weights not allocated (call allocateParams/initWeights)");
    }

    const int W = std::max(1, w);
    const int H = std::max(1, h);
    const int text_len = std::max(1, cfg_.text_ctx_len);
    const int latent_len = std::max(1, cfg_.latent_seq_len);
    const int latent_in_dim = std::max(1, cfg_.latent_in_dim);
    const int latent_raw_dim = latent_len * latent_in_dim;
    const int max_vocab = std::max(1, cfg_.max_vocab);

    // KL warmup (same shape as VAEConv): beta ramps linearly over kl_warmup_steps.
    const float kl_beta = std::max(0.0f, cfg_.kl_beta);
    const int kl_warmup_steps = std::max(0, cfg_.kl_warmup_steps);
    const float kl_progress = (kl_warmup_steps > 0)
        ? std::min(1.0f, static_cast<float>(static_cast<int>(opt.step) + 1) / static_cast<float>(kl_warmup_steps))
        : 1.0f;
    const float kl_beta_eff = kl_beta * kl_progress;

    float logvar_min = cfg_.logvar_clip_min;
    float logvar_max = cfg_.logvar_clip_max;
    if (!std::isfinite(logvar_min)) logvar_min = -10.0f;
    if (!std::isfinite(logvar_max)) logvar_max = 10.0f;
    if (logvar_min > logvar_max) std::swap(logvar_min, logvar_max);

    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::normal_distribution<float> n01(0.0f, 1.0f);

    // Prompt preprocessing + CFG dropout
    std::string used_prompt = sanitize_utf8(prompt);
    if (cfg_.max_text_chars > 0 && used_prompt.size() > static_cast<size_t>(cfg_.max_text_chars)) {
        used_prompt = truncate_utf8_safe(used_prompt, static_cast<size_t>(cfg_.max_text_chars));
    }

    // Support format key/value (caption/theme/keywords/tokens) si activé.
    // On l'applique tôt pour pouvoir compter les termes fréquents sans pré-charger le dataset.
    KvCaption kv;
    if (cfg_.caption_kv_enable) {
        kv = parse_kv_caption(used_prompt);
        if (kv.has_any) {
            used_prompt = compose_kv_caption_prompt(kv);
            if (cfg_.max_text_chars > 0 && used_prompt.size() > static_cast<size_t>(cfg_.max_text_chars)) {
                used_prompt = truncate_utf8_safe(used_prompt, static_cast<size_t>(cfg_.max_text_chars));
            }
        }
    }

    if (cfg_.caption_structured_enable) {
        StructuredCaption cap = parse_structured_caption(used_prompt);
        if (cap.has_any_header) {
            apply_structured_dropout(
                cap,
                rng_,
                std::clamp(cfg_.caption_tags_dropout_prob, 0.0f, 1.0f),
                std::clamp(cfg_.caption_contexte_dropout_prob, 0.0f, 1.0f),
                std::clamp(cfg_.caption_mentalite_dropout_prob, 0.0f, 1.0f),
                std::clamp(cfg_.caption_texte_dropout_prob, 0.0f, 1.0f)
            );

            const std::string composed = compose_structured_caption(cap, cfg_.caption_structured_canonicalize);
            if (!composed.empty()) {
                used_prompt = composed;
                if (cfg_.max_text_chars > 0 && used_prompt.size() > static_cast<size_t>(cfg_.max_text_chars)) {
                    used_prompt = truncate_utf8_safe(used_prompt, static_cast<size_t>(cfg_.max_text_chars));
                }
            }
        }
    }

    // Mettre à jour les fréquences (incrémentales) à partir de tokens/keywords si disponibles,
    // sinon fallback sur les IDs tokenisés du prompt.
    bool term_freq_updated_from_explicit = false;
    if (cfg_.term_freq_boost_enable) {
        std::vector<std::string> terms;
        if (kv.has_any) {
            if (cfg_.term_freq_boost_use_tokens && !kv.tokens.empty()) {
                const auto t = split_terms_simple(kv.tokens);
                terms.insert(terms.end(), t.begin(), t.end());
            }
            if (cfg_.term_freq_boost_use_keywords && !kv.keywords.empty()) {
                const auto t = split_terms_simple(kv.keywords);
                terms.insert(terms.end(), t.begin(), t.end());
            }
        }

        if (!terms.empty()) {
            term_freq_updated_from_explicit = true;
            const int pad_id0 = getMutableTokenizer().getPadId();
            const int unk_id0 = getMutableTokenizer().getUnkId();
            const int bos0 = getMutableTokenizer().getBosId();
            const int eos0 = getMutableTokenizer().getEosId();
            for (const std::string& raw_term : terms) {
                std::string norm = getMutableTokenizer().normalizeToken(raw_term);
                if (norm.empty()) continue;
                const int id = getMutableTokenizer().addToken(norm);
                if (id < 0) continue;
                if (id == pad_id0 || id == unk_id0 || id == bos0 || id == eos0) continue;
                term_freq_counts_[id] += 1;
            }
        }
    }

    if (cfg_.cfg_dropout_prob > 0.0f && u01(rng_) < cfg_.cfg_dropout_prob) {
        used_prompt.clear();
    }

    // Tokenizer -> ids (pad/trunc)
    std::vector<int> text_ids;
    if (!used_prompt.empty()) {
        text_ids = getMutableTokenizer().tokenizeEnsure(used_prompt);
    }

    // Fallback: si aucune liste explicite tokens/keywords n'est présente, compter les IDs du prompt.
    if (cfg_.term_freq_boost_enable && !term_freq_updated_from_explicit && !text_ids.empty()) {
        const int pad_id0 = getMutableTokenizer().getPadId();
        const int unk_id0 = getMutableTokenizer().getUnkId();
        const int bos0 = getMutableTokenizer().getBosId();
        const int eos0 = getMutableTokenizer().getEosId();

        size_t start = 0;
        size_t end = text_ids.size();
        if (start < end && text_ids[start] == bos0) ++start;
        if (end > start && text_ids[end - 1] == eos0) --end;
        for (size_t i = start; i < end; ++i) {
            const int id = text_ids[i];
            if (id < 0) continue;
            if (id == pad_id0 || id == unk_id0 || id == bos0 || id == eos0) continue;
            term_freq_counts_[id] += 1;
        }
    }

    // Rebuild top-K cache periodically (based on optimizer steps).
    if (cfg_.term_freq_boost_enable) {
        const int64_t step = static_cast<int64_t>(opt.step);
        const int start_step = std::max(0, cfg_.term_freq_boost_start_step);
        const int every = std::max(1, cfg_.term_freq_boost_update_every_steps);
        if (step >= start_step && (term_freq_last_rebuild_step_ < 0 || (step - term_freq_last_rebuild_step_) >= every)) {
            const int pad_id0 = getMutableTokenizer().getPadId();
            const int unk_id0 = getMutableTokenizer().getUnkId();
            const int bos0 = getMutableTokenizer().getBosId();
            const int eos0 = getMutableTokenizer().getEosId();
            const int topk = std::max(0, cfg_.term_freq_boost_top_k);

            std::vector<std::pair<int, uint64_t>> pairs;
            pairs.reserve(term_freq_counts_.size());
            for (const auto& kvp : term_freq_counts_) {
                const int id = kvp.first;
                if (id == pad_id0 || id == unk_id0 || id == bos0 || id == eos0) continue;
                if (kvp.second == 0) continue;
                pairs.emplace_back(id, kvp.second);
            }
            std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
                if (a.second != b.second) return a.second > b.second;
                return a.first < b.first;
            });

            term_freq_top_ids_.clear();
            term_freq_top_set_.clear();
            for (int i = 0; i < topk && i < static_cast<int>(pairs.size()); ++i) {
                term_freq_top_ids_.push_back(pairs[static_cast<size_t>(i)].first);
                term_freq_top_set_.insert(pairs[static_cast<size_t>(i)].first);
            }
            term_freq_last_rebuild_step_ = step;
        }
    }

    // Tokens de contexte global (appris), injectés après BOS.
    std::vector<int> gctx_ids;
    {
        const int want = std::max(0, cfg_.global_ctx_tokens);
        gctx_ids.reserve(static_cast<size_t>(want));
        for (int i = 0; i < want; ++i) {
            gctx_ids.push_back(getMutableTokenizer().addToken("<GCTX" + std::to_string(i) + ">"));
        }
    }

    const int pad_id = getMutableTokenizer().getPadId();
    const int unk_id = getMutableTokenizer().getUnkId();
    const int bos = getMutableTokenizer().getBosId();
    const int eos = getMutableTokenizer().getEosId();

    // Enforce: séquence délimitée par <BOS> ... <EOS>.
    if (text_len >= 2 && bos >= 0 && eos >= 0) {
        size_t start = 0;
        size_t end = text_ids.size();
        if (start < end && text_ids[start] == bos) ++start;
        if (end > start && text_ids[end - 1] == eos) --end;

        std::vector<int> content;
        if (end > start) {
            content.assign(text_ids.begin() + static_cast<std::ptrdiff_t>(start),
                           text_ids.begin() + static_cast<std::ptrdiff_t>(end));
        }

        const int cap = std::max(0, text_len - 2 - static_cast<int>(gctx_ids.size()));

        // Majoration: répéter certains tokens fréquents (top-K) pendant l'entraînement.
        if (cfg_.term_freq_boost_enable) {
            const int64_t step = static_cast<int64_t>(opt.step);
            const int start_step = std::max(0, cfg_.term_freq_boost_start_step);
            const int rep = std::max(0, cfg_.term_freq_boost_repeat);
            if (rep > 0 && step >= start_step && !term_freq_top_set_.empty() && cap > 0 && !content.empty()) {
                std::vector<int> boosted;
                boosted.reserve(content.size());
                for (size_t i = 0; i < content.size(); ++i) {
                    const int id = content[i];
                    if (static_cast<int>(boosted.size()) >= cap) break;
                    boosted.push_back(id);
                    if (static_cast<int>(boosted.size()) >= cap) break;
                    if (term_freq_top_set_.find(id) != term_freq_top_set_.end()) {
                        for (int r = 0; r < rep; ++r) {
                            if (static_cast<int>(boosted.size()) >= cap) break;
                            boosted.push_back(id);
                        }
                    }
                }
                content.swap(boosted);
            }
        }

        if (static_cast<int>(content.size()) > cap) {
            content.resize(static_cast<size_t>(cap));
        }

        text_ids.clear();
        text_ids.reserve(static_cast<size_t>(text_len));
        text_ids.push_back(bos);
        text_ids.insert(text_ids.end(), gctx_ids.begin(), gctx_ids.end());
        text_ids.insert(text_ids.end(), content.begin(), content.end());
        while (static_cast<int>(text_ids.size()) < (text_len - 1)) {
            text_ids.push_back(pad_id);
        }
        text_ids.push_back(eos);
    } else {
        // Fallback historique
        if (text_ids.size() > static_cast<size_t>(text_len)) {
            text_ids.resize(static_cast<size_t>(text_len));
        } else if (text_ids.size() < static_cast<size_t>(text_len)) {
            text_ids.resize(static_cast<size_t>(text_len), pad_id);
        }
    }

    for (auto& id : text_ids) {
        if (id == pad_id) continue;
        if (id < 0) id = pad_id;
        if (id >= max_vocab) id = (unk_id >= 0) ? unk_id : pad_id;
    }

    // VAE (pré-entraîné) requis
    if (cfg_.vae_checkpoint.empty()) {
        throw std::runtime_error(
            "PonyXLDDPMModel: vae_checkpoint is required for ponyxl_sdxl training. "
            "Train a VAE (ex: arch=vae_conv) and pass cfg.vae_checkpoint."
        );
    }

    if (!vae_) {
        using json = nlohmann::json;
        json vae_cfg = ModelArchitectures::defaultConfig(cfg_.vae_arch);
        vae_cfg["image_w"] = W;
        vae_cfg["image_h"] = H;
        vae_cfg["image_c"] = 3;

        int lh = cfg_.latent_h;
        int lw = cfg_.latent_w;
        if (lh <= 0 && lw <= 0) {
            lh = 1;
            lw = latent_len;
        } else if (lh <= 0 && lw > 0) {
            lh = (latent_len % lw == 0) ? (latent_len / lw) : 1;
        } else if (lh > 0 && lw <= 0) {
            lw = (latent_len % lh == 0) ? (latent_len / lh) : latent_len;
        }
        lh = std::max(1, lh);
        lw = std::max(1, lw);
        vae_cfg["latent_h"] = lh;
        vae_cfg["latent_w"] = lw;
        vae_cfg["latent_c"] = latent_in_dim;

        if (cfg_.vae_base_channels > 0) {
            vae_cfg["base_channels"] = cfg_.vae_base_channels;
        }

        auto m = ModelArchitectures::create(cfg_.vae_arch, vae_cfg);
        if (!m) {
            throw std::runtime_error("Failed to create VAE model: " + cfg_.vae_arch);
        }
        m->allocateParams();

        Mimir::Serialization::LoadOptions opts;
        const std::string vae_ckpt = resolve_checkpoint_dir_for_loading(cfg_.vae_checkpoint);
        opts.format = Mimir::Serialization::detect_format(vae_ckpt);
        opts.load_tokenizer = false;
        opts.load_encoder = false;
        opts.load_optimizer = false;
        opts.strict_mode = false;
        opts.validate_checksums = false;

        std::string err;
        if (!Mimir::Serialization::load_checkpoint(*m, vae_ckpt, opts, &err)) {
            throw std::runtime_error("Failed to load VAE checkpoint: " + cfg_.vae_checkpoint + " (resolved=" + vae_ckpt + ") | " + err);
        }

        // IMPORTANT: le VAE est pré-entraîné et ne doit jamais être modifié dans PonyXLDDPMModel.
        m->freezeParameters(true);
        vae_ = std::move(m);
    }

    // Encode image -> mu/logvar; use mu as deterministic x0
    const std::vector<float> img_f = imageBytesToFloatRGB(rgb, W, H);
    const std::vector<float> packed = vae_->forwardPass(img_f, false);

    int vae_image_dim = W * H * 3;
    int vae_latent_dim = 0;
    try {
        if (vae_->modelConfig.contains("latent_dim")) {
            vae_latent_dim = vae_->modelConfig["latent_dim"].get<int>();
        } else if (vae_->modelConfig.contains("latent_h") && vae_->modelConfig.contains("latent_w") && vae_->modelConfig.contains("latent_c")) {
            vae_latent_dim = vae_->modelConfig["latent_h"].get<int>() * vae_->modelConfig["latent_w"].get<int>() * vae_->modelConfig["latent_c"].get<int>();
        }
        if (vae_->modelConfig.contains("image_dim")) {
            vae_image_dim = vae_->modelConfig["image_dim"].get<int>();
        }
    } catch (...) {
    }

    if (vae_latent_dim <= 0) {
        throw std::runtime_error("VAE model missing latent_dim in modelConfig");
    }
    if (vae_latent_dim != latent_raw_dim) {
        throw std::runtime_error(
            "VAE latent_dim mismatch: VAE=" + std::to_string(vae_latent_dim) +
            " PonyXL(latent_seq_len*latent_in_dim)=" + std::to_string(latent_raw_dim)
        );
    }
    if (static_cast<int>(packed.size()) < (vae_image_dim + 2 * vae_latent_dim)) {
        throw std::runtime_error("VAE packed output too small");
    }

    // VAEConv exposes mu/logvar in CHW. PonyXL consumes latents as tokens-major (HWC).
    int vae_lh = cfg_.latent_h;
    int vae_lw = cfg_.latent_w;
    int vae_lc = latent_in_dim;
    try {
        if (vae_->modelConfig.contains("latent_h")) vae_lh = vae_->modelConfig["latent_h"].get<int>();
        if (vae_->modelConfig.contains("latent_w")) vae_lw = vae_->modelConfig["latent_w"].get<int>();
        if (vae_->modelConfig.contains("latent_c")) vae_lc = vae_->modelConfig["latent_c"].get<int>();
    } catch (...) {
    }
    vae_lh = std::max(1, vae_lh);
    vae_lw = std::max(1, vae_lw);
    vae_lc = std::max(1, vae_lc);
    if (vae_lc != latent_in_dim || (vae_lh * vae_lw) != latent_len) {
        throw std::runtime_error("VAE latent shape mismatch: expected (latent_h*latent_w==latent_seq_len) and latent_c==latent_in_dim");
    }

    std::vector<float> x0;
    const float scale = std::max(0.0f, cfg_.vae_scale);
    const size_t mu_off = static_cast<size_t>(vae_image_dim);
    if (!latent_chw_to_tokens_hwc_scaled(x0, packed.data() + mu_off, vae_lh, vae_lw, vae_lc, scale)) {
        throw std::runtime_error("Failed to convert VAE mu (CHW) to PonyXL latent (tokens HWC)");
    }

    // For DDPM training, generic activation/block viz taps are usually not what we want.
    // We disable taps during the heavy forward/backward, and re-enable briefly when emitting
    // the custom reconstruction/noise frames at the end.
    // IMPORTANT: do not leave taps disabled when returning, otherwise the Viz (Blocks/Layers)
    // becomes empty after the first step.
    const bool prev_viz_taps_enabled = isVizTapsEnabled();
    if (prev_viz_taps_enabled) {
        setVizTapsEnabled(false);
    }

    // Reset grads once per outer step; we will accumulate across inner steps_per_image.
    zeroGradients();

    // Latent H/W for noise blur (use VAE dims; they are the ground truth here).
    int lat_h = vae_lh;
    int lat_w = vae_lw;
    if (lat_h <= 0 && lat_w <= 0) {
        // Default: assume square-ish.
        const int s = static_cast<int>(std::sqrt(static_cast<double>(latent_len)));
        if (s > 0 && (s * s) == latent_len) {
            lat_h = s;
            lat_w = s;
        } else {
            lat_h = 1;
            lat_w = latent_len;
        }
    } else if (lat_h <= 0 && lat_w > 0) {
        lat_h = (latent_len % lat_w == 0) ? (latent_len / lat_w) : 1;
    } else if (lat_h > 0 && lat_w <= 0) {
        lat_w = (latent_len % lat_h == 0) ? (latent_len / lat_h) : latent_len;
    }
    lat_h = std::max(1, lat_h);
    lat_w = std::max(1, lat_w);

    // Cache alpha_bar schedule (thread-local) to avoid O(T) per step.
    const int T = std::max(2, cfg_.ddpm_steps);
    const float beta0 = std::clamp(cfg_.ddpm_beta_start, 0.0f, 0.999f);
    const float beta1 = std::clamp(cfg_.ddpm_beta_end, 0.0f, 0.999f);
    static thread_local int cache_T = 0;
    static thread_local float cache_b0 = -1.0f;
    static thread_local float cache_b1 = -1.0f;
    static thread_local std::vector<float> cache_ab;
    if (cache_T != T || cache_b0 != beta0 || cache_b1 != beta1 || static_cast<int>(cache_ab.size()) != T) {
        cache_T = T;
        cache_b0 = beta0;
        cache_b1 = beta1;
        cache_ab.assign(static_cast<size_t>(T), 1.0f);
        float ab = 1.0f;
        for (int i = 0; i < T; ++i) {
            const float frac = (T > 1) ? (static_cast<float>(i) / static_cast<float>(T - 1)) : 0.0f;
            const float beta = std::clamp(beta0 + (beta1 - beta0) * frac, 0.0f, 0.999f);
            const float alpha = 1.0f - beta;
            ab *= alpha;
            cache_ab[static_cast<size_t>(i)] = std::clamp(ab, 1e-6f, 1.0f);
        }
    }

    const int steps_per_image = std::max(1, cfg_.ddpm_steps_per_image);
    const float grad_scale = 1.0f / static_cast<float>(steps_per_image);
    std::uniform_int_distribution<int> ut(0, T - 1);

    double loss_sum = 0.0;
    double kl_sum = 0.0;
    double grad_sum = 0.0;
    float grad_max = 0.0f;
    float last_t_norm = 0.0f;
    std::vector<float> last_pred;
    std::vector<float> last_eps;
    float last_loss = 0.0f;
    float last_kl = 0.0f;

    // Optional: image-space auxiliary loss (throttled)
    const float img_loss_w = std::max(0.0f, cfg_.img_loss_weight);
    const int img_loss_every = std::max(0, cfg_.img_loss_every_steps);

    // For visualizer (only last step): keep a copy of the signals we want to show.
    std::vector<float> viz_x0;
    std::vector<float> viz_x_t;
    std::vector<float> viz_eps;
    float viz_sqrt_ab = 1.0f;
    float viz_sqrt_1mab = 0.0f;
    float viz_tnorm = 0.0f;

    for (int s = 0; s < steps_per_image; ++s) {
        const int t = ut(rng_);
        const float t_norm = (T > 1) ? (static_cast<float>(t) / static_cast<float>(T - 1)) : 0.0f;
        last_t_norm = t_norm;

        const float alpha_bar = cache_ab[static_cast<size_t>(t)];
        const float sqrt_ab = std::sqrt(alpha_bar);
        const float sqrt_1mab = std::sqrt(std::max(0.0f, 1.0f - alpha_bar));

        // eps ~ N(0, I) (DDPM standard)
        std::vector<float> eps(static_cast<size_t>(latent_raw_dim), 0.0f);
        for (int i = 0; i < latent_raw_dim; ++i) {
            eps[static_cast<size_t>(i)] = n01(rng_);
        }

        // Keep the last target noise for metrics.
        last_eps = eps;

        std::vector<float> x_t(static_cast<size_t>(latent_raw_dim), 0.0f);
        for (int i = 0; i < latent_raw_dim; ++i) {
            const float e = eps[static_cast<size_t>(i)];
            x_t[static_cast<size_t>(i)] = sqrt_ab * x0[static_cast<size_t>(i)] + sqrt_1mab * e;
        }

        // Keep a copy of x_t for optional image loss (only when needed).
        std::vector<float> x_t_keep;
        const bool want_img_loss = (img_loss_w > 0.0f && img_loss_every > 0 && (static_cast<int>(opt.step) % img_loss_every) == 0 && (s == steps_per_image - 1));
        if (want_img_loss) {
            x_t_keep = x_t;
        }

        if (prev_viz_taps_enabled && (s == steps_per_image - 1)) {
            viz_x0 = x0;
            viz_x_t = x_t;
            viz_eps = eps;
            viz_sqrt_ab = sqrt_ab;
            viz_sqrt_1mab = sqrt_1mab;
            viz_tnorm = t_norm;
        }

        std::unordered_map<std::string, std::vector<float>> fin;
        std::unordered_map<std::string, std::vector<int>> iin;
        fin.emplace("latent", std::move(x_t));
        iin.emplace("text_ids", text_ids);
        if (cfg_.sdxl_time_cond) {
            fin.emplace("timestep", std::vector<float>{compute_time_cond_value(cfg_, t_norm, alpha_bar)});
        }

        const std::vector<float>& pred_view = forwardPassNamedView(fin, iin, true);

        // IMPORTANT:
        // Model::computeLoss requires prediction.size() == target.size().
        // PonyXL's head may output extra channels (e.g. auxiliary outputs), so we compute the
        // diffusion loss only on the first latent_raw_dim values (or fewer if the model outputs less).
        const int pred_dim = static_cast<int>(pred_view.size());
        const int nloss = std::max(0, std::min(latent_raw_dim, pred_dim));

        // Keep the diffusion-relevant prediction slice for visualization/metrics (eps-part).
        // (This also ensures KL/corr are computed on the same signal as the loss.)

        // Slice prediction/target to match sizes for loss.
        std::vector<float> pred_slice;
        std::vector<float> tgt_slice;
        pred_slice.reserve(static_cast<size_t>(nloss));
        tgt_slice.reserve(static_cast<size_t>(nloss));

        // Track moments for KL prior regularization without an extra pass.
        double psum = 0.0;
        double psumsq = 0.0;
        int pcount = 0;
        for (int i = 0; i < nloss; ++i) {
            float pv = pred_view[static_cast<size_t>(i)];
            if (!std::isfinite(pv)) pv = 0.0f;
            pred_slice.push_back(pv);
            tgt_slice.push_back(eps[static_cast<size_t>(i)]);

            psum += static_cast<double>(pv);
            psumsq += static_cast<double>(pv) * static_cast<double>(pv);
            pcount += 1;
        }

        last_pred = pred_slice;

        const float wt = compute_loss_weight(cfg_, alpha_bar);
        const float base_loss = computeLoss(pred_slice, tgt_slice, cfg_.recon_loss);
        const float diff_loss = base_loss * wt;

        // Optional: KL prior regularization on eps_pred distribution (to N(0,1)).
        // KL(N(m, v) || N(0,1)) = 0.5 * (m^2 + v - 1 - log v)
        float kl_prior = 0.0f;
        if (kl_beta_eff > 0.0f && pcount > 0) {
            const double mean = psum / static_cast<double>(pcount);
            double var = (psumsq / static_cast<double>(pcount)) - mean * mean;
            if (!std::isfinite(var) || var < 1e-12) var = 1e-12;
            double lv = std::log(var);
            lv = std::clamp(lv, static_cast<double>(logvar_min), static_cast<double>(logvar_max));
            const double v_use = std::exp(lv);
            kl_prior = static_cast<float>(0.5 * (mean * mean + v_use - 1.0 - lv));
        }

        const float kl_loss = kl_beta_eff * wt * kl_prior;
        last_loss = diff_loss + kl_loss;
        last_kl = kl_prior;

        loss_sum += static_cast<double>(last_loss);
        kl_sum += static_cast<double>(kl_prior);

        // Backward: gradient only on the diffusion portion, zeros elsewhere.
        std::vector<float> grad_slice;
        computeLossGradientInto(pred_slice, tgt_slice, grad_slice, cfg_.recon_loss);
        std::vector<float> loss_grad(static_cast<size_t>(pred_dim), 0.0f);
        for (int i = 0; i < nloss; ++i) {
            loss_grad[static_cast<size_t>(i)] = grad_slice[static_cast<size_t>(i)] * wt;
        }

        // Optional: image-space auxiliary loss (decode x0_hat and backprop through VAE decoder)
        if (want_img_loss && !x_t_keep.empty()) {
            try {
                if (!vae_decode_) {
                    using json = nlohmann::json;
                    json dec_cfg = ModelArchitectures::defaultConfig("vae_conv_decode");
                    dec_cfg["image_w"] = W;
                    dec_cfg["image_h"] = H;
                    dec_cfg["image_c"] = 3;
                    dec_cfg["latent_h"] = lat_h;
                    dec_cfg["latent_w"] = lat_w;
                    dec_cfg["latent_c"] = latent_in_dim;
                    if (cfg_.vae_base_channels > 0) {
                        dec_cfg["base_channels"] = cfg_.vae_base_channels;
                    }

                    auto mdec = ModelArchitectures::create("vae_conv_decode", dec_cfg);
                    if (mdec) {
                        mdec->allocateParams();
                        Mimir::Serialization::LoadOptions lopts;
                        const std::string vae_ckpt = resolve_checkpoint_dir_for_loading(cfg_.vae_checkpoint);
                        lopts.format = Mimir::Serialization::detect_format(vae_ckpt);
                        lopts.load_tokenizer = false;
                        lopts.load_encoder = false;
                        lopts.load_optimizer = false;
                        lopts.strict_mode = false;
                        lopts.validate_checksums = false;
                        std::string err;
                        if (Mimir::Serialization::load_checkpoint(*mdec, vae_ckpt, lopts, &err)) {
                            vae_decode_ = std::move(mdec);
                        }
                    }
                }

                if (vae_decode_) {
                    // x0_hat in latent space
                    std::vector<float> x0_hat(static_cast<size_t>(nloss), 0.0f);
                    for (int i = 0; i < nloss; ++i) {
                        x0_hat[static_cast<size_t>(i)] = (x_t_keep[static_cast<size_t>(i)] - sqrt_1mab * pred_slice[static_cast<size_t>(i)]) / std::max(1e-6f, sqrt_ab);
                    }

                    // Decode requires CHW latents (unscale)
                    const float scale0 = std::max(0.0f, cfg_.vae_scale);
                    const float inv_scale = (scale0 > 0.0f) ? (1.0f / scale0) : 1.0f;
                    std::vector<float> hat_chw;
                    if (!latent_tokens_hwc_to_chw_scaled(hat_chw, x0_hat.data(), lat_h, lat_w, latent_in_dim, inv_scale)) {
                        throw std::runtime_error("img_loss: latent_tokens_hwc_to_chw_scaled failed");
                    }

                    // Forward decoder in training mode so backward is available.
                    const std::vector<float>& img_hat = vae_decode_->forwardPassView(hat_chw, true);
                    const int im_n = std::min<int>((int)img_hat.size(), (int)img_f.size());
                    if (im_n > 0) {
                        double acc = 0.0;
                        for (int i = 0; i < im_n; ++i) {
                            const double d = static_cast<double>(img_hat[static_cast<size_t>(i)]) - static_cast<double>(img_f[static_cast<size_t>(i)]);
                            acc += d * d;
                        }
                        const float img_mse = static_cast<float>(acc / static_cast<double>(im_n));
                        const float img_loss = img_loss_w * wt * img_mse;
                        last_loss += img_loss;
                        loss_sum += static_cast<double>(img_loss);

                        // d/dimg_hat MSE
                        std::vector<float> g_img(static_cast<size_t>(img_hat.size()), 0.0f);
                        const float invN = 1.0f / static_cast<float>(im_n);
                        for (int i = 0; i < im_n; ++i) {
                            g_img[static_cast<size_t>(i)] = (2.0f * invN) * (img_hat[static_cast<size_t>(i)] - img_f[static_cast<size_t>(i)]);
                        }

                        // Scale gradients by weight and timestep weighting.
                        // The global grad_scale (1/steps_per_image) will be applied later to the whole loss_grad.
                        const float gscale = img_loss_w * wt;
                        if (gscale != 1.0f) {
                            for (float& g : g_img) g *= gscale;
                        }

                        const bool was_frozen = vae_decode_->parametersFrozen();
                        if (was_frozen) vae_decode_->freezeParameters(false);
                        vae_decode_->zeroGradients();
                        vae_decode_->backwardPass(g_img);
                        if (was_frozen) vae_decode_->freezeParameters(true);

                        if (vae_decode_->hasLastInputGradient()) {
                            const std::vector<float>& g_hat_chw = vae_decode_->getLastInputGradient();
                            // Convert CHW grad back to tokens and unscale accordingly.
                            std::vector<float> g_x0_tokens;
                            if (latent_chw_to_tokens_hwc_scaled(g_x0_tokens, g_hat_chw.data(), lat_h, lat_w, latent_in_dim, inv_scale)) {
                                const float k = -sqrt_1mab / std::max(1e-6f, sqrt_ab);
                                const int gn = std::min<int>(nloss, (int)g_x0_tokens.size());
                                for (int i = 0; i < gn; ++i) {
                                    loss_grad[static_cast<size_t>(i)] += g_x0_tokens[static_cast<size_t>(i)] * k;
                                }
                            }
                        }
                    }
                }
            } catch (...) {
                // Best-effort: image loss is optional; ignore failures.
            }
        }

        // Add KL prior gradient on the diffusion portion only.
        if (kl_beta_eff > 0.0f && kl_prior > 0.0f && pcount > 0) {
            const float invN = 1.0f / static_cast<float>(pcount);
            const float mean_f = static_cast<float>(psum / static_cast<double>(pcount));
            float var = static_cast<float>((psumsq / static_cast<double>(pcount)) - (psum / static_cast<double>(pcount)) * (psum / static_cast<double>(pcount)));
            if (!std::isfinite(var) || var < 1e-12f) var = 1e-12f;
            float lv = std::log(var);
            lv = std::clamp(lv, logvar_min, logvar_max);
            const float v_use = std::exp(lv);

            const float scale_kl = kl_beta_eff * wt;
            for (int i = 0; i < pcount; ++i) {
                const float x = pred_slice[static_cast<size_t>(i)];
                // d/dx_i KL(N(m,v)||N(0,1)) with m/var estimated over the vector.
                const float g = invN * (x - (x - mean_f) / std::max(1e-12f, v_use));
                loss_grad[static_cast<size_t>(i)] += scale_kl * g;
            }
        }

        // Scale by 1/steps_per_image to match the previous behavior (lr/steps_per_image per inner-step).
        if (grad_scale != 1.0f) {
            for (float& g : loss_grad) g *= grad_scale;
        }
        backwardPass(loss_grad);
    }

    // Compute grad stats after accumulation
    {
        double sum_sq = 0.0;
        float max_abs = 0.0f;
        for (const auto& layer : layers) {
            for (float g : layer.grad_weights) {
                sum_sq += static_cast<double>(g) * static_cast<double>(g);
                max_abs = std::max(max_abs, std::abs(g));
            }
        }
        grad_sum = std::sqrt(sum_sq);
        grad_max = max_abs;
    }

    // One optimizer step per image (matches gradient accumulation above).
    optimizerStep(opt, learning_rate);

    // Custom visualizer frames: reconstruction / denoise / mu (best-effort)
    if (prev_viz_taps_enabled && !viz_x0.empty() && !viz_x_t.empty() && !viz_eps.empty()) {
        // Enable just long enough to record our frames.
        setVizTapsEnabled(true);
        const int max_side = getVizTapsMaxSide();

        auto add_stats_frame = [&](const std::string& short_name, const std::vector<float>& v) {
            // Encode stats into the label (visible in focus/zoom overlay).
            // The thumbnail is a tiny 1x1 grayscale pixel (so it shows up in Blocks/Layers).
            size_t n_total = v.size();
            size_t n_finite = 0;
            size_t n_nonfinite = 0;

            // Welford
            double mean = 0.0;
            double m2 = 0.0;
            double sum_sq = 0.0;
            float vmin = 0.0f;
            float vmax = 0.0f;
            bool has_any = false;

            for (float x : v) {
                if (!std::isfinite(x)) {
                    n_nonfinite++;
                    continue;
                }
                n_finite++;
                sum_sq += static_cast<double>(x) * static_cast<double>(x);

                if (!has_any) {
                    vmin = x;
                    vmax = x;
                    has_any = true;
                } else {
                    vmin = std::min(vmin, x);
                    vmax = std::max(vmax, x);
                }

                const double dx = static_cast<double>(x) - mean;
                mean += dx / static_cast<double>(n_finite);
                const double dx2 = static_cast<double>(x) - mean;
                m2 += dx * dx2;
            }

            const double var = (n_finite > 1) ? (m2 / static_cast<double>(n_finite - 1)) : 0.0;
            const double stdv = std::sqrt(std::max(0.0, var));
            const double rms = (n_finite > 0) ? std::sqrt(sum_sq / static_cast<double>(n_finite)) : 0.0;

            std::ostringstream ss;
            ss.setf(std::ios::scientific);
            ss << std::setprecision(4);
            ss << "n=" << n_finite;
            if (n_total > 0) ss << "/" << n_total;
            if (n_nonfinite > 0) ss << " nonfinite=" << n_nonfinite;
            ss << " mean=" << mean;
            ss << " std=" << stdv;
            ss << " min=" << (has_any ? static_cast<double>(vmin) : 0.0);
            ss << " max=" << (has_any ? static_cast<double>(vmax) : 0.0);
            ss << " rms=" << rms;

            // Visual cue: brightness ~ rms (tanh-compressed), so you can at least see if it is ~0.
            const double t = std::tanh(rms / 3.0);
            const uint8_t p = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(t * 255.0)), 0, 255));

            Model::VizFrame vf;
            vf.w = 1;
            vf.h = 1;
            vf.channels = 1;
            vf.pixels = { p };
            vf.label = "ponyxl_sdxl/viz/stats/" + short_name + " | " + ss.str();
            addVizTapFrame(std::move(vf));
        };

        auto add_latent_frame = [&](const std::string& label, const std::vector<float>& v) {
            Model::VizFrame vf;
            vf.pixels = to_rgb_preview_latent(v, lat_h, lat_w, latent_in_dim, max_side);
            vf.w = std::max(1, (lat_w > max_side) ? (lat_w / std::max(1, (lat_w + max_side - 1) / max_side)) : lat_w);
            vf.h = std::max(1, (lat_h > max_side) ? (lat_h / std::max(1, (lat_h + max_side - 1) / max_side)) : lat_h);
            vf.channels = 3;
            vf.label = label;
            if (!vf.pixels.empty()) addVizTapFrame(std::move(vf));
        };

        auto add_img_frame = [&](const std::string& label, const std::vector<float>& img) {
            Model::VizFrame vf;
            vf.pixels = to_rgb_preview_image(img, W, H, max_side);
            vf.w = std::max(1, (W > max_side) ? (W / std::max(1, (W + max_side - 1) / max_side)) : W);
            vf.h = std::max(1, (H > max_side) ? (H / std::max(1, (H + max_side - 1) / max_side)) : H);
            vf.channels = 3;
            vf.label = label;
            if (!vf.pixels.empty()) addVizTapFrame(std::move(vf));
        };

        // Latent previews
        add_latent_frame("ponyxl_sdxl/viz/latent/x0_mu", viz_x0);
        add_latent_frame("ponyxl_sdxl/viz/latent/x_t_noised", viz_x_t);
        add_latent_frame("ponyxl_sdxl/viz/latent/eps_true", viz_eps);

        // Stats taps (useful when the preview looks uniform)
        add_stats_frame("eps_true_stats", viz_eps);

        // eps_pred is the last model output for the last inner-step.
        std::vector<float> eps_pred = last_pred;
        if (static_cast<int>(eps_pred.size()) >= latent_raw_dim) {
            eps_pred.resize(static_cast<size_t>(latent_raw_dim));

            {
                add_latent_frame("ponyxl_sdxl/viz/latent/eps_pred", eps_pred);
            }

            add_stats_frame("eps_pred_stats", eps_pred);

            // x0_hat (denoised reconstruction in latent space)
            std::vector<float> x0_hat(static_cast<size_t>(latent_raw_dim), 0.0f);
            for (int i = 0; i < latent_raw_dim; ++i) {
                x0_hat[static_cast<size_t>(i)] = (viz_x_t[static_cast<size_t>(i)] - viz_sqrt_1mab * eps_pred[static_cast<size_t>(i)]) / std::max(1e-6f, viz_sqrt_ab);
            }
            {
                add_latent_frame("ponyxl_sdxl/viz/latent/x0_hat", x0_hat);
            }

            // Residual (x0_hat - x0)
            {
                std::vector<float> r(static_cast<size_t>(latent_raw_dim), 0.0f);
                for (int i = 0; i < latent_raw_dim; ++i) {
                    r[static_cast<size_t>(i)] = x0_hat[static_cast<size_t>(i)] - viz_x0[static_cast<size_t>(i)];
                }
                add_latent_frame("ponyxl_sdxl/viz/latent/residual_x0", r);
            }

            // Image-space recon previews via VAE decoder (best-effort)
            try {
                if (!vae_decode_) {
                    using json = nlohmann::json;
                    json dec_cfg = ModelArchitectures::defaultConfig("vae_conv_decode");
                    dec_cfg["image_w"] = W;
                    dec_cfg["image_h"] = H;
                    dec_cfg["image_c"] = 3;
                    dec_cfg["latent_h"] = lat_h;
                    dec_cfg["latent_w"] = lat_w;
                    dec_cfg["latent_c"] = latent_in_dim;
                    if (cfg_.vae_base_channels > 0) {
                        dec_cfg["base_channels"] = cfg_.vae_base_channels;
                    }

                    auto m = ModelArchitectures::create("vae_conv_decode", dec_cfg);
                    if (m) {
                        m->allocateParams();
                        Mimir::Serialization::LoadOptions opts;
                        const std::string vae_ckpt = resolve_checkpoint_dir_for_loading(cfg_.vae_checkpoint);
                        opts.format = Mimir::Serialization::detect_format(vae_ckpt);
                        opts.load_tokenizer = false;
                        opts.load_encoder = false;
                        opts.load_optimizer = false;
                        opts.strict_mode = false;
                        opts.validate_checksums = false;
                        std::string err;
                        if (Mimir::Serialization::load_checkpoint(*m, vae_ckpt, opts, &err)) {
                            m->freezeParameters(true);
                            vae_decode_ = std::move(m);
                        }
                    }
                }

                if (vae_decode_) {
                    const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 1.0f;
                    std::vector<float> mu_chw;
                    std::vector<float> hat_chw;
                    if (!latent_tokens_hwc_to_chw_scaled(mu_chw, viz_x0.data(), lat_h, lat_w, latent_in_dim, inv_scale)) {
                        throw std::runtime_error("Failed to convert latent tokens->CHW for VAE decode");
                    }
                    if (!latent_tokens_hwc_to_chw_scaled(hat_chw, x0_hat.data(), lat_h, lat_w, latent_in_dim, inv_scale)) {
                        throw std::runtime_error("Failed to convert latent tokens->CHW for VAE decode");
                    }
                    const std::vector<float> img_mu = vae_decode_->forwardPass(mu_chw, false);
                    const std::vector<float> img_hat = vae_decode_->forwardPass(hat_chw, false);

                    add_img_frame("ponyxl_sdxl/viz/image/recon_mu", img_mu);
                    add_img_frame("ponyxl_sdxl/viz/image/recon_denoised", img_hat);

                    // Optional: DDPM sequence preview (multiple timesteps), throttled.
                    const int every = std::max(0, cfg_.viz_ddpm_every_steps);
                    const int nsteps = std::clamp(cfg_.viz_ddpm_num_steps, 1, 12);
                    if (every > 0 && (opt.step % every) == 0) {
                        // Reuse one fixed eps for all timesteps so progression is comparable.
                        std::vector<float> eps0(static_cast<size_t>(latent_raw_dim), 0.0f);
                        for (int i = 0; i < latent_raw_dim; ++i) {
                            eps0[static_cast<size_t>(i)] = n01(rng_);
                        }


                        // Emit a compact per-timestep sequence: x_t (decoded) + x0_hat (decoded).
                        for (int si = 0; si < nsteps; ++si) {
                            const float frac = (nsteps <= 1) ? 1.0f : (static_cast<float>(si) / static_cast<float>(nsteps - 1));
                            const int t = std::clamp(static_cast<int>(std::lround(frac * static_cast<float>(T - 1))), 0, T - 1);
                            const float t_norm = (T > 1) ? (static_cast<float>(t) / static_cast<float>(T - 1)) : 0.0f;
                            const float alpha_bar = cache_ab[static_cast<size_t>(t)];
                            const float sqrt_ab = std::sqrt(alpha_bar);
                            const float sqrt_1mab = std::sqrt(std::max(0.0f, 1.0f - alpha_bar));

                            std::vector<float> x_t_seq(static_cast<size_t>(latent_raw_dim), 0.0f);
                            for (int i = 0; i < latent_raw_dim; ++i) {
                                x_t_seq[static_cast<size_t>(i)] = sqrt_ab * viz_x0[static_cast<size_t>(i)] + sqrt_1mab * eps0[static_cast<size_t>(i)];
                            }

                            std::unordered_map<std::string, std::vector<float>> fin_seq;
                            std::unordered_map<std::string, std::vector<int>> iin_seq;
                            fin_seq.emplace("latent", x_t_seq);
                            iin_seq.emplace("text_ids", text_ids);
                            if (cfg_.sdxl_time_cond) {
                                fin_seq.emplace("timestep", std::vector<float>{compute_time_cond_value(cfg_, t_norm, alpha_bar)});
                            }

                            // We don't want extra activation/block tap spam for these preview-only forwards.
                            // Temporarily disable taps, run forward, then restore.
                            const bool prev_taps = isVizTapsEnabled();
                            setVizTapsEnabled(false);
                            const std::vector<float>& pred_view_seq = forwardPassNamedView(fin_seq, iin_seq, false);
                            setVizTapsEnabled(prev_taps);
                            std::vector<float> eps_pred_seq = pred_view_seq;
                            if (static_cast<int>(eps_pred_seq.size()) >= latent_raw_dim) {
                                eps_pred_seq.resize(static_cast<size_t>(latent_raw_dim));
                            } else {
                                continue;
                            }

                            std::vector<float> x0_hat_seq(static_cast<size_t>(latent_raw_dim), 0.0f);
                            for (int i = 0; i < latent_raw_dim; ++i) {
                                x0_hat_seq[static_cast<size_t>(i)] = (x_t_seq[static_cast<size_t>(i)] - sqrt_1mab * eps_pred_seq[static_cast<size_t>(i)]) / std::max(1e-6f, sqrt_ab);
                            }

                            std::vector<float> xt_chw;
                            std::vector<float> x0hat_chw;
                            if (!latent_tokens_hwc_to_chw_scaled(xt_chw, x_t_seq.data(), lat_h, lat_w, latent_in_dim, inv_scale)) {
                                continue;
                            }
                            if (!latent_tokens_hwc_to_chw_scaled(x0hat_chw, x0_hat_seq.data(), lat_h, lat_w, latent_in_dim, inv_scale)) {
                                continue;
                            }

                            const std::vector<float> img_xt = vae_decode_->forwardPass(xt_chw, false);
                            const std::vector<float> img_x0hat = vae_decode_->forwardPass(x0hat_chw, false);

                            const std::string ttag = "t=" + std::to_string(t) + "_" + std::to_string(static_cast<int>(std::lround(t_norm * 1000.0f)));
                            add_img_frame("ponyxl_sdxl/viz/ddpm_seq/" + ttag + "/x_t_noised", img_xt);
                            add_img_frame("ponyxl_sdxl/viz/ddpm_seq/" + ttag + "/x0_hat_denoised", img_x0hat);
                        }
                    }
                }
            } catch (...) {
            }

            // Scalar timestep preview as a tiny 1x1 grayscale frame (helps correlate with denoise strength)
            {
                Model::VizFrame vf;
                vf.w = 1;
                vf.h = 1;
                vf.channels = 1;
                vf.label = "ponyxl_sdxl/viz/meta/t_norm";
                const uint8_t p = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(std::clamp(viz_tnorm, 0.0f, 1.0f) * 255.0f)), 0, 255));
                vf.pixels = {p};
                addVizTapFrame(std::move(vf));
            }

            // Scalar training metrics (loss/lr/grad + distribution/coherence) as tiny 1x1 frames.
            {
                auto add_scalar = [&](const std::string& label, double value, double scale, bool signed_mode, bool log_compress) {
                    std::ostringstream ss;
                    ss.setf(std::ios::scientific);
                    ss << std::setprecision(6);
                    if (!std::isfinite(value)) {
                        ss << "value=nan";
                        Model::VizFrame vf;
                        vf.w = 1;
                        vf.h = 1;
                        vf.channels = 1;
                        vf.pixels = {0};
                        vf.label = label + " | " + ss.str();
                        addVizTapFrame(std::move(vf));
                        return;
                    }
                    ss << "value=" << value;

                    const double s = (scale > 0.0) ? scale : 1.0;
                    double v = value;
                    if (!signed_mode) v = std::max(0.0, v);
                    if (log_compress) v = std::log1p(std::max(0.0, std::abs(v)));

                    double t01 = 0.0;
                    if (signed_mode) {
                        t01 = 0.5 + 0.5 * std::tanh(v / s);
                    } else {
                        t01 = std::tanh(v / s);
                    }
                    const int p = static_cast<int>(std::lround(std::clamp(t01, 0.0, 1.0) * 255.0));

                    Model::VizFrame vf;
                    vf.w = 1;
                    vf.h = 1;
                    vf.channels = 1;
                    vf.pixels = {static_cast<uint8_t>(std::clamp(p, 0, 255))};
                    vf.label = label + " | " + ss.str();
                    addVizTapFrame(std::move(vf));
                };

                // Losses + optimizer
                add_scalar("ponyxl_sdxl/viz/train/metrics/loss_last", static_cast<double>(last_loss), 1.0, /*signed*/false, /*log*/false);
                add_scalar("ponyxl_sdxl/viz/train/metrics/loss_avg", static_cast<double>(loss_sum / static_cast<double>(std::max(1, steps_per_image))), 1.0, /*signed*/false, /*log*/false);
                add_scalar("ponyxl_sdxl/viz/train/metrics/lr", static_cast<double>(learning_rate), 1.0, /*signed*/false, /*log*/true);

                // Grad stats (log-compressed because magnitudes vary a lot)
                add_scalar("ponyxl_sdxl/viz/train/metrics/grad_norm", static_cast<double>(grad_sum), 1.0, /*signed*/false, /*log*/true);
                add_scalar("ponyxl_sdxl/viz/train/metrics/grad_max_abs", static_cast<double>(grad_max), 1.0, /*signed*/false, /*log*/true);

                // DDPM timestep (duplicate of meta/t_norm but placed under metrics)
                add_scalar("ponyxl_sdxl/viz/train/metrics/t_norm", static_cast<double>(viz_tnorm), 1.0, /*signed*/false, /*log*/false);

                // Distribution/coherence diagnostics on last_pred vs last_eps (best-effort)
                const auto mp = compute_moments_local(last_pred);
                const auto mt = compute_moments_local(last_eps);
                const double vp_raw = std::max(mp.var, 1e-12);
                const double vt = std::max(mt.var, 1e-12);

                // KL prior (eps_pred -> N(0,1)), same term as training regularizer.
                double lv = std::log(vp_raw);
                lv = std::clamp(lv, static_cast<double>(logvar_min), static_cast<double>(logvar_max));
                const double vp = std::exp(lv);
                const double kl_prior = 0.5 * (mp.mean * mp.mean + vp - 1.0 - lv);

                // Other diagnostics remain pred vs target (eps).
                const double w2 = (mt.mean - mp.mean) * (mt.mean - mp.mean) +
                                  (std::sqrt(vt) - std::sqrt(vp_raw)) * (std::sqrt(vt) - std::sqrt(vp_raw));
                constexpr double kPi = 3.14159265358979323846;
                constexpr double kE = 2.71828182845904523536;
                const double Hp = 0.5 * std::log(2.0 * kPi * kE * vp_raw);
                const double Ht = 0.5 * std::log(2.0 * kPi * kE * vt);
                const double tvp = mean_abs_adjacent_diff_local(last_pred);
                const double tvt = mean_abs_adjacent_diff_local(last_eps);
                const double corr = pearson_corr_local(last_pred, last_eps);

                add_scalar("ponyxl_sdxl/viz/train/metrics/kl_divergence", std::max(0.0, kl_prior), 1.0, /*signed*/false, /*log*/true);
                add_scalar("ponyxl_sdxl/viz/train/metrics/wasserstein", std::sqrt(std::max(0.0, w2)), 1.0, /*signed*/false, /*log*/true);
                add_scalar("ponyxl_sdxl/viz/train/metrics/entropy_diff", (Hp - Ht), 1.0, /*signed*/true, /*log*/false);
                add_scalar("ponyxl_sdxl/viz/train/metrics/moment_mismatch", std::abs(mp.skew - mt.skew), 1.0, /*signed*/false, /*log*/true);
                add_scalar("ponyxl_sdxl/viz/train/metrics/spatial_coherence", std::abs(tvp - tvt), 1.0, /*signed*/false, /*log*/true);
                add_scalar("ponyxl_sdxl/viz/train/metrics/temporal_consistency", corr, 1.0, /*signed*/true, /*log*/false);
            }

        // Restore the previous taps state so the next step can still emit custom frames.
        // (We still disable taps during the heavy forward/backward at the start of the step.)
        setVizTapsEnabled(prev_viz_taps_enabled);
        }
    }

    StepStats out;
    out.loss = static_cast<float>(loss_sum / static_cast<double>(steps_per_image));
    out.grad_norm = static_cast<float>(grad_sum);
    out.grad_max_abs = grad_max;
    out.timestep = cfg_.sdxl_time_cond ? last_t_norm : 0.0f;
    out.kl_beta_effective = kl_beta_eff;

    // Divergence/coherence metrics (best-effort) on last prediction vs target.
    {
        const auto mp = compute_moments_local(last_pred);
        const auto mt = compute_moments_local(last_eps);
        const double vp = std::max(mp.var, 1e-12);
        const double vt = std::max(mt.var, 1e-12);

        // Metric: report the KL prior term actually used for regularization (eps_pred -> N(0,1)).
        // Use the last computed value to make it step-aligned with other diagnostics.
        // (If kl_beta==0, this stays at 0.)
        out.kl_divergence = last_kl;

        // Wasserstein-2 (1D gaussians)
        const double w2 = (mt.mean - mp.mean) * (mt.mean - mp.mean) +
                          (std::sqrt(vt) - std::sqrt(vp)) * (std::sqrt(vt) - std::sqrt(vp));
        out.wasserstein = static_cast<float>(std::sqrt(std::max(0.0, w2)));

        // Entropy diff
        constexpr double kPi = 3.14159265358979323846;
        constexpr double kE = 2.71828182845904523536;
        const double Hp = 0.5 * std::log(2.0 * kPi * kE * vp);
        const double Ht = 0.5 * std::log(2.0 * kPi * kE * vt);
        out.entropy_diff = static_cast<float>(Hp - Ht);

        // Moment mismatch (skew)
        out.moment_mismatch = static_cast<float>(std::abs(mp.skew - mt.skew));

        // Spatial coherence proxy (1D TV)
        const double tvp = mean_abs_adjacent_diff_local(last_pred);
        const double tvt = mean_abs_adjacent_diff_local(last_eps);
        out.spatial_coherence = static_cast<float>(std::abs(tvp - tvt));

        // Temporal consistency proxy (Pearson corr)
        out.temporal_consistency = static_cast<float>(pearson_corr_local(last_pred, last_eps));
    }
    return out;
}

PonyXLDDPMModel::ValStats PonyXLDDPMModel::validateStepSdxlLatentDiffusion(
    const std::string& prompt,
    const std::string& wrong_prompt,
    const std::vector<uint8_t>& rgb,
    int w,
    int h,
    int seed,
    int ddpm_step
) {
    if (layers.empty()) {
        throw std::runtime_error("PonyXLDDPMModel::validateStepSdxlLatentDiffusion: model not built");
    }

    const int W = std::max(1, w);
    const int H = std::max(1, h);
    const int text_len = std::max(1, cfg_.text_ctx_len);
    const int latent_len = std::max(1, cfg_.latent_seq_len);
    const int latent_in_dim = std::max(1, cfg_.latent_in_dim);
    const int latent_raw_dim = latent_len * latent_in_dim;
    const int max_vocab = std::max(1, cfg_.max_vocab);

    auto make_text_ids = [&](const std::string& p) -> std::vector<int> {
        std::string used = sanitize_utf8(p);
        if (cfg_.max_text_chars > 0 && used.size() > static_cast<size_t>(cfg_.max_text_chars)) {
            used = truncate_utf8_safe(used, static_cast<size_t>(cfg_.max_text_chars));
        }

        if (cfg_.caption_kv_enable) {
            KvCaption kv = parse_kv_caption(used);
            if (kv.has_any) {
                used = compose_kv_caption_prompt(kv);
                if (cfg_.max_text_chars > 0 && used.size() > static_cast<size_t>(cfg_.max_text_chars)) {
                    used = truncate_utf8_safe(used, static_cast<size_t>(cfg_.max_text_chars));
                }
            }
        }

        if (cfg_.caption_kv_enable) {
            KvCaption kv = parse_kv_caption(used);
            if (kv.has_any) {
                used = compose_kv_caption_prompt(kv);
                if (cfg_.max_text_chars > 0 && used.size() > static_cast<size_t>(cfg_.max_text_chars)) {
                    used = truncate_utf8_safe(used, static_cast<size_t>(cfg_.max_text_chars));
                }
            }
        }

        if (cfg_.caption_structured_enable) {
            StructuredCaption cap = parse_structured_caption(used);
            if (cap.has_any_header) {
                const std::string composed = compose_structured_caption(cap, cfg_.caption_structured_canonicalize);
                if (!composed.empty()) {
                    used = composed;
                    if (cfg_.max_text_chars > 0 && used.size() > static_cast<size_t>(cfg_.max_text_chars)) {
                        used = truncate_utf8_safe(used, static_cast<size_t>(cfg_.max_text_chars));
                    }
                }
            }
        }

        std::vector<int> ids;
        if (!used.empty()) ids = getMutableTokenizer().tokenizeEnsure(used);

        // Tokens de contexte global (appris), injectés après BOS.
        std::vector<int> gctx_ids;
        {
            const int want = std::max(0, cfg_.global_ctx_tokens);
            gctx_ids.reserve(static_cast<size_t>(want));
            for (int i = 0; i < want; ++i) {
                gctx_ids.push_back(getMutableTokenizer().addToken("<GCTX" + std::to_string(i) + ">"));
            }
        }

        const int pad_id = getMutableTokenizer().getPadId();
        const int unk_id = getMutableTokenizer().getUnkId();
        const int bos = getMutableTokenizer().getBosId();
        const int eos = getMutableTokenizer().getEosId();

        if (text_len >= 2 && bos >= 0 && eos >= 0) {
            size_t start = 0;
            size_t end = ids.size();
            if (start < end && ids[start] == bos) ++start;
            if (end > start && ids[end - 1] == eos) --end;

            std::vector<int> content;
            if (end > start) {
                content.assign(ids.begin() + static_cast<std::ptrdiff_t>(start),
                               ids.begin() + static_cast<std::ptrdiff_t>(end));
            }

            const int cap = std::max(0, text_len - 2 - static_cast<int>(gctx_ids.size()));
            if (static_cast<int>(content.size()) > cap) {
                content.resize(static_cast<size_t>(cap));
            }

            ids.clear();
            ids.reserve(static_cast<size_t>(text_len));
            ids.push_back(bos);
            ids.insert(ids.end(), gctx_ids.begin(), gctx_ids.end());
            ids.insert(ids.end(), content.begin(), content.end());
            while (static_cast<int>(ids.size()) < (text_len - 1)) {
                ids.push_back(pad_id);
            }
            ids.push_back(eos);
        } else {
            if (ids.size() > static_cast<size_t>(text_len)) {
                ids.resize(static_cast<size_t>(text_len));
            } else if (ids.size() < static_cast<size_t>(text_len)) {
                ids.resize(static_cast<size_t>(text_len), pad_id);
            }
        }

        for (auto& id : ids) {
            if (id == pad_id) continue;
            if (id < 0) id = pad_id;
            if (id >= max_vocab) id = (unk_id >= 0) ? unk_id : pad_id;
        }
        return ids;
    };

    const std::vector<int> text_ids = make_text_ids(prompt);
    const std::vector<int> wrong_ids = make_text_ids(wrong_prompt);

    if (cfg_.vae_checkpoint.empty()) {
        throw std::runtime_error("PonyXLDDPMModel: vae_checkpoint is required for validation");
    }

    // Ensure VAE is loaded (same as training)
    if (!vae_) {
        using json = nlohmann::json;
        json vae_cfg = ModelArchitectures::defaultConfig(cfg_.vae_arch);
        vae_cfg["image_w"] = W;
        vae_cfg["image_h"] = H;
        vae_cfg["image_c"] = 3;

        int lh = cfg_.latent_h;
        int lw = cfg_.latent_w;
        if (lh <= 0 && lw <= 0) {
            lh = 1;
            lw = latent_len;
        } else if (lh <= 0 && lw > 0) {
            lh = (latent_len % lw == 0) ? (latent_len / lw) : 1;
        } else if (lh > 0 && lw <= 0) {
            lw = (latent_len % lh == 0) ? (latent_len / lh) : latent_len;
        }
        lh = std::max(1, lh);
        lw = std::max(1, lw);
        vae_cfg["latent_h"] = lh;
        vae_cfg["latent_w"] = lw;
        vae_cfg["latent_c"] = latent_in_dim;
        if (cfg_.vae_base_channels > 0) {
            vae_cfg["base_channels"] = cfg_.vae_base_channels;
        }

        auto m = ModelArchitectures::create(cfg_.vae_arch, vae_cfg);
        if (!m) {
            throw std::runtime_error("Failed to create VAE model: " + cfg_.vae_arch);
        }
        m->allocateParams();

        Mimir::Serialization::LoadOptions opts;
        const std::string vae_ckpt = resolve_checkpoint_dir_for_loading(cfg_.vae_checkpoint);
        opts.format = Mimir::Serialization::detect_format(vae_ckpt);
        opts.load_tokenizer = false;
        opts.load_encoder = false;
        opts.load_optimizer = false;
        opts.strict_mode = false;
        opts.validate_checksums = false;

        std::string err;
        if (!Mimir::Serialization::load_checkpoint(*m, vae_ckpt, opts, &err)) {
            throw std::runtime_error("Failed to load VAE checkpoint: " + cfg_.vae_checkpoint + " (resolved=" + vae_ckpt + ") | " + err);
        }
        m->freezeParameters(true);
        vae_ = std::move(m);
    }

    // Encode x0
    const std::vector<float> img_f = imageBytesToFloatRGB(rgb, W, H);
    const std::vector<float> packed = vae_->forwardPass(img_f, false);

    int vae_image_dim = W * H * 3;
    int vae_latent_dim = 0;
    try {
        if (vae_->modelConfig.contains("latent_dim")) {
            vae_latent_dim = vae_->modelConfig["latent_dim"].get<int>();
        } else if (vae_->modelConfig.contains("latent_h") && vae_->modelConfig.contains("latent_w") && vae_->modelConfig.contains("latent_c")) {
            vae_latent_dim = vae_->modelConfig["latent_h"].get<int>() * vae_->modelConfig["latent_w"].get<int>() * vae_->modelConfig["latent_c"].get<int>();
        }
        if (vae_->modelConfig.contains("image_dim")) {
            vae_image_dim = vae_->modelConfig["image_dim"].get<int>();
        }
    } catch (...) {
    }

    if (vae_latent_dim <= 0 || vae_latent_dim != latent_raw_dim) {
        throw std::runtime_error("VAE latent_dim mismatch during validation");
    }
    if (static_cast<int>(packed.size()) < (vae_image_dim + 2 * vae_latent_dim)) {
        throw std::runtime_error("VAE packed output too small (validation)");
    }

    int vae_lh = cfg_.latent_h;
    int vae_lw = cfg_.latent_w;
    int vae_lc = latent_in_dim;
    try {
        if (vae_->modelConfig.contains("latent_h")) vae_lh = vae_->modelConfig["latent_h"].get<int>();
        if (vae_->modelConfig.contains("latent_w")) vae_lw = vae_->modelConfig["latent_w"].get<int>();
        if (vae_->modelConfig.contains("latent_c")) vae_lc = vae_->modelConfig["latent_c"].get<int>();
    } catch (...) {
    }
    vae_lh = std::max(1, vae_lh);
    vae_lw = std::max(1, vae_lw);
    vae_lc = std::max(1, vae_lc);
    if (vae_lc != latent_in_dim || (vae_lh * vae_lw) != latent_len) {
        throw std::runtime_error("VAE latent shape mismatch during validation");
    }

    std::vector<float> x0;
    const float scale = std::max(0.0f, cfg_.vae_scale);
    const size_t mu_off = static_cast<size_t>(vae_image_dim);
    if (!latent_chw_to_tokens_hwc_scaled(x0, packed.data() + mu_off, vae_lh, vae_lw, vae_lc, scale)) {
        throw std::runtime_error("Failed to convert VAE mu (CHW) to PonyXL latent (tokens HWC) during validation");
    }

    // Latent H/W (pour previews/viz)
    int lat_h = vae_lh;
    int lat_w = vae_lw;
    if (lat_h <= 0 && lat_w <= 0) {
        const int s = static_cast<int>(std::sqrt(static_cast<double>(latent_len)));
        if (s > 0 && (s * s) == latent_len) {
            lat_h = s;
            lat_w = s;
        } else {
            lat_h = 1;
            lat_w = latent_len;
        }
    } else if (lat_h <= 0 && lat_w > 0) {
        lat_h = (latent_len % lat_w == 0) ? (latent_len / lat_w) : 1;
    } else if (lat_h > 0 && lat_w <= 0) {
        lat_w = (latent_len % lat_h == 0) ? (latent_len / lat_h) : latent_len;
    }
    lat_h = std::max(1, lat_h);
    lat_w = std::max(1, lat_w);

    const int T = std::max(2, cfg_.ddpm_steps);
    const int t = (ddpm_step >= 0) ? std::clamp(ddpm_step, 0, T - 1) : (T / 2);
    const float t_norm = (T > 1) ? (static_cast<float>(t) / static_cast<float>(T - 1)) : 0.0f;

    const float beta0 = std::clamp(cfg_.ddpm_beta_start, 0.0f, 0.999f);
    const float beta1 = std::clamp(cfg_.ddpm_beta_end, 0.0f, 0.999f);
    static thread_local int cache_T = 0;
    static thread_local float cache_b0 = -1.0f;
    static thread_local float cache_b1 = -1.0f;
    static thread_local std::vector<float> cache_ab;
    if (cache_T != T || cache_b0 != beta0 || cache_b1 != beta1 || static_cast<int>(cache_ab.size()) != T) {
        cache_T = T;
        cache_b0 = beta0;
        cache_b1 = beta1;
        cache_ab.assign(static_cast<size_t>(T), 1.0f);
        float ab = 1.0f;
        for (int i = 0; i < T; ++i) {
            const float frac = (T > 1) ? (static_cast<float>(i) / static_cast<float>(T - 1)) : 0.0f;
            const float beta = std::clamp(beta0 + (beta1 - beta0) * frac, 0.0f, 0.999f);
            const float alpha = 1.0f - beta;
            ab *= alpha;
            cache_ab[static_cast<size_t>(i)] = std::clamp(ab, 1e-6f, 1.0f);
        }
    }

    const float alpha_bar = cache_ab[static_cast<size_t>(t)];
    const float sqrt_ab = std::sqrt(alpha_bar);
    const float sqrt_1mab = std::sqrt(std::max(0.0f, 1.0f - alpha_bar));

    // Deterministic eps for validation
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::normal_distribution<float> n01(0.0f, 1.0f);
    std::vector<float> eps(static_cast<size_t>(latent_raw_dim), 0.0f);
    for (int i = 0; i < latent_raw_dim; ++i) {
        eps[static_cast<size_t>(i)] = n01(rng);
    }

    std::vector<float> x_t(static_cast<size_t>(latent_raw_dim), 0.0f);
    for (int i = 0; i < latent_raw_dim; ++i) {
        const float e = eps[static_cast<size_t>(i)];
        x_t[static_cast<size_t>(i)] = sqrt_ab * x0[static_cast<size_t>(i)] + sqrt_1mab * e;
    }

    // Viz taps: avoid generic activation spam during validation forwards.
    const bool prev_viz_taps_enabled = isVizTapsEnabled();
    if (prev_viz_taps_enabled) {
        setVizTapsEnabled(false);
    }

    auto run_pred = [&](const std::vector<int>& ids) -> std::vector<float> {
        std::unordered_map<std::string, std::vector<float>> fin;
        std::unordered_map<std::string, std::vector<int>> iin;
        fin["latent"] = x_t;
        iin["text_ids"] = ids;
        if (cfg_.sdxl_time_cond) {
            fin["timestep"] = std::vector<float>{compute_time_cond_value(cfg_, t_norm, alpha_bar)};
        }
        return forwardPassNamed(fin, iin, false);
    };

    const std::vector<float> pred = run_pred(text_ids);
    const std::vector<float> pred_wrong = run_pred(wrong_ids);

    const int n = std::min<int>(latent_raw_dim, static_cast<int>(pred.size()));
    const int nw = std::min<int>(latent_raw_dim, static_cast<int>(pred_wrong.size()));

    auto mse = [&](const std::vector<float>& a, const std::vector<float>& b, int count) -> double {
        if (count <= 0) return 0.0;
        double s = 0.0;
        for (int i = 0; i < count; ++i) {
            const double d = static_cast<double>(a[static_cast<size_t>(i)]) - static_cast<double>(b[static_cast<size_t>(i)]);
            s += d * d;
        }
        return s / static_cast<double>(count);
    };

    ValStats out;
    out.t_norm = t_norm;
    out.eps_mse = mse(pred, eps, n);
    out.eps_mse_wrong = mse(pred_wrong, eps, nw);
    out.assoc_margin = out.eps_mse_wrong - out.eps_mse;

    // x0_hat from eps_pred
    if (n > 0) {
        std::vector<float> x0_hat(static_cast<size_t>(n), 0.0f);
        for (int i = 0; i < n; ++i) {
            const float ehat = pred[static_cast<size_t>(i)];
            x0_hat[static_cast<size_t>(i)] = (x_t[static_cast<size_t>(i)] - sqrt_1mab * ehat) / std::max(1e-6f, sqrt_ab);
        }
        out.x0_mse = mse(x0_hat, x0, n);

        // Optional: wrong-prompt denoise reconstruction
        std::vector<float> x0_hat_wrong;
        if (nw > 0) {
            const int cnt = std::min(n, nw);
            x0_hat_wrong.assign(static_cast<size_t>(cnt), 0.0f);
            for (int i = 0; i < cnt; ++i) {
                const float ehat = pred_wrong[static_cast<size_t>(i)];
                x0_hat_wrong[static_cast<size_t>(i)] = (x_t[static_cast<size_t>(i)] - sqrt_1mab * ehat) / std::max(1e-6f, sqrt_ab);
            }
        }

        // Image-space reconstruction (best-effort): decode x0_hat via VAE decoder and compare
        // to the original image in [-1,1].
        try {
            if (!vae_decode_) {
                using json = nlohmann::json;
                json dec_cfg = ModelArchitectures::defaultConfig("vae_conv_decode");
                dec_cfg["image_w"] = W;
                dec_cfg["image_h"] = H;
                dec_cfg["image_c"] = 3;
                dec_cfg["latent_h"] = vae_lh;
                dec_cfg["latent_w"] = vae_lw;
                dec_cfg["latent_c"] = latent_in_dim;
                if (cfg_.vae_base_channels > 0) {
                    dec_cfg["base_channels"] = cfg_.vae_base_channels;
                }

                auto m = ModelArchitectures::create("vae_conv_decode", dec_cfg);
                if (m) {
                    m->allocateParams();
                    Mimir::Serialization::LoadOptions opts;
                    const std::string vae_ckpt = resolve_checkpoint_dir_for_loading(cfg_.vae_checkpoint);
                    opts.format = Mimir::Serialization::detect_format(vae_ckpt);
                    opts.load_tokenizer = false;
                    opts.load_encoder = false;
                    opts.load_optimizer = false;
                    opts.strict_mode = false;
                    opts.validate_checksums = false;
                    std::string err;
                    if (!Mimir::Serialization::load_checkpoint(*m, vae_ckpt, opts, &err)) {
                        throw std::runtime_error("Failed to load VAE decoder checkpoint: " + cfg_.vae_checkpoint + " (resolved=" + vae_ckpt + ") | " + err);
                    }
                    m->freezeParameters(true);
                    vae_decode_ = std::move(m);
                }
            }

            if (vae_decode_) {
                const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 1.0f;
                std::vector<float> hat_chw;
                if (latent_tokens_hwc_to_chw_scaled(hat_chw, x0_hat.data(), vae_lh, vae_lw, latent_in_dim, inv_scale)) {
                    const std::vector<float> img_hat = vae_decode_->forwardPass(hat_chw, false);
                    const int im_n = std::min<int>((int)img_hat.size(), (int)img_f.size());
                    if (im_n > 0) {
                        out.img_mse = mse(img_hat, img_f, im_n);
                    }
                }
            }
        } catch (...) {
            // Best-effort only.
        }

        // Custom Viz frames (denoise + text-image assignment proxy) for validation.
        if (prev_viz_taps_enabled) {
            setVizTapsEnabled(true);
            const int ms = getVizTapsMaxSide();

            auto add_lat = [&](const std::string& label, const std::vector<float>& v, int hh, int ww, int cc, const float* override_ptr = nullptr) {
                Model::VizFrame vf;
                vf.pixels = to_rgb_preview_latent(v, hh, ww, cc, ms, override_ptr);
                const int sx = (ww > ms) ? static_cast<int>((ww + ms - 1) / ms) : 1;
                const int sy = (hh > ms) ? static_cast<int>((hh + ms - 1) / ms) : 1;
                vf.w = std::max(1, ww / std::max(1, sx));
                vf.h = std::max(1, hh / std::max(1, sy));
                vf.channels = (cc >= 3) ? 3 : 1;
                vf.label = label;
                if (!vf.pixels.empty()) addVizTapFrame(std::move(vf));
            };

            auto add_stats = [&](const std::string& short_name, const float* ptr, size_t len) {
                size_t n_total = len;
                size_t n_finite = 0;
                size_t n_nonfinite = 0;

                // Welford
                double mean = 0.0;
                double m2 = 0.0;
                double sum_sq = 0.0;
                float vmin = 0.0f;
                float vmax = 0.0f;
                bool has_any = false;

                for (size_t i = 0; i < len; ++i) {
                    const float x = ptr ? ptr[i] : 0.0f;
                    if (!std::isfinite(x)) {
                        n_nonfinite++;
                        continue;
                    }
                    n_finite++;
                    sum_sq += static_cast<double>(x) * static_cast<double>(x);
                    if (!has_any) {
                        vmin = x;
                        vmax = x;
                        has_any = true;
                    } else {
                        vmin = std::min(vmin, x);
                        vmax = std::max(vmax, x);
                    }
                    const double dx = static_cast<double>(x) - mean;
                    mean += dx / static_cast<double>(n_finite);
                    const double dx2 = static_cast<double>(x) - mean;
                    m2 += dx * dx2;
                }

                const double var = (n_finite > 1) ? (m2 / static_cast<double>(n_finite - 1)) : 0.0;
                const double stdv = std::sqrt(std::max(0.0, var));
                const double rms = (n_finite > 0) ? std::sqrt(sum_sq / static_cast<double>(n_finite)) : 0.0;

                std::ostringstream ss;
                ss.setf(std::ios::scientific);
                ss << std::setprecision(4);
                ss << "n=" << n_finite;
                if (n_total > 0) ss << "/" << n_total;
                if (n_nonfinite > 0) ss << " nonfinite=" << n_nonfinite;
                ss << " mean=" << mean;
                ss << " std=" << stdv;
                ss << " min=" << (has_any ? static_cast<double>(vmin) : 0.0);
                ss << " max=" << (has_any ? static_cast<double>(vmax) : 0.0);
                ss << " rms=" << rms;

                const double t = std::tanh(rms / 3.0);
                const uint8_t p = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(t * 255.0)), 0, 255));

                Model::VizFrame vf;
                vf.w = 1;
                vf.h = 1;
                vf.channels = 1;
                vf.pixels = { p };
                vf.label = "ponyxl_sdxl/viz/val/stats/" + short_name + " | " + ss.str();
                addVizTapFrame(std::move(vf));
            };

            // Denoise signals at timestep t
            add_lat("ponyxl_sdxl/viz/val/latent/x_t_noised", x_t, lat_h, lat_w, latent_in_dim);
            add_lat("ponyxl_sdxl/viz/val/latent/eps_true", eps, lat_h, lat_w, latent_in_dim);
            add_stats("eps_true_stats", eps.data(), eps.size());
            if (n > 0) {
                // pred/pred_wrong are flat, but we know they contain at least latent_raw_dim.
                add_lat("ponyxl_sdxl/viz/val/latent/eps_pred", pred, lat_h, lat_w, latent_in_dim);
                add_stats("eps_pred_stats", pred.data(), static_cast<size_t>(std::max(0, n)));
            }
            if (nw > 0) {
                add_lat("ponyxl_sdxl/viz/val/latent/eps_pred_wrong", pred_wrong, lat_h, lat_w, latent_in_dim);
                add_stats("eps_pred_wrong_stats", pred_wrong.data(), static_cast<size_t>(std::max(0, nw)));
            }

            // Text influence map: per-token norm(|eps_pred - eps_pred_wrong|)
            if (n > 0 && nw > 0 && (lat_h * lat_w) == latent_len) {
                const int cnt = std::min(n, nw);
                std::vector<float> diff_field;
                diff_field.assign(static_cast<size_t>(latent_len), 0.0f);
                for (int tok = 0; tok < latent_len; ++tok) {
                    double acc = 0.0;
                    const int base = tok * latent_in_dim;
                    for (int cc = 0; cc < latent_in_dim; ++cc) {
                        const int i = base + cc;
                        if (i >= cnt) break;
                        const double d = static_cast<double>(pred[static_cast<size_t>(i)]) - static_cast<double>(pred_wrong[static_cast<size_t>(i)]);
                        acc += d * d;
                    }
                    diff_field[static_cast<size_t>(tok)] = static_cast<float>(std::sqrt(std::max(0.0, acc)));
                }
                add_lat("ponyxl_sdxl/viz/val/text_image/eps_pred_diff_map", diff_field, lat_h, lat_w, 1);
            }

            // Scalar marker: assoc_margin encoded as a tiny 1x1 grayscale frame.
            {
                auto add_scalar = [&](const std::string& label, double value, double scale, bool signed_mode, bool log_compress) {
                    std::ostringstream ss;
                    ss.setf(std::ios::scientific);
                    ss << std::setprecision(6);
                    if (!std::isfinite(value)) {
                        ss << "value=nan";
                        Model::VizFrame vf;
                        vf.w = 1;
                        vf.h = 1;
                        vf.channels = 1;
                        vf.pixels = {0};
                        vf.label = label + " | " + ss.str();
                        addVizTapFrame(std::move(vf));
                        return;
                    }
                    ss << "value=" << value;

                    const double s = (scale > 0.0) ? scale : 1.0;
                    double v = value;
                    if (!signed_mode) v = std::max(0.0, v);
                    if (log_compress) v = std::log1p(std::max(0.0, std::abs(v)));

                    double t01 = 0.0;
                    if (signed_mode) {
                        t01 = 0.5 + 0.5 * std::tanh(v / s);
                    } else {
                        t01 = std::tanh(v / s);
                    }
                    const int p = static_cast<int>(std::lround(std::clamp(t01, 0.0, 1.0) * 255.0));

                    Model::VizFrame vf;
                    vf.w = 1;
                    vf.h = 1;
                    vf.channels = 1;
                    vf.pixels = {static_cast<uint8_t>(std::clamp(p, 0, 255))};
                    vf.label = label + " | " + ss.str();
                    addVizTapFrame(std::move(vf));
                };

                // Core validation metrics
                add_scalar("ponyxl_sdxl/viz/val/metrics/eps_mse", static_cast<double>(out.eps_mse), 1.0, /*signed*/false, /*log*/true);
                add_scalar("ponyxl_sdxl/viz/val/metrics/x0_mse", static_cast<double>(out.x0_mse), 1.0, /*signed*/false, /*log*/true);
                add_scalar("ponyxl_sdxl/viz/val/metrics/img_mse", static_cast<double>(out.img_mse), 1.0, /*signed*/false, /*log*/true);
                add_scalar("ponyxl_sdxl/viz/val/metrics/eps_mse_wrong", static_cast<double>(out.eps_mse_wrong), 1.0, /*signed*/false, /*log*/true);

                Model::VizFrame vf;
                vf.w = 1;
                vf.h = 1;
                vf.channels = 1;
                {
                    std::ostringstream ss;
                    ss.setf(std::ios::scientific);
                    ss << std::setprecision(6);
                    ss << "value=" << static_cast<double>(out.assoc_margin);
                    vf.label = "ponyxl_sdxl/viz/val/text_image/assoc_margin" + std::string(" | ") + ss.str();
                }
                const float s = static_cast<float>(out.assoc_margin);
                const float t01 = 0.5f + 0.5f * std::tanh(s);
                const int p = static_cast<int>(std::lround(std::clamp(t01, 0.0f, 1.0f) * 255.0f));
                vf.pixels = {static_cast<uint8_t>(std::clamp(p, 0, 255))};
                addVizTapFrame(std::move(vf));
            }
            {
                Model::VizFrame vf;
                vf.w = 1;
                vf.h = 1;
                vf.channels = 1;
                vf.label = "ponyxl_sdxl/viz/val/meta/t_norm";
                const uint8_t p = static_cast<uint8_t>(std::clamp(static_cast<int>(std::lround(std::clamp(t_norm, 0.0f, 1.0f) * 255.0f)), 0, 255));
                vf.pixels = {p};
                addVizTapFrame(std::move(vf));
            }

            // Restore previous taps state.
            setVizTapsEnabled(prev_viz_taps_enabled);
        }
    }

    // Restore taps state if we disabled it before forwards.
    if (prev_viz_taps_enabled) {
        setVizTapsEnabled(prev_viz_taps_enabled);
    }
    return out;
}

PonyXLDDPMModel::ReconPreview PonyXLDDPMModel::reconstructPreviewSdxlLatentDiffusion(
    const std::string& prompt,
    const std::vector<uint8_t>& rgb,
    int w,
    int h,
    int max_side,
    int seed,
    int ddpm_step
) {
    if (layers.empty()) {
        throw std::runtime_error("PonyXLDDPMModel::reconstructPreviewSdxlLatentDiffusion: model not built");
    }

    const int W = std::max(1, w);
    const int H = std::max(1, h);
    const int text_len = std::max(1, cfg_.text_ctx_len);
    const int latent_len = std::max(1, cfg_.latent_seq_len);
    const int latent_in_dim = std::max(1, cfg_.latent_in_dim);
    const int latent_raw_dim = latent_len * latent_in_dim;
    const int max_vocab = std::max(1, cfg_.max_vocab);

    auto make_text_ids = [&](const std::string& p) -> std::vector<int> {
        std::string used = sanitize_utf8(p);
        if (cfg_.max_text_chars > 0 && used.size() > static_cast<size_t>(cfg_.max_text_chars)) {
            used = truncate_utf8_safe(used, static_cast<size_t>(cfg_.max_text_chars));
        }

        if (cfg_.caption_structured_enable) {
            StructuredCaption cap = parse_structured_caption(used);
            if (cap.has_any_header) {
                const std::string composed = compose_structured_caption(cap, cfg_.caption_structured_canonicalize);
                if (!composed.empty()) {
                    used = composed;
                    if (cfg_.max_text_chars > 0 && used.size() > static_cast<size_t>(cfg_.max_text_chars)) {
                        used = truncate_utf8_safe(used, static_cast<size_t>(cfg_.max_text_chars));
                    }
                }
            }
        }

        std::vector<int> ids;
        if (!used.empty()) ids = getMutableTokenizer().tokenizeEnsure(used);

        // Tokens de contexte global (appris), injectés après BOS.
        std::vector<int> gctx_ids;
        {
            const int want = std::max(0, cfg_.global_ctx_tokens);
            gctx_ids.reserve(static_cast<size_t>(want));
            for (int i = 0; i < want; ++i) {
                gctx_ids.push_back(getMutableTokenizer().addToken("<GCTX" + std::to_string(i) + ">"));
            }
        }

        const int pad_id = getMutableTokenizer().getPadId();
        const int unk_id = getMutableTokenizer().getUnkId();
        const int bos = getMutableTokenizer().getBosId();
        const int eos = getMutableTokenizer().getEosId();

        if (text_len >= 2 && bos >= 0 && eos >= 0) {
            size_t start = 0;
            size_t end = ids.size();
            if (start < end && ids[start] == bos) ++start;
            if (end > start && ids[end - 1] == eos) --end;

            std::vector<int> content;
            if (end > start) {
                content.assign(ids.begin() + static_cast<std::ptrdiff_t>(start),
                               ids.begin() + static_cast<std::ptrdiff_t>(end));
            }

            const int cap = std::max(0, text_len - 2 - static_cast<int>(gctx_ids.size()));
            if (static_cast<int>(content.size()) > cap) {
                content.resize(static_cast<size_t>(cap));
            }

            ids.clear();
            ids.reserve(static_cast<size_t>(text_len));
            ids.push_back(bos);
            ids.insert(ids.end(), gctx_ids.begin(), gctx_ids.end());
            ids.insert(ids.end(), content.begin(), content.end());
            while (static_cast<int>(ids.size()) < (text_len - 1)) {
                ids.push_back(pad_id);
            }
            ids.push_back(eos);
        } else {
            if (ids.size() > static_cast<size_t>(text_len)) {
                ids.resize(static_cast<size_t>(text_len));
            } else if (ids.size() < static_cast<size_t>(text_len)) {
                ids.resize(static_cast<size_t>(text_len), pad_id);
            }
        }

        for (auto& id : ids) {
            if (id == pad_id) continue;
            if (id < 0) id = pad_id;
            if (id >= max_vocab) id = (unk_id >= 0) ? unk_id : pad_id;
        }
        return ids;
    };

    const std::vector<int> text_ids = make_text_ids(prompt);

    if (cfg_.vae_checkpoint.empty()) {
        throw std::runtime_error("PonyXLDDPMModel: vae_checkpoint is required for reconstruction preview");
    }

    // Ensure VAE encoder is loaded (same as validation)
    if (!vae_) {
        using json = nlohmann::json;
        json vae_cfg = ModelArchitectures::defaultConfig(cfg_.vae_arch);
        vae_cfg["image_w"] = W;
        vae_cfg["image_h"] = H;
        vae_cfg["image_c"] = 3;

        int lh = cfg_.latent_h;
        int lw = cfg_.latent_w;
        if (lh <= 0 && lw <= 0) {
            lh = 1;
            lw = latent_len;
        } else if (lh <= 0 && lw > 0) {
            lh = (latent_len % lw == 0) ? (latent_len / lw) : 1;
        } else if (lh > 0 && lw <= 0) {
            lw = (latent_len % lh == 0) ? (latent_len / lh) : latent_len;
        }
        lh = std::max(1, lh);
        lw = std::max(1, lw);
        vae_cfg["latent_h"] = lh;
        vae_cfg["latent_w"] = lw;
        vae_cfg["latent_c"] = latent_in_dim;
        if (cfg_.vae_base_channels > 0) {
            vae_cfg["base_channels"] = cfg_.vae_base_channels;
        }

        auto m = ModelArchitectures::create(cfg_.vae_arch, vae_cfg);
        if (!m) {
            throw std::runtime_error("Failed to create VAE model: " + cfg_.vae_arch);
        }
        m->allocateParams();

        Mimir::Serialization::LoadOptions opts;
        const std::string vae_ckpt = resolve_checkpoint_dir_for_loading(cfg_.vae_checkpoint);
        opts.format = Mimir::Serialization::detect_format(vae_ckpt);
        opts.load_tokenizer = false;
        opts.load_encoder = false;
        opts.load_optimizer = false;
        opts.strict_mode = false;
        opts.validate_checksums = false;
        std::string err;
        if (!Mimir::Serialization::load_checkpoint(*m, vae_ckpt, opts, &err)) {
            throw std::runtime_error("Failed to load VAE checkpoint: " + cfg_.vae_checkpoint + " (resolved=" + vae_ckpt + ") | " + err);
        }
        m->freezeParameters(true);
        vae_ = std::move(m);
    }

    // Ensure VAE decoder is loaded (same as viz taps)
    if (!vae_decode_) {
        using json = nlohmann::json;
        json dec_cfg = ModelArchitectures::defaultConfig("vae_conv_decode");
        dec_cfg["image_w"] = W;
        dec_cfg["image_h"] = H;
        dec_cfg["image_c"] = 3;

        int lat_h = cfg_.latent_h;
        int lat_w = cfg_.latent_w;
        if (lat_h <= 0 && lat_w <= 0) {
            const int s = static_cast<int>(std::sqrt(static_cast<double>(latent_len)));
            if (s > 0 && (s * s) == latent_len) {
                lat_h = s;
                lat_w = s;
            } else {
                lat_h = 1;
                lat_w = latent_len;
            }
        } else if (lat_h <= 0 && lat_w > 0) {
            lat_h = (latent_len % lat_w == 0) ? (latent_len / lat_w) : 1;
        } else if (lat_h > 0 && lat_w <= 0) {
            lat_w = (latent_len % lat_h == 0) ? (latent_len / lat_h) : latent_len;
        }
        lat_h = std::max(1, lat_h);
        lat_w = std::max(1, lat_w);

        dec_cfg["latent_h"] = lat_h;
        dec_cfg["latent_w"] = lat_w;
        dec_cfg["latent_c"] = latent_in_dim;
        if (cfg_.vae_base_channels > 0) {
            dec_cfg["base_channels"] = cfg_.vae_base_channels;
        }

        auto m = ModelArchitectures::create("vae_conv_decode", dec_cfg);
        if (!m) {
            throw std::runtime_error("Failed to create VAE decoder model");
        }
        m->allocateParams();
        Mimir::Serialization::LoadOptions opts;
        const std::string vae_ckpt = resolve_checkpoint_dir_for_loading(cfg_.vae_checkpoint);
        opts.format = Mimir::Serialization::detect_format(vae_ckpt);
        opts.load_tokenizer = false;
        opts.load_encoder = false;
        opts.load_optimizer = false;
        opts.strict_mode = false;
        opts.validate_checksums = false;
        std::string err;
        if (!Mimir::Serialization::load_checkpoint(*m, vae_ckpt, opts, &err)) {
            throw std::runtime_error("Failed to load VAE checkpoint for decoder: " + cfg_.vae_checkpoint + " (resolved=" + vae_ckpt + ") | " + err);
        }
        m->freezeParameters(true);
        vae_decode_ = std::move(m);
    }

    // Encode x0
    const std::vector<float> img_f = imageBytesToFloatRGB(rgb, W, H);
    const std::vector<float> packed = vae_->forwardPass(img_f, false);

    int vae_image_dim = W * H * 3;
    int vae_latent_dim = 0;
    try {
        if (vae_->modelConfig.contains("latent_dim")) {
            vae_latent_dim = vae_->modelConfig["latent_dim"].get<int>();
        } else if (vae_->modelConfig.contains("latent_h") && vae_->modelConfig.contains("latent_w") && vae_->modelConfig.contains("latent_c")) {
            vae_latent_dim = vae_->modelConfig["latent_h"].get<int>() * vae_->modelConfig["latent_w"].get<int>() * vae_->modelConfig["latent_c"].get<int>();
        }
        if (vae_->modelConfig.contains("image_dim")) {
            vae_image_dim = vae_->modelConfig["image_dim"].get<int>();
        }
    } catch (...) {
    }

    if (vae_latent_dim <= 0 || vae_latent_dim != latent_raw_dim) {
        throw std::runtime_error("VAE latent_dim mismatch during reconstruction preview");
    }
    if (static_cast<int>(packed.size()) < (vae_image_dim + 2 * vae_latent_dim)) {
        throw std::runtime_error("VAE packed output too small (reconstruction preview)");
    }

    int vae_lh = cfg_.latent_h;
    int vae_lw = cfg_.latent_w;
    int vae_lc = latent_in_dim;
    try {
        if (vae_->modelConfig.contains("latent_h")) vae_lh = vae_->modelConfig["latent_h"].get<int>();
        if (vae_->modelConfig.contains("latent_w")) vae_lw = vae_->modelConfig["latent_w"].get<int>();
        if (vae_->modelConfig.contains("latent_c")) vae_lc = vae_->modelConfig["latent_c"].get<int>();
    } catch (...) {
    }
    vae_lh = std::max(1, vae_lh);
    vae_lw = std::max(1, vae_lw);
    vae_lc = std::max(1, vae_lc);
    if (vae_lc != latent_in_dim || (vae_lh * vae_lw) != latent_len) {
        throw std::runtime_error("VAE latent shape mismatch during reconstruction preview");
    }

    std::vector<float> x0;
    const float scale = std::max(0.0f, cfg_.vae_scale);
    const size_t mu_off = static_cast<size_t>(vae_image_dim);
    if (!latent_chw_to_tokens_hwc_scaled(x0, packed.data() + mu_off, vae_lh, vae_lw, vae_lc, scale)) {
        throw std::runtime_error("Failed to convert VAE mu (CHW) to PonyXL latent (tokens HWC) during reconstruction preview");
    }

    // Latent H/W
    int lat_h = vae_lh;
    int lat_w = vae_lw;
    lat_h = std::max(1, lat_h);
    lat_w = std::max(1, lat_w);

    // Timestep (same default as validation)
    const int T = std::max(2, cfg_.ddpm_steps);
    const int t = (ddpm_step >= 0) ? std::clamp(ddpm_step, 0, T - 1) : (T / 2);
    const float beta0 = std::clamp(cfg_.ddpm_beta_start, 0.0f, 0.999f);
    const float beta1 = std::clamp(cfg_.ddpm_beta_end, 0.0f, 0.999f);
    static thread_local int cache_T = 0;
    static thread_local float cache_b0 = -1.0f;
    static thread_local float cache_b1 = -1.0f;
    static thread_local std::vector<float> cache_ab;
    if (cache_T != T || cache_b0 != beta0 || cache_b1 != beta1 || static_cast<int>(cache_ab.size()) != T) {
        cache_T = T;
        cache_b0 = beta0;
        cache_b1 = beta1;
        cache_ab.assign(static_cast<size_t>(T), 1.0f);
        float ab = 1.0f;
        for (int i = 0; i < T; ++i) {
            const float frac = (T > 1) ? (static_cast<float>(i) / static_cast<float>(T - 1)) : 0.0f;
            const float beta = std::clamp(beta0 + (beta1 - beta0) * frac, 0.0f, 0.999f);
            const float alpha = 1.0f - beta;
            ab *= alpha;
            cache_ab[static_cast<size_t>(i)] = std::clamp(ab, 1e-6f, 1.0f);
        }
    }
    const float alpha_bar = cache_ab[static_cast<size_t>(t)];
    const float sqrt_ab = std::sqrt(alpha_bar);
    const float sqrt_1mab = std::sqrt(std::max(0.0f, 1.0f - alpha_bar));
    const float t_norm = (T > 1) ? (static_cast<float>(t) / static_cast<float>(T - 1)) : 0.0f;

    // Deterministic eps
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::normal_distribution<float> n01(0.0f, 1.0f);
    std::vector<float> eps(static_cast<size_t>(latent_raw_dim), 0.0f);
    for (int i = 0; i < latent_raw_dim; ++i) {
        eps[static_cast<size_t>(i)] = n01(rng);
    }
    std::vector<float> x_t(static_cast<size_t>(latent_raw_dim), 0.0f);
    for (int i = 0; i < latent_raw_dim; ++i) {
        const float e = eps[static_cast<size_t>(i)];
        x_t[static_cast<size_t>(i)] = sqrt_ab * x0[static_cast<size_t>(i)] + sqrt_1mab * e;
    }

    std::unordered_map<std::string, std::vector<float>> fin;
    std::unordered_map<std::string, std::vector<int>> iin;
    fin["latent"] = x_t;
    iin["text_ids"] = text_ids;
    if (cfg_.sdxl_time_cond) {
        fin["timestep"] = std::vector<float>{compute_time_cond_value(cfg_, t_norm, alpha_bar)};
    }
    const std::vector<float> pred = forwardPassNamed(fin, iin, false);
    if (static_cast<int>(pred.size()) < latent_raw_dim) {
        throw std::runtime_error("eps_pred size too small during reconstruction preview");
    }

    std::vector<float> x0_hat(static_cast<size_t>(latent_raw_dim), 0.0f);
    for (int i = 0; i < latent_raw_dim; ++i) {
        const float ehat = pred[static_cast<size_t>(i)];
        x0_hat[static_cast<size_t>(i)] = (x_t[static_cast<size_t>(i)] - sqrt_1mab * ehat) / std::max(1e-6f, sqrt_ab);
    }

    const float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 1.0f;
    std::vector<float> hat_chw;
    if (!latent_tokens_hwc_to_chw_scaled(hat_chw, x0_hat.data(), lat_h, lat_w, latent_in_dim, inv_scale)) {
        throw std::runtime_error("Failed to convert latent tokens->CHW for VAE decode (reconstruction preview)");
    }
    const std::vector<float> img_hat = vae_decode_->forwardPass(hat_chw, false);

    const int ms = std::max(1, max_side);
    const int sx = (W > ms) ? static_cast<int>((W + ms - 1) / ms) : 1;
    const int sy = (H > ms) ? static_cast<int>((H + ms - 1) / ms) : 1;
    const int pw = std::max(1, W / sx);
    const int ph = std::max(1, H / sy);

    ReconPreview out;
    out.pixels = to_rgb_preview_image(img_hat, W, H, ms);
    out.w = pw;
    out.h = ph;
    out.channels = 3;
    return out;
}

void PonyXLDDPMModel::buildInto(Model& model, const Config& cfg) {
    auto sat_mul = [](size_t a, size_t b) -> size_t {
        if (a == 0 || b == 0) return 0;
        if (a > (static_cast<size_t>(-1) / b)) return static_cast<size_t>(-1);
        return a * b;
    };

    auto build_sdxl = [&](Model& m) {
        m.getMutableLayers().clear();
        m.setModelName("PonyXLSDXL");
        m.setHasEncoder(true);
        m.modelConfig["type"] = "ponyxl_sdxl";
        m.modelConfig["task"] = "sdxl_eps_predictor";

        const int d_model = std::max(1, cfg.d_model);
        const int text_len = std::max(1, cfg.text_ctx_len);
        const int latent_len = std::max(1, cfg.latent_seq_len);
        const int latent_in_dim = std::max(1, cfg.latent_in_dim);
        const int vocab = std::max(1, cfg.max_vocab);
        const int pad_id = 0;
        const int heads = std::max(1, cfg.num_heads);
        const int unet_layers = std::max(1, cfg.unet_layers);
        const int text_layers = std::max(1, cfg.text_layers);
        const int mlp_hidden = std::max(4, cfg.mlp_hidden);

        const int unet_depth = std::max(1, cfg.unet_depth);
        const int unet_blocks = std::max(1, cfg.unet_blocks_per_level);
        const int unet_bottleneck_blocks = std::max(1, cfg.unet_bottleneck_blocks);

        const int latent_raw_dim = latent_len * latent_in_dim;
        const int output_dim = latent_raw_dim;

        m.modelConfig["d_model"] = d_model;
        m.modelConfig["text_ctx_len"] = text_len;
        m.modelConfig["latent_seq_len"] = latent_len;
        m.modelConfig["latent_in_dim"] = latent_in_dim;
        m.modelConfig["max_vocab"] = vocab;
        m.modelConfig["padding_idx"] = pad_id;
        m.modelConfig["num_heads"] = heads;
        m.modelConfig["unet_layers"] = unet_layers;
        m.modelConfig["text_layers"] = text_layers;
        m.modelConfig["mlp_hidden"] = mlp_hidden;
        m.modelConfig["sdxl_time_cond"] = cfg.sdxl_time_cond;
        m.modelConfig["unet_depth"] = unet_depth;
        m.modelConfig["unet_blocks_per_level"] = unet_blocks;
        m.modelConfig["unet_bottleneck_blocks"] = unet_bottleneck_blocks;
        m.modelConfig["output_dim"] = output_dim;

        // Training helpers / stabilization knobs
        m.modelConfig["recon_loss"] = cfg.recon_loss;
        m.modelConfig["loss_weighting"] = cfg.loss_weighting;
        m.modelConfig["min_snr_gamma"] = cfg.min_snr_gamma;
        m.modelConfig["kl_beta"] = cfg.kl_beta;
        m.modelConfig["kl_warmup_steps"] = cfg.kl_warmup_steps;
        m.modelConfig["logvar_clip_min"] = cfg.logvar_clip_min;
        m.modelConfig["logvar_clip_max"] = cfg.logvar_clip_max;

        // Input routing (latents = float)
        m.push("ponyxl_sdxl/latent_in", "Identity", 0);
        if (auto* L = m.getLayerByName("ponyxl_sdxl/latent_in")) {
            L->inputs = {"latent"};
            L->output = "ponyxl_sdxl/latent_raw";
        }
        const std::string latent_raw = "ponyxl_sdxl/latent_raw";

        // -----------------------------
        // text_encoder: text_ids(int) -> Embedding entraînable -> mini Transformer
        // -----------------------------
        m.push("ponyxl_sdxl/text_encoder/tok_emb", "Embedding",
               sat_mul(static_cast<size_t>(vocab), static_cast<size_t>(d_model)));
        if (auto* E = m.getLayerByName("ponyxl_sdxl/text_encoder/tok_emb")) {
            E->inputs = {"text_ids"};
            E->output = "ponyxl_sdxl/text_encoder/tok_emb_out";
            E->vocab_size = vocab;
            E->embed_dim = d_model;
            E->padding_idx = pad_id;
            E->seq_len = text_len;
        }

        // -----------------------------
        // text_encoder: mini Transformer sur embeddings
        // -----------------------------
        std::string text = "ponyxl_sdxl/text_encoder/tok_emb_out";

        // Injecter mag/mod (embeddings spéciaux Encoder) dans le flux texte.
        // Add supporte le broadcast: (seq_len*d_model) + (d_model).
        m.push("ponyxl_sdxl/text_encoder/mag_in", "Identity", 0);
        if (auto* L = m.getLayerByName("ponyxl_sdxl/text_encoder/mag_in")) {
            L->inputs = {"mag"};
            L->output = "ponyxl_sdxl/text_encoder/mag_vec";
        }

        m.push("ponyxl_sdxl/text_encoder/add_mag", "Add", 0);
        if (auto* L = m.getLayerByName("ponyxl_sdxl/text_encoder/add_mag")) {
            L->inputs = {text, "ponyxl_sdxl/text_encoder/mag_vec"};
            L->output = "ponyxl_sdxl/text_encoder/tok_plus_mag";
        }
        text = "ponyxl_sdxl/text_encoder/tok_plus_mag";

        m.push("ponyxl_sdxl/text_encoder/mod_in", "Identity", 0);
        if (auto* L = m.getLayerByName("ponyxl_sdxl/text_encoder/mod_in")) {
            L->inputs = {"mod"};
            L->output = "ponyxl_sdxl/text_encoder/mod_vec";
        }

        m.push("ponyxl_sdxl/text_encoder/add_mod", "Add", 0);
        if (auto* L = m.getLayerByName("ponyxl_sdxl/text_encoder/add_mod")) {
            L->inputs = {text, "ponyxl_sdxl/text_encoder/mod_vec"};
            L->output = "ponyxl_sdxl/text_encoder/tok_plus_mag_mod";
        }
        text = "ponyxl_sdxl/text_encoder/tok_plus_mag_mod";
        const size_t attn_params = sat_mul(static_cast<size_t>(4), sat_mul(static_cast<size_t>(d_model), static_cast<size_t>(d_model)));

        for (int i = 0; i < text_layers; ++i) {
            const std::string p = "ponyxl_sdxl/text_encoder/block" + std::to_string(i + 1);

            m.push(p + "/ln1", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
            if (auto* L = m.getLayerByName(p + "/ln1")) {
                L->inputs = {text};
                L->output = p + "/ln1_out";
                L->affine = true;
                L->use_bias = true;
                L->eps = 1e-5f;
                L->in_features = d_model;
            }

            m.push(p + "/self_attn", "MultiHeadAttention", attn_params);
            if (auto* L = m.getLayerByName(p + "/self_attn")) {
                L->inputs = {p + "/ln1_out"};
                L->output = p + "/self_attn_out";
                L->seq_len = text_len;
                L->embed_dim = d_model;
                L->num_heads = heads;
                L->causal = cfg.text_clip_like;
            }

            m.push(p + "/add1", "Add", 0);
            if (auto* L = m.getLayerByName(p + "/add1")) {
                L->inputs = {text, p + "/self_attn_out"};
                L->output = p + "/res1";
            }

            m.push(p + "/ln2", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
            if (auto* L = m.getLayerByName(p + "/ln2")) {
                L->inputs = {p + "/res1"};
                L->output = p + "/ln2_out";
                L->affine = true;
                L->use_bias = true;
                L->eps = 1e-5f;
                L->in_features = d_model;
            }

            m.push(p + "/mlp_fc1", "Linear",
                   sat_mul(static_cast<size_t>(d_model), static_cast<size_t>(mlp_hidden)) + static_cast<size_t>(mlp_hidden));
            if (auto* L = m.getLayerByName(p + "/mlp_fc1")) {
                L->inputs = {p + "/ln2_out"};
                L->output = p + "/mlp_h";
                L->seq_len = text_len;
                L->in_features = d_model;
                L->out_features = mlp_hidden;
                L->use_bias = true;
            }

            m.push(p + "/mlp_act", "GELU", 0);
            if (auto* L = m.getLayerByName(p + "/mlp_act")) {
                L->inputs = {p + "/mlp_h"};
                L->output = p + "/mlp_h_act";
            }

            m.push(p + "/mlp_fc2", "Linear",
                   sat_mul(static_cast<size_t>(mlp_hidden), static_cast<size_t>(d_model)) + static_cast<size_t>(d_model));
            if (auto* L = m.getLayerByName(p + "/mlp_fc2")) {
                L->inputs = {p + "/mlp_h_act"};
                L->output = p + "/mlp_out";
                L->seq_len = text_len;
                L->in_features = mlp_hidden;
                L->out_features = d_model;
                L->use_bias = true;
            }

            m.push(p + "/add2", "Add", 0);
            if (auto* L = m.getLayerByName(p + "/add2")) {
                L->inputs = {p + "/res1", p + "/mlp_out"};
                L->output = p + "/out";
            }

            text = p + "/out";
        }

        if (cfg.text_clip_like) {
            // LN final (CLIP-like)
            m.push("ponyxl_sdxl/text_encoder/final_ln", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
            if (auto* L = m.getLayerByName("ponyxl_sdxl/text_encoder/final_ln")) {
                L->inputs = {text};
                L->output = "ponyxl_sdxl/text_encoder/final_ln_out";
                L->affine = true;
                L->use_bias = true;
                L->eps = 1e-5f;
                L->in_features = d_model;
            }
            text = "ponyxl_sdxl/text_encoder/final_ln_out";
        }

        if (cfg.text_bottleneck_meanpool) {
            // meanpool déterministe: (seq_len*d_model) -> (d_model)
            m.push("ponyxl_sdxl/text_encoder/meanpool", "TokenMeanPool", 0);
            if (auto* P = m.getLayerByName("ponyxl_sdxl/text_encoder/meanpool")) {
                P->inputs = {text};
                P->output = "ponyxl_sdxl/text_encoder/pooled";
                P->seq_len = text_len;
                P->embed_dim = d_model;
            }
            text = "ponyxl_sdxl/text_encoder/pooled";
        }
        const std::string text_ctx = "ponyxl_sdxl/text_encoder/out";
        m.push(text_ctx, "Identity", 0);
        if (auto* L = m.getLayerByName(text_ctx)) {
            L->inputs = {text};
            L->output = "ponyxl_sdxl/text_ctx";
        }

        // -----------------------------
        // vae: projection latent_raw -> latent_tokens(d_model)
        // -----------------------------
        m.push("ponyxl_sdxl/vae/latent_proj", "Linear",
               sat_mul(static_cast<size_t>(latent_in_dim), static_cast<size_t>(d_model)) + static_cast<size_t>(d_model));
        if (auto* L = m.getLayerByName("ponyxl_sdxl/vae/latent_proj")) {
            L->inputs = {latent_raw};
            L->output = "ponyxl_sdxl/vae/latent_tokens";
            L->seq_len = latent_len;
            L->in_features = latent_in_dim;
            L->out_features = d_model;
            L->use_bias = true;
        }

        m.push("ponyxl_sdxl/vae/latent_ln", "LayerNorm", static_cast<size_t>(2) * static_cast<size_t>(d_model));
        if (auto* L = m.getLayerByName("ponyxl_sdxl/vae/latent_ln")) {
            L->inputs = {"ponyxl_sdxl/vae/latent_tokens"};
            L->output = "ponyxl_sdxl/vae/latent_norm";
            L->affine = true;
            L->use_bias = true;
            L->eps = 1e-5f;
            L->in_features = d_model;
        }

        // -----------------------------
        // time embedding (optionnel): timestep(float scalar) -> MLP -> add(latent)
        // -----------------------------
        std::string latent_in_unet = "ponyxl_sdxl/vae/latent_norm";
        if (cfg.sdxl_time_cond) {
            // input: timestep (vector size 1)
            m.push("ponyxl_sdxl/time/in", "Identity", 0);
            if (auto* L = m.getLayerByName("ponyxl_sdxl/time/in")) {
                L->inputs = {"timestep"};
                L->output = "ponyxl_sdxl/time/t";
            }

            m.push("ponyxl_sdxl/time/fc1", "Linear", static_cast<size_t>(d_model) + static_cast<size_t>(d_model));
            if (auto* L = m.getLayerByName("ponyxl_sdxl/time/fc1")) {
                L->inputs = {"ponyxl_sdxl/time/t"};
                L->output = "ponyxl_sdxl/time/h1";
                L->seq_len = 1;
                L->in_features = 1;
                L->out_features = d_model;
                L->use_bias = true;
            }

            m.push("ponyxl_sdxl/time/act", "SiLU", 0);
            if (auto* L = m.getLayerByName("ponyxl_sdxl/time/act")) {
                L->inputs = {"ponyxl_sdxl/time/h1"};
                L->output = "ponyxl_sdxl/time/h1_act";
            }

            m.push("ponyxl_sdxl/time/fc2", "Linear", sat_mul(static_cast<size_t>(d_model), static_cast<size_t>(d_model)) + static_cast<size_t>(d_model));
            if (auto* L = m.getLayerByName("ponyxl_sdxl/time/fc2")) {
                L->inputs = {"ponyxl_sdxl/time/h1_act"};
                L->output = "ponyxl_sdxl/time/emb";
                L->seq_len = 1;
                L->in_features = d_model;
                L->out_features = d_model;
                L->use_bias = true;
            }

            // Broadcast add: (latent_len*d_model) + (d_model)
            m.push("ponyxl_sdxl/vae/add_time", "Add", 0);
            if (auto* L = m.getLayerByName("ponyxl_sdxl/vae/add_time")) {
                L->inputs = {"ponyxl_sdxl/vae/latent_norm", "ponyxl_sdxl/time/emb"};
                L->output = "ponyxl_sdxl/vae/latent_time";
            }
            latent_in_unet = "ponyxl_sdxl/vae/latent_time";
        }

        // -----------------------------
        // UNet 2D multi-échelle (conv) + cross-attn
        // -----------------------------
            auto infer_hw = [&](int& h, int& w, int n) {
                if (h > 0 && w > 0 && h * w == n) return;
                int s = (int)std::sqrt((double)std::max(1, n));
                s = std::max(1, s);
                while (s > 1 && (n % s) != 0) --s;
                h = s;
                w = std::max(1, n / std::max(1, s));
            };

            int lat_h = std::max(0, cfg.latent_h);
            int lat_w = std::max(0, cfg.latent_w);
            infer_hw(lat_h, lat_w, latent_len);
            m.modelConfig["latent_h"] = lat_h;
            m.modelConfig["latent_w"] = lat_w;

            // latent_(norm|time) (seq-major) -> reshape to HWC then permute to CHW
            m.push("ponyxl_sdxl/vae/latent_reshape", "Reshape", 0);
            if (auto* L = m.getLayerByName("ponyxl_sdxl/vae/latent_reshape")) {
                L->inputs = {latent_in_unet};
                L->output = "ponyxl_sdxl/vae/latent_hwc";
                L->target_shape = {lat_h, lat_w, d_model};
            }

            m.push("ponyxl_sdxl/vae/latent_to_chw", "Permute", 0);
            if (auto* L = m.getLayerByName("ponyxl_sdxl/vae/latent_to_chw")) {
                L->inputs = {"ponyxl_sdxl/vae/latent_hwc"};
                L->output = "ponyxl_sdxl/unet/in_chw";
                L->shape = {lat_h, lat_w, d_model};
                L->permute_dims = {2, 0, 1};
            }

            auto conv2d = [&](const std::string& name,
                              const std::string& in,
                              const std::string& out,
                              int in_c,
                              int out_c,
                              int H,
                              int W,
                              int k,
                              int s,
                              int p) {
                m.push(name, "Conv2d",
                       sat_mul(static_cast<size_t>(out_c), sat_mul(static_cast<size_t>(in_c), sat_mul(static_cast<size_t>(k), static_cast<size_t>(k)))));
                if (auto* L = m.getLayerByName(name)) {
                    L->inputs = {in};
                    L->output = out;
                    L->in_channels = in_c;
                    L->out_channels = out_c;
                    L->input_height = H;
                    L->input_width = W;
                    L->kernel_size = k;
                    L->stride = s;
                    L->padding = p;
                    L->use_bias = false;

                    // IMPORTANT (viz taps): Model.cpp capture les activations spatiales
                    // uniquement si la layer connaît ses dimensions de sortie.
                    // Sans ça, ow/oh restent à 0 et la grille des blocs reste vide.
                    const int out_h = std::max(1, (H + 2 * p - k) / std::max(1, s) + 1);
                    const int out_w = std::max(1, (W + 2 * p - k) / std::max(1, s) + 1);
                    L->output_height = out_h;
                    L->output_width = out_w;
                    L->out_h = out_h;
                    L->out_w = out_w;
                }
                m.push(name + "/act", "GELU", 0);
                if (auto* A = m.getLayerByName(name + "/act")) {
                    A->inputs = {out};
                    A->output = out + "_act";
                }
                return out + "_act";
            };

            auto upsample2x = [&](const std::string& name,
                                  const std::string& in,
                                  const std::string& out,
                                  int channels,
                                  int in_h,
                                  int in_w) {
                m.push(name, "UpsampleNearest", 0);
                if (auto* U = m.getLayerByName(name)) {
                    U->inputs = {in};
                    U->output = out;
                    U->in_channels = channels;
                    U->out_h = in_h;
                    U->out_w = in_w;
                    U->scale_h = 2.0f;
                    U->scale_w = 2.0f;
                }
                return out;
            };

            auto cross_attend_chw = [&](const std::string& name,
                                        const std::string& in_chw,
                                        const std::string& out_chw,
                                        int C,
                                        int H,
                                        int W) {
                // CHW -> HWC
                m.push(name + "/to_hwc", "Permute", 0);
                if (auto* P = m.getLayerByName(name + "/to_hwc")) {
                    P->inputs = {in_chw};
                    P->output = name + "/hwc";
                    P->shape = {C, H, W};
                    P->permute_dims = {1, 2, 0};
                }

                // Cross-attn (seq_len = H*W inferred), embed_dim=C
                const size_t attn2d_params = sat_mul(static_cast<size_t>(C), sat_mul(static_cast<size_t>(C), static_cast<size_t>(4)));
                m.push(name + "/xattn", "CrossAttention", attn2d_params);
                if (auto* A = m.getLayerByName(name + "/xattn")) {
                    A->inputs = {name + "/hwc", "ponyxl_sdxl/text_ctx"};
                    A->output = name + "/hwc_out";
                    A->embed_dim = C;
                    A->num_heads = heads;
                    A->causal = false;
                }

                // HWC -> CHW
                m.push(name + "/to_chw", "Permute", 0);
                if (auto* P2 = m.getLayerByName(name + "/to_chw")) {
                    P2->inputs = {name + "/hwc_out"};
                    P2->output = out_chw;
                    P2->shape = {H, W, C};
                    P2->permute_dims = {2, 0, 1};
                }
            };

            // UNet 2D avec channels fixes = d_model
            // Ajuster la profondeur: si (lat_h, lat_w) ne sont pas divisibles par 2^depth,
            // le downsample/upsample ne revient pas à la taille initiale (mismatch + forward/training instables).
            int depth = unet_depth;
            while (depth > 0) {
                const int pow2 = 1 << depth;
                if ((lat_h % pow2) == 0 && (lat_w % pow2) == 0) break;
                --depth;
            }
            m.modelConfig["unet_depth"] = depth;

            const int C = d_model;
            std::string x = "ponyxl_sdxl/unet/in_chw";
            int cur_h = lat_h;
            int cur_w = lat_w;

            std::vector<std::string> skips;
            std::vector<int> skip_h, skip_w;
            skips.reserve((size_t)depth);

            for (int d = 0; d < depth; ++d) {
                const std::string b = "ponyxl_sdxl/unet2d/down" + std::to_string(d + 1);
                for (int bi = 0; bi < unet_blocks; ++bi) {
                    const std::string bb = b + "/b" + std::to_string(bi + 1);
                    x = conv2d(bb + "/conv1", x, bb + "/c1", C, C, cur_h, cur_w, 3, 1, 1);
                    x = conv2d(bb + "/conv2", x, bb + "/c2", C, C, cur_h, cur_w, 3, 1, 1);
                    cross_attend_chw(bb + "/xattn", x, bb + "/xattn_out", C, cur_h, cur_w);
                    x = bb + "/xattn_out";
                }

                skips.push_back(x);
                skip_h.push_back(cur_h);
                skip_w.push_back(cur_w);

                // Downsample (stride 2)
                const std::string ds = b + "/down";
                x = conv2d(ds + "/conv", x, ds + "/y", C, C, cur_h, cur_w, 3, 2, 1);
                cur_h = std::max(1, (cur_h + 2 * 1 - 3) / 2 + 1);
                cur_w = std::max(1, (cur_w + 2 * 1 - 3) / 2 + 1);
            }

            // Bottleneck
            {
                const std::string b = "ponyxl_sdxl/unet2d/bottleneck";
                for (int bi = 0; bi < unet_bottleneck_blocks; ++bi) {
                    const std::string bb = b + "/b" + std::to_string(bi + 1);
                    x = conv2d(bb + "/conv1", x, bb + "/c1", C, C, cur_h, cur_w, 3, 1, 1);
                    cross_attend_chw(bb + "/xattn", x, bb + "/xattn_out", C, cur_h, cur_w);
                    x = bb + "/xattn_out";
                    x = conv2d(bb + "/conv2", x, bb + "/c2", C, C, cur_h, cur_w, 3, 1, 1);
                }
            }

            // Up path
            for (int d = depth - 1; d >= 0; --d) {
                const std::string b = "ponyxl_sdxl/unet2d/up" + std::to_string(d + 1);
                // Upsample
                const int up_in_h = cur_h;
                const int up_in_w = cur_w;
                x = upsample2x(b + "/up", x, b + "/up_y", C, up_in_h, up_in_w);
                cur_h = up_in_h * 2;
                cur_w = up_in_w * 2;

                // Concat skip (channel concat in CHW => flat concat)
                m.push(b + "/concat", "Concat", 0);
                if (auto* L = m.getLayerByName(b + "/concat")) {
                    L->inputs = {x, skips[(size_t)d]};
                    L->output = b + "/cat";
                    L->concat_axis = 0;
                }

                // Reduce back to C
                x = conv2d(b + "/reduce", b + "/cat", b + "/r", C * 2, C, cur_h, cur_w, 3, 1, 1);
                for (int bi = 0; bi < unet_blocks; ++bi) {
                    const std::string bb = b + "/b" + std::to_string(bi + 1);
                    x = conv2d(bb + "/conv", x, bb + "/c", C, C, cur_h, cur_w, 3, 1, 1);
                    cross_attend_chw(bb + "/xattn", x, bb + "/xattn_out", C, cur_h, cur_w);
                    x = bb + "/xattn_out";
                }
            }

            // CHW -> HWC -> (seq-major) -> out_proj
            m.push("ponyxl_sdxl/unet2d/to_hwc", "Permute", 0);
            if (auto* P = m.getLayerByName("ponyxl_sdxl/unet2d/to_hwc")) {
                P->inputs = {x};
                P->output = "ponyxl_sdxl/unet2d/hwc";
                P->shape = {C, lat_h, lat_w};
                P->permute_dims = {1, 2, 0};
            }

            m.push("ponyxl_sdxl/unet/out_proj", "Linear",
                   sat_mul(static_cast<size_t>(d_model), static_cast<size_t>(latent_in_dim)) + static_cast<size_t>(latent_in_dim));
            if (auto* L = m.getLayerByName("ponyxl_sdxl/unet/out_proj")) {
                L->inputs = {"ponyxl_sdxl/unet2d/hwc"};
                L->output = "ponyxl_sdxl/unet/eps";
                L->seq_len = latent_len;
                L->in_features = d_model;
                L->out_features = latent_in_dim;
                L->use_bias = true;
            }
            const std::string out_act = normalize_output_activation(cfg.output_activation);
            const std::string out_type = (out_act == "tanh") ? "Tanh" : "Identity";
            m.push("ponyxl_sdxl/out", out_type, 0);
            if (auto* L = m.getLayerByName("ponyxl_sdxl/out")) {
                L->inputs = {"ponyxl_sdxl/unet/eps"};
                L->output = "x";
            }
    };

    if (!cfg.sdxl_time_cond) {
        throw std::runtime_error("PonyXLDDPMModel::buildInto: ponyxl_sdxl requires sdxl_time_cond=true");
    }

    build_sdxl(model);
}

std::vector<float> PonyXLDDPMModel::imageBytesToFloatRGB(const std::vector<uint8_t>& rgb, int w, int h) {
    const int W = std::max(1, w);
    const int H = std::max(1, h);
    const size_t outN = static_cast<size_t>(W) * static_cast<size_t>(H) * 3ULL;
    std::vector<float> y(outN, 0.0f);

    const size_t limit = std::min(outN, rgb.size());
    for (size_t i = 0; i < limit; ++i) {
        y[i] = (static_cast<float>(rgb[i]) / 255.0f) * 2.0f - 1.0f;
    }
    return y;
}
