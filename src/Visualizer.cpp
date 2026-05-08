#include "Visualizer.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <chrono>

static const char* kVizUISettingsFile = "viz_ui_settings.json";

namespace {
struct ParsedVizLabel {
    std::string raw;
    std::string path;
    std::string extra;
    std::vector<std::string> parts;

    std::string model;
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

static ParsedVizLabel parse_viz_label(const std::string& label_raw) {
    ParsedVizLabel out;
    out.raw = label_raw;

    // Convention: une partie "chemin" peut être suivie d'extra après " | ".
    // Ex: "ponyxl_sdxl/input/dataset/rgb | /path/file.png"
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

    auto is_activation_type = [&](const std::string& t) -> bool {
        // Doit rester simple: uniquement les activations usuelles.
        return t == "ReLU" || t == "LeakyReLU" || t == "GELU" || t == "SiLU" ||
               t == "Tanh" || t == "Sigmoid" || t == "Softmax" || t == "LogSoftmax" ||
               t == "Softplus" || t == "Mish" || t == "HardSigmoid" || t == "HardSwish";
    };

    auto join_tail = [&](size_t start, size_t end) -> std::string {
        if (end <= start) return {};
        // Garder jusqu'à 3 segments de fin (meilleur tradeoff lisibilité / place)
        const size_t keep = std::min<size_t>(3, end - start);
        const size_t from = end - keep;
        std::string tail;
        for (size_t i = from; i < end; ++i) {
            if (!tail.empty()) tail += "/";
            tail += out.parts[i];
        }
        return tail;
    };

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

        if (is_activation_type(t)) {
            out.tag = "ACT";
            out.headline = "Activations";
        } else {
            out.tag = "L";
            out.headline = "Layer";
        }

        const size_t start = static_cast<size_t>(idx_blocks + 1);
        const size_t end = (out.parts.size() > drop) ? (out.parts.size() - drop) : start;
        const std::string tail = join_tail(start, end);
        out.short_text = tail.empty() ? std::string("blocks") : tail;
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
    return out;
}

static sf::Color color_for_tag(const std::string& tag) {
    if (tag == "DS") return sf::Color(120, 180, 220);
    if (tag == "PRE") return sf::Color(140, 200, 220);
    if (tag == "OUT") return sf::Color(220, 160, 120);
    if (tag == "VAL") return sf::Color(160, 220, 160);
    if (tag == "ACT") return sf::Color(160, 160, 220);
    if (tag == "L") return sf::Color(180, 180, 200);
    if (tag == "T") return sf::Color(180, 170, 140);
    return sf::Color(140, 140, 150);
}

static void position_sprite_centered_in_box(sf::Sprite& sprite, float x, float y, float box_size) {
    // sprite est déjà mis à l'échelle par createImageTexture().
    const auto lb = sprite.getLocalBounds();
    const auto sc = sprite.getScale();
    const float dw = lb.width * sc.x;
    const float dh = lb.height * sc.y;
    const float ox = (box_size - dw) * 0.5f;
    const float oy = (box_size - dh) * 0.5f;
    // Ajuster avec left/top au cas où localBounds n'est pas (0,0).
    sprite.setPosition(x + ox - lb.left * sc.x, y + oy - lb.top * sc.y);
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

static float lr_from_t(float t) {
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

static int warmup_from_t(float t) {
    const float tt = clamp01(t);
    const float v = tt * static_cast<float>(kLiveWarmupMax);
    return std::max(0, static_cast<int>(std::lround(v)));
}

static float t_from_warmup(int steps) {
    const int s = std::clamp(steps, 0, kLiveWarmupMax);
    return clamp01(static_cast<float>(s) / static_cast<float>(kLiveWarmupMax));
}

static float klbeta_from_t(float t) {
    return clamp01(t); // 0..1
}

static float t_from_klbeta(float beta) {
    float b = beta;
    if (!std::isfinite(b)) b = 0.0f;
    return clamp01(b);
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

        hide_activation_blocks = viz.value("hide_activation_blocks", false);
        architecture_path = viz.value("architecture_path", std::string());
    }
}

Visualizer::~Visualizer() {
    shutdown();
}

void Visualizer::shutdown() {
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
        window = std::make_unique<sf::RenderWindow>(
            sf::VideoMode(window_width, window_height),
            window_title,
            sf::Style::Titlebar | sf::Style::Close
        );

        // Logo du programme + splash au lancement (best-effort)
        logo_loaded_ = false;
        try {
            sf::Image logo_img;
            if (logo_img.loadFromFile("logo.png")) {
                const auto sz = logo_img.getSize();
                if (sz.x > 0 && sz.y > 0) {
                    // Certains WM ignorent les icônes trop grandes ou atypiques.
                    // On force une taille standard (64x64) via downscale nearest-neighbor.
                    const unsigned target = 64;
                    sf::Image icon_img;
                    icon_img.create(target, target);
                    for (unsigned yy = 0; yy < target; ++yy) {
                        for (unsigned xx = 0; xx < target; ++xx) {
                            const unsigned sx = (sz.x > 0) ? (xx * sz.x) / target : 0;
                            const unsigned sy = (sz.y > 0) ? (yy * sz.y) / target : 0;
                            icon_img.setPixel(xx, yy, logo_img.getPixel(sx, sy));
                        }
                    }
                    window->setIcon(target, target, icon_img.getPixelsPtr());
                }
                if (logo_texture_.loadFromImage(logo_img)) {
                    logo_sprite_.setTexture(logo_texture_, true);
                    logo_sprite_.setColor(sf::Color(255, 255, 255, 220));
                    logo_loaded_ = true;
                    logo_clock_.restart();
                }
            }
        } catch (...) {
            logo_loaded_ = false;
        }

        window->setFramerateLimit(fps_limit);
        syncUIView();

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
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        };
        for (const auto& path : font_candidates) {
            if (font.loadFromFile(path)) {
                font_loaded = true;
                break;
            }
        }

        std::cout << "✓ Fenêtre de visualisation SFML initialisée (" 
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
                } else if (settings.contains("slot1")) {
                    // Compat: certains fichiers peuvent exposer un slot nommé "slot1" au top-level.
                    chosen = &settings["slot1"];
                    chosen_name = "slot1";
                } else if (settings.contains("slots") && settings["slots"].is_object() && settings["slots"].contains("1")) {
                    chosen = &settings["slots"]["1"];
                    chosen_name = "slots[1]";
                }

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
                            window->setSize(sf::Vector2u(static_cast<unsigned>(ww), static_cast<unsigned>(hh)));
                            window_width = ww;
                            window_height = hh;
                            syncUIView();
                        }
                    } catch (...) {
                        // ignore resize errors
                    }

                    const bool ok = applyUILayout(*chosen);
                    if (ok && chosen_name) {
                        std::cout << "✓ Layout appliqué par défaut (" << chosen_name << ")" << std::endl;
                    }
                }
            }
        } catch (...) {
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors de l'initialisation SFML: " << e.what() << std::endl;
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

        if (!layout.contains("panels") || !layout["panels"].is_array()) {
            // Layout incomplet: appliquer seulement les toggles.
            panels_[static_cast<size_t>(PanelId::Generated)].visible = show_generated_images;
            panels_[static_cast<size_t>(PanelId::Training)].visible = show_training_progress;
            panels_[static_cast<size_t>(PanelId::Graph)].visible = show_loss_graph;
            return true;
        }

        for (const auto& jp : layout["panels"]) {
            if (!jp.is_object()) continue;
            if (!jp.contains("id")) continue;
            const int raw_id = jp["id"].get<int>();
            if (raw_id < 0 || raw_id >= static_cast<int>(PanelId::Count)) continue;
            const size_t idx = static_cast<size_t>(raw_id);
            auto& p = panels_[idx];

            if (jp.contains("title") && jp["title"].is_string()) p.title = jp["title"].get<std::string>();
            if (jp.contains("x")) p.pos.x = jp["x"].get<float>();
            if (jp.contains("y")) p.pos.y = jp["y"].get<float>();
            if (jp.contains("w")) p.size.x = std::max(kPanelMinW, jp["w"].get<float>());
            if (jp.contains("h")) p.size.y = std::max(kPanelMinH, jp["h"].get<float>());
            if (jp.contains("visible")) p.visible = jp["visible"].get<bool>();
            if (jp.contains("allow_drag")) p.allow_drag = jp["allow_drag"].get<bool>();
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
    std::cout << "✓ Layout sauvegardé (last) -> " << kVizUISettingsFile << std::endl;
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
    std::cout << "✓ Layout sauvegardé (slot " << slot << ") -> " << kVizUISettingsFile << std::endl;
}

void Visualizer::loadUILayoutFromSlot(int slot) {
    slot = std::clamp(slot, 0, 9);
    json settings;
    if (!loadUISettings(settings)) {
        std::cout << "⚠️  Aucun settings UI trouvé: " << kVizUISettingsFile << std::endl;
        return;
    }
    try {
        if (!settings.contains("slots") || !settings["slots"].is_object()) {
            std::cout << "⚠️  Aucun slot dans settings UI" << std::endl;
            return;
        }
        const std::string key = std::to_string(slot);
        if (!settings["slots"].contains(key)) {
            std::cout << "⚠️  Slot UI inexistant: " << slot << std::endl;
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
                window->setSize(sf::Vector2u(static_cast<unsigned>(ww), static_cast<unsigned>(hh)));
                window_width = ww;
                window_height = hh;
                syncUIView();
            }
        } catch (...) {
            // ignore resize errors
        }

        const bool ok = applyUILayout(layout);
        if (ok) {
            std::cout << "✓ Layout chargé (slot " << slot << ")" << std::endl;
        } else {
            std::cout << "⚠️  Échec application layout (slot " << slot << ")" << std::endl;
        }
    } catch (...) {
        std::cout << "⚠️  Settings UI invalides" << std::endl;
    }
}

void Visualizer::syncUIView() {
    if (!window) return;
    // View UI en coordonnées pixels: (0..w, 0..h)
    // On la force explicitement pour éviter toute déformation au resize.
    const float ww = static_cast<float>(std::max(1, window_width));
    const float hh = static_cast<float>(std::max(1, window_height));
    sf::View v(sf::FloatRect(0.f, 0.f, ww, hh));
    v.setViewport(sf::FloatRect(0.f, 0.f, 1.f, 1.f));
    window->setView(v);
}

bool Visualizer::isOpen() const {
    return window && window->isOpen();
}

bool Visualizer::isPanelVisible(PanelId id) const {
    return panels_[static_cast<size_t>(id)].visible;
}

namespace {
sf::Color with_alpha(sf::Color c, uint8_t a) {
    c.a = a;
    return c;
}
} // namespace

sf::Color Visualizer::panelAccent(PanelId id) const {
    switch (id) {
        case PanelId::Context: return sf::Color(70, 130, 200, 255);
        case PanelId::Blocks: return sf::Color(160, 110, 210, 255);
        case PanelId::Generated: return sf::Color(70, 170, 170, 255);
        case PanelId::Training: return sf::Color(90, 180, 100, 255);
        case PanelId::Metrics: return sf::Color(210, 150, 70, 255);
        case PanelId::Graph: return sf::Color(200, 90, 90, 255);
        default: return sf::Color(140, 140, 150, 255);
    }
}

sf::FloatRect Visualizer::panelRect(PanelId id) const {
    const auto& p = panels_[static_cast<size_t>(id)];
    return sf::FloatRect(p.pos.x, p.pos.y, p.size.x, p.size.y);
}

sf::FloatRect Visualizer::panelTitleRect(PanelId id) const {
    const auto& p = panels_[static_cast<size_t>(id)];
    return sf::FloatRect(p.pos.x, p.pos.y, p.size.x, std::min(kPanelTitleH, p.size.y));
}

sf::FloatRect Visualizer::panelContentRect(PanelId id) const {
    const auto& p = panels_[static_cast<size_t>(id)];
    const float top = p.pos.y + std::min(kPanelTitleH, p.size.y);
    const float inner_w = std::max(0.f, p.size.x - 2.f * kPanelPad);
    const float inner_h = std::max(0.f, p.size.y - std::min(kPanelTitleH, p.size.y) - 2.f * kPanelPad);
    return sf::FloatRect(p.pos.x + kPanelPad, top + kPanelPad, inner_w, inner_h);
}

sf::FloatRect Visualizer::panelResizeHandleRect(PanelId id) const {
    const auto& p = panels_[static_cast<size_t>(id)];
    const float w = std::max(0.f, p.size.x);
    const float h = std::max(0.f, p.size.y);
    const float hs = std::min(kPanelResizeHandle, std::min(w, h));
    return sf::FloatRect(p.pos.x + w - hs, p.pos.y + h - hs, hs, hs);
}

sf::FloatRect Visualizer::panelCloseButtonRect(PanelId id) const {
    const auto tr = panelTitleRect(id);
    const float s = std::max(10.f, std::min(18.f, tr.height - 4.f));
    const float x = tr.left + tr.width - s - 4.f;
    const float y = tr.top + (tr.height - s) * 0.5f;
    return sf::FloatRect(x, y, s, s);
}

std::optional<Visualizer::PanelId> Visualizer::hitTestPanelTitle(const sf::Vector2f& mouse) const {
    // Hit-test du haut vers le bas: si des panneaux se chevauchent,
    // le dernier dessiné (graph/metrics) peut être prioritaire.
    const PanelId order[] = { PanelId::Graph, PanelId::Metrics, PanelId::Training, PanelId::Generated, PanelId::Blocks, PanelId::Context };
    for (PanelId id : order) {
        if (!isPanelVisible(id)) continue;
        const auto& p = panels_[static_cast<size_t>(id)];
        if (!p.allow_drag) continue;
        if (panelTitleRect(id).contains(mouse)) return id;
    }
    return std::nullopt;
}

std::optional<Visualizer::PanelId> Visualizer::hitTestPanelResizeHandle(const sf::Vector2f& mouse) const {
    const PanelId order[] = { PanelId::Graph, PanelId::Metrics, PanelId::Training, PanelId::Generated, PanelId::Blocks, PanelId::Context };
    for (PanelId id : order) {
        if (!isPanelVisible(id)) continue;
        const auto& p = panels_[static_cast<size_t>(id)];
        if (!p.allow_drag) continue;
        if (panelResizeHandleRect(id).contains(mouse)) return id;
    }
    return std::nullopt;
}

std::optional<Visualizer::PanelId> Visualizer::hitTestPanelCloseButton(const sf::Vector2f& mouse) const {
    const PanelId order[] = { PanelId::Graph, PanelId::Metrics, PanelId::Training, PanelId::Generated, PanelId::Blocks, PanelId::Context };
    for (PanelId id : order) {
        if (!isPanelVisible(id)) continue;
        if (panelCloseButtonRect(id).contains(mouse)) return id;
    }
    return std::nullopt;
}

void Visualizer::initCursorsIfNeeded() {
    if (cursors_loaded_) return;
    // Best-effort: certains backends peuvent refuser certains curseurs.
    cursor_ok_arrow_ = cursor_arrow_.loadFromSystem(sf::Cursor::Arrow);
    cursor_ok_hand_ = cursor_hand_.loadFromSystem(sf::Cursor::Hand);
    cursor_ok_cross_ = cursor_cross_.loadFromSystem(sf::Cursor::Cross);
    // Diagonal resize (NW-SE) pour handle bottom-right.
    cursor_ok_resize_ = cursor_resize_.loadFromSystem(sf::Cursor::SizeTopLeftBottomRight);
    cursors_loaded_ = true;
}

void Visualizer::setCursor(CursorKind kind) {
    if (!window) return;
    if (cursor_kind_ == kind) return;
    initCursorsIfNeeded();

    const sf::Cursor* c = nullptr;
    if (kind == CursorKind::Hand && cursor_ok_hand_) c = &cursor_hand_;
    else if (kind == CursorKind::Cross && cursor_ok_cross_) c = &cursor_cross_;
    else if (kind == CursorKind::Resize && cursor_ok_resize_) c = &cursor_resize_;
    else if (cursor_ok_arrow_) c = &cursor_arrow_;

    if (c) {
        window->setMouseCursor(*c);
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

    const float margin = 20.f;
    const float right_w = 520.f;
    const float min_left_w = 520.f;

    const float win_w = static_cast<float>(window_width);
    const float win_h = static_cast<float>(window_height);

    const float left_w = std::clamp(win_w - right_w - 3.f * margin, min_left_w, std::max(min_left_w, win_w - 2.f * margin));
    const float right_x = margin + left_w + margin;
    const float right_safe_w = std::max(320.f, std::min(right_w, win_w - right_x - margin));

    panels_[static_cast<size_t>(PanelId::Context)] = Panel{ sf::Vector2f(margin, margin), sf::Vector2f(left_w, 280.f), "Context", true, true };
    panels_[static_cast<size_t>(PanelId::Blocks)] = Panel{ sf::Vector2f(margin, margin + 300.f), sf::Vector2f(left_w, 320.f), "Blocks / Layers", true, true };
    panels_[static_cast<size_t>(PanelId::Generated)] = Panel{ sf::Vector2f(margin, margin + 640.f), sf::Vector2f(left_w, 280.f), "Generated", true, true };

    panels_[static_cast<size_t>(PanelId::Training)] = Panel{ sf::Vector2f(right_x, margin), sf::Vector2f(right_safe_w, 70.f), "Training", true, true };
    panels_[static_cast<size_t>(PanelId::Graph)] = Panel{ sf::Vector2f(right_x, std::max(margin + 80.f, win_h - 240.f - margin)), sf::Vector2f(right_safe_w, 240.f), "Loss", true, true };
    {
        const float metrics_y = margin + 90.f;
        const float metrics_h = std::max(220.f, win_h - metrics_y - panels_[static_cast<size_t>(PanelId::Graph)].size.y - margin);
        panels_[static_cast<size_t>(PanelId::Metrics)] = Panel{ sf::Vector2f(right_x, metrics_y), sf::Vector2f(right_safe_w, metrics_h), "Metrics", true, true };
    }

    clampPanelsToWindow();
}

void Visualizer::drawPanelChrome(PanelId id) {
    if (!window) return;
    if (!isPanelVisible(id)) return;
    const auto& p = panels_[static_cast<size_t>(id)];
    if (p.size.x <= 1.f || p.size.y <= 1.f) return;

    const sf::Color accent = panelAccent(id);

    sf::RectangleShape bg(p.size);
    bg.setPosition(p.pos);
    bg.setFillColor(sf::Color(18, 18, 22, 170));
    bg.setOutlineColor(with_alpha(accent, 180));
    bg.setOutlineThickness(2);
    window->draw(bg);

    const float th = std::min(kPanelTitleH, p.size.y);
    sf::RectangleShape title(sf::Vector2f(p.size.x, th));
    title.setPosition(p.pos);
    title.setFillColor(with_alpha(accent, 120));
    window->draw(title);

    if (font_loaded && !p.title.empty()) {
        sf::Text t;
        t.setFont(font);
        t.setCharacterSize(14);
        t.setFillColor(sf::Color(235, 235, 240));
        t.setPosition(p.pos.x + 8.f, p.pos.y + 2.f);
        t.setString(sf::String::fromUtf8(p.title.begin(), p.title.end()));
        window->draw(t);
    }

    // Poignée de redimensionnement (bas-droite)
    {
        const auto r = panelResizeHandleRect(id);
        sf::RectangleShape h(sf::Vector2f(r.width, r.height));
        h.setPosition(r.left, r.top);
        h.setFillColor(sf::Color(80, 80, 95, 210));
        window->draw(h);
    }

    // Bouton close [X]
    {
        const auto r = panelCloseButtonRect(id);
        sf::RectangleShape b(sf::Vector2f(r.width, r.height));
        b.setPosition(r.left, r.top);
        b.setFillColor(sf::Color(90, 40, 40, 220));
        b.setOutlineColor(sf::Color(180, 80, 80, 230));
        b.setOutlineThickness(1);
        window->draw(b);

        if (font_loaded) {
            sf::Text t;
            t.setFont(font);
            t.setCharacterSize(14);
            t.setFillColor(sf::Color(245, 240, 240));
            t.setPosition(r.left + 4.f, r.top - 2.f);
            t.setString("X");
            window->draw(t);
        }
    }
}

void Visualizer::processEvents() {
    if (!window) return;

    // Garantir que les coordonnées UI (pixels) sont actives avant tout hit-test.
    // Sinon mapPixelToCoords() peut utiliser une view différente (ex: clipping du panel Blocks)
    // et les clics ne tombent pas dans les hitboxes.
    syncUIView();

    sf::Event event;
    while (window->pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            window->close();
        } else if (event.type == sf::Event::Resized) {
            window_width = static_cast<int>(event.size.width);
            window_height = static_cast<int>(event.size.height);
            // IMPORTANT: si la view n'est pas mise à jour, SFML étire le rendu.
            syncUIView();
            // NOTE: les panneaux sont indépendants de la taille de fenêtre.
            // On ne recale (clamp) pas automatiquement leurs positions/dimensions.
        } else if (event.type == sf::Event::MouseWheelScrolled) {
            if (zoom_active_ || dragging_panel_ || resizing_panel_) {
                continue;
            }

            initDefaultPanelsIfNeeded();
            syncUIView();
            const sf::Vector2f mouse = window->mapPixelToCoords(sf::Vector2i(event.mouseWheelScroll.x, event.mouseWheelScroll.y));

            // Scroll uniquement dans Blocks/Layers (layers)
            if (isPanelVisible(PanelId::Blocks)) {
                const auto content = panelContentRect(PanelId::Blocks);
                if (content.contains(mouse)) {
                    const float delta = event.mouseWheelScroll.delta;
                    blocks_scroll_y_ -= delta * kBlocksScrollSpeed;
                    blocks_scroll_y_ = std::clamp(blocks_scroll_y_, 0.0f, std::max(0.0f, blocks_scroll_max_));
                    continue;
                }
            }
        } else if (event.type == sf::Event::MouseButtonPressed) {
            if (event.mouseButton.button == sf::Mouse::Left) {
                initDefaultPanelsIfNeeded();
                syncUIView();
                const sf::Vector2f mouse = window->mapPixelToCoords(sf::Vector2i(event.mouseButton.x, event.mouseButton.y));

                // STOP training button (dans le panneau Metrics)
                if (isPanelVisible(PanelId::Metrics) && last_stop_button_rect_.has_value() && last_stop_button_rect_->contains(mouse)) {
                    requestStopTraining();
                    continue;
                }

                // Live tuning controls (Metrics): click/drag updates atomics + bumps version.
                if (isPanelVisible(PanelId::Metrics) && !zoom_active_) {
                    auto commit_live = [&]() {
                        // Toute interaction avec les contrôles active le mode LIVE.
                        live_overrides_enabled_.store(true, std::memory_order_relaxed);
                        live_lr_.store(live_ui_lr_, std::memory_order_relaxed);
                        live_lr_warmup_steps_.store(live_ui_lr_warmup_steps_, std::memory_order_relaxed);
                        live_kl_beta_.store(live_ui_kl_beta_, std::memory_order_relaxed);
                        live_kl_warmup_steps_.store(live_ui_kl_warmup_steps_, std::memory_order_relaxed);
                        live_kl_enabled_.store(live_ui_kl_enabled_, std::memory_order_relaxed);
                        live_params_version_.fetch_add(1, std::memory_order_relaxed);
                    };

                    // Toggle LIVE/NATIVE
                    if (last_live_overrides_box_.has_value() && last_live_overrides_box_->contains(mouse)) {
                        const bool cur = live_overrides_enabled_.load(std::memory_order_relaxed);
                        live_overrides_enabled_.store(!cur, std::memory_order_relaxed);
                        live_params_version_.fetch_add(1, std::memory_order_relaxed);
                        // Ne change pas les valeurs UI; le training décidera d'appliquer/ignorer.
                        continue;
                    }

                    auto t_from_mouse_x = [&](const sf::FloatRect& track) -> float {
                        const float denom = std::max(1.f, track.width);
                        return clamp01((mouse.x - track.left) / denom);
                    };

                    // Checkbox KL enable
                    if (last_live_kl_enable_box_.has_value() && last_live_kl_enable_box_->contains(mouse)) {
                        live_ui_kl_enabled_ = !live_ui_kl_enabled_;
                        commit_live();
                        continue;
                    }

                    auto start_drag = [&](LiveDragTarget target,
                                          const std::optional<sf::FloatRect>& track_opt,
                                          const std::optional<sf::FloatRect>& thumb_opt) -> bool {
                        if (!track_opt.has_value()) return false;
                        const bool hit = track_opt->contains(mouse) || (thumb_opt.has_value() && thumb_opt->contains(mouse));
                        if (!hit) return false;
                        live_dragging_ = target;
                        const float t = t_from_mouse_x(*track_opt);
                        if (target == LiveDragTarget::LR) {
                            const float new_lr = lr_from_t(t);
                            if (std::fabs(new_lr - live_ui_lr_) > 0.0f) {
                                live_ui_lr_ = new_lr;
                                commit_live();
                            }
                        } else if (target == LiveDragTarget::LRWarmup) {
                            const int new_steps = warmup_from_t(t);
                            if (new_steps != live_ui_lr_warmup_steps_) {
                                live_ui_lr_warmup_steps_ = new_steps;
                                commit_live();
                            }
                        } else if (target == LiveDragTarget::KLBeta) {
                            const float new_beta = klbeta_from_t(t);
                            if (std::fabs(new_beta - live_ui_kl_beta_) > 1e-6f) {
                                live_ui_kl_beta_ = new_beta;
                                commit_live();
                            }
                        } else if (target == LiveDragTarget::KLWarmup) {
                            const int new_steps = warmup_from_t(t);
                            if (new_steps != live_ui_kl_warmup_steps_) {
                                live_ui_kl_warmup_steps_ = new_steps;
                                commit_live();
                            }
                        }
                        return true;
                    };

                    if (start_drag(LiveDragTarget::LR, last_live_lr_track_, last_live_lr_thumb_)) continue;
                    if (start_drag(LiveDragTarget::LRWarmup, last_live_lrwu_track_, last_live_lrwu_thumb_)) continue;
                    if (start_drag(LiveDragTarget::KLBeta, last_live_klb_track_, last_live_klb_thumb_)) continue;
                    if (start_drag(LiveDragTarget::KLWarmup, last_live_klwu_track_, last_live_klwu_thumb_)) continue;
                }

                // Le zoom overlay est modal: on bloque les interactions de layout,
                // mais on laisse le STOP fonctionner (géré au-dessus).
                if (zoom_active_) {
                    continue;
                }

                // Close button
                if (const auto hit_close = hitTestPanelCloseButton(mouse); hit_close.has_value()) {
                    const PanelId id = *hit_close;
                    panels_[static_cast<size_t>(id)].visible = false;
                    // Si le panneau est lié à un toggle, le refléter.
                    if (id == PanelId::Graph) show_loss_graph = false;
                    if (id == PanelId::Training) show_training_progress = false;
                    if (id == PanelId::Generated) show_generated_images = false;
                    dragging_panel_ = false;
                    resizing_panel_ = false;
                    continue;
                }

                // Scrollbar (slicer) dans Blocks/Layers (prioritaire sur drag panneau)
                if (isPanelVisible(PanelId::Blocks) && blocks_scroll_max_ > 0.0f) {
                    const auto content = panelContentRect(PanelId::Blocks);
                    if (content.contains(mouse)) {
                        if (last_blocks_scroll_thumb_rect_.contains(mouse)) {
                            dragging_blocks_scrollbar_ = true;
                            blocks_scroll_drag_grab_y_ = mouse.y - last_blocks_scroll_thumb_rect_.top;
                            continue;
                        }
                        if (last_blocks_scroll_track_rect_.contains(mouse)) {
                            // Click sur la track: jump au ratio correspondant (thumb centré sous le curseur)
                            const float track_h = std::max(1.0f, last_blocks_scroll_track_rect_.height);
                            const float thumb_h = std::max(1.0f, last_blocks_scroll_thumb_rect_.height);
                            const float denom = std::max(1.0f, track_h - thumb_h);
                            const float y_in_track = std::clamp(mouse.y - last_blocks_scroll_track_rect_.top - thumb_h * 0.5f, 0.0f, denom);
                            const float t = y_in_track / denom;
                            blocks_scroll_y_ = t * blocks_scroll_max_;
                            blocks_scroll_y_ = std::clamp(blocks_scroll_y_, 0.0f, std::max(0.0f, blocks_scroll_max_));
                            dragging_blocks_scrollbar_ = true;
                            blocks_scroll_drag_grab_y_ = thumb_h * 0.5f;
                            continue;
                        }
                    }
                }

                // Sélection images au clic (Context / Generated / Blocks)
                // Note: on s'appuie sur les rects calculés au rendu précédent.
                bool handled_click = false;
                if (isPanelVisible(PanelId::Context)) {
                    if (last_dataset_rect_.has_value() && last_dataset_rect_->contains(mouse)) {
                        focus_target_ = FocusTarget::Dataset;
                        handled_click = true;
                    }
                    if (last_projection_rect_.has_value() && last_projection_rect_->contains(mouse)) {
                        focus_target_ = FocusTarget::Projection;
                        handled_click = true;
                    }
                    if (last_understanding_rect_.has_value() && last_understanding_rect_->contains(mouse)) {
                        focus_target_ = FocusTarget::Understanding;
                        handled_click = true;
                    }
                }

                if (isPanelVisible(PanelId::Generated) && !last_generated_rects_.empty() && last_generated_rects_.size() == last_generated_indices_.size()) {
                    for (size_t i = 0; i < last_generated_rects_.size(); ++i) {
                        if (last_generated_rects_[i].contains(mouse)) {
                            focus_target_ = FocusTarget::Generated;
                            focus_generated_index_ = std::max(0, last_generated_indices_[i]);
                            handled_click = true;
                            break;
                        }
                    }
                }

                if (isPanelVisible(PanelId::Blocks) && !last_block_rects_.empty()) {
                    for (size_t i = 0; i < last_block_rects_.size(); ++i) {
                        if (last_block_rects_[i].contains(mouse)) {
                            focus_target_ = FocusTarget::LayerBlock;
                            focus_block_index_ = static_cast<int>(i);
                            handled_click = true;
                            break;
                        }
                    }
                }

                if (handled_click) {
                    continue;
                }

                // Priorité au resize-handle
                const auto hit_resize = hitTestPanelResizeHandle(mouse);
                if (hit_resize.has_value()) {
                    resized_panel_ = *hit_resize;
                    resizing_panel_ = true;
                    resize_start_mouse_ = mouse;
                    const auto& p = panels_[static_cast<size_t>(resized_panel_)];
                    resize_start_size_ = p.size;
                } else {
                    const auto hit = hitTestPanelTitle(mouse);
                    if (hit.has_value()) {
                        dragged_panel_ = *hit;
                        dragging_panel_ = true;
                        const auto& p = panels_[static_cast<size_t>(dragged_panel_)];
                        drag_grab_offset_ = sf::Vector2f(mouse.x - p.pos.x, mouse.y - p.pos.y);
                    }
                }
            }
        } else if (event.type == sf::Event::MouseButtonReleased) {
            if (event.mouseButton.button == sf::Mouse::Left) {
                dragging_panel_ = false;
                resizing_panel_ = false;
                dragging_blocks_scrollbar_ = false;
                live_dragging_ = LiveDragTarget::None;
            }
        } else if (event.type == sf::Event::MouseMoved) {
            syncUIView();
            const sf::Vector2f mouse = window->mapPixelToCoords(sf::Vector2i(event.mouseMove.x, event.mouseMove.y));

            // Drag live tuning controls (Metrics)
            if (!zoom_active_ && live_dragging_ != LiveDragTarget::None) {
                auto commit_live = [&]() {
                    live_overrides_enabled_.store(true, std::memory_order_relaxed);
                    live_lr_.store(live_ui_lr_, std::memory_order_relaxed);
                    live_lr_warmup_steps_.store(live_ui_lr_warmup_steps_, std::memory_order_relaxed);
                    live_kl_beta_.store(live_ui_kl_beta_, std::memory_order_relaxed);
                    live_kl_warmup_steps_.store(live_ui_kl_warmup_steps_, std::memory_order_relaxed);
                    live_kl_enabled_.store(live_ui_kl_enabled_, std::memory_order_relaxed);
                    live_params_version_.fetch_add(1, std::memory_order_relaxed);
                };

                auto t_from_mouse_x = [&](const sf::FloatRect& track) -> float {
                    const float denom = std::max(1.f, track.width);
                    return clamp01((mouse.x - track.left) / denom);
                };

                bool changed = false;
                if (live_dragging_ == LiveDragTarget::LR && last_live_lr_track_.has_value()) {
                    const float new_lr = lr_from_t(t_from_mouse_x(*last_live_lr_track_));
                    if (std::fabs(new_lr - live_ui_lr_) > 0.0f) {
                        live_ui_lr_ = new_lr;
                        changed = true;
                    }
                } else if (live_dragging_ == LiveDragTarget::LRWarmup && last_live_lrwu_track_.has_value()) {
                    const int new_steps = warmup_from_t(t_from_mouse_x(*last_live_lrwu_track_));
                    if (new_steps != live_ui_lr_warmup_steps_) {
                        live_ui_lr_warmup_steps_ = new_steps;
                        changed = true;
                    }
                } else if (live_dragging_ == LiveDragTarget::KLBeta && last_live_klb_track_.has_value()) {
                    const float new_beta = klbeta_from_t(t_from_mouse_x(*last_live_klb_track_));
                    if (std::fabs(new_beta - live_ui_kl_beta_) > 1e-6f) {
                        live_ui_kl_beta_ = new_beta;
                        changed = true;
                    }
                } else if (live_dragging_ == LiveDragTarget::KLWarmup && last_live_klwu_track_.has_value()) {
                    const int new_steps = warmup_from_t(t_from_mouse_x(*last_live_klwu_track_));
                    if (new_steps != live_ui_kl_warmup_steps_) {
                        live_ui_kl_warmup_steps_ = new_steps;
                        changed = true;
                    }
                }

                if (changed) commit_live();
                setCursor(CursorKind::Hand);
                continue;
            }

            // Drag scrollbar Blocks/Layers
            if (dragging_blocks_scrollbar_ && !zoom_active_) {
                const float track_h = std::max(1.0f, last_blocks_scroll_track_rect_.height);
                const float thumb_h = std::max(1.0f, last_blocks_scroll_thumb_rect_.height);
                const float denom = std::max(1.0f, track_h - thumb_h);
                const float y_in_track = std::clamp(mouse.y - last_blocks_scroll_track_rect_.top - blocks_scroll_drag_grab_y_, 0.0f, denom);
                const float t = y_in_track / denom;
                blocks_scroll_y_ = t * blocks_scroll_max_;
                blocks_scroll_y_ = std::clamp(blocks_scroll_y_, 0.0f, std::max(0.0f, blocks_scroll_max_));
                setCursor(CursorKind::Hand);
                continue;
            }

            // Curseur responsive si pas en interaction
            if (!zoom_active_ && !dragging_panel_ && !resizing_panel_) {
                // Hover scrollbar Blocks/Layers
                if (isPanelVisible(PanelId::Blocks) && blocks_scroll_max_ > 0.0f &&
                    (last_blocks_scroll_thumb_rect_.contains(mouse) || last_blocks_scroll_track_rect_.contains(mouse))) {
                    setCursor(CursorKind::Hand);
                } else if (isPanelVisible(PanelId::Metrics) &&
                           ((last_live_overrides_box_.has_value() && last_live_overrides_box_->contains(mouse)) ||
                            (last_live_kl_enable_box_.has_value() && last_live_kl_enable_box_->contains(mouse)) ||
                            (last_live_lr_track_.has_value() && last_live_lr_track_->contains(mouse)) ||
                            (last_live_lr_thumb_.has_value() && last_live_lr_thumb_->contains(mouse)) ||
                            (last_live_lrwu_track_.has_value() && last_live_lrwu_track_->contains(mouse)) ||
                            (last_live_lrwu_thumb_.has_value() && last_live_lrwu_thumb_->contains(mouse)) ||
                            (last_live_klb_track_.has_value() && last_live_klb_track_->contains(mouse)) ||
                            (last_live_klb_thumb_.has_value() && last_live_klb_thumb_->contains(mouse)) ||
                            (last_live_klwu_track_.has_value() && last_live_klwu_track_->contains(mouse)) ||
                            (last_live_klwu_thumb_.has_value() && last_live_klwu_thumb_->contains(mouse)))) {
                    setCursor(CursorKind::Hand);
                } else if (hitTestPanelCloseButton(mouse).has_value()) {
                    setCursor(CursorKind::Cross);
                } else if (hitTestPanelResizeHandle(mouse).has_value()) {
                    setCursor(CursorKind::Resize);
                } else if (hitTestPanelTitle(mouse).has_value()) {
                    setCursor(CursorKind::Hand);
                } else {
                    setCursor(CursorKind::Arrow);
                }
            }

            if (resizing_panel_ && !zoom_active_) {
                setCursor(CursorKind::Resize);
                auto& p = panels_[static_cast<size_t>(resized_panel_)];
                const sf::Vector2f delta(mouse.x - resize_start_mouse_.x, mouse.y - resize_start_mouse_.y);
                p.size.x = std::max(kPanelMinW, resize_start_size_.x + delta.x);
                p.size.y = std::max(kPanelMinH, resize_start_size_.y + delta.y);
                // Pas de clamp automatique à la fenêtre: taille/panneau indépendant.
            } else if (dragging_panel_ && !zoom_active_) {
                setCursor(CursorKind::Hand);
                auto& p = panels_[static_cast<size_t>(dragged_panel_)];
                p.pos = sf::Vector2f(mouse.x - drag_grab_offset_.x, mouse.y - drag_grab_offset_.y);
                // On peut clamp pendant un drag, pour éviter de perdre un panneau.
                clampPanelsToWindow();
            }
        } else if (event.type == sf::Event::KeyPressed) {
            const bool ctrl = event.key.control;
            const bool shift = event.key.shift;

            // Chord S / S+num
            auto key_to_digit = [](sf::Keyboard::Key k) -> int {
                if (k >= sf::Keyboard::Num0 && k <= sf::Keyboard::Num9) return (int)k - (int)sf::Keyboard::Num0;
                if (k >= sf::Keyboard::Numpad0 && k <= sf::Keyboard::Numpad9) return (int)k - (int)sf::Keyboard::Numpad0;
                return -1;
            };
            const int digit = key_to_digit(event.key.code);

            if (event.key.code == sf::Keyboard::S) {
                save_chord_armed_ = true;
                save_chord_consumed_ = false;
                save_chord_armed_ms_ = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count());
            }

            // Slots UI:
            //  - S+0..9 : sauvegarder un slot
            //  - 0..9   : charger un slot
            if (digit >= 0) {
                const bool s_down = sf::Keyboard::isKeyPressed(sf::Keyboard::S);
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
                continue; // ne pas déclencher d'autres binds sur les chiffres
            }

            // Help
            if (event.key.code == sf::Keyboard::H) {
                show_help_overlay_ = !show_help_overlay_;
            }

            // Toggle graph
            if (event.key.code == sf::Keyboard::G) {
                show_loss_graph = !show_loss_graph;
                panels_[static_cast<size_t>(PanelId::Graph)].visible = show_loss_graph;
            }

            // Toggle prompt text panel
            if (event.key.code == sf::Keyboard::P) {
                show_prompt_text_ = !show_prompt_text_;
            }

            // Zoom overlay
            if (event.key.code == sf::Keyboard::Z || event.key.code == sf::Keyboard::Enter) {
                zoom_active_ = !zoom_active_;
            }
            if (event.key.code == sf::Keyboard::Escape) {
                zoom_active_ = false;
            }

            // Refresh (rebuild textures + reload architecture hints)
            if (event.key.code == sf::Keyboard::R) {
                requestResync();
                rebuildAllTextures();
                architecture_loaded = false;
                arch_layer_names.clear();
                arch_tensor_outputs.clear();
                arch_tensor_sinks.clear();
            }

            // Focus target
            if (event.key.code == sf::Keyboard::Tab) {
                if (focus_target_ == FocusTarget::Dataset) focus_target_ = FocusTarget::Projection;
                else if (focus_target_ == FocusTarget::Projection) focus_target_ = FocusTarget::Understanding;
                else if (focus_target_ == FocusTarget::Understanding) focus_target_ = FocusTarget::LayerBlock;
                else if (focus_target_ == FocusTarget::LayerBlock) focus_target_ = FocusTarget::Generated;
                else focus_target_ = FocusTarget::Dataset;
            }
            if (event.key.code == sf::Keyboard::F1) focus_target_ = FocusTarget::Dataset;
            if (event.key.code == sf::Keyboard::F2) focus_target_ = FocusTarget::Projection;
            if (event.key.code == sf::Keyboard::F3) focus_target_ = FocusTarget::Understanding;
            if (event.key.code == sf::Keyboard::F4) focus_target_ = FocusTarget::LayerBlock;
            if (event.key.code == sf::Keyboard::F5) focus_target_ = FocusTarget::Generated;

            // Resize window (X/Y)
            if (ctrl && (event.key.code == sf::Keyboard::Left || event.key.code == sf::Keyboard::Right ||
                         event.key.code == sf::Keyboard::Up || event.key.code == sf::Keyboard::Down)) {
                const int step = shift ? 128 : 64;
                auto sz = window->getSize();
                int ww = static_cast<int>(sz.x);
                int hh = static_cast<int>(sz.y);
                if (event.key.code == sf::Keyboard::Left) ww -= step;
                if (event.key.code == sf::Keyboard::Right) ww += step;
                if (event.key.code == sf::Keyboard::Up) hh -= step;
                if (event.key.code == sf::Keyboard::Down) hh += step;
                ww = std::max(640, ww);
                hh = std::max(480, hh);
                window->setSize(sf::Vector2u(static_cast<unsigned>(ww), static_cast<unsigned>(hh)));
                window_width = ww;
                window_height = hh;
                // Même quand on resize par code, resynchroniser la view.
                syncUIView();
                // Ne pas modifier les panneaux au resize de fenêtre.
            } else {
                // Navigate layer blocks when focused
                if (focus_target_ == FocusTarget::LayerBlock &&
                    (event.key.code == sf::Keyboard::Left || event.key.code == sf::Keyboard::Right)) {
                    const int n = static_cast<int>(layer_block_images.size());
                    if (n > 0) {
                        if (event.key.code == sf::Keyboard::Left) focus_block_index_ = (focus_block_index_ - 1 + n) % n;
                        if (event.key.code == sf::Keyboard::Right) focus_block_index_ = (focus_block_index_ + 1) % n;
                    }
                }
            }
        }
        else if (event.type == sf::Event::KeyReleased) {
            if (event.key.code == sf::Keyboard::S) {
                if (save_chord_armed_ && !save_chord_consumed_) {
                    initDefaultPanelsIfNeeded();
                    saveUILayoutToLast();
                }
                save_chord_armed_ = false;
                save_chord_consumed_ = false;
            }
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
    p.version = live_params_version_.load(std::memory_order_relaxed);
    return p;
}

void Visualizer::update() {
    if (!enabled) return;
    if (!window || !window->isOpen()) return;

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

    window->clear(sf::Color(30, 30, 35)); // Fond sombre

    renderBackground();

    drawPanelChrome(PanelId::Context);
    renderContextImages();

    drawPanelChrome(PanelId::Blocks);
    renderLayerBlocks();

    if (show_generated_images) {
        drawPanelChrome(PanelId::Generated);
        renderGeneratedImages();
    }

    if (show_training_progress) {
        drawPanelChrome(PanelId::Training);
        renderTrainingProgress();
    }

    drawPanelChrome(PanelId::Metrics);
    renderMetrics();

    if (show_loss_graph) {
        drawPanelChrome(PanelId::Graph);
        renderLossGraph();
    }

    renderFocusOutline();

    if (zoom_active_) {
        renderZoomOverlay();
    }
    if (show_help_overlay_) {
        renderHelpOverlay();
    }

    window->display();
}

void Visualizer::renderFocusOutline() {
    if (!window) return;
    if (zoom_active_) return; // le zoom a déjà sa propre emphase

    // Ne pas afficher si le panneau associé est fermé
    if (focus_target_ == FocusTarget::LayerBlock) {
        if (!isPanelVisible(PanelId::Blocks)) return;
    } else if (focus_target_ == FocusTarget::Generated) {
        if (!isPanelVisible(PanelId::Generated)) return;
    } else {
        if (!isPanelVisible(PanelId::Context)) return;
    }

    std::optional<sf::FloatRect> r;
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
    }

    if (!r.has_value()) return;
    if (r->width <= 0.f || r->height <= 0.f) return;

    // Accent selon la zone focusée
    sf::Color col = sf::Color(245, 245, 250, 230);
    if (focus_target_ == FocusTarget::LayerBlock) {
        col = with_alpha(panelAccent(PanelId::Blocks), 235);
    } else if (focus_target_ == FocusTarget::Generated) {
        col = with_alpha(panelAccent(PanelId::Generated), 235);
    } else {
        col = with_alpha(panelAccent(PanelId::Context), 235);
    }

    // PS (user): quand on scrolle dans Blocks/Layers, ne pas laisser l'outline visible
    // si l'élément focus sort de la zone réellement visible.
    if (focus_target_ == FocusTarget::LayerBlock) {
        const auto area = panelContentRect(PanelId::Blocks);
        const float ww = static_cast<float>(std::max(1, window_width));
        const float hh = static_cast<float>(std::max(1, window_height));

        const float left = std::clamp(area.left, 0.0f, ww);
        const float top = std::clamp(area.top, 0.0f, hh);
        const float right = std::clamp(area.left + area.width, 0.0f, ww);
        const float bottom = std::clamp(area.top + area.height, 0.0f, hh);
        const float w = std::max(0.0f, right - left);
        const float h = std::max(0.0f, bottom - top);
        if (w <= 0.0f || h <= 0.0f) return;

        const sf::FloatRect clip(left, top, w, h);
        sf::FloatRect inter;
        if (!clip.intersects(*r, inter)) {
            // Hors zone visible => désactiver l'overline.
            return;
        }

        const sf::View old_view = window->getView();
        sf::View v(sf::FloatRect(left, top, w, h));
        v.setViewport(sf::FloatRect(left / ww, top / hh, w / ww, h / hh));
        window->setView(v);

        sf::RectangleShape outline(sf::Vector2f(r->width, r->height));
        outline.setPosition(r->left, r->top);
        outline.setFillColor(sf::Color::Transparent);
        outline.setOutlineColor(col);
        outline.setOutlineThickness(3);
        window->draw(outline);

        window->setView(old_view);
        return;
    }

    sf::RectangleShape outline(sf::Vector2f(r->width, r->height));
    outline.setPosition(r->left, r->top);
    outline.setFillColor(sf::Color::Transparent);
    outline.setOutlineColor(col);
    outline.setOutlineThickness(3);
    window->draw(outline);
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
    if (has_output_thumb_) rebuild_one(output_thumb_);
    for (auto& img : layer_block_images) rebuild_one(img);
}

void Visualizer::renderHelpOverlay() {
    if (!window) return;

    sf::RectangleShape bg(sf::Vector2f(static_cast<float>(window_width), static_cast<float>(window_height)));
    bg.setPosition(0, 0);
    bg.setFillColor(sf::Color(0, 0, 0, 160));
    window->draw(bg);

    if (!font_loaded) return;
    const int x = 24;
    int y = 24;
    const int lh = 18;
    auto line = [&](const std::string& s) {
        sf::Text t;
        t.setFont(font);
        t.setCharacterSize(14);
        t.setFillColor(sf::Color(235, 235, 240));
        t.setPosition(static_cast<float>(x), static_cast<float>(y));
        t.setString(sf::String::fromUtf8(s.begin(), s.end()));
        window->draw(t);
        y += lh;
    };

    line("Aide (clavier)");
    line("H : afficher/masquer cette aide");
    line("Z ou Entrée : grossir / réduire l'image sélectionnée");
    line("Tab / F1-F5 : sélectionner (dataset / projection / understanding / blocks / generated)");
    line("←/→ : naviguer dans les blocks (si focus=blocks)");
    line("G : afficher/masquer le graph");
    line("P : afficher/masquer le texte du prompt");
    line("R : actualiser (rebuild textures + reload architecture)");
    line("S : sauvegarder la structure UI (layout last)");
    line("S+0..9 : sauvegarder la structure UI dans un slot");
    line("0..9 : charger/appliquer un slot UI");
    line("Ctrl+←/→ : réduire/agrandir la fenêtre (X)");
    line("Ctrl+↑/↓ : réduire/agrandir la fenêtre (Y)");
    line("Esc : fermer le zoom");
    line("Souris (layout)");
    line("Glisser-déposer sur le titre d'un panneau pour le déplacer");
    line("Glisser la poignée en bas-droite pour redimensionner un panneau");
    line("Cliquer sur [X] dans l'entête pour fermer un panneau");
}

void Visualizer::renderZoomOverlay() {
    if (!window) return;

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
    }

    if (!img) return;
    const auto ts = img->texture.getSize();
    if (ts.x == 0 || ts.y == 0) return;

    if (!label.empty()) {
        label += " (" + std::to_string(img->w) + "x" + std::to_string(img->h) + "x" + std::to_string(img->channels) + ")";
    }

    sf::RectangleShape bg(sf::Vector2f(static_cast<float>(window_width), static_cast<float>(window_height)));
    bg.setPosition(0, 0);
    bg.setFillColor(sf::Color(0, 0, 0, 200));
    window->draw(bg);

    const float pad = 28.0f;
    const float box_w = static_cast<float>(window_width) - 2.0f * pad;
    const float box_h = static_cast<float>(window_height) - 2.0f * pad - 28.0f;
    const float box = std::min(box_w, box_h);

    sf::Sprite spr;
    spr.setTexture(img->texture, true);
    const float sx = box / std::max(1.0f, static_cast<float>(ts.x));
    const float sy = box / std::max(1.0f, static_cast<float>(ts.y));
    const float sc = std::min(sx, sy);
    spr.setScale(sc, sc);

    const float dw = static_cast<float>(ts.x) * sc;
    const float dh = static_cast<float>(ts.y) * sc;
    spr.setPosition((static_cast<float>(window_width) - dw) * 0.5f,
                    (static_cast<float>(window_height) - dh) * 0.5f);
    window->draw(spr);

    if (font_loaded && !label.empty()) {
        sf::Text t;
        t.setFont(font);
        t.setCharacterSize(14);
        t.setFillColor(sf::Color(235, 235, 240));
        t.setPosition(pad, pad);
        const std::string s = clamp_text_end(label, 120);
        t.setString(sf::String::fromUtf8(s.begin(), s.end()));
        window->draw(t);
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
            std::cout << "✓ Visualizer: architecture chargée: " << arch_layer_names.size()
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

void Visualizer::addGeneratedImage(const std::vector<uint8_t>& image, int w, int h, int channels, const std::string& prompt) {
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

    generated_images.push_back(std::move(img_data));

    // Conserver une vignette "Sortie" (120px) pour l'afficher aussi dans Blocks/Layers.
    output_thumb_.pixels = image;
    output_thumb_.prompt = prompt;
    output_thumb_.w = iw;
    output_thumb_.h = ih;
    output_thumb_.channels = ic;
    output_thumb_.display_size = 120;
    createImageTexture(output_thumb_, iw, ih, ic, 120);
    has_output_thumb_ = true;
    output_thumb_label_ = std::string("mimir/output | ") + prompt;

    int max_images = image_grid_cols * image_grid_rows;
    if (generated_images.size() > static_cast<size_t>(max_images)) {
        generated_images.erase(generated_images.begin());
    }
}

void Visualizer::setDatasetImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label) {
    if (!enabled) return;
    if (w <= 0 || h <= 0) return;
    if (channels != 1 && channels != 3 && channels != 4) return;

    dataset_image.pixels = pixels;
    dataset_image.prompt = label;
    dataset_label = label;
    dataset_image.w = w;
    dataset_image.h = h;
    dataset_image.channels = channels;
    dataset_image.display_size = 200;
    createImageTexture(dataset_image, w, h, channels, 200);
    has_dataset_image = true;
}

void Visualizer::setDatasetText(const std::string& raw_text, const std::string& tokenized, const std::string& encoded) {
    if (!enabled) return;
    dataset_text_raw = raw_text;
    dataset_text_tokens = tokenized;
    dataset_text_encoded = encoded;
    has_dataset_text = (!dataset_text_raw.empty() || !dataset_text_tokens.empty() || !dataset_text_encoded.empty());
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

void Visualizer::setLayerBlockImages(const std::vector<BlockFrame>& frames) {
    if (!enabled) return;

    layer_block_images.clear();
    layer_block_labels.clear();
    has_layer_blocks = false;

    if (frames.empty()) return;

    layer_block_images.reserve(frames.size());
    layer_block_labels.reserve(frames.size());

    for (const auto& f : frames) {
        if (f.w <= 0 || f.h <= 0) continue;
        if (f.channels != 1 && f.channels != 3 && f.channels != 4) continue;
        if (f.pixels.empty()) continue;

        const auto p = parse_viz_label(f.label);

        // PS (user): ne pas afficher les blocs d'activation.
        if (hide_activation_blocks) {
            if (p.tag == "ACT" || f.label.find("/activation") != std::string::npos) {
                continue;
            }
        }

        // PS (user): ne pas afficher la validation dans Blocks/Layers.
        // Best-effort: tag heuristique + substring explicite.
        if (p.tag == "VAL" || f.label.find("/validation") != std::string::npos || f.label.find("/val/") != std::string::npos) {
            continue;
        }

        ImageData img;
        img.pixels = f.pixels;
        img.prompt = f.label;
        img.w = f.w;
        img.h = f.h;
        img.channels = f.channels;
        img.display_size = 120;
        createImageTexture(img, f.w, f.h, f.channels, 120);
        layer_block_images.push_back(std::move(img));
        layer_block_labels.push_back(f.label);
    }

    has_layer_blocks = !layer_block_images.empty();
}

void Visualizer::updateMetrics(int epoch, int batch, float loss, float lr, float mse,
                              float kl, float wass, float ent, float mom, float spat, float temp,
                              float timestep,
                              int total_epochs, int total_batches, float avg_loss,
                              int batch_time_ms, size_t memory_mb, float bps, size_t params,
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
                              float kl_beta_effective) {
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
    record.params = params;
    record.mse = mse;
    record.kl_divergence = kl;
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
    full_loss_history.push_back(record);
    
    // Sauvegarder automatiquement l'historique après chaque ajout
    saveLossHistory(loss_log_file);
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
    has_output_thumb_ = false;
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
    sf::RectangleShape line(sf::Vector2f(1, window_height));
    line.setFillColor(sf::Color(40, 40, 45, 100));
    
    for (int x = 0; x < window_width; x += 50) {
        line.setPosition(x, 0);
        window->draw(line);
    }

    line.setSize(sf::Vector2f(window_width, 1));
    for (int y = 0; y < window_height; y += 50) {
        line.setPosition(0, y);
        window->draw(line);
    }

    // Splash logo au lancement
    if (logo_loaded_) {
        const float t = logo_clock_.getElapsedTime().asSeconds();
        if (t >= 0.0f && t < logo_splash_seconds_) {
            const auto lb = logo_sprite_.getLocalBounds();
            const float iw = std::max(1.0f, lb.width);
            const float ih = std::max(1.0f, lb.height);
            const float desired = std::min<float>(
                std::min<float>(window_width, window_height) * 0.35f,
                320.0f
            );
            const float scale = desired / std::max(iw, ih);
            logo_sprite_.setScale(scale, scale);
            logo_sprite_.setOrigin(lb.left + iw * 0.5f, lb.top + ih * 0.5f);
            logo_sprite_.setPosition(window_width * 0.5f, window_height * 0.5f);
            window->draw(logo_sprite_);
        }
    }
}

void Visualizer::renderContextImages() {
    // Affiche (si disponibles) l'image du dataset + la projection associée.
    const int img_display_size = 200;
    const int margin = 16;
    const auto area = panelContentRect(PanelId::Context);
    const int start_x = static_cast<int>(area.left);
    const int start_y = static_cast<int>(area.top);
    const int label_h = 20;

    // Clip au contenu du panneau (overflow: clip)
    const sf::View old_view = window->getView();
    struct ViewGuard {
        sf::RenderWindow* w;
        sf::View v;
        ~ViewGuard() {
            if (w) w->setView(v);
        }
    } guard{window.get(), old_view};
    {
        const float ww = static_cast<float>(std::max(1, window_width));
        const float hh = static_cast<float>(std::max(1, window_height));

        const float left = std::clamp(area.left, 0.0f, ww);
        const float top = std::clamp(area.top, 0.0f, hh);
        const float right = std::clamp(area.left + area.width, 0.0f, ww);
        const float bottom = std::clamp(area.top + area.height, 0.0f, hh);
        const float w = std::max(0.0f, right - left);
        const float h = std::max(0.0f, bottom - top);

        if (w > 0.0f && h > 0.0f) {
            sf::View v(sf::FloatRect(left, top, w, h));
            v.setViewport(sf::FloatRect(left / ww, top / hh, w / ww, h / hh));
            window->setView(v);
        }
    }

    auto toSfUtf8 = [](const std::string& s) -> sf::String {
        return sf::String::fromUtf8(s.begin(), s.end());
    };

    auto drawPanel = [&](int x, int y, ImageData& img, const std::string& label_text, sf::Color outline) -> sf::FloatRect {
        // Cadre
        sf::RectangleShape frame(sf::Vector2f(img_display_size + 4, img_display_size + 4));
        frame.setPosition(x - 2, y - 2);
        frame.setFillColor(sf::Color::Transparent);
        frame.setOutlineColor(outline);
        frame.setOutlineThickness(2);
        window->draw(frame);

        const sf::FloatRect frame_rect(static_cast<float>(x - 2), static_cast<float>(y - 2), static_cast<float>(img_display_size + 4), static_cast<float>(img_display_size + 4));

        // Image
        position_sprite_centered_in_box(img.sprite, static_cast<float>(x), static_cast<float>(y), static_cast<float>(img_display_size));
        window->draw(img.sprite);

        // Label bar
        sf::RectangleShape label(sf::Vector2f(img_display_size, label_h));
        label.setPosition(static_cast<float>(x), static_cast<float>(y + img_display_size + 5));
        label.setFillColor(sf::Color(50, 50, 60, 200));
        window->draw(label);

        if (font_loaded) {
            sf::Text t;
            t.setFont(font);
            t.setCharacterSize(14);
            t.setFillColor(sf::Color(230, 230, 235));
            t.setPosition(static_cast<float>(x + 6), static_cast<float>(y + img_display_size + 3));
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
        if (x + img_display_size > static_cast<int>(area.left + area.width)) {
            x = start_x;
            y += img_display_size + label_h + margin;
        }
    };

    auto fits_row = [&]() {
        return (y + img_display_size + label_h) <= static_cast<int>(area.top + area.height);
    };

    if (!fits_row()) return;

    if (has_dataset_image) {
        auto p = parse_viz_label(dataset_label.empty() ? std::string("dataset") : dataset_label);
        apply_arch_hint(p);
        std::string text = "[" + p.tag + "] ";
        if (!p.model.empty()) text += p.model + " ";
        text += p.short_text;
        if (!p.extra.empty()) text += " " + p.extra;
        last_dataset_rect_ = drawPanel(x, y, dataset_image, clamp_text_end(text, 44), color_for_tag(p.tag));
        advance();
        if (!fits_row()) return;
    }
    if (has_projection_image) {
        auto p = parse_viz_label(projection_label.empty() ? std::string("projection") : projection_label);
        apply_arch_hint(p);
        std::string text = "[" + p.tag + "] ";
        if (!p.model.empty()) text += p.model + " ";
        text += p.short_text;
        if (!p.extra.empty()) text += " " + p.extra;
        last_projection_rect_ = drawPanel(x, y, projection_image, clamp_text_end(text, 44), color_for_tag(p.tag));
        advance();
        if (!fits_row()) return;
    }
    if (has_understanding_image) {
        auto p = parse_viz_label(understanding_label.empty() ? std::string("understanding") : understanding_label);
        apply_arch_hint(p);
        std::string text = "[" + p.tag + "] ";
        if (!p.model.empty()) text += p.model + " ";
        text += p.short_text;
        if (!p.extra.empty()) text += " " + p.extra;
        last_understanding_rect_ = drawPanel(x, y, understanding_image, clamp_text_end(text, 44), color_for_tag(p.tag));
    }

}

void Visualizer::renderLayerBlocks() {
    const bool has_blocks = (has_layer_blocks && !layer_block_images.empty());
    const bool show_projection = (has_projection_thumb_ && has_projection_image && !projection_thumb_.pixels.empty());
    const bool show_output = (has_output_thumb_ && !output_thumb_.pixels.empty());

    const int extras = (show_projection ? 1 : 0) + (show_output ? 1 : 0);
    const int blocks_count = has_blocks ? static_cast<int>(layer_block_images.size()) : 0;
    const int total_items = extras + blocks_count;
    if (total_items <= 0) return;

    last_block_rects_.clear();

    const int thumb = 120;
    const int margin = 14;
    const auto area = panelContentRect(PanelId::Blocks);
    const int start_x = static_cast<int>(area.left);
    const int start_y = static_cast<int>(area.top);

    const int label_h = 18;
    const int max_cols = std::max(1, static_cast<int>(area.width) / (thumb + margin));
    const int cell_h = thumb + label_h + margin;
    const int total_rows = (total_items + max_cols - 1) / std::max(1, max_cols);
    const float content_h = static_cast<float>(std::max(0, total_rows) * std::max(1, cell_h));
    // PS (user): MAJ en temps réel des scrollers avec les valeurs runtime.
    // Ici: si le runtime ajoute/retire des items, on conserve la position relative
    // tant que l'utilisateur n'est pas en train de drag.
    const float prev_max = blocks_scroll_max_;
    const float prev_y = blocks_scroll_y_;

    blocks_scroll_max_ = std::max(0.0f, content_h - area.height);
    if (!dragging_blocks_scrollbar_ && prev_max > 1e-6f) {
        const float t = std::clamp(prev_y / prev_max, 0.0f, 1.0f);
        blocks_scroll_y_ = t * blocks_scroll_max_;
    }
    blocks_scroll_y_ = std::clamp(blocks_scroll_y_, 0.0f, std::max(0.0f, blocks_scroll_max_));

    // Focus rectangles doivent correspondre aux vrais blocks (index = layer_block_images idx).
    last_block_rects_.reserve(static_cast<size_t>(std::max(0, blocks_count)));

    // Clip au contenu du panneau (évite de dessiner en dehors de la zone visible pendant le scroll)
    const sf::View old_view = window->getView();
    {
        const float ww = static_cast<float>(std::max(1, window_width));
        const float hh = static_cast<float>(std::max(1, window_height));

        // Clamp à la partie visible dans la fenêtre (le panneau peut être resizé hors écran)
        const float left = std::clamp(area.left, 0.0f, ww);
        const float top = std::clamp(area.top, 0.0f, hh);
        const float right = std::clamp(area.left + area.width, 0.0f, ww);
        const float bottom = std::clamp(area.top + area.height, 0.0f, hh);
        const float w = std::max(0.0f, right - left);
        const float h = std::max(0.0f, bottom - top);

        if (w > 0.0f && h > 0.0f) {
            sf::View v(sf::FloatRect(left, top, w, h));
            v.setViewport(sf::FloatRect(left / ww, top / hh, w / ww, h / hh));
            window->setView(v);
        }
    }

    int extra_idx = 0;

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

    for (int slot = 0; slot < total_items; ++slot) {
        const int col = slot % max_cols;
        const int row = slot / max_cols;
        const int x = start_x + col * (thumb + margin);
        const int y = start_y + row * (thumb + label_h + margin) - static_cast<int>(std::lround((double)blocks_scroll_y_));

        // Cull (best-effort) pour limiter les draws
        if ((y - cell_h) > static_cast<int>(area.top + area.height)) continue;
        if ((y + 2 * cell_h) < static_cast<int>(area.top)) continue;

        ImageData* img_ptr = nullptr;
        std::string parsed_label;
        std::string text_override;
        std::string tag_override;

        const bool is_extra = (slot < extras);
        if (is_extra) {
            // Ordre: Projection puis Sortie
            if (show_projection && extra_idx == 0) {
                img_ptr = &projection_thumb_;
                parsed_label = projection_label.empty() ? std::string("mimir/output/projection") : projection_label;
                text_override = "Projection";
                tag_override = "OUT";
            } else {
                img_ptr = &output_thumb_;
                parsed_label = output_thumb_label_.empty() ? std::string("mimir/output") : output_thumb_label_;
                text_override = "Sortie";
                tag_override = "OUT";
            }
            extra_idx++;
        } else {
            const int bi = slot - extras;
            if (bi < 0 || bi >= blocks_count) break;
            img_ptr = &layer_block_images[static_cast<size_t>(bi)];
            parsed_label = (bi >= 0 && bi < static_cast<int>(layer_block_labels.size())) ? layer_block_labels[static_cast<size_t>(bi)] : std::string();
        }

        if (!img_ptr) continue;
        auto parsed = parse_viz_label(parsed_label);
        if (!is_extra) {
            apply_arch_hint(parsed);
        }
        const std::string use_tag = is_extra ? tag_override : parsed.tag;

        // Frame
        sf::RectangleShape frame(sf::Vector2f(thumb + 4, thumb + 4));
        frame.setPosition(static_cast<float>(x - 2), static_cast<float>(y - 2));
        frame.setFillColor(sf::Color::Transparent);
        frame.setOutlineColor(color_for_tag(use_tag));
        frame.setOutlineThickness(1);
        window->draw(frame);

        // Focus rectangles: uniquement pour les vrais blocks/layers (ne pas casser le focus/zoom existant).
        if (!is_extra) {
            // S'assurer que l'index correspond à bi (slot - extras)
            const int bi = slot - extras;
            if (bi >= 0 && bi < blocks_count) {
                // Remplir jusqu'à bi si besoin (garde l'alignement index->rect)
                while ((int)last_block_rects_.size() < bi) {
                    last_block_rects_.push_back(sf::FloatRect(0.f, 0.f, 0.f, 0.f));
                }
                if ((int)last_block_rects_.size() == bi) {
                    last_block_rects_.push_back(sf::FloatRect(static_cast<float>(x - 2), static_cast<float>(y - 2), static_cast<float>(thumb + 4), static_cast<float>(thumb + 4)));
                } else {
                    last_block_rects_[static_cast<size_t>(bi)] = sf::FloatRect(static_cast<float>(x - 2), static_cast<float>(y - 2), static_cast<float>(thumb + 4), static_cast<float>(thumb + 4));
                }
            }
        }

        // Image
        position_sprite_centered_in_box(img_ptr->sprite, static_cast<float>(x), static_cast<float>(y), static_cast<float>(thumb));
        window->draw(img_ptr->sprite);

        // Label
        sf::RectangleShape label(sf::Vector2f(static_cast<float>(thumb), static_cast<float>(label_h)));
        label.setPosition(static_cast<float>(x), static_cast<float>(y + thumb + 3));
        label.setFillColor(sf::Color(50, 50, 60, 200));
        window->draw(label);

        if (font_loaded) {
            sf::Text t;
            t.setFont(font);
            t.setCharacterSize(12);
            t.setFillColor(sf::Color(230, 230, 235));
            t.setPosition(static_cast<float>(x + 4), static_cast<float>(y + thumb + 1));

            std::string text;
            if (is_extra) {
                text = text_override;
            } else {
                if (parsed.tag != "ACT") {
                    text = "[" + parsed.tag + "] " + parsed.short_text;
                } else {
                    text = parsed.short_text;
                }
            }
            text = clamp_text_end(text, 18);
            t.setString(sf::String::fromUtf8(text.begin(), text.end()));
            window->draw(t);
        }
    }

    // Restore view
    window->setView(old_view);

    // Scrollbar (slicer) cliquable (dessiné dans l'UI normale, hors view clipping)
    // NB: rects en coordonnées UI (pixels)
    last_blocks_scroll_track_rect_ = sf::FloatRect(0.f, 0.f, 0.f, 0.f);
    last_blocks_scroll_thumb_rect_ = sf::FloatRect(0.f, 0.f, 0.f, 0.f);
    if (blocks_scroll_max_ > 0.0f) {
        const float track_w = 10.0f;
        const float pad = 2.0f;
        const float tx = area.left + std::max(0.0f, area.width - track_w - pad);
        const float ty = area.top;
        const float th = std::max(0.0f, area.height);
        last_blocks_scroll_track_rect_ = sf::FloatRect(tx, ty, track_w, th);

        const float content_h2 = area.height + blocks_scroll_max_;
        const float visible_ratio = (content_h2 > 1.0f) ? std::clamp(area.height / content_h2, 0.05f, 1.0f) : 1.0f;
        const float thumb_h = std::max(24.0f, th * visible_ratio);
        const float denom = std::max(1.0f, th - thumb_h);
        const float t = (blocks_scroll_max_ > 0.0f) ? std::clamp(blocks_scroll_y_ / blocks_scroll_max_, 0.0f, 1.0f) : 0.0f;
        const float thumb_y = ty + t * denom;
        last_blocks_scroll_thumb_rect_ = sf::FloatRect(tx, thumb_y, track_w, thumb_h);

        sf::RectangleShape track(sf::Vector2f(track_w, th));
        track.setPosition(tx, ty);
        track.setFillColor(sf::Color(35, 35, 42, 200));
        window->draw(track);

        sf::RectangleShape thumb(sf::Vector2f(track_w, thumb_h));
        thumb.setPosition(tx, thumb_y);
        thumb.setFillColor(sf::Color(90, 90, 110, 220));
        window->draw(thumb);
    }
}

void Visualizer::renderGeneratedImages() {
    if (generated_images.empty()) return;

    const auto area = panelContentRect(PanelId::Generated);

    const int img_display_size = 200; // Taille d'affichage
    const int margin = 20;
    const int start_x = static_cast<int>(area.left);
    const int start_y = static_cast<int>(area.top);

    const int cell_w = img_display_size + margin;
    const int cell_h = img_display_size + margin + 26;
    const int cols = std::max(1, static_cast<int>(area.width) / std::max(1, cell_w));
    const int rows = std::max(1, static_cast<int>(area.height) / std::max(1, cell_h));
    const int max_items = std::max(1, cols * rows);

    const size_t n = std::min(generated_images.size(), static_cast<size_t>(max_items));
    const size_t start_idx = (generated_images.size() > n) ? (generated_images.size() - n) : 0;

    last_generated_rects_.clear();
    last_generated_indices_.clear();
    last_generated_rects_.reserve(n);
    last_generated_indices_.reserve(n);

    // Clip au contenu du panneau (overflow: clip)
    const sf::View old_view = window->getView();
    {
        const float ww = static_cast<float>(std::max(1, window_width));
        const float hh = static_cast<float>(std::max(1, window_height));

        const float left = std::clamp(area.left, 0.0f, ww);
        const float top = std::clamp(area.top, 0.0f, hh);
        const float right = std::clamp(area.left + area.width, 0.0f, ww);
        const float bottom = std::clamp(area.top + area.height, 0.0f, hh);
        const float w = std::max(0.0f, right - left);
        const float h = std::max(0.0f, bottom - top);

        if (w > 0.0f && h > 0.0f) {
            sf::View v(sf::FloatRect(left, top, w, h));
            v.setViewport(sf::FloatRect(left / ww, top / hh, w / ww, h / hh));
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
        sf::RectangleShape frame(sf::Vector2f(img_display_size + 4, img_display_size + 4));
        frame.setPosition(x - 2, y - 2);
        frame.setFillColor(sf::Color::Transparent);
        frame.setOutlineColor(sf::Color(100, 150, 200));
        frame.setOutlineThickness(2);
        window->draw(frame);

        last_generated_rects_.push_back(sf::FloatRect(static_cast<float>(x - 2), static_cast<float>(y - 2), static_cast<float>(img_display_size + 4), static_cast<float>(img_display_size + 4)));
        last_generated_indices_.push_back(static_cast<int>(start_idx + i));

        // Image
        position_sprite_centered_in_box(img.sprite, static_cast<float>(x), static_cast<float>(y), static_cast<float>(img_display_size));
        window->draw(img.sprite);

        // Titre (simulé avec rectangle - nécessite police pour texte)
        sf::RectangleShape label(sf::Vector2f(img_display_size, 20));
        label.setPosition(x, y + img_display_size + 5);
        label.setFillColor(sf::Color(50, 50, 60, 200));
        window->draw(label);
    }

    window->setView(old_view);
}

void Visualizer::renderTrainingProgress() {
    const auto area = panelContentRect(PanelId::Training);
    int bar_x = static_cast<int>(area.left);
    int bar_y = static_cast<int>(area.top);
    int bar_width = std::max(10, static_cast<int>(area.width));
    int bar_height = std::min(30, std::max(10, static_cast<int>(area.height)));

    // Barre de progression epoch (simulée)
    sf::RectangleShape progress_bg(sf::Vector2f(bar_width, bar_height));
    progress_bg.setPosition(bar_x, bar_y);
    progress_bg.setFillColor(sf::Color(50, 50, 60));
    progress_bg.setOutlineColor(sf::Color(80, 80, 90));
    progress_bg.setOutlineThickness(2);
    window->draw(progress_bg);

    // Barre remplie (exemple: 50% d'avancement)
    float progress = std::min(1.0f, current_batch / 100.0f);
    sf::RectangleShape progress_bar(sf::Vector2f(bar_width * progress, bar_height));
    progress_bar.setPosition(bar_x, bar_y);
    progress_bar.setFillColor(sf::Color(100, 180, 100));
    window->draw(progress_bar);
}

void Visualizer::renderLossGraph() {
    if (!window) return;

    const bool have_full = !full_loss_history.empty();
    if (!have_full && loss_history.empty()) return;

    const auto area = panelContentRect(PanelId::Graph);
    const int graph_x = static_cast<int>(area.left);
    const int graph_y = static_cast<int>(area.top);
    const int graph_width = std::max(120, static_cast<int>(area.width));
    const int graph_height = std::max(120, static_cast<int>(area.height));

    // Fond du graphique
    sf::RectangleShape graph_bg(sf::Vector2f(static_cast<float>(graph_width), static_cast<float>(graph_height)));
    graph_bg.setPosition(static_cast<float>(graph_x), static_cast<float>(graph_y));
    graph_bg.setFillColor(sf::Color(20, 20, 25, 220));
    graph_bg.setOutlineColor(sf::Color(100, 100, 120));
    graph_bg.setOutlineThickness(2);
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

        sf::RectangleShape grid(sf::Vector2f(plot_w, 1.f));
        grid.setPosition(plot_x, y);
        grid.setFillColor(sf::Color(70, 70, 80, (i == 0 || i == ticks - 1) ? 120 : 70));
        window->draw(grid);

        if (font_loaded) {
            const float v = min_loss + t * range;
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << v;
            const std::string label = ss.str();
            sf::Text txt;
            txt.setFont(font);
            txt.setCharacterSize(12);
            txt.setFillColor(sf::Color(220, 220, 228));
            txt.setPosition(static_cast<float>(graph_x) + 6.f, y - 8.f);
            txt.setString(sf::String::fromUtf8(label.begin(), label.end()));
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

    sf::VertexArray line(sf::LineStrip, n_draw);
    for (size_t i = 0; i < n_draw; ++i) {
        const size_t idx = (n_draw <= 1) ? 0 : (i * (n_total - 1)) / (n_draw - 1);
        const float loss = loss_at(idx);
        const float x = plot_x + (static_cast<float>(i) / static_cast<float>(n_draw - 1)) * plot_w;
        const float normalized = (loss - min_loss) / range;
        const float y = plot_y + (1.f - std::clamp(normalized, 0.f, 1.f)) * plot_h;

        line[i].position = sf::Vector2f(x, y);
        line[i].color = getLossColor(loss);
    }
    window->draw(line);

    // Labels X (début/fin) + info
    if (font_loaded) {
        const size_t start_step = 0;
        const size_t end_step = (n_total > 0) ? (n_total - 1) : 0;
        auto drawSmall = [&](float x, float y, const std::string& s) {
            sf::Text t;
            t.setFont(font);
            t.setCharacterSize(12);
            t.setFillColor(sf::Color(220, 220, 228));
            t.setPosition(x, y);
            t.setString(sf::String::fromUtf8(s.begin(), s.end()));
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
    const int panel_w = std::max(120, static_cast<int>(area.width));
    int metrics_x = static_cast<int>(area.left);
    int metrics_y = static_cast<int>(area.top);
    int line_height = 25;

    // Bouton stop: hitbox recalculée à chaque frame.
    last_stop_button_rect_.reset();

    // Live tuning controls: hitboxes recalculées à chaque frame.
    last_live_lr_track_.reset();
    last_live_lr_thumb_.reset();
    last_live_lrwu_track_.reset();
    last_live_lrwu_thumb_.reset();
    last_live_klb_track_.reset();
    last_live_klb_thumb_.reset();
    last_live_klwu_track_.reset();
    last_live_klwu_thumb_.reset();
    last_live_kl_enable_box_.reset();
    last_live_overrides_box_.reset();

    auto drawMetricBox = [&](int y_offset, sf::Color color) {
        sf::RectangleShape box(sf::Vector2f(static_cast<float>(panel_w), 20));
        box.setPosition(metrics_x, metrics_y + y_offset);
        box.setFillColor(color);
        window->draw(box);
    };

    // Epoch (bleu)
    drawMetricBox(0, sf::Color(50, 100, 180, 200));

    // Batch (vert)
    drawMetricBox(line_height, sf::Color(50, 150, 80, 200));

    // Loss (orange/rouge selon valeur)
    sf::Color loss_color = current_loss > 100.0f ? sf::Color(200, 80, 50, 200) : sf::Color(180, 140, 50, 200);
    drawMetricBox(line_height * 2, loss_color);

    // Learning rate (violet)
    drawMetricBox(line_height * 3, sf::Color(140, 80, 180, 200));

    // Bouton STOP (rouge) — dans le panneau Metrics (layer metrics)
    {
        const int pad = 6;
        const int btn_h = 20;
        const int btn_w = std::min(160, std::max(80, panel_w - 2 * pad));
        const int btn_x = metrics_x + panel_w - btn_w;
        const int btn_y = metrics_y;

        const bool stop_requested = stop_training_requested_.load(std::memory_order_relaxed);

        sf::RectangleShape b(sf::Vector2f((float)btn_w, (float)btn_h));
        b.setPosition((float)btn_x, (float)btn_y);
        b.setFillColor(stop_requested ? sf::Color(180, 55, 55, 235) : sf::Color(140, 40, 40, 230));
        b.setOutlineColor(stop_requested ? sf::Color(245, 150, 150, 245) : sf::Color(210, 90, 90, 240));
        b.setOutlineThickness(1);
        window->draw(b);

        last_stop_button_rect_ = sf::FloatRect((float)btn_x, (float)btn_y, (float)btn_w, (float)btn_h);

        if (font_loaded) {
            sf::Text t;
            t.setFont(font);
            t.setCharacterSize(13);
            t.setFillColor(sf::Color(245, 240, 240));
            t.setPosition((float)btn_x + 8.f, (float)btn_y + 1.f);
            t.setString(stop_requested ? "STOP demandé" : "STOP training");
            window->draw(t);
        }
    }

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
            sf::Text t;
            t.setFont(font);
            t.setCharacterSize(14);
            t.setFillColor(sf::Color(230, 230, 235));
            t.setPosition(static_cast<float>(metrics_x + 6), static_cast<float>(metrics_y + y_offset - 2));
            t.setString(sf::String::fromUtf8(s.begin(), s.end()));
            window->draw(t);
        };

        drawText(0, "Epoch " + std::to_string(current_epoch) + "/" + std::to_string(std::max(1, current_total_epochs)));
        drawText(line_height, "Batch " + std::to_string(current_batch) + "/" + std::to_string(std::max(1, current_total_batches)));
        { drawText(line_height * 2, "Loss=" + format_decimal(current_loss, 8) + " avg=" + format_decimal(current_avg_loss, 8)); }
        { drawText(line_height * 3, "LR=" + format_decimal(current_lr, 10)); }

        // -------------------------
        // Live tuning controls
        // -------------------------
        if (!live_ui_inited_) {
            live_ui_lr_ = std::clamp(current_lr, kLiveLRMin, kLiveLRMax);
            live_ui_lr_warmup_steps_ = 0;
            live_ui_kl_beta_ = std::clamp(current_kl_beta_effective, 0.0f, 1.0f);
            live_ui_kl_warmup_steps_ = 0;
            live_ui_kl_enabled_ = (live_ui_kl_beta_ > 0.0f);

            // Démarre en mode NATIVE: aucun override tant que l'utilisateur n'interagit pas.
            live_overrides_enabled_.store(false, std::memory_order_relaxed);
            live_lr_.store(live_ui_lr_, std::memory_order_relaxed);
            live_lr_warmup_steps_.store(live_ui_lr_warmup_steps_, std::memory_order_relaxed);
            live_kl_beta_.store(live_ui_kl_beta_, std::memory_order_relaxed);
            live_kl_warmup_steps_.store(live_ui_kl_warmup_steps_, std::memory_order_relaxed);
            live_kl_enabled_.store(live_ui_kl_enabled_, std::memory_order_relaxed);
            // Ne pas bump la version: on ne veut pas changer le training sans interaction.
            live_ui_inited_ = true;
        }

        // PS (user): MAJ en temps réel des scrollers (sliders) avec les valeurs runtime associées.
        // En mode NATIVE, refléter current_lr/current_kl_beta_effective tant qu'on ne drag pas.
        const bool live_on_rt = live_overrides_enabled_.load(std::memory_order_relaxed);
        if (!live_on_rt && live_dragging_ == LiveDragTarget::None) {
            live_ui_lr_ = std::clamp(current_lr, kLiveLRMin, kLiveLRMax);
            live_ui_lr_warmup_steps_ = 0;
            live_ui_kl_beta_ = std::clamp(current_kl_beta_effective, 0.0f, 1.0f);
            live_ui_kl_warmup_steps_ = 0;
            live_ui_kl_enabled_ = (live_ui_kl_beta_ > 0.0f);
        }

        const int live_y0 = line_height * 4 + 6;
        const int live_row_h = 22;
        const int label_w = 82;
        const int value_w = 110;
        const float track_h = 10.f;
        const float thumb_w = 10.f;
        const float thumb_h = 16.f;
        const int rows = 5;

        auto drawSliderRow = [&](int row,
                                 const std::string& label,
                                 float t,
                                 const std::string& value,
                                 std::optional<sf::FloatRect>& out_track,
                                 std::optional<sf::FloatRect>& out_thumb) {
            const float y = static_cast<float>(metrics_y + live_y0 + row * live_row_h);
            const float x0 = static_cast<float>(metrics_x + 6);
            const float x_track = static_cast<float>(metrics_x + label_w);
            const float x_value = static_cast<float>(metrics_x + panel_w - value_w);
            const float w_track = std::max(20.f, x_value - x_track - 10.f);

            // Label
            {
                sf::Text tt;
                tt.setFont(font);
                tt.setCharacterSize(13);
                tt.setFillColor(sf::Color(225, 225, 235));
                tt.setPosition(x0, y - 2.f);
                tt.setString(sf::String::fromUtf8(label.begin(), label.end()));
                window->draw(tt);
            }

            // Track
            const float tt = clamp01(t);
            const float track_y = y + 7.f;
            sf::RectangleShape track(sf::Vector2f(w_track, track_h));
            track.setPosition(x_track, track_y);
            track.setFillColor(sf::Color(55, 55, 60, 210));
            track.setOutlineColor(sf::Color(90, 90, 95, 220));
            track.setOutlineThickness(1);
            window->draw(track);

            sf::RectangleShape fill(sf::Vector2f(w_track * tt, track_h));
            fill.setPosition(x_track, track_y);
            fill.setFillColor(sf::Color(90, 140, 200, 220));
            window->draw(fill);

            const float thumb_x = x_track + w_track * tt - thumb_w * 0.5f;
            sf::RectangleShape thumb(sf::Vector2f(thumb_w, thumb_h));
            thumb.setPosition(thumb_x, track_y + track_h * 0.5f - thumb_h * 0.5f);
            thumb.setFillColor(sf::Color(210, 210, 220, 235));
            thumb.setOutlineColor(sf::Color(30, 30, 30, 220));
            thumb.setOutlineThickness(1);
            window->draw(thumb);

            out_track = sf::FloatRect(x_track, track_y, w_track, track_h);
            out_thumb = sf::FloatRect(thumb.getPosition().x, thumb.getPosition().y, thumb_w, thumb_h);

            // Value string
            {
                sf::Text tt;
                tt.setFont(font);
                tt.setCharacterSize(13);
                tt.setFillColor(sf::Color(225, 225, 235));
                tt.setPosition(x_value, y - 2.f);
                tt.setString(sf::String::fromUtf8(value.begin(), value.end()));
                window->draw(tt);
            }
        };

        // Indicateur LIVE/NATIVE (cliquable)
        {
            const bool live_on = live_overrides_enabled_.load(std::memory_order_relaxed);
            const float y = static_cast<float>(metrics_y + live_y0 - 22);
            const float box_w = 130.f;
            const float box_h = 18.f;
            const float box_x = static_cast<float>(metrics_x + panel_w) - box_w - 6.f;
            const float box_y = y;

            sf::RectangleShape b(sf::Vector2f(box_w, box_h));
            b.setPosition(box_x, box_y);
            b.setFillColor(live_on ? sf::Color(50, 110, 70, 220) : sf::Color(60, 60, 65, 210));
            b.setOutlineColor(live_on ? sf::Color(130, 220, 160, 235) : sf::Color(100, 100, 105, 220));
            b.setOutlineThickness(1);
            window->draw(b);

            last_live_overrides_box_ = sf::FloatRect(box_x, box_y, box_w, box_h);

            sf::Text tt;
            tt.setFont(font);
            tt.setCharacterSize(12);
            tt.setFillColor(sf::Color(245, 245, 248));
            tt.setPosition(box_x + 6.f, box_y + 1.f);
            tt.setString(live_on ? "LIVE (click -> NATIVE)" : "NATIVE (click -> LIVE)");
            window->draw(tt);

            // Petit suffixe version (debug visuel)
            const uint64_t ver = live_params_version_.load(std::memory_order_relaxed);
            sf::Text vv;
            vv.setFont(font);
            vv.setCharacterSize(12);
            vv.setFillColor(sf::Color(210, 210, 215));
            vv.setPosition(box_x + box_w - 48.f, box_y + 1.f);
            vv.setString("v=" + std::to_string(ver));
            window->draw(vv);
        }

        // LR (log scale)
        { drawSliderRow(0, "LR", t_from_lr(live_ui_lr_), format_decimal(live_ui_lr_, 10), last_live_lr_track_, last_live_lr_thumb_); }

        // LR warmup
        {
            drawSliderRow(1,
                          "LR wu",
                          t_from_warmup(live_ui_lr_warmup_steps_),
                          std::to_string(live_ui_lr_warmup_steps_) + " steps",
                          last_live_lrwu_track_,
                          last_live_lrwu_thumb_);
        }

        // KL beta
        { drawSliderRow(2, "KL beta", t_from_klbeta(live_ui_kl_beta_), format_decimal(live_ui_kl_beta_, 8), last_live_klb_track_, last_live_klb_thumb_); }

        // KL warmup
        {
            drawSliderRow(3,
                          "KL wu",
                          t_from_warmup(live_ui_kl_warmup_steps_),
                          std::to_string(live_ui_kl_warmup_steps_) + " steps",
                          last_live_klwu_track_,
                          last_live_klwu_thumb_);
        }

        // KL enabled checkbox
        {
            const float y = static_cast<float>(metrics_y + live_y0 + 4 * live_row_h);
            const float x0 = static_cast<float>(metrics_x + 6);
            const float box = 14.f;
            const float box_x = static_cast<float>(metrics_x + label_w);
            const float box_y = y + 4.f;

            sf::Text tt;
            tt.setFont(font);
            tt.setCharacterSize(13);
            tt.setFillColor(sf::Color(225, 225, 235));
            tt.setPosition(x0, y - 2.f);
            tt.setString("KL");
            window->draw(tt);

            sf::RectangleShape cb(sf::Vector2f(box, box));
            cb.setPosition(box_x, box_y);
            cb.setFillColor(sf::Color(55, 55, 60, 210));
            cb.setOutlineColor(sf::Color(90, 90, 95, 220));
            cb.setOutlineThickness(1);
            window->draw(cb);

            if (live_ui_kl_enabled_) {
                sf::RectangleShape mark(sf::Vector2f(box - 4.f, box - 4.f));
                mark.setPosition(box_x + 2.f, box_y + 2.f);
                mark.setFillColor(sf::Color(90, 200, 120, 220));
                window->draw(mark);
            }

            last_live_kl_enable_box_ = sf::FloatRect(box_x, box_y, box, box);

            sf::Text st;
            st.setFont(font);
            st.setCharacterSize(13);
            st.setFillColor(sf::Color(225, 225, 235));
            st.setPosition(box_x + box + 8.f, y - 2.f);
            st.setString(live_ui_kl_enabled_ ? "enabled" : "disabled");
            window->draw(st);
        }

        int extra_y = live_y0 + rows * live_row_h + 10;
        {
            std::string s = "t=" + format_decimal(current_timestep, 8) + " mse=" + format_decimal(current_mse, 8);
            if (!current_recon_loss_type.empty()) {
                s = "recon=" + current_recon_loss_type + "  " + s;
            }
            drawText(extra_y, s);
        }

        if (current_kl_beta_effective > 0.0f) {
            drawText(extra_y + line_height, "KL beta_eff=" + format_decimal(current_kl_beta_effective, 8) + "  kl=" + format_decimal(current_kl, 8));
            extra_y += line_height;
        }
        drawText(extra_y + line_height, "grad=" + format_decimal(current_grad_norm, 8) + " max=" + format_decimal(current_grad_max, 8));
        drawText(extra_y + 2 * line_height, "time=" + std::to_string(current_batch_time_ms) + "ms bps=" + format_decimal(current_bps, 8));
        drawText(extra_y + 3 * line_height, "mem=" + std::to_string(current_memory_mb) + "MB params=" + std::to_string(current_params));
        drawText(extra_y + 4 * line_height, "opt step=" + std::to_string(current_opt_step) + " wd=" + format_decimal(current_opt_weight_decay, 10));

        {
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
            drawText(extra_y + 5 * line_height, s);

            // Afficher les métriques de validation dès qu'on en a (même partiellement).
            if (current_val_has || current_val_in_progress) {
                std::ostringstream ss;
                ss << "val recon=" << format_decimal(current_val_recon, 8) << " kl=" << format_decimal(current_val_kl, 8);
                if (std::fabs(current_val_align) > 1e-9f) {
                    ss << " align=" << format_decimal(current_val_align, 8);
                }
                drawText(extra_y + 6 * line_height, ss.str());
            }
        }

        // Texte dataset (si dispo): prompt brut + tokens + résumé encodage.
        if (show_prompt_text_ && has_dataset_text) {
            int y = extra_y + 8 * line_height;
            const int y_max = static_cast<int>(area.top + area.height) - line_height;
            const size_t wrap_chars = static_cast<size_t>(std::max(30, panel_w / 8));

            auto drawWrapped = [&](const std::string& s, int max_lines_hint) {
                if (s.empty()) return;
                const int avail_lines = std::max(0, (y_max - (metrics_y + y)) / line_height);
                const int max_lines = (max_lines_hint > 0) ? std::min(max_lines_hint, avail_lines) : avail_lines;
                const auto lines = wrap_lines(s, wrap_chars, max_lines);
                for (const auto& ln : lines) {
                    if (metrics_y + y > y_max) break;
                    drawText(y, ln);
                    y += line_height;
                }
            };

            // Afficher autant que possible sans tronquer la source.
            // (La zone visible est bornée par la hauteur de la fenêtre.)
            drawWrapped("prompt: " + dataset_text_raw, 0);
            drawWrapped("tokens: " + dataset_text_tokens, 4);
            drawWrapped("enc: " + dataset_text_encoded, 3);
        }
    }
}

void Visualizer::createImageTexture(ImageData& img_data, int w, int h, int channels, int display_size) {
    // Créer une image SFML à partir des pixels (grayscale/RGB/RGBA)
    if (w <= 0 || h <= 0) return;
    if (channels != 1 && channels != 3 && channels != 4) return;

    img_data.w = w;
    img_data.h = h;
    img_data.channels = channels;
    img_data.display_size = display_size;

    sf::Image sfml_image;
    sfml_image.create(static_cast<unsigned>(w), static_cast<unsigned>(h));

    const size_t stride = static_cast<size_t>(channels);
    const size_t expected = static_cast<size_t>(w) * static_cast<size_t>(h) * stride;
    const size_t n = std::min(expected, img_data.pixels.size());

    auto at = [&](size_t idx) -> uint8_t {
        return (idx < n) ? img_data.pixels[idx] : 0;
    };

    for (int yy = 0; yy < h; ++yy) {
        for (int xx = 0; xx < w; ++xx) {
            const size_t base = (static_cast<size_t>(yy) * static_cast<size_t>(w) + static_cast<size_t>(xx)) * stride;
            sf::Color c;
            if (channels == 1) {
                const uint8_t g = at(base);
                c = sf::Color(g, g, g, 255);
            } else if (channels == 3) {
                c = sf::Color(at(base + 0), at(base + 1), at(base + 2), 255);
            } else {
                c = sf::Color(at(base + 0), at(base + 1), at(base + 2), at(base + 3));
            }
            sfml_image.setPixel(static_cast<unsigned>(xx), static_cast<unsigned>(yy), c);
        }
    }

    img_data.texture.loadFromImage(sfml_image);
    img_data.sprite.setTexture(img_data.texture);

    // Mettre à l'échelle pour affichage (fit-to-square)
    const float sx = static_cast<float>(display_size) / static_cast<float>(w);
    const float sy = static_cast<float>(display_size) / static_cast<float>(h);
    const float scale = std::min(sx, sy);
    img_data.sprite.setScale(scale, scale);
}

sf::Color Visualizer::getLossColor(float loss) {
    // Gradient vert -> jaune -> rouge basé sur la loss
    if (loss < 50.0f) {
        return sf::Color(100, 200, 100); // Vert
    } else if (loss < 150.0f) {
        return sf::Color(200, 200, 100); // Jaune
    } else {
        return sf::Color(200, 100, 100); // Rouge
    }
}

void Visualizer::saveLossHistory(const std::string& filepath) const {
    
    std::ofstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Erreur: impossible d'ouvrir " << filepath << " pour écriture" << std::endl;
        return;
    }
    
    // En-tête CSV (métriques complètes)
    file << "step,epoch,total_epochs,batch,total_batches,loss,avg_loss,learning_rate,batch_time_ms,bps,memory_mb,params,mse,kl_divergence,wasserstein,entropy_diff,moment_mismatch,spatial_coherence,temporal_consistency,timestep,grad_norm,grad_max,opt_type,opt_step,opt_beta1,opt_beta2,opt_eps,opt_weight_decay" << std::endl;
    
    // Écrire tout l'historique complet (toutes les epochs et tous les steps)
    for (const auto& record : full_loss_history) {
        file << record.step << "," 
             << record.epoch << "," 
             << record.total_epochs << ","
             << record.batch << "," 
             << record.total_batches << ","
             << std::fixed << std::setprecision(6) << record.loss << ","
             << record.avg_loss << ","
             << std::scientific << record.lr << ","
             << std::fixed << record.batch_time_ms << ","
             << record.bps << ","
             << record.memory_mb << ","
             << record.params << ","
             << std::fixed << std::setprecision(6) << record.mse << ","
             << record.kl_divergence << ","
             << record.wasserstein << ","
             << record.entropy_diff << ","
             << record.moment_mismatch << ","
             << record.spatial_coherence << ","
             << record.temporal_consistency << ","
             << record.timestep << ","
             << record.grad_norm << ","
             << record.grad_max << ","
             << record.opt_type << ","
             << record.opt_step << ","
             << record.opt_beta1 << ","
             << record.opt_beta2 << ","
             // opt_eps est souvent ~1e-8 : en fixed(6) ça apparaît comme 0.000000.
             // On l'encode en scientifique pour préserver l'information.
             << std::scientific << std::setprecision(8) << record.opt_eps << ","
             << std::fixed << std::setprecision(6) << record.opt_weight_decay << std::endl;
    }
    
    file.close();
}
