#pragma once
#include "SkipConnectionControl.hpp"
#include "LiveModelConfig.hpp"
#include "viz/TerminalModelGraph.hpp"
#include "DebugJsonExportDialog.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <streambuf>
#include <memory>
#include <cerrno>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <limits>
#include <atomic>
#if defined(_WIN32)
#include <io.h>
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
struct winsize {
    unsigned short ws_row;
    unsigned short ws_col;
    unsigned short ws_xpixel;
    unsigned short ws_ypixel;
};
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

void framework_log_write(const char* data, size_t size);
void framework_log_write_file_only(const char* data, size_t size);

// Petit streambuf qui écrit directement dans un file descriptor (ex: /dev/tty).
// Objectif: permettre à l'UI d'écrire sur le terminal même si stdout/stderr sont redirigés.
class FdStreamBuf final : public std::streambuf
{
public:
    explicit FdStreamBuf(int fd) : fd_(fd)
    {
        setp(buffer_, buffer_ + sizeof(buffer_) - 1);
    }

    FdStreamBuf(const FdStreamBuf&) = delete;
    FdStreamBuf& operator=(const FdStreamBuf&) = delete;

protected:
    int_type overflow(int_type ch) override
    {
        if (ch != traits_type::eof()) {
            *pptr() = static_cast<char>(ch);
            pbump(1);
        }
        return flushBuffer() == 0 ? ch : traits_type::eof();
    }

    int sync() override
    {
        return flushBuffer() == 0 ? 0 : -1;
    }

private:
    int flushBuffer()
    {
        const std::ptrdiff_t n = pptr() - pbase();
        if (n <= 0) {
            return 0;
        }

        framework_log_write(pbase(), static_cast<size_t>(n));

        const char* data = pbase();
        std::ptrdiff_t remaining = n;
        while (fd_ >= 0 && remaining > 0) {
#if defined(_WIN32)
            const int written = _write(fd_, data, static_cast<unsigned int>(remaining));
#else
            const ssize_t written = ::write(fd_, data, static_cast<size_t>(remaining));
#endif
            if (written < 0) {
                if (errno == EINTR) continue;
                break;
            }
            data += written;
            remaining -= written;
        }

        pbump(-static_cast<int>(n));
        return remaining == 0 ? 0 : -1;
    }

    int fd_;
    char buffer_[8192];
};

// ============================================================================
// Interface Htop-like pour monitoring de l'entraînement
// ============================================================================
class HtopDisplay
{
private:
    struct winsize terminal_size;
    int width, height;
    bool display_enabled;
    std::chrono::steady_clock::time_point start_time;

    // Sortie UI (idéalement /dev/tty ou un dup de l'ancien stdout).
    int out_fd_ = STDOUT_FILENO;
    std::unique_ptr<FdStreamBuf> out_buf_;
    std::unique_ptr<std::ostream> out_;

    bool alt_screen_enabled_ = false;
    std::atomic<int> viz_action_state_{0};
    std::atomic<bool> help_visible_{false};
    std::atomic<bool> validation_enabled_{false};
    std::atomic<bool> live_overrides_enabled_{false};
    std::atomic<float> live_lr_{0.0f};
    std::atomic<int> live_lr_warmup_steps_{0};
    std::atomic<float> live_kl_beta_{0.0f};
    std::atomic<int> live_kl_warmup_steps_{0};
    std::atomic<bool> live_kl_enabled_{false};
    std::atomic<int> live_selected_parameter_{0};
    std::atomic<int> live_recon_loss_index_{0};
    std::atomic<bool> safe_stop_pending_{false};

    // Buffer de logs affiché dans l'UI (collecte via redirection stdout/stderr).
    mutable std::mutex log_mutex_;
    std::deque<std::string> log_lines_;
    std::string log_partial_;
    size_t log_max_lines_ = 300;

    std::ostream& out()
    {
        if (out_) return *out_;
        return std::cerr;
    }

    static std::string rtrimNewlines(std::string s)
    {
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        return s;
    }

    static std::vector<std::string> splitLinesKeepNoNewline(const std::string& s)
    {
        std::vector<std::string> out;
        std::string cur;
        cur.reserve(s.size());
        for (char c : s) {
            if (c == '\r') {
                // ignorer (beaucoup de progress bars font \r)
                continue;
            }
            if (c == '\n') {
                out.push_back(cur);
                cur.clear();
                continue;
            }
            cur.push_back(c);
        }
        out.push_back(cur);
        return out;
    }

    static std::string clipToWidth(const std::string& s, int maxw)
    {
        if (maxw <= 0) return std::string();
        if ((int)s.size() <= maxw) return s;
        if (maxw <= 1) return s.substr(0, (size_t)maxw);
        return s.substr(0, (size_t)maxw - 1) + "…";
    }

    // Wrap simple (sans tenir compte des codes ANSI) pour éviter les débordements.
    static std::vector<std::string> wrapToWidth(const std::string& s, int maxw)
    {
        std::vector<std::string> out;
        if (maxw <= 0) return out;
        if ((int)s.size() <= maxw) {
            out.push_back(s);
            return out;
        }
        size_t i = 0;
        while (i < s.size()) {
            const size_t n = std::min<size_t>((size_t)maxw, s.size() - i);
            out.push_back(s.substr(i, n));
            i += n;
        }
        return out;
    }

    // CSV export (mêmes colonnes que Visualizer::saveLossHistory)
    struct CsvRecord {
        int step;
        int epoch;
        int total_epochs;
        int batch;
        int total_batches;
        float loss;
        float avg_loss;
        float lr;
        int batch_time_ms;
        float bps;
        size_t memory_mb;
        double allocator_memory_mb;
        size_t params;
        float mse;
        float kl_divergence;
        float kl_beta_effective;
        float wasserstein;
        float entropy_diff;
        float moment_mismatch;
        float spatial_coherence;
        float temporal_consistency;
        float timestep;
        float grad_norm;
        float grad_max;
        int opt_type;
        int opt_step;
        float opt_beta1;
        float opt_beta2;
        float opt_eps;
        float opt_weight_decay;
        // Métriques de validation (renseignées uniquement pour is_val=true)
        float val_loss = 0.f;
        float val_mse  = 0.f;
        int   val_step = -1;
        bool  is_val   = false;
    };

    bool csv_enabled = false; // Automatic persistence belongs to AsyncMonitor.
    std::string csv_log_file = "checkpoints/htop_metrics.csv";
    std::vector<CsvRecord> csv_history;

    void saveLossHistoryCsv(const std::string& filepath) const
    {
        try {
            std::filesystem::path p(filepath);
            if (p.has_parent_path()) {
                std::error_code ec;
                std::filesystem::create_directories(p.parent_path(), ec);
            }
        } catch (...) {
            // Best-effort: si on ne peut pas créer le dossier, on tente quand même d'écrire.
        }

        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Erreur: impossible d'ouvrir " << filepath << " pour écriture" << std::endl;
            return;
        }

        file << "step,epoch,total_epochs,batch,total_batches,loss,avg_loss,learning_rate,batch_time_ms,bps,memory_mb,allocator_memory_mb,params,mse,kl_divergence,kl_beta_effective,wasserstein,entropy_diff,moment_mismatch,spatial_coherence,temporal_consistency,timestep,grad_norm,grad_max,opt_type,opt_step,opt_beta1,opt_beta2,opt_eps,opt_weight_decay,val_loss,val_mse,val_step" << std::endl;
        file << std::defaultfloat << std::setprecision(std::numeric_limits<float>::max_digits10);

        for (const auto& record : csv_history) {
            file << record.step << ","
                 << record.epoch << ","
                 << record.total_epochs << ","
                 << record.batch << ","
                 << record.total_batches << ","
                 << record.loss << ","
                 << record.avg_loss << ","
                 << record.lr << ","
                 << record.batch_time_ms << ","
                 << record.bps << ","
                 << record.memory_mb << ","
                 << record.allocator_memory_mb << ","
                 << record.params << ","
                 << record.mse << ","
                 << record.kl_divergence << ","
                 << record.kl_beta_effective << ","
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
                 << record.opt_eps << ","
                 << record.opt_weight_decay << ",";
            if (record.is_val) file << record.val_loss;
            file << ",";
            if (record.is_val) file << record.val_mse;
            file
                 << "," << (record.is_val ? std::to_string(record.val_step)  : "") << std::endl;
        }
    }

    // Statistiques
    struct Stats
    {
        int current_epoch;
        int total_epochs;
        int current_batch;
        int total_batches;
        float current_loss;
        float avg_loss;
        float kl_divergence;
        float wasserstein;
        float entropy_diff;
        float moment_mismatch;
        float spatial_coherence;
        float temporal_consistency;
        float mse_loss;
        float grad_norm;
        float grad_max;
        float learning_rate;
        int opt_type;
        int opt_step;
        float opt_beta1;
        float opt_beta2;
        float opt_eps;
        float opt_weight_decay;
        int batch_time_ms;
        size_t memory_used_mb;
        double allocator_memory_mb;
        size_t memory_freed_mb;
        float batches_per_sec;
        int eta_seconds;
        size_t total_params;
        float timestep;

        // Type de loss recon utilisée par le modèle (ex: "MSE", "L1").
        std::string recon_loss_type;
    } stats;

public:
    enum class VizActionState {
        Ready = 0,
        Opening = 1,
        Active = 2,
        Unavailable = 3
    };

    explicit HtopDisplay(int out_fd = STDOUT_FILENO) : display_enabled(true), out_fd_(out_fd)
    {
        out_buf_ = std::make_unique<FdStreamBuf>(out_fd_);
        out_ = std::make_unique<std::ostream>(out_buf_.get());
        getTerminalSize();
        start_time = std::chrono::steady_clock::now();
        stats = {};
    }

    ~HtopDisplay()
    {
        leaveAltScreen();
        showCursor();
    }

    void getTerminalSize()
    {
#if defined(_WIN32)
        width = 120;
        height = 40;
#else
        ioctl(out_fd_, TIOCGWINSZ, &terminal_size);
        width = terminal_size.ws_col;
        height = terminal_size.ws_row;
#endif
    }

    void enterAltScreen()
    {
        if (alt_screen_enabled_) return;
        // 1049h: alternate screen + save cursor state.
        out() << "\033[?1049h\033[H" << std::flush;
        alt_screen_enabled_ = true;
    }

    void leaveAltScreen()
    {
        if (!alt_screen_enabled_) return;
        out() << "\033[?1049l" << std::flush;
        alt_screen_enabled_ = false;
    }

    void clearScreen()
    {
        out() << "\033[2J\033[H" << std::flush;
    }

    void resetCursor()
    {
        // Retourner au début sans effacer (plus rapide)
        out() << "\033[H" << std::flush;
    }

    void hideCursor()
    {
        out() << "\033[?25l" << std::flush;
    }

    void showCursor()
    {
        out() << "\033[?25h" << std::flush;
    }

    void setVizActionState(VizActionState state)
    {
        viz_action_state_.store(static_cast<int>(state), std::memory_order_relaxed);
    }

    void setRuntimeConfiguration(std::shared_ptr<LiveModelConfig> config) {
        std::atomic_store(&runtime_config_, std::move(config));
    }
    std::shared_ptr<LiveModelConfig> runtime_config_ = std::make_shared<LiveModelConfig>();
    RuntimeConfigEditor config_editor_;
    DebugJsonExportDialog debug_json_dialog_;
    Mimir::Viz::TerminalModelGraphView model_graph_view_;
    bool modelGraphVisible() const { return model_graph_view_.visible; }
    void modelGraphKey(char key) {
        auto config=std::atomic_load(&runtime_config_);
        auto& view=model_graph_view_;
        if(!view.visible) { view.visible=true;view.follow=true; }
        else if(key=='g' || key=='G' || key=='\033')view.visible=false;
        else if(key=='r' || key=='R') { view.selected=0;view.pan_x=view.pan_y=0;view.follow=true; }
        else if(key=='\t')modelGraphMove('B');
        config->requestModelGraph(view.visible);
    }
    void modelGraphMove(char arrow) {
        auto graph=std::atomic_load(&runtime_config_)->modelGraph();
        const size_t count=graph?graph->at("nodes").size():0;
        auto& view=model_graph_view_;
        if(arrow=='A')view.move(-1,count);
        else if(arrow=='B')view.move(1,count);
        else if(arrow=='5' || arrow=='6')view.move((arrow=='5'?-1:1)*std::max(1,(height-13)/4),count);
        else { view.pan_x+=arrow=='D'?-8:8;view.follow=false; }
    }
    void renderModelGraph() {
        const auto config=std::atomic_load(&runtime_config_);
        config->requestModelGraph(true);
        if(width<32 || height<14) {
            resetCursor();clearLine();
            out()<<clipToWidth("Graphe: agrandir le terminal (G fermer)",std::max(1,width))<<std::flush;
            return;
        }
        const auto snapshot=config->modelGraph();
        const auto lines=model_graph_view_.lines(snapshot.get(),width,height);
        for(int row=1;row<=height;++row) {
            moveCursor(row,1);clearLine();
            if(row<=static_cast<int>(lines.size()))
                out()<<"| "<<clipToWidth(lines[row-1],std::max(1,width-4))<<" |";
        }
        out()<<std::flush;
    }
    bool debugJsonVisible() const { return debug_json_dialog_.visible; }
    void debugJsonKey(char key) {
        auto& form = debug_json_dialog_;
        if (!form.visible) { form.open(); return; }
        if (key == '\033') form.visible = false;
        else if (key == '\t') form.move();
        else if (key == '\r' || key == '\n') form.submit(*std::atomic_load(&runtime_config_));
        else if (static_cast<unsigned char>(key) >= 128) {
            auto& value = form.selected == 0 ? form.directory : form.filename;
            if (form.replace) value.clear();
            form.replace = false;
            if (value.size() < 4096) value += key;
        } else form.text(static_cast<unsigned char>(key));
    }
    void renderDebugJsonDialog() {
        const int columns = std::max(1, width - 4);
        auto line = [&](int row, const std::string& value) {
            moveCursor(row, 1); clearLine();
            const auto text = clipToWidth(value, columns);
            out() << "| " << text << std::string(std::max(0, columns-static_cast<int>(text.size())), ' ') << " |";
        };
        auto& form = debug_json_dialog_;
        line(1, std::string(columns, '='));
        line(2, "EXPORT DEBUGJSON");
        line(3, "Tab: champ suivant  Entree: exporter  Esc: annuler");
        line(4, "L'entrainement continue pendant la saisie.");
        line(5, std::string(form.selected == 0 ? "> Dossier : " : "  Dossier : ") + form.directory + (form.selected == 0 ? "_" : ""));
        line(6, std::string(form.selected == 1 ? "> Fichier : " : "  Fichier : ") + form.filename + (form.selected == 1 ? "_" : ""));
        line(7, form.message);
        line(8, "La saisie remplace le champ selectionne. Choisir un nouveau fichier.");
        line(9, std::string(columns, '='));
        for (int row=10; row<=height; ++row) { moveCursor(row,1); clearLine(); }
        out() << std::flush;
    }
    void setSkipConnectionControl(std::shared_ptr<SkipConnectionControl> control) {
        std::atomic_store(&skip_control_, std::move(control));
    }
    std::shared_ptr<SkipConnectionControl> skip_control_ = std::make_shared<SkipConnectionControl>();
    void setHelpVisible(bool visible)
    {
        help_visible_.store(visible, std::memory_order_relaxed);
    }

    void setLiveControlState(bool validation_enabled,
                             bool overrides_enabled,
                             float learning_rate,
                             int lr_warmup_steps,
                             float kl_beta,
                             int kl_warmup_steps,
                             bool kl_enabled,
                             int selected_parameter,
                             int recon_loss_index)
    {
        validation_enabled_.store(validation_enabled, std::memory_order_relaxed);
        live_overrides_enabled_.store(overrides_enabled, std::memory_order_relaxed);
        live_lr_.store(learning_rate, std::memory_order_relaxed);
        live_lr_warmup_steps_.store(lr_warmup_steps, std::memory_order_relaxed);
        live_kl_beta_.store(kl_beta, std::memory_order_relaxed);
        live_kl_warmup_steps_.store(kl_warmup_steps, std::memory_order_relaxed);
        live_kl_enabled_.store(kl_enabled, std::memory_order_relaxed);
        live_selected_parameter_.store(selected_parameter, std::memory_order_relaxed);
        live_recon_loss_index_.store(recon_loss_index, std::memory_order_relaxed);
    }

    void setSafeStopPending(bool pending)
    {
        safe_stop_pending_.store(pending, std::memory_order_relaxed);
    }

    void moveCursor(int row, int col)
    {
        out() << "\033[" << row << ";" << col << "H";
    }

    void clearLine()
    {
        out() << "\033[2K";
    }

    std::string colorText(const std::string &text, int color)
    {
        return "\033[" + std::to_string(color) + "m" + text + "\033[0m";
    }

    std::string progressBar(float percent, int bar_width = 40, bool show_percent = true)
    {
        percent = std::clamp(percent, 0.0f, 100.0f);
        int filled = static_cast<int>(percent / 100.0f * bar_width);
        std::stringstream ss;
        ss << "[";
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < filled)
            {
                if (percent > 80.0f)
                    ss << colorText("█", 32); // Vert
                else if (percent > 50.0f)
                    ss << colorText("█", 33); // Jaune
                else
                    ss << "█";
            }
            else if (i == filled)
                ss << "▓";
            else
                ss << "░";
        }
        ss << "]";
        if (show_percent)
        {
            ss << " " << std::fixed << std::setprecision(1) << percent << "%";
        }
        return ss.str();
    }

    std::string memoryBar(size_t used_mb, size_t total_mb, int bar_width = 30)
    {
        float percent = (float)used_mb / total_mb * 100.0f;
        int filled = static_cast<int>(std::min(percent, 100.0f) / 100.0f * bar_width);
        std::stringstream ss;
        ss << "[";
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < filled)
            {
                if (percent > 100.0f)
                    ss << colorText("█", 91); // Rouge clignotant
                else if (percent > 80.0f)
                    ss << colorText("█", 31); // Rouge
                else if (percent > 60.0f)
                    ss << colorText("█", 33); // Jaune
                else
                    ss << colorText("█", 32); // Vert
            }
            else
            {
                ss << "░";
            }
        }
        ss << "] " << used_mb << "/" << total_mb << " MB";
        if (percent > 100.0f)
        {
            ss << colorText(" ⚠ OVERFILL!", 91);
        }
        return ss.str();
    }

    std::string metricBar(const std::string &label, float value, float scale = 1.0f, int bar_width = 25)
    {
        float percent = std::min(value * scale * 100.0f, 100.0f);
        int filled = static_cast<int>(percent / 100.0f * bar_width);
        std::stringstream ss;
        ss << std::left << std::setw(18) << label << " [";
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < filled)
            {
                if (percent > 75.0f)
                    ss << colorText("█", 31); // Rouge = élevé
                else if (percent > 50.0f)
                    ss << colorText("█", 33); // Jaune
                else if (percent > 25.0f)
                    ss << colorText("█", 32); // Vert
                else
                    ss << "▓";
            }
            else
            {
                ss << "░";
            }
        }
        ss << "] " << std::fixed << std::setprecision(6) << value;
        return ss.str();
    }

    std::string formatTime(int seconds)
    {
        int hours = seconds / 3600;
        int mins = (seconds % 3600) / 60;
        int secs = seconds % 60;
        std::stringstream ss;
        if (hours > 0)
            ss << hours << "h ";
        ss << std::setw(2) << std::setfill('0') << mins << ":"
           << std::setw(2) << std::setfill('0') << secs;
        return ss.str();
    }

    void updateStats(int epoch, int total_epochs, int batch, int total_batches,
                     float loss, float avg_loss, float lr, int batch_time_ms,
                     size_t memory_mb, double allocator_memory_mb,
                     size_t memory_freed, float bps, size_t params,
                     float t, float kl, float kl_beta_effective,
                     float wass, float ent, float mom,
                     float spat, float temp, float mse,
                     const std::string& recon_loss_type,
                     float grad_norm, float grad_max,
                     int opt_type, int opt_step,
                     float opt_beta1, float opt_beta2,
                     float opt_eps, float opt_weight_decay)
    {
        // Guard: éviter divisions par zéro dans l'UI (0/0 => NaN).
        const int safe_total_epochs = std::max(1, total_epochs);
        const int safe_total_batches = std::max(1, total_batches);
        const int safe_epoch = std::clamp(epoch, 0, safe_total_epochs);
        const int safe_batch = std::clamp(batch, 0, safe_total_batches);

        stats.current_epoch = safe_epoch;
        stats.total_epochs = safe_total_epochs;
        stats.current_batch = safe_batch;
        stats.total_batches = safe_total_batches;
        stats.current_loss = loss;
        stats.avg_loss = avg_loss;
        stats.learning_rate = lr;
        stats.batch_time_ms = batch_time_ms;
        stats.memory_used_mb = memory_mb;
        stats.allocator_memory_mb = allocator_memory_mb;
        stats.memory_freed_mb = memory_freed;
        stats.batches_per_sec = bps;
        stats.total_params = params;
        stats.timestep = t;
        stats.kl_divergence = kl;
        stats.wasserstein = wass;
        stats.entropy_diff = ent;
        stats.moment_mismatch = mom;
        stats.spatial_coherence = spat;
        stats.temporal_consistency = temp;
        stats.mse_loss = mse;
        stats.recon_loss_type = recon_loss_type;
        stats.grad_norm = grad_norm;
        stats.grad_max = grad_max;
        stats.opt_type = opt_type;
        stats.opt_step = opt_step;
        stats.opt_beta1 = opt_beta1;
        stats.opt_beta2 = opt_beta2;
        stats.opt_eps = opt_eps;
        stats.opt_weight_decay = opt_weight_decay;

        // Calculer ETA (best-effort mais stable):
        // - préfère bps fourni
        // - sinon utilise batch_time_ms si dispo
        // - sinon moyenne depuis le début (progress/elapsed)
        const int batches_remaining = (safe_total_epochs - safe_epoch) * safe_total_batches + (safe_total_batches - safe_batch);
        float effective_bps = bps;
        if (effective_bps <= 0.0f && batch_time_ms > 0) {
            effective_bps = 1000.0f / static_cast<float>(std::max(1, batch_time_ms));
        }
        if (effective_bps <= 0.0f) {
            const auto now = std::chrono::steady_clock::now();
            const float elapsed_s = std::max(0.001f, (float)std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() / 1000.0f);
            const int processed = std::max(0, (safe_epoch - 1) * safe_total_batches + safe_batch);
            if (processed > 0) {
                effective_bps = (float)processed / elapsed_s;
            }
        }

        stats.batches_per_sec = effective_bps;
        if (effective_bps > 0.0f) {
            stats.eta_seconds = static_cast<int>(std::max(0.0f, (float)batches_remaining / effective_bps));
        } else {
            stats.eta_seconds = 0;
        }

        if (csv_enabled) {
            CsvRecord record;
            record.step = static_cast<int>(csv_history.size());
            record.epoch = safe_epoch;
            record.total_epochs = safe_total_epochs;
            record.batch = safe_batch;
            record.total_batches = safe_total_batches;
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
            record.timestep = t;
            record.grad_norm = grad_norm;
            record.grad_max = grad_max;
            record.opt_type = opt_type;
            record.opt_step = opt_step;
            record.opt_beta1 = opt_beta1;
            record.opt_beta2 = opt_beta2;
            record.opt_eps = opt_eps;
            record.opt_weight_decay = opt_weight_decay;
            csv_history.push_back(record);

            saveLossHistoryCsv(csv_log_file);
        }
    }

    void setCsvEnabled(bool enabled)
    {
        csv_enabled = enabled;
    }

    // Enregistre une ligne de validation dans le CSV (thread-safe : appelé depuis le thread d'entraînement).
    void addValidationRecord(float val_loss_v, float val_mse_v, int val_step_v)
    {
        if (!csv_enabled) return;
        CsvRecord r{};
        r.step     = static_cast<int>(csv_history.size());
        r.val_loss = val_loss_v;
        r.val_mse  = val_mse_v;
        r.val_step = val_step_v;
        r.is_val   = true;
        csv_history.push_back(r);
        saveLossHistoryCsv(csv_log_file);
    }

    void setCsvLogFile(const std::string& filepath)
    {
        csv_log_file = filepath;
        std::cerr << "[htop] csv_log_file=" << csv_log_file << std::endl;
    }

    void configKey(char key) {
        auto config = std::atomic_load(&runtime_config_);
        auto& editor = config_editor_;
        if (!editor.editing && (key == 'c' || key == 'C')) editor.toggle();
        else if (key == '\033') editor.cancel();
        else if (key == '\r' || key == '\n') editor.enter(*config);
        else if (key == '\t') editor.move(1, config->snapshot().size());
        else editor.text(static_cast<unsigned char>(key));
    }
    void configMove(int delta) {
        config_editor_.move(delta, std::atomic_load(&runtime_config_)->snapshot().size());
    }
    bool configVisible() const { return config_editor_.visible; }

    void renderConfiguration() {
        if (width < 32 || height < 10) {
            resetCursor(); clearLine();
            out() << clipToWidth("Config: agrandir le terminal (C fermer)", std::max(1, width)) << std::flush;
            return;
        }
        const auto rows = std::atomic_load(&runtime_config_)->snapshot();
        auto& editor = config_editor_;
        if (!rows.empty()) editor.selected = std::min(editor.selected, rows.size() - 1);
        if (editor.editing) for (size_t i = 0; i < rows.size(); ++i)
            if (rows[i].key == editor.editing_key) editor.selected = i;
        const int columns = std::max(1, width - 4);
        const size_t count = static_cast<size_t>(std::max(1, height - 9));
        const size_t first = (editor.selected / count) * count;
        auto line = [&](int row, const std::string& text) {
            moveCursor(row, 1); clearLine();
            const std::string shown = clipToWidth(text, columns);
            out() << "| " << shown << std::string(std::max(0, columns - static_cast<int>(shown.size())), ' ') << " |";
        };
        line(1, std::string(columns, '='));
        line(2, "CONFIGURATION DU MODELE  [C] fermer  [Esc] annuler");
        line(3, "Fleches: choisir / pages  Entree: modifier puis valider");
        line(4, "DIRECT = modifiable  FIXE = reconstruction / redemarrage requis");
        int row = 5;
        for (size_t i = first; i < rows.size() && i < first + count; ++i) {
            const auto& e = rows[i];
            const bool selected = i == editor.selected;
            std::string value = LiveModelConfig::display(e.value);
            if (e.pending) value += " -> " + LiveModelConfig::display(*e.pending) + " (en attente)";
            const auto text = std::string(selected ? "> " : "  ") +
                (e.owner == LiveModelConfig::Owner::ReadOnly ? "FIXE   " : "DIRECT ") + e.key + " = " + value;
            line(row++, text);
        }
        while (row < height - 3) line(row++, "");
        if (height >= 8) {
            const std::string selected_note = rows.empty() ? "Aucun modele attache" : rows[editor.selected].note;
            line(height - 3, editor.message.empty() ? selected_note : editor.message);
            line(height - 2, editor.editing ? editor.editing_key + " > " + editor.buffer + "_" :
                "Entree pour modifier. Valeurs appliquees par le thread d'entrainement.");
            line(height - 1, std::to_string(rows.empty() ? 0 : editor.selected + 1) + "/" + std::to_string(rows.size()));
            line(height, std::string(columns, '='));
        }
        out() << std::flush;
    }

    void render()
    {
        if (!display_enabled)
            return;

        getTerminalSize();
        if (debug_json_dialog_.visible) { renderDebugJsonDialog(); return; }
        if (config_editor_.visible) { renderConfiguration(); return; }
        if (modelGraphVisible()) { renderModelGraph(); return; }

        // Retourner au début sans effacer l'écran entier (optimisation)
        resetCursor();
        hideCursor();

        int row = 1;

        // Header
        std::string header = "=  Mímir Framework  =";
        int padding = (width > (int)header.length()) ? (width - (int)header.length()) / 2 : 1;
        moveCursor(row++, padding);
        clearLine();
        out() << colorText(header, 96);

        auto now = std::chrono::steady_clock::now();
        int elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        moveCursor(row++, 2);
        clearLine();
        {
            const int remaining = std::max(0,
                (stats.total_epochs - stats.current_epoch) * stats.total_batches +
                (stats.total_batches - stats.current_batch));
            const bool eta_unknown = (remaining > 0) && (stats.batches_per_sec <= 0.0f);
            const std::string eta_txt = eta_unknown ? std::string("--:--") : formatTime(std::max(0, stats.eta_seconds));
            out() << colorText("Uptime:", 36) << " " << formatTime(elapsed)
                  << "  |  " << colorText("ETA:", 36) << " " << eta_txt;
        }

        moveCursor(row++, 1);
        clearLine();
        out() << std::string(std::max(0, width - 2), '_');

        if (help_visible_.load(std::memory_order_relaxed)) {
            const std::vector<std::string> help = {
                "HTOP CONTROLS",
                "H                 Close this help",
                "C                 Open model configuration (Enter to edit/apply)",
                "G                 Open model graph and active planner",
                "V                 Open Viz",
                "D                 Export DebugJSON and continue training",
                "N                 Toggle validation on/off",
                "S                 Toggle encoder-decoder skip connections",
                "Tab / Up / Down   Select LR, warmups, KL beta or recon loss",
                "Left / Right      Decrease / increase selected value",
                "- / +             Decrease / increase selected value",
                "K                 Toggle KL contribution on/off",
                "R                 Reset live parameters to native values",
                "Ctrl+C            Request safe stop and save the current training"
            };
            for (const std::string& line : help) {
                if (row >= height) break;
                moveCursor(row++, 3);
                clearLine();
                out() << clipToWidth(line, std::max(0, width - 4));
            }
            for (int i = row; i < height; ++i) {
                moveCursor(i, 1);
                clearLine();
            }
            out() << std::flush;
            return;
        }

        moveCursor(row++, 2);
        clearLine();
        {
            const int selected = std::clamp(
                live_selected_parameter_.load(std::memory_order_relaxed), 0, 4);
            const char* labels[] = {"LR", "LR-WU", "KL-BETA", "KL-WU", "RECON"};
            const char* recon_losses[] = {"mse", "mae", "huber", "charbonnier", "gaussian_nll", "bce"};
            const int recon_index = std::clamp(
                live_recon_loss_index_.load(std::memory_order_relaxed), 0, 5);
            std::ostringstream controls;
            const auto skips = std::atomic_load(&skip_control_);
            controls << "[S] SKIPS " << (!skips->available.load() ? "N/A" :
                (skips->enabled.load() ? "ON" : "OFF")) << " | LIVE "
                     << (live_overrides_enabled_.load(std::memory_order_relaxed) ? "ON" : "NATIVE")
                     << " | VAL "
                     << (validation_enabled_.load(std::memory_order_relaxed) ? "ON" : "OFF")
                     << " | " << labels[selected] << " selected"
                     << " | lr=" << std::setprecision(6) << std::scientific
                     << live_lr_.load(std::memory_order_relaxed)
                     << " lr-wu=" << std::defaultfloat
                     << live_lr_warmup_steps_.load(std::memory_order_relaxed)
                     << " kl=" << std::setprecision(5)
                     << live_kl_beta_.load(std::memory_order_relaxed)
                     << (live_kl_enabled_.load(std::memory_order_relaxed) ? "" : "(off)")
                     << " kl-wu=" << live_kl_warmup_steps_.load(std::memory_order_relaxed)
                     << " recon=" << recon_losses[recon_index];
            out() << clipToWidth(controls.str(), std::max(0, width - 3));
        }

        // Epoch Progress
        moveCursor(row++, 2);
        clearLine();
        out() << colorText("EPOCH", 33) << " " << stats.current_epoch << "/" << stats.total_epochs;
        float epoch_percent = (float)stats.current_epoch / stats.total_epochs * 100.0f;
        moveCursor(row++, 2);
        clearLine();
        out() << progressBar(epoch_percent, std::max(10, width - 15));

        // Batch Progress
        moveCursor(row++, 2);
        clearLine();
        out() << colorText("BATCH", 33) << " " << stats.current_batch << "/" << stats.total_batches;
        float batch_percent = (float)stats.current_batch / stats.total_batches * 100.0f;
        moveCursor(row++, 2);
        clearLine();
        out() << progressBar(batch_percent, std::max(10, width - 15));

        moveCursor(row++, 1);
        clearLine();
        out() << std::string(std::max(0, width - 2), '_');

        // Loss Metrics
        moveCursor(row++, 2);
        clearLine();
        out() << colorText("● LOSS METRICS", 95);

        moveCursor(row++, 4);
        clearLine();
          out() << "Current: " << colorText(std::to_string(stats.current_loss).substr(0, 10), 93)
              << "  Avg: " << colorText(std::to_string(stats.avg_loss).substr(0, 10), 92);

        if (!stats.recon_loss_type.empty()) {
            moveCursor(row++, 4);
            clearLine();
            out() << "Recon Loss Type  " << colorText(stats.recon_loss_type, 96);
        }

        // Particle Density Loss Components
        moveCursor(row++, 4);
        clearLine();
          out() << "KL Divergence    " << progressBar(std::min(stats.kl_divergence * 10, 100.0f), 25, false)
              << " " << std::fixed << std::setprecision(6) << stats.kl_divergence;

        moveCursor(row++, 4);
        clearLine();
          out() << "Wasserstein      " << progressBar(std::min(stats.wasserstein * 100, 100.0f), 25, false)
              << " " << stats.wasserstein;

        moveCursor(row++, 4);
        clearLine();
          out() << "Entropy Δ        " << progressBar(std::min(stats.entropy_diff * 10, 100.0f), 25, false)
              << " " << stats.entropy_diff;

        moveCursor(row++, 4);
        clearLine();
          out() << "Moment Mismatch  " << progressBar(std::min(stats.moment_mismatch * 20, 100.0f), 25, false)
              << " " << stats.moment_mismatch;

        moveCursor(row++, 4);
        clearLine();
          out() << "Spatial Coher.   " << progressBar(std::min(stats.spatial_coherence * 100, 100.0f), 25, false)
              << " " << stats.spatial_coherence;

        moveCursor(row++, 4);
        clearLine();
          out() << "Temporal Cons.   " << progressBar(std::min(stats.temporal_consistency * 100, 100.0f), 25, false)
              << " " << stats.temporal_consistency;

        moveCursor(row++, 4);
        clearLine();
                {
                    std::string recon_metric_label = "RECON";
                    if (!stats.recon_loss_type.empty()) {
                            recon_metric_label = stats.recon_loss_type;
                    }
                    out() << std::left << std::setw(18) << (recon_metric_label + " (0.1×)")
                            << progressBar(std::min(stats.mse_loss * 100, 100.0f), 25, false)
              << " " << stats.mse_loss;
                }

        moveCursor(row++, 4);
        clearLine();
          out() << "Grad L2          " << progressBar(std::min(stats.grad_norm * 0.1f, 100.0f), 25, false)
              << " " << std::fixed << std::setprecision(6) << stats.grad_norm;

        moveCursor(row++, 4);
        clearLine();
          out() << "Grad MaxAbs      " << progressBar(std::min(stats.grad_max * 1.0f, 100.0f), 25, false)
              << " " << std::fixed << std::setprecision(6) << stats.grad_max;

        moveCursor(row++, 1);
        clearLine();
        out() << std::string(std::max(0, width - 2), '_');

        // System Resources
        moveCursor(row++, 2);
        clearLine();
        out() << colorText("● SYSTEM RESOURCES", 95);

        moveCursor(row++, 4);
        clearLine();
        out() << "Memory: " << memoryBar(stats.memory_used_mb, 10240, 30);

        moveCursor(row++, 4);
        clearLine();
        out() << "Allocator: " << colorText(std::to_string(stats.allocator_memory_mb) + " MB", 96);

        moveCursor(row++, 4);
        clearLine();
        out() << "Freed this batch: " << colorText(std::to_string(stats.memory_freed_mb) + " MB", 92);

        moveCursor(row++, 4);
        clearLine();
          out() << "Parameters: " << colorText(std::to_string(stats.total_params), 96)
              << " (~" << std::fixed << std::setprecision(2)
              << (stats.total_params * 2.0f / 1024.0f / 1024.0f / 1024.0f) << " GB)";

        moveCursor(row++, 1);
        clearLine();
        out() << std::string(std::max(0, width - 2), '_');

        // Training Info
        moveCursor(row++, 2);
        clearLine();
        out() << colorText("● TRAINING INFO", 95);

        moveCursor(row++, 4);
        clearLine();
        out() << "Learning Rate: " << colorText(std::to_string(stats.learning_rate), 93);

        moveCursor(row++, 4);
        clearLine();
        {
            const char* opt_name = "SGD";
            if (stats.opt_type == 1) opt_name = "ADAM";
            else if (stats.opt_type == 2) opt_name = "ADAMW";
            else if (stats.opt_type == 3) opt_name = "LION";
            else if (stats.opt_type == 4) opt_name = "ADAFACTOR";
            else if (stats.opt_type == 5) opt_name = "RADAM";
            else if (stats.opt_type == 6) opt_name = "NADAM";
            else if (stats.opt_type == 7) opt_name = "RMSPROP";
            else if (stats.opt_type == 8) opt_name = "LAMB";

            std::stringstream ss;
            ss << "Optimizer: " << opt_name
               << " | step=" << stats.opt_step
               << " | wd=" << std::fixed << std::setprecision(4) << stats.opt_weight_decay
               << " | b1=" << std::fixed << std::setprecision(3) << stats.opt_beta1
               << " b2=" << std::fixed << std::setprecision(3) << stats.opt_beta2
               << " eps=" << std::scientific << std::setprecision(1) << stats.opt_eps;

            out() << clipToWidth(ss.str(), std::max(0, width - 2));
        }

        moveCursor(row++, 4);
        clearLine();
        out() << "Timestep (t): " << colorText(std::to_string(stats.timestep).substr(0, 6), 96);

        moveCursor(row++, 4);
        clearLine();
        out() << "Batch Time: " << colorText(std::to_string(stats.batch_time_ms) + " ms", 92);

        moveCursor(row++, 4);
        clearLine();
        out() << "Speed: " << colorText(std::to_string(stats.batches_per_sec).substr(0, 5) + " batches/s", 92);

        moveCursor(row++, 1);
        clearLine();
        out() << std::string(std::max(0, width - 2), '_');

        // Logs (stdout/stderr redirigés) : afficher les dernières lignes sans dépasser.
        const int footer_rows = 1;
        const int log_title_rows = 1;
        const int available = std::max(0, height - row - footer_rows - 1);
        const int log_rows = std::max(0, std::min(6, available - log_title_rows));
        if (available >= (log_title_rows + 1)) {
            moveCursor(row++, 2);
            clearLine();
            out() << colorText("● LOGS", 95);

            std::vector<std::string> tail;
            {
                std::lock_guard<std::mutex> lk(log_mutex_);
                // Construire la liste (wrap) depuis la fin.
                const int maxw = std::max(0, width - 6);
                for (auto it = log_lines_.rbegin(); it != log_lines_.rend() && (int)tail.size() < log_rows; ++it) {
                    auto wrapped = wrapToWidth(*it, maxw);
                    // On veut les sous-lignes dans l'ordre (la dernière ligne en bas).
                    for (auto wit = wrapped.rbegin(); wit != wrapped.rend() && (int)tail.size() < log_rows; ++wit) {
                        tail.push_back(*wit);
                    }
                }
                std::reverse(tail.begin(), tail.end());
            }

            for (int i = 0; i < log_rows; ++i) {
                moveCursor(row++, 4);
                clearLine();
                if (i < (int)tail.size()) {
                    out() << clipToWidth(tail[(size_t)i], std::max(0, width - 4));
                }
            }
        }

        // Footer
        moveCursor(row++, 2);
        clearLine();
        const auto viz_state = static_cast<VizActionState>(
            viz_action_state_.load(std::memory_order_relaxed));
        if (safe_stop_pending_.load(std::memory_order_relaxed)) {
            out() << colorText("Stopping safely: checkpoint in progress...", 93);
            out() << std::flush;
            return;
        }
        switch (viz_state) {
            case VizActionState::Ready:
                out() << colorText("[V] Open Viz", 96);
                break;
            case VizActionState::Opening:
                out() << colorText("Viz: opening...", 93);
                break;
            case VizActionState::Active:
                out() << colorText("Viz: active", 92);
                break;
            case VizActionState::Unavailable:
                out() << colorText("Viz: unavailable", 91);
                break;
        }
        out() << colorText("  |  [H] Help  [C] Config  [G] Graph  [D] DebugJSON  |  Ctrl+C: safe stop", 90);

        const auto debug_status = std::atomic_load(&runtime_config_)->debugJsonStatus();
        if (!debug_status.empty()) {
            moveCursor(row++, 2);
            clearLine();
            out() << colorText(debug_status, 96);
        }

        // Effacer les lignes restantes pour éviter les artefacts
        for (int i = row; i < height; ++i)
        {
            moveCursor(i, 1);
            clearLine();
        }

        out() << std::flush;
    }

    // Appelé depuis un thread de capture stdout/stderr.
    void appendLogChunk(const std::string& chunk)
    {
        if (chunk.empty()) return;
        std::lock_guard<std::mutex> lk(log_mutex_);
        log_partial_ += chunk;

        // Split en lignes complètes, conserver le fragment final.
        auto lines = splitLinesKeepNoNewline(log_partial_);
        if (lines.empty()) return;

        // Si le chunk ne se termine pas par \n, la dernière entrée est partielle.
        const bool ends_with_newline = (!log_partial_.empty() && (log_partial_.back() == '\n'));
        const size_t full_count = ends_with_newline ? lines.size() : (lines.size() - 1);

        for (size_t i = 0; i < full_count; ++i) {
            std::string line = rtrimNewlines(lines[i]);
            if (line.empty()) continue;
            log_lines_.push_back(line);
        }
        while (log_lines_.size() > log_max_lines_) {
            log_lines_.pop_front();
        }

        if (ends_with_newline) {
            log_partial_.clear();
        } else {
            log_partial_ = lines.back();
        }
    }
};
