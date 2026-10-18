#ifndef __ASYNC_MONITOR_HPP__
#define __ASYNC_MONITOR_HPP__

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <utility>
#include <optional>
#include <string>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <cerrno>
#include <cstdio>
#include <fstream>
#include <fcntl.h>
#include <cmath>
#include <csignal>
#if defined(_WIN32)
#include <io.h>
#include <conio.h>
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
#define STDERR_FILENO 2
#endif
#else
#include <unistd.h>
#include <termios.h>
#endif

static inline int mimir_os_open(const char* path, int flags) {
#if defined(_WIN32)
    return _open(path, flags);
#else
    return ::open(path, flags);
#endif
}

static inline int mimir_os_dup(int fd) {
#if defined(_WIN32)
    return _dup(fd);
#else
    return ::dup(fd);
#endif
}

static inline int mimir_os_dup2(int oldfd, int newfd) {
#if defined(_WIN32)
    return _dup2(oldfd, newfd);
#else
    return ::dup2(oldfd, newfd);
#endif
}

static inline int mimir_os_close(int fd) {
#if defined(_WIN32)
    return _close(fd);
#else
    return ::close(fd);
#endif
}

static inline int mimir_os_pipe(int fds[2]) {
#if defined(_WIN32)
    return _pipe(fds, 4096, _O_BINARY);
#else
    return ::pipe(fds);
#endif
}

static inline int mimir_os_read(int fd, void* buf, unsigned int count) {
#if defined(_WIN32)
    return _read(fd, buf, count);
#else
    return static_cast<int>(::read(fd, buf, count));
#endif
}
#include "HtopDisplay.hpp"
#include "Visualizer.hpp"
#include "viz/AnnotationFeatureOverlay.hpp"

/**
 * AsyncMonitor - Gestion asynchrone de HtopDisplay et Visualizer
 * 
 * Exécute les moniteurs dans des threads séparés pour ne pas bloquer
 * le processus principal. Synchronisation périodique des métriques.
 */
class AsyncMonitor {
public:
    using LiveTrainParams = Visualizer::LiveTrainParams;

    struct Metrics {
        int epoch = 0;
        int total_epochs = 0;
        int batch = 0;
        int total_batches = 0;
        float loss = 0.0f;
        float avg_loss = 0.0f;
        float lr = 0.0f;
        int batch_time_ms = 0;
        size_t memory_mb = 0;
        double allocator_memory_mb = 0.0;
        size_t memory_freed = 0;
        float bps = 0.0f;
        size_t params = 0;
        float timestep = 0.0f;
        float kl = 0.0f;
        float wass = 0.0f;
        float ent = 0.0f;
        float mom = 0.0f;
        float spat = 0.0f;
        float temp = 0.0f;
        float mse = 0.0f;
        float grad_norm = 0.0f;
        float grad_max = 0.0f;

        // Warmup KL: beta effectif appliqué (après warmup)
        float kl_beta_effective = 0.0f;

        // Nom du type de loss de reconstruction utilisée par le modèle (ex: "MSE", "L1").
        // Laisser vide si non applicable.
        std::string recon_loss_type;

        // Validation (dernières métriques connues)
        bool val_has = false;
        bool val_ok = false;
        bool val_in_progress = false;
        int val_step = 0;
        int val_items = 0;
        int val_done = 0;
        int val_total = 0;
        float val_recon = 0.0f;
        float val_kl = 0.0f;
        float val_align = 0.0f;
        std::string val_feedback;
        Visualizer::DetectionMetrics detection;

        // Optimizer (for display/debug)
        int opt_type = 0;          // 0=SGD, 1=ADAM, 2=ADAMW
        int opt_step = 0;
        float opt_beta1 = 0.0f;
        float opt_beta2 = 0.0f;
        float opt_eps = 0.0f;
        float opt_weight_decay = 0.0f;
    };
    
    AsyncMonitor()
        : running_(false)
        , has_update_(false)
        , update_interval_ms_(100)
        , htop_update_interval_ms_(100)
        , viz_update_interval_ms_(16) {}
    
    ~AsyncMonitor() {
        stop();
    }
    
    // Démarrer les moniteurs
    void start(bool enable_htop = true, bool enable_viz = false,
               const json& viz_config = json(), bool defer_htop = false) {
        // NOTE: start() doit être idempotent.
        // On autorise le cas "htop déjà démarré" puis "activation de la viz" plus tard.
        std::cerr << "[monitor] start htop=" << enable_htop << " viz=" << enable_viz << std::endl;
        if (enable_htop && defer_htop && !htop_) {
            htop_start_pending_.store(true, std::memory_order_release);
            enable_htop = false;
        } else if (enable_htop) {
            htop_start_pending_.store(false, std::memory_order_release);
        }
        if (!running_) {
            running_ = true;
        }
        if (!viz_config.empty()) {
            if (viz_config.contains("visualization") && viz_config["visualization"].is_object()) {
                const auto& v = viz_config["visualization"];
                if (v.contains("loss_log_file") && v["loss_log_file"].is_string()) configureMetricsCsv(v["loss_log_file"].get<std::string>());
            }
            htop_viz_config_ = viz_config;
        }

        if (enable_htop && !htop_) {
            // UI sur un fd dédié (tty) pour que stdout/stderr puissent être redirigés
            // sans casser le rendu.
            if (ui_fd_ < 0) {
                ui_fd_ = mimir_os_open("/dev/tty", O_WRONLY | O_CLOEXEC);
                if (ui_fd_ < 0) {
                    // fallback: dupliquer stdout AVANT redirection
                    ui_fd_ = mimir_os_dup(STDOUT_FILENO);
                }
            }

            if (ui_fd_ < 0) {
                // Dernier recours: écrire sur stdout (peut être capturé, mais évite fd invalide)
                ui_fd_ = STDOUT_FILENO;
            }

            htop_ = std::make_shared<HtopDisplay>(ui_fd_);
            htop_->setSkipConnectionControl(std::atomic_load(&skip_control_));
            htop_->setRuntimeConfiguration(std::atomic_load(&runtime_config_));
            htop_->setCsvLogFile(metrics_csv_file_);
            htop_->setCsvEnabled(false);
            htop_->enterAltScreen();
            htop_->hideCursor();
            htop_->clearScreen();
            htop_->setVizActionState(enable_viz
                ? HtopDisplay::VizActionState::Opening
                : HtopDisplay::VizActionState::Ready);
            setupHtopInput();

            // IMPORTANT: capturer stdout/stderr vers un buffer de logs quand le TUI est actif.
            // Cela évite que des printf cassent le rendu et permet de les afficher dans l'UI.
            startOutputCapture();
            std::cerr << "[monitor] output capture enabled" << std::endl;

            htop_thread_ = std::thread([this]() {
                htopLoop();
            });
        }

        auto current_viz = std::atomic_load_explicit(&viz_, std::memory_order_acquire);
        if (enable_viz && (!current_viz || !viz_thread_.joinable())) {
            viz_launch_requested_ = true;
            json effective_viz_config = viz_config;
            // Si la Viz est explicitement demandée (enable_viz=true), on force
            // le flag d'activation même si le JSON fourni n'a pas ce champ.
            if (!effective_viz_config.contains("visualization") || !effective_viz_config["visualization"].is_object()) {
                effective_viz_config["visualization"] = json::object();
            }
            effective_viz_config["visualization"]["enabled"] = true;

            if (!current_viz) {
                current_viz = std::make_shared<Visualizer>(effective_viz_config);
                current_viz->setSkipConnectionControl(std::atomic_load(&skip_control_));
                current_viz->setRuntimeConfiguration(std::atomic_load(&runtime_config_));
            }

            // Best-effort: permettre de configurer la cadence Viz depuis config.
            // Exemple: {"visualization": {"update_interval_ms": 16}}
            try {
                if (effective_viz_config.contains("visualization")) {
                    const auto& v = effective_viz_config["visualization"];
                    const int ms = v.value("update_interval_ms", static_cast<int>(viz_update_interval_ms_.load()));
                    if (ms > 0) {
                        viz_update_interval_ms_ = ms;
                    }
                }
            } catch (...) {
                // ignore
            }

            // IMPORTANT (SFML): la fenêtre / contexte OpenGL doivent être créés et
            // utilisés dans le même thread. On initialise donc la fenêtre DANS le
            // thread viz avant d'entrer dans la boucle de rendu.
            {
                std::lock_guard<std::mutex> lk(viz_init_mutex_);
                viz_init_done_ = false;
                viz_init_ok_ = false;
                viz_init_err_.clear();
            }

            viz_thread_finished_.store(false, std::memory_order_relaxed);
            viz_thread_ = std::thread([this, current_viz]() {
                bool ok = false;
                std::string err;
                try {
#if !defined(ENABLE_VIZ)
                    ok = false;
                    err = "Visualizer indisponible (MIMIR_VIZ_BACKEND=NONE)";
#else
#if !defined(_WIN32)
                    const char* display = std::getenv("DISPLAY");
                    const char* wayland = std::getenv("WAYLAND_DISPLAY");
                    const bool has_display =
                        (display != nullptr && *display != '\0') ||
                        (wayland != nullptr && *wayland != '\0');
                    if (!has_display) {
                        ok = false;
                        err = "Aucun display graphique (DISPLAY/WAYLAND_DISPLAY absent)";
                    } else
#endif
                    {
                        ok = current_viz->initialize();
                        if (!ok) {
                            err = "Visualizer::initialize() a échoué";
                        } else {
                            {
                                std::lock_guard<std::mutex> lock(mutex_);
                                current_viz->setLossLogFile(metrics_csv_file_);
                            }
                            current_viz->setLossLogEnabled(false);
                            current_viz->setTrainingPauseControl(training_paused_);
                            std::atomic_store_explicit(
                                &viz_, current_viz, std::memory_order_release);
                        }
                    }
#endif
                } catch (const std::exception& e) {
                    ok = false;
                    err = e.what();
                }

                {
                    std::lock_guard<std::mutex> lk(viz_init_mutex_);
                    viz_init_ok_ = ok;
                    viz_init_done_ = true;
                    viz_init_err_ = err;
                }
                viz_init_cv_.notify_all();

                if (!ok) {
                    viz_thread_finished_.store(true, std::memory_order_release);
                    return;
                }

                vizLoop(current_viz);

                // IMPORTANT: détruire la fenêtre SFML dans ce thread.
                current_viz->shutdown();
                setTrainingPaused(false);
                viz_thread_finished_.store(true, std::memory_order_release);
            });

            // Attendre que l'init viz soit terminée (succès ou échec).
            {
                std::unique_lock<std::mutex> lk(viz_init_mutex_);
                viz_init_cv_.wait_for(lk, std::chrono::seconds(2), [&]() { return viz_init_done_; });
            }
            if (htop_) {
                htop_->setVizActionState(vizInitOk()
                    ? HtopDisplay::VizActionState::Active
                    : HtopDisplay::VizActionState::Unavailable);
                htop_->setCsvEnabled(false);
            }
        }
    }
    
    // Arrêter les moniteurs
    void stop() {
        if (!running_ && !csv_thread_.joinable()) return;

        running_ = false;
        setTrainingPaused(false);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            csv_running_ = false;
        }
        csv_cv_.notify_all();
        if (csv_thread_.joinable()) csv_thread_.join();
        cv_.notify_all();
        
        if (htop_thread_.joinable() && std::this_thread::get_id() != htop_thread_.get_id()) {
            htop_thread_.join();
        }
        
        if (viz_thread_.joinable() && std::this_thread::get_id() != viz_thread_.get_id()) {
            viz_thread_.join();
        }

        // Arrêter la capture stdout/stderr (si active) AVANT de détruire htop_.
        stopOutputCapture();
        restoreHtopInput();
        
        if (htop_) {
            htop_->leaveAltScreen();
            htop_->showCursor();
        }

        if (ui_fd_ >= 0 && ui_fd_ != STDOUT_FILENO && ui_fd_ != STDERR_FILENO) {
            mimir_os_close(ui_fd_);
        }
        ui_fd_ = -1;

        // Reset pour permettre un start() ultérieur.
        htop_.reset();
        htop_start_pending_.store(false, std::memory_order_release);
        std::atomic_store_explicit(
            &viz_, std::shared_ptr<Visualizer>{}, std::memory_order_release);
        viz_launch_requested_ = false;
        viz_thread_finished_ = false;
    }

    // Statut init viz (utile pour bindings)
    bool vizInitOk() const {
        std::lock_guard<std::mutex> lk(viz_init_mutex_);
        return viz_init_done_ && viz_init_ok_;
    }
    std::string vizInitError() const {
        std::lock_guard<std::mutex> lk(viz_init_mutex_);
        return viz_init_err_;
    }
    
    // Mettre à jour les métriques (thread-safe)
    void updateMetrics(const Metrics& metrics) {
        if (htop_start_pending_.exchange(false, std::memory_order_acq_rel)) {
            start(true, false, htop_viz_config_);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        // Les updates d'entraînement sont construites depuis un Metrics neuf :
        // leurs champs val_* par défaut ne doivent pas effacer le dernier bilan
        // de validation juste après la reprise du cycle d'entraînement.
        // Une update qui transporte explicitement une validation reste prioritaire.
        if (metrics.val_has || metrics.val_in_progress) {
            metrics_ = metrics;
        } else {
            const bool val_has = metrics_.val_has;
            const bool val_ok = metrics_.val_ok;
            const bool val_in_progress = metrics_.val_in_progress;
            const int val_step = metrics_.val_step;
            const int val_items = metrics_.val_items;
            const int val_done = metrics_.val_done;
            const int val_total = metrics_.val_total;
            const float val_recon = metrics_.val_recon;
            const float val_kl = metrics_.val_kl;
            const float val_align = metrics_.val_align;
            const std::string val_feedback = metrics_.val_feedback;
            const Visualizer::DetectionMetrics previous_detection = metrics_.detection;

            metrics_ = metrics;
            metrics_.val_has = val_has;
            metrics_.val_ok = val_ok;
            metrics_.val_in_progress = val_in_progress;
            metrics_.val_step = val_step;
            metrics_.val_items = val_items;
            metrics_.val_done = val_done;
            metrics_.val_total = val_total;
            metrics_.val_recon = val_recon;
            metrics_.val_kl = val_kl;
            metrics_.val_align = val_align;
            metrics_.val_feedback = val_feedback;
            if (metrics_.detection.enabled && previous_detection.validation_has &&
                !metrics_.detection.validation_has) {
                metrics_.detection.validation_has = true;
                metrics_.detection.precision = previous_detection.precision;
                metrics_.detection.recall = previous_detection.recall;
                metrics_.detection.f1 = previous_detection.f1;
                metrics_.detection.mean_iou = previous_detection.mean_iou;
                metrics_.detection.true_positive = previous_detection.true_positive;
                metrics_.detection.false_positive = previous_detection.false_positive;
                metrics_.detection.false_negative = previous_detection.false_negative;
            }
        }
        recordMetricsLocked();
        has_update_ = true;
        metrics_version_.fetch_add(1, std::memory_order_relaxed);
    }

    // Mettre à jour uniquement l'état de validation (thread-safe) sans écraser
    // les métriques d'entraînement déjà poussées par le C++.
    void updateValidation(bool in_progress,
                          int step,
                          int done,
                          int total,
                          bool has,
                          bool ok,
                          float recon,
                          float kl,
                          float align,
                          const std::string& feedback = std::string()) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (in_progress && !metrics_.val_in_progress) {
            std::lock_guard<std::mutex> viz_lock(viz_mutex_);
            pending_validation_images_reset_ = true;
            pending_images_.erase(std::remove_if(pending_images_.begin(), pending_images_.end(),
                [](const PendingImage& image) { return image.validation; }), pending_images_.end());
        }
        metrics_.val_in_progress = in_progress;
        metrics_.val_step = step;
        metrics_.val_done = std::max(0, done);
        metrics_.val_total = std::max(0, total);

        // Pendant une validation, on peut afficher une progression même si "has" n'est pas encore final.
        if (has || in_progress) {
            metrics_.val_has = true;
        }
        metrics_.val_ok = ok;

        // Afficher les résultats partiels/finals si fournis.
        metrics_.val_recon = recon;
        metrics_.val_kl = kl;
        metrics_.val_align = align;
        metrics_.val_feedback = feedback;

        // Compat UI: val_items représente le volume de validation (total si connu, sinon done).
        if (metrics_.val_total > 0) metrics_.val_items = metrics_.val_total;
        else if (metrics_.val_done > 0) metrics_.val_items = metrics_.val_done;

        if (has && !in_progress && step != last_csv_validation_step_) recordMetricsLocked(true);
        has_update_ = true;
        metrics_version_.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Définir l'intervalle de mise à jour (ms)
    void setUpdateInterval(int ms) {
        update_interval_ms_ = ms;
        htop_update_interval_ms_ = ms;
        viz_update_interval_ms_ = ms;
    }

    // Intervalles séparés (ms)
    void setHtopUpdateInterval(int ms) {
        htop_update_interval_ms_ = ms;
    }
    void setVizUpdateInterval(int ms) {
        viz_update_interval_ms_ = ms;
    }

    void updateRuntimeTrainParams(float lr, int lr_warmup_steps,
                                  float kl_beta, int kl_warmup_steps,
                                  const std::string& recon_loss) {
        if (std::isfinite(lr)) runtime_lr_.store(std::max(0.0f, lr), std::memory_order_relaxed);
        runtime_lr_warmup_steps_.store(std::max(0, lr_warmup_steps), std::memory_order_relaxed);
        if (std::isfinite(kl_beta)) runtime_kl_beta_.store(std::max(0.0f, kl_beta), std::memory_order_relaxed);
        runtime_kl_warmup_steps_.store(std::max(0, kl_warmup_steps), std::memory_order_relaxed);
        runtime_recon_loss_index_.store(reconLossIndex(recon_loss), std::memory_order_relaxed);
        publishRuntimeConfigurationFields();
        if (auto current_viz = getViz()) {
            current_viz->updateRuntimeTrainParams(
                lr, lr_warmup_steps, kl_beta, kl_warmup_steps, recon_loss);
        }
        refreshHtopControlState();
    }

    void publishRuntimeConfigurationFields() {
        auto config = std::atomic_load(&runtime_config_);
        using Owner = LiveModelConfig::Owner;
        std::vector<LiveModelConfig::Entry> entries;
        const auto live = liveTrainParamsSnapshotNoSync();
        const bool enabled = live.overrides_enabled;
        entries.push_back({"learning_rate", enabled ? live.lr : runtime_lr_.load(), Owner::Training,
            1e-12, 100, {}, "Direct; 0 < learning_rate <= 100", {}});
        entries.push_back({"lr_warmup_steps", enabled ? live.lr_warmup_steps : runtime_lr_warmup_steps_.load(),
            Owner::Training, 0, 1000000000, {}, "Direct; entier >= 0", {}});
        entries.push_back({"validation_enabled", validation_enabled_.load(), Owner::Training,
            0, 1, {}, "Direct; true / false", {}});
        for (auto e : config->snapshot()) {
            if (e.owner == Owner::ReadOnly) continue;
            if (e.key == "kl_beta" || e.key == "kl_warmup_steps" || e.key == "recon_loss") {
                e.owner = Owner::Training;
                if (enabled) {
                    if (e.key == "kl_beta") e.value = live.kl_enabled ? live.kl_beta : 0.f;
                    if (e.key == "kl_warmup_steps") e.value = live.kl_warmup_steps;
                    if (e.key == "recon_loss") e.value = live.recon_loss;
                }
                entries.push_back(std::move(e));
            }
        }
        config->publish(std::move(entries));
    }

    void syncRuntimeConfigurationEdits() {
        auto config = std::atomic_load(&runtime_config_);
        json changes = json::object();
        config->apply(LiveModelConfig::Owner::Training, [&](const std::string& key, const json& value) {
            changes[key] = value;
        });
        if (changes.empty()) return;
        if (changes.contains("validation_enabled")) {
            applyValidationControl(changes["validation_enabled"].get<bool>());
            changes.erase("validation_enabled");
        }
        if (!changes.empty()) {
            if (!live_overrides_enabled_.load()) {
                live_lr_ = runtime_lr_.load();
                live_lr_warmup_steps_ = runtime_lr_warmup_steps_.load();
                live_kl_beta_ = runtime_kl_beta_.load();
                live_kl_warmup_steps_ = runtime_kl_warmup_steps_.load();
                live_kl_enabled_ = runtime_kl_beta_.load() > 0;
                live_recon_loss_index_ = runtime_recon_loss_index_.load();
                // Metrics can show beta after warmup. Editing LR must preserve
                // the configured beta, not install its temporary effective value.
                for (const auto& entry : config->snapshot()) {
                    if (entry.key == "kl_beta" && entry.value.is_number()) {
                        live_kl_beta_ = entry.value.get<float>();
                        live_kl_enabled_ = live_kl_beta_.load() > 0;
                    }
                    if (entry.key == "kl_warmup_steps" && entry.value.is_number_integer()) live_kl_warmup_steps_ = entry.value.get<int>();
                    if (entry.key == "recon_loss" && entry.value.is_string()) live_recon_loss_index_ = reconLossIndex(entry.value.get<std::string>());
                }
            }
            if (changes.contains("learning_rate")) live_lr_ = changes["learning_rate"].get<float>();
            if (changes.contains("lr_warmup_steps")) live_lr_warmup_steps_ = changes["lr_warmup_steps"].get<int>();
            if (changes.contains("kl_beta")) {
                live_kl_beta_ = changes["kl_beta"].get<float>();
                live_kl_enabled_ = live_kl_beta_.load() > 0;
            }
            if (changes.contains("kl_warmup_steps")) live_kl_warmup_steps_ = changes["kl_warmup_steps"].get<int>();
            if (changes.contains("recon_loss")) live_recon_loss_index_ = reconLossIndex(changes["recon_loss"].get<std::string>());
            live_overrides_enabled_ = true;
            publishHtopLiveControls(true);
        }
        publishRuntimeConfigurationFields();
    }

    uint64_t liveTrainParamsVersion() {
        syncLiveControlsFromViz();
        syncRuntimeConfigurationEdits();
        return live_params_version_.load(std::memory_order_relaxed);
    }

    LiveTrainParams liveTrainParamsSnapshot() {
        syncLiveControlsFromViz();
        syncRuntimeConfigurationEdits();
        LiveTrainParams params;
        params.overrides_enabled = live_overrides_enabled_.load(std::memory_order_relaxed);
        params.lr = live_lr_.load(std::memory_order_relaxed);
        params.lr_warmup_steps = live_lr_warmup_steps_.load(std::memory_order_relaxed);
        params.kl_beta = live_kl_beta_.load(std::memory_order_relaxed);
        params.kl_warmup_steps = live_kl_warmup_steps_.load(std::memory_order_relaxed);
        params.kl_enabled = live_kl_enabled_.load(std::memory_order_relaxed);
        params.recon_loss = reconLossName(
            live_recon_loss_index_.load(std::memory_order_relaxed));
        params.version = live_params_version_.load(std::memory_order_relaxed);
        return params;
    }

    void bindRuntimeConfiguration(std::shared_ptr<LiveModelConfig> config) {
        std::atomic_store(&runtime_config_, config);
        if (auto viz = getViz()) viz->setRuntimeConfiguration(config);
        if (htop_) htop_->setRuntimeConfiguration(config);
    }

    void bindSkipConnectionControl(std::shared_ptr<SkipConnectionControl> control) {
        std::atomic_store(&skip_control_, control);
        if (auto viz = getViz()) viz->setSkipConnectionControl(control);
        if (htop_) htop_->setSkipConnectionControl(control);
    }

    void updateRuntimeValidationEnabled(bool enabled) {
        if (validation_control_version_.load(std::memory_order_relaxed) == 0) {
            validation_enabled_.store(enabled, std::memory_order_relaxed);
        }
        if (auto current_viz = getViz()) current_viz->updateRuntimeValidationEnabled(enabled);
        refreshHtopControlState();
    }

    bool validationEnabledSnapshot() {
        syncValidationControlFromViz();
        syncRuntimeConfigurationEdits();
        return validation_enabled_.load(std::memory_order_relaxed);
    }

    uint64_t validationControlVersion() {
        syncValidationControlFromViz();
        return validation_control_version_.load(std::memory_order_relaxed);
    }
    
    // Ajouter une image au visualiseur (file "generation")
    void addImage(const std::vector<uint8_t>& pixels, const std::string& prompt) {
        addImage(pixels, 0, 0, 0, prompt);
    }

    void addImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& prompt, bool validation = false) {
        if (!getViz()) return;

        std::lock_guard<std::mutex> lock(viz_mutex_);
        PendingImage img;
        img.pixels = pixels;
        img.w = w;
        img.h = h;
        img.channels = channels;
        img.prompt = prompt;
        img.validation = validation;
        if (staged_dataset_sample_ && !validation) staged_images_.push_back(std::move(img));
        else pending_images_.push_back(std::move(img));
    }

    void clearImages() {
        std::lock_guard<std::mutex> lock(viz_mutex_);
        pending_images_.clear();
        pending_clear_images_ = true;
    }

    // Définir l'image du dataset utilisée (RGB/grayscale)
    void setDatasetImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label, bool bbox_supervision = false, const DatasetBBoxOverlay::Boxes& boxes = {}) {
        if (w <= 0 || h <= 0) return;
        if (channels != 1 && channels != 3 && channels != 4) return;

        std::lock_guard<std::mutex> lock(viz_mutex_);
        context_dirty_ = true;
        if (pending_dataset_sample_) pending_dataset_text_ = pending_dataset_sample_->text;
        pending_dataset_sample_.reset();
        pending_dataset_image_ = PendingFrame{pixels, w, h, channels, label, bbox_supervision, boxes};
    }

    // Stage image + text for the current computation. setLayerBlockImages commits
    // this context together with the completed captures; the UI keeps the previous step.
    void setDatasetSample(
        const std::vector<uint8_t>& pixels,
        int w,
        int h,
        int channels,
        const std::string& label,
        const std::string& raw_text,
        const std::string& tags,
        const std::string& tokenized,
        const std::string& encoded,
        bool bbox_supervision = false,
        const DatasetBBoxOverlay::Boxes& boxes = {},
        const nlohmann::json* annotation = nullptr,
        const nlohmann::json* annotation_config = nullptr
    ) {
        if (w <= 0 || h <= 0) return;
        if (channels != 1 && channels != 3 && channels != 4) return;

        auto annotations=std::make_shared<AnnotationSnapshot>();
        annotations->annotation=annotation?*annotation:nlohmann::json::object();
        annotations->config={{"bbox_loss_weight",annotation_config?annotation_config->value("bbox_loss_weight",1.f):(bbox_supervision?1.f:0.f)}};
        if(annotation_config)for(const char* key:{"relation_primary_weight","relation_secondary_weight","relation_other_weight"})if(annotation_config->contains(key))annotations->config[key]=(*annotation_config)[key];
        if(!annotation && !boxes.empty()) {
            annotations->annotation["entities"]=nlohmann::json::array();size_t i=0;
            for(const auto& box:boxes)annotations->annotation["entities"].push_back({{"id","bbox_"+std::to_string(++i)},{"phrase",""},{"role","object"},{"bbox",box}});
        }
        PendingDatasetSample s;
        s.frame = PendingFrame{pixels, w, h, channels, label, bbox_supervision, boxes};
        s.text = PendingText{raw_text, tags, tokenized, encoded};
        if(annotations->annotation.contains("entities") && !annotations->annotation["entities"].empty()) {
            AnnotationFeatureOverlay::rgb(s.frame.pixels,w,h,channels);
            AnnotationFeatureOverlay::geometry(s.frame.pixels,w,h,annotations->annotation);
            s.frame.channels=3;
        }
        std::lock_guard<std::mutex> lock(viz_mutex_);
        staged_annotations_=std::move(annotations);++annotation_generation_;

        // Input N must not replace the displayed input until outputs N are ready.
        staged_dataset_sample_ = std::move(s);
        staged_projection_image_.reset(); staged_understanding_image_.reset(); staged_detections_.reset();
        staged_images_.clear();
    }

    void stageDatasetImage(const std::vector<uint8_t>& pixels, int w, int h, int channels,
                           const std::string& label, bool bbox_supervision = false,
                           const DatasetBBoxOverlay::Boxes& boxes = {}, const nlohmann::json* annotation = nullptr,
                           const nlohmann::json* annotation_config = nullptr) {
        setDatasetSample(pixels,w,h,channels,label,"","","","",bbox_supervision,boxes,annotation,annotation_config);
    }

    // Définir le texte associé à l'item dataset (si modèle texte)
    void setDatasetText(const std::string& raw_text, const std::string& tags, const std::string& tokenized, const std::string& encoded) {
        std::lock_guard<std::mutex> lock(viz_mutex_);
        if (staged_dataset_sample_) {
            staged_dataset_sample_->text = PendingText{raw_text, tags, tokenized, encoded};
            return;
        }
        context_dirty_ = true;
        if (pending_dataset_sample_) pending_dataset_image_ = pending_dataset_sample_->frame;
        pending_dataset_sample_.reset();
        pending_dataset_text_ = PendingText{raw_text, tags, tokenized, encoded};
    }

    // Définir l'image de projection (souvent une heatmap)
    void setProjectionImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label) {
        if (w <= 0 || h <= 0) return;
        if (channels != 1 && channels != 3 && channels != 4) return;

        std::lock_guard<std::mutex> lock(viz_mutex_);
        if (staged_dataset_sample_) {
            staged_projection_image_ = PendingFrame{pixels, w, h, channels, label, false, {}};
            return;
        }
        context_dirty_ = true;
        pending_projection_image_ = PendingFrame{pixels, w, h, channels, label, false, {}};
    }

    void setUnderstandingImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label) {
        if (w <= 0 || h <= 0) return;
        if (channels != 1 && channels != 3 && channels != 4) return;

        std::lock_guard<std::mutex> lock(viz_mutex_);
        if (staged_dataset_sample_) {
            staged_understanding_image_ = PendingFrame{pixels, w, h, channels, label, false, {}};
            return;
        }
        context_dirty_ = true;
        pending_understanding_image_ = PendingFrame{pixels, w, h, channels, label, false, {}};
    }

    void setDetections(const std::vector<Visualizer::Detection>& detections,
                       Visualizer::DetectionTarget target) {
        if (!getViz()) return;
        std::lock_guard<std::mutex> lock(viz_mutex_);
        if (staged_dataset_sample_) staged_detections_ = PendingDetections{detections, target};
        else pending_detections_ = PendingDetections{detections, target};
    }

    void setDetectionOverlayEnabled(bool enabled) {
        if (!getViz()) return;
        std::lock_guard<std::mutex> lock(viz_mutex_);
        pending_detection_overlay_enabled_ = enabled;
    }

    void setDetectionScoreThreshold(float threshold) {
        if (!getViz()) return;
        std::lock_guard<std::mutex> lock(viz_mutex_);
        pending_detection_score_threshold_ = std::clamp(threshold, 0.0f, 1.0f);
    }

    void setLayerBlockImages(std::vector<Visualizer::BlockFrame> frames) {
        // No capture this step: keep the last completed preview on screen.
        if (frames.empty()) return;
        std::shared_ptr<const AnnotationSnapshot> annotations;
        size_t generation=0;int image_w=1,image_h=1;
        {
            std::lock_guard<std::mutex> lock(viz_mutex_);
            annotations=staged_annotations_;generation=annotation_generation_;
            if(staged_dataset_sample_) {image_w=staged_dataset_sample_->frame.w;image_h=staged_dataset_sample_->frame.h;}
        }
        // Rasterization and feature pooling stay outside the producer/UI lock.
        if(annotations)AnnotationFeatureOverlay::decorate(frames,annotations->annotation,annotations->config,image_w,image_h);
        auto capture = std::make_shared<const std::vector<Visualizer::BlockFrame>>(std::move(frames));
        std::lock_guard<std::mutex> lock(viz_mutex_);
        if(generation!=annotation_generation_)return;
        staged_annotations_.reset();
        if (staged_dataset_sample_) {
            pending_dataset_sample_ = std::move(staged_dataset_sample_);
            staged_dataset_sample_.reset();
            pending_dataset_image_.reset(); pending_dataset_text_.reset();
            pending_projection_image_ = std::move(staged_projection_image_);
            pending_understanding_image_ = std::move(staged_understanding_image_);
            staged_projection_image_.reset(); staged_understanding_image_.reset();
            if (staged_detections_) {
                pending_detections_ = std::move(staged_detections_);
                staged_detections_.reset();
            }
            for (auto& image : staged_images_) pending_images_.push_back(std::move(image));
            staged_images_.clear();
            context_dirty_ = true;
        }
        layer_blocks_dirty_ = true;
        pending_layer_blocks_ = std::move(capture);
    }
    
    // Accesseurs
    std::shared_ptr<HtopDisplay> getHtop() { return htop_; }
    std::shared_ptr<Visualizer> getViz() {
        return std::atomic_load_explicit(&viz_, std::memory_order_acquire);
    }
    bool isRunning() const { return running_; }

    // AsyncMonitor owns the persistent history, independently of either UI.
    void configureMetricsCsv(const std::string& filepath, bool enabled = true) {
        flushMetricsCsv();
        std::lock_guard<std::mutex> lock(mutex_);
        if (!filepath.empty() && filepath != metrics_csv_file_) {
            metrics_csv_file_ = filepath;
            history_loaded_ = false;
        }
        metrics_csv_enabled_ = enabled;
        loadMetricsHistoryLocked();
        metrics_version_.fetch_add(1, std::memory_order_relaxed);
    }

    void flushMetricsCsv() {
        std::unique_lock<std::mutex> lock(mutex_);
        csv_flushed_cv_.wait(lock, [&] { return csv_queue_.empty() && !csv_writing_; });
    }

    bool saveMetricsHistory(const std::string& path) {
        flushMetricsCsv();
        std::lock_guard<std::mutex> lock(mutex_);
        loadMetricsHistoryLocked();
        if (path == metrics_csv_file_) return std::filesystem::exists(path);
        return MetricsCsv::write(path, metrics_history_);
    }

    std::vector<MetricsRecord> metricsHistorySnapshot(size_t offset = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        loadMetricsHistoryLocked();
        offset = std::min(offset, metrics_history_.size());
        return {metrics_history_.begin() + offset, metrics_history_.end()};
    }
    void setTrainingPaused(bool paused) { training_paused_->store(paused); }
    bool trainingPaused() const { return training_paused_->load(); }
    void waitWhileTrainingPaused() {
        while (running_ && trainingPaused()) {
            auto viz = getViz();
            if (interrupt_signal_pending_ || safe_stop_requested_ ||
                (viz && viz->stopTrainingRequested())) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    // Alias historique : le chemin est désormais commun à Htop et Viz.
    void setLossLogFile(const std::string& filepath) {
        if (filepath.empty()) return;
        configureMetricsCsv(filepath, true);
    }

    // Bloquer jusqu'à fermeture de la fenêtre Viz (best-effort).
    // Utile en mode --lua pour éviter que le process se termine dès que le script finit.
    void waitForVizClose() {
        auto current_viz = getViz();
        if (!current_viz) return;
        while (current_viz->isOpen()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    // UI -> training thread: arrêt propre demandé via bouton Viz.
    bool consumeStopTrainingRequested() {
        waitWhileTrainingPaused();
        if (interrupt_signal_pending_ != 0) {
            interrupt_signal_pending_ = 0;
            safe_stop_requested_.store(true, std::memory_order_relaxed);
            if (htop_) {
                htop_->setSafeStopPending(true);
                htop_->appendLogChunk("[htop] Ctrl+C: safe stop requested; saving current training...\n");
            }
        }
        if (safe_stop_requested_.exchange(false, std::memory_order_relaxed)) return true;
        auto current_viz = getViz();
        return current_viz && current_viz->consumeStopTrainingRequested();
    }
    
private:
    friend struct AsyncMonitorPreviewTestAccess;
    void loadMetricsHistoryLocked() {
        if (history_loaded_) return;
        metrics_history_ = MetricsCsv::read(metrics_csv_file_);
        last_csv_validation_step_ = -1;
        for (const auto& record : metrics_history_) if (record.is_val) last_csv_validation_step_ = record.val_step_id;
        history_loaded_ = true;
    }
    void recordMetricsLocked(bool validation_only = false) {
        loadMetricsHistoryLocked();
        MetricsRecord r{};
        r.step = metrics_history_.empty() ? 0 : metrics_history_.back().step + 1;
        r.epoch = metrics_.epoch;
        r.total_epochs = metrics_.total_epochs;
        r.batch = metrics_.batch;
        r.total_batches = metrics_.total_batches;
        r.loss = metrics_.loss;
        r.avg_loss = metrics_.avg_loss;
        r.lr = metrics_.lr;
        r.batch_time_ms = metrics_.batch_time_ms;
        r.bps = metrics_.bps;
        r.memory_mb = metrics_.memory_mb;
        r.allocator_memory_mb = metrics_.allocator_memory_mb;
        r.params = metrics_.params;
        r.mse = metrics_.mse;
        r.kl_divergence = metrics_.kl;
        r.kl_beta_effective = metrics_.kl_beta_effective;
        r.wasserstein = metrics_.wass;
        r.entropy_diff = metrics_.ent;
        r.moment_mismatch = metrics_.mom;
        r.spatial_coherence = metrics_.spat;
        r.temporal_consistency = metrics_.temp;
        r.timestep = metrics_.timestep;
        r.grad_norm = metrics_.grad_norm;
        r.grad_max = metrics_.grad_max;
        r.opt_type = metrics_.opt_type;
        r.opt_step = metrics_.opt_step;
        r.opt_beta1 = metrics_.opt_beta1;
        r.opt_beta2 = metrics_.opt_beta2;
        r.opt_eps = metrics_.opt_eps;
        r.opt_weight_decay = metrics_.opt_weight_decay;
        r.is_val = metrics_.val_has && !metrics_.val_in_progress && metrics_.val_step >= 0 && metrics_.val_step != last_csv_validation_step_;
        if (r.is_val) {
            r.val_loss = metrics_.val_recon; r.val_mse = metrics_.val_kl;
            r.val_step_id = metrics_.val_step; r.val_feedback = metrics_.val_feedback;
            last_csv_validation_step_ = r.val_step_id;
        }
        if (validation_only && !r.is_val) return;
        metrics_history_.push_back(r);
        if (metrics_csv_enabled_) {
            csv_queue_.push_back({metrics_csv_file_, r});
            if (!csv_thread_.joinable()) {
                csv_running_ = true;
                csv_thread_ = std::thread([this] { csvLoop(); });
            }
            csv_cv_.notify_one();
        }
    }

    void csvLoop() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (csv_running_ || !csv_queue_.empty()) {
            csv_cv_.wait(lock, [&] { return !csv_running_ || !csv_queue_.empty(); });
            if (csv_queue_.empty()) continue;
            auto pending = std::move(csv_queue_);
            csv_queue_.clear(); csv_writing_ = true;
            lock.unlock();
            for (const auto& item : pending) {
                if (!MetricsCsv::append(item.first, item.second))
                    std::cerr << "[monitor] Cannot append metrics CSV: " << item.first << std::endl;
            }
            lock.lock(); csv_writing_ = false; csv_flushed_cv_.notify_all();
        }
        csv_flushed_cv_.notify_all();
    }

    static int reconLossIndex(std::string name) {
        std::transform(name.begin(), name.end(), name.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (name == "l1" || name == "mae") return 1;
        if (name == "huber" || name == "smooth_l1" || name == "smoothl1") return 2;
        if (name == "charbonnier") return 3;
        if (name == "gaussian_nll" || name == "nll_gaussian" || name == "gaussian-nll") return 4;
        if (name == "bce") return 5;
        return 0;
    }

    static const char* reconLossName(int index) {
        static constexpr const char* names[] = {
            "mse", "mae", "huber", "charbonnier", "gaussian_nll", "bce"
        };
        return names[std::clamp(index, 0, 5)];
    }

    void startOutputCapture()
    {
        if (capture_running_.load()) return;
        if (!htop_) return;

        // Sauver stdout/stderr actuels.
        saved_stdout_fd_ = mimir_os_dup(STDOUT_FILENO);
        saved_stderr_fd_ = mimir_os_dup(STDERR_FILENO);
        if (saved_stdout_fd_ < 0 || saved_stderr_fd_ < 0) {
            // Best-effort: si dup échoue, ne pas capturer.
            if (saved_stdout_fd_ >= 0) { mimir_os_close(saved_stdout_fd_); saved_stdout_fd_ = -1; }
            if (saved_stderr_fd_ >= 0) { mimir_os_close(saved_stderr_fd_); saved_stderr_fd_ = -1; }
            return;
        }

        if (mimir_os_pipe(pipe_fds_) != 0) {
            mimir_os_close(saved_stdout_fd_);
            mimir_os_close(saved_stderr_fd_);
            saved_stdout_fd_ = -1;
            saved_stderr_fd_ = -1;
            return;
        }

        // Rediriger stdout + stderr vers le pipe.
        ::fflush(stdout);
        ::fflush(stderr);
        mimir_os_dup2(pipe_fds_[1], STDOUT_FILENO);
        mimir_os_dup2(pipe_fds_[1], STDERR_FILENO);

        capture_running_ = true;
        log_thread_ = std::thread([this]() {
            std::string pending;
            pending.reserve(8192);
            std::vector<char> buf;
            buf.resize(4096);
            while (capture_running_.load()) {
                const int n = mimir_os_read(pipe_fds_[0], buf.data(), static_cast<unsigned int>(buf.size()));
                if (n > 0) {
                    if (htop_) {
                        htop_->appendLogChunk(std::string(buf.data(), (size_t)n));
                    }
                    continue;
                }
                if (n == 0) {
                    break; // EOF
                }
                if (errno == EINTR) continue;
                break;
            }
        });
    }

    void stopOutputCapture()
    {
        if (!capture_running_.load()) {
            // Même si capture inactive, s'assurer de fermer des fd restants.
            if (pipe_fds_[0] >= 0) { mimir_os_close(pipe_fds_[0]); pipe_fds_[0] = -1; }
            if (pipe_fds_[1] >= 0) { mimir_os_close(pipe_fds_[1]); pipe_fds_[1] = -1; }
            if (saved_stdout_fd_ >= 0) { mimir_os_close(saved_stdout_fd_); saved_stdout_fd_ = -1; }
            if (saved_stderr_fd_ >= 0) { mimir_os_close(saved_stderr_fd_); saved_stderr_fd_ = -1; }
            return;
        }

        ::fflush(stdout);
        ::fflush(stderr);

        // Restaurer stdout/stderr (cela ferme implicitement les dup2 sur pipe_fds_[1]).
        if (saved_stdout_fd_ >= 0) {
            mimir_os_dup2(saved_stdout_fd_, STDOUT_FILENO);
            mimir_os_close(saved_stdout_fd_);
            saved_stdout_fd_ = -1;
        }
        if (saved_stderr_fd_ >= 0) {
            mimir_os_dup2(saved_stderr_fd_, STDERR_FILENO);
            mimir_os_close(saved_stderr_fd_);
            saved_stderr_fd_ = -1;
        }

        capture_running_ = false;

        // Fermer l'écriture du pipe pour débloquer read().
        if (pipe_fds_[1] >= 0) {
            mimir_os_close(pipe_fds_[1]);
            pipe_fds_[1] = -1;
        }

        if (log_thread_.joinable() && std::this_thread::get_id() != log_thread_.get_id()) {
            log_thread_.join();
        }

        if (pipe_fds_[0] >= 0) {
            mimir_os_close(pipe_fds_[0]);
            pipe_fds_[0] = -1;
        }
    }

    void htopLoop() {
        uint64_t last_ver = 0;
        while (running_) {
            processHtopInput();
            if (viz_thread_finished_.exchange(false, std::memory_order_acq_rel)) {
                if (viz_thread_.joinable()) {
                    viz_thread_.join();
                }
                viz_launch_requested_ = false;
                if (htop_) {
                    htop_->setVizActionState(vizInitOk()
                        ? HtopDisplay::VizActionState::Ready
                        : HtopDisplay::VizActionState::Unavailable);
                    htop_->render();
                }
            }
            if (htop_open_viz_requested_.exchange(false) && !viz_launch_requested_.exchange(true)) {
                if (htop_) {
                    htop_->setVizActionState(HtopDisplay::VizActionState::Opening);
                    htop_->render();
                }
                start(false, true, htop_viz_config_);
                if (htop_) {
                    const bool viz_ok = vizInitOk();
                    htop_->setVizActionState(viz_ok
                        ? HtopDisplay::VizActionState::Active
                        : HtopDisplay::VizActionState::Unavailable);
                    if (!viz_ok) {
                        const std::string error = vizInitError();
                        htop_->appendLogChunk("[viz] " +
                            (error.empty() ? std::string("initialization failed") : error) + "\n");
                    }
                    htop_->render();
                }
            }

            Metrics local_metrics;
            bool has_data = false;
            uint64_t ver = 0;
            
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ver = metrics_version_.load(std::memory_order_relaxed);
                if (ver != last_ver) {
                    local_metrics = metrics_;
                    has_data = true;
                    last_ver = ver;
                }
            }
            
            if (has_data && htop_) {
                htop_->updateStats(
                    local_metrics.epoch, local_metrics.total_epochs,
                    local_metrics.batch, local_metrics.total_batches,
                    local_metrics.loss, local_metrics.avg_loss,
                    local_metrics.lr, local_metrics.batch_time_ms,
                    local_metrics.memory_mb, local_metrics.allocator_memory_mb,
                    local_metrics.memory_freed,
                    local_metrics.bps, local_metrics.params,
                    local_metrics.timestep, local_metrics.kl,
                    local_metrics.kl_beta_effective,
                    local_metrics.wass, local_metrics.ent,
                    local_metrics.mom, local_metrics.spat,
                    local_metrics.temp, local_metrics.mse,
                    local_metrics.recon_loss_type,
                    local_metrics.grad_norm, local_metrics.grad_max,
                    local_metrics.opt_type, local_metrics.opt_step,
                    local_metrics.opt_beta1, local_metrics.opt_beta2,
                    local_metrics.opt_eps, local_metrics.opt_weight_decay
                );
                if (local_metrics.val_has && !local_metrics.val_in_progress &&
                    local_metrics.val_step >= 0 &&
                    local_metrics.val_step != last_htop_validation_step_) {
                    htop_->addValidationRecord(
                        local_metrics.val_recon,
                        local_metrics.val_kl,
                        local_metrics.val_step);
                    last_htop_validation_step_ = local_metrics.val_step;
                }
                htop_->render();
            }
            
            std::this_thread::sleep_for(
                std::chrono::milliseconds(htop_update_interval_ms_.load()));
        }
    }

    void setupHtopInput() {
        interrupt_signal_pending_ = 0;
        previous_sigint_handler_ = std::signal(SIGINT, &AsyncMonitor::handleInterruptSignal);
#if defined(_WIN32)
        htop_input_ready_ = true;
#else
        if (htop_input_fd_ >= 0) return;
        htop_input_fd_ = mimir_os_open("/dev/tty", O_RDONLY | O_NONBLOCK | O_CLOEXEC);
        if (htop_input_fd_ < 0) return;

        if (::tcgetattr(htop_input_fd_, &htop_original_termios_) != 0) {
            mimir_os_close(htop_input_fd_);
            htop_input_fd_ = -1;
            return;
        }
        struct termios raw = htop_original_termios_;
        raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        if (::tcsetattr(htop_input_fd_, TCSANOW, &raw) != 0) {
            mimir_os_close(htop_input_fd_);
            htop_input_fd_ = -1;
            return;
        }
        htop_input_ready_ = true;
#endif
    }

    void restoreHtopInput() {
        if (previous_sigint_handler_ != SIG_ERR) {
            std::signal(SIGINT, previous_sigint_handler_);
            previous_sigint_handler_ = SIG_ERR;
        }
#if defined(_WIN32)
        htop_input_ready_ = false;
#else
        if (htop_input_fd_ >= 0) {
            if (htop_input_ready_) {
                (void)::tcsetattr(htop_input_fd_, TCSANOW, &htop_original_termios_);
            }
            mimir_os_close(htop_input_fd_);
            htop_input_fd_ = -1;
        }
        htop_input_ready_ = false;
#endif
    }

    void processHtopKey(char key) {
        if (htop_ && (htop_->debugJsonVisible() ||
            (!htop_->configVisible() && !htop_->modelGraphVisible() && (key == 'd' || key == 'D')))) {
            htop_->debugJsonKey(key);
            htop_->render();
            return;
        }
        if (htop_ && (htop_->modelGraphVisible() ||
            (!htop_->configVisible() && (key=='g' || key=='G')))) {
            htop_->modelGraphKey(key);
            htop_->render();
            return;
        }
        if (htop_ && (htop_->configVisible() || key == 'c' || key == 'C')) {
            htop_->configKey(key);
            htop_->render();
            return;
        }
        switch (key) {
            case 'h': case 'H':
                htop_help_visible_ = !htop_help_visible_.load(std::memory_order_relaxed);
                if (htop_) htop_->setHelpVisible(htop_help_visible_.load(std::memory_order_relaxed));
                return;
            case 'v': case 'V':
                htop_open_viz_requested_ = true;
                return;
            case 's': case 'S':
                std::atomic_load(&skip_control_)->toggle();
                return;
            case 'n': case 'N':
                applyValidationControl(!validation_enabled_.load(std::memory_order_relaxed));
                return;
            case 'k': case 'K':
                live_overrides_enabled_ = true;
                live_kl_enabled_ = !live_kl_enabled_.load(std::memory_order_relaxed);
                publishHtopLiveControls(true);
                return;
            case 'r': case 'R':
                resetHtopLiveControls();
                return;
            case '\t':
                live_selected_parameter_ = (live_selected_parameter_.load(std::memory_order_relaxed) + 1) % 5;
                refreshHtopControlState();
                return;
            case '+': case '=':
                adjustHtopSelectedParameter(1);
                return;
            case '-': case '_':
                adjustHtopSelectedParameter(-1);
                return;
            default:
                return;
        }
    }

    void processHtopArrow(char arrow) {
        if (htop_ && htop_->debugJsonVisible()) {
            htop_->debug_json_dialog_.move();
            htop_->render();
            return;
        }
        if (htop_ && htop_->modelGraphVisible()) {
            htop_->modelGraphMove(arrow);
            htop_->render();
            return;
        }
        if (htop_ && htop_->configVisible()) {
            htop_->configMove(arrow == 'A' ? -1 : arrow == 'B' ? 1 :
                             (arrow == 'D' || arrow == '5') ? -10 : 10);
            htop_->render();
        } else if (arrow == 'A' || arrow == 'B') {
            const int delta = arrow == 'A' ? 4 : 1;
            live_selected_parameter_ = (live_selected_parameter_.load() + delta) % 5;
            refreshHtopControlState();
        } else if (arrow == 'C' || arrow == 'D') adjustHtopSelectedParameter(arrow == 'C' ? 1 : -1);
    }

    void processHtopInput() {
        if (!htop_input_ready_) return;
#if defined(_WIN32)
        while (_kbhit()) {
            const int key = _getch();
            if (key == 0 || key == 224) {
                const int extended = _getch();
                if (extended == 72) processHtopArrow('A');
                if (extended == 80) processHtopArrow('B');
                if (extended == 75) processHtopArrow('D');
                if (extended == 77) processHtopArrow('C');
            } else processHtopKey(static_cast<char>(key));
        }
#else
        char input[128];
        const int count = mimir_os_read(htop_input_fd_, input, sizeof(input));
        for (int index = 0; index < count; ++index) {
            const char key = input[index];
            if (htop_escape_sequence_.empty()) {
                if (key == '\033') {
                    htop_escape_sequence_ = "\033";
                    htop_escape_started_ = std::chrono::steady_clock::now();
                } else if (!(key == '\n' && htop_previous_cr_)) processHtopKey(key);
                htop_previous_cr_ = key == '\r';
                continue;
            }
            htop_escape_sequence_ += key;
            if (htop_escape_sequence_.size() == 2 && key == '[') continue;
            if (htop_escape_sequence_.size() == 3 && (key == '5' || key == '6')) continue;
            if (htop_escape_sequence_[1] == '[') {
                const char code = htop_escape_sequence_[2];
                if (code == 'A' || code == 'B' || code == 'C' || code == 'D' ||
                    ((code == '5' || code == '6') && key == '~')) processHtopArrow(code);
            } else { processHtopKey('\033'); processHtopKey(key); }
            htop_escape_sequence_.clear();
        }
        if (!htop_escape_sequence_.empty() &&
            std::chrono::steady_clock::now() - htop_escape_started_ > std::chrono::milliseconds(150)) {
            if (htop_escape_sequence_.size() == 1) processHtopKey('\033');
            htop_escape_sequence_.clear();
        }
#endif
    }

    static void handleInterruptSignal(int) {
        interrupt_signal_pending_ = 1;
    }

    void adjustHtopSelectedParameter(int direction) {
        const bool had_overrides = live_overrides_enabled_.load(std::memory_order_relaxed);
        if (!had_overrides) {
            live_lr_ = runtime_lr_.load(std::memory_order_relaxed);
            live_lr_warmup_steps_ = runtime_lr_warmup_steps_.load(std::memory_order_relaxed);
            live_kl_beta_ = runtime_kl_beta_.load(std::memory_order_relaxed);
            live_kl_warmup_steps_ = runtime_kl_warmup_steps_.load(std::memory_order_relaxed);
            live_kl_enabled_ = live_kl_beta_.load(std::memory_order_relaxed) > 0.0f;
            live_recon_loss_index_ = runtime_recon_loss_index_.load(std::memory_order_relaxed);
        }
        live_overrides_enabled_ = true;
        const int selected = live_selected_parameter_.load(std::memory_order_relaxed);
        if (selected == 0) {
            float value = had_overrides
                ? live_lr_.load(std::memory_order_relaxed)
                : runtime_lr_.load(std::memory_order_relaxed);
            value = std::max(1e-12f, value > 0.0f ? value : 1e-4f);
            live_lr_ = direction > 0 ? value * 1.1f : value / 1.1f;
        } else if (selected == 1) {
            const int value = had_overrides
                ? live_lr_warmup_steps_.load(std::memory_order_relaxed)
                : runtime_lr_warmup_steps_.load(std::memory_order_relaxed);
            live_lr_warmup_steps_ = std::max(0, value + direction * std::max(10, value / 10));
        } else if (selected == 2) {
            const float value = had_overrides
                ? live_kl_beta_.load(std::memory_order_relaxed)
                : runtime_kl_beta_.load(std::memory_order_relaxed);
            live_kl_beta_ = std::max(0.0f, value + direction * 0.001f);
            live_kl_enabled_ = live_kl_beta_.load(std::memory_order_relaxed) > 0.0f;
        } else if (selected == 3) {
            const int value = had_overrides
                ? live_kl_warmup_steps_.load(std::memory_order_relaxed)
                : runtime_kl_warmup_steps_.load(std::memory_order_relaxed);
            live_kl_warmup_steps_ = std::max(0, value + direction * std::max(10, value / 10));
        } else {
            const int value = had_overrides
                ? live_recon_loss_index_.load(std::memory_order_relaxed)
                : runtime_recon_loss_index_.load(std::memory_order_relaxed);
            live_recon_loss_index_ = (value + (direction > 0 ? 1 : 5)) % 6;
        }
        publishHtopLiveControls(true);
    }

    void resetHtopLiveControls() {
        live_overrides_enabled_ = false;
        live_lr_ = runtime_lr_.load(std::memory_order_relaxed);
        live_lr_warmup_steps_ = runtime_lr_warmup_steps_.load(std::memory_order_relaxed);
        live_kl_beta_ = runtime_kl_beta_.load(std::memory_order_relaxed);
        live_kl_warmup_steps_ = runtime_kl_warmup_steps_.load(std::memory_order_relaxed);
        live_kl_enabled_ = live_kl_beta_.load(std::memory_order_relaxed) > 0.0f;
        live_recon_loss_index_ = runtime_recon_loss_index_.load(std::memory_order_relaxed);
        publishHtopLiveControls(true);
    }

    void publishHtopLiveControls(bool bump_version) {
        if (!std::isfinite(live_lr_.load(std::memory_order_relaxed)) ||
            live_lr_.load(std::memory_order_relaxed) <= 0.0f) {
            live_lr_ = std::max(1e-12f, runtime_lr_.load(std::memory_order_relaxed));
        }
        if (bump_version) live_params_version_.fetch_add(1, std::memory_order_relaxed);
        if (auto current_viz = getViz()) {
            current_viz->applyLiveTrainParams(liveTrainParamsSnapshotNoSync());
        }
        refreshHtopControlState();
    }

    LiveTrainParams liveTrainParamsSnapshotNoSync() const {
        LiveTrainParams params;
        params.overrides_enabled = live_overrides_enabled_.load(std::memory_order_relaxed);
        params.lr = live_lr_.load(std::memory_order_relaxed);
        params.lr_warmup_steps = live_lr_warmup_steps_.load(std::memory_order_relaxed);
        params.kl_beta = live_kl_beta_.load(std::memory_order_relaxed);
        params.kl_warmup_steps = live_kl_warmup_steps_.load(std::memory_order_relaxed);
        params.kl_enabled = live_kl_enabled_.load(std::memory_order_relaxed);
        params.recon_loss = reconLossName(
            live_recon_loss_index_.load(std::memory_order_relaxed));
        params.version = live_params_version_.load(std::memory_order_relaxed);
        return params;
    }

    void applyValidationControl(bool enabled) {
        validation_enabled_ = enabled;
        validation_control_version_.fetch_add(1, std::memory_order_relaxed);
        if (auto current_viz = getViz()) current_viz->applyValidationControl(enabled);
        refreshHtopControlState();
    }

    void syncLiveControlsFromViz() {
        auto current_viz = getViz();
        if (!current_viz) return;
        const uint64_t viz_version = current_viz->liveTrainParamsVersion();
        if (viz_version == 0 || viz_version == last_viz_live_version_.load(std::memory_order_relaxed)) return;
        last_viz_live_version_ = viz_version;
        const auto params = current_viz->liveTrainParamsSnapshot();
        live_overrides_enabled_ = params.overrides_enabled;
        live_lr_ = params.lr;
        live_lr_warmup_steps_ = params.lr_warmup_steps;
        live_kl_beta_ = params.kl_beta;
        live_kl_warmup_steps_ = params.kl_warmup_steps;
        live_kl_enabled_ = params.kl_enabled;
        live_recon_loss_index_ = reconLossIndex(params.recon_loss);
        live_params_version_.fetch_add(1, std::memory_order_relaxed);
        refreshHtopControlState();
    }

    void syncValidationControlFromViz() {
        auto current_viz = getViz();
        if (!current_viz) return;
        const uint64_t viz_version = current_viz->validationControlVersion();
        if (viz_version == 0 || viz_version == last_viz_validation_version_.load(std::memory_order_relaxed)) return;
        last_viz_validation_version_ = viz_version;
        validation_enabled_ = current_viz->validationEnabledSnapshot();
        validation_control_version_.fetch_add(1, std::memory_order_relaxed);
        refreshHtopControlState();
    }

    void refreshHtopControlState() {
        if (!htop_) return;
        const bool overrides = live_overrides_enabled_.load(std::memory_order_relaxed);
        htop_->setLiveControlState(
            validation_enabled_.load(std::memory_order_relaxed), overrides,
            overrides ? live_lr_.load(std::memory_order_relaxed) : runtime_lr_.load(std::memory_order_relaxed),
            overrides ? live_lr_warmup_steps_.load(std::memory_order_relaxed) : runtime_lr_warmup_steps_.load(std::memory_order_relaxed),
            overrides ? live_kl_beta_.load(std::memory_order_relaxed) : runtime_kl_beta_.load(std::memory_order_relaxed),
            overrides ? live_kl_warmup_steps_.load(std::memory_order_relaxed) : runtime_kl_warmup_steps_.load(std::memory_order_relaxed),
            overrides ? live_kl_enabled_.load(std::memory_order_relaxed)
                      : runtime_kl_beta_.load(std::memory_order_relaxed) > 0.0f,
            live_selected_parameter_.load(std::memory_order_relaxed),
            overrides ? live_recon_loss_index_.load(std::memory_order_relaxed)
                      : runtime_recon_loss_index_.load(std::memory_order_relaxed));
    }
    
    void vizLoop(const std::shared_ptr<Visualizer>& current_viz) {
        uint64_t last_ver = std::numeric_limits<uint64_t>::max();
        size_t history_offset = 0;
        {
            std::lock_guard<std::mutex> lock(viz_mutex_);
            context_dirty_ = true;
            layer_blocks_dirty_ = static_cast<bool>(pending_layer_blocks_);
        }
        while (running_ && current_viz->isOpen()) {
            current_viz->processEvents();
            
            // Mettre à jour métriques
            Metrics local_metrics;
            bool has_new_metrics = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                const uint64_t ver = metrics_version_.load(std::memory_order_relaxed);
                if (ver != last_ver) {
                    local_metrics = metrics_;
                    has_new_metrics = true;
                    last_ver = ver;
                }
            }

            if (has_new_metrics) {
                current_viz->updateMetrics(
                    local_metrics.epoch, local_metrics.batch,
                    local_metrics.loss, local_metrics.lr,
                    local_metrics.mse, local_metrics.kl,
                    local_metrics.wass, local_metrics.ent,
                    local_metrics.mom, local_metrics.spat,
                    local_metrics.temp,
                    local_metrics.timestep,
                    local_metrics.total_epochs, local_metrics.total_batches, local_metrics.avg_loss,
                    local_metrics.batch_time_ms,
                    local_metrics.memory_mb,
                    local_metrics.allocator_memory_mb,
                    local_metrics.bps,
                    local_metrics.params,
                    local_metrics.grad_norm,
                    local_metrics.grad_max,
                    local_metrics.opt_type,
                    local_metrics.opt_step,
                    local_metrics.opt_beta1,
                    local_metrics.opt_beta2,
                    local_metrics.opt_eps,
                    local_metrics.opt_weight_decay,
                    local_metrics.val_has,
                    local_metrics.val_ok,
                    local_metrics.val_step,
                    local_metrics.val_items,
                    local_metrics.val_recon,
                    local_metrics.val_kl,
                    local_metrics.val_align,
                    local_metrics.recon_loss_type,
                    local_metrics.val_in_progress,
                    local_metrics.val_done,
                    local_metrics.val_total,
                    local_metrics.kl_beta_effective,
                    local_metrics.val_feedback,
                    &local_metrics.detection
                );
                auto records = metricsHistorySnapshot(history_offset);
                current_viz->setMetricsHistory(records, history_offset);
                history_offset += records.size();
            }
            
            // Ajouter images en attente
            {
                auto snapshot = takeVizUpdate();

                if (snapshot.pending_loss_log_file_.has_value()) {
                    current_viz->setLossLogFile(snapshot.pending_loss_log_file_.value());
                    snapshot.pending_loss_log_file_.reset();
                }
                if (snapshot.pending_loss_log_enabled_.has_value()) {
                    current_viz->setLossLogEnabled(false);
                    snapshot.pending_loss_log_enabled_.reset();
                }

                if (snapshot.pending_clear_images_) {
                    current_viz->clearImages();
                    snapshot.pending_clear_images_ = false;
                }
                if (snapshot.pending_validation_images_reset_) {
                    current_viz->beginValidationImages();
                    snapshot.pending_validation_images_reset_ = false;
                }
                for (const auto& img : snapshot.pending_images_) {
                    current_viz->addGeneratedImage(img.pixels, img.w, img.h, img.channels, img.prompt, img.validation);
                }
                snapshot.pending_images_.clear();

                if (snapshot.context_dirty_) {
                    // Appliquer d'abord l'update atomique (image+texte) si présent.
                    if (snapshot.pending_dataset_sample_.has_value()) {
                        const auto& s = snapshot.pending_dataset_sample_.value();
                        current_viz->setDatasetImage(s.frame.pixels, s.frame.w, s.frame.h, s.frame.channels, s.frame.label, s.frame.bbox_supervision, s.frame.boxes);
                        current_viz->setDatasetText(s.text.raw, s.text.tags, s.text.tokens, s.text.encoded);
                    }

                    if (snapshot.pending_dataset_image_.has_value()) {
                        const auto& f = snapshot.pending_dataset_image_.value();
                        current_viz->setDatasetImage(f.pixels, f.w, f.h, f.channels, f.label, f.bbox_supervision, f.boxes);
                    }

                    if (snapshot.pending_dataset_text_.has_value()) {
                        const auto& t = snapshot.pending_dataset_text_.value();
                        current_viz->setDatasetText(t.raw, t.tags, t.tokens, t.encoded);
                    }
                    if (snapshot.pending_projection_image_.has_value()) {
                        const auto& f = snapshot.pending_projection_image_.value();
                        current_viz->setProjectionImage(f.pixels, f.w, f.h, f.channels, f.label);
                    }
                    if (snapshot.pending_understanding_image_.has_value()) {
                        const auto& f = snapshot.pending_understanding_image_.value();
                        current_viz->setUnderstandingImage(f.pixels, f.w, f.h, f.channels, f.label);
                    }

                    snapshot.context_dirty_ = false;
                }

                if (snapshot.layer_blocks_dirty_ && snapshot.pending_layer_blocks_) {
                    current_viz->setLayerBlockImages(*snapshot.pending_layer_blocks_);
                    snapshot.layer_blocks_dirty_ = false;
                }

                if (snapshot.pending_detections_.has_value()) {
                    current_viz->setDetections(
                        snapshot.pending_detections_->detections, snapshot.pending_detections_->target);
                    snapshot.pending_detections_.reset();
                }
                if (snapshot.pending_detection_overlay_enabled_.has_value()) {
                    current_viz->setDetectionOverlayEnabled(*snapshot.pending_detection_overlay_enabled_);
                    snapshot.pending_detection_overlay_enabled_.reset();
                }
                if (snapshot.pending_detection_score_threshold_.has_value()) {
                    current_viz->setDetectionScoreThreshold(*snapshot.pending_detection_score_threshold_);
                    snapshot.pending_detection_score_threshold_.reset();
                }


            }
            
            current_viz->update();
            
            std::this_thread::sleep_for(
                std::chrono::milliseconds(viz_update_interval_ms_.load()));
        }
    }
    
    std::shared_ptr<HtopDisplay> htop_;
    std::shared_ptr<Visualizer> viz_;
    
    std::thread csv_thread_;
    std::condition_variable csv_cv_, csv_flushed_cv_;
    bool csv_running_ = false, csv_writing_ = false;
    std::vector<std::pair<std::string, MetricsRecord>> csv_queue_;
    std::thread htop_thread_;
    std::thread viz_thread_;

    // UI fd (tty) + capture stdout/stderr
    int ui_fd_ = -1;
    int htop_input_fd_ = -1;
    bool htop_input_ready_ = false;
#if !defined(_WIN32)
    struct termios htop_original_termios_ {};
#endif
    int saved_stdout_fd_ = -1;
    int saved_stderr_fd_ = -1;
    int pipe_fds_[2] = {-1, -1};
    std::atomic<bool> capture_running_{false};
    std::atomic<bool> htop_start_pending_{false};
    std::atomic<bool> viz_launch_requested_{false};
    std::atomic<bool> viz_thread_finished_{false};
    std::shared_ptr<LiveModelConfig> runtime_config_ = std::make_shared<LiveModelConfig>();
    std::string htop_escape_sequence_;
    std::chrono::steady_clock::time_point htop_escape_started_;
    bool htop_previous_cr_ = false;
    std::shared_ptr<SkipConnectionControl> skip_control_ = std::make_shared<SkipConnectionControl>();
    std::atomic<bool> htop_open_viz_requested_{false};
    std::atomic<bool> htop_help_visible_{false};
    std::atomic<bool> safe_stop_requested_{false};
    std::thread log_thread_;
    json htop_viz_config_;
    using SignalHandler = void (*)(int);
    SignalHandler previous_sigint_handler_ = SIG_ERR;
    inline static volatile std::sig_atomic_t interrupt_signal_pending_ = 0;

    std::atomic<bool> validation_enabled_{false};
    std::atomic<uint64_t> validation_control_version_{0};
    std::atomic<uint64_t> last_viz_validation_version_{0};
    std::atomic<uint64_t> live_params_version_{0};
    std::atomic<uint64_t> last_viz_live_version_{0};
    std::atomic<bool> live_overrides_enabled_{false};
    std::atomic<float> live_lr_{0.0f};
    std::atomic<int> live_lr_warmup_steps_{0};
    std::atomic<float> live_kl_beta_{0.0f};
    std::atomic<int> live_kl_warmup_steps_{0};
    std::atomic<bool> live_kl_enabled_{false};
    std::atomic<int> live_selected_parameter_{0};
    std::atomic<int> live_recon_loss_index_{0};
    std::atomic<float> runtime_lr_{0.0f};
    std::atomic<int> runtime_lr_warmup_steps_{0};
    std::atomic<float> runtime_kl_beta_{0.0f};
    std::atomic<int> runtime_kl_warmup_steps_{0};
    std::atomic<int> runtime_recon_loss_index_{0};
    
    std::mutex mutex_;
    std::mutex viz_mutex_;
    std::condition_variable cv_;

    // Synchronisation init SFML
    mutable std::mutex viz_init_mutex_;
    std::condition_variable viz_init_cv_;
    bool viz_init_done_ = false;
    bool viz_init_ok_ = false;
    std::string viz_init_err_;
    
    std::atomic<bool> running_;
    std::atomic<bool> has_update_;
    int update_interval_ms_;
    std::atomic<int> htop_update_interval_ms_;
    std::atomic<int> viz_update_interval_ms_;
    std::atomic<uint64_t> metrics_version_{0};
    std::shared_ptr<std::atomic<bool>> training_paused_ = std::make_shared<std::atomic<bool>>(false);
    std::vector<MetricsRecord> metrics_history_;
    bool history_loaded_ = false;
    int last_csv_validation_step_ = -1;
    bool metrics_csv_enabled_ = true;
    std::string metrics_csv_file_ = "checkpoints/loss_history.csv";
    int last_htop_validation_step_ = -1;
    
    Metrics metrics_;
    
    struct PendingImage {
        std::vector<uint8_t> pixels;
        int w = 0;
        int h = 0;
        int channels = 0;
        std::string prompt;
        bool validation = false;
    };
    bool pending_validation_images_reset_ = false;
    bool pending_clear_images_ = false;
    std::vector<PendingImage> pending_images_;
    std::vector<PendingImage> staged_images_;

    struct AnnotationSnapshot {nlohmann::json annotation; nlohmann::json config;};
    std::shared_ptr<const AnnotationSnapshot> staged_annotations_;
    size_t annotation_generation_=0;

    struct PendingFrame {
        std::vector<uint8_t> pixels;
        int w = 0;
        int h = 0;
        int channels = 1;
        std::string label;
        bool bbox_supervision = false;
        DatasetBBoxOverlay::Boxes boxes;
    };
    std::optional<PendingFrame> pending_dataset_image_;
    struct PendingText {
        std::string raw;
        std::string tags;
        std::string tokens;
        std::string encoded;
    };
    std::optional<PendingText> pending_dataset_text_;

    struct PendingDatasetSample {
        PendingFrame frame;
        PendingText text;
    };
    std::optional<PendingDatasetSample> pending_dataset_sample_;
    std::optional<PendingDatasetSample> staged_dataset_sample_;
    std::optional<PendingFrame> pending_projection_image_;
    std::optional<PendingFrame> pending_understanding_image_;
    std::optional<PendingFrame> staged_projection_image_, staged_understanding_image_;

    struct PendingDetections {
        std::vector<Visualizer::Detection> detections;
        Visualizer::DetectionTarget target = Visualizer::DetectionTarget::Dataset;
    };
    std::optional<PendingDetections> pending_detections_;
    std::optional<PendingDetections> staged_detections_;
    std::optional<bool> pending_detection_overlay_enabled_;
    std::optional<float> pending_detection_score_threshold_;

    bool context_dirty_ = false;
    bool layer_blocks_dirty_ = false;
    std::shared_ptr<const std::vector<Visualizer::BlockFrame>> pending_layer_blocks_;


    // Only immutable data crosses the mailbox. Texture creation and layout never
    // hold the producer's mutex; completed captures are replaced, not queued.
    struct VizUpdate {
        std::optional<std::string> pending_loss_log_file_;
        std::optional<bool> pending_loss_log_enabled_;
        bool pending_clear_images_ = false, pending_validation_images_reset_ = false;
        std::vector<PendingImage> pending_images_;
        bool context_dirty_ = false, layer_blocks_dirty_ = false;
        std::optional<PendingDatasetSample> pending_dataset_sample_;
        std::optional<PendingFrame> pending_dataset_image_, pending_projection_image_, pending_understanding_image_;
        std::optional<PendingText> pending_dataset_text_;
        std::shared_ptr<const std::vector<Visualizer::BlockFrame>> pending_layer_blocks_;
        std::optional<PendingDetections> pending_detections_;
        std::optional<bool> pending_detection_overlay_enabled_;
        std::optional<float> pending_detection_score_threshold_;
    };

    VizUpdate takeVizUpdate() {
        VizUpdate snapshot;
        std::lock_guard<std::mutex> lock(viz_mutex_);
        snapshot.pending_loss_log_file_ = std::move(pending_loss_log_file_);
        pending_loss_log_file_.reset();
        snapshot.pending_loss_log_enabled_ = pending_loss_log_enabled_;
        pending_loss_log_enabled_.reset();
        snapshot.pending_clear_images_ = std::exchange(pending_clear_images_, false);
        snapshot.pending_validation_images_reset_ = std::exchange(pending_validation_images_reset_, false);
        snapshot.pending_images_.swap(pending_images_);
        snapshot.context_dirty_ = std::exchange(context_dirty_, false);
        if (snapshot.context_dirty_) {
            snapshot.pending_dataset_sample_ = pending_dataset_sample_;
            snapshot.pending_dataset_image_ = pending_dataset_image_;
            snapshot.pending_dataset_text_ = pending_dataset_text_;
            snapshot.pending_projection_image_ = pending_projection_image_;
            snapshot.pending_understanding_image_ = pending_understanding_image_;
        }
        snapshot.layer_blocks_dirty_ = std::exchange(layer_blocks_dirty_, false);
        if (snapshot.layer_blocks_dirty_) snapshot.pending_layer_blocks_ = pending_layer_blocks_;
        snapshot.pending_detections_ = std::move(pending_detections_);
        pending_detections_.reset();
        snapshot.pending_detection_overlay_enabled_ = pending_detection_overlay_enabled_;
        pending_detection_overlay_enabled_.reset();
        snapshot.pending_detection_score_threshold_ = pending_detection_score_threshold_;
        pending_detection_score_threshold_.reset();
        return snapshot;
    }
    // Pending Viz-side settings
    std::optional<std::string> pending_loss_log_file_;
    std::optional<bool> pending_loss_log_enabled_;
};

#endif // __ASYNC_MONITOR_HPP__
