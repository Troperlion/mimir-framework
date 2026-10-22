#ifndef __TENSOR_VISUALIZER_HPP__
#define __TENSOR_VISUALIZER_HPP__
#include "SkipConnectionControl.hpp"
#include "MetricsHistory.hpp"
#include "viz/TensorSelection.hpp"
#include "LiveModelConfig.hpp"
#include "DebugJsonExportDialog.hpp"
#include "viz/DatasetBBoxOverlay.hpp"

#ifdef ENABLE_VIZ
#include "viz/Graphics.hpp"
#include "VizWindow.hpp"
#endif
#include <vector>
#include <string>
#include <deque>
#include <array>
#include <memory>
#include <sstream>
#include <unordered_set>
#include <optional>
#include <atomic>
#include <cstdint>
#include <mutex>
#include "include/json.hpp"

using json = nlohmann::json;

#ifdef ENABLE_VIZ
class Visualizer {
public:
    Visualizer(const json& config);
    ~Visualizer();

    // Ferme la fenêtre et libère les ressources du backend sélectionné.
    // À appeler idéalement depuis le thread qui a créé la fenêtre.
    void shutdown();

    // Initialisation de la fenêtre
    bool initialize();

    // Vérifier si la fenêtre est ouverte
    bool isOpen() const;

    // Traiter les événements (fermeture, etc.)
    void processEvents();
    // Thread-safe mailbox. Scene mutations are applied only by the Viz thread.
    void configureScript(const json& config);
    json pollScriptEvents();

    // Requête de resynchronisation (thread-safe): utilisée pour que le thread
    // d'entraînement puisse détecter que l'utilisateur a pressé 'R' dans la Viz.
    void requestResync();
    bool consumeResyncRequested();

    // Requête d'arrêt de l'entraînement depuis l'UI (bouton dans le panneau Metrics).
    void requestStopTraining();
    bool stopTrainingRequested() const { return stop_training_requested_.load(); }
    bool consumeStopTrainingRequested();
    void setTrainingPauseControl(std::shared_ptr<std::atomic<bool>> control) { std::atomic_store(&training_paused_, std::move(control)); }
    void setMetricsHistory(const std::vector<MetricsRecord>& records, size_t offset = 0);

    // Validation runtime: la configuration initialise l'état, puis tout clic
    // utilisateur devient prioritaire jusqu'à la fin de l'entraînement.
    void setRuntimeConfiguration(std::shared_ptr<LiveModelConfig> config) {
        if (!config) config=std::make_shared<LiveModelConfig>();
        config->gradient_diagnostics.request();
        auto previous=std::atomic_exchange(&runtime_config_,config);
        if(previous && previous!=config) previous->requestModelGraph(false);
    }
    std::shared_ptr<LiveModelConfig> runtime_config_ = std::make_shared<LiveModelConfig>();
    RuntimeConfigEditor config_editor_;
    DebugJsonExportDialog debug_json_dialog_;
    void renderDebugJsonDialog();
    void setSkipConnectionControl(std::shared_ptr<SkipConnectionControl> control) {
        std::atomic_store(&skip_control_, std::move(control));
    }
    std::shared_ptr<SkipConnectionControl> skip_control_ = std::make_shared<SkipConnectionControl>();
    void updateRuntimeValidationEnabled(bool enabled);
    void applyValidationControl(bool enabled);
    bool validationEnabledSnapshot() const;
    uint64_t validationControlVersion() const;

    // Paramètres d'entraînement modifiables en direct depuis l'UI (thread-safe).
    struct LiveTrainParams {
        bool overrides_enabled = false;
        float lr = 0.0f;
        int lr_warmup_steps = 0;
        float kl_beta = 0.0f;
        int kl_warmup_steps = 0;
        bool kl_enabled = false;
        std::string recon_loss = "mse";
        uint64_t version = 0;
    };
    uint64_t liveTrainParamsVersion() const;
    LiveTrainParams liveTrainParamsSnapshot() const;
    void applyLiveTrainParams(const LiveTrainParams& params);

    enum class ValidationFeedbackIcon { None = 0, Reward = 1, Penalty = 2 };
    void updateRuntimeTrainParams(float lr, int lr_warmup_steps,
                                  float kl_beta, int kl_warmup_steps,
                                  const std::string& recon_loss);
    void showValidationFeedback(ValidationFeedbackIcon icon);

    // Mettre à jour l'affichage
    void update();

    // Ajouter une image générée.
    // Compat legacy: l'ancienne API supposait une image carrée en 1 canal (w/h inférés via sqrt).
    void addGeneratedImage(const std::vector<uint8_t>& image, const std::string& prompt);
    // Nouvelle API: dimensions explicites (supporte RGB).
    void addGeneratedImage(const std::vector<uint8_t>& image, int w, int h, int channels, const std::string& prompt, bool validation = false);

    void beginValidationImages();

    // Définir l'image du dataset actuellement utilisée (RGB ou grayscale).
    void setDatasetImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label, bool bbox_supervision = false, const DatasetBBoxOverlay::Boxes& boxes = {});

    // Si le modele utilise du texte provenant du dataset: afficher le prompt,
    // les tags normalises, une forme tokenisee (ids) et un resume de l'encodage.
    void setDatasetText(const std::string& raw_text, const std::string& tags, const std::string& tokenized, const std::string& encoded);

    // Définir la sortie de projection (visualisée en image/grille, généralement grayscale).
    void setProjectionImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label);

    // Définir une visualisation de “compréhension” (feature/projection) en heatmap/grille.
    void setUnderstandingImage(const std::vector<uint8_t>& pixels, int w, int h, int channels, const std::string& label);

    struct Detection {
        float x1 = 0.0f;
        float y1 = 0.0f;
        float x2 = 0.0f;
        float y2 = 0.0f;
        float score = -1.0f;
        int class_id = -1;
        std::string label;
    };
    enum class DetectionTarget { Dataset, Output, Both };
    void setDetections(const std::vector<Detection>& detections, DetectionTarget target);
    void setDetectionOverlayEnabled(bool enabled);
    void setDetectionScoreThreshold(float threshold);

    struct DetectionMetrics {
        bool enabled = false;
        float object_loss = 0.0f;
        float box_loss = 0.0f;
        float class_loss = 0.0f;
        int positive_cells = 0;
        bool validation_has = false;
        float precision = 0.0f;
        float recall = 0.0f;
        float f1 = 0.0f;
        float mean_iou = 0.0f;
        int true_positive = 0;
        int false_positive = 0;
        int false_negative = 0;
    };

    // Définir les images intermédiaires (traitement par blocs/layers)
    struct BlockFrame {
        std::vector<uint8_t> pixels;      // heatmap (ou naturel pour image_like)
        std::vector<uint8_t> pixels_real; // RGB des canaux, vide si image_like
        int w = 0;
        int h = 0;
        int channels = 1;
        std::string label;
        int heatmap_kind = 0; // 0: image/scalar, 1: signed mean + energy
        std::string tensor_info;
        std::vector<float> tensor_values; // interleaved raw samples, before colour mapping
        std::vector<size_t> tensor_indices;
        int tensor_sample_channels = 0;
    };
    void setLayerBlockImages(const std::vector<BlockFrame>& frames);

    // Mettre à jour les métriques d'entraînement
    void updateMetrics(int epoch, int batch, float loss, float lr, float mse = 0.0f,
                      float kl = 0.0f, float wass = 0.0f, float ent = 0.0f,
                      float mom = 0.0f, float spat = 0.0f, float temp = 0.0f,
                      float timestep = 0.0f,
                      int total_epochs = 0, int total_batches = 0, float avg_loss = 0.0f,
                      int batch_time_ms = 0, size_t memory_mb = 0,
                      double allocator_memory_mb = 0.0, float bps = 0.0f, size_t params = 0,
                      float grad_norm = 0.0f, float grad_max = 0.0f,
                      int opt_type = 0, int opt_step = 0,
                      float opt_beta1 = 0.0f, float opt_beta2 = 0.0f,
                      float opt_eps = 0.0f, float opt_weight_decay = 0.0f,
                      bool val_has = false, bool val_ok = false, int val_step = 0, int val_items = 0,
                      float val_recon = 0.0f, float val_kl = 0.0f, float val_align = 0.0f,
                      const std::string& recon_loss_type = std::string(),
                      bool val_in_progress = false,
                      int val_done = 0,
                      int val_total = 0,
                      float kl_beta_effective = 0.0f,
                      const std::string& val_feedback = std::string(),
                      const DetectionMetrics* detection_metrics = nullptr);

    // Ajouter un point au graphique de loss
    void addLossPoint(float loss);

    // Effacer toutes les images
    void clearImages();

    // Activer/désactiver la visualisation
    void setEnabled(bool enabled);
    bool isEnabled() const;

    // Sauvegarder l'historique de loss dans un fichier CSV
    void saveLossHistory(const std::string& filepath) const;
    void setLossLogEnabled(bool enabled);

    // Définir dynamiquement le chemin du CSV loss (appelé typiquement via AsyncMonitor).
    void setLossLogFile(const std::string& filepath);

private:
    friend struct VisualizerEventTestAccess;
    std::string loss_log_file;  // Chemin du fichier de log
    bool loss_log_enabled_ = false;
    std::shared_ptr<std::atomic<bool>> training_paused_ = std::make_shared<std::atomic<bool>>(false);
    std::optional<vizgfx::FloatRect> last_debug_json_button_rect_;
    std::optional<vizgfx::FloatRect> last_pause_button_rect_;
    std::optional<vizgfx::FloatRect> last_commands_rect_;
    std::optional<vizgfx::FloatRect> last_picker_rect_;
    std::string last_picker_text_;
    vizgfx::Vector2f picker_mouse_{-1.f, -1.f};
    std::optional<vizgfx::FloatRect> picker_image_rect_, picker_image_clip_;
    float picker_image_scale_ = 1.f;
    BlockFrame picker_source_, picker_selection_source_;
    bool picker_source_raw_ = false, picker_selection_raw_ = false;
    bool picker_selecting_ = false, picker_selection_active_ = false;
    int picker_anchor_x_ = 0, picker_anchor_y_ = 0;
    int picker_end_x_ = 0, picker_end_y_ = 0;
    std::optional<vizgfx::FloatRect> picker_selection_rect_;
    TensorSelection::Summary picker_selection_summary_;
    void updatePickerSelection(vizgfx::Vector2f point);

    // Configuration
    bool enabled;
    int window_width;
    int window_height;
    std::string window_title;
    int fps_limit;
    bool show_generated_images;
    bool show_training_progress;
    bool show_loss_graph;

    // Flag thread-safe: mis à true quand l'utilisateur demande une resync (touche R).
    std::atomic<bool> resync_requested_{false};

    // Flag thread-safe: mis à true quand l'utilisateur demande d'arrêter l'entraînement.
    std::atomic<bool> stop_training_requested_{false};
    std::atomic<bool> validation_enabled_{false};
    std::atomic<uint64_t> validation_control_version_{0};

    // Paramètres live (écrits par le thread UI, lus par le thread d'entraînement).
    std::atomic<uint64_t> live_params_version_{0};
    std::atomic<bool> live_overrides_enabled_{false};
    std::atomic<float> live_lr_{0.0f};
    std::atomic<int> live_lr_warmup_steps_{0};
    std::atomic<float> live_kl_beta_{0.0f};
    std::atomic<int> live_kl_warmup_steps_{0};
    std::atomic<bool> live_kl_enabled_{false};
    std::atomic<int> live_recon_loss_index_{0};
    std::atomic<uint64_t> external_live_params_version_{0};

    // Valeurs effectivement appliquées par le thread d'entraînement. Elles
    // alimentent les range-box en mode NATIVE sans publier d'override UI.
    std::atomic<float> runtime_lr_{0.0f};
    std::atomic<int> runtime_lr_warmup_steps_{0};
    std::atomic<float> runtime_kl_beta_{0.0f};
    std::atomic<int> runtime_kl_warmup_steps_{0};
    std::atomic<int> runtime_recon_loss_index_{0};
    std::atomic<int> validation_feedback_icon_{0};

    // État UI live (thread UI uniquement, non atomique).
    bool live_ui_inited_ = false;
    float live_ui_lr_ = 0.0f;
    int live_ui_lr_warmup_steps_ = 0;
    float live_ui_kl_beta_ = 0.0f;
    int live_ui_kl_warmup_steps_ = 0;
    bool live_ui_kl_enabled_ = false;
    int live_ui_recon_loss_index_ = 0;
    uint64_t live_ui_external_version_ = 0;

    // Drag des sliders live.
    enum class LiveDragTarget { None, LR, LRWarmup, KLBeta, KLWarmup };
    LiveDragTarget live_dragging_ = LiveDragTarget::None;

    // Saisie clavier directe des sliders live.
    enum class LiveInputTarget { None, LR, LRWarmup, KLBeta, KLWarmup };
    LiveInputTarget live_input_target_ = LiveInputTarget::None;
    std::string live_input_buffer_;
    uint64_t live_input_error_until_ms_ = 0;

    // Rects des contrôles live (reset à chaque frame).
    std::optional<vizgfx::FloatRect> last_live_overrides_box_;
    std::optional<vizgfx::FloatRect> last_live_kl_enable_box_;
    std::optional<vizgfx::FloatRect> last_live_lr_track_;
    std::optional<vizgfx::FloatRect> last_live_lr_thumb_;
    std::optional<vizgfx::FloatRect> last_live_lrwu_track_;
    std::optional<vizgfx::FloatRect> last_live_lrwu_thumb_;
    std::optional<vizgfx::FloatRect> last_live_klb_track_;
    std::optional<vizgfx::FloatRect> last_live_klb_thumb_;
    std::optional<vizgfx::FloatRect> last_live_klwu_track_;
    std::optional<vizgfx::FloatRect> last_live_klwu_thumb_;
    std::optional<vizgfx::FloatRect> last_live_lr_value_box_;
    std::optional<vizgfx::FloatRect> last_live_lrwu_value_box_;
    std::optional<vizgfx::FloatRect> last_live_klb_value_box_;
    std::optional<vizgfx::FloatRect> last_live_klwu_value_box_;
    std::optional<vizgfx::FloatRect> last_live_recon_loss_box_;

    // Toggles UI (panneau Blocks/Layers)
    std::optional<vizgfx::FloatRect> last_blocks_mode_box_, last_blocks_smooth_box_, last_blocks_palette_box_;
    std::optional<vizgfx::FloatRect> last_blocks_hide_act_box_;
    std::optional<vizgfx::FloatRect> last_blocks_hide_norm_box_;
    std::optional<vizgfx::FloatRect> last_blocks_tips_limit_box_;
    std::optional<vizgfx::FloatRect> last_detection_toggle_box_;
    std::optional<vizgfx::FloatRect> last_detection_threshold_track_;
    std::optional<vizgfx::FloatRect> last_detection_threshold_thumb_;
    bool dragging_detection_threshold_ = false;

    // Label parsing / architecture hints
    bool hide_activation_blocks = false;
    bool hide_normalisation_blocks = false;
    bool show_detection_overlay_ = true;
    bool show_feature_probe_ = false;
    float detection_score_threshold_ = 0.60f;
    int blocks_tips_visible_limit_ = 0; // 0 = illimite
    bool blocks_tips_limit_editing_ = false;
    std::string blocks_tips_limit_input_;
    std::string architecture_path;
    bool architecture_loaded = false;
    std::unordered_set<std::string> arch_layer_names;
    std::unordered_set<std::string> arch_tensor_outputs;
    std::unordered_set<std::string> arch_tensor_sinks;
    uint64_t arch_last_check_ms = 0;
    int graph_history_size;
    int image_grid_cols;
    int image_grid_rows;

    // Backend graphique sélectionné à la compilation
    std::unique_ptr<VizWindow> window;
    vizgfx::Font font;
    bool font_loaded = false;

    // Logo framework/programme (chargé depuis ./logo.png)
    vizgfx::Texture logo_texture_;
    vizgfx::Sprite logo_sprite_{logo_texture_};
    bool logo_loaded_ = false;
    vizgfx::Clock logo_clock_;
    float logo_splash_seconds_ = 2.5f;

    vizgfx::Texture validation_reward_texture_;
    vizgfx::Sprite validation_reward_sprite_{validation_reward_texture_};
    vizgfx::Texture validation_penalty_texture_;
    vizgfx::Sprite validation_penalty_sprite_{validation_penalty_texture_};
    bool validation_reward_icon_loaded_ = false;
    bool validation_penalty_icon_loaded_ = false;

    // Données de visualisation
    struct ImageData {
        std::vector<uint8_t> pixels;     // pixels actifs (affichés)
        std::vector<uint8_t> pixels_alt;  // pixels alternatifs (heatmap ↔ naturel)
        std::string prompt;
        int w = 0;
        int h = 0;
        int channels = 1;     // canaux de pixels
        int channels_alt = 1; // canaux de pixels_alt
        int display_size = 0;
        vizgfx::Texture texture;
        vizgfx::Sprite sprite{texture};

        ImageData() = default;

        // Après copie ou déplacement, le sprite doit pointer vers la NOUVELLE adresse
        // de la texture (pas celle d'un ancien objet temporaire/réalloué).
        ImageData(const ImageData& o)
            : pixels(o.pixels), pixels_alt(o.pixels_alt), prompt(o.prompt)
            , w(o.w), h(o.h), channels(o.channels), channels_alt(o.channels_alt)
            , display_size(o.display_size)
            , texture(o.texture)
            , sprite(o.sprite)
        {
            sprite.setTexture(texture, true);
        }

        ImageData(ImageData&& o) noexcept
            : pixels(std::move(o.pixels)), pixels_alt(std::move(o.pixels_alt))
            , prompt(std::move(o.prompt))
            , w(o.w), h(o.h), channels(o.channels), channels_alt(o.channels_alt)
            , display_size(o.display_size)
            , texture(std::move(o.texture))
            , sprite(std::move(o.sprite))
        {
            sprite.setTexture(texture, true);
        }

        ImageData& operator=(const ImageData& o) {
            if (this != &o) {
                pixels = o.pixels; pixels_alt = o.pixels_alt; prompt = o.prompt;
                w = o.w; h = o.h; channels = o.channels; channels_alt = o.channels_alt;
                display_size = o.display_size;
                texture = o.texture;
                sprite = o.sprite;
                sprite.setTexture(texture, true);
            }
            return *this;
        }

        ImageData& operator=(ImageData&& o) noexcept {
            if (this != &o) {
                pixels = std::move(o.pixels); pixels_alt = std::move(o.pixels_alt);
                prompt = std::move(o.prompt);
                w = o.w; h = o.h; channels = o.channels; channels_alt = o.channels_alt;
                display_size = o.display_size;
                texture = std::move(o.texture);
                sprite = std::move(o.sprite);
                sprite.setTexture(texture, true);
            }
            return *this;
        }
    };
    std::vector<ImageData> generated_images;
    size_t validation_image_count_ = 0; // Pinned prefix: latest validation originals/results.

    // Images contextuelles (dataset + projection)
    ImageData dataset_image;
    bool has_dataset_image = false;
    bool dataset_bbox_supervision_ = false;
    DatasetBBoxOverlay::Boxes dataset_boxes_;
    std::string dataset_label;

    ImageData projection_image;
    bool has_projection_image = false;
    std::string projection_label;

    // Thumbnails (120px) réutilisés dans le panneau "Blocks / Layers"
    ImageData projection_thumb_;
    bool has_projection_thumb_ = false;

    ImageData understanding_image;
    bool has_understanding_image = false;
    std::string understanding_label;

    // Dernière sortie du modèle et sa vignette Blocks/Layers.
    ImageData output_image_;
    ImageData output_thumb_;
    bool has_output_thumb_ = false;
    std::string output_thumb_label_;

    std::vector<Detection> detections_;
    DetectionTarget detection_target_ = DetectionTarget::Dataset;

    // Texte dataset (optionnel)
    bool has_dataset_text = false;
    std::string dataset_text_raw;
    std::string dataset_text_tags;
    std::string dataset_text_tokens;
    std::string dataset_text_encoded;
    enum class DatasetTextSection { Prompt = 0, Tags = 1, Tokens = 2, Encoder = 3, Count = 4 };
    std::array<float, static_cast<size_t>(DatasetTextSection::Count)> dataset_text_scroll_y_{};
    std::array<float, static_cast<size_t>(DatasetTextSection::Count)> dataset_text_scroll_max_{};
    std::array<vizgfx::FloatRect, static_cast<size_t>(DatasetTextSection::Count)> dataset_text_section_rects_{};
    static constexpr float kDatasetTextScrollSpeed = 32.0f;

    std::vector<BlockFrame> last_block_frames_;
    std::vector<ImageData> layer_block_images;
    std::vector<std::string> layer_block_labels;
    bool has_layer_blocks = false;

    // Métriques
    int current_epoch;
    int current_total_epochs;
    int current_batch;
    int current_total_batches;
    float current_loss;
    float current_avg_loss;
    float current_lr;
    float current_mse;
    std::string current_recon_loss_type;
    float current_kl;
    float current_wass;
    float current_ent;
    float current_mom;
    float current_spat;
    float current_temp;
    float current_timestep;
    int current_batch_time_ms;
    size_t current_memory_mb;
    double current_allocator_memory_mb;
    float current_bps;
    size_t current_params;
    float current_grad_norm;
    GradientDiagnostics::Snapshot current_gradient_diagnostics_;
    float current_grad_max;
    int current_opt_type;
    int current_opt_step;
    float current_opt_beta1;
    float current_opt_beta2;
    float current_opt_eps;
    float current_opt_weight_decay;

    // Warmup KL: beta effectif (après warmup)
    float current_kl_beta_effective = 0.0f;

    // Validation (dernières valeurs connues)
    bool current_val_has;
    bool current_val_ok;
    bool current_val_in_progress;
    int current_val_step;
    int current_val_items;
    int current_val_done;
    int current_val_total;
    float current_val_recon;
    float current_val_kl;
    float current_val_align;
    std::string current_val_feedback;
    DetectionMetrics current_detection_metrics_;

    // Lissage visuel des barres Training (évite les sauts brusques).
    float progress_epoch_display_ = 0.0f;
    float progress_batch_display_ = 0.0f;
    std::deque<float> loss_history;

    // Stats pour l'échelle du graph (évite un scan O(n) à chaque frame)
    bool has_loss_stats_ = false;
    float loss_min_ = 0.0f;
    float loss_max_ = 0.0f;
    
    // Historique complet pour CSV (epoch, batch, loss, lr, mse, etc.)
    using LossRecord = MetricsRecord;
    std::vector<LossRecord> full_loss_history;
    int last_recorded_validation_step_ = -1;

    // Méthodes de rendu
    void renderBackground();
    void renderContextImages();
    void renderOutputImage();
    void renderDetectionOverlay(const ImageData& image, float x, float y, float box_size,
                                DetectionTarget rendered_target);
    void renderLayerBlocks();
    void renderGeneratedImages();
    void renderTrainingProgress();
    void renderLossGraph();
    void renderMetrics();

    // Helpers
    void createImageTexture(ImageData& img_data, int w, int h, int channels, int display_size);
    void createLayerBlockTexture(ImageData& img_data, const std::string& label);
    vizgfx::Color getLossColor(float loss);

    // UI
    bool show_help_overlay_ = false;
    float help_scroll_y_ = 0.f, help_scroll_max_ = 0.f;
    int help_columns_ = 1;
    std::optional<vizgfx::FloatRect> help_body_rect_, help_close_rect_;
    std::vector<vizgfx::FloatRect> help_row_rects_;
    std::mutex script_mutex_;
    std::optional<json> script_pending_;
    json script_scene_ = json::object();
    json script_events_ = json::array();
    void applyScriptScene();
    void renderScriptControls();
    void renderConfigurationOverlay();
    void renderModelGraphOverlay();
    bool show_model_graph_ = false;
    float model_graph_x_ = 0.f, model_graph_y_ = 0.f;
    float model_graph_max_x_ = 0.f, model_graph_max_y_ = 0.f;
    size_t model_graph_selected_ = 0;
    std::vector<std::pair<vizgfx::FloatRect,size_t>> model_graph_hits_;
    std::shared_ptr<const json> model_graph_display_;

    std::vector<std::pair<vizgfx::FloatRect, size_t>> config_row_rects_;
    void emitScriptEvent(const json& event);
    bool show_prompt_text_ = true;
    bool zoom_active_ = false;
    bool zoom_actual_size_ = false;
    float zoom_info_scroll_ = 0.0f;
    float zoom_info_scroll_max_ = 0.0f;
    bool smooth_layer_block_previews_ = true;
    enum class FocusTarget {
        Dataset,
        Projection,
        Understanding,
        LayerBlock,
        Generated,
        Output,
        Training,
        Metrics,
        Graph
    };
    FocusTarget focus_target_ = FocusTarget::Dataset;
    int focus_block_index_ = 0;
    int focus_generated_index_ = 0;

    // Layout: panneaux déplaçables (drag & drop)
    enum class PanelId { Context = 0, Blocks = 1, Generated = 2, Training = 3, Metrics = 4, Graph = 5, Output = 6, Count = 7 };
    struct Panel {
        vizgfx::Vector2f pos{0.f, 0.f};
        vizgfx::Vector2f size{0.f, 0.f};
        std::string title;
        bool visible = true;
        bool allow_drag = true;
    };
    std::array<Panel, static_cast<size_t>(PanelId::Count)> panels_{};
    bool panels_initialized_ = false;
    bool dragging_panel_ = false;
    PanelId dragged_panel_ = PanelId::Context;
    vizgfx::Vector2f drag_grab_offset_{0.f, 0.f};

    bool resizing_panel_ = false;
    PanelId resized_panel_ = PanelId::Context;
    vizgfx::Vector2f resize_start_mouse_{0.f, 0.f};
    vizgfx::Vector2f resize_start_size_{0.f, 0.f};

    static constexpr float kPanelResizeHandle = 14.f;
    static constexpr float kPanelMinW = 220.f;
    static constexpr float kPanelMinH = 140.f;

    static constexpr float kPanelTitleH = 22.f;
    static constexpr float kPanelPad = 10.f;

    void initDefaultPanelsIfNeeded();
    void clampPanelsToWindow();
    void syncUIView();
    bool isPanelVisible(PanelId id) const;
    vizgfx::FloatRect panelRect(PanelId id) const;
    float panelHeaderHeight(PanelId id) const;
    vizgfx::FloatRect panelTitleRect(PanelId id) const;
    vizgfx::FloatRect panelContentRect(PanelId id) const;
    vizgfx::FloatRect panelResizeHandleRect(PanelId id) const;
    vizgfx::FloatRect panelCloseButtonRect(PanelId id) const;
    std::optional<PanelId> hitTestPanel(const vizgfx::Vector2f& mouse) const;
    std::optional<PanelId> hitTestPanelTitle(const vizgfx::Vector2f& mouse) const;
    std::optional<PanelId> hitTestPanelResizeHandle(const vizgfx::Vector2f& mouse) const;
    std::optional<PanelId> hitTestPanelCloseButton(const vizgfx::Vector2f& mouse) const;
    void drawPanelChrome(PanelId id);
    void setPanelContentHeight(PanelId id, float content_height);
    void drawPanelScrollbar(PanelId id);
    vizgfx::Color panelAccent(PanelId id) const;

    // Cursor
    bool cursors_loaded_ = false;
    bool cursor_ok_arrow_ = false;
    bool cursor_ok_hand_ = false;
    bool cursor_ok_cross_ = false;
    bool cursor_ok_resize_ = false;
    std::optional<vizgfx::Cursor> cursor_arrow_;
    std::optional<vizgfx::Cursor> cursor_hand_;
    std::optional<vizgfx::Cursor> cursor_cross_;
    std::optional<vizgfx::Cursor> cursor_resize_;
    enum class CursorKind { Arrow, Hand, Cross, Resize };
    CursorKind cursor_kind_ = CursorKind::Arrow;
    void initCursorsIfNeeded();
    void setCursor(CursorKind kind);

    void rebuildAllTextures();
    void renderHelpOverlay();
    void renderZoomOverlay();

    // Focus outline
    std::optional<vizgfx::FloatRect> last_dataset_rect_;
    std::optional<vizgfx::FloatRect> last_projection_rect_;
    std::optional<vizgfx::FloatRect> last_understanding_rect_;
    std::optional<vizgfx::FloatRect> last_output_rect_;
    std::vector<vizgfx::FloatRect> last_block_rects_;
    std::vector<vizgfx::FloatRect> last_generated_rects_;
    std::vector<int> last_generated_indices_;
    void renderFocusOutline();

    // Stop button (Metrics panel)
    std::optional<vizgfx::FloatRect> last_stop_button_rect_;
    std::optional<vizgfx::FloatRect> last_validation_button_rect_;
    std::optional<vizgfx::FloatRect> last_skip_button_rect_;

    // Scroll (Blocks/Layers panel)
    float blocks_scroll_y_ = 0.0f;
    float blocks_scroll_max_ = 0.0f;
    static constexpr float kBlocksScrollSpeed = 48.0f;

    // Scrollbar (slicer) cliquable pour Blocks/Layers
    bool dragging_blocks_scrollbar_ = false;
    float blocks_scroll_drag_grab_y_ = 0.0f;
    vizgfx::FloatRect last_blocks_scroll_track_rect_{};
    vizgfx::FloatRect last_blocks_scroll_thumb_rect_{};

    // Scroll vertical commun aux panneaux hors Blocks/Layers.
    std::array<float, static_cast<size_t>(PanelId::Count)> panel_scroll_y_{};
    std::array<float, static_cast<size_t>(PanelId::Count)> panel_scroll_max_{};
    std::array<vizgfx::FloatRect, static_cast<size_t>(PanelId::Count)> panel_scroll_track_rects_{};
    std::array<vizgfx::FloatRect, static_cast<size_t>(PanelId::Count)> panel_scroll_thumb_rects_{};
    bool dragging_panel_scrollbar_ = false;
    PanelId scrolled_panel_ = PanelId::Context;
    float panel_scroll_drag_grab_y_ = 0.0f;
    static constexpr float kPanelScrollSpeed = 48.0f;

    // Refresh auto textures (évite le reload UI)
    uint64_t last_auto_texture_refresh_ms_ = 0;
    bool first_texture_sync_done_ = false;
    static constexpr uint64_t kAutoTextureRefreshPeriodMs = 2000;
    uint64_t last_block_texture_rebuild_ms_ = 0;

    // Evite d'ecrire le CSV a chaque update de metriques.
    uint64_t last_loss_log_flush_ms_ = 0;
    bool pending_loss_log_flush_ = false;

    // Architecture awareness (optional)
    void maybeLoadArchitecture();

    // UI layout settings (persisted to JSON)
    json serializeUILayout() const;
    bool applyUILayout(const json& layout);
    bool loadUISettings(json& out) const;
    bool saveUISettings(const json& s) const;
    void saveUILayoutToLast();
    void saveUILayoutToSlot(int slot);
    void loadUILayoutFromSlot(int slot);

    bool save_chord_armed_ = false;
    uint64_t save_chord_armed_ms_ = 0;
    bool save_chord_consumed_ = false;

    // Capture PNG demandee par la touche E. Le rendu est effectue dans update()
    // afin de rester sur le thread qui possede le contexte graphique.
    bool snapshot_requested_ = false;
    std::string last_snapshot_path_;
    bool saveSnapshotPng();

    // Mode d'affichage des Blocks/Layers : true = heatmap colorée, false = couleurs naturelles
    bool heatmap_mode_ = true;
    enum class HeatmapPalette { Classic = 0, Turbo = 1, Inferno = 2, Viridis = 3 };
    HeatmapPalette heatmap_palette_ = HeatmapPalette::Classic;
    void rebuildLayerBlockTextures();

    std::stringstream log_history;
};

#else

// Version headless (MIMIR_VIZ_BACKEND=NONE) : API compatible, pas de rendu.
class Visualizer {
public:
    explicit Visualizer(const json&) {}
    ~Visualizer() = default;

    bool initialize() { return false; }
    bool isOpen() const { return false; }
    void processEvents() {}
    void configureScript(const json&) { throw std::runtime_error("Viz backend NONE"); }
    json pollScriptEvents() { return json::array(); }
    void update() {}

    void shutdown() {}

    void requestResync() {}
    bool consumeResyncRequested() { return false; }

    void requestStopTraining() {}
    bool stopTrainingRequested() const { return false; }
    bool consumeStopTrainingRequested() { return false; }
    void setTrainingPauseControl(std::shared_ptr<std::atomic<bool>>) {}
    void setMetricsHistory(const std::vector<MetricsRecord>&, size_t = 0) {}
    void setRuntimeConfiguration(std::shared_ptr<LiveModelConfig> config) {
        std::atomic_store(&runtime_config_, std::move(config));
    }
    std::shared_ptr<LiveModelConfig> runtime_config_ = std::make_shared<LiveModelConfig>();
    RuntimeConfigEditor config_editor_;
    void setSkipConnectionControl(std::shared_ptr<SkipConnectionControl> control) {
        std::atomic_store(&skip_control_, std::move(control));
    }
    std::shared_ptr<SkipConnectionControl> skip_control_ = std::make_shared<SkipConnectionControl>();
    void updateRuntimeValidationEnabled(bool) {}
    void applyValidationControl(bool) {}
    bool validationEnabledSnapshot() const { return false; }
    uint64_t validationControlVersion() const { return 0; }

    struct LiveTrainParams {
        bool overrides_enabled = false;
        float lr = 0.0f;
        int lr_warmup_steps = 0;
        float kl_beta = 0.0f;
        int kl_warmup_steps = 0;
        bool kl_enabled = false;
        std::string recon_loss = "mse";
        uint64_t version = 0;
    };
    uint64_t liveTrainParamsVersion() const { return 0; }
    LiveTrainParams liveTrainParamsSnapshot() const { return {}; }
    void applyLiveTrainParams(const LiveTrainParams&) {}
    enum class ValidationFeedbackIcon { None = 0, Reward = 1, Penalty = 2 };
    void updateRuntimeTrainParams(float, int, float, int, const std::string&) {}
    void showValidationFeedback(ValidationFeedbackIcon) {}

    void addGeneratedImage(const std::vector<uint8_t>&, int, int, int, const std::string&, bool = false) {}
    void beginValidationImages() {}

    void setDatasetImage(const std::vector<uint8_t>&, int, int, int, const std::string&, bool = false, const DatasetBBoxOverlay::Boxes& = {}) {}
    void setDatasetText(const std::string&, const std::string&, const std::string&, const std::string&) {}
    void setProjectionImage(const std::vector<uint8_t>&, int, int, int, const std::string&) {}
    void setUnderstandingImage(const std::vector<uint8_t>&, int, int, int, const std::string&) {}

    struct Detection {
        float x1 = 0.0f;
        float y1 = 0.0f;
        float x2 = 0.0f;
        float y2 = 0.0f;
        float score = -1.0f;
        int class_id = -1;
        std::string label;
    };
    enum class DetectionTarget { Dataset, Output, Both };
    void setDetections(const std::vector<Detection>&, DetectionTarget) {}
    void setDetectionOverlayEnabled(bool) {}
    void setDetectionScoreThreshold(float) {}

    struct DetectionMetrics {
        bool enabled = false;
        float object_loss = 0.0f;
        float box_loss = 0.0f;
        float class_loss = 0.0f;
        int positive_cells = 0;
        bool validation_has = false;
        float precision = 0.0f;
        float recall = 0.0f;
        float f1 = 0.0f;
        float mean_iou = 0.0f;
        int true_positive = 0;
        int false_positive = 0;
        int false_negative = 0;
    };

    struct BlockFrame {
        std::vector<uint8_t> pixels;
        std::vector<uint8_t> pixels_real;
        int w = 0;
        int h = 0;
        int channels = 1;
        std::string label;
        int heatmap_kind = 0; // 0: image/scalar, 1: signed mean + energy
        std::string tensor_info;
        std::vector<float> tensor_values; // interleaved raw samples, before colour mapping
        std::vector<size_t> tensor_indices;
        int tensor_sample_channels = 0;
    };
    void setLayerBlockImages(const std::vector<BlockFrame>&) {}

    void updateMetrics(int, int, float, float, float = 0.0f,
                      float = 0.0f, float = 0.0f, float = 0.0f,
                      float = 0.0f, float = 0.0f, float = 0.0f,
                      float = 0.0f,
                      int = 0, int = 0, float = 0.0f,
                      int = 0, size_t = 0, size_t = 0, float = 0.0f, size_t = 0,
                      float = 0.0f, float = 0.0f,
                      int = 0, int = 0,
                      float = 0.0f, float = 0.0f,
                      float = 0.0f, float = 0.0f,
                      bool = false, bool = false, int = 0, int = 0,
                      float = 0.0f, float = 0.0f, float = 0.0f,
                      const std::string& = std::string(),
                      bool = false,
                      int = 0,
                      int = 0,
                      float = 0.0f,
                      const std::string& = std::string(),
                      const DetectionMetrics* = nullptr) {}

    void addLossPoint(float) {}
    void setLossLogFile(const std::string&) {}
    void clearImages() {}
    void setEnabled(bool) {}
    bool isEnabled() const { return false; }
    void saveLossHistory(const std::string&) const {}
    void setLossLogEnabled(bool) {}
};

#endif

#endif // __TENSOR_VISUALIZER_HPP__
