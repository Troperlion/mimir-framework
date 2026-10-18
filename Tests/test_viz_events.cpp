#include "test_utils.hpp"
#include "Visualizer.hpp"
#include "viz/AnnotationFeatureOverlay.hpp"
#include "viz/FeatureProbe.hpp"
#include "viz/LumenOutputRole.hpp"
#include <deque>
#include <cstdlib>
#include <filesystem>
#include <chrono>
namespace fs = std::filesystem;
#undef TASSERT_TRUE
#undef TASSERT_NEAR
#define TASSERT_TRUE(x) do { if (!(x)) throw std::runtime_error("FAIL: " #x); } while (0)
#define TASSERT_NEAR(a,b,e) TASSERT_TRUE(std::abs((a)-(b)) <= (e))

#ifdef ENABLE_VIZ
class EventWindow : public VizWindow {
    vizgfx::RenderTexture canvas_{{1400,1800}};
public:
    std::deque<vizgfx::Event> events;
    bool open=true;
    bool isOpen() const override {return open;}
    void close() override {open=false;}
    bool setActive(bool=true) override {return true;}
    void setIcon(const vizgfx::Image&) override {}
    void setFramerateLimit(unsigned) override {}
    void setSize(vizgfx::Vector2u size) override {(void)canvas_.resize(size);}
    vizgfx::Vector2u getSize() const override {return canvas_.getSize();}
    void setView(const vizgfx::View& v) override {canvas_.setView(v);}
    const vizgfx::View& getView() const override {return canvas_.getView();}
    vizgfx::Vector2f mapPixelToCoords(vizgfx::Vector2i p) const override {return {float(p.x),float(p.y)};}
    void setMouseCursor(const vizgfx::Cursor&,int=0) override {}
    bool isKeyPressed(vizgfx::Keyboard::Key) const override {return false;}
    std::optional<vizgfx::Event> pollEvent() override {
        if(events.empty()) return {};
        auto e=events.front();events.pop_front();
        if(const auto* resized=e.getIf<vizgfx::Event::Resized>()) (void)canvas_.resize(resized->size);
        return e;
    }
    void clear(const vizgfx::Color& c=vizgfx::Color::Black) override {canvas_.clear(c);}
    void display() override {canvas_.display();}
    vizgfx::Image captureImage() const override {return canvas_.getTexture().copyToImage();}
protected:
    vizgfx::RenderTarget& renderTarget() override {return canvas_;}
};
struct VisualizerEventTestAccess {
    static void run() {
        Visualizer v(json{{"visualization",{{"enabled",true},{"window_width",1400},{"window_height",1800}}}});
        v.setLossLogEnabled(false);
        auto window=std::make_unique<EventWindow>(); auto* events=window.get();v.window=std::move(window);
        TASSERT_TRUE(v.font.openFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"));v.font_loaded=true;
        v.initDefaultPanelsIfNeeded();
        using P=Visualizer::PanelId;
        v.panels_[size_t(P::Blocks)].pos={10,400};v.panels_[size_t(P::Blocks)].size={760,400};
        v.panels_[size_t(P::Context)].size={420,280};
        auto redraw=[&] {v.syncUIView();v.drawPanelChrome(P::Blocks);v.drawPanelChrome(P::Context);v.drawPanelChrome(P::Output);v.renderMetrics();};
        redraw();
        auto check_commands=[&] {
            TASSERT_TRUE(v.last_commands_rect_.has_value());
            const std::optional<vizgfx::FloatRect>* buttons[]={&v.last_debug_json_button_rect_,&v.last_pause_button_rect_,&v.last_stop_button_rect_,
                &v.last_validation_button_rect_,&v.last_skip_button_rect_,&v.last_live_overrides_box_,
                &v.last_live_kl_enable_box_,&v.last_live_recon_loss_box_};
            for (auto button:buttons) {
                TASSERT_TRUE(button->has_value());
                const auto r=**button;
                TASSERT_TRUE(v.last_commands_rect_->contains(r.position));
                TASSERT_TRUE(v.last_commands_rect_->contains({r.position.x+r.size.x-0.1f,r.position.y+r.size.y-0.1f}));
            }
            for (size_t i=0;i<8;++i) for (size_t j=i+1;j<8;++j)
                TASSERT_TRUE(!( **buttons[i]).findIntersection(**buttons[j]).has_value());
            TASSERT_TRUE(v.last_pause_button_rect_->size.y>=40.f);
            TASSERT_TRUE(v.last_pause_button_rect_->position.y<v.last_stop_button_rect_->position.y);
        };
        check_commands();
        const auto metrics_panel=v.panels_[size_t(P::Metrics)];
        v.panels_[size_t(P::Metrics)].size.x=290.f;v.renderMetrics();check_commands();
        v.panels_[size_t(P::Metrics)]=metrics_panel;redraw();
        if (const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear(vizgfx::Color(12,14,18));redraw();
            v.window->display();TASSERT_TRUE(v.window->captureImage().saveToFile(capture));
        }
        auto key=[&](vizgfx::Keyboard::Key k) {events->events.emplace_back(vizgfx::Event::KeyPressed{k});v.processEvents();};
        auto click=[&](const std::optional<vizgfx::FloatRect>& r) {
            TASSERT_TRUE(r.has_value()); const auto& b=*r;
            events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Left,
                {int(b.position.x+b.size.x/2),int(b.position.y+b.size.y/2)}});
            v.processEvents();
        };
        auto debug_control = std::atomic_load(&v.runtime_config_);
        key(vizgfx::Keyboard::Key::D);
        TASSERT_TRUE(v.debug_json_dialog_.visible);
        TASSERT_TRUE(!debug_control->consumeDebugJson());
        v.debug_json_dialog_.directory = "/tmp/mimir-debugjson-dialog-test";
        v.debug_json_dialog_.filename = "chosen.json";
        key(vizgfx::Keyboard::Key::Tab);
        TASSERT_TRUE(v.debug_json_dialog_.selected == 1);
        key(vizgfx::Keyboard::Key::Enter);
        TASSERT_TRUE(!v.debug_json_dialog_.visible);
        std::string chosen_path;
        TASSERT_TRUE(debug_control->consumeDebugJson(&chosen_path));
        TASSERT_TRUE(chosen_path == "/tmp/mimir-debugjson-dialog-test/chosen.json");
        TASSERT_TRUE(!debug_control->requestDebugJson());
        debug_control->finishDebugJson("DebugJSON test");
        click(v.last_debug_json_button_rect_);
        TASSERT_TRUE(v.debug_json_dialog_.visible);
        v.renderDebugJsonDialog();
        key(vizgfx::Keyboard::Key::Escape);
        TASSERT_TRUE(!v.debug_json_dialog_.visible);
        TASSERT_TRUE(!debug_control->consumeDebugJson());
        TASSERT_TRUE(!v.stopTrainingRequested());
        TASSERT_TRUE(!std::atomic_load(&v.training_paused_)->load());
        const bool heat=v.heatmap_mode_;click(v.last_blocks_mode_box_);TASSERT_TRUE(v.heatmap_mode_!=heat);
        const bool smooth=v.smooth_layer_block_previews_;click(v.last_blocks_smooth_box_);TASSERT_TRUE(v.smooth_layer_block_previews_!=smooth);
        const auto palette=v.heatmap_palette_;click(v.last_blocks_palette_box_);TASSERT_TRUE(v.heatmap_palette_!=palette);
        const bool act=v.hide_activation_blocks;click(v.last_blocks_hide_act_box_);TASSERT_TRUE(v.hide_activation_blocks!=act && v.consumeResyncRequested());
        const bool norm=v.hide_normalisation_blocks;click(v.last_blocks_hide_norm_box_);TASSERT_TRUE(v.hide_normalisation_blocks!=norm && v.consumeResyncRequested());
        // Native layer captures and legacy previews obey both filters immediately.
        std::vector<Visualizer::BlockFrame> filter_frames;
        auto frame_for=[&](const std::string& label) {
            Visualizer::BlockFrame frame;frame.w=frame.h=1;frame.channels=1;frame.pixels={120};frame.label=label;
            filter_frames.push_back(std::move(frame));
        };
        const std::vector<std::string> activation_types={"ReLU","LeakyReLU","GELU","GEGLU","SiLU","Tanh","Sigmoid", "Softmax","LogSoftmax","Softplus","Mish","HardSigmoid","HardSwish","Swish"};
        const std::vector<std::string> norm_types={"BatchNorm1d","BatchNorm2d","LayerNorm","GroupNorm","InstanceNorm2d","RMSNorm"};
        for(const auto& type:activation_types) frame_for("model/blocks/model/opaque/"+type+"/vec");
        for(const auto& type:norm_types) frame_for("model/blocks/model/opaque/"+type+"/vec");
        frame_for("model/component/activation");frame_for("model/component/norm1");
        frame_for("model/input/dataset/rgb");
        frame_for("model/blocks/model/norm_projection/Conv2d");
        frame_for("model/blocks/model/activation_features/Linear");
        const size_t remaining=3;
        v.setLayerBlockImages(filter_frames);TASSERT_TRUE(v.layer_block_images.size()==remaining);
        TASSERT_TRUE(v.last_block_frames_.size()==filter_frames.size());
        click(v.last_blocks_hide_act_box_);TASSERT_TRUE(!v.hide_activation_blocks);
        TASSERT_TRUE(v.layer_block_images.size()==remaining+activation_types.size()+1);
        click(v.last_blocks_hide_norm_box_);TASSERT_TRUE(!v.hide_normalisation_blocks);
        TASSERT_TRUE(v.layer_block_images.size()==filter_frames.size());
        click(v.last_blocks_hide_act_box_);TASSERT_TRUE(v.layer_block_images.size()==remaining+norm_types.size()+1);
        v.setLayerBlockImages(filter_frames);TASSERT_TRUE(v.layer_block_images.size()==remaining+norm_types.size()+1);
        click(v.last_blocks_hide_norm_box_);TASSERT_TRUE(v.layer_block_images.size()==remaining);
        v.setLayerBlockImages({});
        click(v.last_blocks_tips_limit_box_);TASSERT_TRUE(v.blocks_tips_limit_editing_);
        events->events.emplace_back(vizgfx::Event::TextEntered{U'2'});v.processEvents();key(vizgfx::Keyboard::Key::Enter);TASSERT_TRUE(v.blocks_tips_visible_limit_==2);
        const bool detection=v.show_detection_overlay_;click(v.last_detection_toggle_box_);TASSERT_TRUE(v.show_detection_overlay_!=detection);
        click(v.last_detection_threshold_track_);
        TASSERT_NEAR(v.detection_score_threshold_,0.5f,0.01f);
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,{0,0}});v.processEvents();
        auto pause=std::make_shared<std::atomic<bool>>(false); v.setTrainingPauseControl(pause);
        click(v.last_pause_button_rect_);TASSERT_TRUE(pause->load());
        click(v.last_pause_button_rect_);TASSERT_TRUE(!pause->load());
        const bool validation=v.validationEnabledSnapshot();click(v.last_validation_button_rect_);TASSERT_TRUE(v.validationEnabledSnapshot()!=validation);
        v.skip_control_->available=true;const bool skips=v.skip_control_->enabled;click(v.last_skip_button_rect_);TASSERT_TRUE(v.skip_control_->enabled!=skips);
        click(v.last_live_overrides_box_);TASSERT_TRUE(v.liveTrainParamsSnapshot().overrides_enabled);
        const bool kl=v.live_ui_kl_enabled_;click(v.last_live_kl_enable_box_);TASSERT_TRUE(v.liveTrainParamsSnapshot().kl_enabled!=kl);
        const int recon=v.live_ui_recon_loss_index_;click(v.last_live_recon_loss_box_);TASSERT_TRUE(v.live_ui_recon_loss_index_==(recon+1)%6);
        const std::optional<vizgfx::FloatRect>* inputs[]={&v.last_live_lr_value_box_,&v.last_live_lrwu_value_box_,&v.last_live_klb_value_box_,&v.last_live_klwu_value_box_};
        for(auto input:inputs) {click(*input);TASSERT_TRUE(v.live_input_target_!=Visualizer::LiveInputTarget::None);key(vizgfx::Keyboard::Key::Escape);}
        const std::optional<vizgfx::FloatRect>* tracks[]={&v.last_live_lr_track_,&v.last_live_lrwu_track_,&v.last_live_klb_track_,&v.last_live_klwu_track_};
        for(auto track:tracks) {const auto version=v.liveTrainParamsVersion();click(*track);TASSERT_TRUE(v.liveTrainParamsVersion()>version);events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,{0,0}});v.processEvents();}
        // The correspondence is shown beside the residual and retains raw feature inspection.
        json annotation={{"entities",json::array({
            {{"id","left"},{"role","primary_subject"},{"bbox",{.05,.1,.45,.9}},
             {"skeleton",{{"name","custom"},{"points",json::array({{{"name","hand"},{"position",{.3,.5}},{"parent",nullptr},{"entities",{"right"}}}})}}}},
            {{"id","right"},{"role","object"},{"bbox",{.6,.15,.95,.85}}}
        })},{"relations",json::array({{{"subject","left"},{"predicate","holds"},{"object","right"}}})}};
        Visualizer::BlockFrame feature;feature.w=32;feature.h=16;feature.channels=1;feature.label="vae_conv/recon";
        feature.pixels.assign(512,90);feature.tensor_sample_channels=2;
        for(int y=0;y<16;++y)for(int x=0;x<32;++x){feature.tensor_values.push_back(x<16?1.f:0.f);feature.tensor_values.push_back(x<16?0.f:1.f);}
        auto residual=feature;residual.label="vae_conv/err/resdiff_abs";residual.tensor_values.clear();residual.tensor_sample_channels=0;
        std::vector<Visualizer::BlockFrame> annotation_frames={feature,residual};
        AnnotationFeatureOverlay::decorate(annotation_frames,annotation,json{{"bbox_loss_weight",1.f}},32,16);
        v.blocks_tips_visible_limit_=0;v.setLayerBlockImages(annotation_frames);v.renderLayerBlocks();
        TASSERT_TRUE(v.layer_block_images.size()==4 && v.last_block_rects_.size()==4);
        TASSERT_NEAR(v.last_block_rects_[1].position.y,v.last_block_rects_[2].position.y,1.f);
        TASSERT_TRUE(v.last_block_rects_[2].position.x>v.last_block_rects_[1].position.x);
        if(const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear();v.drawPanelChrome(P::Blocks);v.renderLayerBlocks();v.window->display();
            TASSERT_TRUE(v.window->captureImage().saveToFile(std::string(capture)+"-annotations.png"));
        }
        // F enables the probe on the actual hovered spatial tap, preserving raw data.
        v.focus_target_=Visualizer::FocusTarget::LayerBlock;v.focus_block_index_=0;
        v.zoom_active_=true;v.renderZoomOverlay();
        const auto feature_rect=*v.picker_image_rect_;
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int(feature_rect.position.x+feature_rect.size.x*.25f),int(feature_rect.position.y+feature_rect.size.y*.5f)}});
        key(vizgfx::Keyboard::Key::F);v.renderZoomOverlay();
        TASSERT_TRUE(v.show_feature_probe_);
        TASSERT_TRUE(v.last_picker_text_.find("Section similaire : 256 cellules")!=std::string::npos);
        TASSERT_TRUE(v.last_picker_text_.find("Canal 0 : 1")!=std::string::npos);
        TASSERT_TRUE(v.last_block_frames_[0].tensor_values==feature.tensor_values);
        const auto section=FeatureProbe::analyze(feature.tensor_values,32,16,2,8,8);
        TASSERT_TRUE(section.cells.size()==256 && section.channels[0].first==0);
        TASSERT_TRUE(FeatureProbe::analyze({0,0,0,0},2,1,2,0,0).cells.empty());
        TASSERT_TRUE(FeatureProbe::analyze({1,0,-1,0,1,0},3,1,2,0,0).cells.size()==1);
        if(const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear();v.renderZoomOverlay();v.window->display();
            TASSERT_TRUE(v.window->captureImage().saveToFile(std::string(capture)+"-features.png"));
        }
        key(vizgfx::Keyboard::Key::F);v.renderZoomOverlay();
        TASSERT_TRUE(!v.show_feature_probe_ && v.last_picker_text_.find("Section similaire")==std::string::npos);
        v.blocks_tips_visible_limit_=2;
        Visualizer::BlockFrame sampled;
        sampled.w=sampled.h=2;sampled.channels=1;sampled.label="lumen/blocks/lumen/text/token_embedding/Embedding/vec";
        sampled.pixels={0,0,0,0};sampled.tensor_values={-4.f,12.5f,0.f,8.f};sampled.tensor_indices={10,20,30,40};sampled.tensor_sample_channels=1;
        v.setLayerBlockImages({sampled});
        TASSERT_TRUE(v.has_projection_image && v.has_understanding_image);
        v.focus_target_=Visualizer::FocusTarget::LayerBlock;v.focus_block_index_=0;
        const float sidebar=std::clamp(float(v.window_width)*0.40f,360.f,520.f);
        const float iw=std::max(120.f,float(v.window_width)-56.f-22.f-sidebar);
        const float ih=std::max(120.f,float(v.window_height)-56.f);
        const float box=std::min(iw,ih);
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int(28.f+(iw-box)*0.5f+box*0.75f),int(28.f+(ih-box)*0.5f+box*0.25f)}});
        v.processEvents();v.renderZoomOverlay();
        TASSERT_TRUE(v.last_picker_text_.find("[20] = 12.5")!=std::string::npos);
        TASSERT_TRUE(v.last_picker_rect_.has_value());
        if (const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear();v.renderZoomOverlay();v.window->display();
            TASSERT_TRUE(v.window->captureImage().saveToFile(std::string(capture)+"-picker.png"));
        }
        TASSERT_NEAR(v.last_picker_rect_->size.x,box/2.f,0.01f);
        TASSERT_NEAR(v.last_picker_rect_->position.x,28.f+(iw-box)*0.5f+box/2.f,0.01f);
        v.zoom_actual_size_=true;
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int(28.f+(iw-2.f)*0.5f+1.f),int(28.f+(ih-2.f)*0.5f)}});
        v.processEvents();v.renderZoomOverlay();
        TASSERT_TRUE(v.last_picker_rect_.has_value());TASSERT_NEAR(v.last_picker_rect_->size.x,1.f,0.01f);
        v.zoom_actual_size_=false;
        events->events.emplace_back(vizgfx::Event::MouseMoved{{0,0}});v.processEvents();v.renderZoomOverlay();
        TASSERT_TRUE(!v.last_picker_rect_.has_value());
        // Drag in reverse: inspect all four raw values and keep the capture stable.
        v.zoom_active_=true;
        const auto image_rect=*v.picker_image_rect_;
        auto point=[&](float nx,float ny) {return vizgfx::Vector2i{
            int(image_rect.position.x+image_rect.size.x*nx),int(image_rect.position.y+image_rect.size.y*ny)};};
        events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Left,point(0.75f,0.75f)});
        events->events.emplace_back(vizgfx::Event::MouseMoved{point(0.25f,0.25f)});
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,point(0.25f,0.25f)});
        v.processEvents();v.renderZoomOverlay();
        TASSERT_TRUE(v.picker_selection_active_ && !v.picker_selecting_ && v.picker_selection_rect_);
        const auto& selected=v.picker_selection_summary_;
        TASSERT_TRUE(selected.values.size()==4 && selected.finite==4 && selected.zeros==1);
        TASSERT_NEAR(selected.min,-4.f,1e-6f);TASSERT_NEAR(selected.max,12.5f,1e-6f);
        TASSERT_NEAR(selected.mean,4.125,1e-6f);TASSERT_NEAR(selected.l2,std::sqrt(236.25),1e-6f);
        if(const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear();v.renderZoomOverlay();v.window->display();
            TASSERT_TRUE(v.window->captureImage().saveToFile(std::string(capture)+"-selection.png"));
        }
        sampled.tensor_values={100.f,100.f,100.f,100.f};v.setLayerBlockImages({sampled});v.renderZoomOverlay();
        TASSERT_NEAR(v.picker_selection_summary_.mean,4.125,1e-6f);
        key(vizgfx::Keyboard::Key::Escape);TASSERT_TRUE(!v.picker_selection_active_ && v.zoom_active_);
        // New selection samples the latest capture; right click clears it.
        events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Left,point(0.25f,0.25f)});
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,point(0.25f,0.25f)});
        v.processEvents();TASSERT_NEAR(v.picker_selection_summary_.mean,100.f,1e-6f);
        events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Right,point(0.25f,0.25f)});
        v.processEvents();TASSERT_TRUE(!v.picker_selection_active_);
        // A small viewport still permits scrolling through all selected values.
        const int saved_height=v.window_height;v.window_height=360;v.renderZoomOverlay();
        const auto small=*v.picker_image_clip_;
        const vizgfx::Vector2i center{int(small.position.x+small.size.x/2),int(small.position.y+small.size.y/2)};
        events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Left,center});
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,center});
        v.processEvents();v.renderZoomOverlay();TASSERT_TRUE(v.zoom_info_scroll_max_>0.f);
        events->events.emplace_back(vizgfx::Event::MouseWheelScrolled{vizgfx::Mouse::Wheel::Vertical,-2.f,center});
        v.processEvents();TASSERT_TRUE(v.zoom_info_scroll_>0.f);
        key(vizgfx::Keyboard::Key::T);v.renderZoomOverlay();
        TASSERT_TRUE(v.picker_selection_rect_ && v.picker_selection_rect_->size.x==1.f);
        TASSERT_NEAR(v.picker_selection_summary_.mean,100.f,1e-6f);
        key(vizgfx::Keyboard::Key::T);
        events->events.emplace_back(vizgfx::Event::MouseButtonPressed{vizgfx::Mouse::Button::Right,center});
        v.processEvents();v.window_height=saved_height;v.renderZoomOverlay();
        v.zoom_active_=false;
        // Statistics ignore repeated source indices and count non-finite samples separately.
        const auto exceptional=TensorSelection::analyze({0.f,0.f,std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),-2.f,4.f},{0,0,1,2,3,4},3,2,1,-20,-20,20,20);
        TASSERT_TRUE(exceptional.values.size()==5 && exceptional.finite==3 && exceptional.zeros==1);
        TASSERT_TRUE(exceptional.nan==1 && exceptional.inf==1);
        TASSERT_NEAR(exceptional.mean,2.0/3.0,1e-6f);
        TASSERT_TRUE(TensorSelection::analyze({}, {},0,0,0,0,0,1,1).values.empty());
        MetricsRecord old{};old.loss=3.5f;MetricsRecord next{};next.loss=0.25f;
        v.setMetricsHistory({old,next});TASSERT_TRUE(v.full_loss_history.size()==2 && v.loss_history.size()==2);
        TASSERT_NEAR(v.loss_min_,0.25f,1e-6f);TASSERT_NEAR(v.loss_max_,3.5f,1e-6f);
        MetricsRecord tail{};tail.loss=0.125f;
        v.full_loss_history.push_back(next); // Provisional updateMetrics row.
        v.setMetricsHistory({tail},2);
        TASSERT_TRUE(v.full_loss_history.size()==3 && v.loss_history.size()==3);
        TASSERT_NEAR(v.full_loss_history.back().loss,0.125f,1e-6f);
        v.setMetricsHistory({},3);
        TASSERT_TRUE(v.full_loss_history.size()==3 && v.loss_history.size()==3);
        TASSERT_NEAR(v.loss_min_,0.125f,1e-6f);
        v.setLayerBlockImages({});TASSERT_TRUE(!v.has_projection_image && !v.has_understanding_image);
        // Input preview selection must choose the image subsequently enlarged by Z.
        v.setDatasetImage(std::vector<uint8_t>(8*4*3,128),8,4,3,"input");
        v.renderContextImages();click(v.last_dataset_rect_);
        TASSERT_TRUE(v.focus_target_==Visualizer::FocusTarget::Dataset);
        v.setDetections({{.1f,.1f,.9f,.9f,.9f,2,"main"},{0,0,1,1,.1f,3,"rejected"}},Visualizer::DetectionTarget::Dataset);
        v.zoom_active_=true;v.renderZoomOverlay();
        const auto detected_rect=*v.picker_image_rect_;
        events->events.emplace_back(vizgfx::Event::MouseMoved{{int(detected_rect.position.x+detected_rect.size.x*.5f),int(detected_rect.position.y+detected_rect.size.y*.5f)}});
        key(vizgfx::Keyboard::Key::F);v.renderZoomOverlay();
        TASSERT_TRUE(v.last_picker_text_.find("Detecte : main 90.0%")!=std::string::npos);
        TASSERT_TRUE(v.last_picker_text_.find("rejected")==std::string::npos);
        v.setDetections({{.1f,.1f,.9f,.9f,.9f,2,"wrong target"}},Visualizer::DetectionTarget::Output);
        v.renderZoomOverlay();TASSERT_TRUE(v.last_picker_text_.find("wrong target")==std::string::npos);
        key(vizgfx::Keyboard::Key::F);v.zoom_active_=false;
        key(vizgfx::Keyboard::Key::Z);click(v.last_stop_button_rect_);TASSERT_TRUE(!v.consumeStopTrainingRequested());key(vizgfx::Keyboard::Key::Escape);
        key(vizgfx::Keyboard::Key::H);click(v.last_stop_button_rect_);TASSERT_TRUE(!v.consumeStopTrainingRequested());key(vizgfx::Keyboard::Key::H);
        auto configuration=std::make_shared<LiveModelConfig>();
        configuration->publish({{"learning_rate",0.001,LiveModelConfig::Owner::Training,0,1,{},"",std::nullopt}});
        v.setRuntimeConfiguration(configuration);
        key(vizgfx::Keyboard::Key::C);v.renderConfigurationOverlay();
        TASSERT_TRUE(v.config_editor_.visible && !v.config_row_rects_.empty());
        click(v.config_row_rects_[0].first);TASSERT_TRUE(v.config_editor_.editing);
        v.config_editor_.buffer.clear();
        for(char32_t c:std::u32string(U"0.02")) events->events.emplace_back(vizgfx::Event::TextEntered{c});
        v.processEvents();key(vizgfx::Keyboard::Key::Enter);
        TASSERT_TRUE(configuration->snapshot()[0].pending.has_value());
        TASSERT_NEAR(configuration->snapshot()[0].pending->get<float>(),0.02f,1e-6f);
        key(vizgfx::Keyboard::Key::C);
        configuration->publish({{"bbox_loss_weight",1.0,LiveModelConfig::Owner::Model,0,1e6,{},"",std::nullopt}});
        key(vizgfx::Keyboard::Key::C);v.renderConfigurationOverlay();
        const auto rows=configuration->snapshot();
        for(const auto& row:v.config_row_rects_) if(rows[row.second].key=="bbox_loss_weight") click(row.first);
        TASSERT_TRUE(v.config_editor_.editing_key=="bbox_loss_weight");
        v.config_editor_.buffer.clear();events->events.emplace_back(vizgfx::Event::TextEntered{U'2'});
        v.processEvents();key(vizgfx::Keyboard::Key::Enter);
        bool bbox_requested=false;
        for(const auto& row:configuration->snapshot()) if(row.key=="bbox_loss_weight") bbox_requested=row.pending && *row.pending==2;
        TASSERT_TRUE(bbox_requested);key(vizgfx::Keyboard::Key::C);
        for(const auto* role_key : {"relation_primary_weight","relation_secondary_weight","relation_other_weight"}) {
            configuration->publish({{role_key,1.0,LiveModelConfig::Owner::Model,0,1e6,{},"",std::nullopt}});
            key(vizgfx::Keyboard::Key::C);v.renderConfigurationOverlay();
            const auto role_rows=configuration->snapshot();
            for(const auto& row:v.config_row_rects_) if(role_rows[row.second].key==role_key) click(row.first);
            TASSERT_TRUE(v.config_editor_.editing_key==role_key);
            v.config_editor_.buffer.clear();events->events.emplace_back(vizgfx::Event::TextEntered{U'5'});
            v.processEvents();key(vizgfx::Keyboard::Key::Enter);
            bool requested=false;
            for(const auto& row:configuration->snapshot()) if(row.key==role_key) requested=row.pending && *row.pending==5;
            TASSERT_TRUE(requested);key(vizgfx::Keyboard::Key::C);
        }
        // Panel title and resize handle initiate and release their intended drag.
        click(vizgfx::FloatRect{v.panels_[size_t(P::Blocks)].pos,{40,20}});
        TASSERT_TRUE(v.dragging_panel_);
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,{0,0}});v.processEvents();
        TASSERT_TRUE(!v.dragging_panel_);
        v.panels_[size_t(P::Generated)].visible=false;
        click(v.panelResizeHandleRect(P::Blocks));TASSERT_TRUE(v.resizing_panel_);
        events->events.emplace_back(vizgfx::Event::MouseButtonReleased{vizgfx::Mouse::Button::Left,{0,0}});v.processEvents();
        TASSERT_TRUE(!v.resizing_panel_);v.panels_[size_t(P::Generated)].visible=true;
        // A covered button cannot change the underlying panel.
        auto& graph=v.panels_[size_t(P::Graph)];const auto old_graph=graph;
        graph.pos=v.last_stop_button_rect_->position;graph.size={200,100};
        click(v.last_stop_button_rect_);TASSERT_TRUE(!v.consumeStopTrainingRequested());graph=old_graph;
        click(v.last_stop_button_rect_);TASSERT_TRUE(v.consumeStopTrainingRequested());TASSERT_TRUE(!v.consumeStopTrainingRequested());
        for(P p:{P::Context,P::Output,P::Blocks,P::Generated,P::Training,P::Metrics,P::Graph}) {
            click(v.panelCloseButtonRect(p));TASSERT_TRUE(!v.isPanelVisible(p));
        }
        // Semantic sections ignore the model prefix and the word "output" in time projections.
        using Mimir::Viz::lumenTipSection;
        TASSERT_TRUE(lumenTipSection("lumen_diffusion/blocks/lumen/text/token_embedding/Embedding") == "Text / Cond");
        TASSERT_TRUE(lumenTipSection("lumen/blocks/lumen/time/output/Linear") == "Diffusion");
        TASSERT_TRUE(lumenTipSection("lumen_diffusion/blocks/lumen/dit/block1/cross_attention/CrossAttention") == "Diffusion");
        TASSERT_TRUE(lumenTipSection("lumen_diffusion/blocks/lumen/output/z/Identity | reconstructed image") == "Outputs");
        TASSERT_TRUE(lumenTipSection("diffusion/compare/C_model_decode_z0_pred | timestep=20") == "Validation");
        TASSERT_TRUE(lumenTipSection("resdiff_abs") == "Validation");
        TASSERT_TRUE(lumenTipSection("diffusion/reference") == "Validation");
        TASSERT_TRUE(lumenTipSection("diffusion_out") == "Validation");
        TASSERT_TRUE(lumenTipSection("vae_conv/err/resdiff_abs").empty());
        const auto blocks_before_sections = v.panels_[size_t(P::Blocks)];
        const bool hide_norm_before_sections = v.hide_normalisation_blocks;
        v.hide_normalisation_blocks = false;
        v.panels_[size_t(P::Blocks)].size = {760, 1100};
        v.blocks_tips_visible_limit_ = 0;
        v.blocks_scroll_y_ = v.blocks_scroll_max_ = 0;
        std::vector<Visualizer::BlockFrame> section_frames;
        for (const auto* label : {
            "lumen_diffusion/blocks/lumen/text/token_embedding/Embedding",
            "lumen_diffusion/blocks/lumen/time/output/Linear",
            "lumen_diffusion/blocks/lumen/output/z/Identity | role=model_output tensor=z",
            "diffusion/compare/C_model_decode_z0_pred",
            "resdiff_abs",
            "lumen_diffusion/blocks/lumen/text/final_norm/LayerNorm",
            "lumen_diffusion/blocks/lumen/dit/block1/cross_attention/CrossAttention",
            "diffusion/reference"}) {
            Visualizer::BlockFrame frame;
            frame.w = frame.h = 2;
            frame.channels = 3;
            frame.pixels.assign(12, 42);
            frame.label = label;
            section_frames.push_back(std::move(frame));
        }
        v.setLayerBlockImages(section_frames);
        v.renderLayerBlocks();
        TASSERT_TRUE(v.last_block_rects_.size() == section_frames.size());
        const auto y_of = [&](size_t i) { return v.last_block_rects_[i].position.y; };
        TASSERT_NEAR(y_of(0), y_of(5), 1.f);
        TASSERT_NEAR(y_of(1), y_of(6), 1.f);
        TASSERT_NEAR(y_of(3), y_of(4), 1.f);
        TASSERT_NEAR(y_of(3), y_of(7), 1.f);
        TASSERT_TRUE(y_of(0) < y_of(1) && y_of(1) < y_of(2) && y_of(2) < y_of(3));
        v.panels_[size_t(P::Blocks)] = blocks_before_sections;
        v.hide_normalisation_blocks = hide_norm_before_sections;
        // Only the explicit reconstructed z may own [ST], even when a time
        // projection or an older decoded comparison comes later in the list.
        Visualizer::BlockFrame reconstructed_z;
        reconstructed_z.w = reconstructed_z.h = 2;
        reconstructed_z.channels = 3;
        reconstructed_z.pixels.assign(12, 42);
        reconstructed_z.label = "lumen_diffusion/blocks/lumen/output/z/Identity | role=model_output tensor=z";
        auto time_output = reconstructed_z;
        time_output.pixels.assign(12, 99);
        time_output.label = "lumen_diffusion/blocks/lumen/time/output/Linear";
        auto comparison_output = time_output;
        comparison_output.label = "diffusion_out";
        v.setLayerBlockImages({reconstructed_z, time_output, comparison_output});
        TASSERT_TRUE(v.output_image_.pixels == reconstructed_z.pixels);
        TASSERT_TRUE(v.output_thumb_label_ == reconstructed_z.label);
        v.setLayerBlockImages({comparison_output});
        TASSERT_TRUE(v.output_image_.pixels == reconstructed_z.pixels);
        TASSERT_TRUE(v.output_thumb_label_ == reconstructed_z.label);
        // Validation pairs outlive graph/training updates and the ordinary image cap.
        v.clearImages();
        v.image_grid_cols=1;v.image_grid_rows=1;
        v.addGeneratedImage(std::vector<uint8_t>(12,7),2,2,3,"completed training step");
        const auto completed_output=v.output_image_.pixels;
        v.beginValidationImages();
        for (int sample=0;sample<3;++sample) {
            v.addGeneratedImage(std::vector<uint8_t>(12,10+sample),2,2,3,"VAL target " + std::to_string(sample),true);
            v.addGeneratedImage(std::vector<uint8_t>(12,20+sample),2,2,3,"VAL recon " + std::to_string(sample),true);
        }
        TASSERT_TRUE(v.output_image_.pixels==completed_output);
        for (int step=0;step<8;++step) {
            v.addGeneratedImage(std::vector<uint8_t>(12,90),2,2,3,"training/graph " + std::to_string(step));
            v.setLayerBlockImages({sampled});
        }
        TASSERT_TRUE(v.validation_image_count_==6 && v.generated_images.size()==7);
        for (int sample=0;sample<3;++sample) {
            TASSERT_TRUE(v.generated_images[2*sample].pixels.front()==10+sample);
            TASSERT_TRUE(v.generated_images[2*sample+1].pixels.front()==20+sample);
        }
        v.rebuildAllTextures();
        TASSERT_TRUE(v.validation_image_count_==6 && v.generated_images[0].pixels.front()==10);
        v.renderGeneratedImages();TASSERT_TRUE(v.last_generated_indices_.size()==7);
        v.beginValidationImages();TASSERT_TRUE(v.validation_image_count_==0 && v.generated_images.size()==1);
        v.addGeneratedImage(std::vector<uint8_t>(12,30),2,2,3,"VAL target next",true);
        v.addGeneratedImage(std::vector<uint8_t>(12,40),2,2,3,"VAL recon next",true);
        TASSERT_TRUE(v.generated_images[0].pixels.front()==30 && v.generated_images[1].pixels.front()==40);
        v.clearImages();TASSERT_TRUE(v.generated_images.empty() && v.validation_image_count_==0);

        // Last drawn script button wins when controls overlap.
        v.configureScript({{"controls",{{{"id","under"},{"x",10},{"y",10}},{{"id","above"},{"x",10},{"y",10}}}}});
        click(vizgfx::FloatRect{{10,10},{150,36}});
        const auto log=v.pollScriptEvents();TASSERT_TRUE(!log.empty() && log.back().value("id",std::string())=="above");
        // Help keeps fixed readable text, adapts columns and captures its input.
        key(vizgfx::Keyboard::Key::H);TASSERT_TRUE(v.show_help_overlay_);
        v.renderHelpOverlay();TASSERT_TRUE(v.help_columns_==2 && !v.help_row_rects_.empty());
        auto verify_help=[&] {
            TASSERT_TRUE(v.help_body_rect_->position.x>=0.f && v.help_body_rect_->position.y>=0.f);
            TASSERT_TRUE(v.help_body_rect_->position.x+v.help_body_rect_->size.x<=v.window_width);
            TASSERT_TRUE(v.help_body_rect_->position.y+v.help_body_rect_->size.y<=v.window_height);
            for(size_t i=0;i<v.help_row_rects_.size();++i) for(size_t j=i+1;j<v.help_row_rects_.size();++j)
                TASSERT_TRUE(!v.help_row_rects_[i].findIntersection(v.help_row_rects_[j]));
        };
        verify_help();
        if(const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear();v.renderHelpOverlay();v.window->display();
            TASSERT_TRUE(v.window->captureImage().saveToFile(std::string(capture)+"-help-wide.png"));
        }
        const auto focus=v.focus_target_;
        key(vizgfx::Keyboard::Key::F1);key(vizgfx::Keyboard::Key::C);
        TASSERT_TRUE(v.focus_target_==focus && !v.config_editor_.visible);
        events->events.emplace_back(vizgfx::Event::Resized{{520,400}});v.processEvents();v.renderHelpOverlay();
        TASSERT_TRUE(v.help_columns_==1 && v.help_scroll_max_>0.f);verify_help();
        if(const char* capture=std::getenv("MIMIR_VIZ_TEST_CAPTURE")) {
            v.window->clear();v.renderHelpOverlay();v.window->display();
            TASSERT_TRUE(v.window->captureImage().saveToFile(std::string(capture)+"-help-narrow.png"));
        }
        events->events.emplace_back(vizgfx::Event::MouseWheelScrolled{vizgfx::Mouse::Wheel::Vertical,-2.f,{260,200}});
        v.processEvents();TASSERT_TRUE(v.help_scroll_y_>0.f);
        const float scroll=v.help_scroll_y_;key(vizgfx::Keyboard::Key::Down);TASSERT_TRUE(v.help_scroll_y_>scroll);
        key(vizgfx::Keyboard::Key::Escape);TASSERT_TRUE(!v.show_help_overlay_);
        key(vizgfx::Keyboard::Key::H);TASSERT_TRUE(v.show_help_overlay_ && v.help_scroll_y_==0.f);
        v.renderHelpOverlay();click(v.help_close_rect_);TASSERT_TRUE(!v.show_help_overlay_);
        // The graph is modal and receives immutable metadata from the model.
        auto graph_config=std::atomic_load(&v.runtime_config_);
        graph_config->publishModelGraph(json{{"mode","desactive"},{"planned",false},
            {"nodes",json::array({json{{"index",0},{"name","test"},{"type","Linear"},
                {"lane",0},{"depth",0},{"runtime","non planifie"},{"parameters",4},
                {"inputs",json::array({"x"})},{"output","y"},{"capability","inconnue"}}})},
            {"edges",json::array()},{"external_inputs",json::array()}});
        auto graph_sample=*graph_config->modelGraph();
        auto second=graph_sample["nodes"][0];
        second["index"]=1;second["name"]="activation";second["type"]="ReLU";
        second["depth"]=1;second["inputs"]=json::array({"y"});second["output"]="z";
        graph_sample["nodes"].push_back(second);
        graph_sample["edges"].push_back(json{{"from",0},{"to",1},{"tensor","y"}});
        graph_config->publishModelGraph(std::move(graph_sample));
        key(vizgfx::Keyboard::Key::N);
        TASSERT_TRUE(v.show_model_graph_ && graph_config->modelGraphRequested());
        v.setRuntimeConfiguration(graph_config);
        TASSERT_TRUE(graph_config->modelGraphRequested());
        v.renderModelGraphOverlay();
        TASSERT_TRUE(v.model_graph_hits_.size()==2);
        click(v.model_graph_hits_[1].first);
        TASSERT_TRUE(v.model_graph_selected_==1);
        v.renderModelGraphOverlay();
        if(const char* capture=std::getenv("MIMIR_GRAPH_TEST_CAPTURE")) {
            events->display();TASSERT_TRUE(events->captureImage().saveToFile(capture));
        }
        key(vizgfx::Keyboard::Key::Down);
        TASSERT_TRUE(v.model_graph_y_>0.f);
        key(vizgfx::Keyboard::Key::R);TASSERT_TRUE(v.model_graph_y_==0.f);
        key(vizgfx::Keyboard::Key::Escape);
        TASSERT_TRUE(!v.show_model_graph_ && !graph_config->modelGraphRequested());
        v.shutdown();
    }
};
#endif
int main() {
#ifndef ENABLE_VIZ
    return 77;
#else
#ifdef MIMIR_VIZ_BACKEND_SFML
    if(!std::getenv("DISPLAY")) return 77;
#endif
    const auto dir=fs::temp_directory_path() / ("mimir_viz_events_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto previous=fs::current_path();fs::create_directories(dir);fs::current_path(dir);
    struct Cleanup {fs::path previous,dir;~Cleanup(){fs::current_path(previous);fs::remove_all(dir);}} cleanup{previous,dir};
    try { VisualizerEventTestAccess::run(); }
    catch(const std::exception& error) {std::cerr << error.what() << "\n"; return 1;}
    std::cout << "PASS VIZ button events, modal isolation and overlap routing\n";
#endif
}
