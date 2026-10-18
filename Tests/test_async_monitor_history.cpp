#include "test_utils.hpp"
#include "AsyncMonitor.hpp"
#include <filesystem>
#include <future>
#include <fstream>
#include <iterator>
#include <numeric>

struct AsyncMonitorPreviewTestAccess {
    static int run() {
        AsyncMonitor monitor;
        Visualizer::BlockFrame frame;
        frame.w=frame.h=1; frame.channels=1; frame.pixels={37}; frame.label="layer/first";
        monitor.setDatasetSample({11},1,1,1,"completed 0","caption 0","","","");
        monitor.setLayerBlockImages({frame});
        TASSERT_TRUE(monitor.layer_blocks_dirty_ && monitor.pending_layer_blocks_->size()==1);
        TASSERT_TRUE(monitor.context_dirty_ && monitor.pending_dataset_sample_->frame.pixels.front()==11);
        monitor.layer_blocks_dirty_=false; // The Viz has displayed this capture.
        monitor.context_dirty_=false;
        for (int step=0; step<4; ++step) {
            monitor.setDatasetSample({static_cast<uint8_t>(128+step)},1,1,1,"sample " + std::to_string(step),"prompt","","","");
            monitor.setDatasetText("next prompt","","","");
            monitor.setLayerBlockImages({}); // Capture may be disabled/skipped during this step.
            TASSERT_TRUE(!monitor.layer_blocks_dirty_ && !monitor.context_dirty_);
            TASSERT_TRUE(monitor.pending_dataset_sample_->frame.pixels.front()==11);
            TASSERT_TRUE(monitor.pending_dataset_sample_->text.raw=="caption 0");
            TASSERT_TRUE(monitor.pending_layer_blocks_->size()==1);
            TASSERT_TRUE(monitor.pending_layer_blocks_->front().pixels.front()==37);
        }
        frame.label="layer/next"; frame.pixels={91};
        monitor.setLayerBlockImages({frame});
        TASSERT_TRUE(monitor.layer_blocks_dirty_);
        TASSERT_TRUE(monitor.pending_layer_blocks_->front().pixels.front()==91);
        TASSERT_TRUE(monitor.context_dirty_ && monitor.pending_dataset_sample_->frame.pixels.front()==131);
        TASSERT_TRUE(monitor.pending_dataset_sample_->text.raw=="next prompt");
        TASSERT_TRUE(!monitor.staged_dataset_sample_);
        // N+1 can begin before the UI consumes N without changing the ready snapshot.
        monitor.setDatasetSample({255},1,1,1,"computing N+1","caption N+1","","","");
        TASSERT_TRUE(monitor.pending_dataset_sample_->frame.pixels.front()==131);
        TASSERT_TRUE(monitor.pending_layer_blocks_->front().pixels.front()==91);
        TASSERT_TRUE(monitor.staged_dataset_sample_->frame.pixels.front()==255);
        // Derived images and output are held with their input until the capture is complete.
        std::atomic_store(&monitor.viz_,std::make_shared<Visualizer>(json{{"visualization",{{"enabled",false}}}}));
        monitor.layer_blocks_dirty_=monitor.context_dirty_=false;
        monitor.stageDatasetImage({66},1,1,1,"next image");
        monitor.setUnderstandingImage({67},1,1,1,"next understanding");
        monitor.addImage({68},1,1,1,"next output");
        TASSERT_TRUE(!monitor.context_dirty_ && monitor.pending_images_.empty());
        TASSERT_TRUE(monitor.pending_dataset_sample_->frame.pixels.front()==131);
        TASSERT_TRUE(monitor.staged_understanding_image_->pixels.front()==67 && monitor.staged_images_.size()==1);
        frame.pixels={69};monitor.setLayerBlockImages({frame});
        TASSERT_TRUE(monitor.pending_dataset_sample_->frame.pixels.front()==66);
        TASSERT_TRUE(monitor.pending_understanding_image_->pixels.front()==67);
        TASSERT_TRUE(monitor.pending_images_.front().pixels.front()==68);
        TASSERT_TRUE(monitor.pending_layer_blocks_->front().pixels.front()==69);
        TASSERT_TRUE(monitor.context_dirty_ && monitor.layer_blocks_dirty_);
        // Annotation geometry and genuine spatial feature matches belong to the
        // same immutable input/output snapshot, and preserve the raw samples.
        json annotation={{"entities",json::array({
            {{"id","left"},{"phrase","left object"},{"role","primary_subject"},{"bbox",{0.,0.,.5,1.}},
             {"skeleton",{{"name","custom"},{"points",json::array({{{"name","hand"},{"position",{.2,.5}},{"parent",nullptr},{"entities",{"right"}}}})}}}},
            {{"id","right"},{"phrase","right object"},{"role","object"},{"bbox",{.5,0.,1.,1.}}}
        })},{"relations",json::array({{{"subject","left"},{"predicate","near"},{"object","right"}}})}};
        json config={{"bbox_loss_weight",2.f}};
        Visualizer::BlockFrame features;features.w=8;features.h=4;features.channels=1;
        features.label="vae_conv/blocks/vae_conv/dec/features/Conv2d";features.pixels.assign(32,30);
        features.tensor_sample_channels=2;
        for(int y=0;y<4;++y)for(int x=0;x<8;++x){features.tensor_values.push_back(x<4?1.f:0.f);features.tensor_values.push_back(x<4?0.f:1.f);}
        const auto raw=features.tensor_values;
        Visualizer::BlockFrame diff=features;diff.label="vae_conv/err/resdiff_abs";diff.tensor_values.clear();diff.tensor_sample_channels=0;
        monitor.setDatasetSample(std::vector<uint8_t>(96,80),8,4,3,"annotated","caption","","","",true,DatasetBBoxOverlay::fromAnnotation(&annotation),&annotation,&config);
        annotation["entities"][0]["id"]="changed after staging";
        monitor.setLayerBlockImages({features,diff});
        TASSERT_TRUE(monitor.pending_layer_blocks_->size()==4);
        const auto& blocks=*monitor.pending_layer_blocks_;
        TASSERT_TRUE(blocks[0].tensor_values==raw && blocks[0].channels==3);
        TASSERT_TRUE(blocks[0].pixels!=features.pixels);
        TASSERT_TRUE(blocks[1].label.find("resdiff_abs")!=std::string::npos);
        TASSERT_TRUE(blocks[2].label.find("annotation_features")!=std::string::npos);
        TASSERT_TRUE(blocks[2].tensor_values.size()==32);
        TASSERT_NEAR(blocks[2].tensor_values[9],1.f,1e-6f);
        TASSERT_TRUE(blocks[2].tensor_info.find("region_cos(left,right)=0.000")!=std::string::npos);
        TASSERT_TRUE(blocks[2].tensor_info.find("point_cos(hand->right)=0.000")!=std::string::npos);
        TASSERT_TRUE(blocks[2].tensor_info.find("changed after staging")==std::string::npos);
        TASSERT_TRUE(blocks[3].label.find("annotation_impact")!=std::string::npos);
        TASSERT_NEAR(std::accumulate(blocks[3].tensor_values.begin(),blocks[3].tensor_values.end(),0.f),32.f,1e-4f);
        // No available raw features must be stated, rather than fabricated.
        annotation["entities"][0]["id"]="left";
        monitor.setDatasetSample(std::vector<uint8_t>(96,80),8,4,3,"annotated","caption","","","",true,DatasetBBoxOverlay::fromAnnotation(&annotation),&annotation,&config);
        monitor.setLayerBlockImages({diff});
        TASSERT_TRUE(monitor.pending_layer_blocks_->size()==3);
        TASSERT_TRUE((*monitor.pending_layer_blocks_)[1].tensor_info.find("No aligned raw spatial features")!=std::string::npos);
        // Continue the existing mailbox test with its original unannotated input.
        monitor.setDatasetSample({66},1,1,1,"next image","","","","");
        frame.pixels={69};monitor.setLayerBlockImages({frame});
        // A slow consumer keeps its immutable snapshot without blocking training.
        auto displayed = monitor.takeVizUpdate();
        TASSERT_TRUE(displayed.pending_dataset_sample_->frame.pixels.front()==66);
        TASSERT_TRUE(displayed.pending_layer_blocks_->front().pixels.front()==69);
        std::promise<void> consuming, release;
        auto release_future = release.get_future();
        auto slow_consumer = std::async(std::launch::async, [&] {
            auto capture = std::move(displayed);
            consuming.set_value();
            release_future.wait();
            return capture.pending_layer_blocks_->front().pixels.front()==69 &&
                   capture.pending_dataset_sample_->frame.pixels.front()==66;
        });
        consuming.get_future().wait();
        auto producer = std::async(std::launch::async, [&] {
            for (int i=0; i<32; ++i) {
                monitor.setDatasetSample({static_cast<uint8_t>(i)},1,1,1,"input","","","","");
                auto next = frame; next.pixels={static_cast<uint8_t>(i)};
                monitor.setLayerBlockImages({std::move(next)});
            }
        });
        const bool progressed = producer.wait_for(std::chrono::seconds(1))==std::future_status::ready;
        release.set_value();
        producer.get();
        TASSERT_TRUE(slow_consumer.get() && progressed);
        auto newest = monitor.takeVizUpdate();
        TASSERT_TRUE(newest.pending_dataset_sample_->frame.pixels.front()==31);
        TASSERT_TRUE(newest.pending_layer_blocks_->front().pixels.front()==31);
        TASSERT_TRUE(!monitor.takeVizUpdate().layer_blocks_dirty_);
        return 0;
    }
};

int main() {
    TASSERT_TRUE(AsyncMonitorPreviewTestAccess::run()==0);
    const auto path = std::filesystem::temp_directory_path() / ("mimir_monitor_owned_history_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".csv");
    std::filesystem::remove(path);
    {
        AsyncMonitor monitor;
        monitor.configureMetricsCsv(path.string());
        monitor.start(false, false);
        AsyncMonitor::Metrics m;
        m.epoch=2; m.batch=3; m.loss=0.375f; m.lr=0.001f;
        monitor.updateMetrics(m);
        m.batch=4; m.loss=0.25f; monitor.updateMetrics(m);
        monitor.updateValidation(true, 4, 1, 2, false, false, 0.5f, 0.2f, 0.f);
        monitor.updateValidation(false, 4, 2, 2, true, true, 0.125f, 0.25f, 0.f, "reward");
        monitor.updateValidation(false, 4, 2, 2, true, true, 0.125f, 0.25f, 0.f, "reward");
        TASSERT_TRUE(monitor.metricsHistorySnapshot().size()==3);
        monitor.setTrainingPaused(true);
        std::promise<void> entered;
        auto blocked=std::async(std::launch::async,[&] {entered.set_value(); monitor.waitWhileTrainingPaused();});
        entered.get_future().wait();
        TASSERT_TRUE(blocked.wait_for(std::chrono::milliseconds(60))==std::future_status::timeout);
        monitor.setTrainingPaused(false);
        TASSERT_TRUE(blocked.wait_for(std::chrono::seconds(1))==std::future_status::ready);
        blocked.get();
        monitor.setTrainingPaused(true);
        auto stopping=std::async(std::launch::async,[&] {monitor.waitWhileTrainingPaused();});
        monitor.stop();
        TASSERT_TRUE(stopping.wait_for(std::chrono::seconds(1))==std::future_status::ready);
        stopping.get();
    }
    {
        AsyncMonitor resumed;
        resumed.configureMetricsCsv(path.string());
        auto history=resumed.metricsHistorySnapshot();
        TASSERT_TRUE(history.size()==3 && history.back().is_val);
        TASSERT_TRUE(resumed.metricsHistorySnapshot(2).size()==1);
        TASSERT_TRUE(resumed.metricsHistorySnapshot(3).empty());
        TASSERT_NEAR(history.front().loss,0.375f,1e-7f);
        TASSERT_TRUE(history.back().val_feedback=="reward");
        AsyncMonitor::Metrics m; m.epoch=3; m.batch=1; m.loss=0.0625f;
        resumed.updateMetrics(m);
        resumed.flushMetricsCsv();
        history=MetricsCsv::read(path.string());
        TASSERT_TRUE(history.size()==4 && history.back().step==3);
        TASSERT_NEAR(history.back().loss,0.0625f,1e-7f);
    }
    // Legacy htop header remains appendable without introducing mismatched columns.
    const auto legacy=path.string()+".legacy";
    {std::ofstream file(legacy); file << MetricsCsv::header.substr(0,MetricsCsv::header.find(",val_feedback")) << '\n';}
    MetricsRecord r{}; r.loss=0.5f; r.is_val=true; r.val_step_id=7;
    TASSERT_TRUE(MetricsCsv::append(legacy,r));
    TASSERT_TRUE(MetricsCsv::read(legacy).size()==1);
    {std::ofstream file(legacy,std::ios::app); file << "partial,row";}
    r.val_feedback="reward, quoted \"text\"";
    TASSERT_TRUE(MetricsCsv::append(legacy,r));
    TASSERT_TRUE(MetricsCsv::read(legacy).size()==2);
    const auto exported = path.string()+".export";
    TASSERT_TRUE(MetricsCsv::write(exported,{r}));
    TASSERT_TRUE(MetricsCsv::read(exported).front().val_feedback==r.val_feedback);
    std::filesystem::remove(exported);
    std::filesystem::remove(path);std::filesystem::remove(legacy);
    return 0;
}
