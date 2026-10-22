#include "test_utils.hpp"
#include "Model.hpp"
#include "AsyncMonitor.hpp"
#include <thread>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>

struct AsyncMonitorPreviewTestAccess {
    static void open(AsyncMonitor& monitor) {
        monitor.htop_ = std::make_shared<HtopDisplay>(-1);
        monitor.htop_->setRuntimeConfiguration(std::atomic_load(&monitor.runtime_config_));
        monitor.processHtopKey('D');
    }
    static void key(AsyncMonitor& monitor, char key) { monitor.processHtopKey(key); }
    static bool visible(AsyncMonitor& monitor) { return monitor.htop_->debugJsonVisible(); }
};

static size_t index_of(const std::shared_ptr<LiveModelConfig>& config, const std::string& key) {
    const auto rows = config->snapshot();
    for (size_t i=0; i<rows.size(); ++i) if(rows[i].key==key) return i;
    throw std::runtime_error("missing row " + key);
}
int main(int argc, char** argv) {
    Model model;
    model.modelConfig = {{"type","basic_mlp"},{"bbox_loss_weight",1.0},{"hidden_dim",16},{"training",{{"epochs",10}}}};
    model.push("linear", "Linear", 1);
    auto& layer = model.getMutableLayers()[0];
    layer.inputs={"__input__"}; layer.output="x";
    layer.in_features=layer.out_features=1; layer.use_bias=false;
    model.allocateParams(); layer.getWeights()[0]=1;
    // UI requests do not touch model data; the next forward services the export.
    auto export_control = model.runtimeConfiguration();
    AsyncMonitor export_monitor;
    export_monitor.bindRuntimeConfiguration(export_control);
    AsyncMonitorPreviewTestAccess::open(export_monitor);
    TASSERT_TRUE(AsyncMonitorPreviewTestAccess::visible(export_monitor));
    TASSERT_TRUE(export_control->debugJsonStatus().empty());
    model.forwardPass(std::vector<float>{2}, true); // Training continues during input.
    for (char c : std::string("checkpoints/debug/custom folder")) AsyncMonitorPreviewTestAccess::key(export_monitor,c);
    AsyncMonitorPreviewTestAccess::key(export_monitor,'\t');
    for (char c : std::string("live_test.json")) AsyncMonitorPreviewTestAccess::key(export_monitor,c);
    AsyncMonitorPreviewTestAccess::key(export_monitor,'\n');
    TASSERT_TRUE(!AsyncMonitorPreviewTestAccess::visible(export_monitor));
    TASSERT_TRUE(!export_control->requestDebugJson());
    TASSERT_TRUE(export_control->debugJsonStatus() == "DebugJSON en attente");
    model.forwardPass(std::vector<float>{2}, true);
    const auto export_status = export_control->debugJsonStatus();
    TASSERT_TRUE(export_status.find("DebugJSON : ") == 0);
    const auto export_path = export_status.substr(std::string("DebugJSON : ").size());
    TASSERT_TRUE(export_path == "checkpoints/debug/custom folder/live_test.json");
    std::ifstream debug_file(export_path);
    nlohmann::json debug_dump;
    debug_file >> debug_dump;
    TASSERT_TRUE(debug_dump["format"] == "mimir_debug_dump");
    debug_file.close();
    TASSERT_TRUE(layer.getWeights()[0] == 1);
    model.forwardPass(std::vector<float>{3}, true);
    TASSERT_TRUE(export_control->debugJsonStatus() == export_status);
    std::filesystem::remove(export_path);

    Optimizer opt;
    opt.type=OptimizerType::ADAMW; opt.decay_strategy=LRDecayStrategy::NONE;
    opt.initial_lr=0.1f; opt.weight_decay=0;
    model.publishRuntimeConfiguration(&opt);
    auto config=model.runtimeConfiguration();
    // Repeated publication may reuse unchanged metadata, but direct edits to
    // public model/optimizer state must be visible on the very next publication.
    model.publishRuntimeConfiguration(&opt);
    model.modelConfig["training"]["epochs"] = 12;
    opt.beta1 = 0.8f;
    model.publishRuntimeConfiguration(&opt);
    bool saw_epochs = false, saw_beta = false;
    for (const auto& entry : config->snapshot()) {
        if (entry.key == "training/epochs") saw_epochs = entry.value == 12;
        if (entry.key == "beta1") saw_beta = std::abs(entry.value.get<float>() - 0.8f) < 1e-6f;
    }
    TASSERT_TRUE(saw_epochs && saw_beta);

    TASSERT_TRUE(config->modelGraph()!=nullptr);
    config->requestModelGraph(true);
    model.forwardPass(std::vector<float>{3},true);
    const auto graph_snapshot=config->modelGraph();
    TASSERT_TRUE(graph_snapshot->at("nodes").size()==model.getLayers().size());
    TASSERT_TRUE(graph_snapshot->at("mode")!="en attente");
    config->requestModelGraph(false);

    std::string error;
    TASSERT_TRUE(!config->request("hidden_dim","32",error));
    TASSERT_TRUE(!config->request("training/epochs","2",error));
    TASSERT_TRUE(!config->request("unknown","2",error));
    TASSERT_TRUE(config->snapshot()[index_of(config,"bbox_loss_weight")].owner==LiveModelConfig::Owner::Model);
    TASSERT_TRUE(!config->request("bbox_loss_weight","-1",error));
    TASSERT_TRUE(!config->request("bbox_loss_weight","nan",error));
    TASSERT_TRUE(!config->request("bbox_loss_weight","true",error));
    TASSERT_TRUE(config->request("bbox_loss_weight","0",error));
    for(const auto& key : {"relation_primary_weight","relation_secondary_weight","relation_other_weight"}) {
        TASSERT_TRUE(config->snapshot()[index_of(config,key)].owner==LiveModelConfig::Owner::Model);
        TASSERT_TRUE(!config->request(key,"-1",error));
        TASSERT_TRUE(!config->request(key,"nan",error));
        TASSERT_TRUE(config->request(key,"4",error));
    }
    TASSERT_TRUE(model.modelConfig["bbox_loss_weight"]==1.0);

    TASSERT_TRUE(!config->request("grad_clip_norm","nan",error));
    TASSERT_TRUE(!config->request("grad_clip_norm","-1",error));
    TASSERT_TRUE(!config->request("beta1","1",error));
    TASSERT_TRUE(!config->request("epsilon","0",error));
    TASSERT_TRUE(config->request("weight_decay","0.5",error));
    TASSERT_TRUE(config->request("grad_clip_norm","1",error));
    TASSERT_TRUE(!model.modelConfig.contains("grad_clip_norm"));
    // Forward consumes model changes; optimizer changes wait for its boundary.
    model.forwardPass(std::vector<float>{100},true);
    TASSERT_TRUE(model.modelConfig["grad_clip_norm"]==1);
    TASSERT_TRUE(model.modelConfig["bbox_loss_weight"]==0);
    TASSERT_TRUE(model.modelConfig["relation_primary_weight"]==4);
    TASSERT_TRUE(model.modelConfig["relation_secondary_weight"]==4);
    TASSERT_TRUE(model.modelConfig["relation_other_weight"]==4);
    TASSERT_TRUE(opt.weight_decay==0);
    model.backwardPass({1});
    model.optimizerStep(opt,0.1f,nullptr);
    TASSERT_NEAR(opt.weight_decay,0.5f,1e-6f);
    TASSERT_NEAR(layer.getWeights()[0],0.85f,1e-5f);
    TASSERT_TRUE(model.modelConfig["hidden_dim"]==16);

    AsyncMonitor monitor;
    monitor.bindRuntimeConfiguration(config);
    monitor.updateRuntimeTrainParams(0.001f,0,0,0,"mse");
#ifdef ENABLE_VIZ
    if(argc>1 && std::string(argv[1])=="--viz") {
        Visualizer viz(json{{"visualization", {{"enabled",true},{"window_width",1000},
            {"window_height",720},{"window_title","Mimir configuration test"},{"fps_limit",30}}}});
        viz.setLossLogEnabled(false);
        viz.setRuntimeConfiguration(config);
        TASSERT_TRUE(viz.initialize());
        bool applied=false;
        std::cout << "CONFIG_READY " << index_of(config,"learning_rate") << std::endl;
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        while(viz.isOpen() && std::chrono::steady_clock::now()<deadline) {
            viz.processEvents(); viz.update();
            const auto live=monitor.liveTrainParamsSnapshot();
            if(live.overrides_enabled && std::abs(live.lr-0.02f)<1e-6f) {applied=true;break;}
        }
        viz.shutdown();
        TASSERT_TRUE(applied);
        std::cout << "CONFIG_VIZ_APPLIED" << std::endl;
        return 0;
    }
#endif
    if(argc>1 && std::string(argv[1])=="--htop") {
        // PTY integration: C opens, arrows choose learning_rate, Enter edits,
        // typing 0.02 + Enter submits, and the training thread acknowledges.
        monitor.configureMetricsCsv("",false);
        monitor.start(true,false);
        bool applied=false;
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
        while(std::chrono::steady_clock::now()<deadline) {
            const auto live=monitor.liveTrainParamsSnapshot();
            if(live.overrides_enabled && std::abs(live.lr-0.02f)<1e-6f) {applied=true;break;}
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        monitor.stop();
        TASSERT_TRUE(applied);
        return 0;
    }
    RuntimeConfigEditor editor;
    TASSERT_TRUE(!editor.visible);
    editor.toggle();
    editor.selected=index_of(config,"learning_rate");
    editor.enter(*config);
    TASSERT_TRUE(editor.editing);
    for(char c:std::string("0.02")) editor.text(c);
    editor.enter(*config);
    TASSERT_TRUE(!editor.editing);
    TASSERT_TRUE(config->snapshot()[index_of(config,"learning_rate")].pending.has_value());
    const auto live=monitor.liveTrainParamsSnapshot();
    TASSERT_TRUE(live.overrides_enabled);
    TASSERT_NEAR(live.lr,0.02f,1e-7f);
    monitor.updateRuntimeTrainParams(0.0001f,0,0,0,"mse");
    TASSERT_NEAR(monitor.liveTrainParamsSnapshot().lr,0.02f,1e-7f);
    TASSERT_TRUE(!config->snapshot()[index_of(config,"learning_rate")].pending.has_value());
    TASSERT_TRUE(!config->request("lr_warmup_steps","2.5",error));
    TASSERT_TRUE(!config->request("validation_enabled","1",error));
    TASSERT_TRUE(config->request("validation_enabled","true",error));
    TASSERT_TRUE(monitor.validationEnabledSnapshot());
    editor.enter(*config);
    editor.text('7');
    editor.cancel();
    TASSERT_NEAR(monitor.liveTrainParamsSnapshot().lr,0.02f,1e-7f);
    editor.cancel();
    TASSERT_TRUE(!editor.visible);

    // Publishing a runtime snapshot cannot discard an unconsumed user edit.
    TASSERT_TRUE(config->request("beta2","0.95",error));
    model.publishRuntimeConfiguration(&opt);
    model.applyRuntimeOptimizerConfiguration(opt);
    TASSERT_NEAR(opt.beta2,0.95f,1e-7f);
    TASSERT_TRUE(!config->snapshot()[index_of(config,"beta2")].pending.has_value());
    Model dropout;
    dropout.push("dropout", "Dropout", 0);
    dropout.getMutableLayers()[0].inputs={"__input__"};
    dropout.allocateParams();
    dropout.publishRuntimeConfiguration();
    TASSERT_TRUE(dropout.runtimeConfiguration()->request("dropout","0",error));
    TASSERT_TRUE(dropout.forwardPass(std::vector<float>{1,2,3},true)==std::vector<float>({1,2,3}));
    TASSERT_TRUE(dropout.getLayers()[0].dropout_p==0);
    return 0;
}
