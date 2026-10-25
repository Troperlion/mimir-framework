// Native public Model API counterpart of colab_real_worker.py.
#include "Model.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cmath>
#include <numeric>
#include <sstream>
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
static double rss_mib(){
    std::ifstream f("/proc/self/status");std::string line;
    while(std::getline(f,line))if(line.rfind("VmRSS:",0)==0){std::istringstream value(line.substr(6));double kb=0;value>>kb;return kb/1024.;}
    return 0.;
}
static double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
void framework_log_write_file_only(const char*,size_t){}
void framework_log_write(const char*,size_t){}
template<class T>static std::vector<T> read(const std::string& path){
    std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("Missing file: "+path);
    const auto size=f.tellg();if(size<0 || size%sizeof(T))throw std::runtime_error("Invalid file: "+path);
    std::vector<T> v(static_cast<size_t>(size)/sizeof(T));f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),size);
    if(!f)throw std::runtime_error("Cannot read: "+path);return v;
}
int main(int argc,char** argv){
 if(argc!=4)return 2;
 const auto whole=Clock::now();Json report={{"framework","mimir"},{"backend",argv[1]},{"status","error"}};
 try{
    Json c;std::ifstream(argv[2])>>c;const std::string dir=c.at("data_dir"),backend=argv[1];
    const int batch=c.at("batch"),hidden=c.at("hidden"),steps=c.at("steps");const float lr=c.at("lr");
    if(batch<=0 || hidden<=0 || steps<=0)throw std::runtime_error("Invalid dimensions");
    // Use the production planner and its supported optimizations, rather than
    // forcing every layer onto the requested accelerator independently.
    const Json options=c.value("mimir",Json::object());
    const std::string planner=options.value("planner_mode",std::string("cost"));
    if(planner!="cost" && planner!="static" && planner!="legacy")throw std::runtime_error("Invalid planner mode");
    auto flag=[&](const char* environment,const char* option,bool fallback){
        setenv(environment,options.value(option,fallback)?"1":"0",1);
    };
    const bool allow_cpu=options.value("allow_cpu_fallback",true);
    setenv("MIMIR_DISABLE_CPU",backend=="cpu" || allow_cpu?"0":"1",1);
    setenv("MIMIR_ENABLE_PLANNER","1",1);setenv("MIMIR_PLANNER_MODE",planner.c_str(),1);
    setenv("MIMIR_PLANNER_COST_MODEL","1",1);
    flag("MIMIR_PLANNER_FUSION","fusion",true);flag("MIMIR_ENABLE_FUSION","fusion",true);
    flag("MIMIR_ENABLE_FUSION_TRAIN","fusion_train",true);
    flag("MIMIR_PLANNER_BUFFER_REUSE","buffer_reuse",true);
    flag("MIMIR_PLANNER_DEVICE_RESIDENCY","device_residency",true);
    report["optimizations_requested"]={{"planner_mode",planner},{"fusion",options.value("fusion",true)},
        {"fusion_train",options.value("fusion_train",true)},{"buffer_reuse",options.value("buffer_reuse",true)},
        {"device_residency",options.value("device_residency",true)},{"allow_cpu_fallback",allow_cpu}};
    Model::setFrameworkLogsSuppressed(true);
    const auto data_started=Clock::now();
    std::vector<std::vector<float>> initial;
    for(int i=0;i<4;++i)initial.push_back(read<float>(dir+"/weights-"+std::to_string(hidden)+"-"+std::to_string(i)+".f32"));
    struct Data{std::vector<uint8_t> x,y;};std::map<std::string,Data> data;
    for(const std::string split:{"train","valid","test"}){
        auto& d=data[split];d.x=read<uint8_t>(dir+"/"+split+".images.u8");d.y=read<uint8_t>(dir+"/"+split+".labels.u8");
        if(d.y.empty() || d.x.size()!=d.y.size()*784)throw std::runtime_error("Dataset shape mismatch");
    }
    report["data_allocation_s"]=seconds(data_started);
    const auto initialization=Clock::now();
    Model model;model.modelConfig={{"loss","mse"},{"grad_clip_norm",0}};
    model.push("hidden","Linear",hidden*784+hidden);
    model.push("relu","ReLU",0);model.push("logits","Linear",10*hidden+10);
    auto& layers=model.getMutableLayers();
    layers[0].inputs={"__input__"};layers[0].output="hidden";
    layers[0].in_features=784;layers[0].out_features=hidden;layers[0].seq_len=batch;layers[0].use_bias=true;
    layers[1].inputs={"hidden"};layers[1].output="activation";
    layers[2].inputs={"activation"};layers[2].output="x";
    layers[2].in_features=hidden;layers[2].out_features=10;layers[2].seq_len=batch;layers[2].use_bias=true;
    const auto allocation=Clock::now();model.allocateParams();
    report["parameter_allocation_s"]=seconds(allocation);
    Optimizer opt;const std::string optimizer=c.value("optimizer",std::string("sgd"));
    if(optimizer!="sgd" && optimizer!="adam" && optimizer!="adamw")throw std::runtime_error("Invalid optimizer");
    opt.type=optimizer=="sgd"?OptimizerType::SGD:optimizer=="adam"?OptimizerType::ADAM:OptimizerType::ADAMW;opt.initial_lr=lr;opt.min_lr=lr;
    opt.decay_strategy=LRDecayStrategy::NONE;opt.weight_decay=c.value("weight_decay",0.f);opt.warmup_steps=0;
    auto reset=[&]{
        for(int block=0;block<2;++block){auto& l=layers[block==0?0:2];
            if(l.getWeightsSize()!=initial[2*block].size()+initial[2*block+1].size())throw std::runtime_error("Parameter shape mismatch");
            std::copy(initial[2*block].begin(),initial[2*block].end(),l.getWeights());
            std::copy(initial[2*block+1].begin(),initial[2*block+1].end(),l.getWeights()+initial[2*block].size());
        }opt.step=0;opt.mv_by_param_ptr.clear();opt.m.clear();opt.v.clear();
    };
    auto probe=[&]{
        std::vector<float> flat;
        for(int i:{0,2})flat.insert(flat.end(),layers[i].getWeights(),layers[i].getWeights()+layers[i].getWeightsSize());
        std::vector<float> result;for(size_t index:c.at("probe_indices").get<std::vector<size_t>>())result.push_back(flat.at(index));return result;
    };
    std::vector<float> x(batch*784),y(batch*10);
    auto get_batch=[&](const std::string& split,size_t offset){
        const auto& d=data.at(split);std::fill(y.begin(),y.end(),0.f);
        for(int row=0;row<batch;++row){const size_t index=(offset+row)%d.y.size();
            if(d.y[index]>=10)throw std::runtime_error("Invalid label");y[row*10+d.y[index]]=1.f;
            for(int j=0;j<784;++j)x[row*784+j]=d.x[index*784+j]/255.f;
        }
    };
    auto train=[&](){
        Model::TrainStepRequest request;request.float_inputs["__input__"]=&x;request.target=&y;
        request.optimizer=&opt;request.learning_rate=lr;request.capture_prediction=false;
        auto result=model.trainStep(request);if(!result)throw std::runtime_error("Model.trainStep unavailable");
        if(!std::isfinite(result->loss))throw std::runtime_error("Non-finite loss");return *result;
    };
    report["initialization_s"]=seconds(initialization);
    report["rss_after_initialization_mib"]=rss_mib();
    // Native public model tensors reside on host; accelerator scratch is lazy
    // and therefore included in the first synchronized trainStep.
    {
        const double before=rss_mib();const auto tick=Clock::now();std::vector<float> probe_allocation(4*1024*1024,1.f);
        volatile float touched=probe_allocation.back();(void)touched;
        report["allocation_probe_s"]=seconds(tick);report["allocation_probe_bytes"]=16*1024*1024;
        report["allocation_probe_rss_delta_mib"]=rss_mib()-before;
        report["allocation_probe_device"]="host; lazy device scratch included in cold_step_ms";
    }
    reset();get_batch("train",0);auto cold=Clock::now();auto first=train();
    report["cold_step_ms"]=seconds(cold)*1000;report["first_loss"]=first.loss;
    report["first_prediction"]=model.getTensor("x");report["first_parameter_probe"]=probe();
    if(auto snapshot=model.runtimeConfiguration()->modelGraph())report["training_plan"]=*snapshot;
    reset();cold=Clock::now();model.forwardPass(x,false);report["cold_predict_ms"]=seconds(cold)*1000;
    if(auto snapshot=model.runtimeConfiguration()->modelGraph())report["inference_plan"]=*snapshot;
    // Route diagnostics are deliberately outside the timed training/evaluation.
    std::ostringstream audit;
    auto* old_cerr=std::cerr.rdbuf(audit.rdbuf());
    const char* trace=std::getenv("MIMIR_RUNTIME_TRACE");
    const bool had_trace=trace!=nullptr;
    const std::string previous_trace=trace?trace:"";
    auto restore_trace=[&]{if(had_trace)setenv("MIMIR_RUNTIME_TRACE",previous_trace.c_str(),1);else unsetenv("MIMIR_RUNTIME_TRACE");};
    try {
        setenv("MIMIR_RUNTIME_TRACE","1",1);
        Model::setFrameworkLogsSuppressed(false);
        train();model.forwardPass(x,false);
        Model::setFrameworkLogsSuppressed(true);std::cerr.rdbuf(old_cerr);restore_trace();
    } catch(...) {
        Model::setFrameworkLogsSuppressed(true);std::cerr.rdbuf(old_cerr);restore_trace();throw;
    }
    report["runtime_audit"]=audit.str();
    std::ofstream(std::string(argv[3])+".runtime-audit.log")<<audit.str();
    reset();

    auto evaluate=[&](const std::string& split){
        const auto& d=data.at(split);size_t correct=0;double mse=0;std::vector<double> samples;
        for(size_t offset=0;offset<d.y.size();offset+=batch){
            auto tick=Clock::now();get_batch(split,offset);const auto out=model.forwardPass(x,false);
            samples.push_back(seconds(tick)*1000);
            if(out.size()!=static_cast<size_t>(batch*10))throw std::runtime_error("Wrong prediction shape");
            const size_t count=std::min<size_t>(batch,d.y.size()-offset);
            for(size_t row=0;row<count;++row){
                auto begin=out.begin()+row*10;const int predicted=std::max_element(begin,begin+10)-begin;
                correct+=predicted==d.y[offset+row];
                for(int j=0;j<10;++j){if(!std::isfinite(begin[j]))throw std::runtime_error("Non-finite predictions");
                    const double diff=begin[j]-y[row*10+j];mse+=diff*diff;}
            }
        }
        return Json{{"accuracy",double(correct)/d.y.size()},{"mse",mse/(d.y.size()*10)},
            {"samples_ms",samples},{"examples",d.y.size()}};
    };
    const int eval_every=c.at("eval_every");const double target=c.at("target_accuracy");
    Json history=Json::array(),goal=nullptr;std::vector<double> samples,losses;
    auto started=Clock::now();
    for(int step=0;step<steps;++step){
        auto tick=Clock::now();get_batch("train",size_t(step)*batch);const auto result=train();
        samples.push_back(seconds(tick)*1000);losses.push_back(result.loss);
        if((step+1)%eval_every==0 || step+1==steps){
            const auto valid=evaluate("valid");const double elapsed=seconds(started);
            history.push_back({{"step",step+1},{"elapsed_s",elapsed},{"accuracy",valid.at("accuracy")},{"mse",valid.at("mse")}});
            if(goal.is_null() && valid.at("accuracy").get<double>()>=target)goal=elapsed;
            Json progress={{"status","running"},{"steps_completed",step+1},{"history",history}};
            std::ofstream(std::string(argv[3])+".progress.json")<<progress.dump();
        }
    }
    report["train_wall_s"]=seconds(started);report["train_step_samples_ms"]=samples;report["train_losses"]=losses;
    report["history"]=history;report["time_to_goal_s"]=goal;report["steps_completed"]=steps;
    report["examples_trained"]=size_t(steps)*batch;
    started=Clock::now();report["test"]=evaluate("test");report["inference_wall_s"]=seconds(started);
    report["final_parameter_probe"]=probe();report["case"]=c.at("case");report["status"]="measured";
    report["mode"]="Model.trainStep + Model.forwardPass; planner="+planner+"; production optimizations (see plans/audit)";
    report["device"]="target="+backend+(allow_cpu?"; CPU routes allowed by planner":"; CPU routes disabled");report["optimizer"]=optimizer;
    report["reconstruction_loss_runtime_counts"]=RuntimeRouter::instance().reconstructionLossDispatchCounts();
    report["optimizer_runtime_counts"]=RuntimeRouter::instance().optimizerDispatchCounts();
    report["version"]=c.value("mimir_version",std::string("see protocol"));
 }catch(const std::exception& e){report["error"]=e.what();}
 report["worker_wall_s"]=seconds(whole);std::ofstream(argv[3])<<report.dump(2);
 return 0;
}
