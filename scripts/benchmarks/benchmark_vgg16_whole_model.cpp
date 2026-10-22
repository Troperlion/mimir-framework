// Native VGG training latency, with identical inputs/weights across runs.
// Arguments: image side (128), measured steps (10), base channels (16).
// Build: cmake --build build --target mimir_bench_vgg16_whole_model
// CPU: OMP_NUM_THREADS=10 MIMIR_DISABLE_VULKAN=1 MIMIR_DISABLE_OPENCL=1
//      MIMIR_DISABLE_CUDA=1 MIMIR_DISABLE_ROCM=1
//      ./build/Tests/mimir_bench_vgg16_whole_model 512 5 16
#include "Model.hpp"
#include "Models/Registry/ModelArchitectures.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
int main(int argc,char** argv){try{
    const int side=argc>1?std::stoi(argv[1]):128,steps=argc>2?std::stoi(argv[2]):10,channels=argc>3?std::stoi(argv[3]):16;
    if(side<32 || side>512 || steps<1 || steps>1000 || channels<4 || channels>32)throw std::invalid_argument("side=32..512 steps=1..1000 channels=4..32");
    Model::setFrameworkLogsSuppressed(true);
    auto cfg=ModelArchitectures::defaultConfig("vgg16_feat");
    cfg["image_w"]=side;cfg["image_h"]=side;cfg["base_channels"]=channels;cfg["enc_norm"]="groupnorm";cfg["enc_gn_groups"]=32;
    auto model=ModelArchitectures::create("vgg16_feat",cfg);model->allocateParams();model->initializeWeights("xavier",1337);
    std::vector<float> x(size_t(side)*side*3);for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i%997)*.031f);
    Optimizer opt;opt.type=OptimizerType::LION;opt.weight_decay=1e-6f;
    using Clock=std::chrono::steady_clock;
    std::vector<double> forward,backward,optimizer,total;
    double loss=0,cold_trainstep_ms=0;
    for(int s=-2;s<steps;++s){
        const auto begin=Clock::now();model->zeroGradients();const auto& y=model->forwardPassView(x,true);const auto f=Clock::now();
        if(y.empty())throw std::runtime_error("empty VGG output");
        loss=0;std::vector<float> dy(y.size());for(size_t i=0;i<y.size();++i){const float d=y[i]-.1f;loss+=double(d)*d/y.size();dy[i]=2*d/y.size();}
        if(!std::isfinite(loss))throw std::runtime_error("nonfinite loss");
        model->backwardPass(dy);const auto b=Clock::now();model->optimizerStep(opt,1e-5f);const auto end=Clock::now();
        auto ms=[](auto a,auto z){return std::chrono::duration<double,std::milli>(z-a).count();};
        if(s==-2)cold_trainstep_ms=ms(begin,end);
        if(s>=0){forward.push_back(ms(begin,f));backward.push_back(ms(f,b));optimizer.push_back(ms(b,end));total.push_back(ms(begin,end));}
    }
    auto stats=[](std::vector<double> v){std::sort(v.begin(),v.end());return json{{"median_ms",v[v.size()/2]},{"p95_ms",v[std::min(v.size()-1,v.size()*95/100)]}};};
    double checksum=0;for(const auto& l:model->getLayers())for(size_t i=0;i<l.getWeightsSize();++i)checksum+=double(i+1)*l.getWeights()[i];
    std::cout<<json{{"architecture","vgg16_feat"},{"side",side},{"base_channels",channels},{"steps",steps},{"optimizer","Lion"},{"cold_trainstep_ms",cold_trainstep_ms},{"forward",stats(forward)},{"backward_and_loss",stats(backward)},{"optimizer_step",stats(optimizer)},{"total",stats(total)},{"samples_ms",{{"forward",forward},{"backward_and_loss",backward},{"optimizer_step",optimizer},{"total",total}}},{"final_loss",loss},{"parameter_checksum",checksum}}.dump()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
