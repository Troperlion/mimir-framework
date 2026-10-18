#include "runtimes/tpu/TpuRuntime.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "Layers.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
using json=nlohmann::json;
static std::vector<float> values(size_t n) {
    std::vector<float> v(n);for(size_t i=0;i<n;++i)v[i]=.7f*std::sin(float(i)*1.3f)+.05f;return v;
}
int main(int argc,char** argv) {
    const bool hardware=argc==2 && std::string(argv[1])=="--tpu";
    TpuRuntime tpu;
    if(hardware && !tpu.initialize(RuntimeConfig::fromEnv("TPU"))) {
        std::cerr << "TPU hardware initialization failed: " << tpu.lastError() << '\n';return 1;
    }
    auto compare=[](const std::vector<float>& actual,const std::vector<float>& expected,const std::string& label) {
        if(actual.size()!=expected.size())throw std::runtime_error(label+" size mismatch");
        for(size_t i=0;i<actual.size();++i)
            if(!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i])>3e-5f+4e-4f*std::abs(expected[i]))
                throw std::runtime_error(label+" difference at "+std::to_string(i));
    };
    CpuRuntime cpu;cpu.initialize({});RuntimeRouter::instance().setRuntimes(nullptr,nullptr,nullptr,nullptr,&cpu);json cases=json::array();
    auto emit=[&](Layer l,std::vector<std::vector<float>> x,bool training=false) {
        if(l.weights.empty() && l.params_count)l.weights=values(l.params_count);
        std::vector<const std::vector<float>*> in;for(auto& a:x)in.push_back(&a);
        std::vector<uint8_t> mask;RuntimeForwardContext fc;fc.output_mask=&mask;
        std::vector<std::vector<float>> out;
        if(!cpu.forwardLayerWithContext(in,out,l,training,fc))throw std::runtime_error("CPU forward "+l.type);
        std::vector<std::vector<float>> go;for(auto& a:out)go.push_back(values(a.size()));
        std::vector<const std::vector<float>*> gos;for(auto& a:go)gos.push_back(&a);
        RuntimeBackwardContext bc;bc.output=&out[0];bc.output_mask=mask.empty()?nullptr:&mask;
        std::vector<std::vector<float>> gi;
        if(!cpu.backwardLayerWithContext(in,gos,gi,l,training,bc))throw std::runtime_error("CPU backward "+l.type);
        if(hardware) {
            std::vector<std::vector<float>> tout;std::vector<uint8_t> tmask;
            RuntimeForwardContext tfc;tfc.output_mask=&tmask;
            if(!tpu.forwardLayerWithContext(in,tout,l,training,tfc))throw std::runtime_error("TPU forward "+l.type);
            if(tout.size()!=out.size())throw std::runtime_error("TPU output count "+l.type);
            const bool dropout=l.type_enum==LayerType::Dropout||l.type_enum==LayerType::Dropout2d||l.type_enum==LayerType::AlphaDropout;
            const bool stochastic=training&&(dropout||l.type_enum==LayerType::Reparameterize);
            if(!stochastic)for(size_t k=0;k<out.size();++k)compare(tout[k],out[k],l.type+" forward");
            if(dropout && training) {
                if(tmask.size()!=x[0].size())throw std::runtime_error("missing TPU dropout mask");
                const float p=std::clamp(l.dropout_p,0.f,1.f),ap=-1.6732632423543773f*1.0507009873554805f;
                const float a=p<1?1/std::sqrt((1-p)*(1+p*ap*ap)):0.f;
                std::vector<float> expected(x[0].size());
                for(size_t k=0;k<expected.size();++k)expected[k]=p>=1?0.f:l.type_enum==LayerType::AlphaDropout?
                    a*(tmask[k]?x[0][k]:ap)-a*ap*p:(tmask[k]?x[0][k]/(1-p):0.f);
                compare(tout[0],expected,l.type+" sampled forward");
            }
            RuntimeBackwardContext tbc;tbc.output=&tout[0];tbc.output_mask=tmask.empty()?nullptr:&tmask;
            Layer expected=l,actual=l;expected.grad_weights.clear();expected.grad_bias.clear();actual.grad_weights.clear();actual.grad_bias.clear();
            std::vector<std::vector<float>> want,got;
            if(!cpu.backwardLayerWithContext(in,gos,want,expected,training,tbc)||
               !tpu.backwardLayerWithContext(in,gos,got,actual,training,tbc))throw std::runtime_error("TPU backward "+l.type);
            if(want.size()!=got.size())throw std::runtime_error("TPU gradient count "+l.type);
            for(size_t k=0;k<want.size();++k)compare(got[k],want[k],l.type+" backward");
            compare(actual.grad_weights,expected.grad_weights,l.type+" grad_weights");
            compare(actual.grad_bias,expected.grad_bias,l.type+" grad_bias");
            if(!tpu.backwardLayerWithContext(in,gos,got,actual,training,tbc))throw std::runtime_error("TPU accumulation "+l.type);
            for(auto& v:expected.grad_weights)v*=2;
            for(auto& v:expected.grad_bias)v*=2;
            compare(actual.grad_weights,expected.grad_weights,l.type+" accumulated weights");
            compare(actual.grad_bias,expected.grad_bias,l.type+" accumulated bias");
            // Assert actual router selection, including its planned route.
            RuntimeRouter::instance().setRuntimes(nullptr,nullptr,nullptr,nullptr,&tpu,&cpu);
            AbstractRuntime* selected=nullptr;
            if(!RuntimeRouter::instance().dispatchForwardLayer(in,tout,l,false,&selected)||selected!=&tpu)
                throw std::runtime_error("TPU route not selected "+l.type);
            RuntimeRouter::instance().setRuntimes(nullptr,nullptr,nullptr,nullptr,&cpu);
        }
        cases.push_back({{"layer",Mimir::tpuLayerDescriptor(l)},{"weights",l.weights},{"inputs",x},
            {"training",training},{"outputs",out},{"go",go},{"gi",gi},{"gw",l.grad_weights},{"gb",l.grad_bias},{"mask",mask}});
    };
    for(int raw=0;raw<int(LayerType::UNKNOWN);++raw) {
        auto t=static_cast<LayerType>(raw);if(t==LayerType::Lambda)continue;
        Layer l("fixture",t==LayerType::PatchEmbed?"PatchEmbed":t==LayerType::Reparameterize?"Reparameterize":LayerRegistry::type_to_string(t),0);
        std::vector<std::vector<float>> x={values(8)};
        l.use_bias=true;
        switch(t) {
        case LayerType::Linear:l.in_features=4;l.out_features=3;l.seq_len=2;l.params_count=15;break;
        case LayerType::Bilinear:l.in_features=2;l.out_features=3;l.embed_dim=2;l.params_count=14;x={values(4),values(6)};break;
        case LayerType::Conv2d:case LayerType::ConvTranspose2d:case LayerType::DepthwiseConv2d: {
            l.in_channels=2;l.out_channels=3;l.input_height=3;l.input_width=4;l.kernel_h=2;l.kernel_w=2;l.stride=1;l.stride_h=1;
            l.params_count=t==LayerType::DepthwiseConv2d?10:27;x={values(24)};break; }
        case LayerType::Conv1d:l.in_channels=2;l.out_channels=3;l.kernel_h=3;l.params_count=21;x={values(12)};break;
        case LayerType::Embedding:case LayerType::EmbeddingBag:l.vocab_size=5;l.embed_dim=3;l.padding_idx=0;l.params_count=15;x={{0,1,2,2,-1,4}};if(t==LayerType::EmbeddingBag)x.push_back({0,3,6});break;
        case LayerType::LayerNorm:case LayerType::RMSNorm:l.in_features=4;l.params_count=8;break;
        case LayerType::GroupNorm:case LayerType::InstanceNorm2d:case LayerType::BatchNorm2d:case LayerType::BatchNorm1d:
            l.in_channels=2;l.input_height=2;l.input_width=2;l.num_groups=2;l.params_count=4;break;
        case LayerType::GEGLU:l.seq_len=2;l.out_features=2;break;
        case LayerType::Softmax:case LayerType::LogSoftmax:l.shape={2,2,2};l.axis=1;break;
        case LayerType::MaxPool2d:case LayerType::AvgPool2d:
            l.in_channels=2;l.input_height=2;l.input_width=2;l.kernel_size=2;l.stride=1;break;
        case LayerType::MaxPool1d:case LayerType::AvgPool1d:l.in_channels=2;l.kernel_h=2;break;
        case LayerType::GlobalAvgPool2d:case LayerType::AdaptiveAvgPool2d:l.in_channels=2;l.input_height=2;l.input_width=2;break;
        case LayerType::TokenMeanPool:l.seq_len=2;l.embed_dim=4;break;
        case LayerType::Transpose:l.in_features=2;l.out_features=4;break;
        case LayerType::Permute:l.shape={2,2,2};l.permute_dims={2,0,1};break;
        case LayerType::Add:case LayerType::Subtract:case LayerType::Multiply:case LayerType::Divide:x.push_back(values(8));break;
        case LayerType::Concat:case LayerType::Stack:x.push_back(values(8));break;
        case LayerType::Split:l.split_sizes={3,5};break;
        case LayerType::Chunk:l.num_chunks=3;break;
        case LayerType::MatMul:l.in_features=2;l.out_features=4;l.embed_dim=3;x.push_back(values(12));break;
        case LayerType::BatchMatMul:l.seq_len=2;l.in_features=2;l.out_features=2;l.embed_dim=3;x.push_back(values(12));break;
        case LayerType::NMS:x={{0,0,1,1,.1f,.1f,1.1f,1.1f,2,2,3,3},{.9f,.8f,.7f},{0,0,1}};break;
        case LayerType::SelfAttention:case LayerType::MultiHeadAttention:l.seq_len=2;l.embed_dim=4;l.num_heads=2;l.causal=true;l.params_count=80;break;
        case LayerType::CrossAttention:l.embed_dim=4;l.in_features=4;l.num_heads=2;l.causal=true;l.params_count=68;x.push_back(values(12));break;
        case LayerType::UpsampleNearest:case LayerType::UpsampleBilinear:case LayerType::UpsampleBicubic:
            l.in_channels=2;l.input_height=2;l.input_width=2;l.output_height=4;l.output_width=4;l.scale_h=2;l.scale_w=2;break;
        case LayerType::PixelShuffle:l.in_channels=8;l.input_height=1;l.input_width=1;l.scale_h=2;break;
        case LayerType::ZeroPad2d:case LayerType::ReflectionPad2d:case LayerType::ReplicationPad2d:
            l.in_channels=2;l.input_height=2;l.input_width=2;l.pad_h=1;l.pad_w=t==LayerType::ZeroPad2d?2:1;break;
        case LayerType::RNN:case LayerType::GRU:case LayerType::LSTM: {
            l.seq_len=4;l.in_features=2;l.out_features=3;int g=t==LayerType::RNN?1:t==LayerType::GRU?3:4;l.params_count=g*(6+9+6);break; }
        case LayerType::Constant:l.weights=values(12);l.shape={-1,3};l.seq_len=6;l.embed_dim=3;l.trainable_parameter=true;x.clear();break;
        case LayerType::PatchEmbed:l.seq_text=1;l.num_patches=2;l.patch_dim=2;l.embed_dim=2;l.params_count=6;x={values(8)};break;
        case LayerType::Reparameterize:x.push_back(values(8));break;
        default:break;
        }
        emit(l,x);
        if(t==LayerType::Linear||t==LayerType::Bilinear||t==LayerType::Conv2d||t==LayerType::ConvTranspose2d||
           t==LayerType::Conv1d||t==LayerType::DepthwiseConv2d||t==LayerType::RNN||t==LayerType::GRU||t==LayerType::LSTM) {
            Layer no_bias=l;no_bias.use_bias=false;
            size_t count=t==LayerType::Linear?3:t==LayerType::Bilinear?2:t==LayerType::DepthwiseConv2d?2:
                t==LayerType::RNN?6:t==LayerType::GRU?18:t==LayerType::LSTM?24:3;
            no_bias.params_count-=count;emit(no_bias,x);
        }
        if(t==LayerType::LayerNorm||t==LayerType::RMSNorm||t==LayerType::GroupNorm||t==LayerType::InstanceNorm2d||
           t==LayerType::BatchNorm1d||t==LayerType::BatchNorm2d) {
            Layer plain=l;plain.affine=false;plain.params_count=0;emit(plain,x);
            Layer no_bias=l;no_bias.use_bias=false;emit(no_bias,x);
        }
        if(t==LayerType::Constant) {
            Layer fixed=l;fixed.trainable_parameter=false;emit(fixed,x);
            fixed=l;fixed.seq_len=2;emit(fixed,x);
        }
        if(t==LayerType::SelfAttention||t==LayerType::MultiHeadAttention) {
            Layer no_bias=l;no_bias.params_count=64;emit(no_bias,x);
            no_bias.params_count=68;emit(no_bias,x);
        }
        if(t==LayerType::NMS) { Layer limited=l;limited.nms_max_detections=1;emit(limited,x);emit(l,{{},{}}); }
        if(t==LayerType::Conv2d) { Layer dilated=l;dilated.dilation=2;emit(dilated,x); }
        if(t==LayerType::MaxPool2d||t==LayerType::MaxPool1d)emit(l,{std::vector<float>(8,1.f)});
        if(t==LayerType::GroupNorm) { Layer grouped=l;grouped.num_groups=1;emit(grouped,x); }
        if(t==LayerType::Dropout||t==LayerType::Dropout2d||t==LayerType::AlphaDropout||t==LayerType::Reparameterize) {
            emit(l,x,true);l.dropout_p=1;emit(l,x,true);l.dropout_p=0;emit(l,x,true);
        }
        if(t==LayerType::BatchNorm2d||t==LayerType::BatchNorm1d) {
            emit(l,x,true);l.running_mean={.3f,-.2f};l.running_var={.7f,1.2f};emit(l,x);
        }
        if(t==LayerType::Conv2d||t==LayerType::ConvTranspose2d) {
            l.activation=ActivationType::RELU;emit(l,x);
            l.activation=ActivationType::NONE;l.stride=2;l.padding=1;emit(l,x);
        }
        if(t==LayerType::Add)emit(l,{values(8),values(2)});
        if(t==LayerType::Divide)emit(l,{values(8),{0,1e-10f,-1e-10f,2,3,-2,1,0}});
        if(t==LayerType::ReLU||t==LayerType::HardSigmoid||t==LayerType::HardSwish||t==LayerType::LeakyReLU)emit(l,{{-4,-3,0,3,4}});
    }
    if(hardware)std::cout<<"TPU_HARDWARE_PARITY_OK cases="<<cases.size()<<" device="<<tpu.deviceDescription()<<"\n";
    else if(argc==2) { std::ofstream out(argv[1]);out<<cases.dump(); }
    else std::cout<<"TPU_CPU_FIXTURE_OK cases="<<cases.size()<<"\n";
    return 0;
}
