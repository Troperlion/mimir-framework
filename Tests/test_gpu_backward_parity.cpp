#include "runtimes/opencl/OpenCLRuntime.hpp"
#include "runtimes/vulkan/VulkanRuntime.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/NativeBackward.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include <iostream>
#include <stdexcept>

void compare(const std::vector<float>& actual,const std::vector<float>& expected) {
    if(actual.size()!=expected.size()) throw std::runtime_error("size mismatch");
    for(size_t i=0;i<actual.size();++i) if(!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i])>3e-4f*(1+std::abs(expected[i])))
        throw std::runtime_error("gradient/value mismatch at "+std::to_string(i)+" got="+std::to_string(actual[i])+" expected="+std::to_string(expected[i]));
}
void check(AbstractRuntime& gpu,CpuRuntime& cpu,Layer layer,std::vector<std::vector<float>> in,std::vector<float> go) {
    auto& router=RuntimeRouter::instance();
    router.setRuntimes(nullptr,nullptr,std::string(gpu.name())=="VULKAN"?&gpu:nullptr,
        std::string(gpu.name())=="OPENCL"?&gpu:nullptr,&cpu);
    AbstractRuntime* selected=nullptr;
    std::vector<const std::vector<float>*> ip;
    for(auto& x:in) ip.push_back(&x);
    Layer reference=layer;
    std::vector<std::vector<float>> a,b;
    if(!router.dispatchForwardLayer(ip,a,layer,true,&selected) || !cpu.forwardLayer(ip,b,reference,true)) throw std::runtime_error("forward failed: "+layer.type);
    if(selected!=&gpu) throw std::runtime_error("forward silently fell back to CPU");
    compare(a.at(0),b.at(0));
    for(int accumulation=0;accumulation<2;++accumulation) {
        if(!router.dispatchBackwardLayer(ip,{&go},a,layer,true,&selected) || !cpu.backwardLayer(ip,{&go},b,reference,true)) throw std::runtime_error("backward failed: "+layer.type);
        if(selected!=&gpu) throw std::runtime_error("backward silently fell back to CPU");
        if(a.size()!=b.size()) throw std::runtime_error("gradient count mismatch");
        for(size_t i=0;i<a.size();++i) compare(a[i],b[i]);
        compare(layer.grad_weights,reference.grad_weights);
        compare(layer.grad_bias,reference.grad_bias);
    }
    std::cout<<gpu.name()<<" "<<layer.type<<" OK\n";
}
int main(int argc, char** argv) {
    CpuRuntime cpu; RuntimeConfig cfg; cfg.linear_enabled=true; cfg.linear_min_ops=0; cfg.conv_enabled=true; cfg.conv_min_ops=0; cpu.initialize(cfg);
    std::vector<AbstractRuntime*> backends;
#ifdef ENABLE_OPENCL
    OpenCLRuntime cl;
    if (argc < 2 || std::string(argv[1]) == "opencl") backends.push_back(&cl);
#endif
#ifdef ENABLE_VULKAN
    VulkanRuntime vk;
    if (argc < 2 || std::string(argv[1]) == "vulkan") backends.push_back(&vk);
#endif
    int available=0;
    for(AbstractRuntime* gpu:backends) {
        if(!gpu->initialize(cfg)) { std::cout<<gpu->name()<<" UNAVAILABLE\n"; continue; }
        ++available;
        for(LayerType type:{LayerType::ReLU,LayerType::LeakyReLU,LayerType::Sigmoid,LayerType::Tanh,LayerType::SiLU,LayerType::GELU,LayerType::Softplus,LayerType::Mish,LayerType::HardSigmoid,LayerType::HardSwish}) {
            Layer l("unary",LayerRegistry::type_to_string(type),0);l.leaky_relu_alpha=.2f;
            check(*gpu,cpu,l,{{-2.1f,-.3f,.4f,1.7f}},{.4f,-.8f,.1f,.6f});
        }
        for(LayerType type:{LayerType::Add,LayerType::Subtract,LayerType::Multiply,LayerType::Divide}) {
            Layer l("binary",LayerRegistry::type_to_string(type),0);
            check(*gpu,cpu,l,{{-2.1f,-.3f,.4f,1.7f},{.3f,2.f,-1.f,3.f}},{.4f,-.8f,.1f,.6f});
        }
        Layer linear("linear","Linear",8);linear.in_features=3;linear.out_features=2;linear.use_bias=true;
        linear.weights={.1f,.3f,-.7f,1.f,-.2f,.4f,.3f,-.1f};
        check(*gpu,cpu,linear,{{.2f,.5f,-.3f,-.7f,.3f,.1f}},{.3f,-.4f,.7f,-.2f});
        for(bool batch:{false,true}) {
            Layer l("matmul",batch?"BatchMatMul":"MatMul",0);l.in_features=2;l.out_features=3;l.embed_dim=2;l.seq_len=batch?2:0;
            std::vector<float> a{.2f,.5f,-.3f,-.7f,.3f,.1f},b{.7f,.3f,.1f,.5f,-.3f,.2f},g{.3f,-.4f,.7f,-.2f};
            if(batch){auto aa=a,bb=b,gg=g;a.insert(a.end(),aa.begin(),aa.end());b.insert(b.end(),bb.begin(),bb.end());g.insert(g.end(),gg.begin(),gg.end());}
            check(*gpu,cpu,l,{a,b},g);
        }
    }
    RuntimeRouter::instance().setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
    return available == static_cast<int>(backends.size()) && available > 0 ? 0 : 77;
}
