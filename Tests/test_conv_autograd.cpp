#include "test_utils.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/RuntimeLayerDispatch.hpp"

int main() {
    for (bool accelerated : {false, true}) {
    RuntimeLayerOps::setHardwareAcceleration(accelerated);
    for (bool transposed : {false,true}) for(int stride : {1,2}) for(int dilation : {1,2}) {
        if (transposed && dilation != 1) continue;
        Layer layer("conv", transposed?"ConvTranspose2d":"Conv2d",0);
        layer.in_channels=2;layer.out_channels=2;layer.input_height=5;layer.input_width=4;
        layer.kernel_h=3;layer.kernel_w=3;layer.stride_h=stride;layer.pad_h=1;layer.dilation_h=dilation;
        layer.use_bias=true;layer.weights.resize(38);
        for(size_t i=0;i<layer.weights.size();++i)layer.weights[i]=.1f*std::sin(float(i));
        std::vector<float> x(40);
        for(size_t i=0;i<x.size();++i)x[i]=.2f*std::cos(float(i));
        auto forward=[&](){std::vector<std::vector<float>> out;
            if(!RuntimeLayerDispatch::cpu_forward_layer({&x},out,layer,true)) throw std::runtime_error("forward");
            return out.at(0);
        };
        auto y=forward();std::vector<float> g(y.size());
        for(size_t i=0;i<g.size();++i)g[i]=.3f*std::sin(float(i)+.2f);
        auto loss=[&](){auto values=forward();float result=0;for(size_t i=0;i<g.size();++i)result+=values[i]*g[i];return result;};
        std::vector<std::vector<float>> dx;
        TASSERT_TRUE(RuntimeLayerDispatch::cpu_backward_layer({&x},{&g},dx,layer,true));
        constexpr float h=.001f;
        for(size_t i=0;i<x.size();++i){float old=x[i];x[i]=old+h;float plus=loss();x[i]=old-h;float minus=loss();x[i]=old;TASSERT_NEAR(dx[0][i],(plus-minus)/(2*h),.002f);}
        TASSERT_TRUE(layer.grad_weights.size()==layer.weights.size());
        for(size_t i=0;i<layer.weights.size();++i){float old=layer.weights[i];layer.weights[i]=old+h;float plus=loss();layer.weights[i]=old-h;float minus=loss();layer.weights[i]=old;TASSERT_NEAR(layer.grad_weights[i],(plus-minus)/(2*h),.002f);}
    }
    }
    CpuRuntime cpu;RuntimeConfig config;cpu.initialize(config);
    auto& router=RuntimeRouter::instance();router.setRuntimes(nullptr,nullptr,nullptr,nullptr,&cpu);
    for(auto activation:{ActivationType::RELU,ActivationType::RELU6,ActivationType::LEAKY_RELU,
        ActivationType::ELU,ActivationType::SELU,ActivationType::GELU,ActivationType::SWISH,
        ActivationType::MISH,ActivationType::TANH,ActivationType::SIGMOID,ActivationType::SOFTPLUS,
        ActivationType::SOFTSIGN,ActivationType::SOFTMAX}) {
        Layer l("activated","Conv2d",2);l.in_channels=1;l.out_channels=1;l.input_height=2;l.input_width=2;
        l.kernel_h=1;l.use_bias=true;l.weights={1.2f,.2f};l.activation=activation;l.activation_param=.1f;
        std::vector<float> x{-.7f,.3f,.4f,1.2f},g{-.2f,.4f,-.5f,.2f};
        auto loss=[&](){std::vector<std::vector<float>> y;if(!router.dispatchForwardLayer({&x},y,l,true))throw std::runtime_error("activated forward");float sum=0;for(size_t i=0;i<x.size();++i)sum+=y[0][i]*g[i];return sum;};
        std::vector<std::vector<float>> dx;
        TASSERT_TRUE(router.dispatchBackwardLayer({&x},{&g},dx,l,true));
        for(size_t i=0;i<x.size();++i){float old=x[i];x[i]=old+.001f;float plus=loss();x[i]=old-.001f;float minus=loss();x[i]=old;TASSERT_NEAR(dx[0][i],(plus-minus)/.002f,.002f);}
    }
    router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
    return 0;
}
