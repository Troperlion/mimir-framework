#include "test_utils.hpp"
#include "Model.hpp"
#include "runtimes/NativeBackward.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"

struct FailingDevice {
    int calls=0;
    bool matmulForward(const float*,const float*,float* out,int m,int,int n) {
        std::fill(out,out+static_cast<size_t>(m)*n,99.0f);
        return ++calls<2;
    }
    bool binaryForward(const float*,const float*,float*,int,int,float=0.0f) { return false; }
};

int main() {
    CpuRuntime cpu;
    RuntimeConfig cfg;
    cpu.initialize(cfg);
    auto& router = RuntimeRouter::instance();
    router.setRuntimes(nullptr, nullptr, nullptr, nullptr, &cpu);
    std::vector<float> x{1,2,3,4}, small{2,3}, go{1,2,3,4};
    std::vector<std::vector<float>> gi;
    Layer add("broadcast", "Add", 0);
    TASSERT_TRUE(router.dispatchBackwardLayer({&x,&small},{&go},gi,add,true));
    TASSERT_TRUE(gi[0] == go);
    TASSERT_TRUE(gi[1] == std::vector<float>({4,6}));
    Layer stack("stack", "Stack", 0);
    TASSERT_TRUE(router.dispatchBackwardLayer({&small,&small},{&go},gi,stack,true));
    TASSERT_TRUE(gi == std::vector<std::vector<float>>({{1,2},{3,4}}));
    Layer constant("constant", "Constant", 4);
    constant.trainable_parameter = true;
    constant.weights = x;
    TASSERT_TRUE(router.dispatchBackwardLayer({},{&go},gi,constant,true));
    TASSERT_TRUE(gi.empty());
    TASSERT_TRUE(constant.grad_weights == go);
    Layer linear("linear", "Linear", 6);
    linear.in_features=2; linear.out_features=2; linear.use_bias=true;
    linear.weights={1,0,0,1,0,0};
    TASSERT_TRUE(router.dispatchBackwardLayer({&x},{&go},gi,linear,true));
    TASSERT_TRUE(linear.grad_weights == std::vector<float>({10,14,14,20,4,6}));
    TASSERT_TRUE(router.dispatchBackwardLayer({&x},{&go},gi,linear,true));
    TASSERT_NEAR(linear.grad_weights[4],8,1e-6f);
    Layer dropout("dropout", "Dropout", 0);
    dropout.dropout_p=0.5f;
    std::vector<uint8_t> mask{1,0,1,0};
    RuntimeBackwardContext context;
    context.output_mask=&mask;
    TASSERT_TRUE(router.dispatchBackwardLayer({&x},{&go},gi,dropout,true,nullptr,context));
    TASSERT_TRUE(gi[0] == std::vector<float>({2,0,6,0}));
    // A kept zero input cannot be distinguished from a dropped value by y.
    // The runtime must record the actual sampled mask.
    for (const char* type : {"Dropout","Dropout2d","AlphaDropout"}) {
        Layer stochastic("stochastic",type,0);
        stochastic.dropout_p=.5f;
        std::vector<float> zero(64,0.0f),ones(64,1.0f);
        std::vector<uint8_t> sampled;
        RuntimeForwardContext saved; saved.output_mask=&sampled;
        std::vector<std::vector<float>> output;
        TASSERT_TRUE(router.dispatchForwardLayer({&zero},output,stochastic,true,nullptr,saved));
        TASSERT_TRUE(sampled.size()==zero.size());
        RuntimeBackwardContext backward_saved; backward_saved.output_mask=&sampled;
        TASSERT_TRUE(router.dispatchBackwardLayer({&zero},{&ones},gi,stochastic,true,nullptr,backward_saved));
        const float ap=-1.6732632423543773f*1.0507009873554805f;
        const float scale=stochastic.type_enum==LayerType::AlphaDropout?1/std::sqrt(.5f*(1+.5f*ap*ap)):2.0f;
        size_t kept=0;
        for(size_t i=0;i<sampled.size();++i){kept+=sampled[i];TASSERT_NEAR(gi[0][i],sampled[i]?scale:0.0f,1e-6f);}
        TASSERT_TRUE(kept>0 && kept<sampled.size());
    }
    TASSERT_NEAR(Autograd::gelu_backward(0.5f,1.0f),0.86737f,1e-4f);
    TASSERT_TRUE(Autograd::residual_backward(go)==go);
    auto normalized=Autograd::layernorm_backward(go,x,x);
    TASSERT_TRUE(normalized.size()==x.size());
    FailingDevice failed;
    const auto previous_weights=linear.grad_weights;
    const auto previous_bias=linear.grad_bias;
    const auto previous_inputs=gi;
    TASSERT_TRUE(!NativeBackward::execute(failed,{&x},{&go},gi,linear));
    TASSERT_TRUE(linear.grad_weights==previous_weights && linear.grad_bias==previous_bias && gi==previous_inputs);
    // Avoid leaving pointers to a local runtime in the global router.
    router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
    return 0;
}
