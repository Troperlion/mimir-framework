#include "test_utils.hpp"
#include "Model.hpp"
#include "runtimes/opencl/OpenCLRuntime.hpp"
#include "runtimes/vulkan/VulkanRuntime.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "runtimes/RuntimeRouter.hpp"
#include "runtimes/portable/PortableStaging.hpp"
#include <memory>
#include <string>
#include "runtimes/cpu/RuntimeLayerDispatch.hpp"
#include <cmath>
#include <chrono>
#include <climits>
#include <iostream>
#include <limits>
#include <stdexcept>

static std::string type_name(LayerType type){
    if(type==LayerType::PatchEmbed)return "PatchEmbed";
    if(type==LayerType::Reparameterize)return "Reparameterize";
    return LayerRegistry::type_to_string(type);
}
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static void compare(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size()==b.size(),"GPU/CPU shape mismatch");
    for (size_t i=0;i<a.size();++i) {
        if (a[i]==b[i]) continue;
        if (!std::isfinite(a[i]) || std::abs(a[i]-b[i])>4e-4f*(1+std::abs(b[i]))) {
            std::cerr<<"index="<<i<<" GPU="<<a[i]<<" CPU="<<b[i]<<'\n';
            throw std::runtime_error("GPU/CPU numerical mismatch");
        }
    }
}
static void parity_inputs(AbstractRuntime& runtime, Layer& layer, const std::vector<const std::vector<float>*>& inputs) {
    std::vector<std::vector<float>> gpu,cpu;
    require(runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(layer,inputs,true)),"forward must be native");
    MimirRng::setSeed(4242);
    require(runtime.forwardLayer(inputs,gpu,layer,true),"GPU forward failed");
    MimirRng::setSeed(4242);
    require(RuntimeLayerDispatch::cpu_forward_layer(inputs,cpu,layer,true),"CPU reference forward failed");
    require(gpu.size()==cpu.size(),"output count");
    std::vector<std::vector<float>> dy(gpu.size());
    std::vector<const std::vector<float>*> grads;
    for(size_t o=0;o<gpu.size();++o){
        compare(gpu[o],cpu[o]);dy[o].resize(gpu[o].size());
        for(size_t i=0;i<dy[o].size();++i)dy[o][i]=static_cast<float>(static_cast<int>((i+o)%7)-3)*.17f;
        grads.push_back(&dy[o]);
    }
    require(runtimeCapabilityIsNative(runtime.queryBackwardOperationCapability(layer,inputs,grads,true)),"backward must be native");
    auto backward_inputs=inputs;
    if(layer.type_enum==LayerType::Dropout || layer.type_enum==LayerType::Dropout2d || layer.type_enum==LayerType::AlphaDropout || layer.type_enum==LayerType::Reparameterize)backward_inputs.push_back(&gpu[0]);
    Layer gpu_layer=layer, cpu_layer=layer;
    gpu_layer.grad_weights.assign(layer.getWeightsSize(),.125f);
    cpu_layer.grad_weights=gpu_layer.grad_weights;
    const int bias_count=layer.type_enum==LayerType::DepthwiseConv2d?layer.in_channels:layer.type_enum==LayerType::Bilinear?layer.embed_dim:(layer.type_enum==LayerType::LayerNorm || layer.type_enum==LayerType::RMSNorm)?layer.in_features:((layer.type_enum==LayerType::GroupNorm || layer.type_enum==LayerType::InstanceNorm2d)?layer.in_channels:layer.out_channels);
    if (layer.affine && layer.use_bias && layer.getWeightsSize()) {
        gpu_layer.grad_bias.assign(bias_count,.25f); cpu_layer.grad_bias=gpu_layer.grad_bias;
    }
    for (int repeat=0;repeat<2;++repeat) {
        std::vector<std::vector<float>> gi,ci;
        require(runtime.backwardLayer(backward_inputs,grads,gi,gpu_layer,true),"GPU backward failed");
        require(RuntimeLayerDispatch::cpu_backward_layer(backward_inputs,grads,ci,cpu_layer,true),"CPU reference backward failed");
        require(gi.size()==ci.size(),"gradient count");
        for(size_t i=0;i<gi.size();++i) compare(gi[i],ci[i]);
        compare(gpu_layer.grad_weights,cpu_layer.grad_weights);
        compare(gpu_layer.grad_bias,cpu_layer.grad_bias);
    }
    std::vector<float> wrong(dy[0].size()+1); grads[0]=&wrong;
    require(!runtimeCapabilityIsNative(runtime.queryBackwardOperationCapability(layer,inputs,grads,true)),"reject wrong gradients");
    std::cout<<"GPU_NATIVE_PARITY_OK "<<type_name(layer.type_enum)<<'\n';
}
static void parity(AbstractRuntime& runtime,Layer& layer,const std::vector<float>& x){parity_inputs(runtime,layer,{&x});}
int main(int argc,char** argv) {
    try {
        // Coverage must force native GPU routing; measured placement has its own test.
        setTestEnvironment("MIMIR_PLANNER_MODE","static");
        const std::string backend=argc>1?argv[1]:"VULKAN";
        std::unique_ptr<AbstractRuntime> selected;
        if(backend=="OPENCL") {
#ifdef ENABLE_OPENCL
            selected=std::make_unique<OpenCLRuntime>();
#else
            return 77;
#endif
        } else if(backend=="VULKAN") {
#ifdef ENABLE_VULKAN
            selected=std::make_unique<VulkanRuntime>();
#else
            return 77;
#endif
        } else throw std::runtime_error("expected OPENCL or VULKAN");
        AbstractRuntime& runtime=*selected;RuntimeConfig cfg;
        cfg.linear_enabled=cfg.norm_enabled=cfg.conv_enabled=cfg.attention_enabled=true;
        cfg.linear_min_ops=cfg.norm_min_elements=cfg.conv_min_ops=cfg.attention_min_ops=0;
        int forward=0,backward=0,total=0,cpu_backward=0;
        for (int value=0;value<static_cast<int>(LayerType::UNKNOWN);++value) {
            const auto type=static_cast<LayerType>(value);
            // CPU forward rejects Lambda (Lua callbacks live outside the runtime).
            if(type==LayerType::Lambda) {
                cpu_backward+=RuntimeLayerDispatch::cpu_supports_backward_layer_type(type);
                backward+=runtimeCapabilityIsNative(runtime.queryBackwardCapability(type));
                continue;
            }
            ++total;
            cpu_backward+=RuntimeLayerDispatch::cpu_supports_backward_layer_type(type);
            forward+=runtimeCapabilityIsNative(runtime.queryForwardCapability(type));
            backward+=runtimeCapabilityIsNative(runtime.queryBackwardCapability(type));
            if(!runtimeCapabilityIsNative(runtime.queryForwardCapability(type)))std::cout<<"GPU_FORWARD_REMAINING "<<type_name(type)<<'\n';
            if(RuntimeLayerDispatch::cpu_supports_backward_layer_type(type) && !runtimeCapabilityIsNative(runtime.queryBackwardCapability(type)))std::cout<<"GPU_BACKWARD_REMAINING "<<type_name(type)<<'\n';
        }
        std::cout<<"GPU native types: forward="<<forward<<'/'<<total<<" backward="<<backward<<'/'<<cpu_backward<<'\n';
        require(forward==total && backward==cpu_backward,"GPU/CPU type coverage must match CPU");
        if(!runtime.initialize(cfg)) { std::cout<<"SKIP: "<<backend<<" device unavailable\n";return 77; }
        {
            Layer l; l.type_enum=LayerType::Linear; l.in_features=128; l.out_features=1; l.seq_len=254; l.use_bias=true;
            l.weights.assign(129,.1f); l.grad_weights.assign(129,.125f); l.grad_bias.assign(1,.25f);
            std::vector<float> x(254*128,.2f), dy(254,.3f);
            std::vector<std::vector<float>> gradients;
            auto& device=PortableStaging::contexts[backend+":0"].device;
            // The runtime creates its staging device lazily on first dispatch.
            require(runtime.backwardLayer({&x},{&dy},gradients,l,true),"parallel Linear backward accepts larger staging");
            require(bool(device),"staging device exists");
            auto before=device->batchStatistics();
            Layer probe=l;std::vector<std::vector<float>> scratch;
            require(runtime.backwardLayer({&x},{&dy},scratch,probe,true),"batched Linear backward");
            auto after=device->batchStatistics();
            require(after.submissions-before.submissions==1 && after.kernels-before.kernels==3,"three Linear gradients share one GPU submission");
            require(after.downloaded_bytes-before.downloaded_bytes<after.uploaded_bytes-before.uploaded_bytes,"unchanged Linear inputs are not downloaded");
            std::cout<<"GPU_BATCH_OK backend="<<backend<<" kernels="<<after.kernels-before.kernels
                <<" submissions="<<after.submissions-before.submissions
                <<" uploaded_bytes="<<after.uploaded_bytes-before.uploaded_bytes
                <<" downloaded_bytes="<<after.downloaded_bytes-before.downloaded_bytes<<'\n';
            compare(gradients[0],std::vector<float>(x.size(),.03f));
            std::vector<float> expected(129,.125f+254*.06f);expected.back()=.125f+254*.3f;compare(l.grad_weights,expected);
            compare(l.grad_bias,std::vector<float>(1,.25f+254*.3f));
        }
        {
            std::vector<float> x(8*128*128);
            for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i%997)*.037f);
            Layer norm;norm.type_enum=LayerType::GroupNorm;norm.in_channels=8;
            norm.input_height=norm.input_width=128;norm.num_groups=4;
            norm.affine=true;norm.use_bias=true;norm.weights.assign(16,.3f);
            parity(runtime,norm,x);
            norm.type_enum=LayerType::InstanceNorm2d;parity(runtime,norm,x);
            Layer softmax;softmax.type_enum=LayerType::Softmax;softmax.shape={128,1024};softmax.axis=1;
            parity(runtime,softmax,x);
            softmax.type_enum=LayerType::LogSoftmax;parity(runtime,softmax,x);
            Layer pool;pool.type_enum=LayerType::MaxPool2d;pool.in_channels=8;
            pool.input_height=pool.input_width=128;pool.kernel_size=3;pool.stride=2;pool.padding=1;
            parity(runtime,pool,x);
            pool.type_enum=LayerType::AvgPool2d;parity(runtime,pool,x);
            pool.type_enum=LayerType::GlobalAvgPool2d;parity(runtime,pool,x);
            Layer conv;conv.type_enum=LayerType::Conv2d;conv.in_channels=8;conv.out_channels=4;
            conv.input_height=conv.input_width=128;conv.kernel_size=3;conv.stride=1;conv.padding=1;
            conv.use_bias=true;conv.weights.resize(4*8*9+4);
            for(size_t i=0;i<conv.weights.size();++i)conv.weights[i]=float(int(i%13)-6)*.002f;
            parity(runtime,conv,x);
            Layer constant;constant.type_enum=LayerType::Constant;constant.trainable_parameter=true;
            constant.embed_dim=256;constant.seq_len=512;constant.shape={-1,256};constant.weights.assign(128*256,.3f);
            parity_inputs(runtime,constant,{});
            std::cout<<"LARGE_GENERALIZED_KERNELS_OK elements="<<x.size()<<'\n';
        }
        {
            std::vector<float> large(3 * 1024 * 1024);
            Layer relu; relu.type_enum=LayerType::ReLU;
            Layer norm; norm.type_enum=LayerType::LayerNorm;
            require(runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(relu,{&large},true)),"large parallel activation must remain available");
            require(!runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(norm,{&large},true)),"large serial normalization must be refused");
            require(runtimeCapabilityIsNative(runtime.queryBackwardOperationCapability(relu,{&large},{&large},true)),"large parallel backward available");
            large.front()=2.f;large.back()=-2.f;
            std::vector<std::vector<float>> dx;
            require(runtime.backwardLayer({&large},{&large},dx,relu,true),"large parallel activation backward");
            require(dx[0].size()==large.size() && dx[0].front()==2.f && dx[0].back()==0.f,"large activation gradient");
            const auto norm_start=std::chrono::steady_clock::now();
            norm.in_features=1024;norm.affine=false;norm.use_bias=false;
            require(runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(norm,{&large},true)),"large row normalization available");
            require(runtime.forwardLayer({&large},dx,norm,true),"large parallel normalization forward");
            require(runtime.backwardLayer({&large},{&large},dx,norm,true),"large parallel normalization backward");
            std::vector<std::vector<float>> expected;
            require(RuntimeLayerDispatch::cpu_backward_layer({&large},{&large},expected,norm,true),"large CPU norm reference");
            compare(dx[0],expected[0]);
            std::cout<<"LARGE_NORM_FORWARD_BACKWARD_OK elements="<<large.size()<<" elapsed_with_cpu_check_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-norm_start).count()<<'\n';
            CpuRuntime cpu; require(cpu.initialize(cfg),"CPU fallback initializes");
            auto& router=RuntimeRouter::instance();
            router.setActivators({}, {}, {}, {}, {});
            router.setRuntimes(nullptr,nullptr,backend=="VULKAN"?&runtime:nullptr,backend=="OPENCL"?&runtime:nullptr,&cpu);
            Layer identity; identity.type_enum=LayerType::Identity;
            large.front()=1.f; large.back()=-2.f;
            std::vector<std::vector<float>> output;
            AbstractRuntime* selected=nullptr;
            require(router.dispatchForwardLayer({&large},output,identity,true,&selected),"large reference operation routes to fallback");
            require(selected==&runtime && output.size()==1,"large copy must select native GPU");
            compare(output[0],large);
            require(router.dispatchBackwardLayer({&large},{&large},output,identity,true,&selected),"large backward routes to fallback");
            require(selected==&runtime && output.size()==1,"large copy backward must select native GPU");
            compare(output[0],large);
            router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
        }
        {
            Layer linear;linear.type_enum=LayerType::Linear;linear.in_features=128;linear.out_features=linear.out_channels=128;linear.seq_len=512;linear.use_bias=true;
            linear.weights.resize(128*128+128);
            for(size_t i=0;i<linear.weights.size();++i)linear.weights[i]=float(int(i%13)-6)*.002f;
            std::vector<float> x(512*128);
            for(size_t i=0;i<x.size();++i)x[i]=float(int(i%17)-8)*.03f;
            parity(runtime,linear,x);
            Layer norm;norm.type_enum=LayerType::LayerNorm;norm.in_features=128;norm.affine=true;norm.use_bias=true;norm.weights.assign(256,.3f);
            parity(runtime,norm,x);
            norm.type_enum=LayerType::RMSNorm;parity(runtime,norm,x);
        }
        {
            Layer layer;layer.type_enum=LayerType::Lambda;
            std::vector<float>x{1,2,-3},go{.5f,-.3f,.7f};std::vector<std::vector<float>>gpu,cpu;
            require(!RuntimeLayerDispatch::cpu_forward_layer({&x},cpu,layer,true),"CPU Lambda forward is unsupported");
            require(!runtimeCapabilityIsNative(runtime.queryForwardCapability(layer.type_enum)),"no fictitious Lambda forward");
            require(runtime.backwardLayer({&x},{&go},gpu,layer,true),"Lambda backward native");
            require(RuntimeLayerDispatch::cpu_backward_layer({&x},{&go},cpu,layer,true),"Lambda CPU backward");compare(gpu[0],cpu[0]);
        }
        {
            CpuRuntime cpu;RuntimeConfig cpu_cfg;require(cpu.initialize(cpu_cfg),"CPU oracle initializes");
            auto& router=RuntimeRouter::instance();
            router.setActivators({}, {}, {}, {}, {});
            router.setRuntimes(nullptr,nullptr,backend=="VULKAN"?&runtime:nullptr,backend=="OPENCL"?&runtime:nullptr,&cpu);
            Layer layer;layer.type_enum=LayerType::GroupNorm;layer.in_channels=2;layer.input_height=1;layer.input_width=2;layer.num_groups=1;layer.affine=false;layer.eps=1e-5f;
            std::vector<float>x{1,2,3,5},dy{.2f,-.3f,.4f,-.1f};std::vector<std::vector<float>>out,gi;
            AbstractRuntime* used=nullptr;
            require(router.dispatchForwardLayer({&x},out,layer,true,&used) && used==&runtime,"forward selects requested GPU runtime");
            require(router.dispatchBackwardLayer({&x},{&dy},gi,layer,true,&used) && used==&runtime,"backward selects requested GPU runtime");
            router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
        }
        for(auto type:{LayerType::ReLU,LayerType::LeakyReLU,LayerType::Sigmoid,LayerType::Tanh,LayerType::SiLU,LayerType::GELU,LayerType::Softplus,LayerType::Mish,LayerType::HardSigmoid,LayerType::HardSwish}) {
            Layer l;l.type_enum=type;l.leaky_relu_alpha=.13f;std::vector<float>x{-4,-3,-1,-.2f,0,.3f,1,3,4,21};parity(runtime,l,x);
        }
        for(auto type:{LayerType::Add,LayerType::Subtract,LayerType::Multiply,LayerType::Divide}) {
            Layer l;l.type_enum=type;std::vector<float>a{1,-2,.3f,0,4},b{2,.5f,-.2f,1,-3};parity_inputs(runtime,l,{&a,&b});
        }
        {
            Layer l;l.type_enum=LayerType::Linear;l.in_features=3;l.out_features=4;l.seq_len=2;
            std::vector<float>x{1,-2,3,.3f,.4f,-.5f};
            for(bool bias:{false,true}){l.use_bias=bias;l.weights.resize(12+(bias?4:0));for(size_t i=0;i<l.weights.size();++i)l.weights[i]=std::sin(i*.31f)*.2f;parity(runtime,l,x);}
        }
        for(auto type:{LayerType::MatMul,LayerType::BatchMatMul}) {
            Layer l;l.type_enum=type;l.in_features=2;l.out_features=3;l.embed_dim=4;l.seq_len=2;
            int batches=type==LayerType::BatchMatMul?2:1;std::vector<float>a(batches*6),b(batches*12);for(size_t i=0;i<a.size();++i)a[i]=std::sin(i*.23f);for(size_t i=0;i<b.size();++i)b[i]=std::cos(i*.19f);parity_inputs(runtime,l,{&a,&b});
        }
        {
            Model model; model.push("norm","GroupNorm",8); auto& layer=model.getMutableLayers()[0];
            layer.in_channels=4; layer.input_height=2; layer.input_width=3; layer.num_groups=2;
            layer.affine=true; layer.use_bias=true; layer.eps=1e-5f; layer.params_count=8; model.allocateParams();
            for (int i=0;i<8;++i) layer.getWeights()[i]=.3f+.13f*i;
            std::vector<float> x(24); for (int i=0;i<24;++i) x[i]=std::sin(i*.7f);
            parity(runtime,layer,x);
            layer.affine=false; layer.use_bias=false; parity(runtime,layer,x);
            layer.affine=true; layer.use_bias=false; parity(runtime,layer,x);
            layer.affine=false; layer.num_groups=4; parity(runtime,layer,std::vector<float>(24,3.f));
            layer.num_groups=3; std::vector<const std::vector<float>*> inputs{&x};
            require(!runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(layer,inputs,true)),"reject invalid groups");
            layer.num_groups=2; layer.affine=true; layer.use_bias=true; layer.input_width=257;
            x.resize(2056); for (size_t i=0;i<x.size();++i) x[i]=std::sin(i*.13f);
            parity(runtime,layer,x);
        }
        for (const auto type:{LayerType::UpsampleNearest,LayerType::UpsampleBilinear,LayerType::UpsampleBicubic}) {
            Layer layer; layer.type_enum=type; layer.in_channels=2; layer.input_height=2; layer.input_width=3;
            layer.scale_h=2; layer.scale_w=3;
            std::vector<float> x{1,2,3,4,5,6,-1,-2,-3,-4,-5,-6}; parity(runtime,layer,x);
            if (type!=LayerType::UpsampleNearest) {
                layer.output_height=1; layer.output_width=2; parity(runtime,layer,x);
                layer.output_height=3; layer.output_width=5; parity(runtime,layer,x);
            }
        }
        for (const auto type:{LayerType::Softmax,LayerType::LogSoftmax}) {
            Layer layer; layer.type_enum=type; layer.shape={2,3,4};
            std::vector<float> x(24); for (int i=0;i<24;++i) x[i]=1000.f+static_cast<float>(i%9-4);
            for (int axis:{0,1,-1,9}) { layer.axis=axis; parity(runtime,layer,x); }
            layer.shape={1,513}; layer.axis=-1; x.resize(513);
            for (int i=0;i<513;++i) x[i]=(i%17)*.25f;
            parity(runtime,layer,x);
        }
        for (int stride:{1,2}) for (int dilation:{1,2}) {
            Model model; model.push("conv","Conv2d",38); auto& layer=model.getMutableLayers()[0];
            layer.in_channels=2; layer.out_channels=2; layer.input_height=5; layer.input_width=6;
            layer.kernel_size=3; layer.stride=stride; layer.padding=2; layer.dilation=dilation;
            layer.use_bias=true; layer.params_count=38; model.allocateParams();
            for (int i=0;i<38;++i) layer.getWeights()[i]=static_cast<float>(i%11-5)*.07f;
            std::vector<float> x(60); for (int i=0;i<60;++i) x[i]=std::sin(i*.21f);
            parity(runtime,layer,x);
            layer.use_bias=false; parity(runtime,layer,x);
        }
        for(auto type:{LayerType::Identity,LayerType::Flatten,LayerType::Reshape,LayerType::View,LayerType::Squeeze,LayerType::Unsqueeze,LayerType::Transpose,LayerType::Permute}){
            Layer l;l.type_enum=type;l.in_features=4;l.out_features=6;l.shape={2,3,4};l.permute_dims={2,0,1};l.unsqueeze_dim=0;
            std::vector<float>x(24);for(int i=0;i<24;++i)x[i]=i*.07f;parity(runtime,l,x);
        }
        for(auto type:{LayerType::Concat,LayerType::Stack}){
            Layer l;l.type_enum=type;std::vector<float>a{1,2,3},b{-1,-2,-3};parity_inputs(runtime,l,{&a,&b});
            if(type==LayerType::Concat){b.push_back(-4);parity_inputs(runtime,l,{&a,&b});}
        }
        for(auto type:{LayerType::Split,LayerType::Chunk}){
            Layer l;l.type_enum=type;l.num_splits=3;l.num_chunks=3;std::vector<float>x(12);for(int i=0;i<12;++i)x[i]=i*.1f;parity(runtime,l,x);
            if(type==LayerType::Split){l.split_sizes={2,7,3};parity(runtime,l,x);}else{l.num_chunks=5;parity(runtime,l,x);l.num_chunks=14;parity(runtime,l,x);}
        }
        for(bool trainable:{false,true}){
            Layer l;l.type_enum=LayerType::Constant;l.weights={1,3,-2};l.trainable_parameter=trainable;std::vector<float>x{0};parity(runtime,l,x);
        }
        // Actual Constant graph nodes have no input tensor. Exercise prefix,
        // interpolation, one-row tables and the exact adjoint without CPU routes.
        for(bool trainable:{false,true})for(int base_rows:{1,3})for(int rows:{1,2,3,7}){
            Layer l;l.type_enum=LayerType::Constant;l.trainable_parameter=trainable;
            l.embed_dim=3;l.seq_len=rows;l.shape={-1,3};l.weights.resize(base_rows*3);
            for(size_t i=0;i<l.weights.size();++i)l.weights[i]=std::sin(float(i)*.7f);
            parity_inputs(runtime,l,{});
            auto& router=RuntimeRouter::instance();router.setActivators({}, {}, {}, {}, {});
            router.setRuntimes(nullptr,nullptr,backend=="VULKAN"?&runtime:nullptr,backend=="OPENCL"?&runtime:nullptr,nullptr);
            std::vector<std::vector<float>> output;
            AbstractRuntime* selected=nullptr;
            require(router.dispatchForwardLayer({},output,l,true,&selected) && selected==&runtime,"input-free Constant GPU route");
            std::vector<float> gradient(output.at(0).size(),.25f);
            require(router.dispatchBackwardLayer({},{&gradient},output,l,true,&selected) && selected==&runtime,"input-free Constant backward GPU route");
            require(output.empty(),"Constant must not invent input gradients");
            router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
        }
        {
            Layer l;l.type_enum=LayerType::Constant;l.weights={1,3,-2};l.trainable_parameter=true;
            parity_inputs(runtime,l,{});
            l.shape={-1,3};l.embed_dim=3;l.seq_len=INT_MAX;
            require(!runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(l,{},true)),"oversized Constant refused before allocation");
        }
        for(auto type:{LayerType::MaxPool1d,LayerType::AvgPool1d,LayerType::MaxPool2d,LayerType::AvgPool2d,LayerType::GlobalAvgPool2d,LayerType::AdaptiveAvgPool2d,LayerType::TokenMeanPool}){
            Layer l;l.type_enum=type;l.in_channels=2;l.input_height=3;l.input_width=5;l.seq_len=5;l.embed_dim=6;
            l.kernel_h=3;l.kernel_w=3;l.stride_h=l.stride_w=1;l.pad_h=l.pad_w=1;
            std::vector<float>x(30);for(int i=0;i<30;++i)x[i]=std::sin(i*.7f);parity(runtime,l,x);
            x.assign(30,2.f);parity(runtime,l,x); // first maximum wins, overlapping gradients accumulate.
            if(type==LayerType::AvgPool2d){l.kernel_w=2;l.stride_w=2;l.pad_w=0;parity(runtime,l,x);}
        }
        for(auto type:{LayerType::ZeroPad2d,LayerType::ReflectionPad2d,LayerType::ReplicationPad2d,LayerType::PixelShuffle}){
            Layer l;l.type_enum=type;l.in_channels=8;l.input_height=3;l.input_width=5;l.pad_h=l.pad_w=2;l.scale_h=2;
            std::vector<float>x(120);for(int i=0;i<120;++i)x[i]=std::sin(i*.3f);parity(runtime,l,x);
            if(type==LayerType::ZeroPad2d){l.pad_w=1;parity(runtime,l,x);}
        }
        for(auto type:{LayerType::LayerNorm,LayerType::RMSNorm})for(bool affine:{false,true})for(bool bias:{false,true}){
            Layer l;l.type_enum=type;l.in_features=513;l.affine=affine;l.use_bias=bias;l.eps=1e-5f;l.weights.resize(1026);
            for(int i=0;i<1026;++i)l.weights[i]=.3f+.01f*(i%11);
            std::vector<float>x(1539);for(int i=0;i<1539;++i)x[i]=std::sin(i*.3f);parity(runtime,l,x);
        }
        // Forward RMSNorm dimension hints and missing affine weights are legal
        // CPU configurations; backward keeps the CPU's established width rule.
        for(int hint=0;hint<4;++hint)for(bool affine:{false,true}){
            Layer l;l.type_enum=LayerType::RMSNorm;l.affine=affine;l.use_bias=true;l.eps=1e-5f;
            if(hint==0)l.embed_dim=3;
            if(hint==1)l.shape={2,3};
            if(hint==2)l.target_shape={2,3};
            if(hint==3){l.embed_dim=3;l.shape={2,2};l.target_shape={2,1};}
            if(affine)l.weights={.8f,1.2f,.9f,.6f,1.1f,1.4f};
            parity(runtime,l,std::vector<float>{-2,1,3,.5f,-1,2});
        }
        for(auto type:{LayerType::LayerNorm,LayerType::RMSNorm}){
            Layer l;l.type_enum=type;l.affine=true;l.in_features=3;l.eps=1e-5f;
            const std::vector<float> x{-2,1,3,.5f,-1,2};
            if(type==LayerType::LayerNorm) {
                require(!runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(l,{&x},true)),"LayerNorm requires full affine parameters, like CPU");
                l.weights={.8f,1.2f};
                require(!runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(l,{&x},true)),"partial LayerNorm affine rejected, like CPU");
                continue;
            }
            parity(runtime,l,x); // RMSNorm defaults gamma to 1 without weights
            l.weights={.8f,1.2f};
            std::vector<std::vector<float>> gpu,cpu;
            require(runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(l,{&x},true)),"partial affine forward native");
            require(runtime.forwardLayer({&x},gpu,l,true),"partial affine GPU execution");
            require(RuntimeLayerDispatch::cpu_forward_layer({&x},cpu,l,true),"partial affine CPU reference");
            compare(gpu.at(0),cpu.at(0));
            require(!runtimeCapabilityIsNative(runtime.queryBackwardOperationCapability(l,{&x},{&x},true)),"unsafe partial affine backward rejected");
        }
        for(auto type:{LayerType::Embedding,LayerType::EmbeddingBag}){
            Layer l;l.type_enum=type;l.vocab_size=6;l.embed_dim=4;l.padding_idx=0;l.weights.resize(24);
            for(int i=0;i<24;++i)l.weights[i]=i*.11f;
            std::vector<float>ids{1,1,0,2.5f,-1,7,3};parity(runtime,l,ids);
            if(type==LayerType::EmbeddingBag){std::vector<float> offsets{-2,2,2,5,99};parity_inputs(runtime,l,{&ids,&offsets});}
        }
        {
            Layer l;l.type_enum=LayerType::PatchEmbed;l.embed_dim=4;l.seq_text=2;l.num_patches=3;l.patch_dim=5;l.weights.resize(24);
            for(int i=0;i<24;++i)l.weights[i]=std::sin(i*.2f);
            std::vector<float>x(27);for(int i=0;i<27;++i)x[i]=std::cos(i*.3f);parity(runtime,l,x);
        }
        for(bool bias:{false,true}){
            Layer l;l.type_enum=LayerType::Bilinear;l.in_features=3;l.out_features=2;l.embed_dim=4;l.use_bias=bias;l.weights.resize(24+(bias?4:0));
            for(size_t i=0;i<l.weights.size();++i)l.weights[i]=std::sin(i*.2f);
            std::vector<float>a{1,2,3,-1,0,2},b{3,-1,2,1};parity_inputs(runtime,l,{&a,&b});
        }
        for(bool affine:{false,true}){
            Layer l;l.type_enum=LayerType::InstanceNorm2d;l.in_channels=4;l.input_height=2;l.input_width=257;l.affine=affine;l.use_bias=true;l.eps=1e-5f;l.weights.resize(8);
            for(int i=0;i<8;++i) l.weights[i]=.3f+.13f*i;
            std::vector<float> x(2056);
            for(int i=0;i<2056;++i) x[i]=std::sin(i*.13f);
            parity(runtime,l,x);
        }
        {Layer l;l.type_enum=LayerType::GEGLU;l.seq_len=3;l.out_features=513;std::vector<float>x(3078);for(int i=0;i<3078;++i)x[i]=std::sin(i*.3f)*4;parity(runtime,l,x);}
        {
            Layer l;l.type_enum=LayerType::MaxPool2d;l.in_channels=1;l.input_height=l.input_width=4;l.kernel_h=3;l.kernel_w=2;l.stride_h=l.stride_w=1;
            std::vector<float>x(16),dy(4);require(!runtimeCapabilityIsNative(runtime.queryBackwardOperationCapability(l,{&x},{&dy},true)),"reject inconsistent CPU pooling geometry");
            l.type_enum=LayerType::Permute;l.shape={2,2,4};l.permute_dims={0,0,2};
            require(!runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(l,{&x},true)),"reject duplicate permutation axes");
        }
        for(auto type:{LayerType::Conv1d,LayerType::DepthwiseConv2d,LayerType::ConvTranspose2d})for(int stride:{1,2})for(bool bias:{false,true}){
            Layer l;l.type_enum=type;l.in_channels=2;l.out_channels=3;l.input_height=4;l.input_width=5;
            l.kernel_h=l.kernel_w=l.kernel_size=3;l.stride_h=l.stride_w=l.stride=stride;l.pad_h=l.pad_w=l.padding=1;l.use_bias=bias;
            const int parameters=type==LayerType::Conv1d?18:(type==LayerType::DepthwiseConv2d?18:54);
            const int channels=type==LayerType::DepthwiseConv2d?2:3;l.weights.resize(parameters+(bias?channels:0));
            for(size_t i=0;i<l.weights.size();++i)l.weights[i]=std::sin(i*.31f)*.2f;
            std::vector<float>x(type==LayerType::Conv1d?22:40);for(size_t i=0;i<x.size();++i)x[i]=std::cos(i*.17f);parity(runtime,l,x);
        }
        for(auto type:{LayerType::BatchNorm1d,LayerType::BatchNorm2d})for(bool affine:{false,true})for(bool bias:{false,true}){
            Layer l;l.type_enum=type;l.in_channels=3;l.input_height=2;l.input_width=5;l.affine=affine;l.use_bias=bias;l.eps=1e-5f;
            l.weights={.3f,.7f,1.1f,.1f,.2f,-.2f};l.track_running_stats=true;l.running_mean={1,2,3};l.running_var={.5f,2,3};
            std::vector<float>x(90);for(int i=0;i<90;++i)x[i]=std::sin(i*.37f);parity(runtime,l,x);
            std::vector<std::vector<float>>a,b;require(runtime.forwardLayer({&x},a,l,false),"GPU BatchNorm inference");
            require(RuntimeLayerDispatch::cpu_forward_layer({&x},b,l,false),"CPU BatchNorm inference");compare(a[0],b[0]);
        }
        for(auto type:{LayerType::Dropout,LayerType::Dropout2d,LayerType::AlphaDropout})for(float probability:{0.f,.4f,1.f}){
            Layer l;l.type_enum=type;l.dropout_p=probability;std::vector<float>x(1027);for(int i=0;i<1027;++i)x[i]=std::sin(i*.37f);parity(runtime,l,x);
            // Saved mask rather than output values: a retained zero still has a gradient.
            std::vector<float>zero(513,0.f),go(513,1.f);std::vector<uint8_t>mask;std::vector<std::vector<float>>out,gi;
            RuntimeForwardContext fc;fc.output_mask=&mask;MimirRng::setSeed(4242);
            require(runtime.forwardLayerWithContext({&zero},out,l,true,fc),"GPU dropout context forward");
            require(mask.size()==zero.size(),"GPU saves dropout mask");RuntimeBackwardContext bc;bc.output=&out[0];bc.output_mask=&mask;
            require(runtime.backwardLayerWithContext({&zero},{&go},gi,l,true,bc),"GPU dropout context backward");
            const float ap=-1.6732632423543772848f*1.0507009873554804934f;
            const float scale=probability==1?0:(type==LayerType::AlphaDropout?1/std::sqrt((1-probability)*(1+probability*ap*ap)):1/(1-probability));
            for(size_t i=0;i<zero.size();++i)require(std::abs(gi[0][i]-(mask[i]?scale:0.f))<1e-5f,"retained zero gradient");
        }
        {
            Layer l;l.type_enum=LayerType::Reparameterize;std::vector<float>mu(1027),lv(1027);
            for(int i=0;i<1027;++i){mu[i]=std::sin(i*.1f);lv[i]=(i%5==0)?-21.f:((i%7==0)?21.f:std::cos(i*.2f));}
            parity_inputs(runtime,l,{&mu,&lv});std::vector<std::vector<float>>out;
            RuntimeForwardContext fc;fc.stochastic_latent=false;require(runtime.forwardLayerWithContext({&mu,&lv},out,l,true,fc),"GPU deterministic latent context");compare(out[0],mu);
        }
        for(auto type:{LayerType::SelfAttention,LayerType::MultiHeadAttention,LayerType::CrossAttention})for(bool causal:{false,true})for(int biases:{0,1,4}){
            Layer l;l.type_enum=type;l.embed_dim=8;l.in_features=8;l.seq_len=3;l.num_heads=2;l.causal=causal;
            l.weights.resize(256+8*(type==LayerType::CrossAttention?(biases?1:0):biases));
            for(size_t i=0;i<l.weights.size();++i)l.weights[i]=std::sin(i*.17f)*.2f;
            std::vector<float>q(24),kv(40);for(int i=0;i<24;++i)q[i]=std::cos(i*.31f);for(int i=0;i<40;++i)kv[i]=std::sin(i*.21f);
            if(type==LayerType::CrossAttention)parity_inputs(runtime,l,{&q,&kv});else parity(runtime,l,q);
        }
        for(auto type:{LayerType::RNN,LayerType::GRU,LayerType::LSTM})for(bool bias:{false,true})for(int steps:{1,5}){
            Layer l;l.type_enum=type;l.seq_len=steps;l.in_features=3;l.out_features=4;l.use_bias=bias;
            const int gates=type==LayerType::RNN?1:(type==LayerType::GRU?3:4);l.weights.resize(gates*4*(3+4+(bias?2:0)));
            for(size_t i=0;i<l.weights.size();++i)l.weights[i]=std::sin(i*.17f)*.2f;
            std::vector<float>x(steps*3);for(size_t i=0;i<x.size();++i)x[i]=std::cos(i*.31f);parity(runtime,l,x);
        }
        for(bool agnostic:{false,true})for(int limit:{0,2}){
            Layer l;l.type_enum=LayerType::NMS;l.nms_iou_threshold=.3f;l.nms_score_threshold=.1f;l.nms_max_detections=limit;l.nms_class_agnostic=agnostic;
            std::vector<float>boxes{0,0,2,2,0,0,2,2,1,1,3,3,8,8,9,9,8,8,9,9};
            std::vector<float>scores{.9f,.9f,.8f,.7f,.6f},classes{0,1,0,0,0};
            std::vector<std::vector<float>>a,b;std::vector<const std::vector<float>*>in{&boxes,&scores,&classes};
            require(runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(l,in,false)),"native NMS capability");
            require(runtime.forwardLayer(in,a,l,false),"GPU NMS");require(RuntimeLayerDispatch::cpu_forward_layer(in,b,l,false),"CPU NMS");compare(a[0],b[0]);
            require(runtimeCapabilityIsNative(runtime.queryBackwardCapability(LayerType::NMS)),"native NMS stop-gradient backward");
            std::vector<float> go(a[0].size(),.7f);std::vector<std::vector<float>> gi;
            require(runtime.backwardLayer(in,{&go},gi,l,true),"NMS backward succeeds");
            require(gi.size()==in.size(),"NMS gradient arity");
            {
                CpuRuntime cpu;RuntimeConfig config;require(cpu.initialize(config),"CPU initializes");
                auto& router=RuntimeRouter::instance();const std::string name=runtime.name();
                router.setActivators({}, {}, {}, {}, {});
                router.setRuntimes(nullptr,name=="CUDA"?&runtime:nullptr,name=="VULKAN"?&runtime:nullptr,name=="OPENCL"?&runtime:nullptr,&cpu);
                AbstractRuntime* selected=nullptr;
                require(router.dispatchBackwardLayer(in,{&go},gi,l,true,&selected),"NMS routed backward");
                require(selected==&runtime,"NMS backward must select GPU");
                router.setRuntimes(nullptr,nullptr,nullptr,nullptr,nullptr);
            }

            for(size_t i=0;i<in.size();++i)compare(gi[i],std::vector<float>(in[i]->size(),0.f));
            std::vector<std::vector<float>> cpu_gi;
            require(RuntimeLayerDispatch::cpu_backward_layer(in,{&go},cpu_gi,l,true),"CPU NMS backward reference");
            require(cpu_gi.size()==gi.size(),"CPU/GPU NMS gradient arity");
            for(size_t i=0;i<gi.size();++i)compare(gi[i],cpu_gi[i]);
            const float original=scores[2];scores[2]+=1e-4f;
            std::vector<std::vector<float>> perturbed;
            require(runtime.forwardLayer(in,perturbed,l,false),"NMS finite difference forward");compare(perturbed[0],a[0]);scores[2]=original;
            auto without_classes=in;without_classes.resize(2);
            require(runtime.forwardLayer(without_classes,perturbed,l,false),"NMS without classes");
            std::vector<float> go2(perturbed[0].size(),-.2f);
            require(runtime.backwardLayer(without_classes,{&go2},gi,l,true),"NMS backward without classes");
            require(gi.size()==2,"NMS two-input gradient arity");
            compare(gi[0],std::vector<float>(boxes.size(),0.f));compare(gi[1],std::vector<float>(scores.size(),0.f));
            std::vector<float> bad(scores.size()+1,1.f);
            require(!runtime.backwardLayer(in,{&bad},gi,l,true),"NMS rejects mismatched gradient shape");
            boxes.clear();scores.clear();classes.clear();require(runtime.forwardLayer(in,a,l,false) && a.size()==1 && a[0].empty(),"empty native NMS");
            go.clear();require(runtime.backwardLayer(in,{&go},gi,l,true),"empty NMS backward");
            std::cout<<"NMS_STOP_GRADIENT_OK\n";
            require(gi.size()==3 && gi[0].empty() && gi[1].empty() && gi[2].empty(),"empty NMS gradients");
        }
        {
            Layer l;l.type_enum=LayerType::BatchNorm2d;l.in_channels=3;l.input_height=2;l.input_width=257;l.affine=true;l.use_bias=true;l.eps=1e-5f;l.weights={.3f,.7f,1.1f,.1f,.2f,-.2f};
            std::vector<float>x(3084);for(size_t i=0;i<x.size();++i)x[i]=std::sin(i*.37f);parity(runtime,l,x);
        }
        {
            Layer l;l.type_enum=LayerType::CrossAttention;l.embed_dim=l.in_features=8;l.seq_len=3;l.num_heads=2;l.causal=false;l.weights.resize(256);
            for(size_t i=0;i<l.weights.size();++i)l.weights[i]=std::sin(i*.17f)*.2f;
            std::vector<float>q(24),kv(513*8);for(size_t i=0;i<q.size();++i)q[i]=std::cos(i*.31f);for(size_t i=0;i<kv.size();++i)kv[i]=std::sin(i*.21f);parity_inputs(runtime,l,{&q,&kv});
            l.in_features=5;l.weights.resize(208);kv.resize(5*5);
            std::vector<std::vector<float>>a,b;std::vector<const std::vector<float>*>in{&q,&kv};
            require(runtime.forwardLayer(in,a,l,false),"GPU attention with distinct KV dimensions");require(RuntimeLayerDispatch::cpu_forward_layer(in,b,l,false),"CPU distinct KV reference");compare(a[0],b[0]);
            std::vector<float>dy(q.size());require(!runtimeCapabilityIsNative(runtime.queryBackwardOperationCapability(l,in,{&dy},true)),"respect CPU backward KV dimension limitation");
        }
        {
            Layer l;l.type_enum=LayerType::NMS;l.nms_iou_threshold=.3f;l.nms_score_threshold=.1f;l.nms_max_detections=0;l.nms_class_agnostic=false;
            const float nan=std::numeric_limits<float>::quiet_NaN();
            std::vector<float>boxes{0,0,2,2,0,0,2,2,nan,0,3,3,8,8,7,7,8,8,9,9},scores{.9f,nan,.8f,.7f,.6f},classes{0,1,0,0,nan};
            std::vector<std::vector<float>>a,b;std::vector<const std::vector<float>*>in{&boxes,&scores,&classes};
            require(runtime.forwardLayer(in,a,l,false),"GPU NMS filtering");require(RuntimeLayerDispatch::cpu_forward_layer(in,b,l,false),"CPU NMS filtering");compare(a[0],b[0]);
            l.nms_iou_threshold=1.1f;require(!runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(l,in,false)),"reject invalid NMS threshold");
        }
        cfg.norm_enabled=false; require(runtime.initialize(cfg),"reconfigure runtime");
        Layer disabled; disabled.type_enum=LayerType::GroupNorm; disabled.in_channels=2;
        disabled.input_height=1; disabled.input_width=2; disabled.num_groups=1; disabled.affine=false;
        std::vector<float> x(4,1.f); std::vector<const std::vector<float>*> inputs{&x};
        require(!runtimeCapabilityIsNative(runtime.queryForwardOperationCapability(disabled,inputs,true)),"respect norm disable flag");
        return 0;
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
