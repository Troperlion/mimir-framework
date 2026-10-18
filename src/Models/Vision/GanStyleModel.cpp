#include "GanStyleModel.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

GanStyleModel::GanStyleModel() { setModelName("Gan_Style"); setHasEncoder(false); }

void GanStyleModel::buildFromConfig(const Config& cfg) {
    if (cfg.image_w<3 || cfg.image_h<3 || cfg.image_c!=3 ||
        cfg.base_channels<1 || cfg.discriminator_channels<1 ||
        !std::isfinite(cfg.content_weight) || cfg.content_weight<0 ||
        !std::isfinite(cfg.discriminator_lr) || cfg.discriminator_lr<=0)
        throw std::invalid_argument("Gan_Style: invalid dimensions or loss/learning rate");
    getMutableLayers().clear();
    modelConfig={{"type","Gan_Style"},{"task","image_style_transfer"},
        {"image_w",cfg.image_w},{"image_h",cfg.image_h},{"image_c",3},
        {"base_channels",cfg.base_channels},{"discriminator_channels",cfg.discriminator_channels},
        {"content_weight",cfg.content_weight},{"discriminator_lr",cfg.discriminator_lr},
        {"input_dim",cfg.image_w*cfg.image_h*3},{"output_dim",cfg.image_w*cfg.image_h*3},
        {"bbox_input_layout","CHW"},{"bbox_output_layout","CHW"},
        {"inference_output_tensor","g/image"}};
    auto network=[&](const std::string& prefix,int width,int output_channels) {
        std::string input="__input__";
        int in=3;
        for (int i=0;i<3;++i) {
            const int out=i==2?output_channels:width;
            const std::string name=prefix+"/conv"+std::to_string(i+1);
            push(name,"Conv2d",static_cast<size_t>(in)*out*9+out);
            auto* conv=getLayerByName(name);
            conv->inputs={input}; conv->output=name;
            conv->in_channels=in; conv->out_channels=out;
            conv->input_height=cfg.image_h; conv->input_width=cfg.image_w;
            conv->kernel_size=conv->kernel_h=conv->kernel_w=3;
            conv->stride=conv->stride_h=conv->stride_w=1;
            conv->padding=conv->pad_h=conv->pad_w=1; conv->use_bias=true;
            input=name; in=out;
            if (i<2 || prefix=="g") {
                const std::string act=name+"/act";
                push(act,i==2?"Tanh":(prefix=="g"?"ReLU":"LeakyReLU"),0);
                auto* layer=getLayerByName(act);
                layer->inputs={input}; layer->output=(i==2?"g/image":act); layer->alpha=0.2f;
                input=layer->output;
            }
        }
        return input;
    };
    network("g",cfg.base_channels,3);
    const auto logits=network("d",cfg.discriminator_channels,1);
    push("gan_style/output","Identity",0);
    auto* output=getLayerByName("gan_style/output");
    output->inputs={"g/image"}; output->output="x";
    modelConfig["discriminator_output_tensor"]=logits;
    static_plan_.built=false;
}

void GanStyleModel::selectOutput(bool discriminator) {
    getLayerByName("gan_style/output")->inputs={discriminator?
        modelConfig.at("discriminator_output_tensor").get<std::string>():"g/image"};
    static_plan_.built=false;
}

std::optional<Model::TrainStepResult> GanStyleModel::trainStep(const TrainStepRequest& request) {
    auto input=request.float_inputs.find("__input__");
    if (input==request.float_inputs.end() || !input->second || !request.target || !request.optimizer)
        return std::nullopt;
    if (request.mode!=TrainStepMode::Optimize || request.grad_scale!=1.f)
        throw std::invalid_argument("Gan_Style: alternating optimization requires Optimize and grad_scale=1");
    const size_t count=static_cast<size_t>(modelConfig.at("image_w").get<int>())*
        modelConfig.at("image_h").get<int>()*3;
    auto validate=[&](const std::vector<float>& image) {
        if(image.size()!=count) throw std::invalid_argument("Gan_Style: expected CHW RGB image");
        for(float value:image) if(!std::isfinite(value)) throw std::invalid_argument("Gan_Style: non-finite image");
    };
    validate(*input->second); validate(*request.target);
    const float content=modelConfig.at("content_weight").get<float>();
    const float dlr=modelConfig.at("discriminator_lr").get<float>();
    if (!std::isfinite(request.learning_rate) || request.learning_rate<=0 ||
        !std::isfinite(content) || content<0 || !std::isfinite(dlr) || dlr<=0)
        throw std::invalid_argument("Gan_Style: invalid learning rate or content weight");
    if(request.loss_weights || request.relations)
        throw std::invalid_argument("Gan_Style: spatial loss weights/relations are not supported");
    if(parametersFrozen()) throw std::runtime_error("Gan_Style: parameters frozen");
    const auto frozen=modelConfig.value("frozen_layer_prefixes",json::array());
    if(!frozen.is_array()) throw std::invalid_argument("Gan_Style: frozen_layer_prefixes must be an array");
    // Restore the public inference graph and caller configuration on all exits.
    auto restore=[&]() { selectOutput(false); modelConfig["frozen_layer_prefixes"]=frozen; };
    try {
        TrainStepResult result;
        const auto fake=forwardPass(*input->second,false);
        Optimizer discriminator;
        discriminator.type=OptimizerType::SGD;
        discriminator.decay_strategy=LRDecayStrategy::NONE;
        discriminator.weight_decay=0.f;
        selectOutput(true);
        auto dstep=[&](const std::vector<float>& image,float label) {
            zeroGradients();
            const auto logits=forwardPass(image,true);
            const std::vector<float> target(logits.size(),label);
            const float loss=computeLoss(logits,target,"mse");
            backwardPass(computeLossGradient(logits,target,"mse"));
            modelConfig["frozen_layer_prefixes"]=frozen;
            modelConfig["frozen_layer_prefixes"].push_back("g/");
            optimizerStep(discriminator,dlr);
            return loss;
        };
        const float dr=dstep(*request.target,1.f);
        const float df=dstep(fake,0.f);
        // Differentiate D w.r.t. the generated image without an optimizer step.
        zeroGradients();
        const auto logits=forwardPass(fake,true);
        const std::vector<float> labels(logits.size(),1.f);
        const float adversarial=computeLoss(logits,labels,"mse");
        backwardPass(computeLossGradient(logits,labels,"mse"));
        if(!hasLastInputGradient() || getLastInputGradient().size()!=count)
            throw std::runtime_error("Gan_Style: discriminator input gradient unavailable");
        auto gradient=getLastInputGradient();
        const float l1=computeLoss(fake,*input->second,"mae");
        const auto content_gradient=computeLossGradient(fake,*input->second,"mae");
        for(size_t i=0;i<count;++i) gradient[i]+=content*content_gradient[i];
        selectOutput(false);
        zeroGradients();
        forwardPass(*input->second,true);
        backwardPass(gradient);
        double norm=0;
        for(const auto& layer:getLayers()) {
            if(layer.name.rfind("g/",0)!=0) continue;
            for(float value:layer.grad_weights) {
                if(!std::isfinite(value)) throw std::runtime_error("Gan_Style: non-finite generator gradient");
                norm+=static_cast<double>(value)*value;
                result.grad_max_abs=std::max(result.grad_max_abs,std::abs(value));
            }
        }
        result.grad_norm=static_cast<float>(std::sqrt(norm));
        modelConfig["frozen_layer_prefixes"]=frozen;
        modelConfig["frozen_layer_prefixes"].push_back("d/");
        optimizerStep(*request.optimizer,request.learning_rate);
        result.loss=adversarial+content*l1;
        result.metrics={{"d_real",dr},{"d_fake",df},{"g_adv",adversarial},{"content_l1",l1}};
        if(request.capture_prediction) result.prediction=fake;
        restore();
        return result;
    } catch (...) { restore(); throw; }
}
