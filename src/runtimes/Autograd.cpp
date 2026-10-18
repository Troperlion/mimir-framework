#include "Autograd.hpp"
#include "Layers.hpp"
#include "runtimes/RuntimeRouter.hpp"

namespace Autograd {
namespace {
std::vector<float> layer_backward(Layer& layer, const std::vector<float>& input,
                                 const std::vector<float>& gradient) {
    std::vector<std::vector<float>> result;
    if (!RuntimeRouter::instance().dispatchBackwardLayer({&input}, {&gradient}, result, layer, true) || result.size()!=1)
        throw std::runtime_error("Autograd: no runtime could compute " + layer.type + " backward");
    return std::move(result[0]);
}
}
std::vector<float> layernorm_backward(const std::vector<float>& grad_output,
                                     const std::vector<float>& input,
                                     const std::vector<float>& normalized) {
    if (grad_output.size()!=input.size() || normalized.size()!=input.size())
        throw std::invalid_argument("Autograd::layernorm_backward: size mismatch");
    if (input.empty()) return {};
    Layer layer("autograd/layernorm", "LayerNorm", 0);
    layer.in_features=static_cast<int>(input.size());
    layer.affine=false;
    layer.use_bias=false;
    return layer_backward(layer,input,grad_output);
}
float gelu_backward(float x, float grad_output) {
    Layer layer("autograd/gelu", "GELU", 0);
    return layer_backward(layer,{x},{grad_output}).at(0);
}
std::vector<float> residual_backward(const std::vector<float>& grad_output) {
    Layer layer("autograd/identity", "Identity", 0);
    return layer_backward(layer,grad_output,grad_output);
}
} // namespace Autograd
