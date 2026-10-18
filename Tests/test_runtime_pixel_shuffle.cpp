#include "Model.hpp"
#include "runtimes/cpu/CpuRuntime.hpp"
#include "test_utils.hpp"

#include <vector>

int main() {
    CpuRuntime runtime;
    RuntimeConfig config;
    config.backend = "CPU";
    TASSERT_TRUE(runtime.initialize(config));

    Layer layer("pixel_shuffle", "PixelShuffle", 0);
    layer.scale_h = 2.0f;
    layer.in_channels = 8;
    layer.input_height = 2;
    layer.input_width = 2;

    std::vector<float> input(32);
    for (size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(index + 1);
    }
    const std::vector<const std::vector<float>*> inputs{&input};
    std::vector<std::vector<float>> outputs;
    TASSERT_TRUE(runtime.supportsForwardLayerType(LayerType::PixelShuffle));
    TASSERT_TRUE(runtime.forwardLayer(inputs, outputs, layer, false));
    TASSERT_TRUE(outputs.size() == 1);
    TASSERT_TRUE(outputs[0].size() == 32);

    const int upscale = 2;
    const int input_height = 2;
    const int input_width = 2;
    const int output_channels = 2;
    const int output_height = 4;
    const int output_width = 4;
    for (int channel = 0; channel < output_channels; ++channel) {
        for (int output_y = 0; output_y < output_height; ++output_y) {
            for (int output_x = 0; output_x < output_width; ++output_x) {
                const int input_channel = channel * upscale * upscale +
                    (output_y % upscale) * upscale + output_x % upscale;
                const size_t input_index =
                    (static_cast<size_t>(input_channel) * input_height + output_y / upscale) *
                        input_width + output_x / upscale;
                const size_t output_index =
                    (static_cast<size_t>(channel) * output_height + output_y) *
                        output_width + output_x;
                TASSERT_NEAR(outputs[0][output_index], input[input_index], 1e-6f);
            }
        }
    }

    std::vector<float> grad_output(outputs[0].size());
    for (size_t index = 0; index < grad_output.size(); ++index) {
        grad_output[index] = 100.0f + static_cast<float>(index);
    }
    const std::vector<const std::vector<float>*> grad_outputs{&grad_output};
    std::vector<std::vector<float>> grad_inputs;
    TASSERT_TRUE(runtime.supportsBackwardLayerType(LayerType::PixelShuffle));
    TASSERT_TRUE(runtime.backwardLayer(inputs, grad_outputs, grad_inputs, layer, true));
    TASSERT_TRUE(grad_inputs.size() == 1);
    TASSERT_TRUE(grad_inputs[0].size() == input.size());
    for (int channel = 0; channel < output_channels; ++channel) {
        for (int output_y = 0; output_y < output_height; ++output_y) {
            for (int output_x = 0; output_x < output_width; ++output_x) {
                const int input_channel = channel * upscale * upscale +
                    (output_y % upscale) * upscale + output_x % upscale;
                const size_t input_index =
                    (static_cast<size_t>(input_channel) * input_height + output_y / upscale) *
                        input_width + output_x / upscale;
                const size_t output_index =
                    (static_cast<size_t>(channel) * output_height + output_y) *
                        output_width + output_x;
                TASSERT_NEAR(grad_inputs[0][input_index], grad_output[output_index], 1e-6f);
            }
        }
    }

    std::vector<float> truncated_input(input.size() - 1, 1.0f);
    const std::vector<const std::vector<float>*> truncated_inputs{&truncated_input};
    outputs.clear();
    TASSERT_TRUE(!runtime.forwardLayer(truncated_inputs, outputs, layer, false));

    layer.scale_h = 1.5f;
    outputs.clear();
    TASSERT_TRUE(!runtime.forwardLayer(inputs, outputs, layer, false));
    grad_inputs.clear();
    TASSERT_TRUE(!runtime.backwardLayer(inputs, grad_outputs, grad_inputs, layer, true));

    runtime.shutdown();
    return 0;
}