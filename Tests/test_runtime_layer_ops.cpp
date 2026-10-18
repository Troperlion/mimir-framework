#include "test_utils.hpp"
#include "runtimes/LayerOps.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
    RuntimeLayerOps::setHardwareAcceleration(false);

    {
        std::vector<float> values = {-2.0f, 0.0f, 3.0f};
        RuntimeLayerOps::computeActivation(values, "leaky_relu", 0.1f);
        TASSERT_NEAR(values[0], -0.2f, 1e-6f);
        TASSERT_NEAR(values[1], 0.0f, 1e-6f);
        TASSERT_NEAR(values[2], 3.0f, 1e-6f);

        RuntimeLayerOps::computeActivation(values, "softmax");
        TASSERT_NEAR(values[0] + values[1] + values[2], 1.0f, 1e-6f);
        TASSERT_TRUE(values[2] > values[1] && values[1] > values[0]);
    }

    {
        RuntimeLayerOps::LayerParams params;
        params.in_features = 2;
        params.out_features = 2;
        params.weights = {1.0f, 2.0f, 3.0f, 4.0f};
        params.bias = {0.5f, -0.5f};

        std::vector<float> output;
        RuntimeLayerOps::computeLinear({2.0f, -1.0f}, output, params);
        TASSERT_TRUE(output.size() == 2);
        TASSERT_NEAR(output[0], 0.5f, 1e-6f);
        TASSERT_NEAR(output[1], 1.5f, 1e-6f);
    }

    {
        const std::vector<float> left = {1.0f, 4.0f, -2.0f};
        const std::vector<float> right = {3.0f, 2.0f, 5.0f};
        std::vector<float> merged;
        RuntimeLayerOps::branchMerge(left, right, merged, MergeOperation::AVERAGE);
        TASSERT_TRUE(merged.size() == 3);
        TASSERT_NEAR(merged[0], 2.0f, 1e-6f);
        TASSERT_NEAR(merged[1], 3.0f, 1e-6f);
        TASSERT_NEAR(merged[2], 1.5f, 1e-6f);

        std::vector<std::vector<float>> split;
        RuntimeLayerOps::branchSplit({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, split, {2, 3});
        TASSERT_TRUE(split.size() == 2);
        TASSERT_TRUE(split[0] == std::vector<float>({1.0f, 2.0f}));
        TASSERT_TRUE(split[1] == std::vector<float>({3.0f, 4.0f, 5.0f}));

        bool rejected_merge = false;
        try {
            RuntimeLayerOps::branchMerge({1.0f}, {1.0f, 2.0f}, merged, MergeOperation::ADD);
        } catch (const std::invalid_argument&) {
            rejected_merge = true;
        }
        TASSERT_TRUE(rejected_merge);

        bool rejected_split = false;
        try {
            RuntimeLayerOps::branchSplit({1.0f, 2.0f}, split, {1});
        } catch (const std::invalid_argument&) {
            rejected_split = true;
        }
        TASSERT_TRUE(rejected_split);
    }

    RuntimeLayerOps::setHardwareAcceleration(true);
    return 0;
}