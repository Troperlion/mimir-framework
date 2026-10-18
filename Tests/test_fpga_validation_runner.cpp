#include "runtimes/fpga/FpgaValidationRunner.hpp"
#include "test_utils.hpp"

#include <atomic>
#include <limits>
#include <future>
#include <vector>

int main() {
    std::vector<int8_t> quantized;
    TASSERT_TRUE(FpgaValidationRunner::quantizeNormalizedInput(
        {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f}, quantized));
    TASSERT_TRUE(quantized == std::vector<int8_t>({
        -127, -127, -64, 0, 64, 127, 127
    }));
    TASSERT_TRUE(!FpgaValidationRunner::quantizeNormalizedInput({}, quantized));
    TASSERT_TRUE(!FpgaValidationRunner::quantizeNormalizedInput(
        {std::numeric_limits<float>::quiet_NaN()}, quantized));
    TASSERT_TRUE(quantized.empty());

    FpgaValidationRunner::DatasetInput dataset_input;
    dataset_input.image_rgb = {0, 0, 0, 255, 255, 255};
    dataset_input.text_token_ids = {0, 3};
    dataset_input.tokenizer_vocab_size = 4;
    // Deux frames stereo S16LE: silence puis (32767, 32767).
    dataset_input.audio_pcm_s16le_stereo = {
        0, 0, 0, 0, 0xff, 0x7f, 0xff, 0x7f
    };
    dataset_input.video_rgb = {255, 255, 255, 0, 0, 0};
    std::vector<float> composed;
    TASSERT_TRUE(FpgaValidationRunner::composeDatasetInput(
        dataset_input, 8, composed));
    TASSERT_TRUE(composed.size() == 8);
    TASSERT_TRUE(composed[0] == -1.0f && composed[1] == 1.0f);
    TASSERT_TRUE(composed[2] == -1.0f && composed[3] == 1.0f);
    TASSERT_TRUE(composed[4] == 0.0f && composed[5] > 0.99f);
    TASSERT_TRUE(composed[6] == 1.0f && composed[7] == -1.0f);

    std::promise<void> started;
    std::promise<void> release;
    std::shared_future<void> release_future = release.get_future().share();
    std::atomic<bool> first_job{true};

    FpgaValidationRunner runner([&](
        const std::vector<int8_t>& input,
        std::vector<int32_t>& output
    ) {
        if (first_job.exchange(false)) started.set_value();
        release_future.wait();
        int32_t sum = 0;
        for (const int8_t value : input) sum += value;
        output = {sum};
        return true;
    });

    TASSERT_TRUE(runner.submit(42, {1, 2, 3, 4}, {10}, "alice"));
    started.get_future().wait();
    TASSERT_TRUE(runner.busy());
    TASSERT_TRUE(!runner.submit(43, {5, 6}));

    release.set_value();
    runner.wait();
    TASSERT_TRUE(!runner.busy());

    auto result = runner.poll();
    TASSERT_TRUE(result.has_value());
    TASSERT_TRUE(result->training_step == 42);
    TASSERT_TRUE(result->sample_id == "alice");
    TASSERT_TRUE(result->input_checksum == 10);
    TASSERT_TRUE(result->success);
    TASSERT_TRUE(result->exact);
    TASSERT_TRUE(result->outputs == std::vector<int32_t>({10}));
    TASSERT_TRUE(!runner.poll().has_value());

    TASSERT_TRUE(runner.submit(44, {7, -2}));
    runner.wait();
    result = runner.poll();
    TASSERT_TRUE(result.has_value());
    TASSERT_TRUE(result->training_step == 44);
    TASSERT_TRUE(result->outputs == std::vector<int32_t>({5}));

    TASSERT_TRUE(runner.submitNormalized(47, {-1.0f, 0.5f}));
    runner.wait();
    result = runner.poll();
    TASSERT_TRUE(result.has_value());
    TASSERT_TRUE(result->input_checksum == -63);

    TASSERT_TRUE(runner.submit(45, {3, 4}, {99}, "mismatch"));
    runner.wait();
    result = runner.poll();
    TASSERT_TRUE(result.has_value());
    TASSERT_TRUE(result->success);
    TASSERT_TRUE(!result->exact);

    FpgaValidationRunner failing_runner([](
        const std::vector<int8_t>&,
        std::vector<int32_t>&
    ) -> bool {
        throw 1;
    });
    TASSERT_TRUE(failing_runner.submit(46, {1}));
    failing_runner.wait();
    result = failing_runner.poll();
    TASSERT_TRUE(result.has_value());
    TASSERT_TRUE(result->training_step == 46);
    TASSERT_TRUE(!result->success);
    TASSERT_TRUE(!result->exact);
    TASSERT_TRUE(result->outputs.empty());
    return 0;
}
