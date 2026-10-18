#include "runtimes/fpga/FpgaValidationRunner.hpp"

#include "runtimes/fpga/FpgaRuntime.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

FpgaValidationRunner::FpgaValidationRunner(FpgaRuntime& runtime)
    : FpgaValidationRunner([&runtime](
          const std::vector<int8_t>& input,
          std::vector<int32_t>& output
      ) {
          return runtime.int8MatrixVector(input.data(), input.size(), output);
      }) {}

FpgaValidationRunner::FpgaValidationRunner(Execute execute)
    : execute_(std::move(execute)), worker_(&FpgaValidationRunner::workerLoop, this) {}

FpgaValidationRunner::~FpgaValidationRunner() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
}

bool FpgaValidationRunner::submit(
    uint64_t training_step,
    std::vector<int8_t> input,
    std::vector<int32_t> expected,
    std::string sample_id
) {
    if (input.empty()) return false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || running_ || pending_ || result_) return false;
        pending_ = Job{
            training_step,
            std::move(input),
            std::move(expected),
            std::move(sample_id)
        };
    }
    condition_.notify_one();
    return true;
}

bool FpgaValidationRunner::quantizeNormalizedInput(
    const std::vector<float>& input,
    std::vector<int8_t>& quantized
) {
    quantized.clear();
    if (input.empty()) return false;

    quantized.reserve(input.size());
    for (const float value : input) {
        if (!std::isfinite(value)) {
            quantized.clear();
            return false;
        }
        const float normalized = std::clamp(value, -1.0f, 1.0f);
        const int converted = static_cast<int>(std::lround(normalized * 127.0f));
        quantized.push_back(static_cast<int8_t>(converted));
    }
    return true;
}

namespace {
template <typename Sample>
void sampleModality(size_t count, size_t output_count, Sample sample,
                    std::vector<float>& output) {
    for (size_t i = 0; i < output_count; ++i) {
        const size_t index = std::min(count - 1, (i * count) / output_count);
        output.push_back(std::clamp(sample(index), -1.0f, 1.0f));
    }
}
} // namespace

bool FpgaValidationRunner::composeDatasetInput(
    const DatasetInput& dataset,
    size_t target_elements,
    std::vector<float>& normalized
) {
    normalized.clear();
    if (target_elements == 0) return false;

    const bool has_image = !dataset.image_rgb.empty() &&
        (dataset.image_rgb.size() % 3) == 0;
    const bool has_text = !dataset.text_token_ids.empty() &&
        dataset.tokenizer_vocab_size > 1;
    const bool has_audio = dataset.audio_pcm_s16le_stereo.size() >= 4 &&
        (dataset.audio_pcm_s16le_stereo.size() % 4) == 0;
    const bool has_video = !dataset.video_rgb.empty() &&
        (dataset.video_rgb.size() % 3) == 0;
    const size_t modality_count = static_cast<size_t>(has_image) +
        static_cast<size_t>(has_text) + static_cast<size_t>(has_audio) +
        static_cast<size_t>(has_video);
    if (modality_count == 0 || target_elements < modality_count) return false;

    normalized.reserve(target_elements);
    size_t remaining_modalities = modality_count;
    size_t remaining_elements = target_elements;
    auto allocation = [&]() {
        const size_t count = remaining_elements / remaining_modalities;
        remaining_elements -= count;
        --remaining_modalities;
        return count;
    };

    if (has_image) {
        const size_t pixels = dataset.image_rgb.size() / 3;
        sampleModality(pixels, allocation(), [&](size_t pixel) {
            const size_t offset = pixel * 3;
            const float gray = (static_cast<float>(dataset.image_rgb[offset]) +
                static_cast<float>(dataset.image_rgb[offset + 1]) +
                static_cast<float>(dataset.image_rgb[offset + 2])) / 3.0f;
            return gray / 127.5f - 1.0f;
        }, normalized);
    }
    if (has_text) {
        sampleModality(dataset.text_token_ids.size(), allocation(), [&](size_t index) {
            const int token = std::clamp(
                dataset.text_token_ids[index], 0,
                static_cast<int>(dataset.tokenizer_vocab_size - 1));
            return 2.0f * static_cast<float>(token) /
                static_cast<float>(dataset.tokenizer_vocab_size - 1) - 1.0f;
        }, normalized);
    }
    if (has_audio) {
        const size_t frames = dataset.audio_pcm_s16le_stereo.size() / 4;
        sampleModality(frames, allocation(), [&](size_t frame) {
            const size_t offset = frame * 4;
            auto read_s16le = [&](size_t at) {
                const uint16_t bits = static_cast<uint16_t>(
                    static_cast<uint16_t>(dataset.audio_pcm_s16le_stereo[at]) |
                    (static_cast<uint16_t>(dataset.audio_pcm_s16le_stereo[at + 1]) << 8));
                return static_cast<int16_t>(bits);
            };
            const float mono = 0.5f * (static_cast<float>(read_s16le(offset)) +
                static_cast<float>(read_s16le(offset + 2)));
            return mono / 32768.0f;
        }, normalized);
    }
    if (has_video) {
        const size_t pixels = dataset.video_rgb.size() / 3;
        sampleModality(pixels, allocation(), [&](size_t pixel) {
            const size_t offset = pixel * 3;
            const float gray = (static_cast<float>(dataset.video_rgb[offset]) +
                static_cast<float>(dataset.video_rgb[offset + 1]) +
                static_cast<float>(dataset.video_rgb[offset + 2])) / 3.0f;
            return gray / 127.5f - 1.0f;
        }, normalized);
    }
    return normalized.size() == target_elements;
}

bool FpgaValidationRunner::submitNormalized(
    uint64_t training_step,
    const std::vector<float>& input,
    std::vector<int32_t> expected,
    std::string sample_id
) {
    std::vector<int8_t> quantized;
    if (!quantizeNormalizedInput(input, quantized)) return false;
    return submit(
        training_step,
        std::move(quantized),
        std::move(expected),
        std::move(sample_id)
    );
}

std::optional<FpgaValidationRunner::Result> FpgaValidationRunner::poll() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!result_) return std::nullopt;
    std::optional<Result> result = std::move(result_);
    result_.reset();
    return result;
}

bool FpgaValidationRunner::busy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return running_ || pending_.has_value();
}

void FpgaValidationRunner::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return !running_ && !pending_; });
}

void FpgaValidationRunner::workerLoop() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return stopping_ || pending_.has_value(); });
            if (stopping_ && !pending_) return;
            job = std::move(*pending_);
            pending_.reset();
            running_ = true;
        }

        Result result;
        result.training_step = job.training_step;
        result.sample_id = std::move(job.sample_id);
        for (const int8_t value : job.input) result.input_checksum += value;
        try {
            result.success = execute_ && execute_(job.input, result.outputs);
        } catch (...) {
            result.success = false;
            result.outputs.clear();
        }
        result.exact = result.success &&
            (job.expected.empty() || result.outputs == job.expected);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            result_ = std::move(result);
            running_ = false;
        }
        condition_.notify_all();
    }
}
