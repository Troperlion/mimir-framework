#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

class FpgaRuntime;

class FpgaValidationRunner {
public:
    struct DatasetInput {
        std::vector<uint8_t> image_rgb;
        std::vector<int> text_token_ids;
        size_t tokenizer_vocab_size = 0;
        std::vector<uint8_t> audio_pcm_s16le_stereo;
        std::vector<uint8_t> video_rgb;
    };

    struct Result {
        uint64_t training_step = 0;
        std::string sample_id;
        int input_checksum = 0;
        bool success = false;
        bool exact = false;
        std::vector<int32_t> outputs;
    };

    using Execute = std::function<bool(
        const std::vector<int8_t>&,
        std::vector<int32_t>&
    )>;

    explicit FpgaValidationRunner(FpgaRuntime& runtime);
    explicit FpgaValidationRunner(Execute execute);
    ~FpgaValidationRunner();

    FpgaValidationRunner(const FpgaValidationRunner&) = delete;
    FpgaValidationRunner& operator=(const FpgaValidationRunner&) = delete;

    bool submit(
        uint64_t training_step,
        std::vector<int8_t> input,
        std::vector<int32_t> expected = {},
        std::string sample_id = {}
    );

    // Convertit une entree Mimir normalisee dans [-1, 1] vers le format
    // d'activation signe attendu par le noyau FPGA. La quantification est
    // symetrique (zero-point 0), avec saturation dans [-127, 127].
    // Retourne false si l'entree est vide ou contient une valeur non finie.
    static bool quantizeNormalizedInput(
        const std::vector<float>& input,
        std::vector<int8_t>& quantized
    );

    // Compose les representations deja chargees par Mimir en un vecteur
    // normalise de taille fixe pour le probe matriciel FPGA. Chaque modalite
    // presente recoit une part du vecteur; aucune donnee compressee brute n'est
    // acceptee ici.
    static bool composeDatasetInput(
        const DatasetInput& dataset,
        size_t target_elements,
        std::vector<float>& normalized
    );

    bool submitNormalized(
        uint64_t training_step,
        const std::vector<float>& input,
        std::vector<int32_t> expected = {},
        std::string sample_id = {}
    );
    std::optional<Result> poll();
    bool busy() const;
    void wait();

private:
    struct Job {
        uint64_t training_step = 0;
        std::vector<int8_t> input;
        std::vector<int32_t> expected;
        std::string sample_id;
    };

    void workerLoop();

    Execute execute_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::thread worker_;
    std::optional<Job> pending_;
    std::optional<Result> result_;
    bool running_ = false;
    bool stopping_ = false;
};
