#pragma once

#include "runtimes/AbstractRuntime.hpp"
#include "runtimes/fpga/FpgaProtocol.hpp"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

struct FpgaDeviceInfo {
    std::string sysfs_path;
    std::string serial;
    std::string tty_path;
};

class FpgaRuntime final : public AbstractRuntime {
public:
    const char* name() const override { return "FPGA"; }

    bool initialize(const RuntimeConfig& cfg) override;
    void shutdown() override;
    bool isInitialized() const override { return initialized_; }

    bool linearForward(
        const float* input,
        const float* weights,
        const float* bias_or_null,
        float* output,
        int batch,
        int in_f,
        int out_f
    ) override;

    bool forwardLayer(
        const std::vector<const std::vector<float>*>& inputs,
        std::vector<std::vector<float>>& outputs,
        const Layer& layer,
        bool training
    ) override;

    bool supportsForwardLayerType(LayerType type) const override;
    bool supportsBackwardLayerType(LayerType type) const override;

    const std::optional<FpgaDeviceInfo>& device() const { return device_; }
    const Mimir::FpgaProtocol::Capabilities& capabilities() const { return capabilities_; }
    bool hasComputeCapabilities() const { return capabilities_.operations != 0; }

    bool int8DotProduct(
        const int8_t* lhs,
        const int8_t* rhs,
        size_t elements,
        int32_t& result
    );

    bool uploadInt8Weights(const int8_t* weights, size_t elements);
    bool int8ResidentDotProduct(
        const int8_t* input,
        size_t elements,
        int32_t& result
    );
    bool uploadInt8Matrix(
        const int8_t* weights,
        size_t rows,
        size_t columns
    );
    bool int8MatrixVector(
        const int8_t* input,
        size_t columns,
        std::vector<int32_t>& output
    );

    // mode: 0 dataset bytes, 1 F32->BF16, 2 BF16->F32,
    // 3 F32->F16, 4 F16->F32, 5 checkpoint bytes. <=256 input bytes.
    bool transferBlock(uint8_t mode, const uint8_t* input, size_t size, std::vector<uint8_t>& output);
    bool beginCheckpoint(uint8_t operation); // 0 abort, 1 load, 2 save
    bool endCheckpoint(uint8_t& operation, uint64_t& bytes, uint32_t& crc);

    static std::optional<FpgaDeviceInfo> detectDevice(
        const std::string& sysfs_root = "/sys/bus/usb/devices"
    );

private:
    bool openAndHandshake();

    bool initialized_ = false;
    int serial_descriptor_ = -1;
    std::optional<FpgaDeviceInfo> device_;
    Mimir::FpgaProtocol::Capabilities capabilities_{};
    size_t resident_weight_elements_ = 0;
    size_t resident_matrix_rows_ = 0;
    size_t resident_matrix_columns_ = 0;
    std::mutex transport_mutex_;
};