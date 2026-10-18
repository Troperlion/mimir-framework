#pragma once

#include "DType.hpp"
#include "runtimes/AbstractRuntime.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Mimir {

// Transitional explicit-residency handle. It deliberately performs no copy:
// executors must materialize transfers listed by ExecutionPlan and then mark
// the corresponding representation valid.
struct DeviceStorageHandle {
    RuntimeKind runtime = RuntimeKind::Unknown;
    uintptr_t opaque_handle = 0;
    size_t bytes = 0;
};

class RuntimeTensorHandle {
public:
    RuntimeTensorHandle() = default;
    RuntimeTensorHandle(std::vector<int> shape, DType dtype)
        : shape_(std::move(shape)), dtype_(dtype) {
        size_t elements = 1;
        if (shape_.empty() || dtype_size_bytes(dtype_) == 0) return;
        for (const int dimension : shape_) {
            if (dimension <= 0) throw std::invalid_argument("RuntimeTensorHandle: invalid shape");
            elements *= static_cast<size_t>(dimension);
        }
        bytes_ = elements * dtype_size_bytes(dtype_);
    }

    const std::vector<int>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t bytes() const { return bytes_; }
    RuntimeKind location() const { return location_; }
    bool hostValid() const { return host_valid_; }
    bool deviceValid() const { return device_valid_; }
    const std::vector<uint8_t>& hostStorage() const { return host_storage_; }
    const DeviceStorageHandle& deviceStorage() const { return device_storage_; }

    std::vector<uint8_t>& prepareHostWrite() {
        host_storage_.resize(bytes_);
        host_valid_ = true;
        device_valid_ = false;
        location_ = RuntimeKind::CPU;
        return host_storage_;
    }

    void markDeviceWritten(const DeviceStorageHandle& storage) {
        if (storage.runtime == RuntimeKind::CPU || storage.runtime == RuntimeKind::Unknown ||
            storage.bytes < bytes_ || storage.opaque_handle == 0) {
            throw std::invalid_argument("RuntimeTensorHandle: invalid device storage");
        }
        device_storage_ = storage;
        device_valid_ = true;
        host_valid_ = false;
        location_ = storage.runtime;
    }

    void markHostSynchronized(std::vector<uint8_t> storage) {
        if (storage.size() != bytes_) {
            throw std::invalid_argument("RuntimeTensorHandle: host storage size mismatch");
        }
        host_storage_ = std::move(storage);
        host_valid_ = true;
        // Device remains valid: synchronization made both representations current.
    }

private:
    std::vector<int> shape_;
    DType dtype_ = DType::UNKNOWN;
    size_t bytes_ = 0;
    RuntimeKind location_ = RuntimeKind::CPU;
    std::vector<uint8_t> host_storage_;
    DeviceStorageHandle device_storage_;
    bool host_valid_ = false;
    bool device_valid_ = false;
};

} // namespace Mimir
