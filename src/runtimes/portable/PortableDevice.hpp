#pragma once
#include <memory>
#include <cstdint>
#include <string>
#include <vector>
namespace PortableStaging {
struct BatchStatistics { uint64_t submissions=0,kernels=0,uploaded_bytes=0,downloaded_bytes=0; };
struct MemoryRange { size_t offset=0,count=0; }; // Float elements in the staging arena.
class Device {
protected:
    BatchStatistics batch_statistics_;
public:
    BatchStatistics batchStatistics() const { return batch_statistics_; }
    virtual ~Device() = default;
    virtual bool run(std::vector<float>& memory, const std::vector<int>& arguments) = 0;
    virtual bool runBatch(std::vector<float>& memory,const std::vector<std::vector<int>>& commands,
                          const std::vector<MemoryRange>& downloads={}) {
        for(const auto& command:commands){if(!run(memory,command))return false;++batch_statistics_.submissions;++batch_statistics_.kernels;batch_statistics_.uploaded_bytes+=memory.size()*sizeof(float);batch_statistics_.downloaded_bytes+=memory.size()*sizeof(float);}
        return true;
    }
};
std::unique_ptr<Device> createLossDevice(const std::string& backend, int index);
std::unique_ptr<Device> createOptimizerDevice(const std::string& backend, int index);
std::unique_ptr<Device> createDevice(const std::string& backend, int index);
}
