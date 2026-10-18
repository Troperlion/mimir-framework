#pragma once
#include <memory>
#include <string>
#include <vector>
namespace PortableStaging {
class Device {
public:
    virtual ~Device() = default;
    virtual bool run(std::vector<float>& memory, const std::vector<int>& arguments) = 0;
};
std::unique_ptr<Device> createLossDevice(const std::string& backend, int index);
std::unique_ptr<Device> createOptimizerDevice(const std::string& backend, int index);
std::unique_ptr<Device> createDevice(const std::string& backend, int index);
}
