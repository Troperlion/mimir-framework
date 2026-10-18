#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace Mimir {
// Return false only for unavailable firmware; integrity/transport failures throw.
bool fpgaEncode16(uint16_t* output,const float* input,size_t count,bool bf16);
bool fpgaDecode16(float* output,const uint16_t* input,size_t count,bool bf16);
void fpgaStageDataset(std::vector<uint8_t>& bytes);
class FpgaCheckpointTransaction {
public:
    explicit FpgaCheckpointTransaction(bool saving);
    ~FpgaCheckpointTransaction();
    void verifyFiles(const std::string& path);
    void finish();
private:
    struct State;
    std::unique_ptr<State> state_;
};
}
