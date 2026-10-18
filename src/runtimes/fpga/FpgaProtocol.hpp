#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace Mimir::FpgaProtocol {

inline constexpr std::array<uint8_t, 8> kGetCapabilities = {
    'M', 'I', 'M', 'I', 'R', '?', 1, '\n'
};
inline constexpr std::array<uint8_t, 8> kCapabilitiesSignature = {
    'M', 'I', 'M', 'I', 'R', 'F', 'P', 'G'
};
inline constexpr std::array<uint8_t, 4> kDotCommand = {'D', 'O', 'T', '8'};
inline constexpr std::array<uint8_t, 4> kDotResponse = {'D', 'R', 'E', 'S'};
inline constexpr std::array<uint8_t, 4> kLoadWeightsCommand = {'L', 'O', 'A', 'D'};
inline constexpr std::array<uint8_t, 4> kLoadWeightsResponse = {'W', 'A', 'C', 'K'};
inline constexpr std::array<uint8_t, 4> kExecuteResidentCommand = {'E', 'X', 'E', 'C'};
inline constexpr std::array<uint8_t, 4> kLoadMatrixCommand = {'M', 'W', 'G', 'T'};
inline constexpr std::array<uint8_t, 4> kLoadMatrixResponse = {'M', 'A', 'C', 'K'};
inline constexpr std::array<uint8_t, 4> kExecuteMatrixCommand = {'M', 'V', 'E', 'C'};
inline constexpr std::array<uint8_t, 4> kMatrixResponse = {'M', 'R', 'E', 'S'};
inline constexpr size_t kCapabilitiesResponseSize = 16;
inline constexpr uint8_t kProtocolMajor = 1;

enum Capability : uint32_t {
    DotProductInt8 = 1U << 0,
    ResidentDotProductInt8 = 1U << 1,
    ResidentMatrixVectorInt8 = 1U << 2,
    BufferedIoConversion = 1U << 3
};

struct Capabilities {
    uint8_t protocol_major = 0;
    uint8_t protocol_minor = 0;
    uint32_t operations = 0;
    uint16_t max_vector_elements = 0;
};

bool decodeCapabilities(
    const uint8_t* response,
    size_t response_size,
    Capabilities& capabilities
);

} // namespace Mimir::FpgaProtocol