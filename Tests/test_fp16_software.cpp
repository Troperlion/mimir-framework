// Exercise the software path even on a machine compiled with F16C enabled.
#ifdef __F16C__
#undef __F16C__
#endif
#include "runtimes/cpu/HardwareOpt.hpp"
#include "test_utils.hpp"
#include <limits>

int main() {
    // Every finite half value must survive decode/encode exactly, including
    // signed zero and subnormals. No hardware half instructions are involved.
    for (unsigned bits = 0; bits < 65536; ++bits) {
        if ((bits & 0x7c00u) == 0x7c00u) continue;
        const uint16_t half = static_cast<uint16_t>(bits);
        uint16_t restored;
        float value;
        HardwareOpt::fp16_to_fp32_f16c(&value, &half, 1);
        TASSERT_TRUE(std::isfinite(value));
        HardwareOpt::fp32_to_fp16_f16c(&restored, &value, 1);
        TASSERT_TRUE(restored == half);
    }
    const float values[] = {0.0f, -0.0f, 1.0f, -2.0f, 65504.0f,
        std::ldexp(1.0f, -24), std::ldexp(1.0f, -25),
        1.0f + std::ldexp(1.0f, -11),
        1.0f + 3.0f * std::ldexp(1.0f, -11),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};
    uint16_t encoded[11];
    HardwareOpt::fp32_to_fp16_f16c(encoded, values, 11);
    const uint16_t expected[] = {0, 0x8000, 0x3c00, 0xc000, 0x7bff,
        1, 0, 0x3c00, 0x3c02, 0x7c00};
    for (unsigned i=0; i<10; ++i) TASSERT_TRUE(encoded[i] == expected[i]);
    TASSERT_TRUE((encoded[10] & 0x7c00) == 0x7c00 && (encoded[10] & 0x3ff));
    return 0;
}
