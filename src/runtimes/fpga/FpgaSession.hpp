#pragma once
#include "include/json.hpp"
class FpgaRuntime;
namespace Mimir {
// Process-wide diagnostic connection. Does not claim FP32 layer acceleration.
void initializeFpgaSession();
FpgaRuntime* fpgaSessionRuntime();
nlohmann::json fpgaSessionStatus();
}
