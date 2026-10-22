#include "runtimes/portable/PortableShaderData.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

int main() {
    std::ifstream stream("src/runtimes/portable/reference.cl", std::ios::binary);
    const std::string expected{std::istreambuf_iterator<char>(stream),std::istreambuf_iterator<char>()};
    if (!stream || expected.empty() || expected != PortableStaging::referenceOpenCLStorage ||
        expected != std::string(PortableStaging::referenceOpenCL)) {
        std::cerr << "Embedded OpenCL bytes differ from reference.cl\n";
        return 1;
    }
    std::cout << "EMBEDDED_SHADER_SOURCE_OK " << expected.size() << " bytes\n";
}
