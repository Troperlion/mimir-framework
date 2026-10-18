#include "runtimes/fpga/FpgaRuntime.hpp"
#include "test_utils.hpp"

#include <cstdlib>
#include <array>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

class TempTree {
public:
    TempTree()
        : path_(fs::temp_directory_path() /
                ("mimir-fpga-test-" + std::to_string(std::rand()))) {
        fs::create_directories(path_);
    }

    ~TempTree() {
        std::error_code error;
        fs::remove_all(path_, error);
    }

    const fs::path& path() const { return path_; }

private:
    fs::path path_;
};

void writeFile(const fs::path& path, const std::string& contents) {
    fs::create_directories(path.parent_path());
    std::ofstream(path) << contents;
}

void addUsbDevice(
    const fs::path& root,
    const std::string& name,
    const std::string& vendor,
    const std::string& product,
    const std::string& serial
) {
    const fs::path device = root / name;
    writeFile(device / "idVendor", vendor + "\n");
    writeFile(device / "idProduct", product + "\n");
    writeFile(device / "serial", serial + "\n");
}

} // namespace

int main() {
    TempTree tree;

    TASSERT_TRUE(!FpgaRuntime::detectDevice(tree.path().string()).has_value());

    addUsbDevice(tree.path(), "ext-board", "0d28", "0204", "0700");
    TASSERT_TRUE(!FpgaRuntime::detectDevice(tree.path().string()).has_value());

    addUsbDevice(tree.path(), "icesugar-pro", "1D50", "602B", "0710ABCDEF");
    fs::create_directories(tree.path() / "icesugar-pro" / "1-1:1.0" / "ttyACM0");

    const auto detected = FpgaRuntime::detectDevice(tree.path().string());
    TASSERT_TRUE(detected.has_value());
    TASSERT_TRUE(detected->serial == "0710abcdef");
    TASSERT_TRUE(detected->tty_path == "/dev/ttyACM0");

    const fs::path fake_tty = tree.path() / "ttyACM0";
    writeFile(fake_tty, "");
    setTestEnvironment("MIMIR_FPGA_SYSFS_ROOT", tree.path().string().c_str());
    setTestEnvironment("MIMIR_FPGA_DEVICE", fake_tty.string().c_str());

    FpgaRuntime runtime;
    TASSERT_TRUE(!runtime.initialize(RuntimeConfig::fromEnv("FPGA")));
    TASSERT_TRUE(!runtime.isInitialized());
    TASSERT_TRUE(!runtime.supportsForwardLayerType(LayerType::Linear));

    std::array<uint8_t, Mimir::FpgaProtocol::kCapabilitiesResponseSize> response = {
        'M', 'I', 'M', 'I', 'R', 'F', 'P', 'G',
        1, 0, 7, 0, 0, 0, 64, 0
    };
    Mimir::FpgaProtocol::Capabilities capabilities;
    TASSERT_TRUE(Mimir::FpgaProtocol::decodeCapabilities(
        response.data(), response.size(), capabilities
    ));
    TASSERT_TRUE(capabilities.protocol_major == 1);
    TASSERT_TRUE((capabilities.operations & Mimir::FpgaProtocol::DotProductInt8) != 0);
    TASSERT_TRUE((capabilities.operations &
                  Mimir::FpgaProtocol::ResidentDotProductInt8) != 0);
    TASSERT_TRUE((capabilities.operations &
                  Mimir::FpgaProtocol::ResidentMatrixVectorInt8) != 0);
    TASSERT_TRUE(capabilities.max_vector_elements == 64);

    response[5] = '?';
    TASSERT_TRUE(!Mimir::FpgaProtocol::decodeCapabilities(
        response.data(), response.size(), capabilities
    ));
    response[5] = 'F';
    response[8] = 2;
    TASSERT_TRUE(!Mimir::FpgaProtocol::decodeCapabilities(
        response.data(), response.size(), capabilities
    ));

    setTestEnvironment("MIMIR_DISABLE_FPGA", "1");
    TASSERT_TRUE(!runtime.initialize(RuntimeConfig::fromEnv("FPGA")));
    TASSERT_TRUE(!runtime.isInitialized());

    unsetTestEnvironment("MIMIR_DISABLE_FPGA");
    unsetTestEnvironment("MIMIR_FPGA_DEVICE");
    unsetTestEnvironment("MIMIR_FPGA_SYSFS_ROOT");
    return 0;
}