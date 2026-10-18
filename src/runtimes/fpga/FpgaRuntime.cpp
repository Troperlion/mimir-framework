#include "runtimes/fpga/FpgaRuntime.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>

#if defined(__linux__) || defined(__APPLE__)
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

std::string readTrimmed(const fs::path& path) {
    std::ifstream stream(path);
    std::string value;
    std::getline(stream, value);
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string findTty(const fs::path& device_path) {
    std::error_code error;
    fs::recursive_directory_iterator iterator(
        device_path,
        fs::directory_options::skip_permission_denied,
        error
    );
    const fs::recursive_directory_iterator end;
    while (!error && iterator != end) {
        const std::string name = iterator->path().filename().string();
        if (name.rfind("ttyACM", 0) == 0) {
            return "/dev/" + name;
        }
        iterator.increment(error);
    }
    return {};
}

bool ttyAccessible(const std::string& path) {
    if (path.empty()) return false;
#if defined(__linux__) || defined(__APPLE__)
    return access(path.c_str(), R_OK | W_OK) == 0;
#else
    return fs::exists(path);
#endif
}

#if defined(__linux__) || defined(__APPLE__)
bool writeAll(int descriptor, const uint8_t* data, size_t size, int timeout_ms) {
    size_t written = 0;
    while (written < size) {
        pollfd event{descriptor, POLLOUT, 0};
        if (poll(&event, 1, timeout_ms) <= 0 || (event.revents & POLLOUT) == 0) return false;
        const ssize_t count = write(descriptor, data + written, size - written);
        if (count > 0) written += static_cast<size_t>(count);
        else if (count < 0 && errno != EINTR) return false;
    }
    return true;
}

bool readAll(int descriptor, uint8_t* data, size_t size, int timeout_ms) {
    size_t received = 0;
    while (received < size) {
        pollfd event{descriptor, POLLIN, 0};
        if (poll(&event, 1, timeout_ms) <= 0 || (event.revents & POLLIN) == 0) return false;
        const ssize_t count = read(descriptor, data + received, size - received);
        if (count > 0) received += static_cast<size_t>(count);
        else if (count < 0 && errno != EINTR) return false;
    }
    return true;
}
#endif

} // namespace

bool Mimir::FpgaProtocol::decodeCapabilities(
    const uint8_t* response,
    size_t response_size,
    Capabilities& capabilities
) {
    if (!response || response_size != kCapabilitiesResponseSize ||
        !std::equal(kCapabilitiesSignature.begin(), kCapabilitiesSignature.end(), response)) {
        return false;
    }
    capabilities.protocol_major = response[8];
    capabilities.protocol_minor = response[9];
    capabilities.operations = static_cast<uint32_t>(response[10]) |
        (static_cast<uint32_t>(response[11]) << 8U) |
        (static_cast<uint32_t>(response[12]) << 16U) |
        (static_cast<uint32_t>(response[13]) << 24U);
    capabilities.max_vector_elements = static_cast<uint16_t>(response[14]) |
        static_cast<uint16_t>(static_cast<uint16_t>(response[15]) << 8U);
    return capabilities.protocol_major == kProtocolMajor;
}

std::optional<FpgaDeviceInfo> FpgaRuntime::detectDevice(const std::string& sysfs_root) {
    std::error_code error;
    fs::directory_iterator iterator(
        sysfs_root,
        fs::directory_options::skip_permission_denied,
        error
    );
    const fs::directory_iterator end;
    while (!error && iterator != end) {
        const fs::path path = iterator->path();
        if (readTrimmed(path / "idVendor") == "1d50" &&
            readTrimmed(path / "idProduct") == "602b") {
            const std::string serial = readTrimmed(path / "serial");
            if (serial.rfind("0710", 0) == 0) {
                FpgaDeviceInfo info;
                info.sysfs_path = path.string();
                info.serial = serial;
                info.tty_path = findTty(path);
                if (const char* override_path = std::getenv("MIMIR_FPGA_DEVICE")) {
                    if (override_path[0] != '\0') info.tty_path = override_path;
                }
                return info;
            }
        }
        iterator.increment(error);
    }
    return std::nullopt;
}

bool FpgaRuntime::initialize(const RuntimeConfig& cfg) {
    shutdown();
    config_ = cfg;
    if (cfg.disabled) return false;

    const char* root_override = std::getenv("MIMIR_FPGA_SYSFS_ROOT");
    device_ = detectDevice(
        root_override && root_override[0] != '\0'
            ? root_override
            : "/sys/bus/usb/devices"
    );
    if (!device_ || !ttyAccessible(device_->tty_path)) {
        device_.reset();
        return false;
    }

    if (!openAndHandshake() || !hasComputeCapabilities()) {
        shutdown();
        return false;
    }

    initialized_ = true;
    return true;
}

void FpgaRuntime::shutdown() {
#if defined(__linux__) || defined(__APPLE__)
    if (serial_descriptor_ >= 0) close(serial_descriptor_);
#endif
    serial_descriptor_ = -1;
    initialized_ = false;
    device_.reset();
    capabilities_ = {};
    resident_weight_elements_ = 0;
    resident_matrix_rows_ = 0;
    resident_matrix_columns_ = 0;
}

bool FpgaRuntime::openAndHandshake() {
#if !defined(__linux__) && !defined(__APPLE__)
    return false;
#else
    serial_descriptor_ = open(device_->tty_path.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (serial_descriptor_ < 0) return false;
    if (flock(serial_descriptor_, LOCK_EX | LOCK_NB) != 0) return false;

    termios tty{};
    if (tcgetattr(serial_descriptor_, &tty) != 0) return false;
    cfmakeraw(&tty);
    cfsetispeed(&tty, B115200);
    cfsetospeed(&tty, B115200);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag |= CLOCAL | CREAD;
    tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;
    if (tcsetattr(serial_descriptor_, TCSANOW, &tty) != 0) return false;
    tcflush(serial_descriptor_, TCIOFLUSH);

    std::array<uint8_t, Mimir::FpgaProtocol::kCapabilitiesResponseSize> response{};
    return writeAll(
               serial_descriptor_,
               Mimir::FpgaProtocol::kGetCapabilities.data(),
               Mimir::FpgaProtocol::kGetCapabilities.size(),
               250
           ) &&
        readAll(serial_descriptor_, response.data(), response.size(), 250) &&
        Mimir::FpgaProtocol::decodeCapabilities(response.data(), response.size(), capabilities_);
#endif
}

bool FpgaRuntime::int8DotProduct(
    const int8_t* lhs,
    const int8_t* rhs,
    size_t elements,
    int32_t& result
) {
#if !defined(__linux__) && !defined(__APPLE__)
    (void)lhs; (void)rhs; (void)elements; (void)result;
    return false;
#else
    if (!initialized_ || !lhs || !rhs || elements == 0 || elements > 255 ||
        elements > capabilities_.max_vector_elements ||
        (capabilities_.operations & Mimir::FpgaProtocol::DotProductInt8) == 0) {
        return false;
    }

    std::vector<uint8_t> request;
    request.reserve(5 + elements * 2);
    request.insert(request.end(), Mimir::FpgaProtocol::kDotCommand.begin(),
                   Mimir::FpgaProtocol::kDotCommand.end());
    request.push_back(static_cast<uint8_t>(elements));
    for (size_t index = 0; index < elements; ++index) {
        request.push_back(static_cast<uint8_t>(lhs[index]));
        request.push_back(static_cast<uint8_t>(rhs[index]));
    }

    std::array<uint8_t, 8> response{};
    std::lock_guard<std::mutex> lock(transport_mutex_);
    if (!writeAll(serial_descriptor_, request.data(), request.size(), 500) ||
        !readAll(serial_descriptor_, response.data(), response.size(), 500) ||
        !std::equal(Mimir::FpgaProtocol::kDotResponse.begin(),
                    Mimir::FpgaProtocol::kDotResponse.end(), response.begin())) {
        return false;
    }
    const uint32_t bits = static_cast<uint32_t>(response[4]) |
        (static_cast<uint32_t>(response[5]) << 8U) |
        (static_cast<uint32_t>(response[6]) << 16U) |
        (static_cast<uint32_t>(response[7]) << 24U);
    result = static_cast<int32_t>(bits);
    return true;
#endif
}

bool FpgaRuntime::uploadInt8Weights(const int8_t* weights, size_t elements) {
#if !defined(__linux__) && !defined(__APPLE__)
    (void)weights; (void)elements;
    return false;
#else
    if (!initialized_ || !weights || elements == 0 || elements > 255 ||
        elements > capabilities_.max_vector_elements ||
        (capabilities_.operations & Mimir::FpgaProtocol::ResidentDotProductInt8) == 0) {
        return false;
    }

    std::vector<uint8_t> request;
    request.reserve(5 + elements);
    request.insert(request.end(), Mimir::FpgaProtocol::kLoadWeightsCommand.begin(),
                   Mimir::FpgaProtocol::kLoadWeightsCommand.end());
    request.push_back(static_cast<uint8_t>(elements));
    for (size_t index = 0; index < elements; ++index) {
        request.push_back(static_cast<uint8_t>(weights[index]));
    }

    std::array<uint8_t, 4> response{};
    std::lock_guard<std::mutex> lock(transport_mutex_);
    if (!writeAll(serial_descriptor_, request.data(), request.size(), 500) ||
        !readAll(serial_descriptor_, response.data(), response.size(), 500) ||
        !std::equal(Mimir::FpgaProtocol::kLoadWeightsResponse.begin(),
                    Mimir::FpgaProtocol::kLoadWeightsResponse.end(), response.begin())) {
        resident_weight_elements_ = 0;
        return false;
    }
    resident_weight_elements_ = elements;
    return true;
#endif
}

bool FpgaRuntime::int8ResidentDotProduct(
    const int8_t* input,
    size_t elements,
    int32_t& result
) {
#if !defined(__linux__) && !defined(__APPLE__)
    (void)input; (void)elements; (void)result;
    return false;
#else
    if (!initialized_ || !input || elements == 0 || elements != resident_weight_elements_ ||
        (capabilities_.operations & Mimir::FpgaProtocol::ResidentDotProductInt8) == 0) {
        return false;
    }

    std::vector<uint8_t> request;
    request.reserve(5 + elements);
    request.insert(request.end(), Mimir::FpgaProtocol::kExecuteResidentCommand.begin(),
                   Mimir::FpgaProtocol::kExecuteResidentCommand.end());
    request.push_back(static_cast<uint8_t>(elements));
    for (size_t index = 0; index < elements; ++index) {
        request.push_back(static_cast<uint8_t>(input[index]));
    }

    std::array<uint8_t, 8> response{};
    std::lock_guard<std::mutex> lock(transport_mutex_);
    if (!writeAll(serial_descriptor_, request.data(), request.size(), 500) ||
        !readAll(serial_descriptor_, response.data(), response.size(), 500) ||
        !std::equal(Mimir::FpgaProtocol::kDotResponse.begin(),
                    Mimir::FpgaProtocol::kDotResponse.end(), response.begin())) {
        return false;
    }
    const uint32_t bits = static_cast<uint32_t>(response[4]) |
        (static_cast<uint32_t>(response[5]) << 8U) |
        (static_cast<uint32_t>(response[6]) << 16U) |
        (static_cast<uint32_t>(response[7]) << 24U);
    result = static_cast<int32_t>(bits);
    return true;
#endif
}

bool FpgaRuntime::uploadInt8Matrix(
    const int8_t* weights,
    size_t rows,
    size_t columns
) {
#if !defined(__linux__) && !defined(__APPLE__)
    (void)weights; (void)rows; (void)columns;
    return false;
#else
    if (!initialized_ || !weights || rows == 0 || rows > 8 || columns != 64 ||
        (capabilities_.operations & Mimir::FpgaProtocol::ResidentMatrixVectorInt8) == 0) {
        return false;
    }

    const size_t elements = rows * columns;
    std::vector<uint8_t> request;
    request.reserve(6 + elements);
    request.insert(request.end(), Mimir::FpgaProtocol::kLoadMatrixCommand.begin(),
                   Mimir::FpgaProtocol::kLoadMatrixCommand.end());
    request.push_back(static_cast<uint8_t>(rows));
    request.push_back(static_cast<uint8_t>(columns));
    for (size_t index = 0; index < elements; ++index) {
        request.push_back(static_cast<uint8_t>(weights[index]));
    }

    std::array<uint8_t, 4> response{};
    std::lock_guard<std::mutex> lock(transport_mutex_);
    if (!writeAll(serial_descriptor_, request.data(), request.size(), 1000) ||
        !readAll(serial_descriptor_, response.data(), response.size(), 1000) ||
        !std::equal(Mimir::FpgaProtocol::kLoadMatrixResponse.begin(),
                    Mimir::FpgaProtocol::kLoadMatrixResponse.end(), response.begin())) {
        resident_matrix_rows_ = 0;
        resident_matrix_columns_ = 0;
        return false;
    }
    resident_matrix_rows_ = rows;
    resident_matrix_columns_ = columns;
    return true;
#endif
}

bool FpgaRuntime::int8MatrixVector(
    const int8_t* input,
    size_t columns,
    std::vector<int32_t>& output
) {
#if !defined(__linux__) && !defined(__APPLE__)
    (void)input; (void)columns; (void)output;
    return false;
#else
    if (!initialized_ || !input || columns != resident_matrix_columns_ ||
        resident_matrix_rows_ == 0 ||
        (capabilities_.operations & Mimir::FpgaProtocol::ResidentMatrixVectorInt8) == 0) {
        return false;
    }

    std::vector<uint8_t> request;
    request.reserve(5 + columns);
    request.insert(request.end(), Mimir::FpgaProtocol::kExecuteMatrixCommand.begin(),
                   Mimir::FpgaProtocol::kExecuteMatrixCommand.end());
    request.push_back(static_cast<uint8_t>(columns));
    for (size_t index = 0; index < columns; ++index) {
        request.push_back(static_cast<uint8_t>(input[index]));
    }

    std::vector<uint8_t> response(5 + resident_matrix_rows_ * 4);
    std::lock_guard<std::mutex> lock(transport_mutex_);
    if (!writeAll(serial_descriptor_, request.data(), request.size(), 1000) ||
        !readAll(serial_descriptor_, response.data(), response.size(), 1000) ||
        !std::equal(Mimir::FpgaProtocol::kMatrixResponse.begin(),
                    Mimir::FpgaProtocol::kMatrixResponse.end(), response.begin()) ||
        response[4] != resident_matrix_rows_) {
        return false;
    }

    output.resize(resident_matrix_rows_);
    for (size_t row = 0; row < resident_matrix_rows_; ++row) {
        const size_t offset = 5 + row * 4;
        const uint32_t bits = static_cast<uint32_t>(response[offset]) |
            (static_cast<uint32_t>(response[offset + 1]) << 8U) |
            (static_cast<uint32_t>(response[offset + 2]) << 16U) |
            (static_cast<uint32_t>(response[offset + 3]) << 24U);
        output[row] = static_cast<int32_t>(bits);
    }
    return true;
#endif
}

bool FpgaRuntime::linearForward(
    const float*,
    const float*,
    const float*,
    float*,
    int,
    int,
    int
) {
    return false;
}

bool FpgaRuntime::forwardLayer(
    const std::vector<const std::vector<float>*>&,
    std::vector<std::vector<float>>&,
    const Layer&,
    bool
) {
    return false;
}

bool FpgaRuntime::supportsForwardLayerType(LayerType) const {
    return false;
}

bool FpgaRuntime::supportsBackwardLayerType(LayerType) const {
    return false;
}

namespace {
uint32_t fpgaCrc(const uint8_t* bytes,size_t size) {
    uint32_t crc=0xffffffffU;
    for(size_t i=0;i<size;++i) {
        crc^=bytes[i];
        for(int bit=0;bit<8;++bit) crc=(crc>>1)^((crc&1)?0xedb88320U:0);
    }
    return ~crc;
}
uint32_t readLe32(const uint8_t* p) {
    return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
}
}
bool FpgaRuntime::transferBlock(uint8_t mode,const uint8_t* input,size_t size,std::vector<uint8_t>& output) {
    output.clear();
#if defined(__linux__) || defined(__APPLE__)
    if(!initialized_ || !(capabilities_.operations & Mimir::FpgaProtocol::BufferedIoConversion) ||
        !input || !size || size>256 || mode>5 || ((mode==1 || mode==3) && size%4) ||
        ((mode==2 || mode==4) && size%2)) return false;
    const uint32_t crc=fpgaCrc(input,size);
    std::vector<uint8_t> request={'@','B','U','F',mode,static_cast<uint8_t>(size),static_cast<uint8_t>(size>>8)};
    for(int shift=0;shift<32;shift+=8) request.push_back(static_cast<uint8_t>(crc>>shift));
    request.insert(request.end(),input,input+size);
    std::lock_guard<std::mutex> lock(transport_mutex_);
    std::array<uint8_t,7> header{};
    if(!writeAll(serial_descriptor_,request.data(),request.size(),1000) ||
        !readAll(serial_descriptor_,header.data(),header.size(),1000) ||
        !std::equal(header.begin(),header.begin()+4,"XRES")) return false;
    const size_t length=header[5]+(size_t(header[6])<<8);
    if(length>512) return false;
    std::vector<uint8_t> response(length+4);
    if(!readAll(serial_descriptor_,response.data(),response.size(),1000)) return false;
    const size_t expected=(mode==1 || mode==3)?size/2:(mode==2 || mode==4)?size*2:size;
    if(header[4]!=mode || length!=expected || readLe32(response.data()+length)!=fpgaCrc(response.data(),length)) return false;
    output.assign(response.begin(),response.begin()+length);
    return true;
#else
    (void)mode;(void)input;(void)size;return false;
#endif
}
bool FpgaRuntime::beginCheckpoint(uint8_t operation) {
#if defined(__linux__) || defined(__APPLE__)
    if(!initialized_ || !(capabilities_.operations & Mimir::FpgaProtocol::BufferedIoConversion) || operation>2) return false;
    const std::array<uint8_t,5> request={'@','B','E','G',operation};
    std::array<uint8_t,4> response{};
    std::lock_guard<std::mutex> lock(transport_mutex_);
    return writeAll(serial_descriptor_,request.data(),request.size(),1000) &&
        readAll(serial_descriptor_,response.data(),response.size(),1000) &&
        std::equal(response.begin(),response.end(),"CACK");
#else
    (void)operation;return false;
#endif
}
bool FpgaRuntime::endCheckpoint(uint8_t& operation,uint64_t& bytes,uint32_t& crc) {
#if defined(__linux__) || defined(__APPLE__)
    if(!initialized_ || !(capabilities_.operations & Mimir::FpgaProtocol::BufferedIoConversion)) return false;
    const std::array<uint8_t,4> request={'@','E','N','D'};
    std::array<uint8_t,17> response{};
    std::lock_guard<std::mutex> lock(transport_mutex_);
    if(!writeAll(serial_descriptor_,request.data(),request.size(),1000) ||
        !readAll(serial_descriptor_,response.data(),response.size(),1000) ||
        !std::equal(response.begin(),response.begin()+4,"CRES")) return false;
    operation=response[4]; bytes=0;
    for(int i=0;i<8;++i) bytes|=uint64_t(response[5+i])<<(i*8);
    crc=readLe32(response.data()+13);
    return true;
#else
    (void)operation;(void)bytes;(void)crc;return false;
#endif
}
