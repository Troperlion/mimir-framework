#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mimir::viz {
// Local IPC protocol: fixed-width fields, same executable build on both ends.
// Keys: ASCII uppercase/digits, 256 Escape, 257 Enter, 258 Tab,
// 259 Backspace, 260..263 Left/Right/Up/Down, 281..292 F1..F12.
enum EventType { Close = 1, Move, Down, Up, Wheel, KeyDown, KeyUp, Text, Blur };
struct Event { int32_t type=0, x=0, y=0, value=0, modifiers=0; float delta=0; };
struct Frame { uint32_t width=0, height=0, cursor=0; std::string scene; };
struct FrameHeader { uint32_t width, height, cursor, sceneBytes; };
constexpr int32_t sceneProtocolHandshake = 101;
constexpr uint32_t maxSceneBytes = 128 * 1024 * 1024;
constexpr uint32_t maxDimension = 8192;
inline bool validSize(uint32_t w, uint32_t h) {
    return w && h && w <= maxDimension && h <= maxDimension && uint64_t(w)*h <= 33554432;
}
class Host {
public:
    virtual ~Host() = default;
    virtual void present(Frame frame) = 0;
    virtual std::optional<Event> poll() = 0;
};
std::unique_ptr<Host> makeHost(const std::string& title);
}
