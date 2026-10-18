#include "viz/SoftwareVizWindow.hpp"

#include <stdexcept>
#include <thread>

namespace mimir::viz {

SoftwareVizWindow::SoftwareVizWindow(vizgfx::Vector2u size,
                                     const std::string& title,
                                     unsigned fps_limit) {
    if (!validSize(size.x, size.y)) throw std::runtime_error("Invalid Viz dimensions");
    texture_ = std::make_unique<vizgfx::RenderTexture>(size);
    texture_->recordCommands(true);
    host_ = makeHost(title);
    setFramerateLimit(fps_limit);
}

SoftwareVizWindow::~SoftwareVizWindow() = default;
vizgfx::RenderTarget& SoftwareVizWindow::renderTarget() { return *texture_; }
bool SoftwareVizWindow::isOpen() const { return open_; }
void SoftwareVizWindow::close() { open_ = false; host_.reset(); }
bool SoftwareVizWindow::setActive(bool active) { return texture_->setActive(active); }
void SoftwareVizWindow::setIcon(const vizgfx::Image&) {}
void SoftwareVizWindow::setFramerateLimit(unsigned limit) { limit_ = limit; }

void SoftwareVizWindow::setSize(vizgfx::Vector2u size) {
    if (!validSize(size.x, size.y)) throw std::runtime_error("Invalid Viz dimensions");
    if (size == texture_->getSize()) return;
    const auto view = texture_->getView();
    if (!texture_->resize(size)) throw std::runtime_error("Cannot resize Viz framebuffer");
    texture_->setView(view);
    resize_ = vizgfx::Event::Resized{size};
}

vizgfx::Vector2u SoftwareVizWindow::getSize() const { return texture_->getSize(); }
void SoftwareVizWindow::setView(const vizgfx::View& view) { texture_->setView(view); }
const vizgfx::View& SoftwareVizWindow::getView() const { return texture_->getView(); }
vizgfx::Vector2f SoftwareVizWindow::mapPixelToCoords(vizgfx::Vector2i point) const {
    return texture_->mapPixelToCoords(point);
}
void SoftwareVizWindow::setMouseCursor(const vizgfx::Cursor&, int kind) { cursor_ = kind; }
bool SoftwareVizWindow::isKeyPressed(vizgfx::Keyboard::Key key) const {
    return keys_.count(key) != 0;
}

vizgfx::Keyboard::Key SoftwareVizWindow::decodeKey(int key) {
    using K = vizgfx::Keyboard::Key;
    if (key >= 'A' && key <= 'Z') return static_cast<K>(int(K::A) + key - 'A');
    if (key >= '0' && key <= '9') return static_cast<K>(int(K::Num0) + key - '0');
    if (key >= 281 && key <= 292) return static_cast<K>(int(K::F1) + key - 281);
    switch (key) {
        case 256: return K::Escape;
        case 257: return K::Enter;
        case 258: return K::Tab;
        case 259: return K::Backspace;
        case 260: return K::Left;
        case 261: return K::Right;
        case 262: return K::Up;
        case 263: return K::Down;
        case 32: return K::Space;
        default: return K::Unknown;
    }
}

std::optional<vizgfx::Event> SoftwareVizWindow::pollEvent() {
    if (resize_) {
        auto event = resize_;
        resize_.reset();
        return event;
    }
    if (!host_) return {};
    while (auto event = host_->poll()) {
        const vizgfx::Vector2i position(event->x, event->y);
        const auto button = event->value == 1 ? vizgfx::Mouse::Button::Right
            : event->value == 2 ? vizgfx::Mouse::Button::Middle
                                : vizgfx::Mouse::Button::Left;
        switch (event->type) {
            case Close: return vizgfx::Event::Closed{};
            case Move: return vizgfx::Event::MouseMoved{position};
            case Down: return vizgfx::Event::MouseButtonPressed{button, position};
            case Up: return vizgfx::Event::MouseButtonReleased{button, position};
            case Wheel:
                return vizgfx::Event::MouseWheelScrolled{
                    vizgfx::Mouse::Wheel::Vertical, event->delta, position};
            case Text:
                return vizgfx::Event::TextEntered{static_cast<char32_t>(event->value)};
            case Blur:
                keys_.clear();
                return vizgfx::Event::FocusLost{};
            case KeyDown: {
                const auto key = decodeKey(event->value);
                keys_.insert(key);
                return vizgfx::Event::KeyPressed{
                    key, vizgfx::Keyboard::Scancode::Unknown,
                    bool(event->modifiers & 4), bool(event->modifiers & 1),
                    bool(event->modifiers & 2), bool(event->modifiers & 8)};
            }
            case KeyUp: {
                const auto key = decodeKey(event->value);
                keys_.erase(key);
                return vizgfx::Event::KeyReleased{
                    key, vizgfx::Keyboard::Scancode::Unknown,
                    bool(event->modifiers & 4), bool(event->modifiers & 1),
                    bool(event->modifiers & 2), bool(event->modifiers & 8)};
            }
        }
    }
    return {};
}

void SoftwareVizWindow::clear(const vizgfx::Color& color) { texture_->clear(color); }
void SoftwareVizWindow::display() {
    if (host_) {
        const auto size = texture_->getSize();
        Frame frame;
        frame.width = size.x;
        frame.height = size.y;
        frame.cursor = cursor_;
        frame.scene = texture_->scene();
        host_->present(std::move(frame));
    }
    if (limit_) {
        std::this_thread::sleep_until(
            last_ + std::chrono::microseconds(1000000 / limit_));
    }
    last_ = std::chrono::steady_clock::now();
}

vizgfx::Image SoftwareVizWindow::captureImage() const {
    texture_->display();
    return texture_->getTexture().copyToImage();
}

} // namespace mimir::viz
