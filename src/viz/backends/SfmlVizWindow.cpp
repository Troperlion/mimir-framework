#include "VizWindow.hpp"

#include <memory>

namespace {
class SfmlVizWindow final : public VizWindow {
public:
    SfmlVizWindow(vizgfx::Vector2u size, const std::string& title, unsigned fps_limit)
        : window_(std::make_unique<vizgfx::RenderWindow>(
              vizgfx::VideoMode(size), title,
              vizgfx::Style::Default)) {
        window_->setFramerateLimit(fps_limit);
    }
    bool isOpen() const override { return window_->isOpen(); }
    void close() override { window_->close(); }
    bool setActive(bool active) override { return window_->setActive(active); }
    void setIcon(const vizgfx::Image& icon) override { window_->setIcon(icon); }
    void setFramerateLimit(unsigned limit) override { window_->setFramerateLimit(limit); }
    void setSize(vizgfx::Vector2u size) override { window_->setSize(size); }
    vizgfx::Vector2u getSize() const override { return window_->getSize(); }
    void setView(const vizgfx::View& view) override { window_->setView(view); }
    const vizgfx::View& getView() const override { return window_->getView(); }
    vizgfx::Vector2f mapPixelToCoords(vizgfx::Vector2i point) const override {
        return window_->mapPixelToCoords(point);
    }
    void setMouseCursor(const vizgfx::Cursor& cursor, int) override {
        window_->setMouseCursor(cursor);
    }
    bool isKeyPressed(vizgfx::Keyboard::Key key) const override {
        return vizgfx::Keyboard::isKeyPressed(key);
    }
    std::optional<vizgfx::Event> pollEvent() override { return window_->pollEvent(); }
    void clear(const vizgfx::Color& color) override { window_->clear(color); }
    void display() override { window_->display(); }
    vizgfx::Image captureImage() const override {
        vizgfx::Texture texture(window_->getSize());
        texture.update(*window_);
        return texture.copyToImage();
    }
private:
    vizgfx::RenderTarget& renderTarget() override { return *window_; }
    std::unique_ptr<vizgfx::RenderWindow> window_;
};
} // namespace

std::unique_ptr<VizWindow> createVizWindow(vizgfx::Vector2u size,
                                           const std::string& title,
                                           unsigned fps_limit) {
    return std::make_unique<SfmlVizWindow>(size, title, fps_limit);
}
const char* vizBackendName() { return "SFML"; }
