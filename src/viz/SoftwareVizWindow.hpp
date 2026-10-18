#ifndef MIMIR_SOFTWARE_VIZ_WINDOW_HPP
#define MIMIR_SOFTWARE_VIZ_WINDOW_HPP

#include "VizWindow.hpp"
#include "viz/Host.hpp"
#include <chrono>
#include <set>

namespace mimir::viz {

class SoftwareVizWindow final : public ::VizWindow {
public:
    SoftwareVizWindow(vizgfx::Vector2u size, const std::string& title,
                      unsigned fps_limit);
    ~SoftwareVizWindow() override;

    bool isOpen() const override;
    void close() override;
    bool setActive(bool active) override;
    void setIcon(const vizgfx::Image& icon) override;
    void setFramerateLimit(unsigned limit) override;
    void setSize(vizgfx::Vector2u size) override;
    vizgfx::Vector2u getSize() const override;
    void setView(const vizgfx::View& view) override;
    const vizgfx::View& getView() const override;
    vizgfx::Vector2f mapPixelToCoords(vizgfx::Vector2i point) const override;
    void setMouseCursor(const vizgfx::Cursor& cursor, int kind) override;
    bool isKeyPressed(vizgfx::Keyboard::Key key) const override;
    std::optional<vizgfx::Event> pollEvent() override;
    void clear(const vizgfx::Color& color) override;
    void display() override;
    vizgfx::Image captureImage() const override;

private:
    vizgfx::RenderTarget& renderTarget() override;
    static vizgfx::Keyboard::Key decodeKey(int key);

    std::unique_ptr<vizgfx::RenderTexture> texture_;
    std::unique_ptr<Host> host_;
    bool open_ = true;
    unsigned limit_ = 0;
    int cursor_ = 0;
    std::set<vizgfx::Keyboard::Key> keys_;
    std::optional<vizgfx::Event> resize_;
    std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
};

} // namespace mimir::viz

#endif
