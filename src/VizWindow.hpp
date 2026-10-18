#ifndef MIMIR_VIZ_WINDOW_HPP
#define MIMIR_VIZ_WINDOW_HPP

#include "viz/Graphics.hpp"
#include <memory>
#include <optional>
#include <string>

class VizWindow {
public:
    virtual ~VizWindow() = default;
    VizWindow(const VizWindow&) = delete;
    VizWindow& operator=(const VizWindow&) = delete;

    virtual bool isOpen() const = 0;
    virtual void close() = 0;
    virtual bool setActive(bool active = true) = 0;
    virtual void setIcon(const vizgfx::Image& icon) = 0;
    virtual void setFramerateLimit(unsigned limit) = 0;
    virtual void setSize(vizgfx::Vector2u size) = 0;
    virtual vizgfx::Vector2u getSize() const = 0;
    virtual void setView(const vizgfx::View& view) = 0;
    virtual const vizgfx::View& getView() const = 0;
    virtual vizgfx::Vector2f mapPixelToCoords(vizgfx::Vector2i point) const = 0;
    virtual void setMouseCursor(const vizgfx::Cursor& cursor, int kind = 0) = 0;
    virtual bool isKeyPressed(vizgfx::Keyboard::Key key) const = 0;
    virtual std::optional<vizgfx::Event> pollEvent() = 0;
    virtual void clear(const vizgfx::Color& color = vizgfx::Color::Black) = 0;
    virtual void display() = 0;
    virtual vizgfx::Image captureImage() const = 0;

    template <typename Drawable>
    void draw(const Drawable& drawable) { renderTarget().draw(drawable); }

protected:
    VizWindow() = default;
    virtual vizgfx::RenderTarget& renderTarget() = 0;
};

std::unique_ptr<VizWindow> createVizWindow(vizgfx::Vector2u size,
                                           const std::string& title,
                                           unsigned fps_limit);
const char* vizBackendName();

#endif
