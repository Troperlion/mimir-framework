#include "viz/SoftwareVizWindow.hpp"

std::unique_ptr<VizWindow> createVizWindow(vizgfx::Vector2u size,
                                           const std::string& title,
                                           unsigned fps_limit) {
    return std::make_unique<mimir::viz::SoftwareVizWindow>(size, title, fps_limit);
}
const char* vizBackendName() { return "WEB"; }
