#include "Visualizer.hpp"

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    const int seconds = argc > 1 ? std::stoi(argv[1]) : 5;
    json config = {{"visualization", {
        {"enabled", true}, {"window_width", 960}, {"window_height", 640},
        {"window_title", "Mimir Viz scene smoke"}, {"fps_limit", 30}}}};
    Visualizer viz(config);
    viz.setLossLogEnabled(false);
    if (!viz.initialize()) return 1;

    std::vector<uint8_t> pixels(64 * 48 * 3);
    for (int y = 0; y < 48; ++y) for (int x = 0; x < 64; ++x) {
        pixels[(y * 64 + x) * 3] = static_cast<uint8_t>(x * 4);
        pixels[(y * 64 + x) * 3 + 1] = static_cast<uint8_t>(y * 5);
        pixels[(y * 64 + x) * 3 + 2] = 100;
    }
    viz.setDatasetImage(pixels, 64, 48, 3, "Synthetic RGB");
    viz.addGeneratedImage(pixels, 64, 48, 3, "Synthetic output");
    viz.setDatasetText("Viz backend integration", "synthetic", "1 2 3", "0.1 0.2 0.3");
    for (int i = 0; i < 30; ++i) viz.addLossPoint(1.f / (i + 1));
    viz.updateMetrics(1, 12, .125f, .001f);

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    int frames = 0;
    while (viz.isOpen() && std::chrono::steady_clock::now() < deadline) {
        viz.processEvents();
        if(!viz.isOpen()) break;
        viz.update();
        ++frames;
        if (frames == 1) std::cout << "SCENE_READY\n" << std::flush;
    }
    viz.shutdown();
    std::cout << "SCENE_FRAMES " << frames << '\n';
    return frames > 0 ? 0 : 1;
}
