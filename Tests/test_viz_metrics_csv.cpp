#include "test_utils.hpp"

#include "Visualizer.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

int main() {
    const std::filesystem::path csv_path =
        std::filesystem::temp_directory_path() / "mimir_viz_metrics.csv";

    json config = {
        {"visualization", {{"enabled", false}}}
    };
    Visualizer visualizer(config);
    visualizer.updateMetrics(
        1, 4, 0.8f, 0.001f, 0.7f,
        0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2, 10, 0.9f, 12, 128, 0.5,
        3.0f, 42, 1.0f, 2.0f,
        1, 7, 0.9f, 0.999f, 1e-8f, 0.01f,
        true, false, 12, 8, 0.125f, 0.25f, 0.0f,
        "MSE", false, 8, 8, 0.1f, "penalty");
    visualizer.saveLossHistory(csv_path.string());

    std::ifstream input(csv_path);
    const std::string csv{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    std::filesystem::remove(csv_path);

    TASSERT_TRUE(csv.find("val_loss,val_mse,val_step") != std::string::npos);
    TASSERT_TRUE(csv.find(",0.125,0.25,12,penalty,0,1") != std::string::npos);
    return 0;
}