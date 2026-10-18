#include "test_utils.hpp"

#include "HtopDisplay.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

int main() {
    const std::filesystem::path csv_path =
        std::filesystem::temp_directory_path() / "mimir_htop_metrics.csv";

    HtopDisplay display;
    display.setCsvLogFile(csv_path.string());
    display.setCsvEnabled(true);
    display.addValidationRecord(0.125f, 0.25f, 12);

    std::ifstream input(csv_path);
    const std::string csv{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    std::filesystem::remove(csv_path);

    TASSERT_TRUE(csv.find("val_loss,val_mse,val_step") != std::string::npos);
    TASSERT_TRUE(csv.find(",0.125,0.25,12") != std::string::npos);
    return 0;
}