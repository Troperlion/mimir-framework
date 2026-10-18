#include "test_utils.hpp"

#include "Helpers.hpp"
#include "MemoryGuard.hpp"

#include <cstdint>
#include <string>
#include <vector>

int main() {
    auto& g = MemoryGuard::instance();
    g.reset();
    g.setLimit(100);

    TASSERT_TRUE(g.getCurrentBytes() == 0);

    // Basic limit enforcement.
    TASSERT_TRUE(g.requestAllocation(50, "a"));
    TASSERT_TRUE(g.getCurrentBytes() == 50);
    TASSERT_TRUE(!g.requestAllocation(60, "b"));
    TASSERT_TRUE(g.getCurrentBytes() == 50);

    g.releaseAllocation(50);
    TASSERT_TRUE(g.getCurrentBytes() == 0);

    // Freeze mode blocks allocations.
    g.freezeAllocations(true);
    TASSERT_TRUE(!g.requestAllocation(1, "freeze"));
    g.freezeAllocations(false);

    // Block mode blocks allocations.
    g.blockAllocations(true);
    TASSERT_TRUE(!g.requestAllocation(1, "block"));
    g.blockAllocations(false);

    auto& dataset_memory = DatasetMemoryManager::instance();
    const size_t dataset_baseline = dataset_memory.getCurrentRAM();
    DatasetItem item;
    item.name = "sample";
    item.image_file = "sample.png";
    item.text_file = "sample.txt";
    item.text = std::string(128, 'x');
    item.img = std::vector<uint8_t>(256, 42);
    item.img_loaded = true;
    item.estimated_ram_usage = item.text->size() + item.img.size();
    dataset_memory.trackAllocation(item.text->data(), item.text->size());
    dataset_memory.trackAllocation(item.img.data(), item.img.size());

    {
        DatasetItemUnloadGuard unload_item(item);
        TASSERT_TRUE(item.isLoaded());
        TASSERT_TRUE(dataset_memory.getCurrentRAM() == dataset_baseline + 384);
    }

    TASSERT_TRUE(!item.isLoaded());
    TASSERT_TRUE(item.img.empty());
    TASSERT_TRUE(!item.text.has_value());
    TASSERT_TRUE(item.estimated_ram_usage == 0);
    TASSERT_TRUE(item.name == "sample");
    TASSERT_TRUE(item.image_file == "sample.png");
    TASSERT_TRUE(item.text_file == "sample.txt");
    TASSERT_TRUE(dataset_memory.getCurrentRAM() == dataset_baseline);

    return 0;
}
