#include "test_utils.hpp"
#include "Helpers.hpp"
#include <chrono>
#include <fstream>
#include <numeric>

int main() {
    const auto path = fs::temp_directory_path() / ("mimir_rgb_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".ppm");
    struct Cleanup { fs::path path; ~Cleanup() { fs::remove(path); } } cleanup{path};
    {
        std::ofstream out(path, std::ios::binary);
        out << "P6\n2 1\n255\n";
        const unsigned char pixels[] = {255, 0, 0, 0, 255, 0};
        out.write(reinterpret_cast<const char*>(pixels), sizeof(pixels));
    }
    auto& memory = DatasetMemoryManager::instance();
    const auto baseline = memory.getCurrentRAM();
    DatasetItem item;
    item.image_file = path.string();
    item.w = 2;
    item.h = 1;
    TASSERT_TRUE(item.estimateRAMNeeded() == 6);
    TASSERT_TRUE(item.loadImage(2, 1));
    TASSERT_TRUE(item.img_c == 3 && item.img.size() == 6);
    TASSERT_TRUE(item.img[0] > 250 && item.img[1] < 5 && item.img[2] < 5);
    TASSERT_TRUE(item.img[3] < 5 && item.img[4] > 250 && item.img[5] < 5);
    TASSERT_TRUE(memory.getCurrentRAM() == baseline + 6);
    TASSERT_TRUE(item.loadImage(2, 1));
    TASSERT_TRUE(memory.getCurrentRAM() == baseline + 6);
    TASSERT_TRUE(item.loadImage(4, 2));
    TASSERT_TRUE(item.w == 4 && item.h == 2 && item.img_c == 3 && item.img.size() == 24);
    TASSERT_TRUE(memory.getCurrentRAM() == baseline + 24);
    TASSERT_TRUE(!item.loadImage(0, 2));
    item.unload();
    TASSERT_TRUE(memory.getCurrentRAM() == baseline);
    TASSERT_TRUE(item.loadImageRGB(2, 1));
    TASSERT_TRUE(item.img_c == 3 && item.img.size() == 6);
    item.unload();
    TASSERT_TRUE(memory.getCurrentRAM() == baseline);
    // Le préchargement par lots doit suivre le même contrat RGB.
    std::vector<DatasetItem> items(1);
    items[0].image_file = path.string();
    items[0].w = 2;
    items[0].h = 1;
    DatasetManager manager;
    TASSERT_TRUE(manager.ensureLoaded(items, {0}, 2, 1));
    TASSERT_TRUE(items[0].img_c == 3 && items[0].img.size() == 6);
    TASSERT_TRUE(items[0].img[0] > 250 && items[0].img[1] < 5);
    items[0].unload();
    TASSERT_TRUE(memory.getCurrentRAM() == baseline);

    const auto gif_path = fs::temp_directory_path() / ("mimir_animated_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gif");
    struct GifCleanup { fs::path path; ~GifCleanup() { fs::remove(path); } } gif_cleanup{gif_path};
    const unsigned char animated_gif[] = {
        0x47,0x49,0x46,0x38,0x39,0x61,0x02,0x00,0x01,0x00,0xf0,0x00,
        0x00,0xff,0x00,0x00,0xff,0xff,0xff,0x21,0xff,0x0b,0x4e,0x45,
        0x54,0x53,0x43,0x41,0x50,0x45,0x32,0x2e,0x30,0x03,0x01,0x02,
        0x00,0x00,0x21,0xf9,0x04,0x00,0x0c,0x00,0x00,0x00,0x2c,0x00,
        0x00,0x00,0x00,0x02,0x00,0x01,0x00,0x00,0x02,0x02,0x04,0x0a,
        0x00,0x21,0xf9,0x04,0x00,0x22,0x00,0x00,0x00,0x2c,0x00,0x00,
        0x00,0x00,0x02,0x00,0x01,0x00,0x80,0x00,0x00,0xff,0xff,0xff,
        0xff,0x02,0x02,0x04,0x0a,0x00,0x3b
    };
    {
        std::ofstream out(gif_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(animated_gif), sizeof(animated_gif));
    }
    DatasetItem animated_item;
    animated_item.image_file = gif_path.string();
    TASSERT_TRUE(animated_item.loadImageRGB(2, 1));
    TASSERT_TRUE(animated_item.isAnimatedImage());
    TASSERT_TRUE(animated_item.animationFrameCount() == 2);
    TASSERT_TRUE(animated_item.animation_frame_delays_ms.size() == 2);
    TASSERT_TRUE(animated_item.animation_frame_delays_ms[0] == 120);
    TASSERT_TRUE(animated_item.animation_frame_delays_ms[1] == 340);
    TASSERT_TRUE(animated_item.animation_frames.size() == 12);
    TASSERT_TRUE(animated_item.animation_frames[0] > 250 && animated_item.animation_frames[2] < 5);
    TASSERT_TRUE(animated_item.animation_frames[6] < 5 && animated_item.animation_frames[8] > 250);
    animated_item.unload();
    TASSERT_TRUE(memory.getCurrentRAM() == baseline);

#ifdef MIMIR_ENABLE_FFMPEG
    FFmpegVideoFrameStream animated_stream(gif_path.string(), 2, 1);
    TASSERT_TRUE(animated_stream.valid());
    std::vector<uint8_t> streamed_frame;
    TASSERT_TRUE(animated_stream.nextFrame(streamed_frame));
    TASSERT_TRUE(streamed_frame.size() == 6);
    TASSERT_TRUE(streamed_frame[0] > 250 && streamed_frame[2] < 5);
    TASSERT_TRUE(animated_stream.nextFrame(streamed_frame));
    TASSERT_TRUE(streamed_frame.size() == 6);
    TASSERT_TRUE(streamed_frame[0] < 5 && streamed_frame[2] > 250);
    TASSERT_TRUE(!animated_stream.nextFrame(streamed_frame));
    TASSERT_TRUE(!ffmpegMediaHasAudioStream(gif_path.string()));
#endif

    const auto bmp_path = fs::temp_directory_path() / ("mimir_bmp_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bmp");
    struct BmpCleanup { fs::path path; ~BmpCleanup() { fs::remove(path); } } bmp_cleanup{bmp_path};
    const unsigned char bmp_image[] = {
        0x42,0x4d,0x92,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x8a,0x00,
        0x00,0x00,0x7c,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x01,0x00,
        0x00,0x00,0x01,0x00,0x18,0x00,0x00,0x00,0x00,0x00,0x08,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xff,0x00,0x00,0xff,
        0x00,0x00,0xff,0x00,0x00,0x00,0x00,0x00,0x00,0xff,0x42,0x47,
        0x52,0x73,0x8f,0xc2,0xf5,0x28,0x51,0xb8,0x1e,0x15,0x1e,0x85,
        0xeb,0x01,0x33,0x33,0x33,0x13,0x66,0x66,0x66,0x26,0x66,0x66,
        0x66,0x06,0x99,0x99,0x99,0x09,0x3d,0x0a,0xd7,0x03,0x28,0x5c,
        0x8f,0x32,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x04,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xff,0x00,0x00,0xff
    };
    {
        std::ofstream out(bmp_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bmp_image), sizeof(bmp_image));
    }
    DatasetItem bmp_item;
    bmp_item.image_file = bmp_path.string();
    TASSERT_TRUE(bmp_item.loadImageRGB(2, 1));
    TASSERT_TRUE(bmp_item.img.size() == 6);
    TASSERT_TRUE(bmp_item.img[0] > 250 && bmp_item.img[1] < 5 && bmp_item.img[2] < 5);
    TASSERT_TRUE(!bmp_item.isAnimatedImage());
    bmp_item.unload();
    TASSERT_TRUE(memory.getCurrentRAM() == baseline);

#ifdef MIMIR_ENABLE_FFMPEG
    const auto webp_path = fs::temp_directory_path() / ("mimir_webp_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".webp");
    struct WebpCleanup { fs::path path; ~WebpCleanup() { fs::remove(path); } } webp_cleanup{webp_path};
    const unsigned char webp_image[] = {
        0x52,0x49,0x46,0x46,0x5a,0x00,0x00,0x00,0x57,0x45,0x42,0x50,
        0x56,0x50,0x38,0x58,0x0a,0x00,0x00,0x00,0x10,0x00,0x00,0x00,
        0x01,0x00,0x00,0x00,0x00,0x00,0x41,0x4c,0x50,0x48,0x03,0x00,
        0x00,0x00,0x00,0x80,0x80,0x00,0x56,0x50,0x38,0x20,0x30,0x00,
        0x00,0x00,0x10,0x02,0x00,0x9d,0x01,0x2a,0x02,0x00,0x01,0x00,
        0x02,0x00,0x34,0x25,0xa0,0x02,0x74,0xba,0x01,0xf8,0x01,0xf8,
        0x00,0x03,0xc8,0x00,0xfe,0xe6,0x69,0xdf,0xfb,0x5a,0x03,0x83,
        0x55,0xfe,0x67,0xff,0xfb,0xef,0x85,0x11,0xdb,0xaf,0xf3,0x14,
        0x00,0x00
    };
    {
        std::ofstream out(webp_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(webp_image), sizeof(webp_image));
    }
    DatasetItem webp_item;
    webp_item.image_file = webp_path.string();
    TASSERT_TRUE(webp_item.loadImageRGB(2, 1));
    TASSERT_TRUE(webp_item.img.size() == 6);
    TASSERT_TRUE(webp_item.img[0] < 30 && webp_item.img[1] > 100 && webp_item.img[2] < 30);
    webp_item.unload();
    TASSERT_TRUE(memory.getCurrentRAM() == baseline);
#endif

    const auto dataset_dir = fs::temp_directory_path() / ("mimir_mask_dataset_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct DirectoryCleanup { fs::path path; ~DirectoryCleanup() { fs::remove_all(path); } } directory_cleanup{dataset_dir};
    fs::create_directories(dataset_dir);
    const fs::path standard_path = dataset_dir / "sample.ppm";
    const fs::path mask_path = dataset_dir / "sample_mask.pgm";
    {
        std::ofstream out(standard_path, std::ios::binary);
        out << "P6\n2 1\n255\n";
        const unsigned char pixels[] = {10, 20, 30, 40, 50, 60};
        out.write(reinterpret_cast<const char*>(pixels), sizeof(pixels));
    }
    {
        std::ofstream out(mask_path, std::ios::binary);
        out << "P5\n2 1\n255\n";
        const unsigned char pixels[] = {0, 255};
        out.write(reinterpret_cast<const char*>(pixels), sizeof(pixels));
    }

    auto masked_items = loadDataset(dataset_dir.string(), 2, 1, 1);
    TASSERT_TRUE(masked_items.size() == 1);
    TASSERT_TRUE(masked_items[0].image_file == standard_path.string());
    TASSERT_TRUE(masked_items[0].mask_file == mask_path.string());
    TASSERT_TRUE(masked_items[0].loadImageRGB(2, 1));
    TASSERT_TRUE(masked_items[0].loadMask(2, 1));
    TASSERT_TRUE(masked_items[0].mask.size() == 2);
    TASSERT_TRUE(masked_items[0].mask[0] == 0 && masked_items[0].mask[1] == 255);
    const std::vector<float> weights = masked_items[0].makeMaskLossWeights(3);
    TASSERT_TRUE(weights.size() == 6);
    TASSERT_TRUE(weights[0] < weights[3]);
    const float mean_weight = std::accumulate(weights.begin(), weights.end(), 0.0f) /
        static_cast<float>(weights.size());
    TASSERT_TRUE(std::fabs(mean_weight - 1.0f) < 1e-6f);
    masked_items[0].unload();
    TASSERT_TRUE(memory.getCurrentRAM() == baseline);
    return 0;
}
