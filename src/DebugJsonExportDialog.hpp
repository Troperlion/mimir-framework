#pragma once
#include "LiveModelConfig.hpp"
#include <chrono>
#include <filesystem>

// Local form state; validation only queues a request on the shared control.
struct DebugJsonExportDialog {
    bool visible = false, replace = true;
    int selected = 0;
    std::string directory = "checkpoints/debug", filename, message;
    void open() {
        visible = true; selected = 0; replace = true; message.clear();
        const auto stamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        filename = "model_" + std::to_string(stamp) + ".debug.json";
    }
    void move() { selected = 1 - selected; replace = true; }
    void text(uint32_t cp) {
        auto& value = selected == 0 ? directory : filename;
        if (cp == 8 || cp == 127) {
            if (replace) value.clear();
            else if (!value.empty()) {
                do { auto byte = static_cast<unsigned char>(value.back()); value.pop_back();
                    if ((byte & 0xc0) != 0x80) break; } while (!value.empty());
            }
            replace = false;
        } else if (cp >= 32 && cp < 0x110000 && !(cp >= 0xd800 && cp <= 0xdfff) && value.size() < 4096) {
            if (replace) value.clear();
            replace = false;
            if (cp < 0x80) value += static_cast<char>(cp);
            else {
                if (cp < 0x800) value += static_cast<char>(0xc0 | (cp >> 6));
                else {
                    if (cp < 0x10000) value += static_cast<char>(0xe0 | (cp >> 12));
                    else { value += static_cast<char>(0xf0 | (cp >> 18)); value += static_cast<char>(0x80 | ((cp >> 12) & 63)); }
                    value += static_cast<char>(0x80 | ((cp >> 6) & 63));
                }
                value += static_cast<char>(0x80 | (cp & 63));
            }
        }
    }
    bool submit(LiveModelConfig& control) {
        if (directory.empty() || filename.empty() || filename == "." || filename == ".." ||
            filename.find_first_of("/\\") != std::string::npos) {
            message = "Renseigner un dossier et un nom de fichier simple.";
            return false;
        }
        try {
            const auto path = std::filesystem::path(directory) / filename;
            if (std::filesystem::exists(path)) {
                message = "Ce fichier existe deja. Choisir un autre nom.";
                return false;
            }
            if (!control.requestDebugJson(path.string())) {
                message = "Un export est deja en attente ou en cours.";
                return false;
            }
            visible = false;
            return true;
        } catch (const std::exception& e) { message = e.what(); return false; }
    }
};
