#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace mimir::viz {

// Native GUI helpers must not load the launching editor's Snap runtime. Keep
// the parent untouched, including its training/runtime library configuration.
inline bool isSnapPath(std::string_view path) {
    return path == "/snap" || path.find("/snap/") != std::string_view::npos ||
           path == "/var/lib/snapd" || path.find("/var/lib/snapd/") == 0;
}

inline std::vector<std::string> desktopHostEnvironment(char* const* environment) {
    std::vector<std::string> result;
    for (auto entry = environment; entry && *entry; ++entry) {
        const std::string variable(*entry);
        const auto equals = variable.find('=');
        const std::string_view name(variable.data(), equals == std::string::npos ? variable.size() : equals);
        const bool loader = name == "LD_PRELOAD" || name == "LD_AUDIT";
        const bool paths = loader || name == "LD_LIBRARY_PATH" ||
            name == "GTK_PATH" || name == "GTK_EXE_PREFIX" || name == "GTK_DATA_PREFIX" ||
            name == "GTK_IM_MODULE_FILE" || name == "GSETTINGS_SCHEMA_DIR" || name == "GIO_MODULE_DIR" || name == "GIO_EXTRA_MODULES" ||
            name == "GDK_PIXBUF_MODULE_FILE" || name == "GDK_PIXBUF_MODULEDIR" ||
            name == "QT_PLUGIN_PATH" || name == "QT_QPA_PLATFORM_PLUGIN_PATH" ||
            name == "QML_IMPORT_PATH" || name == "QML2_IMPORT_PATH" || name == "GI_TYPELIB_PATH" ||
            name == "LIBGL_DRIVERS_PATH" || name == "__EGL_VENDOR_LIBRARY_DIRS" ||
            name == "__EGL_VENDOR_LIBRARY_FILENAMES" ||
            name == "XDG_DATA_DIRS" || name == "XDG_DATA_HOME";
        if (!paths || equals == std::string::npos) {
            result.push_back(variable);
            continue;
        }

        const std::string value = variable.substr(equals + 1);
        std::string filtered;
        bool removed = false;
        bool kept = false;
        size_t begin = 0;
        while (begin <= value.size()) {
            const auto end = value.find_first_of(loader ? ": \t" : ":", begin);
            const auto length = end == std::string::npos ? value.size() - begin : end - begin;
            const std::string_view part(value.data() + begin, length);
            if (isSnapPath(part)) {
                removed = true;
            } else {
                if (kept) filtered += ':';
                filtered.append(part);
                kept = true;
            }
            if (end == std::string::npos) break;
            begin = end + 1;
        }
        // An absent variable lets the native toolkit use its system defaults.
        if (!removed) result.push_back(variable);
        else if (kept) result.push_back(std::string(name) + '=' + filtered);
    }
    return result;
}

} // namespace mimir::viz
