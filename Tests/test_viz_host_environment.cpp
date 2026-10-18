#include "viz/HostEnvironment.hpp"
#include "viz/Host.hpp"
#include "viz/Ipc.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unistd.h>

extern char** environ;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        if (std::getenv("MIMIR_VIZ_ENV_TEST_CHILD")) {
            // This branch only runs if the loader survived the sanitized exec.
            require(!std::getenv("LD_PRELOAD"), "Snap preload inherited");
            require(std::string(std::getenv("LD_LIBRARY_PATH")) == "/opt/mimir/native", "Native path lost");
            require(!std::getenv("GTK_PATH"), "Snap GTK modules inherited");
            require(!std::getenv("GIO_MODULE_DIR"), "Snap GIO cache inherited");
            require(!std::getenv("GSETTINGS_SCHEMA_DIR"), "Snap GSettings schemas inherited");
            require(std::string(std::getenv("DISPLAY")) == ":987", "Display changed");
            mimir::viz::Event ready{mimir::viz::sceneProtocolHandshake};
            require(mimir::viz::transfer(STDOUT_FILENO, &ready, sizeof(ready), true), "Handshake failed");
            mimir::viz::FrameHeader header;
            require(mimir::viz::transfer(STDIN_FILENO, &header, sizeof(header), false), "No frame");
            require(header.width == 2 && header.height == 2, "Frame size changed");
            std::string scene(header.sceneBytes, '\0');
            require(mimir::viz::transfer(STDIN_FILENO, scene.data(), scene.size(), false), "No scene");
            require(scene == "{}", "Scene changed");
            mimir::viz::Event close{mimir::viz::Close};
            require(mimir::viz::transfer(STDOUT_FILENO, &close, sizeof(close), true), "Close failed");
            return 0;
        }

        std::vector<std::string> input = {
            "LD_LIBRARY_PATH=/snap/core20/current/lib:/opt/native:/home/user/snap/code/common/lib",
            "LD_PRELOAD=/snap/core20/lib/libpthread.so.0 /opt/native/profiler.so",
            "LD_AUDIT=/snap/code/1/audit.so",
            "GTK_PATH=/snap/code/264/usr/lib/gtk-3.0",
            "GIO_MODULE_DIR=/home/user/snap/code/common/.cache/gio-modules",
            "GSETTINGS_SCHEMA_DIR=/home/user/snap/code/264/.local/share/glib-2.0/schemas",
            "QT_PLUGIN_PATH=/opt/qt/plugins:/var/lib/snapd/snap/code/264/plugins",
            "XDG_DATA_DIRS=/snap/code/264/usr/share:/usr/share:/usr/local/share",
            "XDG_DATA_HOME=/home/user/snap/code/264/.local/share",
            "DISPLAY=:1", "WAYLAND_DISPLAY=wayland-0", "XAUTHORITY=/tmp/auth",
            "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus",
            "XDG_RUNTIME_DIR=/run/user/1000", "QT_QPA_PLATFORM=xcb",
            "UNCHANGED=value with spaces", "LD_DEBUG=libs"
        };
        std::vector<char*> raw;
        for (auto& value : input) raw.push_back(value.data());
        raw.push_back(nullptr);
        const auto filtered = mimir::viz::desktopHostEnvironment(raw.data());
        std::map<std::string, std::string> values;
        for (const auto& entry : filtered) values[entry.substr(0, entry.find('='))] = entry.substr(entry.find('=') + 1);
        require(values.at("LD_LIBRARY_PATH") == "/opt/native", "Library filtering failed");
        require(values.at("LD_PRELOAD") == "/opt/native/profiler.so", "Custom preload lost");
        require(values.at("QT_PLUGIN_PATH") == "/opt/qt/plugins", "Qt filtering failed");
        require(values.at("XDG_DATA_DIRS") == "/usr/share:/usr/local/share", "Data paths lost");
        for (const auto* name : {"GTK_PATH", "GIO_MODULE_DIR", "LD_AUDIT", "XDG_DATA_HOME", "GSETTINGS_SCHEMA_DIR"})
            require(values.count(name) == 0, "Snap-only setting retained");
        require(values.at("WAYLAND_DISPLAY") == "wayland-0", "Wayland changed");
        require(values.at("XAUTHORITY") == "/tmp/auth", "X authority changed");
        require(values.at("DBUS_SESSION_BUS_ADDRESS") == "unix:path=/run/user/1000/bus", "DBus changed");
        require(values.at("UNCHANGED") == "value with spaces", "Unrelated setting changed");
        require(input.front().find("/snap/") != std::string::npos, "Input modified");
        raw.clear();
        for (const auto& value : filtered) raw.push_back(const_cast<char*>(value.c_str()));
        raw.push_back(nullptr);
        require(mimir::viz::desktopHostEnvironment(raw.data()) == filtered, "Native environment modified");

        // Inject after this executable has loaded: only the child loader should
        // be exposed to these variables. No display or GUI dependency required.
        setenv("MIMIR_VIZ_HOST", std::filesystem::read_symlink("/proc/self/exe").c_str(), 1);
        setenv("MIMIR_VIZ_ENV_TEST_CHILD", "1", 1);
        setenv("LD_PRELOAD", "/snap/core20/current/lib/x86_64-linux-gnu/libpthread.so.0", 1);
        setenv("LD_LIBRARY_PATH", "/snap/core20/current/lib/x86_64-linux-gnu:/opt/mimir/native", 1);
        setenv("GTK_PATH", "/snap/code/264/usr/lib/gtk-3.0", 1);
        setenv("GIO_MODULE_DIR", "/home/user/snap/code/common/.cache/gio-modules", 1);
        setenv("GSETTINGS_SCHEMA_DIR", "/home/user/snap/code/264/.local/share/glib-2.0/schemas", 1);
        setenv("DISPLAY", ":987", 1);
        auto host = mimir::viz::makeHost("environment regression");
        mimir::viz::Frame frame{2, 2, 0, "{}"};
        host->present(std::move(frame));
        bool closed = false;
        for (int i = 0; i < 100 && !closed; ++i) {
            if (auto event = host->poll()) closed = event->type == mimir::viz::Close;
            usleep(10000);
        }
        require(closed, "Child failed to finish");
        require(std::string(std::getenv("LD_PRELOAD")).find("/snap/") == 0, "Parent environment modified");
        std::cout << "VIZ_HOST_ENVIRONMENT_OK\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
