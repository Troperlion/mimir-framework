#include <cstddef>

// The production implementation lives in main.cpp. Unit tests link mimir_core
// without the CLI entry point, so provide the intentionally silent test sink.
void framework_log_write_file_only(const char*, std::size_t) {}
void framework_log_write(const char*, std::size_t) {}
