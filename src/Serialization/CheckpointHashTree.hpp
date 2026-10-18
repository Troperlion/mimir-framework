#pragma once

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <iterator>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace Mimir::Serialization {

// One tree per load; only successfully read tensors are included.
class CheckpointHashTree {
    struct Node {
        std::map<std::string, Node> children;
        std::string hash;
    };
    Node root_;
    std::string path_;

    static bool colorEnabled() {
        if (std::getenv("NO_COLOR")) return false;
        const char* force = std::getenv("FORCE_COLOR");
        if (force) return std::string(force) != "0";
        const char* term = std::getenv("TERM");
        if (term && std::string(term) == "dumb") return false;
#ifdef _WIN32
        return _isatty(_fileno(stderr)) != 0;
#else
        return isatty(fileno(stderr)) != 0;
#endif
    }

    static void print(const Node& parent, const std::string& prefix, bool color, size_t depth = 0) {
        for (auto it = parent.children.begin(); it != parent.children.end(); ++it) {
            const bool last = std::next(it) == parent.children.end();
            const auto& node = it->second;
            const std::string branch = prefix + (last ? "└── " : "├── ");
            std::cerr << (color ? "\033[2m" : "") << branch
                      << (color ? "\033[0m" : "")
                      << (color ? (node.children.empty() ? "\033[1;36m" : "\033[1;34m") : "")
                      << it->first << (color ? "\033[0m" : "");
            if (!node.hash.empty()) {
                // Keep the existing 40-column name and algo:hash layout.
                const size_t width = (depth + 1) * 4 + it->first.size();
                std::cerr << std::string(width < 40 ? 40 - width : 0, ' ')
                          << (color ? "\033[2m" : "") << " | "
                          << (color ? "\033[0;32m" : "") << node.hash
                          << (color ? "\033[0m" : "");
            }
            std::cerr << '\n';
            print(node, prefix + (last ? "    " : "│   "), color, depth + 1);
        }
    }
public:
    void begin(const std::string& path) { root_ = {}; path_ = path; }
    void add(const std::string& name, const std::string& algo, const std::string& hash) {
        Node* node = &root_;
        size_t start = 0;
        for (size_t i = 0; i <= name.size(); ++i) {
            if (i == name.size() || name[i] == '/' || name[i] == '.') {
                if (i > start) node = &node->children[name.substr(start, i - start)];
                start = i + 1;
            }
        }
        node->hash = algo + ":" + hash;
    }
    void finish() const {
        if (root_.children.empty()) return;
        const bool color = colorEnabled();
        std::cerr << (color ? "\033[1;34m" : "") << path_
                  << (color ? "\033[0m\033[2m" : "") << " [checkpoint hashsums]"
                  << (color ? "\033[0m" : "") << '\n';
        print(root_, "", color);
        std::cerr.flush();
    }
    struct Scope {
        CheckpointHashTree& tree;
        ~Scope() { try { tree.finish(); } catch (...) {} }
    };
};

} // namespace Mimir::Serialization
