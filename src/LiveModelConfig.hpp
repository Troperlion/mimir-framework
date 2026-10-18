#pragma once
#include "include/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// UI threads submit typed requests; only the training thread applies them.
class LiveModelConfig {
public:
    // The UI queues exports; model access stays on the training thread.
    bool requestDebugJson(const std::string& path = "checkpoints/debug/model.debug.json") {
        std::lock_guard<std::mutex> lock(mutex_);
        if (debug_pending_ || debug_busy_) return false;
        debug_path_ = path;
        debug_pending_ = true;
        debug_status_ = "DebugJSON en attente";
        return true;
    }
    bool consumeDebugJson(std::string* path = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!debug_pending_) return false;
        if (path) *path = debug_path_;
        debug_pending_ = false;
        debug_busy_ = true;
        debug_status_ = "DebugJSON en cours";
        return true;
    }
    void finishDebugJson(const std::string& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        debug_busy_ = false;
        debug_status_ = result;
    }
    std::string debugJsonStatus() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return debug_status_;
    }
    using Json = nlohmann::json;
    void requestModelGraph(bool enabled) { graph_requested_.store(enabled); }
    bool modelGraphRequested() const { return graph_requested_.load(); }
    void publishModelGraph(Json graph) {
        auto snapshot=std::make_shared<const Json>(std::move(graph));
        std::atomic_store(&model_graph_,std::move(snapshot));
    }
    std::shared_ptr<const Json> modelGraph() const { return std::atomic_load(&model_graph_); }

    enum class Owner { ReadOnly, Model, Optimizer, Training };
    struct Entry {
        std::string key;
        Json value;
        Owner owner = Owner::ReadOnly;
        double minimum = -1e30, maximum = 1e30;
        std::vector<std::string> choices;
        std::string note = "Reconstruction / redemarrage requis";
        std::optional<Json> pending;
    };
    void publish(std::vector<Entry> entries) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& e : entries) {
            auto old = entries_.find(e.key);
            if (old != entries_.end()) {
                if (e.owner == Owner::ReadOnly && old->second.owner != Owner::ReadOnly) continue;
                if (old->second.owner == Owner::Training && e.owner != Owner::Training) {
                    // The model publishes the effective value; the monitor owns
                    // how training-loop changes are delivered.
                    old->second.value = e.value;
                    continue;
                }
                e.pending = old->second.pending;
            }
            entries_[e.key] = std::move(e);
        }
    }
    std::vector<Entry> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<Entry> result;
        for (const auto& pair : entries_) result.push_back(pair.second);
        std::stable_sort(result.begin(), result.end(), [](const Entry& a, const Entry& b) {
            return (a.owner != Owner::ReadOnly) > (b.owner != Owner::ReadOnly);
        });
        return result;
    }
    bool request(const std::string& key, const std::string& text, std::string& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if (it == entries_.end()) { error = "Parametre indisponible"; return false; }
        auto& e = it->second;
        if (e.owner == Owner::ReadOnly) { error = e.note; return false; }
        Json value;
        try {
            if (e.value.is_string()) {
                value = text;
                if (!text.empty() && text.front() == '"') value = Json::parse(text);
                if (!value.is_string()) throw std::invalid_argument("string");
                if (!e.choices.empty() && std::find(e.choices.begin(), e.choices.end(), value.get<std::string>()) == e.choices.end())
                    throw std::invalid_argument("choice");
            } else {
                value = Json::parse(text);
                if (e.value.is_boolean()) {
                    if (!value.is_boolean()) throw std::invalid_argument("boolean");
                } else {
                    if (!value.is_number()) throw std::invalid_argument("number");
                    const double number = value.get<double>();
                    if (!std::isfinite(number) || number < e.minimum || number > e.maximum)
                        throw std::invalid_argument("range");
                    if (e.value.is_number_integer() && !value.is_number_integer())
                        throw std::invalid_argument("integer");
                }
            }
        } catch (...) {
            error = e.value.is_boolean() ? "Valeur attendue : true ou false" : "Valeur invalide. " + e.note;
            return false;
        }
        e.pending = std::move(value);
        error.clear();
        return true;
    }
    // The callback runs under the lock; never re-enter this object from it.
    template<class Apply> void apply(Owner owner, Apply callback) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : entries_) {
            auto& e = pair.second;
            if (e.owner != owner || !e.pending) continue;
            callback(e.key, *e.pending);
            e.value = *e.pending;
            e.pending.reset();
            overrides_[e.key] = e.value;
        }
    }
    Json overrides() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return overrides_;
    }
    static std::string display(const Json& value) {
        if (value.is_array() && value.size() > 32) return "[" + std::to_string(value.size()) + " elements]";
        const std::string text = value.is_string() ? value.get<std::string>() : value.dump();
        return text.size() > 1024 ? text.substr(0, 1021) + "..." : text;
    }
private:
    std::shared_ptr<const Json> model_graph_;
    std::atomic<bool> graph_requested_{false};
    mutable std::mutex mutex_;
    std::map<std::string, Entry> entries_;
    Json overrides_ = Json::object();
    bool debug_pending_ = false, debug_busy_ = false;
    std::string debug_status_, debug_path_;
};

// One editor per interface; the request queue and effective values are shared.
struct RuntimeConfigEditor {
    bool visible = false, editing = false, replace = false;
    size_t selected = 0;
    std::string buffer, editing_key, message;
    void toggle() { visible = !visible; editing = false; buffer.clear(); message.clear(); }
    void cancel() { if (editing) { editing = false; buffer.clear(); } else visible = false; message.clear(); }
    void move(int delta, size_t count) {
        if (editing || count == 0) return;
        selected = static_cast<size_t>(std::clamp<long long>(static_cast<long long>(selected) + delta, 0, count - 1));
        message.clear();
    }
    void enter(LiveModelConfig& config) {
        if (editing) {
            if (config.request(editing_key, buffer, message)) {
                editing = false;
                message = "Demande validee; etat affiche dans la ligne";
            }
            return;
        }
        auto rows = config.snapshot();
        if (rows.empty()) return;
        selected = std::min(selected, rows.size() - 1);
        const auto& e = rows[selected];
        if (e.owner == LiveModelConfig::Owner::ReadOnly) { message = e.note; return; }
        editing_key = e.key;
        buffer = LiveModelConfig::display(e.pending ? *e.pending : e.value);
        editing = replace = true;
        message = e.note;
    }
    void text(uint32_t cp) {
        if (!editing) return;
        if (cp == 8 || cp == 127) {
            if (replace) buffer.clear();
            else if (!buffer.empty()) {
                do { const auto c = static_cast<unsigned char>(buffer.back()); buffer.pop_back();
                    if ((c & 0xc0) != 0x80) break; } while (!buffer.empty());
            }
            replace = false;
        } else if (cp >= 32 && cp < 127 && buffer.size() < 1024) {
            if (replace) buffer.clear();
            replace = false;
            buffer.push_back(static_cast<char>(cp));
        }
    }
};
