#pragma once

#include "include/json.hpp"
#include "training/SkeletonAnnotation.hpp"
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_set>

// Editor bookkeeping is never a training modality (nor a cache input).
inline bool isDatasetInternalFile(const std::filesystem::path& path) {
    const auto name = path.filename().string();
    return name == ".annotation_state.json" || name == ".annotation_state.lock" ||
           name == "dataset_cache.json" || name.rfind(".annotation-tmp-", 0) == 0;
}

// Keep the complete document and raw primary caption intact. Training may prepend
// grounded subject/predicate/object triplets for text-conditioned vision models.
// Optional sections support caption-only annotations and future schema extensions.
inline std::string datasetAnnotationCaption(const nlohmann::json& a,
                                            const std::string& image_file) {
    auto require = [](bool condition, const std::string& reason) {
        if (!condition) throw std::runtime_error("Dataset annotation: " + reason);
    };
    require(a.is_object(), "object required");
    require(a.contains("image") && a["image"].is_string(), "image must be a filename");
    require(a["image"].get<std::string>() == std::filesystem::path(image_file).filename().string(), "image filename mismatch");
    require(a.contains("captions") && a["captions"].is_object() &&
            a["captions"].contains("primary") && a["captions"]["primary"].is_string(), "captions.primary must be a string");
    const std::string primary = a["captions"]["primary"].get<std::string>();
    require(primary.find_first_not_of(" \t\r\n") != std::string::npos, "captions.primary is empty");
    if (a["captions"].contains("alternatives")) {
        require(a["captions"]["alternatives"].is_array(), "captions.alternatives must be an array");
        for (const auto& v : a["captions"]["alternatives"]) require(v.is_string(), "alternative must be a string");
    }
    for (const char* key : {"entities", "attributes", "relations"}) {
        if (a.contains(key)) require(a[key].is_array(), std::string(key) + " must be an array");
    }
    std::unordered_set<std::string> ids;
    if (a.contains("entities")) for (const auto& e : a["entities"]) {
        require(e.is_object() && e.contains("id") && e["id"].is_string() &&
                e.contains("phrase") && e["phrase"].is_string() &&
                e.contains("role") && e["role"].is_string(), "invalid entity");
        const auto id = e["id"].get<std::string>();
        require(!id.empty() && ids.insert(id).second, "empty or duplicate entity id");
        if (e.contains("bbox")) {
            const auto& b = e["bbox"];
            require(b.is_array() && b.size() == 4, "bbox must contain four coordinates");
            for (const auto& v : b) {
                require(v.is_number(), "bbox coordinate must be numeric");
                const double n = v.get<double>();
                require(std::isfinite(n) && n >= 0 && n <= 1, "bbox coordinate outside [0,1]");
            }
            require(b[0].get<double>() < b[2].get<double>() && b[1].get<double>() < b[3].get<double>(), "empty or inverted bbox");
        }
    }
    auto stringField = [&](const nlohmann::json& v, const char* key) {
        require(v.is_object() && v.contains(key) && v[key].is_string(), std::string("invalid ") + key);
        return v[key].get<std::string>();
    };
    if (a.contains("attributes")) for (const auto& v : a["attributes"]) {
        require(ids.count(stringField(v, "entity")) != 0, "unknown attribute entity");
        stringField(v, "property"); stringField(v, "value");
    }
    if (a.contains("relations")) for (const auto& v : a["relations"]) {
        require(ids.count(stringField(v, "subject")) != 0 && ids.count(stringField(v, "object")) != 0, "unknown relation entity");
        require(!stringField(v, "predicate").empty(), "empty predicate");
    }
    if (a.contains("lighting")) for (const char* key : {"description", "source", "target"}) stringField(a["lighting"], key);
    if (a.contains("composition")) stringField(a["composition"], "description");
    if (a.contains("quality")) {
        const auto& q = a["quality"];
        require(q.is_object(), "quality must be an object");
        for (const char* key : {"caption_verified", "relations_verified"}) {
            require(q.contains(key) && q[key].is_boolean(), std::string("invalid quality.") + key);
        }
    }
    SkeletonAnnotation::validate(a, true);
    return primary;
}
