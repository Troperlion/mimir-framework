#pragma once
#include "PromptParsing.hpp"

namespace TagAnnotation {
// Shared by vocabulary discovery and VGG train/validation: identical class targets.
inline std::vector<std::string> labels(const std::string& text,
    const nlohmann::json* annotation=nullptr, bool lowercase=true) {
    using json=nlohmann::json;
    json parsed;
    if(!annotation) {
        parsed=json::parse(text,nullptr,false);
        if(parsed.is_object())annotation=&parsed;
    }
    std::vector<std::string> result;
    auto add=[&](std::string value) {
        value=PromptParsing::normalizeSpaces(value);
        if(lowercase)value=PromptParsing::toLowerAscii(value);
        if(!value.empty() && std::find(result.begin(),result.end(),value)==result.end())result.push_back(std::move(value));
    };
    // Composer sidecars contain boxes with class labels; an empty list is a
    // negative sample, never a class named after a JSON key or coordinate.
    if(annotation && annotation->contains("boxes") && (*annotation)["boxes"].is_array()) {
        for(const auto& box:(*annotation)["boxes"])
            if(box.is_object() && box.contains("label") && box["label"].is_string())add(box["label"].get<std::string>());
        return result;
    }
    std::string caption=text;
    if(annotation && annotation->contains("captions") && (*annotation)["captions"].is_object()) {
        const auto& captions=(*annotation)["captions"];
        if(captions.contains("primary") && captions["primary"].is_string())caption=captions["primary"].get<std::string>();
        if(annotation->contains("entities") && (*annotation)["entities"].is_array())
            for(const auto& entity:(*annotation)["entities"])
                if(entity.is_object() && entity.contains("phrase") && entity["phrase"].is_string())add(entity["phrase"].get<std::string>());
    }
    for(const auto& candidate:PromptParsing::extractLabelCandidates(caption,lowercase))add(candidate);
    return result;
}
}
