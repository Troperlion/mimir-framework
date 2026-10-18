#pragma once

#include "include/json.hpp"
#include "SpatialRelation.hpp"
#include "SkeletonAnnotation.hpp"
#include <unordered_map>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

namespace BBoxSupervision {

enum class Layout { HWC, CHW };
inline Layout parseLayout(const std::string& value) {
    if (value == "HWC") return Layout::HWC;
    if (value == "CHW") return Layout::CHW;
    throw std::invalid_argument("BBox layout must be HWC or CHW");
}
struct Weights {
    std::vector<float> values;
    size_t bbox_count = 0;
    size_t skeleton_region_count = 0;
    float strength = 0.0f;
    std::vector<SpatialRelation> relations;
    size_t relation_count = 0;
};

// Role weights apply to relation objectives, not standalone entity losses.
inline float roleWeight(const std::string& role, const nlohmann::json* config) {
    const bool primary=role=="primary_subject" || role=="primary_subjet";
    const bool secondary=role=="secondary_subject" || role=="secondary_subjet";
    const char* key=primary ? "relation_primary_weight" : secondary ? "relation_secondary_weight" : "relation_other_weight";
    const float value=config ? config->value(key,primary ? 3.0f : secondary ? 2.0f : 1.0f) : primary ? 3.0f : secondary ? 2.0f : 1.0f;
    if(!std::isfinite(value) || value<0) throw std::invalid_argument(std::string(key)+" must be finite and non-negative");
    return value;
}

// Equal mixture of the global objective and the mean objective of each entity.
// Each box has equal mass regardless of its size. Exact cell overlap preserves
// boxes smaller than a pixel/latent cell. The final weights have mean one.
inline Weights makeWeights(const nlohmann::json* annotation, int width, int height,
                           int channels, Layout layout, float strength = 1.0f, const nlohmann::json* config = nullptr) {
    if (!std::isfinite(strength) || strength < 0) throw std::invalid_argument("bbox_loss_weight must be finite and non-negative");
    if (width <= 0 || height <= 0 || channels <= 0) throw std::invalid_argument("Invalid bbox supervision dimensions");
    Weights result;
    if (!annotation || annotation->is_null()) return result;
    if (!annotation->is_object()) throw std::invalid_argument("BBox annotation must be an object");
    if (!annotation->contains("entities")) return result;
    const auto& entities = (*annotation)["entities"];
    // Lua represents an empty list as an empty table/object at the JSON bridge.
    if (!entities.is_array() && !(entities.is_object() && entities.empty())) throw std::invalid_argument("BBox entities must be an array");
    std::vector<std::array<double,4>> boxes;
    std::unordered_map<std::string,std::array<double,4>> entity_boxes;
    std::unordered_map<std::string,float> entity_roles;
    for (const auto& entity : entities) {
        if (!entity.is_object()) throw std::invalid_argument("Invalid bbox entity");
        if (!entity.contains("bbox")) continue;
        const auto& b = entity["bbox"];
        if (!b.is_array() || b.size() != 4) throw std::invalid_argument("BBox must contain [x_min,y_min,x_max,y_max]");
        std::array<double,4> box;
        for (size_t i=0; i<4; ++i) {
            if (!b[i].is_number()) throw std::invalid_argument("BBox coordinates must be numeric");
            box[i]=b[i].get<double>();
            if (!std::isfinite(box[i]) || box[i]<0 || box[i]>1) throw std::invalid_argument("BBox outside [0,1]");
        }
        if (box[0]>=box[2] || box[1]>=box[3]) throw std::invalid_argument("Empty or inverted bbox");
        boxes.push_back(box);
        if (entity.contains("id") && entity["id"].is_string()) {
            const auto id=entity["id"].get<std::string>();
            entity_boxes[id]=box;
            entity_roles[id]=roleWeight(entity.value("role",std::string()),config);
        }
    }
    result.bbox_count = boxes.size();
    const auto skeleton_regions=SkeletonAnnotation::regions(*annotation);
    result.skeleton_region_count=skeleton_regions.size();
    boxes.insert(boxes.end(),skeleton_regions.begin(),skeleton_regions.end());
    if (boxes.empty() || strength == 0) return result;
    result.strength = strength;
    if (annotation->contains("relations")) {
        const auto& links=(*annotation)["relations"];
        if (!links.is_array() && !(links.is_object() && links.empty())) throw std::invalid_argument("Relations must be an array");
        for (const auto& link:links) {
            if (!link.is_object() || !link.contains("subject") || !link.contains("object") || !link.contains("predicate") ||
                !link["subject"].is_string() || !link["object"].is_string() || !link["predicate"].is_string())
                throw std::invalid_argument("Invalid spatial relation");
            const auto subject=entity_boxes.find(link["subject"].get<std::string>());
            const auto object=entity_boxes.find(link["object"].get<std::string>());
            // Valid entities without a bbox still participate in text conditioning.
            if (subject==entity_boxes.end() || object==entity_boxes.end()) continue;
            if (link["predicate"].get<std::string>().empty()) throw std::invalid_argument("Empty relation predicate");
            result.relations.push_back({subject->second,object->second,width,height,channels,layout==Layout::CHW,
                strength/(1.0f+strength)*0.5f*(entity_roles.at(subject->first)+entity_roles.at(object->first)),link["predicate"].get<std::string>()});
        }
    }

    result.relation_count = result.relations.size();
    const size_t spatial_size = static_cast<size_t>(width)*height;
    std::vector<double> regional(spatial_size,0.0);
    for (const auto& box : boxes) {
        const double x0=box[0]*width, y0=box[1]*height, x1=box[2]*width, y1=box[3]*height;
        const double area=(x1-x0)*(y1-y0);
        if (!std::isfinite(area) || area<=0) throw std::invalid_argument("BBox area cannot be rasterized");
        const double scale=static_cast<double>(spatial_size)/area/boxes.size();
        const int ix0=std::max(0,static_cast<int>(std::floor(x0))), ix1=std::min(width,static_cast<int>(std::ceil(x1)));
        const int iy0=std::max(0,static_cast<int>(std::floor(y0))), iy1=std::min(height,static_cast<int>(std::ceil(y1)));
        for (int y=iy0; y<iy1; ++y) for (int x=ix0; x<ix1; ++x) {
            const double overlap_x=std::max(0.0,std::min(x1,x+1.0)-std::max(x0,static_cast<double>(x)));
            const double overlap_y=std::max(0.0,std::min(y1,y+1.0)-std::max(y0,static_cast<double>(y)));
            regional[static_cast<size_t>(y)*width+x]+=overlap_x*overlap_y*scale;
        }
    }

    if (!result.relations.empty()) {
        std::vector<double> context(spatial_size,0.0);
        double total_weight=0; for(const auto& relation:result.relations) total_weight+=relation.weight;
        for (const auto& relation:result.relations) {
            if(relation.weight==0) continue;
            const auto& a=relation.subject; const auto& b=relation.object;
            const double ax=(a[0]+a[2])*0.5, ay=(a[1]+a[3])*0.5;
            const double bx=(b[0]+b[2])*0.5, by=(b[1]+b[3])*0.5;
            const double x0=std::max(0.0,std::min(ax,bx)-0.025)*width;
            const double x1=std::min(1.0,std::max(ax,bx)+0.025)*width;
            const double y0=std::max(0.0,std::min(ay,by)-0.025)*height;
            const double y1=std::min(1.0,std::max(ay,by)+0.025)*height;
            const double scale=double(spatial_size)/((x1-x0)*(y1-y0))*relation.weight/total_weight;
            for(int y=int(std::floor(y0));y<std::min(height,int(std::ceil(y1)));++y)
                for(int x=int(std::floor(x0));x<std::min(width,int(std::ceil(x1)));++x) {
                    const double overlap=std::max(0.0,std::min(x1,x+1.0)-std::max(x0,double(x))) *
                                         std::max(0.0,std::min(y1,y+1.0)-std::max(y0,double(y)));
                    context[size_t(y)*width+x]+=overlap*scale;
                }
        }
        if(total_weight>0) for(size_t p=0;p<spatial_size;++p) regional[p]=(regional[p]+context[p])*0.5;
    }
    result.values.resize(spatial_size*channels);
    for (size_t p=0; p<spatial_size; ++p) {
        const float weight=static_cast<float>((1.0+strength*regional[p])/(1.0+strength));
        for (int c=0; c<channels; ++c) {
            const size_t i=layout==Layout::HWC ? p*channels+c : static_cast<size_t>(c)*spatial_size+p;
            result.values[i]=weight;
        }
    }
    return result;
}

// Structured geometry and references enter the text encoder as conditioning,
// independently of loss weighting. Preserve the raw caption in dataset/UI.
inline std::string groundedPrompt(const std::string& caption, const nlohmann::json* annotation) {
    if (!annotation || !annotation->is_object() || !annotation->contains("entities")) return caption;
    SkeletonAnnotation::validate(*annotation);
    std::unordered_map<std::string,std::string> phrases;
    std::ostringstream context;context<<std::fixed<<std::setprecision(3);
    for(const auto& entity:(*annotation)["entities"]) {
        if(!entity.is_object() || !entity.contains("id") || !entity["id"].is_string())continue;
        const auto id=entity["id"].get<std::string>();
        auto phrase=entity.value("phrase",std::string());if(phrase.empty())phrase=id;
        phrases[id]=phrase;
        context<<"entity "<<id<<" "<<phrase;
        if(entity.contains("role") && entity["role"].is_string())context<<" role "<<entity["role"].get<std::string>();
        if(entity.contains("bbox")) {
            const auto& box=entity["bbox"];
            if(!box.is_array() || box.size()!=4)throw std::invalid_argument("Context bbox must contain four coordinates");
            context<<" bbox ";
            for(size_t i=0;i<4;++i) {
                if(!box[i].is_number())throw std::invalid_argument("Context bbox must be numeric");
                const double value=box[i].get<double>();
                if(!std::isfinite(value) || value<0 || value>1)throw std::invalid_argument("Context bbox outside [0,1]");
                if(i)context<<",";
                context<<value;
            }
            if(box[0].get<double>()>=box[2].get<double>() || box[1].get<double>()>=box[3].get<double>())throw std::invalid_argument("Context bbox empty or inverted");
        }
        context<<". ";
    }
    context<<SkeletonAnnotation::prompt(*annotation);
    if(annotation->contains("relations") && (*annotation)["relations"].is_array()) for(const auto& link:(*annotation)["relations"]) {
        if(!link.is_object() || !link.contains("subject") || !link.contains("object") || !link.contains("predicate") ||
            !link["subject"].is_string() || !link["object"].is_string() || !link["predicate"].is_string())continue;
        const auto subject=link["subject"].get<std::string>(),object=link["object"].get<std::string>();
        if(phrases.count(subject) && phrases.count(object))context<<"relation "<<subject<<" "<<link["predicate"].get<std::string>()<<" "<<object<<": "<<phrases.at(subject)<<" "<<link["predicate"].get<std::string>()<<" "<<phrases.at(object)<<". ";
    }
    if(annotation->contains("attributes"))for(const auto& attribute:(*annotation)["attributes"]) {
        if(attribute.is_object() && attribute.contains("entity") && attribute["entity"].is_string() && attribute.contains("property") && attribute["property"].is_string() && attribute.contains("value") && attribute["value"].is_string())
            context<<"attribute "<<attribute["entity"].get<std::string>()<<" "<<attribute["property"].get<std::string>()<<" "<<attribute["value"].get<std::string>()<<". ";
    }
    for(const char* section:{"lighting","composition"})if(annotation->contains(section) && (*annotation)[section].is_object()) {
        const auto& values=(*annotation)[section];
        for(const char* key:{"description","source","target"})if(values.contains(key) && values[key].is_string() && !values[key].get<std::string>().empty())context<<section<<" "<<key<<" "<<values[key].get<std::string>()<<". ";
    }
    return context.str()+caption;
}

// Explicit grids take precedence. No spatial interpretation is guessed for tokens.
inline Weights forModel(const nlohmann::json* annotation, const nlohmann::json& config,
                        size_t input_size, size_t target_size, bool* at_output) {
    const float strength = config.value("bbox_loss_weight", 1.0f);
    auto probe = makeWeights(annotation, 1, 1, 1, Layout::HWC, strength, &config);
    *at_output = true;
    if (probe.values.empty()) return probe;
    const int w = config.value("image_w", 0), h = config.value("image_h", 0);
    const int c = config.value("image_c", 3);
    const std::string type = config.value("type", std::string());
    if (type == "vae" && w > 0 && h > 0 && target_size > static_cast<size_t>(w)*h*c) {
        auto weights=makeWeights(annotation,w,h,c,Layout::HWC,strength,&config);
        weights.values.resize(target_size,1.0f); // Preserve latent statistic losses.
        return weights;
    }
    int ow = config.value("bbox_output_w", 0), oh = config.value("bbox_output_h", 0);
    int oc = config.value("bbox_output_c", c);
    if (ow == 0 && oh == 0 && w > 0 && h > 0 &&
        target_size == static_cast<size_t>(w)*h*c && type != "vit") { ow=w; oh=h; }
    if (ow > 0 && oh > 0) {
        auto weights = makeWeights(annotation, ow, oh, oc,
            parseLayout(config.value("bbox_output_layout", std::string("HWC"))), strength, &config);
        if (weights.values.size() != target_size) throw std::invalid_argument("BBox output grid size mismatch");
        return weights;
    }
    *at_output = false;
    const int iw = config.value("bbox_input_w", w), ih = config.value("bbox_input_h", h);
    const int ic = config.value("bbox_input_c", type == "vit" ? config.value("d_model",c) : c);
    if (iw <= 0 || ih <= 0) throw std::invalid_argument("BBox supervision requires an explicit spatial grid (bbox_input_w/h/c for token or latent inputs)");
    const std::string default_layout = type == "vgg16_feat" || type == "vae" || type == "vit" || type == "patch_discriminator" ? "HWC" : "CHW";
    auto weights = makeWeights(annotation, iw, ih, ic,
        parseLayout(config.value("bbox_input_layout", default_layout)), strength, &config);
    const int prefix_tokens=config.value("bbox_input_prefix_tokens",type == "vit" ? 1 : 0);
    if (prefix_tokens<0) throw std::invalid_argument("Negative bbox input prefix");
    weights.values.insert(weights.values.begin(),static_cast<size_t>(prefix_tokens)*ic,1.0f);
    weights.relations.clear();
    if (weights.values.size() != input_size) throw std::invalid_argument("BBox input grid size mismatch");
    return weights;
}

// Existing segmentation masks retain their own weighting and zero support.
inline void combineMask(Weights& bbox, const std::vector<float>& mask) {
    if (mask.empty()) return;
    if (bbox.values.empty()) { bbox.values=mask; return; }
    if (mask.size()!=bbox.values.size()) throw std::invalid_argument("BBox/mask dimensions mismatch");
    for (size_t i=0;i<mask.size();++i) bbox.values[i]*=mask[i];
}

} // namespace BBoxSupervision
