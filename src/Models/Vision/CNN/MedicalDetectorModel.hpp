#pragma once

#include "../Model.hpp"

#include <optional>
#include <string>
#include <vector>

class MedicalDetectorModel : public Model {
public:
    struct Box {
        float x1 = 0.0f;
        float y1 = 0.0f;
        float x2 = 0.0f;
        float y2 = 0.0f;
        int class_id = 0;
        std::string label;
    };

    struct Detection : Box {
        float score = 0.0f;
    };

    struct Config {
        int image_w = 256;
        int image_h = 256;
        int image_c = 3;
        int base_channels = 8;
        int num_classes = 1;
        float object_loss_weight = 1.0f;
        float box_loss_weight = 5.0f;
        float class_loss_weight = 1.0f;
        float object_pos_weight = 8.0f;
    };

    MedicalDetectorModel();
    void buildFromConfig(const Config& cfg);
    const Config& getConfig() const { return cfg_; }

    static void buildInto(Model& model, const Config& cfg);
    std::optional<TrainStepResult> trainStep(const TrainStepRequest& request) override;

    static bool parseAnnotation(const std::string& json_text,
                                int fallback_width,
                                int fallback_height,
                                int num_classes,
                                std::vector<Box>* boxes,
                                std::string* error = nullptr);
    static bool loadAnnotationForImage(const std::string& image_path,
                                       int fallback_width,
                                       int fallback_height,
                                       int num_classes,
                                       std::vector<Box>* boxes,
                                       std::string* annotation_path = nullptr,
                                       std::string* error = nullptr);
    static size_t encodeTarget(const std::vector<Box>& boxes,
                               int grid_h,
                               int grid_w,
                               int num_classes,
                               std::vector<float>* target);
    static std::vector<Detection> decodeDetections(const std::vector<float>& output,
                                                   int grid_h,
                                                   int grid_w,
                                                   int num_classes,
                                                   float score_threshold,
                                                   float nms_iou_threshold,
                                                   int max_detections = 100);
    static float intersectionOverUnion(const Box& a, const Box& b);

private:
    Config cfg_;
};