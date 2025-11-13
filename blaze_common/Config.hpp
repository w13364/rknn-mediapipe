#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>

namespace blaze {

struct AnchorOptions {
    int num_layers;
    double min_scale;
    double max_scale;
    int input_size_height;
    int input_size_width;
    double anchor_offset_x;
    double anchor_offset_y;
    std::vector<int> strides;
    std::vector<double> aspect_ratios;
    bool reduce_boxes_in_lowest_layer;
    double interpolated_scale_aspect_ratio;
    bool fixed_anchor_size;
};

struct ModelConfig {
    int num_classes;
    int num_anchors;
    int num_coords;
    double score_clipping_thresh;
    double x_scale;
    double y_scale;
    double h_scale;
    double w_scale;
    double min_score_thresh;
    double min_suppression_threshold;
    int num_keypoints;
    
    std::string detection2roi_method;
    int kp1;
    int kp2;
    double theta0;
    double dscale;
    double dy;
};

struct Anchor {
    double x_center;
    double y_center;
    double width;
    double height;
};

class Config {
public:
    Config();
    
    // Get model configuration based on model type and parameters
    ModelConfig get_model_config(const std::string& model_type, 
                                int input_width, 
                                int input_height, 
                                int num_anchors) const;
    
    // Get anchor options based on model type and parameters
    AnchorOptions get_anchor_options(const std::string& model_type, 
                                   int input_width, 
                                   int input_height, 
                                   int num_anchors) const;
    
    // Calculate scale for anchor generation
    double calculate_scale(double min_scale, 
                          double max_scale, 
                          int stride_index, 
                          int num_strides) const;
    
    // Generate anchors based on options
    std::vector<Anchor> generate_anchors(const AnchorOptions& options) const;

private:
    // Palm detection configurations
    AnchorOptions palm_detect_v0_06_anchor_options_;
    ModelConfig palm_detect_v0_06_model_config_;
    AnchorOptions palm_detect_v0_10_anchor_options_;
    ModelConfig palm_detect_v0_10_model_config_;
    
    // Face detection configurations
    AnchorOptions face_front_v0_06_anchor_options_;
    ModelConfig face_front_v0_06_model_config_;
    AnchorOptions face_back_v0_07_anchor_options_;
    ModelConfig face_back_v0_07_model_config_;
    AnchorOptions face_short_range_v0_10_anchor_options_;
    ModelConfig face_short_range_v0_10_model_config_;
    AnchorOptions face_full_range_v0_10_anchor_options_;
    ModelConfig face_full_range_v0_10_model_config_;
    
    // Pose detection configurations
    AnchorOptions pose_detect_v0_07_anchor_options_;
    ModelConfig pose_detect_v0_07_model_config_;
    AnchorOptions pose_detect_v0_10_anchor_options_;
    ModelConfig pose_detect_v0_10_model_config_;
    
    // Initialize all configurations
    void initialize_configurations();
    void initialize_palm_configurations();
    void initialize_face_configurations();
    void initialize_pose_configurations();
};

} // namespace blaze
