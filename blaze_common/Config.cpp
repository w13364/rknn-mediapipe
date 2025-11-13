#include "Config.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace blaze {

Config::Config() {
    initialize_configurations();
}

void Config::initialize_configurations() {
    initialize_palm_configurations();
    initialize_face_configurations();
    initialize_pose_configurations();
}

void Config::initialize_palm_configurations() {
    // Palm detect v0.06 anchor options
    palm_detect_v0_06_anchor_options_ = {
        .num_layers = 5,
        .min_scale = 0.1171875,
        .max_scale = 0.75,
        .input_size_height = 256,
        .input_size_width = 256,
        .anchor_offset_x = 0.5,
        .anchor_offset_y = 0.5,
        .strides = {8, 16, 32, 32, 32},
        .aspect_ratios = {1.0},
        .reduce_boxes_in_lowest_layer = false,
        .interpolated_scale_aspect_ratio = 1.0,
        .fixed_anchor_size = true
    };
    
    // Palm detect v0.06 model config
    palm_detect_v0_06_model_config_ = {
        .num_classes = 1,
        .num_anchors = 2944,
        .num_coords = 18,
        .score_clipping_thresh = 100.0,
        .x_scale = 256.0,
        .y_scale = 256.0,
        .h_scale = 256.0,
        .w_scale = 256.0,
        .min_score_thresh = 0.7,
        .min_suppression_threshold = 0.3,
        .num_keypoints = 7,
        .detection2roi_method = "box",
        .kp1 = 0,
        .kp2 = 2,
        .theta0 = M_PI / 2,
        .dscale = 2.6,
        .dy = -0.5
    };
    
    // Palm detect v0.10 anchor options
    palm_detect_v0_10_anchor_options_ = {
        .num_layers = 4,
        .min_scale = 0.1484375,
        .max_scale = 0.75,
        .input_size_height = 192,
        .input_size_width = 192,
        .anchor_offset_x = 0.5,
        .anchor_offset_y = 0.5,
        .strides = {8, 16, 16, 16},
        .aspect_ratios = {1.0},
        .reduce_boxes_in_lowest_layer = false,
        .interpolated_scale_aspect_ratio = 1.0,
        .fixed_anchor_size = true
    };
    
    // Palm detect v0.10 model config
    palm_detect_v0_10_model_config_ = {
        .num_classes = 1,
        .num_anchors = 2016,
        .num_coords = 18,
        .score_clipping_thresh = 100.0,
        .x_scale = 192.0,
        .y_scale = 192.0,
        .h_scale = 192.0,
        .w_scale = 192.0,
        .min_score_thresh = 0.5,
        .min_suppression_threshold = 0.3,
        .num_keypoints = 7,
        .detection2roi_method = "box",
        .kp1 = 0,
        .kp2 = 2,
        .theta0 = M_PI / 2,
        .dscale = 2.6,
        .dy = -0.5
    };
}

void Config::initialize_face_configurations() {
    // Face front v0.06 anchor options
    face_front_v0_06_anchor_options_ = {
        .num_layers = 4,
        .min_scale = 0.1484375,
        .max_scale = 0.75,
        .input_size_height = 128,
        .input_size_width = 128,
        .anchor_offset_x = 0.5,
        .anchor_offset_y = 0.5,
        .strides = {8, 16, 16, 16},
        .aspect_ratios = {1.0},
        .reduce_boxes_in_lowest_layer = false,
        .interpolated_scale_aspect_ratio = 1.0,
        .fixed_anchor_size = true
    };
    
    // Face front v0.06 model config
    face_front_v0_06_model_config_ = {
        .num_classes = 1,
        .num_anchors = 896,
        .num_coords = 16,
        .score_clipping_thresh = 100.0,
        .x_scale = 128.0,
        .y_scale = 128.0,
        .h_scale = 128.0,
        .w_scale = 128.0,
        .min_score_thresh = 0.75,
        .min_suppression_threshold = 0.3,
        .num_keypoints = 6,
        .detection2roi_method = "box",
        .kp1 = 1,
        .kp2 = 0,
        .theta0 = 0.0,
        .dscale = 1.5,
        .dy = 0.0
    };
    
    // Face back v0.07 anchor options
    face_back_v0_07_anchor_options_ = {
        .num_layers = 4,
        .min_scale = 0.15625,
        .max_scale = 0.75,
        .input_size_height = 256,
        .input_size_width = 256,
        .anchor_offset_x = 0.5,
        .anchor_offset_y = 0.5,
        .strides = {16, 32, 32, 32},
        .aspect_ratios = {1.0},
        .reduce_boxes_in_lowest_layer = false,
        .interpolated_scale_aspect_ratio = 1.0,
        .fixed_anchor_size = true
    };
    
    // Face back v0.07 model config
    face_back_v0_07_model_config_ = {
        .num_classes = 1,
        .num_anchors = 896,
        .num_coords = 16,
        .score_clipping_thresh = 100.0,
        .x_scale = 256.0,
        .y_scale = 256.0,
        .h_scale = 256.0,
        .w_scale = 256.0,
        .min_score_thresh = 0.65,
        .min_suppression_threshold = 0.3,
        .num_keypoints = 6,
        .detection2roi_method = "box",
        .kp1 = 1,
        .kp2 = 0,
        .theta0 = 0.0,
        .dscale = 1.5,
        .dy = 0.0
    };
    
    // Face short range v0.10 anchor options
    face_short_range_v0_10_anchor_options_ = {
        .num_layers = 4,
        .min_scale = 0.1484375,
        .max_scale = 0.75,
        .input_size_height = 128,
        .input_size_width = 128,
        .anchor_offset_x = 0.5,
        .anchor_offset_y = 0.5,
        .strides = {8, 16, 16, 16},
        .aspect_ratios = {1.0},
        .reduce_boxes_in_lowest_layer = false,
        .interpolated_scale_aspect_ratio = 1.0,
        .fixed_anchor_size = true
    };
    
    // Face short range v0.10 model config
    face_short_range_v0_10_model_config_ = {
        .num_classes = 1,
        .num_anchors = 896,
        .num_coords = 16,
        .score_clipping_thresh = 100.0,
        .x_scale = 128.0,
        .y_scale = 128.0,
        .h_scale = 128.0,
        .w_scale = 128.0,
        .min_score_thresh = 0.5,
        .min_suppression_threshold = 0.3,
        .num_keypoints = 6,
        .detection2roi_method = "box",
        .kp1 = 1,
        .kp2 = 0,
        .theta0 = 0.0,
        .dscale = 1.5,
        .dy = 0.0
    };
    
    // Face full range v0.10 anchor options
    face_full_range_v0_10_anchor_options_ = {
        .num_layers = 1,
        .min_scale = 0.1484375,
        .max_scale = 0.75,
        .input_size_height = 192,
        .input_size_width = 192,
        .anchor_offset_x = 0.5,
        .anchor_offset_y = 0.5,
        .strides = {4},
        .aspect_ratios = {1.0},
        .reduce_boxes_in_lowest_layer = false,
        .interpolated_scale_aspect_ratio = 0.0,
        .fixed_anchor_size = true
    };
    
    // Face full range v0.10 model config
    face_full_range_v0_10_model_config_ = {
        .num_classes = 1,
        .num_anchors = 2304,
        .num_coords = 16,
        .score_clipping_thresh = 100.0,
        .x_scale = 192.0,
        .y_scale = 192.0,
        .h_scale = 192.0,
        .w_scale = 192.0,
        .min_score_thresh = 0.6,
        .min_suppression_threshold = 0.3,
        .num_keypoints = 6,
        .detection2roi_method = "box",
        .kp1 = 1,
        .kp2 = 0,
        .theta0 = 0.0,
        .dscale = 1.5,
        .dy = 0.0
    };
}

void Config::initialize_pose_configurations() {
    // Pose detect v0.07 anchor options
    pose_detect_v0_07_anchor_options_ = {
        .num_layers = 4,
        .min_scale = 0.1484375,
        .max_scale = 0.75,
        .input_size_height = 128,
        .input_size_width = 128,
        .anchor_offset_x = 0.5,
        .anchor_offset_y = 0.5,
        .strides = {8, 16, 16, 16},
        .aspect_ratios = {1.0},
        .reduce_boxes_in_lowest_layer = false,
        .interpolated_scale_aspect_ratio = 1.0,
        .fixed_anchor_size = true
    };
    
    // Pose detect v0.07 model config
    pose_detect_v0_07_model_config_ = {
        .num_classes = 1,
        .num_anchors = 896,
        .num_coords = 12,
        .score_clipping_thresh = 100.0,
        .x_scale = 128.0,
        .y_scale = 128.0,
        .h_scale = 128.0,
        .w_scale = 128.0,
        .min_score_thresh = 0.5,
        .min_suppression_threshold = 0.3,
        .num_keypoints = 4,
        .detection2roi_method = "alignment",
        .kp1 = 2,
        .kp2 = 3,
        .theta0 = 90 * M_PI / 180,
        .dscale = 1.5,
        .dy = 0.0
    };
    
    // Pose detect v0.10 anchor options
    pose_detect_v0_10_anchor_options_ = {
        .num_layers = 5,
        .min_scale = 0.1484375,
        .max_scale = 0.75,
        .input_size_height = 224,
        .input_size_width = 224,
        .anchor_offset_x = 0.5,
        .anchor_offset_y = 0.5,
        .strides = {8, 16, 32, 32, 32},
        .aspect_ratios = {1.0},
        .reduce_boxes_in_lowest_layer = false,
        .interpolated_scale_aspect_ratio = 1.0,
        .fixed_anchor_size = true
    };
    
    // Pose detect v0.10 model config
    pose_detect_v0_10_model_config_ = {
        .num_classes = 1,
        .num_anchors = 2254,
        .num_coords = 12,
        .score_clipping_thresh = 100.0,
        .x_scale = 224.0,
        .y_scale = 224.0,
        .h_scale = 224.0,
        .w_scale = 224.0,
        .min_score_thresh = 0.5,
        .min_suppression_threshold = 0.3,
        .num_keypoints = 4,
        .detection2roi_method = "alignment",
        .kp1 = 2,
        .kp2 = 3,
        .theta0 = 90 * M_PI / 180,
        .dscale = 2.5,
        .dy = 0.0
    };
}

ModelConfig Config::get_model_config(const std::string& model_type, 
                                         int input_width, 
                                         int input_height, 
                                         int num_anchors) const {
    if (model_type == "blazepalm") {
        if (input_width == 256) {
            return palm_detect_v0_06_model_config_;
        } else if (input_width == 192) {
            return palm_detect_v0_10_model_config_;
        } else {
            // Default to v0.06 if no exact match
            return palm_detect_v0_06_model_config_;
        }
    } else if (model_type == "blazeface") {
        if (input_width == 128) {
            return face_front_v0_06_model_config_;
        } else if (input_width == 256) {
            return face_back_v0_07_model_config_;
        } else if (input_width == 192) {
            return face_full_range_v0_10_model_config_;
        } else {
            // Default to front face if no exact match
            return face_front_v0_06_model_config_;
        }
    } else if (model_type == "blazepose") {
        if (input_width == 128) {
            return pose_detect_v0_07_model_config_;
        } else if (input_width == 224) {
            return pose_detect_v0_10_model_config_;
        } else {
            // Default to v0.07 if no exact match
            return pose_detect_v0_07_model_config_;
        }
    } else {
        std::cerr << "[Config.get_model_config] ERROR : Unsupported Model Type : " << model_type << std::endl;
    }
    
    // Return default config if no match found
    return ModelConfig{};
}

AnchorOptions Config::get_anchor_options(const std::string& model_type, 
                                             int input_width, 
                                             int input_height, 
                                             int num_anchors) const {
    if (model_type == "blazepalm") {
        if (input_width == 256) {
            return palm_detect_v0_06_anchor_options_;
        } else if (input_width == 192) {
            return palm_detect_v0_10_anchor_options_;
        } else {
            // Default to v0.06 if no exact match
            return palm_detect_v0_06_anchor_options_;
        }
    } else if (model_type == "blazeface") {
        if (input_width == 128) {
            return face_front_v0_06_anchor_options_;
        } else if (input_width == 256) {
            return face_back_v0_07_anchor_options_;
        } else if (input_width == 192) {
            return face_full_range_v0_10_anchor_options_;
        } else {
            // Default to front face if no exact match
            return face_front_v0_06_anchor_options_;
        }
    } else if (model_type == "blazepose") {
        if (input_width == 128) {
            return pose_detect_v0_07_anchor_options_;
        } else if (input_width == 224) {
            return pose_detect_v0_10_anchor_options_;
        } else {
            // Default to v0.07 if no exact match
            return pose_detect_v0_07_anchor_options_;
        }
    } else {
        std::cerr << "[Config.get_anchor_options] ERROR : Unsupported Model Type : " << model_type << std::endl;
    }
    
    // Return default options if no match found
    return AnchorOptions{};
}

double Config::calculate_scale(double min_scale, 
                                   double max_scale, 
                                   int stride_index, 
                                   int num_strides) const {
    if (num_strides == 1) {
        return (max_scale + min_scale) * 0.5;
    } else {
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0);
    }
}

std::vector<Anchor> Config::generate_anchors(const AnchorOptions& options) const {
    size_t strides_size = options.strides.size();
    assert(options.num_layers == static_cast<int>(strides_size));

    std::vector<Anchor> anchors;
    size_t layer_id = 0;
    
    while (layer_id < strides_size) {
        std::vector<double> anchor_height;
        std::vector<double> anchor_width;
        std::vector<double> aspect_ratios;
        std::vector<double> scales;

        // For same strides, we merge the anchors in the same order.
        size_t last_same_stride_layer = layer_id;
        while ((last_same_stride_layer < strides_size) && 
               (options.strides[last_same_stride_layer] == options.strides[layer_id])) {
            
            double scale = calculate_scale(options.min_scale,
                                         options.max_scale,
                                         static_cast<int>(last_same_stride_layer),
                                         static_cast<int>(strides_size));

            if (last_same_stride_layer == 0 && options.reduce_boxes_in_lowest_layer) {
                // For first layer, it can be specified to use predefined anchors.
                aspect_ratios.push_back(1.0);
                aspect_ratios.push_back(2.0);
                aspect_ratios.push_back(0.5);
                scales.push_back(0.1);
                scales.push_back(scale);
                scales.push_back(scale);
            } else {
                for (double aspect_ratio : options.aspect_ratios) {
                    aspect_ratios.push_back(aspect_ratio);
                    scales.push_back(scale);
                }

                if (options.interpolated_scale_aspect_ratio > 0.0) {
                    double scale_next = (last_same_stride_layer == strides_size - 1) ? 
                                      1.0 : 
                                      calculate_scale(options.min_scale,
                                                    options.max_scale,
                                                    static_cast<int>(last_same_stride_layer + 1),
                                                    static_cast<int>(strides_size));
                    scales.push_back(std::sqrt(scale * scale_next));
                    aspect_ratios.push_back(options.interpolated_scale_aspect_ratio);
                }
            }

            last_same_stride_layer++;
        }

        for (size_t i = 0; i < aspect_ratios.size(); ++i) {
            double ratio_sqrts = std::sqrt(aspect_ratios[i]);
            anchor_height.push_back(scales[i] / ratio_sqrts);
            anchor_width.push_back(scales[i] * ratio_sqrts);
        }

        int stride = options.strides[layer_id];
        int feature_map_height = static_cast<int>(std::ceil(static_cast<double>(options.input_size_height) / stride));
        int feature_map_width = static_cast<int>(std::ceil(static_cast<double>(options.input_size_width) / stride));

        for (int y = 0; y < feature_map_height; ++y) {
            for (int x = 0; x < feature_map_width; ++x) {
                for (size_t anchor_id = 0; anchor_id < anchor_height.size(); ++anchor_id) {
                    double x_center = (x + options.anchor_offset_x) / feature_map_width;
                    double y_center = (y + options.anchor_offset_y) / feature_map_height;

                    Anchor new_anchor;
                    new_anchor.x_center = x_center;
                    new_anchor.y_center = y_center;
                    
                    if (options.fixed_anchor_size) {
                        new_anchor.width = 1.0;
                        new_anchor.height = 1.0;
                    } else {
                        new_anchor.width = anchor_width[anchor_id];
                        new_anchor.height = anchor_height[anchor_id];
                    }
                    
                    anchors.push_back(new_anchor);
                }
            }
        }

        layer_id = last_same_stride_layer;
    }

    return anchors;
}

} // namespace blaze
