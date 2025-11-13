#pragma once

#include "Base.hpp"

#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <chrono>
#include <cstring>

// 添加RKNN头文件
#include "rknn_api.h"

// OpenCV is required
#include <opencv2/opencv.hpp>

namespace blaze {

/**
 * Landmark - C++ port of blaze_hailo/blazelandmark.py
 * 
 * This class implements blaze landmark using the RKNN system
 * for actual RK3588 hardware inference.
 */
class Landmark : public LandmarkBase {
public:
    /**
     * Constructor
     * @param blaze_app The application type (e.g., "blazehandlandmark", "blazefacelandmark")
     */

    Landmark(const std::string& blaze_app);
    virtual ~Landmark();
    
    // Model loading and initialization
    bool load_model(const std::string& model_path);
    
    // Main prediction interface
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>
    predict(const std::vector<cv::Mat>& input_images);
    
    // Profiling accessors
    double get_profile_pre() const { return profile_pre; }
    double get_profile_model() const { return profile_model; }
    double get_profile_post() const { return profile_post; }

private:
    // Preprocess image for inference
    cv::Mat preprocess(const cv::Mat& input);

    // Member variables
    std::string blaze_app;
    
    // RKNN相关成员变量
    rknn_context ctx;
    unsigned char* model_data;
    int model_data_size;
    
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    
    // Model configuration
    cv::Size input_shape;
    cv::Size output_shape1;
    cv::Size output_shape2;
    cv::Size output_shape3;
    
    // Profiling
    double profile_pre;
    double profile_model;
    double profile_post;
};

} // namespace blaze