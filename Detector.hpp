#pragma once

#include "Base.hpp"

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>
#include <atomic>

#include "rknn_api.h"
#include <opencv2/opencv.hpp>

namespace blaze {

/**
 * Detector class - C++ port of blaze_hailo/blazedetector.py
 * 
 * This class implements blaze detection using the RKNN system
 * for actual Rockchip hardware inference.
 */
class Detector : public DetectorBase {
public:
    /**
     * Constructor
     * @param blaze_app The application type (e.g., "blazepalm", "blazeface")
     */
    Detector(const std::string& blaze_app);
    
    /**
     * Destructor
     */
    virtual ~Detector();
    
    /**
     * Load a RKNN model from file
     * @param model_path Path to the RKNN model file
     */
    void load_model(const std::string& model_path);
    
    /**
     * Preprocess image data for inference
     * @param input Input image as cv::Mat
     * @return Preprocessed data as cv::Mat ready for inference
     */
    cv::Mat preprocess(const ImageType& input);
    
    /**
     * Make prediction on a single image
     * @param img Input image of shape (H, W, 3)
     * @return Detection results
     */
    std::vector<Detection> predict_on_image(const ImageType& img);
    
    /**
     * Make prediction on a batch of images
     * @param x Input batch of images of shape (b, H, W, 3)
     * @return Vector of detection results for each image
     */
    std::vector<std::vector<Detection>> predict_on_batch(const std::vector<ImageType>& x);
    
    // Profiling accessors
    double get_profile_pre() const { return profile_pre; }
    double get_profile_model() const { return profile_model; }
    double get_profile_post() const { return profile_post; }
    
    // Configuration methods
    void set_min_score_threshold(float threshold);

private:
    /**
     * Process raw model outputs into standardized tensor format
     * @param infer_results Raw model output from RKNN inference
     * @return Pair of processed tensors (scores, boxes)
     */
    std::pair<std::vector<std::vector<std::vector<float>>>, 
              std::vector<std::vector<std::vector<float>>>>
    process_model_outputs(const std::map<std::string, cv::Mat>& infer_results);
    
    /**
     * Process palm detection v0.07 outputs (6 outputs)
     */
    std::pair<std::vector<std::vector<std::vector<float>>>, 
              std::vector<std::vector<std::vector<float>>>>
    process_palm_v07_outputs(const std::map<std::string, cv::Mat>& infer_results);
    
    /**
     * Process palm detection lite outputs (4 outputs)  
     */
    std::pair<std::vector<std::vector<std::vector<float>>>, 
              std::vector<std::vector<std::vector<float>>>>
    process_palm_lite_outputs(const std::map<std::string, cv::Mat>& infer_results);

private:
    std::string blaze_app;
    
    // RKNN model information
    rknn_context ctx;                      // RKNN context
    unsigned char* model_data;             // Model data buffer
    int model_data_size;                   // Model data size
    rknn_input_output_num io_num;          // Number of inputs and outputs
    rknn_tensor_attr input_attrs[16];      // Input tensor attributes
    rknn_tensor_attr output_attrs[16];     // Output tensor attributes
    
    int num_inputs;                        // Number of inputs
    int num_outputs;                       // Number of outputs
    
    int in_shape[4];                       // Input shape [batch, height, width, channel]
    
    // Profiling
    double profile_pre;                    // Preprocessing time
    double profile_model;                  // Model inference time
    double profile_post;                   // Postprocessing time
};

} // namespace blaze