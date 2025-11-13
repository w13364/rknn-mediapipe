#pragma once

#include "Config.hpp"
#include <vector>
#include <string>
#include <memory>
#include <tuple>

// OpenCV is required
#include <opencv2/opencv.hpp>

namespace blaze {

// Type aliases using OpenCV types
using ImageType = cv::Mat;
using Point2DType = cv::Point2f;
using Point3DType = cv::Point3d;
using RectType = cv::Rect2d;
using SizeType = cv::Size;
using AffineMatrixType = cv::Mat;

// Forward declaration
struct Detection;
struct ROI;

/**
 * Base class for all  models
 */
class Base {
public:
    Base();
    virtual ~Base() = default;
    
    // Debug and profiling methods
    void set_debug(bool debug = true);
    void set_model_ref_output(const std::string& model_ref_output1, 
                             const std::string& model_ref_output2);
    void set_dump_data(bool debug = true);
    void set_profile(bool profile = true);
    
    // Utility function for displaying shape and type info
    static void display_shape_type(const std::string& pre_msg, 
                                  const std::string& m_msg, 
                                  const cv::Size& size, 
                                  const std::string& type = "float32");

protected:
    bool DEBUG;
    bool DEBUG_USE_MODEL_REF_OUTPUT;
    std::string model_ref_output1_;
    std::string model_ref_output2_;
    bool DEBUG_DUMP_DATA;
    bool PROFILE;
};

/**
 * ROI (Region of Interest) structure
 */
struct ROI {
    double xc, yc;      // center coordinates
    double theta;       // rotation angle
    double scale;       // scale factor
};

/**
 * Detection structure
 */
struct Detection {
    double ymin, xmin, ymax, xmax;  // bounding box coordinates
    std::vector<Point2DType> keypoints;  // keypoints
    double score;       // confidence score
};

/**
 * Base class for landmark models
 */
class LandmarkBase : public Base {
public:
    LandmarkBase();
    virtual ~LandmarkBase() = default;
    
    // Extract ROI from frame
    std::tuple<std::vector<ImageType>, std::vector<AffineMatrixType>, std::vector<std::vector<Point2DType>>> 
    extract_roi(const ImageType& frame, 
                const std::vector<double>& xc, 
                const std::vector<double>& yc, 
                const std::vector<double>& theta, 
                const std::vector<double>& scale);
    
    // Denormalize landmarks
    std::vector<std::vector<Point3DType>> 
    denormalize_landmarks(std::vector<std::vector<Point3DType>>& landmarks, 
                         const std::vector<AffineMatrixType>& affines);

//protected:
    int resolution;
};

/**
 * Base class for detector models
 */
class DetectorBase : public Base {
public:
    DetectorBase();
    virtual ~DetectorBase() = default;
    
    // Configuration and setup
    void config_model(const std::string& blaze_app);
    
    // Image preprocessing
    std::tuple<ImageType, double, Point2DType> resize_pad(const ImageType& img);
    
    // Detection processing
    std::vector<Detection> denormalize_detections(std::vector<Detection>& detections, 
                                                 double scale, 
                                                 const Point2DType& pad);
    
    // Detection to ROI conversion
    std::vector<ROI> detection2roi(const std::vector<Detection>& detections);
    
    // Tensor processing methods
    std::vector<std::vector<Detection>> 
    tensors_to_detections(const std::vector<std::vector<std::vector<double>>>& raw_box_tensor,
                         const std::vector<std::vector<std::vector<double>>>& raw_score_tensor,
                         const std::vector<Anchor>& anchors);
    
    // Non-maximum suppression
    std::vector<Detection> weighted_non_max_suppression(const std::vector<Detection>& detections);

public:
    std::vector<std::vector<double>> detection_scores;
    double min_score_thresh;
    double min_suppression_threshold;

protected:
    // Configuration parameters
    AnchorOptions anchor_options_;
    std::vector<Anchor> anchors_;
    ModelConfig config_;
    
    // Model parameters
    int num_classes;
    int num_anchors;
    int num_coords;
    double score_clipping_thresh;
    double x_scale, y_scale, h_scale, w_scale;
    int num_keypoints;
    std::string detection2roi_method;
    int kp1, kp2;
    double theta0;
    double dscale;
    double dy;
    
    // Config instance
    std::unique_ptr<Config> blaze_config_;
    
private:
    // Helper methods
    std::vector<std::vector<Detection>> 
    decode_boxes(const std::vector<std::vector<std::vector<double>>>& raw_boxes, 
                const std::vector<Anchor>& anchors);
};

// Utility functions for IOU calculation
double intersect_area(const RectType& box_a, const RectType& box_b);
double jaccard_overlap(const RectType& box_a, const RectType& box_b);
double overlap_similarity(const RectType& box, const std::vector<RectType>& other_boxes, int index);
} // namespace blaze
