#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "Base.hpp"
#include "DebugLog.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace blaze {

// Initialize the global debug stream instance
DebugStream dbgout;

// ============================================================================
// Base Implementation
// ============================================================================

Base::Base() 
    : DEBUG(false)
    , DEBUG_USE_MODEL_REF_OUTPUT(false)
    , model_ref_output1_("")
    , model_ref_output2_("")
    , DEBUG_DUMP_DATA(false)
    , PROFILE(false) {
}

void Base::set_debug(bool debug) {
    DEBUG = debug;
}

void Base::set_model_ref_output(const std::string& model_ref_output1, 
                                    const std::string& model_ref_output2) {
    DEBUG_USE_MODEL_REF_OUTPUT = true;
    model_ref_output1_ = model_ref_output1;
    model_ref_output2_ = model_ref_output2;
}

void Base::set_dump_data(bool debug) {
    DEBUG_DUMP_DATA = debug;
}

void Base::set_profile(bool profile) {
    PROFILE = profile;
}

void Base::display_shape_type(const std::string& pre_msg, 
                                  const std::string& m_msg, 
                                  const cv::Size& size, 
                                  const std::string& type) {
    std::cout << pre_msg << " " << m_msg 
              << " shape=[" << size.height << "x" << size.width << "]"
              << " dtype=" << type << std::endl;
}

// ============================================================================
// LandmarkBase Implementation
// ============================================================================

LandmarkBase::LandmarkBase() 
    : Base()
    , resolution(256) {
}

std::tuple<std::vector<ImageType>, std::vector<AffineMatrixType>, std::vector<std::vector<Point2DType>>> 
LandmarkBase::extract_roi(const ImageType& frame, 
                              const std::vector<double>& xc, 
                              const std::vector<double>& yc, 
                              const std::vector<double>& theta, 
                              const std::vector<double>& scale) {

    // 打印ROI区域参数　roi打印添加
    std::cout << "[LandmarkBase.extract_roi] 提取的ROI区域参数:" << std::endl;
    std::cout << "ROI数量: " << xc.size() << std::endl;
    for (size_t i = 0; i < xc.size(); ++i) {
        std::cout << "ROI " << i << ":" << std::endl;
        std::cout << "  xc = " << xc[i] << " (中心x坐标)" << std::endl;
        std::cout << "  yc = " << yc[i] << " (中心y坐标)" << std::endl;
        std::cout << "  theta = " << theta[i] << " 弧度 (" << theta[i] * 180.0 / M_PI << " 度) (旋转角度)" << std::endl;
        std::cout << "  scale = " << scale[i] << " (缩放因子)" << std::endl;
    }
    
    std::vector<cv::Mat> imgs;
    std::vector<cv::Mat> affines;
    std::vector<std::vector<cv::Point2f>> points_result;
    
    // Define template points (corners of unit square)

    std::vector<cv::Point2f> template_points = {
        {-1.0f, -1.0f}, {-1.0f, 1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}
    };
    
    // Destination points for affine transformation: top-left, bottom-left, top-right
    std::vector<cv::Point2f> dst_points = {
        {0.0f, 0.0f},
        {0.0f, static_cast<float>(resolution - 1)},
        {static_cast<float>(resolution - 1), 0.0f}
    };
    
    for (size_t i = 0; i < xc.size(); ++i) {
        // Scale the template points
        std::vector<cv::Point2f> scaled_points;
        for (const auto& pt : template_points) {
            scaled_points.push_back({pt.x * static_cast<float>(scale[i] / 2), 
                                   pt.y * static_cast<float>(scale[i] / 2)});
        }

        std::cout << "  缩放后模板点(scaled_points):" << std::endl;
        for (size_t j = 0; j < scaled_points.size(); ++j) {
            std::cout << "    点" << j+1 << ": (" << scaled_points[j].x << ", " << scaled_points[j].y << ")" << std::endl;
        }
        
        
        // Apply rotation
        float cos_theta = std::cos(static_cast<float>(theta[i]));
        float sin_theta = std::sin(static_cast<float>(theta[i]));

        std::cout << "  旋转参数: cos(theta)=" << cos_theta << ", sin(theta)=" << sin_theta << std::endl;

        
        std::vector<cv::Point2f> rotated_points;
        for (const auto& pt : scaled_points) {
            float x_rot = cos_theta * pt.x - sin_theta * pt.y;
            float y_rot = sin_theta * pt.x + cos_theta * pt.y;
            rotated_points.push_back({x_rot + static_cast<float>(xc[i]), 
                                    y_rot + static_cast<float>(yc[i])});
        }

        std::cout << "  旋转后点(rotated_points):" << std::endl;
        for (size_t j = 0; j < rotated_points.size(); ++j) {
            std::cout << "    点" << j+1 << ": (" << rotated_points[j].x << ", " << rotated_points[j].y << ")" << std::endl;
        }

        
        // Get first 3 points for affine transformation
        std::vector<cv::Point2f> src_points = {
            rotated_points[0], rotated_points[1], rotated_points[2]
        };

        std::cout << "  仿射变换源点(src_points):" << std::endl;
        for (size_t j = 0; j < src_points.size(); ++j) {
            std::cout << "    点" << j+1 << ": (" << src_points[j].x << ", " << src_points[j].y << ")" << std::endl;
        }
        

        // Calculate affine transformation matrix
        // Compute affine transform in double precision
        cv::Mat M = cv::getAffineTransform(src_points, dst_points);  // CV_64F

        std::cout << "  仿射变换矩阵(M):" << std::endl;
        std::cout << "    [" << M.at<double>(0,0) << ", " << M.at<double>(0,1) << ", " << M.at<double>(0,2) << "]" << std::endl;
        std::cout << "    [" << M.at<double>(1,0) << ", " << M.at<double>(1,1) << ", " << M.at<double>(1,2) << "]" << std::endl;
        
        
        // Apply transformation
        cv::Mat warped_img;
        cv::warpAffine(frame, warped_img, M, cv::Size(resolution, resolution));

        std::cout << "  提取的ROI图像形状: " << warped_img.size() << " 数据类型: " << warped_img.type() << std::endl;

        
        // Convert to float and normalize
        warped_img.convertTo(warped_img, CV_32F, 1.0/255.0);


                // 计算ROI图像的最小值和最大值
        double min_val, max_val;
        cv::minMaxLoc(warped_img, &min_val, &max_val);
        std::cout << "  ROI图像预处理后 - 最小值: " << min_val << ", 最大值: " << max_val << std::endl;
        std::cout << "  预处理后ROI图像形状: " << warped_img.size() << " 数据类型: " << warped_img.type() << std::endl;
        
        // Calculate inverse affine transformation in double precision
        cv::Mat M_inv;
        cv::invertAffineTransform(M, M_inv);  // M_inv is CV_64F
        
        
        std::cout << "  逆仿射变换矩阵(M_inv):" << std::endl;
        std::cout << "    [" << M_inv.at<double>(0,0) << ", " << M_inv.at<double>(0,1) << ", " << M_inv.at<double>(0,2) << "]" << std::endl;
        std::cout << "    [" << M_inv.at<double>(1,0) << ", " << M_inv.at<double>(1,1) << ", " << M_inv.at<double>(1,2) << "]" << std::endl;
        
        
        imgs.push_back(warped_img);
        affines.push_back(M_inv);
        points_result.push_back(rotated_points);
    }
    
    return std::make_tuple(imgs, affines, points_result);
}

std::vector<std::vector<Point3DType>> 
LandmarkBase::denormalize_landmarks(std::vector<std::vector<Point3DType>>& landmarks, 
                                        const std::vector<AffineMatrixType>& affines) {
    
    if (DEBUG) {
        // Debug: Print input data
        std::cout << "[DEBUG denormalize_landmarks] Input resolution: " << resolution << std::endl;
        std::cout << "[DEBUG denormalize_landmarks] Number of landmark batches: " << landmarks.size() << std::endl;
        std::cout << "[DEBUG denormalize_landmarks] Number of affine matrices: " << affines.size() << std::endl;
    }
    // Print first few input landmarks before any processing
    if (!landmarks.empty() && !landmarks[0].empty()) {
        if (DEBUG) {
            std::cout << "[DEBUG denormalize_landmarks] Raw input landmarks (first 3):" << std::endl;
            for (size_t j = 0; j < std::min((size_t)3, landmarks[0].size()); j++) {
                std::cout << "  [" << j << "]: (" << landmarks[0][j].x << ", " << landmarks[0][j].y << ", " << landmarks[0][j].z << ")" << std::endl;
            }
        }
    }
    // Determine if landmarks are normalized
    bool landmarks_are_normalized = false;
    if (!landmarks.empty() && !landmarks[0].empty()) {
        // Check if landmarks are in [0,1] range (normalized) or already in pixel coordinates
        bool all_in_normalized_range = true;
        for (size_t i = 0; i < std::min((size_t)5, landmarks[0].size()); i++) {
            if (landmarks[0][i].x > 1.0 || landmarks[0][i].y > 1.0 || 
                landmarks[0][i].x < 0.0 || landmarks[0][i].y < 0.0) {
                all_in_normalized_range = false;
                break;
            }
        }
        landmarks_are_normalized = all_in_normalized_range;
    }
    
    if (DEBUG) {
        std::cout << "[DEBUG denormalize_landmarks] Landmarks are " << (landmarks_are_normalized ? "normalized [0,1]" : "already in pixel coordinates") << std::endl;
    }
    
    // Apply scaling only if landmarks are normalized
    if (landmarks_are_normalized) {

        for (size_t i = 0; i < landmarks.size(); ++i) {
            for (auto& landmark : landmarks[i]) {
                // Scale only x and y by resolution (z remains in normalized units)
                landmark.x *= resolution;
                landmark.y *= resolution;

            }
        }
        
        // Debug: Print landmarks after scaling by resolution
        if (!landmarks.empty() && !landmarks[0].empty()) {
            if (DEBUG) {
                std::cout << "[DEBUG denormalize_landmarks] After scaling by resolution (first 3):" << std::endl;
                for (size_t j = 0; j < std::min((size_t)3, landmarks[0].size()); j++) {
                    std::cout << "  [" << j << "]: (" << landmarks[0][j].x << ", " << landmarks[0][j].y << ", " << landmarks[0][j].z << ")" << std::endl;
                }
            }
        }
    } else {
        if (DEBUG) {
            std::cout << "[DEBUG denormalize_landmarks] Skipping scaling - landmarks already in pixel coordinates" << std::endl;
        }
    }
    
    // Apply affine transformation (stored affines are already inverse from extract_roi)

    for (size_t i = 0; i < landmarks.size(); ++i) {
        if (i < affines.size()) {
            const cv::Mat& affine = affines[i];  // This is the inverse matrix from extract_roi
            
            if (DEBUG) {
                // Debug: Print affine matrix
                std::cout << "[DEBUG denormalize_landmarks] Affine matrix [" << i << "]:" << std::endl;
                std::cout << "  [" << affine.at<double>(0,0) << ", " << affine.at<double>(0,1) << ", " << affine.at<double>(0,2) << "]" << std::endl;
                std::cout << "  [" << affine.at<double>(1,0) << ", " << affine.at<double>(1,1) << ", " << affine.at<double>(1,2) << "]" << std::endl;
            }
            for (auto& landmark : landmarks[i]) {
                // Store original values for debugging
                double orig_x = landmark.x, orig_y = landmark.y;
                
                // Extract affine matrix components (stored as double in CV_64F)
                double a11 = affine.at<double>(0, 0);
                double a12 = affine.at<double>(0, 1);
                double a13 = affine.at<double>(0, 2);
                double a21 = affine.at<double>(1, 0);
                double a22 = affine.at<double>(1, 1);
                double a23 = affine.at<double>(1, 2);
                
                // Apply transformation: (affine[:,:2] @ landmark[:,:2].T + affine[:,2:]).T
                double x_new = a11 * landmark.x + a12 * landmark.y + a13;
                double y_new = a21 * landmark.x + a22 * landmark.y + a23;
                
                landmark.x = x_new;
                landmark.y = y_new;
                // landmark.z unchanged - stays in normalized depth units
            }
            
            // Debug: Print landmarks after affine transformation
            if (i == 0 && !landmarks[i].empty()) {
                if (DEBUG) {
                    std::cout << "[DEBUG denormalize_landmarks] After affine transformation (first 3):" << std::endl;
                    for (size_t j = 0; j < std::min((size_t)3, landmarks[i].size()); j++) {
                        std::cout << "  [" << j << "]: (" << landmarks[i][j].x << ", " << landmarks[i][j].y << ", " << landmarks[i][j].z << ")" << std::endl;
                    }
                }
            }
        }
    }
    
    return landmarks;
}

// ============================================================================
// DetectorBase Implementation
// ============================================================================

DetectorBase::DetectorBase() 
    : Base()
    , num_classes(1)
    , num_anchors(2944)  // Set default to palm detector anchors
    , num_coords(18)
    , score_clipping_thresh(100.0)
    , x_scale(256.0)     // Set default to palm detector scale
    , y_scale(256.0)
    , h_scale(256.0)
    , w_scale(256.0)
    , min_score_thresh(0.5)
    , min_suppression_threshold(0.3)
    , num_keypoints(7)   // Set default to palm detector keypoints
    , detection2roi_method("box")  // Ensure this is properly initialized
    , kp1(0)
    , kp2(2)
    , theta0(M_PI/2)
    , dscale(2.6)
    , dy(-0.5)
    , blaze_config_(std::make_unique<Config>()) {
}

//生成锚框，调用config。
void DetectorBase::config_model(const std::string& blaze_app) {
    // Get anchor options
    anchor_options_ = blaze_config_->get_anchor_options(blaze_app, 
                                                       static_cast<int>(x_scale), 
                                                       static_cast<int>(y_scale), 
                                                       num_anchors);
    if (DEBUG) {
        std::cout << "[DetectorBase.config_model] Anchor Options: num_layers=" 
                  << anchor_options_.num_layers << std::endl;
    }
    
    // Generate anchors
    anchors_ = blaze_config_->generate_anchors(anchor_options_);
    if (DEBUG) {
        std::cout << "[DetectorBase.config_model] Anchors count: " 
                  << anchors_.size() << std::endl;
    }
    
    // Get model config
    config_ = blaze_config_->get_model_config(blaze_app, 
                                             static_cast<int>(x_scale), 
                                             static_cast<int>(y_scale), 
                                             num_anchors);
    if (DEBUG) {
        std::cout << "[DetectorBase.config_model] Model config loaded" << std::endl;
    }
    
    // Set model config parameters
    num_classes = config_.num_classes;
    num_anchors = config_.num_anchors;
    num_coords = config_.num_coords;
    score_clipping_thresh = config_.score_clipping_thresh;
    x_scale = config_.x_scale;
    y_scale = config_.y_scale;
    h_scale = config_.h_scale;
    w_scale = config_.w_scale;
    min_score_thresh = config_.min_score_thresh;
    min_suppression_threshold = config_.min_suppression_threshold;
    num_keypoints = config_.num_keypoints;
    detection2roi_method = config_.detection2roi_method;
    kp1 = config_.kp1;
    kp2 = config_.kp2;
    theta0 = config_.theta0;
    dscale = config_.dscale;
    dy = config_.dy;
}

// 图像预处理，resize和pad
std::tuple<ImageType, double, Point2DType> DetectorBase::resize_pad(const ImageType& img) {
    /**
     * Resize and pad images to be input to the detectors
     * 
     * The face and palm detector networks take 256x256 and 128x128 images
     * as input. As such the input image is padded and resized to fit the
     * size while maintaining the aspect ratio.
     * 
     * Returns:
     *     img: resized and padded image
     *     scale: scale factor between original image and target image
     *     pad: pixels of padding in the original image
     */
    
    cv::Size original_size = img.size();
    int target_h = static_cast<int>(h_scale);
    int target_w = static_cast<int>(w_scale);
    
    int h1, w1, padh, padw;
    double scale;
    
    if (original_size.height >= original_size.width) {
        h1 = target_h;
        w1 = static_cast<int>(target_w * original_size.width / original_size.height);
        padh = 0;
        padw = target_w - w1;
        scale = static_cast<double>(original_size.width) / w1;
    } else {
        h1 = static_cast<int>(target_h * original_size.height / original_size.width);
        w1 = target_w;
        padh = target_h - h1;
        padw = 0;
        scale = static_cast<double>(original_size.height) / h1;
    }
    
    int padh1 = padh / 2;
    int padh2 = padh / 2 + padh % 2;
    int padw1 = padw / 2;
    int padw2 = padw / 2 + padw % 2;
    
    // Resize image
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(w1, h1));
    
    // Pad image
    cv::Mat padded_img;
    cv::copyMakeBorder(resized_img, padded_img, 
                      padh1, padh2, padw1, padw2, 
                      cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    
    cv::Point2f pad(static_cast<float>(padw1 * scale), static_cast<float>(padh1 * scale));

    // 添加打印语句显示各个值
    std::cout << "padded图像尺寸: " << padded_img.size() << std::endl;
    std::cout << "scale值: " << scale << std::endl;
    std::cout << "pad_left(padw1): " << padw1 << ", pad_top(padh1): " << padh1 << std::endl;
    
    return std::make_tuple(padded_img, scale, pad);
}

std::vector<Detection> DetectorBase::denormalize_detections(std::vector<Detection>& detections, 
                                                                double scale, 
                                                                const Point2DType& pad) {
    /**
     * Maps detection coordinates from [0,1] to image coordinates
     * 
     * The face and palm detector networks take 256x256 and 128x128 images
     * as input. As such the input image is padded and resized to fit the
     * size while maintaining the aspect ratio. This function maps the
     * normalized coordinates back to the original image coordinates.
     */
    
    for (auto& detection : detections) {
        // Denormalize bounding box coordinates
        detection.xmin = detection.xmin * scale * x_scale - pad.x;
        detection.ymin = detection.ymin * scale * y_scale - pad.y;
        detection.xmax = detection.xmax * scale * x_scale - pad.x;
        detection.ymax = detection.ymax * scale * y_scale - pad.y;
        
        // Denormalize keypoints
        for (auto& keypoint : detection.keypoints) {
            keypoint.x = keypoint.x * scale * x_scale - pad.x;
            keypoint.y = keypoint.y * scale * y_scale - pad.y;
        }
    }
    
    return detections;
}

std::vector<ROI> DetectorBase::detection2roi(const std::vector<Detection>& detections) {
    /**
     * Convert detections from detector to an oriented bounding box.
     * 
     * Adapted from:
     * mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
     * 
     * The center and size of the box is calculated from the center 
     * of the detected box. Rotation is calculated from the vector
     * between kp1 and kp2 relative to theta0. The box is scaled
     * and shifted by dscale and dy.
     */
    
    std::vector<ROI> rois;
    
    for (const auto& detection : detections) {
        ROI roi;
        
        if (detection2roi_method == "box") {
            // Compute box center and scale
            // Use mediapipe/calculators/util/detections_to_rects_calculator.cc
            roi.xc = (detection.xmin + detection.xmax) / 2.0;
            roi.yc = (detection.ymin + detection.ymax) / 2.0;
            roi.scale = detection.xmax - detection.xmin; // assumes square boxes
            //添加
            std::cout << "  Calculated roi.scale: " << roi.scale << std::endl;

            
        } else if (detection2roi_method == "alignment") {
            // Compute box center and scale
            // Use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
            if (kp1 < static_cast<int>(detection.keypoints.size()) && 
                kp2 < static_cast<int>(detection.keypoints.size())) {
                
                roi.xc = detection.keypoints[kp1].x;
                roi.yc = detection.keypoints[kp1].y;
                
                double x1 = detection.keypoints[kp2].x;
                double y1 = detection.keypoints[kp2].y;
                
                roi.scale = std::sqrt((roi.xc - x1) * (roi.xc - x1) + 
                                    (roi.yc - y1) * (roi.yc - y1)) * 2.0;
            } else {
                // Fallback to box method if keypoints are not available
                roi.xc = (detection.xmin + detection.xmax) / 2.0;
                roi.yc = (detection.ymin + detection.ymax) / 2.0;
                roi.scale = detection.xmax - detection.xmin;
            }
            
        } else {
            throw std::runtime_error("detection2roi_method [" + detection2roi_method + "] not supported");
        }
        
        // Apply dy offset
        roi.yc += dy * roi.scale;
        
        // Apply dscale
        roi.scale *= dscale;
        
        // Compute box rotation
        if (kp1 < static_cast<int>(detection.keypoints.size()) && 
            kp2 < static_cast<int>(detection.keypoints.size())) {
            
            double x0 = detection.keypoints[kp1].x;
            double y0 = detection.keypoints[kp1].y;
            double x1 = detection.keypoints[kp2].x;
            double y1 = detection.keypoints[kp2].y;
            
            roi.theta = std::atan2(y0 - y1, x0 - x1) - theta0;


                // 添加打印语句，输出各个参数值
            std::cout << std::fixed << std::setprecision(2); // 设置输出精度
            std::cout << "关键点数量: " << detection.keypoints.size() << std::endl;
            std::cout << "kp1索引: " << kp1 << ", kp2索引: " << kp2 << std::endl;
            std::cout << "关键点" << kp1 << "坐标: (x0=" << x0 << ", y0=" << y0 << ")" << std::endl;
            std::cout << "关键点" << kp2 << "坐标: (x1=" << x1 << ", y1=" << y1 << ")" << std::endl;
            std::cout << "基准角度theta0: " << theta0 << "弧度 (" << theta0 * 180.0 / M_PI << "度)" << std::endl;
            std::cout << "向量(y0-y1, x0-x1): (" << (y0-y1) << ", " << (x0-x1) << ")" << std::endl;
    
            double atan2_result = std::atan2(y0 - y1, x0 - x1);
            std::cout << "atan2结果: " << atan2_result << "弧度 (" << atan2_result * 180.0 / M_PI << "度)" << std::endl;
            std::cout << "计算出的旋转角度theta: " << roi.theta << "弧度 (" << roi.theta * 180.0 / M_PI << "度)" << std::endl;
            std::cout << "---" << std::endl;
            std::cout.unsetf(std::ios_base::fixed); // 重置输出格式


        } else {
            roi.theta = 0.0; // Default rotation if keypoints not available
        }
        
        rois.push_back(roi);
    }
    
    return rois;
}

// ============================================================================
// Tensor Processing Functions
// ============================================================================

std::vector<std::vector<Detection>> 
DetectorBase::tensors_to_detections(const std::vector<std::vector<std::vector<double>>>& raw_box_tensor,
                                        const std::vector<std::vector<std::vector<double>>>& raw_score_tensor,
                                        const std::vector<Anchor>& anchors) {
    /**
     * The output of the neural network is an array of shape (b, 896, 12)
     * containing the bounding box regressor predictions, as well as an array 
     * of shape (b, 896, 1) with the classification confidences.
     * 
     * This function converts these two "raw" arrays into proper detections.
     * Returns a list of (num_detections, 13) arrays, one for each image in
     * the batch.
     * 
     * This is based on the source code from:
     * mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
     * mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
     */
    
    // Decode boxes from raw tensor
    auto detection_boxes = decode_boxes(raw_box_tensor, anchors);
    
    // Apply score clipping and sigmoid
    double thresh = score_clipping_thresh;
    //std::vector<std::vector<double>> detection_scores;
    detection_scores.clear();
    
    for (size_t batch_idx = 0; batch_idx < raw_score_tensor.size(); ++batch_idx) {
        std::vector<double> batch_scores;
        for (size_t anchor_idx = 0; anchor_idx < raw_score_tensor[batch_idx].size(); ++anchor_idx) {
            // Clip score
            double clipped_score = std::max(-thresh, std::min(thresh, raw_score_tensor[batch_idx][anchor_idx][0]));
            // Apply sigmoid
            double sigmoid_score = 1.0 / (1.0 + std::exp(-clipped_score));
            batch_scores.push_back(sigmoid_score);
        }
        detection_scores.push_back(batch_scores);
    }

    // Filter detections by score threshold
    std::vector<std::vector<Detection>> output_detections;
    
    // Debug: Track score statistics
    int total_detections = 0;
    int passed_threshold = 0;
    double max_score = -1.0;
    double min_score = 2.0;
    
    for (size_t batch_idx = 0; batch_idx < detection_boxes.size(); ++batch_idx) {
        std::vector<Detection> batch_detections;
        
        for (size_t anchor_idx = 0; anchor_idx < detection_boxes[batch_idx].size(); ++anchor_idx) {
            double score = detection_scores[batch_idx][anchor_idx];
            total_detections++;
            
            // Track score range
            if (score > max_score) max_score = score;
            if (score < min_score) min_score = score;
            
            if (score >= min_score_thresh) {
                passed_threshold++;
                Detection detection;
                const auto& box = detection_boxes[batch_idx][anchor_idx];
                
                // Set bounding box coordinates
                detection.ymin = box.ymin;
                detection.xmin = box.xmin;
                detection.ymax = box.ymax;
                detection.xmax = box.xmax;
                
                // Set keypoints
                detection.keypoints = box.keypoints;
                
                // Set score
                detection.score = score;
                
                batch_detections.push_back(detection);


                std::cout << std::fixed << std::setprecision(4);
                std::cout << "[Detection] 置信度: " << score << std::endl;
                std::cout << "[Detection] 边界框: (" << detection.xmin << ", " << detection.ymin 
                              << ") - (" << detection.xmax << ", " << detection.ymax << ")" << std::endl;
                std::cout << "[Detection] 7个关键点坐标: " << std::endl;
                for (int k = 0; k < num_keypoints && k < static_cast<int>(detection.keypoints.size()); ++k) {
                    const auto& kp = detection.keypoints[k];
                    std::cout << "  关键点[" << k << "]: (" << kp.x << ", " << kp.y << ")" << std::endl;
                }
                std::cout << "----------------------------------------" << std::endl;
                std::cout.unsetf(std::ios_base::fixed);
            
            }
        }
        
        output_detections.push_back(batch_detections);
    }
    
    if (DEBUG) {
        // Debug: Print score statistics
        std::cout << "[DetectorBase.tensors_to_detections] Score statistics:" << std::endl;
        std::cout << "  Total anchors: " << total_detections << std::endl;
        std::cout << "  Passed threshold (" << min_score_thresh << "): " << passed_threshold << std::endl;
        std::cout << "  Score range: " << min_score << " to " << max_score << std::endl;
    }
    return output_detections;
}

std::vector<std::vector<Detection>> 
DetectorBase::decode_boxes(const std::vector<std::vector<std::vector<double>>>& raw_boxes, 
                               const std::vector<Anchor>& anchors) {
    /**
     * Converts the predictions into actual coordinates using
     * the anchor boxes. Processes the entire batch at once.
     */
    
    std::vector<std::vector<Detection>> decoded_boxes;
    
    for (size_t batch_idx = 0; batch_idx < raw_boxes.size(); ++batch_idx) {
        std::vector<Detection> batch_detections;
        
        for (size_t anchor_idx = 0; anchor_idx < raw_boxes[batch_idx].size() && anchor_idx < anchors.size(); ++anchor_idx) {
            const auto& raw_box = raw_boxes[batch_idx][anchor_idx];
            const auto& anchor = anchors[anchor_idx];
            
            Detection detection;
            
            // Decode center coordinates
            double x_center = raw_box[0] / x_scale * anchor.width + anchor.x_center;
            double y_center = raw_box[1] / y_scale * anchor.height + anchor.y_center;
            
            // Decode width and height
            double w = raw_box[2] / w_scale * anchor.width;
            double h = raw_box[3] / h_scale * anchor.height;
            
            // Convert to bounding box coordinates
            detection.ymin = y_center - h / 2.0;
            detection.xmin = x_center - w / 2.0;
            detection.ymax = y_center + h / 2.0;
            detection.xmax = x_center + w / 2.0;
            
            // Decode keypoints
            for (int k = 0; k < num_keypoints; ++k) {
                int offset = 4 + k * 2;
                if (offset + 1 < static_cast<int>(raw_box.size())) {
                    double keypoint_x = raw_box[offset] / x_scale * anchor.width + anchor.x_center;
                    double keypoint_y = raw_box[offset + 1] / y_scale * anchor.height + anchor.y_center;
                    detection.keypoints.push_back(cv::Point2f(static_cast<float>(keypoint_x), static_cast<float>(keypoint_y)));
                }
            }
            
            batch_detections.push_back(detection);
        }
        
        decoded_boxes.push_back(batch_detections);
    }
    
    return decoded_boxes;
}

// ============================================================================
// Utility Functions for IOU Calculation
// ============================================================================

double intersect_area(const RectType& box_a, const RectType& box_b) {
    double max_x = std::min(box_a.x + box_a.width, box_b.x + box_b.width);
    double min_x = std::max(box_a.x, box_b.x);
    double max_y = std::min(box_a.y + box_a.height, box_b.y + box_b.height);
    double min_y = std::max(box_a.y, box_b.y);
    
    double width = std::max(0.0, max_x - min_x);
    double height = std::max(0.0, max_y - min_y);
    
    return width * height;
}

double jaccard_overlap(const RectType& box_a, const RectType& box_b) {
    /**
     * Compute the jaccard overlap of two boxes. The jaccard overlap
     * is simply the intersection over union of two boxes.
     * A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
     */
    double inter = intersect_area(box_a, box_b);
    double area_a = box_a.width * box_a.height;
    double area_b = box_b.width * box_b.height;
    double union_area = area_a + area_b - inter;
    
    if (union_area <= 0.0) {
        return 0.0;
    }
    
    return inter / union_area;
}

double overlap_similarity(const RectType& box, const std::vector<RectType>& other_boxes, int index) {
    /**
     * Computes the IOU between a bounding box and a specific box from a set of other boxes.
     */
    if (index < 0 || index >= static_cast<int>(other_boxes.size())) {
        return 0.0;
    }
    
    return jaccard_overlap(box, other_boxes[index]);
}

std::vector<Detection> DetectorBase::weighted_non_max_suppression(const std::vector<Detection>& detections) {
    /**
     * The alternative NMS method as mentioned in the Face paper:
     * 
     * "We replace the suppression algorithm with a blending strategy that
     * estimates the regression parameters of a bounding box as a weighted
     * mean between the overlapping predictions."
     * 
     * The original MediaPipe code assigns the score of the most confident
     * detection to the weighted detection, but we take the average score
     * of the overlapping detections.
     * 
     * The input detections should be a vector of Detection objects.
     * 
     * Returns a vector of Detection objects, one for each detected object.
     * 
     * This is based on the source code from:
     * mediapipe/calculators/util/non_max_suppression_calculator.cc
     * mediapipe/calculators/util/non_max_suppression_calculator.proto
     */
    
    if (detections.empty()) {
        return {};
    }
    
    std::vector<Detection> output_detections;
    
    // Create a copy of detections with indices for sorting
    std::vector<std::pair<Detection, int>> indexed_detections;
    for (size_t i = 0; i < detections.size(); ++i) {
        indexed_detections.push_back({detections[i], static_cast<int>(i)});
    }
    
    // Sort detections by score (highest first)
    std::sort(indexed_detections.begin(), indexed_detections.end(), 
              [](const std::pair<Detection, int>& a, const std::pair<Detection, int>& b) {
                  return a.first.score > b.first.score;
              });
    
    // Track which detections have been processed
    std::vector<bool> processed(detections.size(), false);
    
    for (size_t i = 0; i < indexed_detections.size(); ++i) {
        if (processed[indexed_detections[i].second]) {
            continue;
        }
        
        Detection current_detection = indexed_detections[i].first;
        std::vector<int> overlapping_indices;
        overlapping_indices.push_back(indexed_detections[i].second);
        
        // Find all overlapping detections
        RectType current_rect(current_detection.xmin, current_detection.ymin,
                             current_detection.xmax - current_detection.xmin,
                             current_detection.ymax - current_detection.ymin);
        
        for (size_t j = i + 1; j < indexed_detections.size(); ++j) {
            if (processed[indexed_detections[j].second]) {
                continue;
            }
            
            const Detection& other_detection = indexed_detections[j].first;
            RectType other_rect(other_detection.xmin, other_detection.ymin,
                               other_detection.xmax - other_detection.xmin,
                               other_detection.ymax - other_detection.ymin);
            
            double iou = jaccard_overlap(current_rect, other_rect);
            
            if (iou > min_suppression_threshold) {
                overlapping_indices.push_back(indexed_detections[j].second);
            }
        }
        
        // Mark overlapping detections as processed
        for (int idx : overlapping_indices) {
            processed[idx] = true;
        }
        
        // Create weighted detection
        Detection weighted_detection = current_detection;
        
        if (overlapping_indices.size() > 1) {
            // Calculate weighted average of coordinates
            double total_score = 0.0;
            double weighted_ymin = 0.0, weighted_xmin = 0.0;
            double weighted_ymax = 0.0, weighted_xmax = 0.0;
            
            // Initialize keypoints accumulator
            std::vector<Point2DType> weighted_keypoints(current_detection.keypoints.size());
            for (auto& kp : weighted_keypoints) {
                kp.x = 0.0f;
                kp.y = 0.0f;
            }
            
            for (int idx : overlapping_indices) {
                const Detection& det = detections[idx];
                double score = det.score;
                total_score += score;
                
                weighted_ymin += det.ymin * score;
                weighted_xmin += det.xmin * score;
                weighted_ymax += det.ymax * score;
                weighted_xmax += det.xmax * score;
                
                // Weight keypoints
                for (size_t k = 0; k < det.keypoints.size() && k < weighted_keypoints.size(); ++k) {
                    weighted_keypoints[k].x += det.keypoints[k].x * score;
                    weighted_keypoints[k].y += det.keypoints[k].y * score;
                }
            }
            
            // Normalize by total score
            if (total_score > 0.0) {
                weighted_detection.ymin = weighted_ymin / total_score;
                weighted_detection.xmin = weighted_xmin / total_score;
                weighted_detection.ymax = weighted_ymax / total_score;
                weighted_detection.xmax = weighted_xmax / total_score;
                
                for (auto& kp : weighted_keypoints) {
                    kp.x /= total_score;
                    kp.y /= total_score;
                }
                weighted_detection.keypoints = weighted_keypoints;
                
                // Use average score
                weighted_detection.score = total_score / overlapping_indices.size();
            }
        }
        

        


        output_detections.push_back(weighted_detection);
    }
    
    return output_detections;
}

} // namespace blaze
