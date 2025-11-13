#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "Base.hpp"  // For Detection struct

namespace blaze {

// Tria color palette
extern cv::Scalar tria_blue;
extern cv::Scalar tria_pink;
extern cv::Scalar tria_white;
extern cv::Scalar tria_gray11;
extern cv::Scalar tria_gray7;
extern cv::Scalar tria_gray3;
extern cv::Scalar tria_purple;
extern cv::Scalar tria_yellow;
extern cv::Scalar tria_aqua;
extern cv::Scalar tria_black;

// Hand landmark connections (21 points)
extern const std::vector<std::pair<int, int>> HAND_CONNECTIONS;

// Face landmark connections (simplified)
extern const std::vector<std::pair<int, int>> FACE_CONNECTIONS;

// Pose landmark connections
extern const std::vector<std::pair<int, int>> POSE_FULL_BODY_CONNECTIONS;
extern const std::vector<std::pair<int, int>> POSE_UPPER_BODY_CONNECTIONS;

// Visualization functions
void draw_detections(cv::Mat& image, const std::vector<Detection>& detections, 
                    bool with_keypoints = true);

void draw_landmarks(cv::Mat& image, const std::vector<cv::Point2f>& landmarks,
                   const std::vector<std::pair<int, int>>& connections,
                   const cv::Scalar& color = cv::Scalar(0, 255, 0), int radius = 3, int thickness = 2);

void draw_roi(cv::Mat& image, const std::vector<std::vector<cv::Point2f>>& roi_boxes);


cv::Mat draw_detection_scores( std::vector<std::vector<double>> detection_scores, double min_score_thresh );    
    
extern const std::vector<cv::Scalar> stacked_bar_generic_colors;
extern const std::vector<cv::Scalar> stacked_bar_latency_colors;
extern const std::vector<cv::Scalar> stacked_bar_performance_colors;

cv::Mat draw_stacked_bar_chart(
    const std::vector<std::string>& prof_title,
    const std::vector<std::string> latency_labels,
    const std::vector<std::vector<double>>& all_latencies, // [component][pipeline]
    const std::vector<cv::Scalar> component_colors = stacked_bar_generic_colors,
    const std::string& chart_name = "Stacked Bar Chart");
    
} // namespace blaze
