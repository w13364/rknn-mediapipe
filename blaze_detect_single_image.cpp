/*
 * Copyright 2025 Tria Technologies Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #include <cstring>
 #include <cstdio>
 #include <cstdlib>
 
 #include <iomanip>
 #include <iostream>
 #include <fstream>
 #include <sstream>
 #include <numeric>
 #include <filesystem>
 #include <algorithm>
 #include <cmath>
 
 #include <opencv2/highgui.hpp>
 #include <opencv2/imgproc.hpp>
 #include <opencv2/opencv.hpp>
 
 #include "Base.hpp"
 #include "Detector.hpp"
 #include "Landmark.hpp"
 #include "visualization.hpp"

// 帮助信息
void print_help() {
    std::cout << "用法: blaze_detect_single_image [选项] <图像路径>" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  -h, --help           显示此帮助信息并退出" << std::endl;
    std::cout << "  -b, --blaze <类型>   应用类型 (hand, face, pose). 默认是 hand" << std::endl;
    std::cout << "  -m, --model1 <路径>  检测模型路径. 默认是 models/hand_detector.rknn" << std::endl;
    std::cout << "  -n, --model2 <路径>  关键点模型路径. 默认是 models/hand_landmarks_detector.rknn" << std::endl;
    std::cout << "  -o, --output <路径>  输出图像保存路径. 默认是显示图像" << std::endl;
    std::cout << "  -v, --verbose        启用详细模式" << std::endl;
}

// 命令行参数结构体
struct Args {
    std::string image_path;
    std::string blaze = "hand";
    std::string model1 = "models/hand_detector.rknn";
    std::string model2 = "models/hand_landmarks_detector.rknn";
    std::string output_path = "";
    bool verbose = false;
};

// 解析命令行参数
Args parse_args(int argc, char* argv[]) {
    Args args;
    
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            print_help();
            exit(0);
        }
        else if (a == "-b" || a == "--blaze") {
            if (i + 1 < argc) {
                args.blaze = argv[++i];
            } else {
                std::cerr << "错误: --blaze 参数需要一个值" << std::endl;
                print_help();
                exit(1);
            }
        }
        else if (a == "-m" || a == "--model1") {
            if (i + 1 < argc) {
                args.model1 = argv[++i];
            } else {
                std::cerr << "错误: --model1 参数需要一个值" << std::endl;
                print_help();
                exit(1);
            }
        }
        else if (a == "-n" || a == "--model2") {
            if (i + 1 < argc) {
                args.model2 = argv[++i];
            } else {
                std::cerr << "错误: --model2 参数需要一个值" << std::endl;
                print_help();
                exit(1);
            }
        }
        else if (a == "-o" || a == "--output") {
            if (i + 1 < argc) {
                args.output_path = argv[++i];
            } else {
                std::cerr << "错误: --output 参数需要一个值" << std::endl;
                print_help();
                exit(1);
            }
        }
        else if (a == "-v" || a == "--verbose") {
            args.verbose = true;
        }
        else if (args.image_path.empty()) {
            args.image_path = a;
        }
        else {
            std::cerr << "错误: 未知参数 \"" << a << "\"" << std::endl;
            print_help();
            exit(1);
        }
    }
    
    if (args.image_path.empty()) {
        std::cerr << "错误: 必须提供图像路径" << std::endl;
        print_help();
        exit(1);
    }
    
    return args;
}

int main(int argc, char* argv[]) {
    // 解析命令行参数
    Args args = parse_args(argc, argv);
    
    // 根据应用类型设置参数
    std::string blaze_detector_type, blaze_landmark_type, blaze_title;
    std::string default_detector_model, default_landmark_model;
    
    if (args.blaze == "hand") {
        blaze_detector_type = "blazepalm";
        blaze_landmark_type = "blazehandlandmark";
        blaze_title = "BlazeHandLandmark";
        default_detector_model = "models/hand_detector.rknn";
        default_landmark_model = "models/hand_landmarks_detector.rknn";
    } else if (args.blaze == "face") {
        blaze_detector_type = "blazeface";
        blaze_landmark_type = "blazefacelandmark";
        blaze_title = "BlazeFaceLandmark";
        // 注意：这里假设已经有对应的face rknn模型
        default_detector_model = "models/face_detection_short_range.rknn";
        default_landmark_model = "models/face_landmark.rknn";
    } else if (args.blaze == "pose") {
        blaze_detector_type = "blazepose";
        blaze_landmark_type = "blazeposelandmark";
        blaze_title = "BlazePoseLandmark";
        // 注意：这里假设已经有对应的pose rknn模型
        default_detector_model = "models/pose_detection.rknn";
        default_landmark_model = "models/pose_landmark_full.rknn";
    } else {
        std::cerr << "错误: 无效的 Blaze 应用类型: " << args.blaze << ". 必须是 hand, face 或 pose 之一." << std::endl;
        return 1;
    }
    
    // 使用命令行参数或默认模型路径
    if (args.model1.empty()) args.model1 = default_detector_model;
    if (args.model2.empty()) args.model2 = default_landmark_model;
    
    // 输出当前配置
    std::cout << "[INFO] 当前配置:" << std::endl;
    std::cout << "  应用类型: " << args.blaze << std::endl;
    std::cout << "  检测模型: " << args.model1 << std::endl;
    std::cout << "  关键点模型: " << args.model2 << std::endl;
    std::cout << "  输入图像: " << args.image_path << std::endl;
    if (!args.output_path.empty()) {
        std::cout << "  输出图像: " << args.output_path << std::endl;
    }
    
    // 创建检测器和关键点模型
    std::unique_ptr<blaze::Detector> blaze_detector = std::make_unique<blaze::Detector>(blaze_detector_type);
    std::unique_ptr<blaze::Landmark> blaze_landmark = std::make_unique<blaze::Landmark>(blaze_landmark_type);
    
    // 加载模型
    std::cout << "[INFO] 正在加载检测模型..." << std::endl;
    blaze_detector->set_debug(args.verbose);
    // 修改：移除 if 条件判断，因为 load_model 是 void 返回类型
    blaze_detector->load_model(args.model1);
    blaze_detector->set_min_score_threshold(.75);
    
    std::cout << "[INFO] 正在加载关键点模型..." << std::endl;
    blaze_landmark->set_debug(args.verbose);
    if (!blaze_landmark->load_model(args.model2)) {
        std::cerr << "错误: 无法加载关键点模型: " << args.model2 << std::endl;
        return 1;
    }
    
    // 加载图像
    std::cout << "[INFO] 正在加载图像..." << std::endl;
    cv::Mat frame = cv::imread(args.image_path);
    if (frame.empty()) {
        std::cerr << "错误: 无法加载输入图像: " << args.image_path << std::endl;
        return 1;
    }
    
    std::cout << "[INFO] 图像尺寸: " << frame.cols << "x" << frame.rows << std::endl;
    
    // 检测开始时间
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 准备输出图像
    cv::Mat output = frame.clone();
    cv::Mat rgb_frame;
    
    // 转换BGR到RGB用于处理
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    
    // 步骤1: 调整图像大小并填充
    auto [resized_img, scale, pad] = blaze_detector->resize_pad(rgb_frame);
    
    // 步骤2: 进行检测
    auto batch_results = blaze_detector->predict_on_batch(std::vector<cv::Mat>{resized_img});
    std::vector<blaze::Detection> normalized_detections = batch_results.empty() ? std::vector<blaze::Detection>() : batch_results[0];
    
    std::cout << "[INFO] 检测到的对象数量: " << normalized_detections.size() << std::endl;
    
    if (!normalized_detections.empty()) {
        // 步骤3: 反归一化检测结果
        std::vector<blaze::Detection> detections = blaze_detector->denormalize_detections(normalized_detections, scale, pad);
        
        // 步骤4: 将检测结果转换为ROI
        std::vector<blaze::ROI> rois = blaze_detector->detection2roi(detections);
        
        std::vector<double> xc, yc, theta, scale_roi;
        for (const auto& roi : rois) {
            xc.push_back(roi.xc);
            yc.push_back(roi.yc);
            theta.push_back(roi.theta);
            scale_roi.push_back(roi.scale);
        }
        
        // 步骤5: 提取ROI图像
        std::vector<cv::Mat> roi_imgs;
        std::vector<cv::Mat> roi_affine;
        std::vector<std::vector<cv::Point2f>> roi_boxes;
        std::tie(roi_imgs, roi_affine, roi_boxes) = blaze_landmark->extract_roi(rgb_frame, xc, yc, theta, scale_roi);
        
        // 步骤6: 关键点检测
        std::vector<std::vector<double>> flags;
        std::vector<std::vector<std::vector<double>>> normalized_landmarks_vec;
        std::vector<std::vector<double>> handedness;
        std::tie(flags, normalized_landmarks_vec, handedness) = blaze_landmark->predict(roi_imgs);
        
        // 处理左右手识别结果 (如果有)
        std::vector<std::string> handedness_results;
        if (!handedness.empty()) {
            for (size_t i = 0; i < handedness.size(); ++i) {
                std::string handedness_result = (handedness[i][0] > 0.5) ? "left" : "right";
                handedness_results.push_back(handedness_result);
                if (args.verbose) {
                    std::cout << "[INFO] 检测对象 " << i+1 << ": 手型 = " << handedness_result << ", 置信度 = " << handedness[i][0] << std::endl;
                }
            }
        }
        
        // 步骤7: 转换为3D点
        std::vector<std::vector<cv::Point3d>> landmarks_3d;
        for (const auto& batch : normalized_landmarks_vec) {
            std::vector<cv::Point3d> landmark_batch;
            for (const auto& landmark : batch) {
                landmark_batch.emplace_back(landmark.size() > 0 ? landmark[0] : 0, 
                                          landmark.size() > 1 ? landmark[1] : 0, 
                                          landmark.size() > 2 ? landmark[2] : 0);
            }
            landmarks_3d.push_back(landmark_batch);
        }
        
        // 步骤8: 反归一化关键点
        std::vector<std::vector<cv::Point3d>> landmarks = blaze_landmark->denormalize_landmarks(landmarks_3d, roi_affine);
        
        // 步骤9: 处理关键点并绘制结果
        double thresh_confidence = 0.5;
        for (size_t i = 0; i < flags.size() && i < landmarks.size(); ++i) {
            double confidence = flags[i].empty() ? 0.0 : flags[i][0];
            if (confidence > thresh_confidence) {
                std::vector<cv::Point2f> landmark_points;
                std::vector<cv::Point3f> landmark_points_3d;
                
                for (const auto& landmark : landmarks[i]) {
                    if (landmark.x >= 0 && landmark.x < rgb_frame.cols && landmark.y >= 0 && landmark.y < rgb_frame.rows) {
                        landmark_points.emplace_back(landmark.x, landmark.y);
                        landmark_points_3d.emplace_back(static_cast<float>(landmark.x), 
                                                     static_cast<float>(landmark.y), 
                                                     static_cast<float>(landmark.z));
                    }
                }
                
                // 绘制关键点
                if (blaze_landmark_type == "blazehandlandmark") {
                    if (handedness_results.empty()) {
                        blaze::draw_landmarks(output, landmark_points, blaze::HAND_CONNECTIONS, cv::Scalar(0,255,0), 4, 2);
                    }
                    else if (handedness_results[i] == "left") {
                        blaze::draw_landmarks(output, landmark_points, blaze::HAND_CONNECTIONS, cv::Scalar(0,255,0), 4, 2);
                    }
                    else {
                        blaze::draw_landmarks(output, landmark_points, blaze::HAND_CONNECTIONS, cv::Scalar(190, 161, 0), 4, 2);
                    }
                } else if (blaze_landmark_type == "blazefacelandmark") {
                    blaze::draw_landmarks(output, landmark_points, blaze::FACE_CONNECTIONS, cv::Scalar(0,255,0), 1, 2);
                } else if (blaze_landmark_type == "blazeposelandmark") {
                    if (landmark_points.size() > 33) {
                        blaze::draw_landmarks(output, landmark_points, blaze::POSE_FULL_BODY_CONNECTIONS, cv::Scalar(0,255,0), 4, 2);
                    } else {
                        blaze::draw_landmarks(output, landmark_points, blaze::POSE_UPPER_BODY_CONNECTIONS, cv::Scalar(0,255,0), 4, 2);
                    }
                }
            }
        }
        
        // 绘制检测框和ROI
        blaze::draw_detections(output, detections);
        blaze::draw_roi(output, roi_boxes);
    }
    
    // 计算检测时间
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "[INFO] 检测完成，耗时: " << std::fixed << std::setprecision(3) << elapsed_time * 1000 << " ms" << std::endl;
    
    // 保存或显示结果
    if (!args.output_path.empty()) {
        if (cv::imwrite(args.output_path, output)) {
            std::cout << "[INFO] 检测结果已保存到: " << args.output_path << std::endl;
        } else {
            std::cerr << "错误: 无法保存输出图像: " << args.output_path << std::endl;
        }
    } else {
        // 显示结果
        std::string window_title = blaze_title + " - 检测结果";
        cv::namedWindow(window_title, cv::WINDOW_NORMAL);
        cv::imshow(window_title, output);
        std::cout << "[INFO] 请按任意键退出..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    
    return 0;
}