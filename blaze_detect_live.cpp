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

 #include <iomanip>
 #include <iostream>
 #include <fstream>
 #include <sstream>
 #include <numeric>
 #include <filesystem>
 #include <algorithm>
 #include <cmath>
 #include <iomanip>
 #include <unistd.h>  // 添加这个头文件以使用 gethostname 函数
 
 
 #include <opencv2/highgui.hpp>
 #include <opencv2/imgproc.hpp>
 #include <opencv2/opencv.hpp>
 
 #include <signal.h>
 #include <thread>
 #include <chrono>
 
 #include "Base.hpp"
 #include "Detector.hpp"
 #include "Landmark.hpp"
 #include "visualization.hpp"
 #include "utils_linux.hpp"
 
 //using namespace blaze; // explicitly use blaze::* instead of using namespace
 
 //  Blaze Detector/Landmark
 std::unique_ptr<blaze::Detector> blaze_detector;
 std::unique_ptr<blaze::Landmark> blaze_landmark;
 
 // Helpers for argument parsing
 struct Args {
     std::string input;
     bool testimage = false;
     std::string blaze = "hand";
     std::string model1 = "./models/hand_detector.rknn";
     std::string model2 = "./models/hand_landmarks_detector.rknn";
     bool verbose = false;
     bool withoutview = false;
     bool profilelog = false;
     bool profileview = false;
     bool fps = false;
 };
 
 Args parse_args(int argc, char* argv[]) {
     Args args;
     for (int i = 1; i < argc; ++i) {
         std::string a = argv[i];
         if (a == "-i" || a == "--input") { args.input = argv[++i]; }
         else if (a == "-I" || a == "--testimage") { args.testimage = true; }
         else if (a == "-b" || a == "--blaze") { args.blaze = argv[++i]; }
         else if (a == "-m" || a == "--model1") { args.model1 = argv[++i]; }
         else if (a == "-n" || a == "--model2") { args.model2 = argv[++i]; }
         else if (a == "-v" || a == "--verbose") { args.verbose = true; }
         else if (a == "-w" || a == "--withoutview") { args.withoutview = true; }
         else if (a == "-z" || a == "--profilelog") { args.profilelog = true; }
         else if (a == "-y" || a == "--profileview") { args.profileview = true; }
         else if (a == "-f" || a == "--fps") { args.fps = false; }
         else if (a == "-h" || a == "--help") {
             std::cout << "                                                                                                          " << std::endl;
             std::cout << "usage: blaze_detect_live [-h] [-i INPUT] [-t] [-b BLAZE] [-m MODEL1] [-n MODEL2] [-d] [-w] [-z] [-y] [-f] " << std::endl;
             std::cout << "                                                                                                          " << std::endl;
             std::cout << "options:                                                                                                  " << std::endl;
             std::cout << "  -h,        --help           Show this help message and exit                                             " << std::endl;
             std::cout << "  -i INPUT,  --input INPUT    Video input device. Default is auto-detect (first usbcam)                   " << std::endl;
             std::cout << "  -I,        --testimage      Use test image as input (womand_hands.jpg). Default is usbcam               " << std::endl;
             std::cout << "  -b BLAZE,  --blaze BLAZE    Application (hand, face, pose). Default is hand                             " << std::endl;           
             std::cout << "  -m MODEL1, --model1 MODEL1  Path of blazepalm model. Default is models/palm_detection_lite.tflite       " << std::endl;
             std::cout << "  -n MODEL2, --model2 MODEL2  Path of blazehandlardmark model. Default is models/hand_landmark_lite.tflite" << std::endl;
             std::cout << "  -v,        --verbose        Enable Verbose mode. Default is off                                         " << std::endl;
             std::cout << "  -w,        --withoutview    Disable Output viewing. Default is on                                       " << std::endl;
             std::cout << "  -z,        --profilelog     Enable Profile Log (Latency). Default is off                                " << std::endl;
             std::cout << "  -y,        --profileview    Enable Profile View (Latency). Default is off                               " << std::endl;
             std::cout << "  -f,        --fps            Enable FPS display. Default is off                                          " << std::endl;
             std::cout << "                                                                                                          " << std::endl;
             exit(0);
         }
     }
     return args;
 }
 
 // track bar callback
 void on_trackbar(int, void*) {
     // DO NOTHING
 }
 
 volatile bool running = true;
 
 // Signal handler (to abort gracefully)
 void signal_handler(int) {
     std::cout << "\nShutting down..." << std::endl;
     running = false;
 }
 
 int main(int argc, char* argv[]) {
     signal(SIGINT, signal_handler);
     signal(SIGTERM, signal_handler);
 
     Args args = parse_args(argc, argv);
 
     std::string user = std::getenv("USER") ? std::getenv("USER") : "user";
     char host[256]; gethostname(host, 256);
     std::string user_host_descriptor = user + "@" + host;
     std::cout << "[INFO] user@hosthame : " << user_host_descriptor << std::endl;
 
     std::cout << "Command line options:" << std::endl;
     std::cout << " --input       : " << args.input << std::endl;
     std::cout << " --testimage   : " << args.testimage << std::endl;
     std::cout << " --blaze       : " << args.blaze << std::endl;
     std::cout << " --model1      : " << args.model1 << std::endl;
     std::cout << " --model2      : " << args.model2 << std::endl;
     std::cout << " --verbose     : " << args.verbose << std::endl;
     std::cout << " --withoutview : " << args.withoutview << std::endl;
     std::cout << " --profilelog  : " << args.profilelog << std::endl;
     std::cout << " --profileview : " << args.profileview << std::endl;
     std::cout << " --fps         : " << args.fps << std::endl;
 
 
     std::cout << "[INFO] Searching for USB camera ..." << std::endl;
 
     // std::string dev_video = get_video_dev_by_name("uvcvideo");
     // std::string dev_media = get_media_dev_by_name("uvcvideo");
     // std::cout << dev_video << std::endl;
     // std::cout << dev_media << std::endl;
 
     // std::string input_video;
     // if (dev_video.empty()) {
     //     input_video = "0";
     // } else if (!args.input.empty()) {
     //     input_video = args.input;
     // } else {
     //     input_video = dev_video;
     // }
     // std::cout << "[INFO] Input Video : " << input_video << std::endl;
     // 修改这部分代码，使用整数而不是字符串来打开摄像头
 
 
     // // Open video
     // cv::VideoCapture cap(input_video);
     // int frame_width = 640;
     // int frame_height = 480;
     // cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
     // cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);
     // std::cout << "camera " << input_video << " (" << frame_width << "," << frame_height << ")" << std::endl;
 
     // std::string output_dir = "./captured-images";
     // if (!std::filesystem::exists(output_dir)) std::filesystem::create_directory(output_dir);
 
     // // Pipelines
     // int nb_blaze_pipelines = 1;
     // 禁用自动检测，直接使用整数摄像头ID
     int camera_id = 0;
     std::string input_video = "0";
     
     // 如果用户通过命令行参数指定了摄像头，使用用户指定的值
     if (!args.input.empty()) {
         input_video = args.input;
         try {
             camera_id = std::stoi(args.input);
         } catch (...) {
             // 如果无法转换为整数，则保持使用字符串路径
             std::cout << "[INFO] Using camera path: " << args.input << std::endl;
         }
     }
     std::cout << "[INFO] Input Video : " << input_video << std::endl;
 
     // 打开摄像头
     cv::VideoCapture cap;
     try {
         // 首先尝试使用整数设备ID打开
         cap.open(camera_id);
     } catch (...) {
         // 如果失败，尝试使用字符串路径打开
         cap.open(input_video);
     }
     
     // 设置摄像头分辨率
     int frame_width = 640;
     int frame_height = 480;
     cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
     cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);
     std::cout << "camera " << input_video << " (" << frame_width << "," << frame_height << ")" << std::endl;
 
     // 检查摄像头是否成功打开
     if (!cap.isOpened()) {
         std::cerr << "[ERROR] Could not open camera!" << std::endl;
         return 1;
     }
 
     std::string output_dir = "./captured-images";
     if (!std::filesystem::exists(output_dir)) std::filesystem::create_directory(output_dir);
 
     // Pipelines
     int nb_blaze_pipelines = 1;   
 
 
     int pipeline_id = 0;
 
     std::string blaze_detector_type, blaze_landmark_type, blaze_title;
     std::string default_detector_model, default_landmark_model;
     if (args.blaze == "hand") {
         blaze_detector_type = "blazepalm";
         blaze_landmark_type = "blazehandlandmark";
         blaze_title = "BlazeHandLandmark";
         default_detector_model = "models/palm_detection_lite.tflite";
         default_landmark_model = "models/hand_landmark_lite.tflite";
     } else if (args.blaze == "face") {
         blaze_detector_type = "blazeface";
         blaze_landmark_type = "blazefacelandmark";
         blaze_title = "BlazeFaceLandmark";
         default_detector_model = "models/face_detection_short_range.tflite";
         default_landmark_model = "models/face_landmark.tflite";
     } else if (args.blaze == "pose") {
         blaze_detector_type = "blazepose";
         blaze_landmark_type = "blazeposelandmark";
         blaze_title = "BlazePoseLandmark";
         default_detector_model = "models/pose_detection.tflite";
         default_landmark_model = "models/pose_landmark_full.tflite";
     } else {
         std::cout << "[ERROR] Invalid Blaze application : " << args.blaze << ".  MUST be one of hand,face,pose." << std::endl;
         return 1;
     }
     if (args.model1.empty()) args.model1 = default_detector_model;
     if (args.model2.empty()) args.model2 = default_landmark_model;
           
     blaze_detector = std::make_unique<blaze::Detector>(blaze_detector_type);
     blaze_landmark = std::make_unique<blaze::Landmark>(blaze_landmark_type);
     
     // Load palm detection model
     blaze_detector->set_debug(args.verbose); 
     blaze_detector->load_model(args.model1);
     blaze_detector->set_min_score_threshold(.75);
         
     // Load hand landmark model
     blaze_landmark->set_debug(args.verbose);
     blaze_landmark->load_model(args.model2);
 
     // Profiling
     std::vector<std::string> prof_title(nb_blaze_pipelines, blaze_title);
     std::vector<int> prof_detector_qty(nb_blaze_pipelines, 0.0);
     std::vector<double> prof_resize(nb_blaze_pipelines, 0.0);
     std::vector<double> prof_detector_pre(nb_blaze_pipelines, 0.0);
     std::vector<double> prof_detector_model(nb_blaze_pipelines, 0.0);
     std::vector<double> prof_detector_post(nb_blaze_pipelines, 0.0);
     std::vector<double> prof_extract_roi(nb_blaze_pipelines, 0.0);
     std::vector<double> prof_landmark_pre(nb_blaze_pipelines, 0.0);
     std::vector<double> prof_landmark_model(nb_blaze_pipelines, 0.0);
     std::vector<double> prof_landmark_post(nb_blaze_pipelines, 0.0);
     std::vector<double> prof_annotate(nb_blaze_pipelines, 0.0);
     //
     std::vector<double> prof_total(nb_blaze_pipelines, 0.0);
     std::vector<double> prof_fps(nb_blaze_pipelines, 0.0);
     
     const std::vector<std::string> latency_labels = {
         "resize",
         "detector[pre]",
         "detector[model]",
         "detector[post]",
         "extract_roi",
         "landmark[pre]",
         "landmark[model]",
         "landmark[post]",
         "annotate"
     };
     const std::vector<std::string> performance_labels = {
         "fps"
     };
     std::string profile_csv = "./blaze_detect_live.csv";
     std::ofstream f_profile_csv;
     bool append = std::filesystem::exists(profile_csv);
     f_profile_csv.open(profile_csv, append ? std::ios::app : std::ios::out);
     if (append) {
         std::cout << "[INFO] Appending to existing profiling results file :" << profile_csv << std::endl;
     } else {
         std::cout << "[INFO] Creating new profiling results file :" << profile_csv << std::endl;
         f_profile_csv << "time,user,hostname,pipeline,detection_qty,resize,detector_pre,detector_model,detector_post,extract_roi,landmark_pre,landmark_model,landmark_post,annotate,total,fps\n";
     }
 
     std::cout << "================================================================" << std::endl;
     std::cout << "Blaze Detect Live Demo" << std::endl;
     std::cout << "================================================================" << std::endl;
     std::cout << "\tPress ESC to quit ..." << std::endl;
     std::cout << "----------------------------------------------------------------" << std::endl;
     std::cout << "\tPress 'p' to pause video ..." << std::endl;
     std::cout << "\tPress 'c' to continue ..." << std::endl;
     std::cout << "\tPress 's' to step one frame at a time ..." << std::endl;
     std::cout << "\tPress 'w' to take a photo ..." << std::endl;
     std::cout << "----------------------------------------------------------------" << std::endl;
     std::cout << "\tPress 't' to toggle between image and live video" << std::endl;
     std::cout << "\tPress 'h' to toggle horizontal mirror on input" << std::endl;
     std::cout << "\tPress 'a' to toggle detection overlay on/off" << std::endl;
     std::cout << "\tPress 'b' to toggle roi overlay on/off" << std::endl;
     std::cout << "\tPress 'l' to toggle landmarks overlay on/off" << std::endl;
     std::cout << "\tPress 'd' to toggle debug image on/off" << std::endl;
     std::cout << "\tPress 'e' to toggle scores image on/off" << std::endl;
     std::cout << "\tPress 'f' to toggle FPS display on/off" << std::endl;
     std::cout << "\tPress 'v' to toggle verbose on/off" << std::endl;
     std::cout << "\tPress 'z' to toggle profiling log on/off" << std::endl;
     std::cout << "\tPress 'y' to toggle profiling view on/off" << std::endl;
     std::cout << "================================================================" << std::endl;
 
     bool bStep = false;
     bool bPause = false;
     bool bWrite = false;
     
     bool bUseImage = args.testimage;
     bool bMirrorImage = false;
     bool bShowDetection = true;
     bool bShowExtractROI = true;
     bool bShowLandmarks = true;
     bool bShowDebugImage = false;
     bool bShowScores = false;
     bool bShowFPS = args.fps;
     bool bVerbose = args.verbose;
     bool bViewOutput = !args.withoutview;
     bool bProfileLog = args.profilelog;
     bool bProfileView = args.profileview;
 
     std::string app_main_title = blaze_title + " Demo";
     std::string app_ctrl_title = blaze_title + " Demo";
     std::string app_debug_title = blaze_title + " Debug";
     std::string app_scores_title = blaze_title + " Detection Scores (sigmoid)";
     //
     std::string profiling_latency_title = "Latency (sec)";
     std::string profiling_performance_title = "Performance (FPS)";
     
     float scale = 1.0;
     int text_fontType = cv::FONT_HERSHEY_SIMPLEX;
     double text_fontSize = 0.75 * scale;
     cv::Scalar text_color(0, 0, 255);
     int text_lineSize = std::max(1, int(2 * scale));
     int text_lineType = cv::LINE_AA;
 
     double thresh_min_score = blaze_detector->min_score_thresh;
     double thresh_min_score_prev = thresh_min_score;
 
     double thresh_nms = blaze_detector->min_suppression_threshold;
     double thresh_nms_prev = thresh_nms;
 
     double thresh_confidence = 0.5;
     double thresh_confidence_prev = thresh_confidence;
     
     int thresh_min_score_percent = int(thresh_min_score*100);
     int thresh_nms_percent = int(thresh_nms*100);
     int thresh_confidence_percent = int(thresh_confidence*100);
     if (bViewOutput) {
         cv::namedWindow(app_main_title);
         
         // Create slider for thresh_min_score
         cv::createTrackbar("threshMinScore", app_ctrl_title, NULL, 100, on_trackbar);
         cv::setTrackbarPos("threshMinScore", app_ctrl_title,thresh_min_score_percent);
 
         // Create slider for thresh_nms
         cv::createTrackbar("threshNMS", app_ctrl_title, NULL, 100, on_trackbar);
         cv::setTrackbarPos("threshNMS", app_ctrl_title,thresh_nms_percent);
 
         // Create slider for thresh_confidence
         cv::createTrackbar("threshConfidence", app_ctrl_title, NULL, 100, on_trackbar);
         cv::setTrackbarPos("threshConfidence", app_ctrl_title,thresh_confidence_percent);
     }
     
     int frame_count = 0;
 
     // init the real-time FPS counter
     int rt_fps_count = 0;
     int rt_fps_valid = 0;
     double rt_fps = 0.0;
     std::string rt_fps_message = "N/A"; // "FPS: {0:.2f}".format(rt_fps)
     int rt_fps_x = int(10 * scale);
     int rt_fps_y = int((frame_height - 10) * scale);
     auto rt_fps_time = std::chrono::high_resolution_clock::now();
 
     while (running) {
         // init the real-time FPS counter
         if ( rt_fps_count == 0 ) {
             rt_fps_time = std::chrono::high_resolution_clock::now();
         }
         frame_count++;
       
         cv::Mat frame;
         if ( bUseImage == true ) {
             frame = cv::imread("../woman_hands.jpg");
             if (frame.empty()) {
                 std::cerr << "Error: Unable to load the input image." << std::endl;
                 break;
             }
         }
         else {
             // Capture a frame from the webcam
             cap >> frame;  
             if (frame.empty()) {
                 std::cerr << "Error: Could not read frame from the webcam." << std::endl;
                 break;
             }
         }
 
         if ( bMirrorImage ) {
             // Mirror horizontally for selfie-mode
             cv::flip(frame, frame, 1);
         }
         
         // Profiling
         for (pipeline_id = 0; pipeline_id < nb_blaze_pipelines; pipeline_id++ ) {
             //prof_title[pipeline_id] = "";
             prof_detector_qty[pipeline_id] = 0;
             prof_resize[pipeline_id] = 0.0;
             prof_detector_pre[pipeline_id] = 0.0;
             prof_detector_model[pipeline_id] = 0.0;
             prof_detector_post[pipeline_id] = 0.0;
             prof_extract_roi[pipeline_id] = 0.0;
             prof_landmark_pre[pipeline_id] = 0.0;
             prof_landmark_model[pipeline_id] = 0.0;
             prof_landmark_post[pipeline_id] = 0.0;
             prof_annotate[pipeline_id] = 0.0;
             //
             prof_total[pipeline_id] = 0.0;
             prof_fps[pipeline_id] = 0.0;
         }
 
         // Pipelines
         pipeline_id = 0;
         
         // Get trackbar values
         if (bViewOutput) {
             // thresh_min_score
             thresh_min_score_percent = cv::getTrackbarPos("threshMinScore", app_ctrl_title);
             if (thresh_min_score_percent < 10) {
                 thresh_min_score_percent = 10;
                 cv::setTrackbarPos("threshMinScore", app_ctrl_title,thresh_min_score_percent);
             }
             thresh_min_score = ((double)thresh_min_score_percent)/100.0;
             if (thresh_min_score != thresh_min_score_prev) {
                 blaze_detector->min_score_thresh = thresh_min_score;
                 thresh_min_score_prev = thresh_min_score;
                 std::cout << "[INFO] thresh_min_score=" << thresh_min_score << std::endl;
             }
             
             // thresh_nms
             thresh_nms_percent = cv::getTrackbarPos("threshNMS", app_ctrl_title);
             if (thresh_nms_percent > 99) {
                 thresh_nms_percent = 99;
                 cv::setTrackbarPos("threshNMS", app_ctrl_title,thresh_nms_percent);
             }
             thresh_nms = ((double)thresh_nms_percent)/100.0;
             if (thresh_nms != thresh_nms_prev) {
                 blaze_detector->min_suppression_threshold = thresh_nms;
                 thresh_nms_prev = thresh_nms;
                 std::cout << "[INFO] thresh_nms=" << thresh_nms << std::endl;
             }
             
             // thresh_confidence
             thresh_confidence_percent = cv::getTrackbarPos("threshConfidence", app_ctrl_title);
             thresh_confidence = ((double)thresh_confidence_percent)/100.0;
             if (thresh_confidence != thresh_confidence_prev) {
                 thresh_confidence_prev = thresh_confidence;
                 std::cout << "[INFO] thresh_confidence=" << thresh_confidence << std::endl;
             }
         }
                   
         // Prepare output image
         cv::Mat output = frame.clone();
         cv::Mat rgb_frame;
         
         // Convert BGR to RGB for processing
         auto resize_start = std::chrono::high_resolution_clock::now();
         cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
         
         // Step 1: Resize and pad image for detector
         auto [resized_img, scale, pad] = blaze_detector->resize_pad(rgb_frame);
         auto resize_end = std::chrono::high_resolution_clock::now();
         prof_resize[pipeline_id] = std::chrono::duration<double>(resize_end - resize_start).count();
 
         cv::Mat debug_img;
         if (bShowDebugImage) {
             cv::resize(resized_img, debug_img, cv::Size(blaze_landmark->resolution, blaze_landmark->resolution));
             //cv::cvtColor(debug_img, debug_img, cv::COLOR_RGB2BGR);
             debug_img.convertTo(debug_img, CV_32F, 1.0 / 255.0);
 
         }
         
         // Step 2: Palm detection
         auto batch_results = blaze_detector->predict_on_batch(std::vector<cv::Mat>{resized_img});
         std::vector<blaze::Detection> normalized_detections = batch_results.empty() ? std::vector<blaze::Detection>() : batch_results[0];
         
         prof_detector_pre[pipeline_id] = blaze_detector->get_profile_pre();
         prof_detector_model[pipeline_id] = blaze_detector->get_profile_model();
         prof_detector_post[pipeline_id] = blaze_detector->get_profile_post();
 
         if (bShowScores) {
             cv::Mat detection_scores_chart;
             detection_scores_chart = blaze::draw_detection_scores( blaze_detector->detection_scores, blaze_detector->min_score_thresh );
             cv::imshow(app_scores_title,detection_scores_chart);
         }        
         prof_detector_qty[pipeline_id] = normalized_detections.size(); 
             
         if (!normalized_detections.empty()) {
             auto extract_start = std::chrono::high_resolution_clock::now();
             
             // Step 3: Denormalize detections
             std::vector<blaze::Detection> detections = blaze_detector->denormalize_detections(normalized_detections, scale, pad);
 
 
             // // 添加打印原始图像坐标的代码
             // std::cout << "\n=== 映射到原始图像后的坐标信息 ===" << std::endl;
             // td::cout << std::fixed << std::setprecision(1);
             // for (size_t i = 0; i < detections.size(); ++i) {
             //     const auto& detection = detections[i];
             //     std::cout << "[原始图像 Detection " << i+1 << "] 置信度: " << detection.score << std::endl;
             //     std::cout << "[原始图像 Detection " << i+1 << "] 边界框: (" << detection.xmin << ", " << detection.ymin 
             //         << ") - (" << detection.xmax << ", " << detection.ymax << ")" << std::endl;
             //     std::cout << "[原始图像 Detection " << i+1 << "] 7个关键点坐标: " << std::endl;
     
             //     // 定义关键点索引与名称的映射（通常的布局）
             //     std::vector<std::string> keypoint_names = {
             //         "手腕中心", "拇指根部", "中指根部", 
             //         "小指根部", "拇指指尖", "食指指尖", "中指指尖"
             //     };
     
             //     for (int k = 0; k < 7 && k < static_cast<int>(detection.keypoints.size()); ++k) {
             //         const auto& kp = detection.keypoints[k];
             //         std::string kp_name = (k < static_cast<int>(keypoint_names.size())) ? keypoint_names[k] : "未知点";
             //         std::cout << "  关键点[" << k << "] (" << kp_name << "): (" << kp.x << ", " << kp.y << ")" << std::endl;
             //     }
             //     std::cout << "----------------------------------------" << std::endl;
             // }
             // std::cout.unsetf(std::ios_base::fixed);
     
             // Step 4: Convert detections to ROIs
             std::vector<blaze::ROI> rois = blaze_detector->detection2roi(detections);
     
             std::vector<double> xc, yc, theta, scale;
             for (const auto& roi : rois) {
                 xc.push_back(roi.xc);
                 yc.push_back(roi.yc);
                 theta.push_back(roi.theta);
                 scale.push_back(roi.scale);
             }
     
             // Step 5: Extract ROI images
             std::vector<cv::Mat> roi_imgs;
             std::vector<cv::Mat> roi_affine;
             std::vector<std::vector<cv::Point2f>> roi_boxes;
             std::tie(roi_imgs, roi_affine, roi_boxes) = blaze_landmark->extract_roi(rgb_frame, xc, yc, theta, scale);
             
             auto extract_end = std::chrono::high_resolution_clock::now();
             prof_extract_roi[pipeline_id] = std::chrono::duration<double>(extract_end - extract_start).count();
             
             // Step 6: landmark detection
             std::vector<std::vector<double>> flags;
             std::vector<std::vector<std::vector<double>>> normalized_landmarks_vec;
             std::vector<std::vector<double>> handedness;
             //std::tie(flags, normalized_landmarks_vec) = blaze_landmark->predict(roi_imgs);
             std::tie(flags, normalized_landmarks_vec, handedness) = blaze_landmark->predict(roi_imgs);
 
             // Process handedness (if available)
             std::vector<std::string> handedness_results;
             if (!handedness.empty()) {
                 for (size_t i = 0; i < handedness.size(); ++i) {
                     std::string handedness_result;
                     if ( bMirrorImage == false ) {
                         handedness[i][0] = 1.0 - handedness[i][0];
                     }
                     if ( handedness[i][0] > 0.5 ) {
                         handedness_result = "left";
                     }
                     else {
                         handedness_result = "right";
                     }
                     handedness_results.push_back(handedness_result);
                 }
             }
             
             prof_landmark_pre[pipeline_id] = blaze_landmark->get_profile_pre();
             prof_landmark_model[pipeline_id] = blaze_landmark->get_profile_model();
             prof_landmark_post[pipeline_id] = blaze_landmark->get_profile_post();
     
             // Step 7: Convert to Point3d
             std::vector<std::vector<cv::Point3d>> landmarks_3d;
             for (const auto& batch : normalized_landmarks_vec) {
                 std::vector<cv::Point3d> landmark_batch;
                 for (const auto& landmark : batch) {
                     landmark_batch.emplace_back(landmark.size() > 0 ? landmark[0] : 0, landmark.size() > 1 ? landmark[1] : 0, landmark.size() > 2 ? landmark[2] : 0);
                 }
                 landmarks_3d.push_back(landmark_batch);
             }
     
             auto annotate_start = std::chrono::high_resolution_clock::now();
     
             // Step 8: Denormalize landmarks
             std::vector<std::vector<cv::Point3d>> landmarks = blaze_landmark->denormalize_landmarks(landmarks_3d, roi_affine);
     
             // Step 9: Process landmarks
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
                     
                     // Draw landmarks
                     if ( bShowLandmarks ) {
                         if (blaze_landmark_type == "blazehandlandmark") {
                             if ( handedness_results.empty() ) {
                                 blaze::draw_landmarks(output, landmark_points, blaze::HAND_CONNECTIONS, cv::Scalar(0,255,0), 4, 2); // green (BGR format)
                             }
                             else if ( handedness_results[i] == "left" ) {
                                 blaze::draw_landmarks(output, landmark_points, blaze::HAND_CONNECTIONS, cv::Scalar(0,255,0), 4, 2); // green (BGR format)
                             }
                             else {
                                 blaze::draw_landmarks(output, landmark_points, blaze::HAND_CONNECTIONS, cv::Scalar(190, 161, 0), 4, 2); // aqua (BGR format)
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
             }
     
             // Draw detections and ROIs
             if ( bShowDetection ) {
                 blaze::draw_detections(output, detections);
             }
             if ( bShowExtractROI ) {
                 blaze::draw_roi(output, roi_boxes);
             }
             
             auto annotate_end = std::chrono::high_resolution_clock::now();
             prof_annotate[pipeline_id] = std::chrono::duration<double>(annotate_end - annotate_start).count();
         
             if (bShowDebugImage) {
             
                 // Visualize each ROI and its landmarks
                 for (size_t i = 0; i < roi_imgs.size() && i < normalized_landmarks_vec.size(); ++i) {
                     cv::Mat roi_disp;
                     roi_imgs[i].copyTo(roi_disp);
             
                     //cv::resize(roi_disp, roi_disp, cv::Size(blaze_landmark->resolution, blaze_landmark->resolution));
                     //cv::cvtColor(roi_disp, roi_disp, cv::COLOR_RGB2BGR);
                     //roi_disp.convertTo(roi_disp, CV_32F, 1.0 / 255.0);
                                 
                     // Convert normalized_landmarks_vec[i] (vector<vector<double>>) to vector<cv::Point2f>
                     std::vector<cv::Point2f> roi_landmarks;
                     for (const auto& pt : normalized_landmarks_vec[i]) {
                         if (pt.size() >= 2) {
                             //roi_landmarks.emplace_back(static_cast<float>(pt[0]) * blaze_landmark->resolution,
                             //                          static_cast<float>(pt[1]) * blaze_landmark->resolution);
                             roi_landmarks.emplace_back(static_cast<float>(pt[0]), static_cast<float>(pt[1]));
                         }
                     }
             
                     // Draw landmarks according to type
                     if (blaze_landmark_type == "blazehandlandmark") {
                         if ( handedness_results.empty() ) {
                             blaze::draw_landmarks(roi_disp, roi_landmarks, blaze::HAND_CONNECTIONS, cv::Scalar(0,255,0), 4, 2); // green (RGB format)
                         }
                         else if ( handedness_results[i] == "left" ) {
                             blaze::draw_landmarks(roi_disp, roi_landmarks, blaze::HAND_CONNECTIONS, cv::Scalar(0,255,0), 4, 2); // green (RGB format)
                         }
                         else {
                             blaze::draw_landmarks(roi_disp, roi_landmarks, blaze::HAND_CONNECTIONS, cv::Scalar(0, 161, 190), 4, 2); // aqua (RGB format)
                         }
                     } else if (blaze_landmark_type == "blazefacelandmark") {
                         blaze::draw_landmarks(roi_disp, roi_landmarks, blaze::FACE_CONNECTIONS, cv::Scalar(0,255,0), 1, 2);
                     } else if (blaze_landmark_type == "blazeposelandmark") {
                         if (roi_landmarks.size() > 33) {
                             blaze::draw_landmarks(roi_disp, roi_landmarks, blaze::POSE_FULL_BODY_CONNECTIONS, cv::Scalar(0,255,0), 4, 2);
                         } else {
                             blaze::draw_landmarks(roi_disp, roi_landmarks, blaze::POSE_UPPER_BODY_CONNECTIONS, cv::Scalar(0,255,0), 4, 2);
                         }
                     }
             
                     // Concatenate the debug images horizontally
                     cv::Mat temp;
                     cv::hconcat(debug_img, roi_disp, temp);
                     debug_img = temp;
                 }
             
             } // if (bShowDebugImage)
 
         } // if (!normalized_detections.empty()) {
 
         if ( bShowDebugImage ) {
             // Show the debug window
             cv::cvtColor(debug_img, debug_img, cv::COLOR_RGB2BGR);
             cv::imshow(app_debug_title, debug_img);
         }
                               
         // display real-time FPS counter (if valid)
         if ( rt_fps_valid == true && bShowFPS ) {
             cv::putText(output,rt_fps_message, cv::Point(rt_fps_x,rt_fps_y),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType);
         }
       
         if (bViewOutput) {
             cv::imshow(app_main_title, output);
         }
 
         // Profiling
         if (bProfileLog || bProfileView) {
            prof_title[pipeline_id] = blaze_title;
            #
            prof_total[pipeline_id] = prof_resize[pipeline_id] + \
                                      prof_detector_pre[pipeline_id] + \
                                      prof_detector_model[pipeline_id] + \
                                      prof_detector_post[pipeline_id];
            if (!normalized_detections.empty()) {
                prof_total[pipeline_id] += prof_extract_roi[pipeline_id] + \
                                           prof_landmark_pre[pipeline_id] + \
                                           prof_landmark_model[pipeline_id] + \
                                           prof_landmark_post[pipeline_id] + \
                                           prof_annotate[pipeline_id];
            }
            prof_fps[pipeline_id] = 1.0 / prof_total[pipeline_id];
            
            if (bProfileLog) {
               // Get timestamp as string
               auto now = std::chrono::system_clock::now();
               std::time_t now_time = std::chrono::system_clock::to_time_t(now);
               std::tm tm_now;
 #ifdef _WIN32
               localtime_s(&tm_now, &now_time);
 #else
               localtime_r(&now_time, &tm_now);
 #endif
               std::ostringstream oss;
               oss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S");
               std::string timestamp = oss.str();
           
               int pipeline_id = 0;
           
               std::ostringstream csv_str;
               csv_str << timestamp << ","
                       << user << ","
                       << host << ","
                       << prof_title[pipeline_id] << ","
                       << prof_detector_qty[pipeline_id] << ","
                       << prof_resize[pipeline_id] << ","
                       << prof_detector_pre[pipeline_id] << ","
                       << prof_detector_model[pipeline_id] << ","
                       << prof_detector_post[pipeline_id] << ","
                       << prof_extract_roi[pipeline_id] << ","
                       << prof_landmark_pre[pipeline_id] << ","
                       << prof_landmark_model[pipeline_id] << ","
                       << prof_landmark_post[pipeline_id] << ","
                       << prof_annotate[pipeline_id] << ","
                       << prof_total[pipeline_id] << ","
                       << prof_fps[pipeline_id] << "\n";
           
               if (bVerbose) {
                   std::cout << "[PROFILING] " << csv_str.str();
               }
           
               f_profile_csv << csv_str.str();
            }
            
            if (bProfileView) {
                std::vector<std::vector<double>> latency_values = {
                    prof_resize,
                    prof_detector_pre,
                    prof_detector_model,
                    prof_detector_post,
                    prof_extract_roi,
                    prof_landmark_pre,
                    prof_landmark_model,
                    prof_landmark_post,
                    prof_annotate
                };
                cv::Mat profiling_latency_chart = blaze::draw_stacked_bar_chart(prof_title, latency_labels, latency_values, blaze::stacked_bar_latency_colors, profiling_latency_title);
                cv::imshow(profiling_latency_title, profiling_latency_chart);
 
                std::vector<std::vector<double>> performance_values = {
                    prof_fps
                };
                cv::Mat profiling_performance_chart = blaze::draw_stacked_bar_chart(prof_title, performance_labels, performance_values, blaze::stacked_bar_performance_colors, profiling_performance_title);
                cv::imshow(profiling_performance_title, profiling_performance_chart);
            }
         }
                 
         if (bWrite == true) {
             std::string filename = "blaze_detect_live_frame" + std::to_string(frame_count) + "_input.tif";
             std::cout << "Capturing " << filename << " ..." << std::endl;
             cv::imwrite(output_dir+"/"+filename, frame);
       
             filename = "blaze_detect_live_frame" + std::to_string(frame_count) + "_detection.tif";
             std::cout << "Capturing " << filename << " ..." << std::endl;
             cv::imwrite(output_dir+"/"+filename, output);
             
             if (bShowDebugImage == true) {
                // ...
             }
         }
 
         if (bProfileLog || bProfileView) {
             
         }
         
         int key;
         if ( bStep == true ) {
             key = cv::waitKey(0);
         } else if ( bPause == true ) {
             key = cv::waitKey(0);
         } else {
             key = cv::waitKey(1);
         }
         //if ( key != -1 ) std::cout << "[INFO] key = " << key << std::endl;
         
         bWrite = false;
         if (key == 119) { // 'w'
             bWrite = true;
         }
       
         if (key == 115) { // 's'
             bStep = true;
         }
       
         if (key == 112) { // 'p'
             bPause = !bPause;
         }
         
         if (key == 99) { // 'c'
             bStep = false;
             bPause = false;
         }
       
         if (key == 116) { // 't'
             bUseImage = !bUseImage;
             std::cout << "[INFO] bUseImage = " << bUseImage << std::endl;
         }
       
         if (key == 104) { // 'h'
             bMirrorImage = !bMirrorImage;
             std::cout << "[INFO] bMirrorImage = " << bMirrorImage << std::endl;
         }
       
         if (key == 97) { // 'a'
             bShowDetection = !bShowDetection;
             std::cout << "[INFO] bShowDetection = " << bShowDetection << std::endl;
         }
       
         if (key == 98) { // 'b'
             bShowExtractROI = !bShowExtractROI;
             std::cout << "[INFO] bShowExtractROI = " << bShowExtractROI << std::endl;
         }
       
         if (key == 108) { // 'l'
             bShowLandmarks = !bShowLandmarks;
             std::cout << "[INFO] bShowLandmarks = " << bShowLandmarks << std::endl;
         }
       
         if (key == 100) { // 'd'
             bShowDebugImage = !bShowDebugImage;
             std::cout << "[INFO] bShowDebugImage = " << bShowDebugImage << std::endl;
             if (!bShowDebugImage) {
                 cv::destroyWindow(app_debug_title);
             }
         }
       
         if (key == 101) { //'e'
             bShowScores = !bShowScores;
             std::cout << "[INFO] bShowScores = " << bShowScores << std::endl;
             //blaze_detector->display_scores(bShowScores);
             if (!bShowScores) { 
                 cv::destroyWindow(app_scores_title);
             } 
         }
       
         if (key == 102) { // 'f'
             bShowFPS = !bShowFPS;
             std::cout << "[INFO] bShowFPS = " << bShowFPS << std::endl;
         }
         
         if (key == 118) { // 'v'
             bVerbose = !bVerbose; 
             std::cout << "[INFO] bVerbose = " << bVerbose << std::endl;
             blaze_detector->set_debug(bVerbose); 
             blaze_landmark->set_debug(bVerbose);    
         }
 
         if (key == 122) {  // 'z'
             bProfileLog = !bProfileLog;
             std::cout << "[INFO] bProfileLog = " << bProfileLog << std::endl;
             blaze_detector->set_profile(bProfileLog||bProfileView); 
             blaze_landmark->set_profile(bProfileLog||bProfileView);
         }
         
         if (key == 121) { // 'y'
             bProfileView = !bProfileView; 
             std::cout << "[INFO] bProfileView = " << bProfileView << std::endl;
             blaze_detector->set_profile(bProfileLog||bProfileView); 
             blaze_landmark->set_profile(bProfileLog||bProfileView);
             if (!bProfileView) { 
                 cv::destroyWindow(profiling_latency_title); 
                 cv::destroyWindow(profiling_performance_title);
             } 
         } 
       
         if (key == 27 || key == 113) { // ESC or 'q'
             break;
         }       
       
         rt_fps_count++;
         if (rt_fps_count == 10) {
             auto now = std::chrono::high_resolution_clock::now();
             std::chrono::duration<double> t = now - rt_fps_time;
             rt_fps_valid = 1;
             rt_fps = 10.0 / t.count();
             std::ostringstream oss;
             oss << "FPS: " << std::fixed << std::setprecision(2) << rt_fps;
             rt_fps_message = oss.str();
             
             rt_fps_count = 0;
         }
       
     } // while (1)
     
     f_profile_csv.close();
     cv::destroyAllWindows();
     
     return 0;
 }
 
