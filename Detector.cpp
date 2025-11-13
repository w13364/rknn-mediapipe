#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cassert>
#include <iomanip>

#include "Detector.hpp"
#include "rknn_api.h"


namespace blaze {

// ============================================================================
// Detector Implementation
// ============================================================================

Detector::Detector(const std::string& blaze_app)
    : DetectorBase()
    , blaze_app(blaze_app)
    , num_inputs(0)
    , num_outputs(0)
    , profile_pre(0.0)
    , profile_model(0.0)
    , profile_post(0.0) {
    ctx = 0;
    model_data = NULL;
    model_data_size = 0;
}

Detector::~Detector() {
    if (ctx != 0) {
        rknn_destroy(ctx);
        ctx = 0;
    }
    if (model_data) {
        free(model_data);
        model_data = NULL;
        model_data_size = 0;
    }
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp) {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = (unsigned char *)malloc(size);
    if (data == NULL) {
        printf("buffer malloc failure.\n");
        fclose(fp);
        return NULL;
    }
    fseek(fp, 0, SEEK_SET);
    fread(data, 1, size, fp);
    fclose(fp);

    *model_size = size;
    return data;
}

void Detector::load_model(const std::string& model_path) {
    if (DEBUG) {
        std::cout << "[Detector.load_model] blaze_app= " << blaze_app << std::endl;
        std::cout << "[Detector.load_model] model_path=" << model_path << std::endl;
    }

    // Load model
    model_data = blaze::load_model(model_path.c_str(), &model_data_size);
    if (!model_data) {
        std::cerr << "[Detector.load_model] FAILED to load model" << model_path.c_str() << std::endl;
        exit(1);
    }
    
    // Initialize RKNN context
    int ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {
        std::cerr << "[Detector.load_model] FAILED to initialize RKNN context, ret=" << ret << std::endl;
        exit(1);
    }
    
    // Query SDK version
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        std::cerr << "[Detector.load_model] FAILED to query SDK version, ret=" << ret << std::endl;
    } else {
        if (DEBUG) {
            std::cout << "[Detector.load_model] SDK version: " << version.api_version 
                      << " driver version: " << version.drv_version << std::endl;
        }
    }
    
    // Query input/output number
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        std::cerr << "[Detector.load_model] FAILED to query input/output number, ret=" << ret << std::endl;
        exit(1);
    }
    
    num_inputs = io_num.n_input;
    num_outputs = io_num.n_output;
    
    if (DEBUG) {
        std::cout << "[Detector.load_model] Number of Inputs : " << num_inputs << std::endl;
        std::cout << "[Detector.load_model] Number of Outputs : " << num_outputs << std::endl;
    }
    
    // Query input attributes
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            std::cerr << "[Detector.load_model] FAILED to query input attributes, ret=" << ret << std::endl;
            exit(1);
        }
        
        if (DEBUG) {
            std::cout << "[Detector.load_model] Input[" << i << "] "
                      << "Name=" << input_attrs[i].name << " "
                      << "Type=" << get_type_string(input_attrs[i].type) << " "
                      << "Shape=";
            std::cout << "[";
            for (int j = 0; j < input_attrs[i].n_dims; ++j) {
                std::cout << input_attrs[i].dims[j];
                if (j < input_attrs[i].n_dims - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    // Query output attributes
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            std::cerr << "[Detector.load_model] FAILED to query output attributes, ret=" << ret << std::endl;
            exit(1);
        }
        
        if (DEBUG) {
            std::cout << "[Detector.load_model] Output[" << i << "] "
                      << "Name=" << output_attrs[i].name << " "
                      << "Type=" << get_type_string(output_attrs[i].type) << " "
                      << "Shape=";
            std::cout << "[";
            for (int j = 0; j < output_attrs[i].n_dims; ++j) {
                std::cout << output_attrs[i].dims[j];
                if (j < output_attrs[i].n_dims - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    // Get input dimensions
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        if (DEBUG) std::cout << "[Detector.load_model] model is NCHW input fmt" << std::endl;
        in_shape[1] = input_attrs[0].dims[1]; // channel
        in_shape[2] = input_attrs[0].dims[2]; // height
        in_shape[3] = input_attrs[0].dims[3]; // width
    } else {
        if (DEBUG) std::cout << "[Detector.load_model] model is NHWC input fmt" << std::endl;
        in_shape[1] = input_attrs[0].dims[1]; // height
        in_shape[2] = input_attrs[0].dims[2]; // width
        in_shape[3] = input_attrs[0].dims[3]; // channel
    }
    
    x_scale = in_shape[2];
    y_scale = in_shape[1];
    h_scale = in_shape[1];
    w_scale = in_shape[2];
    
    // Get num_anchors from output shape
    num_anchors = output_attrs[1].dims[1];
    
    if (DEBUG) {
        std::cout << "[Detector.load_model] in_shape = [1, " << in_shape[1] 
                  << ", " << in_shape[2] << ", " << in_shape[3] << "]" << std::endl;
        std::cout << "[Detector.load_model] Num Anchors : " << num_anchors << std::endl;
    }
    
    config_model(blaze_app);
}

std::string cv_type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += std::to_string(chans);

    return r;
}

cv::Mat Detector::preprocess(const ImageType& input) {
    cv::Mat preprocessed;

    // Convert to float, scaling values to [0, 1] range
    input.convertTo(preprocessed, CV_32FC3, 1.0 / 255.0);
    
    return preprocessed;
}

std::vector<Detection> Detector::predict_on_image(const ImageType& img) {
    // Use resize_pad to handle arbitrary input image sizes
    auto [resized_img, scale, pad] = resize_pad(img);
    
    // Convert single image to batch format
    std::vector<ImageType> batch = {resized_img};
    
    // Call predict_on_batch with properly sized image
    auto detections = predict_on_batch(batch);
    
    // Return first element from batch results
    if (!detections.empty()) {
        return detections[0];
    } else {
        return {};
    }
}

// 在predict_on_batch函数中，需要修改以下代码：
std::vector<std::vector<Detection>> Detector::predict_on_batch(const std::vector<ImageType>& x) {
    profile_pre = 0.0;
    profile_model = 0.0;
    profile_post = 0.0;
    
    // Validate input dimensions
    assert(x[0].channels() == 3);
    assert(x[0].rows == static_cast<int>(y_scale));
    assert(x[0].cols == static_cast<int>(x_scale));
    
    // 1. Preprocess the images
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat img = x[0].clone();
    
    // Convert BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // 关键修复：调用preprocess函数进行归一化处理
    cv::Mat preprocessed_img = preprocess(img);
    
    // 关键问题在这里：
    // Set up RKNN input
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32; // 修改为FLOAT32类型
    inputs[0].size = in_shape[2] * in_shape[1] * 3 * sizeof(float);
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = preprocessed_img.data;
    
    auto end = std::chrono::high_resolution_clock::now();
    profile_pre = std::chrono::duration<double>(end - start).count();

    // 2. Run neural network　推理部分rknn
    start = std::chrono::high_resolution_clock::now();
    
    // Set input
    int ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0) {
        std::cerr << "[Detector.predict_on_batch] FAILED to set inputs, ret=" << ret << std::endl;
        exit(1);
    }
    
    // Run inference
    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        std::cerr << "[Detector.predict_on_batch] FAILED to run inference, ret=" << ret << std::endl;
        exit(1);
    }
    
    // Get outputs
    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1; // Get float results
    }
    
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if (ret < 0) {
        std::cerr << "[Detector.predict_on_batch] FAILED to get outputs, ret=" << ret << std::endl;
        exit(1);
    }



        // 打印输出张量的原始形式
    std::cout << "[RKNN Output Tensors - Raw Form]" << std::endl;

    // 打印第一个输出张量 (outputs[0] - 回归输出)
    std::cout << "Output[0] (Regression) - Raw Data: " << std::endl;
    float* output0_data = (float*)outputs[0].buf;
    int output0_size = outputs[0].size / sizeof(float);
    // 只打印前20个值以避免输出过多
    int print_count0 = std::min(20, output0_size);
    for (int i = 0; i < print_count0; ++i) {
        std::cout << "output0[" << i << "] = " << output0_data[i] << std::endl;
    }
    if (output0_size > print_count0) {
        std::cout << "... and " << (output0_size - print_count0) << " more values" << std::endl;
    }
    std::cout << "Output[0] size: " << output0_size << " floats" << std::endl;

    // 打印第二个输出张量 (outputs[1] - 分类输出)
    std::cout << "Output[1] (Classification) - Raw Data: " << std::endl;
    float* output1_data = (float*)outputs[1].buf;
    int output1_size = outputs[1].size / sizeof(float);
    // 只打印前20个值以避免输出过多
    int print_count1 = std::min(20, output1_size);
    for (int i = 0; i < print_count1; ++i) {
        std::cout << "output1[" << i << "] = " << output1_data[i] << std::endl;
    }
    if (output1_size > print_count1) {
        std::cout << "... and " << (output1_size - print_count1) << " more values" << std::endl;
    }
    std::cout << "Output[1] size: " << output1_size << " floats" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    
    end = std::chrono::high_resolution_clock::now();
    profile_model = std::chrono::duration<double>(end - start).count();

    start = std::chrono::high_resolution_clock::now();

    // Prepare output data structures
    std::vector<std::vector<std::vector<double>>> out1(1);
    out1[0].resize(num_anchors);
    
    std::vector<std::vector<std::vector<double>>> out2(1);
    out2[0].resize(num_anchors);
    
    // Process classification output (out1)
    float* data1 = (float*)outputs[1].buf;
    for (int a = 0; a < num_anchors; ++a) {
        double val = static_cast<double>(data1[a]);
        out1[0][a] = {val};
    }
    
    // Process regression output (out2)
    float* data2 = (float*)outputs[0].buf;
    for (int a = 0; a < num_anchors; ++a) {
        std::vector<double> coords(num_coords);
        for (int e = 0; e < num_coords; ++e) {
            double val = static_cast<double>(data2[a * num_coords + e]);
            coords[e] = val;
        }
        out2[0][a] = coords;
    }
    
    // For DEBUG output, use std::cout or logging as needed
    assert(out1.size() == 1); // batch
    assert(out1[0].size() == num_anchors);
    assert(out1[0][0].size() == 1);

    assert(out2.size() == 1);
    assert(out2[0].size() == num_anchors);
    assert(out2[0][0].size() == num_coords);

    // 打印原始置信度张量的前10个最大值
    std::cout << "[Detector.predict_on_batch] 原始置信度张量的前10个最大值:" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    // 创建一个存储置信度值和索引的向量
    std::vector<std::pair<double, int>> scores_with_indices;
    for (int a = 0; a < num_anchors; ++a) {
        double score = out1[0][a][0];  // 获取原始置信度值
        scores_with_indices.emplace_back(score, a);
    }
    
    // 按置信度值从大到小排序
    std::sort(scores_with_indices.begin(), scores_with_indices.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 打印前10个最大值及其索引
    int num_to_print = std::min(10, static_cast<int>(scores_with_indices.size()));
    for (int i = 0; i < num_to_print; ++i) {
        std::cout << "排名[" << (i+1) << "]: 索引=" << scores_with_indices[i].second 
                  << ", 原始置信度值=" << scores_with_indices[i].first << std::endl;
    }
    std::cout.unsetf(std::ios_base::fixed);
    std::cout << "----------------------------------------" << std::endl;
    
    // 打印前5个锚框的原始关键点值
    std::cout << "[Detector.predict_on_batch] 模型推理原始关键点值 (前5个锚框):" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    
    // 只打印前5个锚框以避免输出过多
    int num_anchors_to_print = std::min(5, static_cast<int>(num_anchors));
    for (int a = 0; a < num_anchors_to_print; ++a) {
        std::cout << "锚框 " << a << ":" << std::endl;
        
        // 打印边界框原始值
        std::cout << "  边界框原始值: x_center=" << out2[0][a][0] 
                  << ", y_center=" << out2[0][a][1] 
                  << ", width=" << out2[0][a][2] 
                  << ", height=" << out2[0][a][3] << std::endl;
        
        // 打印7个关键点的原始值（每个关键点有x,y两个值）
        std::cout << "  7个关键点原始值: " << std::endl;
        for (int k = 0; k < 7; ++k) {
            int offset = 4 + k * 2;  // 前4个是边界框值，然后是7个关键点的x,y值
            if (offset + 1 < num_coords) {
                std::cout << "    关键点[" << k << "]: x_raw=" << out2[0][a][offset] 
                          << ", y_raw=" << out2[0][a][offset + 1] << std::endl;
            }
        }
        std::cout << "----------------------------------------" << std::endl;
    }
    std::cout.unsetf(std::ios_base::fixed);
    
    // 3. Postprocess the raw predictions
    std::vector<std::vector<Detection>> detections = tensors_to_detections(out2, out1, anchors_);

    // 4. Non-maximum suppression
    std::vector<std::vector<Detection>> filtered_detections;

    for (size_t i = 0; i < detections.size(); ++i) {
        std::vector<Detection> wnms_detections = weighted_non_max_suppression(detections[i]);
        if (!wnms_detections.empty()) {
            filtered_detections.push_back(wnms_detections);
        }
    }

    // Release outputs
    // rknn_outputs_release(ctx, io_numio_num.n_output, outputs);
    rknn_outputs_release(ctx, io_num.n_output, outputs);


    end = std::chrono::high_resolution_clock::now();
    profile_post = std::chrono::duration<double>(end - start).count();

    return filtered_detections;
}

std::pair<std::vector<std::vector<std::vector<float>>>, 
          std::vector<std::vector<std::vector<float>>>>
Detector::process_model_outputs(const std::map<std::string, cv::Mat>& infer_results) {
    
    if (blaze_app == "blazepalm" && num_outputs == 6) {
        //return process_palm_v07_outputs(infer_results);
    } else if (blaze_app == "blazepalm" && num_outputs == 4) {
        //return process_palm_lite_outputs(infer_results);
    } else {
        // Default to lite processing
        //return process_palm_lite_outputs(infer_results);
    }
    
    std::vector<std::vector<std::vector<float>>> out1;
    std::vector<std::vector<std::vector<float>>> out2;
    return std::make_pair(out1, out2);
}


void Detector::set_min_score_threshold(float threshold) {
    min_score_thresh = static_cast<double>(threshold);
    std::cout << "[Detector.set_min_score_threshold] Set threshold to: " << min_score_thresh << std::endl;
}

} // namespace blaze
