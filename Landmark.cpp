#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#include "Landmark.hpp"

namespace blaze {

// ============================================================================
// Landmark Implementation
// ============================================================================

Landmark::Landmark(const std::string& blaze_app)
    : LandmarkBase()
    , blaze_app(blaze_app)
    , input_shape(256, 256)
    , output_shape1(1, 1)
    , output_shape2(1, 63)
    , ctx(0)
    , model_data(nullptr)
    , model_data_size(0)
    , input_attrs(nullptr)
    , output_attrs(nullptr)
    , profile_pre(0.0)
    , profile_model(0.0)
    , profile_post(0.0) {
}

Landmark::~Landmark() {
    // 释放RKNN资源
    if (ctx > 0) {
        rknn_destroy(ctx);
        ctx = 0;
    }
    
    if (model_data) {
        free(model_data);
        model_data = nullptr;
        model_data_size = 0;
    }
    
    if (input_attrs) {
        free(input_attrs);
        input_attrs = nullptr;
    }
    
    if (output_attrs) {
        free(output_attrs);
        output_attrs = nullptr;
    }
}

bool Landmark::load_model(const std::string& model_path) {
    if (DEBUG) {
        std::cout << "[Landmark.load_model] blaze_app= " << blaze_app << std::endl;
        std::cout << "[Landmark.load_model] model_path=" << model_path << std::endl;
    }

    // 加载RKNN模型文件
    std::ifstream model_file(model_path, std::ios::binary);
    if (!model_file) {
        std::cerr << "[Landmark.load_model] Failed to open model file: " << model_path << std::endl;
        return false;
    }
    
    // 获取模型文件大小
    model_file.seekg(0, std::ios::end);
    model_data_size = model_file.tellg();
    model_file.seekg(0, std::ios::beg);
    
    // 分配内存并读取模型
    model_data = (unsigned char*)malloc(model_data_size);
    if (!model_data) {
        std::cerr << "[Landmark.load_model] Failed to allocate memory for model" << std::endl;
        model_file.close();
        return false;
    }
    
    model_file.read((char*)model_data, model_data_size);
    model_file.close();
    
    // 初始化RKNN上下文
    int ret = rknn_init(&ctx, model_data, model_data_size, 0, nullptr);
    if (ret != RKNN_SUCC) {
        std::cerr << "[Landmark.load_model] rknn_init failed! ret = " << ret << std::endl;
        return false;
    }
    
    // 查询SDK版本
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret != RKNN_SUCC) {
        std::cerr << "[Landmark.load_model] rknn_query SDK version failed! ret = " << ret << std::endl;
    } else {
        if (DEBUG) {
            std::cout << "[Landmark.load_model] RKNN SDK version: " << version.api_version << ", driver version: " << version.drv_version << std::endl;
        }
    }
    
    // 查询输入输出数量
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        std::cerr << "[Landmark.load_model] rknn_query input output num failed! ret = " << ret << std::endl;
        return false;
    }
    
    if (DEBUG) {
        std::cout << "[Landmark.load_model] Number of Inputs : " << io_num.n_input << std::endl;
        std::cout << "[Landmark.load_model] Number of Outputs : " << io_num.n_output << std::endl;
    }
    
    // 查询输入属性
    input_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * io_num.n_input);
    memset(input_attrs, 0, sizeof(rknn_tensor_attr) * io_num.n_input);
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            std::cerr << "[Landmark.load_model] rknn_query input attr failed! ret = " << ret << std::endl;
            return false;
        }
        
        if (DEBUG) {
            std::cout << "[Landmark.load_model] Input[" << i << "] type: " << input_attrs[i].type << ", shape: [";
            for (int j = 0; j < input_attrs[i].n_dims; j++) {
                std::cout << input_attrs[i].dims[j];
                if (j < input_attrs[i].n_dims - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // 设置输入形状
        if (i == 0) {
            input_shape.width = input_attrs[i].dims[2];
            input_shape.height = input_attrs[i].dims[1];
            resolution = input_shape.width;
        }
    }
    
    // 查询输出属性
    output_attrs = (rknn_tensor_attr*)malloc(sizeof(rknn_tensor_attr) * io_num.n_output);
    memset(output_attrs, 0, sizeof(rknn_tensor_attr) * io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            std::cerr << "[Landmark.load_model] rknn_query output attr failed! ret = " << ret << std::endl;
            return false;
        }
        
        if (DEBUG) {
            std::cout << "[Landmark.load_model] Output[" << i << "] type: " << output_attrs[i].type << ", shape: [";
            for (int j = 0; j < output_attrs[i].n_dims; j++) {
                std::cout << output_attrs[i].dims[j];
                if (j < output_attrs[i].n_dims - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    // 根据应用类型配置输出形状
    if (blaze_app == "blazehandlandmark") {
        if (this->resolution == 224) { // hand_landmark_lite
            output_shape1 = cv::Size(1, 1);
            output_shape2 = cv::Size(21, 3);
            output_shape3 = cv::Size(1, 1);
        } else if (this->resolution == 256) { // hand_landmark_v0_07
            output_shape1 = cv::Size(1, 1);
            output_shape2 = cv::Size(21, 3);
            output_shape3 = cv::Size(1, 1);
        }
    } else if (blaze_app == "blazefacelandmark") {
        output_shape1 = cv::Size(1, 1);
        output_shape2 = cv::Size(468, 3);
    } else if (blaze_app == "blazeposelandmark") {
        // 根据输出大小设置形状
        int out_size = output_attrs[1].dims[1] * output_attrs[1].dims[2];
        if (out_size == 124) {
            output_shape1 = cv::Size(1, 1);
            output_shape2 = cv::Size(31, 4);
        } else if (out_size == 195) {
            output_shape1 = cv::Size(1, 1);
            output_shape2 = cv::Size(39, 5);
        }
    }
    
    if (DEBUG) {
        std::cout << "[Landmark.load_model] Input Shape: " << input_shape << std::endl;
        std::cout << "[Landmark.load_model] Output1 Shape: " << output_shape1 << std::endl;
        std::cout << "[Landmark.load_model] Output2 Shape: " << output_shape2 << std::endl;
        if (blaze_app == "blazehandlandmark") {
            std::cout << "[Landmark.load_model] Output3 Shape: " << output_shape3 << std::endl;
        }
        std::cout << "[Landmark.load_model] Input Resolution: " << resolution << std::endl;
    }

    return true;
}

cv::Mat Landmark::preprocess(const cv::Mat& input) {
    // image was already pre-processed by extract_roi in blaze_common/Base.cpp
    // format = RGB
    // dtype = float32
    // range = 0.0 - 1.0
    return input;
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>>
Landmark::predict(const std::vector<cv::Mat>& input_images) {
    
    profile_pre = 0.0;
    profile_model = 0.0;
    profile_post = 0.0;
    
    std::vector<std::vector<double>> out1_list;
    std::vector<std::vector<std::vector<double>>> out2_list;
    std::vector<std::vector<double>> out3_list;

    for (const auto& input : input_images) {
        if (DEBUG) {
            std::cout << "[Landmark.predict] Processing input image of size: " 
                      << input.size() << " channels: " << input.channels() << std::endl;
        }
        
        // 1. 预处理
        auto pre_start = std::chrono::high_resolution_clock::now();
        cv::Mat processed_input = preprocess(input);
        auto pre_end = std::chrono::high_resolution_clock::now();
        profile_pre += std::chrono::duration<double>(pre_end - pre_start).count();
        
        if (DEBUG) {
            std::cout << "[Landmark.predict] Preprocessed input size: " 
                      << processed_input.size() << " channels: " << processed_input.channels() << std::endl;
        }
        
        // 2. 准备RKNN输入
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_FLOAT32;
        inputs[0].size = processed_input.cols * processed_input.rows * processed_input.channels() * sizeof(float);
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = processed_input.ptr<float>(0);
        
        // 设置输入
        int ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
        if (ret != RKNN_SUCC) {
            std::cerr << "[Landmark.predict] rknn_inputs_set failed! ret = " << ret << std::endl;
            exit(1);
        }
        
        // 3. 执行推理
        ret = rknn_run(ctx, nullptr);
        if (ret != RKNN_SUCC) {
            std::cerr << "[Landmark.predict] rknn_run failed! ret = " << ret << std::endl;
            exit(1);
        }
        
        // 4. 获取输出
        rknn_output outputs[3];
        memset(outputs, 0, sizeof(outputs));
        
        // 根据不同的应用类型获取不同的输出数量
        int output_count = (blaze_app == "blazehandlandmark") ? 3 : 2;
        
        for (int i = 0; i < output_count; i++) {
            outputs[i].index = i;
            outputs[i].want_float = 1;
        }
        
        ret = rknn_outputs_get(ctx, output_count, outputs, nullptr);
        if (ret != RKNN_SUCC) {
            std::cerr << "[Landmark.predict] rknn_outputs_get failed! ret = " << ret << std::endl;
            exit(1);
        }
        
        auto inference_end = std::chrono::high_resolution_clock::now();
        profile_model += std::chrono::duration<double>(inference_end - inference_start).count();
        
        if (DEBUG) {
            std::cout << "[Landmark.predict] Inference completed, processing results..." << std::endl;
        }
        
        // 5. 处理输出结果
        auto post_start = std::chrono::high_resolution_clock::now();
        
        std::vector<double> out1;
        std::vector<std::vector<double>> out2;
        std::vector<double> out3;
        
        // out1_confidence    shape: [1, 1]
        const float* data1 = (const float*)outputs[1].buf;
        out1 = {static_cast<double>(data1[0])};

        // out2_landmarks    shape: [landmarks, 3]
        int nb_landmarks = output_shape2.width;
        int nb_components = output_shape2.height;
        size_t idx = 0;
        out2.resize(nb_landmarks);
        idx = 0;
        const float* data2 = (const float*)outputs[0].buf;
        for (int l = 0; l < nb_landmarks; ++l) {
            std::vector<double> elements(nb_components);
            for (int e = 0; e < nb_components; ++e) {
                double val = static_cast<double>(data2[idx++]);
                elements[e] = val;
            }
            out2[l] = elements;
        }

        if (blaze_app == "blazehandlandmark") {
            // out3_handedness    shape: [1, 1]
            const float* data3 = (const float*)outputs[2].buf;
            out3 = {static_cast<double>(data3[0])};
        }
        
        // 打印未归一化的21个关键点坐标
        if (blaze_app == "blazehandlandmark" && out2.size() == 21) {
            std::cout << "[Landmark.predict] 未归一化的21个关键点坐标:" << std::endl;
            for (int i = 0; i < out2.size(); ++i) {
                std::cout << "点 " << i << ": x=" << out2[i][0] << ", y=" << out2[i][1] << ", z=" << out2[i][2] << std::endl;
            }
        }
        
        // 释放输出缓冲区
        rknn_outputs_release(ctx, output_count, outputs);
        
        auto post_end = std::chrono::high_resolution_clock::now();
        profile_post += std::chrono::duration<double>(post_end - post_start).count();
        
        if (DEBUG) {
            std::cout << "[Landmark.predict] out1 (confidence) size: " << out1.size() << std::endl;
            std::cout << "[Landmark.predict] out2 (landmarks) size: " << out2.size() << "x" << out2[0].size() << std::endl;
            if (blaze_app == "blazehandlandmark") {
                std::cout << "[Landmark.predict] out3 (handedness) size: " << out3.size() << std::endl;
            }
        }
        
        out1_list.push_back(out1);
        out2_list.push_back(out2);
        if (blaze_app == "blazehandlandmark") {
            out3_list.push_back(out3);
        }
    }
    
    return std::make_tuple(out1_list, out2_list, out3_list);
}

} // namespace blaze