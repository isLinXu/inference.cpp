#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>

#include "tnn/core/macro.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/dims_vector_utils.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/core/tnn.h"

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace tnn;

int main(int argc, char** argv) {


    std::string image_path = "your_image_path";
    cv::Mat cv_image = cv::imread(image_path, 1);
    if (cv_image.empty()) {
        std::cerr << "Error: Failed to read image" << std::endl;
        return -1;
    }

    cv::resize(cv_image, cv_image, cv::Size(224, 224));
    cv::cvtColor(cv_image, cv_image, cv::COLOR_BGR2RGB);
    cv_image.convertTo(cv_image, CV_32FC3, 1.0 / 255, 0);

    TNN_NS::Mat image_mat(TNN_NS::DEVICE_NAIVE, TNN_NS::NCHW_FLOAT, {1, 3, 224, 224}, cv_image.data);

    // 加载模型
    std::string proto_file_path = "your_proto_file_path";
    std::string model_file_path = "your_model_file_path";
    TNN_NS::ModelConfig model_config;
    model_config.params.push_back(proto_file_path);
    model_config.params.push_back(model_file_path);
    TNN_NS::TNN tnn;
    auto init_status = tnn.Init(model_config);
    if (init_status != TNN_NS::TNN_OK) {
        std::cerr << "Error: Failed to initialize library" << std::endl;
        return -1;
    }

    // 创建实例
    TNN_NS::NetworkConfig network_config;
    network_config.device_type = TNN_NS::DEVICE_ARM;
    TNN_NS::Status error;
    auto instance = tnn.CreateInst(network_config,error);
//    auto instance = tnn.CreateInst(network_config);
    if (!instance) {
        std::cerr << "Error: Failed to create library instance" << std::endl;
        return -1;
    }

//    // 读取图像
//    TNN_NS::DimsVector image_dims = {1, 3, 224, 224};  // 假设输入图像尺寸为224x224
//    auto image_mat = TNN_NS::MatUtils::GetMatFromImage(image_path, image_dims);
//    if (!image_mat) {
//        std::cerr << "Error: Failed to read image" << std::endl;
//        return -1;
//    }
    // 创建输入Mat
    TNN_NS::DimsVector input_dims = {1, 3, 224, 224};
    auto input_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_ARM, TNN_NS::N8UC3, input_dims, nullptr);

    tnn::MatConvertParam input_cvt_param;

    // 设置输入
    auto input_status = instance->SetInputMat(input_mat, input_cvt_param);
    if (input_status != TNN_NS::TNN_OK) {
        std::cout << "SetInputMat failed: " << input_status.description().c_str();
        return -1;
    }

    // 设置输入

//    auto input_blob = instance->GetBlob("input_blob_name");
//    TNN_NS::BlobConverter converter(input_blob);
//    converter.ConvertFromMat(*image_mat);

    // 前向传播
    instance->Forward();

    // 获取输出

//    auto output_blob = instance->GetBlob("output_blob_name");
//    std::vector<float> output_data(output_blob->GetBlobDesc().dims[1]);
//    TNN_NS::Mat output_mat(TNN_NS::DEVICE_ARM, TNN_NS::NCHW_FLOAT, output_blob->GetBlobDesc().dims, output_data.data());
//    tnn::MatConvertParam MatConvertParams;
//    auto status = instance->GetOutputMat(output_mat,tnn::MatConvertParams());
//    converter.ConvertToMat(output_mat);

    // 获取最大概率的类别索引
//    int max_index = 0;
//    float max_value = output_data[0];
//    for (int i = 1; i < output_data.size(); ++i) {
//        if (output_data[i] > max_value) {
//            max_value = output_data[i];
//            max_index = i;
//        }
//    }

    // 输出结果
//    std::cout << "Predicted class index: " << max_index << std::endl;

    // 销毁实例
//    tnn.DeleteInst(instance);
    return 0;
}

