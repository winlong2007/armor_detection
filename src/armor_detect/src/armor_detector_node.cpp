#include "robotmaster_vision/armor_detector_node.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <memory>
#include <iostream>

using namespace std::placeholders;

ArmorDetectorNode::ArmorDetectorNode() : Node("armor_detector_node"), have_camera_info_(false), maps_initialized_(false)
{
    // 声明参数
    this->declare_parameter<float>("fx", 600.0);
    this->declare_parameter<float>("fy", 600.0);
    this->declare_parameter<std::string>("image_topic", "/camera/image");
    this->declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");

    std::string image_topic = this->get_parameter("image_topic").as_string();
    std::string info_topic = this->get_parameter("camera_info_topic").as_string();

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic, 10, std::bind(&ArmorDetectorNode::image_callback, this, _1));
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        info_topic, 10, std::bind(&ArmorDetectorNode::camera_info_callback, this, _1));

    image_pub_ = image_transport::create_publisher(this, "/armor_detection/image");

    // ===== 加载 ONNX 模型（使用 onnxruntime）=====
    std::string package_share_dir = ament_index_cpp::get_package_share_directory("robotmaster_vision");
    std::string model_path = package_share_dir + "/models/best.onnx";

    // 创建 ONNX Runtime 环境
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "armor_detector");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // 尝试启用 CUDA 加速 (针对你的 5060 显卡)
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0; // 使用第一块显卡
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        RCLCPP_INFO(this->get_logger(), "成功启用 ONNX Runtime CUDA 加速，模型将运行在 GPU 上");
    } catch (const std::exception& e) {
        RCLCPP_WARN(this->get_logger(), "未能启用 CUDA 加速，将回退到 CPU 模式: %s", e.what());
    }

    ort_session_ = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

    // 获取输入输出信息（新版 API）
    Ort::AllocatorWithDefaultOptions allocator;

    // 输入名称和形状
    auto input_name_ptr = ort_session_->GetInputNameAllocated(0, allocator);
    input_names_.push_back(input_name_ptr.get());
    Ort::TypeInfo input_type_info = ort_session_->GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape_ = input_tensor_info.GetShape();
    if (input_shape_.size() != 4) {
        RCLCPP_ERROR(this->get_logger(), "模型输入形状不是4维，实际维度: %zu", input_shape_.size());
    } else {
        RCLCPP_INFO(this->get_logger(), "模型输入形状: %ldx%ldx%ldx%ld",
                    input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]);
    }

    // 输出名称和形状
    auto output_name_ptr = ort_session_->GetOutputNameAllocated(0, allocator);
    output_names_.push_back(output_name_ptr.get());
    Ort::TypeInfo output_type_info = ort_session_->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    output_shape_ = output_tensor_info.GetShape();
    RCLCPP_INFO(this->get_logger(), "模型输出形状: %ldx%ldx%ld", output_shape_[0], output_shape_[1], output_shape_[2]);

    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 设置类别名称（根据训练时的类别数修改，这里假设只有一类）
    class_names_ = {"armor"};

    // 初始化卡尔曼滤波器
    kf_blue_ = init_kalman();
    kf_red_ = init_kalman();

    RCLCPP_INFO(this->get_logger(), "装甲板检测节点已启动");
}

void ArmorDetectorNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    if (!have_camera_info_) {
        cv::Mat camera_matrix(3, 3, CV_64F);
        memcpy(camera_matrix.data, msg->k.data(), 9 * sizeof(double));
        camera_matrix = camera_matrix.clone();

        cv::Mat dist_coeffs(1, msg->d.size(), CV_64F);
        memcpy(dist_coeffs.data, msg->d.data(), msg->d.size() * sizeof(double));
        dist_coeffs = dist_coeffs.clone();
        camera_matrix_ = camera_matrix;
        dist_coeffs_ = dist_coeffs;
        have_camera_info_ = true;

        cv::Size image_size(msg->width, msg->height);
        cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_, cv::Mat(),
                                     camera_matrix_, image_size, CV_32FC1, map1_, map2_);
        maps_initialized_ = true;
        RCLCPP_INFO(this->get_logger(), "相机内参已接收，畸变校正映射已初始化");
    }
}

void ArmorDetectorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    // 1. 图像转换
    cv::Mat frame;
    try {
        frame = cv_bridge::toCvShare(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge 转换失败: %s", e.what());
        return;
    }
    
    // 2. 畸变校正
    cv::Mat frame_undistorted;
    if (maps_initialized_) {
        cv::remap(frame, frame_undistorted, map1_, map2_, cv::INTER_LINEAR);
    } else {
        frame_undistorted = frame.clone();
    }

    // 3. 获取相机参数
    float fx = (have_camera_info_) ? camera_matrix_.at<double>(0,0) : this->get_parameter("fx").as_double();
    float fy = (have_camera_info_) ? camera_matrix_.at<double>(1,1) : this->get_parameter("fy").as_double();
    float cx = (have_camera_info_) ? camera_matrix_.at<double>(0,2) : msg->width / 2.0f;
    float cy = (have_camera_info_) ? camera_matrix_.at<double>(1,2) : msg->height / 2.0f;

    cv::Mat result = frame_undistorted.clone();

    // 4. YOLO推理（使用 onnxruntime）
    std::vector<Armor> armors = run_yolo_inference(frame_undistorted);

    // 5. 处理每个检测到的装甲板
    for (const auto& armor : armors) {
        cv::Point2f pts[4];
        armor.rect.points(pts);
        cv::Scalar color = (armor.class_id == 0) ? cv::Scalar(255,0,0) : cv::Scalar(0,0,255);
        for (int i = 0; i < 4; ++i)
            cv::line(result, pts[i], pts[(i+1)%4], color, 2);
        cv::Point2f center = armor.rect.center;

        if (armor.class_id >= 1 && armor.class_id <= 5) {
            cv::putText(result, std::to_string(armor.class_id), cv::Point(center.x-10, center.y-30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        }

        std::vector<cv::Point2f> img_corners;
        if (!armor.corners.empty()) {
            img_corners = armor.corners;
        } else {
            armor.rect.points(pts);
            img_corners = std::vector<cv::Point2f>(pts, pts+4);
        }

        cv::Mat rvec, tvec;
        if (have_camera_info_ && solve_armor_pnp(img_corners, camera_matrix_, dist_coeffs_, rvec, tvec)) {
            cv::Point3f cam(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
            char coord_text[100];
            sprintf(coord_text, "(%.0f,%.0f,%.0f)mm", cam.x, cam.y, cam.z);
            cv::putText(result, coord_text, cv::Point(center.x-50, center.y-50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,255,255), 1);
            float dist = cam.z;

            cv::KalmanFilter& kf = (armor.class_id == 0) ? kf_blue_ : kf_red_;
            cv::Mat measurement = (cv::Mat_<float>(3,1) << center.x, center.y, dist);
            kf.predict();
            kf.correct(measurement);
        } else {
            float dist = calculate_distance(armor.rect, 230, fx);
            cv::Point3f cam = pixel_to_camera(center.x, center.y, dist, fx, fy, cx, cy);
            char coord_text[100];
            sprintf(coord_text, "(%.0f,%.0f,%.0f)mm", cam.x, cam.y, cam.z);
            cv::putText(result, coord_text, cv::Point(center.x-50, center.y-50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,255,255), 1);
        }
    }

    cv::imshow("Armor Detector", result);
    cv::waitKey(1);

    sensor_msgs::msg::Image::SharedPtr out_msg =
        cv_bridge::CvImage(msg->header, "bgr8", result).toImageMsg();
    image_pub_.publish(out_msg);
}

// ===== 使用 onnxruntime 的推理函数 =====
std::vector<ArmorDetectorNode::Armor> ArmorDetectorNode::run_yolo_inference(const cv::Mat& img)
{
    std::vector<Armor> results;
    if (!ort_session_) return results;

    // 获取模型输入尺寸（从 input_shape_ 获取）
    int input_h = input_shape_[2];
    int input_w = input_shape_[3];

    // 预处理：缩放、归一化、转换为 CHW
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_w, input_h));
    cv::Mat blob;
    resized.convertTo(blob, CV_32FC3, 1.0/255.0);
    cv::Mat chw;
    cv::dnn::blobFromImage(blob, chw);  // 输出形状 {1, 3, H, W}

    // 准备输入 tensor
    std::vector<float> input_data((float*)chw.data, (float*)chw.data + chw.total());
    std::vector<int64_t> input_node_dims = {1, 3, input_h, input_w};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_data.data(), input_data.size(),
        input_node_dims.data(), input_node_dims.size());

    // 推理
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor));
    auto output_tensors = ort_session_->Run(Ort::RunOptions{nullptr},
                                            input_names_.data(), input_tensors.data(), input_tensors.size(),
                                            output_names_.data(), output_names_.size());

    // 解析输出（假设输出形状为 [1, num_classes+5, num_boxes]）
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    int num_boxes = output_shape_[2];
    int num_classes = output_shape_[1] - 5;  // 根据训练时的类别数
    int step = output_shape_[1];             // 每个检测框的数据长度

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float scale_x = img.cols / (float)input_w;
    float scale_y = img.rows / (float)input_h;

    for (int i = 0; i < num_boxes; ++i) {
        float* ptr = output_data + i * step;
        float obj_conf = ptr[4];
        if (obj_conf < confidence_threshold_) continue;
        // 寻找最高类别得分
        float max_class_score = 0;
        int class_id = 0;
        for (int j = 0; j < num_classes; ++j) {
            float score = ptr[5 + j];
            if (score > max_class_score) {
                max_class_score = score;
                class_id = j;
            }
        }
        if (max_class_score > confidence_threshold_) {
            float x = ptr[0] * scale_x;
            float y = ptr[1] * scale_y;
            float w = ptr[2] * scale_x;
            float h = ptr[3] * scale_y;
            int left = int(x - w/2);
            int top = int(y - h/2);
            boxes.push_back(cv::Rect(left, top, int(w), int(h)));
            confidences.push_back(obj_conf * max_class_score);
            class_ids.push_back(class_id);
        }
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_, indices);

    for (int idx : indices) {
        Armor armor;
        armor.class_id = class_ids[idx];
        armor.confidence = confidences[idx];
        cv::Rect box = boxes[idx];
        armor.corners = {
            cv::Point2f(box.tl().x, box.tl().y),
            cv::Point2f(box.br().x, box.tl().y),
            cv::Point2f(box.br().x, box.br().y),
            cv::Point2f(box.tl().x, box.br().y)
        };
        armor.rect = cv::RotatedRect(cv::Point2f(box.x + box.width/2.0f, box.y + box.height/2.0f),
                                     cv::Size2f(box.width, box.height), 0);
        results.push_back(armor);
    }
    return results;
}

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmorDetectorNode>());
    rclcpp::shutdown();
    return 0;
}