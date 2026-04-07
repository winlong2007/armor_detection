#ifndef ARMOR_DETECTOR_NODE_H
#define ARMOR_DETECTOR_NODE_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <map>
#include <onnxruntime_cxx_api.h>
#include <memory>

#include "robotmaster_vision/geometry_utils.h"
#include "robotmaster_vision/kalman_tracker.h"
#include "robotmaster_vision/pnp_solver.h"

class ArmorDetectorNode : public rclcpp::Node
{
public:
  ArmorDetectorNode();

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
  void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  image_transport::Publisher image_pub_;

  cv::KalmanFilter kf_blue_;
  cv::KalmanFilter kf_red_;

  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  bool have_camera_info_;
  cv::Mat map1_, map2_;
  bool maps_initialized_;

  // onnxruntime 相关成员
  std::unique_ptr<Ort::Session> ort_session_;
  Ort::MemoryInfo memory_info_{nullptr};
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;

  std::vector<std::string> class_names_;
  float confidence_threshold_ = 0.5;
  float nms_threshold_ = 0.4;

  struct Armor {
    cv::RotatedRect rect;
    int class_id;
    float confidence;
    std::vector<cv::Point2f> corners;
  };

  std::vector<Armor> run_yolo_inference(const cv::Mat& img);
};

#endif // ARMOR_DETECTOR_NODE_H