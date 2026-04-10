#ifndef KALMAN_TRACKER_H
#define KALMAN_TRACKER_H

#include <opencv2/video/tracking.hpp>
#include <vector>

cv::KalmanFilter init_kalman();

// 核心：基于当前滤波器状态，预测未来几帧的 3D 坐标序列 (实现选项二任务三)
std::vector<cv::Point3f> predict_future_trajectory(const cv::KalmanFilter& kf, int future_steps);

#endif // KALMAN_TRACKER_H
