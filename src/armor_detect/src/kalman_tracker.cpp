#include "robotmaster_vision/kalman_tracker.h"
#include <opencv2/opencv.hpp>

using namespace cv;

KalmanFilter init_kalman() {
    KalmanFilter kf(6, 3, 0);
    // 状态转移矩阵: x_new = x + v*dt (dt=1)
    kf.transitionMatrix = (Mat_<float>(6,6) <<
        1,0,0,1,0,0,
        0,1,0,0,1,0,
        0,0,1,0,0,1,
        0,0,0,1,0,0,
        0,0,0,0,1,0,
        0,0,0,0,0,1);
    
    // 观测矩阵: z = [x, y, z]
    kf.measurementMatrix = (Mat_<float>(3,6) <<
        1,0,0,0,0,0,
        0,1,0,0,0,0,
        0,0,1,0,0,0);

    setIdentity(kf.processNoiseCov, Scalar::all(0.01));
    kf.processNoiseCov.at<float>(3,3) = 5.0f; // 速度过程噪声
    kf.processNoiseCov.at<float>(4,4) = 5.0f;
    kf.processNoiseCov.at<float>(5,5) = 5.0f;
    
    setIdentity(kf.measurementNoiseCov, Scalar::all(1.0));
    
    kf.statePost.at<float>(0) = 0;
    kf.statePost.at<float>(1) = 0;
    kf.statePost.at<float>(2) = 1000;
    kf.statePost.at<float>(3) = 0;
    kf.statePost.at<float>(4) = 0;
    kf.statePost.at<float>(5) = 0;
    
    setIdentity(kf.errorCovPost, Scalar::all(100));
    return kf;
}

// 基于当前滤波器状态，预测未来 N 个位置点 (实现轨迹可视化)
std::vector<cv::Point3f> predict_future_trajectory(const cv::KalmanFilter& kf, int future_steps) {
    std::vector<cv::Point3f> trajectory;
    KalmanFilter kf_copy = kf; // 制作副本，不破坏当前主状态

    for (int i = 0; i < future_steps; ++i) {
        Mat prediction = kf_copy.predict();
        Point3f pt(prediction.at<float>(0), prediction.at<float>(1), prediction.at<float>(2));
        trajectory.push_back(pt);
    }
    return trajectory;
}