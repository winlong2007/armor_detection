#include "robotmaster_vision/kalman_tracker.h"
#include <opencv2/opencv.hpp>

using namespace cv;

KalmanFilter init_kalman() {
    KalmanFilter kf(6, 3, 0);
    kf.transitionMatrix = (Mat_<float>(6,6) <<
        1,0,0,1,0,0,
        0,1,0,0,1,0,
        0,0,1,0,0,1,
        0,0,0,1,0,0,
        0,0,0,0,1,0,
        0,0,0,0,0,1);
    kf.measurementMatrix = (Mat_<float>(3,6) <<
        1,0,0,0,0,0,
        0,1,0,0,0,0,
        0,0,1,0,0,0);
    setIdentity(kf.processNoiseCov, Scalar::all(0.01));
    kf.processNoiseCov.at<float>(3,3) = 5.0f;
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