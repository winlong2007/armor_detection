#ifndef GEOMETRY_UTILS_H
#define GEOMETRY_UTILS_H

#include <opencv2/opencv.hpp>

float calculate_distance(const cv::RotatedRect& armor, float actual_width = 230, float focal = 600);
cv::Point3f pixel_to_camera(float u, float v, float depth, float fx, float fy, float cx, float cy);
cv::Mat get_rotated_roi(const cv::Mat& img, const cv::RotatedRect& rect);

#endif // GEOMETRY_UTILS_H