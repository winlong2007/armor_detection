#include "robotmaster_vision/pnp_solver.h"
#include <opencv2/calib3d.hpp>

using namespace cv;

bool solve_armor_pnp(const std::vector<cv::Point2f>& image_points,
                     ArmorType type,
                     const cv::Mat& camera_matrix,
                     const cv::Mat& dist_coeffs,
                     cv::Mat& rvec, cv::Mat& tvec) {
    if (image_points.size() != 4) return false;

    float w = (type == ArmorType::LARGE) ? 235.0f : 140.0f;
    float h = (type == ArmorType::LARGE) ? 127.0f : 125.0f;
    std::vector<cv::Point3f> object_points = {
        {-w/2,  h/2, 0}, // 左上
        { w/2,  h/2, 0}, // 右上
        { w/2, -h/2, 0}, // 右下
        {-w/2, -h/2, 0}  // 左下
    };
    // 使用SOLVEPNP_IPPE，需要至少4个点
    return solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec, false, SOLVEPNP_IPPE);
}