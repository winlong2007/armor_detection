#include "robotmaster_vision/pnp_solver.h"
#include <opencv2/calib3d.hpp>

using namespace cv;

bool solve_armor_pnp(const std::vector<cv::Point2f>& image_points,
                     const cv::Mat& camera_matrix,
                     const cv::Mat& dist_coeffs,
                     cv::Mat& rvec, cv::Mat& tvec) {
    if (image_points.size() != 4) return false;
    // 使用SOLVEPNP_ITERATIVE，需要至少4个点
    return solvePnP(ARMOR_POINTS, image_points, camera_matrix, dist_coeffs, rvec, tvec, false, SOLVEPNP_ITERATIVE);
}