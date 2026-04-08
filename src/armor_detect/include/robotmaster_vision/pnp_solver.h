#ifndef PNP_SOLVER_H
#define PNP_SOLVER_H

#include <opencv2/opencv.hpp>

// 装甲板的世界坐标（单位：mm），假设装甲板为矩形，四个角点顺序：左下、右下、右上、左上
// // 宽度 230mm，高度 120mm（可根据实际调整）
// const std::vector<cv::Point3f> ARMOR_POINTS = {
//     cv::Point3f(-117.5, -63.5, 0),
//     cv::Point3f( 117.5, -63.5, 0),
//     cv::Point3f( 117.5,  63.5, 0),
//     cv::Point3f(-117.5,  63.5, 0)
// };
enum class ArmorType { SMALL, LARGE };

// 输入：装甲板四个角点的图像坐标（顺序需与 ARMOR_POINTS 对应），相机内参矩阵，畸变系数
// 输出：旋转向量和平移向量
bool solve_armor_pnp(const std::vector<cv::Point2f>& image_points,
                     ArmorType type,
                     const cv::Mat& camera_matrix,
                     const cv::Mat& dist_coeffs,
                     cv::Mat& rvec, cv::Mat& tvec);

#endif // PNP_SOLVER_H