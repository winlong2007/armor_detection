#include "robotmaster_vision/geometry_utils.h"
#include <opencv2/opencv.hpp>

using namespace cv;

float calculate_distance(const RotatedRect& armor, float actual_width, float focal) {
    float width_px = max(armor.size.width, armor.size.height);
    if (width_px == 0) return 0;
    return (actual_width * focal) / width_px;
}

Point3f pixel_to_camera(float u, float v, float depth, float fx, float fy, float cx, float cy) {
    float X = (u - cx) * depth / fx;
    float Y = (v - cy) * depth / fy;
    return Point3f(X, Y, depth);
}

Mat get_rotated_roi(const Mat& img, const RotatedRect& rect) {
    Point2f center = rect.center;
    Size2f size = rect.size;
    float angle = rect.angle;
    Mat M = getRotationMatrix2D(center, angle, 1.0);
    Mat rotated;
    warpAffine(img, rotated, M, img.size());

    // 计算裁剪矩形
    int x = int(center.x - size.width / 2);
    int y = int(center.y - size.height / 2);
    int w = int(size.width);
    int h = int(size.height);

    // 边界保护：确保矩形完全在图像内
    x = max(0, min(x, rotated.cols - 1));
    y = max(0, min(y, rotated.rows - 1));
    if (x + w > rotated.cols) w = rotated.cols - x;
    if (y + h > rotated.rows) h = rotated.rows - y;

    // 如果宽度或高度非正，返回空矩阵
    if (w <= 0 || h <= 0) return Mat();

    Rect roi_rect(x, y, w, h);
    return rotated(roi_rect);
}