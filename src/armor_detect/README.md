# robotmaster_vision - 装甲板检测与位姿解算 ROS 2 节点

## 项目简介

`robotmaster_vision` 是一个基于 ROS 2 的视觉感知节点，用于实时检测机器人对抗赛中的装甲板，并解算其三维位姿（位置与姿态）。节点采用 YOLO 风格的 ONNX 模型进行检测，结合 PnP（Perspective-n-Point）算法计算距离与角度，并使用卡尔曼滤波器对目标进行跟踪。

**应用场景**
- 机器人对抗赛（RoboMaster、RoboCup 等）中的装甲板自动识别
- 移动机器人目标跟踪与定位
- 视觉伺服控制的前端感知模块

## 功能特性

- **装甲板检测**：使用预训练的 ONNX 模型（YOLO 架构）实时检测图像中的装甲板，支持多种类别（红/蓝方、数字编号）。
- **位姿解算**：通过 PnP 算法结合相机内参，计算装甲板相对于相机的三维平移与旋转向量。
- **卡尔曼滤波跟踪**：对检测到的装甲板进行状态估计，平滑运动轨迹，减少抖动。
- **畸变校正**：支持相机标定参数，自动对输入图像进行去畸变处理。
- **实时可视化**：在图像上绘制检测框、类别标签、距离信息，并发布渲染后的图像话题。
- **参数可配置**：支持通过 ROS 2 参数动态调整相机焦距、话题名称、置信度阈值等。

## 依赖安装

### 1. ROS 2 环境
确保已安装 ROS 2（推荐 Humble 或 Foxy）。若未安装，请参考 [ROS 2 官方文档](https://docs.ros.org/) 进行安装。

### 2. 系统依赖
```bash
sudo apt update
sudo apt install libopencv-dev libonnxruntime-dev
```

### 3. ROS 2 包依赖
以下依赖通常通过 `ros-<distro>-<package>` 安装，若已安装 ROS 2 基础包，则可能已满足。如需手动安装：
```bash
sudo apt install ros-${ROS_DISTRO}-rclcpp ros-${ROS_DISTRO}-std-msgs ros-${ROS_DISTRO}-sensor-msgs ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-image-transport ros-${ROS_DISTRO}-ament-index-cpp
```

### 4. ONNX Runtime
若系统包管理器未提供 `libonnxruntime-dev`，可从 [ONNX Runtime 官方 GitHub](https://github.com/microsoft/onnxruntime) 下载预编译库，或使用 vcpkg/pip 安装。确保头文件位于 `/usr/local/include`，库文件位于 `/usr/local/lib`。

## 构建步骤

1. **创建工作空间**
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   ```

2. **克隆本仓库**
   ```bash
   git clone <repository-url> robotmaster_vision
   cd robotmaster_vision
   ```

3. **安装项目依赖**
   ```bash
   sudo apt update
   rosdep install --from-paths . --ignore-src -y
   ```

4. **使用 colcon 构建**
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select robotmaster_vision
   ```

5. **激活工作空间**
   ```bash
   source install/setup.bash
   ```

## 使用方法

### 1. 运行节点
```bash
ros2 run robotmaster_vision armor_detector_node
```

### 2. 发布相机图像
节点默认订阅 `/camera/image`（sensor_msgs/Image）和 `/camera/camera_info`（sensor_msgs/CameraInfo）话题。请确保有相机驱动节点发布相应话题，或使用 bag 文件模拟。

### 3. 查看检测结果
- **可视化图像**：节点会发布 `/armor_detection/image`（sensor_msgs/Image）话题，可使用 `rqt_image_view` 或 RViz 查看。
- **检测信息**：节点会在终端输出检测到的装甲板类别、置信度、距离等信息。

### 4. 参数配置
节点启动时可配置以下参数（通过 `ros2 run` 或 launch 文件）：
| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `image_topic` | string | `/camera/image` | 输入图像话题 |
| `camera_info_topic` | string | `/camera/camera_info` | 相机内参话题 |
| `fx` | float | `600.0` | 相机焦距（x方向），若无相机内参则使用此值 |
| `fy` | float | `600.0` | 相机焦距（y方向），若无相机内参则使用此值 |
| `confidence_threshold` | float | `0.5` | 检测置信度阈值 |
| `nms_threshold` | float | `0.4` | 非极大抑制阈值 |

示例：启动时指定话题名称
```bash
ros2 run robotmaster_vision armor_detector_node --ros-args -p image_topic:=/usb_cam/image_raw
```

## 文件结构

```
robotmaster_vision/
├── CMakeLists.txt          # CMake 构建配置
├── package.xml             # ROS 2 包定义
├── README.md               # 本文档
├── include/robotmaster_vision/
│   ├── armor_detector_node.h   # 节点类声明
│   ├── geometry_utils.h        # 几何计算工具
│   ├── kalman_tracker.h        # 卡尔曼滤波器初始化
│   └── pnp_solver.h            # PnP 解算器
├── src/
│   ├── armor_detector_node.cpp # 节点主实现
│   ├── geometry_utils.cpp
│   ├── kalman_tracker.cpp
│   └── pnp_solver.cpp
├── models/
│   └── best.onnx               # 预训练检测模型
└── templates/
    ├── 1.png                   # 模板图像（用于匹配）
    ├── 2.png
    ├── 3.png
    └── 4.png
```

## 模型与模板

- **模型文件**：`models/best.onnx` 为基于 YOLO 架构训练的装甲板检测模型，输入尺寸为 `1×3×640×640`，输出为 `1×25200×?`（具体维度请参考训练配置）。若需更换模型，请确保输入输出形状与代码中 `run_yolo_inference` 函数兼容。
- **模板图像**：`templates/` 目录下的 PNG 文件用于模板匹配（当前版本未使用，保留供后续扩展）。

## 参数说明（详细）

### 相机参数
- 若提供了 `camera_info` 话题，节点将自动使用其中的内参矩阵和畸变系数进行校正。
- 若未提供，则使用 `fx`、`fy` 参数作为焦距，并假设主点位于图像中心。

### 检测参数
- `confidence_threshold`：过滤低置信度检测框。
- `nms_threshold`：控制重叠框的合并强度。

### 跟踪参数
卡尔曼滤波器的过程噪声与测量噪声已在 `kalman_tracker.cpp` 中硬编码，如需调整请直接修改源码。

## 许可证

本项目基于 Apache License 2.0 开源，详情请参阅 [LICENSE](LICENSE) 文件（若未提供，可补充）。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目仓库 Issues
- 维护者邮箱：your_email@example.com

---

*本项目为 RoboMaster 参赛队内部开发，欢迎 fork 与贡献！*