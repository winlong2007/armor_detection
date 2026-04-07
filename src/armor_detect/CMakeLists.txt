cmake_minimum_required(VERSION 3.8)
project(robotmaster_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ament_index_cpp REQUIRED)

# 查找 onnxruntime 库
find_library(ONNXRUNTIME_LIB onnxruntime PATHS /usr/local/lib REQUIRED)

# 包含头文件目录
include_directories(include)
include_directories(/usr/local/include)   # onnxruntime 头文件路径

# 列出源文件
set(SOURCES
  src/armor_detector_node.cpp
  src/geometry_utils.cpp
  src/kalman_tracker.cpp
  src/pnp_solver.cpp
)

# 添加可执行文件
add_executable(armor_detector_node ${SOURCES})

# 链接依赖库
ament_target_dependencies(armor_detector_node
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  OpenCV
  ament_index_cpp
)

# 链接 onnxruntime
target_link_libraries(armor_detector_node ${ONNXRUNTIME_LIB})

# 安装
install(TARGETS armor_detector_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_package()