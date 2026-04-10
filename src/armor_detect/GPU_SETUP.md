# 🚀 GPU 加速配置指南 (RTX 5060 & CUDA 12)

本项目支持使用 NVIDIA GPU 进行深度学习推理加速。以下是针对 **Ubuntu 22.04 + RTX 5060 Laptop** 的完整环境配置流程。

## 1. 硬件与驱动要求
*   **GPU**: NVIDIA GeForce RTX 5060 (Laptop) 或更高。
*   **Driver**: 建议版本 $\ge$ 525 (本项目测试版本为 580.126)。
    *   检查指令: `nvidia-smi`

## 2. 安装 CUDA 12 运行时
由于 ONNX Runtime 1.18+ 依赖 CUDA 12，需安装核心计算库：

```bash
# 添加 NVIDIA 官方仓库配置
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 安装 CUDA 12 核心组件 (无需安装几GB的完整版 Toolkit)
sudo apt install -y cuda-cudart-12-0 cuda-nvrtc-12-0 libcublas-12-0 libcurand-12-0
```

## 3. 安装 cuDNN 8.9.7 (关键)
**注意**：尽管系统可能支持 cuDNN 9，但 **ONNX Runtime 1.18.0** 预编译版目前强依赖 **cuDNN 8**。

1.  **下载**: 从 [NVIDIA 官网](https://developer.nvidia.com/rdp/cudnn-archive) 下载 `cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz`。
2.  **解压并安装**:
    ```bash
    tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
    sudo cp cudnn-linux-x86_64-8.9.7.29_cuda12-archive/include/* /usr/local/include/
    sudo cp -P cudnn-linux-x86_64-8.9.7.29_cuda12-archive/lib/* /usr/local/lib/
    sudo ldconfig
    ```

## 4. 安装 ONNX Runtime GPU 版
1.  **下载**: `onnxruntime-linux-x64-gpu-1.18.0.tgz`。
2.  **安装**:
    ```bash
    tar -zxvf onnxruntime-linux-x64-gpu-1.18.0.tgz
    sudo cp -r onnxruntime-linux-x64-gpu-1.18.0/include/* /usr/local/include/
    sudo cp -r onnxruntime-linux-x64-gpu-1.18.0/lib/* /usr/local/lib/
    sudo ldconfig
    ```

## 5. 代码实现 (C++)
在 `armor_detector_node.cpp` 的构造函数中，需显式启用 CUDA Provider：

```cpp
Ort::SessionOptions session_options;

// 开启 CUDA 加速
try {
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0; 
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    RCLCPP_INFO(this->get_logger(), "成功启用 ONNX Runtime CUDA 加速");
} catch (const std::exception& e) {
    RCLCPP_WARN(this->get_logger(), "未能启用 CUDA，回退至 CPU: %s", e.what());
}

// 加载模型
ort_session_ = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
```

## 6. 编译与运行
确保在编译时能够链接到 GPU 版库文件：

```bash
cd ~/ros2_ws
# 彻底清理旧编译产物
rm -rf build/robotmaster_vision install/robotmaster_vision

# 编译
colcon build --packages-select robotmaster_vision --cmake-args -DONNXRUNTIME_LIB=/usr/local/lib/libonnxruntime.so

# 运行
source install/setup.bash
ros2 run robotmaster_vision armor_detector_node
```

## 7. 验证 GPU 占用
节点运行后，在另一个终端执行：
```bash
nvidia-smi
```
如果看到 `armor_detector_node` 进程出现在显存占用列表中，即表示配置成功。
