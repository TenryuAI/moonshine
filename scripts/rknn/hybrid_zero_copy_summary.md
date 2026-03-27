# Silero VAD RKNN 异构调度与 Zero-Copy 性能总结

## 1. 架构设计
为了解决 RK3568 NPU 运行 LSTM 算子时产生的累积精度误差，我们将 `silero_vad_k2` 模型切分为三段：
1. **Frontend (CPU)**: 预处理与特征提取 (`Unsqueeze`, `Cast`, `Pad`) -> 输出 `[1, 258, 8]`
2. **Midend (NPU)**: 核心 1D 卷积簇，发挥 NPU 算力优势 -> 输出 `[1, 64, 1]`
3. **Backend (CPU)**: LSTM 解码器与最终分类，保证状态累积的 FP32 精度。

## 2. Zero-Copy 实现 (C++ 层)
在 C++ 层，我们利用了 RKNN C API 的零拷贝机制 (`rknn_create_mem` 和 `rknn_set_io_mem`)，彻底消除了 CPU 与 NPU 之间的张量拷贝开销：
- **Frontend -> Midend**: ONNXRuntime 的输出直接写入由 NPU 驱动分配的 `rknn_input_mem->virt_addr` 内存块。
- **Midend -> Backend**: ONNXRuntime 从 NPU 驱动分配的 `rknn_output_mem->virt_addr` 内存块直接读取输入。
- **数据格式**: 由于 NPU 默认需要 FP16 输入/输出，我们在 C++ 中手动实现了极简的 FP32 <-> FP16 内存位转换，直接在 NPU 共享内存上操作。

## 3. 性能对比 (RK3568, 1387 个音频窗口)

| 运行模式 | 语言/框架 | 耗时 (Elapsed) | CPU 负载率 (Load) | 精度 (Segments) |
| :--- | :--- | :--- | :--- | :--- |
| **纯 CPU 基线** | C++ (ONNXRuntime) | 3.42s | 7.71% | 8 (旧版) / 14 (k2版) |
| **纯 NPU (精度错误)** | Python (RKNNLite) | 18.45s | 41.58% | 3 (严重漂移) |
| **混合调度 (带拷贝)** | Python (ORT + RKNNLite) | 5.70s | 12.86% | 14 (精度恢复) |
| **混合调度 (Zero-Copy)** | C++ (ORT + RKNN C API) | **4.62s** | **10.40%** | **14 (精度恢复)** |

## 4. 结论
1. **精度完全恢复**：将 LSTM 移回 CPU 彻底解决了 NPU 的累积误差问题，`segments` 数量与纯 CPU 模式完全对齐。
2. **Zero-Copy 显著降低开销**：相比于 Python 层的混合调度，C++ 层的 Zero-Copy 将耗时从 5.70s 降至 4.62s，性能提升约 **19%**。
3. **VAD 模型的 NPU 收益瓶颈**：尽管采用了 Zero-Copy，混合调度的耗时 (4.62s) 依然略慢于纯 CPU (3.42s)。这是因为 Silero VAD 模型极小，单次推理仅需几百微秒。在这种极细粒度的频繁调用（每秒 31 次）下，**NPU 的任务提交延迟、中断响应以及跨设备同步开销（`rknn_mem_sync`，约几百微秒）已经超过了 NPU 在卷积计算上节省的时间**。

**最终建议**：
对于像 Silero VAD 这样极小且调用频繁的模型，在 RK3568 这类算力级别的芯片上，**纯 CPU 运行是最佳选择**（耗时最短，负载仅 7.71%）。NPU 更适合用于处理单次推理计算量大、调用频率相对较低的模型（如 ASR 的声学模型或图像检测模型）。
