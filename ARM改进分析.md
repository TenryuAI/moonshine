# Moonshine ARM 改进分析

## 1. 结论先行

如果只用一句话概括当前结论：

**Moonshine 已经具备“可在部分 ARM 平台构建、发布、运行”的基础能力，但还没有形成一套面向嵌入式 ARM 设备的系统化优化方案；当前最大的改进空间不在模型精度本身，而在运行时数据搬移、缓冲策略、构建发布一致性，以及面向弱设备的参数化与执行路径设计。**

从优先级上看，最值得做的不是一上来就重训模型，而是先把下面几类问题解决：

1. 热路径里的多次内存拷贝、重复重采样和缓冲搬移
2. ARM 构建/发布流程和文档不一致的问题
3. Python/JNI 等绑定层在弱设备上的额外开销
4. 面向 ARM 的线程、日志、附加能力开关没有形成明确建议
5. 嵌入式部署文档不够完整，导致“能支持”但“不够好落地”

## 2. 当前 ARM / 嵌入式支持现状

从仓库现状看，Moonshine 并不是“完全没有 ARM 支持”，相反，它已经覆盖了多个 ARM 目标：

- Raspberry Pi / Linux aarch64
- Apple Silicon
- iOS arm64
- Android `arm64-v8a`
- Windows ARM64

但这些支持更接近：

- 能识别平台
- 能选对 ONNX Runtime 库
- 能构建出对应产物
- 文档里有基本的运行路径

而不是：

- 有明确的嵌入式性能预算
- 有针对 ARM 的主链优化
- 有统一的 ARM 调优建议
- 有完整公开的可重复构建链路

也就是说，当前项目在 ARM 方面更像“**已支持**”，但还谈不上“**已针对嵌入式 ARM 深度优化**”。

## 3. 当前已有的 ARM 能力

### 3.1 构建与产物层面

仓库已经在脚本层明确处理了多个 ARM 目标：

- `scripts/build-pip.sh` 会处理 Linux aarch64 / 树莓派
- `scripts/build-pip.bat` 处理 Windows ARM64
- `core/third-party/onnxruntime/find-ort-library-path.cmake` 会按 `aarch64`、`arm64-v8a`、Apple arm64 等路径选择 ORT 库
- `scripts/publish-binary.sh` 区分 `linux-arm64` 与 `rpi-arm64`
- `build.gradle.kts` 中 Android 目前明确只打 `arm64-v8a`

这说明 Moonshine 至少已经把“ARM 平台识别”和“对应 ORT 产物选择”做进了工程层。

### 3.2 文档与示例层面

文档中也存在 ARM 相关入口：

- `README.md` 有 Raspberry Pi 小节
- `examples/raspberry-pi/my-dalek/README.md` 给出了树莓派示例
- Python `pip` 路线是当前最清晰的 ARM 落地路径

这说明官方推荐的 ARM 使用路径，本质上是：

**优先通过 Python 包拿模型和运行时，再在 ARM 目标机上直接运行。**

### 3.3 运行时层面

从核心实现看，ARM 路径目前主要依赖：

- ONNX Runtime CPU 推理
- 模型量化
- streaming 模型减少重复计算
- VAD / speaker / intent 等模块统一复用核心流程

但是仓库内没有明显看到：

- 自研 NEON 内核
- ARM 专用 SIMD 路径
- 专门的低功耗模式
- 针对弱设备分层裁剪的运行时 profile

这意味着当前的 ARM 表现，主要来自模型设计和 ORT 本身，而不是项目在 C++ 层做了很多 ARM 定制优化。

## 4. 当前明显的短板

### 4.1 支持是“可运行”，不是“深度优化”

当前最明显的问题不是“不支持 ARM”，而是“支持了，但没针对 ARM 资源约束做充分优化”。

体现在：

- 热路径里存在多次 `vector` 拷贝
- 部分路径即使采样率相同也会复制音频数据
- VAD 缓冲处理方式对弱 CPU 不够友好
- 绑定层的数据搬运成本在 ARM 上更敏感
- 文档没有给出清晰的 ARM 调参建议

### 4.2 构建发布链不够公开、稳定、可复现

从脚本看，项目对 ARM 的发布能力部分依赖：

- 远端 Linux 主机
- 远端 Raspberry Pi 主机
- 本地已有的 ORT 目录结构

这会带来两个问题：

1. 外部开发者不容易完整复现官方 ARM 构建链
2. 仓库脚本与文档之间存在“支持存在，但过程不够透明”的落差

### 4.3 文档更偏“能跑起来”，而不是“如何跑得更好”

README 对 Raspberry Pi 给出了运行方法，但没有系统回答：

- 最低推荐硬件规格是什么
- 哪些模型更适合弱 ARM
- 哪些参数最影响 ARM 上的体验
- Pi 和通用 Linux aarch64 有什么区别
- 什么时候应该关掉 speaker / word timestamps

对于嵌入式开发者来说，这类信息往往比“怎么 pip install”更重要。

## 5. ARM 上最敏感的热点路径

下面这些地方在桌面端也会影响性能，但在 ARM 边缘设备上更敏感。

### 5.1 `core/transcriber.cpp` 是第一热点

最关键的核心路径仍然是：

- `Transcriber::transcribe_stream()`
- `Transcriber::update_transcript_from_segments()`
- `Transcriber::transcribe_segment_with_streaming_model()`

这里是主链的重计算中心：

- 触发 VAD 分析
- 处理 segment
- 执行流式编码和解码
- 更新 transcript
- 生成附加信息

对于 ARM 来说，这里之所以敏感，是因为它集中了：

- CPU 计算
- 频繁的小块循环
- 锁保护区域
- 文本生成与状态更新

如果这里有不必要的重复处理，ARM 设备的尾延迟会非常明显。

### 5.2 VAD 路径存在额外搬移成本

`core/voice-activity-detector.cpp` 的 `process_audio()` 和 `process_audio_chunk()` 是另一条重要热点。

当前的实现特点包括：

- 把输入先放进 `std::vector`
- 进入 `resample_audio`
- 用 `processing_buffer` 做块级处理
- 逐 hop 调用 VAD
- 对缓冲区头部做 `erase`

这套方案在逻辑上清晰，但在 ARM 上会更容易放大几个问题：

- 多次内存复制
- 小块循环的调度成本
- `erase` 带来的搬移成本
- 缓存命中率不佳

其中最值得注意的是：**头部 `erase` 对长缓冲会造成持续搬移，这在弱 CPU 上非常不划算。**

### 5.3 `resampler.cpp` 对 ARM 不够友好

`core/resampler.cpp` 暴露出一个很典型的可优化点：

- 即使输入和输出采样率相同，也会 `return audio`，产生拷贝返回
- 降采样和升采样都使用简单循环
- 当前没有看到明显的 SIMD / NEON 优化

在 ARM 上，这意味着：

- 每一轮音频输入都可能多走一次内存复制
- 采样率不匹配时，重采样的成本会更明显

如果目标设备本来就能直接提供 16kHz mono PCM，那么最好尽量把“避免重采样”作为一条一等优化策略。

### 5.4 `silero-vad.cpp` 是持续的推理开销点

Silero VAD 本身是合理选择，但当前实现也有几个对 ARM 更敏感的点：

- 它有独立的 ORT 环境和 session
- 每次 `predict()` 都要准备输入张量
- 当前默认单线程
- 与主 ASR 同时存在时，会共同占用 CPU 和内存带宽

对于高端设备这不是大问题，但在嵌入式 ARM 上，**VAD 不只是“很轻的前置判断”，而是一个持续不断发生的推理负担**。

### 5.5 绑定层开销在 ARM 上占比更高

#### Python

`python/src/moonshine_voice/transcriber.py` 在 `add_audio()` 等路径里会把 Python 数据转成 `ctypes` 数组。

这意味着：

- 每次调用都可能发生完整数据拷贝
- 在弱 ARM CPU 上，这种“非模型推理时间”的占比会上升

#### Android JNI

`android/moonshine-jni/moonshine-jni.cpp` 里：

- `GetFloatArrayElements()` 的使用值得特别关注
- `moonshineTranscribeStream()` 中存在热路径 `LOGE`

这两个问题在 ARM 手机上或嵌入式 Android 设备上都很敏感：

- 数组 pinning / 拷贝可能干扰 GC
- 高频日志会增加同步 I/O 负担

#### Swift

Swift 路径相对更轻，但仍然属于需要和核心路径一起评估的数据边界层。

## 6. 为什么这些问题在嵌入式 ARM 上更严重

### 6.1 不是“算得慢一点”，而是“系统余量更小”

ARM 嵌入式设备和桌面 CPU 的本质差异在于：

- 频率更低
- 缓存更小
- 内存带宽更紧
- 功耗预算更严格
- 多数场景没有大算力富余

所以在桌面端只是“还能接受”的额外拷贝，在 ARM 上可能会直接变成：

- UI 不跟手
- 文本刷新变慢
- 功耗升高
- 设备发热
- 最终语音体验变差

### 6.2 ARM 更怕“无意义的搬运”

Moonshine 当前最值得改进的一类问题，并不是“没有用上最强推理后端”，而是：

- 音频被复制了太多次
- 缓冲搬移太频繁
- 数据在边界层来回折腾

这类问题在 ARM 上通常比桌面端更先暴露出来。

### 6.3 ARM 更怕热路径日志和锁竞争

在嵌入式设备上：

- `stderr` / `LOGE` 日志更容易显著拖慢热路径
- 全局锁或大粒度锁更容易造成尾延迟尖刺

因此任何出现在热路径中的日志和锁，都值得作为一类专门优化对象看待。

## 7. 最值得优先改进的方向

下面按价值和投入比来排。

## 8. P0：先减少热路径的数据搬移和重复处理

这是最值得最先做的事。

### 8.1 原因

这类问题有几个特点：

- 不依赖模型重训
- 不依赖新平台能力
- 往往不改变业务接口
- 对 ARM 上的延迟和功耗都有直接帮助

### 8.2 重点位置

- `core/transcriber.cpp`
- `core/voice-activity-detector.cpp`
- `core/resampler.cpp`

### 8.3 最值得做的改动方向

- 避免同采样率下的无意义重采样复制
- 减少 `audio_data -> vector -> resampled_vector -> buffer` 这种串联复制
- 用环形缓冲或索引偏移替代频繁 `erase`
- 审视 `TranscriptStreamOutput` 中是否存在可减少的中间搬移
- 评估 `return_audio_data` 在默认路径中的必要性

### 8.4 预期收益

- 降低 CPU 占用
- 降低内存带宽压力
- 减少尾延迟
- 对 ARM 的收益通常明显大于桌面端

## 9. P1：完善 ARM 构建与发布链路的一致性

### 9.1 当前问题

脚本已经覆盖 ARM，但存在几个现实问题：

- 外部开发者不容易完整复现
- `build-pip-docker.sh` 虽然构建了 `linux/arm64` 镜像，但公开链路仍需进一步核对和解释
- `publish-binary.sh` 对不同平台的上传行为不完全一致
- 文档没有把 ARM 产物和构建过程讲透

### 9.2 为什么这是 P1

因为如果构建和发布路径不清晰，就会导致：

- 别人难以贡献 ARM 优化
- 很难建立稳定测试矩阵
- 优化做了也不容易持续交付

### 9.3 推荐改进方向

- 增加公开可重复的 aarch64 构建说明
- 把 Pi 和通用 Linux aarch64 的差异讲清楚
- 统一脚本与文档里的平台命名、产物说明和上传策略
- 明确哪些 ORT 依赖是仓库自带，哪些需要额外获取

## 10. P1：降低绑定层在弱设备上的额外成本

### 10.1 原因

在高性能设备上，绑定层开销可能只是小头；但在 ARM 嵌入式设备上，小头也可能变大。

### 10.2 Python 路径

建议优先评估：

- 是否能减少 `ctypes` 数组构造拷贝
- 是否能优先支持更连续、更接近 native buffer 的输入路径

### 10.3 Android JNI 路径

建议优先评估：

- `GetFloatArrayElements()` 的释放与生命周期是否完全正确
- 是否可以降低 `moonshineTranscribeStream()` 热路径日志
- 是否值得考虑 `DirectBuffer` 风格输入

### 10.4 为什么值回票价

因为这些优化通常不需要动模型本身，却能直接减少：

- 语言边界拷贝
- GC 干扰
- 日志 I/O

## 11. P2：补一套明确的 ARM 运行配置建议

### 11.1 当前问题

Moonshine 已经有不少可调参数，但对 ARM 场景没有形成明确 profile。

例如：

- `transcription_interval`
- `vad_threshold`
- `vad_window_duration`
- `identify_speakers`
- `word_timestamps`
- `return_audio_data`

### 11.2 建议方向

至少给出几套官方推荐配置：

- `armRealtimeLowLatency`
- `armCommandRecognition`
- `armMeetingTranscription`
- `armLowPower`

即便短期不做成代码 profile，先做成文档建议也很有价值。

### 11.3 价值

这能显著降低用户在 ARM 上的试错成本。

## 12. P2：评估 ARM 专项执行路径

这是中长期方向。

### 12.1 可考虑的方向

- NNAPI
- XNNPACK
- ORT 在 ARM 上更激进的线程/执行策略
- 更严格的生产环境日志级别分层

### 12.2 为什么不是 P0 / P1

因为这类工作：

- 实施成本更高
- 验证矩阵更复杂
- 风险也更高

在当前阶段，先把显而易见的拷贝和缓冲问题解决，收益通常更确定。

### 12.3 当前仓库里真实存在的 P2 入口

如果只看代码现状，而不是只看理想方向，当前仓库里与 `P2` 最相关的事实有这些：

- **主模型和流式模型仍然只走 ORT CPU 路径**  
  目前没有看到项目代码里显式调用 `SessionOptionsAppendExecutionProvider...` 或通用 `SessionOptionsAppendExecutionProvider(...)` 去挂接 NNAPI、XNNPACK、CoreML 等执行后端。

- **仓库自带的 ONNX Runtime 头文件已经具备能力声明**  
  `core/third-party/onnxruntime/include/onnxruntime_c_api.h` 明确声明了通用 provider API；  
  `core/third-party/onnxruntime/include/nnapi_provider_factory.h` 也存在，说明“从 API 能力上看”项目所在 ORT 版本具备 NNAPI 入口。

- **XNNPACK 在当前仓库里没有现成接入代码**  
  头文件目录里没有独立的 `xnnpack_provider_factory.h`，但 ORT 通用 provider API 文档提到了 `XNNPACK`。这说明若要接入，更可能走：
  - 通用 `SessionOptionsAppendExecutionProvider(...)`
  - provider name 方式

- **线程策略目前不统一**
  - `core/silero-vad.cpp`：显式把 `SetIntraOpNumThreads` 和 `SetInterOpNumThreads` 都设为 `1`
  - `core/gemma-embedding-model.cpp`：显式把 `SetIntraOpNumThreads` 设为 `1`
  - `core/moonshine-model.cpp`：没有显式设置线程数，但设置了 `ORT_ENABLE_EXTENDED`、`DisableCpuMemArena`、`session.disable_prepacking`
  - `core/moonshine-streaming-model.cpp`：启用了 `ORT_ENABLE_ALL`，但没有显式线程配置

- **Android 当前平台面很收敛**
  `build.gradle.kts` 里目前只保留了 `arm64-v8a`，这意味着如果后续做 NNAPI 或 Android ARM 侧 provider 试验，验证面相对可控，不需要同时兼顾多个 ABI。

这些事实很重要，因为它们说明：

1. `P2` 不是从零开始
2. 但也远没有到“只差开个开关”的程度
3. 当前最现实的 `P2` 形态，是**先做线程策略与 provider 接入预研，再决定是否进入真实集成**

### 12.4 P2 的几条可行路线对比

下面按“实现难度 / 风险 / 潜在收益”来做一个更实际的对比。

| 路线 | 当前仓库支撑度 | 预期收益 | 风险 | 建议优先级 |
| --- | --- | --- | --- | --- |
| ORT 线程策略调优 | 高 | 中到高 | 中 | 最高 |
| NNAPI 试验接入 | 中 | Android ARM 上可能较高 | 高 | 中 |
| XNNPACK 试验接入 | 中偏低 | CPU 场景可能有帮助 | 高 | 中偏后 |
| 更细的日志/运行档位分层 | 高 | 中 | 低 | 高 |
| 直接大规模更换执行后端 | 低 | 不确定 | 很高 | 最低 |

当前仓库里，NNAPI feasibility spike 的最小入口也已经具备：

- `TranscriberOptions` 已可接收：
  - `ort_use_nnapi`
  - `ort_nnapi_use_fp16`
  - `ort_nnapi_cpu_disabled`
- 这些参数已经能进入：
  - `MoonshineModel`
  - `MoonshineStreamingModel`
- 当前实现只在 **Android** 编译路径下生效
- 当前实现只影响**主转写模型会话**
  - 不影响 VAD
  - 不影响 speaker embedding
  - 不影响 Gemma embedding

这意味着现在已经可以做一个非常保守的 Android provider 试验，而不需要先改默认主链行为。

当前 Android 侧还有一个现实阻断需要明确记录：

- 本地代码已经具备 NNAPI 最小实验入口
- Android instrumentation test 也已经补好
- 但在当前这台开发机上：
  - 没有可用的 `adb`
  - 没有已连接 Android 设备
  - `./gradlew :android:compileDebugAndroidTestJavaWithJavac` 目前还会卡在 Android Gradle Plugin 解析阶段

也就是说，当前已经完成的是：

- **代码接入完成**
- **Android 测试入口完成**
- **本地静态审查完成**

但还没有完成：

- **Android 设备上的实际 NNAPI provider 可用性验证**

这不是代码逻辑阻断，而是本地 Android 运行环境阻断。

#### 路线一：ORT 线程策略调优

这是最适合先做的 `P2` 路线。

原因：

- 不需要先引入新的 provider
- 只是在现有 ORT CPU 路径上做更合理的线程策略
- 可以先从最小实验开始，不破坏主链

优先关注文件：

- `core/moonshine-model.cpp`
- `core/moonshine-streaming-model.cpp`
- `core/silero-vad.cpp`
- `core/gemma-embedding-model.cpp`

优先关注问题：

- 为什么 VAD / Gemma embedding 固定为 1 线程，而主模型没有统一策略
- 主模型和流式模型是否应该允许：
  - 单线程
  - 小核友好
  - “低延迟优先” 与 “吞吐优先” 两种模式

#### 路线二：NNAPI 试验接入

这条路线更偏 Android ARM。

当前有利条件：

- 仓库里已经有 `nnapi_provider_factory.h`
- Android 当前只打 `arm64-v8a`

主要风险：

- 真正可用与否取决于当前打包的 ORT 是否带对应 provider
- 就算 API 在头文件里存在，也不代表当前分发的二进制具备完整支持
- 不同 Android 设备 / SoC / 驱动差异很大，回归面会变宽

所以更合理的做法不是直接接入，而是：

1. 先确认当前 ORT 二进制是否支持 NNAPI provider
2. 再做一个最小 Android 分支实验
3. 最后才考虑是否进入主线

#### 路线三：XNNPACK 试验接入

这条路线理论上对 CPU 场景更有吸引力，但当前仓库的显式支撑度比 NNAPI 还弱一点。

原因：

- 仓库没有独立的 XNNPACK provider factory 头文件
- 只能看到 ORT 的通用 provider API 文档中提到了 `XNNPACK`

因此这条路线更适合放在：

- ORT 线程调优之后
- 并且在明确 ORT 二进制具备相关能力之后

### 12.5 P2 最推荐的实施顺序

如果真的进入 `P2`，我建议按下面顺序推进，而不是直接去改 Android provider：

1. **先做 ORT 线程策略预研**
2. **再做更细的运行 profile / 日志档位**
3. **然后做 Android NNAPI 最小试验**
4. **最后才考虑 XNNPACK 或更深 provider 路线**

这样做的原因是：

- 线程策略更容易复用到所有 ARM 平台
- provider 接入一旦失败，问题定位通常更难
- 先把 CPU 路径调顺，才能判断 provider 的真实增益

### 12.6 P2 最小实验计划

如果下一步继续做 `P2`，建议不要直接改正式主路径，而是先做几组小实验。

#### 实验 A：主模型 / 流式模型线程数实验

目标：

- 比较 1 线程、2 线程、默认线程在 ARM 上的收益和抖动

建议最小改动点：

- `core/moonshine-model.cpp`
- `core/moonshine-streaming-model.cpp`

观察指标：

- `transcribe_stream()` 平均耗时
- `LineCompleted` 延迟
- CPU 峰值
- 长时间运行稳定性

#### 实验 B：VAD 与主模型的线程协同

目标：

- 确认 VAD 固定 1 线程是否仍是最优
- 评估 VAD 和主模型线程策略是否需要分离

建议最小改动点：

- `core/silero-vad.cpp`

观察指标：

- VAD 调用耗时
- 整体转写尾延迟
- 小设备上的抖动

#### 实验 C：Android NNAPI feasibility spike

目标：

- 验证当前 ORT 分发物是否真的能附加 NNAPI provider

建议最小改动点：

- Android 路径中新建一个实验分支
- 不直接改主线默认行为

成功标准：

- 能在 Android 上成功挂载 provider
- 能正常推理
- 相比 CPU 路径确实有收益

失败也有价值，因为它能明确告诉我们：

- 问题出在 ORT 构建
- 还是问题出在接入路径

### 12.7 P2 的前置条件

在正式投入 `P2` 前，建议先确认以下条件已经满足：

1. P0 的主链优化已经稳定
2. 至少有一台可重复测量的 ARM 目标设备
3. 已经定义好统一的验收指标
4. Android / Linux ARM 两条验证路径不要同时起步

否则很容易出现：

- 改了很多地方
- 但不知道收益来自哪里
- 也不知道回归来自哪里

### 12.8 P2 的最小实验入口已经具备

经过前面的改动，当前仓库已经具备一个很适合做 `P2` 第一阶段实验的入口：

- `TranscriberOptions` 已支持：
  - `ort_intra_op_threads`
  - `ort_inter_op_threads`
- 这些参数已经能从 `moonshine-c-api.cpp` 的 option 解析链路进入
- 主模型 `MoonshineModel`
- 流式模型 `MoonshineStreamingModel`
  都已经支持在显式传参时覆盖 ORT 线程数

这意味着现在可以直接做下列对比，而不需要再改默认主链：

1. 默认线程配置
2. `ort_intra_op_threads=1`
3. `ort_intra_op_threads=2`
4. `ort_intra_op_threads=1,ort_inter_op_threads=1`

当前最小实验入口有两个：

- **Python CLI**
  - `python -m moonshine_voice.transcriber --options="ort_intra_op_threads=1"`
- **程序化调用**
  - `Transcriber(..., options={"ort_intra_op_threads": "1"})`

这一步很重要，因为它把 `P2` 从“研究方向”变成了“现在就可以跑的实验路径”。

### 12.9 第一轮线程实验建议怎么做

建议先做一轮非常保守的线程实验，不要同时改太多参数。

#### 实验顺序

1. 默认配置
2. 仅 `ort_intra_op_threads=1`
3. 仅 `ort_intra_op_threads=2`
4. `ort_intra_op_threads=1,ort_inter_op_threads=1`

#### 保持不变的条件

实验时建议以下条件固定：

- 同一个模型
- 同一段测试音频
- 同一设备
- 同样的 `transcription_interval`
- 同样的 `vad_threshold`
- speaker / word timestamps 开关状态不变

#### 第一轮重点看什么

第一轮实验不必追求绝对性能极限，先看这几个问题：

1. 线程参数切换后是否稳定运行
2. `LineCompleted` 延迟是否明显变化
3. CPU 峰值是否下降或抖动是否减轻
4. 是否出现更明显的中间态抖动

如果连“稳定切换线程配置”都做不到，就还不应该进入 NNAPI/XNNPACK 方向。

### 12.10 第一轮线程 smoke test 结果

为了确认这条 `P2` 路线不只是“纸面可行”，当前仓库已经新增了一个最小实验目标：

- `core/ort-thread-benchmark.cpp`

它直接使用：

- `test-assets/tiny-en`
- `test-assets/two_cities.wav`

来跑主模型线程参数对比。

本轮第一批结果是在**当前开发机**上得到的，目的主要是：

- 验证线程参数链路是否生效
- 验证实验入口是否稳定
- 粗略观察“默认 / 单线程 / 双线程”的趋势

这**不是 ARM 实机结论**，但它足以证明后续 ARM 实验值得继续。

#### 本轮测试配置

- 模型：`tiny-en`
- wav：`two_cities.wav`
- `model_arch = MOONSHINE_MODEL_ARCH_TINY`
- 固定 `transcription_interval`
- 关闭 `identify_speakers`
- 关闭 `return_audio_data`

#### 第一轮结果

| 配置 | lines | avg latency | elapsed | load |
| --- | --- | --- | --- | --- |
| 默认线程 | 13 | 93ms | 7.85s | 17.69% |
| `ort_intra_op_threads=1, ort_inter_op_threads=1` | 13 | 84ms | 6.69s | 15.08% |
| `ort_intra_op_threads=1` | 13 | 85ms | 6.94s | 15.64% |
| `ort_intra_op_threads=2` | 13 | 67ms | 5.63s | 12.69% |

#### 从这批结果能得出的结论

1. 线程参数入口已经真实生效，而不是只停留在代码里
2. 默认线程配置并不天然最优
3. 在当前开发机上，`ort_intra_op_threads=2` 这一组明显优于默认配置
4. 这说明继续在 ARM 设备上做同样的 4 组对比是有价值的

#### 当前还不能直接下的结论

这批结果还不能直接推出：

- ARM 上 2 线程一定最好
- `inter_op_threads` 一定应该固定成 1
- 所有模型都适合同样的线程策略

因为这些都必须在真实目标设备上验证。

#### 这批 smoke test 的真正意义

这批结果最重要的意义不是“已经选出最优线程数”，而是：

- 线程调优路线是成立的
- 线程调优收益不是理论上的
- 下一步做 ARM 实机实验不会白费

所以接下来的合理动作，不是立刻把 `2` 线程写成默认值，而是：

1. 在真实 ARM 设备上复跑同样的 4 组
2. 保持其它变量不变
3. 根据设备结果决定是否要进一步做 profile 固化

### 12.11 第一轮 ARM 实机线程实验结果

在完成开发机构建与 smoke test 后，已经进一步通过 SSH 在一台真实 ARM Linux 设备上完成了部署、编译和最小线程实验。

本次远端实验环境要点：

- 设备：`rock@192.168.0.100`
- 架构：`aarch64`
- 系统：Linux 5.10 RK356x
- 原生 `cmake` 版本过低（3.18.4），已通过 `python3 -m pip install --user cmake` 在用户目录补到可用版本
- 已将最小实验所需的 `core/` 与 `test-assets/` 同步到远端
- 已在远端编译并运行：
  - `ort-thread-benchmark`

本轮 ARM 实机实验继续使用：

- 模型：`tiny-en`
- wav：`two_cities.wav`
- `model_arch = MOONSHINE_MODEL_ARCH_TINY`

#### ARM 实机结果

| 配置 | lines | avg latency | elapsed | load |
| --- | --- | --- | --- | --- |
| 默认线程 | 13 | 1152ms | 92.88s | 209.32% |
| `ort_intra_op_threads=1` | 13 | 2786ms | 216.84s | 488.67% |
| `ort_intra_op_threads=2` | 13 | 1646ms | 131.26s | 295.80% |
| `ort_intra_op_threads=1, ort_inter_op_threads=1` | 13 | 2837ms | 222.03s | 500.36% |

#### 这批 ARM 结果和开发机结果的差异

这批结果最重要的一点，是它和开发机 smoke test 的趋势**并不相同**：

- 在开发机上，`ort_intra_op_threads=2` 明显优于默认
- 在这台 ARM 实机上，**默认线程反而是四组里最好的**
- 强行把主模型压成 `1` 线程，性能明显变差
- `intra=1, inter=1` 这一组在 ARM 上也是最差的一档

这正好说明了一个关键结论：

**线程策略不能从开发机结果直接外推到 ARM 设备。**

也就是说：

- `P2` 的实验路线是对的
- 但 profile 固化必须以目标 ARM 设备的实测结果为准

#### 当前能得出的更可靠结论

1. `ort_intra_op_threads` / `ort_inter_op_threads` 的代码入口已经真实可用
2. 不同平台对线程配置的响应差异很大
3. ARM 实机上“少线程更好”并不是普遍规律
4. 对这台 RK356x 设备而言，当前默认线程配置优于本轮测试中的 1 线程和 2 线程实验值

#### 当前还不能下的结论

这轮结果还不能直接推出：

- 所有 ARM 设备默认线程都最好
- RK356x 上永远不需要调线程
- 其它模型（比如 streaming small / medium）也会出现同样趋势

因为这轮实验只覆盖了：

- 一个设备
- 一个模型
- 一段音频
- 少量线程组合

#### 关于 GPU discovery warning

远端实验日志里出现了 ORT 的 warning：

- `GPU device discovery failed`
- `/sys/class/drm/card0/device/vendor` 无法读取

目前看这更像是 ORT 在做设备探测时的非致命告警，而不是本次实验失败原因，因为：

- 所有 4 组实验都正常完成
- 都产出了稳定的结果
- 进程退出码为 `0`

但它提示我们后续如果继续推进 `P2`，应该额外留意：

- 是否需要抑制无用设备探测
- 是否存在更适合嵌入式设备的 ORT 配置

#### 这轮 ARM 实机实验的价值

这轮实验最重要的价值是：

- 证明远端 ARM 部署链已经打通
- 证明 `ort-thread-benchmark` 能在真实 ARM 设备上运行
- 证明线程策略实验必须在真实目标机上做，而不是只看开发机

所以如果后续继续做 `P2`，下一步最值得做的不是立刻上 NNAPI，而是：

1. 用同一台 ARM 设备继续扩充线程组合
2. 分 streaming / non-streaming 模型分别测
3. 再决定是否要把某些线程策略做成 profile 建议

### 12.12 第一轮 ARM 实机 streaming 模型线程实验结果

在完成 non-streaming tiny 模型实验后，又在同一台 ARM 设备上继续对 **`tiny-streaming-en`** 跑了同样的 4 组线程参数对比。

这一步的目的，是确认：

- streaming 模型是否会呈现不同的线程趋势
- 线程策略是不是应该按模型形态区分

本轮 streaming 实验环境保持不变：

- 同一台 ARM 设备
- 同一段 `two_cities.wav`
- `model_arch = MOONSHINE_MODEL_ARCH_TINY_STREAMING`

#### ARM 实机 streaming 结果

| 配置 | lines | avg latency | elapsed | load |
| --- | --- | --- | --- | --- |
| 默认线程 | 13 | 955ms | 74.97s | 168.94% |
| `ort_intra_op_threads=1` | 13 | 2274ms | 171.74s | 387.02% |
| `ort_intra_op_threads=2` | 13 | 1522ms | 114.58s | 258.22% |
| `ort_intra_op_threads=1, ort_inter_op_threads=1` | 13 | 2278ms | 171.97s | 387.55% |

扩展矩阵补充结果：

| 配置 | lines | avg latency | elapsed | load |
| --- | --- | --- | --- | --- |
| `ort_intra_op_threads=3` | 13 | 933ms | 72.88s | 164.23% |
| `ort_intra_op_threads=4` | 13 | 959ms | 75.39s | 169.89% |

#### 和 non-streaming ARM 结果对比后能看出的结论

1. **streaming 模型整体比 non-streaming 更快**
   - 默认线程下：
     - non-streaming：`92.88s`
     - streaming：`74.97s`

2. **线程趋势和 non-streaming 基本一致**
   - 默认线程依然最好
   - `intra=1` 依然明显更差
   - `intra=2` 仍然是介于两者之间

3. **这台 ARM 设备上，“强行压线程”对 streaming 也没有带来收益**

4. **在这台 4 核 Cortex-A55 设备上，streaming tiny 的当前最优点更接近 `intra=3`**

这一点比前一轮结论更具体：

- `intra=1`：明显过低，最差
- `intra=2`：有改善，但仍明显落后于最优
- `intra=3`：当前测试中最佳
- `intra=4`：已经接近默认，但没有继续优于 `intra=3`

#### 当前最重要的推论

经过 non-streaming 和 streaming 两轮 ARM 实机测试后，可以先得到一个比之前更强的阶段性判断：

**在这台 RK356x ARM 设备上，non-streaming tiny 的强基线仍是默认线程；而在 streaming tiny 上，扩展矩阵显示 `ort_intra_op_threads=3` 略优于默认线程。**

这仍然不是“所有 ARM 平台通用”的最终结论，但已经足够说明：

- 线程策略不能简单照搬开发机
- 线程策略最好按真实目标设备做 profile
- 默认策略至少在这台设备上是一个强基线

#### 对后续 `P2` 的实际影响

这批结果会影响后续路线判断：

1. **不要急着把 1 线程或 2 线程写进默认值**
2. streaming 路径在这台 ARM 机上，已经可以把 `intra=3` 作为下一轮重点候选
3. 继续保留线程参数作为实验入口，而不是立即固化为全局默认值
4. 如果后续继续实验，更值得扩充的是：
   - 不同 `transcription_interval`
   - 不同 streaming 模型尺寸
   - streaming / non-streaming 是否需要分开给 profile

#### 这批 streaming 实验的价值

这批结果让 `P2` 的线程调优判断从：

- “开发机上 2 线程看起来更好”

变成了：

- “真实 ARM 设备上，默认线程是强基线，但 streaming tiny 可能在 `intra=3` 上略优”

这对后续是否继续推进 NNAPI/XNNPACK 也有意义，因为它说明：

- 仅仅靠粗暴手动限线程，不一定能拿到更好结果
- 如果要继续深挖性能，下一步更可能需要：
  - 更细的线程矩阵
  - 更具体的 ORT 行为分析
  - 或转向 provider 级实验

## 12.13 根据反馈修订后的方案细化

结合你给出的补充意见，当前方案还可以进一步明确成“**问题 -> 代码落点 -> 建议改法 -> ARM 收益**”四段式。下面是更可执行的一版。

### A. 运行时热路径：优先把“搬运型开销”打掉

#### A1. VAD 缓冲从 `erase` 改成环形缓冲或索引窗口

当前问题已经非常明确：

- `core/voice-activity-detector.cpp` 中 `processing_buffer.erase(...)`
- 长时间流输入下会持续做头部搬移
- ARM 上这类线性搬运非常不划算

建议改法：

1. 用环形缓冲替代当前 `processing_buffer`
2. 或者至少改成“起始索引 + 有效长度”的滑动窗口
3. 让 `process_audio_chunk()` 读取逻辑不再依赖真实擦除

建议改动文件：

- `core/voice-activity-detector.h`
- `core/voice-activity-detector.cpp`

ARM 收益：

- 明显减少内存搬移
- 长语音输入下 CPU 更稳定
- 尾延迟更可控

实现风险：

- 需要重新检查 remainder 逻辑
- 要确保 `just_updated` / `is_complete` 的行为完全不变

#### A2. 重采样增加“同采样率直通”快速路径

当前问题：

- `core/resampler.cpp` 在 `input_sample_rate == output_sample_rate` 时仍然 `return audio`
- 这会返回一个新的 `std::vector<float>`，本质仍是复制

建议改法：

1. 给 `resample_audio()` 增加“零拷贝直通”语义
2. 最理想是把接口改成更明确的两层：
   - 一层判断是否需要重采样
   - 一层只在必要时真正分配新 buffer
3. 如果短期不改接口，至少在调用侧尽量避免“先构造 vector 再原样返回”

建议改动文件：

- `core/resampler.h`
- `core/resampler.cpp`
- `core/transcriber.cpp`
- `core/voice-activity-detector.cpp`

ARM 收益：

- 避免最常见场景下的无意义复制
- 如果设备原生提供 16kHz 音频，收益会非常直接

实现风险：

- 需要小心函数签名变化对调用方的影响

#### A3. 精简 `audio_data -> vector -> resampled_vector -> buffer` 串联复制

当前问题：

- 输入数据在进入 VAD 和 stream buffer 前经历多轮 `vector` 构造与追加
- 对桌面端问题不大，但 ARM 上会放大

建议改法：

1. 优先在 `TranscriberStream::add_to_new_audio_buffer()` 做简化
2. 把“原始输入转 vector”和“重采样结果拼接”这两步合并或减少
3. 评估 VAD 和 stream path 是否能共用某些中间 buffer

建议改动文件：

- `core/transcriber.cpp`
- `core/voice-activity-detector.cpp`

ARM 收益：

- 减少内存带宽压力
- 降低短块音频频繁送入时的额外 CPU 成本

### B. 绑定层：降低“非模型时间”占比

#### B1. Python 绑定减少每次 `ctypes` 全量展开的成本

当前问题：

- `python/src/moonshine_voice/transcriber.py` 每次 `add_audio()` 都走 `(ctypes.c_float * len(audio_data))(*audio_data)`
- 这在 ARM 上会让“边界拷贝”更明显

建议改法：

1. 优先支持 `numpy.ndarray(dtype=float32, contiguous)` 快路径
2. 在 Python 侧尽量接受连续 buffer，而不是一般 list
3. 在文档里明确推荐 ARM 上优先传 `numpy.float32` 连续数组

建议改动文件：

- `python/src/moonshine_voice/transcriber.py`
- `python/src/moonshine_voice/utils.py` 或相关输入辅助层
- 文档层同步说明

ARM 收益：

- 减少 Python 到 native 的边界成本
- 对 Raspberry Pi 这类 Python 路线尤其有价值

实现风险：

- 需要兼容老的 Python list 调用方式

#### B2. Android JNI 修正数组生命周期并去掉热路径日志

当前问题：

- `GetFloatArrayElements()` 的生命周期管理需要特别谨慎
- `moonshineTranscribeStream()` 热路径 `LOGE` 过于频繁

建议改法：

1. 明确补齐 `ReleaseFloatArrayElements()` 或改成更清晰的数组访问模式
2. 把热路径日志降级到 debug 开关下
3. 评估是否引入 `DirectBuffer` 风格输入

建议改动文件：

- `android/moonshine-jni/moonshine-jni.cpp`

ARM 收益：

- 降低 Java/native 边界扰动
- 减少同步日志 I/O
- 改善 Android ARM 设备上的实时稳定性

实现风险：

- 需要注意 JNI 数组释放方式与拷贝模式的正确性

### C. Silero VAD：从“可用”提升到“更适合 ARM”

当前问题：

- 独立 ORT session
- 每次都要构建输入张量
- 与主 ASR 共享有限 CPU / 内存带宽

建议改法：

1. 短期先不替换 VAD 模型
2. 优先增加 ARM 场景下的明确建议：
   - 何时调大 `vad_window_duration`
   - 何时降低更新频率
   - 何时关闭不必要附加能力，给 VAD/ASR 留出预算
3. 中期再评估是否要合并/复用某些 ORT 资源或调整 VAD 调用节奏

建议改动文件：

- `core/silero-vad.cpp`
- 文档层：`README.md`、ARM 专章

ARM 收益：

- 短期可先通过配置减轻负担
- 中期再看是否需要结构性调整

### D. 构建与发布：从“内部可跑”变成“外部可复现”

#### D1. 把 ARM 构建链文档化

当前问题：

- 现在更像维护者自己知道怎么构建
- 外部开发者不一定能一键复现

建议补充：

1. Pi 原生构建路径
2. 通用 Linux aarch64 构建路径
3. Docker / 容器路径
4. ORT 依赖目录说明
5. 常见失败原因和校验方法

建议改动文件：

- `README.md`
- 新增 ARM 专章或附录

#### D2. 统一脚本、产物名和上传行为

当前问题：

- 脚本覆盖了 ARM，但行为并不总是从文档里一眼可见
- 平台名、上传策略和公开产物说明还可以更统一

建议改法：

1. 统一 `linux-arm64` / `rpi-arm64` / `arm64-v8a` 的文档说明
2. 逐项写清脚本产物是什么、传到哪里
3. 把“哪些是官方支持路径”说清楚

建议改动文件：

- `scripts/build-pip.sh`
- `scripts/build-pip-docker.sh`
- `scripts/publish-binary.sh`
- `README.md`

### E. 文档与配置：把 ARM 专项 profile 做出来

这是你反馈里非常关键的一点，我认同应该单独加强。

建议至少形成 4 套官方建议配置：

| Profile | 目标 | 建议特征 |
| --- | --- | --- |
| `armRealtimeLowLatency` | 实时字幕 / 跟手交互 | 较小 `transcription_interval`，中等 VAD，关闭非必要附加能力 |
| `armCommandRecognition` | 命令识别 | 更稳的 VAD，重点保证 `LineCompleted` 后再判 intent |
| `armMeetingTranscription` | 多人会议记录 | 可开启 `identify_speakers` / `word_timestamps`，接受更高成本 |
| `armLowPower` | 低功耗待机 | 较大 `transcription_interval`，关闭 speaker / timestamps / 音频回传 |

建议落地位置：

- `README.md`
- `ARM改进分析.md`
- 未来可扩展到 `原理及工作流程.md`

## 12.14 建议新增一套“ARM 优化验收指标”

如果方案要从分析走向实施，最好补一组统一指标，否则后续优化容易变成主观判断。

建议至少跟踪：

- 平均 CPU 占用
- 峰值 CPU 占用
- 单次 `transcribe_stream()` 耗时
- 最终 `LineCompleted` 平均延迟
- 常驻内存
- 是否出现音频首字丢失或切段变碎

建议按场景分开测：

1. 树莓派实时麦克风转写
2. Android arm64 实时转写
3. 同一设备上 speaker / word timestamps 开关前后对比

这样后续每一项 ARM 优化都能有明确验收口径。

## 13. 文档层还有哪些可以明显改进

当前 ARM 文档最缺的不是“有没有安装命令”，而是“有没有决策信息”。

最建议新增的内容包括：

### 13.1 单独增加 ARM / 嵌入式章节

建议覆盖：

- 当前支持的平台矩阵
- 最低推荐硬件
- Pi 与通用 Linux aarch64 的区别
- 哪些模型更适合弱设备
- 哪些附加能力默认建议关闭

### 13.2 增加 ARM 调参建议

建议直接把这些写成表格：

- 目标：低延迟
- 目标：低功耗
- 目标：命令识别
- 目标：会议记录

### 13.3 增加 ARM 构建说明

建议说明：

- 本机构建方式
- Docker / 容器方式
- 依赖的 ORT 路径
- 常见构建失败原因

## 14. 最值得立刻落地的改进清单

如果按“投入小、收益快”的顺序排，我最推荐下面 8 个动作。

### 14.1 第一组：最快见效

1. 消除同采样率下的多余重采样复制
2. 重构 VAD 缓冲，避免频繁 `erase`
3. 降低 Android JNI 热路径日志
4. 明确 Python 输入拷贝成本，并评估更轻的输入方式

### 14.2 第二组：工程稳定性

1. 校正并文档化 ARM 构建/发布链路
2. 补一套 ARM 推荐参数配置说明

### 14.3 第三组：中期方向

1. 评估 ARM 执行后端与线程策略
2. 单独增加嵌入式 ARM 章节或独立文档

## 15. 如果要落成具体优化任务，建议怎么拆

后续如果你想真正进入改造阶段，我建议把 ARM 优化拆成 4 组任务：

### 15.1 任务组 A：运行时数据路径优化

范围：

- `core/transcriber.cpp`
- `core/voice-activity-detector.cpp`
- `core/resampler.cpp`

目标：

- 降低拷贝
- 降低搬移
- 降低无效 CPU 开销

### 15.2 任务组 B：绑定层轻量化

范围：

- `python/src/moonshine_voice/transcriber.py`
- `android/moonshine-jni/moonshine-jni.cpp`

目标：

- 降低跨语言边界成本
- 降低热路径日志和数组处理开销

### 15.3 任务组 C：工程和发布一致性

范围：

- `scripts/build-pip.sh`
- `scripts/build-pip-docker.sh`
- `scripts/publish-binary.sh`
- `README.md`

目标：

- 让 ARM 路线更可复现
- 让产物、脚本和文档一致

### 15.4 任务组 D：ARM 专章文档

范围：

- `README.md`
- `原理及工作流程.md`
- `二次开发指南.md`

目标：

- 让 ARM 用户知道怎么选模型、怎么调参、怎么排障

## 16. 最后的判断

如果你问“Moonshine 在 ARM/嵌入式场景下最该先改什么”，我的判断是：

**先改运行时数据路径，再补构建发布一致性，再补 ARM 文档和参数 profile，最后再考虑更重的推理后端优化。**

原因很简单：

- 前三类改动投入更小
- 见效更快
- 风险更可控
- 对所有 ARM 设备都有普遍收益

而更换执行后端、引入更重的 ARM 专项优化虽然也值得做，但应该放在“基础工程和热路径已经足够干净”之后。

## 17. 建议的后续动作

如果你下一步要我继续推进，我建议按下面顺序来：

1. 先出一版 ARM 优化任务拆解清单
2. 再从 `P0` 中挑 1 到 2 个最确定的点动手改
3. 同步补一版 ARM 文档章节
4. 最后再考虑是否要评估更深的 ORT / NNAPI / ARM 后端策略

如果只做分析而不马上改代码，那么这份文档已经足够作为后续 ARM 优化的决策基础。
