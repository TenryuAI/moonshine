# Moonshine ARM 优化任务清单

## 1. 文档目标

这份文档是对 `ARM改进分析.md` 中 **P0 优化项** 的进一步拆解，目标不是再解释“为什么要优化”，而是直接回答下面几个问题：

- 具体要改哪几个任务
- 每个任务改哪些文件
- 每个任务建议怎么改
- 每个任务怎么验收
- 每个任务有哪些回归风险

这份清单默认面向 **嵌入式 ARM / 弱算力 ARM64 设备**，例如：

- Raspberry Pi
- ARM Linux 边缘设备
- Android arm64 中低端设备

## 2. P0 总体目标

P0 的核心不是“换模型”，而是先清理主链中的低效数据路径。

P0 总目标：

1. 降低热路径上的无意义内存复制
2. 降低缓冲搬移和小块处理的额外 CPU 开销
3. 降低语言绑定层在 ARM 上的边界成本
4. 在不改变外部行为的前提下，让实时链路更稳定

P0 不追求一次性解决所有 ARM 问题，优先解决这些“投入小、收益快、风险可控”的点。

## 3. P0 任务总览

| 任务 ID | 任务名称 | 主要文件 | 目标 |
| --- | --- | --- | --- |
| `P0-1` | VAD 缓冲去 `erase` 化 | `core/voice-activity-detector.h`、`core/voice-activity-detector.cpp` | 消除头部搬移 |
| `P0-2` | 重采样同采样率直通 | `core/resampler.h`、`core/resampler.cpp`、`core/transcriber.cpp`、`core/voice-activity-detector.cpp` | 消除无意义重采样复制 |
| `P0-3` | 精简 stream 入流路径复制 | `core/transcriber.cpp` | 减少 `audio -> vector -> resampled -> buffer` 串联复制 |
| `P0-4` | 精简 VAD 输入路径复制 | `core/voice-activity-detector.cpp` | 减少进入 VAD 前的临时 buffer 构造 |
| `P0-5` | Android JNI 热路径降噪 | `android/moonshine-jni/moonshine-jni.cpp` | 降低 ARM Android 上日志 I/O 干扰 |
| `P0-6` | Android JNI 数组生命周期修正 | `android/moonshine-jni/moonshine-jni.cpp` | 降低 JNI 边界风险与额外开销 |
| `P0-7` | Python 输入边界轻量化预留 | `python/src/moonshine_voice/transcriber.py` | 为 ARM 上连续 buffer 快路径打基础 |

## 4. 任务 `P0-1`：VAD 缓冲去 `erase` 化

### 4.1 当前问题

当前 `core/voice-activity-detector.cpp` 中：

- `process_audio()` 会先构造 `processing_buffer`
- 每处理一个 hop，就执行一次头部 `erase`

这意味着在长流式音频下，buffer 会不断前移，形成持续的内存搬移。

关键位置：

- `core/voice-activity-detector.cpp`

当前逻辑特征：

- `processing_buffer.insert(...)`
- `while (...)`
- `processing_buffer.erase(begin, begin + hop_size)`

### 4.2 建议改法

推荐优先级：

1. 第一选择：改成“起始索引 + 尾部追加”的滑动窗口
2. 第二选择：改成简单环形缓冲

对于 P0，不一定要一步到位做最复杂的 ring buffer；**先把 `erase` 去掉** 就已经很有价值。

建议实施方式：

1. 把 `processing_buffer` 逻辑改为：
   - 保留一份连续存储
   - 用 `processing_start_index` 标记当前消费位置
2. 每次 `process_audio_chunk()` 只移动索引，不真实擦除
3. 在循环结束后，只把未消费 remainder 拷回一个紧凑缓冲

### 4.3 影响文件

- `core/voice-activity-detector.h`
- `core/voice-activity-detector.cpp`

### 4.4 预计收益

- 长语音输入下 CPU 更稳定
- 降低内存搬移
- 减少尾延迟尖刺

### 4.5 验收标准

功能验收：

1. VAD 切段结果与修改前保持一致或仅有极小可解释差异
2. `LineStarted` / `LineCompleted` 顺序不变
3. `stop()` 后最后一句仍能正确收尾

性能验收：

1. 长流输入下 `process_audio()` 平均耗时下降
2. ARM 设备上 CPU 占用峰值下降或更平稳
3. 连续 1 到 3 分钟输入时无明显性能劣化

### 4.6 回归风险

- remainder 处理不正确导致吞样本或重复样本
- `just_updated` 触发时机变化
- 长句段边界漂移

## 5. 任务 `P0-2`：重采样同采样率直通

### 5.1 当前问题

`core/resampler.cpp` 当前在采样率一致时：

```cpp
if (input_sample_rate == output_sample_rate) {
  return audio;
}
```

虽然逻辑上是“原样返回”，但接口是按值返回 `std::vector<float>`，因此依然会产生复制语义。

### 5.2 建议改法

P0 推荐的保守做法是：

1. 不急着大改整个 resampler API
2. 先在调用侧增加“是否需要重采样”的判断
3. 在采样率一致时，避免先走 `resample_audio()` 再接收一个新 vector

更激进的做法可以留到后续：

- 把 `resample_audio()` 改成输出参数风格
- 或提供“借用输入 buffer”的零拷贝接口

### 5.3 影响文件

- `core/resampler.h`
- `core/resampler.cpp`
- `core/transcriber.cpp`
- `core/voice-activity-detector.cpp`

### 5.4 预计收益

- 16kHz 原生输入路径更轻
- 对麦克风天然就是 16kHz 的 ARM 设备收益最明显
- 降低最常见路径上的无意义分配与复制

### 5.5 验收标准

功能验收：

1. 16kHz 输入下转写结果不变
2. 非 16kHz 输入下转写结果与修改前一致

性能验收：

1. 16kHz 输入路径下 `add_audio_to_stream()` 或 VAD 输入准备耗时下降
2. 16kHz 场景下内存分配次数减少

### 5.6 回归风险

- 调用侧分支不一致导致某些路径漏掉重采样
- 接口调整影响现有调用点

## 6. 任务 `P0-3`：精简 stream 入流路径复制

### 6.1 当前问题

`TranscriberStream::add_to_new_audio_buffer()` 当前路径是：

1. `audio_data` 转 `std::vector<float> audio_vector`
2. 调 `resample_audio(audio_vector, ...)`
3. 把 `resampled_audio` `insert` 到 `new_audio_buffer`

这会形成典型串联复制。

关键位置：

- `core/transcriber.cpp`

### 6.2 建议改法

建议分两步做：

1. 先合并“输入转 vector”和“重采样返回”的中间态
2. 如果采样率相同，直接把原始输入追加到 `new_audio_buffer`

更具体一点：

- 同采样率：直接 `insert(audio_data, audio_data + audio_length)`
- 不同采样率：才构造临时向量并重采样

### 6.3 影响文件

- `core/transcriber.cpp`

### 6.4 预计收益

- 降低实时音频输入路径的额外 CPU 成本
- 降低内存带宽占用
- 对高频小 chunk 输入更友好

### 6.5 验收标准

功能验收：

1. 默认 stream 结果与修改前一致
2. 多 stream 场景行为不变

性能验收：

1. `add_to_new_audio_buffer()` 平均耗时下降
2. 高频送块时 CPU 抖动降低

### 6.6 回归风险

- 原始输入不是 float32 连续内存时误用
- 非 16kHz 路径遗漏重采样

## 7. 任务 `P0-4`：精简 VAD 输入路径复制

### 7.1 当前问题

`VoiceActivityDetector::process_audio()` 当前流程包括：

1. `audio_data` 构造 `input_audio_vector`
2. `resample_audio(input_audio_vector, ...)`
3. 构造 `processing_buffer`
4. 插入 remainder 和新音频

这意味着 VAD 入口本身就有多次拷贝。

### 7.2 建议改法

建议和 `P0-2` 联动：

1. 同采样率时减少 `input_audio_vector` 的不必要持有
2. 把 remainder 合并策略和新输入拼接策略尽量做成单次构造
3. 配合 `P0-1` 去掉 `erase`

### 7.3 影响文件

- `core/voice-activity-detector.cpp`

### 7.4 预计收益

- VAD 前置成本下降
- 让 ASR 主链拿到更多 CPU 预算

### 7.5 验收标准

功能验收：

1. segment 起止时间基本稳定
2. 短句和长句的切段结果与之前一致

性能验收：

1. `VoiceActivityDetector::process_audio()` 耗时下降
2. 长时间输入下无额外性能劣化

### 7.6 回归风险

- VAD 行为看似“只改性能”，实则会因边界处理变化导致切段变化

## 8. 任务 `P0-5`：Android JNI 热路径降噪

### 8.1 当前问题

`android/moonshine-jni/moonshine-jni.cpp` 的 `moonshineTranscribeStream()` 热路径里存在多条 `LOGE`：

- start transcribe stream
- transcription error
- transcript 指针打印

这类日志在 ARM Android 设备上会带来同步 I/O 开销。

### 8.2 建议改法

P0 最保守做法：

1. 默认关闭这些热路径日志
2. 或改成仅在 debug 宏 / 可配置开关下输出

### 8.3 影响文件

- `android/moonshine-jni/moonshine-jni.cpp`

### 8.4 预计收益

- 降低 JNI 热路径干扰
- 减少实时转写过程中不必要的日志成本

### 8.5 验收标准

功能验收：

1. Android 转写行为完全不变
2. 出错路径仍保留必要日志

性能验收：

1. Android 实时转写时日志量明显下降
2. 高频 `transcribeStream` 场景下更平稳

### 8.6 回归风险

- 调试可见性下降

处理方式：

- 用 debug 开关保留可选日志

## 9. 任务 `P0-6`：Android JNI 数组生命周期修正

### 9.1 当前问题

`moonshineAddAudioToStream()` 中使用了：

- `GetFloatArrayElements(audio_data, nullptr)`

但从当前读取范围看，需要非常明确地确认释放和生命周期管理是否完整、是否使用了最适合的访问方式。

### 9.2 建议改法

1. 明确补齐 `ReleaseFloatArrayElements()` 的释放路径
2. 保证异常路径也能正确释放
3. 视情况评估是否改成 `GetPrimitiveArrayCritical` 或 `DirectBuffer` 路线，但这一步可以不放进 P0 首次提交

### 9.3 影响文件

- `android/moonshine-jni/moonshine-jni.cpp`

### 9.4 预计收益

- 降低潜在 JNI 生命周期风险
- 降低 GC 干扰的概率

### 9.5 验收标准

功能验收：

1. Android 音频输入链功能不变
2. 长时间运行无明显 JNI 资源问题

稳定性验收：

1. 连续长时间录音转写无异常
2. 不出现 JNI 相关崩溃或资源警告

### 9.6 回归风险

- 释放模式不正确反而引入数据不可用问题

## 10. 任务 `P0-7`：Python 输入边界轻量化预留

### 10.1 当前问题

Python 目前最常见的输入方式会触发：

- Python list / 序列
- `ctypes` 数组构造
- 完整边界拷贝

这对 Raspberry Pi 一类 Python 主路径尤其不友好。

### 10.2 建议改法

P0 不一定强行上完整零拷贝，但建议至少先做两件事：

1. 增加对 `numpy.ndarray(dtype=float32, contiguous)` 的快路径预留
2. 在文档中明确 ARM 上推荐使用连续 `float32` 数组输入

### 10.3 影响文件

- `python/src/moonshine_voice/transcriber.py`
- 相关文档文件

### 10.4 预计收益

- 降低 Python 路径额外开销
- 为后续更深的边界优化打基础

### 10.5 验收标准

功能验收：

1. 现有 Python list 用法不受影响
2. `numpy.float32` 连续数组输入可正常工作

性能验收：

1. Raspberry Pi 路径下输入边界时间降低

### 10.6 回归风险

- 新输入分支和旧分支行为不一致

## 11. 推荐实施顺序

不建议 7 个任务一起上。推荐顺序如下：

1. `P0-5` Android 热路径降噪
2. `P0-6` Android JNI 生命周期修正
3. `P0-2` 重采样同采样率直通
4. `P0-3` stream 入流复制精简
5. `P0-4` VAD 输入复制精简
6. `P0-1` VAD 缓冲去 `erase` 化
7. `P0-7` Python 连续 buffer 快路径预留

这个顺序的考虑是：

- 先做风险最低、收益立刻可见的项
- 再做调用链较小的复制优化
- 最后做 VAD 缓冲结构调整这种更容易影响行为的改动

## 12. 建议提交顺序与每个提交的改动范围

建议不要把全部 P0 压成一个提交，而是拆成 4 个明确批次。  
这样做的目的不是为了“好看”，而是为了让每次提交都满足下面三个条件：

- 改动范围单一
- 回归来源容易定位
- 每个提交都能独立验证收益

### 提交 1：Android JNI 热路径收敛

#### 提交 1 对应任务

- `P0-5`
- `P0-6`

#### 提交 1 建议改动范围

只改：

- `android/moonshine-jni/moonshine-jni.cpp`

尽量不要在这个提交里同时改：

- `core/`
- `python/`
- `README.md`

#### 提交 1 要完成什么

1. 去掉或降级 `moonshineTranscribeStream()` 热路径里的高频 `LOGE`
2. 明确 `GetFloatArrayElements()` 的释放路径
3. 保证异常路径下也不会遗留 JNI 资源问题

#### 提交 1 为什么要最先做

因为它：

- 改动面最小
- 风险最低
- 对 ARM Android 设备是立刻见效的
- 不会影响转写主链算法行为

#### 提交 1 完成后的最小验证集

1. Android 示例能正常实时转写
2. JNI 路径不崩溃
3. 日志量明显下降
4. 连续录音一段时间后无异常

#### 提交 1 推荐提交信息方向

- `reduce jni hot-path logging on android`
- `fix jni audio array lifecycle handling`

### 提交 2：重采样直通与 stream 入流轻量化

#### 提交 2 对应任务

- `P0-2`
- `P0-3`

#### 提交 2 建议改动范围

主要改：

- `core/resampler.h`
- `core/resampler.cpp`
- `core/transcriber.cpp`

这一批次里可以顺带小改：

- `core/transcriber.h`，仅当需要声明辅助函数或补充注释时

尽量不要在这个提交里同时改：

- `core/voice-activity-detector.cpp`
- `python/`
- `android/`

#### 提交 2 要完成什么

1. 为“输入采样率等于目标采样率”建立直通策略
2. 简化 `TranscriberStream::add_to_new_audio_buffer()` 的中间态复制
3. 让 16kHz 输入路径尽量少分配、少复制

#### 提交 2 为什么放第二个

因为它仍然属于：

- 热路径优化
- 但还没有动 VAD 状态机

因此收益明显，同时行为风险相对可控。

#### 提交 2 完成后的最小验证集

1. 16kHz 输入路径结果与修改前一致
2. 非 16kHz 输入路径结果与修改前一致
3. `add_audio_to_stream()` 平均耗时下降
4. 默认实时转写功能无回归

#### 提交 2 推荐提交信息方向

- `avoid unnecessary resampling copies on matching sample rates`
- `reduce stream input buffering copies`

### 提交 3：VAD 输入路径和缓冲重构

#### 提交 3 对应任务

- `P0-4`
- `P0-1`

#### 提交 3 建议改动范围

主要改：

- `core/voice-activity-detector.h`
- `core/voice-activity-detector.cpp`

必要时联动：

- `core/transcriber.cpp`

但建议控制在“仅为了配合 VAD 输入方式变化而改”，不要顺便改别的逻辑。

#### 提交 3 要完成什么

1. 精简进入 VAD 前的复制路径
2. 去掉或替代 `processing_buffer.erase(...)`
3. 保持 `VoiceActivitySegment` 语义和事件顺序不变

#### 提交 3 为什么放第三个

因为这是 P0 中**最容易引起行为回归**的一组改动：

- 会影响切段边界
- 会影响 `just_updated`
- 会影响最后一句收尾

所以必须放在前两批更稳的改动之后。

#### 提交 3 完成后的最小验证集

1. 短句切段基本一致
2. 长句切段基本一致
3. `LineStarted` / `LineCompleted` 顺序不变
4. `stop()` 后最后一句仍能完成
5. `VoiceActivityDetector::process_audio()` 平均耗时下降

#### 提交 3 推荐提交信息方向

- `reduce vad input copying`
- `replace erase-based vad buffering with sliding window logic`

### 提交 4：Python ARM 输入快路径预留

#### 提交 4 对应任务

- `P0-7`

#### 提交 4 建议改动范围

主要改：

- `python/src/moonshine_voice/transcriber.py`

可选联动：

- Python 文档
- `README.md` 中 ARM/树莓派建议

尽量不要在这个提交里同时改：

- `core/`
- `android/`

#### 提交 4 要完成什么

1. 为 `numpy.ndarray(dtype=float32, contiguous)` 建立更清晰的快路径
2. 保留现有 Python list 行为兼容
3. 在文档中明确 ARM 上推荐的输入形式

#### 提交 4 为什么放最后

因为它不属于核心 C++ 主链，但对 Raspberry Pi Python 路线很重要。放最后的好处是：

- 前面 C++ 主链已经更稳定
- Python 侧优化收益更容易单独观测

#### 提交 4 完成后的最小验证集

1. 现有 list 输入不回归
2. `numpy.float32` 连续数组输入正常
3. Raspberry Pi 路线下边界开销下降

#### 提交 4 推荐提交信息方向

- `add contiguous numpy fast path for python audio input`
- `document arm-friendly python audio input format`

### 为什么不建议把这些合成一个大提交

因为它们涉及的风险类型完全不同：

- 提交 1 是 JNI / 日志风险
- 提交 2 是入流复制和重采样风险
- 提交 3 是 VAD 行为风险
- 提交 4 是 Python 绑定兼容性风险

如果合并成一个提交，后面即使性能变好了，也很难判断到底是哪一项生效，出现回归也很难快速止损。

### 一份最推荐的实际执行顺序

如果你准备真的开始开发，推荐严格按这个顺序推进：

1. 提交 1：先改 Android JNI
2. 提交 2：再改 resampler + stream 入流
3. 提交 3：最后再动 VAD 缓冲
4. 提交 4：补 Python ARM 快路径

这个顺序兼顾了：

- 风险由低到高
- 收益由快到慢
- 回归定位由易到难

## 12.5 四个提交的函数级实施 checklist

这一节把前面的“提交级拆分”继续压到函数级，目的是让后续真正进入编码时可以直接按函数逐项勾选。

### 提交 1 的函数级 checklist：Android JNI 热路径收敛

#### 提交 2 涉及文件

- `android/moonshine-jni/moonshine-jni.cpp`

#### 重点函数 1：`Java_ai_moonshine_voice_JNI_moonshineAddAudioToStream`

当前要解决的问题：

- `GetFloatArrayElements()` 获取后的生命周期不清晰
- 当前函数直接 `return moonshine_transcribe_add_audio_to_stream(...)`
- 这样会让释放路径不够明确

实施 checklist：

1. 先把 `GetFloatArrayElements()` 返回值保存到局部变量
2. 先把 `GetArrayLength()` 结果保存到局部变量
3. 调用 `moonshine_transcribe_add_audio_to_stream(...)` 时不要直接 `return`
4. 把返回值先保存到局部 `int result`
5. 在正常路径里调用 `ReleaseFloatArrayElements(audio_data, audio_data_ptr, JNI_ABORT)` 或等价释放策略
6. 再 `return result`
7. 确保异常路径下也不会遗漏释放

这个函数本提交里不要顺手做的事：

- 不要改 JNI 方法签名
- 不要改 Java 层接口
- 不要在这里同时引入 `DirectBuffer`

#### 重点函数 2：`Java_ai_moonshine_voice_JNI_moonshineTranscribeStream`

当前要解决的问题：

- 热路径里有多条 `LOGE`
- 这些日志不是错误级别的必要日志
- 在 ARM Android 设备上会形成同步 I/O 干扰

实施 checklist：

1. 去掉 `start transcribe stream` 这类热路径日志
2. 去掉成功路径上的 transcript 指针打印
3. 保留真正错误分支的错误日志
4. 如果需要保留调试能力，改成 debug 宏或可开关日志

这个函数本提交里不要顺手做的事：

- 不要改 `c_transcript_to_jobject()` 的对象构造逻辑
- 不要改 transcript 返回结构

#### 提交 1 完成后需要人工复查的函数

- `Java_ai_moonshine_voice_JNI_moonshineStartStream`
- `Java_ai_moonshine_voice_JNI_moonshineStopStream`

复查目的：

- 确认 JNI 生命周期修改没有破坏上下文调用顺序

### 提交 2 的函数级 checklist：重采样直通与 stream 入流轻量化

#### 提交 2 涉及文件

- `core/resampler.h`
- `core/resampler.cpp`
- `core/transcriber.cpp`

#### 重点函数 1：`resample_audio`

当前要解决的问题：

- 同采样率时仍走按值返回 `std::vector<float>`
- 从语义上是“原样返回”，从实现上仍可能发生复制

实施 checklist：

1. 明确本提交是否改接口
2. 如果不改接口：
   - 保持 `resample_audio()` 存在
   - 主要在调用方避免不必要调用
3. 如果要小改接口：
   - 只增加最小辅助函数
   - 不一次性重做整个 resampler 架构

建议优先方案：

1. 保守处理：调用侧先判断采样率是否相同
2. 同采样率时不进入 `resample_audio()`

这个函数本提交里不要顺手做的事：

- 不要引入 NEON 优化
- 不要重写 downsample / upsample 算法

#### 重点函数 2：`TranscriberStream::add_to_new_audio_buffer`

当前要解决的问题：

- 先构造 `audio_vector`
- 再构造 `resampled_audio`
- 再追加到 `new_audio_buffer`

实施 checklist：

1. 保留 `save_audio_data_to_wav()` 行为不变
2. 增加“采样率已等于 `INTERNAL_SAMPLE_RATE`”的分支
3. 在同采样率分支中，直接把 `audio_data` 追加到 `new_audio_buffer`
4. 仅在采样率不同的分支里构造临时 `std::vector<float>`
5. 保持返回行为与调用方无差异

建议本提交不要顺手改的函数：

- `VoiceActivityDetector::process_audio`
- `Transcriber::transcribe_stream`

原因：

- 提交 2 应只解决 stream 入流与 resampler 路径，不混入 VAD 行为风险

#### 提交 2 完成后需要人工复查的函数

- `resample_audio`
- `downsample_audio`
- `upsample_audio`
- `TranscriberStream::clear_new_audio_buffer`

复查目的：

- 确认同采样率快路径没有绕过必要逻辑

### 提交 3 的函数级 checklist：VAD 输入路径和缓冲重构

#### 提交 3 涉及文件

- `core/voice-activity-detector.h`
- `core/voice-activity-detector.cpp`

必要时少量联动：

- `core/transcriber.cpp`

#### 重点函数 1：`VoiceActivityDetector::process_audio`

当前要解决的问题：

- 输入先转 `input_audio_vector`
- 再 `resample_audio`
- 再构造 `processing_buffer`
- 再循环 `erase`

实施 checklist：

1. 先保持函数外部行为完全不变
2. 去掉 `processing_buffer.erase(...)` 依赖
3. 改成：
   - 滑动起始索引
   - 或环形缓冲
4. 循环内只推进消费位置，不真实搬移头部
5. 循环结束后再整理 remainder
6. 确保 `processing_remainder_audio_buffer` 语义不变

这个函数本提交里不要顺手做的事：

- 不要改 VAD 判定阈值逻辑
- 不要改 `process_audio_chunk()` 的核心语义
- 不要改 `on_voice_start()` / `on_voice_end()` 触发规则

#### 重点函数 2：`VoiceActivityDetector::process_audio_chunk`

当前目标不是重写它，而是确认它在新缓冲策略下仍然工作正常。

实施 checklist：

1. 保持输入仍然是单个 `hop_size` 块
2. 确认 `samples_processed_count` 仍正确累积
3. 确认 `look_behind_audio_buffer` 行为不变
4. 确认 `previous_is_voice` 状态机不变

#### 重点函数 3：`VoiceActivityDetector::start` / `stop`

如果提交 3 引入了新缓冲字段或索引变量，必须同步检查：

1. `start()` 是否完整重置新状态
2. `stop()` 是否仍能让最后一句自然完成
3. remainder 和活动段状态是否都被正确清空或收尾

#### 提交 3 完成后需要人工复查的函数

- `on_voice_start`
- `on_voice_continuing`
- `on_voice_end`

复查目的：

- 确保 segment 的开始、持续和结束语义没有被缓冲重构破坏

### 提交 4 的函数级 checklist：Python ARM 输入快路径预留

#### 提交 4 涉及文件

- `python/src/moonshine_voice/transcriber.py`

可选联动文档：

- `README.md`
- ARM 相关文档

#### 重点函数 1：`Stream.add_audio`

当前要解决的问题：

- 始终走 `(ctypes.c_float * len(audio_data))(*audio_data)`
- 无法利用 `numpy.ndarray(float32 contiguous)` 的天然连续内存

实施 checklist：

1. 先增加输入类型分支判断
2. 对 `numpy.ndarray` 快路径，要求：
   - `dtype == float32`
   - contiguous
3. 对不满足条件的 numpy 输入，先安全降级到现有路径
4. 对普通 Python list 保持原有行为
5. 不要让新路径影响 `self._stream_time` 逻辑

建议实现方式：

1. 抽一个私有 helper，例如 `_coerce_audio_buffer(...)`
2. `Stream.add_audio()` 与后续可能的离线路径共用

#### 重点函数 2：`Transcriber.transcribe_without_streaming`

虽然本提交的主要目标是实时路径，但如果已经引入统一 helper，建议顺手评估这里是否也能受益。

实施 checklist：

1. 先确认 helper 是否可以复用
2. 如果复用成本低，则同步接入
3. 如果复用会扩大风险，则本提交先不动

#### 提交 4 完成后需要人工复查的函数

- `Stream.add_audio`
- `Transcriber.transcribe_without_streaming`

复查目的：

- 确认 numpy 快路径和 list 慢路径行为一致

### 四个提交都通用的“不要顺手改”原则

为了保证这 4 个提交都能独立验证，下面这些事情建议明确不要混进去：

1. 不要在提交 1 里改 C++ 核心逻辑
2. 不要在提交 2 里顺手改 VAD 状态机
3. 不要在提交 3 里顺手调参数默认值
4. 不要在提交 4 里顺手改事件系统
5. 不要把文档大改和代码大改混进同一个提交

## 13. 每个提交建议附带的最小验证清单

为了直接进入实施前准备，下面把每个提交要跑的最小验证也定下来。

### 提交 1 验证

- Android 示例可正常转写
- JNI 不崩溃
- 高频日志明显减少
- 长时间运行无 JNI 资源异常

### 提交 2 验证

- 16kHz 输入与修改前结果一致
- 非 16kHz 输入与修改前结果一致
- `add_audio_to_stream()` 或入流路径耗时下降

### 提交 3 验证

- VAD 切段结果基本稳定
- `LineStarted` / `LineCompleted` 顺序不变
- `stop()` 后最后一句能完整结束
- `process_audio()` 耗时下降

### 提交 4 验证

- Python list 输入不回归
- `numpy.float32` 连续数组可正常工作
- ARM Python 路线输入边界开销下降

## 14. 建议的通用验收方式

每个任务都建议至少做三层验收。

### 14.1 功能一致性

检查：

- 文本结果是否明显变化
- `LineStarted` / `LineCompleted` 语义是否一致
- `stop()` 后最后一句是否仍能完成

### 14.2 性能

建议记录：

- `add_audio()` 平均耗时
- `transcribe_stream()` 平均耗时
- CPU 峰值
- CPU 平均值
- 连续运行 1 到 3 分钟的尾延迟

### 14.3 设备实测

至少建议覆盖：

1. Raspberry Pi Python 实时麦克风路径
2. Android arm64 JNI 实时路径
3. 16kHz 原生输入和非 16kHz 输入各一组

## 15. 一份最实用的执行建议

如果现在就要真的开始动手，我最建议从下面这个最小闭环开始：

1. 先做 `P0-5`
2. 再做 `P0-6`
3. 再做 `P0-2`
4. 最后用 Raspberry Pi 和 Android 各跑一次短实测

原因是：

- 这几项改动面最小
- 最容易快速看到收益
- 不太会破坏主链结构

## 16. 实施前准备完成定义

如果要判断“这份任务清单是否已经可以进入实施”，建议看下面 5 条是否都满足：

1. 每个提交只对应一类风险
2. 每个提交有明确的文件边界
3. 每个提交有最小验证清单
4. 已明确哪些地方本批次不要碰
5. 已接受“先低风险、后高风险”的推进顺序

满足这 5 条，就可以直接进入编码阶段。

## 17. 总结

P0 的本质不是“做大优化”，而是把现在最明显的低效路径清干净。

真正最该先做的事只有两类：

- 把热路径里的无意义搬运降下来
- 把绑定层里容易放大为 ARM 问题的开销降下来

只要这两类问题先解决，Moonshine 在 ARM / 嵌入式场景下的整体体验通常就会明显更健康，后面再做更深层的线程、后端或 NNAPI/XNNPACK 评估才更有意义。
