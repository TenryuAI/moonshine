/**
 * RKNN Hybrid Benchmark for Silero VAD.
 *
 * This benchmark tests the performance of a hybrid CPU-NPU-CPU pipeline using
 * ONNXRuntime for the CPU frontend/backend and RKNN C API for the NPU midend.
 * It utilizes RKNN's zero-copy memory API (`rknn_create_mem` / `rknn_set_io_mem`)
 * to avoid unnecessary tensor copying between CPU and NPU.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "moonshine-utils/debug-utils.h"
#include "onnxruntime_c_api.h"
#include "rknn_api.h"

namespace {

struct AudioFile {
  std::vector<float> data;
  int32_t sample_rate = 16000;
};

AudioFile load_wav_or_throw(const std::string &wav_path) {
  float *wav_data = nullptr;
  size_t wav_data_size = 0;
  int32_t wav_sample_rate = 0;
  if (!load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                     &wav_sample_rate)) {
    throw std::runtime_error("Failed to load WAV file: " + wav_path);
  }

  AudioFile result;
  result.data.assign(wav_data, wav_data + wav_data_size);
  result.sample_rate = wav_sample_rate;
  free(wav_data);
  return result;
}

void print_usage() {
  std::fprintf(
      stderr,
      "Usage: rknn-hybrid-benchmark --frontend <onnx> --midend <rknn> "
      "--backend <onnx> --wav-path <wav> [--threshold <float>]\n");
}

void check_status(OrtStatus* status, const OrtApi* ort) {
  if (status != nullptr) {
    const char* msg = ort->GetErrorMessage(status);
    std::fprintf(stderr, "ONNXRuntime Error: %s\n", msg);
    ort->ReleaseStatus(status);
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string frontend_path;
  std::string midend_path;
  std::string backend_path;
  std::string wav_path;
  float threshold = 0.5f;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--frontend" && i + 1 < argc) {
      frontend_path = argv[++i];
    } else if (arg == "--midend" && i + 1 < argc) {
      midend_path = argv[++i];
    } else if (arg == "--backend" && i + 1 < argc) {
      backend_path = argv[++i];
    } else if (arg == "--wav-path" && i + 1 < argc) {
      wav_path = argv[++i];
    } else if (arg == "--threshold" && i + 1 < argc) {
      threshold = std::stof(argv[++i]);
    } else {
      print_usage();
      return 1;
    }
  }

  if (frontend_path.empty() || midend_path.empty() || backend_path.empty() ||
      wav_path.empty()) {
    print_usage();
    return 1;
  }

  // 1. Load Audio
  AudioFile wav = load_wav_or_throw(wav_path);
  const int WINDOW_SAMPLES = 512;
  std::vector<std::vector<float>> windows;
  for (size_t i = 0; i < wav.data.size(); i += WINDOW_SAMPLES) {
    std::vector<float> chunk(WINDOW_SAMPLES, 0.0f);
    size_t copy_size = std::min<size_t>(WINDOW_SAMPLES, wav.data.size() - i);
    std::memcpy(chunk.data(), wav.data.data() + i, copy_size * sizeof(float));
    windows.push_back(std::move(chunk));
  }

  // 2. Initialize ONNXRuntime (Frontend & Backend) using C API
  const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtEnv* env = nullptr;
  check_status(ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "rknn-hybrid", &env), ort);
  
  OrtSessionOptions* session_options = nullptr;
  check_status(ort->CreateSessionOptions(&session_options), ort);
  check_status(ort->SetIntraOpNumThreads(session_options, 1), ort);
  check_status(ort->SetInterOpNumThreads(session_options, 1), ort);

  OrtSession* frontend_sess = nullptr;
  OrtSession* backend_sess = nullptr;
  check_status(ort->CreateSession(env, frontend_path.c_str(), session_options, &frontend_sess), ort);
  check_status(ort->CreateSession(env, backend_path.c_str(), session_options, &backend_sess), ort);

  OrtMemoryInfo* memory_info = nullptr;
  check_status(ort->CreateCpuMemoryInfo(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU, &memory_info), ort);

  // 3. Initialize RKNN (Midend)
  rknn_context rknn_ctx = 0;
  int ret = rknn_init(&rknn_ctx, const_cast<char *>(midend_path.c_str()), 0, 0,
                      nullptr);
  if (ret < 0) {
    std::fprintf(stderr, "rknn_init failed: %d\n", ret);
    return 1;
  }

  rknn_sdk_version version;
  ret = rknn_query(rknn_ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
  if (ret == RKNN_SUCC) {
    std::printf("RKNN API version: %s\n", version.api_version);
    std::printf("RKNN Driver version: %s\n", version.drv_version);
  }

  // 4. Setup RKNN Zero-Copy Memory
  rknn_tensor_attr rknn_input_attr;
  rknn_input_attr.index = 0;
  ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &rknn_input_attr,
                   sizeof(rknn_input_attr));
  if (ret < 0) {
    std::fprintf(stderr, "rknn_query input attr failed: %d\n", ret);
    return 1;
  }

  rknn_tensor_attr rknn_output_attr;
  rknn_output_attr.index = 0;
  ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &rknn_output_attr,
                   sizeof(rknn_output_attr));
  if (ret < 0) {
    std::fprintf(stderr, "rknn_query output attr failed: %d\n", ret);
    return 1;
  }

  // Create NPU memory buffers
  rknn_tensor_mem *rknn_input_mem = rknn_create_mem(rknn_ctx, rknn_input_attr.size_with_stride);
  rknn_tensor_mem *rknn_output_mem = rknn_create_mem(rknn_ctx, rknn_output_attr.size_with_stride);

  // Set pass_through to true to avoid NPU driver format conversion overhead
  rknn_input_attr.pass_through = 1;
  rknn_output_attr.pass_through = 1;

  ret = rknn_set_io_mem(rknn_ctx, rknn_input_mem, &rknn_input_attr);
  if (ret < 0) {
    std::fprintf(stderr, "rknn_set_io_mem input failed: %d\n", ret);
    return 1;
  }
  ret = rknn_set_io_mem(rknn_ctx, rknn_output_mem, &rknn_output_attr);
  if (ret < 0) {
    std::fprintf(stderr, "rknn_set_io_mem output failed: %d\n", ret);
    return 1;
  }
  
  // Print memory sizes for debugging
  std::printf("NPU Input Mem Size: %d, Stride Size: %d\n", rknn_input_attr.size, rknn_input_attr.size_with_stride);
  std::printf("NPU Output Mem Size: %d, Stride Size: %d\n", rknn_output_attr.size, rknn_output_attr.size_with_stride);

  // 5. Pre-allocate Backend States
  std::vector<float> h_state(2 * 1 * 64, 0.0f);
  std::vector<float> c_state(2 * 1 * 64, 0.0f);
  const int64_t state_dims[] = {2, 1, 64};

  int segments = 0;
  bool previous_is_voice = false;

  // 6. Run Hybrid Inference Loop
  auto start_time = std::chrono::high_resolution_clock::now();

  for (auto &window : windows) {
    // --- Frontend (CPU) ---
    const int64_t input_dims[] = {1, WINDOW_SAMPLES};
    OrtValue* input_tensor = nullptr;
    check_status(ort->CreateTensorWithDataAsOrtValue(
        memory_info, window.data(), window.size() * sizeof(float),
        input_dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort);

    const char *frontend_input_names[] = {"x"};
    const char *frontend_output_names[] = {"/Concat_output_0"};

    // Run Frontend
    OrtValue* frontend_outputs[1] = {nullptr};
    check_status(ort->Run(frontend_sess, nullptr, frontend_input_names,
             (const OrtValue* const*)&input_tensor, 1, frontend_output_names,
             1, frontend_outputs), ort);
             
    // Copy data to RKNN input buffer
    float* frontend_out_ptr = nullptr;
    check_status(ort->GetTensorMutableData(frontend_outputs[0], (void**)&frontend_out_ptr), ort);
    
    // Simple FP32 to FP16 conversion (naive)
    uint16_t* rknn_in_ptr = static_cast<uint16_t*>(rknn_input_mem->virt_addr);
    for (int i = 0; i < 258 * 8; ++i) {
        uint32_t f32_val;
        std::memcpy(&f32_val, &frontend_out_ptr[i], sizeof(uint32_t));
        uint16_t sign = (f32_val >> 16) & 0x8000;
        int32_t exp = ((f32_val >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = f32_val & 0x007FFFFF;
        
        uint16_t f16_val;
        if (exp <= 0) {
            f16_val = sign; // Underflow to 0
        } else if (exp >= 31) {
            f16_val = sign | 0x7C00; // Overflow to infinity
        } else {
            f16_val = sign | (exp << 10) | (mantissa >> 13);
        }
        rknn_in_ptr[i] = f16_val;
    }

    // --- Midend (NPU) ---
    // Sync cache from CPU to Device before NPU execution
    rknn_mem_sync(rknn_ctx, rknn_input_mem, RKNN_MEMORY_SYNC_TO_DEVICE);

    ret = rknn_run(rknn_ctx, nullptr);
    if (ret < 0) {
      std::fprintf(stderr, "rknn_run failed: %d\n", ret);
      return 1;
    }

    // Sync cache from Device to CPU after NPU execution
    rknn_mem_sync(rknn_ctx, rknn_output_mem, RKNN_MEMORY_SYNC_FROM_DEVICE);

    // --- Backend (CPU) ---
    // Copy data from RKNN output buffer
    // The NPU output is also FP16 (size 128 = 64*1*2). We need to convert back to FP32.
    std::vector<float> backend_input_data(64);
    uint16_t* rknn_out_ptr = static_cast<uint16_t*>(rknn_output_mem->virt_addr);
    for (int i = 0; i < 64; ++i) {
        uint16_t f16_val = rknn_out_ptr[i];
        uint32_t sign = (f16_val & 0x8000) << 16;
        int32_t exp = ((f16_val >> 10) & 0x1F);
        uint32_t mantissa = (f16_val & 0x03FF) << 13;
        
        uint32_t f32_val;
        if (exp == 0) {
            f32_val = sign; // Zero or subnormal (flush to zero)
        } else if (exp == 31) {
            f32_val = sign | 0x7F800000 | mantissa; // Infinity or NaN
        } else {
            f32_val = sign | ((exp - 15 + 127) << 23) | mantissa;
        }
        std::memcpy(&backend_input_data[i], &f32_val, sizeof(float));
    }

    const int64_t backend_input_dims[] = {1, 64, 1};
    OrtValue* backend_midend_input_tensor = nullptr;
    check_status(ort->CreateTensorWithDataAsOrtValue(
        memory_info, backend_input_data.data(), backend_input_data.size() * sizeof(float),
        backend_input_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &backend_midend_input_tensor), ort);

    OrtValue* h_tensor = nullptr;
    check_status(ort->CreateTensorWithDataAsOrtValue(
        memory_info, h_state.data(), h_state.size() * sizeof(float),
        state_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &h_tensor), ort);
        
    OrtValue* c_tensor = nullptr;
    check_status(ort->CreateTensorWithDataAsOrtValue(
        memory_info, c_state.data(), c_state.size() * sizeof(float),
        state_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &c_tensor), ort);

    const char *backend_input_names[] = {"/encoder.14/Relu_output_0", "h", "c"};
    const OrtValue* backend_input_tensors[] = {
        backend_midend_input_tensor, h_tensor, c_tensor};

    const char *backend_output_names[] = {"prob", "new_h", "new_c"};
    OrtValue* backend_outputs[3] = {nullptr, nullptr, nullptr};
    check_status(ort->Run(backend_sess, nullptr, backend_input_names, backend_input_tensors, 3,
             backend_output_names, 3, backend_outputs), ort);

    float* prob_ptr = nullptr;
    check_status(ort->GetTensorMutableData(backend_outputs[0], (void**)&prob_ptr), ort);
    float speech_prob = prob_ptr[0];
    
    // Update states for next window
    float* out_h_ptr = nullptr;
    check_status(ort->GetTensorMutableData(backend_outputs[1], (void**)&out_h_ptr), ort);
    std::memcpy(h_state.data(), out_h_ptr, h_state.size() * sizeof(float));
    
    float* out_c_ptr = nullptr;
    check_status(ort->GetTensorMutableData(backend_outputs[2], (void**)&out_c_ptr), ort);
    std::memcpy(c_state.data(), out_c_ptr, c_state.size() * sizeof(float));

    bool current_is_voice = speech_prob > threshold;
    if (current_is_voice && !previous_is_voice) {
      segments++;
    }
    previous_is_voice = current_is_voice;
    
    // Release OrtValues for this window
    ort->ReleaseValue(input_tensor);
    ort->ReleaseValue(frontend_outputs[0]);
    ort->ReleaseValue(backend_midend_input_tensor);
    ort->ReleaseValue(h_tensor);
    ort->ReleaseValue(c_tensor);
    ort->ReleaseValue(backend_outputs[0]);
    ort->ReleaseValue(backend_outputs[1]);
    ort->ReleaseValue(backend_outputs[2]);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);
  float duration_seconds = duration_ms.count() / 1000.0f;
  float wav_duration_seconds =
      wav.data.size() / static_cast<float>(wav.sample_rate);
  float load_percentage = (duration_seconds / wav_duration_seconds) * 100.0f;

  std::printf(
      "C++ Hybrid Zero-Copy VAD | threshold=%.2f | windows=%zu | segments=%d | "
      "elapsed=%.2fs | load=%.2f%%\n",
      threshold, windows.size(), segments, duration_seconds, load_percentage);

  // 7. Cleanup
  rknn_destroy_mem(rknn_ctx, rknn_input_mem);
  rknn_destroy_mem(rknn_ctx, rknn_output_mem);
  rknn_destroy(rknn_ctx);
  
  ort->ReleaseSession(frontend_sess);
  ort->ReleaseSession(backend_sess);
  ort->ReleaseSessionOptions(session_options);
  ort->ReleaseEnv(env);
  ort->ReleaseMemoryInfo(memory_info);

  return 0;
}
