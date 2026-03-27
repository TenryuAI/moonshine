#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "moonshine-utils/debug-utils.h"
#include "voice-activity-detector.h"

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
  std::fprintf(stderr,
               "Usage: vad-benchmark [--wav-path path] [--mode block|stream] "
               "[--chunk-seconds value] [--vad-threshold value]\n");
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string wav_path = "../test-assets/two_cities.wav";
  std::string mode = "block";
  float chunk_seconds = 0.1f;
  float vad_threshold = 0.5f;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-w" || arg == "--wav-path") && i + 1 < argc) {
      wav_path = argv[++i];
    } else if ((arg == "-m" || arg == "--mode") && i + 1 < argc) {
      mode = argv[++i];
    } else if ((arg == "-c" || arg == "--chunk-seconds") && i + 1 < argc) {
      chunk_seconds = std::stof(argv[++i]);
    } else if ((arg == "-t" || arg == "--vad-threshold") && i + 1 < argc) {
      vad_threshold = std::stof(argv[++i]);
    } else {
      print_usage();
      return 1;
    }
  }

  AudioFile wav = load_wav_or_throw(wav_path);
  VoiceActivityDetector vad(vad_threshold);

  auto start = std::chrono::high_resolution_clock::now();
  vad.start();

  if (mode == "block") {
    vad.process_audio(wav.data.data(), wav.data.size(), wav.sample_rate);
  } else if (mode == "stream") {
    const size_t chunk_size =
        std::max<size_t>(1, static_cast<size_t>(chunk_seconds * wav.sample_rate));
    for (size_t i = 0; i < wav.data.size(); i += chunk_size) {
      const size_t chunk_data_size = std::min(chunk_size, wav.data.size() - i);
      vad.process_audio(wav.data.data() + i, chunk_data_size, wav.sample_rate);
    }
  } else {
    std::fprintf(stderr, "Unknown mode: %s\n", mode.c_str());
    return 1;
  }

  vad.stop();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  float duration_seconds = duration_ms.count() / 1000.0f;
  float wav_duration_seconds =
      wav.data.size() / static_cast<float>(wav.sample_rate);
  float processing_percentage =
      (duration_seconds / wav_duration_seconds) * 100.0f;
  const std::vector<VoiceActivitySegment> *segments = vad.get_segments();

  std::fprintf(stderr,
               "VAD mode=%s threshold=%.2f chunk=%.3fs | segments=%zu | "
               "elapsed=%.2fs | load=%.2f%%\n",
               mode.c_str(), vad_threshold, chunk_seconds, segments->size(),
               duration_seconds, processing_percentage);
  return 0;
}
