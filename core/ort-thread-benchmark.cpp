#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "moonshine-utils/debug-utils.h"
#include "transcriber.h"

namespace {

class AudioProducer {
 public:
  AudioProducer(const std::string &wav_path,
                float chunk_duration_seconds = 0.0214f)
      : current_index_(0) {
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    if (!load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                       &wav_sample_rate)) {
      throw std::runtime_error("Failed to load WAV file: " + wav_path);
    }
    audio_data_.assign(wav_data, wav_data + wav_data_size);
    free(wav_data);
    sample_rate_ = wav_sample_rate;
    chunk_size_ = static_cast<size_t>(chunk_duration_seconds * sample_rate_);
    if (chunk_size_ == 0) {
      chunk_size_ = 1;
    }
  }

  bool get_next_audio(std::vector<float> &out_audio_data) {
    if (current_index_ >= audio_data_.size()) {
      return false;
    }
    size_t end_index =
        std::min(current_index_ + chunk_size_, audio_data_.size());
    out_audio_data.assign(audio_data_.begin() + current_index_,
                          audio_data_.begin() + end_index);
    current_index_ = end_index;
    return true;
  }

  int32_t sample_rate() const { return sample_rate_; }
  size_t audio_data_size() const { return audio_data_.size(); }

 private:
  size_t chunk_size_;
  int32_t sample_rate_ = 16000;
  size_t current_index_;
  std::vector<float> audio_data_;
};

uint32_t parse_model_arch(const std::string &value) {
  return static_cast<uint32_t>(std::stoi(value));
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string model_path = "../test-assets/tiny-en";
  std::string wav_path = "../test-assets/two_cities.wav";
  uint32_t model_arch = MOONSHINE_MODEL_ARCH_TINY;
  float transcription_interval_seconds = 0.481f;
  int32_t intra_threads = 0;
  int32_t inter_threads = 0;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-m" || arg == "--model-path") && i + 1 < argc) {
      model_path = argv[++i];
    } else if ((arg == "-a" || arg == "--model-arch") && i + 1 < argc) {
      model_arch = parse_model_arch(argv[++i]);
    } else if ((arg == "-w" || arg == "--wav-path") && i + 1 < argc) {
      wav_path = argv[++i];
    } else if ((arg == "-t" || arg == "--transcription-interval") &&
               i + 1 < argc) {
      transcription_interval_seconds = std::stof(argv[++i]);
    } else if (arg == "--ort-intra-op-threads" && i + 1 < argc) {
      intra_threads = std::stoi(argv[++i]);
    } else if (arg == "--ort-inter-op-threads" && i + 1 < argc) {
      inter_threads = std::stoi(argv[++i]);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  AudioProducer audio_producer(wav_path);

  TranscriberOptions options;
  options.model_source = TranscriberOptions::ModelSource::FILES;
  options.model_path = model_path.c_str();
  options.model_arch = model_arch;
  options.transcription_interval = transcription_interval_seconds;
  options.identify_speakers = false;
  options.return_audio_data = false;
  options.ort_intra_op_threads = intra_threads;
  options.ort_inter_op_threads = inter_threads;

  Transcriber transcriber(options);
  int32_t stream_id = transcriber.create_stream();

  auto start = std::chrono::high_resolution_clock::now();
  transcriber.start_stream(stream_id);

  std::vector<float> chunk_audio_data;
  const int32_t samples_between_transcriptions = static_cast<int32_t>(
      transcription_interval_seconds * audio_producer.sample_rate());
  int32_t samples_since_last_transcription = 0;

  transcript_t *transcript = nullptr;
  while (audio_producer.get_next_audio(chunk_audio_data)) {
    transcriber.add_audio_to_stream(stream_id, chunk_audio_data.data(),
                                    chunk_audio_data.size(),
                                    audio_producer.sample_rate());
    samples_since_last_transcription +=
        static_cast<int32_t>(chunk_audio_data.size());
    if (samples_since_last_transcription < samples_between_transcriptions) {
      continue;
    }
    samples_since_last_transcription = 0;
    transcriber.transcribe_stream(stream_id, 0, &transcript);
  }

  transcriber.stop_stream(stream_id);
  transcriber.transcribe_stream(stream_id, 0, &transcript);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  float duration_seconds = duration_ms.count() / 1000.0f;
  float wav_duration_seconds =
      audio_producer.audio_data_size() / static_cast<float>(audio_producer.sample_rate());
  float transcription_percentage =
      (duration_seconds / wav_duration_seconds) * 100.0f;

  int32_t total_latency_ms = 0;
  uint64_t line_count = transcript ? transcript->line_count : 0;
  if (transcript != nullptr) {
    for (uint64_t i = 0; i < transcript->line_count; ++i) {
      total_latency_ms += transcript->lines[i].last_transcription_latency_ms;
    }
  }

  std::fprintf(stderr,
               "ORT threads: intra=%d inter=%d | lines=%llu | avg latency=%.0fms | "
               "elapsed=%.2fs | load=%.2f%%\n",
               intra_threads, inter_threads,
               static_cast<unsigned long long>(line_count),
               line_count == 0 ? 0.0f : total_latency_ms / static_cast<float>(line_count),
               duration_seconds, transcription_percentage);

  return 0;
}
