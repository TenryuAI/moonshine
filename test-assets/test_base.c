#include <stdio.h>
#include "../core/moonshine-c-api.h"
#include <stdlib.h>

int main() {
    printf("Testing base-en model...\n");
    
    int32_t handle = moonshine_load_transcriber_from_files(
        "base-en", MOONSHINE_MODEL_ARCH_BASE, NULL, 0, MOONSHINE_HEADER_VERSION);
    printf("Load handle: %d\n", handle);
    if (handle < 0) { printf("FAILED to load\n"); return 1; }
    
    // Generate 2 seconds of sine wave at 48kHz as test audio
    int sample_rate = 48000;
    int num_samples = 96000;
    float* audio = (float*)malloc(num_samples * sizeof(float));
    for (int i = 0; i < num_samples; i++) {
        audio[i] = 0.3f * sinf(2.0f * 3.14159f * 440.0f * i / sample_rate);
    }
    
    struct transcript_t* transcript = NULL;
    printf("Calling transcribe_without_streaming...\n");
    int32_t err = moonshine_transcribe_without_streaming(
        handle, audio, num_samples, sample_rate, 0, &transcript);
    printf("err=%d\n", err);
    if (transcript) printf("line_count=%llu\n", (unsigned long long)transcript->line_count);
    
    free(audio);
    moonshine_free_transcriber(handle);
    printf("Done.\n");
    return 0;
}
