#!/usr/bin/env python3
"""Minimal RKNN benchmark for Silero-style VAD on Rockchip devices.

This script is intended for device-side validation on RK356x / RK3588 class
boards once a `.rknn` version of the Silero VAD model is available.

Supported model layouts:
- `moonshine`: mirrors `core/silero-vad.cpp`
  - input: `[1, 576]` = `64` context + `512` window
  - state: `[2, 1, 128]`
  - sample rate scalar
- `k2`: the successfully converted `silero_vad_k2.rknn`
  - input: `[1, 512]`
  - h: `[2, 1, 64]`
  - c: `[2, 1, 64]`

The goal is not to perfectly reproduce every detail of the C++ VAD wrapper, but
to make it easy to answer:
- Can the `.rknn` model run on-device?
- What is the per-audio-load percentage compared to the current CPU path?
- Is the NPU path stable across an entire WAV file?
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
try:
    from rknnlite.api import RKNNLite  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - only available on RK device runtime
    RKNNLite = None


MOONSHINE_CONTEXT_SAMPLES = 64
WINDOW_SAMPLES = 512
MOONSHINE_EFFECTIVE_WINDOW_SAMPLES = MOONSHINE_CONTEXT_SAMPLES + WINDOW_SAMPLES
MOONSHINE_STATE_SHAPE = (2, 1, 128)
K2_STATE_SHAPE = (2, 1, 64)
DEFAULT_SAMPLE_RATE = 16000


def load_wav_mono_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        if channels != 1:
            raise ValueError(f"Expected mono wav, got {channels} channels")
        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM wav, got {sample_width * 8}-bit")
        if sample_rate != DEFAULT_SAMPLE_RATE:
            raise ValueError(
                f"Expected {DEFAULT_SAMPLE_RATE} Hz wav, got {sample_rate} Hz"
            )
        raw = wav_file.readframes(frame_count)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def build_moonshine_windows(audio: np.ndarray) -> list[np.ndarray]:
    context = np.zeros(MOONSHINE_CONTEXT_SAMPLES, dtype=np.float32)
    windows: list[np.ndarray] = []
    for start in range(0, len(audio), WINDOW_SAMPLES):
        chunk = audio[start : start + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        full_input = np.concatenate([context, chunk]).astype(np.float32, copy=False)
        windows.append(full_input.reshape(1, MOONSHINE_EFFECTIVE_WINDOW_SAMPLES))
        context = chunk[-MOONSHINE_CONTEXT_SAMPLES:].copy()
    return windows


def build_k2_windows(audio: np.ndarray) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    for start in range(0, len(audio), WINDOW_SAMPLES):
        chunk = audio[start : start + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        windows.append(chunk.astype(np.float32, copy=False).reshape(1, WINDOW_SAMPLES))
    return windows


def benchmark_rknn(
    model_path: Path,
    wav_path: Path,
    threshold: float,
    core_mask: int,
    model_layout: str,
) -> int:
    if RKNNLite is None:
        raise RuntimeError(
            "rknnlite is not available in this Python environment. "
            "Run this benchmark on an RKNN Lite capable device."
        )
    audio = load_wav_mono_16k(wav_path)
    if model_layout == "moonshine":
        windows = build_moonshine_windows(audio)
    elif model_layout == "k2":
        windows = build_k2_windows(audio)
    else:
        raise ValueError(f"Unsupported model layout: {model_layout}")

    rknn = RKNNLite()
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    segments = 0
    previous_is_voice = False

    start = time.perf_counter()
    if model_layout == "moonshine":
        state = np.zeros(MOONSHINE_STATE_SHAPE, dtype=np.float32)
        sample_rate = np.array(DEFAULT_SAMPLE_RATE, dtype=np.int64)
    else:
        h = np.zeros(K2_STATE_SHAPE, dtype=np.float32)
        c = np.zeros(K2_STATE_SHAPE, dtype=np.float32)

    for window in windows:
        if model_layout == "moonshine":
            outputs = rknn.inference(inputs=[window, state, sample_rate])
            if outputs is None or len(outputs) < 2:
                raise RuntimeError("Unexpected RKNN outputs for moonshine layout")
            speech_prob = float(np.array(outputs[0]).reshape(-1)[0])
            state = np.array(outputs[1], dtype=np.float32).reshape(
                MOONSHINE_STATE_SHAPE
            )
        else:
            outputs = rknn.inference(inputs=[window, h, c])
            if outputs is None or len(outputs) < 3:
                raise RuntimeError("Unexpected RKNN outputs for k2 layout")
            speech_prob = float(np.array(outputs[0]).reshape(-1)[0])
            h = np.array(outputs[1], dtype=np.float32).reshape(K2_STATE_SHAPE)
            c = np.array(outputs[2], dtype=np.float32).reshape(K2_STATE_SHAPE)
        current_is_voice = speech_prob > threshold
        if current_is_voice and not previous_is_voice:
            segments += 1
        previous_is_voice = current_is_voice
    elapsed = time.perf_counter() - start

    audio_duration = len(audio) / float(DEFAULT_SAMPLE_RATE)
    load_percentage = (elapsed / audio_duration) * 100.0 if audio_duration else 0.0

    print(
        "RKNN VAD | layout={} | core_mask={} | threshold={:.2f} | windows={} | "
        "segments={} | elapsed={:.2f}s | load={:.2f}%".format(
            model_layout,
            core_mask,
            threshold,
            len(windows),
            segments,
            elapsed,
            load_percentage,
        )
    )

    if hasattr(rknn, "release"):
        rknn.release()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark RKNN Silero VAD")
    parser.add_argument("--model-path", required=True, help="Path to .rknn file")
    parser.add_argument(
        "--wav-path",
        default="test-assets/two_cities_16k.wav",
        help="Path to mono 16k wav file",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Speech probability threshold"
    )
    parser.add_argument(
        "--core-mask",
        type=int,
        default=0,
        help="RKNNLite core_mask value (0 keeps runtime default)",
    )
    parser.add_argument(
        "--model-layout",
        choices=("moonshine", "k2"),
        default="k2",
        help="RKNN model input/output layout (default: k2)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    wav_path = Path(args.wav_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1
    if not wav_path.exists():
        print(f"Wav file not found: {wav_path}", file=sys.stderr)
        return 1

    return benchmark_rknn(
        model_path, wav_path, args.threshold, args.core_mask, args.model_layout
    )


if __name__ == "__main__":
    raise SystemExit(main())
