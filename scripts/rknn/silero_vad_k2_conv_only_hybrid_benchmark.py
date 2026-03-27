#!/usr/bin/env python3
"""Benchmark k2 Silero VAD with CPU frontend, RKNN midend (Conv), and CPU backend (LSTM)."""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    from rknnlite.api import RKNNLite  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - device runtime only
    RKNNLite = None


WINDOW_SAMPLES = 512
STATE_SHAPE = (2, 1, 64)
MIDEND_INPUT_SHAPE = (1, 258, 8)
MIDEND_OUTPUT_SHAPE = (1, 64, 1)
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
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def build_windows(audio: np.ndarray) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    for start in range(0, len(audio), WINDOW_SAMPLES):
        chunk = audio[start : start + WINDOW_SAMPLES]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        windows.append(chunk.astype(np.float32, copy=False).reshape(1, WINDOW_SAMPLES))
    return windows


def benchmark_hybrid(
    frontend_model_path: Path,
    midend_model_path: Path,
    backend_model_path: Path,
    wav_path: Path,
    threshold: float,
    core_mask: int,
) -> int:
    if RKNNLite is None:
        raise RuntimeError("rknnlite is not available in this Python environment.")

    audio = load_wav_mono_16k(wav_path)
    windows = build_windows(audio)

    frontend = ort.InferenceSession(
        str(frontend_model_path),
        providers=["CPUExecutionProvider"],
    )
    
    backend = ort.InferenceSession(
        str(backend_model_path),
        providers=["CPUExecutionProvider"],
    )

    rknn = RKNNLite()
    ret = rknn.load_rknn(str(midend_model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    h = np.zeros(STATE_SHAPE, dtype=np.float32)
    c = np.zeros(STATE_SHAPE, dtype=np.float32)

    segments = 0
    previous_is_voice = False

    start = time.perf_counter()
    for window in windows:
        # 1. CPU Frontend
        features = frontend.run(["/Concat_output_0"], {"x": window})[0]
        features = np.array(features, dtype=np.float32).reshape(MIDEND_INPUT_SHAPE)
        
        # 2. NPU Midend (Conv cluster)
        midend_outputs = rknn.inference(inputs=[features])
        if midend_outputs is None or len(midend_outputs) < 1:
            raise RuntimeError("Unexpected RKNN outputs")
        encoded_features = np.array(midend_outputs[0], dtype=np.float32).reshape(MIDEND_OUTPUT_SHAPE)
        
        # 3. CPU Backend (LSTM)
        backend_outputs = backend.run(["prob", "new_h", "new_c"], {
            "/encoder.14/Relu_output_0": encoded_features,
            "h": h,
            "c": c
        })
        
        speech_prob = float(np.array(backend_outputs[0]).reshape(-1)[0])
        h = np.array(backend_outputs[1], dtype=np.float32).reshape(STATE_SHAPE)
        c = np.array(backend_outputs[2], dtype=np.float32).reshape(STATE_SHAPE)
        
        current_is_voice = speech_prob > threshold
        if current_is_voice and not previous_is_voice:
            segments += 1
        previous_is_voice = current_is_voice
        
    elapsed = time.perf_counter() - start

    audio_duration = len(audio) / float(DEFAULT_SAMPLE_RATE)
    load_percentage = (elapsed / audio_duration) * 100.0 if audio_duration else 0.0
    print(
        "Conv-Only Hybrid VAD | core_mask={} | threshold={:.2f} | windows={} | segments={} | "
        "elapsed={:.2f}s | load={:.2f}%".format(
            core_mask, threshold, len(windows), segments, elapsed, load_percentage
        )
    )

    if hasattr(rknn, "release"):
        rknn.release()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark k2 CPU + RKNN Conv + CPU LSTM")
    parser.add_argument("--frontend-model", required=True)
    parser.add_argument("--midend-model", required=True)
    parser.add_argument("--backend-model", required=True)
    parser.add_argument("--wav-path", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--core-mask", type=int, default=0)
    args = parser.parse_args()

    return benchmark_hybrid(
        frontend_model_path=Path(args.frontend_model),
        midend_model_path=Path(args.midend_model),
        backend_model_path=Path(args.backend_model),
        wav_path=Path(args.wav_path),
        threshold=args.threshold,
        core_mask=args.core_mask,
    )


if __name__ == "__main__":
    raise SystemExit(main())
