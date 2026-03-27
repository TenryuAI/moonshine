#!/usr/bin/env python3
"""Compare all intermediate nodes of k2 backend ONNX vs RKNN for a single window."""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    from rknnlite.api import RKNNLite  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - only available on RK device runtime
    RKNNLite = None


WINDOW_SAMPLES = 512
STATE_SHAPE = (2, 1, 64)
FEATURE_SHAPE = (1, 258, 8)
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


def compare_nodes(
    frontend_model: Path,
    onnx_backend: Path,
    rknn_backend: Path,
    wav_path: Path,
    core_mask: int,
) -> int:
    if RKNNLite is None:
        raise RuntimeError("rknnlite is not available in this Python environment.")

    audio = load_wav_mono_16k(wav_path)
    windows = build_windows(audio)

    frontend = ort.InferenceSession(str(frontend_model), providers=["CPUExecutionProvider"])

    cpu_sess = ort.InferenceSession(str(onnx_backend), providers=["CPUExecutionProvider"])
    output_names = [o.name for o in cpu_sess.get_outputs()]
    
    # RKNN optimizes away the input node and the 4D LSTM outputs
    ignored_outputs = {
        "/Concat_output_0",
        "/decoder/rnn/LSTM_output_0",
        "/decoder/rnn/LSTM_1_output_0",
    }
    filtered_output_names = [name for name in output_names if name not in ignored_outputs]
    
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(rknn_backend))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")
    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    h = np.zeros(STATE_SHAPE, dtype=np.float32)
    c = np.zeros(STATE_SHAPE, dtype=np.float32)
    rknn_h = np.zeros(STATE_SHAPE, dtype=np.float32)
    rknn_c = np.zeros(STATE_SHAPE, dtype=np.float32)

    for window_idx in range(10):
        window = windows[window_idx]
        features = frontend.run(["/Concat_output_0"], {"x": window})[0]
        features = np.array(features, dtype=np.float32).reshape(FEATURE_SHAPE)

        cpu_outputs = cpu_sess.run(output_names, {"/Concat_output_0": features, "h": h, "c": c})
        rknn_outputs = rknn.inference(inputs=[features, rknn_h, rknn_c])

        if rknn_outputs is None or len(rknn_outputs) != len(filtered_output_names):
            raise RuntimeError(f"Unexpected RKNN outputs length: {len(rknn_outputs) if rknn_outputs else 0} vs {len(filtered_output_names)}")
        
        results = []
        cpu_outputs_dict = dict(zip(output_names, cpu_outputs))
        for name, rknn_out in zip(filtered_output_names, rknn_outputs):
            cpu_arr = np.array(cpu_outputs_dict[name], dtype=np.float32)
            rknn_arr = np.array(rknn_out, dtype=np.float32).reshape(cpu_arr.shape)
            abs_diff = np.abs(cpu_arr - rknn_arr)
            mae = float(np.mean(abs_diff))
            max_diff = float(np.max(abs_diff))
            results.append((name, mae, max_diff, cpu_arr.shape))
            
        print(f"\n=== Window {window_idx} ===")
        print(f"{'Node Name':<40} | {'MAE':<10} | {'MaxDiff':<10} | {'Shape'}")
        print("-" * 80)
        for name, mae, max_diff, shape in results:
            if max_diff > 0.001: # Only print nodes with significant error
                print(f"{name:<40} | {mae:<10.6f} | {max_diff:<10.6f} | {shape}")
                
        # Update states
        h = cpu_outputs_dict["new_h"]
        c = cpu_outputs_dict["new_c"]
        
        # Find RKNN new_h and new_c
        rknn_h_idx = filtered_output_names.index("new_h")
        rknn_c_idx = filtered_output_names.index("new_c")
        rknn_h = np.array(rknn_outputs[rknn_h_idx], dtype=np.float32).reshape(STATE_SHAPE)
        rknn_c = np.array(rknn_outputs[rknn_c_idx], dtype=np.float32).reshape(STATE_SHAPE)

    if hasattr(rknn, "release"):
        rknn.release()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare all nodes CPU vs RKNN")
    parser.add_argument("--frontend-model", required=True)
    parser.add_argument("--onnx-backend", required=True)
    parser.add_argument("--rknn-backend", required=True)
    parser.add_argument("--wav-path", required=True)
    parser.add_argument("--core-mask", type=int, default=0)
    args = parser.parse_args()

    return compare_nodes(
        frontend_model=Path(args.frontend_model),
        onnx_backend=Path(args.onnx_backend),
        rknn_backend=Path(args.rknn_backend),
        wav_path=Path(args.wav_path),
        core_mask=args.core_mask,
    )


if __name__ == "__main__":
    raise SystemExit(main())
