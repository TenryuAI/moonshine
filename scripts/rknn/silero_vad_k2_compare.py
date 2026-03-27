#!/usr/bin/env python3
"""Window-by-window compare of k2 ONNX CPU vs RKNN outputs."""

from __future__ import annotations

import argparse
import csv
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


def count_segments(probs: np.ndarray, threshold: float) -> int:
    segments = 0
    previous_is_voice = False
    for prob in probs:
        current_is_voice = float(prob) > threshold
        if current_is_voice and not previous_is_voice:
            segments += 1
        previous_is_voice = current_is_voice
    return segments


def compare_models(
    onnx_model: Path,
    rknn_model: Path,
    wav_path: Path,
    threshold: float,
    core_mask: int,
    top_k: int,
    csv_output: Path | None,
    reset_state_every_window: bool,
    reset_state_interval: int,
) -> int:
    if RKNNLite is None:
        raise RuntimeError("rknnlite is not available in this Python environment.")

    audio = load_wav_mono_16k(wav_path)
    windows = build_windows(audio)

    cpu_sess = ort.InferenceSession(str(onnx_model), providers=["CPUExecutionProvider"])
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(rknn_model))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")
    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    cpu_h = np.zeros(STATE_SHAPE, dtype=np.float32)
    cpu_c = np.zeros(STATE_SHAPE, dtype=np.float32)
    rknn_h = np.zeros(STATE_SHAPE, dtype=np.float32)
    rknn_c = np.zeros(STATE_SHAPE, dtype=np.float32)

    cpu_probs = []
    rknn_probs = []
    h_diffs = []
    c_diffs = []
    rows: list[dict[str, float | int]] = []

    for window_idx, window in enumerate(windows):
        should_reset = False
        if reset_state_every_window:
            should_reset = True
        elif reset_state_interval > 0 and window_idx % reset_state_interval == 0:
            should_reset = True

        if should_reset:
            cpu_h.fill(0.0)
            cpu_c.fill(0.0)
            rknn_h.fill(0.0)
            rknn_c.fill(0.0)

        cpu_prob, cpu_h, cpu_c = cpu_sess.run(
            ["prob", "new_h", "new_c"],
            {"x": window, "h": cpu_h, "c": cpu_c},
        )
        rknn_outputs = rknn.inference(inputs=[window, rknn_h, rknn_c])
        if rknn_outputs is None or len(rknn_outputs) < 3:
            raise RuntimeError("Unexpected RKNN outputs")

        rknn_prob = np.array(rknn_outputs[0], dtype=np.float32)
        rknn_h = np.array(rknn_outputs[1], dtype=np.float32).reshape(STATE_SHAPE)
        rknn_c = np.array(rknn_outputs[2], dtype=np.float32).reshape(STATE_SHAPE)

        cpu_prob_value = float(np.array(cpu_prob).reshape(-1)[0])
        rknn_prob_value = float(rknn_prob.reshape(-1)[0])
        h_diff_value = float(np.mean(np.abs(cpu_h - rknn_h)))
        c_diff_value = float(np.mean(np.abs(cpu_c - rknn_c)))

        cpu_probs.append(cpu_prob_value)
        rknn_probs.append(rknn_prob_value)
        h_diffs.append(h_diff_value)
        c_diffs.append(c_diff_value)
        rows.append(
            {
                "window": window_idx,
                "time_start_s": window_idx * WINDOW_SAMPLES / DEFAULT_SAMPLE_RATE,
                "time_end_s": (window_idx + 1) * WINDOW_SAMPLES / DEFAULT_SAMPLE_RATE,
                "cpu_prob": cpu_prob_value,
                "rknn_prob": rknn_prob_value,
                "prob_diff": abs(cpu_prob_value - rknn_prob_value),
                "h_diff": h_diff_value,
                "c_diff": c_diff_value,
                "cpu_voice": int(cpu_prob_value > threshold),
                "rknn_voice": int(rknn_prob_value > threshold),
            }
        )

    cpu_probs_arr = np.array(cpu_probs, dtype=np.float32)
    rknn_probs_arr = np.array(rknn_probs, dtype=np.float32)
    abs_diff = np.abs(cpu_probs_arr - rknn_probs_arr)
    h_diffs_arr = np.array(h_diffs, dtype=np.float32)
    c_diffs_arr = np.array(c_diffs, dtype=np.float32)
    cpu_voice = cpu_probs_arr > threshold
    rknn_voice = rknn_probs_arr > threshold
    voice_mismatch = cpu_voice != rknn_voice

    print(
        "Compare | reset_state={} | reset_interval={} | windows={} | threshold={:.2f} | cpu_segments={} | rknn_segments={}".format(
            int(reset_state_every_window),
            int(reset_state_interval),
            len(windows),
            threshold,
            count_segments(cpu_probs_arr, threshold),
            count_segments(rknn_probs_arr, threshold),
        )
    )
    print(
        "Compare | prob_mae={:.6f} | prob_max_abs={:.6f} | prob_corr={:.6f}".format(
            float(np.mean(abs_diff)),
            float(np.max(abs_diff)),
            float(np.corrcoef(cpu_probs_arr, rknn_probs_arr)[0, 1]),
        )
    )
    print(
        "Compare | h_mae={:.6f} | c_mae={:.6f}".format(
            float(np.mean(h_diffs)),
            float(np.mean(c_diffs)),
        )
    )
    print(
        "Compare | voice_mismatch_windows={} | max_h_diff={:.6f} | max_c_diff={:.6f}".format(
            int(np.count_nonzero(voice_mismatch)),
            float(np.max(h_diffs_arr)),
            float(np.max(c_diffs_arr)),
        )
    )

    ranked = np.argsort(abs_diff)[::-1][:top_k]
    print("Top probability diffs:")
    for idx in ranked:
        t0 = idx * WINDOW_SAMPLES / DEFAULT_SAMPLE_RATE
        t1 = (idx + 1) * WINDOW_SAMPLES / DEFAULT_SAMPLE_RATE
        print(
            "  window={} time=[{:.2f},{:.2f}] cpu={:.6f} rknn={:.6f} diff={:.6f}".format(
                int(idx),
                t0,
                t1,
                float(cpu_probs_arr[idx]),
                float(rknn_probs_arr[idx]),
                float(abs_diff[idx]),
            )
        )

    ranked_h = np.argsort(h_diffs_arr)[::-1][:top_k]
    print("Top h diffs:")
    for idx in ranked_h:
        t0 = idx * WINDOW_SAMPLES / DEFAULT_SAMPLE_RATE
        t1 = (idx + 1) * WINDOW_SAMPLES / DEFAULT_SAMPLE_RATE
        print(
            "  window={} time=[{:.2f},{:.2f}] h_diff={:.6f} c_diff={:.6f} cpu={:.6f} rknn={:.6f}".format(
                int(idx),
                t0,
                t1,
                float(h_diffs_arr[idx]),
                float(c_diffs_arr[idx]),
                float(cpu_probs_arr[idx]),
                float(rknn_probs_arr[idx]),
            )
        )

    mismatch_indices = np.flatnonzero(voice_mismatch)[:top_k]
    if mismatch_indices.size:
        print("Top voice mismatches:")
        for idx in mismatch_indices:
            t0 = idx * WINDOW_SAMPLES / DEFAULT_SAMPLE_RATE
            t1 = (idx + 1) * WINDOW_SAMPLES / DEFAULT_SAMPLE_RATE
            print(
                "  window={} time=[{:.2f},{:.2f}] cpu_voice={} rknn_voice={} cpu={:.6f} rknn={:.6f}".format(
                    int(idx),
                    t0,
                    t1,
                    int(cpu_voice[idx]),
                    int(rknn_voice[idx]),
                    float(cpu_probs_arr[idx]),
                    float(rknn_probs_arr[idx]),
                )
            )

    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "window",
                    "time_start_s",
                    "time_end_s",
                    "cpu_prob",
                    "rknn_prob",
                    "prob_diff",
                    "h_diff",
                    "c_diff",
                    "cpu_voice",
                    "rknn_voice",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written to: {csv_output}")

    if hasattr(rknn, "release"):
        rknn.release()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare k2 ONNX CPU and RKNN window outputs")
    parser.add_argument("--onnx-model", required=True, help="Path to k2 ONNX model")
    parser.add_argument("--rknn-model", required=True, help="Path to RKNN model")
    parser.add_argument("--wav-path", required=True, help="Path to mono 16k wav")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--core-mask", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional path to write per-window comparison CSV",
    )
    parser.add_argument(
        "--reset-state-every-window",
        action="store_true",
        help="Reset h/c to zero before every window for both CPU and RKNN",
    )
    parser.add_argument(
        "--reset-state-interval",
        type=int,
        default=0,
        help="Reset h/c every N windows for both CPU and RKNN (0 disables)",
    )
    args = parser.parse_args()

    for path_str in (args.onnx_model, args.rknn_model, args.wav_path):
        path = Path(path_str)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 1

    return compare_models(
        onnx_model=Path(args.onnx_model),
        rknn_model=Path(args.rknn_model),
        wav_path=Path(args.wav_path),
        threshold=args.threshold,
        core_mask=args.core_mask,
        top_k=args.top_k,
        csv_output=Path(args.csv_output) if args.csv_output else None,
        reset_state_every_window=args.reset_state_every_window,
        reset_state_interval=args.reset_state_interval,
    )


if __name__ == "__main__":
    raise SystemExit(main())
