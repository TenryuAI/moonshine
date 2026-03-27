#!/usr/bin/env python3
"""Host-side ONNX -> RKNN conversion skeleton for Silero VAD.

This script is intended to run on a host machine with RKNN Toolkit2 installed,
not on the RK3568 target device.

Typical workflow:
1. Obtain a standalone `silero_vad.onnx`
2. Run this script on the host
3. Copy the generated `.rknn` model to the RK3568 target
4. Benchmark on-device with `silero_vad_rknn_benchmark.py`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from rknn.api import RKNN  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - host-side only
    RKNN = None


DEFAULT_TARGET = "rk3568"


def convert_onnx_to_rknn(
    onnx_path: Path,
    output_path: Path,
    target_platform: str,
    do_quantization: bool,
    dataset_path: str | None,
) -> int:
    if RKNN is None:
        raise RuntimeError(
            "rknn-toolkit2 is not available in this Python environment. "
            "Run this script on a host with rknn-toolkit2 installed."
        )

    rknn = RKNN(verbose=True)

    # Audio models such as Silero VAD should keep their original input semantics.
    # Avoid image-style mean/std preprocessing unless a specific model requires it.
    ret = rknn.config(target_platform=target_platform)
    if ret != 0:
        raise RuntimeError(f"rknn.config failed: {ret}")

    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed: {ret}")

    build_kwargs = {"do_quantization": do_quantization}
    if do_quantization and dataset_path:
        build_kwargs["dataset"] = dataset_path

    ret = rknn.build(**build_kwargs)
    if ret != 0:
        raise RuntimeError(f"rknn.build failed: {ret}")

    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed: {ret}")

    rknn.release()
    print(f"Exported RKNN model to: {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Silero VAD ONNX to RKNN")
    parser.add_argument("--onnx-path", required=True, help="Path to silero_vad.onnx")
    parser.add_argument(
        "--output-path",
        default="silero_vad.rknn",
        help="Output RKNN path (default: silero_vad.rknn)",
    )
    parser.add_argument(
        "--target-platform",
        default=DEFAULT_TARGET,
        help=f"RKNN target platform (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable RKNN quantization during build",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional calibration dataset file used when --quantize is enabled",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx_path)
    output_path = Path(args.output_path)
    if not onnx_path.exists():
        print(f"ONNX file not found: {onnx_path}", file=sys.stderr)
        return 1

    if args.quantize and not args.dataset:
        print(
            "--quantize was requested but no --dataset was supplied. "
            "Continuing without dataset is usually not what you want.",
            file=sys.stderr,
        )

    return convert_onnx_to_rknn(
        onnx_path=onnx_path,
        output_path=output_path,
        target_platform=args.target_platform,
        do_quantization=args.quantize,
        dataset_path=args.dataset,
    )


if __name__ == "__main__":
    raise SystemExit(main())
