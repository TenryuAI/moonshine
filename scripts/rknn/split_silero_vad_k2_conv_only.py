#!/usr/bin/env python3
"""Split k2 Silero VAD into three parts:
1. CPU Frontend: feature extraction up to /Concat_output_0
2. NPU Midend: The main Conv block (/Concat_output_0 -> /encoder.14/Relu_output_0)
3. CPU Backend: The LSTM decoder + final Conv/Sigmoid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import utils


FRONTEND_OUTPUT = "/Concat_output_0"
MIDEND_INPUT = "/Concat_output_0"
MIDEND_OUTPUT = "/encoder.14/Relu_output_0"
BACKEND_INPUTS = [MIDEND_OUTPUT, "h", "c"]
BACKEND_OUTPUTS = ["prob", "new_h", "new_c"]

# Known shapes for a single window (batch=1)
MIDEND_INPUT_SHAPE = [1, 258, 8]
MIDEND_OUTPUT_SHAPE = [1, 64, 1]


def fix_shape(model: onnx.ModelProto, tensor_name: str, shape: list[int]) -> None:
    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if value_info.name == tensor_name:
            dims = value_info.type.tensor_type.shape.dim
            del dims[:]
            for value in shape:
                dims.add().dim_value = value


def split_model(src: Path, frontend_out: Path, midend_out: Path, backend_out: Path) -> None:
    # 1. Frontend (CPU)
    utils.extract_model(
        str(src),
        str(frontend_out),
        input_names=["x"],
        output_names=[FRONTEND_OUTPUT],
    )

    # 2. Midend (NPU)
    utils.extract_model(
        str(src),
        str(midend_out),
        input_names=[MIDEND_INPUT],
        output_names=[MIDEND_OUTPUT],
    )

    # 3. Backend (CPU)
    utils.extract_model(
        str(src),
        str(backend_out),
        input_names=BACKEND_INPUTS,
        output_names=BACKEND_OUTPUTS,
    )

    # Fix shapes for RKNN conversion
    midend_model = onnx.load(str(midend_out))
    fix_shape(midend_model, MIDEND_INPUT, MIDEND_INPUT_SHAPE)
    fix_shape(midend_model, MIDEND_OUTPUT, MIDEND_OUTPUT_SHAPE)
    onnx.checker.check_model(midend_model)
    onnx.save(midend_model, str(midend_out))

    frontend_model = onnx.load(str(frontend_out))
    fix_shape(frontend_model, FRONTEND_OUTPUT, MIDEND_INPUT_SHAPE)
    onnx.checker.check_model(frontend_model)
    onnx.save(frontend_model, str(frontend_out))

    backend_model = onnx.load(str(backend_out))
    fix_shape(backend_model, MIDEND_OUTPUT, MIDEND_OUTPUT_SHAPE)
    onnx.checker.check_model(backend_model)
    onnx.save(backend_model, str(backend_out))


def main() -> int:
    parser = argparse.ArgumentParser(description="Split k2 Silero VAD for Conv-only NPU")
    parser.add_argument("--input", required=True)
    parser.add_argument("--frontend-output", required=True)
    parser.add_argument("--midend-output", required=True)
    parser.add_argument("--backend-output", required=True)
    args = parser.parse_args()

    split_model(
        Path(args.input),
        Path(args.frontend_output),
        Path(args.midend_output),
        Path(args.backend_output),
    )
    print(f"Frontend: {args.frontend_output}")
    print(f"Midend:   {args.midend_output}")
    print(f"Backend:  {args.backend_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
