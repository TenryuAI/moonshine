#!/usr/bin/env python3
"""Split k2 Silero VAD into CPU frontend and backend ONNX models."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import utils


FRONTEND_OUTPUT = "/Concat_output_0"
BACKEND_INPUTS = [FRONTEND_OUTPUT, "h", "c"]
BACKEND_OUTPUTS = ["prob", "new_h", "new_c"]
BACKEND_FEATURE_SHAPE = [1, 258, 8]


def split_model(src: Path, frontend_out: Path, backend_out: Path) -> None:
    # Frontend: x -> /Concat_output_0
    utils.extract_model(
        str(src),
        str(frontend_out),
        input_names=["x"],
        output_names=[FRONTEND_OUTPUT],
    )

    # Backend: /Concat_output_0 + recurrent state -> outputs
    utils.extract_model(
        str(src),
        str(backend_out),
        input_names=BACKEND_INPUTS,
        output_names=BACKEND_OUTPUTS,
    )

    frontend_model = onnx.load(str(frontend_out))
    backend_model = onnx.load(str(backend_out))

    for model in (frontend_model, backend_model):
        for value_info in list(model.graph.input) + list(model.graph.output) + list(
            model.graph.value_info
        ):
            if value_info.name == FRONTEND_OUTPUT:
                dims = value_info.type.tensor_type.shape.dim
                del dims[:]
                for value in BACKEND_FEATURE_SHAPE:
                    dims.add().dim_value = value

    onnx.checker.check_model(frontend_model)
    onnx.checker.check_model(backend_model)
    onnx.save(frontend_model, str(frontend_out))
    onnx.save(backend_model, str(backend_out))


def main() -> int:
    parser = argparse.ArgumentParser(description="Split k2 Silero VAD ONNX")
    parser.add_argument("--input", required=True, help="Source ONNX model path")
    parser.add_argument(
        "--frontend-output",
        required=True,
        help="Output path for CPU frontend ONNX",
    )
    parser.add_argument(
        "--backend-output",
        required=True,
        help="Output path for backend ONNX",
    )
    args = parser.parse_args()

    split_model(
        Path(args.input),
        Path(args.frontend_output),
        Path(args.backend_output),
    )
    print(f"Frontend model written to: {args.frontend_output}")
    print(f"Backend model written to: {args.backend_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
