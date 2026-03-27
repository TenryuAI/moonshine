#!/usr/bin/env python3
"""Replace the fixed ReflectPad in `silero_vad_k2_nolog.onnx`.

The model uses a single 1D reflect pad:

    input  shape: [1, 1, 512]
    pads:   [0, 0, 96, 0, 0, 96]
    output: [1, 1, 704]

On the current RK3568 runtime, the converted RKNN model fails around the first
Conv after this reflect pad. This script rewrites the reflect pad into explicit
Slice + Concat nodes with fixed indices.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _const_int64(name: str, output_name: str, values: list[int]) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output_name],
        name=name,
        value=helper.make_tensor(
            name=f"{name}_value",
            data_type=TensorProto.INT64,
            dims=[len(values)],
            vals=values,
        ),
    )


def _make_slice(
    prefix: str,
    data: str,
    output: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
    steps: list[int],
) -> list[onnx.NodeProto]:
    starts_name = f"{prefix}/starts"
    ends_name = f"{prefix}/ends"
    axes_name = f"{prefix}/axes"
    steps_name = f"{prefix}/steps"
    return [
        _const_int64(f"{prefix}/Constant_starts", starts_name, starts),
        _const_int64(f"{prefix}/Constant_ends", ends_name, ends),
        _const_int64(f"{prefix}/Constant_axes", axes_name, axes),
        _const_int64(f"{prefix}/Constant_steps", steps_name, steps),
        helper.make_node(
            "Slice",
            inputs=[data, starts_name, ends_name, axes_name, steps_name],
            outputs=[output],
            name=prefix,
        ),
    ]


def replace_reflect_pad(src: Path, dst: Path) -> None:
    model = onnx.load(str(src))

    pad_idx = None
    pad_node = None
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Pad" and node.name == "/feature_extractor/Pad":
            pad_idx = idx
            pad_node = node
            break

    if pad_node is None or pad_idx is None:
        raise RuntimeError("Target reflect Pad node not found")

    data_input = pad_node.input[0]
    pad_output = pad_node.output[0]
    prefix = "/feature_extractor/Pad_rewrite"

    left_output = f"{prefix}/left_output_0"
    center_output = f"{prefix}/center_output_0"
    right_output = f"{prefix}/right_output_0"

    replacement_nodes: list[onnx.NodeProto] = []

    # Left reflect pad: x[..., 96:0:-1] -> length 96
    replacement_nodes.extend(
        _make_slice(
            f"{prefix}/Slice_left",
            data_input,
            left_output,
            starts=[96],
            ends=[0],
            axes=[2],
            steps=[-1],
        )
    )

    # Center identity: x[..., 0:512:1]
    replacement_nodes.extend(
        _make_slice(
            f"{prefix}/Slice_center",
            data_input,
            center_output,
            starts=[0],
            ends=[512],
            axes=[2],
            steps=[1],
        )
    )

    # Right reflect pad: x[..., 510:414:-1] -> length 96
    replacement_nodes.extend(
        _make_slice(
            f"{prefix}/Slice_right",
            data_input,
            right_output,
            starts=[510],
            ends=[414],
            axes=[2],
            steps=[-1],
        )
    )

    replacement_nodes.append(
        helper.make_node(
            "Concat",
            inputs=[left_output, center_output, right_output],
            outputs=[pad_output],
            name=f"{prefix}/Concat",
            axis=2,
        )
    )

    new_nodes = list(model.graph.node[:pad_idx]) + replacement_nodes + list(
        model.graph.node[pad_idx + 1 :]
    )
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch fixed reflect pad in k2 Silero VAD ONNX")
    parser.add_argument("--input", required=True, help="Source ONNX model path")
    parser.add_argument("--output", required=True, help="Patched ONNX model path")
    args = parser.parse_args()

    replace_reflect_pad(Path(args.input), Path(args.output))
    print(f"Patched model written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
