#!/usr/bin/env python3
"""Replace the single Log op in `silero_vad_k2.onnx`.

The k2 Silero VAD model contains one `Log` node inside adaptive normalization.
RKNN on the current RK3568 runtime rejects that op during execution, even though
the model can be converted successfully.

This script replaces:

    y = log(x)

with a piecewise-linear approximation that only uses:

    Sub / Relu / Mul / Add

The approximation is built on powers-of-two knots, which gives a stable fit for
`ln(x)` over the positive range observed in this model.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _const_node(name: str, output_name: str, value: float) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output_name],
        name=name,
        value=helper.make_tensor(
            name=f"{name}_value",
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[float(value)],
        ),
    )


def build_piecewise_log_nodes(
    input_name: str,
    output_name: str,
    prefix: str,
    max_power: int,
) -> list[onnx.NodeProto]:
    # Knots: 1, 2, 4, ..., 2**max_power
    knots = [2.0**i for i in range(max_power + 1)]
    segment_slopes = [
        (math.log(knots[i + 1]) - math.log(knots[i])) / (knots[i + 1] - knots[i])
        for i in range(len(knots) - 1)
    ]

    nodes: list[onnx.NodeProto] = []
    term_outputs: list[str] = []

    # Base term: s0 * relu(x - 1.0)
    base_threshold_name = f"{prefix}/threshold_0"
    nodes.append(_const_node(f"{prefix}/Constant_threshold_0", base_threshold_name, 1.0))
    base_delta_name = f"{prefix}/delta_0"
    nodes.append(
        _const_node(
            f"{prefix}/Constant_delta_0", base_delta_name, float(segment_slopes[0])
        )
    )
    base_sub = f"{prefix}/Sub_0_output_0"
    nodes.append(
        helper.make_node(
            "Sub",
            inputs=[input_name, base_threshold_name],
            outputs=[base_sub],
            name=f"{prefix}/Sub_0",
        )
    )
    base_relu = f"{prefix}/Relu_0_output_0"
    nodes.append(
        helper.make_node(
            "Relu",
            inputs=[base_sub],
            outputs=[base_relu],
            name=f"{prefix}/Relu_0",
        )
    )
    base_term = f"{prefix}/Mul_0_output_0"
    nodes.append(
        helper.make_node(
            "Mul",
            inputs=[base_relu, base_delta_name],
            outputs=[base_term],
            name=f"{prefix}/Mul_0",
        )
    )
    term_outputs.append(base_term)

    # Hinge terms: (s_i - s_{i-1}) * relu(x - t_i)
    prev_slope = segment_slopes[0]
    for i in range(1, len(segment_slopes)):
        threshold = knots[i]
        slope_delta = segment_slopes[i] - prev_slope
        prev_slope = segment_slopes[i]

        threshold_name = f"{prefix}/threshold_{i}"
        delta_name = f"{prefix}/delta_{i}"
        nodes.append(
            _const_node(
                f"{prefix}/Constant_threshold_{i}", threshold_name, float(threshold)
            )
        )
        nodes.append(
            _const_node(f"{prefix}/Constant_delta_{i}", delta_name, float(slope_delta))
        )

        sub_output = f"{prefix}/Sub_{i}_output_0"
        relu_output = f"{prefix}/Relu_{i}_output_0"
        mul_output = f"{prefix}/Mul_{i}_output_0"
        nodes.append(
            helper.make_node(
                "Sub",
                inputs=[input_name, threshold_name],
                outputs=[sub_output],
                name=f"{prefix}/Sub_{i}",
            )
        )
        nodes.append(
            helper.make_node(
                "Relu",
                inputs=[sub_output],
                outputs=[relu_output],
                name=f"{prefix}/Relu_{i}",
            )
        )
        nodes.append(
            helper.make_node(
                "Mul",
                inputs=[relu_output, delta_name],
                outputs=[mul_output],
                name=f"{prefix}/Mul_{i}",
            )
        )
        term_outputs.append(mul_output)

    current = term_outputs[0]
    for i, term in enumerate(term_outputs[1:], start=1):
        add_output = output_name if i == len(term_outputs) - 1 else f"{prefix}/Add_{i}_output_0"
        nodes.append(
            helper.make_node(
                "Add",
                inputs=[current, term],
                outputs=[add_output],
                name=f"{prefix}/Add_{i}",
            )
        )
        current = add_output

    return nodes


def replace_log_with_piecewise(src: Path, dst: Path, max_power: int) -> None:
    model = onnx.load(str(src))

    log_idx = None
    log_node = None
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Log":
            log_idx = idx
            log_node = node
            break

    if log_node is None or log_idx is None:
        raise RuntimeError("No Log node found in model")

    replacement_nodes = build_piecewise_log_nodes(
        input_name=log_node.input[0],
        output_name=log_node.output[0],
        prefix=f"{log_node.name}_pwlin",
        max_power=max_power,
    )

    new_nodes = list(model.graph.node[:log_idx]) + replacement_nodes + list(
        model.graph.node[log_idx + 1 :]
    )
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch Log op in k2 Silero VAD ONNX")
    parser.add_argument("--input", required=True, help="Source ONNX model path")
    parser.add_argument("--output", required=True, help="Patched ONNX model path")
    parser.add_argument(
        "--max-power",
        type=int,
        default=18,
        help="Largest power-of-two knot to use (default: 18 -> 262144)",
    )
    args = parser.parse_args()

    replace_log_with_piecewise(Path(args.input), Path(args.output), args.max_power)
    print(f"Patched model written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
