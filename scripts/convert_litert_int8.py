#!/usr/bin/env python3
"""Convert frozen TensorFlow GraphDef models to INT8 LiteRT/TFLite models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import tensorflow as tf


@dataclass
class ModelInfo:
    input_arrays: List[str]
    input_shapes: List[List[int]]
    output_arrays: List[str]
    input_dtypes: List[tf.DType]
    has_unknown_shapes: bool = False


def _load_graph_def(model_path: Path) -> tf.compat.v1.GraphDef:
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(str(model_path), "rb") as graph_file:
        graph_def.ParseFromString(graph_file.read())
    return graph_def


def _collect_consumers(graph_def: tf.compat.v1.GraphDef) -> set[str]:
    consumers: set[str] = set()
    for node in graph_def.node:
        for input_name in node.input:
            consumers.add(input_name.split(":")[0])
    return consumers


def _infer_input_shapes(placeholders: Sequence[tf.compat.v1.NodeDef]) -> Tuple[List[List[int]], bool]:
    shapes: List[List[int]] = []
    has_unknown = False
    for node in placeholders:
        shape_attr = node.attr.get("shape")
        if not shape_attr:
            shapes.append([])
            has_unknown = True
            continue
        dims = []
        for dim in shape_attr.shape.dim:
            if dim.size and dim.size > 0:
                dims.append(int(dim.size))
            else:
                dims.append(-1)
                has_unknown = True
        shapes.append(dims or [])
        if not dims:
            has_unknown = True
    return shapes, has_unknown


def _infer_model_info(graph_def: tf.compat.v1.GraphDef) -> ModelInfo:
    placeholders = [node for node in graph_def.node if node.op == "Placeholder"]
    if not placeholders:
        raise ValueError("No Placeholder ops found; please provide --inputs and --input-shapes.")

    input_arrays = [node.name for node in placeholders]
    input_dtypes = [tf.as_dtype(node.attr["dtype"].type) for node in placeholders]
    input_shapes, has_unknown = _infer_input_shapes(placeholders)

    consumers = _collect_consumers(graph_def)
    output_arrays = [
        node.name
        for node in graph_def.node
        if node.name not in consumers and node.op not in {"Const", "NoOp", "Assign"}
    ]
    if not output_arrays:
        raise ValueError("No output nodes inferred; please provide --outputs.")

    return ModelInfo(
        input_arrays=input_arrays,
        input_shapes=input_shapes,
        output_arrays=output_arrays,
        input_dtypes=input_dtypes,
        has_unknown_shapes=has_unknown,
    )


def _parse_shapes(shape_args: Sequence[str]) -> List[List[int]]:
    shapes: List[List[int]] = []
    for shape_arg in shape_args:
        dims = [int(dim) for dim in shape_arg.split(",") if dim]
        if not dims:
            raise ValueError(f"Invalid --input-shape '{shape_arg}'.")
        shapes.append(dims)
    return shapes


def _representative_dataset(
    input_shapes: Sequence[Sequence[int]],
    input_dtypes: Sequence[tf.DType],
    num_samples: int,
    seed: int,
) -> Iterable[List[np.ndarray]]:
    rng = np.random.default_rng(seed)
    for _ in range(num_samples):
        batch = []
        for shape, dtype in zip(input_shapes, input_dtypes, strict=True):
            if dtype == tf.uint8:
                batch.append(rng.integers(0, 256, size=shape, dtype=np.uint8))
            elif dtype == tf.int8:
                batch.append(rng.integers(-128, 128, size=shape, dtype=np.int8))
            elif dtype == tf.int32:
                batch.append(rng.integers(0, 2, size=shape, dtype=np.int32))
            else:
                batch.append(rng.uniform(-1.0, 1.0, size=shape).astype(np.float32))
        yield batch


def _convert_model(
    model_path: Path,
    output_path: Path,
    model_info: ModelInfo,
    num_samples: int,
    seed: int,
) -> None:
    tf.compat.v1.enable_control_flow_v2()
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        str(model_path),
        model_info.input_arrays,
        model_info.output_arrays,
        input_shapes=dict(zip(model_info.input_arrays, model_info.input_shapes, strict=True)),
    )

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(
        model_info.input_shapes,
        model_info.input_dtypes,
        num_samples,
        seed,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    if hasattr(converter, "experimental_new_quantizer"):
        converter.experimental_new_quantizer = True

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)

def _get_tensor_quantization(tensor_details: dict) -> Tuple[List[float], List[int], int]:
    quant_params = tensor_details.get("quantization_parameters", {})
    scales = quant_params.get("scales", [])
    zero_points = quant_params.get("zero_points", [])
    quant_dim = quant_params.get("quantized_dimension", 0)
    return list(scales), list(zero_points), int(quant_dim)


def _check_same_quantization(input_details: dict, output_details: dict) -> List[str]:
    errors = []
    if input_details["dtype"] != output_details["dtype"]:
        errors.append("dtype mismatch")
    if input_details.get("quantization") != output_details.get("quantization"):
        errors.append("quantization mismatch")
    return errors


def _verify_tflite_model(output_path: Path) -> List[str]:
    interpreter = tf.lite.Interpreter(model_path=str(output_path))
    try:
        interpreter.allocate_tensors()
    except RuntimeError as error:
        return [f"Failed to allocate tensors for verification: {error}"]

    tensor_details = {detail["index"]: detail for detail in interpreter.get_tensor_details()}
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    findings: List[str] = []
    for detail in input_details + output_details:
        if detail["dtype"] != np.int8:
            findings.append(f"{detail['name']} is not int8 (dtype={detail['dtype']}).")

    op_details = interpreter._get_ops_details()
    same_quant_ops = {
        "AVERAGE_POOL_2D",
        "MAX_POOL_2D",
        "CONCATENATION",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "SPACE_TO_DEPTH",
        "PAD",
        "PADV2",
        "GATHER",
        "SLICE",
        "SQUEEZE",
        "TRANSPOSE",
        "BATCH_TO_SPACE_ND",
        "SPACE_TO_BATCH_ND",
    }
    fixed_output_specs = {
        "LOGISTIC": (1 / 256, -128),
        "SOFTMAX": (1 / 256, -128),
        "TANH": (1 / 128, 0),
        "L2_NORMALIZATION": (1 / 128, 0),
        "LOG_SOFTMAX": (16 / 256, 127),
    }
    per_axis_ops = {
        "CONV_2D": 0,
        "DEPTHWISE_CONV_2D": 3,
        "FULLY_CONNECTED": 0,
    }

    for op in op_details:
        op_name = op["op_name"]
        inputs = [tensor_details[idx] for idx in op.get("inputs", []) if idx in tensor_details]
        outputs = [tensor_details[idx] for idx in op.get("outputs", []) if idx in tensor_details]

        if op_name in same_quant_ops and inputs and outputs:
            issues = _check_same_quantization(inputs[0], outputs[0])
            if issues:
                findings.append(f"{op_name} requires matching input/output quantization: {', '.join(issues)}.")

        if op_name in fixed_output_specs and outputs:
            expected_scale, expected_zero_point = fixed_output_specs[op_name]
            actual_scale, actual_zero_point = outputs[0].get("quantization", (None, None))
            if (actual_scale, actual_zero_point) != (expected_scale, expected_zero_point):
                findings.append(
                    f"{op_name} output quantization should be ({expected_scale}, {expected_zero_point}), "
                    f"got ({actual_scale}, {actual_zero_point})."
                )

        if op_name in per_axis_ops and len(inputs) >= 3:
            weight_details = inputs[1]
            bias_details = inputs[2]
            weight_scales, weight_zero_points, weight_axis = _get_tensor_quantization(weight_details)
            if len(weight_scales) <= 1:
                findings.append(f"{op_name} weights are not per-axis quantized.")
            if any(zp != 0 for zp in weight_zero_points):
                findings.append(f"{op_name} weights zero_point must be 0.")
            if weight_axis != per_axis_ops[op_name]:
                findings.append(f"{op_name} weight quantized_dimension should be {per_axis_ops[op_name]}.")

            input_scale, input_zero_point = inputs[0].get("quantization", (None, None))
            bias_scales, bias_zero_points, _ = _get_tensor_quantization(bias_details)
            if input_scale is not None and weight_scales:
                expected_bias_scales = [input_scale * scale for scale in weight_scales]
                if len(bias_scales) == len(expected_bias_scales):
                    for expected, actual in zip(expected_bias_scales, bias_scales):
                        if not np.isclose(expected, actual, rtol=1e-3, atol=1e-5):
                            findings.append(f"{op_name} bias scale mismatch (expected {expected}, got {actual}).")
                            break
            if any(zp != 0 for zp in bias_zero_points):
                findings.append(f"{op_name} bias zero_point must be 0.")

            if input_zero_point is not None and not (-128 <= input_zero_point <= 127):
                findings.append(f"{op_name} input zero_point out of int8 range.")

    return findings


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert frozen GraphDef models to INT8 LiteRT/TFLite models.")
    parser.add_argument("--input", type=Path, required=True, help="Path to a frozen .pb model.")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output .tflite file.")
    parser.add_argument("--inputs", type=str, default=None, help="Comma-separated input node names.")
    parser.add_argument("--outputs", type=str, default=None, help="Comma-separated output node names.")
    parser.add_argument(
        "--input-shapes",
        type=str,
        nargs="*",
        default=None,
        help="Input shapes as comma-separated dims per input (e.g. 1,224,224,3).")
    parser.add_argument("--samples", type=int, default=200, help="Number of representative samples to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for representative dataset.")
    parser.add_argument("--verify", action="store_true", help="Report INT8 quantization spec checks for the output model.")
    parser.add_argument(
        "--verify-strict",
        action="store_true",
        help="Fail the conversion if quantization spec checks report issues.",
    )
    parser.add_argument("--show-info", action="store_true", help="Print inferred inputs/outputs and exit.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    graph_def = _load_graph_def(args.input)
    model_info = _infer_model_info(graph_def)

    if args.show_info:
        print("Inferred inputs:", ", ".join(model_info.input_arrays))
        print("Inferred input shapes:", ", ".join(map(str, model_info.input_shapes)))
        print("Inferred input dtypes:", ", ".join(dtype.name for dtype in model_info.input_dtypes))
        print("Inferred outputs:", ", ".join(model_info.output_arrays))
        return

    if args.inputs:
        model_info.input_arrays = [name.strip() for name in args.inputs.split(",") if name.strip()]
    if args.outputs:
        model_info.output_arrays = [name.strip() for name in args.outputs.split(",") if name.strip()]
    if args.input_shapes:
        model_info.input_shapes = _parse_shapes(args.input_shapes)

    if model_info.has_unknown_shapes and not args.input_shapes:
        raise ValueError("Input shapes contain unknown dimensions; please provide --input-shapes explicitly.")

    if len(model_info.input_arrays) != len(model_info.input_shapes):
        raise ValueError("The number of input arrays must match the number of input shapes.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _convert_model(args.input, args.output, model_info, args.samples, args.seed)

    if args.verify:
        issues = _verify_tflite_model(args.output)
        if issues:
            issue_text = "\n".join(f"- {issue}" for issue in issues)
            print(f"Quantization verification issues detected:\\n{issue_text}")
            if args.verify_strict:
                raise RuntimeError(f"Quantization verification failed:\\n{issue_text}")


if __name__ == "__main__":
    main()
