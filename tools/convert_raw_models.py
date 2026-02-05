#!/usr/bin/env python3
import argparse
import datetime as dt
import importlib.util
import os
import sys
from typing import Any

MODEL_OVERRIDES = {
    "deeplabv3.pb": {
        "input_arrays": ["ImageTensor"],
        "output_arrays": ["SemanticPredictions"],
        "input_shapes": {"ImageTensor": [1, 513, 513, 3]},
        "quantization": {
            "experimental_new_quantizer": True,
            "disable_per_channel": True,
        },
    },
    "mobilenetV1ssd.pb": {
        "input_arrays": ["image_tensor"],
        "output_arrays": [
            "detection_boxes",
            "detection_scores",
            "detection_classes",
            "num_detections",
        ],
        "input_shapes": {"image_tensor": [1, 300, 300, 3]},
        "quantization": {
            "experimental_new_quantizer": True,
        },
    },
    "facenet.pb": {
        "input_arrays": ["input"],
        "output_arrays": ["output"],
        "input_shapes": {"input": [1, 160, 160, 3]},
        "quantization": {
            "experimental_new_quantizer": True,
        },
    },
    "mobilebert_quantized.pb": {
        "input_arrays": ["input_ids", "input_mask", "segment_ids"],
        "output_arrays": ["start_logits", "end_logits"],
        "input_shapes": {
            "input_ids": [1, 384],
            "input_mask": [1, 384],
            "segment_ids": [1, 384],
        },
        "quantization": {
            "experimental_new_quantizer": True,
            "disable_per_channel": True,
        },
    },
    "mobilenet_v2_1.4_224_frozen.pb": {
        "input_arrays": ["input"],
        "output_arrays": ["MobilenetV2/Predictions/Reshape_1"],
        "input_shapes": {"input": [1, 224, 224, 3]},
        "quantization": {
            "experimental_new_quantizer": True,
        },
    },
    "vdsr_model.pb": {
        "input_arrays": ["low_resolution"],
        "output_arrays": ["vdsr_output/add"],
        "input_shapes": {"low_resolution": [1, 48, 48, 1]},
        "quantization": {
            "experimental_new_quantizer": True,
        },
    },
    "vgg16_imageNet_notop.pb": {
        "input_arrays": ["input_2"],
        "output_arrays": ["output_1"],
        "input_shapes": {"input_2": [1, 224, 224, 3]},
        "quantization": {
            "experimental_new_quantizer": True,
        },
    },
}

EXCLUDED_OUTPUT_OPS = {
    "Const",
    "Assign",
    "NoOp",
    "Placeholder",
    "PlaceholderWithDefault",
    "Assert",
    "SaveV2",
    "RestoreV2",
    "ControlTrigger",
}

PREFERRED_OUTPUT_OPS = {
    "Identity",
    "Softmax",
    "Sigmoid",
    "Tanh",
    "Relu",
    "Relu6",
    "BiasAdd",
    "Add",
    "AddV2",
    "MatMul",
    "Conv2D",
    "DepthwiseConv2dNative",
    "FusedBatchNorm",
    "FusedBatchNormV3",
    "BatchNormWithGlobalNormalization",
    "Mul",
    "Sub",
    "ArgMax",
    "ConcatV2",
    "Reshape",
}


def protocol_header() -> list[str]:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return [
        "# Raw model int8 conversion protocol",
        "",
        f"- Timestamp (UTC): {now}",
        f"- Python: {sys.version.split()[0]}",
        f"- Platform: {sys.platform}",
        "",
    ]


def load_graph_def(path: str):
    import tensorflow as tf

    graph_def = tf.compat.v1.GraphDef()
    with open(path, "rb") as handle:
        graph_def.ParseFromString(handle.read())
    return graph_def


def infer_io(graph_def) -> tuple[list[str], dict[str, list[int]], list[str], dict[str, str]]:
    input_arrays = []
    input_shapes: dict[str, list[int]] = {}
    dtype_map: dict[str, str] = {}
    for node in graph_def.node:
        if node.op in {"Placeholder", "PlaceholderWithDefault"}:
            input_arrays.append(node.name)
            shape_attr = node.attr.get("shape")
            dims = []
            if shape_attr is not None:
                for dim in shape_attr.shape.dim:
                    dims.append(dim.size if dim.size > 0 else 1)
            input_shapes[node.name] = dims or [1]
            dtype_attr = node.attr.get("dtype")
            if dtype_attr is not None:
                dtype_map[node.name] = dtype_attr.type

    consumers: set[str] = set()
    for node in graph_def.node:
        for input_name in node.input:
            base = input_name.lstrip("^").split(":")[0]
            consumers.add(base)

    candidate_outputs = [
        node.name
        for node in graph_def.node
        if node.name not in consumers and node.op not in EXCLUDED_OUTPUT_OPS
    ]
    preferred_outputs = [
        node.name
        for node in graph_def.node
        if node.name in candidate_outputs and node.op in PREFERRED_OUTPUT_OPS
    ]
    output_arrays = preferred_outputs or candidate_outputs
    return input_arrays, input_shapes, output_arrays, dtype_map


def build_representative_dataset(
    input_shapes: dict[str, list[int]],
    input_arrays: list[str],
    dtype_map: dict[str, Any],
    sample_count: int,
):
    import numpy as np
    import tensorflow as tf

    def representative_dataset():
        for _ in range(sample_count):
            samples = []
            for name in input_arrays:
                shape = input_shapes[name]
                dtype = tf.dtypes.as_dtype(dtype_map.get(name, tf.float32))
                if dtype == tf.uint8:
                    np_dtype = np.uint8
                elif dtype == tf.int8:
                    np_dtype = np.int8
                elif dtype == tf.int32:
                    np_dtype = np.int32
                elif dtype == tf.int64:
                    np_dtype = np.int64
                else:
                    np_dtype = np.float32
                samples.append(np.zeros(shape, dtype=np_dtype))
            yield samples

    return representative_dataset


def convert_frozen_graph(
    graph_path: str,
    output_path: str,
    input_arrays: list[str],
    output_arrays: list[str],
    input_shapes: dict[str, list[int]],
    int8: bool,
    int8_samples: int,
    quantization_settings: dict[str, bool] | None,
) -> None:
    import tensorflow as tf

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_path,
        input_arrays,
        output_arrays,
        input_shapes=input_shapes,
    )
    if int8:
        if quantization_settings:
            if quantization_settings.get("experimental_new_quantizer"):
                converter.experimental_new_quantizer = True
            if quantization_settings.get("disable_per_channel"):
                converter._experimental_disable_per_channel = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        dtype_map = frozen_graph_input_dtypes(graph_path, input_arrays)
        converter.representative_dataset = build_representative_dataset(
            input_shapes, input_arrays, dtype_map, int8_samples
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        input_dtypes = {tf.dtypes.as_dtype(value) for value in dtype_map.values()}
        if input_dtypes and input_dtypes.issubset(
            {tf.float32, tf.float16, tf.uint8, tf.int8}
        ):
            if tf.uint8 in input_dtypes:
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
            else:
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(output_path, "wb") as handle:
        handle.write(tflite_model)


def convert_frozen_graph_with_fallback(
    graph_path: str,
    output_path: str,
    input_arrays: list[str],
    output_arrays: list[str],
    input_shapes: dict[str, list[int]],
    int8: bool,
    int8_samples: int,
    quantization_settings: dict[str, bool] | None,
) -> tuple[bool, str]:
    try:
        convert_frozen_graph(
            graph_path,
            output_path,
            input_arrays,
            output_arrays,
            input_shapes,
            int8,
            int8_samples,
            quantization_settings,
        )
        return True, "Converted frozen graph with default settings."
    except Exception as exc:
        import tensorflow as tf

        tf.compat.v1.enable_control_flow_v2()
        converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
            graph_path,
            input_arrays,
            output_arrays,
            input_shapes=input_shapes,
        )
        converter.allow_custom_ops = True
        if int8:
            if quantization_settings:
                if quantization_settings.get("experimental_new_quantizer"):
                    converter.experimental_new_quantizer = True
                if quantization_settings.get("disable_per_channel"):
                    converter._experimental_disable_per_channel = True
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            dtype_map = frozen_graph_input_dtypes(graph_path, input_arrays)
            converter.representative_dataset = build_representative_dataset(
                input_shapes, input_arrays, dtype_map, int8_samples
            )
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            input_dtypes = {tf.dtypes.as_dtype(value) for value in dtype_map.values()}
            if input_dtypes and input_dtypes.issubset(
                {tf.float32, tf.float16, tf.uint8, tf.int8}
            ):
                if tf.uint8 in input_dtypes:
                    converter.inference_input_type = tf.uint8
                    converter.inference_output_type = tf.uint8
                else:
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
        try:
            tflite_model = converter.convert()
            with open(output_path, "wb") as handle:
                handle.write(tflite_model)
            return True, "Converted frozen graph with control flow v2 + custom ops."
        except Exception as second_exc:
            return False, f"{exc}; fallback failed: {second_exc}"


def frozen_graph_input_dtypes(graph_path: str, input_arrays: list[str]) -> dict[str, Any]:
    import tensorflow as tf

    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(open(graph_path, "rb").read())
    dtype_map = {}
    for node in graph_def.node:
        if node.name in input_arrays:
            dtype = node.attr.get("dtype")
            if dtype:
                dtype_map[node.name] = tf.dtypes.as_dtype(dtype.type)
    return dtype_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert frozen-graph protobuf models in raw_model to int8 TFLite for LiteRT."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="raw_model",
        help="Directory containing .pb raw models.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/converted/raw_model_int8",
        help="Directory to store converted int8 TFLite models.",
    )
    parser.add_argument(
        "--protocol-path",
        default="protocols/raw_model_int8.md",
        help="Path to write conversion protocol output.",
    )
    parser.add_argument(
        "--int8-samples",
        type=int,
        default=100,
        help="Number of samples to generate for int8 calibration.",
    )
    args = parser.parse_args()

    protocol_lines = protocol_header()

    if not os.path.isdir(args.input_dir):
        protocol_lines.append(f"- Input dir not found: {args.input_dir}")
        write_protocol(args.protocol_path, protocol_lines)
        print(f"Protocol written to {args.protocol_path}")
        return

    if importlib.util.find_spec("tensorflow") is None:
        protocol_lines.append("- TensorFlow import failed: module not installed")
        protocol_lines.append("- Conversion skipped (TensorFlow is required).")
        write_protocol(args.protocol_path, protocol_lines)
        print(f"Protocol written to {args.protocol_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.endswith(".pb"):
            protocol_lines.append(f"## {filename}")
            protocol_lines.append("- Skipped (not a .pb file).")
            protocol_lines.append("")
            continue

        model_path = os.path.join(args.input_dir, filename)
        protocol_lines.append(f"## {filename}")

        override = MODEL_OVERRIDES.get(filename, {})
        quantization_settings = override.get("quantization")
        try:
            graph_def = load_graph_def(model_path)
        except Exception as exc:
            protocol_lines.append(f"- Failed to load graph: {exc}")
            protocol_lines.append("")
            continue

        input_arrays = override.get("input_arrays")
        output_arrays = override.get("output_arrays")
        input_shapes = override.get("input_shapes")

        if not (input_arrays and output_arrays and input_shapes):
            inferred_inputs, inferred_shapes, inferred_outputs, _ = infer_io(graph_def)
            input_arrays = input_arrays or inferred_inputs
            output_arrays = output_arrays or inferred_outputs
            input_shapes = input_shapes or inferred_shapes
            protocol_lines.append("- Used inferred IO configuration.")
        else:
            protocol_lines.append("- Used predefined IO configuration.")
        if quantization_settings:
            protocol_lines.append(f"- Quantization settings: {quantization_settings}")

        protocol_lines.append(f"- Inputs: {input_arrays}")
        protocol_lines.append(f"- Input shapes: {input_shapes}")
        protocol_lines.append(f"- Outputs: {output_arrays}")

        if not input_arrays or not output_arrays:
            protocol_lines.append("- Missing IO arrays; conversion skipped.")
            protocol_lines.append("")
            continue

        output_path = os.path.join(
            args.output_dir, os.path.splitext(filename)[0] + "_int8.tflite"
        )

        try:
            converted, message = convert_frozen_graph_with_fallback(
                model_path,
                output_path,
                input_arrays,
                output_arrays,
                input_shapes,
                True,
                args.int8_samples,
                quantization_settings,
            )
            if converted:
                protocol_lines.append(f"- Converted to {output_path}")
                protocol_lines.append(f"- Notes: {message}")
            else:
                protocol_lines.append(f"- Conversion failed: {message}")
        except Exception as exc:
            protocol_lines.append(f"- Conversion failed: {exc}")
        protocol_lines.append("")

    write_protocol(args.protocol_path, protocol_lines)
    print(f"Protocol written to {args.protocol_path}")


def write_protocol(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    main()
