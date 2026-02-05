#!/usr/bin/env python3
import argparse
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
from typing import Any


def protocol_header() -> list[str]:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return [
        "# Raw model int8 runtime protocol",
        "",
        f"- Timestamp (UTC): {now}",
        f"- Python: {sys.version.split()[0]}",
        f"- Platform: {sys.platform}",
        "",
    ]


def load_interpreter(model_path: str):
    if importlib.util.find_spec("tflite_runtime") is not None:
        from tflite_runtime.interpreter import Interpreter

        return "tflite_runtime", Interpreter(model_path=model_path)

    import tensorflow as tf

    return "tensorflow", tf.lite.Interpreter(
        model_path=model_path, experimental_delegates=[]
    )


def build_input_tensor(shape: list[int], dtype: Any):
    import numpy as np

    if dtype == np.uint8:
        return np.zeros(shape, dtype=np.uint8)
    if dtype == np.int8:
        return np.zeros(shape, dtype=np.int8)
    if dtype == np.int32:
        return np.zeros(shape, dtype=np.int32)
    return np.zeros(shape, dtype=np.float32)


def run_model(interpreter) -> tuple[list[str], list[str]]:
    import numpy as np

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_summaries = []
    output_summaries = []

    for detail in input_details:
        shape = [int(dim) if dim > 0 else 1 for dim in detail["shape"]]
        tensor = build_input_tensor(shape, detail["dtype"])
        interpreter.set_tensor(detail["index"], tensor)
        input_summaries.append(
            f"{detail['name']}: shape={shape}, dtype={np.dtype(detail['dtype']).name}"
        )

    interpreter.invoke()

    for detail in output_details:
        data = interpreter.get_tensor(detail["index"])
        output_summaries.append(
            f"{detail['name']}: shape={list(data.shape)}, dtype={data.dtype.name}"
        )

    return input_summaries, output_summaries


def write_protocol(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_single_model(model_path: str) -> dict[str, Any]:
    os.environ.setdefault("TFLITE_ENABLE_XNNPACK", "0")
    os.environ.setdefault("TF_LITE_DISABLE_XNNPACK", "1")
    runtime_label, interpreter = load_interpreter(model_path)
    interpreter.allocate_tensors()
    inputs, outputs = run_model(interpreter)
    return {
        "runtime": runtime_label,
        "inputs": inputs,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run int8 TFLite models with LiteRT/TFLite runtime on x86 Linux."
    )
    parser.add_argument(
        "--model-dir",
        default="models/converted/raw_model_int8",
        help="Directory containing int8 .tflite models.",
    )
    parser.add_argument(
        "--protocol-path",
        default="protocols/raw_model_int8_runtime.md",
        help="Path to write runtime protocol output.",
    )
    parser.add_argument(
        "--single-model",
        help="Internal helper to run a single model and emit JSON results.",
    )
    args = parser.parse_args()

    if args.single_model:
        try:
            result = run_single_model(args.single_model)
            print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
            return
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            sys.exit(1)

    protocol_lines = protocol_header()

    if not os.path.isdir(args.model_dir):
        protocol_lines.append(f"- Model dir not found: {args.model_dir}")
        write_protocol(args.protocol_path, protocol_lines)
        print(f"Protocol written to {args.protocol_path}")
        return

    runtime_label = None
    if importlib.util.find_spec("tflite_runtime") is None:
        protocol_lines.append("- tflite_runtime not available; falling back to tensorflow.")
    if importlib.util.find_spec("tensorflow") is None:
        protocol_lines.append("- TensorFlow not installed; runtime validation skipped.")
        write_protocol(args.protocol_path, protocol_lines)
        print(f"Protocol written to {args.protocol_path}")
        return

    for filename in sorted(os.listdir(args.model_dir)):
        if not filename.endswith(".tflite"):
            continue
        model_path = os.path.join(args.model_dir, filename)
        protocol_lines.append(f"## {filename}")
        result = subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--single-model",
                model_path,
            ],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "TFLITE_ENABLE_XNNPACK": "0",
                "TF_LITE_DISABLE_XNNPACK": "1",
            },
        )
        runtime_label = "unknown"
        if result.stdout:
            try:
                payload = json.loads(result.stdout.strip().splitlines()[-1])
            except json.JSONDecodeError:
                payload = None
        else:
            payload = None
        if payload and payload.get("ok"):
            model_result = payload["result"]
            runtime_label = model_result.get("runtime", runtime_label)
            protocol_lines.append(f"- Runtime: {runtime_label}")
            protocol_lines.append("- Inputs:")
            protocol_lines.extend([f"  - {line}" for line in model_result["inputs"]])
            protocol_lines.append("- Outputs:")
            protocol_lines.extend([f"  - {line}" for line in model_result["outputs"]])
        else:
            error = payload.get("error") if payload else "unknown error"
            stderr_summary = result.stderr.strip()
            protocol_lines.append(
                f"- Runtime: {runtime_label}; failed to run: {error}"
            )
            protocol_lines.append(f"- Return code: {result.returncode}")
            if stderr_summary:
                protocol_lines.append(f"- Stderr: {stderr_summary}")
        protocol_lines.append("")

    write_protocol(args.protocol_path, protocol_lines)
    print(f"Protocol written to {args.protocol_path}")


if __name__ == "__main__":
    main()
