#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile

MODEL_SPECS = [
    {
        "name": "resnet_v2_50",
        "downloads": [
            {
                "label": "tensorflow_checkpoint",
                "urls": [
                    "https://storage.googleapis.com/download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz",
                    "https://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz",
                    "http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz",
                ],
            }
        ],
        "conversion": {
            "type": "tf_slim_checkpoint",
            "notes": (
                "The archive contains TF-Slim checkpoints only. Export a frozen graph "
                "with the TF-Slim export scripts before converting to TFLite."
            ),
        },
    },
    {
        "name": "mobilenet_v2",
        "downloads": [
            {
                "label": "tensorflow_checkpoint",
                "urls": [
                    "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz",
                ],
            },
            {
                "label": "caffe_model",
                "urls": [
                    "https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2.caffemodel",
                ],
            },
        ],
        "conversion": {
            "type": "use_tflite_if_present",
            "notes": "Some MobileNet V2 archives already include a .tflite file.",
        },
    },
    {
        "name": "mobilenet_v1_ssd",
        "downloads": [
            {
                "label": "tensorflow_frozen_graph",
                "urls": [
                    "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz",
                    "https://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz",
                    "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz",
                ],
            }
        ],
        "conversion": {
            "type": "frozen_graph",
            "input_arrays": ["image_tensor"],
            "output_arrays": [
                "detection_boxes",
                "detection_scores",
                "detection_classes",
                "num_detections",
            ],
            "input_shapes": {"image_tensor": [1, 300, 300, 3]},
            "frozen_graph_name": "frozen_inference_graph.pb",
        },
    },
    {
        "name": "vgg16",
        "downloads": [
            {
                "label": "tensorflow_checkpoint",
                "urls": [
                    "https://storage.googleapis.com/download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
                    "https://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
                    "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
                ],
            },
            {
                "label": "caffe_model",
                "urls": [
                    "https://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel",
                    "http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel",
                ],
            },
        ],
        "conversion": {
            "type": "tf_slim_checkpoint",
            "notes": (
                "Export a frozen graph with TF-Slim before converting to TFLite."
            ),
        },
    },
    {
        "name": "deeplabv3_mnv2",
        "downloads": [
            {
                "label": "tensorflow_frozen_graph",
                "urls": [
                    "https://storage.googleapis.com/download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz",
                    "https://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz",
                    "http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz",
                ],
            }
        ],
        "conversion": {
            "type": "frozen_graph",
            "input_arrays": ["ImageTensor"],
            "output_arrays": ["SemanticPredictions"],
            "input_shapes": {"ImageTensor": [1, 513, 513, 3]},
            "frozen_graph_name": "frozen_inference_graph.pb",
        },
    },
    {
        "name": "vdsr",
        "downloads": [
            {
                "label": "tensorflow_checkpoint",
                "urls": [
                    "https://drive.google.com/file/d/1wiej51wFY0oYsoKF7gGiWZT5_t5mFt0f/view?usp=sharing"
                ],
                "needs_manual_download": True,
                "notes": "Google Drive link requires manual download.",
            }
        ],
        "conversion": {
            "type": "manual",
            "notes": "Requires the TensorFlow implementation from VDSR-TensorFlow.",
        },
    },
    {
        "name": "yolov3",
        "downloads": [
            {
                "label": "darknet_weights",
                "urls": [
                    "https://pjreddie.com/media/files/yolov3.weights",
                ],
            }
        ],
        "conversion": {
            "type": "manual",
            "notes": (
                "Convert Darknet weights to a TensorFlow frozen graph "
                "before converting to TFLite."
            ),
        },
    },
    {
        "name": "facenet",
        "downloads": [
            {
                "label": "tensorflow_model",
                "urls": [
                    "https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-"
                ],
                "needs_manual_download": True,
                "notes": "Google Drive link requires manual download.",
            }
        ],
        "conversion": {
            "type": "manual",
            "notes": "Use the provided Facenet preprocessing scripts to export a SavedModel.",
        },
    },
    {
        "name": "mobilebert",
        "downloads": [
            {
                "label": "saved_model",
                "urls": [
                    "https://storage.googleapis.com/cloud-tpu-checkpoints/mobilebert/mobilebert_squad_savedmodels.tar.gz",
                ],
            }
        ],
        "conversion": {
            "type": "saved_model",
            "saved_model_subdir": "mobilebert_squad_savedmodels/float",
        },
    },
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def filename_from_url(url: str) -> str:
    return os.path.basename(urllib.parse.urlparse(url).path)


def sha256sum(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest_path: str, timeout: int) -> None:
    ensure_dir(os.path.dirname(dest_path))
    if os.path.exists(dest_path):
        return
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; LiteRT-Downloader/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response, open(
        dest_path, "wb"
    ) as handle:
        shutil.copyfileobj(response, handle)


def download_with_fallbacks(
    urls: list[str], dest_path: str, timeout: int, use_curl: bool
) -> str:
    errors: list[str] = []
    for url in urls:
        try:
            download(url, dest_path, timeout)
            return url
        except Exception as exc:
            errors.append(f"{url} -> {exc}")
            if use_curl and shutil.which("curl"):
                curl_result = subprocess.run(
                    [
                        "curl",
                        "-L",
                        "-o",
                        dest_path,
                        "-A",
                        "Mozilla/5.0 (compatible; LiteRT-Downloader/1.0)",
                        url,
                    ],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if curl_result.returncode == 0 and os.path.exists(dest_path):
                    return url
                errors.append(
                    f"curl failed for {url}: {curl_result.stderr.strip() or 'unknown error'}"
                )
    raise RuntimeError("; ".join(errors))


def extract_archive(archive_path: str, dest_dir: str) -> None:
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(dest_dir)
        return
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(dest_dir)
        return


def find_file(root: str, filename: str) -> str | None:
    for base, _, files in os.walk(root):
        if filename in files:
            return os.path.join(base, filename)
    return None


def find_saved_model_dir(root: str) -> str | None:
    for base, _, files in os.walk(root):
        if "saved_model.pb" in files:
            return base
    return None


def build_representative_dataset_saved_model(
    saved_model_dir: str, sample_count: int
):
    import numpy as np
    import tensorflow as tf

    loaded = tf.saved_model.load(saved_model_dir, tags=["serve"])
    signature = loaded.signatures.get("serving_default")
    if signature is None:
        signature = next(iter(loaded.signatures.values()))
    inputs = signature.structured_input_signature[1]
    if not inputs:
        raise ValueError("SavedModel signature does not expose inputs.")

    def representative_dataset():
        for _ in range(sample_count):
            sample = {}
            for name, spec in inputs.items():
                shape = [
                    dim if dim is not None else 1 for dim in spec.shape.as_list()
                ]
                sample[name] = np.zeros(shape, dtype=spec.dtype.as_numpy_dtype)
            yield sample

    return representative_dataset


def build_representative_dataset_frozen_graph(
    input_shapes: dict[str, list[int]],
    input_arrays: list[str],
    dtype_map: dict[str, str],
    sample_count: int,
):
    import numpy as np

    def representative_dataset():
        for _ in range(sample_count):
            samples = []
            for name in input_arrays:
                shape = input_shapes[name]
                dtype_name = dtype_map.get(name, "float32")
                if dtype_name == "uint8":
                    dtype = np.uint8
                else:
                    dtype = np.float32
                samples.append(np.zeros(shape, dtype=dtype))
            yield samples

    return representative_dataset


def frozen_graph_input_dtypes(graph_path: str, input_arrays: list[str]) -> dict[str, str]:
    import tensorflow as tf

    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(open(graph_path, "rb").read())
    dtype_map = {}
    for node in graph_def.node:
        if node.name in input_arrays:
            dtype = node.attr.get("dtype")
            if dtype:
                dtype_map[node.name] = tf.dtypes.as_dtype(dtype.type).name
    return dtype_map


def convert_saved_model(
    saved_model_dir: str,
    output_path: str,
    int8: bool,
    int8_samples: int,
) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = build_representative_dataset_saved_model(
            saved_model_dir, int8_samples
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(output_path, "wb") as handle:
        handle.write(tflite_model)


def convert_frozen_graph(
    graph_path: str,
    output_path: str,
    input_arrays: list[str],
    output_arrays: list[str],
    input_shapes: dict[str, list[int]],
    int8: bool,
    int8_samples: int,
) -> None:
    import tensorflow as tf

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_path,
        input_arrays,
        output_arrays,
        input_shapes=input_shapes,
    )
    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        dtype_map = frozen_graph_input_dtypes(graph_path, input_arrays)
        converter.representative_dataset = build_representative_dataset_frozen_graph(
            input_shapes, input_arrays, dtype_map, int8_samples
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if any(dtype == "uint8" for dtype in dtype_map.values()):
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        else:
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(output_path, "wb") as handle:
        handle.write(tflite_model)


def convert_frozen_graph_with_fallbacks(
    graph_path: str,
    output_path: str,
    input_arrays: list[str],
    output_arrays: list[str],
    input_shapes: dict[str, list[int]],
    int8: bool,
    int8_samples: int,
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
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            dtype_map = frozen_graph_input_dtypes(graph_path, input_arrays)
            converter.representative_dataset = build_representative_dataset_frozen_graph(
                input_shapes, input_arrays, dtype_map, int8_samples
            )
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            if any(dtype == "uint8" for dtype in dtype_map.values()):
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


def process_model(
    spec: dict,
    source_root: str,
    output_root: str,
    force: bool,
    timeout: int,
    use_curl: bool,
    int8: bool,
    int8_samples: int,
) -> list[str]:
    name = spec["name"]
    model_source_dir = os.path.join(source_root, name)
    model_output_dir = os.path.join(output_root, name)
    ensure_dir(model_source_dir)
    ensure_dir(model_output_dir)
    notes = []

    for item in spec["downloads"]:
        if item.get("needs_manual_download"):
            urls = ", ".join(item["urls"])
            notes.append(f"Manual download required for {item['label']}: {urls}")
            continue
        urls = item["urls"]
        filename = filename_from_url(urls[0])
        dest_path = os.path.join(model_source_dir, filename)
        if force and os.path.exists(dest_path):
            os.remove(dest_path)
        try:
            selected_url = download_with_fallbacks(urls, dest_path, timeout, use_curl)
            extract_archive(dest_path, model_source_dir)
            checksum = sha256sum(dest_path)
            notes.append(f"Downloaded {filename} ({checksum[:12]}) from {selected_url}")
        except Exception as exc:
            notes.append(f"Download failed for {filename}: {exc}")

    conversion = spec.get("conversion", {})
    conversion_type = conversion.get("type")
    if conversion_type == "saved_model":
        saved_model_subdir = conversion["saved_model_subdir"]
        saved_model_dir = os.path.join(model_source_dir, saved_model_subdir)
        output_path = os.path.join(model_output_dir, f"{name}.tflite")
        resolved_saved_model_dir = (
            saved_model_dir
            if os.path.isdir(saved_model_dir)
            else find_saved_model_dir(model_source_dir)
        )
        if resolved_saved_model_dir:
            try:
                convert_saved_model(
                    resolved_saved_model_dir,
                    output_path,
                    int8,
                    int8_samples,
                )
                notes.append(f"Converted SavedModel to {output_path}")
            except ModuleNotFoundError:
                notes.append("TensorFlow not installed; skipping SavedModel conversion.")
            except Exception as exc:
                notes.append(f"SavedModel conversion failed: {exc}")
        else:
            notes.append(f"SavedModel directory not found under {model_source_dir}")
    elif conversion_type == "frozen_graph":
        graph_path = find_file(model_source_dir, conversion["frozen_graph_name"])
        output_path = os.path.join(model_output_dir, f"{name}.tflite")
        if graph_path:
            try:
                converted, message = convert_frozen_graph_with_fallbacks(
                    graph_path,
                    output_path,
                    conversion["input_arrays"],
                    conversion["output_arrays"],
                    conversion["input_shapes"],
                    int8,
                    int8_samples,
                )
                if converted:
                    notes.append(f"Converted frozen graph to {output_path}. {message}")
                else:
                    notes.append(f"Frozen graph conversion failed: {message}")
            except ModuleNotFoundError:
                notes.append("TensorFlow not installed; skipping frozen graph conversion.")
            except Exception as exc:
                notes.append(f"Frozen graph conversion failed: {exc}")
        else:
            notes.append(
                f"Frozen graph {conversion['frozen_graph_name']} not found in {model_source_dir}"
            )
    elif conversion_type == "use_tflite_if_present":
        tflite_path = None
        for base, _, files in os.walk(model_source_dir):
            for filename in files:
                if filename.endswith(".tflite"):
                    tflite_path = os.path.join(base, filename)
                    break
            if tflite_path:
                break
        if tflite_path:
            output_path = os.path.join(model_output_dir, os.path.basename(tflite_path))
            shutil.copy2(tflite_path, output_path)
            notes.append(f"Copied existing TFLite model to {output_path}")
        else:
            notes.append("No .tflite file found in the MobileNet V2 archive.")
    elif conversion_type == "manual":
        notes.append(conversion.get("notes", "Manual conversion required."))
    elif conversion_type == "tf_slim_checkpoint":
        notes.append(conversion.get("notes", "Export a frozen graph before conversion."))

    return notes


def protocol_header() -> list[str]:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return [
        "# Conversion protocol",
        "",
        f"- Timestamp (UTC): {now}",
        f"- Python: {sys.version.split()[0]}",
        f"- Platform: {sys.platform}",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and convert models from source.txt to LiteRT-compatible TFLite."
    )
    parser.add_argument(
        "--source-root",
        default="models/source",
        help="Directory to store downloaded archives and extracted content.",
    )
    parser.add_argument(
        "--output-root",
        default="models/converted",
        help="Directory to store converted TFLite models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload archives even if they already exist.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each download attempt.",
    )
    parser.add_argument(
        "--use-curl",
        action="store_true",
        help="Use curl as a fallback downloader if urllib fails.",
    )
    parser.add_argument(
        "--protocol-path",
        default="protocols/latest_run.md",
        help="Path to write conversion protocol output.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Attempt full int8 quantization (requires representative dataset).",
    )
    parser.add_argument(
        "--int8-samples",
        type=int,
        default=100,
        help="Number of samples to generate for int8 calibration.",
    )
    args = parser.parse_args()

    ensure_dir(args.source_root)
    ensure_dir(args.output_root)

    protocol_lines = protocol_header()

    for spec in MODEL_SPECS:
        notes = process_model(
            spec,
            args.source_root,
            args.output_root,
            args.force,
            args.timeout,
            args.use_curl,
            args.int8,
            args.int8_samples,
        )
        print(f"{spec['name']}:")
        protocol_lines.append(f"## {spec['name']}")
        for note in notes:
            print(f"  - {note}")
            protocol_lines.append(f"- {note}")
        protocol_lines.append("")

    protocol_path = args.protocol_path
    ensure_dir(os.path.dirname(protocol_path) or ".")
    with open(protocol_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(protocol_lines))
    print(f"Protocol written to {protocol_path}")


if __name__ == "__main__":
    main()
