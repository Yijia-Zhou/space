# LiteRT INT8 conversion plan (SM8850 / MT6993)

This guide converts the frozen GraphDef models under `raw_model/` to INT8 TFLite/LiteRT artifacts that can be compiled and executed with the LiteRT `CompiledModel` API on mobile NPUs such as Qualcomm SM8850 and MediaTek MT6993.

> **Important:** The provided script uses a **random representative dataset** for post-training quantization (PTQ). Replace it with real calibration data before production to minimize accuracy loss and to better match NPU quantization ranges.

## Why these settings

* **Full integer quantization (INT8)** is the typical requirement for mobile NPUs, including SM8850 and MT6993.
* `TFLITE_BUILTINS_INT8` ensures only INT8-capable built-in operators are used.
* Fixed input shapes avoid dynamic-shape incompatibilities that frequently block NPU compilation.

## Prerequisites

* Python 3.10+
* TensorFlow 2.12+ (includes TFLite converter)

## Conversion workflow

1. Inspect the input/output nodes for each frozen graph to ensure placeholders and outputs are correctly inferred:
   ```bash
   python scripts/convert_litert_int8.py --input raw_model/mobilenet_v2_1.4_224_frozen.pb --output converted_models/mobilenet_v2_int8.tflite
   ```
   If the model has unknown input shapes, use `--show-info` to list inputs/outputs/dtypes, then provide explicit `--input-shapes`:
   ```bash
   python scripts/convert_litert_int8.py \
     --input raw_model/mobilenet_v2_1.4_224_frozen.pb \
     --output converted_models/mobilenet_v2_int8.tflite \
     --show-info
   ```
2. If automatic inference fails, specify the inputs/outputs explicitly:
   ```bash
   python scripts/convert_litert_int8.py \
     --input raw_model/vgg16_imageNet_notop.pb \
     --output converted_models/vgg16_int8.tflite \
     --inputs input_1 \
     --outputs vgg16/logits \
     --input-shapes 1,224,224,3
   ```
3. Repeat for each `.pb` model in `raw_model/`.

## Optional: Quantization spec verification

Use `--verify` to report key INT8 quantization constraints (per-axis weights, zero_point requirements, fixed output scales for nonlinear ops, and same-scale requirements for shape ops). This is a best-effort checker aligned with the LiteRT 8-bit quantization specification. Add `--verify-strict` to fail the run when issues are detected.

```bash
python scripts/convert_litert_int8.py \
  --input raw_model/mobilenet_v2_1.4_224_frozen.pb \
  --output converted_models/mobilenet_v2_int8.tflite \
  --verify
```

## Representative dataset (random placeholder)

By default the converter generates `--samples` random inputs to calibrate quantization ranges. This is a **placeholder** step; use real inputs matching the deployment distribution for SM8850/MT6993 NPUs.

To increase randomness or sample size:
```bash
python scripts/convert_litert_int8.py \
  --input raw_model/deeplabv3.pb \
  --output converted_models/deeplabv3_int8.tflite \
  --samples 500 \
  --seed 13
```

## Notes for SM8850 / MT6993

* Prefer **fixed input shapes** (batch=1) for NPU compilation.
* Avoid unsupported ops; inspect conversion logs for operator fallback and use `--verify` to catch common spec mismatches.
* If a model fails to compile on-device, check for unsupported operations and re-export/modify the graph accordingly.

## Models in raw_model

| Model | Status | Notes |
| --- | --- | --- |
| `ResNet-50-model.caffemodel` | Not handled | Convert Caffe -> TF/TFLite first; then apply this workflow. |
| `deeplabv3.pb` | Supported | Frozen GraphDef. |
| `facenet.pb` | Supported | Frozen GraphDef. |
| `mobilebert_quantized.pb` | Supported | Already quantized; re-quantization may reduce accuracy. |
| `mobilenetV1ssd.pb` | Supported | Frozen GraphDef. |
| `mobilenet_v2_1.4_224_frozen.pb` | Supported | Frozen GraphDef. |
| `vdsr_model.pb` | Supported | Frozen GraphDef. |
| `vgg16_imageNet_notop.pb` | Supported | Frozen GraphDef. |
