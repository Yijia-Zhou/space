# LiteRT INT8 模型转换（SM8850 / MT6993）中文指南

本仓库提供的脚本用于将 `raw_model/` 下的冻结 GraphDef（`.pb`）模型转换为 LiteRT/TFLite INT8 模型，并通过 LiteRT 8-bit quantization spec 做基础校验，便于在 SM8850、MT6993 等移动端 NPU 上进一步编译与验证。

> **说明**：
> - 本指南使用“随机代表性数据”做 PTQ（后训练量化）占位。真实项目中应替换为贴合业务分布的校准数据。
> - 一些模型由于控制流或量化限制可能无法转换或会触发校验警告，文末给出已验证的结果与建议。

## 1. 环境准备

建议在独立的 Python 3.10 环境中执行（脚本在本次验证中使用 Python 3.10 + TensorFlow 2.15）：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install tensorflow-cpu==2.15.0 numpy
```

## 2. 快速上手

脚本入口：`scripts/convert_litert_int8.py`

常用参数：
- `--input` / `--output`：输入 `.pb` 与输出 `.tflite`
- `--inputs` / `--outputs`：输入/输出节点名（可用 `--show-info` 自动推断）
- `--input-shapes`：显式指定输入 shape（当模型有 `-1` 或未知维度时必须指定）
- `--verify`：输出量化规范检查（不阻断运行）
- `--verify-strict`：若有问题则直接报错并退出

**先查看模型输入/输出/类型：**
```bash
python scripts/convert_litert_int8.py \
  --input raw_model/mobilenet_v2_1.4_224_frozen.pb \
  --output converted_models/mobilenet_v2_int8.tflite \
  --show-info
```

## 3. 已验证模型的推荐命令

> 说明：为了满足 NPU 运行要求，这里使用固定输入尺寸（batch=1）。

### ✅ MobileNetV2
```bash
python scripts/convert_litert_int8.py \
  --input raw_model/mobilenet_v2_1.4_224_frozen.pb \
  --output converted_models/mobilenet_v2_int8.tflite \
  --inputs input \
  --outputs MobilenetV2/Predictions/Reshape_1 \
  --input-shapes 1,224,224,3 \
  --samples 10 \
  --verify
```

### ⚠️ DeeplabV3（输入为 uint8，输出非 int8）
```bash
python scripts/convert_litert_int8.py \
  --input raw_model/deeplabv3.pb \
  --output converted_models/deeplabv3_int8.tflite \
  --inputs ImageTensor \
  --outputs SemanticPredictions \
  --input-shapes 1,513,513,3 \
  --samples 10 \
  --verify
```

### ⚠️ FaceNet（FC 权重未 per-axis）
```bash
python scripts/convert_litert_int8.py \
  --input raw_model/facenet.pb \
  --output converted_models/facenet_int8.tflite \
  --inputs input \
  --outputs output \
  --input-shapes 1,160,160,3 \
  --samples 10 \
  --verify
```

### ⚠️ MobileBERT（量化后 kernel 校验失败）
```bash
python scripts/convert_litert_int8.py \
  --input raw_model/mobilebert_quantized.pb \
  --output converted_models/mobilebert_int8.tflite \
  --inputs input_ids,input_mask,segment_ids \
  --outputs start_logits,end_logits \
  --input-shapes 1,384 1,384 1,384 \
  --samples 5 \
  --verify
```

### ❌ MobileNetV1 SSD（控制流 V1 无法转换）
```bash
python scripts/convert_litert_int8.py \
  --input raw_model/mobilenetV1ssd.pb \
  --output converted_models/mobilenetV1ssd_int8.tflite \
  --inputs image_tensor \
  --outputs detection_boxes,detection_scores,detection_classes,num_detections \
  --input-shapes 1,300,300,3 \
  --samples 5 \
  --verify
```
> 该模型包含 Control Flow V1，TFLite 转换阶段会报错；建议重新导出为 SavedModel 并启用 Control Flow V2。

### ⚠️ VDSR（Conv 权重未 per-axis）
```bash
python scripts/convert_litert_int8.py \
  --input raw_model/vdsr_model.pb \
  --output converted_models/vdsr_int8.tflite \
  --inputs low_resolution \
  --outputs vdsr_output/add \
  --input-shapes 1,128,128,1 \
  --samples 10 \
  --verify
```

### ✅ VGG16
```bash
python scripts/convert_litert_int8.py \
  --input raw_model/vgg16_imageNet_notop.pb \
  --output converted_models/vgg16_int8.tflite \
  --inputs input_2 \
  --outputs output_1 \
  --input-shapes 1,224,224,3 \
  --samples 10 \
  --verify
```

## 4. 验证结果摘要（当前环境）

| 模型 | 转换结果 | 量化校验结果 | 备注 |
| --- | --- | --- | --- |
| MobileNetV2 | ✅ 成功 | ✅ 无问题 | 可继续做 NPU 编译验证 |
| DeeplabV3 | ✅ 成功 | ⚠️ 输入 uint8 / 输出 int64 | 建议改造模型输出或重新导出 |
| FaceNet | ✅ 成功 | ⚠️ FC 权重非 per-axis | 可尝试重训或 QAT |
| MobileBERT | ✅ 成功 | ⚠️ 校验阶段 kernel 准备失败 | 可能存在量化参数不一致 |
| MobileNetV1 SSD | ❌ 失败 | N/A | Control Flow V1 转换失败 |
| VDSR | ✅ 成功 | ⚠️ Conv 权重非 per-axis | 建议重训或 QAT |
| VGG16 | ✅ 成功 | ✅ 无问题 | 可继续做 NPU 编译验证 |

## 5. 后续建议（SM8850 / MT6993）

1. 替换随机校准数据为真实数据集。
2. 对于 per-axis 失败的模型，考虑 QAT 或重新导出支持 per-channel quant。
3. 对于控制流模型，优先导出为 SavedModel 并启用 Control Flow V2。
4. 在手机端使用 LiteRT CompiledModel API 验证是否完全下沉到 NPU。 
