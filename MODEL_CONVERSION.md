# LiteRT 模型下载与转换说明

本仓库提供了自动化脚本，用于下载 `source.txt` 中列出的模型，并尽可能转换为 LiteRT compiler 可消费的 TFLite 模型。下载产物与转换结果默认放在本地 `models/` 目录，该目录已加入 `.gitignore`。

## 使用方法

1. 安装依赖（至少需要 TensorFlow 支持 TFLite Converter）：

   ```bash
   python -m pip install --upgrade pip
   python -m pip install tensorflow
   ```

2. 运行下载与转换脚本（自动生成 protocol）：

   ```bash
   python tools/download_and_convert_models.py --use-curl
   ```

   若需尝试 int8 量化（脚本会根据模型输入自动生成代表性样本，建议替换为真实样本以提高精度）：

   ```bash
   python tools/download_and_convert_models.py --use-curl --int8 --int8-samples 100
   ```

3. 转换后的模型会输出到 `models/converted/<model_name>/`，protocol 会写入 `protocols/latest_run.md`。

## 注意事项

- 部分模型仅提供 TF-Slim 或 Darknet 的权重，需要先按官方示例导出 frozen graph 或 SavedModel 后再转换。
- 由于部分模型存放在 Google Drive，脚本会提示需要手动下载并放置到 `models/source/<model_name>/` 目录。
- 如果遇到输入/输出张量名称不匹配，请结合官方文档与模型结构调整脚本中的 `input_arrays`、`output_arrays` 与 `input_shapes`。
- 如需延长下载超时或覆盖 protocol 输出路径，可参考 `python tools/download_and_convert_models.py --help`。
- 手动补充步骤可参考 `protocols/MANUAL_FOLLOWUPS.md`。

## raw_model 目录 int8 转换流程

仓库已提供 `tools/convert_raw_models.py` 用于将 `raw_model/` 内的冻结图 (`.pb`) 转换为 int8 量化的 TFLite 模型，供 LiteRT CompiledModel API 使用。输出默认写入 `models/converted/raw_model_int8/`，转换过程记录到 `protocols/raw_model_int8.md`。

```bash
python tools/convert_raw_models.py --int8-samples 100
```

- 默认会尝试推断输入/输出张量；对于常见模型（如 DeepLabV3、MobileNetV1 SSD、Facenet）内置了 IO 覆盖配置。
- 如果 TensorFlow 未安装，脚本会记录原因并跳过转换（可参考上文安装 TensorFlow）。

### x86 Linux LiteRT 运行验证

当 int8 TFLite 模型生成后，可使用 `tools/run_litert_models.py` 在 x86 Linux 环境进行基础推理验证（默认优先尝试 `tflite_runtime`，若不可用则回退到 `tensorflow` 解释器并记录到 protocol）：

```bash
python tools/run_litert_models.py
```

验证结果会写入 `protocols/raw_model_int8_runtime.md`，用于检查模型在 LiteRT/TFLite 运行时的潜在问题。

### raw_model 各模型转换与运行情况（统一汇总）

下表汇总 `raw_model/` 内模型的 int8 转换结果与 x86 Linux 运行验证情况（以最新 protocol 为准）：

| 模型 | 转换结果 | 运行结果 | 说明 |
| --- | --- | --- | --- |
| DeepLabV3 (`deeplabv3.pb`) | 转换成功 | 运行失败（`Return code: -6`） | 运行时进程异常退出并伴随 TFLite/XNNPACK 初始化日志，表明当前 int8 产物在本环境存在不兼容或崩溃问题，需在更接近目标 LiteRT 运行时的环境验证或调整量化配置。【F:protocols/raw_model_int8.md†L10-L17】【F:protocols/raw_model_int8_runtime.md†L7-L20】 |
| FaceNet (`facenet.pb`) | 转换成功 | 运行成功 | 可在 TensorFlow Lite 运行时完成推理（输入/输出已记录）。【F:protocols/raw_model_int8.md†L18-L24】【F:protocols/raw_model_int8_runtime.md†L21-L26】 |
| MobileBERT (`mobilebert_quantized.pb`) | 转换成功 | 运行失败 | 运行时准备阶段失败，`FULLY_CONNECTED` bias 校验不通过（`is_optional_bias_int != true`），通常需要重新导出/量化以满足 TFLite 对 bias dtype 的要求或调整量化配置。【F:protocols/raw_model_int8.md†L26-L32】【F:protocols/raw_model_int8_runtime.md†L28-L39】 |
| MobileNetV1 SSD (`mobilenetV1ssd.pb`) | 转换失败 | 无运行结果 | Frozen graph 中的 Control Flow V1 `Merge` 谓词不一致导致 TFLite functionalization 失败；需要在导出端切换到 Control Flow V2 或使用 TF2/OD API 兼容图后再尝试量化。【F:protocols/raw_model_int8.md†L34-L43】 |
| MobileNetV2 (`mobilenet_v2_1.4_224_frozen.pb`) | 转换成功 | 运行成功 | 可在 TensorFlow Lite 运行时完成推理（输入/输出已记录）。【F:protocols/raw_model_int8.md†L45-L51】【F:protocols/raw_model_int8_runtime.md†L41-L46】 |
| VDSR (`vdsr_model.pb`) | 转换成功 | 运行成功 | 可在 TensorFlow Lite 运行时完成推理（输入/输出已记录）。【F:protocols/raw_model_int8.md†L53-L59】【F:protocols/raw_model_int8_runtime.md†L48-L53】 |
| VGG16 (`vgg16_imageNet_notop.pb`) | 转换成功 | 运行成功 | 可在 TensorFlow Lite 运行时完成推理（输入/输出已记录）。【F:protocols/raw_model_int8.md†L61-L67】【F:protocols/raw_model_int8_runtime.md†L55-L60】 |
