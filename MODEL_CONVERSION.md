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
