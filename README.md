# LiteRT 模型转换与量化快速上手

本仓库提供原始模型的下载、转换与 int8 量化流程，并支持在 x86 Linux 环境进行 LiteRT/TFLite 运行验证。常见流程与完整说明请参考 `MODEL_CONVERSION.md`。

## 快速开始

### 1) 安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow
```

### 2) 批量下载并转换 `source.txt` 模型

```bash
python tools/download_and_convert_models.py --use-curl
```

如需尝试 int8 量化：

```bash
python tools/download_and_convert_models.py --use-curl --int8 --int8-samples 100
```

转换产物默认输出到 `models/converted/<model_name>/`，过程记录到 `protocols/latest_run.md`。

### 3) 转换 `raw_model/` 内的冻结图并生成 int8 模型

```bash
python tools/convert_raw_models.py --int8-samples 100
```

转换产物默认输出到 `models/converted/raw_model_int8/`，过程记录到 `protocols/raw_model_int8.md`。

### 4) 在 x86 Linux 环境验证运行（LiteRT/TFLite）

```bash
python tools/run_litert_models.py
```

验证结果会写入 `protocols/raw_model_int8_runtime.md`，用于排查运行时问题。

## 重要说明

- `raw_model/` 内模型的转换与运行结果（含失败原因）会写入 protocol 文件，建议以最新 protocol 为准。
- 若需查看每个模型的转换/运行状态与已知问题，请阅读 `MODEL_CONVERSION.md` 的统一汇总表。
