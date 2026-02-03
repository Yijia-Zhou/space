# Manual conversion follow-ups

This checklist complements `protocols/latest_run.md` with follow-up steps for models that require manual export before LiteRT conversion. Use these steps in another environment if needed.

## ResNet V2 / VGG16 (TF-Slim checkpoints)

1. Install TF-Slim (in the same environment as TensorFlow):

   ```bash
   pip install tf_slim
   ```

2. Export an inference graph and freeze variables. You can use the TF-Slim export scripts and then freeze the graph. After producing a `.pb`, inspect it to discover input/output names.

   ```bash
   # Example: inspect a frozen graph to find tensors
   python - <<'PY'
   import tensorflow as tf
   graph_def = tf.compat.v1.GraphDef()
   graph_def.ParseFromString(open("frozen_graph.pb","rb").read())
   for node in graph_def.node:
       if node.op == "Placeholder":
           print("INPUT", node.name)
       if "Predictions" in node.name:
           print("OUTPUT", node.name)
   PY
   ```

3. Convert to TFLite:

   ```bash
   python - <<'PY'
   import tensorflow as tf
   converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
       "frozen_graph.pb",
       input_arrays=["<INPUT_NAME>"],
       output_arrays=["<OUTPUT_NAME>"],
       input_shapes={"<INPUT_NAME>": [1, 224, 224, 3]},
   )
   tflite_model = converter.convert()
   open("model.tflite","wb").write(tflite_model)
   PY
   ```

## MobileNet V1 SSD (control-flow conversion failure)

The `ssd_mobilenet_v1_coco_2017_11_17` frozen graph failed conversion due to Control Flow V1 ops. Preferred approach:

1. Use TensorFlow Object Detection API (TF2) to export a SavedModel.
2. Convert with Select TF ops enabled:

   ```bash
   python - <<'PY'
   import tensorflow as tf
   converter = tf.lite.TFLiteConverter.from_saved_model("exported_saved_model")
   converter.target_spec.supported_ops = [
       tf.lite.OpsSet.TFLITE_BUILTINS,
       tf.lite.OpsSet.SELECT_TF_OPS,
   ]
   tflite_model = converter.convert()
   open("ssd_mobilenet_v1.tflite","wb").write(tflite_model)
   PY
   ```

## YOLOv3 (Darknet weights)

1. Use a converter such as `YOLOv3_TensorFlow` or `darknet2tf` to generate a TensorFlow frozen graph or SavedModel.
2. Inspect input/output tensor names, then convert using the TFLite converter (similar to ResNet/VGG steps).

## VDSR / FaceNet (Google Drive downloads)

1. Download from Google Drive using `gdown`:

   ```bash
   pip install gdown
   gdown "https://drive.google.com/uc?id=<FILE_ID>"
   ```

2. Follow the upstream repo instructions to export a SavedModel, then convert using `tf.lite.TFLiteConverter.from_saved_model`.

## MobileBERT

The `mobilebert_squad_savedmodels` archive includes both `float/` and `quant_saved_model/` SavedModels. Use the one that best matches your target deployment. The script defaults to `float/`.

## Int8 量化（通用说明）

脚本支持 `--int8`，会根据模型输入自动生成代表性样本，但**强烈建议**替换为真实样本以获得更准确的量化结果。示例流程如下：

```bash
python - <<'PY'
import numpy as np
import tensorflow as tf

def representative_dataset():
    for _ in range(100):
        # TODO: 使用真实样本，确保形状/类型与模型输入一致
        yield [np.zeros((1, 224, 224, 3), dtype=np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model("exported_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
open("model_int8.tflite", "wb").write(tflite_model)
PY
```
