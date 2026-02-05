# Raw model int8 conversion protocol

- Timestamp (UTC): 2026-02-04T09:33:07.074972+00:00
- Python: 3.12.12
- Platform: linux

## ResNet-50-model.caffemodel
- Skipped (not a .pb file).

## deeplabv3.pb
- Used predefined IO configuration.
- Quantization settings: {'experimental_new_quantizer': True, 'disable_per_channel': True}
- Inputs: ['ImageTensor']
- Input shapes: {'ImageTensor': [1, 513, 513, 3]}
- Outputs: ['SemanticPredictions']
- Converted to models/converted/raw_model_int8/deeplabv3_int8.tflite
- Notes: Converted frozen graph with default settings.

## facenet.pb
- Used predefined IO configuration.
- Quantization settings: {'experimental_new_quantizer': True}
- Inputs: ['input']
- Input shapes: {'input': [1, 160, 160, 3]}
- Outputs: ['output']
- Converted to models/converted/raw_model_int8/facenet_int8.tflite
- Notes: Converted frozen graph with default settings.

## mobilebert_quantized.pb
- Used predefined IO configuration.
- Quantization settings: {'experimental_new_quantizer': True, 'disable_per_channel': True}
- Inputs: ['input_ids', 'input_mask', 'segment_ids']
- Input shapes: {'input_ids': [1, 384], 'input_mask': [1, 384], 'segment_ids': [1, 384]}
- Outputs: ['start_logits', 'end_logits']
- Converted to models/converted/raw_model_int8/mobilebert_quantized_int8.tflite
- Notes: Converted frozen graph with default settings.

## mobilenetV1ssd.pb
- Used predefined IO configuration.
- Quantization settings: {'experimental_new_quantizer': True}
- Inputs: ['image_tensor']
- Input shapes: {'image_tensor': [1, 300, 300, 3]}
- Outputs: ['detection_boxes', 'detection_scores', 'detection_classes', 'num_detections']
- Conversion failed: Merge of two inputs that differ on more than one predicate {s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/Greater:0,else), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/pred_id/_181__cf__184:0,then), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/Greater/_180__cf__183:0,then)} and {s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/Greater:0,else), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/pred_id/_181__cf__184:0,else), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/Greater/_180__cf__183:0,else)}
	for node {{node Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/Merge}}
	Failed to functionalize Control Flow V1 ops. Consider using Control Flow V2 ops instead. See https://www.tensorflow.org/api_docs/python/tf/compat/v1/enable_control_flow_v2.; fallback failed: Merge of two inputs that differ on more than one predicate {s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/Greater:0,else), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/pred_id/_181__cf__184:0,then), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/Greater/_180__cf__183:0,then)} and {s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/Greater:0,else), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/pred_id/_181__cf__184:0,else), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/Greater/_180__cf__183:0,else)}
	for node {{node Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/Merge}}
	Failed to functionalize Control Flow V1 ops. Consider using Control Flow V2 ops instead. See https://www.tensorflow.org/api_docs/python/tf/compat/v1/enable_control_flow_v2.

## mobilenet_v2_1.4_224_frozen.pb
- Used predefined IO configuration.
- Quantization settings: {'experimental_new_quantizer': True}
- Inputs: ['input']
- Input shapes: {'input': [1, 224, 224, 3]}
- Outputs: ['MobilenetV2/Predictions/Reshape_1']
- Converted to models/converted/raw_model_int8/mobilenet_v2_1.4_224_frozen_int8.tflite
- Notes: Converted frozen graph with default settings.

## vdsr_model.pb
- Used predefined IO configuration.
- Quantization settings: {'experimental_new_quantizer': True}
- Inputs: ['low_resolution']
- Input shapes: {'low_resolution': [1, 48, 48, 1]}
- Outputs: ['vdsr_output/add']
- Converted to models/converted/raw_model_int8/vdsr_model_int8.tflite
- Notes: Converted frozen graph with default settings.

## vgg16_imageNet_notop.pb
- Used predefined IO configuration.
- Quantization settings: {'experimental_new_quantizer': True}
- Inputs: ['input_2']
- Input shapes: {'input_2': [1, 224, 224, 3]}
- Outputs: ['output_1']
- Converted to models/converted/raw_model_int8/vgg16_imageNet_notop_int8.tflite
- Notes: Converted frozen graph with default settings.
