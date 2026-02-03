# Conversion protocol

- Timestamp (UTC): 2026-02-03T02:10:05.609233+00:00
- Python: 3.12.12
- Platform: linux

## resnet_v2_50
- Downloaded resnet_v2_50_2017_04_14.tar.gz (2c5c3a852526) from https://storage.googleapis.com/download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz
- The archive contains TF-Slim checkpoints only. Export a frozen graph with the TF-Slim export scripts before converting to TFLite.

## mobilenet_v2
- Downloaded mobilenet_v2_1.4_224.tgz (b19463cb9b84) from https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz
- Downloaded mobilenet_v2.caffemodel (a3124ce7abd2) from https://github.com/shicai/MobileNet-Caffe/raw/master/mobilenet_v2.caffemodel
- No .tflite file found in the MobileNet V2 archive.

## mobilenet_v1_ssd
- Downloaded ssd_mobilenet_v1_coco_2017_11_17.tar.gz (da29bfb388fd) from https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
- Frozen graph conversion failed: Merge of two inputs that differ on more than one predicate {s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/Greater:0,else), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/pred_id/_181__cf__184:0,then), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/Greater/_180__cf__183:0,then)} and {s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/Greater:0,else), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/pred_id/_181__cf__184:0,else), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/Greater/_180__cf__183:0,else)}
	for node {{node Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/Merge}}
	Failed to functionalize Control Flow V1 ops. Consider using Control Flow V2 ops instead. See https://www.tensorflow.org/api_docs/python/tf/compat/v1/enable_control_flow_v2.; fallback failed: Merge of two inputs that differ on more than one predicate {s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/Greater:0,else), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/pred_id/_181__cf__184:0,then), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/Greater/_180__cf__183:0,then)} and {s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/Greater:0,else), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/pred_id/_181__cf__184:0,else), s(Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/Greater/_180__cf__183:0,else)}
	for node {{node Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/cond/Merge}}
	Failed to functionalize Control Flow V1 ops. Consider using Control Flow V2 ops instead. See https://www.tensorflow.org/api_docs/python/tf/compat/v1/enable_control_flow_v2.

## vgg16
- Downloaded vgg_16_2016_08_28.tar.gz (de44460bf79d) from https://storage.googleapis.com/download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
- Downloaded VGG_ILSVRC_16_layers.caffemodel (a6196bc498e4) from https://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
- Export a frozen graph with TF-Slim before converting to TFLite.

## deeplabv3_mnv2
- Downloaded deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz (2b7fe43461c2) from https://storage.googleapis.com/download.tensorflow.org/models/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz
- Converted frozen graph to models/converted/deeplabv3_mnv2/deeplabv3_mnv2.tflite. Converted frozen graph with default settings.

## vdsr
- Manual download required for tensorflow_checkpoint: https://drive.google.com/file/d/1wiej51wFY0oYsoKF7gGiWZT5_t5mFt0f/view?usp=sharing
- Requires the TensorFlow implementation from VDSR-TensorFlow.

## yolov3
- Downloaded yolov3.weights (523e4e69e1d0) from https://pjreddie.com/media/files/yolov3.weights
- Convert Darknet weights to a TensorFlow frozen graph before converting to TFLite.

## facenet
- Manual download required for tensorflow_model: https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-
- Use the provided Facenet preprocessing scripts to export a SavedModel.

## mobilebert
- Downloaded mobilebert_squad_savedmodels.tar.gz (a9be6648506d) from https://storage.googleapis.com/cloud-tpu-checkpoints/mobilebert/mobilebert_squad_savedmodels.tar.gz
- Converted SavedModel to models/converted/mobilebert/mobilebert.tflite
