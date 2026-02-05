# Raw model int8 runtime protocol

- Timestamp (UTC): 2026-02-04T09:36:03.832148+00:00
- Python: 3.12.12
- Platform: linux

- tflite_runtime not available; falling back to tensorflow.
## deeplabv3_int8.tflite
- Runtime: unknown; failed to run: unknown error
- Return code: -6
- Stderr: 2026-02-04 09:36:04.623166: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2026-02-04 09:36:04.643720: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2026-02-04 09:36:04.688022: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1770197764.752227    6422 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1770197764.774767    6422 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2026-02-04 09:36:04.854475: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

## facenet_int8.tflite
- Runtime: tensorflow
- Inputs:
  - input: shape=[1, 160, 160, 3], dtype=int8
- Outputs:
  - output: shape=[1, 512], dtype=int8

## mobilebert_quantized_int8.tflite
- Runtime: unknown; failed to run: tensorflow/lite/kernels/fully_connected.cc:212 is_optional_bias_int != true (0 != 1)Node number 1460 (FULLY_CONNECTED) failed to prepare.
- Return code: 1
- Stderr: 2026-02-04 09:36:16.280282: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2026-02-04 09:36:16.302097: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2026-02-04 09:36:16.345414: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1770197776.414586    6466 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1770197776.436287    6466 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2026-02-04 09:36:16.523783: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.

## mobilenet_v2_1.4_224_frozen_int8.tflite
- Runtime: tensorflow
- Inputs:
  - input: shape=[1, 224, 224, 3], dtype=int8
- Outputs:
  - MobilenetV2/Predictions/Reshape_1: shape=[1, 1001], dtype=int8

## vdsr_model_int8.tflite
- Runtime: tensorflow
- Inputs:
  - low_resolution: shape=[1, 48, 48, 1], dtype=int8
- Outputs:
  - vdsr_output/add: shape=[1, 48, 48, 1], dtype=int8

## vgg16_imageNet_notop_int8.tflite
- Runtime: tensorflow
- Inputs:
  - input_2: shape=[1, 224, 224, 3], dtype=int8
- Outputs:
  - output_1: shape=[7, 7, 512], dtype=int8
