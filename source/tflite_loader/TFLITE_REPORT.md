# TFLite Loader Runtime Report

Run stamp: `20260415T203516Z`

## Status

`PASS_WITH_GENERIC_MULTI_OP_TFLITE_SCHEDULER`

The crate provides a checked TFLite FlatBuffer loader, supported-op graph validator, SPIR-V kernel registry, Vulkan compute dispatch wrapper, and a generic static float32 multi-op scheduler. The MobileNetV2 float `.tflite` now executes through the parsed TFLite operator graph rather than through a MobileNetV2-specific runner.

Unsupported operators, quantized tensors, dynamic tensors, unsupported fused activations, and unavailable kernel lowerings still fail closed.

## Implemented

- Preserved single-op model execution for `LOGISTIC`, `TANH`, and `LEAKY_RELU`.
- Added generic TFLite graph scheduling for the parsed 66-op MobileNetV2 histogram: `Conv2d=36`, `DepthwiseConv2d=17`, `Add=10`, `AveragePool2d=1`, `Reshape=1`, `Softmax=1`.
- Added exact TFLite `SAME` padding offsets for Conv2D and DepthwiseConv2D, including asymmetric stride-2 padding.
- Recompiled and validated the affected SPIR-V kernels: Conv2D and DepthwiseConv2D.

## Evidence

Evidence root: `artifacts/generic_tflite_scheduler/20260415T203516Z/`

Key reports:

- `local-self-test-after-padding.json`: `pass`, LeakyReLU/Sigmoid/Tanh, mismatch count `0`, max abs error `5.9604645e-8`.
- `local-synthetic-logistic-generic.json`: `pass`, model-driven `.tflite` LOGISTIC execution, mismatch count `0`, max abs error `5.9604645e-8`.
- `local-mobilenet-v2-generic.json`: `pass`, generic scheduler, top-1 ImageNet class `282` (`model_class=283` for the 1001-output TFLite head).
- `intel-mobilenet-v2-generic-input.json`: `pass`, Intel HD Graphics 630, generic scheduler, top-1 ImageNet class `282` (`model_class=283`).

## Device Evidence

Remote target: `ahn@192.168.68.129`

Intel execution report device:

- Device: `Intel(R) HD Graphics 630 (KBL GT2)`
- Vendor ID: `0x8086`
- Device ID: `0x5912`

Local unary kernel report device:

- Device: `NVIDIA GeForce RTX 3060`
- Vendor ID: `0x10de`
- Device ID: `0x2544`

## Limits

The scheduler is generic for the supported static float32 subset currently implemented by this crate. It is not a full TensorFlow Lite runtime, does not support quantized graphs, and intentionally fails closed for unsupported ops or option combinations.
