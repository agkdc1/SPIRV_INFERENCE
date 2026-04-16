# TFLite Generic Scheduler Generality Report

Run stamp: `20260415T212020Z`
Artifact root: `<repo-root>/src/cuda-spirv/artifacts/tflite_generality/20260415T212020Z`
Remote Intel target: `ahn@192.168.68.129`

## Verdict

PASS. EfficientNet-Lite0, a real non-MobileNet TFLite architecture, executed through the generic multi-op TFLite scheduler and Vulkan/SPIR-V kernels on the remote Intel HD 630 target. This proves the scheduler path is not MobileNet-specific.

The c3 re-execution also fixed the prior download-command issue by enabling `pipefail`; the first storage.googleapis.com URL returned HTTP 403, then the tfhub.dev URL successfully downloaded the EfficientNet-Lite0 model.

## Model Evidence

- Baseline model: `tflite_loader/models/mobilenet_v2_float.tflite`
- Second model: `artifacts/tflite_generality/20260415T212020Z/models/efficientnet_lite0_fp32.tflite`
- Second model SHA-256: `7276a936443685daae8612fc3cb9370247cb1f4966286cd95e02c44e1183f0dd`
- EfficientNet ops: `Add=9, AveragePool2d=1, Conv2d=33, DepthwiseConv2d=16, FullyConnected=1, Reshape=1, Softmax=1`
- Inspect status: `pass`
- Unsupported ops: `[]`
- Graph validation: `62` operators, `62` supported operators, `164` tensors

## Local Run

- Report: `artifacts/tflite_generality/20260415T212020Z/reports/local-efficientnet_lite0_fp32.json`
- Status: `pass`
- Device: `llvmpipe (LLVM 20.1.2, 256 bits)`
- Top-1 class: `285`
- Output SHA-256: `e4de40d0831301820060832621ed5af6b04c4e0efca8c5f381ca93a973d10aee`

NVIDIA visibility was verified with `nvidia-smi`, `vulkaninfo --summary`, and the kernel self-test on `NVIDIA GeForce RTX 3060`. The explicit full EfficientNet run with `--device nvidia` still returned the structured blocked result `no matching Vulkan compute device found for selector nvidia`, so the local full-model comparison uses the passing `--device any` Vulkan run.

## Intel HD 630 Run

- Report: `artifacts/tflite_generality/20260415T212020Z/reports/intel-efficientnet_lite0_fp32.json`
- Status: `pass`
- Device: `Intel(R) HD Graphics 630 (KBL GT2)`
- Top-1 class: `285`
- Output SHA-256: `e0137e1876c4629bfa929076e0d99b9cd4aaf26684a659d25b3340b4a8ca3498`

## Cross-Device Numeric Parity

- Report: `artifacts/tflite_generality/20260415T212020Z/reports/efficientnet_lite0_fp32-local-vs-intel-parity.json`
- Status: `pass`
- Epsilon: `0.0001`
- Elements: `1000`
- Mismatch count: `0`
- Max absolute error: `4.470348358154297e-7`
- Max relative error: `4.122233718674037e-6`
- Byte identical: `false`

## Verification Commands

- `cargo fmt -p tflite_loader`: pass
- `cargo check -p tflite_loader`: pass
- `cargo test -p tflite_loader`: pass
- `cargo build -p tflite_loader --release`: pass
- GLSL to SPIR-V compilation and `spirv-val`: pass for 14 kernels
- Kernel self-test: pass on `--device any`; pass on explicit `--device nvidia`
- MobileNetV2 baseline: pass, top-1 class `282`
- EfficientNet inspect: pass, zero unsupported ops
- EfficientNet local execution: pass
- EfficientNet Intel HD 630 execution: pass
- Local vs Intel numeric parity: pass

## Non-Claims

This does not claim full TensorFlow or arbitrary TFLite coverage. It proves the existing generic supported-subset scheduler is not MobileNet-specific by running a different real architecture with the supported op set, including float32 `FullyConnected`.
