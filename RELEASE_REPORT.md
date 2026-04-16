# Vulkan Inference Public Evidence Report

Run stamp: 20260415T214515Z

## Status

- Local RTX 3060 Vulkan verification: pass for recorded fixtures.
- Intel HD 630 Vulkan verification: pass for recorded fixtures.
- MobileNetV2 and EfficientNet-Lite0 examples produced expected top-1 classes on recorded devices.
- The public showcase includes source, shaders, SPIR-V modules, reports, and recorded output fixtures. It does not vendor NVIDIA/CUDA binaries, proprietary toolchains, model weights, large tensor dumps, or third-party source trees.

## Local Verification

- Kernel self-test: pass, device: NVIDIA GeForce RTX 3060, mismatches: 0.
- MobileNetV2 top-1 class: 282, confidence: 0.3372005820274353, output SHA-256: d97e9ef0b790ce3fb3481ecee67f196705452c287a7f66d0fde7964e0fb6700b.
- EfficientNet-Lite0 top-1 class: 285, confidence: 0.38234177231788635, output SHA-256: e4de40d0831301820060832621ed5af6b04c4e0efca8c5f381ca93a973d10aee.

## Intel HD 630 Verification

- Kernel self-test: pass, device: Intel HD Graphics 630, mismatches: 0.
- MobileNetV2 top-1 class: 282, confidence: 0.33720120787620544, output SHA-256: 4d016354f9cd3eb53c362683f8c9e65ad16df09e1c89700f51b78ced90d7e57e.
- EfficientNet-Lite0 top-1 class: 285, confidence: 0.3823419511318207, output SHA-256: e0137e1876c4629bfa929076e0d99b9cd4aaf26684a659d25b3340b4a8ca3498.

## Nonclaims

- No production inference framework is claimed.
- No complete TensorFlow, StableHLO, Vulkan, or model-family coverage is claimed.
- No hardware portability guarantee is claimed beyond the recorded devices.
