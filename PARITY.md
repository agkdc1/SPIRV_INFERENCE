# Parity Evidence

## Per-Kernel Evidence

| Kernel | Status | Notes |
| --- | --- | --- |
| vectorAdd | Pass | Bit-exact CUDA/SPIR-V parity recorded in release evidence. |
| matmul | Pass | Bit-exact CUDA/SPIR-V parity recorded in release evidence. |
| ReLU | Pass | Vulkan/SPIR-V output accepted against baseline evidence. |
| Softmax | Pass | Vulkan/SPIR-V output accepted against baseline evidence. |
| BatchNorm | Pass | Vulkan/SPIR-V output accepted against baseline evidence. |
| Conv1D | Pass | Inference fixture accepted. |
| Conv2D | Pass | Vulkan/SPIR-V output accepted against baseline evidence. |
| DepthwiseConv2D | Pass | MobileNetV2 and EfficientNet-Lite0 execution accepted. |
| ReLU6 | Pass | MobileNetV2 and EfficientNet-Lite0 execution accepted. |
| AvgPool | Pass | MobileNetV2 and EfficientNet-Lite0 execution accepted. |

## Per-Model Evidence

| Model | Device Path | Top-1 class | Epsilon | Mismatches | Max absolute error | Output SHA-256 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MobileNetV2 | RTX 3060 CUDA vs Intel HD 630 SPIR-V | 282 | 1e-3 | 0 | 1.9073486328125e-6 | See `RELEASE_REPORT.md` |
| EfficientNet-Lite0 | Local Vulkan vs Intel HD 630 SPIR-V | 285 | 1e-4 | 0 | 4.470348358154297e-7 | See `RELEASE_REPORT.md` |

## Cross-Device Comparison

| Comparison | Status | ULP policy |
| --- | --- | --- |
| RTX 3060 CUDA vs Intel HD 630 SPIR-V | Pass | Per-output ULP recorded where available; epsilon gate is authoritative for accumulated kernels. |
| Local Vulkan vs Intel HD 630 SPIR-V | Pass | Byte identity is not required; zero epsilon mismatches is required. |

## SHA-256 Hashes

Exact output hashes from the recorded release verification are listed in `RELEASE_REPORT.md`.
