# CUDA-SPIRV Inference PoC Report

Status: pass

Pipeline: input -> Conv1D -> BatchNorm -> ReLU -> flatten -> FC/matmul -> Softmax -> output.

Intel iGPU SPIR-V path: Vulkan/SPIR-V only, no CUDA, no NVIDIA proprietary runtime. Device: Intel(R) HD Graphics 630 (KBL GT2), vendor 0x8086.

RTX 3060 baseline: CUDA custom kernels compiled with nvcc for NVIDIA GeForce RTX 3060. No cuBLAS, cuDNN, TensorRT, or TensorFlow runtime is used.

Comparison summary:
- Elements: 10
- Status: pass
- Mismatch count: 0
- Max absolute error: 7.450580596923828e-09
- Max relative error: 8.082866844144868e-08
- Max ULP error: 1
- Thresholds: max_abs <= 1e-5, max_rel <= 1e-4, max_ulp <= 64
- Top-1 class: 6 on both CUDA baseline and Intel SPIR-V

Artifacts:
- Intel report: inference_poc/out/intel_igpu/report.json
- CUDA baseline report: inference_poc/out/nvidia_cuda/report.json
- Comparison: inference_poc/out/compare/inference_comparison.json
- Evidence root: <repo-root>/src/cuda-spirv/artifacts/inference_poc/20260415T114946Z

Execution notes:
- Remote source build on the iGPU host was attempted, but its crates.io proxy returned CONNECT 301 and prevented dependency resolution after omitting the incompatible local lockfile.
- The Intel validation therefore used the locally built release Rust binary transferred to the iGPU host, with the same checked-in fixtures and SPIR-V modules.

Non-claims: not TensorFlow, not full TensorFlow GPU replacement, no cuBLAS/cuDNN, no TensorRT, no hand-written SPIR-V bypass of the verified-kernel scope.
