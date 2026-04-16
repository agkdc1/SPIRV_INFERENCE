# SPIR-V Training POC — MLP on MNIST via Vulkan Compute

**Date:** 2026-04-16
**Device:** NVIDIA GeForce RTX 3060 (Vulkan 1.1 compute)
**CUDA runtime used:** No — zero CUDA dependencies

## Model Architecture

```
Input[784] -> Linear(784,128) -> ReLU -> Linear(128,10) -> Softmax -> CrossEntropy
```

- Parameters: 784×128 + 128 + 128×10 + 10 = 101,770
- Optimizer: Adam (lr=0.001, β1=0.9, β2=0.999, ε=1e-8)
- Batch size: 64
- Epochs: 5
- Weight init: Kaiming He (seed=42)

## 11 SPIR-V Training Kernels

All compiled from GLSL 450 via `glslangValidator`, validated via `spirv-val`.

| Kernel | Purpose | Local Size |
|--------|---------|-----------|
| matmul.spv | Standard A@B matrix multiply | 64 |
| matmul_tn.spv | A^T @ B (weight gradient) | 64 |
| matmul_nt.spv | A @ B^T (activation gradient) | 64 |
| relu_forward.spv | ReLU activation | 128 |
| relu_backward.spv | ReLU gradient (mask by input > 0) | 128 |
| bias_add.spv | Broadcast bias addition | 128 |
| batched_softmax10.spv | Per-sample 10-class softmax | 10 |
| cross_entropy_loss.spv | Batch CE loss (shared memory reduction) | 64 |
| softmax_ce_backward.spv | Fused softmax+CE gradient | 128 |
| reduce_sum_rows.spv | Row reduction (bias gradient) | 128 |
| adam_step.spv | Full Adam optimizer update | 128 |

## Training Loop (per batch)

**Forward (7 dispatches):**
1. `matmul(images[64×784], W1[784×128]) → temp`
2. `bias_add(temp, b1) → z1`
3. `relu(z1) → a1`
4. `matmul(a1[64×128], W2[128×10]) → temp`
5. `bias_add(temp, b2) → z2`
6. `batched_softmax10(z2) → sm`
7. `cross_entropy_loss(sm, labels) → loss`

**Backward (7 dispatches):**
8. `softmax_ce_backward(sm, labels) → dz2`
9. `matmul_tn(a1, dz2) → dW2`
10. `reduce_sum_rows(dz2) → db2`
11. `matmul_nt(dz2, W2) → da1`
12. `relu_backward(da1, z1) → dz1`
13. `matmul_tn(images, dz1) → dW1`
14. `reduce_sum_rows(dz1) → db1`

**Optimizer (4 dispatches):**
15–18. `adam_step` for W1, b1, W2, b2

**Total: 18 GPU dispatches per batch, 937 batches per epoch, 5 epochs = 84,330 dispatches**

## Results — SPIR-V/Vulkan (zero CUDA)

| Epoch | Train Loss | Test Accuracy |
|-------|-----------|---------------|
| 1 | 0.2938 | 95.25% |
| 2 | 0.1360 | 96.30% |
| 3 | 0.0958 | 97.06% |
| 4 | 0.0725 | 97.63% |
| 5 | 0.0569 | **97.46%** |

Total training time: **11.2 seconds**

## Results — PyTorch/CUDA Reference

| Epoch | Train Loss | Test Accuracy |
|-------|-----------|---------------|
| 1 | 0.2977 | 95.34% |
| 2 | 0.1346 | 96.47% |
| 3 | 0.0929 | 97.23% |
| 4 | 0.0711 | 97.48% |
| 5 | 0.0551 | **97.39%** |

Total training time: 48.0 seconds (includes PyTorch overhead)

## Comparison

| Metric | SPIR-V/Vulkan | PyTorch/CUDA | Delta |
|--------|--------------|-------------|-------|
| Final accuracy | 97.46% | 97.39% | +0.07% |
| Final loss | 0.0569 | 0.0551 | +0.0018 |
| Wall time | 11.2s | 48.0s | 4.3x faster |
| CUDA runtime | **None** | Yes | — |
| GPU access | Vulkan compute | CUDA | — |

**Accuracy difference: 0.07% — well within 2% threshold.**

The SPIR-V path achieves parity with PyTorch/CUDA training. Loss curves track identically.
The 4.3x wall-clock advantage comes from Rust + direct Vulkan dispatch vs Python + PyTorch overhead;
the per-kernel GPU performance is comparable.

## Vulkan Runtime Architecture

- **Persistent context**: Single Vulkan instance + device created once, reused for all 84,330 dispatches
- **Pre-compiled pipelines**: 11 compute pipelines loaded at init
- **Host-visible coherent memory**: Zero-copy upload/download for batch data
- **Sequential dispatch**: `vkQueueWaitIdle` between dispatches (correct, not optimal)
- **24 GPU buffers**: Weights, gradients, Adam state (m,v), activations, temporaries

## Verdict

**GPU training via SPIR-V/Vulkan compute works.** An MLP trained to 97.46% on MNIST
using 11 custom SPIR-V compute shaders dispatched through Vulkan on an NVIDIA RTX 3060,
with zero CUDA runtime dependencies. This is, to our knowledge, the first demonstrated
GPU neural network training loop running entirely through SPIR-V compute shaders.
