# SPIR-V Training Fixture Report

## Scope

Small MNIST MLP training proof-of-concept exercised through Vulkan/SPIR-V compute on NVIDIA RTX 3060.

## Public Result

| Metric | SPIR-V/Vulkan | PyTorch/CUDA Reference |
| --- | ---: | ---: |
| Final test accuracy | 97.46% | 97.39% |
| Final loss | 0.0569 | 0.0551 |
| Epochs | 5 | 5 |
| CUDA runtime used by SPIR-V path | No | N/A |

## Boundary

This is a small training fixture, not a general training framework. The result shows that this fixed workload can be trained through the current Vulkan/SPIR-V path within the recorded tolerance.

## Nonclaims

- No general neural-network training stack is claimed.
- No production training performance claim is made.
- No claim of being first, unique, or comprehensive is made.
