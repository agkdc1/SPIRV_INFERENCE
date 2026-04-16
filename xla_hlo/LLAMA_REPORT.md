# TinyLlama-1.1B SPIR-V Fixture Report

## Scope

Fixed-prompt autoregressive TinyLlama-1.1B fixture executed through the current HLO-to-SPIR-V/Vulkan path.

## Public Result

| Metric | Result |
| --- | ---: |
| Prompt | "The capital of France is" |
| Generated tokens compared | 20 |
| CPU reference match on RTX 3060 Vulkan path | 20/20 |
| CPU reference match on Intel HD 630 Vulkan path | 20/20 |
| CUDA/cuBLAS/cuDNN used by SPIR-V path | No |

## Boundary

This is a fixed-fixture parity result. It is not a general LLM runtime claim, not a performance claim, and not a claim of complete model/operator coverage.

## Nonclaims

- No complete LLM serving stack is claimed.
- No production inference framework is claimed.
- No general CUDA replacement claim is made from this fixture alone.
