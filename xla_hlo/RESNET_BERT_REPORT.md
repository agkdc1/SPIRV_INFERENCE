# ResNet-50 and BERT SPIR-V Fixture Report

## Scope

Fixed ResNet-50 image-classification and BERT-base tensor fixtures compared across TF CPU reference, NVIDIA RTX 3060 Vulkan path, and Intel HD 630 Vulkan path.

## Public Result

| Fixture | Compared Elements | RTX 3060 Mismatches | Intel HD 630 Mismatches | Epsilon |
| --- | ---: | ---: | ---: | ---: |
| ResNet-50 | 1000 | 0 | 0 | 0.001 |
| BERT-base | 12288 | 0 | 0 | 0.001 |

## Boundary

The current evidence covers these fixed fixtures and their exercised operator set. It does not claim complete TensorFlow, StableHLO, BERT, or ResNet coverage.

## Nonclaims

- No production inference framework is claimed.
- No complete model-family coverage is claimed.
- No hardware portability guarantee is claimed beyond the recorded devices.
