# Whisper-Tiny SPIR-V Fixture Report

## Scope

Fixed Whisper-tiny speech-to-text fixture compared across TF CPU reference, NVIDIA RTX 3060 Vulkan path, and Intel HD 630 Vulkan path.

## Public Result

| Metric | RTX 3060 Vulkan | Intel HD 630 Vulkan |
| --- | ---: | ---: |
| Encoder output parity vs TF CPU | Pass | Pass |
| Decoder token sequence parity vs TF CPU | Pass | Pass |
| Decoder token IDs matched | 8/8 | 8/8 |
| Encoder mismatch count | 0 | 0 |
| Epsilon | 0.01 | 0.01 |

## Boundary

This is a fixed audio fixture result. It is not a claim of complete Whisper coverage, complete speech-domain coverage, or production speech-to-text readiness.

## Nonclaims

- No general audio model runtime is claimed.
- No production deployment claim is made.
- No safety or certification claim is made.
