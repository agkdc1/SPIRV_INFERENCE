# IEEE 754 Analysis

## Policy

All runtime comparisons use float32 outputs and explicit epsilon thresholds. Bit-exact equality is accepted when observed, but accumulated convolutional workloads are judged by maximum absolute error, relative error, mismatch count, and ULP range where the runner reports ULP.

## Op Analysis

| Op | Expected behavior |
| --- | --- |
| vectorAdd | Bit-exact for matching inputs because each element performs one IEEE 754 addition. |
| matmul | Bit-exact for the validated fixed fixture; larger shapes may vary if accumulation order changes. |
| ReLU / ReLU6 | Bit-exact thresholding except for NaN payload details, which are outside the supported proof fixtures. |
| Softmax | Small ULP variance is allowed because exponentiation, reduction, and reciprocal operations may differ across devices. |
| BatchNorm | Small ULP variance is allowed due to multiply-add ordering. |
| Conv1D / Conv2D / DepthwiseConv2D | ULP variance is expected because accumulation order and fused multiply-add behavior can differ by driver/device. |
| AvgPool / GlobalAvgPool | ULP variance is expected for reductions; epsilon gate remains authoritative. |
| FullyConnected | Equivalent to a small matrix/vector reduction; ULP variance follows accumulation order. |

## Accumulation Order

The release kernels use deterministic shader-local loop ordering for each output element. Cross-device byte identity is not required for reductions. Release acceptance requires zero mismatches under the documented epsilon threshold and records maximum absolute and relative error.
