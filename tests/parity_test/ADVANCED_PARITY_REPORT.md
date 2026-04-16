# CUDA-SPIRV Advanced Neural Network Parity Report

Overall verdict: **pass**

## Scope

This run compares four fixed neural-network operator shapes: ReLU, Softmax, BatchNorm, and Conv1D. CUDA ran on the local RTX 3060 path. SPIR-V ran through Vulkan on the Intel HD Graphics 630 host. This is not a TensorFlow replacement claim.

## Devices

- CUDA baseline: `GPU 0: NVIDIA GeForce RTX 3060 (UUID: GPU-1a8a74f7-080a-5551-f23f-c038c3408e09)`
- SPIR-V runtime: `Intel(R) HD Graphics 630 (KBL GT2)`

## Kernel Results

| Kernel | Status | Elements | Max ULP | Max Abs | Max Rel | Mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ReLU | pass | 4096 | 0 | 0 | 0 | 0 |
| Softmax | pass | 128 | 2 | 1.86264515e-09 | 1.88846196e-07 | 0 |
| BatchNorm | pass | 256 | 0 | 0 | 0 | 0 |
| Conv1D | pass | 512 | 0 | 0 | 0 | 0 |

## IEEE 754 Analysis

- ReLU required bit-exact output, including signed-zero behavior after max-with-zero semantics.
- Softmax used max subtraction and one-workgroup tree reductions on both CUDA and SPIR-V; threshold was max ULP <= 8 and max absolute error <= 1e-6.
- BatchNorm allowed sqrt/FMA differences; threshold was max ULP <= 8 and max absolute error <= 1e-6.
- Conv1D used identical loop ordering in CUDA and GLSL; threshold was max ULP <= 16 and max relative error <= 1e-5.

## Fixture Hashes

- `batchnorm_input`: `e4be769575fd77f769ffc4a77d45728c667399a45821dd80c7614807a11236a0`
- `batchnorm_params`: `5b5867c269a649befc08bc9878db0a88dee1fb6e5d6ae494fc62adbd6267b588`
- `batchnorm_ref`: `de2d9d9458180dbb8a8b518e01d592ccfcc4764af8ee14f475c44a7b5cff47ab`
- `conv1d_filter`: `02cf21162dc7c38b5a035b9152aed00d169baaf08b9070be0b36d6d31d9b5611`
- `conv1d_input`: `01e1926cf274e88d817b7e7a69bd6e6ac0d3c5196e18f5248c6c0a4815abc57f`
- `conv1d_ref`: `10072ca9d332907dec423b16a417088c536f5ab5d132ce7455050154a087db66`
- `relu_input`: `b7c21d15c89d808e3ed5e9a1f56bc9e7551f2947c0b0d0e8ca8a8d20c6f5589f`
- `relu_ref`: `7669adf27dc478c78bd2711ff8252f8a514b94f3735cda40338bfa885edfa810`
- `softmax_input`: `f6727bd02b0294365edec757066883a46716f07e10401c853f68471f43d203d9`
- `softmax_ref`: `2c9f71ca72b4f044c9e14cf0f1c49182c9d6ecb64f03ef0bde7d99dcbe83f7b3`

## Output Hashes

- `relu` CUDA: `7669adf27dc478c78bd2711ff8252f8a514b94f3735cda40338bfa885edfa810`
- `relu` Intel SPIR-V: `7669adf27dc478c78bd2711ff8252f8a514b94f3735cda40338bfa885edfa810`
- `softmax` CUDA: `198734445925b872ca7169b3d3b1ad81d9187310fa76a7306867a37e7b88b593`
- `softmax` Intel SPIR-V: `95242b0638bf3f098ff651702e8ae824f93e6fc6d2009c9c549e4dbd90dbd321`
- `batchnorm` CUDA: `de2d9d9458180dbb8a8b518e01d592ccfcc4764af8ee14f475c44a7b5cff47ab`
- `batchnorm` Intel SPIR-V: `de2d9d9458180dbb8a8b518e01d592ccfcc4764af8ee14f475c44a7b5cff47ab`
- `conv1d` CUDA: `10072ca9d332907dec423b16a417088c536f5ab5d132ce7455050154a087db66`
- `conv1d` Intel SPIR-V: `10072ca9d332907dec423b16a417088c536f5ab5d132ce7455050154a087db66`

## Artifact Paths

- Result JSON: `<repo-root>/src/cuda-spirv/parity_test/out/compare/advanced_parity_result.json`
- Intel SPIR-V report: `<repo-root>/src/cuda-spirv/parity_test/out/intel/intel_spirv_report.json`
- Artifact root: `<repo-root>/src/cuda-spirv`

