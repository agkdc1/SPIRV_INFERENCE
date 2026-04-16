# SPIRV_INFERENCE

Public source and evidence package for selected neural-network inference
fixtures and a small training workload exercised through Vulkan/SPIR-V on
commodity GPUs.

## Tiger Cat Result

| Runtime | Top-1 class | Label | Confidence |
| --- | ---: | --- | ---: |
| PyTorch | 282 | tiger cat | 0.4743793309 |
| CUDA RTX 3060 | 282 | tiger cat | 0.474384 |
| Intel SPIR-V | 282 | tiger cat | 0.4743832946 |

## Demonstrated Models

| Model | Domain | Result |
| --- | --- | --- |
| MobileNetV2 | Image classification | Tiger cat top-1 parity |
| EfficientNet-Lite0 | Image classification | Cross-device SPIR-V parity |
| ResNet-50 | Image classification | HLO to SPIR-V path exercised |
| BERT | NLP | HLO to SPIR-V path exercised |
| Whisper-tiny | Speech | Token-for-token output parity |
| TinyLlama-1.1B | LLM | 20/20 generated tokens match CPU reference |
| MNIST MLP training | Training | 97.46% accuracy after 5 epochs |

## Included

- `source/tflite_loader/`: TFLite graph decoding, scheduling, tensor model, and
  Vulkan-backed execution code.
- `source/xla_hlo/`: StableHLO parser/compiler source, export/reference tools,
  selected HLO fixtures, and report fixtures for ResNet-50, BERT, Whisper,
  TinyLlama, and Stable Diffusion UNet paths.
- `source/training/`: Vulkan training source and reports. Dataset files are not
  vendored.
- `source/inference_poc/`: CUDA baseline, GLSL shader, Vulkan runner, and
  deterministic small inference fixture code.
- `runtime/chained_graph_runtime/`: chained Vulkan graph runtime source.
- `kernels/`: GLSL compute shader sources and compiled SPIR-V modules for the
  packaged kernel set.
- `tests/parity_test/`: parity runner and result fixtures.
- `scripts/fetch_model_sources.sh`: clones upstream TensorFlow and Whisper
  source trees into `external/`.
- Model and kernel parity summaries.
- Inference and training reports.
- IEEE 754 comparison notes.
- SHA-256 manifest for the public source and evidence files.

## Reproduce Source Inputs

```bash
./scripts/fetch_model_sources.sh
```

This clones:

- `https://github.com/tensorflow/tensorflow.git`
- `https://github.com/openai/whisper.git`

TensorFlow, Whisper, downloaded model weights, `.tflite` model files, and raw
tensor dumps are not vendored in this repository.

## Blockers

- Quantized TFLite models are not supported.
- Dynamic tensors are not supported.
- Unsupported fused activations and missing kernel implementations fail closed.
- Quantized model files and large raw tensor/model artifacts are intentionally
  excluded from the repository.

## License

Apache 2.0.
