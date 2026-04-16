#!/usr/bin/env python3
import argparse

import torch
import torchvision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = torchvision.models.squeezenet1_0(
        weights=torchvision.models.SqueezeNet1_0_Weights.DEFAULT
    ).eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )


if __name__ == "__main__":
    main()
