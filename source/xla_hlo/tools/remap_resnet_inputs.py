#!/usr/bin/env python3
import json
import re
from pathlib import Path

import tensorflow as tf


def parse_shape(text: str):
    inside = text[text.index("[") + 1 : text.index("]")]
    if not inside.strip():
        return []
    return [int(part) for part in inside.split(",")]


def canonical_layer(op_name: str) -> str:
    # XLA metadata uses names such as
    # resnet50_1/conv2_block1_1_bn_1/Reshape_3. Keras weight paths use
    # conv2_block1_1_bn/gamma. Strip the model prefix and the export suffix.
    layer = op_name.split("/")
    if len(layer) < 2:
        return op_name
    layer = layer[1]
    return re.sub(r"_1$", "", layer)


def parameter_roles(hlo: str):
    entry = hlo[hlo.index("ENTRY ") :]
    params = {}
    lines = entry.splitlines()
    for line_no, line in enumerate(lines):
        match = re.search(
            r"%arg(\d+)\.1 = (f32\[[^\]]*\])(?:\{[^}]*\})? parameter\((\d+)\)",
            line,
        )
        if not match:
            continue
        arg_name_idx = int(match.group(1))
        param_idx = int(match.group(3))
        if arg_name_idx != param_idx:
            raise SystemExit(f"arg/parameter mismatch: arg{arg_name_idx} parameter({param_idx})")
        params[param_idx] = {"shape": parse_shape(match.group(2)), "line_no": line_no}
    roles = {}
    for param_idx, info in params.items():
        if param_idx == 0:
            roles[param_idx] = ("input", "input")
            continue
        token = f"%arg{param_idx}.1"
        consumers = [line for line in lines[info["line_no"] + 1 :] if token in line]
        if not consumers:
            raise SystemExit(f"no consumer found for parameter {param_idx}")
        consumer = consumers[0]
        op_name_match = re.search(r'op_name="([^"]+)"', consumer)
        if not op_name_match:
            raise SystemExit(f"no op_name metadata for parameter {param_idx}: {consumer}")
        op_name = op_name_match.group(1)
        layer = canonical_layer(op_name)
        if "/convolution" in op_name or "/MatMul" in op_name:
            role = "kernel"
        elif "BiasAdd" in op_name:
            role = "bias"
        elif "/Reshape_1" in op_name:
            role = "moving_variance"
        elif "/Reshape_2" in op_name:
            role = "beta"
        elif "/Reshape_3" in op_name:
            role = "gamma"
        elif "/Reshape" in op_name:
            if layer.endswith("_conv") or layer == "predictions":
                role = "bias"
            else:
                role = "moving_mean"
        else:
            raise SystemExit(f"unsupported parameter consumer for {param_idx}: {consumer}")
        roles[param_idx] = (layer, role)
    return params, roles


def main():
    hlo = Path("xla_hlo/fixtures/resnet50.stablehlo.txt").read_text()
    report = json.loads(Path("xla_hlo/out/tf_resnet50/report.json").read_text())
    params, roles = parameter_roles(hlo)

    model = tf.keras.applications.ResNet50(weights="imagenet")
    weight_index = {}
    for fixture, weight in zip(report["fixtures"][1:], model.weights):
        weight_index[getattr(weight, "path")] = fixture["path"]

    ordered = []
    mapping = []
    for idx in range(len(params)):
        shape = tuple(params[idx]["shape"])
        if idx == 0:
            path = report["fixtures"][0]["path"]
            key = "input"
        else:
            layer, role = roles[idx]
            key = f"{layer}/{role}"
            path = weight_index.get(key)
            if path is None:
                raise SystemExit(f"no Keras fixture for parameter {idx}: {key} shape {shape}")
        ordered.append(path)
        mapping.append({"parameter": idx, "shape": list(shape), "role": key, "path": path})

    out = Path("xla_hlo/fixtures/resnet50_inputs_hlo_order.txt")
    out.write_text("\n".join(ordered) + "\n")
    Path("xla_hlo/reports/resnet50_input_hlo_order_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n"
    )
    print(f"wrote {out} with {len(ordered)} inputs")


if __name__ == "__main__":
    main()
