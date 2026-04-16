#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import struct


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def write_f32(path, values):
    data = b"".join(struct.pack("<f", f32(v)) for v in values)
    with open(path, "wb") as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def relu_ref(xs):
    out = []
    for x in xs:
        out.append(f32(x if x > 0.0 else 0.0))
    return out


def softmax_ref(xs):
    m = max(xs)
    exps = [f32(math.exp(float(f32(x - m)))) for x in xs]
    total = f32(0.0)
    for e in exps:
        total = f32(total + e)
    return [f32(e / total) for e in exps]


def batchnorm_ref(xs, params):
    mean, var, gamma, beta, eps = params
    inv = f32(1.0 / math.sqrt(float(f32(var + eps))))
    return [f32(f32(f32(x - mean) * inv) * gamma + beta) for x in xs]


def conv1d_ref(xs, filt):
    n, ic, oc, k = 16, 64, 32, 3
    out = []
    for row in range(n):
        for o in range(oc):
            acc = f32(0.0)
            for kk in range(k):
                src_row = row + kk - 1
                if src_row < 0 or src_row >= n:
                    continue
                for c in range(ic):
                    xv = xs[src_row * ic + c]
                    wv = filt[(kk * ic + c) * oc + o]
                    acc = f32(acc + f32(xv * wv))
            out.append(acc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    relu_in = []
    for i in range(4096):
        if i == 0:
            relu_in.append(f32(-0.0))
        elif i == 1:
            relu_in.append(f32(0.0))
        elif i % 17 == 0:
            relu_in.append(f32((i % 31 - 15) * 1.0e-7))
        else:
            relu_in.append(f32(((i * 37) % 211 - 105) * 0.03125))

    softmax_in = [f32(((i * 13) % 37 - 18) * 0.0625) for i in range(128)]
    batch_in = [f32(((i * 19) % 127 - 63) * 0.015625) for i in range(256)]
    batch_params = [f32(0.125), f32(1.0), f32(1.0), f32(0.0), f32(0.0)]
    conv_in = [f32(((i * 17) % 97 - 48) * 0.00390625) for i in range(16 * 64)]
    conv_filter = [f32(((i * 11) % 53 - 26) * 0.001953125) for i in range(3 * 64 * 32)]

    tensors = {
        "relu_input": relu_in,
        "relu_ref": relu_ref(relu_in),
        "softmax_input": softmax_in,
        "softmax_ref": softmax_ref(softmax_in),
        "batchnorm_input": batch_in,
        "batchnorm_params": batch_params,
        "batchnorm_ref": batchnorm_ref(batch_in, batch_params),
        "conv1d_input": conv_in,
        "conv1d_filter": conv_filter,
        "conv1d_ref": conv1d_ref(conv_in, conv_filter),
    }

    manifest = {
        "format": "raw little-endian float32",
        "shapes": {
            "relu": [4096],
            "softmax": [128],
            "batchnorm": [256],
            "batchnorm_params": ["mean", "variance", "gamma", "beta", "epsilon"],
            "conv1d_input": [16, 64],
            "conv1d_filter": [3, 64, 32],
            "conv1d_output": [16, 32],
        },
        "thresholds": {
            "relu": {"bit_exact": True, "max_ulp": 0, "max_abs": 0.0},
            "softmax": {"max_ulp": 8, "max_abs": 1.0e-6},
            "batchnorm": {"max_ulp": 8, "max_abs": 1.0e-6},
            "conv1d": {"max_ulp": 16, "max_rel": 1.0e-5},
        },
        "files": {},
    }
    for name, values in tensors.items():
        filename = f"{name}.raw.f32"
        path = os.path.join(args.out, filename)
        sha = write_f32(path, values)
        manifest["files"][name] = {"path": filename, "sha256": sha, "elements": len(values)}

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main()
