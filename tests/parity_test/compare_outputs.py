#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import struct


THRESHOLDS = {
    "relu": {"max_ulp": 0, "max_abs": 0.0, "bit_exact": True},
    "softmax": {"max_ulp": 8, "max_abs": 1.0e-6},
    "batchnorm": {"max_ulp": 8, "max_abs": 1.0e-6},
    "conv1d": {"max_ulp": 16, "max_rel": 1.0e-5},
}


def read_f32(path):
    data = open(path, "rb").read()
    if len(data) % 4:
        raise ValueError(f"{path} is not f32 aligned")
    return data, list(struct.unpack("<" + "f" * (len(data) // 4), data))


def ordered(v):
    bits = struct.unpack("<i", struct.pack("<f", v))[0]
    return -2147483648 - bits if bits < 0 else bits


def ulp(a, b):
    return abs(ordered(a) - ordered(b))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compare_one(name, cuda_dir, spirv_dir):
    cuda_data, cuda_values = read_f32(os.path.join(cuda_dir, name, "output.raw.f32"))
    spirv_data, spirv_values = read_f32(os.path.join(spirv_dir, name, "output.raw.f32"))
    if len(cuda_values) != len(spirv_values):
        raise ValueError(f"{name}: element count mismatch")
    t = THRESHOLDS[name]
    max_abs = 0.0
    max_rel = 0.0
    max_ulp = 0
    mismatches = 0
    bit_mismatches = 0
    for a, b in zip(cuda_values, spirv_values):
        abserr = abs(a - b)
        rel = abserr / max(abs(a), 1.0e-30)
        u = ulp(a, b)
        max_abs = max(max_abs, abserr)
        max_rel = max(max_rel, rel)
        max_ulp = max(max_ulp, u)
        if struct.pack("<f", a) != struct.pack("<f", b):
            bit_mismatches += 1
        if name == "relu":
            bad = u != 0
        elif "max_abs" in t:
            bad = u > t["max_ulp"] or abserr > t["max_abs"]
        else:
            bad = u > t["max_ulp"] or rel > t["max_rel"]
        if bad:
            mismatches += 1
    status = "pass" if mismatches == 0 else "fail"
    return {
        "status": status,
        "elements": len(cuda_values),
        "cuda_sha256": sha(cuda_data),
        "spirv_sha256": sha(spirv_data),
        "threshold": t,
        "mismatch_count": mismatches,
        "bit_mismatch_count": bit_mismatches,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "max_ulp_error": max_ulp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", required=True)
    ap.add_argument("--spirv", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    kernels = ["relu", "softmax", "batchnorm", "conv1d"]
    result = {"kernels": {}, "fixture_manifest": os.path.join(args.fixtures, "manifest.json")}
    for k in kernels:
        result["kernels"][k] = compare_one(k, args.cuda, args.spirv)
    result["overall_status"] = "pass" if all(v["status"] == "pass" for v in result["kernels"].values()) else "fail"
    result["epsilon_policy"] = {
        "relu": "bit-exact required",
        "softmax": "max ULP <= 8 and max abs <= 1e-6",
        "batchnorm": "max ULP <= 8 and max abs <= 1e-6",
        "conv1d": "max ULP <= 16 and max relative <= 1e-5",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main()
