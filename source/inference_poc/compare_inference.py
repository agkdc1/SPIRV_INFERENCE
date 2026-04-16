#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import struct


def read_f32(path):
    data = open(path, "rb").read()
    if len(data) % 4:
        raise ValueError(f"{path} is not f32-aligned")
    return data, list(struct.unpack("<" + "f" * (len(data) // 4), data))


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered(v):
    bits = struct.unpack("<i", struct.pack("<f", v))[0]
    return -2147483648 - bits if bits < 0 else bits


def ulp(a, b):
    return abs(ordered(a) - ordered(b))


def top1(values):
    cls, conf = max(enumerate(values), key=lambda item: item[1])
    return {"class": cls, "confidence": conf}


def compare(a_values, b_values, max_abs_threshold, max_rel_threshold, max_ulp_threshold):
    if len(a_values) != len(b_values):
        raise ValueError("element count mismatch")
    max_abs = 0.0
    max_rel = 0.0
    max_ulp = 0
    mismatches = 0
    for a, b in zip(a_values, b_values):
        abs_err = abs(a - b)
        rel_err = abs_err / max(abs(a), 1.0e-30)
        ulp_err = ulp(a, b)
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        max_ulp = max(max_ulp, ulp_err)
        if (
            abs_err > max_abs_threshold
            or rel_err > max_rel_threshold
            or ulp_err > max_ulp_threshold
        ):
            mismatches += 1
    return {
        "mismatch_count": mismatches,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "max_ulp_error": max_ulp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-report", required=True)
    ap.add_argument("--baseline-output", required=True)
    ap.add_argument("--intel-report", required=True)
    ap.add_argument("--intel-output", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--kernel-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    baseline_data, baseline = read_f32(args.baseline_output)
    intel_data, intel = read_f32(args.intel_output)
    thresholds = {
        "max_abs_error": 1.0e-5,
        "max_rel_error": 1.0e-4,
        "max_ulp_error": 64,
    }
    metrics = compare(
        baseline,
        intel,
        thresholds["max_abs_error"],
        thresholds["max_rel_error"],
        thresholds["max_ulp_error"],
    )
    baseline_report = json.load(open(args.baseline_report, "r", encoding="utf-8"))
    intel_report = json.load(open(args.intel_report, "r", encoding="utf-8"))
    fixture_hashes = {}
    for name in sorted(os.listdir(args.fixtures)):
        path = os.path.join(args.fixtures, name)
        if os.path.isfile(path):
            fixture_hashes[name] = sha_file(path)
    spirv_hashes = {}
    for name in sorted(os.listdir(args.kernel_dir)):
        path = os.path.join(args.kernel_dir, name)
        if name.endswith(".spv"):
            spirv_hashes[name] = sha_file(path)
    result = {
        "status": "pass" if metrics["mismatch_count"] == 0 else "fail",
        "baseline_kind": baseline_report.get("baseline_kind", "unknown"),
        "baseline_device": baseline_report.get("device"),
        "intel_device": intel_report.get("device"),
        "thresholds": thresholds,
        "elements": len(baseline),
        "mismatch_count": metrics["mismatch_count"],
        "max_abs_error": metrics["max_abs_error"],
        "max_rel_error": metrics["max_rel_error"],
        "max_ulp_error": metrics["max_ulp_error"],
        "top1": {"baseline": top1(baseline), "intel": top1(intel)},
        "output_hashes": {
            "baseline_sha256": hashlib.sha256(baseline_data).hexdigest(),
            "intel_sha256": hashlib.sha256(intel_data).hexdigest(),
        },
        "fixture_hashes": fixture_hashes,
        "spirv_hashes": spirv_hashes,
        "reports": {
            "baseline": args.baseline_report,
            "intel": args.intel_report,
        },
        "non_claims": {
            "claims_tensorflow_runtime": False,
            "claims_full_tensorflow_gpu_support": False,
            "uses_cublas": False,
            "uses_cudnn": False,
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
