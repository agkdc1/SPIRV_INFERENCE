#!/usr/bin/env python3
import argparse
import json
import os
import subprocess


def read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def shell(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as exc:
        return exc.output.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--intel", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = read(args.result)
    fixtures = read(args.fixtures)
    intel = read(args.intel)
    lines = []
    lines.append("# CUDA-SPIRV Advanced Neural Network Parity Report")
    lines.append("")
    lines.append(f"Overall verdict: **{result['overall_status']}**")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This run compares four fixed neural-network operator shapes: ReLU, Softmax, BatchNorm, and Conv1D. CUDA ran on the local RTX 3060 path. SPIR-V ran through Vulkan on the Intel HD Graphics 630 host. This is not a TensorFlow replacement claim.")
    lines.append("")
    lines.append("## Devices")
    lines.append("")
    lines.append(f"- CUDA baseline: `{shell('nvidia-smi -L 2>/dev/null || true')}`")
    lines.append(f"- SPIR-V runtime: `{intel.get('device', {}).get('name', 'unknown')}`")
    lines.append("")
    lines.append("## Kernel Results")
    lines.append("")
    lines.append("| Kernel | Status | Elements | Max ULP | Max Abs | Max Rel | Mismatches |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    display = {"relu": "ReLU", "softmax": "Softmax", "batchnorm": "BatchNorm", "conv1d": "Conv1D"}
    for name in ["relu", "softmax", "batchnorm", "conv1d"]:
        k = result["kernels"][name]
        lines.append(f"| {display[name]} | {k['status']} | {k['elements']} | {k['max_ulp_error']} | {k['max_abs_error']:.9g} | {k['max_rel_error']:.9g} | {k['mismatch_count']} |")
    lines.append("")
    lines.append("## IEEE 754 Analysis")
    lines.append("")
    lines.append("- ReLU required bit-exact output, including signed-zero behavior after max-with-zero semantics.")
    lines.append("- Softmax used max subtraction and one-workgroup tree reductions on both CUDA and SPIR-V; threshold was max ULP <= 8 and max absolute error <= 1e-6.")
    lines.append("- BatchNorm allowed sqrt/FMA differences; threshold was max ULP <= 8 and max absolute error <= 1e-6.")
    lines.append("- Conv1D used identical loop ordering in CUDA and GLSL; threshold was max ULP <= 16 and max relative error <= 1e-5.")
    lines.append("")
    lines.append("## Fixture Hashes")
    lines.append("")
    for name, meta in sorted(fixtures["files"].items()):
        lines.append(f"- `{name}`: `{meta['sha256']}`")
    lines.append("")
    lines.append("## Output Hashes")
    lines.append("")
    for name in ["relu", "softmax", "batchnorm", "conv1d"]:
        k = result["kernels"][name]
        lines.append(f"- `{name}` CUDA: `{k['cuda_sha256']}`")
        lines.append(f"- `{name}` Intel SPIR-V: `{k['spirv_sha256']}`")
    lines.append("")
    lines.append("## Artifact Paths")
    lines.append("")
    root = os.path.abspath(os.path.join(os.path.dirname(args.out), ".."))
    lines.append(f"- Result JSON: `{os.path.abspath(args.result)}`")
    lines.append(f"- Intel SPIR-V report: `{os.path.abspath(args.intel)}`")
    lines.append(f"- Artifact root: `{root}`")
    lines.append("")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
