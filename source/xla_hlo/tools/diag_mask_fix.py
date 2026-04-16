#!/usr/bin/env python3
"""Test fix: replace -1e9 mask with -1e4 and re-run decoder through SPIR-V."""
import json, os, pathlib, subprocess, sys, shutil, tempfile
import numpy as np

BASE = pathlib.Path("<repo-root>/src/cuda-spirv")
FIXTURE = BASE / "xla_hlo" / "fixtures"
BIN = str(BASE / "target" / "release" / "xla_hlo")
DECODER_HLO = str(FIXTURE / "whisper_decoder.stablehlo.txt")
KERNEL_DIR = str(BASE / "kernels")

# Step 1: Read original mask and check its values
mask_path = FIXTURE / "whisper_dec_param_003_causal_mask.raw.f32"
mask = np.fromfile(str(mask_path), dtype=np.float32).reshape(1, 1, 64, 64)
print(f"Original mask: min={mask.min()}, max={mask.max()}, nonzero={np.count_nonzero(mask)}")

# Step 2: Create fixed mask with -1e4 instead of -1e9
mask_fixed = mask.copy()
mask_fixed[mask_fixed < -1e6] = -1e4  # Replace all -1e9 values with -1e4
print(f"Fixed mask: min={mask_fixed.min()}, max={mask_fixed.max()}, nonzero={np.count_nonzero(mask_fixed)}")

# Save fixed mask
scratch = pathlib.Path(tempfile.gettempdir())
fixed_mask_path = scratch / "whisper_causal_mask_fixed.raw.f32"
mask_fixed.tofile(str(fixed_mask_path))

# Step 3: Build inputs list with fixed mask
inputs_txt = (FIXTURE / "whisper_decoder_inputs.txt").read_text().strip()
input_paths = inputs_txt.split(",")

# Replace mask path (index 3) with fixed version
resolved = []
for i, p in enumerate(input_paths):
    full = str(BASE / p) if not os.path.isabs(p) else p
    if i == 3:
        full = str(fixed_mask_path)
    resolved.append(full)

# Step 4: Run xla_hlo with fixed mask
print("\n=== Running xla_hlo decoder with FIXED mask (-1e4) ===")
output_path = str(scratch / "whisper_decoder_fixmask_output.raw.f32")
report_path = str(scratch / "whisper_decoder_fixmask_report.json")

cmd = [
    BIN,
    "--input", DECODER_HLO,
    "--run",
    "--kernel-dir", KERNEL_DIR,
    "--device", "any",
    "--inputs-f32", ",".join(resolved),
    "--output-f32", output_path,
    "--report", report_path,
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
print(f"Return code: {result.returncode}")

if result.returncode != 0:
    print(f"FAILED: {result.stderr[:1000]}")
    sys.exit(1)

with open(report_path) as f:
    report = json.load(f)
print(f"Report status: {report.get('status')}")
print(f"Device: {report.get('execution', {}).get('device', {}).get('name')}")

# Step 5: Read and analyze output
logits = np.fromfile(output_path, dtype=np.float32).reshape(1, 64, 51865)
print(f"\nLogits range: min={logits.min():.6f}, max={logits.max():.6f}")
print(f"Any NaN? {np.any(np.isnan(logits))}")
print(f"Any Inf? {np.any(np.isinf(logits))}")

print("\nSPIR-V (fixed mask) logits per position:")
for pos in range(8):
    row = logits[0, pos, :]
    top5_idx = np.argsort(row)[-5:][::-1]
    top5_val = row[top5_idx]
    print(f"  pos {pos}: argmax={np.argmax(row):5d}, top5_idx={top5_idx.tolist()}, top5_val={[f'{v:.4f}' for v in top5_val]}")

# Step 6: Compare with TF numpy reference
print("\n=== TF Numpy reference (for comparison) ===")
# We already know from diag_decoder.py:
# pos 3: argmax=542 (expected next token)
# Let's just verify the argmax at pos 3
spirv_token = int(np.argmax(logits[0, 3, :]))
print(f"SPIR-V next token at pos 3: {spirv_token} (expected: 542)")
print(f"Match: {spirv_token == 542}")

print("\nDone.")
