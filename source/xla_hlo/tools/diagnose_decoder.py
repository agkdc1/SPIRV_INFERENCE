#!/usr/bin/env python3
"""Diagnose why decoder produces all-zero tokens."""
import numpy as np
import sys

def check(name, path, expected_elements=None):
    data = np.fromfile(path, dtype=np.float32)
    print(f"\n=== {name} ({path}) ===")
    print(f"  Elements: {len(data)}")
    nan_count = np.isnan(data).sum()
    inf_count = np.isinf(data).sum()
    print(f"  NaN: {nan_count}  Inf: {inf_count}")
    if nan_count == 0 and inf_count == 0:
        print(f"  Min: {data.min():.6f}  Max: {data.max():.6f}  Mean: {data.mean():.6f}")
        print(f"  Zeros: {(data == 0).sum()} / {len(data)}")
    elif nan_count > 0:
        print(f"  ALL NaN" if nan_count == len(data) else f"  {nan_count}/{len(data)} NaN")
    return data, nan_count, inf_count

# Check encoder SPIR-V output
enc_spirv, enc_nan, _ = check("Encoder SPIR-V output",
    "xla_hlo/out/whisper_encoder_spirv_local.raw.f32")

# Check encoder TF reference
enc_tf, _, _ = check("Encoder TF reference",
    "xla_hlo/out/whisper_encoder_tf_output.raw.f32")

if enc_nan > 0:
    print("\n*** ENCODER OUTPUT IS ALL NaN ***")
    print("The decoder receives NaN encoder output, so all downstream computation is NaN.")
    print("The encoder report said 'pass' but the output file contains NaN.")
    print("Root cause is in the ENCODER, not the decoder.")
    print("\nThe encoder has tanh ops too — the fix may not have been applied")
    print("when the encoder was run (encoder was run BEFORE the tanh fix).")
    sys.exit(1)

# If encoder is fine, compare
if enc_nan == 0:
    diff = np.abs(enc_spirv - enc_tf)
    print(f"\n=== Encoder parity ===")
    print(f"  Max abs diff: {diff.max():.6f}")
    print(f"  Mean abs diff: {diff.mean():.6f}")
    print(f"  Within 0.01: {(diff < 0.01).sum()} / {len(diff)}")
