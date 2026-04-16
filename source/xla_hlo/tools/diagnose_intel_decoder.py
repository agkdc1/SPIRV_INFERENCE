#!/usr/bin/env python3
"""Diagnose Intel iGPU decoder issue — check step 0 logits for NaN."""
import json
import os
import struct
import subprocess
import sys
import tempfile
import time


def read_f32(path):
    with open(path, "rb") as f:
        data = f.read()
    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data))


def write_f32(path, values):
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{len(values)}f", *values))


def main():
    import math

    fdir = sys.argv[1] if len(sys.argv) > 1 else "fixtures"
    xla_hlo = sys.argv[2] if len(sys.argv) > 2 else "./xla_hlo"

    n_text_ctx = 64
    n_text_state = 384
    n_vocab = 51865

    SOT, EN, TRANSCRIBE, NO_TIMESTAMPS = 50258, 50259, 50360, 50364

    # Load encoder output
    enc_path = os.path.join("out", "whisper_encoder_spirv_intel.raw.f32")
    encoder_output = read_f32(enc_path)
    enc_nan = sum(1 for v in encoder_output if math.isnan(v))
    enc_inf = sum(1 for v in encoder_output if math.isinf(v))
    print(f"Encoder output: {len(encoder_output)} elements, NaN={enc_nan}, Inf={enc_inf}")
    if enc_nan > 0:
        print("ENCODER OUTPUT HAS NaN — decoder will fail")
        return 1

    # Load embedding tables
    token_emb_table = read_f32(os.path.join(fdir, "whisper_token_embedding_table.raw.f32"))
    pos_emb_table = read_f32(os.path.join(fdir, "whisper_pos_embedding_table.raw.f32"))

    with open(os.path.join(fdir, "whisper_decoder_inputs.txt")) as f:
        all_dec_paths = f.read().strip().split(",")
    dec_weight_paths = all_dec_paths[4:]

    # Single step (step 0)
    tokens = [SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]
    padded = tokens + [0] * (n_text_ctx - len(tokens))

    tok_emb = []
    for t in padded:
        offset = t * n_text_state
        tok_emb.extend(token_emb_table[offset:offset + n_text_state])

    pos_emb = pos_emb_table[:n_text_ctx * n_text_state]

    causal = []
    for r in range(n_text_ctx):
        for c in range(n_text_ctx):
            if c > r or c >= len(tokens):
                causal.append(-1e9)
            else:
                causal.append(0.0)

    tmpdir = tempfile.mkdtemp(prefix="diag_")
    enc_tmp = os.path.join(tmpdir, "enc.raw.f32")
    tok_path = os.path.join(tmpdir, "tok.raw.f32")
    pos_path = os.path.join(tmpdir, "pos.raw.f32")
    mask_path = os.path.join(tmpdir, "mask.raw.f32")
    write_f32(enc_tmp, encoder_output)
    write_f32(tok_path, tok_emb)
    write_f32(pos_path, pos_emb)
    write_f32(mask_path, causal)

    # Check inputs for NaN
    for name, vals in [("tok_emb", tok_emb), ("pos_emb", pos_emb), ("mask", causal)]:
        nans = sum(1 for v in vals if math.isnan(v))
        print(f"  {name}: {len(vals)} elements, NaN={nans}")

    step_inputs = [enc_tmp, tok_path, pos_path, mask_path] + dec_weight_paths
    step_output = os.path.join(tmpdir, "logits.raw.f32")
    step_report = os.path.join(tmpdir, "report.json")

    cmd = [
        xla_hlo, "--input", os.path.join(fdir, "..", "fixtures", "whisper_decoder.stablehlo.txt"),
        "--run", "--kernel-dir", "kernels", "--device", "any",
        "--inputs-f32", ",".join(step_inputs),
        "--output-f32", step_output,
        "--report", step_report,
        "--nan-trace",
    ]
    # Try without --nan-trace first (may not be supported)
    cmd_nontrace = [c for c in cmd if c != "--nan-trace"]

    print(f"\nRunning decoder step 0...")
    result = subprocess.run(cmd_nontrace, capture_output=True, text=True, timeout=300)
    print(f"  Exit code: {result.returncode}")
    if result.stderr:
        print(f"  Stderr: {result.stderr[:500]}")

    if result.returncode != 0:
        # Try with fixtures path adjusted
        cmd_nontrace[2] = os.path.join(fdir, "whisper_decoder.stablehlo.txt")
        result = subprocess.run(cmd_nontrace, capture_output=True, text=True, timeout=300)
        print(f"  Retry exit code: {result.returncode}")

    if os.path.exists(step_output):
        logits = read_f32(step_output)
        total = len(logits)
        nans = sum(1 for v in logits if math.isnan(v))
        infs = sum(1 for v in logits if math.isinf(v))
        zeros = sum(1 for v in logits if v == 0.0)
        print(f"\nLogits: {total} elements")
        print(f"  NaN: {nans} ({100*nans/total:.1f}%)")
        print(f"  Inf: {infs}")
        print(f"  Zeros: {zeros}")

        if nans == 0:
            # Show logits at pos=3 (current position)
            pos = 3
            offset = pos * n_vocab
            pos_logits = logits[offset:offset + n_vocab]
            best_i = max(range(len(pos_logits)), key=lambda i: pos_logits[i])
            print(f"\n  Pos {pos} argmax: token {best_i}, value {pos_logits[best_i]:.6f}")
            print(f"  Pos {pos} top-5:")
            indexed = sorted(enumerate(pos_logits), key=lambda x: -x[1])[:5]
            for idx, val in indexed:
                print(f"    token {idx}: {val:.6f}")
            print(f"  Pos {pos} logits[0]: {pos_logits[0]:.6f}")
            print(f"  Pos {pos} min: {min(pos_logits):.6f}, max: {max(pos_logits):.6f}")
        else:
            # Find first non-NaN position
            for i, v in enumerate(logits):
                if not math.isnan(v):
                    print(f"  First non-NaN at index {i}: {v}")
                    break
            else:
                print("  ALL LOGITS ARE NaN")
    else:
        print("No output file generated")

    return 0


if __name__ == "__main__":
    sys.exit(main())
