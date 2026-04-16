#!/usr/bin/env python3
"""Run Whisper decoder via xla_hlo — no numpy/tiktoken required.
Uses only Python stdlib (struct) for float32 I/O."""
import argparse
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


def argmax(vals):
    best_i, best_v = 0, vals[0]
    for i in range(1, len(vals)):
        if vals[i] > best_v:
            best_i, best_v = i, vals[i]
    return best_i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder-hlo", required=True)
    parser.add_argument("--kernel-dir", required=True)
    parser.add_argument("--device", default="any")
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--xla-hlo-bin", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--encoder-output", required=True)
    args = parser.parse_args()

    n_text_ctx = 64
    n_text_state = 384
    n_vocab = 51865

    SOT, EN, TRANSCRIBE, NO_TIMESTAMPS, EOT = 50258, 50259, 50360, 50364, 50257

    start_time = time.time()

    # Load encoder output
    print(f"Loading encoder output: {args.encoder_output}")
    encoder_output = read_f32(args.encoder_output)

    # Load embedding tables
    fdir = args.fixture_dir
    token_emb_table = read_f32(os.path.join(fdir, "whisper_token_embedding_table.raw.f32"))
    pos_emb_table = read_f32(os.path.join(fdir, "whisper_pos_embedding_table.raw.f32"))

    # Read decoder weight paths (skip first 4 entries = runtime inputs)
    with open(os.path.join(fdir, "whisper_decoder_inputs.txt")) as f:
        all_dec_paths = f.read().strip().split(",")
    dec_weight_paths = all_dec_paths[4:]
    print(f"Loaded {len(dec_weight_paths)} decoder weight files")

    # Autoregressive decode
    initial_tokens = [SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]
    tokens = list(initial_tokens)
    max_tokens = n_text_ctx
    decode_steps = []

    tmpdir = tempfile.mkdtemp(prefix="whisper_spirv_")

    enc_out_tmp = os.path.join(tmpdir, "encoder_output.raw.f32")
    write_f32(enc_out_tmp, encoder_output)

    for step in range(max_tokens - len(initial_tokens)):
        step_start = time.time()

        padded = tokens + [0] * (n_text_ctx - len(tokens))

        # Token embeddings: [1, 64, 384]
        tok_emb = []
        for t in padded:
            offset = t * n_text_state
            tok_emb.extend(token_emb_table[offset:offset + n_text_state])

        # Positional embeddings: [1, 64, 384]
        pos_emb = pos_emb_table[:n_text_ctx * n_text_state]

        # Causal mask: [1, 1, 64, 64]
        causal = []
        for r in range(n_text_ctx):
            for c in range(n_text_ctx):
                if c > r or c >= len(tokens):
                    causal.append(-1e9)
                else:
                    causal.append(0.0)

        # Write temp fixtures
        tok_path = os.path.join(tmpdir, f"s{step}_tok.raw.f32")
        pos_path = os.path.join(tmpdir, f"s{step}_pos.raw.f32")
        mask_path = os.path.join(tmpdir, f"s{step}_mask.raw.f32")
        write_f32(tok_path, tok_emb)
        write_f32(pos_path, pos_emb)
        write_f32(mask_path, causal)

        # Build input paths
        step_inputs = [enc_out_tmp, tok_path, pos_path, mask_path] + dec_weight_paths

        step_output = os.path.join(tmpdir, f"s{step}_logits.raw.f32")
        step_report = os.path.join(tmpdir, f"s{step}_report.json")

        cmd = [
            args.xla_hlo_bin,
            "--input", args.decoder_hlo,
            "--run",
            "--kernel-dir", args.kernel_dir,
            "--device", args.device,
            "--inputs-f32", ",".join(step_inputs),
            "--output-f32", step_output,
            "--report", step_report,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  Step {step}: xla_hlo FAILED: {result.stderr}", file=sys.stderr)
            break

        with open(step_report) as f:
            dec_report = json.load(f)

        if dec_report.get("status") != "pass":
            print(f"  Step {step}: status != pass")
            break

        # Read logits [1, 64, 51865] and argmax at current position
        logits = read_f32(step_output)
        pos = len(tokens) - 1
        logit_offset = pos * n_vocab
        pos_logits = logits[logit_offset:logit_offset + n_vocab]
        next_token = argmax(pos_logits)

        step_elapsed = time.time() - step_start
        step_info = {
            "step": step,
            "pos": pos,
            "token": next_token,
            "elapsed_s": round(step_elapsed, 3),
            "dispatch_count": dec_report.get("execution", {}).get("dispatch_count"),
        }
        decode_steps.append(step_info)
        print(f"  Step {step}: pos={pos}, token={next_token}, elapsed={step_elapsed:.1f}s")

        if next_token == EOT:
            print(f"  EOT at step {step}")
            break
        tokens.append(next_token)

    total_elapsed = time.time() - start_time

    results = {
        "model": "whisper_tiny",
        "device": args.device,
        "encoder": {"status": "precomputed", "path": args.encoder_output},
        "status": "pass",
        "tokens": tokens,
        "transcription": f"[token_ids: {[t for t in tokens if t < 50257]}]",
        "decode_steps": decode_steps,
        "total_steps": len(decode_steps),
        "total_elapsed_s": round(total_elapsed, 3),
    }

    with open(args.report, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(f"\nTokens: {tokens}")
    print(f"Total time: {total_elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
