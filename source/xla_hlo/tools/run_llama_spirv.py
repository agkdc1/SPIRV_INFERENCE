#!/usr/bin/env python3
"""Run TinyLlama-1.1B end-to-end through SPIR-V via xla_hlo binary.

Autoregressive loop: Python orchestrator calls xla_hlo per step,
reads output logits, does argmax, updates token sequence.

Same pattern as run_whisper_spirv.py but for decoder-only LLM.
"""
import argparse
import hashlib
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np


# ----- TinyLlama-1.1B constants -----
S = 64           # fixed sequence length
HIDDEN = 2048
N_HEADS = 32
N_KV_HEADS = 4
HEAD_DIM = 64
VOCAB = 32000
ROPE_THETA = 10000.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def write_f32(path, array):
    data = np.asarray(array, dtype=np.float32)
    data.tofile(path)
    return str(path)


def read_f32(path, shape=None):
    data = np.fromfile(path, dtype=np.float32)
    if shape is not None:
        data = data.reshape(shape)
    return data


def run_xla_hlo(xla_hlo_bin, hlo_path, kernel_dir, device, inputs_f32_paths,
                output_path, report_path, expected_path=None, epsilon=0.01,
                timeout=600):
    """Run xla_hlo binary and return the report JSON."""
    cmd = [
        xla_hlo_bin,
        "--input", str(hlo_path),
        "--run",
        "--kernel-dir", str(kernel_dir),
        "--device", device,
        "--inputs-f32", ",".join(str(p) for p in inputs_f32_paths),
        "--output-f32", str(output_path),
        "--report", str(report_path),
    ]
    if expected_path:
        cmd.extend(["--expected-f32", str(expected_path)])
        cmd.extend(["--epsilon", str(epsilon)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"xla_hlo failed (rc={result.returncode}):", file=sys.stderr)
        print(f"  stderr: {result.stderr[:2000]}", file=sys.stderr)
        print(f"  stdout: {result.stdout[:2000]}", file=sys.stderr)
        return None

    with open(report_path) as f:
        return json.load(f)


def precompute_rope(seq_len):
    """Precompute RoPE cos/sin for positions 0..seq_len-1."""
    inv_freq = 1.0 / (
        ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM)
    )
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)         # [S, HEAD_DIM//2]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [S, HEAD_DIM]
    cos = np.cos(emb).reshape(1, 1, seq_len, HEAD_DIM).astype(np.float32)
    sin = np.sin(emb).reshape(1, 1, seq_len, HEAD_DIM).astype(np.float32)
    return cos, sin


def build_causal_mask(seq_len, num_valid_tokens):
    """Build causal mask: 0 for allowed, -1e9 for masked."""
    mask = np.zeros((1, 1, seq_len, seq_len), dtype=np.float32)
    for r in range(seq_len):
        for c in range(seq_len):
            if c > r or c >= num_valid_tokens:
                mask[0, 0, r, c] = -1e9
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hlo", required=True,
                        help="Path to tinyllama.stablehlo.txt")
    parser.add_argument("--kernel-dir", required=True,
                        help="Path to SPIR-V kernel directory")
    parser.add_argument("--device", default="any",
                        help="Vulkan device: any, nvidia, intel")
    parser.add_argument("--fixture-dir", required=True,
                        help="Directory with weight .raw.f32 files")
    parser.add_argument("--xla-hlo-bin", required=True,
                        help="Path to xla_hlo binary")
    parser.add_argument("--prompt", default="The capital of France is",
                        help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Maximum tokens to generate")
    parser.add_argument("--report", required=True,
                        help="Output report JSON path")
    parser.add_argument("--reference", default=None,
                        help="Reference report JSON for comparison")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Parity epsilon threshold")
    args = parser.parse_args()

    fixture_dir = pathlib.Path(args.fixture_dir)
    report_path = pathlib.Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = {
        "model": "tinyllama_1.1b",
        "device": args.device,
        "prompt": args.prompt,
    }

    # ----- Load tokenizer -----
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        prompt_ids = tokenizer.encode(args.prompt)
        print(f"Prompt: '{args.prompt}' -> {prompt_ids}")
    except ImportError:
        print("WARNING: transformers not available, using hardcoded tokenization")
        # Fallback: hardcoded for "The capital of France is"
        prompt_ids = [1, 450, 7483, 310, 3444, 338]

    results["prompt_ids"] = prompt_ids

    # ----- Load embedding table -----
    print("Loading embedding table...")
    embed_table = read_f32(
        fixture_dir / "tinyllama_token_embedding.raw.f32",
        (VOCAB, HIDDEN),
    )

    # ----- Load weight paths (skip first 4 dynamic inputs) -----
    inputs_path = fixture_dir / "tinyllama_inputs.txt"
    all_paths = inputs_path.read_text().strip().split(",")
    # Paths 0=token_emb, 1=rope_cos, 2=rope_sin, 3=causal_mask, 4+=weights
    weight_paths = all_paths[4:]
    print(f"Loaded {len(weight_paths)} weight file paths")

    # ----- Precompute RoPE -----
    rope_cos, rope_sin = precompute_rope(S)

    # ----- Autoregressive decode -----
    print("Starting autoregressive generation...")
    tokens = list(prompt_ids)
    decode_steps = []

    # Load reference if provided
    ref_tokens = None
    if args.reference:
        ref_path = pathlib.Path(args.reference)
        if ref_path.exists():
            ref_data = json.loads(ref_path.read_text())
            ref_tokens = ref_data.get("all_token_ids")
            print(f"Loaded reference: {len(ref_data.get('generated_tokens', []))} tokens")

    with tempfile.TemporaryDirectory(prefix="llama_spirv_") as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        for step in range(args.max_tokens):
            step_start = time.time()

            # Pad tokens to S=64
            padded = tokens + [0] * (S - len(tokens))
            if len(padded) > S:
                padded = padded[:S]

            # Look up embeddings: [1, S, 2048]
            tok_emb = embed_table[padded][np.newaxis, :, :]

            # Causal mask: [1, 1, S, S]
            causal = build_causal_mask(S, len(tokens))

            # Write dynamic inputs to temp
            tok_emb_path = tmpdir / f"step{step}_token_emb.raw.f32"
            cos_path = tmpdir / f"step{step}_rope_cos.raw.f32"
            sin_path = tmpdir / f"step{step}_rope_sin.raw.f32"
            mask_path = tmpdir / f"step{step}_causal_mask.raw.f32"

            write_f32(tok_emb_path, tok_emb)
            write_f32(cos_path, rope_cos)
            write_f32(sin_path, rope_sin)
            write_f32(mask_path, causal)

            # Build full input paths: 4 dynamic + N weights
            step_inputs = [
                str(tok_emb_path),
                str(cos_path),
                str(sin_path),
                str(mask_path),
            ] + weight_paths

            # Run xla_hlo
            step_output = tmpdir / f"step{step}_logits.raw.f32"
            step_report = tmpdir / f"step{step}_report.json"

            dec_report = run_xla_hlo(
                args.xla_hlo_bin, args.hlo, args.kernel_dir, args.device,
                step_inputs, step_output, step_report,
                timeout=600,
            )

            if dec_report is None or dec_report.get("status") != "pass":
                err_msg = "unknown"
                if dec_report:
                    err_msg = dec_report.get("error", str(dec_report))
                print(f"  Step {step}: FAILED — {err_msg}")
                results["status"] = "fail"
                results["error"] = f"step {step} failed: {err_msg}"
                results["decode_steps"] = decode_steps
                report_path.write_text(json.dumps(results, indent=2) + "\n")
                return 1

            # Read output logits: [1, S, 32000]
            logits = read_f32(step_output, (1, S, VOCAB))

            # Argmax at current position (last valid token position)
            pos = len(tokens) - 1
            step_logits = logits[0, pos, :]
            next_token = int(np.argmax(step_logits))
            step_elapsed = time.time() - step_start

            # Decode token text
            try:
                token_text = tokenizer.decode([next_token])
            except Exception:
                token_text = f"<{next_token}>"

            step_info = {
                "step": step,
                "pos": pos,
                "token_id": next_token,
                "token_text": token_text,
                "logits_hash": sha256_array(step_logits),
                "logits_top5": [
                    {"id": int(i), "val": float(step_logits[i])}
                    for i in np.argsort(step_logits)[-5:][::-1]
                ],
                "elapsed_s": round(step_elapsed, 3),
                "dispatch_count": dec_report.get("execution", {}).get(
                    "dispatch_count"),
            }

            # Compare with reference if available
            if ref_tokens and len(tokens) < len(ref_tokens):
                expected = ref_tokens[len(tokens)]
                step_info["ref_token_id"] = expected
                step_info["ref_match"] = (next_token == expected)

            decode_steps.append(step_info)
            ref_str = ""
            if "ref_match" in step_info:
                ref_str = f" ref={'MATCH' if step_info['ref_match'] else 'MISMATCH'}"
            print(f"  Step {step}: pos={pos}, token={next_token} "
                  f"('{token_text}') elapsed={step_elapsed:.1f}s{ref_str}")

            # Check for EOS
            if next_token == 2 or next_token == 0:  # EOS or PAD
                print(f"  EOS/PAD at step {step}")
                break
            tokens.append(next_token)

    # ----- Final results -----
    total_elapsed = time.time() - start_time
    generated = tokens[len(prompt_ids):]

    try:
        full_text = tokenizer.decode(tokens, skip_special_tokens=True)
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    except Exception:
        full_text = f"[tokens: {tokens}]"
        generated_text = f"[tokens: {generated}]"

    # Count reference matches
    ref_matches = sum(1 for s in decode_steps if s.get("ref_match"))
    ref_total = sum(1 for s in decode_steps if "ref_match" in s)

    results.update({
        "status": "pass",
        "all_token_ids": tokens,
        "generated_tokens": generated,
        "full_text": full_text,
        "generated_text": generated_text,
        "decode_steps": decode_steps,
        "total_steps": len(decode_steps),
        "total_elapsed_s": round(total_elapsed, 3),
        "avg_step_s": round(total_elapsed / max(len(decode_steps), 1), 3),
    })
    if ref_total > 0:
        results["reference_match"] = {
            "matches": ref_matches,
            "total": ref_total,
            "rate": round(ref_matches / ref_total, 4),
        }

    report_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nFull text: '{full_text}'")
    print(f"Generated: '{generated_text}'")
    print(f"Tokens: {tokens}")
    if ref_total > 0:
        print(f"Reference parity: {ref_matches}/{ref_total} "
              f"({100*ref_matches/ref_total:.1f}%)")
    print(f"Total time: {total_elapsed:.1f}s "
          f"({total_elapsed/max(len(decode_steps),1):.1f}s/token)")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
