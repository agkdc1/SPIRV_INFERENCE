#!/usr/bin/env python3
"""Run Whisper-tiny end-to-end through SPIR-V via xla_hlo binary.

Encoder: single pass through xla_hlo.
Decoder: autoregressive loop — Python orchestrator calls xla_hlo per step,
reads output logits, does argmax, updates token sequence.
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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_f32(path, array):
    data = np.asarray(array, dtype=np.float32)
    data.tofile(path)
    return str(path)


def read_f32(path, shape):
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(shape)


def run_xla_hlo(xla_hlo_bin, hlo_path, kernel_dir, device, inputs_f32_paths,
                output_path, report_path, expected_path=None, epsilon=0.01):
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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"xla_hlo failed: {result.stderr}", file=sys.stderr)
        return None

    with open(report_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-hlo", required=True)
    parser.add_argument("--decoder-hlo", required=True)
    parser.add_argument("--kernel-dir", required=True)
    parser.add_argument("--device", default="any")
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--xla-hlo-bin", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--encoder-output", default=None,
                        help="Pre-computed encoder output (skip encoder run)")
    parser.add_argument("--expected-encoder", default=None,
                        help="Expected encoder output for parity check")
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()

    fixture_dir = pathlib.Path(args.fixture_dir)
    report_path = pathlib.Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    n_text_ctx = 64
    n_text_state = 384
    n_vocab = 51865
    n_audio_ctx = 1500

    # Special tokens
    SOT = 50258
    EN = 50259
    TRANSCRIBE = 50360
    NO_TIMESTAMPS = 50364
    EOT = 50257

    start_time = time.time()
    results = {
        "model": "whisper_tiny",
        "device": args.device,
    }

    # --- Step 1: Run encoder ---
    if args.encoder_output and os.path.exists(args.encoder_output):
        print(f"Using pre-computed encoder output: {args.encoder_output}")
        encoder_output = read_f32(args.encoder_output, (1, n_audio_ctx, n_text_state))
        results["encoder"] = {"status": "precomputed", "path": args.encoder_output}
    else:
        print("Running encoder through SPIR-V...")
        enc_inputs_path = fixture_dir / "whisper_encoder_inputs.txt"
        enc_input_paths = enc_inputs_path.read_text().strip().split(",")

        enc_output_path = report_path.parent / "whisper_encoder_spirv_output.raw.f32"
        enc_report_path = report_path.parent / "whisper_encoder_spirv_subreport.json"

        enc_report = run_xla_hlo(
            args.xla_hlo_bin, args.encoder_hlo, args.kernel_dir, args.device,
            enc_input_paths, enc_output_path, enc_report_path,
            expected_path=args.expected_encoder, epsilon=args.epsilon
        )

        if enc_report is None or enc_report.get("status") != "pass":
            results["status"] = "fail"
            results["error"] = "encoder execution failed"
            report_path.write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(results, indent=2))
            return 1

        encoder_output = read_f32(enc_output_path, (1, n_audio_ctx, n_text_state))
        results["encoder"] = {
            "status": "pass",
            "dispatch_count": enc_report.get("execution", {}).get("dispatch_count"),
            "parity": enc_report.get("execution", {}).get("parity"),
            "device": enc_report.get("execution", {}).get("device"),
        }
        print(f"Encoder done. Output shape: {encoder_output.shape}")

    # --- Step 2: Load decoder weights and embedding tables ---
    print("Loading decoder weights and embedding tables...")

    token_emb_table = read_f32(fixture_dir / "whisper_token_embedding_table.raw.f32",
                                (n_vocab, n_text_state))
    pos_emb_table = read_f32(fixture_dir / "whisper_pos_embedding_table.raw.f32",
                              (448, n_text_state))

    # Read decoder weight paths (skip first 4 entries which are inputs we build per-step)
    dec_inputs_path = fixture_dir / "whisper_decoder_inputs.txt"
    all_dec_paths = dec_inputs_path.read_text().strip().split(",")
    # Paths 0=encoder_output, 1=token_emb, 2=pos_emb, 3=causal_mask, 4+=weights
    dec_weight_paths = all_dec_paths[4:]

    print(f"Loaded {len(dec_weight_paths)} decoder weight files")

    # --- Step 3: Autoregressive decode loop ---
    print("Starting autoregressive decode...")
    initial_tokens = [SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]
    tokens = list(initial_tokens)
    max_tokens = n_text_ctx
    decode_steps = []

    with tempfile.TemporaryDirectory(prefix="whisper_spirv_") as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        # Write encoder output to temp for decoder input
        enc_out_tmp = tmpdir / "encoder_output.raw.f32"
        write_f32(enc_out_tmp, encoder_output)

        for step in range(max_tokens - len(initial_tokens)):
            step_start = time.time()

            # Pad tokens to 64
            padded = tokens + [0] * (n_text_ctx - len(tokens))

            # Look up embeddings
            tok_emb = token_emb_table[padded][np.newaxis, :, :]  # [1, 64, 384]
            pos_emb = pos_emb_table[:n_text_ctx][np.newaxis, :, :]  # [1, 64, 384]

            # Causal mask: 0 for allowed, -1e9 for masked
            causal = np.zeros((1, 1, n_text_ctx, n_text_ctx), dtype=np.float32)
            for r in range(n_text_ctx):
                for c in range(n_text_ctx):
                    if c > r or c >= len(tokens):
                        causal[0, 0, r, c] = -1e9

            # Write temp fixtures
            tok_emb_path = tmpdir / f"step{step}_token_emb.raw.f32"
            pos_emb_path = tmpdir / f"step{step}_pos_emb.raw.f32"
            mask_path = tmpdir / f"step{step}_causal_mask.raw.f32"
            write_f32(tok_emb_path, tok_emb)
            write_f32(pos_emb_path, pos_emb)
            write_f32(mask_path, causal)

            # Build input paths: encoder_output, token_emb, pos_emb, causal_mask, then weights
            step_inputs = [str(enc_out_tmp), str(tok_emb_path), str(pos_emb_path),
                          str(mask_path)] + dec_weight_paths

            # Run decoder
            step_output = tmpdir / f"step{step}_logits.raw.f32"
            step_report = tmpdir / f"step{step}_report.json"

            dec_report = run_xla_hlo(
                args.xla_hlo_bin, args.decoder_hlo, args.kernel_dir, args.device,
                step_inputs, step_output, step_report
            )

            if dec_report is None or dec_report.get("status") != "pass":
                print(f"  Step {step}: FAILED")
                results["status"] = "fail"
                results["error"] = f"decoder step {step} failed"
                results["decode_steps"] = decode_steps
                report_path.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(results, indent=2))
                return 1

            # Read logits [1, 64, 51865]
            logits = read_f32(step_output, (1, n_text_ctx, n_vocab))

            # Argmax at current position
            pos = len(tokens) - 1
            next_token = int(np.argmax(logits[0, pos, :]))
            step_elapsed = time.time() - step_start

            step_info = {
                "step": step,
                "pos": pos,
                "token": next_token,
                "elapsed_s": round(step_elapsed, 3),
                "dispatch_count": dec_report.get("execution", {}).get("dispatch_count"),
            }
            decode_steps.append(step_info)
            print(f"  Step {step}: pos={pos}, token={next_token}, elapsed={step_elapsed:.3f}s")

            if next_token == EOT:
                print(f"  EOT at step {step}")
                break
            tokens.append(next_token)

    # --- Step 4: Decode tokens to text ---
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        text_tokens = [t for t in tokens if t < 50257]
        transcription = enc.decode(text_tokens)
    except ImportError:
        # Fallback: just report token IDs
        transcription = f"[tokens: {tokens}]"
        text_tokens = tokens

    total_elapsed = time.time() - start_time

    results.update({
        "status": "pass",
        "tokens": tokens,
        "transcription": transcription,
        "decode_steps": decode_steps,
        "total_steps": len(decode_steps),
        "total_elapsed_s": round(total_elapsed, 3),
    })

    report_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nTranscription: '{transcription}'")
    print(f"Tokens: {tokens}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
