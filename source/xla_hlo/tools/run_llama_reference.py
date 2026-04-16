#!/usr/bin/env python3
"""Run TinyLlama-1.1B reference inference on CPU via HuggingFace transformers.

Produces reference token IDs, text, and per-step logit hashes for
comparison against the SPIR-V path.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np


def sha256_array(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="The capital of France is",
                        help="Input prompt for text generation")
    parser.add_argument("--max-tokens", type=int, default=20,
                        help="Max new tokens to generate")
    parser.add_argument("--output", required=True,
                        help="Output JSON report path")
    parser.add_argument("--logits-dir", default=None,
                        help="Directory to save per-step logits (optional)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.logits_dir:
        logits_dir = pathlib.Path(args.logits_dir)
        logits_dir.mkdir(parents=True, exist_ok=True)
    else:
        logits_dir = out.parent

    print(f"Loading TinyLlama-1.1B-Chat...")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float32,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    prompt_ids = tokenizer.encode(args.prompt)
    print(f"Prompt: '{args.prompt}'")
    print(f"Prompt token IDs: {prompt_ids}")

    # Autoregressive generation with per-step logit capture
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    generated_tokens = []
    step_details = []
    total_start = time.time()

    with torch.no_grad():
        for step in range(args.max_tokens):
            step_start = time.time()

            outputs = model(input_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]

            # Take logits at last position
            last_logits = logits[0, -1, :].numpy()  # [32000]
            next_token = int(np.argmax(last_logits))
            step_elapsed = time.time() - step_start

            # Save per-step logits
            logits_path = logits_dir / f"tinyllama_tf_logits_step_{step:03d}.raw.f32"
            last_logits.astype(np.float32).tofile(logits_path)

            step_info = {
                "step": step,
                "token_id": next_token,
                "token_text": tokenizer.decode([next_token]),
                "logits_hash": sha256_array(last_logits),
                "logits_path": str(logits_path),
                "logits_top5": [
                    {"id": int(i), "val": float(last_logits[i])}
                    for i in np.argsort(last_logits)[-5:][::-1]
                ],
                "elapsed_s": round(step_elapsed, 4),
            }
            step_details.append(step_info)
            generated_tokens.append(next_token)
            print(f"  Step {step}: token={next_token} "
                  f"('{tokenizer.decode([next_token])}') "
                  f"elapsed={step_elapsed:.3f}s")

            # Check for EOS
            if next_token == tokenizer.eos_token_id:
                print(f"  EOS at step {step}")
                break

            # Append token and continue
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]], dtype=torch.long),
            ], dim=1)

    total_elapsed = time.time() - total_start
    all_ids = prompt_ids + generated_tokens
    full_text = tokenizer.decode(all_ids, skip_special_tokens=True)

    report = {
        "status": "pass",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "torch_version": torch.__version__,
        "prompt": args.prompt,
        "prompt_ids": prompt_ids,
        "generated_tokens": generated_tokens,
        "all_token_ids": all_ids,
        "full_text": full_text,
        "generated_text": tokenizer.decode(generated_tokens,
                                           skip_special_tokens=True),
        "steps": step_details,
        "total_steps": len(step_details),
        "total_elapsed_s": round(total_elapsed, 3),
    }

    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nFull text: '{full_text}'")
    print(f"Generated: '{report['generated_text']}'")
    print(f"Token IDs: {all_ids}")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
