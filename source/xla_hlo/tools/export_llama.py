#!/usr/bin/env python3
"""Export TinyLlama-1.1B as StableHLO text + weight fixtures.

TinyLlama-1.1B-Chat architecture:
  22 decoder layers (LlamaDecoderLayer)
  Hidden: 2048, Intermediate: 5632
  Attention: 32 query heads, 4 KV heads (GQA 8:1), head_dim=64
  RMSNorm (no bias), SiLU activation, RoPE position encoding
  Vocab: 32000, fixed sequence length S=64

All ops map to already-supported HLO ops:
  dot_general, reshape, transpose, reduce, broadcast,
  add, multiply, divide, rsqrt, logistic (sigmoid), negate,
  slice, concatenate, constant
"""
import argparse
import collections
import gc
import json
import os
import pathlib
import re
import sys
import traceback

import numpy as np


def write_f32(path, array, name=None):
    """Save array as raw float32 and return info dict."""
    data = np.asarray(array, dtype=np.float32)
    data.tofile(path)
    return {"path": str(path), "shape": list(data.shape), "name": name or path.stem}


def write_histogram(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def histogram_for_text(model, output, text, tf_version, input_specs, status,
                       total_params=0, error=None):
    stable_ops = [
        m.group(1) for m in re.finditer(r"stablehlo\.([A-Za-z0-9_]+)", text)
    ]
    hlo_ops = [
        m.group(1)
        for m in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_-]*)\(", text)
    ]
    return {
        "model": model,
        "tensorflow_version": tf_version,
        "input_specs": input_specs,
        "output": str(output),
        "bytes": output.stat().st_size if output.exists() else 0,
        "stablehlo_ops": dict(collections.Counter(stable_ops)),
        "hlo_like_tokens": dict(collections.Counter(hlo_ops)),
        "export_blocked": status != "pass",
        "status": status,
        "total_params": total_params,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="HLO text output path")
    parser.add_argument("--histogram", required=True, help="Histogram JSON output")
    parser.add_argument("--fixture-dir", default=None,
                        help="Directory for weight fixtures (default: same as output)")
    args = parser.parse_args()

    out = pathlib.Path(args.output)
    histogram = pathlib.Path(args.histogram)
    out.parent.mkdir(parents=True, exist_ok=True)

    fixture_dir = pathlib.Path(args.fixture_dir) if args.fixture_dir else out.parent
    fixture_dir.mkdir(parents=True, exist_ok=True)

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU for HLO extraction

        import tensorflow as tf
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tf_version = tf.__version__

        # ----- TinyLlama-1.1B dimensions -----
        S = 64              # fixed sequence length (same as Whisper decoder)
        hidden = 2048
        intermediate = 5632
        n_heads = 32
        n_kv_heads = 4
        head_dim = 64
        n_layers = 22
        vocab = 32000
        n_rep = n_heads // n_kv_heads   # 8
        rms_eps = 1e-5
        rope_theta = 10000.0

        # ----- Load model from HuggingFace -----
        print("Loading TinyLlama-1.1B-Chat from HuggingFace...")
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float32,
        )
        model.eval()
        state = model.state_dict()
        print(f"Model loaded. State dict keys: {len(state)}")

        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )

        # ----- Precompute RoPE cos/sin for S positions -----
        inv_freq = 1.0 / (
            rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
        )
        positions = np.arange(S, dtype=np.float32)
        freqs = np.outer(positions, inv_freq)        # [S, head_dim//2]
        emb = np.concatenate([freqs, freqs], axis=-1) # [S, head_dim]
        rope_cos = np.cos(emb).reshape(1, 1, S, head_dim).astype(np.float32)
        rope_sin = np.sin(emb).reshape(1, 1, S, head_dim).astype(np.float32)

        # ----- Save embedding table (Python lookup, NOT in HLO) -----
        embed_table = state["model.embed_tokens.weight"].numpy()  # [32000, 2048]
        write_f32(fixture_dir / "tinyllama_token_embedding.raw.f32",
                  embed_table, "token_embedding")
        print(f"Saved embedding table: {embed_table.shape}")

        # ----- Save RoPE cos/sin (for SPIR-V runner) -----
        write_f32(fixture_dir / "tinyllama_rope_cos.raw.f32", rope_cos, "rope_cos")
        write_f32(fixture_dir / "tinyllama_rope_sin.raw.f32", rope_sin, "rope_sin")
        print(f"Saved RoPE cos/sin: {rope_cos.shape}")

        # ----- Build input_specs (must match TF function arg order) -----
        input_specs = [
            {"shape": [1, S, hidden], "dtype": "float32", "name": "token_emb"},
            {"shape": [1, 1, S, head_dim], "dtype": "float32", "name": "rope_cos"},
            {"shape": [1, 1, S, head_dim], "dtype": "float32", "name": "rope_sin"},
            {"shape": [1, 1, S, S], "dtype": "float32", "name": "causal_mask"},
        ]
        for layer in range(n_layers):
            pfx = f"layer{layer:02d}"
            input_specs.extend([
                {"shape": [hidden], "name": f"{pfx}_input_ln_w"},
                {"shape": [hidden, n_heads * head_dim], "name": f"{pfx}_q_w"},
                {"shape": [hidden, n_kv_heads * head_dim], "name": f"{pfx}_k_w"},
                {"shape": [hidden, n_kv_heads * head_dim], "name": f"{pfx}_v_w"},
                {"shape": [n_heads * head_dim, hidden], "name": f"{pfx}_o_w"},
                {"shape": [hidden], "name": f"{pfx}_post_ln_w"},
                {"shape": [hidden, intermediate], "name": f"{pfx}_gate_w"},
                {"shape": [hidden, intermediate], "name": f"{pfx}_up_w"},
                {"shape": [intermediate, hidden], "name": f"{pfx}_down_w"},
            ])
        input_specs.extend([
            {"shape": [hidden], "name": "final_ln_w"},
            {"shape": [hidden, vocab], "name": "lm_head_w"},
        ])
        for s in input_specs:
            s.setdefault("dtype", "float32")

        print(f"Total HLO parameters: {len(input_specs)}")

        # ----- Save weight fixtures as .raw.f32 -----
        print("Saving weight fixtures...")
        fixtures = []
        param_idx = 0

        # Dynamic inputs: initial-step token_emb, rope_cos, rope_sin, causal_mask
        prompt = "The capital of France is"
        prompt_ids = tokenizer.encode(prompt)
        padded = prompt_ids + [0] * (S - len(prompt_ids))
        tok_emb_init = embed_table[padded][np.newaxis, :, :]  # [1, S, 2048]

        causal_init = np.zeros((1, 1, S, S), dtype=np.float32)
        for r in range(S):
            for c in range(S):
                if c > r or c >= len(prompt_ids):
                    causal_init[0, 0, r, c] = -1e9

        for name, data in [
            ("token_emb", tok_emb_init),
            ("rope_cos", rope_cos),
            ("rope_sin", rope_sin),
            ("causal_mask", causal_init),
        ]:
            info = write_f32(
                fixture_dir / f"tinyllama_param_{param_idx:03d}_{name}.raw.f32",
                data, name,
            )
            fixtures.append(info)
            param_idx += 1

        # Per-layer weights
        for layer in range(n_layers):
            pfx = f"layer{layer:02d}"
            layer_weights = [
                (f"{pfx}_input_ln_w",
                 state[f"model.layers.{layer}.input_layernorm.weight"].numpy()),
                (f"{pfx}_q_w",
                 state[f"model.layers.{layer}.self_attn.q_proj.weight"].numpy().T),
                (f"{pfx}_k_w",
                 state[f"model.layers.{layer}.self_attn.k_proj.weight"].numpy().T),
                (f"{pfx}_v_w",
                 state[f"model.layers.{layer}.self_attn.v_proj.weight"].numpy().T),
                (f"{pfx}_o_w",
                 state[f"model.layers.{layer}.self_attn.o_proj.weight"].numpy().T),
                (f"{pfx}_post_ln_w",
                 state[f"model.layers.{layer}.post_attention_layernorm.weight"].numpy()),
                (f"{pfx}_gate_w",
                 state[f"model.layers.{layer}.mlp.gate_proj.weight"].numpy().T),
                (f"{pfx}_up_w",
                 state[f"model.layers.{layer}.mlp.up_proj.weight"].numpy().T),
                (f"{pfx}_down_w",
                 state[f"model.layers.{layer}.mlp.down_proj.weight"].numpy().T),
            ]
            for wname, wdata in layer_weights:
                info = write_f32(
                    fixture_dir / f"tinyllama_param_{param_idx:03d}_{wname}.raw.f32",
                    wdata, wname,
                )
                fixtures.append(info)
                param_idx += 1
            print(f"  Layer {layer:2d} saved ({param_idx} files)")

        # Final norm + lm_head
        w = state["model.norm.weight"].numpy()
        info = write_f32(
            fixture_dir / f"tinyllama_param_{param_idx:03d}_final_ln_w.raw.f32",
            w, "final_ln_w",
        )
        fixtures.append(info)
        param_idx += 1

        # lm_head: [vocab, hidden] -> transposed to [hidden, vocab]
        w = state["lm_head.weight"].numpy().T
        info = write_f32(
            fixture_dir / f"tinyllama_param_{param_idx:03d}_lm_head_w.raw.f32",
            w, "lm_head_w",
        )
        fixtures.append(info)
        param_idx += 1

        print(f"Total fixture files saved: {param_idx}")

        # Write inputs file (relative paths for portability)
        inputs_path = fixture_dir / "tinyllama_inputs.txt"
        rel_paths = []
        for f in fixtures:
            p = pathlib.Path(f["path"])
            try:
                rel = p.relative_to(pathlib.Path.cwd())
            except ValueError:
                rel = p
            rel_paths.append(str(rel))
        inputs_path.write_text(",".join(rel_paths) + "\n")
        print(f"Inputs file: {inputs_path}")

        # ----- Free PyTorch memory -----
        del model, state
        gc.collect()

        # ----- Build TF function for HLO export -----
        print("Building TF function for HLO export...")
        print(f"  Function will take {len(input_specs)} arguments")

        @tf.function(jit_compile=True)
        def llama_step(*args):
            """TinyLlama-1.1B single forward pass: tokens -> logits."""
            idx = [0]

            def next_arg():
                v = args[idx[0]]
                idx[0] += 1
                return v

            token_emb = next_arg()     # [1, S, 2048]
            r_cos = next_arg()         # [1, 1, S, 64]
            r_sin = next_arg()         # [1, 1, S, 64]
            causal_mask = next_arg()   # [1, 1, S, S]

            def rms_norm(x, w, eps=1e-5):
                variance = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
                return x * tf.math.rsqrt(variance + eps) * w

            def rotate_half(x):
                # x: [batch, heads, seq, head_dim=64]
                x1 = x[:, :, :, :32]   # first half
                x2 = x[:, :, :, 32:]   # second half
                return tf.concat([-x2, x1], axis=-1)

            def apply_rope(x, cos, sin):
                # x: [1, heads, S, 64], cos/sin: [1, 1, S, 64]
                return x * cos + rotate_half(x) * sin

            def silu(x):
                return x * tf.math.sigmoid(x)

            x = token_emb

            for _layer in range(22):
                ln_w = next_arg()       # [2048]
                q_w = next_arg()        # [2048, 2048]
                k_w = next_arg()        # [2048, 256]
                v_w = next_arg()        # [2048, 256]
                o_w = next_arg()        # [2048, 2048]
                post_ln_w = next_arg()  # [2048]
                gate_w = next_arg()     # [2048, 5632]
                up_w = next_arg()       # [2048, 5632]
                down_w = next_arg()     # [5632, 2048]

                # --- RMSNorm + GQA Attention ---
                h = rms_norm(x, ln_w)

                q = tf.linalg.matmul(h, q_w)   # [1, S, 2048]
                k = tf.linalg.matmul(h, k_w)   # [1, S, 256]
                v = tf.linalg.matmul(h, v_w)   # [1, S, 256]

                # Reshape to heads: [1, S, heads*hd] -> [1, heads, S, hd]
                q = tf.transpose(
                    tf.reshape(q, [1, 64, 32, 64]), [0, 2, 1, 3])  # [1,32,S,64]
                k = tf.transpose(
                    tf.reshape(k, [1, 64, 4, 64]), [0, 2, 1, 3])   # [1,4,S,64]
                v = tf.transpose(
                    tf.reshape(v, [1, 64, 4, 64]), [0, 2, 1, 3])   # [1,4,S,64]

                # Apply RoPE to Q and K
                q = apply_rope(q, r_cos, r_sin)
                k = apply_rope(k, r_cos, r_sin)

                # Repeat KV heads 8x: [1,4,S,64] -> [1,32,S,64]
                k = tf.reshape(
                    tf.tile(
                        tf.reshape(k, [1, 4, 1, 64, 64]),
                        [1, 1, 8, 1, 1]),
                    [1, 32, 64, 64])
                v = tf.reshape(
                    tf.tile(
                        tf.reshape(v, [1, 4, 1, 64, 64]),
                        [1, 1, 8, 1, 1]),
                    [1, 32, 64, 64])

                # Scaled dot-product attention
                scale = tf.math.rsqrt(tf.constant(64.0, dtype=tf.float32))
                scores = tf.linalg.matmul(q, k, transpose_b=True) * scale
                scores = scores + causal_mask   # [1,32,S,S]
                probs = tf.nn.softmax(scores, axis=-1)
                att = tf.linalg.matmul(probs, v)  # [1,32,S,64]

                # Merge heads: [1,32,S,64] -> [1,S,2048]
                att = tf.reshape(
                    tf.transpose(att, [0, 2, 1, 3]),
                    [1, 64, 2048])
                att = tf.linalg.matmul(att, o_w)  # [1,S,2048]

                x = x + att  # residual

                # --- RMSNorm + SiLU MLP ---
                h = rms_norm(x, post_ln_w)
                gate = silu(tf.linalg.matmul(h, gate_w))   # [1,S,5632]
                up = tf.linalg.matmul(h, up_w)              # [1,S,5632]
                h = gate * up
                h = tf.linalg.matmul(h, down_w)             # [1,S,2048]

                x = x + h  # residual

            # --- Final norm + output projection ---
            final_ln_w = next_arg()   # [2048]
            lm_head_w = next_arg()    # [2048, 32000]

            x = rms_norm(x, final_ln_w)
            logits = tf.linalg.matmul(x, lm_head_w)  # [1, S, 32000]
            return logits

        # ----- Trace and export HLO -----
        print("Tracing TF function...")
        specs = [tf.TensorSpec(s["shape"], tf.float32) for s in input_specs]
        llama_step.get_concrete_function(*specs)

        print("Running with sample args...")
        sample_args = [tf.zeros(s["shape"], tf.float32) for s in input_specs]
        _ = llama_step(*sample_args)

        print("Extracting HLO IR...")
        hlo = llama_step.experimental_get_compiler_ir(*sample_args)(stage="hlo")
        text = hlo if isinstance(hlo, str) else str(hlo)
        out.write_text(text)
        print(f"HLO text written: {out} ({out.stat().st_size} bytes)")

        # ----- Histogram -----
        hist = histogram_for_text(
            "tinyllama_1.1b", out, text, tf_version, input_specs,
            "pass", total_params=param_idx,
        )
        write_histogram(histogram, hist)
        print(json.dumps(hist, indent=2, sort_keys=True))
        return 0

    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=20),
        }
        text = "EXPORT_BLOCKED: " + repr(exc) + "\n"
        out.write_text(text)
        hist = histogram_for_text(
            "tinyllama_1.1b", out, text,
            locals().get("tf_version", None), [], "blocked",
            error=error,
        )
        write_histogram(histogram, hist)
        print(json.dumps(hist, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
