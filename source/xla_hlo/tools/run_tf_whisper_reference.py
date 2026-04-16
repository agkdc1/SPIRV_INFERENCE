#!/usr/bin/env python3
"""Generate TF reference output for Whisper-tiny model with real weights.

Extracts weights from openai-whisper (PyTorch), transposes conv weights
from PyTorch [C_out, C_in, K] to TF [1, K, C_in, C_out] for Conv2D path,
runs encoder and decoder through TF, saves fixtures and reference outputs.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

import numpy as np
import tensorflow as tf


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_f32(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array, dtype=np.float32)
    data.tofile(path)
    return {
        "path": str(path),
        "shape": list(data.shape),
        "elements": int(data.size),
        "sha256": sha256_file(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--audio", default=None, help="Path to audio file (generates sine wave if omitted)")
    args = parser.parse_args()

    fixture_dir = pathlib.Path(args.fixture_dir)
    out_dir = pathlib.Path(args.out_dir)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import whisper

        # Whisper-tiny dimensions
        n_mels = 80
        n_audio_ctx = 1500
        n_audio_state = 384
        n_audio_head = 6
        n_audio_layer = 4
        n_text_ctx = 64
        n_text_state = 384
        n_text_head = 6
        n_text_layer = 4
        n_vocab = 51865
        head_dim = n_audio_state // n_audio_head  # 64

        # Load Whisper-tiny model
        print("Loading Whisper-tiny model...")
        model = whisper.load_model("tiny", device="cpu")
        sd = model.state_dict()

        print(f"Model loaded. State dict keys: {len(sd)}")

        # --- Generate or load audio ---
        if args.audio and os.path.exists(args.audio):
            print(f"Loading audio from {args.audio}")
            mel_pt = whisper.audio.log_mel_spectrogram(
                whisper.audio.load_audio(args.audio),
                n_mels=n_mels,
            )
        else:
            # Generate a 5-second sine wave test audio (440Hz)
            print("Generating 5-second 440Hz sine wave test audio...")
            sr = 16000
            duration = 5.0
            t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
            # Mix of frequencies for richer spectrogram
            audio = (0.5 * np.sin(2 * np.pi * 440 * t) +
                     0.3 * np.sin(2 * np.pi * 880 * t) +
                     0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

            # Save as raw audio for reproducibility
            audio_path = fixture_dir / "whisper_test_audio.raw.f32"
            audio.tofile(audio_path)

            mel_pt = whisper.audio.log_mel_spectrogram(
                torch.from_numpy(audio),
                n_mels=n_mels,
            )

        # Pad/trim mel to 3000 frames (30 seconds standard Whisper window)
        mel_pt = whisper.audio.pad_or_trim(mel_pt, 3000)
        # mel_pt shape: [80, 3000] (PyTorch: channels-first)
        mel_np = mel_pt.numpy().T  # Transpose to [3000, 80] for TF channels-last
        mel_input = mel_np[np.newaxis, :, :]  # [1, 3000, 80]
        print(f"Mel spectrogram shape: {mel_input.shape}")

        # --- Extract encoder weights ---
        enc_params = {}
        enc_param_order = ["mel"]

        # Conv1: PyTorch [384, 80, 3] -> TF [1, 3, 80, 384]
        conv1_w_pt = sd["encoder.conv1.weight"].numpy()  # [384, 80, 3]
        conv1_w_tf = conv1_w_pt.transpose(2, 1, 0)[np.newaxis, :, :, :]  # [1, 3, 80, 384]
        conv1_b = sd["encoder.conv1.bias"].numpy()  # [384]
        enc_params["conv1_w"] = conv1_w_tf
        enc_params["conv1_b"] = conv1_b
        enc_param_order.extend(["conv1_w", "conv1_b"])

        # Conv2: PyTorch [384, 384, 3] -> TF [1, 3, 384, 384]
        conv2_w_pt = sd["encoder.conv2.weight"].numpy()  # [384, 384, 3]
        conv2_w_tf = conv2_w_pt.transpose(2, 1, 0)[np.newaxis, :, :, :]  # [1, 3, 384, 384]
        conv2_b = sd["encoder.conv2.bias"].numpy()  # [384]
        enc_params["conv2_w"] = conv2_w_tf
        enc_params["conv2_b"] = conv2_b
        enc_param_order.extend(["conv2_w", "conv2_b"])

        # Positional embedding: [1500, 384]
        pos_emb = sd["encoder.positional_embedding"].numpy()  # [1500, 384]
        enc_params["pos_emb"] = pos_emb
        enc_param_order.append("pos_emb")

        # Encoder blocks
        for i in range(n_audio_layer):
            prefix = f"enc{i}"
            pt_prefix = f"encoder.blocks.{i}"

            # Attention layer norm
            enc_params[f"{prefix}_ln1_gamma"] = sd[f"{pt_prefix}.attn_ln.weight"].numpy()
            enc_params[f"{prefix}_ln1_beta"] = sd[f"{pt_prefix}.attn_ln.bias"].numpy()
            enc_param_order.extend([f"{prefix}_ln1_gamma", f"{prefix}_ln1_beta"])

            # Self-attention Q, K, V, Out — PyTorch linear is [out, in], TF needs [in, out]
            enc_params[f"{prefix}_q_w"] = sd[f"{pt_prefix}.attn.query.weight"].numpy().T  # [384, 384]
            enc_params[f"{prefix}_q_b"] = sd[f"{pt_prefix}.attn.query.bias"].numpy()
            enc_params[f"{prefix}_k_w"] = sd[f"{pt_prefix}.attn.key.weight"].numpy().T
            # Whisper key has no bias in the original model, but our TF graph expects one
            if f"{pt_prefix}.attn.key.bias" in sd:
                enc_params[f"{prefix}_k_b"] = sd[f"{pt_prefix}.attn.key.bias"].numpy()
            else:
                enc_params[f"{prefix}_k_b"] = np.zeros(n_audio_state, dtype=np.float32)
            enc_params[f"{prefix}_v_w"] = sd[f"{pt_prefix}.attn.value.weight"].numpy().T
            enc_params[f"{prefix}_v_b"] = sd[f"{pt_prefix}.attn.value.bias"].numpy()
            enc_params[f"{prefix}_out_w"] = sd[f"{pt_prefix}.attn.out.weight"].numpy().T
            enc_params[f"{prefix}_out_b"] = sd[f"{pt_prefix}.attn.out.bias"].numpy()
            enc_param_order.extend([
                f"{prefix}_q_w", f"{prefix}_q_b",
                f"{prefix}_k_w", f"{prefix}_k_b",
                f"{prefix}_v_w", f"{prefix}_v_b",
                f"{prefix}_out_w", f"{prefix}_out_b",
            ])

            # MLP layer norm
            enc_params[f"{prefix}_ln2_gamma"] = sd[f"{pt_prefix}.mlp_ln.weight"].numpy()
            enc_params[f"{prefix}_ln2_beta"] = sd[f"{pt_prefix}.mlp_ln.bias"].numpy()
            enc_param_order.extend([f"{prefix}_ln2_gamma", f"{prefix}_ln2_beta"])

            # MLP fc1, fc2 — transpose
            enc_params[f"{prefix}_fc1_w"] = sd[f"{pt_prefix}.mlp.0.weight"].numpy().T  # [384, 1536]
            enc_params[f"{prefix}_fc1_b"] = sd[f"{pt_prefix}.mlp.0.bias"].numpy()  # [1536]
            enc_params[f"{prefix}_fc2_w"] = sd[f"{pt_prefix}.mlp.2.weight"].numpy().T  # [1536, 384]
            enc_params[f"{prefix}_fc2_b"] = sd[f"{pt_prefix}.mlp.2.bias"].numpy()  # [384]
            enc_param_order.extend([
                f"{prefix}_fc1_w", f"{prefix}_fc1_b",
                f"{prefix}_fc2_w", f"{prefix}_fc2_b",
            ])

        # Final layer norm
        enc_params["ln_f_gamma"] = sd["encoder.ln_post.weight"].numpy()
        enc_params["ln_f_beta"] = sd["encoder.ln_post.bias"].numpy()
        enc_param_order.extend(["ln_f_gamma", "ln_f_beta"])

        print(f"Encoder: {len(enc_param_order)} params (including mel input)")

        # --- Save encoder fixtures ---
        enc_fixtures = []
        # First: mel input
        info = write_f32(fixture_dir / "whisper_enc_param_000_mel.raw.f32", mel_input)
        enc_fixtures.append(info)
        # Then: weights
        for idx, name in enumerate(enc_param_order[1:], 1):  # skip 'mel'
            info = write_f32(fixture_dir / f"whisper_enc_param_{idx:03d}_{name}.raw.f32", enc_params[name])
            enc_fixtures.append(info)

        # Save encoder inputs list
        enc_inputs_path = fixture_dir / "whisper_encoder_inputs.txt"
        enc_inputs_path.write_text(",".join(item["path"] for item in enc_fixtures) + "\n")

        # --- Run TF encoder reference ---
        print("Running TF encoder reference...")

        def layer_norm_np(x, gamma, beta, eps=1e-5):
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
            return (x - mean) / np.sqrt(var + eps) * gamma + beta

        def gelu_np(x):
            c = 0.7978845608  # sqrt(2/pi)
            return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))

        def self_attn_np(x, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, n_head=6):
            B, S, C = x.shape
            hd = C // n_head
            q = x @ q_w + q_b  # [B, S, C]
            k = x @ k_w + k_b
            v = x @ v_w + v_b
            q = q.reshape(B, S, n_head, hd).transpose(0, 2, 1, 3)  # [B, nh, S, hd]
            k = k.reshape(B, S, n_head, hd).transpose(0, 2, 1, 3)
            v = v.reshape(B, S, n_head, hd).transpose(0, 2, 1, 3)
            scale = 1.0 / np.sqrt(hd)
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale  # [B, nh, S, S]
            # Softmax
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            att = probs @ v  # [B, nh, S, hd]
            att = att.transpose(0, 2, 1, 3).reshape(B, S, C)  # [B, S, C]
            return att @ out_w + out_b

        # Run encoder in numpy (matches the TF graph exactly)
        x = mel_input.copy()  # [1, 3000, 80]

        # Conv1: expand to [1, 1, 3000, 80], conv2d with [1, 3, 80, 384], SAME padding
        # Use TF for conv2d since numpy conv is complex
        x_4d = tf.constant(x.reshape(1, 1, 3000, n_mels), dtype=tf.float32)
        x_4d = tf.nn.conv2d(x_4d, tf.constant(enc_params["conv1_w"], dtype=tf.float32),
                            strides=[1, 1, 1, 1], padding="SAME").numpy() + enc_params["conv1_b"]
        x_4d = gelu_np(x_4d)

        # Conv2: stride 2 along time
        x_4d = tf.nn.conv2d(tf.constant(x_4d, dtype=tf.float32),
                            tf.constant(enc_params["conv2_w"], dtype=tf.float32),
                            strides=[1, 1, 2, 1], padding="SAME").numpy() + enc_params["conv2_b"]
        x_4d = gelu_np(x_4d)

        # Squeeze: [1, 1500, 384]
        x = x_4d.reshape(1, n_audio_ctx, n_audio_state)
        x = x + enc_params["pos_emb"]

        # Encoder blocks
        for i in range(n_audio_layer):
            prefix = f"enc{i}"
            # Self-attention
            nx = layer_norm_np(x, enc_params[f"{prefix}_ln1_gamma"], enc_params[f"{prefix}_ln1_beta"])
            att = self_attn_np(nx,
                               enc_params[f"{prefix}_q_w"], enc_params[f"{prefix}_q_b"],
                               enc_params[f"{prefix}_k_w"], enc_params[f"{prefix}_k_b"],
                               enc_params[f"{prefix}_v_w"], enc_params[f"{prefix}_v_b"],
                               enc_params[f"{prefix}_out_w"], enc_params[f"{prefix}_out_b"])
            x = x + att
            # MLP
            nx = layer_norm_np(x, enc_params[f"{prefix}_ln2_gamma"], enc_params[f"{prefix}_ln2_beta"])
            h = nx @ enc_params[f"{prefix}_fc1_w"] + enc_params[f"{prefix}_fc1_b"]
            h = gelu_np(h)
            h = h @ enc_params[f"{prefix}_fc2_w"] + enc_params[f"{prefix}_fc2_b"]
            x = x + h

        # Final LN
        encoder_output = layer_norm_np(x, enc_params["ln_f_gamma"], enc_params["ln_f_beta"])
        print(f"Encoder output shape: {encoder_output.shape}, range: [{encoder_output.min():.4f}, {encoder_output.max():.4f}]")

        # Save encoder output
        enc_output_info = write_f32(out_dir / "whisper_encoder_tf_output.raw.f32", encoder_output)

        # --- Extract decoder weights ---
        dec_params = {}
        dec_param_order = ["encoder_output", "token_emb", "pos_emb_dec", "causal_mask"]

        for i in range(n_text_layer):
            prefix = f"dec{i}"
            pt_prefix = f"decoder.blocks.{i}"

            # Masked self-attention LN
            dec_params[f"{prefix}_ln1_gamma"] = sd[f"{pt_prefix}.attn_ln.weight"].numpy()
            dec_params[f"{prefix}_ln1_beta"] = sd[f"{pt_prefix}.attn_ln.bias"].numpy()

            # Self-attention Q, K, V, Out
            dec_params[f"{prefix}_sq_w"] = sd[f"{pt_prefix}.attn.query.weight"].numpy().T
            dec_params[f"{prefix}_sq_b"] = sd[f"{pt_prefix}.attn.query.bias"].numpy()
            dec_params[f"{prefix}_sk_w"] = sd[f"{pt_prefix}.attn.key.weight"].numpy().T
            if f"{pt_prefix}.attn.key.bias" in sd:
                dec_params[f"{prefix}_sk_b"] = sd[f"{pt_prefix}.attn.key.bias"].numpy()
            else:
                dec_params[f"{prefix}_sk_b"] = np.zeros(n_text_state, dtype=np.float32)
            dec_params[f"{prefix}_sv_w"] = sd[f"{pt_prefix}.attn.value.weight"].numpy().T
            dec_params[f"{prefix}_sv_b"] = sd[f"{pt_prefix}.attn.value.bias"].numpy()
            dec_params[f"{prefix}_sout_w"] = sd[f"{pt_prefix}.attn.out.weight"].numpy().T
            dec_params[f"{prefix}_sout_b"] = sd[f"{pt_prefix}.attn.out.bias"].numpy()

            # Cross-attention LN
            dec_params[f"{prefix}_ln2_gamma"] = sd[f"{pt_prefix}.cross_attn_ln.weight"].numpy()
            dec_params[f"{prefix}_ln2_beta"] = sd[f"{pt_prefix}.cross_attn_ln.bias"].numpy()

            # Cross-attention Q, K, V, Out
            dec_params[f"{prefix}_cq_w"] = sd[f"{pt_prefix}.cross_attn.query.weight"].numpy().T
            dec_params[f"{prefix}_cq_b"] = sd[f"{pt_prefix}.cross_attn.query.bias"].numpy()
            dec_params[f"{prefix}_ck_w"] = sd[f"{pt_prefix}.cross_attn.key.weight"].numpy().T
            if f"{pt_prefix}.cross_attn.key.bias" in sd:
                dec_params[f"{prefix}_ck_b"] = sd[f"{pt_prefix}.cross_attn.key.bias"].numpy()
            else:
                dec_params[f"{prefix}_ck_b"] = np.zeros(n_text_state, dtype=np.float32)
            dec_params[f"{prefix}_cv_w"] = sd[f"{pt_prefix}.cross_attn.value.weight"].numpy().T
            dec_params[f"{prefix}_cv_b"] = sd[f"{pt_prefix}.cross_attn.value.bias"].numpy()
            dec_params[f"{prefix}_cout_w"] = sd[f"{pt_prefix}.cross_attn.out.weight"].numpy().T
            dec_params[f"{prefix}_cout_b"] = sd[f"{pt_prefix}.cross_attn.out.bias"].numpy()

            # MLP LN
            dec_params[f"{prefix}_ln3_gamma"] = sd[f"{pt_prefix}.mlp_ln.weight"].numpy()
            dec_params[f"{prefix}_ln3_beta"] = sd[f"{pt_prefix}.mlp_ln.bias"].numpy()

            # MLP fc1, fc2
            dec_params[f"{prefix}_fc1_w"] = sd[f"{pt_prefix}.mlp.0.weight"].numpy().T
            dec_params[f"{prefix}_fc1_b"] = sd[f"{pt_prefix}.mlp.0.bias"].numpy()
            dec_params[f"{prefix}_fc2_w"] = sd[f"{pt_prefix}.mlp.2.weight"].numpy().T
            dec_params[f"{prefix}_fc2_b"] = sd[f"{pt_prefix}.mlp.2.bias"].numpy()

            dec_param_order.extend([
                f"{prefix}_ln1_gamma", f"{prefix}_ln1_beta",
                f"{prefix}_sq_w", f"{prefix}_sq_b",
                f"{prefix}_sk_w", f"{prefix}_sk_b",
                f"{prefix}_sv_w", f"{prefix}_sv_b",
                f"{prefix}_sout_w", f"{prefix}_sout_b",
                f"{prefix}_ln2_gamma", f"{prefix}_ln2_beta",
                f"{prefix}_cq_w", f"{prefix}_cq_b",
                f"{prefix}_ck_w", f"{prefix}_ck_b",
                f"{prefix}_cv_w", f"{prefix}_cv_b",
                f"{prefix}_cout_w", f"{prefix}_cout_b",
                f"{prefix}_ln3_gamma", f"{prefix}_ln3_beta",
                f"{prefix}_fc1_w", f"{prefix}_fc1_b",
                f"{prefix}_fc2_w", f"{prefix}_fc2_b",
            ])

        # Final LN + projection
        dec_params["dec_ln_f_gamma"] = sd["decoder.ln.weight"].numpy()
        dec_params["dec_ln_f_beta"] = sd["decoder.ln.bias"].numpy()
        # Projection: token_embedding.weight is [51865, 384], transpose to [384, 51865]
        dec_params["proj_w"] = sd["decoder.token_embedding.weight"].numpy().T  # [384, 51865]

        dec_param_order.extend(["dec_ln_f_gamma", "dec_ln_f_beta", "proj_w"])

        # Token embedding table and positional embedding table
        token_emb_table = sd["decoder.token_embedding.weight"].numpy()  # [51865, 384]
        pos_emb_table = sd["decoder.positional_embedding"].numpy()  # [448, 384]

        print(f"Decoder: {len(dec_param_order)} params (including 4 inputs)")

        # --- Run TF decoder reference (greedy decode) ---
        print("Running TF decoder reference (greedy decode)...")

        # Special tokens
        SOT = 50258       # <|startoftranscript|>
        EN = 50259        # <|en|>
        TRANSCRIBE = 50360  # <|transcribe|>
        NO_TIMESTAMPS = 50364  # <|notimestamps|>
        EOT = 50257       # <|endoftext|>

        initial_tokens = [SOT, EN, TRANSCRIBE, NO_TIMESTAMPS]
        tokens = list(initial_tokens)
        max_tokens = n_text_ctx  # 64

        def cross_attn_np(x, enc_out, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, n_head=6):
            B, S, C = x.shape
            _, ES, _ = enc_out.shape
            hd = C // n_head
            q = x @ q_w + q_b
            k = enc_out @ k_w + k_b
            v = enc_out @ v_w + v_b
            q = q.reshape(B, S, n_head, hd).transpose(0, 2, 1, 3)
            k = k.reshape(B, ES, n_head, hd).transpose(0, 2, 1, 3)
            v = v.reshape(B, ES, n_head, hd).transpose(0, 2, 1, 3)
            scale = 1.0 / np.sqrt(hd)
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            att = probs @ v
            att = att.transpose(0, 2, 1, 3).reshape(B, S, C)
            return att @ out_w + out_b

        def masked_self_attn_np(x, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, mask, n_head=6):
            B, S, C = x.shape
            hd = C // n_head
            q = x @ q_w + q_b
            k = x @ k_w + k_b
            v = x @ v_w + v_b
            q = q.reshape(B, S, n_head, hd).transpose(0, 2, 1, 3)
            k = k.reshape(B, S, n_head, hd).transpose(0, 2, 1, 3)
            v = v.reshape(B, S, n_head, hd).transpose(0, 2, 1, 3)
            scale = 1.0 / np.sqrt(hd)
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale + mask
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            att = probs @ v
            att = att.transpose(0, 2, 1, 3).reshape(B, S, C)
            return att @ out_w + out_b

        for step in range(max_tokens - len(initial_tokens)):
            # Pad tokens to 64
            padded = tokens + [0] * (n_text_ctx - len(tokens))
            # Look up embeddings
            tok_emb = token_emb_table[padded]  # [64, 384]
            p_emb = pos_emb_table[:n_text_ctx]  # [64, 384]
            tok_emb = tok_emb[np.newaxis, :, :]  # [1, 64, 384]
            p_emb = p_emb[np.newaxis, :, :]      # [1, 64, 384]

            # Causal mask: 0 for allowed, -1e9 for masked (future + padding)
            causal = np.zeros((1, 1, n_text_ctx, n_text_ctx), dtype=np.float32)
            for r in range(n_text_ctx):
                for c in range(n_text_ctx):
                    if c > r or c >= len(tokens):
                        causal[0, 0, r, c] = -1e9

            x = tok_emb + p_emb

            for i in range(n_text_layer):
                prefix = f"dec{i}"
                # Masked self-attention
                nx = layer_norm_np(x, dec_params[f"{prefix}_ln1_gamma"], dec_params[f"{prefix}_ln1_beta"])
                sa = masked_self_attn_np(nx,
                    dec_params[f"{prefix}_sq_w"], dec_params[f"{prefix}_sq_b"],
                    dec_params[f"{prefix}_sk_w"], dec_params[f"{prefix}_sk_b"],
                    dec_params[f"{prefix}_sv_w"], dec_params[f"{prefix}_sv_b"],
                    dec_params[f"{prefix}_sout_w"], dec_params[f"{prefix}_sout_b"],
                    causal)
                x = x + sa
                # Cross-attention
                nx = layer_norm_np(x, dec_params[f"{prefix}_ln2_gamma"], dec_params[f"{prefix}_ln2_beta"])
                ca = cross_attn_np(nx, encoder_output,
                    dec_params[f"{prefix}_cq_w"], dec_params[f"{prefix}_cq_b"],
                    dec_params[f"{prefix}_ck_w"], dec_params[f"{prefix}_ck_b"],
                    dec_params[f"{prefix}_cv_w"], dec_params[f"{prefix}_cv_b"],
                    dec_params[f"{prefix}_cout_w"], dec_params[f"{prefix}_cout_b"])
                x = x + ca
                # MLP
                nx = layer_norm_np(x, dec_params[f"{prefix}_ln3_gamma"], dec_params[f"{prefix}_ln3_beta"])
                h = nx @ dec_params[f"{prefix}_fc1_w"] + dec_params[f"{prefix}_fc1_b"]
                h = gelu_np(h)
                h = h @ dec_params[f"{prefix}_fc2_w"] + dec_params[f"{prefix}_fc2_b"]
                x = x + h

            x = layer_norm_np(x, dec_params["dec_ln_f_gamma"], dec_params["dec_ln_f_beta"])
            logits = x @ dec_params["proj_w"]  # [1, 64, 51865]

            # Greedy: argmax at current position
            pos = len(tokens) - 1
            next_token = int(np.argmax(logits[0, pos, :]))
            print(f"  Step {step}: pos={pos}, token={next_token}")

            if next_token == EOT:
                print(f"  EOT at step {step}")
                break
            tokens.append(next_token)

        # Decode tokens to text
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        # Filter out special tokens for text decode
        text_tokens = [t for t in tokens if t < 50257]
        transcription = enc.decode(text_tokens)
        print(f"Transcription: '{transcription}'")
        print(f"Token sequence: {tokens}")

        # --- Save decoder fixtures ---
        dec_fixtures = []
        # encoder_output
        info = write_f32(fixture_dir / "whisper_dec_param_000_encoder_output.raw.f32", encoder_output)
        dec_fixtures.append(info)

        # Save token_emb_table and pos_emb_table for the orchestrator
        write_f32(fixture_dir / "whisper_token_embedding_table.raw.f32", token_emb_table)
        write_f32(fixture_dir / "whisper_pos_embedding_table.raw.f32", pos_emb_table)

        # Save first-step decoder inputs for validation
        padded_init = initial_tokens + [0] * (n_text_ctx - len(initial_tokens))
        tok_emb_init = token_emb_table[padded_init][np.newaxis, :, :]
        pos_emb_init = pos_emb_table[:n_text_ctx][np.newaxis, :, :]
        causal_init = np.zeros((1, 1, n_text_ctx, n_text_ctx), dtype=np.float32)
        for r in range(n_text_ctx):
            for c in range(n_text_ctx):
                if c > r or c >= len(initial_tokens):
                    causal_init[0, 0, r, c] = -1e9

        info = write_f32(fixture_dir / "whisper_dec_param_001_token_emb.raw.f32", tok_emb_init)
        dec_fixtures.append(info)
        info = write_f32(fixture_dir / "whisper_dec_param_002_pos_emb.raw.f32", pos_emb_init)
        dec_fixtures.append(info)
        info = write_f32(fixture_dir / "whisper_dec_param_003_causal_mask.raw.f32", causal_init)
        dec_fixtures.append(info)

        # Decoder weight fixtures
        weight_idx = 4
        for name in dec_param_order[4:]:  # skip the 4 inputs
            info = write_f32(fixture_dir / f"whisper_dec_param_{weight_idx:03d}_{name}.raw.f32", dec_params[name])
            dec_fixtures.append(info)
            weight_idx += 1

        # Save decoder inputs list
        dec_inputs_path = fixture_dir / "whisper_decoder_inputs.txt"
        dec_inputs_path.write_text(",".join(item["path"] for item in dec_fixtures) + "\n")

        # --- Save mel input fixture ---
        mel_info = write_f32(fixture_dir / "whisper_mel_input.raw.f32", mel_input)

        # --- Report ---
        report = {
            "status": "pass",
            "model": "whisper_tiny",
            "tensorflow_version": tf.__version__,
            "whisper_version": whisper.__version__,
            "encoder_input_count": len(enc_fixtures),
            "decoder_input_count": len(dec_fixtures),
            "encoder_inputs_list": str(enc_inputs_path),
            "decoder_inputs_list": str(dec_inputs_path),
            "encoder_output": enc_output_info,
            "encoder_output_shape": list(encoder_output.shape),
            "encoder_output_range": {
                "min": float(np.min(encoder_output)),
                "max": float(np.max(encoder_output)),
                "mean": float(np.mean(encoder_output)),
            },
            "decoder_tokens": tokens,
            "decoder_transcription": transcription,
            "decoder_steps": len(tokens) - len(initial_tokens),
            "mel_input": mel_info,
            "mel_shape": list(mel_input.shape),
        }
        report_path = out_dir / "whisper_tf_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "blocked",
            "model": "whisper_tiny",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            },
        }
        (out_dir / "whisper_tf_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
