#!/usr/bin/env python3
"""Export Whisper-tiny encoder and decoder as StableHLO text.

Whisper-tiny architecture (39M params):
  Encoder: mel [1,3000,80] -> encoder_output [1,1500,384]
    - 2x Conv1D (via Conv2D with [1,1,T,C] expansion)
    - Positional embedding addition
    - 4x encoder blocks: LayerNorm + MultiHeadSelfAttention + LayerNorm + MLP(GELU)

  Decoder step: encoder_output + token_emb + pos_emb + causal_mask -> logits [1,S,51865]
    - 4x decoder blocks: LN + MaskedSelfAttn + LN + CrossAttn + LN + MLP(GELU)
    - Final LN + projection

All ops map to already-supported HLO ops:
  Conv2D, dot_general, reshape, transpose, reduce, broadcast,
  add, subtract, multiply, divide, rsqrt, exp, tanh, logistic, constant
"""
import argparse
import collections
import json
import pathlib
import re
import sys
import traceback


def write_histogram(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def histogram_for_text(model, output, text, tf_version, input_specs, status, error=None):
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
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["encoder", "decoder"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--histogram", required=True)
    args = parser.parse_args()

    out = pathlib.Path(args.output)
    histogram = pathlib.Path(args.histogram)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU for HLO extraction
        import tensorflow as tf
        import numpy as np

        tf_version = tf.__version__

        # Whisper-tiny dimensions
        n_mels = 80
        n_audio_ctx = 1500
        n_audio_state = 384
        n_audio_head = 6
        n_audio_layer = 4
        n_text_ctx = 64       # fixed decoder sequence length (padded)
        n_text_state = 384
        n_text_head = 6
        n_text_layer = 4
        n_vocab = 51865
        head_dim = n_audio_state // n_audio_head  # 64

        if args.component == "encoder":
            # --- Encoder ---
            # Inputs: mel [1, 3000, 80]
            # Weights per conv: weight + bias
            # Weights per encoder block: ln1_gamma, ln1_beta, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b,
            #                            ln2_gamma, ln2_beta, fc1_w, fc1_b, fc2_w, fc2_b
            # Final: ln_f_gamma, ln_f_beta
            input_specs = [
                {"shape": [1, 3000, n_mels], "dtype": "float32", "name": "mel"},
                # Conv1 weights: [1, 3, 80, 384] + bias [384]
                {"shape": [1, 3, n_mels, n_audio_state], "dtype": "float32", "name": "conv1_w"},
                {"shape": [n_audio_state], "dtype": "float32", "name": "conv1_b"},
                # Conv2 weights: [1, 3, 384, 384] + bias [384]
                {"shape": [1, 3, n_audio_state, n_audio_state], "dtype": "float32", "name": "conv2_w"},
                {"shape": [n_audio_state], "dtype": "float32", "name": "conv2_b"},
                # Positional embedding: [1500, 384]
                {"shape": [n_audio_ctx, n_audio_state], "dtype": "float32", "name": "pos_emb"},
            ]
            # 4 encoder blocks x 16 params each
            for i in range(n_audio_layer):
                prefix = f"enc{i}"
                input_specs.extend([
                    {"shape": [n_audio_state], "dtype": "float32", "name": f"{prefix}_ln1_gamma"},
                    {"shape": [n_audio_state], "dtype": "float32", "name": f"{prefix}_ln1_beta"},
                    {"shape": [n_audio_state, n_audio_state], "dtype": "float32", "name": f"{prefix}_q_w"},
                    {"shape": [n_audio_state], "dtype": "float32", "name": f"{prefix}_q_b"},
                    {"shape": [n_audio_state, n_audio_state], "dtype": "float32", "name": f"{prefix}_k_w"},
                    {"shape": [n_audio_state], "dtype": "float32", "name": f"{prefix}_k_b"},
                    {"shape": [n_audio_state, n_audio_state], "dtype": "float32", "name": f"{prefix}_v_w"},
                    {"shape": [n_audio_state], "dtype": "float32", "name": f"{prefix}_v_b"},
                    {"shape": [n_audio_state, n_audio_state], "dtype": "float32", "name": f"{prefix}_out_w"},
                    {"shape": [n_audio_state], "dtype": "float32", "name": f"{prefix}_out_b"},
                    {"shape": [n_audio_state], "dtype": "float32", "name": f"{prefix}_ln2_gamma"},
                    {"shape": [n_audio_state], "dtype": "float32", "name": f"{prefix}_ln2_beta"},
                    {"shape": [n_audio_state, n_audio_state * 4], "dtype": "float32", "name": f"{prefix}_fc1_w"},
                    {"shape": [n_audio_state * 4], "dtype": "float32", "name": f"{prefix}_fc1_b"},
                    {"shape": [n_audio_state * 4, n_audio_state], "dtype": "float32", "name": f"{prefix}_fc2_w"},
                    {"shape": [n_audio_state], "dtype": "float32", "name": f"{prefix}_fc2_b"},
                ])
            # Final layer norm
            input_specs.extend([
                {"shape": [n_audio_state], "dtype": "float32", "name": "ln_f_gamma"},
                {"shape": [n_audio_state], "dtype": "float32", "name": "ln_f_beta"},
            ])

            @tf.function(jit_compile=True)
            def whisper_encoder(
                mel,         # [1, 3000, 80]
                conv1_w,     # [1, 3, 80, 384]
                conv1_b,     # [384]
                conv2_w,     # [1, 3, 384, 384]
                conv2_b,     # [384]
                pos_emb,     # [1500, 384]
                # Block 0
                enc0_ln1_gamma, enc0_ln1_beta,
                enc0_q_w, enc0_q_b, enc0_k_w, enc0_k_b, enc0_v_w, enc0_v_b,
                enc0_out_w, enc0_out_b,
                enc0_ln2_gamma, enc0_ln2_beta,
                enc0_fc1_w, enc0_fc1_b, enc0_fc2_w, enc0_fc2_b,
                # Block 1
                enc1_ln1_gamma, enc1_ln1_beta,
                enc1_q_w, enc1_q_b, enc1_k_w, enc1_k_b, enc1_v_w, enc1_v_b,
                enc1_out_w, enc1_out_b,
                enc1_ln2_gamma, enc1_ln2_beta,
                enc1_fc1_w, enc1_fc1_b, enc1_fc2_w, enc1_fc2_b,
                # Block 2
                enc2_ln1_gamma, enc2_ln1_beta,
                enc2_q_w, enc2_q_b, enc2_k_w, enc2_k_b, enc2_v_w, enc2_v_b,
                enc2_out_w, enc2_out_b,
                enc2_ln2_gamma, enc2_ln2_beta,
                enc2_fc1_w, enc2_fc1_b, enc2_fc2_w, enc2_fc2_b,
                # Block 3
                enc3_ln1_gamma, enc3_ln1_beta,
                enc3_q_w, enc3_q_b, enc3_k_w, enc3_k_b, enc3_v_w, enc3_v_b,
                enc3_out_w, enc3_out_b,
                enc3_ln2_gamma, enc3_ln2_beta,
                enc3_fc1_w, enc3_fc1_b, enc3_fc2_w, enc3_fc2_b,
                # Final LN
                ln_f_gamma, ln_f_beta,
            ):
                def layer_norm(x, gamma, beta, eps=1e-5):
                    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
                    var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
                    return (x - mean) * tf.math.rsqrt(var + eps) * gamma + beta

                def gelu(x):
                    # tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    c = tf.constant(0.7978845608, dtype=tf.float32)  # sqrt(2/pi)
                    return 0.5 * x * (1.0 + tf.math.tanh(c * (x + 0.044715 * x * x * x)))

                def self_attention(x, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, num_heads=6):
                    seq_len = 1500
                    hd = 64  # head_dim
                    q = tf.linalg.matmul(x, q_w) + q_b  # [1, S, 384]
                    k = tf.linalg.matmul(x, k_w) + k_b
                    v = tf.linalg.matmul(x, v_w) + v_b

                    q = tf.reshape(q, [1, seq_len, num_heads, hd])
                    q = tf.transpose(q, [0, 2, 1, 3])  # [1, 6, 1500, 64]
                    k = tf.reshape(k, [1, seq_len, num_heads, hd])
                    k = tf.transpose(k, [0, 2, 1, 3])
                    v = tf.reshape(v, [1, seq_len, num_heads, hd])
                    v = tf.transpose(v, [0, 2, 1, 3])

                    scale = tf.math.rsqrt(tf.constant(64.0, dtype=tf.float32))
                    scores = tf.linalg.matmul(q, k, transpose_b=True) * scale  # [1,6,1500,1500]
                    probs = tf.nn.softmax(scores, axis=-1)
                    att = tf.linalg.matmul(probs, v)  # [1,6,1500,64]

                    att = tf.transpose(att, [0, 2, 1, 3])  # [1,1500,6,64]
                    att = tf.reshape(att, [1, seq_len, 384])
                    return tf.linalg.matmul(att, out_w) + out_b

                def encoder_block(x, ln1_g, ln1_b, q_w, q_b, k_w, k_b, v_w, v_b,
                                  out_w, out_b, ln2_g, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b):
                    # Self-attention with residual
                    nx = layer_norm(x, ln1_g, ln1_b)
                    att = self_attention(nx, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b)
                    x = x + att
                    # MLP with residual
                    nx = layer_norm(x, ln2_g, ln2_b)
                    h = tf.linalg.matmul(nx, fc1_w) + fc1_b  # [1, 1500, 1536]
                    h = gelu(h)
                    h = tf.linalg.matmul(h, fc2_w) + fc2_b   # [1, 1500, 384]
                    x = x + h
                    return x

                # Conv1: [1, 3000, 80] -> expand to [1, 1, 3000, 80] -> conv2d -> [1, 1, 3000, 384]
                x = tf.reshape(mel, [1, 1, 3000, n_mels])
                x = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding="SAME") + conv1_b
                x = gelu(x)  # [1, 1, 3000, 384]

                # Conv2: stride 2 along time -> [1, 1, 1500, 384]
                x = tf.nn.conv2d(x, conv2_w, strides=[1, 1, 2, 1], padding="SAME") + conv2_b
                x = gelu(x)

                # Squeeze spatial: [1, 1500, 384]
                x = tf.reshape(x, [1, n_audio_ctx, n_audio_state])

                # Add positional embedding
                x = x + pos_emb

                # Encoder blocks
                x = encoder_block(x, enc0_ln1_gamma, enc0_ln1_beta,
                                  enc0_q_w, enc0_q_b, enc0_k_w, enc0_k_b, enc0_v_w, enc0_v_b,
                                  enc0_out_w, enc0_out_b,
                                  enc0_ln2_gamma, enc0_ln2_beta,
                                  enc0_fc1_w, enc0_fc1_b, enc0_fc2_w, enc0_fc2_b)
                x = encoder_block(x, enc1_ln1_gamma, enc1_ln1_beta,
                                  enc1_q_w, enc1_q_b, enc1_k_w, enc1_k_b, enc1_v_w, enc1_v_b,
                                  enc1_out_w, enc1_out_b,
                                  enc1_ln2_gamma, enc1_ln2_beta,
                                  enc1_fc1_w, enc1_fc1_b, enc1_fc2_w, enc1_fc2_b)
                x = encoder_block(x, enc2_ln1_gamma, enc2_ln1_beta,
                                  enc2_q_w, enc2_q_b, enc2_k_w, enc2_k_b, enc2_v_w, enc2_v_b,
                                  enc2_out_w, enc2_out_b,
                                  enc2_ln2_gamma, enc2_ln2_beta,
                                  enc2_fc1_w, enc2_fc1_b, enc2_fc2_w, enc2_fc2_b)
                x = encoder_block(x, enc3_ln1_gamma, enc3_ln1_beta,
                                  enc3_q_w, enc3_q_b, enc3_k_w, enc3_k_b, enc3_v_w, enc3_v_b,
                                  enc3_out_w, enc3_out_b,
                                  enc3_ln2_gamma, enc3_ln2_beta,
                                  enc3_fc1_w, enc3_fc1_b, enc3_fc2_w, enc3_fc2_b)

                # Final layer norm
                x = layer_norm(x, ln_f_gamma, ln_f_beta)
                return x  # [1, 1500, 384]

            specs = [tf.TensorSpec(s["shape"], tf.float32) for s in input_specs]
            whisper_encoder.get_concrete_function(*specs)
            sample_args = [tf.zeros(s["shape"], tf.float32) for s in input_specs]
            _ = whisper_encoder(*sample_args)
            hlo = whisper_encoder.experimental_get_compiler_ir(*sample_args)(stage="hlo")
            text = hlo if isinstance(hlo, str) else str(hlo)
            out.write_text(text)
            hist = histogram_for_text("whisper_encoder", out, text, tf_version, input_specs, "pass")
            write_histogram(histogram, hist)
            print(json.dumps(hist, indent=2, sort_keys=True))
            return 0

        else:
            # --- Decoder step ---
            S = n_text_ctx  # 64
            input_specs = [
                {"shape": [1, n_audio_ctx, n_text_state], "dtype": "float32", "name": "encoder_output"},
                {"shape": [1, S, n_text_state], "dtype": "float32", "name": "token_emb"},
                {"shape": [1, S, n_text_state], "dtype": "float32", "name": "pos_emb"},
                {"shape": [1, 1, S, S], "dtype": "float32", "name": "causal_mask"},
            ]
            # 4 decoder blocks x 26 params each
            for i in range(n_text_layer):
                prefix = f"dec{i}"
                input_specs.extend([
                    # Masked self-attention
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_ln1_gamma"},
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_ln1_beta"},
                    {"shape": [n_text_state, n_text_state], "dtype": "float32", "name": f"{prefix}_sq_w"},
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_sq_b"},
                    {"shape": [n_text_state, n_text_state], "dtype": "float32", "name": f"{prefix}_sk_w"},
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_sk_b"},
                    {"shape": [n_text_state, n_text_state], "dtype": "float32", "name": f"{prefix}_sv_w"},
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_sv_b"},
                    {"shape": [n_text_state, n_text_state], "dtype": "float32", "name": f"{prefix}_sout_w"},
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_sout_b"},
                    # Cross-attention
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_ln2_gamma"},
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_ln2_beta"},
                    {"shape": [n_text_state, n_text_state], "dtype": "float32", "name": f"{prefix}_cq_w"},
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_cq_b"},
                    {"shape": [n_text_state, n_text_state], "dtype": "float32", "name": f"{prefix}_ck_w"},
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_ck_b"},
                    {"shape": [n_text_state, n_text_state], "dtype": "float32", "name": f"{prefix}_cv_w"},
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_cv_b"},
                    {"shape": [n_text_state, n_text_state], "dtype": "float32", "name": f"{prefix}_cout_w"},
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_cout_b"},
                    # MLP
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_ln3_gamma"},
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_ln3_beta"},
                    {"shape": [n_text_state, n_text_state * 4], "dtype": "float32", "name": f"{prefix}_fc1_w"},
                    {"shape": [n_text_state * 4], "dtype": "float32", "name": f"{prefix}_fc1_b"},
                    {"shape": [n_text_state * 4, n_text_state], "dtype": "float32", "name": f"{prefix}_fc2_w"},
                    {"shape": [n_text_state], "dtype": "float32", "name": f"{prefix}_fc2_b"},
                ])
            # Final LN + projection
            input_specs.extend([
                {"shape": [n_text_state], "dtype": "float32", "name": "dec_ln_f_gamma"},
                {"shape": [n_text_state], "dtype": "float32", "name": "dec_ln_f_beta"},
                {"shape": [n_text_state, n_vocab], "dtype": "float32", "name": "proj_w"},
            ])

            @tf.function(jit_compile=True)
            def whisper_decoder(
                encoder_output,  # [1, 1500, 384]
                token_emb,       # [1, 64, 384]
                pos_emb,         # [1, 64, 384]
                causal_mask,     # [1, 1, 64, 64]
                # Block 0
                dec0_ln1_gamma, dec0_ln1_beta,
                dec0_sq_w, dec0_sq_b, dec0_sk_w, dec0_sk_b, dec0_sv_w, dec0_sv_b, dec0_sout_w, dec0_sout_b,
                dec0_ln2_gamma, dec0_ln2_beta,
                dec0_cq_w, dec0_cq_b, dec0_ck_w, dec0_ck_b, dec0_cv_w, dec0_cv_b, dec0_cout_w, dec0_cout_b,
                dec0_ln3_gamma, dec0_ln3_beta,
                dec0_fc1_w, dec0_fc1_b, dec0_fc2_w, dec0_fc2_b,
                # Block 1
                dec1_ln1_gamma, dec1_ln1_beta,
                dec1_sq_w, dec1_sq_b, dec1_sk_w, dec1_sk_b, dec1_sv_w, dec1_sv_b, dec1_sout_w, dec1_sout_b,
                dec1_ln2_gamma, dec1_ln2_beta,
                dec1_cq_w, dec1_cq_b, dec1_ck_w, dec1_ck_b, dec1_cv_w, dec1_cv_b, dec1_cout_w, dec1_cout_b,
                dec1_ln3_gamma, dec1_ln3_beta,
                dec1_fc1_w, dec1_fc1_b, dec1_fc2_w, dec1_fc2_b,
                # Block 2
                dec2_ln1_gamma, dec2_ln1_beta,
                dec2_sq_w, dec2_sq_b, dec2_sk_w, dec2_sk_b, dec2_sv_w, dec2_sv_b, dec2_sout_w, dec2_sout_b,
                dec2_ln2_gamma, dec2_ln2_beta,
                dec2_cq_w, dec2_cq_b, dec2_ck_w, dec2_ck_b, dec2_cv_w, dec2_cv_b, dec2_cout_w, dec2_cout_b,
                dec2_ln3_gamma, dec2_ln3_beta,
                dec2_fc1_w, dec2_fc1_b, dec2_fc2_w, dec2_fc2_b,
                # Block 3
                dec3_ln1_gamma, dec3_ln1_beta,
                dec3_sq_w, dec3_sq_b, dec3_sk_w, dec3_sk_b, dec3_sv_w, dec3_sv_b, dec3_sout_w, dec3_sout_b,
                dec3_ln2_gamma, dec3_ln2_beta,
                dec3_cq_w, dec3_cq_b, dec3_ck_w, dec3_ck_b, dec3_cv_w, dec3_cv_b, dec3_cout_w, dec3_cout_b,
                dec3_ln3_gamma, dec3_ln3_beta,
                dec3_fc1_w, dec3_fc1_b, dec3_fc2_w, dec3_fc2_b,
                # Final
                dec_ln_f_gamma, dec_ln_f_beta,
                proj_w,
            ):
                def layer_norm(x, gamma, beta, eps=1e-5):
                    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
                    var = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
                    return (x - mean) * tf.math.rsqrt(var + eps) * gamma + beta

                def gelu(x):
                    c = tf.constant(0.7978845608, dtype=tf.float32)
                    return 0.5 * x * (1.0 + tf.math.tanh(c * (x + 0.044715 * x * x * x)))

                def masked_self_attention(x, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, mask):
                    seq = 64
                    hd = 64
                    nh = 6
                    q = tf.linalg.matmul(x, q_w) + q_b
                    k = tf.linalg.matmul(x, k_w) + k_b
                    v = tf.linalg.matmul(x, v_w) + v_b
                    q = tf.transpose(tf.reshape(q, [1, seq, nh, hd]), [0, 2, 1, 3])
                    k = tf.transpose(tf.reshape(k, [1, seq, nh, hd]), [0, 2, 1, 3])
                    v = tf.transpose(tf.reshape(v, [1, seq, nh, hd]), [0, 2, 1, 3])
                    scale = tf.math.rsqrt(tf.constant(64.0, dtype=tf.float32))
                    scores = tf.linalg.matmul(q, k, transpose_b=True) * scale
                    # Apply causal mask: mask is 0 for allowed, -1e9 for masked
                    scores = scores + mask
                    probs = tf.nn.softmax(scores, axis=-1)
                    att = tf.linalg.matmul(probs, v)
                    att = tf.reshape(tf.transpose(att, [0, 2, 1, 3]), [1, seq, 384])
                    return tf.linalg.matmul(att, out_w) + out_b

                def cross_attention(x, enc_out, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b):
                    seq = 64
                    enc_seq = 1500
                    hd = 64
                    nh = 6
                    q = tf.linalg.matmul(x, q_w) + q_b
                    k = tf.linalg.matmul(enc_out, k_w) + k_b
                    v = tf.linalg.matmul(enc_out, v_w) + v_b
                    q = tf.transpose(tf.reshape(q, [1, seq, nh, hd]), [0, 2, 1, 3])
                    k = tf.transpose(tf.reshape(k, [1, enc_seq, nh, hd]), [0, 2, 1, 3])
                    v = tf.transpose(tf.reshape(v, [1, enc_seq, nh, hd]), [0, 2, 1, 3])
                    scale = tf.math.rsqrt(tf.constant(64.0, dtype=tf.float32))
                    scores = tf.linalg.matmul(q, k, transpose_b=True) * scale
                    probs = tf.nn.softmax(scores, axis=-1)
                    att = tf.linalg.matmul(probs, v)
                    att = tf.reshape(tf.transpose(att, [0, 2, 1, 3]), [1, seq, 384])
                    return tf.linalg.matmul(att, out_w) + out_b

                def decoder_block(x, enc_out, mask,
                                  ln1_g, ln1_b, sq_w, sq_b, sk_w, sk_b, sv_w, sv_b, sout_w, sout_b,
                                  ln2_g, ln2_b, cq_w, cq_b, ck_w, ck_b, cv_w, cv_b, cout_w, cout_b,
                                  ln3_g, ln3_b, fc1_w, fc1_b, fc2_w, fc2_b):
                    # Masked self-attention
                    nx = layer_norm(x, ln1_g, ln1_b)
                    sa = masked_self_attention(nx, sq_w, sq_b, sk_w, sk_b, sv_w, sv_b, sout_w, sout_b, mask)
                    x = x + sa
                    # Cross-attention
                    nx = layer_norm(x, ln2_g, ln2_b)
                    ca = cross_attention(nx, enc_out, cq_w, cq_b, ck_w, ck_b, cv_w, cv_b, cout_w, cout_b)
                    x = x + ca
                    # MLP
                    nx = layer_norm(x, ln3_g, ln3_b)
                    h = tf.linalg.matmul(nx, fc1_w) + fc1_b
                    h = gelu(h)
                    h = tf.linalg.matmul(h, fc2_w) + fc2_b
                    x = x + h
                    return x

                x = token_emb + pos_emb

                x = decoder_block(x, encoder_output, causal_mask,
                    dec0_ln1_gamma, dec0_ln1_beta,
                    dec0_sq_w, dec0_sq_b, dec0_sk_w, dec0_sk_b, dec0_sv_w, dec0_sv_b, dec0_sout_w, dec0_sout_b,
                    dec0_ln2_gamma, dec0_ln2_beta,
                    dec0_cq_w, dec0_cq_b, dec0_ck_w, dec0_ck_b, dec0_cv_w, dec0_cv_b, dec0_cout_w, dec0_cout_b,
                    dec0_ln3_gamma, dec0_ln3_beta,
                    dec0_fc1_w, dec0_fc1_b, dec0_fc2_w, dec0_fc2_b)
                x = decoder_block(x, encoder_output, causal_mask,
                    dec1_ln1_gamma, dec1_ln1_beta,
                    dec1_sq_w, dec1_sq_b, dec1_sk_w, dec1_sk_b, dec1_sv_w, dec1_sv_b, dec1_sout_w, dec1_sout_b,
                    dec1_ln2_gamma, dec1_ln2_beta,
                    dec1_cq_w, dec1_cq_b, dec1_ck_w, dec1_ck_b, dec1_cv_w, dec1_cv_b, dec1_cout_w, dec1_cout_b,
                    dec1_ln3_gamma, dec1_ln3_beta,
                    dec1_fc1_w, dec1_fc1_b, dec1_fc2_w, dec1_fc2_b)
                x = decoder_block(x, encoder_output, causal_mask,
                    dec2_ln1_gamma, dec2_ln1_beta,
                    dec2_sq_w, dec2_sq_b, dec2_sk_w, dec2_sk_b, dec2_sv_w, dec2_sv_b, dec2_sout_w, dec2_sout_b,
                    dec2_ln2_gamma, dec2_ln2_beta,
                    dec2_cq_w, dec2_cq_b, dec2_ck_w, dec2_ck_b, dec2_cv_w, dec2_cv_b, dec2_cout_w, dec2_cout_b,
                    dec2_ln3_gamma, dec2_ln3_beta,
                    dec2_fc1_w, dec2_fc1_b, dec2_fc2_w, dec2_fc2_b)
                x = decoder_block(x, encoder_output, causal_mask,
                    dec3_ln1_gamma, dec3_ln1_beta,
                    dec3_sq_w, dec3_sq_b, dec3_sk_w, dec3_sk_b, dec3_sv_w, dec3_sv_b, dec3_sout_w, dec3_sout_b,
                    dec3_ln2_gamma, dec3_ln2_beta,
                    dec3_cq_w, dec3_cq_b, dec3_ck_w, dec3_ck_b, dec3_cv_w, dec3_cv_b, dec3_cout_w, dec3_cout_b,
                    dec3_ln3_gamma, dec3_ln3_beta,
                    dec3_fc1_w, dec3_fc1_b, dec3_fc2_w, dec3_fc2_b)

                x = layer_norm(x, dec_ln_f_gamma, dec_ln_f_beta)
                logits = tf.linalg.matmul(x, proj_w)  # [1, 64, 51865]
                return logits

            specs = [tf.TensorSpec(s["shape"], tf.float32) for s in input_specs]
            whisper_decoder.get_concrete_function(*specs)
            sample_args = [tf.zeros(s["shape"], tf.float32) for s in input_specs]
            _ = whisper_decoder(*sample_args)
            hlo = whisper_decoder.experimental_get_compiler_ir(*sample_args)(stage="hlo")
            text = hlo if isinstance(hlo, str) else str(hlo)
            out.write_text(text)
            hist = histogram_for_text("whisper_decoder", out, text, tf_version, input_specs, "pass")
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
            f"whisper_{args.component}", out, text,
            locals().get("tf_version", None), [], "blocked", error=error,
        )
        write_histogram(histogram, hist)
        print(json.dumps(hist, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
