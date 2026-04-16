#!/usr/bin/env python3
"""Export a Tiny Stable Diffusion UNet denoising step as StableHLO text.

Exercises ALL SD-relevant HLO ops:
  - Conv2d (3x3, stride 1, same padding) for ResBlocks
  - GroupNorm pattern (reduce -> broadcast -> sub -> mul -> rsqrt -> add)
  - SiLU (logistic * input)
  - Self-attention (dot_general with batch dims, softmax, scaled)
  - Cross-attention (same pattern, different K/V source)
  - Skip connection concatenation
  - Timestep embedding (sinusoidal: sine + cosine)
  - Slice (for splitting heads)

Tiny SD shapes:
  Latent: [1, 8, 8, 4]
  Down channels: [64, 128]
  Mid channels: 128
  Attention heads: 4
  Cross-attention context: [1, 8, 64]
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--histogram", required=True)
    args = parser.parse_args()

    out = pathlib.Path(args.output)
    histogram = pathlib.Path(args.histogram)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import tensorflow as tf
        import numpy as np

        tf_version = tf.__version__

        # Tiny SD UNet dimensions
        B = 1
        H, W = 8, 8        # latent spatial
        C_in = 4            # latent channels
        C_down1 = 64
        C_down2 = 128
        C_mid = 128
        heads = 4
        ctx_seq = 8         # cross-attention context sequence length
        ctx_dim = 64        # cross-attention context embedding dim
        t_dim = 128         # timestep embedding dim

        input_specs = [
            {"shape": [B, H, W, C_in], "dtype": "float32", "name": "latent"},
            {"shape": [B, t_dim], "dtype": "float32", "name": "timestep_emb"},
            {"shape": [B, ctx_seq, ctx_dim], "dtype": "float32", "name": "context"},
            # Down block 1 weights
            {"shape": [3, 3, C_in, C_down1], "dtype": "float32", "name": "down1_conv_w"},
            {"shape": [C_down1], "dtype": "float32", "name": "down1_gn_gamma"},
            {"shape": [C_down1], "dtype": "float32", "name": "down1_gn_beta"},
            {"shape": [t_dim, C_down1], "dtype": "float32", "name": "down1_temb_w"},
            # Down block 2 weights
            {"shape": [3, 3, C_down1, C_down2], "dtype": "float32", "name": "down2_conv_w"},
            {"shape": [C_down2], "dtype": "float32", "name": "down2_gn_gamma"},
            {"shape": [C_down2], "dtype": "float32", "name": "down2_gn_beta"},
            {"shape": [t_dim, C_down2], "dtype": "float32", "name": "down2_temb_w"},
            # Mid block weights (self-attention + cross-attention)
            {"shape": [C_down2, C_down2], "dtype": "float32", "name": "mid_self_qkv_w"},
            {"shape": [C_down2, C_down2], "dtype": "float32", "name": "mid_self_out_w"},
            {"shape": [C_down2, C_down2], "dtype": "float32", "name": "mid_cross_q_w"},
            {"shape": [ctx_dim, C_down2], "dtype": "float32", "name": "mid_cross_kv_w"},
            {"shape": [C_down2, C_down2], "dtype": "float32", "name": "mid_cross_out_w"},
            {"shape": [C_down2], "dtype": "float32", "name": "mid_gn_gamma"},
            {"shape": [C_down2], "dtype": "float32", "name": "mid_gn_beta"},
            # Up block weights
            {"shape": [3, 3, C_down2 + C_down2, C_down1], "dtype": "float32", "name": "up_conv_w"},
            {"shape": [C_down1], "dtype": "float32", "name": "up_gn_gamma"},
            {"shape": [C_down1], "dtype": "float32", "name": "up_gn_beta"},
            # Output conv
            {"shape": [3, 3, C_down1, C_in], "dtype": "float32", "name": "out_conv_w"},
        ]

        @tf.function(jit_compile=True)
        def sd_unet_tiny(
            latent,        # [1, 8, 8, 4]
            timestep_emb,  # [1, 128]
            context,       # [1, 8, 64]
            down1_conv_w,  # [3, 3, 4, 64]
            down1_gn_gamma,  # [64]
            down1_gn_beta,   # [64]
            down1_temb_w,    # [128, 64]
            down2_conv_w,  # [3, 3, 64, 128]
            down2_gn_gamma,  # [128]
            down2_gn_beta,   # [128]
            down2_temb_w,    # [128, 128]
            mid_self_qkv_w,  # [128, 128]
            mid_self_out_w,  # [128, 128]
            mid_cross_q_w,   # [128, 128]
            mid_cross_kv_w,  # [64, 128]
            mid_cross_out_w, # [128, 128]
            mid_gn_gamma,    # [128]
            mid_gn_beta,     # [128]
            up_conv_w,     # [3, 3, 256, 64]
            up_gn_gamma,     # [64]
            up_gn_beta,      # [64]
            out_conv_w,    # [3, 3, 64, 4]
        ):
            def group_norm(x, gamma, beta, num_groups=4, eps=1e-5):
                # x: [B, H, W, C]
                shape = tf.shape(x)
                b, h, w, c = shape[0], shape[1], shape[2], x.shape[-1]
                g = num_groups
                gc = c // g
                # reshape to [B, H, W, G, GC]
                x_g = tf.reshape(x, [b, h, w, g, gc])
                # mean/var over H, W, GC (axes 1, 2, 4)
                mean = tf.reduce_mean(x_g, axis=[1, 2, 4], keepdims=True)
                var = tf.reduce_mean(tf.square(x_g - mean), axis=[1, 2, 4], keepdims=True)
                x_norm = (x_g - mean) * tf.math.rsqrt(var + eps)
                x_norm = tf.reshape(x_norm, [b, h, w, c])
                return x_norm * gamma + beta

            def silu(x):
                return x * tf.math.sigmoid(x)

            def self_attention(x_flat, qkv_w, out_w, num_heads=4):
                # x_flat: [B, S, C] where S = H*W
                seq_len = x_flat.shape[1] or tf.shape(x_flat)[1]
                c = x_flat.shape[2] or tf.shape(x_flat)[2]
                head_dim = c // num_heads

                q = tf.linalg.matmul(x_flat, qkv_w)  # [B, S, C]
                q = tf.reshape(q, [1, seq_len, num_heads, head_dim])
                q = tf.transpose(q, [0, 2, 1, 3])  # [B, heads, S, head_dim]

                # self-attention: Q=K=V
                scale = tf.math.rsqrt(tf.cast(head_dim, tf.float32))
                scores = tf.linalg.matmul(q, q, transpose_b=True) * scale
                probs = tf.nn.softmax(scores, axis=-1)
                att = tf.linalg.matmul(probs, q)

                att = tf.transpose(att, [0, 2, 1, 3])  # [B, S, heads, head_dim]
                att = tf.reshape(att, [1, seq_len, c])
                return tf.linalg.matmul(att, out_w)

            def cross_attention(x_flat, context, q_w, kv_w, out_w, num_heads=4):
                # x_flat: [B, S, C], context: [B, ctx_seq, ctx_dim]
                seq_len = x_flat.shape[1] or tf.shape(x_flat)[1]
                c = x_flat.shape[2] or tf.shape(x_flat)[2]
                head_dim = c // num_heads

                q = tf.linalg.matmul(x_flat, q_w)  # [B, S, C]
                k = tf.linalg.matmul(context, kv_w)  # [B, ctx_seq, C]

                q = tf.reshape(q, [1, seq_len, num_heads, head_dim])
                q = tf.transpose(q, [0, 2, 1, 3])  # [B, heads, S, head_dim]
                k = tf.reshape(k, [1, context.shape[1], num_heads, head_dim])
                k = tf.transpose(k, [0, 2, 1, 3])  # [B, heads, ctx_seq, head_dim]

                scale = tf.math.rsqrt(tf.cast(head_dim, tf.float32))
                scores = tf.linalg.matmul(q, k, transpose_b=True) * scale
                probs = tf.nn.softmax(scores, axis=-1)
                att = tf.linalg.matmul(probs, k)

                att = tf.transpose(att, [0, 2, 1, 3])
                att = tf.reshape(att, [1, seq_len, c])
                return tf.linalg.matmul(att, out_w)

            # --- Sinusoidal timestep embedding ---
            # timestep_emb: [B, t_dim] — treat as raw embedding, apply sin/cos
            half = t_dim // 2
            # Slice first half and second half
            t_sin = tf.math.sin(timestep_emb[:, :half])   # [B, half]
            t_cos = tf.math.cos(timestep_emb[:, half:])   # [B, half]
            t_emb = tf.concat([t_sin, t_cos], axis=-1)    # [B, t_dim]
            t_emb = silu(t_emb)

            # --- Down block 1: Conv + GroupNorm + SiLU + timestep inject ---
            h1 = tf.nn.conv2d(latent, down1_conv_w, strides=1, padding="SAME")  # [1,8,8,64]
            h1 = group_norm(h1, down1_gn_gamma, down1_gn_beta, num_groups=4)
            h1 = silu(h1)
            # Inject timestep: project t_emb to channel dim, add
            t_proj1 = tf.linalg.matmul(t_emb, down1_temb_w)  # [B, 64]
            h1 = h1 + tf.reshape(t_proj1, [1, 1, 1, C_down1])
            skip1 = h1  # save for skip connection

            # --- Down block 2: Conv + GroupNorm + SiLU + timestep inject ---
            h2 = tf.nn.conv2d(h1, down2_conv_w, strides=1, padding="SAME")  # [1,8,8,128]
            h2 = group_norm(h2, down2_gn_gamma, down2_gn_beta, num_groups=4)
            h2 = silu(h2)
            t_proj2 = tf.linalg.matmul(t_emb, down2_temb_w)  # [B, 128]
            h2 = h2 + tf.reshape(t_proj2, [1, 1, 1, C_down2])
            skip2 = h2

            # --- Mid block: GroupNorm + Self-Attention + Cross-Attention ---
            hm = group_norm(h2, mid_gn_gamma, mid_gn_beta, num_groups=4)
            # Flatten spatial for attention: [B, H*W, C]
            hm_flat = tf.reshape(hm, [1, 64, C_down2])  # 8*8=64

            # Self-attention
            sa_out = self_attention(hm_flat, mid_self_qkv_w, mid_self_out_w, num_heads=heads)
            hm_flat = hm_flat + sa_out  # residual

            # Cross-attention
            ca_out = cross_attention(hm_flat, context, mid_cross_q_w, mid_cross_kv_w, mid_cross_out_w, num_heads=heads)
            hm_flat = hm_flat + ca_out  # residual

            hm = tf.reshape(hm_flat, [1, 8, 8, C_down2])

            # --- Up block: Concatenate skip + Conv + GroupNorm + SiLU ---
            # Skip connection: concat along channel axis
            hu = tf.concat([hm, skip2], axis=-1)  # [1, 8, 8, 256]
            hu = tf.nn.conv2d(hu, up_conv_w, strides=1, padding="SAME")  # [1, 8, 8, 64]
            hu = group_norm(hu, up_gn_gamma, up_gn_beta, num_groups=4)
            hu = silu(hu)

            # --- Output conv ---
            out = tf.nn.conv2d(hu, out_conv_w, strides=1, padding="SAME")  # [1, 8, 8, 4]
            return out

        # Build concrete function with TensorSpecs
        specs = [tf.TensorSpec(s["shape"], tf.float32) for s in input_specs]
        sd_unet_tiny.get_concrete_function(*specs)

        # Create sample inputs for tracing
        sample_args = [tf.zeros(s["shape"], tf.float32) for s in input_specs]
        _ = sd_unet_tiny(*sample_args)

        # Export HLO — experimental_get_compiler_ir is on tf.function, not ConcreteFunction
        hlo = sd_unet_tiny.experimental_get_compiler_ir(*sample_args)(stage="hlo")
        text = hlo if isinstance(hlo, str) else str(hlo)
        out.write_text(text)

        hist = histogram_for_text(
            "sd_unet_tiny", out, text, tf_version, input_specs, "pass"
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
            "sd_unet_tiny",
            out,
            text,
            locals().get("tf_version", None),
            [],
            "blocked",
            error=error,
        )
        write_histogram(histogram, hist)
        print(json.dumps(hist, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
