#!/usr/bin/env python3
"""Generate TF reference output for SD UNet Tiny model.

Runs the same TF function used for HLO export with deterministic inputs,
saves parameter fixtures and expected output for parity comparison.
"""
import argparse
import hashlib
import json
import pathlib
import sys
import traceback

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
    args = parser.parse_args()

    fixture_dir = pathlib.Path(args.fixture_dir)
    out_dir = pathlib.Path(args.out_dir)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Tiny SD UNet dimensions — must match export_sd_unet.py exactly
        B = 1
        H, W = 8, 8
        C_in = 4
        C_down1 = 64
        C_down2 = 128
        C_mid = 128
        heads = 4
        ctx_seq = 8
        ctx_dim = 64
        t_dim = 128

        # Deterministic inputs using linspace for reproducibility
        params = {
            "latent":        np.linspace(-0.5, 0.5, B*H*W*C_in, dtype=np.float32).reshape(B, H, W, C_in),
            "timestep_emb":  np.linspace(0.0, 3.14, B*t_dim, dtype=np.float32).reshape(B, t_dim),
            "context":       np.linspace(-0.1, 0.1, B*ctx_seq*ctx_dim, dtype=np.float32).reshape(B, ctx_seq, ctx_dim),
            "down1_conv_w":  np.full((3, 3, C_in, C_down1), 0.01, dtype=np.float32),
            "down1_gn_gamma": np.ones(C_down1, dtype=np.float32),
            "down1_gn_beta": np.zeros(C_down1, dtype=np.float32),
            "down1_temb_w":  np.full((t_dim, C_down1), 0.005, dtype=np.float32),
            "down2_conv_w":  np.full((3, 3, C_down1, C_down2), 0.01, dtype=np.float32),
            "down2_gn_gamma": np.ones(C_down2, dtype=np.float32),
            "down2_gn_beta": np.zeros(C_down2, dtype=np.float32),
            "down2_temb_w":  np.full((t_dim, C_down2), 0.005, dtype=np.float32),
            "mid_self_qkv_w": np.eye(C_down2, dtype=np.float32) * 0.01,
            "mid_self_out_w": np.eye(C_down2, dtype=np.float32) * 0.01,
            "mid_cross_q_w":  np.eye(C_down2, dtype=np.float32) * 0.01,
            "mid_cross_kv_w": np.full((ctx_dim, C_down2), 0.01, dtype=np.float32),
            "mid_cross_out_w": np.eye(C_down2, dtype=np.float32) * 0.01,
            "mid_gn_gamma":  np.ones(C_down2, dtype=np.float32),
            "mid_gn_beta":   np.zeros(C_down2, dtype=np.float32),
            "up_conv_w":     np.full((3, 3, C_down2 + C_down2, C_down1), 0.005, dtype=np.float32),
            "up_gn_gamma":   np.ones(C_down1, dtype=np.float32),
            "up_gn_beta":    np.zeros(C_down1, dtype=np.float32),
            "out_conv_w":    np.full((3, 3, C_down1, C_in), 0.01, dtype=np.float32),
        }

        # Order must match the @tf.function signature in export_sd_unet.py
        param_order = [
            "latent", "timestep_emb", "context",
            "down1_conv_w", "down1_gn_gamma", "down1_gn_beta", "down1_temb_w",
            "down2_conv_w", "down2_gn_gamma", "down2_gn_beta", "down2_temb_w",
            "mid_self_qkv_w", "mid_self_out_w",
            "mid_cross_q_w", "mid_cross_kv_w", "mid_cross_out_w",
            "mid_gn_gamma", "mid_gn_beta",
            "up_conv_w", "up_gn_gamma", "up_gn_beta",
            "out_conv_w",
        ]

        # Replicate exact TF function from export_sd_unet.py
        @tf.function(jit_compile=True)
        def sd_unet_tiny(latent, timestep_emb, context,
                         down1_conv_w, down1_gn_gamma, down1_gn_beta, down1_temb_w,
                         down2_conv_w, down2_gn_gamma, down2_gn_beta, down2_temb_w,
                         mid_self_qkv_w, mid_self_out_w,
                         mid_cross_q_w, mid_cross_kv_w, mid_cross_out_w,
                         mid_gn_gamma, mid_gn_beta,
                         up_conv_w, up_gn_gamma, up_gn_beta, out_conv_w):

            def group_norm(x, gamma, beta, num_groups=4, eps=1e-5):
                shape = tf.shape(x)
                b, h, w, c = shape[0], shape[1], shape[2], x.shape[-1]
                g = num_groups
                gc = c // g
                x_g = tf.reshape(x, [b, h, w, g, gc])
                mean = tf.reduce_mean(x_g, axis=[1, 2, 4], keepdims=True)
                var = tf.reduce_mean(tf.square(x_g - mean), axis=[1, 2, 4], keepdims=True)
                x_norm = (x_g - mean) * tf.math.rsqrt(var + eps)
                x_norm = tf.reshape(x_norm, [b, h, w, c])
                return x_norm * gamma + beta

            def silu(x):
                return x * tf.math.sigmoid(x)

            def self_attention(x_flat, qkv_w, out_w, num_heads=4):
                seq_len = x_flat.shape[1] or tf.shape(x_flat)[1]
                c = x_flat.shape[2] or tf.shape(x_flat)[2]
                head_dim = c // num_heads
                q = tf.linalg.matmul(x_flat, qkv_w)
                q = tf.reshape(q, [1, seq_len, num_heads, head_dim])
                q = tf.transpose(q, [0, 2, 1, 3])
                scale = tf.math.rsqrt(tf.cast(head_dim, tf.float32))
                scores = tf.linalg.matmul(q, q, transpose_b=True) * scale
                probs = tf.nn.softmax(scores, axis=-1)
                att = tf.linalg.matmul(probs, q)
                att = tf.transpose(att, [0, 2, 1, 3])
                att = tf.reshape(att, [1, seq_len, c])
                return tf.linalg.matmul(att, out_w)

            def cross_attention(x_flat, context, q_w, kv_w, out_w, num_heads=4):
                seq_len = x_flat.shape[1] or tf.shape(x_flat)[1]
                c = x_flat.shape[2] or tf.shape(x_flat)[2]
                head_dim = c // num_heads
                q = tf.linalg.matmul(x_flat, q_w)
                k = tf.linalg.matmul(context, kv_w)
                q = tf.reshape(q, [1, seq_len, num_heads, head_dim])
                q = tf.transpose(q, [0, 2, 1, 3])
                k = tf.reshape(k, [1, context.shape[1], num_heads, head_dim])
                k = tf.transpose(k, [0, 2, 1, 3])
                scale = tf.math.rsqrt(tf.cast(head_dim, tf.float32))
                scores = tf.linalg.matmul(q, k, transpose_b=True) * scale
                probs = tf.nn.softmax(scores, axis=-1)
                att = tf.linalg.matmul(probs, k)
                att = tf.transpose(att, [0, 2, 1, 3])
                att = tf.reshape(att, [1, seq_len, c])
                return tf.linalg.matmul(att, out_w)

            half = t_dim // 2
            t_sin = tf.math.sin(timestep_emb[:, :half])
            t_cos = tf.math.cos(timestep_emb[:, half:])
            t_emb = tf.concat([t_sin, t_cos], axis=-1)
            t_emb = silu(t_emb)

            h1 = tf.nn.conv2d(latent, down1_conv_w, strides=1, padding="SAME")
            h1 = group_norm(h1, down1_gn_gamma, down1_gn_beta, num_groups=4)
            h1 = silu(h1)
            t_proj1 = tf.linalg.matmul(t_emb, down1_temb_w)
            h1 = h1 + tf.reshape(t_proj1, [1, 1, 1, C_down1])
            skip1 = h1

            h2 = tf.nn.conv2d(h1, down2_conv_w, strides=1, padding="SAME")
            h2 = group_norm(h2, down2_gn_gamma, down2_gn_beta, num_groups=4)
            h2 = silu(h2)
            t_proj2 = tf.linalg.matmul(t_emb, down2_temb_w)
            h2 = h2 + tf.reshape(t_proj2, [1, 1, 1, C_down2])
            skip2 = h2

            hm = group_norm(h2, mid_gn_gamma, mid_gn_beta, num_groups=4)
            hm_flat = tf.reshape(hm, [1, 64, C_down2])
            sa_out = self_attention(hm_flat, mid_self_qkv_w, mid_self_out_w, num_heads=heads)
            hm_flat = hm_flat + sa_out
            ca_out = cross_attention(hm_flat, context, mid_cross_q_w, mid_cross_kv_w, mid_cross_out_w, num_heads=heads)
            hm_flat = hm_flat + ca_out
            hm = tf.reshape(hm_flat, [1, 8, 8, C_down2])

            hu = tf.concat([hm, skip2], axis=-1)
            hu = tf.nn.conv2d(hu, up_conv_w, strides=1, padding="SAME")
            hu = group_norm(hu, up_gn_gamma, up_gn_beta, num_groups=4)
            hu = silu(hu)

            out = tf.nn.conv2d(hu, out_conv_w, strides=1, padding="SAME")
            return out

        # Run TF reference
        tf_args = [tf.constant(params[name]) for name in param_order]
        output = sd_unet_tiny(*tf_args).numpy().astype(np.float32)

        # Save fixtures
        fixtures = []
        for idx, name in enumerate(param_order):
            info = write_f32(fixture_dir / f"sd_unet_param_{idx:03d}_{name}.raw.f32", params[name])
            fixtures.append(info)

        # Save output
        output_info = write_f32(out_dir / "sd_unet_tiny_tf_output.raw.f32", output)

        # Save inputs list
        input_list = fixture_dir / "sd_unet_tiny_inputs.txt"
        input_list.write_text(",".join(item["path"] for item in fixtures) + "\n")

        # Report
        report = {
            "status": "pass",
            "model": "sd_unet_tiny",
            "tensorflow_version": tf.__version__,
            "input_count": len(fixtures),
            "inputs_list": str(input_list),
            "fixtures": fixtures,
            "output": output_info,
            "output_sha256": output_info["sha256"],
            "output_shape": list(output.shape),
            "output_range": {
                "min": float(np.min(output)),
                "max": float(np.max(output)),
                "mean": float(np.mean(output)),
            },
        }
        report_path = out_dir / "sd_unet_tiny_tf_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "blocked",
            "model": "sd_unet_tiny",
            "tensorflow_version": tf.__version__,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            },
        }
        (out_dir / "sd_unet_tiny_tf_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
