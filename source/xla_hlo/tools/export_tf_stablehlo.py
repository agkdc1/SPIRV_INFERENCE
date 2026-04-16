#!/usr/bin/env python3
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
    parser.add_argument(
        "--model", choices=["resnet50", "bert_base", "bert_smoke"], required=True
    )
    parser.add_argument("--out")
    parser.add_argument("--output")
    parser.add_argument("--histogram")
    args = parser.parse_args()

    output_arg = args.output or args.out
    if not output_arg:
        parser.error("--output or --out is required")
    out = pathlib.Path(output_arg)
    histogram = pathlib.Path(args.histogram) if args.histogram else out.with_suffix(out.suffix + ".histogram.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import tensorflow as tf

        tf_version = tf.__version__
        if args.model == "resnet50":
            model = tf.keras.applications.ResNet50(weights="imagenet")
            input_specs = [{"shape": [1, 224, 224, 3], "dtype": "float32"}]
            sample_args = [tf.zeros([1, 224, 224, 3], tf.float32)]
            compiled_fn = tf.function(lambda x: model(x, training=False), jit_compile=True)
            compiled_fn.get_concrete_function(tf.TensorSpec([1, 224, 224, 3], tf.float32))
            compiled_fn(*sample_args)
        elif args.model == "bert_base":
            hidden = 768
            seq = 16
            heads = 12
            head_dim = hidden // heads
            input_specs = [
                {"shape": [1, seq, hidden], "dtype": "float32", "name": "x"},
                {"shape": [hidden, hidden], "dtype": "float32", "name": "qkv_weight"},
                {"shape": [hidden, hidden], "dtype": "float32", "name": "out_weight"},
            ]

            @tf.function(jit_compile=True)
            def bert_base_fn(x, qkv_weight, out_weight):
                q = tf.linalg.matmul(x, qkv_weight)
                q = tf.reshape(q, [1, seq, heads, head_dim])
                q = tf.transpose(q, [0, 2, 1, 3])
                scores = tf.linalg.matmul(q, q, transpose_b=True) * 0.125
                probs = tf.nn.softmax(scores, axis=-1)
                ctx = tf.linalg.matmul(probs, q)
                ctx = tf.transpose(ctx, [0, 2, 1, 3])
                ctx = tf.reshape(ctx, [1, seq, hidden])
                y = tf.linalg.matmul(ctx, out_weight)
                return tf.nn.gelu(y + x)

            sample_args = [
                tf.zeros([1, seq, hidden], tf.float32),
                tf.ones([hidden, hidden], tf.float32) * 0.001,
                tf.ones([hidden, hidden], tf.float32) * 0.001,
            ]
            compiled_fn = bert_base_fn
            compiled_fn.get_concrete_function(
                tf.TensorSpec([1, seq, hidden], tf.float32),
                tf.TensorSpec([hidden, hidden], tf.float32),
                tf.TensorSpec([hidden, hidden], tf.float32),
            )
            compiled_fn(*sample_args)
        else:
            input_specs = [
                {"shape": [1, 16, 64], "dtype": "float32", "name": "x"},
                {"shape": [64, 64], "dtype": "float32", "name": "w"},
            ]

            @tf.function(jit_compile=True)
            def bert_smoke_fn(x, w):
                y = tf.linalg.matmul(x, w)
                return tf.nn.gelu(y + 0.1)

            compiled_fn = bert_smoke_fn
            sample_args = [
                tf.zeros([1, 16, 64], tf.float32),
                tf.ones([64, 64], tf.float32),
            ]
            compiled_fn.get_concrete_function(
                tf.TensorSpec([1, 16, 64], tf.float32),
                tf.TensorSpec([64, 64], tf.float32),
            )
            compiled_fn(*sample_args)

        hlo = compiled_fn.experimental_get_compiler_ir(*sample_args)(stage="hlo")
        text = hlo if isinstance(hlo, str) else str(hlo)
        out.write_text(text)
        hist = histogram_for_text(
            args.model, out, text, tf_version, input_specs, "pass", error=None
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
            args.model,
            out,
            text,
            globals().get("tf", None).__version__ if "tf" in globals() else None,
            [],
            "blocked",
            error=error,
        )
        write_histogram(histogram, hist)
        print(json.dumps(hist, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
