#!/usr/bin/env python3
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
    np.asarray(array, dtype=np.float32).tofile(path)
    return {
        "path": str(path),
        "shape": list(np.asarray(array).shape),
        "elements": int(np.asarray(array).size),
        "sha256": sha256_file(path),
    }


def synthetic_resnet_input():
    x = np.linspace(-1.0, 1.0, num=224 * 224 * 3, dtype=np.float32)
    return x.reshape(1, 224, 224, 3)


def run_resnet(args):
    fixture_dir = pathlib.Path(args.fixture_dir)
    out_dir = pathlib.Path(args.out_dir)
    model = tf.keras.applications.ResNet50(weights="imagenet")
    x = synthetic_resnet_input()
    preprocessed = tf.keras.applications.resnet50.preprocess_input(x.copy())
    output = model(preprocessed, training=False).numpy().astype(np.float32)

    fixtures = []
    fixtures.append(write_f32(fixture_dir / "resnet50_param_000_input.raw.f32", preprocessed))
    for index, weight in enumerate(model.weights, start=1):
        fixtures.append(write_f32(fixture_dir / f"resnet50_param_{index:03d}.raw.f32", weight.numpy()))

    output_info = write_f32(out_dir / "output.raw.f32", output)
    input_list = fixture_dir / "resnet50_inputs.txt"
    input_list.write_text(",".join(item["path"] for item in fixtures) + "\n")
    mapping_path = pathlib.Path("xla_hlo/reports/resnet50_input_hlo_order_mapping.json")
    if mapping_path.exists():
        mapped_paths = [entry["path"] for entry in json.loads(mapping_path.read_text())]
        ordered_fixtures = [
            next(item for item in fixtures if item["path"] == path) for path in mapped_paths
        ]
    else:
        ordered_fixtures = fixtures
    hlo_input_list = fixture_dir / "resnet50_inputs_hlo_order.txt"
    hlo_input_list.write_text(",".join(item["path"] for item in ordered_fixtures) + "\n")
    top = np.argsort(output.reshape(-1))[-5:][::-1]
    report = {
        "status": "pass",
        "model": "resnet50",
        "tensorflow_version": tf.__version__,
        "input_count": len(fixtures),
        "inputs_list": str(input_list),
        "hlo_inputs_list": str(hlo_input_list),
        "fixtures": fixtures,
        "hlo_ordered_fixtures": ordered_fixtures,
        "output": output_info,
        "output_sha256": output_info["sha256"],
        "top5": [
            {"index": int(i), "score": float(output.reshape(-1)[i])}
            for i in top
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def run_bert(args):
    fixture_dir = pathlib.Path(args.fixture_dir)
    out_dir = pathlib.Path(args.out_dir)
    seq = 16
    hidden = 768
    heads = 12
    head_dim = hidden // heads
    x = np.linspace(-0.1, 0.1, num=seq * hidden, dtype=np.float32).reshape(1, seq, hidden)
    qkv = np.full((hidden, hidden), 0.001, dtype=np.float32)
    out_w = np.full((hidden, hidden), 0.001, dtype=np.float32)

    @tf.function(jit_compile=True)
    def bert_base_fn(x_t, qkv_weight, out_weight):
        q = tf.linalg.matmul(x_t, qkv_weight)
        q = tf.reshape(q, [1, seq, heads, head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])
        scores = tf.linalg.matmul(q, q, transpose_b=True) * 0.125
        probs = tf.nn.softmax(scores, axis=-1)
        ctx = tf.linalg.matmul(probs, q)
        ctx = tf.transpose(ctx, [0, 2, 1, 3])
        ctx = tf.reshape(ctx, [1, seq, hidden])
        y = tf.linalg.matmul(ctx, out_weight)
        return tf.nn.gelu(y + x_t)

    output = bert_base_fn(x, qkv, out_w).numpy().astype(np.float32)
    fixtures = [
        write_f32(fixture_dir / "bert_base_param_000_x.raw.f32", x),
        write_f32(fixture_dir / "bert_base_param_001_qkv.raw.f32", qkv),
        write_f32(fixture_dir / "bert_base_param_002_out.raw.f32", out_w),
    ]
    output_info = write_f32(out_dir / "output.raw.f32", output)
    input_list = fixture_dir / "bert_base_inputs.txt"
    input_list.write_text(",".join(item["path"] for item in fixtures) + "\n")
    flat = output.reshape(-1)
    top = np.argsort(flat)[-5:][::-1]
    report = {
        "status": "pass",
        "model": "bert_base",
        "tensorflow_version": tf.__version__,
        "input_count": len(fixtures),
        "inputs_list": str(input_list),
        "fixtures": fixtures,
        "output": output_info,
        "output_sha256": output_info["sha256"],
        "topk_tokens": [{"index": int(i), "score": float(flat[i])} for i in top],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet50", "bert_base"], required=True)
    parser.add_argument("--image")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fixture-dir", required=True)
    args = parser.parse_args()
    try:
        if args.model == "resnet50":
            run_resnet(args)
        else:
            run_bert(args)
        return 0
    except Exception as exc:
        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "blocked",
            "model": args.model,
            "tensorflow_version": tf.__version__,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            },
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
