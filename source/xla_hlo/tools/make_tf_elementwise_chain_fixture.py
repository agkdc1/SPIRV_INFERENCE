import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf


def write_f32(path: pathlib.Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values.astype("<f4").tofile(path)


parser = argparse.ArgumentParser()
parser.add_argument("--hlo", required=True)
parser.add_argument("--input0", required=True)
parser.add_argument("--input1", required=True)
parser.add_argument("--expected", required=True)
parser.add_argument("--report", required=True)
parser.add_argument("--elements", type=int, default=2048)
args = parser.parse_args()

hlo_path = pathlib.Path(args.hlo)
input0_path = pathlib.Path(args.input0)
input1_path = pathlib.Path(args.input1)
expected_path = pathlib.Path(args.expected)
report_path = pathlib.Path(args.report)

a = np.linspace(-0.75, 0.85, args.elements, dtype=np.float32)
b = np.linspace(0.20, 1.20, args.elements, dtype=np.float32)


@tf.function(jit_compile=True)
def chain_fn(x, y):
    add = x + y
    sub = add - y
    mul = sub * tf.constant(0.25, dtype=tf.float32)
    div = mul / (tf.abs(y) + tf.constant(1.25, dtype=tf.float32))
    hi = tf.maximum(div, tf.constant(-0.5, dtype=tf.float32))
    lo = tf.minimum(hi, tf.constant(0.5, dtype=tf.float32))
    mag = tf.abs(lo)
    ex = tf.exp(mag * tf.constant(0.1, dtype=tf.float32))
    lg = tf.math.log(ex + tf.constant(1.0, dtype=tf.float32))
    root = tf.sqrt(lg + tf.constant(1.0, dtype=tf.float32))
    inv = tf.math.rsqrt(root + tf.constant(1.0, dtype=tf.float32))
    neg = -inv
    th = tf.tanh(neg)
    return tf.math.sigmoid(th)


x = tf.constant(a)
y = tf.constant(b)
expected = chain_fn(x, y).numpy()

try:
    hlo = chain_fn.experimental_get_compiler_ir(x, y)(stage="hlo")
    export_status = "pass"
except Exception as exc:
    hlo = f"EXPORT_BLOCKED: {exc!r}\n"
    export_status = "blocked"

hlo_path.parent.mkdir(parents=True, exist_ok=True)
hlo_path.write_text(str(hlo))
write_f32(input0_path, a)
write_f32(input1_path, b)
write_f32(expected_path, expected)
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(
    json.dumps(
        {
            "status": export_status,
            "tensorflow_version": tf.__version__,
            "elements": args.elements,
            "hlo": str(hlo_path),
            "input0": str(input0_path),
            "input1": str(input1_path),
            "expected": str(expected_path),
        },
        indent=2,
    )
)
print(report_path.read_text())
