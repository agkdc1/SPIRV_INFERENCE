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
parser.add_argument("--elements", type=int, default=1024)
args = parser.parse_args()

hlo_path = pathlib.Path(args.hlo)
input0_path = pathlib.Path(args.input0)
input1_path = pathlib.Path(args.input1)
expected_path = pathlib.Path(args.expected)
report_path = pathlib.Path(args.report)

a = np.linspace(-4.0, 4.0, args.elements, dtype=np.float32)
b = np.linspace(9.0, -3.0, args.elements, dtype=np.float32)


@tf.function(jit_compile=True)
def add_fn(x, y):
    return x + y


x = tf.constant(a)
y = tf.constant(b)
expected = add_fn(x, y).numpy()

try:
    hlo = add_fn.experimental_get_compiler_ir(x, y)(stage="hlo")
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
