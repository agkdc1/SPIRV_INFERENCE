#!/usr/bin/env python3
"""Diagnose Whisper decoder SPIR-V failure.
Run one decoder step with the EXACT initial fixtures (from decoder_inputs.txt),
inspect the raw logits, and compare with TF numpy reference."""
import json, os, pathlib, subprocess, sys, tempfile
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

BASE = pathlib.Path("<repo-root>/src/cuda-spirv")
FIXTURE = BASE / "xla_hlo" / "fixtures"
OUT = BASE / "xla_hlo" / "out"
BIN = str(BASE / "target" / "release" / "xla_hlo")
DECODER_HLO = str(FIXTURE / "whisper_decoder.stablehlo.txt")
KERNEL_DIR = str(BASE / "kernels")

# Read the original decoder inputs list
inputs_txt = (FIXTURE / "whisper_decoder_inputs.txt").read_text().strip()
input_paths = inputs_txt.split(",")
print(f"Decoder input count: {len(input_paths)}")

# Verify all input files exist and print shapes
for i, p in enumerate(input_paths):
    full = BASE / p if not os.path.isabs(p) else pathlib.Path(p)
    if not full.exists():
        print(f"  MISSING input {i}: {p}")
        sys.exit(1)
    sz = full.stat().st_size
    elems = sz // 4
    print(f"  param {i:3d}: {elems:>10d} elems  ({sz:>10d} bytes)  {full.name}")

# Step 1: Run xla_hlo with EXACT initial fixtures
print("\n=== Running xla_hlo decoder with initial fixtures ===")
scratch = pathlib.Path(tempfile.gettempdir())
output_path = str(scratch / "whisper_decoder_diag_output.raw.f32")
report_path = str(scratch / "whisper_decoder_diag_report.json")

cmd = [
    BIN,
    "--input", DECODER_HLO,
    "--run",
    "--kernel-dir", KERNEL_DIR,
    "--device", "any",
    "--inputs-f32", ",".join(str(BASE / p) if not os.path.isabs(p) else p for p in input_paths),
    "--output-f32", output_path,
    "--report", report_path,
]
print(f"Command length: {len(cmd)} args, inputs-f32 length: {len(cmd[9])}")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
print(f"Return code: {result.returncode}")
if result.stderr:
    print(f"STDERR (first 2000): {result.stderr[:2000]}")
if result.stdout:
    print(f"STDOUT (first 2000): {result.stdout[:2000]}")

if result.returncode != 0:
    print("FAILED - xla_hlo returned non-zero")
    sys.exit(1)

# Read report
with open(report_path) as f:
    report = json.load(f)
print(f"\nReport status: {report.get('status')}")
exec_info = report.get("execution", {})
print(f"Device: {exec_info.get('device')}")
print(f"Dispatch count: {exec_info.get('dispatch_count')}")

# Read output logits
logits = np.fromfile(output_path, dtype=np.float32)
expected_elems = 1 * 64 * 51865
print(f"\nOutput elements: {logits.size} (expected {expected_elems})")
if logits.size != expected_elems:
    print(f"SIZE MISMATCH!")
    sys.exit(1)

logits = logits.reshape((1, 64, 51865))
print(f"Logits shape: {logits.shape}")
print(f"Logits range: min={logits.min():.6f}, max={logits.max():.6f}, mean={logits.mean():.6f}")
print(f"Logits std: {logits.std():.6f}")
print(f"All zeros? {np.allclose(logits, 0)}")
print(f"Any NaN? {np.any(np.isnan(logits))}")
print(f"Any Inf? {np.any(np.isinf(logits))}")

# Check argmax at positions 0-7
print("\nSPIR-V logits per position:")
for pos in range(8):
    row = logits[0, pos, :]
    top5_idx = np.argsort(row)[-5:][::-1]
    top5_val = row[top5_idx]
    print(f"  pos {pos}: argmax={np.argmax(row):5d}, top5_idx={top5_idx.tolist()}, top5_val={[f'{v:.4f}' for v in top5_val]}")

# Step 2: Run TF numpy reference for the same inputs
print("\n=== Running TF numpy reference decoder (single step) ===")

def read_f32(path, shape=None):
    d = np.fromfile(str(path), dtype=np.float32)
    if shape:
        d = d.reshape(shape)
    return d

# Load the exact same inputs
encoder_output = read_f32(FIXTURE / "whisper_dec_param_000_encoder_output.raw.f32", (1,1500,384))
token_emb = read_f32(FIXTURE / "whisper_dec_param_001_token_emb.raw.f32", (1,64,384))
pos_emb = read_f32(FIXTURE / "whisper_dec_param_002_pos_emb.raw.f32", (1,64,384))
causal_mask = read_f32(FIXTURE / "whisper_dec_param_003_causal_mask.raw.f32", (1,1,64,64))

print(f"encoder_output: range [{encoder_output.min():.4f}, {encoder_output.max():.4f}]")
print(f"token_emb: range [{token_emb.min():.4f}, {token_emb.max():.4f}]")
print(f"pos_emb: range [{pos_emb.min():.4f}, {pos_emb.max():.4f}]")
print(f"causal_mask: range [{causal_mask.min():.4f}, {causal_mask.max():.4f}]")
print(f"causal_mask nonzero: {np.count_nonzero(causal_mask)}/{causal_mask.size}")

# Load all decoder weights
weight_paths = input_paths[4:]
weights = []
for p in weight_paths:
    full = BASE / p if not os.path.isabs(p) else pathlib.Path(p)
    w = np.fromfile(str(full), dtype=np.float32)
    weights.append(w)
print(f"Loaded {len(weights)} weight tensors")

# Numpy decoder functions
def layer_norm_np(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta

def gelu_np(x):
    c = 0.7978845608
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))

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

# Build shaped weights
n_text_state = 384
n_text_layer = 4
n_vocab = 51865
wi = 0
dec_w = {}
for i in range(n_text_layer):
    p = f"dec{i}"
    dec_w[f"{p}_ln1_gamma"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_ln1_beta"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_sq_w"] = weights[wi].reshape(n_text_state, n_text_state); wi += 1
    dec_w[f"{p}_sq_b"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_sk_w"] = weights[wi].reshape(n_text_state, n_text_state); wi += 1
    dec_w[f"{p}_sk_b"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_sv_w"] = weights[wi].reshape(n_text_state, n_text_state); wi += 1
    dec_w[f"{p}_sv_b"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_sout_w"] = weights[wi].reshape(n_text_state, n_text_state); wi += 1
    dec_w[f"{p}_sout_b"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_ln2_gamma"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_ln2_beta"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_cq_w"] = weights[wi].reshape(n_text_state, n_text_state); wi += 1
    dec_w[f"{p}_cq_b"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_ck_w"] = weights[wi].reshape(n_text_state, n_text_state); wi += 1
    dec_w[f"{p}_ck_b"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_cv_w"] = weights[wi].reshape(n_text_state, n_text_state); wi += 1
    dec_w[f"{p}_cv_b"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_cout_w"] = weights[wi].reshape(n_text_state, n_text_state); wi += 1
    dec_w[f"{p}_cout_b"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_ln3_gamma"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_ln3_beta"] = weights[wi].reshape(n_text_state); wi += 1
    dec_w[f"{p}_fc1_w"] = weights[wi].reshape(n_text_state, n_text_state * 4); wi += 1
    dec_w[f"{p}_fc1_b"] = weights[wi].reshape(n_text_state * 4); wi += 1
    dec_w[f"{p}_fc2_w"] = weights[wi].reshape(n_text_state * 4, n_text_state); wi += 1
    dec_w[f"{p}_fc2_b"] = weights[wi].reshape(n_text_state); wi += 1
dec_w["ln_f_gamma"] = weights[wi].reshape(n_text_state); wi += 1
dec_w["ln_f_beta"] = weights[wi].reshape(n_text_state); wi += 1
dec_w["proj_w"] = weights[wi].reshape(n_text_state, n_vocab); wi += 1
print(f"Consumed {wi} weight tensors (expected {len(weights)})")

# Run decoder
x = token_emb + pos_emb
for i in range(n_text_layer):
    p = f"dec{i}"
    nx = layer_norm_np(x, dec_w[f"{p}_ln1_gamma"], dec_w[f"{p}_ln1_beta"])
    sa = masked_self_attn_np(nx, dec_w[f"{p}_sq_w"], dec_w[f"{p}_sq_b"],
                              dec_w[f"{p}_sk_w"], dec_w[f"{p}_sk_b"],
                              dec_w[f"{p}_sv_w"], dec_w[f"{p}_sv_b"],
                              dec_w[f"{p}_sout_w"], dec_w[f"{p}_sout_b"], causal_mask)
    x = x + sa
    nx = layer_norm_np(x, dec_w[f"{p}_ln2_gamma"], dec_w[f"{p}_ln2_beta"])
    ca = cross_attn_np(nx, encoder_output, dec_w[f"{p}_cq_w"], dec_w[f"{p}_cq_b"],
                        dec_w[f"{p}_ck_w"], dec_w[f"{p}_ck_b"],
                        dec_w[f"{p}_cv_w"], dec_w[f"{p}_cv_b"],
                        dec_w[f"{p}_cout_w"], dec_w[f"{p}_cout_b"])
    x = x + ca
    nx = layer_norm_np(x, dec_w[f"{p}_ln3_gamma"], dec_w[f"{p}_ln3_beta"])
    h = nx @ dec_w[f"{p}_fc1_w"] + dec_w[f"{p}_fc1_b"]
    h = gelu_np(h)
    h = h @ dec_w[f"{p}_fc2_w"] + dec_w[f"{p}_fc2_b"]
    x = x + h

x = layer_norm_np(x, dec_w["ln_f_gamma"], dec_w["ln_f_beta"])
ref_logits = x @ dec_w["proj_w"]

print(f"\nTF numpy ref logits shape: {ref_logits.shape}")
print(f"TF numpy ref logits range: min={ref_logits.min():.6f}, max={ref_logits.max():.6f}")
print("\nTF numpy ref logits per position:")
for pos in range(8):
    row = ref_logits[0, pos, :]
    top5_idx = np.argsort(row)[-5:][::-1]
    top5_val = row[top5_idx]
    print(f"  pos {pos}: argmax={np.argmax(row):5d}, top5_idx={top5_idx.tolist()}, top5_val={[f'{v:.4f}' for v in top5_val]}")

# Compare
print("\n=== Comparison ===")
diff = np.abs(logits - ref_logits)
print(f"Max abs diff: {diff.max():.6f}")
print(f"Mean abs diff: {diff.mean():.6f}")
for pos in range(8):
    row_diff = diff[0, pos, :]
    print(f"  pos {pos}: max_diff={row_diff.max():.6f}, mean_diff={row_diff.mean():.6f}")

print("\nDone.")
