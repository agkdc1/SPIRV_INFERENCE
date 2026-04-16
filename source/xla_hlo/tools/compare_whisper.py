#!/usr/bin/env python3
"""Compare SPIR-V whisper output with Python Whisper reference."""
import json
import sys
import os

# Load SPIR-V local report
spirv_report_path = "xla_hlo/reports/whisper_spirv_local_report.json"
tf_report_path = "xla_hlo/out/whisper_tf_report.json"

with open(spirv_report_path) as f:
    spirv = json.load(f)
with open(tf_report_path) as f:
    tf_ref = json.load(f)

print("=== WHISPER SPIR-V vs TF REFERENCE ===")
print()
print(f"TF Reference tokens:   {tf_ref['decoder_tokens']}")
print(f"SPIR-V RTX 3060 tokens: {spirv['tokens']}")
print()
print(f"TF Reference text:   '{tf_ref['decoder_transcription']}'")
print(f"SPIR-V RTX 3060 text: '{spirv['transcription']}'")
print()

tf_tokens = tf_ref['decoder_tokens']
spirv_tokens = spirv['tokens']
match = tf_tokens == spirv_tokens
print(f"Token-for-token match: {match}")

if match:
    print("\nVERDICT: PASS - SPIR-V decoder produces identical tokens to TF reference")
else:
    print(f"\nToken diff: TF has {len(tf_tokens)} tokens, SPIR-V has {len(spirv_tokens)}")
    for i in range(max(len(tf_tokens), len(spirv_tokens))):
        t1 = tf_tokens[i] if i < len(tf_tokens) else "N/A"
        t2 = spirv_tokens[i] if i < len(spirv_tokens) else "N/A"
        marker = " <-- MISMATCH" if t1 != t2 else ""
        print(f"  pos {i}: TF={t1} SPIR-V={t2}{marker}")

# Print encoder parity from encoder report
enc_report_path = "xla_hlo/reports/whisper_encoder_local_report.json"
if os.path.exists(enc_report_path):
    # Read just enough to find parity (file is huge)
    import re
    with open(enc_report_path) as f:
        content = f.read()
    # Find parity section
    parity_match = re.search(r'"parity":\s*\{[^}]+\}', content)
    if parity_match:
        print(f"\nEncoder parity: {parity_match.group()}")
