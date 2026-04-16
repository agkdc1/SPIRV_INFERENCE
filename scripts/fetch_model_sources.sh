#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL="$ROOT/external"
mkdir -p "$EXTERNAL"

clone_or_update() {
  local url="$1"
  local dir="$2"
  if [ -d "$dir/.git" ]; then
    git -C "$dir" fetch --depth 1 origin
  else
    git clone --depth 1 "$url" "$dir"
  fi
}

clone_or_update https://github.com/tensorflow/tensorflow.git "$EXTERNAL/tensorflow"
clone_or_update https://github.com/openai/whisper.git "$EXTERNAL/whisper"

cat > "$EXTERNAL/README.md" <<'EOF'
# External Model Sources

This directory is populated by `scripts/fetch_model_sources.sh`.

The repository intentionally does not vendor TensorFlow, Whisper, downloaded
model weights, `.tflite` model files, or raw tensor dumps. The public source
tree includes the compiler/runtime code, kernel sources, HLO fixtures, reports,
and deterministic small fixtures that are part of the public evidence package.
EOF

printf 'External source checkouts are under %s\n' "$EXTERNAL"
