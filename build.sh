#!/usr/bin/env bash
# Render build step for CaseJog.
# Installs dependencies and pre-downloads the local embedding model so the
# first live query does not pay the model-load cost at request time.
set -euo pipefail

pip install --upgrade pip
pip install -r requirements.txt

# Warm the fastembed ONNX model into the build cache (offline-safe at runtime).
python - <<'PY'
from embed import embed
v = embed(["warmup"])
print(f"embed warmup ok: dim={len(v[0])}")
PY
