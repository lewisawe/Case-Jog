"""Embedding seam for CaseJog.

ONE swappable interface: embed(texts) -> list[vector].

Uses fastembed (ONNX, CPU, fully local) with BAAI/bge-small-en-v1.5.
The corpus never leaves the bench; every query is free and offline.
To swap the model, change EMBED_MODEL below (or the env var of the same name).

Fails LOUD if the model cannot load, rather than hanging or returning junk.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

# Local open embedding model. Swappable in one place (this constant / env var).
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# bge-small-en-v1.5 is 384-dim. Kept here so the store/index can assert it.
EMBED_DIM = int(os.environ.get("EMBED_DIM", "384"))


@lru_cache(maxsize=1)
def _model():
    """Load the fastembed model once. Fail loud on any error."""
    try:
        from fastembed import TextEmbedding
    except Exception as e:  # pragma: no cover - import guard
        sys.stderr.write(
            "FATAL: fastembed is not installed. Run `pip install -r requirements.txt`.\n"
            f"Underlying error: {e}\n"
        )
        raise SystemExit(2)

    try:
        return TextEmbedding(model_name=EMBED_MODEL)
    except Exception as e:
        sys.stderr.write(
            f"FATAL: could not load embedding model '{EMBED_MODEL}'.\n"
            "The model downloads to a local ONNX cache on first use; check disk/network "
            "for the first run, then it is fully offline.\n"
            f"Underlying error: {e}\n"
        )
        raise SystemExit(2)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts into vectors.

    Args:
        texts: list of strings to embed.
    Returns:
        list of float vectors, one per input text, each of length EMBED_DIM.
    """
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        return []
    model = _model()
    # fastembed returns a generator of numpy arrays.
    vectors = [v.tolist() for v in model.embed(texts)]
    if vectors and len(vectors[0]) != EMBED_DIM:
        sys.stderr.write(
            f"FATAL: model '{EMBED_MODEL}' produced dim {len(vectors[0])}, "
            f"expected {EMBED_DIM}. Update EMBED_DIM.\n"
        )
        raise SystemExit(2)
    return vectors


if __name__ == "__main__":
    v = embed(["drum wont spin"])
    print(len(v), len(v[0]))
