"""The dumb baseline: pure cosine over the same embeddings, no filter, no boost.

This is the thing the heavier Mongo/hybrid technique must BEAT in Phase 4.
It is deliberately INDEPENDENT of MongoDB so the comparison runs even if Mongo
misbehaves, and it doubles as the fully-offline fallback retriever.

It loads the corpus from data/corpus.jsonl and embeds every symptom once,
caching the matrix in memory.
"""
from __future__ import annotations

import json
import os

import numpy as np

from embed import embed

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")

_CORPUS: list[dict] | None = None
_MATRIX: np.ndarray | None = None


def load_corpus() -> list[dict]:
    global _CORPUS
    if _CORPUS is None:
        with open(CORPUS_PATH) as f:
            _CORPUS = [json.loads(line) for line in f if line.strip()]
    return _CORPUS


def _ensure_matrix() -> np.ndarray:
    global _MATRIX
    if _MATRIX is None:
        corpus = load_corpus()
        vecs = embed([c["symptom_text"] for c in corpus])
        m = np.array(vecs, dtype=np.float32)
        # normalize rows for cosine via dot product
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _MATRIX = m / norms
    return _MATRIX


def baseline_cosine(query: str, k: int = 5) -> list[dict]:
    """Pure cosine similarity ranking. No metadata filter, no exact boost."""
    corpus = load_corpus()
    matrix = _ensure_matrix()
    qv = np.array(embed([query])[0], dtype=np.float32)
    n = np.linalg.norm(qv)
    if n:
        qv = qv / n
    scores = matrix @ qv
    order = np.argsort(-scores)[:k]
    results = []
    for idx in order:
        c = dict(corpus[idx])
        c["_score"] = float(scores[idx])
        results.append(c)
    return results


if __name__ == "__main__":
    for r in baseline_cosine("drum stuck, water not draining", k=3):
        print(f"{r['_score']:.3f} {r['id']} {r['root_cause']}")
