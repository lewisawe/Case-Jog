"""Phase 3 ingest: embed each symptom_text and load into MongoDB with indexes.

Usage: python ingest.py
"""
from __future__ import annotations

import json
import os

from embed import embed
from store import upsert_cases, ensure_vector_index, ping

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")


def main() -> None:
    ping()
    with open(CORPUS_PATH) as f:
        cases = [json.loads(line) for line in f if line.strip()]
    print(f"embedding {len(cases)} symptom_texts ...")
    vectors = embed([c["symptom_text"] for c in cases])
    count = upsert_cases(cases, vectors)
    print(f"collection count = {count} (corpus size = {len(cases)})")
    status = ensure_vector_index()
    print(f"vector index status = {status}")


if __name__ == "__main__":
    main()
