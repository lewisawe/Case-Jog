"""MongoDB-backed store for CaseJog, behind a swappable seam.

Targets MongoDB Atlas (local via `mongodb/mongodb-atlas-local`, or a real Atlas
cluster via MONGODB_URI). Creates:
  - an Atlas `$vectorSearch` index over the `embedding` field
  - metadata indexes on appliance_type / brand / error_code

Retrieval lives behind `vector_search(...)` so the store is swappable; the
pure-cosine baseline in baseline.py does NOT depend on this module, so the
Phase 4 comparison runs even if Mongo misbehaves.
"""
from __future__ import annotations

import os
import time

from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from embed import EMBED_DIM

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/?directConnection=true")
DB_NAME = os.environ.get("CASEJOG_DB", "casejog")
COLL_NAME = os.environ.get("CASEJOG_COLL", "cases")
VECTOR_INDEX = "casejog_vector_index"


def get_client() -> MongoClient:
    return MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)


def get_collection():
    return get_client()[DB_NAME][COLL_NAME]


def ping() -> bool:
    get_client().admin.command("ping")
    return True


def upsert_cases(cases: list[dict], vectors: list[list[float]]) -> int:
    """Insert cases with their embedding vectors. Replaces the collection."""
    coll = get_collection()
    coll.drop()
    docs = []
    for c, v in zip(cases, vectors):
        d = dict(c)
        d["_id"] = c["id"]
        d["embedding"] = v
        docs.append(d)
    if docs:
        coll.insert_many(docs)
    # metadata indexes for filtered retrieval
    coll.create_index("appliance_type")
    coll.create_index("brand")
    coll.create_index("error_code")
    return coll.count_documents({})


def ensure_vector_index() -> str:
    """Create the Atlas vector-search index if absent; wait until queryable.

    Returns a status string: 'READY' or 'UNAVAILABLE: <reason>'.
    """
    coll = get_collection()
    try:
        existing = {ix["name"] for ix in coll.list_search_indexes()}
    except Exception as e:
        return f"UNAVAILABLE: list_search_indexes failed ({e})"

    if VECTOR_INDEX not in existing:
        model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": EMBED_DIM,
                        "similarity": "cosine",
                    },
                    {"type": "filter", "path": "appliance_type"},
                    {"type": "filter", "path": "brand"},
                    {"type": "filter", "path": "error_code"},
                ]
            },
            name=VECTOR_INDEX,
            type="vectorSearch",
        )
        try:
            coll.create_search_index(model=model)
        except Exception as e:
            return f"UNAVAILABLE: create_search_index failed ({e})"

    # Poll until the index reports queryable.
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            for ix in coll.list_search_indexes():
                if ix["name"] == VECTOR_INDEX:
                    if ix.get("queryable"):
                        return "READY"
                    status = ix.get("status", "PENDING")
                    break
        except Exception as e:
            return f"UNAVAILABLE: poll failed ({e})"
        time.sleep(3)
    return "UNAVAILABLE: index did not become queryable within timeout"


def vector_search(query_vec: list[float], filters: dict | None, k: int,
                  num_candidates: int | None = None) -> list[dict]:
    """Atlas $vectorSearch with an optional metadata filter fused into the stage.

    Returns the matched docs with a `_score` field. Raises on Mongo errors so
    the caller can fall back.
    """
    coll = get_collection()
    stage = {
        "$vectorSearch": {
            "index": VECTOR_INDEX,
            "path": "embedding",
            "queryVector": query_vec,
            "numCandidates": num_candidates or max(100, k * 20),
            "limit": k,
        }
    }
    if filters:
        # only include filters with a truthy value
        clean = {kk: vv for kk, vv in filters.items() if vv}
        if clean:
            if len(clean) == 1:
                kk, vv = next(iter(clean.items()))
                stage["$vectorSearch"]["filter"] = {kk: {"$eq": vv}}
            else:
                stage["$vectorSearch"]["filter"] = {
                    "$and": [{kk: {"$eq": vv}} for kk, vv in clean.items()]
                }
    pipeline = [
        stage,
        {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"embedding": 0}},
    ]
    return list(coll.aggregate(pipeline))
