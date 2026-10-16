"""CaseJog FastAPI app: symptom in -> ranked past cases out (grounded recall).

Serves static/index.html at / and exposes POST /api/search.
Output is always framed as the technician's own past cases, each citing its
source — a recall aid a human verifies, NEVER an autonomous diagnosis.

Retrieval path:
  - default: the filtered-hybrid Atlas $vectorSearch retriever (search.py);
  - offline fallback: the pure-cosine baseline (baseline.py) when MongoDB is
    unreachable, so the open/offline thesis holds even with the container down.

The fallback is scoped to MongoDB/connection failures. A genuine programming
error is NOT swallowed as "fallback" — it surfaces as a 500 — so the demo's
engine label stays honest.
"""
from __future__ import annotations

import os
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pymongo.errors import PyMongoError

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="CaseJog", description="Local semantic recall for a repair tech")


class SearchRequest(BaseModel):
    symptom: str
    appliance_type: str | None = None
    brand: str | None = None
    k: int = 5


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/search")
def api_search(req: SearchRequest):
    filters = {}
    if req.appliance_type:
        filters["appliance_type"] = req.appliance_type
    if req.brand:
        filters["brand"] = req.brand

    t0 = time.time()
    engine = "filtered-hybrid"
    note = None
    try:
        from search import search_filtered
        results = search_filtered(req.symptom, filters or None, k=req.k)
    except PyMongoError as e:
        # Expected offline path: Mongo is unreachable. Fall back to the pure
        # cosine baseline (no metadata filter) so the tool still works offline.
        from baseline import baseline_cosine
        engine = "baseline-cosine (fallback)"
        note = f"MongoDB unavailable ({type(e).__name__}); using offline cosine fallback (no appliance filter)."
        results = baseline_cosine(req.symptom, k=req.k)
    latency_ms = round((time.time() - t0) * 1000, 1)

    return {
        "framing": "Your past cases, ranked — a recall aid, not a diagnosis.",
        "engine": engine,
        "note": note,
        "latency_ms": latency_ms,
        "count": len(results),
        "results": results,
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
