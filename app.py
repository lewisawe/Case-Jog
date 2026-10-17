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

The public demo is hardened so an open, unauthenticated endpoint cannot be
turned into a compute sink: query inputs are clamped (symptom length, k) and a
dependency-free per-IP sliding-window rate limit caps request volume.
"""
from __future__ import annotations

import os
import time
import threading
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Rate-limit config (env-overridable). Sliding window, per client IP.
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "20"))      # requests...
RATE_LIMIT_WINDOW = float(os.environ.get("RATE_LIMIT_WINDOW", "60"))  # ...per N seconds

app = FastAPI(title="CaseJog", description="Local semantic recall for a repair tech")

# In-process hit log per IP. Fine for a single-instance demo; resets on restart.
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    # Render terminates TLS at a proxy; trust the first X-Forwarded-For hop.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    # Only guard the expensive endpoint; static assets and the page are free.
    if request.url.path == "/api/search":
        ip = _client_ip(request)
        now = time.time()
        with _hits_lock:
            dq = _hits[ip]
            cutoff = now - RATE_LIMIT_WINDOW
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= RATE_LIMIT_MAX:
                retry = round(dq[0] + RATE_LIMIT_WINDOW - now, 1)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limited",
                        "detail": (
                            f"Too many requests. Limit is {RATE_LIMIT_MAX} per "
                            f"{int(RATE_LIMIT_WINDOW)}s. Retry in ~{retry}s."
                        ),
                    },
                    headers={"Retry-After": str(int(retry) + 1)},
                )
            dq.append(now)
    return await call_next(request)


class SearchRequest(BaseModel):
    # Clamp inputs so a single request cannot be made arbitrarily expensive.
    symptom: str = Field(min_length=1, max_length=500)
    appliance_type: str | None = Field(default=None, max_length=60)
    brand: str | None = Field(default=None, max_length=60)
    k: int = Field(default=5, ge=1, le=10)


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
