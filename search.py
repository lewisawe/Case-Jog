"""The heavy candidate retriever: filtered, hybrid, grounded.

search_filtered(query, filters, k):
  1. embed the query (local, offline),
  2. run Atlas $vectorSearch with the metadata filter FUSED into the stage
     (filter by appliance_type/brand while ranking by similarity, not a
     post-filter),
  3. hybrid RE-RANK: vector similarity + an exact error_code boost + a mild
     recency boost, so the right case surfaces even when phrasing differs,
  4. each result CITES its source case (id, date, price) — grounded recall,
     never a verdict.

The warrant for the heavy path is (filter + exact-signal + grounding), not
corpus scale. Phase 4 (eval.py) decides whether it is load-bearing.
"""
from __future__ import annotations

import datetime as _dt
import re
from functools import lru_cache

from embed import embed
from store import vector_search, get_collection

# Weighting for the hybrid re-rank. Kept small and explicit.
ERROR_CODE_BOOST = 0.15      # exact error-code match is a strong signal
RECENCY_MAX_BOOST = 0.05     # most-recent case gets up to this much
RECENCY_HALFLIFE_DAYS = 365


@lru_cache(maxsize=1)
def _known_codes() -> tuple[str, ...]:
    """The exact error codes present in the corpus, longest-first.

    Matching against the real code vocabulary (rather than a loose regex that
    also catches ordinary uppercased words like DRUM/MICROWAVE) keeps the
    exact-signal boost precise.
    """
    try:
        codes = {c for c in get_collection().distinct("error_code") if c}
    except Exception:
        return tuple()
    return tuple(sorted(codes, key=len, reverse=True))


def _extract_error_code(query: str) -> str | None:
    """Return a known error code if one appears in the query, else None.

    Matches whole tokens case-insensitively; also tolerates a space-separated
    form (e.g. 'OF OF' for the 'OF-OF' code).
    """
    q = query.upper()
    for code in _known_codes():
        cu = code.upper()
        # whole-token match for the code or its space-separated variant
        pat = r"(?<![A-Z0-9])" + re.escape(cu) + r"(?![A-Z0-9])"
        if re.search(pat, q):
            return cu
        if "-" in cu:
            pat2 = r"(?<![A-Z0-9])" + re.escape(cu.replace("-", " ")) + r"(?![A-Z0-9])"
            if re.search(pat2, q):
                return cu
    return None


def _recency_boost(date_str: str) -> float:
    try:
        d = _dt.date.fromisoformat(date_str)
    except Exception:
        return 0.0
    age_days = (_dt.date.today() - d).days
    if age_days < 0:
        age_days = 0
    # linear decay to zero at RECENCY_HALFLIFE_DAYS, floored at 0
    frac = max(0.0, 1.0 - age_days / RECENCY_HALFLIFE_DAYS)
    return RECENCY_MAX_BOOST * frac


def search_filtered(query: str, filters: dict | None = None, k: int = 5) -> list[dict]:
    """Filtered hybrid vector search with grounded citations.

    Returns a list of result dicts, each carrying cause/parts/time/price, a
    hybrid score, and a `citation` (id, date, price) — grounded recall.
    """
    qv = embed([query])[0]
    query_code = _extract_error_code(query)

    # Pull more candidates than k so the re-rank can reorder meaningfully.
    candidates = vector_search(qv, filters, k=max(k * 4, 20))

    reranked = []
    for c in candidates:
        base = float(c.get("_score", 0.0))
        boost = 0.0
        if query_code and c.get("error_code", "").upper() == query_code:
            boost += ERROR_CODE_BOOST
        boost += _recency_boost(c.get("date", ""))
        final = base + boost
        reranked.append((final, base, boost, c))

    reranked.sort(key=lambda t: t[0], reverse=True)

    out = []
    for final, base, boost, c in reranked[:k]:
        out.append({
            "id": c["id"],
            "appliance_type": c.get("appliance_type"),
            "brand": c.get("brand"),
            "model": c.get("model"),
            "error_code": c.get("error_code"),
            "symptom_text": c.get("symptom_text"),
            "root_cause": c.get("root_cause"),
            "parts": c.get("parts", []),
            "labor_minutes": c.get("labor_minutes"),
            "price": c.get("price"),
            "date": c.get("date"),
            "score": round(final, 4),
            "vector_score": round(base, 4),
            "boost": round(boost, 4),
            # grounded recall: every surfaced case cites its source
            "citation": {"id": c["id"], "date": c.get("date"), "price": c.get("price")},
        })
    return out


if __name__ == "__main__":
    import json
    res = search_filtered("drum stuck, water not draining, E18", {"appliance_type": "washing machine"}, k=3)
    print(json.dumps(res, indent=2))
