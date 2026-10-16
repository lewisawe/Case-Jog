"""Phase 4 honesty centerpiece: precision@k, filtered-hybrid vs baseline.

For each held-out symptom (whose true root_cause is known), retrieve top-k with
BOTH search_filtered (Mongo Atlas filtered-hybrid) and baseline_cosine (pure
cosine), and score precision@k = fraction of top-k whose root_cause matches the
held-out true cause.

Prints a side-by-side table (k, filtered, baseline). Run ONCE; the number is
the point. If Mongo is unavailable, the filtered column is marked N/A and the
baseline still runs (so the comparison degrades honestly rather than crashing).
"""
from __future__ import annotations

import json
import os

from baseline import baseline_cosine

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HELDOUT_PATH = os.path.join(DATA_DIR, "heldout.jsonl")

KS = [1, 3, 5]


def load_heldout() -> list[dict]:
    with open(HELDOUT_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def _precision_at_k(results: list[dict], true_cause: str, k: int) -> float:
    topk = results[:k]
    if not topk:
        return 0.0
    hits = sum(1 for r in topk if r.get("root_cause") == true_cause)
    return hits / len(topk)


def run() -> dict:
    heldout = load_heldout()
    n = len(heldout)

    # Decide once whether the filtered (Mongo) path is usable.
    filtered_ok = True
    try:
        from search import search_filtered
        # smoke call on first held-out item, filtering on its appliance_type
        _ = search_filtered(heldout[0]["symptom_text"],
                            {"appliance_type": heldout[0]["appliance_type"]}, k=1)
    except Exception as e:
        filtered_ok = False
        filtered_err = str(e)

    agg = {k: {"filtered": 0.0, "baseline": 0.0} for k in KS}

    for item in heldout:
        sym = item["symptom_text"]
        true_cause = item["root_cause"]
        base_res = baseline_cosine(sym, k=max(KS))
        if filtered_ok:
            filt_res = search_filtered(sym, {"appliance_type": item["appliance_type"]}, k=max(KS))
        for k in KS:
            agg[k]["baseline"] += _precision_at_k(base_res, true_cause, k)
            if filtered_ok:
                agg[k]["filtered"] += _precision_at_k(filt_res, true_cause, k)

    for k in KS:
        agg[k]["baseline"] /= n
        if filtered_ok:
            agg[k]["filtered"] /= n

    # --- print table ---
    print(f"\nprecision@k over held-out symptoms (n={n}), run once\n")
    print(f"{'k':>3}  {'filtered-hybrid':>16}  {'baseline-cosine':>16}")
    print(f"{'-'*3}  {'-'*16}  {'-'*16}")
    for k in KS:
        f = f"{agg[k]['filtered']:.3f}" if filtered_ok else "N/A"
        b = f"{agg[k]['baseline']:.3f}"
        print(f"{k:>3}  {f:>16}  {b:>16}")
    if not filtered_ok:
        print(f"\n[filtered path UNAVAILABLE: {filtered_err}]")
        print("baseline ran offline, independent of Mongo.")
    print()

    return {"n": n, "filtered_ok": filtered_ok, "agg": agg,
            "filtered_err": None if filtered_ok else filtered_err}


if __name__ == "__main__":
    run()
