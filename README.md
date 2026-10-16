# CaseJog

A 100% local semantic memory for an independent appliance / small-electronics
repair technician. Type a new symptom and instantly get a ranked list of **your
own past jobs** that match — each with root cause, parts, time, and price — so
you recall a fix you have seen before instead of re-diagnosing from scratch.

Output is always framed as *your past cases, ranked* — a recall aid a human
verifies, never an autonomous diagnosis. Every result cites its source case
(id, date, price): grounded recall.

## Why local / open (the open-vs-closed thesis)

The private case corpus is the technician's hard-won edge and **never leaves the
bench**. Local open embeddings (`fastembed`, ONNX, CPU) make every query
**$0, offline, and private**. The embedding model is swappable in **one place**
(`embed.py` / the `EMBED_MODEL` env var) — the open-weight model is the mechanic,
and it is replaceable without touching the rest of the pipeline. A cloud RAG
would upload the corpus and bill per query; CaseJog uploads nothing and bills
nothing. If the Mongo container is down, the app **falls back to a pure in-memory
cosine retriever** and still answers — the open/offline promise holds literally.

> Note: the spec originally named Ollama for embeddings. This build uses
> `fastembed` instead (per build-time steering) — a local ONNX CPU model, same
> open/offline/swappable thesis, no separate model server to run for the core
> retrieval path.

## Architecture

```
symptom ──► embed.py (fastembed, local) ──► query vector
                                              │
          ┌───────────────────────────────────┤
          ▼                                   ▼
  baseline.py                           search.py
  pure cosine (no filter, no boost)     Atlas $vectorSearch (store.py)
  the dumb baseline + offline fallback  + metadata filter fused into the stage
                                        + hybrid re-rank (exact error_code + recency)
                                        + grounded citation per result (id/date/price)
          └───────────────┬───────────────────┘
                          ▼
                       eval.py  precision@k, filtered vs baseline (run once)
                          ▼
                       app.py / static/index.html  (ranked cards + latency)
```

The embeddings + Atlas **surface candidates**; **code owns the ranking and never
fabricates a fix.** No generative model is in the answer path — a surfaced card
is always a real past job, cited.

## Setup (on the build instance)

```bash
cd ~/build/casejog
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # defaults point at the local Atlas-local container
```

Start MongoDB (Atlas-local image — supports `$vectorSearch` locally, no cloud
account needed):

```bash
docker run -d --name casejog-mongo -p 27017:27017 mongodb/mongodb-atlas-local:latest
# if the container already exists but is stopped:  docker start casejog-mongo
```

## Build / run

```bash
python gen_corpus.py          # Phase 2: write data/corpus.jsonl + data/heldout.jsonl
python ingest.py              # Phase 3: embed + load into Mongo, create vector index
python eval.py                # Phase 5: precision@k, filtered-hybrid vs baseline (run once)
uvicorn app:app --host 127.0.0.1 --port 8000   # serve the UI at /
```

Then open `http://127.0.0.1:8000/`, type a varied-phrasing symptom, optionally
pick an appliance/brand filter, and read the ranked cards. Each card shows root
cause, parts, labor minutes, price, match score, and its **source citation**.
The header carries the framing line: *"Your past cases, ranked — a recall aid,
not a diagnosis."*

## The selection gate (Phase 4) — ADOPT, with an honest caveat

CaseJog implements the dumb baseline (pure cosine over the same local
embeddings) **first**, then adopts the heavier filtered-hybrid Atlas path **only
because it demonstrably beats the baseline** on a phrasing-disjoint held-out set.
Full evidence (table + the two hero queries) lives in
`data/phase4-comparison.md`.

### precision@k (held-out n=28, run once, never tuned)

```
  k   filtered-hybrid   baseline-cosine
---  ----------------  ----------------
  1             0.893             0.857
  3             0.869             0.738
  5             0.814             0.657
```

Filtered-hybrid wins at every k; the gap widens at k=3/k=5 as the baseline
drifts to near-but-wrong causes. `n=28` is small — treat the margins as
suggestive, not decisive.

### The wrong-appliance hero query (what the baseline cannot produce)

Query: `door wont lock so it wont start, clicking at the door`

Pure cosine, with no idea which appliance is on the bench, returns **5 wrong
washing-machine door-lock cases**. The filtered path (`appliance_type=microwave`)
returns the **correct 5 microwave door-switch cases**, each citing its source.
The metadata filter fused into `$vectorSearch` is the load-bearing mechanism —
**filter + exact-signal + grounded citations, not corpus scale** (the same
warrant the one MongoDB Atlas weekend winner actually won on).

## Offline fallback (open/offline, demonstrated)

`app.py` runs the filtered-hybrid retriever by default. If MongoDB is
unreachable (a `pymongo` connection error), it falls back to `baseline_cosine`
(pure in-memory cosine, no metadata filter) and labels the response
`engine: baseline-cosine (fallback)` with a `note` explaining why. A genuine
programming error is **not** swallowed as a fallback — it surfaces — so the
engine label stays honest.

## Strands Decider 2B — DEFERRED (smoke-tested, did not elevate)

A per-candidate calibrated match-confidence gate ("does this past case match the
new symptom? strong/weak") was considered. It was smoke-tested locally
(`gemma2:2b` via Ollama, CPU) on CaseJog-style yes/no pairs and **deferred**, for
two concrete reasons captured in `NOTES.md`:

1. **It fails the load-bearing discrimination.** On the near-but-wrong-cause
   pair (a drive-belt case vs a drain-pump `E18` symptom — the exact case the
   filtered-hybrid's error-code boost already gets right) it confidently
   answered `yes, 0.9`, repeatably, even when the prompt explicitly warned that
   a different root cause is not a match. It only emitted two confidence values
   (0.9 / 0.2), so it is a hard yes/no, not a calibrated probability.
2. **It blows the latency budget.** ~20 s per pair on this CPU → ~100 s per
   5-candidate search, which would destroy the sub-second demo that is a core
   selling point.

Per build steering (use Strands **only if it genuinely elevates**), it is not
wired into the code path. No half-wired model is left in. The optional
`confidence` fields the UI can render are simply never populated.

## Files

| File | Role |
|------|------|
| `embed.py` | swappable `embed(texts)->vectors`, fastembed, fail-loud |
| `gen_corpus.py` | realistic synthetic corpus with varied phrasings per root cause |
| `store.py` | MongoDB seam: upsert, metadata + Atlas vector index, `vector_search` |
| `ingest.py` | embed corpus + load into Mongo |
| `baseline.py` | pure-cosine baseline (Mongo-independent, offline fallback) |
| `search.py` | filtered hybrid vector search + grounded citations |
| `eval.py` | precision@k side-by-side (the honesty centerpiece) |
| `app.py` / `static/index.html` | FastAPI + one-page UI |
| `data/phase4-comparison.md` | the selection-gate evidence (table + hero queries) |
| `NOTES.md` | real build notes + Phase 5–9 verification evidence |
| `devSubmission.md` | the DEV submission write-up |

## Test posture

Minimal by design: no unit-test suite. The load-bearing checks (app boots and
returns grounded cards for the hero query; the offline fallback works with Mongo
down; `eval.py` reproduces the table; the Decider smoke test) were each run for
real and their output is captured in `NOTES.md` under "Phase 5-9 verification."
