"""Generate a realistic-size synthetic corpus of past repair jobs for CaseJog.

Design intent (per SPEC §5 and winner evidence):
- A few hundred cases — believable for a working independent tech, NOT inflated
  to manufacture an ANN requirement.
- Multiple appliance types, real-looking brands/models/error codes.
- CRUCIALLY: multiple *varied symptom phrasings* per root cause. This phrasing
  variance (not corpus size) is what lets semantic/hybrid retrieval beat keyword
  search, and is what makes the demo honest.

Schema per case:
  {id, appliance_type, brand, model, error_code, symptom_text, root_cause,
   parts[], labor_minutes, price, date}

Outputs:
  data/corpus.jsonl   — the technician's searchable past jobs
  data/heldout.jsonl  — held-out set (symptom + known true root_cause) used by
                        eval.py for precision@k. Never indexed. The 2 held-out
                        phrasings per template are genuinely RESERVED: the
                        corpus is sampled from phrasings[2:] only, so no
                        held-out core phrasing appears in the corpus.
"""
from __future__ import annotations

import json
import os
import random
from datetime import date, timedelta

random.seed(1337)  # deterministic corpus

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Root-cause "templates". Each defines an appliance family, plausible brands/
# models/error codes, the canonical root cause, typical parts/labor/price, and
# a LIST OF VARIED SYMPTOM PHRASINGS describing the same underlying fault in the
# words different customers / the tech might scribble down.
#
# Each template carries >=7 distinct phrasings so that after RESERVING the
# first 2 (shuffled) for the held-out set, the corpus still draws from >=5
# distinct phrasings per root cause — comfortably meeting the ">=3 distinct
# phrasings per cause in the corpus" invariant WITHOUT sharing any held-out
# phrasing with the corpus.
# ---------------------------------------------------------------------------

BRANDS = {
    "washing machine": [
        ("Bosch", ["WAT28", "WAW32", "Serie 6", "WGG244"]),
        ("Samsung", ["WW90", "WF45", "WA50"]),
        ("LG", ["F4V5", "WM3900", "TurboWash"]),
        ("Whirlpool", ["WFW5620", "WTW5000"]),
    ],
    "fridge": [
        ("Samsung", ["RF28", "RS68", "RF23"]),
        ("LG", ["LFXS26", "GSL361", "InstaView"]),
        ("Bosch", ["KGN39", "Serie 4"]),
        ("Whirlpool", ["WRX735", "WRS325"]),
    ],
    "microwave": [
        ("Panasonic", ["NN-SN", "NN-SD"]),
        ("Samsung", ["ME19R", "MS23"]),
        ("LG", ["MS2595", "NeoChef"]),
        ("Sharp", ["R-21", "R-1874"]),
    ],
    "small electronics": [
        ("Dyson", ["V8", "V11", "DC59"]),
        ("Philips", ["HD9", "Senseo"]),
        ("DeLonghi", ["EC685", "Magnifica"]),
        ("Anker", ["PowerCore", "Soundcore"]),
    ],
}

# Each template: appliance_type, error_code, root_cause, parts, labor_minutes,
# price, and a pool of at least 7 distinct symptom phrasings.
TEMPLATES = [
    # --- washing machine ---
    dict(appliance="washing machine", error_code="E18", root_cause="Drain pump blocked by debris",
         parts=["drain pump", "pump filter"], labor=55, price=145,
         phrasings=[
             "drum won't spin and it throws E18",
             "machine stuck full of water, pump making noise",
             "won't drain, standing water in the tub after cycle",
             "spin cycle dead, water left in drum",
             "no agitation at end of wash, error E18 flashing",
             "grinding from the pump, water not emptying",
             "clothes come out soaked, tub never drains",
         ]),
    dict(appliance="washing machine", error_code="E23", root_cause="Door lock mechanism failed",
         parts=["door interlock", "latch"], labor=40, price=120,
         phrasings=[
             "door won't lock so it won't start, E23",
             "cycle won't begin, clicking at the door",
             "lid latch not engaging, machine dead",
             "says door open but it's shut tight",
             "won't fire up, latch area clicking repeatedly",
             "door shut but control insists it's open",
             "no start, interlock won't catch",
         ]),
    dict(appliance="washing machine", error_code="F21", root_cause="Worn drive belt",
         parts=["drive belt"], labor=50, price=110,
         phrasings=[
             "motor runs but drum doesn't turn",
             "loud squeal then nothing, drum won't rotate",
             "burning rubber smell, drum slips",
             "hears the motor but no spin, belt area smells hot",
             "drum turns by hand but not under power",
             "high-pitched whine and the tub stays still",
             "motor spins free, no drive to the drum",
         ]),
    dict(appliance="washing machine", error_code="E02", root_cause="Clogged inlet valve screen",
         parts=["inlet valve", "inlet screen"], labor=35, price=95,
         phrasings=[
             "takes forever to fill, E02",
             "barely any water coming in",
             "fill is super slow, times out",
             "trickle of water only, won't reach level",
             "fill stage drags on and errors out",
             "weak water intake, never fills properly",
             "long pause at fill, hardly any water",
         ]),
    # --- fridge ---
    dict(appliance="fridge", error_code="5E", root_cause="Defrost heater failure, evaporator iced over",
         parts=["defrost heater", "thermostat"], labor=70, price=185,
         phrasings=[
             "fridge warm but freezer fine, 5E",
             "not cooling up top, ice slab on the back panel",
             "food spoiling in fridge section, freezer ok",
             "frost buildup behind the drawers, poor cooling",
             "fresh food compartment warm, freezer frozen solid",
             "ice sheet on evaporator, fridge side not cold",
             "upper section losing cold, heavy frost at the rear",
         ]),
    dict(appliance="fridge", error_code="PC-ER", root_cause="Faulty main control board",
         parts=["main PCB"], labor=60, price=240,
         phrasings=[
             "display flashing PC-ER and nothing works",
             "random beeping, panel unresponsive",
             "compressor won't kick on, board seems dead",
             "lights flicker, no cooling, control acting up",
             "panel glitching, PC-ER keeps coming back",
             "controller frozen, compressor never starts",
             "erratic beeps and a dead control board",
         ]),
    dict(appliance="fridge", error_code="OF-OF", root_cause="Stuck in demo/showroom mode",
         parts=[], labor=15, price=45,
         phrasings=[
             "shows OF OF and not cooling",
             "panel says off but it's plugged in and lit",
             "not getting cold, screen reads OF OF",
             "customer thinks it's broken, display OF OF",
             "lit up but no cooling, reads OF OF",
             "display stuck on OF OF, compressor silent",
             "powered on yet warm, OF OF on the screen",
         ]),
    dict(appliance="fridge", error_code="22E", root_cause="Evaporator fan motor seized",
         parts=["evaporator fan motor"], labor=65, price=175,
         phrasings=[
             "freezer cold, fridge not, clicking fan noise",
             "loud rattling from the back, uneven cooling",
             "fan not spinning, warm fridge compartment",
             "buzzing then silence, fridge warms up",
             "intermittent fan clatter, cooling all over the place",
             "fan stalls, freezer fine but fridge drifts warm",
             "knocking from the evap fan, patchy temperatures",
         ]),
    # --- microwave ---
    dict(appliance="microwave", error_code="H97", root_cause="Magnetron failure",
         parts=["magnetron"], labor=45, price=160,
         phrasings=[
             "runs but doesn't heat the food, H97",
             "turntable spins, stays cold",
             "no heat at all, humming louder than usual",
             "food never warms up even on high",
             "lights and fan work but zero heating",
             "loud hum and cold food every time",
             "cooks for the full time, food still cold",
         ]),
    dict(appliance="microwave", error_code="SE", root_cause="Membrane keypad shorted",
         parts=["touch membrane panel"], labor=30, price=85,
         phrasings=[
             "buttons pressing themselves, SE error",
             "keypad unresponsive then beeps on its own",
             "ghost touches, panel misbehaving",
             "random key presses, shows SE",
             "panel inputs firing with no one touching it",
             "keys dead one minute, self-triggering the next",
             "touchpad going haywire, SE on the display",
         ]),
    dict(appliance="microwave", error_code="F6", root_cause="Door switch interlock faulty",
         parts=["door switch"], labor=25, price=75,
         phrasings=[
             "won't start with door shut, F6",
             "thinks the door is open when it's closed",
             "dead unless you slam the door",
             "no response on start, door interlock suspect",
             "only runs if you press hard on the door",
             "door closed but it refuses to start",
             "start does nothing, interlock not making contact",
         ]),
    # --- small electronics ---
    dict(appliance="small electronics", error_code="BLINK-RED", root_cause="Battery pack cells degraded",
         parts=["battery pack"], labor=30, price=90,
         phrasings=[
             "cordless vac dies after 2 minutes, red light blinking",
             "won't hold a charge anymore, flashing red",
             "runs for seconds then cuts out",
             "battery drains instantly, red LED flashing",
             "charge lasts no time at all, red blink",
             "full charge gone in moments, flashing red light",
             "dies almost immediately off the dock, red LED",
         ]),
    dict(appliance="small electronics", error_code="E05", root_cause="Clogged brew unit / scale buildup",
         parts=["descaling kit", "brew unit seal"], labor=35, price=80,
         phrasings=[
             "coffee machine won't brew, E05",
             "water not passing through, slow drip",
             "espresso machine jammed, no flow",
             "pump runs but no coffee comes out",
             "brew cycle starts but barely any liquid",
             "machine humming, hardly a trickle of coffee",
             "no espresso, water stuck in the brew unit",
         ]),
    dict(appliance="small electronics", error_code="OVHT", root_cause="Thermal fuse tripped from blocked vents",
         parts=["thermal fuse"], labor=25, price=65,
         phrasings=[
             "hair dryer cuts out when hot",
             "shuts off after a minute, smells warm",
             "turns off by itself when it heats up",
             "overheating and stopping, OVHT",
             "runs cool then dies once it warms up",
             "heat kicks in and it powers off",
             "cuts power the moment it gets hot, OVHT",
         ]),
]

FIRST_NOTES = [
    "customer dropped off", "callout at home", "second visit", "warranty check",
    "walk-in", "recurring complaint", "neighbour referral",
]


def _rand_date() -> str:
    start = date(2023, 1, 1)
    days = random.randint(0, 1000)
    return (start + timedelta(days=days)).isoformat()


def _pick_model(appliance: str) -> tuple[str, str]:
    brand, models = random.choice(BRANDS[appliance])
    model = random.choice(models) + "-" + str(random.randint(100, 999))
    return brand, model


def generate(target_cases: int = 300) -> tuple[list[dict], list[dict]]:
    corpus: list[dict] = []
    heldout: list[dict] = []
    cid = 0

    # Spread cases across templates roughly evenly, cycling phrasings so each
    # root cause gets many distinct wordings in the corpus.
    per_template = max(1, target_cases // len(TEMPLATES))

    for t in TEMPLATES:
        # Reserve 2 held-out phrasings per template and GENUINELY keep them out
        # of the corpus: the corpus is sampled from phrasings[2:] only. This
        # makes the held-out set phrasing-disjoint, not merely string-disjoint.
        phrasings = list(t["phrasings"])
        random.shuffle(phrasings)
        held_phrasings = phrasings[:2]
        corpus_phrasings = phrasings[2:]  # reserved held-out phrasings excluded

        for _ in range(per_template):
            cid += 1
            brand, model = _pick_model(t["appliance"])
            phrasing = random.choice(corpus_phrasings)
            note = random.choice(FIRST_NOTES)
            symptom = f"{brand} {model}: {phrasing} ({note})"
            corpus.append({
                "id": f"JOB-{cid:04d}",
                "appliance_type": t["appliance"],
                "brand": brand,
                "model": model,
                "error_code": t["error_code"],
                "symptom_text": symptom,
                "root_cause": t["root_cause"],
                "parts": t["parts"],
                "labor_minutes": t["labor"],
                "price": t["price"],
                "date": _rand_date(),
            })

        # Build held-out entries using the reserved phrasings and a FRESH
        # brand/model so they are not verbatim corpus rows.
        for hp in held_phrasings:
            brand, model = _pick_model(t["appliance"])
            heldout.append({
                "appliance_type": t["appliance"],
                "brand": brand,
                "model": model,
                "error_code": t["error_code"],
                "symptom_text": f"{brand} {model}: {hp}",
                "root_cause": t["root_cause"],
            })

    random.shuffle(corpus)
    return corpus, heldout


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    corpus, heldout = generate(target_cases=300)

    corpus_path = os.path.join(OUT_DIR, "corpus.jsonl")
    heldout_path = os.path.join(OUT_DIR, "heldout.jsonl")

    with open(corpus_path, "w") as f:
        for c in corpus:
            f.write(json.dumps(c) + "\n")
    with open(heldout_path, "w") as f:
        for h in heldout:
            f.write(json.dumps(h) + "\n")

    # --- spot verification: phrasing variance + disjointness ---
    from collections import defaultdict

    def core_of(txt: str) -> str:
        # strip the "Brand Model:" prefix and trailing note to isolate phrasing
        return txt.split(":", 1)[1].rsplit("(", 1)[0].strip() if ":" in txt else txt

    by_cause = defaultdict(set)
    corpus_cores = set()
    for c in corpus:
        core = core_of(c["symptom_text"])
        by_cause[c["root_cause"]].add(core)
        corpus_cores.add(core)

    corpus_symptoms = {c["symptom_text"] for c in corpus}
    heldout_symptoms = {h["symptom_text"] for h in heldout}
    overlap = corpus_symptoms & heldout_symptoms

    # phrasing-level disjointness: no held-out CORE phrasing appears in corpus
    held_cores = {core_of(h["symptom_text"]) for h in heldout}
    core_overlap = held_cores & corpus_cores

    multi = sum(1 for cause, s in by_cause.items() if len(s) >= 3)
    print(f"corpus_cases={len(corpus)} heldout_cases={len(heldout)}")
    print(f"root_causes={len(by_cause)} causes_with_>=3_distinct_phrasings={multi}")
    print(f"min_distinct_corpus_phrasings_per_cause={min(len(s) for s in by_cause.values())}")
    print(f"heldout_string_overlap_with_corpus={len(overlap)} (must be 0)")
    print(f"heldout_CORE_phrasing_overlap_with_corpus={len(core_overlap)} (must be 0)")
    for cause, s in list(by_cause.items())[:3]:
        print(f"  '{cause[:40]}...': {len(s)} distinct phrasings")


if __name__ == "__main__":
    main()
