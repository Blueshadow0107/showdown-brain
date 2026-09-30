"""jev_benchmark.py — pit TypeSafe/Jev's semantic judgment against kingambit's V(s).

Samples decision-point states from data/rows.jsonl, asks Jev structured
questions about each (win-probability score + who's-ahead choice), and
compares against V(s) predictions and the actual game outcomes.

usage:
  TYPESAFE_API_KEY=... uv run python scripts/jev_benchmark.py --limit 40
  TYPESAFE_API_KEY=... uv run python scripts/jev_benchmark.py --smoke   # 6 states, no V

Calls are batched (~8 states per request) and cached in
data/jev_bench_cache.json — re-runs only pay for new states.
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

API = "https://api.typesafe.ai/v1/systemone"
CACHE = ROOT / "data" / "jev_bench_cache.json"
ROWS = ROOT / "data" / "rows.jsonl"

# score questions take criteria as an ARRAY of 2-10 ordered strings (the API
# rejects dicts here — that's choice's shape). Index 0 = lowest.
WINPROB_LEVELS = [
    "certain loss", "almost certain loss", "heavy disadvantage",
    "clear disadvantage", "slight disadvantage", "even game",
    "slight advantage", "clear advantage", "heavy advantage",
    "almost certain win",
]


def call_api(body: dict, key: str, max_retries: int = 5) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(max_retries):
        req = urllib.request.Request(API, data=data, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 529, 500, 502, 503) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def describe_side(side: dict) -> dict:
    m = side.get("active")
    return {
        "active": m["species"] if m else None,
        "hp_pct": round(100 * m["hp"]) if m else 0,
        "boosts": m.get("boosts", {}) if m else {},
        "status": m.get("status") if m else None,
        "fainted": side["fainted"],
        "remaining": side["remaining"],
        "hazards": side.get("hazards", {}),
        "switches_so_far": side.get("switches", 0),
    }


def build_batch(states: list[dict]) -> dict:
    """states: [{key, mine, theirs, acting}] -> jev request body."""
    entities, questions = [], {}
    for i, s in enumerate(states):
        p = f"states[{i}]"
        entities.append({"mine": s["mine"], "theirs": s["theirs"],
                         "weather": s.get("weather"), "terrain": s.get("terrain")})
        questions[f"s{i}_winprob"] = {
            "type": "score",
            "instructions": (
                f"For the battle position `{p}` (perspective: the player whose "
                f"Pokemon is in `{p}.mine`): estimate THEIR probability of "
                f"winning the game. Consider remaining Pokemon, HP, boosts, "
                f"hazards, and type matchups. Rate against `rubrics.winprob`."
            ),
            "criteria": WINPROB_LEVELS,
        }
        questions[f"s{i}_ahead"] = {
            "type": "choice",
            "instructions": (
                f"For `{p}`: who is currently AHEAD in the game (better "
                f"position to win, ignoring luck)?"
            ),
            "criteria": {"me": "mine is ahead", "even": "roughly even",
                         "foe": "theirs is ahead"},
        }
    return {"model": "jev-latest",
            "state": {"winprob": WINPROB_LEVELS, "states": entities},
            "questions": questions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="6 states, no V compare")
    ap.add_argument("--min-turn", type=int, default=3)
    ap.add_argument("--max-turn", type=int, default=25)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        sys.exit("set TYPESAFE_API_KEY")

    rows = []
    with open(ROWS) as f:
        for line in f:
            r = json.loads(line)
            if args.min_turn <= r["turn"] <= args.max_turn:
                rows.append(r)
    # balanced sample: interleave wins/losses, fixed seed
    import hashlib as _hl
    rows.sort(key=lambda r: (r["outcome"], int(_hl.md5(r["game"].encode()).hexdigest(), 16) & 0xFFFF))
    wins = [r for r in rows if r["outcome"] == 1.0]
    losses = [r for r in rows if r["outcome"] == 0.0]
    n = args.limit if not args.smoke else 6
    sample = []
    for i in range(n):
        src = wins if i % 2 == 0 else losses
        if src:
            sample.append(src.pop())
    print(f"{len(sample)} states sampled from {len(rows)} eligible rows")

    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    pending_keys = []
    for r in sample:
        k = f"{r['game']}:{r['turn']}:{r['player']}"
        if k not in cache:
            pending_keys.append((k, r))

    for i in range(0, len(pending_keys), args.batch):
        chunk = pending_keys[i:i + args.batch]
        states = [{"key": k, "mine": describe_side(r["state_pre"]["my"]),
                   "theirs": describe_side(r["state_pre"]["foe"]),
                   "weather": r["state_pre"].get("weather"),
                   "terrain": r["state_pre"].get("terrain")}
                  for k, r in chunk]
        body = build_batch(states)
        resp = call_api(body, key)
        answers = resp.get("answers", resp)
        for j, (k, r) in enumerate(chunk):
            wp = answers[f"s{j}_winprob"]
            ah = answers[f"s{j}_ahead"]
            cache[k] = {
                # score is a continuous expected index over the levels
                "winprob": round(wp["score"] / (len(WINPROB_LEVELS) - 1) * 100),
                "winprob_conf": round(wp.get("confidence", 0), 3),
                "ahead": max(ah["probabilities"], key=ah["probabilities"].get),
                "outcome": r["outcome"],
                "turn": r["turn"],
            }
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(cache, open(CACHE, "w"))
        print(f"  cached {len(chunk)} states ({len(cache)} total)", flush=True)
        time.sleep(args.sleep)

    # evaluation
    evald = [cache[f"{r['game']}:{r['turn']}:{r['player']}"] for r in sample
             if f"{r['game']}:{r['turn']}:{r['player']}" in cache]
    if not args.smoke:
        from sb import features  # noqa: E402
        import lightgbm as lgb  # noqa: E402
        import numpy as np  # noqa: E402
        features._load()
        v = lgb.Booster(model_file=str(ROOT / "models" / "v_model.txt"))
        v_preds, jev_preds, outs = [], [], []
        for r in sample:
            k = f"{r['game']}:{r['turn']}:{r['player']}"
            if k not in cache:
                continue
            x = [features.state_features(r.get("state_pre") or r["state"])]
            v_preds.append(float(v.predict(np.array(x, dtype="float32"))[0]))
            jev_preds.append(cache[k]["winprob"] / 100)
            outs.append(r["outcome"])
        mae = lambda p: sum(abs(a - b) for a, b in zip(p, outs)) / len(outs)
        corr = lambda a, b: np.corrcoef(a, b)[0, 1]
        agree = sum(1 for jp, vp in zip(jev_preds, v_preds)
                    if (jp > 0.5) == (vp > 0.5)) / len(v_preds)
        print(f"\n=== jev vs kingambit-V on {len(outs)} states ===")
        print(f"MAE   jev {mae(jev_preds):.3f}   V {mae(v_preds):.3f}   (lower better)")
        print(f"corr  jev {corr(jev_preds, outs):.3f}   V {corr(v_preds, outs):.3f}")
        print(f"sign agreement (who's ahead): {agree:.2%}")
    else:
        for e in evald:
            print(f"  t{e['turn']:>2} jev winprob {e['winprob']:>3}% "
                  f"({e['ahead']:>4})  actual {e['outcome']:.0f}")


if __name__ == "__main__":
    main()
