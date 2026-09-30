"""jev_analyze.py — deeper cuts into cached Jev-vs-V benchmark data.

Reads data/jev_bench_cache.json, matches states back to rows.jsonl, recomputes
V(s), and reports: biggest disagreements, calibration buckets, agreement by
turn. Zero API spend (cache-only).
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sb import features  # noqa: E402
import lightgbm as lgb  # noqa: E402

CACHE = ROOT / "data" / "jev_bench_cache.json"


def main():
    features._load()
    v = lgb.Booster(model_file=str(ROOT / "models" / "v_model.txt"))
    cache = json.load(open(CACHE))
    rows = {}
    with open(ROOT / "data" / "rows.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows[f"{r['game']}:{r['turn']}:{r['player']}"] = r

    evald = []
    for k, e in cache.items():
        if k not in rows:
            continue
        r = rows[k]
        x = [features.state_features(r.get("state_pre") or r["state"])]
        vp = float(v.predict(np.array(x, dtype="float32"))[0])
        st = r.get("state_pre") or r["state"]
        evald.append({"k": k, "turn": e["turn"], "outcome": e["outcome"],
                      "jev": e["winprob"] / 100, "v": vp,
                      "diff": abs(e["winprob"] / 100 - vp),
                      "mine": (st["my"]["active"] or {}).get("species"),
                      "theirs": (st["foe"]["active"] or {}).get("species"),
                      "my_hp": round(100 * (st["my"]["active"] or {}).get("hp", 0)),
                      "foe_hp": round(100 * (st["foe"]["active"] or {}).get("hp", 0))})
    print(f"{len(evald)} cached states matched\n")

    evald.sort(key=lambda d: -d["diff"])
    print("=== top disagreements |jev - V| ===")
    for d in evald[:6]:
        print(f"  {d['diff']:.2f}  jev {d['jev']:.2f} vs V {d['v']:.2f}  "
              f"t{d['turn']:>2}  {d['mine']}({d['my_hp']}%) vs {d['theirs']}({d['foe_hp']}%)"
              f"  -> actual {d['outcome']:.0f}")

    for name, preds in (("jev", [d["jev"] for d in evald]),
                        ("V", [d["v"] for d in evald])):
        outs = [d["outcome"] for d in evald]
        print(f"\n=== {name} calibration ===")
        for lo in (0.0, 0.33, 0.45):
            hi = {0.0: 0.33, 0.33: 0.45, 0.45: 1.01}[lo]
            idx = [i for i, p in enumerate(preds) if lo <= p < hi]
            if idx:
                acc = sum(outs[i] for i in idx) / len(idx)
                print(f"  pred[{lo:.2f},{hi:.2f}): n={len(idx):>2}  actual winrate {acc:.2f}")

    early = [d for d in evald if d["turn"] < 12]
    late = [d for d in evald if d["turn"] >= 12]
    for name, seg in (("early (<t12)", early), ("late (>=t12)", late)):
        if not seg:
            continue
        agree = sum(1 for d in seg if (d["jev"] > 0.5) == (d["v"] > 0.5)) / len(seg)
        jmae = sum(abs(d["jev"] - d["outcome"]) for d in seg) / len(seg)
        vmae = sum(abs(d["v"] - d["outcome"]) for d in seg) / len(seg)
        print(f"\n{name}: n={len(seg)}  sign-agree {agree:.0%}  MAE jev {jmae:.3f} V {vmae:.3f}")


if __name__ == "__main__":
    main()
