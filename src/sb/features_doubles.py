"""features_doubles.py — doubles state dict -> fixed-length numeric vector.

Mirrors features.py for parser_doubles' row schema: per side TWO active slots
(`active` is a list of 2 mon dicts), plus the acting `slot` ("p1a"..) as a
feature. Self-contained copy of the singles featurizers — features.py must stay
byte-stable for the singles pipeline, so the doubles volatiles list and the
2-slot side vector live here.

Usage:
    from sb import features, features_doubles as fd
    features._load()                       # loads SPECIES + tag maps (shared)
    X, y, g = fd.build_matrix_d(rows)
"""
import numpy as np

from sb import features as F
from sb.features import one_hot  # noqa: F401  (re-exported for callers)

# singles volatiles plus the doubles-relevant singleturn/singlemove conditions
VOL_D = F.VOLATILES + [v for v in (
    "helpinghand", "followme", "ragepowder", "roost", "glaiverush",
    "wideguard", "quickguard", "craftyshield", "matblock", "maxguard",
) if v not in F.VOLATILES]


def mon_features_d(m: dict | None) -> list[float]:
    """Identical layout to features.mon_features but with the VOL_D volatile set."""
    if not m:
        return [0.0] * MON_DIM_D
    v = []
    sp = F.SPECIES.get(F.resolve_species(m["species"]))
    if sp and "types" not in sp and sp.get("baseSpecies"):
        sp = F.SPECIES.get(F.to_id(sp["baseSpecies"]), sp)
    if sp and "types" in sp:
        t1 = F.TYPES.index(sp["types"][0]) if sp["types"][0] in F.TYPES else len(F.TYPES) - 1
        v += one_hot(t1, len(F.TYPES))
        if len(sp["types"]) > 1 and sp["types"][1] in F.TYPES:
            v += one_hot(F.TYPES.index(sp["types"][1]), len(F.TYPES))
        else:
            v += [0.0] * len(F.TYPES)
        bs = sp["baseStats"]
        v += [bs["hp"] / 255, bs["atk"] / 190, bs["def"] / 230, bs["spa"] / 194,
              bs["spd"] / 230, bs["spe"] / 180]
        v += [1.0]
    else:
        v += [0.0] * (2 * len(F.TYPES) + 6) + [0.0]
    v += [m["hp"], 1.0 if m["hp"] == 0 else 0.0]
    st = m["status"]
    v += one_hot(F.STATUSES.index(st), len(F.STATUSES)) if st in F.STATUSES else [0.0] * len(F.STATUSES)
    v += [m["boosts"].get(b, 0) / 6 for b in F.BOOSTS]
    vol = set(m["volatiles"])
    v += [1.0 if x in vol else 0.0 for x in VOL_D]
    it = m["item"]
    it_tags = F.ITEM_TAGMAP.get(it, set()) if it else set()
    v += [1.0 if it else 0.0] + [1.0 if t in it_tags else 0.0 for t in F.ITEM_TAGS]
    ab = m["ability"]
    ab_tags = F.ABIL_TAGMAP.get(ab, set()) if ab else set()
    v += [1.0 if ab else 0.0] + [1.0 if t in ab_tags else 0.0 for t in F.ABILITY_TAGS]
    tera = m["tera"]
    v += one_hot(F.TYPES.index(tera), len(F.TYPES)) if tera in F.TYPES else [0.0] * len(F.TYPES)
    v += [min(len(m["moves"]), 4) / 4]
    return v


MON_DIM_D = len(F.TYPES) * 2 + 6 + 1 + 2 + len(F.STATUSES) + len(F.BOOSTS) \
    + len(VOL_D) + 1 + len(F.ITEM_TAGS) + 1 + len(F.ABILITY_TAGS) + len(F.TYPES) + 1


def side_features_d(s: dict) -> list[float]:
    # bench in doubles holds at most 4 (team of 6 minus 2 actives)
    hz = s["hazards"]
    sc = s.get("screens", {})
    scr = lambda k: 1.0 if (sc.get(k) or hz.get(k)) else 0.0
    v = [s["fainted"] / 6, s["remaining"] / 6, s["bench_known"] / 4]
    v += [1.0 if hz.get("stealthrock") else 0.0,
          hz.get("spikes", 0) / 3, hz.get("toxicspikes", 0) / 2,
          1.0 if hz.get("stickyweb") else 0.0]
    v += [scr("reflect"), scr("lightscreen"), scr("auroraveil"), scr("tailwind")]
    act = s["active"] if isinstance(s.get("active"), list) else [s.get("active"), None]
    v += mon_features_d(act[0]) + mon_features_d(act[1] if len(act) > 1 else None)
    return v


def state_features_d(state: dict, slot: str | None = None) -> list[float]:
    """slot: "p1a".. — the acting slot; its letter becomes a 2-wide one-hot."""
    w = F.WEATHER_MAP.get(state["weather"], "other") if state["weather"] else None
    t = state["terrain"] if state["terrain"] in F.TERRAINS else ("other" if state["terrain"] else None)
    v = one_hot(F.WEATHERS.index(w), len(F.WEATHERS)) if w else [0.0] * len(F.WEATHERS)
    v += one_hot(F.TERRAINS.index(t), len(F.TERRAINS)) if t else [0.0] * len(F.TERRAINS)
    v += [1.0 if state["trick_room"] else 0.0]
    v += side_features_d(state["my"]) + side_features_d(state["foe"])
    letter = slot[-1] if slot else None
    v += one_hot(0 if letter == "a" else 1, 2) if letter in ("a", "b") else [0.0, 0.0]
    v += _momentum_d(state, slot)
    return v


def _momentum_d(state: dict, slot: str | None) -> list[float]:
    """Doubles momentum: best-speed comparison across both slots, the acting
    slot's kill-clock vs the most dangerous foe clock, priority presence,
    and per-side switch counters. Mirrors features._momentum (13 dims)."""
    from sb import transitions as T
    my_act = [m for m in state["my"]["active"] if m and m["hp"] > 0]
    foe_act = [m for m in state["foe"]["active"] if m and m["hp"] > 0]
    letter = slot[-1] if slot else None
    acting = None
    if letter in ("a", "b"):
        cand = state["my"]["active"][0 if letter == "a" else 1]
        acting = cand if cand and cand["hp"] > 0 else None
    v: list[float] = []
    if my_act and foe_act:
        ms = max(T.effective_speed(m, state) for m in my_act)
        fs = max(T.effective_speed(m, state) for m in foe_act)
        v += one_hot(0 if ms > fs * 1.05 else (2 if fs > ms * 1.05 else 1), 3)
        tgt = acting or my_act[0]
        press = min(T.kill_clock(tgt, f, state) for f in foe_act)
        v += one_hot(T.kill_clock(acting, foe_act[0], state) - 1 if acting else 2, 3)
        v += one_hot(press - 1, 3)
        v += [1.0 if acting and T.has_priority(acting) else 0.0,
              1.0 if any(T.has_priority(f) for f in foe_act) else 0.0]
    else:
        v += [0.0] * 11
    v += [min(state["my"].get("switches", 0), 6) / 6,
          min(state["foe"].get("switches", 0), 6) / 6]
    return v


DIM_D = len(F.WEATHERS) + len(F.TERRAINS) + 1 + 2 * (3 + 4 + 4 + 2 * MON_DIM_D) + 2 + 13


def build_matrix_d(rows):
    """rows -> (X float32 [n, DIM_D], y outcomes float32, g game-id array)."""
    F._load()
    X = np.zeros((len(rows), DIM_D), dtype="float32")
    y = np.zeros(len(rows), dtype="float32")
    g = np.zeros(len(rows), dtype="object")
    for i, r in enumerate(rows):
        X[i] = state_features_d(r["state"], r.get("slot"))
        y[i] = r["outcome"]
        g[i] = r["game"]
    return X, y, g
