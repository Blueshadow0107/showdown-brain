"""features_v2.py — state v2 -> numeric vector: everything v1 knows PLUS the
full-fidelity additions. Designed for the 'measure everything, ablate later'
phase: new dims are grouped in a clearly delimited block so feature-subset
experiments can slice them off with fixed offsets.

New vs v1 (300 dims):
  globals : turn/60                              (1)
  sides   : tera_used, screens active x4 + turns/5 x4, last_action kind x3
            (move/switch/none)                                     (1+8+3 = 12 per side)
  bench   : first 4 known bench mons x [hp, fainted, status one-hot(6)]
                                                                  (4*8 = 32 per side)
  actives : consecutive/4, times_entered/6                       (2 per side)
Total: 300 + 1 + 2*(12 + 32 + 2) = 393
"""
import numpy as np

from sb import features as F
from sb.features import one_hot  # noqa: F401

V2_DIM = 1 + 2 * (1 + 8 + 3 + 32 + 2)
DIM_V2 = F.DIM + V2_DIM  # 393


def _bench_block(side: dict) -> list[float]:
    team = side.get("team") or []
    active_sp = (side.get("active") or {}).get("species")
    slots = []
    for m in team:
        if m is None:
            continue
        if m.get("species") == active_sp and m.get("hp") == (side.get("active") or {}).get("hp"):
            continue  # the active mon (approx match; team[] has no role flag)
        if m.get("fainted"):
            slots.append(m)
        else:
            slots.append(m)
        if len(slots) >= 4:
            break
    v: list[float] = []
    for i in range(4):
        if i < len(slots):
            m = slots[i]
            v += [m.get("hp", 0.0), 1.0 if m.get("fainted") else 0.0]
            st = m.get("status")
            v += one_hot(F.STATUSES.index(st), len(F.STATUSES)) if st in F.STATUSES else [0.0] * len(F.STATUSES)
        else:
            v += [0.0] * 8
    return v


def _side_v2(side: dict) -> list[float]:
    v = [1.0 if side.get("tera_used") else 0.0]
    sc = side.get("screens") or {}
    for k in ("reflect", "lightscreen", "auroraveil", "tailwind"):
        t = sc.get(k, 0)
        v += [1.0 if t > 0 else 0.0, min(t, 5) / 5]
    la = side.get("last_action")
    v += one_hot(0 if la is None else (1 if la.startswith("move:") else 2), 3)
    v += _bench_block(side)
    act = side.get("active")
    if isinstance(act, list):  # doubles: take the healthiest slot's cadence
        act = next((m for m in act if m and m["hp"] > 0), None)
    v += [min((act or {}).get("consecutive") or 0, 4) / 4,
          min((act or {}).get("times_entered") or 0, 6) / 6]
    return v


def state_features_v2(state: dict) -> list[float]:
    v = F.state_features(state)  # v1 block, 300 dims
    v += [min(state.get("turn", 0), 60) / 60]
    v += _side_v2(state["my"]) + _side_v2(state["foe"])
    return v


def build_matrix_v2(rows):
    F._load()
    X = np.zeros((len(rows), DIM_V2), dtype="float32")
    y = np.zeros(len(rows), dtype="float32")
    g = np.zeros(len(rows), dtype="object")
    for i, r in enumerate(rows):
        X[i] = state_features_v2(r.get("state_pre") or r["state"])
        y[i] = r["outcome"]
        g[i] = r["game"]
    return X, y, g
