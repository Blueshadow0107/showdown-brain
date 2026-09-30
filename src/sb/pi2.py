"""pi2.py — two-stage policy prior inference: pi(a|s) = P(kind|s) * P(id|kind, s).

Stage 1 (binary): P(action is a move | state).
Stage 2a (multiclass): move-id ranker over legal moves.
Stage 2b (multiclass): species ranker over legal switches.

Legal actions look like "move:earthquake" / "switch:pikachu". Out-of-vocab ids
get the ranker's "other" class mass (treated as tail). Result is renormalized
to sum to 1 over the legal set.
"""
import json
import pathlib

import lightgbm as lgb
import numpy as np

import sb.features as _F

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "data"

_bundles = {}

# Rows are post-decision snapshots: for switch rows the label species is already
# the acting player's active mon, fingerprinted by its type one-hots + base stats.
# The switch ranker is TRAINED with that block zeroed (see train_pi2.py FP_SLICE),
# so inference must apply the same mask. Offsets derive from features.py constants.
_FP_START = len(_F.WEATHERS) + len(_F.TERRAINS) + 1 + (3 + 4 + 4)
FP_SLICE = slice(_FP_START, _FP_START + 2 * len(_F.TYPES) + 6 + 1)


def load(data_dir=None):
    """Load the pi2 bundle {kind, move, switch, meta}. Cached per data_dir.

    Raises FileNotFoundError with a clear message if any model file is missing.
    """
    key = str(data_dir or DATA)
    if key in _bundles:
        return _bundles[key]
    d = pathlib.Path(data_dir) if data_dir else DATA
    paths = {k: d / f"pi2_{k}.txt" for k in ("kind", "move", "switch")}
    labels = d / "pi2_labels.json"
    missing = [str(p) for p in list(paths.values()) + [labels] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "pi2 policy bundle not found — train it first with "
            "`uv run python scripts/train_pi2.py`. Missing: " + ", ".join(missing))
    _F._load()  # species/tags needed by state_features
    bundle = {
        "kind": lgb.Booster(model_file=str(paths["kind"])),
        "move": lgb.Booster(model_file=str(paths["move"])),
        "switch": lgb.Booster(model_file=str(paths["switch"])),
        "meta": json.load(open(labels)),
    }
    _bundles[key] = bundle
    return bundle


def _rank_probs(bundle, kind, x):
    """Ranker softmax row + tail (other-class) mass for `kind` in {move, switch}."""
    meta = bundle["meta"][kind]
    pred = bundle[kind].predict(x)[0]
    return pred, float(pred[meta["other_idx"]])


def action_probs(bundle, state: dict, legal_actions: list) -> dict:
    """P(a|s) over a legal action set, renormalized to sum to 1.

    P(a|s) = P(kind|s) * P_ranker(id|kind, s). Out-of-vocab ids are assigned
    the ranker's tail (other-class) mass each. Actions whose kind the legal
    set doesn't contain keep their kind mass unused (renormalization handles
    impossible kinds naturally).
    """
    x = np.asarray(_F.state_features(state), dtype="float32").reshape(1, -1)
    x_sw = x.copy()
    x_sw[:, FP_SLICE] = 0.0  # match train-time masking for the switch ranker
    p_move = float(bundle["kind"].predict(x)[0])
    kind_p = {"move": p_move, "switch": 1.0 - p_move}
    move_pred, move_tail = _rank_probs(bundle, "move", x)
    sw_pred, sw_tail = _rank_probs(bundle, "switch", x_sw)
    rank = {"move": (move_pred, move_tail), "switch": (sw_pred, sw_tail)}

    probs = {}
    for a in legal_actions:
        kind, _, aid = a.partition(":")
        if kind not in rank:
            continue
        pred, tail = rank[kind]
        idx = bundle["meta"][kind]["vocab_idx"].get(aid)
        mass = float(pred[idx]) if idx is not None else tail
        probs[a] = kind_p[kind] * mass
    total = sum(probs.values())
    if total > 0:
        probs = {a: p / total for a, p in probs.items()}
    return probs
