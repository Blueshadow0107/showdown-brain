"""pi2doubles.py — two-stage doubles policy prior: pi(a|s) = P(kind|s) * P(id|kind, s).

Doubles mirror of pi2.py, trained by scripts/train_pi2_doubles.py on
parser_doubles rows (see data/pi2d_*.txt / pi2d_labels.json).

Stage 1 (binary): P(action is a move | state, slot).
Stage 2a (multiclass): move-id ranker over legal moves.
Stage 2b (multiclass): species ranker over legal switches.

LEAK / MASKING: unlike singles, NO feature masking is applied at train or
inference time. Doubles rows carry pre-turn snapshots (state_pre == state) and
the bench is represented only as a count, so the switched-in species is not
fingerprintable from the features. The unmasked leak probe in
train_pi2_doubles.py (a 300-round ranker on identical features) confirmed this:
its val top-1 sits near the base rate, not near 1.0. If doubles rows are ever
re-parsed with post-decision states, re-run the probe and add a mask like
pi2.py's FP_SLICE if it trips.

Legal actions look like "move:earthquake" / "switch:pikachu" and must be the
acting SLOT's actions: moves of the mon in `slot` ("p1a"/"p1b"/...), switches
of that slot's player. Out-of-vocab ids get the ranker's "other" class mass
(treated as tail). Result is renormalized to sum to 1 over the legal set.

Agent call convention:
    bundle = pi2doubles.load()                       # or load(data_dir)
    probs = pi2doubles.action_probs(bundle, state, slot, legal_actions)
where `state` is a parser_doubles state dict seen from the acting player's
perspective (state["my"].active[0]["pos"] in {"p1a","p1b"} for player p1) and
`slot` is the deciding slot string, e.g. "p1a".
"""
import json
import pathlib

import lightgbm as lgb
import numpy as np

from sb import features as _F
from sb.features_doubles import state_features_d

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "data"

_bundles = {}


def load(data_dir=None):
    """Load the pi2d bundle {kind, move, switch, meta}. Cached per data_dir.

    Raises FileNotFoundError with a clear message if any model file is missing.
    """
    key = str(data_dir or DATA)
    if key in _bundles:
        return _bundles[key]
    d = pathlib.Path(data_dir) if data_dir else DATA
    paths = {k: d / f"pi2d_{k}.txt" for k in ("kind", "move", "switch")}
    labels = d / "pi2d_labels.json"
    missing = [str(p) for p in list(paths.values()) + [labels] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "pi2d policy bundle not found — train it first with "
            "`uv run python scripts/train_pi2_doubles.py`. Missing: " + ", ".join(missing))
    _F._load()  # species/tags needed by state_features_d
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


def action_probs(bundle, state: dict, slot: str, legal_actions: list) -> dict:
    """P(a|s) over a legal action set for the acting `slot`, renormalized to sum to 1.

    P(a|s) = P(kind|s) * P_ranker(id|kind, s). Out-of-vocab ids are assigned
    the ranker's tail (other-class) mass each. Actions whose kind the legal
    set doesn't contain keep their kind mass unused (renormalization handles
    impossible kinds naturally).
    """
    x = np.asarray(state_features_d(state, slot), dtype="float32").reshape(1, -1)
    p_move = float(bundle["kind"].predict(x)[0])
    kind_p = {"move": p_move, "switch": 1.0 - p_move}
    move_pred, move_tail = _rank_probs(bundle, "move", x)
    sw_pred, sw_tail = _rank_probs(bundle, "switch", x)
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
