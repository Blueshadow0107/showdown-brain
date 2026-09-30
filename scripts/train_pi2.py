"""train_pi2.py — two-stage policy prior.

Stage 1: binary P(action is a move | state), all rows.
Stage 2a: move ranker P(move id | state, move) over move-id vocab.
Stage 2b: switch ranker P(species | state, switch) over species vocab.

Same game-id split as train_pi.py (val = random 20% of games, seed 11).
Rows are streamed once; only the feature matrix + compact labels are held.
"""
import json
import pathlib
import sys
from collections import Counter

import lightgbm as lgb
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sb.features import DIM, TERRAINS, TYPES, WEATHERS, _load as _features_load, state_features  # noqa: E402

ROWS = ROOT / "data" / "rows.jsonl"
MIN_COUNT = 40       # action must appear at least this often to get its own class
TOP_K = 500          # vocab cap
SEED = 11
VAL_FRAC = 0.2

# Rows are POST-decision snapshots: for switch rows the label species is already the
# acting player's active mon, and its type one-hots + base stats uniquely fingerprint
# it. Zero that block for the switch ranker so it learns state->target, not identity.
_FP_START = len(WEATHERS) + len(TERRAINS) + 1 + (3 + 4 + 4)   # header + side header
FP_SLICE = slice(_FP_START, _FP_START + 2 * len(TYPES) + 6 + 1)


def stream_rows():
    """Count rows, then one parse pass: featurize into X + collect compact labels."""
    with open(ROWS) as f:
        n = sum(1 for _ in f)
    _features_load()
    X = np.zeros((n, DIM), dtype="float32")
    y_kind = np.zeros(n, dtype="uint8")          # 1 = move, 0 = switch
    games = np.empty(n, dtype="U40")
    action_ids = [None] * n                      # raw action_id per row (small)
    move_counts, switch_counts = Counter(), Counter()
    kind_counts = Counter()
    with open(ROWS) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            X[i] = state_features(r.get("state_pre") or r["state"])
            games[i] = r["game"]
            kind = r["action_kind"]
            aid = r["action_id"]
            y_kind[i] = 1 if kind == "move" else 0
            kind_counts[kind] += 1
            action_ids[i] = aid
            (move_counts if kind == "move" else switch_counts)[aid] += 1
            if i and i % 50000 == 0:
                print(f"  featurized {i}/{n}", flush=True)
    return X, y_kind, games, action_ids, kind_counts, move_counts, switch_counts


def make_vocab(counts):
    vocab = [a for a, c in counts.most_common(TOP_K) if c >= MIN_COUNT]
    return vocab, {a: i for i, a in enumerate(vocab)}, len(vocab)  # vocab, idx, other_idx


def split_masks(games):
    uniq = np.unique(games)
    rng = np.random.default_rng(SEED)
    val_games = rng.choice(uniq, size=int(VAL_FRAC * len(uniq)), replace=False)
    return np.isin(games, val_games)


def train_binary(Xtr, ytr, Xva, yva):
    params = {"objective": "binary", "metric": ["binary_logloss", "auc"],
              "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 100,
              "feature_fraction": 0.7, "num_threads": 4, "verbose": -1}
    dtrain = lgb.Dataset(Xtr, ytr)
    dval = lgb.Dataset(Xva, yva, reference=dtrain)
    m = lgb.train(params, dtrain, num_boost_round=1500, valid_sets=[dval],
                  callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    return m


def train_ranker(Xtr, ytr, Xva, yva, n_class, boost_rounds=1500):
    # NOTE: lr 0.03 / 63 leaves diverges on these multiclass problems — val logloss
    # worsens from round 1 and early stopping exits at iter 1 (same failure as the
    # flat train_pi.py model, best_iter=1). lr 0.01 / 31 leaves descends properly.
    params = {"objective": "multiclass", "num_class": n_class,
              "metric": ["multi_logloss"], "learning_rate": 0.01,
              "num_leaves": 31, "min_data_in_leaf": 100,
              "feature_fraction": 0.7, "num_threads": 4, "verbose": -1}
    dtrain = lgb.Dataset(Xtr, ytr)
    dval = lgb.Dataset(Xva, yva, reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=boost_rounds, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])


def topk_acc(model, Xva, yva, k=3):
    p = model.predict(Xva)
    top1 = int((p.argmax(1) == yva).sum())
    topk = int(((np.argsort(-p, axis=1)[:, :k]) == yva[:, None]).any(1).sum())
    return p, top1 / len(yva), topk / len(yva)


def main():
    print("streaming rows + building features ...", flush=True)
    X, y_kind, games, action_ids, kind_counts, move_counts, switch_counts = stream_rows()
    n = len(games)
    print(f"{n} rows | kinds: {dict(kind_counts)}", flush=True)

    move_vocab, move_idx, move_other = make_vocab(move_counts)
    sw_vocab, sw_idx, sw_other = make_vocab(switch_counts)
    move_mask = y_kind == 1
    sw_mask = ~move_mask
    # stage-2 labels: idx in vocab, or other_idx; -1 on rows of the other kind
    ids = np.array(action_ids, dtype=object)
    y_move = np.full(n, -1, dtype="int32")
    y_switch = np.full(n, -1, dtype="int32")
    y_move[move_mask] = [move_idx.get(a, move_other) for a in ids[move_mask]]
    y_switch[sw_mask] = [sw_idx.get(a, sw_other) for a in ids[sw_mask]]

    val_mask = split_masks(games)
    train_mask = ~val_mask
    print(f"{len(np.unique(games))} games | val games {int(val_mask.sum() and len(np.unique(games[val_mask])))} "
          f"| val rows {int(val_mask.sum())}", flush=True)

    metrics = {"rows": int(n), "val_rows": int(val_mask.sum()),
               "kinds": {k: int(v) for k, v in kind_counts.items()},
               "min_count": MIN_COUNT, "top_k": TOP_K, "seed": SEED}

    # ---- stage 1: move-vs-switch
    print("stage 1: kind classifier ...", flush=True)
    m_kind = train_binary(X[train_mask], y_kind[train_mask], X[val_mask], y_kind[val_mask])
    p_kind = m_kind.predict(X[val_mask])
    kind_pred = (p_kind > 0.5).astype(int)
    acc = float((kind_pred == y_kind[val_mask]).mean())
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_kind[val_mask], p_kind))
    except Exception:
        auc = None
    metrics["kind"] = {"val_acc": round(acc, 4), "val_auc": round(auc, 4) if auc else None,
                       "best_iter": m_kind.best_iteration,
                       "val_move_share": round(float(y_kind[val_mask].mean()), 4)}
    print(f"  kind: val acc {acc:.4f} auc {auc:.4f}" if auc else f"  kind: val acc {acc:.4f}", flush=True)

    # ---- stage 2a: move ranker
    print("stage 2a: move ranker ...", flush=True)
    tr = train_mask & move_mask
    va = val_mask & move_mask
    n_move_class = len(move_vocab) + 1
    print(f"  {int(tr.sum())} train / {int(va.sum())} val rows, {n_move_class} classes "
          f"({len(move_vocab)} real + other)", flush=True)
    m_move = train_ranker(X[tr], y_move[tr], X[va], y_move[va], n_move_class)
    _, t1, t3 = topk_acc(m_move, X[va], y_move[va])
    other_share = float((y_move[va] == move_other).mean())
    metrics["move"] = {"classes": n_move_class, "vocab": len(move_vocab),
                       "val_rows": int(va.sum()), "top1_acc": round(t1, 4),
                       "top3_acc": round(t3, 4), "other_label_share": round(other_share, 4),
                       "best_iter": m_move.best_iteration}
    print(f"  move: top-1 {t1:.4f} top-3 {t3:.4f} (other share {other_share:.4f})", flush=True)

    # ---- stage 2b: switch ranker (on fingerprint-masked features; see FP_SLICE note)
    print("stage 2b: switch ranker ...", flush=True)
    tr = train_mask & sw_mask
    va = val_mask & sw_mask
    n_sw_class = len(sw_vocab) + 1
    print(f"  {int(tr.sum())} train / {int(va.sum())} val rows, {n_sw_class} classes "
          f"({len(sw_vocab)} real + other)", flush=True)
    X_sw = X.copy()
    X_sw[:, FP_SLICE] = 0.0
    m_switch = train_ranker(X_sw[tr], y_switch[tr], X_sw[va], y_switch[va], n_sw_class)
    _, t1, t3 = topk_acc(m_switch, X_sw[va], y_switch[va])
    other_share = float((y_switch[va] == sw_other).mean())
    metrics["switch"] = {"classes": n_sw_class, "vocab": len(sw_vocab),
                         "val_rows": int(va.sum()), "top1_acc": round(t1, 4),
                         "top3_acc": round(t3, 4), "other_label_share": round(other_share, 4),
                         "best_iter": m_switch.best_iteration,
                         "active_fingerprint_masked": True}
    print(f"  switch (masked): top-1 {t1:.4f} top-3 {t3:.4f} (other share {other_share:.4f})",
          flush=True)

    # leak probe: same ranker on UNMASKED features. Rows are post-decision snapshots
    # (label == my.active.species in 100% of switch rows), so this should score near
    # 1.0 — it measures the leak, and is NOT the model we ship.
    print("  leak probe: unmasked switch ranker (not saved) ...", flush=True)
    m_probe = train_ranker(X[tr], y_switch[tr], X[va], y_switch[va], n_sw_class, boost_rounds=300)
    _, p1, p3 = topk_acc(m_probe, X[va], y_switch[va])
    metrics["switch"]["leak_probe"] = {"unmasked_top1_acc": round(p1, 4),
                                       "unmasked_top3_acc": round(p3, 4),
                                       "note": "label==active species in 100% of switch rows; "
                                               "near-1.0 confirms post-decision snapshot leak"}
    print(f"  leak probe: top-1 {p1:.4f} top-3 {p3:.4f}", flush=True)

    metrics["data_notes"] = {
        "states_are_post_decision": "switch label == my.active.species in 100% of rows; "
                                    "used move is in my.active.moves in 100% of move rows",
        "move_ranker": "no direct identity leak (features see only the COUNT of known moves); "
                       "count includes the just-used move, slightly inflated vs pre-decision",
        "switch_ranker": "trained with my-active type/base-stat fingerprint zeroed (FP_SLICE) "
                         "so it predicts the switch TARGET from state, not the post-decision active",
    }

    # ---- flat baseline for context
    pm = ROOT / "data" / "pi_metrics.json"
    if pm.exists():
        base = json.load(open(pm))
        metrics["flat_baseline"] = {"top1_acc": base.get("top1_acc"), "top3_acc": base.get("top3_acc"),
                                    "note": "scripts/train_pi.py, 500-class flat multiclass"}
        print(f"flat baseline (train_pi.py): top-1 {base.get('top1_acc')} top-3 {base.get('top3_acc')}",
              flush=True)

    # ---- save
    out = ROOT / "data"
    m_kind.save_model(str(out / "pi2_kind.txt"))
    m_move.save_model(str(out / "pi2_move.txt"))
    m_switch.save_model(str(out / "pi2_switch.txt"))
    json.dump({
        "move": {"vocab": move_vocab, "vocab_idx": move_idx, "other_idx": move_other},
        "switch": {"vocab": sw_vocab, "vocab_idx": sw_idx, "other_idx": sw_other},
        "min_count": MIN_COUNT, "top_k": TOP_K,
    }, open(out / "pi2_labels.json", "w"))
    json.dump(metrics, open(out / "pi2_metrics.json", "w"), indent=1)
    print("saved pi2_kind/move/switch + labels + metrics", flush=True)

    sanity_check(out)


def sanity_check(out):
    """Load the bundle via sb.pi2 and score a few real states."""
    print("\nsanity check via sb.pi2:", flush=True)
    import sb.pi2 as pi2
    bundle = pi2.load(out)
    samples = []
    with open(ROWS) as f:
        for i, line in enumerate(f):
            if i in (1, 100, 100000):
                samples.append(json.loads(line))
            if i > 100000:
                break
    meta = json.load(open(out / "pi2_labels.json"))
    for r in samples:
        state = r.get("state_pre") or r["state"]
        known = state["my"]["active"].get("moves") or []
        legal = ["move:" + m for m in known]
        own = f"{r['action_kind']}:{r['action_id']}"
        if own not in legal:
            legal.append(own)
        for m in meta["move"]["vocab"][:2]:
            a = "move:" + m
            if a not in legal:
                legal.append(a)
        for s in meta["switch"]["vocab"][:3]:
            a = "switch:" + s
            if a not in legal:
                legal.append(a)
        probs = pi2.action_probs(bundle, state, legal)
        total = sum(probs.values())
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        print(f"  game {r['game']} turn {r['turn']} true={own} "
              f"P(move|s)={bundle['kind'].predict(np.asarray(state_features(state)).reshape(1, -1))[0]:.3f}")
        print(f"    legal={len(legal)} sum={total:.6f}")
        for a, p in top:
            print(f"    {p:.4f}  {a}")


if __name__ == "__main__":
    main()
