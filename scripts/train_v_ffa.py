"""train_v_ffa.py — V(s) for gen9 FREE-FOR-ALL random battles on the v2 state.
Grouped split by game id (seed 7); val games come from the public corpus only
— personal rows train but never grade. Writes models/v_ffa.txt + metrics."""
import argparse
import json
import pathlib
import sys

import lightgbm as lgb
import numpy as np
from sklearn.metrics import auc, log_loss, roc_curve

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sb.features_v2 import build_matrix_v2  # noqa: E402

PUBLIC = ROOT / "data" / "rows_ffa.jsonl"
PERSONAL = ROOT / "data" / "rows_personal_ffa.jsonl"


def calibration(y_true, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            out.append({"bin": round(float(edges[i] + edges[i + 1]) / 2, 2),
                        "n": int(m.sum()), "actual": round(float(y_true[m].mean()), 3),
                        "pred": round(float(p[m].mean()), 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action=argparse.BooleanOptionalAction, default=True,
                    help="train on data/rows_ffa.jsonl (default: on)")
    ap.add_argument("--personal", action=argparse.BooleanOptionalAction, default=True,
                    help="train on data/rows_personal_ffa.jsonl (default: on)")
    args = ap.parse_args()

    pub, pers = [], []
    if args.public:
        pub = [json.loads(l) for l in open(PUBLIC)]
    if args.personal:
        if PERSONAL.exists():
            pers = [json.loads(l) for l in open(PERSONAL)]
        else:
            print(f"warning: {PERSONAL.name} absent — continuing without personal rows")
    rows = pub + pers
    if not rows:
        sys.exit("no rows: both corpora empty or disabled")
    print(f"{len(rows)} rows ({len(pub)} public, {len(pers)} personal)")
    X, y, g = build_matrix_v2(rows)

    games = np.unique(g)
    pub_games = np.unique([r["game"] for r in pub])
    if len(pub_games):
        val_pool, val_source = pub_games, "public"
    else:
        val_pool, val_source = games, "personal"
        print("note: no public rows — val split falls back to personal games")
    rng = np.random.default_rng(7)
    if len(games) > 1:
        n_val = min(max(1, int(0.2 * len(val_pool))), len(games) - 1)
        val_games = rng.choice(val_pool, size=n_val, replace=False)
        val_mask = np.isin(g, val_games)
    else:
        print("note: too few games for a holdout — evaluating in-sample")
        val_mask = np.ones(len(rows), dtype=bool)

    if val_mask.all():
        dtrain = lgb.Dataset(X, y)
    else:
        dtrain = lgb.Dataset(X[~val_mask], y[~val_mask])
    dval = lgb.Dataset(X[val_mask], y[val_mask], reference=dtrain)
    params = {"objective": "binary", "metric": ["binary_logloss", "auc"],
              "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 50,
              "feature_fraction": 0.8, "num_threads": 4, "verbose": -1}
    model = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

    p = model.predict(X[val_mask])
    fpr, tpr, _ = roc_curve(y[val_mask], p)
    metrics = {
        "rows": int(len(rows)), "public_rows": len(pub), "personal_rows": len(pers),
        "games": int(len(games)), "val_rows": int(val_mask.sum()),
        "val_source": val_source, "dim": int(X.shape[1]),
        "auc": round(auc(fpr, tpr), 4),
        "logloss": round(log_loss(y[val_mask], np.clip(p, 1e-6, 1 - 1e-6)), 4),
        "base_rate": round(float(y[val_mask].mean()), 4),
        "best_iter": model.best_iteration,
        "calibration": calibration(y[val_mask], p),
    }
    out = ROOT / "models"
    model.save_model(str(out / "v_ffa.txt"))
    json.dump(metrics, open(out / "v_ffa_metrics.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "calibration"}, indent=1))


if __name__ == "__main__":
    main()
