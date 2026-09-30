"""train_v_doubles.py — V(s) for gen9 random doubles. LightGBM, grouped split."""
import json
import pathlib
import sys

import lightgbm as lgb
import numpy as np
from sklearn.metrics import auc, log_loss, roc_curve

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sb.features_doubles import build_matrix_d  # noqa: E402

ROWS = ROOT / "data" / "rows_doubles.jsonl"


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
    rows = [json.loads(l) for l in open(ROWS)]
    print(f"{len(rows)} rows")
    X, y, g = build_matrix_d(rows)

    games = np.unique(g)
    rng = np.random.default_rng(7)
    val_games = rng.choice(games, size=int(0.2 * len(games)), replace=False)
    val_mask = np.isin(g, val_games)
    Xtr, ytr = X[~val_mask], y[~val_mask]
    Xva, yva = X[val_mask], y[val_mask]

    dtrain = lgb.Dataset(Xtr, ytr)
    dval = lgb.Dataset(Xva, yva, reference=dtrain)
    params = {"objective": "binary", "metric": ["binary_logloss", "auc"],
              "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 50,
              "feature_fraction": 0.8, "num_threads": 4, "verbose": -1}
    model = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])

    p = model.predict(Xva)
    fpr, tpr, _ = roc_curve(yva, p)
    metrics = {
        "rows": int(len(rows)), "games": int(len(games)),
        "val_rows": int(val_mask.sum()),
        "auc": round(auc(fpr, tpr), 4),
        "logloss": round(log_loss(yva, np.clip(p, 1e-6, 1 - 1e-6)), 4),
        "base_rate": round(float(yva.mean()), 4),
        "best_iter": model.best_iteration,
        "calibration": calibration(yva, p),
    }
    out = ROOT / "models"
    model.save_model(str(out / "v_model_doubles.txt"))
    json.dump(metrics, open(out / "v_metrics_doubles.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "calibration"}, indent=1))
    print("calibration (pred vs actual win rate):")
    for c in metrics["calibration"]:
        print(f"  {c['bin']:>4}: n={c['n']:>5}  pred={c['pred']:.2f}  actual={c['actual']:.2f}")


if __name__ == "__main__":
    main()
