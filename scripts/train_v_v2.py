"""train_v_v2.py — V(s) over the full-fidelity v2 state. Same grouped split
as train_v.py (seed 7); writes models/v2_model.txt + metrics."""
import json
import pathlib
import sys

import lightgbm as lgb
import numpy as np
from sklearn.metrics import auc, log_loss, roc_curve

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sb.features_v2 import build_matrix_v2  # noqa: E402

ROWS = ROOT / "data" / "rows_v2.jsonl"


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
    X, y, g = build_matrix_v2(rows)

    games = np.unique(g)
    rng = np.random.default_rng(7)
    val_games = rng.choice(games, size=int(0.2 * len(games)), replace=False)
    val_mask = np.isin(g, val_games)

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
        "rows": int(len(rows)), "games": int(len(games)),
        "val_rows": int(val_mask.sum()), "dim": int(X.shape[1]),
        "auc": round(auc(fpr, tpr), 4),
        "logloss": round(log_loss(y[val_mask], np.clip(p, 1e-6, 1 - 1e-6)), 4),
        "base_rate": round(float(y[val_mask].mean()), 4),
        "best_iter": model.best_iteration,
        "calibration": calibration(y[val_mask], p),
    }
    out = ROOT / "models"
    model.save_model(str(out / "v2_model.txt"))
    json.dump(metrics, open(out / "v2_metrics.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "calibration"}, indent=1))


if __name__ == "__main__":
    main()
