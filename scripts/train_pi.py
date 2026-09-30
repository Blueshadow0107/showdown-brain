"""train_pi.py — policy prior: P(action | state) via behavior cloning.

Same rows/features as V(s); the label is the action the human took
("move:earthquake" / "switch:pikachu"). Multiclass LightGBM over the top-K
action vocab; renormalize over legal actions at inference time.
"""
import json
import pathlib
import sys
from collections import Counter

import lightgbm as lgb
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sb.features import build_matrix  # noqa: E402

ROWS = ROOT / "data" / "rows.jsonl"
MIN_COUNT = 40       # action must appear at least this often to get its own class
TOP_K = 500          # vocab cap


def main():
    rows = [json.loads(l) for l in open(ROWS)]
    labels = [f"{r['action_kind']}:{r['action_id']}" for r in rows]
    counts = Counter(labels)
    vocab = [a for a, c in counts.most_common(TOP_K) if c >= MIN_COUNT]
    # "other" bucket keeps long tail out of the softmax
    label_to_idx = {a: i for i, a in enumerate(vocab)}
    y = np.array([label_to_idx.get(a, len(vocab)) for a in labels], dtype="int32")
    n_class = len(vocab) + 1
    print(f"{len(rows)} rows, {n_class} classes ({len(vocab)} real + other)")

    X, _, g = build_matrix(rows)

    games = np.unique(g)
    rng = np.random.default_rng(11)
    val_games = rng.choice(games, size=int(0.2 * len(games)), replace=False)
    val_mask = np.isin(g, val_games)
    dtrain = lgb.Dataset(X[~val_mask], y[~val_mask])
    dval = lgb.Dataset(X[val_mask], y[val_mask], reference=dtrain)

    params = {"objective": "multiclass", "num_class": n_class,
              "metric": ["multi_logloss"], "learning_rate": 0.03,
              "num_leaves": 63, "min_data_in_leaf": 100,
              "feature_fraction": 0.7, "num_threads": 4, "verbose": -1}
    model = lgb.train(params, dtrain, num_boost_round=1500, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])

    p = model.predict(X[val_mask])
    top1 = int((p.argmax(1) == y[val_mask]).sum())
    top3 = int(((np.argsort(-p, axis=1)[:, :3]) == y[val_mask][:, None]).any(1).sum())
    n = int(val_mask.sum())
    other_share = float((y[val_mask] == len(vocab)).mean())

    out = ROOT / "models"
    model.save_model(str(out / "pi_model.txt"))
    json.dump({"vocab": vocab, "other_idx": len(vocab),
               "min_count": MIN_COUNT, "top_k": TOP_K},
              open(out / "pi_labels.json", "w"))
    metrics = {"rows": len(rows), "val_rows": n, "classes": n_class,
               "top1_acc": round(top1 / n, 4), "top3_acc": round(top3 / n, 4),
               "other_label_share": round(other_share, 4),
               "base_rate_top_class": round(float((y[val_mask] == y[val_mask].argmax()).mean()), 4),
               "best_iter": model.best_iteration}
    json.dump(metrics, open(out / "pi_metrics.json", "w"), indent=1)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
