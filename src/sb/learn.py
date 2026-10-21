"""learn.py — the bot learns from the games IT plays.

After any battle (vs human, vs baseline, vs itself), dump the raw protocol
log and fold the parsed decision rows into a personal corpus. Retraining on
base + personal rows is one command — the listener triggers it between games.

    from sb.learn import dump_battle, learn_new_games
    dump_battle(battle)                 # after a battle finishes
    learn_new_games()                   # parse -> append -> retrain V(s)
"""
import json
import pathlib

from sb import features, features_doubles, features_v2
from sb.parser import parse_log
from sb.parser_doubles import parse_log as parse_log_doubles
from sb.parser_ffa import parse_log as parse_log_ffa

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
GAMES = ROOT / "data" / "learned_games"
PERSONAL = ROOT / "data" / "rows_personal.jsonl"
PERSONAL_D = ROOT / "data" / "rows_personal_doubles.jsonl"
PERSONAL_FFA = ROOT / "data" / "rows_personal_ffa.jsonl"
DOUBLES_BASE = ROOT / "data" / "rows_v2_doubles.jsonl"
SEEN = ROOT / "data" / "learned_games" / "seen.json"


def dump_battle(battle, games_dir: pathlib.Path | None = None) -> pathlib.Path | None:
    """Write a finished battle's raw protocol log for later learning.
    Returns the path, or None if no replay data was captured."""
    data = getattr(battle, "_replay_data", None)
    if not data:
        return None
    games_dir = games_dir or GAMES
    games_dir.mkdir(parents=True, exist_ok=True)
    path = games_dir / f"{battle.battle_tag}.log"
    with path.open("w") as f:
        for split in data:
            f.write("|".join(str(p) for p in split) + "\n")
    return path


def _seen() -> set:
    return set(json.load(open(SEEN))) if SEEN.exists() else set()


def learn_new_games(retrain: bool = True, games_dir: pathlib.Path | None = None) -> int:
    """Parse unlearned game logs -> append rows to the personal corpora ->
    optionally retrain the affected V models on base+personal rows.

    Retraining rebuilds from scratch (deterministic, ~40s per model) —
    incremental LightGBM updates drift; a rebuild keeps the morning honest.
    Logs route to a parser/corpus by room tag (freeforall / doubles / singles),
    so each format only ever trains on its own rows.
    Returns the number of new games folded in."""
    games_dir = games_dir or GAMES
    seen = _seen()
    new_rows, new_rows_d, new_rows_f, new_games = [], [], [], []
    for path in sorted(games_dir.glob("*.log")):
        tag = path.stem
        if tag in seen:
            continue
        text = path.read_text(errors="replace")
        if "freeforall" in tag:
            rows = parse_log_ffa(text, tag)
            target = new_rows_f
        elif "doubles" in tag:
            rows = parse_log_doubles(text, tag)
            target = new_rows_d
        else:
            rows = parse_log(text, tag)
            target = new_rows
        if not rows:  # unfinished or unparseable game
            continue
        target.extend(rows)
        new_games.append(tag)
    if not new_games:
        return 0
    if new_rows:
        PERSONAL.parent.mkdir(parents=True, exist_ok=True)
        with PERSONAL.open("a") as f:
            for r in new_rows:
                f.write(json.dumps(r) + "\n")
    if new_rows_d:
        with PERSONAL_D.open("a") as f:
            for r in new_rows_d:
                f.write(json.dumps(r) + "\n")
    if new_rows_f:
        with PERSONAL_FFA.open("a") as f:
            for r in new_rows_f:
                f.write(json.dumps(r) + "\n")
    seen.update(new_games)
    json.dump(sorted(seen), open(SEEN, "w"))
    if retrain:
        if new_rows:
            retrain_v2()
        if new_rows_d:
            retrain_v_doubles()
        if new_rows_f:
            retrain_v_ffa()
    return len(new_games)


def retrain_v2() -> dict:
    """Rebuild V_v2 on the public corpus + everything the bot has played.
    Mirrors train_v_v2.py but with the personal rows appended; the eval split
    stays public-only so personal games never grade their own homework."""
    import lightgbm as lgb
    import numpy as np
    from sklearn.metrics import auc, roc_curve

    features._load()
    base = [json.loads(l) for l in open(ROOT / "data" / "rows_v2.jsonl")]
    pers = [json.loads(l) for l in open(PERSONAL)] if PERSONAL.exists() else []
    rows = base + pers
    X, y, g = features_v2.build_matrix_v2(rows)

    base_games = np.unique([r["game"] for r in base])
    rng = np.random.default_rng(7)
    val_games = set(rng.choice(base_games, size=int(0.2 * len(base_games)),
                               replace=False))
    val_mask = np.array([gid in val_games for gid in g])

    dtrain = lgb.Dataset(X[~val_mask], y[~val_mask])
    dval = lgb.Dataset(X[val_mask], y[val_mask], reference=dtrain)
    params = {"objective": "binary", "metric": ["binary_logloss", "auc"],
              "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 50,
              "feature_fraction": 0.8, "num_threads": 4, "verbose": -1}
    model = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    p = model.predict(X[val_mask])
    fpr, tpr, _ = roc_curve(y[val_mask], p)
    metrics = {"rows": int(len(rows)), "personal_rows": len(pers),
               "auc": round(auc(fpr, tpr), 4), "best_iter": model.best_iteration}
    tmp = ROOT / "models" / "v2_model.txt.tmp"
    model.save_model(str(tmp))
    tmp.replace(ROOT / "models" / "v2_model.txt")  # atomic: live listener never
    json.dump(metrics, open(ROOT / "models" / "v2_metrics.json", "w"), indent=1)
    return metrics


def retrain_v_doubles() -> dict:
    """Rebuild the doubles V on the public doubles corpus + personal doubles
    rows; eval split stays public-only (mirrors retrain_v2)."""
    import lightgbm as lgb
    import numpy as np
    from sklearn.metrics import auc, roc_curve

    base = [json.loads(l) for l in open(DOUBLES_BASE)]
    pers = [json.loads(l) for l in open(PERSONAL_D)] if PERSONAL_D.exists() else []
    rows = base + pers
    X, y, g = features_doubles.build_matrix_d(rows)

    base_games = np.unique([r["game"] for r in base])
    rng = np.random.default_rng(7)
    val_games = set(rng.choice(base_games, size=int(0.2 * len(base_games)),
                               replace=False))
    val_mask = np.array([gid in val_games for gid in g])

    dtrain = lgb.Dataset(X[~val_mask], y[~val_mask])
    dval = lgb.Dataset(X[val_mask], y[val_mask], reference=dtrain)
    params = {"objective": "binary", "metric": ["binary_logloss", "auc"],
              "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 50,
              "feature_fraction": 0.8, "num_threads": 4, "verbose": -1}
    model = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    p = model.predict(X[val_mask])
    fpr, tpr, _ = roc_curve(y[val_mask], p)
    metrics = {"rows": int(len(rows)), "personal_rows": len(pers),
               "auc": round(auc(fpr, tpr), 4), "best_iter": model.best_iteration}
    tmp = ROOT / "models" / "v_model_doubles.txt.tmp"
    model.save_model(str(tmp))
    tmp.replace(ROOT / "models" / "v_model_doubles.txt")
    json.dump(metrics, open(ROOT / "models" / "v_metrics_doubles.json", "w"), indent=1)
    return metrics


def retrain_v_ffa() -> dict:
    """Rebuild the FFA V on the public FFA corpus + personal FFA rows; eval
    split stays public-only (mirrors retrain_v2)."""
    import lightgbm as lgb
    import numpy as np
    from sklearn.metrics import auc, roc_curve

    base = [json.loads(l) for l in open(ROOT / "data" / "rows_ffa.jsonl")]
    pers = [json.loads(l) for l in open(PERSONAL_FFA)] if PERSONAL_FFA.exists() else []
    rows = base + pers
    X, y, g = features_v2.build_matrix_v2(rows)

    base_games = np.unique([r["game"] for r in base])
    rng = np.random.default_rng(7)
    if len(base_games) > 1:
        val_games = set(rng.choice(base_games,
                                   size=max(1, int(0.2 * len(base_games))),
                                   replace=False))
        val_mask = np.array([gid in val_games for gid in g])
        dtrain = lgb.Dataset(X[~val_mask], y[~val_mask])
    else:  # too few public games for a holdout — evaluate in-sample
        val_mask = np.ones(len(rows), dtype=bool)
        dtrain = lgb.Dataset(X, y)
    dval = lgb.Dataset(X[val_mask], y[val_mask], reference=dtrain)
    params = {"objective": "binary", "metric": ["binary_logloss", "auc"],
              "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 50,
              "feature_fraction": 0.8, "num_threads": 4, "verbose": -1}
    model = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    p = model.predict(X[val_mask])
    fpr, tpr, _ = roc_curve(y[val_mask], p)
    metrics = {"rows": int(len(rows)), "personal_rows": len(pers),
               "auc": round(auc(fpr, tpr), 4), "best_iter": model.best_iteration}
    tmp = ROOT / "models" / "v_ffa.txt.tmp"
    model.save_model(str(tmp))
    tmp.replace(ROOT / "models" / "v_ffa.txt")
    json.dump(metrics, open(ROOT / "models" / "v_ffa_metrics.json", "w"), indent=1)
    return metrics
