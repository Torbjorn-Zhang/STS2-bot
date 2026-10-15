"""Train the value model on collected self-play data.

usage: train.py data/d0.jsonl [data/d1.jsonl ...] [--out models/value.pkl]
Splits by seed (runs never straddle train/test) and reports how well the model ranks
end-of-turn states compared with the hand-written evaluation's main term (hp_after).
"""

import argparse
import json
import os
import pickle
import random

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score

from value import DEATH_VALUE, FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += [json.loads(l) for l in fh]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "models", "value.pkl"))
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    rows = load(args.data)
    seeds = sorted({r["seed"] for r in rows})
    random.Random(0).shuffle(seeds)
    test_seeds = set(seeds[: int(len(seeds) * args.test_frac)])
    tr = [r for r in rows if r["seed"] not in test_seeds]
    te = [r for r in rows if r["seed"] in test_seeds]
    Xtr, ytr = np.array([r["x"] for r in tr]), np.array([r["y"] for r in tr])
    Xte, yte = np.array([r["x"] for r in te]), np.array([r["y"] for r in te])

    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    hp_after = Xte[:, FEATURES.index("hp_after")]
    died = yte == DEATH_VALUE
    report = {
        "samples": len(rows), "train": len(tr), "test": len(te), "test_runs": len(test_seeds),
        "mae_model": round(float(mean_absolute_error(yte, pred)), 2),
        "mae_hp_after_baseline": round(float(mean_absolute_error(yte, hp_after)), 2),
        "corr_model": round(float(np.corrcoef(yte, pred)[0, 1]), 3),
        "corr_hp_after": round(float(np.corrcoef(yte, hp_after)[0, 1]), 3),
    }
    if died.any() and (~died).any():
        report["death_auc_model"] = round(float(roc_auc_score(died, -pred)), 3)
        report["death_auc_hp_after"] = round(float(roc_auc_score(died, -hp_after)), 3)
    boss = Xte[:, FEATURES.index("is_boss")] == 1
    if boss.sum() > 20 and died[boss].any() and (~died[boss]).any():
        report["boss_death_auc_model"] = round(float(roc_auc_score(died[boss], -pred[boss])), 3)
        report["boss_death_auc_hp_after"] = round(float(roc_auc_score(died[boss], -hp_after[boss])), 3)
    print(json.dumps(report, indent=1))

    # refit on everything for use
    model.fit(np.array([r["x"] for r in rows]), np.array([r["y"] for r in rows]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"model": model, "features": FEATURES, "data": args.data, "report": report}, fh)
    print("saved", args.out)


if __name__ == "__main__":
    main()
