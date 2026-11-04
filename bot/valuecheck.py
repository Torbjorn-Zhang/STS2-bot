"""Can the unified network's value head stand in for MC plays? On recorded runs (run.py --record
rows with their "o" outcome and its --mc-k plays), at each run's first decision of acts 2 and 3:
how well do these predict the run's real win?
  value   the value head (it fits the RL return, upolicy.rl), from that decision's state
  mc      the --mc-k plays' win share (act --mc-act only: the plays start at its first map node)
  bucket  the mean real win of the runs at the same act / floor in this data (no state at all)
Reported per act as AUC (real win vs not) and the Brier-free rank correlation. If the value head
ranks runs at least as well as the plays do, it can replace noisy playouts as a leaf estimate
(rsroll lost to playout noise); if it barely beats the bucket, it has not learned the state yet.

    valuecheck.py models/up8_unified12.pt data/xdarchive/up8_ev12/*_rec.jsonl [--acts 2,3]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from upolicy import Enc, collate, expand, load_net, load_records  # noqa: E402


def auc(score, y):
    """Mann-Whitney AUC of score for y (ties count half)."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    s = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(s):   # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_se(score, y, n=400, seed=0):
    rng, score, y = np.random.default_rng(seed), np.asarray(score, float), np.asarray(y, bool)
    vals = [auc(score[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return float(np.nanstd(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net", help="a unified network .pt (its value head)")
    ap.add_argument("files", nargs="+", help="run.py --record files (with outcome rows)")
    ap.add_argument("--acts", default="2,3")
    args = ap.parse_args()
    blob = torch.load(os.path.join(HERE, args.net), map_location="cpu", weights_only=False)
    enc = Enc(blob=blob["enc"])
    net, _ = load_net(blob)
    net.eval()
    decs, outcome = load_records(expand(args.files))
    first = {}   # (run, act) -> the run's first decision of that act
    for d in decs:
        act = d[1].get("act") or 1
        first.setdefault((d[6], act), d)
    for act in (int(a) for a in args.acts.split(",")):
        rows = [(k[0], d) for k, d in first.items() if k[1] == act and k[0] in outcome]
        if len(rows) < 20:
            print(json.dumps({"act": act, "runs": len(rows), "note": "too few"}))
            continue
        y = np.array([bool(outcome[r].get("victory")) for r, _ in rows])
        with torch.no_grad():
            v = []
            for i in range(0, len(rows), 256):
                items = [enc.encode(d[0], d[1], d[2], d[4]) for _, d in rows[i:i + 256]]
                v.append(net(collate(items, torch.device("cpu")), with_value=True)[1].numpy())
            v = np.concatenate(v)
        by = defaultdict(list)
        for (r, d), w in zip(rows, y):
            by[d[1].get("floor")].append(w)
        bucket = np.array([np.mean(by[d[1].get("floor")]) for _, d in rows])
        res = {"act": act, "runs": len(rows), "wins": int(y.sum()),
               "auc_value": round(auc(v, y), 3), "se_value": round(boot_se(v, y), 3),
               "auc_bucket": round(auc(bucket, y), 3)}
        mc = [outcome[r].get("mc") for r, _ in rows]
        have = np.array([bool(m and m.get("act") == act and m.get("win")) for m in mc])
        if have.sum() >= 20:
            m = np.array([np.mean(x["win"]) if h else np.nan for x, h in zip(mc, have)])
            k = float(np.mean([len(x["win"]) for x, h in zip(mc, have) if h]))
            res.update({"mc_runs": int(have.sum()), "mc_k": round(k, 1), "auc_mc": round(auc(m[have], y[have]), 3),
                        "se_mc": round(boot_se(m[have], y[have]), 3), "auc_value_same_runs": round(auc(v[have], y[have]), 3),
                        # does the value head add to the plays? their average rank, on the same runs
                        "auc_value_plus_mc": round(auc(np.argsort(np.argsort(v[have])) + np.argsort(np.argsort(m[have])),
                                                       y[have]), 3)})
        print(json.dumps(res))


if __name__ == "__main__":
    main()
