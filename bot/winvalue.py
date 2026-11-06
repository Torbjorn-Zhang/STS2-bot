"""Win-only value network (line ③): the unified policy network's trunk and value head refitted to
P(win) alone, the AlphaGo value network's target, instead of the RL return (progress, boss bonus,
floor share) the policy's value head bootstraps.

    winvalue.py train @list.txt --init models/up9_unified18.pt --out models/vwin9.pt \
        --test 'data/xdarchive/up9_ev18/*_rec.jsonl' --report runs/vwin9.json

Training rows: up to --per-run decisions of each self-play run (a run's decisions share one label,
so more of them add little), the label its victory. Held out: 10 % of the training runs, and
--test (greedy A10 evaluation records of a later network: other seeds' outcomes under the policy
the value would serve). Reported per act: AUC, log-loss and Brier against the (ascension, act,
floor) win-rate table of the training runs, the baseline any useful value must beat. Then the
option-level check (valuepick.py --win on the --test records): how often the value's best
card-reward option is the pick, how far apart the options are in win points.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from upolicy import KINDS, Enc, chosen_of, collate, expand, kind_of, load_net, options_of  # noqa: E402


def read_runs(files, per_run, rng):
    """-> [(kind, s, opts, route, run key, won)], up to per_run decisions per run (all if 0)."""
    out = []
    for f in files:
        runs, won = defaultdict(list), {}
        for line in open(f):
            r = json.loads(line)
            key = f"{os.path.basename(f)}/{r.get('seed')}"
            if r.get("k") == "o":
                won[key] = bool(r.get("victory"))
                continue
            if r.get("k") != "d" or kind_of(r) not in KINDS or not r.get("s"):
                continue
            opts = options_of(r)
            if not opts or len(opts) < 2 or chosen_of(r, opts) is None:
                continue
            runs[key].append((kind_of(r), r["s"], opts, (r.get("x") or {}).get("route"), key))
        for key, ds in runs.items():
            if key not in won:
                continue
            if per_run and len(ds) > per_run:
                ds = rng.sample(ds, per_run)
            out += [d + (won[key],) for d in ds]
    return out


def auc(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, float)
    if y.all() or not y.any():
        return None
    r = np.empty(len(p))
    r[np.argsort(p, kind="mergesort")] = np.arange(1, len(p) + 1)
    for v in np.unique(p):   # ties: average rank
        m = p == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def scores(rows, p, table):
    """rows' metrics against the (asc, act, floor) table: per act and overall."""
    y = np.array([d[5] for d in rows], float)
    b = np.array([table.get((d[1].get("asc"), d[1].get("act"), d[1].get("floor")),
                            table.get((d[1].get("asc"), d[1].get("act")), 0.2)) for d in rows])
    act = np.array([d[1].get("act") or 1 for d in rows])
    ll = lambda q, m: float(-np.mean(y[m] * np.log(np.clip(q[m], 1e-4, 1)) + (1 - y[m]) * np.log(np.clip(1 - q[m], 1e-4, 1))))
    res = {}
    for name, m in [("all", np.ones(len(y), bool))] + [(f"act{a}", act == a) for a in (1, 2, 3)]:
        if m.sum() < 50:
            continue
        res[name] = {"n": int(m.sum()), "win": round(float(y[m].mean()), 4),
                     "auc": round(auc(y[m], p[m]) or 0, 4), "auc_table": round(auc(y[m], b[m]) or 0, 4),
                     "ll": round(ll(p, m), 4), "ll_table": round(ll(b, m), 4),
                     "brier": round(float(np.mean((p[m] - y[m]) ** 2)), 4),
                     "brier_table": round(float(np.mean((b[m] - y[m]) ** 2)), 4)}
    return res


def predict(net, items, dev):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(items), 512):
            _, v = net(collate(items[i:i + 512], dev), with_value=True)
            out.append(torch.sigmoid(v).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train(args):
    files = []
    for p in args.files:
        if p.startswith("@"):
            files += [l.strip() for l in open(os.path.join(HERE, p[1:])) if l.strip()]
        else:
            files.append(p)
    rng = random.Random(args.seed)
    rows = read_runs(expand(files), args.per_run, rng)
    keys = sorted({d[4] for d in rows})
    rng.shuffle(keys)
    va_keys = set(keys[:max(1, len(keys) // 10)])
    tr = [d for d in rows if d[4] not in va_keys]
    va = [d for d in rows if d[4] in va_keys]
    test = read_runs(expand([args.test]), 0, rng) if args.test else []
    # the table baseline: win rate by (asc, act, floor) over the training runs, (asc, act) where thin
    cnt = defaultdict(lambda: [0, 0])
    for d in tr:
        for k in ((d[1].get("asc"), d[1].get("act"), d[1].get("floor")), (d[1].get("asc"), d[1].get("act"))):
            cnt[k][0] += d[5]
            cnt[k][1] += 1
    table = {k: (w + 0.2 * 5) / (n + 5) for k, (w, n) in cnt.items() if n >= (20 if len(k) == 3 else 1)}
    blob = torch.load(os.path.join(HERE, args.init), map_location="cpu", weights_only=False)
    enc = Enc(blob=blob["enc"])
    net, kinds = load_net(blob)
    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    net.to(dev)
    enc_items = lambda R: [enc.encode(d[0], d[1], d[2], d[3]) for d in R]
    I_tr, I_va, I_te = enc_items(tr), enc_items(va), enc_items(test)
    Y = torch.tensor([float(d[5]) for d in tr], device=dev)
    print(json.dumps({"files": len(files), "train_rows": len(tr), "val_rows": len(va), "test_rows": len(test),
                      "runs": len(keys), "train_win": round(float(Y.mean()), 4), "device": str(dev)}), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    yva = np.array([d[5] for d in va], float)
    best = (float("inf"), None, None)
    idx = list(range(len(tr)))
    for ep in range(args.epochs):
        net.train()
        random.Random(ep).shuffle(idx)
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            _, v = net(collate([I_tr[j] for j in b], dev), with_value=True)
            loss = bce(v, Y[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        p = predict(net, I_va, dev)
        ll = float(-np.mean(yva * np.log(np.clip(p, 1e-4, 1)) + (1 - yva) * np.log(np.clip(1 - p, 1e-4, 1))))
        print(json.dumps({"epoch": ep + 1, "val_ll": round(ll, 4), "val_auc": round(auc(yva, p) or 0, 4)}), flush=True)
        if ll < best[0]:
            best = (ll, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, ep + 1)
    net.load_state_dict(best[1])
    rep = {"init": args.init, "best_epoch": best[2], "train_rows": len(tr), "runs": len(keys),
           "val": scores(va, predict(net, I_va, dev), table)}
    if test:
        pt = predict(net, I_te, dev)
        rep["test"] = scores(test, pt, table)
        # the policy's own value head (fits return / 100), as a ranking: is the win-only refit any better?
        net0, _ = load_net(blob)
        net0.to(dev)
        net0.eval()
        with torch.no_grad():
            v0 = np.concatenate([net0(collate(I_te[i:i + 512], dev), with_value=True)[1].cpu().numpy()
                                 for i in range(0, len(I_te), 512)])
        y = np.array([d[5] for d in test], bool)
        act = np.array([d[1].get("act") or 1 for d in test])
        rep["test_auc_policy_value"] = {f"act{a}": round(auc(y[act == a], v0[act == a]) or 0, 4) for a in (1, 2, 3)}
        rep["test_auc_policy_value"]["all"] = round(auc(y, v0) or 0, 4)
    torch.save({"enc": blob["enc"], "state": best[1], "kinds": kinds, "win": True}, os.path.join(HERE, args.out))
    if args.test:
        r = subprocess.run([sys.executable, os.path.join(HERE, "valuepick.py"), args.out, args.test, "--win"],
                           capture_output=True, text=True)
        try:
            rep["valuepick"] = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            rep["valuepick"] = {"error": (r.stderr or r.stdout)[-500:]}
    if args.report:
        with open(os.path.join(HERE, args.report), "w") as fh:
            json.dump(rep, fh, indent=1)
    print(json.dumps({"saved": args.out, "best_epoch": best[2],
                      "test_all": (rep.get("test") or {}).get("all"), "valuepick": rep.get("valuepick")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--per-run", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    train(args)


if __name__ == "__main__":
    main()
