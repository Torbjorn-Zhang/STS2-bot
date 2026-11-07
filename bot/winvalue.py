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

    winvalue.py check models/vwin9.pt 'data/xdarchive/up9_it19/*.jsonl' ... [--policy-value]

The validity test (supervisor r13): spread and disagreement can come from noise, so on self-play
the value never saw (sampled picks, so the played option varies), every card reward gets
V_pre (its own state's value) and gap = V(best option) - V(played option), and the run's win is
regressed on both: logit P(win) = a + b * logit(V_pre) + c * gap. The value knows something
about options only if c < 0 at >= 2 SE (bootstrap over runs: a run's decisions share its label).
--policy-value: the same with the policy network's own value head (return points, not win).
"""

import argparse
import json
import re
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
            if re.search(r"/mc\d+$", str(r.get("seed"))):   # run.py --mc-record: an MC play, part of its run
                continue
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


def card_cases(net, enc, files, win):
    """-> [(run key, won, V_pre, gap, chosen is value-best)] over the card rewards whose next
    decision is in the record (valuepick.py's construction: each option's resulting state is the
    next decision's state with the deck that option leaves)."""
    import copy
    runs, won = defaultdict(list), {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if re.search(r"/mc\d+$", str(r.get("seed"))):   # run.py --mc-record: an MC play, part of its run
                continue
            key = f"{os.path.basename(f)}/{r.get('seed')}"
            if r.get("k") == "o":
                won[key] = bool(r.get("victory"))
            elif r.get("k") == "d":
                runs[key].append(r)
    scale = (lambda v: 100.0 * torch.sigmoid(v)) if win else (lambda v: 100.0 * v)
    out = []
    for key, R in runs.items():
        if key not in won:
            continue
        for i, r in enumerate(R[:-1]):
            if r.get("dec") != "card_reward":
                continue
            opts, nxt = options_of(r), R[i + 1]
            a = chosen_of(r, opts)
            if not opts or len(opts) < 2 or a is None or not 0 <= a < len(opts) or options_of(nxt) is None:
                continue
            deck0, deck1 = list(r["s"].get("deck") or []), list(nxt["s"].get("deck") or [])
            kept = [o[1] for o in opts]
            if sorted(deck0 + ([kept[a]] if kept[a] else [])) != sorted(deck1):
                continue
            items = []
            for o in opts:
                s = copy.deepcopy(nxt["s"])
                s["deck"] = deck0 + ([o[1]] if o[1] else [])
                items.append(enc.encode(kind_of(nxt), s, options_of(nxt), (nxt.get("x") or {}).get("route")))
            items.append(enc.encode("card_reward", r["s"], opts, (r.get("x") or {}).get("route")))
            with torch.no_grad():
                _, v = net(collate(items, torch.device("cpu")), with_value=True)
            v = scale(v).numpy()
            vo = v[:-1]
            out.append((key, won[key], float(v[-1]), float(vo.max() - vo[a]), int(np.argmax(vo) == a)))
    return out


def logit_fit(X, y, iters=25):
    """Logistic regression by Newton's method (ridge 1e-6) -> coefficients."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        w += np.linalg.solve(H, g)
    return w


def check(args):
    blob = torch.load(os.path.join(HERE, args.net), map_location="cpu", weights_only=False)
    enc, (net, _) = Enc(blob=blob["enc"]), load_net(blob)
    net.eval()
    win = not args.policy_value
    C = card_cases(net, enc, expand(args.files), win)
    if len(C) < 100:
        print(json.dumps({"cases": len(C)}))
        return
    keys = sorted({c[0] for c in C})
    by = defaultdict(list)
    for j, c in enumerate(C):
        by[c[0]].append(j)
    y = np.array([c[1] for c in C], float)
    vp = np.array([c[2] for c in C]) / 100.0
    gap = np.array([c[3] for c in C])
    if win:
        x1 = np.log(np.clip(vp, 1e-3, 1 - 1e-3) / np.clip(1 - vp, 1e-3, 1 - 1e-3))
    else:
        x1 = (vp - vp.mean()) / (vp.std() + 1e-9)
    X = np.stack([np.ones(len(C)), x1, gap / 10.0], 1)   # c per 10 points of gap
    w = logit_fit(X, y)
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(args.boot):
        idx = np.concatenate([by[keys[k]] for k in rng.integers(0, len(keys), len(keys))])
        boot.append(logit_fit(X[idx], y[idx])[2])
    se = float(np.std(boot))
    # the decile view: within deciles of V_pre, win rate of value-best picks vs picks >= margin below
    dec = np.quantile(x1, np.linspace(0, 1, 11)[1:-1])
    bins = np.digitize(x1, dec)
    best = np.array([c[4] for c in C], bool)
    below = gap >= args.margin
    rows, diffs = [], []
    for b in range(10):
        m = bins == b
        if (m & best).sum() >= 10 and (m & below).sum() >= 10:
            d = y[m & best].mean() - y[m & below].mean()
            diffs.append((d, (m & best).sum(), (m & below).sum()))
            rows.append({"decile": b, "n_best": int((m & best).sum()), "win_best": round(float(y[m & best].mean()), 3),
                         "n_below": int((m & below).sum()), "win_below": round(float(y[m & below].mean()), 3)})
    pooled = (sum(d * min(a, b) for d, a, b in diffs) / max(1, sum(min(a, b) for _, a, b in diffs))) if diffs else None
    res = {"net": args.net, "values": "win" if win else "return", "cases": len(C), "runs": len(keys),
           "win": round(float(y.mean()), 4), "share_value_best": round(float(best.mean()), 3),
           f"share_gap_ge_{args.margin:g}": round(float(below.mean()), 3),
           "gap_median": round(float(np.median(gap)), 2), "gap_p90": round(float(np.percentile(gap, 90)), 2),
           "coef_vpre": round(float(w[1]), 4), "coef_gap_per10": round(float(w[2]), 4), "se_gap_per10": round(se, 4),
           "z_gap": round(float(w[2]) / (se + 1e-12), 2), "pass": bool(w[2] < 0 and w[2] / (se + 1e-12) <= -2),
           "decile_best_minus_below": round(float(pooled), 4) if pooled is not None else None, "deciles": rows}
    if args.report:
        with open(os.path.join(HERE, args.report), "w") as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "deciles"}))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("net")
        ap.add_argument("files", nargs="+")
        ap.add_argument("--policy-value", action="store_true")
        ap.add_argument("--margin", type=float, default=2.0)
        ap.add_argument("--boot", type=int, default=200)
        ap.add_argument("--report", default=None)
        ap.add_argument("--threads", type=int, default=2)
        args = ap.parse_args()
        torch.set_num_threads(args.threads)
        check(args)
        return
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
