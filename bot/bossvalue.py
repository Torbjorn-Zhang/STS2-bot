"""Boss-fight value: P(win this act's boss | the deck, relics, HP and potions brought to it),
fitted on the boss fights of recorded expert-bot runs (record.py "f" header + "fe" result;
the final boss counts as won when the run is a victory).

    bossvalue.py fit DATA... --act 2 --out models/bossv_act2.pkl
prints cross-validated AUC against an HP-only baseline and the cards / relics whose presence
moves the win probability most (in the bot's own hands, not a human's).
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def expand(paths):
    out = []
    for p in paths:
        if p.startswith("@"):
            with open(os.path.join(HERE, p[1:])) as fh:
                out += [l.strip() for l in fh if l.strip()]
        else:
            out.append(p)
    return out


def boss_fights(paths, act):
    """[(run_state at the fight, enemies, won)] for the act's boss fights."""
    out = []
    for p in paths:
        heads, fights, victory = {}, [], {}
        with open(os.path.join(HERE, p)) as fh:
            for line in fh:
                r = json.loads(line)
                k = r.get("k")
                if k == "f" and (r.get("run") or {}).get("room") == "Boss" and (r["run"].get("act") == act):
                    heads[(r["seed"], r["i"])] = r
                elif k == "fe" and (r["seed"], r["i"]) in heads:
                    fights.append((heads.pop((r["seed"], r["i"])), bool(r.get("won"))))
                elif k == "o":
                    victory[r["seed"]] = bool(r.get("victory"))
        for h, won in fights:
            if act == 3:
                won = victory.get(h["seed"], won)
            out.append((h["run"], h.get("enemies") or [], won))
    return out


def featurize(run, enemies, vocab=None):
    f = Counter()
    for c in run.get("deck") or []:
        base = c.rstrip("+")
        f["card:" + base] += 1
        if c.endswith("+"):
            f["up:" + base] += 1
    for rl in run.get("relics") or []:
        f["relic:" + rl] += 1
    f["boss:" + (run.get("boss") or "?")] = 1
    f["hp"] = (run.get("hp") or 0) / 100.0
    f["hp_frac"] = (run.get("hp") or 0) / max(1, run.get("mhp") or 1)
    f["mhp"] = (run.get("mhp") or 0) / 100.0
    f["deck_size"] = len(run.get("deck") or []) / 30.0
    pots = run.get("potions")
    f["potions"] = (pots if isinstance(pots, (int, float)) else len([x for x in pots or [] if x])) / 3.0
    f["gold"] = (run.get("gold") or 0) / 300.0
    return f


class BossValue:
    """P(win the act boss) from a run state, for value-guided deck decisions (nets.CardPolicy)."""

    def __init__(self, path):
        with open(path, "rb") as fh:
            m = pickle.load(fh)
        self.vocab, self.mu, self.sd, self.clf = m["vocab"], m["mu"], m["sd"], m["clf"]
        self.idx = {k: i for i, k in enumerate(self.vocab)}

    def logits(self, run, extra_cards_list):
        """logit P(win) for the run state with each list of extra cards added to the deck."""
        X = np.zeros((len(extra_cards_list), len(self.vocab)))
        for i, extra in enumerate(extra_cards_list):
            r = dict(run)
            r["deck"] = list(run.get("deck") or []) + list(extra)
            for k, v in featurize(r, []).items():
                if k in self.idx:
                    X[i, self.idx[k]] = v
        return self.clf.decision_function((X - self.mu) / self.sd)


def explore_rows(paths):
    rows = []
    for p in paths:
        with open(os.path.join(HERE, p)) as fh:
            for line in fh:
                r = json.loads(line)
                if "deck" in r:
                    rows.append(({"deck": r["deck"], "relics": r.get("relics"), "hp": r.get("hp"), "mhp": r.get("mhp"),
                                  "potions": r.get("potions") or 0, "boss": r.get("boss"), "gold": 0}, r["seed"], bool(r["won"])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "fit-explore"])
    ap.add_argument("data", nargs="+")
    ap.add_argument("--act", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-count", type=int, default=15)
    ap.add_argument("--C", type=float, default=0.3)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    if args.mode == "fit-explore":
        ex = explore_rows(expand(args.data))
        rows = [(r, [], w) for r, _, w in ex]
        groups = [g for _, g, _ in ex]
    else:
        rows = boss_fights(expand(args.data), args.act)
        groups = None
    if len(rows) < 50:
        sys.exit(f"only {len(rows)} boss fights")
    feats = [featurize(r, e) for r, e, _ in rows]
    y = np.asarray([w for _, _, w in rows], dtype=int)
    cnt = Counter(k for f in feats for k in f)
    vocab = sorted(k for k, n in cnt.items() if n >= args.min_count or not k.split(":")[0] in ("card", "up", "relic", "boss"))
    idx = {k: i for i, k in enumerate(vocab)}
    X = np.zeros((len(feats), len(vocab)))
    for i, f in enumerate(feats):
        for k, v in f.items():
            if k in idx:
                X[i, idx[k]] = v
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    base_cols = [idx[k] for k in ("hp", "hp_frac", "mhp") if k in idx] + [idx[k] for k in vocab if k.startswith("boss:")]
    clf = LogisticRegression(C=args.C, max_iter=2000)
    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(5) if groups else 5   # explore data: held-out saves (unseen HP / relics / boss rolls)
    kw = {"groups": groups} if groups else {}
    p = cross_val_predict(clf, Xs, y, cv=cv, method="predict_proba", **kw)[:, 1]
    pb = cross_val_predict(LogisticRegression(C=args.C, max_iter=2000), Xs[:, base_cols], y, cv=cv, method="predict_proba", **kw)[:, 1]
    clf.fit(Xs, y)
    coef = dict(zip(vocab, clf.coef_[0]))
    res = {"act": args.act, "fights": len(rows), "win_rate": round(float(y.mean()), 3), "features": len(vocab),
           "auc": round(float(roc_auc_score(y, p)), 3), "auc_hp_boss_only": round(float(roc_auc_score(y, pb)), 3),
           "top": sorted(((k, round(v, 3)) for k, v in coef.items() if ":" in k and not k.startswith("boss:")),
                         key=lambda x: -x[1])[:25],
           "bottom": sorted(((k, round(v, 3)) for k, v in coef.items() if ":" in k and not k.startswith("boss:")),
                            key=lambda x: x[1])[:15],
           "scalars": {k: round(coef[k], 3) for k in ("hp", "hp_frac", "mhp", "deck_size", "potions", "gold") if k in coef}}
    print(json.dumps(res, indent=1))
    if args.out:
        with open(os.path.join(HERE, args.out), "wb") as fh:
            pickle.dump({"vocab": vocab, "mu": mu, "sd": sd, "clf": clf, "act": args.act}, fh)
    print(json.dumps({"auc": res["auc"], "auc_base": res["auc_hp_boss_only"], "fights": res["fights"]}))


if __name__ == "__main__":
    main()
