"""Deck value network: how well does this deck (with these relics, potions and HP) do against
this act boss? Trained on boss fights of deck variants (bossbench.py vary), where every variant of
a save fights the same hidden futures, so differences between variants are the deck's doing.

    deckvalue.py train runs/dvA_c0.jsonl ... --out models/deckvalue1.pt [--device cuda]
    deckvalue.py cv runs/dvA_c0.jsonl ...          (held-out saves: within-save ranking accuracy)

The score a fight earns: won -> 1 + 0.25 * HP left / max HP; lost -> 0.8 * share of boss HP removed.
DeckValue(path).score(states) predicts it (the AlphaGo value network's role for deck decisions).
"""

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))


def fight_score(r):
    if r["won"]:
        return 1.0 + 0.25 * (r.get("hp_after") or 0) / max(1, r.get("mhp") or 1)
    return 0.8 * (1.0 - min(1.0, max(0.0, r.get("left_frac") or 0.0)))


def expand(paths):
    out = []
    for p in paths:
        if p.startswith("@"):
            with open(os.path.join(HERE, p[1:])) as fh:
                out += [l.strip() for l in fh if l.strip()]
        else:
            out.append(p)
    return out


def load_rows(paths):
    rows = []
    for p in expand(paths):
        with open(os.path.join(HERE, p)) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return [r for r in rows if r.get("deck")]


class Vocab:
    def __init__(self, items, min_count=1):
        cnt = Counter(items)
        self.itos = ["<pad>", "<unk>"] + sorted(k for k, v in cnt.items() if v >= min_count)
        self.stoi = {k: i for i, k in enumerate(self.itos)}

    def __call__(self, k):
        return self.stoi.get(k, 1)

    def __len__(self):
        return len(self.itos)


class Encoder:
    """Run state -> padded index arrays + scalars."""

    def __init__(self, rows=None, blob=None):
        if blob is not None:
            self.cards, self.relics, self.pots, self.bosses = (self._v(blob[k]) for k in ("cards", "relics", "pots", "bosses"))
            return
        self.cards = Vocab([c for r in rows for c in r["deck"]], 2)
        self.relics = Vocab([x for r in rows for x in r.get("relics") or []], 2)
        self.pots = Vocab([x for r in rows for x in r.get("potions") or []], 2)
        self.bosses = Vocab([str(r.get("boss")) for r in rows], 1)

    @staticmethod
    def _v(itos):
        v = Vocab([])
        v.itos, v.stoi = itos, {k: i for i, k in enumerate(itos)}
        return v

    def to_json(self):
        return {"cards": self.cards.itos, "relics": self.relics.itos, "pots": self.pots.itos, "bosses": self.bosses.itos}

    def encode(self, rows):
        n = len(rows)
        lc = max(1, max(len(r["deck"]) for r in rows))
        lr = max(1, max(len(r.get("relics") or []) for r in rows))
        lp = max(1, max(len(r.get("potions") or []) for r in rows))
        C = np.zeros((n, lc), np.int64)
        U = np.zeros((n, lc), np.float32)
        R = np.zeros((n, lr), np.int64)
        P = np.zeros((n, lp), np.int64)
        B = np.zeros(n, np.int64)
        S = np.zeros((n, 8), np.float32)
        for i, r in enumerate(rows):
            for j, c in enumerate(r["deck"]):
                C[i, j] = self.cards(c)
                U[i, j] = 1.0 if c.endswith("+") else 0.0
            for j, x in enumerate(r.get("relics") or []):
                R[i, j] = self.relics(x)
            for j, x in enumerate(r.get("potions") or []):
                P[i, j] = self.pots(x)
            B[i] = self.bosses(str(r.get("boss")))
            hp, mhp = r.get("hp") or 0, max(1, r.get("mhp") or 1)
            act = r.get("act") or 1
            S[i] = [hp / 100.0, mhp / 100.0, hp / mhp, len(r["deck"]) / 30.0, len(r.get("potions") or []) / 3.0,
                    act == 1, act == 2, act == 3]
        return [torch.from_numpy(x) for x in (C, U, R, P, B, S)]


class DeckNet(nn.Module):
    """DeepSets over the deck (per-card MLP, sum and mean pooled), bags of relics and potions,
    a boss embedding and scalars -> (win logit, score)."""

    def __init__(self, enc, d=48, h=192):
        super().__init__()
        self.card = nn.Embedding(len(enc.cards), d, padding_idx=0)
        self.up = nn.Parameter(torch.zeros(d))
        self.phi = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, 2 * d), nn.ReLU())
        self.relic = nn.EmbeddingBag(len(enc.relics), 16, mode="sum", padding_idx=0)
        self.pot = nn.EmbeddingBag(len(enc.pots), 8, mode="sum", padding_idx=0)
        self.boss = nn.Embedding(len(enc.bosses), 12)
        self.rho = nn.Sequential(nn.Linear(4 * d + 16 + 8 + 12 + 8, h), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 2))

    def forward(self, C, U, R, P, B, S):
        mask = (C > 0).float().unsqueeze(-1)
        x = self.phi(self.card(C) + U.unsqueeze(-1) * self.up) * mask
        n = mask.sum(1).clamp(min=1.0)
        pooled = torch.cat([x.sum(1) / 10.0, x.sum(1) / n], -1)
        z = torch.cat([pooled, self.relic(R), self.pot(P), self.boss(B), S], -1)
        out = self.rho(z)
        return out[:, 0], out[:, 1]


def batches(n, bs, rng):
    idx = list(range(n))
    rng.shuffle(idx)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def fit(rows, enc, epochs=60, device="cpu", seed=0, val=None, log=True):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = DeckNet(enc).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    X = [t.to(device) for t in enc.encode(rows)]
    yw = torch.tensor([float(r["won"]) for r in rows], device=device)
    ys = torch.tensor([fight_score(r) for r in rows], device=device)
    bce = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for b in batches(len(rows), 256, rng):
            b = torch.tensor(b, device=device)
            lw, s = net(*[x[b] for x in X])
            loss = bce(lw, yw[b]) + 4.0 * ((s - ys[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        if log and (ep % 10 == 9 or ep == epochs - 1):
            msg = f"epoch {ep + 1} loss {tot / len(rows):.4f}"
            if val is not None:
                msg += " | " + json.dumps(evaluate(net, enc, val, device))
            print(msg, flush=True)
    return net


def predict(net, enc, rows, device="cpu"):
    net.eval()
    out_w, out_s = [], []
    with torch.no_grad():
        for i in range(0, len(rows), 1024):
            X = [t.to(device) for t in enc.encode(rows[i:i + 1024])]
            lw, s = net(*X)
            out_w.append(torch.sigmoid(lw).cpu().numpy())
            out_s.append(s.cpu().numpy())
    return np.concatenate(out_w), np.concatenate(out_s)


def variant_means(rows, pred=None):
    """(save, variant) -> [mean observed score, mean prediction, n]."""
    g = defaultdict(lambda: [0.0, 0.0, 0])
    for i, r in enumerate(rows):
        k = (r["save"], r["variant"])
        g[k][0] += fight_score(r)
        g[k][1] += 0.0 if pred is None else float(pred[i])
        g[k][2] += 1
    return {k: (a / n, b / n, n) for k, (a, b, n) in g.items()}


def within_save_ranking(rows, pred, min_gap=0.1):
    """Share of variant pairs on the same save, observed score differing by >= min_gap, that the
    prediction orders the same way (deck decisions compare options at one state)."""
    vm = variant_means(rows, pred)
    by = defaultdict(list)
    for (s, v), x in vm.items():
        by[s].append(x)
    agree = tot = 0
    for xs in by.values():
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                d = xs[i][0] - xs[j][0]
                if abs(d) >= min_gap:
                    tot += 1
                    agree += (d > 0) == (xs[i][1] - xs[j][1] > 0)
    return round(agree / tot, 3) if tot else None, tot


def evaluate(net, enc, rows, device="cpu"):
    from sklearn.metrics import roc_auc_score
    pw, ps = predict(net, enc, rows, device)
    y = np.array([float(r["won"]) for r in rows])
    ys = np.array([fight_score(r) for r in rows])
    res = {"n": len(rows), "rank": within_save_ranking(rows, ps)[0], "rank_pairs": within_save_ranking(rows, ps)[1],
           "mse": round(float(((ps - ys) ** 2).mean()), 4)}
    if 0 < y.sum() < len(y):
        res["auc"] = round(float(roc_auc_score(y, pw)), 3)
    return res


def hp_baseline_rank(rows):
    """Reference: rank variants by HP alone (hp/mhp) and by deck size (fewer cards better)."""
    hp = np.array([(r.get("hp") or 0) / max(1, r.get("mhp") or 1) for r in rows])
    return {"rank_by_hp": within_save_ranking(rows, hp)[0], "rank_by_small_deck": within_save_ranking(rows, -np.array([len(r["deck"]) for r in rows]))[0]}


def split_by_save(rows, k, folds, seed=0):
    saves = sorted({r["save"] for r in rows})
    random.Random(seed).shuffle(saves)
    held = set(saves[k::folds])
    return [r for r in rows if r["save"] not in held], [r for r in rows if r["save"] in held]


class DeckValue:
    """Load a trained net; score(states) -> predicted fight score per state dict
    {deck, relics, potions, hp, mhp, act, boss}."""

    def __init__(self, path, device="cpu"):
        blob = torch.load(path, map_location=device, weights_only=False)
        self.enc = Encoder(blob=blob["enc"])
        self.net = DeckNet(self.enc, **blob.get("dims", {}))
        self.net.load_state_dict(blob["state"])
        self.net.eval()
        self.device = device

    def score(self, states):
        return predict(self.net, self.enc, states, self.device)[1]

    def win(self, states):
        return predict(self.net, self.enc, states, self.device)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "cv"])
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    rows = load_rows(args.data)
    print(json.dumps({"rows": len(rows), "saves": len({r['save'] for r in rows}),
                      "by_act": {a: sum(r.get('act') == a for r in rows) for a in (1, 2, 3)},
                      "win_rate": round(sum(r["won"] for r in rows) / max(1, len(rows)), 3), "device": device}), flush=True)
    if args.mode == "cv":
        res = []
        for k in range(args.folds):
            tr, va = split_by_save(rows, k, args.folds)
            enc = Encoder(tr)
            net = fit(tr, enc, args.epochs, device, seed=k, log=False)
            r = evaluate(net, enc, va, device)
            r.update(hp_baseline_rank(va))
            print(f"fold {k}: {json.dumps(r)}", flush=True)
            res.append(r)
        keys = [k for k in res[0] if isinstance(res[0][k], float)]
        print(json.dumps({"cv": {k: round(float(np.mean([r[k] for r in res if r.get(k) is not None])), 3) for k in keys}}))
        return
    tr, va = split_by_save(rows, 0, args.folds)
    enc = Encoder(rows)
    fit(tr, enc, args.epochs, device, val=va)          # report held-out quality
    net = fit(rows, enc, args.epochs, device, log=False)   # then train on everything
    if args.out:
        torch.save({"enc": enc.to_json(), "state": net.state_dict(), "dims": {}, "rows": len(rows)},
                   os.path.join(HERE, args.out))
        print(json.dumps({"saved": args.out}))


if __name__ == "__main__":
    main()
