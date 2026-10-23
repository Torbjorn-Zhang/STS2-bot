"""Transformer card-reward policy, trained offline by advantage-weighted imitation (AWR) on every
self-play record we have, on a GPU.

Tokens: one state token (act, floor, ascension, HP share, gold, boss, room), one per deck card
(id + upgraded), relic and potion tokens, and one per option (each offered card, or skip). All
tokens attend to each other (deck synergies of any size, options against the deck), and a linear
head scores the option tokens. Training imitates the recorded choices weighted by
exp(advantage / beta): advantage = the run's return (nets.final_progress, STS2_RETURN) minus the
mean return of runs at the same (ascension, act, floor). Choices that did better than usual there
weigh more: offline policy improvement from all past play (AlphaGo's SL stage, with outcomes).

  tcard.py train FILES... --out models/tcard1.pt [--device cuda] [--epochs 6]
  TCardPolicy(path).scores(st) -> (actions, logits), like nets.CardPolicy (policy.load_deck_nets "card_t")
"""
import argparse
import glob
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


class Vocab:
    def __init__(self, items, min_count=2):
        cnt = Counter(items)
        self.itos = ["<pad>", "<unk>"] + sorted(k for k, v in cnt.items() if v >= min_count)
        self.stoi = {k: i for i, k in enumerate(self.itos)}

    def __call__(self, k):
        return self.stoi.get(k, 1)

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def of(itos):
        v = Vocab([])
        v.itos, v.stoi = list(itos), {k: i for i, k in enumerate(itos)}
        return v


def base(tok):
    return (tok or "").rstrip("+")


class Enc:
    MAXD, MAXR, MAXP = 60, 40, 5

    def __init__(self, items=None, blob=None):
        if blob is not None:
            self.cards, self.relics, self.pots, self.bosses = (Vocab.of(blob[k]) for k in ("cards", "relics", "pots", "bosses"))
            return
        s_list = [s for s, _ in items]
        self.cards = Vocab([base(c) for s, opts in items for c in (s.get("deck") or []) + [o for o in opts if o]])
        self.relics = Vocab([r for s in s_list for r in s.get("relics") or []])
        self.pots = Vocab([p for s in s_list for p in s.get("pots") or []])
        self.bosses = Vocab([str(s.get("boss")) for s in s_list], 1)

    def to_json(self):
        return {k: getattr(self, k).itos for k in ("cards", "relics", "pots", "bosses")}

    def encode(self, s, opts):
        """-> dict of arrays for one decision: deck, relics, pots ids, option ids/upg/skip, scalars."""
        deck = (s.get("deck") or [])[:self.MAXD]
        return {"d": [self.cards(base(c)) for c in deck], "du": [float(c.endswith("+")) for c in deck],
                "r": [self.relics(x) for x in (s.get("relics") or [])[:self.MAXR]],
                "p": [self.pots(x) for x in (s.get("pots") or [])[:self.MAXP]],
                "o": [self.cards(base(o)) if o else 0 for o in opts], "ou": [float(bool(o) and o.endswith("+")) for o in opts],
                "os": [float(o is None) for o in opts], "b": self.bosses(str(s.get("boss"))),
                "x": [(s.get("act") or 1) / 3, (s.get("floor") or 0) / 17, (s.get("asc") or 0) / 10,
                      (s.get("hp") or 0) / max(1, s.get("mhp") or 1), (s.get("mhp") or 80) / 100, (s.get("gold") or 0) / 300,
                      float(s.get("room") == "Elite"), float(s.get("room") == "Boss"), len(deck) / 30]}


def collate(items, device):
    n = len(items)
    L = {k: max(1, max(len(it[k]) for it in items)) for k in ("d", "r", "p", "o")}
    T = {}
    for k in ("d", "r", "p", "o"):
        a = np.zeros((n, L[k]), np.int64)
        for i, it in enumerate(items):
            a[i, :len(it[k])] = it[k]
        T[k] = torch.from_numpy(a).to(device)
    for k, ref in (("du", "d"), ("ou", "o"), ("os", "o")):
        a = np.zeros((n, L[ref]), np.float32)
        for i, it in enumerate(items):
            a[i, :len(it[k])] = it[k]
        T[k] = torch.from_numpy(a).to(device)
    T["omask"] = torch.from_numpy(np.array([[j < len(it["o"]) for j in range(L["o"])] for it in items])).to(device)
    T["b"] = torch.tensor([it["b"] for it in items], device=device)
    T["x"] = torch.tensor([it["x"] for it in items], dtype=torch.float32, device=device)
    return T


class Net(nn.Module):
    def __init__(self, enc, d=128, layers=3, heads=4):
        super().__init__()
        self.card = nn.Embedding(len(enc.cards), d, padding_idx=0)
        self.relic = nn.Embedding(len(enc.relics), d, padding_idx=0)
        self.pot = nn.Embedding(len(enc.pots), d, padding_idx=0)
        self.boss = nn.Embedding(len(enc.bosses), d)
        self.kind = nn.Embedding(5, d)   # state, deck card, relic, potion, option
        self.up = nn.Parameter(torch.zeros(d))
        self.skip = nn.Parameter(torch.zeros(d))
        self.scal = nn.Linear(9, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, T):
        k = self.kind.weight
        st = (self.scal(T["x"]) + self.boss(T["b"]) + k[0]).unsqueeze(1)
        deck = self.card(T["d"]) + T["du"].unsqueeze(-1) * self.up + k[1]
        rel = self.relic(T["r"]) + k[2]
        pot = self.pot(T["p"]) + k[3]
        opt = self.card(T["o"]) + T["ou"].unsqueeze(-1) * self.up + T["os"].unsqueeze(-1) * self.skip + k[4]
        x = torch.cat([st, deck, rel, pot, opt], 1)
        n = T["d"].shape[0]
        pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=x.device), T["d"] == 0, T["r"] == 0, T["p"] == 0, ~T["omask"]], 1)
        h = self.tf(x, src_key_padding_mask=pad)
        no = T["o"].shape[1]
        z = self.head(h[:, -no:]).squeeze(-1)
        return z.masked_fill(~T["omask"], -1e9)


def load_decisions(files):
    """-> list of (state, option tokens, chosen index, return, bucket key, run key)."""
    from nets import final_progress
    runs = defaultdict(list)
    outcome = {}
    for f in files:
        for l in open(f):
            r = json.loads(l)
            key = f"{os.path.basename(f)}/{r.get('seed')}"
            if r.get("k") == "o":
                outcome[key] = r
            elif r.get("k") == "d" and r.get("dec") == "card_reward" and r.get("s") and r.get("a", -1) >= 0 and len(r.get("A") or []) > 1:
                cards = r.get("cards") or []
                opts = []
                for a in r["A"]:
                    i = (a[1] or {}).get("card_index")
                    opts.append(cards[i] if a[0] == "select_card_reward" and i is not None and i < len(cards) else None)
                runs[key].append((r["s"], opts, r["a"]))
    out = []
    for key, decs in runs.items():
        if key not in outcome:
            continue
        ret = final_progress(outcome[key])
        for s, opts, a in decs:
            out.append((s, opts, a, ret, (s.get("asc"), s.get("act"), s.get("floor")), key))
    return out


def train(args):
    files = []
    for p in args.files:
        files += sorted(glob.glob(os.path.join(HERE, p))) if any(ch in p for ch in "*?") else [os.path.join(HERE, p)]
    data = load_decisions(files)
    by = defaultdict(list)
    for d in data:
        by[d[4]].append(d[3])
    base_ = {k: float(np.mean(v)) for k, v in by.items()}
    adv = np.array([d[3] - base_[d[4]] for d in data], dtype=np.float32)
    beta = float(adv.std()) * args.beta + 1e-6
    w = np.clip(np.exp(adv / beta), 0, args.wmax)
    w = w / w.mean()
    keys = sorted({d[5] for d in data})
    random.Random(0).shuffle(keys)
    va_keys = set(keys[:max(1, len(keys) // 10)])
    enc = Enc([(d[0], d[1]) for d in data])
    items = [enc.encode(d[0], d[1]) for d in data]
    tr = [i for i, d in enumerate(data) if d[5] not in va_keys]
    va = [i for i, d in enumerate(data) if d[5] in va_keys]
    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(json.dumps({"decisions": len(data), "runs": len(keys), "train": len(tr), "val": len(va), "device": str(dev),
                      "beta": round(beta, 2), "w_max_share": round(float((w >= args.wmax / w.mean() * 0.999).mean()), 4)}), flush=True)
    torch.manual_seed(args.seed)
    net = Net(enc).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    A = np.array([d[2] for d in data])
    W = torch.tensor(w, device=dev)

    def evaluate(idx):
        net.eval()
        agree, nll, wa, wsum = 0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                b = idx[i:i + 512]
                z = net(collate([items[j] for j in b], dev))
                lp = torch.log_softmax(z, -1)
                a = torch.tensor(A[b], device=dev)
                pick = z.argmax(-1)
                agree += int((pick == a).sum())
                nll += float(-lp.gather(1, a[:, None]).sum())
                ww = W[b]
                wa += float((ww * (pick == a).float()).sum())
                wsum += float(ww.sum())
        return {"agree": round(agree / len(idx), 4), "nll": round(nll / len(idx), 4), "w_agree": round(wa / wsum, 4)}

    best = (float("inf"), None, None)
    for ep in range(args.epochs):
        net.train()
        random.Random(ep).shuffle(tr)
        for i in range(0, len(tr), args.batch):
            b = tr[i:i + args.batch]
            z = net(collate([items[j] for j in b], dev))
            lp = torch.log_softmax(z, -1)
            a = torch.tensor(A[b], device=dev)
            loss = -(W[b] * lp.gather(1, a[:, None]).squeeze(1)).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        res = {"epoch": ep + 1, **evaluate(va)}
        print(json.dumps(res), flush=True)
        if res["nll"] < best[0]:
            best = (res["nll"], {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, res)
    torch.save({"enc": enc.to_json(), "state": best[1], "dims": {}}, os.path.join(HERE, args.out))
    print(json.dumps({"saved": args.out, "decisions": len(data), **best[2]}))


class TCardPolicy:
    """Drop-in for nets.CardPolicy: scores(st) -> (legal actions, logits)."""

    def __init__(self, path):
        blob = torch.load(os.path.join(HERE, path) if not os.path.isabs(path) else path, map_location="cpu", weights_only=False)
        self.enc = Enc(blob=blob["enc"])
        self.net = Net(self.enc)
        self.net.load_state_dict(blob["state"])
        self.net.eval()

    def scores(self, st):
        from engine import legal_actions
        from record import card_tok, run_state
        acts = legal_actions(st)
        by_index = {c["index"]: c for c in st.get("cards") or []}
        opts = [card_tok(by_index[a[1]["card_index"]]) if a[0] == "select_card_reward" else None for a in acts]
        with torch.no_grad():
            z = self.net(collate([self.enc.encode(run_state(st), opts)], "cpu"))[0].numpy()
        return acts, z

    def __call__(self, st):
        acts, z = self.scores(st)
        return acts[int(np.argmax(z))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=1.0, help="AWR temperature in advantage std units")
    ap.add_argument("--wmax", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("STS2_RETURN", "sm")
    import nets
    nets.RETURN = os.environ["STS2_RETURN"]
    torch.set_num_threads(int(os.environ.get("NETS_THREADS", "2")))
    train(args)


if __name__ == "__main__":
    sys.exit(main())
