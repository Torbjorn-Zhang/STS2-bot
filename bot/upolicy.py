"""Unified out-of-combat policy: one transformer for card rewards, rest sites, events, the map and
shops (spire-mentor's run-policy shape), instead of five small separate networks.

Tokens of a decision: a state token (act, floor, ascension, HP, gold, boss, room, the upcoming
route), one per deck card, relic and potion, and one per option. An option token carries its
decision kind and its own fields: a card (card rewards, smiths, shop cards), a relic or potion
(shop), a rest action, an event option (its key and numbers), a map node (type, row, the room
types on its route), a shop line (kind, price share of gold, rarity, card type). Every token
attends to every other; a head scores the option tokens; a softmax over one decision's options.

  upolicy.py train FILES... --out models/up1.pt [--awr] [--device cuda]   imitation (optionally
      advantage-weighted) on self-play records: every decision type at once
Records are run.py --record rows written with DeckNets (the extras carry the option lists).
"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ["card_reward", "rest_site", "event_choice", "map_select", "shop"]
ROOMS = ["Monster", "Elite", "RestSite", "Shop", "Unknown", "Treasure", "Boss", "Event", "Ancient"]
NUMS = ["Cards", "Relics", "Curses", "Gold", "Hp", "HpLoss", "MaxHp", "Potions", "Damage", "Heal", "Energy"]


def base(tok):
    return (tok or "").rstrip("+")


class Vocab:
    def __init__(self, items, min_count=1):
        cnt = Counter(items)
        self.itos = ["<pad>", "<unk>"] + sorted(k for k, v in cnt.items() if v >= min_count and k is not None)
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


def route_counts(route):
    route = route or []
    return [sum(1 for r in route if r == t) / 5.0 for t in ROOMS] + [len(route) / 15.0]


def options_of(r):
    """Option descriptors of a record decision row, in its action order; None if unsupported."""
    dec, x, A = r.get("dec"), r.get("x") or {}, r.get("A") or []
    if dec == "card_reward":
        cards = r.get("cards") or []
        out = []
        for a in A:
            i = (a[1] or {}).get("card_index")
            out.append(("card", cards[i]) if a[0] == "select_card_reward" and i is not None and i < len(cards) else ("skip", None))
        return out
    if dec == "rest_site" and x.get("rkeys"):
        return [("rest", k) for k in x["rkeys"]]
    if dec == "event_choice" and x.get("eopts"):
        return [("event", o) for o in x["eopts"]]
    if dec == "map_select" and x.get("mopts"):
        return [("map", o) for o in x["mopts"]]
    if dec == "shop" and x.get("sopts"):
        return [("shop", o) for o in x["sopts"]]
    return None


def chosen_of(r, opts):
    dec, x = r.get("dec"), r.get("x") or {}
    if dec == "card_reward":
        return r.get("a", -1)
    if dec == "rest_site":
        return x["rkeys"].index(x["rk"]) if x.get("rk") in (x.get("rkeys") or []) else -1
    if dec == "event_choice":
        return x.get("ek", -1)
    if dec == "map_select":
        return x.get("mk", -1)
    if dec == "shop":
        return x.get("sk", -1)
    return -1


def behavior_lp(r):
    x = r.get("x") or {}
    return x.get({"card_reward": "lp", "rest_site": "rlp", "event_choice": "elp", "map_select": "mlp", "shop": "slp"}[r["dec"]])


class Enc:
    MAXD, MAXR, MAXP = 60, 40, 5

    def __init__(self, rows=None, blob=None):
        if blob is not None:
            for k, v in blob.items():
                setattr(self, k, Vocab.of(v))
            return
        cards, relics, pots, bosses, evk, rest = [], [], [], [], [], []
        for s, opts in rows:
            cards += [base(c) for c in s.get("deck") or []]
            relics += s.get("relics") or []
            pots += s.get("pots") or []
            bosses.append(str(s.get("boss")))
            for kind, o in opts:
                if kind == "card" and o:
                    cards.append(base(o))
                elif kind == "rest":
                    a, _, c = o.partition(":")
                    rest.append(a)
                    if c and c != "None":
                        cards.append(base(c))
                elif kind == "event":
                    evk.append(re.sub(r"\.options\.", ".", o[0] or ""))
                elif kind == "shop":
                    if o[0] == "card":
                        cards.append(o[1])
                    elif o[0] == "relic":
                        relics.append(o[1])
                    elif o[0] == "potion":
                        pots.append(o[1])
        self.cards, self.relics, self.pots = Vocab(cards, 2), Vocab(relics, 2), Vocab(pots, 2)
        self.bosses, self.evk, self.rest = Vocab(bosses), Vocab(evk, 3), Vocab(rest)
        self.room = Vocab(ROOMS)
        self.kind = Vocab(["card", "skip", "rest", "event", "map", "shop:remove", "shop:card", "shop:relic", "shop:potion", "shop:leave"])
        self.rar = Vocab(["Common", "Uncommon", "Rare", "Special", "Basic"])
        self.ctype = Vocab(["Attack", "Skill", "Power", "Status", "Curse"])

    def to_json(self):
        return {k: v.itos for k, v in self.__dict__.items() if isinstance(v, Vocab)}

    NOPT = 6 + len(ROOMS) + 1 + len(NUMS)   # option scalars

    def option(self, kind, o, gold):
        """-> (kind id, card id, card upgraded, relic id, potion id, rest id, event key id, room id, scalars)."""
        sc = [0.0] * self.NOPT
        card = up = relic = pot = rid = eid = mid = 0
        if kind == "card":
            card, up, k = self.cards(base(o)), float(o.endswith("+")), "card"
        elif kind == "skip":
            k = "skip"
        elif kind == "rest":
            a, _, c = o.partition(":")
            rid, k = self.rest(a), "rest"
            if c and c != "None":
                card, up = self.cards(base(c)), float(c.endswith("+"))
        elif kind == "event":
            eid, k = self.evk(re.sub(r"\.options\.", ".", o[0] or "")), "event"
            vars_ = o[1] if len(o) > 1 and isinstance(o[1], dict) else {}
            for i, n in enumerate(NUMS):
                v = vars_.get(n)
                if isinstance(v, (int, float)):
                    sc[6 + len(ROOMS) + 1 + i] = float(v) / (100.0 if n in ("Gold",) else 10.0)
        elif kind == "map":
            t, row, route = (o + [None, None, None])[:3]
            mid, k = self.room(t), "map"
            sc[0] = (row or 0) / 15.0
            sc[6:6 + len(ROOMS) + 1] = route_counts(route)
        else:   # shop line: [kind, id/name, cost, rarity, card type]
            sk, name, cost, rar, ct = (list(o) + ["", "", 0, "", ""])[:5]
            k = f"shop:{sk}"
            if sk == "card":
                card = self.cards(name)
            elif sk == "relic":
                relic = self.relics(name)
            elif sk == "potion":
                pot = self.pots(name)
            sc[1] = (cost or 0) / max(1.0, float(gold or 1))
            sc[2] = float((cost or 0) <= (gold or 0))
            sc[3] = self.rar(rar) / 6.0
            sc[4] = self.ctype(ct) / 6.0
        return self.kind(k), card, up, relic, pot, rid, eid, mid, sc

    def encode(self, dec, s, opts, route=None):
        deck = (s.get("deck") or [])[:self.MAXD]
        gold = s.get("gold") or 0
        os_ = [self.option(k, o, gold) for k, o in opts]
        return {"dec": KINDS.index(dec),
                "d": [self.cards(base(c)) for c in deck], "du": [float(c.endswith("+")) for c in deck],
                "r": [self.relics(x) for x in (s.get("relics") or [])[:self.MAXR]],
                "p": [self.pots(x) for x in (s.get("pots") or [])[:self.MAXP]],
                "ok": [o[0] for o in os_], "oc": [o[1] for o in os_], "ou": [o[2] for o in os_], "or": [o[3] for o in os_],
                "op": [o[4] for o in os_], "orest": [o[5] for o in os_], "oev": [o[6] for o in os_], "oroom": [o[7] for o in os_],
                "osc": [o[8] for o in os_],
                "b": self.bosses(str(s.get("boss"))),
                "x": [(s.get("act") or 1) / 3, (s.get("floor") or 0) / 17, (s.get("asc") or 0) / 10,
                      (s.get("hp") or 0) / max(1, s.get("mhp") or 1), (s.get("mhp") or 80) / 100, gold / 300,
                      float(s.get("room") == "Elite"), float(s.get("room") == "Boss"), len(deck) / 30] + route_counts(route)}


SDIM = 9 + len(ROOMS) + 1


def collate(items, device):
    n = len(items)
    L = {k: max(1, max(len(it[k]) for it in items)) for k in ("d", "r", "p", "ok")}
    T = {}

    def pad(key, ref, dtype):
        a = np.zeros((n, L[ref]), dtype)
        for i, it in enumerate(items):
            a[i, :len(it[key])] = it[key]
        return torch.from_numpy(a).to(device)
    T["d"], T["du"] = pad("d", "d", np.int64), pad("du", "d", np.float32)
    T["r"], T["p"] = pad("r", "r", np.int64), pad("p", "p", np.int64)
    for k in ("ok", "oc", "or", "op", "orest", "oev", "oroom"):
        T[k] = pad(k, "ok", np.int64)
    T["ou"] = pad("ou", "ok", np.float32)
    sc = np.zeros((n, L["ok"], Enc.NOPT), np.float32)
    for i, it in enumerate(items):
        if it["osc"]:
            sc[i, :len(it["osc"])] = np.asarray(it["osc"], np.float32)
    T["osc"] = torch.from_numpy(sc).to(device)
    T["omask"] = torch.from_numpy(np.array([[j < len(it["ok"]) for j in range(L["ok"])] for it in items])).to(device)
    T["b"] = torch.tensor([it["b"] for it in items], device=device)
    T["dec"] = torch.tensor([it["dec"] for it in items], device=device)
    T["x"] = torch.tensor([it["x"] for it in items], dtype=torch.float32, device=device)
    return T


class Net(nn.Module):
    def __init__(self, enc, d=128, layers=3, heads=4):
        super().__init__()
        E = lambda v: nn.Embedding(len(v), d, padding_idx=0)
        self.card, self.relic, self.pot, self.boss = E(enc.cards), E(enc.relics), E(enc.pots), nn.Embedding(len(enc.bosses), d)
        self.okind, self.evk, self.rest, self.room = E(enc.kind), E(enc.evk), E(enc.rest), E(enc.room)
        self.dec = nn.Embedding(len(KINDS), d)
        self.tok = nn.Embedding(5, d)   # state, deck card, relic, potion, option
        self.up = nn.Parameter(torch.zeros(d))
        self.scal = nn.Linear(SDIM, d)
        self.osc = nn.Linear(Enc.NOPT, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, T, with_value=False):
        k = self.tok.weight
        st = (self.scal(T["x"]) + self.boss(T["b"]) + self.dec(T["dec"]) + k[0]).unsqueeze(1)
        deck = self.card(T["d"]) + T["du"].unsqueeze(-1) * self.up + k[1]
        rel = self.relic(T["r"]) + k[2]
        pot = self.pot(T["p"]) + k[3]
        misc = self.rest(T["orest"]) + self.evk(T["oev"]) + self.room(T["oroom"])   # 0 = padding where not used
        opt = (self.okind(T["ok"]) + self.card(T["oc"]) + T["ou"].unsqueeze(-1) * self.up + self.relic(T["or"])
               + self.pot(T["op"]) + misc + self.osc(T["osc"]) + k[4])
        x = torch.cat([st, deck, rel, pot, opt], 1)
        n = x.shape[0]
        pad = torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=x.device), T["d"] == 0, T["r"] == 0, T["p"] == 0, ~T["omask"]], 1)
        h = self.tf(x, src_key_padding_mask=pad)
        no = T["ok"].shape[1]
        z = self.head(h[:, -no:]).squeeze(-1).masked_fill(~T["omask"], -1e9)
        if with_value:
            return z, self.value(h[:, 0]).squeeze(-1)
        return z


def load_records(files):
    """-> decisions [(dec, s, opts, chosen, route, behavior lp, run key)], outcomes {run key: row}."""
    decs, outcome = [], {}
    for f in files:
        for l in open(f):
            r = json.loads(l)
            key = f"{os.path.basename(f)}/{r.get('seed')}"
            if r.get("k") == "o":
                outcome[key] = r
                continue
            if r.get("k") != "d" or r.get("dec") not in KINDS or not r.get("s"):
                continue
            opts = options_of(r)
            if not opts or len(opts) < 2:
                continue
            a = chosen_of(r, opts)
            if a is None or a < 0 or a >= len(opts):
                continue
            decs.append((r["dec"], r["s"], opts, a, (r.get("x") or {}).get("route"), behavior_lp(r), key))
    return decs, outcome


def expand(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(HERE, p))) if any(ch in p for ch in "*?") else [os.path.join(HERE, p)]
    return out


def train(args):
    from nets import final_progress
    decs, outcome = load_records(expand(args.files))
    decs = [d for d in decs if d[6] in outcome]
    ret = {k: final_progress(o) for k, o in outcome.items()}
    by = defaultdict(list)
    for d in decs:
        by[(d[1].get("asc"), d[1].get("act"), d[1].get("floor"))].append(ret[d[6]])
    base_ = {k: float(np.mean(v)) for k, v in by.items()}
    adv = np.array([ret[d[6]] - base_[(d[1].get("asc"), d[1].get("act"), d[1].get("floor"))] for d in decs], np.float32)
    if args.awr:
        w = np.clip(np.exp(adv / (adv.std() * args.beta + 1e-6)), 0, args.wmax)
        w = w / w.mean()
    else:
        w = np.ones(len(decs), np.float32)
    keys = sorted({d[6] for d in decs})
    random.Random(0).shuffle(keys)
    va_keys = set(keys[:max(1, len(keys) // 10)])
    enc = Enc([(d[1], d[2]) for d in decs])
    items = [enc.encode(d[0], d[1], d[2], d[4]) for d in decs]
    tr = [i for i, d in enumerate(decs) if d[6] not in va_keys]
    va = [i for i, d in enumerate(decs) if d[6] in va_keys]
    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(json.dumps({"decisions": len(decs), "by_kind": dict(Counter(d[0] for d in decs)), "runs": len(keys),
                      "device": str(dev)}), flush=True)
    torch.manual_seed(args.seed)
    net = Net(enc).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    A = np.array([d[3] for d in decs])
    W = torch.tensor(w, device=dev)
    R = torch.tensor([ret[d[6]] for d in decs], dtype=torch.float32, device=dev)

    def evaluate(idx):
        net.eval()
        per = defaultdict(lambda: [0, 0])
        nll = 0.0
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                b = idx[i:i + 512]
                z = net(collate([items[j] for j in b], dev))
                lp = torch.log_softmax(z, -1)
                a = torch.tensor(A[b], device=dev)
                nll += float(-lp.gather(1, a[:, None]).sum())
                ok = (z.argmax(-1) == a).cpu().numpy()
                for j, o in zip(b, ok):
                    per[decs[j][0]][0] += int(o)
                    per[decs[j][0]][1] += 1
        return {"nll": round(nll / len(idx), 4), **{k: round(v[0] / v[1], 3) for k, v in per.items()}}

    best = (float("inf"), None, None)
    for ep in range(args.epochs):
        net.train()
        random.Random(ep).shuffle(tr)
        for i in range(0, len(tr), args.batch):
            b = tr[i:i + args.batch]
            z, v = net(collate([items[j] for j in b], dev), with_value=True)
            lp = torch.log_softmax(z, -1)
            a = torch.tensor(A[b], device=dev)
            loss = -(W[b] * lp.gather(1, a[:, None]).squeeze(1)).mean() + 0.5 * ((v - R[b] / 100.0) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        res = {"epoch": ep + 1, **evaluate(va)}
        print(json.dumps(res), flush=True)
        if res["nll"] < best[0]:
            best = (res["nll"], {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, res)
    torch.save({"enc": enc.to_json(), "state": best[1]}, os.path.join(HERE, args.out))
    print(json.dumps({"saved": args.out, "decisions": len(decs), **best[2]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--awr", action="store_true")
    ap.add_argument("--beta", type=float, default=1.0)
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
