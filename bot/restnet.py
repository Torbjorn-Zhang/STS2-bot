"""Rest-site policy network: heal, which card to smith, or another option, from the deck,
HP and the route ahead. Learns the path test's state-dependent choices by imitation, then
improves by self-play RL (the same AlphaGo stages as the card network).

    restnet.py sl data/v28path_a10.jsonl --out models/rest_sl0.pt
    restnet.py rl data/rl/card_it3.jsonl --init models/rest_sl0.pt --out models/rest_rl1.pt
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from features import RunEncoder, base
from nets import PolicyNet, _batches, final_progress, grouped, load_runs, split_runs

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ["HEAL", "SMITH", "DIG", "LIFT", "COOK", "HATCH", "KINDLE", "CLONE", "MEND", "OTHER"]
ROUTE = ["Monster", "Elite", "RestSite", "Unknown", "Shop", "Treasure"]
SMITH_CANDIDATES = 3
SOFT_TAU = 4.0


def route_feats(route):
    x = np.zeros(len(ROUTE) + 2, dtype=np.float32)
    for k in route or []:
        if k in ROUTE:
            x[ROUTE.index(k)] += 0.2
    x[-2] = len(route or []) / 15
    x[-1] = float(not route)          # the boss is next
    return x


def option_vec(enc, key):
    """key: 'HEAL:None', 'SMITH:<token>', 'DIG:None', ..."""
    kind, tok = key.split(":", 1)
    x = np.zeros(len(enc.cards) + 1 + len(KINDS), dtype=np.float32)
    if tok not in ("None", ""):
        x[enc.cards[base(tok)]] = 1.0
        x[len(enc.cards)] = float(tok.endswith("+"))
    x[len(enc.cards) + 1 + (KINDS.index(kind) if kind in KINDS else KINDS.index("OTHER"))] = 1.0
    return x


def state_vec(enc, rs, route):
    return np.concatenate([enc.encode(rs), route_feats(route)])


def option_keys(st):
    """The rest options as keys (smith expanded to the top upgrade targets), with the
    moves that make each: {key: (option_index, smith token or None)}."""
    import policy
    from record import card_tok
    opts = [o for o in st.get("options") or [] if o.get("is_enabled", True)]
    deck = (st.get("player") or {}).get("deck") or []
    cand = sorted((c for c in deck if not c.get("upgraded") and c.get("type") not in ("Curse", "Status")),
                  key=policy.upgrade_priority, reverse=True)
    tokens = []
    for c in cand:
        t = card_tok(c)
        if t not in tokens:
            tokens.append(t)
        if len(tokens) >= SMITH_CANDIDATES:
            break
    out = {}
    for o in opts:
        oid = o.get("option_id") or "OTHER"
        if oid == "SMITH":
            for t in tokens:
                out[f"SMITH:{t}"] = (o["index"], t)
        elif oid != "CLONE":
            out[f"{oid}:None"] = (o["index"], None)
    return out


class RestPolicy:
    def __init__(self, path):
        blob = torch.load(path, weights_only=False)
        self.enc = RunEncoder.from_json(blob["enc"])
        self.model = PolicyNet(self.enc.dim + len(ROUTE) + 2, len(self.enc.cards) + 1 + len(KINDS), blob["hidden"])
        self.model.load_state_dict(blob["state"])
        self.model.eval()

    def scores(self, sim, st):
        import decktest
        from record import run_state
        keys = option_keys(st)
        if not keys:
            return {}, [], None, []
        route = [k for k, _ in decktest.route_to_boss(sim, st)]
        ks = list(keys)
        with torch.no_grad():
            z = self.model(torch.from_numpy(state_vec(self.enc, run_state(st), route)[None]),
                           torch.from_numpy(np.stack([option_vec(self.enc, k) for k in ks])),
                           torch.zeros(len(ks), dtype=torch.long)).numpy()
        return keys, ks, z, route


def sl_items(runs):
    """Rest decisions judged by the path test: (state, route, keys, path scores)."""
    out = []
    for rows in runs.values():
        for r in rows:
            x = r.get("x") or {}
            if r["k"] == "d" and r["dec"] == "rest_site" and x.get("path") and len(x["path"]) > 1:
                out.append((r["s"], x.get("route") or [], list(x["path"]), [x["path"][k] for k in x["path"]]))
    return out


def train_sl(paths, out, epochs=60, hidden=128, lr=1e-3):
    runs = load_runs(paths)
    states = [r["s"] if r["k"] == "d" else r["run"] for rows in runs.values() for r in rows if r["k"] in ("d", "f")]
    enc = RunEncoder.build(states)
    tr, va = split_runs(runs)
    data = {}
    for name, part in (("train", tr), ("val", va)):
        items = sl_items(part)
        S = np.stack([state_vec(enc, s, rt) for s, rt, _, _ in items]) if items else np.zeros((0, enc.dim + 8))
        A, start, T, best = [], [], [], []
        for s, rt, keys, sc in items:
            start.append(len(A))
            A.extend(option_vec(enc, k) for k in keys)
            z = np.asarray(sc) / SOFT_TAU
            p = np.exp(z - z.max())
            T.extend(p / p.sum())
            best.append(int(np.argmax(sc)))
        start.append(len(A))
        data[name] = (S, np.stack(A) if A else None, np.asarray(start), np.asarray(T, dtype=np.float32), np.asarray(best))
        print(name, "rest decisions", len(items), flush=True)
    torch.manual_seed(0)
    model = PolicyNet(enc.dim + len(ROUTE) + 2, len(enc.cards) + 1 + len(KINDS), hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def run(part, train):
        S, A, start, T, best = data[part]
        n = len(best)
        if n == 0:
            return 0.0, 0.0
        order = np.random.permutation(n) if train else np.arange(n)
        tot, acc = 0.0, 0.0
        for sel in _batches(order, 64):
            a_idx, owner = [], []
            for k, d in enumerate(sel):
                a_idx.extend(range(start[d], start[d + 1]))
                owner.extend([k] * (start[d + 1] - start[d]))
            logits = model(torch.from_numpy(S[sel]), torch.from_numpy(A[a_idx]), torch.as_tensor(owner))
            sizes = np.asarray([start[d + 1] - start[d] for d in sel])
            logp = F.log_softmax(grouped(logits, np.asarray(owner), sizes), dim=1)
            t = torch.zeros_like(logp)
            col = np.concatenate([np.arange(s) for s in sizes])
            t[torch.as_tensor(owner), torch.as_tensor(col)] = torch.from_numpy(T[a_idx])
            loss = -(t * logp.clamp(min=-30)).sum(1).mean()
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * len(sel)
            acc += (logp.argmax(1).numpy() == best[sel]).sum()
        return tot / n, acc / n

    best_state = (-1, None)
    for ep in range(epochs):
        model.train()
        tl, ta = run("train", True)
        model.eval()
        with torch.no_grad():
            vl, vacc = run("val", False)
        if ep % 10 == 9:
            print(json.dumps({"epoch": ep, "train_acc": round(ta, 3), "val_loss": round(vl, 4), "val_acc": round(vacc, 3)}),
                  flush=True)
        if vacc >= best_state[0]:
            best_state = (vacc, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state[1])
    torch.save({"kind": "rest", "enc": enc.to_json(), "hidden": hidden, "state": model.state_dict(),
                "val_acc": best_state[0], "sources": paths}, out)
    print("saved", out, "val_acc", round(best_state[0], 3))


def train_rl(paths, init, out, temperature=1.0, epochs=3, lr=1e-4, clip=0.2, ent=0.01):
    """PPO on sampled rest choices in self-play; return = final progress, baseline = mean
    final progress at the same (act, floor) in this data."""
    blob = torch.load(init, weights_only=False)
    enc = RunEncoder.from_json(blob["enc"])
    model = PolicyNet(enc.dim + len(ROUTE) + 2, len(enc.cards) + 1 + len(KINDS), blob["hidden"])
    model.load_state_dict(blob["state"])
    runs = load_runs(paths)
    items, by_bucket = [], defaultdict(list)
    for rows in runs.values():
        fin = final_progress(rows[-1])
        for r in rows:
            x = r.get("x") or {}
            if r["k"] == "d" and r["dec"] == "rest_site" and x.get("rlp") is not None and len(x.get("rkeys") or []) > 1:
                key = (r["s"]["act"], r["s"]["floor"])
                items.append((r, fin, key))
                by_bucket[key].append(fin)
    if not items:
        print("no rest decisions")
        torch.save(blob, out)
        return
    basel = {k: float(np.mean(v)) for k, v in by_bucket.items()}
    S = np.stack([state_vec(enc, r["s"], r["x"].get("route")) for r, _, _ in items])
    A, start, chosen = [], [], []
    for r, _, _ in items:
        start.append(len(A))
        A.extend(option_vec(enc, k) for k in r["x"]["rkeys"])
        chosen.append(r["x"]["rkeys"].index(r["x"]["rk"]))
    start.append(len(A))
    A, start, chosen = np.stack(A), np.asarray(start), np.asarray(chosen)
    lp_old = np.asarray([r["x"]["rlp"] for r, _, _ in items], dtype=np.float32)
    adv = np.asarray([fin - (r["V"] if r.get("V") is not None else basel[k]) for r, fin, k in items], dtype=np.float32)
    print("rest rl decisions", len(items), "adv std", round(float(adv.std()), 3), flush=True)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        stats, nb = defaultdict(float), 0
        for sel in _batches(np.random.RandomState(ep).permutation(len(chosen)), 128):
            a_idx, owner = [], []
            for k, d in enumerate(sel):
                a_idx.extend(range(start[d], start[d + 1]))
                owner.extend([k] * (start[d + 1] - start[d]))
            logits = model(torch.from_numpy(S[sel]), torch.from_numpy(A[a_idx]), torch.as_tensor(owner))
            sizes = np.asarray([start[d + 1] - start[d] for d in sel])
            logp = F.log_softmax(grouped(logits / temperature, np.asarray(owner), sizes), dim=1)
            lp_new = logp.gather(1, torch.as_tensor(chosen[sel])[:, None]).squeeze(1)
            ratio = torch.exp(lp_new - torch.from_numpy(lp_old[sel]))
            ad = torch.from_numpy(adv[sel])
            pg = -torch.min(ratio * ad, ratio.clamp(1 - clip, 1 + clip) * ad).mean()
            entropy = -(logp.exp() * logp.clamp(min=-30)).sum(1).mean()
            loss = pg - ent * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats["pg"] += pg.item()
            stats["ent"] += entropy.item()
            nb += 1
        print(json.dumps({"epoch": ep, **{k: round(v / nb, 4) for k, v in stats.items()}}), flush=True)
    torch.save({**blob, "state": model.state_dict(), "rl_from": init, "rl_data": paths}, out)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sl", "rl"])
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.mode == "sl":
        train_sl(args.data, args.out)
    else:
        train_rl(args.data, args.init, args.out, temperature=args.temperature)


if __name__ == "__main__":
    main()
