"""Event policy network: which event option to take, from the deck, HP, gold, the route
ahead and the option itself (its text key, hashed, and its numbers: HP, max HP, gold...).

Stages as for the card and rest networks: start from the rule policy's choices
(imitation), improve by self-play RL on final progress, and learn from the path test's
choices in expert games.

    eventnet.py sl data/rl/events_rules.jsonl --out models/event_sl0.pt
    eventnet.py rl data/rl/deck_it3.jsonl --init models/event_sl0.pt --out models/event_rl1.pt
"""

import argparse
import json
import math
import os
import re
import zlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from features import RunEncoder
from nets import PolicyNet, _batches, final_progress, grouped, load_runs, split_runs
from restnet import ROUTE, route_feats

HASH = 512
NUMS = ["hp", "maxhp", "gold", "heal", "damage", "card", "relic", "potion", "curse", "upgrade", "remove", "transform"]


def option_desc(o):
    """[text key, {var: number}, title + description] of an event option (stored in records)."""
    return [o.get("text_key") or o.get("title") or "", o.get("vars") or {},
            f"{o.get('title') or ''} {o.get('description') or ''}"[:200]]


def option_vec(desc, hp_frac):
    key, vars_, text = desc
    x = np.zeros(HASH * 2 + len(NUMS) * 2 + 3, dtype=np.float32)
    x[zlib.crc32(key.encode()) % HASH] = 1.0
    ev = key.split(".")[0]
    x[HASH + zlib.crc32(ev.encode()) % HASH] = 1.0
    o = 2 * HASH
    for k, v in (vars_ or {}).items():
        if not isinstance(v, (int, float)):
            continue
        kl = k.lower()
        for j, name in enumerate(NUMS):
            if name in kl:
                x[o + j] += min(abs(v), 300) / (100 if name == "gold" else 20)
    t = (text or "").lower()
    o2 = o + len(NUMS)
    for j, name in enumerate(NUMS):
        if re.search(name.replace("maxhp", "max hp"), t):
            x[o2 + j] = 1.0
    x[-3] = float("lose" in t)
    x[-2] = float("gain" in t or "obtain" in t)
    x[-1] = hp_frac
    return x


def state_vec(enc, rs, route):
    return np.concatenate([enc.encode(rs), route_feats(route)])


class EventPolicy:
    def __init__(self, path):
        blob = torch.load(path, weights_only=False)
        self.enc = RunEncoder.from_json(blob["enc"])
        self.model = PolicyNet(self.enc.dim + len(ROUTE) + 2, HASH * 2 + len(NUMS) * 2 + 3, blob["hidden"])
        self.model.load_state_dict(blob["state"])
        self.model.eval()

    def scores(self, sim, st):
        import decktest
        from record import run_state
        opts = [o for o in st.get("options") or [] if not o.get("is_locked")]
        if len(opts) < 2:
            return opts, [], None, []
        try:
            route = [k for k, _ in decktest.route_to_boss(sim, st)]
        except Exception:
            route = []
        rs = run_state(st)
        hpf = (rs["hp"] or 0) / max(1, rs["mhp"] or 1)
        descs = [option_desc(o) for o in opts]
        with torch.no_grad():
            z = self.model(torch.from_numpy(state_vec(self.enc, rs, route)[None]),
                           torch.from_numpy(np.stack([option_vec(d, hpf) for d in descs])),
                           torch.zeros(len(opts), dtype=torch.long)).numpy()
        return opts, descs, z, route


def items_of(runs, need_lp):
    out = []
    for key, rows in runs.items():
        fin = final_progress(rows[-1])
        for r in rows:
            x = r.get("x") or {}
            if r["k"] == "d" and r["dec"] == "event_choice" and x.get("eopts") and len(x["eopts"]) > 1 \
                    and x.get("ek") is not None and (x.get("elp") is not None or not need_lp):
                out.append((r, fin))
    return out


def tensors(enc, items):
    S, A, start, chosen = [], [], [], []
    for r, _ in items:
        x = r["x"]
        hpf = (r["s"]["hp"] or 0) / max(1, r["s"]["mhp"] or 1)
        S.append(state_vec(enc, r["s"], x.get("route")))
        start.append(len(A))
        A.extend(option_vec(d, hpf) for d in x["eopts"])
        chosen.append(x["ek"])
    start.append(len(A))
    return np.stack(S), np.stack(A), np.asarray(start), np.asarray(chosen)


def run_batches(model, S, A, start, chosen, sel, temperature=1.0):
    a_idx, owner = [], []
    for k, d in enumerate(sel):
        a_idx.extend(range(start[d], start[d + 1]))
        owner.extend([k] * (start[d + 1] - start[d]))
    logits = model(torch.from_numpy(S[sel]), torch.from_numpy(A[a_idx]), torch.as_tensor(owner))
    sizes = np.asarray([start[d + 1] - start[d] for d in sel])
    return F.log_softmax(grouped(logits / temperature, np.asarray(owner), sizes), dim=1)


def train_sl(paths, out, epochs=30, hidden=128, lr=1e-3, init_enc=None):
    runs = load_runs(paths)
    if init_enc:
        enc = RunEncoder.from_json(torch.load(init_enc, weights_only=False)["enc"])
    else:
        states = [r["s"] if r["k"] == "d" else r["run"] for rows in runs.values() for r in rows if r["k"] in ("d", "f")]
        enc = RunEncoder.build(states)
    tr, va = split_runs(runs)
    T, V = items_of(tr, False), items_of(va, False)
    print("event decisions", len(T), len(V), flush=True)
    St, At, stt, cht = tensors(enc, T)
    Sv, Av, stv, chv = tensors(enc, V) if V else (None, None, None, None)
    torch.manual_seed(0)
    model = PolicyNet(enc.dim + len(ROUTE) + 2, HASH * 2 + len(NUMS) * 2 + 3, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best = (-1, None)
    for ep in range(epochs):
        model.train()
        for sel in _batches(np.random.RandomState(ep).permutation(len(cht)), 128):
            logp = run_batches(model, St, At, stt, cht, sel)
            loss = F.nll_loss(logp, torch.as_tensor(cht[sel]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        acc = 0.0
        if V:
            with torch.no_grad():
                logp = run_batches(model, Sv, Av, stv, chv, np.arange(len(chv)))
                acc = float((logp.argmax(1).numpy() == chv).mean())
        if ep % 10 == 9:
            print(json.dumps({"epoch": ep, "val_acc": round(acc, 3)}), flush=True)
        if acc >= best[0]:
            best = (acc, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    torch.save({"kind": "event", "enc": enc.to_json(), "hidden": hidden, "state": model.state_dict(),
                "val_acc": best[0], "sources": paths}, out)
    print("saved", out, "val_acc", round(best[0], 3))


def train_rl(paths, init, out, temperature=1.0, epochs=3, lr=1e-4, clip=0.2, ent=0.01):
    blob = torch.load(init, weights_only=False)
    enc = RunEncoder.from_json(blob["enc"])
    model = PolicyNet(enc.dim + len(ROUTE) + 2, HASH * 2 + len(NUMS) * 2 + 3, blob["hidden"])
    model.load_state_dict(blob["state"])
    items = items_of(load_runs(paths), True)
    if not items:
        print("no event decisions")
        torch.save(blob, out)
        return
    by = defaultdict(list)
    for r, fin in items:
        by[(r["s"]["act"], r["s"]["floor"])].append(fin)
    basel = {k: float(np.mean(v)) for k, v in by.items()}
    S, A, start, chosen = tensors(enc, items)
    lp_old = np.asarray([r["x"]["elp"] for r, _ in items], dtype=np.float32)
    adv = np.asarray([fin - basel[(r["s"]["act"], r["s"]["floor"])] for r, fin in items], dtype=np.float32)
    print("event rl decisions", len(items), "adv std", round(float(adv.std()), 3), flush=True)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        stats, nb = defaultdict(float), 0
        for sel in _batches(np.random.RandomState(ep).permutation(len(chosen)), 128):
            logp = run_batches(model, S, A, start, chosen, sel, temperature)
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
    ap.add_argument("--enc-from", default=None, help="sl: reuse this model's run encoder vocabulary")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.mode == "sl":
        train_sl(args.data, args.out, init_enc=args.enc_from)
    else:
        train_rl(args.data, args.init, args.out, temperature=args.temperature)


if __name__ == "__main__":
    main()
