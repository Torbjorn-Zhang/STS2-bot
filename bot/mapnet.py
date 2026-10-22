"""Map policy network: which next node to take, from the deck, HP, gold and each option's
node type plus the route the map DP would follow from it to the boss.

Stages as for the other deck networks: imitation of the DP's choices, then self-play RL
on final progress (cardloop.py --map-init).

    mapnet.py sl data/rl/map_rules.jsonl --out models/map_sl0.pt --enc-from models/deckb_card9.pt
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from features import RunEncoder
from nets import PolicyNet, _batches, final_progress, grouped, load_runs, split_runs
from restnet import ROUTE, route_feats

TYPES = ["Monster", "Elite", "RestSite", "Unknown", "Shop", "Treasure", "Boss"]
OPT_DIM = len(TYPES) + len(ROUTE) + 2 + 3


def option_desc(sim, st, m, ch):
    """[node type, row, route types from it] for one map choice (stored in records)."""
    import decktest
    nodes = {(n["col"], n["row"]): n for row in m.get("rows", []) for n in row}
    n = nodes.get((ch["col"], ch["row"])) or {}
    t = n.get("type") or "Unknown"
    route = [] if t == "Boss" else [k for k, _ in decktest.route_to_boss(sim, st, start=(ch["col"], ch["row"]), m=m)]
    return [t, ch["row"], route]


def option_vec(desc, hp_frac, gold):
    t, row, route = desc
    x = np.zeros(OPT_DIM, dtype=np.float32)
    x[TYPES.index(t) if t in TYPES else TYPES.index("Unknown")] = 1.0
    x[len(TYPES):len(TYPES) + len(ROUTE) + 2] = route_feats(route[1:] if route else [])
    x[-3] = (row or 0) / 15
    x[-2] = hp_frac
    x[-1] = (gold or 0) / 300
    return x


ELITE_BIAS = 0.0   # logit added to elite rooms (run.py --map-elite-bias)
NODE_BIAS = {}     # room type -> logit added (run.py --map-node-bias '{"RestSite": 1, "Shop": 1}')
ELITE_BIAS_ACT = {}   # act -> logit added to elite rooms in that act (run.py --map-elite-bias-act '{"3": -3}')


class MapPolicy:
    def __init__(self, path):
        blob = torch.load(path, weights_only=False)
        self.enc = RunEncoder.from_json(blob["enc"])
        self.model = PolicyNet(self.enc.dim, OPT_DIM, blob["hidden"])
        self.model.load_state_dict(blob["state"])
        self.model.eval()

    def scores(self, sim, st):
        from record import run_state
        chs = st.get("choices") or []
        if len(chs) < 2:
            return chs, [], None
        m = sim.send({"cmd": "get_map"})
        descs = [option_desc(sim, st, m, ch) for ch in chs]
        rs = run_state(st)
        hpf = (rs["hp"] or 0) / max(1, rs["mhp"] or 1)
        with torch.no_grad():
            z = self.model(torch.from_numpy(self.enc.encode(rs)[None]),
                           torch.from_numpy(np.stack([option_vec(d, hpf, rs["gold"]) for d in descs])),
                           torch.zeros(len(chs), dtype=torch.long)).numpy()
        if ELITE_BIAS:   # experiments (run.py --map-elite-bias): humans path to elites 13%, the nets 6.5%
            z = z + np.asarray([ELITE_BIAS if d[0] == "Elite" else 0.0 for d in descs], dtype=z.dtype)
        eb = ELITE_BIAS_ACT.get((st.get("context") or {}).get("act") or 1)
        if eb:   # act-3 elites killed 52 of the 242 runs that reached act 3 (v46db, A10 double boss)
            z = z + np.asarray([eb if d[0] == "Elite" else 0.0 for d in descs], dtype=z.dtype)
        if NODE_BIAS:    # rest sites (now smiths when healthy) and shops (potions, removals) feed deck power
            z = z + np.asarray([NODE_BIAS.get(d[0], 0.0) for d in descs], dtype=z.dtype)
        return chs, descs, z


def items_of(runs, need_lp):
    out = []
    for rows in runs.values():
        fin = final_progress(rows[-1])
        for r in rows:
            x = r.get("x") or {}
            if r["k"] == "d" and r["dec"] == "map_select" and x.get("mopts") and len(x["mopts"]) > 1 \
                    and x.get("mk") is not None and (x.get("mlp") is not None or not need_lp):
                out.append((r, fin))
    return out


def tensors(enc, items):
    S, A, start, chosen = [], [], [], []
    for r, _ in items:
        x = r["x"]
        hpf = (r["s"]["hp"] or 0) / max(1, r["s"]["mhp"] or 1)
        S.append(enc.encode(r["s"]))
        start.append(len(A))
        A.extend(option_vec(d, hpf, r["s"]["gold"]) for d in x["mopts"])
        chosen.append(x["mk"])
    start.append(len(A))
    return np.stack(S), np.stack(A), np.asarray(start), np.asarray(chosen)


def logp_of(model, S, A, start, sel, temperature=1.0):
    a_idx, owner = [], []
    for k, d in enumerate(sel):
        a_idx.extend(range(start[d], start[d + 1]))
        owner.extend([k] * (start[d + 1] - start[d]))
    logits = model(torch.from_numpy(S[sel]), torch.from_numpy(A[a_idx]), torch.as_tensor(owner))
    sizes = np.asarray([start[d + 1] - start[d] for d in sel])
    return F.log_softmax(grouped(logits / temperature, np.asarray(owner), sizes), dim=1)


def train_sl(paths, out, enc_from, epochs=30, hidden=128, lr=1e-3):
    enc = RunEncoder.from_json(torch.load(enc_from, weights_only=False)["enc"])
    tr, va = split_runs(load_runs(paths))
    T, V = items_of(tr, False), items_of(va, False)
    print("map decisions", len(T), len(V), flush=True)
    St, At, stt, cht = tensors(enc, T)
    Sv, Av, stv, chv = tensors(enc, V)
    torch.manual_seed(0)
    model = PolicyNet(enc.dim, OPT_DIM, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best = (-1, None)
    for ep in range(epochs):
        model.train()
        for sel in _batches(np.random.RandomState(ep).permutation(len(cht)), 128):
            loss = F.nll_loss(logp_of(model, St, At, stt, sel), torch.as_tensor(cht[sel]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((logp_of(model, Sv, Av, stv, np.arange(len(chv))).argmax(1).numpy() == chv).mean())
        if ep % 10 == 9:
            print(json.dumps({"epoch": ep, "val_acc": round(acc, 3)}), flush=True)
        if acc >= best[0]:
            best = (acc, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    torch.save({"kind": "map", "enc": enc.to_json(), "hidden": hidden, "state": model.state_dict(),
                "val_acc": best[0], "sources": paths}, out)
    print("saved", out, "val_acc", round(best[0], 3))


def train_rl(paths, init, out, temperature=1.0, epochs=3, lr=1e-4, clip=0.2, ent=0.01):
    lr *= float(os.environ.get("NETS_LR_SCALE", "1"))   # fleet xdeckrl lr_scale
    blob = torch.load(init, weights_only=False)
    enc = RunEncoder.from_json(blob["enc"])
    model = PolicyNet(enc.dim, OPT_DIM, blob["hidden"])
    model.load_state_dict(blob["state"])
    items = items_of(load_runs(paths), True)
    if not items:
        print("no map decisions")
        torch.save(blob, out)
        return
    by = defaultdict(list)
    for r, fin in items:
        by[(r["s"].get("asc"), r["s"]["act"], r["s"]["floor"])].append(fin)
    basel = {k: float(np.mean(v)) for k, v in by.items()}
    S, A, start, chosen = tensors(enc, items)
    lp_old = np.asarray([r["x"]["mlp"] for r, _ in items], dtype=np.float32)
    adv = np.asarray([fin - (r["V"] if r.get("V") is not None else basel[(r["s"].get("asc"), r["s"]["act"], r["s"]["floor"])])
                      for r, fin in items], dtype=np.float32)
    print("map rl decisions", len(items), "adv std", round(float(adv.std()), 3), flush=True)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        stats, nb = defaultdict(float), 0
        for sel in _batches(np.random.RandomState(ep).permutation(len(chosen)), 256):
            logp = logp_of(model, S, A, start, sel, temperature)
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
    ap.add_argument("--enc-from", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()
    torch.set_num_threads(2)
    if args.mode == "sl":
        train_sl(args.data, args.out, args.enc_from)
    else:
        train_rl(args.data, args.init, args.out, temperature=args.temperature)


if __name__ == "__main__":
    main()
