"""Shop policy network: remove a card, buy a card / relic / potion, or leave, one purchase
at a time, from the deck, HP, gold, the route ahead and each offer (item, price, rarity).

Stages as for the other deck networks: imitation of the shop rules (policy.choose_shop),
then self-play RL on final progress (cardloop.py --shop-init).
"""

import argparse
import json
import os
import zlib
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from features import RunEncoder
from nets import PolicyNet, _batches, final_progress, grouped, load_runs, split_runs
from restnet import ROUTE, route_feats

KINDS = ["remove", "card", "relic", "potion", "leave"]
RARITIES = ["Common", "Uncommon", "Rare", "Shop", "Special"]
CTYPES = ["Attack", "Skill", "Power"]
HASH = 128


def shop_options(st, visit):
    """[(desc, action)] of the affordable shop moves. desc = [kind, key, cost, rarity, type]."""
    import policy
    gold = (st.get("player") or {}).get("gold") or 0
    out = []
    cost = st.get("card_removal_cost")
    if not visit.get("removed") and isinstance(cost, int) and gold >= cost:
        out.append((["remove", "", cost, "", ""], ("remove_card", {})))
    for c in st.get("cards") or []:
        if c.get("is_stocked", True) and (c.get("cost") or 9999) <= gold:
            out.append((["card", policy.card_key(c), c.get("cost"), c.get("rarity") or "", c.get("type") or ""],
                        ("buy_card", {"card_index": c["index"]})))
    for r in st.get("relics") or []:
        if r.get("is_stocked", True) and (r.get("cost") or 9999) <= gold:
            out.append((["relic", r.get("name") or "", r.get("cost"), "", ""], ("buy_relic", {"relic_index": r["index"]})))
    slots = (st.get("player") or {}).get("potions") or []
    if any(not p for p in slots):
        for p in st.get("potions") or []:
            if p.get("is_stocked", True) and (p.get("cost") or 9999) <= gold:
                out.append((["potion", p.get("name") or "", p.get("cost"), "", ""], ("buy_potion", {"potion_index": p["index"]})))
    out.append((["leave", "", 0, "", ""], ("leave_room", {})))
    return out


def option_vec(enc, desc, gold):
    kind, key, cost, rarity, ctype = desc
    nc = len(enc.cards)
    x = np.zeros(len(KINDS) + nc + HASH + 2 + len(RARITIES) + len(CTYPES), dtype=np.float32)
    x[KINDS.index(kind)] = 1.0
    o = len(KINDS)
    if kind == "card":
        x[o + enc.cards[key]] = 1.0
    o += nc
    if kind in ("relic", "potion"):
        x[o + zlib.crc32(f"{kind}/{key}".encode()) % HASH] = 1.0
    o += HASH
    x[o] = (cost or 0) / 100
    x[o + 1] = (cost or 0) / max(1, gold)
    o += 2
    if rarity in RARITIES:
        x[o + RARITIES.index(rarity)] = 1.0
    o += len(RARITIES)
    if ctype in CTYPES:
        x[o + CTYPES.index(ctype)] = 1.0
    return x


def opt_dim(enc):
    return len(KINDS) + len(enc.cards) + HASH + 2 + len(RARITIES) + len(CTYPES)


def state_vec(enc, rs, route, visit):
    v = np.array([float(bool(visit.get("removed"))), (visit.get("bought") or 0) / 3], dtype=np.float32)
    return np.concatenate([enc.encode(rs), route_feats(route), v])


def state_dim(enc):
    return enc.dim + len(ROUTE) + 2 + 2


class ShopPolicy:
    def __init__(self, path):
        blob = torch.load(path, weights_only=False)
        self.enc = RunEncoder.from_json(blob["enc"])
        self.model = PolicyNet(state_dim(self.enc), opt_dim(self.enc), blob["hidden"])
        self.model.load_state_dict(blob["state"])
        self.model.eval()

    def scores(self, sim, st, visit):
        import decktest
        from record import run_state
        opts = shop_options(st, visit)
        try:
            route = [k for k, _ in decktest.route_to_boss(sim, st)]
        except Exception:
            route = []
        rs = run_state(st)
        with torch.no_grad():
            z = self.model(torch.from_numpy(state_vec(self.enc, rs, route, visit)[None]),
                           torch.from_numpy(np.stack([option_vec(self.enc, d, rs["gold"] or 0) for d, _ in opts])),
                           torch.zeros(len(opts), dtype=torch.long)).numpy()
        return opts, route, z


def items_of(runs, need_lp):
    out = []
    for rows in runs.values():
        fin = final_progress(rows[-1])
        for r in rows:
            x = r.get("x") or {}
            if r["k"] == "d" and r["dec"] == "shop" and x.get("sopts") and len(x["sopts"]) > 1 \
                    and x.get("sk") is not None and (x.get("slp") is not None or not need_lp):
                out.append((r, fin))
    return out


def tensors(enc, items):
    S, A, start, chosen = [], [], [], []
    for r, _ in items:
        x = r["x"]
        visit = {"removed": x["smemo"][0], "bought": x["smemo"][1]}
        S.append(state_vec(enc, r["s"], x.get("route"), visit))
        start.append(len(A))
        A.extend(option_vec(enc, d, r["s"]["gold"] or 0) for d in x["sopts"])
        chosen.append(x["sk"])
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
    print("shop decisions", len(T), len(V), flush=True)
    St, At, stt, cht = tensors(enc, T)
    Sv, Av, stv, chv = tensors(enc, V)
    torch.manual_seed(0)
    model = PolicyNet(state_dim(enc), opt_dim(enc), hidden)
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
    torch.save({"kind": "shop", "enc": enc.to_json(), "hidden": hidden, "state": model.state_dict(),
                "val_acc": best[0], "sources": paths}, out)
    print("saved", out, "val_acc", round(best[0], 3))


def train_rl(paths, init, out, temperature=1.0, epochs=3, lr=1e-4, clip=0.2, ent=0.01):
    lr *= float(os.environ.get("NETS_LR_SCALE", "1"))   # fleet xdeckrl lr_scale
    blob = torch.load(init, weights_only=False)
    enc = RunEncoder.from_json(blob["enc"])
    model = PolicyNet(state_dim(enc), opt_dim(enc), blob["hidden"])
    model.load_state_dict(blob["state"])
    items = items_of(load_runs(paths), True)
    if not items:
        print("no shop decisions")
        torch.save(blob, out)
        return
    by = defaultdict(list)
    for r, fin in items:
        by[(r["s"]["act"], r["s"]["floor"])].append(fin)
    basel = {k: float(np.mean(v)) for k, v in by.items()}
    S, A, start, chosen = tensors(enc, items)
    lp_old = np.asarray([r["x"]["slp"] for r, _ in items], dtype=np.float32)
    adv = np.asarray([fin - (r["V"] if r.get("V") is not None else basel[(r["s"]["act"], r["s"]["floor"])])
                      for r, fin in items], dtype=np.float32)
    print("shop rl decisions", len(items), "adv std", round(float(adv.std()), 3), flush=True)
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
