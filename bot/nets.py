"""AlphaGo-style networks for STS2 (PyTorch, CPU).

  policy  combat policy p(a|s): imitation of the search planner's moves (the SL policy
          network of AlphaGo), later improved by self-play RL.
  value   run value V(s): from a run state (deck, relics, HP, where we are) predict how
          far the run gets and whether it clears the current act.

    nets.py policy data/sp_a10_r0.jsonl [--out models/policy.pt] [--epochs 8]
    nets.py value  data/sp_a10_r0.jsonl [--out models/value_run.pt]
"""

import argparse
import json
import os
import random
import sys
import time
import zlib
from collections import defaultdict

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

from features import FLOORS_PER_ACT, RunEncoder, Vocab, base, progress

HERE = os.path.dirname(os.path.abspath(__file__))
HITS = ("Attack", "DeathBlow")
MAX_ENEMIES = 5


# ---------------------------------------------------------------- data

def load_runs(paths):
    """seed -> list of records, for runs that finished (have an outcome record)."""
    runs = defaultdict(list)
    for path in paths:
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                runs[(path, r["seed"])].append(r)
    return {k: v for k, v in runs.items() if v and v[-1]["k"] == "o" and not v[-1].get("error")}


def split_of(key, frac=0.1):
    return "val" if (zlib.crc32(str(key).encode()) % 1000) < frac * 1000 else "train"


def split_runs(runs):
    tr = {k: v for k, v in runs.items() if split_of(k) == "train"}
    va = {k: v for k, v in runs.items() if split_of(k) == "val"}
    if not va and len(tr) > 1:   # tiny data sets (debugging)
        k = sorted(tr)[-1]
        va[k] = tr.pop(k)
    return tr, va


ACT_CLEAR_BONUS = float(os.environ.get("STS2_ACT_CLEAR_BONUS", "10"))   # RL reward per act boss beaten
BOSS_DMG_BONUS = float(os.environ.get("STS2_BOSS_DMG_BONUS", "5"))      # ... per share of boss HP removed


def final_progress(o):
    """RL return of a run: floors climbed plus ACT_CLEAR_BONUS per act cleared. Floors
    alone paid almost nothing for beating a boss (dying at the act-2 boss = 33, beating
    it = 34+), so deck learning aimed at surviving hallways, not at winning boss fights."""
    acts = 3 if o.get("victory") else max(0, (o.get("act") or 1) - 1)
    base = 3 * FLOORS_PER_ACT + 2 if o.get("victory") else progress(o.get("act"), o.get("floor"))
    # boss damage: a denser signal than boss wins (a lost boss fight still shows how close
    # the deck came); recorded by selfplay.py as the sum of removed HP shares (a win = 1)
    return base + ACT_CLEAR_BONUS * acts + BOSS_DMG_BONUS * (o.get("boss_dmg") or 0)


# ---------------------------------------------------------------- combat encoding

def incoming(es):
    es = [e for e in es if (e[1] or 0) < 10 ** 8]
    tot = 0
    for e in es:
        for t, d, h in e[5]:
            if t in HITS and d is not None:
                tot += d * (h or 1)
    return tot


def card_stats(h):
    return h[4] if len(h) > 4 and h[4] else {}


def card_dmg(h, target=None):
    if len(h) > 5 and h[5] and target is not None:
        d = h[5].get(target, h[5].get(str(target)))
        if d is not None:
            return d
    st = card_stats(h)
    d = st.get("calculateddamage") if st.get("calculateddamage") is not None else st.get("damage")
    return d or 0


def card_blk(h):
    st = card_stats(h)
    b = st.get("calculatedblock") if st.get("calculatedblock") is not None else st.get("block")
    return b or 0


class CombatEncoder:
    """Sparse features of a combat state (record.combat_state + the fight header's run
    state) and of each legal action. Offsets are fixed by the vocabularies."""

    def __init__(self, cards, powers, enemies, relics, pots, intents):
        self.v = {"cards": cards, "powers": powers, "enemies": enemies, "relics": relics, "pots": pots,
                  "intents": intents}
        nc, npw, ne = len(cards), len(powers), len(enemies)
        self.n_scalar = 26
        o = self.n_scalar
        self.o_hand, o = o, o + 2 * nc
        self.o_deck, o = o, o + 2 * nc
        self.o_sel, o = o, o + nc
        self.o_ppow, o = o, o + 2 * npw
        self.o_rel, o = o, o + len(relics)
        self.o_pot, o = o, o + len(pots)
        self.slot = ne + 8 + len(intents) + npw
        self.o_en, o = o, o + MAX_ENEMIES * self.slot
        self.state_dim = o
        # action: type(6) card(nc) up cost(3) target slot(5) target numbers(6) potion selection(nc)
        self.n_atype = 6
        a = self.n_atype
        self.a_card, a = a, a + nc
        self.a_num, a = a, a + 16
        self.a_slot, a = a, a + MAX_ENEMIES
        self.a_tgt, a = a, a + 6
        self.a_pot, a = a, a + len(pots)
        self.a_sel, a = a, a + nc
        self.action_dim = a

    @classmethod
    def build(cls, runs):
        cards, powers, enemies, relics, pots, intents = (set() for _ in range(6))
        for rows in runs:
            for r in rows:
                if r["k"] == "f":
                    cards.update(base(t) for t in r["run"]["deck"])
                    relics.update(r["run"]["relics"])
                    pots.update(r["run"]["pots"])
                elif r["k"] == "c":
                    s = r["s"]
                    cards.update(base(h[0]) for h in s["hand"])
                    cards.update(base(t) for t in s.get("sel") or [])
                    powers.update(s["pw"])
                    pots.update(s["pots"])
                    for e in s["es"]:
                        enemies.add(e[0])
                        powers.update(e[4])
                        intents.update(i[0] for i in e[5])
        return cls(*(Vocab(x) for x in (cards, powers, enemies, relics, pots, intents)))

    def to_json(self):
        return {k: v.to_json() for k, v in self.v.items()}

    @classmethod
    def from_json(cls, d):
        return cls(*(Vocab.from_json(d[k]) for k in ("cards", "powers", "enemies", "relics", "pots", "intents")))

    def _slots(self, s):
        # Waterfall Giant's ~1e9-HP Steam Eruption placeholder is not a real enemy
        es = sorted((e for e in s["es"] if (e[1] or 0) < 10 ** 8), key=lambda e: e[6] if e[6] is not None else 99)
        return es[:MAX_ENEMIES]

    def state(self, s, run):
        """-> (indices, values)"""
        V = self.v
        idx, val = [], []

        def put(i, x):
            if x:
                idx.append(i)
                val.append(float(x))

        hp, mhp = s["hp"] or 0, max(1, s["mhp"] or 1)
        es = self._slots(s)
        inc = incoming(es)
        need = max(0, inc - (s["blk"] or 0))
        ctx_room = run.get("room")
        sc = [hp / mhp, hp / 100, (s["blk"] or 0) / 50, (s["en"] or 0) / 5, (s["men"] or 0) / 5, (s["r"] or 0) / 20,
              (s["dn"] or 0) / 30, (s["xn"] or 0) / 30, need / 40, inc / 40, len(es) / 5, len(s["hand"]) / 10,
              s["d"] == "card_select", run.get("act") == 1, run.get("act") == 2, run.get("act") == 3,
              ctx_room == "Monster", ctx_room == "Elite", ctx_room == "Boss", (run.get("asc") or 0) / 10,
              need >= hp, (s.get("min") or 0) / 3, (s.get("max") or 0) / 3, len(run.get("deck") or []) / 30,
              sum(card_dmg(h) for h in s["hand"] if h[2]) / 50, sum(card_blk(h) for h in s["hand"] if h[2]) / 30]
        for i, x in enumerate(sc):
            put(i, x)
        nc = len(V["cards"])
        hand = defaultdict(float)
        for h in s["hand"]:
            j = V["cards"][base(h[0])]
            hand[self.o_hand + j] += 1
            if h[0].endswith("+"):
                hand[self.o_hand + nc + j] += 1
        for t in run.get("deck") or []:
            j = V["cards"][base(t)]
            hand[self.o_deck + j] += 0.25
            if t.endswith("+"):
                hand[self.o_deck + nc + j] += 0.25
        for t in s.get("sel") or []:
            hand[self.o_sel + V["cards"][base(t)]] += 1
        npw = len(V["powers"])
        for k, a in s["pw"].items():
            j = V["powers"][k]
            hand[self.o_ppow + j] += (a or 0) / 5
            hand[self.o_ppow + npw + j] = 1.0
        for r in run.get("relics") or []:
            hand[self.o_rel + V["relics"][r]] = 1.0
        for p in s["pots"]:
            hand[self.o_pot + V["pots"][p]] += 1
        ni = len(V["intents"])
        ne = len(V["enemies"])
        for k, e in enumerate(es):
            o = self.o_en + k * self.slot
            name, ehp, emhp, eblk, epw, ints = e[0], e[1] or 0, max(1, e[2] or 1), e[3] or 0, e[4], e[5]
            hand[o + V["enemies"][name]] = 1.0
            dmg = sum((d or 0) * (h or 1) for t, d, h in ints if t in HITS and d is not None)
            hits = sum((h or 1) for t, d, h in ints if t in HITS and d is not None)
            nums = [ehp / 100, ehp / emhp, eblk / 50, dmg / 40, hits / 5, 1.0, "Minion" in epw, ehp <= 10]
            for i, x in enumerate(nums):
                if x:
                    hand[o + ne + i] = float(x)
            for t, d, h in ints:
                hand[o + ne + 8 + V["intents"][t]] = 1.0
            for pk, a in epw.items():
                hand[o + ne + 8 + ni + V["powers"][pk]] += (a or 0) / 5 if a is not None else 1.0
        for i, x in hand.items():
            put(i, x)
        return idx, val

    def action(self, s, a):
        V = self.v
        name, args = a[0], a[1] or {}
        idx, val = [], []

        def put(i, x):
            if x:
                idx.append(i)
                val.append(float(x))

        types = ("end_turn", "play_card", "use_potion", "select_cards", "skip_select")
        put(types.index(name) if name in types else 5, 1.0)
        hand = {h[3]: h for h in s["hand"]}
        es = self._slots(s)
        slot_of = {e[6]: k for k, e in enumerate(es)}
        if name == "play_card":
            h = hand.get(args.get("card_index"))
            if h is not None:
                put(self.a_card + V["cards"][base(h[0])], 1.0)
                cost = h[1]
                put(self.a_num + 0, h[0].endswith("+"))
                put(self.a_num + 1, (cost if isinstance(cost, int) and cost >= 0 else (s["en"] or 0)) / 3)
                put(self.a_num + 2, cost == 0)
                put(self.a_num + 3, not isinstance(cost, int) or cost < 0)
                st_ = card_stats(h)
                t_ = args.get("target_index")
                if t_ is None and len(es) == 1:
                    t_ = es[0][6]
                dmg = card_dmg(h, t_)
                hits = st_.get("repeat") or 1
                blk = card_blk(h)
                need = max(0, incoming(es) - (s["blk"] or 0))
                tgt = next((e for e in es if e[6] == t_), None)
                eff = ((tgt[1] or 0) + (tgt[3] or 0)) if tgt else 0
                kws = h[6] if len(h) > 6 else []
                for i, x in enumerate([dmg / 30, dmg * hits / 40, tgt is not None and dmg * hits >= eff > 0,
                                       blk / 20, min(blk, need) / 20, max(0, blk - need) / 20, "Exhaust" in kws,
                                       "Ethereal" in kws, (st_.get("cards") or 0) / 3, (st_.get("energy") or 0) / 2,
                                       (st_.get("vulnerablepower") or 0) / 3, (st_.get("weakpower") or 0) / 3]):
                    put(self.a_num + 4 + i, x)
        if name == "use_potion":
            pots = s["pots"]
            pi = args.get("potion_index")
            # potion_index is the slot; the compact state lists filled slots in order
            if isinstance(pi, int) and pots:
                put(self.a_pot + V["pots"][pots[min(pi, len(pots) - 1)]], 1.0)
        t = args.get("target_index")
        if t is not None and t in slot_of:
            k = slot_of[t]
            e = es[k]
            put(self.a_slot + k, 1.0)
            ehp, emhp = e[1] or 0, max(1, e[2] or 1)
            dmg = sum((d or 0) * (h or 1) for tt, d, h in e[5] if tt in HITS and d is not None)
            for i, x in enumerate([ehp / 100, ehp / emhp, (e[3] or 0) / 50, dmg / 40, "Vulnerable" in e[4],
                                   ehp == min(x[1] or 0 for x in es)]):
                put(self.a_tgt + i, x)
        if name == "select_cards":
            sel = s.get("sel") or []
            for i in str(args.get("indices", "")).split(","):
                if i.strip().isdigit() and int(i) < len(sel):
                    put(self.a_sel + V["cards"][base(sel[int(i)])], 1.0)
        return idx, val


class PolicyNet(nn.Module):
    """logit(s, a) = MLP(relu(Ws x_s + Wa x_a)): the state half is computed once per decision."""

    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.ws = nn.Linear(state_dim, hidden)
        self.wa = nn.Linear(action_dim, hidden, bias=False)
        self.ss = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, xs, xa, owner):
        """xs: [D, state_dim], xa: [A, action_dim], owner: [A] decision index of each action."""
        hs = self.ss(self.ws(xs))
        return self.head(hs[owner] + self.wa(xa)).squeeze(-1)


def _csr(rows, dim):
    """rows: list of (idx, val) -> torch sparse CSR-like (crow, col, val) numpy arrays."""
    crow = np.zeros(len(rows) + 1, dtype=np.int64)
    cols, vals = [], []
    for i, (ix, vx) in enumerate(rows):
        cols.extend(ix)
        vals.extend(vx)
        crow[i + 1] = crow[i] + len(ix)
    return crow, np.asarray(cols, dtype=np.int64), np.asarray(vals, dtype=np.float32), dim


def _dense(csr, sel):
    crow, cols, vals, dim = csr
    out = np.zeros((len(sel), dim), dtype=np.float32)
    for r, i in enumerate(sel):
        a, b = crow[i], crow[i + 1]
        out[r, cols[a:b]] = vals[a:b]
    return torch.from_numpy(np.clip(out, -5.0, 5.0))


SOFT_TAU = 4.0    # temperature (score units) of the search-score target distribution
SOFT_BETA = 0.5   # weight of the soft (search scores) loss next to the chosen-move loss


def soft_target(r):
    """AlphaGo Zero-style target from the planner's best leaf score per first action
    (only on a plan's first step): softmax(q / tau) over the scored actions, 0 elsewhere."""
    q = (r.get("x") or {}).get("q")
    if not q or len(q) < 2:
        return None
    keys = [json.dumps([a[0], a[1]], sort_keys=True) for a in r["A"]]
    z = np.full(len(keys), -np.inf)
    for i, k in enumerate(keys):
        if k in q:
            z[i] = q[k] / SOFT_TAU
    if not np.isfinite(z).any():
        return None
    z = np.exp(z - z[np.isfinite(z)].max())
    return (z / z.sum()).astype(np.float32)


def policy_dataset(runs, enc):
    """Decisions with >1 legal action and a known choice. Returns state rows, action rows,
    action->decision pointers, chosen offsets, run keys, soft targets (per action, NaN where
    the decision has none)."""
    S, A, start, chosen, keys, soft = [], [], [], [], [], []
    for key, rows in runs.items():
        header = None
        for r in rows:
            if r["k"] == "f":
                header = r["run"]
            elif r["k"] == "c" and header is not None and r["a"] >= 0 and len(r["A"]) > 1:
                S.append(enc.state(r["s"], header))
                start.append(len(A))
                chosen.append(r["a"])
                keys.append(key)
                t = soft_target(r)
                soft.extend(t if t is not None else [np.nan] * len(r["A"]))
                for a in r["A"]:
                    A.append(enc.action(r["s"], a))
    start.append(len(A))
    return S, A, np.asarray(start), np.asarray(chosen), keys, np.asarray(soft, dtype=np.float32)


def _batches(order, size):
    for i in range(0, len(order), size):
        yield order[i:i + size]


def policy_logits(model, Scsr, Acsr, start, sel):
    xs = _dense(Scsr, sel)
    a_idx, owner = [], []
    for k, d in enumerate(sel):
        a_idx.extend(range(start[d], start[d + 1]))
        owner.extend([k] * (start[d + 1] - start[d]))
    xa = _dense(Acsr, a_idx)
    return model(xs, xa, torch.as_tensor(owner)), np.asarray(owner)


def policy_loss(model, Scsr, Acsr, start, chosen, sel, soft=None):
    logits, owner = policy_logits(model, Scsr, Acsr, start, sel)
    n = len(sel)
    sizes = np.asarray([start[d + 1] - start[d] for d in sel])
    m = sizes.max()
    pad = torch.full((n, m), -1e9)
    col = np.concatenate([np.arange(s) for s in sizes])
    pad[torch.as_tensor(owner), torch.as_tensor(col)] = logits
    target = torch.as_tensor(chosen[sel])
    loss = F.cross_entropy(pad, target)
    if soft is not None and SOFT_BETA:
        a_idx = np.concatenate([np.arange(start[d], start[d + 1]) for d in sel])
        t = np.zeros((n, m), dtype=np.float32)
        t[owner, col] = soft[a_idx]
        has = ~np.isnan(t).any(1)
        if has.any():
            logp = F.log_softmax(pad[torch.as_tensor(has)], dim=1)
            loss = loss + SOFT_BETA * -(torch.from_numpy(t[has]) * logp.clamp(min=-30)).sum(1).mean()
    acc = (pad.argmax(1) == target).float().mean().item()
    return loss, acc


def train_policy(paths, out, epochs=8, hidden=256, lr=7e-4, batch=256, max_runs=None):
    t0 = time.time()
    runs = load_runs(paths)
    keys = sorted(runs)
    if max_runs:
        keys = keys[:max_runs]
        runs = {k: runs[k] for k in keys}
    enc = CombatEncoder.build(runs.values())
    tr, va = split_runs(runs)
    data = {}
    for name, part in (("train", tr), ("val", va)):
        S, A, start, chosen, _, soft = policy_dataset(part, enc)
        data[name] = (_csr(S, enc.state_dim), _csr(A, enc.action_dim), start, chosen, soft)
        print(name, "soft targets", int(sum(1 for d in range(len(chosen)) if not np.isnan(soft[start[d]]))), flush=True)
        print(name, "decisions", len(S), "actions", len(A), "runs", len(part), flush=True)
    print("encoded in", round(time.time() - t0), "s; dims", enc.state_dim, enc.action_dim, flush=True)
    torch.manual_seed(0)
    model = PolicyNet(enc.state_dim, enc.action_dim, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    Str, Atr, st_tr, ch_tr, so_tr = data["train"]
    Sva, Ava, st_va, ch_va, _ = data["val"]
    n_tr = len(ch_tr)
    best = (-1, None)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        model.train()
        order = np.random.RandomState(ep).permutation(n_tr)
        tl, ta, nb = 0.0, 0.0, 0
        for sel in _batches(order, batch):
            loss, acc = policy_loss(model, Str, Atr, st_tr, ch_tr, sel, so_tr)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item()
            ta += acc
            nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            vl, vac, vn = 0.0, 0.0, 0
            for sel in _batches(np.arange(len(ch_va)), 1024):
                loss, acc = policy_loss(model, Sva, Ava, st_va, ch_va, sel)
                vl += loss.item() * len(sel)
                vac += acc * len(sel)
                vn += len(sel)
        rand = float(np.mean([1.0 / (st_va[d + 1] - st_va[d]) for d in range(len(ch_va))]))
        print(json.dumps({"epoch": ep, "train_loss": round(tl / nb, 4), "train_acc": round(ta / nb, 3),
                          "val_loss": round(vl / vn, 4), "val_acc": round(vac / vn, 3), "val_random": round(rand, 3),
                          "s": round(time.time() - t0)}), flush=True)
        if vac / vn > best[0]:
            best = (vac / vn, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"kind": "policy", "enc": enc.to_json(), "hidden": hidden, "state": model.state_dict(),
                "val_acc": best[0], "sources": paths}, out)
    print("saved", out, "val_acc", round(best[0], 3))


class PolicyPilot:
    """The policy network as a player: score legal actions of a combat state."""

    def __init__(self, path=os.path.join(HERE, "models", "policy.pt")):
        blob = torch.load(path, weights_only=False)
        self.enc = CombatEncoder.from_json(blob["enc"])
        self.model = PolicyNet(self.enc.state_dim, self.enc.action_dim, blob["hidden"])
        self.model.load_state_dict(blob["state"])
        self.model.eval()

    def scores(self, cs, run, acts):
        S = _csr([self.enc.state(cs, run)], self.enc.state_dim)
        A = _csr([self.enc.action(cs, [a[0], a[1]]) for a in acts], self.enc.action_dim)
        with torch.no_grad():
            return self.model(_dense(S, [0]), _dense(A, range(len(acts))),
                              torch.zeros(len(acts), dtype=torch.long)).numpy()


class NetPolicy:
    """Drop-in for combat.quick_policy: one forward pass and one sim call per action.
    temperature > 0 samples from softmax(logits / T) (self-play exploration)."""

    def __init__(self, path=os.path.join(HERE, "models", "policy.pt"), temperature=0.0, seed=0):
        self.pilot = PolicyPilot(path)
        self.temperature = temperature
        self.rng = random.Random(seed)

    def actions(self, st):
        from combat import potion_actions
        from engine import legal_actions
        d = st.get("decision")
        if d == "combat_play":
            return legal_actions(st) + potion_actions(st)
        if d == "card_select":
            return legal_actions(st)
        return None

    def __call__(self, st):
        from combat import quick_policy
        from record import combat_state, run_state
        acts = self.actions(st)
        if not acts:
            return quick_policy(st)
        if len(acts) == 1:
            return acts[0]
        z = self.pilot.scores(combat_state(st), run_state(st), acts)
        if self.temperature > 0:
            p = np.exp((z - z.max()) / self.temperature)
            return acts[self.rng.choices(range(len(acts)), weights=p.tolist())[0]]
        return acts[int(np.argmax(z))]


# ---------------------------------------------------------------- run value

class ValueNet(nn.Module):
    """Heads: remaining floors (scaled by 1/17) and P(clear the current act)."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, hidden),
                                  nn.ReLU(), nn.Dropout(0.1))
        self.rem = nn.Linear(hidden, 1)
        self.clear = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.rem(h).squeeze(-1), self.clear(h).squeeze(-1)


def value_dataset(runs, enc):
    X, rem, clear, keys, bucket = [], [], [], [], []
    for key, rows in runs.items():
        o = rows[-1]
        fin = final_progress(o)
        end_act = 4 if o.get("victory") else (o.get("act") or 1)
        for r in rows:
            s = r["s"] if r["k"] == "d" else r["run"] if r["k"] == "f" else None
            if s is None or s.get("act") is None:
                continue
            X.append(enc.encode(s))
            now = progress(s["act"], s["floor"])
            rem.append(max(0, fin - now) / FLOORS_PER_ACT)
            clear.append(float(end_act > s["act"]))
            keys.append(key)
            bucket.append((s["act"], s["floor"]))
    return np.stack(X), np.asarray(rem, dtype=np.float32), np.asarray(clear, dtype=np.float32), keys, bucket


def train_value(paths, out, epochs=30, hidden=256, lr=1e-3):
    runs = load_runs(paths)
    states = [r["s"] if r["k"] == "d" else r["run"] for rows in runs.values() for r in rows if r["k"] in ("d", "f")]
    enc = RunEncoder.build(states)
    tr, va = split_runs(runs)
    Xtr, Rtr, Ctr, _, Btr = value_dataset(tr, enc)
    Xva, Rva, Cva, _, Bva = value_dataset(va, enc)
    print("runs", len(tr), len(va), "states", len(Xtr), len(Xva), "dim", enc.dim, flush=True)
    # baseline: mean by (act, floor) bucket
    acc = defaultdict(list)
    for b, r, c in zip(Btr, Rtr, Ctr):
        acc[b].append((r, c))
    mr = float(np.mean(Rtr))
    mc = float(np.mean(Ctr))
    base_r = np.asarray([np.mean([x[0] for x in acc[b]]) if b in acc else mr for b in Bva])
    base_c = np.asarray([np.mean([x[1] for x in acc[b]]) if b in acc else mc for b in Bva]).clip(1e-3, 1 - 1e-3)
    print("baseline (act,floor) val: rem_mse", round(float(np.mean((base_r - Rva) ** 2)), 4),
          "clear_bce", round(float(-np.mean(Cva * np.log(base_c) + (1 - Cva) * np.log(1 - base_c))), 4), flush=True)
    torch.manual_seed(0)
    model = ValueNet(enc.dim, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, Rt, Ct = torch.from_numpy(Xtr), torch.from_numpy(Rtr), torch.from_numpy(Ctr)
    Xv, Rv, Cv = torch.from_numpy(Xva), torch.from_numpy(Rva), torch.from_numpy(Cva)
    best = (1e9, None)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(ep))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            pr, pc = model(Xt[b])
            loss = F.mse_loss(pr, Rt[b]) + F.binary_cross_entropy_with_logits(pc, Ct[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pr, pc = model(Xv)
            mse = F.mse_loss(pr, Rv).item()
            bce = F.binary_cross_entropy_with_logits(pc, Cv).item()
        print(json.dumps({"epoch": ep, "val_rem_mse": round(mse, 4), "val_clear_bce": round(bce, 4)}), flush=True)
        if mse + bce < best[0]:
            best = (mse + bce, {k: v.clone() for k, v in model.state_dict().items()}, mse, bce)
    model.load_state_dict(best[1])
    model.eval()
    # agreement with the deck test on held-out card rewards: does the value network rank
    # the options the way the imagined boss fights do?
    xs, ys, agree, n = [], [], 0, 0
    for rows in va.values():
        for r in rows:
            if r["k"] != "d" or r["dec"] != "card_reward" or not (r.get("x") or {}).get("test"):
                continue
            test = r["x"]["test"]
            arms = [None if k == "None" else int(k) for k in test]
            cards = r.get("cards") or []
            decks = [r["s"]["deck"] if a is None else r["s"]["deck"] + [cards[a]] for a in arms if a is None or a < len(cards)]
            if len(decks) != len(arms):
                continue
            with torch.no_grad():
                pr, _ = model(torch.from_numpy(np.stack([enc.encode(r["s"], d) for d in decks])))
            pr = pr.numpy()
            tv = np.asarray([test[k] for k in test])
            xs.extend(pr - pr.mean())
            ys.extend(tv - tv.mean())
            agree += int(np.argmax(pr) == np.argmax(tv))
            n += 1
    if n:
        print("value vs deck test on", n, "held-out card rewards: corr", round(float(np.corrcoef(xs, ys)[0, 1]), 3),
              "argmax agreement", round(agree / n, 3), flush=True)
    torch.save({"kind": "value", "enc": enc.to_json(), "hidden": hidden, "state": model.state_dict(),
                "val_rem_mse": best[2], "val_clear_bce": best[3], "sources": paths}, out)
    print("saved", out, "val", round(best[2], 4), round(best[3], 4))


class RunValue:
    def __init__(self, path=os.path.join(HERE, "models", "value_run.pt")):
        blob = torch.load(path, weights_only=False)
        self.enc = RunEncoder.from_json(blob["enc"])
        self.model = ValueNet(self.enc.dim, blob["hidden"])
        self.model.load_state_dict(blob["state"])
        self.model.eval()

    def predict(self, states, decks=None):
        """-> (remaining floors, P(clear current act)) arrays for record.run_state dicts."""
        decks = decks or [None] * len(states)
        X = torch.from_numpy(np.stack([self.enc.encode(s, d) for s, d in zip(states, decks)]))
        with torch.no_grad():
            r, c = self.model(X)
        return r.numpy() * FLOORS_PER_ACT, torch.sigmoid(c).numpy()


# ---------------------------------------------------------------- RL (self-play policy improvement)

POTION_HP = 5.0     # HP-equivalent cost of drinking a potion (same as the planner's W["potion"])
DEATH_HP = 30.0     # extra HP-equivalent loss for dying
RETURN_SCALE = 20.0


class CombatValueNet(nn.Module):
    """v(s): expected HP change until the end of the fight (scaled), the RL baseline and
    the combat value network."""

    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, xs):
        return self.net(xs).squeeze(-1)


class CombatValue:
    """The combat value network saved with an RL policy: full sim state -> expected HP
    change until the end of the fight (potions and death priced as in fight_returns)."""

    def __init__(self, path):
        blob = torch.load(path, weights_only=False)
        self.enc = CombatEncoder.from_json(blob["enc"])
        self.net = CombatValueNet(self.enc.state_dim)
        self.net.load_state_dict(blob["value_state"])
        self.net.eval()

    def __call__(self, st):
        from record import combat_state, run_state
        S = _csr([self.enc.state(combat_state(st), run_state(st))], self.enc.state_dim)
        with torch.no_grad():
            return float(self.net(_dense(S, [0]))[0]) * RETURN_SCALE


def fight_returns(rows):
    """Per combat step: HP change from this step to the end of the fight, minus potion
    costs from here on and a death penalty (scaled by 1/RETURN_SCALE)."""
    out = {}
    fights = defaultdict(list)
    ends = {}
    for j, r in enumerate(rows):
        if r["k"] == "c":
            fights[r["i"]].append(j)
        elif r["k"] == "fe":
            ends[r["i"]] = r
    for i, steps in fights.items():
        e = ends.get(i)
        if e is None:
            continue
        hp_end = e.get("hp_last") if e.get("hp_last") is not None else e.get("hp_after") or 0
        won = e.get("won")
        pots_after = 0
        for j in reversed(steps):
            r = rows[j]
            if r["played"][0] == "use_potion":
                pots_after += 1
            hp = r["s"]["hp"] or 0
            g = (hp_end - hp) if won else (-hp - DEATH_HP)
            out[j] = (g - POTION_HP * pots_after) / RETURN_SCALE
    return out


def rl_dataset(runs, enc):
    S, A, start, chosen, lp_old, G = [], [], [], [], [], []
    for key, rows in runs.items():
        rets = fight_returns(rows)
        header = None
        for j, r in enumerate(rows):
            if r["k"] == "f":
                header = r["run"]
            elif r["k"] == "c" and header is not None and j in rets and r["a"] >= 0 and len(r["A"]) > 1 \
                    and (r.get("x") or {}).get("lp") is not None:
                S.append(enc.state(r["s"], header))
                start.append(len(A))
                chosen.append(r["a"])
                lp_old.append(r["x"]["lp"])
                G.append(rets[j])
                for a in r["A"]:
                    A.append(enc.action(r["s"], a))
    start.append(len(A))
    return S, A, np.asarray(start), np.asarray(chosen), np.asarray(lp_old, dtype=np.float32), np.asarray(G, dtype=np.float32)


def grouped(logits, owner, sizes):
    n, m = len(sizes), int(sizes.max())
    dev = logits.device
    pad = torch.full((n, m), -1e9, device=dev)
    col = np.concatenate([np.arange(s) for s in sizes])
    pad = pad.index_put((torch.as_tensor(owner, device=dev), torch.as_tensor(col, device=dev)), logits)
    return pad


def train_rl(paths, init, out, temperature=1.0, epochs=3, lr=1e-4, batch=512, clip=0.2, ent=0.01):
    """One PPO update. Advantages come from the value network as it was before this data
    (standard PPO): fitting the baseline on the same returns first would memorize them and
    leave no advantage signal. A fresh value network gets one light epoch first."""
    t0 = time.time()
    blob = torch.load(init, weights_only=False)
    enc = CombatEncoder.from_json(blob["enc"])
    model = PolicyNet(enc.state_dim, enc.action_dim, blob["hidden"])
    model.load_state_dict(blob["state"])
    vnet = CombatValueNet(enc.state_dim)
    vopt = torch.optim.Adam(vnet.parameters(), lr=1e-3)
    runs = load_runs(paths)
    S, A, start, chosen, lp_old, G = rl_dataset(runs, enc)
    Scsr, Acsr = _csr(S, enc.state_dim), _csr(A, enc.action_dim)
    n = len(chosen)
    print("rl decisions", n, "runs", len(runs), "mean return", round(float(G.mean()), 3), "encoded",
          round(time.time() - t0), "s", flush=True)

    def fit_value(ep):
        order = np.random.RandomState(100 + ep).permutation(n)
        tot, nb = 0.0, 0
        for sel in _batches(order, batch):
            vloss = F.mse_loss(vnet(_dense(Scsr, sel)), torch.from_numpy(G[sel]))
            vopt.zero_grad()
            vloss.backward()
            vopt.step()
            tot += vloss.item()
            nb += 1
        return tot / nb

    if blob.get("value_state"):
        vnet.load_state_dict(blob["value_state"])
    else:
        print(json.dumps({"value_warmup": round(fit_value(-1), 4)}), flush=True)
    with torch.no_grad():
        V = np.concatenate([vnet(_dense(Scsr, sel)).numpy() for sel in _batches(np.arange(n), 4096)])
    ADV = G - V
    print(json.dumps({"adv_std": round(float(ADV.std()), 4), "value_mse_before": round(float((ADV ** 2).mean()), 4),
                      "return_var": round(float(G.var()), 4)}), flush=True)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        order = np.random.RandomState(ep).permutation(n)
        stats = defaultdict(float)
        nb = 0
        for sel in _batches(order, batch):
            adv = torch.from_numpy(ADV[sel].astype(np.float32))
            logits, owner = policy_logits(model, Scsr, Acsr, start, sel)
            sizes = np.asarray([start[d + 1] - start[d] for d in sel])
            pad = grouped(logits / temperature, owner, sizes)
            logp = F.log_softmax(pad, dim=1)
            target = torch.as_tensor(chosen[sel])
            lp_new = logp.gather(1, target[:, None]).squeeze(1)
            ratio = torch.exp(lp_new - torch.from_numpy(lp_old[sel]))
            pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            p = logp.exp()
            entropy = -(p * logp.clamp(min=-30)).sum(1).mean()
            loss = pg - ent * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats["pg"] += pg.item()
            stats["ent"] += entropy.item()
            stats["clipfrac"] += ((ratio - 1).abs() > clip).float().mean().item()
            stats["kl"] += (torch.from_numpy(lp_old[sel]) - lp_new).mean().item()
            nb += 1
        print(json.dumps({"epoch": ep, **{k: round(v / nb, 4) for k, v in stats.items()}, "s": round(time.time() - t0)}),
              flush=True)
    vl = [fit_value(ep) for ep in range(2)]   # the baseline for the next iteration
    print(json.dumps({"value_after": [round(x, 4) for x in vl]}), flush=True)
    torch.save({**blob, "state": model.state_dict(), "value_state": vnet.state_dict(),
                "rl_from": init, "rl_data": paths, "temperature": temperature}, out)
    print("saved", out)

# ---------------------------------------------------------------- card reward policy (imitation of the deck test)

def card_option(enc, tok):
    """Option features for a card reward: card one-hot, upgraded flag; tok None = skip."""
    x = np.zeros(len(enc.cards) + 2, dtype=np.float32)
    if tok is None:
        x[-1] = 1.0
    else:
        x[enc.cards[base(tok)]] = 1.0
        x[-2] = float(tok.endswith("+"))
    return x


def card_dataset(runs, enc):
    S, A, start, chosen = [], [], [], []
    for rows in runs.values():
        for r in rows:
            if r["k"] != "d" or r["dec"] != "card_reward" or r["a"] < 0 or len(r["A"]) < 2:
                continue
            if (r.get("x") or {}).get("explore"):
                continue
            S.append(enc.encode(r["s"]))
            start.append(len(A))
            chosen.append(r["a"])
            cards = r.get("cards") or []
            for a in r["A"]:
                i = (a[1] or {}).get("card_index")
                A.append(card_option(enc, cards[i] if a[0] == "select_card_reward" and i is not None and i < len(cards) else None))
    start.append(len(A))
    return np.stack(S), np.stack(A), np.asarray(start), np.asarray(chosen)


def train_card(paths, out, epochs=40, hidden=128, lr=1e-3):
    runs = load_runs(paths)
    states = [r["s"] if r["k"] == "d" else r["run"] for rows in runs.values() for r in rows if r["k"] in ("d", "f")]
    enc = RunEncoder.build(states)
    tr, va = split_runs(runs)
    data = {k: card_dataset(v, enc) for k, v in (("train", tr), ("val", va))}
    print("card rewards", len(data["train"][3]), len(data["val"][3]), flush=True)
    torch.manual_seed(0)
    model = PolicyNet(enc.dim, len(enc.cards) + 2, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    def run(part, train):
        S, A, start, chosen = data[part]
        order = np.random.permutation(len(chosen)) if train else np.arange(len(chosen))
        tot, acc, n = 0.0, 0.0, 0
        for sel in _batches(order, 128):
            a_idx, owner = [], []
            for k, d in enumerate(sel):
                a_idx.extend(range(start[d], start[d + 1]))
                owner.extend([k] * (start[d + 1] - start[d]))
            logits = model(torch.from_numpy(S[sel]), torch.from_numpy(A[a_idx]), torch.as_tensor(owner))
            sizes = np.asarray([start[d + 1] - start[d] for d in sel])
            pad = grouped(logits, np.asarray(owner), sizes)
            target = torch.as_tensor(chosen[sel])
            loss = F.cross_entropy(pad, target)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * len(sel)
            acc += (pad.argmax(1) == target).float().sum().item()
            n += len(sel)
        return tot / max(1, n), acc / max(1, n)

    best = (-1, None)
    for ep in range(epochs):
        model.train()
        tl, ta = run("train", True)
        model.eval()
        with torch.no_grad():
            vl, vacc = run("val", False)
        if ep % 5 == 4 or ep == epochs - 1:
            print(json.dumps({"epoch": ep, "train_acc": round(ta, 3), "val_loss": round(vl, 4), "val_acc": round(vacc, 3)}),
                  flush=True)
        if vacc > best[0]:
            best = (vacc, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    torch.save({"kind": "card", "enc": enc.to_json(), "hidden": hidden, "state": model.state_dict(),
                "val_acc": best[0], "sources": paths}, out)
    print("saved", out, "val_acc", round(best[0], 3))


SKIP_BIAS = float(os.environ.get("CARD_SKIP_BIAS", "0"))


class CardPolicy:
    """Card reward choice by the imitation network (fast stand-in for the deck test)."""

    def __init__(self, path=os.path.join(HERE, "models", "card.pt")):
        blob = torch.load(path, weights_only=False)
        self.enc = RunEncoder.from_json(blob["enc"])
        self.model = PolicyNet(self.enc.dim, len(self.enc.cards) + 2, blob["hidden"])
        self.model.load_state_dict(blob["state"])
        self.model.eval()

    def scores(self, st):
        from engine import legal_actions
        from record import card_tok, run_state
        acts = legal_actions(st)
        by_index = {c["index"]: c for c in st.get("cards") or []}
        opts = [card_option(self.enc, card_tok(by_index[a[1]["card_index"]]) if a[0] == "select_card_reward" else None)
                for a in acts]
        with torch.no_grad():
            z = self.model(torch.from_numpy(self.enc.encode(run_state(st))[None]), torch.from_numpy(np.stack(opts)),
                           torch.zeros(len(acts), dtype=torch.long)).numpy()
        if SKIP_BIAS:   # experiments: shift the skip option's logit
            for i, a in enumerate(acts):
                if a[0] == "skip_card_reward":
                    z[i] += SKIP_BIAS
        return acts, z

    def __call__(self, st):
        acts, z = self.scores(st)
        return acts[int(np.argmax(z))]


def train_card_rl(paths, init, out, temperature=1.0, epochs=3, lr=1e-4, batch=256, clip=0.2, ent=0.01):
    """PPO on card reward choices in self-play (the RL stage for deck building). Return: the
    run's final progress (floors), baseline: mean final progress of runs at the same
    (act, floor) in this data. NETS_DEVICE=cuda trains on the GPU."""
    t0 = time.time()
    lr *= float(os.environ.get("NETS_LR_SCALE", "1"))   # fleet xdeckrl lr_scale
    dev = torch.device(os.environ.get("NETS_DEVICE", "cpu"))
    blob = torch.load(init, weights_only=False)
    enc = RunEncoder.from_json(blob["enc"])
    model = PolicyNet(enc.dim, len(enc.cards) + 2, blob["hidden"])
    model.load_state_dict(blob["state"])
    model.to(dev)
    runs = load_runs(paths)
    t_load = time.time()
    items, by_bucket = [], defaultdict(list)
    for rows in runs.values():
        fin = final_progress(rows[-1])
        for r in rows:
            if r["k"] == "d" and r["dec"] == "card_reward" and (r.get("x") or {}).get("lp") is not None and len(r["A"]) > 1:
                key = (r["s"]["act"], r["s"]["floor"])
                items.append((r, fin, key))
                by_bucket[key].append(fin)
    base = {k: float(np.mean(v)) for k, v in by_bucket.items()}
    use_v = all(r.get("V") is not None for r, _, _ in items) and items
    S, A, start, chosen, lp_old, adv = [], [], [], [], [], []
    for r, fin, key in items:
        S.append(enc.encode(r["s"]))
        start.append(len(A))
        chosen.append(r["a"])
        lp_old.append(r["x"]["lp"])
        adv.append(fin - (r["V"] if use_v else base[key]))
        cards = r.get("cards") or []
        for a in r["A"]:
            i = (a[1] or {}).get("card_index")
            A.append(card_option(enc, cards[i] if a[0] == "select_card_reward" and i is not None and i < len(cards) else None))
    start.append(len(A))
    S, A, start, chosen = np.stack(S), np.stack(A), np.asarray(start), np.asarray(chosen)
    lp_old, adv = np.asarray(lp_old, dtype=np.float32), np.asarray(adv, dtype=np.float32)
    print("card rl decisions", len(chosen), "runs", len(runs), "adv std", round(float(adv.std()), 3), flush=True)
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    t_enc = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        order = np.random.RandomState(ep).permutation(len(chosen))
        stats, nb = defaultdict(float), 0
        for sel in _batches(order, batch):
            a_idx, owner = [], []
            for k, d in enumerate(sel):
                a_idx.extend(range(start[d], start[d + 1]))
                owner.extend([k] * (start[d + 1] - start[d]))
            logits = model(torch.from_numpy(S[sel]).to(dev), torch.from_numpy(A[a_idx]).to(dev),
                           torch.as_tensor(owner, device=dev))
            sizes = np.asarray([start[d + 1] - start[d] for d in sel])
            logp = F.log_softmax(grouped(logits / temperature, np.asarray(owner), sizes), dim=1)
            lp_new = logp.gather(1, torch.as_tensor(chosen[sel], device=dev)[:, None]).squeeze(1)
            ratio = torch.exp(lp_new - torch.from_numpy(lp_old[sel]).to(dev))
            ad = torch.from_numpy(adv[sel]).to(dev)
            pg = -torch.min(ratio * ad, ratio.clamp(1 - clip, 1 + clip) * ad).mean()
            entropy = -(logp.exp() * logp.clamp(min=-30)).sum(1).mean()
            loss = pg - ent * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats["pg"] += pg.item()
            stats["ent"] += entropy.item()
            stats["clipfrac"] += ((ratio - 1).abs() > clip).float().mean().item()
            nb += 1
        print(json.dumps({"epoch": ep, **{k: round(v / nb, 4) for k, v in stats.items()}}), flush=True)
    t_train = time.time()
    model.to("cpu")
    torch.save({**blob, "state": model.state_dict(), "rl_from": init, "rl_data": paths}, out)
    print(json.dumps({"timing": {"device": str(dev), "load_s": round(t_load - t0, 2), "encode_s": round(t_enc - t_load, 2),
                                 "train_s": round(t_train - t_enc, 2)}}), flush=True)
    print("saved", out)



# ---------------------------------------------------------------- combat value from expert play

class ExpertValueNet(nn.Module):
    """Heads: P(win this fight) and HP change until its end (scaled), under the planner's
    play (trained on the expert games' outcomes, like AlphaGo's value network)."""

    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Dropout(0.2), nn.Linear(state_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.win = nn.Linear(hidden, 1)
        self.dhp = nn.Linear(hidden, 1)

    def forward(self, xs):
        h = self.body(xs)
        return self.win(h).squeeze(-1), self.dhp(h).squeeze(-1)


EV_PHASE = "all"   # "end": only end-of-turn states (the planner's leaves: hand played out)


def expert_value_dataset(runs, enc, rooms=None):
    S, W, D, keys = [], [], [], []
    for key, rows in runs.items():
        header, ends = {}, {}
        for r in rows:
            if r["k"] == "fe":
                ends[r["i"]] = r
        for r in rows:
            if r["k"] == "f":
                header[r["i"]] = r["run"]
            elif r["k"] == "c" and r["i"] in header and r["i"] in ends and r["s"]["d"] == "combat_play":
                h = header[r["i"]]
                if rooms and h.get("room") not in rooms:
                    continue
                if EV_PHASE == "end" and r["played"][0] != "end_turn":
                    continue
                e = ends[r["i"]]
                hp = r["s"]["hp"] or 0
                end = e.get("hp_last") if e.get("hp_last") is not None else e.get("hp_after") or 0
                won = bool(e.get("won"))
                S.append(enc.state(r["s"], h))
                W.append(float(won))
                D.append(((end - hp) if won else (-hp - DEATH_HP)) / RETURN_SCALE)
                keys.append(key)
    return S, np.asarray(W, dtype=np.float32), np.asarray(D, dtype=np.float32), keys


def train_expert_value(paths, out, init_enc, epochs=6, hidden=128, lr=3e-4, batch=512):
    t0 = time.time()
    enc = CombatEncoder.from_json(torch.load(init_enc, weights_only=False)["enc"])
    runs = load_runs(paths)
    tr, va = split_runs(runs)
    Str, Wtr, Dtr, _ = expert_value_dataset(tr, enc)
    Sva, Wva, Dva, _ = expert_value_dataset(va, enc)
    Ctr, Cva = _csr(Str, enc.state_dim), _csr(Sva, enc.state_dim)
    print("expert value states", len(Wtr), len(Wva), "win rate", round(float(Wtr.mean()), 3), "encoded",
          round(time.time() - t0), "s", flush=True)
    torch.manual_seed(0)
    model = ExpertValueNet(enc.state_dim, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best = (1e9, None)
    for ep in range(epochs):
        model.train()
        for sel in _batches(np.random.RandomState(ep).permutation(len(Wtr)), batch):
            pw, pd = model(_dense(Ctr, sel))
            loss = F.binary_cross_entropy_with_logits(pw, torch.from_numpy(Wtr[sel])) + \
                F.mse_loss(pd, torch.from_numpy(Dtr[sel]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pw, pd = [], []
            for sel in _batches(np.arange(len(Wva)), 4096):
                a, b = model(_dense(Cva, sel))
                pw.append(a)
                pd.append(b)
            pw, pd = torch.cat(pw), torch.cat(pd)
            bce = F.binary_cross_entropy_with_logits(pw, torch.from_numpy(Wva)).item()
            mse = F.mse_loss(pd, torch.from_numpy(Dva)).item()
        base_bce = float(-np.mean(Wva * np.log(Wtr.mean()) + (1 - Wva) * np.log(1 - Wtr.mean())))
        print(json.dumps({"epoch": ep, "val_win_bce": round(bce, 4), "base_bce": round(base_bce, 4), "val_dhp_mse": round(mse, 4),
                          "dhp_var": round(float(Dva.var()), 4)}), flush=True)
        if bce + mse < best[0]:
            best = (bce + mse, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    torch.save({"kind": "expert_value", "enc": enc.to_json(), "hidden": hidden, "state": model.state_dict(),
                "sources": paths}, out)
    print("saved", out)


class ExpertValue:
    """Full sim state -> (P(win), expected HP change to the fight's end) under expert play."""

    def __init__(self, path):
        blob = torch.load(path, weights_only=False)
        self.enc = CombatEncoder.from_json(blob["enc"])
        self.net = ExpertValueNet(self.enc.state_dim, blob["hidden"])
        self.net.load_state_dict(blob["state"])
        self.net.eval()

    def __call__(self, st):
        from record import combat_state, run_state
        S = _csr([self.enc.state(combat_state(st), run_state(st))], self.enc.state_dim)
        with torch.no_grad():
            w, d = self.net(_dense(S, [0]))
        return float(torch.sigmoid(w)[0]), float(d[0]) * RETURN_SCALE


# ---------------------------------------------------------------- value baseline for deck RL

def add_value_baseline(path, epochs=4, hidden=64):
    """Value baseline for deck RL with 2-fold cross-fitting: V = mean final progress at the
    same (act, floor) in the other fold + a small network's prediction of the residual from
    the run state (deck, relics, HP...), strongly regularized (raw networks overfit the
    sparse deck features and lost to the plain floor mean). Writes "V" into decision rows."""
    runs = load_runs([path])
    keys = sorted(runs)
    states = [r["s"] for rows in runs.values() for r in rows if r["k"] == "d" and r.get("s")]
    enc = RunEncoder.build(states)
    fold = {k: (zlib.crc32(str(k).encode()) & 1) for k in keys}
    rows_of = {0: [], 1: []}
    for k in keys:
        fin = final_progress(runs[k][-1])
        for r in runs[k]:
            if r["k"] == "d" and r.get("s") and r["s"].get("act") is not None:
                rows_of[fold[k]].append((r["s"], fin))
    buckets, models = {}, {}
    for f in (0, 1):
        by = defaultdict(list)
        for s_, fin in rows_of[f]:
            by[(s_["act"], s_["floor"])].append(fin)
        buckets[f] = ({k: float(np.mean(v)) for k, v in by.items()}, float(np.mean([y for _, y in rows_of[f]])))
        bmean, gmean = buckets[f]
        X = torch.from_numpy(np.stack([enc.encode(s_) for s_, _ in rows_of[f]]))
        Y = torch.from_numpy(np.asarray([fin - bmean.get((s_["act"], s_["floor"]), gmean) for s_, fin in rows_of[f]],
                                        dtype=np.float32)) / FLOORS_PER_ACT
        torch.manual_seed(f)
        m = nn.Sequential(nn.Dropout(0.2), nn.Linear(enc.dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        opt = torch.optim.Adam(m.parameters(), lr=5e-4, weight_decay=1e-3)
        for ep in range(epochs):
            m.train()
            perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(ep))
            for i in range(0, len(perm), 512):
                b = perm[i:i + 512]
                loss = F.mse_loss(m(X[b]).squeeze(-1), Y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        m.eval()
        models[f] = m
    se, se_b, n = 0.0, 0.0, 0
    with open(path) as fh:
        lines = fh.readlines()
    with open(path + ".tmp", "w") as fh:
        for line in lines:
            r = json.loads(line)
            k = (path, r.get("seed"))
            if r.get("k") == "d" and r.get("s") and r["s"].get("act") is not None and k in fold:
                g = 1 - fold[k]
                bmean, gmean = buckets[g]
                b = bmean.get((r["s"]["act"], r["s"]["floor"]), gmean)
                with torch.no_grad():
                    v = b + float(models[g](torch.from_numpy(enc.encode(r["s"])[None]))[0, 0]) * FLOORS_PER_ACT
                r["V"] = round(v, 3)
                fin = final_progress(runs[k][-1])
                se += (fin - v) ** 2
                se_b += (fin - b) ** 2
                n += 1
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    os.replace(path + ".tmp", path)
    print(json.dumps({"value_baseline_rows": n, "rmse": round((se / max(1, n)) ** 0.5, 3),
                      "rmse_floor_mean": round((se_b / max(1, n)) ** 0.5, 3)}), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["policy", "value", "rl", "card", "cardrl", "xvalue", "baseline"])
    ap.add_argument("--init", default=None, help="rl: policy to improve")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args()
    data = []
    for p in args.data:   # '@list.txt' stands for the paths listed there (Windows caps cmd lines at 8191 chars)
        if p.startswith("@"):
            with open(os.path.join(HERE, p[1:])) as fh:
                data += [l.strip() for l in fh if l.strip()]
        else:
            data.append(p)
    args.data = data
    torch.set_num_threads(int(os.environ.get("NETS_THREADS", "2")))
    if args.mode == "policy":
        train_policy(args.data, args.out or os.path.join(HERE, "models", "policy.pt"), epochs=args.epochs or 8,
                     hidden=args.hidden, max_runs=args.max_runs)
    elif args.mode == "baseline":
        add_value_baseline(args.data[0])
    elif args.mode == "xvalue":
        global EV_PHASE
        EV_PHASE = os.environ.get("EV_PHASE", "all")
        train_expert_value(args.data, args.out, args.init, epochs=args.epochs or 6, hidden=args.hidden if args.hidden != 256 else 128)
    elif args.mode == "cardrl":
        train_card_rl(args.data, args.init, args.out, temperature=args.temperature, epochs=args.epochs or 3)
    elif args.mode == "card":
        train_card(args.data, args.out or os.path.join(HERE, "models", "card.pt"), epochs=args.epochs or 40)
    elif args.mode == "rl":
        train_rl(args.data, args.init, args.out, temperature=args.temperature, epochs=args.epochs or 3)
    else:
        train_value(args.data, args.out or os.path.join(HERE, "models", "value_run.pt"), epochs=args.epochs or 30,
                    hidden=args.hidden)


if __name__ == "__main__":
    main()
