"""Run-state value network: how good a deck will this run bring to act 3, judged from any moment
in acts 1-2 (playing for the future, learned from self-play like AlphaGo's value network).

The deck model judges a finished deck at a boss; it cannot tell what an act-1 deck grows into.
This net learns that from self-play records (run.py --record, the policy's own trajectories): every
decision state of a run gets the run's outcome as its target, z = log10(P + 1e-4), where P is the
deck model's P(beat both final bosses) for the deck, HP, relics and potions the run brought to act
3 (validated against real act-3 outcomes), and 0 for runs that died before act 3. Card picks can
then compare the states each option leads to.

  rsvalue.py train FILES... --out models/rsv1.pt [--models deck models] [--epochs 30]
"""
import argparse
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR = 1e-4


def expand(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(HERE, p))) if any(ch in p for ch in "*?") else [os.path.join(HERE, p)]
    return out


def outcome_targets(outcomes, model_paths):
    """seed -> z for each run outcome row."""
    import policy
    from deckvalue import DeckValue
    models = [DeckValue(os.path.join(HERE, p)) for p in model_paths]
    names = {v: k for k, v in policy.RELIC_IDS.items()}
    states, keys, z = [], [], {}
    for seed, o in outcomes.items():
        st = (o.get("act_states") or {}).get("3")
        if o.get("victory"):
            z[seed] = 0.0
        elif not st:
            z[seed] = math.log10(FLOOR)
        else:
            states.append({"deck": st.get("deck") or [], "relics": [names.get(r, r) for r in st.get("relics") or []],
                           "potions": st.get("potions") or [], "hp": st.get("hp"), "mhp": st.get("max_hp"), "act": 3,
                           "boss": st.get("boss")})
            keys.append(seed)
    if states:
        p = np.mean([m.win(states) for m in models], axis=0)
        for k, v in zip(keys, p):
            z[k] = math.log10(float(v) + FLOOR)
    return z


class Net(nn.Module):
    def __init__(self, dim, h=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(dim, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def load(files):
    runs = defaultdict(list)
    outcomes = {}
    for f in files:
        for l in open(f):
            r = json.loads(l)
            key = f"{os.path.basename(f)}/{r.get('seed')}"
            if r.get("k") == "o":
                outcomes[key] = r
            elif r.get("k") == "d" and r.get("s"):
                runs[key].append(r["s"])
    return {k: v for k, v in runs.items() if k in outcomes}, outcomes


def train(args):
    from nets import RunEncoder
    files = expand(args.files)
    runs, outcomes = load(files)
    z = outcome_targets(outcomes, args.models.split(","))
    blob = torch.load(os.path.join(HERE, args.enc_from), weights_only=False)
    enc = RunEncoder.from_json(blob["enc"])
    keys = sorted(runs)
    random.Random(0).shuffle(keys)
    nva = max(1, len(keys) // 10)
    va, tr = set(keys[:nva]), keys[nva:]

    def mk(ks):
        X, Y, A = [], [], []
        for k in ks:
            for s in runs[k]:
                X.append(enc.encode(s))
                Y.append(z[k])
                A.append(s.get("act") or 1)
        return torch.from_numpy(np.stack(X)), torch.tensor(Y, dtype=torch.float32), np.asarray(A)
    Xtr, Ytr, _ = mk(tr)
    Xva, Yva, Ava = mk(sorted(va))
    reach = np.mean([z[k] > math.log10(FLOOR) + 1e-9 for k in keys])
    print(json.dumps({"runs": len(keys), "states_train": len(Ytr), "states_val": len(Yva), "reach_act3": round(float(reach), 3),
                      "z_mean": round(float(Ytr.mean()), 3)}), flush=True)
    torch.manual_seed(args.seed)
    net = Net(Xtr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Ytr))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = ((net(Xtr[b]) - Ytr[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 4 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                p = net(Xva).numpy()
            y = Yva.numpy()
            res = {"epoch": ep + 1, "val_mse": round(float(((p - y) ** 2).mean()), 4),
                   "val_corr": round(float(np.corrcoef(p, y)[0, 1]), 3),
                   "base_mse": round(float(((y - float(Ytr.mean())) ** 2).mean()), 4)}
            for a in (1, 2):
                m = Ava == a
                if m.sum() > 10:
                    res[f"corr_act{a}"] = round(float(np.corrcoef(p[m], y[m])[0, 1]), 3)
            print(json.dumps(res), flush=True)
    if args.out:
        torch.save({"enc": blob["enc"], "state": net.state_dict(), "dim": Xtr.shape[1]}, os.path.join(HERE, args.out))
    print(json.dumps({"saved": args.out, "runs": len(keys), "reach_act3": round(float(reach), 3), **res}))   # fleet logs this line


class RSValue:
    def __init__(self, path):
        from nets import RunEncoder
        blob = torch.load(os.path.join(HERE, path), weights_only=False)
        self.enc = RunEncoder.from_json(blob["enc"])
        self.net = Net(blob["dim"])
        self.net.load_state_dict(blob["state"])
        self.net.eval()

    def value(self, run_states):
        with torch.no_grad():
            return self.net(torch.from_numpy(np.stack([self.enc.encode(s) for s in run_states]))).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--models", default="models/dz_boss_s0.pt,models/dz_boss_s1.pt,models/dz_boss_s2.pt,models/dz_boss_s3.pt")
    ap.add_argument("--enc-from", default="models/frozen/deckh_card4.pt", help="RunEncoder vocabulary (the card net's)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("NETS_THREADS", "2")))
    train(args)


if __name__ == "__main__":
    sys.exit(main())
