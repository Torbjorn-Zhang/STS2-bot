"""Compare fast pilots (no search) on a fight bench: mean graded score per pilot.

usage: pilotbench.py --tag act2a0 [--reps 2] [--pilots quick,net:models/policy.pt]
Pilots: quick (combat.quick_policy), distilled (distill.DistilledPolicy), net:<path>
(nets.NetPolicy). Same fights, same determinized seeds for all, so the comparison is
paired (the last pilot is compared with the first).
"""

import argparse
import glob
import json
import math
import os
import zlib

import decktest
from combat import alive, quick_policy
from engine import fresh_sim

HERE = os.path.dirname(os.path.abspath(__file__))


def fight(sim, meta, rep, pilot):
    sim.send({"cmd": "load_save", "path": meta["path"]})
    col, row = meta.get("node") or meta["boss_node"]
    st = sim.act("select_map_node", col=col, row=row)
    if st.get("decision") != "combat_play":
        return None
    sim.send({"cmd": "determinize", "seed": zlib.crc32(f"{meta['seed']}/{rep}".encode())})
    enemy_max = sum((e.get("max_hp") or 0) for e in st.get("enemies") or [])
    last, end = st, st
    for _ in range(500):
        if end.get("decision") not in ("combat_play", "card_select") or end.get("type") == "error":
            break
        last = end
        a = pilot(end)
        nxt = sim.act(a[0], **a[1])
        end = sim.act("end_turn") if nxt.get("type") == "error" and end.get("decision") == "combat_play" else nxt
    won = end.get("decision") != "game_over" or bool(end.get("victory"))
    hp = (last.get("player") or {}).get("hp") or 0
    left = 0 if won else sum((e.get("hp") or 0) for e in alive(last))
    return decktest.graded(won, hp if won else 0, left, enemy_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="act2a0")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--model", default=os.path.join(HERE, "models", "distill.json"))
    ap.add_argument("--pilots", default="quick,distilled")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    pilots = {}
    for name in args.pilots.split(","):
        if name == "quick":
            pilots[name] = quick_policy
        elif name == "distilled":
            from distill import DistilledPolicy
            pilots[name] = DistilledPolicy(args.model)
        elif name.startswith("net:"):
            import torch
            from nets import NetPolicy
            torch.set_num_threads(1)
            pilots[name] = NetPolicy(name[4:])
    metas = []
    for tag in args.tag.split(","):
        metas += [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "data", f"bossbench_{tag}", "*.save.json")))]
    metas = metas[:args.limit] if args.limit else metas
    sim = fresh_sim()
    res = {k: [] for k in pilots}
    for m in metas:
        for rep in range(args.reps):
            vals = {k: fight(sim, m, rep, p) for k, p in pilots.items()}
            if all(v is not None for v in vals.values()):
                for k, v in vals.items():
                    res[k].append(v)
    sim.close()
    names = list(pilots)
    d = [b - a for a, b in zip(res[names[0]], res[names[-1]])]
    mean = sum(d) / len(d)
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1)) / math.sqrt(len(d))
    wins = {k: sum(1 for v in res[k] if v >= 100) for k in names}
    print(json.dumps({"fights": len(d), **{k: round(sum(res[k]) / len(d), 2) for k in names}, "wins": wins,
                      "diff": round(mean, 2), "se": round(se, 2)}))


if __name__ == "__main__":
    main()
