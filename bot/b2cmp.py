"""Paired comparison on a fight bench by room: deaths and HP lost (a death counts as all HP).
usage: b2cmp.py base.jsonl other.jsonl ... [--tag act2a10]"""
import glob
import json
import math
import statistics as S
import sys

tag = next((a.split("=")[1] for a in sys.argv if a.startswith("--tag=")), "act2a10")
files = [a for a in sys.argv[1:] if not a.startswith("--")]
metas = {}
for p in glob.glob(f"data/bossbench_{tag}/*.save.json"):
    m = json.load(open(p))
    metas[m["seed"]] = m


def load(f):
    return {(r["seed"], r.get("rep")): r for r in map(json.loads, open(f))}


def cost(r):
    return r["hp_before"] if not r["won"] else r["hp_before"] - r["hp_after"]


base = load(files[0])
for f in files:
    o = load(f)
    for room in ("Monster", "Elite", "Boss"):
        ks = [k for k in base if k in o and metas[k[0]]["room"] == room]
        d = [cost(o[k]) - cost(base[k]) for k in ks]
        m = S.mean(d) if d else 0
        se = S.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0
        print(f"{f.split('/')[-1]:28s} {room:7s} won {sum(o[k]['won'] for k in ks):3d}/{len(ks)} "
              f"cost {S.mean(cost(o[k]) for k in ks):5.1f} paired {m:+5.2f}+-{se:.2f}")
