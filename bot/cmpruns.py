"""Paired comparison of full-run evaluations (same seeds): python cmpruns.py base tag [tag ...]"""
import collections
import json
import statistics as S
import sys


def load(t):
    return {json.loads(l)["seed"]: json.loads(l) for l in open(f"runs/{t}.jsonl")}


def prog(r):
    return 3 * 17 + 2 if r["victory"] else ((r["act"] or 1) - 1) * 17 + (r["floor"] or 0)


tags = sys.argv[1:]
D = {t: load(t) for t in tags}
base = D[tags[0]]
for t in tags:
    rows = list(D[t].values())
    b1 = [f for r in rows for f in r["fight_log"] if f["room"] == "Boss" and f["floor"] in (16, 17)]
    dk = [f["deck"] for f in b1 if f.get("deck")]
    s = [k for k in base if k in D[t]]
    d = [prog(D[t][k]) - prog(base[k]) for k in s]
    se = (S.stdev(d) / len(d) ** 0.5) if len(d) > 1 and t != tags[0] else 0
    print(f"{t:22s} n {len(rows)} act1 {sum(r['cleared_acts'] >= 1 for r in rows):3d} act2 {sum(r['cleared_acts'] >= 2 for r in rows)} "
          f"wins {sum(r['victory'] for r in rows)} | boss1 {sum(f['won'] for f in b1)}/{len(b1)} hp_in {S.mean(f['hp_before'] for f in b1):.1f} "
          f"up {S.mean(x['upgraded'] for x in dk):.2f} | floor {S.mean(prog(r) for r in rows):.2f} "
          f"paired {S.mean(d):+.2f}±{se:.2f} | act1 only-base {sum(base[k]['cleared_acts'] >= 1 > D[t][k]['cleared_acts'] for k in s)} "
          f"only-this {sum(D[t][k]['cleared_acts'] >= 1 > base[k]['cleared_acts'] for k in s)}")
