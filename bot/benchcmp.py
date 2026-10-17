"""Paired comparison of fleet bench configs: runs/<tag>_<config>.jsonl or its shard files.
usage: benchcmp.py TAG [base_config]   (wins, mean HP after, paired graded difference vs base)"""
import collections
import glob
import json
import math
import statistics as S
import sys

tag = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "c0"
res = collections.defaultdict(dict)
for f in glob.glob(f"runs/{tag}_c*.jsonl"):
    c = f[len(f"runs/{tag}_"):].split("_")[0].split(".")[0]
    for line in open(f):
        r = json.loads(line)
        res[c][(r["seed"], r.get("rep"))] = r
for c in sorted(res):
    ks = [k for k in res[c] if k in res[base]]
    if not ks:
        continue
    d = [float(res[c][k]["won"]) - float(res[base][k]["won"]) for k in ks]
    g = [res[c][k]["graded"] - res[base][k]["graded"] for k in ks]
    se = lambda x: S.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 0
    print(f"{c:4s} n {len(ks):3d} won {sum(res[c][k]['won'] for k in ks):3d} (base {sum(res[base][k]['won'] for k in ks):3d}) "
          f"dwin {S.mean(d):+.3f}+-{se(d):.3f}  dgraded {S.mean(g):+6.2f}+-{se(g):5.2f}  "
          f"s/fight {S.mean(res[c][k]['seconds'] for k in ks):6.1f}")
