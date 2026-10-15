"""Paired comparison of two bench evaluations (bossbench.py eval --out ...).

usage: benchcmp.py runs/A.jsonl runs/B.jsonl
Pairs fights by (seed, rep) and reports, overall and per room type, the mean change
in HP left after the fight (B - A, deaths count as 0) with its standard error, and
wins on each side.
"""

import json
import math
import sys
from collections import defaultdict


def load(path):
    with open(path) as fh:
        return {(r["seed"], r.get("rep")): r for r in (json.loads(l) for l in fh)}


def stats(pairs):
    d = [b["hp_after"] - a["hp_after"] for a, b in pairs]
    n = len(d)
    if n == 0:
        return {}
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / max(1, n - 1))
    return {"n": n, "d_hp": round(mean, 2), "se": round(sd / math.sqrt(n), 2),
            "A_won": sum(a["won"] for a, _ in pairs), "B_won": sum(b["won"] for _, b in pairs)}


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(a) & set(b), key=str)
    pairs = [(a[k], b[k]) for k in keys]
    by = defaultdict(list)
    for x, y in pairs:
        by[x.get("boss", "?")].append((x, y))
    print(json.dumps({"all": stats(pairs), **{k: stats(v) for k, v in sorted(by.items())}}, indent=1))


if __name__ == "__main__":
    main()
