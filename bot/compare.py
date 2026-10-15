"""Paired comparison of two bot runs on the same seeds.

usage: compare.py runs/A.jsonl runs/B.jsonl
Prints clear rates, the discordant pairs (only A / only B cleared), an exact two-sided
sign-test p-value on them, and boss results for each side.
"""

import json
import sys
from collections import defaultdict
from math import comb


def load(path):
    with open(path) as fh:
        return {r["seed"]: r for r in (json.loads(l) for l in fh)}


def boss_stats(runs):
    by = defaultdict(lambda: [0, 0])
    for r in runs.values():
        for f in r.get("fight_log") or []:
            if f["room"] == "Boss":
                k = (f["enemies"] or ["?"])[-1]
                by[k][0] += 1
                by[k][1] += f["won"]
    return {k: f"{w}/{n}" for k, (n, w) in sorted(by.items())}


def sign_test(a_only, b_only):
    n, k = a_only + b_only, min(a_only, b_only)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    seeds = sorted(set(a) & set(b))
    ca = {s for s in seeds if a[s]["cleared_acts"] >= 1}
    cb = {s for s in seeds if b[s]["cleared_acts"] >= 1}
    a_only, b_only = len(ca - cb), len(cb - ca)
    print(json.dumps({
        "seeds": len(seeds),
        "A": sys.argv[1], "A_cleared": len(ca), "A_rate": round(len(ca) / max(1, len(seeds)), 3),
        "B": sys.argv[2], "B_cleared": len(cb), "B_rate": round(len(cb) / max(1, len(seeds)), 3),
        "both": len(ca & cb), "only_A": a_only, "only_B": b_only,
        "sign_test_p": round(sign_test(a_only, b_only), 4),
        "A_boss": boss_stats({s: a[s] for s in seeds}), "B_boss": boss_stats({s: b[s] for s in seeds}),
        "A_mean_s": round(sum(a[s]["seconds"] for s in seeds) / max(1, len(seeds)), 1),
        "B_mean_s": round(sum(b[s]["seconds"] for s in seeds) / max(1, len(seeds)), 1),
    }, indent=1))


if __name__ == "__main__":
    main()
