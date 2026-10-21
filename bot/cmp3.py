"""Act-3 stage comparison (A10 double boss) on the same act-3 starts: python cmp3.py base tag [tag ...]

Changes that only act in act 3 leave acts 1-2 identical (the sim is deterministic), so the
runs reach the same act-3 start states and their run.py --mc-k plays face the same hidden
futures: a paired comparison of act 3 alone. Every act-3 play (the real one and the K MC ones)
ends at a stage: died before the boss, lost the first final boss (floor 15), lost the second
(floor 16), or won. Shown per tag: P(reach the boss), P(beat boss 1 | reached), P(win | beat
boss 1), P(win), and paired per-seed differences with standard errors."""
import json
import statistics as S
import sys


def load(t):
    return {json.loads(l)["seed"]: json.loads(l) for l in open(f"runs/{t}.jsonl")}


def plays(r):
    """progress values of the act-3 plays of a run: the real one + its MC plays."""
    if not ((r.get("act") or 0) >= 3 or r.get("victory")):
        return []
    real = 53 if r["victory"] else ((r["act"] or 1) - 1) * 17 + (r["floor"] or 0)
    return [real] + list((r.get("mc") or {}).get("prog") or [])


def stages(ps):
    n = len(ps)
    boss = sum(p >= 49 for p in ps)
    b1 = sum(p >= 50 for p in ps)
    win = sum(p >= 53 for p in ps)
    return n, boss, b1, win


def se(x):
    return S.stdev(x) / len(x) ** 0.5 if len(x) > 1 else 0.0


tags = sys.argv[1:]
D = {t: load(t) for t in tags}
base = D[tags[0]]
seeds = [s for s in base if plays(base[s]) and all(s in D[t] and plays(D[t][s]) for t in tags)]
print(f"{len(seeds)} act-3 starts in every tag")
for t in tags:
    ps = [p for s in seeds for p in plays(D[t][s])]
    n, boss, b1, win = stages(ps)
    line = (f"{t:16s} plays {n:5d}  reach boss {boss / n:5.1%}  beat boss1|reach {b1 / max(1, boss):5.1%}  "
            f"win|boss1 {win / max(1, b1):5.1%}  win {win / n:6.2%}")
    if t != tags[0]:
        def per(seed, k, tag):
            ps_ = plays(D[tag][seed])
            st = stages(ps_)
            return st[k] / st[0]
        for k, name in ((1, "reach"), (2, "boss1"), (3, "win")):
            d = [per(s, k, t) - per(s, k, tags[0]) for s in seeds]
            line += f" | d{name} {100 * S.mean(d):+.1f}±{100 * se(d):.1f}"
    print(line)
