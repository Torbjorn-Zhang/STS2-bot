"""Paired A10 trend of an xdeckrl pipeline's evaluations against fixed references.

  trend.py TAG [--refs runs/a.jsonl,runs/b.jsonl] [--ev0 runs/x.jsonl]
Prints, per evaluated iteration: act-1/act-2 clears, wins, mean floor and the paired floor
difference (with its standard error) against each reference on the common seeds.
"""
import argparse
import glob
import json
import os
import re
import statistics as S

HERE = os.path.dirname(os.path.abspath(__file__))


def load(p):
    return {json.loads(l)["seed"]: json.loads(l) for l in open(p)}


def prog(r):
    return 53 if r["victory"] else ((r["act"] or 1) - 1) * 17 + (r["floor"] or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--refs", default="")
    ap.add_argument("--ev0", default=None, help="the starting networks' evaluation when the pipeline has no ev0")
    args = ap.parse_args()
    evs = {int(re.search(r"_ev(\d+)\.jsonl$", p).group(1)): p for p in glob.glob(os.path.join(HERE, "runs", f"{args.tag}_ev*.jsonl"))
           if re.search(r"_ev(\d+)\.jsonl$", p)}
    if args.ev0 and 0 not in evs:
        evs[0] = os.path.join(HERE, args.ev0)
    refs = {os.path.basename(p)[:-6]: load(os.path.join(HERE, p)) for p in args.refs.split(",") if p}
    if 0 in evs:
        refs = {"ev0": load(evs[0]), **refs}
    for it in sorted(evs):
        R = load(evs[it])
        line = (f"ev{it:<3d} n {len(R):3d}  act1 {sum(r['cleared_acts'] >= 1 for r in R.values()):3d}  "
                f"act2 {sum(r['cleared_acts'] >= 2 for r in R.values()):3d}  wins {sum(r['victory'] for r in R.values())}  "
                f"floor {S.mean(prog(r) for r in R.values()):.2f}")
        for nm, B in refs.items():
            c = [s for s in R if s in B]
            if len(c) > 1 and B is not R:
                d = [prog(R[s]) - prog(B[s]) for s in c]
                line += f"  | vs {nm} {S.mean(d):+.2f}±{S.stdev(d) / len(d) ** .5:.2f}"
        print(line)
    log = os.path.join(HERE, "runs", "fleet", "fleet.log")
    if os.path.exists(log):   # the self-play side (mixed ascensions, sampled): wins and act clears per iteration
        pat = re.compile(rf"{re.escape(args.tag)}: iteration (\d+) expert self-play done (\{{.*?\}})")
        for l in open(log):
            m = pat.search(l)
            if m:
                d = json.loads(m.group(2))
                print(f"sp{m.group(1):<3s} runs {d['runs']}  act1 {d['act1']}  act2 {d['act2']}  wins {d['wins']}  floor {d['mean_floor']}")


if __name__ == "__main__":
    main()
