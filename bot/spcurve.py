"""Self-play learning curve by ascension group: python spcurve.py [PIPE ...] [--block N]

Reads the outcome rows ("k": "o") of data/xdarchive/<pipe>_it<N>/ (the xdeckrl self-play
archive) and prints, per iteration (or per block of N iterations), the win rate, the act-2
clear rate and B3 (act-3 boss HP share removed, a win = 2, averaged with the run.py --mc-k
plays) for A0-3, A4-7, A8-9 and A10. Self-play plays every ascension, so this is a free,
low-noise curve of what the policy is learning; the greedy A10 evaluation sees only A10."""
import argparse
import glob
import json
import os
import re
import subprocess
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = [("A0-3", range(0, 4)), ("A4-7", range(4, 8)), ("A8-9", range(8, 10)), ("A10", range(10, 11))]


def b3(r):
    if r.get("victory"):
        return 2.0
    if (r.get("act") or 1) < 3:
        return 0.0
    real = max(0.0, (r.get("boss_dmg") or 0) - 2.0)
    mc = (r.get("mc") or {}).get("bdmg") or []
    return (real + sum(mc)) / (1 + len(mc))


def rows(d):
    out = subprocess.run(f"cat {d}/*.jsonl | grep '\"k\":\"o\"'", shell=True, capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipes", nargs="*", default=["up2", "up3", "up4"])
    ap.add_argument("--block", type=int, default=1, help="average over blocks of N iterations")
    args = ap.parse_args()
    pat = re.compile(r"^(%s)_it(\d+)$" % "|".join(map(re.escape, args.pipes)))
    dirs = []
    for d in glob.glob(os.path.join(HERE, "data", "xdarchive", "*_it*")):
        m = pat.match(os.path.basename(d))
        if m:
            dirs.append((int(m.group(2)), m.group(1), d))
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0.0]))   # label -> group -> n, wins, act-2 clears, B3 sum
    order = []
    for it, pipe, d in sorted(dirs):
        label = f"{pipe} it{it}" if args.block == 1 else f"{pipe} it{(it - 1) // args.block * args.block + 1}-{(it - 1) // args.block * args.block + args.block}"
        if label not in acc:
            order.append(label)
        for r in rows(d):
            g = next((n for n, rg in GROUPS if r.get("asc") in rg), None)
            if g is None:
                continue
            s = acc[label][g]
            s[0] += 1
            s[1] += bool(r.get("victory"))
            s[2] += (r.get("cleared_acts") or 0) >= 2
            s[3] += b3(r)
    print(f"{'':16s}" + "".join(f"| {g:>5s}  n    win   act2    B3  " for g, _ in GROUPS))
    for label in order:
        line = f"{label:16s}"
        for g, _ in GROUPS:
            n, w, a2, b = acc[label][g]
            line += f"| {n:6d} {100 * w / max(1, n):5.1f}% {100 * a2 / max(1, n):5.1f}% {b / max(1, n):.3f} "
        print(line)


if __name__ == "__main__":
    main()
