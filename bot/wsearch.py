"""Coordinate search over the combat evaluation weights (combat.W) on fixed benches.

usage: wsearch.py [--params dmg,dmg_big,...] [--out runs/wsearch.jsonl]
Every candidate is evaluated on the same fights with the same seeds (bossbench eval),
so differences are paired. A change is kept only if it beats the incumbent's mean
graded score by --margin. Each evaluation is appended to --out, so a stopped search
can be read back.
"""

import argparse
import json
import os
import subprocess
import sys

import combat

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
BENCHES = [("act2a0", 1), ("a10", 2)]   # (tag, reps)
ROLLOUTS = '{"k":6,"m":4,"turns":3}'


def evaluate(weights, label):
    vals = []
    for tag, reps in BENCHES:
        out = os.path.join(HERE, "runs", f"ws_{label}_{tag}.jsonl")
        cmd = ["nice", "-n", "19", PY, os.path.join(HERE, "bossbench.py"), "eval", "--tag", tag, "--workers", "6",
               "--rollouts", ROLLOUTS, "--weights", json.dumps(weights), "--out", out]
        if reps > 1:
            cmd += ["--reps", str(reps)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, cwd=HERE)
        with open(out) as fh:
            vals += [json.loads(l)["graded"] for l in fh]
    return sum(vals) / len(vals), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="dmg,dmg_big,kill,power_scale,potion,vuln,scaling,player_debuff")
    ap.add_argument("--factors", default="0.6,1.6")
    ap.add_argument("--margin", type=float, default=0.6)
    ap.add_argument("--sweeps", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "wsearch.jsonl"))
    args = ap.parse_args()
    best = {k: combat.W[k] for k in args.params.split(",")}
    log = open(args.out, "a")
    base, n = evaluate(best, "base")
    log.write(json.dumps({"label": "base", "weights": best, "score": base, "n": n}) + "\n"); log.flush()
    print(json.dumps({"base": round(base, 2), "n": n}), flush=True)
    for sweep in range(args.sweeps):
        for k in list(best):
            trials = []
            for f in map(float, args.factors.split(",")):
                w = dict(best, **{k: round(best[k] * f, 3)})
                sc, _ = evaluate(w, f"{k}x{f}")
                trials.append((sc, w))
                log.write(json.dumps({"label": f"{k}x{f}", "weights": w, "score": sc}) + "\n"); log.flush()
                print(json.dumps({"param": k, "factor": f, "score": round(sc, 2), "incumbent": round(base, 2)}), flush=True)
            sc, w = max(trials, key=lambda t: t[0])
            if sc > base + args.margin:
                best, base = w, sc
                print(json.dumps({"accepted": k, "value": w[k], "score": round(sc, 2)}), flush=True)
    print(json.dumps({"best": best, "score": round(base, 2)}), flush=True)


if __name__ == "__main__":
    main()
