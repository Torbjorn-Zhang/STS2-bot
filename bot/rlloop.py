"""Self-play policy improvement loop (AlphaGo's RL stage).

Each iteration: N fast self-play runs with the current policy network (sampled at a
temperature), one PPO update on them, then a paired fight bench against quick_policy.

    rlloop.py --init models/policy.pt --iters 10 --runs 300 [--card-policy models/card.pt]
Writes models/policy_rl<i>.pt, data/rl/it<i>.jsonl and runs/rlloop.jsonl (one line per iteration).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
# low CPU priority: nice on Linux/macOS; on Windows the launcher starts us at low priority
# and child processes inherit it
NICE = ["nice", "-n", "19"] if shutil.which("nice") else []


def sh(args, log):
    print("+", " ".join(args), flush=True)
    out = subprocess.run(args, cwd=HERE, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
    with open(log, "a") as fh:
        fh.write(out.stdout + out.stderr)
    if out.returncode != 0:
        raise RuntimeError(f"{args[1]} failed: {out.stderr[-800:]}")
    return out.stdout.strip().splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--card-policy", default=None)
    ap.add_argument("--bench", default="a10v2")
    ap.add_argument("--bench-reps", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "data", "rl"), exist_ok=True)
    log = os.path.join(HERE, "runs", "rlloop.log")
    cur = args.init
    for it in range(args.start, args.start + args.iters):
        data = os.path.join("data", "rl", f"it{it}.jsonl")
        if os.path.exists(os.path.join(HERE, data)):
            os.remove(os.path.join(HERE, data))
        sp = NICE + [PY, "selfplay.py", str(args.runs), "--policy", cur, "--temperature",
              str(args.temperature), "--workers", str(args.workers), "--seed-prefix", f"rl{it}", "--out", data]
        if args.card_policy:
            sp += ["--card-policy", args.card_policy]
        play = json.loads(sh(sp, log)[-1])
        new = os.path.join("models", f"policy_rl{it}.pt")
        sh(NICE + [PY, "nets.py", "rl", data, "--init", cur, "--out", new, "--temperature",
            str(args.temperature), "--epochs", str(args.epochs)], log)
        bench = json.loads(sh(NICE + [PY, "pilotbench.py", "--tag", args.bench, "--reps", str(args.bench_reps),
                               "--pilots", f"quick,net:{cur},net:{new}"], log)[-1])
        row = {"iter": it, "from": cur, "to": new, "selfplay": play, "bench": bench}
        with open(os.path.join(HERE, "runs", "rlloop.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        cur = new


if __name__ == "__main__":
    main()
