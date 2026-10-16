"""Deck-building RL loop: card reward choices improved by self-play outcomes.

Each iteration: N fast self-play runs (combat by a fixed policy network, played greedily;
card rewards sampled from the card network), one PPO update of the card network on the
runs' final progress, then a greedy evaluation on fixed seeds (paired across iterations).

    cardloop.py --combat models/policy_rl8.pt --init models/card_sl0.pt --iters 10 --runs 2000
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def sh(args, log):
    print("+", " ".join(args), flush=True)
    out = subprocess.run(args, cwd=HERE, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
    with open(log, "a") as fh:
        fh.write(out.stdout + out.stderr)
    if out.returncode != 0:
        raise RuntimeError(f"{args[1]} failed: {out.stderr[-800:]}")
    return out.stdout.strip().splitlines()


def evaluate(combat, card, runs, workers, log, rest=None, tag="card"):
    out = os.path.join("data", "rl", f"{tag}_eval.jsonl")
    if os.path.exists(os.path.join(HERE, out)):
        os.remove(os.path.join(HERE, out))
    res = json.loads(sh(["nice", "-n", "19", PY, "selfplay.py", str(runs), "--policy", combat, "--temperature", "0",
                         "--card-policy", card, "--workers", str(workers), "--seed-prefix", "cardeval", "--out", out]
                        + (["--rest-policy", rest] if rest else []), log)[-1])
    os.remove(os.path.join(HERE, out))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combat", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--eval-runs", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rest-init", default=None, help="restnet.py model: rest sites learned too")
    ap.add_argument("--tag", default="cardloop")
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "data", "rl"), exist_ok=True)
    log = os.path.join(HERE, "runs", f"{args.tag}.log")
    cur, rest = args.init, args.rest_init
    if args.start == 1:
        row = {"iter": 0, "card": cur, "rest": rest, "eval": evaluate(args.combat, cur, args.eval_runs, args.workers, log, rest, args.tag)}
        with open(os.path.join(HERE, "runs", f"{args.tag}.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    for it in range(args.start, args.start + args.iters):
        data = os.path.join("data", "rl", f"{args.tag}_it{it}.jsonl")
        if os.path.exists(os.path.join(HERE, data)):
            os.remove(os.path.join(HERE, data))
        play = json.loads(sh(["nice", "-n", "19", PY, "selfplay.py", str(args.runs), "--policy", args.combat,
                              "--temperature", "0", "--card-policy", cur, "--card-temperature", str(args.temperature),
                              "--workers", str(args.workers), "--seed-prefix", f"{args.tag}{it}", "--out", data]
                             + (["--rest-policy", rest, "--rest-temperature", str(args.temperature)] if rest else []),
                             log)[-1])
        new = os.path.join("models", f"{args.tag}_card{it}.pt")
        sh(["nice", "-n", "19", PY, "nets.py", "cardrl", data, "--init", cur, "--out", new, "--temperature",
            str(args.temperature)], log)
        new_rest = None
        if rest:
            new_rest = os.path.join("models", f"{args.tag}_rest{it}.pt")
            sh(["nice", "-n", "19", PY, "restnet.py", "rl", data, "--init", rest, "--out", new_rest, "--temperature",
                str(args.temperature)], log)
        ev = evaluate(args.combat, new, args.eval_runs, args.workers, log, new_rest or rest, args.tag)
        row = {"iter": it, "card": new, "rest": new_rest, "selfplay": play, "eval": ev}
        with open(os.path.join(HERE, "runs", f"{args.tag}.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        os.remove(os.path.join(HERE, data))   # self-play data is large; the outcome is in the row
        cur, rest = new, new_rest or rest


if __name__ == "__main__":
    main()
