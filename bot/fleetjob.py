"""Jobs the fleet coordinator (fleet.py) runs on a training machine.

    fleetjob.py train --tag decki --it 6 --temperature 2.0 --card M --rest M --event M --map M --shop M DATA...
        one PPO update of each deck network on these self-play files (the trainers run in
        parallel; NETS_DEVICE=cuda puts the card network's update on the GPU). Prints
        {"models": {kind: path}, "timing": {...}} as the last line.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

TRAINERS = {"card": ["nets.py", "cardrl"], "rest": ["restnet.py", "rl"], "event": ["eventnet.py", "rl"],
            "map": ["mapnet.py", "rl"], "shop": ["shopnet.py", "rl"], "unified": ["upolicy.py", "rl"]}


def expand(paths):
    """'@list.txt' stands for the paths listed in that file (Windows caps a cmd line at 8191 chars)."""
    out = []
    for p in paths:
        if p.startswith("@"):
            with open(os.path.join(HERE, p[1:])) as fh:
                out += [l.strip() for l in fh if l.strip()]
        else:
            out.append(p)
    return out


def train(args):
    t0 = time.time()
    args.data = expand(args.data)
    procs, models = {}, {}
    for kind, (script, mode) in TRAINERS.items():
        init = getattr(args, kind)
        if not init:
            continue
        out = f"models/{args.tag}_{kind}{args.it}.pt"
        cmd = [PY, script, mode] + args.data + ["--init", init, "--out", out, "--temperature", str(args.temperature)]
        log = open(os.path.join(HERE, "fleet", "jobs", f"train_{args.tag}{args.it}_{kind}.log"), "w")
        procs[kind] = (subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "OMP_NUM_THREADS": "2", "NETS_THREADS": "2",
                                             "NETS_LR_SCALE": str(args.lr_scale),
                                             **({k: str(v) for k, v in json.loads(args.reward).items()} if args.reward else {})}),
                                        log, time.time())
        models[kind] = out
    timing, failed = {}, []
    for kind, (p, log, ts) in procs.items():
        rc = p.wait()
        log.close()
        timing[kind] = round(time.time() - ts, 1)
        if rc != 0:
            failed.append(kind)
    if failed:
        for kind in failed:
            with open(os.path.join(HERE, "fleet", "jobs", f"train_{args.tag}{args.it}_{kind}.log")) as fh:
                sys.stderr.write(f"--- {kind}\n" + fh.read()[-1500:])
        sys.exit(1)
    print(json.dumps({"models": models, "timing": {**timing, "total": round(time.time() - t0, 1)},
                      "device": os.environ.get("NETS_DEVICE", "cpu")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train"])
    ap.add_argument("data", nargs="+")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--it", type=int, required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr-scale", type=float, default=1.0, help="multiplies every trainer's learning rate")
    ap.add_argument("--reward", default=None, help='JSON env overrides of the RL return, e.g. {"STS2_WIN_BONUS": 50}')
    for kind in TRAINERS:
        ap.add_argument(f"--{kind}", default=None)
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "fleet", "jobs"), exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
