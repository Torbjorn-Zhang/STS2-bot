"""Compute benchmark for a training machine: which hardware the pipeline should use.

    benchcompute.py selfplay [--splits 1x8,2x8,4x4] [--runs 480]
        deck-RL self-play throughput (the real workload) for P processes x W worker threads
    benchcompute.py infer --data data/rl/x.jsonl
        combat-policy decision cost: state encoding, forward pass on CPU and CUDA, batched
    benchcompute.py train --data data/rl/x.jsonl --init models/x_card.pt
        card PPO update (nets.py cardrl) on CPU (2 and 8 threads) and CUDA, by phase
One JSON line per measurement.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

NETS = {"card": "models/decki_card5.pt", "rest": "models/decki_rest5.pt", "event": "models/decki_event5.pt",
        "map": "models/decki_map5.pt", "shop": "models/decki_shop5.pt"}


def selfplay_cmd(n, workers, prefix, start, out, records=False):
    cmd = [PY, "selfplay.py", str(n), "--policy", "models/policy_rl105.pt", "--temperature", "0",
           "--workers", str(workers), "--seed-prefix", prefix, "--seed-start", str(start), "--out", out, "--calibrate"]
    for k, flag in (("card", "card"), ("rest", "rest"), ("event", "event"), ("map", "map"), ("shop", "shop")):
        cmd += [f"--{flag}-policy", NETS[k], f"--{flag}-temperature", "2.0"]
    if not records:
        cmd.append("--no-combat-records")
    return cmd


def bench_selfplay(args):
    tmp = tempfile.mkdtemp(prefix="benchsp")
    for split in args.splits.split(","):
        rec = split.endswith("r")
        p, w = (int(x) for x in split.rstrip("r").split("x"))
        per = args.runs // p
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        t0 = time.time()
        procs = [subprocess.Popen(selfplay_cmd(per, w, "bench", i * per, os.path.join(tmp, f"{split}_{i}.jsonl"), rec),
                                  cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                 for i in range(p)]
        outs = [json.loads(pr.communicate()[0].strip().splitlines()[-1]) for pr in procs]
        wall = time.time() - t0
        runs = sum(o["runs"] for o in outs)
        size = sum(os.path.getsize(os.path.join(tmp, f)) for f in os.listdir(tmp) if f.startswith(f"{split}_"))
        print(json.dumps({"bench": "selfplay", "split": split, "procs": p, "workers": w, "records": rec,
                          "runs": runs, "errors": sum(o["errors"] for o in outs), "wall_s": round(wall, 1),
                          "runs_per_min": round(runs / wall * 60, 1), "mb_per_1000": round(size / 1e6 / runs * 1000, 1),
                          "mean_floor": round(sum(o["mean_floor"] * o["runs"] for o in outs) / max(1, runs), 2)}), flush=True)
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))


def bench_infer(args):
    import numpy as np
    import torch
    import nets
    torch.set_num_threads(1)
    pilot = nets.PolicyPilot(os.path.join(HERE, "models", "policy_rl105.pt"))
    items, header = [], {}
    with open(args.data) as fh:
        for line in fh:
            r = json.loads(line)
            if r["k"] == "f":
                header[r["seed"]] = r["run"]
            elif r["k"] == "c" and len(r["A"]) > 1 and r["seed"] in header:
                items.append((r["s"], header[r["seed"]], r["A"]))
                if len(items) >= args.n:
                    break
    t0 = time.time()
    enc = []
    for cs, run, acts in items:
        S = nets._csr([pilot.enc.state(cs, run)], pilot.enc.state_dim)
        A = nets._csr([pilot.enc.action(cs, [a[0], a[1]]) for a in acts], pilot.enc.action_dim)
        enc.append((nets._dense(S, [0]), nets._dense(A, range(len(acts))), torch.zeros(len(acts), dtype=torch.long)))
    t_enc = (time.time() - t0) / len(items)
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    out = {"bench": "infer", "decisions": len(items), "mean_actions": round(float(np.mean([len(a) for _, _, a in items])), 1),
           "encode_ms": round(t_enc * 1e3, 3)}
    for dev in devs:
        m = pilot.model.to(dev)
        sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
        with torch.no_grad():
            for s, a, o in enc[:50]:   # warm-up
                m(s.to(dev), a.to(dev), o.to(dev)).cpu()
            sync()
            t0 = time.time()
            for s, a, o in enc:
                m(s.to(dev), a.to(dev), o.to(dev)).cpu().numpy()
            sync()
            out[f"{dev}_single_ms"] = round((time.time() - t0) / len(enc) * 1e3, 3)
            for b in (64, 256):
                batches = []
                for i in range(0, len(enc) - b + 1, b):
                    grp = enc[i:i + b]
                    xs = torch.cat([g[0] for g in grp])
                    xa = torch.cat([g[1] for g in grp])
                    own = torch.cat([torch.full((len(g[2]),), k, dtype=torch.long) for k, g in enumerate(grp)])
                    batches.append((xs, xa, own))
                t0 = time.time()
                for xs, xa, own in batches:
                    m(xs.to(dev), xa.to(dev), own.to(dev)).cpu().numpy()
                sync()
                out[f"{dev}_batch{b}_ms_per_decision"] = round((time.time() - t0) / (len(batches) * b) * 1e3, 4)
    print(json.dumps(out), flush=True)


def bench_train(args):
    import torch
    configs = [("cpu", "2"), ("cpu", "8")] + ([("cuda", "2")] if torch.cuda.is_available() else [])
    for dev, th in configs:
        env = {**os.environ, "OMP_NUM_THREADS": th, "NETS_THREADS": th, "NETS_DEVICE": dev}
        out = os.path.join(tempfile.mkdtemp(prefix="benchtr"), "card.pt")
        t0 = time.time()
        res = subprocess.run([PY, "nets.py", "cardrl", args.data, "--init", args.init, "--out", out, "--temperature", "2.0"],
                             cwd=HERE, env=env, capture_output=True, text=True)
        wall = time.time() - t0
        timing = next((json.loads(l)["timing"] for l in res.stdout.splitlines() if l.startswith('{"timing"')), None)
        print(json.dumps({"bench": "train", "device": dev, "threads": int(th), "wall_s": round(wall, 1), "timing": timing,
                          "rc": res.returncode, "err": res.stderr[-300:] if res.returncode else ""}), flush=True)


def bench_profile(args):
    """Where one self-play worker's time goes: engine (waiting on the game process), network
    forward passes, state encoding, the rest (Python bookkeeping)."""
    import cProfile
    import pstats
    import random
    import torch
    torch.set_num_threads(1)
    import selfplay
    from engine import fresh_sim
    from eventnet import EventPolicy
    from mapnet import MapPolicy
    from nets import CardPolicy, NetPolicy
    from restnet import RestPolicy
    from shopnet import ShopPolicy
    selfplay.CALIBRATE = True
    pilot = NetPolicy(os.path.join(HERE, "models", "policy_rl105.pt"))
    card, rest, event = CardPolicy(NETS["card"]), RestPolicy(NETS["rest"]), EventPolicy(NETS["event"])
    mapp, shop = MapPolicy(NETS["map"]), ShopPolicy(NETS["shop"])
    sim = fresh_sim()
    prof = cProfile.Profile()
    t0 = time.time()
    prof.enable()
    for i in range(args.n):
        s = f"bench_{i}"
        selfplay.play_fast_run(sim, s, pilot, 10, 3, selfplay.NullRecorder(), 0.0, random.Random(s), card, 2.0, rest, 2.0,
                               event, 2.0, mapp, 2.0, False, shop, 2.0, False)
    prof.disable()
    wall = time.time() - t0
    sim.close()
    st = pstats.Stats(prof)
    def cum(pred):
        return sum(v[3] for k, v in st.stats.items() if pred(k))
    wait = cum(lambda k: k[0].endswith("queue.py") and k[2] == "get")             # the game process computing
    send = cum(lambda k: k[0].endswith("simclient.py") and k[2] == "send")          # + JSON both ways
    fwd = cum(lambda k: k[2] == "_call_impl")                                       # network forward passes
    enc = cum(lambda k: k[2] in ("state", "action", "encode") and ("nets.py" in k[0] or "features.py" in k[0]))
    print(json.dumps({"bench": "profile", "runs": args.n, "wall_s": round(wall, 1), "s_per_run": round(wall / args.n, 3),
                      "engine_wait": round(wait / wall, 3), "engine_json": round((send - wait) / wall, 3),
                      "forward": round(fwd / wall, 3), "encode": round(enc / wall, 3),
                      "other_python": round(1 - send / wall - fwd / wall - enc / wall, 3)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selfplay", "infer", "train", "profile"])
    ap.add_argument("--splits", default="1x8,1x16,2x8,4x4,8x2,16x1,4x6")
    ap.add_argument("--runs", type=int, default=480)
    ap.add_argument("--data")
    ap.add_argument("--init", default=NETS["card"])
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    {"selfplay": bench_selfplay, "infer": bench_infer, "train": bench_train, "profile": bench_profile}[args.mode](args)


if __name__ == "__main__":
    main()
