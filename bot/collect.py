"""Self-play data for the value model.

Plays runs with the current planner (optionally a learned model) plus a little
exploration. Records every end-of-turn state and labels it with the fight's outcome:
HP when the fight ends, or DEATH_VALUE if the player died.

usage: collect.py N --tag d0 [--workers 8] [--explore 0.1] [--model path] [--acts 1]
"""

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import policy
from combat import TurnPlanner
from engine import fresh_sim, maybe_recycle
from run import _key
from value import DEATH_VALUE, ValueModel, features

HERE = os.path.dirname(os.path.abspath(__file__))


def collect_run(sim, seed, args, model):
    rng = random.Random(seed)
    pending = []
    planner = TurnPlanner(sim, max_nodes=args.nodes, value=model, explore=args.explore, rng=rng,
                          recorder=lambda s: pending.append((features(s), s.get("round"))))
    st = sim.send({"cmd": "start_run", "character": args.character, "seed": seed})
    rows, fight, steps, dead_ends, memo = [], 0, 0, {}, {}
    while steps < 5000:
        d = st.get("decision")
        ctx = st.get("context") or {}
        room = ctx.get("room_type")
        if d == "game_over" or (ctx.get("act") or 1) > args.acts or st.get("type") == "error":
            break
        if d == "combat_play" or (d == "card_select" and room in ("Monster", "Elite", "Boss")):
            pending.clear()
            st = planner.play_combat(st)
            died = st.get("decision") == "game_over" and not st.get("victory")
            y = DEATH_VALUE if died else float((st.get("player") or {}).get("hp") or 0)
            for x, rnd in pending:
                rows.append({"x": x, "y": y, "died": died, "seed": seed, "fight": fight, "room": room, "round": rnd})
            fight += 1
            continue
        a = policy.decide(sim, st, avoid=dead_ends.get(_key(st), ()), memo=memo)
        nxt = sim.act(a[0], **a[1])
        steps += 1
        if nxt.get("type") == "error" or _key(nxt) == _key(st):
            dead_ends.setdefault(_key(st), set()).add(json.dumps(a, sort_keys=True))
            if nxt.get("type") == "error":
                continue
        st = nxt
    return rows, {"seed": seed, "cleared": ((st.get("context") or {}).get("act") or 1) > args.acts,
                  "died": st.get("decision") == "game_over" and not st.get("victory")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--explore", type=float, default=0.1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--acts", type=int, default=1)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--seed-prefix", default="data")
    args = ap.parse_args()

    model = ValueModel(args.model) if args.model else None
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    out = os.path.join(HERE, "data", f"{args.tag}.jsonl")
    done = set()
    if os.path.exists(out):  # resume: skip runs already written
        with open(out) as f:
            done = {json.loads(l)["seed"] for l in f}
    seeds = [s for s in (f"{args.seed_prefix}_{args.tag}_{i}" for i in range(args.n)) if s not in done]
    lock = threading.Lock()
    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    t0 = time.perf_counter()

    def worker(chunk):
        sim = fresh_sim()
        rows, summ = [], []
        try:
            for s in chunk:
                sim = maybe_recycle(sim)
                r, info = collect_run(sim, s, args, model)
                with lock:  # append as we go: an interrupted collection keeps what it has
                    with open(out, "a") as f:
                        for row in r:
                            f.write(json.dumps(row) + "\n")
                rows += r
                summ.append(info)
                print(json.dumps({**info, "samples": len(r)}), flush=True)
        finally:
            sim.close()
        return rows, summ

    with ThreadPoolExecutor(args.workers) as ex:
        parts = list(ex.map(worker, chunks))
    rows = [r for p, _ in parts for r in p]
    summ = [s for _, ss in parts for s in ss]
    print(json.dumps({"runs": len(summ), "cleared": sum(s["cleared"] for s in summ), "samples": len(rows),
                      "boss_samples": sum(1 for r in rows if r["room"] == "Boss"),
                      "wall_s": round(time.perf_counter() - t0, 1)}))


if __name__ == "__main__":
    main()
