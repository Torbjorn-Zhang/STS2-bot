"""Play seeds with the search bot and report how far it gets.

usage: run.py N [--acts 1] [--workers 6] [--character Ironclad] [--nodes 400] [--tag name]
Writes one JSON line per run to bot/runs/<tag>.jsonl and prints a summary.
"""

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import policy
from combat import TurnPlanner
from engine import Sim

HERE = os.path.dirname(os.path.abspath(__file__))


def _key(st):
    return json.dumps(st, sort_keys=True)


def play_run(sim, seed, character, acts, max_nodes):
    planner = TurnPlanner(sim, max_nodes=max_nodes)
    st = sim.send({"cmd": "start_run", "character": character, "seed": seed})
    t0, steps, fights, hp_log = time.perf_counter(), 0, 0, []
    dead_ends, loops = {}, 0
    while steps < 5000:
        d = st.get("decision")
        act = (st.get("context") or {}).get("act") or 1
        if d == "game_over" or act > acts or st.get("type") == "error":
            break
        room = (st.get("context") or {}).get("room_type")
        if d == "combat_play" or (d == "card_select" and room in ("Monster", "Elite", "Boss")):
            hp0 = (st.get("player") or {}).get("hp")
            names = [e.get("name") for e in st.get("enemies") or []]
            floor0 = (st.get("context") or {}).get("floor")
            last_enemies = {}
            def watch(s):  # remember the last enemy HP we saw (for deaths)
                if s.get("enemies") is not None:
                    last_enemies["hp"] = [e.get("hp") for e in s["enemies"]]
                    last_enemies["round"] = s.get("round")
            st = planner.play_combat(st, on_state=watch)
            fights += 1
            hp_log.append({"floor": floor0, "room": room, "enemies": names, "hp_before": hp0,
                           "hp_after": (st.get("player") or {}).get("hp"), "won": st.get("decision") != "game_over",
                           "enemy_hp_left": last_enemies.get("hp"), "rounds": last_enemies.get("round")})
            continue
        a = policy.decide(sim, st, avoid=dead_ends.get(_key(st), ()))
        nxt = sim.act(a[0], **a[1])
        steps += 1
        if nxt.get("type") != "error" and _key(nxt) == _key(st):
            # no progress (e.g. an event option that starts an event fight, which the engine
            # does not support yet): remember it and pick something else next time
            dead_ends.setdefault(_key(st), set()).add(json.dumps(a, sort_keys=True))
            loops += 1
        if nxt.get("type") == "error":
            dead_ends.setdefault(_key(st), set()).add(json.dumps(a, sort_keys=True))
            continue
        st = nxt
    ctx = st.get("context") or {}
    p = st.get("player") or {}
    return {
        "seed": seed, "character": character,
        "cleared_acts": (ctx.get("act") or 1) - 1,
        "victory": bool(st.get("victory")),
        "died": st.get("decision") == "game_over" and not st.get("victory"),
        "act": ctx.get("act"), "floor": ctx.get("floor"), "room": ctx.get("room_type"),
        "hp": p.get("hp"), "max_hp": p.get("max_hp"), "deck_size": p.get("deck_size"),
        "fights": fights, "seconds": round(time.perf_counter() - t0, 1),
        "planner": dict(planner.stats), "dead_end_actions": loops, "fight_log": hp_log, "error": st.get("message") if st.get("type") == "error" else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--acts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--tag", default="act1")
    ap.add_argument("--seed-prefix", default="bot")
    args = ap.parse_args()

    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    out_path = os.path.join(HERE, "runs", f"{args.tag}.jsonl")
    seeds = [f"{args.seed_prefix}_{i}" for i in range(args.n)]
    chunks = [seeds[i::args.workers] for i in range(args.workers)]

    def worker(chunk):
        sim = Sim("Release", timeout=60)
        res = []
        try:
            for s in chunk:
                r = play_run(sim, s, args.character, args.acts, args.nodes)
                res.append(r)
                print(json.dumps({k: r[k] for k in ("seed", "cleared_acts", "died", "act", "floor", "room", "hp", "deck_size", "seconds")}), flush=True)
        finally:
            sim.close()
        return res

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as ex:
        results = [r for chunk in ex.map(worker, chunks) for r in chunk]
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    cleared = [r for r in results if r["cleared_acts"] >= args.acts]
    print(json.dumps({
        "runs": len(results), "acts_target": args.acts,
        "cleared": len(cleared), "clear_rate": round(len(cleared) / max(1, len(results)), 3),
        "median_floor_reached": statistics.median([r["floor"] or 0 for r in results]) if results else None,
        "death_rooms": dict(sorted({k: sum(1 for r in results if r["died"] and r["room"] == k) for k in {r["room"] for r in results}}.items())),
        "wall_s": round(time.perf_counter() - t0, 1),
        "mean_run_s": round(statistics.mean([r["seconds"] for r in results]), 1) if results else None,
    }, indent=1))


if __name__ == "__main__":
    main()
