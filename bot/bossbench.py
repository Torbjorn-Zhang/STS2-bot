"""Boss-fight benchmark: fixed pre-boss save states, replayed under different bot configs.

1) collect: play runs with the current bot and save the map state right before the act
   boss node (deck, HP, relics and potions as the bot built them):
       bossbench.py collect N --tag b0
2) eval: load every saved state, enter the boss, fight it with a planner config:
       bossbench.py eval --tag b0 [--model m.pkl] [--value-weight w] [--weights '{"dmg_big":1.0}']
Same states for every config, so the comparison is paired and far less noisy than whole runs.
"""

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import combat
import policy
from combat import TurnPlanner
from engine import fresh_sim, maybe_recycle
from run import _key

HERE = os.path.dirname(os.path.abspath(__file__))


def collect_one(sim, seed, out_dir, ascension=0, act=1):
    """Play with the default bot until the boss node is the next choice; save there."""
    planner = TurnPlanner(sim)
    st = sim.send({"cmd": "start_run", "character": "Ironclad", "seed": seed, "ascension": ascension})
    dead_ends, memo, steps = {}, {}, 0
    while steps < 5000:
        d = st.get("decision")
        ctx = st.get("context") or {}
        room = ctx.get("room_type")
        if d == "game_over" or (ctx.get("act") or 1) > act or st.get("type") == "error":
            return None
        if d == "map_select" and (ctx.get("act") or 1) == act and any(c.get("type") == "Boss" for c in st.get("choices") or []):
            boss = next(c for c in st["choices"] if c.get("type") == "Boss")
            path = os.path.join(out_dir, f"{seed}.save")
            r = sim.send({"cmd": "write_continue_save", "path": path})
            if not r.get("success"):
                return None
            meta = {"seed": seed, "path": path, "boss_node": [boss["col"], boss["row"]],
                    "hp": (st.get("player") or {}).get("hp"), "max_hp": (st.get("player") or {}).get("max_hp"),
                    "deck": (st.get("player") or {}).get("deck_size"), "boss": (ctx.get("boss") or {}).get("name")}
            with open(path + ".json", "w") as fh:
                json.dump(meta, fh)
            return meta
        if d == "combat_play" or (d == "card_select" and room in ("Monster", "Elite", "Boss")):
            st = planner.play_combat(st)
            continue
        a = policy.decide(sim, st, avoid=dead_ends.get(_key(st), ()), memo=memo)
        nxt = sim.act(a[0], **a[1])
        steps += 1
        if nxt.get("type") == "error" or _key(nxt) == _key(st):
            dead_ends.setdefault(_key(st), set()).add(json.dumps(a, sort_keys=True))
            if nxt.get("type") == "error":
                continue
        st = nxt
    return None


def fight_one(sim, meta, model, value_weight, nodes, rollouts=None):
    st = sim.send({"cmd": "load_save", "path": meta["path"]})
    col, row = meta["boss_node"]
    st = sim.act("select_map_node", col=col, row=row)
    planner = TurnPlanner(sim, max_nodes=nodes, value=model, value_weight=value_weight, rollouts=rollouts)
    hp0 = (st.get("player") or {}).get("hp")
    rounds = [0]

    def watch(s):
        if s.get("round"):
            rounds[0] = s["round"]

    t0 = time.perf_counter()
    st = planner.play_combat(st, on_state=watch)
    won = st.get("decision") != "game_over" or bool(st.get("victory"))
    return {"seed": meta["seed"], "boss": meta["boss"], "won": won, "hp_before": hp0,
            "hp_after": (st.get("player") or {}).get("hp") if won else 0, "rounds": rounds[0],
            "seconds": round(time.perf_counter() - t0, 1)}


def parallel(items, fn, workers):
    chunks = [items[i::workers] for i in range(workers)]

    def worker(chunk):
        sim, out = fresh_sim(), []
        try:
            for it in chunk:
                sim = maybe_recycle(sim)
                out.append(fn(sim, it))
        finally:
            sim.close()
        return out

    with ThreadPoolExecutor(workers) as ex:
        return [r for part in ex.map(worker, chunks) for r in part]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "eval"])
    ap.add_argument("n", type=int, nargs="?", default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=None)
    ap.add_argument("--value-weight", type=float, default=None)
    ap.add_argument("--rollouts", default=None, help='JSON rollout config, e.g. {"k":4,"m":3,"turns":2}')
    ap.add_argument("--weights", default=None, help="JSON overrides for combat.W")
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--act", type=int, default=1, help="collect: which act's boss")
    ap.add_argument("--subset", default=None, help="'even' or 'odd' seeds only (train/validation split)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    bench_dir = os.path.join(HERE, "data", f"bossbench_{args.tag}")

    if args.mode == "collect":
        os.makedirs(bench_dir, exist_ok=True)
        seeds = [f"boss_{args.tag}_{i}" for i in range(args.n)]
        metas = [m for m in parallel(seeds, lambda sim, s: collect_one(sim, s, bench_dir, args.ascension, args.act), args.workers) if m]
        print(json.dumps({"saved": len(metas), "of_runs": len(seeds), "dir": bench_dir}))
        return

    if args.weights:
        combat.W.update(json.loads(args.weights))
    model = None
    if args.model:
        from value import ValueModel
        model = ValueModel(args.model)
    metas = []
    for p in sorted(glob.glob(os.path.join(bench_dir, "*.save.json"))):
        with open(p) as fh:
            metas.append(json.load(fh))
    if args.subset:
        metas = [m for m in metas if (int(m["seed"].rsplit("_", 1)[1]) % 2 == 0) == (args.subset == "even")]
    t0 = time.perf_counter()
    res = parallel(metas, lambda sim, m: fight_one(sim, m, model, args.value_weight, args.nodes,
                                                    json.loads(args.rollouts) if args.rollouts else None), args.workers)
    if args.out:
        with open(args.out, "w") as fh:
            for r in res:
                fh.write(json.dumps(r) + "\n")
    by = {}
    for r in res:
        b = by.setdefault(r["boss"], [0, 0])
        b[0] += 1
        b[1] += r["won"]
    print(json.dumps({
        "fights": len(res), "won": sum(r["won"] for r in res),
        "win_rate": round(sum(r["won"] for r in res) / max(1, len(res)), 3),
        "mean_hp_after": round(sum(r["hp_after"] for r in res) / max(1, len(res)), 2),
        "by_boss": {k: f"{w}/{n}" for k, (n, w) in sorted(by.items())},
        "wall_s": round(time.perf_counter() - t0, 1),
    }))


if __name__ == "__main__":
    main()
