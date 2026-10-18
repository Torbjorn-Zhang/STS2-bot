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
import hashlib
import json
import math
import os
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import combat
import decktest
import policy
from combat import TurnPlanner
from engine import fresh_sim, maybe_recycle
from run import _key

HERE = os.path.dirname(os.path.abspath(__file__))


def collect_fights(sim, seed, out_dir, ascension=0, act=2, rollouts=None, min_act=None):
    """Play a run with the default bot; in acts min_act..act (default: only `act`), save
    before every combat node the bot picks (hallway, elite, boss). Returns the saved metas."""
    min_act = act if min_act is None else min_act
    policy.ASCENSION = ascension
    planner = TurnPlanner(sim, rollouts=rollouts)
    st = sim.send({"cmd": "start_run", "character": "Ironclad", "seed": seed, "ascension": ascension})
    dead_ends, memo, steps, saved = {}, {}, 0, []
    while steps < 6000:
        d = st.get("decision")
        ctx = st.get("context") or {}
        room = ctx.get("room_type")
        if d == "game_over" or (ctx.get("act") or 1) > act or st.get("type") == "error":
            return saved
        if d == "combat_play" or (d == "card_select" and room in ("Monster", "Elite", "Boss")):
            st = planner.play_combat(st)
            continue
        a = policy.decide(sim, st, avoid=dead_ends.get(_key(st), ()), memo=memo)
        if d == "map_select" and min_act <= (ctx.get("act") or 1) <= act and a[0] == "select_map_node":
            node = next((c for c in st.get("choices") or [] if c["col"] == a[1]["col"] and c["row"] == a[1]["row"]), {})
            if node.get("type") in ("Monster", "Elite", "Boss"):
                name = f"{seed}_a{ctx.get('act')}f{ctx.get('floor')}"
                path = os.path.join(out_dir, f"{name}.save")
                if sim.send({"cmd": "write_continue_save", "path": path}).get("success"):
                    p = st.get("player") or {}
                    meta = {"seed": name, "path": path, "node": [node["col"], node["row"]], "room": node["type"],
                            "act": ctx.get("act"), "floor": ctx.get("floor"), "hp": p.get("hp"), "max_hp": p.get("max_hp"),
                            "deck": p.get("deck_size"), "boss": node["type"]}
                    with open(path + ".json", "w") as fh:
                        json.dump(meta, fh)
                    saved.append(meta)
        nxt = sim.act(a[0], **a[1])
        steps += 1
        if nxt.get("type") == "error" or _key(nxt) == _key(st):
            dead_ends.setdefault(_key(st), set()).add(json.dumps(a, sort_keys=True))
            if nxt.get("type") == "error":
                continue
        st = nxt
    return saved


def collect_one(sim, seed, out_dir, ascension=0, act=1, rollouts=None):
    """Play with the default bot until the boss node is the next choice; save there."""
    policy.ASCENSION = ascension
    planner = TurnPlanner(sim, rollouts=rollouts)
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


SET_DECK = None   # diagnostic (--set-deck): replace the saved deck with these card ids before the fight


def fight_one(sim, meta, model, value_weight, nodes, rollouts=None, rep=None):
    st = sim.send({"cmd": "load_save", "path": meta["path"]})
    if SET_DECK:
        sim.send({"cmd": "set_player", "deck": SET_DECK})
    col, row = meta.get("node") or meta["boss_node"]
    st = sim.act("select_map_node", col=col, row=row)
    if rep is not None:   # a different hidden future per repetition, the same for every config
        sim.send({"cmd": "determinize", "seed": zlib.crc32(f"{meta['seed']}/{rep}".encode())})
    planner = TurnPlanner(sim, max_nodes=nodes, value=model, value_weight=value_weight, rollouts=rollouts)
    hp0 = (st.get("player") or {}).get("hp")
    rounds = [0]

    def watch(s):
        if s.get("round"):
            rounds[0] = s["round"]

    t0 = time.perf_counter()
    enemy_max = sum((e.get("max_hp") or 0) for e in st.get("enemies") or [])
    last = {"s": st}

    def watch2(s):
        watch(s)
        if s.get("enemies") is not None:
            last["s"] = s

    st = planner.play_combat(st, on_state=watch2)
    won = st.get("decision") != "game_over" or bool(st.get("victory"))
    hp = (last["s"].get("player") or {}).get("hp") if won else 0   # (a won act boss heals for the next act)
    left = 0 if won else sum((e.get("hp") or 0) for e in combat.alive(last["s"]))
    return {"seed": meta["seed"], "rep": rep, "boss": meta["boss"], "won": won, "hp_before": hp0,
            "hp_after": hp or 0, "graded": decktest.graded(won, hp or 0, left, enemy_max), "rounds": rounds[0],
            "seconds": round(time.perf_counter() - t0, 1)}


def shard_of(seed, n):
    """md5, not crc32: crc32 is linear, so near-identical seed names crowd a few low-bit buckets."""
    return int(hashlib.md5(seed.encode()).hexdigest(), 16) % n


def explore_one(sim, meta, variant, offset, pool, nodes, rollouts):
    """One deck variant on a boss save: add pool cards to the saved deck (70%) or build a fresh
    deck of basics + pool cards (30%), fight the boss, return the deck and the result. Deck
    diversity the bot's own runs lack (boss-value training data)."""
    import random
    from record import card_tok
    rng = random.Random(f"{meta['seed']}/{variant}/{offset}")
    st = sim.send({"cmd": "load_save", "path": meta["path"]})
    p = st.get("player") or {}
    deck = [card_tok(c) for c in p.get("deck") or []]
    ids, wts = zip(*pool)
    if rng.random() < 0.7:
        add = [rng.choices(ids, weights=wts)[0] for _ in range(rng.randint(1, 8))]
        sim.send({"cmd": "set_player", "add_cards": add})
        deck = deck + add
    else:
        add = [rng.choices(ids, weights=wts)[0] for _ in range(rng.randint(6, 16))]
        deck = ["BASH"] + ["STRIKE_IRONCLAD"] * 3 + ["DEFEND_IRONCLAD"] * 3 + add
        sim.send({"cmd": "set_player", "deck": deck})
    ctx = st.get("context") or {}
    col, row = meta.get("node") or meta["boss_node"]
    st = sim.act("select_map_node", col=col, row=row)
    sim.send({"cmd": "determinize", "seed": zlib.crc32(f"{meta['seed']}/x{variant}/{offset}".encode())})
    planner = TurnPlanner(sim, max_nodes=nodes, rollouts=rollouts)
    hp0 = (st.get("player") or {}).get("hp")
    enemy_max = sum((e.get("max_hp") or 0) for e in st.get("enemies") or [])
    last = {"s": st}

    def watch(s_):
        if s_.get("enemies") is not None:
            last["s"] = s_

    st = planner.play_combat(st, on_state=watch)
    won = st.get("decision") != "game_over" or bool(st.get("victory"))
    hp = (last["s"].get("player") or {}).get("hp") if won else 0
    left = 0 if won else sum((e.get("hp") or 0) for e in combat.alive(last["s"]))
    return {"seed": meta["seed"], "variant": variant, "act": ctx.get("act"), "boss": (ctx.get("boss") or {}).get("id"),
            "deck": deck, "relics": [r.get("name") for r in p.get("relics") or []], "hp": hp0, "mhp": p.get("max_hp"),
            "potions": len([x for x in p.get("potions") or [] if x]), "won": won, "hp_after": hp or 0,
            "graded": decktest.graded(won, hp or 0, left, enemy_max)}


def in_shard(seed, k, n):
    return shard_of(seed, n) == k


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
    ap.add_argument("mode", choices=["collect", "collect-fights", "eval", "explore"])
    ap.add_argument("--variants", type=int, default=20, help="explore: deck variants per save")
    ap.add_argument("n", type=int, nargs="?", default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=None)
    ap.add_argument("--value-weight", type=float, default=None)
    ap.add_argument("--rollouts", default=None, help='JSON rollout config, e.g. {"k":4,"m":3,"turns":2}')
    ap.add_argument("--weights", default=None, help="JSON overrides for combat.W")
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--net-rollouts", default=None, help="policy network as the planner's rollout policy")
    ap.add_argument("--rollout-value", default=None, help="RL policy file whose combat value net ends rollouts")
    ap.add_argument("--rollout-xvalue", default=None, help="expert value net (nets.py xvalue) ends rollouts")
    ap.add_argument("--leaf-xvalue", default=None, help="expert value net scores the planner's leaves")
    ap.add_argument("--leaf-mu", type=float, default=1.0)
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--act", type=int, default=1, help="collect: which act's boss")
    ap.add_argument("--min-act", type=int, default=None, help="collect-fights: also save fights from this act on")
    ap.add_argument("--subset", default=None, help="'even' or 'odd' seeds only (train/validation split)")
    ap.add_argument("--boss", default=None, help="only fights against this boss (name)")
    ap.add_argument("--room", default=None, help="only fights in this room type (Monster, Elite, Boss)")
    ap.add_argument("--shard", default=None, help="K/N: only saves in shard K of N (split across machines)")
    ap.add_argument("--deck-nets", default=None, help='collect: JSON for policy.load_deck_nets (decks built like v33)')
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps", type=int, default=1, help=">1: fight each save this many times, determinized")
    ap.add_argument("--rep-offset", type=int, default=0, help="first repetition index (fresh hidden futures per tuning generation)")
    ap.add_argument("--set-deck", default=None, help='diagnostic: JSON list of card ids replacing every saved deck')
    args = ap.parse_args()
    bench_dir = os.path.join(HERE, "data", f"bossbench_{args.tag}")
    if args.deck_nets:
        policy.load_deck_nets(json.loads(args.deck_nets))

    if args.mode == "collect-fights":
        os.makedirs(bench_dir, exist_ok=True)
        seeds = [f"fb_{args.tag}_{i}" for i in range(args.n)]
        roll = json.loads(args.rollouts) if args.rollouts else None
        metas = [m for part in parallel(seeds, lambda sim, s: collect_fights(sim, s, bench_dir, args.ascension, args.act, roll,
                                                                             args.min_act),
                                        args.workers) for m in part]
        print(json.dumps({"saved": len(metas), "of_runs": len(seeds), "dir": bench_dir,
                          "rooms": {k: sum(1 for m in metas if m["room"] == k) for k in ("Monster", "Elite", "Boss")}}))
        return

    if args.mode == "collect":
        os.makedirs(bench_dir, exist_ok=True)
        seeds = [f"boss_{args.tag}_{i}" for i in range(args.n)]
        roll = json.loads(args.rollouts) if args.rollouts else None
        metas = [m for m in parallel(seeds, lambda sim, s: collect_one(sim, s, bench_dir, args.ascension, args.act, roll),
                                     args.workers) if m]
        print(json.dumps({"saved": len(metas), "of_runs": len(seeds), "dir": bench_dir}))
        return

    if args.weights:
        combat.W.update(json.loads(args.weights))
    if args.set_deck:
        global SET_DECK
        SET_DECK = json.loads(args.set_deck)
    if args.net_rollouts or args.rollout_value or args.rollout_xvalue or args.leaf_xvalue:
        import torch
        torch.set_num_threads(1)
        import nets
        if args.net_rollouts:
            combat.FAST_POLICY = nets.NetPolicy(args.net_rollouts)
        if args.rollout_value:
            combat.ROLLOUT_VALUE = nets.CombatValue(args.rollout_value)
        if args.rollout_xvalue:
            combat.ROLLOUT_VALUE = nets.ExpertValue(args.rollout_xvalue)
        if args.leaf_xvalue:
            combat.LEAF_VALUE = nets.ExpertValue(args.leaf_xvalue)
            combat.LEAF_MU = args.leaf_mu
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
    if args.boss:
        metas = [m for m in metas if m.get("boss") == args.boss]
    if args.room:
        metas = [m for m in metas if m.get("room", "Boss") == args.room]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        metas = [m for m in metas if in_shard(m["seed"], k, n)]
    for m in metas:   # saves copied to another machine: the path recorded at collection is elsewhere
        if not os.path.exists(m["path"]):
            m["path"] = os.path.join(bench_dir, os.path.basename(m["path"]))
    t0 = time.perf_counter()
    roll = json.loads(args.rollouts) if args.rollouts else None
    if args.mode == "explore":
        with open(os.path.join(HERE, "data", "human_card_stats.json")) as fh:
            hum = json.load(fh)["cards"]
        pool = [(k.replace("CARD.", ""), math.exp((v["elo"] - 1200) / 250.0)) for k, v in hum.items()
                if v["chars"] and max(v["chars"], key=v["chars"].get) == "IRONCLAD" and v["elo_n"] >= 100]
        items = [(m, v) for m in metas for v in range(args.variants)]
        res = parallel(items, lambda sim, it: explore_one(sim, it[0], it[1], args.rep_offset, pool, args.nodes, roll),
                       args.workers)
        with open(args.out, "w") as fh:
            for r in res:
                fh.write(json.dumps(r) + "\n")
        print(json.dumps({"fights": len(res), "won": sum(r["won"] for r in res)}))
        return
    if args.reps <= 1 and not args.rep_offset:
        items = [(m, None) for m in metas]
    else:
        items = [(m, r) for m in metas for r in range(args.rep_offset, args.rep_offset + args.reps)]
    res = parallel(items, lambda sim, it: fight_one(sim, it[0], model, args.value_weight, args.nodes, roll, it[1]),
                   args.workers)
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
