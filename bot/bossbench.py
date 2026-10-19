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


def collect_bosses(sim, seed, out_dir, ascension=10, rollouts=None, acts=3):
    """Play a whole run with the current bot (policy globals as set by main: deck nets, shop
    potions) and save at every act boss's map node, i.e. after the pre-boss rest: the states
    the bot really brings to its bosses (deck value training data). Returns the metas."""
    policy.ASCENSION = ascension
    planner = TurnPlanner(sim, rollouts=rollouts)
    st = sim.send({"cmd": "start_run", "character": "Ironclad", "seed": seed, "ascension": ascension})
    dead_ends, memo, steps, saved, done = {}, {}, 0, [], set()
    while steps < 8000:
        d = st.get("decision")
        ctx = st.get("context") or {}
        room = ctx.get("room_type")
        act = ctx.get("act") or 1
        if d == "game_over" or act > acts or st.get("type") == "error":
            break
        if d == "combat_play" or (d == "card_select" and room in ("Monster", "Elite", "Boss")):
            st = planner.play_combat(st)
            continue
        if d == "map_select" and act not in done and any(c.get("type") == "Boss" for c in st.get("choices") or []):
            done.add(act)
            boss = next(c for c in st["choices"] if c.get("type") == "Boss")
            name = f"{seed}_a{act}"
            path = os.path.join(out_dir, f"{name}.save")
            if sim.send({"cmd": "write_continue_save", "path": path}).get("success"):
                p = st.get("player") or {}
                meta = {"seed": name, "path": path, "node": [boss["col"], boss["row"]], "room": "Boss", "act": act,
                        "floor": ctx.get("floor"), "hp": p.get("hp"), "max_hp": p.get("max_hp"),
                        "deck": p.get("deck_size"), "boss": (ctx.get("boss") or {}).get("id")}
                with open(path + ".json", "w") as fh:
                    json.dump(meta, fh)
                saved.append(meta)
        a = policy.decide(sim, st, avoid=dead_ends.get(_key(st), ()), memo=memo)
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
DECK_OP = None    # diagnostic (--deck-op): change the saved deck (removals, upgrades) or HP, see deck_op()

BASICS = ("STRIKE_IRONCLAD", "DEFEND_IRONCLAD")
COMMON_POTIONS = ("FIRE_POTION", "EXPLOSIVE_AMPOULE", "BLOCK_POTION", "STRENGTH_POTION", "DEXTERITY_POTION",
                  "ENERGY_POTION", "SWIFT_POTION", "WEAK_POTION", "VULNERABLE_POTION", "ATTACK_POTION",
                  "SKILL_POTION", "POWER_POTION", "FLEX_POTION", "BLOOD_POTION", "REGEN_POTION", "LIQUID_BRONZE")


def potion_slots(st):
    from policy import potion_slots as ps
    return ps(st)


def deck_op(sim, st, op, seed):
    """What is a card removal, an upgrade or a heal worth at this boss? Applies `op` to the
    loaded save: base | rmK (K basics, Strikes first) | nostrike | nobasic | upK (K random
    upgradable cards, non-basics first) | upall | hpmK (K HP less). Needs the Diag engine
    (set_player remove_cards / upgrade)."""
    import random
    import re
    if not op or op == "base":
        return
    p = st.get("player") or {}
    deck = p.get("deck") or []
    ids = [c["id"].replace("CARD.", "") for c in deck]
    rng = random.Random(f"{seed}/{op}")
    m = re.fullmatch(r"([a-z]+)(\d*)", op)
    kind, k = m.group(1), int(m.group(2) or 0)
    if kind == "rm":
        order = [i for i in ids if i == BASICS[0]] + [i for i in ids if i == BASICS[1]]
        sim.send({"cmd": "set_player", "remove_cards": order[:k]})
    elif kind == "nostrike":
        sim.send({"cmd": "set_player", "remove_cards": [i for i in ids if i == BASICS[0]]})
    elif kind == "nobasic":
        sim.send({"cmd": "set_player", "remove_cards": [i for i in ids if i in BASICS]})
    elif kind == "upall":
        sim.send({"cmd": "set_player", "upgrade": ["*"]})
    elif kind == "up":
        cand = [c for c in deck if not c.get("upgraded") and c.get("after_upgrade")]
        rng.shuffle(cand)
        cand.sort(key=lambda c: c["id"].replace("CARD.", "") in BASICS)   # stable: non-basics first
        sim.send({"cmd": "set_player", "upgrade": [c["id"].replace("CARD.", "") for c in cand[:k]]})
    elif kind == "hpm":
        sim.send({"cmd": "set_player", "hp": max(1, (p.get("hp") or 1) - k)})
    elif kind in ("pot", "potfill"):   # potK: K more random common potions; potfill: every free slot
        from policy import potion_id
        have = [potion_id(x) for x in p.get("potions") or [] if x and potion_id(x)]
        cap = potion_slots(st)
        add = max(0, cap - len(have)) if kind == "potfill" else min(k, max(0, cap - len(have)))
        sim.send({"cmd": "set_player", "potions": have + [rng.choice(COMMON_POTIONS) for _ in range(add)]})
    else:
        raise ValueError(f"unknown deck op {op}")


def fight_one(sim, meta, model, value_weight, nodes, rollouts=None, rep=None):
    st = sim.send({"cmd": "load_save", "path": meta["path"]})
    if SET_DECK:
        sim.send({"cmd": "set_player", "deck": SET_DECK})
    deck_op(sim, st, DECK_OP, meta["seed"])
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


def load_offer_pool(min_count=20):
    """act -> (card ids, weights): how often each card was offered in the expert bot's card
    rewards (data/offer_pool.json), so variant decks get cards the bot really sees."""
    with open(os.path.join(HERE, "data", "offer_pool.json")) as fh:
        pool = json.load(fh)["pool"]
    out = {}
    for a, cnt in pool.items():
        items = [(k, v) for k, v in cnt.items() if v >= min_count]
        out[int(a)] = ([k for k, _ in items], [v for _, v in items])
    return out


def vary_ops(meta, deck, variant, offset, pool, act):
    """The changes that make deck variant `variant` of a boss save, drawn once so every
    repetition fights the same variant (variant 0: the deck as saved). Removals, upgrades,
    added offer-pool cards, lower HP and potion changes, mixed: the decisions a deck value
    must rank (which card to take, remove, upgrade; heal or smith; buy potions)."""
    import random
    if variant == 0:
        return {}
    rng = random.Random(f"{meta['seed']}/v{variant}/{offset}")
    ids, wts = pool.get(act) or pool[max(pool)]
    toks = [c["id"].replace("CARD.", "") for c in deck]
    basics = [t for t in toks if t in BASICS]
    others = [t for t in toks if t not in BASICS and t not in ("ASCENDERS_BANE", "BASH")]
    rm = []
    for _ in range(rng.choice([0, 0, 0, 1, 1, 2, 3])):
        src = basics if basics and (rng.random() < 0.7 or not others) else others
        if src:
            rm.append(src.pop(rng.randrange(len(src))))
    upg = [c["id"].replace("CARD.", "") for c in deck if not c.get("upgraded") and c.get("after_upgrade")]
    up = rng.sample(upg, min(len(upg), rng.choice([0, 0, 0, 1, 1, 2, 3, 4])))
    add = [rng.choices(ids, weights=wts)[0] + ("+" if rng.random() < 0.15 else "")
           for _ in range(rng.choice([0, 0, 1, 1, 1, 2, 2, 3]))]
    ops = {"remove": rm, "upgrade": up, "add": add}
    if rng.random() < 0.4:
        ops["hp_frac"] = round(rng.uniform(0.5, 1.0), 3)
    u = rng.random()
    if u < 0.25:
        ops["potions"] = [rng.choice(COMMON_POTIONS) for _ in range(3)]   # fill (capped by the slots)
    elif u < 0.4:
        ops["potions"] = []
    return ops


def vary_one(sim, meta, variant, rep, offset, pool, nodes, rollouts):
    """One repetition of one deck variant on a boss save: apply the variant, fight the boss
    (hidden future shared by every variant of this save and repetition), record deck and result."""
    from policy import potion_id
    path = meta["path"] if os.path.exists(meta["path"]) else os.path.join(HERE, "data", os.path.basename(os.path.dirname(meta["path"])),
                                                                           os.path.basename(meta["path"]))
    st = sim.send({"cmd": "load_save", "path": path})
    p = st.get("player") or {}
    act = meta.get("act") or (st.get("context") or {}).get("act") or 1
    ops = vary_ops(meta, p.get("deck") or [], variant, offset, pool, act)
    if ops.get("remove"):
        sim.send({"cmd": "set_player", "remove_cards": ops["remove"]})
    if ops.get("upgrade"):
        sim.send({"cmd": "set_player", "upgrade": ops["upgrade"]})
    if ops.get("add"):
        sim.send({"cmd": "set_player", "add_cards": ops["add"]})
    if "hp_frac" in ops:
        sim.send({"cmd": "set_player", "hp": max(1, round((p.get("hp") or 1) * ops["hp_frac"]))})
    if "potions" in ops:
        have = [potion_id(x) for x in p.get("potions") or [] if x and potion_id(x)]
        sim.send({"cmd": "set_player", "potions": (have + ops["potions"]) if ops["potions"] else []})
    col, row = meta.get("node") or meta["boss_node"]
    st = sim.act("select_map_node", col=col, row=row)
    if st.get("decision") != "combat_play":
        return None
    sim.send({"cmd": "determinize", "seed": zlib.crc32(f"{meta['seed']}/vr{rep}/{offset}".encode())})
    pl = st.get("player") or {}
    from record import card_tok
    row_ = {"save": meta["seed"], "act": act, "boss": (st.get("context") or {}).get("boss", {}).get("id") if isinstance((st.get("context") or {}).get("boss"), dict) else meta.get("boss"),
            "enemies": [e.get("name") for e in st.get("enemies") or []], "variant": variant, "rep": rep, "offset": offset, "ops": ops,
            "deck": [card_tok(c) for c in pl.get("deck") or []], "relics": [r.get("name") for r in pl.get("relics") or []],
            "hp": pl.get("hp"), "mhp": pl.get("max_hp"), "potions": [potion_id(x) or x.get("name") for x in pl.get("potions") or [] if x],
            "gold": pl.get("gold")}
    planner = TurnPlanner(sim, max_nodes=nodes, rollouts=rollouts)
    enemy_max = sum((e.get("max_hp") or 0) for e in st.get("enemies") or [])
    last = {"s": st}

    def watch(s_):
        if s_.get("enemies") is not None:
            last["s"] = s_

    t0 = time.perf_counter()
    end = planner.play_combat(st, on_state=watch)
    won = end.get("decision") != "game_over" or bool(end.get("victory"))
    hp = (last["s"].get("player") or {}).get("hp") if won else 0
    left = 0 if won else sum((e.get("hp") or 0) for e in combat.alive(last["s"]))
    row_.update(won=won, hp_after=hp or 0, enemy_max=enemy_max, enemy_left=left, rounds=last["s"].get("round"),
                left_frac=round(min(1.0, left / max(1, enemy_max)), 4), graded=decktest.graded(won, hp or 0, min(left, enemy_max), enemy_max),
                seconds=round(time.perf_counter() - t0, 1))
    return row_


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
    ap.add_argument("mode", choices=["collect", "collect-fights", "collect-bosses", "eval", "explore", "vary"])
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
    ap.add_argument("--deck-op", default=None, help="diagnostic: base|rmK|nostrike|nobasic|upK|upall|hpmK (deck_op)")
    ap.add_argument("--engine-config", default=None, help="engine build to use (Diag: one built for diagnostics)")
    ap.add_argument("--seed-start", type=int, default=0, help="collect-bosses: first seed index")
    ap.add_argument("--shop-potions", action="store_true", help="collect-bosses: the bot buys potions (policy.SHOP_POTIONS)")
    ap.add_argument("--tar", default=None, help="collect-bosses: pack the new saves into this tar (one fetchable output)")
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

    if args.engine_config:
        import engine
        engine.ENGINE_CONFIG = args.engine_config
    policy.SHOP_POTIONS = args.shop_potions
    if args.mode == "collect-bosses":
        import tarfile
        os.makedirs(bench_dir, exist_ok=True)
        seeds = [f"bc_{args.tag}_{i}" for i in range(args.seed_start, args.seed_start + args.n)]
        roll = json.loads(args.rollouts) if args.rollouts else None
        metas = [m for ms in parallel(seeds, lambda sim, s: collect_bosses(sim, s, bench_dir, args.ascension, roll),
                                      args.workers) for m in ms]
        if args.tar:
            with tarfile.open(os.path.join(HERE, args.tar), "w") as tf:
                for m in metas:
                    for p in (m["path"], m["path"] + ".json"):
                        tf.add(p, arcname=os.path.relpath(p, HERE))
        print(json.dumps({"saved": len(metas), "by_act": {a: sum(m["act"] == a for m in metas) for a in (1, 2, 3)},
                          "of_runs": len(seeds)}))
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
    if args.engine_config:
        import engine
        engine.ENGINE_CONFIG = args.engine_config
    if args.deck_op:
        global DECK_OP
        DECK_OP = args.deck_op
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
    if args.mode == "vary":
        pool = load_offer_pool()
        items = [(m, v, r) for m in metas for v in range(args.variants) for r in range(args.reps)]
        res = [r for r in parallel(items, lambda sim, it: vary_one(sim, it[0], it[1], it[2], args.rep_offset, pool, args.nodes, roll),
                                   args.workers) if r]
        with open(args.out, "w") as fh:
            for r in res:
                fh.write(json.dumps(r) + "\n")
        print(json.dumps({"fights": len(res), "won": sum(r["won"] for r in res), "saves": len(metas)}))
        return
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
