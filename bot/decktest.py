"""Deck test: judge a card-reward option by fighting the act boss with it.

At a card reward, each option (and skipping) is played forward in a sandbox: take the
card, go to the map, heal to full, enter the act boss's encounter (known from the
map), determinize the fight and play it with a fast planner; then restore. The boss is
public information, and determinize keeps the imagined fights from seeing the real
future. The graded score rewards winning and HP left, and for a loss, how much of the
boss's HP the deck removed (early decks lose to every boss; this still ranks them).
"""

import json
import os
import zlib

from combat import TurnPlanner, alive, quick_policy

NET_PILOT = None   # optional nets.NetPolicy for the imagined fights instead of quick_policy


def _pilot(st):
    return NET_PILOT(st) if NET_PILOT is not None else quick_policy(st)

REPS = 16
UPGRADE_REPS = 5   # the engine's smith step sleeps 200 ms, so fewer seeds here
PILOT = "quick"   # "quick": quick_policy (fast) or "planner": TurnPlanner(NODES)
DEBUG = None   # set to a list to record how imagined fights ended
NODES = 80
REFINE_NODES = 40   # planner pilot for the refinement stage


def graded(won, hp, enemy_left, enemy_max):
    if won:
        return 100.0 + hp
    return 80.0 * (1 - enemy_left / max(1, enemy_max))


def _to_map(sim, st):
    """Finish the current room with simple, sim-free choices until the map shows."""
    import policy
    for _ in range(40):
        d = st.get("decision")
        if d == "map_select":
            return st
        if d is None or st.get("type") == "error" or d == "game_over":
            return None
        if d == "card_reward":
            a = policy.choose_card_reward_tier(st)
        elif d == "card_select":
            a = _pilot(st) if (st.get("context") or {}).get("room_type") in ("Monster", "Elite", "Boss") \
                else policy.choose_card_select(st)
        elif d == "combat_play":
            a = _pilot(st)
        elif d == "event_choice":
            a = policy.choose_event(st)
        elif d == "bundle_select":
            a = ("select_bundle", {"bundle_index": 0})
        elif d in ("rest_site", "shop"):
            a = ("leave_room", {})
        else:
            a = ("proceed", {})
        nxt = sim.act(a[0], **a[1])
        if nxt.get("type") == "error":
            if d == "combat_play":
                nxt = sim.act("end_turn")
            elif d in ("event_choice", "rest_site", "shop", "card_reward"):
                nxt = sim.act("leave_room") if d != "card_reward" else sim.act("skip_card_reward")
            if nxt.get("type") == "error":
                return None
        st = nxt
    return st if st.get("decision") == "map_select" else None


PATH_MARGIN = 0.5   # path test: points per HP lost in a won boss fight (HP left over is not banked)


def graded_win(won, hp_start, hp_end, enemy_left, enemy_max):
    """Path-test boss score: winning, then a small margin for HP not lost. Unlike graded(),
    HP carried into the fight earns nothing by itself, only through more wins (the next
    act starts with a heal anyway)."""
    if won:
        return 100.0 - PATH_MARGIN * max(0, hp_start - hp_end)
    return 80.0 * (1 - enemy_left / max(1, enemy_max))


def boss_fight(sim, map_state, boss_id, seed, pilot=None, hp=None):
    """From a map state: heal to full (or set `hp`), enter the boss encounter, fight; return the graded score
    (with `hp` given: graded_win, the path test's score)."""
    hp_start = hp
    sim.send({"cmd": "set_player", "hp": hp if hp is not None else (map_state.get("player") or {}).get("max_hp") or 80})
    st = sim.send({"cmd": "enter_room", "type": "combat", "encounter": boss_id})
    if st.get("decision") != "combat_play":
        return None
    sim.send({"cmd": "determinize", "seed": seed})
    enemy_max = sum((e.get("max_hp") or 0) for e in alive(st))
    last = st
    if (pilot or PILOT) == "planner":
        holder = {}

        def watch(s):
            if s.get("enemies") is not None:
                holder["s"] = s

        end = TurnPlanner(sim, max_nodes=REFINE_NODES if pilot else NODES).play_combat(st, on_state=watch, max_actions=300)
        last = holder.get("s", st)
    else:
        end = st
        for _ in range(400):
            if end.get("decision") not in ("combat_play", "card_select") or end.get("type") == "error":
                break
            last = end
            a = _pilot(end)
            nxt = sim.act(a[0], **a[1])
            if nxt.get("type") == "error":
                nxt = sim.act("end_turn") if end.get("decision") == "combat_play" else nxt
            end = nxt
    if end.get("type") == "error" or end.get("decision") in ("combat_play", "card_select"):
        return None   # stuck or broken: no evidence either way
    won = end.get("decision") != "game_over" or bool(end.get("victory"))
    # HP from the last combat state: winning a boss moves to the next act, which heals
    hp = (last.get("player") or {}).get("hp") or 0
    left = 0 if won else sum((e.get("hp") or 0) for e in alive(last))
    if DEBUG is not None:
        DEBUG.append((won, hp, left, last.get("round")))
    if hp_start is not None:
        return graded_win(won, hp_start, hp, left, enemy_max)
    return graded(won, hp, left, enemy_max)


# Hallway fights that punish weak decks, by act (game knowledge of each act's pool,
# not the run's actual upcoming fights). One is added to the boss, rotating by floor.
HARD_HALLWAY = {
    1: ["CUBEX_CONSTRUCT_NORMAL", "SLITHERING_STRANGLER_NORMAL", "VINE_SHAMBLER_NORMAL"],
    2: ["HUNTER_KILLER_NORMAL", "SLUMBERING_BEETLE_NORMAL", "SPINY_TOAD_NORMAL", "MYTES_NORMAL", "THE_OBSCURA_NORMAL"],
    3: ["FABRICATOR_NORMAL", "FROG_KNIGHT_NORMAL", "SLIMED_BERSERKER_NORMAL", "GLOBE_HEAD_NORMAL"],
}
HALLWAY_FROM_FLOOR = 8   # in act 1, before this floor only the boss matters (boss-only all act: neutral, v24)


# act-1 variants have their own hallway pools (the run's act 1 is Overgrowth or Underdocks)
HARD_HALLWAY_ACT1 = {
    "Overgrowth": HARD_HALLWAY[1],
    "Underdocks": ["GREMLIN_MERC_NORMAL", "SEWER_CLAM_NORMAL", "TWO_TAILED_RATS_NORMAL", "FOSSIL_STALKER_NORMAL"],
}


def tests_for(st):
    ctx = st.get("context") or {}
    boss = (ctx.get("boss") or {}).get("id")
    act, floor = ctx.get("act") or 1, ctx.get("floor") or 0
    tests = [boss] if boss else []
    nxt = act + 1 if act == 1 and floor >= HALLWAY_FROM_FLOOR else act
    if nxt > 1 or floor >= HALLWAY_FROM_FLOOR:
        pool = HARD_HALLWAY.get(min(nxt, 3), []) if nxt > 1 else \
            HARD_HALLWAY_ACT1.get(ctx.get("act_name"), HARD_HALLWAY[1])
        if pool:
            tests.append(pool[floor % len(pool)])
    return tests


def _apply(sim, s, moves):
    """Apply an arm's moves; a move is (action, kwargs) or a function state -> (action, kwargs)."""
    for m in moves:
        a, kw = m(s) if callable(m) else m
        if a == "noop":
            continue
        if a is None:
            break
        s = sim.act(a, **kw)
        if s.get("type") == "error":
            return s
    return s


# ---------------------------------------------------------------- path test
# Every out-of-combat choice trades deck strength against HP, max HP, potions, relics and
# gold, and which matters more depends on the deck: an upgrade can be worth more than a
# heal when the deck cannot win without it, or nothing when the deck wins anyway. The
# path test plays the rest of the act forward from each alternative: walk the bot's
# planned route to the boss, lose the HP this deck loses in each hallway / elite fight
# (simulated against the act's pools, public knowledge, scaled to the planner's play),
# heal at rest sites like the bot would, then fight the boss at the HP that is left.

PATH_TEST = os.environ.get("STS2_PATH_TEST", "0") == "1"
PATH_REPS = 12          # boss fights per alternative (same seeds for every alternative)
COST_REPS = 1           # simulated fights per pool encounter for the HP cost
UNKNOWN_FIGHT = 0.35    # share of "?" rooms that turn out to be fights
NEAR_FIGHTS = 2         # fights fought with this deck's own simulated losses; later ones (the
                        # deck will have changed) draw from the bot's real losses at that floor


def _load_losses():
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fight_losses_a10.json")) as fh:
            return json.load(fh)["losses"]
    except (OSError, ValueError, KeyError):
        return {}


EMPIRICAL = _load_losses()


def empirical_losses(act, kind, row):
    """The A10 bot's real HP-fraction losses for this act, room type and floor bucket."""
    for a in (act, act - 1, 1):
        for key in (f"{a}/{kind}/{min(4, max(0, row) // 3)}", f"{a}/{kind}/all"):
            v = EMPIRICAL.get(key)
            if v and len(v) >= 20:
                return v
    return None
GOLD_POINTS = 0.02      # graded points per gold (not simulated: shops later in the act)


def end_of_combat_heal(st):
    """HP healed after each fight by relics such as Burning Blood."""
    heal = 0
    for r in (st.get("player") or {}).get("relics") or []:
        d = (r.get("description") or "").lower()
        if "end of combat" in d and "heal" in d:
            heal += (r.get("vars") or {}).get("Heal") or 0
    return heal


def route_to_boss(sim, st, start=None, m=None):
    """Node types of the bot's planned path from the current map position (or through
    the node `start` = (col, row)) up to the boss (exclusive), using the map DP's choices."""
    import policy
    m = m or sim.send({"cmd": "get_map"})
    nodes = {(n["col"], n["row"]): n for row in m.get("rows", []) for n in row}
    cur = m.get("current_coord") or {}
    here = nodes.get((cur.get("col"), cur.get("row")))
    if start is not None:
        kids = [{"col": start[0], "row": start[1]}]
    elif here is None:   # not on the map yet (e.g. act start): every first-row node is a start
        kids = [{"col": n["col"], "row": n["row"]} for n in (m.get("rows") or [[]])[0]]
    else:
        kids = here.get("children") or []
    act = (st.get("context") or {}).get("act") or 1
    gold = (st.get("player") or {}).get("gold") or 0
    memo = {}

    def value(c, hp):
        key = (c, round(hp, 2))
        if key not in memo:
            n = nodes.get(c)
            if n is None:
                memo[key] = (0.0, [])
            else:
                v = policy.node_value(n["type"], hp, n["row"], act, gold)
                hp2 = policy._hp_after(n["type"], hp, act)
                if hp2 <= 0.05:
                    v -= 20.0
                best = max([value((k["col"], k["row"]), hp2) for k in n.get("children") or []] or [(0.0, [])],
                           key=lambda t: t[0])
                memo[key] = (v + best[0], [(n["type"], n["row"])] + best[1])
        return memo[key]

    hp = policy.hp_ratio(st)
    opts = [value((k["col"], k["row"]), hp) for k in kids]
    if not opts:
        return []
    route = max(opts, key=lambda t: t[0])[1]
    return [t for t in route if t[0] != "Boss"]


# HP lost by the planner per fight relative to the pilot, on the same decks and pools at
# full HP (A10 boss-bench decks, act-1 pools: planner 0.115 hallway / 0.29 elite of max HP,
# quick_policy 0.151 / 0.434, policy network rl11 0.144 / 0.333)
PATH_HANDICAP = {"quick": {"Monster": 0.76, "Elite": 0.67}, "net": {"Monster": 0.8, "Elite": 0.87}}


def path_value(sim, s_map, route, seed_tag, pilot=None, reps=PATH_REPS, losses=None):
    """From a map state: walk `route` with sampled fight losses (this deck's simulated
    losses in the act's pools, scaled to the planner), heal at rest sites, then fight the
    boss at the HP left. Each rep samples its own losses, so low HP risks dying on the way
    (scored 0). Returns (mean graded score, mean HP at the boss)."""
    import random
    ctx = s_map.get("context") or {}
    boss = (ctx.get("boss") or {}).get("id")
    if not boss:
        return None, None
    p = s_map.get("player") or {}
    hp0, mhp = p.get("hp") or 0, max(1, p.get("max_hp") or 1)
    kinds = [k for k, _ in route]
    near = set()
    for k in kinds:
        if k in ("Monster", "Elite", "Unknown") and len(near) < NEAR_FIGHTS:
            near.add("Monster" if k == "Unknown" else k)
    if losses is None:
        losses = _costs(sim, s_map, near, seed_tag) if near else {}
    act = ctx.get("act") or 1
    heal = end_of_combat_heal(s_map)
    snap = sim.send({"cmd": "snapshot"})["id"]
    vals, hps = [], []
    try:
        for rep in range(reps):
            rng = random.Random(f"route/{seed_tag}/{rep}")
            hp, rests_left, dead, fought = hp0, kinds.count("RestSite"), False, 0
            for kind, row in route:
                fight = kind in ("Monster", "Elite") or (kind == "Unknown" and rng.random() < UNKNOWN_FIGHT)
                if fight:
                    k2 = "Elite" if kind == "Elite" else "Monster"
                    pool = losses.get(k2) if fought < NEAR_FIGHTS else None
                    pool = pool or empirical_losses(act, k2, row) or losses.get(k2) or [0.0]
                    fought += 1
                    hp -= rng.choice(pool) * mhp
                    if hp <= 0:
                        dead = True
                        break
                    hp = min(mhp, hp + heal)
                elif kind == "RestSite":
                    rests_left -= 1
                    if hp < (0.9 if rests_left == 0 else 0.6) * mhp:
                        hp = min(mhp, hp + int(0.3 * mhp))
            if dead:
                vals.append(0.0)
                hps.append(0)
                continue
            hp = max(1, int(round(hp)))
            sim.send({"cmd": "restore", "id": snap})
            v = boss_fight(sim, s_map, boss, zlib.crc32(f"path/{seed_tag}/{rep}".encode()), pilot, hp=hp)
            if v is not None:
                vals.append(v)
                hps.append(hp)
    finally:
        sim.send({"cmd": "restore", "id": snap})
        sim.send({"cmd": "drop_snapshots", "from": snap})
    if not vals:
        return None, None
    return sum(vals) / len(vals), round(sum(hps) / len(hps), 1)


def _costs(sim, s_map, kinds, seed_tag):
    """Sampled HP fractions lost per hallway / elite fight for this deck at full HP (the
    act's pools), scaled to the planner's play. {kind: [loss, ...]}"""
    name = (s_map.get("context") or {}).get("act_name")
    pools = {"Elite": ELITES.get(name), "Monster": HALLWAYS.get(name)}
    hc = PATH_HANDICAP["net" if NET_PILOT is not None else "quick"]
    p = s_map.get("player") or {}
    full = dict(s_map, player=dict(p, hp=p.get("max_hp")))
    snap = sim.send({"cmd": "snapshot"})["id"]
    out = {}
    try:
        for kind in kinds:
            vals = []
            for enc in pools.get(kind) or []:
                for rep in range(COST_REPS):
                    sim.send({"cmd": "restore", "id": snap})
                    sim.send({"cmd": "set_player", "hp": p.get("max_hp")})
                    v = fight_cost(sim, full, enc, zlib.crc32(f"pcost/{seed_tag}/{enc}/{rep}".encode()))
                    if v is not None:
                        vals.append(hc[kind] * v)
            if vals:
                out[kind] = vals
    finally:
        sim.send({"cmd": "restore", "id": snap})
        sim.send({"cmd": "drop_snapshots", "from": snap})
    return out


# Rewards of map nodes the path test does not simulate (graded points): an elite's relic,
# card and gold (A10 data: each elite before the act-1 boss ~ +0.37 logit of winning it at
# equal HP, ~6-8 points), a chest's relic, a hallway's card and gold, an event's average.
NODE_POINTS = {"Elite": 7.0, "Treasure": 4.0, "Monster": 1.0, "Unknown": 1.5, "RestSite": 0.0}
MAP_PATH = os.environ.get("STS2_MAP_PATH", "0") == "1"


def shop_points(gold):
    return 4.0 if gold >= 200 else 2.0 if gold >= 100 else 0.5


def evaluate_map(sim, st):
    """Map choice by the path test: each next node with the DP's best continuation from it,
    walked to the boss with this deck's sampled fight losses, plus NODE_POINTS for what
    the walk does not simulate. Returns {(col, row): score}; restores st."""
    global LAST_PATH
    choices = st.get("choices") or []
    m = sim.send({"cmd": "get_map"})
    nodes = {(n["col"], n["row"]): n for row in m.get("rows", []) for n in row}
    routes = {}
    for ch in choices:
        c = (ch["col"], ch["row"])
        n = nodes.get(c)
        if n is None:
            continue
        if n["type"] == "Boss":
            return {}
        routes[c] = route_to_boss(sim, st, start=c, m=m)
    if len({tuple(k for k, _ in r) for r in routes.values()}) < 2:
        return {}   # every choice leads the same way: nothing to decide
    ctx = st.get("context") or {}
    seed_tag = f"map/{ctx.get('act')}/{ctx.get('floor')}"
    losses = _costs(sim, st, {"Monster", "Elite"}, seed_tag)
    gold = (st.get("player") or {}).get("gold") or 0
    out, info = {}, {}
    for c, route in routes.items():
        v, hp = path_value(sim, st, route, seed_tag, losses=losses)
        if v is None:
            continue
        pts = sum(shop_points(gold) if k == "Shop" else NODE_POINTS.get(k, 0.0) for k, _ in route)
        out[c] = v + pts
        info[str(c)] = (round(v, 2), round(pts, 1), hp, [k for k, _ in route])
    LAST_PATH = {"map": info}
    return out


LAST_PATH = None   # debug / records: {key: (score, projected HP)} of the last path test


def evaluate_arms_path(sim, st, arms, tag="", pilot=None, reps=PATH_REPS):
    """Like evaluate_arms, but each alternative is judged by the path test (HP carried to
    the boss). Gold differences add GOLD_POINTS per gold. Returns {key: score}; restores st."""
    global LAST_PATH
    if len(arms) < 2:
        return {}
    ctx = st.get("context") or {}
    snap = sim.send({"cmd": "snapshot"})["id"]
    route = None
    out, info = {}, {}
    seed_tag = f"{tag}{ctx.get('act')}/{ctx.get('floor')}"
    gold0 = (st.get("player") or {}).get("gold") or 0
    try:
        for key, moves in arms.items():
            sim.send({"cmd": "restore", "id": snap})
            s = _apply(sim, st, moves)
            s = _to_map(sim, s) if s.get("type") != "error" else None
            if s is None:
                continue
            if route is None:   # one route for every alternative (same draws, less noise)
                route = route_to_boss(sim, s)
            v, hp = path_value(sim, s, route, seed_tag, pilot, reps)
            if v is None:
                continue
            gold = (s.get("player") or {}).get("gold") or 0
            out[key] = v + GOLD_POINTS * (gold - gold0)
            info[key] = (round(out[key], 2), hp)
    finally:
        sim.send({"cmd": "restore", "id": snap})
        sim.send({"cmd": "drop_snapshots", "from": snap})
    LAST_PATH = {"route": [k for k, _ in route or []], "arms": {str(k): v for k, v in info.items()}}
    return out


def evaluate_arms(sim, st, arms, reps=REPS, tag="", pilot=None):
    """arms: {key: [(action, kwargs), ...]}, the moves that make each alternative from st.
    Each arm is applied in the sandbox, then the bot walks to the map and fights the test
    encounters (same seeds for every arm). Returns {key: mean graded score}; restores st."""
    if PATH_TEST:
        return evaluate_arms_path(sim, st, arms, tag=tag, pilot=pilot)
    tests = tests_for(st)
    if not tests or len(arms) < 2:
        return {}
    snap = sim.send({"cmd": "snapshot"})["id"]
    ctx = st.get("context") or {}
    out = {}
    try:
        for key, moves in arms.items():
            vals = []
            for rep in range(reps):
                sim.send({"cmd": "restore", "id": snap})
                s = _apply(sim, st, moves)
                s = _to_map(sim, s) if s.get("type") != "error" else None
                if s is None:
                    continue
                enc = tests[rep % len(tests)]
                v = boss_fight(sim, s, enc, zlib.crc32(f"{tag}{ctx.get('act')}/{ctx.get('floor')}/{rep}".encode()), pilot)
                if v is not None:
                    vals.append(v)
            if vals:
                out[key] = sum(vals) / len(vals)
    finally:
        sim.send({"cmd": "restore", "id": snap})
        sim.send({"cmd": "drop_snapshots", "from": snap})
    return out


def evaluate_options(sim, st):
    """Card reward: {card_index or None (skip): mean graded score}."""
    arms = {c["index"]: [("select_card_reward", {"card_index": c["index"]})] for c in st.get("cards") or []}
    if st.get("can_skip", True):
        arms[None] = [("skip_card_reward", {})]
    return evaluate_arms(sim, st, arms)


def evaluate_upgrades(sim, st, candidates):
    """Smith prompt: {card index: mean graded score} for upgrading each candidate."""
    arms = {idx: [("select_cards", {"indices": str(idx)})] for idx in candidates}
    return evaluate_arms(sim, st, arms, reps=UPGRADE_REPS, tag="up/")


def evaluate_buys(sim, st, card_indices):
    """Shop: {card index or None (buy nothing): mean graded score}."""
    arms = {i: [("buy_card", {"card_index": i})] for i in card_indices}
    arms[None] = []
    return evaluate_arms(sim, st, arms, tag="shop/")


def evaluate_event(sim, st):
    """Event / Neow / Ancient options: {option index: mean graded score}."""
    opts = [o for o in st.get("options") or [] if not o.get("is_locked")]
    arms = {o["index"]: [("choose_option", {"option_index": o["index"]})] for o in opts}
    return evaluate_arms(sim, st, arms, tag="event/")


# Each act's elite and hallway pools (game knowledge, not the run's actual map).
ELITES = {
    "Overgrowth": ["BYGONE_EFFIGY_ELITE", "BYRDONIS_ELITE", "PHROG_PARASITE_ELITE"],
    "Underdocks": ["PHANTASMAL_GARDENERS_ELITE", "SKULKING_COLONY_ELITE", "TERROR_EEL_ELITE"],
    "Hive": ["DECIMILLIPEDE_ELITE", "ENTOMANCER_ELITE", "INFESTED_PRISMS_ELITE"],
    "Glory": ["KNIGHTS_ELITE", "MECHA_KNIGHT_ELITE", "SOUL_NEXUS_ELITE"],
}
HALLWAYS = {
    "Overgrowth": ["CUBEX_CONSTRUCT_NORMAL", "VINE_SHAMBLER_NORMAL", "SLIMES_NORMAL", "INKLETS_NORMAL"],
    "Underdocks": ["CORPSE_SLUGS_NORMAL", "GREMLIN_MERC_NORMAL", "SEWER_CLAM_NORMAL", "TWO_TAILED_RATS_NORMAL"],
    "Hive": HARD_HALLWAY[2][:4],
    "Glory": HARD_HALLWAY[3],
}
# The planner loses far less HP than quick_policy: at A10, real act-1 costs were 0.05
# (hallway) and 0.25 (elite) of max HP where quick_policy loses about 0.29 and 0.75.
PILOT_HANDICAP = {"Monster": 0.2, "Elite": 0.35}


def fight_cost(sim, map_state, encounter, seed):
    """HP fraction lost fighting `encounter` from the current HP (1.0 = the full current HP, i.e. death)."""
    p = map_state.get("player") or {}
    hp, max_hp = p.get("hp") or 1, p.get("max_hp") or 1
    st = sim.send({"cmd": "enter_room", "type": "combat", "encounter": encounter})
    if st.get("decision") != "combat_play":
        return None
    sim.send({"cmd": "determinize", "seed": seed})
    last, end = st, st
    for _ in range(400):
        if end.get("decision") not in ("combat_play", "card_select") or end.get("type") == "error":
            break
        last = end
        a = _pilot(end)
        nxt = sim.act(a[0], **a[1])
        end = sim.act("end_turn") if nxt.get("type") == "error" and end.get("decision") == "combat_play" else nxt
    if end.get("type") == "error" or end.get("decision") in ("combat_play", "card_select"):
        return None
    if end.get("decision") == "game_over" and not end.get("victory"):
        return hp / max_hp
    return max(0, hp - ((last.get("player") or {}).get("hp") or 0)) / max_hp


def expected_costs(sim, st, reps=2):
    """From a map state: expected HP fraction lost in this act's elite and hallway fights."""
    ctx = st.get("context") or {}
    name = ctx.get("act_name")
    pools = {"Elite": ELITES.get(name), "Monster": HALLWAYS.get(name)}
    if not any(pools.values()):
        return {}
    snap = sim.send({"cmd": "snapshot"})["id"]
    out = {}
    try:
        for kind, encs in pools.items():
            vals = []
            for i, enc in enumerate(encs or []):
                for rep in range(reps):
                    sim.send({"cmd": "restore", "id": snap})
                    v = fight_cost(sim, st, enc, zlib.crc32(f"cost/{enc}/{ctx.get('floor')}/{rep}".encode()))
                    if v is not None:
                        vals.append(v)
            if vals:
                out[kind] = PILOT_HANDICAP[kind] * sum(vals) / len(vals)
    finally:
        sim.send({"cmd": "restore", "id": snap})
        sim.send({"cmd": "drop_snapshots", "from": snap})
    return out
