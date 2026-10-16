"""Deck test: judge a card-reward option by fighting the act boss with it.

At a card reward, each option (and skipping) is played forward in a sandbox: take the
card, go to the map, heal to full, enter the act boss's encounter (known from the
map), determinize the fight and play it with a fast planner; then restore. The boss is
public information, and determinize keeps the imagined fights from seeing the real
future. The graded score rewards winning and HP left, and for a loss, how much of the
boss's HP the deck removed (early decks lose to every boss; this still ranks them).
"""

import zlib

from combat import TurnPlanner, alive, quick_policy

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
            a = quick_policy(st) if (st.get("context") or {}).get("room_type") in ("Monster", "Elite", "Boss") \
                else policy.choose_card_select(st)
        elif d == "combat_play":
            a = quick_policy(st)
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


def boss_fight(sim, map_state, boss_id, seed, pilot=None):
    """From a map state: heal to full, enter the boss encounter, fight; return the graded score."""
    sim.send({"cmd": "set_player", "hp": (map_state.get("player") or {}).get("max_hp") or 80})
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
            a = quick_policy(end)
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
    return graded(won, hp, left, enemy_max)


# Hallway fights that punish weak decks, by act (game knowledge of each act's pool,
# not the run's actual upcoming fights). One is added to the boss, rotating by floor.
HARD_HALLWAY = {
    1: ["CUBEX_CONSTRUCT_NORMAL", "SLITHERING_STRANGLER_NORMAL", "VINE_SHAMBLER_NORMAL"],
    2: ["HUNTER_KILLER_NORMAL", "SLUMBERING_BEETLE_NORMAL", "SPINY_TOAD_NORMAL", "MYTES_NORMAL", "THE_OBSCURA_NORMAL"],
    3: ["FABRICATOR_NORMAL", "FROG_KNIGHT_NORMAL", "SLIMED_BERSERKER_NORMAL", "GLOBE_HEAD_NORMAL"],
}
HALLWAY_FROM_FLOOR = 8   # in act 1, before this floor only the boss matters


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


def evaluate_arms(sim, st, arms, reps=REPS, tag="", pilot=None):
    """arms: {key: [(action, kwargs), ...]}, the moves that make each alternative from st.
    Each arm is applied in the sandbox, then the bot walks to the map and fights the test
    encounters (same seeds for every arm). Returns {key: mean graded score}; restores st."""
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
                s = st
                for a, kw in moves:
                    s = sim.act(a, **kw)
                    if s.get("type") == "error":
                        break
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
        a = quick_policy(end)
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
