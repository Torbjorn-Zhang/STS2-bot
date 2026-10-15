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

REPS = 10
PILOT = "quick"   # "quick": quick_policy (fast) or "planner": TurnPlanner(NODES)
DEBUG = None   # set to a list to record how imagined fights ended
NODES = 80


def graded(won, hp, enemy_left, enemy_max):
    if won:
        return 100.0 + hp
    return 80.0 * (1 - enemy_left / max(1, enemy_max))


def _to_map(sim, st):
    for _ in range(6):
        d = st.get("decision")
        if d == "map_select":
            return st
        if d == "card_reward":
            st = sim.act("skip_card_reward")
        else:
            st = sim.act("proceed")
        if st.get("type") == "error":
            return None
    return st if st.get("decision") == "map_select" else None


def boss_fight(sim, map_state, boss_id, seed):
    """From a map state: heal to full, enter the boss encounter, fight; return the graded score."""
    sim.send({"cmd": "set_player", "hp": (map_state.get("player") or {}).get("max_hp") or 80})
    st = sim.send({"cmd": "enter_room", "type": "combat", "encounter": boss_id})
    if st.get("decision") != "combat_play":
        return None
    sim.send({"cmd": "determinize", "seed": seed})
    enemy_max = sum((e.get("max_hp") or 0) for e in alive(st))
    last = st
    if PILOT == "planner":
        holder = {}

        def watch(s):
            if s.get("enemies") is not None:
                holder["s"] = s

        end = TurnPlanner(sim, max_nodes=NODES).play_combat(st, on_state=watch, max_actions=300)
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


def evaluate_options(sim, st):
    """{arm: mean graded score}; arm is a card_index or None (skip). Restores st."""
    boss = ((st.get("context") or {}).get("boss") or {}).get("id")
    if not boss:
        return {}
    snap = sim.send({"cmd": "snapshot"})["id"]
    arms = [c["index"] for c in st.get("cards") or []] + ([None] if st.get("can_skip", True) else [])
    ctx = st.get("context") or {}
    out = {}
    try:
        for arm in arms:
            vals = []
            for rep in range(REPS):
                sim.send({"cmd": "restore", "id": snap})
                s = sim.act("skip_card_reward") if arm is None else sim.act("select_card_reward", card_index=arm)
                s = _to_map(sim, s) if s.get("type") != "error" else None
                if s is None:
                    continue
                v = boss_fight(sim, s, boss, zlib.crc32(f"{ctx.get('act')}/{ctx.get('floor')}/{rep}".encode()))
                if v is not None:
                    vals.append(v)
            if vals:
                out[arm] = sum(vals) / len(vals)
    finally:
        sim.send({"cmd": "restore", "id": snap})
        sim.send({"cmd": "drop_snapshots", "from": snap})
    return out
