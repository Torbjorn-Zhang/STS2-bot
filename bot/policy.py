"""Out-of-combat decisions: small, explainable rules (map DP, card picks, rest, events)."""

import json
import re

from engine import legal_actions

RARITY = {"Rare": 3.0, "Uncommon": 2.0, "Common": 1.0, "Basic": 0.0}
BASIC = ("STRIKE_", "DEFEND_")
# Ironclad cards that are clearly good early (STS2 v0.111 names); a nudge on top of rarity.
GOOD = {
    "SHRUG_IT_OFF": 1.5, "INFLAME": 1.5, "BATTLE_TRANCE": 1.0, "UPPERCUT": 1.0, "THUNDERCLAP": 1.0,
    "IRON_WAVE": 0.8, "POMMEL_STRIKE": 1.0, "TWIN_STRIKE": 0.8, "HEMOKINESIS": 0.8, "BLOODLETTING": 0.5,
    "CARNAGE": 1.0, "BLUDGEON": 1.0, "OFFERING": 1.0, "DEMON_FORM": 1.0, "FEED": 1.0, "TRUE_GRIT": 0.8,
    "SETUP_STRIKE": 0.5, "STOMP": 0.5, "DISMANTLE": 0.5,
}

# Keyword scoring for Neow blessings and event options (title + description).
GOOD_WORDS = [(r"max hp", 2.0), (r"\brelic", 1.5), (r"remove", 1.5), (r"upgrade", 1.0), (r"heal", 0.8),
              (r"gain .*gold", 1.0), (r"potion", 0.5), (r"transform", 0.5), (r"card", 0.3)]
BAD_WORDS = [(r"curse", -2.5), (r"lose all gold", -1.5), (r"lose .*max hp", -2.0), (r"lose .*hp", -1.2),
             (r"take .*damage", -1.0), (r"fight", -0.5)]


def hp_ratio(st):
    p = st.get("player") or {}
    return (p.get("hp") or 0) / max(1, p.get("max_hp") or 1)


def text_score(opt, hp):
    t = f"{opt.get('title', '')} {opt.get('description', '')}".lower()
    s = sum(v for pat, v in GOOD_WORDS if re.search(pat, t))
    bad = sum(v for pat, v in BAD_WORDS if re.search(pat, t))
    return s + bad * (2.0 if hp < 0.5 else 1.0)


def node_value(kind, hp, row):
    if kind == "Monster":
        return 1.0 if hp > 0.4 else -1.0
    if kind == "Elite":
        return -3.0  # act-1 elites kill the starter deck far more often than they pay off
    if kind == "RestSite":
        return 3.0 if hp < 0.5 else 2.0 if hp < 0.7 else 0.8
    if kind == "Treasure":
        return 2.0
    if kind == "Unknown":
        return 1.0
    if kind == "Shop":
        return 0.3
    return 0.0


def choose_map(sim, st):
    m = sim.send({"cmd": "get_map"})
    nodes = {(n["col"], n["row"]): n for row in m.get("rows", []) for n in row}
    hp, memo = hp_ratio(st), {}

    def value(c):
        if c not in memo:
            n = nodes.get(c)
            memo[c] = 0.0 if n is None else node_value(n["type"], hp, n["row"]) + max(
                [value((k["col"], k["row"])) for k in n.get("children") or []] or [0.0])
        return memo[c]

    best = max(st["choices"], key=lambda ch: (value((ch["col"], ch["row"])), -ch["col"]))
    return ("select_map_node", {"col": best["col"], "row": best["row"]})


def card_value(c, deck_size):
    cid = (c.get("id") or "").replace("CARD.", "")
    v = RARITY.get(c.get("rarity"), 1.0) + GOOD.get(cid, 0.0)
    if c.get("type") == "Attack":
        v += 0.5
    elif c.get("type") == "Power":
        v += 0.3
    if c.get("type") in ("Curse", "Status"):
        v -= 10
    return v - 0.05 * max(0, deck_size - 15)


def choose_card_reward(st):
    deck = (st.get("player") or {}).get("deck_size") or 10
    cards = st.get("cards") or []
    if not cards:
        return ("skip_card_reward", {})
    best = max(cards, key=lambda c: card_value(c, deck))
    if card_value(best, deck) < 1.0 and st.get("can_skip", True):
        return ("skip_card_reward", {})
    return ("select_card_reward", {"card_index": best["index"]})


def upgrade_priority(c):
    cid = (c.get("id") or "").replace("CARD.", "")
    if cid == "BASH":
        return 10
    if cid.startswith(BASIC):
        return 0 if cid.startswith("STRIKE_") else -1
    return 3 + RARITY.get(c.get("rarity"), 1.0)


def choose_card_select(st):
    cards = st.get("cards") or []
    lo, hi = st.get("min_select") or 0, st.get("max_select") or 1
    if not cards:
        return ("skip_select", {})
    room = (st.get("context") or {}).get("room_type")
    if room == "RestSite":                       # smith
        order = sorted(cards, key=upgrade_priority, reverse=True)
    else:                                        # shop removal / event remove-transform: basics first
        order = sorted(cards, key=lambda c: (0 if (c.get("id") or "").replace("CARD.", "").startswith(BASIC) else 1,
                                             RARITY.get(c.get("rarity"), 1.0)))
    n = max(1, lo) if hi >= 1 else lo
    return ("select_cards", {"indices": ",".join(str(c.get("index", i)) for i, c in enumerate(order[:n]))})


def choose_rest(st):
    opts = [o for o in st.get("options") or [] if o.get("is_enabled", True)]
    floor = (st.get("context") or {}).get("floor") or 0
    # arrive at the boss healthy: the last rest sites before it heal unless nearly full
    want = "HEAL" if hp_ratio(st) < (0.9 if floor >= 15 else 0.65) else "SMITH"
    pick = next((o for o in opts if o.get("option_id") == want), None) or (opts[0] if opts else None)
    return ("choose_option", {"option_index": pick["index"]}) if pick else ("leave_room", {})


def choose_event(st):
    opts = [o for o in st.get("options") or [] if not o.get("is_locked")]
    if not opts:
        return ("leave_room", {})
    hp = hp_ratio(st)
    best = max(opts, key=lambda o: (text_score(o, hp), -o["index"]))
    return ("choose_option", {"option_index": best["index"]})


def decide(sim, st, avoid=()):
    """avoid: JSON-encoded actions that made no progress from this exact state before."""
    a = _decide(sim, st)
    if json.dumps(a, sort_keys=True) not in avoid:
        return a
    alts = [x for x in legal_actions(st) if json.dumps(x, sort_keys=True) not in avoid]
    return alts[-1] if alts else ("leave_room", {})


def _decide(sim, st):
    d = st.get("decision")
    if d == "map_select":
        return choose_map(sim, st)
    if d == "card_reward":
        return choose_card_reward(st)
    if d == "card_select":
        return choose_card_select(st)
    if d == "rest_site":
        return choose_rest(st)
    if d == "event_choice":
        return choose_event(st)
    if d == "bundle_select":
        return ("select_bundle", {"bundle_index": 0})
    if d == "shop":
        return ("leave_room", {})
    acts = legal_actions(st)
    return acts[0]
