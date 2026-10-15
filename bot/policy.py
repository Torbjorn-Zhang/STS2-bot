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


def card_key(c):
    """Card id without prefix; shop entries only carry a display name."""
    cid = (c.get("id") or "").replace("CARD.", "")
    return cid or re.sub(r"[^A-Z0-9]+", "_", (c.get("name") or "").upper()).strip("_")


def stat_value(c):
    """Damage / block per energy from the card's own numbers (no tier list needed)."""
    st = c.get("stats") or {}
    cost = c.get("card_cost", c.get("cost"))
    energy = cost if isinstance(cost, int) and cost > 0 else 0.7
    desc = (c.get("description") or "").lower()
    hits = 2 if "twice" in desc else 3 if "3 times" in desc or "three times" in desc else 1
    dmg = (st.get("damage") or 0) * hits
    if "all enemies" in desc:
        dmg *= 1.5
    return 0.12 * dmg / energy + 0.06 * (st.get("block") or 0) / energy


def card_value(c, deck_size):
    cid = card_key(c)
    v = RARITY.get(c.get("rarity"), 1.0) + GOOD.get(cid, 0.0) + stat_value(c)
    if c.get("type") == "Attack":
        v += 0.3
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
    cid = card_key(c)
    if cid == "BASH":
        return 10
    if cid.startswith(BASIC):
        return 0 if cid.startswith("STRIKE_") else -1
    return 3 + RARITY.get(c.get("rarity"), 1.0) + stat_value(c)


def choose_card_select(st):
    cards = st.get("cards") or []
    lo, hi = st.get("min_select") or 0, st.get("max_select") or 1
    if not cards:
        return ("skip_select", {})
    room = (st.get("context") or {}).get("room_type")
    if room == "RestSite":                       # smith
        order = sorted(cards, key=upgrade_priority, reverse=True)
    else:                                        # shop removal / event remove-transform: Strikes, then Defends
        def rank(c):
            k = card_key(c)
            return (0 if k.startswith("STRIKE_") else 1 if k.startswith("DEFEND_") else 2, RARITY.get(c.get("rarity"), 1.0))
        order = sorted(cards, key=rank)
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


def choose_shop(st, memo):
    """Remove a Strike, then buy up to two good cards and one potion if affordable."""
    ctx = st.get("context") or {}
    m = memo.setdefault(("shop", ctx.get("act"), ctx.get("floor")), {"removed": False, "cards": 0, "potions": 0})
    gold = (st.get("player") or {}).get("gold") or 0
    cost = st.get("card_removal_cost")
    if not m["removed"] and isinstance(cost, int) and gold >= cost:
        m["removed"] = True
        return ("remove_card", {})
    deck = (st.get("player") or {}).get("deck_size") or 10
    cards = [c for c in st.get("cards") or [] if c.get("is_stocked", True) and (c.get("cost") or 9999) <= gold]
    if cards and m["cards"] < 2:
        best = max(cards, key=lambda c: card_value(c, deck))
        if card_value(best, deck) >= 2.6:
            m["cards"] += 1
            return ("buy_card", {"card_index": best["index"]})
    held = len([p for p in (st.get("player") or {}).get("potions") or [] if p])
    pots = [p for p in st.get("potions") or [] if p.get("is_stocked", True) and (p.get("cost") or 9999) <= gold]
    if pots and held < 2 and m["potions"] < 1:
        m["potions"] += 1
        return ("buy_potion", {"potion_index": min(pots, key=lambda p: p.get("cost") or 0)["index"]})
    return ("leave_room", {})


def decide(sim, st, avoid=(), memo=None):
    """avoid: JSON-encoded actions that made no progress from this exact state before.
    memo: per-run scratch dict (shop visits)."""
    if st.get("decision") == "shop":
        a = choose_shop(st, memo if memo is not None else {})
        if json.dumps(a, sort_keys=True) not in avoid:
            return a
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
