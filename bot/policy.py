"""Out-of-combat decisions: small, explainable rules (map DP, card picks, rest, events)."""

import json
import re

from engine import legal_actions

RARITY = {"Rare": 3.0, "Uncommon": 2.0, "Common": 1.0, "Basic": 0.0}
BASIC = ("STRIKE_", "DEFEND_")
# Ironclad tier list for the v0.111 pool (0 = never take, 5 = take over almost anything).
# Single-player view: cards that only help "another player" are worthless here.
TIER = {
    # attacks
    "BASH": 1.5, "STRIKE_IRONCLAD": 0.0, "ANGER": 1.2, "ASHEN_STRIKE": 1.3, "BLUDGEON": 2.6, "BODY_SLAM": 1.0,
    "BREAK": 2.2, "BREAKTHROUGH": 2.6, "BULLY": 1.6, "CINDER": 2.0, "CONFLAGRATION": 3.2, "DISMANTLE": 2.0,
    "FEED": 3.0, "FIEND_FIRE": 2.0, "FIGHT_ME": 1.8, "HEADBUTT": 1.6, "HEMOKINESIS": 2.6, "HOWL_FROM_BEYOND": 2.6,
    "IRON_WAVE": 1.6, "MANGLE": 2.6, "MIDNIGHT": 1.4, "MOLTEN_FIST": 1.6, "OUTRAGE": 1.0, "PACTS_END": 1.0,
    "PERFECTED_STRIKE": 2.0, "PILLAGE": 1.6, "POMMEL_STRIKE": 3.0, "RAMPAGE": 1.6, "SETUP_STRIKE": 1.9,
    "SPITE": 1.5, "STOMP": 2.5, "SWORD_BOOMERANG": 2.1, "TEAR_ASUNDER": 2.0, "THRASH": 2.0, "THUNDERCLAP": 2.6,
    "TWIN_STRIKE": 2.1, "UNRELENTING": 2.0, "UPPERCUT": 3.0, "WHIRLWIND": 3.0,
    # skills
    "ARMAMENTS": 2.0, "BATTLE_TRANCE": 3.0, "BLOOD_WALL": 2.0, "BLOODLETTING": 1.6, "BRAND": 2.4,
    "BURNING_PACT": 2.4, "CASCADE": 1.4, "COLOSSUS": 2.0, "DEFEND_IRONCLAD": 0.0, "DOMINATE": 2.4,
    "DRUM_OF_BATTLE": 2.0, "EVIL_EYE": 1.6, "EXPECT_A_FIGHT": 2.0, "FLAME_BARRIER": 3.0, "FORGOTTEN_RITUAL": 1.6,
    "HAVOC": 1.0, "IMPERVIOUS": 3.2, "INFERNAL_BLADE": 1.5, "NOT_YET": 1.4, "OFFERING": 3.5, "ONE_TWO_PUNCH": 2.0,
    "PRIMAL_FORCE": 1.4, "SECOND_WIND": 2.0, "SHRUG_IT_OFF": 3.0, "STOKE": 1.4, "TAUNT": 2.2, "TREMBLE": 1.6,
    "TRUE_GRIT": 2.5, "BLAZE": 0.0, "DEMONIC_SHIELD": 0.0,
    # powers
    "AGGRESSION": 2.0, "BARRICADE": 1.8, "CORRUPTION": 2.4, "CRIMSON_MANTLE": 2.6, "CRUELTY": 1.5,
    "DARK_EMBRACE": 2.0, "DEMON_FORM": 3.2, "FEEL_NO_PAIN": 2.5, "HELLRAISER": 1.5, "INFERNO": 2.0,
    "INFLAME": 3.0, "JUGGERNAUT": 2.0, "JUGGLING": 1.4, "PYRE": 3.0, "RAGE": 2.0, "RUPTURE": 1.5,
    "STAMPEDE": 2.0, "STONE_ARMOR": 2.6, "UNMOVABLE": 1.5, "VICIOUS": 1.0, "TANK": 0.0,
}
AOE = {"THUNDERCLAP", "WHIRLWIND", "BREAKTHROUGH", "CONFLAGRATION", "STOMP", "HOWL_FROM_BEYOND", "SWORD_BOOMERANG",
       "PACTS_END"}
BLOCK = {"SHRUG_IT_OFF", "FLAME_BARRIER", "IMPERVIOUS", "TRUE_GRIT", "BLOOD_WALL", "COLOSSUS", "EXPECT_A_FIGHT",
         "IRON_WAVE", "TAUNT", "SECOND_WIND", "EVIL_EYE", "ARMAMENTS", "STONE_ARMOR", "CRIMSON_MANTLE", "FEEL_NO_PAIN"}
DRAW = {"POMMEL_STRIKE", "BATTLE_TRANCE", "OFFERING", "SHRUG_IT_OFF", "BURNING_PACT", "DRUM_OF_BATTLE", "PILLAGE",
        "DARK_EMBRACE"}
SCALING = {"INFLAME", "DEMON_FORM", "SETUP_STRIKE", "BRAND", "DOMINATE", "RUPTURE", "FIGHT_ME", "PYRE", "RAMPAGE"}
FRONT = {"BLUDGEON", "HEMOKINESIS", "POMMEL_STRIKE", "TWIN_STRIKE", "UPPERCUT", "CARNAGE", "BREAKTHROUGH",
         "THUNDERCLAP", "WHIRLWIND", "CONFLAGRATION", "FEED", "SWORD_BOOMERANG", "MANGLE", "HOWL_FROM_BEYOND"}


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


ELITE_HP = {1: 0.55, 2: 0.7, 3: 0.7}   # take elites (relics) only from this HP fraction up
ELITE_VALUE = {1: 3.5, 2: 2.0, 3: 2.0}


def node_value(kind, hp, row, act=1, gold=0):
    if kind == "Monster":
        return 1.0 if hp > 0.4 else -1.0
    if kind == "Elite":
        # the starter deck loses to elites on the first floors; later, healthy, they pay in relics
        a = min(max(act, 1), 3)
        return ELITE_VALUE[a] if hp >= ELITE_HP[a] and row >= 4 else -3.0
    if kind == "RestSite":
        return 3.0 if hp < 0.5 else 2.0 if hp < 0.7 else 0.8
    if kind == "Treasure":
        return 2.0
    if kind == "Unknown":
        return 1.0
    if kind == "Shop":
        # removals and relics: gold left unspent at the end of an act is wasted strength
        return 4.0 if gold >= 350 else 3.0 if gold >= 200 else 1.5 if gold >= 90 else 0.3
    return 0.0


# expected HP change (fraction of max HP) per node type, per act, for projecting the path
HP_COST = {"Monster": (0.07, 0.12, 0.14), "Elite": (0.25, 0.30, 0.32), "Boss": (0.45, 0.5, 0.5)}


def _hp_after(kind, hp, act):
    a = min(max(act, 1), 3) - 1
    if kind in HP_COST:
        return hp - HP_COST[kind][a]
    if kind == "RestSite" and hp < 0.7:
        return min(1.0, hp + 0.3)
    return hp


def choose_map(sim, st):
    """Best path by DP over the map, with HP projected along it (an elite deep in the
    path is judged at the HP the bot will likely have there, not the HP it has now)."""
    m = sim.send({"cmd": "get_map"})
    nodes = {(n["col"], n["row"]): n for row in m.get("rows", []) for n in row}
    act = (st.get("context") or {}).get("act") or 1
    gold = (st.get("player") or {}).get("gold") or 0
    memo = {}

    def value(c, hp):
        key = (c, round(hp, 2))
        if key not in memo:
            n = nodes.get(c)
            if n is None:
                memo[key] = 0.0
            else:
                v = node_value(n["type"], hp, n["row"], act, gold)
                hp2 = _hp_after(n["type"], hp, act)
                if hp2 <= 0.05:
                    v -= 20.0
                memo[key] = v + max([value((k["col"], k["row"]), hp2) for k in n.get("children") or []] or [0.0])
        return memo[key]

    hp = hp_ratio(st)
    best = max(st["choices"], key=lambda ch: (value((ch["col"], ch["row"]), hp), -ch["col"]))
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


def base_key(c):
    return card_key(c).rstrip("+").split("+")[0]


def card_value(c, deck_size, deck=None, act=1):
    """Tier value of a card for this deck. deck: list of the deck's cards (for needs)."""
    cid = base_key(c)
    if c.get("type") in ("Curse", "Status"):
        return -10.0
    v = TIER.get(cid)
    if v is None:  # colorless / unknown: rarity plus raw numbers
        v = 0.6 * RARITY.get(c.get("rarity"), 1.0) + stat_value(c)
    if c.get("upgraded") or "+" in card_key(c):
        v += 0.3
    if deck is not None:
        keys = [base_key(x) for x in deck]
        n = lambda group: sum(1 for k in keys if k in group)
        if cid in AOE:
            v += 0.8 if n(AOE) == 0 else 0.3 if n(AOE) == 1 else -0.3
        if cid in BLOCK and n(BLOCK) < 2 + act:
            v += 0.4
        if cid in DRAW and n(DRAW) < 3:
            v += 0.3
        if cid in SCALING:
            v += 0.4 if n(SCALING) < 2 else -0.3 * (n(SCALING) - 1)
        if cid in FRONT and act == 1 and n(FRONT) < 3:
            v += 0.4
        powers_n = sum(1 for x in deck if x.get("type") == "Power")
        if c.get("type") == "Power" and powers_n >= 3:
            v -= 0.4 * (powers_n - 2)
        dup = keys.count(cid)
        if dup:
            v -= 0.5 * dup if cid not in ("POMMEL_STRIKE", "SHRUG_IT_OFF", "OFFERING", "FLAME_BARRIER") else 0.25 * dup
    return v - 0.04 * max(0, deck_size - 18)


SKIP_BELOW = 1.7   # a weaker card only dilutes the deck


DECK_TEST = True        # judge card rewards by imagined boss fights (decktest.py)
TIER_WEIGHT = 4.0       # graded-score points per tier point, on top of the boss test


def choose_card_reward(st, sim=None):
    if DECK_TEST and sim is not None and st.get("cards"):
        import decktest
        scores = decktest.evaluate_options(sim, st)
        if scores:
            full = (st.get("player") or {}).get("deck")
            deck_n = (st.get("player") or {}).get("deck_size") or 10
            act = (st.get("context") or {}).get("act") or 1
            by_index = {c["index"]: c for c in st["cards"]}

            def total(arm):
                prior = SKIP_BELOW if arm is None else card_value(by_index[arm], deck_n, full, act)
                return scores[arm] + TIER_WEIGHT * prior

            best = max(scores, key=total)
            return ("skip_card_reward", {}) if best is None else ("select_card_reward", {"card_index": best})
    return choose_card_reward_tier(st)


def choose_card_reward_tier(st):
    deck = (st.get("player") or {}).get("deck_size") or 10
    cards = st.get("cards") or []
    if not cards:
        return ("skip_card_reward", {})
    full = (st.get("player") or {}).get("deck")
    act = (st.get("context") or {}).get("act") or 1
    best = max(cards, key=lambda c: card_value(c, deck, full, act))
    if card_value(best, deck, full, act) < SKIP_BELOW and st.get("can_skip", True):
        return ("skip_card_reward", {})
    return ("select_card_reward", {"card_index": best["index"]})


def upgrade_priority(c):
    cid = card_key(c)
    if cid == "BASH":
        return 10
    if cid.startswith(BASIC):
        return 0 if cid.startswith("STRIKE_") else -1
    return 3 + TIER.get(base_key(c), RARITY.get(c.get("rarity"), 1.0)) + 0.5 * stat_value(c)


def choose_card_select(st, sim=None):
    cards = st.get("cards") or []
    lo, hi = st.get("min_select") or 0, st.get("max_select") or 1
    if not cards:
        return ("skip_select", {})
    room = (st.get("context") or {}).get("room_type")
    if room == "RestSite":                       # smith
        order = sorted(cards, key=upgrade_priority, reverse=True)
        if DECK_TEST and sim is not None and hi == 1 and len(order) > 1:
            import decktest
            top = [c.get("index", i) for i, c in enumerate(order[:4])]
            scores = decktest.evaluate_upgrades(sim, st, top)
            if scores:
                prio = {c.get("index", i): upgrade_priority(c) for i, c in enumerate(cards)}
                best = max(scores, key=lambda k: scores[k] + 1.0 * prio.get(k, 0))
                return ("select_cards", {"indices": str(best)})
    else:                                        # shop removal / event remove-transform: Strikes, then Defends
        def rank(c):
            k = card_key(c)
            first = (-1 if c.get("type") in ("Curse", "Status") else 0 if k.startswith("STRIKE_")
                     else 1 if k.startswith("DEFEND_") else 2)
            return (first, TIER.get(base_key(c), RARITY.get(c.get("rarity"), 1.0)))
        order = sorted(cards, key=rank)
    n = max(1, lo) if hi >= 1 else lo
    return ("select_cards", {"indices": ",".join(str(c.get("index", i)) for i, c in enumerate(order[:n]))})


def boss_is_next(sim, st):
    """True when the boss is the next row of the map (floors differ by act, rows don't)."""
    if sim is None:
        return ((st.get("context") or {}).get("floor") or 0) >= 15
    m = sim.send({"cmd": "get_map"})
    boss, cur = m.get("boss") or {}, m.get("current_coord") or {}
    return boss.get("row") is not None and cur.get("row") is not None and cur["row"] >= boss["row"] - 1


HEAL_BELOW = 0.5   # away from the boss, smith unless HP is below this


def choose_rest(st, sim=None):
    opts = [o for o in st.get("options") or [] if o.get("is_enabled", True)]
    # arrive at the boss healthy: the last rest site before it heals unless nearly full
    want = "HEAL" if hp_ratio(st) < (0.9 if boss_is_next(sim, st) else HEAL_BELOW) else "SMITH"
    pick = next((o for o in opts if o.get("option_id") == want), None) or (opts[0] if opts else None)
    return ("choose_option", {"option_index": pick["index"]}) if pick else ("leave_room", {})


EVENT_TEXT_WEIGHT = 3.0   # graded-score points per keyword-score point in event choices
# Neow's opening options, by title (the boss test says little on floor 1, where every
# option still loses to the boss). Values are in keyword-score units; unknown titles
# fall back to the keyword score.
NEOW = {
    "Precarious Shears": 3.0,   # remove 2 cards for a little HP
    "Hefty Tablet": 2.5,        # choose a Rare card (plus an Injury)
    "Booming Conch": 2.0,       # extra draw and energy in elite fights
    "Neow's Torment": 1.8,      # a strong card
    "Lead Paperweight": 1.5,    # choose a Colorless card
    "Golden Pearl": 1.5,        # gold: removals and relics in shops
    "Neow's Bones": 1.2,        # random relics, but curses
    "Neow's Talisman": 0.8,     # upgrades a Strike and a Defend
    "Winged Boots": 0.3,        # path freedom the bot does not use
}


def choose_event(st, sim=None):
    opts = [o for o in st.get("options") or [] if not o.get("is_locked")]
    if not opts:
        return ("leave_room", {})
    hp = hp_ratio(st)
    neow = [o for o in opts if o.get("title") in NEOW]
    if len(neow) == len(opts):
        best = max(opts, key=lambda o: NEOW[o["title"]])
        return ("choose_option", {"option_index": best["index"]})
    if DECK_TEST and sim is not None and len(opts) > 1:
        import decktest
        scores = decktest.evaluate_event(sim, st)
        if len(scores) > 1:
            by = {o["index"]: o for o in opts}
            best = max(scores, key=lambda k: scores[k] + EVENT_TEXT_WEIGHT * text_score(by[k], hp))
            return ("choose_option", {"option_index": best})
    best = max(opts, key=lambda o: (text_score(o, hp), -o["index"]))
    return ("choose_option", {"option_index": best["index"]})


def choose_shop(st, memo, sim=None):
    """Remove a Strike, then buy up to two good cards and one potion if affordable."""
    ctx = st.get("context") or {}
    m = memo.setdefault(("shop", ctx.get("act"), ctx.get("floor")), {"removed": False, "cards": 0, "potions": 0})
    gold = (st.get("player") or {}).get("gold") or 0
    cost = st.get("card_removal_cost")
    if not m["removed"] and isinstance(cost, int) and gold >= cost:
        m["removed"] = True
        return ("remove_card", {})
    relics = [r for r in st.get("relics") or [] if r.get("is_stocked", True) and (r.get("cost") or 9999) <= gold]
    if relics and m.get("relics", 0) < 2:
        best = max(relics, key=lambda r: r.get("cost") or 0)   # price follows rarity
        veto = False
        if DECK_TEST and sim is not None:
            import decktest
            sc = decktest.evaluate_arms(sim, st, {"buy": [("buy_relic", {"relic_index": best["index"]})], "none": []},
                                        reps=6, tag="relic/")
            veto = bool(sc) and sc.get("buy", 0) < sc.get("none", 0) - 5.0
        m["relics"] = m.get("relics", 0) + 1
        if not veto:
            return ("buy_relic", {"relic_index": best["index"]})
    deck = (st.get("player") or {}).get("deck_size") or 10
    full = (st.get("player") or {}).get("deck")
    act = ctx.get("act") or 1
    cards = [c for c in st.get("cards") or [] if c.get("is_stocked", True) and (c.get("cost") or 9999) <= gold]
    if cards and m["cards"] < 2 and DECK_TEST and sim is not None:
        import decktest
        top = sorted(cards, key=lambda c: card_value(c, deck, full, act), reverse=True)[:3]
        scores = decktest.evaluate_buys(sim, st, [c["index"] for c in top])
        if scores:
            by = {c["index"]: c for c in top}
            # buying must beat keeping the gold by a margin (gold also buys removals, potions)
            total = {k: v + (TIER_WEIGHT * card_value(by[k], deck, full, act) - 8.0 if k is not None else TIER_WEIGHT * 2.8)
                     for k, v in scores.items()}
            best = max(total, key=total.get)
            if best is None:
                m["cards"] = 2
            else:
                m["cards"] += 1
                return ("buy_card", {"card_index": best})
    if cards and m["cards"] < 2:
        best = max(cards, key=lambda c: card_value(c, deck, full, act))
        if card_value(best, deck, full, act) >= 2.8:
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
        a = choose_shop(st, memo if memo is not None else {}, sim)
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
        return choose_card_reward(st, sim)
    if d == "card_select":
        return choose_card_select(st, sim)
    if d == "rest_site":
        return choose_rest(st, sim)
    if d == "event_choice":
        return choose_event(st, sim)
    if d == "bundle_select":
        return ("select_bundle", {"bundle_index": 0})
    if d == "shop":
        return ("leave_room", {})
    acts = legal_actions(st)
    return acts[0]
