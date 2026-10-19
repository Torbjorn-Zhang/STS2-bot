"""Out-of-combat decisions: small, explainable rules (map DP, card picks, rest, events)."""

import json
import os
import re
import threading

import numpy as np

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
ELITE_ASC_HP = 0.01        # extra HP fraction required per ascension level
ELITE_ASC_DISCOUNT = 0.0   # (0.15 per level made A10 worse: v17)
LATE_ELITE_ROW = 10        # elites from this map row on (the boss is row ~15) cost a share of the HP for the boss
LATE_ELITE_PENALTY = 0.0   # (2.5 in v25: no gain)


MAP_COSTS = False  # map DP uses simulated fight costs for this deck when available


def node_value(kind, hp, row, act=1, gold=0, costs=None):
    if kind == "Monster":
        return 1.0 if hp > 0.4 else -1.0
    if kind == "Elite":
        # the starter deck loses to elites on the first floors; later, healthy, they pay in relics
        a = min(max(act, 1), 3)
        if costs and "Elite" in costs:
            # worth it when the fight leaves enough HP to carry on (relic ~ ELITE_VALUE)
            after = hp - costs["Elite"]
            return ELITE_VALUE[a] if after >= 0.35 and row >= 3 else -3.0 - 10.0 * max(0.0, 0.2 - after)
        # at high ascension HP on reaching the boss decides more than relics do (A10 bench:
        # boss wins vs HP r=+0.66, vs relic count r=0.00), so elites need more HP and pay less
        value = ELITE_VALUE[a] - ELITE_ASC_DISCOUNT * ASCENSION
        if row >= LATE_ELITE_ROW:   # guides: fight elites early while HP is full, skip the late ones
            value -= LATE_ELITE_PENALTY
        return value if hp >= ELITE_HP[a] + ELITE_ASC_HP * ASCENSION and row >= 4 else -3.0
    if kind == "RestSite":
        return 3.0 if hp < 0.5 else 2.0 if hp < 0.7 else 0.8
    if kind == "Treasure":
        return 2.0
    if kind == "Unknown":
        return 1.0
    if kind == "Shop":
        # removals and relics: gold left unspent at the end of an act is wasted strength
        # a removal (~75 gold) is the cheapest deck upgrade there is
        return 4.0 if gold >= 350 else 3.0 if gold >= 200 else 1.5 if gold >= 90 else 0.3
    return 0.0


# expected HP change (fraction of max HP) per node type, per act, for projecting the path
HP_COST = {"Monster": (0.07, 0.12, 0.14), "Elite": (0.25, 0.30, 0.32), "Boss": (0.45, 0.5, 0.5)}


ASCENSION = 0   # set by the runner; tougher and deadlier enemies make every fight cost more


def _hp_after(kind, hp, act, costs=None):
    a = min(max(act, 1), 3) - 1
    if costs and kind in costs:          # simulated for this deck (decktest.expected_costs)
        return hp - costs[kind]
    if kind in HP_COST:
        return hp - HP_COST[kind][a] * (1 + 0.04 * ASCENSION)
    if kind == "RestSite" and hp < 0.7:
        return min(1.0, hp + 0.3)
    return hp


def choose_map(sim, st):
    """Best path by DP over the map, with HP projected along it (an elite deep in the
    path is judged at the HP the bot will likely have there, not the HP it has now).
    With the map path test on, the next node is chosen by walking each option's route to
    the boss with this deck's simulated fight losses (decktest.evaluate_map)."""
    if MAP_NET is not None and sim is not None and len(st.get("choices") or []) > 1:
        chs, _, z = MAP_NET.scores(sim, st)
        if z is not None:
            ch = chs[int(np.argmax(z))]
            return ("select_map_node", {"col": ch["col"], "row": ch["row"]})
    if DECK_TEST and sim is not None and len(st.get("choices") or []) > 1:
        import decktest
        if decktest.MAP_PATH:
            scores = decktest.evaluate_map(sim, st)
            if scores:
                _LAST_SCORES.v = {"map": (decktest.LAST_PATH or {}).get("map")}
                col, row = max(scores, key=lambda c: (scores[c], -c[0]))
                return ("select_map_node", {"col": col, "row": row})
    m = sim.send({"cmd": "get_map"})
    nodes = {(n["col"], n["row"]): n for row in m.get("rows", []) for n in row}
    act = (st.get("context") or {}).get("act") or 1
    gold = (st.get("player") or {}).get("gold") or 0
    costs = None
    if DECK_TEST and MAP_COSTS:
        import decktest
        costs = decktest.expected_costs(sim, st) or None
    memo = {}

    def value(c, hp):
        key = (c, round(hp, 2))
        if key not in memo:
            n = nodes.get(c)
            if n is None:
                memo[key] = 0.0
            else:
                v = node_value(n["type"], hp, n["row"], act, gold, costs)
                hp2 = _hp_after(n["type"], hp, act, costs)
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


# Ironclad archetypes (PCGamesN Slay the Spire 2 guide, same v0.11x card pool): enablers
# and payoffs. A payoff is worth more for every enabler already in the deck and vice versa;
# per-card tiers and the quick-pilot deck test do not see these combinations.
ARCHETYPES = {
    "vulnerable": ({"BASH", "UPPERCUT", "THUNDERCLAP", "TREMBLE", "TAUNT", "BREAK", "DOMINATE", "MOLTEN_FIST"},
                   {"DISMANTLE", "MOLTEN_FIST", "BULLY", "CRUELTY", "DOMINATE", "COLOSSUS", "VICIOUS"}),
    "block": (BLOCK, {"BODY_SLAM", "UNMOVABLE", "BARRICADE", "JUGGERNAUT", "CRIMSON_MANTLE", "IMPERVIOUS"}),
    "exhaust": ({"TRUE_GRIT", "BURNING_PACT", "SECOND_WIND", "FIEND_FIRE", "BRAND", "STOKE", "CINDER", "THRASH",
                 "HAVOC", "CORRUPTION", "FORGOTTEN_RITUAL"},
                {"FEEL_NO_PAIN", "DARK_EMBRACE", "ASHEN_STRIKE", "PACTS_END", "EVIL_EYE", "CORRUPTION", "MIDNIGHT",
                 "DRUM_OF_BATTLE"}),
}
SYNERGY = False   # archetype bonus (v25: 22 vs 24 act-1 clears of 60, off)


def synergy_bonus(cid, keys):
    """Tier points from archetype fit: payoffs scale with the deck's enablers, enablers
    with its payoffs (capped, so one combo does not dominate the pick)."""
    bonus = 0.0
    for enablers, payoffs in ARCHETYPES.values():
        n_en = sum(1 for k in keys if k in enablers)
        n_pay = sum(1 for k in keys if k in payoffs)
        if cid in payoffs:
            bonus += 0.35 * min(4, n_en) - (0.5 if n_en == 0 else 0.0)
        if cid in enablers:
            bonus += 0.25 * min(3, n_pay)
    return bonus


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
        if SYNERGY:
            v += synergy_bonus(cid, keys)
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
LAB_WEIGHT = 0.5        # weight of the card lab's average effect (card_prior_a10.json)
REFINE_REPS = 0         # card rewards: planner-piloted fights for the top two options (0 = off; 3 was 4-5x slower, no gain)
CARD_NET = None         # optional nets.CardPolicy: card rewards without the deck test (fast self-play)
CARD_PRIOR = None       # optional nets.CardPolicy whose log-probabilities are added to the deck test's scores
CARD_PRIOR_WEIGHT = 5.0 # graded points per unit of log-probability
REST_NET = None         # optional restnet.RestPolicy: rest-site choice by the network (greedy)
EVENT_NET = None        # optional eventnet.EventPolicy: event choice by the network (greedy)
MAP_NET = None          # optional mapnet.MapPolicy: next map node by the network (greedy)
EVENT_NET_NEOW = False  # the event network also picks Neow's opening option (it learns them in self-play)
CARD_NET_DECIDES = False  # card rewards by the card network alone (CARD_PRIOR), no deck test
DECKTEST_FROM_ACT = 0   # with CARD_NET_DECIDES: from this act on, the deck test decides again (net as prior)
SHOP_NET = None         # optional shopnet.ShopPolicy: shop purchases by the network (greedy)
STRAT = None            # optional stratsearch.StratSearch: rest sites by rollouts of the rest of the act
RUN_VALUE = None        # optional nets.RunValue: the run value network (trained on self-play outcomes)
VALUE_WEIGHT = 3.0      # graded-score points per floor of predicted further progress


def value_bonus(st, arms):
    """Run value network: predicted further floors for each card reward option (None =
    skip), relative to skipping. AlphaGo mixes its value network with rollouts; here the
    deck test is the rollout."""
    if RUN_VALUE is None:
        return {}
    from record import card_tok, run_state
    rs = run_state(st)
    by_index = {c["index"]: c for c in st.get("cards") or []}
    decks = [rs["deck"] if a is None else rs["deck"] + [card_tok(by_index[a])] for a in arms]
    rem, _ = RUN_VALUE.predict([rs] * len(arms), decks)
    ref = rem[list(arms).index(None)] if None in arms else float(np.mean(rem))
    return {a: float(r - ref) for a, r in zip(arms, rem)}


def _load_prior(name):
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name)) as fh:
            return json.load(fh)["values"]
    except (OSError, ValueError, KeyError):
        return {}


LAB = _load_prior("card_prior_a10.json")              # act-1 boss fights
LAB_HALL2 = _load_prior("card_prior_act2_hall.json")  # act-2 hallway fights


def lab_value(cid, act, floor):
    """Card lab prior: the boss value, averaged with the act-2 hallway value once the
    deck is heading into act 2 or beyond (mirrors decktest.tests_for)."""
    boss = LAB.get(cid, 0.0)
    if act >= 2 or floor >= 8:
        return 0.5 * (boss + LAB_HALL2.get(cid, 0.0))
    return boss


def _potion_ids():
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "sts2-cli", "localization_eng", "potions.json")) as fh:
            return {v: k[:-len(".title")] for k, v in json.load(fh).items() if k.endswith(".title")}
    except (OSError, ValueError):
        return {}


POTION_IDS = _potion_ids()   # display name -> POTION id (the state shows potions by name only)


def potion_id(p):
    return POTION_IDS.get(p.get("name") or "")


def potion_slots(st):
    """Potion slots: 3, one fewer from ascension 4, +2 each with Alchemical Coffer / Potion Belt
    (a guess at their {PotionSlots}; an over-count only makes a purchase fail). The state lists filled slots
    only, so free slots = this - len(player.potions)."""
    p = st.get("player") or {}
    asc = (st.get("context") or {}).get("ascension") or 0
    cap = 3 - (1 if asc >= 4 else 0)
    names = {(r.get("name") if isinstance(r, dict) else r) for r in p.get("relics") or []}
    cap += 2 * len(names & {"Alchemical Coffer", "Potion Belt"})
    return cap


def free_potion_slots(st):
    return max(0, potion_slots(st) - len([x for x in (st.get("player") or {}).get("potions") or [] if x]))


SHOP_POTIONS = False   # with the shop net: gold it leaves buys potions for free slots
# boss bench, filling the free slots: act-1 bosses +11+-4% wins, act-2 +6+-3% (the bot had never
# bought one: see free_potion_slots)
POTION_PREF = {"FAIRY_IN_A_BOTTLE": 3.0, "GHOST_IN_A_JAR": 2.5, "FIRE_POTION": 2.0, "EXPLOSIVE_AMPOULE": 2.0,
               "STRENGTH_POTION": 2.0, "BLOCK_POTION": 1.7, "ENERGY_POTION": 1.6, "SWIFT_POTION": 1.6,
               "LIQUID_BRONZE": 1.5, "REGEN_POTION": 1.5, "BLOOD_POTION": 1.5, "VULNERABLE_POTION": 1.5,
               "WEAK_POTION": 1.5, "DEXTERITY_POTION": 1.4, "HEART_OF_IRON": 1.4, "FLEX_POTION": 1.2}


def shop_potion(st):
    """The potion to buy for a free slot (preferred kind, then the cheaper), or None."""
    if free_potion_slots(st) <= 0:
        return None
    gold = (st.get("player") or {}).get("gold") or 0
    pots = [x for x in st.get("potions") or [] if x.get("is_stocked", True) and (x.get("cost") or 9999) <= gold]
    if not pots:
        return None
    best = max(pots, key=lambda x: (POTION_PREF.get(potion_id(x), 1.0), -(x.get("cost") or 0)))
    return ("buy_potion", {"potion_index": best["index"]})


def choose_card_reward(st, sim=None):
    if CARD_NET is not None and not DECK_TEST and st.get("cards"):
        return CARD_NET(st)
    if CARD_NET_DECIDES and CARD_PRIOR is not None and st.get("cards"):
        # the nets build act-1 decks well; act-2/3 bosses (Kaiser Crab: 428 HP, 25-57 a turn)
        # need decks judged against the boss itself
        act0 = (st.get("context") or {}).get("act") or 1
        if not (DECKTEST_FROM_ACT and act0 >= DECKTEST_FROM_ACT and DECK_TEST and sim is not None):
            return CARD_PRIOR(st)
    if DECK_TEST and sim is not None and st.get("cards"):
        import decktest
        scores = decktest.evaluate_options(sim, st)
        if scores:
            vb = value_bonus(st, list(scores))
            if CARD_PRIOR is not None:   # the RL card network as a prior (AlphaGo's policy prior)
                acts_, z_ = CARD_PRIOR.scores(st)
                z_ = z_ - z_.max()
                lp_ = z_ - np.log(np.exp(z_).sum())
                for a_, l_ in zip(acts_, lp_):
                    k_ = a_[1].get("card_index") if a_[0] == "select_card_reward" else None
                    if k_ in scores:
                        vb[k_] = vb.get(k_, 0.0) + CARD_PRIOR_WEIGHT * float(l_) / VALUE_WEIGHT
            full = (st.get("player") or {}).get("deck")
            deck_n = (st.get("player") or {}).get("deck_size") or 10
            act = (st.get("context") or {}).get("act") or 1
            by_index = {c["index"]: c for c in st["cards"]}

            def total(arm):
                if arm is None:
                    return scores[arm] + VALUE_WEIGHT * vb.get(arm, 0.0)
                c = by_index[arm]
                # the deck test for this deck, the hand-made tier prior, and the card lab's
                # average effect across 48 A10 decks (less noisy, same units as the test)
                return (scores[arm] + TIER_WEIGHT * (card_value(c, deck_n, full, act) - SKIP_BELOW)
                        + LAB_WEIGHT * lab_value(base_key(c), act, (st.get("context") or {}).get("floor") or 0)
                        + VALUE_WEIGHT * vb.get(arm, 0.0))

            best = max(scores, key=total)
            if REFINE_REPS and len(scores) > 1:
                # second stage: the two best by the quick test, fought by the real planner
                top = sorted(scores, key=total, reverse=True)[:2]
                arms = {k: ([("skip_card_reward", {})] if k is None else [("select_card_reward", {"card_index": k})])
                        for k in top}
                fine = decktest.evaluate_arms(sim, st, arms, reps=REFINE_REPS, tag="fine/", pilot="planner")
                if len(fine) == 2:
                    best = max(fine, key=lambda k: fine[k] + total(k) - scores[k])
            _LAST_SCORES.v = {"test": {str(k): round(v, 2) for k, v in scores.items()},
                              "total": {str(k): round(total(k), 2) for k in scores}}
            if vb:
                _LAST_SCORES.v["value"] = {str(k): round(v, 3) for k, v in vb.items()}
            return ("skip_card_reward", {}) if best is None else ("select_card_reward", {"card_index": best})
    return choose_card_reward_tier(st)


_LAST_SCORES = threading.local()   # per worker thread: deck-test scores of the last card reward


def pop_last_scores():
    v = getattr(_LAST_SCORES, "v", None)
    _LAST_SCORES.v = None
    return v


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
    if room == "RestSite" and getattr(_LAST_REST, "option", None) != "COOK":   # smith (Cook's prompt is a removal)
        order = sorted(cards, key=upgrade_priority, reverse=True)
        token = getattr(_REST_PICK, "token", None)
        if token and hi == 1:   # the path test already chose this upgrade
            from record import card_tok
            _REST_PICK.token = None
            for c in cards:
                if card_tok(c) == token:
                    return ("select_cards", {"indices": str(c.get("index"))})
        if DECK_TEST and sim is not None and hi == 1 and len(order) > 1:
            import decktest
            top = [c.get("index", i) for i, c in enumerate(order[:4])]
            scores = decktest.evaluate_upgrades(sim, st, top)
            if scores:
                prio = {c.get("index", i): upgrade_priority(c) for i, c in enumerate(cards)}
                best = max(scores, key=lambda k: scores[k] + 1.0 * prio.get(k, 0))
                return ("select_cards", {"indices": str(best)})
    elif room == "Event" and event_prompt_kind() == "better":   # event upgrade / duplicate / enchant
        order = sorted(cards, key=upgrade_priority, reverse=True)
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


HEAL_BELOW = float(os.environ.get("STS2_HEAL_BELOW", "0.5"))   # away from the boss, smith unless HP is below this (0.4 lost HP and elites at A10: v21)


# Rest-site options besides Rest and Smith come from relics. Cook removes cards and adds
# max HP, Dig finds a relic, Train (LIFT) gives +1 Strength in every later battle, Hatch
# hatches the Byrdonis Egg, Kindle recharges a relic, Clone copies enchanted cards, Mend
# heals another player (nothing in single player).
REST_PRIORITY = {"COOK": 5.0, "DIG": 4.0, "LIFT": 3.5, "HATCH": 3.0, "SMITH": 2.0, "KINDLE": 1.5,
                 "CLONE": 1.0, "HEAL": 0.0, "MEND": -1.0}
_LAST_REST = threading.local()   # per worker thread: the option that opened the next card prompt


REST_VALUE = False   # heal vs smith by the run value network (needs RUN_VALUE)


def rest_by_value(st):
    """HEAL or SMITH by the run value network: predicted further floors after healing 30%
    of max HP vs after upgrading the card the smith would pick."""
    from record import card_tok, run_state
    rs = run_state(st)
    p = st.get("player") or {}
    deck = p.get("deck") or []
    cand = [c for c in deck if not c.get("upgraded") and c.get("type") not in ("Curse", "Status")]
    if not cand:
        return "HEAL"
    best = max(cand, key=upgrade_priority)
    i = deck.index(best)
    smith_deck = [card_tok(c) + ("+" if j == i else "") for j, c in enumerate(deck)]
    healed = dict(rs, hp=min(rs["mhp"], rs["hp"] + int(0.3 * rs["mhp"])))
    rem, _ = RUN_VALUE.predict([healed, rs], [rs["deck"], smith_deck])
    return "HEAL" if rem[0] > rem[1] else "SMITH"


REST_SMITH_CANDIDATES = 3   # path test: upgrades considered at a rest site (by upgrade_priority)
_REST_PICK = threading.local()   # per worker thread: the card the path test chose to smith


def _set_rest(opt_id):
    def move(s):
        _LAST_REST.option = opt_id
        return ("noop", {})
    return move


def _smith_card(token):
    """Arm move for the smith prompt: upgrade the card with this token."""
    from record import card_tok

    def move(s):
        for c in s.get("cards") or []:
            if card_tok(c) == token:
                return ("select_cards", {"indices": str(c.get("index"))})
        return (None, {})
    return move


def rest_by_path(sim, st, opts):
    """Every rest option (heal, each of the top smith targets, dig, lift, ...) played
    forward to the act boss by the path test; the best one wins. Returns (option, card token)."""
    import decktest
    from record import card_tok
    deck = (st.get("player") or {}).get("deck") or []
    cand = sorted((c for c in deck if not c.get("upgraded") and c.get("type") not in ("Curse", "Status")),
                  key=upgrade_priority, reverse=True)
    seen, tokens = set(), []
    for c in cand:
        t = card_tok(c)
        if t not in seen:
            seen.add(t)
            tokens.append(t)
        if len(tokens) >= REST_SMITH_CANDIDATES:
            break
    arms = {}
    for o in opts:
        oid = o.get("option_id")
        if oid == "SMITH":
            for t in tokens:
                arms[("SMITH", t)] = [_set_rest("SMITH"), ("choose_option", {"option_index": o["index"]}), _smith_card(t)]
        elif oid not in ("CLONE",):   # (clone's prompt picks a card to copy: tier order handles it)
            arms[(oid, None)] = [_set_rest(oid), ("choose_option", {"option_index": o["index"]})]
    scores = decktest.evaluate_arms_path(sim, st, {k: v for k, v in arms.items()}, tag="rest/")
    if len(scores) < 2:
        return None, None
    _LAST_SCORES.v = {"path": {f"{k[0]}:{k[1]}": round(v, 2) for k, v in scores.items()},
                      "route": (decktest.LAST_PATH or {}).get("route")}
    best = max(scores, key=lambda k: (scores[k], REST_PRIORITY.get(k[0], 0.5)))
    return best


def choose_rest(st, sim=None):
    opts = [o for o in st.get("options") or [] if o.get("is_enabled", True)]
    if not opts:
        return ("leave_room", {})
    by = {o.get("option_id"): o for o in opts}
    _REST_PICK.token = None
    if STRAT is not None and sim is not None and len(opts) > 1:
        from restnet import option_keys
        keys = option_keys(st)
        arms = {k: [_set_rest(k.split(":")[0]), ("choose_option", {"option_index": idx})]
                + ([_smith_card(tok)] if tok else []) for k, (idx, tok) in keys.items()}
        scores = STRAT.evaluate(sim, st, arms, tag="rest")
        if scores:
            best = max(scores, key=scores.get)
            idx, tok = keys[best]
            _LAST_SCORES.v = {"strat": {k: round(v, 2) for k, v in scores.items()}}
            _LAST_REST.option = best.split(":")[0]
            _REST_PICK.token = tok
            return ("choose_option", {"option_index": idx})
    if REST_NET is not None and sim is not None and len(opts) > 1:
        keys, ks, z, _ = REST_NET.scores(sim, st)
        if ks:
            k = ks[int(np.argmax(z))]
            idx, tok = keys[k]
            _LAST_REST.option = k.split(":")[0]
            _REST_PICK.token = tok
            return ("choose_option", {"option_index": idx})
    if DECK_TEST and sim is not None and len(opts) > 1:
        import decktest
        if decktest.PATH_TEST:
            oid, token = rest_by_path(sim, st, opts)
            if oid in by:
                _LAST_REST.option = oid
                _REST_PICK.token = token
                return ("choose_option", {"option_index": by[oid]["index"]})
    if REST_VALUE and RUN_VALUE is not None and "HEAL" in by and "SMITH" in by \
            and max(opts, key=lambda o: REST_PRIORITY.get(o.get("option_id"), 0.5)).get("option_id") in ("HEAL", "SMITH"):
        pick = by[rest_by_value(st)]
        _LAST_REST.option = pick.get("option_id")
        return ("choose_option", {"option_index": pick["index"]})
    # arrive at the boss healthy: the last rest site before it heals unless nearly full
    if "HEAL" in by and hp_ratio(st) < (0.9 if boss_is_next(sim, st) else HEAL_BELOW):
        pick = by["HEAL"]
    else:
        pick = max(opts, key=lambda o: REST_PRIORITY.get(o.get("option_id"), 0.5))
    _LAST_REST.option = pick.get("option_id")
    return ("choose_option", {"option_index": pick["index"]})


EVENT_TEXT_WEIGHT = 3.0   # graded-score points per keyword-score point in event choices
EVENT_TEXT_WEIGHT_PATH = 1.0   # ... with the path test (it simulates the HP side itself)
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
    if len(neow) == len(opts) and not (EVENT_NET is not None and EVENT_NET_NEOW and sim is not None):
        best = max(opts, key=lambda o: NEOW[o["title"]])
        _remember_event(best)
        return ("choose_option", {"option_index": best["index"]})
    if EVENT_NET is not None and sim is not None and len(opts) > 1:
        eo, _, z, _ = EVENT_NET.scores(sim, st)
        if z is not None:
            best = eo[int(np.argmax(z))]
            _remember_event(best)
            return ("choose_option", {"option_index": best["index"]})
    if DECK_TEST and sim is not None and len(opts) > 1:
        import decktest
        scores = decktest.evaluate_event(sim, st)
        if len(scores) > 1:
            by = {o["index"]: o for o in opts}
            # the path test simulates HP, max HP, relics and cards; the keyword score then
            # only adds what it cannot see (gold, curses' later cost, ...)
            w = EVENT_TEXT_WEIGHT_PATH if decktest.PATH_TEST else EVENT_TEXT_WEIGHT
            best = max(scores, key=lambda k: scores[k] + w * text_score(by[k], hp))
            _remember_event(by[best])
            return ("choose_option", {"option_index": best})
    best = max(opts, key=lambda o: (text_score(o, hp), -o["index"]))
    _remember_event(best)
    return ("choose_option", {"option_index": best["index"]})


_LAST_EVENT = threading.local()   # per worker thread: text of the event option just chosen


def _remember_event(opt):
    _LAST_EVENT.text = f"{opt.get('title', '')} {opt.get('description', '')}".lower()


def event_prompt_kind():
    """What the card prompt after an event option is for: 'better' (upgrade, duplicate,
    enchant: pick the best cards) or 'worse' (remove, transform: pick the worst)."""
    t = getattr(_LAST_EVENT, "text", "") or ""
    if re.search(r"upgrade|duplicate|copy|clone|enchant", t) and not re.search(r"remove|transform", t):
        return "better"
    return "worse"


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
    empty = free_potion_slots(st)   # the state lists filled slots only (no None for empty ones)
    pots = [p for p in st.get("potions") or [] if p.get("is_stocked", True) and (p.get("cost") or 9999) <= gold]
    if pots and empty > 0 and m["potions"] < 2:   # potions decide boss fights; fill free slots
        m["potions"] += 1
        return ("buy_potion", {"potion_index": min(pots, key=lambda p: p.get("cost") or 0)["index"]})
    return ("leave_room", {})


def load_deck_nets(cfg):
    """cfg: {"card": path, "rest": path, "event": path, "map": path, "shop": path,
    "card_decides": bool, "neow": bool} -> sets the deck-network globals (the v33 setup)."""
    import torch
    torch.set_num_threads(1)
    global CARD_PRIOR, REST_NET, EVENT_NET, MAP_NET, SHOP_NET, CARD_NET_DECIDES, EVENT_NET_NEOW
    if cfg.get("card"):
        import nets
        CARD_PRIOR = nets.CardPolicy(cfg["card"])
    if cfg.get("rest"):
        from restnet import RestPolicy
        REST_NET = RestPolicy(cfg["rest"])
    if cfg.get("event"):
        from eventnet import EventPolicy
        EVENT_NET = EventPolicy(cfg["event"])
    if cfg.get("map"):
        from mapnet import MapPolicy
        MAP_NET = MapPolicy(cfg["map"])
    if cfg.get("shop"):
        from shopnet import ShopPolicy
        SHOP_NET = ShopPolicy(cfg["shop"])
    CARD_NET_DECIDES = bool(cfg.get("card_decides"))
    EVENT_NET_NEOW = bool(cfg.get("neow"))


def decide(sim, st, avoid=(), memo=None):
    """avoid: JSON-encoded actions that made no progress from this exact state before.
    memo: per-run scratch dict (shop visits)."""
    if st.get("decision") == "shop" and SHOP_NET is not None and sim is not None:
        memo = memo if memo is not None else {}
        ctx = st.get("context") or {}
        visit = memo.setdefault(("shopnet", ctx.get("act"), ctx.get("floor")), {"removed": False, "bought": 0})
        opts, _, z = SHOP_NET.scores(sim, st, visit)
        for i in np.argsort(-z):
            a = opts[int(i)][1]
            if json.dumps(a, sort_keys=True) not in avoid:
                if a[0] == "leave_room" and SHOP_POTIONS:   # the net is done: potions with what is left
                    pa = shop_potion(st)
                    if pa is not None and json.dumps(pa, sort_keys=True) not in avoid:
                        return pa
                if a[0] == "remove_card":
                    visit["removed"] = True
                elif a[0].startswith("buy_"):
                    visit["bought"] += 1
                return a
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
