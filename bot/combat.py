"""Combat turn planner: depth-first search over this turn's plays with snapshot/restore.

Fairness: the search only expands moves whose outcome is already visible. An action
that reveals hidden information (draws or creates cards, shows draw-pile cards) ends
its branch; the bot executes up to that point and replans with the real result. The
enemy turn is never simulated: incoming damage comes from the visible intents.
"""

import json
from collections import Counter

from engine import legal_actions

# Leaf evaluation weights (HP is the main currency).
W = {
    "hp": 1.0,          # player HP after the enemy's announced attacks
    "dmg": 0.45,        # enemy HP removed this turn
    "dmg_big": 0.8,     # ... in elite / boss fights (long fights lose to scaling)
    "scaling": 1.3,     # extra factor vs. an enemy that grows (Strength / Ritual)
    "kill": 7.0,        # enemy removed (no more attacks from it)
    "vuln": 1.2,        # per Vulnerable stack left on a living enemy
    "weak": 1.0,        # per Weak stack left on a living enemy
    "power_scale": 1.0, # lasting powers: POWER_TURN etc. x expected remaining turns
    "player_debuff": 1.2,
    "potion": 5.0,      # cost of drinking a potion (keeps them for fights that need them)
    "win": 300.0,
    "fight_model": 0.0,     # 0..1: replace fixed damage weights by -future_loss (turns to kill x threat)
    "leader": 1.0,          # damage weight on the leader of Minions (they leave when it dies)
    "illusion_credit": 1.0, # kill/damage credit for Illusions (they revive; killing still stops one attack)
    "stun": 8.0,            # fully blocking an attacking Imbalanced enemy (it gets Stunned)
    "wake": 4.0,            # per Slumber count knocked off a sleeping enemy this turn
    "late_hallway_big": 0,  # 1: act-2+ hallway fights use dmg_big too
    "potion_late_hp": 0.65, # act-2+ hallway fights: potions allowed below this HP fraction
    "potions_any_fight_asc": 99,  # from this ascension potions are considered in every fight
    "eot_hand": 1,      # count end-of-turn damage / HP loss of cards left in hand (Beckon, Burn)
    "lethal": 150.0,
    "draw_card": 4.0,     # HP-equivalent of one card drawn per turn (Mind Rot-style debuffs)
    "danger": 0.0,        # extra weight per HP below the danger line (0 = off)
    "danger_turns": 1.5,  # danger line = this many turns of incoming damage ...
    "danger_floor": 15,   # ... counting at least this much per turn    # announced attacks kill us at the end of this turn (unblocked >= HP)
    "rollout_tiebreak": 0.01,   # weight of the leaf evaluation added to rollout means (breaks ties)
    # graded rollout deaths: a rollout that dies scores ROLLOUT_DEATH plus these per turn it
    # survived and per share of the root's enemy HP it removed, so in fights where every line
    # dies within the horizon the search still prefers dying later with the enemy lower
    "death_turn": 0.0,
    "death_dmg": 0.0,
    # focus fire (bottled_ai ranks "lowest-HP monster" above total enemy HP): per HP taken
    # off the weakest living enemy in multi-enemy fights (Kaiser Crab's two halves)
    "focus": 0.0,
    # diagnostic: 0 = rollouts replay the real hidden future (draw order, enemy rolls) instead
    # of a determinized sample, i.e. a planner that knows the future
    "determinize": 1,
    "steam_lethal": 0.0,    # Waterfall Giant: penalty when its death blast (Steam Eruption) would kill us (80: 9 vs 7 of 44 WG fights)
}
PLAYER_DEBUFFS = ("Vulnerable", "Weak", "Frail")
COUNTED = ("Strength", "Dexterity") + PLAYER_DEBUFFS

# Lasting player powers, valued per remaining turn of the fight (HP-equivalents per unit
# of the power's amount; the fight length is estimated from enemy HP / deck damage).
POWER_TURN = {
    "Strength": 1.2, "Dexterity": 0.9, "Thorns": 0.7, "Pyre": 4.0, "Barricade": 2.0, "Feel No Pain": 0.4,
    "Dark Embrace": 1.0, "Rupture": 0.6, "Inferno": 0.5, "Juggernaut": 0.6, "Corruption": 2.0,
    "Crimson Mantle": 0.5, "Aggression": 2.5, "Hellraiser": 2.0, "Stampede": 2.5, "Vicious": 0.8,
    "Cruelty": 0.04, "Unmovable": 2.0, "Juggling": 2.0, "Metallicize": 0.9,
}
POWER_GROWTH = {"Demon Form": 1.2, "Ritual": 1.2}          # Strength per turn, compounding
POWER_ONCE = {"Artifact": 2.0, "Intangible": 6.0, "Buffer": 6.0, "Radiance": 4.0, "Vigor": 0.6, "Regen": 1.5}
POWER_THIS_TURN = {"Rage", "One-Two Punch", "Colossus", "No Draw", "Flame Barrier"}


# A killed enemy that still acts once more shows up with a huge placeholder HP (the
# Waterfall Giant's Steam Eruption: ~1e9 HP and a DeathBlow intent at the end of the next
# turn). It is dead for damage and kills, but its blow still has to be blocked.
PLACEHOLDER_HP = 10 ** 8
HITS = ("Attack", "DeathBlow")


def alive(st):
    return [e for e in st.get("enemies") or [] if 0 < (e.get("hp") or 0) < PLACEHOLDER_HP]


def eid(e):
    """Stable enemy identity: "index" is the position among living enemies and shifts
    when one dies; "id" is the game's CombatId (older engines: fall back to index)."""
    return e["id"] if e.get("id") is not None else ("i", e.get("index"))


def powers(lst):
    return {p.get("name"): p.get("amount") or 0 for p in lst or []}


def incoming_damage(st):
    total = 0
    for e in st.get("enemies") or []:   # placeholders too: their DeathBlow still lands
        if (e.get("hp") or 0) <= 0:
            continue
        for i in e.get("intents") or []:
            if i.get("type") in HITS and i.get("damage") is not None:
                total += i["damage"] * (i.get("hits") or i.get("repeat") or 1)
    return total


def end_of_turn_hand(st):
    """(blockable damage, HP loss) from cards still in hand at the end of the turn:
    statuses / curses like Beckon ("if this is in your Hand, lose 6 HP") or Burn."""
    dmg, loss = 0, 0
    for c in st.get("hand") or []:
        desc = (c.get("description") or "").lower()
        if "end of your turn" not in desc or "in your hand" not in desc:
            continue
        stt = c.get("stats") or {}
        if "lose" in desc and "hp" in desc:
            loss += stt.get("hploss") or stt.get("hp") or 0
        elif "damage" in desc:
            dmg += stt.get("damage") or 0
    return dmg, loss


def damage_rate(st):
    """Rough damage per turn of this deck (to estimate how long the fight will last)."""
    deck = (st.get("player") or {}).get("deck") or []
    per_card = [(c.get("stats") or {}).get("damage") or 0 for c in deck if c.get("type") == "Attack"]
    avg = sum(per_card) / max(1, len(deck))
    strength = powers(st.get("player_powers")).get("Strength", 0)
    return max(6.0, 4.5 * avg + 2 * strength) if deck else 12.0


def future_weight(st):
    """Discounted number of turns the fight is still expected to last (0.9 per turn, <= 10)."""
    ehp = sum((e.get("hp") or 0) + (e.get("block") or 0) for e in alive(st))
    t = max(1.0, min(10.0, ehp / damage_rate(st)))
    whole = int(t)
    return sum(0.9 ** k for k in range(1, whole + 1)) + (t - whole) * 0.9 ** (whole + 1), t


def future_loss(st):
    """HP still to be lost in this fight: turns to kill everything (enemy HP / our damage
    rate) times what the enemies deal per turn beyond the block we usually make."""
    en = alive(st)
    if not en:
        return 0.0
    rate = damage_rate(st)
    ehp = 0.0
    for e in en:
        vul = 1.25 if powers(e.get("powers")).get("Vulnerable") else 1.0
        ehp += ((e.get("hp") or 0) + (e.get("block") or 0)) / vul
    turns = min(12.0, ehp / rate)
    threat = 0.0
    for e in en:
        pw = powers(e.get("powers"))
        shown = sum(i["damage"] * (i.get("hits") or i.get("repeat") or 1) for i in e.get("intents") or []
                    if i.get("type") in HITS and i.get("damage") is not None)
        base = max(shown, min(15.0, 0.1 * (e.get("max_hp") or 0))) + 1.5 * pw.get("Strength", 0)
        threat += base * (0.75 if pw.get("Weak") else 1.0)
    # we block about a third of it on average; the rest is HP
    return turns * 0.65 * threat


def self_damage_powers(st):
    """HP the player's own debuffs take at the end of this turn (Knowledge Demon's
    Disintegration: "At the end of your turn, take X damage")."""
    total = 0
    for p in st.get("player_powers") or []:
        d = (p.get("description") or "").lower()
        if p.get("type") == "Debuff" and "end of your turn" in d and "take" in d and "damage" in d:
            total += p.get("amount") or 0
    return total


def power_value(st, f, t):
    v = 0.0
    kinds = {p.get("name"): (p.get("type"), (p.get("description") or "").lower()) for p in st.get("player_powers") or []}
    for name, amt in powers(st.get("player_powers")).items():
        amt = amt or 1
        ptype, desc = kinds.get(name, (None, ""))
        if name in PLAYER_DEBUFFS:
            v -= W["player_debuff"] * amt
        elif ptype == "Debuff":
            # unknown debuffs used to count as buffs (+f): Disintegration 6 looked good
            if "end of your turn" in desc and "take" in desc and "damage" in desc:
                v -= amt * f                      # that much HP every remaining turn
            elif "fewer" in desc and "card" in desc:
                v -= W["draw_card"] * amt * f      # Mind Rot: cards not drawn every turn
            else:
                v -= W["player_debuff"] * amt * min(f, 3.0)
        elif name in POWER_THIS_TURN:
            if name == "Flame Barrier":
                hits = sum((i.get("hits") or i.get("repeat") or 1) for e in alive(st) for i in e.get("intents") or []
                           if i.get("type") in HITS)
                v += 0.6 * amt * hits
        elif name in POWER_GROWTH:
            v += POWER_GROWTH[name] * amt * f * min(t, 10.0) / 2
        elif name == "Plating":
            v += 0.8 * sum(max(0, amt - k) for k in range(int(t) + 1))
        elif name in POWER_ONCE:
            v += POWER_ONCE[name] * amt
        else:
            v += POWER_TURN.get(name, 1.0 / max(1, amt)) * amt * f
    for e in alive(st):   # enemy scaling costs HP every remaining turn
        pw = powers(e.get("powers"))
        v -= 0.5 * pw.get("Strength", 0) * f + 0.5 * pw.get("Ritual", 0) * f * min(t, 10.0) / 2
    return v


def deck_block(st):
    """Rough block the next hand can make (5 cards, 3 energy) from the deck's block cards."""
    deck = (st.get("player") or {}).get("deck") or []
    blocks = sorted(((c.get("stats") or {}).get("block") or 0 for c in deck), reverse=True)
    if not blocks:
        return 0.0
    per_card = sum(blocks) / len(blocks)
    return min(5 * per_card, sum(blocks[:3]))


def steam_threat(root, st):
    """Waterfall Giant explodes one turn after it dies, for its Steam Eruption stacks (+3
    a turn while it lives). Guides: kill it early and keep enough HP + block for the blast.
    Returns the expected blast, or 0 (no Giant, or the blast is already an announced intent)."""
    blast = 0
    for e in st.get("enemies") or []:
        pw = powers(e.get("powers"))
        if (e.get("hp") or 0) >= PLACEHOLDER_HP:
            if any(i.get("type") == "DeathBlow" for i in e.get("intents") or []):
                return 0   # counted by incoming_damage
            root_steam = max([powers(r.get("powers")).get("Steam Eruption", 0) for r in root.get("enemies") or []] + [0])
            blast = max(blast, pw.get("Steam Eruption", 0) or root_steam)
        elif "Steam Eruption" in pw and (e.get("hp") or 0) > 0:
            blast = max(blast, pw["Steam Eruption"] + 3)
    return blast


def evaluate(root, st):
    d = st.get("decision")
    if d == "game_over":
        return W["win"] * 10 if st.get("victory") else -W["win"] * 10
    if d not in ("combat_play", "card_select"):
        # combat is over and we are alive
        return W["win"] + W["hp"] * ((st.get("player") or {}).get("hp") or 0)
    p = st.get("player") or {}
    hp, block = p.get("hp") or 0, p.get("block") or 0
    eot_dmg, eot_loss = end_of_turn_hand(st) if W["eot_hand"] else (0, 0)
    eot_dmg += self_damage_powers(st)
    taken = max(0, incoming_damage(st) + eot_dmg - block) + eot_loss
    score = W["hp"] * (hp - taken)
    if d == "combat_play" and hp > 0 and taken >= hp:
        # the linear HP term let a big power (Demon Form) outscore surviving the turn
        score -= W["lethal"]
    elif d == "combat_play" and any(0 < powers(e.get("powers")).get("Sandpit", 99) <= 1 for e in alive(st)):
        # The Insatiable: Sandpit drops by 1 each round and eats the player at 0 (Frantic
        # Escape adds 1)
        score -= W["lethal"]
    if W["danger"] and d == "combat_play":
        # HP is worth more when little is left: below what the enemies hit for in about a
        # turn and a half, each HP counts extra (boss races: attack instead of Defend at 42
        # HP into 24 damage, then dead two turns later)
        danger = W["danger_turns"] * max(incoming_damage(st), W["danger_floor"])
        score -= W["danger"] * max(0.0, danger - (hp - taken))
    root_alive, now_alive = alive(root), alive(st)
    room = (root.get("context") or {}).get("room_type")
    late = ((root.get("context") or {}).get("act") or 1) >= 2 and W["late_hallway_big"]
    w_dmg = W["dmg_big"] if room in ("Elite", "Boss") or late else W["dmg"]
    now_by_id = {eid(e): e for e in now_alive}
    fm = W["fight_model"]
    # Minions leave with their leader; Illusions revive at full HP. Damage to an Illusion
    # is wasted (its attack prevented this turn still counts via incoming damage), and the
    # leader's HP is what ends the fight.
    has_minions = any("Minion" in powers(e.get("powers")) for e in root_alive)
    for e in root_alive:
        pw = powers(e.get("powers"))
        dealt = e["hp"] - (now_by_id[eid(e)]["hp"] if eid(e) in now_by_id else 0)
        grows = any(k in pw for k in ("Strength", "Ritual"))
        mult = W["illusion_credit"] if "Illusion" in pw else W["leader"] if has_minions and "Minion" not in pw else 1.0
        score += (1 - fm) * w_dmg * (W["scaling"] if grows else 1.0) * dealt * mult
    if fm:
        score -= fm * future_loss(st)
    if W["focus"] and len(root_alive) >= 2:
        weakest = min(e["hp"] for e in root_alive)
        killed = any(eid(e) not in now_by_id for e in root_alive)
        now_weakest = 0 if killed else min((e["hp"] for e in now_alive), default=0)
        score += W["focus"] * max(0, weakest - now_weakest)
    # enemies from the root that are dead now (a kill that spawns new enemies is still a kill)
    score += W["kill"] * sum((W["illusion_credit"] if "Illusion" in powers(e.get("powers")) else 1.0)
                             for e in root_alive if eid(e) not in now_by_id)
    # mechanics the one-turn view misses: a fully blocked Imbalanced attacker is stunned;
    # every HP loss of a Slumbering enemy brings its waking closer
    inc_now = incoming_damage(st)
    for e in now_alive:
        pw = powers(e.get("powers"))
        if "Imbalanced" in pw and e.get("intends_attack") and inc_now > 0 and block >= inc_now:
            score += W["stun"]
    root_slumber = {eid(e): powers(e.get("powers")).get("Slumber", 0) for e in root_alive}
    for e in now_alive:
        before = root_slumber.get(eid(e), 0)
        if before > 0:
            score -= W["wake"] * max(0, before - powers(e.get("powers")).get("Slumber", 0))
    for e in now_alive:
        pw = powers(e.get("powers"))
        score += W["vuln"] * pw.get("Vulnerable", 0) + W["weak"] * pw.get("Weak", 0)
    f, t = future_weight(st)
    score += W["power_scale"] * power_value(st, f, t)
    score -= W["potion"] * max(0, len(potions(root)) - len(potions(st)))
    if W["steam_lethal"]:
        blast = steam_threat(root, st)
        if blast:
            left = hp - max(0, incoming_damage(st) - block)
            deficit = blast - deck_block(st) - left
            if deficit >= 0:
                score -= W["steam_lethal"] + deficit
    return score


def potions(st):
    return [x for x in (st.get("player") or {}).get("potions") or [] if x]


def potion_actions(st):
    """use_potion moves, only where they matter: elite/boss fights or low HP."""
    room = (st.get("context") or {}).get("room_type")
    p = st.get("player") or {}
    late = ((st.get("context") or {}).get("act") or 1) >= 2
    low = (p.get("hp") or 0) < (W["potion_late_hp"] if late else 0.5) * max(1, p.get("max_hp") or 1)
    # from A4 on (one potion slot fewer) guides say: drink to prevent HP loss now rather than
    # hoard for a boss you may not reach; the planner still charges W["potion"] per potion
    early = ((st.get("context") or {}).get("ascension") or 0) >= W["potions_any_fight_asc"]
    if st.get("decision") != "combat_play" or not (room in ("Elite", "Boss") or low or early):
        return []
    acts = []
    targets = [e["index"] for e in alive(st)]
    for pot in potions(st):
        if pot.get("target_type") == "AnyEnemy":
            acts += [("use_potion", {"potion_index": pot["index"], "target_index": t}) for t in targets]
        else:
            acts.append(("use_potion", {"potion_index": pot["index"]}))
    return acts


def reveals_hidden_info(prev, cur):
    """True if the action that led from prev to cur showed cards the player didn't know."""
    if (cur.get("draw_pile_count") or 0) < (prev.get("draw_pile_count") or 0):
        return True
    before = Counter(c.get("id") for c in prev.get("hand") or [])
    if cur.get("decision") == "card_select":
        shown = Counter(c.get("id") for c in cur.get("cards") or [])
        return bool(shown - before)
    after = Counter(c.get("id") for c in cur.get("hand") or [])
    return bool(after - before)


def canon(st):
    return json.dumps(st, sort_keys=True)


def card_numbers(c, energy):
    """(damage per target, hits, block) of a card from its stats (calculated values first)."""
    stt = c.get("stats") or {}
    desc = (c.get("description") or "").lower()
    dmg = stt.get("calculateddamage") if stt.get("calculateddamage") is not None else stt.get("damage") or 0
    if stt.get("repeat"):
        hits = stt["repeat"]
    elif "x times" in desc:
        hits = energy
    else:
        hits = 2 if "twice" in desc else 3 if "3 times" in desc or "three times" in desc else 1
    blk = stt.get("calculatedblock") if stt.get("calculatedblock") is not None else stt.get("block") or 0
    return dmg or 0, hits, blk or 0


def quick_card_value(c, target, st, need, energy):
    """Rough worth of playing card c now (HP-equivalents), for the rollout policy."""
    stt = c.get("stats") or {}
    desc = (c.get("description") or "").lower()
    en = alive(st)
    dmg, hits, blk = card_numbers(c, energy)
    v = 0.0
    if dmg:
        strength = powers(st.get("player_powers")).get("Strength", 0)
        per = dmg + strength
        targets = en if "all enemies" in desc else ([target] if target else en[:1])
        minions = any("Minion" in powers(x.get("powers")) for x in en)
        for e in targets:
            epw = powers(e.get("powers"))
            vul = 1.5 if epw.get("Vulnerable") else 1.0
            total = per * hits * vul
            eff = (e.get("hp") or 0) + (e.get("block") or 0)
            if "Illusion" in epw and W["illusion_credit"] < 1.0:   # revives at full HP
                v += (4.0 if total >= eff and e.get("intends_attack") else 0.0)
                continue
            v += min(total, eff) * (W["leader"] if minions and "Minion" not in epw else 1.0)
            if total >= eff:
                v += 6.0 + (4.0 if e.get("intends_attack") else 0.0)
    pp = powers(st.get("player_powers"))
    if blk:
        keep = 0.6 if "Barricade" in pp else 0.1   # Barricade: block beyond this turn's attacks carries over
        v += 1.2 * min(blk, need) + keep * max(0, blk - need)
    v += 1.5 * (stt.get("vulnerablepower") or 0) * (1 if en else 0)
    v += 1.0 * (stt.get("cards") or 0) + 2.5 * (stt.get("energy") or 0) - 0.8 * (stt.get("hploss") or 0)
    if c.get("type") == "Power":
        # worth its effect over the rest of the fight (the planner's view of powers), so a
        # 3-cost Demon Form is played before Strikes in a long fight but not in a short one
        f, t = future_weight(st)
        per_turn = 2.0 + 1.5 * (stt.get("strengthpower") or 0)
        if "start of your turn" in desc or "each turn" in desc:
            per_turn *= 1 + min(t, 10.0) / 4
        v += 2.0 + f * per_turn
    elif c.get("type") in ("Status", "Curse"):
        v += 0.5
    elif v == 0.0:
        v = 2.0
    return v


def quick_policy(st):
    """Fast rule policy for rollouts: no search, one sim call per action.
    Play the card with the best value per energy (kills, needed block, powers early),
    end the turn when nothing is worth playing."""
    d = st.get("decision")
    if d == "card_select":
        cards = st.get("cards") or []
        lo = st.get("min_select") or 0
        if not cards or lo == 0:
            return ("skip_select", {})
        def junk(c):
            k = (c.get("id") or "")
            return (0 if c.get("type") in ("Status", "Curse") else 1 if ("STRIKE_" in k or "DEFEND_" in k) else 2)
        order = sorted(range(len(cards)), key=lambda i: junk(cards[i]))
        return ("select_cards", {"indices": ",".join(str(cards[i].get("index", i)) for i in order[:max(1, lo)])})
    if d != "combat_play":
        return ("end_turn", {})
    p = st.get("player") or {}
    need = max(0, incoming_damage(st) - (p.get("block") or 0))
    energy = st.get("energy") or 0
    en = {e["index"]: e for e in alive(st)}
    hand = {c["index"]: c for c in st.get("hand") or []}
    best, best_v = ("end_turn", {}), 0.4
    for a in legal_actions(st):
        if a[0] != "play_card":
            continue
        c = hand.get(a[1]["card_index"])
        if c is None:
            continue
        cost = c.get("cost")
        cost = energy if not isinstance(cost, int) or cost < 0 else cost
        v = quick_card_value(c, en.get(a[1].get("target_index")), st, need, energy) / max(cost, 0.6)
        if v > best_v:
            best, best_v = a, v
    return best


ROLLOUT_DEATH = -300.0

# Optional learned fast policy (nets.NetPolicy: the imitation-trained policy network) that
# replaces quick_policy in the planner's rollouts, like AlphaGo's rollout policy.
FAST_POLICY = None
# Optional combat value network (nets.CombatValue: expected HP change until the fight ends,
# learned from self-play). A rollout's end state is then judged by
# (1 - lam) * evaluate + lam * (win + HP + predicted change), lam from the rollout config.
ROLLOUT_VALUE = None
# Optional expert value network scoring the planner's leaves directly (AlphaGo's value
# network at the leaves): score = (1 - LEAF_MU) * evaluate + LEAF_MU * (win * P(win) + HP + change).
LEAF_VALUE = None
LEAF_MU = 1.0


def fast_policy(st):
    return FAST_POLICY(st) if FAST_POLICY is not None else quick_policy(st)


STALL_ROUND = 30   # past this round the learned value is ignored for the rest of the fight


class TurnPlanner:
    """value: optional learned model (value.ValueModel) that scores leaves instead of
    the hand-written evaluate(); explore: probability of picking a random top-3 leaf
    (data collection only); recorder(state): called with each end-of-turn state."""

    def __init__(self, sim, max_nodes=400, max_depth=10, value=None, explore=0.0, rng=None, recorder=None,
                 value_weight=None, rollouts=None):
        self.sim, self.max_nodes, self.max_depth = sim, max_nodes, max_depth
        # rollouts: {"k": leaves to roll out, "m": rollouts per leaf, "turns": future turns}.
        # The top-k leaves (by evaluate) are re-scored by the mean outcome of m determinized
        # rollouts (draw order shuffled, combat RNG reseeded: no peeking) with quick_policy.
        self.rollouts = rollouts
        self.value, self.explore, self.rng, self.recorder = value, explore, rng, recorder
        # None: the model's prediction is the score. A number w: hand-written evaluation
        # plus w * prediction (the model only adds its longer-horizon judgement).
        self.value_weight = value_weight
        self.stats = Counter()

    def _snap(self):
        return self.sim.send({"cmd": "snapshot"})["id"]

    def _restore(self, sid):
        return self.sim.send({"cmd": "restore", "id": sid})

    def _score_leaves(self, root, leaves):
        """leaves: [(kind, path, state)], kind in end / terminal / partial."""
        scores = [evaluate(root, st) for _, _, st in leaves]
        if LEAF_VALUE is not None and (root.get("round") or 0) <= STALL_ROUND:
            out = list(scores)
            for i, (kind, _, st) in enumerate(leaves):
                if kind != "terminal" and st.get("decision") in ("combat_play", "card_select"):
                    pw, dhp = LEAF_VALUE(st)
                    hp = (st.get("player") or {}).get("hp") or 0
                    out[i] = (1 - LEAF_MU) * scores[i] + LEAF_MU * (W["win"] * pw + W["hp"] * (hp + dhp))
            return [sc - (0.01 if kind == "partial" else 0.0) for sc, (kind, _, _) in zip(out, leaves)]
        if self.value is None or (root.get("round") or 0) > STALL_ROUND:
            # a fight this long means the model is stalling (e.g. predicting more than the
            # current HP for a live position, so it never lands the kill): finish it by hand
            return [sc - (0.01 if kind == "partial" else 0.0) for sc, (kind, _, _) in zip(scores, leaves)]
        from value import DEATH_VALUE
        idx = [i for i, (kind, _, st) in enumerate(leaves) if kind != "terminal"]
        preds = self.value.predict([leaves[i][2] for i in idx])
        out = list(scores)
        for i, v in zip(idx, preds):
            if self.value_weight is None:
                # the fight can't end with more HP than now (in-combat heals are rare), and a
                # prediction above it would rank "keep the enemy alive" over winning
                hp = (leaves[i][2].get("player") or {}).get("hp") or 0
                out[i] = min(float(v), hp) + 1e-3 * scores[i]      # tiny heuristic tie-break
            else:
                out[i] = scores[i] + self.value_weight * float(v)
        for i, (kind, _, st) in enumerate(leaves):
            if kind == "terminal" and self.value_weight is None:
                won = st.get("decision") != "game_over" or st.get("victory")
                out[i] = ((st.get("player") or {}).get("hp") or 0) + 2.0 if won else DEATH_VALUE
        return out

    def _death_value(self, root, seen, future):
        e0 = sum(e.get("hp") or 0 for e in alive(root))
        e1 = sum(e.get("hp") or 0 for e in alive(seen)) if seen is not None else e0
        return ROLLOUT_DEATH + W["death_turn"] * future + W["death_dmg"] * max(0.0, (e0 - e1) / max(1, e0))

    def _rollout(self, root, snap, last, seed, turns, pilot_nodes=0, lam=0.0):
        self._restore(snap)
        if W["determinize"]:
            self.sim.send({"cmd": "determinize", "seed": seed})
        s = self.sim.act(last[0], **last[1])
        future = 1 if last[0] == "end_turn" else 0
        seen = s if s.get("enemies") is not None else None   # last state that shows the enemies
        if pilot_nodes:
            s = self._planner_rollout(s, future, turns, pilot_nodes)
            self.stats["rollouts"] += 1
            if s.get("type") == "error":
                return None
            if s.get("decision") == "game_over" and not s.get("victory"):
                return ROLLOUT_DEATH
            return evaluate(root, s)
        for _ in range(80):
            if s.get("type") == "error" or s.get("decision") not in ("combat_play", "card_select"):
                break
            a = fast_policy(s)
            if a[0] == "end_turn" and future >= turns:
                break
            s2 = self.sim.act(a[0], **a[1])
            if s2.get("type") == "error":
                if s.get("decision") != "combat_play":
                    break
                a, s2 = ("end_turn", {}), self.sim.act("end_turn")
                if future >= turns or s2.get("type") == "error":
                    break
            if a[0] == "end_turn":
                future += 1
            s = s2
            if s.get("enemies") is not None:
                seen = s
        self.stats["rollouts"] += 1
        if s.get("type") == "error":
            return None
        if s.get("decision") == "game_over" and not s.get("victory"):
            return self._death_value(root, seen, future)
        sc = evaluate(root, s)
        if lam and ROLLOUT_VALUE is not None and s.get("decision") in ("combat_play", "card_select"):
            hp = (s.get("player") or {}).get("hp") or 0
            v = ROLLOUT_VALUE(s)
            if isinstance(v, tuple):   # expert value net: (P(win), HP change to the end)
                pw, dhp = v
                sc = (1 - lam) * sc + lam * (W["win"] * pw + W["hp"] * (hp + dhp))
            else:
                sc = (1 - lam) * sc + lam * (W["win"] + W["hp"] * (hp + v))
        elif lam and ROLLOUT_VALUE is not None:
            sc = (1 - lam) * sc + lam * evaluate(root, s)
        return sc

    def _planner_rollout(self, s, future, turns, nodes):
        """Play the imagined future with a small planner (one-turn search, heuristic leaves)
        instead of quick_policy; stop before the end of turn `turns`."""
        inner = TurnPlanner(self.sim, max_nodes=nodes)
        for _ in range(40):
            if s.get("type") == "error" or s.get("decision") not in ("combat_play", "card_select"):
                return s
            path, _ = inner.plan(s)
            for a in path:
                if a[0] == "end_turn" and future >= turns:
                    return s
                s2 = self.sim.act(a[0], **a[1])
                if s2.get("type") == "error":
                    if s.get("decision") != "combat_play":
                        return s
                    a, s2 = ("end_turn", {}), self.sim.act("end_turn")
                if a[0] == "end_turn":
                    future += 1
                s = s2
                if s.get("type") == "error" or s.get("decision") not in ("combat_play", "card_select"):
                    return s
        return s

    def _rollout_cfg(self, root):
        """The "big" budget (if given) for elite/boss fights and every fight from act 2 on."""
        ctx = root.get("context") or {}
        if ctx.get("room_type") == "Boss" and "boss" in self.rollouts:
            return self.rollouts["boss"]
        big = ctx.get("room_type") in ("Elite", "Boss") or (ctx.get("act") or 1) >= 2
        return self.rollouts.get("big", self.rollouts) if big else self.rollouts

    def _rollout_rescore(self, root, leaves, scores, cfg=None):
        cfg = cfg or self.rollouts
        cand = [i for i in sorted(range(len(leaves)), key=lambda i: scores[i], reverse=True)
                if leaves[i][0] != "terminal"][:cfg.get("k", 4)]
        if len(cand) < 2:
            return scores
        # hash() of an int tuple is the same in every process and Python version; None hashes
        # by address (a different rollout seed in every process), so it becomes -1
        key = (root.get("round"), len(root.get("hand") or []), root.get("energy"))
        base = hash(tuple(-1 if v is None else v for v in key)) & 0xFFFFFFFF
        out = [float("-inf")] * len(leaves)
        for i, (kind, _, st, snap, last) in enumerate(leaves):
            if kind == "terminal":   # a finished fight needs no rollout; put it on the rollout scale
                out[i] = scores[i] if (st.get("decision") != "game_over" or st.get("victory")) else ROLLOUT_DEATH
        for i in cand:
            _, _, st, snap, last = leaves[i]
            vals = [self._rollout(root, snap, last, base * 1000 + j, cfg.get("turns", 2), cfg.get("pilot_nodes", 0),
                                  cfg.get("lam", 0.0)) for j in range(cfg.get("m", 3))]
            vals = [v for v in vals if v is not None]
            mean = sum(vals) / len(vals) if vals else scores[i]
            # shrink toward the leaf's own evaluation: with few noisy rollouts, the max
            # over candidates otherwise picks whichever one got lucky
            beta = cfg.get("blend", 0.0)
            # tie-break by the leaf's own evaluation: when every rollout dies (hopeless-looking
            # positions in hard fights) the means are all ROLLOUT_DEATH and the first leaf won,
            # even one that does not survive this turn (Kaiser Crab: no block, back turned)
            out[i] = (1 - beta) * mean + beta * scores[i] + W["rollout_tiebreak"] * scores[i]
        return out

    def plan(self, root):
        """Best action sequence from `root`. It ends with end_turn, a terminal state,
        or an action that revealed hidden information (then the caller replans)."""
        root_snap = self._snap()
        leaves = []
        seen = {canon(root)}
        nodes = 0

        def visit(st, path, snap_id):
            nonlocal nodes
            if st.get("decision") == "combat_play":
                leaves.append(("end", path + [("end_turn", {})], st, snap_id, ("end_turn", {})))
            for a in legal_actions(st) + potion_actions(st):
                if a[0] == "end_turn":
                    continue
                if nodes >= self.max_nodes:
                    self.stats["node_cap"] += 1
                    return
                self._restore(snap_id)
                s2 = self.sim.act(a[0], **a[1])
                nodes += 1
                if s2.get("type") == "error":
                    continue
                key = canon(s2)
                if key in seen:
                    continue
                seen.add(key)
                p2 = path + [a]
                d2 = s2.get("decision")
                if d2 not in ("combat_play", "card_select"):
                    leaves.append(("terminal", p2, s2, snap_id, a))
                elif reveals_hidden_info(st, s2) or len(p2) >= self.max_depth:
                    # stop here: what follows depends on information we don't have yet
                    leaves.append(("partial", p2, s2, snap_id, a))
                else:
                    visit(s2, p2, self._snap())

        visit(root, [], root_snap)
        self.stats["plans"] += 1
        self.stats["nodes"] += nodes
        if not leaves:
            self._restore(root_snap)
            self.sim.send({"cmd": "drop_snapshots", "from": root_snap})
            return [("end_turn", {})], float("-inf")
        scores = self._score_leaves(root, [l[:3] for l in leaves])
        heur = scores
        if self.rollouts and (root.get("round") or 0) <= STALL_ROUND:
            scores = self._rollout_rescore(root, leaves, scores, self._rollout_cfg(root))
        # best leaf score per first action (search targets for the policy network)
        firsts, hfirst = {}, {}
        for leaf, sc, hs in zip(leaves, scores, heur):
            if leaf[1]:
                k = json.dumps([leaf[1][0][0], leaf[1][0][1]], sort_keys=True)
                if sc != float("-inf"):
                    firsts[k] = max(firsts.get(k, float("-inf")), sc)
                hfirst[k] = max(hfirst.get(k, float("-inf")), hs)
        self.last_root, self.last_first = root, {"q": firsts, "h": hfirst}
        self._restore(root_snap)
        self.sim.send({"cmd": "drop_snapshots", "from": root_snap})
        order = sorted(range(len(leaves)), key=lambda i: scores[i], reverse=True)
        pick = order[0]
        if self.explore and self.rng is not None and len(order) > 1 and self.rng.random() < self.explore:
            pick = self.rng.choice(order[:3])
            self.stats["explored"] += 1
        return leaves[pick][1], scores[pick]

    def play_combat(self, state, max_actions=600, on_state=None, on_step=None):
        """Play until the combat ends. Returns the first non-combat state.
        on_step(state, action): called before each executed action (self-play records)."""
        n = 0
        if on_state:
            on_state(state)
        while state.get("decision") in ("combat_play", "card_select") and n < max_actions:
            path, _ = self.plan(state)
            before = self._snap()
            for a in path:
                if a[0] == "end_turn" and self.recorder is not None:
                    self.recorder(state)
                if on_step:
                    on_step(state, a)
                nxt = self.sim.act(a[0], **a[1])
                n += 1
                if nxt.get("type") == "error":
                    # the protocol has no "get state": go back to the last good state
                    self.stats["exec_error"] += 1
                    state = self._restore(before)
                    if state.get("decision") == "combat_play":
                        state = self.sim.act("end_turn")
                    break
                state = nxt
                if on_state:
                    on_state(state)
            self.sim.send({"cmd": "drop_snapshots", "from": before})
            # replan after every executed path (end of turn, revealed info, or prompt)
        return state
