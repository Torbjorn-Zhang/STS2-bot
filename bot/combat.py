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
    "stun": 8.0,            # fully blocking an attacking Imbalanced enemy (it gets Stunned)
    "wake": 4.0,            # per Slumber count knocked off a sleeping enemy this turn
    "late_hallway_big": 0,  # 1: act-2+ hallway fights use dmg_big too
    "potion_late_hp": 0.65, # act-2+ hallway fights: potions allowed below this HP fraction
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


def alive(st):
    return [e for e in st.get("enemies") or [] if (e.get("hp") or 0) > 0]


def eid(e):
    """Stable enemy identity: "index" is the position among living enemies and shifts
    when one dies; "id" is the game's CombatId (older engines: fall back to index)."""
    return e["id"] if e.get("id") is not None else ("i", e.get("index"))


def powers(lst):
    return {p.get("name"): p.get("amount") or 0 for p in lst or []}


def incoming_damage(st):
    total = 0
    for e in alive(st):
        for i in e.get("intents") or []:
            if i.get("type") == "Attack" and i.get("damage") is not None:
                total += i["damage"] * (i.get("hits") or i.get("repeat") or 1)
    return total


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
                    if i.get("type") == "Attack" and i.get("damage") is not None)
        base = max(shown, min(15.0, 0.1 * (e.get("max_hp") or 0))) + 1.5 * pw.get("Strength", 0)
        threat += base * (0.75 if pw.get("Weak") else 1.0)
    # we block about a third of it on average; the rest is HP
    return turns * 0.65 * threat


def power_value(st, f, t):
    v = 0.0
    for name, amt in powers(st.get("player_powers")).items():
        amt = amt or 1
        if name in PLAYER_DEBUFFS:
            v -= W["player_debuff"] * amt
        elif name in POWER_THIS_TURN:
            if name == "Flame Barrier":
                hits = sum((i.get("hits") or i.get("repeat") or 1) for e in alive(st) for i in e.get("intents") or []
                           if i.get("type") == "Attack")
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


def evaluate(root, st):
    d = st.get("decision")
    if d == "game_over":
        return W["win"] * 10 if st.get("victory") else -W["win"] * 10
    if d not in ("combat_play", "card_select"):
        # combat is over and we are alive
        return W["win"] + W["hp"] * ((st.get("player") or {}).get("hp") or 0)
    p = st.get("player") or {}
    hp, block = p.get("hp") or 0, p.get("block") or 0
    score = W["hp"] * (hp - max(0, incoming_damage(st) - block))
    root_alive, now_alive = alive(root), alive(st)
    room = (root.get("context") or {}).get("room_type")
    late = ((root.get("context") or {}).get("act") or 1) >= 2 and W["late_hallway_big"]
    w_dmg = W["dmg_big"] if room in ("Elite", "Boss") or late else W["dmg"]
    now_by_id = {eid(e): e for e in now_alive}
    fm = W["fight_model"]
    for e in root_alive:
        dealt = e["hp"] - (now_by_id[eid(e)]["hp"] if eid(e) in now_by_id else 0)
        grows = any(k in powers(e.get("powers")) for k in ("Strength", "Ritual"))
        score += (1 - fm) * w_dmg * (W["scaling"] if grows else 1.0) * dealt
    if fm:
        score -= fm * future_loss(st)
    # enemies from the root that are dead now (a kill that spawns new enemies is still a kill)
    score += W["kill"] * sum(1 for e in root_alive if eid(e) not in now_by_id)
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
    return score


def potions(st):
    return [x for x in (st.get("player") or {}).get("potions") or [] if x]


def potion_actions(st):
    """use_potion moves, only where they matter: elite/boss fights or low HP."""
    room = (st.get("context") or {}).get("room_type")
    p = st.get("player") or {}
    late = ((st.get("context") or {}).get("act") or 1) >= 2
    low = (p.get("hp") or 0) < (W["potion_late_hp"] if late else 0.5) * max(1, p.get("max_hp") or 1)
    if st.get("decision") != "combat_play" or not (room in ("Elite", "Boss") or low):
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
        for e in targets:
            vul = 1.5 if powers(e.get("powers")).get("Vulnerable") else 1.0
            total = per * hits * vul
            eff = (e.get("hp") or 0) + (e.get("block") or 0)
            v += min(total, eff)
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

    def _rollout(self, root, snap, last, seed, turns):
        self._restore(snap)
        self.sim.send({"cmd": "determinize", "seed": seed})
        s = self.sim.act(last[0], **last[1])
        future = 1 if last[0] == "end_turn" else 0
        for _ in range(80):
            if s.get("type") == "error" or s.get("decision") not in ("combat_play", "card_select"):
                break
            a = quick_policy(s)
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
        self.stats["rollouts"] += 1
        if s.get("type") == "error":
            return None
        if s.get("decision") == "game_over" and not s.get("victory"):
            return ROLLOUT_DEATH
        return evaluate(root, s)

    def _rollout_cfg(self, root):
        """The "big" budget (if given) for elite/boss fights and every fight from act 2 on."""
        ctx = root.get("context") or {}
        big = ctx.get("room_type") in ("Elite", "Boss") or (ctx.get("act") or 1) >= 2
        return self.rollouts.get("big", self.rollouts) if big else self.rollouts

    def _rollout_rescore(self, root, leaves, scores, cfg=None):
        cfg = cfg or self.rollouts
        cand = [i for i in sorted(range(len(leaves)), key=lambda i: scores[i], reverse=True)
                if leaves[i][0] != "terminal"][:cfg.get("k", 4)]
        if len(cand) < 2:
            return scores
        base = hash((root.get("round"), len(root.get("hand") or []), root.get("energy"))) & 0xFFFFFFFF
        out = [float("-inf")] * len(leaves)
        for i, (kind, _, st, snap, last) in enumerate(leaves):
            if kind == "terminal":   # a finished fight needs no rollout; put it on the rollout scale
                out[i] = scores[i] if (st.get("decision") != "game_over" or st.get("victory")) else ROLLOUT_DEATH
        for i in cand:
            _, _, st, snap, last = leaves[i]
            vals = [self._rollout(root, snap, last, base * 1000 + j, cfg.get("turns", 2)) for j in range(cfg.get("m", 3))]
            vals = [v for v in vals if v is not None]
            mean = sum(vals) / len(vals) if vals else scores[i]
            # shrink toward the leaf's own evaluation: with few noisy rollouts, the max
            # over candidates otherwise picks whichever one got lucky
            beta = cfg.get("blend", 0.0)
            out[i] = (1 - beta) * mean + beta * scores[i]
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
        if self.rollouts and (root.get("round") or 0) <= STALL_ROUND:
            scores = self._rollout_rescore(root, leaves, scores, self._rollout_cfg(root))
        self._restore(root_snap)
        self.sim.send({"cmd": "drop_snapshots", "from": root_snap})
        order = sorted(range(len(leaves)), key=lambda i: scores[i], reverse=True)
        pick = order[0]
        if self.explore and self.rng is not None and len(order) > 1 and self.rng.random() < self.explore:
            pick = self.rng.choice(order[:3])
            self.stats["explored"] += 1
        return leaves[pick][1], scores[pick]

    def play_combat(self, state, max_actions=600, on_state=None):
        """Play until the combat ends. Returns the first non-combat state."""
        n = 0
        if on_state:
            on_state(state)
        while state.get("decision") in ("combat_play", "card_select") and n < max_actions:
            path, _ = self.plan(state)
            before = self._snap()
            for a in path:
                if a[0] == "end_turn" and self.recorder is not None:
                    self.recorder(state)
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
