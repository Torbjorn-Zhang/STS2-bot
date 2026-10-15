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
    "strength": 2.5,    # per player Strength
    "dexterity": 2.0,
    "other_power": 1.5, # any other lasting player power (Demon Form, Barricade, ...)
    "player_debuff": 1.2,
    "potion": 5.0,      # cost of drinking a potion (keeps them for fights that need them)
    "win": 300.0,
}
PLAYER_DEBUFFS = ("Vulnerable", "Weak", "Frail")
COUNTED = ("Strength", "Dexterity") + PLAYER_DEBUFFS


def alive(st):
    return [e for e in st.get("enemies") or [] if (e.get("hp") or 0) > 0]


def powers(lst):
    return {p.get("name"): p.get("amount") or 0 for p in lst or []}


def incoming_damage(st):
    total = 0
    for e in alive(st):
        for i in e.get("intents") or []:
            if i.get("type") == "Attack" and i.get("damage") is not None:
                total += i["damage"] * (i.get("hits") or i.get("repeat") or 1)
    return total


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
    w_dmg = W["dmg_big"] if room in ("Elite", "Boss") else W["dmg"]
    now_by_index = {e.get("index"): e for e in now_alive}
    for e in root_alive:
        dealt = e["hp"] - (now_by_index[e.get("index")]["hp"] if e.get("index") in now_by_index else 0)
        grows = any(k in powers(e.get("powers")) for k in ("Strength", "Ritual"))
        score += w_dmg * (W["scaling"] if grows else 1.0) * dealt
    score += W["kill"] * (len(root_alive) - len(now_alive))
    for e in now_alive:
        pw = powers(e.get("powers"))
        score += W["vuln"] * pw.get("Vulnerable", 0) + W["weak"] * pw.get("Weak", 0)
    pp = powers(st.get("player_powers"))
    score += W["strength"] * pp.get("Strength", 0) + W["dexterity"] * pp.get("Dexterity", 0)
    score += W["other_power"] * sum(1 for k in pp if k not in COUNTED)
    score -= W["player_debuff"] * sum(pp.get(k, 0) for k in PLAYER_DEBUFFS)
    score -= W["potion"] * max(0, len(potions(root)) - len(potions(st)))
    return score


def potions(st):
    return [x for x in (st.get("player") or {}).get("potions") or [] if x]


def potion_actions(st):
    """use_potion moves, only where they matter: elite/boss fights or low HP."""
    room = (st.get("context") or {}).get("room_type")
    p = st.get("player") or {}
    low = (p.get("hp") or 0) < 0.5 * max(1, p.get("max_hp") or 1)
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


STALL_ROUND = 30   # past this round the learned value is ignored for the rest of the fight


class TurnPlanner:
    """value: optional learned model (value.ValueModel) that scores leaves instead of
    the hand-written evaluate(); explore: probability of picking a random top-3 leaf
    (data collection only); recorder(state): called with each end-of-turn state."""

    def __init__(self, sim, max_nodes=400, max_depth=10, value=None, explore=0.0, rng=None, recorder=None,
                 value_weight=None):
        self.sim, self.max_nodes, self.max_depth = sim, max_nodes, max_depth
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
                leaves.append(("end", path + [("end_turn", {})], st))
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
                    leaves.append(("terminal", p2, s2))
                elif reveals_hidden_info(st, s2) or len(p2) >= self.max_depth:
                    # stop here: what follows depends on information we don't have yet
                    leaves.append(("partial", p2, s2))
                else:
                    visit(s2, p2, self._snap())

        visit(root, [], root_snap)
        self._restore(root_snap)
        self.sim.send({"cmd": "drop_snapshots"})
        self.stats["plans"] += 1
        self.stats["nodes"] += nodes
        if not leaves:
            return [("end_turn", {})], float("-inf")
        scores = self._score_leaves(root, leaves)
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
            # replan after every executed path (end of turn, revealed info, or prompt)
        return state
