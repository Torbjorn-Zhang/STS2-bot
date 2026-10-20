"""Self-play records for the AlphaGo-style pipeline (policy nets by imitation, value nets
from outcomes). One JSON line per record; a run's records are written together when the
run ends, followed by its outcome, so a crash never leaves a run without a label.

Record kinds ("k"):
  f   fight header: run state (deck, relics, potions, HP) and enemy names
  c   combat step: compact combat state, legal actions, index of the one played
  fe  fight end: won, HP before/after, rounds
  d   out-of-combat decision: run state, legal actions, the one taken, extras (deck-test scores)
  o   run outcome (last line of the run)
"""

import json


def card_tok(c):
    k = (c.get("id") or "").replace("CARD.", "")
    up = c.get("upgraded") or (c.get("name") or "").endswith("+")
    return k + ("+" if up else "")


def pot_tok(x):
    return (x.get("id") or x.get("name") or "?").replace("POTION.", "")


def powers_dict(lst):
    return {p.get("name"): p.get("amount") for p in lst or [] if p.get("name")}


def run_state(st):
    ctx = st.get("context") or {}
    p = st.get("player") or {}
    return {"act": ctx.get("act"), "floor": ctx.get("floor"), "asc": ctx.get("ascension"),
            "an": ctx.get("act_name"), "boss": (ctx.get("boss") or {}).get("id"), "room": ctx.get("room_type"),
            "hp": p.get("hp"), "mhp": p.get("max_hp"), "gold": p.get("gold"),
            "deck": [card_tok(c) for c in p.get("deck") or []],
            "relics": [r.get("name") for r in p.get("relics") or []],
            "pots": [pot_tok(x) for x in p.get("potions") or [] if x]}


def enemy(e):
    ints = [[i.get("type"), i.get("damage"), i.get("hits") or i.get("repeat")] for i in e.get("intents") or []]
    return [e.get("name"), e.get("hp"), e.get("max_hp"), e.get("block"), powers_dict(e.get("powers")), ints,
            e.get("index"), e.get("id")]


def hand_card(c):
    """[token, cost, playable, hand index, stats (damage/block/... as shown, with modifiers),
    {target index: damage}, keywords]"""
    dbt = {d.get("target_index"): d.get("damage") for d in c.get("damage_by_target") or []}
    return [card_tok(c), c.get("cost"), bool(c.get("can_play")), c.get("index"), c.get("stats") or {}, dbt,
            c.get("keywords") or []]


def combat_state(st):
    p = st.get("player") or {}
    s = {"d": st.get("decision"), "r": st.get("round"), "en": st.get("energy"), "men": st.get("max_energy"),
         "hp": p.get("hp"), "mhp": p.get("max_hp"), "blk": p.get("block"), "pw": powers_dict(st.get("player_powers")),
         "hand": [hand_card(c) for c in st.get("hand") or []],
         "dn": st.get("draw_pile_count"), "xn": st.get("discard_pile_count"),
         "es": [enemy(e) for e in st.get("enemies") or [] if (e.get("hp") or 0) > 0],
         "pots": [pot_tok(x) for x in p.get("potions") or [] if x]}
    if st.get("decision") == "card_select":
        s["sel"] = [card_tok(c) for c in st.get("cards") or []]
        s["min"], s["max"] = st.get("min_select"), st.get("max_select")
    return s


def act_key(a):
    return json.dumps([a[0], a[1]], sort_keys=True)


class GameRecorder:
    def __init__(self, path, lock, seed, meta=None):
        self.path, self.lock, self.seed = path, lock, seed
        self.meta = meta or {}
        self.rows, self.fight = [], -1

    def _add(self, row):
        row["seed"] = self.seed
        self.rows.append(row)

    def fight_start(self, st):
        self.fight += 1
        self._add({"k": "f", "i": self.fight, "run": run_state(st),
                   "enemies": [e.get("name") for e in st.get("enemies") or []]})

    def combat_step(self, st, legal, a, extra=None):
        keys = [act_key(x) for x in legal]
        k = act_key(a)
        row = {"k": "c", "i": self.fight, "s": combat_state(st), "A": [[x[0], x[1]] for x in legal],
               "a": keys.index(k) if k in keys else -1, "played": [a[0], a[1]]}
        if extra:
            row["x"] = extra
        self._add(row)

    def fight_end(self, st, hp_before, won, rounds, hp_last=None):
        """hp_after: first state after the fight (post-combat heals included); hp_last: HP in
        the last combat state (an act boss win heals for the next act)."""
        self._add({"k": "fe", "i": self.fight, "won": won, "hp_before": hp_before,
                   "hp_after": (st.get("player") or {}).get("hp"), "hp_last": hp_last, "rounds": rounds})

    def decision(self, st, legal, a, extra=None):
        row = {"k": "d", "dec": st.get("decision"), "s": run_state(st), "A": [[x[0], x[1]] for x in legal],
               "played": [a[0], a[1]]}
        keys = [act_key(x) for x in legal]
        row["a"] = keys.index(act_key(a)) if act_key(a) in keys else -1
        if st.get("decision") in ("card_reward", "card_select"):
            row["cards"] = [card_tok(c) for c in st.get("cards") or []]
        if st.get("decision") in ("event_choice", "rest_site"):
            row["opts"] = [(o.get("option_id") or o.get("id") or o.get("title") or o.get("text") or "")[:60] for o in st.get("options") or []]
        if st.get("decision") == "map_select":
            row["opts"] = [c.get("type") or c.get("room_type") for c in st.get("choices") or []]
        if extra:
            row["x"] = extra
        self._add(row)

    def finish(self, result):
        out = {k: result.get(k) for k in ("cleared_acts", "victory", "died", "act", "floor", "room", "hp", "max_hp", "boss_dmg",
                                          "boss_mc", "error", "stuck", "mc", "act_states")}
        self._add({"k": "o", **out, **self.meta})
        with self.lock:
            with open(self.path, "a") as fh:
                for r in self.rows:
                    fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        self.rows = []
