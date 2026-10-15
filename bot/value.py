"""Learned position evaluation: features of a combat state and the model wrapper.

The model predicts the outcome of the current fight from an end-of-turn state:
target = player HP when the fight ends, or DEATH_VALUE if the player dies.
"""

import os

# Workers already run in parallel; one predict call must not grab every core
# (HistGradientBoosting uses OpenMP: 6 workers x 10 threads pushed the load to ~40).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pickle

from combat import alive, incoming_damage, potions, powers
from policy import card_key, stat_value

DEATH_VALUE = -40.0
BOSSES = ("Ceremonial Beast", "Vantom", "Kin Priest")
SCALING = ("Strength", "Ritual")
FEATURES = [
    "is_boss", "is_elite", "act", "round",
    "hp", "max_hp", "hp_frac", "block", "energy_left", "strength", "dexterity",
    "p_vuln", "p_weak", "p_frail", "p_other_powers", "potions",
    "incoming", "unblocked", "hp_after",
    "n_enemies", "enemy_hp", "enemy_max_hp", "enemy_hp_max_single", "enemy_block",
    "enemy_strength", "enemy_scaling", "enemy_vuln", "enemy_weak", "n_attacking",
    "hand", "draw", "discard", "deck", "deck_attacks", "deck_skills", "deck_powers",
    "deck_basic", "deck_value", "deck_dmg", "deck_block",
] + [f"boss_{b.split()[0].lower()}" for b in BOSSES]


def features(st):
    ctx = st.get("context") or {}
    p = st.get("player") or {}
    room = ctx.get("room_type")
    hp, max_hp, block = p.get("hp") or 0, p.get("max_hp") or 1, p.get("block") or 0
    pp = powers(st.get("player_powers"))
    inc = incoming_damage(st)
    en = alive(st)
    epw = [powers(e.get("powers")) for e in en]
    deck = p.get("deck") or []
    names = {e.get("name") for e in en}
    f = {
        "is_boss": float(room == "Boss"), "is_elite": float(room == "Elite"),
        "act": ctx.get("act") or 1, "round": st.get("round") or 0,
        "hp": hp, "max_hp": max_hp, "hp_frac": hp / max(1, max_hp), "block": block,
        "energy_left": st.get("energy") or 0,
        "strength": pp.get("Strength", 0), "dexterity": pp.get("Dexterity", 0),
        "p_vuln": pp.get("Vulnerable", 0), "p_weak": pp.get("Weak", 0), "p_frail": pp.get("Frail", 0),
        "p_other_powers": sum(1 for k in pp if k not in ("Strength", "Dexterity", "Vulnerable", "Weak", "Frail")),
        "potions": len(potions(st)),
        "incoming": inc, "unblocked": max(0, inc - block), "hp_after": hp - max(0, inc - block),
        "n_enemies": len(en), "enemy_hp": sum(e.get("hp") or 0 for e in en),
        "enemy_max_hp": sum(e.get("max_hp") or 0 for e in en),
        "enemy_hp_max_single": max([e.get("hp") or 0 for e in en] or [0]),
        "enemy_block": sum(e.get("block") or 0 for e in en),
        "enemy_strength": sum(w.get("Strength", 0) for w in epw),
        "enemy_scaling": sum(1 for w in epw if any(k in w for k in SCALING)),
        "enemy_vuln": sum(w.get("Vulnerable", 0) for w in epw), "enemy_weak": sum(w.get("Weak", 0) for w in epw),
        "n_attacking": sum(1 for e in en if e.get("intends_attack")),
        "hand": len(st.get("hand") or []), "draw": st.get("draw_pile_count") or 0,
        "discard": st.get("discard_pile_count") or 0, "deck": len(deck),
        "deck_attacks": sum(1 for c in deck if c.get("type") == "Attack"),
        "deck_skills": sum(1 for c in deck if c.get("type") == "Skill"),
        "deck_powers": sum(1 for c in deck if c.get("type") == "Power"),
        "deck_basic": sum(1 for c in deck if card_key(c).startswith(("STRIKE_", "DEFEND_"))),
        "deck_value": sum(stat_value(c) for c in deck) / max(1, len(deck)),
        "deck_dmg": sum((c.get("stats") or {}).get("damage") or 0 for c in deck) / max(1, len(deck)),
        "deck_block": sum((c.get("stats") or {}).get("block") or 0 for c in deck) / max(1, len(deck)),
    }
    for b in BOSSES:
        f[f"boss_{b.split()[0].lower()}"] = float(b in names)
    return [float(f[k]) for k in FEATURES]


class ValueModel:
    def __init__(self, path):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        self.model, self.features = blob["model"], blob["features"]
        assert self.features == FEATURES, "feature list changed since this model was trained"

    def predict(self, states):
        if not states:
            return []
        return list(self.model.predict([features(s) for s in states]))


def default_model_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "value.pkl")
