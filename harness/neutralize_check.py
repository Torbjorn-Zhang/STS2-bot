"""Does sts2-cli's hand-written Neutralize replacement still go through the attack pipeline?

Compare Neutralize (patched OnPlay) against Strike (unpatched) under effects that hook
attack damage: Strength (Vajra), Vulnerable (Bash first), The Boot, and Envenom.
"""

import json

from simclient import Sim

ENC = "SHRINKER_BEETLE_WEAK"


def enemy(st):
    e = st["enemies"][0]
    return e["hp"], e.get("block", 0), {p.get("id") or p.get("name"): p.get("amount") for p in (e.get("powers") or [])}


def play(sim, st, card_id, target=True):
    c = next(c for c in st["hand"] if c["id"] == "CARD." + card_id)
    args = {"card_index": c["index"]}
    if target:
        args["target_index"] = 0
    return sim.act("play_card", **args)


def run_case(test_card, relics, setup_card=None):
    sim = Sim("Debug")
    try:
        sim.send({"cmd": "start_run", "character": "Silent", "seed": "neut"})
        deck = [test_card] + ([setup_card] if setup_card else []) + ["DEFEND_SILENT"] * 3
        sim.send({"cmd": "set_player", "hp": 999, "max_hp": 999, "relics": relics, "deck": deck})
        st = sim.send({"cmd": "enter_room", "type": "combat", "encounter": ENC})
        if setup_card:
            st = play(sim, st, setup_card, target=(setup_card != "ENVENOM"))
        hp0, blk0, pw0 = enemy(st)
        st2 = play(sim, st, test_card)
        if st2.get("type") == "error":
            return {"error": st2.get("message")}
        hp1, blk1, pw1 = enemy(st2)
        return {"damage": (hp0 + blk0) - (hp1 + blk1), "powers_after": pw1}
    finally:
        sim.close()


CASES = [
    ("baseline", [], None),
    ("Vajra +1 Strength", ["VAJRA"], None),
    ("Bash first (Vulnerable)", [], "BASH"),
    ("The Boot (min 5 unblocked)", ["THE_BOOT"], None),
    ("Envenom first", [], "ENVENOM"),
]

for name, relics, setup in CASES:
    n = run_case("NEUTRALIZE", relics, setup)
    s = run_case("STRIKE_SILENT", relics, setup)
    print(f"{name:28s} Neutralize: {json.dumps(n)}\n{'':28s} Strike:     {json.dumps(s)}")
