"""Vine data for card picks: every option of a card reward played forward by the real engine.

Deck-model rollouts imagined the hallway fights and chose slow cards that died in act 1. Here the
near future is real. At a sampled card reward (acts 1-2) the state is snapshotted, and each option
(each card, or skipping) is taken and the run is played on by the bot itself (planner combat, a
cheaper budget) to the start of the next act, K times with the same K hidden futures for every
option (determinize_run seeds: common random numbers). A future that dies scores 0. One that
survives scores the deck model's P(win every boss left) for the deck and HP it holds at that act
start (the act boss of that act known, act-3 bosses averaged). The far future comes from the model;
surviving the act and the HP it costs come from the engine. Rows feed a card-net update toward the
options that score best (AlphaGo: search results train the policy).

run.py --vine-out F.jsonl [--vine-p 0.25 --vine-k 3 --vine-nodes 400] records rows while playing.
"""
import json
import threading
import time
import zlib

import numpy as np

CFG = None   # set by run.py: {"out", "p", "k", "acts", "nodes", "rollouts", "models"}
_LOCK = threading.Lock()
_MODELS = []
BOSSES3 = ("AEONGLASS_BOSS", "QUEEN_BOSS", "TEST_SUBJECT_BOSS")


def _models():
    if not _MODELS:
        import os
        from deckvalue import DeckValue
        here = os.path.dirname(os.path.abspath(__file__))
        _MODELS.extend(DeckValue(os.path.join(here, p)) for p in CFG["models"])
    return _MODELS


def horizon_value(end, from_act):
    """0 if the future died (or stalled); else P(win every boss left) by the deck model for the
    deck, HP, relics and potions the run holds at the next act's start."""
    from policy import potion_id
    from record import card_tok
    if end.get("victory"):
        return 1.0
    ctx = end.get("context") or {}
    act = ctx.get("act") or from_act
    if end.get("decision") == "game_over" or end.get("type") == "error" or act <= from_act:
        return 0.0
    p = end.get("player") or {}
    base = {"deck": [card_tok(c) for c in p.get("deck") or []], "relics": [r.get("name") for r in p.get("relics") or []],
            "potions": [potion_id(x) or x.get("name") for x in p.get("potions") or [] if x],
            "hp": p.get("hp"), "mhp": p.get("max_hp")}
    states = []
    if act <= 2:   # act 2 start: its boss (known) at this HP, then act 3 at full HP (averaged over the bosses)
        states.append([dict(base, act=2, boss=(ctx.get("boss") or {}).get("id"))])
        states.append([dict(base, act=3, boss=b, hp=base["mhp"]) for b in BOSSES3])
    else:
        b = (ctx.get("boss") or {}).get("id")
        states.append([dict(base, act=3, boss=b)] if b else [dict(base, act=3, boss=x) for x in BOSSES3])
    v = 1.0
    for group in states:
        v *= float(np.mean([np.mean(m.win(group)) for m in _models()]))
    return v


def maybe_vine(sim, st, seed, rng):
    """At a card reward: with probability p, value every option by K real futures; append a row."""
    from combat import TurnPlanner
    from engine import legal_actions
    from run import continue_run
    ctx = st.get("context") or {}
    act = ctx.get("act") or 1
    if CFG is None or act not in CFG["acts"] or rng.random() >= CFG["p"]:
        return
    opts = [a for a in legal_actions(st) if a[0] in ("select_card_reward", "skip_card_reward")]
    if len(opts) < 2:
        return
    t0 = time.perf_counter()
    snap = sim.send({"cmd": "snapshot"})["id"]
    vals = {}
    try:
        for a in opts:
            key = json.dumps(a, sort_keys=True)
            vals[key] = []
            for j in range(CFG["k"]):
                sim.send({"cmd": "restore", "id": snap})
                sim.send({"cmd": "determinize_run",
                          "seed": zlib.crc32(f"{seed}/vine/{act}/{ctx.get('floor')}/{j}".encode()) * 2654435761 % (2 ** 62)})
                s = sim.act(a[0], **a[1])
                if s.get("type") == "error":
                    break
                planner = TurnPlanner(sim, max_nodes=CFG["nodes"], rollouts=CFG["rollouts"])
                end = continue_run(sim, s, planner, act)
                vals[key].append(horizon_value(end, act))
    finally:
        sim.send({"cmd": "restore", "id": snap})
        sim.send({"cmd": "drop_snapshots", "from": snap})
    row = {"seed": seed, "act": act, "floor": ctx.get("floor"), "state": st, "values": vals,
           "secs": round(time.perf_counter() - t0, 1)}
    with _LOCK, open(CFG["out"], "a") as fh:
        fh.write(json.dumps(row) + "\n")
