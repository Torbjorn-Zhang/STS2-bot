"""Fast self-play with the policy network (AlphaGo's RL stage and value-net data).

Combat is played by nets.NetPolicy (one forward pass per action, sampled at a
temperature for exploration; the behaviour log-probability of each move is recorded for
the PPO update in nets.py rl). Out-of-combat decisions use the rule policy without the
deck test, so a run takes seconds instead of minutes.

    selfplay.py 300 --policy models/policy.pt --temperature 1.0 --out data/rl_it1.jsonl
"""

import argparse
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import policy
from engine import fresh_sim, legal_actions, maybe_recycle
from record import GameRecorder, combat_state, run_state

HERE = os.path.dirname(os.path.abspath(__file__))
FIGHT_ROOMS = ("Monster", "Elite", "Boss")


def sample(pilot, st, acts, temperature, rng):
    """-> (action, log-prob of it under softmax(z / T))."""
    if len(acts) == 1:
        return acts[0], 0.0
    z = pilot.pilot.scores(combat_state(st), run_state(st), acts) / max(temperature, 1e-6)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    i = rng.choices(range(len(acts)), weights=p.tolist())[0] if temperature > 0 else int(np.argmax(z))
    return acts[i], float(math.log(max(p[i], 1e-12)))


def play_fast_run(sim, seed, pilot, ascension, acts, rec, temperature, rng):
    policy.ASCENSION = ascension
    st = sim.send({"cmd": "start_run", "character": "Ironclad", "seed": seed, "ascension": ascension})
    memo, steps, in_fight, hp0, last = {}, 0, False, None, None
    fights, fight_steps = 0, 0
    while steps < 6000:
        d = st.get("decision")
        ctx = st.get("context") or {}
        if d == "game_over" or (ctx.get("act") or 1) > acts or st.get("type") == "error":
            break
        room = ctx.get("room_type")
        combat = d == "combat_play" or (d == "card_select" and room in FIGHT_ROOMS and in_fight)
        if d == "combat_play" and not in_fight:
            in_fight, hp0, fight_steps = True, (st.get("player") or {}).get("hp"), 0
            rec.fight_start(st)
            fights += 1
        if in_fight and not combat:
            rec.fight_end(st, hp0, True, (last or {}).get("round"), (last or st).get("player", {}).get("hp"))
            in_fight = False
        steps += 1
        if combat:
            if st.get("enemies") is not None:
                last = st
            legal = pilot.actions(st) or legal_actions(st)
            a, lp = sample(pilot, st, legal, temperature, rng)
            fight_steps += 1
            if fight_steps > 400:   # a stalled fight: end the turn until it resolves
                a, lp = ("end_turn", {}), None
            rec.combat_step(st, legal, a, {"lp": lp} if lp is not None else None)
            nxt = sim.act(a[0], **a[1])
            if nxt.get("type") == "error" and d == "combat_play":
                nxt = sim.act("end_turn")
            if nxt.get("type") == "error":
                break
            st = nxt
            continue
        a = policy.decide(sim, st, avoid=(), memo=memo)
        rec.decision(st, legal_actions(st), a, None)
        nxt = sim.act(a[0], **a[1])
        if nxt.get("type") == "error" or json.dumps(nxt, sort_keys=True) == json.dumps(st, sort_keys=True):
            alts = [x for x in legal_actions(st) if x != a]   # no progress: try something else
            if alts:
                nxt = sim.act(alts[-1][0], **alts[-1][1])
        if nxt.get("type") == "error":
            break
        st = nxt
    if in_fight:
        won = st.get("decision") != "game_over"
        rec.fight_end(st, hp0, won, (last or {}).get("round"), (last or st).get("player", {}).get("hp"))
    ctx = st.get("context") or {}
    p = st.get("player") or {}
    res = {"seed": seed, "victory": bool(st.get("victory")), "died": st.get("decision") == "game_over" and not st.get("victory"),
           "cleared_acts": 3 if st.get("victory") else (ctx.get("act") or 1) - 1, "act": ctx.get("act"),
           "floor": ctx.get("floor"), "room": ctx.get("room_type"), "hp": p.get("hp"), "max_hp": p.get("max_hp"),
           "fights": fights, "error": st.get("message") if st.get("type") == "error" else None, "stuck": steps >= 6000}
    rec.finish(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--policy", default=os.path.join(HERE, "models", "policy.pt"))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--acts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed-prefix", default="rl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import random
    import torch
    torch.set_num_threads(1)
    from nets import NetPolicy
    policy.DECK_TEST = False
    pilot = NetPolicy(args.policy)
    seeds = [f"{args.seed_prefix}_{i}" for i in range(args.n)]
    lock = threading.Lock()
    results = []
    t0 = time.time()

    def worker(chunk):
        sim = fresh_sim()
        rng = random.Random(chunk[0] if chunk else 0)
        for s in chunk:
            sim = maybe_recycle(sim)
            rec = GameRecorder(args.out, lock, s, {"tag": "selfplay", "asc": args.ascension, "T": args.temperature,
                                                   "policy": os.path.basename(args.policy)})
            try:
                r = play_fast_run(sim, s, pilot, args.ascension, args.acts, rec, args.temperature, rng)
            except Exception as e:
                r = {"seed": s, "error": repr(e)[:200]}
                try:
                    sim.close()
                except Exception:
                    pass
                sim = fresh_sim()
            with lock:
                results.append(r)
        sim.close()

    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(worker, chunks))
    ok = [r for r in results if not r.get("error")]
    print(json.dumps({"runs": len(results), "errors": len(results) - len(ok),
                      "act1_clears": sum(1 for r in ok if (r.get("cleared_acts") or 0) >= 1),
                      "act2_clears": sum(1 for r in ok if (r.get("cleared_acts") or 0) >= 2),
                      "wins": sum(1 for r in ok if r.get("victory")),
                      "mean_floor": round(sum(((r.get("act") or 1) - 1) * 17 + (r.get("floor") or 0) for r in ok) / max(1, len(ok)), 2),
                      "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
