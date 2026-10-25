"""Replay a recorded bot run (bot/run.py --record rows) in the engine, action by action.

    replay_rec.py REC.jsonl SEED [--asc 10] [--config Fix2] [--dump STATES.jsonl]

The record holds every action the bot executed: "d" rows (decisions outside combat) and "c"
rows (each combat action, card selections in combat included), in order. The engine is
deterministic, so the same seed and the same actions give the same run; --dump writes the
engine state before every action (the reference for replaying the run in the real game).
"""

import argparse
import json

from simclient import Sim


def recorded_actions(path, seed):
    out = []
    for line in open(path):
        r = json.loads(line)
        if r.get("seed") == seed and r.get("k") in ("d", "c"):
            out.append((r["k"], r["played"][0], r["played"][1] or {}))
    return out


def start(sim, seed, asc, character="Ironclad"):
    return sim.send({"cmd": "start_run", "character": character, "seed": seed, "ascension": asc})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("seed")
    ap.add_argument("--asc", type=int, default=10)
    ap.add_argument("--config", default="Fix2")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()
    acts = recorded_actions(args.rec, args.seed)
    sim = Sim(args.config, timeout=60)
    st = start(sim, args.seed, args.asc)
    dump = open(args.dump, "w") if args.dump else None
    for i, (kind, a, kw) in enumerate(acts):
        if dump:
            dump.write(json.dumps({"i": i, "k": kind, "state": st, "action": [a, kw]}) + "\n")
        nxt = sim.act(a, **kw)
        if nxt.get("type") == "error":
            print(json.dumps({"error_at": i, "action": [a, kw], "message": nxt.get("message"),
                              "decision": st.get("decision")}))
            break
        st = nxt
    ctx, p = st.get("context") or {}, st.get("player") or {}
    print(json.dumps({"actions": len(acts), "decision": st.get("decision"), "victory": st.get("victory"),
                      "act": ctx.get("act"), "floor": ctx.get("floor"), "hp": p.get("hp"), "max_hp": p.get("max_hp")}))
    sim.close()


if __name__ == "__main__":
    main()
