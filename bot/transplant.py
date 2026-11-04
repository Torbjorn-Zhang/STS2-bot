"""Replay our engine's elite / boss fights inside sts2-rs: which engine's fight is harder?

A run.py --rs-fight --rs-dump-wires eval keeps each elite / boss fight's first combat_wire in its
fight-log entry ("wire0"). Here every such fight is rebuilt as rsfight.RsFighter builds its host
(the same import, the last boss on its map node) and played to its end by sts2-rs's own
MctsCombatAgent, K times (other agent seeds), the whole fight inside sts2-rs as in an sts2-rs run
(rssim.RsSim). Against what the fight did in our engine, on the same entry board:
  sts2-rs much better -> the fight's gap is in combat (sts2-rs's enemies / cards, or what
  rsfight's per-decision import loses); about equal -> it is in the run (decks, maps, events).

    transplant.py TAG --extract      # runs/TAG.jsonl -> runs/TAG_w0.jsonl.gz (the fights with a wire)
    transplant.py TAG [--k 4] [--sims 1000 --boss-sims 4000] [--rooms Boss] [--start I --end J] [--reuse] [--out OUT] [--workers W]

A --rs --rs-dump-wires eval dumps sts2-rs's own boards ("state"): replaying those is the null control
(this host vs the fight inside the sts2-rs run).

Rows -> runs/OUT.jsonl (default TAG_tp), one per fight: the run seed, act, floor, room, enemies,
ours (hp_before, hp_after, won) and rs: [{hp, won, steps, error}] per replay.
"""

import argparse
import gzip
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))
from rsfight import GAME_VERSION, TIER, last_boss_node, want_override  # noqa: E402
from rssim import sts2_rs  # noqa: E402
from sts2_hint import adapter  # noqa: E402


def g(rm, name):
    v = getattr(rm, name)
    return v() if callable(v) else v


def host(w0):
    """The fight's entry board as a fresh sts2-rs run in it (RsFighter._host)."""
    ctx = w0["ctx"]
    asc = int(ctx.get("ascension") or 0)
    if "state" in w0:   # an sts2-rs run's own board (run.py --rs --rs-dump-wires): the null control
        cs = sts2_rs.state_from_dict(w0["state"])
    else:
        wire = {**w0["wire"], "game_version": GAME_VERSION}
        cs = sts2_rs.state_from_dict(adapter.translate(wire, character="Ironclad", ascension=asc))
    rm = sts2_rs.RunManager(seed=0, character="Ironclad", ascension_level=asc, game_version=GAME_VERSION)
    if w0.get("last_boss"):
        col, row = last_boss_node(asc)
        rm.debug_place_on_map(3, col, row, [], int(ctx.get("floor") or 0), 50)
    rm.debug_enter_combat(cs, TIER.get(ctx.get("room_type"), "monster"))
    return rm


def replay(job):
    w0, k, sims, boss_sims = job
    out = []
    for j in range(k):
        try:
            rm = host(w0)
        except Exception as e:  # noqa: BLE001  (an import the adapter refuses: the row says so)
            return [{"error": f"import {type(e).__name__}: {str(e)[:160]}"}]
        seed = zlib.crc32(f"{w0['ctx'].get('floor')}:{j}".encode()) & 0x7FFFFFFF
        agent = sts2_rs.MctsCombatAgent(seed=seed, sims=sims, boss_sims=boss_sims)
        steps, err = 0, None
        try:
            while steps < 4000 and not g(rm, "is_over") and g(rm, "phase") == "combat":
                if agent.step(rm) is None:
                    err = "no action"
                    break
                steps += 1
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:160]}"
        hp = int(g(rm, "current_hp"))
        out.append({"hp": hp, "won": hp > 0 and (g(rm, "phase") != "combat" or bool(g(rm, "player_won"))),
                    "steps": steps, **({"error": err} if err else {})})
    return out


def extract(tag):
    """runs/TAG.jsonl's dumped fights -> runs/TAG_w0.jsonl.gz, one per line (the replay tasks' input)."""
    out, n = os.path.join(HERE, "runs", f"{tag}_w0.jsonl.gz"), 0
    with open(os.path.join(HERE, "runs", f"{tag}.jsonl")) as fh, gzip.open(out, "wt") as gz:
        for line in fh:
            r = json.loads(line)
            for f in r.get("fight_log") or []:
                if f.get("wire0"):
                    gz.write(json.dumps({"seed": r["seed"], **f}) + "\n")
                    n += 1
    print(json.dumps({"tag": tag, "fights": n, "out": out}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="runs/TAG.jsonl: a --rs-dump-wires eval")
    ap.add_argument("--extract", action="store_true", help="only write runs/TAG_w0.jsonl.gz (the fights with a wire)")
    ap.add_argument("--k", type=int, default=4, help="replays per fight")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--boss-sims", type=int, default=4000)
    ap.add_argument("--rooms", default=None, help="only these room types (e.g. Boss); --start/--end index the kept fights")
    ap.add_argument("--start", type=int, default=0, help="first fight of runs/TAG_w0.jsonl.gz")
    ap.add_argument("--end", type=int, default=None, help="end fight, exclusive")
    ap.add_argument("--reuse", action="store_true", help="sts2-rs's tree reuse in the replays (as --rs-reuse)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.extract:
        return extract(args.tag)
    if args.reuse:
        want_override("tree_reuse=1")
    with gzip.open(os.path.join(HERE, "runs", f"{args.tag}_w0.jsonl.gz"), "rt") as fh:
        fights = [json.loads(line) for line in fh]
    if args.rooms:
        fights = [f for f in fights if f["wire0"]["ctx"].get("room_type") in args.rooms.split(",")]
    fights = fights[args.start:args.end]
    jobs = [(f["wire0"], args.k, args.sims, args.boss_sims) for f in fights]
    path = os.path.join(HERE, "runs", f"{args.out or args.tag + '_tp'}.jsonl")
    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(path, "a") as fh:
        for f, rs in zip(fights, ex.map(replay, jobs)):
            ctx = f["wire0"]["ctx"]
            fh.write(json.dumps({"seed": f["seed"], "act": ctx.get("act"), "floor": f.get("floor"), "room": f.get("room"),
                                 "enemies": f.get("enemies"), "last_boss": f["wire0"].get("last_boss"),
                                 "ours": {"hp_before": f.get("hp_before"), "hp_after": f.get("hp_after"),
                                          "won": f.get("won")}, "rs": rs}) + "\n")
            fh.flush()
            n += 1
    print(json.dumps({"tag": args.tag, "fights": n, "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
