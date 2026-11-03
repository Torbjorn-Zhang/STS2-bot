"""Whole A10 runs on sts2-rs played by spire-mentor's PPO run policy (every non-combat decision)
with sts2-rs's MCTS for the fights (PpoRunAgent), on our eval seeds: how strong that deck policy
is next to ours on the same engine (run.py --rs plays our nets on it).

    rsppo.py END --seed-start S --tag T --model models/ext/ppo_fullrun_ironclad_cont300_it1150.onnx
             [--features card_mods,reward_upgrade] [--sims 1000 --boss-sims 4000] [--workers W]

The RunManager is built like rssim.RsSim's (Neow start, act 1 rolled, interactive removal / upgrade
/ transform, direct event relic grant, game v0.111.0, seeds hashed raw): spire-mentor's arena setup.
Rows -> runs/T.jsonl: seed, victory, act, floor, hp, deck, and every boss fight (act, hp in/out, won).
PpoRunAgent.step holds the GIL, so the workers are processes.
"""

import argparse
import json
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

from rssim import sts2_rs  # noqa: E402  (rssim locates the module and sets STS2_RS_RAW_SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
GAME_VERSION = "v0.111.0"


def g(rm, name):
    v = getattr(rm, name)
    return v() if callable(v) else v


def play(job):
    seed, asc, model, features, sims, boss_sims = job
    t0 = time.time()
    rm = sts2_rs.RunManager(character="Ironclad", string_seed=seed, ascension_level=asc, neow_start=True, act1="roll",
                            interactive_card_removal=True, interactive_card_upgrade=True,
                            interactive_card_transform=True, direct_event_relic_grant=True, game_version=GAME_VERSION)
    agent = sts2_rs.PpoRunAgent(seed=zlib.crc32(seed.encode()) & 0x7FFFFFFF, sims=sims, boss_sims=boss_sims,
                                model_path=model, features=features or None)
    fights, cur, steps, err = [], None, 0, None
    try:
        while not g(rm, "is_over") and steps < 20000:
            in_combat = g(rm, "phase") == "combat"
            if in_combat and cur is None:
                room = (g(rm, "current_room") or {}).get("room_type")
                cur = {"act": int(g(rm, "act_index") or 1), "floor": int(g(rm, "total_floor")), "room": room,
                       "hp_before": int(g(rm, "current_hp")), "potions_before": len(g(rm, "potions") or [])}
            elif not in_combat and cur is not None:
                cur.update(hp_after=int(g(rm, "current_hp")), potions_left=len(g(rm, "potions") or []),
                           won=int(g(rm, "current_hp")) > 0)
                fights.append(cur)
                cur = None
            if agent.step(rm) is None:
                err = "no action"
                break
            steps += 1
    except Exception as e:  # noqa: BLE001  (a crash ends the run; the row says so)
        err = f"{type(e).__name__}: {e}"[:300]
    if cur is not None:
        cur.update(hp_after=int(g(rm, "current_hp")), potions_left=len(g(rm, "potions") or []),
                   won=bool(g(rm, "player_won")))
        fights.append(cur)
    won = bool(g(rm, "player_won"))
    bosses = [f for f in fights if f["room"] in ("Boss", "boss")]
    return {"seed": seed, "ascension": asc, "victory": won, "died": not won and err is None,
            "act": int(g(rm, "act_index") or 1), "floor": int(g(rm, "total_floor")), "hp": int(g(rm, "current_hp")),
            "max_hp": int(g(rm, "max_hp")), "deck": list(g(rm, "master_deck") or []), "relics": list(g(rm, "relics") or []),
            "bosses_won": sum(1 for f in bosses if f["won"]), "boss_log": bosses, "fights": len(fights),
            "steps": steps, "error": err, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, help="end seed index (exclusive)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-prefix", default="full")
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", default="card_mods,reward_upgrade")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--boss-sims", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    model = args.model if os.path.isabs(args.model) else os.path.join(HERE, args.model)
    jobs = [(f"{args.seed_prefix}_{i}", args.ascension, model, args.features, args.sims, args.boss_sims)
            for i in range(args.seed_start, args.n)]
    out = os.path.join(HERE, "runs", f"{args.tag}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(out, "a") as fh:
        for r in ex.map(play, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
    n = len(rows)
    print(json.dumps({"tag": args.tag, "runs": n, "wins": sum(r["victory"] for r in rows),
                      "act3": sum(1 for r in rows if r["act"] >= 3), "errors": sum(1 for r in rows if r["error"]),
                      "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
