"""spire-mentor's PPO run policy improved by rollouts: whole A10 runs on sts2-rs played like rsppo.py
(PPO picks every non-combat decision, sts2-rs's MCTS the fights), except at the decisions where the
PPO is unsure (its top probability < --pmax): its --m most likely options (each >= --pmin) are tried
on clones of the run, each played out --k times by a cheaper copy of the same agent (--roll-sims),
and the PPO's pick is left only for an option that beats it on the paired playouts by more than --z
standard errors (one-step rollout policy improvement).

    rsroll.py END --seed-start S --tag T --model models/ext/ppo_fullrun_ironclad_cont300_it1150.onnx
              [--k 4 --m 3 --pmax 0.85 --pmin 0.03] [--roll-sims 100,400] [--sims 1000 --boss-sims 4000]
              [--horizon run|act] [--workers W]

Playouts are paired across options (playout j of every option reseeds its clone with the same
seed, RunManager.reseed_run: same future rolls, none of them the real run's) and score
2 x victory + acts cleared + floor / 20 at the end of the run (--horizon run) or of the current
act (--horizon act: reaching the next act scores like clearing it).
Rows -> runs/T.jsonl as rsppo.py's plus "searched": one entry per searched decision (the PPO's
probabilities over the tried options, their mean scores, the pick).
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
MAX_STEPS = 20000

_AGENTS = {}   # per worker process: (model, features, sims, boss_sims, seed) -> PpoRunAgent


def g(rm, name):
    v = getattr(rm, name)
    return v() if callable(v) else v


def agent(model, features, sims, boss_sims, seed):
    k = (model, features, sims, boss_sims, seed)
    if k not in _AGENTS:
        _AGENTS[k] = sts2_rs.PpoRunAgent(seed=seed, sims=sims, boss_sims=boss_sims, model_path=model,
                                         features=features or None)
    return _AGENTS[k]


def score(rm, act0, horizon):
    act, floor = int(g(rm, "act_index") or 1), int(g(rm, "act_floor") or 0)
    if bool(g(rm, "player_won")):
        return 2.0 + 3.0
    if horizon == "act" and act > act0 and not g(rm, "is_over"):
        return float(act - 1)
    return float(act - 1) + floor / 20.0


def playout(rm, ag, horizon):
    act0, steps = int(g(rm, "act_index") or 1), 0
    while not g(rm, "is_over") and steps < MAX_STEPS:
        if horizon == "act" and int(g(rm, "act_index") or 1) > act0:
            break
        if ag.step(rm) is None:
            break
        steps += 1
    return score(rm, act0, horizon)


def search(rm, acts, probs, cfg, roll, salt):
    """-> (index into acts, the searched entry) for one decision: the PPO's pick unless another tried
    option beats it on the paired playouts by more than z standard errors (the best such one)."""
    order = sorted(range(len(acts)), key=lambda i: -probs[i])
    cand = [i for i in order[:cfg["m"]] if probs[i] >= cfg["pmin"]]
    sc = {}
    for i in cand:
        sc[i] = []
        for j in range(cfg["k"]):
            cp = rm.clone()
            cp.reseed_run((salt * 1000003 + j * 7919) & 0x7FFFFFFF)
            cp.take_action(acts[i])
            sc[i].append(playout(cp, roll, cfg["horizon"]))
    a, k = order[0], cfg["k"]
    best, lcb_best = a, 0.0
    for i in cand[1:]:
        d = [x - y for x, y in zip(sc[i], sc[a])]
        mu = sum(d) / k
        sd = (sum((x - mu) ** 2 for x in d) / max(k - 1, 1)) ** 0.5
        lcb = mu - cfg["z"] * sd / k ** 0.5
        if lcb > lcb_best:
            best, lcb_best = i, lcb
    return best, {"floor": int(g(rm, "total_floor")), "act": int(g(rm, "act_index") or 1),
                  "kind": acts[a].get("action"), "p": [round(probs[i], 3) for i in cand],
                  "q": [round(sum(sc[i]) / k, 3) for i in cand], "pick": cand.index(best)}


def play(job):
    seed, asc, model, features, sims, boss_sims, cfg = job
    t0 = time.time()
    rm = sts2_rs.RunManager(character="Ironclad", string_seed=seed, ascension_level=asc, neow_start=True, act1="roll",
                            interactive_card_removal=True, interactive_card_upgrade=True,
                            interactive_card_transform=True, direct_event_relic_grant=True, game_version=GAME_VERSION)
    s0 = zlib.crc32(seed.encode()) & 0x7FFFFFFF
    main = sts2_rs.PpoRunAgent(seed=s0, sims=sims, boss_sims=boss_sims, model_path=model, features=features or None)
    roll = agent(model, features, cfg["roll_sims"], cfg["roll_boss_sims"], 12345)
    fights, cur, steps, err, searched, decisions = [], None, 0, None, [], 0
    try:
        while not g(rm, "is_over") and steps < MAX_STEPS:
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
            res = None if in_combat else main.evaluate_with_actions(rm)
            if res is not None and len(res[0]) >= 2:
                acts, probs, _value, argmax = res
                probs = [float(p) for p in probs]
                decisions += 1
                if max(probs) < cfg["pmax"] and sum(p >= cfg["pmin"] for p in probs) >= 2 \
                        and len(searched) < cfg["max_searches"]:
                    i, entry = search(rm, acts, probs, cfg, roll, s0 + 7 * decisions)
                    entry["changed"] = i != argmax
                    searched.append(entry)
                else:
                    i = argmax
                rm.take_action(acts[i])
            elif main.step(rm) is None:
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
            "steps": steps, "error": err, "decisions": decisions, "searched": searched,
            "changed": sum(e["changed"] for e in searched), "seconds": round(time.time() - t0, 1)}


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
    ap.add_argument("--k", type=int, default=4, help="playouts per tried option")
    ap.add_argument("--m", type=int, default=3, help="options tried per searched decision (the PPO's most likely)")
    ap.add_argument("--pmax", type=float, default=0.85, help="search only when the PPO's top probability is below this")
    ap.add_argument("--pmin", type=float, default=0.03, help="try only options the PPO gives at least this")
    ap.add_argument("--z", type=float, default=1.0, help="leave the PPO's pick only for an option whose paired "
                    "playout gain is over this many standard errors")
    ap.add_argument("--roll-sims", default="100,400", help="the playouts' combat budget: SIMS,BOSS_SIMS")
    ap.add_argument("--horizon", default="run", choices=["run", "act"])
    ap.add_argument("--max-searches", type=int, default=1000, help="at most this many searched decisions per run")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    model = args.model if os.path.isabs(args.model) else os.path.join(HERE, args.model)
    rs, rb = (int(x) for x in args.roll_sims.split(","))
    cfg = {"k": args.k, "m": args.m, "pmax": args.pmax, "pmin": args.pmin, "z": args.z, "roll_sims": rs,
           "roll_boss_sims": rb, "horizon": args.horizon, "max_searches": args.max_searches}
    jobs = [(f"{args.seed_prefix}_{i}", args.ascension, model, args.features, args.sims, args.boss_sims, cfg)
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
                      "searched": sum(len(r["searched"]) for r in rows), "changed": sum(r["changed"] for r in rows),
                      "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
