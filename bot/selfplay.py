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
# Calibrated fast world: the policy network loses more HP than the search planner, so
# strategy learned with it overvalues healing and block. With CALIBRATE, hallway and elite
# losses are refunded down to the planner's (A10 bench decks: planner/net loss 0.8 hallway,
# 0.87 elite) and boss fights start with BOSS_BONUS extra HP (net 52/108 boss wins at +20
# vs the planner's 54/108).
CALIBRATE = False
LOSS_SCALE = {"Monster": 0.8, "Elite": 0.87}
BOSS_BONUS = 20
# Act 2+ (A10 act-2 bench, policy_rl105 vs the planner): the network also dies far more
# often mid-fight in hallways (23/170 vs 11/175), which refunds afterwards cannot undo. So
# from act 2 every hallway / elite fight starts with a buffer of FIGHT_BUFFER HP (net then
# wins 163/170, planner 164/175) and the HP lost is scaled to the planner's.
FIGHT_BUFFER = {"Monster": 15, "Elite": 15}
LOSS_SCALE_LATE = {"Monster": 0.75, "Elite": 0.82}


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


def play_fast_run(sim, seed, pilot, ascension, acts, rec, temperature, rng, card_pol=None, card_temp=0.0,
                  rest_pol=None, rest_temp=0.0, event_pol=None, event_temp=0.0, map_pol=None, map_temp=0.0,
                  record_map=False, shop_pol=None, shop_temp=0.0, record_shop=False):
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
            fight_room, fight_act, buffer = room, (ctx.get("act") or 1), 0
            if CALIBRATE:
                buffer = BOSS_BONUS if room == "Boss" else FIGHT_BUFFER.get(room, 0) if fight_act >= 2 else 0
                if buffer:
                    st = dict(st, player=dict(st.get("player") or {}, hp=(hp0 or 0) + buffer))
                    sim.send({"cmd": "set_player", "hp": (hp0 or 0) + buffer})
            rec.fight_start(st)
            fights += 1
        if in_fight and not combat:
            hp_last = (last or st).get("player", {}).get("hp")
            rec.fight_end(st, hp0, True, (last or {}).get("round"), hp_last)
            in_fight = False
            if CALIBRATE and st.get("decision") != "game_over":
                p = st.get("player") or {}
                cur, mhp = p.get("hp") or 0, p.get("max_hp") or 80
                new = cur
                scale = (LOSS_SCALE if fight_act < 2 else LOSS_SCALE_LATE).get(fight_room)
                if scale is not None and hp0 is not None and hp_last is not None:
                    # HP before the fight, minus the scaled loss (buffer included), plus
                    # end-of-combat heals (relics) seen after the fight
                    new = hp0 - scale * max(0, hp0 + buffer - hp_last) + max(0, cur - hp_last)
                new = int(max(1, min(mhp, new)))
                if new != cur:
                    sim.send({"cmd": "set_player", "hp": new})
                    st = dict(st, player=dict(p, hp=new))
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
        extra = None
        if d == "card_reward" and card_pol is not None and card_temp > 0 and st.get("cards"):
            cacts, z = card_pol.scores(st)
            z = (z - z.max()) / card_temp
            pr = np.exp(z)
            pr /= pr.sum()
            i = rng.choices(range(len(cacts)), weights=pr.tolist())[0]
            a, extra = cacts[i], {"lp": float(math.log(max(pr[i], 1e-12))), "T": card_temp}
        elif d == "rest_site" and rest_pol is not None:
            keys, ks, z, route = rest_pol.scores(sim, st)
            if len(ks) >= 1:
                zz = (z - z.max()) / max(rest_temp, 1e-6)
                pr = np.exp(zz)
                pr /= pr.sum()
                i = rng.choices(range(len(ks)), weights=pr.tolist())[0] if rest_temp > 0 else int(np.argmax(z))
                idx, tok = keys[ks[i]]
                policy._LAST_REST.option = ks[i].split(":")[0]
                policy._REST_PICK.token = tok
                a = ("choose_option", {"option_index": idx})
                extra = {"rlp": float(math.log(max(pr[i], 1e-12))), "rkeys": ks, "rk": ks[i], "route": route}
            else:
                a = policy.decide(sim, st, avoid=(), memo=memo)
        elif d == "shop" and (shop_pol is not None or record_shop):
            from shopnet import shop_options
            visit = memo.setdefault(("shopnet", ctx.get("act"), ctx.get("floor")), {"removed": False, "bought": 0})
            before = [bool(visit["removed"]), visit["bought"]]
            if shop_pol is not None:
                opts, route, z = shop_pol.scores(sim, st, visit)
                zz = (z - z.max()) / max(shop_temp, 1e-6)
                pr = np.exp(zz)
                pr /= pr.sum()
                i = rng.choices(range(len(opts)), weights=pr.tolist())[0] if shop_temp > 0 else int(np.argmax(z))
                a = opts[i][1]
                extra = {"sopts": [o for o, _ in opts], "sk": i, "slp": float(math.log(max(pr[i], 1e-12))),
                         "route": route, "smemo": before}
            else:
                a = policy.decide(sim, st, avoid=(), memo=memo)
                opts = shop_options(st, visit)
                keys = [json.dumps(list(x), sort_keys=True) for _, x in opts]
                k = json.dumps([a[0], a[1]], sort_keys=True)
                if k in keys and len(opts) > 1:
                    import decktest
                    route = [t for t, _ in decktest.route_to_boss(sim, st)]
                    extra = {"sopts": [o for o, _ in opts], "sk": keys.index(k), "route": route, "smemo": before}
            if a[0] == "remove_card":
                visit["removed"] = True
            elif a[0].startswith("buy_"):
                visit["bought"] += 1
        elif d == "map_select" and map_pol is not None and len(st.get("choices") or []) > 1:
            chs, descs, z = map_pol.scores(sim, st)
            zz = (z - z.max()) / max(map_temp, 1e-6)
            pr = np.exp(zz)
            pr /= pr.sum()
            i = rng.choices(range(len(chs)), weights=pr.tolist())[0] if map_temp > 0 else int(np.argmax(z))
            a = ("select_map_node", {"col": chs[i]["col"], "row": chs[i]["row"]})
            extra = {"mopts": descs, "mk": i, "mlp": float(math.log(max(pr[i], 1e-12)))}
        elif d == "event_choice" and event_pol is not None:
            opts, descs, z, route = event_pol.scores(sim, st)
            if z is not None:
                zz = (z - z.max()) / max(event_temp, 1e-6)
                pr = np.exp(zz)
                pr /= pr.sum()
                i = rng.choices(range(len(opts)), weights=pr.tolist())[0] if event_temp > 0 else int(np.argmax(z))
                policy._remember_event(opts[i])
                a = ("choose_option", {"option_index": opts[i]["index"]})
                extra = {"eopts": descs, "ek": i, "elp": float(math.log(max(pr[i], 1e-12))), "route": route}
            else:
                a = policy.decide(sim, st, avoid=(), memo=memo)
        else:
            a = policy.decide(sim, st, avoid=(), memo=memo)
            if d == "map_select" and record_map and len(st.get("choices") or []) > 1:   # DP choice: map imitation data
                from mapnet import option_desc as map_desc
                m = sim.send({"cmd": "get_map"})
                chs = st.get("choices") or []
                keys = [(c["col"], c["row"]) for c in chs]
                k = (a[1].get("col"), a[1].get("row"))
                if k in keys:
                    extra = {"mopts": [map_desc(sim, st, m, c) for c in chs], "mk": keys.index(k)}
            if d == "event_choice":   # rule choice: imitation data for the event network
                from eventnet import option_desc
                opts = [o for o in st.get("options") or [] if not o.get("is_locked")]
                idx = [o["index"] for o in opts]
                if len(opts) > 1 and a[0] == "choose_option" and a[1].get("option_index") in idx:
                    extra = {"eopts": [option_desc(o) for o in opts], "ek": idx.index(a[1]["option_index"])}
        rec.decision(st, legal_actions(st), a, extra)
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
    ap.add_argument("--card-policy", default=None, help="nets.py card model for card rewards (default: tier rules)")
    ap.add_argument("--card-temperature", type=float, default=0.0, help=">0: sample card rewards (deck RL)")
    ap.add_argument("--rest-policy", default=None, help="restnet.py model for rest sites (default: rules)")
    ap.add_argument("--rest-temperature", type=float, default=0.0, help=">0: sample rest choices (RL)")
    ap.add_argument("--event-policy", default=None, help="eventnet.py model for events (default: rules)")
    ap.add_argument("--event-temperature", type=float, default=0.0, help=">0: sample event choices (RL)")
    ap.add_argument("--calibrate", action="store_true", help="refund HP to the planner's losses, boss HP bonus")
    ap.add_argument("--map-policy", default=None, help="mapnet.py model for map choices (default: the DP)")
    ap.add_argument("--map-temperature", type=float, default=0.0)
    ap.add_argument("--record-map", action="store_true", help="record the DP's map choices (imitation data)")
    ap.add_argument("--shop-policy", default=None, help="shopnet.py model for shops (default: rules)")
    ap.add_argument("--shop-temperature", type=float, default=0.0)
    ap.add_argument("--record-shop", action="store_true", help="record the rules' shop choices (imitation data)")
    args = ap.parse_args()
    import random
    import torch
    torch.set_num_threads(1)
    from nets import NetPolicy
    policy.DECK_TEST = False
    global CALIBRATE
    CALIBRATE = args.calibrate
    card_pol = None
    if args.card_policy:
        from nets import CardPolicy
        policy.CARD_NET = card_pol = CardPolicy(args.card_policy)
    rest_pol = None
    if args.rest_policy:
        from restnet import RestPolicy
        rest_pol = RestPolicy(args.rest_policy)
    event_pol = None
    if args.event_policy:
        from eventnet import EventPolicy
        event_pol = EventPolicy(args.event_policy)
    shop_pol = None
    if args.shop_policy:
        from shopnet import ShopPolicy
        shop_pol = ShopPolicy(args.shop_policy)
    map_pol = None
    if args.map_policy:
        from mapnet import MapPolicy
        map_pol = MapPolicy(args.map_policy)
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
                r = play_fast_run(sim, s, pilot, args.ascension, args.acts, rec, args.temperature, rng,
                                  card_pol, args.card_temperature, rest_pol, args.rest_temperature,
                                  event_pol, args.event_temperature, map_pol, args.map_temperature, args.record_map,
                                  shop_pol, args.shop_temperature, args.record_shop)
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
