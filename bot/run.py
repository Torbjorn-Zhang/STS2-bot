"""Play seeds with the search bot and report how far it gets.

usage: run.py N [--acts 1] [--workers 6] [--character Ironclad] [--nodes 400] [--tag name]
Writes one JSON line per run to bot/runs/<tag>.jsonl and prints a summary.
"""

import argparse
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import policy
from combat import TurnPlanner, potion_actions
from engine import fresh_sim, legal_actions, maybe_recycle

HERE = os.path.dirname(os.path.abspath(__file__))


def _key(st):
    return json.dumps(st, sort_keys=True)


def play_run(sim, seed, character, acts, max_nodes, model=None, value_weight=None, ascension=0, rollouts=None,
             rec=None, explore=0.0, decknets=None, combat_records=True):
    """rec: optional record.GameRecorder (self-play data for the policy and value nets).
    explore: chance to take a uniformly random card reward option instead (value-net data
    needs decks the policy would not build; such decisions are flagged in the record)."""
    import random
    rng = random.Random(f"explore/{seed}")
    policy.ASCENSION = ascension
    planner = TurnPlanner(sim, max_nodes=max_nodes, value=model, value_weight=value_weight, rollouts=rollouts)
    st = sim.send({"cmd": "start_run", "character": character, "seed": seed, "ascension": ascension})
    t0, steps, fights, hp_log = time.perf_counter(), 0, 0, []
    dead_ends, loops, memo, stuck = {}, 0, {}, False
    boss_dmg = 0.0   # sum over boss fights of the share of boss HP removed (a win = 1), as selfplay.py
    while steps < 5000:
        d = st.get("decision")
        act = (st.get("context") or {}).get("act") or 1
        if d == "game_over" or act > acts or st.get("type") == "error":
            break
        room = (st.get("context") or {}).get("room_type")
        if d == "combat_play" or (d == "card_select" and room in ("Monster", "Elite", "Boss")):
            hp0 = (st.get("player") or {}).get("hp")
            pots0 = len([x for x in (st.get("player") or {}).get("potions") or [] if x])
            deck0 = (st.get("player") or {}).get("deck") or []
            deck_stats = {"size": len(deck0), "upgraded": sum(1 for c in deck0 if c.get("upgraded")),
                          "basics": sum(1 for c in deck0 if any(b in (c.get("id") or "") for b in ("STRIKE_", "DEFEND_"))),
                          "relics": len((st.get("player") or {}).get("relics") or [])} if room == "Boss" else None
            names = [e.get("name") for e in st.get("enemies") or []]
            enemy_max = sum((e.get("max_hp") or 0) for e in st.get("enemies") or [])
            floor0 = (st.get("context") or {}).get("floor")
            last_enemies = {}
            def watch(s):  # remember the last enemy HP we saw (for deaths)
                if s.get("enemies") is not None:
                    last_enemies["hp"] = [e.get("hp") for e in s["enemies"]]
                    last_enemies["round"] = s.get("round")
                    last_enemies["potions"] = len([x for x in (s.get("player") or {}).get("potions") or [] if x])
                    last_enemies["php"] = (s.get("player") or {}).get("hp")
            on_step = None
            if rec:
                rec.fight_start(st)
            if rec and combat_records:
                def on_step(s, a):
                    extra = None
                    if s is getattr(planner, "last_root", None):   # first action of a fresh plan
                        extra = {k: {kk: round(v, 2) for kk, v in d.items()} for k, d in planner.last_first.items()}
                    rec.combat_step(s, legal_actions(s) + potion_actions(s) if s.get("decision") == "combat_play"
                                    else legal_actions(s), a, extra)
            st = planner.play_combat(st, on_state=watch, on_step=on_step)
            if rec:
                rec.fight_end(st, hp0, st.get("decision") != "game_over", last_enemies.get("round"),
                              last_enemies.get("php"))
            fights += 1
            if room == "Boss":
                won = st.get("decision") != "game_over" or bool(st.get("victory"))
                left = sum(h or 0 for h in last_enemies.get("hp") or [])
                boss_dmg += 1.0 if won else max(0.0, 1 - left / max(1, enemy_max))
            hp_log.append({"floor": floor0, "room": room, "enemies": names, "hp_before": hp0,
                           "hp_after": (st.get("player") or {}).get("hp"), "won": st.get("decision") != "game_over",
                           "enemy_hp_left": last_enemies.get("hp"), "rounds": last_enemies.get("round"),
                           "potions_before": pots0, "potions_left": last_enemies.get("potions"),
                           **({"deck": deck_stats} if deck_stats else {})})
            if st.get("decision") == "combat_play":
                stuck = True   # play_combat hit its action cap: give up on the run
                break
            continue
        a, extra = (decknets.decide(sim, st) if decknets is not None else (None, None))
        if a is None or json.dumps(a, sort_keys=True) in dead_ends.get(_key(st), ()):
            a = policy.decide(sim, st, avoid=dead_ends.get(_key(st), ()), memo=memo)
            extra = policy.pop_last_scores()
        if explore and d == "card_reward" and rng.random() < explore:
            a = rng.choice(legal_actions(st))
            extra = {**(extra or {}), "explore": True}
        if rec:
            rec.decision(st, legal_actions(st), a, extra)
        nxt = sim.act(a[0], **a[1])
        steps += 1
        if nxt.get("type") != "error" and _key(nxt) == _key(st):
            # no progress (e.g. an event option that starts an event fight, which the engine
            # does not support yet): remember it and pick something else next time
            dead_ends.setdefault(_key(st), set()).add(json.dumps(a, sort_keys=True))
            loops += 1
        if nxt.get("type") == "error":
            dead_ends.setdefault(_key(st), set()).add(json.dumps(a, sort_keys=True))
            continue
        st = nxt
    ctx = st.get("context") or {}
    p = st.get("player") or {}
    res = {
        "seed": seed, "character": character, "ascension": ascension,
        "cleared_acts": 3 if st.get("victory") else (ctx.get("act") or 1) - 1,
        "victory": bool(st.get("victory")),
        "died": st.get("decision") == "game_over" and not st.get("victory"),
        "act": ctx.get("act"), "floor": ctx.get("floor"), "room": ctx.get("room_type"),
        "hp": p.get("hp"), "max_hp": p.get("max_hp"), "deck_size": p.get("deck_size"),
        "fights": fights, "seconds": round(time.perf_counter() - t0, 1),
        "boss_dmg": round(boss_dmg, 3), "planner": dict(planner.stats), "dead_end_actions": loops, "fight_log": hp_log, "stuck": stuck, "error": st.get("message") if st.get("type") == "error" else None,
    }
    if rec:
        rec.finish(res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--acts", type=int, default=1)
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--character", default="Ironclad")
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--tag", default="act1")
    ap.add_argument("--seed-prefix", default="bot")
    ap.add_argument("--seed-start", type=int, default=0, help="run seeds prefix_start .. prefix_(n-1) (split work across machines)")
    ap.add_argument("--model", default=None, help="value model (models/value.pkl); default: hand-written evaluation")
    ap.add_argument("--rollouts", default=None, help='JSON rollout config, e.g. {"k":4,"m":3,"turns":2}')
    ap.add_argument("--no-deck-test", action="store_true", help="pick card rewards by the tier list only")
    ap.add_argument("--record", default=None, help="append self-play records to this .jsonl (record.py)")
    ap.add_argument("--no-combat-records", action="store_true", help="--record without combat steps (deck RL)")
    ap.add_argument("--explore", type=float, default=0.0, help="chance of a random card reward pick (data collection)")
    ap.add_argument("--net-rollouts", default=None, help="policy network (nets.py) as the planner's rollout policy")
    ap.add_argument("--net-decktest", default=None, help="policy network as the deck test's pilot")
    ap.add_argument("--run-value", default=None, help="run value network (nets.py value) for card rewards")
    ap.add_argument("--value-weight-run", type=float, default=None, help="graded points per predicted floor")
    ap.add_argument("--rest-value", action="store_true", help="heal vs smith by the run value network")
    ap.add_argument("--rollout-value", default=None, help="RL policy file whose combat value net ends rollouts")
    ap.add_argument("--rollout-xvalue", default=None, help="expert value net (nets.py xvalue) ends rollouts")
    ap.add_argument("--rest-net", default=None, help="restnet.py model decides rest sites")
    ap.add_argument("--event-net", default=None, help="eventnet.py model decides events")
    ap.add_argument("--map-net", default=None, help="mapnet.py model decides map choices")
    ap.add_argument("--event-net-neow", action="store_true", help="the event net also decides Neow")
    ap.add_argument("--card-net-decides", action="store_true", help="card rewards by the --card-prior net alone")
    ap.add_argument("--shop-net", default=None, help="shopnet.py model decides shop purchases")
    ap.add_argument("--deck-nets", default=None, help="JSON file for policy.load_deck_nets (e.g. models/frozen/v34.json)")
    ap.add_argument("--strat", type=int, default=0, help="rest sites by K strategic rollouts per option (stratsearch.py)")
    ap.add_argument("--deck-sample", type=float, default=None,
                    help="RL in the expert bot: card/rest/event/map nets (the --*-net / --card-prior files) sample at this temperature")
    ap.add_argument("--card-prior", default=None, help="card network log-probs added to card reward scores")
    ap.add_argument("--card-prior-weight", type=float, default=None)
    ap.add_argument("--value-weight", type=float, default=None,
                    help="with --model: score = hand-written + w * model (default: model only)")
    args = ap.parse_args()
    if args.no_deck_test:
        policy.DECK_TEST = False
    if args.net_rollouts or args.net_decktest or args.run_value or args.rollout_value or args.rollout_xvalue or args.rest_net \
            or args.event_net or args.card_prior or args.map_net or args.shop_net:
        import torch
        torch.set_num_threads(1)
        import combat
        import decktest
        import nets
        if args.net_rollouts:
            combat.FAST_POLICY = nets.NetPolicy(args.net_rollouts)
        if args.net_decktest:
            decktest.NET_PILOT = nets.NetPolicy(args.net_decktest)
        if args.run_value:
            policy.RUN_VALUE = nets.RunValue(args.run_value)
        if args.rollout_value:
            combat.ROLLOUT_VALUE = nets.CombatValue(args.rollout_value)
        if args.rollout_xvalue:
            combat.ROLLOUT_VALUE = nets.ExpertValue(args.rollout_xvalue)
        if args.rest_net:
            from restnet import RestPolicy
            policy.REST_NET = RestPolicy(args.rest_net)
        if args.event_net:
            from eventnet import EventPolicy
            policy.EVENT_NET = EventPolicy(args.event_net)
        if args.card_prior:
            policy.CARD_PRIOR = nets.CardPolicy(args.card_prior)
        if args.map_net:
            from mapnet import MapPolicy
            policy.MAP_NET = MapPolicy(args.map_net)
        policy.EVENT_NET_NEOW = args.event_net_neow
        if args.shop_net:
            from shopnet import ShopPolicy
            policy.SHOP_NET = ShopPolicy(args.shop_net)
        policy.CARD_NET_DECIDES = args.card_net_decides
        if args.card_prior_weight is not None:
            policy.CARD_PRIOR_WEIGHT = args.card_prior_weight
        if args.value_weight_run is not None:
            policy.VALUE_WEIGHT = args.value_weight_run
        policy.REST_VALUE = args.rest_value
    if args.deck_nets:
        with open(args.deck_nets) as fh:
            policy.load_deck_nets(json.load(fh))
    if args.strat:
        from stratsearch import StratSearch
        with open(args.deck_nets) as fh:
            policy.STRAT = StratSearch(json.load(fh), k=args.strat)
    model = None
    if args.model:
        from value import ValueModel
        model = ValueModel(args.model)

    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    out_path = os.path.join(HERE, "runs", f"{args.tag}.jsonl")
    seeds = [f"{args.seed_prefix}_{i}" for i in range(args.seed_start, args.n)]

    done = {}
    if os.path.exists(out_path):  # resume: keep finished runs
        with open(out_path) as f:
            for l in f:
                r = json.loads(l)
                done[r["seed"]] = r
    todo = [s for s in seeds if s not in done]
    chunks = [todo[i::args.workers] for i in range(args.workers)]
    lock = threading.Lock()
    rec_lock = threading.Lock()

    def worker(chunk):
        sim = fresh_sim()
        res = []
        try:
            for s in chunk:
                sim = maybe_recycle(sim)
                try:
                    dn = None
                    if args.deck_sample is not None:
                        from decknets import DeckNets
                        paths = (args.card_prior, args.rest_net, args.event_net, args.map_net)
                        if args.deck_nets and not any(paths):   # sample the deck networks the bot plays with
                            with open(args.deck_nets) as fh:
                                cfg = json.load(fh)
                            paths = (cfg.get("card"), cfg.get("rest"), cfg.get("event"), cfg.get("map"))
                        dn = DeckNets(*paths, args.deck_sample, seed=s)
                    rec = None
                    if args.record:
                        from record import GameRecorder
                        rec = GameRecorder(args.record, rec_lock, s, {"tag": args.tag, "asc": args.ascension})
                    r = play_run(sim, s, args.character, args.acts, args.nodes, model, args.value_weight, args.ascension,
                                 json.loads(args.rollouts) if args.rollouts else None, rec=rec,
                                 explore=args.explore, decknets=dn, combat_records=not args.no_combat_records)
                except Exception as e:  # engine hang/crash: record it, restart the engine, go on
                    r = {"seed": s, "character": args.character, "ascension": args.ascension, "cleared_acts": 0,
                         "victory": False, "died": False, "act": None, "floor": None, "room": None, "hp": None,
                         "max_hp": None, "deck_size": None, "fights": 0, "seconds": 0, "planner": {},
                         "dead_end_actions": 0, "fight_log": [], "error": f"{type(e).__name__}: {e}"}
                    try:
                        sim.close()
                    except Exception:
                        pass
                    sim = fresh_sim()
                with lock:
                    with open(out_path, "a") as f:
                        f.write(json.dumps(r) + "\n")
                res.append(r)
                print(json.dumps({k: r[k] for k in ("seed", "cleared_acts", "died", "act", "floor", "room", "hp", "deck_size", "seconds", "error")}), flush=True)
        finally:
            sim.close()
        return res

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(worker, chunks))
    with open(out_path) as f:
        results = [json.loads(l) for l in f]
    cleared = [r for r in results if r["cleared_acts"] >= args.acts]
    print(json.dumps({
        "runs": len(results), "acts_target": args.acts,
        "cleared": len(cleared), "clear_rate": round(len(cleared) / max(1, len(results)), 3),
        "median_floor_reached": statistics.median([r["floor"] or 0 for r in results]) if results else None,
        "death_rooms": dict(sorted({str(k): sum(1 for r in results if r["died"] and r["room"] == k) for k in {r["room"] for r in results}}.items())),
        "wall_s": round(time.perf_counter() - t0, 1),
        "mean_run_s": round(statistics.mean([r["seconds"] for r in results]), 1) if results else None,
    }, indent=1))


if __name__ == "__main__":
    main()
