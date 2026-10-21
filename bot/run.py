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


MC_BOSS = 0   # run.py --mc-boss K: K extra determinized plays of every boss fight (lower-variance RL return)
MC_BOSS_ACTS = None   # --mc-boss-acts: replay only these acts' bosses (the record's boss_mc then covers only them)
# --mc-act A --mc-k K: save at act A's first map node; after the run, play the rest of the run K more
# times from that save, each with a fresh hidden future (determinize_run). The share of wins estimates
# P(win | the state the bot brought to act A): an unbiased, much less noisy win measure than the one
# real outcome, and a "future value" label for that deck (cmpwin.py)
MC_ACT, MC_K, MC_DIR = 0, 0, None
STOP_AT_ACT = 0   # --stop-at-act A: end the run at act A's first map node (after recording act_states)


def act_state(st):
    """Compact run state at an act's first map node (future-value training data)."""
    from record import card_tok, pot_tok
    p = st.get("player") or {}
    ctx = st.get("context") or {}
    return {"act": ctx.get("act"), "floor": ctx.get("floor"), "hp": p.get("hp"), "max_hp": p.get("max_hp"),
            "gold": p.get("gold"), "deck": [card_tok(c) for c in p.get("deck") or []],
            "relics": [policy.relic_id(r) or (r.get("name") if isinstance(r, dict) else r) for r in p.get("relics") or []],
            "potions": [pot_tok(x) for x in p.get("potions") or [] if x], "boss": (ctx.get("boss") or {}).get("id")}


def continue_run(sim, st, planner, acts, max_steps=4000):
    """Play on from st with the bot (as play_run, without records) until the run ends."""
    dead_ends, memo, steps = {}, {}, 0
    while steps < max_steps:
        d = st.get("decision")
        ctx = st.get("context") or {}
        if d == "game_over" or (ctx.get("act") or 1) > acts or st.get("type") == "error":
            break
        if d == "combat_play" or (d == "card_select" and ctx.get("room_type") in ("Monster", "Elite", "Boss")):
            st = planner.play_combat(st)
            if st.get("decision") == "combat_play":
                break   # the planner's action cap: counts as a loss
            continue
        a = policy.decide(sim, st, avoid=dead_ends.get(_key(st), ()), memo=memo)
        policy.pop_last_scores()
        nxt = sim.act(a[0], **a[1])
        steps += 1
        if nxt.get("type") == "error" or _key(nxt) == _key(st):
            dead_ends.setdefault(_key(st), set()).add(json.dumps(a, sort_keys=True))
            if nxt.get("type") == "error":
                continue
        st = nxt
    return st


def mc_rollouts(sim, path, seed, acts, max_nodes, model=None, value_weight=None, rollouts=None):
    """MC_K plays of the rest of the run from the save at act MC_ACT's first map node.
    Returns (sim, {"win": [...], "prog": [...], ...}); the engine may be recycled or replaced."""
    import zlib
    out = {"act": MC_ACT, "k": MC_K, "win": [], "prog": [], "secs": [], "err": 0}
    for j in range(MC_K):
        sim = maybe_recycle(sim)
        t0 = time.perf_counter()
        try:
            st = sim.send({"cmd": "load_save", "path": path})
            if st.get("type") == "error":
                raise RuntimeError(st.get("message"))
            sim.send({"cmd": "determinize_run", "seed": zlib.crc32(f"{seed}/mc{MC_ACT}/{j}".encode()) * 2654435761 % (2 ** 62)})
            planner = TurnPlanner(sim, max_nodes=max_nodes, value=model, value_weight=value_weight, rollouts=rollouts)
            end = continue_run(sim, st, planner, acts)
        except Exception as e:
            out["err"] += 1
            out.setdefault("errors", []).append(f"{type(e).__name__}: {e}"[:200])
            try:
                sim.close()
            except Exception:
                pass
            sim = fresh_sim()
            continue
        ctx = end.get("context") or {}
        win = bool(end.get("victory"))
        out["win"].append(int(win))
        out["prog"].append(53 if win else ((ctx.get("act") or 1) - 1) * 17 + (ctx.get("floor") or 0))
        out["secs"].append(round(time.perf_counter() - t0, 1))
    return sim, out


def boss_mc(sim, st, planner, seed, enemy_max):
    """Mean boss-HP share removed (a win = 1) over MC_BOSS replays of this boss fight from its
    first state, each with a different hidden future; the sim is left as it was."""
    import zlib
    import combat
    snap = sim.send({"cmd": "snapshot"})["id"]
    shares = []
    try:
        for i in range(MC_BOSS):
            sim.send({"cmd": "restore", "id": snap})
            sim.send({"cmd": "determinize", "seed": zlib.crc32(f"{seed}/mcboss/{(st.get('context') or {}).get('act')}/{i}".encode())})
            last = {"s": st}

            def w(s_):
                if s_.get("enemies") is not None:
                    last["s"] = s_

            end = planner.play_combat(dict(st), on_state=w)
            # still in combat = the planner's action cap: not a win
            won = end.get("decision") not in ("game_over", "combat_play", "card_select") or bool(end.get("victory"))
            left = 0 if won else sum((e.get("hp") or 0) for e in combat.alive(last["s"]))
            shares.append(1.0 if won else max(0.0, 1 - left / max(1, enemy_max)))
    finally:
        sim.send({"cmd": "restore", "id": snap})
        sim.send({"cmd": "drop_snapshots", "from": snap})
    return shares


def play_run(sim, seed, character, acts, max_nodes, model=None, value_weight=None, ascension=0, rollouts=None,
             rec=None, explore=0.0, decknets=None, combat_records=True):
    """rec: optional record.GameRecorder (self-play data for the policy and value nets).
    explore: chance to take a uniformly random card reward option instead (value-net data
    needs decks the policy would not build; such decisions are flagged in the record)."""
    import random
    rng = random.Random(f"explore/{seed}")
    vine_rng = random.Random(f"vine/{seed}")
    policy.ASCENSION = ascension
    planner = TurnPlanner(sim, max_nodes=max_nodes, value=model, value_weight=value_weight, rollouts=rollouts)
    st = sim.send({"cmd": "start_run", "character": character, "seed": seed, "ascension": ascension})
    t0, steps, fights, hp_log = time.perf_counter(), 0, 0, []
    dead_ends, loops, memo, stuck = {}, 0, {}, False
    boss_dmg = 0.0   # sum over boss fights of the share of boss HP removed (a win = 1), as selfplay.py
    mc_boss = []     # per boss fight: the MC_BOSS replay shares
    mc_boss_acts = []
    act_states, mc_save = {}, None
    while steps < 5000:
        d = st.get("decision")
        act = (st.get("context") or {}).get("act") or 1
        if d == "game_over" or act > acts or st.get("type") == "error":
            break
        if d == "map_select" and act not in act_states:
            act_states[act] = act_state(st)
            if MC_K and act == MC_ACT and MC_DIR:
                path = os.path.join(MC_DIR, f"{seed}_a{act}.save")
                if sim.send({"cmd": "write_continue_save", "path": path}).get("success"):
                    mc_save = path
            if STOP_AT_ACT and act >= STOP_AT_ACT:
                break   # deck-building benches: the run's act-STOP_AT_ACT state is all they need
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
            if room == "Boss" and MC_BOSS and (MC_BOSS_ACTS is None or act in MC_BOSS_ACTS):
                mc_boss.append(boss_mc(sim, st, planner, seed, enemy_max))
                mc_boss_acts.append(act)
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
                           "hp_after": (st.get("player") or {}).get("hp"),
                           "won": st.get("decision") != "game_over" or bool(st.get("victory")),
                           "enemy_hp_left": last_enemies.get("hp"), "rounds": last_enemies.get("round"),
                           "potions_before": pots0, "potions_left": last_enemies.get("potions"),
                           **({"deck": deck_stats} if deck_stats else {})})
            if st.get("decision") == "combat_play":
                stuck = True   # play_combat hit its action cap: give up on the run
                break
            continue
        if d == "card_reward":
            import vine
            if vine.CFG is not None:
                vine.maybe_vine(sim, st, seed, vine_rng)
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
        "boss_dmg": round(boss_dmg, 3), "boss_mc": [[round(x, 3) for x in m] for m in mc_boss] if MC_BOSS else None,
        **({"boss_mc_acts": mc_boss_acts} if MC_BOSS_ACTS is not None else {}),
        "act_states": act_states, **({"mc_save": mc_save} if mc_save else {}),
        "planner": dict(planner.stats), "dead_end_actions": loops, "fight_log": hp_log, "stuck": stuck, "error": st.get("message") if st.get("type") == "error" else None,
    }
    if rec and not (MC_K and mc_save):   # with act-start rollouts pending, the worker finishes the record
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
    ap.add_argument("--weights", default=None, help="JSON overrides for combat.W (e.g. graded rollout deaths)")
    ap.add_argument("--human-prior", type=float, default=0.0, help="card rewards: + this x human pick Elo (per 100 over 1200)")
    ap.add_argument("--human-prior-from-act", type=int, default=1)
    ap.add_argument("--skip-bias", type=float, default=None, help="card rewards: logit added to skipping")
    ap.add_argument("--skip-bias-from-act", type=int, default=1)
    ap.add_argument("--human-skip-elo", type=float, default=None, help="with --human-prior: skip counts as a card of this Elo")
    ap.add_argument("--decktest-from-act", type=int, default=0,
                    help="with card_decides nets: card rewards from this act on by the deck test, the net as prior")
    ap.add_argument("--engine-config", default=None, help="engine build (bin/<config>), e.g. Fix1 while Release jobs still run")
    ap.add_argument("--deck-value", default=None, help="deckvalue.py model: card rewards by the boss-fight deck value")
    ap.add_argument("--deck-value-beta", type=float, default=None, help="weight of the card net's log-probability")
    ap.add_argument("--deck-value-from-act", type=int, default=1)
    ap.add_argument("--deck-value-margin", type=float, default=None,
                    help="gated: keep the card net's pick unless the value net rates another option this much higher")
    ap.add_argument("--rest-smith-hp", type=float, default=None, help="rest net: smith whenever HP >= this share of max")
    ap.add_argument("--rest-smith-hp-boss", type=float, default=None, help="... when the boss is the next node")
    ap.add_argument("--map-node-bias", default=None, help='map net: JSON {room type: logit}, e.g. {"RestSite": 1}')
    ap.add_argument("--force-neow", type=int, default=None, help="Neow lab: take option K of the opening choice")
    ap.add_argument("--card-lab-rule", action="store_true", help="act 2+: card lab values overrule the card net by a margin")
    ap.add_argument("--ancient-rule", action="store_true", help="Ancient relic choices by the relic lab's measured values")
    ap.add_argument("--shop-potions", action="store_true", help="with the shop net: leftover gold buys potions for free slots")
    ap.add_argument("--remove-first", action="store_true", help="with the shop net: remove a card first whenever affordable")
    ap.add_argument("--decktest-pilot", default=None, help="quick (quick_policy) or planner (TurnPlanner, --decktest-nodes)")
    ap.add_argument("--decktest-nodes", type=int, default=None)
    ap.add_argument("--decktest-reps", type=int, default=None)
    ap.add_argument("--decktest-boss-only", action="store_true", help="deck test against the act boss only")
    ap.add_argument("--rest-smith-bias", type=float, default=0.0, help="rest network: logit added to smith options")
    ap.add_argument("--boss-value", default=None, help='JSON {act: bossvalue model} for value-guided card rewards')
    ap.add_argument("--map-elite-bias-act", default=None, help='JSON {act: logit} added to elite rooms, e.g. {"3": -3}')
    ap.add_argument("--deck-roll", default=None, help="card rewards by imagined deck futures: comma-separated deck models (deckroll.py)")
    ap.add_argument("--deck-roll-m", type=int, default=8, help="--deck-roll: futures per option")
    ap.add_argument("--deck-roll-from-act", type=int, default=1)
    ap.add_argument("--deck-roll-margin", type=float, default=None, help="--deck-roll: overrule the card net only by this margin")
    ap.add_argument("--deck-roll-beta", type=float, default=0.0)
    ap.add_argument("--deck-roll-select", action="store_true", help="--deck-roll: also shop removals and smiths")
    ap.add_argument("--deck-roll-hall", default=None, help="--deck-roll: comma-separated hallway models (deckvalue --target hpfrac)")
    ap.add_argument("--mc-boss", type=int, default=0, help="K extra determinized plays of each boss fight (RL return)")
    ap.add_argument("--mc-boss-acts", default=None, help='--mc-boss only at these acts\' bosses, e.g. "3"')
    ap.add_argument("--remove-defend-first", action="store_true", help="removals take Defends before Strikes")
    ap.add_argument("--rs-value", default=None, help="card rewards by the run-state value net (rsvalue.py model)")
    ap.add_argument("--rs-beta", type=float, default=0.0, help="--rs-value: + beta x the card net's log-probability")
    ap.add_argument("--rs-from-act", type=int, default=1)
    ap.add_argument("--vine-out", default=None, help="vine.py: value sampled card rewards by real futures, rows to this file")
    ap.add_argument("--vine-p", type=float, default=0.25)
    ap.add_argument("--vine-k", type=int, default=3)
    ap.add_argument("--vine-acts", default="1,2")
    ap.add_argument("--vine-nodes", type=int, default=400)
    ap.add_argument("--vine-rollouts", default='{"k":4,"m":4,"turns":2}')
    ap.add_argument("--vine-models", default="models/dz_boss_s0.pt,models/dz_boss_s1.pt,models/dz_boss_s2.pt,models/dz_boss_s3.pt")
    ap.add_argument("--stop-at-act", type=int, default=0, help="end runs at this act's first map node (deck-building benches)")
    ap.add_argument("--mc-act", type=int, default=3, help="--mc-k: the act whose first map node the rollouts start from")
    ap.add_argument("--mc-k", type=int, default=0, help="K plays of the rest of the run from act --mc-act's start (cmpwin.py)")
    ap.add_argument("--mc-dir", default=None, help="--mc-k: where the act-start saves go (default runs/mcsaves/<tag>)")
    ap.add_argument("--boss-beta", type=float, default=0.0, help="weight of the boss-win logit gain on card rewards")
    ap.add_argument("--map-elite-bias", type=float, default=0.0, help="map network: logit added to elite rooms")
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
    if args.engine_config:
        import engine
        engine.ENGINE_CONFIG = args.engine_config
    if args.no_deck_test:
        policy.DECK_TEST = False
    if args.weights:
        import combat
        combat.W.update(json.loads(args.weights))
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
        if args.value_weight_run is not None:
            policy.VALUE_WEIGHT = args.value_weight_run
        policy.REST_VALUE = args.rest_value
    if args.deck_nets:
        with open(args.deck_nets) as fh:
            policy.load_deck_nets(json.load(fh))
    if args.card_prior_weight is not None:   # also with --deck-nets alone
        policy.CARD_PRIOR_WEIGHT = args.card_prior_weight
    if args.human_prior:
        import nets
        nets.set_human_prior(args.human_prior)
        nets.HUMAN_PRIOR_MIN_ACT = args.human_prior_from_act
        nets.HUMAN_SKIP_ELO = args.human_skip_elo
    if args.skip_bias is not None:
        import nets
        nets.SKIP_BIAS = args.skip_bias
        nets.SKIP_BIAS_MIN_ACT = args.skip_bias_from_act
    global MC_BOSS, MC_BOSS_ACTS, MC_ACT, MC_K, MC_DIR, STOP_AT_ACT
    STOP_AT_ACT = args.stop_at_act
    if args.mc_boss:
        MC_BOSS = args.mc_boss
    if args.mc_boss_acts:
        MC_BOSS_ACTS = {int(a) for a in args.mc_boss_acts.split(",")}
    if args.mc_k:
        MC_ACT, MC_K = args.mc_act, args.mc_k
        MC_DIR = args.mc_dir or os.path.join(HERE, "runs", "mcsaves", args.tag)
        os.makedirs(MC_DIR, exist_ok=True)
    if args.boss_value and args.boss_beta:
        import nets
        from bossvalue import BossValue
        nets.BOSS_VALUES = {int(a): BossValue(os.path.join(HERE, p)) for a, p in json.loads(args.boss_value).items()}
        nets.BOSS_BETA = args.boss_beta
    policy.DECKTEST_FROM_ACT = args.decktest_from_act
    policy.SHOP_POTIONS = args.shop_potions
    policy.ANCIENT_RULE = args.ancient_rule
    policy.CARD_LAB_RULE = args.card_lab_rule
    policy.FORCE_NEOW = args.force_neow
    policy.REST_SMITH_HP = args.rest_smith_hp
    policy.REST_SMITH_HP_BOSS = args.rest_smith_hp_boss
    if args.deck_value:
        import torch
        torch.set_num_threads(1)
        from deckvalue import DeckValue
        policy.DECK_VALUE = DeckValue(os.path.join(HERE, args.deck_value))
        policy.DECK_VALUE_FROM_ACT = args.deck_value_from_act
        policy.DECK_VALUE_MARGIN = args.deck_value_margin
        if args.deck_value_beta is not None:
            policy.DECK_VALUE_BETA = args.deck_value_beta
    policy.REMOVE_FIRST = args.remove_first
    policy.REMOVE_DEFEND_FIRST = args.remove_defend_first
    if args.rs_value:
        import torch
        torch.set_num_threads(1)
        from rsvalue import RSValue
        policy.RS_VALUE = RSValue(args.rs_value)
        policy.RS_BETA = args.rs_beta
        policy.RS_FROM_ACT = args.rs_from_act
    if args.vine_out:
        import torch
        torch.set_num_threads(1)
        import vine
        vine.CFG = {"out": os.path.join(HERE, args.vine_out), "p": args.vine_p, "k": args.vine_k,
                    "acts": {int(a) for a in args.vine_acts.split(",")}, "nodes": args.vine_nodes,
                    "rollouts": json.loads(args.vine_rollouts), "models": args.vine_models.split(",")}
        os.makedirs(os.path.dirname(vine.CFG["out"]), exist_ok=True)
    if args.deck_roll:
        import torch
        torch.set_num_threads(1)
        from deckroll import DeckRoller
        policy.DECK_ROLL = DeckRoller(args.deck_roll.split(","), m=args.deck_roll_m,
                                      hall_paths=args.deck_roll_hall.split(",") if args.deck_roll_hall else None)
        policy.DECK_ROLL_FROM_ACT = args.deck_roll_from_act
        policy.DECK_ROLL_MARGIN = args.deck_roll_margin
        policy.DECK_ROLL_BETA = args.deck_roll_beta
        policy.DECK_ROLL_SELECT = args.deck_roll_select
    if args.decktest_pilot or args.decktest_nodes or args.decktest_reps or args.decktest_boss_only:
        import decktest   # card rewards only; upgrades, shops and events keep the defaults
        if args.decktest_pilot == "planner":
            decktest.CARD_PILOT = "planner"
            decktest.REFINE_NODES = args.decktest_nodes or decktest.REFINE_NODES
        decktest.CARD_REPS = args.decktest_reps
        decktest.CARD_BOSS_ONLY = args.decktest_boss_only
    if args.rest_smith_bias:
        import restnet
        restnet.SMITH_BIAS = args.rest_smith_bias
    if args.map_elite_bias:
        import mapnet
        mapnet.ELITE_BIAS = args.map_elite_bias
    if args.map_elite_bias_act:
        import mapnet
        mapnet.ELITE_BIAS_ACT = {int(k): float(v) for k, v in json.loads(args.map_elite_bias_act).items()}
    if args.map_node_bias:
        import mapnet
        mapnet.NODE_BIAS = json.loads(args.map_node_bias)
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
                        paths, shop = (args.card_prior, args.rest_net, args.event_net, args.map_net), args.shop_net
                        if args.deck_nets and not any(paths):   # sample the deck networks the bot plays with
                            with open(args.deck_nets) as fh:
                                cfg = json.load(fh)
                            paths = (cfg.get("card"), cfg.get("rest"), cfg.get("event"), cfg.get("map"))
                            shop = cfg.get("shop")
                        dn = DeckNets(*paths, args.deck_sample, seed=s, shop=shop)
                    rec = None
                    if args.record:
                        from record import GameRecorder
                        rec = GameRecorder(args.record, rec_lock, s, {"tag": args.tag, "asc": args.ascension})
                    r = play_run(sim, s, args.character, args.acts, args.nodes, model, args.value_weight, args.ascension,
                                 json.loads(args.rollouts) if args.rollouts else None, rec=rec,
                                 explore=args.explore, decknets=dn, combat_records=not args.no_combat_records)
                    if MC_K and r.get("mc_save") and not r.get("error"):
                        sim, r["mc"] = mc_rollouts(sim, r["mc_save"], s, args.acts, args.nodes, model, args.value_weight,
                                                   json.loads(args.rollouts) if args.rollouts else None)
                    if rec and MC_K and r.get("mc_save"):
                        rec.finish(r)
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
