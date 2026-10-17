"""Turn-by-turn trace of the bot's costly fights, for diagnosing combat play.

usage: trace.py SEED [--acts 2] [--min-loss 15] [--ascension 0]
       trace.py - --save data/bossbench_act2a10/X.save   (one bench fight from its save)
Plays one run with the default bot and prints, for every fight that cost at least
--min-loss HP (or killed the bot), each turn: HP/block/energy, hand, enemies with
intents, and the actions the planner executed.
"""

import argparse
import json

import engine
import policy
from combat import TurnPlanner, alive, incoming_damage, powers
from run import _key


def brief(st):
    p = st.get("player") or {}
    en = [f"{e['name'][:12]} {e.get('hp')}/{e.get('max_hp')} b{e.get('block') or 0} "
          f"{','.join(f'{k}{v}' for k, v in powers(e.get('powers')).items())} "
          f"[{' '.join((i.get('type') or '?')[:3] + (str(i.get('damage')) + 'x' + str(i.get('hits') or i.get('repeat') or 1) if i.get('damage') is not None else '') for i in e.get('intents') or [])}]"
          for e in alive(st)]
    hand = " ".join(f"{c['id'].replace('CARD.', '')}{'+' if c.get('upgraded') else ''}({c.get('cost')})" for c in st.get("hand") or [])
    pw = ",".join(f"{k}{v}" for k, v in powers(st.get("player_powers")).items())
    return (f"  r{st.get('round')} hp {p.get('hp')} blk {p.get('block')} E{st.get('energy')} pw[{pw}] inc {incoming_damage(st)}\n"
            f"    hand: {hand}\n    foes: {' | '.join(en)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed")
    ap.add_argument("--acts", type=int, default=2)
    ap.add_argument("--min-loss", type=int, default=15)
    ap.add_argument("--ascension", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--rollouts", default=None, help='JSON, e.g. {"k":4,"m":3,"turns":2}')
    ap.add_argument("--only", default=None, help="only print fights in this room type (Boss, Elite, ...)")
    ap.add_argument("--save", default=None, help="trace the fight of this bench save (bossbench.py) instead of a run")
    args = ap.parse_args()
    sim = engine.fresh_sim()
    planner = TurnPlanner(sim, max_nodes=args.nodes, rollouts=json.loads(args.rollouts) if args.rollouts else None)
    log = []
    real_act = sim.act

    if args.save:
        import os
        with open(args.save + ".json") as fh:
            meta = json.load(fh)
        sim.send({"cmd": "load_save", "path": os.path.abspath(args.save)})
        col, row = meta.get("node") or meta["boss_node"]
        st = sim.act("select_map_node", col=col, row=row)
        args.min_loss, args.acts = -999, 99
    else:
        st = sim.send({"cmd": "start_run", "character": "Ironclad", "seed": args.seed, "ascension": args.ascension})
    dead, memo = {}, {}
    for _ in range(4000):
        ctx = st.get("context") or {}
        if st.get("decision") == "game_over" or (ctx.get("act") or 1) > args.acts:
            break
        room = ctx.get("room_type")
        if st.get("decision") == "combat_play" or (st.get("decision") == "card_select" and room in ("Monster", "Elite", "Boss")):
            lines, hp0 = [f"== act {ctx.get('act')} floor {ctx.get('floor')} {room}: "
                          f"{', '.join(e['name'] for e in st.get('enemies') or [])}  deck {len((st.get('player') or {}).get('deck') or [])}"], (st.get("player") or {}).get("hp")
            turn = {"acts": []}

            def on_state(s):
                if s.get("decision") == "combat_play" and (not turn["acts"] or turn["acts"][-1] == "end_turn"):
                    lines.append(brief(s))
                    turn["acts"] = []

            def logged_act(a, **kw):  # executed plans only (search uses restore around its probes)
                return real_act(a, **kw)

            # record executed paths by wrapping plan()
            real_plan = planner.plan

            def plan(root):
                path, score = real_plan(root)
                lines.append("    plays: " + " ; ".join(a if not k else f"{a} {json.dumps(k)}" for a, k in path) + f"   (score {score:.1f})")
                return path, score

            planner.plan = plan
            st = planner.play_combat(st, on_state=on_state)
            planner.plan = real_plan
            hp1 = (st.get("player") or {}).get("hp") or 0
            if (args.only is None or room == args.only) and (st.get("decision") == "game_over" or (hp0 or 0) - hp1 >= args.min_loss):
                print("\n".join(lines) + f"\n   => hp {hp0} -> {hp1 if st.get('decision') != 'game_over' else 'DEAD'}\n")
            if args.save:
                break
            continue
        a = policy.decide(sim, st, avoid=dead.get(_key(st), ()), memo=memo)
        nxt = sim.act(a[0], **a[1])
        if nxt.get("type") == "error" or _key(nxt) == _key(st):
            dead.setdefault(_key(st), set()).add(json.dumps(a, sort_keys=True))
            if nxt.get("type") == "error":
                continue
        st = nxt
    ctx = st.get("context") or {}
    print(f"END act {ctx.get('act')} floor {ctx.get('floor')} {st.get('decision')} victory={st.get('victory')}")
    sim.close()


if __name__ == "__main__":
    main()
