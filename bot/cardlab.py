"""Card lab: how much does adding one card to the deck help in a boss fight?

usage: cardlab.py --tag a0 [--cards POMMEL_STRIKE,INFLAME | --pool ironclad] [--reps 3]
                  [--rollouts '{"k":4,"m":2,"turns":2}'] [--workers 6]
For every pre-boss save of the bench (bossbench.py collect), every repetition and every
arm (the deck as it is, or the deck plus one card): load the save, add the card, enter
the boss, determinize the hidden future with a seed shared by all arms of that
(save, rep), and fight. The summary is paired against the unchanged deck: mean HP
change and win-rate change per card, with a standard error. Results append to
data/cardlab_<tag>_<label>.jsonl, so an interrupted lab resumes.
"""

import argparse
import glob
import json
import math
import os
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

from combat import TurnPlanner, alive, quick_policy
import decktest
from engine import fresh_sim, maybe_recycle

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = "planner"   # or "quick" (quick_policy: ~100x faster, ranks decks similarly, corr 0.79)
IRONCLAD = """ANGER ARMAMENTS ASHEN_STRIKE BARRICADE BATTLE_TRANCE BLOOD_WALL BLOODLETTING BLUDGEON BODY_SLAM
BRAND BREAK BREAKTHROUGH BULLY BURNING_PACT CASCADE CINDER COLOSSUS CONFLAGRATION CORRUPTION CRIMSON_MANTLE
CRUELTY DARK_EMBRACE DEMON_FORM DISMANTLE DOMINATE DRUM_OF_BATTLE EVIL_EYE EXPECT_A_FIGHT FEED FEEL_NO_PAIN
FIEND_FIRE FIGHT_ME FLAME_BARRIER FORGOTTEN_RITUAL HAVOC HEADBUTT HELLRAISER HEMOKINESIS HOWL_FROM_BEYOND
IMPERVIOUS INFERNAL_BLADE INFERNO INFLAME IRON_WAVE JUGGERNAUT JUGGLING MANGLE MIDNIGHT MOLTEN_FIST NOT_YET
OFFERING ONE_TWO_PUNCH OUTRAGE PACTS_END PERFECTED_STRIKE PILLAGE POMMEL_STRIKE PRIMAL_FORCE PYRE RAGE
RAMPAGE RUPTURE SECOND_WIND SETUP_STRIKE SHRUG_IT_OFF SPITE STAMPEDE STOKE STOMP STONE_ARMOR SWORD_BOOMERANG
TAUNT TEAR_ASUNDER THRASH THUNDERCLAP TREMBLE TRUE_GRIT TWIN_STRIKE UNMOVABLE UNRELENTING UPPERCUT VICIOUS
WHIRLWIND STRIKE_IRONCLAD DEFEND_IRONCLAD""".split()


def fight(sim, meta, card, rep, rollouts, nodes):
    st = sim.send({"cmd": "load_save", "path": meta["path"]})
    if card:
        r = sim.send({"cmd": "set_player", "add_cards": [card]})
        if r.get("type") == "error":
            return {"seed": meta["seed"], "rep": rep, "card": card, "error": r.get("message")}
    col, row = meta.get("node") or meta["boss_node"]
    st = sim.act("select_map_node", col=col, row=row)
    sim.send({"cmd": "determinize", "seed": zlib.crc32(f"{meta['seed']}/{rep}".encode())})
    enemy_max = sum((e.get("max_hp") or 0) for e in st.get("enemies") or [])
    rounds, last = [0], {"s": st}

    def watch(s):
        rounds[0] = s.get("round") or rounds[0]
        if s.get("enemies") is not None:
            last["s"] = s

    if PILOT == "quick":
        for _ in range(400):
            if st.get("decision") not in ("combat_play", "card_select") or st.get("type") == "error":
                break
            watch(st)
            a = quick_policy(st)
            nxt = sim.act(a[0], **a[1])
            st = sim.act("end_turn") if nxt.get("type") == "error" and st.get("decision") == "combat_play" else nxt
    else:
        st = TurnPlanner(sim, max_nodes=nodes, rollouts=rollouts).play_combat(st, on_state=watch)
    won = st.get("decision") != "game_over" or bool(st.get("victory"))
    # HP from the last combat state: winning the act boss heals for the next act
    hp = (last["s"].get("player") or {}).get("hp") if won else 0
    left = 0 if won else sum((e.get("hp") or 0) for e in alive(last["s"]))
    return {"seed": meta["seed"], "rep": rep, "card": card, "boss": meta.get("boss"), "won": won,
            "hp_after": hp, "graded": decktest.graded(won, hp or 0, left, enemy_max), "rounds": rounds[0]}


def summarize(rows):
    base = {(r["seed"], r["rep"]): r for r in rows if r["card"] is None and "error" not in r}
    by = {}
    for r in rows:
        if r["card"] is None or "error" in r or (r["seed"], r["rep"]) not in base:
            continue
        b = base[(r["seed"], r["rep"])]
        key = "graded" if "graded" in r and "graded" in b else "hp_after"
        by.setdefault(r["card"], []).append((r[key] - b[key], int(r["won"]) - int(b["won"])))
    out = []
    for card, d in by.items():
        n = len(d)
        mean = sum(x for x, _ in d) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x, _ in d) / max(1, n - 1))
        out.append({"card": card, "n": n, "d_score": round(mean, 2), "se": round(sd / math.sqrt(n), 2),
                    "d_win": round(sum(w for _, w in d) / n, 3)})
    out.sort(key=lambda x: -x["d_score"])
    base_rows = list(base.values())
    return {"baseline_win": round(sum(r["won"] for r in base_rows) / max(1, len(base_rows)), 3),
            "baseline_hp": round(sum(r["hp_after"] for r in base_rows) / max(1, len(base_rows)), 2),
            "cards": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cards", default=None)
    ap.add_argument("--pool", default="ironclad")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rollouts", default=None)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--pilot", default="planner", choices=["planner", "quick"])
    ap.add_argument("--room", default=None, help="fight benches: only this room type (Monster, Elite, Boss)")
    ap.add_argument("--label", default="heur", help="results file suffix (one per planner config)")
    args = ap.parse_args()

    global PILOT
    PILOT = args.pilot
    out_path = os.path.join(HERE, "data", f"cardlab_{args.tag}_{args.label}.jsonl")
    rows = []
    if os.path.exists(out_path):
        with open(out_path) as fh:
            rows = [json.loads(l) for l in fh]
    if not args.summary_only:
        metas = []
        for p in sorted(glob.glob(os.path.join(HERE, "data", f"bossbench_{args.tag}", "*.save.json"))):
            with open(p) as fh:
                metas.append(json.load(fh))
        if args.room:
            metas = [m for m in metas if m.get("room", "Boss") == args.room]
        cards = args.cards.split(",") if args.cards else IRONCLAD
        done = {(r["seed"], r["rep"], r["card"]) for r in rows}
        jobs = [(m, c, rep) for c in [None] + cards for m in metas for rep in range(args.reps)
                if (m["seed"], rep, c) not in done]
        rollouts = json.loads(args.rollouts) if args.rollouts else None
        lock = threading.Lock()
        t0 = time.perf_counter()
        print(json.dumps({"jobs": len(jobs), "fights_per_card": len(metas) * args.reps}), flush=True)

        def worker(chunk):
            sim = fresh_sim()
            try:
                for i, (m, c, rep) in enumerate(chunk):
                    sim = maybe_recycle(sim)
                    try:
                        r = fight(sim, m, c, rep, rollouts, args.nodes)
                    except Exception as e:  # engine died: record and restart
                        r = {"seed": m["seed"], "rep": rep, "card": c, "error": repr(e)[:200]}
                        try:
                            sim.close()
                        except Exception:
                            pass
                        sim = fresh_sim()
                    with lock:
                        rows.append(r)
                        with open(out_path, "a") as fh:
                            fh.write(json.dumps(r) + "\n")
            finally:
                sim.close()

        chunks = [jobs[i::args.workers] for i in range(args.workers)]
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(worker, chunks))
        print(json.dumps({"wall_s": round(time.perf_counter() - t0, 1)}), flush=True)
    print(json.dumps(summarize(rows), indent=1))


if __name__ == "__main__":
    main()
