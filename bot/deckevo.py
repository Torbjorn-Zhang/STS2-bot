"""Deck evolution: what decks beat a given boss? Let the engine say, without human archetypes.

A genetic search over decks for one boss save. The save keeps its real HP, relics and
potions; only the deck changes. Fitness is the planner's graded boss result (a win = 100 + HP
left, a loss = 80 x boss HP share removed), averaged over hidden futures that are the same for
every deck of a generation (common random numbers). Mutations: add a pool card, remove a card,
upgrade one, copy one, swap one for a pool card. Survivors are re-scored every generation, so
lucky draws do not stick.

The result shows deck directions the bot never builds (target archetypes per boss). It also
shows how far deck building alone could take the act-3 bosses with the HP and relics the bot
really brings.

usage: deckevo.py SAVE.json --out F.jsonl [--gens 16 --pop 20 --reps 3 --workers W]
       deckevo.py report F.jsonl [...]
"""
import argparse
import json
import os
import random
import statistics as S
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_DECK, MAX_DECK = 8, 40


def fight(sim, meta, deck, rep, nodes, rollouts, detail=False):
    """deck: card ids, "ID+" upgraded. -> (graded score, won[, info]) (needs the Fix1 or Diag engine)."""
    import combat
    import decktest
    from combat import TurnPlanner
    st = sim.send({"cmd": "load_save", "path": meta["path"]})
    if st.get("type") == "error":
        raise RuntimeError(st.get("message"))
    plain = [c.rstrip("+") for c in deck]
    r = sim.send({"cmd": "set_player", "deck": plain})
    if len((r.get("player") or {}).get("deck") or []) != len(plain):
        raise RuntimeError(f"deck not set: {r.get('message') or ''}")
    ups = [c.rstrip("+") for c in deck if c.endswith("+")]
    if ups:
        r = sim.send({"cmd": "set_player", "upgrade": ups})
    player = r.get("player") or {}   # (curses and statuses stay unupgraded: the row records the real deck)
    col, row = meta["node"]
    st = sim.act("select_map_node", col=col, row=row)
    sim.send({"cmd": "determinize", "seed": zlib.crc32(f"{meta['seed']}/evo/{rep}".encode())})
    planner = TurnPlanner(sim, max_nodes=nodes, rollouts=rollouts)
    last = {"s": st, "dmg": 0.0, "e": {}}

    def watch(s):   # cumulative damage: multi-phase bosses (Test Subject) refill their HP
        if s.get("enemies") is not None:
            last["s"] = s
            now = {e.get("id"): (e.get("hp") or 0) for e in s["enemies"]}
            for k, h0 in last["e"].items():
                if now.get(k, 0) < h0:
                    last["dmg"] += h0 - now.get(k, 0)
            last["e"] = now

    watch(st)
    end = planner.play_combat(st, on_state=watch)
    # Ascension 10 (Double Boss, engine Fix2+): after the first act-3 boss the map offers the
    # second boss (no heal in between); the save's fight is only won once both are dead
    import policy
    for _ in range(30):
        if (meta.get("act") or 0) < 3 or end.get("victory") or end.get("decision") in ("game_over", "combat_play"):
            break
        ctx = end.get("context") or {}
        if end.get("decision") == "card_select" and ctx.get("room_type") in ("Monster", "Elite", "Boss"):
            end = planner.play_combat(end, on_state=watch)
            continue
        if end.get("decision") == "map_select":
            boss = [c for c in end.get("choices") or [] if c.get("type") == "Boss"]
            if not boss:
                break
            last["dmg1"] = last["dmg"]
            last["e"] = {}
            end = sim.act("select_map_node", col=boss[0]["col"], row=boss[0]["row"])
            watch(end)
            last["bosses"] = last.get("bosses", 1) + 1
            if end.get("decision") in ("combat_play", "card_select"):
                end = planner.play_combat(end, on_state=watch)
            continue
        a = policy.decide(sim, end, avoid=(), memo={})
        end = sim.act(a[0], **a[1])
    won = end.get("decision") not in ("game_over", "combat_play", "card_select") or bool(end.get("victory"))
    if (meta.get("act") or 0) >= 3 and engine_double_boss():
        won = bool(end.get("victory"))
    hp = (last["s"].get("player") or {}).get("hp") if won else 0
    left = 0 if won else sum((e.get("hp") or 0) for e in combat.alive(last["s"]))
    from record import card_tok
    info = {"player": player, "deck": sorted(card_tok(c) for c in player.get("deck") or []), "hp_after": hp or 0, "rounds": last["s"].get("round"), "bosses": last.get("bosses", 1),
            "left_frac": 0.0 if won else left / max(1.0, last["dmg"] + left)}
    if won:
        g = decktest.graded(True, hp or 0, 0, 1)
    elif (meta.get("act") or 0) >= 3 and engine_double_boss():   # two final bosses: 40 each
        if "dmg1" in last:
            d2 = last["dmg"] - last["dmg1"]
            g = 40.0 + 40.0 * d2 / max(1.0, d2 + left)
        else:
            g = 40.0 * last["dmg"] / max(1.0, last["dmg"] + left)
        info["left_frac"] = 1.0 - g / 80.0   # deckvalue.fight_score: a loss scores 0.8 x (1 - left_frac)
    else:
        g = 80.0 * last["dmg"] / max(1.0, last["dmg"] + left)
    return (g, won, info) if detail else (g, won)


def engine_double_boss():
    """Engines from Fix2 on play Ascension 10's second act-3 boss; older ones end the run early."""
    import engine
    return engine.ENGINE_CONFIG not in ("Release", "Debug", "Diag", "Fix1", "Exp", "ReleaseExp")


def mutate(deck, pool, rng):
    d = list(deck)
    for _ in range(rng.choice((1, 1, 2, 3))):
        op = rng.random()
        if op < 0.35 and len(d) < MAX_DECK:
            d.append(rng.choice(pool))
        elif op < 0.6 and len(d) > MIN_DECK:
            d.pop(rng.randrange(len(d)))
        elif op < 0.75:
            i = rng.randrange(len(d))
            if not d[i].endswith("+"):
                d[i] += "+"
        elif op < 0.85 and len(d) < MAX_DECK:
            d.append(rng.choice(d))
        else:
            d[rng.randrange(len(d))] = rng.choice(pool)
    return sorted(d)


def crossover(a, b, rng):
    """Half of each parent's cards (decks are multisets; order does not matter)."""
    child = [c for c in a if rng.random() < 0.5] + [c for c in b if rng.random() < 0.5]
    return sorted(child[:MAX_DECK]) if len(child) >= MIN_DECK else sorted(a)


def run(args):
    import bossbench
    from combolab import ironclad_pool
    with open(args.save) as fh:
        meta = json.load(fh)
    meta["path"] = os.path.abspath(args.save[:-5] if args.save.endswith(".json") else args.save)
    pool = ironclad_pool() + ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD"]
    rng = random.Random(f"{meta['seed']}/{args.seed}")
    rollouts = json.loads(args.rollouts) if args.rollouts else None

    from engine import fresh_sim
    sim = fresh_sim()
    sim.send({"cmd": "load_save", "path": meta["path"]})
    st = sim.send({"cmd": "set_player"})   # no changes: replies with the player summary (deck)
    sim.close()
    start = sorted(c["id"].replace("CARD.", "") + ("+" if c.get("upgraded") else "") for c in (st.get("player") or {}).get("deck") or [])
    if not start:
        raise SystemExit(f"no deck in {meta['path']}: {str(st)[:200]}")
    pop = [start] + [mutate(start, pool, rng) for _ in range(args.pop - 1)]
    hist = {}   # deck key -> list of (gen, score, won)

    def key(d):
        return ",".join(d)

    def score(sim_, item):
        d, rep = item
        try:
            g, w = fight(sim_, meta, d, rep, args.nodes, rollouts)
        except Exception as e:
            return key(d), rep, None, False, f"{type(e).__name__}: {e}"[:160]
        return key(d), rep, g, w, None

    t0 = time.time()
    for gen in range(args.gens):
        reps = [args.seed * 1000 + gen * args.reps + i for i in range(args.reps)]
        uniq = {key(d): d for d in pop}
        items = [(d, r) for d in uniq.values() for r in reps]
        res = bossbench.parallel(items, score, args.workers)
        for k, rep, g, w, err in res:
            if g is not None:
                hist.setdefault(k, []).append((gen, g, w))
        # rank by mean over every rep a deck has had (survivors accumulate evidence)
        def fit(d):
            h = hist.get(key(d)) or [(0, 0.0, False)]
            return S.mean(x[1] for x in h)
        ranked = sorted(uniq.values(), key=fit, reverse=True)
        best = ranked[0]
        bh = hist.get(key(best)) or []
        rec = {"gen": gen, "save": meta["seed"], "boss": meta.get("boss"), "best": best, "best_fit": round(fit(best), 1),
               "best_wins": sum(x[2] for x in bh), "best_n": len(bh),
               "start_fit": round(fit(start), 1) if key(start) in hist else None,
               "pop_mean": round(S.mean(fit(d) for d in ranked), 1), "errors": sum(1 for x in res if x[4]),
               "err": next((x[4] for x in res if x[4]), None), "wall_s": round(time.time() - t0)}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("gen", "best_fit", "best_wins", "best_n", "start_fit", "pop_mean", "errors")}), flush=True)
        elite = ranked[:max(2, args.pop // 4)]
        nxt = list(elite)
        if key(start) not in {key(d) for d in nxt}:
            nxt.append(start)   # the bot's own deck stays as the reference
        while len(nxt) < args.pop:
            a, b = rng.sample(elite, 2) if len(elite) > 1 else (elite[0], elite[0])
            child = crossover(a, b, rng) if rng.random() < 0.3 else list(a)
            nxt.append(mutate(child, pool, rng))
        pop = nxt
    final = sorted(hist.items(), key=lambda kv: -S.mean(x[1] for x in kv[1]))[:5]
    with open(args.out, "a") as fh:
        for k, h in final:
            fh.write(json.dumps({"final": k.split(","), "save": meta["seed"], "boss": meta.get("boss"), "n": len(h),
                                 "fit": round(S.mean(x[1] for x in h), 1), "wins": sum(x[2] for x in h)}) + "\n")


def report(files):
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        gens = [r for r in rows if "gen" in r]
        fin = [r for r in rows if "final" in r]
        if not gens:
            continue
        g0, g1 = gens[0], gens[-1]
        print(f"{g1['save']} {g1['boss']}: start {g0.get('start_fit')} -> best {g1['best_fit']} "
              f"(wins {g1['best_wins']}/{g1['best_n']}) after {g1['gen'] + 1} gens, {g1['wall_s']} s")
        start = set(gens[0]["best"]) if gens else set()
        for r in fin[:2]:
            from collections import Counter
            c = Counter(r["final"])
            print(f"   fit {r['fit']} wins {r['wins']}/{r['n']} size {len(r['final'])}: "
                  + " ".join(f"{k}x{v}" if v > 1 else k for k, v in sorted(c.items(), key=lambda x: -x[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("save")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--gens", type=int, default=16)
    ap.add_argument("--pop", type=int, default=20)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--rollouts", default='{"k":4,"m":4,"turns":2}')
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine-config", default="Fix2", help="needs set_player upgrade (Fix1+); Fix2+: A10 double boss")
    args = ap.parse_args()
    import engine
    engine.ENGINE_CONFIG = args.engine_config
    if args.save == "report":
        report(args.files)
    else:
        run(args)


if __name__ == "__main__":
    sys.exit(main())
