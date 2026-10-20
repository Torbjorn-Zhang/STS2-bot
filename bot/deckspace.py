"""Deck-space exploration: learn what makes a WHOLE deck strong, not card by card or pair by pair.

The game's answer is not linear. A combo can have any number of pieces; keeping or removing
the starter cards, deck size, upgrades and relics all interact. So no hand-made hypotheses
(pairs, "remove Strikes first"): sample decks of every shape on many real boss saves (real HP,
relics, potions), fight them in the engine, and fit a set model with attention (deckvalue.py
--arch attn) on the outcomes. Then let a search over decks guided by the model propose the decks
it believes in most (or is least sure about), fight those in the engine, and retrain: the
AlphaZero loop, in deck space.

  deckspace.py explore --saves DIR --shard k/n --per-save N --out F.jsonl   engine fights of sampled decks
  deckspace.py propose MODEL.pt --saves DIR --out sets.json [--per-save N]  model-guided deck search
  deckspace.py fight sets.json --saves DIR --shard k/n --out F.jsonl        engine fights of proposed decks
  deckspace.py report F.jsonl [...]                                          what the data says
Rows are deckvalue.py rows (+ family, rep, graded).
"""
import argparse
import glob
import hashlib
import json
import os
import random
import statistics as S
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STRIKE, DEFEND, BASH = "STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"
BASICS = (STRIKE, DEFEND)


def load_saves(d):
    metas = []
    for p in sorted(glob.glob(os.path.join(HERE, d, "*.save.json"))):
        with open(p) as fh:
            m = json.load(fh)
        m["path"] = p[:-5]
        metas.append(m)
    return metas


def pool_weights():
    """Ironclad card-reward pool with offer counts (how likely a card is to be seen)."""
    with open(os.path.join(HERE, "data", "offer_pool.json")) as fh:
        pool = json.load(fh)["pool"]
    tot = {}
    for a in pool:
        for c, n in pool[a].items():
            tot[c] = tot.get(c, 0) + n
    return {c: n for c, n in tot.items() if n >= 300}


# Human archetypes (Chinese and English high-ascension guides, 2026; see the research notes):
# a starting point for exploration, not an answer; the engine judges them like any other deck
GUIDE = {
    "loop": ["POMMEL_STRIKE", "SHRUG_IT_OFF", "BURNING_PACT", "BATTLE_TRANCE", "OFFERING", "PILLAGE", "BLOODLETTING",
             "FORGOTTEN_RITUAL", "EXPECT_A_FIGHT", "PYRE", "TRUE_GRIT", "THRASH", "SECOND_WIND", "ASHEN_STRIKE", "PACTS_END"],
    "selfdmg": ["RUPTURE", "TEAR_ASUNDER", "INFERNO", "CRIMSON_MANTLE", "BLOODLETTING", "OFFERING", "BRAND", "SPITE",
                "BREAKTHROUGH", "BLOOD_WALL", "HEMOKINESIS"],
    "exhaust": ["ASHEN_STRIKE", "DARK_EMBRACE", "FEEL_NO_PAIN", "TRUE_GRIT", "BURNING_PACT", "SECOND_WIND", "FIEND_FIRE",
                "STOKE", "HAVOC", "PACTS_END", "BRAND"],
    "vuln": ["VICIOUS", "TAUNT", "THUNDERCLAP", "UPPERCUT", "COLOSSUS", "DOMINATE", "BASH", "OFFERING", "PYRE"],
    "strike": ["HELLRAISER", "POMMEL_STRIKE", "PERFECTED_STRIKE", "PILLAGE", "TWIN_STRIKE", "SETUP_STRIKE", "ASHEN_STRIKE"],
    "thrash": ["THRASH", "SHRUG_IT_OFF", "TAUNT", "COLOSSUS", "IMPERVIOUS", "FLAME_BARRIER", "BASH", "BURNING_PACT"],
    "strength": ["DEMON_FORM", "INFLAME", "BRAND", "FIGHT_ME", "TWIN_STRIKE", "SWORD_BOOMERANG", "WHIRLWIND", "RUPTURE"],
}
GUIDE_BLOCK = ["UNMOVABLE", "CRIMSON_MANTLE", "COLOSSUS", "RAGE", "IMPERVIOUS", "FLAME_BARRIER", "FEEL_NO_PAIN", "TAUNT",
               "SHRUG_IT_OFF", "EVIL_EYE", "SECOND_WIND", "ARMAMENTS", "TRUE_GRIT"]
GUIDE_AOE = ["INFERNO", "CONFLAGRATION", "BREAKTHROUGH", "WHIRLWIND", "THUNDERCLAP", "STOMP"]


def guide_deck(pool, rng):
    arch = rng.choice(sorted(GUIDE))
    ok = lambda c: c in pool or c == "BASH"
    d = [STRIKE] * rng.randint(0, 5 if arch != "strike" else 5) + [DEFEND] * rng.randint(0, 4) + ([BASH] if rng.random() < 0.7 else [])
    for c in GUIDE[arch]:
        if ok(c) and rng.random() < 0.75:
            d += [c] * rng.choice((1, 1, 2))
    d += [c for c in rng.sample(GUIDE_BLOCK, rng.randint(2, 4)) if ok(c)]
    d += [c for c in rng.sample(GUIDE_AOE, rng.randint(1, 2)) if ok(c)]
    cards = list(pool)
    w = [pool[c] for c in cards]
    size = rng.randint(15, 32)
    while len(d) < size:
        d.append(rng.choices(cards, w)[0])
    up = rng.uniform(0.2, 1.0)
    return "guide_" + arch, sorted(c + "+" if rng.random() < up else c for c in d)


USE_GUIDE = False


def sample_deck(real, pool, rng):
    """One deck of a random shape. real: the save's deck (card tokens, "+" = upgraded)."""
    cards = list(pool)
    w = [pool[c] for c in cards]
    fams, wts = ["local", "thin", "random", "theme", "big"], [0.3, 0.15, 0.25, 0.2, 0.1]
    if USE_GUIDE:   # human archetypes: off by default (no human knowledge in the data; a control only)
        fams, wts = fams + ["guide"], [0.25, 0.12, 0.2, 0.15, 0.1, 0.18]
    fam = rng.choices(fams, wts)[0]
    if fam == "guide":
        return guide_deck(pool, rng)
    if fam == "local":   # the bot's deck, a few edits away
        d = list(real)
        for _ in range(rng.randint(1, 8)):
            op = rng.random()
            if op < 0.35:
                d.append(rng.choices(cards, w)[0])
            elif op < 0.65 and len(d) > 5:
                d.pop(rng.randrange(len(d)))
            elif op < 0.8:
                i = rng.randrange(len(d))
                d[i] = d[i] if d[i].endswith("+") else d[i] + "+"
            elif op < 0.9:
                d.append(rng.choice(d))
            else:
                d[rng.randrange(len(d))] = rng.choices(cards, w)[0]
    elif fam == "thin":   # the bot's deck without some or all starter cards (and maybe more)
        d = list(real)
        basics = [i for i, c in enumerate(d) if c.rstrip("+") in BASICS]
        drop = set(rng.sample(basics, rng.randint(1, len(basics)))) if basics else set()
        d = [c for i, c in enumerate(d) if i not in drop]
        for _ in range(rng.randint(0, 4)):
            if len(d) > 5:
                d.pop(rng.randrange(len(d)))
    else:
        size = {"random": rng.randint(6, 32), "theme": rng.randint(8, 28), "big": rng.randint(30, 45)}[fam]
        d = [STRIKE] * rng.randint(0, 5) + [DEFEND] * rng.randint(0, 5) + ([BASH] if rng.random() < 0.6 else [])
        if fam == "theme":   # a few core cards in several copies: concentrated synergies
            core = rng.choices(cards, w, k=rng.randint(2, 4))
            for c in core:
                d += [c] * rng.randint(2, 4)
        sub = rng.sample(cards, min(len(cards), rng.randint(5, 30)))
        sw = [pool[c] * rng.random() ** 2 for c in sub]   # uneven weights: some cards repeat
        while len(d) < size:
            d.append(rng.choices(sub, sw)[0])
        up = rng.random()
        d = [c + "+" if not c.endswith("+") and rng.random() < up else c for c in d[:45]]
    if len(d) < 5:
        d += [STRIKE] * (5 - len(d))
    return fam, sorted(d)


def deck_key(d):
    return hashlib.md5(",".join(sorted(d)).encode()).hexdigest()[:12]


def fight_row(sim, meta, deck, rep, nodes, rollouts, extra):
    """Engine fight of `deck` on the save -> deckvalue row."""
    import deckevo
    from policy import potion_id
    g, won, info = deckevo.fight(sim, meta, deck, rep, nodes, rollouts, detail=True)
    pl = info["player"]
    return {"save": meta["seed"], "act": meta.get("act"), "boss": meta.get("boss"), "deck": info.get("deck") or deck,
            "relics": [r.get("name") for r in pl.get("relics") or []], "hp": pl.get("hp"), "mhp": pl.get("max_hp"),
            "potions": [potion_id(x) or x.get("name") for x in pl.get("potions") or [] if x],
            "won": won, "hp_after": info["hp_after"], "left_frac": info["left_frac"], "graded": round(g, 2),
            "rounds": info["rounds"], "rep": rep, "dk": deck_key(deck), "bosses": info.get("bosses", 1),
            "engine": __import__("engine").ENGINE_CONFIG, **extra}


def run_items(items, args, fn_extra):
    import bossbench
    rollouts = json.loads(args.rollouts) if args.rollouts else None
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for l in fh:
                r = json.loads(l)
                done.add((r["save"], r["dk"], r["rep"]))
    items = [it for it in items if (it[0]["seed"], deck_key(it[1]), it[2]) not in done]
    lock = threading.Lock()

    def fn(sim, it):
        meta, deck, rep, extra = it
        try:
            r = fight_row(sim, meta, deck, rep, args.nodes, rollouts, extra)
        except Exception as e:
            r = {"save": meta["seed"], "dk": deck_key(deck), "rep": rep, "deck": deck, "err": f"{type(e).__name__}: {e}"[:200]}
        with lock, open(args.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        return r

    t0 = time.time()
    res = bossbench.parallel(items, fn, args.workers)
    ok = [r for r in res if "err" not in r]
    print(json.dumps({"fights": len(res), "errors": len(res) - len(ok), "wins": sum(r["won"] for r in ok),
                      "wall_s": round(time.time() - t0)}))


def real_deck(meta):
    from engine import fresh_sim
    sim = fresh_sim()
    try:
        sim.send({"cmd": "load_save", "path": meta["path"]})
        p = sim.send({"cmd": "set_player"}).get("player") or {}
    finally:
        sim.close()
    return sorted(c["id"].replace("CARD.", "") + ("+" if c.get("upgraded") else "") for c in p.get("deck") or [])


def cmd_explore(args):
    metas = load_saves(args.saves)
    if args.only:   # one save by name (a host only has the saves its jobs pushed)
        metas = [m for m in metas if m["seed"] == args.only]
    else:
        k, n = map(int, args.shard.split("/"))
        metas = [m for m in metas if int(hashlib.md5(m["seed"].encode()).hexdigest(), 16) % n == k]
    pool = pool_weights()
    items = []
    for m in metas:
        real = real_deck(m)
        rng = random.Random(f"{m['seed']}/{args.seed}")
        items.append((m, real, 0, {"family": "real"}))
        items.append((m, real, 1, {"family": "real"}))
        for i in range(args.per_save):
            fam, d = sample_deck(real, pool, rng)
            items.append((m, d, i % 2, {"family": fam}))
    run_items(items, args, None)


def cmd_fight(args):
    with open(args.sets) as fh:
        sets = json.load(fh)
    metas = {m["seed"]: m for m in load_saves(args.saves)}
    k, n = map(int, args.shard.split("/"))
    real = {}
    for s_ in sets:   # deck "real": the save's own deck
        if s_["deck"] == "real" and s_["save"] in metas and s_["save"] not in real:
            real[s_["save"]] = real_deck(metas[s_["save"]])
    items = [(metas[s["save"]], real[s["save"]] if s["deck"] == "real" else s["deck"], s.get("rep", 0), {"family": s.get("family", "proposed"),
             **({"pred": s["pred"]} if "pred" in s else {})})
             for i, s in enumerate(sets) if s["save"] in metas and i % n == k]
    run_items(items, args, None)


def mutate(d, pool, rng):
    cards = list(pool)
    w = [pool[c] for c in cards]
    d = list(d)
    for _ in range(rng.choice((1, 1, 2, 3, 4))):
        op = rng.random()
        if op < 0.3 and len(d) < 45:
            d.append(rng.choices(cards, w)[0])
        elif op < 0.55 and len(d) > 5:
            d.pop(rng.randrange(len(d)))
        elif op < 0.65:   # drop every copy of one card (all Strikes, say)
            c = rng.choice(d)
            if sum(x != c for x in d) >= 5:
                d = [x for x in d if x != c]
        elif op < 0.78:
            i = rng.randrange(len(d))
            d[i] = d[i] if d[i].endswith("+") else d[i] + "+"
        elif op < 0.88 and len(d) < 45:
            d.append(rng.choice(d))
        else:
            d[rng.randrange(len(d))] = rng.choices(cards, w)[0]
    return sorted(d)


def edits(real, d):
    """(cards added, cards removed, upgrades) turning `real` into `d` (an upgrade is not an add)."""
    from collections import Counter
    rb, db = Counter(c.rstrip("+") for c in real), Counter(c.rstrip("+") for c in d)
    adds = sum((db - rb).values())
    rems = sum((rb - db).values())
    ups = max(0, sum(c.endswith("+") for c in d) - sum(c.endswith("+") for c in real))
    return adds, rems, ups


def feasible(real, d, mode):
    """budget: what one act can do (<= 4 cards added, <= 3 removed, <= 4 upgrades);
    sized: any composition, but within 3 cards of the real deck's size (composition, not size)."""
    if mode == "budget":
        a, r, u = edits(real, d)
        return a <= 4 and r <= 3 and u <= 4
    if mode == "sized":
        return abs(len(d) - len(real)) <= 3
    return True


def edit_summary(real, d):
    from collections import Counter
    rc, dc = Counter(real), Counter(d)
    plus = sorted((dc - rc).elements())
    minus = sorted((rc - dc).elements())
    return f"+[{' '.join(plus)}] -[{' '.join(minus)}]"[:300]


def cmd_propose(args):
    """Model-guided deck search per save: evolve decks under an ensemble of deck models,
    score = mean + beta * spread (optimism where the models disagree: that is where the engine
    teaches them most), keep the best distinct decks for engine fights."""
    import numpy as np
    from deckvalue import DeckValue
    models = [DeckValue(p, device=args.device) for p in args.files]
    rows = [json.loads(l) for f in args.data for l in open(f)]
    ctx = {}
    for r in rows:
        if "err" not in r and r.get("family") == "real":
            ctx[r["save"]] = {k: r[k] for k in ("relics", "potions", "hp", "mhp", "act", "boss")}
            ctx[r["save"]]["real"] = r["deck"]
    pool = pool_weights()
    out = []
    for save, base in sorted(ctx.items()):
        rng = random.Random(f"{save}/propose/{args.seed}")

        def score(decks):
            st = [dict(base, deck=d) for d in decks]
            p = np.stack([m.score(st) for m in models])
            return p.mean(0), p.std(0)

        real = base["real"]
        for mode in args.modes.split(","):
            ok = lambda d, mode=mode: feasible(real, d, mode)
            popn = [real]
            while len(popn) < args.pop:
                d = mutate(rng.choice(popn), pool, rng)
                if ok(d):
                    popn.append(d)
            seen = {}
            for gen in range(args.gens):
                kids = []
                for _ in range(args.pop * 6):
                    d = mutate(rng.choice(popn), pool, rng)
                    if ok(d):
                        kids.append(d)
                    if len(kids) >= args.pop * 2:
                        break
                cand = {",".join(d): d for d in popn + kids}
                keys = [k for k in cand if k not in seen]
                if keys:
                    mu, sd = score([cand[k] for k in keys])
                    for k, a, b in zip(keys, mu, sd):
                        seen[k] = (float(a), float(b))
                ranked = sorted(cand, key=lambda k: -(seen[k][0] + args.beta * seen[k][1]))
                popn = [cand[k] for k in ranked[:args.pop]]
            mu_real = seen[",".join(real)][0]
            best = [k for k in sorted(seen, key=lambda k: -(seen[k][0] + args.beta * seen[k][1])) if k != ",".join(real)][:args.per_save]
            greedy = [k for k in sorted(seen, key=lambda k: -seen[k][0]) if k not in best and k != ",".join(real)][:max(1, args.per_save // 4)]
            for k in best + greedy:
                for rep in (0, 1):
                    out.append({"save": save, "deck": k.split(","), "rep": rep, "family": f"prop_{mode}",
                                "pred": [round(seen[k][0], 3), round(seen[k][1], 3), round(mu_real, 3)]})
            b0 = best[0].split(",")
            print(f"{save} {mode}: real {mu_real:.3f}  best {seen[best[0]][0]:.3f}±{seen[best[0]][1]:.3f} size {len(real)}->{len(b0)} "
                  f"{edit_summary(real, b0)}", flush=True)
        for rep in (0, 1):   # the real deck on the same futures, as the reference
            out.append({"save": save, "deck": real, "rep": rep, "family": "real"})
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"{len(out)} fights proposed -> {args.out}")


def cmd_report(args):
    rows = [json.loads(l) for f in args.files for l in open(f)]
    err = [r for r in rows if "err" in r]
    rows = [r for r in rows if "err" not in r]
    print(f"{len(rows)} fights, {len(err)} errors, {len({r['save'] for r in rows})} saves, wins {sum(r['won'] for r in rows)}")
    for e in err[:3]:
        print("  err", e["err"][:150])
    real = {}
    for r in rows:
        if r.get("family") == "real":
            real.setdefault(r["save"], []).append(r["graded"])
    fams = {}
    for r in rows:
        if r["save"] in real:
            fams.setdefault(r.get("family"), []).append((r["graded"] - S.mean(real[r["save"]]), r["won"]))
    print("by family (graded minus the save's real deck; win share):")
    for f, v in sorted(fams.items(), key=lambda x: -S.mean(y[0] for y in x[1])):
        print(f"  {f:10s} n {len(v):5d}  {S.mean(y[0] for y in v):+6.1f}  wins {sum(y[1] for y in v) / len(v):.1%}")
    by_boss = {}
    for r in rows:
        by_boss.setdefault(r.get("boss"), []).append(r)
    print("best decks per boss (beat the real deck on the same save):")
    for b, rs in sorted(by_boss.items()):
        wins = [r for r in rs if r["won"]]
        best = sorted(rs, key=lambda r: -(r["graded"] - S.mean(real.get(r["save"], [0]))))[:2]
        print(f"  {b}: {len(wins)}/{len(rs)} wins ({sum(1 for r in rs if r.get('family') == 'real' and r['won'])} by real decks)")
        for r in best:
            basics = sum(1 for c in r["deck"] if c.rstrip("+") in BASICS)
            print(f"     {r['family']:7s} graded {r['graded']:.0f} (real {S.mean(real.get(r['save'], [0])):.0f}) size {len(r['deck'])} "
                  f"basics {basics}: {' '.join(sorted(set(c for c in r['deck'] if c.rstrip('+') not in BASICS)))[:200]}")
    # deck shape vs outcome (relative to the save's real deck)
    for name, fn in (("size", lambda r: len(r["deck"]) // 5 * 5), ("basics", lambda r: min(10, sum(1 for c in r["deck"] if c.rstrip("+") in BASICS)) // 2 * 2),
                     ("upgraded share", lambda r: round(sum(c.endswith("+") for c in r["deck"]) / len(r["deck"]) * 4) / 4)):
        g = {}
        for r in rows:
            if r["save"] in real:
                g.setdefault(fn(r), []).append(r["graded"] - S.mean(real[r["save"]]))
        print(f"{name}: " + "  ".join(f"{k}: {S.mean(v):+.1f} (n {len(v)})" for k, v in sorted(g.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["explore", "fight", "report", "propose"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--saves", default="data/deckspace_saves")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default=None, help="explore: just this save (its seed name)")
    ap.add_argument("--guide", action="store_true", help="explore: also sample human-archetype decks (a control)")
    ap.add_argument("--per-save", type=int, default=150)
    ap.add_argument("--out")
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--rollouts", default='{"k":4,"m":4,"turns":2}')
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine-config", default="Fix2", help="Fix2+: act-3 saves fight both A10 final bosses")
    ap.add_argument("--data", nargs="*", default=[], help="propose: explore rows (the saves' contexts)")
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--modes", default="budget,sized", help="propose: budget, sized and/or free (see feasible())")
    args = ap.parse_args()
    import engine
    engine.ENGINE_CONFIG = args.engine_config
    global USE_GUIDE
    USE_GUIDE = args.guide
    if args.cmd == "explore":
        cmd_explore(args)
    elif args.cmd == "fight":
        args.sets = args.files[0]
        cmd_fight(args)
    elif args.cmd == "report":
        cmd_report(args)
    else:
        cmd_propose(args)


if __name__ == "__main__":
    sys.exit(main())
