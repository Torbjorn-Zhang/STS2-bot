"""Combo miner: find card (and relic) sets whose value is more than the sum of their parts,
including combos nobody has named, by measuring them in the real game engine.

Every probe takes a boss save, replaces the deck with the starter deck plus the candidate set,
gives the player a huge HP pool (so the fight measures the deck's output, not whether it
survives), and plays T rounds with the planner. Per probe it records the boss damage dealt and
the HP lost (both per round), and the most actions in one turn (loops and infinites show up
as huge turns). Deck power is L = log((damage rate + 10) / (HP-loss rate + 5)): boss HP removed
per HP lost, the trade a real fight is won or lost on. Synergy of a pair a, b is
    L(a+b) - L(a) - L(b) + L(starter)
averaged over the same saves (and the same hidden futures) for every set.

usage:
  combolab.py sets --out data/combo1_sets.json [--pairs]     write the candidate sets
  combolab.py run SETS --saves DIR --shard k/n --out F.jsonl  probe (one fleet task per shard)
  combolab.py report F.jsonl [...]                            singles, top synergies, loops
"""
import argparse
import glob
import hashlib
import json
import math
import os
import statistics as S
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
STARTER = ["STRIKE_IRONCLAD"] * 5 + ["DEFEND_IRONCLAD"] * 4 + ["BASH"]
GOD_HP = 5000


class Stop(Exception):
    pass


def ironclad_pool(min_offers=300):
    """Cards the Ironclad is offered (card rewards, all acts), most offered first."""
    with open(os.path.join(HERE, "data", "offer_pool.json")) as fh:
        pool = json.load(fh)["pool"]
    tot = {}
    for a in pool:
        for c, n in pool[a].items():
            tot[c] = tot.get(c, 0) + n
    return [c for c, n in sorted(tot.items(), key=lambda x: -x[1]) if n >= min_offers]


def make_sets(pairs):
    cards = ironclad_pool()
    sets = [{"id": "base", "cards": []}]
    sets += [{"id": c, "cards": [c]} for c in cards]
    sets += [{"id": f"{c}x2", "cards": [c, c]} for c in cards]
    if pairs:
        sets += [{"id": f"{a}+{b}", "cards": [a, b]} for i, a in enumerate(cards) for b in cards[i + 1:]]
    return sets


def probe(sim, meta, cset, T, nodes, rollouts, rep=0):
    from combat import TurnPlanner
    st = sim.send({"cmd": "load_save", "path": meta["path"]})
    if st.get("type") == "error":
        return {"err": f"load: {st.get('message')}"}
    r = sim.send({"cmd": "set_player", "deck": STARTER + cset["cards"]})
    have = [c.get("id", "").replace("CARD.", "") for c in (r.get("player") or {}).get("deck") or []]
    if sorted(have) != sorted(STARTER + cset["cards"]):   # set_player skips unknown ids silently
        return {"err": f"deck mismatch: {sorted(set(STARTER + cset['cards']) - set(have))}"}
    for r in cset.get("relics") or []:
        sim.send({"cmd": "set_player", "add_relics": [r]})
    sim.send({"cmd": "set_player", "hp": GOD_HP})
    col, row = meta["node"]
    st = sim.act("select_map_node", col=col, row=row)
    sim.send({"cmd": "determinize", "seed": zlib.crc32(f"{meta['seed']}/combo/{rep}".encode())})
    planner = TurnPlanner(sim, max_nodes=nodes, rollouts=rollouts)
    acc = {"dmg": 0.0, "taken": 0.0, "round": 1, "acts": {}, "hp": None, "e": {}}

    def on_state(s):
        rnd = s.get("round") or acc["round"]
        if rnd > T:
            raise Stop()
        acc["round"] = rnd
        p = (s.get("player") or {}).get("hp")
        if p is not None:
            if acc["hp"] is not None and p < acc["hp"]:
                acc["taken"] += acc["hp"] - p
            acc["hp"] = p
        if s.get("enemies") is not None:
            now = {e.get("id"): (e.get("hp") or 0) for e in s["enemies"]}
            for k, h0 in acc["e"].items():   # damage: every drop in an enemy's HP (gone = killed)
                h1 = now.get(k, 0)
                if h1 < h0:
                    acc["dmg"] += h0 - h1
            acc["e"] = now

    def on_step(s, a):
        acc["acts"][acc["round"]] = acc["acts"].get(acc["round"], 0) + 1

    on_state(st)
    t0 = time.perf_counter()
    killed = False
    try:
        end = planner.play_combat(st, on_state=on_state, on_step=on_step)
        killed = end.get("decision") != "game_over" or bool(end.get("victory"))
    except Stop:
        pass
    rounds = acc["round"]
    return {"dmg": round(acc["dmg"], 1), "taken": round(acc["taken"], 1), "rounds": rounds, "killed": killed,
            "max_acts": max(acc["acts"].values()) if acc["acts"] else 0, "secs": round(time.perf_counter() - t0, 1)}


def power(r):
    n = max(1, r["rounds"])
    return math.log((r["dmg"] / n + 10) / (r["taken"] / n + 5))


def cmd_run(args):
    import bossbench
    with open(args.sets) as fh:
        sets = json.load(fh)
    metas = []
    for p in sorted(glob.glob(os.path.join(HERE, args.saves, "*.save.json"))):
        with open(p) as fh:
            m = json.load(fh)
        m["path"] = p[:-5]
        metas.append(m)
    k, n = map(int, args.shard.split("/"))
    items = [(c, m) for c in sets for m in metas
             if int(hashlib.md5(f"{c['id']}/{m['seed']}".encode()).hexdigest(), 16) % n == k]
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done = {(r["set"], r["save"]) for r in map(json.loads, fh)}
    items = [it for it in items if (it[0]["id"], it[1]["seed"]) not in done]
    rollouts = json.loads(args.rollouts) if args.rollouts else None
    import threading
    lock = threading.Lock()

    def fn(sim, it):
        c, m = it
        try:
            r = probe(sim, m, c, args.turns, args.nodes, rollouts)
        except Exception as e:
            r = {"err": f"{type(e).__name__}: {e}"[:200]}
        r.update(set=c["id"], cards=c["cards"], relics=c.get("relics") or [], save=m["seed"], boss=m.get("boss"))
        with lock, open(args.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        return r

    t0 = time.time()
    res = bossbench.parallel(items, fn, args.workers)
    ok = [r for r in res if "err" not in r]
    print(json.dumps({"probes": len(res), "errors": len(res) - len(ok), "wall_s": round(time.time() - t0),
                      "mean_s": round(S.mean(r["secs"] for r in ok), 1) if ok else None}))


def cmd_report(args):
    rows = [json.loads(l) for f in args.files for l in open(f)]
    errs = [r for r in rows if "err" in r]
    rows = [r for r in rows if "err" not in r]
    L = {}
    for r in rows:
        L.setdefault(r["set"], {})[r["save"]] = r
    base = L.get("base", {})
    saves = sorted(base)
    print(f"{len(rows)} probes, {len(errs)} errors, {len(saves)} saves, {len(L)} sets")
    for e in errs[:5]:
        print("  err", e["set"], e["save"], e["err"][:120])

    def lp(sid):
        d = L.get(sid, {})
        return {s: power(d[s]) for s in saves if s in d}

    lb = lp("base")
    single = {}
    for sid in L:
        if sid == "base" or "+" in sid:
            continue
        l = lp(sid)
        diff = [l[s] - lb[s] for s in l if s in lb]
        if diff:
            single[sid] = (S.mean(diff), S.stdev(diff) / len(diff) ** 0.5 if len(diff) > 1 else 0, len(diff))
    print("\nsingle cards (power gain over the starter deck, log units):")
    for sid, (m, se, n) in sorted(single.items(), key=lambda x: -x[1][0])[:args.top]:
        print(f"  {sid:24s} {m:+.3f}±{se:.3f} (n {n})")
    syn = []
    for sid in L:
        if "+" not in sid:
            continue
        a, b = sid.split("+")
        la, lb2, lab = lp(a), lp(b), lp(sid)
        d = [lab[s] - la[s] - lb2[s] + lb[s] for s in lab if s in la and s in lb2 and s in lb]
        if len(d) >= 2:
            syn.append((S.mean(d), S.stdev(d) / len(d) ** 0.5, len(d), sid, S.mean(lab[s] - lb[s] for s in lab if s in lb)))
    if syn:
        print("\npairs with the most synergy (pair gain beyond the two single gains):")
        for m, se, n, sid, g in sorted(syn, reverse=True)[:args.top]:
            print(f"  {sid:36s} synergy {m:+.3f}±{se:.3f}  pair gain {g:+.3f} (n {n})")
    loops = sorted((r["max_acts"], r["set"], r["save"]) for r in rows)[-10:]
    print("\nlongest turns (actions in one turn; loops/infinites):")
    for a, sid, sv in reversed(loops):
        print(f"  {a:4d}  {sid}  {sv}")
    kills = {}
    for r in rows:
        kills.setdefault(r["set"], []).append(r["killed"])
    top = sorted(((sum(v) / len(v), k) for k, v in kills.items()), reverse=True)[:10]
    print("\nsets that kill the boss within the rounds most often:", ", ".join(f"{k} {f:.0%}" for f, k in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sets", "run", "report"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--saves", default="data/combo_saves")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--rollouts", default='{"k":4,"m":4,"turns":2}')
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    if args.cmd == "sets":
        sets = make_sets(args.pairs)
        with open(args.out, "w") as fh:
            json.dump(sets, fh)
        print(f"{len(sets)} sets -> {args.out}")
    elif args.cmd == "run":
        args.sets = args.files[0]
        cmd_run(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
