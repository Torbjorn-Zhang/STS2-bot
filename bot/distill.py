"""Distill the planner into a fast action scorer for rollouts and the deck test.

The planner's chosen actions are logged with features of every legal action at that
moment (collect mode). A linear ranking model is fit on "chosen beats the others"
pairs (train mode). DistilledPolicy then plays like quick_policy, one sim call per
action, but picks argmax w . phi(state, action).

    distill.py collect N --ascension 10 --workers 2      # appends data/distill.jsonl
    distill.py train [--out models/distill.json]
"""

import argparse
import json
import os
import random
import threading
import time

import numpy as np

from combat import alive, card_numbers, future_weight, incoming_damage, legal_actions, powers, quick_policy

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = ("Attack", "Skill", "Power", "Status", "Curse")
NUM = ["is_end", "cost", "free", "energy_after", "dmg_target", "dmg_total", "aoe", "kills", "kills_attacker",
       "target_hp_frac", "target_threat", "block", "block_needed", "block_excess", "draw", "energy_gain", "hploss",
       "vuln", "weak", "strength", "power_f", "exhaust", "round", "hand", "need", "energy", "playables",
       "illusion", "minion", "leader"]


def card_ids():
    from cardlab import IRONCLAD
    return IRONCLAD + ["SLIMED", "WOUND", "BURN", "DAZED", "INFECTION", "ASCENDERS_BANE"]


CARDS = card_ids()
FEATURES = NUM + [f"t_{t}" for t in TYPES] + [f"c_{c}" for c in CARDS]
INDEX = {f: i for i, f in enumerate(FEATURES)}


def phi(st, a):
    """Feature vector of action a in combat state st."""
    x = np.zeros(len(FEATURES))
    p = st.get("player") or {}
    energy = st.get("energy") or 0
    inc = incoming_damage(st)
    need = max(0, inc - (p.get("block") or 0))
    hand = {c["index"]: c for c in st.get("hand") or []}
    playables = sum(1 for b in legal_actions(st) if b[0] == "play_card")
    common = {"round": min(st.get("round") or 1, 15) / 15, "hand": len(hand) / 10, "need": need / 30,
              "energy": energy / 5, "playables": playables / 10}
    for k, v in common.items():
        x[INDEX[k]] = v
    if a[0] == "end_turn":
        x[INDEX["is_end"]] = 1.0
        return x
    c = hand.get(a[1].get("card_index"))
    if c is None:
        return x
    stt = c.get("stats") or {}
    cost = c.get("cost")
    cost = energy if not isinstance(cost, int) or cost < 0 else cost
    en = {e["index"]: e for e in alive(st)}
    minions = any("Minion" in powers(e.get("powers")) for e in en.values())
    dmg, hits, blk = card_numbers(c, energy)
    desc = (c.get("description") or "").lower()
    strength = powers(st.get("player_powers")).get("Strength", 0)
    aoe = "all enemies" in desc
    tgt = en.get(a[1].get("target_index")) or (list(en.values())[0] if en else None)
    dtot, kills, kills_att = 0.0, 0, 0
    for e in (en.values() if aoe else [tgt] if tgt else []):
        vul = 1.5 if powers(e.get("powers")).get("Vulnerable") else 1.0
        d = (dmg + strength) * hits * vul if dmg else 0.0
        eff = (e.get("hp") or 0) + (e.get("block") or 0)
        dtot += min(d, eff)
        if d and d >= eff:
            kills += 1
            kills_att += 1 if e.get("intends_attack") else 0
    tpw = powers(tgt.get("powers")) if tgt else {}
    threat = sum(i["damage"] * (i.get("hits") or i.get("repeat") or 1) for i in (tgt or {}).get("intents") or []
                 if i.get("type") in ("Attack", "DeathBlow") and i.get("damage") is not None) if tgt else 0
    f, _ = future_weight(st)
    vals = {"cost": cost / 3, "free": float(cost == 0), "energy_after": (energy - cost) / 5,
            "dmg_target": min(dtot, 60) / 30 if not aoe else 0.0, "dmg_total": min(dtot, 80) / 30, "aoe": float(aoe),
            "kills": kills, "kills_attacker": kills_att,
            "target_hp_frac": ((tgt or {}).get("hp") or 0) / max(1, (tgt or {}).get("max_hp") or 1) if tgt else 0.0,
            "target_threat": threat / 30, "block": blk / 20, "block_needed": min(blk, need) / 20,
            "block_excess": max(0, blk - need) / 20, "draw": (stt.get("cards") or 0) / 3,
            "energy_gain": (stt.get("energy") or 0) / 2, "hploss": (stt.get("hploss") or 0) / 6,
            "vuln": (stt.get("vulnerablepower") or 0) / 3, "weak": (stt.get("weakpower") or stt.get("power") or 0) / 3 if c.get("type") != "Power" else 0.0,
            "strength": (stt.get("strengthpower") or 0) / 3, "power_f": f / 6 if c.get("type") == "Power" else 0.0,
            "exhaust": float("Exhaust" in (c.get("keywords") or [])),
            "illusion": float("Illusion" in tpw), "minion": float("Minion" in tpw),
            "leader": float(minions and tgt is not None and "Minion" not in tpw)}
    for k, v in vals.items():
        x[INDEX[k]] = v
    if c.get("type") in TYPES:
        x[INDEX[f"t_{c['type']}"]] = 1.0
    cid = (c.get("id") or "").replace("CARD.", "")
    if f"c_{cid}" in INDEX:
        x[INDEX[f"c_{cid}"]] = 1.0
    return x


class DistilledPolicy:
    def __init__(self, path=os.path.join(HERE, "models", "distill.json")):
        with open(path) as fh:
            blob = json.load(fh)
        assert blob["features"] == FEATURES, "feature list changed since training"
        self.w = np.array(blob["w"])

    def __call__(self, st):
        if st.get("decision") != "combat_play":
            return quick_policy(st)
        acts = legal_actions(st)
        if not acts:
            return ("end_turn", {})
        scores = [float(self.w @ phi(st, a)) for a in acts]
        return acts[int(np.argmax(scores))]


# ---------------------------------------------------------------- collection

def record_planner(planner, out, lock):
    """Log every executed step of the planner's paths: (state, all legal actions, the one
    the planner played). Steps are matched to states through play_combat's on_state."""
    real_play = planner.play_combat

    def log(state, action):
        if state.get("decision") != "combat_play" or action[0] not in ("play_card", "end_turn"):
            return
        acts = legal_actions(state)
        keys = [json.dumps(a, sort_keys=True) for a in acts]
        key = json.dumps(action, sort_keys=True)
        if key in keys and len(acts) > 1:
            X = [phi(state, a).round(4).tolist() for a in acts]
            with lock:
                out.write(json.dumps({"chosen": keys.index(key), "X": X}) + "\n")

    def play_combat(state, max_actions=600, on_state=None):
        cur = {"path": [], "step": 0}
        real_plan = planner.plan

        def plan(root):
            path, score = real_plan(root)
            cur["path"], cur["step"] = path, 0
            if path:
                log(root, path[0])
            return path, score

        def watch(s):
            cur["step"] += 1
            if cur["step"] < len(cur["path"]):
                log(s, cur["path"][cur["step"]])
            if on_state:
                on_state(s)

        planner.plan = plan
        try:
            return real_play(state, max_actions=max_actions, on_state=watch)
        finally:
            planner.plan = real_plan

    planner.play_combat = play_combat


def collect(n, ascension, workers, prefix):
    import policy
    from engine import fresh_sim, maybe_recycle
    import run as R
    out = open(os.path.join(HERE, "data", "distill.jsonl"), "a")
    lock = threading.Lock()
    seeds = [f"{prefix}_{i}" for i in range(n)]
    orig = R.TurnPlanner

    def make(sim, **kw):
        pl = orig(sim, **kw)
        record_planner(pl, out, lock)
        return pl

    R.TurnPlanner = make

    def worker(chunk):
        sim = fresh_sim()
        for s in chunk:
            sim = maybe_recycle(sim)
            try:
                R.play_run(sim, s, "Ironclad", 3, 400, ascension=ascension, rollouts={"k": 6, "m": 4, "turns": 3})
            except Exception as e:
                print("error", s, repr(e)[:120], flush=True)
                sim = fresh_sim()
            print("done", s, flush=True)
        sim.close()

    threads = [threading.Thread(target=worker, args=(seeds[i::workers],)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


# ---------------------------------------------------------------- training

def train(out_path, c_reg=1.0):
    from sklearn.linear_model import LogisticRegression
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "distill.jsonl"))]
    random.Random(0).shuffle(rows)
    split = int(len(rows) * 0.8)
    tr, te = rows[:split], rows[split:]
    Xd, yd = [], []
    for r in tr:
        X = np.array(r["X"]); c = r["chosen"]
        for j in range(len(X)):
            if j != c:
                Xd.append(X[c] - X[j]); yd.append(1)
                Xd.append(X[j] - X[c]); yd.append(0)
    m = LogisticRegression(C=c_reg, fit_intercept=False, max_iter=2000)
    m.fit(np.array(Xd), np.array(yd))
    w = m.coef_[0]

    def acc(rs, scorer):
        return sum(int(np.argmax([scorer(np.array(x)) for x in r["X"]]) == r["chosen"]) for r in rs) / max(1, len(rs))

    report = {"decisions": len(rows), "train": len(tr), "test": len(te),
              "test_top1": round(acc(te, lambda x: float(w @ x)), 3),
              "test_top1_random": round(sum(1 / len(r["X"]) for r in te) / max(1, len(te)), 3)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"features": FEATURES, "w": w.tolist(), "report": report}, fh)
    print(json.dumps(report))
    top = sorted(zip(FEATURES, w), key=lambda t: -abs(t[1]))[:25]
    print("largest weights:", [(f, round(v, 2)) for f, v in top])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["collect", "train"])
    ap.add_argument("n", type=int, nargs="?", default=10)
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prefix", default="dist")
    ap.add_argument("--out", default=os.path.join(HERE, "models", "distill.json"))
    args = ap.parse_args()
    if args.mode == "collect":
        collect(args.n, args.ascension, args.workers, args.prefix)
    else:
        train(args.out)


if __name__ == "__main__":
    main()
