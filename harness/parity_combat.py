"""Parity check 2: step-by-step combat comparison, real game (via STS2MCP) vs sts2-cli.

Precondition: the real game is at the start of a fight, and `--save` is the real
current_run.save written on entering it. The sim loads that save and enters the
next fight from the map (same encounter, same RNG position). Both sides then get
the same deterministic policy: play the first affordable card in hand order at the
first living enemy, otherwise end the turn. After every action both states are
normalised and compared.
"""

import argparse
import json
import time
import urllib.request

from simclient import Sim

API = "http://localhost:15526/api/v1/singleplayer"


def real_get():
    with urllib.request.urlopen(API + "?format=json", timeout=10) as r:
        return json.load(r)


def real_post(body):
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def real_settled(timeout=30.0):
    """Poll until the real game is back in the player's play phase (or left combat) and stable."""
    t0, last = time.time(), None
    while time.time() - t0 < timeout:
        st = real_get()
        b = st.get("battle")
        ready = (b is None) or (b.get("is_play_phase") and b.get("turn") == "player")
        key = json.dumps(norm_real(st), sort_keys=True) if b else st.get("state_type")
        if ready and key == last:
            return st
        last = key if ready else None
        time.sleep(0.4)
    raise TimeoutError("real game did not settle")


def intent_tuple(t, dmg, hits=None):
    return (t, dmg, hits or 1)


def norm_real(st):
    b, p = st.get("battle") or {}, st.get("player") or {}
    enemies = []
    for e in b.get("enemies", []):
        intents = []
        for i in e.get("intents", []):
            lab = str(i.get("label") or "")
            if "x" in lab or "×" in lab:
                a, h = lab.replace("×", "x").split("x")[:2]
                intents.append(intent_tuple(i["type"], int(a), int(h)))
            else:
                intents.append(intent_tuple(i["type"], int(lab) if lab.isdigit() else None))
        enemies.append({"hp": e["hp"], "max_hp": e["max_hp"], "block": e["block"], "intents": intents,
                        "status": sorted((s.get("id") or s.get("name"), s.get("amount")) for s in e.get("status", []))})
    return {"round": b.get("round"), "enemies": enemies,
            "player": {"hp": p.get("hp"), "block": p.get("block"), "energy": p.get("energy"),
                       "hand": [c["id"] for c in p.get("hand", [])],
                       "draw": p.get("draw_pile_count"), "discard": p.get("discard_pile_count"),
                       "exhaust": p.get("exhaust_pile_count"),
                       "status": sorted((s.get("id") or s.get("name"), s.get("amount")) for s in p.get("status", []))}}


def norm_sim(st):
    p = st.get("player") or {}
    enemies = []
    for e in st.get("enemies") or []:
        intents = [intent_tuple(i["type"], i.get("damage"), i.get("hits")) for i in e.get("intents") or []]
        enemies.append({"hp": e["hp"], "max_hp": e["max_hp"], "block": e.get("block", 0), "intents": intents,
                        "status": sorted((s.get("id") or s.get("name"), s.get("amount")) for s in e.get("powers") or [])})
    return {"round": st.get("round"), "enemies": enemies,
            "player": {"hp": p.get("hp"), "block": p.get("block"), "energy": st.get("energy"),
                       "hand": [c["id"].replace("CARD.", "") for c in st.get("hand", [])],
                       "draw": st.get("draw_pile_count"), "discard": st.get("discard_pile_count"),
                       "exhaust": st.get("exhaust_pile_count"),
                       "status": sorted((s.get("id") or s.get("name"), s.get("amount")) for s in st.get("player_powers") or [])}}


def comparable(a, b):
    """Compare everything except power identifiers (languages differ); amounts must match."""
    diffs = []
    for k in ("round",):
        if a[k] != b[k]:
            diffs.append((k, a[k], b[k]))
    if len(a["enemies"]) != len(b["enemies"]):
        diffs.append(("enemies.len", len(a["enemies"]), len(b["enemies"])))
    for i, (x, y) in enumerate(zip(a["enemies"], b["enemies"])):
        for k in ("hp", "max_hp", "block"):
            if x[k] != y[k]:
                diffs.append((f"enemy{i}.{k}", x[k], y[k]))
        # sts2-cli only exports numbers for Attack intents (e.g. StatusCard count is null),
        # so non-attack intents are compared by type only.
        ix = [t if t[0] == "Attack" else (t[0],) for t in x["intents"]]
        iy = [t if t[0] == "Attack" else (t[0],) for t in y["intents"]]
        if ix != iy:
            diffs.append((f"enemy{i}.intents", x["intents"], y["intents"]))
        if sorted(v for _, v in x["status"]) != sorted(v for _, v in y["status"]):
            diffs.append((f"enemy{i}.status", x["status"], y["status"]))
    for k in ("hp", "block", "energy", "hand", "draw", "discard"):
        if a["player"][k] != b["player"][k]:
            diffs.append((f"player.{k}", a["player"][k], b["player"][k]))
    if sorted(v for _, v in a["player"]["status"]) != sorted(v for _, v in b["player"]["status"]):
        diffs.append(("player.status", a["player"]["status"], b["player"]["status"]))
    return diffs


def choose(n, sim_state, real_state):
    """Same policy for both sides, decided from the (equal) normalised state."""
    energy = n["player"]["energy"]
    hand_real = real_state["player"]["hand"]
    for i, c in enumerate(hand_real):
        cost = c.get("cost")
        cost = int(cost) if isinstance(cost, str) and cost.isdigit() else cost
        if c.get("can_play") and isinstance(cost, int) and cost <= energy:
            alive = [j for j, e in enumerate(n["enemies"]) if e["hp"] > 0]
            return ("play", i, c.get("target_type"), alive[0] if alive else None)
    return ("end",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", required=True)
    ap.add_argument("--max-steps", type=int, default=200)
    args = ap.parse_args()

    sim = Sim("Debug", timeout=60)
    st = sim.send({"cmd": "load_save", "path": args.save})
    ch = st["choices"][0]
    sst = sim.act("select_map_node", col=ch["col"], row=ch["row"])
    rst = real_settled()
    log = []
    for step in range(args.max_steps):
        if sst.get("decision") != "combat_play" or rst.get("state_type") not in ("monster", "elite", "boss"):
            print("combat over:", "sim", sst.get("decision"), "| real", rst.get("state_type"))
            break
        rn, sn = norm_real(rst), norm_sim(sst)
        diffs = comparable(rn, sn)
        log.append({"step": step, "real": rn, "sim": sn, "diffs": diffs})
        print(f"step {step:3d} round {rn['round']} enemy_hp {[e['hp'] for e in rn['enemies']]}/{[e['hp'] for e in sn['enemies']]} "
              f"player_hp {rn['player']['hp']}/{sn['player']['hp']} hand {len(rn['player']['hand'])} -> {'SAME' if not diffs else 'DIFF ' + json.dumps(diffs, ensure_ascii=False)[:400]}", flush=True)
        if diffs:
            break
        a = choose(rn, sst, rst)
        if a[0] == "play":
            _, idx, ttype, tgt = a
            body = {"action": "play_card", "card_index": idx}
            sargs = {"card_index": sst["hand"][idx]["index"]}
            if ttype == "AnyEnemy" and tgt is not None:
                body["target"] = rst["battle"]["enemies"][tgt]["entity_id"]
                sargs["target_index"] = sst["enemies"][tgt]["index"]
            r = real_post(body)
            sst = sim.act("play_card", **sargs)
        else:
            r = real_post({"action": "end_turn"})
            sst = sim.act("end_turn")
        if r.get("status") == "error":
            print("real action error:", r)
            break
        rst = real_settled()
    sim.close()
    json.dump(log, open(args.save + ".combat_parity.json", "w"), ensure_ascii=False, indent=1)
    print("final real state_type:", rst.get("state_type"), "| sim decision:", sst.get("decision"))
    if sst.get("decision") == "card_reward":
        print("sim card reward:", [c["id"] for c in sst.get("cards", [])], "gold", sst.get("gold_earned"))
    print("real after-combat keys:", list(rst.keys()))


if __name__ == "__main__":
    main()
