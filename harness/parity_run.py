"""Parity check 3: several consecutive fights, each started from the real game's own save.

For each fight: the real game enters a combat room and writes current_run.save; the sim
loads that save and enters the same encounter (id taken from the save's encounter list at
the visited counter); then both play the same deterministic policy and are compared after
every action. After the fight, rewards are compared and the real game moves on to the
next Monster/Elite node.
"""

import glob
import json
import os
import shutil
import sys
import time

from parity_combat import choose, comparable, norm_real, norm_sim, real_get, real_post, real_settled
from simclient import Sim

def _find_run_save():
    """The real game's modded-profile current_run.save (STS2_RUN_SAVE overrides the search)."""
    if os.environ.get("STS2_RUN_SAVE"):
        return os.environ["STS2_RUN_SAVE"]
    base = os.path.expanduser("~/Library/Application Support/SlayTheSpire2/steam")
    hits = glob.glob(os.path.join(base, "*", "modded", "profile1", "saves", "current_run.save"))
    return max(hits, key=os.path.getmtime) if hits else os.path.join(base, "<steam-id>", "modded", "profile1", "saves", "current_run.save")


SAVE = _find_run_save()
# real-game saves stay local (gitignored): they contain the Steam profile id
PAR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parity")
COMBAT = ("monster", "elite", "boss")


def wait_state(pred, timeout=40):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = real_get()
        if pred(st):
            return st
        time.sleep(0.5)
    raise TimeoutError("waiting for real state")


def current_encounter(save, room_type):
    rooms = save["acts"][save["current_act_index"]]["rooms"]
    if room_type == "elite":
        return rooms["elite_encounter_ids"][rooms["elite_encounters_visited"]].split(".", 1)[1], "elite"
    if room_type == "boss":
        return rooms["boss_id"].split(".", 1)[1], "combat"
    return rooms["normal_encounter_ids"][rooms["normal_encounters_visited"]].split(".", 1)[1], "combat"


def fight(tag, room_type, resume_snap=None, resume_actions=None):
    time.sleep(1.0)
    snap = resume_snap or os.path.join(PAR, f"real_{tag}.save")
    if not resume_snap:
        shutil.copy(SAVE, snap)
    save = json.load(open(snap))
    enc, rtype = current_encounter(save, room_type)
    # Most faithful entry: drop the room's own coordinate from a copy of the save, load it,
    # and walk into that same coordinate from the map (same floor, same node, same RNG).
    # enter_room (debug) skips floor bookkeeping: rewards and random encounter make-up drift.
    alt = snap + ".prev_node"
    alt_save = json.load(open(snap))
    here = alt_save["visited_map_coords"].pop()
    json.dump(alt_save, open(alt, "w"))
    sim = Sim("Debug", timeout=60)
    st = sim.send({"cmd": "load_save", "path": alt})
    node = next((c for c in st.get("choices", []) if (c["col"], c["row"]) == (here["col"], here["row"])), None)
    if node:
        sst = sim.act("select_map_node", col=node["col"], row=node["row"])
        path = "same_node"
    else:
        sim.close()
        sim = Sim("Debug", timeout=60)
        sim.send({"cmd": "load_save", "path": snap})
        sst = sim.send({"cmd": "enter_room", "type": rtype, "encounter": enc})
        path = "enter_room"
    actions = []
    for a in resume_actions or []:
        sst = sim.act(a[0], **a[1])
        actions.append(a)
    rst = real_settled()
    steps, first_diff = len(actions), None
    while sst.get("decision") == "combat_play" and rst.get("state_type") in COMBAT and steps < 300:
        rn, sn = norm_real(rst), norm_sim(sst)
        d = comparable(rn, sn)
        for _ in range(4):  # the real API can lag behind animations (discard count, consumed powers): re-read
            if not d:
                break
            time.sleep(1.5)
            rst = real_get()
            rn = norm_real(rst)
            d = comparable(rn, sn)
        if d:
            first_diff = {"step": steps, "round": rn["round"], "diffs": d, "real": rn, "sim": sn}
            break
        a = choose(rn, sst, rst)
        if a[0] == "play":
            _, idx, ttype, tgt = a
            body, sargs = {"action": "play_card", "card_index": idx}, {"card_index": sst["hand"][idx]["index"]}
            if ttype == "AnyEnemy" and tgt is not None:
                body["target"] = rst["battle"]["enemies"][tgt]["entity_id"]
                sargs["target_index"] = sst["enemies"][tgt]["index"]
            r = real_post(body)
            sst = sim.act("play_card", **sargs)
            actions.append(("play_card", sargs))
        else:
            r = real_post({"action": "end_turn"})
            sst = sim.act("end_turn")
            actions.append(("end_turn", {}))
        if r.get("status") == "error":
            first_diff = {"step": steps, "real_error": r}
            break
        rst = real_settled()
        steps += 1
    result = {"actions": actions, "tag": tag, "encounter": enc, "sim_path": path, "steps": steps, "identical": first_diff is None, "first_diff": first_diff}
    # Rewards (only meaningful if the fight finished in lockstep)
    if first_diff is None and sst.get("decision") == "card_reward":
        rr = wait_state(lambda s: s.get("state_type") == "rewards")
        items = rr["rewards"]["items"]
        gold = next((i.get("gold_amount") for i in items if i["type"] == "gold"), None)
        result["gold"] = {"real": gold, "sim": sst.get("gold_earned")}
        result["hp_after"] = {"real": rr["player"]["hp"], "sim": sst.get("player", {}).get("hp")}
        card_idx = next((i["index"] for i in items if i["type"] == "card"), None)
        for i in sorted((i["index"] for i in items if i["type"] != "card"), reverse=True):
            real_post({"action": "claim_reward", "index": i})
            time.sleep(1)
        if card_idx is not None:
            rr = wait_state(lambda s: s.get("state_type") == "rewards")
            ci = next(i["index"] for i in rr["rewards"]["items"] if i["type"] == "card")
            real_post({"action": "claim_reward", "index": ci})
            cr = wait_state(lambda s: s.get("state_type") == "card_reward")
            result["card_reward"] = {"real": [c["id"] for c in cr.get("card_reward", cr)["cards"]],
                                     "sim": [c["id"].replace("CARD.", "") for c in sst.get("cards", [])]}
            real_post({"action": "select_card_reward", "card_index": 0})
            time.sleep(1.5)
        st = real_get()
        if st.get("state_type") == "rewards":
            real_post({"action": "proceed"})
    sim.close()
    return result


def go_to_next_fight():
    """From the real map, pick a Monster (else Elite) node; walk through simple non-combat rooms."""
    for _ in range(12):
        st = real_get()
        t = st.get("state_type")
        if t in COMBAT:
            return t
        if t == "map":
            opts = st["map"]["next_options"]
            pick = next((o for o in opts if o["type"] == "Monster"), None) or \
                   next((o for o in opts if o["type"] == "Elite"), None) or opts[0]
            real_post({"action": "choose_map_node", "index": pick["index"]})
        elif t == "rewards":
            real_post({"action": "proceed"})
        elif t == "card_reward":
            real_post({"action": "skip_card_reward"})
        elif t == "card_select":
            real_post({"action": "select_card", "index": 0})
            time.sleep(1)
            real_post({"action": "confirm_selection"})
        elif t == "event":
            r = real_post({"action": "choose_event_option", "index": 0})
            if r.get("status") == "error":
                real_post({"action": "advance_dialogue"})
        elif t == "rest_site":
            real_post({"action": "choose_rest_option", "index": 0})
            time.sleep(2)
            real_post({"action": "proceed"})
        else:
            real_post({"action": "proceed"})
        time.sleep(3)
    raise RuntimeError(f"could not reach a fight, state {real_get().get('state_type')}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    results = []
    for k in range(n):
        room = go_to_next_fight()
        try:
            res = fight(f"fight{k + 3}_{int(time.time())}", room)
        except Exception as e:  # keep partial results instead of losing the whole run
            res = {"tag": f"fight{k + 3}", "error": repr(e), "first_diff": None}
            print(json.dumps(res), flush=True)
            results.append(res)
            break
        results.append(res)
        print(json.dumps({x: res[x] for x in res if x != "first_diff"}, ensure_ascii=False), flush=True)
        if res["first_diff"]:
            print("FIRST DIFF:", json.dumps(res["first_diff"], ensure_ascii=False)[:1500], flush=True)
        st = real_get()
        if st.get("state_type") in ("game_over", "menu"):
            break
    json.dump(results, open(os.path.join(PAR, "parity_run_results.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
