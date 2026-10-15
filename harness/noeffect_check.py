"""Counterfactual check for cards the scan flagged as "no visible effect".

Same setup as scan.py. Branch A plays the card (resolving prompts) then ends the turn;
branch B just ends the turn. Next-turn states are compared on a wider signature
(energy, hand card costs/stats, powers, piles...). Any difference means the card did
something; identical outcomes are listed with their card text for manual reading.
"""

import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from simclient import CLI, Sim, legal_actions

ENC = "SLIMES_WEAK"
TEXT = json.load(open(os.path.join(CLI, "localization_eng", "cards.json")))


def wide(st, cid):
    p = st.get("player") or {}
    return {
        "decision": st.get("decision"),
        "round": st.get("round"),
        "energy": st.get("energy"), "max_energy": st.get("max_energy"),
        "enemies": [(e.get("hp"), e.get("block"), json.dumps(e.get("powers"), sort_keys=True), json.dumps(e.get("intents"), sort_keys=True))
                    for e in st.get("enemies") or []],
        "hp": p.get("hp"), "block": p.get("block"), "gold": p.get("gold"), "max_hp": p.get("max_hp"),
        "player_powers": json.dumps(st.get("player_powers"), sort_keys=True),
        "hand": sorted(json.dumps([c.get("id"), c.get("cost"), c.get("stats")], sort_keys=True)
                       for c in st.get("hand") or [] if c.get("id") != cid),
        "draw": st.get("draw_pile_count"),
        "osty": json.dumps(st.get("osty"), sort_keys=True), "orbs": json.dumps(st.get("orbs"), sort_keys=True),
        "stars": json.dumps(st.get("stars"), sort_keys=True), "potions": json.dumps(p.get("potions"), sort_keys=True),
        "deck_size": p.get("deck_size"),
    }


def branch(card, char, play):
    sim = Sim("Debug", timeout=20)
    try:
        sim.send({"cmd": "start_run", "character": char.capitalize(), "seed": "scan"})
        filler = [f"STRIKE_{char}"] * 4 + [f"DEFEND_{char}"] * 4
        sim.send({"cmd": "set_player", "hp": 999, "max_hp": 999, "deck": [card, card] + filler})
        st = sim.send({"cmd": "enter_room", "type": "combat", "encounter": ENC})
        cid = "CARD." + card
        for _ in range(5):
            if any(c.get("id") == cid for c in st.get("hand", [])):
                break
            st = sim.act("end_turn")
        c = next(c for c in st["hand"] if c.get("id") == cid)
        if play:
            args = {"card_index": c["index"]}
            if c.get("target_type") == "AnyEnemy":
                args["target_index"] = 0
            st = sim.act("play_card", **args)
            for _ in range(6):
                if st.get("type") == "error" or st.get("decision") in ("combat_play", "game_over", None):
                    break
                a = legal_actions(st)[0]
                st = sim.act(a[0], **a[1])
        after_play = wide(st, cid)
        nxt = sim.act("end_turn")
        return after_play, wide(nxt, cid)
    finally:
        sim.close()


def check(job):
    card, char = job
    try:
        a_play, a_next = branch(card, char, True)
        b_play, b_next = branch(card, char, False)
    except Exception as e:
        return {"card": card, "char": char, "error": repr(e)}
    now = [k for k in a_play if a_play[k] != b_play[k] and k != "energy"]
    if a_play["energy"] != b_play["energy"]:
        now.append("energy_spent")
    later = [k for k in a_next if a_next[k] != b_next[k]]
    return {"card": card, "char": char, "diff_now": now, "diff_next_turn": later}


def main():
    flagged = defaultdict(list)
    for line in open("scan_results.jsonl"):
        r = json.loads(line)
        if r["status"] == "no_visible_effect":
            flagged[r["card"]].append(r["char"])
    jobs = [(c, ch) for c, chs in sorted(flagged.items()) for ch in chs]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(check, jobs))
    json.dump(res, open("noeffect_results.json", "w"), indent=1)
    by_card = defaultdict(list)
    for r in res:
        by_card[r["card"]].append(r)
    real_effect, nothing, errors = [], [], []
    for card, rs in sorted(by_card.items()):
        if any(r.get("error") for r in rs):
            errors.append((card, [r.get("error") for r in rs if r.get("error")][:1]))
        # effect = anything beyond paying the energy cost, now or next turn
        eff = [r for r in rs if (set(r.get("diff_now", [])) - {"energy_spent"}) or r.get("diff_next_turn")]
        if eff:
            real_effect.append((card, sorted({k for r in eff for k in r.get("diff_now", []) + r.get("diff_next_turn", [])})))
        else:
            nothing.append((card, TEXT.get(card + ".description", "")[:150].replace("\n", " ")))
    print("flagged cards:", len(by_card), "| pairs:", len(res))
    print(f"\n== effect found with wider signature / next turn: {len(real_effect)}")
    for x in real_effect:
        print("  ", x)
    print(f"\n== still no effect in this setup (read the text): {len(nothing)}")
    for x in nothing:
        print("  ", x)
    print(f"\n== errors: {errors}")


if __name__ == "__main__":
    main()
