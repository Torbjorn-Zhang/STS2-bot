"""Full card scan on the headless engine.

For every card id x every character: fresh engine, 999 HP, deck = 2 copies of the
card + 8 Strike/Defend, enter a fixed 3-enemy fight (SLIMES_WEAK), play the card
when it shows up, resolve any follow-up prompt, then end the turn once.

Flags per case: error response, timeout, engine exit, exception/warning lines on
stderr after the play, or no visible state change from playing the card.
"""

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from simclient import CLI, Sim, legal_actions

CHARS = ["IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER"]
ENCOUNTER = "SLIMES_WEAK"
WARN_MARKERS = ("Exception", "[WARN]", "NullReference", "MissingMethod", "InvalidOperation",
                "InvalidCast", "KeyNotFound", "ArgumentOutOfRange", "not implemented")


def card_ids():
    d = json.load(open(os.path.join(CLI, "localization_eng", "cards.json")))
    return sorted({k.split(".")[0] for k in d})


def signature(st, card_id):
    """Everything playing a card could plausibly change, minus the played copy itself."""
    p = st.get("player", {}) or {}
    hand = Counter(c.get("id") for c in st.get("hand", []) or [])
    return {
        "enemies": [(e.get("name"), e.get("hp"), e.get("block"), json.dumps(e.get("powers"), sort_keys=True))
                    for e in st.get("enemies", []) or []],
        "hp": p.get("hp"), "block": p.get("block"), "max_hp": p.get("max_hp"), "gold": p.get("gold"),
        "player_powers": json.dumps(st.get("player_powers"), sort_keys=True),
        "draw": st.get("draw_pile_count"),
        "hand_other": sorted((k, v) for k, v in hand.items() if k != card_id),
        "osty": json.dumps(st.get("osty"), sort_keys=True),
        "orbs": json.dumps(st.get("orbs"), sort_keys=True),
        "stars": json.dumps(st.get("stars"), sort_keys=True),
        "potions": json.dumps(p.get("potions"), sort_keys=True),
        "deck_size": p.get("deck_size"),
    }


def warn_lines(lines):
    return [l[:300] for l in lines if any(m in l for m in WARN_MARKERS)]


def scan_one(card, char):
    res = {"card": card, "char": char, "status": None}
    sim = None
    try:
        sim = Sim("Release", timeout=20, capture_stderr=True)
        sim.send({"cmd": "start_run", "character": char.capitalize(), "seed": "scan"})
        filler = [f"STRIKE_{char}"] * 4 + [f"DEFEND_{char}"] * 4
        r = sim.send({"cmd": "set_player", "hp": 999, "max_hp": 999, "deck": [card, card] + filler})
        if r.get("player", {}).get("deck_size") != 10:
            res["status"] = "unknown_id"
            return res
        st = sim.send({"cmd": "enter_room", "type": "combat", "encounter": ENCOUNTER})
        if st.get("type") == "error":
            res.update(status="enter_error", message=st.get("message"))
            return res
        cid = "CARD." + card
        for turn in range(5):
            if st.get("decision") != "combat_play":
                res.update(status="left_combat_before_play", decision=st.get("decision"))
                return res
            mine = [c for c in st.get("hand", []) if c.get("id") == cid]
            if mine:
                break
            st = sim.act("end_turn")
        else:
            res["status"] = "never_drawn"
            return res
        c = mine[0]
        res.update(cost=c.get("cost"), target=c.get("target_type"), type=c.get("type"))
        if not c.get("can_play"):
            res["status"] = "cost_too_high" if isinstance(c.get("cost"), int) and c["cost"] > st.get("energy", 0) else "unplayable"
            return res
        before = signature(st, cid)
        mark = len(sim.stderr_lines)
        args = {"card_index": c["index"]}
        if c.get("target_type") == "AnyEnemy":
            args["target_index"] = 0
        after = sim.act("play_card", **args)
        prompts = []
        for _ in range(6):  # resolve follow-up prompts (choose a card, etc.)
            if after.get("type") == "error" or after.get("decision") in ("combat_play", "game_over", None):
                break
            prompts.append(after.get("decision"))
            acts = legal_actions(after)
            after = sim.act(acts[0][0], **acts[0][1])
        time.sleep(0.05)
        res["prompts"] = prompts
        res["warn_play"] = warn_lines(sim.stderr_lines[mark:])
        if after.get("type") == "error":
            res.update(status="play_error", message=after.get("message"))
            return res
        if after.get("decision") == "combat_play" or after.get("decision") == "game_over":
            changed = [k for k, v in signature(after, cid).items() if before.get(k) != v]
            res["changed"] = changed
            if after.get("decision") == "game_over":
                res["status"] = "combat_ended"
            else:
                res["status"] = "ok" if changed or prompts else "no_visible_effect"
                mark = len(sim.stderr_lines)
                et = sim.act("end_turn")
                time.sleep(0.05)
                res["warn_end_turn"] = warn_lines(sim.stderr_lines[mark:])
                if et.get("type") == "error":
                    res["end_turn_error"] = et.get("message")
        else:
            res.update(status="stuck_after_play", decision=after.get("decision"))
        return res
    except TimeoutError:
        res["status"] = "timeout"
        return res
    except RuntimeError as e:
        res.update(status="engine_exit", message=str(e))
        return res
    finally:
        if sim:
            res["stderr_tail"] = [l[:200] for l in sim.stderr_lines[-3:]] if res.get("status") in ("timeout", "engine_exit") else None
            sim.close()


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "scan_results.jsonl"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    cards = card_ids()
    jobs = [(c, ch) for c in cards for ch in CHARS]
    t0 = time.perf_counter()
    done = 0
    with open(out_path, "w") as f, ThreadPoolExecutor(workers) as ex:
        for res in ex.map(lambda j: scan_one(*j), jobs):
            f.write(json.dumps(res) + "\n")
            done += 1
            if done % 250 == 0:
                print(f"{done}/{len(jobs)} {time.perf_counter() - t0:.0f}s", flush=True)
    print("cards", len(cards), "jobs", len(jobs), f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
