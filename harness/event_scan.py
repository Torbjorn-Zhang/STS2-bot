"""Event scan: every event, every first-page option, played through to the map.

usage: event_scan.py [Release|Exp]
For each event id (localization) and each option on its first page: a fresh start_run,
enter_room event, choose the option, then follow the flow with simple choices (first
option, tier-free card picks, quick fights) until the map or game over. Reports error
responses, exception lines on stderr, and flows that never return to the map.
"""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bot"))
from simclient import CLI, Sim  # noqa: E402
from combat import quick_policy  # noqa: E402

MARKERS = ("Exception", "NullReference", "MissingMethod", "InvalidOperation")


def event_ids():
    d = json.load(open(os.path.join(CLI, "localization_eng", "events.json")))
    return sorted({k.split(".")[0] for k in d if k.endswith(".title") and "." not in k[:-6]})


def step(sim, st):
    d = st.get("decision")
    if d == "event_choice":
        opts = [o for o in st.get("options") or [] if not o.get("is_locked")]
        return sim.act("choose_option", option_index=opts[0]["index"]) if opts else sim.act("leave_room")
    if d == "card_reward":
        return sim.act("skip_card_reward")
    if d == "card_select":
        if (st.get("context") or {}).get("room_type") in ("Monster", "Elite", "Boss"):
            a = quick_policy(st)
            return sim.act(a[0], **a[1])
        cards, lo = st.get("cards") or [], st.get("min_select") or 0
        return sim.act("select_cards", indices=",".join(str(c.get("index", i)) for i, c in enumerate(cards[:max(1, lo)]))) \
            if lo or cards else sim.act("skip_select")
    if d == "combat_play":
        a = quick_policy(st)
        r = sim.act(a[0], **a[1])
        return sim.act("end_turn") if r.get("type") == "error" else r
    if d == "bundle_select":
        return sim.act("select_bundle", bundle_index=0)
    return sim.act("proceed")


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else "Release"
    sim = Sim(cfg, timeout=30, capture_stderr=True)
    results, summary = [], Counter()
    for ev in event_ids():
        sim.send({"cmd": "start_run", "character": "Ironclad", "seed": f"evscan_{ev}", "ascension": 10})
        st = sim.send({"cmd": "enter_room", "type": "event", "event": ev})
        if st.get("decision") != "event_choice":
            results.append({"event": ev, "option": None, "issue": f"enter: {st.get('type')} {str(st.get('message'))[:80]}"})
            summary["enter_failed"] += 1
            continue
        n_opts = len([o for o in st.get("options") or [] if not o.get("is_locked")])
        for k in range(n_opts):
            sim.send({"cmd": "start_run", "character": "Ironclad", "seed": f"evscan_{ev}", "ascension": 10})
            st = sim.send({"cmd": "enter_room", "type": "event", "event": ev})
            opts = [o for o in st.get("options") or [] if not o.get("is_locked")]
            if k >= len(opts):
                continue
            title = opts[k].get("title")
            mark = len(sim.stderr_lines)
            st = sim.act("choose_option", option_index=opts[k]["index"])
            issue, steps = None, 0
            for steps in range(80):
                if st.get("type") == "error":
                    issue = f"error: {str(st.get('message'))[:90]}"
                    break
                if st.get("decision") in ("map_select", "game_over"):
                    break
                st = step(sim, st)
            else:
                issue = f"stuck in {st.get('decision')}"
            exc = [l for l in sim.stderr_lines[mark:] if any(m in l for m in MARKERS)]
            if exc and not issue:
                issue = "exception: " + exc[0][:90]
            summary["issue" if issue else "ok"] += 1
            if issue:
                results.append({"event": ev, "option": title, "issue": issue})
    sim.proc.kill()
    print(json.dumps({"summary": dict(summary), "issues": results}, indent=1))


if __name__ == "__main__":
    main()
