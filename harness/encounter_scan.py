"""Fight every encounter once per character in one reused engine process.

999x HP, random legal play until the fight ends (or 60 rounds). Collects missing Godot
stub methods, other exception lines, stalls and timeouts per encounter.
"""

import json
import os
import random
import re
import sys
from collections import defaultdict

from simclient import CLI, Sim, legal_actions

CHARS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Ironclad", "Defect"]


def encounters():
    d = json.load(open(os.path.join(CLI, "localization_eng", "encounters.json")))
    return sorted({k.split(".")[0] for k in d})


def fight(sim, ch, enc, rng):
    sim.send({"cmd": "start_run", "character": ch, "seed": f"enc_{enc}"})
    sim.send({"cmd": "set_player", "hp": 9999, "max_hp": 9999})
    rtype = "elite" if enc.endswith("_ELITE") else "combat"
    st = sim.send({"cmd": "enter_room", "type": rtype, "encounter": enc})
    if st.get("type") == "error":
        return {"result": "enter_error", "message": st.get("message", "")[:200]}
    steps = 0
    while st.get("decision") == "combat_play" and (st.get("round") or 0) <= 60 and steps < 1500:
        acts = legal_actions(st)
        if len(acts) > 1 and rng.random() > 0.2:
            acts = acts[1:]
        a = rng.choice(acts)
        st = sim.act(a[0], **a[1])
        while st.get("decision") not in ("combat_play", "game_over", "card_reward", "map_select", None) and steps < 1500:
            a2 = legal_actions(st)[0]
            st = sim.act(a2[0], **a2[1])
            steps += 1
        if st.get("type") == "error":
            return {"result": "action_error", "message": st.get("message", "")[:200], "steps": steps}
        steps += 1
    d = st.get("decision")
    res = "won" if d in ("card_reward", "map_select") else "died" if d == "game_over" else "round_cap" if d == "combat_play" else f"other:{d}"
    return {"result": res, "rounds": st.get("round"), "steps": steps}


def main():
    sim = Sim("Release", timeout=30, capture_stderr=True)
    report = defaultdict(dict)
    missing = defaultdict(set)
    for enc in encounters():
        for ch in CHARS:
            mark = len(sim.stderr_lines)
            try:
                r = fight(sim, ch, enc, random.Random(enc + ch))
            except TimeoutError:
                r = {"result": "timeout"}
                sim.close()
                sim = Sim("Release", timeout=30, capture_stderr=True)
            lines = sim.stderr_lines[mark:]
            mm = sorted({m.group(1) for l in lines for m in [re.search(r"Method not found: '([^']+)'", l)] if m})
            for m in mm:
                missing[m].add(enc)
            r["missing_methods"] = mm
            r["exception_lines"] = sum(1 for l in lines if "Exception" in l and "--->" not in l and "Failed to save progress" not in l
                                       and "InitProfileId" not in l)
            r["stuck_logs"] = sum(1 for l in lines if "stuck" in l.lower())
            report[enc][ch] = r
        print(enc, {c: (v["result"], v.get("rounds"), len(v["missing_methods"]), v["exception_lines"]) for c, v in report[enc].items()}, flush=True)
    sim.close()
    json.dump(report, open("encounter_scan.json", "w"), indent=1)
    print("\nMISSING METHODS:")
    for m, encs in sorted(missing.items()):
        print(" ", m, "<-", sorted(encs))
    bad = {e: {c: v["result"] for c, v in r.items()} for e, r in report.items()
           if any(v["result"] not in ("won", "died", "round_cap") or v["stuck_logs"] for v in r.values())}
    print("\nNON-CLEAN OUTCOMES:", json.dumps(bad, indent=1))
    noisy = {e: {c: v["exception_lines"] for c, v in r.items()} for e, r in report.items() if any(v["exception_lines"] for v in r.values())}
    print("\nENCOUNTERS WITH EXCEPTION LINES:", json.dumps(noisy))


if __name__ == "__main__":
    main()
