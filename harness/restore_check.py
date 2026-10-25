"""Verify snapshot/restore exactness and measure its cost.

One engine process plays several random runs back to back. At random decision points a
snapshot is taken (in combat, on the map, in events, at rewards...). After each run every
snapshot is restored and must give the recorded state and checksums; then the next K
recorded actions are replayed and every state must match the original run. Branch test:
restore, take a different action, restore again, replay the original actions -> must
still match.
"""

import json
import os
import random
import statistics
import sys
import time
from collections import Counter

from simclient import Sim, legal_actions

CHARS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]
GODMODE = "--godmode" in sys.argv


def canon(st):
    return json.dumps(st, sort_keys=True)


def pick(st, rng):
    acts = legal_actions(st)
    if st.get("decision") == "combat_play" and len(acts) > 1 and rng.random() > 0.15:
        acts = acts[1:]
    return rng.choice(acts)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    runs_per_char = int(argv[0]) if len(argv) > 0 else 2
    k_follow = int(argv[1]) if len(argv) > 1 else 8
    snap_prob = float(argv[2]) if len(argv) > 2 else 0.08
    sim = Sim(os.environ.get("STS2_ENGINE_CONFIG", "Release"), timeout=60, capture_stderr=True)
    stats = Counter()
    restore_ms, replay_len, by_decision = [], [], Counter()
    failures = []
    for ch in CHARS:
        for k in range(runs_per_char):
            seed = f"restore_{ch}_{k}"
            rng = random.Random(seed)
            states = [sim.send({"cmd": "start_run", "character": ch, "seed": seed})]
            if GODMODE:  # survive to acts 2-3 so act transitions, elites, shops get covered
                sim.send({"cmd": "set_player", "hp": 9999, "max_hp": 9999})
            actions, snaps = [], []
            while len(actions) < 2500 and states[-1].get("decision") not in ("game_over", None):
                st = states[-1]
                # in god mode states[0] predates set_player, so only snapshot after a real action
                if rng.random() < snap_prob and (actions or not GODMODE):
                    sid = sim.send({"cmd": "snapshot"})
                    cs = sim.send({"cmd": "checksum"})
                    snaps.append((len(actions), sid["id"], sid["replay_length"], cs))
                a = pick(st, rng)
                actions.append(a)
                states.append(sim.act(a[0], **a[1]))
            stats["runs"] += 1
            stats["steps"] += len(actions)
            for i, sid, rlen, cs in snaps:
                stats["snapshots"] += 1
                by_decision[states[i].get("decision")] += 1
                t0 = time.perf_counter()
                r = sim.send({"cmd": "restore", "id": sid})
                restore_ms.append(1000 * (time.perf_counter() - t0))
                replay_len.append(rlen)
                cs2 = sim.send({"cmd": "checksum"})
                ok = canon(r) == canon(states[i]) and cs2.get("combat") == cs.get("combat") and cs2.get("run") == cs.get("run")
                bad_at = None
                if ok:
                    for j in range(i, min(i + k_follow, len(actions))):
                        s = sim.act(actions[j][0], **actions[j][1])
                        if canon(s) != canon(states[j + 1]):
                            ok, bad_at = False, j + 1
                            break
                if not ok:
                    failures.append({"seed": seed, "snap_step": i, "decision": states[i].get("decision"),
                                     "bad_at": bad_at, "restored": canon(r)[:300], "expected": canon(states[i])[:300],
                                     "cs": cs, "cs_after": cs2})
                    continue
                stats["restore_exact"] += 1
                # branch: diverge, restore again, replay the original continuation
                if i + 1 < len(actions):
                    alt = [a for a in legal_actions(states[i]) if a != actions[i]]
                    if alt:
                        sim.send({"cmd": "restore", "id": sid})
                        sim.act(*random.Random(sid).choice(alt)[:1], **random.Random(sid).choice(alt)[1])
                        sim.send({"cmd": "restore", "id": sid})
                        s = None
                        good = True
                        for j in range(i, min(i + k_follow, len(actions))):
                            s = sim.act(actions[j][0], **actions[j][1])
                            if canon(s) != canon(states[j + 1]):
                                good = False
                                break
                        stats["branch_tests"] += 1
                        stats["branch_exact"] += int(good)
                        if not good:
                            failures.append({"seed": seed, "snap_step": i, "branch": True, "bad_at": j + 1})
            sim.send({"cmd": "drop_snapshots"})
            last = states[-1]
            print(ch, k, "steps", len(actions), "snapshots", len(snaps), "final", last.get("decision"),
                  "act", (last.get("context") or {}).get("act"), "floor", (last.get("context") or {}).get("floor"),
                  "failures so far", len(failures), flush=True)
    exc = [l[:200] for l in sim.stderr_lines if "Exception" in l and "---> " not in l]
    sim.close()
    out = dict(stats)
    out["snapshot_decisions"] = dict(by_decision)
    out["restore_ms_median"] = round(statistics.median(restore_ms), 1)
    out["restore_ms_p90"] = round(sorted(restore_ms)[int(0.9 * len(restore_ms))], 1)
    out["restore_ms_max"] = round(max(restore_ms), 1)
    out["replay_len_median"] = statistics.median(replay_len)
    out["replay_len_max"] = max(replay_len)
    pairs = sorted(zip(replay_len, restore_ms))
    out["restore_ms_by_replay_len"] = {f"{lo}-{hi}": round(statistics.median([m for l, m in pairs if lo <= l <= hi]), 1)
                                       for lo, hi in [(0, 5), (6, 20), (21, 50), (51, 200)]
                                       if any(lo <= l <= hi for l, _ in pairs)}
    out["exception_lines"] = len(exc)
    out["failures"] = failures[:6]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
