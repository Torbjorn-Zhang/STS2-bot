"""Process reuse must not leak state between runs.

Record each (character, seed) run with a random policy in its own fresh process, then
replay all of them back to back in ONE process and require identical state streams.
Also counts stderr exception lines per mode.
"""

import json
import random
import sys
import time

from determinism import canon, first_diff
from simclient import Sim, legal_actions

CHARS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]


def record(sim, ch, seed, rng, max_steps=3000):
    states = [sim.send({"cmd": "start_run", "character": ch, "seed": seed})]
    actions = []
    for _ in range(max_steps):
        st = states[-1]
        if st.get("decision") == "game_over" or st.get("type") == "error":
            break
        acts = legal_actions(st)
        if st.get("decision") == "combat_play" and len(acts) > 1 and rng.random() > 0.15:
            acts = acts[1:]
        a = rng.choice(acts)
        actions.append(a)
        states.append(sim.act(a[0], **a[1]))
    return actions, states


def replay(sim, ch, seed, actions):
    states = [sim.send({"cmd": "start_run", "character": ch, "seed": seed})]
    for a in actions:
        states.append(sim.act(a[0], **a[1]))
    return states


def exc_lines(lines):
    return sum(1 for l in lines if "Exception" in l and "---> " not in l)


def main():
    per_char = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    config = sys.argv[2] if len(sys.argv) > 2 else "Debug"
    jobs = [(ch, f"reuse_{ch}_{k}") for k in range(per_char) for ch in CHARS]
    fresh, fresh_exc = {}, 0
    for ch, seed in jobs:
        sim = Sim(config, capture_stderr=True)
        fresh[seed] = record(sim, ch, seed, random.Random(seed))
        time.sleep(0.05)
        fresh_exc += exc_lines(sim.stderr_lines)
        sim.close()
    sim = Sim(config, capture_stderr=True)
    t0 = time.perf_counter()
    same, diffs = 0, []
    for ch, seed in jobs:
        actions, states = fresh[seed]
        again = replay(sim, ch, seed, actions)
        d = first_diff(states, again)
        if d is None:
            same += 1
        else:
            diffs.append({"seed": seed, "step": d, "fresh": canon(states[d])[:300], "reused": canon(again[d])[:300]})
    wall = time.perf_counter() - t0
    starts = [s for n, s in sim.latencies if n == "start_run"]
    steps = sum(len(fresh[s][0]) for _, s in jobs)
    time.sleep(0.05)
    reused_exc = exc_lines(sim.stderr_lines)
    sim.close()
    print(json.dumps({
        "runs": len(jobs), "identical_in_reused_process": same,
        "steps": steps, "reused_wall_s": round(wall, 2), "reused_steps_per_s": round(steps / wall, 1),
        "start_run_ms_first": round(1000 * starts[0], 1),
        "start_run_ms_median_after_first": round(1000 * sorted(starts[1:])[len(starts[1:]) // 2], 1),
        "exception_lines_fresh_total": fresh_exc, "exception_lines_reused": reused_exc,
        "diffs": diffs[:5]}, indent=1))


if __name__ == "__main__":
    main()
