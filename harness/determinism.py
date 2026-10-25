"""Determinism checks.

1. Same seed + same action sequence in two fresh processes -> identical state stream?
2. Save at a map_select checkpoint, load in a fresh process, replay the remaining
   actions -> identical to the original continuation? (This is the "clone by
   save + replay" primitive a search agent would need.)
"""

import json
import os
import random
import sys
import tempfile

from simclient import Sim, legal_actions

CHARS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]


def canon(state):
    return json.dumps(state, sort_keys=True)


def record_run(character, seed, rng, save_dir=None, max_steps=3000):
    """Random run; returns (actions, states, checkpoints{step: save_path})."""
    sim = Sim(os.environ.get("STS2_ENGINE_CONFIG", "Release"))
    states = [sim.send({"cmd": "start_run", "character": character, "seed": seed})]
    actions, checkpoints = [], {}
    for step in range(max_steps):
        st = states[-1]
        if st.get("decision") == "game_over" or st.get("type") == "error":
            break
        if save_dir and st.get("decision") == "map_select" and len(checkpoints) < 3 and step > 0:
            path = os.path.join(save_dir, f"{seed}_{step}.save")
            res = sim.send({"cmd": "write_continue_save", "path": path})
            if res.get("success"):
                checkpoints[step] = path
        acts = legal_actions(st)
        if st.get("decision") == "combat_play" and len(acts) > 1 and rng.random() > 0.15:
            acts = acts[1:]
        a = rng.choice(acts)
        actions.append(a)
        states.append(sim.act(a[0], **a[1]))
    sim.close()
    return actions, states, checkpoints


def replay(first_cmd, actions):
    sim = Sim(os.environ.get("STS2_ENGINE_CONFIG", "Release"))
    states = [sim.send(first_cmd)]
    for a in actions:
        states.append(sim.act(a[0], **a[1]))
    sim.close()
    return states


def first_diff(xs, ys):
    for i, (x, y) in enumerate(zip(xs, ys)):
        if canon(x) != canon(y):
            return i
    return None if len(xs) == len(ys) else min(len(xs), len(ys))


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tmp = tempfile.mkdtemp(prefix="sts2det_", dir=os.path.dirname(os.path.abspath(__file__)))
    summary = {"same_seed_replays": 0, "same_seed_identical": 0,
               "save_load_replays": 0, "save_load_identical": 0, "diffs": []}
    for ch in CHARS:
        for k in range(seeds):
            seed = f"det_{ch}_{k}"
            actions, states, cps = record_run(ch, seed, random.Random(seed), save_dir=tmp)
            again = replay({"cmd": "start_run", "character": ch, "seed": seed}, actions)
            summary["same_seed_replays"] += 1
            d = first_diff(states, again)
            if d is None:
                summary["same_seed_identical"] += 1
            else:
                summary["diffs"].append({"kind": "same_seed", "seed": seed, "step": d,
                                         "a": canon(states[d])[:400], "b": canon(again[d])[:400]})
            for step, path in cps.items():
                cont = replay({"cmd": "load_save", "path": path}, actions[step:])
                summary["save_load_replays"] += 1
                d = first_diff(states[step:], cont)
                if d is None:
                    summary["save_load_identical"] += 1
                else:
                    summary["diffs"].append({"kind": "save_load", "seed": seed, "from_step": step,
                                             "offset": d, "a": canon(states[step + d])[:600],
                                             "b": canon(cont[d])[:600]})
            print(ch, k, "steps", len(actions), "checkpoints", len(cps), flush=True)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
