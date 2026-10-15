"""Parity check 1: same seed, compare run generation field by field.

Starts a run in sts2-cli with the real game's seed, writes a continue save, and diffs
it against the real game's current_run.save (generated content only: acts, encounter /
event / elite sequences, boss, map, odds, rng stream seeds).
"""

import json
import os
import sys

from simclient import Sim

# real-game saves stay local (gitignored): they contain the Steam profile id
PAR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "parity")


def ddiff(a, b, path="", out=None, limit=60):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if type(a) != type(b):
        out.append((path, a, b))
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            ddiff(a.get(k), b.get(k), f"{path}.{k}", out, limit)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append((path + ".len", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            ddiff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append((path, a, b))
    return out


def main():
    real_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PAR, "real_floor1_combat.save")
    real = json.load(open(real_path))
    seed = real["rng"]["seed"]
    char = real["players"][0]["character_id"].split(".")[-1].capitalize()
    sim = Sim("Debug")
    st = sim.send({"cmd": "start_run", "character": char, "seed": seed, "ascension": real["ascension"], "unlock_from_save": real_path})
    sim_path = os.path.join(PAR, f"sim_start_{seed}.save")
    r = sim.send({"cmd": "write_continue_save", "path": sim_path})
    sim.close()
    print("sim start decision:", st.get("decision"), "| save:", r.get("success"))
    sim_save = json.load(open(sim_path))

    print("seed real/sim:", seed, sim_save["rng"]["seed"])
    for i, (ra, sa) in enumerate(zip(real["acts"], sim_save["acts"])):
        print(f"--- act {i}: real {ra['id']} / sim {sa['id']}")
        for k in sorted(set(ra["rooms"]) | set(sa["rooms"])):
            rv, sv = ra["rooms"].get(k), sa["rooms"].get(k)
            mark = "SAME" if rv == sv else "DIFF"
            print(f"   {mark} rooms.{k}: " + ("" if rv == sv else f"\n        real={json.dumps(rv)[:300]}\n        sim ={json.dumps(sv)[:300]}"))
        md = ddiff(ra.get("saved_map"), sa.get("saved_map"))
        print(f"   map: {'SAME' if not md else f'{len(md)}+ differences, first: {md[:3]}'}")
    print("odds:", "SAME" if real["odds"] == sim_save["odds"] else f"real={real['odds']} sim={sim_save['odds']}")
    for name in sorted(set(real["rng"]["rngs"]) | set(sim_save["rng"]["rngs"])):
        a, b = real["rng"]["rngs"].get(name), sim_save["rng"]["rngs"].get(name)
        same_state = a and b and all(a[k] == b[k] for k in a if k.startswith("s"))
        print(f"rng {name:28s} real counter={a and a['counter']} sim counter={b and b['counter']} state={'SAME' if same_state else 'DIFF'}")


if __name__ == "__main__":
    main()
