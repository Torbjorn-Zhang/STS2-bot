"""Run-start parity: same seed string in our engine (real v0.111.0 DLL) and sts2-rs (v0.111.0 mode).
Compares the act-1 boss, the Neow offers and the whole act-1 map (node types and edges)."""
import json, os, sys
os.environ["STS2_RS_RAW_SEED"] = "1"
sys.path.insert(0, "/Users/szha0569/Daily/sts2-ai/harness")
sys.path.insert(0, "/Users/szha0569/Daily/sts2-ai/local/rs-py")
from simclient import Sim
import sts2_rs

TYPES = {"Monster": "monster", "Unknown": "unknown", "Shop": "shop", "RestSite": "rest_site", "Elite": "elite",
         "Treasure": "treasure", "Boss": "boss", "Ancient": "ancient", "Event": "event"}

def ours(sim, seed):
    st = sim.send({"cmd": "start_run", "character": "Ironclad", "seed": seed, "ascension": 10})
    m = sim.send({"cmd": "get_map"})
    neow = [o["text_key"].rsplit(".", 1)[-1] for o in st["options"]]
    nodes = {}
    for row in m["rows"]:
        for n in row:
            nodes[(n["col"], n["row"])] = (TYPES.get(n["type"], n["type"]), tuple(sorted((c["col"], c["row"]) for c in n["children"] if c["row"] < 16)))
    return {"act": st["context"]["act_name"], "boss": st["context"]["boss"]["id"], "neow": neow, "nodes": nodes,
            "hp": st["player"]["hp"], "gold": st["player"]["gold"]}

def rs(seed):
    rm = sts2_rs.RunManager(character="Ironclad", string_seed=seed, ascension_level=10, neow_start=True, act1="roll",
                            interactive_card_removal=True, interactive_card_upgrade=True, interactive_card_transform=True,
                            direct_event_relic_grant=True, game_version="v0.111.0")
    g = lambda n: (getattr(rm, n)() if callable(getattr(rm, n)) else getattr(rm, n))
    # our get_map leaves out the Ancient start (row 0) and the boss node; so do we here
    nodes = {(n["col"], n["row"]): (n["point_type"], tuple(sorted(tuple(c) for c in n["children"] if c[1] < 16)))
             for n in g("act_map") if n["point_type"] not in ("ancient", "boss")}
    return {"boss": g("pinned_boss_encounter_id"), "neow": list(g("neow_offers")), "nodes": nodes,
            "hp": g("current_hp"), "gold": g("gold")}

seeds = [f"full_{i}" for i in range(int(sys.argv[1]) if len(sys.argv) > 1 else 60)]
sim = Sim("Fix10", timeout=60)
agree = {"boss": 0, "neow": 0, "map": 0, "hp_gold": 0}
for s in seeds:
    a, b = ours(sim, s), rs(s)
    ok = {"boss": a["boss"] == b["boss"], "neow": a["neow"] == b["neow"], "hp_gold": (a["hp"], a["gold"]) == (b["hp"], b["gold"])}
    # our map lists the boss row separately; compare the nodes both have
    ka, kb = set(a["nodes"]), set(b["nodes"])
    same = [k for k in ka & kb if a["nodes"][k] == b["nodes"][k]]
    ok["map"] = ka == kb and len(same) == len(ka)
    for k, v in ok.items():
        agree[k] += v
    if not all(ok.values()):
        diff = sorted(k for k in ka | kb if a["nodes"].get(k) != b["nodes"].get(k))[:4]
        print(s, a["act"], {k: v for k, v in ok.items() if not v}, "boss", a["boss"], b["boss"], "neow", a["neow"], b["neow"],
              "map nodes", len(ka), len(kb), "first diffs", [(k, a["nodes"].get(k), b["nodes"].get(k)) for k in diff][:2])
print("AGREE of", len(seeds), agree)
sim.close()
