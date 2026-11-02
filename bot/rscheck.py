"""Check rssim.RsSim's rendering against our engine: one deterministic policy plays the same seed in
both (RsSim with expose_combat, our engine Fix11), and every state the deck policy reads is
compared field by field. Stops at the first difference.

  ../.venv/bin/python rscheck.py SEED [SEED...]    (RSCHECK_GOD=9999 gives both players 9999 HP)
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
from simclient import Sim  # noqa: E402
from rssim import RsSim  # noqa: E402

GOD = int(os.environ.get("RSCHECK_GOD", "0"))


def view(st):
    """The fields the deck policy and run.py read, normalised for comparison."""
    d = st.get("decision") or st.get("type")
    ctx = st.get("context") or {}
    p = st.get("player") or {}
    v = {"d": d, "act": ctx.get("act"), "an": ctx.get("act_name"), "floor": ctx.get("floor"),
         "room": ctx.get("room_type"), "boss": (ctx.get("boss") or {}).get("id")}
    if d != "combat_play":
        v.update(hp=p.get("hp"), mhp=p.get("max_hp"), gold=p.get("gold"),
                 relics=[r.get("name") for r in p.get("relics") or []],
                 pots=sorted(x.get("name") for x in p.get("potions") or [] if x),
                 deck=sorted(Counter((c.get("id"), bool(c.get("upgraded"))) for c in p.get("deck") or []).items()))
    if d == "card_reward":
        v["cards"] = [(c.get("id"), bool(c.get("upgraded")), c.get("type"), c.get("rarity")) for c in st.get("cards") or []]
    if d == "bundle_select":
        v["bundles"] = [[c.get("name") for c in b.get("cards") or []] for b in st.get("bundles") or []]
    if d == "event_choice":
        v["ev"] = st.get("event_name")
        v["opts"] = [(o.get("text_key"), bool(o.get("is_locked"))) for o in st.get("options") or []]
    if d == "rest_site":
        v["opts"] = [(o.get("option_id"), bool(o.get("is_enabled", True))) for o in st.get("options") or []]
    if d == "map_select":
        v["choices"] = sorted((c.get("col"), c.get("row"), c.get("type")) for c in st.get("choices") or [])
    if d == "shop":
        v["shop"] = ([(c.get("name"), c.get("cost")) for c in st.get("cards") or []],
                     [(r.get("name"), r.get("cost")) for r in st.get("relics") or []],
                     [(x.get("name"), x.get("cost")) for x in st.get("potions") or []], st.get("card_removal_cost"))
    if d == "card_select":
        v["cards"] = [(c.get("id"), bool(c.get("upgraded"))) for c in st.get("cards") or []]
        v["n"] = (st.get("min_select"), min(st.get("max_select") or 0, len(st.get("cards") or [])))
    if d == "combat_play":
        v.update(round=st.get("round"), energy=st.get("energy"), php=p.get("hp"),
                 hand=[(c.get("id"), bool(c.get("upgraded")), bool(c.get("can_play"))) for c in st.get("hand") or []],
                 enemies=[(e.get("hp"), e.get("block")) for e in st.get("enemies") or [] if (e.get("hp") or 0) > 0])
    return v


def policy(st):
    """One deterministic choice in our engine's action vocabulary."""
    d = st.get("decision")
    if d == "event_choice":
        o = [x for x in st.get("options") or [] if not x.get("is_locked")]
        return ("choose_option", {"option_index": o[0]["index"]}) if o else ("proceed", {})
    if d == "rest_site":
        o = [x for x in st.get("options") or [] if x.get("is_enabled", True)]
        return ("choose_option", {"option_index": o[0]["index"]})
    if d == "map_select":
        c = min(st["choices"], key=lambda c: (c["row"], c["col"]))
        return ("select_map_node", {"col": c["col"], "row": c["row"]})
    if d == "card_reward":
        return ("select_card_reward", {"card_index": st["cards"][0]["index"]}) if st.get("cards") else ("skip_card_reward", {})
    if d == "bundle_select":
        return ("select_bundle", {"bundle_index": 0})
    if d == "shop":
        return ("leave_room", {})
    if d == "card_select":
        cards = st.get("cards") or []
        n = max(1, st.get("min_select") or 0)
        return ("select_cards", {"indices": ",".join(str(c.get("index", i)) for i, c in enumerate(cards[:n]))}) if cards else ("skip_select", {})
    if d == "combat_play":
        for c in st.get("hand") or []:
            if c.get("can_play"):
                return ("play_card", {"card_index": c["index"], "target_index": 0})
        return ("end_turn", {})
    return ("proceed", {})


def check(seed, steps=3000):
    ours, rs = Sim("Fix11", timeout=60), RsSim(expose_combat=True)
    try:
        a = ours.send({"cmd": "start_run", "character": "Ironclad", "seed": seed, "ascension": 10})
        b = rs.send({"cmd": "start_run", "character": "Ironclad", "seed": seed, "ascension": 10})
        if GOD:
            a["player"] = ours.send({"cmd": "set_player", "hp": GOD, "max_hp": GOD})["player"]
            rs.rm.set_player(hp=GOD, max_hp=GOD)
            b = rs._render()
        for step in range(steps):
            va, vb = view(a), view(b)
            if va != vb:
                diff = {k: (va.get(k), vb.get(k)) for k in sorted(set(va) | set(vb)) if va.get(k) != vb.get(k)}
                return {"seed": seed, "step": step, "diff": diff, "decision": va.get("d")}
            if va["d"] == "game_over":
                return {"seed": seed, "step": step, "ok": True}
            act = policy(a)
            a = ours.act(act[0], **act[1])
            b = rs.act(act[0], **act[1])
            if b.get("type") == "error":
                return {"seed": seed, "step": step, "error": b.get("message"), "action": act}
        return {"seed": seed, "ok": True, "note": "max steps"}
    finally:
        ours.close()


if __name__ == "__main__":
    for s in sys.argv[1:] or ["full_145"]:
        print(json.dumps(check(s), default=str)[:2500], flush=True)
