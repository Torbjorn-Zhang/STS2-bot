"""Our engine's JSON protocol on top of sts2-rs (tfriedel/spire-mentor's Rust port of StS2, MIT),
in its v0.111.0 mode (build 41cef1ea = our game DLL). Out-of-combat decisions are rendered in our
engine's state format (sts2-cli RunSimulator), so run.py, policy.py and the deck networks run
unchanged; fights are played inside by sts2-rs's MCTS (plays_combat), or exposed as combat_play
states (expose_combat=True) so a parity harness can drive both engines with one policy.

  sim = RsSim(sims=250, boss_sims=1000)          # like simclient.Sim: send(cmd) / act(action, **args)
  st = sim.send({"cmd": "start_run", "character": "Ironclad", "seed": "full_3", "ascension": 10})

Seeds hash as given (STS2_RS_RAW_SEED), like RunState.CreateForTest in our engine, so a seed starts
the same run in both engines (verified on 240 eval seeds: map, Neow, bosses). The extension module
is looked up in STS2_RS_PY (default: the Mac's local/rs-py, else the Apollo Lustre dir)."""
import copy
import json
import os
import random
import re
import sys

os.environ.setdefault("STS2_RS_RAW_SEED", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("STS2_RS_PY"), os.path.join(HERE, "..", "local", "rs-py"),
           "/scratch/xkzhang/slzhang_cvpr27/rs/py"):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import sts2_rs  # noqa: E402

LOC_DIR = next(p for p in (os.environ.get("STS2_LOC_DIR"), os.path.join(HERE, "..", "sts2-cli", "localization_eng"))
               if p and os.path.isdir(p))


def _loc(name):
    with open(os.path.join(LOC_DIR, name + ".json")) as f:
        return json.load(f)


_CARDS, _RELICS, _POTIONS = _loc("cards"), _loc("relics"), _loc("potions")
CARD_IDS = {k[:-6] for k in _CARDS if k.endswith(".title")}
_CARD_SPECIAL = {"SLOTH_STATUS": "SLOTH"}


def card_id(name):
    """sts2-rs card name (Debug 'ColossusCard', python 'COLOSSUS_CARD', 'CARD.X') -> the C# id."""
    s = str(name).split(".", 1)[1] if str(name).startswith("CARD.") else str(name)
    s = _CARD_PY.get(s, s)
    if s in _CARD_SPECIAL:
        return _CARD_SPECIAL[s]
    if s not in CARD_IDS and s.endswith("_CARD") and s[:-5] in CARD_IDS:
        return s[:-5]
    return s


_CARD_PY = dict(sts2_rs.list_registry_id_forms("card"))
_RELIC_PY = dict(sts2_rs.list_registry_id_forms("relic_id"))
_POTION_PY = dict(sts2_rs.list_registry_id_forms("potion"))
_KIND = {card_id(n): (t, r) for n, t, r in sts2_rs.card_kinds()}
_TYPE = {"ATTACK": "Attack", "SKILL": "Skill", "POWER": "Power", "STATUS": "Status", "CURSE": "Curse", "QUEST": "Quest"}


def relic_id(name):
    s = str(name).split(".", 1)[1] if str(name).startswith("RELIC.") else str(name)
    return _RELIC_PY.get(s, s)


def potion_id(name):
    s = str(name).split(".", 1)[1] if str(name).startswith("POTION.") else str(name)
    return _POTION_PY.get(s, s)


def card_name(cid):
    return _CARDS.get(cid + ".title", cid)


def relic_name(rid):
    return _RELICS.get(rid + ".title", rid)


def potion_name(pid):
    return _POTIONS.get(pid + ".title", pid)


# Tables made from our engine (Fix11 "card_db" / "relic_db", derived from the game DLL, so kept out
# of git: `python rssim.py --make-db [DIR]` writes them), looked up in STS2_RS_DATA, bot/data, then
# the Apollo Lustre dir
DATA_DIRS = [p for p in (os.environ.get("STS2_RS_DATA"), os.path.join(HERE, "data"),
                         "/scratch/xkzhang/slzhang_cvpr27/rs/data") if p]


def _data(name):
    for d in DATA_DIRS:
        if os.path.exists(os.path.join(d, name)):
            with open(os.path.join(d, name)) as f:
                return json.load(f)
    return {}


def make_db(out_dir, config="Fix11"):
    """Write card_db_v0111.json / relic_db_v0111.json from our engine (needs a started run)."""
    sys.path.insert(0, os.path.join(HERE, "..", "harness"))
    from simclient import Sim
    sim = Sim(config, timeout=60)
    try:
        sim.send({"cmd": "start_run", "character": "Ironclad", "seed": "db", "ascension": 0})
        cards = {c["id"]: c for c in sim.send({"cmd": "card_db"})["cards"]}
        relics = {r["id"]: r["vars"] for r in sim.send({"cmd": "relic_db"})["relics"]}
    finally:
        sim.close()
    os.makedirs(out_dir, exist_ok=True)
    for name, d in (("card_db_v0111.json", cards), ("relic_db_v0111.json", relics)):
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(d, f, sort_keys=True)
    return len(cards), len(relics)


# v0.111.0 card table (the real DLL's type / rarity / printed cost); sts2-rs's card_kinds() carries
# v0.107.1 rarities (Taunt, Cruelty, ... changed in v0.111.0)
CARD_DB = _data("card_db_v0111.json")


def card_entry(name, upgraded=False, index=None):
    """One card in our engine's format (id, name, cost, type, rarity, upgraded)."""
    cid = card_id(name)
    db = CARD_DB.get(cid)
    if db:
        t, r, cost = db.get("type"), db.get("rarity"), db.get("cost")
    else:
        t, r = _KIND.get(cid, ("", ""))
        t, r, cost = _TYPE.get(t, t.title()), r.title(), None
    e = {"id": "CARD." + cid, "name": card_name(cid), "cost": cost, "type": t, "rarity": r, "upgraded": bool(upgraded)}
    if index is not None:
        e = {"index": index, **e}
    return e


_ANCIENTS = _loc("ancients")
_EVENTS = _loc("events")


def event_name(rs_name):
    """sts2-rs event name ('ZenWeaver') -> our engine's display title ('Zen Weaver')."""
    if rs_name is None:
        return None
    key = upper_snake(rs_name) + ".title"
    return _EVENTS.get(key) or _ANCIENTS.get(key) or rs_name


RELIC_VARS = _data("relic_db_v0111.json")   # relic id -> canonical dynamic vars (Ancient options)


def upper_snake(pascal):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", str(pascal)).upper()


ROOM = {"monster": "Monster", "elite": "Elite", "boss": "Boss", "shop": "Shop", "rest_site": "RestSite",
        "treasure": "Treasure", "event": "Event", "unknown": "Unknown", "ancient": "Ancient"}
ACT1_BOSSES_UNDERDOCKS = {"LAGAVULIN_MATRIARCH_BOSS", "WATERFALL_GIANT_BOSS", "SOUL_FYSH_BOSS"}


class RsSim:
    plays_combat = True

    def __init__(self, sims=250, boss_sims=1000, game_version="v0.111.0", expose_combat=False, mcts_seed=0):
        self.sims, self.boss_sims, self.gv = sims, boss_sims, game_version
        self.expose_combat = expose_combat
        self.mcts_seed = mcts_seed
        self.rm = None
        self._snaps, self._saves, self._sid = {}, {}, 1
        self.act1 = None   # "Overgrowth" / "Underdocks" of this run
        self._at_ancient = False   # act start: the player walked into the Ancient node
        self._next = None          # act start: the next act's context / moves, peeked on a copy

    # ── protocol ────────────────────────────────────────────────────────────
    def send(self, cmd):
        c = cmd.get("cmd")
        if c == "start_run":
            return self._start(cmd)
        if c == "action":
            return self._action(cmd.get("action"), cmd.get("args") or {})
        if c == "get_map":
            return self._map()
        if c in ("snapshot", "write_continue_save"):
            key = cmd.get("path") or self._sid
            self._sid += 1
            (self._saves if c == "write_continue_save" else self._snaps)[key] = copy.deepcopy(self.rm)
            return {"id": key} if c == "snapshot" else {"success": True, "path": key}
        if c in ("restore", "load_save"):
            src = self._snaps if c == "restore" else self._saves
            key = cmd.get("id") if c == "restore" else cmd.get("path")
            if key not in src:
                return {"type": "error", "message": f"rssim: no {c} {key}"}
            self.rm = copy.deepcopy(src[key])
            return {"type": "ok"} if cmd.get("quiet") else self._render()
        if c == "drop_snapshots":
            frm = cmd.get("from")
            for k in [k for k in self._snaps if frm is None or (isinstance(k, int) and k >= frm)]:
                del self._snaps[k]
            return {"type": "ok"}
        if c == "determinize_run":
            return self._determinize(cmd.get("seed") or 0)
        if c == "gc_stats":
            return {"working_set_mb": 0}
        if c == "quit":
            return {"type": "ok"}
        return {"type": "error", "message": f"rssim: unsupported command {c}"}

    def act(self, action, **args):
        return self.send({"cmd": "action", "action": action, "args": args})

    def close(self):
        self.rm = None

    # ── run ─────────────────────────────────────────────────────────────────
    def _start(self, cmd):
        self.rm = sts2_rs.RunManager(character=cmd.get("character") or "Ironclad", string_seed=str(cmd["seed"]),
                                     ascension_level=int(cmd.get("ascension") or 0), neow_start=True, act1="roll",
                                     interactive_card_removal=True, interactive_card_upgrade=True,
                                     interactive_card_transform=True, direct_event_relic_grant=True,
                                     game_version=self.gv)
        self.agent = sts2_rs.MctsCombatAgent(seed=self.mcts_seed, sims=self.sims, boss_sims=self.boss_sims)
        boss = self.rm.pinned_boss_encounter_id()
        self.act1 = "Underdocks" if boss in ACT1_BOSSES_UNDERDOCKS else "Overgrowth"
        return self._render()

    def _g(self, n):
        v = getattr(self.rm, n)
        return v() if callable(v) else v

    def _acts(self):
        return self.rm.available_actions()

    def _phase(self):
        ph = self._g("phase")
        if ph == "event" and self._g("current_event_name") is None:
            self.rm.materialize_event()
        return ph

    def _deck_prompt(self):
        acts = [a for a in self._acts() if str(a.get("action", "")).endswith("_deck_card")]
        return acts[0]["action"].split("_")[0] if acts else None

    # ── rendering in our engine's format ────────────────────────────────────
    def _context(self):
        act = self._g("act_index") or 1
        room = (self.rm.current_room() or {}).get("room_type")
        ph = self._g("phase")
        # a deck prompt left by an event / shop keeps sts2-rs in map_choice; our engine still shows its room
        # after the first of A10's two act-3 bosses our engine stays in the Boss room for the map pick
        rt = "Map" if ph == "map_choice" and not self._deck_prompt() and room != "boss" else ROOM.get(room, (room or "").title())
        if not rt:
            rt = "Event"   # a prompt from an Ancient's relic (sts2-rs has no room there; our engine is in the event)
        boss = self.rm.pinned_boss_encounter_id()
        return {"act": act, "act_name": [None, self.act1, "Hive", "Glory"][min(act, 3)],
                "floor": max(1, self._g("act_floor") or 0), "room_type": rt, "ascension": self._g("ascension_level"),
                "boss": {"id": boss, "name": boss}}

    def _player(self):
        deck = [card_entry(c, up) for c, up in self._g("master_deck_detailed")]
        return {"hp": self._g("current_hp"), "max_hp": self._g("max_hp"), "block": 0, "gold": self._g("gold"),
                "relics": [{"name": relic_name(relic_id(r))} for r in self._g("relics")],
                "potions": [{"index": i, "name": potion_name(potion_id(p))} for i, p in enumerate(self._g("potions"))],
                "deck_size": len(deck), "deck": deck}

    def _base(self, decision, **kw):
        return {"type": "decision", "decision": decision, "context": self._context(), "player": self._player(), **kw}

    def _render(self):
        if self.rm is None:
            return {"type": "error", "message": "rssim: no run"}
        for _ in range(200):
            if self._g("is_over"):
                ctx = self._context()
                return {"type": "decision", "decision": "game_over", "context": ctx, "player": self._player(),
                        "victory": bool(self._g("player_won")), "act": ctx["act"], "floor": ctx["floor"]}
            ph = self._phase()
            kind = self._deck_prompt()
            if kind:
                return self._render_deck_prompt(kind)
            if ph == "combat":
                # run.py's combat branch reads the opening state (enemies, HP) and then calls
                # play_combat (plays_combat); a parity harness drives it itself (expose_combat)
                return self._render_combat()
            if ph == "treasure":
                self._take_first("collect_treasure")
                continue
            if ph == "card_reward":
                st = self._render_rewards()
                if st is not None:
                    return st
                continue
            if ph == "map_choice":
                ch = [a for a in self._acts() if a.get("action") == "move"]
                nodes = {(n["col"], n["row"]): n for n in self.rm.act_map()}
                row = (self.rm.current_room() or {}).get("map_coord")
                nxt = (row[1] + 1) if row else 1
                choices = [{"col": a["col"], "row": a.get("row", nxt),
                            "type": ROOM.get((nodes.get((a["col"], a.get("row", nxt))) or {}).get("point_type"), "Unknown")}
                           for a in ch]
                ctx = self._context()
                return self._base("map_select", choices=choices, act=ctx["act"], act_name=ctx["act_name"], floor=ctx["floor"])
            if ph == "boss_relic":
                return self._render_act_start()
            if ph == "event":
                if self._g("current_event_name") == "FakeMerchant":
                    # our adapter cannot present the Fake Merchant's custom layout and walks out
                    leave = sorted(a.get("idx") for a in self._acts() if a.get("action") == "event_choice")
                    if leave:
                        self.rm.take_action({"action": "event_choice", "idx": leave[-1]})
                        continue
                return self._render_event(ph)
            if ph == "rest_site" and any(a.get("action") == "smith_card" for a in self._acts()):
                # SMITH picked: our engine asks which card to upgrade with a card_select
                cards = [card_entry(c, False, i) for i, c in enumerate(self._g("smith_options") or [])]
                self._prompt = "smith"
                return self._base("card_select", cards=cards, min_select=1, max_select=1)
            if ph == "rest_site":
                opts = [{"index": i, "option_id": str(n).upper(), "name": str(n).title().replace("_", "") + "RestSiteOption",
                         "is_enabled": bool(en)} for i, (n, en) in enumerate(self._g("rest_site_options") or [])]
                return self._base("rest_site", options=opts)
            if ph == "shop":
                return self._render_shop()
            return {"type": "error", "message": f"rssim: unknown phase {ph}"}
        return {"type": "error", "message": "rssim: no decision after 200 auto steps"}

    def _take_first(self, name):
        for a in self._acts():
            if a.get("action") == name:
                return self.rm.take_action(a)
        acts = self._acts()
        if acts:
            return self.rm.take_action(acts[0])
        return None

    def _render_rewards(self):
        """Relic rewards are auto-taken (our engine auto-collects them); card offers are decisions."""
        acts = self._acts()
        if any(a.get("action") == "pick_relic_reward" for a in acts):
            self._take_first("pick_relic_reward")
            return None
        bundles = self._g("card_reward_bundles")
        if bundles:
            return self._base("bundle_select", bundles=[{"index": i, "cards": [card_entry(c) for c in b]}
                                                        for i, b in enumerate(bundles)])
        ids = self._g("card_reward_option_ids") or []
        ups = self._g("card_reward_option_upgraded") or [False] * len(ids)
        if ids and self._g("card_offer_choose_screen"):
            # Neow / event "choose a card" screens: our engine shows a (skippable) card_select
            skip = bool(self._g("card_offer_can_skip"))
            return self._base("card_select", cards=[card_entry(c, u, i) for i, (c, u) in enumerate(zip(ids, ups))],
                              min_select=0 if skip else 1, max_select=1, purpose="add")
        if not ids:
            if acts:
                self.rm.take_action(acts[0])
                return None
            return {"type": "error", "message": "rssim: empty card reward"}
        return self._base("card_reward", cards=[card_entry(c, u, i) for i, (c, u) in enumerate(zip(ids, ups))],
                          can_skip=bool(self._g("card_offer_can_skip")))

    def _peek_next_act(self):
        """Our engine opens the next act's map before the Ancient (which the player may skip);
        sts2-rs offers the Ancient's relics first. Advance a copy past the offer to read the
        next act's context, Ancient node and first-row moves."""
        if self._next is not None:
            return self._next
        peek = copy.deepcopy(self.rm)
        acts = peek.available_actions()
        skip = [a for a in acts if a.get("action") == "skip_relic"]
        peek.take_action(skip[0] if skip else acts[0])
        g = lambda n: (getattr(peek, n)() if callable(getattr(peek, n)) else getattr(peek, n))
        act = g("act_index") or 1
        boss = peek.pinned_boss_encounter_id()
        nodes = {(n["col"], n["row"]): n for n in peek.act_map()}
        anc = [(c, n) for (c, r), n in nodes.items() if r == 0 and n.get("point_type") == "ancient"]
        kids = sorted({tuple(k) for k in (anc[0][1].get("children") or [])}) if anc else []
        moves = [{"col": c, "row": r, "type": ROOM.get((nodes.get((c, r)) or {}).get("point_type"), "Unknown")} for c, r in kids]
        anc = [c for c, _ in anc]
        ctx = {"act": act, "act_name": [None, self.act1, "Hive", "Glory"][min(act, 3)], "floor": 0, "room_type": "Map",
               "ascension": g("ascension_level"), "boss": {"id": boss, "name": boss}}
        # the next act's opening heal: our engine applies it on entering the Ancient node,
        # sts2-rs only after the relic pick, so the Ancient event shows the copy's HP
        self._next = {"ctx": ctx, "ancient_col": anc[0] if anc else 3, "moves": moves,
                      "hp": g("current_hp"), "max_hp": g("max_hp"), "rm": peek}
        return self._next

    def _render_act_start(self):
        nxt = self._peek_next_act()
        if not self._at_ancient:
            choices = [{"col": nxt["ancient_col"], "row": 0, "type": "Ancient"}] + nxt["moves"]
            st = self._base("map_select", choices=choices, act=nxt["ctx"]["act"], act_name=nxt["ctx"]["act_name"], floor=0)
            st["context"] = dict(nxt["ctx"])
            return st
        anc = self._g("next_ancient") or "Ancient"
        key = upper_snake(anc)
        name = _ANCIENTS.get(key + ".title", anc)
        rel = [relic_id(r) for r in self._g("boss_relic_options") or []]
        opts = [{"index": i, "title": relic_name(r), "text_key": f"{key}.pages.INITIAL.options.{r}", "is_locked": False,
                 "vars": RELIC_VARS.get(r)} for i, r in enumerate(rel)]
        st = self._base("event_choice", event_name=name, options=opts)
        st["context"] = {**nxt["ctx"], "floor": 1, "room_type": "Event"}
        st["player"]["hp"], st["player"]["max_hp"] = nxt["hp"], nxt["max_hp"]
        return st

    def _render_event(self, ph):
        if ph == "boss_relic":
            rel = [relic_id(r) for r in self._g("boss_relic_options") or []]
            name = (self._g("current_event_name") or "ANCIENT")
            opts = [{"index": i, "title": relic_name(r), "text_key": f"{str(name).upper()}.pages.INITIAL.options.{r}",
                     "is_locked": False, "vars": None} for i, r in enumerate(rel)]
            return self._base("event_choice", event_name=name, options=opts)
        name = self._g("current_event_name")
        infos = self.rm.event_option_infos() or []
        avail = {a.get("idx") for a in self._acts() if a.get("action") == "event_choice"}
        opts = []
        for i, o in enumerate(infos):
            key = o.get("loc_key") or (f"NEOW.pages.INITIAL.options.{o.get('label')}" if name == "Neow" else "")
            locked = i not in avail
            if locked and (key + "_LOCKED.title") in _EVENTS:
                key += "_LOCKED"   # our engine shows a locked option under its _LOCKED text
            opts.append({"index": i, "title": o.get("label"), "text_key": key, "is_locked": locked,
                         "vars": o.get("vars") or None,
                         "description": _EVENTS.get(key + ".description") or o.get("description")})
        return self._base("event_choice", event_name=event_name(name), options=opts)

    def _render_shop(self):
        inv = self.rm.shop_inventory() or {}
        cards = [{"index": i, **card_entry(c.get("card") or c.get("id") or c.get("name")), "cost": c.get("price") or c.get("cost"),
                  "is_stocked": True} for i, c in enumerate(inv.get("cards") or [])]
        relics = [{"index": i, "name": relic_name(relic_id(r.get("relic") or r.get("id") or r.get("name"))),
                   "cost": r.get("price") or r.get("cost"), "is_stocked": True} for i, r in enumerate(inv.get("relics") or [])]
        pots = [{"index": i, "name": potion_name(potion_id(p.get("potion") or p.get("id") or p.get("name"))),
                 "cost": p.get("price") or p.get("cost"), "is_stocked": True} for i, p in enumerate(inv.get("potions") or [])]
        return self._base("shop", cards=cards, relics=relics, potions=pots,
                          card_removal_cost=inv.get("removal_price") or inv.get("card_removal_cost"))

    def _render_deck_prompt(self, kind):
        deck = self._g("master_deck_detailed")
        if kind == "grant":
            cards = [card_entry(c, up, i) for i, (c, up) in enumerate(self._g("card_grant_options") or [])]
            n = self._g("pending_card_grants") or 1
        else:
            idx = {"enchant": "enchant_select_deck_indices", "upgrade": "upgrade_select_deck_indices",
                   "transform": "transform_select_deck_indices"}.get(kind)
            if idx:
                cards = [card_entry(deck[j][0], deck[j][1], i) for i, j in enumerate(self._g(idx))]
            else:
                cards = [card_entry(c, up, i) for i, (c, up) in enumerate(self._g("card_removal_options") or [])]
            n = self._g({"enchant": "pending_deck_enchants", "upgrade": "pending_deck_upgrades",
                         "transform": "pending_deck_transforms"}.get(kind, "pending_deck_removals")) or 1
        self._prompt = kind
        return self._base("card_select", cards=cards, min_select=min(n, len(cards)), max_select=min(n, len(cards)),
                          purpose={"remove": "remove", "upgrade": "upgrade", "transform": "transform",
                                   "enchant": "enchant", "grant": "add"}.get(kind, kind))

    def _render_combat(self):
        cs = self.rm.combat_state()
        s = cs.get_state()
        if s.get("type") == "card_select":
            st = self._base("card_select", cards=[card_entry(c["id"], c.get("upgraded"), i) for i, c in enumerate(s.get("cards") or [])],
                            min_select=0 if s.get("can_skip") else s.get("min_select"), max_select=s.get("max_select"))
            # the run-level HP is only written back when the fight ends: read the fight's own
            st["player"]["hp"], st["player"]["max_hp"], st["player"]["block"] = cs.player_hp()
            if st["context"]["room_type"] == "Event":
                st["context"]["room_type"] = "Monster"
            return st
        acts = self._acts()
        playable = {a["hand_idx"] for a in acts if a.get("action") == "play_card"}
        hand = [{**card_entry(c["id"], c.get("upgraded"), i), "cost": c.get("cost"), "can_play": i in playable,
                 "target_type": c.get("target")} for i, c in enumerate(s.get("hand") or [])]
        alive = [e for e in s.get("enemies") or [] if e.get("is_alive", e.get("hp", 0) > 0)]
        enemies = [{"index": i, "name": e.get("id"), "hp": e.get("hp"), "max_hp": e.get("max_hp"), "block": e.get("block"),
                    "powers": [{"name": p.get("id"), "amount": p.get("amount")} for p in e.get("powers") or []],
                    "intents": [{"type": str(e.get("intent", "")).title(), "damage": e.get("intent_damage"),
                                 "hits": e.get("intent_hits")}]} for i, e in enumerate(alive)]
        pl = s.get("player") or {}
        st = self._base("combat_play", round=s.get("round"), energy=pl.get("energy"), max_energy=pl.get("max_energy"),
                        hand=hand, enemies=enemies, draw_pile_count=s.get("draw_pile_count"),
                        discard_pile_count=s.get("discard_pile_count"))
        st["player"]["hp"], st["player"]["block"], st["player"]["max_hp"] = pl.get("hp"), pl.get("block"), pl.get("max_hp")
        if st["context"]["room_type"] == "Event":   # our engine labels an event's fight a Monster room
            st["context"]["room_type"] = "Monster"
        return st

    def _map(self):
        rm = self.rm
        if self._g("phase") == "boss_relic":
            # at the act start our engine already shows the next act's map: read it off the
            # peeked copy (past the Ancient's offer)
            self._peek_next_act()
            rm = self._next.get("rm") or rm
        nodes = rm.act_map()
        rows = {}
        for n in nodes:
            if n["point_type"] in ("ancient", "boss"):
                continue
            rows.setdefault(n["row"], []).append({"col": n["col"], "row": n["row"], "type": ROOM.get(n["point_type"], "Unknown"),
                                                  "children": [{"col": c[0], "row": c[1]} for c in n["children"] if c[1] < 16],
                                                  "visited": False, "current": False})
        cur = None if rm is not self.rm else (self.rm.current_room() or {}).get("map_coord")
        ctx = self._next["ctx"] if rm is not self.rm else self._context()
        return {"type": "map", "context": ctx, "rows": [sorted(rows[r], key=lambda n: n["col"]) for r in sorted(rows)],
                "boss": {"id": rm.pinned_boss_encounter_id()},
                "current_coord": {"col": cur[0], "row": cur[1]} if cur else None}

    # ── actions in our engine's vocabulary ─────────────────────────────────
    def _action(self, name, args):
        acts = self._acts()

        def take(pred, what):
            for a in acts:
                if pred(a):
                    self.rm.take_action(a)
                    return self._render()
            return {"type": "error", "message": f"rssim: no sts2-rs action for {what}; have {sorted({a.get('action') for a in acts})}"}

        if name == "choose_option" and self._g("phase") == "boss_relic":
            i = int(args.get("option_index", 0))
            self._at_ancient, self._next = False, None
            return take(lambda a: a.get("action") == "pick_boss_relic" and a.get("idx") == i, f"boss relic {i}")
        if name == "choose_option":
            i = int(args.get("option_index", 0))
            return take(lambda a: a.get("action") in ("event_choice", "rest_option", "pick_boss_relic") and a.get("idx") == i,
                        f"option {i}")
        if name == "select_map_node" and self._g("phase") == "boss_relic":
            if int(args.get("row", 1)) == 0:
                self._at_ancient = True
                return self._render()
            skip = [a for a in acts if a.get("action") == "skip_relic"]   # walking past the Ancient
            self.rm.take_action(skip[0] if skip else acts[0])
            self._at_ancient, self._next = False, None
            acts = self._acts()
        if name == "select_map_node":
            return take(lambda a: a.get("action") == "move" and a.get("col") == int(args["col"]), f"map col {args.get('col')}")

        if name in ("select_card_reward", "select_bundle"):
            i = int(args.get("card_index", args.get("bundle_index", 0)))
            return take(lambda a: a.get("action") == "pick_card_reward" and a.get("idx") == i, f"reward {i}")
        if name == "skip_card_reward":
            return take(lambda a: a.get("action") == "pick_card_reward" and a.get("idx") is None, "skip reward")
        if name == "select_cards":
            idx = [int(x) for x in str(args.get("indices", "")).split(",") if x != ""]
            if self._g("phase") == "combat":
                multi = ((self.rm.combat_state().get_state() or {}).get("max_select") or 1) > 1
                if len(idx) == 1 and not multi and any(a.get("action") == "combat_resolve_choice" for a in acts):
                    return take(lambda a: a.get("action") == "combat_resolve_choice" and a.get("option") == idx[0], "combat pick")
                self.rm.take_action({"action": "combat_resolve_choice_selection", "selection": idx})
                return self._render()
            if any(a.get("action") == "smith_card" for a in acts):
                return take(lambda a: a.get("action") == "smith_card" and a.get("idx") == idx[0], f"smith {idx}")
            if self._g("phase") == "card_reward" and self._g("card_offer_choose_screen"):
                return take(lambda a: a.get("action") == "pick_card_reward" and a.get("idx") == idx[0], f"card offer {idx}")
            kind = self._deck_prompt()
            if kind == "grant":
                names = [c for c, _ in self._g("card_grant_options")]
                want = [names[i] for i in idx]
                for w in want:
                    cur = [c for c, _ in self._g("card_grant_options")]
                    self.rm.take_action({"action": "grant_deck_card", "idx": cur.index(w)})
                return self._render()
            # one pick per action; picked cards leave the eligible list, so map the original
            # positions onto the shrinking list
            for k, i in enumerate(sorted(idx)):
                cur = [a for a in self._acts() if str(a.get("action", "")).endswith("_deck_card")]
                if not cur:
                    break
                j = i - k
                self.rm.take_action(next((a for a in cur if a.get("idx") == j), cur[0]))
            return self._render()
        if name == "skip_select":
            if self._g("phase") == "combat":
                self.rm.take_action({"action": "combat_resolve_choice_selection", "selection": []})
                return self._render()
            if self._g("card_offer_choose_screen"):
                return take(lambda a: a.get("action") == "pick_card_reward" and a.get("idx") is None, "skip card offer")
            if self._g("pending_enchant_optional"):
                self.rm.decline_optional_deck_enchants()
                return self._render()
            return take(lambda a: a.get("action") in ("skip_relic",) or a.get("idx") is None, "skip select")
        if name == "leave_room":
            return take(lambda a: a.get("action") in ("leave_shop",), "leave")
        if name == "buy_card":
            i = int(args.get("card_index", 0))
            return take(lambda a: a.get("action") == "shop_buy_card" and a.get("idx") == i, f"buy card {i}")
        if name == "buy_relic":
            i = int(args.get("relic_index", 0))
            return take(lambda a: a.get("action") == "shop_buy_relic" and a.get("idx") == i, f"buy relic {i}")
        if name == "buy_potion":
            i = int(args.get("potion_index", 0))
            return take(lambda a: a.get("action") == "shop_buy_potion" and a.get("idx") == i, f"buy potion {i}")
        if name == "remove_card":
            return take(lambda a: a.get("action") == "shop_remove_card", "remove card")
        if name == "play_card":
            i = int(args["card_index"])
            cand = [a for a in acts if a.get("action") == "play_card" and a.get("hand_idx") == i]
            if not cand:
                return {"type": "error", "message": f"rssim: card {i} not playable"}
            t = args.get("target_index")
            alive = [k for k, e in enumerate((self.rm.combat_state().get_state() or {}).get("enemies") or [])
                     if e.get("is_alive", e.get("hp", 0) > 0)]
            slot = alive[int(t)] if t is not None and int(t) < len(alive) else None
            a = next((c for c in cand if c.get("target_index") == slot), None) or \
                min(cand, key=lambda x: -1 if x.get("target_index") is None else x["target_index"])
            self.rm.take_action(a)
            return self._render()
        if name == "end_turn":
            return take(lambda a: a.get("action") == "end_turn", "end turn")
        if name in ("proceed", "noop"):
            if acts:
                self.rm.take_action(acts[0])
            return self._render()
        return {"type": "error", "message": f"rssim: unsupported action {name}"}

    def resolve(self, name, args):
        """The sts2-rs action dict _action would take for one of our out-of-combat actions, without
        taking it (None when it is not one step: the Ancient walk, multi-card picks). For a teacher
        policy on the same RunManager (teacher.py)."""
        acts, phase = self._acts(), self._g("phase")

        def find(pred):
            return next((a for a in acts if pred(a)), None)

        i = args.get("option_index", args.get("card_index", args.get("bundle_index", args.get("relic_index",
                     args.get("potion_index", 0)))))
        if name == "choose_option" and phase == "boss_relic":
            return find(lambda a: a.get("action") == "pick_boss_relic" and a.get("idx") == int(i))
        if name == "choose_option":
            return find(lambda a: a.get("action") in ("event_choice", "rest_option", "pick_boss_relic") and a.get("idx") == int(i))
        if name == "select_map_node":
            return None if phase == "boss_relic" else find(lambda a: a.get("action") == "move" and a.get("col") == int(args["col"]))
        if name in ("select_card_reward", "select_bundle"):
            return find(lambda a: a.get("action") == "pick_card_reward" and a.get("idx") == int(i))
        if name == "skip_card_reward":
            return find(lambda a: a.get("action") == "pick_card_reward" and a.get("idx") is None)
        if name == "select_cards" and phase != "combat":
            idx = [int(x) for x in str(args.get("indices", "")).split(",") if x != ""]
            if len(idx) != 1:
                return None
            if any(a.get("action") == "smith_card" for a in acts):
                return find(lambda a: a.get("action") == "smith_card" and a.get("idx") == idx[0])
            if phase == "card_reward" and self._g("card_offer_choose_screen"):
                return find(lambda a: a.get("action") == "pick_card_reward" and a.get("idx") == idx[0])
            if self._deck_prompt() == "grant":
                return find(lambda a: a.get("action") == "grant_deck_card" and a.get("idx") == idx[0])
            return find(lambda a: str(a.get("action", "")).endswith("_deck_card") and a.get("idx") == idx[0])
        if name == "leave_room":
            return find(lambda a: a.get("action") == "leave_shop")
        if name in ("buy_card", "buy_relic", "buy_potion"):
            kind = {"buy_card": "shop_buy_card", "buy_relic": "shop_buy_relic", "buy_potion": "shop_buy_potion"}[name]
            return find(lambda a: a.get("action") == kind and a.get("idx") == int(i))
        if name == "remove_card":
            return find(lambda a: a.get("action") == "shop_remove_card")
        return None

    # ── combat by sts2-rs's MCTS ────────────────────────────────────────────
    def play_combat(self, st, on_state=None, on_step=None):
        """Play the current fight to its end with MctsCombatAgent; on_state gets a light combat
        state (enemies, round, player HP, potions) after every step. Returns the next state."""
        for _ in range(4000):
            if self._g("is_over") or self._g("phase") != "combat":
                break
            if on_state is not None:
                on_state(self._light_combat())
            if self.agent.step(self.rm) is None:
                break
        return self._render()

    def _light_combat(self):
        s = self.rm.combat_state().get_state()
        if "enemies" not in s:
            return {}
        return {"enemies": [{"name": e.get("id"), "hp": e.get("hp") if e.get("is_alive", True) else 0, "max_hp": e.get("max_hp")}
                            for e in s.get("enemies") or []],
                "round": s.get("round"), "player": {"hp": (s.get("player") or {}).get("hp"),
                                                    "potions": [{"name": p} for p in self._g("potions")]}}

    def _determinize(self, seed):
        """Shift the run's RNG stream counters by seeded amounts so act-start MC plays see
        different futures (card rewards, encounters, shuffles)."""
        r = random.Random(seed)
        ctr = self.rm.rng_counters()
        self.rm.debug_set_rng_counters(self._g("run_seed"), {k: v + r.randrange(1, 50) for k, v in ctr.items()})
        return {"type": "ok"}


if __name__ == "__main__":
    if sys.argv[1:2] == ["--make-db"]:
        print(make_db(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "data")))
