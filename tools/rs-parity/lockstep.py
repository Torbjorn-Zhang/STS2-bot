"""Lockstep parity: play the same seeded A10 Ironclad run in our engine (real v0.111.0 DLL) and in
sts2-rs (v0.111.0 mode) with one deterministic, engine-neutral policy, comparing the observable
state before every action. Stops at the first mismatch and prints both sides.

  python lockstep.py SEED [SEED...]       (seeds as our runs name them, e.g. full_145)

Policy: Neow/events take the first enabled option, the map takes the leftmost child, combat plays
the first playable card in hand order at the first living enemy (else ends the turn), card
rewards are skipped, rest sites rest, shops are left, prompts take the first card(s)."""
import json
import os
import sys
from collections import Counter

os.environ["STS2_RS_RAW_SEED"] = "1"
sys.path.insert(0, "/Users/szha0569/Daily/sts2-ai/harness")
sys.path.insert(0, "/Users/szha0569/Daily/sts2-ai/local/rs-py")
from simclient import Sim  # noqa: E402
import sts2_rs  # noqa: E402

VERBOSE = os.environ.get("LOCKSTEP_VERBOSE") == "1"
GOD = int(os.environ.get("LOCKSTEP_GOD", "0"))        # >0: both players start with this HP / max HP (deep coverage)
PICK = os.environ.get("LOCKSTEP_PICK", "first")        # card rewards: "first" card or "skip"


CARD_ALIAS = {"slothstatus": "sloth"}   # sts2-rs status-card ids


def ncard(cid):
    """'CARD.STRIKE_IRONCLAD' / 'STRIKE_IRONCLAD' / 'StrikeIronclad' / 'RageCard+' -> 'strikeironclad'."""
    s = str(cid)
    s = s.split(".", 1)[1] if s.startswith("CARD.") else s
    up = s.endswith("+")
    s = s.rstrip("+")
    if s.endswith("Card") and len(s) > 4 and s != s.upper():
        s = s[:-4]   # sts2-rs names some cards XCard / X_CARD to dodge enum clashes (ColossusCard, RAGE_CARD)
    elif s.endswith("_CARD"):
        s = s[:-5]
    n = s.replace("_", "").lower()
    return CARD_ALIAS.get(n, n) + ("+" if up else "")


ALIAS = {"backattackleft": "backattack", "backattackright": "backattack",   # sts2-rs power-id variants
         "battleworndummytimelimit": "timelimit"}


def nname(s):
    n = "".join(ch for ch in str(s).lower() if ch.isalnum())
    # ours falls back to the loc key for some powers ("MANGLE_POWER.title"); sts2-rs ids end in Power
    for suf in ("powertitle", "power"):
        if n.endswith(suf) and len(n) > len(suf):
            n = n[: -len(suf)]
            break
    return ALIAS.get(n, n)


class Ours:
    def __init__(self, seed):
        self.sim = Sim("Fix11", timeout=60)   # Fix11 = Fix10 + "upgraded" on hand cards
        self.st = self.sim.send({"cmd": "start_run", "character": "Ironclad", "seed": seed, "ascension": 10})
        if GOD:
            r = self.sim.send({"cmd": "set_player", "hp": GOD, "max_hp": GOD})
            self.st["player"] = r["player"]

    def kind(self):
        d = self.st.get("decision")
        return {"event_choice": "event", "map_select": "map", "combat_play": "combat", "card_reward": "card_reward",
                "rest_site": "rest", "shop": "shop", "card_select": "card_select", "bundle_select": "bundle",
                "game_over": "over"}.get(d, d or self.st.get("type"))

    def obs(self):
        s, k = self.st, self.kind()
        p = s.get("player") or {}
        o = {"kind": k}
        if k == "combat":
            o.update(round=s.get("round"), energy=s.get("energy"), draw=s.get("draw_pile_count"),
                     discard=s.get("discard_pile_count"),
                     # our hand entries carry no "upgraded" flag; the display name ends in "+"
                     hand=[ncard(c["id"]) + ("+" if c.get("upgraded") else "") for c in s.get("hand", [])],
                     # can_play already checks energy (and X-cost cards report a resolved cost of 3)
                     playable=[bool(c.get("can_play")) for c in s.get("hand", [])],
                     # intents are not compared: ours shows damage after Strength etc., sts2-rs the base value;
                     # a real damage difference shows up as an HP mismatch one step later
                     # names are not compared (ours shows display names, sts2-rs model ids such as
                     # DecimillipedeSegmentFront); order, HP, block and powers identify the enemies
                     enemies=[(e["hp"], e.get("block", 0),
                               tuple(sorted((nname(pw["name"]), pw.get("amount")) for pw in (e.get("powers") or []))))
                              for e in s.get("enemies", []) if e.get("hp", 0) > 0],
                     ppowers=tuple(sorted((nname(pw["name"]), pw.get("amount")) for pw in (s.get("player_powers") or []))))
        self._reward_is_select = False
        in_combat = self._in_combat()
        if k == "combat":
            self._last_kind = "combat"
        elif k == "card_select" and in_combat:
            self._last_kind = "card_select_combat"
        else:
            self._last_kind = k
        if k == "card_select" and s.get("min_select") == 0 and s.get("max_select") == 1 and not in_combat:
            k = "card_reward"   # a skippable single pick outside combat is a card reward in sts2-rs terms
            o["kind"] = k
            self._sel_index = [c.get("index", i) for i, c in enumerate(s.get("cards", []))]
            self._reward_is_select = True
        if k == "card_select":
            o.update(cards=[ncard(c["id"]) + ("+" if c.get("upgraded") else "") for c in s.get("cards", [])],
                     min_select=s.get("min_select"), max_select=min(s.get("max_select") or 0, len(s.get("cards", []))))
            self._sel_index = [c.get("index", i) for i, c in enumerate(s.get("cards", []))]
            return o
        self._rcount = {nname(r["name"]): r.get("counter") for r in (p.get("relics") or []) if r.get("counter") is not None}
        o.update(hp=p.get("hp"), block=p.get("block") if k == "combat" else None)
        if k != "combat":
            o.update(gold=p.get("gold"), max_hp=p.get("max_hp"),
                     deck=sorted(Counter(ncard(c["id"]) + ("+" if c.get("upgraded") else "") for c in (p.get("deck") or [])).items()),
                     n_relics=len(p.get("relics") or []), potions=sorted(nname(x["name"]) for x in (p.get("potions") or [])))
        if k == "event":
            o["options"] = [o2.get("text_key", "").rsplit(".", 1)[-1] for o2 in s.get("options", [])]
            o["avail"] = [o2.get("index", i) for i, o2 in enumerate(s.get("options", [])) if not o2.get("is_locked")]
        if k == "rest":
            o["options"] = [nname(o2.get("id") or o2.get("name") or o2.get("title")).replace("restsiteoption", "") for o2 in s.get("options", []) if o2.get("is_enabled", True)]
            self._rest_raw = s.get("options", [])
        if k == "map":
            o["choices"] = sorted((c["col"], c["row"]) for c in s.get("choices", []))
        if k == "card_reward":
            o["cards"] = [ncard(c["id"]) for c in s.get("cards", [])]
        self._bundle = k == "bundle"
        if self._bundle:   # sts2-rs shows a bundle pick as a card reward of bundles
            o["kind"] = "card_reward"
            o["bundles"] = [[nname(c["name"]) for c in b.get("cards", [])] for b in s.get("bundles", [])]
        return o

    def _in_combat(self):
        """A prompt right after a combat state (or another in-combat prompt) is inside that combat."""
        return getattr(self, "_last_kind", None) in ("combat", "card_select_combat")

    def act(self, a):
        """a: an engine-neutral action from policy()."""
        t = a[0]
        if t == "pick_reward" and getattr(self, "_bundle", False):
            self.st = self.sim.act("select_bundle", bundle_index=a[1])
            return
        if t in ("pick_reward", "skip_reward") and getattr(self, "_reward_is_select", False):
            if t == "pick_reward":
                self.st = self.sim.act("select_cards", indices=str(self._sel_index[a[1]]))
            else:
                self.st = self.sim.act("skip_select")
            return
        if t == "option":
            self.st = self.sim.act("choose_option", option_index=a[1])
        elif t == "move":
            self.st = self.sim.act("select_map_node", col=a[1], row=a[2])
        elif t == "play":
            kw = {"card_index": a[1]}
            if a[2] is not None:
                kw["target_index"] = a[2]
            self.st = self.sim.act("play_card", **kw)
        elif t == "end_turn":
            self.st = self.sim.act("end_turn")
        elif t == "skip_reward":
            self.st = self.sim.act("skip_card_reward")
        elif t == "pick_reward":
            self.st = self.sim.act("select_card_reward", card_index=a[1])
        elif t == "leave":
            self.st = self.sim.act("leave_room")
        elif t == "select_first":
            n = a[1]
            self.st = self.sim.act("select_cards", indices=",".join(str(i) for i in self._sel_index[:n]))
        elif t == "skip_select":
            self.st = self.sim.act("skip_select")
        else:
            self.st = self.sim.act("proceed")

    def close(self):
        self.sim.close()


class Rust:
    def __init__(self, seed):
        self.rm = sts2_rs.RunManager(character="Ironclad", string_seed=seed, ascension_level=10, neow_start=True, act1="roll",
                                     interactive_card_removal=True, interactive_card_upgrade=True,
                                     interactive_card_transform=True, direct_event_relic_grant=True, game_version="v0.111.0")
        if GOD:
            self.rm.set_player(hp=GOD, max_hp=GOD)

    def g(self, n):
        v = getattr(self.rm, n)
        return v() if callable(v) else v

    def kind(self):
        ph = self.g("phase")
        if ph == "event" and self.g("current_event_name") is None:
            self.rm.materialize_event()   # an Unknown room rolls its event lazily
        if any(str(a.get("action", "")).endswith("_deck_card") for a in self.rm.available_actions()):
            return "deck_select"          # enchant / upgrade / transform / remove picks queue as actions
        if ph == "combat":
            cs = self.rm.combat_state()
            if cs is not None and cs.get_state().get("type") == "card_select":
                return "card_select"
        return {"map_choice": "map", "run_over": "over", "rest_site": "rest"}.get(ph, ph)

    def obs(self):
        k = self.kind()
        o = {"kind": k}
        if k == "deck_select":
            acts = [a for a in self.rm.available_actions() if str(a.get("action", "")).endswith("_deck_card")]
            kind = acts[0]["action"].split("_")[0]
            deck = self.g("master_deck_detailed")
            idx = {"enchant": "enchant_select_deck_indices", "upgrade": "upgrade_select_deck_indices",
                   "transform": "transform_select_deck_indices"}.get(kind)
            if kind == "grant":   # pick-k-of-n card offers (Gorge): one grant_deck_card per pick
                n = self.g("pending_card_grants") or 1
                o.update(kind="card_select", cards=[ncard(c) + ("+" if up else "") for c, up in self.g("card_grant_options")],
                         min_select=n, max_select=n)
                return o
            if idx:
                cards = [ncard(deck[i][0]) + ("+" if deck[i][1] else "") for i in self.g(idx)]
            else:
                cards = [ncard(c) + ("+" if up else "") for c, up in self.g("card_removal_options")]
            n = self.g({"enchant": "pending_deck_enchants", "upgrade": "pending_deck_upgrades",
                        "transform": "pending_deck_transforms"}.get(kind, "pending_deck_removals")) or 1
            o.update(kind="card_select", cards=cards, min_select=n, max_select=min(n, len(cards)))
            return o
        if k == "card_select" and self.g("phase") == "combat":
            s = self.rm.combat_state().get_state()
            o.update(cards=[ncard(c["id"]) + ("+" if c.get("upgraded") else "") for c in s.get("cards", [])],
                     min_select=s.get("min_select"), max_select=min(s.get("max_select") or 0, len(s.get("cards", []))))
            return o
        if k == "combat":
            s = self.rm.combat_state().get_state()
            if "player" not in s:
                raise RuntimeError("rust combat state without player: " + json.dumps(s, default=str)[:1500]
                                   + " | actions " + json.dumps(self.rm.available_actions()[:10], default=str))
            pl = s["player"]
            o.update(round=s.get("round"), energy=pl.get("energy"), draw=s.get("draw_pile_count"),
                     discard=s.get("discard_pile_count"),
                     hand=[ncard(c["id"]) + ("+" if c.get("upgraded") else "") for c in s.get("hand", [])],
                     # the state's "playable" flag ignores Unplayable (Ascender's Bane); the legal list does not
                     playable=[any(a.get("action") == "play_card" and a.get("hand_idx") == i for a in self.rm.available_actions())
                               for i in range(len(s.get("hand", [])))],
                     enemies=[(e["hp"], e.get("block", 0),
                               tuple(sorted((nname(pw["id"]), pw.get("amount")) for pw in (e.get("powers") or []))))
                              for e in s.get("enemies", []) if e.get("is_alive", e.get("hp", 0) > 0)],
                     ppowers=tuple(sorted((nname(pw["id"]), pw.get("amount")) for pw in (pl.get("powers") or []))))
            o.update(hp=pl.get("hp"), block=pl.get("block"))
        else:
            o.update(hp=self.g("current_hp"), block=None, gold=self.g("gold"), max_hp=self.g("max_hp"),
                     deck=sorted(Counter(ncard(c) + ("+" if up else "") for c, up in self.g("master_deck_detailed")).items()),
                     n_relics=len(self.g("relics")), potions=sorted(nname(x) for x in self.g("potions")))
        if k == "event":
            o["options"] = [x[1] for x in (self.g("event_options") or [])]
            o["avail"] = sorted(a["idx"] for a in self.rm.available_actions() if a.get("action") == "event_choice")
        if k == "rest":
            o["options"] = [nname(n) for n, en in (self.g("rest_site_options") or []) if en]
        if k == "boss_relic":
            o["kind"] = "event"   # the act-start Ancient's offer; ours shows it as an event after the row-0 node
            o["options"] = [ncard(x.split(".", 1)[-1]) for x in (self.g("boss_relic_options") or [])]
            o["avail"] = sorted(a["idx"] for a in self.rm.available_actions() if a.get("idx") is not None)
            o["_rust_boss_relic"] = True
        if k == "map":
            o["choices"] = sorted((a["col"], a.get("row")) for a in self.rm.available_actions() if a.get("action") == "move")
        if k == "card_reward":
            bundles = self.g("card_reward_bundles")
            if bundles:
                o["bundles"] = [[ncard(c) for c in b] for b in bundles]
            else:
                o["cards"] = [ncard(c) for c in self.g("card_reward_option_ids")]
        return o

    def act(self, a):
        acts = self.rm.available_actions()
        t = a[0]

        def pick(pred):
            for x in acts:
                if pred(x):
                    return self.rm.take_action(x)
            raise RuntimeError(f"rust: no action for {a} among {acts[:8]}")
        if t == "option":
            pick(lambda x: x.get("action") in ("event_choice", "rest_option", "choose_option", "pick_boss_relic", "boss_relic") and x.get("idx") == a[1])
        elif t == "move":
            pick(lambda x: x.get("action") == "move" and x.get("col") == a[1])
        elif t == "play":
            # policy target 0 = the first living enemy: ours indexes the living enemies, sts2-rs the
            # enemy slots, so take this card's legal play with the lowest slot
            cand = [x for x in acts if x.get("action") == "play_card" and x.get("hand_idx") == a[1]]
            if not cand:
                raise RuntimeError(f"rust: no play for {a} among {acts[:8]}")
            self.rm.take_action(min(cand, key=lambda x: -1 if x.get("target_index") is None else x["target_index"]))
        elif t == "end_turn":
            pick(lambda x: x.get("action") == "end_turn")
        elif t == "skip_reward":
            pick(lambda x: x.get("action") == "pick_card_reward" and x.get("idx") is None)
        elif t == "pick_reward":
            pick(lambda x: x.get("action") == "pick_card_reward" and x.get("idx") == a[1])
        elif t == "select_first" and self.kind() == "deck_select":
            if any(x.get("action") == "grant_deck_card" for x in acts):
                want = [c for c, _ in self.g("card_grant_options")][:a[1]]
                for name in want:
                    cur = [c for c, _ in self.g("card_grant_options")]
                    self.rm.take_action({"action": "grant_deck_card", "idx": cur.index(name)})
                return
            # one pick per action; the picked card leaves the eligible list, so idx 0 each time
            # takes the same cards as ours selecting its first n
            for _ in range(a[1]):
                cur = [x for x in self.rm.available_actions() if str(x.get("action", "")).endswith("_deck_card")]
                if not cur:
                    break
                self.rm.take_action(min(cur, key=lambda x: x.get("idx", 0)))
        elif t == "select_first" and self.g("phase") == "combat":
            multi = (self.rm.combat_state().get_state().get("max_select") or 1) > 1
            if a[1] == 1 and not multi:
                pick(lambda x: x.get("action") == "combat_resolve_choice" and x.get("option") == 0)
            else:
                self.rm.take_action({"action": "combat_resolve_choice_selection", "selection": list(range(a[1]))})
        elif t in ("leave", "proceed"):
            names = [x.get("action") for x in acts]
            for want in ("shop_leave", "leave_shop", "leave", "proceed", "skip", "continue"):
                hit = [x for x in acts if str(x.get("action", "")) == want or str(x.get("action", "")).endswith("_" + want)]
                if hit:
                    return self.rm.take_action(hit[0])
            raise RuntimeError(f"rust: no leave/proceed among {sorted(set(names))}")
        else:
            raise RuntimeError(f"rust: unmapped action {a}; available {acts[:8]}")


def policy(o):
    k = o["kind"]
    if k == "event":
        return ("option", (o.get("avail") or [0])[0])
    if k == "map":
        c = min(o["choices"])
        return ("move", c[0], c[1])
    if k == "combat":
        for i, ok in enumerate(o["playable"]):
            if ok:
                return ("play", i, 0)
        return ("end_turn",)
    if k == "card_reward":
        if o.get("bundles"):
            return ("pick_reward", 0)
        return ("pick_reward", 0) if PICK == "first" and o.get("cards") else ("skip_reward",)
    if k == "rest":
        return ("option", 0)
    if k == "shop":
        return ("leave",)
    if k == "card_select":
        if not o.get("cards"):
            return ("skip_select",)
        return ("select_first", min(len(o["cards"]), max(1, o.get("min_select") or 0)))
    return ("proceed",)


SOFT = {"ppowers", "rcount", "min_select"}   # sts2-rs export gaps; min_select: whether a pick may be skipped (Discovery)
SOFT_SEEN = Counter()


def compare(a, b):
    keys = sorted(set(a) | set(b))
    d = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
    for k in [k for k in d if k in SOFT]:
        SOFT_SEEN[(k, str(d[k]))[:120]] += 1
        del d[k]
    return d


def lockstep(seed, max_steps=3000):
    us, rs = Ours(seed), Rust(seed)
    try:
        for step in range(max_steps):
            oa, ob = us.obs(), rs.obs()
            if oa["kind"] == "map" and ob.get("_rust_boss_relic"):
                row0 = [c for c in oa["choices"] if c[1] == 0]
                if row0:
                    us.act(("move", row0[0][0], 0))   # sts2-rs skips the map step into the Ancient node
                    oa = us.obs()
            if oa["kind"] == "map" and ob["kind"] == "event" and rs.g("current_event_name") == "FakeMerchant":
                # our adapter cannot present the Fake Merchant's custom layout and walks out (known):
                # take sts2-rs's Leave too
                SOFT_SEEN[("fake_merchant_skipped", seed)] += 1
                rs.act(("option", (ob.get("avail") or [0])[-1]))
                ob = rs.obs()
            ancient = ob.pop("_rust_boss_relic", None)
            if oa["kind"] == "event" and ob["kind"] == "event":
                xa = [nname(x) for x in oa.get("options", [])]
                xb = [nname(x) for x in ob.get("options", [])]
                # labels differ in decoration only ("Bone Tea" vs "Bone Tea (50g)"): prefix match
                # ours gives text keys (READ_PASSAGE), sts2-rs display labels ("Read a Passage"): the count
                # must match, the labels only roughly (logged as soft when they differ)
                same = len(xa) == len(xb)
                if same and xa != xb and not all(p.startswith(q) or q.startswith(p) for p, q in zip(xa, xb)):
                    SOFT_SEEN[("event_labels", str((xa, xb)))[:120]] += 1
                oa = dict(oa); ob = dict(ob)
                oa["options"] = xa
                ob["options"] = xa if same else xb
            if ancient:
                # ours applies the next-act heal on entering the Ancient node; check HP one step later
                oa = dict(oa); ob = dict(ob); oa.pop("hp", None); ob.pop("hp", None)
            # the Rust move action carries only the column: compare columns, act on ours
            ca, cb = dict(oa), dict(ob)
            rc_r = {nname(k): v for k, v in rs.rm.relic_counters().items()}
            rc_o = getattr(us, "_rcount", {})
            ca["rcount"] = tuple(sorted(rc_o.items()))
            cb["rcount"] = tuple(sorted((k, rc_r.get(k)) for k in rc_o))
            if oa["kind"] == "map" and ob["kind"] == "map":
                ca["choices"] = sorted(c for c, _ in oa["choices"])
                cb["choices"] = sorted(c for c, _ in ob["choices"])
            d = compare(ca, cb)
            if VERBOSE:
                print(step, oa)
            if d:
                return {"seed": seed, "step": step, "kind": (oa["kind"], ob["kind"]), "diff": d, "ours": oa, "rust": ob}
            if oa["kind"] == "over":
                return {"seed": seed, "step": step, "ok": True, "hp": oa.get("hp")}
            if oa["kind"] == "event" and "Crystal Sphere" in str(us.st.get("event_name")):
                # its minigame is auto-played by each harness's own heuristic: a known, by-design difference
                return {"seed": seed, "step": step, "ok": True, "note": "stopped at Crystal Sphere"}
            a = policy(oa)
            us.act(a)
            rs.act(a)
        return {"seed": seed, "step": max_steps, "ok": True, "note": "max_steps"}
    finally:
        us.close()


if __name__ == "__main__":
    for s in sys.argv[1:] or ["full_145"]:
        r = lockstep(s)
        print(json.dumps(r, default=str)[:3000], flush=True)
    if SOFT_SEEN:
        print("soft diffs:", dict(SOFT_SEEN.most_common(8)))
