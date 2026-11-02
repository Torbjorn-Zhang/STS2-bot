"""Fights in OUR engine (the real game DLL) played by sts2-rs's combat MCTS.

At each combat_play decision the board (Fix12 "combat_wire") goes through spire-mentor's
sts2_hint adapter (vendor/) into sts2_rs.state_from_dict, is installed as the live fight of a
fresh host RunManager (debug_enter_combat: the room tier picks sims or boss_sims) and the
deployed run agent (MctsCombatAgent: honest beliefdraw search, the draw order hidden at the root)
picks one action, which is played here. A card_select the played card opens is answered by the
same agent on the host, which took that play on the true board, when its pending choice offers
the same cards; anything else (an import the adapter refuses, a prompt that does not match) is
left to the fallback planner for that decision.

  fighter = RsFighter(sims=1000, boss_sims=4000)
  st = fighter.play_combat(sim, st, fallback=planner)      # like TurnPlanner.play_combat
"""
import os
import sys
import zlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))
from rssim import card_id, sts2_rs  # noqa: E402  (rssim locates the sts2_rs module)
from sts2_hint import adapter  # noqa: E402
from sts2_hint.errors import UnsupportedStateError  # noqa: E402

GAME_VERSION = "v0.111.0"
TIER = {"Monster": "monster", "Elite": "elite", "Boss": "boss"}


class RsFighter:
    def __init__(self, sims=1000, boss_sims=4000, seed=0):
        self.sims, self.boss_sims, self.seed = sims, boss_sims, seed
        self.stats = Counter()
        self.k = 0            # decisions searched (the agent seed mixes it in)
        self.host = None      # host RunManager after the last searched play (answers its prompt)

    # ── one decision ────────────────────────────────────────────────────────
    def _agent(self, st):
        ctx = st.get("context") or {}
        self.k += 1
        s = zlib.crc32(f"{self.seed}:{ctx.get('act')}:{ctx.get('floor')}:{self.k}".encode()) & 0x7FFFFFFF
        return sts2_rs.MctsCombatAgent(seed=s, sims=self.sims, boss_sims=self.boss_sims)

    def _host(self, st, wire):
        ctx = st.get("context") or {}
        asc = int(ctx.get("ascension") or 0)
        wire = {**wire, "game_version": GAME_VERSION}
        cs = sts2_rs.state_from_dict(adapter.translate(wire, ascension=asc))
        rm = sts2_rs.RunManager(seed=0, character="Ironclad", ascension_level=asc, game_version=GAME_VERSION)
        rm.debug_enter_combat(cs, TIER.get(ctx.get("room_type"), "monster"))
        return rm

    def choose(self, sim, st):
        """Our engine's (action, args) for this combat decision, or None (use the fallback)."""
        d = st.get("decision")
        if d == "combat_play":
            self.host = None
            wire = sim.send({"cmd": "combat_wire"})
            if wire.get("type") != "combat_wire":
                self.stats["wire_error"] += 1
                return None
            try:
                rm = self._host(st, wire)
            except (UnsupportedStateError, ValueError, KeyError, TypeError, IndexError) as e:
                self.stats["import_fail"] += 1
                self.stats["import_fail:" + type(e).__name__] += 1
                return None
            rec = self._agent(st).step(rm)
            if rec is None or not rec.get("is_combat"):
                self.stats["no_action"] += 1
                return None
            act = self._ours(st, rec["action"])
            if act is not None:
                self.stats["searched"] += 1
                self.host = rm if act[0] == "play_card" else None
            return act
        if d == "card_select":
            return self._answer_prompt(st)
        return None

    def _ours(self, st, a):
        kind = a.get("action")
        if kind == "end_turn":
            return ("end_turn", {})
        if kind == "play_card":
            hand = st.get("hand") or []
            i = a.get("hand_idx")
            if i is None or not 0 <= i < len(hand):
                self.stats["bad_hand_idx"] += 1
                return None
            args = {"card_index": hand[i].get("index", i)}
            if a.get("target_index") is not None:
                args["target_index"] = a["target_index"]
            return ("play_card", args)
        if kind == "drink_potion":
            args = {"potion_index": a.get("potion_slot")}
            if a.get("target_index") is not None:
                args["target_index"] = a["target_index"]
            return ("use_potion", args)
        self.stats["unmapped:" + str(kind)] += 1
        return None

    def _answer_prompt(self, st):
        """A card_select opened by the play the host also took: ask the agent there."""
        rm, self.host = self.host, None
        if rm is None:
            self.stats["prompt_no_host"] += 1
            return None
        cs = rm.combat_state()
        s = cs.get_state() if cs is not None else {}
        if s.get("type") != "card_select":
            self.stats["prompt_host_not_prompt"] += 1
            return None
        rs_cards = [(card_id(c["id"]), bool(c.get("upgraded"))) for c in s.get("cards") or []]
        ours = st.get("cards") or []
        our_cards = [((c.get("id") or "").split(".", 1)[-1], bool(c.get("upgraded"))) for c in ours]
        if Counter(rs_cards) != Counter(our_cards):
            self.stats["prompt_mismatch"] += 1
            return None
        rec = self._agent(st).step(rm)
        if rec is None:
            self.stats["prompt_no_action"] += 1
            return None
        a = rec["action"]
        kind = a.get("action")
        if kind == "combat_resolve_choice":
            pick = [] if a.get("option") is None else [a["option"]]
        elif kind == "combat_resolve_choice_take_first":
            pick = list(range(int(a.get("count") or 0)))
        elif "selection" in a:
            pick = list(a.get("selection") or [])
        else:
            self.stats["prompt_unmapped:" + str(kind)] += 1
            return None
        if not pick:
            self.stats["prompt_skip"] += 1
            return ("skip_select", {})
        used, idx = set(), []
        for j in pick:   # rs option j -> the first unused card of ours that is the same card
            want = rs_cards[j]
            k = next((k for k, c in enumerate(our_cards) if c == want and k not in used), None)
            if k is None:
                self.stats["prompt_mismatch"] += 1
                return None
            used.add(k)
            idx.append(str(ours[k].get("index", k)))
        self.stats["prompt_answered"] += 1
        return ("select_cards", {"indices": ",".join(idx)})

    # ── a whole fight ───────────────────────────────────────────────────────
    def play_combat(self, sim, state, fallback=None, max_actions=600, on_state=None, on_step=None):
        """Play until the combat ends (TurnPlanner.play_combat's contract); decisions the agent
        cannot take go to fallback.plan(state)'s first action (or end_turn / skip)."""
        n = 0
        if on_state:
            on_state(state)
        while state.get("decision") in ("combat_play", "card_select") and n < max_actions:
            a = self.choose(sim, state)
            if a is None:
                self.stats["fallback"] += 1
                path = fallback.plan(state)[0] if fallback is not None else []
                a = path[0] if path else (("end_turn", {}) if state.get("decision") == "combat_play" else ("skip_select", {}))
            if on_step:
                on_step(state, a)
            nxt = sim.act(a[0], **a[1])
            n += 1
            if nxt.get("type") == "error":
                self.stats["exec_error"] += 1
                self.host = None
                nxt = sim.act("end_turn") if state.get("decision") == "combat_play" else sim.act("skip_select")
                if nxt.get("type") == "error":
                    break
            state = nxt
            if on_state:
                on_state(state)
        return state
