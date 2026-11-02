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
_LAST_BOSS = {}   # ascension -> the host run's last boss node on its act-3 map


def last_boss_node(asc):
    """(col, row) of the run's last boss on the host run's act-3 map (seed 0): the second boss
    at A10 (Double Boss), the only one below."""
    if asc not in _LAST_BOSS:
        rm = sts2_rs.RunManager(seed=0, character="Ironclad", ascension_level=asc, game_version=GAME_VERSION)
        bosses = [(n["col"], n["row"]) for n in rm.act_map_for(3) if n["point_type"] == "boss"]
        _LAST_BOSS[asc] = max(bosses, key=lambda c: c[1])
    return _LAST_BOSS[asc]


def is_last_boss(ctx):
    """Our engine's act-3 bosses are on floors 15 and 16 (A10's second boss), every run."""
    asc = int(ctx.get("ascension") or 0)
    return ctx.get("act") == 3 and ctx.get("room_type") == "Boss" and ctx.get("floor") == (16 if asc >= 10 else 15)


def want_override(tok):
    """Add a "key=value" knob to STS2_RUNAI_MCTS_OVERRIDES unless the key is set (the agents
    read it when they are built)."""
    cur = os.environ.get("STS2_RUNAI_MCTS_OVERRIDES", "").strip()
    if tok.split("=")[0] not in {t.split("=")[0].strip() for t in cur.split(",") if t.strip()}:
        os.environ["STS2_RUNAI_MCTS_OVERRIDES"] = ",".join(x for x in (cur, tok) if x)


def _piles(rm):
    d = sts2_rs.state_to_dict(rm.combat_state()).get("deck") or {}
    return {k: sorted((c.get("id"), c.get("upgraded")) for c in v) for k, v in d.items() if isinstance(v, list)}


def same_board(a, b):
    """Two hosts show the same board: the visible state, potions, relic counters and the cards
    of every pile (the draw order is not compared: the search never reads it)."""
    try:
        return (a.combat_state().get_state() == b.combat_state().get_state() and a.potions() == b.potions()
                and a.relic_counters() == b.relic_counters() and _piles(a) == _piles(b))
    except Exception:
        return False


class RsFighter:
    def __init__(self, sims=1000, boss_sims=4000, boss3_sims=None, seed=0, elite_sims=None, reuse=False,
                 keep_host=False):
        # boss3_sims: the act-3 bosses' budget (default boss_sims): they gate the run;
        # elite_sims: the elites' (default sims)
        self.sims, self.boss_sims, self.boss3_sims, self.seed = sims, boss_sims, boss3_sims or boss_sims, seed
        self.elite_sims = elite_sims or sims
        # reuse: one agent per fight under sts2-rs's tree reuse (GH #830): a search starts from the
        # subtree of the previous decision's move when the new board matches one of its nodes
        self.reuse, self.agent = reuse, None
        # keep_host (implied by reuse): search on the host that made the last move while it matches our
        # board, so sts2-rs's own per-turn counters stay exact (combat_wire does not carry them)
        self.keep = reuse or keep_host
        self.cont = None      # keep: the host after the agent's last move, kept while it matches our board
        if reuse:
            want_override("tree_reuse=1")
        self.stats = Counter()
        self.fails = []       # the first failures of the run (kind, detail), for the run record
        self.k = 0            # decisions searched (the agent seed mixes it in)
        self.host = None      # host RunManager after the last searched play (answers its prompt)

    # ── one decision ────────────────────────────────────────────────────────
    def _agent(self, st):
        if self.agent is not None:
            return self.agent
        ctx = st.get("context") or {}
        self.k += 1
        s = zlib.crc32(f"{self.seed}:{ctx.get('act')}:{ctx.get('floor')}:{self.k}".encode()) & 0x7FFFFFFF
        room = ctx.get("room_type")
        boss = self.boss3_sims if ctx.get("act") == 3 and room == "Boss" else self.boss_sims
        agent = sts2_rs.MctsCombatAgent(seed=s, sims=self.elite_sims if room == "Elite" else self.sims, boss_sims=boss)
        if self.reuse:
            self.agent = agent
        return agent

    def _step(self, st, rm):
        rec = self._agent(st).step(rm)
        if rec is not None and rec.get("retained_visits") is not None:
            self.stats["reused"] += 1
            self.stats["retained_visits"] += int(rec["retained_visits"])
        return rec

    def _host(self, st, wire):
        ctx = st.get("context") or {}
        asc = int(ctx.get("ascension") or 0)
        wire = {**wire, "game_version": GAME_VERSION}
        # character: the adapter otherwise infers the class from card ids, which fails for a
        # deck with no Ironclad-prefixed card left
        cs = sts2_rs.state_from_dict(adapter.translate(wire, character="Ironclad", ascension=asc))
        rm = sts2_rs.RunManager(seed=0, character="Ironclad", ascension_level=asc, game_version=GAME_VERSION)
        if is_last_boss(ctx):
            # the run's last fight: stand the host on its last boss node, so the agent searches
            # for the win alone (no HP / potions kept for later; sts2-rs's last-boss objective)
            col, row = last_boss_node(asc)
            rm.debug_place_on_map(3, col, row, [], int(ctx.get("floor") or 0), 50)
            self.stats["last_boss"] += 1
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
            cont, self.cont = self.cont, None
            try:
                rm = self._host(st, wire)
            except (UnsupportedStateError, ValueError, KeyError, TypeError, IndexError) as e:
                self.stats["import_fail"] += 1
                self.stats["import_fail:" + type(e).__name__] += 1
                self._fail("import", f"{type(e).__name__}: {str(e)[:200]}")
                return None
            if cont is not None:
                # the host that made the last move carries what the wire lacks (cards played, HP
                # lost, hits this turn...), which tree reuse's node match reads: go on with it
                # while it shows the board we have (a chance edge, e.g. the enemy turn, mostly differs)
                if same_board(cont, rm):
                    rm = cont
                    self.stats["host_kept"] += 1
                else:
                    self.stats["host_rebuilt"] += 1
            rec = self._step(st, rm)
            if rec is None or not rec.get("is_combat"):
                self.stats["no_action"] += 1
                return None
            act = self._ours(st, rec["action"])
            if act is not None:
                self.stats["searched"] += 1
                self.host = rm   # took the same action on the true board: it answers the next prompt
                if self.keep:
                    self.cont = rm
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
            slot = a.get("potion_slot")
            args = {"potion_index": slot}
            if a.get("target_index") is not None:
                args["target_index"] = a["target_index"]
            else:
                # sts2-rs drinks every potion untargeted and aims an AnyEnemy one at the first
                # living enemy (as the original Sts2Headless did); our engine wants the target
                pot = next((p for p in (st.get("player") or {}).get("potions") or [] if p and p.get("index") == slot), None)
                if pot is not None and pot.get("target_type") == "AnyEnemy":
                    args["target_index"] = 0
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
        rec = self._step(st, rm)
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
            if self.keep:
                self.cont = rm
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
        if self.keep:
            self.cont = rm
        return ("select_cards", {"indices": ",".join(idx)})

    def _fail(self, kind, detail):
        if len(self.fails) < 12:
            self.fails.append((kind, detail))

    # ── a whole fight ───────────────────────────────────────────────────────
    def play_combat(self, sim, state, fallback=None, max_actions=600, on_state=None, on_step=None):
        """Play until the combat ends (TurnPlanner.play_combat's contract); decisions the agent
        cannot take go to fallback.plan(state)'s first action (or end_turn / skip)."""
        n = 0
        self.agent = self.cont = None   # a fight's own agent and host (reuse)
        if on_state:
            on_state(state)
        while state.get("decision") in ("combat_play", "card_select") and n < max_actions:
            a = self.choose(sim, state)
            if a is None:
                self.stats["fallback"] += 1
                self.cont = None   # the host did not take the fallback's move
                path = fallback.plan(state)[0] if fallback is not None else []
                a = path[0] if path else (("end_turn", {}) if state.get("decision") == "combat_play" else ("skip_select", {}))
            if on_step:
                on_step(state, a)
            nxt = sim.act(a[0], **a[1])
            n += 1
            if nxt.get("type") == "error":
                self.stats["exec_error"] += 1
                self._fail("exec", f"{a} -> {str(nxt.get('message'))[:160]} | hand "
                                   f"{[(c.get('id'), c.get('cost'), c.get('can_play')) for c in state.get('hand') or []]} "
                                   f"energy {state.get('energy')} enemies {len(state.get('enemies') or [])} "
                                   f"pots {[(p.get('index'), p.get('name')) for p in (state.get('player') or {}).get('potions') or [] if p]}")
                self.host = self.cont = None
                nxt = sim.act("end_turn") if state.get("decision") == "combat_play" else sim.act("skip_select")
                if nxt.get("type") == "error":
                    break
            state = nxt
            if on_state:
                on_state(state)
        if self.agent is not None:   # what tree reuse did in this fight (sts2-rs's tally by reason)
            for k, v in self.agent.tree_reuse_stats().items():
                if isinstance(v, int) and k not in ("reused", "retained_visits"):
                    self.stats["tr_" + k] += v
        self.agent = self.cont = None
        return state
