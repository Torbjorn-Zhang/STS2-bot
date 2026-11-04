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


def is_first_double_boss(ctx):
    """The first act-3 boss under A10 DoubleBoss (floor 15): its end HP and potions are the
    second boss's, with no rest between."""
    return (int(ctx.get("ascension") or 0) >= 10 and ctx.get("act") == 3 and ctx.get("room_type") == "Boss"
            and ctx.get("floor") == 15)


def want_override(tok):
    """Add a "key=value" knob to STS2_RUNAI_MCTS_OVERRIDES unless the key is set (the agents
    read it when they are built)."""
    cur = os.environ.get("STS2_RUNAI_MCTS_OVERRIDES", "").strip()
    if tok.split("=")[0] not in {t.split("=")[0].strip() for t in cur.split(",") if t.strip()}:
        os.environ["STS2_RUNAI_MCTS_OVERRIDES"] = ",".join(x for x in (cur, tok) if x)


def _piles(rm):
    d = sts2_rs.state_to_dict(rm.combat_state()).get("deck") or {}
    return {k: sorted((c.get("id"), c.get("upgraded")) for c in v) for k, v in d.items() if isinstance(v, list)}


def _piles_any(rm):
    """_piles that also keeps the deck's non-card lists (_piles raises on them, so same_board says
    "differs" there; left as it is, the diagnostic must not)."""
    d = sts2_rs.state_to_dict(rm.combat_state()).get("deck") or {}
    return {k: sorted((c.get("id"), c.get("upgraded")) for c in v) if all(isinstance(c, dict) for c in v) else v
            for k, v in d.items() if isinstance(v, list)}


def _diff(a, b, path="", out=None, limit=6):
    """Paths (with both values, short) where two nested dicts/lists differ, at most `limit`."""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            _diff(a.get(k), b.get(k), f"{path}.{k}", out, limit)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: {str(a)[:40]} | {str(b)[:40]}")
    return out


def _norm(x, key=None):
    """A board without the differences that are only representation: power lists in another order,
    dead enemies the host still lists (our engine drops them)."""
    if isinstance(x, dict):
        return {k: _norm(v, k) for k, v in x.items()}
    if isinstance(x, list):
        xs = [_norm(v) for v in x]
        if key == "enemies":
            xs = [e for e in xs if not (isinstance(e, dict) and (e.get("hp") or 0) <= 0)]
        if xs and all(isinstance(v, dict) and "id" in v for v in xs) and key != "enemies":
            xs = sorted(xs, key=lambda v: (str(v.get("id")), str(v.get("amount"))))
        return xs
    return x


def _view(rm):
    """rm's visible state as a copy rebuilt from its state_to_dict shows it. sts2-rs's live
    get_state() keeps listing the powers the fight was imported with and none gained since
    (Inflame's STRENGTH, Battle Trance's NO_DRAW, a potion's...), though the fight itself has them
    (state_to_dict does, and so does the copy's get_state)."""
    cs = rm.combat_state()
    try:
        return sts2_rs.state_from_dict(sts2_rs.state_to_dict(cs)).get_state()
    except Exception:  # noqa: BLE001  (a board state_from_dict refuses, e.g. a cost that expires on play)
        return cs.get_state()


def board_diff(host, real):
    """Where the host that took a move shows another board than the true one after the same move
    (same_board's fields, the visible state as _view shows it): host value | true value per path."""
    try:
        ga, gb = _norm(_view(host)), _norm(_view(real))
        return (_diff(ga, gb, "board") + _diff(_piles_any(host), _piles_any(real), "piles")
                + _diff(host.potions(), real.potions(), "potions") + _diff(host.relic_counters(), real.relic_counters(), "relics"))[:6]
    except Exception as e:  # noqa: BLE001  (a diagnostic only)
        return [f"diff failed: {type(e).__name__}: {str(e)[:120]}"]


def turn_diff(host, real):
    """After end_turn: where the host's own enemy turn (and the next turn's start) left another board
    than the true one, on the fields no draw decides: each enemy's hp / block / intent / powers, the
    player's hp / block / energy / powers. [(path, host value, true value)]"""
    try:
        ga, gb = _norm(_view(host)), _norm(_view(real))
    except Exception as e:  # noqa: BLE001  (a diagnostic only)
        return [("failed", type(e).__name__, str(e)[:60])]
    ea, eb = ga.get("enemies") or [], gb.get("enemies") or []
    if [e.get("id") for e in ea] != [e.get("id") for e in eb]:
        return [("enemies", [e.get("id") for e in ea], [e.get("id") for e in eb])]
    out = [(f"{x.get('id')}.{k}", x.get(k), y.get(k)) for x, y in zip(ea, eb)
           for k in ("hp", "block", "intent", "intent_damage", "intent_hits", "powers") if x.get(k) != y.get(k)]
    pa, pb = ga.get("player") or {}, gb.get("player") or {}
    return out + [(f"player.{k}", pa.get(k), pb.get(k)) for k in ("hp", "block", "energy", "powers") if pa.get(k) != pb.get(k)]


def same_board(a, b, view=False):
    """Two hosts show the same board: the visible state, potions, relic counters and the cards
    of every pile (the draw order is not compared: the search never reads it). view: the visible
    state as _view shows it (with the powers gained in the fight; the live one misses them, so a
    host that gained one never matched)."""
    try:
        state = _view if view else (lambda rm: rm.combat_state().get_state())
        return (state(a) == state(b) and a.potions() == b.potions()
                and a.relic_counters() == b.relic_counters() and _piles(a) == _piles(b))
    except Exception:
        return False


class RsFighter:
    def __init__(self, sims=1000, boss_sims=4000, boss3_sims=None, seed=0, elite_sims=None, reuse=False,
                 keep_host=False, boss1_pow=None, true_view=False, dump_wires=False, carry=False, carry_rooms=None):
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
        self.true_view = true_view   # keep: match the host's board with the powers it gained (_view)
        # dump_wires: each elite / boss fight's first combat_wire (and its context) in self.wire0, for
        # the fight log (transplant.py replays the fight from it inside sts2-rs)
        self.dump_wires, self.wire0 = dump_wires, None
        # carry: a host rebuilt from the wire takes what the wire cannot say from the last host that
        # searched (CombatState.transplant_hidden, as sts2-rs's own CombatMirror): each enemy's move
        # machine (state log, counter, first-move flags), undisclosed relic latches, the turn's hidden
        # tallies, the draw pins. Without it every rebuild (after nearly every enemy turn) restarts the
        # bosses' move machines blank. The RNG cursors stay the world's (the wire carries them)
        self.carry, self.prev = carry, None
        # carry_rooms: carry only in these room types (e.g. {"Boss"}); None: every fight
        self.carry_rooms = set(carry_rooms) if carry_rooms else None
        if reuse:
            want_override("tree_reuse=1")
        # boss1_pow k: the first boss of A10's double boss values a win at 0.5 + (the share of a full
        # HP bar and belt it keeps)^k, as the second boss's win rate climbs steeply with its entry HP
        # (sts2-rs win_carry_pow; the deployed value prices 1 HP at ~1/300 of the win)
        self.boss1_pow = boss1_pow
        self.stats = Counter()
        self.fails = []       # the first failures of the run (kind, detail), for the run record
        self.k = 0            # decisions searched (the agent seed mixes it in)
        self.host = None      # host RunManager after the last searched play (answers its prompt)
        # fidelity diagnostic, per fight (play_combat resets it): after a move that is not end_turn
        # (no enemy turn in between), does the host that made it show the true board? A port or
        # import mismatch shows as "mid_diff" on enemy fields; a draw the host guessed, on the piles
        self.fight, self.diffs, self.cont_move = Counter(), [], None
        # and after end_turn: did the host's own enemy turn leave the true board (turn_diff)? The
        # search plays every later turn on sts2-rs's enemies, so a boss it models wrong shows here
        self.tdiffs = []

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
        if self.boss1_pow and is_first_double_boss(ctx):
            # this agent only: the env bundle is shared by every worker thread of the process
            cur = os.environ.get("STS2_RUNAI_MCTS_OVERRIDES", "").strip()
            agent.set_mcts_overrides(",".join(x for x in (cur, f"win_carry_pow={self.boss1_pow}") if x))
            self.stats["boss1_pow"] += 1
        if self.reuse:
            self.agent = agent
        return agent

    def _step(self, st, rm):
        rec = self._agent(st).step(rm)
        if rec is not None and rec.get("retained_visits") is not None:
            self.stats["reused"] += 1
            self.stats["retained_visits"] += int(rec["retained_visits"])
        return rec

    def _carry(self, cs, carried, imported):
        """transplant_hidden from the carried host's board onto the fresh import cs. Each enemy takes
        the closest unused carried enemy of its kind (dead or alive, HP, powers): a corpse one side
        keeps and the other reaps shifts the slots, and same-kind enemies change places (a
        reattached Decimillipede segment comes back elsewhere in the host than in the game, so
        matching in order swapped their move machines). Relic counters carry for the relics the
        wire left at None."""
        def far(o, e):
            pw = lambda x: sorted((p.get("name"), p.get("amount")) for p in x.get("powers") or [])
            return (1000 * (bool(o.get("is_dead")) != bool(e.get("is_dead"))) + abs((o.get("hp") or 0) - (e.get("hp") or 0))
                    + 10 * (pw(o) != pw(e)))
        try:
            new = sts2_rs.state_to_dict(cs).get("encounter") or []
            old = sts2_rs.state_to_dict(carried).get("encounter") or []
            used, slots = set(), []
            for e in new:
                cand = [j for j, o in enumerate(old) if j not in used and o.get("kind") == e.get("kind")]
                j = min(cand, key=lambda j: (far(old[j], e), j)) if cand else None
                if j is not None:
                    used.add(j)
                slots.append(j)
            hidden = [r.get("id") for r in (imported.get("player") or {}).get("relics") or [] if r.get("counter") is None]
            cs.transplant_hidden(carried, enemy_slots=slots, carry_rng=False, carry_relic_counters=hidden)
            self.stats["carried"] += 1
        except Exception as e:  # noqa: BLE001  (the fresh import still stands)
            self.stats["carry_fail"] += 1
            self.stats["carry_fail:" + type(e).__name__] += 1
            if self.stats["carry_fail"] <= 3:
                self._fail("carry", f"{type(e).__name__}: {str(e)[:200]}")

    def _host(self, st, wire, carried=None):
        ctx = st.get("context") or {}
        asc = int(ctx.get("ascension") or 0)
        wire = {**wire, "game_version": GAME_VERSION}
        # character: the adapter otherwise infers the class from card ids, which fails for a
        # deck with no Ironclad-prefixed card left
        imported = adapter.translate(wire, character="Ironclad", ascension=asc)
        cs = sts2_rs.state_from_dict(imported)
        if carried is not None:
            self._carry(cs, carried, imported)
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
            ctx = st.get("context") or {}
            if self.dump_wires and self.wire0 is None and ctx.get("room_type") in ("Elite", "Boss"):
                self.wire0 = {"ctx": {k: ctx.get(k) for k in ("act", "floor", "room_type", "ascension")},
                              "last_boss": is_last_boss(ctx), "wire": wire}
            cont, self.cont = self.cont, None
            src = cont if cont is not None else self.prev
            try:
                carry = self.carry and src is not None and (self.carry_rooms is None or ctx.get("room_type") in self.carry_rooms)
                rm = self._host(st, wire, src.combat_state() if carry else None)
            except (UnsupportedStateError, ValueError, KeyError, TypeError, IndexError) as e:
                self.stats["import_fail"] += 1
                self.stats["import_fail:" + type(e).__name__] += 1
                self._fail("import", f"{type(e).__name__}: {str(e)[:200]}")
                return None
            if cont is not None:
                # the host that made the last move carries what the wire lacks (cards played, HP
                # lost, hits this turn...), which tree reuse's node match reads: go on with it
                # while it shows the board we have (a chance edge, e.g. the enemy turn, mostly differs)
                mid = self.cont_move is not None and self.cont_move[0] != "end_turn"
                if self.cont_move is not None and not mid:
                    td = turn_diff(cont, rm)
                    self.fight["turn_diff" if td else "turn_same"] += 1
                    for p, _, _ in td:
                        self.fight["td:" + p] += 1
                    if td and len(self.tdiffs) < 8:
                        self.tdiffs.append({"round": st.get("round"),
                                            "diff": [[p, str(a)[:60], str(b)[:60]] for p, a, b in td[:6]]})
                if same_board(cont, rm, self.true_view):
                    if mid:
                        self.fight["mid_same"] += 1
                    rm = cont
                    self.stats["host_kept"] += 1
                else:
                    if mid:
                        diff = board_diff(cont, rm)
                        if not diff:
                            self.fight["mid_repr"] += 1   # only power order / dead enemies listed
                        else:
                            self.fight["mid_diff"] += 1
                            if len(self.diffs) < 3:
                                self.diffs.append({"move": list(self.cont_move), "diff": diff})
                    self.stats["host_rebuilt"] += 1
            rec = self._step(st, rm)
            if rec is None or not rec.get("is_combat"):
                self.stats["no_action"] += 1
                return None
            act = self._ours(st, rec["action"])
            if act is not None:
                self.stats["searched"] += 1
                self.host = self.prev = rm   # took the same action on the true board: it answers the next prompt
                if self.keep:
                    self.cont = rm
                    hand = st.get("hand") or []
                    i = rec["action"].get("hand_idx")
                    self.cont_move = (rec["action"].get("action"),
                                      hand[i].get("id") if i is not None and 0 <= i < len(hand) else None)
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
        self.agent = self.cont = self.prev = None   # a fight's own agent and host (reuse, carry)
        self.fight, self.diffs, self.cont_move, self.wire0, self.tdiffs = Counter(), [], None, None, []
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
        self.agent = self.cont = self.prev = None
        return state
