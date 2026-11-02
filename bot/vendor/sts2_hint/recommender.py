"""MCTS hint recommender.

Bead: ``sts2-rs-tjtr`` (hint-T8, sts2-pycli side).

Pipeline:

1. Take a Sts2Headless ``combat_play`` state dict (the same dict that
   flows through :func:`play.show_combat`).
2. Run :func:`sts2_hint.adapter.translate` to produce the wire-schema
   dict.
3. Call :func:`sts2_rs.state_from_dict` + :func:`sts2_rs.mcts_recommend`.
4. Map each sts2-rs action back to the C# ``stable_idx`` via the hand
   ordering preserved in the wire dict.
5. Return ``{stable_idx: HintTag}``.

The recommender is intentionally state-free; one call per decision
point. ``sts2_client.Game`` is also accepted via duck-typing so the
caller may pass it directly if a future refactor stores the latest
state on the Game instance (it does not at the time of writing).

Sentinel ``stable_idx`` values for non-card actions:

* ``-1``    — end-turn
* ``-100 - slot`` — ``potion`` / ``drink_potion`` (slot ∈ [0, 4])
* ``-200``  — channel
* ``-201``  — evoke
* ``-300 - index`` — pending-choice ``choose`` selection

Card play maps to the matching ``CardSpec.stable_idx`` already emitted
by :mod:`sts2_hint.adapter` (C# ``card.index``).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any, NamedTuple

# Sentinel stable_idx values for non-card actions. Kept as module
# constants so callers (display, tests) can refer to them by name.
END_TURN_STABLE_IDX = -1
POTION_STABLE_IDX_BASE = -100  # potion slot s -> -100 - s
CHANNEL_STABLE_IDX = -200
EVOKE_STABLE_IDX = -201
CHOOSE_STABLE_IDX_BASE = -300  # choose option i -> -300 - i
#: The pending-choice FINALIZE/SKIP action (`{"action": "choose",
#: "selection": []}`). The engine's wire dict deliberately collapses its
#: back-compat `index` to 0 (crates/sts2-py/src/lib.rs, `legal_actions`:
#: "the full selection list disambiguates the empty-selection SKIP from
#: picking option 0"), so mapping choose actions by `index` alone encoded
#: the skip as a phantom "choose option 0". On Gambling Chip's turn-1
#: mulligan that phantom became a DUPLICATE index in the world's
#: `select_cards`, and the C# `DiscardAndDraw(list, list.Count)`
#: (GamblingChip.cs:20-23) then drew one MORE card than it discarded —
#: the whole mirror_d1 "-1 hand" family (seeds 1004/1047).
CHOOSE_SKIP_STABLE_IDX = -299


class HintTag(NamedTuple):
    """One action's hint summary.

    Attributes:
        stable_idx: C# ``card.index`` for card-plays; sentinel for
            non-card actions (see module docstring).
        win_prob: ``p_win`` from MCTS, in ``[0, 1]``.
        bucket_label: Label from ``sts2_rs.WIN_PROB_BUCKETS``
            (e.g. ``"likely-win"``).
        is_best: True iff this is the search's own pick (``mcts_recommend``
            ``is_best``: the engine's root rule, GH #1022), or, when a caller
            dropped that row, the engine's next choice among the survivors
            (:func:`crown_engine_heir`). Exactly one entry per ``recommend``
            call has this set.
        visits: Raw visit count — useful for debug overlay; not
            displayed in the default tag format.
        q_saturated: True when the rollout policy couldn't distinguish
            actions (Q-spread below epsilon). When true, the displayed
            tag should be treated as advisory only.
        low_confidence: True when this action's visit count is well
            below the most-visited action's. Independent of
            ``q_saturated``.
        target_index: Enemy index for actions with target selection
            (cards / potions with ``AnyEnemy`` target). ``None`` for
            untargeted actions. The MCTS action space lists one entry
            per (action, target) pair, so multiple HintTags can share
            a ``stable_idx`` while differing only on this field.
        ally_index: Co-op: the teammate (``allies[]`` index) an ``AnyAlly``
            card is aimed at (``PlayCardAtAlly``, GH #1017). ``None`` for every
            other action, including an ``AnyAlly`` card with one teammate.
        pick_rank: The engine's preference order (``mcts_recommend`` row
            ``pick_rank``, GH #1022): 0 is its pick, and a caller that has to
            drop rows crowns the lowest surviving rank. ``None`` when the
            result carries none.
        in_band: The engine cannot tell this action from its pick (inside
            the ``tie_eps`` band its rule decided in).
    """

    stable_idx: int
    win_prob: float
    bucket_label: str
    is_best: bool
    visits: int = 0
    q_saturated: bool = False
    low_confidence: bool = False
    target_index: int | None = None
    # Continuous Q (value_sum / visits) from the lightspeed-style reward.
    # Bead `sts2-rs-v2oh`: this is what action ranking actually consumes.
    # Range roughly [-1.0, 1.0]; > 0.5 reliably means win, < -0.5 loss.
    q: float = 0.0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    ally_index: int | None = None
    pick_rank: int | None = None
    in_band: bool = False


def belief_draw_c_puct(sims: int, threads: int = 1) -> float:
    """Exploration constant for a BeliefDraw search of ``sims`` simulations.

    Mirror of ``sts2_runai::policy::belief_draw_c_puct`` — keep the two in step.

    The optimum ``c_puct`` rises with budget. Measured on the 168-position book
    at eight budgets, 24 seeds each, paired against the deployed 2.5
    (``logs/2026-08-15-cpuct-sims/FINDINGS.md``): 2.5 loses 8.2pp at 200 sims
    (p<1e-4), 4.2pp at 300 (p=0.007) and 3.1pp at 500 (p=0.0003), and is not
    significantly beaten anywhere from 1000 up.

    The rule only moves off 2.5 where that evidence is significant, so **every
    deployed budget is unchanged**. What it fixes is the hazard: ``c_puct`` was
    pinned in ``HONEST_SEARCH`` alongside ``hidden_info`` while ``--sims`` is
    chosen separately, so trimming the budget for latency silently kept 2.5 and
    walked into the mis-tuned corner.

    ``threads`` matters because root-parallel search SPLITS the budget across
    workers (``mcts_recommend``: "the sim budget is SPLIT across workers") and
    ``c_puct`` governs one tree, so the per-tree budget is what the sweep
    measured. NOTE this part is an extrapolation — the sweep itself ran
    single-tree. The advisor's defaults (4000 sims, 16000 at bosses, one
    tree) land on 2.5.
    """
    per_tree = sims // max(1, threads)
    if per_tree < 500:
        return 0.8
    if per_tree < 1000:
        return 1.41
    return 2.5


@dataclass(frozen=True)
class RecommendConfig:
    """Knobs for one :func:`recommend` call.

    Defaults: sims=300, depth=200, c_puct=1.4. ``threads=1`` and
    ``fair=False`` are the cheap path; flip to ``fair=True`` when you need
    unbiased per-action win-prob.

    ``depth`` is the rollout ply cap. It must be deep enough for a rollout to
    reach a real terminal (combat won/lost); a too-shallow cap (the old
    anchor was 10) truncates every Act-1 rollout before the fight ends, so
    the leaf-eval scores every action ~0 — a flat, useless win-prob where
    ``is_best`` collapses to "end turn". 200 matches ``sts2_rs.mcts_recommend``'s
    own default and lets most rollouts terminate.
    """

    sims: int = 300
    depth: int = 200
    #: Exploration constant. ``None`` (the default) means "whatever the
    #: resolved ``hidden_info`` mode deploys" — see :meth:`resolved_c_puct`.
    #: Set a number only to override that deliberately (an ablation, or a
    #: recorder pinning its historical value).
    c_puct: float | None = None
    seed: int = 0
    threads: int = 1
    fair: bool = False
    cheat_draw_order: bool = False  # reserved for future use
    timeout_ms: int = 0  # 0 = no timeout (not yet enforced)
    # Optional explicit character override forwarded to the adapter.
    character: str | None = None
    # Optional ascension passed through to adapter / Rust state.
    ascension: int = 0
    # Forwarded as the ``combat_id`` argument to the adapter.
    combat_id: int = 0
    # Rollout policy forwarded to sts2_rs.mcts_recommend. None (default) runs
    # what the hidden-info mode deploys: the playout net under "beliefdraw"
    # (the fight character's built-in net; all five characters ship one),
    # the heuristic otherwise. A named policy — "heuristic" (softmax over
    # expert priorities), "uniform" (un-biased random) or "net" — always runs.
    rollout_policy: str | None = None
    # Hidden-information mode forwarded to sts2_rs.mcts_recommend. Default
    # "beliefdraw" — the DEPLOYED honest search, the same mode
    # ``sts2_runai::policy::resolve_hidden_info`` picks when
    # ``STS2_RUNAI_HONEST`` is unset, i.e. what every agent that plays a run
    # actually uses.
    #
    # It used to default to "clairvoyant", the historical leaky search whose
    # root sees the true draw order / RNG cursors. That made every consumer of
    # this config — the workbench hint panel, the regret review, the real-game
    # advisor — judge play with information the player and the agent never
    # had, so an honest move routinely scored as a blunder against a line only
    # a cheater could pick. Two further reasons it is wrong as a default: on
    # reconstructed states (real-DLL bridge, dict round-trip) the RNG cursors
    # are faked, so a clairvoyant root plans against a fictional future; and
    # the G2 3-way A/B retired clairvoyance as the deployed mode outright
    # (docs/plans/2026-07-12-clairvoyance-removal-plan.md).
    #
    # Options: beliefdraw|clairvoyant|drawchance|rootscramble. "clairvoyant"
    # remains selectable for ablations — it is no longer anybody's default.
    hidden_info: str = "beliefdraw"
    # Draw-node progressive-widening constant. None means "the value the
    # resolved mode deploys": 2.0 under BeliefDraw (the F2 retune), else the
    # engine default. See :meth:`resolved_c_puct` for the c_puct half.
    draw_widening_c: float | None = None
    # EndTurn-node progressive-widening pair. None (the default) means "what
    # the resolved mode deploys", resolved in Rust by
    # ``MctsConfig::apply_mode_dialect`` — under BeliefDraw that is the PR
    # #467 retune (5.787 / 0.436) the run agent plays with (GH #758). Set
    # numbers only for an ablation; 0.5 / 0.4 is the pre-#467 dialect.
    endturn_widening_c: float | None = None
    endturn_widening_alpha: float | None = None
    # Common random numbers: give every stochastic edge off one decision node
    # the SAME reseed stream, so sibling alternatives are compared under
    # matched luck instead of each rolling its own draw. Off by default — it
    # changes every seeded fixture, and its matched-luck argument is weaker
    # here than in the lightspeed original (after different card plays the
    # deck composition itself differs, so a shared reseed is an
    # approximation). Judge it on per-decision STABILITY (flip rate, modal
    # share), not on a 60-seed win rate that noise dominates.
    common_random_numbers: bool = False
    # Engine-proven move-order pruning (``MctsConfig::dominance_pruning``):
    # drop a follow-up card play when playing it earlier in the same turn is
    # proven at least as good. Off here, matching the run agent; the browser
    # workbench turns it on (``DEFAULT_MCTS_CONFIG``).
    dominance_pruning: bool = False
    # Boss win value by what the fight carries forward (GH #771): the run
    # agent's `boss_win_carry`, applied by default to the run's last boss
    # only, where winning is all that counts. `None` keeps the default win
    # value; the live advisor fills these per fight (`sts2_rs.boss_win_carry`).
    win_hp_carry: float | None = None
    win_potion_carry: float | None = None
    # Optional path to a JSONL log file. When set, every recommend()
    # call appends one line with the full per-action MCTS output plus
    # a state digest, for post-hoc analysis.
    log_path: str | None = None

    def resolved_c_puct(self) -> float:
        """The exploration constant this config actually searches with.

        ``c_puct=None`` defers to the mode: under the deployed BeliefDraw
        honest search that is :func:`belief_draw_c_puct` of the budget (the F2
        retune, which steps down below 500 sims/tree), and under any other mode
        it is the engine default 1.4. An explicit number always wins.

        Deferring matters because ``sims`` is chosen separately from the
        constant — a caller that trimmed the budget for latency used to keep a
        c_puct tuned for a budget it no longer runs at.
        """
        if self.c_puct is not None:
            return self.c_puct
        if self.hidden_info == "beliefdraw":
            return belief_draw_c_puct(self.sims, self.threads)
        return 1.4

    def resolved_draw_widening_c(self) -> float | None:
        """The draw-node widening constant this config searches with.

        ``None`` defers to the mode: 2.0 under BeliefDraw (the F2 retune, which
        travels with the c_puct above), else ``None`` = the engine default.
        That is the base dialect on purpose: the run agent's Ironclad layer
        (draw widening 4.0 among others, GH #830) is not taken by the hint.
        """
        if self.draw_widening_c is not None:
            return self.draw_widening_c
        return 2.0 if self.hidden_info == "beliefdraw" else None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RecommendConfig:
        """Build from a (possibly over-specified) config dict, dropping
        unknown keys.

        Tolerates stale keys carried by older frontends / saved reasoning
        configs — notably ``chance_nodes``, removed when the engine retired
        the legacy single-determinization PIMC search (chance-node MCTS is now
        the only mode), so a persisted config that still stamps it no longer
        raises ``TypeError`` here.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def _resolve_state(game: Any) -> dict[str, Any]:
    """Best-effort accessor.

    The bead spec says "takes the live ``sts2_client.Game``" but the
    current Game class does not store its latest combat-play state.
    To stay flexible:

    * If ``game`` is already a ``decision="combat_play"`` dict, use it.
    * Else if it has a ``.state`` / ``.last_state`` / ``.combat_state``
      attribute that looks like a combat_play dict, use that.
    * Else raise :class:`TypeError`.
    """
    if isinstance(game, dict) and game.get("decision") == "combat_play":
        return game
    for attr in ("state", "last_state", "combat_state"):
        v = getattr(game, attr, None)
        if isinstance(v, dict) and v.get("decision") == "combat_play":
            return v
    raise TypeError(
        "recommend(): expected a Sts2Headless combat_play state dict or "
        "an object exposing one via .state/.last_state/.combat_state; "
        f"got {type(game).__name__}"
    )


def _state_digest(raw_state: dict[str, Any]) -> dict[str, Any]:
    """Tiny state summary suitable for an MCTS log line.

    Captures only the fields that make the line readable when scanning
    a JSONL file by eye — turn, player HP, energy, hand sizes, enemy
    HP and **intents list with per-hit damage** (the load-bearing
    field for diagnosing rollout behavior). Not a full state snapshot.

    Schema notes — the Sts2Headless wire dict (see
    ``RunSimulator.cs``) puts hand/pile state at the top level, not
    under a ``deck`` sub-object, and enemy intents are under the
    plural key ``intents`` as a list of
    ``{type, damage?, hits?, label, ...}`` records.
    """
    player = raw_state.get("player") or {}
    enemies = raw_state.get("enemies") or []
    return {
        "decision": raw_state.get("decision"),
        "round": raw_state.get("round"),
        "energy": raw_state.get("energy"),
        "player": {
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "block": player.get("block"),
            "potions": player.get("potions"),
            "powers": player.get("powers"),
        },
        "hand_size": len(raw_state.get("hand") or []),
        "draw_size": raw_state.get("draw_pile_count") or len(raw_state.get("draw_pile") or []),
        "discard_size": raw_state.get("discard_pile_count")
        or len(raw_state.get("discard_pile") or []),
        "enemies": [
            {
                "idx": e.get("idx") if e.get("idx") is not None else e.get("index"),
                "name": e.get("name"),
                "hp": e.get("hp") if e.get("hp") is not None else e.get("current_hp"),
                "max_hp": e.get("max_hp"),
                "block": e.get("block"),
                "intents": [
                    {
                        "type": it.get("type"),
                        "damage": it.get("damage"),
                        "hits": it.get("hits"),
                        "label": it.get("label"),
                    }
                    for it in (e.get("intents") or [])
                ],
                "powers": e.get("powers"),
            }
            for e in enemies
        ],
    }


def _append_log(
    log_path: str,
    cfg: RecommendConfig,
    raw_state: dict[str, Any],
    result: dict[str, Any],
    hand: list[dict[str, Any]],
) -> None:
    """Append one JSONL record describing this recommend() call.

    Best-effort: any I/O or serialization failure prints a one-line
    warning to stderr and otherwise is swallowed (logging must never
    take down the hint pipeline).
    """
    import json
    import os
    import sys
    import time

    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
        record = {
            "ts": time.time(),
            "cfg": {
                "sims": cfg.sims,
                "depth": cfg.depth,
                "c_puct": cfg.resolved_c_puct(),
                "seed": cfg.seed,
                "threads": cfg.threads,
                "fair": cfg.fair,
                "rollout_policy": cfg.rollout_policy,
                "ascension": cfg.ascension,
                "combat_id": cfg.combat_id,
            },
            "state": _state_digest(raw_state),
            "hand": [
                {
                    "stable_idx": h.get("stable_idx"),
                    "name": h.get("name") or h.get("card_id"),
                    "cost": h.get("cost"),
                    "upgraded": h.get("upgraded"),
                }
                for h in hand
            ],
            "actions": result.get("actions", []),
            "q_saturated": result.get("q_saturated"),
            "pv": result.get("pv"),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        print(f"[hint] mcts log write failed: {e}", file=sys.stderr)


def _bucket_label(p_win: float, buckets: list[tuple[float, str]]) -> str:
    """Return the bucket label for ``p_win``.

    ``buckets`` is the value of ``sts2_rs.WIN_PROB_BUCKETS`` — a list
    of ``(low_inclusive, label)`` pairs sorted **descending** by
    threshold (see ``crates/sts2-mcts/src/hint.rs``). The first row
    with ``low <= p_win`` wins.
    """
    for low, label in buckets:
        if p_win >= low:
            return label
    # Defensive fallback — shouldn't trigger because the last bucket
    # in WIN_PROB_BUCKETS has low=0.0.
    return buckets[-1][1] if buckets else "unknown"


class UnmappableAction(KeyError):
    """A ranked action that has no handle in THIS world's hand. GH #360.

    Distinct from the plain :class:`KeyError` the unknown-action-kind arm
    raises, because the two want opposite handling. An unknown kind is
    sts2-rs/adapter drift — a contract break the caller must not paper over. A
    play whose ``card_index`` runs past the world's hand is a *recoverable
    runtime condition*: the searched mirror is holding a card the world is not,
    so that ONE row is meaningless while every other row in the ranking is
    still perfectly good.

    Measured on the 2026-08-03 live corpus (88 logs, 38 487 decisions): 151
    such rows, 150 of them exactly one card past the end, and **151 of 151** on
    the decision immediately following a ``FromChooseACardScreen`` choose —
    where ``CombatMirror.reconcile`` deliberately defers grading for one
    decision (``choice_in_flight_skips``) and the search therefore runs on a
    hand the world has not confirmed. Aborting the whole ranking turned each
    into a full collapse to the uniform random-legal policy; dropping the row
    keeps the best MAPPABLE MCTS action instead.
    """


def _stable_idx_for_action(action: dict[str, Any], hand: list[dict[str, Any]]) -> int:
    """Map one sts2-rs legal-action dict to its C# stable_idx.

    Sentinels documented in the module docstring. Raises
    :class:`UnmappableAction` when a play indexes past this world's hand —
    recoverable, and dropped by :func:`tags_from_result`. Raises a plain
    :class:`KeyError` for shapes that aren't in the wire contract at all; the
    caller should not catch that one (it indicates sts2-rs/adapter drift).
    """
    kind = action["action"]
    if kind == "play":
        card_index = int(action["card_index"])
        if not 0 <= card_index < len(hand):
            raise UnmappableAction(
                f"recommend(): play action card_index={card_index} out of range "
                f"for hand of size {len(hand)}"
            )
        return int(hand[card_index]["stable_idx"])
    if kind == "end_turn":
        return END_TURN_STABLE_IDX
    if kind in ("potion", "drink_potion"):
        return POTION_STABLE_IDX_BASE - int(action["slot"])
    if kind == "channel":
        return CHANNEL_STABLE_IDX
    if kind == "evoke":
        return EVOKE_STABLE_IDX
    if kind == "choose":
        # `selection` is authoritative: the engine emits it precisely so the
        # empty-selection FINALIZE/SKIP is distinguishable from "choose option
        # 0" — both collapse to `index == 0` (crates/sts2-py/src/lib.rs,
        # `legal_actions` choose arm). Reading only `index` turned every
        # ranked skip into a phantom pick of option 0.
        selection = action.get("selection")
        if selection is not None:
            if len(selection) == 0:
                return CHOOSE_SKIP_STABLE_IDX
            return CHOOSE_STABLE_IDX_BASE - int(selection[0])
        return CHOOSE_STABLE_IDX_BASE - int(action.get("index", 0))
    raise KeyError(f"recommend(): unknown sts2-rs action kind {kind!r}")


def tags_from_result(result: dict[str, Any], hand: list[dict[str, Any]]) -> list[HintTag]:
    """Turn a raw ``sts2_rs.mcts_recommend`` result into ranked :class:`HintTag`s.

    Split out of :func:`recommend_all` so a caller that already owns the
    searched state — the persistent combat mirror, which searches a clone of
    itself rather than a fresh reconstruction — reuses the exact same
    action→``stable_idx`` mapping instead of a second copy that could drift.

    Args:
        result: The ``mcts_recommend`` dict.
        hand: The wire dict's ``deck.hand``, whose ``stable_idx`` values are
            the world's own card handles.

    Returns:
        MCTS-best first, then descending win probability. One entry per
        (action, target) pair — rows sharing a ``stable_idx`` are NOT
        collapsed. Rows the world's hand cannot address are DROPPED (see
        :class:`UnmappableAction`); an empty result means every row was
        unmappable, which the caller still treats as "no legal actions".
    """
    import sts2_rs

    buckets: list[tuple[float, str]] = list(sts2_rs.WIN_PROB_BUCKETS)
    out: list[HintTag] = []
    dropped: list[str] = []
    for entry in result["actions"]:
        action = entry["action"]
        target = action.get("target_index")
        ally = action.get("ally_index")
        p_win = float(entry["p_win"])
        try:
            stable_idx = _stable_idx_for_action(action, hand)
        except UnmappableAction as e:
            # GH #360: one phantom card must not cost the whole ranking. Never
            # silent — a dropped row means the searched state disagreed with
            # the world, which is exactly what the corpus mining needs to see.
            dropped.append(str(e))
            continue
        out.append(
            HintTag(
                stable_idx=stable_idx,
                win_prob=p_win,
                bucket_label=_bucket_label(p_win, buckets),
                is_best=bool(entry["is_best"]),
                visits=int(entry["visits"]),
                q_saturated=bool(entry.get("q_saturated", False)),
                low_confidence=bool(entry.get("low_confidence", False)),
                target_index=int(target) if target is not None else None,
                q=float(entry.get("q", entry.get("value", 0.0))),
                wins=int(entry.get("wins", 0)),
                losses=int(entry.get("losses", 0)),
                ties=int(entry.get("ties", 0)),
                ally_index=int(ally) if ally is not None else None,
                pick_rank=int(entry["pick_rank"]) if entry.get("pick_rank") is not None else None,
                in_band=bool(entry.get("in_band", False)),
            )
        )
    if dropped:
        print(
            f"[hint] dropped {len(dropped)} unmappable action row(s) from the ranking, "
            f"kept {len(out)}: {dropped[0]}",
            file=sys.stderr,
        )
    out = crown_engine_heir(out)
    out.sort(key=lambda t: (not t.is_best, -t.win_prob))
    return out


def crown_engine_heir(tags: list[HintTag]) -> list[HintTag]:
    """Re-crown ``tags`` when the engine's pick is no longer among them.

    A caller that drops rows after the search -- an unmappable row here, a
    play the live game refuses in ``realgame_advisor.veto_unplayable`` --
    must not crown a move by a rule of its own. The heir is the engine's own
    next choice among the survivors: the lowest ``pick_rank`` (GH #1022),
    falling back to the first tag only when no tag carries a rank.

    Returns:
        ``tags`` unchanged when one is still ``is_best``; otherwise the heir
        first with ``is_best`` set, then the rest in their order.
    """
    if not tags or any(t.is_best for t in tags):
        return tags
    ranked = [t for t in tags if t.pick_rank is not None]
    heir = min(ranked, key=lambda t: t.pick_rank) if ranked else tags[0]
    return [heir._replace(is_best=True)] + [t for t in tags if t is not heir]


def choice_picks_from_tags(
    tags: list[HintTag],
    *,
    n_options: int,
    min_select: int,
    max_select: int,
) -> list[int]:
    """Option indices to answer a pending-choice screen with, best-first.

    Shared by BOTH mirrored ``card_select`` drivers
    (``headless_eval.pick_card_select_mirrored`` and
    ``realgame_mcts.pick_card_select_mirrored``) so the tag→picks reading
    cannot fork and drift. Three properties are load-bearing:

    * **The FINALIZE/SKIP tag is a stop signal, not a pick.** The engine's
      multi-pick root enumerates one incremental pick per option plus one
      empty-selection finalize (sts2-core ``legal_actions``); the search
      ranking the finalize above a remaining option means "stop here".
      The old reading encoded it as a phantom "choose option 0"
      (:data:`CHOOSE_SKIP_STABLE_IDX`), which both destroyed the stop
      signal (every mulligan discarded ``max_select`` cards) and sent a
      DUPLICATE index to the world — whose ``ResolvePendingByIndices``
      (RunSimulator.cs:7119) keeps duplicates, so Gambling Chip's
      ``DiscardAndDraw(list, list.Count)`` (GamblingChip.cs:20-23) drew one
      more card than it discarded while the engine's resolver
      (``resolve_discard_from_hand_and_draw``) skips the already-discarded
      duplicate. That asymmetry was the entire mirror_d1 "post-mulligan
      hand short by 1" family (seeds 1004/1047).
    * **No duplicates, ever.** A selection index names one concrete option;
      the world's screen consumes each at most once.
    * **Arity is honored.** At most ``max_select`` picks; a skip ranked
      before ``min_select`` picks were collected is ignored (stopping there
      is not legal), and a search that returned fewer than ``min_select``
      option rows is padded with the lowest unused indices (defensive —
      every option is a root action, so this should not happen).

    Args:
        tags: :func:`tags_from_result` output (best-first).
        n_options: How many options the world's screen shows.
        min_select / max_select: The screen's (already clamped) arity.

    Returns:
        Distinct option indices, best-first. Empty means skip.
    """
    picks: list[int] = []
    stop_when_legal = False
    for t in tags:
        if t.stable_idx == CHOOSE_SKIP_STABLE_IDX:
            if len(picks) >= min_select:
                break
            # Stopping is not legal yet: remember the wish and stop the
            # moment the minimum is satisfied.
            stop_when_legal = True
            continue
        if t.stable_idx > CHOOSE_STABLE_IDX_BASE:
            continue  # not a choose tag
        idx = CHOOSE_STABLE_IDX_BASE - t.stable_idx
        if not 0 <= idx < n_options or idx in picks:
            continue
        picks.append(idx)
        if len(picks) >= max_select or (stop_when_legal and len(picks) >= min_select):
            break
    if len(picks) < min_select:
        picks.extend(i for i in range(n_options) if i not in picks)
        del picks[min_select:]
    return picks


#: What :func:`next_choice_pick_from_tags` returns when the search ranked the
#: FINALIZE/SKIP row above every remaining option: "stop here".
FINALIZE = object()


def next_choice_pick_from_tags(
    tags: list[HintTag],
    *,
    n_options: int,
    taken: Sequence[int],
    allow_finalize: bool = True,
) -> int | object | None:
    """The single best next pick for an open choice — ONE incremental step.

    :func:`choice_picks_from_tags` reads a whole multi-pick answer off one
    search. This reads only the FIRST pick, so the caller can apply it and
    search again with that pick in the root state — which is how the engine
    itself models a multi-pick (`legal_actions` re-opens the same choice after
    every incremental `ResolveChoice`, bead ``sts2-rs-sopgf``). Second-best at
    a root where nothing is picked yet is not the same question as best-given-
    the-first-pick, and only the sequential read asks the second question.

    Args:
        tags: :func:`tags_from_result` output (best-first) for the CURRENT
            root — i.e. re-derived after each applied pick.
        n_options: How many options the world's screen shows.
        taken: Option indices already picked in this choice.
        allow_finalize: When False the FINALIZE row is skipped over instead of
            returned — the caller has not reached ``min_select`` yet, so
            stopping is not legal. Same rule as
            :func:`choice_picks_from_tags`'s "a skip ranked too early is
            ignored".

    Returns:
        The option index to pick next, :data:`FINALIZE` when the search wants
        to stop, or ``None`` when no choose row was usable at all.
    """
    for t in tags:
        if t.stable_idx == CHOOSE_SKIP_STABLE_IDX:
            if allow_finalize:
                return FINALIZE
            continue
        if t.stable_idx > CHOOSE_STABLE_IDX_BASE:
            continue  # not a choose tag
        idx = CHOOSE_STABLE_IDX_BASE - t.stable_idx
        if 0 <= idx < n_options and idx not in taken:
            return idx
    return None


def recommend_all(game: Any, cfg: RecommendConfig | None = None) -> list[HintTag]:
    """Return every legal action's :class:`HintTag`, one per (action, target).

    Unlike :func:`recommend`, this does NOT collapse rows that share a
    ``stable_idx`` but differ on ``target_index``. The result is sorted
    with the MCTS-best entry first, then by descending ``win_prob``.

    Use this when you want to surface per-target win probabilities
    (e.g. show a multi-enemy attack's score against each enemy
    separately). For the legacy "best-tag-per-stable_idx" dict, call
    :func:`recommend` instead.
    """
    cfg = cfg or RecommendConfig()
    raw_state = _resolve_state(game)

    import sts2_rs

    from sts2_hint import adapter

    wire = adapter.translate(
        raw_state,
        character=cfg.character,
        combat_id=cfg.combat_id,
        seed=cfg.seed,
        ascension=cfg.ascension,
    )
    hand = wire["deck"]["hand"]

    cs = sts2_rs.state_from_dict(wire)
    result = sts2_rs.mcts_recommend(
        cs,
        sims=cfg.sims,
        c_puct=cfg.resolved_c_puct(),
        depth=cfg.depth,
        seed=cfg.seed,
        threads=cfg.threads,
        fair=cfg.fair,
        rollout_policy=cfg.rollout_policy,
        hidden_info=cfg.hidden_info,
        draw_widening_c=cfg.resolved_draw_widening_c(),
        endturn_widening_c=cfg.endturn_widening_c,
        endturn_widening_alpha=cfg.endturn_widening_alpha,
        common_random_numbers=cfg.common_random_numbers,
        dominance_pruning=cfg.dominance_pruning,
    )

    if cfg.log_path:
        _append_log(cfg.log_path, cfg, raw_state, result, hand)

    return tags_from_result(result, hand)


def search_tree(game: Any, cfg: RecommendConfig | None = None) -> dict[str, Any]:
    """Run one MCTS search and return the full search tree for inspection.

    Same translation path as :func:`recommend`, but calls
    ``sts2_rs.mcts_search_tree`` (single non-fair tree) and returns its raw
    serialization: ``{"nodes": [...], "root": 0, "pv": [...], "sims": int,
    "c_puct": float}``. Each node carries ``id, parent, visits, q, value_sum,
    wins/losses/ties, is_terminal, proven_win, proven_loss, children[]`` and
    the incoming edge (``kind``/``label``/``target_index``).

    Args:
        game: A ``combat_play`` state dict or an object exposing one.
        cfg: Hint config (sims, depth, c_puct, seed, rollout_policy).

    Raises:
        Same as :func:`recommend` (adapter / ``state_from_dict`` errors).
    """
    cfg = cfg or RecommendConfig()
    raw_state = _resolve_state(game)

    import sts2_rs

    from sts2_hint import adapter

    wire = adapter.translate(
        raw_state,
        character=cfg.character,
        combat_id=cfg.combat_id,
        seed=cfg.seed,
        ascension=cfg.ascension,
    )
    cs = sts2_rs.state_from_dict(wire)
    return sts2_rs.mcts_search_tree(
        cs,
        sims=cfg.sims,
        c_puct=cfg.resolved_c_puct(),
        depth=cfg.depth,
        seed=cfg.seed,
        rollout_policy=cfg.rollout_policy,
        # Same reason :func:`recommend` forwards them: dropping these pinned
        # the tree to `hidden_info="clairvoyant"` regardless of the caller, so
        # the inspector showed a leaky search next to an honest hint panel.
        hidden_info=cfg.hidden_info,
        draw_widening_c=cfg.resolved_draw_widening_c(),
        endturn_widening_c=cfg.endturn_widening_c,
        endturn_widening_alpha=cfg.endturn_widening_alpha,
        dominance_pruning=cfg.dominance_pruning,
    )


def recommend(game: Any, cfg: RecommendConfig | None = None) -> dict[int, HintTag]:
    """Return ``{stable_idx: HintTag}`` for the current decision point.

    See module docstring for the sentinel ``stable_idx`` table. The
    output dict has exactly one ``HintTag`` whose ``is_best`` is True
    (the MCTS top pick).

    Args:
        game: A Sts2Headless ``combat_play`` state dict or an object
            exposing one. See :func:`_resolve_state`.
        cfg: Hint config (sims, depth, …). Uses :class:`RecommendConfig`
            defaults when None.

    Returns:
        ``dict[int, HintTag]``. Empty when the state has no legal
        actions (terminal).

    Raises:
        sts2_hint.errors.UnsupportedStateError / ValueError: From the
            adapter or ``sts2_rs.state_from_dict`` (unmapped fields).
        TypeError: When ``game`` is neither a dict nor exposes a
            recognized state accessor.
    """
    cfg = cfg or RecommendConfig()
    raw_state = _resolve_state(game)

    # Local imports keep import-time graph clean for callers that just
    # want to introspect the module (e.g. tests that monkeypatch).
    import sts2_rs

    from sts2_hint import adapter

    wire = adapter.translate(
        raw_state,
        character=cfg.character,
        combat_id=cfg.combat_id,
        seed=cfg.seed,
        ascension=cfg.ascension,
    )
    hand = wire["deck"]["hand"]

    cs = sts2_rs.state_from_dict(wire)
    result = sts2_rs.mcts_recommend(
        cs,
        sims=cfg.sims,
        c_puct=cfg.resolved_c_puct(),
        depth=cfg.depth,
        seed=cfg.seed,
        threads=cfg.threads,
        fair=cfg.fair,
        rollout_policy=cfg.rollout_policy,
        hidden_info=cfg.hidden_info,
        draw_widening_c=cfg.resolved_draw_widening_c(),
        endturn_widening_c=cfg.endturn_widening_c,
        endturn_widening_alpha=cfg.endturn_widening_alpha,
        common_random_numbers=cfg.common_random_numbers,
        dominance_pruning=cfg.dominance_pruning,
    )

    if cfg.log_path:
        _append_log(cfg.log_path, cfg, raw_state, result, hand)

    buckets: list[tuple[float, str]] = list(sts2_rs.WIN_PROB_BUCKETS)
    out: dict[int, HintTag] = {}
    for entry in result["actions"]:
        action = entry["action"]
        stable_idx = _stable_idx_for_action(action, hand)
        p_win = float(entry["p_win"])
        tag = HintTag(
            stable_idx=stable_idx,
            win_prob=p_win,
            bucket_label=_bucket_label(p_win, buckets),
            is_best=bool(entry["is_best"]),
            visits=int(entry["visits"]),
            q_saturated=bool(entry.get("q_saturated", False)),
            low_confidence=bool(entry.get("low_confidence", False)),
            q=float(entry.get("q", entry.get("value", 0.0))),
            wins=int(entry.get("wins", 0)),
            losses=int(entry.get("losses", 0)),
            ties=int(entry.get("ties", 0)),
        )
        # Multiple sts2-rs actions can map to the same stable_idx
        # (e.g. the same card with different targets when target
        # selection is open). Keep whichever entry has is_best=True;
        # otherwise prefer the higher win_prob — that's the row the
        # display layer will use to summarize the card.
        prior = out.get(stable_idx)
        if prior is None:
            out[stable_idx] = tag
            continue
        if (tag.is_best and not prior.is_best) or (
            (not prior.is_best) and tag.win_prob > prior.win_prob
        ):
            out[stable_idx] = tag

    return out


__all__ = [
    "CHANNEL_STABLE_IDX",
    "CHOOSE_SKIP_STABLE_IDX",
    "CHOOSE_STABLE_IDX_BASE",
    "END_TURN_STABLE_IDX",
    "EVOKE_STABLE_IDX",
    "POTION_STABLE_IDX_BASE",
    "HintTag",
    "RecommendConfig",
    "choice_picks_from_tags",
    "recommend",
    "recommend_all",
    "tags_from_result",
]
