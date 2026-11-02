"""Hint annotate entry point.

Bead: ``sts2-rs-7al4`` (hint-T6).

Lives in a private submodule so that :mod:`sts2_hint.__init__` can stay
docstrings + re-exports only (per the project's ruff RUF067 rule). The
public name is :func:`sts2_hint.annotate`.

This is a **T6 skeleton**. The real implementation will land in:

* T7 — adapter: read Sts2Headless state, emit the wire dict, call
  ``sts2_rs.state_from_dict`` + ``sts2_rs.mcts_recommend``.
* T8 — recommender + display: bucketise win-prob, format tags.

Calling :func:`annotate` here raises :class:`NotImplementedError` so
callers don't silently get an empty map and assume the hint pipeline
ran.
"""

from __future__ import annotations

from typing import Any

from sts2_hint.errors import get_logger


def annotate(game: Any, cfg: Any) -> dict[int, str]:
    """Return a ``{stable_idx: hint_tag}`` map for the current decision
    point in ``game``.

    Args:
        game: A :class:`sts2_client.Game` instance at a player-decision
            point (``phase == "PlayerPlayPhase"``).
        cfg: A hint-config namespace carrying ``mcts_sims``,
            ``mcts_depth``, ``mcts_c_puct``, ``mcts_seed``,
            ``mcts_threads``, ``mcts_fair``, ``hint_timeout_ms``, …
            (parsed by T10's CLI-flags bead).

    Returns:
        Mapping from ``CardSpec.stable_idx`` (or sentinel ``-1`` for
        end-turn) to a rendered tag string.

    Raises:
        UnsupportedStateError: If the snapshot references a relic /
            card / power / monster / mid-resolution state that
            ``sts2_rs`` cannot model. The exception message carries
            the offending JSON path.
        NotImplementedError: T6 skeleton — adapter/recommender land in
            T7+.
    """
    get_logger().debug("annotate() called on T6 skeleton; raising NotImplementedError")
    raise NotImplementedError(
        "sts2_hint.annotate is a T6 skeleton; adapter + recommender land "
        "in beads T7+ (see ../sts2-rs/docs/plans/2026-05-14-integrate-"
        "sts2-rs-mcts-hint-system-into-projects-sts2-sts2-c.md)."
    )
