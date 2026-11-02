"""Hint-tag formatting.

Bead: ``sts2-rs-tjtr`` (hint-T8, sts2-pycli side).

Renders a :class:`sts2_hint.recommender.HintTag` as a fixed-width
string suitable for interleaving with un-tagged rows in
:func:`play.show_combat`. The width is chosen so column alignment
survives across actions with and without a tag.

Format:
    ``[<LABEL> <pct>%] <best-arrow>``

Examples:
    ``[LIKELY-WIN 72%] ▲``   — best action, high win-prob
    ``[EVEN 47%]     ``      — non-best, even bucket
    ``[LIKELY-LOSS 12%] ?``  — q-saturated marker (engine couldn't
                               distinguish actions)

The bucket label text comes from ``sts2_rs.WIN_PROB_BUCKETS`` —
canonical labels at the time of writing: ``likely-win``, ``even``,
``likely-loss``.

The right-padded width is :data:`TAG_WIDTH`. Pick a value wide enough
to fit the longest label (``LIKELY-LOSS``) + percentage + flags;
adjust here when adding new buckets / markers.
"""

from __future__ import annotations

from sts2_hint.recommender import HintTag

# Width chosen so that ``[LIKELY-WIN 100%] ▲`` (the longest plausible
# bucket label + 3-digit pct + best arrow) fits. ``▲`` is one display
# column under monospaced terminals despite being a 3-byte UTF-8
# sequence, so we measure by ``len`` of the rendered string treating
# the arrow as 1 char.
TAG_WIDTH = 22

# Width for the Q/visits annotation column rendered by
# :func:`format_extras`. Sized for the longest plausible body:
# ``Q=-1.00 v=12345 w1000/l1000/t1000`` ≈ 33 chars. Bump as needed.
EXTRAS_WIDTH = 34

BEST_ARROW = "▲"  # ▲
SAT_MARKER = "?"  # appended when the MCTS Q-spread is saturated


def _pct(p: float) -> int:
    """Round to nearest integer percent, clamped to [0, 100]."""
    if p < 0.0:
        return 0
    if p > 1.0:
        return 100
    return round(p * 100.0)


def format_tag(tag: HintTag) -> str:
    """Render one :class:`HintTag` as a fixed-width column.

    The output is *always* :data:`TAG_WIDTH` characters wide (right-
    padded with spaces) so caller code can interleave tagged and
    untagged rows without losing alignment.
    """
    label = tag.bucket_label.upper()
    pct = _pct(tag.win_prob)
    body = f"[{label} {pct}%]"
    suffix_parts: list[str] = []
    if tag.is_best:
        suffix_parts.append(BEST_ARROW)
    if tag.q_saturated:
        suffix_parts.append(SAT_MARKER)
    suffix = " " + " ".join(suffix_parts) if suffix_parts else ""
    rendered = body + suffix
    # Left-justify within TAG_WIDTH so the percentage column lines up.
    if len(rendered) < TAG_WIDTH:
        rendered = rendered + " " * (TAG_WIDTH - len(rendered))
    return rendered


def format_empty() -> str:
    """Return a blank cell of :data:`TAG_WIDTH` width.

    Use this for rows that have no hint (e.g. un-recommendable actions
    or untagged context lines) so the column alignment matches tagged
    rows.
    """
    return " " * TAG_WIDTH


def format_extras(tag: HintTag) -> str:
    """Render Q + visit breakdown for the verbose hint column.

    Example outputs:
        ``Q=+0.93 v=386``              — clean win, no ties/losses
        ``Q=+0.85 v=256 w243/t13``     — mostly wins, a few depth-cap ties
        ``Q=-0.10 v=120 w50/l40/t30``  — mixed; losses surfaced

    The Q value is the **continuous** mean (`value_sum / visits`) from
    the v2oh reward — that's the scalar action ranking actually uses,
    distinct from the bucket-derived win-rate shown in
    :func:`format_tag`. Padded to a fixed width so columns line up
    across rows.
    """
    sign = "+" if tag.q >= 0 else "-"
    body = f"Q={sign}{abs(tag.q):.2f} v={tag.visits}"
    # Compact w/l/t suffix: omit zero buckets so simple wins stay short.
    parts: list[str] = []
    if tag.wins:
        parts.append(f"w{tag.wins}")
    if tag.losses:
        parts.append(f"l{tag.losses}")
    if tag.ties:
        parts.append(f"t{tag.ties}")
    # Only annotate when the row isn't a pure-win sweep (single-bucket
    # rows are obvious from Q≈1 already; suppress to reduce noise).
    if len(parts) > 1:
        body += " " + "/".join(parts)
    return body.ljust(EXTRAS_WIDTH)


def format_extras_empty() -> str:
    """Padded blank cell with :data:`EXTRAS_WIDTH` width."""
    return " " * EXTRAS_WIDTH


def format_target(target_index: int | None, enemies: list[dict] | None) -> str:
    """Return a short ``→ N name`` label for a target enemy, or ``""``.

    ``target_index`` is the C# enemy index from the action dict.
    ``enemies`` is the live ``state["enemies"]`` list (each entry has an
    ``index`` and ``name``). If the index doesn't resolve, falls back
    to ``→ N``.
    """
    if target_index is None:
        return ""
    if enemies:
        for e in enemies:
            if e.get("index") == target_index:
                name = e.get("name") or "?"
                return f"→ {target_index} {name}"
    return f"→ {target_index}"


__all__ = [
    "BEST_ARROW",
    "EXTRAS_WIDTH",
    "SAT_MARKER",
    "TAG_WIDTH",
    "format_empty",
    "format_extras",
    "format_extras_empty",
    "format_tag",
    "format_target",
]
