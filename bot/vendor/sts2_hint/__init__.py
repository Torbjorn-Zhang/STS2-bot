"""sts2_hint — MCTS hint annotations for sts2-pycli's interactive REPL.

Bead: ``sts2-rs-7al4`` (hint-T6) — module skeleton.

This package translates Sts2Headless decision-point snapshots into the
``sts2_rs`` wire schema, calls into the Rust MCTS engine, and renders
hint tags (e.g. ``[WIN 71%] ▲``) next to numbered choices in
``play.py``'s ``show_combat`` view.

T6 ships the package skeleton: schema mirror (:mod:`sts2_hint.schema`),
error types + rotating log (:mod:`sts2_hint.errors`), import-time-
verified canonical maps (:mod:`sts2_hint.canonical_maps`), and an
:func:`annotate` entry point with a fail-loud stub body — actual hint
generation lands in T7+ (adapter, recommender, display).

Plan: ``../sts2-rs/docs/plans/2026-05-14-integrate-sts2-rs-mcts-hint-system-into-projects-sts2-sts2-c.md``
"""

from __future__ import annotations

from sts2_hint._annotate import annotate
from sts2_hint.errors import UnsupportedStateError
from sts2_hint.schema import WIRE_VERSION, StateDict

__all__ = [
    "WIRE_VERSION",
    "StateDict",
    "UnsupportedStateError",
    "annotate",
]
