"""TypedDict mirror of the ``sts2_rs`` wire schema.

Bead: ``sts2-rs-7al4`` (hint-T6).

This module is the **Python-side type contract** for the dict the
adapter (T7) emits and the Rust ``sts2_rs.parse_state_dict_py`` consumes
(``crates/sts2-py/src/state_dict.rs``). Keep it mechanically aligned
with the Rust parser — every required key in
``REQUIRED_TOP_KEYS`` / ``REQUIRED_PLAYER_KEYS`` / etc. appears here as
a non-:class:`~typing.NotRequired` field; optional fields are
:class:`~typing.NotRequired`.

Wire-version policy: bump :data:`WIRE_VERSION` (and the Rust
``WIRE_VERSION`` constant) whenever a non-additive schema change lands.
Additive fields (new optional key) do NOT require a bump; older Rust
builds will ignore the unknown key and parse what they recognise.

This module is intentionally side-effect-free — it imports cleanly even
when the maturin-built ``sts2_rs`` extension is absent. (Import-time
verification against the Rust registry lives in
:mod:`sts2_hint.canonical_maps`, which logs a warning and skips if
``sts2_rs`` isn't importable.)
"""

from __future__ import annotations

from typing import Any, Literal, Optional

try:  # sts2-ai: our venv is Python 3.9 (typing has NotRequired from 3.11)
    from typing import NotRequired, TypedDict
except ImportError:
    from typing_extensions import NotRequired, TypedDict

# ---------------------------------------------------------------------------
# Wire-format version
# ---------------------------------------------------------------------------

#: Wire-format version. Mirrors ``crates/sts2-py/src/state_dict.rs::WIRE_VERSION``.
#: Bump in lockstep with the Rust side on any non-additive change.
WIRE_VERSION: int = 1

# ---------------------------------------------------------------------------
# Enum literals — mirror the ``VALID_*`` arrays in the Rust parser.
# ---------------------------------------------------------------------------

Phase = Literal[
    "Setup",
    "PlayerDraw",
    "PlayerPlayPhase",
    "EnemyPhase",
    "EndOfTurn",
    "Victory",
    "Defeat",
]

Side = Literal["Player", "Enemy"]

PlayerClass = Literal["ironclad", "silent", "defect", "necrobinder", "regent"]

CardKind = Literal["card", "status", "curse"]

#: Which convention a state dict's Attack telegraphs use — see
#: :class:`StateDict`. Absent means ``"effective"``.
IntentAmounts = Literal["effective", "base"]

CardFlag = Literal[
    "free_once",
    "free_turn",
    "retain",
    "ethereal",
    "unplayable",
    "innate",
]


# ---------------------------------------------------------------------------
# Leaf types
# ---------------------------------------------------------------------------


class RelicSpec(TypedDict):
    """A relic entry on ``player.relics``. ``counter`` is ``None`` for
    relics that don't carry per-instance state (most of them).
    """

    id: str
    counter: NotRequired[int | None]


class PowerSpec(TypedDict):
    """A power entry on either ``player.powers`` or ``encounter[i].powers``.

    ``extra`` is opaque to the Python side — the Rust parser only
    checks for its presence at T1 and T2 routes it through the
    matching power's apply hook.
    """

    name: str
    amount: int
    extra: NotRequired[dict[str, Any] | None]


class OrbSpec(TypedDict):
    """A Defect orb slot. ``type`` is one of EMPTY / LIGHTNING / FROST /
    DARK / PLASMA — matches ``OrbKind::python_name``.
    """

    type: str
    passive: int
    evoke: int


class OstySpec(TypedDict):
    """Necrobinder's Osty companion."""

    alive: bool
    hp: int
    max_hp: int
    block: int


class IntentSpec(TypedDict, total=False):
    """An entry in ``encounter[i].intents``.

    Opaque to the Rust parser at T1 (only ``len(intents)`` is read), so
    this type is **structurally permissive** — fields are
    ``total=False`` and additional keys are tolerated. T7's adapter
    will populate whichever shape Sts2Headless emits.
    """

    kind: str
    amount: int
    times: int


class CardSpec(TypedDict):
    """One card across draw/discard/hand/exhaust/master_deck.

    The ``id`` ↔ ``kind`` rules mirror the Rust parser:

    * ``kind == "card"`` → ``id`` must resolve to a canonical CardId.
    * ``kind == "status"`` → ``id`` is either a real CardId or a
      synthetic ``"status:Burn"`` / ``"status:Wound"`` form.
    * ``kind == "curse"`` → same as ``"status"`` but with ``"curse:"``
      prefix when synthetic.

    ``stable_idx`` is the **hand-position key** carried through to the
    recommender (T8). The adapter assigns it from the C# ``card.index``
    field; the recommender uses it to align C# action indices with
    sts2-rs action ids.

    ``cost_override`` is the card's *resolved* cost, every modifier folded
    in. ``base_cost`` is the instance's own cost — local modifiers applied,
    the global power hooks excluded — which is what sts2-rs stores in
    ``Card.cost`` and prefers when rebuilding, so a spent power's discount
    does not reconstruct as a permanent one. ``-1`` marks an X-cost card.

    ``persistent_cost`` is what ``base_cost`` becomes once this turn's cost
    modifiers wear off. Local modifiers carry a lifetime (end of combat / end
    of turn / when played) that a single number cannot express, and the
    difference is what a reconstruction needs to know to let a *transient*
    discount expire on schedule. Absent when it cannot differ from
    ``base_cost``.

    ``played_cost`` is what ``base_cost`` becomes the moment the card is
    PLAYED — a different boundary from ``persistent_cost``. The game's "costs N
    this turn" helpers are really ``SetThisTurnOrUntilPlayed``
    (``EndOfTurn | WhenPlayed``, whichever comes first), so a discount can
    retire at either edge and the two numbers do not coincide. Absent when it
    cannot differ from ``base_cost``.

    ``stable_cost`` is the cost once BOTH edges have passed. It is not
    redundant — though not because the other two collide: a single
    ``EndOfTurn | WhenPlayed`` modifier folds to ``(base, base)`` while two
    separate modifiers fold to ``(M2(base), M1(base))``, which differ unless
    both are no-ops. What the pair cannot say is how much of the shave SURVIVES
    the first edge — combined, the whole shave retires at whichever fires first;
    split, each edge takes only its own half — and this third number is what
    separates those. The three are ALL-OR-NOTHING on the wire: ``sts2_rs``
    rejects a card that carries some of them, because its re-bank step reads
    ``stable_cost`` back through a default-0 accessor and cannot otherwise tell
    an absent value from a real zero.

    ``combat_vars`` is opaque per-instance state; it's round-tripped
    but not interpreted on the Python side.
    """

    id: str
    stable_idx: int
    kind: CardKind
    upgraded: bool
    cost_override: NotRequired[int | None]
    base_cost: NotRequired[int | None]
    persistent_cost: NotRequired[int | None]
    played_cost: NotRequired[int | None]
    stable_cost: NotRequired[int | None]
    flags: NotRequired[list[CardFlag]]
    enchantment: NotRequired[str | None]
    enchantment_amount: NotRequired[int | None]
    combat_vars: NotRequired[dict[str, Any] | None]


class EnemySpec(TypedDict):
    """One entry in ``encounter``. The Sts2Headless side keeps dead /
    escaped enemies in the list; sts2-rs preserves both flags so that
    multi-enemy logic can re-key against ``uid``.
    """

    kind: str
    uid: int
    hp: int
    max_hp: int
    block: int
    powers: list[PowerSpec]
    intents: list[IntentSpec]
    intent_history: list[int]
    is_dead: bool
    is_escaped: bool


# ---------------------------------------------------------------------------
# Player / deck / rng
# ---------------------------------------------------------------------------


class RngSpec(TypedDict, total=False):
    """Optional RNG block. Both fields are individually optional on the
    Rust side; ``ascension`` defaults to 0 when omitted.
    """

    seed: int
    ascension: int
    #: The RUN's once-hashed master seed (C# ``RunRngSet.Seed``), which is what
    #: sts2-rs's ``RunRngSet::new`` takes. Unrelated to ``seed`` above, which
    #: is the *search* seed.
    run_seed: int
    #: Per-stream read counts, keyed by the ``RunRngType`` snake_case names.
    #: Applied on top of the streams derived from ``run_seed``, which puts a
    #: reconstruction on the same die roll the producer is about to make.
    #: Meaningless — and ignored — without ``run_seed``.
    counters: dict[str, int]


class PlayerSpec(TypedDict):
    """The ``player`` sub-object. ``orbs``/``orb_slots``/``stars``/``osty``
    are character-specific and only present when relevant.

    ``stance`` was an accepted key until GH #408 removed the dead STS1
    stance state from the engine. The Rust parser ignores unknown keys, so
    a producer still emitting it parses fine — the value is simply dropped.
    """

    class_: PlayerClass  # exposed as ``class`` in the wire dict; alias below
    hp: int
    max_hp: int
    block: int
    energy: int
    max_energy: int
    energy_regen_per_turn: int
    relics: list[RelicSpec]
    powers: list[PowerSpec]
    potions: list[str | None]
    #: Belt capacity (C# ``Player.MaxPotionCount``). ``potions`` is not the
    #: slot count on every producer, so it rides in its own key (GH #893).
    potion_slots: NotRequired[int]
    orbs: NotRequired[list[OrbSpec] | None]
    orb_slots: NotRequired[int | None]
    stars: NotRequired[int | None]
    osty: NotRequired[OstySpec | None]
    #: Run gold. Optional so older producers still parse; gates in-combat
    #: relic hooks (SealOfGold spends 5 gold for +1 energy each turn start).
    gold: NotRequired[int]


# Note: TypedDict cannot use the reserved keyword ``class`` directly as
# a field name. Adapters MUST emit the literal key ``"class"`` in the
# JSON dict; the TypedDict above uses ``class_`` so static type
# checkers can name the field, with the runtime key resolved via the
# functional-syntax alias below.
#
# Use this alias when constructing dicts that you want mypy to check:
#     player: PlayerSpec = {"class": "ironclad", "hp": 75, ...}  # type: ignore[typeddict-item]
# or build from this functional-form alias:
PlayerSpecRuntime = TypedDict(
    "PlayerSpecRuntime",
    {
        "class": PlayerClass,
        "hp": int,
        "max_hp": int,
        "block": int,
        "energy": int,
        "max_energy": int,
        "energy_regen_per_turn": int,
        "relics": list[RelicSpec],
        "powers": list[PowerSpec],
        "potions": list[Optional[str]],
        "potion_slots": NotRequired[int],
        "orbs": NotRequired[Optional[list[OrbSpec]]],
        "orb_slots": NotRequired[Optional[int]],
        "stars": NotRequired[Optional[int]],
        "osty": NotRequired[Optional[OstySpec]],
        "gold": NotRequired[int],
    },
)


class DeckSpec(TypedDict):
    """The four combat piles + an optional ``master_deck`` snapshot.

    The Rust parser treats ``master_deck`` as best-effort: if the
    Sts2Headless JSON doesn't carry it, omit the key (or set to None)
    and sts2-rs leaves its internal master deck empty.

    The optional tail below exists for the persistent combat mirror
    (``python/combat_mirror.py``) and is absent from every older producer:

    * ``play_pile`` — cards played but not yet flushed. Non-empty only in a
      mid-resolution snapshot, which is exactly when a card-select screen is
      open.
    * ``pinned_top`` / ``pinned_bottom`` — draw-pile **indices** naming the
      cards the player knows the position of (Headbutt-style pins). Indices,
      not uids: the Rust rebuild reallocates every uid.
    * ``deferred_draw_order`` — the draw pile's unknown middle is a canonical
      belief multiset rather than a materialized order.
    """

    draw_pile: list[CardSpec]
    discard_pile: list[CardSpec]
    hand: list[CardSpec]
    exhaust_pile: list[CardSpec]
    master_deck: NotRequired[list[CardSpec] | None]
    play_pile: NotRequired[list[CardSpec]]
    pinned_top: NotRequired[list[int]]
    pinned_bottom: NotRequired[list[int]]
    deferred_draw_order: NotRequired[bool]


# ---------------------------------------------------------------------------
# Top-level state
# ---------------------------------------------------------------------------


class StateDict(TypedDict):
    """Top-level wire dict, mirroring ``REQUIRED_TOP_KEYS`` in
    ``crates/sts2-py/src/state_dict.rs``.

    The fields ``active_attack``, ``pending_choice``, ``pending_play``
    are documented to be ``None`` at hint-time; the Rust parser raises
    :class:`~sts2_hint.errors.UnsupportedStateError` if any of them is
    set, since mid-resolution states cannot be safely rolled out.

    ``intent_amounts`` declares which convention this dict's Attack
    telegraphs use, because the two producers disagree and the difference is
    load-bearing. Sts2Headless / the live game / this adapter emit the
    **UI-effective** number (Strength and Vulnerable already folded in), and
    the Rust parser inverts them so ``perform_intent`` doesn't double-count;
    ``sts2_rs.state_to_dict`` emits the engine's own **base** amount and
    stamps ``"base"`` so it is ingested verbatim. Omitting the key means
    ``"effective"`` — the behavior every producer predating it relies on.
    """

    wire_version: int
    phase: Phase
    round: int
    side: Side
    is_elite: bool
    player: PlayerSpec
    deck: DeckSpec
    encounter: list[EnemySpec]
    turn: int
    combat_id: int
    rng: NotRequired[RngSpec]
    intent_amounts: NotRequired[IntentAmounts]
    active_attack: NotRequired[None]
    pending_choice: NotRequired[None]
    pending_play: NotRequired[None]


__all__ = [
    "WIRE_VERSION",
    "CardFlag",
    "CardKind",
    "CardSpec",
    "DeckSpec",
    "EnemySpec",
    "IntentAmounts",
    "IntentSpec",
    "OrbSpec",
    "OstySpec",
    "Phase",
    "PlayerClass",
    "PlayerSpec",
    "PlayerSpecRuntime",
    "PowerSpec",
    "RelicSpec",
    "RngSpec",
    "Side",
    "StateDict",
]
