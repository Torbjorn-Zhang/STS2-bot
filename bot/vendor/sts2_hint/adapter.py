"""Sts2Headless ``combat_play`` JSON → ``sts2_rs`` wire-schema dict.

Bead: ``sts2-rs-lv9d`` (hint-T7, sts2-pycli side).

The C# headless game emits a heterogeneous JSON state object at every
decision point; ``sts2_rs.state_from_dict`` consumes a much stricter
wire dict modelled in :mod:`sts2_hint.schema`. This module bridges the
two:

1. Reads the raw ``combat_play`` dict surfaced by
   :class:`sts2_client.Game` (or any test fixture matching the same
   shape).
2. Looks up every display name (relics, potions, powers, monsters,
   intents, orbs, status cards) against :mod:`sts2_hint.canonical_maps`.
3. Re-emits the data as a :class:`sts2_hint.schema.StateDict` ready for
   ``sts2_rs.state_from_dict``.

Fail-loud philosophy
====================

Anything we can't map — unknown relic title, unmodelled power name,
mid-resolution ``pending_choice`` — raises
:class:`sts2_hint.errors.UnsupportedStateError` with the offending
``<path>: <reason>`` message. We *never* silently elide a field that
matters for play correctness; the recommender depends on a strict
state and the hint system would happily emit confident-but-wrong
win-prob tags if we tolerated drift here.

Limitations the C# state surfaces force on us
=============================================

* **No canonical ids.** Sts2Headless emits localized display names for
  relics / potions / powers / monsters. We reverse-lookup via the
  English localization (see :mod:`sts2_hint.canonical_maps`). If
  Sts2Headless is run with ``lang != "en"`` this adapter will fail
  loud — that's intentional; running hints in a non-English
  localization would otherwise silently mismap.
* **No base cost.** The C# JSON only carries the *resolved* cost. We
  emit it as ``cost_override`` so the Rust parser uses it directly.
  This means even *cards-not-discounted* cards carry ``cost_override``;
  the round-trip ``state_equiv()`` predicate is robust to this
  (cost is a derived field on the Rust side too).
* **No enemy uid.** We synthesise a uid from ``encounter[i].index``.
  Stable within a turn; across turns when an enemy dies and the list
  re-indexes, uids drift — the hint recommender re-reads on every
  ``annotate()`` call so this is OK.
* **No master_deck from Sts2Headless.** The historical combat-scope JSON
  doesn't include the run-scope deck; when the input dict has no
  ``"master_deck"`` key at all we still emit ``master_deck=None`` (the
  ``sts2_rs.state_from_dict`` best-effort no-op documented on
  ``ParsedDeck.master_deck``), so event-reward relics that depend on it
  will mis-fire in rollouts fed from that source. Documented in the plan's
  Risk 6 / 11. A caller that *does* carry the run-scope deck (e.g. the
  real-game bridge's ``realgame_adapter.to_headless_combat_play``, which
  forwards ``run_state.master_deck``) gets it translated and threaded
  through -- see ``master_deck_raw`` below.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
import collections
import functools
import logging as _logging
import re
import threading
from typing import Any

from sts2_hint.canonical_maps import (
    CARD_ID_PREFIX_TO_CLASS,
    CARD_TITLE_TO_ID,
    INTENT_TYPES,
    MONSTER_TITLE_TO_ID,
    ORB_TITLE_TO_ID,
    POTION_TITLE_TO_ID,
    POWER_TITLE_TO_ID,
    RELIC_TITLE_TO_ID,
    STATUS_CARD_TITLES,
)
from sts2_hint.errors import UnsupportedStateError
from sts2_hint.schema import WIRE_VERSION

_log = _logging.getLogger("sts2_hint")

#: Engine monster-kind names (the map's values) — used to resolve raw
#: ``<ENTRY>.name`` loc-key leaks from Sts2Headless directly to a kind.
#: Extended with kinds that share a localized title and therefore never
#: appear as map values: the Decimillipede segment variants all render as
#: one title, but the engine models Front/Middle/Back as distinct kinds
#: with their own AIs — collapsing them to the generic segment kind gave
#: every segment the wrong per-position moveset.
_KNOWN_MONSTER_KINDS = frozenset(MONSTER_TITLE_TO_ID.values()) | {
    "DecimillipedeSegmentFront",
    "DecimillipedeSegmentMiddle",
    "DecimillipedeSegmentBack",
}

#: Same specimen-template stripping ``display_maps.monster_title`` already
#: applies in the id->title direction (workbench display), reused here for
#: title->id. A monster whose in-combat ``Title`` embeds an unresolved loc
#: variable (Test Subject's ``"Test Subject #C{Count}"`` -- ``{Count}`` is
#: filled from cross-run save progress the simulator doesn't model) also has
#: a SEPARATE, plain-text title used outside combat: the C# encounter class's
#: own title (``TEST_SUBJECT_BOSS.title`` = ``"Test Subject"``,
#: ``localization_eng/encounters.json``) — the Act-3 boss-fight/map-node
#: label. Both titles name the same species/``MonsterKind``, but only the
#: templated form is a key in ``MONSTER_TITLE_TO_ID`` (built from
#: ``monsters.json``'s ``.name`` entries), so the plain encounter-title form
#: fell through every fallback and raised `unknown monster name 'Test
#: Subject'` on real Act-3 boss fights.
_SPECIMEN_TEMPLATE = re.compile(r"\s*#\S*\{[^}]+\}.*$")
_LOC_TEMPLATE_VAR = re.compile(r"\s*\{[^}]+\}")


@functools.lru_cache(maxsize=1)
def _bare_species_title_to_id() -> dict[str, str]:
    """``{bare species title -> monster id}`` for every templated entry in
    ``MONSTER_TITLE_TO_ID`` (currently just Test Subject; generalises to any
    future monster whose title embeds an unresolved loc variable)."""
    out: dict[str, str] = {}
    for title, mid in MONSTER_TITLE_TO_ID.items():
        bare = _LOC_TEMPLATE_VAR.sub("", _SPECIMEN_TEMPLATE.sub("", title)).strip()
        if bare and bare != title:
            out.setdefault(bare, mid)
    return out


# --------------------------------------------------------------------------- #
# Soft-drop telemetry. A soft-drop silently removes an entity the search can't
# map from the state the MCTS plans over. At INFO level with no counter it hid a
# 55%->0% real-world combat regression (67 `_CARD`-suffixed cards + the starter
# relic dropped). Count drops by kind so an eval harness can SURFACE them: a
# degraded search should read as "N dropped", not a silent 0 wins. Thread-safe
# because recommend_all runs the translation off-GIL across worker threads.
# --------------------------------------------------------------------------- #
_DROP_LOCK = threading.Lock()
_DROP_COUNTS: collections.Counter[str] = collections.Counter()


def reset_drop_counts() -> None:
    """Zero the soft-drop counters — call before a run to scope the tally."""
    with _DROP_LOCK:
        _DROP_COUNTS.clear()


def drop_counts() -> dict[str, int]:
    """Snapshot of soft-drop counts by kind (``card`` / ``relic`` / ``power`` /
    ``potion`` / ``other``) plus ``total``. Empty == the search saw the full
    state; a non-empty tally means the MCTS planned over a degraded state."""
    with _DROP_LOCK:
        return dict(_DROP_COUNTS)


def _drop_kind(path_prefix: str) -> str:
    p = path_prefix.lower()
    if "relic" in p:
        return "relic"
    if "power" in p:
        return "power"
    if "potion" in p:
        return "potion"
    if any(k in p for k in ("hand", "draw", "discard", "exhaust", "deck", "card")):
        return "card"
    return "other"


def _record_drop(path_prefix: str) -> None:
    kind = _drop_kind(path_prefix)
    with _DROP_LOCK:
        _DROP_COUNTS[kind] += 1
        _DROP_COUNTS["total"] += 1


def _soft_translate_list(
    items: list[Any],
    translator: Any,
    path_prefix: str,
) -> list[Any]:
    """Translate a list of raw items via ``translator(raw, path)``,
    dropping any whose lookup fails on a content-port gap.

    The fail-loud policy is preserved for genuine schema bugs (those
    bubble out of ``translator`` via other paths). Only the narrow
    ``UnsupportedStateError`` raised by an unknown-id lookup is
    converted to a logged drop here, so MCTS hints stay usable when
    sts2-rs's registry lags STS2 content.
    """
    out: list[Any] = []
    for i, raw in enumerate(items):
        try:
            out.append(translator(raw, f"{path_prefix}[{i}]"))
        except UnsupportedStateError as e:
            _log.info("adapter: soft-dropping %s[%d]: %s", path_prefix, i, e)
            _record_drop(path_prefix)
    return out


def _require(obj: Any, key: str, path: str, *, allow_none: bool = False) -> Any:
    """Return ``obj[key]`` or raise ``UnsupportedStateError`` with a
    full json-path message."""
    if not isinstance(obj, dict):
        raise UnsupportedStateError(f"{path}: expected dict, got {type(obj).__name__}")
    if key not in obj:
        raise UnsupportedStateError(f"{path}.{key}: required key missing")
    value = obj[key]
    if value is None and not allow_none:
        raise UnsupportedStateError(f"{path}.{key}: required value is null")
    return value


def _flags_for_card(card: dict[str, Any]) -> list[str]:
    """Extract CardSpec flags from a hand-card dict.

    Two producer dialects reach here. The C# headless exposes the keyword set
    as a ``keywords`` list; the SpireMentor mod exposes it as individual bools
    (``retain``, ``sly``, ``ethereal``, ...) and ships no ``keywords`` key at
    all. Both are read for the keyword SET, but only the headless list is
    mapped onto keyword flags — see the emission note below.

    On top of the keyword set, ``will_retain`` / ``sly_this_turn`` carry the
    SINGLE-TURN halves of Retain and Sly (GH #370) — C#
    ``ShouldRetainThisTurn`` / ``IsSlyThisTurn``, each of which is
    ``keyword || private single-turn bool``. Subtracting the keyword leaves
    exactly the transient grant (Well-Laid Plans, Hand Trick), which is what
    the engine's per-turn uid lists hold and what a rebuilt mirror otherwise
    lost. Anything unmapped is dropped (informational, not load-bearing).
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(flag: str) -> None:
        if flag not in seen:
            out.append(flag)
            seen.add(flag)

    # Map keyword names to flags — case-insensitive.
    mapping = {
        "retain": "retain",
        "sly": "sly",
        "ethereal": "ethereal",
        "innate": "innate",
        "unplayable": "unplayable",
        "freeonce": "free_once",
        "free_once": "free_once",
        "freeturn": "free_turn",
        "free_turn": "free_turn",
    }
    for kw in card.get("keywords") or []:
        if not isinstance(kw, str):
            continue
        flag = mapping.get(kw.lower().replace(" ", "_"))
        if flag is not None:
            add(flag)

    # SpireMentor dialect: one bool per keyword instead of a `keywords` list.
    #
    # Only `sly` is EMITTED from these. The mirror diffs the flag tuple, so a
    # flag the world names and `state_to_dict` does not is a per-decision
    # desync on every affected card — and sts2_rs emits `sly` (GH #370) but
    # deliberately not `retain`. `ethereal`/`innate` are simply absent on the
    # live arm today; folding them in is a real fix but a separate one, with
    # its own symmetry question to answer first. `keyword_retain` therefore
    # feeds the subtraction below WITHOUT being emitted.
    if card.get("sly") is True:
        add("sly")
    keyword_retain = "retain" in seen or card.get("retain") is True

    # The single-turn halves. Ordered AFTER both keyword readers above, so the
    # keyword is known whichever dialect supplied it.
    if card.get("will_retain") is True and not keyword_retain:
        add("retain_turn")
    if card.get("sly_this_turn") is True and "sly" not in seen:
        add("sly_turn")
    return out


# ---------------------------------------------------------------------------
# Sub-element translators
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=8)
def _registry_names(kind: str) -> frozenset[str]:
    """Valid ``sts2_rs`` registry ids for *kind* (``card``/``relic``/``power``).

    Empty when ``sts2_rs`` is unavailable, so callers keep their prior
    title-map behaviour instead of over-dropping in a build-less environment.
    """
    try:
        import sts2_rs  # type: ignore[import-not-found]

        list_names = getattr(sts2_rs, "list_registry_names", None)
        if list_names is None:
            return frozenset()
        return frozenset(list_names(kind))
    except Exception:
        return frozenset()


def _registry_id_from_leaked(name: str, kind: str, suffixes: tuple[str, ...] = ()) -> str | None:
    """Resolve a *leaked localization key* (``BURNING_BLOOD.title``,
    ``TAINTED_POWER.title``) to a registry id.

    The C# backend emits a resolved display title for most content, but content
    without a ``localization_eng`` entry leaks its raw loc-key (``<STEM>.title``)
    instead. The title maps key on display names and miss these, so the adapter
    was soft-dropping the power/relic out of the search state (real-world combat
    under-modelling). Strip ``.title`` and try the stem against the registry,
    optionally stripping a kind-specific suffix the registry drops (``_POWER``:
    the registry stores ``TAINTED``/``FLEX_POTION``, not ``TAINTED_POWER``).
    Returns None when there is no engine registry or no candidate matches.
    """
    stem = name[: -len(".title")] if name.endswith(".title") else name
    stem = stem.upper()
    candidates = [stem, *(stem[: -len(s)] for s in suffixes if stem.endswith(s))]
    reg = _registry_names(kind)
    for cand in candidates:
        if cand in reg:
            return cand
    return None


def _relic_id_from_wire(raw: dict[str, Any]) -> str | None:
    """A registry-valid relic id straight off the wire's ``id`` field, if any.

    Both producers already publish the canonical ``RelicModel.Id.Entry``
    (Sts2Headless ``RunSimulator.cs:5047``; SpireMentor
    ``SimStateExtractor.SerializeRelic``), and it is strictly more reliable than
    the rendered title:

    * ``RelicModel.Title`` PREFIXES a wax relic with ToyBox's ``"Wax {Title}"``
      template (RelicModel.cs:48-61, ToyBox.cs:26, ``relics.json``
      ``"TOY_BOX.waxRelicPrefix"``), so a waxed Burning Blood arrives as
      ``"Wax Burning Blood"`` and misses ``RELIC_TITLE_TO_ID`` entirely -- for
      all 294 relics, and ToyBox hands out four at a time (ToyBox.cs:87-95).
    * Relics present in the live build but absent from this repo's
      ``localization_eng/relics.json`` snapshot (FISHING_ROD, SILKEN_TRESS)
      have no title key to look up at all.

    Both used to raise ``UnsupportedStateError`` and get soft-dropped out of the
    reconstructed state. This is additive: the id is only accepted when the
    engine registry actually knows it (and the registry is empty when
    ``sts2_rs`` is unavailable), so the title path stays in charge everywhere
    it already worked -- including the headless arm, whose ``id`` is the same
    ``Id.Entry`` and resolves identically.
    """
    raw_id = raw.get("id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        return None
    candidate = raw_id.strip().upper()
    return candidate if candidate in _registry_names("relic") else None


def _translate_relic(raw: dict[str, Any], path: str) -> dict[str, Any]:
    relic_id = _relic_id_from_wire(raw)
    if relic_id is None:
        name = _require(raw, "name", path)
        relic_id = RELIC_TITLE_TO_ID.get(name)
        if relic_id is None:
            # Leaked loc-key (`BURNING_BLOOD.title`) — the starter relic and
            # other unlocalized relics arrive this way; resolve the stem via
            # the registry.
            relic_id = _registry_id_from_leaked(name, "relic")
        if relic_id is None:
            raise UnsupportedStateError(f"{path}: unknown relic title {name!r}")
    # The relic's live counter. `vars` are its canonical THRESHOLDS — for
    # Nunchaku, `{Cards: 10, Energy: 1}` — and the progress toward them is a
    # separate value the game exposes as `DisplayAmount`.
    #
    # This used to guess: "pick the first int-valued var". For every
    # counter-driven relic that guess produced the threshold instead of the
    # progress, which is worse than nothing — a reconstructed Nunchaku started
    # at 10 of 10 attacks and then fired on the 20th rather than the 10th. On
    # the v10 sweep the counter relics were the top lift over baseline in the
    # single-field `$.player.energy` class (188 records with the mirror exactly
    # 1 energy low): Nunchaku 3.0x, Kusarigama 3.3x, HappyFlower 2.3x.
    #
    # Equivalence is worth stating, because C# and the engine count
    # differently and it does not matter: C# reports progress modulo the
    # threshold (`AttacksPlayedThisTurn % Cards`) while the engine keeps a
    # running total and triggers on `counter % N == 0`. Since
    # `(total + k) % N == 0` exactly when `(total % N + k) % N == 0`, the two
    # fire on the same play. Checked against the decomp for Nunchaku,
    # Kusarigama and HappyFlower.
    #
    # A producer that sends no counter leaves it unset rather than guessed.
    counter = raw.get("counter")
    if isinstance(counter, bool) or not isinstance(counter, int):
        counter = None
    out: dict[str, Any] = {"id": relic_id, "counter": counter}
    # ToyBox wax/melt. A melted relic stays in the inventory but is filtered
    # from every hook dispatch (CombatState.cs:130-137), and both producers
    # ship the flags (RunSimulator.cs relic serializer, SimStateExtractor.cs
    # :567). Dropping them here made every reconstructed mirror play melted
    # relics as live — the d1 corpus's dominant class (54/158 records, block
    # the mirror over-gains on ToyBox boards, seeds 1001/1018). Same
    # only-when-true convention as the engine's own wire.
    if raw.get("is_wax"):
        out["is_wax"] = True
    if raw.get("is_melted"):
        out["is_melted"] = True
    return out


def _translate_potion(name: str | None, path: str) -> str | None:
    if name is None:
        return None
    pid = POTION_TITLE_TO_ID.get(name)
    if pid is None:
        # Soft-drop: unported potion → empty slot. The MCTS rollouts
        # won't model its effect, but hints remain usable.
        _log.info("adapter: soft-dropping %s: unknown potion title %r", path, name)
        _record_drop(path)
        return None
    return pid


@functools.lru_cache(maxsize=1)
def _registry_power_names() -> frozenset[str]:
    """The valid ``sts2_rs`` power ids (best-effort).

    Empty when ``sts2_rs`` is unavailable (e.g. a build-less CI), in which case
    callers fall back to trusting a carried ``id`` — there is no engine to feed
    the translation to anyway.
    """
    try:
        import sts2_rs  # type: ignore[import-not-found]

        list_names = getattr(sts2_rs, "list_registry_names", None)
        if list_names is None:
            return frozenset()
        # Every build's powers ("power" is v0.107.1's alone, an embedding
        # vocabulary that must not grow); older engines lack the kind.
        try:
            return frozenset(list_names("power_all"))
        except ValueError:
            return frozenset(list_names("power"))
    except Exception:
        return frozenset()


#: (engine power id, C# power `DynamicVar` name) -> the engine field that
#: holds the same quantity. Deliberately a short, explicit, power-scoped
#: table: a var only appears here once its meaning has been checked against
#: the engine's own use of that field, because forwarding an unmapped var
#: would put a number the engine reads for something else into a slot that
#: changes behaviour. Power-scoped because one var name can mean different
#: things on different powers (CrabRagePower also declares a `Block` var,
#: but its Rust arm reads constants, not `counter`).
_POWER_VAR_TO_ENGINE_FIELD = {
    # CrimsonMantlePower / InfernoPower: damage-to-self accumulator,
    # incremented once per copy played. `Power.counter` on the engine side.
    # (The only two powers in the v0.107.1 decompile with a `SelfDamage` var.)
    ("CRIMSON_MANTLE", "SelfDamage"): "counter",
    ("INFERNO", "SelfDamage"): "counter",
    # SkittishPower's once-per-turn `hasGainedBlockThisTurn`, which the mod
    # reads off its public getter and sends with the vars (GH #989). The
    # engine keeps the same latch on `Power.counter`.
    ("SKITTISH", "HasGainedBlockThisTurn"): "counter",
    # SurroundedPower's `Facing` (Right 0, Left 1), which the mod reads off
    # its public getter and sends with the vars (GH #989): the side a player
    # turned to decides which crab arm's hits take 1.5x. The engine keeps it
    # on `Power.counter` (`SURROUNDED_FACING_RIGHT` / `_LEFT`, same values).
    # Ours is inferred from the arms' intents without it; a teammate's is not.
    ("SURROUNDED", "Facing"): "counter",
    # ToricToughnessPower.SetBlock stashes the block RE-GRANTED at the start
    # of the owner's next turns (the play-time GainBlock result: 5/7 upgraded,
    # Dex/Frail already folded in) in `DynamicVars.Block`; `amount` is only
    # the turns-left counter and says nothing about it. The engine keeps the
    # stored block on `Power.counter` and re-grants it Unpowered — dropping
    # the var rebuilt every mirror at counter=0, so the re-grant handed out 0
    # block: 303 of the 311 positive `$.player.block` end_turn divergences on
    # the 2026-08-01 v27wr sweep.
    ("TORIC_TOUGHNESS", "Block"): "counter",
}


def _translate_power(raw: dict[str, Any], path: str) -> dict[str, Any]:
    name = _require(raw, "name", path)
    # Prefer the engine power id when the source carries one that is a real
    # registry power. The Rust backend's ``state_to_dict`` / recorded-run path
    # emits ``id`` = the registry python_name (e.g. ``MANGLE_POWER``), which
    # ``state_from_dict`` accepts verbatim and which — unlike the localization
    # title — exists for *every* engine power and is unambiguous. That round-
    # trips powers whose display title has no ``powers.json`` entry (Mangle, the
    # TemporaryStrength/-Dexterity/-Focus family); without it they fell through
    # the title map and were silently dropped, stranding a temporary stat change
    # as permanent and defanging the enemy (the Entomancer "MCTS suggests end
    # turn" bug).
    #
    # The id is validated against the registry first because the C# backend
    # emits its canonical ``Id.Entry``, which keeps a ``_POWER`` suffix the Rust
    # registry strips for most powers (``VULNERABLE_POWER`` vs the engine's
    # ``VULNERABLE``). Such an id is NOT a registry power, so it falls back to
    # the (correct) localization title map; only a genuinely valid id is trusted
    # directly. When ``sts2_rs`` is unavailable the id is trusted as-is.
    raw_id = raw.get("id")
    registry = _registry_power_names()
    if raw_id and (not registry or raw_id in registry):
        pid: str | None = raw_id
    elif raw_id and raw_id.removesuffix("_POWER") in registry:
        # C# canonical id with the ``_POWER`` suffix the Rust registry strips
        # (``BACK_ATTACK_RIGHT_POWER`` → ``BACK_ATTACK_RIGHT``). Must resolve
        # BEFORE the title map: distinct powers can share one display title —
        # both crab-arm back-attacks are titled "Back Attack", so the title
        # map collapsed Rocket's RIGHT onto Crusher's LEFT and every rollout
        # mis-assigned the Surrounded 1.5× (2026-07-17 Crusher+Rocket audit).
        pid = raw_id.removesuffix("_POWER")
    else:
        pid = POWER_TITLE_TO_ID.get(name)
    if pid is None:
        # Leaked loc-key (`TAINTED_POWER.title`, `FLEX_POTION_POWER.title`) —
        # potion/relic-granted powers without a `powers.json` entry arrive this
        # way. The registry stores them stripped of `_POWER` (`TAINTED`,
        # `FLEX_POTION`), so resolve the stem there before dropping.
        pid = _registry_id_from_leaked(name, "power", ("_POWER",))
    if pid is None:
        raise UnsupportedStateError(f"{path}: unknown power name {name!r}")
    try:
        amount = int(raw.get("amount", 0) or 0)
    except (TypeError, ValueError) as e:
        raise UnsupportedStateError(
            f"{path}.amount: non-integer power amount {raw.get('amount')!r}"
        ) from e
    out = {"name": pid, "amount": amount}
    # Forward the C# ``DisplayAmount`` when it diverges from ``amount``: for
    # countdown-style powers (WitheringPresence) ``amount`` is the static
    # cycle length while ``display_amount`` is the live cards-left counter.
    # Dropping it reset every mid-fight reconstruction to a full cycle
    # (2026-07-17 Aeonglass audit). Best-effort: a non-integer display is
    # cosmetic, never worth failing the state over.
    # A power's per-instance accumulator lives in a DynamicVar, and `amount` /
    # `display_amount` say nothing about it. The engine keeps that accumulator
    # in `Power.counter`, so the two have to be connected by name — but only
    # where the meaning is actually the same, which is why this is an explicit
    # table and not "forward every var".
    #
    # `SelfDamage` is the clear case: CrimsonMantle's `amount` is the block it
    # grants, while the damage it deals you every turn is a var bumped once per
    # copy played (`CrimsonMantle.cs:28` -> `IncrementSelfDamage`), and Inferno
    # is its twin. With no wire representation the engine rebuilt it at zero
    # and simply skipped that damage: 104 of the 107 single-field
    # `$.player.hp` divergences on the v10 sweep were a mirror sitting exactly
    # 1 HP high at `end_turn` with CrimsonMantle up.
    pvars = raw.get("vars")
    if isinstance(pvars, dict):
        for var_name, v in pvars.items():
            target = _POWER_VAR_TO_ENGINE_FIELD.get((pid, var_name))
            if target is not None and isinstance(v, int) and not isinstance(v, bool):
                out[target] = v
    disp = raw.get("display_amount")
    if disp is not None:
        try:
            out["display"] = int(disp)
        except (TypeError, ValueError):
            pass
    # The player who applied it (GH #1004): the engine's seat id when the
    # source is an engine board, else the mod's net id, which
    # `_resolve_power_appliers` turns into one once every seat is known.
    # Imitation Learning's seat is instead the teammate it copies (C#
    # `PlayerTarget`, the mod's `player_target`, GH #992). Its `applier` is
    # its own holder, so without `player_target` (an older mod) the seat is
    # left to the engine's default: the holder's first teammate (exact with
    # two players; three or more is #1017).
    seat = raw.get("applier_seat")
    if isinstance(seat, int) and not isinstance(seat, bool):
        out["applier_seat"] = seat
    elif pid == "IMITATION_LEARNING":
        if raw.get("player_target") is not None:
            out["applier"] = str(raw["player_target"])
    elif raw.get("applier") is not None:
        out["applier"] = str(raw["applier"])
    return out


def _resolve_power_appliers(state: dict[str, Any], out: dict[str, Any]) -> None:
    """Every power's ``applier`` net id -> ``applier_seat`` (0 us, ``i + 1``
    for ``$.allies[i]``): a teammate's Flanking, Knockdown and Tag Team help
    only the players who did not apply them (GH #1004). An applier the board
    does not name, or any applier solo, is dropped: the engine reads the
    power as ours, as before."""
    seats: dict[str, int] = {}
    if state.get("net_id") is not None:
        seats[str(state["net_id"])] = 0
    for i, ally in enumerate(state.get("allies") or []):
        if ally.get("net_id") is not None:
            seats[str(ally["net_id"])] = i + 1
    lists = [(out.get("player") or {}).get("powers")]
    lists += [e.get("powers") for e in out.get("encounter") or []]
    lists += [a.get("powers") for a in out.get("allies") or []]
    for powers in lists:
        for p in powers or []:
            net_id = p.pop("applier", None)
            if net_id is not None and net_id in seats and len(seats) > 1:
                p["applier_seat"] = seats[net_id]


def _translate_intent(raw: dict[str, Any], path: str) -> dict[str, Any]:
    kind = _require(raw, "type", path)
    if kind not in INTENT_TYPES:
        # We don't fail loud here: unknown intent types are diagnostic-
        # only on the sts2-rs side at T1 (intents are largely opaque to
        # the Rust parser). Round-trip via the literal string.
        pass
    out: dict[str, Any] = {"kind": kind}
    if "damage" in raw and raw["damage"] is not None:
        out["amount"] = int(raw["damage"])
    if "hits" in raw and raw["hits"] is not None:
        out["times"] = int(raw["hits"])
    return out


def _resolve_card_id(c_id: str, path: str) -> tuple[str, str]:
    """Resolve a Sts2Headless ``c.Id.ToString()`` like ``IRONCLAD/STRIKE``
    or ``COLORLESS/MAYHEM`` to ``(wire_class_or_None, registry_card_id)``.

    Returns the *class prefix* (lower-cased, or empty when not a
    character-prefixed id like ``STATUS/BURN``) and the canonical
    registry id.
    """
    if "/" not in c_id:
        # Sts2Headless emits ``CARD.STRIKE_IRONCLAD`` style; strip the
        # ``CARD.`` prefix. The bare remainder matches most registry ids
        # directly (``STRIKE_IRONCLAD``, ``BASH``), but 67 cards are
        # registered with a ``_CARD`` suffix (``INFERNO_CARD``,
        # ``JUGGERNAUT_CARD``, ``DEMON_FORM_CARD``, ``SERPENT_FORM_CARD`` …)
        # whose C# id arrives WITHOUT the suffix (``CARD.INFERNO``). The old
        # bare-only lookup missed those, so the adapter soft-dropped the card
        # out of the search state entirely — silently gutting the deck the
        # MCTS plans over (real-world combat under-play). Try the ``_CARD``
        # variant too. No ``_CARD`` registry id has a bare duplicate, so
        # preferring the exact bare form first is unambiguous and
        # regression-free.
        #
        # ``_STATUS`` is the same shape with one member: the Knowledge Demon's
        # Sloth arrives as ``CARD.SLOTH`` and is registered ``SLOTH_STATUS``
        # (the C# class is plain ``Sloth``; see ``display_maps`` for the loc-key
        # side of the same split). Without it the resolve raised
        # ``UnsupportedStateError`` and KILLED the run — 11 of 60 seeds in the
        # 2026-07-27 rung-3.5 sweep, 9 at floor 33 and 2 (1004, 1054) at 35.
        # The two later aborts are the tell that the blast radius is not just
        # the boss's card_select: once the Sloth is TAKEN, any hand, pile or
        # later card_select carrying it raises too, because ``Sloth`` is absent
        # from ``STATUS_CARD_TITLES`` and so falls through to this resolver.
        # ``SLOTH_STATUS`` is the only ``_STATUS`` id in the 580-card registry
        # and has no bare duplicate, so the trailing position is unambiguous.
        bare = c_id.upper()
        if bare.startswith("CARD."):
            bare = bare[len("CARD.") :]
        return "", _select_card_candidate([bare, f"{bare}_CARD", f"{bare}_STATUS"], c_id, path)
    prefix, entry = c_id.split("/", 1)
    entry_u = entry.upper()

    # Try character-suffixed form first for STRIKE/DEFEND duplicates.
    char_lower_class = CARD_ID_PREFIX_TO_CLASS.get(prefix.upper(), "")
    candidates: list[str] = []
    if char_lower_class:
        candidates.append(f"{entry_u}_{prefix.upper()}")
    candidates.append(entry_u)
    candidates.append(f"{entry_u}_CARD")
    candidates.append(f"{entry_u}_STATUS")

    # If sts2_rs is available, prefer the strictest candidate that
    # exists in the card registry. Otherwise pick the first.
    chosen = _select_card_candidate(candidates, c_id, path)
    return char_lower_class, chosen


def registry_card_id(c_id: str, path: str = "$") -> str:
    """The canonical ``sts2_rs`` registry id for a Sts2Headless / SpireMentor
    card id, dropping the character-class prefix :func:`_resolve_card_id`
    also returns.

    Public because a ``card_select`` message names cards outside any pile —
    the combat mirror has to resolve them to registry ids before injecting the
    option list, and re-deriving the ``_CARD``-suffix and STRIKE/DEFEND
    disambiguation rules there would be a second copy that could drift.

    Raises:
        UnsupportedStateError: When no candidate exists in the registry.
    """
    return _resolve_card_id(c_id, path)[1]


def _select_card_candidate(candidates: list[str], c_id: str, path: str) -> str:
    """Pick a candidate id from the registry if available; else the
    first candidate. Falls back to fail-loud if we have a registry and
    none of the candidates match it.

    Reads the registry through the ``lru_cache``d :func:`_registry_names`
    rather than rebuilding a 580-entry set here — this runs once per card per
    state translation, and the set build dwarfs the handful of membership
    tests it exists to serve.

    Caveat, unchanged and deliberate: with no registry (build-less env) this
    returns ``candidates[0]``, so ``CARD.SLOTH`` becomes the nonexistent
    ``SLOTH`` instead of raising. The suffix candidates below are only
    guaranteed unambiguous on the registry-backed path.
    """
    known = _registry_names("card")
    if not known:
        return candidates[0]
    for cand in candidates:
        if cand in known:
            return cand
    raise UnsupportedStateError(
        f"{path}: card id {c_id!r} doesn't map to any of {candidates!r} in sts2_rs card registry"
    )


def _verify_card_id(bare: str, original: str, path: str) -> str:
    """Verify a bare card id against the sts2_rs registry (best-effort).

    Used by the no-slash fixture path. Fail loud only when sts2_rs is
    importable and the id is genuinely missing.
    """
    try:
        import sts2_rs  # type: ignore[import-not-found]

        list_names = getattr(sts2_rs, "list_registry_names", None)
        if list_names is None:
            return bare
        if bare not in set(list_names("card")):
            raise UnsupportedStateError(
                f"{path}: card id {original!r} → {bare!r} not in sts2_rs card registry"
            )
    except UnsupportedStateError:
        raise
    except Exception:
        return bare
    return bare


def _translate_card(raw: dict[str, Any], path: str, *, hand: bool = False) -> dict[str, Any]:
    """Translate one Sts2Headless ``CombatCard`` dict to a wire CardSpec.

    ``hand=True`` enables status/curse detection via the C# ``type``
    field (``Status``, ``Curse``) which is only fully populated on
    hand cards in the plan's edge-case table.
    """
    c_id = _require(raw, "id", path)
    stable_idx = int(_require(raw, "index", path))
    type_str = (raw.get("type") or "").lower()
    cost_resolved = raw.get("cost")

    # Status / Curse synthesis
    if type_str in ("status", "curse"):
        title = raw.get("affliction") or raw.get("name") or ""
        # Strip bilingual suffix if present (English half before " / ").
        eng_title = title.split(" / ", 1)[0].strip() if isinstance(title, str) else ""
        if eng_title in STATUS_CARD_TITLES:
            kind_label = "status" if type_str == "status" else "curse"
            wire_id = f"{kind_label}:{eng_title}"
        else:
            # Some status/curse cards are first-class registry entries
            # (BURN, WOUND, SLIMED, DAZED). Fall through to id-based
            # resolution so we use the canonical card id when present.
            _cls, wire_id = _resolve_card_id(c_id, path)
            kind_label = "status" if type_str == "status" else "curse"
        spec: dict[str, Any] = {
            "id": wire_id,
            "stable_idx": stable_idx,
            "kind": kind_label,
            "upgraded": bool(raw.get("upgraded")),
        }
    else:
        _cls, card_id = _resolve_card_id(c_id, path)
        spec = {
            "id": card_id,
            "stable_idx": stable_idx,
            "kind": "card",
            "upgraded": bool(raw.get("upgraded")),
        }

    # Cost. Two values, and both matter:
    #
    # * ``cost_override`` — the RESOLVED cost, every modifier folded in. This
    #   is what a display consumer shows and what legality is judged on.
    # * ``base_cost`` — the instance's own cost, local modifiers applied and
    #   the global power hooks excluded (C# ``GetWithModifiers(Local)``),
    #   which is precisely what the engine keeps in ``Card.cost``.
    #
    # sts2-rs rebuilds from ``base_cost`` when it is present and re-derives
    # the globals from the powers on the same wire. Rebuilding from the
    # resolved cost alone turns a TRANSIENT discount into a PERMANENT one: on
    # the 60-run realdll_v7 sweep the mirror was cheaper than the game on 1380
    # of 1662 diverging cards — a spent FreeAttack (Unrelenting) left every
    # Strike in hand at cost 0 for the rest of the fight, and the search then
    # planned hands it could not afford. Older C# builds omit ``base_cost``;
    # there we still fall back to the resolved cost, as before.
    #
    # ``persistent_cost`` is the third: what ``base_cost`` becomes once this
    # turn's cost modifiers wear off. A local modifier has a LIFETIME the
    # other two numbers cannot express — Stomp's "1 less per Attack played
    # this turn" and Enlightenment's "costs 1 this combat" both read as a
    # cheaper card, but only the first springs back at turn start — so a
    # reconstruction that keeps every discount forever diverges on exactly the
    # turn boundary that should have undone it (v8: 163 of STOMP's 175 cost
    # divergences sat on ``end_turn``). Absent when it cannot differ from
    # ``base_cost``, which is the overwhelmingly common case.
    if isinstance(cost_resolved, int):
        spec["cost_override"] = cost_resolved
    base_cost = raw.get("base_cost")
    if isinstance(base_cost, int):
        spec["base_cost"] = base_cost
    persistent_cost = raw.get("persistent_cost")
    if isinstance(persistent_cost, int):
        spec["persistent_cost"] = persistent_cost
    # ``played_cost`` is the fourth: what ``base_cost`` becomes the moment the
    # card is played. "This turn" discounts are really "this turn OR until
    # played" (C# ``SetThisTurnOrUntilPlayed``), so restoring them only at the
    # turn boundary leaves a spent discount alive on the card that spent it.
    played_cost = raw.get("played_cost")
    if isinstance(played_cost, int):
        spec["played_cost"] = played_cost
    # And the cost once BOTH edges have passed. Load-bearing rather than
    # belt-and-braces: one "this turn or until played" modifier and two
    # separate modifiers give identical (persistent, played) pairs, and only
    # this third number says whether the shave is shared or independent.
    stable_cost = raw.get("stable_cost")
    if isinstance(stable_cost, int):
        spec["stable_cost"] = stable_cost
    # The STAR half of the same story. C# ``SetToFreeThisTurn`` zeroes the
    # star cost as well as the energy cost (``CardModel.cs:1267-1271``), and
    # the energy fields above cannot say so: a Silent's Splash-generated
    # Falling Star rebuilt at 0 energy but its printed 2 stars, which a
    # star-less character can never pay, so the search listed a free attack as
    # unplayable and ended the turn over it. ``star_cost`` is the current star
    # cost (``CurrentStarCost``), ``persistent_star_cost`` what it is once this
    # turn's temporary costs expire (``GetStarCostThisCombat``); the engine maps
    # a 0 in either onto its this-turn / this-combat star-free rider.
    for star_key in ("star_cost", "persistent_star_cost"):
        v = raw.get(star_key)
        if isinstance(v, int) and not isinstance(v, bool):
            spec[star_key] = v

    # Current base attack / block from the C# `stats` block. These are the
    # card's own base values (post-upgrade, pre-external-modifier — verified
    # constant across player Strength -8..+21), NOT the modified preview, so
    # they map straight to the engine's `base_damage` / `base_block`. Without
    # them state_from_dict rebuilds from the UNUPGRADED registry spec and every
    # upgraded/enchanted card is planned at its unupgraded value (upgraded
    # Strike 9 -> 6). The C# headless never emits an `upgraded` flag, so `stats`
    # is the ONLY signal that a card is stronger than its registry default
    # (2026-07-15 csview recon audit).
    stats = raw.get("stats")
    if isinstance(stats, dict):
        dmg = stats.get("damage")
        blk = stats.get("block")
        if isinstance(dmg, int):
            spec["base_damage"] = dmg
        if isinstance(blk, int):
            spec["base_block"] = blk

    flags = _flags_for_card(raw) if hand else []
    if flags:
        spec["flags"] = flags

    enchantment = raw.get("enchantment")
    if isinstance(enchantment, str) and enchantment:
        # Bilingual title — keep the English half only.
        spec["enchantment"] = enchantment.split(" / ", 1)[0].strip()
        # The stack size, which is the whole payload for the scaling
        # enchantments (Nimble +N block, Sharp +N damage): forwarding the
        # name alone rebuilt every one of them at one stack. C# omits the
        # field when the amount is 0, and the engine reads absent as one
        # stack — see `ParsedCard::enchantment_amount`.
        amount = raw.get("enchantment_amount")
        if isinstance(amount, int) and not isinstance(amount, bool):
            spec["enchantment_amount"] = amount

    # Per-card affliction (Bound from the Queen's ChainsOfBinding, Tainted
    # from Infested Prism, ...). Two input dialects reach here: the C#
    # headless emits a bilingual localized *title* ("Bound / ..."), the
    # SpireMentor mod emits the bare C# class name ("Bound"). English titles
    # equal the class names (localization_eng/afflictions.json), so one
    # title-keyed map normalizes both onto the wire vocabulary sts2_rs
    # pins. Disambiguation caveat: on STATUS/CURSE cards the same field is
    # the legacy status-identity title ("Burn", consumed above as the
    # wire-id source), so an unknown name there is expected and skipped;
    # on regular cards an unknown name fails loud — soft-dropping is how
    # the move_state/card-id dialect bugs stayed invisible.
    affliction = raw.get("affliction")
    if isinstance(affliction, str) and affliction:
        eng = _normalize_affliction_name(affliction.split(" / ", 1)[0].strip())
        if eng in _AFFLICTION_WIRE_NAMES:
            spec["affliction"] = eng
            stacks = raw.get("affliction_stacks", raw.get("affliction_amount"))
            if isinstance(stacks, int):
                spec["affliction_stacks"] = stacks
        elif spec["kind"] == "card":
            raise UnsupportedStateError(f"{path}: unknown card affliction {affliction!r}")

    # Per-instance card TYPE. Consumed engine-side only by MadScience, whose
    # TinkerTime variant is Attack/Skill/Power *per instance* and therefore
    # cannot be recovered from the registry spec (that spec is the
    # default-constructed Attack build). Dropping it here rebuilt every live
    # Power-mode MadScience as an Attack — wrong damage, wrong cost, and
    # invisible to every card-type-gated surface (CuriousPower's own Power
    # discount, targeting legality). `realgame_adapter._translate_card` already
    # reads `type` into its intermediate dict; this is the leg that was
    # missing (GH #353). `ParsedCard` lowercases it, so pass it verbatim, and
    # unknown spellings are tolerated there rather than rejected.
    card_type = raw.get("type")
    if isinstance(card_type, str) and card_type:
        spec["type"] = card_type

    # MadScience's rider (Curious / Improvement / Expertise). No producer emits
    # this yet — the SpireMentor mod still has to put its public `TinkerTimeRider`
    # `[SavedProperty]` on the bridge Card object (the remaining half of
    # GH #353) — but forwarding it now means that lands as a mod-only change,
    # with `TinkerTimeRider::from_wire_name` already accepting the enum names.
    rider = raw.get("tinker_time_rider")
    if isinstance(rider, str) and rider:
        spec["tinker_time_rider"] = rider

    # Per-instance combat scratch the producer disclosed (the engine's
    # `VarKey` wire names, e.g. Bolas / Thrumming Hatchet's
    # `return_if_played_round`). Integers only; absent otherwise.
    combat_vars = raw.get("combat_vars")
    if isinstance(combat_vars, dict):
        clean = {
            k: v
            for k, v in combat_vars.items()
            if isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
        }
        if clean:
            spec["combat_vars"] = clean
    # Extra plays the instance gained this combat (C# `BaseReplayCount`:
    # Soldier's Stew, Hidden Gem); absent means none.
    replays = raw.get("base_replay_count")
    if isinstance(replays, int) and not isinstance(replays, bool) and replays:
        spec["base_replay_count"] = replays

    return spec


def _normalize_affliction_name(name: str) -> str:
    """Normalize a per-card affliction name onto the wire vocabulary
    (the C# class name, e.g. ``"Tainted"``).

    Three dialects reach the adapter:
      * SpireMentor mod — the bare class name ``"Tainted"`` (pass through).
      * C# headless with a localized entry — the English title, which
        equals the class name (pass through).
      * C# headless WITHOUT a localized entry — an unresolved loc key
        ``"TAINTED.title"`` (``_loc.Bilingual`` echoes the key when
        ``localization_eng/afflictions.json`` lacks it). Tainted/Devoured/
        Weighted currently hit this path. Convert ``"<NAME>.title"`` ->
        ``"Name"`` so the reconstructed deck keeps its afflictions instead
        of soft-dropping the whole card (the Infested Prism un-tainted-deck
        gap).
    """
    if name.endswith(".title"):
        stem = name[: -len(".title")]
        return stem.replace("_", " ").title().replace(" ", "")
    return name


#: Wire vocabulary for per-card afflictions — must match sts2_rs's
#: `card_affliction_kind_from_wire` (state_dict.rs). v0.107.1 ships 7
#: affliction classes; Devoured/Weighted are older-build leftovers the
#: engine still models, kept for headless fixture compatibility.
_AFFLICTION_WIRE_NAMES = frozenset(
    {
        "Bound",
        "Devoured",
        "Entangled",
        "Galvanized",
        "Hexed",
        "Ringing",
        "Smog",
        "Weighted",
        "Tainted",
    }
)


def _translate_orb(raw: dict[str, Any], path: str) -> dict[str, Any]:
    orb_type = _require(raw, "type", path)
    orb_id = ORB_TITLE_TO_ID.get(orb_type)
    if orb_id is None:
        raise UnsupportedStateError(f"{path}: unknown orb type {orb_type!r}")
    return {
        "type": orb_id,
        "passive": int(raw.get("passive", 0)),
        "evoke": int(raw.get("evoke", 0)),
    }


def _translate_osty(raw: dict[str, Any] | None, path: str) -> dict[str, Any] | None:
    """Translate the wire's ``osty`` block.

    A dead Osty is NOT gone: ``DieForYouPower.ShouldCreatureBeRemovedFromCombat
    AfterDeath`` returns false for its own owner (DieForYouPower.cs:40-47), so
    ``PlayerCombatState.OnPetDied`` never removes it from ``Pets``
    (PlayerCombatState.cs:288-297) and both producers ship
    ``alive: false, hp: 0, max_hp: N`` (verbatim live capture:
    ``logs/realgame/20260803T000422Z_19120_2.jsonl``). Osty's ``MaxHp``
    COMPOUNDS across summons and is the Necrobinder's main accumulated
    resource, so the corpse's ``max_hp`` (and its ``block``) must survive the
    round trip -- zeroing it made ``restore_osty`` clamp ``max_hp`` to 1 and the
    next summon rebuild a fresh 1-HP spawn. Only ``hp`` is forced to 0.
    """
    if raw is None:
        return None
    if not raw.get("alive", False):
        return {
            "alive": False,
            "hp": 0,
            "max_hp": int(raw.get("max_hp") or 0),
            "block": int(raw.get("block") or 0),
        }
    return {
        "alive": True,
        "hp": int(_require(raw, "hp", path)),
        "max_hp": int(_require(raw, "max_hp", path)),
        "block": int(raw.get("block", 0)),
    }


def _translate_enemy(
    raw: dict[str, Any], path: str, idx: int, *, is_dead: bool = False
) -> dict[str, Any]:
    name = _require(raw, "name", path)
    monster_id = MONSTER_TITLE_TO_ID.get(name)
    if monster_id is None:
        # Some titles embed a live-resolved localization variable — e.g. the
        # Act-3 TestSubject boss renders as "Test Subject #C8" while the
        # derived map keeps the raw template key "Test Subject #C{Count}".
        # Retry with digit runs folded back into "{Count}".
        monster_id = MONSTER_TITLE_TO_ID.get(re.sub(r"\d+", "{Count}", name))
    if monster_id is None:
        # Plain bare-species title (e.g. "Test Subject", the encounter's own
        # out-of-combat title) rather than the in-combat templated form. See
        # ``_bare_species_title_to_id`` above.
        monster_id = _bare_species_title_to_id().get(name)
    if monster_id is None:
        # Raw loc-key leak: Sts2Headless emits the entry key verbatim when
        # its loc lookup misses (e.g. "DECIMILLIPEDE_SEGMENT_FRONT.name").
        # PascalCase the entry and strip trailing segments until a known
        # engine kind matches (FRONT/MIDDLE/BACK variants all collapse to
        # the one DecimillipedeSegment kind, like their shared title does).
        m = re.fullmatch(r"([A-Z0-9_]+)\.name", str(name))
        if m:
            parts = m.group(1).split("_")
            known = _KNOWN_MONSTER_KINDS
            for k in range(len(parts), 0, -1):
                candidate = "".join(p.capitalize() for p in parts[:k])
                if candidate in known:
                    monster_id = candidate
                    break
    if monster_id is None:
        raise UnsupportedStateError(f"{path}: unknown monster name {name!r}")
    intents_raw = raw.get("intents") or []
    powers_raw = raw.get("powers") or []
    intents = [_translate_intent(it, f"{path}.intents[{i}]") for i, it in enumerate(intents_raw)]
    powers = _soft_translate_list(powers_raw, _translate_power, f"{path}.powers")
    out = {
        "kind": monster_id,
        "uid": int(raw.get("index", idx)) + 1,  # 1-based uid; C# index is 0-based
        "hp": int(_require(raw, "hp", path)),
        "max_hp": int(_require(raw, "max_hp", path)),
        "block": int(raw.get("block", 0)),
        "powers": powers,
        "intents": intents,
        "intent_history": [],  # not surfaced by Sts2Headless
        # Sts2Headless pre-filters its alive list and ships corpses via the
        # separate ``dead_enemies`` field (the kwarg); the live bridge ships
        # its full creature list INLINE with per-enemy flags (forwarded by
        # realgame_adapter as ``is_dead``). Either way, corpses are revivers
        # the game deliberately kept in combat.
        "is_dead": bool(raw.get("is_dead", is_dead)),
        "is_escaped": False,
        # AI move-state id, when the source carries it (the Rust state_to_dict
        # path / recorded runs). state_from_dict rebuilds the enemy AI from it
        # so it keeps acting in rollouts (sts2-rs-3f3rz). The C# backend never
        # emits it → None → enemy stays AI-less, as before.
        "move_state": raw.get("move_state"),
    }
    # Allowlisted private-AI-state disclosures both C# producers emit
    # (2026-08-02 wire-completeness round): Fabricator's last-spawned bot kind
    # (canonical Id.Entry, e.g. "ZAPBOT") and KnowledgeDemon's curse counter
    # (indexes the Disintegration 6/7/8 table). Forwarded verbatim into the
    # engine dict — sts2_rs.state_from_dict tolerates (currently ignores)
    # unknown enemy keys, so this is forward-compat plumbing for the engine's
    # reviver logic to prefer over its max-uid / zero-counter inferences.
    last_spawned = raw.get("fabricator_last_spawned")
    if isinstance(last_spawned, str) and last_spawned:
        out["fabricator_last_spawned"] = last_spawned
    curse_counter = raw.get("curse_of_knowledge_counter")
    if isinstance(curse_counter, int) and not isinstance(curse_counter, bool):
        out["curse_of_knowledge_counter"] = curse_counter
    return out


# ---------------------------------------------------------------------------
# Main translator
# ---------------------------------------------------------------------------


def translate(
    state: dict[str, Any],
    *,
    character: str | None = None,
    combat_id: int = 0,
    seed: int = 0,
    ascension: int = 0,
) -> dict[str, Any]:
    """Translate a Sts2Headless ``combat_play`` JSON state to the
    sts2_rs wire-schema dict.

    Args:
        state: The raw JSON dict (``type == "decision"`` and
            ``decision == "combat_play"``).
        character: Optional explicit character name (``"Ironclad"`` …).
            If not given we infer from card-id prefix.
        combat_id: Optional opaque combat identifier. Sts2Headless
            doesn't surface one; the recommender doesn't read this.
        seed: Optional RNG seed for the rollouts. The hint pipeline
            seeds with ``seed + turn`` per the plan's RNG-divergence
            decision.
        ascension: Ascension level (0 by default — Sts2Headless tracks
            this at run-start, not in combat state).

    Returns:
        A dict matching :class:`sts2_hint.schema.StateDict`.

    Raises:
        UnsupportedStateError: On any unmapped field, with the offending
            JSON path annotated.
    """
    decision = state.get("decision")
    if decision != "combat_play":
        raise UnsupportedStateError(f"$: expected decision='combat_play', got {decision!r}")

    # Mid-resolution state guards — plan §pending_play/choice.
    for key in ("pending_choice", "pending_play", "active_attack"):
        if state.get(key) not in (None, [], {}):
            raise UnsupportedStateError(
                f"$.{key}: mid-resolution state not supported (got {state[key]!r})"
            )

    player_raw = _require(state, "player", "$.player")

    # Resolve player class
    cls: str | None = None
    if character is not None:
        cls = {
            "Ironclad": "ironclad",
            "Silent": "silent",
            "Defect": "defect",
            "Necrobinder": "necrobinder",
            "Regent": "regent",
        }.get(character)
    if cls is None:
        cls = _infer_class_from_cards(state)
    if cls is None:
        raise UnsupportedStateError(
            "$.player.class: could not infer character class from state "
            "(no card id with known character prefix). Pass character= explicitly."
        )

    # Player block
    relics_raw = player_raw.get("relics") or []
    relics = _soft_translate_list(relics_raw, _translate_relic, "$.player.relics")

    # Potions: C# filters null slots; we get a list of dicts. The wire
    # schema is a list[str | None] of fixed-ish length. We emit just
    # what's present.
    potions_raw = player_raw.get("potions") or []
    potions: list[str | None] = []
    for i, p in enumerate(potions_raw):
        if not isinstance(p, dict):
            potions.append(None)
            continue
        potions.append(_translate_potion(p.get("name"), f"$.player.potions[{i}]"))

    player_powers_raw = state.get("player_powers") or []
    p_powers = _soft_translate_list(player_powers_raw, _translate_power, "$.player_powers")

    # Piles
    hand_raw = state.get("hand") or []
    draw_raw = state.get("draw_pile") or []
    discard_raw = state.get("discard_pile") or []
    exhaust_raw = state.get("exhaust_pile") or []
    # The card mid-play. C# parks it in its own pile so it counts toward
    # neither hand nor discard, and moves it to the discard only when the
    # play finishes — which for a card whose OnPlay awaits a selection is
    # after the pick. A board serialized during that screen shows the card
    # in NO pile unless this one travels, and a state rebuilt from it has
    # nothing to flush when the choice resolves, so the played card never
    # reaches the discard at all.
    play_raw = state.get("play_pile") or []
    # ``master_deck`` is optional -- absent (the historical Sts2Headless
    # combat_play shape, and any hand-crafted test fixture that predates
    # this field) must translate to exactly today's ``None`` behaviour, so
    # this is a presence check (``state.get`` -> ``None``), not a
    # falsy/empty check: a caller sending an explicit ``[]`` (e.g. a
    # not-yet-populated run-scope deck) gets an explicit empty list through
    # to ``sts2_rs.state_from_dict``, not "unknown, leave master_deck alone".
    master_deck_raw = state.get("master_deck")

    # Hand cards keep ``hand=True`` for status/flag handling. Non-hand
    # piles share the default translator.
    def _hand_translator(raw: dict[str, Any], path: str) -> dict[str, Any]:
        return _translate_card(raw, path, hand=True)

    hand = _soft_translate_list(hand_raw, _hand_translator, "$.hand")
    draw = _soft_translate_list(draw_raw, _translate_card, "$.draw_pile")
    discard = _soft_translate_list(discard_raw, _translate_card, "$.discard_pile")
    exhaust = _soft_translate_list(exhaust_raw, _translate_card, "$.exhaust_pile")
    play = _soft_translate_list(play_raw, _translate_card, "$.play_pile")
    master_deck: list[dict[str, Any]] | None = None
    if master_deck_raw is not None:
        master_deck = _soft_translate_list(master_deck_raw, _translate_card, "$.master_deck")

    enemies_raw = state.get("enemies") or []
    encounter = [_translate_enemy(e, f"$.enemies[{i}]", i) for i, e in enumerate(enemies_raw)]
    # Dead-but-reviving enemies, APPENDED so the alive entries keep their
    # positions (target_index semantics are alive-relative on both the
    # headless wire and the engine's reconstructed slot order). The engine
    # needs the corpse to exist for its native revive machinery to fire in
    # rollouts (DecimillipedeSegment REATTACH, IllusionPower revive);
    # without it every kill looks permanent to the search.
    dead_raw = state.get("dead_enemies") or []
    n_alive = len(encounter)
    encounter += [
        _translate_enemy(e, f"$.dead_enemies[{i}]", n_alive + i, is_dead=True)
        for i, e in enumerate(dead_raw)
    ]

    # Character-specific
    orbs_raw = state.get("orbs")
    orbs: list[dict[str, Any]] | None = None
    if orbs_raw is not None:
        orbs = [_translate_orb(o, f"$.orbs[{i}]") for i, o in enumerate(orbs_raw)]
    orb_slots = state.get("orb_slots")
    stars = state.get("stars")
    osty = _translate_osty(state.get("osty"), "$.osty")

    ctx = state.get("context") or {}
    is_elite = ctx.get("room_type") == "Elite"

    rnd = int(state.get("round", 1))
    energy = int(state.get("energy", 0))
    max_energy = int(state.get("max_energy", energy))

    player_block = int(player_raw.get("block", 0))
    player_hp = int(_require(player_raw, "hp", "$.player"))
    player_max_hp = int(_require(player_raw, "max_hp", "$.player"))

    player: dict[str, Any] = {
        "class": cls,
        "hp": player_hp,
        "max_hp": player_max_hp,
        "block": player_block,
        "energy": energy,
        "max_energy": max_energy,
        "energy_regen_per_turn": max_energy,  # best-effort
        "relics": relics,
        "powers": p_powers,
        "potions": potions,
    }
    # Combat state that gates relic hooks. SealOfGold spends 5 gold for +1
    # energy at every player turn start; rebuilt with the (former) hardcoded
    # gold=0 that hook could never fire — 90 of the 178 `$.player.energy`
    # divergences on the v27wr sweep were the mirror sitting exactly 1 energy
    # low at end_turn with the seal equipped. Both producers send it
    # (RunSimulator.cs PlayerSummary, SimStateExtractor.cs).
    gold = player_raw.get("gold")
    if isinstance(gold, int) and not isinstance(gold, bool):
        player["gold"] = gold
    # Belt CAPACITY (C# ``Player.MaxPotionCount``). ``potions`` above is not
    # the slot count on every producer (the headless ``PlayerSummary`` drops
    # empty slots), so the capacity rides in its own key: the headless
    # ``potion_slots`` (RunSimulator.cs PlayerSummary) or the live bridge's
    # ``run_state.potion_slot_count`` via ``realgame_adapter``. GH #893.
    potion_slots = player_raw.get("potion_slots")
    if isinstance(potion_slots, int) and not isinstance(potion_slots, bool) and potion_slots > 0:
        player["potion_slots"] = potion_slots
    if orbs is not None:
        player["orbs"] = orbs
        if orb_slots is not None:
            player["orb_slots"] = int(orb_slots)
    if stars is not None:
        player["stars"] = int(stars)
    if osty is not None:
        player["osty"] = osty
    # Per-turn play counters (5 buckets, Rust CardType order:
    # Attack/Skill/Power/Curse/Status). Optional wire field: mid-turn
    # ShouldPlay gates (Ringing one-card-per-turn, Normality, Sloth) read
    # these; absent (older mod builds / start of turn) => engine default
    # of all-zeros.
    cppt = player_raw.get("cards_played_this_turn")
    if isinstance(cppt, list) and len(cppt) == 5:
        player["cards_played_this_turn"] = [max(0, int(n)) for n in cppt]
    # ChainsOfBinding's once-per-turn Bound gate (Queen fight): optional
    # bool the bridge computes from combat history; absent (older mod /
    # headless builds) => engine default false.
    if player_raw.get("bound_card_played_this_turn") is True:
        player["bound_card_played_this_turn"] = True
    # Spite's "lost HP this turn"; absent => false.
    if player_raw.get("hp_lost_this_turn") is True:
        player["hp_lost_this_turn"] = True
    _copy_seat_history(player_raw, player)

    deck: dict[str, Any] = {
        "draw_pile": draw,
        "discard_pile": discard,
        "hand": hand,
        "exhaust_pile": exhaust,
        "master_deck": master_deck,
    }
    # Omitted rather than sent empty: absent means "no card is mid-play",
    # which is what every producer that predates the field also means.
    if play:
        deck["play_pile"] = play

    out: dict[str, Any] = {
        "wire_version": WIRE_VERSION,
        "phase": "PlayerPlayPhase",
        "round": rnd,
        "side": "Player",
        "is_elite": is_elite,
        "player": player,
        "deck": deck,
        "encounter": encounter,
        "turn": rnd,
        "combat_id": int(combat_id),
        "rng": _rng_block(state, seed, ascension),
        "active_attack": None,
        "pending_choice": None,
        "pending_play": None,
    }
    # The game build the board belongs to (GH #800; the live adapter
    # `realgame_adapter.to_headless_combat_play` sets it from the mod's
    # envelope). `state_from_dict` builds the engine for `$.game_version` and
    # defaults to v0.107.1 without it, so it is forwarded only for another
    # build: every v0.107.1 board yields the wire it always did, and the wire
    # agrees with `state_to_dict`, which also omits the default.
    game_version = _wire_game_version(state)
    if game_version is not None:
        out["game_version"] = game_version
    # The act (1-based) and a boss room: co-op enemy scaling, the
    # persistent-gain horizon and act-gated relics read them (GH #989).
    # Forwarded only off their defaults (act 1, not a boss), like
    # `state_to_dict`.
    act = state.get("act")
    if isinstance(act, int) and not isinstance(act, bool) and act > 1:
        out["act_index"] = act
    if ctx.get("room_type") == "Boss":
        out["is_boss"] = True
    # Co-op teammates (``$.allies``): omitted when solo, so a single-player
    # board yields the wire it always did.
    allies = state.get("allies")
    if allies:
        out["allies"] = [_translate_ally(a, i) for i, a in enumerate(allies)]
        # Every teammate came as a whole seat (a current mod): a whole co-op
        # fight, in which they keep drawing and playing every round. Without
        # it they play out only this round and then stand idle, so every line
        # past it is ours alone against enemies scaled for the whole party
        # (GH #1046). One seat short of whole keeps the one-round board.
        if all(_is_whole_seat(a) for a in out["allies"]):
            out["coop_full"] = True
            # Our place in the run's player order: the teammates ahead of us
            # set up, draw and end their turn first.
            if state.get("seat_index"):
                out["coop_seat_index"] = state["seat_index"]
    _resolve_power_appliers(state, out)
    return out


_CLASS_OF_CHARACTER = {
    "Ironclad": "ironclad",
    "Silent": "silent",
    "Defect": "defect",
    "Necrobinder": "necrobinder",
    "Regent": "regent",
}


def _translate_ally(ally: dict[str, Any], index: int) -> dict[str, Any]:
    """One ``$.allies`` entry: HP and block, plus -- when the teammate came
    with a hand -- the whole seat the engine plays out (``sts2_core::coop``)."""
    out: dict[str, Any] = {
        "hp": int(ally["hp"]),
        "max_hp": int(ally["max_hp"]),
        "block": int(ally["block"]),
    }
    if "hand" not in ally:
        return out
    path = f"$.allies[{index}]"

    def _hand_translator(raw: dict[str, Any], card_path: str) -> dict[str, Any]:
        return _translate_card(raw, card_path, hand=True)

    cls = _CLASS_OF_CHARACTER.get(ally.get("character") or "")
    if cls is not None:
        out["class"] = cls
    out["energy"] = int(ally.get("energy") or 0)
    out["max_energy"] = int(ally.get("max_energy") or 0)
    out["powers"] = _soft_translate_list(
        ally.get("powers") or [], _translate_power, f"{path}.powers"
    )
    out["hand"] = _soft_translate_list(ally["hand"], _hand_translator, f"{path}.hand")
    out["ended_turn"] = bool(ally.get("ended_turn"))
    # The rest of the seat, when the producer sent it (a recorded fight to be
    # replayed as a whole co-op fight); omitted otherwise.
    for key in ("draw_pile", "discard_pile", "exhaust_pile", "play_pile"):
        if isinstance(ally.get(key), list):
            out[key] = _soft_translate_list(ally[key], _translate_card, f"{path}.{key}")
    if isinstance(ally.get("relics"), list):
        out["relics"] = [
            _translate_relic(r, f"{path}.relics[{i}]") for i, r in enumerate(ally["relics"])
        ]
    if isinstance(ally.get("stars"), int):
        out["stars"] = ally["stars"]
    if ally.get("hp_lost_this_turn") is True:
        out["hp_lost_this_turn"] = True
    _copy_seat_history(ally, out)
    # Their plays this turn, as ours (GH #989: their Ripple Basin).
    cppt = ally.get("cards_played_this_turn")
    if isinstance(cppt, list) and len(cppt) == 5:
        out["cards_played_this_turn"] = [max(0, int(n)) for n in cppt]
    if isinstance(ally.get("orbs"), list):
        out["orbs"] = [_translate_orb(o, f"{path}.orbs[{i}]") for i, o in enumerate(ally["orbs"])]
        if isinstance(ally.get("orb_slots"), int):
            out["orb_slots"] = ally["orb_slots"]
    # Their Osty, as ours (GH #1017).
    osty = _translate_osty(ally.get("osty"), f"{path}.osty")
    if osty is not None:
        out["osty"] = osty
    if isinstance(ally.get("potions"), list):
        out["potions"] = [
            _translate_potion(p.get("name"), f"{path}.potions[{i}]")
            if isinstance(p, dict)
            else None
            for i, p in enumerate(ally["potions"])
        ]
    return out


def _is_whole_seat(ally: dict[str, Any]) -> bool:
    """A translated ``$.allies`` entry the engine can play for a whole fight:
    a hand, the draw and discard piles it draws and reshuffles from, and the
    relics (an empty list included) that answer its turns."""
    return all(
        isinstance(ally.get(k), list) for k in ("hand", "draw_pile", "discard_pile", "relics")
    )


#: A seat's turn counts a rebuilt board cannot redo (GH #989), on the wire
#: whenever the producer sent them, zero included.
TURN_HISTORY_KEYS = ("cards_exhausted_this_turn", "card_block_gains_this_turn")


def _copy_seat_history(src: dict[str, Any], out: dict[str, Any]) -> None:
    """A seat's History counts a rebuilt board cannot redo (GH #989), when
    the producer sent them: the hits that took the seat's HP this combat
    (Tear Asunder) and its powered hits on each enemy this turn, as
    ``[{enemy, hits}]`` (a teammate's Gang Up), and the cards it exhausted
    this combat (Midnight counts everyone's), and this turn's exhausts (Evil
    Eye) and card block gains (Unmovable). Zero and empty are omitted, the
    wire's default -- except the two turn counts, kept at 0 too: their
    presence says the producer disclosed them, so the live mirror lets
    reality win on them (``combat_mirror``, GH #989)."""
    for key in (
        "unblocked_hits_this_combat",
        "cards_exhausted_this_combat",
        "cards_exhausted_this_turn",
        "card_block_gains_this_turn",
    ):
        n = src.get(key)
        if isinstance(n, int) and not isinstance(n, bool) and (n > 0 or key in TURN_HISTORY_KEYS):
            out[key] = max(0, n)
    hits = src.get("powered_hits_this_turn")
    if isinstance(hits, list):
        rows = [
            {"enemy": int(h["enemy"]), "hits": int(h["hits"])}
            for h in hits
            if isinstance(h, dict) and h.get("hits")
        ]
        if rows:
            out["powered_hits_this_turn"] = rows


def _wire_game_version(state: dict[str, Any]) -> str | None:
    """``$.game_version`` for the wire: the normalized build, or None for the default."""
    raw = state.get("game_version")
    if raw is None:
        return None
    from sts2_client import DEFAULT_GAME_VERSION, normalize_game_version

    if not isinstance(raw, str):
        raise UnsupportedStateError(f"$.game_version: must be a string, got {raw!r}")
    try:
        ver = normalize_game_version(raw)
    except ValueError as exc:
        raise UnsupportedStateError(f"$.game_version: {exc}") from None
    return None if ver == DEFAULT_GAME_VERSION else ver


def _rng_block(state: dict[str, Any], seed: int, ascension: int) -> dict[str, Any]:
    """The wire's ``$.rng``: the search seed, and the RUN's own dice.

    Two unrelated things share this block, and confusing them is expensive.
    ``seed`` is the *search* seed the caller chose for rollouts. ``run_seed``
    and ``counters`` are the game's, forwarded verbatim from the C# producer:
    the once-hashed master seed every stream salts from, plus each stream's
    read count.

    A reconstruction without them has no RNG lineage at all — its ``monster_ai``
    and ``shuffle`` streams come from a default seed and have nothing to do
    with the game's — so every enemy intent and reshuffle order it predicts is
    a guess, and a guess that misses proves nothing about fidelity. With them,
    the engine draws the same card the game is about to draw, and a mismatch
    means the two engines did different things with the *same* roll.

    Omitted when the producer doesn't carry them (older C# builds, hand-built
    fixtures), which leaves the previous carried-cursor behaviour in place.

    Forwarded ALL-OR-NOTHING, because half of it is worse than none:
    :meth:`combat_mirror.CombatMirror.reconcile` reads the presence of
    ``run_seed`` alone as "the world owns the dice" and stops carrying its own
    valid cursors, while ``state_from_dict`` given a seed and no counters
    rebuilds every stream at count 0. A producer that publishes the seed and
    then throws part-way through its counters would therefore not degrade to
    the pre-field behaviour — it would destroy lineage that was intact. The
    emitters build the pair locally and assign it together for the same reason;
    this is the consumer-side half of that contract, so a partial block from
    *any* producer (including a future one) is refused here too.
    """
    out: dict[str, Any] = {"seed": int(seed), "ascension": int(ascension)}
    raw = state.get("rng")
    if not isinstance(raw, dict):
        return out
    run_seed = raw.get("run_seed")
    if not isinstance(run_seed, int) or isinstance(run_seed, bool):
        return out
    counters = raw.get("counters")
    if not isinstance(counters, dict):
        return out
    clean = {
        str(k): int(v)
        for k, v in counters.items()
        if isinstance(v, int) and not isinstance(v, bool)
    }
    if not clean:
        return out
    out["run_seed"] = run_seed
    out["counters"] = clean
    return out


def _infer_class_from_cards(state: dict[str, Any]) -> str | None:
    """Best-effort: scan hand → draw → discard → exhaust for the first
    card whose id prefix matches a known character. Returns the wire
    class literal or None if nothing matches.
    """
    for pile_key in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        for c in state.get(pile_key) or []:
            cid = c.get("id") if isinstance(c, dict) else None
            if not isinstance(cid, str):
                continue
            # Slash-prefix form (e.g. ``IRONCLAD/STRIKE``): take the prefix.
            # Bare/CARD.-prefixed form (e.g. ``CARD.STRIKE_IRONCLAD``): take
            # the trailing ``_CHARACTER`` suffix, which Sts2Headless appends
            # for character-shared basics (STRIKE_IRONCLAD, DEFEND_SILENT,
            # …) and matches the sts2_rs registry directly.
            if "/" in cid:
                prefix = cid.split("/", 1)[0].upper()
            else:
                bare = cid.upper()
                if bare.startswith("CARD."):
                    bare = bare[len("CARD.") :]
                prefix = bare.rsplit("_", 1)[-1] if "_" in bare else ""
            cls = CARD_ID_PREFIX_TO_CLASS.get(prefix)
            if cls is not None:
                return cls
    return None


__all__ = ["translate"]
