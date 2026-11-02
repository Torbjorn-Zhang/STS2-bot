"""Canonical name → id tables for the ``sts2_hint`` adapter.

Bead: ``sts2-rs-lv9d`` (hint-T7).

The adapter translates Sts2Headless's *display names* (because the C#
JSON does not surface canonical ids — see
``docs/sts2headless-combat-state-fields.md`` Not-Found section) into the
SCREAMING_SNAKE_CASE / CamelCase ids that ``sts2_rs`` expects on the
wire.

Source of truth for the display strings is the English localization
under ``localization_eng/`` shipped with sts2-pycli. The maps are built
at import time by:

1. Parsing the relevant ``*.json`` localization file for keys of the
   form ``{REGISTRY_ID}.title`` (or ``.name`` for monsters).
2. Inverting to ``{display_title -> REGISTRY_ID}``.
3. Verifying every ``REGISTRY_ID`` against
   ``sts2_rs.list_registry_names(kind)`` and dropping any pair whose
   id isn't recognised by the Rust engine — so we never hand
   ``sts2_rs.state_from_dict`` something it will reject. (Drops are
   logged at INFO; missing ``sts2_rs`` falls back to "keep everything"
   and logs a WARNING — matches the T6 contract.)

Monsters are special: their registry ids are CamelCase
(``Nibbit``, ``ShrinkerBeetle``) but their localization keys are
SCREAMING_SNAKE_CASE (``NIBBIT.name``). We hand-curate the mapping
from registry-id to display-title for the small set of monsters
sts2-rs actually models (see ``MONSTER_TITLE_TO_ID`` below). New
monsters land as fail-loud entries until they're added here.

Orbs are similarly hand-curated (5 values; trivially small).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Final

import runtime_paths

from sts2_hint.errors import get_logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# runtime_paths keeps this working in the frozen advisor exe, where the
# localization files are PyInstaller datas rather than a checkout sibling.
_LOCALIZATION_DIR: Final[Path] = runtime_paths.data_root() / "localization_eng"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, str]:
    """Load a localization JSON file. Returns ``{}`` if missing — the
    adapter degrades to fail-loud at translate time rather than at
    import time.

    The packaged advisor ships these as ``<name>.json.gz`` (the game reads
    every ``.json`` under a mod's folder as a mod manifest), so a missing
    ``.json`` falls back to its gzipped copy."""
    gz = path.with_name(path.name + ".gz")
    try:
        if not path.is_file() and gz.is_file():
            with gzip.open(gz, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        if not isinstance(data, dict):
            return {}
        # Values may be lists/dicts in some files; coerce to str-only.
        return {k: v for k, v in data.items() if isinstance(v, str)}
    except (OSError, json.JSONDecodeError) as e:
        get_logger().warning("canonical_maps: could not load %s: %s", path, e)
        return {}


def _harvest_title_map(loc: dict[str, str], suffix: str = ".title") -> dict[str, str]:
    """Invert ``{ID}{suffix} -> "Display Title"`` to ``"Display Title" -> ID``.

    On display-title collisions we keep the *first* registration and
    log a warning — Sts2Headless never sends back a colliding name
    because the engine only loads one of them per combat slot, but the
    log surfaces the ambiguity for the hint side to react to if it
    ever matters.
    """
    out: dict[str, str] = {}
    log = get_logger()
    for key, value in loc.items():
        if not key.endswith(suffix):
            continue
        reg_id = key[: -len(suffix)]
        if value in out and out[value] != reg_id:
            log.info(
                "canonical_maps: display-title collision %r → %r vs %r; keeping first",
                value,
                out[value],
                reg_id,
            )
            continue
        out[value] = reg_id
    return out


def _strip_power_suffix(title_to_id: dict[str, str]) -> dict[str, str]:
    """Powers in the registry are bare (``STRENGTH``, ``AFTERIMAGE``) but
    the localization keys have a ``_POWER`` suffix (``STRENGTH_POWER``).
    Strip the suffix from the values. If two power-localization-ids
    collapse to the same bare id we keep the first and log."""
    out: dict[str, str] = {}
    log = get_logger()
    for title, pid in title_to_id.items():
        bare = pid[:-6] if pid.endswith("_POWER") else pid
        if title in out and out[title] != bare:
            log.info(
                "canonical_maps: power title collision %r → %r vs %r; keeping first",
                title,
                out[title],
                bare,
            )
            continue
        out[title] = bare
    return out


# ---------------------------------------------------------------------------
# Build raw maps from localization
# ---------------------------------------------------------------------------

_RELIC_LOC = _load_json(_LOCALIZATION_DIR / "relics.json")
_POTION_LOC = _load_json(_LOCALIZATION_DIR / "potions.json")
_POWER_LOC = _load_json(_LOCALIZATION_DIR / "powers.json")
_CARD_LOC = _load_json(_LOCALIZATION_DIR / "cards.json")

RELIC_TITLE_TO_ID: dict[str, str] = _harvest_title_map(_RELIC_LOC)
POTION_TITLE_TO_ID: dict[str, str] = _harvest_title_map(_POTION_LOC)
POWER_TITLE_TO_ID: dict[str, str] = _strip_power_suffix(_harvest_title_map(_POWER_LOC))
CARD_TITLE_TO_ID: dict[str, str] = _harvest_title_map(_CARD_LOC)


def _augment_power_titles_from_registry(title_to_id: dict[str, str]) -> None:
    """Fill in engine powers whose display title has no ``powers.json`` entry.

    ``POWER_TITLE_TO_ID`` is harvested from localization, but ~36 real engine
    powers — the TemporaryStrength family (``TEMPORARY_STRENGTH``, ``…_FOCUS``,
    ``…_DEXTERITY``, ``MANGLE_POWER`` …), ``METALLICIZE``, ``ENTANGLED``,
    ``DARK_SHACKLES`` … — have no loc entry. Their title then resolves to
    ``None`` and :func:`sts2_hint.adapter._translate_power` silently dropped
    them, stranding e.g. a *temporary* −Strength debuff as permanent and
    defanging the enemy in the dict-reconstructed search (the Entomancer "MCTS
    suggests end turn" bug).

    The Rust backend now carries the engine power ``id`` directly, which covers
    the live-hint path; this closes the same gap for the **C# backend** and for
    states **recorded before the id field existed**, where only the display
    title survives. The title used here is the engine's title-cased fallback
    (``power_id`` stem → ``Title Case``) — exactly what ``display_maps.power_title``
    emits for a power absent from loc, so it matches those recorded titles.

    Registry-sourced (best-effort: a no-op when ``sts2_rs`` is unavailable, e.g.
    a build-less CI) so it can't silently drift as content is ported. Purely
    additive — a loc-derived title is never overwritten. When a base power and
    its ``…Power`` variant share a fallback title (``MANGLE``/``MANGLE_POWER``,
    ``GRAPPLE``/``GRAPPLE_POWER``, ``FREE``/``FREE_POWER``), the
    creature-applied ``_POWER`` variant wins — a title→id map structurally
    cannot represent both, and only the ``_POWER`` variant lands on creatures.
    """
    try:
        import sts2_rs  # local import: adapter/canonical_maps stay importable build-less
    except Exception:  # pragma: no cover - exercised only in build-less envs
        return
    try:
        names = list(sts2_rs.list_registry_names("power"))
    except Exception:  # pragma: no cover - defensive against binding drift
        return
    loc_titles = frozenset(title_to_id)
    for pname in names:
        stem = pname[:-6] if pname.endswith("_POWER") else pname
        title = stem.replace("_", " ").title()
        if title in loc_titles:
            continue  # localization is authoritative; never override it
        existing = title_to_id.get(title)
        if existing is None or (pname.endswith("_POWER") and not existing.endswith("_POWER")):
            title_to_id[title] = pname


_augment_power_titles_from_registry(POWER_TITLE_TO_ID)

# Hand-curated maps for things whose registry ids don't trivially match
# the localization key shape.

#: Defect orb localization-id stem (``DARK_ORB.title``) → registry id.
ORB_TITLE_TO_ID: Final[dict[str, str]] = {
    "Dark": "DARK",
    "Frost": "FROST",
    "Glass": "GLASS",
    "Lightning": "LIGHTNING",
    "Plasma": "PLASMA",
    "Orb Slot": "EMPTY",  # Empty slot's display name
}

#: Monster display-name → CamelCase registry id, covering every monster the
#: ``sts2_rs`` engine models. Single-sourced (not hand-curated): generated from
#: ``sts2_rs.list_registry_names("monster")`` × the display names in
#: ``localization_eng/monsters.json``, and pinned by
#: ``tests/workbench/test_monster_map_sync.py`` — which fails loud if the engine
#: grows a monster that isn't represented here, so the map can't silently drift
#: again (it did: it covered 30 of 121 until 2026-06-20).
#:
#: Deliberately EXCLUDED (kept in sync with the test's allow-list):
#:   * synthetic test monsters with no localization entry —
#:     ``OneHpMonster``, ``TenHpMonster``, ``SingleAttackMoveMonster``,
#:     ``MultiAttackMoveMonster``;
#:   * ids whose display name COLLIDES with another id, which a title→id map
#:     structurally cannot disambiguate — ``DecimillipedeSegment{Back,Front,
#:     Middle}`` (all show as "Decimillipede"; the base ``DecimillipedeSegment``
#:     is mapped) and ``DenseVegetationWriggler`` (shows as "Wriggler", already
#:     bound to ``Wriggler``).
#: To regenerate after porting monsters, re-run the snippet in the sync test.
MONSTER_TITLE_TO_ID: Final[dict[str, str]] = {
    "Assassin Raider": "AssassinRubyRaider",
    "Axe Raider": "AxeRubyRaider",
    "Axebot": "Axebot",
    "Battle Friend V1.0": "BattleFriendV1",
    "Battle Friend V2.0": "BattleFriendV2",
    "Battle Friend V3.0": "BattleFriendV3",
    "Big Dummy": "BigDummy",
    "Bowlbug (Egg)": "BowlbugEgg",
    "Bowlbug (Nectar)": "BowlbugNectar",
    "Bowlbug (Rock)": "BowlbugRock",
    "Bowlbug (Silk)": "BowlbugSilk",
    "Brute Raider": "BruteRubyRaider",
    "Bygone Effigy": "BygoneEffigy",
    "Byrdonis": "Byrdonis",
    "Byrdpip": "Byrdpip",
    "Calcified Cultist": "CalcifiedCultist",
    "Ceremonial Beast": "CeremonialBeast",
    "Chomper": "Chomper",
    "Corpse Slug": "CorpseSlug",
    "Crossbow Raider": "CrossbowRubyRaider",
    "Crusher": "Crusher",
    "Cubex Construct": "CubexConstruct",
    "Damp Cultist": "DampCultist",
    "Decimillipede": "DecimillipedeSegment",
    "Aeonglass": "Aeonglass",
    "Devoted Sculptor": "DevotedSculptor",
    "Entomancer": "Entomancer",
    "Exoskeleton": "Exoskeleton",
    "Eye with Teeth": "EyeWithTeeth",
    "Fabricator": "Fabricator",
    "Fat Gremlin": "FatGremlin",
    "Flail Knight": "FlailKnight",
    "Flyconid": "Flyconid",
    "Fogmog": "Fogmog",
    "Fossil Stalker": "FossilStalker",
    "Frog Knight": "FrogKnight",
    "Fuzzy Wurm Crawler": "FuzzyWurmCrawler",
    "Gas Bomb": "GasBomb",
    "Globe Head": "GlobeHead",
    "Gremlin Merc": "GremlinMerc",
    "Guardbot": "Guardbot",
    "Haunted Ship": "HauntedShip",
    "Hunter Killer": "HunterKiller",
    "Infested Prism": "InfestedPrism",
    "Inklet": "Inklet",
    "Kin Follower": "KinFollower",
    "Kin Priest": "KinPriest",
    "Knowledge Demon": "KnowledgeDemon",
    "Lagavulin Matriarch": "LagavulinMatriarch",
    "Leaf Slime (M)": "LeafSlimeM",
    "Leaf Slime (S)": "LeafSlimeS",
    "Living Fog": "LivingFog",
    "Living Shield": "LivingShield",
    "Louse Progenitor": "LouseProgenitor",
    "Magi Knight": "MagiKnight",
    "Mawler": "Mawler",
    "Mecha Knight": "MechaKnight",
    "Mysterious Knight": "MysteriousKnight",
    "Myte": "Myte",
    "Nibbit": "Nibbit",
    "Noisebot": "Noisebot",
    "Osty": "Osty",
    "Ovicopter": "Ovicopter",
    "Owl Magistrate": "OwlMagistrate",
    "Pael's Legion": "PaelsLegion",
    "Parafright": "Parafright",
    "Phantasmal Gardener": "PhantasmalGardener",
    "Phrog Parasite": "PhrogParasite",
    "Punch Construct": "PunchConstruct",
    "Queen": "Queen",
    "Rocket": "Rocket",
    "Scroll of Biting": "ScrollOfBiting",
    "Seapunk": "Seapunk",
    "Sewer Clam": "SewerClam",
    "Shrinker Beetle": "ShrinkerBeetle",
    "Skulking Colony": "SkulkingColony",
    "Slimed Berserker": "SlimedBerserker",
    "Slithering Strangler": "SlitheringStrangler",
    "Sludge Spinner": "SludgeSpinner",
    "Slumbering Beetle": "SlumberingBeetle",
    "Snapping Jaxfruit": "SnappingJaxfruit",
    "Sneaky Gremlin": "SneakyGremlin",
    "Soul Fysh": "SoulFysh",
    "Soul Nexus": "SoulNexus",
    "Spectral Knight": "SpectralKnight",
    "Spiny Toad": "SpinyToad",
    "Stabbot": "Stabbot",
    "Terror Eel": "TerrorEel",
    "Test Subject #C{Count}": "TestSubject",
    "The Adversary Mk 1": "TheAdversaryMkOne",
    "The Adversary Mk 2": "TheAdversaryMkTwo",
    "The Adversary Mk 3": "TheAdversaryMkThree",
    "The Architect": "Architect",
    "The Forgotten": "TheForgotten",
    "The Insatiable": "TheInsatiable",
    "The Lost": "TheLost",
    "The Merchant???": "FakeMerchantMonster",
    "The Obscura": "TheObscura",
    "Thieving Hopper": "ThievingHopper",
    "Toadpole": "Toadpole",
    "Torch Head Amalgam": "TorchHeadAmalgam",
    "Tough Egg": "ToughEgg",
    "Tracker Raider": "TrackerRubyRaider",
    "Tunneler": "Tunneler",
    "Turret Operator": "TurretOperator",
    "Twig Slime (M)": "TwigSlimeM",
    "Twig Slime (S)": "TwigSlimeS",
    "Two-Tailed Rat": "TwoTailedRat",
    "Vantom": "Vantom",
    "Vine Shambler": "VineShambler",
    "Waterfall Giant": "WaterfallGiant",
    "Wriggler": "Wriggler",
    "Zapbot": "Zapbot",
}

#: Extra runtime-only display titles that the standard
#: ``registry-id -> SCREAMING_SNAKE -> "<ID>.name"`` derivation above can't
#: produce, because the monster's C# ``Title`` getter switches to a
#: *different* localization key under some in-combat state. Merged into
#: ``MONSTER_TITLE_TO_ID`` below (same id, an additional title) rather than
#: folded into the derived table so ``test_monster_map_sync.py`` can
#: document each one explicitly instead of silently tolerating extra keys.
#:
#: * ``ToughEgg`` overrides ``Title`` to ``HATCHLING.name`` ("Hatchling")
#:   once its ``HATCH`` move fires (``ToughEgg.cs`` in the v0.107.1 decomp:
#:   ``if (!_hatched) return L10N(Id.Entry + ".name"); return
#:   L10N("HATCHLING.name");``) -- the live bridge sends that post-hatch
#:   name, not "Tough Egg", for the rest of the fight.
MONSTER_TITLE_ALIASES: Final[dict[str, str]] = {
    "Hatchling": "ToughEgg",
}
MONSTER_TITLE_TO_ID.update(MONSTER_TITLE_ALIASES)

#: Status / curse card affliction titles → synthetic ``status:<Name>``
#: ids. The Rust side parses ``status:Burn`` directly; canonical
#: status-card "registry" entries also exist for the ones that are
#: real cards (``BURN``, ``WOUND``, ``SLIMED``, ``DAZED``, etc.) but
#: the synthetic form is the documented wire shape from the plan.
STATUS_CARD_TITLES: Final[set[str]] = {
    "Burn",
    "Wound",
    "Slimed",
    "Dazed",
    "Void",
    "Shame",
    "Pain",
    "Decay",
    "Doubt",
    "Injury",
    "Normality",
    "Parasite",
    "Regret",
    "Writhe",
    "Clumsy",
    "Curse of the Bell",
    "Necronomicurse",
    "Pride",
}

#: Intent type display names (already canonical strings emitted by C#).
#: These pass through unchanged to ``IntentSpec.kind``.
INTENT_TYPES: Final[set[str]] = {
    "Attack",
    "Defend",
    "Buff",
    "Heal",
    "Debuff",
    "DebuffStrong",
    "CardDebuff",
    "StatusCard",
    "DeathBlow",
    "Escape",
    "Summon",
    "Sleep",
    "Stun",
    "Hidden",
    "Unknown",
}

#: Sts2Headless character names (from ``start_run``) → wire-schema
#: ``player.class`` literals.
CHARACTER_TO_CLASS: Final[dict[str, str]] = {
    "Ironclad": "ironclad",
    "Silent": "silent",
    "Defect": "defect",
    "Necrobinder": "necrobinder",
    "Regent": "regent",
}

#: Card ``id`` prefix (the part before ``/`` in ``c.Id.ToString()``) →
#: wire-schema class literal. Used to recover ``player.class`` when the
#: adapter caller doesn't pass character explicitly.
CARD_ID_PREFIX_TO_CLASS: Final[dict[str, str]] = {
    "IRONCLAD": "ironclad",
    "SILENT": "silent",
    "DEFECT": "defect",
    "NECROBINDER": "necrobinder",
    "REGENT": "regent",
}

# ---------------------------------------------------------------------------
# Optional registry cross-check (drops unknown ids)
# ---------------------------------------------------------------------------


def _prune_against_registry() -> None:
    """For each populated map, drop entries whose registry id isn't in
    the corresponding ``sts2_rs.list_registry_names(kind)`` set.

    Behaviour matches the T6 spec:

    * ``sts2_rs`` not importable → log WARNING, keep everything (the
      adapter will fail loud at translate time anyway).
    * ``list_registry_names`` missing → log WARNING, keep everything.
    * Individual id missing → DROP it from the in-memory map and log
      INFO. Rationale: if the Rust side doesn't know an id, the only
      thing handing it through can do is provoke an
      ``UnsupportedStateError`` deeper in the pipeline; pruning here
      gives the adapter a chance to fail loud with a clear
      ``"<path>: unmapped name"`` message instead.

    NOTE: ``CARD_TITLE_TO_ID`` is *not* pruned by display title alone
    — cards are resolved by C# id (``IRONCLAD/STRIKE``) via the
    adapter's :func:`adapter.resolve_card_id` helper, which performs
    its own registry lookup. Keeping the localization-derived map
    around lets us fall back to title-based lookup for content
    sts2-rs doesn't yet model (where we'd want a clear "title not
    in registry" diagnostic).
    """
    log = get_logger()
    try:
        import sts2_rs  # type: ignore[import-not-found]
    except ImportError:
        log.warning(
            "sts2_rs extension not importable; skipping canonical-map "
            "verification. Build it with `maturin develop` to enable."
        )
        return

    list_names = getattr(sts2_rs, "list_registry_names", None)
    if list_names is None:
        log.warning(
            "sts2_rs.list_registry_names not found; this build predates "
            "bead sts2-rs-92sk. Skipping canonical-map verification."
        )
        return

    pairs: list[tuple[str, dict[str, str]]] = [
        ("relic", RELIC_TITLE_TO_ID),
        ("potion", POTION_TITLE_TO_ID),
        ("power", POWER_TITLE_TO_ID),
        ("monster", dict(MONSTER_TITLE_TO_ID)),  # frozen — verify only
        ("orb", dict(ORB_TITLE_TO_ID)),
        ("card", CARD_TITLE_TO_ID),
    ]
    for kind, table in pairs:
        try:
            known: set[str] = set(list_names(kind))
        except Exception as e:
            log.warning("sts2_rs.list_registry_names(%r) failed: %s", kind, e)
            continue
        to_drop = [src for src, dst in table.items() if dst not in known]
        for src in to_drop:
            log.info(
                "canonical_maps: dropping %s %r → %r (not in registry)",
                kind,
                src,
                table[src],
            )
            # Mutating frozen-typed dicts is intentional here — Final on
            # the binding, not on the dict contents. Same as upstream
            # ``RELIC_NAME_TO_ID`` in the T6 skeleton.
            if kind == "relic":
                RELIC_TITLE_TO_ID.pop(src, None)
            elif kind == "potion":
                POTION_TITLE_TO_ID.pop(src, None)
            elif kind == "power":
                POWER_TITLE_TO_ID.pop(src, None)
            elif kind == "card":
                CARD_TITLE_TO_ID.pop(src, None)
            # monster + orb maps stay as their hand-curated Final
            # binding — drops there are advisory only.


_prune_against_registry()


# ---------------------------------------------------------------------------
# Backwards-compat aliases for the T6 names. These will be removed
# once any consumers migrate to the *_TITLE_TO_ID names.
# ---------------------------------------------------------------------------

RELIC_NAME_TO_ID: Final[dict[str, str]] = RELIC_TITLE_TO_ID
CARD_NAME_TO_ID: Final[dict[str, str]] = CARD_TITLE_TO_ID
MONSTER_NAME_TO_ID: Final[dict[str, str]] = dict(MONSTER_TITLE_TO_ID)
POWER_NAME_TO_ID: Final[dict[str, str]] = POWER_TITLE_TO_ID
POTION_NAME_TO_ID: Final[dict[str, str]] = POTION_TITLE_TO_ID
ORB_NAME_TO_ID: Final[dict[str, str]] = dict(ORB_TITLE_TO_ID)


__all__ = [
    "CARD_ID_PREFIX_TO_CLASS",
    "CARD_NAME_TO_ID",
    "CARD_TITLE_TO_ID",
    "CHARACTER_TO_CLASS",
    "INTENT_TYPES",
    "MONSTER_NAME_TO_ID",
    "MONSTER_TITLE_ALIASES",
    "MONSTER_TITLE_TO_ID",
    "ORB_NAME_TO_ID",
    "ORB_TITLE_TO_ID",
    "POTION_NAME_TO_ID",
    "POTION_TITLE_TO_ID",
    "POWER_NAME_TO_ID",
    "POWER_TITLE_TO_ID",
    "RELIC_NAME_TO_ID",
    "RELIC_TITLE_TO_ID",
    "STATUS_CARD_TITLES",
]
