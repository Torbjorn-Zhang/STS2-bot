"""Inverse (canonical-id → display) maps for the Rust workbench backend.

The :mod:`sts2_hint.canonical_maps` tables go *display title → registry id*
(used to translate Sts2Headless display names into the ids ``sts2_rs`` wants).
The Rust backend needs the reverse: it gets canonical ids straight off the
``sts2_rs`` engine and must turn them back into the display strings / loc keys
the workbench's server-side enrichment expects.

Two id shapes show up from ``sts2_rs`` and are normalised here:

* Combat hand cards (``state_to_dict``) use ``SCREAMING_SNAKE`` ids
  (``STRIKE_IRONCLAD``); the master deck (``master_deck_detailed``) uses
  ``PascalCase`` (``StrikeIronclad``); relics (``relics()``) carry a
  ``RELIC.`` prefix.
* The localization keys (and card-art slugs) are all ``SCREAMING_SNAKE``
  (``STRIKE_IRONCLAD.title``), so :func:`loc_id` collapses every shape to it.
"""

from __future__ import annotations

import re

from sts2_hint.canonical_maps import (
    MONSTER_TITLE_TO_ID,
    POWER_TITLE_TO_ID,
)

# Registry id -> display title. Built by inverting the *_TITLE_TO_ID maps.
MONSTER_ID_TO_TITLE: dict[str, str] = {v: k for k, v in MONSTER_TITLE_TO_ID.items()}
POWER_ID_TO_TITLE: dict[str, str] = {v: k for k, v in POWER_TITLE_TO_ID.items()}

# Ids whose display title collides with another id are kept OUT of the
# title->id map (a title->id dict can't disambiguate them), so they have no
# inverse entry above and would otherwise fall back to CamelCase spacing. The
# id->title direction has no collision, so add them here explicitly. Without
# this, the three directional Decimillipede segments rendered as "Decimillipede
# Segment Front"/etc. — a name whose slug finds no sprite, so the combatant
# showed as a blank box in the workbench (#188). All three localize to
# "Decimillipede" (DECIMILLIPEDE_SEGMENT.name on the shared base class), whose
# slug resolves to monsters/decimillipede.png. The connected body DOES have
# per-segment Spine rigs (monster_rigs/decimillipede_segment_{front,middle,
# back}), but they are resolved by MonsterKind — not by this shared display
# name — via ``AssetResolver._MONSTER_KIND_RIGS`` (the name alone can't tell the
# segments apart). So the display name maps to the sprite; the rig is picked by
# kind. The static sprite is only the fallback (e.g. a stale bucket manifest).
MONSTER_ID_TO_TITLE.update(
    {
        "DecimillipedeSegmentFront": "Decimillipede",
        "DecimillipedeSegmentMiddle": "Decimillipede",
        "DecimillipedeSegmentBack": "Decimillipede",
    }
)

# Insert an underscore before every uppercase letter preceded by an
# alphanumeric — matching the game's own ``StringHelper.Slugify`` (which splits
# runs of capitals too). The narrower ``(?<=[a-z0-9])`` boundary missed the
# split inside a one-letter "word" + next word, e.g. ``ExpectAFight`` ->
# ``EXPECT_AFIGHT`` (should be ``EXPECT_A_FIGHT``) and ``IAmInvincible`` ->
# ``IAM_INVINCIBLE``, so those cards showed the raw id and found no art.
_PASCAL_BOUNDARY = re.compile(r"(?<=[A-Za-z0-9])(?=[A-Z])")

# A trailing specimen designation carrying an unresolved loc template variable,
# e.g. Test Subject's ``"Test Subject #C{Count}"`` (the game fills ``{Count}``
# from ``SaveManager.Progress.TestSubjectKills + 8`` — cross-run save progress
# the simulator doesn't model, so there's no faithful value to substitute). Left
# verbatim it both shows the raw ``{Count}`` in the UI and breaks art lookup
# (``slugify`` turns it into ``test_subject_c_count``, missing
# ``monsters/test_subject.png``), so it is stripped to the bare species name.
_SPECIMEN_TEMPLATE = re.compile(r"\s*#\S*\{[^}]+\}.*$")
# Safety net for any other unresolved ``{Var}`` token, so a raw template never
# leaks to the UI / art slug again as monsters are ported.
_LOC_TEMPLATE_VAR = re.compile(r"\s*\{[^}]+\}")


def pascal_to_screaming_snake(s: str) -> str:
    """``StrikeIronclad`` -> ``STRIKE_IRONCLAD``; idempotent on snake/all-caps.

    Inserts an underscore before each uppercase letter that follows an
    alphanumeric character (so capital runs split too, e.g. ``ExpectAFight`` ->
    ``EXPECT_A_FIGHT``), then upper-cases — mirroring the game's
    ``StringHelper.Slugify`` that generates the loc keys / art slugs.
    A string that already contains ``_`` (or has no lowercase letters) is
    assumed to be SCREAMING_SNAKE already and is returned upper-cased as-is.
    """
    if "_" in s or not any(c.islower() for c in s):
        return s.upper()
    return _PASCAL_BOUNDARY.sub("_", s).upper()


def loc_id(raw_id: str) -> str:
    """Normalise any ``sts2_rs`` id to the SCREAMING_SNAKE loc-key/art stem.

    Strips a registry prefix (``RELIC.WHETSTONE`` -> ``WHETSTONE``) and
    converts PascalCase to SCREAMING_SNAKE (``StrikeIronclad`` ->
    ``STRIKE_IRONCLAD``).

    A subset of ``sts2_rs`` card ids carry a disambiguating suffix that the
    localization keys and art slugs omit, so it is stripped here so those names
    resolve and their art is found:

    * ``_CARD`` (``CardId::StampedeCard`` -> ``STAMPEDE_CARD``) ->
      ``STAMPEDE.title`` / ``stampede.png``.
    * ``_STATUS`` (``CardId::SlothStatus`` -> ``SLOTH_STATUS``) ->
      ``SLOTH.title`` / ``sloth.png``. The C# card class is plain ``Sloth``;
      the Rust port appends ``_STATUS`` to disambiguate from ``SlothPower`` in
      its shared id space, but the game's loc key / art stem is just ``SLOTH``.

    ``SLOTH_STATUS`` is the only ``_STATUS``-suffixed card id and no loc key
    ends in ``_STATUS``, so the strip is safe for cards. It is NOT applied to a
    ``RELIC.``-prefixed id: the Membership Card relic's loc key really is
    ``MEMBERSHIP_CARD.title``, and stripping it to ``MEMBERSHIP`` left the name
    unresolved and made the act-2/3 parity probe report a phantom relic
    mismatch against C#'s ``MEMBERSHIPCARD``.
    """
    prefix, bare = raw_id.split(".", 1) if "." in raw_id else ("", raw_id)
    stem = pascal_to_screaming_snake(bare)
    if prefix.upper() == "RELIC":
        return stem
    for suffix in ("_CARD", "_STATUS"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def power_title(power_id: str) -> str:
    """Display name for a power id (``STRENGTH`` -> ``Strength``).

    Falls back to a title-cased form of the id when the power isn't in the
    curated localization map.
    """
    pid = power_id[:-6] if power_id.endswith("_POWER") else power_id
    return POWER_ID_TO_TITLE.get(pid) or pid.replace("_", " ").title()


def monster_title(monster_id: str) -> str:
    """Display name for a monster kind (``Nibbit`` -> ``Nibbit``).

    Falls back to spacing the CamelCase id when the monster isn't curated.

    Localization templates carry the names verbatim, so a title may hold an
    unresolved loc variable (Test Subject's ``"Test Subject #C{Count}"``). The
    workbench has no value to fill it with (see :data:`_SPECIMEN_TEMPLATE`), so
    such templates are stripped to the bare species name (``"Test Subject"``) —
    which also lets the slug-based art lookup find the sprite/rig.
    """
    if monster_id in MONSTER_ID_TO_TITLE:
        title = MONSTER_ID_TO_TITLE[monster_id]
        title = _SPECIMEN_TEMPLATE.sub("", title)
        return _LOC_TEMPLATE_VAR.sub("", title).strip()
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", monster_id)
