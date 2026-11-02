"""Shim for the game-version helpers of spire-mentor's sts2_client (sts2_hint.adapter imports
them); same values and behaviour as the original (sts2-pycli/python/sts2_client.py)."""
DEFAULT_GAME_VERSION = "v0.107.1"
SUPPORTED_GAME_VERSIONS = ("v0.107.1", "v0.111.0")


def normalize_game_version(version):
    v = version.strip()
    if v and not v.startswith("v"):
        v = "v" + v
    if v not in SUPPORTED_GAME_VERSIONS:
        raise ValueError(f"unsupported game version {version!r} (supported: {', '.join(SUPPORTED_GAME_VERSIONS)})")
    return v
