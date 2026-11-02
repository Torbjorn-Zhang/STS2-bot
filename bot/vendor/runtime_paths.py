"""Shim for spire-mentor's runtime_paths (sts2_hint imports it): data_root() is our sts2-cli
checkout, whose localization_eng/ is the v0.111.0 game's English localization."""
from pathlib import Path

_SCLI = Path(__file__).resolve().parent.parent.parent / "sts2-cli"


def is_frozen() -> bool:
    return False


def data_root() -> Path:
    return _SCLI


def app_data_dir() -> Path:
    return Path(__file__).resolve().parent / "sts2_hint"
