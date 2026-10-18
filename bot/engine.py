"""Shared engine client: reuses harness/simclient.py (one long-lived engine process)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

from simclient import Sim, legal_actions  # noqa: E402,F401

# Restoring many times leaks game objects inside the engine process, so recycle it once its
# working set passes this many MB (a fresh process costs ~0.7 s). Keeps memory bounded.
RECYCLE_MB = int(os.environ.get("STS2_RECYCLE_MB", "1500"))


# build configuration of the engine (bin/<config>/net9.0): diagnostics that need engine changes
# build and use their own (STS2_ENGINE_CONFIG=Diag) while jobs keep running on Release
ENGINE_CONFIG = os.environ.get("STS2_ENGINE_CONFIG", "Release")


def fresh_sim():
    return Sim(ENGINE_CONFIG, timeout=60)


def maybe_recycle(sim):
    """Call between runs. Returns the same sim, or a fresh one if the old one grew too big."""
    try:
        ws = sim.send({"cmd": "gc_stats"}).get("working_set_mb") or 0
    except Exception:
        ws = RECYCLE_MB + 1
    if ws <= RECYCLE_MB:
        return sim
    sim.close()
    return fresh_sim()
