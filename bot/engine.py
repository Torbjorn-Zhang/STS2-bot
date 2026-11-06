"""Shared engine client: reuses harness/simclient.py (one long-lived engine process)."""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

from simclient import Sim, legal_actions  # noqa: E402,F401

# Restoring many times leaks game objects inside the engine process, so recycle it once its
# working set passes this many MB (a fresh process costs ~0.7 s). Keeps memory bounded.
RECYCLE_MB = int(os.environ.get("STS2_RECYCLE_MB", "1500"))


# build configuration of the engine (bin/<config>/net9.0): diagnostics that need engine changes
# build and use their own (STS2_ENGINE_CONFIG=Diag) while jobs keep running on Release
ENGINE_CONFIG = os.environ.get("STS2_ENGINE_CONFIG", "Release")


# "SIMS,BOSS_SIMS": play on sts2-rs instead (rssim.RsSim; its MCTS plays the fights). run.py --rs sets it.
RS = os.environ.get("STS2_RS")


# run.py --run-timeout: a wall-clock deadline for the run each worker thread is playing. Every
# engine call checks it, so a run that never ends (11-04: seed up83_2400 kept a chunk busy ~13 h,
# both processes busy, no output) raises RunTimeout and the worker records it and moves on.
_DEADLINE = threading.local()


class RunTimeout(Exception):
    pass


def set_deadline(seconds):
    """This thread's engine calls raise RunTimeout once `seconds` from now have passed (None or 0: never)."""
    _DEADLINE.t = time.time() + seconds if seconds else None


class _Guarded:
    def send(self, cmd):
        t = getattr(_DEADLINE, "t", None)
        if t is not None and time.time() > t:
            raise RunTimeout("run wall-clock limit reached")
        r = super().send(cmd)
        tr = getattr(self, "trace", None)   # run.py --repro: every reply's game-state signature
        if tr is not None and isinstance(r, dict) and r.get("decision") is not None:
            p, ctx = r.get("player") or {}, r.get("context") or {}
            tr.append([cmd.get("action") or cmd.get("cmd"), ctx.get("act"), ctx.get("floor"), r.get("decision"),
                       p.get("hp"), p.get("gold"), len(p.get("deck") or []), r.get("round"),
                       sum((e.get("hp") or 0) for e in r.get("enemies") or [])])
        return r


class GuardedSim(_Guarded, Sim):
    pass


_RS_SIM = None


def fresh_sim():
    global _RS_SIM
    if RS:
        if _RS_SIM is None:
            from rssim import RsSim
            _RS_SIM = type("GuardedRsSim", (_Guarded, RsSim), {})
        sims, boss = (int(x) for x in str(RS).split(","))
        return _RS_SIM(sims=sims, boss_sims=boss)
    return GuardedSim(ENGINE_CONFIG, timeout=60)


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
