"""Shared engine client: reuses harness/simclient.py (one long-lived engine process)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))

from simclient import Sim, legal_actions  # noqa: E402,F401
