"""Map the search's raw ``p_win`` to the chance of winning the fight.

The search's ``p_win`` is the share of its playouts that win, and its
playouts are played by the playout net, which plays far worse than the
search does: a raw 5-10% was a fight won about half the time. The ranking
is sound (higher is likelier), so a monotone map fixes the number shown
without touching the choice. Platt scaling, fitted on fights played to the
end by the advisor's own search (``realgame_advisor``: 4000 sims, 16000 for
bosses, BeliefDraw): ``docs/experiments/2026-10-30-pwin-calibration.md``.
Refit when the search's budget, playouts or knobs change.
"""

from __future__ import annotations

import math

#: ``p_cal = sigmoid(INTERCEPT + SLOPE * logit(p_raw))``.
INTERCEPT = 2.4748
SLOPE = 0.9319

#: Raw values are clipped here before the logit: the search reports exact
#: 0 and 1 (every playout lost or won), which no map should pass through as
#: certainty it did not measure.
_EPS = 1e-3


def calibrated_p_win(raw: float) -> float:
    """The fight-win chance for a raw search ``p_win``.

    Args:
        raw: The search's ``p_win`` in ``[0, 1]``.

    Returns:
        The calibrated chance in ``(0, 1)``, increasing in ``raw``.
    """
    p = min(max(raw, _EPS), 1 - _EPS)
    z = INTERCEPT + SLOPE * math.log(p / (1 - p))
    return 1 / (1 + math.exp(-z))
