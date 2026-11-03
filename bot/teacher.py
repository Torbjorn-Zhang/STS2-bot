"""A teacher for the deck decisions: spire-mentor's PPO run policy scores the options of a decision
our bot faces on sts2-rs (run.py --rs ... --teacher MODEL), for imitation (upolicy.py train
--teacher-only). DeckNets asks it before its own networks: the teacher's best option is played and
recorded (x.teach = 1, x.tp = its probabilities over our options), so the records are the teacher's
decisions on the states our bot reaches.

Our options map onto the teacher's actions through RsSim.resolve (the sts2-rs action each of ours
takes). Rest-site smith options share one sts2-rs action (rest_option smith); their card comes
from the teacher's smith pick on a copy of the run after it.
"""

import threading

from rssim import card_name, sts2_rs

_LOCAL = threading.local()


class PpoTeacher:
    def __init__(self, model, features="card_mods,reward_upgrade", play=1.0):
        # play: probability its pick is played at a decision it labels (< 1: DAgger, our pick
        # plays otherwise and the record keeps its label, x.tplay = 0)
        self.model, self.features, self.play = model, features, play

    def _agent(self):
        a = getattr(_LOCAL, "agent", None)
        if a is None or getattr(_LOCAL, "model", None) != self.model:
            a = sts2_rs.PpoRunAgent(seed=0, sims=1, boss_sims=1, model_path=self.model, features=self.features or None)
            _LOCAL.agent, _LOCAL.model = a, self.model
        return a

    def dist(self, rm):
        """{action key: prob} of the teacher at rm's decision, or None (combat, one option)."""
        res = self._agent().evaluate_with_actions(rm)
        if res is None:
            return None
        acts, probs, _value, _argmax = res
        return {key(a): float(p) for a, p in zip(acts, probs)}

    def choose(self, sim, cands, smith=None):
        """-> (index of the teacher's best candidate, its probabilities over cands) or None.
        cands: our (action, args) per option; smith: {option index: card token} for rest-site
        smith options (same rest action, the card decides)."""
        rm = getattr(sim, "rm", None)
        if rm is None or len(cands) < 2:
            return None
        d = self.dist(rm)
        if not d:
            return None
        ks = [sim.resolve(a, args) for a, args in cands]
        if any(k is None for k in ks):
            return None
        p = [d.get(key(k), 0.0) for k in ks]
        if smith:
            p = self._split_smith(rm, ks, p, smith)
        tot = sum(p)
        if tot <= 0:
            return None
        p = [x / tot for x in p]
        return max(range(len(p)), key=lambda i: p[i]), p

    def _split_smith(self, rm, ks, p, smith):
        """Share the smith action's probability among our SMITH:<card> options by the teacher's
        card pick right after it (on a copy)."""
        idx = [i for i in smith if i < len(ks)]
        if not idx:
            return p
        cp = rm.__copy__()
        cp.take_action(ks[idx[0]])
        d = self.dist(cp) or {}
        deck = list(_g(cp, "master_deck_detailed") or [])
        cand = list(_g(cp, "upgrade_select_deck_indices") or [])   # smith_card idx -> deck index
        by_card = {}
        for k, q in d.items():
            if k[0] == "smith_card" and k[1] is not None and k[1] < len(cand) and cand[k[1]] < len(deck):
                nm = norm(card_name(deck[cand[k[1]]][0]))
                by_card[nm] = max(by_card.get(nm, 0.0), q)
        q = [by_card.get(norm(smith[i]), 0.0) for i in idx]
        if sum(q) <= 0:
            return p   # cannot tell the cards apart: keep the smith share on each
        out = list(p)
        share = p[idx[0]]
        for i, qi in zip(idx, q):
            out[i] = share * qi / sum(q)
        return out


def _g(rm, name):
    v = getattr(rm, name)
    return v() if callable(v) else v


def norm(card):
    return str(card or "").rstrip("+").replace(" ", "").replace("_", "").lower()


def key(a):
    """Hashable identity of an sts2-rs action dict."""
    return (a.get("action"), a.get("idx"), a.get("col"), a.get("row"), a.get("option"), a.get("hand_idx"),
            a.get("target_index"))
