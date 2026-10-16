"""Feature encoders for the policy and value networks (record.py compact states).

Vocabularies are built from the training data and stored with each model, so a model
always encodes states the way it was trained. Unknown tokens fall into a shared slot.
"""

import numpy as np

ACT_NAMES = ["Overgrowth", "Underdocks", "Hive", "Glory"]
FLOORS_PER_ACT = 17


def progress(act, floor):
    """Floors climbed so far, counting every act as 17 floors (the boss is floor 16-17)."""
    return ((act or 1) - 1) * FLOORS_PER_ACT + (floor or 0)


def base(tok):
    return tok[:-1] if tok.endswith("+") else tok


class Vocab:
    def __init__(self, items=None):
        self.items = ["<unk>"] + sorted(set(items or []) - {"<unk>"})
        self.index = {t: i for i, t in enumerate(self.items)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, t):
        return self.index.get(t, 0)

    def to_json(self):
        return self.items

    @classmethod
    def from_json(cls, items):
        v = cls()
        v.items = items
        v.index = {t: i for i, t in enumerate(items)}
        return v


class RunEncoder:
    """Run state (deck, relics, potions, HP, gold, where we are) -> fixed vector."""

    def __init__(self, cards, relics, pots, bosses):
        self.cards, self.relics, self.pots, self.bosses = cards, relics, pots, bosses
        self.n_scalar = 12 + len(ACT_NAMES)
        self.dim = self.n_scalar + 2 * len(cards) + len(relics) + len(pots) + len(bosses)

    @classmethod
    def build(cls, states):
        cards, relics, pots, bosses = set(), set(), set(), set()
        for s in states:
            cards.update(base(t) for t in s.get("deck") or [])
            relics.update(s.get("relics") or [])
            pots.update(s.get("pots") or [])
            bosses.add(s.get("boss") or "")
        return cls(Vocab(cards), Vocab(relics), Vocab(pots), Vocab(bosses))

    def to_json(self):
        return {"cards": self.cards.to_json(), "relics": self.relics.to_json(), "pots": self.pots.to_json(),
                "bosses": self.bosses.to_json()}

    @classmethod
    def from_json(cls, d):
        return cls(Vocab.from_json(d["cards"]), Vocab.from_json(d["relics"]), Vocab.from_json(d["pots"]),
                   Vocab.from_json(d["bosses"]))

    def encode(self, s, deck=None):
        """s: record.run_state dict. deck: optional replacement deck (token list)."""
        x = np.zeros(self.dim, dtype=np.float32)
        deck = s.get("deck") or [] if deck is None else deck
        hp, mhp = s.get("hp") or 0, max(1, s.get("mhp") or 1)
        act, floor = s.get("act") or 1, s.get("floor") or 0
        n_up = sum(1 for t in deck if t.endswith("+"))
        basics = sum(1 for t in deck if t.startswith(("STRIKE_", "DEFEND_")))
        sc = [hp / mhp, hp / 100, mhp / 100, (s.get("gold") or 0) / 300, floor / FLOORS_PER_ACT,
              progress(act, floor) / 51, act == 1, act == 2, act == 3, (s.get("asc") or 0) / 10,
              len(deck) / 30, (n_up + 0.0) / 10 + basics / 100]
        x[:12] = sc
        an = s.get("an")
        if an in ACT_NAMES:
            x[12 + ACT_NAMES.index(an)] = 1.0
        o = self.n_scalar
        nc = len(self.cards)
        for t in deck:
            j = self.cards[base(t)]
            x[o + j] += 1.0
            if t.endswith("+"):
                x[o + nc + j] += 1.0
        o += 2 * nc
        for r in s.get("relics") or []:
            x[o + self.relics[r]] = 1.0
        o += len(self.relics)
        for p in s.get("pots") or []:
            x[o + self.pots[p]] += 1.0
        o += len(self.pots)
        x[o + self.bosses[s.get("boss") or ""]] = 1.0
        return x
