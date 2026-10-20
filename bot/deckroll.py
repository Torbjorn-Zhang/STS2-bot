"""Card picks by imagined futures (playing for the future, "战个未来"): judge each option not by
what it adds now but by the deck it can grow into.

For every option of a card reward (each card, or skipping), play M imagined futures of the rest
of the run in deck space: the remaining card rewards drawn from the real offer pool of each act,
each one picked greedily by the deck model (or skipped), a card removal per shop and upgrades at
rest sites; each future faces two sampled act-3 bosses (Ascension 10's double boss). The option's
value is the deck model's mean value of those final decks. A combo piece that is weak alone wins
when the futures it opens are better than the ones the others open. All futures of all options
are stepped together in batches, with the same random futures for every option (common random
numbers), so options differ only by the option.

The deck model (deckvalue.py, trained on deckspace.py engine fights) is the AlphaGo value
network's counterpart; these rollouts are its rollouts.
"""
import json
import os
import random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ACT3_BOSSES = ("AEONGLASS_BOSS", "QUEEN_BOSS", "TEST_SUBJECT_BOSS")
REWARDS_PER_ACT = 7     # card rewards per act (hallway fights, elites, the boss)
REMOVES_PER_ACT = 1     # shop removals
UPGRADES_PER_ACT = 2    # rest-site smiths


class DeckRoller:
    def __init__(self, model_paths, m=8, device="cpu", seed=0, inner=1):
        from deckvalue import DeckValue
        self.models = [DeckValue(os.path.join(HERE, p), device=device) for p in model_paths]
        self.inner = self.models[:max(1, inner)]   # the greedy steps use a cheaper subset
        self.m = m
        self.seed = seed
        with open(os.path.join(HERE, "data", "offer_pool.json")) as fh:
            pool = json.load(fh)["pool"]
        self.pool = {}
        for a in ("1", "2", "3"):
            cards = [c for c, n in pool.get(a, {}).items() if n >= 20]
            self.pool[int(a)] = (cards, np.array([pool[a][c] for c in cards], dtype=float))
        self.stats = {"decisions": 0, "evals": 0}

    def _score(self, models, states):
        self.stats["evals"] += len(states) * len(models)
        return np.mean([m.score(states) for m in models], axis=0)

    def schedule(self, act, floor):
        """The rest of the run as steps: ("reward", act) / ("remove",) / ("upgrade",)."""
        steps = []
        for a in range(act, 4):
            left = 1.0 if a > act else max(0.0, 1.0 - (floor or 0) / 17.0)
            n = int(round(REWARDS_PER_ACT * left))
            per = [("reward", a)] * n
            if left > 0.3:
                per.insert(len(per) // 2, ("remove",))
            per += [("upgrade",)] * int(round(UPGRADES_PER_ACT * left))
            steps += per
        return steps

    def futures(self, decks, ctx, act, floor, key):
        """decks: the option decks. -> (values per option, per-future values)."""
        rng = random.Random(f"{key}/{self.seed}")
        steps = self.schedule(act, floor)
        # thread t = option o, future j; the random draws depend on j only (common random numbers)
        bosses = [rng.sample(ACT3_BOSSES, 2) for _ in range(self.m)]
        draws = []
        for j in range(self.m):
            r = random.Random(f"{key}/{self.seed}/{j}")
            d = []
            for s in steps:
                if s[0] == "reward":
                    cards, w = self.pool[s[1]]
                    p = w / w.sum()
                    d.append([cards[i] for i in np.random.default_rng(r.randrange(2 ** 31)).choice(len(cards), 3, replace=False, p=p)])
                else:
                    d.append(None)
            draws.append(d)
        threads = [(o, j, list(decks[o])) for o in range(len(decks)) for j in range(self.m)]
        for si, s in enumerate(steps):
            cand, owner = [], []
            for t, (o, j, d) in enumerate(threads):
                if s[0] == "reward":
                    opts = [d] + [d + [c] for c in draws[j][si]]
                elif s[0] == "remove":
                    uniq = sorted(set(d))
                    opts = [d] + [_without(d, c) for c in uniq] if len(d) > 5 else [d]
                else:
                    ups = sorted({c for c in d if not c.endswith("+") and not c.startswith(("ASCENDERS_BANE", "CURSE"))})
                    opts = [d] + [_upgraded(d, c) for c in ups]
                for x in opts:
                    cand.append(dict(ctx, deck=x, boss=bosses[j][0], act=3))
                    owner.append(t)
            v = self._score(self.inner, cand)
            best = {}
            for i, t in enumerate(owner):
                if t not in best or v[i] > best[t][0]:
                    best[t] = (v[i], cand[i]["deck"])
            threads = [(o, j, best[t][1]) for t, (o, j, d) in enumerate(threads)]
        final = []
        for o, j, d in threads:
            for b in bosses[j]:
                final.append(dict(ctx, deck=d, boss=b, act=3))
        v = self._score(self.models, final).reshape(len(threads), 2).mean(1)
        per = v.reshape(len(decks), self.m)
        return per.mean(1), per

    def choose(self, st, legal, prior_lp=None, beta=0.0):
        """legal: the reward screen's actions. -> (best action, {action json: value})."""
        from policy import potion_id
        from record import card_tok
        ctx_ = st.get("context") or {}
        p = st.get("player") or {}
        deck = [card_tok(c) for c in p.get("deck") or []]
        ctx = {"relics": [r.get("name") for r in p.get("relics") or []],
               "potions": [potion_id(x) or x.get("name") for x in p.get("potions") or [] if x],
               "hp": p.get("max_hp"), "mhp": p.get("max_hp")}
        by_index = {c["index"]: c for c in st.get("cards") or []}
        opts, decks = [], []
        for a in legal:
            if a[0] == "select_card_reward" and a[1].get("card_index") in by_index:
                opts.append(a)
                decks.append(deck + [card_tok(by_index[a[1]["card_index"]])])
            elif a[0] in ("skip_card_reward", "proceed") and not any(x[0] == a[0] for x in opts):
                opts.append(a)
                decks.append(list(deck))
        if len(opts) < 2:
            return None, {}
        key = f"{ctx_.get('act')}/{ctx_.get('floor')}/{len(deck)}/{','.join(sorted(deck))[:200]}"
        val, _ = self.futures(decks, ctx, ctx_.get("act") or 1, ctx_.get("floor") or 0, key)
        tot = val + (beta * np.asarray(prior_lp) if prior_lp is not None else 0.0)
        self.stats["decisions"] += 1
        i = int(np.argmax(tot))
        return opts[i], {json.dumps(a, sort_keys=True): round(float(x), 4) for a, x in zip(opts, val)}


def choose_deck(roller, st, options):
    """options: [(action, resulting deck tokens)] -> (best action, values); for removals and
    upgrades (the deck after the choice, judged by its futures like a card reward)."""
    from policy import potion_id
    ctx_ = st.get("context") or {}
    p = st.get("player") or {}
    ctx = {"relics": [r.get("name") for r in p.get("relics") or []],
           "potions": [potion_id(x) or x.get("name") for x in p.get("potions") or [] if x],
           "hp": p.get("max_hp"), "mhp": p.get("max_hp")}
    if len(options) < 2:
        return None, {}
    decks = [d for _, d in options]
    key = f"sel/{ctx_.get('act')}/{ctx_.get('floor')}/{','.join(sorted(decks[0]))[:200]}"
    val, _ = roller.futures(decks, ctx, ctx_.get("act") or 1, ctx_.get("floor") or 0, key)
    roller.stats["decisions"] += 1
    i = int(np.argmax(val))
    return options[i][0], {json.dumps(a, sort_keys=True): round(float(x), 4) for (a, _), x in zip(options, val)}


def _without(d, c):
    d = list(d)
    d.remove(c)
    return d


def _upgraded(d, c):
    d = list(d)
    d[d.index(c)] = c + "+"
    return d
