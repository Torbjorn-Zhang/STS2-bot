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
BOSSES = {1: ("CEREMONIAL_BEAST_BOSS", "THE_KIN_BOSS", "SOUL_FYSH_BOSS", "VANTOM_BOSS", "WATERFALL_GIANT_BOSS",
              "LAGAVULIN_MATRIARCH_BOSS"),
          2: ("KAISER_CRAB_BOSS", "THE_INSATIABLE_BOSS", "KNOWLEDGE_DEMON_BOSS"), 3: ACT3_BOSSES}
REWARDS_PER_ACT = 7     # card rewards per act (hallway fights, elites, the boss)
REMOVES_PER_ACT = 1     # shop removals
UPGRADES_PER_ACT = 2    # rest-site smiths (or heals when HP is low)
ELITE_SHARE = 0.15      # hallway fights that are elites


class DeckRoller:
    def __init__(self, model_paths, m=8, device="cpu", seed=0, inner=1, hall_paths=None):
        from deckvalue import DeckValue
        self.models = [DeckValue(os.path.join(HERE, p), device=device) for p in model_paths]
        # hallway models (deckvalue.py --target hpfrac on hallway/elite fights): survival and HP left
        self.hall = [DeckValue(os.path.join(HERE, p), device=device) for p in (hall_paths or [])]
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
        """The rest of the run as steps: ("reward", act) / ("remove",) / ("upgrade",) / ("boss", act)."""
        steps = []
        for a in range(act, 4):
            left = 1.0 if a > act else max(0.0, 1.0 - (floor or 0) / 17.0)
            n = int(round(REWARDS_PER_ACT * left))
            per = [("reward", a)] * n
            if left > 0.3:
                per.insert(len(per) // 2, ("remove",))
            per += [("upgrade",)] * int(round(UPGRADES_PER_ACT * left))
            steps += per + [("boss", a)]
        return steps

    def _logp(self, models, states):
        """log win probability (the deck model's win head), clipped."""
        self.stats["evals"] += len(states) * len(models)
        p = np.mean([m.win(states) for m in models], axis=0)
        return np.log(np.clip(p, 1e-6, 1.0 - 1e-6))

    def _hall(self, states):
        """-> (log P(survive), expected HP left as a share of max HP) for hallway fight states."""
        self.stats["evals"] += len(states) * len(self.hall)
        w = np.mean([m.win(states) for m in self.hall], axis=0)
        h = np.mean([m.score(states) for m in self.hall], axis=0)
        return np.log(np.clip(w, 1e-6, 1.0 - 1e-6)), np.clip(h, 0.0, 1.0)

    def futures(self, decks, ctx, act, floor, key, boss_now=None, hp=None):
        """decks: the option decks. -> (values per option, per-future values). A future's value is
        log P(it wins every boss left): each act boss scores the deck it holds there (act 3 = both
        final bosses, as the model was trained). With hallway models, every card reward is preceded
        by a hallway fight (an elite now and then) played by the deck of that moment: it can kill
        the future and costs HP the next boss fight sees (A10 act starts heal 80% of missing HP,
        rest sites heal below half HP). So a pick that pays off late must not cost the early acts."""
        rng = random.Random(f"{key}/{self.seed}")
        steps = self.schedule(act, floor)
        acts = list(range(act, 4))
        mhp = float(ctx.get("mhp") or 80)
        hp0 = float(hp if hp is not None else mhp)
        bosses = []   # per future: the boss of each act left (the current one when known)
        elites = []
        for j in range(self.m):
            b = {a: rng.choice(BOSSES[a]) for a in acts}
            if boss_now:
                b[act] = boss_now
            bosses.append(b)
            elites.append([rng.random() < ELITE_SHARE for _ in steps])
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
        T = len(threads)
        hpv = np.full(T, hp0)
        logp = np.zeros(T)
        cur_act = act
        hall = bool(self.hall)
        for si, s in enumerate(steps):
            if s[0] == "boss":
                st_ = [dict(ctx, deck=d, boss=bosses[j][s[1]], act=s[1], hp=int(max(1, hpv[t])), mhp=int(mhp))
                       for t, (o, j, d) in enumerate(threads)]
                logp += self._logp(self.models, st_)
                cur_act = s[1] + 1
                hpv = hpv + 0.8 * (mhp - hpv)   # Ascension 2: the next act's Ancient heals 80% of missing HP
                continue
            if s[0] == "reward" and hall:   # the fight before this reward, with the deck of this moment
                st_ = [dict(ctx, deck=d, boss=("ELITE" if elites[j][si] else "MONSTER") + f"_A{s[1]}", act=s[1],
                            hp=int(max(1, hpv[t])), mhp=int(mhp)) for t, (o, j, d) in enumerate(threads)]
                lw, hf = self._hall(st_)
                logp += lw
                hpv = np.minimum(hpv, np.maximum(1.0, hf * mhp))
            if s[0] == "upgrade" and hall:   # rest site: heal when low, else smith
                low = hpv < 0.5 * mhp
                hpv = np.where(low, np.minimum(mhp, hpv + 0.3 * mhp), hpv)
            else:
                low = np.zeros(T, dtype=bool)
            cand, owner = [], []
            for t, (o, j, d) in enumerate(threads):
                if s[0] == "reward":
                    opts = [d] + [d + [c] for c in draws[j][si]]
                elif s[0] == "remove":
                    uniq = sorted(set(d))
                    opts = [d] + [_without(d, c) for c in uniq] if len(d) > 5 else [d]
                else:
                    ups = sorted({c for c in d if not c.endswith("+") and not c.startswith(("ASCENDERS_BANE", "CURSE"))})
                    opts = [d] + ([] if low[t] else [_upgraded(d, c) for c in ups])
                for x in opts:
                    cand.append((t, x))
            if all(len([1 for c in cand if c[0] == t]) == 1 for t in range(T)):
                continue
            # greedy step: the candidate deck against every boss still ahead (the current act's at
            # this HP, later ones at full HP), plus surviving the next hallway fight
            ahead = [a for a in acts if a >= cur_act] or [3]
            states, idx = [], []
            for i, (t, x) in enumerate(cand):
                j = threads[t][1]
                for a in ahead:
                    states.append(dict(ctx, deck=x, boss=bosses[j][a], act=a,
                                       hp=int(max(1, hpv[t])) if a == cur_act else int(mhp), mhp=int(mhp)))
                    idx.append(i)
            lp = self._logp(self.inner, states)
            tot = np.zeros(len(cand))
            np.add.at(tot, idx, lp)
            if hall and cur_act <= 3:
                hs = [dict(ctx, deck=x, boss=f"MONSTER_A{min(cur_act, 3)}", act=min(cur_act, 3), hp=int(max(1, hpv[t])), mhp=int(mhp))
                      for t, x in cand]
                lw, _ = self._hall(hs)
                tot += lw
            best = {}
            for i, (t, x) in enumerate(cand):
                if t not in best or tot[i] > best[t][0]:
                    best[t] = (tot[i], x)
            threads = [(o, j, best[t][1]) for t, (o, j, d) in enumerate(threads)]
        per = logp.reshape(len(decks), self.m)
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
        val, _ = self.futures(decks, ctx, ctx_.get("act") or 1, ctx_.get("floor") or 0, key,
                              boss_now=(ctx_.get("boss") or {}).get("id"), hp=p.get("hp"))
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
    val, _ = roller.futures(decks, ctx, ctx_.get("act") or 1, ctx_.get("floor") or 0, key,
                            boss_now=(ctx_.get("boss") or {}).get("id"), hp=p.get("hp"))
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
