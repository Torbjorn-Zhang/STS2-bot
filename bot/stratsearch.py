"""Strategic search: judge a deck decision by playing the rest of the act forward from the
real state, AlphaGo-style (Monte-Carlo rollouts with the learned policies).

For each option: restore the decision state, make the run's hidden future random but
reproducible (engine determinize_run: rewards, shops, events, encounters, combat RNG), take
the option, then play to the end of the act in the calibrated fast world (policy-network
pilot, deck networks greedy). The return is the RL return (floors + act clears + boss HP
share). The same seeds are used for every option (common random numbers).
"""

import json
import os
import random
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


class StratSearch:
    def __init__(self, nets_cfg, pilot=os.path.join(HERE, "models", "policy_rl105.pt"), k=6, prior=0.0):
        import torch
        torch.set_num_threads(1)
        import nets
        import selfplay
        from eventnet import EventPolicy
        from mapnet import MapPolicy
        from restnet import RestPolicy
        from shopnet import ShopPolicy
        selfplay.CALIBRATE = True
        self.pilot = nets.NetPolicy(pilot)
        self.card = nets.CardPolicy(nets_cfg["card"]) if nets_cfg.get("card") else None
        self.rest = RestPolicy(nets_cfg["rest"]) if nets_cfg.get("rest") else None
        self.event = EventPolicy(nets_cfg["event"]) if nets_cfg.get("event") else None
        self.map = MapPolicy(nets_cfg["map"]) if nets_cfg.get("map") else None
        self.shop = ShopPolicy(nets_cfg["shop"]) if nets_cfg.get("shop") else None
        self.k, self.prior = k, prior
        self.stats = {"decisions": 0, "rollouts": 0}

    def rollout(self, sim, st, seed):
        import policy
        import selfplay
        from nets import final_progress
        act = (st.get("context") or {}).get("act") or 1
        # rollouts decide rests, events, map and shops by the networks directly (never
        # through policy.choose_rest, so no search recursion) and card rewards by the card
        # network (policy.CARD_NET_DECIDES); policy globals stay untouched (worker threads)
        res = selfplay.play_fast_run(sim, None, self.pilot, policy.ASCENSION, act, None, 0.0, random.Random(seed),
                                     self.card, 0.0, self.rest, 0.0, self.event, 0.0, self.map, 0.0, False,
                                     self.shop, 0.0, False, st0=st)
        self.stats["rollouts"] += 1
        if res.get("error"):
            return None
        return final_progress(res)

    def evaluate(self, sim, st, arms, tag=""):
        """arms: {key: moves} (decktest._apply format). Returns {key: mean return}; restores st."""
        import decktest
        if len(arms) < 2:
            return {}
        ctx = st.get("context") or {}
        base = f"strat/{tag}/{ctx.get('act')}/{ctx.get('floor')}"
        snap = sim.send({"cmd": "snapshot"})["id"]
        out = {}
        try:
            for key, moves in arms.items():
                vals = []
                for j in range(self.k):
                    sim.send({"cmd": "restore", "id": snap})
                    seed = zlib.crc32(f"{base}/{j}".encode()) * 2654435761 % (2 ** 62)
                    sim.send({"cmd": "determinize_run", "seed": seed})
                    s = decktest._apply(sim, st, moves)
                    if s.get("type") == "error":
                        continue
                    v = self.rollout(sim, s, seed)
                    if v is not None:
                        vals.append(v)
                if vals:
                    out[key] = float(np.mean(vals))
        finally:
            sim.send({"cmd": "restore", "id": snap})
            sim.send({"cmd": "drop_snapshots", "from": snap})
        self.stats["decisions"] += 1
        return out


def load(path, **kw):
    with open(path) as fh:
        return StratSearch(json.load(fh), **kw)
