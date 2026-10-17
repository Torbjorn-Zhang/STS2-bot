"""Deck-decision networks inside the expert bot, sampled for RL (the self-play stage run
with the full planner bot instead of the fast world).

DeckNets(card, rest, event, map paths, temperature, shop=path) decides card rewards, rest
sites, events, map choices and shop purchases by sampling softmax(logits / T) and returns
the record extras the RL trainers read (card x.lp, rest x.rlp/rkeys/rk/route, event
x.elp/eopts/ek/route, map x.mlp/mopts/mk, shop x.slp/sopts/sk/route/smemo), the same format
as selfplay.py.
"""

import math
import random

import numpy as np


class DeckNets:
    def __init__(self, card=None, rest=None, event=None, mapp=None, temperature=1.0, seed=0, shop=None):
        self.card = self.rest = self.event = self.map = self.shop = None
        self.visits = {}   # shop visit -> {"removed", "bought"} (the shop network's own state input)
        if card:
            from nets import CardPolicy
            self.card = CardPolicy(card)
        if rest:
            from restnet import RestPolicy
            self.rest = RestPolicy(rest)
        if event:
            from eventnet import EventPolicy
            self.event = EventPolicy(event)
        if mapp:
            from mapnet import MapPolicy
            self.map = MapPolicy(mapp)
        if shop:
            from shopnet import ShopPolicy
            self.shop = ShopPolicy(shop)
        self.temperature = temperature
        self.rng = random.Random(seed)

    def _sample(self, z):
        z = np.asarray(z, dtype=np.float64)
        if self.temperature <= 0:
            i = int(np.argmax(z))
            return i, 0.0
        zz = (z - z.max()) / self.temperature
        p = np.exp(zz)
        p /= p.sum()
        i = self.rng.choices(range(len(z)), weights=p.tolist())[0]
        return i, float(math.log(max(p[i], 1e-12)))

    def decide(self, sim, st):
        """-> (action, extra) or (None, None) when no network covers this decision."""
        import policy
        d = st.get("decision")
        if d == "card_reward" and self.card is not None and st.get("cards"):
            acts, z = self.card.scores(st)
            i, lp = self._sample(z)
            return acts[i], {"lp": lp, "T": self.temperature}
        if d == "rest_site" and self.rest is not None:
            keys, ks, z, route = self.rest.scores(sim, st)
            if len(ks) >= 2:
                i, lp = self._sample(z)
                idx, tok = keys[ks[i]]
                policy._LAST_REST.option = ks[i].split(":")[0]
                policy._REST_PICK.token = tok
                return ("choose_option", {"option_index": idx}), {"rlp": lp, "rkeys": ks, "rk": ks[i], "route": route}
        if d == "event_choice" and self.event is not None:
            opts, descs, z, route = self.event.scores(sim, st)
            if z is not None:
                i, lp = self._sample(z)
                policy._remember_event(opts[i])
                return ("choose_option", {"option_index": opts[i]["index"]}), \
                    {"eopts": descs, "ek": i, "elp": lp, "route": route}
        if d == "shop" and self.shop is not None and sim is not None:
            ctx = st.get("context") or {}
            visit = self.visits.setdefault((ctx.get("act"), ctx.get("floor")), {"removed": False, "bought": 0})
            before = [bool(visit["removed"]), visit["bought"]]
            opts, route, z = self.shop.scores(sim, st, visit)
            if len(opts) >= 2:
                i, lp = self._sample(z)
                a = opts[i][1]
                if a[0] == "remove_card":
                    visit["removed"] = True
                elif a[0].startswith("buy_"):
                    visit["bought"] += 1
                return a, {"sopts": [o for o, _ in opts], "sk": i, "slp": lp, "route": route, "smemo": before}
        if d == "map_select" and self.map is not None and len(st.get("choices") or []) > 1:
            chs, descs, z = self.map.scores(sim, st)
            if z is not None:
                i, lp = self._sample(z)
                return ("select_map_node", {"col": chs[i]["col"], "row": chs[i]["row"]}), \
                    {"mopts": descs, "mk": i, "mlp": lp}
        return None, None
