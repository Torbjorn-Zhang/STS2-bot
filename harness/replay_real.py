"""Replay an engine-recorded run in the real game (STS2MCP mod, localhost:15526), action by action.

    replay_rec.py REC.jsonl SEED --dump STATES.jsonl      (engine states before every action)
    replay_real.py STATES.jsonl [--from N] [--check-only]

The real game must be at character select with the run's character and ascension chosen, or
already inside the run at step --from (the checkpoint file says where a stopped replay was).
Each engine action is translated to the mod's API using the engine state it was taken in and
the real state the game shows (hand position, enemy order, map coordinates, shop item ids).
The real game has screens the engine skips (the rewards list, treasure chests, dialogue, room
exits): those are walked through the way the engine resolves them (gold, relics and potions
taken in list order, the first card reward offered, the first treasure relic). After every
action the visible state (HP, gold, hand, enemy HP) is compared; on a difference the replay
stops instead of playing on in a different game.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

API = "http://localhost:15526/api/v1/singleplayer"
COMBAT = ("monster", "elite", "boss")
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "local", "replay_real.ckpt")
LOG = None
TRIED = set()   # rewards already claimed once (a claim the game ignores must not loop)


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    if LOG:
        LOG.write(msg + "\n")
        LOG.flush()


def real_get():
    for _ in range(20):
        try:
            with urllib.request.urlopen(API + "?format=json", timeout=15) as r:
                return json.load(r)
        except Exception:
            time.sleep(1)
    raise RuntimeError("mod API not answering")


def post(action, **kw):
    body = {"action": action, **kw}
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)
    log("   real <-", json.dumps(body, ensure_ascii=False), "->", res.get("status"), (res.get("message") or res.get("error") or "")[:120])
    return res


def stable_key(st):
    t = st.get("state_type")
    p = st.get("player") or {}
    b = st.get("battle") or {}
    key = [t, p.get("hp"), p.get("block"), p.get("gold"), p.get("energy"), [c.get("id") for c in p.get("hand") or []],
           p.get("draw_pile_count"), p.get("discard_pile_count"), p.get("exhaust_pile_count"),
           [(e.get("hp"), e.get("block")) for e in b.get("enemies") or []], b.get("turn"), b.get("is_play_phase")]
    for k in ("event", "rewards", "card_reward", "card_select", "hand_select", "shop", "rest_site", "treasure", "map"):
        if k in st:
            key.append(json.dumps(st[k], sort_keys=True)[:4000])
    return json.dumps(key, ensure_ascii=False)


def settle(timeout=60.0):
    """The real state once it stops changing and is actionable (the player's play phase in combat)."""
    t0, last, same = time.time(), None, 0
    while time.time() - t0 < timeout:
        st = real_get()
        t = st.get("state_type")
        b = st.get("battle") or {}
        busy = (t in COMBAT and not (b.get("is_play_phase") and b.get("turn") == "player")) or \
               (t == "treasure" and not (st.get("treasure") or {}).get("relics") and not (st.get("treasure") or {}).get("can_proceed")) or \
               (t == "shop" and (st.get("shop") or {}).get("error"))
        k = stable_key(st)
        if not busy and k == last:
            same += 1
            if same >= 3:
                return st
        else:
            same = 0
        last = k
        time.sleep(0.35)
    raise TimeoutError(f"real game did not settle (state {real_get().get('state_type')})")


def cid(x):
    return (x or "").replace("CARD.", "").replace("RELIC.", "").replace("POTION.", "")


# ---------------------------------------------------------------- engine-state helpers

def sim_decision(S):
    return S.get("decision")


def sim_in_combat(S):
    return (S.get("context") or {}).get("room_type") in ("Monster", "Elite", "Boss") and S.get("hand") is not None


def matches(d, S, R):
    t = R.get("state_type")
    if d == "map_select":
        return t == "map"
    if d == "combat_play":
        return t in COMBAT
    if d == "card_select":
        return t in ("hand_select", "card_select")
    if d == "event_choice":
        return t == "event" and not (R.get("event") or {}).get("in_dialogue")
    if d == "rest_site":
        return t == "rest_site"
    if d == "card_reward":
        return t == "card_reward"
    if d == "shop":
        return t == "shop"
    if d == "bundle_select":
        return t == "bundle_select"
    if d == "game_over":
        return t == "game_over"
    return False


def claim_rewards(d, R):
    """The engine takes gold, relics and potions in list order, then offers the first card reward."""
    items = (R.get("rewards") or {}).get("items") or []
    p = R.get("player") or {}
    belt_full = len(p.get("potions") or []) >= (p.get("max_potion_slots") or 3)
    for it in items:
        if it.get("type") == "potion" and belt_full:
            continue   # the engine leaves a potion it has no slot for (the mod still answers "ok")
        key = (R.get("run") or {}).get("floor"), it.get("type"), it.get("description")
        if it.get("type") in ("gold", "relic", "potion") and key not in TRIED:
            TRIED.add(key)
            res = post("claim_reward", index=it["index"])
            time.sleep(0.8)
            if res.get("status") == "ok":
                return True
            if it.get("type") != "potion":
                raise RuntimeError(f"could not claim {it}")
    # nothing else to take: the card reward when the engine offers one, else leave
    if d == "card_reward":
        card = next((it for it in items if it.get("type") in ("card", "special_card")), None)
        if card is None:
            raise RuntimeError("engine offers a card reward, real rewards have none")
        post("claim_reward", index=card["index"])
        return True
    post("proceed")
    return True


def reach(d, S, R, limit=25):
    """Walk the real game through screens the engine skips until it shows decision d."""
    for _ in range(limit):
        if matches(d, S, R):
            return R
        t = R.get("state_type")
        if t == "event" and (R.get("event") or {}).get("in_dialogue"):
            post("advance_dialogue")
        elif t == "rewards":
            claim_rewards(d, R)
        elif t == "treasure":
            tr = R.get("treasure") or {}
            if tr.get("relics"):
                post("claim_treasure_relic", index=0)
            else:
                post("proceed")
        elif t == "card_reward" and d != "card_reward":
            post("skip_card_reward")   # a second card reward: the engine offers only the first
        elif t in ("rest_site", "shop", "fake_merchant") and d != t:
            post("proceed")
        elif t == "event" and d != "event_choice":
            opts = (R.get("event") or {}).get("options") or []
            pr = next((o for o in opts if o.get("is_proceed")), None)
            if pr is None:
                raise RuntimeError(f"event without a proceed option while the engine is at {d}")
            post("choose_event_option", index=pr["index"])
        elif t == "menu" and R.get("menu_screen") in ("tutorial", "popup"):
            opts = [o.get("name") for o in R.get("options") or []]
            post("menu_select", option="advance" if "advance" in opts else opts[0])
        elif t == "relic_select":
            raise RuntimeError("relic_select: not in the engine's flow")
        else:
            raise RuntimeError(f"cannot reach engine decision {d} from real state {t}")
        R = settle()
    raise RuntimeError(f"gave up reaching {d}")


# ---------------------------------------------------------------- comparisons

def compare(d, S, R):
    diffs = []
    sp, rp = S.get("player") or {}, R.get("player") or {}
    for k in ("hp", "max_hp", "gold"):
        if sp.get(k) is not None and rp.get(k) is not None and sp.get(k) != rp.get(k):
            diffs.append((k, sp.get(k), rp.get(k)))
    if d == "combat_play":   # card ids only: the engine's hand cards carry no upgrade flag
        sh = [cid(c.get("id")) for c in S.get("hand") or []]
        rh = [cid(c.get("id")) for c in rp.get("hand") or []]
        if sh != rh:
            diffs.append(("hand", sh, rh))
        se = [e.get("hp") for e in S.get("enemies") or [] if (e.get("hp") or 0) > 0]
        re_ = [e.get("hp") for e in (R.get("battle") or {}).get("enemies") or [] if (e.get("hp") or 0) > 0]
        if se != re_:
            diffs.append(("enemy_hp", se, re_))
        if S.get("energy") is not None and rp.get("energy") is not None and S.get("energy") != rp.get("energy"):
            diffs.append(("energy", S.get("energy"), rp.get("energy")))
    if d == "card_reward":
        sc = [cid(c.get("id")) for c in S.get("cards") or []]
        rc = [cid(c.get("id")) for c in (R.get("card_reward") or {}).get("cards") or []]
        if sc != rc:
            diffs.append(("card_reward", sc, rc))
    return diffs


# ---------------------------------------------------------------- actions

def alive_enemies_real(R):
    return [e for e in (R.get("battle") or {}).get("enemies") or [] if (e.get("hp") or 0) > 0]


def target_id(S, R, target_index):
    sim_alive = [e for e in S.get("enemies") or [] if (e.get("hp") or 0) > 0]
    real_alive = alive_enemies_real(R)
    if target_index is None:
        return real_alive[0]["entity_id"] if len(real_alive) == 1 else None
    pos = next(i for i, e in enumerate(sim_alive) if e.get("index") == target_index)
    return real_alive[pos]["entity_id"]


def act(d, S, R, a, kw):
    if a == "select_map_node":
        opts = (R.get("map") or {}).get("next_options") or []
        o = next(o for o in opts if o["col"] == kw["col"] and o["row"] == kw["row"])
        return post("choose_map_node", index=o["index"])
    if a == "end_turn":
        return post("end_turn")
    if a == "play_card":
        hand = S.get("hand") or []
        pos = next(i for i, c in enumerate(hand) if c.get("index") == kw["card_index"])
        card = (R.get("player") or {}).get("hand")[pos]
        args = {"card_index": card["index"]}
        if card.get("target_type") == "AnyEnemy":
            t = target_id(S, R, kw.get("target_index"))
            if t:
                args["target"] = t
        return post("play_card", **args)
    if a in ("use_potion", "discard_potion"):
        # the engine numbers the potions it holds 0.. (no gaps, no ids); the game keeps belt slots
        pots = sorted((R.get("player") or {}).get("potions") or [], key=lambda p: p.get("slot"))
        rp = pots[kw["potion_index"]]
        if a == "discard_potion":
            return post("discard_potion", slot=rp["slot"])
        args = {"slot": rp["slot"]}
        if rp.get("target_type") == "AnyEnemy":
            t = target_id(S, R, kw.get("target_index"))
            if t:
                args["target"] = t
        return post("use_potion", **args)
    if a == "choose_option" and d == "event_choice":
        return post("choose_event_option", index=kw["option_index"])
    if a == "choose_option" and d == "rest_site":
        so = next(o for o in S.get("options") or [] if o.get("index") == kw["option_index"])
        want = (so.get("option_id") or "").lower()
        alias = {"heal": ("heal", "rest"), "smith": ("smith",)}.get(want, (want,))
        ro = next((o for o in (R.get("rest_site") or {}).get("options") or [] if (o.get("id") or "").lower() in alias), None)
        return post("choose_rest_option", index=ro["index"] if ro else kw["option_index"])
    if a == "select_card_reward":
        return post("select_card_reward", card_index=kw["card_index"])
    if a == "skip_card_reward":
        return post("skip_card_reward")
    if a in ("select_cards", "skip_select"):
        # by card identity (id and upgrade): the real grid may list the cards in another order
        idx = [int(x) for x in str(kw.get("indices", "")).split(",") if x != ""] if a == "select_cards" else []
        t = R.get("state_type")
        cards = S.get("cards") or []
        real_cards = (R.get("hand_select") or {}).get("cards") if t == "hand_select" else (R.get("card_select") or {}).get("cards")
        used = set()
        for i in idx:
            sc = next(c for j, c in enumerate(cards) if c.get("index", j) == i)
            want = (cid(sc.get("id")), bool(sc.get("upgraded")))
            rc = next((c for c in real_cards or [] if c["index"] not in used
                       and (cid(c.get("id")), bool(c.get("is_upgraded"))) == want), None)
            if rc is None:   # no upgrade flag on one side: match the id alone
                rc = next(c for c in real_cards or [] if c["index"] not in used and cid(c.get("id")) == want[0])
            used.add(rc["index"])
            if t == "hand_select":
                post("combat_select_card", card_index=rc["index"])
            else:
                post("select_card", index=rc["index"])
            time.sleep(0.5)
        R2 = real_get()
        if t == "hand_select":
            if (R2.get("hand_select") or {}).get("can_confirm", True) or not idx:
                return post("combat_confirm_selection")
            return {"status": "ok"}
        cs = R2.get("card_select") or {}
        if R2.get("state_type") == "card_select" and (cs.get("can_confirm") or cs.get("preview_showing")):
            return post("confirm_selection")
        if R2.get("state_type") == "card_select" and not idx:
            return post("cancel_selection")
        return {"status": "ok"}
    if a in ("buy_card", "buy_relic", "buy_potion", "remove_card"):
        items = (R.get("shop") or {}).get("items") or []
        if a == "remove_card":
            it = next(i for i in items if i.get("category") == "card_removal")
        else:   # the engine lists the shop's cards/relics/potions in the game's order, without ids: by position
            cat = a.split("_")[1]
            key = {"card": "card_index", "relic": "relic_index", "potion": "potion_index"}[cat]
            pool = S.get({"card": "cards", "relic": "relics", "potion": "potions"}[cat]) or []
            pos = next(j for j, x in enumerate(pool) if x.get("index", j) == kw.get(key))
            it = [i for i in items if i.get("category") == cat][pos]
            if not it.get("is_stocked"):
                raise RuntimeError(f"shop item {it} is sold out in the real game")
        return post("shop_purchase", index=it["index"])
    if a == "select_bundle":
        post("select_bundle", index=kw["bundle_index"])
        time.sleep(0.8)
        return post("confirm_bundle_selection")
    if a in ("leave_room", "proceed"):
        t = R.get("state_type")
        if t == "event":
            opts = (R.get("event") or {}).get("options") or []
            pr = next((o for o in opts if o.get("is_proceed")), None)
            if pr is not None:
                return post("choose_event_option", index=pr["index"])
        return post("proceed")
    raise RuntimeError(f"no translation for {a} at {d}")


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("states")
    ap.add_argument("--from", dest="start", type=int, default=None)
    ap.add_argument("--seed", default=None, help="embark from character select with this seed first")
    ap.add_argument("--to", type=int, default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    LOG = open(args.log, "a") if args.log else None
    steps = [json.loads(l) for l in open(args.states)]
    start = args.start
    if start is None:
        start = int(open(CKPT).read()) if os.path.exists(CKPT) else 0
    if args.seed:
        res = post("menu_select", option="embark", seed=args.seed)
        if res.get("status") != "ok":
            sys.exit(f"embark failed: {res}")
        time.sleep(6)
    end = args.to if args.to is not None else len(steps)
    for i in range(start, end):
        S, (a, kw) = steps[i]["state"], steps[i]["action"]
        d = sim_decision(S)
        R = settle()
        R = reach(d, S, R)
        diffs = compare(d, S, R)
        for _ in range(6):   # card effects that play out (auto-played cards) can outlast settle(): look again
            if not diffs:
                break
            time.sleep(1.5)
            R = settle()
            diffs = compare(d, S, R) if matches(d, S, R) else diffs
        ctx = S.get("context") or {}
        log(f"[{i}] act {ctx.get('act')} floor {ctx.get('floor')} {d}: {a} {json.dumps(kw)}")
        if diffs:
            log("DIFF", json.dumps(diffs, ensure_ascii=False)[:1500])
            sys.exit(f"stopped at step {i}: the real game differs from the engine")
        res = act(d, S, R, a, kw)
        if res.get("status") == "error":
            sys.exit(f"stopped at step {i}: the real game refused {a}: {res}")
        with open(CKPT, "w") as fh:
            fh.write(str(i + 1))
        time.sleep(0.3)
    log("replay finished at step", end)


if __name__ == "__main__":
    main()
