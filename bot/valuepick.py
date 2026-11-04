"""Would a one-step value lookahead change the card picks? For every recorded card_reward decision
whose next decision is in the record, the value head scores each option's resulting state: the
next decision's state with the deck the option would have left (the reward deck plus that card,
or unchanged for a skip), in that next decision's context. Reported: how often the value's best
option is the policy's pick, how far apart the options' values are (return points, the scale
the head fits), and whether runs whose pick the value rated well below its best did worse.

    valuepick.py models/up8_unified12.pt data/xdarchive/up8_ev12/*_rec.jsonl [--margin 2]
"""

import argparse
import copy
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from upolicy import Enc, chosen_of, collate, expand, kind_of, load_net, options_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--margin", type=float, default=2.0, help="return points: the pick is 'rated below' the best by more")
    args = ap.parse_args()
    blob = torch.load(os.path.join(HERE, args.net), map_location="cpu", weights_only=False)
    enc, (net, _) = Enc(blob=blob["enc"]), load_net(blob)
    net.eval()
    runs, outcome = defaultdict(list), {}
    for f in expand(args.files):
        for line in open(f):
            r = json.loads(line)
            key = f"{os.path.basename(f)}/{r.get('seed')}"
            if r.get("k") == "o":
                outcome[key] = r
            elif r.get("k") == "d":
                runs[key].append(r)
    cases = []   # (run, act, pick, policy probs, values)
    skipped = Counter()
    for key, R in runs.items():
        for i, r in enumerate(R[:-1]):
            if r.get("dec") != "card_reward":
                continue
            opts, nxt = options_of(r), R[i + 1]
            a = chosen_of(r, opts)
            if not opts or len(opts) < 2 or a is None or not 0 <= a < len(opts) or options_of(nxt) is None:
                skipped["no options"] += 1
                continue
            deck0, deck1 = list(r["s"].get("deck") or []), list(nxt["s"].get("deck") or [])
            kept = [o[1] for o in opts]
            if sorted(deck0 + ([kept[a]] if kept[a] else [])) != sorted(deck1):
                skipped["next deck is not reward deck + pick"] += 1   # something else changed the deck in between
                continue
            items = []
            for o in opts:
                s = copy.deepcopy(nxt["s"])
                s["deck"] = deck0 + ([o[1]] if o[1] else [])
                items.append(enc.encode(kind_of(nxt), s, options_of(nxt), (nxt.get("x") or {}).get("route")))
            items.append(enc.encode("card_reward", r["s"], opts, (r.get("x") or {}).get("route")))
            with torch.no_grad():
                z, v = net(collate(items, torch.device("cpu")), with_value=True)
            p = torch.softmax(z[-1][:len(opts)], -1).numpy()
            cases.append((key, r["s"].get("act"), a, p, 100.0 * v[:-1].numpy()))
    if not cases:
        print(json.dumps({"cases": 0, "skipped": dict(skipped)}))
        return
    agree_pick = np.mean([int(np.argmax(v) == a) for _, _, a, _, v in cases])
    agree_pol = np.mean([int(np.argmax(v) == np.argmax(p)) for _, _, _, p, v in cases])
    spread = np.array([v.max() - v.min() for *_, v in cases])
    gap = np.array([v.max() - v[a] for _, _, a, _, v in cases])
    win = np.array([bool(outcome.get(k, {}).get("victory")) for k, *_ in cases])
    below = gap > args.margin
    res = {"cases": len(cases), "skipped": dict(skipped), "value_best_is_pick": round(float(agree_pick), 3),
           "value_best_is_policy_top": round(float(agree_pol), 3),
           "spread_median": round(float(np.median(spread)), 2), "spread_p90": round(float(np.percentile(spread, 90)), 2),
           "pick_gap_median": round(float(np.median(gap)), 2),
           f"share_pick_below_best_by_{args.margin:g}": round(float(below.mean()), 3)}
    # runs: did the runs with any pick the value rated well below its best do worse?
    worst = defaultdict(float)
    for (k, *_), g in zip(cases, gap):
        worst[k] = max(worst[k], g)
    ks = list(worst)
    w = np.array([bool(outcome.get(k, {}).get("victory")) for k in ks]); g = np.array([worst[k] for k in ks])
    res.update({"runs": len(ks), "win_runs_no_big_gap": round(float(w[g <= args.margin].mean()), 3) if (g <= args.margin).any() else None,
                "win_runs_big_gap": round(float(w[g > args.margin].mean()), 3) if (g > args.margin).any() else None,
                "n_runs_big_gap": int((g > args.margin).sum())})
    by_act = defaultdict(list)
    for (_, act, a, p, v) in cases:
        by_act[act].append(int(np.argmax(v) == a))
    res["value_best_is_pick_by_act"] = {str(k): round(float(np.mean(x)), 3) for k, x in sorted(by_act.items())}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
