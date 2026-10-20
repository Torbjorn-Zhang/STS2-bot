"""Paired comparison by win probability, not by floors: python cmpwin.py base tag [tag ...]

Floors reward going far, and a bot that heals, hoards potions and dodges elites goes far without
ever winning. Wins are the goal but are rare (~1.5% at A10), so one outcome per seed is very
noisy. Runs made with run.py --mc-k K carry K extra plays of the rest of the run from act
--mc-act's first map node, each with a fresh hidden future. Per seed
    W = 1{reached that act} * (real win + MC wins) / (1 + K)
is an unbiased estimate of the seed's win chance with far less variance. The table shows W
(in %), raw wins, and paired differences with standard errors."""
import json
import statistics as S
import sys


def load(t):
    return {json.loads(l)["seed"]: json.loads(l) for l in open(f"runs/{t}.jsonl")}


def w_est(r):
    """(estimate, has MC data)"""
    mc = r.get("mc")
    if mc and mc.get("win") is not None and (mc["win"] or r["victory"]):
        return (r["victory"] + sum(mc["win"])) / (1 + len(mc["win"])), True
    return float(r["victory"]), bool(r.get("act_states"))


def se(x):
    return S.stdev(x) / len(x) ** 0.5 if len(x) > 1 else 0.0


tags = sys.argv[1:]
D = {t: load(t) for t in tags}
base = D[tags[0]]
for t in tags:
    rows = list(D[t].values())
    est = [w_est(r)[0] for r in rows]
    raw = [float(r["victory"]) for r in rows]
    mcs = [r["mc"] for r in rows if r.get("mc")]
    k = S.mean(len(m["win"]) for m in mcs) if mcs else 0
    reach = sum(1 for r in rows if (r["act"] or 0) >= 3 or r["victory"])
    line = (f"{t:22s} n {len(rows):4d} reach3 {reach / len(rows):5.1%} wins {int(sum(raw)):3d} ({S.mean(raw):.2%}) | "
            f"W {100 * S.mean(est):.2f}%±{100 * se(est):.2f} (mc runs {len(mcs)}, k {k:.1f}, "
            f"err {sum(m.get('err', 0) for m in mcs)})")
    if t != tags[0]:
        s = [x for x in base if x in D[t]]
        dw = [w_est(D[t][x])[0] - w_est(base[x])[0] for x in s]
        dr = [float(D[t][x]["victory"]) - float(base[x]["victory"]) for x in s]
        line += f" | paired n {len(s)} dW {100 * S.mean(dw):+.2f}±{100 * se(dw):.2f} raw {100 * S.mean(dr):+.2f}±{100 * se(dr):.2f}"
    print(line)
    if mcs and len(rows) > 1:
        v_raw, v_w = S.pvariance(raw), S.pvariance(est)
        print(f"{'':22s} variance raw {v_raw:.5f} W {v_w:.5f} -> {v_raw / max(v_w, 1e-12):.1f}x fewer seeds needed; "
              f"P(win | reached act {mcs[0]['act']}) {S.mean(sum(m['win']) / max(1, len(m['win'])) for m in mcs):.2%}; "
              f"rollout {S.mean(x for m in mcs for x in m['secs']):.0f} s")
