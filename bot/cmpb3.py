"""usage (from bot/): cmpb3.py BASE TAG ...  paired act-3 boss share (B3, 0..2, win = 2), win
estimate W and the smcb return over runs/<tag>.jsonl, all from the act-3-start MC plays."""
import json, os, sys, statistics as S
os.environ.setdefault("STS2_RETURN", "smcb"); os.environ.setdefault("STS2_BOSS3_BONUS", "15")
sys.path.insert(0, os.getcwd())
import nets

def load(t):
    return {json.loads(l)["seed"]: json.loads(l) for l in open(f"runs/{t}.jsonl")}

def m(r):
    a3 = (r.get("act") or 1) >= 3 or r.get("victory")
    return {"B3": nets.boss3_estimate(r) if a3 else 0.0, "W": 100 * nets.win_estimate(r) if a3 else 0.0,
            "R": nets.final_progress(r), "beat1": (sum(1 for x in ([nets.boss3_real(r)] + ((r.get("mc") or {}).get("bdmg") or [])) if x >= 1)
                                                  / (1 + len((r.get("mc") or {}).get("bdmg") or []))) if a3 else 0.0}

def se(x):
    return S.stdev(x) / len(x) ** 0.5
tags = sys.argv[1:]
D = {t: load(t) for t in tags}
base = D[tags[0]]
for t in tags:
    rows = D[t]
    M = {s: m(r) for s, r in rows.items()}
    mc = sum(1 for r in rows.values() if r.get("mc"))
    line = f"{t:12s} n {len(rows)} mc {mc:3d} | " + "  ".join(f"{k} {S.mean(v[k] for v in M.values()):.4f}" for k in ("B3", "beat1", "W", "R"))
    if t != tags[0]:
        common = [s for s in rows if s in base]
        Mb = {s: m(base[s]) for s in common}
        line += " | paired " + "  ".join(f"d{k} {S.mean(M[s][k] - Mb[s][k] for s in common):+.4f}±{se([M[s][k] - Mb[s][k] for s in common]):.4f}" for k in ("B3", "beat1", "R"))
    print(line)
