"""Where A10 eval runs die: python deaths.py TAG ... (runs/<tag>.jsonl)."""
import json, sys
from collections import Counter, defaultdict
rows = []
for t in sys.argv[1:]:
    rows += [json.loads(l) for l in open(f"runs/{t}.jsonl")]
n = len(rows)
by_room, by_enemy, hp_frac = Counter(), Counter(), defaultdict(list)
boss_full = Counter()
for r in rows:
    if r.get("victory"):
        by_room["WIN"] += 1
        continue
    fl = r.get("fight_log") or []
    last = fl[-1] if fl and not fl[-1].get("won") else None
    if last is None:
        by_room[(r.get("act"), "non-combat/other")] += 1
        continue
    key = (r.get("act"), last["room"])
    by_room[key] += 1
    by_enemy[(r.get("act"), last["room"], "+".join(sorted(set(last["enemies"]))))] += 1
    hp_frac[key].append(last["hp_before"] / max(1, r.get("max_hp") or 1))
print(f"runs {n}")
for k, v in sorted(by_room.items(), key=lambda kv: -kv[1]):
    hf = hp_frac.get(k)
    print(f"  {str(k):28s} {v:4d} {100*v/n:5.1f}%" + (f"   hp_before/max mean {sum(hf)/len(hf):.2f}" if hf else ""))
print("top killers:")
for k, v in by_enemy.most_common(18):
    print(f"  {v:4d}  {k}")
