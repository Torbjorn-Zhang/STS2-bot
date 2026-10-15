"""Summarise scan_results.jsonl: per-status counts and the cards that need a closer look."""

import json
import sys
from collections import Counter, defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "scan_results.jsonl"
rows = [json.loads(l) for l in open(path)]
by_card = defaultdict(dict)
for r in rows:
    by_card[r["card"]][r["char"]] = r

print("jobs:", len(rows), "cards:", len(by_card))
print("status counts:", dict(Counter(r["status"] for r in rows)))

known = {c: v for c, v in by_card.items() if not all(x["status"] == "unknown_id" for x in v.values())}
print("card ids in ModelDb:", len(known), "| not in ModelDb (stale loc keys):", len(by_card) - len(known))

BAD = ("play_error", "timeout", "engine_exit", "stuck_after_play", "enter_error", "left_combat_before_play")
groups = defaultdict(list)
for c, v in sorted(known.items()):
    st = {ch: x["status"] for ch, x in v.items()}
    bad_chars = [ch for ch, s in st.items() if s in BAD]
    warn_chars = [ch for ch, x in v.items() if x.get("warn_play") or x.get("warn_end_turn") or x.get("end_turn_error")]
    noeff = [ch for ch, s in st.items() if s == "no_visible_effect"]
    if bad_chars:
        groups["bad_all" if len(bad_chars) == len(st) else "bad_some"].append((c, bad_chars, v[bad_chars[0]].get("message")))
    if warn_chars:
        x = v[warn_chars[0]]
        groups["warn"].append((c, warn_chars, (x.get("warn_play") or x.get("warn_end_turn") or [x.get("end_turn_error")])[:2]))
    if noeff and len(noeff) == len(st):
        groups["no_effect_all"].append((c, v[noeff[0]].get("type"), v[noeff[0]].get("target")))
    elif noeff:
        groups["no_effect_some"].append((c, noeff))
    if all(s in ("unplayable", "cost_too_high") for s in st.values()):
        groups["never_playable"].append((c, sorted(set(st.values()))))

for g in ("bad_all", "bad_some", "warn", "no_effect_all", "no_effect_some", "never_playable"):
    print(f"\n== {g}: {len(groups[g])}")
    for item in groups[g][:80]:
        print("  ", json.dumps(item, ensure_ascii=False)[:300])
