"""Deck exam: how strong are the decks a policy brings to act 3, apart from act-3 luck?

Every run that reached act 3 has its act-3-start deck in its record (run.py act_states). The exam
fights that deck, transplanted into a fixed panel of act-3 boss saves (Ascension 10 double boss:
each save's HP, relics and potions, its two final bosses), with the same hidden futures for every
deck (deckspace.py fight). A deck-building change is then judged by
    P(reach act 3) x exam score of the decks that got there
both of which have far less noise than full-run wins (~0.1% at A10).

  deckexam.py make TAG [TAG ...] --out sets.json [--panel 3 --reps 1]   exam fights for those runs
  (deckspace.py fight sets.json --saves data/deckspace_saves ...)       the fights (fleet tasks)
  deckexam.py report F.jsonl [...] --runs TAG [TAG ...]                 exam scores per tag
"""
import argparse
import glob
import json
import os
import statistics as S
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_BOSSES = ("AEONGLASS_BOSS", "QUEEN_BOSS", "TEST_SUBJECT_BOSS")


def load_runs(tag):
    R = {}
    for f in [os.path.join(HERE, "runs", f"{tag}.jsonl")] + glob.glob(os.path.join(HERE, "runs", f"{tag}_c*.jsonl")):
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l)
                R[r["seed"]] = r
    return R


def panel(n_per_boss):
    metas = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "data", "deckspace_saves", "*_a3.save.json")))]
    out = []
    for b in PANEL_BOSSES:
        out += [m["seed"] for m in metas if m.get("boss") == b][:n_per_boss]
    return out


def cmd_make(args):
    saves = panel(args.panel)
    sets = []
    for tag in args.tags:
        for seed, r in load_runs(tag).items():
            st = (r.get("act_states") or {}).get("3")
            if not st:
                continue
            for s in saves:
                for rep in range(100, 100 + args.reps):
                    sets.append({"save": s, "deck": st["deck"], "rep": rep, "family": "exam", "run": f"{tag}/{seed}"})
    with open(args.out, "w") as fh:
        json.dump(sets, fh)
    print(f"{len(sets)} exam fights ({len(saves)} panel saves) -> {args.out}")


def cmd_report(args):
    from deckspace import deck_key
    rows = [json.loads(l) for f in args.files for l in open(f)]
    rows = [r for r in rows if "err" not in r]
    by_deck = {}
    for r in rows:
        by_deck.setdefault(r["dk"], []).append(r)
    for tag in args.runs:
        R = load_runs(tag)
        reach = [r for r in R.values() if (r.get("act_states") or {}).get("3")]
        scores, wins, b2 = [], [], []
        for r in reach:
            ex = by_deck.get(deck_key(r["act_states"]["3"]["deck"]))
            if not ex:
                continue
            scores.append(S.mean(x["graded"] for x in ex))
            wins.append(S.mean(x["won"] for x in ex))
            b2.append(S.mean(x.get("bosses", 1) >= 2 for x in ex))
        n = len(R)
        if not scores:
            print(f"{tag}: no exam rows yet")
            continue
        se = S.stdev(scores) / len(scores) ** 0.5 if len(scores) > 1 else 0
        print(f"{tag:18s} runs {n:4d} reach act 3 {len(reach) / max(1, n):5.1%}  examined {len(scores):3d}  "
              f"exam graded {S.mean(scores):5.1f}±{se:.1f}  boss2 {S.mean(b2):5.1%}  win {S.mean(wins):5.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "report"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--panel", type=int, default=1, help="exam saves per act-3 boss")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args()
    if a.cmd == "make":
        a.tags = a.args
        cmd_make(a)
    else:
        a.files = a.args
        cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
