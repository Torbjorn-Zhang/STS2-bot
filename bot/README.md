# bot

A combat search bot running on the headless engine (`../sts2-cli`), with simple rules outside combat.

- `combat.py`: the turn planner. At each decision point it snapshots and runs a depth-first search over this turn's plays (cards, potions, card prompts), restoring between branches and merging identical states. It scores the end-of-turn state: HP after the enemies' announced attacks, damage dealt, kills, debuffs left on enemies, and lasting player powers.
  - **Fair play:** a branch stops at any action that reveals hidden information (draws or creates cards, shows draw-pile cards). The bot executes up to that point and replans with the real result. The enemy turn is never simulated; incoming damage comes from the visible intents.
- `policy.py`: map DP, card rewards, rest/smith, events, Neow, card prompts outside combat.
- `run.py`: plays seeds in parallel (one reused engine process per worker) and writes `runs/<tag>.jsonl`.

```bash
cd bot && ../.venv/bin/python run.py 30 --workers 6 --tag test     # Ironclad A0, stop after act 1
```

Results so far (Ironclad, Ascension 0, stop after act 1). Rules were tuned by looking at losses on the `bot_*` seeds; the `val_*` seeds were never looked at:

| Version | tuning seeds `bot_0..59` | held-out seeds `val_0..59` |
|---|---|---|
| random legal actions | 0 / 30 (first 30 only) | not run |
| v1 search + rules | 5 / 30 (first 30 only) | not run |
| v3 + potions in elite/boss fights, skip act-1 elites, heal before the boss | 29 / 60 | 32 / 60 |
| v4 + card value from card stats, shop (remove a Strike, buy cards and a potion), smith priority | **48 / 60** | **35 / 60** |

So most of v4's gain on the tuning seeds does not carry over, and 60 runs are too few to judge rule changes (roughly ±9 points). Next step: tune the evaluation weights automatically on large seed sets, with separate validation seeds, then learn the evaluation itself from self-play.

Most remaining losses are to act-1 bosses, especially the Ceremonial Beast. It gains Strength every turn, stuns at 150 HP, and its Ringing debuff limits you to one card per turn, so it is a damage race.
