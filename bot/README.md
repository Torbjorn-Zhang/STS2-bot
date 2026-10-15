# bot

A combat search bot running on the headless engine (`../sts2-cli`), with simple rules outside combat.

- `combat.py`: the turn planner. At each decision point it snapshots and runs a depth-first search over this turn's plays (cards, potions, card prompts), restoring between branches and merging identical states. It scores the end-of-turn state: HP after the enemies' announced attacks, damage dealt, kills, debuffs left on enemies, and lasting player powers.
  - **Fair play:** a branch stops at any action that reveals hidden information (draws or creates cards, shows draw-pile cards). The bot executes up to that point and replans with the real result. The enemy turn is never simulated; incoming damage comes from the visible intents.
- `policy.py`: map DP, card rewards, rest/smith, events, Neow, card prompts outside combat.
- `run.py`: plays seeds in parallel (one reused engine process per worker) and writes `runs/<tag>.jsonl`.

```bash
cd bot && ../.venv/bin/python run.py 30 --workers 6 --tag test     # Ironclad A0, stop after act 1
```

Results so far (Ironclad, Ascension 0, act 1, seeds `bot_0..29`):

| Version | Act 1 cleared |
|---|---|
| random legal actions | 0 / 30 |
| v1 search + rules | 5 / 30 |
| v2 + potions in elite/boss fights, skip act-1 elites, heal before the boss | 17 / 30 |

Most remaining losses are to the Ceremonial Beast boss: it gains Strength every turn, and the one-turn evaluation does not race it hard enough.
