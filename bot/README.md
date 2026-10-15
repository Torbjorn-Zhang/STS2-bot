# bot

A combat search bot running on the headless engine (`../sts2-cli`), with rules plus simulation outside combat.

- `combat.py`: the turn planner.
  - At each decision point it snapshots and runs a depth-first search over this turn's plays (cards, potions, card prompts), restoring between branches and merging identical states.
  - Leaves are scored by `evaluate`: HP after the enemies' announced attacks, damage dealt, kills, debuffs left on enemies, and lasting powers. Powers are valued per remaining turn, using a fight-length estimate of enemy HP divided by the deck's damage rate.
  - With `rollouts={"k":6,"m":4,"turns":3}`, the top k leaves are re-scored by m **determinized rollouts**. Each rollout plays three more turns with a fast rule policy (`quick_policy`) after the engine's `determinize` command shuffles the draw pile and reseeds every combat RNG.
  - **Fair play:**
    - A branch stops at any action that reveals hidden information, such as drawing or creating cards. The bot executes up to that point and replans with the real result.
    - Rollouts see an imagined future, never the real one.
    - Incoming damage for the current turn comes from the visible intents.
- `decktest.py`: deck decisions by simulation.
  - Covers card rewards, smith upgrades and shop purchases. Each alternative is played forward in a sandbox: take it, walk to the map, heal to full, and fight the act boss plus a hard hallway fight of the act ahead. Those fights are determinized and played by `quick_policy`. Then the sandbox restores.
  - The act boss is public (it is shown on the map), and the hallway fights come from each act's known encounter pool, not the run's actual future.
  - The tier list in `policy.py` acts as a prior on top of these results.
- `policy.py`: map DP with HP projected along the path, the Ironclad tier list with deck-aware adjustments, rest/smith, events, shop, and card prompts outside combat.
- `bossbench.py`: fixed pre-boss saves, replayed under different planner configs (paired, optionally `--reps` determinized repetitions).
- `cardlab.py`: the marginal value of each card in boss fights (bench decks plus one card, paired against the unchanged deck).
- `trace.py`: turn-by-turn trace of a run's costly fights.
- `value.py`, `collect.py`, `train.py`: a learned evaluation (gradient-boosted trees on end-of-turn features). It lost to the hand-written evaluation online (55.8% vs 63.3% act-1 clears), so it is shelved.
- `engine.py`: restarts a worker's engine process once it passes 1.5 GB (`STS2_RECYCLE_MB`).
- `run.py`: plays seeds in parallel (one reused engine process per worker), resumable, and writes `runs/<tag>.jsonl`. `compare.py` does a paired comparison of two runs files with a sign test.

```bash
cd bot && nice -n 19 ../.venv/bin/python run.py 120 --seed-prefix val --workers 6 --rollouts '{"k":6,"m":4,"turns":3}' --tag test
cd bot && nice -n 19 ../.venv/bin/python run.py 60 --acts 3 --ascension 10 --seed-prefix full --workers 6 --rollouts '{"k":6,"m":4,"turns":3}' --tag a10
```

Results (Ironclad, Ascension 0, stop after act 1, the same 120 held-out seeds `val_0..119`, paired comparisons with `compare.py`):

| Version | act-1 clears | notes |
|---|---|---|
| v4: search + rules, card value from card stats | 76 / 120 | 40 of 44 deaths at the boss |
| v6: v4 + learned value model (monotonic GBT) | 67 / 120 | shelved |
| v7: rollouts k6 m4 t3, power valuation, Ironclad tier list, elites when healthy | 81 / 120 | boss wins 74% (v4: 66%); 1.5 elites per run (v4: 0.1) |
| **v9: v7 + deck test for card rewards** | **107 / 120** | vs v7: 34 seeds only v9 clears, 8 only v7, sign test p = 0.0001 |

Boss bench (26 fixed A0 act-1 boss fights): heuristic 15 wins, rollouts 19, rollouts plus power valuation 20.

Full 3-act runs at A0 with v7 (60 seeds): no run cleared act 2; deaths were 33 at bosses, 21 in hallway fights and 6 at elites. Act 2 (the Hive) is the wall: Kaiser Crab (Crusher + Rocket, about 400 HP between them) and hallway fights such as Hunter Killer and Slumbering Beetle.
