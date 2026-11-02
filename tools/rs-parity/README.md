# sts2-rs parity (lockstep vs the real v0.111.0 DLL)

sts2-rs is the MIT Rust port of StS2 in github.com/tfriedel/spire-mentor (sparse-cloned to
`local/spire-mentor`, gitignored). It already has a v0.111.0 mode (build 41cef1ea = our game).

- `sts2-rs-local.patch`: our local changes to it (`git diff` of the clone; `patch -p1` in a pristine
  checkout of 52bfee16):
  - knobs and bindings: `STS2_ARENA_GAME_VERSION` for `run_arena`, `STS2_RS_RAW_SEED=1` (hash the
    seed string as given, like `RunState.CreateForTest`), PyO3 abi3-py39, `relic_counters()` and
    `next_ancient()` on the PyO3 RunManager, `player_hp()` on PyCombatState, `removal_price` in the
    shop inventory, and `MctsCombatAgent.step` searches without the GIL (run.py's worker threads
    search in parallel);
  - fixes found by lockstep: the side-turn-start walk is skipped once a `SetupPlayerTurn` kill
    ended the combat (C# `IterateCombatHookListeners` yields nothing when `IsOverOrEnding`), Pael's
    Tears clears its leftover-energy latch at combat end, Sunken Treasury rolls, Gremlin Merc's gold
    share, Philosopher's Stone strength on spawned enemies, Thunderclap's kills are deferred, Vexing
    Puzzlebox fires after the player turn starts, Bowler Hat's x1.25 on run-level gold.
- `parity_start.py N`: run-start parity (act-1 boss, Neow offers, whole act-1 map) for full_0..N-1.
- `lockstep.py SEED...`: plays one deterministic policy in both engines and stops at the first
  state difference. `LOCKSTEP_GOD=9999` gives both players 9999 HP for whole-run coverage,
  `LOCKSTEP_PICK=skip` skips card rewards. Our side needs engine config Fix11.
- `batch.py LO HI`: lockstep over full_LO..full_HI-1 in parallel (BATCH_PROCS, BATCH_OUT).

Build the extension: `PYO3_PYTHON=.venv/bin/python cargo build --release -p sts2-py --features
pyo3/extension-module` in `sts2-rs`, then copy `target/release/libsts2_rs.dylib` to
`local/rs-py/sts2_rs.abi3.so`.

## RsSim: training on sts2-rs

`bot/rssim.py` renders sts2-rs in our engine's JSON protocol, so `run.py`, the deck networks and the
MC plays run unchanged while sts2-rs's MCTS plays the fights: `run.py ... --rs SIMS,BOSS_SIMS`
(e.g. `--rs 250,1000`; ~3 s per A10 run on one Mac core vs minutes on our engine).

- `bot/rscheck.py SEED...`: one deterministic policy in our engine (Fix11) and in
  `RsSim(expose_combat=True)`, every state the deck policy reads compared field by field
  (`RSCHECK_GOD=9999` for whole runs). Remaining differences are engine divergences (see lockstep).
- Card / relic tables come from our engine (Fix11 `card_db` / `relic_db`, derived from the game DLL,
  not in git): `python bot/rssim.py --make-db [DIR]`; looked up in `STS2_RS_DATA`, `bot/data`, then
  the Apollo Lustre dir.
- `apollo_install.sh`: rustup + the patched source + the builds on the Apollo pods' Lustre dir
  (`/scratch/xkzhang/slzhang_cvpr27/rs`: `py/sts2_rs.abi3.so`, `bin/run_arena`, `data/`); a changed
  patch re-extracts the pristine source and re-applies it.

## rsfight: sts2-rs's MCTS playing fights in our engine

`bot/rsfight.py` (`run.py ... --engine-config Fix12 --rs-fight SIMS,BOSS_SIMS`): at every combat
decision our engine's board (Fix12 `combat_wire`, ported from spire-mentor's Sts2Headless) goes
through spire-mentor's `sts2_hint` adapter (vendored in `bot/vendor`, MIT) into
`sts2_rs.state_from_dict`, is installed in a host RunManager (`debug_enter_combat`) and the
deployed run agent (`MctsCombatAgent`, honest beliefdraw search) picks the action, which is played
in our engine. The prompt a play opens is answered on the host when its pending choice offers the
same cards; whatever the agent cannot take goes to our planner.

