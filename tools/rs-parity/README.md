# sts2-rs parity (lockstep vs the real v0.111.0 DLL)

sts2-rs is the MIT Rust port of StS2 in github.com/tfriedel/spire-mentor (sparse-cloned to
`local/spire-mentor`, gitignored). It already has a v0.111.0 mode (build 41cef1ea = our game).

- `sts2-rs-local.patch`: our local changes to it (`git apply` inside the clone):
  - `STS2_ARENA_GAME_VERSION` for `run_arena`, `STS2_RS_RAW_SEED=1` (hash the seed string as
    given, like `RunState.CreateForTest`), `relic_counters()` on the PyO3 RunManager, PyO3 abi3-py39;
  - fixes found by lockstep: the side-turn-start walk is skipped once a `SetupPlayerTurn` kill
    ended the combat (C# `IterateCombatHookListeners` yields nothing when `IsOverOrEnding`), and
    Pael's Tears clears its leftover-energy latch at combat end.
- `parity_start.py N`: run-start parity (act-1 boss, Neow offers, whole act-1 map) for full_0..N-1.
- `lockstep.py SEED...`: plays one deterministic policy in both engines and stops at the first
  state difference. `LOCKSTEP_GOD=9999` gives both players 9999 HP for whole-run coverage,
  `LOCKSTEP_PICK=skip` skips card rewards. Our side needs engine config Fix11.
- `batch.py LO HI`: lockstep over full_LO..full_HI-1 in parallel (BATCH_PROCS, BATCH_OUT).

Build the extension: `PYO3_PYTHON=.venv/bin/python cargo build --release -p sts2-py --features
pyo3/extension-module` in `sts2-rs`, then copy `target/release/libsts2_rs.dylib` to
`local/rs-py/sts2_rs.abi3.so`.
