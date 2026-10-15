# STS2 v0.111.0 compatibility verification

Tested on macOS ARM64 on 2026-09-07 against Steam public-beta build **24724944**,
game commit **41cef1ea** (installed release metadata).

- Setup and .NET 9 build succeeded (three existing nullable warnings).
- Existing pytest suite: **57 passed**; setup/runner regression checks: **7 passed**.
- Five deterministic batch runs per character, seeds `run_1` through `run_5`.
- All 25 reached normal game-over; no crashes, stalls, protocol errors, or unobserved task exceptions in the final batch logs.
- The simple first-playable-card agent lost every run. These checks establish execution compatibility, not winning strategy or exhaustive card/event coverage.

| Character | Completed | Wins | Final act/floor per seed |
|---|---:|---:|---|
| Ironclad | 5/5 | 0/5 | 1/6, 1/7, 1/15, 1/6, 1/15 |
| Silent | 5/5 | 0/5 | 1/12, 1/7, 1/17, 1/9, 1/5 |
| Defect | 5/5 | 0/5 | 1/13, 1/9, 2/11, 1/11, 1/17 |
| Regent | 5/5 | 0/5 | 1/6, 1/11, 1/17, 1/17, 1/15 |
| Necrobinder | 5/5 | 0/5 | 1/6, 1/7, 1/15, 1/11, 1/5 |

Reproduce after running `./setup.sh`:

```bash
python3 -m pytest -q
for character in Ironclad Silent Defect Regent Necrobinder; do
    python3 python/play_full_run.py 5 "$character" || exit 1
done
```

## Interactive-launch dependency regression

The first validation used `STS2_GAME_DIR` pointing at the Steam installation,
which hid a missing `Sentry.Godot.dll` in `lib/`. The interactive launcher only
searched `lib/` and failed at the game module initializer. Setup now copies this
new dependency; the launcher reruns setup when it is missing, and protocol
errors retain inner exception details.

The game test fixture now uses only `lib/`. Additional tests launch the real
interactive program for all five characters with `STS2_GAME_DIR` unset and
verify automatic repair is triggered for the missing dependency. The reported
`cli_5005` seed successfully reached Neow through the normal Chinese launcher.
All five characters were also replayed five times using only `lib/`: 25/25
normal game-over states. The lib-only full suite passed 69 tests; the additional
auto-repair check passed in a targeted seven-test launcher run. Both `launch.py`
and `python/play.py` reached Neow without a Steam fallback.

The batch runner records action/state JSONL files under `logs/` (gitignored).
Native save/load tests use temporary saves; game DLLs remain local and are not distributed.
