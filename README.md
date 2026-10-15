# STS2-bot

Experiments toward an AI for **Slay the Spire 2**. The simulator is the game itself: `sts2-cli` runs the game's own `sts2.dll` headless, with no Godot and no rendering. On top of it, this repo adds process reuse, mid-combat snapshot/restore for search, a validation harness, and (in progress) a combat search bot.

> 中文简介：用游戏自己的 `sts2.dll` 当模拟器（基于 wuhao21/sts2-cli），加上同进程多局、战斗中途快照/恢复、对照真游戏的验证工具，下一步是战斗搜索 bot。

**No game files are included.** You need your own copy of Slay the Spire 2 (Steam). `sts2-cli/setup.sh` copies the game DLLs from your local install into `sts2-cli/lib/`, which is git-ignored, and they stay on your machine. This project is not affiliated with Mega Crit.

## Status

Measured on game version **v0.111.0** (Steam public-beta, build 24724944), macOS arm64:

| Check | Result |
|---|---|
| Lockstep vs. the real game (via the STS2MCP mod) | 7 fights, 106 actions: every state identical. Gold, HP and card rewards identical in every fight entered via the map. |
| Run generation vs. the real game (same seed, same unlock state) | Maps, all 12 RNG streams, encounters, events and bosses identical, except for the game's first-run tutorial overrides. |
| Card sweep (602 cards x 5 characters) | No wrong effects found. 4 cards hung the engine; all fixed. |
| Encounter sweep (95 encounters x 5 characters) | Clean after adding 7 missing Godot stub methods and guarding 3 monster death hooks. |
| Determinism | Same seed replay 15/15, save/load replay 45/45, reused process vs. fresh 50/50. |
| Snapshot / restore | 392 snapshots restored exactly (state + full combat checksum), 309 branch-then-restore tests exact. Restore median about 4 ms. |
| Throughput (random policy, Apple M5) | 1 process: about 400 steps/s. 10 processes: about 2,350 steps/s. |

## Layout

```
sts2-cli/   headless engine adapter. Fork of wuhao21/sts2-cli (MIT), history kept, with our patches.
harness/    validation and benchmark scripts (see harness/README.md)
patches/    STS2MCP-v0.111.0.diff: STS2MCP mod patch for game v0.111.0 (for real-game parity checks)
bot/        combat search bot (in progress)
```

## What changed in sts2-cli

- **Many runs per process.** `start_run` / `load_save` clean up the previous run. Starting takes about 45 ms instead of about 700 ms for a new process.
- **`snapshot` / `restore` / `checksum` / `drop_snapshots`.** Restore rebuilds a checkpoint (run start, loaded save, or the pre-entry save of a map node) and replays the logged commands. This is the same design as the game's own `CombatReplay`. `checksum` uses the game's `NetFullCombatState`.
- **Determinism fixes:**
  - Event effects now settle before state is read.
  - Card-selection re-entrancy fixed: Glam/Replay prompts were lost and combat hung.
  - `load_save` keeps map point flags, "?" room odds and the Neow flag.
  - Profile progress is reset between runs.
- **Engine fixes:**
  - Godot stubs added: `GodotObject.Connect` / `CallDeferred`, `Node.GetIndex`, `Mathf.Log`, `Callable.From<T>`, `CanvasItem` setters.
  - Presentation-only death hooks guarded (Kaiser Crab arms, Soul Nexus).
  - Start-of-turn card prompts are surfaced (Tools of the Trade, Tyranny, Stratagem).
  - Single-enemy cards are unplayable when no enemy is targetable.
  - `set_player` relics are registered correctly.
  - `start_run` accepts `unlock_from_save`.

See `sts2-cli/README.md` for the protocol and `sts2-cli/CLAUDE.md` for the engine conventions.

## Quick start (macOS / Linux)

```bash
# .NET 9 SDK (user-local install is fine) and Python 3.9+
./sts2-cli/setup.sh              # copies the DLLs from your Steam install and builds
dotnet build -c Release sts2-cli/src/Sts2Headless/Sts2Headless.csproj
python3 -m venv .venv && .venv/bin/pip install pytest
cd sts2-cli && ../.venv/bin/python -m pytest -q      # engine tests
cd ../harness && ../.venv/bin/python reuse_check.py 4 Release
```

## Credits

- [wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli) (MIT): the headless engine adapter this builds on.
- [Gennadiyev/STS2MCP](https://github.com/Gennadiyev/STS2MCP) (MIT): the mod used to drive the real game for parity checks.
- Slay the Spire 2 is © Mega Crit. No game code or assets are distributed here.

## License

MIT for the code in this repository (see `LICENSE`). `sts2-cli/` keeps its upstream MIT license (`sts2-cli/LICENSE`).
