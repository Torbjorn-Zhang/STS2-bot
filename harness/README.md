# harness

Scripts that check the headless engine (`../sts2-cli`) and measure it. Run them with the repo's Python (`../.venv/bin/python`) from this directory after `sts2-cli/setup.sh` and a Release build.

| Script | What it checks |
|---|---|
| `simclient.py` | Client for the engine's JSON protocol, plus a random legal-action policy. |
| `reuse_check.py N` | Plays the same runs in fresh processes and then back to back in one process. Every state must be identical. |
| `restore_check.py N K P [--godmode]` | Takes random snapshots, restores them, and replays the next K actions. Restores must be exact (state JSON and both checksums), and so must branch-then-restore. `--godmode` sets 9999 HP so runs reach acts 2-3. |
| `determinism.py N` | Same seed with the same actions must replay identically, and save, load, replay must match. |
| `scan_reuse.py` / `scan_report.py` | Card sweep: every card x every character, played in a fixed fight. Flags errors, hangs, exceptions and "no visible effect". |
| `noeffect_check.py` | Counterfactual (play vs. don't play) check for the cards the sweep flagged as having no visible effect. |
| `encounter_scan.py` | Fights every encounter with every character. Flags missing Godot stub methods, exceptions and stalls. |
| `neutralize_check.py` | Checks sts2-cli's hand-written Neutralize against Strike under Strength, Vulnerable, The Boot and Envenom. |
| `bench.py`, `par.py`, `par_reuse.py` | Throughput, with one process per run and with reused processes. |
| `parity_save.py`, `parity_combat.py`, `parity_run.py` | Lockstep comparison with the real game through the STS2MCP mod (see `../patches`). Needs the game running with the mod. |
| `reflect/`, `ildump/` | Small .NET tools: list types and members in `sts2.dll`, and dump the IL calls of a method. |
