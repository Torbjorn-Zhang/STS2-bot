# Vendored: spire-mentor's sts2_hint (MIT)

`sts2_hint/` is copied from tfriedel/spire-mentor (commit 52bfee16,
`sts2-pycli/python/sts2_hint`, MIT licence in `LICENSE-spire-mentor`). It translates a
Sts2Headless `combat_play` board into the wire dict `sts2_rs.state_from_dict` reads. Our engine
emits that board with the Fix12 `combat_wire` command (sts2-cli `CombatWire.cs`, ported from the
same repository). `runtime_paths.py` is our shim for the package's path helper.

One local change, for our venv's Python 3.9: `schema.py` takes `NotRequired` / `TypedDict` from
`typing_extensions` when `typing` lacks them, and its one functional TypedDict spells `X | None`
as `Optional[X]`. `sts2_client.py` is a shim with the original's game-version helpers.
