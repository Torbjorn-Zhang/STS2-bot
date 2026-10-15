"""snapshot / restore / checksum: restoring must reproduce the exact state."""

import json


def _canon(state):
    return json.dumps(state, sort_keys=True)


def _into_combat(game, seed):
    state = game.skip_neow(game.start(seed=seed))
    while state.get("decision") == "map_select":
        c = state["choices"][0]
        state = game.act("select_map_node", col=c["col"], row=c["row"])
        state = game.skip_neow(state) if state.get("decision") not in ("combat_play", "map_select") else state
    assert state["decision"] == "combat_play"
    return state


def test_restore_mid_combat_reproduces_state_and_continuation(game):
    state = _into_combat(game, "snap-mid")
    state = game.auto_combat(state)
    snap = game.send({"cmd": "snapshot"})
    assert snap["type"] == "snapshot"
    before = game.send({"cmd": "checksum"})
    assert "combat" in before

    # original continuation
    original = []
    s = state
    for _ in range(6):
        if s.get("decision") != "combat_play":
            break
        s = game.auto_combat(s)
        original.append(_canon(s))

    restored = game.send({"cmd": "restore", "id": snap["id"]})
    assert _canon(restored) == _canon(state)
    after = game.send({"cmd": "checksum"})
    assert after["combat"] == before["combat"] and after["run"] == before["run"]

    s = restored
    for expected in original:
        s = game.auto_combat(s)
        assert _canon(s) == expected


def test_branching_does_not_leak_into_next_restore(game):
    state = _into_combat(game, "snap-branch")
    snap = game.send({"cmd": "snapshot"})
    first = game.act("end_turn")
    game.send({"cmd": "restore", "id": snap["id"]})
    game.auto_combat(state)  # a different branch: play a card instead
    game.send({"cmd": "restore", "id": snap["id"]})
    again = game.act("end_turn")
    assert _canon(again) == _canon(first)


def test_unknown_snapshot_is_an_error(game):
    game.start(seed="snap-err")
    r = game.send({"cmd": "restore", "id": 999})
    assert r["type"] == "error"


def test_determinize_is_reproducible_and_undone_by_restore(game):
    state = _into_combat(game, "snap-detz")
    snap = game.send({"cmd": "snapshot"})
    before = game.send({"cmd": "checksum"})

    def future(seed):
        game.send({"cmd": "restore", "id": snap["id"]})
        if seed is not None:
            r = game.send({"cmd": "determinize", "seed": seed})
            assert r["type"] == "ok", r
        out = []
        s = state
        for _ in range(3):
            if s.get("decision") != "combat_play":
                break
            s = game.act("end_turn")
            out.append(_canon(s))
        return out

    a, b = future(11), future(11)
    assert a == b                                     # same seed, same hidden future
    assert any(future(s) != a for s in (12, 13, 14))  # other seeds change it
    game.send({"cmd": "restore", "id": snap["id"]})
    after = game.send({"cmd": "checksum"})
    assert after["combat"] == before["combat"] and after["run"] == before["run"]
