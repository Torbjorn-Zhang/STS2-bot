"""Reusing one headless process across runs must behave like fresh processes."""

import json

from conftest import Game


def _play_first_steps(game, character, seed, steps=40):
    state = game.start(character=character, seed=seed)
    trace = [json.dumps(state, sort_keys=True)]
    state = game.skip_neow(state)
    trace.append(json.dumps(state, sort_keys=True))
    for _ in range(steps):
        dec = state.get("decision")
        if dec == "map_select":
            c = state["choices"][0]
            state = game.act("select_map_node", col=c["col"], row=c["row"])
        elif dec == "combat_play":
            state = game.auto_combat(state)
        elif dec == "card_reward":
            state = game.act("skip_card_reward")
        elif dec in ("game_over", None):
            break
        else:
            state = game.skip_neow(state)
        trace.append(json.dumps(state, sort_keys=True))
    return trace


def test_start_run_can_be_called_twice_in_one_process(game):
    first = game.start(seed="reuse-first")
    assert first["type"] == "decision" and first["decision"] == "event_choice"
    second = game.start(seed="reuse-second")
    assert second["type"] == "decision" and second["decision"] == "event_choice"


def test_reused_process_matches_fresh_processes():
    jobs = [("Ironclad", "reuse-a"), ("Silent", "reuse-b"), ("Necrobinder", "reuse-c")]
    fresh = {}
    for character, seed in jobs:
        g = Game()
        try:
            fresh[seed] = _play_first_steps(g, character, seed)
        finally:
            g.close()
    g = Game()
    try:
        for character, seed in jobs:
            assert _play_first_steps(g, character, seed) == fresh[seed], seed
    finally:
        g.close()
