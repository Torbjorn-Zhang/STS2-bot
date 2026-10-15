"""Thin client for the sts2-cli headless engine (real sts2.dll + Godot stubs).

Launches the built Sts2Headless.dll directly (no `dotnet run`), speaks the
line-delimited JSON protocol, and records per-command latency.
"""

import json
import os
import queue
import random
import shutil
import subprocess
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "sts2-cli")
DOTNET = os.environ.get("DOTNET") or next(
    (p for p in (os.path.expanduser("~/.dotnet/dotnet"), os.path.expanduser("~/.dotnet-arm64/dotnet")) if os.path.exists(p)),
    shutil.which("dotnet") or "dotnet")


def engine_dll(config="Debug"):
    return os.path.join(CLI, "src", "Sts2Headless", "bin", config, "net9.0", "Sts2Headless.dll")


class Sim:
    def __init__(self, config="Debug", timeout=30.0, capture_stderr=False):
        self.timeout = timeout
        self.latencies = []  # (cmd_name, seconds)
        self.stderr_lines = []
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            [DOTNET, engine_dll(config)],
            cwd=CLI,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        if capture_stderr:
            threading.Thread(target=self._pump_err, daemon=True).start()
        ready = self._read()
        if ready.get("type") != "ready":
            raise RuntimeError(f"bad ready message: {ready}")
        self.startup_s = time.perf_counter() - t0

    def _pump(self):
        for line in self.proc.stdout:
            self._q.put(line)
        self._q.put(None)

    def _pump_err(self):
        for line in self.proc.stderr:
            self.stderr_lines.append(line.rstrip("\n"))

    def _read(self):
        while True:
            try:
                line = self._q.get(timeout=self.timeout)
            except queue.Empty:
                raise TimeoutError("engine did not answer in time") from None
            if line is None:
                raise RuntimeError("engine exited")
            if line.startswith("{"):
                return json.loads(line)

    def send(self, cmd):
        t0 = time.perf_counter()
        self.proc.stdin.write(json.dumps(cmd) + "\n")
        self.proc.stdin.flush()
        resp = self._read()
        name = cmd.get("action") or cmd.get("cmd")
        self.latencies.append((name, time.perf_counter() - t0))
        return resp

    def act(self, action, **args):
        cmd = {"cmd": "action", "action": action}
        if args:
            cmd["args"] = args
        return self.send(cmd)

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def legal_actions(state):
    """Enumerate (action, args) pairs for a decision point. Shops are left immediately."""
    d = state.get("decision")
    if d == "map_select":
        return [("select_map_node", {"col": c["col"], "row": c["row"]}) for c in state.get("choices", [])]
    if d == "combat_play":
        acts = [("end_turn", {})]
        energy = state.get("energy", 0)
        alive = [e["index"] for e in state.get("enemies", []) if e.get("hp", 0) > 0]
        for c in state.get("hand", []):
            if not c.get("can_play"):
                continue
            cost = c.get("cost", 0)
            if isinstance(cost, int) and cost > energy:
                continue
            if c.get("target_type") == "AnyEnemy" and len(alive) > 1:
                acts += [("play_card", {"card_index": c["index"], "target_index": t}) for t in alive]
            else:
                acts.append(("play_card", {"card_index": c["index"]}))
        return acts
    if d == "event_choice":
        opts = [o for o in state.get("options", []) if not o.get("is_locked")]
        if not opts:
            return [("leave_room", {})]
        return [("choose_option", {"option_index": o["index"]}) for o in opts]
    if d == "rest_site":
        opts = [o for o in state.get("options", []) if o.get("is_enabled", True)]
        if not opts:
            return [("leave_room", {})]
        return [("choose_option", {"option_index": o["index"]}) for o in opts]
    if d == "card_reward":
        acts = [("select_card_reward", {"card_index": c["index"]}) for c in state.get("cards", [])]
        if state.get("can_skip", True) or not acts:
            acts.append(("skip_card_reward", {}))
        return acts
    if d == "bundle_select":
        return [("select_bundle", {"bundle_index": b["index"]}) for b in state.get("bundles", [])] or [("proceed", {})]
    if d == "card_select":
        cards = state.get("cards", [])
        lo = state.get("min_select", 1) or 0
        acts = []
        if cards:
            n = max(1, lo)
            acts.append(("select_cards", {"indices": ",".join(str(i) for i in range(min(n, len(cards))))}))
            if n == 1:
                acts += [("select_cards", {"indices": str(c.get("index", i))}) for i, c in enumerate(cards)][1:]
        if lo == 0 or not cards:
            acts.append(("skip_select", {}))
        return acts
    if d == "shop":
        return [("leave_room", {})]
    return [("proceed", {})]


def play_random_run(sim, character, seed, rng, max_steps=3000, end_turn_bias=0.15, trace=None):
    """Play one run with a uniformly random legal policy. Returns a summary dict."""
    state = sim.send({"cmd": "start_run", "character": character, "seed": seed})
    steps = 0
    t0 = time.perf_counter()
    errors = 0
    while steps < max_steps:
        if state.get("type") == "error":
            errors += 1
            if errors > 20:
                return {"ok": False, "error": state.get("message"), "steps": steps}
        d = state.get("decision")
        if d == "game_over":
            return {"ok": True, "victory": state.get("victory", False), "steps": steps,
                    "act": state.get("context", {}).get("act"), "floor": state.get("context", {}).get("floor"),
                    "seconds": time.perf_counter() - t0, "errors": errors}
        acts = legal_actions(state)
        if d == "combat_play" and len(acts) > 1 and rng.random() > end_turn_bias:
            acts = acts[1:]
        action, args = rng.choice(acts)
        if trace is not None:
            trace.append((action, args))
        state = sim.act(action, **args)
        steps += 1
    return {"ok": False, "error": "max_steps", "steps": steps}
