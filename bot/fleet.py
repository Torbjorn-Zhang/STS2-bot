"""Distributed training on the user's Windows machines; this Mac only schedules and copies.

The coordinator keeps a queue of jobs (self-play chunks, evaluation chunks, network updates,
expert-world evaluation chunks) and runs them over SSH on the machines in deploy/fleet.json,
at below-normal CPU priority, in process slots (several processes per machine: one Python
process stops scaling at ~6 worker threads, see benchcompute.py). A machine can leave at any
time and its work moves elsewhere:
  * it stops answering (shut down, asleep, offline): its running jobs go back to the queue;
    it rejoins by itself when it answers again
  * drained: `fleet.py drain HOST`, or a PAUSE file in its root (the desktop "STS2 pause"
    script makes one): our processes there are killed at once and it gets no more work
    until `fleet.py undrain HOST` / the PAUSE file is removed
  * auto-yield ("yield_gpu" hosts): another program keeps the GPU busy (a game) -> drained
    until the GPU has been quiet for YIELD_QUIET seconds
Network updates go to a free GPU machine (CUDA) while the CPUs keep playing.

    fleet.py run                       the coordinator (state in runs/fleet/, resumes after a restart)
    fleet.py deckrl --tag decki --start 6 --init decki:5 [--iters 40 --runs 4000 ...]
    fleet.py expert --deck-nets models/frozen/v42.json --tag v42_a10 [--n 240]
    fleet.py status | drain HOST | undrain HOST | stop
"""

import argparse
import base64
import fnmatch
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FLEET = os.path.join(HERE, "runs", "fleet")
STATE = os.path.join(FLEET, "state.json")
CONFIG = os.path.join(REPO, "deploy", "fleet.json")
POLL = 10            # seconds between polls
DOWN_AFTER = 3       # failed polls before a host counts as gone
GPU_BUSY = 40        # % GPU utilisation by other programs that makes a yield_gpu host yield
YIELD_QUIET = 600    # seconds of a quiet GPU before such a host gets work again
MAX_TRIES = 4
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10"]


def log(msg):
    line = time.strftime("%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(os.path.join(FLEET, "fleet.log"), "a") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------- remote primitives (Windows)

def ps(host, script, timeout=60):
    """Run a PowerShell script on the host; returns stdout (raises on failure)."""
    pre = '[Console]::OutputEncoding=[Text.Encoding]::UTF8; $ProgressPreference="SilentlyContinue"\n'
    enc = base64.b64encode((pre + script).encode("utf-16-le")).decode()
    res = subprocess.run(SSH + [host["ssh"], f"powershell -NoProfile -NonInteractive -EncodedCommand {enc}"],
                         capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"ps rc {res.returncode}: {res.stderr[-300:]}")
    return res.stdout


def sftp(host, lines, timeout=900):
    if not lines:
        return
    with tempfile.NamedTemporaryFile("w", suffix=".sftp", delete=False) as fh:
        fh.write("\n".join(lines) + "\n")
        batch = fh.name
    try:
        res = subprocess.run(["sftp", "-C", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-b", batch,
                              host["ssh"]], capture_output=True, text=True, timeout=timeout)
    finally:
        os.remove(batch)
    if res.returncode != 0:
        raise RuntimeError(f"sftp rc {res.returncode}: {(res.stderr or res.stdout)[-300:]}")


def rpath(host, rel=""):
    """Remote path for sftp ("/D:/sts2/bot/<rel>")."""
    return "/" + host["root"].replace("\\", "/").rstrip("/") + "/bot/" + rel.replace("\\", "/")


def wroot(host):
    return host["root"].replace("/", "\\").rstrip("\\")


KILL_PS = r"""
$root = '%ROOT%'
function Ours { Get-CimInstance Win32_Process | Where-Object { ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($root, 'OrdinalIgnoreCase')) -or ($_.Name -eq 'cmd.exe' -and $_.CommandLine -like "*$root\bot\fleet\jobs\*") } }
for ($round = 0; $round -lt 3; $round++) {
  $left = @(Ours)
  if ($left.Count -eq 0) { break }
  $left | Where-Object { $_.Name -eq 'cmd.exe' } | ForEach-Object { taskkill /PID $_.ProcessId /T /F 2>&1 | Out-Null }
  @(Ours) | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Milliseconds 500
}
"killed, left $(@(Ours).Count)"
"""

POLL_PS = r"""
$root = '%ROOT%'; $ids = @(%IDS%)
$alive = @{}
Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" | ForEach-Object {
  if ($_.CommandLine -match 'fleet\\jobs\\([\w\.\-]+)\.cmd') { $alive[$matches[1]] = $true } }
$jobs = @{}
foreach ($id in $ids) {
  $d = "$root\bot\fleet\jobs\$id"
  if (Test-Path "$d.rc") {
    $rc = (Get-Content "$d.rc" -Raw).Trim()
    $out = if (Test-Path "$d.out") { Get-Content "$d.out" -Tail 1 } else { '' }
    $err = if ($rc -ne '0' -and (Test-Path "$d.err")) { (Get-Content "$d.err" -Tail 12) -join "`n" } else { '' }
    $jobs[$id] = @{ state = 'done'; rc = $rc; out = "$out"; err = $err }
  } elseif ($alive[$id]) { $jobs[$id] = @{ state = 'running' } }
  else { $jobs[$id] = @{ state = 'lost' } }
}
$gpu = $null
try { $gpu = [int]((nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) | Select-Object -First 1) } catch {}
@{ pause = (Test-Path "$root\PAUSE"); jobs = $jobs; gpu = $gpu; ours = $alive.Count } | ConvertTo-Json -Compress -Depth 4
"""

LAUNCH_PS = r"""
$root = '%ROOT%'; $ids = @(%IDS%)
$su = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{PriorityClass = [uint32]16384; ShowWindow = [uint16]0}
$pids = @{}
foreach ($id in $ids) {
  $d = "$root\bot\fleet\jobs\$id"
  Remove-Item "$d.rc","$d.out","$d.err" -ErrorAction SilentlyContinue
  $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "cmd.exe /c $d.cmd"; CurrentDirectory = "$root\bot"; ProcessStartupInformation = $su }
  $pids[$id] = [int]$r.ProcessId
}
$pids | ConvertTo-Json -Compress
"""


def job_cmd(host, job):
    root = wroot(host)
    py = root + ("\\.venv-cu" if job.get("gpu_now") else "\\.venv") + "\\Scripts\\python.exe"
    env = {"DOTNET": root + "\\dotnet\\dotnet.exe", "DOTNET_ROOT": root + "\\dotnet", "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
           "OMP_NUM_THREADS": "1", "NETS_DEVICE": "cuda" if job.get("gpu_now") else "cpu", "PYTHONUNBUFFERED": "1"}
    jid = job["id"]
    argv = [str(host["workers"]) if a == "%W%" else str(a) for a in job["argv"]]
    lines = ["@echo off"] + [f"set {k}={v}" for k, v in env.items()] + [f"cd /d {root}\\bot"] + [
        f'if exist "{o.replace("/", chr(92))}" del /q "{o.replace("/", chr(92))}"' for o in job["outputs"]] + [
        f'"{py}" {subprocess.list2cmdline(argv)} > fleet\\jobs\\{jid}.out 2> fleet\\jobs\\{jid}.err',
        f"> fleet\\jobs\\{jid}.rc echo %ERRORLEVEL%"]
    return "\r\n".join(lines) + "\r\n"


# ---------------------------------------------------------------- pipelines

def merge_selfplay(sums):
    ok = [s for s in sums if s]
    runs = sum(s["runs"] for s in ok)
    good = sum(s["runs"] - s["errors"] for s in ok)
    return {"runs": runs, "errors": sum(s["errors"] for s in ok),
            "act1_clears": sum(s["act1_clears"] for s in ok), "act2_clears": sum(s["act2_clears"] for s in ok),
            "wins": sum(s["wins"] for s in ok),
            "mean_floor": round(sum(s["mean_floor"] * (s["runs"] - s["errors"]) for s in ok) / max(1, good), 2)}


class DeckRL:
    """cardloop.py on the fleet. Iteration i: self-play chunks with the current networks ->
    one update of every network (GPU host) -> evaluation chunks with the new networks
    (queued behind the next iteration's self-play, so the CPUs never wait for training)."""

    KINDS = ("card", "rest", "event", "map", "shop")

    def __init__(self, spec, st=None):
        self.spec = spec
        self.name = spec["tag"]
        self.st = st or {"it": spec["start"], "cur": spec["init"], "phase": "start", "evals": {}, "sp": {}}

    def selfplay_args(self, n, start, prefix, out, models, temperature):
        a = ["selfplay.py", n, "--policy", self.spec["combat"], "--temperature", 0, "--workers", "%W%",
             "--seed-prefix", prefix, "--seed-start", start, "--calibrate"]
        for k in self.KINDS:
            if models.get(k):
                a += [f"--{k}-policy", models[k]] + ([f"--{k}-temperature", temperature] if temperature else [])
        return a + (["--out", out, "--no-combat-records"] if out else ["--out", "fleet/null.jsonl", "--no-record"])

    def inputs(self, models):
        return [self.spec["combat"]] + [m for m in models.values() if m]

    def start_selfplay(self, fl, it):
        sp = self.spec
        jobs = []
        for k, s in enumerate(range(0, sp["runs"], sp["chunk"])):
            n = min(sp["chunk"], sp["runs"] - s)
            out = f"fleetdata/{self.name}_it{it}_c{k}.jsonl"
            jobs.append(fl.add_job(self.name, f"{self.name}{it}_sp{k}", "selfplay", 1,
                                   self.selfplay_args(n, s, f"{self.name}{it}", out, self.st["cur"], sp["temperature"]),
                                   self.inputs(self.st["cur"]), [out]))
        self.st["sp"][str(it)] = {"jobs": jobs, "t0": time.time()}
        self.st["phase"] = "selfplay"
        log(f"{self.name}: iteration {it} self-play, {len(jobs)} chunks")

    def start_eval(self, fl, it, models, sp_summary):
        sp = self.spec
        jobs = [fl.add_job(self.name, f"{self.name}{it}_ev{k}", "eval", 2,
                           self.selfplay_args(min(sp["eval_chunk"], sp["eval_runs"] - s), s, "cardeval", None, models, 0),
                           self.inputs(models), [])
                for k, s in enumerate(range(0, sp["eval_runs"], sp["eval_chunk"]))]
        self.st["evals"][str(it)] = {"jobs": jobs, "models": models, "selfplay": sp_summary, "t0": time.time()}

    def step(self, fl):
        st, sp = self.st, self.spec
        if st["phase"] == "start":
            if sp["start"] == 1 or sp.get("eval_init"):
                self.start_eval(fl, sp["start"] - 1, st["cur"], None)
            self.start_selfplay(fl, st["it"])
        elif st["phase"] == "selfplay":
            it = st["it"]
            ids = st["sp"][str(it)]["jobs"]
            if all(fl.jobs[j]["state"] in ("done", "failed") for j in ids):
                done = [fl.jobs[j] for j in ids if fl.jobs[j]["state"] == "done"]
                summary = merge_selfplay([j["result"] for j in done])
                summary["wall_s"] = round(time.time() - st["sp"][str(it)]["t0"], 1)
                st["sp"][str(it)]["summary"] = summary
                data = [o for j in done for o in j["outputs"]]
                argv = ["fleetjob.py", "train"] + data + ["--tag", self.name, "--it", it, "--temperature", sp["temperature"]]
                for k in self.KINDS:
                    if st["cur"].get(k):
                        argv += [f"--{k}", st["cur"][k]]
                outs = [f"models/{self.name}_{k}{it}.pt" for k in self.KINDS if st["cur"].get(k)]
                st["train"] = fl.add_job(self.name, f"{self.name}{it}_train", "train", 0, argv,
                                         self.inputs(st["cur"]) + data, outs, gpu=True)
                st["phase"] = "train"
                log(f"{self.name}: iteration {it} self-play done {json.dumps(summary)}; training")
        elif st["phase"] == "train":
            j = fl.jobs[st["train"]]
            it = st["it"]
            if j["state"] == "failed":
                log(f"{self.name}: training {it} failed; pipeline stopped")
                st["phase"] = "failed"
            elif j["state"] == "done":
                new = dict(st["cur"])
                new.update({k: f"models/{self.name}_{k}{it}.pt" for k in j["result"]["models"]})
                log(f"{self.name}: iteration {it} trained on {j['host']} ({j['result'].get('device')}, "
                    f"{j['result']['timing'].get('total')} s)")
                self.start_eval(fl, it, new, st["sp"][str(it)].get("summary"))
                fl.cleanup(f"fleetdata/{self.name}_it{it}_c*.jsonl")
                st["cur"] = new
                if it + 1 < sp["start"] + sp["iters"]:
                    st["it"] = it + 1
                    self.start_selfplay(fl, it + 1)
                else:
                    st["phase"] = "finishing"
        for it, ev in list(st["evals"].items()):
            if all(fl.jobs[j]["state"] in ("done", "failed") for j in ev["jobs"]):
                res = merge_selfplay([fl.jobs[j]["result"] for j in ev["jobs"] if fl.jobs[j]["state"] == "done"])
                res["wall_s"] = round(time.time() - ev["t0"], 1)
                row = {"iter": int(it), **{k: ev["models"].get(k) for k in self.KINDS}, "selfplay": ev["selfplay"],
                       "eval": res, "fleet": True}
                with open(os.path.join(HERE, "runs", f"{self.name}.jsonl"), "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                log(f"{self.name}: iteration {it} eval {json.dumps(res)}")
                del st["evals"][it]
        if st["phase"] == "finishing" and not st["evals"]:
            st["phase"] = "finished"

    def finished(self):
        return self.st["phase"] in ("finished", "failed", "cancelled")


class Expert:
    """Expert-world (planner) evaluation, seeds split into chunks; rows land in runs/<tag>.jsonl."""

    def __init__(self, spec, st=None):
        self.spec = spec
        self.name = spec["tag"]
        self.st = st or {"phase": "start"}

    def step(self, fl):
        sp, st = self.spec, self.st
        if st["phase"] == "start":
            with open(os.path.join(HERE, sp["deck_nets"])) as fh:
                nets = [v for v in json.load(fh).values() if isinstance(v, str) and v.endswith(".pt")]
            # seed ranges: contiguous runs of the wanted seeds, at most `chunk` per job
            want = sp.get("seeds") or list(range(sp["start"], sp["n"]))
            ranges = []
            for i in want:
                if ranges and ranges[-1][1] == i and ranges[-1][1] - ranges[-1][0] < sp["chunk"]:
                    ranges[-1][1] = i + 1
                else:
                    ranges.append([i, i + 1])
            jobs = []
            for k, (s, e) in enumerate(ranges):
                argv = ["run.py", e, "--seed-start", s, "--acts", 3, "--ascension", 10, "--seed-prefix", sp["prefix"],
                        "--workers", "%W%", "--rollouts", sp["rollouts"], "--deck-nets", sp["deck_nets"],
                        "--tag", f"{self.name}_c{k}"] + sp.get("extra", [])
                jobs.append(fl.add_job(self.name, f"{self.name}_c{k}", "expert", sp.get("prio", 1.5), argv,
                                       [sp["deck_nets"]] + nets, [f"runs/{self.name}_c{k}.jsonl"]))
            st.update(phase="running", jobs=jobs, t0=time.time())
            log(f"{self.name}: expert evaluation, {len(jobs)} chunks")
        elif st["phase"] == "running":
            if all(fl.jobs[j]["state"] in ("done", "failed") for j in st["jobs"]):
                rows = []
                for j in st["jobs"]:
                    p = os.path.join(HERE, fl.jobs[j]["outputs"][0])
                    if fl.jobs[j]["state"] == "done" and os.path.exists(p):
                        with open(p) as fh:
                            rows += [json.loads(l) for l in fh if l.strip()]
                        os.remove(p)
                with open(os.path.join(HERE, "runs", f"{self.name}.jsonl"), "a") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")
                n = len(rows)
                summ = {"runs": n, "act1": sum(1 for r in rows if (r.get("cleared_acts") or 0) >= 1),
                        "act2": sum(1 for r in rows if (r.get("cleared_acts") or 0) >= 2),
                        "wins": sum(1 for r in rows if r.get("victory")),
                        "mean_floor": round(sum(((r.get("act") or 1) - 1) * 17 + (r.get("floor") or 0) for r in rows) / max(1, n), 2),
                        "wall_s": round(time.time() - st["t0"], 1)}
                st.update(phase="finished", summary=summ)
                log(f"{self.name}: expert evaluation done {json.dumps(summ)}")

    def finished(self):
        return self.st["phase"] == "finished"


class Bench:
    """Fight-bench evaluation (bossbench.py eval) of several planner configs; the saves are
    split into shards, one job per (config, shard). Rows land in runs/<tag>_<config>.jsonl."""

    def __init__(self, spec, st=None):
        self.spec = spec
        self.name = spec["tag"]
        self.st = st or {"phase": "start"}

    def step(self, fl):
        sp, st = self.spec, self.st
        if st["phase"] == "start":
            bdir = f"data/bossbench_{sp['bench']}"
            shards = {}
            for p in sorted(glob.glob(os.path.join(HERE, bdir, "*.save.json"))):
                with open(p) as fh:
                    m = json.load(fh)
                if sp.get("room") and m.get("room", "Boss") != sp["room"]:
                    continue
                k = int(hashlib.md5(m["seed"].encode()).hexdigest(), 16) % sp["shards"]   # = bossbench.shard_of
                base = os.path.basename(m["path"])
                shards.setdefault(k, []).extend([f"{bdir}/{base}", f"{bdir}/{base}.json"])
            jobs = []
            for c, cfg in sp["configs"].items():
                for k, files in sorted(shards.items()):
                    out = f"runs/{self.name}_{c}_s{k}.jsonl"
                    # "nodes": the planner's search size, "weights": combat.W overrides
                    roll = {a: v for a, v in cfg.items() if a not in ("nodes", "weights")}
                    argv = ["bossbench.py", "eval", "--tag", sp["bench"], "--rollouts", json.dumps(roll, separators=(",", ":")),
                            "--reps", sp.get("reps", 1), "--shard", f"{k}/{sp['shards']}", "--workers", "%W%", "--out", out]
                    argv += ["--nodes", cfg["nodes"]] if "nodes" in cfg else []
                    argv += ["--weights", json.dumps(cfg["weights"], separators=(",", ":"))] if "weights" in cfg else []
                    argv += (["--room", sp["room"]] if sp.get("room") else []) + sp.get("extra", [])
                    jobs.append(fl.add_job(self.name, f"{self.name}_{c}_s{k}", "bench", sp.get("prio", 0.8), argv, files, [out]))
            st.update(phase="running", jobs=jobs, t0=time.time())
            log(f"{self.name}: bench {sp['bench']}, {len(sp['configs'])} configs x {len(shards)} shards")
        elif st["phase"] == "running" and all(fl.jobs[j]["state"] in ("done", "failed") for j in st["jobs"]):
            summ = {}
            for c in sp["configs"]:
                rows = []
                for j in st["jobs"]:
                    job = fl.jobs[j]
                    p = os.path.join(HERE, job["outputs"][0])
                    if job["id"].startswith(f"{self.name}_{c}_s") and job["state"] == "done" and os.path.exists(p):
                        with open(p) as fh:
                            rows += [json.loads(l) for l in fh if l.strip()]
                        os.remove(p)
                with open(os.path.join(HERE, "runs", f"{self.name}_{c}.jsonl"), "w") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")
                by = {}
                for r in rows:
                    b = by.setdefault(r["boss"], [0, 0])
                    b[0] += 1
                    b[1] += bool(r["won"])
                summ[c] = {"fights": len(rows), "won": sum(bool(r["won"]) for r in rows),
                           "mean_s": round(sum(r["seconds"] for r in rows) / max(1, len(rows)), 1),
                           "by_boss": {k: f"{w}/{n}" for k, (n, w) in sorted(by.items())}}
                log(f"{self.name}: {c} {json.dumps(summ[c])}")
            st.update(phase="finished", summary=summ)

    def finished(self):
        return self.st["phase"] == "finished"


def summarize_runs(rows):
    n = len(rows)
    return {"runs": n, "act1": sum(1 for r in rows if (r.get("cleared_acts") or 0) >= 1),
            "act2": sum(1 for r in rows if (r.get("cleared_acts") or 0) >= 2),
            "wins": sum(1 for r in rows if r.get("victory")),
            "mean_floor": round(sum(((r.get("act") or 1) - 1) * 17 + (r.get("floor") or 0) for r in rows) / max(1, n), 2),
            "boss_dmg": round(sum(r.get("boss_dmg") or 0 for r in rows) / max(1, n), 3)}


def read_rows(paths, remove=True):
    rows = []
    for rel in paths:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            with open(p) as fh:
                rows += [json.loads(l) for l in fh if l.strip()]
            if remove:
                os.remove(p)
    return rows


class XDeckRL:
    """Deck RL in the expert world: self-play is the real bot (planner combat, run.py) with its
    deck networks sampled at a temperature (--deck-sample), so learning sees the game it is
    scored on instead of the calibrated fast world. Iteration i: self-play chunks -> one update
    of the sampled networks (card, rest, event, map; the shop network stays) -> the A10
    benchmark (240 paired seeds, greedy) with the new networks, queued ahead of the next
    iteration's self-play so every iteration gets its real score."""

    def __init__(self, spec, st=None):
        self.spec = spec
        self.name = spec["tag"]
        self.KINDS = tuple(spec.get("kinds") or ("card", "rest", "event", "map"))   # the networks sampled and trained
        self.st = st or {"it": spec["start"], "cur": spec["init"], "phase": "start", "evals": {}, "sp": {}}

    def nets_json(self, it, models):
        rel = f"models/{self.name}_nets{it}.json"
        with open(os.path.join(HERE, rel), "w") as fh:
            json.dump({**models, "card_decides": True, "neow": True}, fh)
        return rel

    def run_args(self, end, start, prefix, cfg, tag, extra=()):
        return ["run.py", end, "--seed-start", start, "--acts", 3, "--ascension", 10, "--seed-prefix", prefix,
                "--workers", "%W%", "--rollouts", self.spec["rollouts"], "--deck-nets", cfg, "--tag", tag] + \
            list(self.spec.get("extra", [])) + list(extra)

    def inputs(self, models, cfg):
        return [cfg] + [m for m in models.values() if m]

    def start_selfplay(self, fl, it):
        sp = self.spec
        cfg = self.nets_json(it - 1, self.st["cur"])
        jobs = []
        for k, s0 in enumerate(range(0, sp["runs"], sp["chunk"])):
            rec, rows = f"fleetdata/{self.name}_it{it}_c{k}.jsonl", f"runs/{self.name}{it}_sp{k}.jsonl"
            argv = self.run_args(min(sp["runs"], s0 + sp["chunk"]), s0, f"{self.name}{it}", cfg, f"{self.name}{it}_sp{k}",
                                 ["--deck-sample", sp["temperature"], "--record", rec, "--no-combat-records"])
            jobs.append(fl.add_job(self.name, f"{self.name}{it}_sp{k}", "xselfplay", 1, argv,
                                   self.inputs(self.st["cur"], cfg), [rec, rows]))
        self.st["sp"][str(it)] = {"jobs": jobs, "t0": time.time()}
        self.st["phase"] = "selfplay"
        log(f"{self.name}: iteration {it} expert self-play, {len(jobs)} chunks")

    def start_eval(self, fl, it, models, sp_summary):
        sp = self.spec
        cfg = self.nets_json(it, models)
        jobs = []
        for k, s0 in enumerate(range(0, sp["eval_runs"], sp["eval_chunk"])):
            out = f"runs/{self.name}_ev{it}_c{k}.jsonl"
            argv = self.run_args(min(sp["eval_runs"], s0 + sp["eval_chunk"]), s0, "full", cfg, f"{self.name}_ev{it}_c{k}")
            jobs.append(fl.add_job(self.name, f"{self.name}{it}_ev{k}", "xeval", sp.get("eval_prio", 0.95), argv,
                                   self.inputs(models, cfg), [out]))
        self.st["evals"][str(it)] = {"jobs": jobs, "models": models, "selfplay": sp_summary, "t0": time.time()}

    def step(self, fl):
        st, sp = self.st, self.spec
        if st["phase"] == "start":
            if sp.get("eval_init"):
                self.start_eval(fl, sp["start"] - 1, st["cur"], None)
            self.start_selfplay(fl, st["it"])
        elif st["phase"] == "selfplay":
            it = st["it"]
            ids = st["sp"][str(it)]["jobs"]
            if all(fl.jobs[j]["state"] in ("done", "failed") for j in ids):
                done = [fl.jobs[j] for j in ids if fl.jobs[j]["state"] == "done"]
                summary = summarize_runs(read_rows([j["outputs"][1] for j in done]))
                summary["wall_s"] = round(time.time() - st["sp"][str(it)]["t0"], 1)
                st["sp"][str(it)]["summary"] = summary
                data = [j["outputs"][0] for j in done]
                argv = ["fleetjob.py", "train"] + data + ["--tag", self.name, "--it", it, "--temperature", sp["temperature"]]
                for k in self.KINDS:
                    if st["cur"].get(k):
                        argv += [f"--{k}", st["cur"][k]]
                outs = [f"models/{self.name}_{k}{it}.pt" for k in self.KINDS if st["cur"].get(k)]
                st["train"] = fl.add_job(self.name, f"{self.name}{it}_train", "train", 0, argv,
                                         [m for m in st["cur"].values() if m] + data, outs, gpu=True)
                st["phase"] = "train"
                log(f"{self.name}: iteration {it} expert self-play done {json.dumps(summary)}; training")
        elif st["phase"] == "train":
            j = fl.jobs[st["train"]]
            it = st["it"]
            if j["state"] == "failed":
                log(f"{self.name}: training {it} failed; pipeline stopped")
                st["phase"] = "failed"
            elif j["state"] == "done":
                new = dict(st["cur"])
                new.update({k: f"models/{self.name}_{k}{it}.pt" for k in j["result"]["models"]})
                log(f"{self.name}: iteration {it} trained on {j['host']} ({j['result'].get('device')}, "
                    f"{j['result']['timing'].get('total')} s)")
                self.start_eval(fl, it, new, st["sp"][str(it)].get("summary"))
                fl.cleanup(f"fleetdata/{self.name}_it{it}_c*.jsonl")
                st["cur"] = new
                if it + 1 < sp["start"] + sp["iters"]:
                    st["it"] = it + 1
                    self.start_selfplay(fl, it + 1)
                else:
                    st["phase"] = "finishing"
        for it, ev in list(st["evals"].items()):
            if all(fl.jobs[j]["state"] in ("done", "failed") for j in ev["jobs"]):
                rows = read_rows([fl.jobs[j]["outputs"][0] for j in ev["jobs"] if fl.jobs[j]["state"] == "done"])
                with open(os.path.join(HERE, "runs", f"{self.name}_ev{it}.jsonl"), "w") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")
                res = summarize_runs(rows)
                res["wall_s"] = round(time.time() - ev["t0"], 1)
                row = {"iter": int(it), **{k: ev["models"].get(k) for k in ev["models"]}, "selfplay": ev["selfplay"],
                       "a10": res, "fleet": True, "world": "expert"}
                with open(os.path.join(HERE, "runs", f"{self.name}.jsonl"), "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                log(f"{self.name}: iteration {it} A10 {json.dumps(res)}")
                del st["evals"][it]
        if st["phase"] == "finishing" and not st["evals"]:
            st["phase"] = "finished"

    def finished(self):
        return self.st["phase"] in ("finished", "failed", "cancelled")


PIPES = {"deckrl": DeckRL, "expert": Expert, "bench": Bench, "xdeckrl": XDeckRL}


# ---------------------------------------------------------------- coordinator

class Fleet:
    def __init__(self):
        with open(CONFIG) as fh:
            self.cfg = json.load(fh)
        self.hosts = {}
        for name, h in self.cfg["hosts"].items():
            self.hosts[name] = {**h, "name": name, "state": "unknown", "fails": 0, "files": {}, "drain": None,
                                "gpu_busy": 0, "gpu_quiet_since": None, "gpu_yield": False,
                                "clean": []}
        self.jobs, self.pipes, self.seq, self.until_done = {}, [], 0, False
        if os.path.exists(STATE):
            with open(STATE) as fh:
                s = json.load(fh)
            self.jobs, self.seq = s["jobs"], s["seq"]
            for name, h in s["hosts"].items():
                if name in self.hosts:
                    for k in ("files", "drain", "clean", "state", "gpu_quiet_since", "gpu_yield"):
                        self.hosts[name][k] = h.get(k, self.hosts[name][k])
            self.pipes = [PIPES[p["type"]](p["spec"], p["st"]) for p in s["pipes"]]
            for j in self.jobs.values():
                if j["state"] == "fetch":
                    j["state"] = "running"

    def save(self):
        s = {"jobs": self.jobs, "seq": self.seq, "hosts": self.hosts,
             "pipes": [{"type": next(k for k, v in PIPES.items() if isinstance(p, v)), "spec": p.spec, "st": p.st}
                       for p in self.pipes]}
        tmp = STATE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(s, fh)
        os.replace(tmp, STATE)

    def add_job(self, pipe, jid, kind, prio, argv, inputs, outputs, gpu=False):
        self.seq += 1
        self.jobs[jid] = {"id": jid, "pipe": pipe, "kind": kind, "prio": prio, "seq": self.seq, "argv": argv,
                          "inputs": inputs, "outputs": outputs, "gpu": gpu, "state": "queued", "tries": 0,
                          "host": None, "result": None}
        return jid

    def cleanup(self, pattern):
        for h in self.hosts.values():
            h["clean"].append(pattern)
        for p in glob.glob(os.path.join(HERE, pattern)):
            os.remove(p)
        for h in self.hosts.values():
            for rel in [r for r in h["files"] if fnmatch.fnmatch(r, pattern)]:
                del h["files"][rel]

    # ---- host state

    def running_on(self, h):
        return [j for j in self.jobs.values() if j["state"] == "running" and j["host"] == h["name"]]

    def requeue(self, h, why, count_try=False):
        for j in self.running_on(h):
            j["state"], j["host"] = "queued", None
            if count_try:
                j["tries"] += 1
            log(f"  requeued {j['id']} ({why})")

    def kill(self, h):
        try:
            out = ps(h, KILL_PS.replace("%ROOT%", wroot(h)), timeout=60).strip().splitlines()[-1]
            if not out.endswith("left 0"):
                log(f"{h['name']}: {out}")
        except Exception as e:
            log(f"{h['name']}: kill failed: {e}")

    def drain(self, h, why):
        if h["drain"]:
            return
        log(f"{h['name']}: draining ({why})")
        h["drain"] = why
        if h["state"] == "up":
            self.kill(h)
        self.requeue(h, f"{h['name']} drained")

    def undrain(self, h, why):
        if h["drain"]:
            log(f"{h['name']}: back in the fleet ({why})")
            h["drain"] = None

    def poll(self, h):
        ids = [j["id"] for j in self.running_on(h)]
        script = POLL_PS.replace("%ROOT%", wroot(h)).replace("%IDS%", ",".join(f"'{i}'" for i in ids) or "")
        try:
            out = ps(h, script, timeout=60)
            return json.loads(out.strip().splitlines()[-1])
        except Exception as e:
            return {"error": str(e)[:200]}

    def handle_poll(self, h, r):
        if "error" in r:
            h["fails"] += 1
            if h["state"] != "down" and h["fails"] >= DOWN_AFTER:
                log(f"{h['name']}: not answering ({r['error'][:80]}); marked down")
                h["state"] = "down"
                self.requeue(h, f"{h['name']} down")
            return
        h["fails"] = 0
        if h["state"] != "up":
            log(f"{h['name']}: up")
            if h["state"] == "down":
                self.kill(h)          # leftovers of jobs that were moved elsewhere
                self.requeue(h, "rejoin")
            h["state"] = "up"
            if h["clean"]:
                self.remote_clean(h)
        # drain reasons: the Mac-side flag (fleet.py drain), PAUSE on the machine, its GPU
        # kept busy by another program (yield_gpu hosts)
        reasons = []
        if os.path.exists(os.path.join(FLEET, "drain", h["name"])):
            reasons.append("fleet.py drain")
        if r.get("pause"):
            reasons.append("PAUSE file")
        if h.get("yield_gpu") and r.get("gpu") is not None:
            ours_gpu = any(j.get("gpu_now") for j in self.running_on(h))
            if r["gpu"] >= GPU_BUSY and not ours_gpu:
                h["gpu_busy"] += 1
                h["gpu_quiet_since"] = None
                if h["gpu_busy"] >= 2:
                    h["gpu_yield"] = True
            else:
                h["gpu_busy"] = 0
                if h["gpu_yield"]:
                    h["gpu_quiet_since"] = h["gpu_quiet_since"] or time.time()
                    if time.time() - h["gpu_quiet_since"] >= YIELD_QUIET:
                        h["gpu_yield"] = False
        if h["gpu_yield"]:
            reasons.append(f"GPU busy ({r.get('gpu')}%)")
        if reasons and not h["drain"]:
            self.drain(h, ", ".join(reasons))
        elif not reasons and h["drain"]:
            self.undrain(h, "no reason left")
        if reasons:
            h["drain"] = ", ".join(reasons)
            if r.get("ours"):         # a kill that did not take (or jobs started by hand): again
                log(f"{h['name']}: {r['ours']} of our jobs still alive while drained; killing")
                self.kill(h)
            return
        fetch = []
        for jid, jr in (r.get("jobs") or {}).items():
            j = self.jobs.get(jid)
            if not j or j["state"] != "running" or j["host"] != h["name"]:
                continue
            if jr["state"] == "done":
                if str(jr["rc"]) == "0":
                    try:
                        j["result"] = json.loads(jr["out"]) if jr["out"].strip().startswith("{") else None
                    except ValueError:
                        j["result"] = None
                    j["state"], j["t_end"] = "fetch", time.time()
                    fetch.append(j)
                else:
                    j["tries"] += 1
                    log(f"{h['name']}: {jid} failed rc {jr['rc']} (try {j['tries']}): {jr.get('err', '')[-300:]}")
                    j["state"], j["host"] = ("failed" if j["tries"] >= MAX_TRIES else "queued"), None
            elif jr["state"] == "lost":
                j["tries"] += 1
                log(f"{h['name']}: {jid} vanished (try {j['tries']})")
                j["state"], j["host"] = ("failed" if j["tries"] >= MAX_TRIES else "queued"), None
        if fetch:
            lines = []
            for j in fetch:
                for o in j["outputs"]:
                    os.makedirs(os.path.dirname(os.path.join(HERE, o)), exist_ok=True)
                    lines.append(f'get "{rpath(h, o)}" "{os.path.join(HERE, o)}"')
            try:
                sftp(h, lines)
                for j in fetch:
                    j["state"] = "done"
                    for o in j["outputs"]:
                        lo = os.path.join(HERE, o)
                        h["files"][o] = [os.path.getsize(lo), int(os.path.getmtime(lo))]
                    log(f"{h['name']}: {j['id']} done in {round(j['t_end'] - j['t_start'])} s")
            except Exception as e:
                log(f"{h['name']}: fetch failed ({e}); rerunning")
                for j in fetch:
                    j["state"], j["host"] = "queued", None

    def remote_clean(self, h):
        pats = ",".join(f"'{wroot(h)}\\bot\\{p.replace('/', chr(92))}'" for p in h["clean"])
        try:
            ps(h, f"foreach ($p in @({pats})) {{ Remove-Item $p -ErrorAction SilentlyContinue }}")
            h["clean"] = []
        except Exception:
            pass

    # ---- scheduling

    def free_slots(self, h):
        return h["slots"] - len(self.running_on(h))

    def pick(self, job, hosts):
        if job["gpu"]:
            gpu = [h for h in hosts if h.get("gpu") and self.free_slots(h) > 0
                   and not any(j.get("gpu_now") for j in self.running_on(h))]
            if gpu:
                return sorted(gpu, key=lambda h: h.get("order", 9))[0], True
            # a GPU machine in the fleet frees a slot soon: wait for it (10 min at most)
            if any(h.get("gpu") for h in hosts) and time.time() - job.get("queued_since", time.time()) < 600:
                return None, False
        cand = [h for h in hosts if self.free_slots(h) > 0]
        if not cand:
            return None, False
        return max(cand, key=lambda h: (self.free_slots(h) / h["slots"], -h.get("order", 9))), False

    def schedule(self):
        hosts = [h for h in self.hosts.values() if h["state"] == "up" and not h["drain"]]
        plan = {}
        for j in sorted([j for j in self.jobs.values() if j["state"] == "queued"], key=lambda j: (j["prio"], j["seq"])):
            j.setdefault("queued_since", time.time())
            h, gpu = self.pick(j, hosts)
            if h is None:
                continue
            j["state"], j["host"], j["gpu_now"] = "running", h["name"], gpu
            plan.setdefault(h["name"], []).append(j)
        for name, js in plan.items():
            self.launch(self.hosts[name], js)

    def launch(self, h, js):
        lines, made = [], set()
        try:
            for j in js:
                for rel in j["inputs"]:
                    loc = os.path.join(HERE, rel)
                    size = [os.path.getsize(loc), int(os.path.getmtime(loc))]
                    if h["files"].get(rel) == size:
                        continue
                    d = os.path.dirname(rel)
                    parts = d.split("/") if d else []
                    for i in range(1, len(parts) + 1):
                        sub = "/".join(parts[:i])
                        if sub not in made:
                            lines.append(f'-mkdir "{rpath(h, sub)}"')
                            made.add(sub)
                    lines.append(f'put "{loc}" "{rpath(h, rel)}"')
                    h["files"][rel] = size
                cmdf = os.path.join(FLEET, "jobs", f"{j['id']}.cmd")
                with open(cmdf, "w", newline="") as fh:
                    fh.write(job_cmd(h, j))
                lines.append(f'put "{cmdf}" "{rpath(h, "fleet/jobs/" + j["id"] + ".cmd")}"')
            sftp(h, [f'-mkdir "{rpath(h, d)}"' for d in ("fleet", "fleet/jobs", "fleetdata", "runs")] + lines)
            out = ps(h, LAUNCH_PS.replace("%ROOT%", wroot(h)).replace("%IDS%", ",".join(f"'{j['id']}'" for j in js)))
            json.loads(out.strip().splitlines()[-1])
            for j in js:
                j["t_start"] = time.time()
            log(f"{h['name']}: started {', '.join(j['id'] + (' (GPU)' if j.get('gpu_now') else '') for j in js)}")
        except Exception as e:
            log(f"{h['name']}: launch failed ({str(e)[:200]})")
            for j in js:
                j["state"], j["host"] = "queued", None
                if isinstance(e, FileNotFoundError):    # a missing input will not appear by retrying
                    j["tries"] += 1
                    if j["tries"] >= MAX_TRIES:
                        j["state"] = "failed"
            for rel in [r for r in list(h["files"]) if any(r in j["inputs"] for j in js)]:
                del h["files"][rel]

    def inbox(self):
        for p in sorted(glob.glob(os.path.join(FLEET, "inbox", "*.json"))):
            with open(p) as fh:
                spec = json.load(fh)
            if spec["type"] == "cancel":
                tag = spec["spec"]["tag"]
                for x in self.pipes:
                    if x.name == tag and not x.finished():
                        x.st["phase"] = "cancelled"
                n = 0
                for j in self.jobs.values():
                    if j["pipe"] == tag and j["state"] == "queued":
                        j["state"] = "failed"
                        n += 1
                log(f"pipeline {tag} cancelled ({n} queued jobs dropped; running ones finish)")
                os.remove(p)
                continue
            if any(x.name == spec["spec"]["tag"] for x in self.pipes):
                log(f"pipeline {spec['spec']['tag']} exists; ignored")
            else:
                self.pipes.append(PIPES[spec["type"]](spec["spec"]))
                log(f"pipeline {spec['spec']['tag']} ({spec['type']}) submitted")
            os.remove(p)

    def loop(self):
        log("coordinator started: " + ", ".join(self.hosts))
        while True:
            self.inbox()
            if os.path.exists(os.path.join(FLEET, "STOP")):
                os.remove(os.path.join(FLEET, "STOP"))
                for h in self.hosts.values():
                    if h["state"] == "up":
                        self.kill(h)
                    self.requeue(h, "stop")
                self.save()
                log("stopped (jobs requeued; `fleet.py run` resumes)")
                return
            with ThreadPoolExecutor(len(self.hosts)) as ex:
                polls = dict(zip(self.hosts, ex.map(self.poll, self.hosts.values())))
            for name, r in polls.items():
                self.handle_poll(self.hosts[name], r)
            for p in self.pipes:
                if not p.finished():
                    p.step(self)
            self.schedule()
            self.save()
            if self.until_done and self.pipes and all(p.finished() for p in self.pipes) \
                    and not any(j["state"] in ("running", "queued") for j in self.jobs.values()):
                log("all pipelines finished")
                return
            time.sleep(POLL)


def status():
    if not os.path.exists(STATE):
        print("no fleet state")
        return
    with open(STATE) as fh:
        s = json.load(fh)
    for name, h in s["hosts"].items():
        run = [j["id"] for j in s["jobs"].values() if j["state"] == "running" and j["host"] == name]
        print(f"{name:8s} {h['state']:5s} {('drained: ' + h['drain']) if h.get('drain') else 'active':24s} running {len(run)}: {' '.join(run)}")
    for p in s["pipes"]:
        st = p["st"]
        q = sum(1 for j in s["jobs"].values() if j["pipe"] == p["spec"]["tag"] and j["state"] == "queued")
        print(f"pipeline {p['spec']['tag']} ({p['type']}): phase {st.get('phase')} it {st.get('it')} queued {q}")
    try:
        with open(os.path.join(FLEET, "fleet.log")) as fh:
            print("".join(fh.readlines()[-8:]), end="")
    except OSError:
        pass


def submit(kind, spec):
    os.makedirs(os.path.join(FLEET, "inbox"), exist_ok=True)
    with open(os.path.join(FLEET, "inbox", f"{spec['tag']}.json"), "w") as fh:
        json.dump({"type": kind, "spec": spec}, fh)
    print(f"submitted {kind} {spec['tag']}")


def main():
    for d in ("", "jobs", "drain", "inbox"):
        os.makedirs(os.path.join(FLEET, d), exist_ok=True)
    os.makedirs(os.path.join(HERE, "fleetdata"), exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status", "drain", "undrain", "stop", "deckrl", "xdeckrl", "expert", "bench",
                                    "cancel"])
    ap.add_argument("host", nargs="?")
    ap.add_argument("--until-done", action="store_true", help="run: exit once every pipeline has finished")
    ap.add_argument("--tag")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--init", help="deckrl: TAG:IT of the starting networks (models/TAG_<kind>IT.pt)")
    ap.add_argument("--combat", default="models/policy_rl105.pt")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--runs", type=int, default=4000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--eval-runs", type=int, default=1500)
    ap.add_argument("--eval-chunk", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--eval-init", action="store_true", help="deckrl: evaluate the starting networks first")
    ap.add_argument("--deck-nets")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--expert-chunk", type=int, default=8)
    ap.add_argument("--prefix", default="full")
    ap.add_argument("--rollouts", default='{"k":6,"m":4,"turns":3}')
    ap.add_argument("--missing", action="store_true", help="expert: only seeds without a row in runs/<tag>.jsonl yet")
    ap.add_argument("--prio", type=float, default=None, help="expert/bench: queue priority (self-play 1, evaluation 2)")
    ap.add_argument("--bench", help="bench: fight bench tag (data/bossbench_<bench>)")
    ap.add_argument("--configs", help='bench: JSON {name: rollout config}')
    ap.add_argument("--room", default=None)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--extra", default=None, help='expert/xdeckrl: JSON list of extra run.py arguments')
    args = ap.parse_args()
    if args.cmd == "run":
        lock = os.path.join(FLEET, "coordinator.pid")
        if os.path.exists(lock):
            try:
                os.kill(int(open(lock).read()), 0)
                sys.exit("a coordinator is already running")
            except (OSError, ValueError):
                pass
        with open(lock, "w") as fh:
            fh.write(str(os.getpid()))
        try:
            fl = Fleet()
            fl.until_done = args.until_done
            fl.loop()
        finally:
            os.remove(lock)
    elif args.cmd == "status":
        status()
    elif args.cmd in ("drain", "undrain"):
        flag = os.path.join(FLEET, "drain", args.host)
        if args.cmd == "drain":
            open(flag, "w").close()
        elif os.path.exists(flag):
            os.remove(flag)
        print(f"{args.cmd} {args.host}: the coordinator acts within {POLL} s")
    elif args.cmd == "stop":
        open(os.path.join(FLEET, "STOP"), "w").close()
        print("stop requested")
    elif args.cmd == "deckrl":
        tag, it = args.init.split(":")
        init = {k: f"models/{tag}_{k}{it}.pt" for k in DeckRL.KINDS if os.path.exists(os.path.join(HERE, f"models/{tag}_{k}{it}.pt"))}
        submit("deckrl", {"tag": args.tag, "start": args.start, "iters": args.iters, "runs": args.runs, "chunk": args.chunk,
                          "eval_runs": args.eval_runs, "eval_chunk": args.eval_chunk, "temperature": args.temperature,
                          "combat": args.combat, "init": init, "eval_init": args.eval_init})
    elif args.cmd == "expert":
        seeds = None
        if args.missing:
            done = set()
            path = os.path.join(HERE, "runs", f"{args.tag}.jsonl")
            if os.path.exists(path):
                with open(path) as fh:
                    for line in fh:
                        try:
                            done.add(json.loads(line)["seed"])
                        except (ValueError, KeyError):
                            pass
            seeds = [i for i in range(args.n) if f"{args.prefix}_{i}" not in done]
            print(f"{len(seeds)} seeds missing")
        submit("expert", {"tag": args.tag, "deck_nets": args.deck_nets, "n": args.n, "start": 0, "chunk": args.expert_chunk,
                          "prefix": args.prefix, "rollouts": args.rollouts, "seeds": seeds,
                          "prio": 1.5 if args.prio is None else args.prio, "extra": json.loads(args.extra or "[]")})
    elif args.cmd == "xdeckrl":
        with open(os.path.join(HERE, args.deck_nets)) as fh:
            cfg = json.load(fh)
        init = {k: cfg[k] for k in ("card", "rest", "event", "map", "shop") if cfg.get(k)}
        submit("xdeckrl", {"tag": args.tag, "start": args.start, "iters": args.iters, "runs": args.runs,
                           "chunk": args.expert_chunk, "eval_runs": args.n, "eval_chunk": args.expert_chunk,
                           "temperature": args.temperature, "rollouts": args.rollouts, "init": init,
                           "eval_init": args.eval_init, "kinds": ["card", "rest", "event", "map", "shop"],
                           "extra": json.loads(args.extra or "[]")})
    elif args.cmd == "cancel":
        submit("cancel", {"tag": args.tag})
    elif args.cmd == "bench":
        submit("bench", {"tag": args.tag, "bench": args.bench, "configs": json.loads(args.configs), "room": args.room,
                         "reps": args.reps, "shards": args.shards, "prio": 0.8 if args.prio is None else args.prio})


if __name__ == "__main__":
    main()
