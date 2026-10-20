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
import math
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

import numpy as np

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
BACKUP_AFTER = 120   # a job running this long while the queue is empty gets a backup copy on an idle host
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


def remote_argv(host, cmd):
    """ssh argv running `cmd` on the host. Hosts with "via" (lab2/lab3: this Mac's key is not
    authorised there) are reached by running ssh on the jump host with its own key."""
    if host.get("via"):
        inner = (f"ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -p {host.get('port', 22)} "
                 f"{host['target']} {shlex.quote(cmd)}")
        return SSH + [host["via"], inner]
    return SSH + [host["ssh"], cmd]


def sftp_via(host, lines, timeout=900):
    """The sftp batch subset the fleet uses (put / get / -mkdir with quoted absolute remote
    paths) as tar streams through the jump host."""
    import io
    import re
    import tarfile
    puts, gets, dirs = [], [], []
    for line in lines:
        m = re.fullmatch(r'-?(put|get|mkdir) "([^"]*)"(?: "([^"]*)")?', line.strip())
        if not m:
            raise RuntimeError(f"sftp_via: cannot parse {line!r}")
        op, a, b = m.groups()
        if op == "put":
            puts.append((a, b))
        elif op == "get":
            gets.append((a, b))
        else:
            dirs.append(a)
    if puts or dirs:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for loc, rem in puts:
                tf.add(loc, arcname=rem.lstrip("/"), recursive=False)
        cmd = (f"mkdir -p {' '.join(shlex.quote(d) for d in dirs)} && " if dirs else "") + "tar -xf - -C /"
        res = subprocess.run(remote_argv(host, cmd), input=buf.getvalue(), capture_output=True, timeout=timeout)
        if res.returncode != 0:
            raise RuntimeError(f"put via rc {res.returncode}: {res.stderr.decode(errors='replace')[-300:]}")
    if gets:
        want = {rem.lstrip("/"): loc for rem, loc in gets}
        cmd = "tar -cf - -C / " + " ".join(shlex.quote(r) for r in want)
        res = subprocess.run(remote_argv(host, cmd), capture_output=True, timeout=timeout)
        got = 0
        with tarfile.open(fileobj=io.BytesIO(res.stdout), mode="r") as tf:
            for mem in tf.getmembers():
                if mem.isfile() and mem.name in want:
                    os.makedirs(os.path.dirname(want[mem.name]) or ".", exist_ok=True)
                    with open(want[mem.name], "wb") as out:
                        out.write(tf.extractfile(mem).read())
                    got += 1
        if res.returncode != 0 or got < len(want):
            raise RuntimeError(f"get via rc {res.returncode}, {got}/{len(want)} files: "
                               f"{res.stderr.decode(errors='replace')[-300:]}")


def sftp(host, lines, timeout=900):
    if not lines:
        return
    if host.get("via"):
        return sftp_via(host, lines, timeout)
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


def linux(host):
    return host.get("os") == "linux"


def rpath(host, rel=""):
    """Remote path for sftp ("/D:/sts2/bot/<rel>" on Windows, "<root>/bot/<rel>" on Linux)."""
    if linux(host):
        return host["root"].rstrip("/") + "/bot/" + rel
    return "/" + host["root"].replace("\\", "/").rstrip("/") + "/bot/" + rel.replace("\\", "/")


def lx(host, script, timeout=60, interp="bash -s"):
    """Run a script on a Linux host (bash, or python3 with interp='python3 -'); returns stdout."""
    res = subprocess.run(remote_argv(host, interp), input=script, capture_output=True, text=True, timeout=timeout,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"ssh rc {res.returncode}: {res.stderr[-300:]}")
    return res.stdout


def lx_pat(path):
    """pgrep/pkill pattern for a path that does not match the pkill command line itself."""
    return "[" + path[0] + "]" + path[1:]


LX_POLL = r"""
import json, os, re, subprocess
root, ids = %ROOT%, %IDS%
ps = subprocess.run(["ps", "-eww", "-o", "args"], capture_output=True, text=True).stdout.splitlines()
alive = set()
for line in ps:
    m = re.search(r"fleet/jobs/([\w.\-]+)\.sh", line)
    if m and root in line:
        alive.add(m.group(1))
jobs = {}
for i in ids:
    d = root + "/bot/fleet/jobs/" + i
    if os.path.exists(d + ".rc"):
        rc = open(d + ".rc").read().strip()
        try:
            lines = open(d + ".out", errors="replace").read().splitlines()
        except OSError:
            lines = []
        err = ""
        if rc != "0":
            try:
                err = "\n".join(open(d + ".err", errors="replace").read().splitlines()[-12:])
            except OSError:
                pass
        jobs[i] = {"state": "done", "rc": rc, "out": lines[-1] if lines else "", "err": err}
    elif i in alive:
        jobs[i] = {"state": "running"}
    else:
        jobs[i] = {"state": "lost"}
def sample():
    # (all CPU ticks, idle ticks, ticks of our processes: command line under our root)
    v = [int(x) for x in open("/proc/stat").readline().split()[1:9]]
    mine = 0
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            if root.encode() not in open("/proc/" + p + "/cmdline", "rb").read():
                continue
            s = open("/proc/" + p + "/stat").read()
            f = s[s.rfind(")") + 2:].split()
            mine += int(f[11]) + int(f[12])
        except (OSError, ValueError, IndexError):
            pass
    return sum(v), v[3] + v[4], mine
import time
a = sample(); time.sleep(1.5); b = sample()
ncpu = os.cpu_count()
per_cpu = max(1, (b[0] - a[0]) / ncpu)
busy = ((b[0] - a[0]) - (b[1] - a[1])) / per_cpu
mine = max(0, b[2] - a[2]) / per_cpu
print(json.dumps({"pause": os.path.exists(root + "/PAUSE"), "jobs": jobs, "gpu": None, "ours": len(alive),
                  "load": float(open("/proc/loadavg").read().split()[0]), "ncpu": ncpu,
                  "others_now": round(max(0.0, busy - mine), 2), "ours_cpu": round(mine, 2)}))
"""


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

KILL_JOB_PS = r"""
$root = '%ROOT%'; $id = '%ID%'
Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" | Where-Object { $_.CommandLine -like "*$root\bot\fleet\jobs\$id.cmd*" } |
  ForEach-Object { taskkill /PID $_.ProcessId /T /F 2>&1 | Out-Null }
'killed'
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


def code_sig():
    """Size and mtime of every Python/shell file the jobs run (cheap: stat only)."""
    hs = hashlib.sha1()
    for d in (HERE, os.path.join(REPO, "harness"), os.path.join(REPO, "deploy"), os.path.join(REPO, "sts2-cli", "src", "Sts2Headless")):
        for p in sorted(glob.glob(os.path.join(d, "*.py")) + glob.glob(os.path.join(d, "*.sh")) + glob.glob(os.path.join(d, "*.json"))
                        + glob.glob(os.path.join(d, "*.cs"))):
            st = os.stat(p)
            hs.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
    return hs.hexdigest()


def job_sh(host, job):
    """Linux job script: env on the data disk (HOME too), below-normal CPU and idle IO come from the launcher."""
    R = host["root"].rstrip("/")
    jid = job["id"]
    argv = [str(host["workers"]) if a == "%W%" else str(a) for a in job["argv"]]
    env = {"HOME": f"{R}/home", "DOTNET": f"{R}/dotnet/dotnet", "DOTNET_ROOT": f"{R}/dotnet", "DOTNET_CLI_HOME": f"{R}/cache/dotnethome",
           "DOTNET_CLI_TELEMETRY_OPTOUT": "1", "OMP_NUM_THREADS": "1", "XDG_CACHE_HOME": f"{R}/cache", "NETS_DEVICE": "cpu",
           "PYTHONUNBUFFERED": "1"}
    py = f"{R}/.venv/bin/python"
    gpu = []
    if job.get("gpu_now"):
        # lab rule: only a fully idle card (memory ~0), never the avoided one (lab0's A6000);
        # none idle or no CUDA venv: run on the CPU instead
        avoid = host.get("gpu_avoid", "^$")
        gpu = [f"G=$(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader,nounits 2>/dev/null | "
               f"awk -F', ' '$2 !~ /{avoid}/ && $3 < 100 {{print $1; exit}}')",
               f'PY={py}',
               f'if [ -n "$G" ] && [ -x {R}/.venv-cu/bin/python ]; then export CUDA_VISIBLE_DEVICES=$G NETS_DEVICE=cuda; PY={R}/.venv-cu/bin/python; fi',
               f'echo "gpu=${{G:-none}}" > fleet/jobs/{jid}.gpu']
        py = "$PY"
    lines = ["#!/bin/bash"] + [f"export {k}={shlex.quote(v)}" for k, v in env.items()] + [f"mkdir -p {R}/home", f"cd {R}/bot"] + \
        [f"rm -f {shlex.quote(o)}" for o in job["outputs"]] + gpu + \
        [f"{py} {' '.join(shlex.quote(a) for a in argv)} > fleet/jobs/{jid}.out 2> fleet/jobs/{jid}.err",
         f"echo $? > fleet/jobs/{jid}.rc"]
    return "\n".join(lines) + "\n"


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
                                       [sp["deck_nets"]] + nets + sp.get("inputs", []), [f"runs/{self.name}_c{k}.jsonl"]))
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
                    roll = {a: v for a, v in cfg.items() if a not in ("nodes", "weights", "extra")}
                    argv = ["bossbench.py", sp.get("mode", "eval"), "--tag", sp["bench"], "--rollouts", json.dumps(roll, separators=(",", ":")),
                            "--reps", sp.get("reps", 1), "--shard", f"{k}/{sp['shards']}", "--workers", "%W%", "--out", out]
                    argv += ["--variants", sp["variants"], "--rep-offset", sp.get("offset", 0)] if sp.get("mode") in ("explore", "vary") else []
                    argv += ["--nodes", cfg["nodes"]] if "nodes" in cfg else []
                    argv += ["--weights", json.dumps(cfg["weights"], separators=(",", ":"))] if "weights" in cfg else []
                    argv += (["--room", sp["room"]] if sp.get("room") else []) + sp.get("extra", []) + cfg.get("extra", [])
                    jobs.append(fl.add_job(self.name, f"{self.name}_{c}_s{k}", "bench", sp.get("prio", 0.8), argv,
                                           files + sp.get("inputs", []), [out]))
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
                    b = by.setdefault(str(r.get("boss")), [0, 0])
                    b[0] += 1
                    b[1] += bool(r["won"])
                summ[c] = {"fights": len(rows), "won": sum(bool(r["won"]) for r in rows),
                           "mean_s": round(sum(r.get("seconds", 0) for r in rows) / max(1, len(rows)), 1),
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
            rec = f"fleetdata/{self.name}_ev{it}_c{k}_rec.jsonl"   # greedy-play records (value-network data)
            argv = self.run_args(min(sp["eval_runs"], s0 + sp["eval_chunk"]), s0, "full", cfg, f"{self.name}_ev{it}_c{k}",
                                 ["--record", rec, "--no-combat-records"])
            jobs.append(fl.add_job(self.name, f"{self.name}{it}_ev{k}", "xeval", sp.get("eval_prio", 0.95), argv,
                                   self.inputs(models, cfg), [out, rec]))
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
                # "reuse": N also trains on the previous N iterations' self-play (archived), PPO's
                # ratio clipping covering the slightly older behaviour policies
                for back in range(1, sp.get("reuse", 0) + 1):
                    d = os.path.join(HERE, "data", "xdarchive", f"{self.name}_it{it - back}")
                    data += sorted(os.path.relpath(p, HERE) for p in glob.glob(os.path.join(d, f"{self.name}_it{it - back}_c*.jsonl")))
                listing = f"fleetdata/{self.name}_it{it}_train.txt"   # the file list (cmd lines cap at 8191 chars)
                with open(os.path.join(HERE, listing), "w") as fh:
                    fh.write("\n".join(data) + "\n")
                argv = ["fleetjob.py", "train", "@" + listing, "--tag", self.name, "--it", it, "--temperature", sp["temperature"],
                        "--lr-scale", sp.get("lr_scale", 1.0)]
                if sp.get("reward"):
                    argv += ["--reward", json.dumps(sp["reward"], separators=(",", ":"))]
                for k in self.KINDS:
                    if st["cur"].get(k):
                        argv += [f"--{k}", st["cur"][k]]
                outs = [f"models/{self.name}_{k}{it}.pt" for k in self.KINDS if st["cur"].get(k)]
                st["train"] = fl.add_job(self.name, f"{self.name}{it}_train", "train", 0, argv,
                                         [m for m in st["cur"].values() if m] + data + [listing], outs, gpu=True)
                st["phase"] = "train"
                log(f"{self.name}: iteration {it} expert self-play done {json.dumps(summary)}; training on {len(data)} files")
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
                fl.cleanup(f"fleetdata/{self.name}_it{it}_c*.jsonl", keep=f"data/xdarchive/{self.name}_it{it}")
                st["cur"] = new
                if it + 1 < sp["start"] + sp["iters"]:
                    st["it"] = it + 1
                    self.start_selfplay(fl, it + 1)
                else:
                    st["phase"] = "finishing"
        for it, ev in list(st["evals"].items()):
            if all(fl.jobs[j]["state"] in ("done", "failed") for j in ev["jobs"]):
                rows = read_rows([fl.jobs[j]["outputs"][0] for j in ev["jobs"] if fl.jobs[j]["state"] == "done"])
                fl.cleanup(f"fleetdata/{self.name}_ev{it}_c*_rec.jsonl", keep=f"data/xdarchive/{self.name}_ev{it}")
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


class Task:
    """One ad-hoc job (training a network, an analysis): argv run with the bot's Python, input
    files pushed first, output files fetched back to this Mac. The last stdout line (JSON or
    text) is logged."""

    def __init__(self, spec, st=None):
        self.spec = spec
        self.name = spec["tag"]
        self.st = st or {"phase": "start"}

    def step(self, fl):
        sp, st = self.spec, self.st
        if st["phase"] == "start":
            st["job"] = fl.add_job(self.name, f"{self.name}_task", "task", sp.get("prio", 0.5), sp["argv"],
                                   sp.get("inputs", []), sp.get("outputs", []), gpu=sp.get("gpu", False))
            st["phase"] = "running"
        elif st["phase"] == "running" and fl.jobs[st["job"]]["state"] in ("done", "failed"):
            j = fl.jobs[st["job"]]
            st.update(phase="finished", state=j["state"], result=j.get("result"))
            log(f"{self.name}: task {j['state']} {json.dumps(j.get('result'))[:300]}")

    def finished(self):
        return self.st["phase"] == "finished"


def fight_score(r):
    """Bench fight -> tuning score: a win 100 + HP left, a loss 0..80 by enemy HP removed."""
    return max(0.0, min(200.0, float(r.get("graded") or 0.0)))


class Tune:
    """CMA-ES over planner evaluation weights (combat.W), scored on fight benches. Every
    generation samples `popsize` weight vectors, plays the same fights with the same hidden
    futures for all of them (common random numbers; a fresh set of futures each generation, so
    the search cannot fit one draw order), and moves the distribution toward the best half.
    The distribution mean is played too, as the progress trace. Parameters: {"name": {"x0": v,
    "log": true}} (value = x0 * exp(x)) or {"x0": v, "scale": s} (value = max(0, x0 + s * x))."""

    def __init__(self, spec, st=None):
        self.spec = spec
        self.name = spec["tag"]
        self.names = list(spec["params"])
        n = len(self.names)
        self.st = st or {"gen": 0, "phase": "start", "m": [0.0] * n, "sigma": spec.get("sigma", 0.3),
                         "C": np.eye(n).tolist(), "pc": [0.0] * n, "ps": [0.0] * n, "history": []}

    def weights_of(self, x):
        w = {}
        for k, xi in zip(self.names, x):
            p = self.spec["params"][k]
            w[k] = round(p["x0"] * math.exp(xi), 4) if p.get("log") else round(max(0.0, p["x0"] + p.get("scale", 1.0) * xi), 4)
        return w

    def consts(self):
        n = len(self.names)
        lam = self.spec.get("popsize", 4 + int(3 * math.log(n)))
        mu = lam // 2
        wts = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        wts /= wts.sum()
        mueff = 1.0 / (wts ** 2).sum()
        cs = (mueff + 2) / (n + mueff + 5)
        ds = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        return lam, mu, wts, mueff, cs, ds, cc, c1, cmu, chin

    def start_gen(self, fl):
        st, sp = self.st, self.spec
        n = len(self.names)
        lam = self.consts()[0]
        rng = np.random.RandomState(1000 + st["gen"])
        C = np.asarray(st["C"])
        evals, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(evals, 1e-12))
        ys = [(B @ (D * rng.randn(n))).tolist() for _ in range(lam)]
        xs = [(np.asarray(st["m"]) + st["sigma"] * np.asarray(y)).tolist() for y in ys]
        cands = [{"y": y, "x": x, "w": self.weights_of(x)} for y, x in zip(ys, xs)]
        cands.append({"y": None, "x": list(st["m"]), "w": self.weights_of(st["m"]), "mean": True})
        # control: the starting weights on the same fights, for a paired mean-vs-start difference
        cands.append({"y": None, "x": [0.0] * n, "w": self.weights_of([0.0] * n), "control": True})
        off = 100 * (st["gen"] + 1)   # hidden futures of this generation
        for ci, c in enumerate(cands):
            c["jobs"] = []
            for bi, b in enumerate(sp["benches"]):
                for k in b.get("use") or range(b["shards"]):
                    out = f"runs/{self.name}_g{st['gen']}_c{ci}_b{bi}_s{k}.jsonl"
                    argv = ["bossbench.py", "eval", "--tag", b["bench"], "--rollouts", sp["rollouts"], "--reps", b["reps"],
                            "--rep-offset", off, "--shard", f"{k}/{b['shards']}", "--workers", "%W%", "--out", out,
                            "--weights", json.dumps(c["w"], separators=(",", ":"))] + (["--room", b["room"]] if b.get("room") else [])
                    files = bench_files(b["bench"], b.get("room"), k, b["shards"])
                    if files:
                        c["jobs"].append(fl.add_job(self.name, f"{self.name}_g{st['gen']}_c{ci}_b{bi}_s{k}", "tune",
                                                    sp.get("prio", 0.8), argv, files, [out]))
        st["cands"] = cands
        st["phase"] = "running"
        log(f"{self.name}: generation {st['gen']}, {lam} candidates + mean, sigma {st['sigma']:.3f}")

    def finish_gen(self, fl):
        st = self.st
        n = len(self.names)
        lam, mu, wts, mueff, cs, ds, cc, c1, cmu, chin = self.consts()
        for c in st["cands"]:
            rows = read_rows([fl.jobs[j]["outputs"][0] for j in c["jobs"] if fl.jobs[j]["state"] == "done"])
            c["fights"] = len(rows)
            c["won"] = sum(bool(r["won"]) for r in rows)
            c["fit"] = sum(fight_score(r) for r in rows) / max(1, len(rows))
        samples = [c for c in st["cands"] if not c.get("mean") and not c.get("control")]
        mean_c = next(c for c in st["cands"] if c.get("mean"))
        ctrl = next((c for c in st["cands"] if c.get("control")), None)
        order = sorted(range(len(samples)), key=lambda i: -samples[i]["fit"])
        Y = np.asarray([samples[i]["y"] for i in order[:mu]])
        yw = wts @ Y
        m = np.asarray(st["m"]) + st["sigma"] * yw
        C = np.asarray(st["C"])
        evals, B = np.linalg.eigh(C)
        Cinvsqrt = B @ np.diag(1 / np.sqrt(np.maximum(evals, 1e-12))) @ B.T
        ps = (1 - cs) * np.asarray(st["ps"]) + math.sqrt(cs * (2 - cs) * mueff) * (Cinvsqrt @ yw)
        hs = float(np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (st["gen"] + 1))) < (1.4 + 2 / (n + 1)) * chin)
        pc = (1 - cc) * np.asarray(st["pc"]) + hs * math.sqrt(cc * (2 - cc) * mueff) * yw
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C) \
            + cmu * sum(w * np.outer(y, y) for w, y in zip(wts, Y))
        sigma = st["sigma"] * math.exp((cs / ds) * (np.linalg.norm(ps) / chin - 1))
        best = samples[order[0]]
        row = {"gen": st["gen"], "mean_fit": round(mean_c["fit"], 3), "mean_won": mean_c["won"], "fights": mean_c["fights"],
               "best_fit": round(best["fit"], 3), "best_won": best["won"], "pop_fit": round(float(np.mean([c["fit"] for c in samples])), 3),
               "sigma": round(sigma, 4), "mean_w": mean_c["w"], "best_w": best["w"]}
        if ctrl is not None:
            row.update(control_fit=round(ctrl["fit"], 3), control_won=ctrl["won"])
        st["history"].append(row)
        with open(os.path.join(HERE, "runs", f"{self.name}.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
        log(f"{self.name}: generation {st['gen']} mean fit {row['mean_fit']} ({row['mean_won']}/{row['fights']} won), "
            f"best {row['best_fit']} ({row['best_won']}), pop {row['pop_fit']}, sigma {row['sigma']}"
            + (f", start weights {row['control_fit']} ({row['control_won']} won)" if ctrl is not None else ""))
        st.update(m=m.tolist(), C=C.tolist(), ps=ps.tolist(), pc=pc.tolist(), sigma=sigma, gen=st["gen"] + 1)
        with open(os.path.join(HERE, "models", f"{self.name}_mean.json"), "w") as fh:
            json.dump(self.weights_of(st["m"]), fh)
        st.pop("cands", None)

    def step(self, fl):
        st = self.st
        if st["phase"] == "start":
            self.start_gen(fl)
        elif st["phase"] == "running":
            if all(fl.jobs[j]["state"] in ("done", "failed") for c in st["cands"] for j in c["jobs"]):
                self.finish_gen(fl)
                if st["gen"] >= self.spec["generations"]:
                    st["phase"] = "finished"
                    log(f"{self.name}: tuning finished; weights in models/{self.name}_mean.json")
                else:
                    self.start_gen(fl)

    def finished(self):
        return self.st["phase"] in ("finished", "cancelled")


def bench_files(bench, room, k, shards):
    """bot/-relative save files (and metadata) of a bench shard, as bossbench.py --shard sees it."""
    bdir = f"data/bossbench_{bench}"
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, bdir, "*.save.json"))):
        with open(p) as fh:
            m = json.load(fh)
        if room and m.get("room", "Boss") != room:
            continue
        if int(hashlib.md5(m["seed"].encode()).hexdigest(), 16) % shards != k:
            continue
        base = os.path.basename(m["path"])
        out += [f"{bdir}/{base}", f"{bdir}/{base}.json"]
    return out


PIPES = {"deckrl": DeckRL, "expert": Expert, "bench": Bench, "xdeckrl": XDeckRL, "task": Task, "tune": Tune}


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

    def cleanup(self, pattern, keep=None):
        """Delete files matching pattern on every host; on this Mac move them to `keep`
        (a directory under bot/) instead when given."""
        for h in self.hosts.values():
            h["clean"].append(pattern)
        if keep:
            os.makedirs(os.path.join(HERE, keep), exist_ok=True)
        for p in glob.glob(os.path.join(HERE, pattern)):
            if keep:
                os.replace(p, os.path.join(HERE, keep, os.path.basename(p)))
            else:
                os.remove(p)
        for h in self.hosts.values():
            for rel in [r for r in h["files"] if fnmatch.fnmatch(r, pattern)]:
                del h["files"][rel]

    # ---- host state

    def running_on(self, h):
        """Jobs running on h, primary copies and backup copies."""
        return [j for j in self.jobs.values() if j["state"] == "running"
                and (j["host"] == h["name"] or j.get("backup") == h["name"])]

    def drop_copy(self, j, host):
        """Forget the copy of j on host; True if another copy keeps running (then no requeue)."""
        if j.get("backup") == host:
            j["backup"] = None
            return True
        if j.get("backup"):
            j["host"], j["backup"] = j["backup"], None   # the backup becomes the job
            return True
        return False

    def requeue(self, h, why, count_try=False):
        for j in self.running_on(h):
            if self.drop_copy(j, h["name"]):
                log(f"  {j['id']}: copy on {h['name']} dropped, the other copy goes on ({why})")
                continue
            j["state"], j["host"] = "queued", None
            if count_try:
                j["tries"] += 1
            log(f"  requeued {j['id']} ({why})")

    def kill_job(self, host, jid):
        h = self.hosts.get(host)
        if h and h["state"] == "up":
            try:
                if linux(h):   # the job runs in its own session (setsid): stop the whole process group
                    pat = lx_pat(h["root"].rstrip("/") + "/bot/fleet/jobs/" + jid + ".sh")
                    lx(h, f"for p in $(pgrep -f '{pat}'); do kill -TERM -- -$p 2>/dev/null; done; sleep 1; "
                          f"for p in $(pgrep -f '{pat}'); do kill -KILL -- -$p 2>/dev/null; done; echo ok")
                    return
                ps(h, KILL_JOB_PS.replace("%ROOT%", wroot(h)).replace("%ID%", jid), timeout=60)
            except Exception as e:
                log(f"{host}: kill of {jid} failed: {e}")

    def kill(self, h):
        try:
            if linux(h):
                pat = lx_pat(h["root"].rstrip("/") + "/")
                out = lx(h, f"pkill -TERM -f '{pat}'; sleep 1; pkill -KILL -f '{pat}'; sleep 0.5; "
                            f"echo killed, left $(pgrep -fc '{pat}')").strip().splitlines()[-1]
                if not out.endswith("left 0"):
                    log(f"{h['name']}: {out}")
                return
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

    def sync_code(self, h):
        if h.get("via"):   # deploy/winsync.sh's file list, piped through the jump host
            files = subprocess.run("git ls-files -co --exclude-standard bot deploy harness sts2-cli/src sts2-cli/localization_eng "
                                   "sts2-cli/localization_zhs | grep -v '\\.pt$'", shell=True, cwd=REPO,
                                   capture_output=True, text=True).stdout
            tar = subprocess.run(["tar", "--no-mac-metadata", "--no-xattrs", "-cf", "-", "-T", "-"], cwd=REPO, input=files.encode(),
                                 capture_output=True, env={**os.environ, "COPYFILE_DISABLE": "1"}, timeout=300)
            res = subprocess.run(remote_argv(h, f"tar -xf - -C {shlex.quote(h['root'])}"), input=tar.stdout,
                                 capture_output=True, timeout=600)
            return res.returncode == 0 and tar.returncode == 0, (res.stderr or tar.stderr).decode(errors="replace")[-200:]
        res = subprocess.run([os.path.join(REPO, "deploy", "winsync.sh"), h["ssh"], h["root"]],
                             capture_output=True, text=True, timeout=600)
        return res.returncode == 0, (res.stderr or res.stdout)[-200:]

    # engine builds that jobs may ask for (run.py / bossbench.py --engine-config); Windows hosts
    # build missing ones after a code sync (the Linux hosts get theirs copied from lab0)
    # Fix2 (2026-10-20): Ascension 10's Double Boss (the sim used to report victory after the
    # first act-3 boss). A new config name per engine change: running jobs keep their DLL
    ENGINE_BUILDS = ("Fix1", "Diag", "Fix2")

    def ensure_builds(self, h):
        if linux(h):
            R = h["root"].rstrip("/")
            dll = "sts2-cli/src/Sts2Headless/bin/{c}/net9.0/Sts2Headless.dll"
            try:
                have = lx(h, "; ".join(f"[ -f {R}/{dll.format(c=c)} ] && echo have:{c}" for c in self.ENGINE_BUILDS) + "; echo ok")
                missing = [c for c in self.ENGINE_BUILDS if f"have:{c}" not in have]
                if not missing:
                    return
                if h.get("via"):   # lab2/lab3 have the .NET runtime only: copy lab0's builds through the jump
                    J = self.hosts[h["via"]]
                    JR = J["root"].rstrip("/")
                    got = []
                    for c in missing:
                        cmd = (f"[ -f {JR}/{dll.format(c=c)} ] && tar -cf - -C {JR}/sts2-cli/src/Sts2Headless/bin {c} | "
                               f"ssh -o BatchMode=yes -p {h.get('port', 22)} {h['target']} "
                               f"{shlex.quote(f'tar -xf - -C {R}/sts2-cli/src/Sts2Headless/bin && echo copied')}")
                        res = subprocess.run(SSH + [h["via"], cmd], capture_output=True, text=True, timeout=600)
                        if "copied" in res.stdout:
                            got.append(c)
                    log(f"{h['name']}: engine builds copied from {h['via']}: {' '.join(got) or 'none'}"
                        f"{'' if len(got) == len(missing) else ' (missing there: ' + ' '.join(sorted(set(missing) - set(got))) + ')'}")
                    return
                cmd = " ; ".join(
                    f"( cd {R}/sts2-cli && DOTNET_CLI_TELEMETRY_OPTOUT=1 nice -n 19 ionice -c3 {R}/dotnet/dotnet build "
                    f"src/Sts2Headless/Sts2Headless.csproj -c {c} -p:Optimize=true --no-incremental > /dev/null 2>&1 ; "
                    f"[ -f {R}/{dll.format(c=c)} ] && echo built:{c} )" for c in missing)
                out = lx(h, cmd + " ; echo ok", timeout=1200)
                built = [c for c in missing if f"built:{c}" in out]
                log(f"{h['name']}: engine builds: {' '.join(built) or 'none'}"
                    f"{'' if len(built) == len(missing) else ' (FAILED: ' + ' '.join(sorted(set(missing) - set(built))) + ')'}")
            except Exception as e:
                log(f"{h['name']}: engine build check failed ({str(e)[:100]})")
            return
        R = wroot(h)
        script = "$env:DOTNET_CLI_TELEMETRY_OPTOUT='1'; Set-Location '" + R + "\\sts2-cli'\n" + "\n".join(
            f"if (-not (Test-Path '{R}\\sts2-cli\\src\\Sts2Headless\\bin\\{c}\\net9.0\\Sts2Headless.dll')) {{ "
            f"& '{R}\\dotnet\\dotnet.exe' build src\\Sts2Headless\\Sts2Headless.csproj -c {c} -p:Optimize=true --no-incremental "
            f"2>&1 | Select-Object -Last 2; 'built {c}' }}" for c in self.ENGINE_BUILDS)
        try:
            out = ps(h, script, timeout=600)
            if "built" in out:
                log(f"{h['name']}: engine builds: {' '.join(l for l in out.split() if l in self.ENGINE_BUILDS)}")
        except Exception as e:
            log(f"{h['name']}: engine build check failed ({str(e)[:100]})")

    def undrain(self, h, why):
        if h["drain"]:
            ok, msg = self.sync_code(h)   # the code may have changed while it was drained
            log(f"{h['name']}: back in the fleet ({why}){'' if ok else ' (code sync failed: ' + msg + ')'}")
            h["drain"] = None

    def poll(self, h):
        ids = [j["id"] for j in self.running_on(h)]
        try:
            if linux(h):
                out = lx(h, LX_POLL.replace("%ROOT%", json.dumps(h["root"].rstrip("/"))).replace("%IDS%", json.dumps(ids)),
                         interp="python3 -")
                return json.loads(out.strip().splitlines()[-1])
            script = POLL_PS.replace("%ROOT%", wroot(h)).replace("%IDS%", ",".join(f"'{i}'" for i in ids) or "")
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
            # the code may have changed while it was away: same code everywhere before any job
            sig = code_sig()
            ok, msg = self.sync_code(h)
            if ok:
                h["code_sig"] = sig
            if ok:
                self.ensure_builds(h)
            if not ok:
                h["sync_fails"] = h.get("sync_fails", 0) + 1
                if h["sync_fails"] < 3:
                    log(f"{h['name']}: code sync failed ({msg}); retry next poll")
                    return
                log(f"{h['name']}: code sync failed 3 times; joining anyway")
            else:
                log(f"{h['name']}: code synced")
            h["sync_fails"] = 0
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
        if h.get("dynamic") and r.get("ncpu"):
            # shared machine: at most `share` (half) of the threads other users leave idle; when they
            # need more, stop our newest jobs (only ours, found by our own job names)
            if r.get("others_now") is not None:
                # measured over 1.5 s (/proc/stat minus our own processes), not the load
                # average, which lags our own starts and stops by minutes and made us start
                # jobs, see the load rise, and stop them again; rises at once, falls slowly
                prev = h.get("others_ema")
                others = r["others_now"] if prev is None else max(r["others_now"], 0.75 * prev + 0.25 * r["others_now"])
                h["others_ema"] = others
                # new jobs only within the peak of the last 10 minutes (others' use swings
                # 20 -> 40 threads within a minute); stopping goes by the current estimate
                now = time.time()
                hist = [x for x in h.get("others_hist", []) if now - x[0] < 600] + [[now, r["others_now"]]]
                h["others_hist"] = hist
                peak = max([others] + [x[1] for x in hist])
            else:
                ours_threads = sum(h["workers"] for _ in self.running_on(h))
                others = peak = max(0.0, r["load"] - ours_threads)
            per = max(1, h["workers"])
            allow = int((r["ncpu"] - others) * h.get("share", 0.5)) // per
            h["slots_now"] = max(0, min(h["slots"], int((r["ncpu"] - peak) * h.get("share", 0.5)) // per))
            h["others_load"] = round(others, 1)
            extra = len(self.running_on(h)) - max(0, min(h["slots"], allow))
            if extra > 1:   # one slot of slack against load-average jitter
                for j in sorted(self.running_on(h), key=lambda j: -j.get("t_start", 0))[:extra]:
                    self.kill_job(h["name"], j["id"])
                    if not self.drop_copy(j, h["name"]):
                        j["state"], j["host"] = "queued", None
                log(f"{h['name']}: others' load {others:.0f} of {r['ncpu']} threads; stopped {extra} of our jobs")
        if reasons:
            h["drain"] = ", ".join(reasons)
            if r.get("ours"):         # a kill that did not take (or jobs started by hand): again
                log(f"{h['name']}: {r['ours']} of our jobs still alive while drained; killing")
                self.kill(h)
            return
        fetch = []
        for jid, jr in (r.get("jobs") or {}).items():
            j = self.jobs.get(jid)
            if not j or j["state"] != "running" or h["name"] not in (j["host"], j.get("backup")):
                continue
            if jr["state"] == "done" and str(jr["rc"]) == "0":
                try:
                    j["result"] = json.loads(jr["out"]) if jr["out"].strip().startswith("{") else None
                except ValueError:
                    j["result"] = None
                other = j.get("backup") if j["host"] == h["name"] else j["host"]
                if j.get("backup"):      # first copy to finish wins; stop the other one
                    if other:
                        self.kill_job(other, jid)
                    log(f"{h['name']}: {jid} finished first ({'backup' if j.get('backup') == h['name'] else 'primary'} copy)")
                j["host"], j["backup"] = h["name"], None
                j["state"], j["t_end"] = "fetch", time.time()
                fetch.append(j)
                continue
            what = f"failed rc {jr['rc']}" if jr["state"] == "done" else "vanished" if jr["state"] == "lost" else None
            if what is None:
                continue
            if self.drop_copy(j, h["name"]):
                log(f"{h['name']}: {jid} {what} here; the other copy goes on")
                continue
            j["tries"] += 1
            log(f"{h['name']}: {jid} {what} (try {j['tries']}): {jr.get('err', '')[-300:]}")
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
        if linux(h):
            try:
                R = h["root"].rstrip("/")
                lx(h, "cd " + shlex.quote(R + "/bot") + " && rm -f " + " ".join(p for p in h["clean"]) + "; echo ok")
                h["clean"] = []
            except Exception:
                pass
            return
        pats = ",".join(f"'{wroot(h)}\\bot\\{p.replace('/', chr(92))}'" for p in h["clean"])
        try:
            ps(h, f"foreach ($p in @({pats})) {{ Remove-Item $p -ErrorAction SilentlyContinue }}")
            h["clean"] = []
        except Exception:
            pass

    # ---- scheduling

    def free_slots(self, h):
        cap = h.get("slots_now", h["slots"]) if h.get("dynamic") else h["slots"]
        return cap - len(self.running_on(h))

    def pick(self, job, hosts):
        if job["gpu"]:
            gpu = [h for h in hosts if h.get("gpu") and self.free_slots(h) > 0
                   and not any(j.get("gpu_now") for j in self.running_on(h))]
            if gpu:
                return sorted(gpu, key=lambda h: h.get("gpu_order", h.get("order", 9)))[0], True
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
        # backups: with nothing left in the queue, an idle slot runs a copy of the oldest
        # job still running elsewhere (stragglers on the slower machine end an iteration late)
        if any(j["state"] == "queued" for j in self.jobs.values()):
            return
        for h in sorted(hosts, key=lambda h: h.get("order", 9)):
            free = self.free_slots(h)
            if free <= 0:
                continue
            cand = sorted([j for j in self.jobs.values() if j["state"] == "running" and j["host"] != h["name"]
                           and not j.get("backup") and j["kind"] != "train" and not j.get("gpu_now")
                           and time.time() - j.get("t_start", time.time()) > BACKUP_AFTER],
                          key=lambda j: j.get("t_start", 0))[:free]
            if cand:
                for j in cand:
                    j["backup"] = h["name"]
                self.launch(h, cand, backup=True)

    def launch(self, h, js, backup=False):
        lines, made = [], set()
        try:
            # the code changed since this host's last sync (e.g. new run.py flags): sync first, or
            # the jobs fail on the old code (hosts used to sync only when they joined)
            sig = code_sig()
            if h.get("code_sig") != sig:
                ok, msg = self.sync_code(h)
                if not ok:
                    raise RuntimeError(f"code sync failed: {msg}")
                h["code_sig"] = sig
                log(f"{h['name']}: code changed, synced")
                self.ensure_builds(h)   # a new engine config may be wanted by these jobs
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
                ext = "sh" if linux(h) else "cmd"
                cmdf = os.path.join(FLEET, "jobs", f"{j['id']}.{ext}")
                with open(cmdf, "w", newline="") as fh:
                    fh.write(job_sh(h, j) if linux(h) else job_cmd(h, j))
                lines.append(f'put "{cmdf}" "{rpath(h, "fleet/jobs/" + j["id"] + "." + ext)}"')
            sftp(h, [f'-mkdir "{rpath(h, d)}"' for d in ("fleet", "fleet/jobs", "fleetdata", "runs")] + lines)
            if linux(h):
                R = h["root"].rstrip("/")
                # one statement per job with its own redirections, so nothing keeps the ssh channel open
                cmds = [f"cd {R}/bot"]
                for j in js:
                    cmds.append(f"rm -f fleet/jobs/{j['id']}.rc fleet/jobs/{j['id']}.out fleet/jobs/{j['id']}.err")
                    cmds.append(f"setsid nohup nice -n 19 ionice -c3 bash {R}/bot/fleet/jobs/{j['id']}.sh > /dev/null 2>&1 < /dev/null &")
                lx(h, "\n".join(cmds) + "\necho ok")
            else:
                out = ps(h, LAUNCH_PS.replace("%ROOT%", wroot(h)).replace("%IDS%", ",".join(f"'{j['id']}'" for j in js)))
                json.loads(out.strip().splitlines()[-1])
            for j in js:
                if not backup:
                    j["t_start"] = time.time()
            log(f"{h['name']}: started {'backup copies of ' if backup else ''}"
                f"{', '.join(j['id'] + (' (GPU)' if j.get('gpu_now') else '') for j in js)}")
        except Exception as e:
            log(f"{h['name']}: launch failed ({str(e)[:200]})")
            for j in js:
                if backup:
                    j["backup"] = None
                    continue
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
                n = k = 0
                for j in self.jobs.values():
                    if j["pipe"] == tag and j["state"] == "queued":
                        j["state"] = "failed"
                        n += 1
                    elif j["pipe"] == tag and j["state"] == "running":   # stop them too: the slots go to live work
                        for host in (j["host"], j.get("backup")):
                            if host:
                                self.kill_job(host, j["id"])
                        j["state"], j["backup"] = "failed", None
                        k += 1
                log(f"pipeline {tag} cancelled ({n} queued jobs dropped, {k} running ones stopped)")
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
        import signal
        term = []
        # `kill` (SIGTERM): finish this round, save the state, exit; dying mid-round lost
        # whatever the inbox had just applied (a cancel came back on restart)
        signal.signal(signal.SIGTERM, lambda *_: term.append(1))
        while True:
            if term:
                self.save()
                log("coordinator stopped (SIGTERM; state saved)")
                return
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
        run = [j["id"] + ("*" if j.get("backup") == name else "") for j in s["jobs"].values()
               if j["state"] == "running" and name in (j["host"], j.get("backup"))]
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
                                    "cancel", "task"])
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
    ap.add_argument("--mode", default=None, help="bench: eval (default), explore or vary")
    ap.add_argument("--variants", type=int, default=20, help="bench explore/vary: deck variants per save")
    ap.add_argument("--offset", type=int, default=0, help="bench explore/vary: variant/repetition seed offset")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--extra", default=None, help='expert/xdeckrl: JSON list of extra run.py arguments')
    ap.add_argument("--argv", default=None, help='task: JSON list, the script and its arguments')
    ap.add_argument("--inputs", default=None, help='task: JSON list of bot/-relative files to push')
    ap.add_argument("--outputs", default=None, help='task: JSON list of bot/-relative files to fetch')
    ap.add_argument("--gpu", action="store_true", help="task: run on a GPU machine with CUDA")
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
                          "prio": 1.5 if args.prio is None else args.prio, "extra": json.loads(args.extra or "[]"),
                          "inputs": json.loads(args.inputs or "[]")})
    elif args.cmd == "xdeckrl":
        with open(os.path.join(HERE, args.deck_nets)) as fh:
            cfg = json.load(fh)
        init = {k: cfg[k] for k in ("card", "rest", "event", "map", "shop") if cfg.get(k)}
        submit("xdeckrl", {"tag": args.tag, "start": args.start, "iters": args.iters, "runs": args.runs,
                           "chunk": args.expert_chunk, "eval_runs": args.n, "eval_chunk": args.expert_chunk,
                           "temperature": args.temperature, "rollouts": args.rollouts, "init": init,
                           "eval_init": args.eval_init, "kinds": ["card", "rest", "event", "map", "shop"],
                           "extra": json.loads(args.extra or "[]")})
    elif args.cmd == "task":
        submit("task", {"tag": args.tag, "argv": json.loads(args.argv), "inputs": json.loads(args.inputs or "[]"),
                        "outputs": json.loads(args.outputs or "[]"), "gpu": args.gpu,
                        "prio": 0.5 if args.prio is None else args.prio})
    elif args.cmd == "cancel":
        submit("cancel", {"tag": args.tag})
    elif args.cmd == "bench":
        spec = {"tag": args.tag, "bench": args.bench, "configs": json.loads(args.configs), "room": args.room,
                "reps": args.reps, "shards": args.shards, "prio": 0.8 if args.prio is None else args.prio,
                "inputs": json.loads(args.inputs or "[]"), "extra": json.loads(args.extra or "[]")}
        if args.mode:   # explore / vary: deck variants per save
            spec.update(mode=args.mode, variants=args.variants, offset=args.offset)
        submit("bench", spec)


if __name__ == "__main__":
    main()
