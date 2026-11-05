#!/usr/bin/env python3
"""A command channel into Apollo's pods over the lab0 relay, for when runai exec is down.

runai exec, the fleet's only way into a pod, fails whenever Apollo's control plane does (41 % of
11-05/06, up to 2.5 h at a time) while the pods compute on: they finish their jobs and then sit
idle. A pod can still rsync with the relay (its key is rrsync-restricted to the transfer
directory), so the channel is files in <relay>/q/<pod>/:

  agent <pod> <target> <port> <key> <known_hosts> <rel>      (in the pod; fleet.py starts it)
      every few seconds mirrors q/<pod>/*.cmd from the relay, runs each new command (bash -c,
      output to files: a background child holding a pipe would block it), and pushes
      q/<pod>/done/<id>.json (rc, out, err) and a heartbeat q/<pod>/done/alive back
  call <pod> <relay dir> <b64 command> <timeout>             (on lab0, from fleet.remote_argv)
      queues one command, waits for its result, prints it and exits with its rc;
      97: no agent heard from lately (nothing queued, nothing ran)
  alive <pod> <relay dir>                                    (on lab0) exit 0 when the agent is up

Commands run once: the agent remembers the ids it took (also across its own restarts), and
call deletes the command file when it has the result.
"""

import base64
import json
import os
import subprocess
import sys
import threading
import time

ALIVE_MAX = 45      # s: an agent heard from within this is up (it pushes every 2-6 s)
OUT_MAX = 4 << 20   # bytes of stdout kept (the fleet's polls print a few kB)


def _age(done):
    try:
        with open(os.path.join(done, "alive")) as fh:
            return time.time() - float(fh.read().strip() or 0)
    except (OSError, ValueError):
        return 1e9


def alive(pod, d):
    sys.exit(0 if _age(os.path.join(d, "q", pod, "done")) < ALIVE_MAX else 1)


def call(pod, d, b64, timeout):
    q = os.path.join(d, "q", pod)
    done = os.path.join(q, "done")
    if _age(done) >= ALIVE_MAX:
        sys.stderr.write(f"podrelay: no agent on {pod}\n")
        sys.exit(97)
    i = f"{int(time.time() * 1000)}_{os.getpid()}"
    cmd, res = os.path.join(q, i + ".cmd"), os.path.join(done, i + ".json")
    tmp = os.path.join(q, i + ".part")
    with open(tmp, "w") as fh:
        json.dump({"id": i, "cmd": base64.b64decode(b64).decode(), "timeout": timeout}, fh)
    os.rename(tmp, cmd)
    end = time.time() + timeout + 60
    try:
        while time.time() < end:
            if os.path.exists(res):
                try:
                    with open(res) as fh:
                        m = json.load(fh)
                except ValueError:   # not yet complete (rsync writes a temp file, then renames)
                    time.sleep(0.2)
                    continue
                os.remove(res)
                sys.stdout.write(m.get("out", ""))
                sys.stderr.write(m.get("err", ""))
                sys.stdout.flush()
                sys.exit(int(m.get("rc", 1)))
            if _age(done) >= 3 * ALIVE_MAX:
                sys.stderr.write(f"podrelay: agent on {pod} went quiet with {i} queued\n")
                sys.exit(98)
            time.sleep(0.2)
        sys.stderr.write(f"podrelay: {i} timed out on {pod}\n")
        sys.exit(124)
    finally:
        try:
            os.remove(cmd)
        except OSError:
            pass


def _run(m, out):
    """One command: bash -c, stdin closed, output to files; result json renamed into out/."""
    i = m["id"]
    o, e = os.path.join(out, i + ".o"), os.path.join(out, i + ".e")
    try:
        with open(o, "wb") as fo, open(e, "wb") as fe:
            p = subprocess.Popen(["bash", "-c", m["cmd"]], stdin=subprocess.DEVNULL, stdout=fo, stderr=fe)
            try:
                rc = p.wait(timeout=float(m.get("timeout", 1200)))
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
                rc = 124
        with open(o, "rb") as fh:
            so = fh.read()[-OUT_MAX:].decode(errors="replace")
        with open(e, "rb") as fh:
            se = fh.read()[-65536:].decode(errors="replace")
    except Exception as x:
        rc, so, se = 125, "", f"podrelay agent: {type(x).__name__}: {x}\n"
    for f in (o, e):
        try:
            os.remove(f)
        except OSError:
            pass
    tmp = os.path.join(out, i + ".tmp")
    with open(tmp, "w") as fh:
        json.dump({"id": i, "rc": rc, "out": so, "err": se}, fh)
    os.rename(tmp, os.path.join(out, i + ".json"))


def agent(pod, target, port, key, kh, rel):
    import fcntl
    base = f"/tmp/podrelay_{pod}"
    inn, out = os.path.join(base, "in"), os.path.join(base, "out")
    os.makedirs(inn, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    lock = open(os.path.join(base, "lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(0)   # one agent per pod
    seen_f = os.path.join(base, "seen")
    try:
        with open(seen_f) as fh:
            seen = set(fh.read().split())
    except OSError:
        seen = set()
    e = (f"ssh -p {port} -i {key} -o UserKnownHostsFile={kh} -o StrictHostKeyChecking=yes -o BatchMode=yes "
         f"-o ConnectTimeout=20")
    remote = f"{target}:{rel}/q/{pod}/" if target != "-" else f"{rel}/q/{pod}/"   # "-": a local directory (tests)
    ssh = ["-e", e] if target != "-" else []
    last, fails = 0.0, 0
    log = lambda s: print(time.strftime("%m-%d %H:%M:%S"), s, flush=True)
    log(f"agent for {pod} up ({remote})")
    while True:
        t0 = time.time()
        r = subprocess.run(["rsync", "-a", "--delete", "--include=*.cmd", "--exclude=*"] + ssh + [remote, inn + "/"],
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
        if r.returncode == 0:
            fails = 0
            now = set()
            for f in sorted(os.listdir(inn)):
                if not f.endswith(".cmd"):
                    continue
                i = f[:-4]
                now.add(i)
                if i in seen:
                    continue
                try:
                    with open(os.path.join(inn, f)) as fh:
                        m = json.load(fh)
                except (OSError, ValueError):
                    continue
                seen.add(i)
                with open(seen_f, "a") as fh:
                    fh.write(i + "\n")
                last = time.time()
                threading.Thread(target=_run, args=(m, out), daemon=True).start()
            if len(seen) > 5000:   # ids the caller already deleted never come back
                seen = {i for i in seen if i in now or os.path.exists(os.path.join(out, i + ".json"))}
                with open(seen_f, "w") as fh:
                    fh.write("".join(i + "\n" for i in seen))
        else:
            fails += 1
            if fails in (1, 10) or fails % 100 == 0:
                log(f"pull failed ({fails}): {r.stderr.decode(errors='replace').strip()[-200:]}")
        with open(os.path.join(out, "alive.tmp"), "w") as fh:
            fh.write(f"{time.time():.3f}")
        os.rename(os.path.join(out, "alive.tmp"), os.path.join(out, "alive"))
        ready = [f for f in os.listdir(out) if f.endswith(".json")]
        r = subprocess.run(["rsync", "-a", "--exclude=*.tmp", "--exclude=*.o", "--exclude=*.e"] + ssh + [out + "/",
                            remote + "done/"], stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
        if r.returncode == 0:
            for f in ready:
                os.remove(os.path.join(out, f))
        elif fails == 0:
            log(f"push failed: {r.stderr.decode(errors='replace').strip()[-200:]}")
        busy = time.time() - last < 120
        time.sleep(max(0.2, (1.5 if busy else 5.0) - (time.time() - t0)))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "agent":
        agent(*a[1:7])
    elif a[0] == "call":
        call(a[1], a[2], a[3], float(a[4]))
    elif a[0] == "alive":
        alive(a[1], a[2])
    else:
        raise SystemExit(__doc__)
