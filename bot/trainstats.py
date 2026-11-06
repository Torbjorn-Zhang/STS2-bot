"""trainstats.py TAG IT: one line of an RL pipeline update's train stats from runs/fleet/state.json
(lr, runs, ret_std/ret_max, ev_value by act, v loss, final KL, gcos), for session watchers."""
import json, sys
tag, it = sys.argv[1], sys.argv[2]
j = json.load(open("runs/fleet/state.json"))["jobs"].get(f"{tag}{it}_train") or {}
a = j.get("argv", [])
out = {"lr": [a[i] for i in range(1, len(a)) if a[i - 1] == "--lr-scale"], "state": j.get("state")}
for rows in ((j.get("result") or {}).get("stats") or {}).values():
    for r in rows:
        if "ev_value" in r:
            out.update({k: r[k] for k in r if k == "runs" or k.startswith(("ret_", "ev_value"))})
        if "epoch" in r:
            out.update(v=r["v"], kl=r["kl"], pg=r["pg"])
        if "gcos_all" in r:
            out.update({k: r[k] for k in r if k.startswith("gcos")})
print(json.dumps(out))
