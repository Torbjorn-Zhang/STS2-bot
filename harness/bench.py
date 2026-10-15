"""Throughput benchmark: startup cost, per-step latency, and process reuse across runs."""

import argparse
import json
import random
import statistics
import time

from simclient import Sim, play_random_run

CHARS = ["Ironclad", "Silent", "Defect", "Regent", "Necrobinder"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10, help="runs per character, all in one process")
    ap.add_argument("--config", default="Debug")
    args = ap.parse_args()

    out = {"config": args.config, "per_char": {}}
    for ch in CHARS:
        # One engine process per run: start_run cannot be reused within a process.
        results, lat, starts, boots = [], [], [], []
        t0 = time.perf_counter()
        for i in range(args.runs):
            sim = Sim(args.config)
            rng = random.Random(f"bench-{ch}-{i}")
            results.append(play_random_run(sim, ch, f"bench_{ch}_{i}", rng))
            lat += [s for name, s in sim.latencies if name != "start_run"]
            starts += [s for name, s in sim.latencies if name == "start_run"]
            boots.append(sim.startup_s)
            sim.close()
        wall = time.perf_counter() - t0
        steps = sum(r.get("steps", 0) for r in results)
        out["per_char"][ch] = {
            "startup_s": round(statistics.median(boots), 3),
            "runs": len(results),
            "completed": sum(1 for r in results if r.get("ok")),
            "failures": [r for r in results if not r.get("ok")],
            "max_floor": max((r.get("floor") or 0) for r in results),
            "acts_reached": sorted({r.get("act") for r in results if r.get("ok")}),
            "steps": steps,
            "wall_s": round(wall, 2),
            "steps_per_s": round(steps / wall, 1),
            "steps_per_s_excl_startup": round(steps / sum(lat), 1),
            "runs_per_hour_1core": round(3600 * len(results) / wall),
            "start_run_ms_median": round(1000 * statistics.median(starts), 1),
            "step_ms_median": round(1000 * statistics.median(lat), 2),
            "step_ms_p95": round(1000 * sorted(lat)[int(0.95 * len(lat))], 2),
            "step_ms_max": round(1000 * max(lat), 1),
        }
        print(ch, json.dumps(out["per_char"][ch]), flush=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
