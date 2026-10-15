"""Aggregate throughput, one long-lived engine process per worker (process reuse)."""
import random, sys, time
from concurrent.futures import ThreadPoolExecutor
from simclient import Sim, play_random_run

def worker(args):
    wid, runs = args
    sim = Sim("Release")
    steps = 0
    for i in range(runs):
        r = play_random_run(sim, "Ironclad", f"parr_{wid}_{i}", random.Random(f"{wid}-{i}"))
        steps += r.get("steps", 0)
    sim.close()
    return steps

for n in [int(x) for x in sys.argv[1].split(",")]:
    runs = 30
    t0 = time.perf_counter()
    with ThreadPoolExecutor(n) as ex:
        steps = sum(ex.map(worker, [(w, runs) for w in range(n)]))
    wall = time.perf_counter() - t0
    print(f"reuse workers={n} runs={n*runs} steps={steps} wall={wall:.1f}s steps/s={steps/wall:.0f} runs/h={3600*n*runs/wall:.0f}", flush=True)
