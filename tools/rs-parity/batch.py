"""Run lockstep over many seeds in parallel; one JSON line per seed."""
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, "/Users/szha0569/Daily/sts2-ai/local/rs-trial")

def one(seed):
    import lockstep as L
    try:
        r = L.lockstep(seed)
    except Exception as e:
        r = {"seed": seed, "error": repr(e)[:1500]}
    r["soft"] = dict(L.SOFT_SEEN.most_common(3))
    return r

if __name__ == "__main__":
    seeds = sys.argv[3:] or [f"full_{i}" for i in range(int(sys.argv[1]), int(sys.argv[2]))]
    out = os.environ.get("BATCH_OUT", "/Users/szha0569/Daily/sts2-ai/local/rs-trial/batch.jsonl")
    with Pool(int(os.environ.get("BATCH_PROCS", "4"))) as p, open(out, "a") as f:
        for r in p.imap_unordered(one, seeds):
            f.write(json.dumps(r, default=str) + "\n"); f.flush()
            print(r["seed"], "OK" if r.get("ok") else ("ERR" if "error" in r else f"DIFF@{r.get('step')} {list((r.get('diff') or {}).keys())}"), flush=True)
