"""Act-1/act-2 clear rates of self-play by ascension group per 6-iteration block (up2/up4/up5 archives):
python asctrend.py. A9 (DeadlyEnemies) is the cliff; A9-10 act-1 clear is the number to watch."""
HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
import glob, json, os, re, subprocess
from collections import defaultdict
B = 6
acc = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
for d in sorted(glob.glob(os.path.join(HERE, "data", "xdarchive", "up[245]_it*"))):
    m = re.search(r"/(up\d)_it(\d+)$", d); pipe, it = m.group(1), int(m.group(2))
    lab = f"{pipe} it{(it-1)//B*B+1:02d}"
    out = subprocess.run(f"cat {d}/*.jsonl | grep '\"k\":\"o\"'", shell=True, capture_output=True, text=True).stdout
    for l in out.splitlines():
        r = json.loads(l); a = r.get("asc")
        g = "A8" if a == 8 else "A9-10" if a in (9, 10) else "A5-7" if a in (5, 6, 7) else None
        if g is None: continue
        s = acc[lab][g]; s[0] += 1; ca = r.get("cleared_acts") or 0; s[1] += ca >= 1; s[2] += ca >= 2
for lab in acc:
    print(lab, "  ".join(f"{g} n{s[0]:4d} a1 {s[1]/s[0]:5.1%} a2 {s[2]/s[0]:5.1%}" for g, s in sorted(acc[lab].items())))
