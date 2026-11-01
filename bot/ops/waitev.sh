#!/bin/zsh
# waitev.sh N: wait for up5 eval N (or a runai token expiry), then print the checkpoint summary
N=$1; S=${0:A:h}
cd ${S:h} || exit 1
L0=$(wc -l < runs/fleet/fleet.log)
while true; do
  if grep -q "up5: iteration $N A10" runs/fleet/fleet.log; then echo "EVAL$N"; break; fi
  if tail -n +$((L0+1)) runs/fleet/fleet.log | grep -q "token has expired"; then
    echo "TOKEN_EXPIRED"; tail -n +$((L0+1)) runs/fleet/fleet.log | grep -m1 "token has expired" | cut -c1-120; exit 0
  fi
  sleep 120
done
grep -E "up5: iteration $N A10" runs/fleet/fleet.log | cut -c1-200
../.venv/bin/python cmpb3.py mcref45 up5_ev$((N-18)) up5_ev$((N-12)) up5_ev$((N-6)) up5_ev$N 2>&1 | tail -5
../.venv/bin/python deaths.py up5_ev$N 2>&1 | grep -E "WIN|top killers" -A3 | head -6
grep -E "up5: iteration [0-9]+ expert self-play done" runs/fleet/fleet.log | python3 -c "
import sys,re,json,math
N=$N
d={}
for l in sys.stdin:
    m=re.search(r'iteration (\d+) expert self-play done (\{.*?\})',l); d[int(m.group(1))]=json.loads(m.group(2))
for b in range(N-29,N+7,6):
    its=[i for i in range(b,b+6) if i in d]
    if not its: continue
    n=sum(d[i]['runs'] for i in its); w=sum(d[i]['wins'] for i in its); p=w/n
    print(f'it{b}-{b+5} ({len(its)}) win {p:6.2%} ±{math.sqrt(p*(1-p)/n):.2%}')
"
../.venv/bin/python spcurve.py up5 --block 6 2>&1 | tail -3
for it in $((N-4)) $((N-2)) $N; do zsh $S/trainlog.sh $it 2>&1 | grep -E "^it |epoch\": [01]" | cut -c1-120; done
