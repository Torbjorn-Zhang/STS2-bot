#!/bin/bash
# Keep one low-priority A10 hunt (Fix5 = Fix3 engine, faster; the newest up2 policy, fresh seeds) queued until a
# Fix3 A10 game is won, either by a hunt or by an up2 evaluation; then print it and exit.
cd "$(dirname "$0")"
n=${1:-6}
while true; do
    tag=hunt${n}_a10
    if ! grep -q "pipeline $tag (expert) submitted" runs/fleet/fleet.log; then
        nets=$(ls -t models/up2_nets*.json | head -1)
        ../.venv/bin/python fleet.py expert --tag $tag --deck-nets "$nets" --n 480 --expert-chunk 16 --prefix hunt$n \
            --rollouts '{"k":6,"m":8,"turns":3}' --prio 1.5 \
            --extra '["--engine-config","Fix5","--nodes","1000","--deck-sample","0"]' > /dev/null
        echo "$(date '+%m-%d %H:%M') submitted $tag with $nets"
    fi
    win=$(grep -E "hunt[0-9]+_a10: expert evaluation done .*\"wins\": [1-9]|up2: iteration (1[1-9]|[2-9][0-9]) A10 .*\"wins\": [1-9]" runs/fleet/fleet.log | tail -1)
    if [ -n "$win" ]; then echo "WIN: $win"; exit 0; fi
    if grep -q "$tag: expert evaluation done" runs/fleet/fleet.log; then n=$((n + 1)); continue; fi
    sleep 600
done
