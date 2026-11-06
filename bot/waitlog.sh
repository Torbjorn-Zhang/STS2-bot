#!/bin/bash
# waitlog.sh LOG REGEX [MAX_S] -- wait for a line matching REGEX (grep -E) to be appended to LOG.
#   exit 0: prints the line            exit 2: LOG unreadable (fails at once, loudly)
#   exit 3: TIMEOUT after MAX_S (default 7200), so a waiting session hears back at least that often
# Byte offsets, not `tail -n +$(wc -l)`: macOS wc pads its count with spaces, and "tail -n +  N"
# failed every poll of two session watchers for 3.75 h on 11-06 while they looked alive.
log=$1; re=$2; max=${3:-7200}
[ -n "$log" ] && [ -n "$re" ] || { echo "usage: waitlog.sh LOG REGEX [MAX_S]"; exit 2; }
off=$(wc -c < "$log" 2>/dev/null | tr -d ' ') || { echo "waitlog: cannot read $log"; exit 2; }
[ -n "$off" ] || { echo "waitlog: cannot read $log"; exit 2; }
end=$(( $(date +%s) + max ))
while :; do
    new=$(tail -c +"$((off + 1))" "$log") || { echo "waitlog: tail failed on $log"; exit 2; }
    m=$(printf '%s\n' "$new" | grep -m1 -E -- "$re")
    if [ -n "$m" ]; then
        echo "$m"
        exit 0
    fi
    if [ "$(date +%s)" -ge "$end" ]; then
        echo "waitlog: TIMEOUT after ${max}s waiting for: $re"
        exit 3
    fi
    sleep 30
done
