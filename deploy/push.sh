#!/usr/bin/env bash
# From the Mac: copy the repo (git checkout) and the private game DLLs to a remote host.
#   deploy/push.sh <ssh-host>        (an alias from ~/.ssh/config)
set -euo pipefail
HOST="$1"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ssh "$HOST" 'mkdir -p ~/sts2 && chmod 700 ~/sts2'
# code + frozen models (what git tracks), no run outputs or data
rsync -az --delete --exclude .git --exclude .venv --exclude local --exclude parity --exclude STS2MCP \
  --exclude 'bot/data' --exclude 'bot/runs' --exclude 'bot/models/*.pt' --exclude 'sts2-cli/src/*/bin' \
  --exclude 'sts2-cli/src/*/obj' --exclude 'sts2-cli/lib' "$HERE/" "$HOST:sts2/"
rsync -az "$HERE/bot/models/frozen/" "$HOST:sts2/bot/models/frozen/"
rsync -az "$HERE/bot/models/policy_rl105.pt" "$HOST:sts2/bot/models/"
# the user's own game DLLs: private directory only
rsync -az "$HERE/sts2-cli/lib/" "$HOST:sts2/sts2-cli/lib/"
ssh "$HOST" 'chmod -R go-rwx ~/sts2'
echo "pushed; now run: ssh $HOST bash sts2/deploy/setup_remote.sh"
